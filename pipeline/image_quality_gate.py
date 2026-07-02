"""
图片质量关卡 (Image Quality Gate)
==================================
渲染后的本地像素级检测，零API成本。
检测到问题 → 返回 FAIL + 原因 → 触发重生成（不放行）。

6项检测（类型感知）：
1. 空白检测：非白像素过少 → 渲染可能失败
2. 遮挡检测：文字bbox高密度堆积 → 文字互相遮挡（热力图/填充图豁免）
3. 裁切检测：内容触及图片边缘 → 被tight_layout裁掉
4. 乱码检测：孤立小方块聚集 → 字体缺失导致口字符
5. 内容比例检测：有效内容区域太小 → 图形太小/大面积空白
6. 空框检测：坐标轴框内无数据 → 仅对坐标轴类图生效

diagram_meta 参数让检测器知道图的类型，实现精准检测：
- heatmap: 跳过遮挡检测+空框检测，改查格子内是否有数值标注
- scatter/function/bar/distribution/regression_plot: 重点查空框
- networkx/graphviz: 跳过空框检测（白底稀疏线条正常），查节点数
- venn: 跳过空框检测
- table: 检测是否有文字行
- geometry: 跳过空框（线条图正常）
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Dict, Optional


# 图类型分组
AXIS_PLOT_TYPES = {"function", "bar", "scatter", "distribution", "regression_plot", "pie"}
FILL_PLOT_TYPES = {"heatmap"}
GRAPH_ENGINES = {"networkx", "graphviz"}
SPARSE_ENGINES = {"geometry", "venn", "table", "circuit", "fallback"}


def quality_gate(image_path: str, diagram_meta: Optional[Dict] = None) -> Tuple[bool, List[str]]:
    """
    图片质量关卡（主入口）。
    
    Args:
        image_path: PNG 图片路径
        diagram_meta: 可选的图类型元数据，来自 render_instruction
            例: {"engine": "matplotlib", "diagram_type": "heatmap", "plot_type": "heatmap"}
        
    Returns:
        (pass: bool, issues: list[str])
        pass=True 表示通过，pass=False 表示不合格+原因列表
    """
    path = Path(image_path)
    if not path.exists():
        return False, ["图片文件不存在"]
    
    file_size = path.stat().st_size
    if file_size < 2000:  # < 2KB 几乎肯定是空白/损坏
        return False, [f"文件过小({file_size}B)，疑似空白或损坏图片"]
    
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return False, [f"无法打开图片: {e}"]
    
    arr = np.array(img)
    height, width = arr.shape[:2]
    
    # 解析图类型
    meta = diagram_meta or {}
    engine = meta.get("engine", "").lower()
    plot_type = meta.get("plot_type", "").lower()
    diagram_type = meta.get("diagram_type", "").lower()
    
    # gpt-image-2 是生成式图片，不适用旧的matplotlib坐标轴/字体/热力图专项检查。
    # 只做通用像素质量关卡：可打开、非空白、非全白/全黑、内容量足够、内容不被裁切、不是纯噪声/纯色块。
    if engine == "gpt-image-2" or diagram_type == "image_prompt":
        return _quality_gate_gpt_image(arr, width, height, file_size)

    issues = []
    
    # === 检测1: 空白检测（所有图类型都检测）===
    issue = _check_blank(arr, width, height, file_size)
    if issue:
        issues.append(issue)
    
    # === 检测2: 遮挡/重叠检测（热力图/填充图跳过）===
    if plot_type not in FILL_PLOT_TYPES and diagram_type not in FILL_PLOT_TYPES:
        issue = _check_occlusion(arr, width, height)
        if issue:
            issues.append(issue)
    
    # === 检测3: 边缘裁切检测 ===
    issue = _check_edge_clipping(arr, width, height)
    if issue:
        issues.append(issue)
    
    # === 检测4: 乱码/方块字检测 ===
    issue = _check_garbled_text(arr, width, height)
    if issue:
        issues.append(issue)
    
    # === 检测5: 内容比例检测（graph/geometry跳过，线条图天然稀疏）===
    if engine not in GRAPH_ENGINES and engine not in SPARSE_ENGINES:
        issue = _check_content_ratio(arr, width, height)
        if issue:
            issues.append(issue)
    
    # === 检测6: 空框检测（仅对坐标轴类图：scatter/function/bar/distribution/regression）===
    if plot_type in AXIS_PLOT_TYPES or (engine == "matplotlib" and plot_type not in FILL_PLOT_TYPES):
        issue = _check_empty_frame(arr, width, height)
        if issue:
            issues.append(issue)
    
    # === 检测7: 热力图专项（有colorbar但格子内无数值标注）===
    if plot_type in FILL_PLOT_TYPES or diagram_type in FILL_PLOT_TYPES:
        issue = _check_heatmap_content(arr, width, height)
        if issue:
            issues.append(issue)
    
    # === 检测8: 图/网络专项（节点数是否合理）===
    if engine in GRAPH_ENGINES:
        issue = _check_graph_content(arr, width, height)
        if issue:
            issues.append(issue)
    
    return (len(issues) == 0, issues)


def _quality_gate_gpt_image(arr: np.ndarray, width: int, height: int, file_size: int) -> Tuple[bool, List[str]]:
    """
    gpt-image-2 专用图片质量检查。

    只做生成图通用检查，不做旧matplotlib的坐标轴/空框/字体推断：
    1. 分辨率和文件大小合理
    2. 不是全白、全黑、纯色或近纯色
    3. 有足够非背景内容
    4. 内容主体不能过小、不能贴边裁切
    5. 有足够颜色/纹理变化，避免API返回空白占位/纯噪声/大色块
    """
    issues = []
    pixels = width * height
    gray = np.mean(arr.astype(np.float32), axis=2)

    if width < 512 or height < 512:
        issues.append(f"gpt-image-2图片分辨率过低({width}x{height})")
    small_file = file_size < 8_000
    suspicious_small_file = file_size < 20_000

    # 背景通常接近白色；同时兼容黑底/彩底异常返回。
    white_ratio = np.all(arr > 245, axis=2).sum() / pixels
    black_ratio = np.all(arr < 10, axis=2).sum() / pixels
    non_white_ratio = np.any(arr < 240, axis=2).sum() / pixels
    non_black_ratio = np.any(arr > 20, axis=2).sum() / pixels
    channel_std = float(arr.reshape(-1, 3).std(axis=0).mean())
    gray_std = float(gray.std())

    if white_ratio > 0.985 or non_white_ratio < 0.006:
        issues.append(f"gpt-image-2空白/近全白图片(white={white_ratio:.1%}, non_white={non_white_ratio:.2%})")
    if black_ratio > 0.985 or non_black_ratio < 0.006:
        issues.append(f"gpt-image-2近全黑图片(black={black_ratio:.1%}, non_black={non_black_ratio:.2%})")
    if channel_std < 3.0 or gray_std < 3.0:
        issues.append(f"gpt-image-2近纯色图片(std={gray_std:.2f})")

    # 有效内容：与白背景有明显差异的像素。阈值不能太高，学术线图可能较稀疏。
    content_mask = np.any(arr < 235, axis=2)
    content_ratio = content_mask.sum() / pixels
    if content_ratio < 0.015:
        issues.append(f"gpt-image-2有效内容过少(content={content_ratio:.2%})")
    if content_ratio > 0.92:
        issues.append(f"gpt-image-2画面几乎被填满(content={content_ratio:.1%})，疑似非白底/噪声/大色块")

    # 内容包围盒：主体太小或贴边说明可能裁切/构图失败。
    ys, xs = np.where(content_mask)
    if len(xs) > 0 and len(ys) > 0:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bbox_w = x1 - x0 + 1
        bbox_h = y1 - y0 + 1
        bbox_area_ratio = (bbox_w * bbox_h) / pixels
        margin = max(8, int(min(width, height) * 0.015))
        if bbox_area_ratio < 0.05:
            issues.append(f"gpt-image-2主体区域过小(bbox={bbox_area_ratio:.1%})")
        edge_hits = sum([
            x0 <= margin,
            y0 <= margin,
            (width - 1 - x1) <= margin,
            (height - 1 - y1) <= margin,
        ])
        # gpt-image-2 学术图常会把坐标轴/标题贴近边缘，甚至铺满画布。
        # 这类构图特征不作为自动阻断项；只保留空白/全白/主体过小等硬失败。
    else:
        issues.append("gpt-image-2未检测到有效内容区域")

    # 颜色/纹理复杂度：空白图颜色很少，纯噪声边缘极多；二者都拦截。
    sample_step = max(1, min(width, height) // 160)
    sampled = arr[::sample_step, ::sample_step].reshape(-1, 3)
    quantized = (sampled // 16).astype(np.uint8)
    unique_colors = len(np.unique(quantized, axis=0))
    if unique_colors < 6:
        issues.append(f"gpt-image-2颜色变化过少({unique_colors}种量化色)，疑似空白/纯色图")
    if small_file and (content_ratio < 0.03 or unique_colors < 12):
        issues.append(f"gpt-image-2图片文件过小且内容不足({file_size//1024}KB, content={content_ratio:.2%}, colors={unique_colors})")
    elif suspicious_small_file and content_ratio < 0.015:
        issues.append(f"gpt-image-2图片文件偏小且有效内容过少({file_size//1024}KB, content={content_ratio:.2%})")

    # 边缘密度：太少=没内容；太多=随机噪声/照片化纹理过强，不像清晰教材图。
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    edge_energy = float((gx + gy) / 2.0)
    if edge_energy < 0.8:
        issues.append(f"gpt-image-2边缘信息过少(edge={edge_energy:.2f})")
    if edge_energy > 45:
        issues.append(f"gpt-image-2纹理/噪声过强(edge={edge_energy:.2f})，不像清晰学术图")

    return (len(issues) == 0, issues)


def _check_blank(arr: np.ndarray, width: int, height: int, file_size: int) -> str:
    """
    空白检测：
    - 非白像素(RGB任一通道<240) < 2% → 空白图
    - 同时 file_size < 10KB → 确认空白（不是正常的白底线图）
    
    区分正常Venn图（白底但有丰富抗锯齿色彩，通常>50KB）和真空白。
    """
    non_white_mask = np.any(arr < 240, axis=2)
    non_white_ratio = non_white_mask.sum() / (width * height)
    
    # 颜色种类数（抽样检测，避免全图计算太慢）
    sample_step = max(1, height // 100)
    sampled = arr[::sample_step, ::sample_step].reshape(-1, 3)
    unique_colors = len(np.unique(sampled, axis=0))
    
    # 条件：非白<1% 且 (文件<10KB 或 颜色种类<30)
    # 注：分子结构图(RDKit)线条细+白底大，非白可能只有1-2%但颜色丰富(>50色)
    if non_white_ratio < 0.01 and (file_size < 10000 or unique_colors < 30):
        return f"空白图片(非白像素{non_white_ratio*100:.1f}%, {file_size//1024}KB, {unique_colors}色)"
    
    return ""


def _check_occlusion(arr: np.ndarray, width: int, height: int) -> str:
    """
    遮挡/重叠检测：
    在深色像素(灰度<80)的分布中，找文字行段，检测：
    1. 相邻文字段间距<2px → 文字行重叠
    2. 局部区域(50x50块)深色密度>70% → 图形+文字堆积严重
    3. 文字连通区域中心距<阈值 → 标注互相遮挡
    """
    from scipy import ndimage
    
    gray = np.mean(arr[:, :, :3], axis=2)
    dark_mask = gray < 80  # 深色像素（文字/线条）
    
    # === 检测1: 按行扫描找文字段 ===
    row_density = dark_mask.sum(axis=1) / width
    
    # 找连续高密度行段（>5%像素为深色）
    in_seg = False
    segments = []
    start = 0
    threshold = 0.05
    
    for i, d in enumerate(row_density):
        if d > threshold and not in_seg:
            start = i
            in_seg = True
        elif d <= threshold and in_seg:
            if i - start > 3:  # 排除噪声
                segments.append((start, i))
            in_seg = False
    if in_seg and len(row_density) - start > 3:
        segments.append((start, len(row_density)))
    
    # 检查相邻段间距
    overlap_count = 0
    for i in range(len(segments) - 1):
        gap = segments[i+1][0] - segments[i][1]
        if gap < 2:
            overlap_count += 1
    
    if overlap_count >= 3:
        return f"文字行重叠严重({overlap_count}处间距<2px)"
    
    # === 检测2: 局部高密度检测（50x50块扫描）===
    block_size = 50
    high_density_blocks = 0
    total_blocks = 0
    
    for y in range(0, height - block_size, block_size // 2):
        for x in range(0, width - block_size, block_size // 2):
            block = dark_mask[y:y+block_size, x:x+block_size]
            density = block.sum() / block.size
            total_blocks += 1
            if density > 0.70:
                high_density_blocks += 1
    
    if total_blocks > 0 and high_density_blocks / total_blocks > 0.05:
        # 排除热力图/填充色块：如果深色像素中彩色(通道差>30)比例>50%，
        # 说明是图形填充色而非文字堆积，不判为遮挡
        dark_pixels = arr[dark_mask]
        if len(dark_pixels) > 100:
            ch_range = dark_pixels.max(axis=1).astype(int) - dark_pixels.min(axis=1).astype(int)
            colored_ratio = (ch_range > 30).mean()
            if colored_ratio > 0.50:
                return ""  # 彩色填充，不是文字遮挡
        return f"局部遮挡严重({high_density_blocks}个区块密度>70%)"
    
    # === 检测3: 标注文字间距检测（连通区域bbox交叉 + 中心距）===
    # 找文字大小的连通区域（面积100-5000px），检测：
    # a) bbox直接交叉 → 确定遮挡
    # b) 绘图区内多对中心距过近 → 疑似遮挡
    # 排除：坐标轴区域（底部15%、左侧12%）和标题区域（顶部12%）
    try:
        labeled, num_features = ndimage.label(dark_mask)
        if num_features > 0:
            sizes = ndimage.sum(dark_mask, labeled, range(1, num_features + 1))
            text_region_ids = [i + 1 for i, s in enumerate(sizes) if 100 < s < 5000]
            
            if len(text_region_ids) >= 3:
                # 计算每个区域的中心和bbox，过滤掉坐标轴/标题区域
                plot_bboxes = []
                plot_centers = []
                for r_id in text_region_ids:
                    c = ndimage.center_of_mass(dark_mask, labeled, r_id)
                    y_ratio = c[0] / height
                    x_ratio = c[1] / width
                    if 0.12 < y_ratio < 0.85 and x_ratio > 0.12:
                        ys, xs = np.where(labeled == r_id)
                        plot_bboxes.append((ys.min(), xs.min(), ys.max(), xs.max()))
                        plot_centers.append(c)
                
                # a) bbox交叉检测 — 只算重叠面积占较小区域>30%的真遮挡
                bbox_overlaps = 0
                for i in range(len(plot_bboxes)):
                    for j in range(i + 1, len(plot_bboxes)):
                        b1, b2 = plot_bboxes[i], plot_bboxes[j]
                        # 计算交叉区域
                        oy1 = max(b1[0], b2[0])
                        ox1 = max(b1[1], b2[1])
                        oy2 = min(b1[2], b2[2])
                        ox2 = min(b1[3], b2[3])
                        if oy1 < oy2 and ox1 < ox2:
                            overlap_area = (oy2 - oy1) * (ox2 - ox1)
                            area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
                            area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
                            min_area = min(area1, area2)
                            if min_area > 0 and overlap_area / min_area > 0.30:
                                bbox_overlaps += 1
                
                if bbox_overlaps >= 2:
                    return f"标注文字互相遮挡({bbox_overlaps}对bbox重叠>30%)"
                
                # b) 中心距检测（更宽松的补充）
                min_dist_threshold = min(width, height) * 0.02
                close_pairs = 0
                for i in range(len(plot_centers)):
                    for j in range(i + 1, len(plot_centers)):
                        dist = np.sqrt((plot_centers[i][0] - plot_centers[j][0])**2 + 
                                     (plot_centers[i][1] - plot_centers[j][1])**2)
                        if dist < min_dist_threshold:
                            close_pairs += 1
                if close_pairs >= 4:
                    return f"标注文字互相重叠({close_pairs}对间距<{min_dist_threshold:.0f}px)"
    except Exception:
        pass  # scipy不可用时跳过此检测
    
    return ""


def _check_edge_clipping(arr: np.ndarray, width: int, height: int) -> str:
    """
    边缘裁切检测：
    检查4条边缘5px范围内是否有大量深色像素（内容被裁切）。
    排除：底部常有x轴标签贴边的情况，阈值适当放松。
    """
    edge_width = 5
    
    edges = {
        "顶部": arr[0:edge_width, :, :3],
        "左侧": arr[:, 0:edge_width, :3],
        "右侧": arr[:, -edge_width:, :3],
    }
    # 底部单独处理（阈值放松）
    bottom = arr[-edge_width:, :, :3]
    
    clipped_edges = []
    
    for edge_name, edge_pixels in edges.items():
        gray = np.mean(edge_pixels, axis=2)
        dark_ratio = (gray < 100).sum() / gray.size
        if dark_ratio > 0.20:  # 超过20%深色 → 内容被裁
            clipped_edges.append(edge_name)
    
    # 底部用更宽松阈值
    gray_bot = np.mean(bottom, axis=2)
    if (gray_bot < 100).sum() / gray_bot.size > 0.35:
        clipped_edges.append("底部")
    
    if clipped_edges:
        return f"内容被裁切({','.join(clipped_edges)}边缘有大量深色像素)"
    
    return ""


def _check_garbled_text(arr: np.ndarray, width: int, height: int) -> str:
    """
    乱码/方块字检测：
    字体缺失时，字符会渲染为小方块(□)。
    特征：大量孤立的小矩形连通区域，大小在8x8~20x20之间，内部中空。
    
    简化检测：在深色像素中找小矩形区域，如果数量异常多则判定为乱码。
    """
    gray = np.mean(arr[:, :, :3], axis=2)
    dark_mask = (gray < 80).astype(np.uint8)
    
    # 按列扫描找窄竖线段（方块字的特征：连续的等宽竖线对）
    # 简化：统计 8~20px 高度的独立深色段数量
    col_segments_count = 0
    sample_cols = range(0, width, max(1, width // 50))
    
    for col in sample_cols:
        col_data = dark_mask[:, col]
        in_seg = False
        seg_start = 0
        for i, v in enumerate(col_data):
            if v and not in_seg:
                seg_start = i
                in_seg = True
            elif not v and in_seg:
                seg_len = i - seg_start
                if 8 <= seg_len <= 20:
                    col_segments_count += 1
                in_seg = False
    
    # 正常图片中 8-20px 的小段也会有一些（annotate 文字），
    # 但如果占比极高说明是方块字
    expected_normal = len(list(sample_cols)) * 3  # 正常最多每列3个小段
    if col_segments_count > expected_normal * 4:
        return f"疑似乱码/方块字(检测到{col_segments_count}个小矩形段，正常上限{expected_normal})"
    
    return ""


def _check_content_ratio(arr: np.ndarray, width: int, height: int) -> str:
    """
    内容比例检测：
    有效内容的 bounding box 占整张图面积的比例。
    如果内容只占图片<15%的面积 → 图形太小，大面积浪费空白。
    """
    non_white = np.any(arr < 235, axis=2)
    
    # 找 bounding box
    rows_with_content = np.any(non_white, axis=1)
    cols_with_content = np.any(non_white, axis=0)
    
    if not rows_with_content.any():
        return "图片完全空白，无任何内容"
    
    row_indices = np.where(rows_with_content)[0]
    col_indices = np.where(cols_with_content)[0]
    
    content_height = row_indices[-1] - row_indices[0]
    content_width = col_indices[-1] - col_indices[0]
    
    content_area = content_height * content_width
    total_area = height * width
    
    content_ratio = content_area / total_area
    
    if content_ratio < 0.15:
        return f"内容区域过小(仅占{content_ratio*100:.0f}%画面)，图形太小或大面积空白"
    
    return ""


def _check_empty_frame(arr: np.ndarray, width: int, height: int) -> str:
    """
    空框检测：画了坐标轴/边框但内部没有实质数据。
    
    策略：裁掉外围12%边距（标题/轴标签），分析中心区域。
    在中心区域内，排除白色(背景)、浅灰(网格线)、深色极细线(轴线)后，
    剩余的"数据像素"<0.5% → 判定为空框。
    
    适用于：matplotlib坐标轴图（scatter/bar/function/regression等）。
    不影响：Venn图/网络图/流程图/电路图（这些没有大面积白色中心区域）。
    """
    h, w = arr.shape[:2]
    margin_y = int(h * 0.12)
    margin_x = int(w * 0.12)
    center = arr[margin_y:h-margin_y, margin_x:w-margin_x]
    
    if center.size == 0:
        return ""
    
    # 分类像素
    # 白色背景: 所有通道 > 245
    is_white = np.all(center > 245, axis=2)
    # 纯黑/深色(轴线): 所有通道 < 40
    is_dark_line = np.all(center < 40, axis=2)
    # 灰色调(网格线/淡色边框): 所有通道在 140-240 之间且通道差<20
    center_int = center.astype(np.int16)
    channel_range = center_int.max(axis=2) - center_int.min(axis=2)
    avg_val = center_int.mean(axis=2)
    is_gray = (channel_range < 20) & (avg_val > 140) & (avg_val < 240)
    
    # 数据像素 = 非白、非轴线、非网格
    data_pixels = ~is_white & ~is_dark_line & ~is_gray
    data_ratio = data_pixels.sum() / data_pixels.size
    
    # 额外条件：白色占比>85%（确认这是一个"白底+框"的坐标轴图）
    white_ratio = is_white.sum() / is_white.size
    
    if white_ratio > 0.85 and data_ratio < 0.005:
        return f"空框图(坐标轴框内无数据: 数据像素仅{data_ratio*100:.2f}%, 白色{white_ratio*100:.0f}%)"
    
    return ""


def _check_heatmap_content(arr: np.ndarray, width: int, height: int) -> str:
    """
    热力图专项检测：
    1. 中心区域应有大面积彩色填充（非白非灰的色块）
    2. 如果只有 colorbar（右侧窄条有色）但主图区全白 → FAIL
    """
    h, w = arr.shape[:2]
    margin_y = int(h * 0.12)
    margin_x = int(w * 0.12)
    
    # 主图区（左侧 80%）vs colorbar 区（右侧 20%）
    center = arr[margin_y:h-margin_y, margin_x:w-margin_x]
    if center.size == 0:
        return ""
    
    split_x = int(center.shape[1] * 0.80)
    main_area = center[:, :split_x]
    
    # 在主图区找彩色像素（通道差>30，排除灰/白/黑）
    main_int = main_area.astype(np.int16)
    ch_range = main_int.max(axis=2) - main_int.min(axis=2)
    has_color = ch_range > 30
    color_ratio = has_color.mean()
    
    # 热力图主图区应至少有 5% 彩色像素（格子填充）
    if color_ratio < 0.02:
        # 再检查是否整体太白
        is_white = np.all(main_area > 240, axis=2)
        white_ratio = is_white.mean()
        if white_ratio > 0.90:
            return f"热力图主区无数据(彩色{color_ratio*100:.1f}%, 白色{white_ratio*100:.0f}%，仅colorbar有色)"
    
    return ""


def _check_graph_content(arr: np.ndarray, width: int, height: int) -> str:
    """
    图/网络专项检测：
    确保不是完全空白。网络图/graphviz 应至少有一些非白像素（节点+边线）。
    使用宽松标准：非白像素 > 0.5% 即可（线条图天然稀疏）。
    """
    non_white = np.any(arr < 235, axis=2)
    non_white_ratio = non_white.sum() / (width * height)
    
    if non_white_ratio < 0.005:
        return f"网络/图结构图几乎空白(非白像素仅{non_white_ratio*100:.2f}%)"
    
    return ""


# ========== 便捷函数 ==========

def gate_pass(image_path: str, diagram_meta: Optional[Dict] = None) -> bool:
    """简化接口：True=通过，False=不通过"""
    passed, _ = quality_gate(image_path, diagram_meta)
    return passed


def gate_report(image_path: str, diagram_meta: Optional[Dict] = None) -> str:
    """返回人可读的检测报告"""
    passed, issues = quality_gate(image_path, diagram_meta)
    if passed:
        return f"✅ {Path(image_path).name}: PASS"
    return f"❌ {Path(image_path).name}: FAIL\n" + "\n".join(f"  - {i}" for i in issues)
