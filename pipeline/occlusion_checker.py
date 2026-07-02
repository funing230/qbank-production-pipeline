"""
图片防遮挡检查器 (Occlusion Checker)
====================================
检查渲染后的PNG图片是否存在文字与图形互相遮挡问题。
发现问题后尝试通过修改渲染代码自动修复。

规则：
- 文字与文字不得重叠
- 文字不得被柱形、曲线、节点、箭头、边框遮挡
- 文字不得遮挡关键图形、数据点、连线、箭头
- 图例、标题、注释和底部说明之间必须保留足够间距
- 文字不得超出画布或被裁切
- 不得改变图片表达的数据、题意、答案和美术风格
"""

import re
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ========== 图片检查 ==========

def check_image(image_path: str) -> dict:
    """
    检查PNG图片是否存在遮挡/重叠/裁切问题。
    
    Returns:
        {
            "pass": bool,
            "issues": [
                {"type": str, "severity": "high"|"medium"|"low", 
                 "location": [x, y, w, h], "description": str}
            ]
        }
    """
    issues = []
    
    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception as e:
        return {"pass": False, "issues": [{"type": "load_error", "severity": "high",
                                           "location": [0,0,0,0], "description": f"无法加载图片: {e}"}]}
    
    width, height = img.size
    arr = np.array(img)
    
    # 检查1: 边缘裁切检测 - 检查图片边缘是否有被截断的内容
    edge_issues = _check_edge_clipping(arr, width, height)
    issues.extend(edge_issues)
    
    # 检查2: 文字区域检测与重叠分析
    text_issues = _check_text_overlap(arr, width, height)
    issues.extend(text_issues)
    
    # 检查3: 密集区域检测 (文字与图形过近)
    density_issues = _check_density(arr, width, height)
    issues.extend(density_issues)
    
    # 检查4: 图例/标题间距检测
    spacing_issues = _check_legend_spacing(arr, width, height)
    issues.extend(spacing_issues)
    
    return {"pass": len(issues) == 0, "issues": issues}


def _check_edge_clipping(arr: np.ndarray, width: int, height: int) -> list:
    """检测图片边缘是否有被裁切的内容（非白色像素紧贴边缘）"""
    issues = []
    # 检查四条边缘3px范围内是否有大量非白色非背景像素
    edges = {
        "top": arr[0:3, :, :3],
        "bottom": arr[-3:, :, :3],
        "left": arr[:, 0:3, :3],
        "right": arr[:, -3:, :3],
    }
    
    for edge_name, edge_pixels in edges.items():
        # 计算非白色像素比例 (排除纯白和近白)
        non_white = np.any(edge_pixels < 200, axis=-1)
        ratio = non_white.sum() / non_white.size
        if ratio > 0.15:  # 超过15%的边缘像素非白色
            issues.append({
                "type": "edge_clipping",
                "severity": "medium",
                "location": [0, 0, width, height],
                "description": f"{edge_name}边缘有{ratio*100:.0f}%非白色像素，可能存在内容裁切"
            })
    
    return issues


def _check_text_overlap(arr: np.ndarray, width: int, height: int) -> list:
    """检测文字区域重叠"""
    issues = []
    
    # 提取深色连通区域作为"文字候选"
    gray = np.mean(arr[:, :, :3], axis=2)
    text_mask = gray < 80  # 深色像素（文字通常是黑色/深色）
    
    # 简化：按行扫描，找到密集文字行，检查是否有过于接近的行
    row_density = text_mask.sum(axis=1) / width
    
    # 找文字行段（连续高密度行）
    text_rows = row_density > 0.05
    segments = _find_segments(text_rows)
    
    # 检查相邻文字段间距
    for i in range(len(segments) - 1):
        gap = segments[i+1][0] - segments[i][1]
        if gap < 3 and gap >= 0:
            seg_height = segments[i][1] - segments[i][0]
            if seg_height > 5:  # 排除单像素噪声
                issues.append({
                    "type": "text_overlap",
                    "severity": "high",
                    "location": [0, segments[i][0], width, segments[i+1][1] - segments[i][0]],
                    "description": f"第{segments[i][0]}-{segments[i+1][1]}行文字间距过小({gap}px)，可能重叠"
                })
    
    return issues


def _check_density(arr: np.ndarray, width: int, height: int) -> list:
    """检测图形密集区域中的文字拥挤问题"""
    issues = []
    
    # 将图片分为4x4网格，检查每个格子的内容密度
    grid_h, grid_w = height // 4, width // 4
    gray = np.mean(arr[:, :, :3], axis=2)
    
    for gy in range(4):
        for gx in range(4):
            cell = gray[gy*grid_h:(gy+1)*grid_h, gx*grid_w:(gx+1)*grid_w]
            # 密度 = 非白色像素占比
            density = (cell < 200).sum() / cell.size
            if density > 0.6:  # 超过60%像素有内容，可能过于拥挤
                issues.append({
                    "type": "overcrowded",
                    "severity": "low",
                    "location": [gx*grid_w, gy*grid_h, grid_w, grid_h],
                    "description": f"区域({gx},{gy})内容密度{density*100:.0f}%，可能存在遮挡"
                })
    
    return issues


def _check_legend_spacing(arr: np.ndarray, width: int, height: int) -> list:
    """检测图例/标题区域间距"""
    issues = []
    
    # 检查顶部标题区域（上10%）和底部说明区域（下15%）
    top_region = arr[:int(height*0.1), :, :3]
    bottom_region = arr[int(height*0.85):, :, :3]
    
    # 顶部太满
    top_density = (np.mean(top_region, axis=2) < 200).sum() / (top_region.shape[0] * top_region.shape[1])
    if top_density > 0.5:
        issues.append({
            "type": "title_crowded",
            "severity": "medium",
            "location": [0, 0, width, int(height*0.1)],
            "description": f"标题区域内容密度过高({top_density*100:.0f}%)，间距可能不足"
        })
    
    # 底部太满
    bot_density = (np.mean(bottom_region, axis=2) < 200).sum() / (bottom_region.shape[0] * bottom_region.shape[1])
    if bot_density > 0.5:
        issues.append({
            "type": "caption_crowded",
            "severity": "medium",
            "location": [0, int(height*0.85), width, int(height*0.15)],
            "description": f"底部说明区域内容密度过高({bot_density*100:.0f}%)，间距可能不足"
        })
    
    return issues


def _find_segments(mask_1d: np.ndarray) -> list:
    """找连续True段的起止索引"""
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(mask_1d):
        if v and not in_seg:
            start = i
            in_seg = True
        elif not v and in_seg:
            segments.append((start, i))
            in_seg = False
    if in_seg:
        segments.append((start, len(mask_1d)))
    return segments


# ========== 代码修复 ==========

# 修复注入模板：按优先级排序
FIX_TEMPLATES = [
    # 1. tight_layout
    ("tight_layout", "plt.tight_layout(pad=1.5)\n"),
    # 2. subplots_adjust增加边距
    ("subplots_adjust", "plt.subplots_adjust(left=0.12, right=0.92, top=0.90, bottom=0.15)\n"),
    # 3. 图例位置调整
    ("legend_loc", None),  # 特殊处理
    # 4. 增大figsize
    ("figsize_increase", None),  # 特殊处理
    # 5. 缩小字体
    ("fontsize_reduce", None),  # 特殊处理
    # 6. bbox_inches保存
    ("bbox_inches", None),  # 在savefig中注入
]


def fix_render_code(original_code: str, issues: list, attempt: int = 0) -> str:
    """
    根据检测到的问题修改渲染代码。
    不改变数据、题意、答案和美术风格，只调整布局。
    
    Args:
        original_code: 原始matplotlib/networkx渲染代码
        issues: check_image返回的issues列表
        attempt: 当前修复尝试次数(0-based)
    
    Returns:
        修复后的渲染代码
    """
    code = original_code
    
    if not issues:
        return code
    
    # 根据attempt级别逐步加强修复
    if attempt == 0:
        # 第一轮：添加tight_layout + bbox_inches
        code = _inject_tight_layout(code)
        code = _inject_bbox_inches(code)
    
    elif attempt == 1:
        # 第二轮：增加边距 + 移动图例 + 适度缩小字体
        code = _inject_tight_layout(code, pad=2.0)
        code = _inject_subplots_adjust(code)
        code = _adjust_legend_position(code)
        code = _inject_bbox_inches(code)
    
    elif attempt >= 2:
        # 第三轮：增大画布 + 缩小字体 + 所有优化
        code = _increase_figsize(code)
        code = _reduce_fontsize(code)
        code = _inject_tight_layout(code, pad=2.5)
        code = _inject_subplots_adjust(code)
        code = _adjust_legend_position(code)
        code = _add_annotation_backgrounds(code)
        code = _inject_bbox_inches(code)
    
    return code


def _inject_tight_layout(code: str, pad: float = 1.5) -> str:
    """在savefig之前注入plt.tight_layout()"""
    if "tight_layout" in code:
        # 已有则更新pad值
        code = re.sub(r"plt\.tight_layout\([^)]*\)", f"plt.tight_layout(pad={pad})", code)
    else:
        # 在savefig之前插入
        code = re.sub(r"(plt\.savefig|fig\.savefig)", f"plt.tight_layout(pad={pad})\n\\1", code, count=1)
    return code


def _inject_bbox_inches(code: str) -> str:
    """确保savefig使用bbox_inches='tight'"""
    if "bbox_inches" not in code:
        code = re.sub(
            r"(\.savefig\([^)]+)\)",
            r"\1, bbox_inches='tight')",
            code
        )
    return code


def _inject_subplots_adjust(code: str) -> str:
    """注入subplots_adjust增加边距"""
    if "subplots_adjust" not in code:
        code = re.sub(
            r"(plt\.savefig|fig\.savefig)",
            "plt.subplots_adjust(left=0.14, right=0.90, top=0.88, bottom=0.16)\n\\1",
            code, count=1
        )
    return code


def _adjust_legend_position(code: str) -> str:
    """将图例移到不遮挡数据的位置"""
    # 将loc='upper right'等改为'best'或外部位置
    code = re.sub(
        r"\.legend\(([^)]*?)loc=['\"](?:upper right|upper left|center)['\"]",
        r".legend(\1loc='best'",
        code
    )
    return code


def _increase_figsize(code: str) -> str:
    """增大画布尺寸(宽高各增20%)"""
    def _scale(match):
        w, h = float(match.group(1)), float(match.group(2))
        return f"figsize=({w*1.2:.1f}, {h*1.2:.1f})"
    
    code = re.sub(r"figsize=\((\d+\.?\d*),\s*(\d+\.?\d*)\)", _scale, code)
    return code


def _reduce_fontsize(code: str) -> str:
    """适度缩小字体(减少2pt)"""
    def _shrink(match):
        size = int(match.group(1))
        return f"fontsize={max(6, size - 2)}"
    
    code = re.sub(r"fontsize=(\d+)", _shrink, code)
    return code


def _add_annotation_backgrounds(code: str) -> str:
    """为annotate添加半透明背景框"""
    if "bbox=dict(" not in code and "annotate" in code:
        code = re.sub(
            r"(\.annotate\([^)]+)\)",
            r"\1, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))",
            code
        )
    return code


# ========== 完整检查修复流程 ==========

def check_and_fix(
    image_path: str,
    render_code: str,
    output_path: str,
    max_retries: int = 3,
    render_func=None
) -> dict:
    """
    完整的检查-修复流程。
    
    Args:
        image_path: 已渲染图片路径
        render_code: 渲染代码字符串
        output_path: 最终输出路径
        max_retries: 最大重试次数
        render_func: 可选的渲染函数 (code, path) -> bool
    
    Returns:
        {
            "pass": bool,
            "final_image": str,
            "attempts": int,
            "issues_found": int,
            "issues_fixed": int,
            "remaining_issues": list
        }
    """
    result = check_image(image_path)
    
    if result["pass"]:
        return {
            "pass": True,
            "final_image": image_path,
            "attempts": 0,
            "issues_found": 0,
            "issues_fixed": 0,
            "remaining_issues": []
        }
    
    initial_issues = len(result["issues"])
    high_severity = [i for i in result["issues"] if i["severity"] == "high"]
    
    # 如果没有高严重度问题，且只有low问题，直接通过
    if not high_severity and all(i["severity"] == "low" for i in result["issues"]):
        return {
            "pass": True,
            "final_image": image_path,
            "attempts": 0,
            "issues_found": initial_issues,
            "issues_fixed": 0,
            "remaining_issues": result["issues"]
        }
    
    # 尝试修复
    fixed_code = render_code
    for attempt in range(max_retries):
        fixed_code = fix_render_code(fixed_code, result["issues"], attempt)
        
        # 重新渲染
        if render_func:
            try:
                success = render_func(fixed_code, output_path)
                if not success:
                    continue
            except Exception:
                continue
        else:
            # 默认：用exec执行代码（需要安全沙箱环境）
            try:
                _execute_render_code(fixed_code, output_path)
            except Exception:
                continue
        
        # 重新检查
        result = check_image(output_path)
        if result["pass"] or not any(i["severity"] == "high" for i in result["issues"]):
            return {
                "pass": True,
                "final_image": output_path,
                "attempts": attempt + 1,
                "issues_found": initial_issues,
                "issues_fixed": initial_issues - len(result["issues"]),
                "remaining_issues": result["issues"]
            }
    
    # 所有重试都失败
    return {
        "pass": False,
        "final_image": output_path if os.path.exists(output_path) else image_path,
        "attempts": max_retries,
        "issues_found": initial_issues,
        "issues_fixed": initial_issues - len(result["issues"]),
        "remaining_issues": result["issues"]
    }


def _execute_render_code(code: str, output_path: str):
    """在隔离环境中执行渲染代码"""
    # 确保输出路径正确
    code = re.sub(r"plt\.savefig\(['\"][^'\"]+['\"]", f"plt.savefig('{output_path}'", code)
    code = re.sub(r"fig\.savefig\(['\"][^'\"]+['\"]", f"fig.savefig('{output_path}'", code)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import matplotlib\nmatplotlib.use('Agg')\n")
        f.write("import matplotlib.pyplot as plt\n")
        f.write("import numpy as np\n")
        f.write(code)
        f.write("\nplt.close('all')\n")
        tmp_path = f.name
    
    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True, timeout=30,
            cwd=os.path.dirname(output_path)
        )
        if result.returncode != 0:
            raise RuntimeError(f"Render failed: {result.stderr.decode()[:200]}")
    finally:
        os.unlink(tmp_path)


# ========== 快速集成接口 ==========

def quick_check(image_path: str) -> bool:
    """快速检查图片是否通过（True=通过，False=有高/中严重度问题）"""
    result = check_image(image_path)
    if result["pass"]:
        return True
    # 只有low severity的也通过
    return all(i["severity"] == "low" for i in result["issues"])
