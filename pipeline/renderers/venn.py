"""
Venn 图渲染函数
===============
使用 matplotlib-venn 库，圆的位置/大小/标签全部自动计算。
GPT 只需提供区域数值和标签。

data 格式:
{
  "set_labels": ["A", "B", "C"],        # 集合名称
  "region_values": {                     # 7个区域的值（可以是数字或文字）
    "100": 5, "010": 3, "110": 8,
    "001": 4, "101": 1, "011": 6, "111": 2
  },
  "highlight_regions": ["110", "100"],   # 可选：高亮哪些区域
  "elements": [                          # 可选：在区域内放置字母/符号
    {"name": "p", "region": "100"},
    {"name": "q", "region": "011"}
  ],
  "num_sets": 3                          # 2 或 3（默认3）
}
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.global_style import (
    finalize_figure,
    FIGSIZE, DPI, BACKGROUND, VENN_ALPHA, VENN_FONTSIZE,
    VENN_LABEL_FONTSIZE, PRIMARY_COLORS, HIGHLIGHT_COLOR,
    HIGHLIGHT_COLOR_ALT, LINE_WIDTH, apply_global_style
)


def render_venn(data: dict, diagram_type: str, output_path: str, style_override: dict = None) -> tuple:
    """渲染 Venn 图"""
    plt = apply_global_style()
    from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles

    # 从 data 或 diagram_type 推断集合数量
    num_sets = data.get("num_sets", None)
    if num_sets is None:
        # 从 diagram_type 推断: "venn_2sets" → 2, 否则 3
        if "2" in diagram_type:
            num_sets = 2
        elif "3" in diagram_type:
            num_sets = 3
        else:
            # 从 region_values 的 key 长度推断
            keys = list(data.get("region_values", {}).keys())
            if keys and max(len(k) for k in keys) <= 2:
                num_sets = 2
            else:
                num_sets = 3
    set_labels = data.get("set_labels", ["A", "B", "C"][:num_sets])
    region_values = data.get("region_values", {})
    highlight_regions = data.get("highlight_regions", [])
    elements = data.get("elements", [])

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax.set_facecolor(BACKGROUND)

    # 设置颜色
    colors = PRIMARY_COLORS[:num_sets]

    if num_sets == 2:
        # 2-set Venn
        subsets = (
            region_values.get("10", 0),
            region_values.get("01", 0),
            region_values.get("11", 0),
        )
        v = venn2(subsets=subsets, set_labels=set_labels,
                  set_colors=colors, alpha=VENN_ALPHA, ax=ax)
        venn2_circles(subsets=subsets, linewidth=LINE_WIDTH, ax=ax)
        region_ids = ["10", "01", "11"]
    else:
        # 3-set Venn
        subsets = (
            region_values.get("100", 0),
            region_values.get("010", 0),
            region_values.get("110", 0),
            region_values.get("001", 0),
            region_values.get("101", 0),
            region_values.get("011", 0),
            region_values.get("111", 0),
        )
        v = venn3(subsets=subsets, set_labels=set_labels,
                  set_colors=colors, alpha=VENN_ALPHA, ax=ax)
        venn3_circles(subsets=subsets, linewidth=LINE_WIDTH, ax=ax)
        region_ids = ["100", "010", "110", "001", "101", "011", "111"]

    if v is None:
        plt.close(fig)
        return False, "matplotlib-venn 返回 None（可能所有区域为0）"

    # 设置区域数字字号
    for rid in region_ids:
        label = v.get_label_by_id(rid)
        if label:
            label.set_fontsize(VENN_FONTSIZE)
            label.set_fontweight('bold')

    # 设置集合标签字号
    for sl in (v.set_labels or []):
        if sl:
            sl.set_fontsize(VENN_LABEL_FONTSIZE)
            sl.set_fontweight('bold')

    # 高亮区域
    for rid in highlight_regions:
        patch = v.get_patch_by_id(rid)
        if patch:
            patch.set_color(HIGHLIGHT_COLOR)
            patch.set_alpha(0.7)
            patch.set_edgecolor('#333333')
            patch.set_linewidth(2.0)

    # 放置元素标记（如点 p, q）
    if elements:
        # 获取每个区域的中心位置，放置标记
        for elem in elements:
            name = elem.get("name", "")
            region = elem.get("region", "")
            label_obj = v.get_label_by_id(region)
            if label_obj:
                # 在原来的数字标签旁边放置元素名
                pos = label_obj.get_position()
                ax.plot(pos[0], pos[1] - 0.08, 'ko', markersize=5)
                ax.text(pos[0] + 0.03, pos[1] - 0.08, name,
                        fontsize=VENN_FONTSIZE - 2, fontweight='bold',
                        ha='left', va='center')

    # 去掉坐标轴
    ax.set_axis_off()

    # 保存
    finalize_figure(fig, output_path, pad=1.0)

    return True, "Venn图渲染成功"
