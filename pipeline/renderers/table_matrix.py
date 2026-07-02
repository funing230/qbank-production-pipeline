"""
表格/矩阵渲染函数
=================
使用 Pillow 精确绘制表格和矩阵。
格子布局天然不会遮挡。

data 格式 (表格):
{
  "table_type": "table | matrix | truth_table",
  "headers": ["变量", "A", "B", "A∧B"],
  "rows": [
    ["T", "T", "T"],
    ["T", "F", "F"],
    ["F", "T", "F"],
    ["F", "F", "F"]
  ],
  "highlight_cells": [[1, 2]],  # 可选：高亮哪些单元格 [row, col]
  "title": ""                    # 可选
}

data 格式 (矩阵):
{
  "table_type": "matrix",
  "matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
  "brackets": "square",  # square | round | none
  "label": "A ="
}
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.global_style import (
    finalize_figure,
    DPI, BACKGROUND, LABEL_FONTSIZE, LINE_COLOR,
    PRIMARY_COLORS, HIGHLIGHT_COLOR, apply_global_style
)


def render_table_matrix(data: dict, diagram_type: str, output_path: str, style_override: dict = None) -> tuple:
    """渲染表格或矩阵"""
    table_type = data.get("table_type", "table")

    if table_type == "matrix":
        return _render_matrix(data, output_path)
    else:
        return _render_table(data, output_path)


def _render_table(data, output_path):
    """渲染表格（使用 matplotlib table）"""
    plt = apply_global_style()

    headers = data.get("headers", [])
    rows = data.get("rows", [])
    highlight_cells = data.get("highlight_cells", [])

    if not rows:
        return False, "rows 为空"

    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(headers) * 1.5), max(4, len(rows) * 0.6 + 1)))
    ax.set_axis_off()

    # 构建表格数据
    cell_text = rows
    if headers:
        col_labels = headers
    else:
        col_labels = None

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )

    # 设置样式
    table.auto_set_font_size(False)
    table.set_fontsize(LABEL_FONTSIZE)
    table.scale(1.3, 1.8)

    # 表头样式
    if headers:
        for j in range(len(headers)):
            cell = table[0, j]
            cell.set_facecolor('#D6EAF8')
            cell.set_text_props(fontweight='bold')
            cell.set_edgecolor('#666666')

    # 数据行样式
    for i in range(len(rows)):
        for j in range(len(rows[i]) if i < len(rows) else 0):
            row_idx = i + (1 if headers else 0)
            cell = table[row_idx, j]
            cell.set_edgecolor('#999999')
            # 交替行颜色
            if i % 2 == 1:
                cell.set_facecolor('#F8F9FA')

    # 高亮单元格
    for hc in highlight_cells:
        if len(hc) == 2:
            row_idx = hc[0] + (1 if headers else 0)
            col_idx = hc[1]
            try:
                cell = table[row_idx, col_idx]
                cell.set_facecolor(HIGHLIGHT_COLOR)
            except KeyError:
                pass

    finalize_figure(fig, output_path, pad=1.0)
    return True, "表格渲染成功"


def _render_matrix(data, output_path):
    """渲染数学矩阵"""
    plt = apply_global_style()

    matrix = data.get("matrix", [[]])
    brackets = data.get("brackets", "square")
    label = data.get("label", "")

    n_rows = len(matrix)
    n_cols = len(matrix[0]) if matrix else 0

    fig, ax = plt.subplots(1, 1, figsize=(max(6, n_cols * 1.2 + 2), max(4, n_rows * 0.8 + 1)))
    ax.set_axis_off()
    ax.set_xlim(-2, n_cols + 2)
    ax.set_ylim(-1, n_rows + 1)

    # 画括号
    bracket_x_left = -0.3
    bracket_x_right = n_cols - 0.7
    bracket_y_top = n_rows - 0.3
    bracket_y_bottom = -0.7

    if brackets == "square":
        left_bracket = "["
        right_bracket = "]"
    elif brackets == "round":
        left_bracket = "("
        right_bracket = ")"
    else:
        left_bracket = ""
        right_bracket = ""

    # 大括号
    if left_bracket:
        ax.text(bracket_x_left, (bracket_y_top + bracket_y_bottom) / 2,
                left_bracket, fontsize=LABEL_FONTSIZE * 2 + n_rows * 4,
                ha='center', va='center', fontfamily='DejaVu Sans')
        ax.text(bracket_x_right + 1.0, (bracket_y_top + bracket_y_bottom) / 2,
                right_bracket, fontsize=LABEL_FONTSIZE * 2 + n_rows * 4,
                ha='center', va='center', fontfamily='DejaVu Sans')

    # 填充数值
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i][j] if j < len(matrix[i]) else ""
            ax.text(j, n_rows - 1 - i, str(val),
                    fontsize=LABEL_FONTSIZE + 2, ha='center', va='center',
                    fontweight='bold')

    # 标签 (如 "A = ")
    if label:
        ax.text(bracket_x_left - 1.2, (bracket_y_top + bracket_y_bottom) / 2,
                label, fontsize=LABEL_FONTSIZE + 2, ha='right', va='center',
                fontweight='bold')

    finalize_figure(fig, output_path, pad=1.0)
    return True, "矩阵渲染成功"
