"""
流程图/状态机渲染函数
====================
使用 graphviz 画流程图、状态机(DFA/NFA)、编译流程。
Graphviz 自带排版引擎，天然不会出现遮挡。

data 格式:
{
  "nodes": [
    {"id": "q0", "label": "开始", "shape": "circle"},
    {"id": "q1", "label": "状态1", "shape": "doublecircle"},
  ],
  "edges": [
    {"from": "q0", "to": "q1", "label": "a"},
    {"from": "q1", "to": "q0", "label": "b"},
  ],
  "rankdir": "LR",  # LR=左到右, TB=上到下
  "title": ""       # 可选标题
}
"""

import sys
import os
import subprocess
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.global_style import (
    DPI, BACKGROUND, LINE_COLOR, NODE_COLOR, HIGHLIGHT_COLOR,
    LABEL_FONTSIZE, NODE_FONT_SIZE, EDGE_FONT_SIZE, apply_global_style
)

# 节点填充色调色板 — 柔和、区分度高、白底下好看
_NODE_PALETTE = [
    '#E8F4FD',  # 浅蓝
    '#FDE8E8',  # 浅红/粉
    '#E8FDE8',  # 浅绿
    '#FDF4E8',  # 浅橙
    '#F0E8FD',  # 浅紫
    '#E8FDFD',  # 浅青
    '#FDE8F4',  # 浅玫瑰
    '#F4FDE8',  # 浅黄绿
]

# 边颜色调色板 — 稍深、与节点搭配
_EDGE_PALETTE = [
    '#4A90D9',  # 蓝
    '#E85D5D',  # 红
    '#5CB85C',  # 绿
    '#F0AD4E',  # 橙
    '#9B59B6',  # 紫
    '#17A2B8',  # 青
    '#E84393',  # 玫瑰
    '#6C757D',  # 灰
]


def render_flowchart(data: dict, diagram_type: str, output_path: str, style_override: dict = None) -> tuple:
    """渲染流程图/状态机"""
    import graphviz

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    rankdir = data.get("rankdir", "LR")

    # 构建 dot graph
    dot = graphviz.Digraph(format='png')
    dot.attr(rankdir=rankdir, dpi=str(DPI), bgcolor=BACKGROUND,
             size='10,8', ratio='compress', pad='0.5')
    dot.attr('node', style='filled', fillcolor='#E8F4FD',
             color=LINE_COLOR, fontsize=str(NODE_FONT_SIZE),
             fontname='DejaVu Sans', penwidth='1.5')
    dot.attr('edge', color=LINE_COLOR, fontsize=str(EDGE_FONT_SIZE),
             fontname='DejaVu Sans', penwidth='1.5', arrowsize='0.8')

    # 添加节点（按序号轮换颜色）
    for i, node in enumerate(nodes):
        nid = str(node.get("id", ""))
        label = str(node.get("label", nid))
        shape = node.get("shape", "circle")
        fill = node.get("color", None)
        if fill == "highlight":
            fill = HIGHLIGHT_COLOR
        elif fill is None or fill == "#E8F4FD":
            # 自动分配调色板颜色
            fill = _NODE_PALETTE[i % len(_NODE_PALETTE)]
        dot.node(nid, label=label, shape=shape, fillcolor=fill)

    # 添加边（按序号轮换颜色）
    for i, edge in enumerate(edges):
        src = str(edge.get("from", ""))
        dst = str(edge.get("to", ""))
        label = str(edge.get("label", ""))
        style = edge.get("style", "solid")
        edge_color = edge.get("color", _EDGE_PALETTE[i % len(_EDGE_PALETTE)])
        dot.edge(src, dst, label=label, style=style, color=edge_color,
                 fontcolor=edge_color)

    # 渲染为 PNG
    try:
        # graphviz render 返回输出文件路径
        # 先渲染到临时文件再移动
        out_base = output_path.replace('.png', '')
        dot.render(out_base, cleanup=True)
        # graphviz 可能会加 .png 后缀
        rendered = out_base + '.png'
        if os.path.exists(rendered) and rendered != output_path:
            os.rename(rendered, output_path)
        elif not os.path.exists(output_path):
            return False, f"graphviz 渲染后文件未找到: {rendered}"
        return True, "流程图渲染成功"
    except Exception as e:
        return False, f"graphviz 渲染失败: {str(e)}"
