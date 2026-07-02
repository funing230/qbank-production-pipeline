"""
图论/网络图渲染函数
==================
使用 networkx 画有向图、无向图、树、Hasse图。
节点位置由库的布局算法自动计算，不会重叠。

data 格式:
{
  "graph_type": "directed | undirected | tree | hasse",
  "nodes": ["A", "B", "C", "D"],
  "edges": [["A","B"], ["B","C"], ["A","D"]],
  "edge_labels": {"A-B": "5", "B-C": "3"},     # 可选：边的标签
  "node_colors": {"A": "highlight"},             # 可选：特殊着色
  "weighted": false,                             # 可选：是否加权
  "layout": "spring | circular | tree | planar"  # 可选：布局算法
}
"""

import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.global_style import (
    finalize_figure,
    FIGSIZE, DPI, BACKGROUND, NODE_SIZE, NODE_FONT_SIZE,
    EDGE_FONT_SIZE, NODE_COLOR, NODE_EDGE_COLOR, LINE_COLOR,
    LINE_WIDTH, HIGHLIGHT_COLOR, PRIMARY_COLORS, apply_global_style
)

# 节点颜色调色板（柔和、区分度高）
_NODE_PALETTE = [
    '#AED6F1',  # 浅蓝
    '#F9C0C0',  # 浅红
    '#A9DFBF',  # 浅绿
    '#FAD7A0',  # 浅橙
    '#D2B4DE',  # 浅紫
    '#A3E4D7',  # 浅青
    '#F5CBA7',  # 浅棕
    '#AED9E0',  # 浅湖蓝
]

# 边颜色调色板
_EDGE_PALETTE = [
    '#2980B9',  # 蓝
    '#C0392B',  # 红
    '#27AE60',  # 绿
    '#E67E22',  # 橙
    '#8E44AD',  # 紫
    '#16A085',  # 青
    '#D35400',  # 深橙
    '#2C3E50',  # 深灰蓝
]

# Unicode subscript/superscript → ASCII 映射
_UNICODE_SUB_MAP = str.maketrans(
    '₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₔₕₖₗₘₙₚₛₜ',
    '0123456789+-=()aeoxəhklmnpst'
)
_UNICODE_SUP_MAP = str.maketrans(
    '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ',
    '0123456789+-=()ni'
)


def _has_cjk(s: str) -> bool:
    """检测字符串是否包含 CJK 字符"""
    for ch in s:
        if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf':
            return True
    return False


def _unicode_to_ascii(s: str) -> str:
    """将 Unicode subscript/superscript 转回 ASCII（用于不支持的字体）"""
    s = s.translate(_UNICODE_SUB_MAP)
    s = s.translate(_UNICODE_SUP_MAP)
    return s


def _pick_font(label: str) -> str:
    """根据标签内容选择字体：含中文用 WenQuanYi，纯英文/数学用 DejaVu Sans"""
    if _has_cjk(label):
        return 'WenQuanYi Zen Hei'
    return 'DejaVu Sans'


def _prepare_label(label: str) -> str:
    """
    准备标签文本：
    - 含中文 → 把 Unicode subscript 转回 ASCII（WenQuanYi 没有 subscript glyph）
    - 纯英文 → 保留 Unicode subscript（DejaVu Sans 支持）
    """
    if _has_cjk(label):
        return _unicode_to_ascii(label)
    return label


def render_graph(data: dict, diagram_type: str, output_path: str, style_override: dict = None) -> tuple:
    """渲染图论/网络图"""
    plt = apply_global_style()
    import networkx as nx

    graph_type = data.get("graph_type", "directed")
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    edge_labels_raw = data.get("edge_labels", {})
    node_colors_map = data.get("node_colors", {})
    layout_algo = data.get("layout", "spring")

    # 创建图
    if graph_type in ("directed", "tree"):
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    G.add_nodes_from(nodes)
    for edge in edges:
        if len(edge) >= 3:
            G.add_edge(edge[0], edge[1], weight=edge[2])
        elif len(edge) == 2:
            G.add_edge(edge[0], edge[1])

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax.set_facecolor(BACKGROUND)

    # 选择布局算法
    if layout_algo == "circular":
        pos = nx.circular_layout(G)
    elif layout_algo == "tree" or graph_type == "tree":
        try:
            root = nodes[0] if nodes else None
            pos = nx.bfs_layout(G, root) if root else nx.spring_layout(G, seed=42)
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=2.0)
    elif layout_algo == "planar":
        try:
            pos = nx.planar_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=2.0)
    elif layout_algo == "shell":
        pos = nx.shell_layout(G)
    elif layout_algo == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        # spring layout with wider spacing
        pos = nx.spring_layout(G, seed=42, k=2.5 / (len(nodes) ** 0.5 + 0.1))

    # 节点颜色（按序号轮换调色板）
    colors = []
    for i, n in enumerate(G.nodes()):
        if node_colors_map.get(str(n)) == "highlight":
            colors.append(HIGHLIGHT_COLOR)
        elif node_colors_map.get(str(n)) in PRIMARY_COLORS:
            colors.append(node_colors_map[str(n)])
        else:
            colors.append(_NODE_PALETTE[i % len(_NODE_PALETTE)])

    # 画节点
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=colors,
                           node_size=NODE_SIZE,
                           edgecolors=NODE_EDGE_COLOR,
                           linewidths=LINE_WIDTH)

    # 边颜色（按序号轮换）
    edge_colors = [_EDGE_PALETTE[i % len(_EDGE_PALETTE)] for i in range(len(G.edges()))]
    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color=edge_colors,
                           width=LINE_WIDTH,
                           style='solid',
                           arrows=(graph_type == "directed"),
                           arrowsize=20,
                           arrowstyle='-|>',
                           connectionstyle='arc3,rad=0.1' if graph_type == "directed" else 'arc3,rad=0')

    # 逐个画节点标签（根据内容动态选择字体）
    for node, (x, y) in pos.items():
        label = _prepare_label(str(node))
        font = _pick_font(str(node))
        ax.text(x, y, label,
                fontsize=NODE_FONT_SIZE, fontweight='bold',
                color='#1a1a1a', fontfamily=font,
                ha='center', va='center', zorder=5)

    # 画边标签
    if edge_labels_raw:
        edge_labels = {}
        for key, val in edge_labels_raw.items():
            parts = key.replace("-", ",").replace("→", ",").split(",")
            if len(parts) == 2:
                src = parts[0].strip()
                dst = parts[1].strip()
                if (src, dst) in G.edges() or (dst, src) in G.edges():
                    edge_labels[(src, dst)] = str(val)
        if edge_labels:
            # 检测边标签是否含中文决定字体
            sample_val = next(iter(edge_labels.values()), "")
            edge_font = _pick_font(sample_val)
            # 准备标签文本
            display_labels = {k: _prepare_label(v) for k, v in edge_labels.items()}
            nx.draw_networkx_edge_labels(G, pos, ax=ax,
                                         edge_labels=display_labels,
                                         font_size=EDGE_FONT_SIZE,
                                         font_color='#333333',
                                         font_family=edge_font)

    ax.set_axis_off()
    finalize_figure(fig, output_path, pad=1.5)

    return True, "图论图渲染成功"
