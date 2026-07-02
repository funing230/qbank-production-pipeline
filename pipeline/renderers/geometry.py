"""
几何图形渲染函数（V2 扩展版）
===============================
支持两种 data 格式:

格式A (shapes 列表):
{
  "shapes": [
    {"type": "polygon|circle|line|arc|point|arrow|support|dimension|angle_arc|curved_arrow|text|rect", ...}
  ],
  "show_grid": false,
  "show_axes": true,
  "equal_aspect": true,
  "x_range": null, "y_range": null,
  "title": ""
}

格式B (3D / 扩展结构):
{
  "coordinate_system": {...},
  "points": [...],
  "vectors": [...],
  "rotations": [...],
  "view": {"azimuth_degrees": 35, "elevation_degrees": 25}
}

字段名兼容: from/to ↔ start/end, position ↔ pos
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.global_style import (
    finalize_figure,
    FIGSIZE, DPI, BACKGROUND, LABEL_FONTSIZE, LINE_WIDTH,
    THICK_LINE_WIDTH, PRIMARY_COLORS, EXTENDED_COLORS, LINE_COLOR,
    HIGHLIGHT_COLOR, GRID_COLOR, apply_global_style
)


def _get_start(shape):
    """兼容 start/from 字段"""
    return shape.get("start") or shape.get("from") or [0, 0]


def _get_end(shape):
    """兼容 end/to 字段"""
    return shape.get("end") or shape.get("to") or [1, 1]


def _get_pos(shape):
    """兼容 pos/position/center 字段"""
    return shape.get("pos") or shape.get("position") or shape.get("center") or [0, 0]


def _draw_support(ax, shape, color):
    """绘制支座符号（铰支座/滚动支座/固定端）"""
    from matplotlib.patches import FancyArrowPatch, RegularPolygon, Circle
    pos = _get_pos(shape)
    support_type = shape.get("support_type", "pin")
    label = shape.get("label", "")
    x, y = pos[0], pos[1]
    size = 0.3

    if support_type in ("pin", "hinge", "铰支座"):
        # 三角形
        tri_pts = np.array([[x, y], [x - size, y - size * 1.2], [x + size, y - size * 1.2]])
        from matplotlib.patches import Polygon
        tri = Polygon(tri_pts, closed=True, fill=False, edgecolor=color, linewidth=LINE_WIDTH)
        ax.add_patch(tri)
        # 接地线
        ax.plot([x - size * 1.2, x + size * 1.2], [y - size * 1.2, y - size * 1.2],
                color=color, linewidth=LINE_WIDTH)
        # 斜线表示地面
        for xi in np.linspace(x - size * 1.1, x + size * 0.9, 5):
            ax.plot([xi, xi - 0.12], [y - size * 1.2, y - size * 1.5],
                    color=color, linewidth=0.8)

    elif support_type in ("roller", "滚动支座"):
        # 三角形 + 圆
        tri_pts = np.array([[x, y], [x - size, y - size], [x + size, y - size]])
        from matplotlib.patches import Polygon
        tri = Polygon(tri_pts, closed=True, fill=False, edgecolor=color, linewidth=LINE_WIDTH)
        ax.add_patch(tri)
        # 小圆表示滚动
        circle_r = size * 0.2
        for cx in [x - size * 0.5, x, x + size * 0.5]:
            c = Circle((cx, y - size - circle_r), circle_r, fill=False,
                       edgecolor=color, linewidth=0.8)
            ax.add_patch(c)
        # 地面线
        ax.plot([x - size * 1.2, x + size * 1.2], [y - size - circle_r * 2.2, y - size - circle_r * 2.2],
                color=color, linewidth=LINE_WIDTH)

    elif support_type in ("fixed", "固定端"):
        # 固定墙壁
        ax.plot([x, x], [y - size, y + size], color=color, linewidth=THICK_LINE_WIDTH)
        for yi in np.linspace(y - size * 0.8, y + size * 0.8, 5):
            ax.plot([x, x - 0.2], [yi, yi - 0.1], color=color, linewidth=0.8)

    if label:
        ax.text(x, y - size * 2, label, fontsize=LABEL_FONTSIZE - 2,
                ha='center', va='top', color=color)


def _draw_dimension(ax, shape, color):
    """绘制标注线（dimension line with arrows）"""
    start = _get_start(shape)
    end = _get_end(shape)
    label = shape.get("label", "")

    # 双向箭头
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle='<->', color=color, lw=1.2))
    # 标注文字
    mx = (start[0] + end[0]) / 2
    my = (start[1] + end[1]) / 2
    ax.text(mx, my - 0.15, label, fontsize=LABEL_FONTSIZE - 2,
            ha='center', va='top', color=color,
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.8))


def _draw_angle_arc(ax, shape, color):
    """绘制角度弧线"""
    from matplotlib.patches import Arc
    center = shape.get("center", [0, 0])
    radius = shape.get("radius", 0.5)
    start_angle = shape.get("start_angle") or shape.get("angle_start", 0)
    end_angle = shape.get("end_angle") or shape.get("angle_end", 90)
    label = shape.get("label", "")

    arc = Arc(center, 2 * radius, 2 * radius,
              angle=0, theta1=start_angle, theta2=end_angle,
              color=color, linewidth=LINE_WIDTH)
    ax.add_patch(arc)
    if label:
        mid_angle = np.radians((start_angle + end_angle) / 2)
        lx = center[0] + radius * 1.4 * np.cos(mid_angle)
        ly = center[1] + radius * 1.4 * np.sin(mid_angle)
        ax.text(lx, ly, label, fontsize=LABEL_FONTSIZE - 2,
                ha='center', va='center', color=color)


def _draw_curved_arrow(ax, shape, color):
    """绘制弯曲箭头（力矩符号）"""
    from matplotlib.patches import FancyArrowPatch, Arc
    center = shape.get("center", [0, 0])
    radius = shape.get("radius", 0.4)
    direction = shape.get("direction", "counterclockwise")
    label = shape.get("label", "")

    # 画弧
    if direction in ("clockwise", "cw"):
        theta1, theta2 = 30, 330
    else:
        theta1, theta2 = 30, 330

    arc = Arc(center, 2 * radius, 2 * radius,
              angle=0, theta1=theta1, theta2=theta2,
              color=color, linewidth=LINE_WIDTH)
    ax.add_patch(arc)

    # 箭头
    end_angle = np.radians(theta2 if direction in ("clockwise", "cw") else theta1)
    arrow_x = center[0] + radius * np.cos(end_angle)
    arrow_y = center[1] + radius * np.sin(end_angle)
    ax.annotate("", xy=(arrow_x, arrow_y),
                xytext=(arrow_x - 0.05, arrow_y + 0.05),
                arrowprops=dict(arrowstyle='->', color=color, lw=LINE_WIDTH))

    if label:
        ax.text(center[0], center[1] + radius + 0.2, label,
                fontsize=LABEL_FONTSIZE - 2, ha='center', va='bottom', color=color)


def _draw_text(ax, shape, color):
    """绘制独立文本标签"""
    pos = _get_pos(shape)
    text = shape.get("text") or shape.get("label", "")
    fontsize = shape.get("fontsize", LABEL_FONTSIZE)
    ha = shape.get("ha", "center")
    va = shape.get("va", "center")
    ax.text(pos[0], pos[1], text, fontsize=fontsize, ha=ha, va=va, color=color)


def _draw_rect(ax, shape, color):
    """绘制矩形"""
    from matplotlib.patches import Rectangle
    pos = _get_pos(shape)  # 左下角
    width = shape.get("width", 1)
    height = shape.get("height", 1)
    fill = shape.get("fill", False)
    style = shape.get("style", "solid")
    label = shape.get("label", "")

    rect = Rectangle(pos, width, height, fill=fill,
                     edgecolor=color, facecolor=HIGHLIGHT_COLOR if fill else 'none',
                     linewidth=LINE_WIDTH, linestyle=style, alpha=0.3 if fill else 1.0)
    ax.add_patch(rect)
    if label:
        cx = pos[0] + width / 2
        cy = pos[1] + height / 2
        ax.text(cx, cy, label, fontsize=LABEL_FONTSIZE - 1, ha='center', va='center')


def render_geometry_3d(data: dict, output_path: str) -> tuple:
    """渲染3D几何图形（坐标/旋转/向量）"""
    plt = apply_global_style()
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection='3d')

    view = data.get("view", {})
    azim = view.get("azimuth_degrees", 35)
    elev = view.get("elevation_degrees", 25)
    ax.view_init(elev=elev, azim=azim)

    # 绘制坐标轴
    coord_sys = data.get("coordinate_system", {})
    axes_info = coord_sys.get("axes", [])
    axis_len = 2.0
    for axis in axes_info:
        d = axis.get("direction", [1, 0, 0])
        label = axis.get("label", "")
        ax.quiver(0, 0, 0, d[0] * axis_len, d[1] * axis_len, d[2] * axis_len,
                  color='#555555', arrow_length_ratio=0.08, linewidth=1.5)
        ax.text(d[0] * axis_len * 1.15, d[1] * axis_len * 1.15, d[2] * axis_len * 1.15,
                label, fontsize=LABEL_FONTSIZE, fontweight='bold')

    origin_label = coord_sys.get("origin_label", "O")
    ax.text(0, 0, 0, f"  {origin_label}", fontsize=LABEL_FONTSIZE - 1)

    # 绘制向量
    color_map = {"blue": "#4A90D9", "red": "#E85D5D", "green": "#5CB85C",
                 "gray": "#888888", "orange": "#F0AD4E", "purple": "#9B59B6"}
    vectors = data.get("vectors", [])
    for vec in vectors:
        fr = vec.get("from", [0, 0, 0])
        to = vec.get("to", [1, 0, 0])
        color = color_map.get(vec.get("color", "blue"), "#4A90D9")
        label = vec.get("label", "")
        dx, dy, dz = to[0] - fr[0], to[1] - fr[1], to[2] - fr[2]
        ax.quiver(fr[0], fr[1], fr[2], dx, dy, dz,
                  color=color, arrow_length_ratio=0.1, linewidth=THICK_LINE_WIDTH)
        if label:
            mx = (fr[0] + to[0]) / 2
            my = (fr[1] + to[1]) / 2
            mz = (fr[2] + to[2]) / 2
            ax.text(mx, my, mz, f" {label}", fontsize=LABEL_FONTSIZE - 2, color=color)

    # 绘制点
    points = data.get("points", [])
    for pt in points:
        coords = pt.get("coords", [0, 0, 0])
        label = pt.get("label", "")
        style = pt.get("style", "filled")
        marker = 'o' if 'filled' in style else '^'
        color = "#E85D5D" if "highlight" in style else "#4A90D9"
        ax.scatter(*coords, c=color, s=80, marker=marker, zorder=5)
        if label:
            ax.text(coords[0] + 0.1, coords[1] + 0.1, coords[2] + 0.1,
                    label, fontsize=LABEL_FONTSIZE - 1, fontweight='bold')

    # 绘制旋转弧线
    rotations = data.get("rotations", [])
    for rot in rotations:
        from_pt = None
        to_pt = None
        for pt in points:
            if pt.get("id") == rot.get("from_point"):
                from_pt = pt.get("coords")
            if pt.get("id") == rot.get("to_point"):
                to_pt = pt.get("coords")
        if from_pt and to_pt:
            # 画虚线弧线近似
            t = np.linspace(0, 1, 20)
            # 简单线性插值（实际应该是旋转弧，但视觉近似足够）
            arc_x = from_pt[0] + (to_pt[0] - from_pt[0]) * t
            arc_y = from_pt[1] + (to_pt[1] - from_pt[1]) * t
            arc_z = from_pt[2] + (to_pt[2] - from_pt[2]) * t
            # 给弧线一些弯曲
            mid_offset = 0.3
            arc_x += mid_offset * np.sin(np.pi * t)
            arc_y += mid_offset * np.sin(np.pi * t) * 0.5
            ax.plot(arc_x, arc_y, arc_z, '--', color=EXTENDED_COLORS[rot.get("order", 1) % len(EXTENDED_COLORS)],
                    linewidth=1.5)
            # 标签
            label = rot.get("arc_label", "")
            if label:
                mid_idx = len(t) // 2
                ax.text(arc_x[mid_idx], arc_y[mid_idx], arc_z[mid_idx],
                        f" {label}", fontsize=LABEL_FONTSIZE - 3)

    # 设置范围
    all_coords = [[0, 0, 0]]
    for pt in points:
        all_coords.append(pt.get("coords", [0, 0, 0]))
    for vec in vectors:
        all_coords.append(vec.get("to", [0, 0, 0]))
    all_coords = np.array(all_coords)
    margin = 0.5
    ax.set_xlim(all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
    ax.set_ylim(all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)
    ax.set_zlim(all_coords[:, 2].min() - margin, all_coords[:, 2].max() + margin)

    if data.get("show_grid", True):
        ax.grid(True, alpha=0.3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    finalize_figure(fig, output_path)
    return True, "3D几何图形渲染成功"


def render_geometry(data: dict, diagram_type: str, output_path: str, style_override: dict = None) -> tuple:
    """渲染几何图形（2D / 3D 自动判断）"""

    # 判断是否为3D格式
    if "coordinate_system" in data or "vectors" in data or "view" in data:
        # 检查是否含3D坐标
        points = data.get("points", [])
        vectors = data.get("vectors", [])
        is_3d = False
        for pt in points:
            if len(pt.get("coords", [])) == 3:
                is_3d = True
                break
        for vec in vectors:
            if len(vec.get("to", [])) == 3:
                is_3d = True
                break
        if is_3d:
            return render_geometry_3d(data, output_path)

    # 2D 渲染
    plt = apply_global_style()
    from matplotlib.patches import Circle, Arc, FancyArrowPatch, Polygon

    shapes = data.get("shapes", [])

    # 如果没有 shapes 但有 points/vectors（2D版本），转换为 shapes
    if not shapes and ("points" in data or "vectors" in data):
        shapes = []
        for pt in data.get("points", []):
            coords = pt.get("coords", pt.get("pos", [0, 0]))
            shapes.append({"type": "point", "pos": coords[:2], "label": pt.get("label", "")})
        for vec in data.get("vectors", []):
            fr = vec.get("from", [0, 0])[:2]
            to = vec.get("to", [1, 0])[:2]
            shapes.append({"type": "arrow", "start": fr, "end": to,
                           "label": vec.get("label", ""), "color_name": vec.get("color")})

    show_grid = data.get("show_grid", False)
    show_axes = data.get("show_axes", True)
    equal_aspect = data.get("equal_aspect", True)
    x_range = data.get("x_range", None)
    y_range = data.get("y_range", None)

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    color_map = {"blue": "#4A90D9", "red": "#E85D5D", "green": "#5CB85C",
                 "gray": "#888888", "orange": "#F0AD4E", "purple": "#9B59B6",
                 "black": "#2C3E50", "brown": "#8B4513"}

    for i, shape in enumerate(shapes):
        stype = shape.get("type", "")
        color_idx = shape.get("color", i)
        # 支持颜色名称
        color_name = shape.get("color_name")
        if color_name and color_name in color_map:
            color = color_map[color_name]
        elif isinstance(color_idx, str) and color_idx in color_map:
            color = color_map[color_idx]
        elif isinstance(color_idx, int):
            color = EXTENDED_COLORS[color_idx % len(EXTENDED_COLORS)]
        else:
            color = EXTENDED_COLORS[i % len(EXTENDED_COLORS)]

        fill = shape.get("fill", False)
        style = shape.get("style", "solid")
        label = shape.get("label", "")

        if stype == "polygon":
            vertices = shape.get("vertices", [])
            if not vertices:
                continue
            poly = Polygon(vertices, closed=True, fill=fill,
                           edgecolor=color, facecolor=HIGHLIGHT_COLOR if fill else 'none',
                           linewidth=THICK_LINE_WIDTH, linestyle=style, alpha=0.3 if fill else 1.0)
            ax.add_patch(poly)

            # 顶点标签
            vertex_labels = shape.get("vertex_labels", [])
            for j, v in enumerate(vertices):
                ax.plot(v[0], v[1], 'o', color=color, markersize=6, zorder=5)
                if j < len(vertex_labels):
                    cx = sum(vv[0] for vv in vertices) / len(vertices)
                    cy = sum(vv[1] for vv in vertices) / len(vertices)
                    dx = v[0] - cx
                    dy = v[1] - cy
                    norm = (dx**2 + dy**2) ** 0.5 + 0.001
                    offset_x = dx / norm * 0.3
                    offset_y = dy / norm * 0.3
                    ax.text(v[0] + offset_x, v[1] + offset_y, vertex_labels[j],
                            fontsize=LABEL_FONTSIZE, fontweight='bold',
                            ha='center', va='center')

        elif stype == "circle":
            center = shape.get("center", [0, 0])
            radius = shape.get("radius", 1)
            circle = Circle(center, radius, fill=fill,
                            edgecolor=color, facecolor=HIGHLIGHT_COLOR if fill else 'none',
                            linewidth=LINE_WIDTH, linestyle=style, alpha=0.3 if fill else 1.0)
            ax.add_patch(circle)
            if label:
                ax.text(center[0], center[1], label,
                        fontsize=LABEL_FONTSIZE - 1, ha='center', va='center')

        elif stype == "line":
            start = _get_start(shape)
            end = _get_end(shape)
            linestyle = style if style != "force" else "solid"
            lw = THICK_LINE_WIDTH if shape.get("style") == "force" else LINE_WIDTH
            ax.plot([start[0], end[0]], [start[1], end[1]],
                    color=color, linewidth=lw, linestyle=linestyle)
            if label:
                mx = (start[0] + end[0]) / 2
                my = (start[1] + end[1]) / 2
                ax.text(mx, my + 0.15, label, fontsize=LABEL_FONTSIZE - 1,
                        ha='center', va='bottom', color=color)

        elif stype == "arc":
            _draw_angle_arc(ax, shape, color)

        elif stype == "angle_arc":
            _draw_angle_arc(ax, shape, color)

        elif stype == "point":
            pos = _get_pos(shape)
            ax.plot(pos[0], pos[1], 'o', color=color, markersize=8, zorder=5)
            if label:
                ax.text(pos[0] + 0.15, pos[1] + 0.15, label,
                        fontsize=LABEL_FONTSIZE, fontweight='bold',
                        ha='left', va='bottom')

        elif stype == "arrow":
            start = _get_start(shape)
            end = _get_end(shape)
            ax.annotate("", xy=end, xytext=start,
                        arrowprops=dict(arrowstyle='->', color=color, lw=LINE_WIDTH + 0.5))
            if label:
                mx = (start[0] + end[0]) / 2
                my = (start[1] + end[1]) / 2
                ax.text(mx + 0.1, my + 0.1, label, fontsize=LABEL_FONTSIZE - 1, color=color)

        elif stype == "support":
            _draw_support(ax, shape, color)

        elif stype == "dimension":
            _draw_dimension(ax, shape, color)

        elif stype == "curved_arrow":
            _draw_curved_arrow(ax, shape, color)

        elif stype in ("text", "annotation"):
            _draw_text(ax, shape, color)

        elif stype in ("rect", "rectangle"):
            _draw_rect(ax, shape, color)

        elif stype == "distributed_load":
            # 分布荷载（梯形/均布荷载箭头组）
            start = _get_start(shape)
            end = _get_end(shape)
            n_arrows = shape.get("n_arrows", 5)
            intensity_start = shape.get("intensity_start", 0.5)
            intensity_end = shape.get("intensity_end", 0.5)
            for j in range(n_arrows):
                t = j / max(n_arrows - 1, 1)
                x = start[0] + (end[0] - start[0]) * t
                y_base = start[1] + (end[1] - start[1]) * t
                h = intensity_start + (intensity_end - intensity_start) * t
                ax.annotate("", xy=(x, y_base), xytext=(x, y_base + h),
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.0))
            # 顶部连线
            ax.plot([start[0], end[0]],
                    [start[1] + intensity_start, end[1] + intensity_end],
                    color=color, linewidth=1.0)
            if label:
                mx = (start[0] + end[0]) / 2
                my = max(start[1] + intensity_start, end[1] + intensity_end) + 0.15
                ax.text(mx, my, label, fontsize=LABEL_FONTSIZE - 2,
                        ha='center', va='bottom', color=color)

        else:
            # 未知类型，尝试作为点绘制
            pos = _get_pos(shape)
            if pos != [0, 0]:
                ax.plot(pos[0], pos[1], 's', color=color, markersize=6, zorder=4)
                if label:
                    ax.text(pos[0] + 0.1, pos[1] + 0.1, label,
                            fontsize=LABEL_FONTSIZE - 2, color=color)

    # 坐标轴和网格
    if equal_aspect:
        ax.set_aspect('equal')
    if show_grid:
        ax.grid(True, alpha=0.3, color=GRID_COLOR)
    if not show_axes:
        ax.set_axis_off()
    else:
        ax.axhline(y=0, color='#888888', linewidth=0.5, zorder=0)
        ax.axvline(x=0, color='#888888', linewidth=0.5, zorder=0)

    if x_range:
        ax.set_xlim(x_range)
    if y_range:
        ax.set_ylim(y_range)

    # 自动调整范围
    ax.autoscale_view(tight=True)
    margin = 0.5
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0] - margin, xlim[1] + margin)
    ax.set_ylim(ylim[0] - margin, ylim[1] + margin)

    finalize_figure(fig, output_path)
    return True, "几何图形渲染成功"
