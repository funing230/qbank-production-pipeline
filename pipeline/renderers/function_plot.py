"""
函数曲线/坐标系渲染函数
======================
使用 matplotlib + sympy 画函数图、概率分布、柱状图、散点图等。
这是最通用的渲染函数，覆盖大量科目。

data 格式 (函数图):
{
  "plot_type": "function | bar | scatter | distribution | pie | heatmap",
  "functions": [
    {"expr": "x**2 - 3*x + 2", "label": "f(x)", "color": 0, "style": "solid"},
    {"expr": "2*x - 1", "label": "g(x)", "color": 1, "style": "dashed"}
  ],
  "x_range": [-5, 5],
  "y_range": null,                # 可选，自动计算
  "markers": [                     # 可选：标记点
    {"x": 1, "y": 0, "label": "A(1,0)"},
  ],
  "annotations": [],               # 可选：标注文字
  "grid": true,
  "axes_labels": {"x": "x", "y": "y"}
}

data 格式 (柱状图):
{
  "plot_type": "bar",
  "categories": ["A", "B", "C", "D"],
  "values": [10, 25, 15, 30],
  "y_label": "频率",
  "colors": null  # 使用默认色板
}

data 格式 (分布图):
{
  "plot_type": "distribution",
  "dist_type": "normal | uniform | exponential | poisson",
  "params": {"mu": 0, "sigma": 1},
  "shade_region": {"from": -1, "to": 1},  # 可选：阴影区域
  "x_range": [-4, 4]
}
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.global_style import (
    finalize_figure,
    FIGSIZE, DPI, BACKGROUND, LABEL_FONTSIZE, AXIS_LABEL_FONTSIZE,
    TICK_FONTSIZE, LINE_WIDTH, THICK_LINE_WIDTH, PRIMARY_COLORS,
    EXTENDED_COLORS, GRID_COLOR, HIGHLIGHT_COLOR, LINE_COLOR,
    TITLE_FONTSIZE, apply_global_style
)


def render_function_plot(data: dict, diagram_type: str, output_path: str, style_override: dict = None) -> tuple:
    """渲染函数曲线/统计图"""
    plt = apply_global_style()

    plot_type = data.get("plot_type", "function")

    if plot_type == "function":
        return _render_function(data, plt, output_path)
    elif plot_type in ("line", "function_graph", "stratigraphic_section", 
                       "soil_depth_profile", "boundary_layer_plate_with_cf"):
        return _render_series(data, plt, output_path)
    elif plot_type == "bar":
        return _render_bar(data, plt, output_path)
    elif plot_type == "scatter":
        return _render_scatter(data, plt, output_path)
    elif plot_type == "distribution":
        return _render_distribution(data, plt, output_path)
    elif plot_type == "pie":
        return _render_pie(data, plt, output_path)
    elif plot_type == "heatmap":
        return _render_heatmap(data, plt, output_path)
    elif plot_type == "regression_plot":
        return _render_regression(data, plt, output_path)
    else:
        # 如果有series字段，走series渲染；否则走function
        if "series" in data or "segments" in data:
            return _render_series(data, plt, output_path)
        return _render_function(data, plt, output_path)


def _render_function(data, plt, output_path):
    """函数曲线图"""
    from sympy import symbols, sympify, lambdify
    from sympy.parsing.sympy_parser import parse_expr

    functions = data.get("functions", [])
    x_range = data.get("x_range", [-5, 5])
    y_range = data.get("y_range", None)
    markers = data.get("markers", [])
    show_grid = data.get("grid", True)
    axes_labels = data.get("axes_labels", {"x": "x", "y": "y"})

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    x = symbols('x')
    x_vals = np.linspace(x_range[0], x_range[1], 500)

    for i, func in enumerate(functions):
        expr_str = func.get("expr", "x")
        label = func.get("label", f"f_{i}(x)")
        color_idx = func.get("color", i)
        style = func.get("style", "solid")

        try:
            expr = parse_expr(expr_str)
            f_lambda = lambdify(x, expr, modules=['numpy'])
            
            color = EXTENDED_COLORS[color_idx % len(EXTENDED_COLORS)]
            
            # points_only 模式：只画离散点（整数x）
            if style == "points_only" or func.get("integer_x_only", False):
                x_int = np.arange(int(x_range[0]), int(x_range[1]) + 1)
                y_int = np.array([float(f_lambda(xi)) for xi in x_int])
                ax.scatter(x_int, y_int, color=color, s=60, zorder=5, label=label)
            elif style == "dashed":
                y_vals = f_lambda(x_vals)
                ax.plot(x_vals, y_vals, color=color, linewidth=THICK_LINE_WIDTH,
                        linestyle='dashed', label=label)
            else:
                y_vals = f_lambda(x_vals)
                ax.plot(x_vals, y_vals, color=color, linewidth=THICK_LINE_WIDTH,
                        linestyle=style if style in ('solid','dashed','dotted','dashdot') else 'solid',
                        label=label)
        except Exception as e:
            continue

    # 标记点
    for marker in markers:
        mx, my = marker.get("x", 0), marker.get("y", 0)
        mlabel = marker.get("label", "")
        ax.plot(mx, my, 'o', color=PRIMARY_COLORS[0], markersize=8, zorder=5)
        if mlabel:
            ax.annotate(mlabel, (mx, my), textcoords="offset points",
                        xytext=(10, 10), fontsize=LABEL_FONTSIZE - 1,
                        fontweight='bold')

    # 坐标轴
    ax.axhline(y=0, color='#888888', linewidth=0.8, zorder=1)
    ax.axvline(x=0, color='#888888', linewidth=0.8, zorder=1)

    if y_range:
        ax.set_ylim(y_range)
    ax.set_xlim(x_range)

    ax.set_xlabel(axes_labels.get("x", "x"), fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(axes_labels.get("y", "y"), fontsize=AXIS_LABEL_FONTSIZE)

    if show_grid:
        ax.grid(True, alpha=0.3, color=GRID_COLOR)

    if functions:
        ax.legend(fontsize=LABEL_FONTSIZE - 1, loc='best')

    finalize_figure(fig, output_path)
    return True, "函数图渲染成功"


def _render_series(data, plt, output_path):
    """通用折线/散点序列图 — 支持 series 数组、segments 箭头、layers 地层、annotations"""
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    
    series = data.get("series", [])
    segments = data.get("segments", [])
    layers = data.get("layers", [])
    markers = data.get("markers", [])
    annotations = data.get("annotations", [])
    x_range = data.get("x_range", None)
    y_range = data.get("y_range", None)
    x_scale = data.get("x_scale", "linear")
    y_scale = data.get("y_scale", "linear")
    title = data.get("title", "")
    x_label = data.get("x_label", "")
    y_label = data.get("y_label", "")
    show_grid = data.get("grid", True)
    equal_aspect = data.get("equal_aspect", False)
    
    # 设置坐标轴缩放
    if x_scale == "log":
        ax.set_xscale("log")
    if y_scale == "log":
        ax.set_yscale("log")
    
    # 渲染 series（折线/散点）
    # 兼容多种字段名: x/x_values, y/y_values, points[{x,y}]
    has_data = False
    # 如果顶级有 x_values（公共x轴），各series只需y_values
    shared_x = data.get("x_values", None)
    
    for i, s in enumerate(series):
        x_vals = s.get("x", [])
        y_vals = s.get("y", s.get("y_values", []))
        
        # 兼容 points: [{x:..., y:...}, ...] 格式
        if not x_vals and not y_vals and "points" in s:
            pts = s["points"]
            x_vals = [p.get("x", p[0] if isinstance(p, list) else 0) for p in pts]
            y_vals = [p.get("y", p[1] if isinstance(p, list) else 0) for p in pts]
        
        # 使用共享x轴
        if not x_vals and shared_x and y_vals:
            x_vals = shared_x
        
        if not x_vals or not y_vals:
            continue
        has_data = True
        label = s.get("label", f"Series {i+1}")
        color = s.get("color", EXTENDED_COLORS[i % len(EXTENDED_COLORS)])
        if isinstance(color, int):
            color = EXTENDED_COLORS[color % len(EXTENDED_COLORS)]
        linestyle = s.get("linestyle", "solid")
        marker = s.get("marker", None)
        linewidth = s.get("linewidth", THICK_LINE_WIDTH)
        
        if s.get("line", True):
            ax.plot(x_vals, y_vals, color=color, linewidth=linewidth,
                    linestyle=linestyle, marker=marker, label=label,
                    markersize=6, zorder=3)
        else:
            ax.scatter(x_vals, y_vals, color=color, s=50, label=label, zorder=3)
    
    # 渲染 segments（箭头/线段）
    for i, seg in enumerate(segments):
        fr = seg.get("from", [0, 0])
        to = seg.get("to", [1, 1])
        label = seg.get("label", "")
        color = seg.get("color", EXTENDED_COLORS[i % len(EXTENDED_COLORS)])
        if isinstance(color, int):
            color = EXTENDED_COLORS[color % len(EXTENDED_COLORS)]
        has_data = True
        
        if seg.get("arrow", False):
            ax.annotate("", xy=to, xytext=fr,
                        arrowprops=dict(arrowstyle="->", color=color, lw=THICK_LINE_WIDTH))
        else:
            ax.plot([fr[0], to[0]], [fr[1], to[1]], color=color, linewidth=THICK_LINE_WIDTH)
        
        if label:
            mid_x = (fr[0] + to[0]) / 2
            mid_y = (fr[1] + to[1]) / 2
            ax.annotate(label, (mid_x, mid_y), fontsize=LABEL_FONTSIZE - 1,
                        ha='center', va='bottom')
    
    # 渲染 layers（地层/区块填充）
    for layer in layers:
        top = layer.get("top", 0)
        bottom = layer.get("bottom", -1)
        x_start = layer.get("x_start", 0)
        x_end = layer.get("x_end", 10)
        color = layer.get("color", "#cccccc")
        hatch = layer.get("hatch", "")
        name = layer.get("name", "")
        has_data = True
        
        rect = plt.Rectangle((x_start, bottom), x_end - x_start, top - bottom,
                              facecolor=color, edgecolor='black', linewidth=1,
                              hatch=hatch if hatch != "none" else "")
        ax.add_patch(rect)
        
        lp = layer.get("label_position")
        if lp:
            ax.text(lp[0], lp[1], name, fontsize=LABEL_FONTSIZE - 1,
                    ha='center', va='center', fontweight='bold')
    
    # 标记点
    for marker in markers:
        mx, my = marker.get("x", 0), marker.get("y", 0)
        mlabel = marker.get("label", "")
        ax.plot(mx, my, 'o', color=PRIMARY_COLORS[0], markersize=8, zorder=5)
        if mlabel:
            ax.annotate(mlabel, (mx, my), textcoords="offset points",
                        xytext=(10, 10), fontsize=LABEL_FONTSIZE - 2)
    
    # 标注 annotations
    for ann in annotations:
        text = ann.get("text", ann.get("label", ""))
        pos = ann.get("position", None)
        ax_val = ann.get("x"), ann.get("y")
        if pos:
            ax.text(pos[0], pos[1], text, fontsize=LABEL_FONTSIZE - 1,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
        elif ax_val[0] is not None and ax_val[1] is not None:
            ax.text(ax_val[0], ax_val[1], text, fontsize=LABEL_FONTSIZE - 1,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
    
    # 坐标轴设置
    if x_range:
        ax.set_xlim(x_range)
    if y_range:
        ax.set_ylim(y_range)
    if equal_aspect:
        ax.set_aspect('equal')
    if x_label:
        ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    if y_label:
        ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    if title:
        ax.set_title(title, fontsize=AXIS_LABEL_FONTSIZE + 2, fontweight='bold')
    if show_grid:
        ax.grid(True, alpha=0.3, color=GRID_COLOR)
    if series and any(s.get("label") for s in series):
        ax.legend(fontsize=LABEL_FONTSIZE - 1, loc='best')
    
    finalize_figure(fig, output_path)
    return True, "序列图渲染成功"


def _render_bar(data, plt, output_path):
    """柱状图"""
    categories = data.get("categories", [])
    values = data.get("values", [])
    y_label = data.get("y_label", "")
    x_label = data.get("x_label", "")

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    colors = [EXTENDED_COLORS[i % len(EXTENDED_COLORS)] for i in range(len(categories))]

    bars = ax.bar(categories, values, color=colors, edgecolor='white', linewidth=0.8)

    # 在柱子上方标数值
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                str(val), ha='center', va='bottom', fontsize=LABEL_FONTSIZE, fontweight='bold')

    if y_label:
        ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    if x_label:
        ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)

    ax.grid(True, axis='y', alpha=0.3, color=GRID_COLOR)
    finalize_figure(fig, output_path)
    return True, "柱状图渲染成功"


def _render_scatter(data, plt, output_path):
    """散点图 — 支持 points 格式和 series 格式"""
    x_label = data.get("x_label", "x")
    y_label = data.get("y_label", "y")
    title = data.get("title", "")

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    # 方式1: series 格式（多组散点，每组有 label/color/marker）
    series = data.get("series", [])
    x_scale = data.get("x_scale", "linear")
    y_scale = data.get("y_scale", "linear")
    if x_scale == "log":
        ax.set_xscale("log")
    if y_scale == "log":
        ax.set_yscale("log")
    
    if series:
        markers_list = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']
        for i, s in enumerate(series):
            xs = s.get("x", [])
            ys = s.get("y", [])
            # 兼容 points: [{x,y},...] 或 [[x,y],...]
            if not xs and not ys and "points" in s:
                pts = s["points"]
                if pts and isinstance(pts[0], dict):
                    xs = [p.get("x", 0) for p in pts]
                    ys = [p.get("y", 0) for p in pts]
                elif pts and isinstance(pts[0], (list, tuple)):
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
            label = s.get("label", f"系列{i+1}")
            color = s.get("color", i)
            if isinstance(color, int):
                color = PRIMARY_COLORS[color % len(PRIMARY_COLORS)]
            marker = s.get("marker", markers_list[i % len(markers_list)])
            
            ax.scatter(xs, ys, color=color, s=100, marker=marker,
                       edgecolors=LINE_COLOR, linewidths=1.0, zorder=5, label=label)
            # 如果指定了 line=true，在散点之上叠加连线
            if s.get("line", False):
                ax.plot(xs, ys, color=color, linewidth=THICK_LINE_WIDTH - 0.5, zorder=4)
        
        # fit_lines 拟合线
        fit_lines = data.get("fit_lines", [])
        for fl in fit_lines:
            fx = fl.get("x", [])
            fy = fl.get("y", [])
            if fx and fy:
                ax.plot(fx, fy, color=fl.get("color", "gray"), linewidth=1.5,
                        linestyle='--', label=fl.get("label", ""), zorder=3)
        
        # trend_line
        trend = data.get("trend_line")
        if trend and isinstance(trend, dict):
            tx = trend.get("x", [])
            ty = trend.get("y", [])
            if tx and ty:
                ax.plot(tx, ty, color=trend.get("color", "gray"), linewidth=1.5,
                        linestyle='--', label=trend.get("label", "趋势线"), zorder=3)
        
        # invert axes
        if data.get("invert_x_axis", False):
            ax.invert_xaxis()
        if data.get("invert_y_axis", False):
            ax.invert_yaxis()
        
        ax.legend(fontsize=LABEL_FONTSIZE)
    else:
        # 方式2: points 格式（简单 [[x,y], ...] 列表）
        points = data.get("points", [])
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.scatter(xs, ys, color=PRIMARY_COLORS[0], s=80, edgecolors=LINE_COLOR,
                       linewidths=1.0, zorder=5)

    # 标注 annotations（兼容多种格式）
    annotations = data.get("annotations", [])
    if not isinstance(annotations, list):
        annotations = []
    text_objects = []
    for ann in annotations:
        text = ann.get("text", ann.get("label", ""))
        # 优先 xy 字段（列表格式），其次 x/y 分字段
        xy = ann.get("xy")
        if xy is None:
            xy = (ann.get("x", 0), ann.get("y", 0))
        else:
            xy = tuple(xy)
        xytext = ann.get("xytext")
        has_arrow = ann.get("arrow", False)
        style = ann.get("style", "annotate")

        if style == "text" or (not has_arrow and xytext is None):
            # 纯文字标注
            t = ax.text(xy[0], xy[1], text, fontsize=LABEL_FONTSIZE - 1,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                                  edgecolor='none', alpha=0.7))
            text_objects.append(t)
        else:
            # 带箭头的标注
            xytext_val = tuple(xytext) if xytext else (xy[0] + 0.05, xy[1] + 0.05)
            t = ax.annotate(text, xy=xy, xytext=xytext_val,
                            fontsize=LABEL_FONTSIZE - 1, fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
            text_objects.append(t)

    # markers（带标签的标记点）
    markers = data.get("markers", [])
    for mk in markers:
        mx = mk.get("x", 0)
        my = mk.get("y", 0)
        mlabel = mk.get("label", "")
        ax.plot(mx, my, 'o', color=PRIMARY_COLORS[1], markersize=10, zorder=6)
        if mlabel:
            t = ax.annotate(mlabel, (mx, my),
                            textcoords="offset points", xytext=(8, 8),
                            fontsize=LABEL_FONTSIZE, fontweight='bold')
            text_objects.append(t)

    # guide_lines（辅助线）
    guide_lines = data.get("guide_lines", [])
    for gl in guide_lines:
        gf = gl.get("from", gl.get("start", [0, 0]))
        gt = gl.get("to", gl.get("end", [1, 1]))
        gstyle = gl.get("style", "dashed")
        glabel = gl.get("label", "")
        ax.plot([gf[0], gt[0]], [gf[1], gt[1]],
                color='#888888', linestyle=gstyle, linewidth=1.2, zorder=2)
        if glabel:
            gmx = (gf[0] + gt[0]) / 2
            gmy = (gf[1] + gt[1]) / 2
            t = ax.text(gmx + 0.02, gmy, glabel, fontsize=LABEL_FONTSIZE - 2,
                        ha='left', va='center', color='#555555')
            text_objects.append(t)

    # 标记带标签的点（旧格式兼容）
    labeled_points = data.get("labeled_points", [])
    for lp in labeled_points:
        ax.plot(lp["x"], lp["y"], 'o', color=PRIMARY_COLORS[1], markersize=10, zorder=6)
        t = ax.annotate(lp.get("label", ""), (lp["x"], lp["y"]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=LABEL_FONTSIZE, fontweight='bold')
        text_objects.append(t)

    # 轴范围
    x_range = data.get("x_range")
    y_range = data.get("y_range")
    if x_range:
        ax.set_xlim(x_range)
    if y_range:
        ax.set_ylim(y_range)
    else:
        # 自动 padding：如果 y 范围太窄（例如全为0），加 padding
        y_lo, y_hi = ax.get_ylim()
        if abs(y_hi - y_lo) < 0.5:
            mid = (y_hi + y_lo) / 2
            ax.set_ylim(mid - 1, mid + 1)

    if data.get("equal_aspect", False):
        ax.set_aspect('equal')
    if title:
        ax.set_title(title, fontsize=AXIS_LABEL_FONTSIZE + 2, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(data.get("grid", True), alpha=0.3, color=GRID_COLOR)

    finalize_figure(fig, output_path)
    return True, "散点图渲染成功"


def _render_regression(data, plt, output_path):
    """回归图 — 散点 + 多条候选回归曲线"""
    import re

    title = data.get("title", "")
    x_label = data.get("x_label", "x")
    y_label = data.get("y_label", "y")
    points = data.get("points", [])
    curves = data.get("curves", [])
    point_label = data.get("point_label", "数据点")
    x_range = data.get("x_range")

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    # 画散点
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, color=PRIMARY_COLORS[0], s=80, edgecolors=LINE_COLOR,
                   linewidths=1.0, zorder=5, label=point_label)

        # 确定 x 范围
        if not x_range:
            x_range = [min(xs) * 0.8, max(xs) * 1.2]

    if not x_range:
        x_range = [0, 10]

    # 画候选回归曲线
    x_dense = np.linspace(x_range[0], x_range[1], 300)
    for i, curve in enumerate(curves):
        expr = curve.get("expr", "")
        label = curve.get("label", f"曲线{i+1}")
        color = PRIMARY_COLORS[(i + 1) % len(PRIMARY_COLORS)]

        # 安全地计算表达式
        try:
            # 替换常见数学函数
            safe_expr = expr.replace("^", "**")
            safe_expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', safe_expr)  # 2x -> 2*x
            safe_expr = safe_expr.replace("exp(", "np.exp(")
            safe_expr = safe_expr.replace("log(", "np.log(")
            safe_expr = safe_expr.replace("sin(", "np.sin(")
            safe_expr = safe_expr.replace("cos(", "np.cos(")
            safe_expr = safe_expr.replace("sqrt(", "np.sqrt(")

            x = x_dense  # noqa: F841
            y_curve = eval(safe_expr)  # noqa: S307
            ax.plot(x_dense, y_curve, color=color, linewidth=LINE_WIDTH,
                    label=label, zorder=3)
        except Exception:
            # 表达式解析失败，跳过这条曲线
            continue

    ax.legend(fontsize=LABEL_FONTSIZE - 1, loc='best')
    if title:
        ax.set_title(title, fontsize=AXIS_LABEL_FONTSIZE + 2, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(data.get("grid", True), alpha=0.3, color=GRID_COLOR)

    finalize_figure(fig, output_path)
    return True, "回归图渲染成功"


def _render_distribution(data, plt, output_path):
    """概率分布图"""
    from scipy import stats

    dist_type = data.get("dist_type", "normal")
    params = data.get("params", {"mu": 0, "sigma": 1})
    x_range = data.get("x_range", [-4, 4])
    shade_region = data.get("shade_region", None)

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    x_vals = np.linspace(x_range[0], x_range[1], 500)

    if dist_type == "normal":
        mu, sigma = params.get("mu", 0), params.get("sigma", 1)
        y_vals = stats.norm.pdf(x_vals, mu, sigma)
        label = f"N({mu}, {sigma}²)"
    elif dist_type == "uniform":
        a, b = params.get("a", 0), params.get("b", 1)
        y_vals = stats.uniform.pdf(x_vals, a, b - a)
        label = f"U({a}, {b})"
    elif dist_type == "exponential":
        lam = params.get("lambda", 1)
        y_vals = stats.expon.pdf(x_vals, scale=1 / lam)
        label = f"Exp({lam})"
    else:
        y_vals = stats.norm.pdf(x_vals, 0, 1)
        label = "N(0, 1)"

    ax.plot(x_vals, y_vals, color=PRIMARY_COLORS[0], linewidth=THICK_LINE_WIDTH, label=label)

    # 阴影区域
    if shade_region:
        x_shade = np.linspace(shade_region["from"], shade_region["to"], 200)
        if dist_type == "normal":
            y_shade = stats.norm.pdf(x_shade, params.get("mu", 0), params.get("sigma", 1))
        else:
            y_shade = stats.norm.pdf(x_shade, 0, 1)
        ax.fill_between(x_shade, y_shade, alpha=0.3, color=HIGHLIGHT_COLOR)

    ax.axhline(y=0, color='#888888', linewidth=0.5)
    ax.set_xlabel("x", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("f(x)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.legend(fontsize=LABEL_FONTSIZE, loc='best')
    ax.grid(True, alpha=0.3, color=GRID_COLOR)

    finalize_figure(fig, output_path)
    return True, "分布图渲染成功"


def _render_pie(data, plt, output_path):
    """饼图"""
    labels = data.get("labels", [])
    values = data.get("values", [])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    colors = EXTENDED_COLORS[:len(labels)]

    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': LABEL_FONTSIZE}
    )
    for at in autotexts:
        at.set_fontweight('bold')

    ax.axis('equal')
    finalize_figure(fig, output_path, pad=1.0)
    return True, "饼图渲染成功"


def _render_heatmap(data, plt, output_path):
    """热力图/矩阵可视化"""
    # 兼容多种字段名: matrix/values/z_values, row_labels/y_categories/y_labels, col_labels/x_categories/x_values
    matrix = data.get("matrix") or data.get("values") or data.get("z_values", [[]])
    row_labels = data.get("row_labels") or data.get("y_categories") or data.get("y_labels", [])
    col_labels = data.get("col_labels") or data.get("x_categories") or data.get("x_values", [])
    colormap = data.get("colormap") or data.get("color_map", "YlOrRd")
    annotate = data.get("annotate", data.get("show_values", True))
    colorbar_label = data.get("colorbar_label") or data.get("z_label", "")

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    arr = np.array(matrix, dtype=float)

    im = ax.imshow(arr, cmap=colormap, aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=AXIS_LABEL_FONTSIZE)

    if row_labels:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=TICK_FONTSIZE)
    if col_labels:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=TICK_FONTSIZE)

    # 在格子中写数值
    if annotate and arr.size > 0 and arr.size <= 100:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                fmt = f"{val:.2f}" if abs(val) < 1 else f"{val:.1f}"
                ax.text(j, i, fmt, ha='center', va='center',
                        fontsize=LABEL_FONTSIZE, fontweight='bold',
                        color='white' if val > arr.max() * 0.6 else 'black')

    x_label = data.get("x_label", "")
    y_label = data.get("y_label", "")
    if x_label:
        ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    if y_label:
        ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)

    finalize_figure(fig, output_path)
    return True, "热力图渲染成功"
