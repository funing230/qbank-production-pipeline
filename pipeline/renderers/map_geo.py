"""
地图渲染函数
============
使用 cartopy 画地图（地理/气候/海洋/环境科学）。

data 格式:
{
  "map_type": "world | regional | china",
  "projection": "PlateCarree | Mollweide | Orthographic",
  "extent": [lon_min, lon_max, lat_min, lat_max],  # 可选：区域范围
  "features": ["coastlines", "borders", "rivers", "lakes"],
  "points": [
    {"lon": 116.4, "lat": 39.9, "label": "北京", "color": 0},
  ],
  "regions_highlight": [],   # 可选：高亮区域
  "contour_data": null,      # 可选：等值线数据
  "title": ""
}
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.global_style import (
    finalize_figure,
    MAP_FIGSIZE, DPI, BACKGROUND, LABEL_FONTSIZE, LINE_WIDTH,
    PRIMARY_COLORS, EXTENDED_COLORS, apply_global_style
)


def render_map(data: dict, diagram_type: str, output_path: str, style_override: dict = None) -> tuple:
    """渲染地图"""
    plt = apply_global_style()

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        return False, "cartopy 未安装"

    map_type = data.get("map_type", "world")
    projection_name = data.get("projection", "PlateCarree")
    extent = data.get("extent", None)
    features = data.get("features", ["coastlines", "borders"])
    points = data.get("points", data.get("markers", []))

    # 选择投影
    projection_map = {
        "PlateCarree": ccrs.PlateCarree(),
        "Mollweide": ccrs.Mollweide(),
        "Orthographic": ccrs.Orthographic(central_longitude=105, central_latitude=35),
        "Robinson": ccrs.Robinson(),
    }
    projection = projection_map.get(projection_name, ccrs.PlateCarree())

    fig, ax = plt.subplots(1, 1, figsize=MAP_FIGSIZE,
                           subplot_kw={'projection': projection})

    # 设置范围
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    elif map_type == "china":
        ax.set_extent([73, 135, 18, 54], crs=ccrs.PlateCarree())

    # 添加地图要素
    if "coastlines" in features:
        ax.add_feature(cfeature.COASTLINE, linewidth=LINE_WIDTH * 0.6)
    if "borders" in features:
        ax.add_feature(cfeature.BORDERS, linewidth=LINE_WIDTH * 0.4, linestyle='--')
    if "rivers" in features:
        ax.add_feature(cfeature.RIVERS, linewidth=LINE_WIDTH * 0.3, color='#4A90D9')
    if "lakes" in features:
        ax.add_feature(cfeature.LAKES, alpha=0.5, color='#AED6F1')
    if "land" in features:
        ax.add_feature(cfeature.LAND, facecolor='#F5F5DC')
    if "ocean" in features:
        ax.add_feature(cfeature.OCEAN, facecolor='#E8F4FD')

    # 添加网格线（某些投影+shapely版本组合可能失败，非致命）
    try:
        gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5)
    except Exception:
        pass  # 网格线非核心，失败不影响图片

    # 标记点
    for i, pt in enumerate(points):
        lon = pt.get("lon", 0)
        lat = pt.get("lat", 0)
        label = pt.get("label", "")
        color_idx = pt.get("color", i)
        color = EXTENDED_COLORS[color_idx % len(EXTENDED_COLORS)]

        ax.plot(lon, lat, 'o', color=color, markersize=10,
                transform=ccrs.PlateCarree(), zorder=5)
        if label:
            ax.text(lon + 1, lat + 1, label, fontsize=LABEL_FONTSIZE - 2,
                    transform=ccrs.PlateCarree(), fontweight='bold',
                    color=color)

    finalize_figure(fig, output_path, pad=1.5)
    return True, "地图渲染成功"
