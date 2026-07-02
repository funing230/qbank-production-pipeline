"""
电路图渲染函数
=============
使用 schemdraw 画电路图、逻辑门。
schemdraw 的组件有固定间距，天然不会遮挡。

data 格式:
{
  "components": [
    {"type": "resistor", "label": "R1=10kΩ", "direction": "right"},
    {"type": "capacitor", "label": "C1=100μF", "direction": "down"},
    {"type": "voltage_source", "label": "V=5V", "direction": "up"},
    {"type": "ground"},
    {"type": "wire", "direction": "left"},
    {"type": "dot"},  # 连接点
  ],
  "title": ""  # 可选
}

支持的 component types:
  resistor, capacitor, inductor, voltage_source, current_source,
  diode, led, ground, wire, dot, switch, opamp, transistor_npn, transistor_pnp
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.global_style import (
    FIGSIZE, DPI, BACKGROUND, CIRCUIT_SCALE, CIRCUIT_FONTSIZE, apply_global_style
)


# schemdraw 组件映射
COMPONENT_MAP = {
    'resistor': 'Resistor',
    'capacitor': 'Capacitor',
    'inductor': 'Inductor',
    'voltage_source': 'SourceV',
    'current_source': 'SourceI',
    'diode': 'Diode',
    'led': 'LED',
    'ground': 'Ground',
    'wire': 'Line',
    'dot': 'Dot',
    'switch': 'Switch',
    'opamp': 'Opamp',
    'transistor_npn': 'BjtNpn',
    'transistor_pnp': 'BjtPnp',
}

DIRECTION_MAP = {
    'right': 'right',
    'left': 'left',
    'up': 'up',
    'down': 'down',
}

# 电路组件颜色调色板（较深、清晰）
_CIRCUIT_COLORS = [
    '#2980B9',  # 蓝
    '#C0392B',  # 红
    '#27AE60',  # 绿
    '#8E44AD',  # 紫
    '#D35400',  # 橙
    '#16A085',  # 青
    '#2C3E50',  # 深灰蓝
    '#E67E22',  # 黄橙
]


def render_circuit(data: dict, diagram_type: str, output_path: str, style_override: dict = None) -> tuple:
    """渲染电路图"""
    try:
        import schemdraw
        import schemdraw.elements as elm
    except ImportError:
        return False, "schemdraw 未安装"

    components = data.get("components", [])
    if not components:
        return False, "components 列表为空"

    with schemdraw.Drawing(show=False) as d:
        d.config(fontsize=CIRCUIT_FONTSIZE, unit=CIRCUIT_SCALE * 3)

        for i, comp in enumerate(components):
            comp_type = comp.get("type", "wire")
            label = comp.get("label", "")
            direction = comp.get("direction", "right")

            # 获取 schemdraw 元素类
            elem_name = COMPONENT_MAP.get(comp_type, 'Line')
            elem_cls = getattr(elm, elem_name, elm.Line)

            # 创建元素
            element = elem_cls()

            # 设置方向
            dir_method = DIRECTION_MAP.get(direction, 'right')
            element = getattr(element, dir_method)()

            # 按序号分配颜色（wire/dot/ground 保持黑色不着色）
            if comp_type not in ('wire', 'dot', 'ground'):
                element = element.color(_CIRCUIT_COLORS[i % len(_CIRCUIT_COLORS)])

            # 设置标签
            if label:
                element = element.label(label)

            d.add(element)

        # 保存
        d.save(output_path, dpi=DPI)

    if os.path.exists(output_path):
        return True, "电路图渲染成功"
    else:
        return False, "schemdraw 保存失败"
