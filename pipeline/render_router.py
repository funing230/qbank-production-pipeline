"""
渲染路由器 (Render Router)
==========================
接收 GPT 的 render_instruction JSON，根据 engine 字段分发给对应的渲染函数。

GPT 输出格式:
{
  "render_instruction": {
    "engine": "matplotlib-venn | networkx | graphviz | schemdraw | rdkit | matplotlib | pillow | cartopy",
    "diagram_type": "具体图形子类型",
    "data": { ... },
    "style_override": { ... }  // 可选，覆盖全局风格
  }
}
"""

import json
import os
import traceback
from pathlib import Path

# 渲染函数导入
from pipeline.renderers.venn import render_venn
from pipeline.renderers.graph import render_graph
from pipeline.renderers.flowchart import render_flowchart
from pipeline.renderers.circuit import render_circuit
from pipeline.renderers.molecule import render_molecule
from pipeline.renderers.function_plot import render_function_plot
from pipeline.renderers.geometry import render_geometry
from pipeline.renderers.table_matrix import render_table_matrix
from pipeline.renderers.map_geo import render_map
from pipeline.renderers.fallback import render_fallback

# engine → 渲染函数映射
ENGINE_MAP = {
    "matplotlib-venn": render_venn,
    "venn": render_venn,
    "networkx": render_graph,
    "graph": render_graph,
    "graphviz": render_flowchart,
    "flowchart": render_flowchart,
    "schemdraw": render_circuit,
    "circuit": render_circuit,
    "rdkit": render_molecule,
    "molecule": render_molecule,
    "matplotlib": render_function_plot,
    "function_plot": render_function_plot,
    "sympy": render_function_plot,
    "geometry": render_geometry,
    "shapely": render_geometry,
    "pillow": render_table_matrix,
    "table": render_table_matrix,
    "matrix": render_table_matrix,
    "cartopy": render_map,
    "map": render_map,
}


def render_from_instruction(render_instruction: dict, output_path: str) -> tuple[bool, str]:
    """
    主入口：根据 render_instruction 路由到对应的渲染函数。
    
    Args:
        render_instruction: GPT 输出的画图指令字典
        output_path: 图片保存路径
    
    Returns:
        (success: bool, message: str)
    """
    engine = render_instruction.get("engine", "").lower().strip()
    diagram_type = render_instruction.get("diagram_type", "")
    data = render_instruction.get("data", {})
    style_override = render_instruction.get("style_override", {})

    if not engine:
        return False, "render_instruction 缺少 engine 字段"

    if not data:
        return False, "render_instruction 缺少 data 字段"

    # 白名单校验：diagram_type 必须在已实现列表中，否则直接拒绝
    VALID_DIAGRAM_TYPES = {
        "matplotlib-venn": {"venn_2sets", "venn_3sets"},
        "venn": {"venn_2sets", "venn_3sets"},
        "networkx": {"directed", "undirected", "tree", "hasse"},
        "graph": {"directed", "undirected", "tree", "hasse"},
        "graphviz": {"flowchart", "state_machine", "dfa"},
        "flowchart": {"flowchart", "state_machine", "dfa"},
        "schemdraw": {"circuit", "logic_gate"},
        "circuit": {"circuit", "logic_gate"},
        "rdkit": {"molecule"},
        "molecule": {"molecule"},
        "matplotlib": {"function", "bar", "scatter", "distribution", "pie", "heatmap"},
        "function_plot": {"function", "bar", "scatter", "distribution", "pie", "heatmap"},
        "geometry": {"polygon", "circle", "coordinate"},
        "shapely": {"polygon", "circle", "coordinate"},
        "table": {"table", "truth_table", "matrix"},
        "pillow": {"table", "truth_table", "matrix"},
        "matrix": {"table", "truth_table", "matrix"},
        "cartopy": {"world_map", "regional_map"},
        "map": {"world_map", "regional_map"},
        "fallback": None,  # fallback 不限制
    }
    allowed = VALID_DIAGRAM_TYPES.get(engine)
    if allowed is not None and diagram_type not in allowed:
        # 对 matplotlib engine，也检查 data.plot_type
        plot_type = data.get("plot_type", diagram_type)
        valid_plot_types = {"function", "bar", "scatter", "distribution", "pie", "heatmap", "line", "grouped_bar"}
        if engine in ("matplotlib", "function_plot") and plot_type in valid_plot_types:
            pass  # plot_type 合法，允许通过
        else:
            return False, f"非法diagram_type '{diagram_type}'(engine={engine})，不在白名单中。请用fallback engine+完整代码"

    # 共性方案：LaTeX → Unicode，确保所有引擎图片上显示正规公式符号
    from pipeline.global_style import latex_sanitize_data
    data = latex_sanitize_data(data)

    # 查找对应的渲染函数
    render_fn = ENGINE_MAP.get(engine)

    if render_fn is None:
        # 未知 engine，走保底
        render_fn = render_fallback

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 调用渲染函数
    try:
        success, msg = render_fn(
            data=data,
            diagram_type=diagram_type,
            output_path=output_path,
            style_override=style_override,
        )
        return success, msg
    except Exception as e:
        tb = traceback.format_exc()
        return False, f"渲染异常 [{engine}]: {str(e)}\n{tb[-500:]}"


def parse_render_instruction(question_json: dict) -> dict | None:
    """
    从 GPT 生成的 question_json 中提取 render_instruction。
    
    支持两种格式:
    1. question_json["render_instruction"] = {...}  (新格式)
    2. question_json["render_code"] = "..." (旧格式，走 fallback)
    
    Returns:
        render_instruction dict, 或 None（如果没有图片需求）
    """
    # 新格式：直接有 render_instruction
    if "render_instruction" in question_json:
        ri = question_json["render_instruction"]
        if isinstance(ri, str):
            try:
                ri = json.loads(ri)
            except json.JSONDecodeError:
                return None
        return ri

    # 旧格式：有 render_code（兼容）
    if "render_code" in question_json and question_json["render_code"]:
        return {
            "engine": "fallback",
            "diagram_type": "legacy_code",
            "data": {"code": question_json["render_code"]},
        }

    return None
