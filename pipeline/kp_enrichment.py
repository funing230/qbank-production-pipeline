"""
Knowledge-point enrichment for image_prompt generation.

The population CSV guarantees basic Chinese metadata for all 1688 KPs, while
input_snapshot only covers a subset with detailed visual constraints. This module
adds deterministic fallback fields for every KP and augments existing fields with
image_prompt-specific precision requirements. It also builds a quota-aware
question blueprint so each KP's real production_quota controls the distribution
of difficulty, archetype, and visual type.
"""
from copy import deepcopy

SUBJECT_DEFAULTS = {
    "S01": {"image_types": ["mathematical_diagram", "graph_diagram", "table"], "competencies": ["calculation", "reasoning", "diagram_interpretation"]},
    "S02": {"image_types": ["chart", "scatter_plot", "distribution_plot", "table"], "competencies": ["statistical_reasoning", "calculation", "data_interpretation"]},
    "S03": {"image_types": ["topology_diagram", "network_diagram", "annotated_structure"], "competencies": ["spatial_reasoning", "classification", "diagram_interpretation"]},
    "S04": {"image_types": ["function_plot", "algebraic_diagram", "table"], "competencies": ["symbolic_reasoning", "calculation", "pattern_recognition"]},
    "S05": {"image_types": ["geometry_diagram", "coordinate_plot", "annotated_figure"], "competencies": ["spatial_reasoning", "calculation", "proof_reasoning"]},
    "S06": {"image_types": ["logic_diagram", "truth_table", "number_line", "graph_diagram"], "competencies": ["logical_reasoning", "calculation", "classification"]},
    "S13": {"image_types": ["mechanical_schematic", "force_diagram", "annotated_structure"], "competencies": ["engineering_reasoning", "calculation", "diagram_interpretation"]},
    "S14": {"image_types": ["circuit_diagram", "signal_plot", "block_diagram"], "competencies": ["circuit_analysis", "signal_reasoning", "calculation"]},
    "S15": {"image_types": ["flowchart", "data_structure_diagram", "network_diagram", "state_machine"], "competencies": ["algorithmic_reasoning", "logic", "structure_interpretation"]},
    "S16": {"image_types": ["structural_diagram", "section_view", "load_diagram", "site_plan"], "competencies": ["engineering_reasoning", "spatial_reasoning", "calculation"]},
    "S17": {"image_types": ["microstructure_diagram", "phase_diagram", "stress_strain_chart"], "competencies": ["materials_reasoning", "data_interpretation", "structure_property_mapping"]},
    "S18": {"image_types": ["thermodynamic_cycle_diagram", "system_schematic", "performance_chart"], "competencies": ["energy_balance", "cycle_analysis", "calculation"]},
    "S19": {"image_types": ["biological_process_diagram", "system_schematic", "data_chart"], "competencies": ["process_reasoning", "data_interpretation", "modeling"]},
    "S20": {"image_types": ["engineering_schematic", "process_flow_diagram", "data_chart"], "competencies": ["engineering_reasoning", "calculation", "system_analysis"]},
    "S21": {"image_types": ["biological_structure_diagram", "pathway_diagram", "phylogenetic_tree", "data_chart"], "competencies": ["structure_function_reasoning", "process_reasoning", "classification"]},
    "S22": {"image_types": ["chemical_structure", "reaction_scheme", "energy_diagram", "molecular_diagram"], "competencies": ["chemical_reasoning", "calculation", "structure_interpretation"]},
    "S23": {"image_types": ["physics_diagram", "vector_diagram", "wave_plot", "field_diagram"], "competencies": ["physical_reasoning", "calculation", "modeling"]},
    "S24": {"image_types": ["earth_system_diagram", "map_diagram", "cross_section", "data_chart"], "competencies": ["systems_reasoning", "spatial_interpretation", "data_interpretation"]},
}

IMAGE_PROMPT_PRECISION_RULE = (
    "For image_prompt generation, explicitly describe the diagram type, global layout, "
    "exact element counts, spatial positions, connections/arrows/relationships, and every visible label. "
    "Use concrete numbers and avoid vague words such as several/some/many. "
    "If the image contains mathematical or scientific formulas, describe them as rendered formulas, not raw LaTeX source; "
    "the final image must show standard mathematical notation, never visible dollar signs or LaTeX commands."
)

ENGLISH_SLOT_PERIOD = 6


def _as_list(value):
    if not value:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _dedupe(seq):
    out = []
    seen = set()
    for item in seq:
        marker = str(item)
        if marker not in seen:
            seen.add(marker)
            out.append(item)
    return out


def _difficulty_pattern(importance: float) -> list:
    if importance >= 4.5:
        return [3, 4, 4, 5, 3, 4, 5, 4]
    if importance >= 3.5:
        return [2, 3, 3, 4, 3, 4, 2, 3]
    if importance >= 2.5:
        return [2, 3, 3, 2, 4, 3, 2, 3]
    return [1, 2, 2, 3, 2, 3, 1, 2]


def _complexity_for_difficulty(difficulty: int) -> str:
    if difficulty <= 2:
        return "single-step reading/calculation; 3-5 main visual elements"
    if difficulty == 3:
        return "two-step reasoning; 4-6 main visual elements"
    if difficulty == 4:
        return "multi-step reasoning with one distractor relation; 5-7 main visual elements"
    return "advanced synthesis; 6-8 main visual elements, but still visually uncluttered"


def build_question_blueprint(kp_info: dict) -> list:
    """Build per-question blueprint from the KP's real production_quota."""
    quota = int(kp_info.get("production_quota") or kp_info.get("target_quota") or 1)
    quota = max(1, quota)
    archetypes = _as_list(kp_info.get("question_archetypes")) or ["基于图示信息判断关键概念"]
    image_types = _as_list(kp_info.get("allowed_image_types")) or ["academic_diagram"]
    competencies = _as_list(kp_info.get("competency_types")) or ["diagram_interpretation"]
    importance = float(kp_info.get("importance") or 3)
    diff_pattern = _difficulty_pattern(importance)
    blueprint = []
    for idx in range(quota):
        difficulty = diff_pattern[idx % len(diff_pattern)]
        archetype = archetypes[idx % len(archetypes)]
        image_type = image_types[(idx + idx // max(1, len(archetypes))) % len(image_types)]
        competency = competencies[idx % len(competencies)]
        question_language = "en" if (idx + 1) % ENGLISH_SLOT_PERIOD == 0 else "zh"
        blueprint.append({
            "slot": idx + 1,
            "question_language": question_language,
            "difficulty": difficulty,
            "archetype": archetype,
            "image_type": image_type,
            "competency": competency,
            "visual_complexity": _complexity_for_difficulty(difficulty),
            "design_goal": f"第{idx + 1}题必须使用{'英文' if question_language == 'en' else '中文'}题干/选项/解析，围绕{archetype}，使用{image_type}，考查{competency}，难度为{difficulty}/5。",
        })
    return blueprint


def enrich_kp_for_image_prompt(kp_info: dict) -> dict:
    """Return a copy of kp_info with deterministic fallback + augmented visual fields."""
    info = deepcopy(kp_info)
    subject_id = info.get("subject_id") or str(info.get("kp_id", "")).split("-M")[0]
    defaults = SUBJECT_DEFAULTS.get(subject_id, {"image_types": ["academic_diagram", "data_chart", "annotated_figure"], "competencies": ["reasoning", "calculation", "diagram_interpretation"]})

    kp_name = info.get("knowledge_point_name") or info.get("kp_name") or info.get("kp_id", "该知识点")
    subject_name = info.get("subject_name") or subject_id
    module_name = info.get("module_name") or info.get("module_id") or "未指定模块"

    info.setdefault("subject_name", subject_name)
    info.setdefault("kp_name", kp_name)
    info.setdefault("knowledge_point_name", kp_name)
    info.setdefault("module_name", module_name)

    if not info.get("scope_boundary"):
        info["scope_boundary"] = (
            f"围绕{subject_name}中{module_name}模块的“{kp_name}”出题；"
            f"不得扩展到医学类知识、临床诊断、药物治疗或人体疾病处置；"
            f"题目必须考查该知识点本身的概念、结构、过程、公式、数据判读或工程/科学应用。"
        )
    else:
        info["scope_boundary"] = (
            f"{info['scope_boundary']}；同时不得扩展到医学类知识、临床诊断、药物治疗或人体疾病处置。"
        )

    info["allowed_image_types"] = _dedupe(_as_list(info.get("allowed_image_types")) + defaults["image_types"])
    info["competency_types"] = _dedupe(_as_list(info.get("competency_types")) + defaults["competencies"])

    if not info.get("question_archetypes"):
        info["question_archetypes"] = [
            f"基于图示信息判断{kp_name}的关键概念或结构",
            f"基于图中数据/标注计算{kp_name}相关结果",
            f"比较图中两个或多个对象在{kp_name}上的差异",
        ]
    else:
        info["question_archetypes"] = _dedupe(_as_list(info["question_archetypes"]) + [
            f"基于精确图示信息考查{kp_name}",
        ])

    base_visual = info.get("required_visual_information") or (
        f"图片必须提供解题所需的{kp_name}关键信息，至少包含明确对象、数量、位置关系、连接/流程/变化关系和可读标注。"
    )
    info["required_visual_information"] = f"{base_visual} {IMAGE_PROMPT_PRECISION_RULE}"

    info.setdefault("validation_methods", ["answer_consistency_check", "image_prompt_consistency_check", "expert_logic_check"])
    info.setdefault("visual_adaptability", "fallback_enriched")
    info.setdefault("variation_capacity", max(4, int(float(info.get("importance") or 3))))
    info.setdefault("professional_risk", "standard")
    info.setdefault("ambiguity_risk", "controlled_by_prompt_precision")
    info.setdefault("duplication_risk", "controlled_by_archetype_rotation")
    info["kp_enrichment_applied"] = True
    info["question_blueprint"] = build_question_blueprint(info)
    return info
