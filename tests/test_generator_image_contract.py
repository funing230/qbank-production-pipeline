import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.generator import QuestionGenerator


def make_generator():
    return QuestionGenerator("https://example.invalid/v1", "dummy", max_concurrent=1)


def valid_question():
    return {
        "question_text": "根据图中给出的网络结构，哪一条边的权重满足最短路径的关键条件？",
        "options": {"A": "A→B", "B": "B→C", "C": "C→D", "D": "A→D"},
        "correct_answer": "B",
        "explanation": "图中 A 到 D 的候选路径需要比较边权，B→C 是满足条件的关键边。",
        "difficulty": 3,
        "question_language": "zh",
        "image_prompt": (
            "A directed graph diagram showing a weighted path comparison, viewed from above with four circular nodes arranged in a diamond layout. "
            "The top node is labeled A, the left node is labeled B, the right node is labeled C, and the bottom node is labeled D. "
            "Show four directed arrows: A to B labeled 2, B to D labeled 5, A to C labeled 4, and C to D labeled 1. "
            "Use a thin blue highlight around the C to D arrow only, while the other arrows stay black. "
            "Place a small legend labeled path weight beside the graph and keep all numeric labels clear."
        ),
        "image_contract": {
            "version": "1.0",
            "diagram_type": "directed graph diagram",
            "visual_facts": ["Four nodes A, B, C, D form a diamond layout."],
            "answer_relevant_facts": ["The C to D arrow has weight 1 and is highlighted."],
            "labels": ["A", "B", "C", "D", "path weight"],
            "quantities": {"A_to_B": 2, "B_to_D": 5, "A_to_C": 4, "C_to_D": 1},
            "relations": ["A->B", "B->D", "A->C", "C->D"],
            "constraints": ["Use only the listed nodes, arrows, weights, and labels."],
            "distractor_rationales": {
                "A": "Confuses the first local edge with the required comparison.",
                "B": "正确",
                "C": "Chooses the highlighted downstream edge without checking the condition.",
                "D": "Treats the direct edge as automatically decisive.",
            },
            "qwen36_difficulty_rationale": "Requires combining direction, edge weights, and the stated condition rather than reading one label.",
            "forbidden_text_leakage": ["C to D arrow has weight 1 and is highlighted"],
        },
        "image_dependency_reason": "必须读取图中边权才能判断。",
    }


def test_validate_question_requires_image_contract():
    q = valid_question()
    del q["image_contract"]
    assert make_generator()._validate_question(q) is False


def test_validate_question_rejects_missing_contract_schema_key():
    q = valid_question()
    del q["image_contract"]["relations"]
    assert make_generator()._validate_question(q) is False


def test_validate_question_rejects_forbidden_text_leakage_in_question_or_options():
    q = valid_question()
    q["question_text"] += " C to D arrow has weight 1 and is highlighted"
    assert make_generator()._validate_question(q) is False


def test_validate_question_rejects_final_conclusion_inside_contract():
    q = valid_question()
    q["image_contract"]["answer_relevant_facts"].append("The shortest path is A to C to D.")
    assert make_generator()._validate_question(q) is False


def test_validate_question_requires_distractor_rationales_mapping():
    q = valid_question()
    del q["image_contract"]["distractor_rationales"]["D"]
    assert make_generator()._validate_question(q) is False


def test_build_messages_adds_qwen36_difficulty_gate():
    messages = make_generator()._build_messages(
        {
            "subject_id": "S07",
            "subject_name": "基础医学",
            "kp_id": "S07-M01-001",
            "kp_name": "心电图基础",
        },
        batch_size=1,
    )
    system_prompt = messages[0]["content"]
    assert "synthetic educational medical diagrams" in system_prompt
    assert "simulated ECG" in system_prompt
    assert "真实 patient image" in system_prompt
    assert "Qwen3.6 难度门" in system_prompt
    assert "至少需要 2-3 个推理步骤" in system_prompt
    assert "distractor_rationales" in system_prompt
    assert "禁止写 `shortest path is A-C-D`" in system_prompt
