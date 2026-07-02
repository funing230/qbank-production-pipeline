#!/usr/bin/env python3
"""One-question end-to-end smoke for qwen3.6 pre-image pilot.

Runs exactly one isolated question through:
GPT generation -> local validation -> qwen3.6-flash 5-rollout review -> image render.
Does not write production.db or start queue workers.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_env() -> None:
    env_path = ROOT / "config" / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()


def pick_key(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def main() -> int:
    load_env()

    from pipeline.generator import QuestionGenerator
    from pipeline.reviewer import QwenReviewer, pre_validate
    from pipeline.image_renderer import GPTImageRenderer
    from pipeline.image_quality_gate import quality_gate

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "smoke_runs" / f"qwen36_one_question_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gpt_base = os.environ.get("GPT5_BASE_URL") or os.environ.get("GPT_BASE_URL") or "https://api.lk888.ai/v1"
    gpt_key = pick_key("GPT_WORKER1_API_KEY", "GPT_API_KEY", "GPT5_API_KEY")
    gpt_model = os.environ.get("GPT5_MODEL") or os.environ.get("GPT_MODEL") or "gpt-5.5"

    qwen_base = os.environ.get("QWEN_BASE_URL", "https://yuanlansj.xin/v1")
    qwen_key = pick_key("QWEN_API_KEY", "YUANLAN_API_KEY")
    qwen_model = "qwen3.6-flash"

    if not gpt_key:
        raise RuntimeError("Missing GPT key in config/.env")
    if not qwen_key:
        raise RuntimeError("Missing Qwen/Yuanlan key in config/.env")

    kp_info = {
        "subject_id": "S01",
        "subject_name": "组合与离散数学",
        "module_name": "图论基础",
        "kp_id": "S01-SMOKE-001",
        "kp_name": "最短路径与加权图",
        "scope_boundary": "只考查加权无向图中从指定起点到指定终点的最短路径判断，不涉及负权边或复杂算法证明。",
        "question_archetypes": ["根据图示边权判断最短路径"],
        "competency_types": ["diagram_interpretation", "quantitative_reasoning"],
        "allowed_image_types": ["weighted_graph_diagram"],
        "required_visual_information": "图中必须显示4个节点、5条带权边，以及节点标签。",
        "question_blueprint": [
            {
                "slot": 1,
                "question_language": "zh",
                "difficulty": 3,
                "archetype": "根据图示边权判断最短路径",
                "image_type": "weighted_graph_diagram",
                "competency": "diagram_interpretation",
                "visual_complexity": "4 nodes, 5 weighted edges, one shortest-path comparison",
                "design_goal": "生成一题必须依赖图中边权才能判断的中文选择题。",
            }
        ],
        "prohibited_patterns": ["题干直接列出所有边权", "选项泄露完整路径长度计算过程"],
    }

    t0 = time.time()
    generator = QuestionGenerator(
        base_url=gpt_base,
        api_key=gpt_key,
        model=gpt_model,
        max_concurrent=1,
        response_log_dir=str(out_dir / "api_responses"),
    )
    questions = generator.generate_batch(kp_info, batch_size=1, existing_questions=[])
    if not questions:
        raise RuntimeError("GPT generation returned no valid question")
    question = questions[0]
    gen_sec = time.time() - t0
    (out_dir / "question.json").write_text(json.dumps(question, ensure_ascii=False, indent=2), encoding="utf-8")

    pre_ok, pre_issues = pre_validate(question)
    if not pre_ok:
        raise RuntimeError(f"pre_validate failed: {pre_issues}")

    t1 = time.time()
    reviewer = QwenReviewer(qwen_base, qwen_key, qwen_model)
    review = reviewer.review_rollouts(question, kp_info, rollouts=5)
    review_sec = time.time() - t1
    (out_dir / "qwen_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")

    image_ok = False
    image_msg = "skipped: qwen decision not PASS"
    gate_ok = False
    gate_issues = []
    image_path = out_dir / "smoke.png"
    image_sec = 0.0
    if review.get("decision") == "PASS":
        os.environ.setdefault("GPT_IMAGE_LK888_TIMEOUT", "330")
        os.environ.setdefault("GPT_IMAGE_LK888_POLL_INTERVAL", "6")
        t2 = time.time()
        renderer = GPTImageRenderer(name="smoke")
        image_ok, image_msg = renderer.render(question.get("image_prompt", ""), str(image_path))
        image_sec = time.time() - t2
        if image_ok:
            gate_ok, gate_issues = quality_gate(str(image_path), {"diagram_type": question.get("blueprint_image_type", "")})

    summary = {
        "run_id": run_id,
        "out_dir": str(out_dir),
        "gpt_model": gpt_model,
        "qwen_model": qwen_model,
        "question_id": question.get("question_id"),
        "correct_answer": question.get("correct_answer"),
        "has_image_contract": isinstance(question.get("image_contract"), dict),
        "has_image_prompt": bool(question.get("image_prompt")),
        "pre_validate_pass": pre_ok,
        "pre_validate_issues": pre_issues,
        "qwen_decision": review.get("decision"),
        "qwen_wrong_count": review.get("wrong_count"),
        "qwen_correct_count": review.get("correct_count"),
        "qwen_valid_count": review.get("valid_count"),
        "image_render_pass": image_ok,
        "image_message": image_msg[:500],
        "image_path": str(image_path) if image_path.exists() else "",
        "image_size_bytes": image_path.stat().st_size if image_path.exists() else 0,
        "quality_gate_pass": gate_ok,
        "quality_gate_issues": gate_issues,
        "timing_sec": {
            "generation": round(gen_sec, 2),
            "qwen_review": round(review_sec, 2),
            "image_render": round(image_sec, 2),
            "total": round(time.time() - t0, 2),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0 if (pre_ok and review.get("decision") == "PASS" and image_ok and gate_ok) else 2


if __name__ == "__main__":
    raise SystemExit(main())
