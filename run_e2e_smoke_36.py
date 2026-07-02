#!/usr/bin/env python3
"""Lightweight E2E smoke test: 18 subjects × 2 KPs × 1 question = 36 questions."""
import csv
import json
import os
import random
import shutil
import sqlite3
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.generator import QuestionGenerator
from pipeline.image_renderer import GPTImageRenderer
from pipeline.image_quality_gate import quality_gate
from pipeline.kp_enrichment import enrich_kp_for_image_prompt
from pipeline.orchestrator import Config, ProductionOrchestrator
from pipeline.reviewer import QwenReviewer

SUBJECTS = ["S01", "S02", "S03", "S04", "S05", "S06", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20", "S21", "S22", "S23", "S24"]
MEDICAL = {f"S{i:02d}" for i in range(7, 13)}
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SMOKE_DIR = PROJECT_ROOT / "smoke_runs" / f"e2e_36_{RUN_ID}"
IMAGES_ROOT = SMOKE_DIR / "images"
DB_PATH = SMOKE_DIR / "smoke.db"
DESKTOP = Path("/mnt/c/Users/admin/Desktop")
FINAL_DIR = DESKTOP / f"qbank_e2e_36_submission_{RUN_ID}"
LOG_FILE = SMOKE_DIR / "run.log"
STOP_EVENT = threading.Event()
LOG_LOCK = threading.Lock()


def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with LOG_LOCK:
        print(line, flush=True)
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def feishu(msg: str):
    # Use Hermes CLI if available; otherwise keep log-only. The chat agent also mirrors key stages.
    log("FEISHU_REPORT: " + msg.replace("\n", " | "))


def load_env():
    env_file = PROJECT_ROOT / "config" / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ[k] = v


def load_population_details():
    details = {}
    pop = PROJECT_ROOT.parent / "population" / "full_18subject_kp_population.csv"
    with pop.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kp_id = row.get("kp_id", "")
            if not kp_id:
                continue
            details[kp_id] = {
                "knowledge_point_id": kp_id,
                "kp_id": kp_id,
                "subject_id": row.get("subject_id", ""),
                "subject_name": row.get("subject_name", ""),
                "knowledge_point_name": row.get("kp_name", ""),
                "kp_name": row.get("kp_name", ""),
                "module_id": row.get("module_id", ""),
                "module_name": row.get("module_name", ""),
                "importance": float(row.get("importance") or 0),
                "target_quota": float(row.get("target_quota") or 0),
            }

    input_dir = PROJECT_ROOT.parent / "input_snapshot"
    if input_dir.exists():
        for fp in sorted(input_dir.glob("*.json")):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                items = []
                if "modules" in data:
                    for module in data["modules"]:
                        items.extend(module.get("knowledge_points", []))
                elif "knowledge_points" in data:
                    items = data["knowledge_points"]
                for kp in items:
                    kp_id = kp["knowledge_point_id"]
                    details.setdefault(kp_id, {}).update(kp)
            except Exception as exc:
                log(f"WARN input_snapshot read failed {fp.name}: {exc}")
    return details


def choose_kps(config: Config, details: dict):
    quotas = config.quotas.get("kp_quotas", {})
    rng = random.Random(20260621)
    selected = []
    for subject_id in SUBJECTS:
        candidates = [
            kp_id for kp_id, info in quotas.items()
            if info.get("subject_id") == subject_id
            and subject_id not in MEDICAL
            and kp_id in details
            and info.get("production_quota", 0) > 0
        ]
        if len(candidates) < 2:
            raise RuntimeError(f"{subject_id} candidates < 2: {len(candidates)}")
        selected.extend(rng.sample(sorted(candidates), 2))
    return selected


def init_db():
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            question_id TEXT PRIMARY KEY,
            subject_id TEXT,
            subject_name TEXT,
            kp_id TEXT,
            kp_name TEXT,
            question_json TEXT,
            image_path TEXT,
            status TEXT,
            detail TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    return conn


def make_question_json(q, kp_info, image_path=""):
    return {
        "question_id": q.get("question_id", ""),
        "subject_id": kp_info.get("subject_id", ""),
        "subject_name": kp_info.get("subject_name", ""),
        "kp_id": kp_info.get("kp_id") or kp_info.get("knowledge_point_id", ""),
        "kp_name": kp_info.get("knowledge_point_name") or kp_info.get("kp_name", ""),
        "question_language": q.get("question_language", "zh"),
        "question_text": q.get("question_text", ""),
        "options": q.get("options", {}),
        "correct_answer": q.get("correct_answer", ""),
        "explanation": q.get("explanation", ""),
        "difficulty": q.get("difficulty", 0),
        "blueprint_slot": q.get("blueprint_slot"),
        "blueprint_archetype": q.get("blueprint_archetype", ""),
        "blueprint_image_type": q.get("blueprint_image_type", ""),
        "image_path": image_path,
        "image_prompt": q.get("image_prompt", ""),
        "final_image_prompt": QuestionGenerator.assemble_final_image_prompt(q.get("image_prompt", "")),
        "image_dependency_reason": q.get("image_dependency_reason", ""),
        "truth_spec": q.get("truth_spec", {}),
    }


def process_kp(kp_id, kp_info, generator, reviewer, image_renderer, conn_lock, conn):
    if STOP_EVENT.is_set():
        return {"kp_id": kp_id, "status": "SKIPPED_STOP"}
    subject_id = kp_info["subject_id"]
    try:
        last_error = ""
        for item_attempt in range(3):
            questions = []
            last_gen_error = ""
            for gen_attempt in range(3):
                questions = generator.generate_batch(kp_info, 1, existing_questions=[])
                if questions:
                    break
                last_gen_error = f"empty_generation_attempt_{gen_attempt + 1}"
                time.sleep(3)
            if not questions:
                last_error = f"GPT returned no valid question after 3 attempts: {last_gen_error}"
                continue

            q = questions[0]
            q["question_id"] = f"{kp_id}-SMOKE-{uuid.uuid4().hex[:8]}"
            qj_no_image = make_question_json(q, kp_info)

            review = None
            for review_attempt in range(3):
                review = reviewer.review(qj_no_image, kp_info)
                if review.get("decision") == "PASS":
                    break
                source = review.get("source", "")
                issues_text = " ; ".join(str(i) for i in review.get("issues", []))
                retryable = source in {"exception", "http_error", "parse_error"} or "timeout" in issues_text.lower() or "控制字符" in issues_text or "control character" in issues_text.lower()
                if retryable and review_attempt < 2:
                    log(f"Qwen retry {review_attempt + 1}/3 for {kp_id}: {source} {issues_text[:120]}")
                    time.sleep(5 * (review_attempt + 1))
                    continue
                break
            if not review or review.get("decision") != "PASS":
                last_error = f"Qwen FAIL item_attempt={item_attempt + 1}: {review.get('issues') if review else 'no_review'}"
                log(f"Regenerate after Qwen quality FAIL for {kp_id}: {last_error[:180]}")
                time.sleep(2)
                continue

            img_dir = IMAGES_ROOT / subject_id
            img_dir.mkdir(parents=True, exist_ok=True)
            img_path = img_dir / f"{q['question_id']}.png"
            ok, msg = image_renderer.render(q.get("image_prompt", ""), str(img_path))
            if not ok:
                last_error = f"image render failed item_attempt={item_attempt + 1}: {msg}"
                log(f"Regenerate after image render FAIL for {kp_id}: {last_error[:180]}")
                time.sleep(2)
                continue
            gate_ok, gate_issues = quality_gate(str(img_path), {"engine": "gpt-image-2", "diagram_type": "image_prompt"})
            if not gate_ok:
                last_error = f"quality gate failed item_attempt={item_attempt + 1}: {gate_issues}"
                log(f"Regenerate after quality gate FAIL for {kp_id}: {last_error[:180]}")
                time.sleep(2)
                continue

            qj = make_question_json(q, kp_info, str(img_path))
            with conn_lock:
                conn.execute(
                    "INSERT INTO questions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (qj["question_id"], subject_id, qj["subject_name"], kp_id, qj["kp_name"], json.dumps(qj, ensure_ascii=False), str(img_path), "FINAL_PASS", "", datetime.now().isoformat())
                )
                conn.commit()
            return {"kp_id": kp_id, "subject_id": subject_id, "status": "FINAL_PASS", "question_id": qj["question_id"], "image_path": str(img_path)}

        raise RuntimeError(last_error or "failed after item regeneration attempts")
    except Exception as exc:
        STOP_EVENT.set()
        err = f"{type(exc).__name__}: {str(exc)[:500]}"
        with conn_lock:
            conn.execute(
                "INSERT INTO questions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"ERR-{kp_id}", subject_id, kp_info.get("subject_name", ""), kp_id, kp_info.get("knowledge_point_name", ""), "{}", "", "FAIL", err, datetime.now().isoformat())
            )
            conn.commit()
        feishu(f"🚨 轻量E2E测试失败：{kp_id}\n{err}")
        raise

def export_submission(conn):
    if FINAL_DIR.exists():
        shutil.rmtree(FINAL_DIR)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = conn.execute("SELECT * FROM questions WHERE status='FINAL_PASS' ORDER BY subject_id, kp_id").fetchall()
    grouped = {}
    for row in rows:
        qj = json.loads(row[5])
        sid = qj["subject_id"]
        sname = qj["subject_name"]
        subject_dir = FINAL_DIR / f"{sid}_{sname}"
        images_dir = subject_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        src = Path(qj["image_path"])
        dst = images_dir / src.name
        shutil.copy2(src, dst)
        qj["image_path"] = f"images/{dst.name}"
        grouped.setdefault((sid, sname), []).append(qj)
    subjects = []
    total = 0
    for (sid, sname), items in sorted(grouped.items()):
        subject_dir = FINAL_DIR / f"{sid}_{sname}"
        json_path = subject_dir / f"{sid}_{sname}.json"
        json_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        subjects.append({"subject_id": sid, "subject_name": sname, "count": len(items), "json_file": f"{sid}_{sname}/{json_path.name}", "images_dir": f"{sid}_{sname}/images"})
        total += len(items)
    index = {"version": "e2e_smoke_36", "generated_at": datetime.now().isoformat(), "total_questions": total, "subjects": subjects}
    (FINAL_DIR / "dataset_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"final_dir": str(FINAL_DIR), "total_questions": total, "subjects": len(subjects)}


def main():
    start = time.time()
    load_env()
    feishu("🚀 轻量E2E测试开始：18科×2KP×1题=36题。阶段1：加载配置和抽样KP。")
    config = Config()
    details = load_population_details()
    selected = choose_kps(config, details)
    enriched = []
    for kp_id in selected:
        info = dict(details[kp_id])
        quota = config.quotas.get("kp_quotas", {}).get(kp_id, {})
        info.update({"production_quota": quota.get("production_quota", 1), "original_quota": quota.get("original_quota", 1)})
        enriched.append(enrich_kp_for_image_prompt(info))
    feishu(f"✅ 阶段1完成：已抽样 {len(enriched)} 个KP，覆盖18科，每科2个。")

    conn = init_db()
    conn_lock = threading.Lock()
    gpt_keys = [os.environ.get("GPT5_API_KEY", "")] + [os.environ.get(f"GPT_WORKER{i}_API_KEY", "") for i in range(1, 9)]
    gpt_keys = [k for k in gpt_keys if k]
    if len(gpt_keys) < 1:
        raise RuntimeError("No GPT keys configured")
    gpt_base_url = os.environ.get("GPT5_BASE_URL", "https://api.lk888.ai/v1")
    gpt_model = "gpt-5.5"
    generators = [QuestionGenerator(gpt_base_url, key, gpt_model, max_concurrent=1, response_log_dir=str(SMOKE_DIR / f"api_responses_w{i}")) for i, key in enumerate(gpt_keys[:9])]
    reviewer = QwenReviewer(config.qwen_base_url, config.qwen_api_key, config.qwen_model)
    image_renderer = GPTImageRenderer()

    feishu(f"🚧 阶段2开始：并发生成/审核/出图/检查。GPT workers={len(generators)}, Qwen并发按API调用，gpt-image-2={image_renderer.model}。")
    results = []
    with ThreadPoolExecutor(max_workers=min(9, len(generators))) as pool:
        futures = []
        for idx, kp_info in enumerate(enriched):
            gen = generators[idx % len(generators)]
            futures.append(pool.submit(process_kp, kp_info["kp_id"], kp_info, gen, reviewer, image_renderer, conn_lock, conn))
        done_count = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            done_count += 1
            if done_count % 6 == 0 or done_count == len(futures):
                feishu(f"📍 阶段2进度：{done_count}/{len(futures)} 完成，当前PASS={sum(1 for r in results if r['status']=='FINAL_PASS')}。")
            if STOP_EVENT.is_set():
                break

    pass_count = sum(1 for r in results if r["status"] == "FINAL_PASS")
    if pass_count != 36:
        raise RuntimeError(f"Expected 36 FINAL_PASS, got {pass_count}")
    feishu("✅ 阶段2完成：36题全部 FINAL_PASS，开始导出桌面提交包。")
    export_info = export_submission(conn)
    if export_info["total_questions"] != 36:
        raise RuntimeError(f"Export integrity failed: expected 36 questions, got {export_info['total_questions']}")
    if export_info["subjects"] != 18:
        raise RuntimeError(f"Export integrity failed: expected 18 subjects, got {export_info['subjects']}")
    elapsed = time.time() - start
    feishu(f"🎉 轻量E2E测试完成：{export_info['total_questions']}题，{export_info['subjects']}科。桌面目录：{export_info['final_dir']}。耗时 {elapsed/60:.1f} 分钟。")
    print(json.dumps({"results": results, "export": export_info, "elapsed_seconds": elapsed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
