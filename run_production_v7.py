#!/usr/bin/env python3
"""
V7 全量生产脚本 — image_prompt 端到端并发
=============================================
架构:
  - 9路GPT worker 生成题目文本 + image_prompt
  - Qwen 15路并发纯文本审核（题目文本 ↔ image_prompt 一致性）
  - gpt-image-2 渲染：Qwen PASS 后才调用
  - image_quality_gate：渲染后像素级检查
  - 每100道FINAL_PASS随机抽3道发飞书审核
  - FAIL→GPT重生成，最多2轮（共3次机会）

流程:
  KP队列 → GPT生成batch → Qwen文本审核 → gpt-image-2渲染 → quality_gate → FINAL_PASS
       ↑                                  ↓ FAIL
       └──────── GPT重生成(最多2轮) ←──────┘
"""
import sys
import os
import csv
import json
import time
import signal
import random
import threading
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.orchestrator import ProductionOrchestrator, Config
from pipeline.generator import QuestionGenerator
from pipeline.db import ProductionDB
from pipeline.reviewer import QwenReviewer
from pipeline.image_renderer import GPTImageRenderer
from pipeline.image_quality_gate import quality_gate
from pipeline.kp_enrichment import enrich_kp_for_image_prompt

MEDICAL_EXCLUDED_SUBJECTS = {"S07", "S08", "S09", "S10", "S11", "S12"}

SUBJECTS = [
    "S01", "S02", "S03", "S04", "S05", "S06",
    "S13", "S14", "S15", "S16", "S17", "S18",
    "S19", "S20", "S21", "S22", "S23", "S24",
]

if any(subject_id in MEDICAL_EXCLUDED_SUBJECTS for subject_id in SUBJECTS):
    raise RuntimeError("医学类科目 S07-S12 不允许进入本次18科生产流程")

NUM_GPT_WORKERS = 9
QWEN_CONCURRENCY = 15
BATCH_SIZE = 8
MAX_REGEN_ROUNDS = 2
API_COOLDOWN = 0
SAMPLE_EVERY_N = 100
SAMPLE_COUNT = 3
TAIL_FILL_MODE = os.environ.get("TAIL_FILL_MODE", "1").lower() not in {"0", "false", "no"}
TAIL_FILL_DEFAULT_SUBJECT_TARGET = int(os.environ.get("TAIL_FILL_SUBJECT_TARGET", "1333"))
TAIL_FILL_MAX_ACTIVE_SUBJECTS = int(os.environ.get("TAIL_FILL_MAX_ACTIVE_SUBJECTS", "1"))
TAIL_FILL_MAX_KPS_PER_SUBJECT = int(os.environ.get("TAIL_FILL_MAX_KPS_PER_SUBJECT", "1"))

LOG_FILE = PROJECT_ROOT / "production_v7.log"
FEISHU_SAMPLE_DIR = PROJECT_ROOT / "feishu_samples"

_log_lock = threading.Lock()
_db_lock = threading.Lock()
_qwen_semaphore = None
_image_semaphore = None
_image_executor = None
_pass_counter_lock = threading.Lock()
_pass_counter = 0
_last_sample_at = 0
_recent_passes = []
_running = True


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with _log_lock:
        print(line, flush=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


def maybe_sample_to_feishu(question_info: dict):
    global _pass_counter, _last_sample_at, _recent_passes
    with _pass_counter_lock:
        _pass_counter += 1
        _recent_passes.append(question_info)
        if _pass_counter - _last_sample_at >= SAMPLE_EVERY_N:
            pool = _recent_passes[-SAMPLE_EVERY_N:]
            samples = random.sample(pool, min(SAMPLE_COUNT, len(pool)))
            FEISHU_SAMPLE_DIR.mkdir(exist_ok=True)
            batch_id = f"sample_{_pass_counter}"
            for i, sample in enumerate(samples):
                out_file = FEISHU_SAMPLE_DIR / f"{batch_id}_{i}.json"
                out_file.write_text(json.dumps(sample, ensure_ascii=False, indent=2))
            log(f"  📋 抽样: PASS={_pass_counter}, 抽取{len(samples)}道 → feishu_samples/{batch_id}_*.json")
            _last_sample_at = _pass_counter
            if len(_recent_passes) > 200:
                _recent_passes = _recent_passes[-100:]


def make_question_json(q: dict, subject_id: str, kp_id: str, kp_info: dict, image_path: str = "") -> dict:
    return {
        "question_id": q.get("question_id", ""),
        "subject_id": subject_id,
        "kp_id": kp_id,
        "kp_name": kp_info.get("knowledge_point_name", "") or kp_info.get("kp_name", ""),
        "question_text": q.get("question_text", ""),
        "options": q.get("options", {}),
        "correct_answer": q.get("correct_answer", ""),
        "explanation": q.get("explanation", ""),
        "difficulty": q.get("difficulty", 0),
        "question_language": q.get("question_language", "zh"),
        "blueprint_slot": q.get("blueprint_slot"),
        "blueprint_archetype": q.get("blueprint_archetype", ""),
        "blueprint_image_type": q.get("blueprint_image_type", ""),
        "image_prompt": q.get("image_prompt", ""),
        "final_image_prompt": QuestionGenerator.assemble_final_image_prompt(q.get("image_prompt", "")),
        "image_path": image_path,
        "truth_spec": q.get("truth_spec", {}),
        "image_dependency_reason": q.get("image_dependency_reason", ""),
        "difference_from_others": q.get("difference_from_others", ""),
    }


def normalize_question(q: dict, subject_id: str, kp_id: str, kp_info: dict) -> dict:
    import uuid
    q.setdefault("subject_id", subject_id)
    q.setdefault("module_id", kp_id.rsplit("-", 1)[0] if "-" in kp_id else "")
    q.setdefault("kp_id", kp_id)
    q.setdefault("kp_name", kp_info.get("knowledge_point_name", "") or kp_info.get("kp_name", ""))
    if not q.get("question_id"):
        q["question_id"] = f"{kp_id}-Q{uuid.uuid4().hex[:6]}"
    q["question_json"] = make_question_json(q, subject_id, kp_id, kp_info)
    return q


def review_text(q: dict, kp_info: dict, reviewer: QwenReviewer) -> dict:
    with _qwen_semaphore:
        result = reviewer.review(q.get("question_json", q), kp_info)
    return {
        "verdict": result.get("decision", "FAIL"),
        "issues": result.get("issues", []),
        "checks": result.get("checks", {}),
        "confidence": result.get("confidence", 0),
        "source": result.get("source", "qwen_api"),
        "latency": result.get("latency", 0),
    }


def _render_and_gate_worker(q: dict, img_path: str, image_renderer: GPTImageRenderer) -> tuple:
    ok, msg = image_renderer.render(q.get("image_prompt", ""), img_path)
    if not ok:
        return False, f"image_render: {msg}"
    gate_ok, gate_issues = quality_gate(img_path, {"engine": "gpt-image-2", "diagram_type": "image_prompt"})
    if not gate_ok:
        return False, "quality_gate: " + "; ".join(gate_issues)
    return True, msg


def render_and_gate(q: dict, img_path: str, image_renderer: GPTImageRenderer) -> tuple:
    """Submit image generation + quality_gate to the global image thread pool.

    Each task owns a unique img_path ({question_id}.png), so concurrent image jobs
    cannot overwrite or mix results.
    """
    if _image_executor is None:
        return _render_and_gate_worker(q, img_path, image_renderer)
    future = _image_executor.submit(_render_and_gate_worker, q, img_path, image_renderer)
    return future.result()


def process_single_question(
    q: dict, qid: str, img_path: str,
    kp_info: dict, generator: QuestionGenerator,
    reviewer: QwenReviewer, image_renderer: GPTImageRenderer,
    db: ProductionDB, worker_id: int, existing: list,
    subject_id: str, kp_id: str,
) -> bool:
    current_q = q
    for regen_round in range(MAX_REGEN_ROUNDS + 1):
        review = review_text(current_q, kp_info, reviewer)
        if review["verdict"] != "PASS":
            issues = "; ".join(str(i) for i in review.get("issues", [])) or review.get("source", "unknown")
            if regen_round >= MAX_REGEN_ROUNDS:
                with _db_lock:
                    db.update_question_status(qid, "SENTINEL_FAIL_FINAL", f"qwen_text_fail: {issues[:400]}")
                log(f"    [W{worker_id}][{qid}] ✗ Qwen FAIL after {regen_round+1} rounds: {issues[:80]}")
                return False
            log(f"    [W{worker_id}][{qid}] Qwen FAIL(round {regen_round}): {issues[:60]} → regen...")
            with _db_lock:
                db.update_question_status(qid, "SENTINEL_REGEN", f"round={regen_round}: {issues[:300]}")
            retry_questions = generator.generate_batch(kp_info, 1, existing)
            if not retry_questions:
                with _db_lock:
                    db.update_question_status(qid, "SENTINEL_FAIL_FINAL", f"regen_empty_round_{regen_round+1}")
                return False
            current_q = normalize_question(retry_questions[0], subject_id, kp_id, kp_info)
            continue

        with _db_lock:
            db.update_question_status(qid, "ACCEPTED", f"qwen_conf={review.get('confidence', 0)},round={regen_round}")

        ok, render_msg = render_and_gate(current_q, img_path, image_renderer)
        if not ok:
            if regen_round >= MAX_REGEN_ROUNDS:
                with _db_lock:
                    db.update_question_status(qid, "RENDER_FAIL", render_msg[:500])
                log(f"    [W{worker_id}][{qid}] ✗ Image FAIL after {regen_round+1} rounds: {render_msg[:80]}")
                return False
            log(f"    [W{worker_id}][{qid}] Image FAIL(round {regen_round}): {render_msg[:60]} → regen...")
            with _db_lock:
                db.update_question_status(qid, "SENTINEL_REGEN", f"image_fail_round={regen_round}: {render_msg[:300]}")
            retry_questions = generator.generate_batch(kp_info, 1, existing)
            if not retry_questions:
                with _db_lock:
                    db.update_question_status(qid, "SENTINEL_FAIL_FINAL", f"image_regen_empty_round_{regen_round+1}")
                return False
            current_q = normalize_question(retry_questions[0], subject_id, kp_id, kp_info)
            continue

        rel_path = str(Path(img_path).relative_to(Path(img_path).parents[2])) if False else img_path
        current_q["image_path"] = rel_path
        current_q["question_json"] = make_question_json(current_q, subject_id, kp_id, kp_info, rel_path)
        with _db_lock:
            db.update_question_status(qid, "FINAL_PASS", f"qwen_conf={review.get('confidence', 0)},round={regen_round}; {render_msg[:120]}")
            db.update_question_json(qid, current_q["question_json"])
        maybe_sample_to_feishu({
            "question_id": qid,
            "kp_id": kp_id,
            "question_json": current_q["question_json"],
            "image_path": rel_path,
        })
        return True
    return False


def tail_fill_quota(need: int) -> int:
    if need <= 0:
        return 0
    if need <= 3:
        return need + 3
    if need <= 10:
        return int(need * 1.5 + 0.999)
    return int(need * 1.3 + 0.999)


def subject_target(config: Config, subject_id: str) -> int:
    targets = config.quotas.get("subject_targets", {})
    return int(targets.get(subject_id, TAIL_FILL_DEFAULT_SUBJECT_TARGET))


def build_tail_fill_queue(config: Config, orch: ProductionOrchestrator) -> tuple[Queue, int, int, dict]:
    quotas = config.quotas.get("kp_quotas", {})
    kp_queue = Queue()
    skipped_medical = 0
    subject_rows = []
    with _db_lock:
        for subject_id in SUBJECTS:
            if subject_id in MEDICAL_EXCLUDED_SUBJECTS:
                skipped_medical += 1
                continue
            target = subject_target(config, subject_id)
            have = orch.db.count_final_pass_questions_for_subject(subject_id)
            gap = max(0, target - have)
            if gap > 0:
                subject_rows.append((gap, subject_id, target, have))

    subject_rows.sort()
    if TAIL_FILL_MAX_ACTIVE_SUBJECTS > 0:
        subject_rows = subject_rows[:TAIL_FILL_MAX_ACTIVE_SUBJECTS]
    active_subjects = {subject_id for _, subject_id, _, _ in subject_rows}

    total_target = 0
    plan = {}
    for gap, subject_id, target, have in subject_rows:
        subject_plan = []
        subject_cap = tail_fill_quota(gap)
        subject_candidates = []
        for kp_id, info in sorted(quotas.items()):
            if info.get("subject_id") != subject_id:
                continue
            if kp_id.split("-M")[0] in MEDICAL_EXCLUDED_SUBJECTS:
                skipped_medical += 1
                continue
            quota = int(info.get("production_quota", 0))
            if quota <= 0:
                continue
            with _db_lock:
                kp_have = orch.db.count_questions_for_kp(kp_id)
            kp_need = quota - kp_have
            if kp_need <= 0:
                continue
            subject_candidates.append((kp_need, kp_id, kp_have, quota))
        subject_candidates.sort(reverse=True)
        for kp_need, kp_id, kp_have, quota in subject_candidates[:TAIL_FILL_MAX_KPS_PER_SUBJECT]:
            fill_quota = kp_have + tail_fill_quota(kp_need)
            kp_queue.put((kp_id, fill_quota))
            subject_plan.append({"kp_id": kp_id, "have": kp_have, "quota": quota, "need": kp_need, "fill_quota": fill_quota})
        total_target += subject_cap
        plan[subject_id] = {"target": target, "have": have, "gap": gap, "kps": subject_plan}

    if not active_subjects:
        log("Tail-fill: 所有科目已达到目标，无需继续生产")
    else:
        desc = ", ".join(f"{sid}:have={have}/target={target},gap={gap}" for gap, sid, target, have in subject_rows)
        log(f"Tail-fill active subjects: {desc}")
    return kp_queue, total_target, skipped_medical, plan


def build_full_queue(config: Config) -> tuple[Queue, int, int]:
    quotas = config.quotas.get("kp_quotas", {})
    kp_queue = Queue()
    total_target = 0
    skipped_medical = 0
    for subject_id in SUBJECTS:
        if subject_id in MEDICAL_EXCLUDED_SUBJECTS:
            skipped_medical += 1
            continue
        subject_kps = sorted([
            (kp_id, info["production_quota"])
            for kp_id, info in quotas.items()
            if info.get("subject_id") == subject_id
            and info.get("subject_id") not in MEDICAL_EXCLUDED_SUBJECTS
            and info.get("production_quota", 0) > 0
        ])
        for kp_id, quota in subject_kps:
            if kp_id.split("-M")[0] in MEDICAL_EXCLUDED_SUBJECTS:
                skipped_medical += 1
                continue
            kp_queue.put((kp_id, quota))
            total_target += quota
    return kp_queue, total_target, skipped_medical


def produce_kp(
    worker_id: int, kp_id: str, kp_quota: int,
    generator: QuestionGenerator, reviewer: QwenReviewer, image_renderer: GPTImageRenderer,
    orch: ProductionOrchestrator, stats: dict, stats_lock: threading.Lock
) -> int:
    global _running
    subject_id = kp_id.split("-M")[0]
    kp_info = orch.get_kp_info(kp_id)
    with _db_lock:
        have = orch.db.count_questions_for_kp(kp_id)
    remaining = kp_quota - have
    if remaining <= 0:
        return 0
    log(f"  [W{worker_id}][{kp_id}] quota={kp_quota}, have={have}, need={remaining}")

    with _db_lock:
        cur = orch.db.conn.execute(
            "SELECT question_json FROM questions WHERE kp_id = ? AND quality_status = 'FINAL_PASS'",
            (kp_id,)
        )
        rows = cur.fetchall()
    existing = []
    for row in rows:
        try:
            existing.append(json.loads(row[0]) if isinstance(row[0], str) else row[0])
        except Exception:
            pass

    produced_pass = 0
    consecutive_errors = 0
    while produced_pass < remaining and _running:
        if TAIL_FILL_MODE:
            with _db_lock:
                subject_have = orch.db.count_final_pass_questions_for_subject(subject_id)
            subject_gap = subject_target(orch.config, subject_id) - subject_have
            if subject_gap <= 0:
                log(f"  [W{worker_id}][{kp_id}] 科目{subject_id}已满({subject_have})，停止补洞")
                break
        else:
            subject_gap = remaining - produced_pass
        if consecutive_errors >= 5:
            log(f"  [W{worker_id}][{kp_id}] ⚠ 连续{consecutive_errors}次错误，跳过")
            break
        batch_size = min(BATCH_SIZE, remaining - produced_pass, max(1, subject_gap))
        try:
            questions = generator.generate_batch(kp_info, batch_size, existing)
            if not questions:
                consecutive_errors += 1
                time.sleep(5)
                continue
            questions = [normalize_question(q, subject_id, kp_id, kp_info) for q in questions]

            with _db_lock:
                batch_db_id = orch.db.create_batch(subject_id, kp_id, len(questions))
                q_ids = orch.db.add_questions(batch_db_id, questions)

            batch_pass = 0
            for q, qid in zip(questions, q_ids):
                if not _running:
                    break
                img_dir = orch.config.output_dir / subject_id / "images"
                img_dir.mkdir(parents=True, exist_ok=True)
                img_path = str(img_dir / f"{qid}.png")
                passed = process_single_question(
                    q, qid, img_path, kp_info, generator,
                    reviewer, image_renderer, orch.db, worker_id, existing,
                    subject_id, kp_id,
                )
                if passed:
                    batch_pass += 1
                    existing.append(q)

            produced_pass += batch_pass
            consecutive_errors = 0
            with stats_lock:
                stats["total_generated"] += len(questions)
                stats["total_reviewed"] += len(questions)
                stats["total_pass"] += batch_pass
                stats[f"w{worker_id}_pass"] = stats.get(f"w{worker_id}_pass", 0) + batch_pass
            log(f"  [W{worker_id}][{kp_id}] batch: {len(questions)} gen/reviewed, {batch_pass} PASS (累计{produced_pass}/{remaining})")
            time.sleep(API_COOLDOWN)
        except Exception as e:
            err_msg = str(e)[:200]
            consecutive_errors += 1
            if "429" in err_msg or "502" in err_msg:
                log(f"  [W{worker_id}][{kp_id}] RATE LIMITED, wait 45s...")
                time.sleep(45)
            elif "timeout" in err_msg.lower():
                log(f"  [W{worker_id}][{kp_id}] TIMEOUT, wait 30s...")
                time.sleep(30)
            else:
                log(f"  [W{worker_id}][{kp_id}] ERROR: {err_msg}")
                traceback.print_exc()
                time.sleep(10)
    log(f"  [W{worker_id}][{kp_id}] ✓ 完成: {produced_pass} PASS")
    return produced_pass


def main():
    global _running, _qwen_semaphore, _image_semaphore, _image_executor
    env_file = PROJECT_ROOT / "config" / ".env"
    if env_file.exists():
        for line in env_file.read_text().strip().split('\n'):
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value

    log("=" * 60)
    log("V7 IMAGE_PROMPT PRODUCTION — GPT + Qwen文本审核 + gpt-image-2")
    log(f"Workers={NUM_GPT_WORKERS}, Qwen并发={QWEN_CONCURRENCY}, Batch={BATCH_SIZE}, 抽样={SAMPLE_COUNT}/{SAMPLE_EVERY_N}")
    log("=" * 60)

    _qwen_semaphore = threading.Semaphore(QWEN_CONCURRENCY)
    image_concurrency = int(os.environ.get("GPT_IMAGE_CONCURRENCY", "38"))
    _image_semaphore = threading.Semaphore(image_concurrency)
    _image_executor = ThreadPoolExecutor(max_workers=image_concurrency, thread_name_prefix="gpt-image-gate")
    config = Config()
    orch = ProductionOrchestrator(config)
    reviewer = QwenReviewer(config.qwen_base_url, config.qwen_api_key, config.qwen_model)
    image_renderer = GPTImageRenderer()

    all_kp_details = {}

    # 基础全量信息：1688个KP均应从 population CSV 获得中文科目名、中文模块名、中文知识点名。
    population_csv = PROJECT_ROOT.parent / "population" / "full_18subject_kp_population.csv"
    if population_csv.exists():
        with population_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                kp_id = row.get("kp_id", "")
                if not kp_id:
                    continue
                all_kp_details[kp_id] = {
                    "knowledge_point_id": kp_id,
                    "kp_id": kp_id,
                    "subject_id": row.get("subject_id", ""),
                    "subject_name": row.get("subject_name", ""),
                    "knowledge_point_name": row.get("kp_name", ""),
                    "kp_name": row.get("kp_name", ""),
                    "module_id": row.get("module_id", ""),
                    "module_name": row.get("module_name", ""),
                    "importance": int(float(row.get("importance") or 0)),
                    "target_quota": float(row.get("target_quota") or 0),
                }
    log(f"Loaded {len(all_kp_details)} KP base records from population CSV")

    # 增强详细信息：input_snapshot 覆盖/补充 scope_boundary、allowed_image_types、archetypes 等高级字段。
    input_dir = PROJECT_ROOT.parent / "input_snapshot"
    detailed_count = 0
    if input_dir.exists():
        for file_path in sorted(input_dir.glob("*.json")):
            try:
                data = json.loads(file_path.read_text())
                if "modules" in data:
                    for module in data["modules"]:
                        for kp in module.get("knowledge_points", []):
                            kp_id = kp["knowledge_point_id"]
                            all_kp_details.setdefault(kp_id, {}).update(kp)
                            detailed_count += 1
                elif "knowledge_points" in data:
                    for kp in data["knowledge_points"]:
                        kp_id = kp["knowledge_point_id"]
                        all_kp_details.setdefault(kp_id, {}).update(kp)
                        detailed_count += 1
            except Exception:
                pass
    log(f"Loaded/merged {detailed_count} detailed KP records from input_snapshot")

    orig_get = orch.get_kp_info
    def patched_get(kp_id):
        base = orig_get(kp_id)
        if kp_id in all_kp_details:
            base.update(all_kp_details[kp_id])
        return enrich_kp_for_image_prompt(base)
    orch.get_kp_info = patched_get

    gpt_keys = [
        os.environ.get("GPT5_API_KEY", ""),
        os.environ.get("GPT_WORKER1_API_KEY", ""),
        os.environ.get("GPT_WORKER2_API_KEY", ""),
        os.environ.get("GPT_WORKER3_API_KEY", ""),
        os.environ.get("GPT_WORKER4_API_KEY", ""),
        os.environ.get("GPT_WORKER5_API_KEY", ""),
        os.environ.get("GPT_WORKER6_API_KEY", ""),
        os.environ.get("GPT_WORKER7_API_KEY", ""),
        os.environ.get("GPT_WORKER8_API_KEY", ""),
    ]
    gpt_keys = [key for key in gpt_keys if key]
    worker_limit = NUM_GPT_WORKERS
    if TAIL_FILL_MODE:
        worker_limit = min(worker_limit, int(os.environ.get("TAIL_FILL_GPT_WORKERS", "6")))
    actual_workers = min(worker_limit, len(gpt_keys))
    gpt_base_url = os.environ.get("GPT5_BASE_URL", "https://api.lk888.ai/v1")
    gpt_model = "gpt-5.5"
    os.environ["GPT_MODEL"] = gpt_model

    generators = []
    for i, key in enumerate(gpt_keys[:actual_workers]):
        gen = QuestionGenerator(
            base_url=gpt_base_url,
            api_key=key,
            model=gpt_model,
            max_concurrent=1,
            response_log_dir=str(PROJECT_ROOT / f"api_responses_w{i}"),
        )
        generators.append(gen)
        log(f"  GPT Worker {i}: key=...{key[-4:]}")
    log(f"  实际可用GPT workers: {actual_workers}")
    log(f"  Qwen model: {config.qwen_model} @ {config.qwen_base_url}")
    log(f"  Image model: {image_renderer.model} @ {image_renderer.base_url}")

    def signal_handler(signum, frame):
        global _running
        log(f"Signal {signum} received, stopping...")
        _running = False
        orch.running = False
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    tail_fill_plan = {}
    if TAIL_FILL_MODE:
        kp_queue, total_target, skipped_medical, tail_fill_plan = build_tail_fill_queue(config, orch)
        log(f"\nTail-fill KP队列: {kp_queue.qsize()} 个KP, 估算补洞目标≈{total_target}题, 医学跳过={skipped_medical}")
    else:
        kp_queue, total_target, skipped_medical = build_full_queue(config)
        log(f"\nKP队列: {kp_queue.qsize()} 个KP, 总目标≈{total_target}题, 医学跳过={skipped_medical}")

    stats = {
        "start_time": time.time(),
        "total_generated": 0,
        "total_reviewed": 0,
        "total_pass": 0,
    }
    stats_lock = threading.Lock()

    def worker_loop(worker_id: int):
        gen = generators[worker_id]
        while _running:
            try:
                kp_id, kp_quota = kp_queue.get_nowait()
            except Exception:
                break
            try:
                produce_kp(worker_id, kp_id, kp_quota, gen, reviewer, image_renderer, orch, stats, stats_lock)
            except Exception as e:
                log(f"  [W{worker_id}] FATAL on {kp_id}: {str(e)[:150]}")
                traceback.print_exc()
            kp_queue.task_done()

    log(f"\n{'='*60}")
    log(f"开始生产! {actual_workers}路GPT并行...")
    log(f"{'='*60}\n")

    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        futures = [pool.submit(worker_loop, i) for i in range(actual_workers)]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                log(f"Worker exception: {e}")

    if _image_executor:
        _image_executor.shutdown(wait=True)

    elapsed_min = (time.time() - stats["start_time"]) / 60
    rate = stats["total_pass"] / max(1, elapsed_min)
    log(f"\n{'='*60}")
    log("PRODUCTION COMPLETE")
    log(f"{'='*60}")
    log(f"耗时: {elapsed_min:.0f}分钟 | 速率: {rate:.1f} PASS/分钟")
    log(f"生成: {stats['total_generated']} | Qwen审核: {stats['total_reviewed']} | PASS: {stats['total_pass']}")
    log("Workers: " + " | ".join(f"W{i}={stats.get(f'w{i}_pass', 0)}" for i in range(actual_workers)))
    log(f"全局PASS计数: {_pass_counter}")

    report_path = PROJECT_ROOT / "production_v7_report.json"
    report = {
        "elapsed_minutes": round(elapsed_min, 1),
        "rate_per_min": round(rate, 1),
        "stats": stats,
        "pass_counter": _pass_counter,
        "timestamp": datetime.now().isoformat(),
        "pipeline": "image_prompt_qwen_text_gpt_image_2",
        "tail_fill_mode": TAIL_FILL_MODE,
        "tail_fill_plan": tail_fill_plan,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"Report: {report_path}")


if __name__ == "__main__":
    main()
