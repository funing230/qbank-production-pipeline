#!/usr/bin/env python3
"""
V5 正式生产脚本 - Phase 2 (v4: 5路并行)
策略: 5个独立API_KEY × 5路KP并行生产 + 渲染并行
预期速度: 12-15题/min (原3.5题/min, 提升4x)
"""
import sys, json, time, os, signal, traceback, threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.orchestrator import ProductionOrchestrator, Config
from pipeline.generator import QuestionGenerator
from pipeline.db import ProductionDB

# ========== 配置 ==========
READY_SUBJECTS = ["S01", "S02", "S03", "S04", "S05", "S06", "S13"]

KP_CONCURRENCY = 5          # 5路KP并行（9key测试全通，保守用5）
BATCH_SIZE = 12             # 每批12题
RENDER_WORKERS = 6          # 渲染并行度
REPORT_INTERVAL = 600       # 进度报告间隔(秒)
API_COOLDOWN = 3            # 每批后间隔（防502）

LOG_FILE = PROJECT_ROOT / "production_v4.log"
_log_lock = threading.Lock()
_db_lock = threading.Lock()  # 保护所有SQLite操作

# ========== 9路API密钥 ==========
GPT_KEYS = [
    os.environ["GPT5_API_KEY"],
    os.environ["GPT_WORKER1_API_KEY"],
    os.environ["GPT_WORKER2_API_KEY"],
    os.environ["GPT_WORKER3_API_KEY"],
    os.environ["GPT_WORKER4_API_KEY"],
    os.environ["GPT_WORKER5_API_KEY"],
    os.environ["GPT_WORKER6_API_KEY"],
    os.environ["GPT_WORKER7_API_KEY"],
    os.environ["GPT_WORKER8_API_KEY"],
]
GPT_BASE_URL = os.environ["GPT5_BASE_URL"]
GPT_MODEL = "gpt-5.5-openai-compact"

# ========== 日志 ==========
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with _log_lock:
        print(line, flush=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


# ========== 并行渲染 ==========
_render_pool = ThreadPoolExecutor(max_workers=RENDER_WORKERS)

def render_batch_parallel(orch, questions, q_ids, subject_id):
    """并行渲染一批题目的图片"""
    rendered = []
    
    def render_one(q, qid):
        engine = q.get("render_engine", "MATPLOTLIB")
        code = q.get("render_code", "")
        img_dir = orch.config.output_dir / subject_id / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        img_path = img_dir / f"{qid}.png"
        
        result = orch.renderer.dispatch(engine, code, str(img_path))
        
        if result["success"]:
            with _db_lock:
                orch.db.update_question_status(qid, "RENDERED", f"engine={engine}")
            rel_path = str(img_path.relative_to(orch.config.output_dir))
            q["image_path"] = rel_path
            if isinstance(q.get("question_json"), dict):
                q["question_json"]["image_path"] = rel_path
                with _db_lock:
                    orch.db.update_question_json(qid, q["question_json"])
            return (q, qid, str(img_path))
        else:
            try:
                with _db_lock:
                    orch.db.add_render_job(qid, engine, code)
            except Exception:
                pass
            return None
    
    futures = {_render_pool.submit(render_one, q, qid): qid 
               for q, qid in zip(questions, q_ids)}
    for future in as_completed(futures):
        try:
            result = future.result()
            if result:
                rendered.append(result)
        except Exception as e:
            log(f"    RENDER ERR: {str(e)[:80]}")
    
    return rendered


# ========== 单KP生产（由特定worker执行） ==========
def produce_kp(orch, generator, worker_id, kp_id, kp_quota, subject_id, stats, stats_lock):
    """用指定的generator实例生产一个KP的所有题目"""
    # 检查已有进度
    with _db_lock:
        cur = orch.db.conn.execute(
            "SELECT COUNT(*) FROM questions WHERE kp_id = ?", (kp_id,))
        have = cur.fetchone()[0]
    remaining = kp_quota - have
    
    if remaining <= 0:
        return 0
    
    # 获取已有题目用于去重
    with _db_lock:
        cur2 = orch.db.conn.execute(
            "SELECT question_json FROM questions WHERE kp_id = ?", (kp_id,))
        rows = cur2.fetchall()
    existing_for_kp = []
    for row in rows:
        try:
            existing_for_kp.append(json.loads(row[0]) if isinstance(row[0], str) else row[0])
        except:
            pass
    
    log(f"  [W{worker_id}][{kp_id}] quota={kp_quota}, have={have}, need={remaining}")
    
    produced = 0
    batch_num = 0
    
    while remaining > 0 and orch.running:
        batch_size = min(remaining, BATCH_SIZE)
        batch_num += 1
        
        try:
            # GPT 生成 (使用自己的generator实例 = 自己的API key)
            kp_info = orch.get_kp_info(kp_id)
            questions = generator.generate_batch(kp_info, batch_size, existing_for_kp)
            
            if not questions:
                log(f"  [W{worker_id}][{kp_id}] batch {batch_num} empty, retry...")
                time.sleep(5)
                questions = generator.generate_batch(kp_info, batch_size, existing_for_kp)
                if not questions:
                    log(f"  [W{worker_id}][{kp_id}] still empty, skip")
                    break
            
            # 补充字段
            import uuid
            for q in questions:
                q.setdefault("subject_id", subject_id)
                q.setdefault("module_id", kp_id.rsplit("-", 1)[0] if "-" in kp_id else "")
                q.setdefault("kp_id", kp_id)
                q.setdefault("kp_name", kp_info.get("knowledge_point_name", ""))
                if not q.get("question_id"):
                    q["question_id"] = f"{kp_id}-Q{uuid.uuid4().hex[:6]}"
                delivery_fields = {
                    "question_id": q.get("question_id", ""),
                    "subject_id": subject_id,
                    "kp_id": kp_id,
                    "kp_name": kp_info.get("knowledge_point_name", ""),
                    "question_text": q.get("question_text", ""),
                    "options": q.get("options", {}),
                    "correct_answer": q.get("correct_answer", ""),
                    "explanation": q.get("explanation", ""),
                    "difficulty": q.get("difficulty", 0),
                    "image_description": q.get("image_description", ""),
                    "image_path": "",
                    "render_engine": q.get("render_engine", ""),
                }
                q["question_json"] = delivery_fields
            
            # 入库
            with _db_lock:
                batch_db_id = orch.db.create_batch(subject_id, kp_id, len(questions))
                q_ids = orch.db.add_questions(batch_db_id, questions)
            
            # 并行渲染
            rendered = render_batch_parallel(orch, questions, q_ids, subject_id)
            rendered_ok = len(rendered)
            
            with stats_lock:
                stats["total_generated"] += len(questions)
                stats["total_rendered"] += rendered_ok
                stats[f"worker_{worker_id}_count"] = stats.get(f"worker_{worker_id}_count", 0) + len(questions)
            
            produced += len(questions)
            remaining -= len(questions)
            existing_for_kp.extend(questions)
            
            log(f"  [W{worker_id}][{kp_id}] batch {batch_num}: {len(questions)} gen, {rendered_ok} rendered")
            
            # 短暂间隔
            time.sleep(API_COOLDOWN)
            
        except Exception as e:
            err_msg = str(e)[:200]
            if "timeout" in err_msg.lower():
                with stats_lock:
                    stats["timeouts"] = stats.get("timeouts", 0) + 1
                log(f"  [W{worker_id}][{kp_id}] TIMEOUT batch {batch_num}, wait 30s...")
                time.sleep(30)
            elif "429" in err_msg or "502" in err_msg or "forbidden" in err_msg.lower():
                with stats_lock:
                    stats["rate_limits"] = stats.get("rate_limits", 0) + 1
                log(f"  [W{worker_id}][{kp_id}] RATE LIMITED/502, wait 45s...")
                time.sleep(45)
            else:
                with stats_lock:
                    stats["errors"] = stats.get("errors", 0) + 1
                log(f"  [W{worker_id}][{kp_id}] ERROR: {err_msg}")
                traceback.print_exc()
                time.sleep(10)
            continue
    
    return produced


# ========== 主流程 ==========
def main():
    log("=== V5 PRODUCTION (v4: 5-WAY PARALLEL) ===")
    log(f"Config: KP_CONCURRENCY={KP_CONCURRENCY}, BATCH={BATCH_SIZE}, RENDER_WORKERS={RENDER_WORKERS}")
    log(f"Workers: main + worker1-4 = 5 API keys")
    
    config = Config()
    
    # 加载KP详情
    input_dir = PROJECT_ROOT.parent / "input_snapshot"
    all_kp_details = {}
    for f in sorted(input_dir.glob("*.json")):
        data = json.loads(f.read_text())
        if "modules" in data:
            for mod in data["modules"]:
                for kp in mod.get("knowledge_points", []):
                    all_kp_details[kp["knowledge_point_id"]] = kp
        elif "knowledge_points" in data:
            for kp in data["knowledge_points"]:
                all_kp_details[kp["knowledge_point_id"]] = kp
    
    log(f"Loaded {len(all_kp_details)} KP details")
    
    # 创建5个Generator实例（各自独立的API key）
    generators = []
    for i, key in enumerate(GPT_KEYS):
        gen = QuestionGenerator(
            base_url=GPT_BASE_URL,
            api_key=key,
            model=GPT_MODEL,
            max_concurrent=1,
            response_log_dir=str(PROJECT_ROOT / f"api_responses_w{i}"),
        )
        generators.append(gen)
        log(f"  Generator W{i} initialized (key=...{key[-4:]})")
    
    # 初始化orchestrator
    orch = ProductionOrchestrator(config)
    
    # Patch get_kp_info
    orig_get = orch.get_kp_info
    def patched_get(kp_id):
        base = orig_get(kp_id)
        if kp_id in all_kp_details:
            base.update(all_kp_details[kp_id])
        return base
    orch.get_kp_info = patched_get
    
    # 优雅退出
    def signal_handler(signum, frame):
        log(f"Signal {signum}, stopping all workers...")
        orch.running = False
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # 统计
    stats = {
        "start_time": time.time(),
        "total_generated": 0,
        "total_rendered": 0,
        "subjects_done": [],
        "last_report_time": time.time(),
        "timeouts": 0,
        "rate_limits": 0,
        "errors": 0,
    }
    stats_lock = threading.Lock()
    
    quotas = config.quotas.get("kp_quotas", {})
    subject_targets = config.quotas.get("subject_targets", {})
    
    for subject_id in READY_SUBJECTS:
        if not orch.running:
            break
        
        target = subject_targets.get(subject_id, 1333)
        subject_kps = sorted([kp_id for kp_id, info in quotas.items() 
                             if info.get("subject_id") == subject_id])
        
        # 过滤已完成KP
        pending_kps = []
        for kp_id in subject_kps:
            kp_quota = quotas[kp_id]["production_quota"]
            if kp_quota <= 0:
                continue
            with _db_lock:
                cur = orch.db.conn.execute(
                    "SELECT COUNT(*) FROM questions WHERE kp_id = ?", (kp_id,))
                have = cur.fetchone()[0]
            if have < kp_quota:
                pending_kps.append((kp_id, kp_quota))
        
        if not pending_kps:
            log(f"[{subject_id}] Complete, skipping")
            stats["subjects_done"].append(subject_id)
            continue
        
        total_remaining = 0
        with _db_lock:
            for k, q in pending_kps:
                cur = orch.db.conn.execute(
                    "SELECT COUNT(*) FROM questions WHERE kp_id = ?", (k,))
                total_remaining += q - cur.fetchone()[0]
        log(f"\n{'='*50}")
        log(f"[{subject_id}] {len(pending_kps)} pending KPs, ~{total_remaining} questions remaining")
        log(f"{'='*50}")
        
        subject_produced = 0
        
        # 5路KP并行
        with ThreadPoolExecutor(max_workers=KP_CONCURRENCY) as kp_pool:
            futures = {}
            kp_iter = iter(pending_kps)
            worker_cycle = 0  # 轮转分配worker
            
            # 初始提交（最多KP_CONCURRENCY个）
            for _ in range(min(KP_CONCURRENCY, len(pending_kps))):
                item = next(kp_iter, None)
                if item and orch.running:
                    kp_id, kp_quota = item
                    wid = worker_cycle % len(generators)
                    worker_cycle += 1
                    future = kp_pool.submit(
                        produce_kp, orch, generators[wid], wid,
                        kp_id, kp_quota, subject_id, stats, stats_lock
                    )
                    futures[future] = (kp_id, wid)
            
            # 完成一个提交下一个
            while futures and orch.running:
                done_futures = [f for f in futures if f.done()]
                
                for future in done_futures:
                    kp_id, wid = futures.pop(future)
                    try:
                        produced = future.result()
                        subject_produced += produced
                    except Exception as e:
                        log(f"  [W{wid}][{kp_id}] FATAL: {str(e)[:150]}")
                    
                    # 提交下一个KP
                    next_item = next(kp_iter, None)
                    if next_item and orch.running:
                        next_kp, next_quota = next_item
                        next_wid = worker_cycle % len(generators)
                        worker_cycle += 1
                        new_future = kp_pool.submit(
                            produce_kp, orch, generators[next_wid], next_wid,
                            next_kp, next_quota, subject_id, stats, stats_lock
                        )
                        futures[new_future] = (next_kp, next_wid)
                
                # 进度报告
                now = time.time()
                if now - stats["last_report_time"] > REPORT_INTERVAL:
                    elapsed_min = (now - stats["start_time"]) / 60
                    rate = stats["total_generated"] / max(1, elapsed_min)
                    remaining_all = 24000 - 197 - stats["total_generated"]
                    eta_h = remaining_all / max(1, rate) / 60
                    log(f"\n  ╔══ PROGRESS REPORT ══╗")
                    log(f"  ║ Elapsed: {elapsed_min:.0f}min | Rate: {rate:.1f}/min")
                    log(f"  ║ Generated: {stats['total_generated']} | Rendered: {stats['total_rendered']}")
                    log(f"  ║ Timeouts: {stats['timeouts']} | 429s: {stats['rate_limits']} | Errors: {stats['errors']}")
                    log(f"  ║ Workers: " + " | ".join(f"W{i}={stats.get(f'worker_{i}_count',0)}" for i in range(5)))
                    log(f"  ║ Subject: {subject_id} | Done: {stats['subjects_done']}")
                    log(f"  ║ ETA: {eta_h:.1f}h")
                    log(f"  ╚═══════════════════════╝\n")
                    stats["last_report_time"] = now
                
                if not done_futures:
                    time.sleep(0.5)
        
        stats["subjects_done"].append(subject_id)
        log(f"[{subject_id}] DONE: +{subject_produced} questions")
    
    # 最终报告
    elapsed = (time.time() - stats["start_time"]) / 60
    rate = stats["total_generated"] / max(1, elapsed)
    log(f"\n{'='*50}")
    log(f"PRODUCTION COMPLETE")
    log(f"{'='*50}")
    log(f"Elapsed: {elapsed:.0f}min | Rate: {rate:.1f}/min")
    log(f"Generated: {stats['total_generated']} | Rendered: {stats['total_rendered']}")
    log(f"Timeouts: {stats['timeouts']} | 429s: {stats['rate_limits']} | Errors: {stats['errors']}")
    log(f"Workers: " + " | ".join(f"W{i}={stats.get(f'worker_{i}_count',0)}" for i in range(5)))
    log(f"Subjects: {stats['subjects_done']}")
    
    final_report = PROJECT_ROOT / "production_final_report.json"
    progress = orch.db.get_overall_progress()
    progress["production_stats"] = stats
    progress["elapsed_minutes"] = elapsed
    final_report.write_text(json.dumps(progress, ensure_ascii=False, indent=2, default=str))
    log(f"Report: {final_report}")


if __name__ == "__main__":
    main()
