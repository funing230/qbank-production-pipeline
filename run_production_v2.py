#!/usr/bin/env python3
"""
V5 正式生产脚本 - Phase 2 (v2: 并行加速版)
18科24,000道图文题目批量生产。
加速策略:
  1. 2个KP并行生产 (ThreadPoolExecutor)
  2. 每批内图片并行渲染 (4 workers)
  3. 断点续跑：跳过已完成的KP
"""
import sys, json, time, os, signal, traceback, threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.orchestrator import ProductionOrchestrator, Config
from pipeline.db import ProductionDB

# ========== 配置 ==========
READY_SUBJECTS = ["S01", "S02", "S03", "S04", "S05", "S06", "S13"]

# 并行度
KP_CONCURRENCY = 2          # 同时处理2个KP
RENDER_WORKERS = 4          # 渲染并行度
BATCH_SIZE = 8              # 每批题数
REPORT_INTERVAL = 600       # 进度报告间隔(秒)

LOG_FILE = PROJECT_ROOT / "production.log"
_log_lock = threading.Lock()

# ========== 日志 ==========
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with _log_lock:
        print(line, flush=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


# ========== 并行渲染 ==========
def render_batch_parallel(orch, questions, q_ids, subject_id):
    """并行渲染一批题目的图片，返回成功列表"""
    rendered = []
    render_lock = threading.Lock()
    
    def render_one(q, qid):
        engine = q.get("render_engine", "MATPLOTLIB")
        code = q.get("render_code", "")
        img_dir = orch.config.output_dir / subject_id / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        img_path = img_dir / f"{qid}.png"
        
        result = orch.renderer.dispatch(engine, code, str(img_path))
        
        if result["success"]:
            with render_lock:
                orch.db.update_question_status(qid, "RENDERED", f"engine={engine}")
            rel_path = str(img_path.relative_to(orch.config.output_dir))
            q["image_path"] = rel_path
            if isinstance(q.get("question_json"), dict):
                q["question_json"]["image_path"] = rel_path
                with render_lock:
                    orch.db.update_question_json(qid, q["question_json"])
            return (q, qid, str(img_path))
        else:
            with render_lock:
                # RENDER_FAIL not in QUALITY_STATUSES; keep as GENERATED + add render_job for retry
                try:
                    orch.db.add_render_job(qid, engine, code)
                except Exception:
                    pass
            return None
    
    with ThreadPoolExecutor(max_workers=RENDER_WORKERS) as pool:
        futures = {pool.submit(render_one, q, qid): (q, qid) 
                   for q, qid in zip(questions, q_ids)}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    rendered.append(result)
            except Exception as e:
                log(f"    RENDER EXCEPTION: {str(e)[:100]}")
    
    return rendered


# ========== 单KP生产（线程安全） ==========
def produce_kp(orch, kp_id, kp_quota, subject_id, all_kp_details, stats, stats_lock):
    """生产一个KP的所有题目"""
    # 检查已有进度 - 直接SQL计数
    cur = orch.db.conn.execute(
        "SELECT COUNT(*) FROM questions WHERE kp_id = ?", (kp_id,))
    have = cur.fetchone()[0]
    remaining = kp_quota - have
    
    if remaining <= 0:
        return 0
    
    # 获取已有题目用于去重
    cur2 = orch.db.conn.execute(
        "SELECT question_json FROM questions WHERE kp_id = ?", (kp_id,))
    existing_for_kp = []
    for row in cur2.fetchall():
        try:
            existing_for_kp.append(json.loads(row[0]) if isinstance(row[0], str) else row[0])
        except:
            pass
    
    log(f"  [{kp_id}] quota={kp_quota}, have={have}, need={remaining}")
    
    produced = 0
    batch_num = 0
    
    while remaining > 0 and orch.running:
        batch_size = min(remaining, BATCH_SIZE)
        batch_num += 1
        
        try:
            # GPT 生成
            kp_info = orch.get_kp_info(kp_id)
            questions = orch.generator.generate_batch(kp_info, batch_size, existing_for_kp)
            
            if not questions:
                log(f"    [{kp_id}] batch {batch_num} empty, retry...")
                time.sleep(5)
                questions = orch.generator.generate_batch(kp_info, batch_size, existing_for_kp)
                if not questions:
                    log(f"    [{kp_id}] still empty, skip")
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
            batch_db_id = orch.db.create_batch(subject_id, kp_id, len(questions))
            q_ids = orch.db.add_questions(batch_db_id, questions)
            
            # 并行渲染
            rendered = render_batch_parallel(orch, questions, q_ids, subject_id)
            rendered_ok = len(rendered)
            
            with stats_lock:
                stats["total_generated"] += len(questions)
                stats["total_rendered"] += rendered_ok
            
            produced += len(questions)
            remaining -= len(questions)
            existing_for_kp.extend(questions)
            
            log(f"    [{kp_id}] batch {batch_num}: {len(questions)} gen, {rendered_ok} rendered")
            
        except Exception as e:
            log(f"    [{kp_id}] ERROR batch {batch_num}: {str(e)[:200]}")
            traceback.print_exc()
            time.sleep(10)
            continue
    
    return produced


# ========== 主流程 ==========
def main():
    log("=== V5 PRODUCTION PHASE 2 (v2 PARALLEL) START ===")
    log(f"Config: KP_CONCURRENCY={KP_CONCURRENCY}, RENDER_WORKERS={RENDER_WORKERS}, BATCH_SIZE={BATCH_SIZE}")
    
    config = Config()
    
    # 加载所有科目的KP详细信息
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
    
    log(f"Loaded {len(all_kp_details)} KP details from input_snapshot")
    
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
        log(f"Received signal {signum}, stopping...")
        orch.running = False
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # 生产统计
    stats = {
        "start_time": time.time(),
        "total_generated": 0,
        "total_rendered": 0,
        "subjects_done": [],
        "last_report_time": time.time(),
    }
    stats_lock = threading.Lock()
    
    # 获取配额
    quotas = config.quotas.get("kp_quotas", {})
    subject_targets = config.quotas.get("subject_targets", {})
    
    for subject_id in READY_SUBJECTS:
        if not orch.running:
            break
        
        target = subject_targets.get(subject_id, 1333)
        subject_kps = sorted([kp_id for kp_id, info in quotas.items() 
                             if info.get("subject_id") == subject_id])
        
        # 过滤已完成的KP
        pending_kps = []
        for kp_id in subject_kps:
            kp_quota = quotas[kp_id]["production_quota"]
            if kp_quota <= 0:
                continue
            cur = orch.db.conn.execute(
                "SELECT COUNT(*) FROM questions WHERE kp_id = ?", (kp_id,))
            have = cur.fetchone()[0]
            if have < kp_quota:
                pending_kps.append(kp_id)
        
        if not pending_kps:
            log(f"[{subject_id}] All KPs complete, skipping")
            stats["subjects_done"].append(subject_id)
            continue
        
        log(f"\n{'='*50}")
        log(f"[{subject_id}] Starting: {len(pending_kps)} pending KPs (of {len(subject_kps)} total), target={target}")
        log(f"{'='*50}")
        
        subject_produced = 0
        
        # 2个KP并行
        with ThreadPoolExecutor(max_workers=KP_CONCURRENCY) as kp_pool:
            futures = {}
            kp_iter = iter(pending_kps)
            
            # 初始提交
            for _ in range(min(KP_CONCURRENCY, len(pending_kps))):
                kp_id = next(kp_iter, None)
                if kp_id and orch.running:
                    kp_quota = quotas[kp_id]["production_quota"]
                    future = kp_pool.submit(
                        produce_kp, orch, kp_id, kp_quota, 
                        subject_id, all_kp_details, stats, stats_lock
                    )
                    futures[future] = kp_id
            
            # 处理完成的，提交新的
            while futures and orch.running:
                done_futures = []
                for future in list(futures.keys()):
                    if future.done():
                        done_futures.append(future)
                
                for future in done_futures:
                    kp_id = futures.pop(future)
                    try:
                        produced = future.result()
                        subject_produced += produced
                    except Exception as e:
                        log(f"  [{kp_id}] FATAL: {str(e)[:200]}")
                    
                    # 提交下一个
                    next_kp = next(kp_iter, None)
                    if next_kp and orch.running:
                        kp_quota = quotas[next_kp]["production_quota"]
                        new_future = kp_pool.submit(
                            produce_kp, orch, next_kp, kp_quota,
                            subject_id, all_kp_details, stats, stats_lock
                        )
                        futures[new_future] = next_kp
                
                # 进度报告
                now = time.time()
                if now - stats["last_report_time"] > REPORT_INTERVAL:
                    elapsed_min = (now - stats["start_time"]) / 60
                    rate = stats["total_generated"] / max(1, elapsed_min)
                    log(f"\n  --- PROGRESS REPORT ---")
                    log(f"  Elapsed: {elapsed_min:.0f}min | Generated: {stats['total_generated']} | Rate: {rate:.1f}/min")
                    log(f"  Rendered: {stats['total_rendered']} | Current: {subject_id}")
                    log(f"  Active KPs: {list(futures.values())}")
                    log(f"  -------------------------\n")
                    stats["last_report_time"] = now
                
                if not done_futures:
                    time.sleep(1)  # avoid busy-wait
        
        stats["subjects_done"].append(subject_id)
        log(f"[{subject_id}] DONE: {subject_produced} questions produced")
    
    # 最终报告
    elapsed = (time.time() - stats["start_time"]) / 60
    log(f"\n{'='*50}")
    log(f"PRODUCTION COMPLETE")
    log(f"{'='*50}")
    log(f"Elapsed: {elapsed:.0f} minutes")
    log(f"Generated: {stats['total_generated']}")
    log(f"Rendered: {stats['total_rendered']}")
    log(f"Subjects done: {stats['subjects_done']}")
    
    # 写最终状态文件
    final_report = PROJECT_ROOT / "production_final_report.json"
    progress = orch.db.get_overall_progress()
    progress["production_stats"] = stats
    progress["elapsed_minutes"] = elapsed
    final_report.write_text(json.dumps(progress, ensure_ascii=False, indent=2, default=str))
    log(f"Final report: {final_report}")


if __name__ == "__main__":
    main()
