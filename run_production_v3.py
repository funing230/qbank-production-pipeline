#!/usr/bin/env python3
"""
V5 正式生产脚本 - Phase 2 (v3: 渲染并行+大batch)
策略: 单KP串行调GPT(避免endpoint并发timeout) + 渲染并行化 + batch_size=12
预期速度: ~5题/min (vs原版3.3题/min)
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

BATCH_SIZE = 12             # 每批12题 (原8)
RENDER_WORKERS = 4          # 渲染并行度
REPORT_INTERVAL = 600       # 进度报告间隔(秒)
API_COOLDOWN = 2            # GPT调用间隔(秒)，避免429

LOG_FILE = PROJECT_ROOT / "production_v3.log"

# ========== 日志 ==========
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ========== 并行渲染 ==========
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
            orch.db.update_question_status(qid, "RENDERED", f"engine={engine}")
            rel_path = str(img_path.relative_to(orch.config.output_dir))
            q["image_path"] = rel_path
            if isinstance(q.get("question_json"), dict):
                q["question_json"]["image_path"] = rel_path
                orch.db.update_question_json(qid, q["question_json"])
            return (q, qid, str(img_path))
        else:
            # Keep as GENERATED; add render_job for retry
            try:
                orch.db.add_render_job(qid, engine, code)
            except Exception:
                pass
            return None
    
    # 串行渲染但用ThreadPool并行（SQLite WAL模式线程安全）
    with ThreadPoolExecutor(max_workers=RENDER_WORKERS) as pool:
        futures = {pool.submit(render_one, q, qid): qid 
                   for q, qid in zip(questions, q_ids)}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    rendered.append(result)
            except Exception as e:
                log(f"    RENDER ERR: {str(e)[:100]}")
    
    return rendered


# ========== 主流程 ==========
def main():
    log("=== V5 PRODUCTION (v3: serial GPT + parallel render + batch12) ===")
    log(f"Config: BATCH_SIZE={BATCH_SIZE}, RENDER_WORKERS={RENDER_WORKERS}, API_COOLDOWN={API_COOLDOWN}s")
    
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
    
    # 初始化
    orch = ProductionOrchestrator(config)
    
    orig_get = orch.get_kp_info
    def patched_get(kp_id):
        base = orig_get(kp_id)
        if kp_id in all_kp_details:
            base.update(all_kp_details[kp_id])
        return base
    orch.get_kp_info = patched_get
    
    # 优雅退出
    def signal_handler(signum, frame):
        log(f"Signal {signum}, stopping gracefully...")
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
        "errors": 0,
        "timeouts": 0,
    }
    
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
            cur = orch.db.conn.execute(
                "SELECT COUNT(*) FROM questions WHERE kp_id = ?", (kp_id,))
            have = cur.fetchone()[0]
            if have < kp_quota:
                pending_kps.append((kp_id, kp_quota, have))
        
        if not pending_kps:
            log(f"[{subject_id}] Complete, skipping")
            stats["subjects_done"].append(subject_id)
            continue
        
        total_pending = sum(quota - have for _, quota, have in pending_kps)
        log(f"\n{'='*50}")
        log(f"[{subject_id}] {len(pending_kps)} pending KPs, ~{total_pending} questions remaining")
        log(f"{'='*50}")
        
        subject_produced = 0
        
        for kp_id, kp_quota, have in pending_kps:
            if not orch.running:
                break
            
            remaining = kp_quota - have
            
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
            
            batch_num = 0
            while remaining > 0 and orch.running:
                batch_size = min(remaining, BATCH_SIZE)
                batch_num += 1
                
                try:
                    # GPT生成
                    kp_info = orch.get_kp_info(kp_id)
                    questions = orch.generator.generate_batch(kp_info, batch_size, existing_for_kp)
                    
                    if not questions:
                        log(f"    [{kp_id}] batch {batch_num} empty, retry...")
                        time.sleep(5)
                        questions = orch.generator.generate_batch(kp_info, batch_size, existing_for_kp)
                        if not questions:
                            log(f"    [{kp_id}] still empty, skip KP")
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
                    
                    stats["total_generated"] += len(questions)
                    stats["total_rendered"] += rendered_ok
                    subject_produced += len(questions)
                    remaining -= len(questions)
                    existing_for_kp.extend(questions)
                    
                    log(f"    batch {batch_num}: {len(questions)} gen, {rendered_ok}/{len(questions)} rendered")
                    
                    # API cooldown
                    time.sleep(API_COOLDOWN)
                    
                except Exception as e:
                    err_msg = str(e)[:200]
                    if "timeout" in err_msg.lower() or "Timeout" in err_msg:
                        stats["timeouts"] += 1
                        log(f"    [{kp_id}] TIMEOUT batch {batch_num}, wait 30s...")
                        time.sleep(30)
                    elif "429" in err_msg:
                        stats["errors"] += 1
                        log(f"    [{kp_id}] RATE LIMITED, wait 60s...")
                        time.sleep(60)
                    else:
                        stats["errors"] += 1
                        log(f"    [{kp_id}] ERROR batch {batch_num}: {err_msg}")
                        traceback.print_exc()
                        time.sleep(10)
                    continue
                
                # 定期进度报告
                now = time.time()
                if now - stats["last_report_time"] > REPORT_INTERVAL:
                    elapsed_min = (now - stats["start_time"]) / 60
                    rate = stats["total_generated"] / max(1, elapsed_min)
                    log(f"\n  === PROGRESS ===")
                    log(f"  Elapsed: {elapsed_min:.0f}min | Generated: {stats['total_generated']} | Rate: {rate:.1f}/min")
                    log(f"  Rendered: {stats['total_rendered']} | Timeouts: {stats['timeouts']} | Errors: {stats['errors']}")
                    log(f"  Current: {subject_id}/{kp_id} | Subjects done: {stats['subjects_done']}")
                    remaining_total = sum(subject_targets.get(s, 1333) for s in READY_SUBJECTS) - stats['total_generated'] - 155
                    if rate > 0:
                        log(f"  ETA: {remaining_total/rate/60:.1f}h")
                    log(f"  ===============\n")
                    stats["last_report_time"] = now
        
        stats["subjects_done"].append(subject_id)
        log(f"[{subject_id}] DONE: +{subject_produced} questions")
    
    # 最终报告
    elapsed = (time.time() - stats["start_time"]) / 60
    log(f"\n{'='*50}")
    log(f"PRODUCTION COMPLETE")
    log(f"{'='*50}")
    log(f"Elapsed: {elapsed:.0f} min | Generated: {stats['total_generated']} | Rendered: {stats['total_rendered']}")
    log(f"Timeouts: {stats['timeouts']} | Errors: {stats['errors']}")
    log(f"Subjects: {stats['subjects_done']}")
    
    final_report = PROJECT_ROOT / "production_final_report.json"
    progress = orch.db.get_overall_progress()
    progress["production_stats"] = stats
    progress["elapsed_minutes"] = elapsed
    final_report.write_text(json.dumps(progress, ensure_ascii=False, indent=2, default=str))
    log(f"Report: {final_report}")


if __name__ == "__main__":
    main()
