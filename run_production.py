#!/usr/bin/env python3
"""
V5 正式生产脚本 - Phase 2
18科24,000道图文题目批量生产。
先跑已就绪的7科，后续科目等千问审核完成后继续。
"""
import sys, json, time, os, signal, traceback
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.orchestrator import ProductionOrchestrator, Config
from pipeline.db import ProductionDB

# ========== 配置 ==========
# 已通过千问审核的7科（≥95% KP通过率）
READY_SUBJECTS = ["S01", "S02", "S03", "S04", "S05", "S06", "S13"]

# 进度报告间隔（秒）
REPORT_INTERVAL = 600  # 每10分钟

LOG_FILE = PROJECT_ROOT / "production.log"

# ========== 日志 ==========
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ========== 主流程 ==========
def main():
    log("=== V5 PRODUCTION PHASE 2 START ===")
    
    config = Config()
    
    # 加载所有科目的KP详细信息
    input_dir = PROJECT_ROOT.parent / "input_snapshot"
    all_kp_details = {}
    for f in sorted(input_dir.glob("*.json")):
        data = json.loads(f.read_text())
        sid = data.get("subject_id", f.stem.upper())
        # 两种格式
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
    
    # Patch get_kp_info to include full KP details
    orig_get = orch.get_kp_info
    def patched_get(kp_id):
        base = orig_get(kp_id)
        if kp_id in all_kp_details:
            base.update(all_kp_details[kp_id])
        return base
    orch.get_kp_info = patched_get
    
    # 生产统计
    stats = {
        "start_time": time.time(),
        "total_generated": 0,
        "total_rendered": 0,
        "total_sentinel_pass": 0,
        "total_sentinel_fail": 0,
        "total_render_fail": 0,
        "subjects_done": [],
        "last_report_time": time.time(),
    }
    
    # 获取配额
    quotas = config.quotas.get("kp_quotas", {})
    subject_targets = config.quotas.get("subject_targets", {})
    
    for subject_id in READY_SUBJECTS:
        if not orch.running:
            break
        
        target = subject_targets.get(subject_id, 1333)
        subject_kps = sorted([kp_id for kp_id, info in quotas.items() 
                             if info.get("subject_id") == subject_id])
        
        log(f"\n{'='*50}")
        log(f"[{subject_id}] Starting: {len(subject_kps)} KPs, target={target} questions")
        log(f"{'='*50}")
        
        subject_produced = 0
        
        for kp_id in subject_kps:
            if not orch.running:
                break
            
            kp_quota = quotas[kp_id]["production_quota"]
            if kp_quota <= 0:
                continue
            
            # 检查已有进度
            existing = orch.db.get_questions_by_status(None, subject_id)
            existing_for_kp = [q for q in existing if q.get("kp_id") == kp_id]
            remaining = kp_quota - len(existing_for_kp)
            
            if remaining <= 0:
                continue
            
            log(f"  [{kp_id}] quota={kp_quota}, have={len(existing_for_kp)}, need={remaining}")
            
            # 分批生产
            batch_num = 0
            while remaining > 0 and orch.running:
                batch_size = min(remaining, 8)  # 默认8题/批
                batch_num += 1
                
                try:
                    results = orch.produce_batch(kp_id, batch_size, existing_for_kp)
                    
                    if not results:
                        log(f"    [{kp_id}] batch {batch_num} empty, retry once...")
                        time.sleep(5)
                        results = orch.produce_batch(kp_id, batch_size, existing_for_kp)
                        if not results:
                            log(f"    [{kp_id}] still empty, moving to next KP")
                            break
                    
                    rendered_ok = sum(1 for _, _, p in results if Path(p).exists())
                    stats["total_generated"] += len(results)
                    stats["total_rendered"] += rendered_ok
                    subject_produced += len(results)
                    remaining -= len(results)
                    
                    log(f"    batch {batch_num}: {len(results)} generated, {rendered_ok} rendered OK")
                    
                except Exception as e:
                    log(f"    ERROR in batch {batch_num}: {str(e)[:200]}")
                    traceback.print_exc()
                    time.sleep(10)
                    continue
                
                # 定期进度报告
                now = time.time()
                if now - stats["last_report_time"] > REPORT_INTERVAL:
                    elapsed_min = (now - stats["start_time"]) / 60
                    rate = stats["total_generated"] / max(1, elapsed_min)
                    log(f"\n  --- PROGRESS REPORT ---")
                    log(f"  Elapsed: {elapsed_min:.0f}min | Generated: {stats['total_generated']} | Rate: {rate:.1f}/min")
                    log(f"  Rendered: {stats['total_rendered']} | Current: {subject_id}")
                    log(f"  -------------------------\n")
                    stats["last_report_time"] = now
        
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
    final_report.write_text(json.dumps(progress, ensure_ascii=False, indent=2))
    log(f"Final report: {final_report}")


if __name__ == "__main__":
    main()
