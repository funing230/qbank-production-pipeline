#!/usr/bin/env python3
"""
试跑5KP + 每题实时发飞书审核。
每生产一道题（渲染成功后），立刻把图片和JSON内容发到飞书群。
"""
import sys, json, time, os, threading, traceback
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.orchestrator import ProductionOrchestrator, Config
from pipeline.db import ProductionDB

# ========== 配置 ==========
PILOT_KPS = [
    "S01-M01-001",
    "S02-M01-001",
    "S03-M01-001",
    "S04-M01-001",
    "S05-M01-001",
]

BATCH_SIZE = 8
LOG_FILE = PROJECT_ROOT / "pilot_5kp.log"
# 飞书发送记录（避免重复发送）
SENT_FILE = PROJECT_ROOT / "pilot_sent.json"

_log_lock = threading.Lock()


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with _log_lock:
        print(line, flush=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


def load_sent():
    if SENT_FILE.exists():
        return set(json.loads(SENT_FILE.read_text()))
    return set()


def save_sent(sent_set):
    SENT_FILE.write_text(json.dumps(list(sent_set), ensure_ascii=False))


def send_to_feishu(question_id, subject_id, question_json, image_path):
    """把一道题的图片+内容写入待发送目录，由外部监控进程发送"""
    # 写入 feishu_queue 目录
    queue_dir = PROJECT_ROOT / "feishu_queue"
    queue_dir.mkdir(exist_ok=True)
    
    item = {
        "question_id": question_id,
        "subject_id": subject_id,
        "image_path": str(image_path),
        "question_json": question_json,
        "timestamp": datetime.now().isoformat(),
    }
    
    out_file = queue_dir / f"{question_id}.json"
    out_file.write_text(json.dumps(item, ensure_ascii=False, indent=2))
    log(f"  → Queued for Feishu: {question_id}")


def main():
    log("=" * 50)
    log("PILOT RUN: 5 KPs + Feishu live review")
    log("=" * 50)

    config = Config()
    # 单线程模式：逐题生产，方便审核
    config.gpt_concurrency = 1
    config.qwen_concurrency = 1
    config.render_concurrency = 1
    # 试跑阶段：关闭Qwen哨兵审核，由人工审核
    config.sentinel_sample_rate = 1.0  # 100%全量哨兵审核，启用冻结的重生成闭环
    config.sentinel_high_risk_rate = 0.0
    orch = ProductionOrchestrator(config)
    quotas = config.quotas.get("kp_quotas", {})
    
    sent = load_sent()
    total_new = 0

    for kp_id in PILOT_KPS:
        if not orch.running:
            break

        kp_info = quotas.get(kp_id, {})
        subject_id = kp_info.get("subject_id", kp_id.split("-M")[0])
        target = kp_info.get("production_quota", 10)

        have = orch.db.count_questions_for_kp(kp_id)
        need = target - have
        
        if need <= 0:
            log(f"[{kp_id}] Already complete ({have}/{target})")
            continue

        log(f"\n[{kp_id}] ({subject_id}) need={need}, target={target}")

        produced_this_kp = 0
        consecutive_errors = 0
        while produced_this_kp < need and orch.running:
            if consecutive_errors >= 3:
                log(f"  ⚠ 连续{consecutive_errors}次错误，跳过此KP")
                break
            batch_size = min(BATCH_SIZE, need - produced_this_kp)
            log(f"  Generating batch of {batch_size}...")

            try:
                batch_result = orch.produce_batch(kp_id, batch_size, [])
                
                if not batch_result:
                    log(f"  Batch returned empty")
                    consecutive_errors += 1
                    time.sleep(10)
                    continue
                
                for item in batch_result:
                    if isinstance(item, tuple) and len(item) >= 3:
                        q, qid, img_path = item
                    elif isinstance(item, tuple) and len(item) >= 2:
                        q, qid = item
                        img_path = q.get("image_path", "")
                    else:
                        continue
                    
                    produced_this_kp += 1
                    total_new += 1
                    consecutive_errors = 0  # 成功了，重置计数
                    
                    # 发到飞书队列
                    if qid not in sent:
                        send_to_feishu(qid, subject_id, q, img_path)
                        sent.add(qid)
                        save_sent(sent)
                        
            except Exception as e:
                log(f"  ERROR: {str(e)[:300]}")
                traceback.print_exc()
                consecutive_errors += 1
                time.sleep(10)
                # 502/429 等临时错误：重试当前batch，不break
                continue

        log(f"  [{kp_id}] produced {produced_this_kp} this run")

    log(f"\n{'=' * 50}")
    log(f"PILOT COMPLETE: {total_new} new questions")
    log(f"Check feishu_queue/ for items to send")
    log(f"{'=' * 50}")


if __name__ == "__main__":
    main()
