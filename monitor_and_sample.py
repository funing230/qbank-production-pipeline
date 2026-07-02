"""
监控 production.db，每新增 100 道 FINAL_PASS 题，
随机抽 3 道发送到飞书群审核。
"""
import sqlite3, json, random, time, subprocess, sys
from pathlib import Path

DB_PATH = Path(__file__).parent / "production.db"
OUTPUT_DIR = Path(__file__).parent / "output"
SAMPLE_LOG = Path(__file__).parent / "sample_sent.json"

# 记录已经发过的里程碑
def load_sent():
    if SAMPLE_LOG.exists():
        return json.loads(SAMPLE_LOG.read_text())
    return {"last_milestone": 0, "sent_ids": []}

def save_sent(data):
    SAMPLE_LOG.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def get_pass_count():
    conn = sqlite3.connect(str(DB_PATH))
    count = conn.execute("SELECT COUNT(*) FROM questions WHERE quality_status='FINAL_PASS'").fetchone()[0]
    conn.close()
    return count

def get_random_samples(milestone_start, milestone_end, n=3):
    """从 milestone_start+1 到 milestone_end 范围内随机抽 n 道"""
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        """SELECT question_id, kp_id, question_json FROM questions 
           WHERE quality_status='FINAL_PASS' 
           ORDER BY created_at
           LIMIT ? OFFSET ?""",
        (milestone_end - milestone_start, milestone_start)
    ).fetchall()
    conn.close()
    
    if len(rows) <= n:
        return rows
    return random.sample(rows, n)

def send_to_feishu(qid, kp_id, qj, milestone, sample_idx):
    """通过写文件到 feishu_queue 的方式不行，直接 print 让外层处理"""
    opts = qj.get("options", {})
    opts_str = "  ".join(f"{k}. {v}" for k, v in opts.items())
    
    img_path = OUTPUT_DIR / qj.get("image_path", "")
    img_exists = img_path.exists()
    
    msg = f"🎲 抽样审核 [里程碑 {milestone}题, 样本{sample_idx}] {kp_id}\n\n"
    msg += f"题目：{qj.get('question_text', '')}\n\n"
    msg += f"选项：\n{opts_str}\n\n"
    msg += f"正确答案：{qj.get('correct_answer', '')}\n\n"
    msg += f"解析：{qj.get('explanation', '')}\n\n"
    msg += f"难度：{qj.get('difficulty', '?')}/5"
    
    if img_exists:
        msg += f"\n\nMEDIA:{img_path}"
    
    print(f"SEND|{msg}", flush=True)
    return True

def main():
    print(f"[Monitor] Started. Checking every 30s. DB: {DB_PATH}", flush=True)
    state = load_sent()
    
    while True:
        try:
            current = get_pass_count()
            last = state["last_milestone"]
            next_milestone = last + 100
            
            if current >= next_milestone:
                # 达到新里程碑
                print(f"[Monitor] Milestone reached: {next_milestone} (current={current})", flush=True)
                samples = get_random_samples(last, next_milestone, n=3)
                
                for idx, (qid, kp_id, qj_str) in enumerate(samples, 1):
                    qj = json.loads(qj_str) if qj_str else {}
                    send_to_feishu(qid, kp_id, qj, next_milestone, idx)
                    state["sent_ids"].append(qid)
                
                state["last_milestone"] = next_milestone
                save_sent(state)
                print(f"[Monitor] Samples sent for milestone {next_milestone}", flush=True)
            else:
                print(f"[Monitor] {current}/{next_milestone} FINAL_PASS (need {next_milestone - current} more)", flush=True)
        except Exception as e:
            print(f"[Monitor] Error: {e}", flush=True)
        
        time.sleep(30)

if __name__ == "__main__":
    main()
