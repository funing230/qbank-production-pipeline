#!/usr/bin/env python3
"""
每次执行时检查 production.db 的 FINAL_PASS 数量，
如果达到新的 100 题里程碑，随机抽 3 道输出到 stdout 供 hermes 发送。
如果没有新里程碑，stdout 为空（cron no_agent 模式下不会发消息）。
"""
import sqlite3, json, random
from pathlib import Path

BASE = Path("/home/flyer8258/research_projects/multimodal_question_bank_24x1000/runs/qwen_gpt_closed_loop_18subjects_24000_20260619_v2/final_questionbank_production_18subjects_24000_v5")
DB_PATH = BASE / "production.db"
OUTPUT_DIR = BASE / "output"
STATE_FILE = BASE / "sample_state.json"

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_milestone": 0}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state))

def main():
    if not DB_PATH.exists():
        return  # DB not ready yet
    
    state = load_state()
    conn = sqlite3.connect(str(DB_PATH))
    current = conn.execute("SELECT COUNT(*) FROM questions WHERE quality_status='FINAL_PASS'").fetchone()[0]
    
    last = state["last_milestone"]
    next_milestone = last + 100
    
    if current < next_milestone:
        conn.close()
        return  # 没达到里程碑，静默退出
    
    # 达到里程碑！抽样
    rows = conn.execute(
        """SELECT question_id, kp_id, question_json FROM questions 
           WHERE quality_status='FINAL_PASS' 
           ORDER BY created_at
           LIMIT ? OFFSET ?""",
        (100, last)
    ).fetchall()
    conn.close()
    
    samples = random.sample(rows, min(3, len(rows)))
    
    output_lines = [f"🎲 里程碑 {next_milestone} 题达成！当前总通过: {current} 题\n随机抽样 3 道审核：\n"]
    
    for idx, (qid, kp_id, qj_str) in enumerate(samples, 1):
        qj = json.loads(qj_str) if qj_str else {}
        opts = qj.get("options", {})
        opts_str = " | ".join(f"{k}.{v}" for k, v in opts.items())
        img_rel = qj.get("image_path", "")
        img_full = OUTPUT_DIR / img_rel if img_rel else None
        
        output_lines.append(f"--- 样本 {idx}/3 [{kp_id}] ---")
        output_lines.append(f"题目: {qj.get('question_text', '')}")
        output_lines.append(f"选项: {opts_str}")
        output_lines.append(f"答案: {qj.get('correct_answer', '')} | 难度: {qj.get('difficulty','?')}/5")
        output_lines.append(f"解析: {qj.get('explanation', '')}")
        if img_full and img_full.exists():
            output_lines.append(f"MEDIA:{img_full}")
        output_lines.append("")
    
    # 更新状态
    state["last_milestone"] = next_milestone
    save_state(state)
    
    # 输出到 stdout
    print("\n".join(output_lines))

if __name__ == "__main__":
    main()
