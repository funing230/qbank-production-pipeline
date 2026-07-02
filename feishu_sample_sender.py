#!/usr/bin/env python3
"""
监控 feishu_samples/ 目录，发现新的抽样文件就发送到飞书群。
由 hermes cron job 每2分钟执行一次。
stdout 非空时发送（no_agent=True 模式）。
"""
import json
from pathlib import Path

SAMPLE_DIR = Path("/home/flyer8258/research_projects/multimodal_question_bank_24x1000/runs/qwen_gpt_closed_loop_18subjects_24000_20260619_v2/final_questionbank_production_18subjects_24000_v5/feishu_samples")
SENT_FILE = SAMPLE_DIR / ".sent_log"

def main():
    if not SAMPLE_DIR.exists():
        return  # 目录不存在=还没开始生产，静默退出
    
    # 读取已发送记录
    sent = set()
    if SENT_FILE.exists():
        sent = set(SENT_FILE.read_text().strip().split("\n"))
    
    # 扫描新文件
    new_samples = []
    for f in sorted(SAMPLE_DIR.glob("sample_*.json")):
        if f.name not in sent:
            try:
                data = json.loads(f.read_text())
                new_samples.append((f.name, data))
            except Exception:
                pass
    
    if not new_samples:
        return  # 无新抽样，静默退出（不发消息）
    
    # 构造飞书消息
    lines = [f"📋 题库抽样审核 ({len(new_samples)}道题)\n"]
    
    for fname, data in new_samples:
        qj = data.get("question_json", {})
        lines.append(f"━━━━━━━━━━━━━━━━━━")
        lines.append(f"📌 {data.get('kp_id', '?')} | ID: {data.get('question_id', '?')[:12]}")
        lines.append(f"题目: {qj.get('question_text', '?')[:80]}")
        opts = qj.get("options", {})
        for k, v in opts.items():
            lines.append(f"  {k}. {v}")
        lines.append(f"答案: {qj.get('correct_answer', '?')}")
        lines.append(f"解析: {qj.get('explanation', '?')[:100]}")
        lines.append(f"图片: {data.get('image_path', 'N/A')}")
        lines.append("")
    
    lines.append(f"请审核以上题目质量。如有问题请回复。")
    
    # 输出到stdout（hermes cron no_agent模式会发送）
    print("\n".join(lines))
    
    # 记录已发送
    with open(SENT_FILE, "a") as f:
        for fname, _ in new_samples:
            f.write(fname + "\n")


if __name__ == "__main__":
    main()
