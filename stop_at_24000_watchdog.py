#!/usr/bin/env python3
"""硬性终止看门狗: FINAL_PASS >= TARGET 立刻停止所有生产流程。

设计 (no_agent cron, 每1分钟):
- 未达标 -> 空输出 (静默, 不打扰)
- 达标   -> 写 stop 标志 + kill 生产进程 + 落触发标记, 输出终止报告 (投递飞书群)
- 已触发过 -> 静默退出 (幂等, 不重复 kill/报告)
"""
import os, sqlite3, subprocess, json, datetime, sys

ROOT = "/home/flyer8258/research_projects/multimodal_question_bank_24x1000/runs/qwen_gpt_closed_loop_18subjects_24000_20260619_v2/final_questionbank_production_18subjects_24000_v5"
DB = os.path.join(ROOT, "production.db")
CONTROL = os.path.join(ROOT, "runtime_control.json")
TRIGGER_MARK = os.path.join(ROOT, "STOP_AT_24000.TRIGGERED")
TARGET = 24000  # FINAL_PASS >= 此值立即全停

def final_pass_count():
    c = sqlite3.connect(DB); cur = c.cursor()
    cur.execute("SELECT count(*) FROM questions WHERE quality_status='FINAL_PASS'")
    n = cur.fetchone()[0]; c.close()
    return n

def main():
    # 已触发过 -> 幂等静默
    if os.path.exists(TRIGGER_MARK):
        sys.exit(0)

    try:
        fp = final_pass_count()
    except Exception as e:
        # DB 暂时锁住等下一分钟, 不报错刷屏
        sys.exit(0)

    if fp < TARGET:
        # 未达标: 静默
        sys.exit(0)

    # ===== 达标: 硬性全停 =====
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    actions = []

    # 1) 写 stop 标志 (进程热加载后会自停, 双保险)
    try:
        with open(CONTROL, "w") as f:
            json.dump({"stop": True, "drain_only": True, "gpt_enabled": False,
                       "qwen_enabled": False, "image_enabled": False, "regen_enabled": False}, f)
        actions.append("✅ 写入 stop 标志 (gpt/qwen/image/regen 全关)")
    except Exception as e:
        actions.append(f"⚠️ 写 stop 标志失败: {e}")

    # 2) kill 所有生产进程 (主进程死 = 所有 worker 线程当场停)
    killed = []
    self_pid = str(os.getpid())
    for pat in ["run_production_v7_queue.py", "questionbank_export", "monitor_export"]:
        try:
            out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout.split()
            for pid in out:
                if pid == self_pid:
                    continue
                try:
                    subprocess.run(["kill", "-TERM", pid], timeout=5)
                    killed.append(f"{pat}:{pid}")
                except Exception:
                    pass
        except Exception:
            pass
    actions.append(f"✅ 已终止进程: {killed if killed else '无匹配进程'}")

    # 3) 落触发标记 (幂等, 防重复)
    try:
        with open(TRIGGER_MARK, "w") as f:
            f.write(f"{ts}  FINAL_PASS={fp} >= {TARGET}\n")
        actions.append("✅ 已落触发标记 (后续静默, 不重复)")
    except Exception as e:
        actions.append(f"⚠️ 落标记失败: {e}")

    # 投递终止报告
    print("🎯🎯🎯 项目目标达成! 已硬性终止所有生产流程")
    print(f"")
    print(f"FINAL_PASS = {fp}  (>= 目标 {TARGET})")
    print(f"终止时间: {ts}")
    print(f"")
    print("执行的动作:")
    for a in actions:
        print(f"  {a}")
    print(f"")
    print("所有 GPT/Qwen/出图 worker 已停止。项目根本目的已完成。")
    print("⚠️ 此看门狗将进入静默 (已落标记), 不会再 kill 任何进程。")

if __name__ == "__main__":
    main()
