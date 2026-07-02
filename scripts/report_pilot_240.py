#!/usr/bin/env python3
"""Generate a Markdown/PDF report for the 24-subject qwen3.6-flash pilot.

Read-only: inspects production.db and writes report files only.
"""
import argparse
import json
import re
import shutil
import sqlite3
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

SUBJECTS = [f"S{i:02d}" for i in range(1, 25)]
MEDICAL_SUBJECTS = {f"S{i:02d}" for i in range(7, 13)}
DEFAULT_TARGET_PER_SUBJECT = 10

STATUS_COLUMNS = [
    "GENERATED",
    "ACCEPTED",
    "SENTINEL_REGEN",
    "REGENERATE",
    "REPAIR_PENDING",
    "RENDER_FAIL",
    "RENDER_FAILED",
    "FINAL_PASS",
    "FINAL_FAIL",
    "HOLD",
    "DISCARDED",
]


def connect_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def load_rows(conn: sqlite3.Connection):
    return conn.execute(
        "SELECT question_id, subject_id, kp_id, question_json, quality_status, "
        "sentinel_result, image_path, render_attempts, created_at, updated_at "
        "FROM questions"
    ).fetchall()


def parse_json_field(value):
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def extract_rollout(row, question_json):
    candidates = []
    for key in ("qwen36_rollout_result", "qwen_rollout_result", "qwen_audit", "audit_result"):
        if isinstance(question_json.get(key), dict):
            candidates.append(question_json[key])
    sentinel = row["sentinel_result"] or ""
    detail = sentinel.lower()
    parsed = {}
    for field in ("wrong", "correct"):
        match = re.search(rf"{field}=([0-5])", detail)
        if match:
            parsed[f"{field}_count"] = int(match.group(1))
    answers_match = re.search(r"answers=([A-D, ]+)", sentinel)
    if answers_match:
        parsed["answers"] = [x.strip() for x in answers_match.group(1).split(",") if x.strip()]
    if parsed:
        candidates.append(parsed)
    return candidates[0] if candidates else {}


def summarize(rows):
    by_subject = {sid: Counter() for sid in SUBJECTS}
    subject_totals = Counter()
    rollout_wrong = Counter()
    failure_types = Counter()
    qwen = Counter()
    render_fail = 0
    final_pass = 0

    for row in rows:
        sid = row["subject_id"] or (row["kp_id"] or "")[:3]
        status = row["quality_status"] or "UNKNOWN"
        if sid not in by_subject:
            by_subject[sid] = Counter()
        by_subject[sid][status] += 1
        subject_totals[sid] += 1
        if status == "FINAL_PASS":
            final_pass += 1
        if status in {"RENDER_FAIL", "RENDER_FAILED"}:
            render_fail += 1

        question_json = parse_json_field(row["question_json"])
        rollout = extract_rollout(row, question_json)
        if rollout:
            wrong = rollout.get("wrong_count")
            correct = rollout.get("correct_count")
            if wrong is not None:
                rollout_wrong[int(wrong)] += 1
                qwen["rollout_questions_with_wrong_count"] += 1
                qwen["rollout_calls_estimated"] += int(rollout.get("rollouts") or 5)
            if wrong is not None and int(wrong) >= 3:
                qwen["candidate_pass"] += 1
            elif correct is not None and int(correct) >= 3:
                qwen["candidate_quality_fail"] += 1
        sentinel = (row["sentinel_result"] or "").lower()
        if "qwen_technical" in sentinel or "technical_failure" in sentinel:
            qwen["technical_fail"] += 1
            failure_types["qwen technical failure"] += 1
        elif "qwen36_candidate_quality_fail" in sentinel or "candidate_quality_fail" in sentinel:
            failure_types["qwen candidate too easy"] += 1
        elif "render" in sentinel and "fail" in sentinel:
            failure_types["render failure"] += 1
        elif status in {"SENTINEL_REGEN", "REGENERATE", "REPAIR_PENDING"}:
            failure_types["regen / repair pending"] += 1
        elif status == "HOLD":
            failure_types["hold"] += 1

    return {
        "by_subject": by_subject,
        "subject_totals": subject_totals,
        "rollout_wrong": rollout_wrong,
        "failure_types": failure_types,
        "qwen": qwen,
        "render_fail": render_fail,
        "final_pass": final_pass,
        "total_rows": len(rows),
    }


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def build_markdown(db_path: Path, summary):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("# 240题 Qwen3.6 Flash Thinking Pilot 测试报告")
    lines.append("")
    lines.append(f"生成时间：{now}")
    lines.append(f"数据库：`{db_path}`")
    lines.append("")
    lines.append("## 1. 项目概述")
    lines.append("")
    lines.append("- 目标：24科 × 每科10题 = 240题 pilot。")
    lines.append("- 审核模型：`qwen3.6-flash` thinking。")
    lines.append("- 审核规则：每题独立作答5次，错≥3次通过；答对≥3次进入重生成。")
    lines.append("- 医学图像规则：仅使用 synthetic/simulated medical diagrams，不使用真实患者影像。")
    lines.append("- 图片生成：通过 `image_prompt` 交给 image worker；Qwen 审核使用 `image_contract`/visual facts。")
    lines.append("")
    lines.append("## 2. 总体完成情况")
    lines.append("")
    lines.append(md_table(["指标", "数值"], [
        ["目标题数", 240],
        ["DB总记录", summary["total_rows"]],
        ["FINAL_PASS", summary["final_pass"]],
        ["Render Fail", summary["render_fail"]],
        ["Qwen rollout题数", summary["qwen"].get("rollout_questions_with_wrong_count", 0)],
        ["Qwen调用估算", summary["qwen"].get("rollout_calls_estimated", 0)],
        ["Qwen难度通过", summary["qwen"].get("candidate_pass", 0)],
        ["Qwen判太容易", summary["qwen"].get("candidate_quality_fail", 0)],
        ["Qwen技术失败", summary["qwen"].get("technical_fail", 0)],
    ]))
    lines.append("")
    lines.append("## 3. 分科完成情况")
    lines.append("")
    subject_rows = []
    for sid in SUBJECTS:
        c = summary["by_subject"].get(sid, Counter())
        subject_rows.append([
            sid,
            DEFAULT_TARGET_PER_SUBJECT,
            "yes" if sid in MEDICAL_SUBJECTS else "no",
            sum(c.values()),
            c.get("FINAL_PASS", 0),
            c.get("GENERATED", 0),
            c.get("ACCEPTED", 0),
            c.get("SENTINEL_REGEN", 0) + c.get("REGENERATE", 0) + c.get("REPAIR_PENDING", 0),
            c.get("RENDER_FAIL", 0) + c.get("RENDER_FAILED", 0),
            c.get("HOLD", 0),
        ])
    lines.append(md_table(["科目", "目标", "医学模拟图", "总记录", "FINAL_PASS", "GENERATED", "ACCEPTED", "REGEN", "RENDER_FAIL", "HOLD"], subject_rows))
    lines.append("")
    lines.append("## 4. Qwen 5次 Rollout 统计")
    lines.append("")
    wrong_rows = [[wrong, summary["rollout_wrong"].get(wrong, 0)] for wrong in range(0, 6)]
    lines.append(md_table(["wrong_count", "题数"], wrong_rows))
    lines.append("")
    lines.append("## 5. 失败类型摘要")
    lines.append("")
    failure_rows = [[k, v] for k, v in summary["failure_types"].most_common()]
    if failure_rows:
        lines.append(md_table(["失败类型", "数量"], failure_rows))
    else:
        lines.append("暂无失败类型记录。")
    lines.append("")
    lines.append("## 6. 最终结论")
    lines.append("")
    if summary["final_pass"] >= 240:
        lines.append("- 当前 DB 中 FINAL_PASS 已达到或超过 240，pilot 数量目标达成。")
    else:
        lines.append(f"- 当前 DB 中 FINAL_PASS={summary['final_pass']}，尚未达到 240。")
    lines.append("- 正式验收仍需补充真实交付抽检、近似去重、语言比例和医学合规说明。")
    lines.append("")
    return "\n".join(lines)


def try_pdf(md_path: Path, pdf_path: Path):
    pandoc = shutil.which("pandoc")
    if pandoc:
        subprocess.run([pandoc, str(md_path), "-o", str(pdf_path)], check=True)
        return True, "pandoc"
    return False, "pandoc not found"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("production.db"))
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-pdf", type=Path)
    args = parser.parse_args()

    conn = connect_db(args.db)
    rows = load_rows(conn)
    summary = summarize(rows)
    markdown = build_markdown(args.db, summary)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown, encoding="utf-8")
    print(f"WROTE_MD {args.out_md}")

    if args.out_pdf:
        ok, detail = try_pdf(args.out_md, args.out_pdf)
        if ok:
            print(f"WROTE_PDF {args.out_pdf} via {detail}")
        else:
            print(f"PDF_SKIPPED {detail}; markdown report is available")


if __name__ == "__main__":
    main()
