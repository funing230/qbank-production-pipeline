#!/usr/bin/env python3
"""Random FINAL_PASS multimodal QA audit with qwen3.5-flash via Hermes CLI.

Read-only against production.db. Samples N FINAL_PASS questions, calls model R times per
question with attached image, extracts final A/B/C/D, compares to DB correct_answer,
and writes JSON + Markdown reports.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shlex
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ANSWER_RE_LIST = [
    re.compile(r"最终答案\s*[:：]\s*([ABCD])", re.I),
    re.compile(r"答案\s*[:：]\s*([ABCD])", re.I),
    re.compile(r"\b([ABCD])\b", re.I),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="production.db")
    p.add_argument("--sample-size", type=int, default=10)
    p.add_argument("--status", default="FINAL_PASS")
    p.add_argument("--provider", default="custom:gpt5")
    p.add_argument("--model", default="qwen3.5-flash")
    p.add_argument("--runs-per-question", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--source", default="qwen35_flash_finalpass_eval")
    return p.parse_args()


def load_rows(db_path: Path, status: str, sample_size: int, seed: int | None) -> list[sqlite3.Row]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        count = con.execute(
            "SELECT COUNT(*) FROM questions WHERE quality_status=? AND question_json IS NOT NULL",
            (status,),
        ).fetchone()[0]
        if count < sample_size:
            raise SystemExit(f"Only {count} rows with status={status}, cannot sample {sample_size}")
        if seed is None:
            rows = con.execute(
                "SELECT question_id, subject_id, module_id, kp_id, kp_name, question_json, image_path "
                "FROM questions WHERE quality_status=? AND question_json IS NOT NULL "
                "ORDER BY random() LIMIT ?",
                (status, sample_size),
            ).fetchall()
        else:
            all_rows = con.execute(
                "SELECT question_id, subject_id, module_id, kp_id, kp_name, question_json, image_path "
                "FROM questions WHERE quality_status=? AND question_json IS NOT NULL",
                (status,),
            ).fetchall()
            rnd = random.Random(seed)
            rows = rnd.sample(all_rows, sample_size)
        return rows
    finally:
        con.close()


def qget(q: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for k in keys:
        v = q.get(k)
        if v not in (None, ""):
            return v
    return default


def resolve_image(root: Path, row: sqlite3.Row, q: dict[str, Any]) -> str:
    candidates: list[str] = []
    for v in [row["image_path"], q.get("image_path"), q.get("final_image_path"), q.get("image_file")]:
        if v:
            candidates.append(str(v))
    qids = [row["question_id"], q.get("question_id")]
    for v in candidates:
        p = Path(v)
        if not p.is_absolute():
            p = root / p
        if p.exists() and p.is_file():
            return str(p)
    # common production layout: output/Sxx/images/<db-question-id>.png
    for qid in [x for x in qids if x]:
        p = root / "output" / str(row["subject_id"]) / "images" / f"{qid}.png"
        if p.exists():
            return str(p)
    # bounded fallback search under subject image dir
    img_dir = root / "output" / str(row["subject_id"]) / "images"
    if img_dir.exists():
        for qid in [x for x in qids if x]:
            hits = list(img_dir.glob(f"*{qid}*.*"))
            hits = [h for h in hits if h.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
            if hits:
                return str(hits[0])
    return ""


def build_prompt(row: sqlite3.Row, q: dict[str, Any]) -> str:
    opts = qget(q, "options", default={})
    if isinstance(opts, dict):
        opt_text = "\n".join(f"{k}. {v}" for k, v in opts.items())
    elif isinstance(opts, list):
        opt_text = "\n".join(str(x) for x in opts)
    else:
        opt_text = str(opts)
    return f"""你正在独立作答一道中文多模态选择题。请结合随附图片和题干信息判断正确选项。\n\n要求：\n1. 只能在 A/B/C/D 中选择一个最终答案。\n2. 可以简要推理，但不要引用标准答案或外部上下文。\n3. 最后一行必须严格写成：最终答案：A 或 最终答案：B 或 最终答案：C 或 最终答案：D\n\n题目ID：{row['question_id']}\n科目：{row['subject_id']}\n知识点：{row['kp_id']} {row['kp_name'] or ''}\n\n题干：\n{qget(q, 'question_text')}\n\n选项：\n{opt_text}\n"""


def extract_answer(text: str) -> str:
    tail = "\n".join(text.strip().splitlines()[-10:])
    for rgx in ANSWER_RE_LIST[:2]:
        m = rgx.search(tail)
        if m:
            return m.group(1).upper()
    for rgx in ANSWER_RE_LIST:
        ms = list(rgx.finditer(text))
        if ms:
            return ms[-1].group(1).upper()
    return ""


def call_model(prompt: str, image: str, provider: str, model: str, source: str, timeout: int) -> dict[str, Any]:
    cmd = [
        "hermes", "chat",
        "--ignore-rules",
        "--source", source,
        "-Q",
        "--provider", provider,
        "-m", model,
        "--max-turns", "1",
        "-q", prompt,
    ]
    if image:
        cmd[2:2] = ["--image", image]
    t0 = time.time()
    try:
        cp = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        out = cp.stdout.strip()
        err = cp.stderr.strip()
        return {
            "ok": cp.returncode == 0,
            "returncode": cp.returncode,
            "latency_sec": round(time.time() - t0, 2),
            "stdout": out,
            "stderr_tail": err[-2000:],
            "answer": extract_answer(out),
            "cmd_preview": " ".join(shlex.quote(x) for x in cmd[:12]) + " ...",
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "returncode": "timeout",
            "latency_sec": round(time.time() - t0, 2),
            "stdout": (e.stdout or "") if isinstance(e.stdout, str) else "",
            "stderr_tail": "TIMEOUT",
            "answer": "",
            "cmd_preview": " ".join(shlex.quote(x) for x in cmd[:12]) + " ...",
        }


def write_reports(data: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# qwen3.5-flash FINAL_PASS 随机抽检报告")
    lines.append("")
    lines.append(f"- generated_at: `{data['generated_at']}`")
    lines.append(f"- db: `{data['db']}`")
    lines.append(f"- provider/model: `{data['provider']}` / `{data['model']}`")
    lines.append(f"- sample_size: `{data['sample_size']}`")
    lines.append(f"- runs_per_question: `{data['runs_per_question']}`")
    lines.append(f"- total_runs: `{data['summary']['total_runs']}`")
    lines.append(f"- correct_runs: `{data['summary']['correct_runs']}`")
    lines.append(f"- run_accuracy: `{data['summary']['run_accuracy']:.2%}`")
    lines.append(f"- questions_all_correct: `{data['summary']['questions_all_correct']}/{data['sample_size']}`")
    lines.append(f"- questions_majority_correct: `{data['summary']['questions_majority_correct']}/{data['sample_size']}`")
    lines.append("")
    lines.append("## Per-question results")
    lines.append("")
    for item in data["items"]:
        lines.append(f"### {item['question_id']} ({item['subject_id']} / {item['kp_id']})")
        lines.append(f"- 标准答案: `{item['correct_answer']}`")
        lines.append(f"- 图片: `{item['image_path']}`")
        lines.append(f"- 题干: {item['question_text']}")
        lines.append(f"- 5次答案: `{', '.join(r['answer'] or '?' for r in item['runs'])}`")
        correct_count = item.get('correct_count', sum(1 for r in item['runs'] if r.get('is_correct')))
        lines.append(f"- 正确数: `{correct_count}/{len(item['runs'])}`")
        bad = [f"run{r['run_index']}={r['answer'] or '?'}" for r in item['runs'] if not r['is_correct']]
        if bad:
            lines.append(f"- 错误: {', '.join(bad)}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = root / db_path
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    rows = load_rows(db_path, args.status, args.sample_size, args.seed)

    data: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "db": str(db_path),
        "provider": args.provider,
        "model": args.model,
        "sample_size": args.sample_size,
        "runs_per_question": args.runs_per_question,
        "seed": args.seed,
        "items": [],
        "summary": {},
    }

    total_runs = correct_runs = 0
    q_all_correct = q_majority_correct = 0
    for qi, row in enumerate(rows, 1):
        q = json.loads(row["question_json"])
        correct = str(qget(q, "correct_answer")).strip().upper()[:1]
        image = resolve_image(root, row, q)
        prompt = build_prompt(row, q)
        item = {
            "question_id": row["question_id"],
            "json_question_id": q.get("question_id"),
            "subject_id": row["subject_id"],
            "module_id": row["module_id"],
            "kp_id": row["kp_id"],
            "kp_name": row["kp_name"],
            "question_text": qget(q, "question_text"),
            "options": qget(q, "options", default={}),
            "correct_answer": correct,
            "image_path": image,
            "runs": [],
        }
        print(f"[{qi}/{len(rows)}] {row['question_id']} correct={correct} image={'YES' if image else 'NO'}", flush=True)
        for ri in range(1, args.runs_per_question + 1):
            res = call_model(prompt, image, args.provider, args.model, args.source, args.timeout)
            ans = res.get("answer", "")
            is_correct = bool(ans and ans == correct)
            total_runs += 1
            correct_runs += int(is_correct)
            run = {
                "run_index": ri,
                "ok": res["ok"],
                "returncode": res["returncode"],
                "latency_sec": res["latency_sec"],
                "answer": ans,
                "is_correct": is_correct,
                "stdout": res["stdout"],
                "stderr_tail": res["stderr_tail"],
            }
            item["runs"].append(run)
            print(f"  run {ri}/{args.runs_per_question}: ans={ans or '?'} correct={is_correct} latency={res['latency_sec']}s ok={res['ok']}", flush=True)
            # incremental checkpoint after every call
            tmp = dict(data)
            tmp["items"] = data["items"] + [item]
            tmp["summary"] = {
                "total_runs": total_runs,
                "correct_runs": correct_runs,
                "run_accuracy": correct_runs / total_runs if total_runs else 0.0,
                "questions_all_correct": q_all_correct,
                "questions_majority_correct": q_majority_correct,
                "in_progress": True,
            }
            write_reports(tmp, out_json, out_md)
        c = sum(1 for r in item["runs"] if r["is_correct"])
        item["correct_count"] = c
        item["all_correct"] = c == len(item["runs"])
        item["majority_correct"] = c >= (len(item["runs"]) // 2 + 1)
        q_all_correct += int(item["all_correct"])
        q_majority_correct += int(item["majority_correct"])
        data["items"].append(item)

    data["summary"] = {
        "total_runs": total_runs,
        "correct_runs": correct_runs,
        "run_accuracy": correct_runs / total_runs if total_runs else 0.0,
        "questions_all_correct": q_all_correct,
        "questions_majority_correct": q_majority_correct,
        "in_progress": False,
    }
    write_reports(data, out_json, out_md)
    print(f"DONE json={out_json} md={out_md}")
    print(json.dumps(data["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
