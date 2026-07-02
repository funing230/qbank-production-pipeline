#!/usr/bin/env python3
"""Re-audit existing FINAL_PASS candidates with qwen3.6-flash thinking 5-rollout logic.

Read-only for source DB. Writes an independent audit SQLite + JSONL checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sqlite3
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

DEFAULT_MODEL = "qwen3.6-flash"
DEFAULT_BASE_URL = "https://yuanlansj.xin/v1"
ROLL_OUTS = 5


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def normalize_answer(answer) -> str:
    text = str(answer or "").strip().upper()
    match = re.search(r"[ABCD]", text)
    return match.group(0) if match else ""


def parse_answer(content: str) -> str:
    if not content:
        return ""
    text = re.sub(r"<think>.*?</think>", "", str(content), flags=re.DOTALL).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return normalize_answer(data.get("answer"))
    except Exception:
        pass
    match = re.search(r'"answer"\s*:\s*"?([ABCD])"?', text, flags=re.I)
    if match:
        return normalize_answer(match.group(1))
    match = re.search(r"(?:最终答案|答案|answer)\s*[:：]?\s*([ABCD])", text, flags=re.I)
    if match:
        return normalize_answer(match.group(1))
    candidates = re.findall(r"\b([ABCD])\b", text.upper())
    if len(candidates) == 1:
        return candidates[0]
    return ""


def build_prompt(question: dict) -> str:
    safe_question = {
        "question_text": question.get("question_text", ""),
        "options": question.get("options", {}),
    }
    if question.get("image_contract") is not None:
        safe_question["image_contract"] = question.get("image_contract")
    elif question.get("visual_facts") is not None:
        safe_question["visual_facts"] = question.get("visual_facts")
    subject = question.get("subject") or question.get("subject_name")
    if subject:
        safe_question["subject"] = subject
    kp = question.get("kp") or question.get("knowledge_point") or question.get("kp_name")
    if kp:
        safe_question["kp"] = kp
    return (
        "You are a student taking a multiple-choice exam. "
        "Use only the question, options, and visual facts below. "
        "Return exactly one JSON object and no explanation: {\"answer\":\"A\"}\n\n"
        f"QUESTION_JSON:\n{json.dumps(safe_question, ensure_ascii=False, sort_keys=True)}"
    )


def call_qwen(prompt: str, *, api_key: str, base_url: str, model: str, timeout: int) -> tuple[str, dict]:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "temperature": 0.7,
        "enable_thinking": True,
    }).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    msg = body.get("choices", [{}])[0].get("message", {})
    content = msg.get("content", "")
    meta = {
        "model": body.get("model"),
        "message_keys": sorted(msg.keys()),
        "has_reasoning_content": bool(msg.get("reasoning_content")),
        "usage": body.get("usage", {}),
    }
    return content, meta


def init_audit_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS qwen36_audit (
            question_id TEXT PRIMARY KEY,
            subject_id TEXT,
            source_status TEXT,
            correct_answer TEXT,
            answers_json TEXT,
            raw_summaries_json TEXT,
            correct_count INTEGER,
            wrong_count INTEGER,
            valid_count INTEGER,
            rollouts INTEGER,
            decision TEXT,
            technical_failure INTEGER,
            error_summary TEXT,
            started_at TEXT,
            updated_at TEXT
        )
        """
    )
    conn.commit()
    return conn


def load_candidates(source_db: Path, limit: int | None = None) -> list[dict]:
    conn = sqlite3.connect(source_db)
    conn.row_factory = sqlite3.Row
    sql = """
        SELECT question_id, subject_id, quality_status, question_json
        FROM questions
        WHERE quality_status='FINAL_PASS'
        ORDER BY subject_id, question_id
    """
    rows = conn.execute(sql).fetchall()
    conn.close()
    out = []
    for row in rows[:limit]:
        try:
            question = json.loads(row["question_json"] or "{}")
        except Exception:
            question = {}
        out.append({
            "question_id": row["question_id"],
            "subject_id": row["subject_id"],
            "source_status": row["quality_status"],
            "question": question,
            "correct_answer": normalize_answer(question.get("correct_answer")),
        })
    return out


def already_done(conn: sqlite3.Connection, question_id: str) -> bool:
    row = conn.execute(
        "SELECT decision, valid_count FROM qwen36_audit WHERE question_id=?",
        (question_id,),
    ).fetchone()
    return bool(row and row[0] in {"PASS", "FAIL"} and int(row[1] or 0) == ROLL_OUTS)


def write_result(conn: sqlite3.Connection, lock: threading.Lock, result: dict, jsonl_path: Path) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with lock:
        conn.execute(
            """
            INSERT INTO qwen36_audit (
                question_id, subject_id, source_status, correct_answer, answers_json,
                raw_summaries_json, correct_count, wrong_count, valid_count, rollouts,
                decision, technical_failure, error_summary, started_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(question_id) DO UPDATE SET
                answers_json=excluded.answers_json,
                raw_summaries_json=excluded.raw_summaries_json,
                correct_count=excluded.correct_count,
                wrong_count=excluded.wrong_count,
                valid_count=excluded.valid_count,
                rollouts=excluded.rollouts,
                decision=excluded.decision,
                technical_failure=excluded.technical_failure,
                error_summary=excluded.error_summary,
                updated_at=excluded.updated_at
            """,
            (
                result["question_id"], result["subject_id"], result["source_status"],
                result["correct_answer"], json.dumps(result["answers"], ensure_ascii=False),
                json.dumps(result["raw_summaries"], ensure_ascii=False), result["correct_count"],
                result["wrong_count"], result["valid_count"], result["rollouts"],
                result["decision"], int(result["technical_failure"]), result["error_summary"],
                result.get("started_at") or now, now,
            ),
        )
        conn.commit()
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**result, "updated_at": now}, ensure_ascii=False) + "\n")


def audit_one(item: dict, *, api_key: str, base_url: str, model: str, timeout: int) -> dict:
    started = datetime.now().isoformat(timespec="seconds")
    expected = item["correct_answer"]
    prompt = build_prompt(item["question"])
    answers = []
    raw_summaries = []
    failures = []
    for idx in range(ROLL_OUTS):
        try:
            raw, meta = call_qwen(prompt, api_key=api_key, base_url=base_url, model=model, timeout=timeout)
            ans = parse_answer(raw)
            raw_summaries.append({
                "rollout": idx + 1,
                "answer": ans,
                "raw_prefix": str(raw)[:160],
                "meta": meta,
            })
            if ans:
                answers.append(ans)
            else:
                failures.append({"rollout": idx + 1, "error": "parse_error"})
        except Exception as exc:
            failures.append({"rollout": idx + 1, "error": type(exc).__name__, "detail": str(exc)[:160]})
            raw_summaries.append({"rollout": idx + 1, "answer": "", "error": type(exc).__name__})
    correct_count = sum(1 for a in answers if a == expected)
    wrong_count = sum(1 for a in answers if a != expected)
    valid_count = len(answers)
    technical_failure = valid_count < ROLL_OUTS
    if technical_failure:
        decision = "TECH_FAIL"
    elif wrong_count >= 3:
        decision = "PASS"
    else:
        decision = "FAIL"
    return {
        "question_id": item["question_id"],
        "subject_id": item["subject_id"],
        "source_status": item["source_status"],
        "correct_answer": expected,
        "answers": answers,
        "raw_summaries": raw_summaries,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "valid_count": valid_count,
        "rollouts": ROLL_OUTS,
        "decision": decision,
        "technical_failure": technical_failure,
        "error_summary": json.dumps(failures, ensure_ascii=False)[:500] if failures else "",
        "started_at": started,
    }


def summarize(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    total = cur.execute("SELECT COUNT(*) FROM qwen36_audit").fetchone()[0]
    done = cur.execute("SELECT COUNT(*) FROM qwen36_audit WHERE valid_count=5 AND decision IN ('PASS','FAIL')").fetchone()[0]
    passed = cur.execute("SELECT COUNT(*) FROM qwen36_audit WHERE decision='PASS' AND wrong_count>=3 AND valid_count=5").fetchone()[0]
    failed = cur.execute("SELECT COUNT(*) FROM qwen36_audit WHERE decision='FAIL' AND valid_count=5").fetchone()[0]
    tech = cur.execute("SELECT COUNT(*) FROM qwen36_audit WHERE decision='TECH_FAIL'").fetchone()[0]
    by_subject = cur.execute("""
        SELECT subject_id,
               SUM(CASE WHEN decision='PASS' AND wrong_count>=3 AND valid_count=5 THEN 1 ELSE 0 END) pass_n,
               COUNT(*) total_n
        FROM qwen36_audit GROUP BY subject_id ORDER BY subject_id
    """).fetchall()
    subjects = ", ".join(f"{s}:{p}/{t}" for s,p,t in by_subject)
    return f"audited={total} done={done} pass={passed} fail={failed} tech={tech} by_subject={subjects}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-db", default="pilot_240_qwen36/pilot_240.db")
    ap.add_argument("--audit-db", default="pilot_240_qwen36/qwen36_reaudit/qwen36_reaudit.db")
    ap.add_argument("--jsonl", default="pilot_240_qwen36/qwen36_reaudit/qwen36_reaudit.jsonl")
    ap.add_argument("--env", default="config/.env")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--base-url", default="")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    load_env(Path(args.env))
    api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("YUANLAN_API_KEY")
    if not api_key:
        raise SystemExit("missing QWEN_API_KEY/YUANLAN_API_KEY")
    base_url = args.base_url or os.environ.get("QWEN_BASE_URL") or DEFAULT_BASE_URL
    if args.model != DEFAULT_MODEL:
        raise SystemExit(f"refusing non-required model: {args.model}")

    source_db = Path(args.source_db)
    audit_db = Path(args.audit_db)
    jsonl = Path(args.jsonl)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    conn = init_audit_db(audit_db)
    lock = threading.Lock()
    candidates = load_candidates(source_db, args.limit or None)
    work = queue.Queue()
    skipped = 0
    for item in candidates:
        if args.resume and already_done(conn, item["question_id"]):
            skipped += 1
            continue
        work.put(item)

    print(f"START candidates={len(candidates)} queued={work.qsize()} skipped={skipped} model={args.model} base={base_url} workers={args.workers} rollouts={ROLL_OUTS}", flush=True)

    def worker(worker_id: int):
        while True:
            try:
                item = work.get_nowait()
            except queue.Empty:
                return
            try:
                result = audit_one(item, api_key=api_key, base_url=base_url, model=args.model, timeout=args.timeout)
                write_result(conn, lock, result, jsonl)
                print(
                    f"DONE worker={worker_id} qid={item['question_id']} subj={item['subject_id']} "
                    f"decision={result['decision']} wrong={result['wrong_count']} correct={result['correct_count']} "
                    f"valid={result['valid_count']} answers={','.join(result['answers'])}",
                    flush=True,
                )
            except Exception as exc:
                result = {
                    "question_id": item["question_id"], "subject_id": item["subject_id"],
                    "source_status": item["source_status"], "correct_answer": item["correct_answer"],
                    "answers": [], "raw_summaries": [], "correct_count": 0, "wrong_count": 0,
                    "valid_count": 0, "rollouts": ROLL_OUTS, "decision": "TECH_FAIL",
                    "technical_failure": True, "error_summary": f"{type(exc).__name__}: {str(exc)[:300]}",
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                }
                write_result(conn, lock, result, jsonl)
                print(f"ERROR worker={worker_id} qid={item['question_id']} {type(exc).__name__}: {exc}", flush=True)
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(max(1, args.workers))]
    for t in threads:
        t.start()
    last = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        if time.time() - last >= 30:
            with lock:
                print("PROGRESS", summarize(conn), flush=True)
            last = time.time()
    for t in threads:
        t.join()
    print("FINAL", summarize(conn), flush=True)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
