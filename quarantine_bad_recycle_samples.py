#!/usr/bin/env python3
"""Quarantine malformed recycle samples before they consume Qwen retries."""
import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB = PROJECT_ROOT / "production.db"
RECYCLE_STATUSES = (
    "GENERATED",
    "HOLD",
    "SENTINEL_REGEN",
    "REGENERATE",
    "REPAIR_PENDING",
    "RENDER_FAIL",
    "RENDER_FAILED",
    "ACCEPTED",
)
QUARANTINE_STATUS = "HOLD"
QUARANTINE_MARKER = "recycle_quarantine"
ILLEGAL_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
LATEX_SIGNAL_RE = re.compile(r"\\[A-Za-z]+|\\[\[(]|\$\$|(?<!\\)\$(?!\$)")
LATEX_TEXT_FIELDS = (
    "question_text",
    "explanation",
    "image_prompt",
    "final_image_prompt",
    "image_dependency_reason",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_text(value: Any, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for nested in value.values():
            append_text(nested, out)
    elif isinstance(value, list):
        for nested in value:
            append_text(nested, out)


def latex_validation_text(question: dict) -> str:
    chunks: list[str] = []
    for field_name in LATEX_TEXT_FIELDS:
        append_text(question.get(field_name, ""), chunks)
    append_text(question.get("options", {}), chunks)
    append_text(question.get("truth_spec", {}), chunks)
    return "\n".join(chunks)


def quarantine_reasons(raw: Any) -> list[str]:
    reasons: list[str] = []
    if raw is None or not str(raw).strip():
        return ["empty_question_json"]
    match = ILLEGAL_CONTROL_CHAR_RE.search(str(raw))
    if match:
        reasons.append(f"illegal_control_char:U+{ord(match.group()):04X}")
    try:
        question = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as exc:
        reasons.append(f"json_parse_error:{str(exc)[:120]}")
        return reasons
    if not isinstance(question, dict):
        reasons.append(f"invalid_question_json_type:{type(question).__name__}")
        return reasons
    latex_text = latex_validation_text(question)
    if not LATEX_SIGNAL_RE.search(latex_text):
        return reasons
    if latex_text.count("\\(") != latex_text.count("\\)"):
        reasons.append("unbalanced_latex_inline_delimiter")
    if latex_text.count("\\[") != latex_text.count("\\]"):
        reasons.append("unbalanced_latex_display_delimiter")
    if latex_text.count("$$") % 2:
        reasons.append("unbalanced_latex_double_dollar")
    return reasons


def already_quarantined(detail: str | None) -> bool:
    return QUARANTINE_MARKER in (detail or "")


def fetch_candidates(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    placeholders = ",".join("?" for _ in RECYCLE_STATUSES)
    sql = (
        "SELECT q.question_id, q.subject_id, q.kp_id, q.quality_status, q.question_json, "
        "COALESCE(a.detail, '') AS last_audit_detail "
        "FROM questions q "
        "LEFT JOIN (SELECT question_id, MAX(id) AS max_id FROM audit_log GROUP BY question_id) latest "
        "ON latest.question_id = q.question_id "
        "LEFT JOIN audit_log a ON a.id = latest.max_id "
        f"WHERE q.quality_status IN ({placeholders}) "
        "ORDER BY q.updated_at, q.created_at"
    )
    return conn.execute(sql, RECYCLE_STATUSES).fetchall()


def write_audit(conn: sqlite3.Connection, row: sqlite3.Row, detail: str) -> None:
    ts = now_iso()
    conn.execute(
        "UPDATE questions SET quality_status = ?, sentinel_result = ?, updated_at = ? WHERE question_id = ?",
        (QUARANTINE_STATUS, detail[:500], ts, row["question_id"]),
    )
    conn.execute(
        "INSERT INTO audit_log (question_id, action, old_status, new_status, detail, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (row["question_id"], "recycle_quarantine", row["quality_status"], QUARANTINE_STATUS, detail[:500], ts),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite production DB path")
    parser.add_argument("--apply", action="store_true", help="write HOLD quarantine markers; default is dry-run")
    parser.add_argument("--manifest", default="", help="output JSON manifest path")
    args = parser.parse_args()

    db_path = Path(args.db)
    manifest_path = Path(args.manifest) if args.manifest else PROJECT_ROOT / f"recycle_quarantine_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")

    quarantines = []
    skipped_existing = 0
    for row in fetch_candidates(conn):
        if already_quarantined(row["last_audit_detail"]):
            skipped_existing += 1
            continue
        reasons = quarantine_reasons(row["question_json"])
        if not reasons:
            continue
        detail = f"{QUARANTINE_MARKER}: " + "; ".join(reasons)
        quarantines.append({
            "question_id": row["question_id"],
            "subject_id": row["subject_id"],
            "kp_id": row["kp_id"],
            "old_status": row["quality_status"],
            "reasons": reasons,
        })
        if args.apply:
            write_audit(conn, row, detail)

    if args.apply:
        conn.commit()
    manifest = {
        "mode": "apply" if args.apply else "dry_run",
        "db": str(db_path),
        "timestamp": now_iso(),
        "candidate_statuses": list(RECYCLE_STATUSES),
        "quarantine_count": len(quarantines),
        "already_quarantined_skipped": skipped_existing,
        "items": quarantines,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("mode", "quarantine_count", "already_quarantined_skipped")}, ensure_ascii=False))
    print(f"manifest={manifest_path}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
