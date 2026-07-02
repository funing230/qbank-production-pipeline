#!/usr/bin/env python3
"""Build a verified pilot DB seeded only with qwen3.6 five-rollout PASS rows."""
from __future__ import annotations
import argparse
import shutil
import sqlite3
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-db', default='pilot_240_qwen36/pilot_240.db')
    ap.add_argument('--audit-db', default='pilot_240_qwen36/qwen36_reaudit/qwen36_reaudit.db')
    ap.add_argument('--out-db', default='pilot_240_qwen36_verified/verified_pilot.db')
    args = ap.parse_args()
    source = Path(args.source_db)
    audit = Path(args.audit_db)
    out = Path(args.out_db)
    if out.exists():
        out.unlink()
    for suffix in ('-wal', '-shm'):
        p = Path(str(out) + suffix)
        if p.exists():
            p.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)

    src = sqlite3.connect(source)
    src.row_factory = sqlite3.Row
    dst = sqlite3.connect(out)
    aud = sqlite3.connect(audit)
    aud.row_factory = sqlite3.Row

    schema = src.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='questions'").fetchone()[0]
    dst.execute(schema)
    indexes = src.execute("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='questions' AND sql IS NOT NULL").fetchall()
    for (sql,) in indexes:
        try:
            dst.execute(sql)
        except sqlite3.OperationalError:
            pass

    pass_ids = [r['question_id'] for r in aud.execute(
        "SELECT question_id FROM qwen36_audit WHERE decision='PASS' AND wrong_count>=3 AND valid_count=5 ORDER BY subject_id, question_id"
    )]
    if not pass_ids:
        raise SystemExit('no verified pass rows found')
    cols = [r[1] for r in src.execute('PRAGMA table_info(questions)').fetchall()]
    placeholders = ','.join('?' for _ in cols)
    col_sql = ','.join(cols)
    copied = 0
    for qid in pass_ids:
        row = src.execute(f"SELECT {col_sql} FROM questions WHERE question_id=?", (qid,)).fetchone()
        if not row:
            continue
        values = [row[c] for c in cols]
        dst.execute(f"INSERT INTO questions ({col_sql}) VALUES ({placeholders})", values)
        copied += 1
    dst.commit()

    print(f'COPIED_VERIFIED_PASS {copied}')
    print('BY_SUBJECT')
    for sid, n in dst.execute("SELECT subject_id, COUNT(*) FROM questions GROUP BY subject_id ORDER BY subject_id"):
        print(sid, n, 'gap', max(0, 10 - n))
    src.close(); aud.close(); dst.close()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
