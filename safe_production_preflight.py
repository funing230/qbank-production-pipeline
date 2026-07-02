#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_FINAL_PASS = 3663
EXPECTED_STATUS = {
    "FINAL_PASS": 3663,
    "ACCEPTED": 214,
    "GENERATED": 997,
    "RENDER_FAIL": 1272,
    "SENTINEL_REGEN": 1073,
}
WATCHED_PROCESS_NAMES = (
    "run_production_v7_queue.py",
    "run_production_v7.py",
    "monitor_export_every10.py",
)
STATE_FILES = (
    "monitor_export_every10_state.json",
    ".monitor_export_feishu_state.json",
)
PYTHON_REL = Path(".venv-qbank-render/bin/python3")


def fail(message: str, payload: dict | None = None) -> None:
    print(json.dumps({"ok": False, "error": message, "details": payload or {}}, ensure_ascii=False, indent=2))
    raise SystemExit(1)


def status_counts(db_path: Path) -> tuple[dict[str, int], int]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT quality_status, COUNT(*) FROM questions GROUP BY quality_status").fetchall()
    finally:
        conn.close()
    status = {str(key): int(value) for key, value in rows}
    return status, sum(status.values())


def check_baseline() -> dict:
    db_path = PROJECT_ROOT / "production.db"
    if not db_path.exists():
        fail("production.db not found", {"path": str(db_path)})
    status, total_rows = status_counts(db_path)
    final_pass = status.get("FINAL_PASS", 0)
    report = {"final_pass": final_pass, "total_rows": total_rows, "status": status}
    if final_pass != EXPECTED_FINAL_PASS:
        fail("FINAL_PASS baseline mismatch; refusing to start", report)
    return report


def matching_processes() -> list[dict[str, str | int]]:
    result = subprocess.run(
        ["ps", "-eo", "pid,ppid,stat,etime,args"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    matches = []
    current_pid = str(Path("/proc/self").resolve().name) if Path("/proc/self").exists() else ""
    for line in result.stdout.splitlines()[1:]:
        if not any(name in line for name in WATCHED_PROCESS_NAMES):
            continue
        fields = line.split(maxsplit=4)
        if len(fields) < 5:
            continue
        pid, ppid, stat, etime, args = fields
        if pid == current_pid or "safe_production_preflight.py" in args:
            continue
        matches.append({"pid": int(pid), "ppid": int(ppid), "stat": stat, "etime": etime, "args": args})
    return matches


def check_no_duplicate_processes() -> list[dict[str, str | int]]:
    matches = matching_processes()
    if matches:
        fail("duplicate production/monitor process detected; refusing to start", {"processes": matches})
    return matches


def clean_pycache() -> list[str]:
    removed = []
    for path in PROJECT_ROOT.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(str(path.relative_to(PROJECT_ROOT)))
    return removed


def backup_state_files(timestamp: str) -> list[dict[str, str]]:
    backups = []
    for name in STATE_FILES:
        path = PROJECT_ROOT / name
        if not path.exists():
            continue
        target = PROJECT_ROOT / f"{name}.bak_before_restart_{timestamp}"
        if target.exists():
            fail("state backup target already exists", {"source": str(path), "target": str(target)})
        path.rename(target)
        backups.append({"source": name, "backup": target.name})
    return backups


def compile_entrypoints() -> None:
    python = PROJECT_ROOT / PYTHON_REL
    if not python.exists():
        fail("project Python not found", {"python": str(python)})
    result = subprocess.run(
        [
            str(python),
            "-m",
            "py_compile",
            "run_production_v7_queue.py",
            "monitor_export_every10.py",
            "pipeline/reviewer.py",
            "pipeline/db.py",
            "pipeline/image_renderer.py",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        fail("py_compile failed", {"output": result.stdout})


def verify_provider_plan() -> dict:
    python = PROJECT_ROOT / PYTHON_REL
    code = """
import json
import os
from pathlib import Path
for line in Path('config/.env').read_text().splitlines():
    if '=' in line and not line.strip().startswith('#'):
        key, value = line.split('=', 1)
        os.environ[key] = value
import run_production_v7_queue as m
providers, total = m.build_image_renderers()
print(json.dumps({
    'providers': [
        {'name': name, 'concurrency': concurrency, 'ramp': ramp, 'provider': getattr(renderer, 'provider', ''), 'model': getattr(renderer, 'model', ''), 'key_present': bool(getattr(renderer, 'api_key', ''))}
        for name, renderer, concurrency, ramp in providers
    ],
    'total': total,
}, ensure_ascii=False))
"""
    result = subprocess.run(
        [str(python), "-c", code],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        fail("image provider plan check failed", {"output": result.stdout})
    lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        fail("image provider plan missing JSON output", {"output": result.stdout})
    return json.loads(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Safe preflight for qbank production restart")
    parser.add_argument("--apply", action="store_true", help="Perform cache cleanup and state-file backup after checks pass.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline = check_baseline()
    processes = check_no_duplicate_processes()
    compile_entrypoints()
    provider_plan = verify_provider_plan()
    removed_pycache = []
    state_backups = []
    if args.apply:
        removed_pycache = clean_pycache()
        state_backups = backup_state_files(timestamp)

    print(json.dumps({
        "ok": True,
        "applied": args.apply,
        "baseline": baseline,
        "duplicate_processes": processes,
        "provider_plan": provider_plan,
        "removed_pycache": removed_pycache,
        "state_backups": state_backups,
        "monitor_command": [str(PYTHON_REL), "monitor_export_every10.py", "--db", "production.db", "--baseline", str(EXPECTED_FINAL_PASS), "--target", "24000", "--interval", "60", "--chunk", "10"],
        "production_command": [str(PYTHON_REL), "run_production_v7_queue.py", "--enable-db-recycle", "--recycle-gpt-workers", "2"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
