#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / 'production.db'
PY = ROOT / '.venv-qbank-render/bin/python3'
CONTROL = ROOT / 'runtime_control.json'
LOG = ROOT / 'qbank_final_pass_guard.log'
CHECK_INTERVAL = 60
STALL_SECONDS = 10 * 60
DRAIN_SECONDS = 150
TARGET = 24000
EXPORT_CHUNK = 10
EXPORT_INTERVAL = 60
EXPORT_DIR = Path('/mnt/c/Users/admin/Desktop/test')


def log(message: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')


def final_pass_count() -> int:
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM questions WHERE quality_status='FINAL_PASS'").fetchone()[0])
    finally:
        conn.close()


def status_distribution() -> dict[str, int]:
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    try:
        return dict(conn.execute('SELECT quality_status, COUNT(*) FROM questions GROUP BY quality_status').fetchall())
    finally:
        conn.close()


def pgrep(pattern: str) -> list[tuple[int, str]]:
    out = subprocess.run(['pgrep', '-af', pattern], text=True, capture_output=True).stdout.strip()
    rows = []
    own_pid = os.getpid()
    for line in out.splitlines():
        if not line:
            continue
        parts = line.split(' ', 1)
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == own_pid:
            continue
        cmd = parts[1] if len(parts) > 1 else ''
        rows.append((pid, cmd))
    return rows


def stop_processes(pattern: str, graceful: bool = True):
    sig = signal.SIGTERM if graceful else signal.SIGKILL
    for pid, cmd in pgrep(pattern):
        log(f"sending {sig.name} to pid={pid} cmd={cmd[:180]}")
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def wait_absent(pattern: str, timeout_sec: int) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if not pgrep(pattern):
            return True
        time.sleep(2)
    return not pgrep(pattern)


def tail(path: Path, lines: int = 30) -> str:
    if not path.exists():
        return f'{path.name}: MISSING'
    data = path.read_text(errors='replace').splitlines()
    return '\n'.join(data[-lines:])


def start_monitor(baseline: int) -> subprocess.Popen:
    cmd = [str(PY), '-u', 'monitor_export_every10.py', '--db', 'production.db', '--baseline', str(baseline), '--target', str(TARGET), '--interval', str(EXPORT_INTERVAL), '--chunk', str(EXPORT_CHUNK)]
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=False)


def start_production() -> subprocess.Popen:
    env = os.environ.copy()
    env.update({
        'GPT_IMAGE_CONCURRENCY': '38',
        'LK888_IMAGE_CONCURRENCY_2': '20',
        'LK888_IMAGE_RAMP_STEPS_2': '28,36,42,48',
    })
    cmd = [str(PY), '-u', 'run_production_v7_queue.py', '--enable-db-recycle', '--recycle-gpt-workers', '2']
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=False)


def restart_after_stall(previous_count: int, stall_for: int):
    prod = pgrep('run_production_v7_queue.py')
    mon = pgrep('monitor_export_every10.py')
    log(f"STALL detected: final_pass={previous_count}, unchanged_for={stall_for}s, prod={prod}, monitor={mon}")
    CONTROL.write_text(json.dumps({
        'gpt_enabled': False,
        'regen_enabled': False,
        'qwen_enabled': True,
        'image_enabled': True,
        'drain_only': True,
        'qwen_high_watermark': 0.8,
        'qwen_low_watermark': 0.5,
        'image_high_watermark': 0.85,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f"runtime_control drain_only written; waiting {DRAIN_SECONDS}s before SIGTERM")
    time.sleep(DRAIN_SECONDS)
    stop_processes('run_production_v7_queue.py', graceful=True)
    stop_processes('monitor_export_every10.py', graceful=True)
    prod_gone = wait_absent('run_production_v7_queue.py', 60)
    mon_gone = wait_absent('monitor_export_every10.py', 60)
    forced = False
    if not prod_gone:
        forced = True
        stop_processes('run_production_v7_queue.py', graceful=False)
        prod_gone = wait_absent('run_production_v7_queue.py', 20)
    if not mon_gone:
        forced = True
        stop_processes('monitor_export_every10.py', graceful=False)
        mon_gone = wait_absent('monitor_export_every10.py', 20)
    baseline = final_pass_count()
    dist = status_distribution()
    CONTROL.unlink(missing_ok=True)
    monitor_proc = start_monitor(baseline)
    time.sleep(3)
    prod_proc = start_production()
    report = {
        'event': 'qbank_final_pass_stall_restart',
        'time': datetime.now().isoformat(),
        'stop_before_final_pass': previous_count,
        'restart_baseline': baseline,
        'stall_seconds': stall_for,
        'forced_kill_used': forced,
        'prod_gone': prod_gone,
        'monitor_gone': mon_gone,
        'new_monitor_pid': monitor_proc.pid,
        'new_production_pid': prod_proc.pid,
        'status_distribution': dist,
        'production_log_tail': tail(ROOT / 'production_v7_queue.log', 20),
        'monitor_log_tail': tail(ROOT / 'monitor_export_every10.log', 20),
    }
    log('RESTART_REPORT ' + json.dumps(report, ensure_ascii=False))
    return baseline


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    last = final_pass_count()
    last_growth = time.time()
    log(f"guard started baseline={last}, check_interval={CHECK_INTERVAL}s, stall_seconds={STALL_SECONDS}")
    while True:
        time.sleep(CHECK_INTERVAL)
        current = final_pass_count()
        prod_running = bool(pgrep('run_production_v7_queue.py'))
        if current > last:
            log(f"growth final_pass {last}->{current}")
            last = current
            last_growth = time.time()
            continue
        stalled_for = int(time.time() - last_growth)
        log(f"no_growth final_pass={current}, stalled_for={stalled_for}s, prod_running={prod_running}")
        if prod_running and stalled_for >= STALL_SECONDS:
            last = restart_after_stall(current, stalled_for)
            last_growth = time.time()


if __name__ == '__main__':
    main()
