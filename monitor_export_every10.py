#!/usr/bin/env python3
import argparse
import json
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def choose_desktop_test():
    users = Path('/mnt/c/Users')
    if users.exists():
        for user_dir in users.iterdir():
            if user_dir.name.lower() in {'public', 'default', 'default user', 'all users'}:
                continue
            desktop = user_dir / 'Desktop'
            if desktop.exists():
                target = desktop / 'test'
                target.mkdir(parents=True, exist_ok=True)
                return target
    target = Path.home() / 'Desktop' / 'test'
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_state(path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {'baseline': None, 'exported_question_ids': [], 'last_report_total': None, 'chunks': 0}


def save_state(path, state):
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')


def rows_since_baseline(conn, baseline_created_at, baseline_ids, limit):
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT question_id, subject_id, module_id, kp_id, kp_name, batch_id,
               question_json, image_path, quality_status, render_engine,
               created_at, updated_at
        FROM questions
        WHERE quality_status = 'FINAL_PASS'
        ORDER BY datetime(created_at), question_id
        """
    ).fetchall()
    new_rows = []
    baseline_ids = set(baseline_ids)
    for row in rows:
        if row['question_id'] in baseline_ids:
            continue
        new_rows.append(row)
    return new_rows[:limit] if limit else new_rows


def sanitize(value):
    return ''.join(c if c.isalnum() or c in {'-', '_'} else '_' for c in str(value))[:80]


def export_row(row, export_dir, seq):
    qid = row['question_id']
    kp = row['kp_id']
    stem = f"{seq:05d}_{sanitize(qid)}_{sanitize(kp)}"
    payload = dict(row)
    json_path = export_dir / f"{stem}.json"
    parsed = None
    try:
        parsed = json.loads(row['question_json']) if row['question_json'] else None
    except Exception:
        parsed = row['question_json']
    payload['question_json_parsed'] = parsed
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    image_src = row['image_path']
    if (not image_src) and isinstance(parsed, dict):
        image_src = parsed.get('image_path') or parsed.get('image') or parsed.get('figure_path')
    image_dst = None
    image_missing = False
    if image_src:
        src = Path(image_src)
        if not src.is_absolute():
            src = Path.cwd() / src
        if src.exists():
            suffix = src.suffix or '.png'
            image_dst = export_dir / f"{stem}{suffix}"
            shutil.copy2(src, image_dst)
        else:
            image_missing = True
    else:
        image_missing = True
    return str(json_path), str(image_dst) if image_dst else None, image_missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='production.db')
    parser.add_argument('--baseline', type=int, default=508)
    parser.add_argument('--target', type=int, default=24000)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--chunk', type=int, default=10)
    args = parser.parse_args()

    root = Path.cwd()
    db = root / args.db
    export_dir = choose_desktop_test()
    state_path = root / 'monitor_export_every10_state.json'
    report_path = root / 'monitor_export_every10_latest_report.txt'
    log_path = root / 'monitor_export_every10.log'
    state = load_state(state_path)

    while True:
        try:
            conn = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
            conn.row_factory = sqlite3.Row
            total = conn.execute("SELECT COUNT(*) FROM questions WHERE quality_status='FINAL_PASS'").fetchone()[0]
            if state.get('baseline') is None:
                baseline_rows = conn.execute(
                    "SELECT question_id FROM questions WHERE quality_status='FINAL_PASS' ORDER BY datetime(created_at), question_id LIMIT ?",
                    (args.baseline,),
                ).fetchall()
                state['baseline'] = args.baseline
                state['baseline_question_ids'] = [r['question_id'] for r in baseline_rows]
                state['exported_question_ids'] = []
                state['chunks'] = 0
                save_state(state_path, state)

            new_rows = rows_since_baseline(conn, None, state.get('baseline_question_ids', []), None)
            exported = set(state.get('exported_question_ids', []))
            pending = [r for r in new_rows if r['question_id'] not in exported]
            messages = []

            while len(pending) >= args.chunk:
                chunk_rows = pending[:args.chunk]
                exported_files = []
                missing = []
                start_seq = len(exported) + 1
                for offset, row in enumerate(chunk_rows):
                    json_file, image_file, image_missing = export_row(row, export_dir, start_seq + offset)
                    exported_files.append(Path(json_file).name)
                    if image_file:
                        exported_files.append(Path(image_file).name)
                    if image_missing:
                        missing.append(row['question_id'])
                    exported.add(row['question_id'])
                state['exported_question_ids'] = sorted(exported)
                state['chunks'] = int(state.get('chunks', 0)) + 1
                state['last_report_total'] = total
                save_state(state_path, state)
                msg = {
                    'time': now(),
                    'total_final_pass': total,
                    'new_since_508': max(0, total - args.baseline),
                    'export_dir': str(export_dir),
                    'chunk_index': state['chunks'],
                    'exported_count_total': len(exported),
                    'this_chunk_questions': [r['question_id'] for r in chunk_rows],
                    'this_chunk_file_count': len(exported_files),
                    'missing_images': missing,
                    'observation': '前100条观察窗口内' if total - args.baseline <= 100 else '100条后继续生产中',
                }
                messages.append(msg)
                pending = pending[args.chunk:]

            status = {
                'time': now(),
                'total_final_pass': total,
                'new_since_508': max(0, total - args.baseline),
                'export_dir': str(export_dir),
                'exported_questions_total': len(exported),
                'pending_new_not_yet_exported': len(pending),
                'target': args.target,
                'messages': messages,
            }
            report_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
            with log_path.open('a', encoding='utf-8') as f:
                if messages:
                    for msg in messages:
                        f.write(json.dumps(msg, ensure_ascii=False) + '\n')
                else:
                    f.write(json.dumps({k: status[k] for k in ['time','total_final_pass','new_since_508','exported_questions_total','pending_new_not_yet_exported']}, ensure_ascii=False) + '\n')
            conn.close()
            if total >= args.target:
                break
        except Exception as exc:
            err = {'time': now(), 'error': repr(exc)}
            report_path.write_text(json.dumps(err, ensure_ascii=False, indent=2), encoding='utf-8')
            with log_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(err, ensure_ascii=False) + '\n')
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
