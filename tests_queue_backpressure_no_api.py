#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import time
from collections import defaultdict
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import cast

import run_production_v7_queue as m
from pipeline.db import ProductionDB


class FakeRows(list):
    def fetchall(self):
        return self


class FakeDB:
    def __init__(self):
        self.status = []
        self.next_id = 0
        self.conn = self

    def execute(self, *args, **kwargs):
        return FakeRows()

    def count_questions_for_kp(self, kp_id):
        return 0

    def count_total_questions_for_kp(self, kp_id):
        return 0

    def count_final_pass_questions_for_subject(self, subject_id):
        return 0

    def count_total_questions_for_subject(self, subject_id):
        return 0

    def create_batch(self, subject_id, kp_id, n):
        return f"batch-{self.next_id}"

    def add_questions(self, batch_id, questions):
        ids = []
        for _ in questions:
            self.next_id += 1
            ids.append(f"q{self.next_id}")
        return ids

    def update_question_status(self, question_id, status, notes=""):
        self.status.append((question_id, status, notes))

    def update_question_json(self, question_id, question_json):
        pass


class FakeGenerator:
    def __init__(self):
        self.calls = 0

    def generate_batch(self, kp_info, batch_size, existing):
        self.calls += 1
        return [
            {
                "question_text": f"q-{self.calls}-{i}",
                "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
                "correct_answer": "A",
                "explanation": "ok",
                "difficulty": 1,
                "image_prompt": "A clean simple educational diagram with two labeled boxes and one arrow.",
            }
            for i in range(batch_size)
        ]


class FakeOrch:
    def __init__(self, db):
        self.db = db

    def get_kp_info(self, kp_id):
        return {"kp_name": "kp", "knowledge_point_name": "kp", "subject_name": "subject"}


class Renderer:
    name = "primary"
    fallback_active = False
    fallback_reason = ""

    def render(self, prompt, path):
        Path(path).write_bytes(b"fake")
        return True, "fake"


def reset():
    m.stop_event.clear()
    m.retry_scheduler = m.RetryScheduler()
    m.runtime_control = m.RuntimeControl(Path(tempfile.mkdtemp()) / "runtime_control.json")
    m.QUEUE_PUT_TIMEOUT_SEC = 0.05
    m.BACKPRESSURE_SLEEP_SEC = 0.02
    m.RETRY_SCHEDULER_INTERVAL_SEC = 0.01
    m.QWEN_HIGH_WATERMARK = 0.5
    m.IMAGE_HIGH_WATERMARK = 0.5
    m.STALL_CHECK_INTERVAL_SEC = 0.05
    m.STALL_WINDOW_COUNT = 3


def test_safe_put_stops_on_full_queue():
    reset()
    q = Queue(maxsize=1)
    q.put("filled")
    stats = defaultdict(int)
    lock = Lock()
    result = []
    t = Thread(target=lambda: result.append(m.safe_put(q, "x", "unit", stats, lock, timeout=0.02)))
    t.start()
    time.sleep(0.08)
    assert t.is_alive(), "safe_put should wait without throwing while full"
    m.stop_event.set()
    t.join(timeout=1)
    assert result == [False], result
    assert stats["queue_full_unit"] >= 1, dict(stats)
    print("OK safe_put_stops_on_full_queue", dict(stats))


def test_qwen_technical_retry_is_delayed_not_sleeping_worker():
    reset()
    db = FakeDB()
    qwen_q = Queue(maxsize=1)
    task = m.Task({"question_json": {}}, "q1", "S01-M01-001", "S01", {}, cast(ProductionDB, db))
    stats = defaultdict(int)
    lock = Lock()
    start = time.time()
    m.handle_qwen_technical_failure(0, task, qwen_q, {"decision": "FAIL", "source": "http_error", "issues": ["timeout"]}, stats, lock)
    elapsed = time.time() - start
    assert elapsed < 1.0, elapsed
    assert m.retry_scheduler.pending_count() == 1
    assert qwen_q.qsize() == 0
    print("OK qwen_technical_retry_is_delayed", dict(stats), db.status)


def test_gpt_backpressure_waits_instead_of_overfilling_qwen():
    reset()
    db = FakeDB()
    orch = FakeOrch(db)
    kp_q = Queue()
    kp_q.put(("S01-M01-001", 4))
    qwen_q = Queue(maxsize=2)
    qwen_q.put("filled1")
    qwen_q.put("filled2")
    regen_q = Queue(maxsize=4)
    stats = defaultdict(int)
    lock = Lock()
    t = Thread(target=m.gpt_worker, args=(0, FakeGenerator(), kp_q, qwen_q, orch, stats, lock, Renderer(), "final_pass", None))
    t.start()
    time.sleep(0.12)
    assert t.is_alive(), "gpt_worker should be waiting on backpressure, not crashed"
    assert stats["backpressure_wait_gpt0_to_qwen"] >= 1, dict(stats)
    m.stop_event.set()
    t.join(timeout=1)
    print("OK gpt_backpressure_waits", dict(stats))


def test_monitor_detects_stall_and_writes_state():
    reset()
    tempdir = Path(tempfile.mkdtemp())
    m.PIPELINE_STATE_FILE = tempdir / "pipeline_state.json"
    kp_q = Queue()
    qwen_q = Queue(maxsize=2)
    image_q = Queue(maxsize=2)
    qwen_q.put("a")
    qwen_q.put("b")
    stats = defaultdict(int)
    lock = Lock()
    thread = Thread(target=m.monitor_worker, args=(kp_q, qwen_q, image_q, stats, lock, [Renderer()]))
    thread.start()
    time.sleep(0.25)
    m.stop_event.set()
    thread.join(timeout=1)
    assert stats["stall_detected"] >= 1, dict(stats)
    assert m.PIPELINE_STATE_FILE.exists()
    print("OK monitor_detects_stall", dict(stats), m.PIPELINE_STATE_FILE)


def test_regen_defer_does_not_block_on_full_queue():
    reset()
    q = Queue(maxsize=1)
    q.put("filled")
    stats = defaultdict(int)
    lock = Lock()
    start = time.time()
    ok = m.try_put_or_defer(q, "regen-task", "qwen0_to_regen", stats, lock)
    elapsed = time.time() - start
    assert ok is False
    assert elapsed < 0.2, elapsed
    assert stats["deferred_qwen0_to_regen"] == 1, dict(stats)
    assert q.qsize() == 1
    print("OK regen_defer_does_not_block", dict(stats))


if __name__ == "__main__":
    test_safe_put_stops_on_full_queue()
    test_qwen_technical_retry_is_delayed_not_sleeping_worker()
    test_gpt_backpressure_waits_instead_of_overfilling_qwen()
    test_monitor_detects_stall_and_writes_state()
    test_regen_defer_does_not_block_on_full_queue()
    print("ALL_QUEUE_BACKPRESSURE_TESTS_OK")
