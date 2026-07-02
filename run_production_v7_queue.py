#!/usr/bin/env python3
"""
V7 queue-based production pipeline.

Independent queues:
  KP queue -> 9 GPT workers -> Qwen queue -> 15 Qwen workers
  -> Image queue -> 38 gpt-image-2+quality_gate workers -> FINAL_PASS

Each image task writes to a unique {question_id}.png path. DB writes are guarded
by a single lock; API calls are isolated by worker threads.
"""
import argparse
import csv
import json
import os
import random
import re
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.db import ProductionDB
from pipeline.generator import QuestionGenerator
from pipeline.image_quality_gate import quality_gate
from pipeline.image_renderer import GPTImageRenderer
from pipeline.kp_enrichment import enrich_kp_for_image_prompt
from pipeline.orchestrator import Config, ProductionOrchestrator
from pipeline.reviewer import QwenReviewer

MEDICAL_EXCLUDED_SUBJECTS = {"S07", "S08", "S09", "S10", "S11", "S12"}  # 医学类：不生产、不读 KP、直接跳过
SKIP_SUBJECTS = set()  # 无暂弃科目
SKIP_KP_MODULES = {"S21-M11", "S21-M12"}  # DeepSeek 对这俩模块完全不响应：先跳过，最后处理
SUBJECTS = [
    "S01", "S02", "S03", "S04", "S05", "S06",
    "S13", "S14", "S15", "S16", "S17", "S18",
    "S19", "S20", "S21", "S22", "S23", "S24",
]

NUM_GPT_WORKERS = 12
RECYCLE_GPT_WORKERS_DEFAULT = 2
QWEN_ROLLOUTS = 5

QWEN_AUDIT_TRAIL_PATH = PROJECT_ROOT / "output" / "qwen36_audit_trail.jsonl"
_audit_trail_lock = threading.Lock()


def _write_audit_trail(question_id: str, subject_id: str, result: dict):
    entry = {
        "question_id": question_id,
        "subject_id": subject_id,
        "decision": result.get("decision", "?"),
        "wrong_count": result.get("wrong_count", 0),
        "correct_count": result.get("correct_count", 0),
        "answers": result.get("answers", []),
        "rollouts": result.get("rollouts", 0),
        "timestamp": datetime.utcnow().isoformat(),
    }
    with _audit_trail_lock:
        with open(QWEN_AUDIT_TRAIL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
QWEN_API_MAX_INFLIGHT = 8
QWEN_CONCURRENCY = QWEN_API_MAX_INFLIGHT
QWEN_QUEUE_MAXSIZE = 120
IMAGE_CONCURRENCY_DEFAULT = 38
IMAGE_CONCURRENCY_RAMP_MAX = 49
IMAGE_CONCURRENCY_RAMP_INTERVAL_SEC = 5 * 60
IMAGE_CONCURRENCY_RAMP_RETRIES = 3
LK888_IMAGE_RAMP_STEPS_DEFAULT = "20,28,36,42,48"
FALLBACK_THROTTLE_LIMIT = 4
BATCH_SIZE = 4
MAX_REGEN_ROUNDS = 2
MAX_QWEN_TECH_RETRIES = 3
MAX_EMPTY_GENERATIONS = 10
QWEN_TECH_FAILURE_SOURCES = {"exception", "http_error", "parse_error", "qwen_technical_failure"}
QWEN_TECH_FAILURE_MARKERS = (
    "timeout",
    "timed out",
    "http",
    "urlopen",
    "connection",
    "rate limit",
)
SAMPLE_EVERY_N = 100
SAMPLE_COUNT = 3
RECYCLE_QWEN_STATUSES = {"GENERATED"}
RECYCLE_IMAGE_STATUSES = {"ACCEPTED", "RENDER_FAIL", "RENDER_FAILED"}
RECYCLE_REGEN_STATUSES = {"SENTINEL_REGEN", "REGENERATE", "REPAIR_PENDING", "SENTINEL_FAIL_FINAL", "HOLD"}
RECYCLE_HOLD_MARKERS = ("qwen_tech", "http", "timeout", "parse", "control character", "控制字符")
RECYCLE_QUARANTINE_STATUS = "HOLD"
RECYCLE_QUARANTINE_MARKER = "recycle_quarantine"
ILLEGAL_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
LATEX_SIGNAL_RE = re.compile(r"\\[A-Za-z]+|\\[\[(]|\$\$|(?<!\\)\$(?!\$)")
LATEX_TEXT_FIELDS = (
    "question_text",
    "explanation",
    "image_prompt",
    "final_image_prompt",
    "image_dependency_reason",
)
QUEUE_PUT_TIMEOUT_SEC = 5
BACKPRESSURE_SLEEP_SEC = 2
QWEN_HIGH_WATERMARK = 0.85
QWEN_LOW_WATERMARK = 0.55
IMAGE_HIGH_WATERMARK = 0.85
RETRY_SCHEDULER_INTERVAL_SEC = 0.5
STALL_CHECK_INTERVAL_SEC = 60
STALL_WINDOW_COUNT = 5
RUNTIME_CONTROL_FILE = PROJECT_ROOT / "runtime_control.json"
PIPELINE_STATE_FILE = PROJECT_ROOT / "pipeline_state.json"


# ── 自适应节流：GPT 线程 + Qwen worker 联动 ──
class AdaptiveThrottle:
    """GPT 线程自适应节流阀：根据 qwen_queue 填充率动态调整活跃槽位"""

    def __init__(self, max_slots: int = 32):
        self._cond = threading.Condition()
        self._max_slots = max(1, min(32, max_slots))
        self._active = 0

    def set_max(self, n: int):
        with self._cond:
            new_max = max(1, min(32, n))
            old_max = self._max_slots
            self._max_slots = new_max
            # 只在扩容时唤醒，缩容时等待中的线程自然会被 while 条件拦住
            if new_max > old_max:
                self._cond.notify_all()

    def acquire(self, timeout: float = 60.0) -> bool:
        """获取一个槽位，超时返回 False。调用方应检查返回值并跳过本轮"""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._active >= self._max_slots:
                if stop_event.is_set():
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(timeout=min(remaining, 5.0))
            self._active += 1
            return True

    def release(self):
        with self._cond:
            if self._active <= 0:
                # 下溢保护：release 被多调时 fail-fast，避免永久容量膨胀
                raise RuntimeError("AdaptiveThrottle.release() called without matching acquire()")
            self._active -= 1
            self._cond.notify()


_qwen_active_slots = QWEN_CONCURRENCY
_qwen_slots_lock = threading.Lock()
_qwen_resize_event = threading.Event()

_image_active_slots = 0  # computed at startup from total image concurrency
_image_slots_lock = threading.Lock()
_image_resize_event = threading.Event()


def get_qwen_slots() -> int:
    with _qwen_slots_lock:
        return _qwen_active_slots


def set_qwen_slots(n: int):
    global _qwen_active_slots
    with _qwen_slots_lock:
        _qwen_active_slots = max(8, min(QWEN_CONCURRENCY, n))
    _qwen_resize_event.set()


def get_image_slots() -> int:
    with _image_slots_lock:
        return _image_active_slots


def set_image_slots(n: int):
    global _image_active_slots
    with _image_slots_lock:
        _image_active_slots = max(0, n)
    _image_resize_event.set()  # 唤醒休眠中的 worker 重新读取


def adaptive_controller(throttle: AdaptiveThrottle, qwen_queue, image_queue, stats: dict,
                        stats_lock: threading.Lock, max_image_concurrency: int = 0,
                        db: "ProductionDB | None" = None):
    """后台线程：每秒检查 Qwen + Image 队列，联动调节 GPT 节流阀、Qwen 工作线程数、Image 工作线程数
    带 hysteresis 避免阈值抖动：连续 3 次同一区间才切换
    
    三队列联动：取 qwen_queue 和 image_queue 填充率的最大值决定整体压力等级。
    Image worker 始终保持在 max（消费者不应被缩减），但高 image_queue 会通过 zone 提升
    来压制 GPT → Qwen 产线。"""
    last_zone = -1
    stable_count = 0
    image_slots_max = max_image_concurrency
    # ── GPT 动态降档：FINAL_PASS ≥ 22000 后 GPT 槽位砍半（Qwen/Image 不变）──
    GPT_HALVE_THRESHOLD = 99999  # 临时取消砍半：冲刺最后<1000条难产尾部，需要满配32槽火力(原值22000)
    _gpt_halve_active = False
    _last_fp_check = 0.0
    _fp_cached = 0
    while not stop_event.is_set():
        qwen_ratio = queue_fill_ratio(qwen_queue)
        image_ratio = queue_fill_ratio(image_queue)
        effective_ratio = max(qwen_ratio, image_ratio)
        if effective_ratio < 0.30:
            zone = 0
        elif effective_ratio < 0.50:
            zone = 1
        elif effective_ratio < 0.70:
            zone = 2
        elif effective_ratio < 0.80:
            zone = 3
        else:
            zone = 4

        if zone == last_zone:
            stable_count += 1
        else:
            last_zone = zone
            stable_count = 1

        # 连续 3 秒同一区间才执行调节（防抖动）
        if stable_count >= 3:
            levels = [
                (12, 32),  # zone 0: 空闲 → 全力
                (12, 24),  # zone 1: 轻度
                (12, 16),  # zone 2: 中度
                (8, 8),    # zone 3: Qwen 降压 + GPT 降速
                (8, 8),    # zone 4: 保命 (GPT≥8: each worker gets 1 slot)
            ]
            qwen_n, gpt_n = levels[min(zone, len(levels) - 1)]
            # ── GPT 砍半判断：每 30s 查一次全库 FINAL_PASS，≥阈值后 GPT 槽位减半 ──
            if db is not None:
                now = time.time()
                if now - _last_fp_check >= 30:
                    _last_fp_check = now
                    try:
                        with db_lock:
                            _fp_cached = db.count_final_pass_questions_total()
                    except Exception as e:
                        log(f"[GPT_HALVE] FINAL_PASS 查询失败(忽略): {e}")
                    new_active = _fp_cached >= GPT_HALVE_THRESHOLD
                    if new_active and not _gpt_halve_active:
                        log(f"[GPT_HALVE] FINAL_PASS={_fp_cached} ≥ {GPT_HALVE_THRESHOLD}，GPT 槽位开始砍半（Qwen 不变）")
                    _gpt_halve_active = new_active
            if _gpt_halve_active:
                gpt_n = max(4, gpt_n // 2)  # 砍半，地板 4
            # Image workers: always at max when queue has work (consumers should not be reduced)
            # Only trim when image_queue is nearly empty
            if image_slots_max > 0:
                if image_ratio < 0.05:
                    image_n = max(1, image_slots_max // 4)
                elif image_ratio < 0.15:
                    image_n = max(image_slots_max // 2, image_slots_max * 2 // 3)
                else:
                    image_n = image_slots_max
                set_image_slots(image_n)
            set_qwen_slots(qwen_n)
            throttle.set_max(gpt_n)
            with stats_lock:
                stats["qwen_active_slots"] = _qwen_active_slots
                stats["image_active_slots"] = _image_active_slots
                stats["gpt_max_slots"] = throttle._max_slots
                stats["gpt_active_slots"] = throttle._active
                stats["adaptive_zone"] = zone
                stats["adaptive_qwen_ratio"] = round(qwen_ratio, 3)
                stats["adaptive_image_ratio"] = round(image_ratio, 3)
        time.sleep(1)
# ── 自适应节流 结束 ──


LOG_FILE = PROJECT_ROOT / "production_v7_queue.log"
FEISHU_SAMPLE_DIR = PROJECT_ROOT / "feishu_samples"

log_lock = threading.Lock()
db_lock = threading.Lock()
pass_counter_lock = threading.Lock()
stop_event = Event()
pass_counter = 0
last_sample_at = 0
recent_passes = []


@dataclass
class Task:
    question: dict
    question_id: str
    kp_id: str
    subject_id: str
    kp_info: dict
    db: ProductionDB
    attempt: int = 0
    worker_id: int = -1
    existing: list = field(default_factory=list)


@dataclass
class RegenTask:
    failed_task: Task
    reason: str


class RuntimeControl:
    """Hot-readable control plane for queue workers.

    Operators can edit runtime_control.json while the process is running:
      {"gpt_enabled": false, "regen_enabled": true, "drain_only": false}
    Missing file/keys keep safe defaults.
    """

    DEFAULTS = {
        "gpt_enabled": True,
        "regen_enabled": True,
        "qwen_enabled": True,
        "image_enabled": True,
        "drain_only": False,
        "qwen_active_workers": QWEN_CONCURRENCY,
        "image_active_workers": 0,  # 0 = use all available (max concurrency)
        "qwen_high_watermark": QWEN_HIGH_WATERMARK,
        "qwen_low_watermark": QWEN_LOW_WATERMARK,
        "image_high_watermark": IMAGE_HIGH_WATERMARK,
    }

    def __init__(self, path: Path = RUNTIME_CONTROL_FILE):
        self.path = path
        self._data = dict(self.DEFAULTS)
        self._mtime = 0.0
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        self.reload_if_needed()
        with self._lock:
            return dict(self._data)

    def reload_if_needed(self):
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            return
        except Exception as exc:
            log(f"[CONTROL] cannot stat {self.path.name}: {str(exc)[:160]}")
            return
        if mtime <= self._mtime:
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("runtime control root must be object")
            merged = dict(self.DEFAULTS)
            merged.update(raw)
            with self._lock:
                self._data = merged
                self._mtime = mtime
            log(f"[CONTROL] reloaded {self.path.name}: {json.dumps(merged, ensure_ascii=False, sort_keys=True)}")
        except Exception as exc:
            log(f"[CONTROL] invalid {self.path.name}: {str(exc)[:200]}")

    def enabled(self, key: str) -> bool:
        return bool(self.snapshot().get(key, self.DEFAULTS.get(key, True)))


class RetryScheduler:
    def __init__(self):
        self._items = []
        self._lock = threading.Lock()

    def schedule(self, ready_at: float, queue: Queue, item, stage: str):
        with self._lock:
            self._items.append((ready_at, queue, item, stage))
            self._items.sort(key=lambda entry: entry[0])

    def due(self):
        now_ts = time.time()
        ready = []
        with self._lock:
            while self._items and self._items[0][0] <= now_ts:
                ready.append(self._items.pop(0))
        return ready

    def pending_count(self) -> int:
        with self._lock:
            return len(self._items)


runtime_control = RuntimeControl()
retry_scheduler = RetryScheduler()


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with log_lock:
        print(line, flush=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


def queue_fill_ratio(queue: Queue) -> float:
    maxsize = getattr(queue, "maxsize", 0) or 0
    if maxsize <= 0:
        return 0.0
    return min(1.0, qsize_safe(queue) / maxsize)


def queue_over_watermark(queue: Queue, watermark: float) -> bool:
    maxsize = getattr(queue, "maxsize", 0) or 0
    return maxsize > 0 and qsize_safe(queue) >= max(1, int(maxsize * watermark))


def wait_for_queue_below(queue: Queue, watermark: float, stage: str, stats: dict, stats_lock: threading.Lock) -> bool:
    logged = False
    while not stop_event.is_set() and queue_over_watermark(queue, watermark):
        control = runtime_control.snapshot()
        if control.get("drain_only", False):
            return False
        with stats_lock:
            stats[f"backpressure_wait_{stage}"] += 1
        if not logged:
            log(f"[BACKPRESSURE][{stage}] waiting: qsize={qsize_safe(queue)}, max={getattr(queue, 'maxsize', 0)}, watermark={watermark}")
            logged = True
        time.sleep(BACKPRESSURE_SLEEP_SEC)
    return not stop_event.is_set()


def wait_until_enabled(control_key: str, stage: str, stats: dict, stats_lock: threading.Lock) -> bool:
    logged = False
    while not stop_event.is_set():
        control = runtime_control.snapshot()
        if control.get("drain_only", False) and control_key in {"gpt_enabled", "regen_enabled"}:
            return False
        if control.get(control_key, True):
            return True
        with stats_lock:
            stats[f"runtime_paused_{stage}"] += 1
        if not logged:
            log(f"[CONTROL][{stage}] paused by runtime_control.{control_key}=false")
            logged = True
        time.sleep(BACKPRESSURE_SLEEP_SEC)
    return False


def safe_put(queue: Queue, item, stage: str, stats: dict | None = None,
             stats_lock: Any = None, timeout: float = QUEUE_PUT_TIMEOUT_SEC) -> bool:
    while not stop_event.is_set():
        try:
            queue.put(item, timeout=timeout)
            return True
        except Full:
            if stats is not None and stats_lock is not None:
                with stats_lock:
                    stats[f"queue_full_{stage}"] += 1
            log(f"[QUEUE_FULL][{stage}] qsize={qsize_safe(queue)}, max={getattr(queue, 'maxsize', 0)}; retrying")
            time.sleep(BACKPRESSURE_SLEEP_SEC)
    if stats is not None and stats_lock is not None:
        with stats_lock:
            stats[f"queue_put_aborted_{stage}"] += 1
    return False


def try_put_or_defer(queue: Queue, item, stage: str, stats: dict, stats_lock: threading.Lock,
                     reason: str = "db_state_preserved") -> bool:
    try:
        queue.put_nowait(item)
        return True
    except Full:
        with stats_lock:
            stats[f"queue_full_{stage}"] += 1
            stats[f"deferred_{stage}"] += 1
        log(
            f"[QUEUE_DEFER][{stage}] qsize={qsize_safe(queue)}, max={getattr(queue, 'maxsize', 0)}; "
            f"not blocking producer; reason={reason}"
        )
        return False


def retry_scheduler_worker(stats: dict, stats_lock: threading.Lock):
    while not stop_event.is_set() or retry_scheduler.pending_count() > 0:
        delivered = 0
        for _ready_at, queue, item, stage in retry_scheduler.due():
            if safe_put(queue, item, stage, stats, stats_lock, timeout=1):
                delivered += 1
            elif not stop_event.is_set():
                retry_scheduler.schedule(time.time() + BACKPRESSURE_SLEEP_SEC, queue, item, stage)
        if delivered:
            with stats_lock:
                stats["retry_scheduler_delivered"] += delivered
        time.sleep(RETRY_SCHEDULER_INTERVAL_SEC)


def write_pipeline_state(state: dict):
    try:
        tmp = PIPELINE_STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp.replace(PIPELINE_STATE_FILE)
    except Exception as exc:
        log(f"[STATE] failed to write {PIPELINE_STATE_FILE.name}: {str(exc)[:160]}")


def maybe_sample_to_feishu(question_info: dict):
    global pass_counter, last_sample_at, recent_passes
    with pass_counter_lock:
        pass_counter += 1
        recent_passes.append(question_info)
        if pass_counter - last_sample_at >= SAMPLE_EVERY_N:
            pool = recent_passes[-SAMPLE_EVERY_N:]
            samples = random.sample(pool, min(SAMPLE_COUNT, len(pool)))
            FEISHU_SAMPLE_DIR.mkdir(exist_ok=True)
            batch_id = f"sample_{pass_counter}"
            for i, sample in enumerate(samples):
                out_file = FEISHU_SAMPLE_DIR / f"{batch_id}_{i}.json"
                out_file.write_text(json.dumps(sample, ensure_ascii=False, indent=2))
            log(f"  📋 抽样: PASS={pass_counter}, 抽取{len(samples)}道 → feishu_samples/{batch_id}_*.json")
            last_sample_at = pass_counter
            if len(recent_passes) > 200:
                recent_passes = recent_passes[-100:]


def make_question_json(q: dict, subject_id: str, kp_id: str, kp_info: dict, image_path: str = "") -> dict:
    return {
        "question_id": q.get("question_id", ""),
        "subject_id": subject_id,
        "subject_name": kp_info.get("subject_name", ""),
        "kp_id": kp_id,
        "kp_name": kp_info.get("knowledge_point_name", "") or kp_info.get("kp_name", ""),
        "question_language": q.get("question_language", "zh"),
        "question_text": q.get("question_text", ""),
        "options": q.get("options", {}),
        "correct_answer": q.get("correct_answer", ""),
        "explanation": q.get("explanation", ""),
        "difficulty": q.get("difficulty", 0),
        "blueprint_slot": q.get("blueprint_slot"),
        "blueprint_archetype": q.get("blueprint_archetype", ""),
        "blueprint_image_type": q.get("blueprint_image_type", ""),
        "image_path": image_path,
        "image_prompt": q.get("image_prompt", ""),
        "final_image_prompt": QuestionGenerator.assemble_final_image_prompt(q.get("image_prompt", "")),
        "image_dependency_reason": q.get("image_dependency_reason", ""),
        "truth_spec": q.get("truth_spec", {}),
        "difference_from_others": q.get("difference_from_others", ""),
    }


def normalize_question(q: dict, subject_id: str, kp_id: str, kp_info: dict) -> dict:
    import uuid
    q.setdefault("subject_id", subject_id)
    q.setdefault("module_id", kp_id.rsplit("-", 1)[0] if "-" in kp_id else "")
    q.setdefault("kp_id", kp_id)
    q.setdefault("kp_name", kp_info.get("knowledge_point_name", "") or kp_info.get("kp_name", ""))
    if not q.get("question_id"):
        q["question_id"] = f"{kp_id}-Q{uuid.uuid4().hex[:6]}"
    q["question_json"] = make_question_json(q, subject_id, kp_id, kp_info)
    return q


def load_env():
    env_file = PROJECT_ROOT / "config" / ".env"
    if env_file.exists():
        for line in env_file.read_text().strip().split("\n"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ[key] = value


def get_gpt_worker_keys() -> list[tuple[str, str]]:
    return [
        ("GPT5_API_KEY", os.environ.get("GPT5_API_KEY", "")),
        ("GPT_WORKER1_API_KEY", os.environ.get("GPT_WORKER1_API_KEY", "")),
        ("GPT_WORKER2_API_KEY", os.environ.get("GPT_WORKER2_API_KEY", "")),
        ("GPT_WORKER3_API_KEY", os.environ.get("GPT_WORKER3_API_KEY", "")),
        ("GPT_WORKER4_API_KEY", os.environ.get("GPT_WORKER4_API_KEY", "")),
        ("GPT_WORKER5_API_KEY", os.environ.get("GPT_WORKER5_API_KEY", "")),
        ("GPT_WORKER6_API_KEY", os.environ.get("GPT_WORKER6_API_KEY", "")),
        ("GPT_WORKER7_API_KEY", os.environ.get("GPT_WORKER7_API_KEY", "")),
        ("GPT_WORKER8_API_KEY", os.environ.get("GPT_WORKER8_API_KEY", "")),
    ]


def get_deepseek_worker_keys() -> list[tuple[str, str]]:
    """Return 9 worker-name/key pairs for DeepSeek, sharing a single API key."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    return [
        (f"DEEPSEEK_WORKER{i}_API_KEY", api_key)
        for i in range(NUM_GPT_WORKERS)
    ]


def probe_one_gpt_worker(name: str, api_key: str, base_url: str, model: str, timeout_sec: int) -> dict:
    if not api_key:
        return {"worker": name, "ok": False, "error": "missing_key"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": "Reply exactly: {\"ok\":true}"},
        ],
        "max_tokens": 20,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {
            "worker": name,
            "ok": True,
            "status": "HTTP_200",
            "latency_sec": round(time.time() - started, 2),
            "returned_model": body.get("model", ""),
            "content_preview": str(content)[:80],
        }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300]
        return {
            "worker": name,
            "ok": False,
            "status": f"HTTP_{exc.code}",
            "latency_sec": round(time.time() - started, 2),
            "error": body,
        }
    except Exception as exc:
        return {
            "worker": name,
            "ok": False,
            "latency_sec": round(time.time() - started, 2),
            "error": str(exc)[:300],
        }


def require_gpt_api_health(worker_keys: list[tuple[str, str]], base_url: str, model: str, fail_threshold: float = 0.30) -> list[dict]:
    configured = [(name, key) for name, key in worker_keys if key]
    if not configured:
        raise RuntimeError("GPT API preflight failed: no GPT worker keys configured")
    results = []
    result_lock = threading.Lock()

    def run_probe(name: str, api_key: str):
        result = probe_one_gpt_worker(name, api_key, base_url, model, timeout_sec=60)
        with result_lock:
            results.append(result)

    threads = [Thread(target=run_probe, args=(name, key), name=f"gpt-preflight-{name}") for name, key in configured]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    results.sort(key=lambda item: item["worker"])

    ok_count = sum(1 for item in results if item.get("ok"))
    fail_count = len(results) - ok_count
    fail_rate = fail_count / len(results)
    for item in results:
        status = "OK" if item.get("ok") else "FAIL"
        detail = item.get("status") or item.get("error", "")
        log(f"[GPT_PREFLIGHT] {item['worker']}: {status} latency={item.get('latency_sec', '?')}s detail={detail}")
    log(f"[GPT_PREFLIGHT] summary ok={ok_count}/{len(results)} fail_rate={fail_rate:.1%} threshold={fail_threshold:.0%}")
    if fail_rate > fail_threshold:
        raise RuntimeError(f"GPT API preflight failed: fail_rate={fail_rate:.1%} > {fail_threshold:.0%}; production start blocked")
    return results


def patch_kp_loader(orch: ProductionOrchestrator):
    all_kp_details = {}
    population_csv = PROJECT_ROOT.parent / "population" / "full_18subject_kp_population.csv"
    if population_csv.exists():
        with population_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                kp_id = row.get("kp_id", "")
                if not kp_id:
                    continue
                all_kp_details[kp_id] = {
                    "knowledge_point_id": kp_id,
                    "kp_id": kp_id,
                    "subject_id": row.get("subject_id", ""),
                    "subject_name": row.get("subject_name", ""),
                    "knowledge_point_name": row.get("kp_name", ""),
                    "kp_name": row.get("kp_name", ""),
                    "module_id": row.get("module_id", ""),
                    "module_name": row.get("module_name", ""),
                    "importance": int(float(row.get("importance") or 0)),
                    "target_quota": float(row.get("target_quota") or 0),
                }
    input_dirs = [
        PROJECT_ROOT.parent / "input_snapshot",
        PROJECT_ROOT.parents[2] / "runs" / "remaining20_gate_20260618_2310" / "input_snapshot",
        PROJECT_ROOT.parents[2] / "runs" / "remaining20_gate_20260618_2308" / "input_snapshot",
        PROJECT_ROOT.parents[2] / "taxonomy_versions" / "v2_2_master" / "subjects",
    ]
    detailed_count = 0
    for input_dir in input_dirs:
        if input_dir.exists():
            log(f"Loading KP details from {input_dir}")
            for file_path in sorted(input_dir.glob("*.json")):
                try:
                    data = json.loads(file_path.read_text())
                    if "modules" in data:
                        for module in data["modules"]:
                            for kp in module.get("knowledge_points", []):
                                kp_id = kp["knowledge_point_id"]
                                all_kp_details.setdefault(kp_id, {}).update(kp)
                                detailed_count += 1
                    if "knowledge_points" in data:
                        for kp in data["knowledge_points"]:
                            kp_id = kp["knowledge_point_id"]
                            all_kp_details.setdefault(kp_id, {}).update(kp)
                            detailed_count += 1
                except Exception:
                    pass
    log(f"Loaded {len(all_kp_details)} KP base records; merged {detailed_count} detailed records")
    orig_get = orch.get_kp_info

    def patched_get(kp_id):
        base = orig_get(kp_id) or {}
        if kp_id in all_kp_details:
            base.update(all_kp_details[kp_id])
        return enrich_kp_for_image_prompt(base)

    orch.get_kp_info = patched_get


def count_subject_progress_for_mode(db: ProductionDB, subject_id: str, count_mode: str) -> int:
    if count_mode == "final_pass":
        return db.count_final_pass_questions_for_subject(subject_id)
    return db.count_total_questions_for_subject(subject_id)


def count_kp_progress_for_mode(db: ProductionDB, kp_id: str, count_mode: str) -> int:
    if count_mode == "final_pass":
        return db.count_questions_for_kp(kp_id)
    return db.count_total_questions_for_kp(kp_id)


def _pilot_taxonomy_kps_for_subject(subject_id: str) -> list[tuple[str, int]]:
    roots = [
        PROJECT_ROOT.parent / "input_snapshot",
        PROJECT_ROOT.parents[2] / "runs" / "remaining20_gate_20260618_2310" / "input_snapshot",
        PROJECT_ROOT.parents[2] / "runs" / "remaining20_gate_20260618_2308" / "input_snapshot",
        PROJECT_ROOT.parents[2] / "taxonomy_versions" / "v2_2_master" / "subjects",
    ]
    found: list[tuple[str, int]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            if data.get("subject_id") != subject_id:
                continue
            for module in data.get("modules", []):
                for kp in module.get("knowledge_points", []):
                    kp_id = kp.get("knowledge_point_id")
                    if kp_id and kp_id not in seen:
                        seen.add(kp_id)
                        found.append((kp_id, int(float(kp.get("production_quota") or kp.get("target_quota") or 1))))
            for kp in data.get("knowledge_points", []):
                kp_id = kp.get("knowledge_point_id")
                if kp_id and kp_id not in seen:
                    seen.add(kp_id)
                    found.append((kp_id, int(float(kp.get("production_quota") or kp.get("target_quota") or 1))))
    return found


def _distribute_subject_quota(total: int, weights: list[int]) -> list[int]:
    """Split a per-subject quota across selected KPs, preserving the total."""
    n = len(weights)
    if n == 0:
        return []
    if total <= 0:
        return [0] * n
    if n == 1:
        return [total]

    # Keep every selected KP alive with at least 1 item when possible.
    base = [1] * n if total >= n else [0] * n
    remaining = total - sum(base)
    if remaining <= 0:
        return base

    weight_sum = sum(max(0, w) for w in weights) or n
    raw = [remaining * max(0, w) / weight_sum for w in weights]
    alloc = [int(x) for x in raw]
    leftover = remaining - sum(alloc)
    frac_order = sorted(
        range(n),
        key=lambda i: (raw[i] - alloc[i], weights[i], -i),
        reverse=True,
    )
    for i in frac_order[:leftover]:
        alloc[i] += 1
    return [b + a for b, a in zip(base, alloc)]


def build_kp_queue(
    config: Config,
    db: ProductionDB | None = None,
    count_mode: str = "total",
    subject_ids: list[str] | None = None,
    subject_target_quota: int | None = None,
    kp_limit_per_subject: int = 2,
    include_medical_subjects: bool = False,
) -> tuple[Queue, int]:
    if count_mode not in {"total", "final_pass"}:
        raise ValueError(f"Unsupported count_mode={count_mode}")
    quotas = config.quotas.get("kp_quotas", {})
    kp_queue = Queue()
    total_target = 0
    skipped_medical = 0
    skipped_full_subjects = 0
    skipped_full_kps = 0
    queued_deficit = 0
    # Default: keep the existing production ordering.
    if subject_ids:
        ordered_subjects = [s for s in subject_ids if re.match(r"^S\d{2}$", s)]
    else:
        # KP 读取优先级：S24 优先 → S23 其次 → 其他未满科目按原顺序。
        # 仅影响 KP 入队顺序，配额/计数/其他逻辑不变。
        PRIORITY_SUBJECTS = ["S24", "S23"]
        ordered_subjects = PRIORITY_SUBJECTS + [s for s in SUBJECTS if s not in PRIORITY_SUBJECTS]
    custom_subject_mode = subject_target_quota is not None
    kp_limit_per_subject = max(1, kp_limit_per_subject)
    for subject_id in ordered_subjects:
        if subject_id in MEDICAL_EXCLUDED_SUBJECTS and not include_medical_subjects:
            skipped_medical += 1
            continue
        if subject_id in SKIP_SUBJECTS:
            log(f"[SKIP][{subject_id}] 暂弃科目，跳过")
            continue
        subject_kps = sorted([
            (kp_id, int(info.get("production_quota", 0) or 0))
            for kp_id, info in quotas.items()
            if info.get("subject_id") == subject_id
            and (include_medical_subjects or info.get("subject_id") not in MEDICAL_EXCLUDED_SUBJECTS)
            and info.get("production_quota", 0) > 0
        ])
        if not subject_kps and include_medical_subjects and custom_subject_mode:
            subject_kps = _pilot_taxonomy_kps_for_subject(subject_id)
            if subject_kps:
                log(f"[PILOT][{subject_id}] using taxonomy KP fallback: {len(subject_kps)} KPs")
        if not subject_kps:
            continue

        if custom_subject_mode:
            selected_kps = subject_kps[: min(kp_limit_per_subject, len(subject_kps))]
            selected_weights = [quota for _, quota in selected_kps]
            additional_quotas = _distribute_subject_quota(int(subject_target_quota), selected_weights)
            selected_quotas = []
            for (kp_id, _), additional in zip(selected_kps, additional_quotas):
                have = count_kp_progress_for_mode(db, kp_id, count_mode) if db is not None else 0
                selected_quotas.append(have + additional)
            subject_target = sum(selected_quotas)
        else:
            selected_kps = subject_kps
            selected_quotas = [quota for _, quota in subject_kps]
            subject_target = sum(selected_quotas)

        total_target += subject_target
        if db is not None and not custom_subject_mode:
            subject_have = count_subject_progress_for_mode(db, subject_id, count_mode)
            if subject_have >= subject_target:
                skipped_full_subjects += 1
                log(f"[RESUME][{subject_id}] {count_mode}记录 {subject_have}/{subject_target} 已占满，跳过整个科目")
                continue

        for (kp_id, _), quota in zip(selected_kps, selected_quotas):
            if kp_id.split("-M")[0] in MEDICAL_EXCLUDED_SUBJECTS and not include_medical_subjects:
                skipped_medical += 1
                continue
            have = count_kp_progress_for_mode(db, kp_id, count_mode) if db is not None else 0
            remaining = max(0, quota - have)
            if remaining <= 0:
                skipped_full_kps += 1
                continue
            kp_queue.put((kp_id, quota))
            queued_deficit += remaining
    log(
        f"KP队列: {kp_queue.qsize()} 个KP, 模式={count_mode}, 总目标≈{total_target}题, "
        f"待补≈{queued_deficit}题, 科目跳过={skipped_full_subjects}, "
        f"KP跳过={skipped_full_kps}, 医学跳过={skipped_medical}"
    )
    return kp_queue, total_target


def write_topup_dry_run_report(config: Config, db: ProductionDB) -> Path:
    quotas = config.quotas.get("kp_quotas", {})
    rows = []
    subject_summary = defaultdict(lambda: {"target": 0, "total_rows": 0, "final_pass": 0, "missing_final_pass": 0, "kp_count": 0, "kp_missing_count": 0})
    for kp_id, info in sorted(quotas.items()):
        subject_id = info.get("subject_id", "")
        quota = int(info.get("production_quota", 0) or 0)
        if subject_id in MEDICAL_EXCLUDED_SUBJECTS or kp_id.split("-M")[0] in MEDICAL_EXCLUDED_SUBJECTS or quota <= 0:
            continue
        total_rows = db.count_total_questions_for_kp(kp_id)
        final_pass = db.count_questions_for_kp(kp_id)
        missing = max(0, quota - final_pass)
        rows.append({
            "subject_id": subject_id,
            "kp_id": kp_id,
            "target": quota,
            "total_rows": total_rows,
            "final_pass": final_pass,
            "missing_final_pass": missing,
        })
        summary = subject_summary[subject_id]
        summary["target"] += quota
        summary["total_rows"] += total_rows
        summary["final_pass"] += final_pass
        summary["missing_final_pass"] += missing
        summary["kp_count"] += 1
        if missing > 0:
            summary["kp_missing_count"] += 1

    report = {
        "mode": "topup_final_pass_dry_run",
        "timestamp": datetime.now().isoformat(),
        "total_target": sum(row["target"] for row in rows),
        "total_rows": sum(row["total_rows"] for row in rows),
        "final_pass": sum(row["final_pass"] for row in rows),
        "missing_final_pass": sum(row["missing_final_pass"] for row in rows),
        "subjects": dict(sorted(subject_summary.items())),
        "missing_kps": [row for row in rows if row["missing_final_pass"] > 0],
    }
    report_path = PROJECT_ROOT / f"topup_final_pass_dry_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    log(
        f"TOPUP DRY-RUN: target={report['total_target']}, total_rows={report['total_rows']}, "
        f"final_pass={report['final_pass']}, missing_final_pass={report['missing_final_pass']}, "
        f"missing_kps={len(report['missing_kps'])}, report={report_path.name}"
    )
    return report_path


def qsize_safe(queue: Queue) -> int:
    try:
        return queue.qsize()
    except NotImplementedError:
        return -1


def load_question_payload(row: dict) -> dict:
    raw = row.get("question_json")
    if isinstance(raw, dict):
        question = dict(raw)
    else:
        try:
            question = json.loads(raw) if raw else {}
        except Exception:
            question = {}
    for key in ("question_id", "subject_id", "kp_id", "kp_name", "module_id", "image_path"):
        if row.get(key) and not question.get(key):
            question[key] = row.get(key)
    return question


def _append_text(value: Any, out: list[str]):
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for nested in value.values():
            _append_text(nested, out)
    elif isinstance(value, list):
        for nested in value:
            _append_text(nested, out)


def _latex_validation_text(question: dict) -> str:
    chunks: list[str] = []
    for field_name in LATEX_TEXT_FIELDS:
        _append_text(question.get(field_name, ""), chunks)
    _append_text(question.get("options", {}), chunks)
    _append_text(question.get("truth_spec", {}), chunks)
    return "\n".join(chunks)


def find_recycle_quarantine_reasons(row: dict) -> list[str]:
    raw = row.get("question_json") or ""
    reasons: list[str] = []
    if not str(raw).strip():
        reasons.append("empty_question_json")
        return reasons
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
    latex_text = _latex_validation_text(question)
    if not LATEX_SIGNAL_RE.search(latex_text):
        return reasons
    if latex_text.count("\\(") != latex_text.count("\\)"):
        reasons.append("unbalanced_latex_inline_delimiter")
    if latex_text.count("\\[") != latex_text.count("\\]"):
        reasons.append("unbalanced_latex_display_delimiter")
    if latex_text.count("$$") % 2:
        reasons.append("unbalanced_latex_double_dollar")
    return reasons


def quarantine_recycle_row(db: ProductionDB, question_id: str, reasons: list[str]):
    detail = f"{RECYCLE_QUARANTINE_MARKER}: " + "; ".join(reasons)
    db.update_question_status(question_id, RECYCLE_QUARANTINE_STATUS, detail[:500])


def is_recoverable_hold(row: dict) -> bool:
    text = " ".join(str(row.get(key, "")) for key in ("sentinel_result", "last_audit_detail", "question_json", "quality_status")).lower()
    if RECYCLE_QUARANTINE_MARKER in text:
        return False
    return any(marker in text for marker in RECYCLE_HOLD_MARKERS)


def classify_recycle_row(row: dict) -> str:
    status = str(row.get("quality_status", ""))
    if status in RECYCLE_IMAGE_STATUSES:
        return "image"
    if status in RECYCLE_QWEN_STATUSES or (status == "HOLD" and is_recoverable_hold(row)):
        return "qwen"
    if status in RECYCLE_REGEN_STATUSES:
        return "regen"
    return "skip"


def build_recycle_tasks(db: ProductionDB, orch: ProductionOrchestrator, limit: int = 0, allowed_statuses: set[str] | None = None) -> dict[str, deque]:
    statuses = sorted(
        allowed_statuses
        if allowed_statuses is not None
        else (RECYCLE_QWEN_STATUSES | RECYCLE_IMAGE_STATUSES | RECYCLE_REGEN_STATUSES | {"HOLD"})
    )
    placeholders = ",".join("?" for _ in statuses)
    sql = (
        "SELECT q.question_id, q.subject_id, q.module_id, q.kp_id, q.kp_name, q.batch_id, "
        "q.question_json, q.image_path, q.quality_status, q.sentinel_result, "
        "q.created_at, q.updated_at, COALESCE(a.detail, '') AS last_audit_detail "
        "FROM questions q "
        "LEFT JOIN ("
        "  SELECT question_id, MAX(id) AS max_id FROM audit_log GROUP BY question_id"
        ") latest ON latest.question_id = q.question_id "
        "LEFT JOIN audit_log a ON a.id = latest.max_id "
        f"WHERE q.quality_status IN ({placeholders}) "
        "ORDER BY q.updated_at, q.created_at"
    )
    params: list = list(statuses)
    if limit > 0:
        sql += " LIMIT ?"
        params.append(limit)
    with db_lock:
        if db.conn is None:
            raise RuntimeError("production DB connection is not initialized")
        rows = [dict(row) for row in db.conn.execute(sql, params).fetchall()]
    buckets = {"qwen": deque(), "image": deque(), "regen": deque(), "skip": deque()}
    quarantined_count = 0
    for row in rows:
        if RECYCLE_QUARANTINE_MARKER in str(row.get("last_audit_detail", "")):
            buckets["skip"].append(row)
            continue
        reasons = find_recycle_quarantine_reasons(row)
        if reasons:
            with db_lock:
                quarantine_recycle_row(db, row["question_id"], reasons)
            row["quality_status"] = RECYCLE_QUARANTINE_STATUS
            row["last_audit_detail"] = f"{RECYCLE_QUARANTINE_MARKER}: " + "; ".join(reasons)
            buckets["skip"].append(row)
            quarantined_count += 1
            continue
        route = classify_recycle_row(row)
        if route == "skip":
            buckets["skip"].append(row)
            continue
        kp_id = row.get("kp_id", "")
        subject_id = row.get("subject_id", "") or kp_id.split("-M")[0]
        if subject_id in MEDICAL_EXCLUDED_SUBJECTS or kp_id.split("-M")[0] in MEDICAL_EXCLUDED_SUBJECTS or subject_id in SKIP_SUBJECTS:
            buckets["skip"].append(row)
            continue
        kp_info = orch.get_kp_info(kp_id)
        question = load_question_payload(row)
        question = normalize_question(question, subject_id, kp_id, kp_info)
        question["question_id"] = row["question_id"]
        question["question_json"] = make_question_json(question, subject_id, kp_id, kp_info, question.get("image_path", ""))
        task = Task(question, row["question_id"], kp_id, subject_id, kp_info, db, 0, -1, [])
        if route == "regen":
            buckets[route].append(RegenTask(task, f"db_recycle:{row.get('quality_status', '')}"))
        else:
            buckets[route].append(task)
    if quarantined_count:
        log(f"[RECYCLE] quarantined malformed samples before queueing: count={quarantined_count}")
    return buckets


def enqueue_recycle_tasks(qwen_queue: Queue, image_queue: Queue, regen_queue: Queue,
                          recycle_buckets: dict[str, deque], stats: dict, stats_lock: threading.Lock):
    qwen_count = image_count = regen_count = 0
    while recycle_buckets["qwen"] and not stop_event.is_set():
        if not wait_for_queue_below(qwen_queue, QWEN_HIGH_WATERMARK, "recycle_qwen", stats, stats_lock):
            break
        if not safe_put(qwen_queue, recycle_buckets["qwen"].popleft(), "recycle_qwen", stats, stats_lock):
            break
        qwen_count += 1
    while recycle_buckets["image"] and not stop_event.is_set():
        if not try_put_or_defer(image_queue, recycle_buckets["image"].popleft(), "recycle_image", stats, stats_lock, reason="image_queue_full_recycle_deferred"):
            deferred = len(recycle_buckets["image"])
            recycle_buckets["skip"].extend(recycle_buckets["image"])
            recycle_buckets["image"].clear()
            log(f"[RECYCLE] deferred remaining image tasks via retry_scheduler, count={deferred + 1}")
            break
        image_count += 1
    while recycle_buckets["regen"] and not stop_event.is_set():
        if not try_put_or_defer(regen_queue, recycle_buckets["regen"].popleft(), "recycle_regen", stats, stats_lock):
            deferred = len(recycle_buckets["regen"])
            recycle_buckets["skip"].extend(recycle_buckets["regen"])
            recycle_buckets["regen"].clear()
            log(f"[RECYCLE] deferred remaining regen tasks to DB state, count={deferred + 1}")
            break
        regen_count += 1
    with stats_lock:
        stats["recycle_qwen"] += qwen_count
        stats["recycle_image"] += image_count
        stats["recycle_regen"] += regen_count
        stats["recycle_skipped"] += len(recycle_buckets["skip"])
    log(
        f"DB回收投递: qwen={qwen_count}, image={image_count}, regen={regen_count}, "
        f"skip={len(recycle_buckets['skip'])}"
    )


def should_retire_for_fallback(worker_id: int, image_renderer: GPTImageRenderer) -> bool:
    return image_renderer.name == "primary" and image_renderer.fallback_active and worker_id >= FALLBACK_THROTTLE_LIMIT


def record_fallback_retirement(worker_kind: str, worker_id: int, stats: dict, stats_lock: threading.Lock):
    key = f"fallback_retired_{worker_kind}"
    with stats_lock:
        stats[key] += 1
    log(f"[{worker_kind.upper()}{worker_id}] fallback active; retiring worker to enforce {FALLBACK_THROTTLE_LIMIT}-lane pipeline")


def is_qwen_candidate_rollout_mode(result: dict | None = None) -> bool:
    if os.environ.get("QWEN_REVIEW_MODE") == "candidate_5rollout":
        return True
    return bool(result and result.get("source") == "qwen_candidate_5rollout")


def format_qwen_rollout_detail(prefix: str, result: dict, order: str = "correct_wrong") -> str:
    answers = result.get("answers", [])
    if not isinstance(answers, list):
        answers = [str(answers)]
    if order == "wrong_correct":
        counts = f"wrong={result.get('wrong_count', 0)} correct={result.get('correct_count', 0)}"
    else:
        counts = f"correct={result.get('correct_count', 0)} wrong={result.get('wrong_count', 0)}"
    return f"{prefix} {counts} answers={','.join(str(answer) for answer in answers)}"[:500]


def record_qwen_rollout_stats(result: dict, stats: dict, stats_lock: threading.Lock):
    rollouts = int(result.get("rollouts") or QWEN_ROLLOUTS)
    valid_count = int(result.get("valid_count") or 0)
    wrong_count = int(result.get("wrong_count") or 0)
    wrong_count = max(0, min(QWEN_ROLLOUTS, wrong_count))
    with stats_lock:
        stats["qwen_rollout_calls_total"] += rollouts
        stats["qwen_rollout_valid_total"] += valid_count
        stats[f"qwen_wrong_count_{wrong_count}"] += 1



def qwen_issues_text(result: dict) -> str:
    issues = result.get("issues", [])
    if isinstance(issues, list):
        return "; ".join(str(issue) for issue in issues)
    return str(issues)


def is_qwen_technical_failure(result: dict) -> bool:
    source = str(result.get("source", "")).lower()
    if source in QWEN_TECH_FAILURE_SOURCES:
        return True
    text = f"{source} {qwen_issues_text(result)}".lower()
    return any(marker in text for marker in QWEN_TECH_FAILURE_MARKERS)


def handle_qwen_technical_failure(worker_id: int, task: Task, qwen_queue: Queue, result: dict,
                                  stats: dict, stats_lock: threading.Lock):
    issues = qwen_issues_text(result) or result.get("source", "qwen_technical_failure")
    retry_count = getattr(task, "qwen_tech_retries", 0)
    if retry_count < MAX_QWEN_TECH_RETRIES:
        setattr(task, "qwen_tech_retries", retry_count + 1)
        delay = min(30, 5 * (retry_count + 1))
        with db_lock:
            task.db.update_question_status(
                task.question_id,
                "GENERATED",
                f"qwen_tech_retry={retry_count + 1}/{MAX_QWEN_TECH_RETRIES}: {issues[:300]}",
            )
        with stats_lock:
            stats["qwen_tech_retry"] += 1
        log(f"[QWEN{worker_id}][{task.question_id}] TECH_FAIL retry {retry_count + 1}/{MAX_QWEN_TECH_RETRIES} scheduled in {delay}s: {issues[:80]}")
        retry_scheduler.schedule(time.time() + delay, qwen_queue, task, "qwen_retry")
        return
    with db_lock:
        task.db.update_question_status(task.question_id, "HOLD", f"qwen_tech_hold: {issues[:400]}")
    with stats_lock:
        stats["qwen_tech_hold"] += 1
    log(f"[QWEN{worker_id}][{task.question_id}] TECH_FAIL → HOLD, no regen: {issues[:80]}")


def gpt_worker(worker_id: int, generator: QuestionGenerator, kp_queue: Queue, qwen_queue: Queue,
               orch: ProductionOrchestrator, stats: dict, stats_lock: threading.Lock,
               image_renderer: GPTImageRenderer, count_mode: str, regen_queue: Queue | None = None):
    retired = False
    while not stop_event.is_set() or not kp_queue.empty():
        if should_retire_for_fallback(worker_id, image_renderer):
            record_fallback_retirement("gpt", worker_id, stats, stats_lock)
            return
        try:
            kp_id, kp_quota = kp_queue.get_nowait()
        except Empty:
            if regen_queue is not None:
                with stats_lock:
                    stats["gpt_workers_reassigned_to_regen"] += 1
                log(f"[GPT{worker_id}] KP queue empty; switching to regen_queue")
                recycle_regen_worker(worker_id, generator, regen_queue, qwen_queue, stats, stats_lock, image_renderer)
            return
        if should_retire_for_fallback(worker_id, image_renderer):
            safe_put(kp_queue, (kp_id, kp_quota), "gpt_retire_kp", stats, stats_lock)
            kp_queue.task_done()
            record_fallback_retirement("gpt", worker_id, stats, stats_lock)
            return
        subject_id = kp_id.split("-M")[0]
        module_prefix = "-".join(kp_id.split("-")[:2])  # e.g. "S21-M11"
        if module_prefix in SKIP_KP_MODULES:
            safe_put(kp_queue, (kp_id, kp_quota), "gpt_deferred_kp", stats, stats_lock)
            kp_queue.task_done()
            log(f"[GPT{worker_id}][{kp_id}] DEFERRED (module={module_prefix} in SKIP_KP_MODULES)")
            continue
        try:
            kp_info = orch.get_kp_info(kp_id)
            with db_lock:
                have = count_kp_progress_for_mode(orch.db, kp_id, count_mode)
                rows = orch.db.conn.execute(
                    "SELECT question_json FROM questions WHERE kp_id = ? AND quality_status = 'FINAL_PASS'",
                    (kp_id,),
                ).fetchall()
            existing = []
            for row in rows:
                try:
                    existing.append(json.loads(row[0]) if isinstance(row[0], str) else row[0])
                except Exception:
                    pass
            remaining = max(0, kp_quota - have)
            empty_streak = 0
            log(f"[GPT{worker_id}][{kp_id}] mode={count_mode}, quota={kp_quota}, have={have}, need={remaining}")
            while remaining > 0 and not stop_event.is_set():
                if should_retire_for_fallback(worker_id, image_renderer):
                    safe_put(kp_queue, (kp_id, kp_quota), "gpt_retire_kp", stats, stats_lock)
                    record_fallback_retirement("gpt", worker_id, stats, stats_lock)
                    return
                if not wait_until_enabled("gpt_enabled", f"gpt{worker_id}", stats, stats_lock):
                    safe_put(kp_queue, (kp_id, kp_quota), "gpt_paused_kp", stats, stats_lock)
                    return
                if not wait_for_queue_below(qwen_queue, float(runtime_control.snapshot().get("qwen_high_watermark", QWEN_HIGH_WATERMARK)), f"gpt{worker_id}_to_qwen", stats, stats_lock):
                    safe_put(kp_queue, (kp_id, kp_quota), "gpt_backpressure_kp", stats, stats_lock)
                    return
                batch_size = remaining  # 方案D: 一把梭全部remaining题，消除分批间隙
                questions = generator.generate_batch(kp_info, batch_size, existing)
                if not questions:
                    empty_streak += 1
                    if empty_streak >= MAX_EMPTY_GENERATIONS:
                        log(f"[GPT{worker_id}][{kp_id}] {MAX_EMPTY_GENERATIONS} consecutive empty generations; giving up KP, remaining={remaining}")
                        with stats_lock:
                            stats["gpt_empty_generation_hold"] += 1
                        return
                    log(f"[GPT{worker_id}][{kp_id}] empty generation (streak={empty_streak}/{MAX_EMPTY_GENERATIONS}), retry later")
                    time.sleep(5)
                    continue
                empty_streak = 0
                questions = [normalize_question(q, subject_id, kp_id, kp_info) for q in questions]
                with db_lock:
                    batch_id = orch.db.create_batch(subject_id, kp_id, len(questions))
                    qids = orch.db.add_questions(batch_id, questions)
                queued = 0
                for question, qid in zip(questions, qids):
                    task = Task(question, qid, kp_id, subject_id, kp_info, orch.db, 0, worker_id, existing)
                    if safe_put(qwen_queue, task, f"gpt{worker_id}_to_qwen", stats, stats_lock):
                        queued += 1
                    else:
                        break
                with stats_lock:
                    stats["total_generated"] += len(questions)
                    stats["queued_qwen"] += queued
                remaining -= queued
                if queued < len(questions):
                    log(f"[GPT{worker_id}][{kp_id}] queued={queued}/{len(questions)} before stop/backpressure")
                    return
                log(f"[GPT{worker_id}][{kp_id}] generated={len(questions)}, qwen_queue={qsize_safe(qwen_queue)}")
        except Exception as exc:
            log(f"[GPT{worker_id}][{kp_id}] ERROR: {str(exc)[:200]}")
            traceback.print_exc()
        finally:
            kp_queue.task_done()


def recycle_regen_worker(worker_id: int, generator: QuestionGenerator, regen_queue: Queue, qwen_queue: Queue,
                         stats: dict, stats_lock: threading.Lock,
                         image_renderer: GPTImageRenderer):
    while not stop_event.is_set() or not regen_queue.empty():
        if should_retire_for_fallback(worker_id, image_renderer):
            record_fallback_retirement("regen", worker_id, stats, stats_lock)
            return
        task_done = False
        try:
            regen_task = regen_queue.get(timeout=1)
        except Empty:
            continue
        if should_retire_for_fallback(worker_id, image_renderer):
            safe_put(regen_queue, regen_task, "regen_retire", stats, stats_lock)
            regen_queue.task_done()
            task_done = True
            record_fallback_retirement("regen", worker_id, stats, stats_lock)
            return
        if not wait_until_enabled("regen_enabled", f"regen{worker_id}", stats, stats_lock):
            try_put_or_defer(regen_queue, regen_task, "regen_paused", stats, stats_lock, reason="regen_paused_db_state_preserved")
            regen_queue.task_done()
            task_done = True
            return
        failed = regen_task.failed_task
        next_attempt = failed.attempt + 1
        try:
            if next_attempt > MAX_REGEN_ROUNDS:
                with db_lock:
                    failed.db.update_question_status(
                        failed.question_id,
                        "SENTINEL_FAIL_FINAL",
                        f"regen_limit_reached: {regen_task.reason[:360]}",
                    )
                with stats_lock:
                    stats["final_fail"] += 1
                continue
            retry_questions = generator.generate_batch(failed.kp_info, 1, failed.existing)
            if not retry_questions:
                try_put_or_defer(regen_queue, regen_task, "regen_empty_retry", stats, stats_lock, reason="empty_generation_retry_later")
                with stats_lock:
                    stats["regen_empty_retry"] += 1
                time.sleep(5)
                continue
            replacement = normalize_question(retry_questions[0], failed.subject_id, failed.kp_id, failed.kp_info)
            with db_lock:
                batch_id = failed.db.create_batch(failed.subject_id, failed.kp_id, 1)
                replacement_id = failed.db.add_questions(batch_id, [replacement])[0]
                # Defer DISCARDED: only discard old AFTER new is safely in qwen_queue
            task = Task(
                replacement,
                replacement_id,
                failed.kp_id,
                failed.subject_id,
                failed.kp_info,
                failed.db,
                next_attempt,
                worker_id,
                failed.existing,
            )
            if not wait_for_queue_below(qwen_queue, float(runtime_control.snapshot().get("qwen_high_watermark", QWEN_HIGH_WATERMARK)), f"regen{worker_id}_to_qwen", stats, stats_lock):
                # Backpressure: delay via retry_scheduler instead of wrapping as RegenTask
                # Both old (SENTINEL_REGEN) and new (GENERATED) stay in DB; recycle picks up either
                retry_scheduler.schedule(time.time() + BACKPRESSURE_SLEEP_SEC, qwen_queue, task, "regen_to_qwen_delayed")
                with stats_lock:
                    stats["regen_qwen_delayed"] += 1
                log(f"[REGEN{worker_id}][{failed.question_id}] -> {replacement_id} delayed via retry_scheduler (qwen backpressure)")
                return
            if not safe_put(qwen_queue, task, f"regen{worker_id}_to_qwen", stats, stats_lock):
                # Stop/failure: both old and new stay in DB; next recycle picks up
                log(f"[REGEN{worker_id}][{failed.question_id}] -> {replacement_id} qwen enqueue failed; DB state preserved for recycle")
                return
            # Success: now safe to discard old
            with db_lock:
                failed.db.update_question_status(
                    failed.question_id,
                    "DISCARDED",
                    f"recycled_by={replacement_id}; reason={regen_task.reason[:260]}",
                )
            with stats_lock:
                stats["regen_generated"] += 1
                stats["queued_qwen"] += 1
            log(f"[REGEN{worker_id}][{failed.question_id}] -> {replacement_id}, attempt={next_attempt}, qwen_queue={qsize_safe(qwen_queue)}")
        except Exception as exc:
            log(f"[REGEN{worker_id}][{failed.question_id}] ERROR: {str(exc)[:200]}")
        finally:
            if not task_done:
                regen_queue.task_done()


def qwen_worker(worker_id: int, qwen_queue: Queue, image_queue: Queue, regen_queue: Queue, reviewer: QwenReviewer,
                stats: dict, stats_lock: threading.Lock,
                image_renderer: GPTImageRenderer):
    while not stop_event.is_set() or not qwen_queue.empty():
        control = runtime_control.snapshot()
        active_workers = get_qwen_slots()
        if worker_id >= active_workers:
            with stats_lock:
                stats[f"qwen_worker_inactive_{worker_id}"] += 1
            _qwen_resize_event.wait(timeout=2)
            _qwen_resize_event.clear()
            continue
        if should_retire_for_fallback(worker_id, image_renderer):
            record_fallback_retirement("qwen", worker_id, stats, stats_lock)
            return
        task_done = False
        try:
            task = qwen_queue.get(timeout=1)
        except Empty:
            continue
        if should_retire_for_fallback(worker_id, image_renderer):
            safe_put(qwen_queue, task, "qwen_retire", stats, stats_lock)
            qwen_queue.task_done()
            task_done = True
            record_fallback_retirement("qwen", worker_id, stats, stats_lock)
            return
        if not wait_until_enabled("qwen_enabled", f"qwen{worker_id}", stats, stats_lock):
            safe_put(qwen_queue, task, "qwen_paused", stats, stats_lock)
            qwen_queue.task_done()
            task_done = True
            return
        try:
            review_question = task.question.get("question_json", task.question)
            if is_qwen_candidate_rollout_mode():
                result = reviewer.review_rollouts(review_question, task.kp_info, rollouts=QWEN_ROLLOUTS)
            else:
                result = reviewer.review(review_question, task.kp_info)
            decision = result.get("decision", "FAIL")
            candidate_rollout_mode = is_qwen_candidate_rollout_mode(result)
            with stats_lock:
                stats["total_reviewed"] += 1
            if candidate_rollout_mode:
                record_qwen_rollout_stats(result, stats, stats_lock)
                _write_audit_trail(task.question_id, task.subject_id, result)
                if decision == "PASS":
                    detail = format_qwen_rollout_detail("qwen36_rollout_pass", result, order="wrong_correct")
                    with db_lock:
                        task.db.update_question_status(task.question_id, "ACCEPTED", detail)
                    image_high_wm = float(runtime_control.snapshot().get("image_high_watermark", IMAGE_HIGH_WATERMARK))
                    if queue_over_watermark(image_queue, image_high_wm):
                        retry_scheduler.schedule(time.time() + BACKPRESSURE_SLEEP_SEC, image_queue, task, "qwen_to_image_delayed")
                        with stats_lock:
                            stats["queued_image_delayed"] += 1
                            stats["qwen_candidate_pass"] += 1
                    elif try_put_or_defer(image_queue, task, f"qwen{worker_id}_to_image", stats, stats_lock, reason="image_q_full"):
                        with stats_lock:
                            stats["queued_image"] += 1
                            stats["qwen_candidate_pass"] += 1
                    else:
                        retry_scheduler.schedule(time.time() + BACKPRESSURE_SLEEP_SEC, image_queue, task, "qwen_to_image_retry")
                        with stats_lock:
                            stats["queued_image_delayed"] += 1
                            stats["qwen_candidate_pass"] += 1
                elif result.get("technical_failure") or result.get("source") == "qwen_technical_failure" or is_qwen_technical_failure(result):
                    with stats_lock:
                        stats["qwen_rollout_technical_fail"] += 1
                    handle_qwen_technical_failure(worker_id, task, qwen_queue, result, stats, stats_lock)
                else:
                    detail = format_qwen_rollout_detail("qwen36_candidate_quality_fail", result)
                    with db_lock:
                        task.db.update_question_status(task.question_id, "SENTINEL_REGEN", detail)
                    regen_rt = RegenTask(task, detail)
                    if try_put_or_defer(regen_queue, regen_rt, f"qwen{worker_id}_candidate_to_regen",
                                        stats, stats_lock, reason="candidate_quality_fail_db_state_preserved"):
                        with stats_lock:
                            stats["queued_regen"] += 1
                            stats["qwen_candidate_quality_fail"] += 1
                        log(f"[QWEN{worker_id}][{task.question_id}] CANDIDATE_QUALITY_FAIL → regen_queue: {detail[:80]}")
                    else:
                        with stats_lock:
                            stats["qwen_candidate_quality_fail"] += 1
                        log(f"[QWEN{worker_id}][{task.question_id}] CANDIDATE_QUALITY_FAIL deferred: {detail[:80]}")
            elif decision == "PASS":
                with db_lock:
                    task.db.update_question_status(task.question_id, "ACCEPTED", f"qwen_conf={result.get('confidence', 0)},attempt={task.attempt}")
                # Non-blocking image_queue delivery: check watermark + try_put_or_defer
                # Never blocks the Qwen worker — if image_queue is full, defer to retry_scheduler
                image_high_wm = float(runtime_control.snapshot().get("image_high_watermark", IMAGE_HIGH_WATERMARK))
                if queue_over_watermark(image_queue, image_high_wm):
                    retry_scheduler.schedule(time.time() + BACKPRESSURE_SLEEP_SEC, image_queue, task, "qwen_to_image_delayed")
                    with stats_lock:
                        stats["queued_image_delayed"] += 1
                elif try_put_or_defer(image_queue, task, f"qwen{worker_id}_to_image", stats, stats_lock, reason="image_q_full"):
                    with stats_lock:
                        stats["queued_image"] += 1
                else:
                    retry_scheduler.schedule(time.time() + BACKPRESSURE_SLEEP_SEC, image_queue, task, "qwen_to_image_retry")
                    with stats_lock:
                        stats["queued_image_delayed"] += 1
            else:
                issues = qwen_issues_text(result) or result.get("source", "qwen_fail")
                if is_qwen_technical_failure(result):
                    handle_qwen_technical_failure(worker_id, task, qwen_queue, result, stats, stats_lock)
                elif task.attempt >= MAX_REGEN_ROUNDS:
                    with db_lock:
                        task.db.update_question_status(task.question_id, "SENTINEL_FAIL_FINAL", f"qwen_text_fail: {issues[:400]}")
                    with stats_lock:
                        stats["final_fail"] += 1
                else:
                    # ── Qwen self-revise: fix issues → image_queue ──
                    qwen_issues = result.get("issues", [])
                    revised = reviewer.revise(task.question, qwen_issues, task.kp_info)
                    if revised:
                        revised_q = normalize_question(revised, task.subject_id, task.kp_id, task.kp_info)
                        revised_q["question_id"] = task.question_id
                        task.question = revised_q
                        task.question["question_json"] = make_question_json(
                            revised_q, task.subject_id, task.kp_id, task.kp_info,
                            revised_q.get("image_path", ""))
                        with db_lock:
                            task.db.update_question_json(task.question_id, task.question["question_json"])
                            task.db.update_question_status(task.question_id, "ACCEPTED",
                                f"qwen_self_revised: {issues[:200]}")
                        if try_put_or_defer(image_queue, task, f"qwen{worker_id}_revised_to_image",
                                            stats, stats_lock, reason="revised_image_q_full"):
                            with stats_lock:
                                stats["queued_image"] += 1
                                stats["qwen_self_revised"] += 1
                            log(f"[QWEN{worker_id}][{task.question_id}] SELF_REVISED → image_queue: {issues[:80]}")
                        else:
                            retry_scheduler.schedule(time.time() + BACKPRESSURE_SLEEP_SEC,
                                                     image_queue, task, "qwen_revised_to_image_retry")
                            with stats_lock:
                                stats["queued_image_delayed"] += 1
                            log(f"[QWEN{worker_id}][{task.question_id}] SELF_REVISED delayed: {issues[:80]}")
                    else:
                        # Fallback: revision failed, keep existing regen path
                        with db_lock:
                            task.db.update_question_status(task.question_id, "SENTINEL_REGEN",
                                f"qwen_round={task.attempt}: {issues[:300]}")
                        regen_rt = RegenTask(task, issues)
                        if try_put_or_defer(regen_queue, regen_rt, f"qwen{worker_id}_to_regen",
                                            stats, stats_lock, reason="sentinel_regen_db_state_preserved"):
                            with stats_lock:
                                stats["queued_regen"] += 1
                            log(f"[QWEN{worker_id}][{task.question_id}] REVISE_FAILED → regen_queue: {issues[:80]}")
                        else:
                            log(f"[QWEN{worker_id}][{task.question_id}] REVISE_FAILED deferred: {issues[:80]}")
        except Exception as exc:
            log(f"[QWEN{worker_id}][{task.question_id}] ERROR: {str(exc)[:200]}")
            try:
                with db_lock:
                    task.db.update_question_status(task.question_id, "HOLD", f"qwen_worker_exception: {str(exc)[:300]}")
            except Exception:
                pass
        finally:
            if not task_done:
                qwen_queue.task_done()


def image_worker(worker_id: int, image_queue: Queue, image_renderer: GPTImageRenderer,
                 output_dir: Path, stats: dict, stats_lock: threading.Lock,
                 provider_name: str = "primary"):
    worker_label = f"{provider_name}-{worker_id}"
    pass_key = f"image_pass_{provider_name}"
    fail_key = f"image_fail_{provider_name}"
    while not stop_event.is_set() or not image_queue.empty():
        active_slots = get_image_slots()
        if active_slots > 0 and worker_id >= active_slots:
            with stats_lock:
                stats[f"image_worker_inactive_{provider_name}_{worker_id}"] += 1
            _image_resize_event.wait(timeout=2)
            _image_resize_event.clear()
            continue
        if should_retire_for_fallback(worker_id, image_renderer):
            record_fallback_retirement("image", worker_id, stats, stats_lock)
            return
        task_done = False
        try:
            task = image_queue.get(timeout=1)
        except Empty:
            continue
        if should_retire_for_fallback(worker_id, image_renderer):
            safe_put(image_queue, task, "image_retire", stats, stats_lock)
            image_queue.task_done()
            task_done = True
            record_fallback_retirement("image", worker_id, stats, stats_lock)
            return
        if not wait_until_enabled("image_enabled", f"image{worker_label}", stats, stats_lock):
            safe_put(image_queue, task, "image_paused", stats, stats_lock)
            image_queue.task_done()
            task_done = True
            return
        try:
            img_dir = output_dir / task.subject_id / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            img_path = str(img_dir / f"{task.question_id}.png")
            ok, msg = image_renderer.render(task.question.get("image_prompt", ""), img_path)
            if ok:
                gate_ok, gate_issues = quality_gate(img_path, {"engine": "gpt-image-2", "diagram_type": "image_prompt", "provider": provider_name})
                if gate_ok:
                    rel_path = img_path
                    task.question["image_path"] = rel_path
                    task.question["question_json"] = make_question_json(task.question, task.subject_id, task.kp_id, task.kp_info, rel_path)
                    with db_lock:
                        task.db.update_question_status(task.question_id, "FINAL_PASS", f"image_worker={worker_label}; {msg[:120]}")
                        task.db.update_question_json(task.question_id, task.question["question_json"])
                    maybe_sample_to_feishu({
                        "question_id": task.question_id,
                        "kp_id": task.kp_id,
                        "question_json": task.question["question_json"],
                        "image_path": rel_path,
                    })
                    with stats_lock:
                        stats["total_pass"] += 1
                        stats[pass_key] += 1
                else:
                    reason = f"quality_gate[{provider_name}]: " + "; ".join(gate_issues)
                    with db_lock:
                        task.db.update_question_status(task.question_id, "RENDER_FAIL", reason[:500])
                    with stats_lock:
                        stats["image_fail"] += 1
                        stats[fail_key] += 1
            else:
                with db_lock:
                    task.db.update_question_status(task.question_id, "RENDER_FAIL", f"{provider_name}: {msg[:480]}")
                with stats_lock:
                    stats["image_fail"] += 1
                    stats[fail_key] += 1
        except Exception as exc:
            log(f"[IMG-{worker_label}][{task.question_id}] ERROR: {str(exc)[:200]}")
            try:
                with db_lock:
                    task.db.update_question_status(task.question_id, "RENDER_FAIL", f"image_worker_exception: {str(exc)[:400]}")
            except Exception:
                pass
        finally:
            if not task_done:
                image_queue.task_done()


def current_image_worker_count(stats: dict, stats_lock: threading.Lock, provider_name: str = "primary") -> int:
    key = f"image_workers_current_{provider_name}"
    with stats_lock:
        return int(stats.get(key, 0) or 0)


def start_image_worker(worker_id: int, image_queue: Queue, image_renderer: GPTImageRenderer,
                       output_dir: Path, stats: dict, stats_lock: threading.Lock,
                       provider_name: str = "primary") -> Thread:
    thread = Thread(
        target=image_worker,
        args=(worker_id, image_queue, image_renderer, output_dir, stats, stats_lock, provider_name),
        name=f"image-{provider_name}-{worker_id}",
    )
    thread.start()
    return thread


def ramp_image_workers(image_threads: list, image_queue: Queue, image_renderer: GPTImageRenderer,
                       output_dir: Path, stats: dict, stats_lock: threading.Lock,
                       provider_name: str = "primary", ramp_steps: list[int] | None = None):
    if ramp_steps is None:
        ramp_steps = list(range(current_image_worker_count(stats, stats_lock, provider_name) + 1, IMAGE_CONCURRENCY_RAMP_MAX + 1))
    while ramp_steps and not stop_event.wait(IMAGE_CONCURRENCY_RAMP_INTERVAL_SEC):
        if image_renderer.fallback_active:
            log(f"[IMAGE_RAMP][{provider_name}] fallback active; keep image_workers={current_image_worker_count(stats, stats_lock, provider_name)}")
            continue
        current = current_image_worker_count(stats, stats_lock, provider_name)
        next_targets = [step for step in ramp_steps if step > current]
        if not next_targets:
            log(f"[IMAGE_RAMP][{provider_name}] reached target image_workers={current}; ramp complete")
            return
        target = next_targets[0]
        if target > IMAGE_CONCURRENCY_RAMP_MAX:
            target = IMAGE_CONCURRENCY_RAMP_MAX
        for attempt in range(1, IMAGE_CONCURRENCY_RAMP_RETRIES + 1):
            try:
                for worker_id in range(current, target):
                    thread = start_image_worker(worker_id, image_queue, image_renderer, output_dir, stats, stats_lock, provider_name)
                    image_threads.append(thread)
                with stats_lock:
                    stats[f"image_workers_current_{provider_name}"] = target
                log(f"[IMAGE_RAMP][{provider_name}] image workers {current} -> {target} succeeded on attempt {attempt}")
                break
            except Exception as exc:
                log(f"[IMAGE_RAMP][{provider_name}] image workers {current} -> {target} failed attempt {attempt}/{IMAGE_CONCURRENCY_RAMP_RETRIES}: {str(exc)[:200]}")
                time.sleep(5)
        else:
            log(f"[IMAGE_RAMP][{provider_name}] image workers stay at {current}; failed to expand to {target} after {IMAGE_CONCURRENCY_RAMP_RETRIES} attempts")


def monitor_worker(kp_queue: Queue, qwen_queue: Queue, image_queue: Queue, stats: dict,
                   stats_lock: threading.Lock, renderers: list[GPTImageRenderer]):
    stall_history = deque(maxlen=STALL_WINDOW_COUNT)
    while not stop_event.is_set():
        time.sleep(STALL_CHECK_INTERVAL_SEC)
        with stats_lock:
            snap = dict(stats)
        primary_renderer = renderers[0]
        fallback_reason = primary_renderer.fallback_reason if primary_renderer.fallback_active else ""
        provider_workers = ",".join(
            f"{renderer.name}={current_image_worker_count(stats, stats_lock, renderer.name)}"
            for renderer in renderers
        )
        if primary_renderer.fallback_active:
            effective = f"fallback {FALLBACK_THROTTLE_LIMIT}/{FALLBACK_THROTTLE_LIMIT}/{FALLBACK_THROTTLE_LIMIT}; providers={provider_workers}"
        else:
            effective = (
                f"primary gpt={snap.get('production_gpt_workers', NUM_GPT_WORKERS)}"
                f"+recycle={snap.get('recycle_gpt_workers', 0)}/{QWEN_CONCURRENCY}"
                f" img={_image_active_slots}"
                f"; providers={provider_workers}"
            )
        qwen_maxsize = getattr(qwen_queue, "maxsize", 0) or 0
        image_maxsize = getattr(image_queue, "maxsize", 0) or 0
        queue_snapshot = {
            "kp_q": qsize_safe(kp_queue),
            "qwen_q": qsize_safe(qwen_queue),
            "image_q": qsize_safe(image_queue),
            "qwen_maxsize": qwen_maxsize,
            "image_maxsize": image_maxsize,
            "qwen_fill_ratio": round(queue_fill_ratio(qwen_queue), 3),
            "qwen_rollouts_per_question": QWEN_ROLLOUTS,
            "qwen_effective_api_backlog": qsize_safe(qwen_queue) * QWEN_ROLLOUTS,
            "image_fill_ratio": round(queue_fill_ratio(image_queue), 3),
            "retry_pending": retry_scheduler.pending_count(),
            "gpt_max_slots": snap.get("gpt_max_slots", 32),
            "gpt_active_slots": snap.get("gpt_active_slots", 0),
            "qwen_max_workers": QWEN_CONCURRENCY,
            "qwen_active_workers": snap.get("qwen_active_slots", QWEN_CONCURRENCY),
            "image_active_workers": _image_active_slots,
            "production_gpt_workers": snap.get("production_gpt_workers", NUM_GPT_WORKERS),
            "recycle_gpt_workers": snap.get("recycle_gpt_workers", 0),
        }
        progress_key = (
            snap.get("total_pass", 0),
            snap.get("total_reviewed", 0),
            snap.get("queued_image", 0),
            snap.get("queued_regen", 0),
            snap.get("qwen_tech_retry", 0),
        )
        stall_history.append(progress_key)
        stalled = (
            len(stall_history) == STALL_WINDOW_COUNT
            and len(set(stall_history)) == 1
            and queue_over_watermark(qwen_queue, float(runtime_control.snapshot().get("qwen_high_watermark", QWEN_HIGH_WATERMARK)))
        )
        if stalled:
            with stats_lock:
                stats["stall_detected"] += 1
            log(
                f"[STALL] progress unchanged for {STALL_WINDOW_COUNT} checks and qwen_q high; "
                f"suggest drain_only or lower Qwen/GPT pressure. queues={queue_snapshot}"
            )
        state = {
            "timestamp": datetime.now().isoformat(),
            "mode": effective,
            "fallback_reason": fallback_reason,
            "queues": queue_snapshot,
            "stats": snap,
            "runtime_control": runtime_control.snapshot(),
            "stalled": stalled,
        }
        write_pipeline_state(state)
        log(
            f"PROGRESS kp_q={queue_snapshot['kp_q']} qwen_q={queue_snapshot['qwen_q']} "
            f"image_q={queue_snapshot['image_q']} retry_pending={queue_snapshot['retry_pending']} "
            f"qwen_rollouts_per_question={queue_snapshot['qwen_rollouts_per_question']} "
            f"qwen_effective_api_backlog={queue_snapshot['qwen_effective_api_backlog']} "
            f"mode={effective} fallback_reason={fallback_reason[:160]} stalled={stalled} stats={snap}"
        )


def parse_int_env(name: str, default: int, min_value: int = 0, max_value: int | None = None) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        log(f"{name}={raw!r} invalid; using {default}")
        return default
    if value < min_value:
        log(f"{name}={value} below min {min_value}; using {min_value}")
        value = min_value
    if max_value is not None and value > max_value:
        log(f"{name}={value} above max {max_value}; using {max_value}")
        value = max_value
    return value


def parse_ramp_steps(raw: str, start: int) -> list[int]:
    steps = []
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            log(f"Ignoring invalid ramp step {item!r}")
            continue
        if value > start:
            steps.append(min(value, IMAGE_CONCURRENCY_RAMP_MAX))
    return sorted(set(steps))


def build_image_renderers() -> tuple[list[tuple[str, GPTImageRenderer, int, list[int]]], int]:
    primary_concurrency = parse_int_env(
        "GPT_IMAGE_PRIMARY_CONCURRENCY",
        parse_int_env("GPT_IMAGE_CONCURRENCY", IMAGE_CONCURRENCY_DEFAULT, 1, IMAGE_CONCURRENCY_RAMP_MAX),
        1,
        IMAGE_CONCURRENCY_RAMP_MAX,
    )
    if primary_concurrency < IMAGE_CONCURRENCY_DEFAULT:
        log(f"primary image concurrency={primary_concurrency}; below legacy default but allowed for controlled restart")
    providers = [(
        "primary",
        GPTImageRenderer(name="primary"),
        primary_concurrency,
        list(range(primary_concurrency + 1, IMAGE_CONCURRENCY_RAMP_MAX + 1)),
    )]

    lk888_key = os.environ.get("LK888_IMAGE_API_KEY_2", "").strip()
    lk888_concurrency = parse_int_env("LK888_IMAGE_CONCURRENCY_2", 0, 0, IMAGE_CONCURRENCY_RAMP_MAX)
    if lk888_key and lk888_concurrency > 0:
        lk888_base_url = os.environ.get("LK888_IMAGE_BASE_URL_2", "https://api.lk888.ai/v1")
        lk888_model = os.environ.get("LK888_IMAGE_MODEL_2", "gpt-image-2")
        lk888_size = os.environ.get("LK888_IMAGE_SIZE_2") or os.environ.get("GPT_IMAGE_SIZE", "1024x1024")
        ramp_steps = parse_ramp_steps(os.environ.get("LK888_IMAGE_RAMP_STEPS_2", LK888_IMAGE_RAMP_STEPS_DEFAULT), lk888_concurrency)
        providers.append((
            "lk888_b",
            GPTImageRenderer(
                name="lk888_b",
                base_url=lk888_base_url,
                api_key=lk888_key,
                model=lk888_model,
                size=lk888_size,
                provider="lk888",
            ),
            lk888_concurrency,
            ramp_steps,
        ))
        log(f"LK888 second image provider enabled: concurrency={lk888_concurrency}, ramp_steps={ramp_steps or 'disabled'}, key_present=yes")
    elif lk888_concurrency > 0:
        log("LK888_IMAGE_CONCURRENCY_2 set but LK888_IMAGE_API_KEY_2 missing; second image provider disabled")
    return providers, sum(concurrency for _, _, concurrency, _ in providers)


def main():
    parser = argparse.ArgumentParser(description="V7 queue production with safe resume/topup modes")
    parser.add_argument(
        "--topup-final-pass",
        action="store_true",
        help="Build queues by FINAL_PASS deficits instead of total DB rows.",
    )
    parser.add_argument(
        "--dry-run-topup-report",
        action="store_true",
        help="Write a FINAL_PASS deficit report and exit without starting workers.",
    )
    parser.add_argument(
        "--enable-db-recycle",
        action="store_true",
        help="Recycle existing non-FINAL_PASS DB rows into the normal Qwen/Image/Regen queues.",
    )
    parser.add_argument(
        "--recycle-limit",
        type=int,
        default=0,
        help="Maximum existing DB rows to recycle this run; 0 means no limit.",
    )
    parser.add_argument(
        "--recycle-gpt-workers",
        type=int,
        default=RECYCLE_GPT_WORKERS_DEFAULT,
        help="Number of GPT keys reserved for regeneration/recycle workers.",
    )
    parser.add_argument(
        "--model",
        choices=("gpt", "deepseek"),
        default="gpt",
        help="GPT-side generator backend.",
    )
    parser.add_argument(
        "--no-new-kp",
        action="store_true",
        help="Disable fresh KP generation; only recycle/drain existing DB rows.",
    )
    parser.add_argument(
        "--subject-list",
        default="S01,S02,S03,S04,S05,S06,S13,S14,S15,S16,S17,S18,S19,S20",
        help="Comma-separated subject allowlist for KP reading, e.g. S01,S02.",
    )
    parser.add_argument(
        "--subject-target-quota",
        type=int,
        default=30,
        help="If >0, cap each listed subject to this many generated questions.",
    )
    parser.add_argument(
        "--kp-limit-per-subject",
        type=int,
        default=2,
        help="Maximum KPs to read per subject when --subject-target-quota is used.",
    )
    parser.add_argument(
        "--include-medical-subjects",
        action="store_true",
        help="Pilot mode: include S07-S12 with synthetic/simulated medical diagrams.",
    )
    args = parser.parse_args()
    count_mode = "final_pass" if args.topup_final_pass or args.dry_run_topup_report else "total"
    run_mode = "topup_final_pass" if count_mode == "final_pass" else "safe_resume_total_rows"

    load_env()
    image_provider_specs, total_image_concurrency = build_image_renderers()
    primary_renderer = image_provider_specs[0][1]
    image_renderers = [renderer for _, renderer, _, _ in image_provider_specs]
    log("=" * 60)
    log(f"V7 QUEUE PRODUCTION — {NUM_GPT_WORKERS} GPT + {QWEN_CONCURRENCY} Qwen + image queue")
    log(f"Mode={run_mode}, count_mode={count_mode}")
    log(
        f"GPT={NUM_GPT_WORKERS}, Qwen={QWEN_CONCURRENCY}, "
        f"ImageTotal={total_image_concurrency}, Providers="
        + ",".join(f"{name}:{concurrency}" for name, _, concurrency, _ in image_provider_specs)
        + f", Batch={BATCH_SIZE}"
    )
    log("=" * 60)

    config = Config()
    orch = ProductionOrchestrator(config)
    patch_kp_loader(orch)

    if args.dry_run_topup_report:
        report_path = write_topup_dry_run_report(config, orch.db)
        log(f"Dry-run only; no production workers started. report={report_path}")
        return

    reviewer = QwenReviewer(config.qwen_base_url, config.qwen_api_key, config.qwen_model)
    image_renderer = GPTImageRenderer()

    gpt_base_url = os.environ.get("GPT5_BASE_URL", "https://api.lk888.ai/v1")
    gpt_model = "gpt-5.5"
    if args.model == "deepseek":
        gpt_base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://yuanlansj.xin/v1")
        gpt_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
        gpt_worker_keys = get_deepseek_worker_keys()
        log(f"[MODEL] deepseek-v4-pro @ {gpt_base_url} (NUM_KEYS={len(set(k for _, k in gpt_worker_keys if k))})")
    else:
        gpt_worker_keys = get_gpt_worker_keys()
        log(f"[MODEL] gpt-5.5 @ {gpt_base_url} (NUM_KEYS={len(set(k for _, k in gpt_worker_keys if k))})")
    os.environ["GPT_MODEL"] = gpt_model
    require_gpt_api_health(gpt_worker_keys, gpt_base_url, gpt_model)
    gpt_keys = [key for _, key in gpt_worker_keys if key]
    recycle_gpt_workers = max(1, min(args.recycle_gpt_workers, len(gpt_keys))) if gpt_keys else 0
    production_gpt_workers = max(0, min(NUM_GPT_WORKERS - recycle_gpt_workers, len(gpt_keys) - recycle_gpt_workers))
    if production_gpt_workers <= 0 and recycle_gpt_workers > 0 and not args.enable_db_recycle:
        production_gpt_workers = min(1, len(gpt_keys))
        recycle_gpt_workers = max(0, min(args.recycle_gpt_workers, len(gpt_keys) - production_gpt_workers))
    production_generators = [
        QuestionGenerator(
            base_url=gpt_base_url,
            api_key=key,
            model=gpt_model,
            max_concurrent=4,
            response_log_dir=str(PROJECT_ROOT / f"api_responses_w{i}"),
        )
        for i, key in enumerate(gpt_keys[:production_gpt_workers])
    ]
    recycle_generators = [
        QuestionGenerator(
            base_url=gpt_base_url,
            api_key=key,
            model=gpt_model,
            max_concurrent=4,
            response_log_dir=str(PROJECT_ROOT / f"api_responses_recycle_w{i}"),
        )
        for i, key in enumerate(gpt_keys[production_gpt_workers:production_gpt_workers + recycle_gpt_workers])
    ]

    if args.no_new_kp:
        kp_queue = Queue()
        total_target = 0
        log("[NO_NEW_KP] KP queue disabled by operator; this run will only recycle/drain existing DB rows.")
    else:
        subject_list = [s.strip() for s in args.subject_list.split(",") if s.strip()]
        subject_target_quota = args.subject_target_quota if args.subject_target_quota > 0 else None
        kp_queue, total_target = build_kp_queue(
            config,
            orch.db,
            count_mode=count_mode,
            subject_ids=subject_list or None,
            subject_target_quota=subject_target_quota,
            kp_limit_per_subject=args.kp_limit_per_subject,
            include_medical_subjects=args.include_medical_subjects,
        )
    qwen_queue = Queue(maxsize=QWEN_QUEUE_MAXSIZE)
    regen_queue = Queue(maxsize=max(200, max(1, recycle_gpt_workers) * BATCH_SIZE * 8))
    image_queue = Queue(maxsize=max(500, total_image_concurrency * 4))
    stats = defaultdict(int)
    stats["start_time"] = time.time()
    stats["total_target"] = total_target
    stats["production_gpt_workers"] = production_gpt_workers
    stats["recycle_gpt_workers"] = recycle_gpt_workers
    stats["qwen_candidate_pass"] = 0
    stats["qwen_candidate_quality_fail"] = 0
    stats["qwen_rollout_calls_total"] = 0
    stats["qwen_rollout_valid_total"] = 0
    stats["qwen_rollout_technical_fail"] = 0
    for wrong_count in range(QWEN_ROLLOUTS + 1):
        stats[f"qwen_wrong_count_{wrong_count}"] = 0
    for provider_name, _, concurrency, _ in image_provider_specs:
        stats[f"image_workers_current_{provider_name}"] = concurrency
    stats_lock = threading.Lock()
    log(f"GPT worker split: production={production_gpt_workers}, recycle_regen={recycle_gpt_workers}, db_recycle={args.enable_db_recycle}")

    def signal_handler(signum, frame):
        log(f"Signal {signum} received, stopping...")
        stop_event.set()
        orch.running = False

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    threads = []
    image_threads = []
    threads.append(Thread(target=retry_scheduler_worker, args=(stats, stats_lock), name="retry-scheduler", daemon=True))
    for i, gen in enumerate(production_generators):
        threads.append(Thread(target=gpt_worker, args=(i, gen, kp_queue, qwen_queue, orch, stats, stats_lock, primary_renderer, count_mode, regen_queue), name=f"gpt-{i}"))
    for i, gen in enumerate(recycle_generators):
        threads.append(Thread(target=recycle_regen_worker, args=(i, gen, regen_queue, qwen_queue, stats, stats_lock, primary_renderer), name=f"regen-{i}"))
    for i in range(QWEN_CONCURRENCY):
        threads.append(Thread(target=qwen_worker, args=(i, qwen_queue, image_queue, regen_queue, reviewer, stats, stats_lock, primary_renderer), name=f"qwen-{i}"))
    for provider_name, renderer, concurrency, ramp_steps in image_provider_specs:
        provider_threads = []
        for i in range(concurrency):
            provider_threads.append(start_image_worker(i, image_queue, renderer, config.output_dir, stats, stats_lock, provider_name))
        image_threads.extend(provider_threads)
        if ramp_steps:
            threads.append(Thread(
                target=ramp_image_workers,
                args=(provider_threads, image_queue, renderer, config.output_dir, stats, stats_lock, provider_name, ramp_steps),
                name=f"image-ramp-{provider_name}",
                daemon=True,
            ))
    threads.append(Thread(target=monitor_worker, args=(kp_queue, qwen_queue, image_queue, stats, stats_lock, image_renderers), name="monitor", daemon=True))

    # ── 自适应节流：创建并绑定 ──
    throttle = AdaptiveThrottle(max_slots=32)
    for gen in production_generators:
        gen._throttle = throttle
    for gen in recycle_generators:
        gen._throttle = throttle
    # Initialize image slots to full concurrency (all workers active at startup)
    global _image_active_slots
    _image_active_slots = total_image_concurrency
    threads.append(Thread(target=adaptive_controller, args=(throttle, qwen_queue, image_queue, stats, stats_lock, total_image_concurrency, orch.db), name="adaptive-ctrl", daemon=True))
    log(f"[ADAPTIVE] throttle initialized: max_slots=32, qwen_max={QWEN_CONCURRENCY}, image_max={total_image_concurrency}")
    # ── 自适应节流 结束 ──

    for thread in threads:
        thread.start()

    # Auto-recover orphaned questions AFTER workers start (queues have consumers)
    # --- GENERATED orphans → qwen_queue ---
    orphan_count = quarantined_orphan_count = 0
    with db_lock:
        orphan_rows = orch.db.conn.execute(
            "SELECT question_id, subject_id, kp_id, question_json FROM questions WHERE quality_status = 'GENERATED' ORDER BY created_at LIMIT 500"
        ).fetchall()
    for row in orphan_rows:
        if stop_event.is_set():
            break
        qid, subject_id, kp_id = row[0], row[1], row[2]
        reasons = find_recycle_quarantine_reasons({"question_id": qid, "question_json": row[3]})
        if reasons:
            with db_lock:
                quarantine_recycle_row(orch.db, qid, reasons)
            quarantined_orphan_count += 1
            continue
        question = load_question_payload({"question_id": qid, "subject_id": subject_id, "kp_id": kp_id, "question_json": row[3]})
        kp_info = orch.get_kp_info(kp_id) if kp_id else {}
        question = normalize_question(question, subject_id, kp_id, kp_info)
        task = Task(question, qid, kp_id, subject_id, kp_info, orch.db, 0, -1, [])
        if safe_put(qwen_queue, task, "orphan_recovery", stats, stats_lock):
            orphan_count += 1
        else:
            break
    if orphan_count or quarantined_orphan_count:
        log(f"[STARTUP] auto-enqueued {orphan_count} orphaned GENERATED questions (quarantined={quarantined_orphan_count}) into qwen_queue")

    # --- ACCEPTED orphans → image_queue ---
    accepted_count = accepted_quarantined = 0
    with db_lock:
        accepted_rows = orch.db.conn.execute(
            "SELECT question_id, subject_id, kp_id, question_json FROM questions WHERE quality_status = 'ACCEPTED' ORDER BY created_at LIMIT 500"
        ).fetchall()
    for row in accepted_rows:
        if stop_event.is_set():
            break
        qid, subject_id, kp_id = row[0], row[1], row[2]
        reasons = find_recycle_quarantine_reasons({"question_id": qid, "question_json": row[3]})
        if reasons:
            with db_lock:
                quarantine_recycle_row(orch.db, qid, reasons)
            accepted_quarantined += 1
            continue
        question = load_question_payload({"question_id": qid, "subject_id": subject_id, "kp_id": kp_id, "question_json": row[3]})
        kp_info = orch.get_kp_info(kp_id) if kp_id else {}
        question = normalize_question(question, subject_id, kp_id, kp_info)
        task = Task(question, qid, kp_id, subject_id, kp_info, orch.db, 0, -1, [])
        if safe_put(image_queue, task, "orphan_accepted", stats, stats_lock):
            accepted_count += 1
        else:
            break
    if accepted_count or accepted_quarantined:
        log(f"[STARTUP] auto-enqueued {accepted_count} orphaned ACCEPTED questions (quarantined={accepted_quarantined}) into image_queue")

    if args.enable_db_recycle:
        recycle_buckets = build_recycle_tasks(orch.db, orch, max(0, args.recycle_limit))
        enqueue_recycle_tasks(qwen_queue, image_queue, regen_queue, recycle_buckets, stats, stats_lock)

    # Wait for GPT to finish all KP dispatch, then repeatedly drain regen and Qwen
    # because Qwen can enqueue new regen tasks while it is being drained.
    kp_queue.join()
    log("All KP generation tasks dispatched. Draining regen/Qwen queues...")
    while True:
        regen_queue.join()
        qwen_queue.join()
        pending = retry_scheduler.pending_count()
        if regen_queue.empty() and qwen_queue.empty() and pending == 0:
            break
        if pending:
            log(f"Drain loop: retry_scheduler has {pending} pending tasks, waiting...")
            time.sleep(RETRY_SCHEDULER_INTERVAL_SEC * 2)
        log(f"Drain loop continues: regen_q={qsize_safe(regen_queue)}, qwen_q={qsize_safe(qwen_queue)}, retry_pending={pending}")
    log("All regen and Qwen tasks finished. Waiting image queue...")
    image_queue.join()
    stop_event.set()

    for thread in threads:
        if thread.is_alive() and not thread.daemon:
            thread.join(timeout=5)
    for thread in image_threads:
        if thread.is_alive() and not thread.daemon:
            thread.join(timeout=5)

    elapsed_min = (time.time() - stats["start_time"]) / 60
    report = {
        "pipeline": "v7_queue_gpt_qwen_image_gate",
        "elapsed_minutes": round(elapsed_min, 1),
        "stats": dict(stats),
        "pass_counter": pass_counter,
        "timestamp": datetime.now().isoformat(),
    }
    report_path = PROJECT_ROOT / "production_v7_queue_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"QUEUE PRODUCTION COMPLETE: {json.dumps(report, ensure_ascii=False, default=str)}")


if __name__ == "__main__":
    main()
