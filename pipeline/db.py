"""
V5 题库生产流水线 - SQLite 状态管理模块
Production Pipeline State Management via SQLite

管理24000道多模态试题从生成、渲染到质检的全流程状态跟踪。
Tracks 24,000 multimodal questions across 18 subjects through generation,
rendering, and quality review stages.
"""

import sqlite3
import json
import uuid
import threading
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Optional


# 题目质量状态枚举值
QUALITY_STATUSES = (
    'GENERATED',       # 已生成，待渲染
    'RENDERED',        # 已渲染，待质检
    'SENTINEL_PASS',   # 哨兵模型质检通过
    'SENTINEL_FAIL',   # 哨兵模型质检失败
    'SENTINEL_REGEN',  # 哨兵审核FAIL，正在重生成
    'SENTINEL_FAIL_FINAL',  # 重生成N轮后仍FAIL
    'REGENERATE',      # 标记为需重生成
    'REPAIR_PENDING',  # 等待修复
    'FINAL_PASS',      # 最终通过
    'FINAL_FAIL',      # 最终失败
    'HOLD',            # 暂挂（人工复核）
    'ACCEPTED',        # V7: Qwen审核通过
    'FROZEN_FAIL',     # V7: 重生成2次后仍FAIL，冻结
    'DISCARDED',       # V7: 丢弃（已被重生成替代）
    'RENDER_FAILED',   # V7: 渲染失败
    'RENDER_FAIL',     # 渲染失败（orchestrator用）
)

# 批次状态枚举值
BATCH_STATUSES = (
    'PENDING', 'IN_PROGRESS', 'GENERATED', 'RENDERED',
    'REVIEWED', 'COMPLETE', 'FAILED',
)

# 渲染任务状态枚举值
RENDER_JOB_STATUSES = ('PENDING', 'RUNNING', 'SUCCESS', 'FAIL')


# ─── SQL Schema ───────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS questions (
    question_id   TEXT PRIMARY KEY,
    subject_id    TEXT NOT NULL,
    module_id     TEXT,
    kp_id         TEXT NOT NULL,
    kp_name       TEXT,
    batch_id      TEXT NOT NULL,
    question_json TEXT,
    image_path    TEXT,
    quality_status TEXT NOT NULL DEFAULT 'GENERATED'
        CHECK(quality_status IN ('GENERATED','RENDERED','SENTINEL_PASS','SENTINEL_FAIL','SENTINEL_REGEN','SENTINEL_FAIL_FINAL','REGENERATE','REPAIR_PENDING','FINAL_PASS','FINAL_FAIL','HOLD','ACCEPTED','FROZEN_FAIL','DISCARDED','RENDER_FAILED','RENDER_FAIL')),
    render_engine TEXT,
    render_code   TEXT,
    render_attempts INTEGER DEFAULT 0,
    sentinel_result TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS batches (
    batch_id     TEXT PRIMARY KEY,
    subject_id   TEXT NOT NULL,
    kp_id        TEXT NOT NULL,
    size         INTEGER NOT NULL,
    status       TEXT NOT NULL DEFAULT 'PENDING'
        CHECK(status IN ('PENDING','IN_PROGRESS','GENERATED','RENDERED','REVIEWED','COMPLETE','FAILED')),
    created_at   TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS render_jobs (
    job_id       TEXT PRIMARY KEY,
    question_id  TEXT NOT NULL REFERENCES questions(question_id),
    engine       TEXT NOT NULL,
    code         TEXT,
    status       TEXT NOT NULL DEFAULT 'PENDING'
        CHECK(status IN ('PENDING','RUNNING','SUCCESS','FAIL')),
    attempts     INTEGER DEFAULT 0,
    error_log    TEXT,
    output_path  TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id  TEXT NOT NULL,
    action       TEXT NOT NULL,
    old_status   TEXT,
    new_status   TEXT,
    detail       TEXT,
    timestamp    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS production_stats (
    subject_id   TEXT PRIMARY KEY,
    total_target INTEGER DEFAULT 0,
    generated    INTEGER DEFAULT 0,
    rendered     INTEGER DEFAULT 0,
    passed       INTEGER DEFAULT 0,
    failed       INTEGER DEFAULT 0,
    hold         INTEGER DEFAULT 0,
    last_updated TEXT
);
"""

# 索引：加速常见查询（按状态筛选、按科目筛选、按批次关联）
_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_questions_status ON questions(quality_status);
CREATE INDEX IF NOT EXISTS idx_questions_subject ON questions(subject_id);
CREATE INDEX IF NOT EXISTS idx_questions_subject_status ON questions(subject_id, quality_status);
CREATE INDEX IF NOT EXISTS idx_questions_batch ON questions(batch_id);
CREATE INDEX IF NOT EXISTS idx_questions_kp ON questions(kp_id);
CREATE INDEX IF NOT EXISTS idx_batches_subject ON batches(subject_id);
CREATE INDEX IF NOT EXISTS idx_batches_status ON batches(status);
CREATE INDEX IF NOT EXISTS idx_render_jobs_status ON render_jobs(status);
CREATE INDEX IF NOT EXISTS idx_render_jobs_question ON render_jobs(question_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_question ON audit_log(question_id);
"""


def _now_iso() -> str:
    """返回当前UTC时间的ISO格式字符串"""
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _gen_id() -> str:
    """生成短UUID作为主键"""
    return uuid.uuid4().hex[:16]


class ProductionDB:
    """
    V5 题库生产流水线状态数据库

    线程安全，使用WAL模式以支持并发读写。
    所有写操作通过上下文管理器确保事务完整性。
    """

    def __init__(self, db_path: str):
        """
        初始化数据库连接，创建表和索引。

        Args:
            db_path: SQLite数据库文件路径，传 ':memory:' 可用于测试
        """
        self.db_path = db_path
        self._lock = threading.Lock()  # 线程安全锁
        # check_same_thread=False 允许多线程访问同一连接
        self.conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            timeout=30.0,
        )
        self.conn.row_factory = sqlite3.Row
        # 启用WAL模式：提高并发读写性能
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        """创建所有表和索引（幂等操作）"""
        with self._transaction() as cur:
            cur.executescript(_SCHEMA_SQL)
            cur.executescript(_INDEX_SQL)

    @contextmanager
    def _transaction(self):
        """
        事务上下文管理器：自动提交或回滚。
        确保每个写操作的原子性。线程安全。
        """
        with self._lock:
            cur = self.conn.cursor()
            try:
                yield cur
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
            finally:
                cur.close()

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None

    # ─── Batch Operations ─────────────────────────────────────────────────

    def create_batch(self, subject_id: str, kp_id: str, size: int) -> str:
        """
        创建一个新的生产批次。

        Args:
            subject_id: 科目ID
            kp_id: 知识点ID
            size: 本批次计划生成的题目数量

        Returns:
            batch_id: 新批次的唯一标识
        """
        batch_id = _gen_id()
        now = _now_iso()
        with self._transaction() as cur:
            cur.execute(
                "INSERT INTO batches (batch_id, subject_id, kp_id, size, status, created_at) "
                "VALUES (?, ?, ?, ?, 'PENDING', ?)",
                (batch_id, subject_id, kp_id, size, now),
            )
        return batch_id

    def get_batch_status(self, batch_id: str) -> Optional[dict]:
        """获取批次状态详情"""
        cur = self.conn.execute(
            "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    # ─── Question Operations ──────────────────────────────────────────────

    def add_questions(self, batch_id: str, questions: list) -> list:
        """
        批量插入题目记录。

        Args:
            batch_id: 所属批次ID
            questions: 题目字典列表，每个至少包含:
                - subject_id, kp_id, kp_name
                - 可选: module_id, question_json, image_path

        Returns:
            新插入题目的question_id列表
        """
        now = _now_iso()
        ids = []
        with self._transaction() as cur:
            for q in questions:
                qid = _gen_id()
                # question_json 若为dict则序列化
                qjson = q.get('question_json')
                if isinstance(qjson, dict):
                    qjson = json.dumps(qjson, ensure_ascii=False)
                cur.execute(
                    """INSERT INTO questions
                    (question_id, subject_id, module_id, kp_id, kp_name, batch_id,
                     question_json, image_path, quality_status, render_attempts,
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'GENERATED', 0, ?, ?)""",
                    (
                        qid,
                        q['subject_id'],
                        q.get('module_id', ''),
                        q['kp_id'],
                        q.get('kp_name', ''),
                        batch_id,
                        qjson,
                        q.get('image_path', ''),
                        now,
                        now,
                    ),
                )
                ids.append(qid)
            # 同步更新批次状态为 IN_PROGRESS
            cur.execute(
                "UPDATE batches SET status = 'IN_PROGRESS' WHERE batch_id = ? AND status = 'PENDING'",
                (batch_id,),
            )
        return ids

    def update_question_status(self, question_id: str, new_status: str, detail: str = ''):
        """
        更新题目质量状态，并自动记录审计日志。

        状态流转约束由CHECK约束保证合法值，
        业务层负责流转逻辑的合理性。
        """
        if new_status not in QUALITY_STATUSES:
            raise ValueError(f"Invalid status: {new_status}. Must be one of {QUALITY_STATUSES}")

        now = _now_iso()
        with self._transaction() as cur:
            # 先获取旧状态用于审计
            cur.execute(
                "SELECT quality_status FROM questions WHERE question_id = ?",
                (question_id,),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Question not found: {question_id}")
            old_status = row['quality_status']

            # 更新状态
            cur.execute(
                "UPDATE questions SET quality_status = ?, updated_at = ? WHERE question_id = ?",
                (new_status, now, question_id),
            )
            # 写审计日志
            cur.execute(
                "INSERT INTO audit_log (question_id, action, old_status, new_status, detail, timestamp) "
                "VALUES (?, 'STATUS_CHANGE', ?, ?, ?, ?)",
                (question_id, old_status, new_status, detail, now),
            )

    def get_questions_by_status(
        self, status: str, subject_id: Optional[str] = None, limit: int = 100
    ) -> list:
        """
        按状态查询题目，可选科目过滤。

        Returns:
            题目字典列表
        """
        if subject_id:
            cur = self.conn.execute(
                "SELECT * FROM questions WHERE quality_status = ? AND subject_id = ? "
                "ORDER BY created_at LIMIT ?",
                (status, subject_id, limit),
            )
        else:
            cur = self.conn.execute(
                "SELECT * FROM questions WHERE quality_status = ? "
                "ORDER BY created_at LIMIT ?",
                (status, limit),
            )
        return [dict(row) for row in cur.fetchall()]

    def get_subject_progress(self, subject_id: str) -> dict:
        """
        获取单个科目的生产进度：按状态分组计数。

        Returns:
            {'GENERATED': n, 'RENDERED': n, ..., 'total': N}
        """
        cur = self.conn.execute(
            "SELECT quality_status, COUNT(*) as cnt FROM questions "
            "WHERE subject_id = ? GROUP BY quality_status",
            (subject_id,),
        )
        result = {s: 0 for s in QUALITY_STATUSES}
        total = 0
        for row in cur.fetchall():
            result[row['quality_status']] = row['cnt']
            total += row['cnt']
        result['total'] = total
        return result

    def count_questions_for_kp(self, kp_id: str) -> int:
        """统计某知识点已通过审核的题目数量（只算FINAL_PASS）"""
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM questions WHERE kp_id = ? AND quality_status = 'FINAL_PASS'",
            (kp_id,),
        )
        return cur.fetchone()[0]

    def count_final_pass_questions_for_subject(self, subject_id: str) -> int:
        """统计某科目已通过审核的题目数量（只算FINAL_PASS）。"""
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM questions WHERE subject_id = ? AND quality_status = 'FINAL_PASS'",
            (subject_id,),
        )
        return cur.fetchone()[0]

    def count_final_pass_questions_total(self) -> int:
        """统计全库 FINAL_PASS 总数（用于动态降档判断）。"""
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM questions WHERE quality_status = 'FINAL_PASS'"
        )
        return cur.fetchone()[0]

    def count_total_questions_for_kp(self, kp_id: str) -> int:
        """统计某知识点已产生的全部题目记录，用于失败也占配额的安全续跑。"""
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM questions WHERE kp_id = ?",
            (kp_id,),
        )
        return cur.fetchone()[0]

    def count_total_questions_for_subject(self, subject_id: str) -> int:
        """统计某科目已产生的全部题目记录，用于科目级安全续跑。"""
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM questions WHERE subject_id = ?",
            (subject_id,),
        )
        return cur.fetchone()[0]

    def insert_question(self, question_id: str, kp_id: str, subject_id: str, question_data: dict):
        """V7: 单题插入"""
        now = _now_iso()
        # 在序列化前提取字段
        module_id = question_data.get('module_id', '') if isinstance(question_data, dict) else ''
        kp_name = question_data.get('knowledge_point_name', '') if isinstance(question_data, dict) else ''
        qjson = json.dumps(question_data, ensure_ascii=False) if isinstance(question_data, dict) else question_data
        with self._transaction() as cur:
            cur.execute(
                """INSERT OR IGNORE INTO questions
                (question_id, subject_id, module_id, kp_id, kp_name, batch_id,
                 question_json, image_path, quality_status, render_attempts,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'GENERATED', 0, ?, ?)""",
                (
                    question_id,
                    subject_id,
                    module_id,
                    kp_id,
                    kp_name,
                    f"v7_{subject_id}",
                    qjson,
                    '',
                    now,
                    now,
                ),
            )

    def update_question_review(self, question_id: str, verdict):
        """V7: 记录审核结论"""
        now = _now_iso()
        # verdict 可能是 dict，需要序列化为 JSON string
        if isinstance(verdict, dict):
            verdict_str = json.dumps(verdict, ensure_ascii=False)
        else:
            verdict_str = str(verdict)
        with self._transaction() as cur:
            cur.execute(
                "UPDATE questions SET sentinel_result = ?, updated_at = ? WHERE question_id = ?",
                (verdict_str, now, question_id),
            )

    def get_overall_progress(self) -> dict:
        """
        获取全局生产进度汇总。

        Returns:
            {
                'by_status': {'GENERATED': n, ...},
                'by_subject': {'subject_id': {'GENERATED': n, ...}, ...},
                'total': N
            }
        """
        # 全局按状态汇总
        cur = self.conn.execute(
            "SELECT quality_status, COUNT(*) as cnt FROM questions GROUP BY quality_status"
        )
        by_status = {s: 0 for s in QUALITY_STATUSES}
        total = 0
        for row in cur.fetchall():
            by_status[row['quality_status']] = row['cnt']
            total += row['cnt']

        # 按科目分组
        cur = self.conn.execute(
            "SELECT subject_id, quality_status, COUNT(*) as cnt FROM questions "
            "GROUP BY subject_id, quality_status"
        )
        by_subject = {}
        for row in cur.fetchall():
            sid = row['subject_id']
            if sid not in by_subject:
                by_subject[sid] = {s: 0 for s in QUALITY_STATUSES}
            by_subject[sid][row['quality_status']] = row['cnt']

        return {'by_status': by_status, 'by_subject': by_subject, 'total': total}

    # ─── Render Job Operations ────────────────────────────────────────────

    def add_render_job(self, question_id: str, engine: str, code: str) -> str:
        """
        创建渲染任务。

        Args:
            question_id: 关联题目ID
            engine: 渲染引擎 (matplotlib/tikz/svg/pillow等)
            code: 渲染代码

        Returns:
            job_id
        """
        job_id = _gen_id()
        now = _now_iso()
        with self._transaction() as cur:
            cur.execute(
                """INSERT INTO render_jobs
                (job_id, question_id, engine, code, status, attempts, created_at)
                VALUES (?, ?, ?, ?, 'PENDING', 0, ?)""",
                (job_id, question_id, engine, code, now),
            )
            # 同步更新题目的渲染引擎和代码字段
            cur.execute(
                "UPDATE questions SET render_engine = ?, render_code = ?, updated_at = ? "
                "WHERE question_id = ?",
                (engine, code, now, question_id),
            )
        return job_id

    def update_render_job(
        self, job_id: str, status: str, output_path: str = '', error: str = ''
    ):
        """
        更新渲染任务状态。

        成功时更新题目的image_path和状态；
        失败时记录错误日志并递增重试次数。
        """
        if status not in RENDER_JOB_STATUSES:
            raise ValueError(f"Invalid render status: {status}")

        now = _now_iso()
        with self._transaction() as cur:
            # 递增尝试次数
            cur.execute(
                "UPDATE render_jobs SET status = ?, output_path = ?, error_log = ?, "
                "attempts = attempts + 1 WHERE job_id = ?",
                (status, output_path, error, job_id),
            )
            # 获取关联的question_id
            cur.execute("SELECT question_id FROM render_jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            if row:
                qid = row['question_id']
                # 渲染成功：更新题目状态和图片路径
                if status == 'SUCCESS':
                    cur.execute(
                        "UPDATE questions SET image_path = ?, quality_status = 'RENDERED', "
                        "render_attempts = render_attempts + 1, updated_at = ? "
                        "WHERE question_id = ?",
                        (output_path, now, qid),
                    )
                    cur.execute(
                        "INSERT INTO audit_log (question_id, action, old_status, new_status, detail, timestamp) "
                        "VALUES (?, 'RENDER_SUCCESS', 'GENERATED', 'RENDERED', ?, ?)",
                        (qid, f"job={job_id}", now),
                    )
                elif status == 'FAIL':
                    # 失败时仅递增重试计数
                    cur.execute(
                        "UPDATE questions SET render_attempts = render_attempts + 1, updated_at = ? "
                        "WHERE question_id = ?",
                        (now, qid),
                    )

    def get_pending_render_jobs(self, limit: int = 50) -> list:
        """获取待处理的渲染任务列表（按创建时间排序）"""
        cur = self.conn.execute(
            "SELECT * FROM render_jobs WHERE status = 'PENDING' "
            "ORDER BY created_at LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]

    # ─── Audit Log ────────────────────────────────────────────────────────

    def record_audit(
        self,
        question_id: str,
        action: str,
        old_status: str,
        new_status: str,
        detail: str = '',
    ):
        """手动记录审计日志条目"""
        now = _now_iso()
        with self._transaction() as cur:
            cur.execute(
                "INSERT INTO audit_log (question_id, action, old_status, new_status, detail, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (question_id, action, old_status, new_status, detail, now),
            )

    # ─── Production Stats ─────────────────────────────────────────────────

    def update_production_stats(self, subject_id: str):
        """
        重新计算并更新指定科目的生产统计。
        使用 UPSERT 语义（INSERT OR REPLACE）确保幂等。
        """
        now = _now_iso()
        progress = self.get_subject_progress(subject_id)
        with self._transaction() as cur:
            # 获取已有的total_target（若存在）
            cur.execute(
                "SELECT total_target FROM production_stats WHERE subject_id = ?",
                (subject_id,),
            )
            row = cur.fetchone()
            total_target = row['total_target'] if row else 0

            cur.execute(
                """INSERT OR REPLACE INTO production_stats
                (subject_id, total_target, generated, rendered, passed, failed, hold, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    subject_id,
                    total_target,
                    progress.get('GENERATED', 0),
                    progress.get('RENDERED', 0),
                    progress.get('SENTINEL_PASS', 0) + progress.get('FINAL_PASS', 0),
                    progress.get('SENTINEL_FAIL', 0) + progress.get('FINAL_FAIL', 0),
                    progress.get('HOLD', 0),
                    now,
                ),
            )

    def get_production_stats(self) -> list:
        """获取所有科目的生产统计"""
        cur = self.conn.execute(
            "SELECT * FROM production_stats ORDER BY subject_id"
        )
        return [dict(row) for row in cur.fetchall()]

    def set_subject_target(self, subject_id: str, target: int):
        """设置科目的目标题目数量"""
        now = _now_iso()
        with self._transaction() as cur:
            cur.execute(
                """INSERT OR REPLACE INTO production_stats
                (subject_id, total_target, generated, rendered, passed, failed, hold, last_updated)
                VALUES (?, ?,
                    COALESCE((SELECT generated FROM production_stats WHERE subject_id = ?), 0),
                    COALESCE((SELECT rendered FROM production_stats WHERE subject_id = ?), 0),
                    COALESCE((SELECT passed FROM production_stats WHERE subject_id = ?), 0),
                    COALESCE((SELECT failed FROM production_stats WHERE subject_id = ?), 0),
                    COALESCE((SELECT hold FROM production_stats WHERE subject_id = ?), 0),
                    ?)""",
                (subject_id, target, subject_id, subject_id, subject_id, subject_id, subject_id, now),
            )

    # ─── Utility ──────────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def update_question_json(self, question_id: str, question_json: dict):
        """更新题目的question_json字段（如回填image_path）"""
        import json as _json
        now = _now_iso()
        data = _json.dumps(question_json, ensure_ascii=False)
        with self._transaction() as cur:
            cur.execute(
                "UPDATE questions SET question_json = ?, updated_at = ? WHERE question_id = ?",
                (data, now, question_id),
            )

    def __repr__(self):
        return f"ProductionDB(db_path='{self.db_path}')"


# ─── 模块可独立运行：验证Schema创建 ──────────────────────────────────────────
if __name__ == '__main__':
    import tempfile
    import os

    # 使用临时文件验证
    tmp = tempfile.mktemp(suffix='.db')
    try:
        with ProductionDB(tmp) as db:
            # 创建批次
            bid = db.create_batch('math', 'kp_001', 10)
            print(f"Created batch: {bid}")

            # 添加题目
            questions = [
                {'subject_id': 'math', 'kp_id': 'kp_001', 'kp_name': '函数极限',
                 'question_json': {'stem': '求极限...', 'options': ['A', 'B', 'C', 'D']}},
                {'subject_id': 'math', 'kp_id': 'kp_001', 'kp_name': '函数极限',
                 'question_json': {'stem': '计算导数...', 'options': ['A', 'B', 'C', 'D']}},
            ]
            qids = db.add_questions(bid, questions)
            print(f"Added {len(qids)} questions: {qids}")

            # 更新状态
            db.update_question_status(qids[0], 'RENDERED', '渲染完成')

            # 查询进度
            progress = db.get_subject_progress('math')
            print(f"Math progress: {progress}")

            # 添加渲染任务
            jid = db.add_render_job(qids[1], 'matplotlib', 'plt.plot([1,2,3])')
            print(f"Render job: {jid}")

            # 获取待渲染
            pending = db.get_pending_render_jobs()
            print(f"Pending render jobs: {len(pending)}")

            # 更新统计
            db.update_production_stats('math')
            stats = db.get_production_stats()
            print(f"Production stats: {stats}")

            print("\n✓ All operations completed successfully.")
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
