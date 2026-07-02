"""
飞书进度汇报模块 v1.0（冻结版）

实现 FEISHU_PROGRESS_REPORT_SPEC.md 中定义的：
- 每15分钟定时汇报
- 异常三级处理（自动/累积/暂停）
- 暂停心跳
- 发送失败重试
- 本地日志

发送目标（不可变）：oc_0fac3a04b055503ceefc95f636cb7daf
"""
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable, Awaitable

logger = logging.getLogger(__name__)

# 不可变发送目标
FEISHU_REPORT_TARGET = "feishu:oc_0fac3a04b055503ceefc95f636cb7daf"
REPORT_INTERVAL_SECONDS = 15 * 60  # 15分钟


@dataclass
class ProductionStats:
    """生产统计数据"""
    start_time: float = field(default_factory=time.time)
    generated_items: int = 0
    generated_images: int = 0
    auto_passed: int = 0
    final_passed: int = 0
    completed_subjects: int = 0
    total_subjects: int = 18
    total_target: int = 24000
    # 最近15分钟
    period_new_items: int = 0
    period_new_images: int = 0
    period_reviewed: int = 0
    period_pass: int = 0
    period_revise: int = 0
    period_reject: int = 0
    period_regen: int = 0
    # 通道状态
    gpt_healthy: int = 9
    gpt_active: int = 0
    qwen_healthy: int = 6
    qwen_active: int = 0
    generation_queue: int = 0
    review_queue: int = 0
    regen_queue: int = 0
    # 质量
    layout_failures: int = 0
    font_failures: int = 0
    image_text_failures: int = 0
    naming_failures: int = 0
    duplicate_risks: int = 0
    # 状态
    system_status: str = "RUNNING"  # RUNNING | PAUSED_WAITING_USER_DECISION | USER_TERMINATED
    current_phase: str = "生产中"
    current_bottleneck: str = "无"
    risk_level: str = "🟢正常"  # 🟢正常 | 🟡注意 | 🔴异常

    def reset_period(self):
        """重置周期计数器"""
        self.period_new_items = 0
        self.period_new_images = 0
        self.period_reviewed = 0
        self.period_pass = 0
        self.period_revise = 0
        self.period_reject = 0
        self.period_regen = 0

    def elapsed_str(self) -> str:
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        return f"{hours}h{minutes}m"

    def eta_str(self) -> str:
        elapsed = time.time() - self.start_time
        if self.final_passed <= 0 or elapsed < 60:
            return "计算中..."
        rate = self.final_passed / (elapsed / 3600)
        remaining = self.total_target - self.final_passed
        if rate <= 0:
            return "无法估计"
        hours = remaining / rate
        return f"约{hours:.1f}小时"

    def hourly_throughput(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            return 0
        return self.final_passed / (elapsed / 3600)


@dataclass
class Incident:
    """异常事件"""
    incident_id: str = field(default_factory=lambda: f"INC-{uuid.uuid4().hex[:8]}")
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    level: str = "WARNING"  # INFO | WARNING | ERROR | CRITICAL
    phase: str = ""
    subject_id: str = ""
    knowledge_point_id: str = ""
    item_id: str = ""
    description: str = ""
    affected_items: int = 0
    recommendation: str = ""


class FeishuProgressReporter:
    """飞书进度汇报器"""

    def __init__(self, send_fn: Callable[[str, str], Awaitable[bool]], log_dir: Path):
        """
        Args:
            send_fn: 异步发送函数 async (target, message) -> bool
            log_dir: 日志目录
        """
        self._send_fn = send_fn
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "feishu_reports.jsonl"
        self._report_no = 0
        self._stats = ProductionStats()
        self._paused = False
        self._pause_incident: Optional[Incident] = None
        self._accumulated_warnings: list = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @property
    def stats(self) -> ProductionStats:
        return self._stats

    async def start(self):
        """启动定时汇报循环"""
        self._running = True
        self._task = asyncio.create_task(self._report_loop())
        logger.info("Feishu progress reporter started")

    async def stop(self):
        """停止汇报循环"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _report_loop(self):
        """每15分钟发送进度报告"""
        while self._running:
            try:
                await asyncio.sleep(REPORT_INTERVAL_SECONDS)
                if self._paused:
                    await self._send_pause_heartbeat()
                else:
                    await self._send_progress_report()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Report loop error: {e}")

    async def _send_progress_report(self):
        """发送正常进度报告"""
        self._report_no += 1
        s = self._stats

        msg = f"""【24,000题图文题库进度｜第{self._report_no}次｜{s.risk_level}】

一、时间
- 当前时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 已运行：{s.elapsed_str()}
- 当前阶段：{s.current_phase}
- 系统状态：{s.system_status}

二、总体进度
- 已生成题目：{s.generated_items} / {s.total_target}
- 已生成图片：{s.generated_images} / {s.total_target}
- 最终通过：{s.final_passed} / {s.total_target}
- 已完成科目：{s.completed_subjects} / {s.total_subjects}

三、最近15分钟
- 新生成：{s.period_new_items}题 / {s.period_new_images}图
- 审核：{s.period_reviewed}（PASS:{s.period_pass} REVISE:{s.period_revise} REJECT:{s.period_reject}）
- 重生成：{s.period_regen}

四、并行通道
- GPT：{s.gpt_active}/{s.gpt_healthy}活跃
- Qwen：{s.qwen_active}/{s.qwen_healthy}活跃
- 队列：生成{s.generation_queue} / 审核{s.review_queue} / 重生成{s.regen_queue}

五、质量
- 排版问题：{s.layout_failures} | 字体问题：{s.font_failures}
- 图文不一致：{s.image_text_failures} | 重复风险：{s.duplicate_risks}

六、ETA
- 吞吐：{s.hourly_throughput():.0f}题/小时
- 预计完成：{s.eta_str()}
- 瓶颈：{s.current_bottleneck}"""

        # 附加累积警告
        if self._accumulated_warnings:
            msg += f"\n\n七、累积警告（{len(self._accumulated_warnings)}条）"
            for w in self._accumulated_warnings[-5:]:
                msg += f"\n- [{w['level']}] {w['description']}"
            self._accumulated_warnings.clear()

        await self._send_with_retry(msg)
        s.reset_period()
        self._log_report("progress", msg)

    async def _send_pause_heartbeat(self):
        """暂停期间心跳"""
        s = self._stats
        inc = self._pause_incident

        msg = f"""【题库生产暂停中｜等待用户确认】

- 异常编号：{inc.incident_id if inc else 'N/A'}
- 暂停开始时间：{inc.timestamp if inc else 'N/A'}
- 当前状态：PAUSED_WAITING_USER_DECISION
- 当前完成题目：{s.final_passed} / {s.total_target}
- 已完成科目：{s.completed_subjects} / {s.total_subjects}
- 异常摘要：{inc.description if inc else '未知'}

请回复处理命令：继续 / 忽略继续 / 废弃重做 / 保持暂停 / 终止"""

        await self._send_with_retry(msg)
        self._log_report("pause_heartbeat", msg)

    async def report_incident(self, incident: Incident) -> bool:
        """
        报告异常并决定是否暂停。
        
        Returns:
            True = 已暂停流程，等待用户确认
            False = 已自动处理或累积
        """
        # 三级判定
        if incident.level == "CRITICAL":
            # L3: 立即暂停
            return await self._pause_and_alert(incident)
        elif incident.level == "ERROR":
            # L2: 累积到5条时暂停
            self._accumulated_warnings.append({
                "level": incident.level,
                "description": incident.description,
                "time": incident.timestamp,
            })
            if len(self._accumulated_warnings) >= 5:
                return await self._pause_and_alert(incident)
            return False
        else:
            # L1: 自动处理，记录
            self._accumulated_warnings.append({
                "level": incident.level,
                "description": incident.description,
                "time": incident.timestamp,
            })
            return False

    async def _pause_and_alert(self, incident: Incident) -> bool:
        """暂停流程并发送异常报告"""
        self._paused = True
        self._pause_incident = incident
        self._stats.system_status = "PAUSED_WAITING_USER_DECISION"

        msg = f"""【题库生产异常告警｜流程已暂停】

一、异常信息
- 异常编号：{incident.incident_id}
- 发生时间：{incident.timestamp}
- 当前状态：PAUSED_WAITING_USER_DECISION
- 已运行时间：{self._stats.elapsed_str()}
- 异常级别：{incident.level}
- 异常阶段：{incident.phase}

二、异常对象
- 科目：{incident.subject_id}
- 知识点：{incident.knowledge_point_id}
- 题目：{incident.item_id}

三、异常说明
{incident.description}

四、影响范围
- 受影响题目：{incident.affected_items}

五、建议
{incident.recommendation}

六、请回复处理命令
- 继续（确认修改并继续）
- 忽略继续（忽略异常并继续）
- 废弃重做（废弃当前题目重新生成）
- 保持暂停
- 终止"""

        await self._send_with_retry(msg)
        self._log_report("incident", msg)
        return True

    async def resume(self, user_decision: str):
        """用户确认后恢复"""
        self._paused = False
        self._stats.system_status = "RUNNING"

        msg = f"""【题库生产流程已恢复】

- 异常编号：{self._pause_incident.incident_id if self._pause_incident else 'N/A'}
- 用户决定：{user_decision}
- 恢复时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 当前题目进度：{self._stats.final_passed} / {self._stats.total_target}
- GPT通道：{self._stats.gpt_active}/{self._stats.gpt_healthy}"""

        await self._send_with_retry(msg)
        self._log_report("resume", msg)
        self._pause_incident = None

    async def report_subject_complete(self, subject_id: str, subject_name: str, 
                                       kp_count: int, pass_count: int, 
                                       status_dist: dict):
        """单科完成时报告"""
        msg = f"""【科目完成报告】{subject_id} {subject_name}

- KP总数：{kp_count}
- 最终通过：{pass_count}
- 状态分布：PASS={status_dist.get('PASS',0)} REVISE→PASS={status_dist.get('REGEN_PASS',0)} FROZEN_FAIL={status_dist.get('FROZEN_FAIL',0)}
- 科目准入：{'✅ READY' if pass_count >= kp_count * 0.95 else '⚠️ PARTIAL'}"""

        await self._send_with_retry(msg)
        self._log_report("subject_complete", msg)

    async def _send_with_retry(self, message: str, max_retries: int = 3):
        """带重试的发送"""
        delays = [10, 30, 60]
        for attempt in range(max_retries):
            try:
                success = await self._send_fn(FEISHU_REPORT_TARGET, message)
                if success:
                    return True
            except Exception as e:
                logger.warning(f"Feishu send attempt {attempt+1} failed: {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(delays[attempt])
        
        # 三次失败，保存到本地
        pending_dir = self._log_dir / "feishu_pending_reports"
        pending_dir.mkdir(exist_ok=True)
        pending_file = pending_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        pending_file.write_text(message, encoding="utf-8")
        logger.error(f"Feishu send failed after {max_retries} retries, saved to {pending_file}")
        return False

    def _log_report(self, report_type: str, message: str):
        """写入本地日志"""
        record = {
            "report_id": f"RPT-{uuid.uuid4().hex[:8]}",
            "report_type": report_type,
            "timestamp": datetime.now().isoformat(),
            "system_status": self._stats.system_status,
            "generated_items": self._stats.generated_items,
            "final_passed": self._stats.final_passed,
            "completed_subjects": self._stats.completed_subjects,
            "risk_level": self._stats.risk_level,
        }
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
