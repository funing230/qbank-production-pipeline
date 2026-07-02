"""
15路模型通道动态并行调度 (Provider Pool)
========================================
GPT Generator Pool: 9 providers (gpt5, gpt-worker-1..8)
Qwen Reviewer Pool: 6 providers (qwen, qwen-worker-1..5)

核心原则：
- 每个Provider独立限速(>=2s间隔)
- 不同Provider之间允许并行
- 动态扩缩容基于实时健康状态
- 任务租约机制防止重复处理
- API Key仅从环境变量读取，不记录到日志
"""

import asyncio
import os
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from collections import deque

import aiohttp

logger = logging.getLogger(__name__)


# ========== 数据结构 ==========

class ProviderStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    COOLDOWN = "COOLDOWN"
    DISABLED = "DISABLED"


class TaskStatus(Enum):
    PENDING = "PENDING"
    LEASED = "LEASED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


@dataclass
class ProviderConfig:
    """单个Provider配置"""
    name: str
    base_url: str
    api_key_env_var: str
    model: str
    role: str  # 'generator' | 'reviewer'
    status: ProviderStatus = ProviderStatus.HEALTHY
    active: bool = False  # 是否已激活（扩容控制）

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env_var)

    @property
    def available(self) -> bool:
        return (self.active and 
                self.status in (ProviderStatus.HEALTHY, ProviderStatus.DEGRADED) and
                self.api_key is not None)


@dataclass
class ProviderHealth:
    """Provider健康状态追踪"""
    success_count: int = 0
    fail_count: int = 0
    total_latency: float = 0.0
    latencies: list = field(default_factory=lambda: deque(maxlen=50))
    last_error_time: float = 0.0
    consecutive_errors: int = 0
    json_complete_count: int = 0
    json_total_count: int = 0
    last_request_time: float = 0.0
    in_flight: int = 0
    cooldown_until: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 1.0

    @property
    def avg_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def p95_latency(self) -> float:
        if len(self.latencies) < 5:
            return self.avg_latency
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat)-1)]

    @property
    def json_completeness_rate(self) -> float:
        if self.json_total_count == 0:
            return 1.0
        return self.json_complete_count / self.json_total_count

    @property
    def weight(self) -> float:
        """负载均衡权重：越高越优先分配"""
        if self.success_rate < 0.5:
            return 0.1
        w = self.success_rate * 10.0
        if self.avg_latency > 0:
            w /= (1 + self.avg_latency / 30.0)
        w /= (1 + self.in_flight * 2)
        return max(0.1, w)


@dataclass
class TaskLease:
    """任务租约"""
    task_id: str
    provider_id: str
    leased_at: float
    lease_expires_at: float
    attempt: int
    status: TaskStatus


# ========== Provider Pool ==========

class ProviderPool:
    """一个阵营的Provider池（GPT或Qwen）"""
    
    MIN_REQUEST_INTERVAL = 2.0  # 每个Provider最小请求间隔(秒)
    COOLDOWN_DURATION = 60.0    # 冷却时长(秒)
    MAX_CONSECUTIVE_ERRORS = 5  # 连续错误阈值 -> COOLDOWN
    DISABLE_THRESHOLD = 15      # 连续错误阈值 -> DISABLED

    def __init__(self, pool_type: str, providers: list):
        """
        Args:
            pool_type: 'gpt' 或 'claude'
            providers: ProviderConfig列表
        """
        self.pool_type = pool_type
        self.providers = {p.name: p for p in providers}
        self.health = {p.name: ProviderHealth() for p in providers}
        self._locks = {p.name: asyncio.Lock() for p in providers}
        self._pool_lock = asyncio.Lock()
        
        # 扩容等级
        self._scale_level = 1  # 1=初始, 2=中等, 3=全部
        
    def _get_scale_counts(self) -> int:
        """根据扩容等级返回应激活的Provider数量"""
        total = len(self.providers)
        if self.pool_type == 'gpt':
            # GPT: 3 -> 6 -> 9
            return [3, 6, total][min(self._scale_level - 1, 2)]
        else:
            # Claude: 2 -> 4 -> 6
            return [2, 4, total][min(self._scale_level - 1, 2)]

    def initialize(self, initial_active: Optional[int] = None):
        """初始化：激活初始Provider数量"""
        count = initial_active or self._get_scale_counts()
        activated = 0
        for name, provider in self.providers.items():
            if activated < count and provider.api_key:
                provider.active = True
                provider.status = ProviderStatus.HEALTHY
                activated += 1
                logger.info(f"[{self.pool_type}] Activated provider: {name}")
            else:
                provider.active = False

    async def acquire(self, task_type: str = "default") -> Optional[ProviderConfig]:
        """
        获取最佳可用Provider（加权最少任务算法）。
        
        Args:
            task_type: 任务类型（用于日志）
        
        Returns:
            ProviderConfig 或 None（无可用Provider）
        """
        async with self._pool_lock:
            # 先检查冷却到期的Provider
            now = time.time()
            for name, health in self.health.items():
                provider = self.providers[name]
                if (provider.status == ProviderStatus.COOLDOWN and 
                    now >= health.cooldown_until):
                    provider.status = ProviderStatus.HEALTHY
                    health.consecutive_errors = 0
                    logger.info(f"[{self.pool_type}] Provider {name} recovered from COOLDOWN")
            
            # 筛选可用Provider
            available = []
            for name, provider in self.providers.items():
                if not provider.available:
                    continue
                health = self.health[name]
                # 检查限速：距离上次请求至少2秒
                if now - health.last_request_time < self.MIN_REQUEST_INTERVAL:
                    continue
                # 每个Provider同时只处理一个在途请求
                if health.in_flight >= 1:
                    continue
                available.append((name, provider, health))
            
            if not available:
                return None
            
            # 加权选择：weight最高的优先
            available.sort(key=lambda x: x[2].weight, reverse=True)
            chosen_name, chosen_provider, chosen_health = available[0]
            
            # 标记在途
            chosen_health.in_flight += 1
            chosen_health.last_request_time = now
            
            return chosen_provider

    async def release(self, provider_name: str, success: bool, latency: float, 
                      json_complete: bool = True):
        """
        归还Provider，更新健康状态。
        
        Args:
            provider_name: Provider名称
            success: 请求是否成功
            latency: 响应时间(秒)
            json_complete: JSON是否完整（未截断）
        """
        if provider_name not in self.health:
            return
            
        health = self.health[provider_name]
        provider = self.providers[provider_name]
        
        health.in_flight = max(0, health.in_flight - 1)
        health.latencies.append(latency)
        health.json_total_count += 1
        
        if success:
            health.success_count += 1
            health.consecutive_errors = 0
            if json_complete:
                health.json_complete_count += 1
            # 从DEGRADED恢复
            if provider.status == ProviderStatus.DEGRADED and health.success_rate > 0.95:
                provider.status = ProviderStatus.HEALTHY
        else:
            health.fail_count += 1
            health.consecutive_errors += 1
            health.last_error_time = time.time()
            
            # 状态降级
            if health.consecutive_errors >= self.DISABLE_THRESHOLD:
                provider.status = ProviderStatus.DISABLED
                logger.warning(f"[{self.pool_type}] Provider {provider_name} DISABLED "
                             f"({health.consecutive_errors} consecutive errors)")
            elif health.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                provider.status = ProviderStatus.COOLDOWN
                health.cooldown_until = time.time() + self.COOLDOWN_DURATION
                logger.warning(f"[{self.pool_type}] Provider {provider_name} COOLDOWN "
                             f"({health.consecutive_errors} consecutive errors)")
            elif health.consecutive_errors >= 3:
                provider.status = ProviderStatus.DEGRADED

    def scale_up(self):
        """扩容：增加活跃Provider数量"""
        if self._scale_level >= 3:
            return
        self._scale_level += 1
        target = self._get_scale_counts()
        
        activated = sum(1 for p in self.providers.values() if p.active)
        for name, provider in self.providers.items():
            if activated >= target:
                break
            if not provider.active and provider.api_key:
                provider.active = True
                provider.status = ProviderStatus.HEALTHY
                activated += 1
                logger.info(f"[{self.pool_type}] Scale UP: activated {name} "
                          f"(level={self._scale_level}, total={activated})")

    def scale_down(self):
        """缩容：减少活跃Provider数量"""
        if self._scale_level <= 1:
            return
        self._scale_level -= 1
        target = self._get_scale_counts()
        
        # 从末尾开始停用
        active_list = [n for n, p in self.providers.items() if p.active]
        for name in reversed(active_list):
            if sum(1 for p in self.providers.values() if p.active) <= target:
                break
            health = self.health[name]
            if health.in_flight == 0:
                self.providers[name].active = False
                logger.info(f"[{self.pool_type}] Scale DOWN: deactivated {name} "
                          f"(level={self._scale_level})")

    def get_healthy_count(self) -> int:
        return sum(1 for p in self.providers.values() 
                   if p.active and p.status == ProviderStatus.HEALTHY)

    def get_active_count(self) -> int:
        return sum(1 for p in self.providers.values() if p.active)

    def get_status_report(self) -> dict:
        """生成状态报告（用于飞书15分钟汇报）"""
        report = {
            "pool_type": self.pool_type,
            "scale_level": self._scale_level,
            "total_providers": len(self.providers),
            "active_providers": self.get_active_count(),
            "healthy_providers": self.get_healthy_count(),
            "providers": {}
        }
        
        for name, provider in self.providers.items():
            if not provider.active:
                continue
            health = self.health[name]
            report["providers"][name] = {
                "status": provider.status.value,
                "success_count": health.success_count,
                "fail_count": health.fail_count,
                "avg_latency": round(health.avg_latency, 2),
                "p95_latency": round(health.p95_latency, 2),
                "success_rate": round(health.success_rate * 100, 1),
                "in_flight": health.in_flight,
                "consecutive_errors": health.consecutive_errors,
            }
        
        return report


# ========== 动态调度器 ==========

class DynamicScheduler:
    """
    中央动态调度器：管理GPT和Qwen两个Pool，
    实现生成-审核流水并行和背压控制。
    """
    
    # 扩容条件
    SCALE_UP_SUCCESS_RATE = 0.99
    SCALE_UP_429_RATE = 0.01
    SCALE_UP_MIN_QUEUE = 5
    
    # 背压阈值
    BACKPRESSURE_RATIO = 2.0  # 待审核 > Qwen 1小时能力 × 2
    
    def __init__(self):
        self.gpt_pool = self._build_gpt_pool()
        self.qwen_pool = self._build_qwen_pool()
        
        # 任务队列
        self.generation_queue = asyncio.Queue()
        self.review_queue = asyncio.Queue()
        self.regen_queue = asyncio.Queue()
        self.fix_queue = asyncio.Queue()  # kept for backward compat
        
        # 任务租约跟踪
        self.leases: dict[str, TaskLease] = {}
        self._lease_lock = asyncio.Lock()
        
        # 统计
        self.stats = {
            "generated": 0,
            "reviewed": 0,
            "fixed": 0,
            "rejected": 0,
            "errors": 0,
            "429_count": 0,
            "5xx_count": 0,
            "start_time": time.time(),
        }
        
        # 背压状态
        self._backpressure_active = False
        
    def _build_gpt_pool(self) -> ProviderPool:
        """构建GPT Provider池 — 1主key + 8个worker并发"""
        base_url = os.environ.get("GPT5_BASE_URL", "https://api.lk888.ai/v1")
        model = os.environ.get("GPT_MODEL", "gpt-5.5")
        
        providers = [
            ProviderConfig("gpt5", base_url, "GPT5_API_KEY", model, "generator"),
            ProviderConfig("gpt-worker-1", base_url, "GPT_WORKER1_API_KEY", model, "generator"),
            ProviderConfig("gpt-worker-2", base_url, "GPT_WORKER2_API_KEY", model, "generator"),
            ProviderConfig("gpt-worker-3", base_url, "GPT_WORKER3_API_KEY", model, "generator"),
            ProviderConfig("gpt-worker-4", base_url, "GPT_WORKER4_API_KEY", model, "generator"),
            ProviderConfig("gpt-worker-5", base_url, "GPT_WORKER5_API_KEY", model, "generator"),
            ProviderConfig("gpt-worker-6", base_url, "GPT_WORKER6_API_KEY", model, "generator"),
            ProviderConfig("gpt-worker-7", base_url, "GPT_WORKER7_API_KEY", model, "generator"),
            ProviderConfig("gpt-worker-8", base_url, "GPT_WORKER8_API_KEY", model, "generator"),
        ]
        
        # 过滤出有 key 的 provider
        valid_providers = [p for p in providers if os.environ.get(p.api_key_env_var)]
        if not valid_providers:
            raise RuntimeError("No GPT API keys found in environment!")
        
        pool = ProviderPool("gpt", valid_providers)
        pool.initialize(initial_active=len(valid_providers))  # 全部9路并发启动
        return pool

    def _build_qwen_pool(self) -> ProviderPool:
        """构建Qwen3.7-max Reviewer池（单key 6路并发）"""
        base_url = os.environ.get("QWEN_BASE_URL", "https://yuanlansj.xin/v1")
        model = os.environ.get("QWEN_MODEL", "qwen3.7-max")
        api_key_env = "QWEN_API_KEY"
        
        providers = [
            ProviderConfig("qwen", base_url, api_key_env, model, "reviewer"),
            ProviderConfig("qwen-worker-1", base_url, api_key_env, model, "reviewer"),
            ProviderConfig("qwen-worker-2", base_url, api_key_env, model, "reviewer"),
            ProviderConfig("qwen-worker-3", base_url, api_key_env, model, "reviewer"),
            ProviderConfig("qwen-worker-4", base_url, api_key_env, model, "reviewer"),
            ProviderConfig("qwen-worker-5", base_url, api_key_env, model, "reviewer"),
        ]
        
        pool = ProviderPool("qwen", providers)
        pool.initialize(initial_active=6)  # 全部6路并发启动
        return pool

    def initialize(self):
        """验证环境变量并初始化池"""
        # 只验证存在性，不打印值
        gpt_keys = sum(1 for p in self.gpt_pool.providers.values() if p.api_key)
        qwen_keys = sum(1 for p in self.qwen_pool.providers.values() if p.api_key)
        
        logger.info(f"Provider Pool initialized: GPT={gpt_keys}/9 keys, Qwen={qwen_keys}/6 keys")
        
        if gpt_keys == 0:
            raise RuntimeError("No GPT API keys found in environment")
        if qwen_keys == 0:
            logger.warning("No Qwen API keys found - review tasks will queue")

    async def call_gpt(self, messages: list, task_id: str, 
                       task_type: str = "generate") -> Optional[dict]:
        """
        通过GPT Pool发送请求。自动选择Provider、限速、重试。
        
        Args:
            messages: OpenAI格式消息列表
            task_id: 唯一任务ID
            task_type: generate|fix|spec
        
        Returns:
            API响应dict 或 None（失败）
        """
        max_attempts = 3
        
        for attempt in range(max_attempts):
            provider = await self.gpt_pool.acquire(task_type)
            if provider is None:
                # 所有Provider忙，等待后重试
                await asyncio.sleep(2.0)
                continue
            
            # 创建租约
            lease = TaskLease(
                task_id=task_id,
                provider_id=provider.name,
                leased_at=time.time(),
                lease_expires_at=time.time() + 120,
                attempt=attempt,
                status=TaskStatus.LEASED
            )
            async with self._lease_lock:
                self.leases[task_id] = lease
            
            start = time.time()
            try:
                result = await self._make_api_call(
                    provider.base_url, provider.api_key, provider.model, messages
                )
                latency = time.time() - start
                
                # 检查JSON完整性
                json_ok = result is not None and "choices" in result
                await self.gpt_pool.release(provider.name, True, latency, json_ok)
                
                lease.status = TaskStatus.COMPLETED
                return result
                
            except RateLimitError:
                latency = time.time() - start
                await self.gpt_pool.release(provider.name, False, latency)
                self.stats["429_count"] += 1
                lease.status = TaskStatus.FAILED
                await asyncio.sleep(5 * (attempt + 1))  # 递增等待
                
            except ServerError:
                latency = time.time() - start
                await self.gpt_pool.release(provider.name, False, latency)
                self.stats["5xx_count"] += 1
                lease.status = TaskStatus.FAILED
                await asyncio.sleep(3 * (attempt + 1))
                
            except Exception as e:
                latency = time.time() - start
                await self.gpt_pool.release(provider.name, False, latency)
                self.stats["errors"] += 1
                lease.status = TaskStatus.FAILED
                logger.error(f"[GPT] Task {task_id} error on {provider.name}: {type(e).__name__}")
                await asyncio.sleep(2)
        
        return None

    async def call_qwen(self, messages: list, task_id: str,
                          task_type: str = "review") -> Optional[dict]:
        """通过Qwen Pool发送审核请求"""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            provider = await self.qwen_pool.acquire(task_type)
            if provider is None:
                await asyncio.sleep(2.0)
                continue
            
            lease = TaskLease(
                task_id=task_id,
                provider_id=provider.name,
                leased_at=time.time(),
                lease_expires_at=time.time() + 180,
                attempt=attempt,
                status=TaskStatus.LEASED
            )
            async with self._lease_lock:
                self.leases[task_id] = lease
            
            start = time.time()
            try:
                result = await self._make_api_call(
                    provider.base_url, provider.api_key, provider.model, messages
                )
                latency = time.time() - start
                json_ok = result is not None and "choices" in result
                await self.qwen_pool.release(provider.name, True, latency, json_ok)
                lease.status = TaskStatus.COMPLETED
                return result
                
            except RateLimitError:
                latency = time.time() - start
                await self.qwen_pool.release(provider.name, False, latency)
                self.stats["429_count"] += 1
                lease.status = TaskStatus.FAILED
                await asyncio.sleep(5 * (attempt + 1))
                
            except ServerError:
                latency = time.time() - start
                await self.qwen_pool.release(provider.name, False, latency)
                self.stats["5xx_count"] += 1
                lease.status = TaskStatus.FAILED
                await asyncio.sleep(3 * (attempt + 1))
                
            except Exception as e:
                latency = time.time() - start
                await self.qwen_pool.release(provider.name, False, latency)
                self.stats["errors"] += 1
                lease.status = TaskStatus.FAILED
                logger.error(f"[Qwen] Task {task_id} error on {provider.name}: {type(e).__name__}")
                await asyncio.sleep(2)
        
        return None

    async def _make_api_call(self, base_url: str, api_key: str, 
                             model: str, messages: list) -> dict:
        """执行实际的API HTTP请求"""
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 429:
                    raise RateLimitError(f"429 from {base_url}")
                elif resp.status >= 500:
                    raise ServerError(f"{resp.status} from {base_url}")
                elif resp.status != 200:
                    text = await resp.text()
                    raise APIError(f"{resp.status}: {text[:200]}")
                return await resp.json()

    def check_backpressure(self) -> dict:
        """检查背压状态：生成速度是否超过审核能力"""
        review_queue_size = self.review_queue.qsize()
        qwen_active = self.qwen_pool.get_active_count()
        
        # 估算Qwen每小时审核能力（Qwen平均3秒/次，比Claude快很多）
        qwen_hourly_capacity = qwen_active * (3600 / 5)
        
        ratio = review_queue_size / max(1, qwen_hourly_capacity / 2)
        
        should_throttle = ratio > self.BACKPRESSURE_RATIO
        
        if should_throttle and not self._backpressure_active:
            self._backpressure_active = True
            logger.warning(f"Backpressure ACTIVE: review queue={review_queue_size}, "
                         f"ratio={ratio:.1f}x")
        elif not should_throttle and self._backpressure_active:
            self._backpressure_active = False
            logger.info("Backpressure released")
        
        return {
            "active": self._backpressure_active,
            "review_queue_size": review_queue_size,
            "generation_queue_size": self.generation_queue.qsize(),
            "fix_queue_size": self.fix_queue.qsize(),
            "qwen_hourly_capacity": int(qwen_hourly_capacity),
            "ratio": round(ratio, 2),
        }

    def auto_scale(self):
        """根据当前状态自动扩缩容"""
        # GPT扩容检查
        gpt_health = self.gpt_pool.get_status_report()
        gpt_providers = gpt_health.get("providers", {})
        if gpt_providers:
            avg_success = sum(p["success_rate"] for p in gpt_providers.values()) / len(gpt_providers)
            has_queue = self.generation_queue.qsize() > self.SCALE_UP_MIN_QUEUE
            
            if avg_success >= self.SCALE_UP_SUCCESS_RATE * 100 and has_queue:
                if not self._backpressure_active:
                    self.gpt_pool.scale_up()
            elif avg_success < 80:
                self.gpt_pool.scale_down()
        
        # Qwen扩容检查
        qwen_health = self.qwen_pool.get_status_report()
        qwen_providers = qwen_health.get("providers", {})
        if qwen_providers:
            has_review_queue = self.review_queue.qsize() > 3
            avg_success = sum(p["success_rate"] for p in qwen_providers.values()) / len(qwen_providers)
            
            if avg_success >= self.SCALE_UP_SUCCESS_RATE * 100 and has_review_queue:
                self.qwen_pool.scale_up()
            elif avg_success < 80:
                self.qwen_pool.scale_down()

    def get_progress_report(self) -> dict:
        """生成完整进度报告（用于飞书15分钟汇报）"""
        elapsed = time.time() - self.stats["start_time"]
        
        return {
            "elapsed_minutes": round(elapsed / 60, 1),
            "generated": self.stats["generated"],
            "reviewed": self.stats["reviewed"],
            "fixed": self.stats["fixed"],
            "rejected": self.stats["rejected"],
            "errors": self.stats["errors"],
            "rate_limits_429": self.stats["429_count"],
            "server_errors_5xx": self.stats["5xx_count"],
            "gpt_pool": self.gpt_pool.get_status_report(),
            "qwen_pool": self.qwen_pool.get_status_report(),
            "backpressure": self.check_backpressure(),
            "queues": {
                "generation": self.generation_queue.qsize(),
                "review": self.review_queue.qsize(),
                "regen": self.regen_queue.qsize(),
                "fix": self.fix_queue.qsize(),
            }
        }


# ========== 异常类 ==========

class RateLimitError(Exception):
    pass

class ServerError(Exception):
    pass

class APIError(Exception):
    pass


# ========== 便捷初始化函数 ==========

def create_scheduler() -> DynamicScheduler:
    """创建并初始化调度器（验证环境变量）"""
    scheduler = DynamicScheduler()
    scheduler.initialize()
    return scheduler
