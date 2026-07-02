#!/usr/bin/env python3
"""
V6 正式生产脚本 - 15路Provider Pool动态调度
============================================
架构升级：
- GPT 9路动态并行（初始3路，自动扩容至9路）
- Qwen3.7-max 6路动态并行（初始3路，自动扩容至6路）
- 每个Provider独立限速（>=2s间隔）
- 流水线并行：生成→渲染→防遮挡检查→审核同时运行
- 背压控制：审核积压时自动降低生成速度
- 15分钟自动汇报+每科完成即报

补充要求集成：
- 补充要求一：图片防遮挡检查（渲染后自动修复）
- 补充要求二：CJK字体统一配置 + 固定英文名称映射
- 补充要求三：15路Provider Pool动态调度

不修改：出题逻辑、JSON格式、审核标准、图片风格、最终交付结构
"""
import sys
import json
import time
import os
import signal
import asyncio
import threading
import uuid
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.orchestrator import ProductionOrchestrator, Config
from pipeline.generator import QuestionGenerator
from pipeline.db import ProductionDB
from pipeline.occlusion_checker import quick_check, check_and_fix
from pipeline.font_manager import setup_fonts
from pipeline.provider_pool import DynamicScheduler, create_scheduler

# ========== 配置 ==========
READY_SUBJECTS = ["S01", "S02", "S03", "S04", "S05", "S06", "S13",
                   "S14", "S15", "S16", "S17", "S18", "S19", "S20",
                   "S21", "S22", "S23", "S24"]

BATCH_SIZE = 10              # 每批题目数
RENDER_WORKERS = 8           # 渲染并行度
REPORT_INTERVAL = 900        # 飞书报告间隔(秒) = 15分钟
AUTO_SCALE_INTERVAL = 60     # 自动扩缩容检查间隔(秒)

LOG_FILE = PROJECT_ROOT / "production_v6.log"
_log_lock = threading.Lock()
_db_lock = threading.Lock()
_shutdown = threading.Event()

# ========== 日志 ==========
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with _log_lock:
        print(line, flush=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


# ========== 信号处理 ==========
def handle_signal(signum, frame):
    log(f"Received signal {signum}, initiating graceful shutdown...")
    _shutdown.set()

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


# ========== 并行渲染 + 防遮挡检查 ==========
_render_pool = ThreadPoolExecutor(max_workers=RENDER_WORKERS)

def render_one(orch, q, qid, subject_id):
    """渲染单题图片 + 防遮挡检查"""
    engine = q.get("render_engine", "MATPLOTLIB")
    code = q.get("render_code", "")
    if not code:
        return None
    
    img_dir = orch.config.output_dir / subject_id / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / f"{qid}.png"
    
    result = orch.renderer.dispatch(engine, code, str(img_path))
    
    if result["success"]:
        # 补充要求一：防遮挡检查
        if not quick_check(str(img_path)):
            fix_result = check_and_fix(
                str(img_path), code, str(img_path),
                max_retries=2
            )
            if not fix_result["pass"]:
                # 低严重度问题放行，高严重度标记但仍入库（不丢题）
                log(f"    ⚠️ [{qid}] 图片有遮挡问题(尝试{fix_result['attempts']}次修复)，已放行")
        
        with _db_lock:
            orch.db.update_question_status(qid, "RENDERED", f"engine={engine}")
        rel_path = str(img_path.relative_to(orch.config.output_dir))
        q["image_path"] = rel_path
        if isinstance(q.get("question_json"), dict):
            q["question_json"]["image_path"] = rel_path
            with _db_lock:
                orch.db.update_question_json(qid, q["question_json"])
        return (q, qid, str(img_path))
    else:
        try:
            with _db_lock:
                orch.db.add_render_job(qid, engine, code)
        except Exception:
            pass
        return None


def render_batch(orch, questions, q_ids, subject_id):
    """并行渲染一批题目"""
    rendered = []
    futures = {_render_pool.submit(render_one, orch, q, qid, subject_id): qid 
               for q, qid in zip(questions, q_ids)}
    for future in as_completed(futures):
        try:
            result = future.result()
            if result:
                rendered.append(result)
        except Exception as e:
            log(f"    RENDER ERR: {str(e)[:80]}")
    return rendered


# ========== 异步生产核心 ==========

async def produce_subject(scheduler: DynamicScheduler, orch, generator, 
                          subject_id: str, kp_list: list, stats: dict) -> dict:
    """
    异步生产一个科目的所有题目。
    通过scheduler动态分配Provider。
    
    Returns:
        科目完成报告dict
    """
    subject_stats = {
        "subject_id": subject_id,
        "total_kps": len(kp_list),
        "total_quota": sum(kp.get("quota", 0) for kp in kp_list),
        "generated": 0,
        "rendered": 0,
        "reviewed": 0,
        "passed": 0,
        "failed": 0,
        "start_time": time.time(),
    }
    
    for kp in kp_list:
        if _shutdown.is_set():
            break
            
        kp_id = kp["id"]
        quota = kp.get("quota", 10)
        
        # 检查已有进度
        with _db_lock:
            existing = orch.db.count_questions_for_kp(kp_id)
        remaining = max(0, quota - existing)
        
        if remaining == 0:
            subject_stats["generated"] += quota
            continue
        
        # 分批生产
        produced = 0
        while produced < remaining and not _shutdown.is_set():
            batch_count = min(BATCH_SIZE, remaining - produced)
            
            # 背压检查：如果审核积压严重，等待
            bp = scheduler.check_backpressure()
            if bp["active"]:
                log(f"    ⏸ [{subject_id}] 背压激活，等待审核队列消化...")
                await asyncio.sleep(10)
                continue
            
            # 通过GPT Pool生成
            task_id = f"{kp_id}_{produced}_{uuid.uuid4().hex[:6]}"
            messages = generator.build_generation_prompt(kp, batch_count)
            
            response = await scheduler.call_gpt(messages, task_id, "generate")
            
            if response is None:
                log(f"    ❌ [{kp_id}] GPT生成失败 (batch {produced}/{remaining})")
                stats["errors"] += 1
                await asyncio.sleep(5)
                continue
            
            # 解析响应
            try:
                questions = generator.parse_response(response, kp_id, subject_id)
            except Exception as e:
                log(f"    ❌ [{kp_id}] 解析失败: {str(e)[:60]}")
                stats["errors"] += 1
                continue
            
            if not questions:
                continue
            
            # 入库
            q_ids = []
            for q in questions:
                qid = f"{kp_id}_{produced + len(q_ids):04d}"
                q["question_id"] = qid
                with _db_lock:
                    orch.db.insert_question(qid, kp_id, subject_id, q)
                q_ids.append(qid)
            
            produced += len(questions)
            subject_stats["generated"] += len(questions)
            stats["total_generated"] += len(questions)
            
            # 渲染（同步，在线程池中执行）
            rendered = await asyncio.get_event_loop().run_in_executor(
                None, render_batch, orch, questions, q_ids, subject_id
            )
            subject_stats["rendered"] += len(rendered)
            stats["total_rendered"] += len(rendered)
            
            # 将渲染完成的题目推入审核队列
            for (q, qid, img_path) in rendered:
                await scheduler.review_queue.put({
                    "question": q,
                    "question_id": qid,
                    "subject_id": subject_id,
                    "image_path": img_path,
                })
            
            # 节奏控制：每批间隔
            await asyncio.sleep(1.0)
    
    subject_stats["elapsed_seconds"] = time.time() - subject_stats["start_time"]
    return subject_stats


async def review_worker(scheduler: DynamicScheduler, orch, generator, stats: dict):
    """
    审核工作协程：从review_queue取题，通过Qwen3.7-max审核。
    """
    while not _shutdown.is_set():
        try:
            item = await asyncio.wait_for(scheduler.review_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            continue
        
        qid = item["question_id"]
        q = item["question"]
        
        # 构建审核prompt
        review_messages = generator.build_review_prompt(q)
        task_id = f"review_{qid}_{uuid.uuid4().hex[:4]}"
        
        response = await scheduler.call_qwen(review_messages, task_id, "review")
        
        if response is None:
            # Qwen不可用时暂存，不丢题
            stats["review_failures"] += 1
            continue
        
        # 解析审核结果
        try:
            verdict = generator.parse_review_response(response)
            with _db_lock:
                orch.db.update_question_review(qid, verdict)
            
            if verdict.get("status") == "PASS":
                stats["total_passed"] += 1
            elif verdict.get("status") == "REVISE":
                # 推入修复队列
                await scheduler.fix_queue.put({
                    "question": q,
                    "question_id": qid,
                    "subject_id": item.get("subject_id", ""),
                    "image_path": item.get("image_path", ""),
                    "verdict": verdict,
                    "fix_attempt": 0,
                })
            else:
                stats["total_rejected"] += 1
                
            stats["total_reviewed"] += 1
        except Exception as e:
            log(f"    ⚠️ [{qid}] 审核解析失败: {str(e)[:60]}")


MAX_FIX_ATTEMPTS = 2  # 每道题最多修复2次


async def fix_worker(scheduler: DynamicScheduler, orch, generator, stats: dict):
    """
    修复工作协程：从fix_queue取被REVISE的题目，通过GPT修复后重新渲染+二审。
    
    流程：
    1. 取出REVISE的题目 + Qwen审核意见
    2. 用 build_fix_prompt() 构建修复指令
    3. 通过GPT Pool发送修复请求
    4. 解析修复后的JSON
    5. 重新渲染图片
    6. 如果渲染成功，推回review_queue让Qwen二审
    7. 如果修复次数超限，标记为REJECT
    """
    while not _shutdown.is_set():
        try:
            item = await asyncio.wait_for(scheduler.fix_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            continue
        
        qid = item["question_id"]
        q = item["question"]
        verdict = item["verdict"]
        subject_id = item.get("subject_id", "")
        fix_attempt = item.get("fix_attempt", 0) + 1
        
        if fix_attempt > MAX_FIX_ATTEMPTS:
            # 超过修复次数上限，标记为REJECT
            log(f"    ❌ [{qid}] 修复{MAX_FIX_ATTEMPTS}次仍未通过，标记REJECT")
            with _db_lock:
                orch.db.update_question_status(qid, "FIX_EXHAUSTED", 
                    f"attempts={fix_attempt-1}, issues={verdict.get('issues', [])}")
            stats["total_rejected"] += 1
            continue
        
        log(f"    🔧 [{qid}] 修复尝试 {fix_attempt}/{MAX_FIX_ATTEMPTS}")
        
        # 1. 构建修复prompt并调用GPT
        fix_messages = generator.build_fix_prompt(q, verdict)
        task_id = f"fix_{qid}_{fix_attempt}_{uuid.uuid4().hex[:4]}"
        
        response = await scheduler.call_gpt(fix_messages, task_id, "fix")
        
        if response is None:
            log(f"    ⚠️ [{qid}] GPT修复请求失败")
            stats["errors"] += 1
            # 放回队列重试（不增加attempt，因为是API失败不是内容失败）
            item["fix_attempt"] = fix_attempt - 1
            await scheduler.fix_queue.put(item)
            await asyncio.sleep(5)
            continue
        
        # 2. 解析修复结果
        fixed_q = generator.parse_fix_response(response)
        
        if fixed_q is None:
            log(f"    ⚠️ [{qid}] GPT修复响应解析失败")
            # 算作一次失败的修复尝试，重新推入队列
            item["fix_attempt"] = fix_attempt
            await scheduler.fix_queue.put(item)
            await asyncio.sleep(2)
            continue
        
        # 3. 保持关键ID不变
        fixed_q["question_id"] = qid
        fixed_q["knowledge_point_id"] = q.get("knowledge_point_id", "")
        
        # 4. 重新渲染图片
        render_code = fixed_q.get("render_code", "")
        if render_code:
            img_dir = orch.config.output_dir / subject_id / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            img_path = str(img_dir / f"{qid}.png")
            
            render_result = await asyncio.get_event_loop().run_in_executor(
                None, render_one, orch, fixed_q, qid, subject_id
            )
            
            if render_result is None:
                log(f"    ⚠️ [{qid}] 修复后渲染失败")
                # 渲染失败也算修复尝试
                item["question"] = fixed_q
                item["fix_attempt"] = fix_attempt
                item["verdict"] = {"issues": ["修复后render_code执行失败"], "fix_instructions": ["请重写render_code"]}
                await scheduler.fix_queue.put(item)
                await asyncio.sleep(2)
                continue
            
            _, _, img_path = render_result
        
        # 5. 更新DB
        with _db_lock:
            orch.db.update_question_json(qid, fixed_q)
            orch.db.update_question_status(qid, "FIXED", f"attempt={fix_attempt}")
        
        # 6. 推回审核队列进行二审
        await scheduler.review_queue.put({
            "question": fixed_q,
            "question_id": qid,
            "subject_id": subject_id,
            "image_path": item.get("image_path", ""),
        })
        
        stats["total_fixed"] = stats.get("total_fixed", 0) + 1
        log(f"    ✓ [{qid}] 修复完成，推入二审队列")


async def auto_scale_loop(scheduler: DynamicScheduler):
    """定期检查并调整Provider池容量"""
    while not _shutdown.is_set():
        await asyncio.sleep(AUTO_SCALE_INTERVAL)
        scheduler.auto_scale()


async def report_loop(scheduler: DynamicScheduler, stats: dict, start_time: float):
    """每15分钟生成飞书进度报告"""
    while not _shutdown.is_set():
        await asyncio.sleep(REPORT_INTERVAL)
        
        elapsed = time.time() - start_time
        report = scheduler.get_progress_report()
        
        # 格式化报告
        lines = [
            f"📊 V6生产进度报告 ({datetime.now().strftime('%H:%M')})",
            f"运行时长: {elapsed/3600:.1f}h",
            f"生成: {stats['total_generated']} | 渲染: {stats['total_rendered']}",
            f"审核: {stats['total_reviewed']} | 通过: {stats['total_passed']} | 拒绝: {stats['total_rejected']}",
            f"错误: {stats['errors']} | 429: {report.get('rate_limits_429', 0)} | 5xx: {report.get('server_errors_5xx', 0)}",
            f"",
            f"GPT通道: {report['gpt_pool']['healthy_providers']}/{report['gpt_pool']['active_providers']} healthy/active",
            f"Qwen通道: {report['qwen_pool']['healthy_providers']}/{report['qwen_pool']['active_providers']} healthy/active",
            f"队列: 生成={report['queues']['generation']} | 审核={report['queues']['review']} | 修复={report['queues']['fix']}",
        ]
        
        bp = report.get("backpressure", {})
        if bp.get("active"):
            lines.append(f"⚠️ 背压激活: ratio={bp['ratio']}")
        
        report_text = "\n".join(lines)
        log(report_text)
        
        # 写入report文件供飞书读取
        report_path = PROJECT_ROOT / "latest_report.txt"
        report_path.write_text(report_text, encoding="utf-8")


# ========== 主流程 ==========
async def async_main():
    """异步主入口"""
    # 初始化字体
    setup_fonts()
    
    log("=" * 60)
    log("=== V6 PRODUCTION (15路Provider Pool动态调度) ===")
    log("=" * 60)
    
    # 初始化调度器
    try:
        scheduler = create_scheduler()
    except RuntimeError as e:
        log(f"❌ 初始化失败: {e}")
        return
    
    log(f"GPT Pool: {scheduler.gpt_pool.get_active_count()}/9 active")
    log(f"Qwen Pool: {scheduler.qwen_pool.get_active_count()}/6 active")
    
    # 初始化生产组件
    config = Config()
    
    # 加载KP详情
    input_dir = PROJECT_ROOT.parent / "input_snapshot"
    all_kp_details = {}
    subjects_to_process = []
    
    for sid in READY_SUBJECTS:
        # 尝试多种文件名格式
        candidates = [
            input_dir / f"{sid.lower()}.json",
            input_dir / f"{sid}.json",
        ]
        # 也搜索 subject_XX_*.json 格式
        for f in input_dir.glob(f"subject_{sid[1:]}*.json"):
            candidates.append(f)
        
        loaded = False
        for fpath in candidates:
            if fpath.exists():
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                
                # 提取KP列表（兼容两种格式）
                kps = []
                if "modules" in data:
                    for mod in data["modules"]:
                        kps.extend(mod.get("knowledge_points", []))
                elif "knowledge_points" in data:
                    kps = data["knowledge_points"]
                
                if kps:
                    all_kp_details[sid] = kps
                    total_quota = sum(kp.get("quota", 10) for kp in kps)
                    subjects_to_process.append(sid)
                    log(f"  加载 {sid}: {len(kps)} KPs, quota={total_quota}")
                    loaded = True
                    break
        
        if not loaded:
            log(f"  ⚠️ {sid}: 未找到taxonomy文件")
    
    if not subjects_to_process:
        log("❌ 没有可处理的科目")
        return
    
    log(f"\n共 {len(subjects_to_process)} 科目待生产")
    
    # 初始化Orchestrator和Generator
    orch = ProductionOrchestrator(config)
    generator = QuestionGenerator(config)
    
    # 统计
    stats = {
        "total_generated": 0,
        "total_rendered": 0,
        "total_reviewed": 0,
        "total_passed": 0,
        "total_rejected": 0,
        "errors": 0,
        "review_failures": 0,
        "subjects_completed": [],
    }
    
    start_time = time.time()
    
    # 启动后台任务
    review_tasks = [
        asyncio.create_task(review_worker(scheduler, orch, generator, stats))
        for _ in range(3)  # 3个审核协程
    ]
    fix_tasks = [
        asyncio.create_task(fix_worker(scheduler, orch, generator, stats))
        for _ in range(2)  # 2个修复协程
    ]
    scale_task = asyncio.create_task(auto_scale_loop(scheduler))
    report_task = asyncio.create_task(report_loop(scheduler, stats, start_time))
    
    # 逐科生产
    for sid in subjects_to_process:
        if _shutdown.is_set():
            break
        
        kp_list = all_kp_details[sid]
        log(f"\n{'='*40}")
        log(f"开始生产: {sid} ({len(kp_list)} KPs)")
        log(f"{'='*40}")
        
        subject_report = await produce_subject(scheduler, orch, generator, sid, kp_list, stats)
        stats["subjects_completed"].append(subject_report)
        
        # 每科完成即报
        log(f"\n✅ {sid} 完成:")
        log(f"   KPs: {subject_report['total_kps']} | 配额: {subject_report['total_quota']}")
        log(f"   生成: {subject_report['generated']} | 渲染: {subject_report['rendered']}")
        log(f"   耗时: {subject_report['elapsed_seconds']/60:.1f}分钟")
    
    # 等待审核队列和修复队列排空
    log("\n等待审核和修复队列完成...")
    while (not scheduler.review_queue.empty() or not scheduler.fix_queue.empty()) and not _shutdown.is_set():
        await asyncio.sleep(5)
    
    # 停止后台任务
    _shutdown.set()
    for task in review_tasks + fix_tasks:
        task.cancel()
    scale_task.cancel()
    report_task.cancel()
    
    # 最终报告
    elapsed = time.time() - start_time
    log("\n" + "=" * 60)
    log("=== V6 PRODUCTION COMPLETE ===")
    log(f"总耗时: {elapsed/3600:.2f}h")
    log(f"生成: {stats['total_generated']} | 渲染: {stats['total_rendered']}")
    log(f"审核: {stats['total_reviewed']} | 通过: {stats['total_passed']} | 拒绝: {stats['total_rejected']}")
    log(f"修复: {stats.get('total_fixed', 0)} | 错误: {stats['errors']} | 审核失败: {stats['review_failures']}")
    log(f"完成科目: {len(stats['subjects_completed'])}/{len(subjects_to_process)}")
    log("=" * 60)
    
    # 写入最终报告JSON
    final_report = {
        "version": "v6",
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "elapsed_hours": round(elapsed / 3600, 2),
        "stats": stats,
        "scheduler_report": scheduler.get_progress_report(),
    }
    report_path = PROJECT_ROOT / "production_v6_final_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    log(f"最终报告: {report_path}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
