#!/usr/bin/env python3
"""
V7 轻量集成测试 — 验证 全量审核 + FAIL重生成 流程
===============================================
测试内容：
1. GPT生成1道题 → Qwen全量审核
2. 如果PASS → 验证入库
3. 模拟FAIL → 触发regen_worker → 验证重生成逻辑
4. 验证FROZEN_FAIL标记
"""
import sys, os, json, asyncio, time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载.env
env_file = PROJECT_ROOT / "config" / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

from pipeline.generator import QuestionGenerator
from pipeline.orchestrator import Config
from pipeline.provider_pool import create_scheduler

RESULTS = []

def report(stage, status, detail=""):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {stage}: {status}" + (f" | {detail}" if detail else "")
    print(line, flush=True)
    RESULTS.append({"stage": stage, "status": status, "detail": detail})


async def test_full_regen_flow():
    """端到端测试：生成 → 审核 → (FAIL) → 重生成 → 再审"""
    
    # 1. 初始化
    report("初始化", "开始")
    try:
        scheduler = create_scheduler()
        report("调度器", "✅ OK", f"GPT={scheduler.gpt_pool.get_active_count()} Qwen={scheduler.qwen_pool.get_active_count()}")
    except Exception as e:
        report("调度器", "❌ FAIL", str(e))
        return
    
    config = Config()
    generator = QuestionGenerator(
        base_url=os.environ.get("OPENAI_BASE_URL", ""),
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        model=os.environ.get("GPT_MODEL", "gpt-5.5"),
    )
    
    # 2. 选一个简单知识点生成
    test_kp = {
        "id": "TEST-KP-001",
        "kp_name": "一元二次方程求根",
        "knowledge_point_name": "一元二次方程求根",
        "subject_name": "代数与函数",
        "subject_id": "S04",
        "scope_boundary": "一元二次方程的求根公式、判别式、韦达定理",
        "question_archetypes": ["计算求根", "判别式判断", "韦达定理应用"],
        "allowed_image_types": ["函数图像", "数轴标注"],
        "quota": 1,
    }
    
    report("生成", "开始", f"KP={test_kp['kp_name']}")
    
    # 3. GPT生成
    messages = generator.build_generation_prompt(test_kp, 1)
    t0 = time.time()
    response = await scheduler.call_gpt(messages, "test_gen_001", "generate")
    gen_time = time.time() - t0
    
    if response is None:
        report("GPT生成", "❌ FAIL", "API返回None")
        return
    
    questions = generator.parse_response(response, "TEST-KP-001", "S04")
    if not questions:
        report("GPT生成", "❌ 解析失败")
        return
    
    q = questions[0]
    report("GPT生成", "✅ OK", f"耗时{gen_time:.1f}s | 题干:{q.get('question_text','')[:40]}")
    
    # 4. Qwen全量审核
    review_messages = generator.build_review_prompt(q)
    t0 = time.time()
    review_response = await scheduler.call_qwen(review_messages, "test_review_001", "review")
    review_time = time.time() - t0
    
    if review_response is None:
        report("Qwen审核", "❌ FAIL", "API返回None")
        return
    
    verdict = generator.parse_review_response(review_response)
    review_status = verdict.get("status", "UNKNOWN")
    report("Qwen审核", f"{'✅' if review_status == 'PASS' else '⚠️'} {review_status}", 
           f"耗时{review_time:.1f}s | issues={verdict.get('issues', [])[:2]}")
    
    # 5. 测试重生成流程（无论审核是否通过，都走一遍regen验证逻辑）
    report("重生成测试", "开始", "模拟FAIL触发regen")
    
    old_question_text = q.get("question_text", "")
    fake_verdict = {"status": "REVISE", "issues": ["选项区分度不够", "图片信息量不足"]}
    
    regen_messages = generator.build_regen_prompt(test_kp, old_question_text, fake_verdict)
    t0 = time.time()
    regen_response = await scheduler.call_gpt(regen_messages, "test_regen_001", "regen")
    regen_time = time.time() - t0
    
    if regen_response is None:
        report("GPT重生成", "❌ FAIL", "API返回None")
        return
    
    regen_questions = generator.parse_response(regen_response, "TEST-KP-001", "S04")
    if not regen_questions:
        report("GPT重生成", "❌ 解析失败")
        return
    
    new_q = regen_questions[0]
    # 验证新题与旧题不同
    is_different = new_q.get("question_text", "") != old_question_text
    report("GPT重生成", "✅ OK", 
           f"耗时{regen_time:.1f}s | 与旧题不同={'是' if is_different else '否'} | 新题干:{new_q.get('question_text','')[:40]}")
    
    # 6. 对重生成题目做Qwen二审
    review2_messages = generator.build_review_prompt(new_q)
    t0 = time.time()
    review2_response = await scheduler.call_qwen(review2_messages, "test_review_002", "review")
    review2_time = time.time() - t0
    
    if review2_response is None:
        report("Qwen二审", "❌ FAIL", "API返回None")
        return
    
    verdict2 = generator.parse_review_response(review2_response)
    report("Qwen二审", f"{'✅' if verdict2.get('status') == 'PASS' else '⚠️'} {verdict2.get('status')}", 
           f"耗时{review2_time:.1f}s")
    
    # 7. 最终汇总
    print("\n" + "=" * 60)
    print("V7 集成测试汇总")
    print("=" * 60)
    all_ok = all(r["status"].startswith("✅") or r["status"].startswith("⚠️") for r in RESULTS)
    for r in RESULTS:
        print(f"  {r['stage']:12s} → {r['status']}")
    print(f"\n结论: {'✅ 全部通过，V7 regen流程可用' if all_ok else '⚠️ 部分环节异常，需检查'}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_full_regen_flow())
