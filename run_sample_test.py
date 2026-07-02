#!/usr/bin/env python3
"""
采样测试脚本：每科 5 题，验证新模板渲染架构的效果。
- 从 18 科目中各随机选 5 个 KP
- 每个 KP 生成 1 题（batch_size=1）
- 渲染图片 + 质量关卡
- 输出统计报告
"""

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from collections import defaultdict

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.generator import QuestionGenerator
from pipeline.render_router import render_from_instruction
from pipeline.render_executor import render_question_image
from pipeline.image_quality_gate import quality_gate
from pipeline.global_style import apply_global_style

# 初始化全局风格
apply_global_style()

# ========== 配置 ==========
SUBJECTS = [
    "S01", "S02", "S03", "S04", "S05", "S06",
    "S13", "S14", "S15", "S16", "S17", "S18",
    "S19", "S20", "S21", "S22", "S23", "S24"
]
KPS_PER_SUBJECT = 5
BATCH_SIZE = 1  # 每个KP生成1题

# API配置 — 从 config/.env 加载
def _load_env():
    env_file = PROJECT_ROOT / "config" / ".env"
    env = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"')
    return env

_env = _load_env()
GPT_BASE_URL = _env.get("GPT_BASE_URL", "https://api.lk888.ai/v1")
GPT_API_KEY = _env.get("GPT_API_KEY", "")
GPT_MODEL = _env.get("GPT_MODEL", "gpt-5.5")

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "test_sample_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ========== 加载KP ==========
def load_all_kps():
    """从taxonomy加载所有科目的KP"""
    tax_dir = PROJECT_ROOT.parent.parent.parent / "taxonomy_versions" / "v2_2_master" / "subjects"
    all_kps = {}
    
    for sid in SUBJECTS:
        kps = []
        # 尝试多种文件名
        candidates = [
            tax_dir / f"{sid.lower()}.json",
        ]
        # subject_XX_name.json 格式
        num = sid[1:]  # "01", "14" etc
        for f in tax_dir.glob(f"subject_{int(num)}_*.json"):
            candidates.append(f)
        
        for fpath in candidates:
            if fpath.exists():
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                
                subject_name = data.get("subject_name", data.get("name", sid))
                
                if "modules" in data:
                    for mod in data["modules"]:
                        mod_name = mod.get("module_name", mod.get("name", ""))
                        mod_id = mod.get("module_id", mod.get("id", ""))
                        for kp in mod.get("knowledge_points", []):
                            kp["_subject_name"] = subject_name
                            kp["_subject_id"] = sid
                            kp["module_name"] = mod_name
                            kp["module_id"] = mod_id
                            kp.setdefault("id", kp.get("knowledge_point_id", ""))
                            kp.setdefault("kp_name", kp.get("knowledge_point_name", ""))
                            kp.setdefault("kp_id", kp.get("knowledge_point_id", ""))
                            kp.setdefault("quota", 1)
                            kp.setdefault("subject_name", subject_name)
                            kps.append(kp)
                
                if not kps and "knowledge_points" in data:
                    for kp in data["knowledge_points"]:
                        kp["_subject_name"] = subject_name
                        kp["_subject_id"] = sid
                        kp.setdefault("module_name", "")
                        kp.setdefault("module_id", "")
                        kp.setdefault("id", kp.get("knowledge_point_id", ""))
                        kp.setdefault("kp_name", kp.get("knowledge_point_name", ""))
                        kp.setdefault("kp_id", kp.get("knowledge_point_id", ""))
                        kp.setdefault("quota", 1)
                        kp.setdefault("subject_name", subject_name)
                        kps.append(kp)
                
                if kps:
                    break
        
        if kps:
            all_kps[sid] = kps
            print(f"  {sid}: {len(kps)} KPs 可用")
        else:
            print(f"  ⚠️ {sid}: 未找到KP数据")
    
    return all_kps


def run_sample_test():
    """主测试流程"""
    print("=" * 60)
    print("🧪 采样测试: 每科5题，验证模板渲染架构")
    print("=" * 60)
    
    # 读取API key
    api_key = GPT_API_KEY
    if not api_key:
        print("❌ 未找到 API key，请在 config/.env 中设置 GPT_API_KEY")
        return
    
    # 加载KP
    print("\n📚 加载科目KP...")
    all_kps = load_all_kps()
    print(f"\n加载了 {len(all_kps)} 个科目")
    
    # 随机采样
    print(f"\n🎲 每科随机选 {KPS_PER_SUBJECT} 个KP...")
    sampled = {}
    for sid, kps in all_kps.items():
        n = min(KPS_PER_SUBJECT, len(kps))
        sampled[sid] = random.sample(kps, n)
    
    total_tasks = sum(len(v) for v in sampled.values())
    print(f"共 {total_tasks} 个任务")
    
    # 初始化生成器
    generator = QuestionGenerator(
        base_url=GPT_BASE_URL,
        api_key=api_key,
        model=GPT_MODEL,
    )
    
    # 统计
    stats = defaultdict(lambda: {"generated": 0, "rendered": 0, "passed": 0, 
                                  "render_fail": 0, "gate_fail": 0, "gen_fail": 0,
                                  "engines_used": defaultdict(int)})
    overall_start = time.time()
    
    # 逐科目处理
    for sid in sorted(sampled.keys()):
        kps = sampled[sid]
        subject_name = kps[0].get("_subject_name", sid)
        img_dir = OUTPUT_DIR / sid / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*40}")
        print(f"📝 {sid} ({subject_name}) — {len(kps)} 题")
        print(f"{'='*40}")
        
        for i, kp in enumerate(kps):
            kp_name = kp.get("kp_name", kp.get("id", "?"))
            print(f"  [{i+1}/{len(kps)}] KP: {kp_name[:40]}...", end=" ", flush=True)
            
            # 生成题目
            try:
                questions = generator.generate_batch(kp, batch_size=BATCH_SIZE)
                if not questions:
                    print("❌ 生成失败(空)")
                    stats[sid]["gen_fail"] += 1
                    continue
            except Exception as e:
                print(f"❌ 生成异常: {str(e)[:60]}")
                stats[sid]["gen_fail"] += 1
                continue
            
            stats[sid]["generated"] += len(questions)
            
            # 渲染每道题
            for qi, q in enumerate(questions):
                qid = f"{sid}_{kp.get('id', 'X')}_{qi:03d}"
                img_path = str(img_dir / f"{qid}.png")
                
                render_instr = q.get("render_instruction")
                render_code = q.get("render_code", "")
                engine_name = "unknown"
                
                if isinstance(render_instr, dict) and "engine" in render_instr:
                    # 新格式：走路由器
                    engine_name = render_instr.get("engine", "unknown")
                    ok, msg = render_from_instruction(render_instr, img_path)
                elif render_code:
                    # 旧格式：走子进程执行
                    engine_name = q.get("render_engine", "MATPLOTLIB")
                    ok, _ = render_question_image(
                        render_code=render_code,
                        output_path=img_path,
                        render_engine=engine_name,
                        timeout=60
                    )
                    msg = "旧格式渲染" if ok else "旧格式渲染失败"
                else:
                    print(f"❌ 无渲染数据")
                    stats[sid]["render_fail"] += 1
                    continue
                
                stats[sid]["engines_used"][engine_name] += 1
                
                if not ok:
                    print(f"❌ 渲染失败[{engine_name}]: {msg[:40]}")
                    stats[sid]["render_fail"] += 1
                    continue
                
                stats[sid]["rendered"] += 1
                
                # 质量关卡
                gate_ok, issues = quality_gate(img_path)
                if gate_ok:
                    stats[sid]["passed"] += 1
                    print(f"✅ [{engine_name}]")
                else:
                    stats[sid]["gate_fail"] += 1
                    print(f"⚠️ [{engine_name}] 关卡未过: {issues[0][:40]}")
                
                # 保存题目JSON（含image_path）
                q["image_path"] = f"images/{qid}.png"
                q_file = OUTPUT_DIR / sid / f"{qid}.json"
                q_file.write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # ========== 最终报告 ==========
    elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print("📊 采样测试结果")
    print("=" * 60)
    
    total_gen = sum(s["generated"] for s in stats.values())
    total_rendered = sum(s["rendered"] for s in stats.values())
    total_passed = sum(s["passed"] for s in stats.values())
    total_render_fail = sum(s["render_fail"] for s in stats.values())
    total_gate_fail = sum(s["gate_fail"] for s in stats.values())
    total_gen_fail = sum(s["gen_fail"] for s in stats.values())
    
    print(f"\n总计: {total_tasks} 任务 | 耗时 {elapsed:.0f}s")
    print(f"  生成成功: {total_gen}/{total_tasks}")
    print(f"  渲染成功: {total_rendered}/{total_gen}")
    print(f"  质量通过: {total_passed}/{total_rendered}")
    print(f"  渲染失败: {total_render_fail}")
    print(f"  关卡失败: {total_gate_fail}")
    print(f"  生成失败: {total_gen_fail}")
    
    # 渲染引擎分布
    all_engines = defaultdict(int)
    for s in stats.values():
        for eng, cnt in s["engines_used"].items():
            all_engines[eng] += cnt
    
    print(f"\n🔧 引擎使用分布:")
    for eng, cnt in sorted(all_engines.items(), key=lambda x: -x[1]):
        print(f"  {eng}: {cnt}")
    
    # 各科目详情
    print(f"\n📋 各科目详情:")
    print(f"  {'科目':<8} {'生成':>4} {'渲染':>4} {'通过':>4} {'通过率':>6}")
    print(f"  {'-'*36}")
    for sid in sorted(stats.keys()):
        s = stats[sid]
        rate = f"{s['passed']/max(s['generated'],1)*100:.0f}%" if s['generated'] else "N/A"
        print(f"  {sid:<8} {s['generated']:>4} {s['rendered']:>4} {s['passed']:>4} {rate:>6}")
    
    overall_rate = total_passed / max(total_gen, 1) * 100
    print(f"\n  {'总体':>8} {total_gen:>4} {total_rendered:>4} {total_passed:>4} {overall_rate:.0f}%")
    
    # 保存报告
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "total_tasks": total_tasks,
        "total_generated": total_gen,
        "total_rendered": total_rendered,
        "total_passed": total_passed,
        "total_render_fail": total_render_fail,
        "total_gate_fail": total_gate_fail,
        "total_gen_fail": total_gen_fail,
        "overall_pass_rate": overall_rate,
        "engines": dict(all_engines),
        "per_subject": {sid: dict(s) for sid, s in stats.items()},
    }
    # defaultdict不能直接json序列化，转换一下
    for sid in report["per_subject"]:
        report["per_subject"][sid]["engines_used"] = dict(report["per_subject"][sid]["engines_used"])
    
    report_file = OUTPUT_DIR / "test_report.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 报告已保存: {report_file}")
    print(f"📁 图片目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    random.seed(42)  # 可复现
    run_sample_test()
