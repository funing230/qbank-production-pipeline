#!/usr/bin/env python3
"""
抽检测试 V2 — 18科目 × 5知识点 = 90道全新题
与上次采样测试 (test_sample_output) 完全不重叠。
"""
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
BATCH_SIZE = 1
MAX_RETRIES = 2  # 每题最多额外重试2次

# API配置
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

# 输出目录（独立于上次测试）
OUTPUT_DIR = PROJECT_ROOT / "spot_check_v2_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ========== 加载KP ==========
def load_all_kps():
    """从 input_snapshot 加载所有科目的KP"""
    input_dir = PROJECT_ROOT.parent / "input_snapshot"
    all_kps = {}

    for sid in SUBJECTS:
        kps = []
        # 尝试多种文件名
        candidates = list(input_dir.glob(f"{sid.lower()}*.json"))
        candidates += list(input_dir.glob(f"subject_{int(sid[1:])}_*.json"))

        for fpath in candidates:
            if not fpath.exists():
                continue
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
                        kp.setdefault("subject_id", sid)
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
                    kp.setdefault("subject_id", sid)
                    kps.append(kp)

            if kps:
                break

        if kps:
            all_kps[sid] = kps
            print(f"  {sid}: {len(kps)} KPs 可用")
        else:
            print(f"  ⚠️ {sid}: 未找到KP数据")

    return all_kps


def get_tested_kps():
    """获取上次测试已经测试过的KP ID集合"""
    tested = set()
    test_dir = PROJECT_ROOT / "test_sample_output"
    if test_dir.is_dir():
        for sub in test_dir.iterdir():
            if sub.is_dir():
                for f in sub.glob("*.json"):
                    # S01_S01-M03-001_000.json -> S01-M03-001
                    parts = f.stem.split("_")
                    if len(parts) >= 2:
                        tested.add(parts[1])
    return tested


def run_spot_check_v2():
    """主测试流程"""
    print("=" * 60)
    print("🧪 抽检测试 V2: 每科5题，全新知识点（0重叠）")
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

    # 获取已测试的KP
    tested_kps = get_tested_kps()
    print(f"上次已测试 KP: {len(tested_kps)} 个")

    # 随机采样（排除已测试的）
    random.seed(2026)
    print(f"\n🎲 每科从未测试过的KP中随机选 {KPS_PER_SUBJECT} 个...")
    sampled = {}
    for sid, kps in all_kps.items():
        untested = [kp for kp in kps if kp.get("id", "") not in tested_kps]
        if len(untested) < KPS_PER_SUBJECT:
            untested = kps  # fallback
        n = min(KPS_PER_SUBJECT, len(untested))
        sampled[sid] = random.sample(untested, n)

    total_tasks = sum(len(v) for v in sampled.values())
    overlap = sum(1 for sid in sampled for kp in sampled[sid] if kp.get("id", "") in tested_kps)
    print(f"共 {total_tasks} 个任务, 与上次重叠: {overlap}")

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
            kp_id = kp.get("id", kp.get("kp_id", "?"))
            print(f"  [{i+1}/{len(kps)}] KP: {kp_id} {kp_name[:40]}...", end=" ", flush=True)

            success = False
            for attempt in range(MAX_RETRIES + 1):
                # 生成题目
                try:
                    questions = generator.generate_batch(kp, batch_size=BATCH_SIZE)
                    if not questions:
                        if attempt < MAX_RETRIES:
                            continue
                        print("❌ 生成失败(空)")
                        stats[sid]["gen_fail"] += 1
                        break
                except Exception as e:
                    if attempt < MAX_RETRIES:
                        continue
                    print(f"❌ 生成异常: {str(e)[:60]}")
                    stats[sid]["gen_fail"] += 1
                    break

                q = questions[0]
                stats[sid]["generated"] += 1

                # 渲染
                qid = f"{sid}_{kp_id}_000"
                img_path = str(img_dir / f"{qid}.png")

                render_instr = q.get("render_instruction")
                render_code = q.get("render_code", "")
                engine_name = "unknown"

                if isinstance(render_instr, dict) and "engine" in render_instr:
                    engine_name = render_instr.get("engine", "unknown")
                    ok, msg = render_from_instruction(render_instr, img_path)
                elif render_code:
                    engine_name = q.get("render_engine", "MATPLOTLIB")
                    ok, msg = render_question_image(
                        render_code=render_code,
                        output_path=img_path,
                        render_engine=engine_name,
                        timeout=60
                    )
                else:
                    if attempt < MAX_RETRIES:
                        continue
                    print("❌ 无渲染数据")
                    stats[sid]["render_fail"] += 1
                    break

                stats[sid]["engines_used"][engine_name] += 1

                if not ok:
                    if attempt < MAX_RETRIES:
                        continue
                    print(f"❌ 渲染失败[{engine_name}]: {msg[:40] if isinstance(msg, str) else msg}")
                    stats[sid]["render_fail"] += 1
                    break

                stats[sid]["rendered"] += 1

                # 质量关卡
                diagram_meta = render_instr if isinstance(render_instr, dict) else None
                gate_ok, issues = quality_gate(img_path, diagram_meta)
                if not gate_ok:
                    if attempt < MAX_RETRIES:
                        continue
                    stats[sid]["gate_fail"] += 1
                    print(f"⚠️ [{engine_name}] 关卡未过: {issues[0][:40]}")
                    # 仍然保存，标记gate_fail
                    q["_gate_failed"] = True
                    q["_gate_issues"] = issues

                if gate_ok:
                    stats[sid]["passed"] += 1
                    print(f"✅ [{engine_name}]")

                # 保存题目JSON
                q["image_path"] = f"images/{qid}.png"
                q_file = OUTPUT_DIR / sid / f"{qid}.json"
                q_file.write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")
                success = True
                break

    # ========== 最终报告 ==========
    elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print("📊 抽检测试 V2 结果")
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
    print(f"\n渲染引擎分布:")
    for eng, cnt in sorted(all_engines.items(), key=lambda x: -x[1]):
        print(f"  {eng}: {cnt}")

    # 保存报告
    report = {
        "total_tasks": total_tasks,
        "elapsed_seconds": elapsed,
        "generated": total_gen,
        "rendered": total_rendered,
        "passed": total_passed,
        "render_fail": total_render_fail,
        "gate_fail": total_gate_fail,
        "gen_fail": total_gen_fail,
        "engines": dict(all_engines),
        "per_subject": {sid: dict(s) for sid, s in stats.items()},
    }
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n报告已保存: {OUTPUT_DIR / 'report.json'}")


if __name__ == "__main__":
    run_spot_check_v2()
