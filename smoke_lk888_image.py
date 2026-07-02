#!/usr/bin/env python3
"""Smoke test: lk888 出图两个模型 (gpt-image-2 / gpt-image-2-medium), 真实落盘 PNG."""
import os, sys, time, tempfile

ROOT = "/home/flyer8258/research_projects/multimodal_question_bank_24x1000/runs/qwen_gpt_closed_loop_18subjects_24000_20260619_v2/final_questionbank_production_18subjects_24000_v5"
sys.path.insert(0, ROOT)

env = {}
for line in open(os.path.join(ROOT, "config/.env")):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        env[k] = v.strip().strip('"')
os.environ.update(env)
# smoke 用更短超时，避免 4 个挂死用例各等满 240s
os.environ["GPT_IMAGE_LK888_TIMEOUT"] = "90"

from pipeline.image_renderer import GPTImageRenderer

PROMPT = ("A clean physics diagram: a charged particle moving in a circular path inside a "
          "uniform magnetic field, arrows showing B field direction and velocity, labeled axes, white background.")

# lk888 的两个模型 + 两组 key
cases = [
    ("gpt-image-2 (主key GPT_IMAGE_API_KEY)",        env["GPT_IMAGE_BASE_URL"], env["GPT_IMAGE_API_KEY"],  "gpt-image-2"),
    ("gpt-image-2-medium (主key GPT_IMAGE_API_KEY)",  env["GPT_IMAGE_BASE_URL"], env["GPT_IMAGE_API_KEY"],  "gpt-image-2-medium"),
    ("gpt-image-2 (key2 LK888_IMAGE_API_KEY_2)",      env.get("LK888_IMAGE_BASE_URL_2", env["GPT_IMAGE_BASE_URL"]), env.get("LK888_IMAGE_API_KEY_2",""), "gpt-image-2"),
    ("gpt-image-2-medium (key2 LK888_IMAGE_API_KEY_2)", env.get("LK888_IMAGE_BASE_URL_2", env["GPT_IMAGE_BASE_URL"]), env.get("LK888_IMAGE_API_KEY_2",""), "gpt-image-2-medium"),
]

print("="*72)
print("LK888 出图 SMOKE TEST — 两个模型真实落盘 (submit→poll→PNG)")
print("base=%s  超时=%ss  轮询间隔=%ss" % (env["GPT_IMAGE_BASE_URL"], env.get("GPT_IMAGE_LK888_TIMEOUT"), env.get("GPT_IMAGE_LK888_POLL_INTERVAL")))
print("="*72)

results = []
for label, base, key, model in cases:
    print(f"\n>>> {label}")
    if not key:
        print("    🔴 SKIP: 无 key")
        results.append((label, False, "no key")); continue
    r = GPTImageRenderer(base_url=base, api_key=key, model=model, provider="lk888", name="smoke")
    out = os.path.join(tempfile.gettempdir(), f"lk_smoke_{model}_{int(time.time())}.png")
    if os.path.exists(out): os.remove(out)
    t0 = time.time()
    try:
        ok, msg = r.render(PROMPT, out)
    except Exception as e:
        ok, msg = False, f"EXCEPTION {e}"
    dt = time.time() - t0
    sz = os.path.getsize(out) if os.path.exists(out) else 0
    passed = ok and sz > 1000
    print(f"    {'✅ PASS' if passed else '🔴 FAIL'}  {dt:.1f}s  file={sz}B")
    print(f"    msg: {str(msg)[:200]}")
    results.append((label, passed, f"{dt:.1f}s {sz}B {str(msg)[:120]}"))

print("\n" + "="*72)
print("汇总")
print("="*72)
for label, passed, info in results:
    print(f"  {'✅' if passed else '🔴'} {label}")
n_ok = sum(1 for _,p,_ in results if p)
print(f"\n  通过 {n_ok}/{len(results)} 条 lk888 出图通道")
sys.exit(0 if n_ok>0 else 1)
