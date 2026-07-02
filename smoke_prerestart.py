#!/usr/bin/env python3
"""Pre-restart smoke test: GPT worker 1-8 + Qwen + GPT image (real render path)."""
import os, sys, json, time, urllib.request, urllib.error, concurrent.futures, tempfile

ROOT = "/home/flyer8258/research_projects/multimodal_question_bank_24x1000/runs/qwen_gpt_closed_loop_18subjects_24000_20260619_v2/final_questionbank_production_18subjects_24000_v5"
sys.path.insert(0, ROOT)

# load .env
env = {}
for line in open(os.path.join(ROOT, "config/.env")):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        env[k] = v.strip().strip('"')
os.environ.update(env)

def chat(base_url, api_key, model, messages, max_tokens=1200, timeout=120):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(url, data=payload, headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read())
        dt = time.time() - t0
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        return (200, dt, len(content or ""), content[:80])
    except urllib.error.HTTPError as e:
        return (e.code, time.time()-t0, 0, e.read()[:120].decode("utf-8","replace"))
    except Exception as e:
        return ("ERR", time.time()-t0, 0, str(e)[:120])

# Production-length prompt (~realistic generation load)
PROD_PROMPT = [
    {"role": "system", "content": "你是一个严格的多模态理科题目生成器。输出必须是合法 JSON，包含 question/options/answer/explanation/image_prompt 字段。"},
    {"role": "user", "content": "请生成一道高中物理多选题，主题为带电粒子在匀强磁场中的圆周运动。"
     "要求：题干完整给出已知量（电荷量 q、质量 m、磁感应强度 B、初速度 v 及方向），"
     "4 个选项 A/B/C/D 覆盖半径、周期、动能、洛伦兹力做功等概念，answer 给正确选项字母，"
     "explanation 给出完整推导（含公式 r=mv/(qB)、T=2πm/(qB)），"
     "image_prompt 用英文描述一张清晰的物理示意图（粒子轨迹、磁场方向、坐标轴）。"
     "严格输出 JSON，不要 markdown 代码块包裹。" * 3}
]

print("="*70)
print("SMOKE TEST — 重启前全链路自检")
print("="*70)

# ---- GPT workers 1-8 ----
print("\n【1】GPT WORKER 1-8 (model=%s, 生产长度 prompt)" % env.get("GPT_MODEL"))
gpt_base = env["GPT_BASE_URL"]; gpt_model = env["GPT_MODEL"]
worker_results = {}
def test_worker(i):
    key = env.get(f"GPT_WORKER{i}_API_KEY")
    if not key:
        return i, ("NOKEY", 0, 0, "no key")
    return i, chat(gpt_base, key, gpt_model, PROD_PROMPT)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    for i, res in ex.map(lambda i: test_worker(i), range(1, 9)):
        worker_results[i] = res
gpt_ok = 0
for i in range(1, 9):
    code, dt, ln, snip = worker_results[i]
    ok = (code == 200 and ln > 200)
    gpt_ok += ok
    print(f"  worker{i}: HTTP={code} {dt:5.1f}s len={ln:<5} {'✅' if ok else '🔴'} {snip if not ok else ''}")
print(f"  → GPT worker 通过: {gpt_ok}/8")

# ---- Qwen ----
print("\n【2】QWEN (model=%s)" % env.get("QWEN_MODEL"))
qcode, qdt, qln, qsnip = chat(env["QWEN_BASE_URL"], env["QWEN_API_KEY"], env["QWEN_MODEL"],
    [{"role":"user","content":"请审核以下题目是否合规并返回 JSON {\"verdict\":\"PASS/FAIL\"}：一道关于洛伦兹力的物理题，题干完整，选项无歧义。"}])
qwen_ok = (qcode == 200 and qln > 5)
print(f"  qwen: HTTP={qcode} {qdt:.1f}s len={qln} {'✅' if qwen_ok else '🔴'} {qsnip}")

# ---- GPT Image (real render path: submit -> poll -> write PNG) ----
print("\n【3】GPT IMAGE (真实出图路径 submit→poll→落盘 PNG, provider=%s)" % env.get("GPT_IMAGE_PROVIDER"))
img_ok = False; img_msg = ""
try:
    from pipeline.image_renderer import GPTImageRenderer
    r = GPTImageRenderer(name="smoke")
    out = os.path.join(tempfile.gettempdir(), "smoke_img.png")
    if os.path.exists(out): os.remove(out)
    t0 = time.time()
    ok, msg = r.render("A clean physics diagram: a charged particle moving in a circular path inside a uniform magnetic field, arrows showing B field direction and velocity, labeled axes, white background.", out)
    dt = time.time() - t0
    sz = os.path.getsize(out) if os.path.exists(out) else 0
    img_ok = ok and sz > 1000
    img_msg = f"ok={ok} {dt:.1f}s file={sz}B msg={msg[:150]}"
    print(f"  image: {'✅' if img_ok else '🔴'} {img_msg}")
except Exception as e:
    import traceback
    print(f"  image: 🔴 EXCEPTION {e}")
    traceback.print_exc()

# ---- summary ----
print("\n" + "="*70)
print("SMOKE TEST 结果汇总")
print("="*70)
print(f"  GPT worker 1-8 : {gpt_ok}/8  {'✅' if gpt_ok==8 else '🔴 不全'}")
print(f"  Qwen           : {'✅ PASS' if qwen_ok else '🔴 FAIL'}")
print(f"  GPT Image      : {'✅ PASS' if img_ok else '🔴 FAIL'}")
all_ok = (gpt_ok==8 and qwen_ok and img_ok)
print(f"\n  整体: {'✅ 全部通过，可重启' if all_ok else '🔴 有失败项，不可重启'}")
sys.exit(0 if all_ok else 1)
