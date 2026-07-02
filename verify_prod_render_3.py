#!/usr/bin/env python3
"""用生产类 GPTImageRenderer.render() 原样连出 3 张真图, 验证 bug 修复。"""
import os, time, threading
ROOT = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(ROOT, "config", ".env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

from pipeline.image_renderer import GPTImageRenderer

PROMPTS = [
    ("triangle", "a right triangle with legs a and b and hypotenuse c, clean black line art on white background"),
    ("circle",   "a circle with an inscribed square, geometry diagram, clean black line art on white background"),
    ("parabola", "a parabola y=x^2 with labeled vertex and axis, clean black line art on white background"),
]
results = {}
def work(tag, prompt):
    r = GPTImageRenderer()
    out = f"/tmp/prod_verify_{tag}.png"
    t = time.time()
    ok, msg = r.render(prompt, out)
    dt = time.time() - t
    sz = os.path.getsize(out) if (ok and os.path.exists(out)) else 0
    results[tag] = (ok, dt, sz, msg[:200])
    print(f"[{tag}] ok={ok} {dt:.0f}s size={sz} :: {msg[:160]}", flush=True)

ts = [threading.Thread(target=work, args=p) for p in PROMPTS]
for t in ts: t.start()
for t in ts: t.join()

print("\n===== 汇总 =====")
n_ok = sum(1 for v in results.values() if v[0])
for tag,(ok,dt,sz,msg) in results.items():
    print(f"  {tag:9s} {'✅' if ok else '🔴'} {dt:.0f}s {sz} bytes")
print(f"\n{'✅ 全部成功 3/3' if n_ok==3 else f'🔴 仅 {n_ok}/3 成功'}")
