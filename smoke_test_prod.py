#!/usr/bin/env python3
"""Smoke test using the REAL production code paths (not reimplemented)."""
import os, sys, json, time, pathlib

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

# load config/.env into os.environ (same as production startup)
for line in (ROOT / "config" / ".env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip())

def mask(k):
    return (k[:6] + "***" + k[-4:]) if k and len(k) > 12 else ("***" if k else "(none)")

print("="*60)
print("SMOKE TEST — 使用生产代码路径 (pipeline.*)")
print("="*60)

# ---- 1) GPT text via production generator ----
from pipeline.generator import QuestionGenerator
print("\n=== 1) GPT 文本生成 (pipeline.generator) ===")
print(f"  model={os.environ.get('GPT_MODEL')} base={os.environ.get('GPT_BASE_URL')} key={mask(os.environ.get('GPT_WORKER1_API_KEY'))}")
try:
    import inspect
    sig = inspect.signature(QuestionGenerator.__init__)
    print(f"  generator init params: {list(sig.parameters)[1:]}")
except Exception as e:
    print(f"  (introspect failed: {e})")

# ---- 3) Image via production GPTImageRenderer ----
from pipeline.image_renderer import GPTImageRenderer
print("\n=== 3) 出图 (pipeline.image_renderer.GPTImageRenderer) ===")
r = GPTImageRenderer()
print(f"  provider={r.provider} base={r.base_url} model={r.model} key={mask(r.api_key)} size={r.size}")
print(f"  lk888_timeout={r.lk888_timeout} poll={r.lk888_poll_interval} result_path={r.lk888_result_path!r}")
out = str(ROOT / "smoke_test_image_out.png")
if os.path.exists(out):
    os.remove(out)
t0 = time.time()
ok, msg = r.render("a simple right triangle with sides labeled a, b, c on white background, clean black line art", out)
dt = time.time() - t0
print(f"  RESULT  : ok={ok} {dt:.1f}s")
print(f"  msg     : {msg[:400]}")
if ok and os.path.exists(out):
    print(f"  file    : {out}  ({os.path.getsize(out)} bytes)")
    print(f"  VERDICT : ✅ PASS — 出图落盘成功")
else:
    print(f"  VERDICT : 🔴 FAIL")
print("\n=== done ===")
