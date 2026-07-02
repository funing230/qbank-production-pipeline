#!/usr/bin/env python3
"""两个出图通道各跑一次, 全程打印真实状态, 判定 模型 vs 代码。"""
import os, json, time, urllib.request, urllib.error, threading

ROOT = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(ROOT, "config", ".env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

def mask(k): return (k[:6]+"***"+k[-4:]) if k and len(k)>12 else "***"

def run_channel(label, base, key, model, size):
    def post(u, p):
        d = json.dumps(p).encode()
        r = urllib.request.Request(u, data=d, method="POST",
            headers={"Content-Type":"application/json","Authorization":f"Bearer {key}"})
        return json.loads(urllib.request.urlopen(r, timeout=120).read())
    def get(u):
        r = urllib.request.Request(u, headers={"Authorization":f"Bearer {key}"})
        return json.loads(urllib.request.urlopen(r, timeout=60).read())
    log = []
    def p(s): log.append(s)
    p(f"\n{'='*58}\n[{label}] base={base}\n  model={model} size={size} key={mask(key)}")
    try:
        b = post(f"{base}/media/generate", {"model":model,"prompt":"a right triangle with sides a b c, clean black line art on white background","size":size,"n":1})
        p(f"  SUBMIT -> {json.dumps(b, ensure_ascii=False)[:200]}")
        tid = b.get("data",{}).get("task_id") or b.get("task_id")
        if not tid:
            p(f"  🔴 no task_id"); print("\n".join(log)); return
        p(f"  task_id={tid}")
    except urllib.error.HTTPError as e:
        p(f"  🔴 SUBMIT HTTP {e.code}: {e.read().decode()[:200]}"); print("\n".join(log)); return
    except Exception as e:
        p(f"  🔴 SUBMIT {type(e).__name__}: {e}"); print("\n".join(log)); return
    deadline = time.time() + 280
    n = 0
    while time.time() < deadline:
        time.sleep(8); n += 1
        got = False
        for path in (f"media/generate/{tid}", f"media/task/{tid}", f"media/result/{tid}"):
            try:
                r = get(f"{base}/{path}")
                d = r.get("data", r)
                prog = d.get("progress"); st = d.get("state"); url = d.get("result_url") or ""
                p(f"  [{n*8:3d}s] state={st} status={d.get('status')} progress={prog} is_final={d.get('is_final')} url={url[:50]}")
                got = True
                if url:
                    p(f"  ✅✅ IMAGE READY after {n*8}s: {url[:80]}")
                    print("\n".join(log)); return
                break
            except urllib.error.HTTPError as e:
                if e.code == 404: continue
                p(f"  query HTTP {e.code}"); break
        if not got:
            p(f"  [{n*8:3d}s] all query paths 404")
    p(f"  🔴 TIMEOUT 280s — progress 始终未推进")
    print("\n".join(log))

ch = [
    ("通道1 GPT_IMAGE", os.environ["GPT_IMAGE_BASE_URL"], os.environ["GPT_IMAGE_API_KEY"], os.environ["GPT_IMAGE_MODEL"], "1024x1024"),
    ("通道2 LK888_2",    os.environ["LK888_IMAGE_BASE_URL_2"], os.environ["LK888_IMAGE_API_KEY_2"], os.environ["LK888_IMAGE_MODEL_2"], os.environ.get("LK888_IMAGE_SIZE_2","1024x1024")),
]
ts = [threading.Thread(target=run_channel, args=c) for c in ch]
for t in ts: t.start()
for t in ts: t.join()
print("\n=== 双通道测试结束 ===")
