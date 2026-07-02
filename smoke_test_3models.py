#!/usr/bin/env python3
"""Smoke test for the 3 production models: GPT(text) / Qwen(text) / Image(media)."""
import json, os, time, urllib.request, urllib.error, urllib.parse, pathlib

ROOT = pathlib.Path(__file__).parent
# load config/.env
env = {}
for line in (ROOT / "config" / ".env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    env[k.strip()] = v.strip()

def mask(k):
    return (k[:6] + "***" + k[-4:]) if k and len(k) > 12 else "***"

def post_json(url, key, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode()), time.time() - t0

def get_json(url, key, timeout):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())

def chat_test(label, base, key, model):
    print(f"\n=== {label} ===")
    print(f"  endpoint : {base}/chat/completions")
    print(f"  model    : {model}")
    print(f"  key      : {mask(key)}")
    payload = {"model": model, "messages": [{"role": "user",
        "content": "用一句话说明勾股定理。只输出这句话。"}], "max_tokens": 200}
    try:
        st, body, dt = post_json(f"{base}/chat/completions", key, payload, 120)
        msg = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"  RESULT   : HTTP={st} {dt:.1f}s len={len(msg)}")
        print(f"  reply    : {msg[:120]}")
        print(f"  VERDICT  : {'✅ PASS' if st==200 and msg else '🔴 FAIL'}")
    except urllib.error.HTTPError as e:
        print(f"  RESULT   : 🔴 HTTP {e.code} {e.read().decode()[:200]}")
    except Exception as e:
        print(f"  RESULT   : 🔴 {type(e).__name__}: {str(e)[:200]}")

def image_test(label, base, key, model):
    print(f"\n=== {label} ===")
    print(f"  endpoint : {base}/media/generate  (异步出图接口, 非 chat)")
    print(f"  model    : {model}")
    print(f"  key      : {mask(key)}")
    payload = {"model": model, "prompt": "a simple right triangle on white background, clean line art",
               "size": "1024x1024", "n": 1}
    try:
        st, body, dt = post_json(f"{base}/media/generate", key, payload, 120)
        print(f"  SUBMIT   : HTTP={st} {dt:.1f}s body={json.dumps(body, ensure_ascii=False)[:300]}")
        # extract task id
        tid = None
        for k in ("task_id", "id", "taskId"):
            tid = body.get(k) or (body.get("data") or {}).get(k) if isinstance(body.get("data"), dict) else body.get(k)
            if tid: break
        if not tid and isinstance(body.get("data"), dict):
            tid = body["data"].get("task_id") or body["data"].get("id")
        if not tid:
            # maybe synchronous image url
            print(f"  VERDICT  : ⚠️ no task_id; inspect body above")
            return
        print(f"  task_id  : {tid}")
        deadline = time.time() + 120
        last = None
        while time.time() < deadline:
            for path in (f"media/generate/{tid}", f"media/task/{tid}", f"media/result/{tid}"):
                try:
                    s2, b2 = get_json(f"{base}/{path}", key, 60)
                    last = b2
                    blob = json.dumps(b2, ensure_ascii=False)
                    if "http" in blob and (".png" in blob or ".jpg" in blob or "result_url" in blob):
                        # check there is a real url
                        prog = str((b2.get("data") or b2).get("progress", ""))
                        url = (b2.get("data") or b2).get("result_url") or ""
                        if url:
                            print(f"  POLL     : ✅ image ready url={url[:80]}")
                            print(f"  VERDICT  : ✅ PASS")
                            return
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        continue
                    raise
            time.sleep(5)
        print(f"  POLL     : 🔴 timeout 120s, last={json.dumps(last, ensure_ascii=False)[:300]}")
        print(f"  VERDICT  : 🔴 FAIL (no image)")
    except urllib.error.HTTPError as e:
        print(f"  SUBMIT   : 🔴 HTTP {e.code} {e.read().decode()[:200]}")
    except Exception as e:
        print(f"  SUBMIT   : 🔴 {type(e).__name__}: {str(e)[:200]}")

# 1) GPT text (worker1 key)
chat_test("1) GPT 文本生成", env["GPT_BASE_URL"], env.get("GPT_WORKER1_API_KEY", env["GPT_API_KEY"]), env["GPT_MODEL"])
# 2) Qwen text (review)
chat_test("2) Qwen 审核", env["QWEN_BASE_URL"], env["QWEN_API_KEY"], env["QWEN_MODEL"])
# 3) Image (media/generate)
image_test("3) 出图 (lk888 media)", env["GPT_IMAGE_BASE_URL"], env["GPT_IMAGE_API_KEY"], env["GPT_IMAGE_MODEL"])
print("\n=== smoke test done ===")
