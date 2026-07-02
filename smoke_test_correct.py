#!/usr/bin/env python3
"""穷尽式正确异步出图测试: 用 task_id + 对话组ID + OpenAI同步式 三路探测, 打印真实响应体。"""
import os, json, time, urllib.request, urllib.error, urllib.parse

ROOT = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(ROOT, "config", ".env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

BASE = os.environ["GPT_IMAGE_BASE_URL"].rstrip("/")
KEY  = os.environ["GPT_IMAGE_API_KEY"]
MODEL = os.environ["GPT_IMAGE_MODEL"]
SIZE = "1024x1024"
PROMPT = "a right triangle with sides labeled a b c, clean black line art on white background"

def post(url, payload, timeout=120):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST",
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.getcode(), r.read().decode("utf-8","replace")

def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"Authorization":f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.getcode(), r.read().decode("utf-8","replace")

print(f"BASE={BASE} MODEL={MODEL}")

# ---------- TEST A: OpenAI 同步式 /images/generations ----------
print("\n========== TEST A: /images/generations (OpenAI sync) ==========")
try:
    code, body = post(f"{BASE}/images/generations", {"model":MODEL,"prompt":PROMPT,"size":SIZE,"n":1})
    print(f"  HTTP {code}: {body[:400]}")
except urllib.error.HTTPError as e:
    print(f"  HTTP {e.code}: {e.read().decode('utf-8','replace')[:400]}")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

# ---------- TEST B: /media/generate 下单 + 穷尽轮询 ----------
print("\n========== TEST B: /media/generate async ==========")
try:
    code, body = post(f"{BASE}/media/generate", {"model":MODEL,"prompt":PROMPT,"size":SIZE,"n":1})
    print(f"  SUBMIT HTTP {code}: {body[:400]}")
    j = json.loads(body)
    data = j.get("data", j)
    task_id = str(data.get("task_id") or (data.get("任务ids") or [""])[0] or "")
    group_id = str(data.get("对话组ID") or data.get("group_id") or "")
    print(f"  task_id={task_id}  group_id={group_id}")
except Exception as e:
    print(f"  SUBMIT failed: {e}"); raise SystemExit

ids = {"task_id": task_id, "group_id": group_id}
# 候选查询路径模板, 用 task_id 和 group_id 各试一遍
templates = [
    "media/generate/{v}", "media/task/{v}", "media/tasks/{v}", "media/result/{v}",
    "media/results/{v}", "media/status/{v}", "media/query/{v}", "media/get/{v}",
    "media/generations/{v}", "media/group/{v}", "media/conversation/{v}",
    "media/status?task_id={v}", "media/result?task_id={v}", "media/query?id={v}",
    "media/generate?task_id={v}", "media/generate?group_id={v}", "media/generate?id={v}",
]

found_paths = []
deadline = time.time() + 200
n = 0
while time.time() < deadline:
    n += 1
    any_non404 = False
    for label, idv in ids.items():
        if not idv: continue
        for tpl in templates:
            url = f"{BASE}/{tpl.format(v=urllib.parse.quote(idv, safe=''))}"
            try:
                code, body = get(url)
                any_non404 = True
                print(f"  [{n}] ✅NON-404 ({label}) {tpl}: HTTP {code}: {body[:300]}")
                if any(x in body.lower() for x in ['b64_json','"url"','image_url','result_url','data:image','.png','.jpg','.jpeg','http']):
                    found_paths.append((url, body[:500]))
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue
                any_non404 = True
                print(f"  [{n}] ({label}) {tpl}: HTTP {e.code}: {e.read().decode('utf-8','replace')[:200]}")
            except Exception as e:
                print(f"  [{n}] ({label}) {tpl}: {type(e).__name__}")
    if found_paths:
        print("\n  🎯 候选含图URL路径:")
        for u,b in found_paths: print(f"    {u}\n      {b}")
        break
    if not any_non404:
        print(f"  [poll {n}] 全部路径 404 (task_id 和 group_id 都查不到)")
    time.sleep(8)

print("\n=== 测试结束 ===")
