import os, json, urllib.request, urllib.error
env={}
for l in open("config/.env"):
    l=l.strip()
    if "=" in l and not l.startswith("#"):
        k,v=l.split("=",1); env[k]=v.strip().strip('"')

base=env["GPT_IMAGE_BASE_URL"].rstrip("/"); key=env["GPT_IMAGE_API_KEY"]
def get(path):
    req=urllib.request.Request(base+path, headers={"Authorization":f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req,timeout=30) as r: return r.status, r.read()[:2000].decode("utf-8","replace")
    except urllib.error.HTTPError as e: return e.code, e.read()[:500].decode("utf-8","replace")
    except Exception as e: return "ERR", str(e)[:300]

print("=== /models (确认 key 活 & 出图模型是否列出) ===")
st,body=get("/models")
print("HTTP",st)
try:
    j=json.loads(body); ids=[m.get("id") for m in j.get("data",[])]
    imgs=[i for i in ids if "image" in str(i).lower()]
    print("总模型数:",len(ids),"| 含image的:",imgs)
except Exception:
    print(body[:800])

print("\n=== 复查历史卡死 task 现态 ===")
for tid in ["53231248","53231369","53219692"]:
    hit=False
    for p in ["/media/task/"+tid, "/media/query?task_id="+tid, "/task/"+tid]:
        st,body=get(p)
        if st!=404:
            print("task",tid,"via",p,": HTTP",st,"->",body[:300]); hit=True; break
    if not hit:
        print("task",tid,": 所有查询路径均 404")
