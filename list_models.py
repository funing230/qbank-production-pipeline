import os, json, urllib.request
env={}
for l in open("config/.env"):
    l=l.strip()
    if "=" in l and not l.startswith("#"):
        k,v=l.split("=",1); env[k]=v.strip().strip('"')
base=env["GPT_IMAGE_BASE_URL"].rstrip("/"); key=env["GPT_IMAGE_API_KEY"]
req=urllib.request.Request(base+"/models", headers={"Authorization":"Bearer "+key})
j=json.loads(urllib.request.urlopen(req,timeout=30).read())
ids=sorted(m.get("id") for m in j.get("data",[]))
print("总模型数:", len(ids))
print("\n含 image:", [i for i in ids if "image" in i.lower()])
print("含 gpt-image:", [i for i in ids if "gpt-image" in i.lower()])
print("含 dall/flux/sd/seedream/draw/pic:", [i for i in ids if any(t in i.lower() for t in ["dall","flux","sd","seedream","draw","pic","banana","gemini-3"])])
print("\n全部模型:")
for i in ids: print("  ",i)
