"""随机6科各1KP，生成题目+渲染图片，打包ZIP发桌面"""
import json, os, sys, time, zipfile, shutil, random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from pipeline.orchestrator import Config
from pipeline.generator import QuestionGenerator
from pipeline.render_router import RenderEngineRouter
from pipeline.render_executor import render_question_image
import httpx

PROJECT = Path(__file__).parent.resolve()
OUTPUT_DIR = PROJECT / "human_review_sample_v2"
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)

config = Config()
generator = QuestionGenerator(base_url=config.gpt_base_url, api_key=config.gpt_api_key, model=config.gpt_model)
router = RenderEngineRouter()
router.smoke_test()

# 随机选6科
tax_dir = Path("/home/flyer8258/research_projects/multimodal_question_bank_24x1000/taxonomy_versions/v2_2_master/subjects")
all_subjects = {}
for f in sorted(tax_dir.glob("*.json")):
    data = json.loads(f.read_text())
    sid = data.get("subject_id", f.stem)
    name = data.get("subject_name", sid)
    kps = []
    if "modules" in data:
        for mod in data["modules"]:
            for kp in mod.get("knowledge_points", []):
                kp["module_name"] = mod.get("module_name", "")
                kps.append(kp)
    elif "knowledge_points" in data:
        kps = data["knowledge_points"]
    if kps:
        all_subjects[sid] = {"name": name, "kps": kps}

# 只从V7生产范围的18科中抽样（排除医学S07-S12）
PRODUCTION_SUBJECTS = {"S01","S02","S03","S04","S05","S06",
                       "S13","S14","S15","S16","S17","S18",
                       "S19","S20","S21","S22","S23","S24"}
valid_sids = [s for s in all_subjects.keys() if s in PRODUCTION_SUBJECTS]
chosen_sids = random.sample(valid_sids, 6)
KPS = []
for sid in chosen_sids:
    subj = all_subjects[sid]
    kp = random.choice(subj["kps"])
    kp["_subject_id"] = sid
    kp["subject_name"] = subj["name"]
    kp["kp_id"] = kp.get("knowledge_point_id", "unknown")
    kp["kp_name"] = kp.get("knowledge_point_name", "unknown")
    KPS.append(kp)
    print(f"  选: {sid} {subj['name']} / {kp['kp_id']} {kp['kp_name']}")

def call_gpt(messages):
    url = f"{config.gpt_base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {config.gpt_api_key}", "Content-Type": "application/json"}
    payload = {"model": config.gpt_model, "messages": messages, "temperature": 0.7, "max_tokens": 4096}
    with httpx.Client(timeout=120) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()

all_questions = {}
now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+08:00")

for kp in KPS:
    sid = kp["_subject_id"]
    sname = kp["subject_name"]
    kp_id = kp["kp_id"]
    kp_name = kp["kp_name"]
    print(f"\n{'='*50}\n生成: {sid} {sname} / {kp_name}\n{'='*50}")

    subject_dir = OUTPUT_DIR / f"{sid}_{sname}"
    images_dir = subject_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    messages = generator.build_generation_prompt(kp, 1)
    print("  调用GPT...")
    try:
        response = call_gpt(messages)
    except Exception as e:
        print(f"  ❌ GPT失败: {e}")
        continue

    questions = generator.parse_response(response, kp_id, sid)
    if not questions:
        print("  ❌ 解析失败")
        continue

    q = questions[0]
    q_id = f"{kp_id}_001"
    q["question_id"] = q_id
    print(f"  题目: {q.get('question_text','')[:55]}...")

    decision = router.route(q, kp)
    final_engine = decision["final_engine"]
    print(f"  路由: {decision['gpt_engine']}→{final_engine} ({decision['action']})")

    img_filename = f"{q_id}.png"
    img_path_abs = str((images_dir / img_filename).resolve())

    render_code = q.get("render_code", "")
    render_ok = False
    if render_code:
        print("  渲染...")
        success, err = render_question_image(render_code, img_path_abs, final_engine)
        if success and Path(img_path_abs).exists() and Path(img_path_abs).stat().st_size > 500:
            print(f"  ✅ 图片OK ({Path(img_path_abs).stat().st_size//1024}KB)")
            render_ok = True
        else:
            print(f"  ⚠️ 渲染失败: {(err or 'unknown')[:80]}")

    if not render_ok:
        os.makedirs(os.path.dirname(img_path_abs), exist_ok=True)
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"RENDER FAILED\n{kp_name}", ha='center', va='center',
                fontsize=16, color='red', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', edgecolor='red'))
        ax.axis('off')
        plt.savefig(img_path_abs, dpi=100, bbox_inches='tight')
        plt.close('all')
        print("  📌 占位图")

    delivery = {
        "question_id": q_id, "subject_id": sid, "subject_name": sname,
        "knowledge_point_id": kp_id, "knowledge_point_name": kp_name,
        "question_text": q.get("question_text",""), "options": q.get("options",{}),
        "correct_answer": q.get("correct_answer",""), "explanation": q.get("explanation",""),
        "difficulty": q.get("difficulty",3), "image_path": f"images/{img_filename}",
        "render_success": render_ok, "generated_by": config.gpt_model, "generated_at": now_str,
    }
    all_questions.setdefault(sid, []).append(delivery)
    print(f"  ✅ 完成")

# 写JSON
for sid, items in all_questions.items():
    sdir = OUTPUT_DIR / f"{sid}_{items[0]['subject_name']}"
    (sdir / f"{sid}_{items[0]['subject_name']}.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

# index
render_ok_count = sum(1 for v in all_questions.values() for i in v if i["render_success"])
idx = {"version":"v2(6科随机)", "generated_at":now_str,
       "total": len(all_questions), "render_ok": render_ok_count,
       "subjects": [{"sid":s,"name":v[0]["subject_name"],"kp":v[0]["knowledge_point_name"],
                     "render":v[0]["render_success"]} for s,v in all_questions.items()]}
(OUTPUT_DIR/"dataset_index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=2))

# ZIP
zip_path = PROJECT / "human_review_sample_v2.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fp in OUTPUT_DIR.rglob("*"):
        if fp.is_file():
            zf.write(fp, fp.relative_to(PROJECT))

shutil.copy2(zip_path, "/mnt/c/Users/admin/Desktop/human_review_sample_v2.zip")
print(f"\n{'='*50}")
print(f"✅ 完成! {len(all_questions)}/6科 | 渲染成功: {render_ok_count}/{len(all_questions)}")
print(f"✅ ZIP: {zip_path.stat().st_size//1024}KB → 桌面")
