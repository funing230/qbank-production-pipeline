"""
人工审核样本生成：2科各1个KP，每KP生成1道题 + 渲染图片 + 打包ZIP
"""
import json, os, sys, time, zipfile, shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.orchestrator import Config
from pipeline.generator import QuestionGenerator
from pipeline.render_router import RenderEngineRouter
from pipeline.render_executor import render_question_image, generate_placeholder_image

import httpx


PROJECT = Path(__file__).parent
OUTPUT_DIR = PROJECT / "human_review_sample"
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)

config = Config()
generator = QuestionGenerator(
    base_url=config.gpt_base_url,
    api_key=config.gpt_api_key,
    model=config.gpt_model,
)
router = RenderEngineRouter()
router.smoke_test()

# 两个选中的知识点
KPS = [
    {
        "_subject_id": "S05",
        "subject_name": "几何",
        "module_name": "几何补充二",
        "kp_id": "S05-M11-014",
        "kp_name": "向量积的几何应用",
        "knowledge_point_id": "S05-M11-014",
        "knowledge_point_name": "向量积的几何应用",
        "scope_boundary": "叉积求面积/体积、混合积的几何意义、向量积证明方法",
        "allowed_image_types": ["vector_diagram", "geometric_figure"],
        "question_archetypes": ["平行四边形面积", "四面体体积", "共面判定"],
        "competency_types": ["calculation", "spatial_reasoning", "proof"],
        "quota": 12,
    },
    {
        "_subject_id": "S21",
        "subject_name": "生物学",
        "module_name": "分子生物学",
        "kp_id": "S21-M02-008",
        "kp_name": "基因组学与测序",
        "knowledge_point_id": "S21-M02-008",
        "knowledge_point_name": "基因组学与测序",
        "scope_boundary": "NGS/基因组组装/注释/比较基因组",
        "allowed_image_types": ["scientific_diagram", "annotated_figure", "data_visualization"],
        "question_archetypes": ["实验分析", "机制推断", "数据解读"],
        "competency_types": ["experiment_analysis", "mechanism_inference", "data_interpretation"],
        "quota": 10,
    },
]


def call_gpt(messages):
    """同步调用GPT"""
    url = f"{config.gpt_base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.gpt_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.gpt_model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 4096,
    }
    with httpx.Client(timeout=120) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


# 主流程
all_questions = {}  # {sid: [questions]}
now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+08:00")

for kp in KPS:
    sid = kp["_subject_id"]
    subject_name = kp["subject_name"]
    kp_id = kp["kp_id"]
    kp_name = kp["kp_name"]
    
    print(f"\n{'='*50}")
    print(f"生成: {sid} {subject_name} / {kp_name}")
    print(f"{'='*50}")
    
    # 构建prompt
    messages = generator.build_generation_prompt(kp, 1)
    
    # 调用GPT
    print("  调用GPT...")
    response = call_gpt(messages)
    print(f"  GPT响应OK")
    
    # 解析
    questions = generator.parse_response(response, kp_id, sid)
    if not questions:
        print("  ❌ 解析失败")
        continue
    
    q = questions[0]
    q_id = f"{kp_id}_001"
    q["question_id"] = q_id
    
    print(f"  题目: {q.get('question_text', '')[:60]}...")
    print(f"  引擎: {q.get('render_engine', 'N/A')}")
    
    # 渲染路由
    decision = router.route(q, kp)
    final_engine = decision["final_engine"]
    print(f"  路由决策: {decision['gpt_engine']} → {final_engine} ({decision['action']})")
    
    # 创建科目目录
    subject_dir = OUTPUT_DIR / f"{sid}_{subject_name}"
    images_dir = subject_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 渲染图片
    img_filename = f"{q_id}.png"
    img_path = images_dir / img_filename
    
    render_code = q.get("render_code", "")
    if render_code:
        print(f"  渲染图片...")
        success, err = render_question_image(render_code, str(img_path), final_engine)
        if success:
            print(f"  ✅ 图片生成: {img_path.name}")
        else:
            print(f"  ⚠️ 渲染失败: {err[:100]}")
            generate_placeholder_image(str(img_path))
    else:
        print("  ⚠️ 无render_code，生成占位图")
        generate_placeholder_image(str(img_path))
    
    # 构建最终交付格式（14字段）
    delivery_item = {
        "question_id": q_id,
        "subject_id": sid,
        "subject_name": subject_name,
        "knowledge_point_id": kp_id,
        "knowledge_point_name": kp_name,
        "question_text": q.get("question_text", ""),
        "options": q.get("options", {}),
        "correct_answer": q.get("correct_answer", ""),
        "explanation": q.get("explanation", ""),
        "difficulty": q.get("difficulty", 3),
        "image_path": f"images/{img_filename}",
        "generated_by": config.gpt_model,
        "reviewed_by": "待Qwen审核",
        "generated_at": now_str,
    }
    
    if sid not in all_questions:
        all_questions[sid] = []
    all_questions[sid].append(delivery_item)
    
    print(f"  ✅ 题目生成完成")

# 写入JSON文件
for sid, items in all_questions.items():
    subject_name = items[0]["subject_name"]
    subject_dir = OUTPUT_DIR / f"{sid}_{subject_name}"
    json_path = subject_dir / f"{sid}_{subject_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"\n✅ {json_path.name} 写入 ({len(items)}题)")

# 生成dataset_index.json
index = {
    "version": "人工审核样本",
    "generated_at": now_str,
    "total_questions": sum(len(v) for v in all_questions.values()),
    "subjects": [
        {"subject_id": sid, "subject_name": items[0]["subject_name"], "count": len(items)}
        for sid, items in all_questions.items()
    ]
}
with open(OUTPUT_DIR / "dataset_index.json", "w", encoding="utf-8") as f:
    json.dump(index, f, ensure_ascii=False, indent=2)

# 打包ZIP
zip_path = OUTPUT_DIR.parent / "human_review_sample.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            file_path = Path(root) / file
            arcname = file_path.relative_to(OUTPUT_DIR.parent)
            zf.write(file_path, arcname)

print(f"\n{'='*50}")
print(f"✅ ZIP打包完成: {zip_path}")
print(f"   大小: {zip_path.stat().st_size / 1024:.1f} KB")

# 复制到桌面
desktop = Path("/mnt/c/Users/admin/Desktop")
if desktop.exists():
    dest = desktop / "human_review_sample.zip"
    shutil.copy2(zip_path, dest)
    print(f"✅ 已发送到桌面: {dest}")
else:
    print(f"⚠️ 桌面路径不存在: {desktop}")
    # 尝试其他路径
    for user in Path("/mnt/c/Users").iterdir():
        d = user / "Desktop"
        if d.exists() and user.name not in ("Public", "Default", "Default User", "All Users"):
            dest = d / "human_review_sample.zip"
            shutil.copy2(zip_path, dest)
            print(f"✅ 已发送到桌面: {dest}")
            break

print("\n完成！请检查桌面上的 human_review_sample.zip")
