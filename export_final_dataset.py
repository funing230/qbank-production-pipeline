#!/usr/bin/env python3
"""
Export FINAL_PASS questions into the final submission package.

Output layout:
  final_submission_v7_image_prompt/
    dataset_index.json
    S01_组合与离散数学/
      S01_组合与离散数学.json
      images/*.png
    ...
"""
import json
import os
import shutil
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DB_PATH = PROJECT_ROOT / "production.db"
OUTPUT_DIR = Path(os.environ.get("EXPORT_OUTPUT_DIR", str(PROJECT_ROOT / "final_submission_v7_image_prompt")))
SUBJECT_NAME_FILE = PROJECT_ROOT / "config" / "subject_name_mapping.json"
EXPECTED_TOTAL_QUESTIONS = 24000

SUBJECTS = [
    "S01", "S02", "S03", "S04", "S05", "S06",
    "S13", "S14", "S15", "S16", "S17", "S18",
    "S19", "S20", "S21", "S22", "S23", "S24",
]
EXPECTED_TOTAL = int(os.environ.get("EXPECTED_TOTAL_QUESTIONS", "24000"))
STRICT_EXPORT = os.environ.get("STRICT_EXPORT", "1") != "0"
REQUIRED_FIELDS = [
    "question_id",
    "subject_id",
    "subject_name",
    "kp_id",
    "kp_name",
    "question_language",
    "question_text",
    "options",
    "correct_answer",
    "explanation",
    "difficulty",
    "blueprint_slot",
    "blueprint_archetype",
    "blueprint_image_type",
    "image_path",
    "image_prompt",
    "final_image_prompt",
    "image_dependency_reason",
    "truth_spec",
]


def load_subject_names() -> dict:
    if SUBJECT_NAME_FILE.exists():
        data = json.loads(SUBJECT_NAME_FILE.read_text(encoding="utf-8"))
        out = {}
        for k, v in data.items():
            if isinstance(v, str):
                out[k] = v
            elif isinstance(v, dict):
                # 嵌套结构 {"zh": "...", "en": "..."}: 取中文名作权威
                name = v.get("zh") or v.get("name") or v.get("en")
                if isinstance(name, str):
                    out[k] = name
        return out
    return {}


def resolve_image_path(raw_path: str, subject_id: str) -> Path | None:
    if not raw_path:
        return None
    candidates = []
    raw_image_path = Path(raw_path)
    if raw_image_path.is_absolute():
        candidates.append(raw_image_path)
    else:
        candidates.extend([
            PROJECT_ROOT / raw_image_path,
            PROJECT_ROOT / "output" / raw_image_path,
            PROJECT_ROOT / "output" / subject_id / "images" / raw_image_path.name,
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def normalize_item(raw: dict, row: sqlite3.Row, subject_names: dict, subject_dir: Path) -> dict:
    subject_id = raw.get("subject_id") or row["subject_id"]
    # 以权威 mapping 为准, 避免 raw.subject_name 时缺时有导致同科目分裂成两个文件夹
    subject_name = subject_names.get(subject_id) or raw.get("subject_name") or subject_id
    raw_image_path = raw.get("image_path") or row["image_path"] or ""
    image_src = resolve_image_path(raw_image_path, subject_id)
    if image_src:
        images_dir = subject_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        image_dst = images_dir / image_src.name
        if image_src.resolve() != image_dst.resolve():
            shutil.copy2(image_src, image_dst)
        image_rel = f"images/{image_dst.name}"
    else:
        image_rel = raw_image_path

    item = {
        "question_id": raw.get("question_id") or row["question_id"],
        "subject_id": subject_id,
        "subject_name": subject_name,
        "kp_id": raw.get("kp_id") or row["kp_id"],
        "kp_name": raw.get("kp_name") or row["kp_name"],
        "question_language": raw.get("question_language", "zh"),
        "question_text": raw.get("question_text", ""),
        "options": raw.get("options", {}),
        "correct_answer": raw.get("correct_answer", ""),
        "explanation": raw.get("explanation", ""),
        "difficulty": raw.get("difficulty", 0),
        "blueprint_slot": raw.get("blueprint_slot"),
        "blueprint_archetype": raw.get("blueprint_archetype", ""),
        "blueprint_image_type": raw.get("blueprint_image_type", ""),
        "image_path": image_rel,
        "image_prompt": raw.get("image_prompt", ""),
        "final_image_prompt": raw.get("final_image_prompt", ""),
        "image_dependency_reason": raw.get("image_dependency_reason", ""),
        "truth_spec": raw.get("truth_spec", {}),
    }
    missing = [field for field in REQUIRED_FIELDS if item.get(field) in (None, "", [], {}) and field not in {"blueprint_slot"}]
    if missing:
        item["_export_warnings"] = [f"missing_or_empty: {field}" for field in missing]
    return item


def main():
    if not DB_PATH.exists():
        raise SystemExit(f"DB not found: {DB_PATH}")
    subject_names = load_subject_names()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM questions WHERE quality_status='FINAL_PASS' ORDER BY subject_id, kp_id, question_id"
    ).fetchall()

    grouped = defaultdict(list)
    warnings = []
    for row in rows:
        try:
            raw = json.loads(row["question_json"] or "{}")
        except json.JSONDecodeError:
            warnings.append({"question_id": row["question_id"], "warning": "invalid question_json"})
            continue
        subject_id = raw.get("subject_id") or row["subject_id"]
        if subject_id not in SUBJECTS:
            warnings.append({"question_id": row["question_id"], "warning": f"excluded_subject:{subject_id}"})
            continue
        subject_name = subject_names.get(subject_id) or raw.get("subject_name") or subject_id
        subject_dir = OUTPUT_DIR / f"{subject_id}_{subject_name}"
        item = normalize_item(raw, row, subject_names, subject_dir)
        grouped[subject_id].append(item)
        if "_export_warnings" in item:
            warnings.append({"question_id": item["question_id"], "warnings": item["_export_warnings"]})

    subjects_index = []
    validation_errors = []
    for subject_id in SUBJECTS:
        items = grouped.get(subject_id, [])
        subject_name = subject_names.get(subject_id) or (items[0]["subject_name"] if items else subject_id)
        subject_dir = OUTPUT_DIR / f"{subject_id}_{subject_name}"
        images_dir = subject_dir / "images"
        subject_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        json_path = subject_dir / f"{subject_id}_{subject_name}.json"
        json_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

        image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        referenced_images = [item.get("image_path", "") for item in items]
        missing_image_refs = [ref for ref in referenced_images if not ref or not (subject_dir / ref).exists()]
        extra_images = [p.name for p in image_files if f"images/{p.name}" not in referenced_images]
        if len(items) != len(image_files):
            validation_errors.append({
                "subject_id": subject_id,
                "error": "question_image_count_mismatch",
                "questions": len(items),
                "images": len(image_files),
            })
        if missing_image_refs:
            validation_errors.append({
                "subject_id": subject_id,
                "error": "missing_referenced_images",
                "count": len(missing_image_refs),
                "sample": missing_image_refs[:5],
            })
        if extra_images:
            validation_errors.append({
                "subject_id": subject_id,
                "error": "extra_unreferenced_images",
                "count": len(extra_images),
                "sample": extra_images[:5],
            })
        subjects_index.append({
            "subject_id": subject_id,
            "subject_name": subject_name,
            "count": len(items),
            "image_count": len(image_files),
            "json_file": f"{subject_id}_{subject_name}/{json_path.name}",
            "images_dir": f"{subject_id}_{subject_name}/images",
        })

    total_questions = sum(len(v) for v in grouped.values())
    total_images = sum(s["image_count"] for s in subjects_index)
    if total_questions != EXPECTED_TOTAL:
        validation_errors.append({
            "error": "total_question_count_mismatch",
            "expected": EXPECTED_TOTAL,
            "actual": total_questions,
        })
    if total_images != EXPECTED_TOTAL:
        validation_errors.append({
            "error": "total_image_count_mismatch",
            "expected": EXPECTED_TOTAL,
            "actual": total_images,
        })

    index = {
        "version": "v7_image_prompt_final_submission",
        "generated_at": datetime.now().isoformat(),
        "source_db": str(DB_PATH),
        "expected_total_questions": EXPECTED_TOTAL,
        "total_questions": total_questions,
        "total_images": total_images,
        "subjects": subjects_index,
        "schema_fields": REQUIRED_FIELDS,
        "validation_status": "PASS" if not validation_errors and not warnings else "FAIL",
        "validation_errors": validation_errors,
        "warnings_count": len(warnings),
        "warnings_sample": warnings[:50],
    }
    (OUTPUT_DIR / "dataset_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(OUTPUT_DIR),
        "expected_total_questions": EXPECTED_TOTAL,
        "total_questions": index["total_questions"],
        "total_images": index["total_images"],
        "subjects": len(subjects_index),
        "validation_status": index["validation_status"],
        "validation_errors_count": len(validation_errors),
        "warnings_count": len(warnings),
    }, ensure_ascii=False, indent=2))

    if STRICT_EXPORT and (validation_errors or warnings):
        raise SystemExit(
            "STRICT_EXPORT failed: final package is not submission-ready. "
            f"validation_errors={len(validation_errors)}, warnings={len(warnings)}. "
            f"See {OUTPUT_DIR / 'dataset_index.json'}"
        )


if __name__ == "__main__":
    main()
