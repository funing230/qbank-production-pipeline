#!/usr/bin/env python3
"""Build a read-only 24-subject pilot plan from taxonomy metadata.

This script prints a JSON plan only. It does not start production, call models,
or write queue/runtime state.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAXONOMY_DIR = (
    ROOT.parents[2]
    / "taxonomy_versions"
    / "v2_2_master"
    / "subjects"
)
MEDICAL_SUBJECTS = {f"S{i:02d}" for i in range(7, 13)}


def iter_subject_files(taxonomy_dir: Path):
    for path in taxonomy_dir.glob("*.json"):
        yield path


def load_subject(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    modules = data.get("modules", [])
    kp_count = sum(len(module.get("knowledge_points", [])) for module in modules)
    return {
        "subject_id": data.get("subject_id"),
        "subject_name": data.get("subject_name", ""),
        "quota": 10,
        "synthetic_medical_only": data.get("subject_id") in MEDICAL_SUBJECTS,
        "taxonomy_file": str(path),
        "module_count": len(modules),
        "knowledge_point_count": kp_count,
    }


def build_plan(taxonomy_dir: Path) -> dict:
    subjects = [load_subject(path) for path in iter_subject_files(taxonomy_dir)]
    subjects = [subject for subject in subjects if subject["subject_id"]]
    subjects.sort(key=lambda item: int(item["subject_id"][1:]))
    if len(subjects) != 24:
        raise RuntimeError(f"Expected 24 subjects, found {len(subjects)} in {taxonomy_dir}")
    return {
        "version": "pilot_240_qwen36_subjects_plan_v1",
        "taxonomy_dir": str(taxonomy_dir),
        "total_target": sum(subject["quota"] for subject in subjects),
        "quota_per_subject": 10,
        "medical_subjects": sorted(MEDICAL_SUBJECTS),
        "subjects": subjects,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy-dir", type=Path, default=DEFAULT_TAXONOMY_DIR)
    args = parser.parse_args()
    plan = build_plan(args.taxonomy_dir)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
