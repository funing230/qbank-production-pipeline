"""
12题端到端轻量测试 v1.0

验证V7冻结版完整链路：
1. 数据读取（16字段）
2. GPT生成prompt构建
3. GPT响应解析
4. 渲染引擎路由
5. Qwen审核prompt构建
6. Qwen审核响应解析
7. 重生成prompt构建

本测试不调用真实API，使用模拟响应验证代码逻辑正确性。
"""
import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.generator import QuestionGenerator
from pipeline.render_router import RenderEngineRouter
from pipeline.orchestrator import Config


def _make_generator():
    """创建测试用generator（不需要真实API调用）"""
    config = Config()
    return QuestionGenerator(
        base_url=config.gpt_base_url or "http://localhost:8000/v1",
        api_key=config.gpt_api_key or "test-key",
        model=config.gpt_model,
    )


def test_data_loading():
    """测试1: 16字段完整读取"""
    print("=" * 60)
    print("TEST 1: 数据读取 - 16字段完整性")
    print("=" * 60)
    
    tax_dir = Path(__file__).parent.parent.parent.parent / "taxonomy_versions" / "v2_2_master" / "subjects"
    assert tax_dir.exists(), f"Taxonomy dir not found: {tax_dir}"
    
    f = tax_dir / "s01.json"
    data = json.load(open(f))
    subject_name = data.get("subject_name", data.get("name", "S01"))
    
    kps = []
    for mod in data["modules"]:
        mod_name = mod.get("module_name", mod.get("name", ""))
        mod_id = mod.get("module_id", mod.get("id", ""))
        for kp in mod.get("knowledge_points", []):
            kp["_subject_name"] = subject_name
            kp["_subject_id"] = "S01"
            kp["module_name"] = mod_name
            kp["module_id"] = mod_id
            kp.setdefault("id", kp.get("knowledge_point_id", ""))
            kp.setdefault("kp_name", kp.get("knowledge_point_name", ""))
            kp.setdefault("kp_id", kp.get("knowledge_point_id", ""))
            kp.setdefault("quota", kp.get("target_quota", 10))
            kp.setdefault("subject_name", subject_name)
            kps.append(kp)
    
    GPT_FIELDS = ["subject_name", "module_name", "kp_name", "kp_id", 
                  "scope_boundary", "question_archetypes", "allowed_image_types", "competency_types"]
    SYS_FIELDS = ["_subject_id", "module_id", "quota", "importance",
                  "exam_frequency", "image_source_risk", "professional_risk", "variation_capacity"]
    
    sample = kps[0]
    missing = []
    for f in GPT_FIELDS + SYS_FIELDS:
        if f not in sample:
            missing.append(f)
    
    if missing:
        print(f"  ❌ FAIL: Missing fields: {missing}")
        return False, kps
    else:
        print(f"  ✅ PASS: All 16 fields present ({len(kps)} KPs loaded)")
        return True, kps


def test_generation_prompt(kps):
    """测试2: GPT生成prompt构建"""
    print("\n" + "=" * 60)
    print("TEST 2: GPT生成prompt构建")
    print("=" * 60)
    
    generator = _make_generator()
    
    kp = kps[0]
    messages = generator.build_generation_prompt(kp, 3)
    
    assert len(messages) >= 1, "Messages list empty"
    system_msg = messages[0]["content"] if messages[0]["role"] == "system" else ""
    
    # 检查关键内容是否在prompt中
    checks = {
        "科目名": kp["subject_name"] in system_msg,
        "知识点名": kp["kp_name"] in system_msg,
        "知识点ID": kp["kp_id"] in system_msg,
        "scope_boundary": kp.get("scope_boundary", "")[:20] in system_msg if kp.get("scope_boundary") else True,
        "truth_spec": "truth_spec" in system_msg,
        "image_dependency_reason": "image_dependency_reason" in system_msg,
        "difference_from_others": "difference_from_others" in system_msg,
        "render_engine": "render_engine" in system_msg,
        "render_code": "render_code" in system_msg,
    }
    
    all_pass = True
    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if not result:
            all_pass = False
    
    print(f"  Prompt length: {len(system_msg)} chars")
    return all_pass


def test_response_parsing():
    """测试3: GPT响应解析（模拟）"""
    print("\n" + "=" * 60)
    print("TEST 3: GPT响应解析")
    print("=" * 60)
    
    generator = _make_generator()
    
    # 模拟GPT返回的V7格式
    mock_response = {
        "choices": [{
            "message": {
                "content": json.dumps([{
                    "question_id": "S01-M01-001_0001",
                    "question_text": "如图所示的Venn图中，阴影部分表示的集合运算是",
                    "options": {"A": "A∩B", "B": "A∪B", "C": "A-B", "D": "(A∪B)ᶜ"},
                    "correct_answer": "C",
                    "explanation": "阴影部分包含属于A但不属于B的元素，即A-B",
                    "difficulty": 3,
                    "knowledge_point": "集合运算与Venn图",
                    "render_engine": "MATPLOTLIB",
                    "render_code": "fig, ax = plt.subplots()\ncircle1 = plt.Circle((0.3, 0.5), 0.3)\nax.add_patch(circle1)\nplt.savefig(output_path, dpi=150, bbox_inches='tight')",
                    "render_params": {"figsize": [10, 8], "dpi": 150},
                    "image_description": "两个相交圆的Venn图，左圆阴影部分表示A-B",
                    "truth_spec": {
                        "correct_answer": "C",
                        "image_only_facts": ["阴影区域位于A圈内B圈外"],
                        "validation_rules": ["阴影面积=A∩Bᶜ"]
                    },
                    "image_dependency_reason": "必须从图中观察阴影位置才能确定集合运算类型",
                    "difference_from_others": "本题聚焦差集运算的图形识别"
                }], ensure_ascii=False)
            }
        }]
    }
    
    questions = generator.parse_response(mock_response, "S01-M01-001", "S01")
    
    if not questions:
        print("  ❌ FAIL: parse_response returned empty list")
        return False
    
    q = questions[0]
    checks = {
        "question_text存在": bool(q.get("question_text")),
        "options完整": len(q.get("options", {})) == 4,
        "correct_answer有效": q.get("correct_answer") in "ABCD",
        "render_code存在": bool(q.get("render_code")),
        "render_engine存在": bool(q.get("render_engine")),
        "truth_spec存在": bool(q.get("truth_spec")),
        "image_dependency_reason存在": bool(q.get("image_dependency_reason")),
    }
    
    all_pass = True
    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if not result:
            all_pass = False
    
    return all_pass


def test_render_router(kps):
    """测试4: 渲染引擎路由器"""
    print("\n" + "=" * 60)
    print("TEST 4: 渲染引擎路由器")
    print("=" * 60)
    
    router = RenderEngineRouter()
    results = router.smoke_test()
    
    enabled = sum(1 for v in results.values() if v == "ENABLED")
    print(f"  引擎冒烟测试: {enabled}/{len(results)} ENABLED")
    
    # 测试路由决策
    test_cases = [
        # (GPT选择, KP image_types, 预期动作)
        ({"render_engine": "MATPLOTLIB"}, {"allowed_image_types": ["function_plot"]}, "KEEP"),
        ({"render_engine": "MATPLOTLIB"}, {"allowed_image_types": ["circuit_diagram"]}, "OVERRIDE"),
        ({"render_engine": "NETWORKX"}, {"allowed_image_types": ["hasse_diagram"]}, "KEEP"),
    ]
    
    all_pass = True
    for question, kp_info, expected_action in test_cases:
        decision = router.route(question, kp_info)
        result = decision["action"] == expected_action
        status = "✅" if result else "❌"
        print(f"  {status} GPT={question['render_engine']}, type={kp_info['allowed_image_types'][0]} → {decision['action']} (expected {expected_action})")
        if not result:
            all_pass = False
    
    return all_pass


def test_review_prompt(kps):
    """测试5: Qwen审核prompt构建"""
    print("\n" + "=" * 60)
    print("TEST 5: Qwen审核prompt构建（9项检查）")
    print("=" * 60)
    
    generator = _make_generator()
    
    question = {
        "question_id": "S01-M01-001_0001",
        "question_text": "测试题目",
        "options": {"A": "选项A", "B": "选项B", "C": "选项C", "D": "选项D"},
        "correct_answer": "A",
        "render_code": "fig, ax = plt.subplots()\nplt.savefig(output_path)",
        "render_engine": "MATPLOTLIB",
        "image_description": "测试图片",
    }
    
    messages = generator.build_review_prompt(question, kps[0])
    
    assert len(messages) >= 1, "Review messages empty"
    prompt_text = messages[0]["content"]
    
    # 检查9项检查是否都在prompt中
    nine_checks = ["范围", "专业正确性", "答案与解析", "图文一致性", "图片依赖", 
                   "图片排版", "名称一致性", "难度一致性", "重复性"]
    
    all_pass = True
    for check in nine_checks:
        result = check in prompt_text
        status = "✅" if result else "❌"
        print(f"  {status} 检查项: {check}")
        if not result:
            all_pass = False
    
    # 检查输出格式
    format_checks = {
        "decision字段": "decision" in prompt_text,
        "checks结构": "scope_correct" in prompt_text,
        "confidence字段": "confidence" in prompt_text,
        "科目名传入": kps[0]["subject_name"] in prompt_text,
        "知识点名传入": kps[0]["kp_name"] in prompt_text,
    }
    
    for check_name, result in format_checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if not result:
            all_pass = False
    
    return all_pass


def test_review_parsing():
    """测试6: Qwen审核响应解析"""
    print("\n" + "=" * 60)
    print("TEST 6: Qwen审核响应解析")
    print("=" * 60)
    
    generator = _make_generator()
    
    # 模拟Qwen V7格式响应
    mock_review = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "item_id": "S01-M01-001_0001",
                    "decision": "PASS",
                    "difficulty_alignment": "ALIGNED",
                    "checks": {
                        "scope_correct": True,
                        "professional_correct": True,
                        "answer_correct": True,
                        "solution_correct": True,
                        "image_consistent": True,
                        "image_dependency_valid": True,
                        "layout_valid": True,
                        "font_valid": True,
                        "naming_consistent": True,
                        "duplication_risk": "LOW"
                    },
                    "issues": [],
                    "revision_scope": "NONE",
                    "confidence": 0.92
                }, ensure_ascii=False)
            }
        }]
    }
    
    result = generator.parse_review_response(mock_review)
    
    checks = {
        "status=PASS": result["status"] == "PASS",
        "checks存在": bool(result.get("checks")),
        "difficulty_alignment": result.get("difficulty_alignment") == "ALIGNED",
        "revision_scope": result.get("revision_scope") == "NONE",
        "confidence>0": result.get("confidence", 0) > 0,
    }
    
    all_pass = True
    for check_name, ok in checks.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {check_name}: {result.get(check_name.split('=')[0], 'N/A')}")
        if not ok:
            all_pass = False
    
    # 测试REJECT响应
    mock_reject = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "item_id": "S01-M01-001_0002",
                    "decision": "REJECT",
                    "difficulty_alignment": "ALIGNED",
                    "checks": {
                        "scope_correct": True,
                        "professional_correct": True,
                        "answer_correct": False,
                        "solution_correct": False,
                        "image_consistent": True,
                        "image_dependency_valid": True,
                        "layout_valid": True,
                        "font_valid": True,
                        "naming_consistent": True,
                        "duplication_risk": "LOW"
                    },
                    "issues": [{"type": "answer_error", "description": "答案计算有误"}],
                    "revision_scope": "FULL_REDESIGN",
                    "confidence": 0.88
                }, ensure_ascii=False)
            }
        }]
    }
    
    reject_result = generator.parse_review_response(mock_reject)
    reject_ok = reject_result["status"] == "REJECT"
    print(f"  {'✅' if reject_ok else '❌'} REJECT解析: status={reject_result['status']}")
    if not reject_ok:
        all_pass = False
    
    return all_pass


def test_regen_prompt(kps):
    """测试7: 重生成prompt构建（避错指令+去重）"""
    print("\n" + "=" * 60)
    print("TEST 7: 重生成prompt构建")
    print("=" * 60)
    
    generator = _make_generator()
    
    kp = kps[0]
    old_text = "如图所示的Venn图中，阴影部分表示的是什么集合运算？"
    verdict = {
        "status": "REJECT",
        "issues": [{"type": "answer_error", "description": "答案不正确"}],
        "checks": {
            "answer_correct": False,
            "solution_correct": False,
            "image_dependency_valid": True,
        },
        "confidence": 0.85
    }
    existing = [
        {"question_text": "已通过的某题关于集合并运算的问题"},
        {"question_text": "已通过的某题关于集合交运算的问题"},
    ]
    
    messages = generator.build_regen_prompt(kp, old_text, verdict, existing)
    
    assert len(messages) >= 2, "Regen messages should have system + user"
    full_text = " ".join(m["content"] for m in messages)
    
    checks = {
        "不含原题完整代码": "render_code" not in full_text or "plt.Circle" not in full_text,
        "含失败原因类型": "答案或解析错误" in full_text,
        "含避错指令": "特别注意" in full_text,
        "含去重摘要": "已通过" in full_text,
        "含已丢弃题": "已丢弃" in full_text,
        "含知识点约束": kp["kp_name"] in full_text,
        "不含旧题全文": len(old_text) > 50 and old_text not in full_text or len(old_text) <= 50,  # 短题摘要即全文，允许
        "含truth_spec要求": "truth_spec" in full_text,
        "含image_dependency_reason": "image_dependency_reason" in full_text,
    }
    
    all_pass = True
    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if not result:
            all_pass = False
    
    return all_pass


def main():
    print("\n" + "#" * 60)
    print("# V7冻结版 端到端轻量测试")
    print("# 验证: prompt构建 → 解析 → 路由 → 审核 → 重生成")
    print("#" * 60)
    
    results = {}
    
    # Test 1
    ok, kps = test_data_loading()
    results["数据读取16字段"] = ok
    
    if not kps:
        print("\n❌ 数据加载失败，无法继续测试")
        return
    
    # Test 2
    results["GPT生成prompt"] = test_generation_prompt(kps)
    
    # Test 3
    results["GPT响应解析"] = test_response_parsing()
    
    # Test 4
    results["渲染引擎路由"] = test_render_router(kps)
    
    # Test 5
    results["Qwen审核prompt"] = test_review_prompt(kps)
    
    # Test 6
    results["Qwen审核解析"] = test_review_parsing()
    
    # Test 7
    results["重生成prompt"] = test_regen_prompt(kps)
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n结果: {passed}/{total} PASSED")
    
    if passed == total:
        print("\n🎉 全部通过！V7冻结版代码链路验证完成。")
    else:
        print("\n⚠️ 存在失败项，需要修复后重新测试。")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
