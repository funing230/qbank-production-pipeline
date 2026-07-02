"""
Qwen 审核模块（新版：纯文本逻辑一致性审核）
================================================================
FROZEN (2026-06-21) — 未经爸爸明确批准不得修改审核逻辑。

职责：验证 GPT 输出的题目文本与 image_prompt 之间的逻辑一致性。
不涉及图片渲染、像素检测、多模态。纯文本 → 纯文本。

审核 6 项：
  1. 答案正确性
  2. 选项合理性
  3. image_prompt 与题目一致性
  4. 图片依赖性
  5. 知识点范围
  6. image_prompt 规范合规
"""
import json
import os
import re
import time
import urllib.request
import urllib.error
from pathlib import Path


# 加载 image_prompt 配置（用于合规性校验）
_CONFIG_DIR = Path(__file__).parent.parent / "config"
_STYLE_CONFIG_PATH = _CONFIG_DIR / "image_prompt_style.json"

def _load_style_config() -> dict:
    if _STYLE_CONFIG_PATH.exists():
        return json.loads(_STYLE_CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def _word_count(text: str) -> int:
    """统计英文词数（中文字符按1词计）"""
    # 去掉中文后统计英文词
    english_parts = re.sub(r'[\u4e00-\u9fff]+', ' ZH ', text)
    return len(english_parts.split())


def _iter_text_fields(question: dict):
    yield "question_text", str(question.get("question_text", ""))
    options = question.get("options", {})
    if isinstance(options, dict):
        for key in ["A", "B", "C", "D"]:
            yield f"options.{key}", str(options.get(key, ""))
    yield "explanation", str(question.get("explanation", ""))


def _has_illegal_control_chars(text: str) -> bool:
    return any((ord(ch) < 32 and ch not in "\n\r\t") for ch in text)


def _math_spans(text: str):
    spans = []
    for pattern in (r"\$\$.*?\$\$", r"\$[^$\n]+\$", r"\\\(.*?\\\)", r"\\\[.*?\\\]"):
        spans.extend(match.span() for match in re.finditer(pattern, text, flags=re.DOTALL))
    return sorted(spans)


def _inside_any_span(index: int, spans: list) -> bool:
    return any(start <= index < end for start, end in spans)


def _strip_math_spans(text: str) -> str:
    stripped = text
    for start, end in reversed(_math_spans(text)):
        stripped = stripped[:start] + " " * (end - start) + stripped[end:]
    return stripped


def _check_balanced_math_delimiters(text: str) -> list:
    issues = []
    if text.count("$") % 2 != 0:
        issues.append("美元符号 $ 数量不成对")
    if text.count("\\(") != text.count("\\)"):
        issues.append("LaTeX 行内公式分隔符 \\( / \\) 不成对")
    if text.count("\\[") != text.count("\\]"):
        issues.append("LaTeX 展示公式分隔符 \\[ / \\] 不成对")
    return issues


def _consume_braced_group(text: str, pos: int) -> int:
    """从 pos 处期望一个 {...} 组(允许前导空白与可选 [..] 选项)。
    支持任意深度嵌套花括号。返回组结束后的索引；解析失败返回 -1。"""
    n = len(text)
    i = pos
    while i < n and text[i] in " \t":
        i += 1
    if i >= n or text[i] != "{":
        return -1
    depth = 0
    while i < n:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1  # 未闭合


def _has_complete_args(text: str, start: int, n_args: int, optional_bracket: bool = False) -> bool:
    """校验 text[start:] 处的命令后面是否跟着 n_args 个完整 {..} 参数。
    支持嵌套花括号(如 \\frac{1}{2\\sqrt{y}}、\\frac{\\frac{1}{2}}{3})。
    optional_bracket=True 允许首个参数前出现可选 [..](如 \\sqrt[3]{8})。"""
    i = start
    n = len(text)
    if optional_bracket:
        j = i
        while j < n and text[j] in " \t":
            j += 1
        if j < n and text[j] == "[":
            close = text.find("]", j)
            if close == -1:
                return False
            i = close + 1
    for _ in range(n_args):
        i = _consume_braced_group(text, i)
        if i == -1:
            return False
    return True


def _check_command_braces(text: str) -> list:
    issues = []
    # (命令, 必需 {} 参数个数, 是否允许可选 [..])
    command_specs = [
        ("\\frac", 2, False),
        ("\\sqrt", 1, True),
        ("\\binom", 2, False),
    ]
    for command, n_args, opt in command_specs:
        for match in re.finditer(re.escape(command), text):
            after = match.start() + len(command)
            if not _has_complete_args(text, after, n_args, optional_bracket=opt):
                issues.append(f"LaTeX 命令 {command} 参数不完整")
                break
    begins = re.findall(r"\\begin\{([^{}]+)\}", text)
    ends = re.findall(r"\\end\{([^{}]+)\}", text)
    if begins != ends:
        issues.append("LaTeX begin/end 环境不匹配")
    return issues


def _find_raw_latex_source_marker(text: str) -> str:
    """Return the first raw LaTeX source marker in visible image text, if any.

    Unicode-rendered math/science symbols are allowed: α, β, Δ, ≤, ±, ², ₀, etc.
    Raw source forms are not allowed because image models may draw them literally:
    $, \frac, \\frac, \alpha, \\alpha, ...
    """
    if "$" in text:
        return "$"
    for command in ("frac", "alpha", "beta", "gamma", "sum", "int", "sqrt", "begin", "end"):
        for marker in (f"\\{command}", f"\\\\{command}"):
            if marker in text:
                return marker
    return ""


def validate_latex_format(question: dict) -> list:
    """本地 LaTeX/公式格式检查；只做确定性格式门，不判断数学语义。"""
    issues = []
    typo_markers = ["\\rac", "\\farc", "\\frc"]
    math_commands = ["\\frac", "\\sqrt", "\\sum", "\\int", "\\alpha", "\\beta", "\\gamma", "\\theta", "\\lambda", "\\binom"]

    for field, text in _iter_text_fields(question):
        if not text:
            continue
        if _has_illegal_control_chars(text):
            issues.append(f"{field} 含非法控制字符")
        for detail in _check_balanced_math_delimiters(text):
            issues.append(f"{field}: {detail}")
        for detail in _check_command_braces(text):
            issues.append(f"{field}: {detail}")
        text_lower = text.lower()
        for marker in typo_markers:
            if marker.lower() in text_lower:
                issues.append(f"{field} 疑似残缺 LaTeX 命令: {marker}")
                break
        if re.search(r"(^|[^a-zA-Z\\\\])rac\{", text):
            issues.append(f"{field} 疑似残缺 LaTeX 命令: rac{{")
        non_math_text = _strip_math_spans(text)
        for command in math_commands:
            for match in re.finditer(re.escape(command), non_math_text):
                if not _inside_any_span(match.start(), _math_spans(text)):
                    issues.append(f"{field} 中 LaTeX 命令 {command} 未放在公式分隔符内")
                    break
            if issues and issues[-1].startswith(f"{field} 中 LaTeX 命令 {command}"):
                break

    image_prompt = str(question.get("image_prompt", ""))
    if _has_illegal_control_chars(image_prompt):
        issues.append("image_prompt 含非法控制字符")
    # Unicode math/science symbols such as α, β, Δ, ≤, ±, ² are allowed in image_prompt.
    # What is forbidden here is raw LaTeX source that gpt-image-2 may render literally.
    marker = _find_raw_latex_source_marker(image_prompt)
    if marker:
        issues.append(f"image_prompt 包含图片中不应显示的 LaTeX 源码标记: {marker}")

    return issues


def pre_validate(question: dict) -> tuple:
    """
    本地预校验（不调API），检查 image_prompt 基本合规性。
    
    Returns:
        (pass: bool, issues: list[str])
    """
    issues = []
    config = _load_style_config()
    constraints = config.get("constraints", {})
    validation = config.get("validation", {})
    
    image_prompt = question.get("image_prompt", "")
    
    # 1. 存在性
    if not image_prompt or not image_prompt.strip():
        issues.append("image_prompt 字段为空")
        return False, issues
    
    # 2. 词数检查
    wc = _word_count(image_prompt)
    min_words = constraints.get("min_words", 50)
    max_words = constraints.get("max_words", 200)
    if wc < min_words:
        issues.append(f"image_prompt 词数过少: {wc} < {min_words}")
    if wc > max_words:
        issues.append(f"image_prompt 词数过多: {wc} > {max_words}")
    
    # 3. 不应包含风格后缀关键词
    reject_keywords = validation.get("reject_if_contains_suffix_keywords", [])
    prompt_lower = image_prompt.lower()
    for kw in reject_keywords:
        if kw.lower() in prompt_lower:
            issues.append(f"image_prompt 包含风格后缀关键词: '{kw}'")
            break  # 报一个就够
    
    # 4. 开头应为图类型相关
    patterns = validation.get("required_opening_patterns", [])
    if patterns:
        first_word_ok = any(re.match(p, image_prompt.strip(), re.IGNORECASE) for p in patterns)
        if not first_word_ok:
            issues.append(f"image_prompt 开头不是图类型声明: '{image_prompt[:30]}...'")
    
    # 5. 模糊词检测
    vague_words = ["several", "some", "a few", "many", "various", "multiple"]
    for vw in vague_words:
        if re.search(r'\b' + vw + r'\b', image_prompt, re.IGNORECASE):
            issues.append(f"image_prompt 包含模糊词: '{vw}'（应使用明确数字）")
            break
    
    # 6. 图片prompt禁止可见 LaTeX 源码；允许 Unicode 数学/科学符号（α、β、Δ、≤、±、² 等）
    marker = _find_raw_latex_source_marker(image_prompt)
    if marker:
        issues.append(f"image_prompt 包含图片中不应显示的LaTeX源码标记: {marker}")

    # 7. 基本题目字段完整性
    required_fields = ["question_text", "options", "correct_answer", "explanation", "question_language"]
    for f in required_fields:
        if not question.get(f):
            issues.append(f"缺少必填字段: {f}")
    
    if question.get("question_language") not in ["zh", "en"]:
        issues.append(f"question_language 不合法: {question.get('question_language')}")

    # 选项检查
    options = question.get("options", {})
    if options:
        if not all(k in options for k in ["A", "B", "C", "D"]):
            issues.append("选项不完整，必须有 A/B/C/D")
        if question.get("correct_answer") not in ["A", "B", "C", "D"]:
            issues.append(f"correct_answer 不合法: {question.get('correct_answer')}")

    # 8. 本地 LaTeX/公式格式硬门：失败则不调用 Qwen，避免把确定性格式问题交给模型审
    issues.extend(validate_latex_format(question))
    
    return len(issues) == 0, issues


def build_review_prompt(question: dict, kp_info: dict = None) -> str:
    """
    构建发给 Qwen 的审核 prompt（纯文本）。
    
    Args:
        question: GPT 生成的完整题目 dict
        kp_info: 知识点信息 dict（含 kp_name, scope_boundary 等）
    
    Returns:
        prompt 字符串
    """
    # 提取题目信息
    question_text = question.get("question_text", "")
    options = question.get("options", {})
    correct_answer = question.get("correct_answer", "")
    explanation = question.get("explanation", "")
    image_prompt = question.get("image_prompt", "")
    image_dependency_reason = question.get("image_dependency_reason", "")
    question_language = question.get("question_language", "")
    difficulty = question.get("difficulty", "")
    
    # 提取知识点信息
    kp_name = ""
    scope_boundary = ""
    subject_name = ""
    if kp_info:
        kp_name = kp_info.get("kp_name", "") or kp_info.get("knowledge_point_name", "")
        scope_boundary = kp_info.get("scope_boundary", "")
        subject_name = kp_info.get("subject_name", "") or kp_info.get("_subject_name", "")
    
    # 格式化选项
    options_str = "\n".join(f"  {k}: {v}" for k, v in sorted(options.items()))
    
    prompt = f"""你是大学考试题目质量审核员。请严格检查以下题目的文本逻辑一致性。

重要边界：你不审核最终图片的视觉质量，因为你看不到 gpt-image-2 实际生成的图片。空白、全白、裁切、清晰度、主体大小、边缘触碰等真实图片质量问题，由后续 Python image_quality_gate 判断。你只审核题目文本、知识点范围、答案解析、以及 image_prompt 是否与题目一致且足够明确可画。

## 题目信息

- 科目：{subject_name}
- 知识点：{kp_name}
- 知识点范围：{scope_boundary}
- 难度：{difficulty}
- 题目语言：{question_language}（zh=中文题，en=英文题）

题干：{question_text}

选项：
{options_str}

正确答案：{correct_answer}

解析：{explanation}

图片描述提示词(image_prompt)：
{image_prompt}

图片依赖理由：{image_dependency_reason}

## 审核要求（8项检查）

1. **答案正确性**：explanation 的推导是否正确？correct_answer 是否能从解析唯一推出？
2. **选项合理性**：四选项是否互不重复？干扰项是否合理？是否存在多个正确答案？
3. **image_prompt 与题目一致性**：image_prompt 描述的元素（数量、名称、连接关系、标注）是否与题目答案匹配？答案依赖的关键信息是否体现在 image_prompt 中？
4. **图片依赖性**：如果删掉图片只看题干和选项，能否直接选出答案？题目是否真正需要图片才能作答？
5. **知识点范围**：题目是否属于"{kp_name}"？是否超出范围"{scope_boundary}"？
6. **image_prompt 规范**：描述是否具体明确（无"several/some"等模糊词）？元素数量是否写了具体数字？
7. **公式格式**：question_text/options/explanation 中的公式是否以 LaTeX `$...$` 保存？image_prompt 中是否避免了可见 `$...$`、`\\frac`、`\\alpha` 等LaTeX源码，并改用 rendered formula / standard mathematical notation 描述图片里的公式？
8. **语言一致性**：若 question_language=zh，题干/选项/解析应为中文；若 question_language=en，题干/选项/解析应为英文。

## 输出格式

只输出合法JSON，不要输出解释或思维过程：

{{"decision": "PASS或FAIL", "checks": {{"answer_correct": true, "options_reasonable": true, "prompt_consistent": true, "image_dependency_valid": true, "scope_correct": true, "prompt_compliant": true, "formula_format_valid": true, "language_valid": true}}, "issues": ["如有问题写在这里"], "confidence": 0.0}}"""

    return prompt


def build_revise_prompt(question: dict, issues: list, kp_info: dict = None) -> str:
    """构建让 Qwen 自己修改题目的 prompt"""
    question_json = json.dumps(question, ensure_ascii=False, indent=2)
    issues_text = "\n".join(f"- {issue}" for issue in issues)
    kp_name = ""
    subject_name = ""
    if kp_info:
        kp_name = kp_info.get("kp_name", "") or kp_info.get("knowledge_point_name", "")
        subject_name = kp_info.get("subject_name", "") or kp_info.get("_subject_name", "")
    
    return f"""你是试题修复助手。下面是一道试题及其审核发现的问题。请逐一修复所有问题，返回完整的修复后题目JSON。

## 科目/知识点
- 科目：{subject_name}
- 知识点：{kp_name}

## 审核发现的问题
{issues_text}

## 原始题目
```json
{question_json}
```

## 修复要求
1. 逐一修复上述所有问题，确保每个问题都被解决
2. 只改有问题的部分，其他内容保持不变
3. LaTeX 公式必须完整正确（如 \\frac{{分子}}{{分母}}，花括号必须配对）
4. 选项 A/B/C/D 必须齐全，correct_answer 必须为 A/B/C/D 之一
5. image_prompt 中不能出现 $...$、\\frac、\\alpha 等LaTeX源码
6. image_prompt 必须描述具体元素和数量，避免模糊词
7. 返回完整题目JSON，包含所有字段（question_text, options, correct_answer, explanation, image_prompt, image_dependency_reason, question_language, difficulty, knowledge_point, truth_spec, difference_from_others）
8. 直接返回JSON，不要加任何解释文字"""


def normalize_answer(answer) -> str:
    """Normalize a candidate answer to A/B/C/D, or empty string if invalid."""
    if answer is None:
        return ""
    text = str(answer).strip().upper()
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    if text in {"A", "B", "C", "D"}:
        return text
    return ""


def parse_answer(content: str) -> str:
    """Parse Qwen single-rollout output. Prefer JSON; fall back to regex."""
    if not content:
        return ""
    text = re.sub(r'<think>.*?</think>', '', str(content), flags=re.DOTALL).strip()

    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start:i + 1])
                        parsed = normalize_answer(data.get("answer"))
                        if parsed:
                            return parsed
                    except Exception:
                        pass
                    break

    match = re.search(r'"?answer"?\s*[:：]\s*"?([ABCD])"?', text, flags=re.IGNORECASE)
    if match:
        return normalize_answer(match.group(1))
    match = re.search(r"\b([ABCD])\b", text, flags=re.IGNORECASE)
    if match:
        return normalize_answer(match.group(1))
    return ""


def build_rollout_prompt(question: dict, kp_info: dict = None) -> str:
    """Build a candidate-student prompt from an explicit safe whitelist only."""
    safe_question = {
        "question_text": question.get("question_text", ""),
        "options": question.get("options", {}),
    }
    if question.get("image_contract") is not None:
        safe_question["image_contract"] = question.get("image_contract")
    elif question.get("visual_facts") is not None:
        safe_question["visual_facts"] = question.get("visual_facts")

    subject = question.get("subject") or question.get("subject_name")
    if subject:
        safe_question["subject"] = subject

    if kp_info:
        kp_name = kp_info.get("kp_name") or kp_info.get("knowledge_point_name")
        subject_name = kp_info.get("subject_name") or kp_info.get("_subject_name")
        if kp_name:
            safe_question["kp"] = kp_name
        if subject_name and "subject" not in safe_question:
            safe_question["subject"] = subject_name
    elif question.get("kp") or question.get("knowledge_point"):
        safe_question["kp"] = question.get("kp") or question.get("knowledge_point")

    return (
        "You are a student taking a multiple-choice exam. "
        "Use only the question, options, and visual facts below. "
        "Return exactly one JSON object and no explanation: {\"answer\":\"A\"}\n\n"
        f"QUESTION_JSON:\n{json.dumps(safe_question, ensure_ascii=False, sort_keys=True)}"
    )


class QwenReviewer:
    """Qwen 纯文本审核器"""
    
    def __init__(self, base_url: str, api_key: str, model: str = "qwen3.7-max"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def _call_qwen_candidate(self, prompt: str) -> str:
        api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("YUANLAN_API_KEY") or self.api_key
        payload = json.dumps({
            "model": "qwen3.6-flash",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0.7,
            "enable_thinking": True,
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        req = urllib.request.Request(
            "https://yuanlansj.xin/v1/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = json.loads(resp.read().decode())
        return body.get("choices", [{}])[0].get("message", {}).get("content", "")

    def review_rollouts(self, question: dict, kp_info: dict = None, rollouts=5) -> dict:
        """Candidate-student 5-rollout review with local answer comparison.
        
        Early termination: stops as soon as wrong_count reaches 3 (PASS)
        or correct_count exceeds (5-3)=2 making PASS impossible (FAIL).
        """
        t0 = time.time()
        expected = normalize_answer(question.get("correct_answer"))
        prompt = build_rollout_prompt(question, kp_info)
        answers = []
        failures = []
        actual_rollouts = 0

        for idx in range(int(rollouts)):
            actual_rollouts = idx + 1
            try:
                raw = self._call_qwen_candidate(prompt)
                answer = parse_answer(raw)
                if answer:
                    answers.append(answer)
                else:
                    failures.append({"rollout": idx + 1, "error": "parse_error"})
            except Exception as e:
                failures.append({"rollout": idx + 1, "error": type(e).__name__})

            # Early termination: wrong_count >= 3 → PASS immediately
            wrong_so_far = sum(1 for a in answers if a != expected)
            correct_so_far = sum(1 for a in answers if a == expected)
            if wrong_so_far >= 3:
                break
            # Early termination: can't reach wrong_count >= 3 → FAIL immediately
            remaining = int(rollouts) - actual_rollouts
            if correct_so_far > (int(rollouts) - 3):
                break

        correct_count = sum(1 for answer in answers if answer == expected)
        wrong_count = sum(1 for answer in answers if answer != expected)
        valid_count = len(answers)

        base = {
            "correct_count": correct_count,
            "wrong_count": wrong_count,
            "valid_count": valid_count,
            "answers": answers,
            "rollouts": actual_rollouts,
            "latency": round(time.time() - t0, 2),
        }
        if failures:
            base["failures"] = failures

        if valid_count < actual_rollouts:
            return {
                **base,
                "decision": "FAIL",
                "source": "qwen_technical_failure",
                "technical_failure": True,
                "issues": [f"valid answers {valid_count}/{int(rollouts)}"],
                "confidence": 0,
            }
        if wrong_count >= 3:
            decision = "PASS"
        elif correct_count >= 3:
            decision = "FAIL"
        else:
            decision = "FAIL"
        return {
            **base,
            "decision": decision,
            "source": "qwen_candidate_5rollout",
            "technical_failure": False,
            "issues": [],
            "confidence": 1.0,
        }
    
    def review(self, question: dict, kp_info: dict = None) -> dict:
        """
        审核单道题：先本地预校验，再调 Qwen API。
        
        Returns:
            {
                "decision": "PASS|FAIL",
                "checks": {...},
                "issues": [...],
                "confidence": float,
                "latency": float,
                "source": "pre_validate|qwen_api"
            }
        """
        if os.environ.get("QWEN_REVIEW_MODE") == "candidate_5rollout":
            return self.review_rollouts(question, kp_info)

        t0 = time.time()
        
        # 阶段1：本地预校验
        pre_pass, pre_issues = pre_validate(question)
        if not pre_pass:
            return {
                "decision": "FAIL",
                "checks": {},
                "issues": pre_issues,
                "confidence": 1.0,
                "latency": round(time.time() - t0, 2),
                "source": "pre_validate",
            }
        
        # 阶段2：Qwen API 审核
        prompt = build_review_prompt(question, kp_info)
        
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400,
            "temperature": 0,
        }).encode()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        url = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                body = json.loads(resp.read().decode())
                content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                elapsed = time.time() - t0
                
                # 清除 Qwen thinking 标签（qwen3.7-max 默认开启思考模式）
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                
                # 解析 JSON — 花括号计数法精确截取
                start = content.find("{")
                if start >= 0:
                    depth = 0
                    end = start
                    for i in range(start, len(content)):
                        if content[i] == '{':
                            depth += 1
                        elif content[i] == '}':
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                break
                    result = json.loads(content[start:end])
                    
                    decision = result.get("decision", "FAIL").upper()
                    if decision not in ("PASS", "FAIL"):
                        decision = "FAIL"
                    
                    return {
                        "decision": decision,
                        "checks": result.get("checks", {}),
                        "issues": result.get("issues", []),
                        "confidence": float(result.get("confidence", 0)),
                        "latency": round(elapsed, 2),
                        "source": "qwen_api",
                    }
                
                return {
                    "decision": "FAIL",
                    "checks": {},
                    "issues": ["Qwen响应无法解析为JSON"],
                    "confidence": 0,
                    "latency": round(elapsed, 2),
                    "source": "parse_error",
                    "raw": content[:200],
                }
        
        except urllib.error.HTTPError as e:
            return {
                "decision": "FAIL",
                "checks": {},
                "issues": [f"Qwen API HTTP错误: {e.code}"],
                "confidence": 0,
                "latency": round(time.time() - t0, 2),
                "source": "http_error",
            }
        except Exception as e:
            return {
                "decision": "FAIL",
                "checks": {},
                "issues": [f"Qwen API异常: {str(e)[:100]}"],
                "confidence": 0,
                "latency": round(time.time() - t0, 2),
                "source": "exception",
            }

    def revise(self, question: dict, issues: list, kp_info: dict = None):
        """Qwen 自修改：根据审核发现的问题修复题目，返回修复后的完整题目 dict。失败返回 None。"""
        if not issues:
            return question
        prompt = build_revise_prompt(question, issues, kp_info)
        prompt_len = len(prompt)
        max_tokens = min(8000, max(2000, prompt_len // 2 + 500))
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode())
                content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                start = content.find("{")
                if start >= 0:
                    depth = 0
                    end = start
                    for i in range(start, len(content)):
                        if content[i] == '{':
                            depth += 1
                        elif content[i] == '}':
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                break
                    return json.loads(content[start:end])
                return None
        except Exception:
            return None
