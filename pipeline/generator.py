"""
GPT Question Generation Module — image_prompt pipeline.

Generates university-level Chinese multiple-choice questions with a structured
English image_prompt for gpt-image-2 rendering. The old render_code and
render_instruction paths are intentionally removed.
"""
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
try:
    from json_repair import repair_json
except ImportError:  # Optional production hardening; strict JSON paths still work.
    repair_json = None

logger = logging.getLogger(__name__)

_RUN_DIR = Path.home() / "research_projects" / "multimodal_question_bank_24x1000" / "runs" / "qwen_gpt_closed_loop_18subjects_24000_20260619_v2"
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "image_prompt_style.json"


def _load_image_prompt_config() -> dict:
    if _CONFIG_PATH.exists():
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "style_suffix": "Clean academic textbook illustration style, white background, thin black outlines, no shading, no 3D perspective, no drop shadows, no gradients, high contrast, clearly legible labels.",
        "constraints": {
            "min_words": 50,
            "max_words": 200,
            "target_words_min": 100,
            "target_words_max": 160,
            "max_main_elements": 8,
        },
    }


_IMAGE_PROMPT_CONFIG = _load_image_prompt_config()
_CONSTRAINTS = _IMAGE_PROMPT_CONFIG.get("constraints", {})
_STYLE_SUFFIX = _IMAGE_PROMPT_CONFIG.get("style_suffix", "")

SYSTEM_PROMPT_TEMPLATE = """\
你是一个专业的大学考试图文选择题生成器。

任务：为指定知识点生成高质量中文选择题。每道题必须配有一张功能性图片，但你不要写任何渲染代码。每道题必须同时输出 image_contract（图片事实契约）和 image_prompt（一段给 gpt-image-2 使用的英文图片描述）。

## 科目与知识点

- 科目：{subject_name}
- 模块：{module_name}
- 知识点：{kp_name}
- 知识点ID：{kp_id}

## 知识点边界（严禁超出）

{scope_boundary_section}

## 出题方向

{archetype_section}

## 图片类型约束

{image_type_section}

## 本批逐题生产蓝图（必须严格执行）

本知识点的总生产题数不是固定值，而是由 `production_quota` 决定。本次只生成其中一批。
每道题必须严格对应下面的 slot：question_language、difficulty、archetype、image_type、competency、visual_complexity 都不能自行改动。
输出JSON数组的第1个元素对应蓝图第1行，第2个元素对应蓝图第2行，依此类推。
每道题必须在JSON中填写 `question_language`、`blueprint_slot`、`blueprint_archetype`、`blueprint_image_type`，分别等于对应蓝图行的 question_language、slot、archetype、image_type。

{blueprint_section}

## image_contract 与 image_prompt 生成规范（必须严格遵守）

每道题必须输出字段 `image_contract` 和 `image_prompt`。

### image_contract 图片事实契约

`image_contract` 是 JSON 子对象，是图片中允许出现的全部解题事实来源。它必须像“图上可见事实清单”，不能像“解析摘要”。schema 至少包含以下键：
- `version`: 字符串，例如 "1.0"。
- `diagram_type`: 图片类型，必须匹配蓝图 image_type。
- `visual_facts`: 图片中可见但不一定直接决定答案的底层事实列表。
- `answer_relevant_facts`: 解题所需、只能从图片获得的底层事实列表；只能写节点、边、权重、标签、坐标、方向、颜色、局部关系等原子事实，禁止写最终结论。
- `labels`: 图片中允许出现的标签列表。
- `quantities`: 图片中允许出现的数值/计量/坐标/权重，使用对象或数组保存。
- `relations`: 图片中允许出现的连接、空间、因果、流程或对应关系列表。
- `constraints`: 绘图约束列表，包括布局、元素数量、颜色高亮、医学合成限制等。
- `distractor_rationales`: 4个选项中每个错误选项对应的合理误解来源，例如忽略约束、混淆标签、少算一步、把局部最优当全局最优；正确选项写“正确”。
- `qwen36_difficulty_rationale`: 说明本题为什么对 qwen3.6-flash thinking 有挑战，必须来自公平的多步视觉推理，不得来自歧义、错误答案或缺失信息。
- `forbidden_text_leakage`: 不得被 question_text/options 完整泄露的答案相关短语列表。

`image_contract` 禁止出现这些“最终结论式事实”：正确答案、正确选项、最终排序、最短路径、最大/最小对象、最终类别、最终数值、path total、shortest path、maximum/minimum、best/worst、answer is、option A/B/C/D is correct。最终结论只能放在 `explanation` 和 `truth_spec.validation_rules` 中。

### image_prompt 图片生成提示词

`image_prompt` 是 gpt-image-2 的英文图片生成提示词，不是代码，不是JSON子对象。必须从 `image_contract` 派生：只能重写、组织和视觉化 contract 中已有的 diagram_type、visual_facts、answer_relevant_facts、labels、quantities、relations、constraints，不得引入 contract 外的新解题事实。

1. 语言：英文为主体；如果图片中需要中文标签，中文标注直接写在 prompt 里。
2. 长度：目标 {target_words_min}-{target_words_max} 个英文词；允许范围 {min_words}-{max_words} 词。不要把统一风格后缀写进 image_prompt，系统会自动追加。
3. 结构必须按顺序包含：
   - 图类型声明：开头说明是什么图，如 diagram / chart / graph / circuit / structure / table / map。
   - 全局布局：视角、区域划分、方向，如 viewed from above, split into left and right regions。
   - 具体元素：节点、组件、结构、数量、位置关系，必须写具体数字。
   - 连接关系：边、线、箭头、流程、映射关系，逐条写明谁连谁。
   - 标注要求：需要显示的元素必须写 `(labeled "xxx")`，中文标注直接写中文。
   - 视觉收尾：简短说明局部颜色或高亮，但不要写统一风格后缀。
4. 精确性：元素数量必须明确，禁止 several / some / a few / many / various / multiple 这类模糊词。
5. 复杂度：主要元素不超过 {max_main_elements} 个；超过时必须简化成核心可判题结构。
6. 事实闭包：image_prompt 中的解题相关事实必须能在 image_contract 中逐项找到；不得添加 contract 没写的新数值、新标签、新关系、新高亮或新答案线索。
7. 文本防泄露：question_text/options 不得完整泄露 image_contract.forbidden_text_leakage 或 answer_relevant_facts 中的任何条目；题干只能说明作答任务，不能把图片里的关键答案事实原样写出。
8. 医学限制：{medical_image_instruction}
9. 禁止事项：禁止输出 render_code、render_instruction、engine、diagram_type 顶层字段、Python、matplotlib、TikZ、LaTeX绘图代码、Markdown。

系统自动追加的统一风格后缀如下，你不要重复写：
{style_suffix}

## 公式与语言规则（必须严格遵守）

1. 题目文本保存规则：question_text、options、explanation 中所有数学/物理/化学公式必须用标准 LaTeX 行内格式 `$...$` 保存，例如 `$E=mc^2$`、`$\\frac{{1}}{{2}}mv^2$`、`$K_c=\\frac{{[C]^c[D]^d}}{{[A]^a[B]^b}}$`。
2. 图片显示规则：image_prompt 中如果需要图片显示公式，必须描述为 rendered formula / standard mathematical notation，例如 `a rendered formula E=mc² above the curve`。禁止要求图片显示 `$...$`、`\\frac`、`\\alpha` 等 LaTeX 源码字符。
3. 语言比例规则：按蓝图的 `question_language` 生成题目。`zh` 表示 question_text/options/explanation 用中文；`en` 表示 question_text/options/explanation 用英文。整体英文题约每6题1题（约16.7%）。
4. image_prompt 仍以英文为主体；若题目是中文题，图片中必要中文标签可以直接写中文；若题目是英文题，图片标签优先英文。

## Qwen3.6 难度门（必须严格遵守 — 这是题目的唯一通过标准）

本题库的唯一质量门是：qwen3.6-flash thinking 独立作答 5 次，必须有 ≥3 次答错（wrong_count ≥ 3）。如果题目让 qwen3.6-flash 5 次全对或只错 1-2 次，则该题被拒，需回炉重写。这意味着你生成的题目必须足够难，让一个强大的视觉推理模型在 5 次尝试中至少 3 次失败。

目标不是制造烂题或骗模型，而是生成"人类大学生可严谨解答、但 qwen3.6-flash thinking 容易不稳定"的高区分度题。难度只能来自公平因素：多视觉事实组合、多步计算、条件约束、相近干扰项、常见概念误解、局部事实与全局结论冲突。

每题必须满足：
1. 至少需要 3-4 个推理步骤：读取多个图中局部事实 → 组合计算/比较/约束筛选 → 排除干扰项得到唯一答案。仅需 2 步的题大概率会被 qwen3.6 全对。
2. 禁止只读一个标签、一个数值、一个箭头方向、一个颜色或一个局部关系即可作答。
3. 至少有一个干扰项必须利用"常见概念误解"或"漏算一个约束条件"导致的错误路径。
4. 题目涉及的数量关系必须足够复杂：至少 2 个数值需要从图中提取并运算，且运算结果不能显而易见。
5. 题干和选项不能把关键视觉事实完整说出；正确答案不能从文本本身推出。
6. `image_contract` 只能给底层可见事实，不能给最终结论；例如允许写 `edge A-B has weight 2`，禁止写 `shortest path is A-C-D` 或 `path A-C-D total is 6`。
7. 每个错误选项必须是合理干扰项，并在 `image_contract.distractor_rationales` 中说明它对应的误解来源。
8. 题目不得靠歧义、答案错误、缺失条件、超出知识点、文字陷阱或图片无法表达的细节来让 Qwen 出错。
9. **强制性自查**：在提交前，问自己"qwen3.6-flash thinking 能否轻松答对这题？"如果答案是"很可能 5 次全对"，必须立即重写：增加一个公平约束、增加一步图中事实组合、或把一个干扰项改成常见错误路径。目标是让 qwen3.6 至少错 3 次。

## 生成要求

1. 每道题必须是4个选项（A/B/C/D），仅一个正确答案。
2. 大学本科及以上难度，不能靠常识或简单读图直接秒选。
3. 图片必须承载解题所需的关键信息；删除图片后无法确定答案。
4. 题干不得完整复述图片中的全部关键信息。
5. 答案必须唯一且可验证。
6. explanation 必须完整到能独立复核答案。
7. image_prompt 中的数值、标签、结构、连接关系必须与题目答案和解析完全一致。
8. 文本公式必须按上面的LaTeX保存规则；图片公式必须按上面的标准公式显示规则。
9. 本批题目之间不得高度相似，不得仅改数字、变量名、颜色或布局。
10. 不得出现任何老渲染字段：render_code、render_instruction、render_engine、render_params、image_description。

## 避免重复

{dedup_instruction}

## 禁止模式

{prohibited_patterns}

## 输出格式

只输出一个 JSON 数组，不要输出解释、Markdown 或代码块。数组长度必须等于 {batch_size}。

[
  {{{{
    "question_id": "{item_id_prefix}_001",
    "question_text": "题目文本（中文）",
    "options": {{{{"A": "选项A", "B": "选项B", "C": "选项C", "D": "选项D"}}}},
    "correct_answer": "A",
    "explanation": "详细解析过程（中文）",
    "difficulty": 5,
    "question_language": "zh",
    "blueprint_slot": 1,
    "blueprint_archetype": "本题对应的蓝图题型",
    "blueprint_image_type": "本题对应的蓝图图片类型",
    "knowledge_point": "{kp_name}",
    "image_contract": {{{{
      "version": "1.0",
      "diagram_type": "本题对应的蓝图图片类型",
      "visual_facts": ["图片中允许出现的非答案泄露可见事实"],
      "answer_relevant_facts": ["只在图片中出现、题干和选项不得完整泄露的关键解题事实"],
      "labels": ["图片中允许显示的标签"],
      "quantities": {{{{"example_value": 1}}}},
      "relations": ["图片中允许出现的连接/空间/流程/对应关系"],
      "constraints": ["布局、元素数量、颜色高亮、医学合成图限制等"],
      "distractor_rationales": {{{{"A": "正确", "B": "错误选项B对应的合理误解", "C": "错误选项C对应的合理误解", "D": "错误选项D对应的合理误解"}}}},
      "qwen36_difficulty_rationale": "说明本题为何需要多步视觉推理、为何可能让 qwen3.6-flash thinking 不稳定，但不是靠歧义或错误条件",
      "forbidden_text_leakage": ["不得在question_text/options完整出现的短语"]
    }}}},
    "image_prompt": "A precise English prompt derived only from image_contract, following the required structure, with Chinese labels if needed.",
    "truth_spec": {{{{
      "correct_answer": "A",
      "image_only_facts": ["只在图片中出现、题干未完整给出的关键信息"],
      "validation_rules": ["验证答案正确性的规则"]
    }}}},
    "image_dependency_reason": "说明为什么必须看图才能作答",
    "difference_from_others": "说明本题与本批其他题的本质区别"
  }}}}
]

## 输出前自检

1. 题目数量是否等于 {batch_size}。
2. 每道题是否真正依赖图片。
3. correct_answer 与 truth_spec.correct_answer 是否一致。
4. image_contract 是否包含 version、diagram_type、visual_facts、answer_relevant_facts、labels、quantities、relations、constraints、distractor_rationales、qwen36_difficulty_rationale、forbidden_text_leakage。
5. image_contract 是否只写底层视觉事实，没有写正确答案、最短路径、最大/最小对象、最终排序、最终类别、最终数值等解析结论。
6. question_text/options 是否没有完整泄露 forbidden_text_leakage 或 answer_relevant_facts。
7. 题目是否至少需要 2-3 个推理步骤，且不能只读单一标签/数值/颜色即可作答。
8. 每个错误选项是否有合理误解来源，而不是随机干扰项。
9. image_prompt 是否为英文主体、目标 {target_words_min}-{target_words_max} 词、没有统一风格后缀。
10. image_prompt 是否按图类型声明→全局布局→具体元素→连接关系→标注→视觉收尾的顺序写。
11. image_prompt 是否只使用 image_contract 中已有事实，并与答案、解析完全一致。
12. 医学 S07-S12 是否只使用 synthetic/simulated 教学图，不声称真实 patient image。
13. 是否出现任何 render_code/render_instruction/render_engine/render_params 字段；如果出现就是错误。
14. JSON是否合法可解析。
"""


class QuestionGenerator:
    """Generates exam questions via GPT API with image_prompt fields."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gpt-5.5",
        max_concurrent: int = 16,
        response_log_dir: Optional[str] = None,
        api_mode: str = "openai",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.api_mode = api_mode
        self.max_concurrent = max_concurrent
        self.response_log_dir = Path(response_log_dir) if response_log_dir else _RUN_DIR / "api_responses"
        self.response_log_dir.mkdir(parents=True, exist_ok=True)
        self._seq_counter = {}
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        logger.info(
            "QuestionGenerator initialized: model=%s, max_concurrent=%s, base_url=%s",
            model,
            max_concurrent,
            self.base_url,
        )

    def _call_api(self, messages: list, max_tokens: int = 16000, temperature: float = 0.7) -> requests.Response:
        throttle = getattr(self, '_throttle', None)
        acquired = False
        if throttle is not None:
            acquired = throttle.acquire(timeout=600.0)  # 10min: 不因队列满而丢题
            if not acquired:
                raise RuntimeError("AdaptiveThrottle acquire timeout or stop_event")
        try:
            if self.api_mode == "anthropic":
                url = f"{self.base_url}/messages"
                headers = {
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                }
                system_text = ""
                api_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        system_text += msg["content"] + "\n"
                    else:
                        api_messages.append(msg)
                payload = {
                    "model": self.model,
                    "messages": api_messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if system_text.strip():
                    payload["system"] = system_text.strip()
            else:
                url = f"{self.base_url}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            return requests.post(url, headers=headers, json=payload, timeout=300)
        finally:
            if acquired:
                throttle.release()

    def _log_response(self, kp_id: str, attempt: int, raw_text: str, status_code: int):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = self.response_log_dir / f"{kp_id}_{timestamp}_attempt{attempt}.json"
        log_data = {
            "kp_id": kp_id,
            "attempt": attempt,
            "status_code": status_code,
            "timestamp": timestamp,
            "raw_response": raw_text,
        }
        try:
            log_file.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to log response: %s", e)

    def _parse_json_response(self, text: str) -> Optional[list]:
        if not text:
            return None
        code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
        for match in re.findall(code_block_pattern, text, re.DOTALL):
            try:
                result = json.loads(match)
                if self._looks_like_question_array(result):
                    return result
            except json.JSONDecodeError:
                continue
        bracket_start = text.find("[")
        if bracket_start != -1:
            depth = 0
            in_string = False
            escape = False
            for i in range(bracket_start, len(text)):
                ch = text[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\" and in_string:
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            result = json.loads(text[bracket_start:i + 1])
                            if self._looks_like_question_array(result):
                                return result
                        except json.JSONDecodeError:
                            break
        try:
            result = json.loads(text.strip())
            if self._looks_like_question_array(result):
                return result
        except json.JSONDecodeError:
            pass
        cleaned = text.strip()
        if cleaned.startswith("["):
            cleaned = re.sub(r",\s*\]", "]", cleaned)
            cleaned = re.sub(r",\s*\}", "}", cleaned)
            try:
                result = json.loads(cleaned)
                if self._looks_like_question_array(result):
                    return result
            except json.JSONDecodeError:
                pass

        # ── JSON repair strategies (python post-processing, no API cost) ──

        # Strategy 5: repair_json on bracket-extracted array segment
        bracket_start = text.find("[")
        if repair_json is not None and bracket_start != -1:
            end = self._find_matching_bracket(text, bracket_start)
            if end != -1:
                candidate = text[bracket_start:end + 1]
                try:
                    repaired = repair_json(candidate)
                    result = json.loads(repaired)
                    if self._looks_like_question_array(result):
                        logger.info("JSON repaired via bracket-extract+repair_json (length=%s)", len(candidate))
                        return result
                except Exception:
                    pass

        # Strategy 6: repair_json on full text
        if repair_json is not None:
            try:
                repaired = repair_json(text.strip())
                result = json.loads(repaired)
                if self._looks_like_question_array(result):
                    logger.info("JSON repaired via full-text repair_json (length=%s)", len(text))
                    return result
            except Exception:
                pass

        logger.error("Failed to parse JSON from response (length=%s)", len(text))
        return None

    @staticmethod
    def _looks_like_question_array(result: object) -> bool:
        if not isinstance(result, list):
            return False
        if not result:
            return True
        return all(isinstance(item, dict) and "question_text" in item for item in result)

    @staticmethod
    def _find_matching_bracket(text: str, start: int) -> int:
        """Find the matching ']' for a '[' at position start, respecting strings."""
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return i
        return -1

    def _build_messages(self, kp_info: dict, batch_size: int, existing_questions: Optional[list] = None) -> list:
        subject_name = kp_info.get("subject_name") or kp_info.get("_subject_name") or kp_info.get("subject_id", "未知学科")
        subject_id = kp_info.get("subject_id", "")
        module_name = kp_info.get("module_name", "")
        kp_name = kp_info.get("kp_name") or kp_info.get("knowledge_point_name") or "未知知识点"
        kp_id = kp_info.get("kp_id") or kp_info.get("knowledge_point_id", "unknown")
        item_id_prefix = f"{kp_id}_Q"

        scope = kp_info.get("scope_boundary", "")
        if scope:
            scope_boundary_section = f"本知识点的精确范围：{scope}\n严禁超出此边界出题。"
        else:
            scope_boundary_section = "无明确边界定义。请严格围绕知识点名称出题，不得偏离。"

        archetypes = kp_info.get("question_archetypes", [])
        competency = kp_info.get("competency_types", [])
        if archetypes:
            archetype_section = "推荐出题方向（请覆盖多种，不要集中在同一种）：\n" + "\n".join(f"- {a}" for a in archetypes)
        else:
            archetype_section = "无特定出题方向约束，请根据知识点自然设计多种变式。"
        if competency:
            archetype_section += f"\n\n考查能力类型：{', '.join(competency)}"

        allowed_images = kp_info.get("allowed_image_types", [])
        required_visual = kp_info.get("required_visual_information", "")
        if allowed_images:
            image_type_section = f"本知识点允许的图片类型：{', '.join(allowed_images)}\n请只使用上述图片类型。"
        else:
            image_type_section = "无特定图片类型限制，请根据知识点选择最合适的可视化方式。"
        if required_visual:
            image_type_section += f"\n图片必须展示的信息：{required_visual}"

        if subject_id in {"S07", "S08", "S09", "S10", "S11", "S12"}:
            medical_image_instruction = (
                "医学科目必须使用 synthetic educational medical diagrams / simulated ECG / "
                "simulated pathology-style diagram / anatomy teaching diagram / lab chart / "
                "pharmacokinetic curve 等教学图；image_contract.constraints 必须写明 synthetic_medical_only；"
                "禁止声称或暗示真实 patient image、真实病人照片、真实影像或真实病例图。"
            )
        else:
            medical_image_instruction = "非医学科目按知识点选择合适的学术教学图；不得引入真实病人图像声称。"

        if existing_questions:
            existing_texts = [q.get("question_text", "")[:60] for q in existing_questions[:20]]
            dedup_instruction = "以下题目已存在，新题必须与它们有本质区别：\n" + "\n".join(f"- {t}..." for t in existing_texts if t)
        else:
            dedup_instruction = "无需避免重复（这是第一批生成）。"

        prohibited = kp_info.get("prohibited_patterns", [])
        if prohibited:
            prohibited_patterns = "以下模式禁止出现：\n" + "\n".join(f"- {p}" for p in prohibited)
        else:
            prohibited_patterns = "无额外禁止模式。"

        full_blueprint = kp_info.get("question_blueprint") or []
        start_slot = len(existing_questions or [])
        batch_blueprint = full_blueprint[start_slot:start_slot + batch_size]
        if not batch_blueprint:
            batch_blueprint = [
                {
                    "slot": start_slot + i + 1,
                    "question_language": "en" if (start_slot + i + 1) % 6 == 0 else "zh",
                    "difficulty": 5,
                    "archetype": (archetypes or ["基于图示信息判断关键概念"])[i % max(1, len(archetypes or [1]))],
                    "image_type": (allowed_images or ["academic_diagram"])[i % max(1, len(allowed_images or [1]))],
                    "competency": (competency or ["diagram_interpretation"])[i % max(1, len(competency or [1]))],
                    "visual_complexity": "multi-step reasoning requiring 3+ visual fact combinations; 5-8 elements with non-trivial spatial/quantitative relations; at least one distractor exploiting common misconception path",
                    "design_goal": "按知识点生成一题图文一致的选择题。",
                }
                for i in range(batch_size)
            ]
        blueprint_lines = []
        for bp in batch_blueprint:
            blueprint_lines.append(
                f"- slot {bp.get('slot')}: question_language={bp.get('question_language')}; "
                f"difficulty={bp.get('difficulty')}/5; "
                f"archetype={bp.get('archetype')}; image_type={bp.get('image_type')}; "
                f"competency={bp.get('competency')}; visual_complexity={bp.get('visual_complexity')}; "
                f"design_goal={bp.get('design_goal')}"
            )
        blueprint_section = "\n".join(blueprint_lines)

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            kp_name=kp_name,
            subject_name=subject_name,
            module_name=module_name,
            kp_id=kp_id,
            batch_size=batch_size,
            scope_boundary_section=scope_boundary_section,
            archetype_section=archetype_section,
            image_type_section=image_type_section,
            blueprint_section=blueprint_section,
            dedup_instruction=dedup_instruction,
            prohibited_patterns=prohibited_patterns,
            item_id_prefix=item_id_prefix,
            style_suffix=_STYLE_SUFFIX,
            min_words=_CONSTRAINTS.get("min_words", 50),
            max_words=_CONSTRAINTS.get("max_words", 200),
            target_words_min=_CONSTRAINTS.get("target_words_min", 100),
            target_words_max=_CONSTRAINTS.get("target_words_max", 160),
            max_main_elements=_CONSTRAINTS.get("max_main_elements", 8),
            medical_image_instruction=medical_image_instruction,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请为知识点「{kp_name}」生成 {batch_size} 道带 image_prompt 的选择题。"},
        ]

    def _get_next_seq(self, kp_id: str, count: int = 1) -> int:
        if kp_id not in self._seq_counter:
            self._seq_counter[kp_id] = 0
        start = self._seq_counter[kp_id]
        self._seq_counter[kp_id] += count
        return start

    def _assign_question_ids(self, questions: list, kp_info: dict) -> list:
        subject_id = kp_info.get("subject_id", "SUB00")
        kp_id = kp_info.get("kp_id", "KP000")
        kp_id_short = kp_id.split("-")[-1] if "-" in kp_id else kp_id[:8]
        start_seq = self._get_next_seq(kp_id, len(questions))
        for i, q in enumerate(questions):
            seq = start_seq + i + 1
            q["question_id"] = f"{subject_id}-{kp_id_short}-Q{seq:04d}"
            q["knowledge_point_id"] = kp_id
        return questions

    @staticmethod
    def _word_count(text: str) -> int:
        english_parts = re.sub(r"[\u4e00-\u9fff]+", " ZH ", text or "")
        return len(english_parts.split())

    @staticmethod
    def _normalise_leak_text(value) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip().lower()

    @staticmethod
    def _flatten_contract_text(value) -> str:
        if isinstance(value, dict):
            return "\n".join(QuestionGenerator._flatten_contract_text(v) for v in value.values())
        if isinstance(value, list):
            return "\n".join(QuestionGenerator._flatten_contract_text(v) for v in value)
        return str(value or "")

    @classmethod
    def _contract_contains_final_conclusion(cls, contract: dict) -> bool:
        text = cls._normalise_leak_text(
            "\n".join(
                cls._flatten_contract_text(contract.get(key))
                for key in ("visual_facts", "answer_relevant_facts", "relations", "constraints", "quantities")
            )
        )
        conclusion_patterns = [
            r"\banswer\s+is\b",
            r"\boption\s+[abcd]\s+(?:is\s+)?correct\b",
            r"\bcorrect\s+(?:answer|option)\b",
            r"\bshortest\s+path\b",
            r"\blongest\s+path\b",
            r"\bpath\s+total\b",
            r"\btotal\s+(?:cost|weight|length)\s+(?:is|=)\b",
            r"\bminimum\b",
            r"\bmaximum\b",
            r"\bmin\b",
            r"\bmax\b",
            r"\bbest\b",
            r"\bworst\b",
            r"最短路径",
            r"最长路径",
            r"正确答案",
            r"正确选项",
            r"答案是",
            r"最大值",
            r"最小值",
            r"最终排序",
            r"最终类别",
        ]
        return any(re.search(pattern, text) for pattern in conclusion_patterns)

    def _validate_image_contract(self, contract, q: dict) -> bool:
        if not isinstance(contract, dict):
            logger.debug("Invalid image_contract: not an object")
            return False

        required_keys = [
            "version",
            "diagram_type",
            "visual_facts",
            "answer_relevant_facts",
            "labels",
            "quantities",
            "relations",
            "constraints",
            "distractor_rationales",
            "qwen36_difficulty_rationale",
            "forbidden_text_leakage",
        ]
        for key in required_keys:
            if key not in contract:
                logger.debug("Invalid image_contract: missing %s", key)
                return False

        list_keys = [
            "visual_facts",
            "answer_relevant_facts",
            "labels",
            "relations",
            "constraints",
            "forbidden_text_leakage",
        ]
        for key in list_keys:
            if not isinstance(contract.get(key), list):
                logger.debug("Invalid image_contract.%s: expected list", key)
                return False

        if not str(contract.get("version", "")).strip() or not str(contract.get("diagram_type", "")).strip():
            return False
        if not isinstance(contract.get("quantities"), (dict, list)):
            logger.debug("Invalid image_contract.quantities: expected object or list")
            return False
        if not contract.get("answer_relevant_facts"):
            logger.debug("Invalid image_contract: answer_relevant_facts is empty")
            return False
        rationales = contract.get("distractor_rationales")
        if not isinstance(rationales, dict) or not all(key in rationales for key in ["A", "B", "C", "D"]):
            logger.debug("Invalid image_contract.distractor_rationales: expected A/B/C/D mapping")
            return False
        if not str(contract.get("qwen36_difficulty_rationale", "")).strip():
            logger.debug("Invalid image_contract: missing qwen36_difficulty_rationale")
            return False
        if self._contract_contains_final_conclusion(contract):
            logger.debug("Invalid image_contract: contains final-answer conclusion instead of atomic visual facts")
            return False

        visible_text = self._normalise_leak_text(
            " ".join(
                [str(q.get("question_text", ""))]
                + [str(v) for v in (q.get("options") or {}).values()]
            )
        )
        protected_items = contract.get("forbidden_text_leakage", []) + contract.get("answer_relevant_facts", [])
        for item in protected_items:
            item_text = self._normalise_leak_text(item)
            if item_text and len(item_text) >= 8 and item_text in visible_text:
                logger.debug("Invalid question: text leaks protected image fact: %s", item)
                return False

        return True

    def _validate_question(self, q: dict) -> bool:
        required_fields = [
            "question_text",
            "options",
            "correct_answer",
            "explanation",
            "difficulty",
            "question_language",
            "image_contract",
            "image_prompt",
            "image_dependency_reason",
        ]
        for field in required_fields:
            if field not in q or not q[field]:
                logger.debug("Invalid question: missing %s", field)
                return False

        forbidden_fields = ["render_code", "render_instruction", "render_engine", "render_params", "image_description"]
        for field in forbidden_fields:
            if field in q:
                logger.debug("Invalid question: old render field present: %s", field)
                return False

        options = q.get("options", {})
        if not isinstance(options, dict) or not all(k in options for k in ["A", "B", "C", "D"]):
            return False
        option_values = [str(options[k]).strip() for k in ["A", "B", "C", "D"]]
        if len(set(option_values)) != 4:
            return False

        if q["correct_answer"] not in ["A", "B", "C", "D"]:
            return False

        if q.get("question_language") not in ["zh", "en"]:
            return False

        contract = q.get("image_contract")
        if not self._validate_image_contract(contract, q):
            return False

        try:
            diff = int(q["difficulty"])
            if diff < 1 or diff > 5:
                return False
            q["difficulty"] = diff
        except (ValueError, TypeError):
            return False

        image_prompt = str(q.get("image_prompt", "")).strip()
        wc = self._word_count(image_prompt)
        if wc < _CONSTRAINTS.get("min_words", 50) or wc > _CONSTRAINTS.get("max_words", 200):
            logger.debug("Invalid image_prompt word count: %s", wc)
            return False

        vague_words = ["several", "some", "a few", "many", "various", "multiple"]
        if any(re.search(r"\b" + re.escape(word) + r"\b", image_prompt, re.IGNORECASE) for word in vague_words):
            return False

        reject_keywords = _IMAGE_PROMPT_CONFIG.get("validation", {}).get("reject_if_contains_suffix_keywords", [])
        prompt_lower = image_prompt.lower()
        if any(keyword.lower() in prompt_lower for keyword in reject_keywords):
            return False

        patterns = _IMAGE_PROMPT_CONFIG.get("validation", {}).get("required_opening_patterns", [])
        if patterns and not any(re.match(pattern, image_prompt, re.IGNORECASE) for pattern in patterns):
            return False

        raw_latex_markers = ["$", "\\\\frac", "\\\\alpha", "\\\\beta", "\\\\gamma", "\\\\sum", "\\\\int", "\\\\sqrt"]
        if any(marker in image_prompt for marker in raw_latex_markers):
            logger.debug("Invalid image_prompt: raw LaTeX marker appears in visible image prompt")
            return False

        if "truth_spec" not in q or not isinstance(q.get("truth_spec"), dict):
            q["truth_spec"] = {
                "correct_answer": q.get("correct_answer", ""),
                "image_only_facts": [],
                "validation_rules": [],
            }
        if "difference_from_others" not in q:
            q["difference_from_others"] = ""
        if "knowledge_point" not in q:
            q["knowledge_point"] = ""

        return True

    def _generate_one(self, kp_info: dict, existing_questions: list, slot_index: int = 0) -> Optional[dict]:
        """Generate a single question via one API call with retry logic.

        Called concurrently by ThreadPoolExecutor from generate_batch().
        The AdaptiveThrottle in _call_api() limits global in-flight API calls
        and is hot-adjusted by adaptive_controller based on qwen_queue fill ratio.
        """
        kp_id = kp_info.get("kp_id", "unknown")
        max_retries = 3
        backoff_times = [60, 120, 240]

        for attempt in range(1, max_retries + 1):
            messages = self._build_messages(kp_info, 1, existing_questions)
            try:
                response = self._call_api(messages)
                status_code = response.status_code
                raw_text = response.text
                self._log_response(kp_id, attempt, raw_text, status_code)

                if status_code == 429:
                    wait_time = backoff_times[min(attempt - 1, len(backoff_times) - 1)]
                    logger.warning("Rate limited (429) for KP %s slot %s, waiting %ss", kp_id, slot_index, wait_time)
                    time.sleep(wait_time)
                    continue
                if status_code != 200:
                    logger.error("API error %s for KP %s slot %s: %s", status_code, kp_id, slot_index, raw_text[:200])
                    time.sleep(5)
                    continue

                response_data = response.json()
                content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not content:
                    reasoning = response_data.get("choices", [{}])[0].get("message", {}).get("reasoning_content", "")
                    if reasoning:
                        logger.warning(
                            "Empty content but reasoning_content=%s chars, using reasoning as content fallback",
                            len(reasoning),
                        )
                        content = reasoning
                if not content:
                    logger.warning("Empty content in response for KP %s slot %s", kp_id, slot_index)
                    continue

                questions = self._parse_json_response(content)
                if questions is None:
                    logger.warning("JSON parse failed for KP %s slot %s (attempt %s)", kp_id, slot_index, attempt)
                    continue

                valid_questions = [q for q in questions if isinstance(q, dict) and self._validate_question(q)]
                if valid_questions:
                    q = valid_questions[0]
                    q["_slot_index"] = slot_index
                    logger.info("KP %s slot %s: generated 1 valid question (attempt %s)", kp_id, slot_index, attempt)
                    return q

            except requests.exceptions.Timeout:
                logger.error("Timeout for KP %s slot %s (attempt %s)", kp_id, slot_index, attempt)
                time.sleep(10)
            except requests.exceptions.ConnectionError as e:
                logger.error("Connection error for KP %s slot %s: %s", kp_id, slot_index, e)
                time.sleep(10)
            except Exception as e:
                logger.error("Unexpected error for KP %s slot %s: %s", kp_id, slot_index, e, exc_info=True)
                time.sleep(5)

        return None

    def generate_batch(self, kp_info: dict, batch_size: int, existing_questions: Optional[list] = None) -> list:
        """Generate a batch of questions by concurrent independent API calls.

        Each question is generated via a separate API call, executed concurrently
        through ThreadPoolExecutor(max_workers=self.max_concurrent).
        AdaptiveThrottle in _call_api() provides hot-pluggable global concurrency
        control, adjusted by adaptive_controller based on qwen_queue fill ratio.
        """
        kp_id = kp_info.get("kp_id", "unknown")
        if existing_questions is None:
            existing_questions = []

        # Submit batch_size independent generation tasks
        futures = {}
        for slot in range(batch_size):
            future = self._executor.submit(
                self._generate_one, kp_info, existing_questions, slot
            )
            futures[future] = slot

        all_questions = []
        for future in as_completed(futures):
            try:
                question = future.result()
                if question is not None:
                    all_questions.append(question)
                    # Feed back into existing_questions for subsequent batches
                    # (not perfect for full dedup but acceptable for concurrent generation)
                    existing_questions.append(question)
            except Exception as e:
                logger.error("Concurrent generation failed for KP %s slot %s: %s",
                             kp_id, futures[future], e)

        all_questions = all_questions[:batch_size]
        all_questions = self._assign_question_ids(all_questions, kp_info)
        logger.info("KP %s: generated %s/%s questions via %s-way concurrency",
                    kp_id, len(all_questions), batch_size, self.max_concurrent)
        return all_questions

    def generate_batch_async(self, kp_infos: list, batch_size: int, existing_questions_map: Optional[dict] = None) -> dict:
        if existing_questions_map is None:
            existing_questions_map = {}
        results = {}
        futures = {}
        for kp_info in kp_infos:
            kp_id = kp_info.get("kp_id", "unknown")
            existing = existing_questions_map.get(kp_id, [])
            future = self._executor.submit(self.generate_batch, kp_info, batch_size, existing)
            futures[future] = kp_id
        for future in as_completed(futures):
            kp_id = futures[future]
            try:
                results[kp_id] = future.result()
            except Exception as e:
                logger.error("Async generation failed for KP %s: %s", kp_id, e)
                results[kp_id] = []
        return results

    def build_generation_prompt(self, kp_info: dict, batch_size: int = 1) -> list:
        return self._build_messages(kp_info, batch_size)

    def build_review_prompt(self, question: dict, kp_info: dict = None) -> list:
        from pipeline.reviewer import build_review_prompt
        return [{"role": "user", "content": build_review_prompt(question, kp_info)}]

    def parse_review_response(self, response: dict) -> dict:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return {"status": "FAIL", "issues": ["无法解析审核响应"], "checks": {}, "confidence": 0}
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        start = content.find("{")
        if start >= 0:
            depth = 0
            end = start
            for i in range(start, len(content)):
                if content[i] == "{":
                    depth += 1
                elif content[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            try:
                result = json.loads(content[start:end])
                decision = (result.get("decision") or result.get("verdict") or "FAIL").upper()
                if decision not in ("PASS", "FAIL"):
                    decision = "FAIL"
                return {
                    "status": decision,
                    "issues": result.get("issues", []),
                    "checks": result.get("checks", {}),
                    "confidence": float(result.get("confidence", 0)),
                }
            except (json.JSONDecodeError, ValueError):
                pass
        return {"status": "FAIL", "issues": ["审核响应非法JSON"], "checks": {}, "confidence": 0}

    def build_fix_prompt(self, question: dict, verdict: dict) -> list:
        return self.build_regen_prompt({}, question.get("question_text", ""), verdict, existing_questions=[])

    def parse_fix_response(self, response: dict) -> Optional[dict]:
        questions = self.parse_response(response)
        return questions[0] if questions else None

    def build_regen_prompt(self, kp_info: dict, old_question_text: str, verdict: dict, existing_questions: Optional[list] = None) -> list:
        kp_name = kp_info.get("kp_name") or kp_info.get("knowledge_point_name") or "未知知识点"
        subject_name = kp_info.get("subject_name") or kp_info.get("_subject_name") or "未知学科"
        module_name = kp_info.get("module_name", "")
        kp_id = kp_info.get("kp_id") or kp_info.get("knowledge_point_id", "unknown")
        scope = kp_info.get("scope_boundary", "")
        archetypes = kp_info.get("question_archetypes", [])
        allowed_images = kp_info.get("allowed_image_types", [])
        competency = kp_info.get("competency_types", [])

        issues = verdict.get("issues", [])
        checks = verdict.get("checks", {})
        failure_types = []
        avoidance_instructions = []
        if checks.get("answer_correct") is False:
            failure_types.append("答案或解析错误")
            avoidance_instructions.append("答案必须唯一且可验证，解析必须完整推导出正确选项。")
        if checks.get("options_reasonable") is False:
            failure_types.append("选项不合理")
            avoidance_instructions.append("四个选项必须互不重复，且只有一个正确答案。")
        if checks.get("prompt_consistent") is False:
            failure_types.append("image_prompt与题目不一致")
            avoidance_instructions.append("image_prompt中的数值、标签、结构、连接关系必须与题目、答案、解析完全一致。")
        if checks.get("image_dependency_valid") is False:
            failure_types.append("图片依赖不成立")
            avoidance_instructions.append("题目必须真正依赖图片信息，删除图片后不能确定答案。")
        if checks.get("scope_correct") is False:
            failure_types.append("超出知识点范围")
            avoidance_instructions.append("严格在scope_boundary范围内出题，不要涉及相邻但不属于本知识点的内容。")
        if checks.get("prompt_compliant") is False:
            failure_types.append("image_prompt不合规")
            avoidance_instructions.append("image_prompt必须英文主体、目标100-160词、结构明确、元素数量清楚、禁止模糊词和统一风格后缀。")
        if not failure_types and issues:
            failure_types.append("审核未通过")
            avoidance_instructions.append("审核指出的问题类型：" + "; ".join(str(i) for i in issues[:3]))

        failure_type_text = "、".join(failure_types) if failure_types else "质量未达标"
        avoidance_text = "\n".join(f"- {i}" for i in avoidance_instructions) or "- 严格按 image_prompt 新模板重新设计。"

        dedup_items = []
        if existing_questions:
            for q in existing_questions[:15]:
                text = q.get("question_text", "")[:50]
                if text:
                    dedup_items.append(f"[已通过] {text}...")
        if old_question_text:
            dedup_items.append(f"[已丢弃] {old_question_text[:50]}...")
        dedup_section = "\n".join(f"- {item}" for item in dedup_items) if dedup_items else "无。"

        regen_kp = {
            "subject_name": subject_name,
            "module_name": module_name,
            "kp_name": kp_name,
            "kp_id": kp_id,
            "scope_boundary": scope,
            "question_archetypes": archetypes,
            "allowed_image_types": allowed_images,
            "competency_types": competency,
            "prohibited_patterns": [],
        }
        messages = self._build_messages(regen_kp, 1, existing_questions=[])
        messages[1]["content"] = f"请为知识点「{kp_name}」重新生成 1 道全新题目。上一题因「{failure_type_text}」失败。\n\n避错指令：\n{avoidance_text}\n\n已有或丢弃题目：\n{dedup_section}\n\n只输出单元素JSON数组。"
        return messages

    def parse_response(self, response: dict, kp_id: str = "", subject_id: str = "") -> list:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return []
        result = self._parse_json_response(content)
        if result:
            return result
        text = content
        if "```" in text:
            blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
            if blocks:
                text = blocks[0]
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start:end])
                if isinstance(obj, dict) and obj.get("question_text"):
                    return [obj]
            except json.JSONDecodeError:
                pass
        return []

    @staticmethod
    def assemble_final_image_prompt(image_prompt: str) -> str:
        """Append the frozen style suffix before calling gpt-image-2."""
        image_prompt = (image_prompt or "").strip()
        if not image_prompt:
            return _STYLE_SUFFIX
        return f"{image_prompt}\n{_STYLE_SUFFIX}"

    def shutdown(self):
        self._executor.shutdown(wait=True)


def test_generate():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "test-key")
    sample_kp = {
        "subject_id": "MATH",
        "subject_name": "高等数学",
        "kp_id": "MATH-CALC-001",
        "kp_name": "多元函数微分学-偏导数与全微分",
    }
    generator = QuestionGenerator(base_url=base_url, api_key=api_key, model="gpt-5.5", max_concurrent=2)
    try:
        questions = generator.generate_batch(sample_kp, batch_size=2)
        print(f"Generated {len(questions)} questions")
        for q in questions:
            print(f"ID: {q['question_id']} | Answer: {q['correct_answer']} | Prompt words: {generator._word_count(q.get('image_prompt', ''))}")
        return questions
    finally:
        generator.shutdown()


if __name__ == "__main__":
    test_generate()
