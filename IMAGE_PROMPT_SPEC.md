# GPT 图片提示词（image_prompt）生成规范

> **状态：FROZEN (2026-06-21)**
> 第2步更新：新增 `image_contract`，作为 `image_prompt` 的事实来源。

## 概述

GPT 生成题目时，不再输出 render_code（Python绘图代码），而是输出 `image_contract` 和结构化英文 `image_prompt` 字段。
`image_contract` 是图片事实契约；`image_prompt` 必须由该 contract 派生，再由 gpt-image-2 直接渲染成图片。

## 1. image_contract

每题必须包含 JSON 子对象 `image_contract`。schema 至少包含：

| 字段 | 说明 |
|------|------|
| `version` | contract schema 版本，例如 `"1.0"` |
| `diagram_type` | 图片类型，需与蓝图 image_type 一致 |
| `visual_facts` | 图片中允许出现的普通可见事实 |
| `answer_relevant_facts` | 解题所需、只能从图片获得的关键事实 |
| `labels` | 图片中允许显示的标签 |
| `quantities` | 图片中允许显示或隐含的数值、计量、坐标、权重 |
| `relations` | 连接、空间、因果、流程或对应关系 |
| `constraints` | 布局、元素数量、颜色高亮、医学合成图等约束 |
| `forbidden_text_leakage` | question_text/options 不得完整泄露的答案相关短语 |

`question_text` 和 `options` 不得完整泄露 `forbidden_text_leakage` 或 `answer_relevant_facts` 中的条目。

## 2. image_prompt 与 image_contract 的关系

`image_prompt` 是英文为主体的图片生成提示词，不是代码，不是 JSON 子对象。
它必须从 `image_contract` 派生：只能重写、组织和视觉化 contract 中已有的 `diagram_type`、`visual_facts`、`answer_relevant_facts`、`labels`、`quantities`、`relations`、`constraints`。

禁止在 `image_prompt` 中引入 contract 外的新解题事实，包括新数值、新标签、新关系、新高亮、新答案线索。

## 3. 语言

英文为主体，中文标注直接写在 prompt 里（gpt-image-2 能渲染中文）。

## 4. 长度

100–160 词（不含系统自动追加的风格后缀）。代码侧允许 50–200 词容差。

## 5. 结构要素（按顺序）

GPT 输出的 image_prompt 必须按以下顺序包含全部要素：

| 序号 | 要素 | 说明 | 示例 |
|------|------|------|------|
| ① | 图类型声明 | 开头第一句说明图的类型 | "A directed graph diagram showing..." |
| ② | 全局布局 | 视角、区域划分、方向 | "viewed from above, split into left and right regions" |
| ③ | 具体元素 | 节点/组件/结构名称、数量、位置 | "7 nodes arranged in 3 levels: top level has nodes A, B; middle level has..." |
| ④ | 连接关系 | 边/线/箭头，逐条写明谁连谁 | "A→C, A→D, B→D, B→E, C→F, D→F, E→F" |
| ⑤ | 标注要求 | 每个需标注的元素写明 labeled "xxx" | '(labeled "起始节点")' |
| ⑥ | 视觉风格收尾 | 配色、线条粗细、背景色 | "black edges, blue nodes, white background" |

## 6. 医学科目限制

S07-S12 医学科目必须使用 synthetic/simulated 教学图，包括但不限于：

- synthetic educational medical diagrams
- simulated ECG
- simulated pathology-style diagram
- anatomy teaching diagram
- lab chart
- pharmacokinetic curve

禁止声称或暗示真实 patient image、真实病人照片、真实影像或真实病例图。

## 7. 精确性原则

- 元素数量必须明确数字（"4 chromosomes"、"7 nodes"），禁止 "several"、"some"、"a few"
- 空间位置必须明确（"at opposite poles"、"bottom level"、"upper-right"）
- 连接关系必须逐条列出，不能含糊
- 主要元素不超过 8 个；超过时必须简化为核心可判题结构

## 8. 禁止事项

| 禁止 | 原因 |
|------|------|
| 模糊描述（"some lines connecting them"） | gpt-image-2 会随机画 |
| 装饰性要求（"make it beautiful"、"artistic style"） | 浪费词数，与学术风格冲突 |
| 超过 8 个主要元素 | 元素过多导致图拥挤/标注不清 |
| 写风格后缀（GPT不需要写） | 系统自动追加，GPT写了就重复 |
| 在 prompt 中加入 contract 外的新事实 | 破坏题图一致性和可审计性 |

## 9. 统一风格后缀

**由 pipeline 代码自动追加**，GPT 不需要写。

后缀内容（frozen）：

```
Clean academic textbook illustration style, white background, thin black outlines, no shading, no 3D perspective, no drop shadows, no gradients, high contrast, clearly legible labels.
```

## 10. 拼接流程

```
GPT 输出 image_contract
          ↓
GPT 根据 image_contract 输出 image_prompt（100-160词，纯内容描述）
          ↓
pipeline 校验 contract、泄露、prompt 格式
          ↓
pipeline 拼接：final_prompt = image_prompt + "\n" + STYLE_SUFFIX
          ↓
调用 gpt-image-2 API：generate_image(prompt=final_prompt)
```

## 11. 验证规则（代码侧）

pipeline 在拼接前应校验：
1. `image_contract` 字段存在且包含必需 schema key
2. `question_text/options` 不完整泄露 `forbidden_text_leakage` 或 `answer_relevant_facts`
3. `image_prompt` 字段存在且非空
4. `image_prompt` 词数在 50–200 范围内（允许少量容差）
5. `image_prompt` 不包含风格后缀中的关键词（避免GPT重复写）
6. `image_prompt` 开头第一个词应为图类型相关（A/An + noun，或直接 diagram/chart/graph 等）
7. 旧渲染字段 `render_code/render_instruction/render_engine/render_params/image_description` 不得出现
