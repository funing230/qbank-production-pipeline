# GPT 图片提示词（image_prompt）生成规范

> **状态：FROZEN (2026-06-21)**
> 未经爸爸明确批准不得修改本规范。

## 概述

GPT 生成题目时，不再输出 render_code（Python绘图代码），而是输出结构化英文 `image_prompt` 字段。
该 prompt 由 gpt-image-2 直接渲染成图片。

## 1. 语言

英文为主体，中文标注直接写在 prompt 里（gpt-image-2 能渲染中文）。

## 2. 长度

100–160 词（不含系统自动追加的风格后缀）。

## 3. 结构要素（按顺序）

GPT 输出的 image_prompt 必须按以下顺序包含全部要素：

| 序号 | 要素 | 说明 | 示例 |
|------|------|------|------|
| ① | 图类型声明 | 开头第一句说明图的类型 | "A directed graph diagram showing..." |
| ② | 全局布局 | 视角、区域划分、方向 | "viewed from above, split into left and right regions" |
| ③ | 具体元素 | 节点/组件/结构名称、数量、位置 | "7 nodes arranged in 3 levels: top level has nodes A, B; middle level has..." |
| ④ | 连接关系 | 边/线/箭头，逐条写明谁连谁 | "A→C, A→D, B→D, B→E, C→F, D→F, E→F" |
| ⑤ | 标注要求 | 每个需标注的元素写明 labeled "xxx" | '(labeled "起始节点")' |
| ⑥ | 视觉风格收尾 | 配色、线条粗细、背景色 | "black edges, blue nodes, white background" |

## 4. 精确性原则

- 元素数量必须明确数字（"4 chromosomes"、"7 nodes"），**禁止** "several"、"some"、"a few"
- 空间位置必须明确（"at opposite poles"、"bottom level"、"upper-right"）
- 连接关系必须逐条列出，不能含糊

## 5. 禁止事项

| 禁止 | 原因 |
|------|------|
| 模糊描述（"some lines connecting them"） | gpt-image-2 会随机画 |
| 装饰性要求（"make it beautiful"、"artistic style"） | 浪费词数，与学术风格冲突 |
| 超过 8 个主要元素 | 元素过多导致图拥挤/标注不清 |
| 写风格后缀（GPT不需要写） | 系统自动追加，GPT写了就重复 |

## 6. 统一风格后缀

**由 pipeline 代码自动追加**，GPT 不需要写。

后缀内容（frozen）：

```
Clean academic textbook illustration style, white background, thin black outlines, no shading, no 3D perspective, no drop shadows, no gradients, high contrast, clearly legible labels.
```

## 7. 拼接流程

```
GPT 输出 image_prompt（100-160词，纯内容描述）
          ↓
pipeline 拼接：final_prompt = image_prompt + "\n" + STYLE_SUFFIX
          ↓
调用 gpt-image-2 API：generate_image(prompt=final_prompt)
```

## 8. 验证规则（代码侧）

pipeline 在拼接前应校验：
1. `image_prompt` 字段存在且非空
2. 词数在 50–200 范围内（允许少量容差）
3. 不包含风格后缀中的关键词（避免GPT重复写）
4. 开头第一个词应为图类型相关（A/An + noun，或直接 diagram/chart/graph 等）
