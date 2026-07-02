# Qwen 审核规范（新版：纯文本逻辑一致性）

> **状态：FROZEN (2026-06-21)**
> 未经爸爸明确批准不得修改本规范。

## 概述

Qwen 审核的对象是 **GPT 按模板输出的一整套文本**，不再涉及渲染代码或图片像素。
审核目标：验证题目文本与 image_prompt 之间的逻辑一致性。

## 审核输入

| 字段 | 来源 | 说明 |
|------|------|------|
| question_text | GPT输出 | 题干 |
| options | GPT输出 | A/B/C/D 四选项 |
| correct_answer | GPT输出 | 正确答案字母 |
| explanation | GPT输出 | 解析过程 |
| image_prompt | GPT输出 | 100-160词图片描述 |
| image_dependency_reason | GPT输出 | 为什么必须看图 |
| knowledge_point | 知识点库 | 当前知识点名称 |
| scope_boundary | 知识点库 | 知识点精确范围 |

## 审核标准（6项检查）

### 1. 答案正确性
- explanation 的推导过程是否逻辑正确
- correct_answer 是否能从 explanation 唯一推出
- 计算/推理过程有无错误

### 2. 选项合理性
- 四选项是否互不重复
- 干扰项是否合理（不能一眼排除）
- 是否存在多个正确答案

### 3. image_prompt 与题目一致性
- image_prompt 描述的元素（数量、名称、连接关系）是否与题目答案匹配
- image_prompt 中的标注文字是否与题目引用的标注一致
- 题目说"图中有7个节点"，image_prompt 是否也描述了7个节点
- 答案依赖的关键信息是否体现在 image_prompt 中

### 4. 图片依赖性
- 题目是否真正依赖图片信息才能作答
- 如果删掉图片（只看题干+选项），能否直接选出答案
- image_dependency_reason 是否成立

### 5. 知识点范围
- 题目是否属于指定知识点
- 是否超出 scope_boundary 定义的边界

### 6. image_prompt 规范合规
- 词数是否在 50-200 范围内
- 是否以图类型声明开头
- 元素数量是否明确（无 "several"/"some" 等模糊词）
- 主要元素是否 ≤ 8 个
- 是否包含了风格后缀中的关键词（不应包含）

## 判定规则

### PASS
全部 6 项检查通过。

### FAIL
存在以下任一情况：
- 答案错误或解析推导不出答案
- image_prompt 描述与题目矛盾（元素数/连接/标注不匹配）
- 图片依赖不成立（不看图也能答）
- 超出知识点范围
- 多个正确答案
- image_prompt 严重不合规（词数<30 或完全缺失结构）

## 输出格式

```json
{
  "decision": "PASS|FAIL",
  "checks": {
    "answer_correct": true,
    "options_reasonable": true,
    "prompt_consistent": true,
    "image_dependency_valid": true,
    "scope_correct": true,
    "prompt_compliant": true
  },
  "issues": [],
  "confidence": 0.0
}
```

## 与老版对比

| 项目 | 老版 | 新版 |
|------|------|------|
| 审核对象 | render_code + 题目JSON + 图片base64 | 题目文本 + image_prompt |
| 判定档位 | PASS/REVISE/REJECT | PASS/FAIL |
| 是否需要图片 | 是（多模态） | 否（纯文本） |
| 核心关注 | 代码画的对不对 | 描述和题目对不对 |

## FAIL 后处理

FAIL → GPT 重生成（最多 2 轮，共 3 次机会），与老版一致。

## 拼接流程位置

```
GPT 生成题目（含 image_prompt）
          ↓
Qwen 审核（本规范）← 纯文本，不调图片API
          ↓ PASS
gpt-image-2 渲染图片（拼接 style_suffix）
          ↓
像素级 quality_gate（全白/全黑/过小检测）
          ↓ PASS
入库
```
