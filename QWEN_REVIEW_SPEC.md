# Qwen 审核规范（新版：qwen3.6-flash thinking 五次考生作答）

> **状态：QWEN36 PILOT (2026-06-30)**
> 本规范用于 24科×10题 pilot 与后续 qwen3.6-flash thinking 难度门槛。

## 概述

Qwen 审核不再是“审稿人 PASS/FAIL”模式，而是“考生独立作答”模式。

核心目标：验证题目对 `qwen3.6-flash` thinking 是否足够困难。

判定规则：

```text
每题独立作答 5 次
wrong_count >= 3 → 通过难度门槛，进入 image_queue
correct_count >= 3 → 题目太容易或泄露答案，进入 regen
```

## 固定模型

```text
model: qwen3.6-flash
base_url: https://yuanlansj.xin/v1
enable_thinking: true
rollouts: 5
```

## Qwen 输入白名单

给 Qwen 的 prompt 只能包含：

| 字段 | 说明 |
|---|---|
| `question_text` | 题干 |
| `options` | A/B/C/D 四个选项 |
| `image_contract` / `visual_facts` | 虚拟图片事实，等价于考生看到图片后的视觉信息 |
| `subject` / `kp` | 可选，仅提供学科语境 |

严禁传入：

```text
correct_answer
explanation
truth_spec
image_prompt
final_image_prompt
image_dependency_reason（若包含答案线索）
```

`correct_answer` 只允许由本地程序用于结果比对，不能进入 Qwen prompt。

## image_contract 要求

`image_contract` 是 Qwen 审核用的虚拟图片事实合同，至少包含：

```json
{
  "version": "1.0",
  "diagram_type": "diagram/chart/graph/circuit/synthetic medical diagram/etc",
  "visual_facts": [],
  "answer_relevant_facts": [],
  "labels": [],
  "quantities": {},
  "relations": [],
  "constraints": [],
  "forbidden_text_leakage": []
}
```

其中：

- `visual_facts`：图片中可见事实；
- `answer_relevant_facts`：解题必须依赖的视觉事实；
- `forbidden_text_leakage`：题干和选项不得直接泄露的答案关键事实。

## 单次 rollout 输出格式

Qwen 必须只输出 JSON：

```json
{"answer":"A"}
```

允许本地解析器做 JSON parse 和正则兜底；无法解析的输出记为 technical failure，不计为 wrong。

## 五次结果判定

本地程序统计：

```text
correct_count = Qwen答案与 correct_answer 相同的次数
wrong_count = 有效答案中不等于 correct_answer 的次数
```

| 条件 | 决策 | 后续 |
|---|---|---|
| `wrong_count >= 3` | PASS | `ACCEPTED -> image_queue` |
| `correct_count >= 3` | FAIL | 逻辑 `QUALITY_FAIL`，物理状态先用 `SENTINEL_REGEN -> regen_queue` |
| 有效答案不足 5 | technical failure | 重试 qwen_queue；超限后 HOLD |

第一阶段不修改 SQLite schema，不新增物理 `QUALITY_FAIL` 状态。

## 多线程规则

不要每题内部 5 并发。

推荐：

```text
QWEN_CONCURRENCY = 8
每个 qwen_worker 对一题串行执行 5 次 rollout
全局瞬时 Qwen API 并发约 8
qwen_queue maxsize = 120
```

这样避免：

```text
8 workers × 5 rollouts = 40 并发
```

造成限流、超时和 retry storm。

## 医学图像规则

医学类只使用合成/模拟教学图：

```text
synthetic educational medical diagram
simulated ECG
simulated pathology-style diagram
anatomy teaching diagram
lab chart
pharmacokinetic curve
```

不得声称使用真实患者影像、真实 CT/MRI/X-ray/病理切片或任何含隐私的临床图像。

## 流程位置

```text
GPT 生成题目 + image_contract + image_prompt
          ↓
Qwen3.6-flash thinking 5次作答审核（本规范）
          ↓ wrong_count >= 3
image worker 根据 image_prompt 生成图片
          ↓
像素级 quality_gate
          ↓
FINAL_PASS
```
