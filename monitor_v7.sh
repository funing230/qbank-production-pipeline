#!/bin/bash
# V7 生产监控脚本 — 被 cron job 调用
# 功能：检查产出进度、每100题抽样3题发飞书、异常报警

cd /home/flyer8258/research_projects/multimodal_question_bank_24x1000/runs/qwen_gpt_closed_loop_18subjects_24000_20260619_v2/final_questionbank_production_18subjects_24000_v5

# 检查生产进程是否还在跑
PROD_PID=$(pgrep -f "run_production_v7.py" 2>/dev/null)
OUTPUT_DIR="./output"

# 统计当前题目数
CURRENT_COUNT=$(find "$OUTPUT_DIR" -name "*.png" 2>/dev/null | wc -l)

# 读取上次计数
STATE_FILE="./monitor_state_simple.txt"
LAST_COUNT=0
if [ -f "$STATE_FILE" ]; then
    LAST_COUNT=$(cat "$STATE_FILE")
fi

# 计算新增
NEW=$((CURRENT_COUNT - LAST_COUNT))

# 检查是否需要抽样（每100题）
LAST_MILESTONE=$((LAST_COUNT / 100))
CURR_MILESTONE=$((CURRENT_COUNT / 100))

echo "当前: ${CURRENT_COUNT}题, 上次: ${LAST_COUNT}题, 新增: ${NEW}题, PID: ${PROD_PID:-无}"

# 异常检测
ALERT=""

# 1. 进程是否存活
if [ -z "$PROD_PID" ] && [ "$CURRENT_COUNT" -lt 24000 ]; then
    if [ "$CURRENT_COUNT" -gt 0 ]; then
        ALERT="🚨 V7生产进程已退出！当前仅完成 ${CURRENT_COUNT}/24000 题。请检查日志。"
    fi
fi

# 2. 渲染失败检测（小于5KB的PNG）
if [ "$CURRENT_COUNT" -gt 50 ]; then
    FAILURES=$(find "$OUTPUT_DIR" -name "*.png" -size -5k 2>/dev/null | wc -l)
    RATE=$((FAILURES * 100 / CURRENT_COUNT))
    if [ "$RATE" -gt 20 ]; then
        ALERT="🚨 渲染失败率过高: ${FAILURES}/${CURRENT_COUNT} = ${RATE}%"
    fi
fi

# 3. 长时间无进度
if [ "$NEW" -eq 0 ] && [ "$CURRENT_COUNT" -gt 0 ] && [ "$CURRENT_COUNT" -lt 24000 ]; then
    if [ -n "$PROD_PID" ]; then
        ALERT="⚠️ 生产进程在运行但15分钟无新增题目。当前: ${CURRENT_COUNT}/24000"
    fi
fi

# 输出报警（如果有）
if [ -n "$ALERT" ]; then
    echo "$ALERT"
fi

# 抽样逻辑（达到新的100整数倍）
SAMPLE_MSG=""
if [ "$CURR_MILESTONE" -gt "$LAST_MILESTONE" ] && [ "$CURRENT_COUNT" -ge 100 ]; then
    # Python 抽样
    SAMPLE_MSG=$(python3 -c "
import json, random
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
pngs = list(output_dir.rglob('*.png'))
if len(pngs) < 3:
    exit(0)

samples = random.sample(pngs, 3)
lines = ['🔍 V7抽样审核 (已完成 ${CURRENT_COUNT}/24000 题)', '随机抽3题：', '']

for i, png in enumerate(samples, 1):
    qid = png.stem
    subj_dir = png.parent.parent
    # 找JSON
    qdata = None
    for jf in subj_dir.glob('*.json'):
        try:
            data = json.loads(jf.read_text())
            if isinstance(data, list):
                for q in data:
                    if q.get('question_id') == qid:
                        qdata = q; break
            elif isinstance(data, dict) and data.get('question_id') == qid:
                qdata = data
        except: pass
        if qdata: break
    
    if qdata:
        lines.append(f'【{i}】{qid}')
        lines.append(f'  科目: {qdata.get(\"subject_name\",\"?\")}')
        lines.append(f'  知识点: {qdata.get(\"knowledge_point_name\",\"?\")}')
        lines.append(f'  题目: {qdata.get(\"question_text\",\"\")[:80]}...')
        lines.append(f'  答案: {qdata.get(\"correct_answer\",\"?\")}')
        lines.append(f'  图片: {png.stat().st_size//1024}KB')
    else:
        lines.append(f'【{i}】{qid} ({subj_dir.name}) {png.stat().st_size//1024}KB')
    lines.append('')

# 输出图片路径（供发送）
lines.append('---IMAGES---')
for png in samples:
    lines.append(str(png.resolve()))

print('\n'.join(lines))
" 2>/dev/null)
    echo "$SAMPLE_MSG"
fi

# 更新状态
echo "$CURRENT_COUNT" > "$STATE_FILE"

# 构建最终输出供 cron agent 使用
if [ -n "$ALERT" ] || [ -n "$SAMPLE_MSG" ]; then
    if [ -n "$ALERT" ]; then
        echo "ALERT:$ALERT"
    fi
    if [ -n "$SAMPLE_MSG" ]; then
        echo "SAMPLE:$SAMPLE_MSG"
    fi
else
    # 静默 — 没有达到里程碑也没有报警
    if [ "$CURRENT_COUNT" -eq 0 ]; then
        echo "V7生产尚未开始产出（output/为空）"
    fi
fi
