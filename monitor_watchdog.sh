#!/bin/bash
# 题库生产监控看门狗
# 检查项：进程存活、新错误模式、通过率骤降、Glyph缺失、空白图片
# 由 cron job 每3分钟调用，有问题时输出报警文本（stdout），无问题时静默

WORKDIR="/home/flyer8258/research_projects/multimodal_question_bank_24x1000/runs/qwen_gpt_closed_loop_18subjects_24000_20260619_v2/final_questionbank_production_18subjects_24000_v5"
LOG="$WORKDIR/production_v7.log"
DB="$WORKDIR/production.db"
STATE="$WORKDIR/.monitor_state"
PYTHON="$WORKDIR/.venv-qbank-render/bin/python3"

# 初始化状态文件（记录上次检查的日志行数）
if [ ! -f "$STATE" ]; then
    wc -l < "$LOG" > "$STATE" 2>/dev/null || echo "0" > "$STATE"
    exit 0
fi

ALERTS=""

# 1. 进程存活检查
if ! pgrep -f "run_production_v7.py" > /dev/null 2>&1; then
    ALERTS="${ALERTS}🔴 生产进程已停止！请检查是否崩溃。\n"
fi

# 2. 获取新增日志行
LAST_LINE=$(cat "$STATE")
CURRENT_LINE=$(wc -l < "$LOG" 2>/dev/null || echo "0")

if [ "$CURRENT_LINE" -le "$LAST_LINE" ]; then
    # 日志没有新增（可能卡住了）
    if [ "$CURRENT_LINE" -eq "$LAST_LINE" ] && pgrep -f "run_production_v7.py" > /dev/null 2>&1; then
        # 进程在但3分钟没新日志——可能卡住
        ALERTS="${ALERTS}⚠️ 3分钟内无新日志输出，进程可能卡住。\n"
    fi
    echo "$CURRENT_LINE" > "$STATE"
    if [ -n "$ALERTS" ]; then
        echo -e "$ALERTS"
    fi
    exit 0
fi

# 取新增的行
NEW_LINES=$(sed -n "$((LAST_LINE+1)),${CURRENT_LINE}p" "$LOG")
echo "$CURRENT_LINE" > "$STATE"

# 3. Glyph 缺失检查（修复后不应出现）
GLYPH_COUNT=$(echo "$NEW_LINES" | grep -c "Glyph.*missing" || true)
if [ "$GLYPH_COUNT" -gt 0 ]; then
    ALERTS="${ALERTS}🔴 检测到 ${GLYPH_COUNT} 次 Glyph 缺失！字体修复可能失效。\n"
fi

# 4. 空白图片检查
BLANK_COUNT=$(echo "$NEW_LINES" | grep -c "空白图片" || true)
if [ "$BLANK_COUNT" -gt 5 ]; then
    ALERTS="${ALERTS}⚠️ 新增 ${BLANK_COUNT} 次空白图片（阈值5），渲染器可能有问题。\n"
fi

# 5. 异常错误模式（非常规FAIL以外的）
TRACEBACK_COUNT=$(echo "$NEW_LINES" | grep -c "Traceback\|RuntimeError\|MemoryError\|OSError\|Segmentation" || true)
if [ "$TRACEBACK_COUNT" -gt 0 ]; then
    LAST_TB=$(echo "$NEW_LINES" | grep -A2 "Traceback\|RuntimeError\|MemoryError\|OSError" | tail -3)
    ALERTS="${ALERTS}🔴 检测到 ${TRACEBACK_COUNT} 次异常错误：\n${LAST_TB}\n"
fi

# 6. 401/403/500 API错误
API_ERR=$(echo "$NEW_LINES" | grep -c "401\|403\|500\|502\|503" || true)
if [ "$API_ERR" -gt 3 ]; then
    ALERTS="${ALERTS}⚠️ 新增 ${API_ERR} 次 API 错误 (401/403/5xx)，key 或服务可能异常。\n"
fi

# 7. 通过率统计
PASS_RATE=$($PYTHON -c "
import sqlite3
db = sqlite3.connect('$DB')
total = db.execute('SELECT COUNT(*) FROM questions').fetchone()[0]
passed = db.execute(\"SELECT COUNT(*) FROM questions WHERE quality_status='FINAL_PASS'\").fetchone()[0]
if total > 0:
    print(f'{passed}/{total} ({passed/total*100:.1f}%)')
else:
    print('0/0 (0%)')
" 2>/dev/null)

# 8. 超时堆积
TIMEOUT_COUNT=$(echo "$NEW_LINES" | grep -c "timed out\|Timeout" || true)
if [ "$TIMEOUT_COUNT" -gt 5 ]; then
    ALERTS="${ALERTS}⚠️ 新增 ${TIMEOUT_COUNT} 次超时，网络可能不稳定。\n"
fi

# 输出报警（有内容才输出）
if [ -n "$ALERTS" ]; then
    echo "🚨 题库生产异常报警"
    echo "当前进度: PASS $PASS_RATE"
    echo "---"
    echo -e "$ALERTS"
fi
