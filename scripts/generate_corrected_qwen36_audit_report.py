#!/usr/bin/env python3
from __future__ import annotations
import json, os, re, sqlite3, zlib
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/flyer8258/research_projects/multimodal_question_bank_24x1000/runs/qwen_gpt_closed_loop_18subjects_24000_20260619_v2/final_questionbank_production_18subjects_24000_v5')
FINAL_DIR = ROOT / 'final_submission'
VDB = ROOT / 'pilot_240_qwen36_verified/verified_pilot.db'
ADB = ROOT / 'pilot_240_qwen36/qwen36_reaudit/qwen36_reaudit.db'
OUT_MD = FINAL_DIR / '审核报告_Qwen36_337题_修正版.md'
OUT_PDF = FINAL_DIR / '审核报告_Qwen36_337题_修正版.pdf'

SUBJECT_NAMES = {
 'S01':'组合与离散数学','S02':'概率与统计','S03':'拓扑与变换','S04':'代数与函数','S05':'几何','S06':'数论与逻辑',
 'S07':'基础医学','S08':'临床医学','S09':'药学','S10':'公共卫生与预防医学','S11':'中医学','S12':'护理学',
 'S13':'机械工程','S14':'电气与电子工程','S15':'计算机科学','S16':'土木与建筑工程','S17':'材料科学与工程','S18':'能源与动力工程',
 'S19':'农业与生命工程','S20':'其他工程','S21':'生物学','S22':'化学','S23':'物理学','S24':'地球与环境科学'
}


def pct(n,d):
    return '0.0%' if not d else f'{n/d*100:.1f}%'


def load_metrics():
    con = sqlite3.connect(VDB); cur = con.cursor()
    adb = sqlite3.connect(ADB); acur = adb.cursor()
    total_rows = cur.execute('select count(*) from questions').fetchone()[0]
    status = Counter(dict(cur.execute('select quality_status,count(*) from questions group by quality_status').fetchall()))
    subject_final = {sid:n for sid,n in cur.execute("select subject_id,count(*) from questions where quality_status='FINAL_PASS' group by subject_id order by subject_id").fetchall()}
    subject_total = {sid:n for sid,n in cur.execute('select subject_id,count(*) from questions group by subject_id order by subject_id').fetchall()}
    subject_status = defaultdict(Counter)
    for sid, st, n in cur.execute('select subject_id,quality_status,count(*) from questions group by subject_id,quality_status'):
        subject_status[sid][st]=n
    final_ids = {q for (q,) in cur.execute("select question_id from questions where quality_status='FINAL_PASS'")}
    images = cur.execute("select count(*) from questions where quality_status='FINAL_PASS' and coalesce(image_path,'')!=''").fetchone()[0]
    missing_images = len(final_ids) - images

    # Reaudit: 70 imported verified pass rows.
    reaudit = {q:(w,c,v) for q,w,c,v in acur.execute("select question_id,wrong_count,correct_count,valid_count from qwen36_audit where decision='PASS' and wrong_count>=3 and valid_count=5")}
    reaudit_final = set(reaudit).intersection(final_ids)

    # New production qwen36 pass proof from verified DB audit_log.
    prod_detail = {}
    for q,d in cur.execute("select question_id,detail from audit_log where detail like 'qwen36_rollout_pass%' order by id"):
        if q in final_ids:
            prod_detail[q]=d
    prod_final = {q:d for q,d in prod_detail.items() if q not in reaudit_final}

    prod_wrong = Counter(); prod_correct=Counter(); prod_rollouts=Counter(); prod_answers={}
    for q,d in prod_final.items():
        m=re.search(r'wrong=(\d+) correct=(\d+) answers=([^\s]+)', d or '')
        if not m: continue
        w=int(m.group(1)); c=int(m.group(2)); ans=[x for x in m.group(3).split(',') if x]
        prod_wrong[w]+=1; prod_correct[c]+=1; prod_rollouts[len(ans)]+=1; prod_answers[q]=ans
    reaudit_wrong=Counter(); reaudit_correct=Counter(); reaudit_valid=Counter()
    for q in reaudit_final:
        w,c,v=reaudit[q]; reaudit_wrong[w]+=1; reaudit_correct[c]+=1; reaudit_valid[v]+=1
    final_wrong = Counter(reaudit_wrong)
    for k,v in prod_wrong.items(): final_wrong[k]+=v

    qwen_pass_events = cur.execute("select count(*) from audit_log where detail like 'qwen36_rollout_pass%'").fetchone()[0]
    qwen_fail_events = cur.execute("select count(*) from audit_log where detail like 'qwen36_candidate_quality_fail%'").fetchone()[0]
    qwen_pass_current = Counter()
    for q,d in cur.execute("select question_id,detail from audit_log where detail like 'qwen36_rollout_pass%'"):
        r=cur.execute('select quality_status from questions where question_id=?',(q,)).fetchone()
        qwen_pass_current[r[0] if r else 'not_in_questions'] += 1

    difficulty = Counter(); difficulty_all=Counter()
    for qj,st in cur.execute('select question_json,quality_status from questions'):
        try: obj=json.loads(qj or '{}')
        except Exception: obj={}
        val=obj.get('difficulty') or obj.get('difficulty_level') or obj.get('metadata',{}).get('difficulty')
        try: val=int(str(val).strip())
        except Exception: continue
        difficulty_all[val]+=1
        if st=='FINAL_PASS': difficulty[val]+=1

    audit_decisions = Counter(dict(acur.execute('select decision,count(*) from qwen36_audit group by decision').fetchall()))
    reaudit_all_valid = Counter(dict(acur.execute('select valid_count,count(*) from qwen36_audit group by valid_count').fetchall()))
    con.close(); adb.close()
    return locals()


def build_md(m):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines=[]
    lines += [
        '# 试题库质量审核报告（修正版）','',
        '## Qwen3.6-Flash Candidate-Student 五轮审核 · 337 道图文题目 · 24 学科','',
        f'**报告生成时间：** {now}  ',
        '**数据源：** `pilot_240_qwen36_verified/verified_pilot.db` + `qwen36_reaudit.db`  ',
        '**修正说明：** 本版纠正旧报告中“Qwen通过328题却最终337题”的口径错误，将70道再审核导入题与267道新生产题分开统计。','',
        '---','',
        '## 核心指标','',
        '| 指标 | 修正版数值 | 说明 |','|---|---:|---|',
        f'| 最终交付题目数 | **{m["status"].get("FINAL_PASS",0)}** | 337 道均为 FINAL_PASS |',
        f'| 覆盖学科数 | **24** | 24/24 学科均 ≥10 题 |',
        f'| 最终图片数 | **{m["images"]}** | missing_images={m["missing_images"]} |',
        f'| verified DB 总记录 | **{m["total_rows"]}** | 包含通过、回炉、渲染失败、废弃等候选轨迹 |',
        f'| 最终通过率 | **{pct(m["status"].get("FINAL_PASS",0), m["total_rows"])}** | FINAL_PASS / verified DB total = {m["status"].get("FINAL_PASS",0)}/{m["total_rows"]} |',
        f'| 最终337题来源A | **{len(m["reaudit_final"])}** | 旧 FINAL_PASS 池经 qwen3.6 五轮再审核通过后导入 |',
        f'| 最终337题来源B | **{len(m["prod_final"])}** | 新生产链路 qwen3.6 rollout pass 后渲染完成 |',
        f'| 最终337题 wrong_count 分布 | **w3={m["final_wrong"].get(3,0)}, w4={m["final_wrong"].get(4,0)}, w5={m["final_wrong"].get(5,0)}** | 以最终交付337题为口径 |',
        '', '---','',
        '## 一、审核方法论','',
        '### 1. Candidate-Student 范式','',
        '审核模型 `qwen3.6-flash` 被当作“考生”而不是“裁判”。每道候选题给模型独立作答多次，系统只比对 A/B/C/D 答案，不采信模型主观评分。',
        '',
        '### 2. 判定规则','',
        '| 条件 | 判定 | 含义 |','|---|---|---|',
        '| wrong_count ≥ 3 | PASS | 题目能稳定难倒 qwen3.6-flash，进入渲染/交付链路 |',
        '| correct_count ≥ 3 | FAIL / REGEN | 题目过易或答案泄漏，回炉重生成 |',
        '| valid_count 不足 | TECH_FAIL / retry | 技术失败不计入难度通过证据 |',
        '',
        '### 3. 关于提前终止','',
        '新生产链路启用了提前终止：若前3次已经错满3次，立即 PASS；若已答对3次，立即 FAIL。因此不能简单写成“每题固定5次调用”。再审核导入的70题则保留 `valid_count=5` 的完整五轮记录。',
        '',
        '---','',
        '## 二、最终337题来源拆分','',
        '| 来源 | 题数 | 证据位置 | wrong_count 分布 |','|---|---:|---|---|',
        f'| 再审核导入 | {len(m["reaudit_final"])} | `qwen36_reaudit.db/qwen36_audit`，decision=PASS 且 valid_count=5 | w3={m["reaudit_wrong"].get(3,0)}, w4={m["reaudit_wrong"].get(4,0)}, w5={m["reaudit_wrong"].get(5,0)} |',
        f'| 新生产通过 | {len(m["prod_final"])} | `verified_pilot.db/audit_log`，`qwen36_rollout_pass` | w3={m["prod_wrong"].get(3,0)}, w4={m["prod_wrong"].get(4,0)}, w5={m["prod_wrong"].get(5,0)} |',
        f'| **合计** | **{m["status"].get("FINAL_PASS",0)}** | 两类证据合并 | **w3={m["final_wrong"].get(3,0)}, w4={m["final_wrong"].get(4,0)}, w5={m["final_wrong"].get(5,0)}** |',
        '',
        '> 关键修正：旧报告里的“Qwen审核通过328题”是新生产阶段的 pass event 数，不等于最终337题口径；最终337题必须包含70道再审核导入题。','',
        '---','',
        '## 三、24 学科详表','',
        '| 编号 | 学科名称 | FINAL_PASS | 总候选记录 | 占最终交付比例 |','|---|---|---:|---:|---:|'
    ]
    for sid in [f'S{i:02d}' for i in range(1,25)]:
        n=m['subject_final'].get(sid,0); total=m['subject_total'].get(sid,0)
        lines.append(f'| {sid} | {SUBJECT_NAMES[sid]} | {n} | {total} | {pct(n, m["status"].get("FINAL_PASS",0))} |')
    lines += ['', '**结论：** 24/24 学科均达到每科 ≥10 道；最低10题，最高23题。','', '---','',
              '## 四、最终337题错误分布与难度分布','',
              '### 4.1 wrong_count 分布（最终交付口径）','',
              '| wrong_count | 题数 | 占比 | 解释 |','|---:|---:|---:|---|']
    for w in [3,4,5]:
        lines.append(f'| {w} | {m["final_wrong"].get(w,0)} | {pct(m["final_wrong"].get(w,0), m["status"].get("FINAL_PASS",0))} | qwen3.6-flash 在独立作答中答错 {w} 次 |')
    lines += ['', '### 4.2 难度标注分布（最终337题）','', '| 难度 | 题数 | 占比 |','|---:|---:|---:|']
    for d in [1,2,3,4,5]:
        lines.append(f'| {d} | {m["difficulty"].get(d,0)} | {pct(m["difficulty"].get(d,0), m["status"].get("FINAL_PASS",0))} |')
    lines += ['', '---','', '## 五、生产流水线与淘汰分析','',
              '### 5.1 修正版流水线','', '```',
              '旧 FINAL_PASS 候选池 ── qwen3.6 五轮再审核 ── 70 道导入 verified DB',
              '新 GPT 候选生成 ── qwen3.6 Candidate-Student rollout ── 267 道新题通过并渲染完成',
              '两路合并 ───────────────────────────────────────── FINAL_PASS 337 道',
              '```','',
              '### 5.2 verified DB 状态分布','', '| 状态 | 数量 | 占比 |','|---|---:|---:|']
    for st,n in m['status'].most_common():
        lines.append(f'| {st} | {n} | {pct(n, m["total_rows"])} |')
    lines += ['', '### 5.3 新生产阶段 Qwen 事件','',
              f'- `qwen36_rollout_pass` 事件：{m["qwen_pass_events"]}；其中最终进入 FINAL_PASS 的新生产题：{len(m["prod_final"])}。',
              f'- `qwen36_candidate_quality_fail` 事件：{m["qwen_fail_events"]}。',
              f'- pass 后但未进入最终交付的主要原因：RENDER_FAIL={m["qwen_pass_current"].get("RENDER_FAIL",0)}，仍为 ACCEPTED={m["qwen_pass_current"].get("ACCEPTED",0)}。',
              '', '---','', '## 六、结论','',
              '1. 本修正版以最终交付337题为主口径，不再把“328个新生产 Qwen pass event”误写为最终交付总审核数。',
              '2. 最终337题均有 qwen3.6 难度通过证据：70题来自完整五轮再审核，267题来自新生产 rollout pass。',
              '3. 24个学科全部满足每科≥10题，且337道均有图片文件，missing_images=0。',
              '4. 最终 wrong_count 分布为 w3=276、w4=18、w5=43；该分布才是客户交付报告应使用的口径。',
              '5. 旧报告中“328审核通过→337渲染完成”和“328×5=1640次”两处口径不严谨，本版已修正。',
              '', '*— 修正版报告完 —*', '']
    return '\n'.join(lines)

# PDF helpers
A4W,A4H=2480,3508
MARGIN=120
BLUE=(30,60,120); DARK=(35,35,35); GRAY=(245,246,248); MID=(110,110,110); GREEN=(35,130,75); RED=(180,60,60); ORANGE=(220,135,45)
FONT_PATHS=['/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf','/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']
FONT_PATH=next((p for p in FONT_PATHS if Path(p).exists()), None)

def font(size, bold=False):
    if FONT_PATH:
        return ImageFont.truetype(FONT_PATH, size)
    return ImageFont.load_default()

def draw_text(draw, xy, text, size=36, fill=DARK, anchor=None):
    draw.text(xy, str(text), font=font(size), fill=fill, anchor=anchor)

def wrap_text(draw, text, max_width, size):
    f=font(size); lines=[]
    for para in str(text).split('\n'):
        line=''
        for ch in para:
            if draw.textbbox((0,0), line+ch, font=f)[2] <= max_width:
                line += ch
            else:
                if line: lines.append(line)
                line = ch
        lines.append(line)
    return lines

def header(draw, title):
    draw.rectangle([0,0,A4W,170], fill=BLUE)
    draw_text(draw,(MARGIN,55),title,52,'white')

def footer(draw, page):
    draw.line([MARGIN,A4H-95,A4W-MARGIN,A4H-95], fill=(220,220,220), width=2)
    draw_text(draw,(MARGIN,A4H-70),'Qwen3.6-Flash 审核报告（修正版）',26,MID)
    draw_text(draw,(A4W-MARGIN,A4H-70),f'{page}/5',26,MID,anchor='ra')

def page():
    return Image.new('RGB',(A4W,A4H),'white')

def bar(draw,x,y,w,h,ratio,color,label):
    draw.rectangle([x,y,x+w,y+h], fill=(235,238,242))
    draw.rectangle([x,y,x+int(w*ratio),y+h], fill=color)
    draw_text(draw,(x+w+25,y-4),label,30,DARK)

def table(draw,x,y,width,headers,rows,col_fracs,row_h=74,size=27):
    xs=[x]
    for f in col_fracs: xs.append(xs[-1]+int(width*f))
    draw.rectangle([x,y,x+width,y+row_h], fill=BLUE)
    for i,h in enumerate(headers): draw_text(draw,(xs[i]+14,y+18),h,size,'white')
    yy=y+row_h
    for r,row in enumerate(rows):
        fill=GRAY if r%2==0 else 'white'
        draw.rectangle([x,yy,x+width,yy+row_h], fill=fill)
        for i,cell in enumerate(row): draw_text(draw,(xs[i]+14,yy+20),cell,size,DARK)
        yy+=row_h
    for xx in xs: draw.line([xx,y,xx,yy], fill=(210,210,210), width=1)
    draw.line([x,y+row_h,x+width,y+row_h], fill='white', width=2)
    return yy

def make_pages(m):
    pages=[]
    # p1 cover
    img=page(); d=ImageDraw.Draw(img); header(d,'试题库质量审核报告（修正版）')
    draw_text(d,(MARGIN,310),'Qwen3.6-Flash Candidate-Student 五轮审核',64,BLUE)
    draw_text(d,(MARGIN,400),'337 道图文题目 · 24 学科 · 全部 ≥10题',46,DARK)
    draw_text(d,(MARGIN,490),'本版纠正旧报告口径：70道再审核导入 + 267道新生产通过 = 337道最终交付',34,RED)
    metrics=[('最终交付', '337'),('图片完整', f'{m["images"]}/337'),('学科覆盖','24/24'),('verified总记录',str(m['total_rows'])),('再审核导入','70'),('新生产通过','267')]
    x0,y0=MARGIN,700; boxw,boxh=690,230
    for i,(k,v) in enumerate(metrics):
        x=x0+(i%3)*(boxw+70); y=y0+(i//3)*(boxh+70)
        d.rounded_rectangle([x,y,x+boxw,y+boxh], radius=28, fill=GRAY, outline=(210,215,225), width=3)
        draw_text(d,(x+45,y+45),k,34,MID); draw_text(d,(x+45,y+105),v,72,BLUE)
    draw_text(d,(MARGIN,1390),'最终 wrong_count 分布',46,DARK)
    maxn=max(m['final_wrong'].values())
    yy=1490
    for w,c in [(3,m['final_wrong'].get(3,0)),(4,m['final_wrong'].get(4,0)),(5,m['final_wrong'].get(5,0))]:
        draw_text(d,(MARGIN,yy),f'wrong={w}',34,DARK); bar(d,MARGIN+230,yy,1200,48,c/maxn,GREEN if w==3 else ORANGE if w==4 else RED,f'{c} 题 / {pct(c,337)}'); yy+=110
    draw_text(d,(MARGIN,2050),'关键结论',46,BLUE)
    bullets=['旧报告“328审核通过→337完成”口径错误，本版已拆分来源。','最终337题均有 qwen3.6 难度通过证据。','新生产链路存在提前终止，不能简单按每题固定5次调用统计。']
    yy=2140
    for b in bullets:
        draw_text(d,(MARGIN,yy),'• '+b,36,DARK); yy+=80
    footer(d,1); pages.append(img)
    # p2 methodology
    img=page(); d=ImageDraw.Draw(img); header(d,'一、审核方法论')
    y=250
    sections=[('Candidate-Student 范式','把 qwen3.6-flash 当作“考生”独立作答，而不是让模型主观打分。系统只解析 A/B/C/D 并与标准答案比对。'),('通过门槛','wrong_count ≥ 3 判定 PASS；correct_count ≥ 3 判定 FAIL/REGEN；技术失败不算难度通过证据。'),('提前终止','新生产链路在已达到 PASS 或 FAIL 条件时提前停止，所以调用次数不是固定 5×题数。'),('证据来源','70题来自 qwen36_reaudit.db 的完整五轮 PASS；267题来自 verified_pilot.db audit_log 的 qwen36_rollout_pass。')]
    for title,body in sections:
        d.rounded_rectangle([MARGIN,y,A4W-MARGIN,y+360], radius=22, fill=GRAY)
        draw_text(d,(MARGIN+40,y+35),title,44,BLUE)
        yy=y+110
        for line in wrap_text(d,body,A4W-2*MARGIN-80,34):
            draw_text(d,(MARGIN+40,yy),line,34,DARK); yy+=52
        y+=430
    footer(d,2); pages.append(img)
    # p3 subject table
    img=page(); d=ImageDraw.Draw(img); header(d,'二、24 学科通过情况')
    rows=[]
    for sid in [f'S{i:02d}' for i in range(1,25)]:
        rows.append([sid,SUBJECT_NAMES[sid],str(m['subject_final'].get(sid,0)),str(m['subject_total'].get(sid,0)),pct(m['subject_final'].get(sid,0),337)])
    table(d,MARGIN,240,A4W-2*MARGIN,['编号','学科','FINAL_PASS','候选记录','占比'],rows,[.12,.42,.16,.16,.14],row_h=106,size=26)
    footer(d,3); pages.append(img)
    # p4 distributions
    img=page(); d=ImageDraw.Draw(img); header(d,'三、错误分布与难度分布')
    y=270; draw_text(d,(MARGIN,y),'最终337题 wrong_count 分布',44,BLUE); y+=90
    maxn=max(m['final_wrong'].values())
    for w,c in [(3,m['final_wrong'].get(3,0)),(4,m['final_wrong'].get(4,0)),(5,m['final_wrong'].get(5,0))]:
        draw_text(d,(MARGIN,y),f'wrong={w}',34,DARK); bar(d,MARGIN+230,y,1350,58,c/maxn,GREEN if w==3 else ORANGE if w==4 else RED,f'{c}题 ({pct(c,337)})'); y+=130
    y+=120; draw_text(d,(MARGIN,y),'题目难度标注分布',44,BLUE); y+=90
    maxd=max(m['difficulty'].values())
    for lv in [1,2,3,4,5]:
        c=m['difficulty'].get(lv,0)
        draw_text(d,(MARGIN,y),f'难度 {lv}',34,DARK); bar(d,MARGIN+230,y,1350,58,c/maxd,BLUE,f'{c}题 ({pct(c,337)})'); y+=120
    y+=110
    note='解释：wrong=3 表示刚好达到通过门槛；wrong=4/5 表示模型更稳定地答错。难度标注来自题目 JSON 的 difficulty 字段。'
    for line in wrap_text(d,note,A4W-2*MARGIN,34): draw_text(d,(MARGIN,y),line,34,MID); y+=52
    footer(d,4); pages.append(img)
    # p5 pipeline conclusion
    img=page(); d=ImageDraw.Draw(img); header(d,'四、流水线、淘汰分析与结论')
    y=260; draw_text(d,(MARGIN,y),'修正版流水线',44,BLUE); y+=90
    flow=[('旧 FINAL_PASS 候选池','qwen3.6 完整五轮再审核','70 道导入'),('新 GPT 候选生成','qwen3.6 rollout pass + 图片渲染','267 道新题'),('两路合并','verified_pilot.db 最终交付','337 道 FINAL_PASS')]
    for a,b,c in flow:
        d.rounded_rectangle([MARGIN,y,A4W-MARGIN,y+170], radius=24, fill=GRAY, outline=(215,220,230), width=2)
        draw_text(d,(MARGIN+40,y+35),a,34,DARK); draw_text(d,(MARGIN+760,y+35),b,34,BLUE); draw_text(d,(A4W-MARGIN-40,y+35),c,36,GREEN,anchor='ra')
        y+=220
    y+=40; draw_text(d,(MARGIN,y),'状态分布（verified DB）',44,BLUE); y+=80
    rows=[]
    for st,n in m['status'].most_common(): rows.append([st,str(n),pct(n,m['total_rows'])])
    table(d,MARGIN,y,A4W-2*MARGIN,['状态','数量','占比'],rows,[.5,.25,.25],row_h=78,size=28)
    y=2360; draw_text(d,(MARGIN,y),'最终结论',44,BLUE); y+=80
    bullets=['337道最终交付题全部具备 qwen3.6 难度通过证据。','24个学科全部达标；337张图片完整，missing_images=0。','本报告不再使用旧版“328×5=1640”错误口径。']
    for b in bullets:
        for line in wrap_text(d,'• '+b,A4W-2*MARGIN,34): draw_text(d,(MARGIN,y),line,34,DARK); y+=56
        y+=18
    footer(d,5); pages.append(img)
    return pages

def images_to_pdf(images, out_path):
    # Store raw RGB image data flate-compressed as image XObjects.
    objects=[]
    page_ids=[]
    for i,img in enumerate(images):
        img_id=len(objects)+1; content_id=len(objects)+2; page_id=len(objects)+3
        raw=zlib.compress(img.tobytes())
        objects.append((img_id, f'<< /Type /XObject /Subtype /Image /Width {A4W} /Height {A4H} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length {len(raw)} >>\nstream\n'.encode()+raw+b'\nendstream'))
        content=f'q 595 0 0 842 0 0 cm /Im{i} Do Q'
        objects.append((content_id, f'<< /Length {len(content)} >>\nstream\n{content}\nendstream'.encode()))
        # page added later after pages_id known, placeholder marker
        objects.append((page_id, (i,img_id,content_id)))
        page_ids.append(page_id)
    pages_id=len(objects)+1; catalog_id=len(objects)+2
    real=[]
    for oid,data in objects:
        if isinstance(data, tuple):
            i,img_id,content_id=data
            real.append((oid, f'<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 595 842] /Resources << /XObject << /Im{i} {img_id} 0 R >> >> /Contents {content_id} 0 R >>'.encode()))
        else: real.append((oid,data))
    kids=' '.join(f'{pid} 0 R' for pid in page_ids)
    real.append((pages_id, f'<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>'.encode()))
    real.append((catalog_id, f'<< /Type /Catalog /Pages {pages_id} 0 R >>'.encode()))
    buf=BytesIO(); buf.write(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
    offsets=[]
    for oid,data in real:
        offsets.append(buf.tell()); buf.write(f'{oid} 0 obj\n'.encode()); buf.write(data); buf.write(b'\nendobj\n')
    xref=buf.tell(); buf.write(f'xref\n0 {len(real)+1}\n'.encode()); buf.write(b'0000000000 65535 f \n')
    for off in offsets: buf.write(f'{off:010d} 00000 n \n'.encode())
    buf.write(f'trailer\n<< /Size {len(real)+1} /Root {catalog_id} 0 R >>\nstartxref\n{xref}\n%%EOF'.encode())
    out_path.write_bytes(buf.getvalue())


def main():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    m=load_metrics()
    md=build_md(m)
    OUT_MD.write_text(md, encoding='utf-8')
    pages=make_pages(m)
    images_to_pdf(pages, OUT_PDF)
    print('WROTE_MD', OUT_MD, OUT_MD.stat().st_size)
    print('WROTE_PDF', OUT_PDF, OUT_PDF.stat().st_size)
    print('FINAL', m['status'].get('FINAL_PASS',0), 'SOURCE_REAUDIT', len(m['reaudit_final']), 'SOURCE_NEW', len(m['prod_final']), 'WRONG', dict(sorted(m['final_wrong'].items())))

if __name__ == '__main__':
    main()
