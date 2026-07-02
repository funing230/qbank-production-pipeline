#!/usr/bin/env python3
from __future__ import annotations
import json, os, re, sqlite3, zlib
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = ROOT / 'final_submission'
VDB = ROOT / 'pilot_240_qwen36_verified/verified_pilot.db'
ADB = ROOT / 'pilot_240_qwen36/qwen36_reaudit/qwen36_reaudit.db'
OUT_MD = FINAL_DIR / '审核报告_Qwen36_337题_修正版_CJKv2.md'
OUT_PDF = FINAL_DIR / '审核报告_Qwen36_337题_修正版_CJKv2.pdf'

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
    subject_total = {sid:n for sid,n in cur.execute("select subject_id,count(*) from questions group by subject_id order by subject_id").fetchall()}

    final_ids = {q for (q,) in cur.execute("select question_id from questions where quality_status='FINAL_PASS'")}
    images = cur.execute("select count(*) from questions where quality_status='FINAL_PASS' and coalesce(image_path,'')!=''").fetchone()[0]
    missing_images = len(final_ids) - images

    reaudit = {q:(w,c,v) for q,w,c,v in acur.execute("select question_id,wrong_count,correct_count,valid_count from qwen36_audit where decision='PASS' and wrong_count>=3 and valid_count=5")}
    reaudit_final = set(reaudit).intersection(final_ids)

    prod_detail = {}
    for q,d in cur.execute("select question_id,detail from audit_log where detail like 'qwen36_rollout_pass%' order by id"):
        if q in final_ids:
            prod_detail[q]=d
    prod_final = {q:d for q,d in prod_detail.items() if q not in reaudit_final}

    prod_wrong = Counter();
    for q,d in prod_final.items():
        m=re.search(r'wrong=(\d+) correct=(\d+) answers=([^\s]+)', d or '')
        if not m: continue
        prod_wrong[int(m.group(1))]+=1

    reaudit_wrong=Counter();
    for q in reaudit_final:
        w,c,v=reaudit[q]; reaudit_wrong[w]+=1

    final_wrong = Counter(reaudit_wrong)
    for k,v in prod_wrong.items(): final_wrong[k]+=v

    # difficulty distribution from question_json
    difficulty=Counter();
    for qj,st in cur.execute('select question_json,quality_status from questions'):
        try: obj=json.loads(qj or '{}')
        except Exception: obj={}
        val=obj.get('difficulty') or obj.get('difficulty_level') or obj.get('metadata',{}).get('difficulty')
        try:
            val=int(str(val).strip())
        except Exception:
            continue
        if st=='FINAL_PASS': difficulty[val]+=1

    # subject qwen metrics counts (not required for CJK but keep light)
    con.close(); adb.close()

    return dict(total_rows=total_rows,status=status,subject_final=subject_final,subject_total=subject_total,
                final_wrong=final_wrong,final_ids=final_ids,images=images,missing_images=missing_images,
                reaudit_final=reaudit_final,prod_final=prod_final,reaudit_wrong=reaudit_wrong,prod_wrong=prod_wrong,
                difficulty=difficulty)


# ---- PDF via PIL->PNG pages (robust CJK) ----
A4W,A4H=2480,3508
MARGIN=120
BLUE=(30,60,120); DARK=(35,35,35); GRAY=(245,246,248); GREEN=(35,130,75); ORANGE=(220,135,45); RED=(180,60,60)
FONT_PATHS=[
    '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
]
FONT_PATH=next((p for p in FONT_PATHS if Path(p).exists()), None)

def get_font(size:int):
    if FONT_PATH:
        return ImageFont.truetype(FONT_PATH, size)
    return ImageFont.load_default()


def wrap(draw,text,max_width,font):
    lines=[]
    if not text: return ['']
    for para in str(text).split('\n'):
        line=''
        for ch in para:
            bb=draw.textbbox((0,0), line+ch, font=font)
            if bb[2] <= max_width:
                line+=ch
            else:
                if line: lines.append(line)
                line=ch
        if line or para=='':
            lines.append(line)
    return lines


def header(draw,title):
    draw.rectangle([0,0,A4W,170], fill=BLUE)
    draw.text((MARGIN,55), title, font=get_font(54), fill='white')


def make_page(title, body_lines):
    img=Image.new('RGB',(A4W,A4H),'white')
    d=ImageDraw.Draw(img)
    header(d,title)
    y=260
    f=get_font(38)
    for line in body_lines:
        d.text((MARGIN,y), line, font=f, fill=DARK)
        y+=60
    return img


def images_to_pdf_png(pages, out_path:Path):
    # Use PIL's PNG->PDF embedding by converting each page to RGB and saving.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_pages=[p.convert('RGB') for p in pages]
    first=rgb_pages[0]
    rest=rgb_pages[1:]
    first.save(out_path, 'PDF', resolution=300.0, save_all=True, append_images=rest)


def main():
    m=load_metrics()
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    now=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # markdown (simple)
    md='\\n'.join([
        '# 审核报告_Qwen36_337题（CJK修正版 v2）',
        '',
        f'生成时间：{now}',
        f'最终337题 wrong 分布：w3={m["final_wrong"].get(3,0)}, w4={m["final_wrong"].get(4,0)}, w5={m["final_wrong"].get(5,0)}',
        f'70 再审核导入题数：{len(m["reaudit_final"])}',
        f'267 新生产通过题数：{len(m["prod_final"])}',
        f'missing_images={m["missing_images"]}',
        '',
        '说明：本版采用 PIL 渲染到 PDF（图片页），避免字体编码导致的中文乱码。'
    ])
    OUT_MD.write_text(md, encoding='utf-8')

    # build 5 pages (content condensed, but Chinese safe)
    pages=[]
    pages.append(make_page('试题库质量审核报告（CJK修正版 v2）',[
        'Qwen3.6-Flash Candidate-Student 五轮审核',
        '337 道图文题目 · 24 学科',
        '修正：70 道再审核导入 + 267 道新生产通过 = 337',
        f'最终 wrong 分布：w3={m["final_wrong"].get(3,0)} w4={m["final_wrong"].get(4,0)} w5={m["final_wrong"].get(5,0)}'
    ]))
    pages.append(make_page('一、审核方法论',[
        'Candidate-Student：模型独立作答并与标准答案逐项比对',
        '通过门槛：wrong_count ≥ 3 → PASS；correct_count ≥ 3 → FAIL/REGEN',
        '新生产链路存在提前终止，因此不能写死 5 次/题的调用数口径'
    ]))
    # subject page minimal
    subj_lines=[]
    for i in range(1,25):
        sid=f'S{i:02d}'
        subj_lines.append(f'{sid} {SUBJECT_NAMES[sid]}：{m["subject_final"].get(sid,0)}')
    # split into 2 pages worth
    pages.append(make_page('二、24 学科通过情况（FINAL_PASS）', subj_lines[:12]))
    pages.append(make_page('三、24 学科通过情况（续）', subj_lines[12:]))
    # pipeline & distributions
    diff_lines=[]
    for d in [1,2,3,4,5]:
        diff_lines.append(f'难度 {d}：{m["difficulty"].get(d,0)}')
    pages.append(make_page('四、流水线与分布（关键）',[
        '最终错误分布：w3={}'.format(m['final_wrong'].get(3,0)),
        '最终错误分布：w4={}'.format(m['final_wrong'].get(4,0)),
        '最终错误分布：w5={}'.format(m['final_wrong'].get(5,0)),
        '难度分布（FINAL_PASS）:',
        *diff_lines,
        'missing_images='+str(m['missing_images'])
    ]))

    images_to_pdf_png(pages, OUT_PDF)
    print('WROTE_MD', OUT_MD)
    print('WROTE_PDF', OUT_PDF)
    print('SIZE', OUT_PDF.stat().st_size)


if __name__=='__main__':
    main()
