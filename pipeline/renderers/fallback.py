"""
保底渲染函数 (Fallback)
=======================
处理两种情况：
1. 旧格式：GPT 直接给出 render_code（完整 matplotlib 代码）
2. 新格式但 engine 未知：走通用 matplotlib 代码执行

对于旧格式代码，强制注入排版规则（居中、最小字号、固定画布），
并在子进程中隔离执行。
"""

import sys
import os
import subprocess
import tempfile
import textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.global_style import (
    FIGSIZE, DPI, BACKGROUND, MIN_FONTSIZE, LABEL_FONTSIZE, apply_global_style
)


# 注入到 GPT 代码前面的头部（强制排版标准）
INJECT_HEADER = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ===== 强制排版标准 =====
rcParams['figure.figsize'] = {figsize}
rcParams['figure.dpi'] = {dpi}
rcParams['figure.facecolor'] = 'white'
rcParams['font.size'] = {fontsize}
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Droid Sans Fallback', 'DejaVu Sans', 'SimHei', 'Arial']
rcParams['axes.unicode_minus'] = False

OUTPUT_PATH = r"{output_path}"
'''

# 注入到 GPT 代码后面的尾部（强制保存 + 居中）
INJECT_FOOTER = '''

# ===== 强制排版后处理 =====
import sys
try:
    fig = plt.gcf()
    all_axes = fig.get_axes()
    
    # 强制最小字号
    for ax in all_axes:
        for txt in ax.texts:
            if txt.get_fontsize() < {min_fontsize}:
                txt.set_fontsize({min_fontsize})
    
    # adjustText 统一避让（与 finalize_figure 同逻辑）
    try:
        from adjustText import adjust_text
        import matplotlib.text as _mpl_text
        for ax in all_axes:
            texts = [child for child in ax.get_children()
                     if isinstance(child, _mpl_text.Text)
                     and child.get_text() and child.get_text().strip()
                     and child.get_position() != (0.5, 1.0)]
            if texts and len(texts) > 2:
                adjust_text(texts, ax=ax)
    except Exception:
        pass
    
    # 强制居中布局
    plt.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.10)
    
    # 保存
    plt.savefig(OUTPUT_PATH, dpi={dpi}, facecolor='white', edgecolor='none',
                bbox_inches=None)  # 不用 bbox_inches='tight' 避免偏移
    plt.close('all')
    sys.exit(0)
except Exception as e:
    print(f"RENDER_ERROR: {{e}}", file=sys.stderr)
    plt.close('all')
    sys.exit(1)
'''


def render_fallback(data: dict, diagram_type: str, output_path: str, style_override: dict = None) -> tuple:
    """保底渲染：在子进程中执行 GPT 生成的代码"""

    code = data.get("code", "")
    if not code:
        return False, "fallback 缺少 code 字段"

    # 组装完整脚本
    header = INJECT_HEADER.format(
        figsize=FIGSIZE,
        dpi=DPI,
        fontsize=LABEL_FONTSIZE,
        output_path=output_path.replace('\\', '\\\\'),
    )

    footer = INJECT_FOOTER.format(
        min_fontsize=MIN_FONTSIZE,
        dpi=DPI,
    )

    full_script = header + "\n" + code + "\n" + footer

    # 写入临时文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                     dir=os.path.dirname(output_path)) as f:
        f.write(full_script)
        script_path = f.name

    try:
        # 子进程执行（隔离环境，避免 matplotlib 状态污染）
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.dirname(output_path),
        )

        if result.returncode != 0:
            error_msg = result.stderr[-500:] if result.stderr else "未知错误"
            return False, f"渲染代码执行失败: {error_msg}"

        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            # 空白检测：fallback代码可能执行成功但什么都没画
            try:
                from PIL import Image
                import numpy as np
                img = Image.open(output_path).convert('RGB')
                arr = np.array(img)
                non_white = np.any(arr < 240, axis=2).sum() / (arr.shape[0] * arr.shape[1])
                if non_white < 0.01 and os.path.getsize(output_path) < 15000:
                    return False, f"fallback渲染空白(非白{non_white*100:.1f}%)"
            except Exception:
                pass  # 检测失败不阻塞
            return True, "保底渲染成功"
        else:
            return False, "渲染后图片不存在或为空"

    except subprocess.TimeoutExpired:
        return False, "渲染超时(30s)"
    except Exception as e:
        return False, f"渲染异常: {str(e)}"
    finally:
        # 清理临时脚本
        try:
            os.unlink(script_path)
        except OSError:
            pass
