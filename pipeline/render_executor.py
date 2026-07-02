"""
渲染执行器 v2.0（adjustText + 强化排版规则版）

统一的render_code执行入口：
- 自动注入字体配置
- adjustText 自动推开重叠文字（"整理工"）
- 强制最小字号、最小图片尺寸
- 强制 tight_layout + bbox_inches='tight'
- 图例自动移到图外防止遮挡数据
"""
import subprocess
import tempfile
import os
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# 渲染代码头部模板（强制注入）
RENDER_HEADER = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# adjustText：自动推开重叠文字
try:
    from adjustText import adjust_text as _adjust_text
    _HAS_ADJUST_TEXT = True
except ImportError:
    _HAS_ADJUST_TEXT = False

# 强制字体配置
rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 150
rcParams['figure.figsize'] = [10, 8]

# 强化排版规则
rcParams['figure.autolayout'] = False  # 我们手动控制
rcParams['axes.titlepad'] = 15
rcParams['axes.labelpad'] = 10
rcParams['font.size'] = 12  # 全局最小字号基准

output_path = r"{output_path}"
'''

# 渲染代码尾部模板（adjustText整理工 + 强化排版 + 保存）
RENDER_FOOTER = '''

# ========== adjustText整理工 + 强化排版规则（自动注入） ==========
import sys

def _layout_enforcer():
    """渲染后自动排版整理（整理工）"""
    fig = plt.gcf()
    all_axes = fig.get_axes()
    
    if not all_axes:
        return
    
    for ax in all_axes:
        # === 1. adjustText：自动推开所有重叠文字 ===
        if _HAS_ADJUST_TEXT:
            # 收集所有 ax.text 对象
            texts = [t for t in ax.texts if t and t.get_text().strip()]
            if len(texts) >= 2:
                try:
                    _adjust_text(texts, ax=ax, 
                                 force_text=(0.5, 0.5),
                                 force_static=(0.3, 0.3),
                                 expand=(1.2, 1.4),
                                 ensure_inside_axes=True,
                                 arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
                except Exception:
                    pass  # adjustText内部错误不影响渲染
            
            # 也处理 annotate 生成的文字
            annots = [child for child in ax.get_children()
                      if hasattr(child, 'get_text') and hasattr(child, 'xyann')
                      and child.get_text().strip()]
            if len(annots) >= 2:
                try:
                    _adjust_text(annots, ax=ax,
                                 force_text=(0.3, 0.3),
                                 ensure_inside_axes=True)
                except Exception:
                    pass
        
        # === 2. 强制最小字号：所有文字不低于9pt ===
        for t in ax.texts:
            if t and t.get_fontsize() < 9:
                t.set_fontsize(9)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            if label.get_fontsize() < 8:
                label.set_fontsize(8)
        
        # === 3. 图例防遮挡：移到最佳位置 ===
        legend = ax.get_legend()
        if legend:
            try:
                legend.set_loc('best')
            except Exception:
                pass
        
        # === 4. 长x轴标签自动旋转 ===
        try:
            for label in ax.get_xticklabels():
                if len(label.get_text()) > 12:
                    label.set_rotation(30)
                    label.set_ha('right')
        except Exception:
            pass
    
    # === 5. 全局布局调整 ===
    try:
        plt.tight_layout(pad=2.0)
    except Exception:
        try:
            plt.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.15)
        except Exception:
            pass

try:
    _layout_enforcer()
except Exception as e:
    sys.stderr.write(f"Layout enforcer warning: {e}\\n")

# 保存
try:
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3, facecolor='white')
    plt.close('all')
except Exception as e:
    # 备用保存（不带bbox_inches）
    try:
        plt.savefig(output_path, dpi=150, facecolor='white')
        plt.close('all')
    except:
        sys.stderr.write(f"Save failed: {e}\\n")
        sys.exit(1)
'''


def render_question_image(
    render_code: str,
    output_path: str,
    render_engine: str = "MATPLOTLIB",
    timeout: int = 30,
    workdir: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    执行render_code并生成图片。
    
    自动注入：
    - 字体配置
    - 防遮挡后处理
    - tight_layout + bbox_inches='tight'
    
    Args:
        render_code: GPT生成的渲染代码
        output_path: 图片输出路径
        render_engine: 引擎名（当前只支持MATPLOTLIB系）
        timeout: 超时秒数
        workdir: 工作目录
        
    Returns:
        (success: bool, error_msg: str)
    """
    # 预处理render_code：移除GPT可能自己加的savefig/show
    import re
    # 删除GPT自己写的plt.savefig行（我们会在footer中统一处理）
    render_code_clean = re.sub(
        r'plt\.savefig\([^)]*\)\s*', 
        '# [removed: savefig handled by renderer]\n', 
        render_code
    )
    render_code_clean = re.sub(
        r'plt\.show\(\)\s*', 
        '# [removed: show not needed]\n', 
        render_code_clean
    )
    # 删除GPT自己写的tight_layout（防止重复调用）
    render_code_clean = re.sub(
        r'plt\.tight_layout\([^)]*\)\s*',
        '# [removed: tight_layout handled by renderer]\n',
        render_code_clean
    )
    
    # 组装完整脚本
    full_code = RENDER_HEADER.format(output_path=output_path) + "\n" + render_code_clean + "\n" + RENDER_FOOTER
    
    # 写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(full_code)
        script_path = f.name
    
    try:
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir or str(Path(output_path).parent),
        )
        
        if result.returncode != 0:
            error_msg = result.stderr[:500] if result.stderr else "Unknown render error"
            logger.warning(f"Render failed: {error_msg}")
            return False, error_msg
        
        if not Path(output_path).exists():
            return False, "Output file not created"
        
        # 检查文件大小（空图片通常<1KB）
        size = Path(output_path).stat().st_size
        if size < 1024:
            return False, f"Image too small ({size} bytes), likely blank"
        
        return True, ""
        
    except subprocess.TimeoutExpired:
        return False, f"Render timeout ({timeout}s)"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.unlink(script_path)
        except:
            pass


def generate_placeholder_image(output_path: str, message: str = "渲染失败\n需人工检查"):
    """生成占位图"""
    code = f'''
fig, ax = plt.subplots(figsize=(10, 8))
ax.text(0.5, 0.5, '{message}', ha='center', va='center', fontsize=24,
        fontfamily='WenQuanYi Zen Hei', color='red',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', edgecolor='red'))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_facecolor('#f8f8f8')
'''
    success, err = render_question_image(code, output_path)
    if not success:
        logger.error(f"Even placeholder failed: {err}")
