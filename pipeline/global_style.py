"""
全局风格表 (Global Style Sheet)
================================
所有渲染函数共用的视觉参数。
保证 18 科目的图片风格统一（像同一套教材出的配图）。

修改这里 = 修改所有图片的外观。
"""

# ==================== 画布 ====================
FIGSIZE = (10, 8)           # 统一画布尺寸（英寸）
DPI = 150                   # 统一分辨率
BACKGROUND = "white"        # 统一白底

# ==================== 字体 ====================
FONT_FAMILY = "DejaVu Sans"  # 主字体（英文+数学）
FONT_FAMILY_CJK = "SimHei"   # 中文字体（备选 WenQuanYi Micro Hei）
LABEL_FONTSIZE = 14           # 数据标签字号
TITLE_FONTSIZE = 0            # 图内不放标题（标题在题干里）
LEGEND_FONTSIZE = 12          # 图例字号
AXIS_LABEL_FONTSIZE = 13      # 坐标轴标签字号
TICK_FONTSIZE = 11            # 刻度数字字号
MIN_FONTSIZE = 10             # 最小允许字号（低于此值强制提升）

# ==================== 颜色方案 ====================
# 学术四色（对色盲友好的配色，来自 matplotlib tab10 + 微调）
PRIMARY_COLORS = [
    "#4A90D9",  # 蓝色（主色）
    "#E85D5D",  # 红色
    "#5CB85C",  # 绿色
    "#F0AD4E",  # 橙色
]

# 扩展色板（需要更多颜色时使用）
EXTENDED_COLORS = [
    "#4A90D9", "#E85D5D", "#5CB85C", "#F0AD4E",
    "#9B59B6", "#1ABC9C", "#E67E22", "#34495E",
]

# 功能色
HIGHLIGHT_COLOR = "#AED6F1"     # 高亮填充（浅蓝）
HIGHLIGHT_COLOR_ALT = "#FADBD8" # 高亮填充备选（浅红）
LINE_COLOR = "#2C3E50"          # 主线条颜色（深灰蓝）
EDGE_COLOR = "#555555"          # 边框/边线颜色
GRID_COLOR = "#CCCCCC"          # 网格线颜色
NODE_COLOR = "#AED6F1"          # 图论节点默认填充
NODE_EDGE_COLOR = "#2C3E50"     # 图论节点边框

# ==================== 线条 ====================
LINE_WIDTH = 1.8            # 主线宽
THIN_LINE_WIDTH = 1.0       # 细线宽（辅助线、网格）
THICK_LINE_WIDTH = 2.5      # 粗线宽（强调线）
ARROW_WIDTH = 1.5           # 箭头线宽
BORDER_WIDTH = 2.0          # 图形边框线宽

# ==================== 布局 ====================
PADDING_LEFT = 0.12         # 左边距比例
PADDING_RIGHT = 0.88        # 右边界比例
PADDING_TOP = 0.92          # 上边界比例
PADDING_BOTTOM = 0.10       # 下边距比例
TIGHT_PAD = 2.0             # tight_layout padding

# ==================== 节点/顶点 ====================
NODE_SIZE = 600             # networkx 节点大小
NODE_FONT_SIZE = 12         # 节点内文字
EDGE_FONT_SIZE = 10         # 边标签文字

# ==================== Venn 图专用 ====================
VENN_ALPHA = 0.4            # Venn 圆填充透明度
VENN_FONTSIZE = 15          # Venn 区域数字字号
VENN_LABEL_FONTSIZE = 16    # Venn 集合标签字号

# ==================== 电路图专用 ====================
CIRCUIT_SCALE = 1.5         # schemdraw 缩放比例
CIRCUIT_FONTSIZE = 13       # 电路标注字号

# ==================== 分子图专用 ====================
MOLECULE_IMG_SIZE = (600, 500)  # RDKit 图片尺寸（像素）

# ==================== 地图专用 ====================
MAP_FIGSIZE = (12, 8)       # 地图稍大一些

# ==================== matplotlib rcParams 设置 ====================
def apply_global_style():
    """在渲染前调用此函数，统一 matplotlib 全局设置"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams['figure.figsize'] = FIGSIZE
    rcParams['figure.dpi'] = DPI
    rcParams['figure.facecolor'] = BACKGROUND
    rcParams['axes.facecolor'] = BACKGROUND

    # 字体
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', FONT_FAMILY_CJK, FONT_FAMILY, 'Arial', 'Helvetica']
    rcParams['font.size'] = LABEL_FONTSIZE
    rcParams['axes.labelsize'] = AXIS_LABEL_FONTSIZE
    rcParams['xtick.labelsize'] = TICK_FONTSIZE
    rcParams['ytick.labelsize'] = TICK_FONTSIZE
    rcParams['legend.fontsize'] = LEGEND_FONTSIZE

    # 线条
    rcParams['lines.linewidth'] = LINE_WIDTH
    rcParams['axes.linewidth'] = THIN_LINE_WIDTH
    rcParams['grid.linewidth'] = THIN_LINE_WIDTH * 0.6

    # 颜色
    from cycler import cycler
    rcParams['axes.prop_cycle'] = cycler(color=EXTENDED_COLORS)
    rcParams['grid.color'] = GRID_COLOR
    rcParams['grid.alpha'] = 0.5

    # 布局
    rcParams['figure.constrained_layout.use'] = False  # 我们手动控制

    # 学术风格微调
    rcParams['axes.spines.top'] = True
    rcParams['axes.spines.right'] = True
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['xtick.major.size'] = 4
    rcParams['ytick.major.size'] = 4

    return plt


# ====== LaTeX → Unicode 转换（共性方案）======
# 所有引擎的文本标签在渲染前统一调用此函数，
# 确保图片上显示正规数学/物理/化学符号，不显示 LaTeX 源码。

import re as _re

_LATEX_GREEK = {
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε',
    'zeta': 'ζ', 'eta': 'η', 'theta': 'θ', 'iota': 'ι', 'kappa': 'κ',
    'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ', 'pi': 'π',
    'rho': 'ρ', 'sigma': 'σ', 'tau': 'τ', 'upsilon': 'υ', 'phi': 'φ',
    'chi': 'χ', 'psi': 'ψ', 'omega': 'ω',
    'Alpha': 'Α', 'Beta': 'Β', 'Gamma': 'Γ', 'Delta': 'Δ', 'Epsilon': 'Ε',
    'Zeta': 'Ζ', 'Eta': 'Η', 'Theta': 'Θ', 'Iota': 'Ι', 'Kappa': 'Κ',
    'Lambda': 'Λ', 'Mu': 'Μ', 'Nu': 'Ν', 'Xi': 'Ξ', 'Pi': 'Π',
    'Rho': 'Ρ', 'Sigma': 'Σ', 'Tau': 'Τ', 'Upsilon': 'Υ', 'Phi': 'Φ',
    'Chi': 'Χ', 'Psi': 'Ψ', 'Omega': 'Ω',
    'infty': '∞', 'partial': '∂', 'nabla': '∇', 'hbar': 'ℏ',
    'ell': 'ℓ', 'forall': '∀', 'exists': '∃', 'emptyset': '∅',
    'pm': '±', 'mp': '∓', 'times': '×', 'div': '÷', 'cdot': '·',
    'leq': '≤', 'geq': '≥', 'neq': '≠', 'approx': '≈', 'equiv': '≡',
    'sim': '∼', 'propto': '∝',
    'leftarrow': '←', 'rightarrow': '→', 'leftrightarrow': '↔',
    'Leftarrow': '⇐', 'Rightarrow': '⇒', 'Leftrightarrow': '⇔',
    'uparrow': '↑', 'downarrow': '↓',
    'sum': '∑', 'prod': '∏', 'int': '∫',
    'sqrt': '√', 'angle': '∠', 'triangle': '△', 'degree': '°',
    'circ': '°', 'perp': '⊥', 'parallel': '∥',
}

_LATEX_ACCENTS = {
    'vec': '→',     # 用箭头后缀替代 combining character（字体兼容性更好）
    'hat': '^',     # 简化表示
    'bar': '‾',     # overline
    'dot': '·',     # 点
    'ddot': '··',   # 双点
    'tilde': '~',   # 波浪
}

_SUPERSCRIPTS = str.maketrans('0123456789n', '⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ')
_SUBSCRIPTS = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')


def _convert_super(match):
    """^{content} → Unicode superscript（仅限 WenQuanYi 支持的字符）"""
    content = match.group(1) if match.group(1) else match.group(0)[1:]
    # WenQuanYi 支持: ¹²³⁴ⁿ，不支持: ⁰⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁱ
    _SAFE_SUPER = str.maketrans('1234n', '¹²³⁴ⁿ')
    result = content.translate(_SAFE_SUPER)
    return result


def _convert_sub(match):
    """_{content} → Unicode subscript（仅限 WenQuanYi 支持的字符）"""
    content = match.group(1) if match.group(1) else match.group(0)[1:]
    # WenQuanYi 支持: ₁₂₃₄，不支持: ₀₅₆₇₈₉₊₋₌₍₎ₐₑₒₓ
    _SAFE_SUB = str.maketrans('1234', '₁₂₃₄')
    result = content.translate(_SAFE_SUB)
    return result


def latex_to_unicode(text: str) -> str:
    """
    将 LaTeX 数学标记转换为 Unicode 字符显示。
    
    适用于所有渲染引擎的图片标签文本。
    JSON 存储中保留 LaTeX 格式，仅在渲染到图片时调用此函数。
    
    Examples:
        "$\\vec{B}$ 向里"  → "B⃗ 向里"
        "$\\alpha = 0.05$" → "α = 0.05"
        "$\\Delta T$"      → "ΔT"
        "$x^2 + y^2$"     → "x² + y²"
        "$V_1$"           → "V₁"
    """
    if not text or '$' not in text:
        return text

    def _process_math(m):
        """处理 $...$ 内部的数学内容"""
        content = m.group(1)

        # 处理 \mathrm{...}, \text{...}, \textrm{...} → 直接取内容
        content = _re.sub(r'\\(?:mathrm|text|textrm|textbf|mathbf)\{([^}]*)\}', r'\1', content)

        # 处理 \vec{X} → X⃗
        for accent, combining in _LATEX_ACCENTS.items():
            content = _re.sub(
                rf'\\{accent}\{{([^}}]+)\}}',
                lambda mm: mm.group(1) + combining,
                content
            )
            # 也处理 \vec X (无花括号，单字符)
            content = _re.sub(
                rf'\\{accent}\s+([A-Za-z])',
                lambda mm: mm.group(1) + combining,
                content
            )

        # 处理希腊字母 \alpha → α
        for latex_name, unicode_char in _LATEX_GREEK.items():
            content = content.replace(f'\\{latex_name}', unicode_char)

        # 处理 \frac{a}{b} → a/b
        content = _re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1/\2', content)

        # 处理上标 ^{...}
        content = _re.sub(r'\^\{([^}]+)\}', _convert_super, content)
        # 单字符上标 ^2
        content = _re.sub(r'\^([0-9n+-])', _convert_super, content)

        # 处理下标 _{...}
        content = _re.sub(r'_\{([^}]+)\}', _convert_sub, content)
        # 单字符下标 _1
        content = _re.sub(r'_([0-9a-z])', _convert_sub, content)

        # 处理 \, (thin space) → 空格
        content = content.replace('\\,', ' ')
        # 处理 \; \: \! (各种空格)
        content = _re.sub(r'\\[;:!]', ' ', content)
        # 处理 \quad \qquad
        content = content.replace('\\quad', '  ').replace('\\qquad', '    ')

        # 处理 \left \right (只是定界符标记，去掉)
        content = content.replace('\\left', '').replace('\\right', '')

        # 去掉剩余的反斜杠命令（如果还有未处理的）
        content = _re.sub(r'\\([A-Za-z]+)', r'\1', content)

        # 清理多余花括号
        content = content.replace('{', '').replace('}', '')

        return content.strip()

    # 匹配 $...$ (非贪婪)
    result = _re.sub(r'\$([^$]+)\$', _process_math, text)
    return result


def latex_sanitize_data(data: dict) -> dict:
    """
    递归遍历 render_instruction 的 data 字典，
    将所有字符串值中的 LaTeX 公式转为 Unicode。
    
    这是共性方案：在 render_router 调用任何引擎之前执行一次，
    确保所有引擎的图片上只显示正规公式符号。
    """
    if isinstance(data, str):
        return latex_to_unicode(data)
    elif isinstance(data, list):
        return [latex_sanitize_data(item) for item in data]
    elif isinstance(data, dict):
        return {latex_sanitize_data(k): latex_sanitize_data(v) for k, v in data.items()}
    else:
        return data


def finalize_figure(fig, output_path, pad=2.0):
    """
    统一的图片保存出口 — 所有渲染器最终都应调用此函数。
    
    执行顺序:
    1. adjustText 自动避让所有文字标签（无引导线，纯位移）
    2. tight_layout
    3. savefig
    4. close
    
    这是解决标签重叠的共性方案 — 不需要每个渲染器单独处理。
    """
    import matplotlib.pyplot as plt
    import matplotlib.text as mpl_text

    # 对所有 axes 执行 adjustText
    try:
        from adjustText import adjust_text
        for ax in fig.get_axes():
            # 收集所有非空、非轴标签、非标题的文本对象
            texts = []
            for child in ax.get_children():
                if not isinstance(child, mpl_text.Text):
                    continue
                t = child.get_text()
                if not t or not t.strip():
                    continue
                # 排除坐标轴标签和标题（它们不需要避让）
                if child == ax.title or child == ax.xaxis.label or child == ax.yaxis.label:
                    continue
                # 排除刻度标签
                if child in ax.get_xticklabels() or child in ax.get_yticklabels():
                    continue
                texts.append(child)
            
            if len(texts) > 1:
                # 只做位移，不画引导线（避免莫名其妙的黑色小线段）
                adjust_text(texts, ax=ax)
    except ImportError:
        pass
    except Exception:
        pass  # adjustText 失败不阻塞渲染

    plt.figure(fig.number)
    plt.tight_layout(pad=pad)
    plt.savefig(output_path, dpi=DPI, facecolor=BACKGROUND, edgecolor='none')
    plt.close(fig)

