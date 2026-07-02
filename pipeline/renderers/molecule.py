"""
分子结构渲染函数
===============
使用 RDKit 画化学分子 2D 结构图。
RDKit 自动计算原子位置、键角，不会重叠。

data 格式:
{
  "smiles": "CCO",                    # SMILES 分子式
  "highlight_atoms": [0, 1],           # 可选：高亮的原子序号
  "highlight_bonds": [0],              # 可选：高亮的键序号
  "show_atom_numbers": false,          # 可选：显示原子编号
  "molecule_name": "乙醇"             # 可选：分子名（不画在图上）
}
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.global_style import (
    DPI, BACKGROUND, MOLECULE_IMG_SIZE, HIGHLIGHT_COLOR, apply_global_style
)


def render_molecule(data: dict, diagram_type: str, output_path: str, style_override: dict = None) -> tuple:
    """渲染分子结构图"""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw, AllChem
    except ImportError:
        return False, "rdkit 未安装"

    smiles = data.get("smiles", "")
    if not smiles:
        return False, "缺少 smiles 字段"

    highlight_atoms = data.get("highlight_atoms", [])
    highlight_bonds = data.get("highlight_bonds", [])
    show_numbers = data.get("show_atom_numbers", False)

    # 解析分子
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, f"无法解析 SMILES: {smiles}"

    # 生成 2D 坐标
    AllChem.Compute2DCoords(mol)

    # 设置高亮颜色
    highlight_atom_colors = {}
    if highlight_atoms:
        for idx in highlight_atoms:
            highlight_atom_colors[idx] = (0.68, 0.85, 0.95)  # 浅蓝

    highlight_bond_colors = {}
    if highlight_bonds:
        for idx in highlight_bonds:
            highlight_bond_colors[idx] = (0.91, 0.36, 0.36)  # 红色

    # 绘制
    img_size = style_override.get("img_size", MOLECULE_IMG_SIZE) if style_override else MOLECULE_IMG_SIZE

    drawer = Draw.MolDraw2DSVG(img_size[0], img_size[1])
    opts = drawer.drawOptions()
    opts.setBackgroundColour((1, 1, 1))
    opts.bondLineWidth = 2.5
    opts.fixedFontSize = 16
    if show_numbers:
        opts.addAtomIndices = True

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms or [],
        highlightAtomColors=highlight_atom_colors or {},
        highlightBonds=highlight_bonds or [],
        highlightBondColors=highlight_bond_colors or {},
    )
    drawer.FinishDrawing()
    svg_text = drawer.GetDrawingText()

    # SVG → PNG
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_text.encode(), write_to=output_path, dpi=DPI)
    except ImportError:
        # fallback: 用 rdkit 直接生成 PNG
        img = Draw.MolToImage(mol, size=img_size,
                              highlightAtoms=highlight_atoms or [])
        img.save(output_path)

    if os.path.exists(output_path):
        return True, "分子结构图渲染成功"
    else:
        return False, "分子图保存失败"
