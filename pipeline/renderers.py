"""
V5 渲染引擎模块 - 7核心引擎 + 4辅助引擎
每个引擎执行GPT生成的绘图代码，输出PNG图片。
"""
import os
import time
import tempfile
import subprocess
import hashlib
import threading
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np

# 中文字体路径
CHINESE_FONT = "/mnt/c/Windows/Fonts/simhei.ttf"
DEFAULT_DPI = 150
DEFAULT_TIMEOUT = 30
TIKZ_TIMEOUT = 60


def validate_output(path: str) -> dict:
    """验证渲染输出的PNG图片质量"""
    from PIL import Image
    result = {"valid": False, "issues": []}
    p = Path(path)
    if not p.exists():
        result["issues"].append("file_not_found")
        return result
    size = p.stat().st_size
    result["file_size"] = size
    if size < 500:
        result["issues"].append("file_too_small")
        return result
    try:
        img = Image.open(p)
        img.load()
        w, h = img.size
        result["width"] = w
        result["height"] = h
        if w < 100 or h < 100:
            result["issues"].append("dimensions_too_small")
        arr = np.array(img.convert("RGB"))
        if arr.mean() > 254:
            result["issues"].append("all_white")
        if arr.mean() < 3:
            result["issues"].append("all_black")
    except Exception as e:
        result["issues"].append(f"pillow_error: {str(e)[:100]}")
        return result
    result["valid"] = len(result["issues"]) == 0
    return result


class BaseRenderer(ABC):
    """渲染引擎基类"""
    engine_name: str = "BASE"

    @abstractmethod
    def render(self, code: str, output_path: str, params: dict = None) -> dict:
        """执行渲染，返回 {success, path, error, engine, elapsed_s}"""
        pass

    def _make_result(self, success, path, error, elapsed):
        return {
            "success": success,
            "path": path,
            "error": error,
            "engine": self.engine_name,
            "elapsed_s": round(elapsed, 2),
        }


class MatplotlibRenderer(BaseRenderer):
    """Matplotlib渲染器 - 函数图、统计图、数据可视化"""
    engine_name = "MATPLOTLIB"

    def render(self, code: str, output_path: str, params: dict = None) -> dict:
        t0 = time.time()
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib import font_manager
            # 注册中文字体
            if os.path.exists(CHINESE_FONT):
                font_manager.fontManager.addfont(CHINESE_FONT)
                plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False

            # 在隔离命名空间执行代码
            ns = {"plt": plt, "np": np, "output_path": output_path}
            exec(code, ns)
            # 如果代码没有自行保存，尝试保存当前figure
            if not Path(output_path).exists():
                fig = plt.gcf()
                if fig.get_axes():
                    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
            plt.close('all')

            v = validate_output(output_path)
            if v["valid"]:
                return self._make_result(True, output_path, "", time.time() - t0)
            else:
                return self._make_result(False, output_path, f"validation: {v['issues']}", time.time() - t0)
        except Exception as e:
            plt.close('all')
            return self._make_result(False, "", str(e)[:500], time.time() - t0)


class NetworkXRenderer(BaseRenderer):
    """NetworkX渲染器 - 图论、网络图、树"""
    engine_name = "NETWORKX"

    def render(self, code: str, output_path: str, params: dict = None) -> dict:
        t0 = time.time()
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib import font_manager
            import networkx as nx
            if os.path.exists(CHINESE_FONT):
                font_manager.fontManager.addfont(CHINESE_FONT)
                plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False

            ns = {"plt": plt, "np": np, "nx": nx, "output_path": output_path}
            exec(code, ns)
            if not Path(output_path).exists():
                fig = plt.gcf()
                if fig.get_axes():
                    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
            plt.close('all')

            v = validate_output(output_path)
            if v["valid"]:
                return self._make_result(True, output_path, "", time.time() - t0)
            else:
                return self._make_result(False, output_path, f"validation: {v['issues']}", time.time() - t0)
        except Exception as e:
            plt.close('all')
            return self._make_result(False, "", str(e)[:500], time.time() - t0)


class TikZRenderer(BaseRenderer):
    """TikZ渲染器 - 几何构造、坐标几何、精确图形"""
    engine_name = "TIKZ"

    def render(self, code: str, output_path: str, params: dict = None) -> dict:
        t0 = time.time()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tex_file = Path(tmpdir) / "figure.tex"
                # 如果代码不包含documentclass，包装它
                if "\\documentclass" not in code:
                    code = (
                        "\\documentclass[border=5pt]{standalone}\n"
                        "\\usepackage{tikz}\n"
                        "\\usepackage{amsmath}\n"
                        "\\begin{document}\n"
                        f"{code}\n"
                        "\\end{document}"
                    )
                tex_file.write_text(code)

                r = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, str(tex_file)],
                    capture_output=True, text=True, timeout=TIKZ_TIMEOUT
                )
                pdf_file = Path(tmpdir) / "figure.pdf"
                if not pdf_file.exists():
                    return self._make_result(False, "", f"pdflatex failed: {r.stderr[-500:]}", time.time() - t0)

                # PDF → PNG
                out_stem = Path(output_path).with_suffix('')
                subprocess.run(
                    ["pdftoppm", "-png", "-r", str(DEFAULT_DPI), "-singlefile", str(pdf_file), str(out_stem)],
                    capture_output=True, timeout=15
                )
                # pdftoppm输出 stem.png
                actual_out = Path(str(out_stem) + ".png")
                if actual_out.exists() and str(actual_out) != output_path:
                    actual_out.rename(output_path)

            v = validate_output(output_path)
            if v["valid"]:
                return self._make_result(True, output_path, "", time.time() - t0)
            else:
                return self._make_result(False, output_path, f"validation: {v['issues']}", time.time() - t0)
        except subprocess.TimeoutExpired:
            return self._make_result(False, "", "timeout", time.time() - t0)
        except Exception as e:
            return self._make_result(False, "", str(e)[:500], time.time() - t0)


class CircuitikZRenderer(BaseRenderer):
    """CircuitikZ渲染器 - 电路图"""
    engine_name = "CIRCUITIKZ"

    def render(self, code: str, output_path: str, params: dict = None) -> dict:
        t0 = time.time()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tex_file = Path(tmpdir) / "circuit.tex"
                if "\\documentclass" not in code:
                    code = (
                        "\\documentclass[border=5pt]{standalone}\n"
                        "\\usepackage{circuitikz}\n"
                        "\\begin{document}\n"
                        f"{code}\n"
                        "\\end{document}"
                    )
                tex_file.write_text(code)

                r = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, str(tex_file)],
                    capture_output=True, text=True, timeout=TIKZ_TIMEOUT
                )
                pdf_file = Path(tmpdir) / "circuit.pdf"
                if not pdf_file.exists():
                    return self._make_result(False, "", f"pdflatex failed: {r.stderr[-500:]}", time.time() - t0)

                out_stem = Path(output_path).with_suffix('')
                subprocess.run(
                    ["pdftoppm", "-png", "-r", str(DEFAULT_DPI), "-singlefile", str(pdf_file), str(out_stem)],
                    capture_output=True, timeout=15
                )
                actual_out = Path(str(out_stem) + ".png")
                if actual_out.exists() and str(actual_out) != output_path:
                    actual_out.rename(output_path)

            v = validate_output(output_path)
            if v["valid"]:
                return self._make_result(True, output_path, "", time.time() - t0)
            else:
                return self._make_result(False, output_path, f"validation: {v['issues']}", time.time() - t0)
        except subprocess.TimeoutExpired:
            return self._make_result(False, "", "timeout", time.time() - t0)
        except Exception as e:
            return self._make_result(False, "", str(e)[:500], time.time() - t0)


class SchemdrawRenderer(BaseRenderer):
    """Schemdraw渲染器 - 简单电路原理图"""
    engine_name = "SCHEMDRAW"

    def render(self, code: str, output_path: str, params: dict = None) -> dict:
        t0 = time.time()
        try:
            import matplotlib
            matplotlib.use('Agg')
            import schemdraw
            import schemdraw.elements as elm

            ns = {"schemdraw": schemdraw, "elm": elm, "output_path": output_path}
            exec(code, ns)

            v = validate_output(output_path)
            if v["valid"]:
                return self._make_result(True, output_path, "", time.time() - t0)
            else:
                return self._make_result(False, output_path, f"validation: {v['issues']}", time.time() - t0)
        except Exception as e:
            return self._make_result(False, "", str(e)[:500], time.time() - t0)


class PillowRenderer(BaseRenderer):
    """Pillow渲染器 - 复合图、表格、文字密集型图"""
    engine_name = "PILLOW"

    def render(self, code: str, output_path: str, params: dict = None) -> dict:
        t0 = time.time()
        try:
            from PIL import Image, ImageDraw, ImageFont

            font = None
            if os.path.exists(CHINESE_FONT):
                font = ImageFont.truetype(CHINESE_FONT, 16)

            ns = {
                "Image": Image, "ImageDraw": ImageDraw, "ImageFont": ImageFont,
                "output_path": output_path, "CHINESE_FONT": CHINESE_FONT, "font": font,
                "np": np,
            }
            exec(code, ns)

            v = validate_output(output_path)
            if v["valid"]:
                return self._make_result(True, output_path, "", time.time() - t0)
            else:
                return self._make_result(False, output_path, f"validation: {v['issues']}", time.time() - t0)
        except Exception as e:
            return self._make_result(False, "", str(e)[:500], time.time() - t0)


class PlotlyRenderer(BaseRenderer):
    """Plotly渲染器 - 3D图、等高线、热力图"""
    engine_name = "PLOTLY"

    def render(self, code: str, output_path: str, params: dict = None) -> dict:
        t0 = time.time()
        try:
            import plotly.graph_objects as go
            import plotly.express as px

            ns = {"go": go, "px": px, "np": np, "output_path": output_path}
            exec(code, ns)
            # 如果代码没有自行write_image，检查是否有fig变量
            if not Path(output_path).exists() and "fig" in ns:
                ns["fig"].write_image(output_path, width=800, height=600)

            v = validate_output(output_path)
            if v["valid"]:
                return self._make_result(True, output_path, "", time.time() - t0)
            else:
                return self._make_result(False, output_path, f"validation: {v['issues']}", time.time() - t0)
        except Exception as e:
            return self._make_result(False, "", str(e)[:500], time.time() - t0)


class SympyRenderer(BaseRenderer):
    """SymPy渲染器 - 符号数学图形"""
    engine_name = "SYMPY"

    def render(self, code: str, output_path: str, params: dict = None) -> dict:
        t0 = time.time()
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import sympy as sp
            from sympy import symbols, diff, integrate, solve, lambdify

            if os.path.exists(CHINESE_FONT):
                from matplotlib import font_manager
                font_manager.fontManager.addfont(CHINESE_FONT)
                plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False

            ns = {
                "plt": plt, "np": np, "sp": sp, "symbols": symbols,
                "diff": diff, "integrate": integrate, "solve": solve,
                "lambdify": lambdify, "output_path": output_path,
            }
            exec(code, ns)
            if not Path(output_path).exists():
                fig = plt.gcf()
                if fig.get_axes():
                    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
            plt.close('all')

            v = validate_output(output_path)
            if v["valid"]:
                return self._make_result(True, output_path, "", time.time() - t0)
            else:
                return self._make_result(False, output_path, f"validation: {v['issues']}", time.time() - t0)
        except Exception as e:
            plt.close('all')
            return self._make_result(False, "", str(e)[:500], time.time() - t0)


class ShapelyRenderer(BaseRenderer):
    """Shapely渲染器 - 几何构造"""
    engine_name = "SHAPELY"

    def render(self, code: str, output_path: str, params: dict = None) -> dict:
        t0 = time.time()
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from shapely.geometry import Point, Polygon, LineString, MultiPoint, MultiPolygon
            from shapely.ops import unary_union

            if os.path.exists(CHINESE_FONT):
                from matplotlib import font_manager
                font_manager.fontManager.addfont(CHINESE_FONT)
                plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False

            ns = {
                "plt": plt, "np": np, "Point": Point, "Polygon": Polygon,
                "LineString": LineString, "MultiPoint": MultiPoint,
                "MultiPolygon": MultiPolygon, "unary_union": unary_union,
                "output_path": output_path,
            }
            exec(code, ns)
            if not Path(output_path).exists():
                fig = plt.gcf()
                if fig.get_axes():
                    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
            plt.close('all')

            v = validate_output(output_path)
            if v["valid"]:
                return self._make_result(True, output_path, "", time.time() - t0)
            else:
                return self._make_result(False, output_path, f"validation: {v['issues']}", time.time() - t0)
        except Exception as e:
            plt.close('all')
            return self._make_result(False, "", str(e)[:500], time.time() - t0)


class RDKitRenderer(BaseRenderer):
    """RDKit渲染器 - 化学结构式"""
    engine_name = "RDKIT"

    def render(self, code: str, output_path: str, params: dict = None) -> dict:
        t0 = time.time()
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw, AllChem

            ns = {
                "Chem": Chem, "Draw": Draw, "AllChem": AllChem,
                "output_path": output_path,
            }
            exec(code, ns)

            v = validate_output(output_path)
            if v["valid"]:
                return self._make_result(True, output_path, "", time.time() - t0)
            else:
                return self._make_result(False, output_path, f"validation: {v['issues']}", time.time() - t0)
        except Exception as e:
            return self._make_result(False, "", str(e)[:500], time.time() - t0)


class GeoPandasRenderer(BaseRenderer):
    """GeoPandas渲染器 - 地图"""
    engine_name = "GEOPANDAS"

    def render(self, code: str, output_path: str, params: dict = None) -> dict:
        t0 = time.time()
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import geopandas as gpd
            from shapely.geometry import Point, Polygon

            if os.path.exists(CHINESE_FONT):
                from matplotlib import font_manager
                font_manager.fontManager.addfont(CHINESE_FONT)
                plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False

            ns = {
                "plt": plt, "np": np, "gpd": gpd,
                "Point": Point, "Polygon": Polygon,
                "output_path": output_path,
            }
            exec(code, ns)
            if not Path(output_path).exists():
                fig = plt.gcf()
                if fig.get_axes():
                    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
            plt.close('all')

            v = validate_output(output_path)
            if v["valid"]:
                return self._make_result(True, output_path, "", time.time() - t0)
            else:
                return self._make_result(False, output_path, f"validation: {v['issues']}", time.time() - t0)
        except Exception as e:
            plt.close('all')
            return self._make_result(False, "", str(e)[:500], time.time() - t0)


# ========== 调度器 ==========

class RenderDispatcher:
    """渲染调度器 - 根据引擎名称分发渲染任务"""

    def __init__(self):
        self._engines = {
            "MATPLOTLIB": MatplotlibRenderer(),
            "NETWORKX": NetworkXRenderer(),
            "TIKZ": TikZRenderer(),
            "CIRCUITIKZ": CircuitikZRenderer(),
            "SCHEMDRAW": SchemdrawRenderer(),
            "PILLOW": PillowRenderer(),
            "PLOTLY": PlotlyRenderer(),
            "SYMPY": SympyRenderer(),
            "SHAPELY": ShapelyRenderer(),
            "RDKIT": RDKitRenderer(),
            "GEOPANDAS": GeoPandasRenderer(),
        }
        self._lock = threading.Lock()

    def dispatch(self, engine_name: str, code: str, output_path: str, params: dict = None) -> dict:
        """分发渲染任务到对应引擎"""
        engine_name = engine_name.upper()
        if engine_name not in self._engines:
            return {
                "success": False, "path": "", "engine": engine_name,
                "error": f"Unknown engine: {engine_name}. Available: {list(self._engines.keys())}",
                "elapsed_s": 0,
            }
        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        renderer = self._engines[engine_name]
        return renderer.render(code, output_path, params)

    def get_available_engines(self) -> list:
        return list(self._engines.keys())

    def health_check(self) -> dict:
        """对每个引擎做最小化健康检查"""
        results = {}
        for name, renderer in self._engines.items():
            with tempfile.TemporaryDirectory() as tmpdir:
                out = os.path.join(tmpdir, "health.png")
                # 最简单的测试代码
                test_codes = {
                    "MATPLOTLIB": "fig,ax=plt.subplots()\nax.fill([0,1,1,0],[0,0,1,1],color='blue')\nax.set_title('Test')\nfig.savefig(output_path,dpi=72)",
                    "NETWORKX": "G=nx.complete_graph(5)\nfig,ax=plt.subplots()\nnx.draw(G,ax=ax,node_color='red',node_size=300)\nfig.savefig(output_path,dpi=72)",
                    "TIKZ": "\\begin{tikzpicture}\n\\fill[blue] (0,0) rectangle (2,2);\n\\draw[thick,red] (0,0) -- (2,2);\n\\end{tikzpicture}",
                    "CIRCUITIKZ": "\\begin{circuitikz}\n\\draw (0,0) to[R,l=$R_1$] (3,0) to[C,l=$C_1$] (3,-2) -- (0,-2) to[V,v=$V$] (0,0);\n\\end{circuitikz}",
                    "SCHEMDRAW": "import schemdraw\nimport schemdraw.elements as elm\nwith schemdraw.Drawing(show=False) as d:\n    d += elm.Resistor().label('R1')\n    d += elm.Capacitor().down().label('C1')\nd.save(output_path,dpi=72)",
                    "PILLOW": "from PIL import Image,ImageDraw\nimg=Image.new('RGB',(200,200),'white')\ndraw=ImageDraw.Draw(img)\ndraw.rectangle([10,10,190,190],fill='blue',outline='black')\ndraw.ellipse([30,30,170,170],fill='red')\nimg.save(output_path)",
                    "PLOTLY": "import plotly.graph_objects as go\nfig=go.Figure(go.Bar(x=['A','B','C'],y=[3,7,2]))\nfig.write_image(output_path,width=400,height=300)",
                    "SYMPY": "import numpy as np\nx=np.linspace(0,6.28,100)\nfig,ax=plt.subplots()\nax.fill_between(x,np.sin(x),alpha=0.5)\nax.set_title('sin')\nfig.savefig(output_path,dpi=72)",
                    "SHAPELY": "from shapely.geometry import Point\nfig,ax=plt.subplots()\np=Point(0,0).buffer(1)\nx,y=p.exterior.xy\nax.fill(x,y,color='green')\nax.set_aspect('equal')\nfig.savefig(output_path,dpi=72)",
                    "RDKIT": "from rdkit import Chem\nfrom rdkit.Chem import Draw\nmol=Chem.MolFromSmiles('c1ccccc1')\nimg=Draw.MolToImage(mol,size=(300,300))\nimg.save(output_path)",
                    "GEOPANDAS": "import geopandas as gpd\nfrom shapely.geometry import Polygon\ngdf=gpd.GeoDataFrame(geometry=[Polygon([(0,0),(2,0),(2,2),(0,2)])],crs='EPSG:4326')\nfig,ax=plt.subplots()\ngdf.plot(ax=ax,color='orange',edgecolor='black')\nfig.savefig(output_path,dpi=72)",
                }
                code = test_codes.get(name, "")
                r = renderer.render(code, out)
                results[name] = r["success"]
        return results


if __name__ == "__main__":
    print("=== Renderer Module Self-Test ===")
    dispatcher = RenderDispatcher()
    print(f"Available engines: {dispatcher.get_available_engines()}")
    print("\nHealth check...")
    health = dispatcher.health_check()
    for eng, ok in health.items():
        print(f"  {'✅' if ok else '❌'} {eng}")
    print(f"\n{sum(health.values())}/{len(health)} engines healthy")
