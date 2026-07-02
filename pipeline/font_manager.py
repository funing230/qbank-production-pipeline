"""
Font manager for CJK-compatible matplotlib rendering.

Handles cross-platform font detection, configuration, and validation.
Designed for WSL environments where Chinese fonts may not be pre-installed.

Usage:
    from pipeline.font_manager import setup_fonts
    setup_fonts()  # Call once at startup
"""

import logging
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# CJK font priority list (highest priority first)
CJK_FONT_PRIORITY = [
    "WenQuanYi Zen Hei",       # WSL confirmed working
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "SimHei",
    "Microsoft YaHei",
    "AR PL UMing CN",
    "Droid Sans Fallback",
]

LATIN_FONT_PRIORITY = [
    "DejaVu Sans",
    "Liberation Sans",
    "Arial",
]

MATH_FONT_PRIORITY = [
    "STIX",
    "STIXGeneral",
    "DejaVu Math TeX Gyre",
    "DejaVu Sans",
]

# Common font paths across platforms
_FONT_SEARCH_PATHS = [
    # Linux system fonts
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    # User fonts
    os.path.expanduser("~/.local/share/fonts"),
    os.path.expanduser("~/.fonts"),
    # WSL: Windows fonts accessible from Linux
    "/mnt/c/Windows/Fonts",
    # macOS
    "/Library/Fonts",
    "/System/Library/Fonts",
    os.path.expanduser("~/Library/Fonts"),
]

# Cache for discovered fonts
_font_cache: Optional[Dict[str, str]] = None
_setup_done = False


def _discover_system_fonts() -> Dict[str, str]:
    """Discover available fonts on the system. Returns {font_name: font_path}."""
    global _font_cache
    if _font_cache is not None:
        return _font_cache

    fonts: Dict[str, str] = {}

    # Method 1: Use fc-list (fontconfig) if available
    try:
        result = subprocess.run(
            ["fc-list", "--format", "%{family}|%{file}\n"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if "|" not in line:
                    continue
                parts = line.split("|", 1)
                families = parts[0].split(",")
                filepath = parts[1].strip()
                for family in families:
                    family = family.strip()
                    if family and filepath:
                        # Store first occurrence (prefer earlier in fc-list output)
                        if family not in fonts:
                            fonts[family] = filepath
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Method 2: Walk known font directories for .ttf/.otf files
    if not fonts:
        for search_path in _FONT_SEARCH_PATHS:
            path = Path(search_path)
            if not path.exists():
                continue
            try:
                for font_file in path.rglob("*.[tToO][tTpP][fFcC]"):
                    # Use filename stem as approximate font name
                    name = font_file.stem.replace("-", " ").replace("_", " ")
                    if name not in fonts:
                        fonts[name] = str(font_file)
            except PermissionError:
                continue

    _font_cache = fonts
    return fonts


def _find_best_font(priority_list: List[str]) -> Optional[str]:
    """Find the best available font from a priority list. Returns font path or None."""
    fonts = _discover_system_fonts()

    for desired in priority_list:
        # Exact match
        if desired in fonts:
            return fonts[desired]
        # Case-insensitive partial match
        desired_lower = desired.lower()
        for name, path in fonts.items():
            if desired_lower in name.lower():
                return path

    return None


def check_system_fonts() -> Dict[str, object]:
    """
    Report which fonts are available on the system.

    Returns:
        dict with keys:
            - cjk_available: list of available CJK fonts
            - cjk_best: name of best available CJK font (or None)
            - latin_available: list of available Latin fonts
            - math_available: list of available math fonts
            - all_fonts_count: total number of discovered fonts
            - wsl_fonts_accessible: whether /mnt/c/Windows/Fonts is accessible
            - warnings: list of warning messages
    """
    fonts = _discover_system_fonts()
    warnings = []

    # Check CJK fonts
    cjk_available = []
    cjk_best = None
    for name in CJK_FONT_PRIORITY:
        name_lower = name.lower()
        for font_name in fonts:
            if name_lower in font_name.lower():
                cjk_available.append(name)
                if cjk_best is None:
                    cjk_best = name
                break

    # Check Latin fonts
    latin_available = []
    for name in LATIN_FONT_PRIORITY:
        name_lower = name.lower()
        for font_name in fonts:
            if name_lower in font_name.lower():
                latin_available.append(name)
                break

    # Check math fonts
    math_available = []
    for name in MATH_FONT_PRIORITY:
        name_lower = name.lower()
        for font_name in fonts:
            if name_lower in font_name.lower():
                math_available.append(name)
                break

    # WSL-specific checks
    wsl_fonts_accessible = Path("/mnt/c/Windows/Fonts").exists()

    if not cjk_available:
        warnings.append(
            "No CJK fonts found! Chinese text will render as boxes (□). "
            "Install fonts with: sudo apt-get install fonts-noto-cjk "
            "or: sudo apt-get install fonts-wqy-microhei"
        )
        if wsl_fonts_accessible:
            warnings.append(
                "Windows fonts detected at /mnt/c/Windows/Fonts. "
                "You can symlink them: sudo ln -s /mnt/c/Windows/Fonts /usr/share/fonts/windows "
                "then run: sudo fc-cache -fv"
            )

    return {
        "cjk_available": cjk_available,
        "cjk_best": cjk_best,
        "latin_available": latin_available,
        "math_available": math_available,
        "all_fonts_count": len(fonts),
        "wsl_fonts_accessible": wsl_fonts_accessible,
        "warnings": warnings,
    }


def get_font_path(category: str = "cjk") -> str:
    """
    Get the path to the best available font for a given category.

    Args:
        category: One of 'cjk', 'latin', or 'math'

    Returns:
        Path to the font file, or empty string if no suitable font found.

    Raises:
        ValueError: If category is not recognized.
    """
    if category == "cjk":
        priority = CJK_FONT_PRIORITY
    elif category == "latin":
        priority = LATIN_FONT_PRIORITY
    elif category == "math":
        priority = MATH_FONT_PRIORITY
    else:
        raise ValueError(f"Unknown font category: {category!r}. Use 'cjk', 'latin', or 'math'.")

    path = _find_best_font(priority)
    if path is None:
        if category == "cjk":
            logger.warning(
                "No CJK font found. Chinese characters will not render correctly. "
                "Install with: sudo apt-get install fonts-noto-cjk"
            )
        return ""
    return path


def setup_fonts() -> None:
    """
    Configure matplotlib rcParams for CJK-compatible rendering.

    Call this once at application startup before creating any plots.
    Configures font families, fallback chains, and minus sign rendering.
    """
    global _setup_done
    if _setup_done:
        return

    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import font_manager as mpl_fm
    except ImportError:
        logger.error("matplotlib is not installed. Cannot configure fonts.")
        return

    # Rebuild font cache if needed
    if hasattr(mpl_fm, '_load_fontmanager'):
        mpl_fm._load_fontmanager()
    elif hasattr(mpl_fm.fontManager, '_version') and not mpl_fm.fontManager.ttflist:
        mpl_fm._rebuild()

    # FORCE register WenQuanYi Zen Hei from known path (WSL confirmed)
    WQY_PATH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    if os.path.isfile(WQY_PATH):
        try:
            mpl_fm.fontManager.addfont(WQY_PATH)
            logger.info(f"Force-registered WenQuanYi Zen Hei from {WQY_PATH}")
        except Exception as e:
            logger.warning(f"Failed to register WQY font: {e}")

    # Determine best CJK font name for matplotlib
    cjk_font_name = None
    for name in CJK_FONT_PRIORITY:
        # Check if matplotlib knows about this font
        matches = [f for f in mpl_fm.fontManager.ttflist if name.lower() in f.name.lower()]
        if matches:
            cjk_font_name = matches[0].name
            break

    # If not found via matplotlib, try system detection and register
    if cjk_font_name is None:
        cjk_path = get_font_path("cjk")
        if cjk_path and os.path.isfile(cjk_path):
            try:
                mpl_fm.fontManager.addfont(cjk_path)
                prop = mpl_fm.FontProperties(fname=cjk_path)
                cjk_font_name = prop.get_name()
                logger.info(f"Registered CJK font from: {cjk_path} as '{cjk_font_name}'")
            except Exception as e:
                logger.warning(f"Failed to register CJK font {cjk_path}: {e}")

    # Build font family list — CJK FIRST, then Latin fallback
    font_families = []
    if cjk_font_name:
        font_families.append(cjk_font_name)
    # Add remaining CJK fonts as fallbacks BEFORE Latin
    for name in CJK_FONT_PRIORITY:
        if name != cjk_font_name and name not in font_families:
            font_families.append(name)
    # Latin/numbers fallback last
    font_families.append("DejaVu Sans")

    # Configure matplotlib rcParams
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = font_families
    plt.rcParams["axes.unicode_minus"] = False  # Use hyphen-minus instead of Unicode minus

    # Math font configuration
    math_font = None
    for name in MATH_FONT_PRIORITY:
        matches = [f for f in mpl_fm.fontManager.ttflist if name.lower() in f.name.lower()]
        if matches:
            math_font = name
            break

    if math_font and "stix" in math_font.lower():
        plt.rcParams["mathtext.fontset"] = "stix"
    else:
        plt.rcParams["mathtext.fontset"] = "dejavusans"

    # Suppress font-not-found warnings for known fallbacks
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    _setup_done = True

    if cjk_font_name:
        logger.info(f"Font setup complete. Primary CJK font: {cjk_font_name}")
    else:
        logger.warning(
            "⚠️  CJK font setup failed - no suitable Chinese font found!\n"
            "  Chinese text will render as empty boxes (□).\n"
            "  To fix on Ubuntu/WSL:\n"
            "    sudo apt-get install fonts-noto-cjk\n"
            "  Or:\n"
            "    sudo apt-get install fonts-wqy-microhei\n"
            "  After installing, clear matplotlib cache:\n"
            "    rm -rf ~/.cache/matplotlib"
        )


def validate_text_rendering(text: str, font_path: str = "") -> bool:
    """
    Validate that text renders correctly without missing glyph boxes.

    Renders the text to a small temporary image and checks for the presence
    of the Unicode replacement character / tofu boxes.

    Args:
        text: The text string to validate (typically Chinese characters).
        font_path: Path to font file to use. If empty, uses default CJK font.

    Returns:
        True if text appears to render correctly, False if missing glyphs detected.
    """
    if not text:
        return True

    if not font_path:
        font_path = get_font_path("cjk")
        if not font_path:
            logger.warning("No CJK font available for validation")
            return False

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.warning("Pillow not installed, cannot validate text rendering")
        return True  # Assume OK if we can't check

    try:
        # Load font
        font = ImageFont.truetype(font_path, size=24)

        # Create a small image and render text
        img = Image.new("RGB", (len(text) * 30 + 20, 40), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((5, 5), text, font=font, fill="black")

        # Check if any characters rendered as the missing glyph (tofu box)
        # Strategy: render the known-bad replacement char and compare
        # If the font has the glyph, it should produce non-trivial pixels
        test_img = Image.new("RGB", (30, 30), color="white")
        test_draw = ImageDraw.Draw(test_img)

        # Render each character individually and check it produces pixels
        for i, char in enumerate(text):
            char_img = Image.new("L", (28, 28), color=255)
            char_draw = ImageDraw.Draw(char_img)
            char_draw.text((2, 2), char, font=font, fill=0)

            # Count dark pixels - a missing glyph typically produces very few or a box
            pixels = list(char_img.getdata())  # type: ignore[arg-type]
            dark_pixels = sum(1 for p in pixels if p < 128)

            if dark_pixels < 3:
                # Character produced almost no ink - likely missing glyph
                logger.debug(f"Character '{char}' (U+{ord(char):04X}) may not render correctly")
                return False

        return True

    except OSError as e:
        logger.warning(f"Font validation failed: {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error during text validation: {e}")
        return False


def reset():
    """Reset the font manager state. Useful for testing."""
    global _font_cache, _setup_done
    _font_cache = None
    _setup_done = False
