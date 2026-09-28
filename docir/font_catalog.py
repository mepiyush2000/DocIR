"""The bundled open-licence font families (docir/assets/fonts), registered
with reportlab for embedding, plus the real metrics typography needs.

Only bundled fonts are ever drawn, so output PDFs are self-contained and
identical on every machine; matching (see `typography`) picks among them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")
METRIC_RENDER_PX = 400


@dataclass(frozen=True)
class FontFamily:
    key: str
    category: str            # "serif" | "sans_serif" | "monospace"
    regular: str             # paths relative to FONT_DIR
    bold: str | None = None
    italic: str | None = None
    bold_italic: str | None = None

    def face_path(self, bold: bool, italic: bool) -> tuple[str, bool]:
        """(file, needs synthetic slant) for the requested style; a missing
        italic falls back to the upright face, drawn slanted."""
        if bold and italic and self.bold_italic:
            return self.bold_italic, False
        if italic and not bold and self.italic:
            return self.italic, False
        upright = self.bold if bold and self.bold else self.regular
        return upright, italic


def _fam(key, category, folder, regular, bold=None, italic=None, bold_italic=None) -> FontFamily:
    j = lambda f: f and os.path.join(folder, f)  # noqa: E731
    return FontFamily(key, category, j(regular), j(bold), j(italic), j(bold_italic))


FAMILIES: dict[str, FontFamily] = {f.key: f for f in (
    _fam("LiberationSans", "sans_serif", "liberation", "LiberationSans-Regular.ttf",
         "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf", "LiberationSans-BoldItalic.ttf"),
    _fam("LiberationSansNarrow", "sans_serif", "liberation-narrow", "LiberationSansNarrow-Regular.ttf",
         "LiberationSansNarrow-Bold.ttf", "LiberationSansNarrow-Italic.ttf",
         "LiberationSansNarrow-BoldItalic.ttf"),
    _fam("DejaVuSans", "sans_serif", "dejavu", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
         "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf"),
    _fam("DejaVuSansCondensed", "sans_serif", "dejavu", "DejaVuSansCondensed.ttf",
         "DejaVuSansCondensed-Bold.ttf", "DejaVuSansCondensed-Oblique.ttf",
         "DejaVuSansCondensed-BoldOblique.ttf"),
    _fam("NotoSans", "sans_serif", "noto", "NotoSans-Regular.ttf", "NotoSans-Bold.ttf",
         "NotoSans-Italic.ttf", "NotoSans-BoldItalic.ttf"),
    _fam("PTSansNarrow", "sans_serif", "pt-sans-narrow", "PT_Sans-Narrow-Web-Regular.ttf",
         "PT_Sans-Narrow-Web-Bold.ttf"),
    _fam("LiberationSerif", "serif", "liberation", "LiberationSerif-Regular.ttf",
         "LiberationSerif-Bold.ttf", "LiberationSerif-Italic.ttf", "LiberationSerif-BoldItalic.ttf"),
    _fam("DejaVuSerif", "serif", "dejavu", "DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf",
         "DejaVuSerif-Italic.ttf", "DejaVuSerif-BoldItalic.ttf"),
    _fam("DejaVuSerifCondensed", "serif", "dejavu", "DejaVuSerifCondensed.ttf",
         "DejaVuSerifCondensed-Bold.ttf", "DejaVuSerifCondensed-Italic.ttf",
         "DejaVuSerifCondensed-BoldItalic.ttf"),
    _fam("NotoSerif", "serif", "noto", "NotoSerif-Regular.ttf", "NotoSerif-Bold.ttf",
         "NotoSerif-Italic.ttf", "NotoSerif-BoldItalic.ttf"),
    _fam("LiberationMono", "monospace", "liberation", "LiberationMono-Regular.ttf",
         "LiberationMono-Bold.ttf", "LiberationMono-Italic.ttf", "LiberationMono-BoldItalic.ttf"),
    _fam("DejaVuSansMono", "monospace", "dejavu", "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf",
         "DejaVuSansMono-Oblique.ttf", "DejaVuSansMono-BoldOblique.ttf"),
)}

DEFAULT_FAMILY = {"sans_serif": "LiberationSans", "serif": "LiberationSerif",
                  "monospace": "LiberationMono"}
FALLBACK_FAMILY = "DejaVuSans"   # widest glyph coverage, for characters others lack


def families_in(category: str) -> list[FontFamily]:
    return [f for f in FAMILIES.values() if f.category == category]


def available(family: FontFamily) -> bool:
    return os.path.isfile(os.path.join(FONT_DIR, family.regular))


@dataclass(frozen=True)
class Face:
    name: str                # reportlab font name
    path: str
    synthetic_italic: bool


@lru_cache(maxsize=None)
def face(family_key: str, bold: bool = False, italic: bool = False) -> Face:
    """The registered reportlab face for a family/style (registered once)."""
    family = FAMILIES[family_key]
    rel, synthetic = family.face_path(bold, italic)
    path = os.path.join(FONT_DIR, rel)
    name = os.path.splitext(os.path.basename(rel))[0]
    if name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(name, path))
    return Face(name, path, synthetic)


@lru_cache(maxsize=None)
def x_height_em(path: str) -> float:
    """Height of a lowercase 'x' as a fraction of the font size, measured
    from the font's own outline."""
    return _glyph_height_em(path, "x")


@lru_cache(maxsize=None)
def cap_height_em(path: str) -> float:
    return _glyph_height_em(path, "H")


def _glyph_height_em(path: str, glyph: str) -> float:
    font = ImageFont.truetype(path, METRIC_RENDER_PX)
    img = Image.new("L", (METRIC_RENDER_PX * 2, METRIC_RENDER_PX * 2), 0)
    ImageDraw.Draw(img).text((METRIC_RENDER_PX // 2, METRIC_RENDER_PX // 2), glyph, font=font, fill=255)
    rows = np.where(np.asarray(img).max(axis=1) > 127)[0]
    return float(rows[-1] - rows[0] + 1) / METRIC_RENDER_PX if rows.size else 0.5


@lru_cache(maxsize=None)
def _glyph_set(font_name: str) -> frozenset[int]:
    return frozenset(pdfmetrics.getFont(font_name).face.charToGlyph.keys())


def split_by_coverage(text: str, primary: Face, fallback: Face) -> list[tuple[str, Face]]:
    """Runs of `text`, each drawn with `primary` unless it lacks a glyph."""
    have, spare = _glyph_set(primary.name), _glyph_set(fallback.name)
    runs: list[tuple[str, Face]] = []
    for ch in text:
        use = primary if ord(ch) in have or ord(ch) not in spare or ch.isspace() else fallback
        if runs and runs[-1][1] is use:
            runs[-1] = (runs[-1][0] + ch, use)
        else:
            runs.append((ch, use))
    return runs


def string_width(text: str, font_name: str, size: float) -> float:
    return pdfmetrics.stringWidth(text, font_name, size)
