"""Draw a `Page` model as a fully vector PDF.

Draw order: page background -> block fills -> rules -> pictures -> text.
Text is written in reading order, every line at its measured baseline and
tilt. The role's font, size and glyph width are shared by all its lines; a
line's small width differences are absorbed by word spacing, then letter
spacing, and only then by at most +/-5% extra glyph width -- so lines keep
one consistent look yet end exactly where the original's did. Blocks with
no measured lines fall back to a paragraph laid out inside their box.
"""

from __future__ import annotations

import html
import logging
import math
import re

import numpy as np
from PIL import Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Paragraph

from . import font_catalog
from .page_model import WHITE, Block, Color, Line, Page, TextStyle
from .page_ratios import closest_standard_ratio

logger = logging.getLogger(__name__)

TARGET_PAGE_WIDTH_PT = 612.0
STANDARD_RATIO_TOLERANCE = 0.05
WORD_SPACE_RANGE_EM = (-0.12, 0.6)
CHAR_SPACE_RANGE_EM = (-0.02, 0.05)
LINE_WIDTH_SLACK = 0.05           # extra glyph width a line may take beyond its role's
LINE_SHRINK_SLACK = 0.12          # a line may condense this much before gaps shrink
MIN_SPACE_KEEP = 0.5              # word gaps keep at least half a normal space
SYNTHETIC_ITALIC_DEG = 11.0
FILL_PAD_PX = 2.0
PARAGRAPH_LEADING = 1.2
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _rgb(c: Color) -> tuple[float, float, float]:
    return c[0] / 255, c[1] / 255, c[2] / 255


class _Geometry:
    """Source pixels -> PDF points: one uniform scale from the top-left, on
    a canvas snapped to a standard paper ratio (never shorter than content)."""

    def __init__(self, width_px: float, height_px: float):
        self.scale = TARGET_PAGE_WIDTH_PT / width_px
        self.page_w = TARGET_PAGE_WIDTH_PT
        content_h = height_px * self.scale
        ratio = closest_standard_ratio(width_px, height_px, STANDARD_RATIO_TOLERANCE)
        self.page_h = max(content_h, self.page_w / ratio.ratio if ratio else content_h)

    def pt(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale, self.page_h - y * self.scale


def _clean(text: str) -> str:
    return _CONTROL.sub("", text).replace("\t", " ")


# ---------- line fitting ----------

def fit_line(words: list[str], word_widths: list[float], space_width: float, n_chars: int,
             target: float, size: float, role_scale: float) -> tuple[float, float, float]:
    """(word gap addition, char spacing, glyph width scale) that make the
    words span `target`. Widths are unscaled (at the font size); the glyph
    scale multiplies everything, so the gaps are solved in unscaled units.
    Stretching goes into word gaps first (justified text); shrinking goes
    into glyph width and letter spacing first, so words never collide."""
    natural = sum(word_widths) + space_width * (len(words) - 1)
    if natural <= 0 or target <= 0:
        return 0.0, 0.0, role_scale
    gaps = len(words) - 1
    ws_lo, ws_hi = (v * size for v in WORD_SPACE_RANGE_EM)
    ws_lo = max(ws_lo, -MIN_SPACE_KEEP * space_width)
    cs_lo, cs_hi = (v * size for v in CHAR_SPACE_RANGE_EM)
    hz_lo, hz_hi = role_scale * (1 - LINE_SHRINK_SLACK), role_scale * (1 + LINE_WIDTH_SLACK)

    def width(ws, cs):
        return natural + ws * gaps + cs * n_chars

    if target >= natural * role_scale:
        excess = target / role_scale - natural
        ws = min(excess / gaps, ws_hi) if gaps else 0.0
        excess -= ws * gaps
        cs = min(max(excess / n_chars, cs_lo), cs_hi) if n_chars > 1 else 0.0
        hz = min(max(target / width(ws, cs), hz_lo), hz_hi)
        return ws, cs, hz
    hz = max(target / natural, hz_lo)
    excess = target / hz - natural
    cs = max(excess / n_chars, cs_lo) if n_chars > 1 else 0.0
    excess -= cs * n_chars
    ws = max(excess / gaps, ws_lo) if gaps else 0.0
    return ws, cs, hz


def _runs_width(runs, size: float) -> float:
    return sum(font_catalog.string_width(text, f.name, size) for text, f in runs)


def _draw_line(c: canvas.Canvas, geo: _Geometry, line: Line, style: TextStyle) -> None:
    text = _clean(line.text).strip()
    if not text:
        return
    size = (line.size_px or style.size_px) * geo.scale
    if size <= 0.5:
        return
    face = font_catalog.face(style.font_family, style.bold, style.italic)
    fallback = font_catalog.face(font_catalog.FALLBACK_FAMILY, style.bold, style.italic)
    words = text.split()
    word_runs = [font_catalog.split_by_coverage(w, face, fallback) for w in words]
    widths = [_runs_width(r, size) for r in word_runs]
    space_w = font_catalog.string_width(" ", face.name, size)
    n_chars = sum(len(w) for w in words)
    if line.drop_cap:
        ws, cs, hz = 0.0, 0.0, 1.0
    else:
        ws, cs, hz = fit_line(words, widths, space_w, n_chars, line.length * geo.scale,
                              size, style.width_scale)

    x0, y0 = geo.pt(*line.baseline_start)
    c.saveState()
    c.translate(x0, y0)
    if abs(line.angle_deg) > 0.05:
        c.rotate(line.angle_deg)
    if face.synthetic_italic:
        c.skew(0, SYNTHETIC_ITALIC_DEG)
    c.setFillColorRGB(*_rgb(line.color))
    t = c.beginText()
    t.setHorizScale(hz * 100)
    t.setCharSpace(cs)
    x = 0.0
    for runs, w in zip(word_runs, widths):
        t.setTextOrigin(x, 0)
        for run, f in runs:
            t.setFont(f.name, size)
            t.textOut(run)
        # letter spacing follows every glyph, the last one included
        x += (w + cs * sum(len(r) for r, _ in runs) + space_w + ws) * hz
    c.drawText(t)
    c.restoreState()


# ---------- paragraph fallback ----------

def _draw_paragraph(c: canvas.Canvas, geo: _Geometry, block: Block, style: TextStyle | None) -> None:
    x0, y0, x1, y1 = block.bbox
    w, h = (x1 - x0) * geo.scale, (y1 - y0) * geo.scale
    if w <= 1 or h <= 1:
        return
    family = style.font_family if style else font_catalog.DEFAULT_FAMILY["sans_serif"]
    face = font_catalog.face(family, bool(style and style.bold), bool(style and style.italic))
    text = html.escape(_clean(block.content)).replace("\n", "<br/>")
    lo, hi = 2.0, max(4.0, h)
    best = None
    for _ in range(16):
        size = (lo + hi) / 2
        para = Paragraph(text, ParagraphStyle("p", fontName=face.name, fontSize=size,
                                              leading=size * PARAGRAPH_LEADING))
        _, need = para.wrap(w, 1e6)
        if need <= h:
            best, lo = para, size
        else:
            hi = size
    if best is None:
        return
    left, top = geo.pt(x0, y0)
    Frame(left, top - h, w, h, leftPadding=0, rightPadding=0, topPadding=0,
          bottomPadding=0, showBoundary=0).addFromList([best], c)


# ---------- pictures / fills / rules ----------

def _draw_picture(c, geo: _Geometry, image: Image.Image | None, bbox) -> None:
    if image is None:
        return
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1)
    if x1 <= x0 or y1 <= y0:
        return
    left, top = geo.pt(x0, y0)
    c.drawImage(ImageReader(image.crop((x0, y0, x1, y1))), left, top - (y1 - y0) * geo.scale,
                (x1 - x0) * geo.scale, (y1 - y0) * geo.scale)


def _fill_rect(c, geo: _Geometry, bbox, color: Color, pad: float = 0.0) -> None:
    x0, y0, x1, y1 = bbox
    left, top = geo.pt(x0 - pad, y0 - pad)
    c.setFillColorRGB(*_rgb(color))
    c.rect(left, top - (y1 - y0 + 2 * pad) * geo.scale, (x1 - x0 + 2 * pad) * geo.scale,
           (y1 - y0 + 2 * pad) * geo.scale, fill=1, stroke=0)


def render_page_model(page: Page, output_path: str, source_image: Image.Image | None = None) -> str:
    geo = _Geometry(page.width, page.height)
    c = canvas.Canvas(output_path, pagesize=(geo.page_w, geo.page_h))
    c.setTitle("DocIR reconstruction")
    if page.background != WHITE:
        c.setFillColorRGB(*_rgb(page.background))
        c.rect(0, 0, geo.page_w, geo.page_h, fill=1, stroke=0)
    for block in page.blocks:
        if block.fill is not None:
            _fill_rect(c, geo, block.bbox, block.fill, FILL_PAD_PX)
    c.setLineCap(1)
    for g in page.graphics:
        if g.kind == "rule" and len(g.points) == 2:
            c.setStrokeColorRGB(*_rgb(g.color))
            c.setLineWidth(max(g.stroke_px * geo.scale, 0.25))
            (ax, ay), (bx, by) = (geo.pt(*p) for p in g.points)
            c.line(ax, ay, bx, by)
        elif g.kind == "fill":
            _fill_rect(c, geo, g.bbox, g.color)
    for pic in page.pictures:
        _draw_picture(c, geo, source_image, pic.bbox)
    n_lines = n_fallback = 0
    for block in sorted(page.blocks, key=lambda b: b.reading_index):
        if block.lines and block.style is not None:
            for line in block.lines:
                _draw_line(c, geo, line, block.style)
                n_lines += 1
        else:
            _draw_paragraph(c, geo, block, block.style)
            n_fallback += 1
    c.showPage()
    c.save()
    logger.debug("rendered %d lines, %d paragraph-fallback blocks", n_lines, n_fallback)
    return output_path
