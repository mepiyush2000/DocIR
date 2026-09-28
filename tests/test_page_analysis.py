import numpy as np
from PIL import Image, ImageDraw, ImageFont

from docir import font_catalog
from docir.color import snap_background, snap_ink
from docir.dewarp import dewarp
from docir.graphics import detect_rules
from docir.page_model import BLACK, WHITE, Page
from docir.preprocess import measure_distortion


def test_snapping_keeps_real_colours():
    assert snap_background((236, 232, 226)) == WHITE
    assert snap_ink((70, 66, 60)) == BLACK
    assert snap_ink((40, 110, 70)) == (40, 110, 70)
    assert snap_background((250, 225, 150)) == (250, 225, 150)


def test_rule_detection_finds_separator_not_edges():
    img = Image.new("RGB", (800, 600), "white")
    d = ImageDraw.Draw(img)
    d.line([(100, 300), (700, 300)], fill="black", width=2)
    d.line([(3, 0), (3, 599)], fill="gray", width=4)       # scan-edge shadow
    rules = detect_rules(Page(width=800, height=600), img)
    assert len(rules) >= 1
    assert all(abs(g.points[0][1] - 300) < 3 for g in rules)
    assert min(g.points[0][0] for g in rules) < 110 and max(g.points[1][0] for g in rules) > 690


def _curved_page() -> Image.Image:
    font = ImageFont.truetype(font_catalog.face("LiberationSerif").path, 18)
    flat = Image.new("L", (900, 1100), 255)
    d = ImageDraw.Draw(flat)
    for i in range(30):
        d.text((60, 60 + i * 32), "the quick brown fox jumps over the lazy dog again and again ok",
               font=font, fill=0)
    a = np.asarray(flat)
    h, w = a.shape
    xs = np.arange(w)
    shift = (18 * ((xs - w / 2) / (w / 2)) ** 2).astype(int)   # a book-page curl
    out = np.full_like(a, 255)
    for x in range(w):
        out[:, x] = np.roll(a[:, x], shift[x])
    return Image.fromarray(out).convert("RGB")


def test_dewarp_levels_curved_lines():
    page = _curved_page()
    before = measure_distortion(page)
    flat = dewarp(page)
    assert flat is not None
    after = measure_distortion(flat)
    assert before.is_curved
    assert after.curvature_deg < before.curvature_deg
    assert not after.is_curved
