import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from docir import font_catalog
from docir.line_geometry import measure_line, pixel_lines
from docir.render_model import fit_line
from docir.typography import size_for_x_height

FONT = font_catalog.face("LiberationSans").path


def _text_image(lines, size=24, angle=0.0, width=700, pitch=40):
    img = Image.new("L", (width, pitch * (len(lines) + 2)), 255)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, size)
    for i, t in enumerate(lines):
        d.text((20, pitch * (i + 1)), t, font=font, fill=0, anchor="ls")
    return img.rotate(angle, resample=Image.BICUBIC, fillcolor=255) if angle else img


def test_measure_line_finds_baseline_and_x_height():
    img = _text_image(["the quick brown fox jumps"])
    m = measure_line(np.asarray(img), [(0, 10), (700, 10), (700, 55), (0, 55)])
    assert m is not None
    assert m.baseline_start[1] == pytest.approx(40, abs=1.5)
    assert m.x_height == pytest.approx(font_catalog.x_height_em(FONT) * 24, rel=0.15)
    assert m.baseline_start[0] == pytest.approx(20, abs=3)


def test_size_from_measured_x_height_recovers_font_size():
    img = _text_image(["the quick brown fox jumps over"], size=30, pitch=50)
    m = measure_line(np.asarray(img), [(0, 10), (700, 10), (700, 70), (0, 70)])
    assert size_for_x_height(m.x_height, FONT) == pytest.approx(30, rel=0.06)


def test_pixel_lines_separates_tilted_lines():
    text = ["the quick brown fox jumps over", "a lazy dog while seven zebras", "quietly wander home again"]
    img = _text_image(text, angle=2.0).convert("RGB")
    quads = pixel_lines(img, (0, 0, img.width, img.height))
    assert len(quads) == 3


def test_fit_line_stretches_gaps_first_and_shrinks_glyphs_first():
    words, widths = ["aa", "bb", "cc"], [20.0, 20.0, 20.0]
    ws, cs, hz = fit_line(words, widths, 5.0, 6, 80.0, 10.0, 1.0)
    assert ws > 0 and hz == pytest.approx(1.0, abs=0.06)
    ws, cs, hz = fit_line(words, widths, 5.0, 6, 64.0, 10.0, 1.0)
    assert hz < 1.0 and ws >= -2.5
