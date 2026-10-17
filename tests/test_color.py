import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from docir.color import ILLUMINATION_SIGMA_FRAC, UNEVEN_LIGHT_RANGE, assign_colors, flat_field
from docir.page_model import Block, Line, Page

BAR = (10, 40, 140)


def _line(x0, x1, baseline, xh=12.0):
    return Line(text="Address", baseline_start=(x0, baseline), baseline_end=(x1, baseline),
                x_height=xh, cap_top=1.4 * xh)


def test_light_text_on_a_coloured_bar_gets_the_bar_as_fill():
    img = Image.new("RGB", (600, 300), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((40, 180, 560, 260), fill=BAR)
    for i in range(10):                                 # white word-like glyphs
        d.rectangle((120 + i * 30, 210, 128 + i * 30, 230), fill="white")
    block = Block(id=0, role="footer", reading_index=0, bbox=(110, 200, 420, 235),
                  content="Address", lines=[_line(120, 410, 230)])
    page = Page(width=600, height=300, blocks=[block])
    assign_colors(page, img)
    assert block.fill is not None and max(abs(a - b) for a, b in zip(block.fill, BAR)) < 10
    x0, y0, x1, y1 = block.fill_bbox
    assert abs(x0 - 40) <= 3 and abs(y0 - 180) <= 3 and abs(x1 - 561) <= 3 and abs(y1 - 261) <= 3
    assert min(block.lines[0].color) > 200


def test_dark_text_on_white_paper_has_no_panel():
    img = Image.new("RGB", (600, 300), "white")
    d = ImageDraw.Draw(img)
    for i in range(10):
        d.rectangle((120 + i * 30, 210, 140 + i * 30, 230), fill="black")
    block = Block(id=0, role="text", reading_index=0, bbox=(110, 200, 420, 235),
                  content="Address", lines=[_line(120, 410, 230)])
    page = Page(width=600, height=300, blocks=[block])
    assign_colors(page, img)
    assert block.fill is None and block.fill_bbox is None


@pytest.mark.parametrize("shape", [(43, 51), (180, 240), (320, 180), (1414, 1000)])
def test_flat_field_matches_gaussian_reference(shape):
    h, w = shape
    light = np.linspace(90, 240, w)[None, :, None]
    rgb = np.broadcast_to(light * np.array([1.0, 0.9, 0.8]), (h, w, 3)).astype(np.uint8).copy()
    rgb[h // 3:h // 3 + 2, w // 4:3 * w // 4] = 20
    k = max(5, int(w * 0.02) | 1)
    closed = cv2.morphologyEx(
        rgb, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    illum = cv2.GaussianBlur(closed.astype(np.float32), (0, 0), max(w, h) * ILLUMINATION_SIGMA_FRAC)
    lum = illum.mean(axis=2)
    assert np.percentile(lum, 90) - np.percentile(lum, 10) >= UNEVEN_LIGHT_RANGE
    paper = np.percentile(illum.reshape(-1, 3), 90, axis=0)
    expected = np.clip(rgb.astype(np.float32) / np.maximum(illum, 1.0) * paper, 0, 255).astype(np.uint8)
    np.testing.assert_allclose(flat_field(rgb), expected, atol=1, rtol=0)


def test_flat_field_preserves_evenly_lit_page():
    rgb = np.full((160, 200, 3), 230, dtype=np.uint8)
    rgb[40:42, 30:170] = 0
    assert flat_field(rgb) is rgb
