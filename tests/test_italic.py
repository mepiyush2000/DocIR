import os

import pytest
from PIL import Image, ImageDraw, ImageFont

from docir.italic import ITALIC_ANGLE_THRESHOLD_DEG, classify_italic_by_label, measure_slant_angle

_REGULAR_FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
_ITALIC_FONT = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"
_HAS_SYSTEM_FONTS = os.path.isfile(_REGULAR_FONT) and os.path.isfile(_ITALIC_FONT)


def _render_text(path: str) -> Image.Image:
    font = ImageFont.truetype(path, 40)
    img = Image.new("L", (500, 90), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((10, 20), "Sample Text Here", font=font, fill=0)
    return img.convert("RGB")


@pytest.mark.skipif(not _HAS_SYSTEM_FONTS, reason="no local Arial/Arial Italic fonts found")
def test_measure_slant_angle_distinguishes_upright_from_italic():
    upright_angle = measure_slant_angle(_render_text(_REGULAR_FONT), (0, 0, 500, 90))
    italic_angle = measure_slant_angle(_render_text(_ITALIC_FONT), (0, 0, 500, 90))
    assert upright_angle is not None and italic_angle is not None
    assert abs(upright_angle) < ITALIC_ANGLE_THRESHOLD_DEG
    assert abs(italic_angle) > ITALIC_ANGLE_THRESHOLD_DEG


def test_measure_slant_angle_none_without_image():
    assert measure_slant_angle(None, (0, 0, 10, 10)) is None


def test_measure_slant_angle_none_for_empty_bbox():
    img = Image.new("RGB", (100, 100), color="white")
    assert measure_slant_angle(img, (5, 5, 5, 5)) is None


def test_classify_italic_by_label_uses_absolute_threshold():
    angles = {
        "text": [0.5, -1.0, 0.2],
        "vision_footnote": [11.0, 12.5, 10.0],
    }
    result = classify_italic_by_label(angles)
    assert result["text"] is False
    assert result["vision_footnote"] is True


def test_classify_italic_by_label_empty():
    assert classify_italic_by_label({}) == {}
