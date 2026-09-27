import os

import pytest
from PIL import Image, ImageDraw, ImageFont

from docir.fonts import discover_catalog, match_style_category

_SANS_FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
_HAS_SANS_FONT = os.path.isfile(_SANS_FONT)

_TEXT = "The quick brown fox jumps over the lazy dog and runs away quickly"


def _render_text(path: str) -> Image.Image:
    font = ImageFont.truetype(path, 40)
    img = Image.new("L", (900, 100), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((10, 20), _TEXT, font=font, fill=0)
    return img.convert("RGB")


def test_discover_catalog_entries_have_valid_categories():
    for face in discover_catalog():
        assert face.style_category in {"serif", "sans_serif", "monospace"}
        assert os.path.isfile(face.path)


@pytest.mark.skipif(not _HAS_SANS_FONT, reason="no local Arial font found")
def test_match_style_category_recovers_sans_serif():
    category, score = match_style_category(_render_text(_SANS_FONT), (0, 0, 900, 100), _TEXT)
    assert category == "sans_serif"
    assert score > 0


def test_match_style_category_unknown_without_image():
    assert match_style_category(None, (0, 0, 10, 10), _TEXT) == ("unknown", 0.0)


def test_match_style_category_unknown_for_short_text():
    img = _render_text(_SANS_FONT) if _HAS_SANS_FONT else Image.new("RGB", (100, 30), color="white")
    category, score = match_style_category(img, (0, 0, 100, 30), "ab")
    assert category == "unknown"
    assert score == 0.0
