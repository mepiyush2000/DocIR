import os

import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from docir.fonts import (
    _classify_font_file,
    classify_word_samples,
    collect_word_samples,
    discover_catalog,
    match_style_category,
)

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


_SERIF_FONT = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
_MONO_FONT = "/System/Library/Fonts/Supplemental/Courier New.ttf"


@pytest.mark.skipif(not os.path.isfile(_SERIF_FONT), reason="no local Times New Roman font found")
def test_match_style_category_recovers_serif():
    category, score = match_style_category(_render_text(_SERIF_FONT), (0, 0, 900, 100), _TEXT)
    assert category == "serif"
    assert score > 0


@pytest.mark.skipif(not os.path.isfile(_SERIF_FONT), reason="no local Times New Roman font found")
def test_match_style_category_recovers_serif_at_scan_resolution():
    """Small, blurred text (like a real scan) must still read as serif --
    renders are degraded to the same resolution before comparing."""
    font = ImageFont.truetype(_SERIF_FONT, 14)
    img = Image.new("L", (460, 40), color=255)
    ImageDraw.Draw(img).text((6, 10), _TEXT, font=font, fill=0)
    img = img.filter(ImageFilter.GaussianBlur(0.6)).convert("RGB")
    category, _ = match_style_category(img, (0, 0, 460, 40), _TEXT)
    assert category == "serif"


@pytest.mark.parametrize("path, expected", [
    (_SANS_FONT, "sans_serif"),
    (_SERIF_FONT, "serif"),
    (_MONO_FONT, "monospace"),
])
def test_font_files_are_categorized_by_their_rendered_glyphs(path, expected):
    if not os.path.isfile(path):
        pytest.skip(f"{path} not installed")
    assert _classify_font_file(path) == expected


def test_script_fonts_are_not_catalogued():
    path = "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc"
    if not os.path.isfile(path):
        pytest.skip("no local script font found")
    assert _classify_font_file(path) is None


@pytest.mark.skipif(not (_HAS_SANS_FONT and os.path.isfile(_SERIF_FONT)), reason="reference fonts missing")
def test_pooled_word_samples_decide_category():
    samples = collect_word_samples(_render_text(_SERIF_FONT), (0, 0, 900, 100), _TEXT)
    assert len(samples) >= 3
    category, margin = classify_word_samples(samples)
    assert category == "serif" and margin > 0
    assert classify_word_samples([]) == ("unknown", 0.0)
