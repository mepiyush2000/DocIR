import os

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from docir.ink import component_boxes, estimate_skew, measure_text_lines, rotate_mask

_SANS_FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
_LINES = ["the quick brown fox jumps over", "a lazy dog while seven zebras", "quietly wander home again", "short"]


def _paragraph_mask(px_size=30, pitch=42, lines=_LINES, angle=0.0):
    font = ImageFont.truetype(_SANS_FONT, px_size)
    img = Image.new("L", (px_size * 18, pitch * (len(lines) + 2)), 255)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((20, 20 + i * pitch), line, font=font, fill=0)
    if angle:
        img = img.rotate(angle, expand=True, fillcolor=255)
    return np.asarray(img) < 128


@pytest.mark.skipif(not os.path.isfile(_SANS_FONT), reason="no local Arial font found")
def test_measure_text_lines_counts_lines_including_short_last_line():
    stats = measure_text_lines(_paragraph_mask())
    assert stats.n_lines == len(_LINES)
    assert stats.pitch == pytest.approx(42, abs=2)
    # Arial's x-height is 0.519 em
    assert stats.x_height == pytest.approx(0.519 * 30, abs=2)


@pytest.mark.skipif(not os.path.isfile(_SANS_FONT), reason="no local Arial font found")
def test_measure_text_lines_handles_tilted_text():
    stats = measure_text_lines(_paragraph_mask(angle=6.0))
    assert stats.n_lines == len(_LINES)
    assert stats.pitch == pytest.approx(42, abs=3)


@pytest.mark.skipif(not os.path.isfile(_SANS_FONT), reason="no local Arial font found")
def test_estimate_skew_recovers_rotation():
    mask = _paragraph_mask()
    assert estimate_skew(rotate_mask(mask, 5.0)) == pytest.approx(-5.0, abs=0.5)


def test_measure_text_lines_none_for_empty_mask():
    assert measure_text_lines(np.zeros((20, 50), dtype=bool)) is None


def test_component_boxes_finds_separate_and_diagonal_components():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True
    mask[3, 3] = True  # diagonal neighbor -> same 8-connected component
    mask[6:9, 6:8] = True
    boxes = sorted(component_boxes(mask))
    assert boxes == [(1, 4, 1, 4), (6, 9, 6, 8)]
