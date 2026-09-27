import os

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from docir.ink import measure_text_lines
from docir.render_pdf import (
    _consensus_cluster_size,
    _line_count_size_range,
    _measured_font_size,
    _wrapped_line_count,
)

_SANS_FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
_TEXT = ("Automation technologies can be broadly grouped into three categories that "
         "differ in how much they can learn from the data they are given every day")


def test_line_count_size_range_reproduces_the_target_line_count():
    width = 200.0
    lo, hi = _line_count_size_range("text", _TEXT, width, 3, False, False, "sans_serif")
    assert lo < hi
    for size in (lo + 0.01, (lo + hi) / 2, hi):
        assert _wrapped_line_count("text", _TEXT, width, size, False, False, "sans_serif") == 3


def test_line_count_size_range_respects_explicit_line_breaks():
    """Hard line breaks in OCR text make fewer lines impossible; the range
    must still exist (for the smallest achievable count) instead of None."""
    text = "one\ntwo\nthree"
    rng = _line_count_size_range("text", text, 300.0, 1, False, False, "sans_serif")
    assert rng is not None
    assert _wrapped_line_count("text", text, 300.0, rng[1], False, False, "sans_serif") == 3


@pytest.mark.skipif(not os.path.isfile(_SANS_FONT), reason="no local Arial font found")
def test_measured_font_size_recovers_rendered_size():
    """Arial ~ Helvetica metrics: a paragraph rendered at a known pixel size
    should be measured back at that size (scale 1 => 1px == 1pt)."""
    px_size = 24
    font = ImageFont.truetype(_SANS_FONT, px_size)
    lines = ["Automation technologies can be broadly", "grouped into three categories that differ",
             "in how much they can learn from the data"]
    img = Image.new("L", (560, 160), 255)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((10, 10 + i * 32), line, font=font, fill=0)
    stats = measure_text_lines(np.asarray(img) < 128)
    size, size_range = _measured_font_size("text", " ".join(lines), 540.0, stats, 1.0,
                                           False, False, "sans_serif")
    assert size == pytest.approx(px_size, rel=0.08)
    assert size_range is not None and size_range[0] <= size <= size_range[1] * 1.1


def test_consensus_cluster_size_prefers_a_size_most_blocks_can_share():
    cluster = [(1, 9.0), (2, 9.2), (3, 9.1)]
    ranges = {1: (7.0, 8.0), 2: (7.5, 8.4), 3: (6.0, 9.5)}
    size = _consensus_cluster_size(cluster, ranges)
    assert all(lo <= size <= hi for lo, hi in ranges.values())
