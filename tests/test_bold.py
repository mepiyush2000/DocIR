import json

import numpy as np
from PIL import Image

from docir.bold import classify_bold_by_label, measure_stroke_ratio
from docir.schemas import PageLayout

from .conftest import FIXTURE_IMAGE, FIXTURE_JSON


def _synthetic_text_image(stroke_px: int, gap_px: int = 3, width: int = 120, height: int = 30) -> Image.Image:
    """A single horizontal line of ink bars at a fixed line height, used to
    simulate thin (regular) vs. thick (bold) strokes."""
    arr = np.full((height, width), 255, dtype=np.uint8)
    y0, y1 = 8, 20
    x = 4
    while x < width - stroke_px:
        arr[y0:y1, x:x + stroke_px] = 0
        x += stroke_px + gap_px
    return Image.fromarray(arr, mode="L").convert("RGB")


def test_measure_stroke_ratio_increases_with_stroke_width():
    thin = measure_stroke_ratio(_synthetic_text_image(stroke_px=2), (0, 0, 120, 30))
    thick = measure_stroke_ratio(_synthetic_text_image(stroke_px=6), (0, 0, 120, 30))
    assert thin is not None and thick is not None
    assert thick > thin


def test_measure_stroke_ratio_none_without_image():
    assert measure_stroke_ratio(None, (0, 0, 10, 10)) is None


def test_measure_stroke_ratio_none_for_empty_bbox():
    img = _synthetic_text_image(stroke_px=2)
    assert measure_stroke_ratio(img, (5, 5, 5, 5)) is None


def test_classify_bold_by_label_relative_to_document_median():
    ratios = {
        "text": [0.20, 0.21, 0.22, 0.23],
        "title": [0.38, 0.37, 0.39],
    }
    result = classify_bold_by_label(ratios)
    assert result["text"] is False
    assert result["title"] is True


def test_classify_bold_by_label_empty():
    assert classify_bold_by_label({}) == {}


def test_bold_classification_regression_on_real_document():
    """Locks in the fix for a real bug: a fixed absolute stroke-ratio
    threshold previously classified every block in this document as bold."""
    with open(FIXTURE_JSON) as f:
        layout = PageLayout.from_paddle_dict(json.load(f))
    img = Image.open(FIXTURE_IMAGE).convert("RGB")

    ratios_by_label: dict[str, list[float]] = {}
    for block in layout.blocks:
        if not block.content:
            continue
        ratio = measure_stroke_ratio(img, block.bbox)
        if ratio is not None:
            ratios_by_label.setdefault(block.label, []).append(ratio)

    consensus = classify_bold_by_label(ratios_by_label)
    assert consensus["text"] is False
    assert consensus["header"] is True
    assert consensus["paragraph_title"] is True
