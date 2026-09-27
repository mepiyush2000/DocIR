"""Shared ink/background binarization for pixel-based style measurement
(bold, italic, and font-family matching all start from the same mask)."""

from __future__ import annotations

import numpy as np
from PIL import Image


def _otsu_threshold(gray: np.ndarray) -> float:
    """Pick the gray-level splitting ink from background via Otsu's method."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 127
    sum_all = np.dot(np.arange(256), hist)
    sum_bg, weight_bg = 0.0, 0.0
    # track the *range* of gray levels tying for max variance, not just the
    # first -- a pure bilevel image (e.g. a binarized scan) has a wide flat
    # plateau, and picking its first edge (0) instead of its midpoint would
    # make the ink/background split degenerate to nothing.
    best_thresh_start, best_thresh_end, best_var = 127, 127, -1.0
    for i in range(256):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += i * hist[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh_start = best_thresh_end = i
        elif var_between == best_var:
            best_thresh_end = i
    return (best_thresh_start + best_thresh_end) / 2


def _row_ink_bands(binary: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous rows of ink (text-line bands) to estimate line height."""
    row_density = binary.mean(axis=1)
    peak = row_density.max()
    if peak <= 0:
        return []
    is_text_row = row_density > (peak * 0.15)
    bands = []
    start = None
    for i, v in enumerate(is_text_row):
        if v and start is None:
            start = i
        elif not v and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, len(is_text_row)))
    return bands


def binarize_region(source_image: Image.Image | None, bbox) -> np.ndarray | None:
    """Crop `bbox` out of `source_image` and return a boolean ink mask
    (True = ink), or None if it can't be measured (no image, empty crop, or
    no clear ink/background split)."""
    if source_image is None:
        return None
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, source_image.width), min(y2, source_image.height)
    if x2 <= x1 or y2 <= y1:
        return None

    gray = np.asarray(source_image.crop((x1, y1, x2, y2)).convert("L"), dtype=np.float64)
    if gray.size == 0:
        return None

    thresh = _otsu_threshold(gray)
    dark_frac = (gray < thresh).mean()
    # text is the minority-coverage class, whichever side of the threshold that is
    binary = (gray < thresh) if dark_frac < 0.5 else (gray >= thresh)
    ink_frac = binary.mean()
    if ink_frac == 0 or ink_frac > 0.6:
        return None
    return binary
