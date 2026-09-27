"""Pixel-based text-weight measurement.

Detects bold vs. regular purely from ink stroke thickness in the source
image -- never inferred from a block's label/role. Absolute stroke-width
ratios vary a lot by font/rendering, so classification compares each label's
ratio against this document's own median rather than a fixed cutoff.
"""

from __future__ import annotations

import statistics

from .ink import _row_ink_bands, binarize_region

BOLD_RELATIVE_FACTOR = 1.15  # a label's median ratio above (doc median * this) is bold


def measure_stroke_ratio(source_image, bbox) -> float | None:
    """Measure this block's ink-stroke-width / text-line-height ratio from
    real pixels -- returns None if it can't tell (no image, empty crop, or
    no clear text signal)."""
    binary = binarize_region(source_image, bbox)
    if binary is None:
        return None

    bands = _row_ink_bands(binary)
    line_height = statistics.median([e - s for s, e in bands]) if bands else binary.shape[0]
    if line_height <= 0:
        return None

    starts = binary.copy()
    starts[:, 1:] &= ~binary[:, :-1]
    num_runs = starts.sum()
    if num_runs == 0:
        return None

    return (binary.sum() / num_runs) / line_height


def classify_bold_by_label(ratios_by_label: dict[str, list[float]]) -> dict[str, bool]:
    """Collapse per-block stroke ratios into one bold/regular verdict per
    label, using this document's own median ratio as the baseline (so a
    role like "paragraph_title" is either bold or not throughout the doc)."""
    all_ratios = [r for ratios in ratios_by_label.values() for r in ratios]
    if not all_ratios:
        return {}
    threshold = statistics.median(all_ratios) * BOLD_RELATIVE_FACTOR
    return {
        label: bool(statistics.median(ratios) > threshold)
        for label, ratios in ratios_by_label.items()
    }
