"""Pixel-based text-weight measurement.

Detects bold vs. regular purely from ink stroke thickness in the source
image -- never inferred from a block's label/role. Absolute stroke-width
ratios vary a lot by font/rendering, so classification compares each label's
ratio against this document's own median rather than a fixed cutoff.
"""

from __future__ import annotations

import statistics

from .ink import binarize_region, measure_text_lines

BOLD_RELATIVE_FACTOR = 1.15  # a label's median ratio above (doc median * this) is bold
# one block this much heavier than the body is bold even when its role is not
# (bold questions or lead-ins inside regular text); all-caps text measures light, never heavy
BLOCK_BOLD_FACTOR = 1.3


def measure_stroke_ratio(source_image, bbox) -> float | None:
    """This block's stem width relative to its x-height: the mean
    horizontal ink-run length (a sub-pixel stem estimate) over the measured
    x-height, so the weight doesn't depend on how many ascenders or capitals
    the text contains. None if it can't tell (no image, empty crop, or no
    clear text signal)."""
    binary = binarize_region(source_image, bbox)
    if binary is None:
        return None

    stats = measure_text_lines(binary)
    if stats is None or stats.x_height <= 0:
        return None

    starts = binary.copy()
    starts[:, 1:] &= ~binary[:, :-1]
    num_runs = starts.sum()
    if num_runs == 0:
        return None

    return float(binary.sum() / num_runs) / stats.x_height


def baseline_ratio(ratios_by_label: dict[str, list[float]], baseline_label: str | None = None) -> float | None:
    """The document's regular-weight stroke ratio: the body label's median when
    given (headings can outnumber paragraphs), else the median of all blocks."""
    all_ratios = [r for ratios in ratios_by_label.values() for r in ratios]
    if not all_ratios:
        return None
    base = ratios_by_label.get(baseline_label) if baseline_label else None
    return statistics.median(base or all_ratios)


def classify_bold_by_label(ratios_by_label: dict[str, list[float]],
                           baseline_label: str | None = None) -> dict[str, bool]:
    """Collapse per-block stroke ratios into one bold/regular verdict per
    label, relative to this document's own baseline (so a role like
    "paragraph_title" is either bold or not throughout the doc)."""
    base = baseline_ratio(ratios_by_label, baseline_label)
    if base is None:
        return {}
    threshold = base * BOLD_RELATIVE_FACTOR
    return {
        label: bool(statistics.median(ratios) > threshold)
        for label, ratios in ratios_by_label.items()
    }


def is_bold_outlier(ratio: float | None, baseline: float | None) -> bool:
    return ratio is not None and baseline is not None and ratio > baseline * BLOCK_BOLD_FACTOR
