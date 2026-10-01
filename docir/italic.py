"""Pixel-based italic/oblique detection via shear-search deskewing.

For each block we try shearing its binarized ink mask by a range of
candidate slant angles and keep the one that makes the vertical stroke
projection profile sharpest -- upright strokes collapse into thin, dense
columns when the shear exactly cancels their slant, while the wrong shear
smears ink across columns. Unlike bold's stroke-width ratio (which needs a
document-relative baseline), 0 degrees is a real physical baseline for
upright text, so a fixed absolute threshold is used for classification.
"""

from __future__ import annotations

import statistics

import numpy as np

from .ink import binarize_region

ITALIC_ANGLE_THRESHOLD_DEG = 6.0
_SEARCH_ANGLES_DEG = tuple(range(-24, 25, 3))  # coarse shear search, refined by callers if needed


def _shear_mask(binary: np.ndarray, angle_deg: float) -> np.ndarray:
    """Shift each row horizontally in proportion to its distance from the
    vertical center, simulating a shear by `angle_deg`."""
    h, w = binary.shape
    shear = np.tan(np.radians(angle_deg))
    center = h / 2.0
    sheared = np.zeros_like(binary)
    for y in range(h):
        shift = int(round(shear * (y - center)))
        if abs(shift) >= w:
            continue  # sheared entirely out of a narrow crop
        if shift == 0:
            sheared[y] = binary[y]
        elif shift > 0:
            sheared[y, shift:] = binary[y, : w - shift]
        else:
            sheared[y, : w + shift] = binary[y, -shift:]
    return sheared


def _peakiness(binary: np.ndarray) -> float:
    """How sharply ink concentrates into narrow columns (higher = more
    vertical/upright strokes)."""
    col_density = binary.mean(axis=0)
    return float(np.square(col_density).sum())


def measure_slant_angle(source_image, bbox) -> float | None:
    """Estimate this block's average stroke slant in degrees (0 = upright,
    positive = leaning right, as in italics) -- returns None if it can't
    tell (no image, empty crop, or no clear text signal)."""
    binary = binarize_region(source_image, bbox)
    if binary is None or binary.shape[0] < 6 or binary.shape[1] < 6:
        return None

    best_angle, best_score = 0.0, -1.0
    for angle in _SEARCH_ANGLES_DEG:
        sheared = _shear_mask(binary, angle)
        score = _peakiness(sheared)
        if score > best_score:
            best_score = score
            best_angle = angle

    # the shear that best deskews the ink is the inverse of the text's own slant
    return -float(best_angle)


def classify_italic_by_label(angles_by_label: dict[str, list[float]]) -> dict[str, bool]:
    """Collapse per-block slant angles into one italic/upright verdict per
    label, using a fixed absolute angle threshold (0 degrees is a
    meaningful upright baseline, unlike bold's relative stroke-ratio)."""
    return {
        label: bool(abs(statistics.median(angles)) > ITALIC_ANGLE_THRESHOLD_DEG)
        for label, angles in angles_by_label.items()
        if angles
    }
