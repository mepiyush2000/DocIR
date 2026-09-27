"""Shared ink/background binarization for pixel-based style measurement
(bold, italic, and font-family matching all start from the same mask)."""

from __future__ import annotations

from dataclasses import dataclass

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
    result = ink_maps(source_image, bbox)
    return None if result is None else result[0]


def ink_maps(source_image: Image.Image | None, bbox) -> tuple[np.ndarray, np.ndarray] | None:
    """Like `binarize_region`, but also returns a soft ink-intensity map
    (float 0..1, 1 = full ink) of the same crop, whose anti-aliased edges
    keep sub-pixel shape detail that the hard mask throws away."""
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
    ink_is_dark = dark_frac < 0.5
    binary = (gray < thresh) if ink_is_dark else (gray >= thresh)
    ink_frac = binary.mean()
    if ink_frac == 0 or ink_frac > 0.6:
        return None
    lo, hi = np.percentile(gray, 2), np.percentile(gray, 98)
    soft = np.clip((gray - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    if ink_is_dark:
        soft = 1.0 - soft
    return binary, soft


# ---------- rotation / morphology / connected components (numpy-only) ----------

def rotate_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a boolean mask counter-clockwise, expanding the canvas."""
    if angle_deg == 0:
        return mask
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(img.rotate(angle_deg, resample=Image.NEAREST, expand=True, fillcolor=0)) > 127


def estimate_skew(mask: np.ndarray, max_deg: float = 24.0) -> float:
    """Angle (degrees) that makes text rows horizontal, via the classic
    projection-profile criterion: rows of text concentrate ink into as few
    rows as possible when level, maximizing the sum of squared row counts.
    Coarse-to-fine search keeps it cheap."""
    if not mask.any():
        return 0.0

    def energy(a):
        p = rotate_mask(mask, a).sum(axis=1).astype(np.float64)
        return float((p * p).sum())

    best = max(np.arange(-max_deg, max_deg + 1e-6, 2.0), key=energy)
    for step in (0.5, 0.25):
        span = step * 3
        best = max(np.arange(best - span, best + span + 1e-6, step), key=energy)
    return float(best)


def runs_1d(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges of consecutive True values."""
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(padded.astype(np.int8))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def close_horizontal(mask: np.ndarray, gap: int) -> np.ndarray:
    """Fill background gaps of at most `gap` px between ink pixels in the
    same row (joins letters of a word into one blob, keeps word spaces)."""
    h, w = mask.shape
    idx = np.broadcast_to(np.arange(w), (h, w))
    prev = np.maximum.accumulate(np.where(mask, idx, -(10 ** 9)), axis=1)
    nxt = np.minimum.accumulate(np.where(mask, idx, 10 ** 9)[:, ::-1], axis=1)[:, ::-1]
    return mask | ((nxt - prev) <= gap + 1)


def component_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding boxes (y0, y1, x0, x1) of 8-connected components, via
    union-find over per-row runs."""
    row_runs = [runs_1d(row) for row in mask]
    parent: list[int] = []
    boxes: list[list[int]] = []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    prev_ids: list[tuple[int, int, int]] = []
    for y, runs in enumerate(row_runs):
        cur_ids = []
        j = 0
        for s, e in runs:
            rid = len(parent)
            parent.append(rid)
            boxes.append([y, y + 1, s, e])
            while j < len(prev_ids) and prev_ids[j][1] < s:
                j += 1
            k = j
            while k < len(prev_ids) and prev_ids[k][0] <= e:
                a, b = find(prev_ids[k][2]), find(rid)
                if a != b:
                    parent[b] = a
                k += 1
            cur_ids.append((s, e, rid))
        prev_ids = cur_ids

    merged: dict[int, list[int]] = {}
    for i, (y0, y1, x0, x1) in enumerate(boxes):
        r = find(i)
        m = merged.get(r)
        if m is None:
            merged[r] = [y0, y1, x0, x1]
        else:
            m[0], m[1] = min(m[0], y0), max(m[1], y1)
            m[2], m[3] = min(m[2], x0), max(m[3], x1)
    return [tuple(b) for b in merged.values()]


# ---------- text-line geometry ----------

LINE_CORE_FRAC = 0.45  # rows above this fraction of the peak row count are x-height core
MIN_STRIP_WIDTH_PX = 60
MAX_STRIPS = 8


@dataclass(frozen=True)
class TextLineStats:
    n_lines: int
    x_height: float  # px, median height of the dense lowercase core band
    pitch: float | None  # px, median baseline-to-baseline distance (None if 1 line)


def _line_cores(profile: np.ndarray) -> list[tuple[int, int]]:
    """x-height core bands of a (deskewed) row profile: dense-row runs, with
    fragments of one line merged and stray thin runs dropped."""
    if profile.max() <= 0:
        return []
    # a robust peak, so an underline or one oversized glyph (e.g. a
    # decorative quote mark) can't make every ordinary text row look sparse
    peak = np.percentile(profile[profile > 0], 85)
    cores = runs_1d(profile > LINE_CORE_FRAC * peak)
    if not cores:
        return []
    max_h = max(e - s for s, e in cores)
    merged = [list(cores[0])]
    for s, e in cores[1:]:
        if s - merged[-1][1] <= max(1, 0.2 * max_h):
            merged[-1][1] = e
        else:
            merged.append([s, e])
    max_h = max(e - s for s, e in merged)
    return [(s, e) for s, e in merged if e - s >= 0.4 * max_h]


def measure_text_lines(binary: np.ndarray) -> TextLineStats | None:
    """Count physical text lines and measure x-height/line pitch from a
    block's ink mask. The block is split into vertical strips that are
    deskewed independently, so slanted *and* curved (warped-page) lines are
    each locally level; a short trailing line only needs to reach the
    leftmost strips to be counted."""
    h, w = binary.shape
    if not binary.any() or h < 4:
        return None
    # level the block as a whole first: a strongly rotated line leaves too
    # few glyphs in any one strip for strip-level deskew to find its angle
    binary = rotate_mask(binary, estimate_skew(binary))
    h, w = binary.shape
    block_cores = _line_cores(binary.sum(axis=1))
    # fixed-width strips, not ones sized off a first-pass line height: a
    # drop cap or an embedded picture would inflate that estimate and
    # collapse everything into one strip
    n_strips = int(np.clip(w // MIN_STRIP_WIDTH_PX, 1, MAX_STRIPS))
    edges = np.linspace(0, w, n_strips + 1).astype(int)

    strip_cores = []
    for a, b in zip(edges[:-1], edges[1:]):
        strip = binary[:, a:b]
        if not strip.any():
            continue
        cores = _line_cores(rotate_mask(strip, estimate_skew(strip)).sum(axis=1))
        if cores:
            strip_cores.append(cores)
    if not strip_cores:
        return None
    strip_typical = float(np.median([e - s for cores in strip_cores for s, e in cores]))
    # the whole-block profile anchors the line height: a strip holding a
    # picture fragment the bbox clipped in yields tall bogus "lines"
    typical = float(np.median([e - s for s, e in block_cores])) if block_cores else strip_typical
    typical = min(typical, strip_typical) if block_cores else typical
    # slivers of a neighboring block's text clipped in at the bbox edge are
    # much thinner than real lines; drop caps/pictures are much taller
    strip_cores = [[(s, e) for s, e in cores if 0.5 * typical <= e - s <= 1.6 * max(typical, strip_typical)]
                   for cores in strip_cores]
    counts = [len(cores) for cores in strip_cores if cores]
    if not counts:
        return None
    heights = [e - s for cores in strip_cores for s, e in cores]
    x_height = float(np.median([v for v in heights if v <= 1.6 * typical] or heights))
    pitches = [(s2 + e2 - s1 - e1) / 2 for cores in strip_cores for (s1, e1), (s2, e2) in zip(cores, cores[1:])]
    # a short trailing line may only reach the leftmost strip(s), so the
    # strip seeing the most lines is the one that saw all of them
    n_lines = max(counts)
    pitch = float(np.median(pitches)) if pitches else None
    return TextLineStats(n_lines=n_lines, x_height=x_height, pitch=pitch)
