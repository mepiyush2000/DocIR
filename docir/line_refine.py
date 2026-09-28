"""Validate detector text lines against the page's own pixels.

A text detector occasionally returns one quad covering two or more
physical lines (tightly set or slightly blurred text). Such a quad is much
taller than its block's measured line pitch; it is rectified, its ink row
profile split into line cores, and replaced by one quad per core.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .ink import binarize_region, line_cores, measure_text_lines
from .schemas import Block, PageLayout, PageLines, TextLine

IMAGE_LABELS = {"image", "figure", "chart", "seal"}
MERGED_PITCH_RATIO = 1.6     # a quad this many line pitches tall holds several lines
PITCH_PER_X_HEIGHT = 2.3     # typical pitch when a block has a single line


def _ordered_quad(poly) -> np.ndarray:
    pts = sorted(((float(x), float(y)) for x, y in poly), key=lambda p: p[0])
    left = sorted(pts[:2], key=lambda p: p[1])
    right = sorted(pts[2:], key=lambda p: p[1])
    return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)


def _quad_height(q: np.ndarray) -> float:
    return float((np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2)


def _quad_width(q: np.ndarray) -> float:
    return float((np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2)


def _rectify(gray: np.ndarray, q: np.ndarray) -> np.ndarray:
    w, h = max(2, round(_quad_width(q))), max(2, round(_quad_height(q)))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(gray, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def _sub_quad(q: np.ndarray, v0: float, v1: float) -> list[tuple[float, float]]:
    """The band between fractions v0..v1 of the quad's height."""
    tl, tr, br, bl = q
    a0, b0 = tl + (bl - tl) * v0, tr + (br - tr) * v0
    a1, b1 = tl + (bl - tl) * v1, tr + (br - tr) * v1
    return [tuple(map(float, p)) for p in (a0, b0, b1, a1)]


def _block_pitches(layout: PageLayout, image: Image.Image) -> list[tuple[Block, float]]:
    out = []
    for b in layout.blocks:
        if b.label in IMAGE_LABELS or not b.content:
            continue
        mask = binarize_region(image, b.bbox)
        stats = measure_text_lines(mask) if mask is not None else None
        if stats is None:
            continue
        out.append((b, stats.pitch or PITCH_PER_X_HEIGHT * stats.x_height))
    return out


def _pitch_at(pitches: list[tuple[Block, float]], x: float, y: float) -> float | None:
    for b, pitch in pitches:
        x0, y0, x1, y1 = b.bbox
        if x0 <= x <= x1 and y0 <= y <= y1:
            return pitch
    return None


def split_line(gray: np.ndarray, line: TextLine, pitch: float) -> list[TextLine]:
    """One TextLine per physical line inside `line`'s quad (itself if single)."""
    q = _ordered_quad(line.polygon)
    height = _quad_height(q)
    if height < MERGED_PITCH_RATIO * pitch:
        return [line]
    crop = _rectify(gray, q)
    thresh, _ = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = crop < thresh
    cores = line_cores(ink.sum(axis=1).astype(np.float64))
    expected = max(2, round(height / pitch))
    if not 2 <= len(cores) <= expected + 1:
        # profile unusable: split evenly
        edges = np.linspace(0, 1, expected + 1)
    else:
        mids = [(e1 + s2) / 2 / crop.shape[0] for (_, e1), (s2, _) in zip(cores, cores[1:])]
        edges = np.array([0.0, *mids, 1.0])
    # the whole quad's reading can't be divided between its lines reliably
    return [TextLine(polygon=_sub_quad(q, a, b), text="", score=line.score)
            for a, b in zip(edges, edges[1:])]


def split_merged_lines(page_lines: PageLines, layout: PageLayout, image: Image.Image) -> PageLines:
    pitches = _block_pitches(layout, image)
    gray = np.asarray(image.convert("L"))
    out = []
    for line in page_lines.lines:
        q = _ordered_quad(line.polygon)
        cx, cy = q.mean(axis=0)
        pitch = _pitch_at(pitches, cx, cy)
        out.extend(split_line(gray, line, pitch) if pitch else [line])
    return page_lines.model_copy(update={"lines": out})
