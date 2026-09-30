"""Non-text vector graphics: horizontal/vertical rules (separators,
underlines, table and box edges) found by morphology on the page's ink
once text lines and pictures are masked out.

Slightly curved rules (photographed pages) come out as a few short
straight segments, each following its part of the curve.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .color import snap_ink
from .page_model import Graphic, Page

MIN_RULE_FRAC = 0.06          # of the page width (horizontal) / height (vertical)
MIN_RULE_ASPECT = 25          # length / thickness
EDGE_MARGIN_FRAC = 0.03       # rules this close to the page edge are scan shadows
PICTURE_EDGE_FRAC = 0.015     # rules this close to a picture are its border
TABLE_GRID_MIN_SPAN = 0.25    # of a table's width/height: shorter strokes inside it are glyphs
FILL_MIN_DARKENING = 40       # gray levels below the table's paper that count as shading
FILL_ROW_PERCENTILE = 30      # a shaded row is dark at this percentile; a text row is still paper
FILL_MIN_ROW_COVER = 0.35     # of the table's width a band must span
FILL_MIN_HEIGHT_XH = 1.2      # thinner shaded runs are rules, not bands
MAX_RULE_THICKNESS_FRAC = 0.006
TEXT_MASK_PAD_FRAC = 0.35     # of a line's x-height, around each line's box
SEGMENT_FRAC = 0.12           # long rules are fitted in pieces this long (page width)
ADAPTIVE_BLOCK_FRAC = 0.02
ADAPTIVE_C = 12


def _ink(gray: np.ndarray) -> np.ndarray:
    block = max(15, int(ADAPTIVE_BLOCK_FRAC * max(gray.shape)) | 1)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                 block, ADAPTIVE_C) > 0


def _text_and_picture_mask(page: Page, shape) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for block in page.blocks:
        for line in block.lines:
            pad = TEXT_MASK_PAD_FRAC * line.x_height
            (x0, y0), (x1, y1) = line.baseline_start, line.baseline_end
            top = line.cap_top + pad
            desc = 0.5 * line.x_height + pad
            poly = np.array([[x0 - pad, y0 - top], [x1 + pad, y1 - top],
                             [x1 + pad, y1 + desc], [x0 - pad, y0 + desc]], dtype=np.int32)
            cv2.fillPoly(mask, [poly], 1)
    for pic in page.pictures:
        x0, y0, x1, y1 = (int(round(v)) for v in pic.bbox)
        mask[max(0, y0):y1, max(0, x0):x1] = 1
    return mask.astype(bool)


def _segments(component: np.ndarray, x0: int, y0: int, horizontal: bool, seg_len: float):
    """Straight pieces (p0, p1, thickness) along one rule component."""
    ys, xs = np.nonzero(component)
    along, across = (xs, ys) if horizontal else (ys, xs)
    lo, hi = along.min(), along.max() + 1
    n = max(1, int(round((hi - lo) / seg_len)))
    edges = np.linspace(lo, hi, n + 1)
    out = []
    for a, b in zip(edges, edges[1:]):
        sel = (along >= a) & (along < b) if b < hi else (along >= a)
        if sel.sum() < 2:
            continue
        coef = np.polyfit(along[sel], across[sel], 1) if np.ptp(along[sel]) > 0 else (0.0, across[sel].mean())
        thickness = sel.sum() / max(np.ptp(along[sel]) + 1, 1)
        p0 = (a, np.polyval(coef, a))
        p1 = (b, np.polyval(coef, b))
        if not horizontal:
            p0, p1 = p0[::-1], p1[::-1]
        out.append(((p0[0] + x0, p0[1] + y0), (p1[0] + x0, p1[1] + y0), float(thickness)))
    return out


def _plausible_rule(page: Page, bbox, w: int, h: int) -> bool:
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    in_table = any(b.tabular and b.bbox[0] <= cx <= b.bbox[2] and b.bbox[1] <= cy <= b.bbox[3]
                   for b in page.blocks)
    mx, my = w * EDGE_MARGIN_FRAC, h * EDGE_MARGIN_FRAC
    # a full-width table's border legitimately runs close to the page edge
    if not in_table and (x0 < mx or y0 < my or x1 > w - mx or y1 > h - my):
        return False
    # a picture's own border (photo edges) is part of the picture
    pad = PICTURE_EDGE_FRAC * w
    for p in page.pictures:
        px0, py0, px1, py1 = p.bbox
        if px0 - pad <= x1 and x0 <= px1 + pad and py0 - pad <= y1 and y0 <= py1 + pad:
            return False
    # nor may it run through a text line (strike-through artefacts)
    for b in page.blocks:
        for l in b.lines:
            lx0, lx1 = sorted((l.baseline_start[0], l.baseline_end[0]))
            ly = (l.baseline_start[1] + l.baseline_end[1]) / 2
            if lx0 < x1 and x0 < lx1 and ly - l.cap_top <= cy <= ly:
                return False
    # text residue (glyph parts the line masks missed) sits inside text
    # blocks -- except a table's own grid, which spans a good part of it
    for b in page.blocks:
        if not (b.bbox[0] <= cx <= b.bbox[2] and b.bbox[1] <= cy <= b.bbox[3]):
            continue
        # a rebuilt table draws its own clean grid
        if b.table is not None:
            return False
        if not b.tabular:
            return False
        horizontal = (x1 - x0) >= (y1 - y0)
        span, full = ((x1 - x0), b.bbox[2] - b.bbox[0]) if horizontal else ((y1 - y0), b.bbox[3] - b.bbox[1])
        if span < TABLE_GRID_MIN_SPAN * full:
            return False
    return True


def detect_rules(page: Page, image: Image.Image) -> list[Graphic]:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = _ink(gray) & ~_text_and_picture_mask(page, gray.shape)
    h, w = gray.shape
    max_thick = max(2, int(MAX_RULE_THICKNESS_FRAC * max(w, h)))
    graphics = []
    for horizontal in (True, False):
        length = int(MIN_RULE_FRAC * (w if horizontal else h))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1) if horizontal else (1, length))
        # a gently curved or tilted rule is still long in a slightly thicker band
        closed = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        lines = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            extent, thick = (bw, bh) if horizontal else (bh, bw)
            if extent < length or area / extent > max_thick or extent * extent / area < MIN_RULE_ASPECT:
                continue
            if not _plausible_rule(page, (x, y, x + bw, y + bh), w, h):
                continue
            comp = labels[y:y + bh, x:x + bw] == i
            color = snap_ink(np.median(rgb[y:y + bh, x:x + bw][comp], axis=0))
            for p0, p1, t in _segments(comp, x, y, horizontal, SEGMENT_FRAC * w):
                graphics.append(Graphic(
                    kind="rule", color=color, stroke_px=max(1.0, t), points=[p0, p1],
                    bbox=(min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1]))))
    return graphics


def detect_table_fills(page: Page, image: Image.Image) -> list[Graphic]:
    """Shaded bands inside tables (header rows, section bars), often printed
    as a halftone. A row counts as shaded when most of its pixels -- not
    just its glyphs -- are darker than the table's paper: in a text row the
    lower percentile is still paper, in a shaded band it is not. Each band
    becomes one filled rectangle in its average colour."""
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fills = []
    for b in page.blocks:
        if not b.tabular or not b.lines:
            continue
        x0, y0, x1, y1 = (int(round(v)) for v in b.bbox)
        x0, y0 = max(0, x0), max(0, y0)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        xh = float(np.median([l.x_height for l in b.lines]))
        paper = float(np.percentile(crop, 95))
        shaded = np.percentile(crop, FILL_ROW_PERCENTILE, axis=1) < paper - FILL_MIN_DARKENING
        for s, e in _runs(shaded):
            if e - s < FILL_MIN_HEIGHT_XH * xh:
                continue
            cols = np.where(np.percentile(crop[s:e], FILL_ROW_PERCENTILE, axis=0)
                            < paper - FILL_MIN_DARKENING)[0]
            if cols.size < FILL_MIN_ROW_COVER * crop.shape[1]:
                continue
            band = (x0 + cols[0], y0 + s, x0 + cols[-1] + 1, y0 + e)
            region = rgb[band[1]:band[3], band[0]:band[2]].reshape(-1, 3)
            color = tuple(int(v) for v in region.mean(axis=0))
            fills.append(Graphic(kind="fill", bbox=tuple(float(v) for v in band), color=color))
    return fills


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(padded)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def detect_graphics(page: Page, image: Image.Image) -> list[Graphic]:
    return detect_table_fills(page, image) + detect_rules(page, image)
