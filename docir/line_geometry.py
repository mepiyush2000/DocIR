"""Pixel-exact geometry of individual text lines.

`measure_line` refines any line quad (from a detector or from
`pixel_lines`) against the page's own ink: where its baseline runs, how
tall its lowercase letters are, and where its ink starts and ends -- the
three things a renderer needs to put a line of text exactly where the
original one was.

`pixel_lines` segments a block into line quads from its ink alone; it is
the fallback when detector lines are missing or disagree with the pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .ink import binarize_region, estimate_skew, line_cores

QUAD_PAD_FRAC = 0.15          # vertical padding around a quad before measuring
ASCENDER_X_HEIGHT = 0.85      # band above the x-height core reached by ascenders/caps
DESCENDER_X_HEIGHT = 0.45     # band below the baseline reached by descenders
MIN_INK_COL_FRAC = 0.02       # of the band's height: columns with less ink are empty
MAX_SKEW_DEG = 12.0
STRIP_MIN_XH = 12             # strip width in x-heights (enough glyphs for a clear core)
MAX_STRIPS = 8

Point = tuple[float, float]


@dataclass(frozen=True)
class LineMetrics:
    baseline_start: Point     # page pixels, at the first ink column
    baseline_end: Point       # page pixels, at the last ink column
    x_height: float           # px
    cap_top: float            # px above the baseline reached by the tallest ink

    @property
    def angle_deg(self) -> float:
        """Counter-clockwise tilt of the baseline (page y grows downwards)."""
        (x0, y0), (x1, y1) = self.baseline_start, self.baseline_end
        return float(np.degrees(np.arctan2(y0 - y1, max(x1 - x0, 1e-6))))

    @property
    def length(self) -> float:
        (x0, y0), (x1, y1) = self.baseline_start, self.baseline_end
        return float(np.hypot(x1 - x0, y1 - y0))


def ordered_quad(poly) -> np.ndarray:
    """TL, TR, BR, BL of any 4-point polygon."""
    pts = sorted(((float(x), float(y)) for x, y in poly), key=lambda p: p[0])
    left = sorted(pts[:2], key=lambda p: p[1])
    right = sorted(pts[2:], key=lambda p: p[1])
    return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)


def quad_height(q: np.ndarray) -> float:
    return float((np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2)


def quad_width(q: np.ndarray) -> float:
    return float((np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2)


def _pad_quad(q: np.ndarray, frac: float) -> np.ndarray:
    tl, tr, br, bl = q
    dl, dr = (bl - tl) * frac, (br - tr) * frac
    return np.array([tl - dl, tr - dr, br + dr, bl + dl], dtype=np.float32)


def _ink_mask(gray: np.ndarray) -> np.ndarray | None:
    if gray.size == 0 or gray.max() - gray.min() < 24:
        return None
    thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = gray < thresh
    # text is the minority class, whichever side of the threshold it is on
    return dark if dark.mean() < 0.5 else ~dark


def _core(profile: np.ndarray) -> tuple[int, int] | None:
    """The x-height band: the heaviest dense-row run."""
    cores = line_cores(profile.astype(np.float64))
    if not cores:
        return None
    return max(cores, key=lambda c: profile[c[0]:c[1]].sum())


def measure_line(gray: np.ndarray, polygon) -> LineMetrics | None:
    """Baseline, x-height and ink extent of the single text line in `polygon`
    (`gray` is the whole page, uint8)."""
    q = _pad_quad(ordered_quad(polygon), QUAD_PAD_FRAC)
    w, h = max(2, round(quad_width(q))), max(2, round(quad_height(q)))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    rect = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(q, dst), (w, h),
                               flags=cv2.INTER_LINEAR, borderValue=int(np.median(gray)))
    ink = _ink_mask(rect)
    if ink is None:
        return None
    core = _core(ink.sum(axis=1))
    if core is None:
        return None
    top, base = core
    x_height = float(base - top)
    band = ink[max(0, int(top - ASCENDER_X_HEIGHT * x_height)):
               min(h, int(base + DESCENDER_X_HEIGHT * x_height) + 1)]
    cols = np.where(band.sum(axis=0) > max(1, MIN_INK_COL_FRAC * band.shape[0]))[0]
    if cols.size == 0:
        return None
    rows = np.where(ink[:base].any(axis=1))[0]
    cap_top = float(base - rows[0]) if rows.size else x_height
    # rectified (u, v) -> page: bilinear on the padded quad
    tl, tr, br, bl = q

    def to_page(u: float, v: float) -> Point:
        a, b = u / w, v / h
        top_pt, bot_pt = tl + (tr - tl) * a, bl + (br - bl) * a
        p = top_pt + (bot_pt - top_pt) * b
        return float(p[0]), float(p[1])

    scale_v = quad_height(q) / h
    return LineMetrics(baseline_start=to_page(float(cols[0]), float(base)),
                       baseline_end=to_page(float(cols[-1] + 1), float(base)),
                       x_height=x_height * scale_v, cap_top=cap_top * scale_v)


def measure_glyph(gray: np.ndarray, polygon) -> tuple[Point, Point, float] | None:
    """(baseline start, baseline end, ink height) of a single large glyph
    (drop cap) -- its x-height band is meaningless, its ink box is not."""
    q = ordered_quad(polygon)
    w, h = max(2, round(quad_width(q))), max(2, round(quad_height(q)))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    rect = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(q, dst), (w, h),
                               flags=cv2.INTER_LINEAR, borderValue=int(np.median(gray)))
    ink = _ink_mask(rect)
    if ink is None or not ink.any():
        return None
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    tl, tr, br, bl = q

    def to_page(u: float, v: float) -> Point:
        a, b = u / w, v / h
        top_pt, bot_pt = tl + (tr - tl) * a, bl + (br - bl) * a
        p = top_pt + (bot_pt - top_pt) * b
        return float(p[0]), float(p[1])

    base = float(rows[-1] + 1)
    return (to_page(float(cols[0]), base), to_page(float(cols[-1] + 1), base),
            (rows[-1] + 1 - rows[0]) * quad_height(q) / h)


def _rotation(shape: tuple[int, int], angle_deg: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Affine rotating an image counter-clockwise onto an expanded canvas."""
    h, w = shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(np.ceil(h * sin + w * cos)), int(np.ceil(h * cos + w * sin))
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return m, (nw, nh)


def _strip_cores(level: np.ndarray, n_strips: int) -> list[tuple[float, list[tuple[int, int]]]]:
    """(strip centre x, x-height cores) per vertical strip."""
    edges = np.linspace(0, level.shape[1], n_strips + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        prof = level[:, a:b].sum(axis=1).astype(np.float64)
        if prof.max() > 0:
            out.append(((a + b) / 2, line_cores(prof)))
    return out


def _chain_lines(strips, pitch: float) -> list[list[tuple[float, float, float]]]:
    """Link cores strip to strip into lines: each line is a list of
    (x, centre y, core height). Starts from the strip seeing most lines and
    walks outwards, following each line's drift (tilt / curvature)."""
    if not strips:
        return []
    ref = max(range(len(strips)), key=lambda i: (len(strips[i][1]), -i))
    x_ref, cores = strips[ref]
    lines = [[(x_ref, (s + e) / 2, e - s)] for s, e in cores]
    for order in (range(ref + 1, len(strips)), range(ref - 1, -1, -1)):
        last = [l[-1] if order.step > 0 else l[0] for l in lines]
        for i in order:
            x, cs = strips[i]
            taken = set()
            for k, (_, cy, _) in enumerate(last):
                best = None
                for j, (s, e) in enumerate(cs):
                    d = abs((s + e) / 2 - cy)
                    if j not in taken and d < 0.5 * pitch and (best is None or d < best[0]):
                        best = (d, j)
                if best is not None:
                    taken.add(best[1])
                    s, e = cs[best[1]]
                    pt = (x, (s + e) / 2, e - s)
                    if order.step > 0:
                        lines[k].append(pt)
                    else:
                        lines[k].insert(0, pt)
                    last[k] = pt
    return lines


def pixel_lines(image: Image.Image, bbox) -> list[list[Point]]:
    """Line quads of the block at `bbox`, segmented from its ink. The block
    is levelled as a whole, then x-height cores are found per vertical strip
    and chained into lines; each line gets its own fitted tilt (so lines of
    a gently curved page don't merge), ascender/descender room, and is
    trimmed to its ink before being mapped back onto the page."""
    x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    mask = binarize_region(image, (x1, y1, bbox[2], bbox[3]))
    if mask is None or not mask.any():
        return []
    angle = estimate_skew(mask, max_deg=MAX_SKEW_DEG)
    m, size = _rotation(mask.shape, angle)
    level = cv2.warpAffine(mask.astype(np.uint8), m, size, flags=cv2.INTER_NEAREST) > 0
    inv = cv2.invertAffineTransform(m)
    whole = line_cores(level.sum(axis=1).astype(np.float64))
    if not whole:
        return []
    xh0 = float(np.median([e - s for s, e in whole]))
    centres = [(s + e) / 2 for s, e in whole]
    pitch = float(np.median(np.diff(centres))) if len(centres) > 1 else 2.5 * xh0
    n_strips = int(np.clip(level.shape[1] // max(STRIP_MIN_XH * xh0, 1), 1, MAX_STRIPS))
    strips = _strip_cores(level, n_strips)
    typical = float(np.median([e - s for _, cs in strips for s, e in cs] or [xh0]))
    # slivers of neighbouring text and drop caps / pictures are not lines
    strips = [(x, [(s, e) for s, e in cs if 0.5 * typical <= e - s <= 1.8 * typical])
              for x, cs in strips]
    quads = []
    h_img, w_img = level.shape
    for pts in _chain_lines(strips, max(pitch, typical)):
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        xh = float(np.median([p[2] for p in pts]))
        slope, icpt = np.polyfit(xs, ys, 1) if len(pts) > 1 else (0.0, ys[0])
        up, down = (0.5 + ASCENDER_X_HEIGHT) * xh, (0.5 + DESCENDER_X_HEIGHT) * xh
        cols = np.arange(w_img)
        mid = slope * cols + icpt
        band = np.zeros_like(level)
        for c0 in range(0, w_img, 8):
            c1 = min(w_img, c0 + 8)
            yc = float(mid[c0:c1].mean())
            band[max(0, int(yc - up)):min(h_img, int(yc + down) + 1), c0:c1] = True
        ink_cols = np.where((level & band).any(axis=0))[0]
        if ink_cols.size == 0:
            continue
        a, b = float(ink_cols[0]), float(ink_cols[-1] + 1)
        ya, yb = slope * a + icpt, slope * b + icpt
        corners = np.array([[a, ya - up], [b, yb - up], [b, yb + down], [a, ya + down]], dtype=np.float64)
        page = corners @ inv[:, :2].T + inv[:, 2] + (x1, y1)
        quads.append([(float(x), float(y)) for x, y in page])
    return quads
