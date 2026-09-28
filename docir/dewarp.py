"""Text-line dewarping for photographed pages.

Printed lines are straight on paper, so their curvature in a photo *is*
the page's distortion. Each text line is found as a blob (words closed
together), its centreline slope sampled, and a smooth slope field
dy/dx = f(x, y) fitted over the page. Tracing that field from the page's
centre column gives, for every output row, the curve it came from in the
photo; remapping along those curves makes every line level. Only rows move
vertically (a shear-free vertical remap), which is exact for the gentle
curl of a book page or a slightly tilted camera.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

WORK_WIDTH_PX = 1000
MIN_BLOB_LEN_TEXT = 8          # line blobs at least this many text heights long
MAX_BLOB_THICK_TEXT = 2.6      # and at most this many text heights thick
SAMPLE_STEP_TEXT = 2.0         # slope samples every this many text heights
POLY_DEGREE_X, POLY_DEGREE_Y = 3, 2
MIN_BLOBS = 6
MAX_ABS_SLOPE = 0.2            # ~11 degrees; steeper samples are not text lines
GRID_STEP_PX = 8


def _text_height(ink: np.ndarray) -> float | None:
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    ws = stats[1:, cv2.CC_STAT_WIDTH]
    keep = (hs >= 4) & (hs <= 80) & (ws <= 3 * hs)
    return float(np.median(hs[keep])) if keep.sum() >= 20 else None


def _slope_samples(ink: np.ndarray, s: float) -> list[tuple[float, float, float, float]]:
    """(x, y, slope, weight) along every text-line blob's centreline."""
    close = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(1.6 * s)), 1))
    blobs = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, close)
    # separate lines that touch through ascenders/descenders
    blobs = cv2.morphologyEx(blobs, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(s)), 1)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blobs, connectivity=8)
    step = max(2, int(SAMPLE_STEP_TEXT * s))
    samples = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w < MIN_BLOB_LEN_TEXT * s or h > MAX_BLOB_THICK_TEXT * s + w * MAX_ABS_SLOPE:
            continue
        comp = labels[y:y + h, x:x + w] == i
        cols = np.arange(w)
        counts = comp.sum(axis=0)
        valid = counts > 0
        if valid.sum() < MIN_BLOB_LEN_TEXT * s * 0.6:
            continue
        cy = (comp * np.arange(h)[:, None]).sum(axis=0)[valid] / counts[valid]
        cx = cols[valid]
        # the centreline must be a thin line, not a paragraph merged into one blob
        if np.median(counts[valid]) > MAX_BLOB_THICK_TEXT * s:
            continue
        for a in range(0, len(cx) - step, step):
            seg_x, seg_y = cx[a:a + 2 * step], cy[a:a + 2 * step]
            if np.ptp(seg_x) < step:
                continue
            slope = np.polyfit(seg_x, seg_y, 1)[0]
            if abs(slope) <= MAX_ABS_SLOPE:
                samples.append((float(x + seg_x.mean()), float(y + seg_y.mean()), float(slope),
                                float(np.ptp(seg_x))))
    return samples


def _design(x: np.ndarray, y: np.ndarray, w: int, h: int) -> np.ndarray:
    u, v = x / w * 2 - 1, y / h * 2 - 1
    return np.stack([u ** i * v ** j for i in range(POLY_DEGREE_X + 1)
                     for j in range(POLY_DEGREE_Y + 1)], axis=-1)


def fit_slope_field(samples, w: int, h: int):
    """Weighted least squares polynomial slope field, or None."""
    if len(samples) < (POLY_DEGREE_X + 1) * (POLY_DEGREE_Y + 1) * 2:
        return None
    x, y, s, wt = (np.array(v) for v in zip(*samples))
    a = _design(x, y, w, h) * np.sqrt(wt)[:, None]
    coef, *_ = np.linalg.lstsq(a, s * np.sqrt(wt), rcond=None)
    # one robust refit without the worst outliers (figures, stray blobs)
    resid = np.abs(_design(x, y, w, h) @ coef - s)
    keep = resid <= np.percentile(resid, 90)
    a = _design(x[keep], y[keep], w, h) * np.sqrt(wt[keep])[:, None]
    coef, *_ = np.linalg.lstsq(a, s[keep] * np.sqrt(wt[keep]), rcond=None)
    return lambda xx, yy: _design(xx, yy, w, h) @ coef


def _source_rows(field, w: int, h: int) -> np.ndarray:
    """For a coarse grid of output rows, the source y at each grid column,
    traced along the slope field from the centre column."""
    xs = np.arange(0, w + GRID_STEP_PX, GRID_STEP_PX, dtype=np.float64)
    xs[-1] = min(xs[-1], w - 1)
    ys0 = np.arange(0, h + GRID_STEP_PX, GRID_STEP_PX, dtype=np.float64)
    c = int(np.argmin(np.abs(xs - w / 2)))
    grid = np.zeros((len(ys0), len(xs)))
    grid[:, c] = ys0
    for direction in (1, -1):
        y = ys0.copy()
        j = c
        while 0 <= j + direction < len(xs):
            dx = xs[j + direction] - xs[j]
            # midpoint (RK2) step along dy/dx = field(x, y)
            k1 = field(np.full_like(y, xs[j]), y)
            k2 = field(np.full_like(y, xs[j] + dx / 2), y + k1 * dx / 2)
            y = y + k2 * dx
            j += direction
            grid[:, j] = y
    return grid


def dewarp(image: Image.Image) -> Image.Image | None:
    """The page with its text lines levelled, or None if too few lines."""
    rgb = np.asarray(image.convert("RGB"))
    H, W = rgb.shape[:2]
    scale = min(1.0, WORK_WIDTH_PX / W)
    small = cv2.resize(rgb, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    block = max(15, int(0.03 * max(gray.shape)) | 1)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                block, 15) > 0
    s = _text_height(ink)
    if s is None:
        return None
    samples = _slope_samples(ink, s)
    blobs = len({(round(x / (4 * s)), round(y / s)) for x, y, _, _ in samples})
    if blobs < MIN_BLOBS:
        return None
    h, w = gray.shape
    field = fit_slope_field(samples, w, h)
    if field is None:
        return None
    grid = _source_rows(field, w, h).astype(np.float32)
    # exact grid coordinates of every full-resolution pixel
    gx = np.broadcast_to((np.arange(W, dtype=np.float32) * scale / GRID_STEP_PX), (H, W))
    gy = np.broadcast_to((np.arange(H, dtype=np.float32) * scale / GRID_STEP_PX)[:, None], (H, W))
    map_y = cv2.remap(grid, np.ascontiguousarray(gx), np.ascontiguousarray(gy),
                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) / scale
    map_x = np.ascontiguousarray(np.broadcast_to(np.arange(W, dtype=np.float32), (H, W)))
    border = tuple(int(v) for v in np.median(np.concatenate([rgb[0], rgb[-1]]), axis=0))
    out = cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    logger.info("dewarp: %d slope samples, text height %.1fpx", len(samples), s / scale)
    return Image.fromarray(out)
