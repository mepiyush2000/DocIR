"""Colours measured from the page: page background, block background
panels, and text colour per block.

Photographed paper is rarely pure white and printed black is rarely pure
black; near-neutral light paper is snapped to white and near-neutral dark
ink to black so the vector output looks clean, while genuinely coloured
paper, panels, and text (coloured headings) keep their measured colour.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .page_model import BLACK, WHITE, Block, Color, Line, Page

ANALYSIS_WIDTH_PX = 1000
NEUTRAL_SATURATION = 0.18       # below this (HSV S) a colour counts as grey
PAPER_MIN_VALUE = 0.62          # neutral backgrounds at least this light become white
INK_MAX_VALUE = 0.62            # neutral inks at most this light become black
FILL_MIN_DISTANCE = 45          # RGB distance from the page colour that makes a panel
FILL_MAX_SPREAD = 28            # per-channel std of a uniform panel
INK_DARKEST_FRAC = 0.35         # blur lightens glyph edges: use the core of the strokes
UNEVEN_LIGHT_RANGE = 14         # paper luminance spread (p90 - p10) that means uneven lighting
ILLUMINATION_SIGMA_FRAC = 1 / 12
DESCENDER_BAND = 0.35           # of the x-height, below the baseline
REVERSE_MAX_BG_LUMA = 110       # a background darker than this can carry light text
RULE_ROW_FILL = 0.85            # a region row this dark across is a rule, not text


def flat_field(rgb: np.ndarray) -> np.ndarray:
    """Divide out uneven lighting (camera photos): the paper's illumination
    is estimated by closing away the (dark, thin) print and blurring
    heavily. Evenly lit pages are returned unchanged."""
    h, w = rgb.shape[:2]
    k = max(5, int(w * 0.02) | 1)
    closed = cv2.morphologyEx(rgb, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    illum = cv2.GaussianBlur(closed.astype(np.float32), (0, 0), max(w, h) * ILLUMINATION_SIGMA_FRAC)
    lum = illum.mean(axis=2)
    if np.percentile(lum, 90) - np.percentile(lum, 10) < UNEVEN_LIGHT_RANGE:
        return rgb
    paper = np.percentile(illum.reshape(-1, 3), 90, axis=0)
    out = rgb.astype(np.float32) / np.maximum(illum, 1.0) * paper
    return np.clip(out, 0, 255).astype(np.uint8)


def _hsv(rgb: np.ndarray) -> tuple[float, float, float]:
    r, g, b = (float(v) / 255 for v in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    s = 0.0 if mx == 0 else (mx - mn) / mx
    return 0.0, s, mx


def snap_background(rgb) -> Color:
    _, s, v = _hsv(np.asarray(rgb))
    if s < NEUTRAL_SATURATION and v >= PAPER_MIN_VALUE:
        return WHITE
    return tuple(int(c) for c in rgb)


def snap_ink(rgb) -> Color:
    _, s, v = _hsv(np.asarray(rgb))
    if s < NEUTRAL_SATURATION * 1.5 and v <= INK_MAX_VALUE:
        return BLACK
    return tuple(int(c) for c in rgb)


def _ink_split(region: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(ink, background) boolean masks of an RGB region, ink = minority."""
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    if gray.size < 16 or int(gray.max()) - int(gray.min()) < 24:
        return None
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = gray < t
    # rules crossing the region are neither text nor background
    keep = np.broadcast_to((dark.mean(axis=1) < RULE_ROW_FILL)[:, None], dark.shape)
    if not keep.any():
        return None
    # light text is only on a dark panel; a grey halftone panel's dots and
    # text outnumber its gaps yet the text is still the dark class
    light_ink = dark[keep].mean() >= 0.5 and float(np.median(gray[dark & keep])) < REVERSE_MAX_BG_LUMA
    ink = (~dark if light_ink else dark) & keep
    return ink, ~ink & keep


def page_background(rgb: np.ndarray) -> np.ndarray:
    """Measured (unsnapped) paper colour."""
    split = _ink_split(rgb)
    pixels = rgb[split[1]] if split is not None else rgb.reshape(-1, 3)
    return np.median(pixels, axis=0)


def _region(rgb: np.ndarray, bbox, scale: float) -> np.ndarray:
    x0, y0, x1, y1 = (int(round(v * scale)) for v in bbox)
    h, w = rgb.shape[:2]
    return rgb[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]


def block_colors(rgb: np.ndarray, block: Block, scale: float,
                 paper: np.ndarray) -> tuple[Color, Color | None]:
    """(text colour, background fill or None) of one block; `paper` is the
    measured page colour."""
    region = _region(rgb, block.bbox, scale)
    split = _ink_split(region) if region.size else None
    if split is None:
        return BLACK, None
    ink_mask, bg_mask = split
    ink, bg = region[ink_mask], region[bg_mask]
    fill = None
    if len(bg):
        med = np.median(bg, axis=0)
        if np.linalg.norm(med - paper) > FILL_MIN_DISTANCE and bg.std(axis=0).max() < FILL_MAX_SPREAD:
            fill = snap_background(med)
            if fill == snap_background(paper):
                fill = None
    if not len(ink):
        return BLACK, fill
    return _core_color(ink, bg, paper), fill


def _core_color(ink: np.ndarray, bg: np.ndarray, paper: np.ndarray) -> Color:
    # stroke cores: the ink pixels furthest from the local background
    ref = np.median(bg, axis=0) if len(bg) else paper
    dist = np.linalg.norm(ink.astype(float) - ref, axis=1)
    core = ink[dist >= np.quantile(dist, 1 - INK_DARKEST_FRAC)]
    return snap_ink(white_balance(np.median(core, axis=0), ref))


def white_balance(color, local_paper) -> np.ndarray:
    """Ink colour as it would look on white paper: tinted paper (yellow
    receipts, warm lighting) casts its hue onto black print."""
    paper = np.maximum(np.asarray(local_paper, dtype=float), 1.0)
    return np.clip(np.asarray(color, dtype=float) * 255.0 / paper, 0, 255)


def _line_bbox(line: Line) -> tuple[float, float, float, float]:
    (x0, y0), (x1, y1) = line.baseline_start, line.baseline_end
    return (min(x0, x1), min(y0, y1) - line.cap_top, max(x0, x1),
            max(y0, y1) + DESCENDER_BAND * line.x_height)


def line_color(rgb: np.ndarray, line: Line, scale: float, paper: np.ndarray) -> Color | None:
    region = _region(rgb, _line_bbox(line), scale)
    split = _ink_split(region) if region.size else None
    if split is None or not split[0].any():
        return None
    return _core_color(region[split[0]], region[split[1]], paper)


def assign_colors(page: Page, image: Image.Image) -> None:
    scale = min(1.0, ANALYSIS_WIDTH_PX / image.width)
    small = image.convert("RGB")
    if scale < 1.0:
        small = small.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
    rgb = flat_field(np.asarray(small))
    paper = page_background(rgb)
    page.background = snap_background(paper)
    for block in page.blocks:
        color, fill = block_colors(rgb, block, scale, paper)
        block.fill = fill
        measured = {id(l): line_color(rgb, l, scale, paper) for l in block.lines}
        body = [c for l in block.lines if not l.drop_cap and (c := measured[id(l)]) is not None]
        # one colour for a block's running text; a drop cap keeps its own
        shared = tuple(int(v) for v in np.median(np.array(body), axis=0)) if body else color
        for line in block.lines:
            line.color = (measured[id(line)] or shared) if line.drop_cap else snap_ink(shared)
