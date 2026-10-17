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
RULE_MAX_HEIGHT_FRAC = 0.2      # ... in a run this thin; a taller dark run is a panel
PANEL_MIN_LINE_FRAC = 0.5       # share of a block's lines printed light on one panel
PANEL_COLOR_TOL = 80            # RGB distance from the panel colour still inside it (gradients)
PANEL_MAX_SPREAD = 40           # robust per-channel spread of a printed (textured) panel
PANEL_SEED_COVER = 0.35         # panel share of the text lines' own box (large light glyphs)
PANEL_COVER = 0.35              # a panel row / column (light text cuts into it) is this much panel
PANEL_HUE_TOL = 12              # OpenCV hue units (of 180): a gradient keeps a saturated panel's hue
PANEL_MIN_SATURATION = 90       # OpenCV saturation (of 255) for the hue test
PANEL_MAX_PAGE_FRAC = 0.5       # a larger 'panel' is tinted paper, not a printed box


def flat_field(rgb: np.ndarray) -> np.ndarray:
    """Divide out uneven lighting (camera photos): the paper's illumination
    is estimated by closing away the (dark, thin) print and blurring
    heavily. Evenly lit pages are returned unchanged."""
    h, w = rgb.shape[:2]
    k = max(5, int(w * 0.02) | 1)
    closed = cv2.morphologyEx(rgb, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    sigma = max(w, h) * ILLUMINATION_SIGMA_FRAC
    # Match GaussianBlur's float kernel, using FFT filtering for its large support.
    kernel = cv2.getGaussianKernel(int(round(sigma * 8 + 1)) | 1, sigma, cv2.CV_32F)
    illum = cv2.filter2D(cv2.filter2D(closed.astype(np.float32), -1, kernel.T), -1, kernel)
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
    rule_rows = dark.mean(axis=1) >= RULE_ROW_FILL
    for s, e in _runs(rule_rows):
        if e - s > max(2, RULE_MAX_HEIGHT_FRAC * len(rule_rows)):
            rule_rows[s:e] = False
    keep = np.broadcast_to(~rule_rows[:, None], dark.shape)
    if not keep.any():
        return None
    # light text is only on a dark panel; a grey halftone panel's dots and
    # text outnumber its gaps yet the text is still the dark class
    light_ink = dark[keep].mean() >= 0.5 and float(np.median(gray[dark & keep])) < REVERSE_MAX_BG_LUMA
    ink = (~dark if light_ink else dark) & keep
    return ink, ~ink & keep


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


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


def _line_panel(rgb: np.ndarray, line: Line, scale: float, paper: np.ndarray) -> np.ndarray | None:
    """Colour of the uniform panel behind a line of light text, if any."""
    region = _region(rgb, _line_bbox(line), scale)
    split = _ink_split(region) if region.size else None
    if split is None or not split[0].any() or not split[1].any():
        return None
    ink, bg = region[split[0]], region[split[1]]
    med = np.median(bg, axis=0)
    if med.mean() >= np.median(ink, axis=0).mean():
        return None
    # robust spread: anti-aliased glyph edges and print texture widen the std
    spread = 1.4826 * np.median(np.abs(bg.astype(float) - med), axis=0)
    if np.linalg.norm(med - paper) <= FILL_MIN_DISTANCE or spread.max() >= PANEL_MAX_SPREAD:
        return None
    return med


def _panel_extent(rgb: np.ndarray, color: np.ndarray, lines: list[Line], scale: float):
    """Page box (source pixels) of the panel of `color` the lines sit on:
    grown from the lines' box while the next row / column is mostly panel."""
    near = np.linalg.norm(rgb.astype(np.float32) - color, axis=2) < PANEL_COLOR_TOL
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ref = cv2.cvtColor(np.uint8([[np.clip(color, 0, 255)]]), cv2.COLOR_RGB2HSV)[0, 0].astype(int)
    if ref[1] >= PANEL_MIN_SATURATION:
        dh = np.abs(hsv[..., 0].astype(int) - ref[0])
        near |= (np.minimum(dh, 180 - dh) <= PANEL_HUE_TOL) & (hsv[..., 1] >= PANEL_MIN_SATURATION)
    h, w = near.shape
    boxes = [_line_bbox(l) for l in lines]
    x0, y0 = max(0, int(min(b[0] for b in boxes) * scale)), max(0, int(min(b[1] for b in boxes) * scale))
    x1 = min(w, int(np.ceil(max(b[2] for b in boxes) * scale)))
    y1 = min(h, int(np.ceil(max(b[3] for b in boxes) * scale)))
    if x1 <= x0 or y1 <= y0 or near[y0:y1, x0:x1].mean() < PANEL_SEED_COVER:
        return None
    for _ in range(2):
        while y0 > 0 and near[y0 - 1, x0:x1].mean() >= PANEL_COVER:
            y0 -= 1
        while y1 < h and near[y1, x0:x1].mean() >= PANEL_COVER:
            y1 += 1
        while x0 > 0 and near[y0:y1, x0 - 1].mean() >= PANEL_COVER:
            x0 -= 1
        while x1 < w and near[y0:y1, x1].mean() >= PANEL_COVER:
            x1 += 1
    if (x1 - x0) * (y1 - y0) > PANEL_MAX_PAGE_FRAC * h * w:
        return None
    return (x0 / scale, y0 / scale, x1 / scale, y1 / scale)


def block_panel(rgb: np.ndarray, raw: np.ndarray, block: Block, scale: float, paper: np.ndarray):
    """(colour, page box) of a printed panel behind the block's light text
    (banners, coloured header/footer bars), or None. Found on the
    flat-fielded `rgb`; measured on `raw`, since flat-fielding brightens
    large dark areas it mistakes for shadow."""
    lines = [l for l in block.lines if l.x_height > 0]
    on_panel = [l for l in lines if _line_panel(rgb, l, scale, paper) is not None]
    if not lines or len(on_panel) < PANEL_MIN_LINE_FRAC * len(lines):
        return None
    colors = []
    for l in on_panel:
        region = _region(raw, _line_bbox(l), scale)
        if (split := _ink_split(region)) is not None and split[1].any():
            colors.append(np.median(region[split[1]], axis=0))
    if not colors:
        return None
    color = np.median(np.array(colors), axis=0)
    extent = _panel_extent(raw, color, lines, scale)
    return (snap_background(color), extent) if extent is not None else None


def _merge_panels(blocks: list[Block]) -> None:
    """Blocks measured on one panel (overlapping boxes, close colours) share
    the union box and one colour."""
    panels = [b for b in blocks if b.fill_bbox is not None]
    changed = True
    while changed:
        changed = False
        for a in panels:
            for b in panels:
                if a is b or a.fill_bbox == b.fill_bbox:
                    continue
                ax0, ay0, ax1, ay1 = a.fill_bbox
                bx0, by0, bx1, by1 = b.fill_bbox
                overlap = ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1
                if overlap and np.linalg.norm(np.subtract(a.fill, b.fill)) < PANEL_COLOR_TOL:
                    box = (min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1))
                    a.fill_bbox = b.fill_bbox = box
                    b.fill = a.fill
                    changed = True


def assign_colors(page: Page, image: Image.Image) -> None:
    scale = min(1.0, ANALYSIS_WIDTH_PX / image.width)
    small = image.convert("RGB")
    if scale < 1.0:
        small = small.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
    raw = np.asarray(small)
    rgb = flat_field(raw)
    paper = page_background(rgb)
    page.background = snap_background(paper)
    for block in page.blocks:
        color, fill = block_colors(rgb, block, scale, paper)
        block.fill = fill
        if (panel := block_panel(rgb, raw, block, scale, paper)) is not None:
            block.fill, block.fill_bbox = panel
        measured = {id(l): line_color(rgb, l, scale, paper) for l in block.lines}
        body = [c for l in block.lines if not l.drop_cap and (c := measured[id(l)]) is not None]
        # one colour for a block's running text; a drop cap keeps its own
        shared = tuple(int(v) for v in np.median(np.array(body), axis=0)) if body else color
        for line in block.lines:
            line.color = (measured[id(line)] or shared) if line.drop_cap else snap_ink(shared)
    _merge_panels(page.blocks)
