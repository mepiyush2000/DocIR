"""Text printed sideways (page numbers and stamps along a margin).

A block much taller than wide whose wording can't fit across it is one line
turned by 90 degrees. Both turns are tried: the crop is rotated upright each
way and compared with the wording rendered normally, and the line is then
measured on the upright crop and mapped back to the page.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import font_catalog
from .line_geometry import measure_line
from .page_model import Line

VERTICAL_ASPECT = 2.5     # a block this many times taller than wide may be sideways text
PROFILE_BINS = 64


def is_vertical(bbox, text: str) -> bool:
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    return w > 0 and h >= VERTICAL_ASPECT * w and len(text.strip()) >= 3 and "\n" not in text.strip()


def _profile(ink: np.ndarray) -> np.ndarray:
    p = ink.sum(axis=0).astype(np.float64)
    p = cv2.resize(p[None, :], (PROFILE_BINS, 1), interpolation=cv2.INTER_AREA)[0]
    return (p - p.mean()) / (p.std() + 1e-6)


def _rendered_profile(text: str) -> np.ndarray:
    font = ImageFont.truetype(font_catalog.face(font_catalog.FALLBACK_FAMILY).path, 48)
    img = Image.new("L", (int(font.getlength(text)) + 8, 72), 255)
    ImageDraw.Draw(img).text((4, 8), text, font=font, fill=0)
    a = np.asarray(img) < 128
    cols = np.nonzero(a.any(axis=0))[0]
    return _profile(a[:, cols[0]:cols[-1] + 1]) if cols.size else np.zeros(PROFILE_BINS)


def _ink(gray: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask > 0


def vertical_line(gray: np.ndarray, bbox, text: str) -> Line | None:
    """The block as one line turned by 90 degrees, or None if it can't be read."""
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    crop = gray[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0:
        return None
    m_rows, n_cols = crop.shape
    want = _rendered_profile(text.strip())
    best = None
    for k in (1, -1):    # 1: reads top-to-bottom (turned clockwise); -1: bottom-to-top
        upright = np.ascontiguousarray(np.rot90(crop, k))
        ink = _ink(upright)
        cols = np.nonzero(ink.any(axis=0))[0]
        if cols.size < 2:
            continue
        score = float(np.dot(_profile(ink[:, cols[0]:cols[-1] + 1]), want)) / PROFILE_BINS
        if best is None or score > best[0]:
            best = (score, k, upright)
    if best is None:
        return None
    _, k, upright = best
    h, w = upright.shape
    m = measure_line(upright, [(0, 0), (w, 0), (w, h), (0, h)])
    if m is None:
        return None
    (u0, b0), (u1, b1) = m.baseline_start, m.baseline_end
    b = (b0 + b1) / 2
    if k == 1:     # upright row -> page x falls, upright column -> page y rises
        start, end = (x0 + n_cols - 1 - b, y0 + u0), (x0 + n_cols - 1 - b, y0 + u1)
    else:          # upright row -> page x, upright column -> page y falls
        start, end = (x0 + b, y0 + m_rows - 1 - u0), (x0 + b, y0 + m_rows - 1 - u1)
    return Line(text=text.strip(), baseline_start=start, baseline_end=end,
                x_height=m.x_height, cap_top=m.cap_top)
