"""Page preprocessing: make the page flat and upright *before* anything is
measured, so every downstream coordinate (layout blocks, text lines,
typography, scoring) refers to one clean page image.

Only pages that need it are touched -- UVDoc resamples the whole image, so
a page that is already flat keeps its original pixels:

  * wrong orientation (90/180/270)  -> PaddleOCR doc-orientation classifier
  * curved / warped (camera photos)  -> UVDoc unwarping
  * uniformly skewed only            -> plain rotation
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

from .config import paddle_device_kwargs
from .dewarp import dewarp
from .ink import binarize_region, estimate_skew

logger = logging.getLogger(__name__)

ANALYSIS_WIDTH_PX = 900        # distortion is measured on a downscaled copy
TILE_GRID = (3, 4)             # columns x rows
MIN_TILE_INK_FRAC = 0.01
MIN_MEASURED_TILES = 3
CURVATURE_DEG = 2.0            # spread of per-column skew that means "curved"
SKEW_DEG = 0.5                 # uniform skew worth rotating away
ORIENTATION_MIN_SCORE = 0.85


@dataclass(frozen=True)
class Distortion:
    skew_deg: float        # median counter-clockwise rotation that levels text rows
    curvature_deg: float   # max - min of the per-column median angles
    tiles_measured: int

    @property
    def is_curved(self) -> bool:
        return self.curvature_deg > CURVATURE_DEG

    @property
    def is_skewed(self) -> bool:
        return abs(self.skew_deg) > SKEW_DEG


@dataclass(frozen=True)
class PreprocessResult:
    image_path: str        # the page every later step should use
    changed: bool
    rotation_deg: int      # orientation fix applied (0/90/180/270)
    unwarped: bool
    deskew_deg: float
    distortion: Distortion | None


def measure_distortion(image: Image.Image) -> Distortion | None:
    """Per-tile text-row angles on a downscaled page. A flat page's tiles
    agree; a curved page's left and right columns tilt opposite ways."""
    scale = min(1.0, ANALYSIS_WIDTH_PX / image.width)
    small = image.convert("RGB").resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    cols, rows = TILE_GRID
    xs = np.linspace(0, small.width, cols + 1).astype(int)
    ys = np.linspace(0, small.height, rows + 1).astype(int)
    per_col: list[list[float]] = [[] for _ in range(cols)]
    for ci in range(cols):
        for ri in range(rows):
            mask = binarize_region(small, (xs[ci], ys[ri], xs[ci + 1], ys[ri + 1]))
            if mask is None or mask.mean() < MIN_TILE_INK_FRAC:
                continue
            per_col[ci].append(estimate_skew(mask, max_deg=20.0))
    angles = [a for col in per_col for a in col]
    if len(angles) < MIN_MEASURED_TILES:
        return None
    col_medians = [float(np.median(col)) for col in per_col if col]
    return Distortion(
        skew_deg=float(np.median(angles)),
        curvature_deg=max(col_medians) - min(col_medians) if len(col_medians) > 1 else 0.0,
        tiles_measured=len(angles),
    )


@lru_cache(maxsize=1)
def _orientation_model():
    from paddleocr import DocImgOrientationClassification  # optional dependency

    return DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori", **paddle_device_kwargs())


@lru_cache(maxsize=1)
def _unwarp_model():
    from paddleocr import TextImageUnwarping  # optional dependency

    return TextImageUnwarping(model_name="UVDoc", **paddle_device_kwargs())


def detect_orientation(image_path: str) -> int:
    """Counter-clockwise rotation (0/90/180/270) that makes the page upright,
    if the classifier is confident (PaddleX applies its label the same way)."""
    res = next(iter(_orientation_model().predict(image_path))).json["res"]
    label, score = int(res["label_names"][0]), float(res["scores"][0])
    return label if score >= ORIENTATION_MIN_SCORE else 0


def unwarp(image_path: str) -> Image.Image:
    res = next(iter(_unwarp_model().predict(image_path)))
    bgr = np.asarray(res["doctr_img"]).astype(np.uint8)
    return Image.fromarray(bgr[..., ::-1].copy())


def _border_color(image: Image.Image) -> tuple[int, ...]:
    arr = np.asarray(image.convert("RGB"))
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    return tuple(int(v) for v in np.median(border, axis=0))


def flat_image_path(image_path: str) -> str:
    stem, _ = os.path.splitext(image_path)
    return f"{stem}_flat.png"


def flatten_page(image_path: str, output_path: str | None = None, mode: str = "auto") -> PreprocessResult:
    """Return the page image downstream steps should use.

    mode: "auto" fixes only what is detected, "off" never changes the page,
    "force" always unwarps. A changed page is written to `output_path`
    (default `<image>_flat.png`); an untouched page keeps its own path.
    """
    if mode == "off":
        return PreprocessResult(image_path, False, 0, False, 0.0, None)
    output_path = output_path or flat_image_path(image_path)
    image = Image.open(image_path).convert("RGB")
    work_path = image_path

    rotation = detect_orientation(image_path)
    if rotation:
        image = image.rotate(rotation, expand=True)
        image.save(output_path)
        work_path = output_path

    distortion = measure_distortion(image)
    unwarped, deskew = False, 0.0
    if mode == "force" or (distortion is not None and distortion.is_curved):
        # text-line dewarping keeps the photo sharp; UVDoc is the fallback
        # for pages with too little running text to trace
        flat = dewarp(image)
        after = measure_distortion(flat) if flat is not None else None
        if flat is None or after is None or (
                distortion is not None and after.curvature_deg >= distortion.curvature_deg):
            flat = unwarp(work_path)
            after = measure_distortion(flat)
        image = flat
        unwarped = True
        if after is not None and after.is_skewed:
            deskew = after.skew_deg
    elif distortion is not None and distortion.is_skewed:
        deskew = distortion.skew_deg
    if deskew:
        image = image.rotate(deskew, resample=Image.BICUBIC, fillcolor=_border_color(image))

    changed = bool(rotation or unwarped or deskew)
    if changed:
        image.save(output_path)
    logger.info("preprocess %s: %s rotation=%d unwarped=%s deskew=%.2f distortion=%s",
                image_path, "->" + output_path if changed else "unchanged",
                rotation, unwarped, deskew, distortion)
    return PreprocessResult(output_path if changed else image_path, changed, rotation,
                            unwarped, deskew, distortion)
