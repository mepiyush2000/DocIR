"""Render-and-compare fidelity score: how closely an output PDF's ink matches
the source scan.

The output PDF is rasterized back to source-image pixels (content is placed
at a uniform `page_w / img_w` scale from the top-left, so this is a pure
resampling) and compared ink-to-ink per text block:

  precision = rendered ink lying near source ink   (nothing drawn extra)
  recall    = source ink lying near rendered ink   (nothing missing)

"Near" is a tolerance of ~0.3 x-height, so a faithful re-typeset in a close
substitute font scores ~1 while misplaced lines, wrong sizes, or wrong line
breaks do not. Picture blocks are masked out; rendered ink outside every
text block where the source has none ("stray" ink) also counts against
page precision.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from PIL import Image

from .ink import ink_maps, measure_text_lines
from .labels import PICTURE_LABELS as IMAGE_LABELS
from .schemas import Block, PageLayout

TOLERANCE_X_HEIGHT_FRAC = 0.3
MIN_TOLERANCE_PX = 1
BLOCK_PAD_FRAC = 0.25       # of the block's x-height (or height); catches slight overflow
STRAY_INK_DELTA = 80        # gray levels darker than the page background counted as ink
STRAY_TOLERANCE_PX = 3      # rendered ink this close to dark source pixels is not stray


@dataclass
class BlockScore:
    block_id: int
    label: str
    precision: float
    recall: float
    f1: float
    source_ink: int
    rendered_ink: int
    tolerance_px: int


@dataclass
class FidelityReport:
    precision: float
    recall: float
    f1: float
    stray_ink: int
    blocks: list[BlockScore] = field(default_factory=list)

    def worst(self, n: int) -> list[BlockScore]:
        return sorted(self.blocks, key=lambda b: b.f1)[:n]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else float(2 * p * r / (p + r))


def rasterize_pdf(pdf_path: str, width_px: int, height_px: int) -> Image.Image:
    """Render page 1 of `pdf_path` so its width is `width_px`, then crop/pad
    (white) to exactly `height_px` rows -- the output page may be taller
    than the content when its canvas snapped to a standard paper ratio."""
    import pypdfium2 as pdfium

    from .document import PDFIUM_LOCK

    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(pdf_path)
        try:
            page = pdf[0]
            img = page.render(scale=width_px / page.get_width()).to_pil().convert("RGB")
        finally:
            pdf.close()
    if img.width != width_px:
        img = img.resize((width_px, round(img.height * width_px / img.width)), Image.LANCZOS)
    canvas = Image.new("RGB", (width_px, height_px), "white")
    canvas.paste(img.crop((0, 0, width_px, min(height_px, img.height))), (0, 0))
    return canvas


def _clip_box(bbox, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)
    return None if x2 <= x1 or y2 <= y1 else (x1, y1, x2, y2)


def _block_ink(image: Image.Image, box) -> np.ndarray:
    maps = ink_maps(image, box)
    if maps is None:
        return np.zeros((box[3] - box[1], box[2] - box[0]), dtype=bool)
    return maps[0]


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def _picture_mask(blocks: list[Block], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for b in blocks:
        if b.label in IMAGE_LABELS and (box := _clip_box(b.bbox, width, height)):
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = True
    return mask


def _text_blocks(layout: PageLayout) -> list[Block]:
    # empty text blocks still cover real printed text
    return [b for b in layout.blocks if b.label not in IMAGE_LABELS]


def _block_geometry(block: Block, source: Image.Image) -> tuple[tuple[int, int, int, int], int] | None:
    """Padded scoring box and match tolerance, both scaled by the block's
    measured x-height (falling back to its height for unmeasurable crops)."""
    box = _clip_box(block.bbox, source.width, source.height)
    if box is None:
        return None
    maps = ink_maps(source, box)
    stats = measure_text_lines(maps[0]) if maps is not None else None
    x_height = stats.x_height if stats is not None else 0.5 * (box[3] - box[1])
    tol = max(MIN_TOLERANCE_PX, int(round(TOLERANCE_X_HEIGHT_FRAC * x_height)))
    pad = int(round(BLOCK_PAD_FRAC * x_height))
    x1, y1, x2, y2 = block.bbox
    padded = _clip_box((x1 - pad, y1 - pad, x2 + pad, y2 + pad), source.width, source.height)
    return padded, tol


@dataclass
class _Comparison:
    """Page-sized boolean maps shared by scoring and the diff image."""
    source: np.ndarray
    rendered: np.ndarray
    matched_src: np.ndarray    # source ink near rendered ink
    matched_ren: np.ndarray    # rendered ink near source ink
    covered: np.ndarray        # union of scored text regions
    stray: np.ndarray          # rendered ink outside every text region / picture
    blocks: list[BlockScore]


def _compare(layout: PageLayout, source: Image.Image, rendered: Image.Image) -> _Comparison:
    w, h = source.width, source.height
    pictures = _picture_mask(layout.blocks, w, h)
    src_all = np.zeros((h, w), dtype=bool)
    ren_all = np.zeros_like(src_all)
    m_src = np.zeros_like(src_all)
    m_ren = np.zeros_like(src_all)
    covered = np.zeros_like(src_all)
    scores: list[BlockScore] = []

    for block in _text_blocks(layout):
        geom = _block_geometry(block, source)
        if geom is None:
            continue
        box, tol = geom
        x1, y1, x2, y2 = box
        keep = ~pictures[y1:y2, x1:x2]
        src = _block_ink(source, box) & keep
        ren = _block_ink(rendered, box) & keep
        near_src = src & _dilate(ren, tol)
        near_ren = ren & _dilate(src, tol)
        n_src, n_ren = int(src.sum()), int(ren.sum())
        p = near_ren.sum() / n_ren if n_ren else 0.0
        r = near_src.sum() / n_src if n_src else (1.0 if not n_ren else 0.0)
        scores.append(BlockScore(block.id, block.label, round(float(p), 4), round(float(r), 4),
                                 round(_f1(p, r), 4), n_src, n_ren, tol))
        # overlapping blocks: OR so shared pixels count once at page level
        src_all[y1:y2, x1:x2] |= src
        ren_all[y1:y2, x1:x2] |= ren
        m_src[y1:y2, x1:x2] |= near_src
        m_ren[y1:y2, x1:x2] |= near_ren
        covered[y1:y2, x1:x2] |= keep

    gray = np.asarray(rendered.convert("L"), dtype=np.int16)
    background = int(np.median(gray))
    src_gray = np.asarray(source.convert("L"), dtype=np.int16)
    # rules, panels and bars printed outside text blocks are matched, not stray
    src_dark = _dilate(int(np.median(src_gray)) - src_gray > STRAY_INK_DELTA, STRAY_TOLERANCE_PX)
    stray = (background - gray > STRAY_INK_DELTA) & ~src_dark & ~covered & ~pictures
    return _Comparison(src_all, ren_all, m_src, m_ren, covered, stray, scores)


def _report(cmp: _Comparison) -> FidelityReport:
    n_src = int(cmp.source.sum())
    n_stray = int(cmp.stray.sum())
    n_ren = int(cmp.rendered.sum()) + n_stray
    p = cmp.matched_ren.sum() / n_ren if n_ren else 0.0
    r = cmp.matched_src.sum() / n_src if n_src else 0.0
    return FidelityReport(round(float(p), 4), round(float(r), 4), round(_f1(p, r), 4),
                          n_stray, cmp.blocks)


def score_images(layout: PageLayout, source: Image.Image, rendered: Image.Image) -> FidelityReport:
    """Score an already-rasterized `rendered` page against `source` (same size)."""
    if rendered.size != source.size:
        raise ValueError(f"rendered size {rendered.size} != source size {source.size}")
    return _report(_compare(layout, source.convert("RGB"), rendered.convert("RGB")))


def diff_image(layout: PageLayout, source: Image.Image, rendered: Image.Image) -> Image.Image:
    """Visual diff: matched ink dark gray, missing source ink red, extra
    rendered ink blue (stray ink outside text blocks too), pictures pale."""
    cmp = _compare(layout, source.convert("RGB"), rendered.convert("RGB"))
    out = np.full((source.height, source.width, 3), 255, dtype=np.uint8)
    out[_picture_mask(layout.blocks, source.width, source.height)] = (235, 235, 235)
    out[cmp.covered & ~cmp.source & ~cmp.rendered] = (250, 250, 250)
    out[cmp.matched_src | cmp.matched_ren] = (70, 70, 70)
    out[cmp.source & ~cmp.matched_src] = (220, 30, 30)
    out[(cmp.rendered & ~cmp.matched_ren) | cmp.stray] = (30, 80, 230)
    return Image.fromarray(out)


def score_pdf(
    layout: PageLayout, pdf_path: str, source: Image.Image,
) -> tuple[FidelityReport, Image.Image]:
    """Rasterize `pdf_path` at the source resolution and score it; also
    returns the rasterized page for callers that want a diff image."""
    source = source.convert("RGB")
    rendered = rasterize_pdf(pdf_path, source.width, source.height)
    return score_images(layout, source, rendered), rendered
