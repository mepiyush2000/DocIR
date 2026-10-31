"""High-level entry points wiring preprocessing -> OCR -> layout schema ->
PDF rendering (and scoring)."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import statistics
import time

from PIL import Image

from .config import line_ocr_enabled, preprocess_mode, render_mode
from .document import image_page_pdf, is_pdf, merge_pdfs, pdf_page_images
from .ink import binarize_region, measure_text_lines
from .labels import PICTURE_LABELS as IMAGE_LABELS
from .ocr import LineOcrEngine, OcrEngine, PaddleLineOcrEngine, PaddleOcrVlEngine
from .preprocess import flatten_page
from .render_pdf import render_page_to_pdf
from .schemas import PageLayout, PageLines

logger = logging.getLogger(__name__)

ASPECT_TOLERANCE = 0.01          # a resized copy of the analysed image keeps its aspect ratio


def analyze_image(image_path: str, engine: OcrEngine | None = None) -> PageLayout:
    """Run layout detection + OCR on a raw image, returning our versioned
    PageLayout schema (never a PaddleOCR-specific object)."""
    engine = engine or PaddleOcrVlEngine()
    return engine.analyze(image_path)


def _resolve_source_image(source_image_path: str | None) -> Image.Image | None:
    if not source_image_path:
        return None
    if not os.path.isfile(source_image_path):
        logger.warning(
            "Could not locate source image '%s'; rendering all text at regular "
            "weight and image blocks as placeholders.", source_image_path,
        )
        return None
    return Image.open(source_image_path).convert("RGB")


def estimate_text_height(layout: PageLayout, image: Image.Image) -> float | None:
    """Median x-height (px) over the page's text blocks."""
    heights = []
    for b in layout.blocks:
        if b.label in IMAGE_LABELS or not b.content:
            continue
        mask = binarize_region(image, b.bbox)
        stats = measure_text_lines(mask) if mask is not None else None
        if stats is not None:
            heights.append(stats.x_height)
    return statistics.median(heights) if heights else None


def lines_json_path(layout_json_path: str) -> str:
    stem, ext = os.path.splitext(layout_json_path)
    if stem.endswith("_res"):
        stem = stem[: -len("_res")]
    return f"{stem}_lines{ext or '.json'}"


def load_lines(path: str) -> PageLines:
    with open(path) as f:
        return PageLines.model_validate_json(f.read())


def detect_lines(
    layout: PageLayout, image_path: str, cache_path: str | None = None,
    engine: LineOcrEngine | None = None,
) -> PageLines | None:
    """Physical text lines for the page: the cached result when present,
    otherwise a fresh line-OCR pass (saved to `cache_path`)."""
    if cache_path and os.path.isfile(cache_path):
        return load_lines(cache_path)
    if engine is None and not line_ocr_enabled():
        return None
    try:
        engine = engine or PaddleLineOcrEngine()
        image = Image.open(image_path).convert("RGB")
        lines = engine.detect_lines(image_path, text_height_px=estimate_text_height(layout, image))
    except ImportError:
        logger.warning("line OCR unavailable (install the 'ocr' extra); using block layout only")
        return None
    if cache_path:
        with open(cache_path, "w") as f:
            f.write(lines.model_dump_json(indent=1))
    logger.info("detected %d text lines in %s", len(lines.lines), image_path)
    return lines


def render_pdf_from_layout(
    layout: PageLayout, output_path: str, source_image_path: str | None = None,
    page_lines: PageLines | None = None, mode: str | None = None,
) -> str:
    """Render `layout` to `output_path`, resolving the source image from
    `source_image_path` (falling back to the layout's own `input_path`)."""
    path = source_image_path or layout.input_path
    source_image = _resolve_source_image(path)
    if source_image is not None:
        source_image = match_layout_space(source_image, layout, path)
    if page_lines is not None:
        page_lines = lines_in_layout_space(page_lines, layout)
    return render_page_to_pdf(layout, output_path, source_image=source_image,
                              page_lines=page_lines, mode=mode or render_mode())


def _same_aspect(w1: float, h1: float, w2: float, h2: float) -> bool:
    return abs((w1 / h1) / (w2 / h2) - 1) <= ASPECT_TOLERANCE


def match_layout_space(image: Image.Image, layout: PageLayout, path: str | None = None) -> Image.Image:
    """The source image in the layout's pixel space. Every coordinate is in
    pixels of the image the layout was detected on; a uniformly resized copy
    is rescaled to it, a different image (e.g. the original instead of its
    `_flat.png`) is refused rather than silently misaligned."""
    size = (round(layout.width), round(layout.height))
    if image.size == size:
        return image
    if not _same_aspect(image.width, image.height, *size):
        raise ValueError(
            f"source image {path or ''} is {image.width}x{image.height} but the layout was "
            f"detected on a {size[0]}x{size[1]} page; pass the image it was analysed on "
            f"(for a flattened page, <name>_flat.png)")
    logger.warning("source image %s is %dx%d, layout is %dx%d; rescaling the image",
                   path, image.width, image.height, *size)
    return image.resize(size, Image.LANCZOS)


def lines_in_layout_space(lines: PageLines, layout: PageLayout) -> PageLines | None:
    """Detector lines rescaled to the layout's pixel space (None if they
    belong to a different page shape)."""
    if (round(lines.width), round(lines.height)) == (round(layout.width), round(layout.height)):
        return lines
    if not _same_aspect(lines.width, lines.height, layout.width, layout.height):
        logger.warning("cached lines (%dx%d) don't match the layout page (%dx%d); ignoring them",
                       lines.width, lines.height, layout.width, layout.height)
        return None
    sx, sy = layout.width / lines.width, layout.height / lines.height
    return lines.model_copy(update={
        "width": layout.width, "height": layout.height,
        "lines": [l.model_copy(update={"polygon": [(x * sx, y * sy) for x, y in l.polygon]})
                  for l in lines.lines]})


def _schema_json_path(output_json: str) -> str:
    """Where our own versioned PageLayout schema is saved, kept separate
    from `output_json` -- which holds PaddleOCR-VL's own raw result, in its
    own native "_res.json" naming convention (never overwritten)."""
    stem, ext = os.path.splitext(output_json)
    if stem.endswith("_res"):
        stem = stem[: -len("_res")]
    return f"{stem}_schema{ext}"


def process_image(
    image_path: str,
    output_pdf: str | None = None,
    output_json: str | None = None,
    engine: OcrEngine | None = None,
    line_engine: LineOcrEngine | None = None,
    preprocess: str | None = None,
    model_lock=None,
) -> str:
    """End-to-end: raw image -> flat page -> layout + lines (+ optional
    JSON) -> PDF.

    The page is flattened first (see `preprocess.py`) and every later step
    uses the flat image, so all coordinates refer to it. When `output_json`
    is given, PaddleOCR-VL's raw result is saved there as-is, our schema
    next to it (`_schema.json`), and the text lines as `_lines.json`.
    `model_lock` (if given) is held only while models run, not while the
    CPU-bound page model is built and rendered.
    """
    lock = model_lock or contextlib.nullcontext()
    started = time.perf_counter()
    with lock:
        prep = flatten_page(image_path, mode=preprocess or preprocess_mode())
    preprocessed = time.perf_counter()
    page_path = prep.image_path
    engine = engine or PaddleOcrVlEngine()
    with lock:
        layout = engine.analyze(page_path)
    analyzed = time.perf_counter()
    if output_json:
        raw = getattr(engine, "last_raw", None)
        if raw is not None:
            with open(output_json, "w") as f:
                json.dump(raw, f, indent=2)
            with open(_schema_json_path(output_json), "w") as f:
                f.write(layout.model_dump_json(indent=2))
        else:
            with open(output_json, "w") as f:
                f.write(layout.model_dump_json(indent=2))
    with lock:
        lines = detect_lines(layout, page_path, lines_json_path(output_json) if output_json else None,
                             engine=line_engine)
    detected = time.perf_counter()
    output_pdf = output_pdf or _default_output_path(image_path, ".pdf")
    result = render_pdf_from_layout(layout, output_pdf, source_image_path=page_path, page_lines=lines)
    finished = time.perf_counter()
    logger.info(
        "pipeline %s: total=%.3fs preprocess=%.3fs layout=%.3fs lines_and_json=%.3fs render=%.3fs",
        image_path, finished - started, preprocessed - started, analyzed - preprocessed,
        detected - analyzed, finished - detected,
    )
    return result


def document_pages(input_path: str, work_dir: str, max_pages: int | None = None) -> list[str]:
    """Page images of an input: every page of a PDF, or the image itself."""
    return pdf_page_images(input_path, work_dir, max_pages) if is_pdf(input_path) else [input_path]


def convert_page(image_path: str, work_dir: str, engine: OcrEngine | None = None,
                 line_engine: LineOcrEngine | None = None, preprocess: str | None = None,
                 model_lock=None) -> dict:
    """One page to `<work_dir>/<stem>.pdf`. A page that fails is replaced by its
    scan, so a document never loses a page; the error is reported in the row."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    pdf, js = os.path.join(work_dir, f"{stem}.pdf"), os.path.join(work_dir, f"{stem}_res.json")
    row = {"image": image_path, "pdf": pdf, "json": js, "error": ""}
    try:
        process_image(image_path, output_pdf=pdf, output_json=js, engine=engine,
                      line_engine=line_engine, preprocess=preprocess, model_lock=model_lock)
    except Exception as e:  # keep the document whole; the page falls back to the scan
        logger.exception("page %s failed; using the scan instead", image_path)
        row["error"] = f"{type(e).__name__}: {e}"
        image_page_pdf(image_path, pdf)
    return row


def process_document(input_path: str, output_pdf: str, work_dir: str, engine: OcrEngine | None = None,
                     line_engine: LineOcrEngine | None = None, preprocess: str | None = None) -> list[dict]:
    """A scanned PDF or a page image in, one vector PDF with every page out."""
    os.makedirs(work_dir, exist_ok=True)
    engine = engine or PaddleOcrVlEngine()
    if line_engine is None and line_ocr_enabled():
        line_engine = PaddleLineOcrEngine()
    rows = [convert_page(p, work_dir, engine, line_engine, preprocess)
            for p in document_pages(input_path, work_dir)]
    merge_pdfs([r["pdf"] for r in rows], output_pdf)
    return rows


def _locate_legacy_source_image(json_path: str, input_path: str | None) -> str | None:
    """Resolve the original page image for a legacy layout JSON via its
    `input_path` field, tried as-is and relative to the JSON's directory."""
    if not input_path:
        return None
    json_dir = os.path.dirname(os.path.abspath(json_path))
    for candidate in (
        input_path,
        os.path.join(json_dir, input_path),
        os.path.join(os.path.dirname(json_dir), input_path),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def load_layout_json(json_path: str) -> PageLayout:
    with open(json_path) as f:
        return PageLayout.from_paddle_dict(json.load(f))


def render_json_file(
    json_path: str, output_path: str | None = None, source_image: str | None = None,
    mode: str | None = None, line_engine: LineOcrEngine | None = None,
) -> str:
    """Render a previously saved layout JSON instead of running layout OCR
    on a raw image. Text lines come from `<name>_lines.json` when cached,
    or are detected (and cached) when line OCR is available."""
    layout = load_layout_json(json_path)
    output_path = output_path or _default_output_path(json_path, ".pdf")
    resolved_source = source_image or _locate_legacy_source_image(json_path, layout.input_path)
    mode = mode or render_mode()
    lines = None
    if mode == "lines" and resolved_source and os.path.isfile(resolved_source):
        lines = detect_lines(layout, resolved_source, lines_json_path(json_path), engine=line_engine)
    return render_pdf_from_layout(layout, output_path, source_image_path=resolved_source,
                                  page_lines=lines, mode=mode)


def detect_lines_for_json(json_path: str, source_image: str | None = None,
                          refresh: bool = False) -> str:
    """Detect/cache lines for a saved layout and summarize how they matched
    its blocks."""
    from .reconcile import reconcile

    layout = load_layout_json(json_path)
    resolved = source_image or _locate_legacy_source_image(json_path, layout.input_path)
    if not resolved:
        raise FileNotFoundError(f"source image not found for {json_path!r}; pass --source-image")
    cache = lines_json_path(json_path)
    if refresh and os.path.isfile(cache):
        os.remove(cache)
    lines = detect_lines(layout, resolved, cache, engine=PaddleLineOcrEngine())
    rec = reconcile(layout, lines)
    n_cont = sum(l.continuation for b in rec.blocks.values() for l in b.lines)
    n_text = sum(1 for b in layout.blocks if b.label not in IMAGE_LABELS and b.content)
    weak = sorted(rec.blocks.values(), key=lambda b: b.text_match)[:5]
    return (f"{cache}: {len(lines.lines)} lines; {len(rec.blocks)}/{n_text} text blocks matched, "
            f"{n_cont} continuation lines, {len(rec.orphans)} orphans; weakest text match: "
            + ", ".join(f"#{b.block_id}={b.text_match:.2f}" for b in weak))


def score_pdf_file(
    json_path: str,
    pdf_path: str,
    source_image: str | None = None,
    diff_path: str | None = None,
    report_path: str | None = None,
):
    """Score `pdf_path` against the source scan of the layout in `json_path`;
    optionally write a red/blue diff PNG and a JSON report."""
    from .fidelity import diff_image, score_pdf

    layout = load_layout_json(json_path)
    resolved = source_image or _locate_legacy_source_image(json_path, layout.input_path)
    if not resolved or not os.path.isfile(resolved):
        raise FileNotFoundError(f"source image not found for {json_path!r}; pass --source-image")
    source = Image.open(resolved).convert("RGB")
    size = (int(round(layout.width)), int(round(layout.height)))
    if source.size != size:
        source = source.resize(size, Image.LANCZOS)
    report, rendered = score_pdf(layout, pdf_path, source)
    if diff_path:
        diff_image(layout, source, rendered).save(diff_path)
    if report_path:
        with open(report_path, "w") as f:
            f.write(report.to_json())
    return report


def _default_output_path(path: str, new_ext: str) -> str:
    stem, _ = os.path.splitext(path)
    return stem + new_ext
