"""High-level entry points wiring OCR -> layout schema -> PDF rendering."""

from __future__ import annotations

import json
import logging
import os

from PIL import Image

from .ocr import OcrEngine, PaddleOcrVlEngine
from .render_pdf import render_page_to_pdf
from .schemas import PageLayout

logger = logging.getLogger(__name__)


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


def render_pdf_from_layout(
    layout: PageLayout, output_path: str, source_image_path: str | None = None,
) -> str:
    """Render `layout` to `output_path`, resolving the source image from
    `source_image_path` (falling back to the layout's own `input_path`)."""
    path = source_image_path or layout.input_path
    source_image = _resolve_source_image(path)
    return render_page_to_pdf(layout, output_path, source_image=source_image)


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
) -> str:
    """End-to-end: raw image -> layout (+ optional JSON) -> PDF.

    When `output_json` is given and the engine exposes the raw PaddleOCR-VL
    result (`last_raw`), it's saved there as-is -- so it never needs to be
    re-run just to get the raw data again -- and our own schema is saved to
    a separate, automatically-derived path instead of overwriting it.
    """
    engine = engine or PaddleOcrVlEngine()
    layout = engine.analyze(image_path)
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
    output_pdf = output_pdf or _default_output_path(image_path, ".pdf")
    return render_pdf_from_layout(layout, output_pdf, source_image_path=image_path)


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


def render_json_file(
    json_path: str, output_path: str | None = None, source_image: str | None = None,
) -> str:
    """Backward-compatible entry point: render a previously saved layout
    JSON (e.g. from an older run) instead of running OCR on a raw image."""
    with open(json_path) as f:
        data = json.load(f)
    layout = PageLayout.from_paddle_dict(data)
    output_path = output_path or _default_output_path(json_path, ".pdf")
    resolved_source = source_image or _locate_legacy_source_image(json_path, layout.input_path)
    return render_pdf_from_layout(layout, output_path, source_image_path=resolved_source)


def _default_output_path(path: str, new_ext: str) -> str:
    stem, _ = os.path.splitext(path)
    return stem + new_ext
