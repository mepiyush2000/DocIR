"""Shared helpers for the standalone bbox/polygon overlay diagnostic tools.

These are read-only visualization aids -- they don't affect rendering, they
just draw what the detector actually reported directly on top of the source
image, so bbox-only and polygon-only footprints can be compared by eye.
"""

from __future__ import annotations

import json

from PIL import Image, ImageDraw

from ..schemas import PageLayout

BBOX_COLOR = (220, 30, 30)
POLYGON_COLOR = (30, 90, 220)
NO_POLYGON_COLOR = (140, 140, 140)
LABEL_FILL = (255, 255, 255)


def load_page(json_path: str) -> PageLayout:
    """Load a `PageLayout` from either our own clean-schema JSON (has a
    top-level "blocks" key) or a raw PaddleOCR-VL result (has
    "parsing_res_list")."""
    with open(json_path) as f:
        data = json.load(f)
    if "parsing_res_list" in data:
        return PageLayout.from_paddle_dict(data)
    return PageLayout.model_validate(data)


def _label_tag(block) -> str:
    return f"{block.id}:{block.label}"


def _draw_label(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, color) -> None:
    draw.rectangle([x, y - 11, x + 7 * len(text) + 4, y], fill=LABEL_FILL, outline=color)
    draw.text((x + 2, y - 11), text, fill=color)


def draw_bbox_overlay(json_path: str, image_path: str, out_path: str) -> str:
    """Draw only `block.bbox` rectangles (axis-aligned) over the source
    image, in red, with an "id:label" tag per block."""
    layout = load_page(json_path)
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for block in layout.blocks:
        x1, y1, x2, y2 = block.bbox
        draw.rectangle([x1, y1, x2, y2], outline=BBOX_COLOR, width=2)
        _draw_label(draw, x1, max(y1, 11), _label_tag(block), BBOX_COLOR)
    img.save(out_path)
    return out_path


def draw_polygon_overlay(json_path: str, image_path: str, out_path: str) -> str:
    """Draw only `block.polygon` (blue) over the source image, with an
    "id:label" tag per block. Blocks with no polygon fall back to a thin
    gray bbox outline, so it's visually obvious which blocks lack polygon
    data."""
    layout = load_page(json_path)
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for block in layout.blocks:
        x1, y1, x2, y2 = block.bbox
        if block.polygon:
            points = [(p[0], p[1]) for p in block.polygon]
            draw.polygon(points, outline=POLYGON_COLOR, width=2)
            label_y = max(min(p[1] for p in points), 11)
            label_x = min(p[0] for p in points)
            _draw_label(draw, label_x, label_y, _label_tag(block), POLYGON_COLOR)
        else:
            draw.rectangle([x1, y1, x2, y2], outline=NO_POLYGON_COLOR, width=1)
            _draw_label(draw, x1, max(y1, 11), _label_tag(block), NO_POLYGON_COLOR)
    img.save(out_path)
    return out_path
