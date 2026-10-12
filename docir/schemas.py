"""Versioned, engine-agnostic data contracts for a page's recovered layout.

`PageLayout.from_paddle_dict` is the adapter boundary: PaddleOCR-VL's raw
result dict and field names (`block_label`, `block_bbox`, ...) must never
leak past this module into rendering, tests, or the CLI.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

SCHEMA_VERSION = "1.0"

# VL writes some glyphs as markup: a footnote star as `$ ^{*} $` (attached
# to the word before it), and in-cell line breaks as a literal backslash-n
_INLINE_SCRIPT = re.compile(r"[ \t]*\$\s*[\^_]\s*\{+\s*([^{}$]{1,8}?)\s*\}+\s*\$")
# other inline math holding commands or braced scripts, e.g. `$ \underline{\text{PO Box 1}} $`,
# `$ a_{x} $`, `$ \frac{1}{6} $` (handwriting the VL model read as a formula)
_INLINE_MATH = re.compile(r"\$\s([^$]*[\\{][^$]*)\$")
_FRACTION = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_TEX_COMMAND = re.compile(r"\\[A-Za-z]+\*?")
# fill-in blanks: VL writes a form's empty lines as runs of underscores (the
# lines themselves are drawn as graphics)
_BLANK = re.compile(r"_{2,}")


def _math_text(m: re.Match) -> str:
    text = _FRACTION.sub(r"\1/\2", m.group(1))
    text = re.sub(r"[\^_](?=\s*\{)", "", _TEX_COMMAND.sub("", text))
    return " ".join(re.sub(r"[{}]", "", text).split())


def clean_vl_text(text: str) -> str:
    """VL block text as printed: markup for super/subscripts, formatting
    commands, fill-in blanks and escaped line breaks resolved (a bare `$` is
    currency and stays)."""
    text = text.replace("\\n", "\n")
    text = _INLINE_SCRIPT.sub(r"\1", text)
    text = _INLINE_MATH.sub(_math_text, text)
    text = _BLANK.sub(" ", text)
    return "\n".join(" ".join(line.split()) for line in text.split("\n")).strip()


class Block(BaseModel):
    id: int
    label: str
    content: str
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2), source-image pixels
    polygon: list[tuple[float, float]] | None = None  # raw detected quad, if any;
                                                       # None for legacy/hand-built layouts


class PageLayout(BaseModel):
    schema_version: str = SCHEMA_VERSION
    input_path: str | None = None
    width: float
    height: float
    blocks: list[Block]

    @classmethod
    def from_paddle_dict(cls, data: dict) -> "PageLayout":
        """Map PaddleOCR-VL's raw `res.json["res"]` payload (or an equivalent
        saved JSON file) into our schema."""
        width, height = data.get("width"), data.get("height")
        if not width or not height:
            raise ValueError(f"Invalid page dimensions: width={width!r}, height={height!r}")

        raw_blocks = data.get("parsing_res_list") or []
        blocks = [
            Block(
                id=b.get("block_id", i),
                label=b.get("block_label") or "text",
                content=clean_vl_text((b.get("block_content") or "").strip()),
                bbox=tuple(b["block_bbox"]),
                polygon=[tuple(p) for p in b["block_polygon_points"]]
                if b.get("block_polygon_points") else None,
            )
            for i, b in enumerate(raw_blocks)
        ]
        return cls(
            input_path=data.get("input_path"),
            width=width,
            height=height,
            blocks=blocks,
        )


class TextLine(BaseModel):
    polygon: list[tuple[float, float]]  # detected quad, source-image pixels
    text: str                           # the line OCR's own reading
    score: float


class PageLines(BaseModel):
    """Physical text lines from a line-level OCR pass -- geometry comes from
    here, wording from the (cleaner) layout blocks."""

    schema_version: str = SCHEMA_VERSION
    input_path: str | None = None
    engine: str = ""
    width: float
    height: float
    lines: list[TextLine]

    @classmethod
    def from_paddle_dict(cls, data: dict, width: float, height: float, engine: str = "",
                         scale: float = 1.0) -> "PageLines":
        """Map PaddleOCR's raw `res.json["res"]` payload; `scale` maps the
        coordinates of an image resized before OCR back to the original."""
        lines = [
            TextLine(polygon=[(float(x) / scale, float(y) / scale) for x, y in poly],
                     text=text, score=float(score))
            for poly, text, score in zip(data.get("rec_polys") or [], data.get("rec_texts") or [],
                                         data.get("rec_scores") or [])
        ]
        return cls(input_path=data.get("input_path"), engine=engine, width=width,
                   height=height, lines=lines)
