"""Versioned, engine-agnostic data contracts for a page's recovered layout.

`PageLayout.from_paddle_dict` is the adapter boundary: PaddleOCR-VL's raw
result dict and field names (`block_label`, `block_bbox`, ...) must never
leak past this module into rendering, tests, or the CLI.
"""

from __future__ import annotations

from pydantic import BaseModel

SCHEMA_VERSION = "1.0"


class Block(BaseModel):
    id: int
    label: str
    content: str
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2), source-image pixels


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
                content=(b.get("block_content") or "").strip(),
                bbox=tuple(b["block_bbox"]),
            )
            for i, b in enumerate(raw_blocks)
        ]
        return cls(
            input_path=data.get("input_path"),
            width=width,
            height=height,
            blocks=blocks,
        )
