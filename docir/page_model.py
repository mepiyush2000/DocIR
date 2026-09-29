"""The intermediate page model: everything measured about a page, in
source-image pixels, ready to be drawn. Analysis modules fill it in
(`page_builder`, `typography`, `color`, `graphics`); the renderer only
reads it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Point = tuple[float, float]
Color = tuple[int, int, int]
BBox = tuple[float, float, float, float]

BLACK: Color = (0, 0, 0)
WHITE: Color = (255, 255, 255)


@dataclass
class Line:
    text: str
    baseline_start: Point          # left end of the baseline, at the first ink
    baseline_end: Point            # right end, at the last ink
    x_height: float                # px, measured
    cap_top: float                 # px above the baseline reached by the tallest ink
    source: str = "detector"       # "detector" | "pixels"
    continuation: bool = False     # lies below its block's detected box
    color: Color = BLACK
    size_px: float | None = None   # own size (drop caps); None = the block style's size
    drop_cap: bool = False         # a single enlarged initial; cap_top = its ink height

    @property
    def angle_deg(self) -> float:
        """Counter-clockwise baseline tilt (page y grows downwards)."""
        (x0, y0), (x1, y1) = self.baseline_start, self.baseline_end
        return math.degrees(math.atan2(y0 - y1, max(x1 - x0, 1e-6)))

    @property
    def length(self) -> float:
        (x0, y0), (x1, y1) = self.baseline_start, self.baseline_end
        return math.hypot(x1 - x0, y1 - y0)


@dataclass
class TextStyle:
    font_family: str               # bundled family key (see font_catalog)
    category: str                  # "serif" | "sans_serif" | "monospace"
    bold: bool = False
    italic: bool = False
    size_px: float = 12.0          # font size in source pixels
    width_scale: float = 1.0       # horizontal glyph scale shared by the role


@dataclass
class Block:
    id: int
    role: str                      # layout label: text, paragraph_title, ...
    reading_index: int
    bbox: BBox
    content: str
    lines: list[Line] = field(default_factory=list)
    style: TextStyle | None = None
    fill: Color | None = None      # background panel behind the block, if not the page's
    tabular: bool = False          # rows of separate cells: each line keeps its own text

    @property
    def has_lines(self) -> bool:
        return bool(self.lines)


@dataclass
class Picture:
    bbox: BBox
    role: str = "image"


@dataclass
class Graphic:
    kind: str                      # "rule" | "box" | "fill"
    bbox: BBox                     # rules: the stroke's extent
    color: Color = BLACK
    stroke_px: float = 1.0
    points: list[Point] = field(default_factory=list)  # rules: centre-line end points


@dataclass
class Page:
    width: float
    height: float
    background: Color = WHITE
    blocks: list[Block] = field(default_factory=list)      # reading order
    pictures: list[Picture] = field(default_factory=list)
    graphics: list[Graphic] = field(default_factory=list)

    def text_blocks(self) -> list[Block]:
        return [b for b in self.blocks if b.content]
