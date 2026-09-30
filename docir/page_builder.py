"""Analysis -> page model: turns a layout (+ optional detector lines) and
the page image into a fully measured `Page`.

Per text block the line geometry comes from the detector when it agrees
with the block's own pixels (line count, no merged or crossing quads), and
from pixel segmentation otherwise; every line is then measured against the
ink (baseline, x-height, extent) and given its share of the block's text.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from . import table_layout, tables
from .color import assign_colors
from .config import font_size_offset_pt, straighten_max_deg
from .graphics import detect_graphics
from .ink import binarize_region, measure_text_lines
from .labels import PICTURE_LABELS as IMAGE_LABELS
from .labels import TABLE_LABELS
from .line_geometry import measure_glyph, measure_line, ordered_quad, pixel_lines, quad_height
from .line_fit import refit_page
from .page_model import Block, Line, Page, Picture
from .reconcile import LinePlacement, order_lines, reconcile, split_text_to_lines, vl_wording
from .render_model import TARGET_PAGE_WIDTH_PT as PAGE_WIDTH_PT
from .schemas import PageLayout, PageLines, TextLine
from .straighten import straighten_page
from .typography import assign_styles, offset_sizes

logger = logging.getLogger(__name__)

MERGED_PITCH_RATIO = 1.7         # a quad this many pitches tall holds several lines
MIN_PIXEL_LINE_HEIGHT_RATIO = 0.75  # of the block's typical line height
MIN_ORPHAN_SCORE = 0.8
FALLBACK_BASELINE_FRAC = 0.78    # of a quad's height, when its ink can't be measured
FALLBACK_X_HEIGHT_FRAC = 0.42
DROP_CAP_HEIGHT_RATIO = 2.2      # a drop cap's quad vs the block's typical line quad
DROP_CAP_LEFT_FRAC = 0.12        # ... and it sits at the block's left edge
PICTURE_TRIM_DESCENDER = 0.5     # of the x-height below a baseline kept clear of pictures
ORPHAN_ALIGN_XH = 1.5            # stacked orphan lines: left edges within this many x-heights
ORPHAN_MAX_GAP_XH = 3.0          # ... and at most this far below the previous one
TABLE_RULE_XH = 0.12             # printed table rules, as a fraction of the x-height


def _split_drop_cap(block, own: list[LinePlacement]):
    """(drop-cap placement or None, remaining placements)."""
    if len(own) < 3 or not block.content[:1].isalpha():
        return None, own
    heights = [quad_height(ordered_quad(l.polygon)) for l in own]
    typical = float(np.median(heights))
    x0, _, x1, _ = block.bbox
    for l, h in zip(own, heights):
        text = l.ocr_text.strip()
        if h >= DROP_CAP_HEIGHT_RATIO * typical and len(text) <= 2 \
                and l.bbox[0] <= x0 + DROP_CAP_LEFT_FRAC * (x1 - x0) \
                and (not text or text[0].lower() == block.content[0].lower()):
            return l, [o for o in own if o is not l]
    return None, own


def _detector_agrees(own: list[LinePlacement], expected: int, pitch: float | None) -> bool:
    if not own or len(own) != expected:
        return False
    if pitch and any(quad_height(ordered_quad(l.polygon)) > MERGED_PITCH_RATIO * pitch for l in own):
        return False
    # crossing quads: two lines sharing most of their height in the same column
    boxes = sorted((l.bbox for l in own), key=lambda b: b[1])
    for a, b in zip(boxes, boxes[1:]):
        x_overlap = min(a[2], b[2]) - max(a[0], b[0])
        y_overlap = min(a[3], b[3]) - max(a[1], b[1])
        if x_overlap > 0.5 * min(a[2] - a[0], b[2] - b[0]) and \
                y_overlap > 0.5 * min(a[3] - a[1], b[3] - b[1]):
            return False
    return True


def _measured_line(gray: np.ndarray, polygon, text: str, source: str, continuation: bool) -> Line:
    m = measure_line(gray, polygon)
    if m is not None:
        return Line(text=text, baseline_start=m.baseline_start, baseline_end=m.baseline_end,
                    x_height=m.x_height, cap_top=m.cap_top, source=source, continuation=continuation)
    tl, tr, br, bl = ordered_quad(polygon)
    f = FALLBACK_BASELINE_FRAC
    start, end = tl + (bl - tl) * f, tr + (br - tr) * f
    h = quad_height(ordered_quad(polygon))
    return Line(text=text, baseline_start=(float(start[0]), float(start[1])),
                baseline_end=(float(end[0]), float(end[1])), x_height=FALLBACK_X_HEIGHT_FRAC * h,
                cap_top=0.7 * h, source=source, continuation=continuation)


def _center_y(poly) -> float:
    return float(np.mean([p[1] for p in poly]))


def _overlaps(box, other, min_frac: float = 0.3) -> bool:
    """At least `min_frac` of `box`'s area lies inside `other`."""
    w = max(0.0, min(box[2], other[2]) - max(box[0], other[0]))
    h = max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
    return w * h >= min_frac * area


def _mostly_inside(poly, bbox) -> bool:
    """The quad's x-height band (its middle) lies inside the block box."""
    ys = sorted(p[1] for p in poly)
    top, bottom = ys[0], ys[-1]
    mid_top, mid_bottom = top + 0.3 * (bottom - top), bottom - 0.3 * (bottom - top)
    return bbox[1] <= mid_top and mid_bottom <= bbox[3]


def _merge_with_pixels(own: list[LinePlacement], pix: list[list], pitch: float | None, bbox):
    """Pixel lines as the skeleton; each is replaced by the (unmerged)
    detector lines centred inside it, detector lines no pixel line covers
    are kept, and pixel lines with no detector line fill the gaps."""
    usable = [l for l in own
              if not pitch or quad_height(ordered_quad(l.polygon)) <= MERGED_PITCH_RATIO * pitch]
    heights = [quad_height(ordered_quad(q)) for q in pix]
    ref = (float(np.median([quad_height(ordered_quad(l.polygon)) for l in usable])) if usable
           else float(np.median(heights)) if heights else 0.0)
    used: set[int] = set()
    out: list[tuple[list, str, str, bool]] = []
    for q, qh in zip(pix, heights):
        ys = [p[1] for p in q]
        inside = [l for l in usable if min(ys) <= _center_y(l.polygon) <= max(ys)]
        if inside:
            for l in inside:
                if id(l) not in used:
                    used.add(id(l))
                    out.append((l.polygon, l.ocr_text, "detector", False))
        elif qh >= MIN_PIXEL_LINE_HEIGHT_RATIO * ref and _mostly_inside(q, bbox):
            # others are slivers of a neighbouring line cut by the box edge
            out.append((q, "", "pixels", False))
    out += [(l.polygon, l.ocr_text, "detector", False) for l in usable if id(l) not in used]
    return out


def _block_lines(block, placements: list[LinePlacement], image: Image.Image,
                 gray: np.ndarray) -> list[Line]:
    own = [l for l in placements if not l.continuation]
    extra = [l for l in placements if l.continuation]
    drop, own = _split_drop_cap(block, own)
    drop_line = None
    content = block.content
    if drop is not None and (g := measure_glyph(gray, drop.polygon)) is not None:
        start, end, ink_h = g
        drop_line = Line(text=content[0], baseline_start=start, baseline_end=end, x_height=ink_h,
                         cap_top=ink_h, drop_cap=True)
        content = content[1:].lstrip()
    if block.role in TABLE_LABELS and tables.is_html_table(content) and \
            (grid := tables.parse_html_table(content)) is not None:
        return _html_table_lines(block, grid, own + extra, gray)
    if tables.is_tabular(block.role, [l.polygon for l in own]):
        block.tabular = True
        return ([drop_line] if drop_line else []) + _table_lines(content, own + extra, gray)
    mask = binarize_region(image, block.bbox)
    stats = measure_text_lines(mask) if mask is not None else None
    polys: list[tuple[list, str, str, bool]]  # polygon, ocr text, source, continuation
    if stats is None or _detector_agrees(own, stats.n_lines, stats.pitch):
        polys = [(l.polygon, l.ocr_text, "detector", False) for l in own]
    else:
        polys = _merge_with_pixels(own, pixel_lines(image, block.bbox), stats.pitch, block.bbox)
        logger.debug("block %d: %d detector lines merged with pixels -> %d (expected %d)",
                     block.id, len(own), len(polys), stats.n_lines)
    polys += [(l.polygon, l.ocr_text, "detector", True) for l in extra]
    if not polys:
        return [drop_line] if drop_line else []
    # reading order over the final set (continuations sit below by construction)
    tmp = [TextLine(polygon=p, text=t, score=1.0) for p, t, _, _ in polys]
    order = {id(t): i for i, t in enumerate(tmp)}
    ordered = [polys[order[id(t)]] for t in order_lines(tmp)]
    widths = [float(np.ptp([p[0] for p in poly])) for poly, *_ in ordered]
    texts, _ = split_text_to_lines(content, [t for _, t, _, _ in ordered], widths)
    lines = [_measured_line(gray, poly, text, source, cont)
             for (poly, _, source, cont), text in zip(ordered, texts) if text]
    return ([drop_line] if drop_line else []) + lines


def _html_table_lines(block, grid, placements: list[LinePlacement], gray: np.ndarray) -> list[Line]:
    """Grid geometry for a table the layout model returned as HTML. The
    measured detector lines returned here only inform typography (size,
    face, colour); `table_layout.typeset` replaces them afterwards."""
    measured = [_measured_line(gray, p.polygon, p.ocr_text, "detector", p.continuation)
                 for p in placements if p.ocr_text.strip()]
    x_height = float(np.median([l.x_height for l in measured])) if measured else 10.0
    block.tabular = True
    block.table = table_layout.analyze_table(grid, block.bbox, placements, gray, x_height)
    table_layout.detect_bold_rows(block.table, gray)
    # text inside the table region that the HTML doesn't contain (captions, sub-headers)
    block.table.loose_lines = [_measured_line(gray, poly, text, "detector", False)
                               for poly, text in block.table.loose if text.strip()]
    return measured


def _table_lines(content: str, cells: list[LinePlacement], gray: np.ndarray) -> list[Line]:
    """One line per detected cell, row by row, each with its own wording."""
    rows = tables.group_rows([c.polygon for c in cells])
    ordered = [(r, cells[i]) for r, row in enumerate(rows) for i in row]
    texts = tables.assign_cell_text([c.ocr_text for _, c in ordered], tables.vl_tokens(content))
    out = []
    for (r, c), t in zip(ordered, texts):
        if t:
            line = _measured_line(gray, c.polygon, t, "detector", c.continuation)
            line.row = r
            out.append(line)
    return out


def _continues_column(block: Block, box, line: Line) -> bool:
    """`line` (box) sits just below `block`'s last line, left edges aligned."""
    last = block.lines[-1]
    xh = max(min(last.x_height, line.x_height), 1.0)
    gap = box[1] - block.bbox[3]
    return (abs(box[0] - block.bbox[0]) <= ORPHAN_ALIGN_XH * xh
            and -0.5 * xh <= gap <= ORPHAN_MAX_GAP_XH * xh
            and 0.7 <= line.x_height / last.x_height <= 1.4)


def _trim_pictures(page: Page) -> None:
    """Pull a picture's top/bottom edge off any text line it cuts into, so
    the cropped photo carries no strip of scanned text under the vector text."""
    for pic in page.pictures:
        x0, y0, x1, y1 = pic.bbox
        for block in page.blocks:
            for l in block.lines:
                lx0, lx1 = sorted((l.baseline_start[0], l.baseline_end[0]))
                if min(lx1, x1) - max(lx0, x0) <= 0:
                    continue
                base = max(l.baseline_start[1], l.baseline_end[1])
                top = min(l.baseline_start[1], l.baseline_end[1]) - l.cap_top
                bottom = base + PICTURE_TRIM_DESCENDER * l.x_height
                if top < y0 < bottom:
                    y0 = bottom
                if top < y1 < bottom:
                    y1 = top
        if y1 > y0:
            pic.bbox = (x0, y0, x1, y1)


def build_page(layout: PageLayout, image: Image.Image, page_lines: PageLines | None = None,
               straighten_deg: float | None = None, size_offset_pt: float | None = None) -> Page:
    image = image.convert("RGB")
    gray = np.asarray(image.convert("L"))
    page = Page(width=layout.width, height=layout.height)
    rec = reconcile(layout, page_lines) if page_lines is not None else None

    for index, b in enumerate(layout.blocks):
        if b.label in IMAGE_LABELS:
            page.pictures.append(Picture(bbox=b.bbox, role=b.label))
            continue
        if not b.content:
            continue
        block = Block(id=b.id, role=b.label, reading_index=index, bbox=b.bbox, content=b.content)
        placements = rec.blocks[b.id].lines if rec is not None and b.id in rec.blocks else []
        block.lines = _block_lines(block, placements, image, gray)
        page.blocks.append(block)

    # text the layout model missed entirely (page numbers, captions, ...)
    next_id = max((b.id for b in layout.blocks), default=0) + 1
    first_orphan_id = next_id
    contents = [b.content for b in layout.blocks if b.content]
    drawn = ["".join(" ".join(l.text for l in b.lines).split()).lower() for b in page.blocks]
    for orphan in (rec.orphans if rec is not None else []):
        text = orphan.text.strip()
        if orphan.score < MIN_ORPHAN_SCORE or len(text) < 2:
            continue
        text = vl_wording(text, contents)
        key = "".join(text.split()).lower()
        # a line bending out of its block's box: its words are already drawn
        if any(key in d for d in drawn):
            continue
        xs, ys = [p[0] for p in orphan.polygon], [p[1] for p in orphan.polygon]
        box = (min(xs), min(ys), max(xs), max(ys))
        # a quad narrower than it is tall is not a horizontal text line
        if box[2] - box[0] < box[3] - box[1]:
            continue
        # lettering on a picture (book covers, logos) stays part of the picture
        if any(_overlaps(box, p.bbox) for p in page.pictures):
            continue
        line = _measured_line(gray, orphan.polygon, text, "detector", False)
        # lines stacked left-aligned right below the previous orphan: one block
        prev = page.blocks[-1] if page.blocks and page.blocks[-1].id >= first_orphan_id else None
        if prev is not None and _continues_column(prev, box, line):
            prev.lines.append(line)
            prev.content += "\n" + text
            prev.bbox = (min(prev.bbox[0], box[0]), prev.bbox[1], max(prev.bbox[2], box[2]), box[3])
            continue
        block = Block(id=next_id, role="text", reading_index=len(page.blocks), bbox=box, content=text)
        block.lines = [line]
        page.blocks.append(block)
        next_id += 1

    assign_styles(page, image)
    offset = font_size_offset_pt() if size_offset_pt is None else size_offset_pt
    offset_sizes(page.blocks, offset * image.width / PAGE_WIDTH_PT)
    refitted = refit_page(page.blocks)
    assign_colors(page, image)
    for block in page.blocks:
        if block.table is not None and block.style is not None:
            color = block.lines[0].color if block.lines else (0, 0, 0)
            block.lines = table_layout.typeset(block, block.table)
            for line in block.lines:
                line.color = color
    _trim_pictures(page)
    page.graphics = detect_graphics(page, image)
    for block in page.blocks:
        if block.table is not None:
            stroke = max(1.0, TABLE_RULE_XH * block.table.x_height)
            page.graphics += table_layout.grid_graphics(block.table, stroke_px=stroke)
    # after all pixel measurements, which need the real (tilted) geometry
    levelled = straighten_page(page, straighten_max_deg() if straighten_deg is None else straighten_deg)
    for block in page.blocks:
        if block.tabular and block.table is None:
            tables.align_row_baselines(block)
    logger.info("page model: %d text blocks (%d lines; %d from pixels; %d levelled; %d re-flowed), "
                "%d pictures, %d graphics", len(page.blocks), sum(len(b.lines) for b in page.blocks),
                sum(l.source == "pixels" for b in page.blocks for l in b.lines), levelled, refitted,
                len(page.pictures), len(page.graphics))
    return page
