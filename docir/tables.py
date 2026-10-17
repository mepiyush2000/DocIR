"""Tables and table-like blocks (receipts, invoices, key/value lists).

Paragraph logic breaks on these: a row is several separate *cells*, not
one line, so splitting a block's text at line breaks moves words between
cells and rows. Here a block is recognised as tabular from its detected
line fragments (several rows made of several cells) or from the layout
label, and each cell receives its own wording: the VL text (plain, or the
cells of an HTML table) is aligned onto the cells in reading order by a
monotone dynamic program that maximises reading similarity. Cells the VL
text does not cover keep their own OCR reading; VL words that belong to no
cell (e.g. a neighbouring block's row repeated by the layout model) are
skipped.
"""

from __future__ import annotations

import difflib
import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser

import numpy as np

from .labels import TABLE_LABELS

MIN_TABULAR_ROWS = 2             # rows made of 2+ cells needed to call a block tabular
MIN_TABULAR_ROW_FRAC = 0.3       # ... and at least this share of its rows
ROW_OVERLAP_FRAC = 0.5           # fragments sharing this much height are one row
TALL_BOX_RATIO = 1.6             # boxes taller than this many median line heights span rows
MAX_CELL_WORDS = 14
MIN_CELL_MATCH = 0.45            # below this similarity a cell keeps its OCR reading
SKIP_WORD_COST = 0.05            # small price for leaving a VL word unused
ROW_BASELINE_TOLERANCE = 1.2     # of the x-height: baselines this close are one row
CROP_RATIO = 1.5                 # VL text this much longer than a box's reading is cropped to it
CROP_MIN_CHARS = 3               # ... when the reading is long enough to align


# ---------- structure ----------

def _bbox(poly):
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _text_angle(polygons: list) -> float:
    """Median direction (radians) of quadrilateral line polygons (tl, tr,
    br, bl), weighted by length; 0 when the polygons are not quads."""
    angles, weights = [], []
    for p in polygons:
        if len(p) != 4:
            return 0.0
        lx, ly = (p[0][0] + p[3][0]) / 2, (p[0][1] + p[3][1]) / 2
        rx, ry = (p[1][0] + p[2][0]) / 2, (p[1][1] + p[2][1]) / 2
        if rx - lx <= 0:
            continue
        angles.append(np.arctan2(ry - ly, rx - lx))
        weights.append(rx - lx)
    if not angles:
        return 0.0
    order = np.argsort(angles)
    cum = np.cumsum(np.asarray(weights)[order])
    return float(np.asarray(angles)[order][np.searchsorted(cum, cum[-1] / 2)])


def group_rows(polygons: list) -> list[list[int]]:
    """Indices of `polygons` grouped into rows (top to bottom, left to right).

    Rows are found in a frame aligned with the text direction: on a slightly
    tilted page a long line's axis-aligned box is taller than the gap to
    the next line, which would merge consecutive lines into one row."""
    a = _text_angle(polygons)
    c, s = np.cos(-a), np.sin(-a)
    boxes = [_bbox([(x * c - y * s, x * s + y * c) for x, y in p]) for p in polygons]
    if not boxes:
        return []
    # a box spanning several rows (a column of tick marks read as one) would
    # chain them into one row: it joins the row it overlaps most instead
    line_h = float(np.median([b[3] - b[1] for b in boxes]))
    tall = {i for i, b in enumerate(boxes) if b[3] - b[1] > TALL_BOX_RATIO * line_h}
    order = sorted((i for i in range(len(boxes)) if i not in tall), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
    rows: list[list[int]] = []
    for i in order:
        x0, y0, x1, y1 = boxes[i]
        if rows:
            r = rows[-1]
            ry0 = min(boxes[j][1] for j in r)
            ry1 = max(boxes[j][3] for j in r)
            # a box under one already in the row is that cell's next line
            stacked = any(min(x1, boxes[j][2]) - max(x0, boxes[j][0]) > 0 for j in r)
            if not stacked and min(y1, ry1) - max(y0, ry0) >= ROW_OVERLAP_FRAC * min(y1 - y0, ry1 - ry0):
                r.append(i)
                continue
        rows.append([i])
    for i in sorted(tall, key=lambda i: boxes[i][1]):
        _, y0, _, y1 = boxes[i]
        spans = [(min(boxes[j][1] for j in r), max(boxes[j][3] for j in r)) for r in rows]
        ov = [min(y1, b) - max(y0, a) for a, b in spans]
        if ov and max(ov) > 0:
            rows[int(np.argmax(ov))].append(i)
        else:
            rows.append([i])
    rows.sort(key=lambda r: min(boxes[j][1] for j in r))
    return [sorted(r, key=lambda j: boxes[j][0]) for r in rows]


def is_tabular(label: str, polygons: list) -> bool:
    if label in TABLE_LABELS:
        return True
    rows = group_rows(polygons)
    multi = sum(len(r) >= 2 for r in rows)
    return multi >= MIN_TABULAR_ROWS and multi >= MIN_TABULAR_ROW_FRAC * len(rows)


# ---------- row geometry ----------

def align_row_baselines(block) -> None:
    """Cells of one table row share a baseline. Measured baselines of short
    cells (digits have no lowercase to anchor them) wander by a few pixels;
    a row is drawn on its median baseline. Rows come from detection when
    known (`Line.row`), else from baseline proximity. Run after levelling."""
    if block.lines and all(l.row is not None for l in block.lines):
        by_row: dict[int, list] = {}
        for l in block.lines:
            by_row.setdefault(l.row, []).append(l)
        rows = list(by_row.values())
    else:
        rows = _rows_by_baseline(block.lines)
    for row in rows:
        if len(row) < 2:
            continue
        base = float(np.median([(l.baseline_start[1] + l.baseline_end[1]) / 2 for l in row]))
        for l in row:
            dy = base - (l.baseline_start[1] + l.baseline_end[1]) / 2
            l.baseline_start = (l.baseline_start[0], l.baseline_start[1] + dy)
            l.baseline_end = (l.baseline_end[0], l.baseline_end[1] + dy)


def _rows_by_baseline(lines) -> list[list]:
    rows: list[list] = []
    for l in sorted(lines, key=lambda l: l.baseline_start[1]):
        y = (l.baseline_start[1] + l.baseline_end[1]) / 2
        if rows:
            ref = rows[-1]
            ry = float(np.median([(r.baseline_start[1] + r.baseline_end[1]) / 2 for r in ref]))
            if abs(y - ry) <= ROW_BASELINE_TOLERANCE * min(l.x_height, ref[0].x_height):
                ref.append(l)
                continue
        rows.append([l])
    return rows


# ---------- VL text ----------

class _CellParser(HTMLParser):
    """Collects cells as (row, text, colspan, rowspan), in document order."""

    def __init__(self):
        super().__init__()
        self.cells: list[str] = []
        self.spans: list[tuple[int, str, int, int]] = []
        self._row = -1
        self._buf: list[str] | None = None
        self._span = (1, 1)

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row += 1
        elif tag in ("td", "th"):
            a = dict(attrs)
            self._buf = []
            self._span = (_int(a.get("colspan")), _int(a.get("rowspan")))

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._buf is not None:
            text = " ".join("".join(self._buf).split())
            self.cells.append(text)
            self.spans.append((max(self._row, 0), text, *self._span))
            self._buf = None

    def handle_data(self, data):
        if self._buf is not None:
            self._buf.append(data)


def _int(v) -> int:
    try:
        return max(1, int(v))
    except (TypeError, ValueError):
        return 1


@dataclass(frozen=True)
class GridCell:
    row: int
    col: int
    text: str
    colspan: int = 1
    rowspan: int = 1


@dataclass
class TableGrid:
    n_rows: int
    n_cols: int
    cells: list[GridCell]

    def at(self, row: int) -> list[GridCell]:
        return [c for c in self.cells if c.row == row]


def is_html_table(content: str) -> bool:
    return "<td" in content or "<th" in content


def parse_html_table(content: str) -> TableGrid | None:
    """Row/column grid of an HTML table, honouring colspan and rowspan: a
    cell occupies the first column slot not already taken by a cell
    spanning down from a row above."""
    parser = _CellParser()
    parser.feed(content)
    if not parser.spans:
        return None
    taken: set[tuple[int, int]] = set()
    cells: list[GridCell] = []
    col_by_row: dict[int, int] = {}
    for row, text, cs, rs in parser.spans:
        col = col_by_row.get(row, 0)
        while (row, col) in taken:
            col += 1
        for r in range(row, row + rs):
            for c in range(col, col + cs):
                taken.add((r, c))
        cells.append(GridCell(row, col, text, cs, rs))
        col_by_row[row] = col + cs
    n_rows = max(r for r, _ in taken) + 1
    n_cols = max(c for _, c in taken) + 1
    return TableGrid(n_rows, n_cols, cells)


def vl_tokens(content: str) -> list[str]:
    """VL wording as a word sequence in reading order; an HTML table
    contributes its cells' words cell by cell."""
    if "<td" in content or "<th" in content:
        parser = _CellParser()
        parser.feed(content)
        return [w for cell in parser.cells for w in cell.split()]
    return content.split()


# ---------- cell wording ----------

def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _similarity(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def assign_cell_text(cell_ocr: list[str], words: list[str]) -> list[str]:
    """Wording for each cell (in reading order): the consecutive run of VL
    `words` that best matches its OCR reading, found jointly for all cells
    so runs never overlap or cross; the OCR reading where nothing matches."""
    n, m = len(cell_ocr), len(words)
    neg = float("-inf")
    # best[k][j]: score after assigning the first k cells using words[:j]
    best = [[neg] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[str, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for k in range(n + 1):
        for j in range(m + 1):
            cur = best[k][j]
            if cur == neg:
                continue
            if j < m and cur - SKIP_WORD_COST > best[k][j + 1]:
                best[k][j + 1], back[k][j + 1] = cur - SKIP_WORD_COST, ("skip", j)
            if k == n:
                continue
            # the cell takes no VL words (keeps its OCR reading)
            if cur > best[k + 1][j]:
                best[k + 1][j], back[k + 1][j] = cur, ("empty", j)
            for e in range(j + 1, min(m, j + MAX_CELL_WORDS) + 1):
                s = _similarity(cell_ocr[k], " ".join(words[j:e]))
                if s < MIN_CELL_MATCH:
                    continue
                if cur + s > best[k + 1][e]:
                    best[k + 1][e], back[k + 1][e] = cur + s, ("take", j)
    j = max(range(m + 1), key=lambda t: best[n][t])
    out = list(cell_ocr)
    k = n
    while k > 0 or j > 0:
        step = back[k][j]
        if step is None:
            break
        kind, prev = step
        if kind == "skip":
            j = prev
            continue
        k -= 1
        if kind == "take":
            out[k] = " ".join(_trim_run(cell_ocr[k], words[prev:j]))
        j = prev
    return [t.strip() for t in out]


def _trim_run(ocr: str, run: list[str]) -> list[str]:
    """A cell's VL run without end words that aren't in its reading: taking
    a neighbour's word costs the DP less than skipping it."""
    while len(run) > 1:
        s = _similarity(ocr, " ".join(run))
        if _similarity(ocr, " ".join(run[1:])) > s:
            run = run[1:]
        elif _similarity(ocr, " ".join(run[:-1])) > s:
            run = run[:-1]
        else:
            break
    return _crop_to_reading(ocr, " ".join(run)).split()


def _crop_to_reading(ocr: str, text: str) -> str:
    """The part of `text` the box's reading covers, when the box holds only
    part of it (the detector cut a printed line the VL read as one word)."""
    if len(_norm(ocr)) < CROP_MIN_CHARS or len(_norm(text)) <= CROP_RATIO * len(_norm(ocr)):
        return text
    blocks = [b for b in difflib.SequenceMatcher(None, text, ocr, autojunk=False).get_matching_blocks() if b.size]
    if not blocks:
        return text
    start = max(0, blocks[0].a - blocks[0].b)
    end = min(len(text), blocks[-1].a + blocks[-1].size + len(ocr) - blocks[-1].b - blocks[-1].size)
    return text[start:end].strip() or text
