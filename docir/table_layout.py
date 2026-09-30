"""Simple, robust table rendering from the layout model's HTML.

For a table block the layout model returns the full structure (rows,
cells, colspan / rowspan) with clean text; detector lines are only
fragments that may wrap, merge or be misread. So a table is rebuilt as a
grid, not as free lines:

  1. structure   -- rows x columns from the HTML (`tables.parse_html_table`);
  2. rows        -- the detector's fragments are grouped into visual rows,
                    and HTML rows are aligned onto them in order by a
                    monotone dynamic program on text similarity (an HTML row
                    may take several visual rows: wrapped cells);
  3. columns     -- inside each matched row, cells are paired with that
                    row's fragments left to right, which locates even short
                    cells ("1", "30"); a column starts at the median left
                    edge of its cells over all rows;
  4. typesetting -- after typography: one font size for the table, each
                    cell's text wrapped inside its cell and shrunk only if it
                    still doesn't fit, so cells can't overlap; rows without a
                    visual match get the font height plus padding;
  5. grid        -- only strong printed rules, redrawn as clean strokes.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import font_catalog
from .page_model import Block, Graphic, Line
from .tables import GridCell, TableGrid

ROW_OVERLAP_FRAC = 0.5        # fragments sharing this much height form one visual row
MAX_ROWS_PER_CELL_ROW = 4     # an HTML row may wrap over at most this many visual rows
MAX_ROWS_SPANNING_CELL = 10   # ... or this many when one cell spans most of the table
EXTRA_ROW_PENALTY = 0.12      # cost of each extra visual row an HTML row takes
LOOSE_MAX_SIM = 0.5           # fragments less similar than this to every cell aren't table text
MIN_ROW_MATCH = 0.3           # below this an HTML row is left unmatched
CELL_PAD_XH = 0.4             # inner cell padding, in x-heights
ROW_PAD_XH = 0.5              # padding above/below text for rows sized from the font
LEADING = 1.2                 # line spacing of wrapped cell text, in font sizes
MIN_SHRINK = 0.6              # a cell's text may shrink to this fraction of the table size
RULE_MIN_FRAC = 0.5           # a printed rule spans at least this much of the table
RULE_SNAP_XH = 1.0            # a row/column edge this close to a printed rule snaps onto it
RULE_PROJECTION_COVER = 0.45  # a broken vertical rule is inked along this share of the height
GRID_RULED_ROW_FRAC = 0.6     # rows separated by printed rules -> draw the full grid
GRID_MIN_ROWS = 3             # ... in a table with at least this many rows
LOOSE_MIN_CHARS = 3           # shorter unmatched fragments are cell parts, not captions
WRAP_PART_SIM = 0.8           # a fragment this much contained in a cell is one of its lines
PAGE_MARGIN_XH = 1.0          # an unruled table's last value stops this far from the page edge
CAP_HEIGHT_EM = 0.72          # baseline offset below a text line's top, in font sizes
DESCENT_EM = 0.22             # descenders below the last baseline, in font sizes
CELL_VPAD_XH = 0.25           # text keeps this far from a cell's top/bottom edge
DESCENDER_FRAC = 0.25         # a detector box's baseline sits this far above its bottom
BOLD_RATIO = 1.25             # header rows with stems this much heavier than the body are bold
HEADER_MAX_CHARS = 24         # longest cell text a heading row can hold


@dataclass
class TableLayout:
    grid: TableGrid
    row_edges: list[float]                                  # n_rows + 1 page y's
    col_edges: list[float]                                  # n_cols + 1 page x's
    x_height: float
    row_tops: dict[int, float] = field(default_factory=dict)    # measured first text top per row
    cell_lefts: dict[tuple[int, int], float] = field(default_factory=dict)  # measured cell text starts
    row_boxes: dict[int, list[tuple]] = field(default_factory=dict)       # measured text boxes per row
    row_frags: dict[int, list[tuple]] = field(default_factory=dict)       # (box, ocr text) per row
    loose: list[tuple[list, str]] = field(default_factory=list)  # (polygon, text) not in the HTML
    loose_lines: list[Line] = field(default_factory=list)
    page_right: float = 0.0                  # how far a row's last value may run (page margin)
    ruled: bool = False                      # drawn as a full grid
    h_rules: list[tuple[float, float, float]] = field(default_factory=list)   # y, x0, x1
    v_rules: list[tuple[float, float, float]] = field(default_factory=list)   # x, y0, y1
    bold_rows: set[int] = field(default_factory=set)


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _box(poly):
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _sim(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


# ---------- visual rows ----------

@dataclass
class _VRow:
    top: float
    bottom: float
    frags: list[tuple[tuple, str]]       # (box, text), left to right

    @property
    def text(self) -> str:
        return " ".join(t for _, t in self.frags)


def visual_rows(placements) -> list[_VRow]:
    items = sorted(((_box(p.polygon), p.ocr_text) for p in placements if p.ocr_text.strip()),
                   key=lambda t: (t[0][1] + t[0][3]) / 2)
    rows: list[_VRow] = []
    for box, text in items:
        if rows:
            r = rows[-1]
            ov = min(box[3], r.bottom) - max(box[1], r.top)
            if ov >= ROW_OVERLAP_FRAC * min(box[3] - box[1], r.bottom - r.top):
                r.frags.append((box, text))
                r.top, r.bottom = min(r.top, box[1]), max(r.bottom, box[3])
                continue
        rows.append(_VRow(box[1], box[3], [(box, text)]))
    for r in rows:
        r.frags.sort(key=lambda f: f[0][0])
    return rows


def match_rows(grid: TableGrid, vrows: list[_VRow]) -> dict[int, list[int]]:
    """HTML row -> visual rows it occupies, in order: a monotone dynamic
    program where each HTML row takes 0..MAX_ROWS_PER_CELL_ROW consecutive
    visual rows and visual rows may also be skipped (text around the table)."""
    n, m = grid.n_rows, len(vrows)
    row_text = [" ".join(c.text for c in grid.at(r)) for r in range(n)]
    max_k = []
    for r in range(n):
        filled = [c for c in grid.at(r) if _norm(c.text)]
        spanning = len(filled) == 1 and filled[0].colspan >= grid.n_cols / 2
        max_k.append(MAX_ROWS_SPANNING_CELL if spanning else MAX_ROWS_PER_CELL_ROW)
    neg = float("-inf")
    best = [[neg] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            cur = best[i][j]
            if cur == neg:
                continue
            if j < m and cur > best[i][j + 1]:          # skip a visual row
                best[i][j + 1], back[i][j + 1] = cur, (i, j)
            if i == n:
                continue
            if cur > best[i + 1][j]:                    # HTML row with no visual row
                best[i + 1][j], back[i + 1][j] = cur, (i, j)
            for k in range(1, min(max_k[i], m - j) + 1):
                s = _sim(row_text[i], " ".join(v.text for v in vrows[j:j + k]))
                if s < MIN_ROW_MATCH:
                    continue
                # a wrapped row needs its extra lines to raise the match
                s -= EXTRA_ROW_PENALTY * (k - 1) * (max_k[i] == MAX_ROWS_PER_CELL_ROW)
                if cur + s > best[i + 1][j + k]:
                    best[i + 1][j + k], back[i + 1][j + k] = cur + s, (i, j)
    j = max(range(m + 1), key=lambda t: best[n][t])
    i, out = n, {}
    while i > 0 or j > 0:
        step = back[i][j]
        if step is None:
            break
        pi, pj = step
        if pi == i - 1 and pj < j:
            out[pi] = list(range(pj, j))
        i, j = pi, pj
    return out


def absorb_wrapped_rows(grid: TableGrid, vrows: list[_VRow], matched: dict[int, list[int]]) -> None:
    """Give a matched HTML row the unmatched visual rows next to it whose
    every fragment is part of that row's cells (the other lines of a
    wrapped header or label), so they're typeset with it, not as loose text."""
    owner = {k: r for r, idx in matched.items() for k in idx}

    def part_of(r: int, k: int) -> bool:
        texts = [c.text for c in grid.at(r) if _norm(c.text)]
        return bool(texts) and all(max(_frag_sim(t, c) for c in texts) >= WRAP_PART_SIM
                                   for _, t in vrows[k].frags)

    changed = True
    while changed:
        changed = False
        for k in range(len(vrows)):
            if k in owner:
                continue
            for nb in (k - 1, k + 1):
                r = owner.get(nb)
                if r is not None and part_of(r, k):
                    owner[k] = r
                    matched[r] = sorted(matched[r] + [k])
                    changed = True
                    break


def _cell_lefts(grid: TableGrid, r: int, rows: list[_VRow]) -> dict[int, float]:
    """Left edge of each non-empty cell of HTML row r, by pairing its cells
    with the row's fragments in reading order (both sorted by x)."""
    cells = [c for c in grid.at(r) if _norm(c.text)]
    frags = [f for v in rows for f in v.frags]
    frags.sort(key=lambda f: f[0][0])
    if not cells or not frags:
        return {}
    # monotone alignment: each cell takes the fragment it best resembles
    n, m = len(cells), len(frags)
    neg = float("-inf")
    best = [[neg] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int, bool] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            cur = best[i][j]
            if cur == neg:
                continue
            if j < m and cur > best[i][j + 1]:
                best[i][j + 1], back[i][j + 1] = cur, (i, j, False)
            if i < n and cur > best[i + 1][j]:
                best[i + 1][j], back[i + 1][j] = cur, (i, j, False)
            if i < n and j < m:
                s = _sim(cells[i].text, frags[j][1])
                if s > 0.4 and cur + s > best[i + 1][j + 1]:
                    best[i + 1][j + 1], back[i + 1][j + 1] = cur + s, (i, j, True)
    i, j, out = n, max(range(m + 1), key=lambda t: best[n][t]), {}
    while i > 0 or j > 0:
        step = back[i][j]
        if step is None:
            break
        pi, pj, took = step
        if took:
            out[cells[pi].col] = frags[pj][0][0]
        i, j = pi, pj
    return out


# ---------- printed rules ----------

def printed_rules(gray: np.ndarray, bbox, x_height: float):
    """Strong horizontal (y, x0, x1) and vertical (x, y0, y1) rules printed
    in the table: long thin strokes spanning at least RULE_MIN_FRAC of it."""
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    pad = int(x_height)
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
    crop = gray[Y0:y1 + pad, X0:x1 + pad]
    if crop.size == 0:
        return [], []
    block = max(15, int(2 * x_height) | 1)
    # faint, watermark-broken rules: a low contrast offset, then bridge gaps
    ink = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 6)
    h, w = ink.shape
    out = []
    for horizontal in (True, False):
        span = w if horizontal else h
        # short kernel first (tolerates small gaps), then require the length
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(4 * x_height), 1) if horizontal else (1, int(4 * x_height)))
        lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k)
        close = cv2.getStructuringElement(cv2.MORPH_RECT, (int(4 * x_height), 1) if horizontal else (1, int(4 * x_height)))
        lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, close)
        n, _, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
        segs = []
        for i in range(1, n):
            x, y, bw, bh, _ = stats[i]
            length, thick = (bw, bh) if horizontal else (bh, bw)
            if length < RULE_MIN_FRAC * span or thick > max(4, 0.5 * x_height):
                continue
            segs.append((Y0 + y + bh / 2, X0 + x, X0 + x + bw) if horizontal
                        else (X0 + x + bw / 2, Y0 + y, Y0 + y + bh))
        out.append(segs)
    if not out[1]:
        out[1] = _projected_verticals(ink, X0, Y0, x_height)
    return out[0], out[1]


def _projected_verticals(ink: np.ndarray, X0: int, Y0: int, x_height: float):
    """Vertical rules broken into pieces (faint print, watermarks): after
    keeping only tall thin strokes, a rule's column is inked along a large
    share of the table's height even though no single piece spans it."""
    h = ink.shape[0]
    tall = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(3 * x_height)))))
    cover = cv2.dilate(tall, np.ones((1, 3), np.uint8)).astype(bool).mean(axis=0)
    cols = np.where(cover >= RULE_PROJECTION_COVER)[0]
    segs = []
    for a, b in _runs(np.isin(np.arange(ink.shape[1]), cols)):
        x = X0 + (a + b) / 2
        rows = np.where(tall[:, a:b].any(axis=1))[0]
        segs.append((x, Y0 + rows[0], Y0 + rows[-1] + 1))
    return segs if len(segs) >= 2 else []


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(padded)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _snap(edges: list[float], rules: list[float], tol: float) -> list[float]:
    out = []
    for e in edges:
        near = min(rules, key=lambda r: abs(r - e)) if rules else None
        out.append(near if near is not None and abs(near - e) <= tol else e)
    return out


def _fill(values: list[float | None], lo: float, hi: float) -> list[float]:
    """Missing entries interpolated by index, anchored to [lo, hi] at the ends."""
    n = len(values)
    known = [(i, v) for i, v in enumerate(values) if v is not None]
    xs = [-1.0] + [float(i) for i, _ in known] + [float(n)]
    ys = [lo] + [v for _, v in known] + [hi]
    return [float(np.interp(i, xs, ys)) if values[i] is None else float(values[i]) for i in range(n)]


# ---------- geometry ----------

def absorb_punctuation(grid: TableGrid) -> TableGrid:
    """A cell holding only punctuation (\":\" between a label and its value)
    is printed attached to the value; merge it into the next filled cell of
    its row so it doesn't claim a column of its own."""
    cells = list(grid.cells)
    drop = set()
    for i, c in enumerate(cells):
        if not c.text or any(ch.isalnum() for ch in c.text):
            continue
        later = [(j, d) for j, d in enumerate(cells)
                 if d.row == c.row and d.col > c.col and _norm(d.text) and j not in drop]
        if not later:
            continue
        j, nxt = min(later, key=lambda t: t[1].col)
        cells[j] = GridCell(nxt.row, nxt.col, f"{c.text} {nxt.text}", nxt.colspan, nxt.rowspan)
        drop.add(i)
    return TableGrid(grid.n_rows, grid.n_cols, [c for i, c in enumerate(cells) if i not in drop])


def analyze_table(grid: TableGrid, bbox, placements, gray: np.ndarray, x_height: float) -> TableLayout:
    grid = absorb_punctuation(grid)
    page_w = gray.shape[1]
    x0, y0, x1, y1 = bbox
    vrows = visual_rows(placements)
    matched = match_rows(grid, vrows)
    absorb_wrapped_rows(grid, vrows, matched)
    tops: list[float | None] = [None] * grid.n_rows
    bots: list[float | None] = [None] * grid.n_rows
    lefts: dict[int, list[float]] = {}
    cell_lefts: dict[tuple[int, int], float] = {}
    row_boxes: dict[int, list[tuple]] = {}
    row_frags: dict[int, list[tuple]] = {}
    for r, idx in matched.items():
        rows = [vrows[k] for k in idx]
        tops[r], bots[r] = min(v.top for v in rows), max(v.bottom for v in rows)
        row_boxes[r] = [box for v in rows for box, _ in v.frags]
        row_frags[r] = [f for v in rows for f in v.frags]
        for col, left in _cell_lefts(grid, r, rows).items():
            lefts.setdefault(col, []).append(left)
            cell_lefts[(r, col)] = left
    used = {k for idx in matched.values() for k in idx}
    cell_texts = [c.text for c in grid.cells if _norm(c.text)]
    loose = [(_poly(box), text) for k, v in enumerate(vrows) for box, text in v.frags
             if len(_norm(text)) >= LOOSE_MIN_CHARS and (
                 k not in used or max((_frag_sim(text, t) for t in cell_texts), default=0.0) < LOOSE_MAX_SIM)]

    # rows: between neighbouring rows' text; unmatched rows interpolated and
    # given at least one text line plus padding (the font-height fallback)
    row_h = x_height * 2 * (1 + ROW_PAD_XH)
    centres = [None if t is None else (t + b) / 2 for t, b in zip(tops, bots)]
    centres = _fill(centres, y0 - row_h / 2, y1 + row_h / 2)
    row_edges = [y0] + [(a + b) / 2 for a, b in zip(centres, centres[1:])] + [y1]
    for i in range(1, len(row_edges)):
        nxt_measured = i < len(tops) and tops[i] is not None
        if tops[i - 1] is None and not nxt_measured:
            row_edges[i] = max(row_edges[i], row_edges[i - 1] + row_h)

    # columns: median left edge of each column's cells, padded
    pad = CELL_PAD_XH * x_height
    col_starts = [float(np.median(lefts[c])) if c in lefts else None for c in range(grid.n_cols)]
    col_starts = _fill(col_starts, x0 - (x1 - x0) / grid.n_cols, x1)
    col_edges = [x0] + [col_starts[c] - pad for c in range(1, grid.n_cols)] + [x1]
    for i in range(1, len(col_edges)):
        col_edges[i] = max(col_edges[i], col_edges[i - 1] + 2 * x_height)
    col_edges[-1] = max(col_edges[-1], x1)

    h_rules, v_rules = printed_rules(gray, bbox, x_height)
    tol = RULE_SNAP_XH * x_height
    row_edges = _snap(row_edges, [r[0] for r in h_rules], tol)
    col_edges = _snap(col_edges, [r[0] for r in v_rules], tol)
    row_tops = {r: t for r, t in enumerate(tops) if t is not None}
    layout = TableLayout(grid=grid, row_edges=row_edges, col_edges=col_edges, x_height=x_height,
                         row_tops=row_tops, cell_lefts=cell_lefts, row_boxes=row_boxes,
                         row_frags=row_frags, loose=loose, h_rules=h_rules, v_rules=v_rules)
    # a ruled table's last column is closed by its border; otherwise a row's
    # last value may run on to the page margin
    ruled = len(v_rules) >= 2 or (grid.n_rows >= GRID_MIN_ROWS and len(h_rules) >= 2
                                  and _ruled_rows(layout) >= GRID_RULED_ROW_FRAC)
    layout.ruled = ruled
    layout.page_right = layout.col_edges[-1] if ruled else max(layout.col_edges[-1], page_w - PAGE_MARGIN_XH * x_height)
    return layout


def _poly(box):
    x0, y0, x1, y1 = box
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _frag_sim(frag: str, cell: str) -> float:
    """How much of a fragment's reading appears in a cell's text (a
    fragment is often part of a longer, wrapped cell)."""
    a, b = _norm(frag), _norm(cell)
    if not a or not b:
        return 0.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(m.size for m in sm.get_matching_blocks()) / len(a)


def detect_bold_rows(layout: TableLayout, gray: np.ndarray) -> None:
    """Rows printed heavier than the table's typical row (header rows),
    judged only inside each row's measured text boxes (their x-height
    band) so shading and rules around the text don't count."""
    ratios = {}
    for r, boxes in layout.row_boxes.items():
        runs = []
        for x0, y0, x1, y1 in boxes:
            h = y1 - y0
            band = gray[max(0, int(y0 + 0.3 * h)):int(y1 - 0.3 * h), max(0, int(x0)):int(x1)]
            w = _stem_width(band)
            if w:
                runs.append(w)
        if runs:
            ratios[r] = float(np.median(runs))
    if len(ratios) < 3:
        return
    base = float(np.median(list(ratios.values())))
    layout.bold_rows = {r for r, v in ratios.items() if v > BOLD_RATIO * base and _header_like(layout, r)}


def _header_like(layout: TableLayout, r: int) -> bool:
    """Column headings / section labels: every filled cell is short."""
    filled = [c.text for c in layout.grid.at(r) if _norm(c.text)]
    return bool(filled) and max(len(t) for t in filled) <= HEADER_MAX_CHARS


def _stem_width(crop: np.ndarray) -> float | None:
    if crop.size == 0 or int(crop.max()) - int(crop.min()) < 40:
        return None
    t, _ = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = crop < t
    # printed rules would dominate the run lengths: drop near-full rows/cols
    ink[ink.mean(axis=1) > 0.6] = False
    ink[:, ink.mean(axis=0) > 0.6] = False
    starts = ink.copy()
    starts[:, 1:] &= ~ink[:, :-1]
    n = int(starts.sum())
    return float(ink.sum() / n) if n else None


# ---------- typesetting ----------

def _wrap(text: str, font: str, size: float, width: float) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        trial = f"{cur} {w}" if cur else w
        if cur and font_catalog.string_width(trial, font, size) > width:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def _cell_extent(layout: TableLayout, cell, pad: float) -> tuple[float, float]:
    """Horizontal space of a cell in its own row: from where its text was
    measured (else its column edge) up to where the next non-empty cell of
    the row starts (else the end of its column span). Sub-tables and
    label/value rows with their own column positions then keep them."""
    grid = layout.grid
    x_end = layout.col_edges[min(cell.col + cell.colspan, len(layout.col_edges) - 1)] - pad
    start = layout.cell_lefts.get((cell.row, cell.col))
    # a start at (or before) where an earlier cell of the row begins is one
    # detector box merged across both cells, not where this cell begins
    if start is not None:
        for c in grid.at(cell.row):
            if c.col < cell.col and _norm(c.text):
                prev = layout.cell_lefts.get((c.row, c.col), layout.col_edges[c.col] + pad)
                if start <= prev + 2 * layout.x_height:
                    start = None
                    break
    x0 = start if start is not None else layout.col_edges[cell.col] + pad
    # in a ruled table text never touches its column's rule
    if layout.ruled:
        x0 = max(x0, layout.col_edges[cell.col] + pad)
    later = [c for c in grid.at(cell.row) if c.col > cell.col and _norm(c.text)]
    if later:
        nxt = min(later, key=lambda c: c.col)
        nxt_start = layout.cell_lefts.get((cell.row, nxt.col))
        if nxt_start is None:
            nxt_start = layout.col_edges[min(nxt.col, len(layout.col_edges) - 1)] + pad
        if nxt_start > x0:
            x_end = max(x_end, nxt_start - pad)
    else:
        # the last filled cell of its row may run on to the page margin, as
        # printed values do, rather than wrap into a sliver at the table edge
        x_end = max(x_end, layout.col_edges[-1] - pad, layout.page_right - pad)
    return x0, max(x_end, x0 + layout.x_height)


def _is_text_row(layout: TableLayout, cell) -> bool:
    """A cell alone in its row that the detector saw as several lines: a
    paragraph inside the table (notes, instructions), not a grid cell."""
    filled = [c for c in layout.grid.at(cell.row) if _norm(c.text)]
    frags = layout.row_frags.get(cell.row, [])
    rows = {round((b[1] + b[3]) / 2 / max(layout.x_height, 1)) for b, _ in frags}
    return len(filled) == 1 and len(rows) >= 2


def _measured_lines(layout: TableLayout, cell, text: str, font: str, size: float, bold: bool,
                    frags=None, min_x: float | None = None, max_x: float | None = None) -> list[Line]:
    """The cell's clean text split across its measured lines, each drawn
    where it was printed (the table's size, shrunk only to fit its line)."""
    from .reconcile import split_text_to_lines

    frags = _merge_line_frags(layout.row_frags[cell.row] if frags is None else frags, layout.x_height)
    widths = [b[2] - b[0] for b, _ in frags]
    pieces, _ = split_text_to_lines(text, [t for _, t in frags], widths)
    out = []
    for (box, _), piece in zip(frags, pieces):
        if not piece:
            continue
        x0, y0, x1, y1 = box
        if min_x is not None and x0 < min_x:
            x0, x1 = min_x, max(x1, min_x + layout.x_height)
        natural = font_catalog.string_width(piece, font, size)
        # a cell's own extent bounds it (a detector box may run into the next cell)
        room = (max_x - x0) if max_x is not None else x1 - x0
        s = size if natural <= room * 1.05 else max(size * room / natural, MIN_SHRINK * size)
        base = y1 - DESCENDER_FRAC * (y1 - y0)
        w = font_catalog.string_width(piece, font, s)
        out.append(Line(text=piece, baseline_start=(x0, base), baseline_end=(x0 + w, base),
                        x_height=layout.x_height, cap_top=CAP_HEIGHT_EM * s, row=cell.row,
                        size_px=None if abs(s - size) < 1e-6 else s, bold=bold, fit=False))
    return out


def _cell_frags(layout: TableLayout, cell, cx0: float, cx1: float) -> list[tuple]:
    """The measured fragments that are lines of this cell: centred within
    its horizontal extent and made of its text."""
    rows = range(cell.row, cell.row + cell.rowspan)
    return [(b, t) for r in rows for b, t in layout.row_frags.get(r, [])
            if cx0 - layout.x_height <= (b[0] + b[2]) / 2 <= cx1
            and _frag_sim(t, cell.text) >= WRAP_PART_SIM]


def _merge_line_frags(frags, x_height: float) -> list[tuple[tuple, str]]:
    """Fragments printed on one line (a label and its text) as one line box,
    top to bottom."""
    lines: list[list] = []
    for box, text in sorted(frags, key=lambda f: ((f[0][1] + f[0][3]) / 2, f[0][0])):
        cy = (box[1] + box[3]) / 2
        if lines and abs(cy - lines[-1][2]) <= 0.6 * x_height:
            b, t, _ = lines[-1]
            lines[-1] = [(min(b[0], box[0]), min(b[1], box[1]), max(b[2], box[2]), max(b[3], box[3])),
                         f"{t} {text}", cy]
        else:
            lines.append([box, text, cy])
    return [(b, t) for b, t, _ in lines]


def typeset(block: Block, layout: TableLayout) -> list[Line]:
    """Final lines of a table block: each cell's text wrapped inside its
    cell at the table's font size (shrunk per cell only if it can't fit)."""
    style = block.style
    size, xh = style.size_px, layout.x_height
    pad = CELL_PAD_XH * xh
    out: list[Line] = []
    for cell in layout.grid.cells:
        text = cell.text.strip()
        if not text:
            continue
        bold = cell.row in layout.bold_rows or style.bold
        face = font_catalog.face(style.font_family, bold, style.italic)
        if _is_text_row(layout, cell):
            out += _measured_lines(layout, cell, text, face.name, size, bold)
            continue
        last_row = min(cell.row + cell.rowspan, len(layout.row_edges) - 1)
        cx0, cx1 = _cell_extent(layout, cell, pad)
        # a cell printed over several lines keeps its printed line breaks
        frags = _cell_frags(layout, cell, cx0, cx1)
        if len(_merge_line_frags(frags, xh)) >= 2:
            out += _measured_lines(layout, cell, text, face.name, size, bold, frags,
                                   min_x=cx0 if layout.ruled else None, max_x=cx1)
            continue
        cy0, cy1 = layout.row_edges[cell.row], layout.row_edges[last_row]
        width, height = cx1 - cx0, max(cy1 - cy0, xh)
        vpad = CELL_VPAD_XH * xh
        s = size
        while True:
            lines = _wrap(text, face.name, s, width)
            widest = max(font_catalog.string_width(t, face.name, s) for t in lines)
            ink_h = (CAP_HEIGHT_EM + (len(lines) - 1) * LEADING + DESCENT_EM) * s
            # one line always fits its row (receipt rows are set tight);
            # wrapped cells must fit the row height
            fits_v = len(lines) == 1 or ink_h <= height - 2 * vpad
            if (fits_v and widest <= width) or s <= MIN_SHRINK * size:
                break
            s *= 0.94
        ink_h = (CAP_HEIGHT_EM + (len(lines) - 1) * LEADING + DESCENT_EM) * s
        measured = layout.row_tops.get(cell.row)
        # where the row's text was printed (kept inside the cell), else centred
        top = measured if measured is not None else cy0 + (height - ink_h) / 2
        top = min(max(top, cy0 + vpad), max(cy0 + vpad, cy1 - vpad - ink_h))
        for k, t in enumerate(lines):
            base = top + CAP_HEIGHT_EM * s + k * LEADING * s
            w = font_catalog.string_width(t, face.name, s)
            out.append(Line(text=t, baseline_start=(cx0, base), baseline_end=(cx0 + w, base),
                            x_height=xh, cap_top=CAP_HEIGHT_EM * s, row=cell.row,
                            size_px=None if abs(s - size) < 1e-6 else s, bold=bold, fit=False))
    return out + layout.loose_lines


def _ruled_rows(layout: TableLayout) -> float:
    """Share of inner row boundaries that carry a printed horizontal rule
    (a fully ruled table even when its faint verticals weren't found)."""
    inner = layout.row_edges[1:-1]
    if not inner:
        return 0.0
    tol = RULE_SNAP_XH * 2 * layout.x_height
    ys = [r[0] for r in layout.h_rules]
    return sum(any(abs(y - e) <= tol for y in ys) for e in inner) / len(inner)


def grid_graphics(layout: TableLayout, color=(0, 0, 0), stroke_px: float = 2.0) -> list[Graphic]:
    """Clean rules drawn on the table's own row/column edges, so they always
    fall between the typeset text:
      * ruled vertically in print  -> a full grid (segments a spanning cell
        covers are left out);
      * only horizontal rules      -> just those, snapped to row edges;
      * no printed rules           -> nothing."""
    rows, cols = layout.row_edges, layout.col_edges
    x0, x1 = cols[0], cols[-1]
    g: list[Graphic] = []

    def seg(a, b):
        g.append(Graphic(kind="rule", bbox=(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])),
                         color=color, stroke_px=stroke_px, points=[a, b]))

    if layout.ruled:
        covered_h = {(c.row + k, col) for c in layout.grid.cells for k in range(1, c.rowspan)
                     for col in range(c.col, c.col + c.colspan)}
        covered_v = {(row, c.col + k) for c in layout.grid.cells for k in range(1, c.colspan)
                     for row in range(c.row, c.row + c.rowspan)}
        for r, y in enumerate(rows):
            for c in range(len(cols) - 1):
                if (r, c) not in covered_h:
                    seg((cols[c], y), (cols[c + 1], y))
        for c, x in enumerate(cols):
            for r in range(len(rows) - 1):
                if (r, c) not in covered_v:
                    seg((x, rows[r]), (x, rows[r + 1]))
        return g
    tol = RULE_SNAP_XH * 2 * layout.x_height
    for y, a, b in layout.h_rules:
        near = min(rows, key=lambda e: abs(e - y))
        seg((max(a, x0), near if abs(near - y) <= tol else y), (min(b, x1), near if abs(near - y) <= tol else y))
    return g
