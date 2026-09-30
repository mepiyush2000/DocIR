"""Reconcile line-level OCR with PaddleOCR-VL layout blocks.

The two sources are complementary:
  * VL blocks: role labels, reading order and the cleanest wording, but a
    block's box often stops short of its paragraph (e.g. text wrapping
    around a photo) and carries no line geometry.
  * Line OCR: exact physical line polygons, but noisy wording and lines
    that occasionally run across a column gap.

Result: every text block gets its physical lines (geometry from line OCR)
carrying the block's own VL wording, split at the same line breaks.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import numpy as np

from .labels import PICTURE_LABELS as IMAGE_LABELS
from .schemas import Block, PageLayout, PageLines, TextLine

MIN_CLAIM_FRAC = 0.5             # of a line's area inside a block to belong to it
SPLIT_MIN_FRAC = 0.25            # of a line's width a second block must cover to split it
SPLIT_SMALL_BLOCK_FRAC = 0.6     # ... or this much of the (small) block itself
MIN_ROW_OVERLAP_FRAC = 0.5       # of a line's height overlapping a block to count as "beside" it
CONTINUATION_MIN_X_OVERLAP = 0.6
CONTINUATION_MAX_GAP_LINES = 2.0  # max gap below a block, in its line pitches
CONTINUATION_MIN_LEFTOVER = 0.5  # of the line's text the block's content must still hold
WORD_SNAP_CHARS = 12             # how far a line break may move to reach a space
MIN_OCR_LENGTH_FRAC = 0.4        # OCR text shorter than this x the expected length is "unread"
UNREAD_CHAR = "\x00"             # placeholder that occupies an unread line's width
TRIM_MIN_COVERAGE = 0.7          # of OCR characters aligned before a text tail is dropped
TRIM_MIN_CHARS = 8
TRIM_TAIL_FRAC = 0.35            # of the last line's length
ANCHOR_CHARS = 4                 # next-line prefix that must match to break mid-word
VL_WORDING_MIN_MATCH = 0.85      # of an orphan's characters found in a VL passage
VL_WORDING_MAX_SPAN = 1.3        # the passage may be at most this much longer


@dataclass
class LinePlacement:
    polygon: list[tuple[float, float]]   # TL, TR, BR, BL in source pixels
    text: str                            # wording to draw (from the VL block)
    ocr_text: str
    score: float
    continuation: bool = False           # attached below the block's own box

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return _bbox(self.polygon)


@dataclass
class BlockLines:
    block_id: int
    lines: list[LinePlacement]
    text_match: float                    # 0..1 agreement of VL text with the OCR lines


@dataclass
class Reconciliation:
    blocks: dict[int, BlockLines] = field(default_factory=dict)
    orphans: list[TextLine] = field(default_factory=list)   # text outside every block


# ---------- geometry ----------

def _bbox(poly) -> tuple[float, float, float, float]:
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap_1d(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _area_frac_inside(box, region) -> float:
    x0, y0, x1, y1 = box
    area = max(x1 - x0, 1e-6) * max(y1 - y0, 1e-6)
    return _overlap_1d(x0, x1, region[0], region[2]) * _overlap_1d(y0, y1, region[1], region[3]) / area


def _ordered_quad(poly) -> list[tuple[float, float]]:
    """TL, TR, BR, BL of any 4-point polygon."""
    pts = sorted(((float(x), float(y)) for x, y in poly), key=lambda p: p[0])
    left = sorted(pts[:2], key=lambda p: p[1])
    right = sorted(pts[2:], key=lambda p: p[1])
    return [left[0], right[0], right[1], left[1]]


def _edge_y(p, q, x) -> float:
    if abs(q[0] - p[0]) < 1e-6:
        return p[1]
    return p[1] + (q[1] - p[1]) * (x - p[0]) / (q[0] - p[0])


def clip_quad(poly, x0: float, x1: float) -> list[tuple[float, float]]:
    """The part of a line quad between x0 and x1, following its tilt."""
    tl, tr, br, bl = _ordered_quad(poly)
    x0, x1 = max(x0, min(tl[0], bl[0])), min(x1, max(tr[0], br[0]))
    return [(x0, _edge_y(tl, tr, x0)), (x1, _edge_y(tl, tr, x1)),
            (x1, _edge_y(bl, br, x1)), (x0, _edge_y(bl, br, x0))]


# ---------- step 1: split lines that run across a column gap ----------

def _split_across_blocks(line: TextLine, blocks: list[Block]) -> list[TextLine]:
    x0, y0, x1, y1 = _bbox(line.polygon)
    width, height = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    beside = []
    for b in blocks:
        bx0, by0, bx1, by1 = b.bbox
        if _overlap_1d(y0, y1, by0, by1) < MIN_ROW_OVERLAP_FRAC * height:
            continue
        # covering much of the line, or most of a small neighbour (a page number)
        overlap = _overlap_1d(x0, x1, bx0, bx1)
        if overlap >= SPLIT_MIN_FRAC * width or overlap >= SPLIT_SMALL_BLOCK_FRAC * (bx1 - bx0):
            beside.append(b)
    if len(beside) < 2:
        return [line]
    beside.sort(key=lambda b: b.bbox[0])
    # cut halfway through each gap between neighbouring blocks
    cuts = [(a.bbox[2] + b.bbox[0]) / 2 for a, b in zip(beside, beside[1:])]
    edges = [x0, *cuts, x1]
    text = line.text
    pieces = []
    for a, b in zip(edges, edges[1:]):
        if b - a < 1:
            continue
        i0 = round(len(text) * (a - x0) / width)
        i1 = round(len(text) * (b - x0) / width)
        pieces.append(TextLine(polygon=clip_quad(line.polygon, a, b), text=text[i0:i1].strip(),
                               score=line.score))
    return pieces


def _clip_out_of_pictures(line: TextLine, pictures: list[Block]) -> TextLine:
    """A line whose quad starts (or ends) inside a picture beside it -- a
    book cover next to its blurb -- is cut at the picture's edge."""
    x0, y0, x1, y1 = _bbox(line.polygon)
    a, b = x0, x1
    for p in pictures:
        px0, py0, px1, py1 = p.bbox
        if _overlap_1d(y0, y1, py0, py1) < MIN_ROW_OVERLAP_FRAC * (y1 - y0):
            continue
        if px0 <= a < px1 < b:
            a = px1
        if a < px0 < b <= px1:
            b = px0
    if a == x0 and b == x1:
        return line
    return TextLine(polygon=clip_quad(line.polygon, a, b), text=line.text, score=line.score)


# ---------- step 2: assign lines to blocks ----------

def _assign(lines: list[TextLine], text_blocks: list[Block], pictures: list[Block]):
    owned: dict[int, list[TextLine]] = {b.id: [] for b in text_blocks}
    unclaimed: list[TextLine] = []
    for line in lines:
        box = _bbox(line.polygon)
        best, best_frac = None, 0.0
        for b in text_blocks:
            frac = _area_frac_inside(box, b.bbox)
            if frac > best_frac:
                best, best_frac = b, frac
        if best is not None and best_frac >= MIN_CLAIM_FRAC:
            owned[best.id].append(line)
        elif any(_area_frac_inside(box, p.bbox) >= MIN_CLAIM_FRAC for p in pictures):
            continue  # text printed inside a picture (book covers, logos)
        else:
            unclaimed.append(line)
    return owned, unclaimed


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _line_pitch(lines: list[TextLine]) -> float:
    boxes = sorted((_bbox(l.polygon) for l in lines), key=lambda b: b[1])
    heights = [b[3] - b[1] for b in boxes]
    tops = [b[1] for b in boxes]
    gaps = [t2 - t1 for t1, t2 in zip(tops, tops[1:]) if t2 - t1 > 0.5 * np.median(heights)]
    return float(np.median(gaps)) if gaps else 1.4 * float(np.median(heights))


def _attach_continuations(owned, unclaimed, text_blocks):
    """Unclaimed lines directly below a block, in its column, while the
    block's VL text still holds words its own lines don't account for."""
    by_id = {b.id: b for b in text_blocks}
    continuation: set[int] = set()
    leftovers = []
    for line in sorted(unclaimed, key=lambda l: _bbox(l.polygon)[1]):
        lx0, ly0, lx1, ly1 = _bbox(line.polygon)
        best = None
        for bid, blines in owned.items():
            if not blines:
                continue
            block = by_id[bid]
            bottom = max(block.bbox[3], *(_bbox(l.polygon)[3] for l in blines))
            gap = ly0 - bottom
            x_frac = _overlap_1d(lx0, lx1, block.bbox[0], block.bbox[2]) / max(lx1 - lx0, 1e-6)
            # below the block (a slight overlap with its last line is fine)
            if x_frac < CONTINUATION_MIN_X_OVERLAP or gap < -0.5 * (ly1 - ly0):
                continue
            if gap > CONTINUATION_MAX_GAP_LINES * _line_pitch(blines):
                continue
            leftover = len(_norm(block.content)) - sum(len(_norm(l.text)) for l in blines)
            if leftover < CONTINUATION_MIN_LEFTOVER * len(_norm(line.text)):
                continue
            if best is None or gap < best[1]:
                best = (bid, gap)
        if best is None:
            leftovers.append(line)
        else:
            owned[best[0]].append(line)
            continuation.add(id(line))
    return leftovers, continuation


# ---------- step 3: reading order of a block's lines ----------

def order_lines(lines: list[TextLine]) -> list[TextLine]:
    """Top-to-bottom rows (a row = lines sharing most of their height),
    left-to-right within a row."""
    items = sorted(lines, key=lambda l: (_bbox(l.polygon)[1] + _bbox(l.polygon)[3]) / 2)
    rows: list[list[TextLine]] = []
    for line in items:
        _, y0, _, y1 = _bbox(line.polygon)
        if rows:
            _, ry0, _, ry1 = _bbox(rows[-1][-1].polygon)
            if _overlap_1d(y0, y1, ry0, ry1) >= 0.5 * min(y1 - y0, ry1 - ry0):
                rows[-1].append(line)
                continue
        rows.append([line])
    return [l for row in rows for l in sorted(row, key=lambda l: _bbox(l.polygon)[0])]


# ---------- step 4: VL wording onto OCR lines ----------

def _nonspace_index(text: str) -> list[int]:
    return [i for i, ch in enumerate(text) if not ch.isspace()]


def _map_positions(src: str, dst: str, positions: list[int]) -> tuple[list[float], float, float]:
    """Map character offsets in `src` onto `dst` via their alignment,
    interpolating between matched anchors. Returns (offsets, match ratio,
    fraction of `src` matched)."""
    sm = difflib.SequenceMatcher(None, src, dst, autojunk=False)
    anchors = [(0, 0)]
    for a, b, n in sm.get_matching_blocks():
        if n:
            anchors += [(a, b), (a + n, b + n)]
    xs, ys = zip(*sorted(set(anchors)))
    matched = sum(n for *_, n in sm.get_matching_blocks())
    ratio = 2 * matched / max(len(src) + len(dst), 1)

    def at(p: float) -> float:
        # past the last match, characters advance one-for-one
        if p >= xs[-1]:
            return min(ys[-1] + (p - xs[-1]), len(dst))
        return float(np.interp(p, xs, ys))

    return [at(p) for p in positions], ratio, matched / max(len(src), 1)


def _snap_to_space(text: str, pos: int) -> int:
    if pos <= 0 or pos >= len(text) or text[pos].isspace() or text[pos - 1].isspace():
        return pos
    for d in range(1, WORD_SNAP_CHARS + 1):
        for p in (pos - d, pos + d):
            if 0 < p < len(text) and text[p].isspace():
                return p
    return pos


def _with_unread_placeholders(texts: list[str], widths: list[float]) -> list[str]:
    """Stand-in text for lines the OCR could not read (empty or truncated):
    placeholders sized by the line's width, so alignment still spaces the
    line breaks correctly around them."""
    lengths = [len(t) for t in texts]
    read = [(n, w) for n, w in zip(lengths, widths) if n and w > 0]
    if not read:
        return texts
    chars_per_px = sum(n for n, _ in read) / sum(w for _, w in read)
    out = []
    for t, w in zip(texts, widths):
        expected = w * chars_per_px
        out.append(t if len(t) >= MIN_OCR_LENGTH_FRAC * expected
                   else UNREAD_CHAR * max(1, round(expected)))
    return out


def _trim_unseen_tail(content: str, idx: list[int], end_mapped: float, last_ocr: str,
                      coverage: float) -> str:
    """Drop text past the end of the last line when the OCR read the lines
    well and clearly ends earlier: layout models sometimes repeat the start
    of the next column's text inside a block."""
    if coverage < TRIM_MIN_COVERAGE or UNREAD_CHAR in last_ocr:
        return content
    j = int(round(end_mapped))
    if j >= len(idx):
        return content
    cut = _snap_to_space(content, idx[j])
    tail = content[cut:].strip()
    if len(tail.replace(" ", "")) <= max(TRIM_MIN_CHARS, TRIM_TAIL_FRAC * len(last_ocr)):
        return content
    return content[:cut].rstrip()


def split_text_to_lines(content: str, line_texts: list[str],
                        line_widths: list[float]) -> tuple[list[str], float]:
    """Split `content` into len(line_texts) consecutive pieces whose breaks
    match where the OCR broke lines. OCR wording is only used to locate the
    breaks (spaces are ignored: line OCR often drops them); line widths
    are the fallback when OCR and content disagree too much."""
    n = len(line_texts)
    content = " ".join(content.split())
    if n == 0:
        return [], 0.0
    idx = _nonspace_index(content)
    flat = "".join(content[i] for i in idx).lower()
    hyphenated = [t.rstrip().endswith("-") for t in line_texts]
    # a trailing hyphen may be missing from the VL text (word re-joined)
    ocr = ["".join(t.split()).lower().rstrip("-") for t in line_texts]
    ocr = _with_unread_placeholders(ocr, line_widths)
    bounds, total = [], 0
    for t in ocr:
        total += len(t)
        bounds.append(total)
    mapped, ratio, coverage = _map_positions("".join(ocr), flat, bounds)
    content = _trim_unseen_tail(content, idx, mapped[-1], ocr[-1], coverage)
    idx = _nonspace_index(content)
    flat = flat[:len(idx)]
    mapped = mapped[:-1]
    if n == 1:
        return [content], ratio
    if ratio < 0.3:
        widths = np.cumsum(line_widths)[:-1] / max(sum(line_widths), 1e-6)
        mapped = list(widths * len(flat))
    cuts = []
    for k, m in enumerate(mapped):
        j = int(round(min(max(m, 0), len(flat))))
        pos = idx[j] if j < len(idx) else len(content)
        nxt = ocr[k + 1][:ANCHOR_CHARS]
        # the next line's reading starts exactly here: break here, even mid-word
        anchored = len(nxt) == ANCHOR_CHARS and UNREAD_CHAR not in nxt and flat[j:j + ANCHOR_CHARS] == nxt
        if hyphenated[k]:
            if pos < len(content) and content[pos] == "-":
                pos += 1
        elif not anchored:
            pos = _snap_to_space(content, pos)
        mid_word = 0 < pos < len(content) and content[pos - 1].isalpha() and content[pos].isalpha()
        cuts.append((pos, hyphenated[k] or (anchored and mid_word)))
    pieces, start = [], 0
    for pos, hyphen in cuts:
        pos = max(pos, start)
        piece = content[start:pos].strip()
        # a word the source hyphenated across lines, joined in the VL text
        if hyphen and piece and pos < len(content) and not content[pos].isspace() \
                and not piece.endswith("-"):
            piece += "-"
        pieces.append(piece)
        start = pos
    pieces.append(content[start:].strip())
    return pieces, ratio


# ---------- entry point ----------

def vl_wording(ocr_text: str, contents: list[str]) -> str:
    """The passage of any VL block text that `ocr_text` reads, with VL's
    spacing and spelling -- for lines outside every block, whose OCR
    reading often lacks spaces. Falls back to the OCR text itself."""
    key = _norm(ocr_text)
    if len(key) < ANCHOR_CHARS:
        return ocr_text
    best: tuple[int, str] | None = None
    for content in contents:
        content = " ".join(content.split())
        idx = _nonspace_index(content)
        flat = "".join(content[i] for i in idx).lower()
        sm = difflib.SequenceMatcher(None, key, flat, autojunk=False)
        blocks = [m for m in sm.get_matching_blocks() if m.size >= 2]
        if not blocks:
            continue
        # the passage must cover the reading from its start to its end
        if blocks[0].a > 2 or blocks[-1].a + blocks[-1].size < len(key) - 2:
            continue
        matched = sum(m.size for m in blocks)
        start, end = blocks[0].b, blocks[-1].b + blocks[-1].size
        if matched < VL_WORDING_MIN_MATCH * len(key) or end - start > VL_WORDING_MAX_SPAN * len(key):
            continue
        s, e = idx[start], idx[end - 1] + 1
        # grow to whole words only over a letter or two the OCR dropped: a
        # line starting/ending mid-word is one half of a hyphenated word
        s2, e2 = s, e
        while s2 > 0 and not content[s2 - 1].isspace():
            s2 -= 1
        while e2 < len(content) and not content[e2].isspace():
            e2 += 1
        s = s2 if s - s2 <= 2 else s
        hyphen = ocr_text.rstrip().endswith("-")
        e = e2 if e2 - e <= 2 and not hyphen else e
        passage = content[s:e] + ("-" if hyphen and e < e2 and not content[e - 1] == "-" else "")
        if best is None or matched > best[0]:
            best = (matched, passage)
    return best[1] if best else ocr_text


def reconcile(layout: PageLayout, page_lines: PageLines) -> Reconciliation:
    text_blocks = [b for b in layout.blocks if b.label not in IMAGE_LABELS and b.content]
    pictures = [b for b in layout.blocks if b.label in IMAGE_LABELS]
    lines = [_clip_out_of_pictures(piece, pictures)
             for line in page_lines.lines for piece in _split_across_blocks(line, text_blocks)]
    owned, unclaimed = _assign(lines, text_blocks, pictures)
    orphans, continuation = _attach_continuations(owned, unclaimed, text_blocks)

    result = Reconciliation(orphans=orphans)
    for block in text_blocks:
        blines = order_lines(owned[block.id])
        if not blines:
            continue
        widths = [_bbox(l.polygon)[2] - _bbox(l.polygon)[0] for l in blines]
        texts, ratio = split_text_to_lines(block.content, [l.text for l in blines], widths)
        result.blocks[block.id] = BlockLines(
            block_id=block.id,
            lines=[LinePlacement(polygon=_ordered_quad(l.polygon), text=t, ocr_text=l.text,
                                 score=l.score, continuation=id(l) in continuation)
                   for l, t in zip(blines, texts)],
            text_match=round(ratio, 3),
        )
    return result
