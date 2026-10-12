"""Re-flow a block's text through its measured line slots.

Line breaks normally come from the OCR reading of each line. On small or
degraded print that reading can be garbage, and a line then receives far
more (or less) text than its slot can hold -- it overflows into the next
column or leaves a stray word behind. Since the typeface, size and every
slot's width are known, the breaks can instead be chosen so that each
line's rendered width matches its slot: optimal line breaking with fixed
line lengths (dynamic programming over word boundaries). Lines keep their
measured positions; only the words move.
"""

from __future__ import annotations

import math

import numpy as np

from . import font_catalog
from .page_model import Block

# beyond what the renderer can absorb (glyph width -12%, gaps) -> the line overflows
OVERFLOW_TRIGGER_LOG = math.log(1.15)
UNDERFILL_TRIGGER_LOG = math.log(0.55)  # a slot left mostly empty (stray word)
OVERFLOW_WEIGHT = 4.0            # overflow collides with neighbours; slack only spaces out
SIZE_CLASS_RATIO = 1.25          # a line whose x-height is this much larger is set in a larger size
SIZE_CLASS_MAX_RATIO = 1.6       # ... beyond this it is a mis-measurement (lines merged), not a size


def _pieces(texts: list[str]) -> tuple[list[str], list[bool]]:
    """Words of the block's lines, and whether each is preceded by a space.
    A word hyphenated across two lines becomes two unspaced pieces without
    the hyphen, so it rejoins whole unless a break falls between them."""
    words, spaced = [], []
    for k, text in enumerate(texts):
        for i, w in enumerate(text.split()):
            glued = i == 0 and k > 0 and words and words[-1].endswith("-") \
                and words[-1][-2:-1].isalpha() and w[:1].islower()
            if glued:
                words[-1] = words[-1][:-1]
            words.append(w)
            spaced.append(bool(words[:-1]) and not glued)
    return words, spaced


def _line_cost(natural: float, target: float) -> float:
    r = math.log(max(natural, 1e-6) / max(target, 1e-6))
    return r * r * (OVERFLOW_WEIGHT if r > 0 else 1.0)


def _join(words: list[str], spaced: list[bool]) -> str:
    out = ""
    for i, (w, s) in enumerate(zip(words, spaced)):
        out += (" " if s and i else "") + w
    return out


def _targets(lines) -> list[float]:
    """Each slot's width in block-size text: a line printed clearly larger
    than the block's typical line holds proportionally fewer characters
    per pixel."""
    med = float(np.median([l.x_height for l in lines]))
    out = []
    for l in lines:
        r = l.x_height / med if med > 0 else 1.0
        scaled = SIZE_CLASS_RATIO <= r <= SIZE_CLASS_MAX_RATIO
        out.append(l.length / r if scaled else l.length)
    return out


def refit_block(block: Block) -> bool:
    """Re-flow the block's words through its slots if some line misfits
    and the re-flow fits better. Returns whether it changed anything."""
    style = block.style
    lines = [l for l in block.lines if not l.drop_cap]
    if style is None or len(lines) < 2 or block.tabular:
        return False
    face = font_catalog.face(style.font_family, style.bold, style.italic)
    size, hz = style.size_px, style.width_scale

    def natural(text: str) -> float:
        return font_catalog.string_width(text, face.name, size) * hz

    current = [natural(l.text) for l in lines]
    targets = _targets(lines)
    ratios = [math.log(max(n, 1e-6) / max(t, 1e-6)) for n, t in zip(current, targets)]
    # the last line of a paragraph is legitimately short
    if not any(r > OVERFLOW_TRIGGER_LOG for r in ratios) and \
            not any(r < UNDERFILL_TRIGGER_LOG for r in ratios[:-1]):
        return False
    words, spaced = _pieces([l.text for l in lines])
    m, n = len(words), len(lines)
    if m < n:
        return False
    word_w = np.array([natural(w) for w in words])
    space_w = natural(" ")
    gaps = np.array([space_w if s else 0.0 for s in spaced])
    # width of words[i:j] = cw[j] - cw[i] + (cg[j] - cg[i+1]) (gaps between them)
    cw = np.concatenate([[0.0], np.cumsum(word_w)])
    cg = np.concatenate([[0.0], np.cumsum(gaps)])

    def width(i: int, j: int) -> float:
        return cw[j] - cw[i] + cg[j] - cg[i + 1]

    inf = float("inf")
    # best[k][j]: cost of placing words[:j] on the first k lines
    best = np.full((n + 1, m + 1), inf)
    back = np.zeros((n + 1, m + 1), dtype=int)
    best[0][0] = 0.0
    for k in range(1, n + 1):
        target = targets[k - 1]
        for j in range(k, m - (n - k) + 1):
            for i in range(k - 1, j):
                if best[k - 1][i] == inf:
                    continue
                c = best[k - 1][i] + _line_cost(width(i, j), target)
                if c < best[k][j]:
                    best[k][j], back[k][j] = c, i
    if best[n][m] == inf:
        return False
    old = sum(_line_cost(c, t) for c, t in zip(current, targets))
    if best[n][m] >= old:
        return False
    cuts, j = [], m
    for k in range(n, 0, -1):
        i = back[k][j]
        cuts.append((i, j))
        j = i
    for line, (i, j) in zip(lines, reversed(cuts)):
        seg_spaced = [False] + spaced[i + 1:j]
        text = _join(words[i:j], seg_spaced)
        # the break falls inside a word: hyphenate it as the original did
        if j < m and not spaced[j]:
            text += "-"
        line.text = text
    return True


def refit_page(blocks: list[Block]) -> int:
    return sum(refit_block(b) for b in blocks)
