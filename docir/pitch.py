"""Typewriter (monospaced) text, recognised from measured line lengths.

In a monospaced face every character advances the same distance, so within
one block (one size) a line's length is proportional to its character count.
In a proportional face it follows the glyphs' own widths instead. Whichever
model predicts the block's line lengths better tells which kind of face
printed it -- no glyph shapes needed, which scans and faxes blur away.
"""

from __future__ import annotations

import numpy as np

from . import font_catalog
from .page_model import Block

MIN_CHARS = 12          # shorter lines are too noisy to judge pitch
MIN_LINES = 6           # pooled lines needed before deciding
MONO_RATIO = 0.35       # monospace fits this much better than proportional: typewriter
MONO_AGREE_RATIO = 1.0  # on a typewriter page, a role where monospace still fits better is typed too


def pitch_ratio(blocks: list[Block]) -> tuple[float, int] | None:
    """(monospace deviation / proportional deviation, lines used): each
    model's median spread of per-line pitch around its block's median."""
    reference = font_catalog.face(font_catalog.DEFAULT_FAMILY["sans_serif"]).name
    mono, prop = [], []
    for b in blocks:
        lines = [l for l in b.lines if len(l.text.strip()) >= MIN_CHARS and l.length > 0 and not l.drop_cap]
        if len(lines) < 2:
            continue
        m = np.array([l.length / len(l.text.strip()) for l in lines])
        p = np.array([l.length / font_catalog.string_width(l.text.strip(), reference, 1.0) for l in lines])
        mono += list(np.abs(np.log(m / np.median(m))))
        prop += list(np.abs(np.log(p / np.median(p))))
    if len(mono) < MIN_LINES:
        return None
    return float(np.median(mono)) / max(float(np.median(prop)), 1e-9), len(mono)


def looks_monospaced(blocks: list[Block]) -> bool:
    r = pitch_ratio(blocks)
    return r is not None and r[0] < MONO_RATIO
