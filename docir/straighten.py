"""Level small tilts: a line leaning a degree or two is almost always a
scanning / photographing artefact, not part of the design, so the output
draws it straight.

Each block with a small median tilt is rotated rigidly about its own
centre (so its lines keep their spacing and alignment, and the left edge
of a tilted paragraph doesn't stay staggered), then every line is made
exactly horizontal. Nearly horizontal / vertical rules are levelled too.
Larger angles are kept -- they are deliberate (rotated labels, stamps).
"""

from __future__ import annotations

import math

import numpy as np

from .page_model import Block, Page, Point


def _rotate(p: Point, pivot: Point, angle_deg: float) -> Point:
    # page y grows downwards; this undoes a counter-clockwise tilt of angle_deg
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    dx, dy = p[0] - pivot[0], p[1] - pivot[1]
    return pivot[0] + c * dx - s * dy, pivot[1] + s * dx + c * dy


def straighten_block(block: Block, max_deg: float) -> bool:
    if not block.lines:
        return False
    tilt = float(np.median([l.angle_deg for l in block.lines]))
    if abs(tilt) > max_deg:
        return False
    mids = [((l.baseline_start[0] + l.baseline_end[0]) / 2, (l.baseline_start[1] + l.baseline_end[1]) / 2)
            for l in block.lines]
    pivot = (float(np.mean([m[0] for m in mids])), float(np.mean([m[1] for m in mids])))
    for line in block.lines:
        start = _rotate(line.baseline_start, pivot, tilt)
        end = _rotate(line.baseline_end, pivot, tilt)
        if abs(line.angle_deg - tilt) <= max_deg:
            y = (start[1] + end[1]) / 2
            start, end = (start[0], y), (end[0], y)
        line.baseline_start, line.baseline_end = start, end
    return True


def straighten_rules(page: Page, max_deg: float) -> None:
    for g in page.graphics:
        if g.kind != "rule" or len(g.points) != 2:
            continue
        (x0, y0), (x1, y1) = g.points
        angle = math.degrees(math.atan2(abs(y1 - y0), abs(x1 - x0)))
        if angle <= max_deg:
            y = (y0 + y1) / 2
            g.points = [(x0, y), (x1, y)]
        elif angle >= 90 - max_deg:
            x = (x0 + x1) / 2
            g.points = [(x, y0), (x, y1)]


def straighten_page(page: Page, max_deg: float) -> int:
    """Level every block tilted by at most `max_deg` degrees (0 disables);
    returns how many blocks were levelled."""
    if max_deg <= 0:
        return 0
    n = sum(straighten_block(b, max_deg) for b in page.blocks)
    straighten_rules(page, max_deg)
    return n
