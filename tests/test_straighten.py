import pytest

from docir.page_model import Block, Graphic, Line, Page
from docir.straighten import straighten_page


def _tilted_line(x0, y0, length, angle_deg, text="abc"):
    import math
    a = math.radians(angle_deg)
    return Line(text=text, baseline_start=(x0, y0),
                baseline_end=(x0 + length * math.cos(a), y0 - length * math.sin(a)),
                x_height=10, cap_top=14)


def _block(angle):
    lines = [_tilted_line(100 + i * 30 * __import__("math").sin(__import__("math").radians(angle)),
                          100 + i * 30, 400, angle) for i in range(4)]
    return Block(id=0, role="text", reading_index=0, bbox=(90, 80, 520, 220), content="x", lines=lines)


def test_small_tilt_is_levelled_and_spacing_kept():
    block = _block(2.0)
    page = Page(width=800, height=600, blocks=[block])
    assert straighten_page(page, 5.0) == 1
    ys = [l.baseline_start[1] for l in block.lines]
    for l in block.lines:
        assert l.angle_deg == pytest.approx(0, abs=1e-6)
        assert l.length == pytest.approx(400, rel=1e-3)
    assert [b - a for a, b in zip(ys, ys[1:])] == pytest.approx([30, 30, 30], abs=0.5)
    # a tilted paragraph's staggered left edge becomes one aligned edge
    xs = [l.baseline_start[0] for l in block.lines]
    assert max(xs) - min(xs) < 0.5


def test_large_tilt_is_kept_and_zero_disables():
    block = _block(15.0)
    assert straighten_page(Page(width=800, height=600, blocks=[block]), 5.0) == 0
    assert block.lines[0].angle_deg == pytest.approx(15.0, abs=1e-6)
    small = _block(2.0)
    assert straighten_page(Page(width=800, height=600, blocks=[small]), 0) == 0


def test_nearly_straight_rules_are_levelled():
    rule = Graphic(kind="rule", bbox=(0, 0, 1, 1), points=[(100, 200), (600, 212)])
    vert = Graphic(kind="rule", bbox=(0, 0, 1, 1), points=[(300, 100), (306, 500)])
    straighten_page(Page(width=800, height=600, graphics=[rule, vert]), 5.0)
    assert rule.points[0][1] == rule.points[1][1] == pytest.approx(206)
    assert vert.points[0][0] == vert.points[1][0] == pytest.approx(303)
