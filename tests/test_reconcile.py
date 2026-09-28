import pytest

from docir.reconcile import clip_quad, order_lines, reconcile, split_text_to_lines, vl_wording
from docir.schemas import Block, PageLayout, PageLines, TextLine


def _line(x0, y0, x1, y1, text, score=0.95):
    return TextLine(polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)], text=text, score=score)


def test_split_follows_ocr_line_breaks_with_vl_wording():
    content = "The quick brown fox jumps over the lazy dog near the river bank."
    ocr = ["The quick brownfox", "jumps overthe lazy dog", "near the river bank."]
    pieces, ratio = split_text_to_lines(content, ocr, [100, 120, 100])
    assert pieces == ["The quick brown fox", "jumps over the lazy dog", "near the river bank."]
    assert ratio > 0.8


def test_split_restores_hyphen_of_rejoined_word():
    content = "The Teej Sakhi Celebration at Grand Omaxe was vibrant"
    pieces, _ = split_text_to_lines(content, ["The Teej Sakhi Celebra-", "tion at Grand Omaxe was vibrant"],
                                    [100, 100])
    assert pieces == ["The Teej Sakhi Celebra-", "tion at Grand Omaxe was vibrant"]


def test_split_breaks_mid_word_when_next_line_is_anchored_there():
    content = "The Teej Sakhi Celebration at Grand Omaxe was"
    pieces, _ = split_text_to_lines(content, ["", "tion at Grand Omaxe was"], [100, 100])
    assert pieces == ["The Teej Sakhi Celebra-", "tion at Grand Omaxe was"]


def test_unseen_tail_of_vl_text_is_dropped():
    content = "Teej Tambola: A fun Teej-themed Tambo- la was organized, where everyone participated."
    pieces, _ = split_text_to_lines(content, ["Teej Tambola:Afun Teej-themed Tamo"], [270])
    assert pieces == ["Teej Tambola: A fun Teej-themed Tambo-"]


def test_unread_lines_are_placed_by_width():
    content = "aaaa bbbb cccc dddd eeee ffff"
    pieces, _ = split_text_to_lines(content, ["", "", ""], [100, 100, 100])
    assert pieces == ["aaaa bbbb", "cccc dddd", "eeee ffff"]


def test_vl_wording_recovers_spacing_and_hyphen():
    contents = ["Tambo- la was organized, where everyone participated enthusiastically and enjoyed the game."]
    assert vl_wording("ed enthusiasticallyandenjoyedtheame", contents) == \
        "ed enthusiastically and enjoyed the game."
    assert vl_wording("la was organized, where everyone participat-", contents) == \
        "la was organized, where everyone participat-"


def test_clip_quad_follows_tilt():
    q = clip_quad([(0, 0), (100, 10), (100, 30), (0, 20)], 50, 100)
    assert q[0] == pytest.approx((50, 5))
    assert q[3] == pytest.approx((50, 25))


def test_order_lines_rows_then_columns():
    a, b, c = _line(0, 0, 50, 10, "a"), _line(60, 1, 100, 11, "b"), _line(0, 20, 50, 30, "c")
    assert [l.text for l in order_lines([c, b, a])] == ["a", "b", "c"]


def test_reconcile_assigns_splits_and_continues():
    layout = PageLayout(width=400, height=300, blocks=[
        Block(id=0, label="text", content="left one left two left three", bbox=(10, 10, 150, 45)),
        Block(id=1, label="text", content="right one", bbox=(200, 10, 350, 25)),
        Block(id=2, label="image", content="", bbox=(10, 200, 100, 290)),
    ])
    lines = PageLines(width=400, height=300, lines=[
        _line(10, 10, 350, 24, "left one right one"),   # runs across the column gap
        _line(10, 28, 150, 42, "left two"),
        _line(10, 48, 150, 62, "left three"),            # below block 0's box
        _line(20, 220, 90, 235, "COVER"),                # printed on the picture
    ])
    rec = reconcile(layout, lines)
    assert [l.text for l in rec.blocks[0].lines] == ["left one", "left two", "left three"]
    assert rec.blocks[0].lines[-1].continuation
    assert [l.text for l in rec.blocks[1].lines] == ["right one"]
    assert rec.orphans == []
