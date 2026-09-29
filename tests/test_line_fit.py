import pytest

from docir import font_catalog
from docir.line_fit import refit_block
from docir.page_model import Block, Line, TextStyle

STYLE = TextStyle(font_family="LiberationSerif", category="serif", size_px=20.0)


def _natural(text: str) -> float:
    face = font_catalog.face(STYLE.font_family)
    return font_catalog.string_width(text, face.name, STYLE.size_px)


def _block(texts, lengths):
    lines = [Line(text=t, baseline_start=(0, 30 * i), baseline_end=(length, 30 * i), x_height=9, cap_top=13)
             for i, (t, length) in enumerate(zip(texts, lengths))]
    return Block(id=0, role="text", reading_index=0, bbox=(0, 0, 600, 200), content=" ".join(texts),
                 lines=lines, style=STYLE)


def test_overflowing_line_is_reflowed_into_its_slots():
    good = ["the quick brown fox jumps over", "the lazy dog while seven zebras", "wander home"]
    slots = [_natural(t) for t in good]
    # misread breaks: the first line got most of the text, the second a stray word
    block = _block(["the quick brown fox jumps over the lazy dog while seven", "zebras",
                    "wander home"], slots)
    assert refit_block(block)
    assert [l.text for l in block.lines] == good


def test_fitting_block_is_left_alone():
    texts = ["the quick brown fox jumps over", "the lazy dog while seven zebras", "home"]
    block = _block(texts, [_natural(t) * 1.02 for t in texts])
    assert not refit_block(block)
    assert [l.text for l in block.lines] == texts


def test_hyphenated_word_rejoins_or_rehyphenates():
    good = ["alpha beta gamma delta", "epsilon zeta eta theta"]
    slots = [_natural(t) for t in good]
    block = _block(["alpha beta gamma delta epsi-", "lon zeta eta theta"], slots)
    assert refit_block(block)
    assert [l.text for l in block.lines] == good
    # slots sized for the split word: the break stays inside it, hyphenated
    split = ["alpha beta gamma delta epsi-", "lon zeta eta theta"]
    block = _block(["alpha beta gamma", "delta epsi- lon zeta eta theta"],
                   [_natural(split[0]), _natural(split[1])])
    assert refit_block(block)
    assert [l.text for l in block.lines] == split


def test_reflow_never_loses_or_reorders_words():
    block = _block(["one two three four five six seven eight nine ten eleven", "twelve", "thirteen"],
                   [_natural("one two three four"), _natural("five six seven eight"),
                    _natural("nine ten eleven twelve thirteen")])
    before = " ".join(l.text for l in block.lines).split()
    refit_block(block)
    assert " ".join(l.text for l in block.lines).split() == before
    assert all(l.text for l in block.lines)
