import pytest

from docir.page_model import Block, Line, TextStyle
from docir.typography import harmonize_styles


def _block(bid, style, text, length=300.0):
    line = Line(text=text, baseline_start=(0, 40 * bid), baseline_end=(length, 40 * bid), x_height=9, cap_top=13)
    return Block(id=bid, role="text", reading_index=bid, bbox=(0, 0, 400, 400), content=text,
                 lines=[line], style=style)


def _style(family, size, bold=False, italic=False, category="serif"):
    return TextStyle(font_family=family, category=category, bold=bold, italic=italic, size_px=size)


BODY = "the quick brown fox jumps over the lazy dog again"


def test_near_sizes_and_faces_become_one_style():
    blocks = [_block(0, _style("NotoSerif", 17.6), BODY * 3),
              _block(1, _style("NotoSerif", 18.6), BODY),
              _block(2, _style("LiberationSerif", 16.9), BODY)]
    assert harmonize_styles(blocks) == 1
    assert len({id(b.style) for b in blocks}) == 1
    assert blocks[0].style.font_family == "NotoSerif"
    assert blocks[0].style.size_px == pytest.approx(17.6)


def test_distinct_sizes_bold_and_display_titles_stay_apart():
    blocks = [_block(0, _style("LiberationSerif", 14.0), BODY * 4),
              _block(1, _style("LiberationSerif", 17.5), BODY * 2),
              _block(2, _style("NotoSerif", 17.5, bold=True), "Notes"),
              _block(3, _style("DejaVuSerif", 40.0, bold=True), "A Big Display Title")]
    harmonize_styles(blocks)
    assert blocks[0].style.size_px != blocks[1].style.size_px
    assert blocks[2].style.bold and blocks[2].style.font_family == "LiberationSerif"
    assert blocks[3].style.font_family == "DejaVuSerif"


def test_short_text_adopts_slant_of_nearby_style():
    blocks = [_block(0, _style("LiberationSans", 30.0, bold=True, category="sans_serif"), BODY * 2),
              _block(1, _style("LiberationSans", 32.0, bold=True, italic=True, category="sans_serif"), "37")]
    harmonize_styles(blocks)
    assert not blocks[1].style.italic
    assert blocks[1].style is blocks[0].style
