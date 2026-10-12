from PIL import Image, ImageDraw

from docir.color import assign_colors
from docir.page_model import Block, Line, Page

BAR = (10, 40, 140)


def _line(x0, x1, baseline, xh=12.0):
    return Line(text="Address", baseline_start=(x0, baseline), baseline_end=(x1, baseline),
                x_height=xh, cap_top=1.4 * xh)


def test_light_text_on_a_coloured_bar_gets_the_bar_as_fill():
    img = Image.new("RGB", (600, 300), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((40, 180, 560, 260), fill=BAR)
    for i in range(10):                                 # white word-like glyphs
        d.rectangle((120 + i * 30, 210, 128 + i * 30, 230), fill="white")
    block = Block(id=0, role="footer", reading_index=0, bbox=(110, 200, 420, 235),
                  content="Address", lines=[_line(120, 410, 230)])
    page = Page(width=600, height=300, blocks=[block])
    assign_colors(page, img)
    assert block.fill is not None and max(abs(a - b) for a, b in zip(block.fill, BAR)) < 10
    x0, y0, x1, y1 = block.fill_bbox
    assert abs(x0 - 40) <= 3 and abs(y0 - 180) <= 3 and abs(x1 - 561) <= 3 and abs(y1 - 261) <= 3
    assert min(block.lines[0].color) > 200


def test_dark_text_on_white_paper_has_no_panel():
    img = Image.new("RGB", (600, 300), "white")
    d = ImageDraw.Draw(img)
    for i in range(10):
        d.rectangle((120 + i * 30, 210, 140 + i * 30, 230), fill="black")
    block = Block(id=0, role="text", reading_index=0, bbox=(110, 200, 420, 235),
                  content="Address", lines=[_line(120, 410, 230)])
    page = Page(width=600, height=300, blocks=[block])
    assign_colors(page, img)
    assert block.fill is None and block.fill_bbox is None
