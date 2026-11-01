"""Form handling: typewriter detection, sideways text, rules inside text,
and the layout model's form markup."""

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from docir import font_catalog
from docir.ink import measure_text_lines, without_rules
from docir.page_model import Block, Line
from docir.pitch import looks_monospaced
from docir.schemas import clean_vl_text
from docir.vertical import is_vertical, vertical_line

TEXTS = ["the quick brown fox jumps over", "lazy dogs sleep in the warm sun",
         "while seven zebras quietly wander", "home again after a long day out",
         "every illusion fills a wide mirror", "minimal impact with little width"]


def _block(face_key: str) -> Block:
    name = font_catalog.face(face_key).name
    lines = [Line(text=t, baseline_start=(0, 40 * i), baseline_end=(font_catalog.string_width(t, name, 20), 40 * i),
                  x_height=10, cap_top=14) for i, t in enumerate(TEXTS)]
    return Block(id=0, role="text", reading_index=0, bbox=(0, 0, 900, 300), content="\n".join(TEXTS), lines=lines)


def test_typewriter_text_is_told_from_line_lengths():
    assert looks_monospaced([_block("LiberationMono")])
    assert not looks_monospaced([_block("LiberationSans")])


def _sideways(text: str, k: int) -> np.ndarray:
    font = ImageFont.truetype(font_catalog.face("LiberationSans").path, 30)
    img = Image.new("L", (int(font.getlength(text)) + 40, 60), 255)
    ImageDraw.Draw(img).text((20, 45), text, font=font, fill=0, anchor="ls")
    page = np.full((600, 300), 255, np.uint8)
    turned = np.rot90(np.asarray(img), -k)    # undo of the upright turn
    page[100:100 + turned.shape[0], 120:120 + turned.shape[1]] = turned
    return page, (120, 100, 120 + turned.shape[1], 100 + turned.shape[0])


@pytest.mark.parametrize("k, angle", [(1, -90), (-1, 90)])
def test_sideways_stamp_becomes_one_turned_line(k, angle):
    page, bbox = _sideways("87052570", k)
    assert is_vertical(bbox, "87052570")
    line = vertical_line(page, bbox, "87052570")
    assert line is not None and line.angle_deg == pytest.approx(angle, abs=2)
    assert 120 <= line.baseline_start[0] <= bbox[2]


def test_underline_is_not_counted_as_a_text_line():
    font = ImageFont.truetype(font_catalog.face("LiberationSans").path, 30)
    img = Image.new("L", (700, 70), 255)
    d = ImageDraw.Draw(img)
    d.text((20, 40), "SUBJECT: STYLE LIGHTS PROGRESS REPORT", font=font, fill=0, anchor="ls")
    d.line([(150, 50), (680, 50)], fill=0, width=3)
    mask = np.asarray(img) < 128
    assert not without_rules(mask)[49:52].any()
    assert measure_text_lines(mask, ignore_rules=True).n_lines == 1


def test_form_markup_and_blanks_are_cleaned_but_currency_stays():
    assert clean_vl_text("Mailing Address: $ \\underline{\\text{PO Box 35090}} $") == "Mailing Address: PO Box 35090"
    assert clean_vl_text("Company: ___ Date: _____") == "Company: Date:"
    assert clean_vl_text("a gift of more than $100 in value") == "a gift of more than $100 in value"
    assert clean_vl_text("file_name stays") == "file_name stays"
    assert clean_vl_text(" $ a_{x} $") == "ax"
    assert clean_vl_text(" $ \\frac{1}{6110126} $") == "1/6110126"
    assert clean_vl_text("costs $5 or {x} $6") == "costs $5 or {x} $6"


def test_math_symbols_print_as_characters_not_markup():
    assert clean_vl_text("<td>$ \\pm5\\% $</td>") == "<td>±5%</td>"
    assert clean_vl_text("on our  $ 16 \\times 16 $ grids") == "on our 16 × 16 grids"
    assert clean_vl_text("GT Diff $ \\leq 10\\% $") == "GT Diff ≤ 10%"
    assert clean_vl_text("25 $ ^{\\circ} $C") == "25°C"
    assert clean_vl_text("$ \\alpha = 0.05 $, $ \\mathrm{p} < 10^{-3} $") == "α = 0.05, p < 10-3"
    assert clean_vl_text("C:\\temp\\pm stays outside math") == "C:\\temp\\pm stays outside math"
