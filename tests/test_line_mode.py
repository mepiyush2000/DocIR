"""End-to-end line mode: a synthetic page rendered with a known font is
analysed into a page model and re-rendered; the output must match it."""

import os

import pytest
from PIL import Image, ImageDraw, ImageFont

from docir import font_catalog
from docir.fidelity import score_pdf
from docir.ocr import FakeLineOcrEngine
from docir.page_builder import build_page
from docir.pipeline import render_json_file
from docir.render_model import render_page_model
from docir.schemas import Block, PageLayout, PageLines, TextLine

from .conftest import FIXTURE_JSON

TEXT = ["the quick brown fox jumps over the lazy dog", "while seven zebras quietly wander home",
        "again and again until the evening ends"]


def _page():
    font = ImageFont.truetype(font_catalog.face("LiberationSerif").path, 26)
    img = Image.new("RGB", (1000, 400), "white")
    d = ImageDraw.Draw(img)
    for i, t in enumerate(TEXT):
        d.text((80, 100 + i * 40), t, font=font, fill="black", anchor="ls")
    d.text((80, 50), "Heading", font=ImageFont.truetype(font_catalog.face("LiberationSerif", True).path, 36),
           fill="black", anchor="ls")
    layout = PageLayout(width=1000, height=400, blocks=[
        Block(id=0, label="paragraph_title", content="Heading", bbox=(75, 18, 260, 62)),
        Block(id=1, label="text", content=" ".join(TEXT), bbox=(75, 75, 720, 190)),
    ])
    lines = PageLines(width=1000, height=400, lines=[
        TextLine(polygon=[(76, 78 + i * 40), (700, 78 + i * 40), (700, 108 + i * 40), (76, 108 + i * 40)],
                 text=t, score=0.99) for i, t in enumerate(TEXT)])
    return img, layout, lines


def test_page_model_has_measured_lines_and_one_style_per_role():
    img, layout, lines = _page()
    page = build_page(layout, img, lines)
    body = next(b for b in page.blocks if b.id == 1)
    assert [l.text for l in body.lines] == TEXT
    assert body.style.category == "serif"
    assert body.style.size_px == pytest.approx(26, rel=0.1)
    assert body.lines[0].baseline_start[1] == pytest.approx(100, abs=2)


def test_line_mode_render_matches_source(tmp_path):
    img, layout, lines = _page()
    out = str(tmp_path / "p.pdf")
    render_page_model(build_page(layout, img, lines), out, source_image=img)
    report, _ = score_pdf(layout, out, img)
    assert report.f1 > 0.9


def test_render_json_file_lines_mode_uses_injected_engine(tmp_path):
    # a copy of the fixture JSON so the line cache lands in tmp_path
    json_copy = tmp_path / "fixture_res.json"
    json_copy.write_text(open(FIXTURE_JSON).read())
    engine = FakeLineOcrEngine(PageLines(width=1, height=1, lines=[]))
    from .conftest import FIXTURE_IMAGE
    out = render_json_file(str(json_copy), str(tmp_path / "o.pdf"), source_image=FIXTURE_IMAGE,
                           mode="lines", line_engine=engine)
    assert os.path.getsize(out) > 1000
    assert (tmp_path / "fixture_lines.json").is_file()
