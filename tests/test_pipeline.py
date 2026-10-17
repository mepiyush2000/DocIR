import os

from docir.ocr import FakeLineOcrEngine, FakeOcrEngine
from docir.pipeline import analyze_image, lines_json_path, process_image, render_json_file
from docir.schemas import PageLines

from .conftest import FIXTURE_IMAGE, FIXTURE_JSON


def _no_lines(layout) -> FakeLineOcrEngine:
    return FakeLineOcrEngine(PageLines(width=layout.width, height=layout.height, lines=[]))


def test_analyze_image_uses_injected_fake_engine(fixture_layout):
    engine = FakeOcrEngine(fixture_layout)
    result = analyze_image("unused.png", engine=engine)
    assert result is fixture_layout


def test_process_image_end_to_end_with_fake_engine(tmp_path, fixture_layout, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="docir.pipeline")
    engine = FakeOcrEngine(fixture_layout)
    output_pdf = tmp_path / "out.pdf"
    output_json = tmp_path / "out.json"

    result_path = process_image(
        FIXTURE_IMAGE, output_pdf=str(output_pdf), output_json=str(output_json), engine=engine,
        line_engine=_no_lines(fixture_layout), preprocess="off",
    )

    assert result_path == str(output_pdf)
    assert os.path.isfile(output_pdf)
    assert os.path.getsize(output_pdf) > 1000
    assert os.path.isfile(output_json)
    assert os.path.isfile(lines_json_path(str(output_json)))
    assert "total=" in caplog.text
    assert all(f"{stage}=" in caplog.text for stage in
               ("preprocess", "layout", "lines_and_json", "render"))


def test_render_json_file_legacy_entry_point(tmp_path):
    output_pdf = tmp_path / "out.pdf"
    result_path = render_json_file(FIXTURE_JSON, str(output_pdf), mode="paragraph")
    assert result_path == str(output_pdf)
    assert os.path.isfile(output_pdf)
    assert os.path.getsize(output_pdf) > 1000


def test_render_pdf_falls_back_gracefully_without_source_image(tmp_path, fixture_layout):
    """No source image resolvable -> still renders (regular weight, placeholders)."""
    from docir.pipeline import render_pdf_from_layout

    output_pdf = tmp_path / "out.pdf"
    result_path = render_pdf_from_layout(
        fixture_layout, str(output_pdf), source_image_path="does/not/exist.png",
    )
    assert os.path.isfile(result_path)


def test_source_image_and_lines_are_brought_into_layout_pixel_space():
    import pytest
    from PIL import Image

    from docir.pipeline import lines_in_layout_space, match_layout_space
    from docir.schemas import PageLayout, TextLine

    layout = PageLayout(width=1000, height=1400, blocks=[])
    assert match_layout_space(Image.new("RGB", (500, 700)), layout).size == (1000, 1400)
    with pytest.raises(ValueError):
        match_layout_space(Image.new("RGB", (700, 1400)), layout)
    half = PageLines(width=500, height=700, lines=[
        TextLine(polygon=[(10, 20), (110, 20), (110, 40), (10, 40)], text="x", score=1.0)])
    scaled = lines_in_layout_space(half, layout)
    assert scaled.lines[0].polygon[1] == (220, 40)
    assert lines_in_layout_space(PageLines(width=700, height=1400, lines=[]), layout) is None
