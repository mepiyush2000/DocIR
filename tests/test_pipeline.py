import os

from docir.ocr import FakeOcrEngine
from docir.pipeline import analyze_image, process_image, render_json_file

from .conftest import FIXTURE_IMAGE, FIXTURE_JSON


def test_analyze_image_uses_injected_fake_engine(fixture_layout):
    engine = FakeOcrEngine(fixture_layout)
    result = analyze_image("unused.png", engine=engine)
    assert result is fixture_layout


def test_process_image_end_to_end_with_fake_engine(tmp_path, fixture_layout):
    engine = FakeOcrEngine(fixture_layout)
    output_pdf = tmp_path / "out.pdf"
    output_json = tmp_path / "out.json"

    result_path = process_image(
        FIXTURE_IMAGE, output_pdf=str(output_pdf), output_json=str(output_json), engine=engine,
    )

    assert result_path == str(output_pdf)
    assert os.path.isfile(output_pdf)
    assert os.path.getsize(output_pdf) > 1000
    assert os.path.isfile(output_json)


def test_render_json_file_legacy_entry_point(tmp_path):
    output_pdf = tmp_path / "out.pdf"
    result_path = render_json_file(FIXTURE_JSON, str(output_pdf))
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
