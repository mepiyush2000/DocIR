import json
import logging
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from docir import config, ocr, pipeline


@pytest.mark.parametrize("suffix", [".jpg", ".pdf"])
def test_benchmark_uses_fresh_inputs_and_reports_end_to_end(tmp_path, monkeypatch, suffix):
    source = tmp_path / f"sample.page{suffix}"
    Image.new("RGB", (80, 100), "white").save(source)
    original = source.read_bytes()
    output = tmp_path / "benchmark"
    monkeypatch.setattr(sys, "argv", [
        "benchmark_vlm.py", str(source), "--backend", "native",
        "--output-dir", str(output), "--rounds", "2",
    ])
    monkeypatch.setitem(sys.modules, "paddle", SimpleNamespace(
        disable_static=lambda: None, set_device=lambda device: None))
    monkeypatch.setattr(config, "paddle_device_kwargs", lambda: {"device": "cpu"})
    monkeypatch.setattr(config, "load_config", lambda: {"vlm": {"backend": "native"}})
    monkeypatch.setattr(ocr, "PaddleOcrVlEngine", lambda: object())
    monkeypatch.setattr(ocr, "PaddleLineOcrEngine", lambda: object())
    processed = []

    def process(image_path, *, output_pdf, output_json, **kwargs):
        image = Path(image_path)
        assert image.parent != source.parent
        assert image.is_file()
        assert not Path(output_json).exists()
        processed.append(image)
        image.with_name(f"{image.stem}_flat.png").write_bytes(b"preprocessed")
        Path(output_pdf).write_bytes(b"pdf")
        Path(output_json).write_text("{}")
        logging.getLogger("docir.pipeline").info(
            "pipeline %s: total=%.3fs preprocess=%.3fs layout=%.3fs lines_and_json=%.3fs render=%.3fs",
            image_path, 0.005, 0.001, 0.002, 0.001, 0.001,
        )

    monkeypatch.setattr(pipeline, "process_image", process)
    monkeypatch.setattr(pipeline, "score_pdf_file", lambda *args: SimpleNamespace(
        precision=0.9, recall=0.8, f1=0.85))
    logger = logging.getLogger("docir.pipeline")
    handlers = logger.handlers[:]
    level = logger.level
    logger.setLevel(logging.INFO)
    try:
        runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "benchmark_vlm.py"), run_name="__main__")
    finally:
        logger.handlers[:] = handlers
        logger.setLevel(level)
    assert source.read_bytes() == original
    assert len(processed) == len(set(processed)) == 2
    rows = json.loads((output / "results.json").read_text())
    assert [row["warm"] for row in rows] == [False, True]
    for row in rows:
        assert row["stages"]["layout"] == 0.002
        assert row["f1"] == 0.85
        assert row["end_to_end_seconds"] >= (
            row["input_seconds"] + row["seconds"] + row["score_seconds"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary[0]["warm_runs"] == 1
    assert summary[0]["median_end_to_end_seconds"] == rows[1]["end_to_end_seconds"]
    assert summary[0]["warmup_end_to_end_seconds"] == rows[0]["end_to_end_seconds"]
