"""Compare VLM backends using fresh full-pipeline conversions, not OCR caches."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import shutil
import statistics
import time


class PipelineTimings(logging.Handler):
    def __init__(self):
        super().__init__()
        self.stages: dict[str, float] = {}

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == "docir.pipeline" and record.msg.startswith("pipeline %s:"):
            self.stages = {
                name: float(value) for name, value in re.findall(
                    r"(total|preprocess|layout|lines_and_json|render)=([0-9.]+)s",
                    record.getMessage(),
                )
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="PDFs or page images.")
    parser.add_argument("--backend", choices=("native", "vllm-server"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3, help="First round is warm-up.")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if args.rounds < 2 or args.dpi < 1:
        parser.error("--rounds must be at least 2 and --dpi must be positive")
    image_suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    for path in args.inputs:
        if not path.is_file():
            parser.error(f"Input does not exist: {path}")
        if path.suffix.lower() not in image_suffixes | {".pdf"}:
            parser.error(f"Unsupported input format: {path}")

    import paddle
    import pypdfium2 as pdfium

    from docir.config import load_config, paddle_device_kwargs
    from docir.ocr import PaddleLineOcrEngine, PaddleOcrVlEngine
    from docir.pipeline import process_image, score_pdf_file

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    timings = PipelineTimings()
    logging.getLogger("docir.pipeline").addHandler(timings)
    # Override only this benchmark process, never the application's config file.
    load_config()["vlm"]["backend"] = args.backend
    paddle.disable_static()
    device = paddle_device_kwargs().get("device")
    if device is None:
        device = "gpu:0" if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() else "cpu"
    paddle.set_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pages: list[tuple[str, Path, int | None]] = []
    for number, path in enumerate(args.inputs):
        if path.suffix.lower() == ".pdf":
            with pdfium.PdfDocument(path) as document:
                pages.extend(
                    (f"{number:02d}-{path.stem}-page{index + 1}", path, index)
                    for index in range(len(document))
                )
        else:
            pages.append((f"{number:02d}-{path.stem}", path, None))

    engine, line_engine = PaddleOcrVlEngine(), PaddleLineOcrEngine()
    rows = []
    for iteration in range(args.rounds):
        for name, source_path, page_index in pages:
            page_dir = args.output_dir / f"{name}-r{iteration}"
            page_dir.mkdir()
            stem = page_dir / name
            suffix = ".png" if page_index is not None else source_path.suffix.lower()
            image_path = page_dir / f"{name}{suffix}"
            output_pdf = f"{stem}.pdf"
            output_json = f"{stem}_res.json"
            timings.stages = {}
            if device.startswith("gpu"):
                paddle.device.synchronize()
            end_to_end_start = time.perf_counter()
            if page_index is None:
                shutil.copyfile(source_path, image_path)
            else:
                with pdfium.PdfDocument(source_path) as document:
                    page = document[page_index]
                    try:
                        bitmap = page.render(scale=args.dpi / 72)
                        try:
                            bitmap.to_pil().convert("RGB").save(image_path)
                        finally:
                            bitmap.close()
                    finally:
                        page.close()
            input_seconds = time.perf_counter() - end_to_end_start
            start = time.perf_counter()
            process_image(
                str(image_path), output_pdf=output_pdf, output_json=output_json,
                engine=engine, line_engine=line_engine,
            )
            if device.startswith("gpu"):
                paddle.device.synchronize()
            elapsed = time.perf_counter() - start
            score_start = time.perf_counter()
            score = score_pdf_file(output_json, output_pdf)
            score_seconds = time.perf_counter() - score_start
            end_to_end_seconds = time.perf_counter() - end_to_end_start
            row = dict(
                backend=args.backend, page=name, iteration=iteration, warm=iteration > 0,
                seconds=elapsed, stages=dict(timings.stages),
                input_seconds=input_seconds, score_seconds=score_seconds,
                end_to_end_seconds=end_to_end_seconds, source=str(source_path.resolve()),
                precision=score.precision, recall=score.recall, f1=score.f1,
                pdf=output_pdf, layout_json=output_json,
            )
            rows.append(row)
            (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2))
            print(f"RESULT {json.dumps(row)}", flush=True)
    summary = []
    for name, _, _ in pages:
        warm = [r for r in rows if r["page"] == name and r["warm"]]
        cold = next(r for r in rows if r["page"] == name and not r["warm"])
        summary.append(dict(
            page=name, backend=args.backend, warm_runs=len(warm),
            median_seconds=statistics.median(r["seconds"] for r in warm),
            median_layout_seconds=statistics.median(r["stages"]["layout"] for r in warm),
            median_f1=statistics.median(r["f1"] for r in warm),
            median_score_seconds=statistics.median(r["score_seconds"] for r in warm),
            median_end_to_end_seconds=statistics.median(r["end_to_end_seconds"] for r in warm),
            warmup_end_to_end_seconds=cold["end_to_end_seconds"],
        ))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"SUMMARY {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
