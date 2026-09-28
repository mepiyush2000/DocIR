"""Command-line entry point.

    docir analyze <image>            # raw image -> OCR/layout -> PDF
    docir render  <layout.json>       # legacy layout JSON -> PDF
    docir score   <layout.json> <pdf> # ink fidelity of a PDF vs. the scan
    docir lines   <layout.json>       # detect + cache text lines, show matching
    docir flatten <image>             # orientation / unwarp / deskew only
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import is_debug_logging_enabled
from .pipeline import detect_lines_for_json, process_image, render_json_file, score_pdf_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docir",
        description="Recover document layout via PaddleOCR-VL and rebuild a geometry-preserving PDF.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser(
        "analyze", help="Run OCR/layout detection on a raw image and render a PDF.",
    )
    p_analyze.add_argument("image_path", help="Path to the source page image.")
    p_analyze.add_argument(
        "output_pdf", nargs="?", default=None,
        help="Path to write the output PDF (default: <image_path> with .pdf extension).",
    )
    p_analyze.add_argument(
        "--json", dest="output_json", default=None,
        help="Also save the recovered layout as a versioned JSON file.",
    )
    p_analyze.add_argument(
        "--preprocess", choices=("auto", "off", "force"), default=None,
        help="Page flattening (default: [preprocess] mode in docir_config.toml).",
    )

    p_render = sub.add_parser(
        "render", help="Render a PDF from an existing layout JSON file (legacy input).",
    )
    p_render.add_argument("json_path", help="Path to a layout-detection JSON file.")
    p_render.add_argument(
        "output_pdf", nargs="?", default=None,
        help="Path to write the output PDF (default: <json_path> with .pdf extension).",
    )
    p_render.add_argument(
        "--source-image", default=None,
        help="Path to the original page image, used to detect bold vs. regular text "
             "weight and to embed real images (default: the JSON's input_path, "
             "resolved relative to json_path).",
    )
    p_render.add_argument(
        "--score", action="store_true",
        help="Score the rendered PDF against the source image afterwards.",
    )
    p_render.add_argument(
        "--mode", choices=("lines", "paragraph"), default=None,
        help="Line placement or paragraph reflow (default: [render] mode in the config).",
    )

    p_score = sub.add_parser(
        "score", help="Compare a rendered PDF's ink against the source scan.",
    )
    p_score.add_argument("json_path", help="Layout JSON the PDF was rendered from.")
    p_score.add_argument("pdf_path", help="PDF to score.")
    p_score.add_argument("--source-image", default=None,
                         help="Original page image (default: resolved from the JSON).")
    p_score.add_argument("--diff", default=None,
                         help="Write a diff PNG (red = missing ink, blue = extra ink).")
    p_score.add_argument("--report", default=None, help="Write the full JSON report.")
    p_score.add_argument("--worst", type=int, default=5,
                         help="How many lowest-scoring blocks to list (default 5).")

    p_lines = sub.add_parser(
        "lines", help="Detect (and cache) text lines for a layout JSON; summarize matching.",
    )
    p_lines.add_argument("json_path", help="Layout JSON (PaddleOCR-VL result).")
    p_lines.add_argument("--source-image", default=None,
                         help="Page image (default: resolved from the JSON).")
    p_lines.add_argument("--refresh", action="store_true",
                         help="Re-run line OCR even if <name>_lines.json exists.")

    p_flat = sub.add_parser(
        "flatten", help="Fix orientation / unwarp / deskew a page image (as analyze does).",
    )
    p_flat.add_argument("image_path")
    p_flat.add_argument("output_path", nargs="?", default=None)
    p_flat.add_argument("--mode", choices=("auto", "force"), default="auto")
    return parser


def _print_report(report, worst: int) -> None:
    print(f"page  P={report.precision:.3f}  R={report.recall:.3f}  F1={report.f1:.3f}  "
          f"stray_ink={report.stray_ink}px  blocks={len(report.blocks)}")
    for b in report.worst(worst):
        print(f"  block {b.block_id:>3} {b.label:<18} P={b.precision:.3f} R={b.recall:.3f} "
              f"F1={b.f1:.3f}")


def main(argv: list[str] | None = None) -> int:
    level = logging.DEBUG if is_debug_logging_enabled() else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)

    if args.command == "analyze":
        result_path = process_image(
            args.image_path, output_pdf=args.output_pdf, output_json=args.output_json,
            preprocess=args.preprocess,
        )
    elif args.command == "score":
        report = score_pdf_file(args.json_path, args.pdf_path, source_image=args.source_image,
                                diff_path=args.diff, report_path=args.report)
        _print_report(report, args.worst)
        return 0
    elif args.command == "lines":
        summary = detect_lines_for_json(args.json_path, source_image=args.source_image,
                                        refresh=args.refresh)
        print(summary)
        return 0
    elif args.command == "flatten":
        from .preprocess import flatten_page

        result = flatten_page(args.image_path, args.output_path, mode=args.mode)
        print(f"{'Wrote ' + result.image_path if result.changed else 'Unchanged'} "
              f"(rotation={result.rotation_deg} unwarped={result.unwarped} "
              f"deskew={result.deskew_deg:.2f} distortion={result.distortion})")
        return 0
    else:
        result_path = render_json_file(
            args.json_path, args.output_pdf, source_image=args.source_image, mode=args.mode,
        )
        if args.score:
            print(f"Wrote {result_path}")
            _print_report(score_pdf_file(args.json_path, result_path,
                                         source_image=args.source_image), 5)
            return 0

    print(f"Wrote {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
