"""Command-line entry point.

    docir analyze <image>            # raw image -> OCR/layout -> PDF
    docir render  <layout.json>       # legacy layout JSON -> PDF
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import is_debug_logging_enabled
from .pipeline import process_image, render_json_file


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
    return parser


def main(argv: list[str] | None = None) -> int:
    level = logging.DEBUG if is_debug_logging_enabled() else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)

    if args.command == "analyze":
        result_path = process_image(
            args.image_path, output_pdf=args.output_pdf, output_json=args.output_json,
        )
    else:
        result_path = render_json_file(
            args.json_path, args.output_pdf, source_image=args.source_image,
        )

    print(f"Wrote {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
