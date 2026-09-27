"""Deprecated entry point -- kept for backward compatibility.

Use instead:
    python -m docir.cli render <json_path> [output_pdf] [--source-image ...]

The actual implementation now lives in the `docir` package
(docir.schemas / docir.bold / docir.render_pdf / docir.pipeline).
"""
import sys

from docir.cli import main as _cli_main

if __name__ == "__main__":
    sys.exit(_cli_main(["render", *sys.argv[1:]]))
