"""Deprecated entry point -- kept for backward compatibility.

Use instead:
    python -m docir.cli analyze <image_path> [output_pdf] [--json out.json]
"""
import sys

from docir.cli import main as _cli_main

if __name__ == "__main__":
    sys.exit(_cli_main(["analyze", *sys.argv[1:]]))

