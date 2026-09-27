"""Standalone diagnostic: draw only detected bboxes (axis-aligned) over the
source image.

Usage:
    python -m docir.tools.overlay_bbox <json_path> <source_image> <out_png>
"""

from __future__ import annotations

import sys

from ._overlay_common import draw_bbox_overlay


def main() -> None:
    if len(sys.argv) != 4:
        print("usage: python -m docir.tools.overlay_bbox <json_path> <source_image> <out_png>")
        raise SystemExit(1)
    json_path, image_path, out_path = sys.argv[1:4]
    draw_bbox_overlay(json_path, image_path, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
