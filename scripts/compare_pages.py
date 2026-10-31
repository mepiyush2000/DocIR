"""Side-by-side source | render | diff images for pages scored by eval_corpus.py.

    .docir_env/bin/python scripts/compare_pages.py SCORES_CSV NAME [NAME ...] [--width 1400]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from PIL import Image  # noqa: E402

from docir.fidelity import diff_image, rasterize_pdf  # noqa: E402
from docir.pipeline import _locate_legacy_source_image, load_layout_json  # noqa: E402


def compare(row: dict, out_dir: str, width: int) -> str:
    layout = load_layout_json(row["json"])
    src_path = _locate_legacy_source_image(row["json"], layout.input_path)
    size = (round(layout.width), round(layout.height))
    source = Image.open(src_path).convert("RGB").resize(size)
    rendered = rasterize_pdf(row["pdf"], *size)
    diff = diff_image(layout, source, rendered)
    panels = [source, rendered, diff]
    scale = width / (3 * size[0])
    small = [p.resize((round(size[0] * scale), round(size[1] * scale))) for p in panels]
    sheet = Image.new("RGB", (sum(p.width for p in small), small[0].height), "white")
    x = 0
    for p in small:
        sheet.paste(p, (x, 0))
        x += p.width
    path = os.path.join(out_dir, f"{row['category']}_{row['name']}.png")
    sheet.save(path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--width", type=int, default=1800)
    args = ap.parse_args()
    rows = {r["name"]: r for r in csv.DictReader(open(args.scores))}
    out_dir = os.path.join(os.path.dirname(args.scores), "cmp")
    os.makedirs(out_dir, exist_ok=True)
    for n in args.names:
        print(compare(rows[n], out_dir, args.width))
    return 0


if __name__ == "__main__":
    sys.exit(main())
