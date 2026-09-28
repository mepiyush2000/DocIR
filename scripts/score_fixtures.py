"""Render + score every fixture page; prints a table and writes diffs.

    .docir_env/bin/python scripts/score_fixtures.py [name ...]
"""

from __future__ import annotations

import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from PIL import Image  # noqa: E402

from docir.fidelity import rasterize_pdf  # noqa: E402
from docir.pipeline import render_json_file, score_pdf_file  # noqa: E402

FIXTURES = {
    "vl1_6_10": "vl1_6_10.png",
    "demo_img": "demo_img.jpeg",
    "vl1_6_3": "vl1_6_3.jpeg",
    "vl1_6_3_flat": "vl1_6_3_flat.png",
    "vl1_6_4": "vl1_6_4.png",
}
EVAL_DIR = os.path.join(REPO, "output", "_eval")


def main(names: list[str]) -> int:
    os.makedirs(EVAL_DIR, exist_ok=True)
    print(f"{'fixture':<10} {'P':>6} {'R':>6} {'F1':>6} {'stray':>8} {'sec':>6}")
    for name in names or FIXTURES:
        image = os.path.join(REPO, FIXTURES[name])
        json_path = os.path.join(REPO, "output", f"{name}_res.json")
        pdf = os.path.join(EVAL_DIR, f"{name}.pdf")
        t0 = time.time()
        render_json_file(json_path, pdf, source_image=image)
        elapsed = time.time() - t0
        report = score_pdf_file(json_path, pdf, source_image=image,
                                diff_path=os.path.join(EVAL_DIR, f"{name}_diff.png"),
                                report_path=os.path.join(EVAL_DIR, f"{name}_score.json"))
        print(f"{name:<10} {report.precision:6.3f} {report.recall:6.3f} {report.f1:6.3f} "
              f"{report.stray_ink:8d} {elapsed:6.1f}", flush=True)
        w, h = Image.open(image).size
        rasterize_pdf(pdf, w, h).save(os.path.join(EVAL_DIR, f"{name}_render.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
