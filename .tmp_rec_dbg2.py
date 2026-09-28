import sys

from PIL import Image

from docir.line_refine import split_merged_lines
from docir.pipeline import load_layout_json, load_lines
from docir.reconcile import reconcile

name, image, ids = sys.argv[1], sys.argv[2], [int(v) for v in sys.argv[3].split(",")]
layout = load_layout_json(f"output/{name}_res.json")
lines = load_lines(f"output/{name}_lines.json")
refined = split_merged_lines(lines, layout, Image.open(image).convert("RGB"))
print("lines", len(lines.lines), "->", len(refined.lines))
rec = reconcile(layout, refined)
print("orphans", [(o.text, [int(v) for v in o.polygon[0]]) for o in rec.orphans])
for bid in ids:
    b = rec.blocks[bid]
    print(f"== block {bid} match={b.text_match}")
    for l in b.lines:
        print(f"   {[int(v) for v in l.bbox]} cont={l.continuation} ocr={l.ocr_text[:40]!r}\n      -> {l.text!r}")
