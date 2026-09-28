import sys

from docir.pipeline import lines_json_path, load_layout_json, load_lines
from docir.reconcile import reconcile

name, ids = sys.argv[1], [int(v) for v in sys.argv[2].split(",")]
layout = load_layout_json(f"output/{name}_res.json")
rec = reconcile(layout, load_lines(lines_json_path(f"output/{name}_res.json")))
for b in layout.blocks:
    if b.id in ids:
        print(f"== #{b.id} {b.label} bbox={[int(v) for v in b.bbox]} content={b.content[:160]!r}")
        for l in (rec.blocks[b.id].lines if b.id in rec.blocks else []):
            print(f"   {[int(v) for v in l.bbox]} C={int(l.continuation)} ocr={l.ocr_text!r}")
