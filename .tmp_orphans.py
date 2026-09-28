from docir.pipeline import lines_json_path, load_layout_json, load_lines
from docir.reconcile import reconcile, vl_wording

name = "demo_img"
layout = load_layout_json(f"output/{name}_res.json")
rec = reconcile(layout, load_lines(lines_json_path(f"output/{name}_res.json")))
contents = [b.content for b in layout.blocks if b.content]
for o in rec.orphans:
    xs, ys = [p[0] for p in o.polygon], [p[1] for p in o.polygon]
    print([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))], round(o.score, 2), repr(o.text),
          "->", repr(vl_wording(o.text, contents)))
