import sys

from PIL import Image

from docir.page_builder import build_page
from docir.pipeline import lines_json_path, load_layout_json, load_lines

name, image = sys.argv[1], sys.argv[2]
ids = {int(v) for v in sys.argv[3].split(",")} if len(sys.argv) > 3 else None
layout = load_layout_json(f"output/{name}_res.json")
page = build_page(layout, Image.open(image).convert("RGB"), load_lines(lines_json_path(f"output/{name}_res.json")))
print("background", page.background, "graphics", len(page.graphics))
for b in page.blocks:
    if ids and b.id not in ids:
        continue
    s = b.style
    print(f"#{b.id} {b.role} lines={len(b.lines)} src={sorted({l.source for l in b.lines})} fill={b.fill} "
          f"color={b.lines[0].color if b.lines else None} "
          f"style={s.font_family if s else None} {s.size_px if s else 0:.1f}px w{s.width_scale if s else 0:.2f} "
          f"b{int(bool(s and s.bold))}i{int(bool(s and s.italic))}")
    for l in b.lines[:6]:
        print(f"     xh={l.x_height:.1f} len={l.length:.0f} ang={l.angle_deg:.1f} {'D' if l.drop_cap else ''}{'C' if l.continuation else ''} {l.text[:70]!r}")
