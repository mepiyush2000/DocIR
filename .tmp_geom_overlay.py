import sys

import numpy as np
from PIL import Image, ImageDraw

from docir.line_geometry import measure_line, pixel_lines
from docir.pipeline import load_layout_json, load_lines

name, image_path = sys.argv[1], sys.argv[2]
layout = load_layout_json(f"output/{name}_res.json")
img = Image.open(image_path).convert("RGB")
gray = np.asarray(img.convert("L"))
out = img.copy()
d = ImageDraw.Draw(out)
n = 0
for b in layout.blocks:
    if b.label in {"image", "figure", "chart", "seal"} or not b.content:
        continue
    quads = pixel_lines(img, b.bbox)
    n += len(quads)
    for q in quads:
        d.polygon(q, outline="blue")
for l in load_lines(f"output/{name}_lines.json").lines:
    m = measure_line(gray, l.polygon)
    if m:
        d.line([m.baseline_start, m.baseline_end], fill="red", width=1)
        (x0, y0), (x1, y1) = m.baseline_start, m.baseline_end
        d.line([(x0, y0 - m.x_height), (x1, y1 - m.x_height)], fill="green", width=1)
print("pixel lines", n)
out.save(f"output/_eval/{name}_geom_overlay.png")
