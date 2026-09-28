import sys

from PIL import Image, ImageDraw

from docir.line_refine import split_merged_lines
from docir.pipeline import load_layout_json, load_lines

name, image_path = sys.argv[1], sys.argv[2]
layout = load_layout_json(f"output/{name}_res.json")
img = Image.open(image_path).convert("RGB")
lines = split_merged_lines(load_lines(f"output/{name}_lines.json"), layout, img)
d = ImageDraw.Draw(img)
for i, l in enumerate(lines.lines):
    d.polygon([tuple(p) for p in l.polygon], outline=("red", "blue", "green")[i % 3])
for b in layout.blocks:
    d.rectangle(b.bbox, outline="orange")
img.save(f"output/_eval/{name}_lines_overlay.png")
