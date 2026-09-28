import sys

import numpy as np
from PIL import Image

from docir.ink import binarize_region, measure_text_lines, runs_1d
from docir.pipeline import load_layout_json

name, image = sys.argv[1], sys.argv[2]
im = Image.open(image).convert("RGB")
for b in load_layout_json(f"output/{name}_res.json").blocks:
    if not b.content or b.label in {"image"}:
        continue
    m = binarize_region(im, b.bbox)
    st = measure_text_lines(m) if m is not None else None
    if st is None:
        continue
    runs = [e - s for row in m for s, e in runs_1d(row)]
    stem = float(np.median(runs))
    p60 = float(np.percentile(runs, 60))
    print(b.id, b.label[:14], f"xh={st.x_height:.1f} stem={stem:.1f} p60={p60:.1f} "
          f"ratio={stem / st.x_height:.3f} r60={p60 / st.x_height:.3f}")
