from PIL import Image

from docir.dewarp import dewarp
from docir.preprocess import measure_distortion

for p in ["vl1_6_3.jpeg", "demo_img.jpeg", "vl1_6_4.png"]:
    im = Image.open(p)
    out = dewarp(im)
    print(p, "before", measure_distortion(im), flush=True)
    if out is not None:
        print("   after", measure_distortion(out), flush=True)
        out.save(f"output/_eval/{p.split('.')[0]}_dewarp.png")
