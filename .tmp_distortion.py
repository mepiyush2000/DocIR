from PIL import Image

from docir.preprocess import measure_distortion

for p in ["vl1_6_10.png", "demo_img.jpeg", "vl1_6_3.jpeg", "vl1_6_4.png", "output/_eval/vl1_6_3_flat_probe.png"]:
    print(p, measure_distortion(Image.open(p)), flush=True)
