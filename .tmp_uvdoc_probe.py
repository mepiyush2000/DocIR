import sys
import time

import numpy as np
from paddleocr import DocImgOrientationClassification, TextImageUnwarping
from PIL import Image

path = sys.argv[1]
t = time.time()
ori = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori").predict(path)[0].json["res"]
print("orientation", ori.get("label_names"), ori.get("scores"), f"{time.time() - t:.1f}s", flush=True)
t = time.time()
res = TextImageUnwarping(model_name="UVDoc").predict(path)[0]
print("unwarp keys", sorted(res.json["res"].keys()), f"{time.time() - t:.1f}s", flush=True)
out = res["doctr_img"]
arr = np.asarray(out)
print("in", Image.open(path).size, "out", arr.shape, arr.dtype, arr.min(), arr.max())
Image.fromarray(arr[..., ::-1].astype(np.uint8)).save(sys.argv[2])
