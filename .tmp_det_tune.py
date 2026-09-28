import sys
import time

import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

path, scale, unclip = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
                text_detection_model_name="PP-OCRv5_server_det", text_recognition_model_name="PP-OCRv5_server_rec",
                text_det_limit_side_len=32, text_det_limit_type="min", text_det_unclip_ratio=unclip)
img = Image.open(path).convert("RGB")
img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
t = time.time()
j = ocr.predict(np.ascontiguousarray(np.asarray(img)[..., ::-1]))[0].json["res"]
hs = sorted(round((max(p[1] for p in q) - min(p[1] for p in q)) / scale) for q in j["rec_polys"])
print(f"scale={scale} unclip={unclip}: {len(hs)} lines, heights {hs}, empty={sum(not t for t in j['rec_texts'])} {time.time() - t:.0f}s", flush=True)
