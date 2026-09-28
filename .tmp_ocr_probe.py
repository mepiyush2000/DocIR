import json
import sys
import time

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
    text_detection_model_name="PP-OCRv5_server_det", text_recognition_model_name="PP-OCRv5_server_rec",
    text_det_limit_side_len=int(sys.argv[2]) if len(sys.argv) > 2 else 64,
    text_det_limit_type=sys.argv[3] if len(sys.argv) > 3 else "min",
)
for path in sys.argv[1].split(","):
    t = time.time()
    res = ocr.predict(path)[0]
    j = res.json["res"]
    print(path, "lines", len(j["rec_texts"]), "sec %.1f" % (time.time() - t), "keys", sorted(j.keys()), flush=True)
    for poly, text, sc in list(zip(j["rec_polys"], j["rec_texts"], j["rec_scores"]))[:6]:
        print("   ", [list(map(int, p)) for p in poly], repr(text), round(sc, 3))
    with open(f"output/_eval/{path.rsplit('.', 1)[0]}_ocrtest.json", "w") as f:
        json.dump({k: j[k] for k in ("rec_polys", "rec_texts", "rec_scores")}, f)
