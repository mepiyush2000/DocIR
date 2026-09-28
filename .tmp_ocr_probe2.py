import sys
import time

from paddleocr import PaddleOCR

det, side, ltype = sys.argv[2], int(sys.argv[3]), sys.argv[4]
ocr = PaddleOCR(
    use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
    text_detection_model_name=det, text_recognition_model_name="PP-OCRv5_server_rec",
    text_det_limit_side_len=side, text_det_limit_type=ltype,
)
for path in sys.argv[1].split(","):
    t = time.time()
    j = ocr.predict(path)[0].json["res"]
    good = sum(s > 0.8 for s in j["rec_scores"])
    print(f"{det} {side} {ltype} {path}: lines {len(j['rec_texts'])} conf>0.8 {good} sec {time.time() - t:.1f}", flush=True)
