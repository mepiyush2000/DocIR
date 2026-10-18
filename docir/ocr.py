"""OCR/layout-detection adapters.

`OcrEngine` is the protocol the rest of the pipeline depends on; concrete
engines (real or fake) live behind it so no PaddleOCR-specific object ever
reaches schemas, rendering, or tests.
"""

from __future__ import annotations

from typing import Protocol

from .schemas import PageLayout, PageLines


class OcrEngine(Protocol):
    def analyze(self, image_path: str) -> PageLayout: ...


class PaddleOcrVlEngine:
    """Adapter around `paddleocr.PaddleOCRVL`. The pipeline is created lazily
    on first use so importing this module -- and this package in general --
    never requires paddleocr/paddlepaddle to be installed (they're an
    optional dependency group; see the "ocr" extra in pyproject.toml)."""

    def __init__(self, pipeline_version: str = "v1.6"):
        self._pipeline_version = pipeline_version
        self._pipeline = None
        self.last_raw: dict | None = None  # raw dict from the most recent analyze() call

    def _pipeline_instance(self):
        if self._pipeline is None:
            from paddleocr import PaddleOCRVL  # optional dependency

            from .config import paddle_device_kwargs, vlm_backend_kwargs

            self._pipeline = PaddleOCRVL(
                pipeline_version=self._pipeline_version,
                **paddle_device_kwargs(), **vlm_backend_kwargs(),
            )
        return self._pipeline

    def analyze(self, image_path: str) -> PageLayout:
        pipeline = self._pipeline_instance()
        results = list(pipeline.predict(image_path))
        if not results:
            raise RuntimeError(f"PaddleOCR-VL returned no result for '{image_path}'")
        # res.json == {"res": <same payload as the saved *_res.json file>}
        self.last_raw = results[0].json["res"]
        return PageLayout.from_paddle_dict(self.last_raw)


class FakeOcrEngine:
    """Deterministic stand-in for tests: returns a pre-built PageLayout
    instead of running real OCR (no model download, no inference, no
    network access)."""

    def __init__(self, layout: PageLayout):
        self._layout = layout

    def analyze(self, image_path: str) -> PageLayout:
        return self._layout


class LineOcrEngine(Protocol):
    def detect_lines(self, image_path: str, text_height_px: float | None = None) -> PageLines: ...


LINE_DET_TARGET_X_HEIGHT_PX = 12.0   # detector works best around this text size
LINE_DET_SCALE_RANGE = (0.35, 2.5)
LINE_DET_MAX_SIDE_PX = 2400


def line_detection_scale(width: int, height: int, text_height_px: float | None) -> float:
    """Resize factor that brings the page's x-height near the detector's
    sweet spot: small photographed text is upscaled (otherwise most lines
    are missed), large scans downscaled (detection cost grows with area)."""
    lo, hi = LINE_DET_SCALE_RANGE
    scale = 1.0 if not text_height_px else LINE_DET_TARGET_X_HEIGHT_PX / text_height_px
    scale = min(max(scale, lo), hi)
    return min(scale, LINE_DET_MAX_SIDE_PX / max(width, height))


class PaddleLineOcrEngine:
    """PP-OCRv5 text-line detection + recognition (lazy, optional dependency)."""

    def __init__(self, det_model: str = "PP-OCRv5_server_det",
                 rec_model: str = "PP-OCRv5_server_rec"):
        self._det_model, self._rec_model = det_model, rec_model
        self._pipeline = None

    @property
    def name(self) -> str:
        return f"{self._det_model}+{self._rec_model}"

    def _pipeline_instance(self):
        if self._pipeline is None:
            from paddleocr import PaddleOCR  # optional dependency

            from .config import paddle_device_kwargs

            self._pipeline = PaddleOCR(
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False,
                text_detection_model_name=self._det_model,
                text_recognition_model_name=self._rec_model,
                # resizing is done here, by text size, not by PaddleX's side limits
                text_det_limit_side_len=32, text_det_limit_type="min",
                **paddle_device_kwargs(),
            )
        return self._pipeline

    def detect_lines(self, image_path: str, text_height_px: float | None = None) -> PageLines:
        import numpy as np
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        scale = line_detection_scale(image.width, image.height, text_height_px)
        if abs(scale - 1.0) > 0.05:
            image = image.resize((round(image.width * scale), round(image.height * scale)),
                                 Image.LANCZOS)
        else:
            scale = 1.0
        bgr = np.ascontiguousarray(np.asarray(image)[..., ::-1])
        results = list(self._pipeline_instance().predict(bgr))
        if not results:
            raise RuntimeError(f"line OCR returned no result for '{image_path}'")
        raw = dict(results[0].json["res"], input_path=image_path)
        w, h = Image.open(image_path).size
        return PageLines.from_paddle_dict(raw, w, h, engine=self.name, scale=scale)


class FakeLineOcrEngine:
    def __init__(self, lines: PageLines):
        self._lines = lines

    def detect_lines(self, image_path: str, text_height_px: float | None = None) -> PageLines:
        return self._lines
