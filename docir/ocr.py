"""OCR/layout-detection adapters.

`OcrEngine` is the protocol the rest of the pipeline depends on; concrete
engines (real or fake) live behind it so no PaddleOCR-specific object ever
reaches schemas, rendering, or tests.
"""

from __future__ import annotations

from typing import Protocol

from .schemas import PageLayout


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

            self._pipeline = PaddleOCRVL(pipeline_version=self._pipeline_version)
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
