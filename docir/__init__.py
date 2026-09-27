"""docir: recover document layout via PaddleOCR-VL and rebuild a
geometry-preserving PDF from it."""

from .schemas import Block, PageLayout

__all__ = ["Block", "PageLayout"]

__version__ = "0.1.0"
