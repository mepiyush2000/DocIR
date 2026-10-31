"""Multi-page documents: scanned PDFs in, one merged vector PDF out.

A PDF input is rasterised page by page and each page goes through the
single-page pipeline; the per-page PDFs are then joined in order.
"""

from __future__ import annotations

import os
import threading

# pdfium is not thread-safe; every pdfium call in the process goes through this lock
PDFIUM_LOCK = threading.RLock()

# the resolution the vLLM benchmarks were run at; small print stays above ~16 px tall
PDF_RENDER_DPI = 200
MAX_PAGE_SIDE_PX = 4200        # caps huge poster-size pages


def is_pdf(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(5) == b"%PDF-"


def pdf_page_images(pdf_path: str, out_dir: str, max_pages: int | None = None) -> list[str]:
    """Rasterise every page to `<out_dir>/<stem>_p001.png`, ... and return the paths.
    Refuses (ValueError) a PDF with more than `max_pages` pages before rendering any."""
    import pypdfium2 as pdfium

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    paths = []
    with PDFIUM_LOCK:
        doc = pdfium.PdfDocument(pdf_path)
        try:
            if max_pages is not None and len(doc) > max_pages:
                raise ValueError(f"the PDF has {len(doc)} pages; the limit is {max_pages}")
            for i, page in enumerate(doc, 1):
                w_pt, h_pt = page.get_size()
                scale = min(PDF_RENDER_DPI / 72, MAX_PAGE_SIDE_PX / max(w_pt, h_pt))
                path = os.path.join(out_dir, f"{stem}_p{i:03d}.png")
                page.render(scale=scale).to_pil().convert("RGB").save(path)
                paths.append(path)
        finally:
            doc.close()
    return paths


def image_page_pdf(image_path: str, output_path: str, page_width_pt: float = 612.0) -> str:
    """A one-page PDF showing the scan itself: the stand-in for a page that could not be converted."""
    from reportlab.pdfgen import canvas

    from .preprocess import load_upright

    image, _ = load_upright(image_path)
    height_pt = page_width_pt * image.height / image.width
    c = canvas.Canvas(output_path, pagesize=(page_width_pt, height_pt))
    c.drawInlineImage(image, 0, 0, page_width_pt, height_pt)
    c.showPage()
    c.save()
    return output_path


def merge_pdfs(page_pdfs: list[str], output_path: str) -> str:
    import pypdfium2 as pdfium

    with PDFIUM_LOCK:
        merged = pdfium.PdfDocument.new()
        for path in page_pdfs:
            src = pdfium.PdfDocument(path)
            merged.import_pages(src)
            src.close()
        merged.save(output_path)
        merged.close()
    return output_path
