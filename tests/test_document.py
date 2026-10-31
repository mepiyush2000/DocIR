import pypdfium2 as pdfium
from PIL import Image
from reportlab.pdfgen import canvas

from docir.document import image_page_pdf, is_pdf, merge_pdfs, pdf_page_images
from docir.pipeline import document_pages
from docir.preprocess import flatten_page, load_upright


def _sideways_photo(path, size=(300, 200)):
    """A landscape-stored photo whose EXIF tag says: rotate 90 degrees to view it upright."""
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", size, "white").save(path, exif=exif.tobytes())


def _two_page_pdf(path):
    c = canvas.Canvas(str(path), pagesize=(612, 792))
    for text in ("first", "second"):
        c.drawString(72, 700, text)
        c.showPage()
    c.save()


def test_load_upright_applies_exif_rotation(tmp_path):
    photo = tmp_path / "photo.jpg"
    _sideways_photo(photo)
    image, tagged = load_upright(str(photo))
    assert tagged and image.size == (200, 300)


def test_flatten_off_still_writes_an_upright_copy_of_a_tagged_photo(tmp_path):
    photo = tmp_path / "photo.jpg"
    _sideways_photo(photo)
    result = flatten_page(str(photo), mode="off")
    assert result.changed and result.image_path != str(photo)
    assert Image.open(result.image_path).size == (200, 300)


def test_flatten_off_leaves_an_untagged_image_alone(tmp_path):
    page = tmp_path / "page.png"
    Image.new("RGB", (200, 300), "white").save(page)
    result = flatten_page(str(page), mode="off")
    assert not result.changed and result.image_path == str(page)


def test_pdf_pages_are_rasterised_in_order(tmp_path):
    pdf = tmp_path / "scan.pdf"
    _two_page_pdf(pdf)
    assert is_pdf(str(pdf))
    pages = document_pages(str(pdf), str(tmp_path / "pages"))
    assert [p.rsplit("/", 1)[1] for p in pages] == ["scan_p001.png", "scan_p002.png"]
    # 200 dpi letter page (pdfium may round up a pixel)
    w, h = Image.open(pages[0]).size
    assert abs(w - 1700) <= 1 and abs(h - 2200) <= 1


def test_an_image_input_is_its_own_single_page(tmp_path):
    page = tmp_path / "page.png"
    Image.new("RGB", (200, 300), "white").save(page)
    assert not is_pdf(str(page))
    assert document_pages(str(page), str(tmp_path / "pages")) == [str(page)]


def test_merge_keeps_every_page_including_scan_fallbacks(tmp_path):
    pdf = tmp_path / "scan.pdf"
    _two_page_pdf(pdf)
    page = tmp_path / "page.png"
    Image.new("RGB", (400, 600), "white").save(page)
    fallback = image_page_pdf(str(page), str(tmp_path / "fallback.pdf"))
    merged = merge_pdfs([str(pdf), fallback], str(tmp_path / "merged.pdf"))
    doc = pdfium.PdfDocument(merged)
    assert len(doc) == 3
    assert doc[2].get_size() == (612.0, 918.0)
