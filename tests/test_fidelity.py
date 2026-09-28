import numpy as np
import pytest
from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas

from docir.fidelity import diff_image, rasterize_pdf, score_images
from docir.schemas import Block, PageLayout

W, H = 400, 300


def _page(text_rows: list[tuple[int, int]], shift: int = 0) -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    for x, y in text_rows:
        # word-like bars: tall enough to measure an x-height
        for i in range(8):
            d.rectangle((x + shift + i * 30, y, x + shift + i * 30 + 20, y + 12), fill="black")
    return img


def _layout(blocks: list[Block]) -> PageLayout:
    return PageLayout(width=W, height=H, blocks=blocks)


TEXT = Block(id=0, label="text", content="hello world", bbox=(40, 40, 300, 120))


def test_identical_render_scores_one():
    src = _page([(50, 50), (50, 90)])
    report = score_images(_layout([TEXT]), src, src.copy())
    assert report.f1 == pytest.approx(1.0)
    assert report.blocks[0].f1 == pytest.approx(1.0)


def test_shift_within_tolerance_is_forgiven_but_misplaced_lines_are_not():
    src = _page([(50, 50), (50, 90)])
    near = score_images(_layout([TEXT]), src, _page([(50, 52), (50, 92)]))
    far = score_images(_layout([TEXT]), src, _page([(50, 70), (50, 110)]))
    assert near.f1 > 0.95
    assert far.f1 < near.f1 - 0.2


def test_blank_render_has_zero_recall():
    src = _page([(50, 50)])
    report = score_images(_layout([TEXT]), src, Image.new("RGB", (W, H), "white"))
    assert report.recall == 0.0
    assert report.f1 == 0.0


def test_picture_regions_are_ignored_and_stray_ink_counts():
    src = _page([(50, 50)])
    ren = src.copy()
    d = ImageDraw.Draw(ren)
    d.rectangle((320, 200, 390, 290), fill="black")      # inside picture: ignored
    d.rectangle((10, 200, 60, 240), fill="black")        # outside everything: stray
    picture = Block(id=1, label="image", content="", bbox=(310, 190, 400, 300))
    report = score_images(_layout([TEXT, picture]), src, ren)
    assert report.blocks[0].f1 == pytest.approx(1.0)
    assert report.stray_ink == 51 * 41
    assert report.precision < 0.9


def test_mismatched_sizes_rejected():
    with pytest.raises(ValueError):
        score_images(_layout([TEXT]), _page([]), Image.new("RGB", (W + 1, H), "white"))


def test_diff_image_colors_missing_red_and_extra_blue():
    src = _page([(50, 50)])
    ren = _page([(50, 90)])
    diff = np.asarray(diff_image(_layout([TEXT]), src, ren))
    assert tuple(diff[55, 55]) == (220, 30, 30)
    assert tuple(diff[95, 55]) == (30, 80, 230)


def test_rasterize_pdf_matches_source_pixels(tmp_path):
    # content placed at a uniform scale from the top-left, on a taller page
    path = str(tmp_path / "p.pdf")
    page_w, scale = 200.0, 200.0 / W
    c = canvas.Canvas(path, pagesize=(page_w, H * scale + 50))
    c.rect(50 * scale, (H * scale + 50) - 120 * scale, 100 * scale, 40 * scale, fill=1, stroke=0)
    c.save()
    img = np.asarray(rasterize_pdf(path, W, H).convert("L"))
    assert img.shape == (H, W)
    assert img[100, 100] < 50            # inside the rect (x 50..150, y 80..120)
    assert img[60, 100] > 200 and img[100, 20] > 200
