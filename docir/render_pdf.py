"""Render a PageLayout into a PDF that visually mirrors the source page's
geometry -- regardless of orientation, column count, or label taxonomy.

Design notes (what makes this "general" vs. a hardcoded per-document script):
  1. Columns are discovered automatically via 1D gap-clustering on each
     block's x1, instead of a hand-picked bucket size for one document.
  2. Any block much wider than a single column (a masthead/section header)
     is detected by width-fraction and rendered full-width, pushing every
     column's cursor down below it.
  3. Unknown labels fall back to a generic body-text style instead of
     crashing or being skipped.
  4. Text is XML-escaped and literal newlines become <br/>, so content with
     "&", "<", ">" or embedded line breaks won't break ReportLab's mini-XML
     parser.
  5. Per-column cursors + "does this text actually fit" checks (via
     Paragraph.wrap) grow a block's box downward when the real font needs
     more room than the original tight pixel box implied, and push
     subsequent same-column blocks down to match, so text is never silently
     dropped or overlapped.
  6. Image/figure/chart/seal blocks are cropped directly out of the source
     page image by bbox (no dependency on separately saved per-block crop
     files) and scaled to fit their box, preserving aspect ratio.
  7. Bold vs. regular weight, italic slant, and serif/sans-serif/monospace
     family are all measured from real pixels (see `bold.py`, `italic.py`,
     `fonts.py`), never assumed from a block's label.
"""

from __future__ import annotations

import html
import logging
import statistics
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Paragraph

from .bold import classify_bold_by_label, measure_stroke_ratio
from .fonts import classify_word_samples, collect_word_samples, discover_catalog
from .ink import TextLineStats, binarize_region, measure_text_lines
from .italic import classify_italic_by_label, measure_slant_angle
from .page_ratios import closest_standard_ratio
from .schemas import PageLayout

logger = logging.getLogger(__name__)

# ---------- tunables (safe defaults, override per-call if needed) ----------
TARGET_PAGE_WIDTH_PT = 612.0     # scale output to a standard 8.5in-wide page
COLUMN_GAP_THRESHOLD_FRAC = 0.05  # x1 gap (as a fraction of page width) that starts a new column
FULL_WIDTH_FRAC = 0.6            # a block this wide (as a fraction of page width) is a spanning header
FONT_FIT_ITERATIONS = 18         # binary-search steps; 18 halvings easily beats 0.1pt precision
SIZE_CLUSTER_GAP_RATIO = 1.5      # per-label fitted sizes more than this multiplicative
                                  # gap apart are treated as separate visual roles
MIN_LABEL_FONT_WORDS = 8          # a label with fewer matched words than this defers to the
                                  # document-wide category instead of its own noisy signal
STANDARD_RATIO_TOLERANCE = 0.05  # snap the page canvas to a known paper/photo ratio
                                  # within this relative difference; further off keeps
                                  # the source image's own raw ratio
FONT_POOL_WORDS_PER_LABEL = 24    # matched words pooled per label for its font-family call
LABEL_FONT_OVERRIDE_MARGIN = 0.05  # margin a label needs to differ from the document's family
SIZE_SEARCH_MIN_PT = 2.0
SIZE_SEARCH_MAX_PT = 200.0
# a substitute font's x-height can differ from the original's, so a size
# matched on x-height may exceed the measured line pitch; lines may then sit
# a little tighter than the size, but never so tight that they collide
MIN_LEADING_RATIO = 0.85
CLUSTER_SIZE_SLACK = 1.2  # a role's shared size may move this far from its median to fit more blocks
MIN_HSCALE, MAX_HSCALE = 0.6, 1.25  # horizontal glyph scaling allowed to mimic narrow/wide fonts

# x-height as a fraction of the em, straight from the standard-14 AFM metrics
X_HEIGHT_EM = {
    "Helvetica": 0.523, "Helvetica-Oblique": 0.523,
    "Helvetica-Bold": 0.532, "Helvetica-BoldOblique": 0.532,
    "Times-Roman": 0.448, "Times-Italic": 0.441,
    "Times-Bold": 0.461, "Times-BoldItalic": 0.462,
    "Courier": 0.426, "Courier-Oblique": 0.426,
    "Courier-Bold": 0.439, "Courier-BoldOblique": 0.439,
}

# fontSize/leading below are only fallbacks (used if auto-fit can't find a
# size within bounds); leading-to-size ratio is preserved per label when
# fitting. There's no fontName here -- font family (serif/sans-serif/
# monospace), bold, and italic are all decided per block from actual pixels,
# never assumed from the label.
DEFAULT_STYLES = {
    "doc_title":        dict(fontSize=13, leading=15.5, color="#1F2A36", minSize=8, maxSize=30),
    "title":            dict(fontSize=13, leading=15.5, color="#1F2A36", minSize=8, maxSize=30),
    "paragraph_title":  dict(fontSize=10.5, leading=12.5, color="#1F2A36", minSize=7, maxSize=26),
    "figure_title":     dict(fontSize=8, leading=9.5, color="#1F2A36", minSize=5.5, maxSize=18),
    "vision_footnote":  dict(fontSize=7.6, leading=10, color="#2E3A46", minSize=5, maxSize=16),
    "source":           dict(fontSize=6.3, leading=7.8, color="#8A94A0", minSize=4.5, maxSize=12),
    "header":           dict(fontSize=7, leading=9, color="#8A94A0", align="center", minSize=4.5, maxSize=12),
    "footer":           dict(fontSize=6.5, leading=8, color="#8A94A0", minSize=4.5, maxSize=12),
    "number":           dict(fontSize=7, leading=8.5, color="#8A94A0", align="center", minSize=4.5, maxSize=14),
    "text":             dict(fontSize=7.4, leading=9.8, color="#2E3A46", minSize=5, maxSize=18),
}
FALLBACK_STYLE = DEFAULT_STYLES["text"]

SKIP_LABELS: set[str] = set()  # labels to omit entirely even if content exists
IMAGE_LABELS = {"image", "figure", "chart", "seal"}

# ReportLab's standard-14 fonts, keyed by detected style category -- no
# embedding/licensing needed since these ship with every PDF viewer.
FONT_FAMILIES = {
    "sans_serif": dict(regular="Helvetica", bold="Helvetica-Bold",
                       italic="Helvetica-Oblique", bold_italic="Helvetica-BoldOblique"),
    "serif":      dict(regular="Times-Roman", bold="Times-Bold",
                       italic="Times-Italic", bold_italic="Times-BoldItalic"),
    "monospace":  dict(regular="Courier", bold="Courier-Bold",
                       italic="Courier-Oblique", bold_italic="Courier-BoldOblique"),
}
DEFAULT_FONT_CATEGORY = "sans_serif"  # used when a label's family can't be measured


def _sanitize(text):
    text = html.escape(text, quote=False)
    text = text.replace("\n", "<br/>")
    return text


def _resolve_font_name(font_category, bold, italic):
    """Pick the standard-14 font name for this category/weight/slant
    combination, falling back to the default category if unrecognized."""
    variants = FONT_FAMILIES.get(font_category, FONT_FAMILIES[DEFAULT_FONT_CATEGORY])
    if bold and italic:
        return variants["bold_italic"]
    if bold:
        return variants["bold"]
    if italic:
        return variants["italic"]
    return variants["regular"]


def _build_style(label, font_size=None, bold=False, italic=False, font_category=None, leading=None):
    """Build a ParagraphStyle for a label, optionally overriding the font size
    (leading scales with it, preserving the label's original leading/size
    ratio, unless an explicit measured `leading` is given)."""
    spec = DEFAULT_STYLES.get(label, FALLBACK_STYLE)
    align = TA_CENTER if spec.get("align") == "center" else TA_LEFT
    size = font_size if font_size is not None else spec["fontSize"]
    leading_ratio = spec["leading"] / spec["fontSize"]
    font_name = _resolve_font_name(font_category or DEFAULT_FONT_CATEGORY, bold, italic)
    return ParagraphStyle(
        label, fontName=font_name, fontSize=size,
        leading=leading if leading is not None else size * leading_ratio,
        textColor=colors.HexColor(spec["color"]),
        alignment=align,
    )


def _fit_font_size(label, content, width_pt, height_pt, bold=False, italic=False, font_category=None):
    """Binary-search the largest font size whose wrapped paragraph height
    fits within height_pt at width_pt, so each block's rendered text size
    approximates its original box instead of one fixed size per label."""
    spec = DEFAULT_STYLES.get(label, FALLBACK_STYLE)
    lo, hi = spec.get("minSize", 5.0), spec.get("maxSize", 18.0)
    best = lo
    for _ in range(FONT_FIT_ITERATIONS):
        mid = (lo + hi) / 2
        style = _build_style(label, font_size=mid, bold=bold, italic=italic, font_category=font_category)
        para = Paragraph(_sanitize(content), style)
        _, h = para.wrap(max(width_pt, 1), 20000)
        if h <= height_pt:
            best = mid
            lo = mid
        else:
            hi = mid
    return round(best, 1)


def _wrapped_line_count(label, content, width_pt, size, bold, italic, font_category):
    style = _build_style(label, font_size=size, bold=bold, italic=italic, font_category=font_category)
    para = Paragraph(_sanitize(content), style)
    para.wrap(max(width_pt, 1), 1e6)
    return len(para.blPara.lines)


def _line_count_size_range(label, content, width_pt, n_lines, bold, italic, font_category):
    """(smallest, largest) font size at which `content` wraps into exactly
    `n_lines` lines at `width_pt` -- line count only grows with size, so
    both ends are found by bisection. None for an end that doesn't exist."""
    def count(s):
        return _wrapped_line_count(label, content, width_pt, s, bold, italic, font_category)

    # explicit line breaks in the OCR text can make fewer lines impossible
    n_lines = max(n_lines, count(SIZE_SEARCH_MIN_PT))

    def bisect(pred):  # largest size in range where pred(count) holds
        lo, hi = SIZE_SEARCH_MIN_PT, SIZE_SEARCH_MAX_PT
        if not pred(count(lo)):
            return None
        if pred(count(hi)):
            return hi
        for _ in range(FONT_FIT_ITERATIONS):
            mid = (lo + hi) / 2
            if pred(count(mid)):
                lo = mid
            else:
                hi = mid
        return lo

    largest = bisect(lambda c: c <= n_lines)
    below = bisect(lambda c: c < n_lines)  # the size where it's still too few lines
    smallest = SIZE_SEARCH_MIN_PT if below is None else below
    if largest is None or smallest > largest:
        return None
    return smallest, largest


@lru_cache(maxsize=1)
def _x_band_bias() -> float:
    """How `measure_text_lines`' x-height (a dense-row band) relates to a
    font's true x-height, calibrated once by measuring a rendered sample in
    any local font -- so measured sizes aren't systematically off."""
    catalog = discover_catalog()
    face = next((f for f in catalog if f.style_category == "sans_serif"), None)
    if face is None:
        return 1.0
    px_size = 40
    font = ImageFont.truetype(face.path, px_size)
    sample = ["the quick brown fox jumps over", "a lazy dog while seven zebras", "quietly wander home again"]
    img = Image.new("L", (px_size * 20, px_size * 5), 255)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(sample):
        draw.text((10, 10 + i * px_size * 1.3), line, font=font, fill=0)
    stats = measure_text_lines(np.asarray(img) < 128)
    x_img = Image.new("L", (px_size * 2, px_size * 2), 255)
    ImageDraw.Draw(x_img).text((10, 10), "x", font=font, fill=0)
    rows = np.where((np.asarray(x_img) < 128).any(axis=1))[0]
    if stats is None or rows.size == 0:
        return 1.0
    return stats.x_height / (rows[-1] - rows[0] + 1)


def _measured_font_size(label, content, width_pt, stats: TextLineStats, scale, bold, italic, font_category):
    """(size, size_range) from real pixels: the size whose x-height matches
    the block's measured x-height, and the range of sizes that reproduce
    the block's measured line count at its width (None if impossible)."""
    font_name = _resolve_font_name(font_category, bold, italic)
    x_height_pt = stats.x_height * scale / _x_band_bias()
    size = x_height_pt / X_HEIGHT_EM.get(font_name, 0.5)
    size_range = _line_count_size_range(label, content, width_pt, stats.n_lines, bold, italic, font_category)
    return size, size_range


def _draw_fitted_placeholder(c, x, y, w, h):
    """Draw a caption inside an image placeholder box, shrinking the font
    (or skipping the text entirely for very small boxes) so it never
    spills outside the box into neighboring columns."""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    full_label = "[ image unavailable -- source page not found ]"
    short_label = "[ image ]"
    pad = 6
    avail_w = max(w - pad, 0)
    c.setFillColor(colors.HexColor("#8A94A0"))
    for label in (full_label, short_label):
        size = 7.5
        while size >= 5.0:
            if stringWidth(label, "Helvetica-Oblique", size) <= avail_w:
                c.setFont("Helvetica-Oblique", size)
                c.drawCentredString(x + w / 2, y + h / 2, label)
                return
            size -= 0.5
    # box too small even for the short label at minimum size -> leave it blank


def _crop_block_image(source_image, bbox):
    """Crop this block's region directly out of the source page image."""
    if source_image is None:
        return None
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, source_image.width), min(y2, source_image.height)
    if x2 <= x1 or y2 <= y1:
        return None
    return source_image.crop((x1, y1, x2, y2))


def _draw_pil_image(c, pil_image, x, y, w, h):
    """Draw a PIL image scaled to fit inside the block box, preserving
    aspect ratio and centering it (letterboxed) within the box."""
    iw, ih = pil_image.size
    if iw <= 0 or ih <= 0:
        return
    fit_scale = min(w / iw, h / ih)
    draw_w, draw_h = iw * fit_scale, ih * fit_scale
    draw_x = x + (w - draw_w) / 2
    draw_y = y + (h - draw_h) / 2
    c.drawImage(ImageReader(pil_image), draw_x, draw_y, width=draw_w, height=draw_h, mask="auto")


def _polygon_extent(polygon: list[tuple[float, float]]) -> tuple[float, float] | None:
    """For a simple 4-point quad ordered [top-left, top-right, bottom-right,
    bottom-left] (PaddleOCR-VL's convention), return its true (width,
    height) measured along its own edges -- not the axis-aligned bbox
    around it, which is inflated whenever the quad is rotated/slanted."""
    if len(polygon) != 4:
        return None
    p0, p1, p2, p3 = polygon

    def dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    width = (dist(p0, p1) + dist(p2, p3)) / 2
    height = (dist(p1, p2) + dist(p3, p0)) / 2
    return width, height


def _deskew_bbox_height(bbox: list[float], polygon: list[tuple[float, float]] | None) -> list[float]:
    """Shrink `bbox`'s height to the polygon's true (rotated) height when
    the block is a rotated single quad -- an axis-aligned box around a
    rotated rectangle is always taller than the rectangle itself, and that
    inflation is what makes e.g. a slightly slanted title's bbox reach down
    into whatever is below it. Left untouched for non-quad (multi-line/
    legacy) blocks, or when the polygon doesn't reveal any inflation."""
    if not polygon:
        return bbox
    extent = _polygon_extent(polygon)
    if extent is None:
        return bbox
    _, true_height = extent
    x1, y1, x2, y2 = bbox
    if true_height <= 0 or true_height >= (y2 - y1):
        return bbox
    return [x1, y1, x2, y1 + true_height]


def _cluster_columns(blocks, page_width, gap_frac):
    """Assign each block a column_key via 1D gap clustering on x1."""
    threshold = gap_frac * page_width
    ordered = sorted(blocks, key=lambda b: b["bbox"][0])
    col_key = 0
    last_x1 = None
    for b in ordered:
        x1 = b["bbox"][0]
        if last_x1 is not None and (x1 - last_x1) > threshold:
            col_key += 1
        b["_col"] = col_key
        last_x1 = x1
    return blocks


def _consensus_cluster_size(cluster, size_ranges) -> float:
    """One size for a visual-role cluster: near its median measured size,
    but chosen so as many of its blocks as possible keep their measured
    line count at that exact size -- so same-role blocks don't each get
    squeezed to a slightly different size."""
    median = statistics.median(size for _, size in cluster)
    ranges = [size_ranges[block_id] for block_id, _ in cluster if size_ranges.get(block_id)]
    lo, hi = median / CLUSTER_SIZE_SLACK, median * CLUSTER_SIZE_SLACK
    candidates = [median] + [v for r in ranges for v in r if lo <= v <= hi]

    def coverage(s):
        return sum(1 for r in ranges if r[0] <= s <= r[1])

    return max(candidates, key=lambda s: (coverage(s), -abs(s - median)))


def _best_shared_scale(intervals, prefer: float) -> float:
    """The value inside the most (lo, hi) intervals, ties broken by closeness
    to `prefer`; `prefer` itself when there are no intervals."""
    if not intervals:
        return prefer
    candidates = [prefer] + [v for iv in intervals for v in iv]

    def coverage(s):
        return sum(1 for lo, hi in intervals if lo <= s <= hi)

    return max(candidates, key=lambda s: (coverage(s), -abs(s - prefer)))


def _cluster_label_sizes(id_size_pairs):
    """Split one label's per-block fitted sizes into groups via relative-gap
    clustering (same idea as `_cluster_columns`) -- a label can be reused for
    visually distinct elements (e.g. a pull-quote sharing "paragraph_title"
    with normal short titles), and forcing all of them to one document-wide
    size would fit neither well."""
    ordered = sorted(id_size_pairs, key=lambda pair: pair[1])
    clusters = [[ordered[0]]]
    for block_id, size in ordered[1:]:
        if size > clusters[-1][-1][1] * SIZE_CLUSTER_GAP_RATIO:
            clusters.append([])
        clusters[-1].append((block_id, size))
    return clusters


def render_page_to_pdf(
    layout: PageLayout,
    output_path: str,
    source_image: Image.Image | None = None,
    extra_styles: dict | None = None,
) -> str:
    """Render `layout` into a geometry-preserving PDF at `output_path`.

    source_image: the original page image (PIL), used to detect bold vs.
    regular text weight and to crop real images for image/figure/chart/seal
    blocks. If None, text renders at regular weight and image blocks render
    as placeholders.
    extra_styles: optional dict merged over DEFAULT_STYLES for custom labels.
    """
    styles_spec = dict(DEFAULT_STYLES)
    if extra_styles:
        styles_spec.update(extra_styles)

    img_w, img_h = layout.width, layout.height

    blocks = [
        {
            "id": b.id, "label": b.label, "content": b.content,
            "bbox": _deskew_bbox_height(list(b.bbox), b.polygon),
            "src_bbox": list(b.bbox),
        }
        for b in layout.blocks
    ]

    # Snap the canvas to the closest standard paper/photo ratio (books,
    # invoices, letters, prints, screens, ...) so the page isn't shaped by
    # whatever arbitrary crop the source scan/photo happened to have --
    # unless it's distinctively its own shape, in which case its own ratio
    # is kept. Content keeps a single uniform scale either way (never
    # stretched), and the canvas is never shrunk below what the content
    # actually needs, so nothing gets clipped.
    scale = TARGET_PAGE_WIDTH_PT / img_w
    page_w = TARGET_PAGE_WIDTH_PT
    content_h = img_h * scale
    matched_ratio = closest_standard_ratio(img_w, img_h, STANDARD_RATIO_TOLERANCE)
    standard_h = page_w / matched_ratio.ratio if matched_ratio else content_h
    page_h = max(content_h, standard_h)
    if matched_ratio:
        logger.debug(
            "page ratio %.4f snapped to standard %r (%.4f)",
            img_w / img_h, matched_ratio.name, matched_ratio.ratio,
        )
    else:
        logger.debug(
            "page ratio %.4f is distinctive (no standard match within tolerance %.2f); keeping raw ratio",
            img_w / img_h, STANDARD_RATIO_TOLERANCE,
        )

    def px(v):
        return v * scale

    blocks = _cluster_columns(blocks, img_w, COLUMN_GAP_THRESHOLD_FRAC)
    # process in original document order (usually already reading order)
    blocks.sort(key=lambda b: b["id"])

    # Detect each block's ink-stroke ratio, slant angle, and font family from
    # real pixels, then classify each per label using this document's own
    # data (bold/italic vote on this label's own measurements; font family
    # is this label's most common non-"unknown" match) -- so a role like
    # "paragraph_title" is either bold/italic/serif or not throughout the doc.
    label_stroke_ratios: dict[str, list[float]] = {}
    label_slant_angles: dict[str, list[float]] = {}
    label_word_samples: dict[str, list] = {}
    line_stats: dict[int, TextLineStats | None] = {}
    for block in blocks:
        label, content = block["label"], block["content"]
        if label in IMAGE_LABELS or label in SKIP_LABELS or not content:
            continue
        fit_label = label if label in styles_spec else "text"
        bbox = block["bbox"]
        ratio = measure_stroke_ratio(source_image, bbox)
        if ratio is not None:
            label_stroke_ratios.setdefault(fit_label, []).append(ratio)
        angle = measure_slant_angle(source_image, bbox)
        if angle is not None:
            label_slant_angles.setdefault(fit_label, []).append(angle)
        # pixel measurements use the detector's own box: the deskewed one
        # is shorter than a rotated block's actual ink
        pool = label_word_samples.setdefault(fit_label, [])
        if len(pool) < FONT_POOL_WORDS_PER_LABEL:
            pool += collect_word_samples(source_image, block["src_bbox"], content, max_words=4)
        binary = binarize_region(source_image, block["src_bbox"])
        line_stats[block["id"]] = measure_text_lines(binary) if binary is not None else None
    consensus_bold = classify_bold_by_label(label_stroke_ratios)
    consensus_italic = classify_italic_by_label(label_slant_angles)
    # A label with too few matched words of its own (titles/footers are
    # often 1-2 blocks) is noisy -- defer to the whole document's pooled
    # words instead, since headings usually share body text's family.
    global_font_category, global_margin = classify_word_samples(
        [s for samples in label_word_samples.values() for s in samples]
    )
    if global_font_category == "unknown":
        global_font_category = DEFAULT_FONT_CATEGORY
    consensus_font_category = {}
    label_margins = {}
    for label, samples in label_word_samples.items():
        category, margin = "unknown", 0.0
        if len(samples) >= MIN_LABEL_FONT_WORDS:
            category, margin = classify_word_samples(samples)
        label_margins[label] = (category, round(margin, 3), len(samples))
        # a role only departs from the document's family on clear evidence
        if category == "unknown" or (category != global_font_category and margin < LABEL_FONT_OVERRIDE_MARGIN):
            category = global_font_category
        consensus_font_category[label] = category
    logger.debug("font family: document=%s (margin %.3f) per-label=%s evidence=%s",
                 global_font_category, global_margin, consensus_font_category, label_margins)

    # Size each block from its own pixels -- the size whose x-height matches
    # the measured one, kept within the range that reproduces the block's
    # measured line count at its width -- then collapse per label to one
    # size per visual-role cluster (real documents use one size per role),
    # still honoring each block's own line-count range. Blocks that can't
    # be measured fall back to fitting the text into the box.
    label_fit_sizes: dict[str, list[tuple[int, float]]] = {}
    block_size_range: dict[int, tuple[float, float] | None] = {}
    block_leading: dict[int, float] = {}
    block_text_width: dict[int, tuple[str, str, float]] = {}  # content, font, box width (pt)
    for block in blocks:
        label, content = block["label"], block["content"]
        if label in IMAGE_LABELS or label in SKIP_LABELS or not content:
            continue
        x1, y1, x2, y2 = block["bbox"]
        fit_label = label if label in styles_spec else "text"
        bold = consensus_bold.get(fit_label, False)
        italic = consensus_italic.get(fit_label, False)
        font_category = consensus_font_category.get(fit_label, DEFAULT_FONT_CATEGORY)
        stats = line_stats.get(block["id"])
        if stats is not None:
            size, size_range = _measured_font_size(fit_label, content, px(x2 - x1), stats, scale,
                                                   bold, italic, font_category)
            block_size_range[block["id"]] = size_range
            if stats.n_lines == 1 and "\n" not in content:
                block_text_width[block["id"]] = (
                    content, _resolve_font_name(font_category, bold, italic), px(x2 - x1))
            if stats.n_lines >= 2 and stats.pitch:
                block_leading[block["id"]] = stats.pitch * scale
            else:
                # one line: its box is its line height, so don't let the
                # label's default leading make it taller than the original
                spec = styles_spec.get(fit_label, FALLBACK_STYLE)
                block_leading[block["id"]] = min(size * spec["leading"] / spec["fontSize"],
                                                 max(px(y2 - y1), size))
        else:
            size = _fit_font_size(fit_label, content, px(x2 - x1), px(y2 - y1),
                                  bold=bold, italic=italic, font_category=font_category)
        label_fit_sizes.setdefault(fit_label, []).append((block["id"], size))

    # A block's size that reproduces its measured line breaks can differ a
    # lot from the size that matches its measured x-height whenever the
    # original font is narrower/wider than the standard-14 substitute (e.g. a
    # condensed sans). Rather than trading one for the other, draw at the
    # x-height size and scale glyphs horizontally so the same words land on
    # each line: wrapping at size s in width W is identical to size s/k in
    # width W/k, drawn k-times wider.
    block_resolved_size: dict[int, float] = {}
    block_hscale: dict[int, float] = {}
    cluster_plans = []
    for pairs in label_fit_sizes.values():
        for cluster in _cluster_label_sizes(pairs):
            # single-line x-height readings are noisy; prefer multi-line blocks
            reliable = [s for bid, s in cluster
                        if (st := line_stats.get(bid)) is not None and st.n_lines >= 2]
            visual_size = statistics.median(reliable or [s for _, s in cluster])
            ideal_hscales = []  # per block: (lowest, highest) width keeping its line breaks
            for bid, _ in cluster:
                if bid in block_text_width:
                    # one line: the detector's box is the line's own width
                    text, font_name, box_w = block_text_width[bid]
                    natural_w = stringWidth(text, font_name, visual_size)
                    if natural_w > 0:
                        v = box_w / natural_w
                        ideal_hscales.append((v * 0.97, v * 1.03))
                elif (r := block_size_range.get(bid)):
                    ideal_hscales.append((r[0] / visual_size, r[1] / visual_size))
            cluster_plans.append((cluster, visual_size, ideal_hscales))
    # the original typeface's width is a document-wide trait: pick the width
    # that keeps the most blocks' line breaks overall, then let each role
    # deviate only as far as its own blocks require
    doc_hscale = _best_shared_scale([iv for _, _, ivs in cluster_plans for iv in ivs], 1.0)
    for cluster, visual_size, ideal_hscales in cluster_plans:
        hscale = _best_shared_scale(ideal_hscales, doc_hscale)
        hscale = min(max(hscale, MIN_HSCALE), MAX_HSCALE)
        # one size per role; a block that would wrap onto extra lines at
        # the role's width only condenses a bit more (never a new size)
        for block_id, _ in cluster:
            block_resolved_size[block_id] = visual_size
            r = block_size_range.get(block_id)
            block_hscale[block_id] = (hscale if r is None or r[1] / visual_size >= hscale
                                      else max(r[1] / visual_size, MIN_HSCALE))
        logger.debug("size cluster %s: %.2fpt x%.2f wide",
                     [bid for bid, _ in cluster], visual_size, hscale)

    col_cursor_px: dict[int, float] = {}  # column_key -> lowest y used so far (image-pixel space)
    global_cursor_px = 0.0  # advanced by full-width spanning blocks

    c = canvas.Canvas(output_path, pagesize=(page_w, page_h))

    for block in blocks:
        label, content = block["label"], block["content"]
        x1, y1, x2, y2 = block["bbox"]
        col = block["_col"]
        width_frac = (x2 - x1) / img_w

        eff_y1 = max(y1, col_cursor_px.get(col, 0), global_cursor_px)

        x = px(x1)
        w = px(x2 - x1)
        top_y_pdf = page_h - px(eff_y1)
        bbox_h = px(y2 - y1)

        if label in IMAGE_LABELS:
            h = bbox_h
            y = top_y_pdf - h
            crop = _crop_block_image(source_image, block["bbox"])
            if crop is not None:
                _draw_pil_image(c, crop, x, y, w, h)
            else:
                c.setFillColor(colors.HexColor("#F5F7F9"))
                c.setStrokeColor(colors.HexColor("#D5DAE0"))
                c.rect(x, y, w, h, fill=1, stroke=1)
                _draw_fitted_placeholder(c, x, y, w, h)
            new_bottom_px = eff_y1 + (y2 - y1)
        elif label in SKIP_LABELS or not content:
            new_bottom_px = eff_y1 + (y2 - y1)
        else:
            fit_label = label if label in styles_spec else "text"
            font_size = block_resolved_size[block["id"]]
            bold = consensus_bold.get(fit_label, False)
            italic = consensus_italic.get(fit_label, False)
            font_category = consensus_font_category.get(fit_label, DEFAULT_FONT_CATEGORY)
            leading = block_leading.get(block["id"])
            style = _build_style(fit_label, font_size=font_size, bold=bold,
                                  italic=italic, font_category=font_category,
                                  leading=None if leading is None else max(leading, MIN_LEADING_RATIO * font_size))
            para = Paragraph(_sanitize(content), style)
            hscale = block_hscale.get(block["id"], 1.0)
            _, needed_h = para.wrap(max(w / hscale, 1), 20000)
            stats = line_stats.get(block["id"])
            logger.debug(
                "block %d (%s): %s %.2fpt x%.2f wide, leading %.2fpt, lines rendered %d / measured %s, "
                "height %.1f / box %.1f",
                block["id"], label, style.fontName, font_size, hscale, style.leading, len(para.blPara.lines),
                stats.n_lines if stats else "?", needed_h, bbox_h,
            )
            h = max(bbox_h, needed_h + 2)
            y = top_y_pdf - h
            c.saveState()
            c.scale(hscale, 1)
            frame = Frame(x / hscale, y, w / hscale, h, leftPadding=0, rightPadding=0,
                           topPadding=0, bottomPadding=0, showBoundary=0)
            frame.addFromList([para], c)
            c.restoreState()
            new_bottom_px = eff_y1 + (h / scale)

        if width_frac >= FULL_WIDTH_FRAC:
            global_cursor_px = max(global_cursor_px, new_bottom_px)
        col_cursor_px[col] = max(col_cursor_px.get(col, 0), new_bottom_px)

    c.showPage()
    c.save()
    return output_path
