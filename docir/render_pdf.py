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
import statistics

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Paragraph

from .bold import classify_bold_by_label, measure_stroke_ratio
from .fonts import match_style_category
from .italic import classify_italic_by_label, measure_slant_angle
from .page_ratios import closest_standard_ratio
from .schemas import PageLayout

# ---------- tunables (safe defaults, override per-call if needed) ----------
TARGET_PAGE_WIDTH_PT = 612.0     # scale output to a standard 8.5in-wide page
COLUMN_GAP_THRESHOLD_FRAC = 0.05  # x1 gap (as a fraction of page width) that starts a new column
FULL_WIDTH_FRAC = 0.6            # a block this wide (as a fraction of page width) is a spanning header
FONT_FIT_ITERATIONS = 18         # binary-search steps; 18 halvings easily beats 0.1pt precision
SIZE_CLUSTER_GAP_RATIO = 1.5      # per-label fitted sizes more than this multiplicative
                                  # gap apart are treated as separate visual roles
MIN_FONT_CATEGORY_SAMPLES = 3     # a label with fewer matches than this defers to the
                                  # document-wide category instead of its own noisy signal
STANDARD_RATIO_TOLERANCE = 0.05  # snap the page canvas to a known paper/photo ratio
                                  # within this relative difference; further off keeps
                                  # the source image's own raw ratio

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


def _build_style(label, font_size=None, bold=False, italic=False, font_category=None):
    """Build a ParagraphStyle for a label, optionally overriding the font size
    (leading scales with it, preserving the label's original leading/size ratio)."""
    spec = DEFAULT_STYLES.get(label, FALLBACK_STYLE)
    align = TA_CENTER if spec.get("align") == "center" else TA_LEFT
    size = font_size if font_size is not None else spec["fontSize"]
    leading_ratio = spec["leading"] / spec["fontSize"]
    font_name = _resolve_font_name(font_category or DEFAULT_FONT_CATEGORY, bold, italic)
    return ParagraphStyle(
        label, fontName=font_name, fontSize=size,
        leading=size * leading_ratio, textColor=colors.HexColor(spec["color"]),
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
        {"id": b.id, "label": b.label, "content": b.content, "bbox": list(b.bbox)}
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
    label_font_category_scores: dict[str, dict[str, float]] = {}
    label_font_category_counts: dict[str, int] = {}
    global_font_category_scores: dict[str, float] = {}
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
        category, score = match_style_category(source_image, bbox, content)
        if category != "unknown":
            scores = label_font_category_scores.setdefault(fit_label, {})
            scores[category] = scores.get(category, 0.0) + score
            label_font_category_counts[fit_label] = label_font_category_counts.get(fit_label, 0) + 1
            global_font_category_scores[category] = global_font_category_scores.get(category, 0.0) + score
    consensus_bold = classify_bold_by_label(label_stroke_ratios)
    consensus_italic = classify_italic_by_label(label_slant_angles)
    # A label with too few of its own matches (titles/footers are often 1-2
    # blocks) is noisy -- defer to the document-wide winner instead, since
    # headings usually share body text's font family anyway.
    global_font_category = (
        max(global_font_category_scores, key=global_font_category_scores.get)
        if global_font_category_scores else DEFAULT_FONT_CATEGORY
    )
    consensus_font_category = {
        label: (
            max(scores, key=scores.get)
            if label_font_category_counts[label] >= MIN_FONT_CATEGORY_SAMPLES
            else global_font_category
        )
        for label, scores in label_font_category_scores.items()
    }

    # Fit each block individually first, then collapse per-label to one
    # size per visual-role cluster (real documents use one size per role,
    # not a size tailored to each block's box) -- a label can still cover
    # more than one cluster if it's reused for genuinely different roles.
    label_fit_sizes: dict[str, list[tuple[int, float]]] = {}
    for block in blocks:
        label, content = block["label"], block["content"]
        if label in IMAGE_LABELS or label in SKIP_LABELS or not content:
            continue
        x1, y1, x2, y2 = block["bbox"]
        fit_label = label if label in styles_spec else "text"
        bold = consensus_bold.get(fit_label, False)
        italic = consensus_italic.get(fit_label, False)
        font_category = consensus_font_category.get(fit_label, DEFAULT_FONT_CATEGORY)
        size = _fit_font_size(fit_label, content, px(x2 - x1), px(y2 - y1),
                               bold=bold, italic=italic, font_category=font_category)
        label_fit_sizes.setdefault(fit_label, []).append((block["id"], size))

    block_resolved_size: dict[int, float] = {}
    for pairs in label_fit_sizes.values():
        for cluster in _cluster_label_sizes(pairs):
            cluster_size = statistics.median(size for _, size in cluster)
            for block_id, _ in cluster:
                block_resolved_size[block_id] = cluster_size

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
            style = _build_style(fit_label, font_size=font_size, bold=bold,
                                  italic=italic, font_category=font_category)
            para = Paragraph(_sanitize(content), style)
            _, needed_h = para.wrap(max(w, 1), 20000)
            h = max(bbox_h, needed_h + 2)
            y = top_y_pdf - h
            frame = Frame(x, y, w, h, leftPadding=0, rightPadding=0,
                           topPadding=0, bottomPadding=0, showBoundary=0)
            frame.addFromList([para], c)
            new_bottom_px = eff_y1 + (h / scale)

        if width_frac >= FULL_WIDTH_FRAC:
            global_cursor_px = max(global_cursor_px, new_bottom_px)
        col_cursor_px[col] = max(col_cursor_px.get(col, 0), new_bottom_px)

    c.showPage()
    c.save()
    return output_path
