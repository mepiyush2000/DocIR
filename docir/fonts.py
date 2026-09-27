"""Local, filesystem-based font catalog + render-and-compare style matching.

No font files are bundled or redistributed by this package -- it only reads
a small, well-known shortlist of common system fonts that may already be
installed locally, purely to compare their rendered shapes against a
block's own ink pixels. If none of them are present on a given machine,
matching gracefully degrades to "unknown" rather than guessing.

This only distinguishes broad style *categories* (serif / sans_serif /
monospace) and never produces a numeric font-weight class -- bold vs.
regular stays a separate binary signal (see `docir.bold`).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .ink import binarize_region

_CANDIDATE_DIRS = [
    "/System/Library/Fonts/Supplemental",
    "/System/Library/Fonts",
    "/Library/Fonts",
    os.path.expanduser("~/Library/Fonts"),
    "/usr/share/fonts/truetype/dejavu",  # common on Linux
    "/usr/share/fonts/truetype/liberation",  # common on Linux
    "/usr/share/fonts/truetype/msttcorefonts",  # common on Linux
    "/usr/share/fonts/opentype/noto",  # common on Linux
    "/usr/share/fonts/truetype/noto",  # common on Linux
    "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/.local/share/fonts"),
    "C:/Windows/Fonts",  # Windows
]

# A broad shortlist of common system fonts -- enough to tell serif from
# sans-serif from monospace by stroke-terminal shape, not to identify one
# exact family. Several (family, filename) entries exist per family to
# cover filename-casing/format differences across macOS/Linux/Windows;
# `discover_catalog()` only keeps entries whose file actually exists.
_KNOWN_FONTS = [
    # Serif
    ("Times New Roman", "serif", "Times New Roman.ttf"),
    ("Times New Roman", "serif", "times.ttf"),
    ("Georgia", "serif", "Georgia.ttf"),
    ("Georgia", "serif", "georgia.ttf"),
    ("Cambria", "serif", "cambria.ttc"),
    ("Cambria", "serif", "Cambria.ttf"),
    ("Garamond", "serif", "Garamond.ttf"),
    ("Book Antiqua", "serif", "Book Antiqua.ttf"),
    ("Palatino", "serif", "Palatino.ttc"),
    ("Hoefler Text", "serif", "Hoefler Text.ttc"),
    ("PT Serif", "serif", "PTSerif-Regular.ttf"),
    ("Noto Serif", "serif", "NotoSerif-Regular.ttf"),
    ("Liberation Serif", "serif", "LiberationSerif-Regular.ttf"),
    ("DejaVu Serif", "serif", "DejaVuSerif.ttf"),
    ("Cochin", "serif", "Cochin.ttc"),
    ("Didot", "serif", "Didot.ttc"),
    ("American Typewriter", "serif", "AmericanTypewriter.ttc"),

    # Sans-serif
    ("Arial", "sans_serif", "Arial.ttf"),
    ("Arial", "sans_serif", "arial.ttf"),
    ("Helvetica", "sans_serif", "Helvetica.ttc"),
    ("Verdana", "sans_serif", "Verdana.ttf"),
    ("Verdana", "sans_serif", "verdana.ttf"),
    ("Tahoma", "sans_serif", "Tahoma.ttf"),
    ("Tahoma", "sans_serif", "tahoma.ttf"),
    ("Calibri", "sans_serif", "calibri.ttf"),
    ("Segoe UI", "sans_serif", "segoeui.ttf"),
    ("Century Gothic", "sans_serif", "GOTHIC.TTF"),
    ("Trebuchet MS", "sans_serif", "Trebuchet MS.ttf"),
    ("PT Sans", "sans_serif", "PTSans-Regular.ttf"),
    ("Noto Sans", "sans_serif", "NotoSans-Regular.ttf"),
    ("Liberation Sans", "sans_serif", "LiberationSans-Regular.ttf"),
    ("DejaVu Sans", "sans_serif", "DejaVuSans.ttf"),
    ("Futura", "sans_serif", "Futura.ttc"),
    ("Optima", "sans_serif", "Optima.ttc"),
    ("Avenir", "sans_serif", "Avenir.ttc"),
    ("Geneva", "sans_serif", "Geneva.ttf"),
    ("Comic Sans MS", "sans_serif", "Comic Sans MS.ttf"),

    # Monospace
    ("Courier New", "monospace", "Courier New.ttf"),
    ("Courier New", "monospace", "cour.ttf"),
    ("Consolas", "monospace", "consola.ttf"),
    ("Menlo", "monospace", "Menlo.ttc"),
    ("Monaco", "monospace", "Monaco.ttf"),
    ("PT Mono", "monospace", "PTMono-Regular.ttf"),
    ("Noto Sans Mono", "monospace", "NotoSansMono-Regular.ttf"),
    ("Liberation Mono", "monospace", "LiberationMono-Regular.ttf"),
    ("DejaVu Sans Mono", "monospace", "DejaVuSansMono.ttf"),
    ("Andale Mono", "monospace", "Andale Mono.ttf"),
]

MIN_MATCH_TEXT_LEN = 3  # too-short text can't discriminate styles reliably
# Pixel-level render-and-compare is inherently noisy (subpixel alignment,
# hinting) -- even a perfect same-font match rarely exceeds ~0.2 IoU, so this
# is calibrated well below "looks like a clean match" and instead just
# filters out near-zero/no-signal scores, relying on relative best-of-catalog
# ranking (like `docir.bold`'s document-relative threshold) rather than an
# absolute confidence cutoff.
MIN_MATCH_SCORE = 0.12

_FONT_EXTENSIONS = (".ttf", ".ttc", ".otf")

# Strips weight/style suffixes (repeatedly, so "Arial Narrow Bold Italic"
# collapses the same as "Arial") so every weight/style variant of one family
# found on disk becomes a single match candidate instead of a redundant one.
_STYLE_SUFFIX_RE = re.compile(
    r"[-_ ]?(bold|italic|oblique|light|medium|semibold|narrow|condensed|"
    r"regular|book|heavy|thin|black|extrabold|extralight|ultralight|"
    r"unicode|w\d+)$",
    re.IGNORECASE,
)

_MONO_KEYWORDS = ("mono", "consol", "courier", "code", "terminal", "typewriter")
_SERIF_KEYWORDS = (
    "serif", "times", "georgia", "garamond", "palatino", "cambria", "antiqua",
    "didot", "cochin", "hoefler", "minion", "caslon", "baskerville", "cardo",
    "playfair", "merriweather", "lora", "charter", "bodoni", "athelas",
)

# System font directories (esp. macOS's Supplemental folder) mostly contain
# non-Latin-script, symbol, or emoji fonts -- hundreds of them. They can't
# usefully discriminate Latin serif/sans/monospace shape, render our target
# text as fallback "tofu" glyphs, and dominate matching time for no benefit,
# so the opportunistic full-directory scan skips anything matching these
# (the curated `_KNOWN_FONTS` list above is unaffected).
_SCAN_EXCLUDE_KEYWORDS = (
    "noto", "braille", "emoji", "symbol", "dingbat", "wingding", "webding",
    "stix", "lastresort", "cjk", "hiragino", "heiti", "songti", "pingfang",
    "applegothic", "applemyungjo", "applesdgothicneo", "aquakana",
    "arabic", "geeza", "damascus", "farisi", "mishafi", "nadeem", "waseem",
    "diwan", "kufi", "baghdad", "al bayan", "albayan", "al nile", "al tarikh",
    " sana.", "farah", "raanana", "muna", "decotype",
    "hebrew", "armenian", "sfgeorgian",
    "devanagari", "gujarati", "gurmukhi", "kannada", "malayalam", "myanmar",
    "oriya", "sinhala", "tamil", "telugu", "khmer", "lao ", "laosangam",
    "thonburi", "sukhumvit", "silom", "krungthep", "bangla", "kohinoor",
    "shree714", "muktamahee", "itfdevanagari", "zither", "euphemia",
    "plantagenetcherokee", "adtnumeric", "inaimathi", "newpeninimmt",
    "mshtakan", "galvji",
)


def _base_family_key(filename: str) -> str:
    """Collapse a filename down to one key per family, ignoring extension
    and weight/style suffixes."""
    stem = os.path.splitext(filename)[0]
    prev = None
    while prev != stem:
        prev = stem
        stem = _STYLE_SUFFIX_RE.sub("", stem).strip(" -_")
    return stem.lower()


def _guess_category(filename: str) -> str:
    """Rough serif/sans/monospace guess for a font with no curated
    metadata -- filename keywords only, no font parsing. Defaults to
    sans_serif since that's the more common fallback family."""
    name = filename.lower()
    if any(k in name for k in _MONO_KEYWORDS):
        return "monospace"
    if any(k in name for k in _SERIF_KEYWORDS):
        return "serif"
    return "sans_serif"


def _is_scannable(filename: str) -> bool:
    """Whether a discovered (non-curated) font file is worth adding as a
    match candidate -- excludes non-Latin-script, symbol, and emoji fonts
    that only add matching cost without discriminating Latin text shape."""
    name = filename.lower()
    return not any(k in name for k in _SCAN_EXCLUDE_KEYWORDS)


def _file_identity(path: str):
    """Identity key that's correct even on case-insensitive filesystems
    (where e.g. "Georgia.ttf" and "georgia.ttf" are the same file)."""
    try:
        st = os.stat(path)
        return (st.st_dev, st.st_ino)
    except OSError:
        return path


@dataclass(frozen=True)
class FontFace:
    family: str
    style_category: str  # "serif" | "sans_serif" | "monospace"
    path: str


@lru_cache(maxsize=1)
def discover_catalog() -> tuple[FontFace, ...]:
    """Build the match-candidate list from every renderable font file this
    machine actually has: curated entries (`_KNOWN_FONTS`) use their known
    category, and any other font file found in the same directories is
    still included as a candidate with a filename-based category guess --
    collapsed to one representative per family/weight-group so e.g. "Arial
    Bold" and "Arial Italic" don't each become a redundant match candidate
    alongside plain "Arial"."""
    faces: dict[tuple, FontFace] = {}  # keyed by file identity, for dedup
    seen_base_keys: set[str] = set()

    def add(family, category, path, base_key):
        identity = _file_identity(path)
        if identity in faces:
            return
        faces[identity] = FontFace(family, category, path)
        seen_base_keys.add(base_key)

    for family, category, filename in _KNOWN_FONTS:
        for directory in _CANDIDATE_DIRS:
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                add(family, category, path, _base_family_key(filename))
                break

    for directory in _CANDIDATE_DIRS:
        if not os.path.isdir(directory):
            continue
        for entry in sorted(os.listdir(directory)):
            if not entry.lower().endswith(_FONT_EXTENSIONS):
                continue
            if not _is_scannable(entry):
                continue
            base_key = _base_family_key(entry)
            if base_key in seen_base_keys:
                continue
            path = os.path.join(directory, entry)
            add(os.path.splitext(entry)[0], _guess_category(entry), path, base_key)

    return tuple(faces.values())


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont | None:
    """Load+parse a font file only once per (path, size) -- re-parsing a
    TTF/TTC from disk on every single render call dominates matching cost
    once the catalog has hundreds of candidates."""
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return None


_load_font = lru_cache(maxsize=4096)(_load_font)


def _render_text_mask(face: FontFace, text: str, height_px: int) -> np.ndarray | None:
    """Rasterize `text` in `face` at a size matched to `height_px`, and
    return a tight boolean ink mask cropped to its own bounding box."""
    size = max(int(height_px), 8)
    font = _load_font(face.path, size)
    if font is None:
        return None
    canvas_w = size * max(len(text), 1) * 2
    canvas_h = size * 3
    img = Image.new("L", (canvas_w, canvas_h), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((size, size), text, font=font, fill=0)
    arr = np.asarray(img)
    ink_cols = np.where((arr < 128).any(axis=0))[0]
    ink_rows = np.where((arr < 128).any(axis=1))[0]
    if ink_cols.size == 0 or ink_rows.size == 0:
        return None
    cropped = arr[ink_rows[0] : ink_rows[-1] + 1, ink_cols[0] : ink_cols[-1] + 1]
    return cropped < 128


def _first_line_mask(binary: np.ndarray) -> np.ndarray:
    """A multi-line block's overall ink mask can't be compared against one
    rendered line of text (line breaks, inter-line gaps), so restrict the
    comparison to just its first physical text line -- including the full
    ascender-to-descender extent, since font shape (e.g. serifs) lives in
    those extremes, not just the x-height core."""
    has_ink = binary.any(axis=1)
    rows = np.where(has_ink)[0]
    if rows.size == 0:
        return binary
    # a real inter-line gap is several consecutive fully-blank rows; a
    # single-row gap can just be a natural dip inside one line of glyphs
    gap_rows = np.where(~has_ink[rows[0] : rows[-1] + 1])[0] + rows[0]
    line_end = rows[-1] + 1
    if gap_rows.size:
        run_start = gap_rows[0]
        run_len = 1
        for r in gap_rows[1:]:
            if r == run_start + run_len:
                run_len += 1
            else:
                if run_len >= 2:
                    line_end = run_start
                    break
                run_start, run_len = r, 1
        else:
            if run_len >= 2:
                line_end = run_start
    return binary[rows[0] : line_end]


def _crop_to_ink(mask: np.ndarray) -> np.ndarray | None:
    """Tightly crop `mask` to its own ink bounding box."""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _scale_to_height(mask: np.ndarray, target_height: int) -> np.ndarray:
    """Uniformly scale `mask` (same factor on both axes, so glyph
    proportions like cap-height/x-height ratios are preserved) so its
    height matches `target_height`."""
    h, w = mask.shape
    if h == 0 or w == 0:
        return mask
    scale = target_height / h
    new_w = max(int(round(w * scale)), 1)
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = img.resize((new_w, target_height), resample=Image.BILINEAR)
    return np.asarray(resized) > 127


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union else 0.0


def _match_from_mask(observed_mask: np.ndarray, text: str) -> tuple[str, float]:
    catalog = discover_catalog()
    clean_text = text.strip()
    cropped_observed = _crop_to_ink(_first_line_mask(observed_mask))
    if not catalog or len(clean_text) < MIN_MATCH_TEXT_LEN or cropped_observed is None:
        return "unknown", 0.0

    # font hinting/grid-fitting is size-dependent -- rendering a candidate at
    # a wildly different pixel size than the observed text (then rescaling)
    # distorts its width/height proportions. Render close to the observed
    # size instead, so both go through comparable hinting.
    observed_height = cropped_observed.shape[0]
    best_category, best_score = "unknown", 0.0
    for face in catalog:
        rendered = _render_text_mask(face, clean_text, observed_height)
        if rendered is None:
            continue
        rendered_scaled = _scale_to_height(rendered, observed_height)
        # a block's ink mask may only be its first line, while `text` is the
        # whole block's content -- compare just the overlapping left portion
        # rather than penalizing for the length difference.
        width = min(cropped_observed.shape[1], rendered_scaled.shape[1])
        if width == 0:
            continue
        a = cropped_observed[:, :width]
        b = rendered_scaled[:, :width]
        score = _iou(a, b)
        if score > best_score:
            best_score = score
            best_category = face.style_category

    if best_score < MIN_MATCH_SCORE:
        return "unknown", best_score
    return best_category, best_score


def match_style_category(source_image, bbox, text: str) -> tuple[str, float]:
    """Render `text` with each known local font, compare its ink mask
    against this block's real pixels, and return the best-matching
    (style_category, score). Returns ("unknown", 0.0) if it can't tell (no
    image, no local catalog, text too short, or no confident match)."""
    binary = binarize_region(source_image, bbox)
    if binary is None:
        return "unknown", 0.0
    return _match_from_mask(binary, text)
