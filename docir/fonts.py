"""Local, filesystem-based font catalog + word-level render-and-compare
style matching.

No font files are bundled or redistributed by this package -- it only reads
fonts that are already installed locally, purely to compare their rendered
shapes against a block's own ink pixels. If none are present on a given
machine, matching gracefully degrades to "unknown" rather than guessing.

This only distinguishes broad style *categories* (serif / sans_serif /
monospace) and never produces a numeric font-weight class -- bold vs.
regular stays a separate binary signal (see `docir.bold`).

How matching works (and why):
  1. Every candidate font file is *categorized by rendering it*: a serif
     "I"/"l" has feet much wider than its stem, a monospace font gives "i"
     and "m" the same advance, and script/decorative/small-caps/symbol
     fonts fail simple upright-text-glyph checks and are dropped. Filename
     guesses mislabel far too many fonts (e.g. "Microsoft Sans Serif" as
     serif, or dozens of handwriting fonts as sans-serif) to be trusted.
  2. The block is cut into word blobs, and each blob is aligned to the
     OCR'd word it shows (monotonic alignment scored against a few
     reference fonts), so every comparison is the *same word* rendered vs.
     observed -- not a whole line whose glyph set and length differ.
  3. Each observed word is deskewed and compared at sub-pixel precision
     (upsampled soft ink, not a hard mask) against renders that are first
     pushed through the same resolution bottleneck as the scan, so a
     crisp high-res serif isn't unfairly penalized against a blurry one.
     Scores are band-pass NCC (shape detail, not overall ink mass) with a
     small stroke-weight search, so bold text still matches its family.
  4. Categories are scored by the mean of their best few fonts, so a
     category with more installed fonts doesn't win just by having more
     chances at a lucky match; the returned score is the winning margin.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .ink import close_horizontal, component_boxes, estimate_skew, ink_maps, measure_text_lines, rotate_mask

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

# Well-known text fonts looked up first (in this order) so they win the
# one-file-per-family dedup below; their category is still verified by
# rendering, like every other font.
_KNOWN_FONT_FILES = [
    "Times New Roman.ttf", "times.ttf", "Georgia.ttf", "georgia.ttf",
    "cambria.ttc", "Cambria.ttf", "Garamond.ttf", "Book Antiqua.ttf",
    "Palatino.ttc", "Hoefler Text.ttc", "PTSerif-Regular.ttf",
    "NotoSerif-Regular.ttf", "LiberationSerif-Regular.ttf", "DejaVuSerif.ttf",
    "Cochin.ttc", "Didot.ttc", "Charter.ttc", "Baskerville.ttc",
    "Arial.ttf", "arial.ttf", "Helvetica.ttc", "Verdana.ttf", "verdana.ttf",
    "Tahoma.ttf", "tahoma.ttf", "calibri.ttf", "segoeui.ttf", "GOTHIC.TTF",
    "Trebuchet MS.ttf", "PTSans-Regular.ttf", "NotoSans-Regular.ttf",
    "LiberationSans-Regular.ttf", "DejaVuSans.ttf", "Futura.ttc", "Optima.ttc",
    "Avenir.ttc", "Geneva.ttf",
    "Courier New.ttf", "cour.ttf", "consola.ttf", "Menlo.ttc", "Monaco.ttf",
    "PTMono-Regular.ttf", "NotoSansMono-Regular.ttf",
    "LiberationMono-Regular.ttf", "DejaVuSansMono.ttf", "Andale Mono.ttf",
]

MIN_MATCH_TEXT_LEN = 3  # too-short text can't discriminate styles reliably
# Category scores closer than this are too close to call -> "unknown".
MIN_CATEGORY_MARGIN = 0.01
CATEGORY_TOP_K = 3       # a category's score = mean of its best K fonts
MAX_WORDS_PER_BLOCK = 6  # widest aligned words compared against every font
ALIGN_MAX_BLOBS = 18     # reading-order prefix of blobs/words considered
ALIGN_MAX_WORDS = 28     # for alignment (plenty to find MAX_WORDS_PER_BLOCK)
ALIGN_MIN_SCORE = 0.45   # reference-font NCC needed to accept a blob<->word pair
ALIGN_MAX_ASPECT_LOG = 0.3  # |log(aspect ratio)| gate before scoring a pair
MIN_WORD_HEIGHT_PX = 6   # smaller blobs have no usable shape detail
WORD_GAP_FRAC = 0.35     # letter gaps below this x-height fraction join a word
COMPARE_HEIGHT_PX = 48   # words are compared at this (upsampled) height
RENDER_SIZE_PX = 64      # candidate words are rendered at this font size
BLOCK_PAD_PX = 4         # a bbox cut tight to the text shouldn't clip edge glyphs
BLUR_OPTIONS_PX = (0.0, 0.5, 0.9)  # candidate scan blur (sigma, native px) per word

_FONT_EXTENSIONS = (".ttf", ".ttc", ".otf")

# Heavy-only files (e.g. "DIN Condensed Bold") would give bold-looking scans
# an unfair head start; stroke weight is matched separately by the
# stroke-weight search in `_shape_score`, so only regular weights compete.
_HEAVY_WEIGHT_RE = re.compile(r"(bold|black|heavy)", re.IGNORECASE)

# Strips weight/style suffixes (repeatedly, so "Arial Narrow Bold Italic"
# collapses the same as "Arial") so every weight/style variant of one family
# found on disk becomes a single match candidate instead of a redundant one.
_STYLE_SUFFIX_RE = re.compile(
    r"[-_ ]?(bold|italic|oblique|light|medium|semibold|narrow|condensed|"
    r"regular|book|heavy|thin|black|extrabold|extralight|ultralight|"
    r"unicode|w\d+)$",
    re.IGNORECASE,
)

# System font directories (esp. macOS's Supplemental folder) mostly contain
# non-Latin-script, symbol, emoji, or novelty fonts. They can't represent
# ordinary document text, so they're skipped before any rendering. Anything
# not caught here still has to pass `_classify_font_file`'s glyph checks.
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
    "mshtakan", "galvji", "ヒラギノ",
    # informal/handwriting faces pass upright-glyph checks but aren't
    # plausible document body/title fonts
    "comic", "chalk", "marker", "noteworthy", "luminari", "keyboard",
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


def _is_scannable(filename: str) -> bool:
    name = filename.lower()
    if _HEAVY_WEIGHT_RE.search(os.path.splitext(name)[0]):
        return False
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


# ---------- categorizing font files by rendering them ----------

_CLASSIFY_SIZE_PX = 160


def _glyph_mask(font: ImageFont.FreeTypeFont, ch: str) -> np.ndarray | None:
    size = _CLASSIFY_SIZE_PX
    img = Image.new("L", (size * 3, size * 3), 255)
    ImageDraw.Draw(img).text((size, size // 2), ch, font=font, fill=0)
    return _crop(np.asarray(img) < 128)


def _stem_stats(mask: np.ndarray) -> dict | None:
    """Stem width at mid-height, ink extent at the very top/bottom relative
    to it (serif feet), stem slant, and edge roughness."""
    h, _ = mask.shape
    widths, centers = [], []
    for row in mask[int(h * 0.35): int(h * 0.65)]:
        xs = np.where(row)[0]
        if xs.size == 0 or xs[-1] - xs[0] + 1 != xs.size:  # not one solid run
            continue
        widths.append(xs.size)
        centers.append(xs.mean())
    if len(widths) < 3:
        return None
    stem = float(np.median(widths))
    band = max(2, int(h * 0.06))
    return dict(
        bottom=mask[h - band:].any(axis=0).sum() / stem,
        top=mask[:band].any(axis=0).sum() / stem,
        slope=float(np.polyfit(np.arange(len(centers)), centers, 1)[0]),
        rough=float(np.std(widths) / max(np.mean(widths), 1e-6)),
    )


def _classify_font_file(path: str) -> str | None:
    """"serif" | "sans_serif" | "monospace" for an ordinary upright Latin
    text font, or None for anything else (script, decorative, small-caps,
    symbol, italic-only, or no Latin glyphs)."""
    try:
        font = ImageFont.truetype(path, _CLASSIFY_SIZE_PX)
        adv_i, adv_m, adv_w = (font.getlength(ch) for ch in "imW")
    except Exception:
        return None
    glyphs = {ch: _glyph_mask(font, ch) for ch in "IlxHaA"}
    if any(m is None for m in glyphs.values()):
        return None
    # lowercase x vs. capital H: ordinary text faces sit around 0.65-0.78;
    # caps-only/small-caps/script faces fall well outside
    if not 0.55 <= glyphs["x"].shape[0] / glyphs["H"].shape[0] <= 0.82:
        return None
    # small caps: "a" is just a smaller "A"
    a = _resize_mask(glyphs["a"], (64, 64))
    big_a = _resize_mask(glyphs["A"], (64, 64))
    if (a & big_a).sum() / max((a | big_a).sum(), 1) > 0.7:
        return None
    stats_l, stats_i = _stem_stats(glyphs["l"]), _stem_stats(glyphs["I"])
    if stats_l is None or stats_i is None:
        return None
    if max(abs(stats_l["slope"]), abs(stats_i["slope"])) > 0.06:
        return None  # slanted/script
    if max(stats_l["rough"], stats_i["rough"]) > 0.12:
        return None  # distressed/textured
    if adv_m > 0 and abs(adv_i - adv_m) / adv_m < 0.03 and abs(adv_w - adv_m) / adv_m < 0.03:
        return "monospace"
    has_serifs = stats_i["bottom"] > 1.5 and stats_i["top"] > 1.5 and stats_l["bottom"] > 1.4
    return "serif" if has_serifs else "sans_serif"


@lru_cache(maxsize=1)
def discover_catalog() -> tuple[FontFace, ...]:
    """Every usable text font on this machine, one file per family, each
    categorized from its own rendered glyphs."""
    faces: dict[tuple, FontFace] = {}
    seen_base_keys: set[str] = set()

    def consider(path: str, filename: str):
        base_key = _base_family_key(filename)
        identity = _file_identity(path)
        if base_key in seen_base_keys or identity in faces:
            return
        category = _classify_font_file(path)
        if category is None:
            return
        faces[identity] = FontFace(os.path.splitext(filename)[0], category, path)
        seen_base_keys.add(base_key)

    for filename in _KNOWN_FONT_FILES:
        for directory in _CANDIDATE_DIRS:
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                consider(path, filename)
                break

    for directory in _CANDIDATE_DIRS:
        if not os.path.isdir(directory):
            continue
        for entry in sorted(os.listdir(directory)):
            if entry.lower().endswith(_FONT_EXTENSIONS) and _is_scannable(entry):
                consider(os.path.join(directory, entry), entry)

    return tuple(faces.values())


# ---------- small numpy image helpers ----------

def _crop(mask: np.ndarray) -> np.ndarray | None:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return mask[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(img.resize((shape[1], shape[0]), Image.BILINEAR)) > 127


def _resize_float(a: np.ndarray, shape: tuple[int, int], resample) -> np.ndarray:
    img = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))
    return np.asarray(img.resize((max(shape[1], 1), max(shape[0], 1)), resample), dtype=np.float64) / 255


@lru_cache(maxsize=8)
def _gauss_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    return k / k.sum()


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with edge padding."""
    k = _gauss_kernel(sigma)
    r = len(k) // 2
    h, w = a.shape
    p = np.pad(a, ((r, r), (0, 0)), mode="edge")
    a = sum(k[i] * p[i: i + h] for i in range(len(k)))
    p = np.pad(a, ((0, 0), (r, r)), mode="edge")
    return sum(k[i] * p[:, i: i + w] for i in range(len(k)))


def _max_filter(a: np.ndarray, radius: int) -> np.ndarray:
    h, w = a.shape
    n = 2 * radius + 1
    p = np.pad(a, ((radius, radius), (0, 0)), mode="edge")
    a = np.max([p[i: i + h] for i in range(n)], axis=0)
    p = np.pad(a, ((0, 0), (radius, radius)), mode="edge")
    return np.max([p[:, i: i + w] for i in range(n)], axis=0)


def _band_pass(a: np.ndarray) -> np.ndarray:
    return _blur(a, 0.8) - _blur(a, 3.0)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom else 0.0


# ---------- rendering candidate words ----------

def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont | None:
    """Load+parse a font file only once per (path, size)."""
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return None


_load_font = lru_cache(maxsize=4096)(_load_font)


@lru_cache(maxsize=8192)
def _render_word(path: str, word: str) -> np.ndarray | None:
    """`word` rendered at RENDER_SIZE_PX, cropped to its ink, as 0..1 ink."""
    size = RENDER_SIZE_PX
    font = _load_font(path, size)
    if font is None:
        return None
    try:
        width = int(font.getlength(word)) + 2 * size
    except Exception:
        return None
    img = Image.new("L", (width, size * 3), 0)
    ImageDraw.Draw(img).text((size, size), word, font=font, fill=255)
    arr = np.asarray(img, dtype=np.float64) / 255
    rows = np.where((arr > 0.5).any(axis=1))[0]
    cols = np.where((arr > 0.5).any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return arr[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]


def _degraded_render(path: str, word: str, native_shape, compare_shape, blur: float = 0.0) -> np.ndarray | None:
    """A render squeezed through the observed word's native pixel size (and
    optical blur), then upsampled exactly like the observation -- so both
    lose the same detail."""
    hi = _render_word(path, word)
    if hi is None:
        return None
    low = _resize_float(hi, native_shape, Image.BOX)
    if blur > 0:
        low = _blur(low, blur)
    return _resize_float(low, compare_shape, Image.BICUBIC)


def _shape_score(obs_bp: np.ndarray, render: np.ndarray) -> float:
    """Band-pass NCC, best over a small stroke-weight search (so a bold
    observation still matches a regular-weight file of the same family)."""
    best = _ncc(obs_bp, _band_pass(render))
    for radius in (1, 2):
        best = max(best, _ncc(obs_bp, _band_pass(_max_filter(render, radius))))
    return best


# ---------- observed words ----------

@dataclass(frozen=True)
class _ObservedWord:
    soft: np.ndarray  # deskewed, upsampled to COMPARE_HEIGHT_PX, 0..1 ink
    band_pass: np.ndarray
    native_shape: tuple[int, int]
    width: int
    blur: float = 0.0  # native-scale blur that best explains this scan


def _word_boxes(binary: np.ndarray, x_height: float) -> list[tuple[int, int, int, int]]:
    """Word blobs (y0, y1, x0, x1) in reading order: letters joined by
    closing small horizontal gaps, then chained into lines left-to-right by
    vertical-center proximity (so curved lines still chain correctly)."""
    gap = max(2, int(round(WORD_GAP_FRAC * x_height)))
    boxes = [
        b for b in component_boxes(close_horizontal(binary, gap))
        if 0.8 * x_height <= b[1] - b[0] <= 3.5 * x_height and b[3] - b[2] >= 0.8 * x_height
    ]
    boxes.sort(key=lambda b: b[2])
    lines: list[list[tuple[int, int, int, int]]] = []
    for b in boxes:
        cy = (b[0] + b[1]) / 2
        best = None
        for line in lines:
            last = line[-1]
            dy = abs(cy - (last[0] + last[1]) / 2)
            if b[2] >= last[3] - 2 and b[2] - last[3] < 6 * x_height and dy < 0.9 * x_height:
                if best is None or dy < best[0]:
                    best = (dy, line)
        if best:
            best[1].append(b)
        else:
            lines.append([b])
    lines.sort(key=lambda line: (line[0][0] + line[0][1]) / 2)
    return [b for line in lines for b in line]


def _observe_word(binary: np.ndarray, soft: np.ndarray, box) -> _ObservedWord | None:
    y0, y1, x0, x1 = box
    if y1 - y0 < MIN_WORD_HEIGHT_PX:
        return None
    angle = estimate_skew(binary[y0:y1, x0:x1], max_deg=8.0)
    pad = 2
    patch = soft[max(0, y0 - pad): y1 + pad, max(0, x0 - pad): x1 + pad]
    img = Image.fromarray((patch * 255).astype(np.uint8))
    if angle:
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=0)
    arr = np.asarray(img, dtype=np.float64) / 255
    rows = np.where((arr > 0.5).any(axis=1))[0]
    cols = np.where((arr > 0.5).any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    arr = arr[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]
    h, w = arr.shape
    if h < MIN_WORD_HEIGHT_PX:
        return None
    compare = (COMPARE_HEIGHT_PX, max(int(round(w * COMPARE_HEIGHT_PX / h)), 4))
    up = _resize_float(arr, compare, Image.BICUBIC)
    return _ObservedWord(up, _band_pass(up), (h, w), w)


def _reference_faces(catalog) -> list[FontFace]:
    """A couple of common fonts per category, used only to align blobs to
    words (not to decide the category)."""
    preferred = ("Helvetica", "Arial", "Verdana", "Times New Roman", "Times", "Georgia")
    chosen = [f for name in preferred for f in catalog if f.family == name]
    for category in ("sans_serif", "serif"):
        if not any(f.style_category == category for f in chosen):
            chosen += [f for f in catalog if f.style_category == category][:2]
    return chosen[:4]


def _align_words(observed, words, references) -> list[tuple[_ObservedWord, str]]:
    """Monotonic max-weight matching of observed blobs to text words: a
    pair only counts if it looks like that word in some reference font.
    Skipping is free, so stray blobs (drop caps, a caption fragment the
    bbox clipped in) and OCR words with no clean blob just go unmatched."""
    ref_aspects = {}
    for word in set(words):
        ref_aspects[word] = [r.shape[1] / r.shape[0] for f in references
                             if (r := _render_word(f.path, word)) is not None]

    nb, nt = len(observed), len(words)
    score = np.full((nb, nt), -1.0)
    for i, obs in enumerate(observed):
        if obs is None:
            continue
        aspect = obs.soft.shape[1] / obs.soft.shape[0]
        for j, word in enumerate(words):
            aspects = ref_aspects[word]
            if not aspects or min(abs(math.log(aspect / a)) for a in aspects) > ALIGN_MAX_ASPECT_LOG:
                continue
            for f in references:
                r = _degraded_render(f.path, word, obs.native_shape, obs.soft.shape)
                if r is not None:
                    score[i, j] = max(score[i, j], _ncc(obs.band_pass, _band_pass(r)))

    gain = np.where(score > ALIGN_MIN_SCORE, score - ALIGN_MIN_SCORE, -np.inf)
    dp = np.zeros((nb + 1, nt + 1))
    for i in range(1, nb + 1):
        for j in range(1, nt + 1):
            dp[i, j] = max(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1] + gain[i - 1, j - 1])
    pairs = []
    i, j = nb, nt
    while i > 0 and j > 0:
        if np.isfinite(gain[i - 1, j - 1]) and dp[i, j] == dp[i - 1, j - 1] + gain[i - 1, j - 1]:
            pairs.append((observed[i - 1], words[j - 1]))
            i, j = i - 1, j - 1
        elif dp[i, j] == dp[i - 1, j]:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


WordSample = tuple[_ObservedWord, str]


def _collect_from_maps(binary: np.ndarray, soft: np.ndarray, text: str, max_words: int) -> list[WordSample]:
    catalog = discover_catalog()
    words = [w for w in text.split() if any(ch.isalnum() for ch in w)]
    if not catalog or sum(ch.isalnum() for ch in text) < MIN_MATCH_TEXT_LEN or not words:
        return []
    stats = measure_text_lines(binary)
    if stats is None:
        return []
    boxes = _word_boxes(binary, stats.x_height)[:ALIGN_MAX_BLOBS]
    observed = [_observe_word(binary, soft, b) for b in boxes]
    references = _reference_faces(catalog)
    pairs = _align_words(observed, words[:ALIGN_MAX_WORDS], references)
    pairs = sorted(pairs, key=lambda p: -p[0].width)[:max_words]
    return [(_with_blur(obs, word, references), word) for obs, word in pairs]


def _with_blur(obs: _ObservedWord, word: str, references) -> _ObservedWord:
    """A photo/scan is usually blurrier than a clean downsample; pick the
    blur level that lets the reference fonts best explain this word, so
    every candidate font is later degraded the same way."""
    def fit(blur):
        scores = [_shape_score(obs.band_pass, r) for f in references
                  if (r := _degraded_render(f.path, word, obs.native_shape, obs.soft.shape, blur)) is not None]
        return max(scores, default=-1.0)
    return replace(obs, blur=max(BLUR_OPTIONS_PX, key=fit))


def collect_word_samples(source_image, bbox, text: str, max_words: int = MAX_WORDS_PER_BLOCK) -> list[WordSample]:
    """This block's clearest word blobs, each paired with the OCR'd word it
    shows -- evidence for `classify_word_samples`, which can pool samples
    from many blocks (one role, or a whole document) for a steadier call
    than any single short block gives."""
    if source_image is None:
        return []
    x1, y1, x2, y2 = bbox
    padded = (max(x1 - BLOCK_PAD_PX, 0), y1, min(x2 + BLOCK_PAD_PX, source_image.width), y2)
    maps = ink_maps(source_image, padded)
    if maps is None:
        return []
    binary, soft = _deskew_block(*maps)
    return _collect_from_maps(binary, soft, text, max_words)


def classify_word_samples(samples: list[WordSample]) -> tuple[str, float]:
    """(style_category, margin) for pooled word samples: every local font
    renders the same words, a category scores the mean of its best
    CATEGORY_TOP_K fonts, and margin is how far the winner beat the
    runner-up. ("unknown", 0.0) if there's no evidence or no clear winner."""
    catalog = discover_catalog()
    if not samples or not catalog:
        return "unknown", 0.0
    by_category: dict[str, list[float]] = {}
    for face in catalog:
        scores = []
        for obs, word in samples:
            r = _degraded_render(face.path, word, obs.native_shape, obs.soft.shape, obs.blur)
            if r is not None:
                scores.append(_shape_score(obs.band_pass, r))
        if scores:
            by_category.setdefault(face.style_category, []).append(float(np.mean(scores)))

    ranked = sorted(
        ((float(np.mean(sorted(v, reverse=True)[:CATEGORY_TOP_K])), k) for k, v in by_category.items()),
        reverse=True,
    )
    if not ranked:
        return "unknown", 0.0
    best_score, best_category = ranked[0]
    margin = best_score - (ranked[1][0] if len(ranked) > 1 else 0.0)
    if margin < MIN_CATEGORY_MARGIN:
        return "unknown", 0.0
    return best_category, margin


def match_style_category(source_image, bbox, text: str) -> tuple[str, float]:
    """Compare this block's real word shapes against every local font and
    return (style_category, confidence), where confidence is how far the
    winning category beat the runner-up. Returns ("unknown", 0.0) if it
    can't tell (no image, no local catalog, text too short, no word could
    be matched, or no clear winner)."""
    return classify_word_samples(collect_word_samples(source_image, bbox, text))


def _deskew_block(binary: np.ndarray, soft: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Level a whole slanted block (e.g. a photographed, tilted page) before
    cutting it into words, so word blobs don't merge across tilted lines."""
    angle = estimate_skew(binary)
    if abs(angle) < 0.5:
        return binary, soft
    img = Image.fromarray((soft * 255).astype(np.uint8))
    soft = np.asarray(img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=0), dtype=np.float64) / 255
    return rotate_mask(binary, angle), soft
