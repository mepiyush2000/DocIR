"""Role typography: one consistent style per visual role.

Everything is decided per role (layout label), because real documents
style a role the same way throughout:
  * bold / italic / serif-sans-mono category -- measured from pixels,
    pooled over the role's blocks (see `bold`, `italic`, `fonts`);
  * size -- the role's blocks are clustered by measured x-height, each
    cluster gets ONE size (a role like paragraph_title may genuinely come
    in two sizes, never in five);
  * family + width -- among the bundled families of the category, the one
    whose natural line widths best match the measured ones at that size
    (so condensed originals get a condensed face, not squeezed glyphs);
    the small remaining ratio becomes the role's shared glyph width.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import font_catalog
from .bold import classify_bold_by_label, measure_stroke_ratio
from .fonts import classify_word_samples, collect_word_samples
from .italic import classify_italic_by_label, measure_slant_angle
from .line_geometry import measure_line
from .page_model import Block, Line, Page, TextStyle

logger = logging.getLogger(__name__)

DEFAULT_CATEGORY = "sans_serif"
MIN_ROLE_FONT_WORDS = 8          # fewer matched words: defer to the document's category
ROLE_FONT_OVERRIDE_MARGIN = 0.05
FONT_POOL_WORDS_PER_ROLE = 24
SIZE_CLUSTER_RATIO = 1.22        # block x-heights further apart than this are separate sizes
RELIABLE_CHARS = 40              # one-line blocks shorter than this have a noisy x-height
NOISY_JOIN_RATIO = 1.45          # how far such a block may sit from the cluster it joins
MIN_WIDTH_CHARS = 8              # lines shorter than this don't vote on width
WIDTH_PERCENTILE = 30            # justified lines are stretched: favour the tighter ones
MIN_FAMILY_EVIDENCE_CHARS = 60   # less text than this inherits the body family
FAMILY_TIE_LOG = 0.015           # prefer the default family within this width-fit margin
WIDTH_SCALE_RANGE = (0.82, 1.18)
SIZE_MERGE_RATIO = 1.12          # styles closer in size than this are one size
DISPLAY_SIZE_RATIO = 1.6         # text this much larger than body may keep its own face
MIN_TRAIT_CHARS = 40             # less text than this can't establish its own slant
TRAIT_ADOPT_RATIO = 1.35         # ... so it adopts those of a style this close in size
CAPS_MAX_SHORT_FRAC = 0.5        # fewer x-band-only letters than this: the line may measure caps
CAPS_MAX_RISE = 1.25             # ... confirmed by ink barely rising above the measured band
XH_PER_CAP = 0.74                # typical x-height / cap height, before a family is chosen


@dataclass(frozen=True)
class RoleTraits:
    bold: bool
    italic: bool
    category: str


# ---------- bold / italic / category per role ----------

def classify_roles(items: list[tuple[str, tuple, str]], image: Image.Image | None) -> dict[str, RoleTraits]:
    """`items`: (role, bbox, content) per text block."""
    strokes: dict[str, list[float]] = {}
    slants: dict[str, list[float]] = {}
    samples: dict[str, list] = {}
    for role, bbox, content in items:
        if (r := measure_stroke_ratio(image, bbox)) is not None:
            strokes.setdefault(role, []).append(r)
        if (a := measure_slant_angle(image, bbox)) is not None:
            slants.setdefault(role, []).append(a)
        pool = samples.setdefault(role, [])
        if len(pool) < FONT_POOL_WORDS_PER_ROLE:
            pool += collect_word_samples(image, bbox, content, max_words=4)
    chars: dict[str, int] = {}
    for role, _, content in items:
        chars[role] = chars.get(role, 0) + len(content)
    body_role = max(chars, key=chars.get) if chars else None
    bold = classify_bold_by_label(strokes, baseline_label=body_role)
    italic = classify_italic_by_label(slants)
    doc_category, doc_margin = classify_word_samples([s for v in samples.values() for s in v])
    if doc_category == "unknown":
        doc_category = DEFAULT_CATEGORY
    traits = {}
    for role in {r for r, _, _ in items}:
        category, margin = "unknown", 0.0
        if len(samples.get(role, [])) >= MIN_ROLE_FONT_WORDS:
            category, margin = classify_word_samples(samples[role])
        # a role only departs from the document's category on clear evidence
        if category == "unknown" or (category != doc_category and margin < ROLE_FONT_OVERRIDE_MARGIN):
            category = doc_category
        traits[role] = RoleTraits(bold.get(role, False), italic.get(role, False), category)
    logger.debug("role traits: %s (document category %s, margin %.3f)", traits, doc_category, doc_margin)
    return traits


# ---------- measured x-height -> font size ----------

@lru_cache(maxsize=None)
def x_band_bias(font_path: str) -> float:
    """How `measure_line`'s x-height relates to the font's true x-height,
    calibrated on a rendered sample line of that very font."""
    px = 48
    font = ImageFont.truetype(font_path, px)
    img = Image.new("L", (px * 22, px * 3), 255)
    ImageDraw.Draw(img).text((px, px // 2), "the quick brown fox jumps over a lazy dog", font=font, fill=0)
    m = measure_line(np.asarray(img), [(0, 0), (img.width, 0), (img.width, img.height), (0, img.height)])
    true_xh = font_catalog.x_height_em(font_path) * px
    return m.x_height / true_xh if m is not None and true_xh > 0 else 1.0


def size_for_x_height(x_height_px: float, font_path: str) -> float:
    return x_height_px / x_band_bias(font_path) / font_catalog.x_height_em(font_path)


# lowercase letters that stay inside the x-height band
_X_ONLY = frozenset("acegimnopqrsuvwxyz")


def measures_caps(line: Line) -> bool:
    """Whether a line's measured x-height is really its cap height: text of
    capitals and digits has no x-band, so its densest band is the caps."""
    chars = [c for c in line.text if c.isalnum()]
    if not chars:
        return False
    short = sum(c in _X_ONLY for c in chars) / len(chars)
    return short < CAPS_MAX_SHORT_FRAC and line.cap_top < CAPS_MAX_RISE * line.x_height


def line_x_height(line: Line, font_path: str | None = None) -> float:
    """The line's x-height, converted from its cap height (in the given
    font's proportions) when that is what was measured."""
    if not measures_caps(line):
        return line.x_height
    ratio = (font_catalog.x_height_em(font_path) / font_catalog.cap_height_em(font_path)
             if font_path else XH_PER_CAP)
    return line.x_height * ratio


def lines_x_height(lines: list[Line], font_path: str | None = None) -> float:
    hs = [line_x_height(l, font_path) for l in lines if len(l.text.strip()) >= 3]
    return float(np.median(hs or [line_x_height(l, font_path) for l in lines]))


# ---------- size clusters ----------

def _block_x_height(block: Block) -> float | None:
    hs = [line_x_height(l) for l in block.lines if len(l.text.strip()) >= 3 and not l.drop_cap]
    return float(np.median(hs)) if hs else None


def _reliable(block: Block) -> bool:
    lines = [l for l in block.lines if not l.drop_cap]
    return len(lines) >= 2 or sum(len(l.text) for l in lines) >= RELIABLE_CHARS


def cluster_by_x_height(blocks: list[Block], ratio: float = SIZE_CLUSTER_RATIO) -> list[list[Block]]:
    """Size clusters of one role. Short one-line blocks (capitals and digits
    distort their x-height) join the nearest cluster of the longer ones."""
    measured = sorted(((xh, b) for b in blocks if (xh := _block_x_height(b))), key=lambda t: t[0])
    solid = [t for t in measured if _reliable(t[1])]
    noisy = [t for t in measured if not _reliable(t[1])] if solid else []
    if not solid:
        solid, ratio = measured, NOISY_JOIN_RATIO
    clusters: list[list[tuple[float, Block]]] = []
    for xh, b in solid:
        if clusters and xh / clusters[-1][0][0] <= ratio:
            clusters[-1].append((xh, b))
        else:
            clusters.append([(xh, b)])
    for xh, b in noisy:
        med = [float(np.median([v for v, _ in c])) for c in clusters]
        k = int(np.argmin([abs(math.log(xh / m)) for m in med]))
        if max(xh, med[k]) / min(xh, med[k]) <= NOISY_JOIN_RATIO:
            clusters[k].append((xh, b))
        else:
            clusters.append([(xh, b)])
    return [[b for _, b in c] for c in clusters]


# ---------- family + width ----------

def _width_ratios(lines: list[Line], face: font_catalog.Face, size_px: float) -> list[float]:
    out = []
    for l in lines:
        text = l.text.strip()
        if len(text.replace(" ", "")) < MIN_WIDTH_CHARS or l.length <= 0:
            continue
        natural = font_catalog.string_width(text, face.name, size_px)
        if natural > 0:
            out.append(l.length / natural)
    return out


def fit_family(lines: list[Line], category: str, bold: bool, italic: bool) -> tuple[str, float, float] | None:
    """(family, size_px, width ratio) of the bundled family in `category`
    whose natural widths best match these lines, or None without evidence."""
    candidates = []
    for family in font_catalog.families_in(category):
        if not font_catalog.available(family):
            continue
        face = font_catalog.face(family.key, bold, italic)
        size = size_for_x_height(lines_x_height(lines, face.path), face.path)
        ratios = _width_ratios(lines, face, size)
        if not ratios:
            return None
        ratio = float(np.percentile(ratios, WIDTH_PERCENTILE))
        candidates.append((abs(math.log(ratio)), family.key, size, ratio))
    if not candidates:
        return None
    candidates.sort()
    best = candidates[0]
    default = next((c for c in candidates if c[1] == font_catalog.DEFAULT_FAMILY[category]), None)
    if default is not None and default[0] - best[0] <= FAMILY_TIE_LOG:
        best = default
    logger.debug("family fit (%s): %s", category,
                 ", ".join(f"{k}={r:.3f}" for _, k, _, r in candidates))
    return best[1], best[2], best[3]


def _chars(lines: list[Line]) -> int:
    return sum(len(l.text.replace(" ", "")) for l in lines)


def assign_styles(page: Page, image: Image.Image | None) -> None:
    """Give every text block with lines its role's (cluster's) TextStyle."""
    blocks = [b for b in page.text_blocks() if b.lines]
    if not blocks:
        return
    traits = classify_roles([(b.role, b.bbox, b.content) for b in blocks], image)

    # the body family: fitted on the role cluster with the most text
    plans = []
    for role in {b.role for b in blocks}:
        t = traits[role]
        for cluster in cluster_by_x_height([b for b in blocks if b.role == role]):
            lines = [l for b in cluster for l in b.lines if not l.drop_cap]
            if not lines:
                continue
            plans.append((role, t, cluster, lines))
    plans.sort(key=lambda p: -_chars(p[3]))
    body_family: dict[str, str] = {}
    for role, t, cluster, lines in plans:
        fit = None
        if _chars(lines) >= MIN_FAMILY_EVIDENCE_CHARS or t.category not in body_family:
            fit = fit_family(lines, t.category, t.bold, t.italic)
        if fit is None or (_chars(lines) < MIN_FAMILY_EVIDENCE_CHARS and t.category in body_family):
            family = body_family.get(t.category, font_catalog.DEFAULT_FAMILY[t.category])
            face = font_catalog.face(family, t.bold, t.italic)
            size = size_for_x_height(lines_x_height(lines, face.path), face.path)
            ratios = _width_ratios(lines, face, size)
            ratio = float(np.percentile(ratios, WIDTH_PERCENTILE)) if ratios else 1.0
        else:
            family, size, ratio = fit
        body_family.setdefault(t.category, family)
        style = TextStyle(font_family=family, category=t.category, bold=t.bold, italic=t.italic,
                          size_px=size, width_scale=float(np.clip(ratio, *WIDTH_SCALE_RANGE)))
        for b in cluster:
            b.style = style
            for l in b.lines:
                if l.drop_cap:
                    face = font_catalog.face(family, t.bold, t.italic)
                    l.size_px = l.cap_top / font_catalog.cap_height_em(face.path)
        logger.debug("style %s x%d blocks: %s %.1fpx width %.3f (raw %.3f)", role, len(cluster),
                     family, size, style.width_scale, ratio)

    # blocks too short to measure (e.g. 1-2 letter lines) take their role's
    # nearest-sized style, else a default one sized from their own lines
    for b in blocks:
        if b.style is not None:
            continue
        xh = float(np.median([line_x_height(l) for l in b.lines]))
        peers = [p.style for p in blocks if p.role == b.role and p.style is not None]
        if peers:
            b.style = min(peers, key=lambda s: abs(math.log(max(s.size_px, 1e-6) / max(xh * 2, 1e-6))))
            continue
        t = traits[b.role]
        family = body_family.get(t.category, font_catalog.DEFAULT_FAMILY[t.category])
        face = font_catalog.face(family, t.bold, t.italic)
        b.style = TextStyle(font_family=family, category=t.category, bold=t.bold, italic=t.italic,
                            size_px=size_for_x_height(lines_x_height(b.lines, face.path), face.path))

    harmonize_styles(blocks)


# ---------- page-wide harmony ----------

def _style_groups(blocks: list[Block]) -> list[tuple[TextStyle, list[Block], int]]:
    """Distinct styles in use, with their blocks and amount of text."""
    groups: dict[int, tuple[TextStyle, list[Block]]] = {}
    for b in blocks:
        if b.style is not None:
            groups.setdefault(id(b.style), (b.style, []))[1].append(b)
    return [(s, bs, _chars([l for b in bs for l in b.lines if not l.drop_cap]))
            for s, bs in groups.values()]


def _weighted_median(values: list[float], weights: list[float]) -> float:
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights, dtype=float)[order]
    return float(v[np.searchsorted(np.cumsum(w), w.sum() / 2)])


def harmonize_styles(blocks: list[Block]) -> int:
    """Make nearby styles identical: a document uses a handful of styles,
    and measurement noise must not turn one into several.

      1. styles with too little text to judge slant take it from the
         nearest-sized established style;
      2. one typeface per category for all text up to display size
         (large titles may use their own face);
      3. sizes within SIZE_MERGE_RATIO of each other (same weight/slant)
         become one size, weighted by amount of text;
    and every merged style's glyph width is refitted to all its lines.
    Returns the number of distinct styles left."""
    groups = _style_groups(blocks)
    if len(groups) < 2:
        return len(groups)
    strong = [(s, n) for s, _, n in groups if n >= MIN_TRAIT_CHARS]
    for s, _, n in groups:
        if n >= MIN_TRAIT_CHARS or not strong:
            continue
        near, _ = min(strong, key=lambda t: abs(math.log(t[0].size_px / s.size_px)))
        if abs(math.log(near.size_px / s.size_px)) <= math.log(TRAIT_ADOPT_RATIO):
            # slant needs many glyphs to measure; stem weight doesn't
            s.italic = near.italic

    # the body face of each category: the one carrying the most text
    by_family: dict[tuple[str, str], int] = {}
    for s, _, n in groups:
        by_family[(s.category, s.font_family)] = by_family.get((s.category, s.font_family), 0) + n
    body_face = {}
    for (cat, fam), n in sorted(by_family.items(), key=lambda t: -t[1]):
        body_face.setdefault(cat, fam)
    body_size = {}
    for s, _, n in sorted(groups, key=lambda t: -t[2]):
        body_size.setdefault(s.category, s.size_px)
    for s, _, _ in groups:
        if s.size_px <= DISPLAY_SIZE_RATIO * body_size[s.category]:
            s.font_family = body_face[s.category]

    # merge near-equal sizes of the same look
    merged: list[list[tuple[TextStyle, list[Block], int]]] = []
    for g in sorted(groups, key=lambda t: (t[0].category, t[0].font_family, t[0].bold, t[0].italic,
                                             t[0].size_px)):
        s = g[0]
        if merged:
            head = merged[-1][0][0]
            if (head.category, head.font_family, head.bold, head.italic) == \
                    (s.category, s.font_family, s.bold, s.italic) and \
                    s.size_px / head.size_px <= SIZE_MERGE_RATIO:
                merged[-1].append(g)
                continue
        merged.append([g])

    for cluster in merged:
        base = cluster[0][0]
        members = [b for _, bs, _ in cluster for b in bs]
        weights = [max(n, 1) for _, _, n in cluster]
        size = _weighted_median([s.size_px for s, _, _ in cluster], weights)
        face = font_catalog.face(base.font_family, base.bold, base.italic)
        lines = [l for b in members for l in b.lines if not l.drop_cap]
        ratios = _width_ratios(lines, face, size)
        width = float(np.percentile(ratios, WIDTH_PERCENTILE)) if ratios else \
            _weighted_median([s.width_scale for s, _, _ in cluster], weights)
        style = TextStyle(font_family=base.font_family, category=base.category, bold=base.bold,
                          italic=base.italic, size_px=size,
                          width_scale=float(np.clip(width, *WIDTH_SCALE_RANGE)))
        for b in members:
            b.style = style
            for l in b.lines:
                if l.drop_cap:
                    l.size_px = l.cap_top / font_catalog.cap_height_em(face.path)
    logger.debug("harmonized %d styles into %d", len(groups), len(merged))
    return len(merged)
