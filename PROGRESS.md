# DocIR Progress Log

> Living status doc for picking up work across sessions. Update this file
> whenever a non-trivial change lands. `docstyle_build_plan.md` is the
> original aspirational design spec (OCR ingestion, web UI, milestones
> M0-M12) and has **not** been kept in sync with the actual codebase -- this
> file describes what actually exists today in `docir/`.

## What this project actually does (current state)

Takes a PaddleOCR-VL-style layout-detection JSON (blocks with `label`,
`content`, `bbox`) plus the original source page image, and renders a
geometry-preserving PDF: same columns, same approximate box positions, and
per-block text styling (bold / italic / serif-sans-serif-monospace family /
font size) inferred **from the source image's pixels**, never guessed from
the label alone.

Entry points:
- `docir render <layout.json> [output_pdf] [--source-image path]` -- renders
  an existing layout JSON (this is the path used for all real testing so
  far).
- `docir analyze <image> [output_pdf] [--json path]` -- full pipeline
  (OCR/layout detection via `ocr.py` + `pipeline.py`) then render. Less
  exercised than `render`; `ocr.py`/`pipeline.py` were not part of this
  session's work.

## Module map

- **`docir/ink.py`** -- shared binarization. `binarize_region(source_image,
  bbox)` crops + Otsu-thresholds a region into a boolean ink mask (minority
  class = ink), returns `None` if there's no clear ink signal. Also has
  `_row_ink_bands()` (density-threshold line-band detection -- good for
  line-height/bold estimation, **not** for font-shape matching since it
  clips ascenders/descenders).
- **`docir/bold.py`** -- `measure_stroke_ratio()` (stroke-width / line-height
  from pixels) + `classify_bold_by_label()` (per-label median vs.
  document-wide median * `BOLD_RELATIVE_FACTOR = 1.15`).
- **`docir/italic.py`** -- shear-search slant detection. Tries angles -24..24
  deg in 3 deg steps, shifts rows by `tan(angle) * (row - center)`, keeps
  the angle that maximizes column-density "peakiness". `measure_slant_angle()`
  + `classify_italic_by_label()` (threshold `ITALIC_ANGLE_THRESHOLD_DEG = 6.0`
  on `abs(angle)`).
- **`docir/fonts.py`** -- local font-catalog discovery (`discover_catalog()`)
  + render-and-compare style-category matching (`match_style_category()` ->
  `(category, score)` where category is `serif` / `sans_serif` / `monospace`
  / `unknown`). Discovery is now a full-directory scan, not just a fixed
  curated list -- see "Font catalog: curated list -> full-directory scan"
  below. Key calibration constants: `MIN_MATCH_TEXT_LEN = 3`,
  `MIN_MATCH_SCORE = 0.12` (pixel IoU matching is inherently noisy -- even a
  perfect same-font match rarely exceeds ~0.2 IoU, so this only filters
  near-zero signal and otherwise relies on relative best-of-catalog ranking,
  not absolute confidence). Candidates are rendered at the **same pixel
  height as the observed text** (matching hinting behavior) and compared
  only on the first physical text line via `_first_line_mask()` (any-ink-row
  + gap-run merge, preserves full ascender-to-descender extent --
  deliberately different from `ink._row_ink_bands()`).
- **`docir/render_pdf.py`** -- the actual PDF renderer. Column detection via
  gap-clustering on x1 (`_cluster_columns`, reused pattern for
  `_cluster_label_sizes`), full-width header detection, XML-escaping,
  per-column downward-growing cursors so text never overlaps/gets dropped,
  cropping image/figure/chart/seal blocks directly from the source image.
  Per block, bold/italic/font-category are measured from pixels and
  aggregated per-label (see "Font-style consensus" below), then a font size
  is fit per block and collapsed to per-visual-role clusters.
- **`docir/schemas.py`** -- `PageLayout` / block dataclasses,
  `PageLayout.from_paddle_dict()` loader.
- **`docir/cli.py`** -- argparse entry point (`analyze` / `render`).
- **`docir/pipeline.py`**, **`docir/ocr.py`** -- OCR/layout-detection
  pipeline for the `analyze` path; not touched this session.

## Font-style consensus logic (the tricky part, `render_pdf.py`)

Real documents reuse one layout *label* (e.g. `paragraph_title`) for
genuinely different visual roles (a short real title vs. a large pull-quote
sidebar), and per-block pixel measurements are individually noisy
(especially font-category matching, scores ~0.12-0.25 even for confident
matches). Two independent problems were found and fixed this session:

### 1. Font **size** consensus (bimodal label fix)

**Symptom:** user reported some rendered font sizes still didn't visually
match after the initial font-family/italic work landed.

**Root cause:** the original code fit each block's own best size, then
collapsed all blocks sharing a label to one `statistics.median` size. When a
label covers two visually distinct roles (e.g. `paragraph_title` = normal
short titles ~9pt *and* a pull-quote sidebar ~15.7-18.4pt in the
`vl1_6_10` fixture), the median (~12.5pt) fit neither.

**Fix:** `_cluster_label_sizes(id_size_pairs)` -- relative-gap 1D clustering
(same technique as `_cluster_columns`, tunable `SIZE_CLUSTER_GAP_RATIO =
1.5`) splits a label's per-block fitted sizes into separate clusters when
consecutive sorted sizes differ by more than the ratio, then each cluster
gets its own median. Verified on `vl1_6_10_res.json`: `paragraph_title`
correctly splits into a ~9.2pt cluster (real titles) and a ~17.0pt cluster
(pull-quote), while `figure_title`'s legitimately-uniform variance (7.1 vs
10.0) stays merged as one ~8.6pt cluster.

### 2. Font **category** consensus (small-sample tie-break fix)

**Symptom:** user reported `demo_img`'s titles/headings rendered serif when
they were visually sans-serif.

**Root cause:** per-label font-category consensus used
`statistics.mode()` on a list of per-block category guesses. A label with
very few matched blocks (titles/footers are often only 1-2 blocks) can land
exactly tied (e.g. `doc_title` in `demo_img`: 2 samples, one scored `serif`
0.165 and the other `sans_serif` 0.16 -- a near-exact tie within pixel-match
noise), and `mode()` breaks ties by insertion order, arbitrarily picking
whichever category was encountered first.

**Fix:** switched to score-weighted voting (sum `match_style_category`'s
confidence score per category, not just count) *and* added
`MIN_FONT_CATEGORY_SAMPLES = 3`: a label with fewer than 3 matched samples
now defers to the document-wide, score-weighted-winner category instead of
trusting its own thin/noisy sample (mirrors the existing
document-relative-baseline philosophy in `bold.py`/`italic.py`). Verified:
`demo_img`'s `doc_title` (tied 2-sample case) now resolves to `sans_serif`,
matching the document's dominant `text` signal (18 samples, sans_serif
sum ~1.95 vs serif ~0.97).

## Key design lessons (apply these if extending pixel-based detection further)

1. **Font rendering is size-dependent (hinting)** -- always render/compare
   candidate glyphs at the *same* pixel size as the observed text; a fixed
   large render size then downscaled produces genuinely different
   width/height aspect ratios (confirmed empirically: same font+text at
   size 40 vs 200 had a 26% aspect-ratio discrepancy from hinting alone).
2. **Density-threshold line detection is wrong for shape matching** --
   `ink._row_ink_bands()` (density-threshold) is fine for line-height/bold
   estimation but clips ascenders/descenders, destroying serif-distinguishing
   shape info; `fonts._first_line_mask()` (any-ink-row + gap-run merge) is
   the correct tool for font-shape comparison.
3. **A single global threshold/ratio can't separate bimodal distributions**
   -- when one label may represent multiple visual roles, cluster
   (relative-gap clustering) rather than compare against one group-wide
   summary statistic.
4. **Small samples + near-threshold scores need a fallback anchor, not a
   raw vote** -- ties/thin samples should defer to a larger, more reliable
   aggregate (document-wide consensus) rather than trusting `mode()`/first-
   encountered ordering.
5. **Absolute confidence thresholds must be empirically calibrated** against
   realistic baselines for pixel-comparison metrics -- even "perfect"
   matches can score low; prefer relative/document-local ranking over
   assumed-a-priori cutoffs.

## Standard page-ratio canvas snapping (this session)

New module `docir/page_ratios.py`: a catalog of ~24 well-known paper/photo/
screen aspect ratios (ISO 216, US Letter/Legal/Ledger/Executive/Statement/
Folio/Quarto/Foolscap, index cards, photo prints, business/credit cards,
square, 4:3, 3:2, 5:4, 16:9, 16:10, 21:9, panoramic) plus
`closest_standard_ratio(width, height, tolerance=0.05)` which checks both
portrait and landscape orientation of every entry and returns the closest
match within a relative tolerance, or `None` if the shape is distinctively
its own.

Wired into `render_pdf.py`'s `render_page_to_pdf()`: the page canvas width
stays fixed at `TARGET_PAGE_WIDTH_PT`, but the page height now snaps to the
nearest standard ratio (`STANDARD_RATIO_TOLERANCE = 0.05`) instead of being
a raw uniform scale of the source image. Content itself is never
stretched/distorted -- only the page height is adjusted, and only upward:
`page_h = max(content_h, standard_h)`, so a snapped standard height can
never be smaller than what the content actually needs (guards against
clipping). Verified against real fixtures: `vl1_6_10.png` (2160x3840, exact
16:9) snaps its page to 612x1088pt; `demo_img.jpeg` (984x1280, ~US Letter
but not quite) keeps its raw content height (796.1pt) because the standard
US Letter height (792.0pt) would have been slightly shorter than the actual
content.

Added `tests/test_page_ratios.py` (6 tests, pure math, no fixtures needed).

Note: this was implemented in response to a user report of "overlapping
text" that they attributed to aspect-ratio skew. Investigation showed the
pre-existing scale computation was already a uniform (non-distorting)
transform, so this feature addresses the user's explicit ask (snap to
standard ratios) but may NOT be the actual root cause of any real text
overlap -- that would need separate diagnosis with real bbox data (e.g. for
`vl1_6_3.jpeg`, which has no saved layout JSON yet).

## Font catalog: curated list -> full-directory scan (this session)

**Ask:** "include all whatever we can render" -- broaden the font catalog
beyond the original curated 11-font list.

**What changed in `docir/fonts.py`:**
1. `_KNOWN_FONTS` grew from 11 to ~45 curated `(family, category, filename)`
   entries (more serif/sans/monospace system fonts across macOS/Linux/
   Windows naming conventions, incl. filename-casing variants).
2. `_CANDIDATE_DIRS` grew to also check `/usr/local/share/fonts`,
   `~/.fonts`, `~/.local/share/fonts`, and `C:/Windows/Fonts`.
3. `discover_catalog()` no longer *only* looks up the curated filenames --
   it now also opportunistically lists every `.ttf`/`.ttc`/`.otf` file in
   each candidate directory and adds any not already covered, guessing its
   category from filename keywords (`_guess_category()`) when it isn't in
   the curated list.
4. To keep this from re-adding every bold/italic/condensed variant of a
   family as a separate (redundant) candidate, filenames are collapsed to a
   `_base_family_key()` (strips weight/style suffixes like Bold/Italic/
   Narrow/Condensed/W1-W9/etc., repeatedly) and only the first file per key
   is kept.
5. **Performance problem found and fixed:** an unqualified full scan on
   macOS pulled in ~356 candidates (dominated by ~90 `NotoSans*`/`NotoSerif*`
   per-script files, dozens of CJK/Indic/Arabic/Hebrew/Armenian/Braille/
   Emoji/Symbol/STIX fonts that can't usefully discriminate Latin
   serif/sans/mono shape anyway) and blew up render time to ~90s for a
   ~32-block document. Added `_SCAN_EXCLUDE_KEYWORDS` (a keyword blocklist:
   `noto`, CJK families, Arabic/Hebrew/Armenian/Indic-script families,
   braille/emoji/symbol/dingbat/STIX) + `_is_scannable()`, applied only to
   the opportunistic scan (never to the curated list) -- cut the real
   catalog down to ~120 candidates on this machine, and full-document
   render time down to ~12-30s (still slower than the original ~25-font
   catalog, but a deliberate breadth-for-speed tradeoff the user asked for).
6. Also discovered and fixed a **pre-existing bug**: `_render_text_mask()`
   called an undefined `_load_font()` (referenced but never defined) --
   added it as an `lru_cache`-wrapped `ImageFont.truetype()` loader. This
   didn't recover much speed on its own (per-block render heights vary too
   much for cache hits to be common) but is a real correctness fix
   independent of the catalog-size work, and helps when many blocks happen
   to share a font size.
7. Catalog dedup is now by filesystem identity (`_file_identity()` ->
   `(st_dev, st_ino)`) instead of raw path string, so case-insensitive
   filesystem duplicates (e.g. `Georgia.ttf` vs `georgia.ttf` resolving to
   the same inode on macOS) don't produce two redundant match candidates.

**Verified:** `discover_catalog()` finds ~120 fonts on this dev machine
(93 sans_serif / 20 serif / 7 monospace); `26/26` tests still pass; both
fixture PDFs regenerate successfully (`vl1_6_10` ~12s, `demo_img` ~30s).

**If this needs to be faster later:** the main remaining lever is
`MIN_FONT_CATEGORY_SAMPLES`-style early-exit (stop scanning candidates once
a very high-confidence match is found instead of always scanning the full
catalog), or trimming `_SCAN_EXCLUDE_KEYWORDS` further / capping the
opportunistic scan to a max candidate count. Not done this session since
render times were judged acceptable for a batch/CLI tool.

## Test status

`26/26` passing as of the last change (`.docir_env/bin/python -m pytest -q`).
Covers `test_bold.py`, `test_fonts.py`, `test_italic.py`, `test_pipeline.py`,
`test_schemas.py`.

## Real fixtures used for empirical validation

- `output/vl1_6_10_res.json` + `vl1_6_10.png` -- multi-column business-style
  document (labels: `header`, `paragraph_title`, `text`, `figure_title`,
  `footer`, `number`, `image`; 18 blocks). Contains the bimodal
  `paragraph_title` pull-quote case used to validate size clustering.
- `output/demo_img_res.json` + `demo_img.jpeg` -- casual social-post-style
  image (labels include `doc_title`, `paragraph_title`, `text`,
  `vision_footnote`, `footer`). Used to validate the font-category
  small-sample tie-break fix.

Regenerate either with:
```bash
.docir_env/bin/python -m docir.cli render output/vl1_6_10_res.json output/vl1_6_10.pdf --source-image vl1_6_10.png
.docir_env/bin/python -m docir.cli render output/demo_img_res.json output/demo_img.pdf --source-image demo_img.jpeg
```

## Environment

Python 3.13.5 venv at `.docir_env` -- always use `.docir_env/bin/python`.
Key deps: Pillow (ImageFont/ImageDraw/Image), numpy, reportlab
(standard-14 fonts only, no embedding/licensing concerns). `fontTools` was
tried and explicitly removed -- filename-based hints on macOS system fonts
turned out to be sufficient, no TTC/variable-font parsing needed.

## Open items / possible next steps

- No known open bugs as of this writing; both font-size and font-category
  consensus issues raised by the user this session are fixed and verified
  against real fixtures.
- `docir/pipeline.py` / `docir/ocr.py` (the `analyze` path) were not
  exercised or modified this session -- if the `analyze` CLI command is
  reported broken, start there.
- `docstyle_build_plan.md` describes a much larger product surface (web UI,
  FastAPI service, PDF ingestion/editing, evaluation harness) that has not
  been started; treat it as aspirational/reference only unless a user
  explicitly asks to pick up one of its milestones.
- If font-category or font-size heuristics misfire again on a new fixture,
  the fastest diagnostic path is a throwaway script (see pattern used this
  session) that replays the same per-label aggregation loop from
  `render_pdf.py` and prints per-block scores/samples/consensus -- delete it
  after use, don't leave it in the repo root.
