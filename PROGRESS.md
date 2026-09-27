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
  class = ink), returns `None` if there's no clear ink signal;
  `ink_maps()` also returns a soft (anti-aliased) ink map. Also has
  `_row_ink_bands()` (used by `bold.py`), numpy-only geometry helpers
  (`estimate_skew`, `rotate_mask`, `close_horizontal`, `component_boxes`),
  and `measure_text_lines()` -> `TextLineStats(n_lines, x_height, pitch)`
  (see "Font type + size rewrite" below).
- **`docir/bold.py`** -- `measure_stroke_ratio()` (stroke-width / line-height
  from pixels) + `classify_bold_by_label()` (per-label median vs.
  document-wide median * `BOLD_RELATIVE_FACTOR = 1.15`).
- **`docir/italic.py`** -- shear-search slant detection. Tries angles -24..24
  deg in 3 deg steps, shifts rows by `tan(angle) * (row - center)`, keeps
  the angle that maximizes column-density "peakiness". `measure_slant_angle()`
  + `classify_italic_by_label()` (threshold `ITALIC_ANGLE_THRESHOLD_DEG = 6.0`
  on `abs(angle)`).
- **`docir/fonts.py`** -- local font catalog (`discover_catalog()`, every
  font file categorized by rendering its own glyphs) + word-level
  render-and-compare matching: `collect_word_samples()` (aligned
  observed-word/OCR-word pairs from one block), `classify_word_samples()`
  (pooled -> `(category, margin)`), and `match_style_category()` (one-block
  wrapper). See "Font type + size rewrite" below.
- **`docir/render_pdf.py`** -- the actual PDF renderer. Column detection via
  gap-clustering on x1 (`_cluster_columns`, reused pattern for
  `_cluster_label_sizes`), full-width header detection, XML-escaping,
  per-column downward-growing cursors so text never overlaps/gets dropped,
  cropping image/figure/chart/seal blocks directly from the source image.
  Per block, bold/italic are measured from pixels and aggregated per-label;
  font family is decided from pooled matched words per label/document, and
  font size from measured x-height + line count (see "Font type + size
  rewrite" below).
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

## Config-driven debug logging + real overlap root cause (this session)

Added `docir/config.py` (`load_config()`, `is_debug_logging_enabled()`) and
`docir_config.toml` at repo root (`[logging] debug = true|false`), read via
stdlib `tomllib` (no new dependency). `cli.py`'s `logging.basicConfig` now
uses `DEBUG` when the config flag is on, `INFO` otherwise. First debug log
added: `render_pdf.py` logs which standard ratio a page snapped to (or that
it was distinctive and kept raw).

This led to actually diagnosing the "overlapping text" complaint on
`vl1_6_3.jpeg` (previously unconfirmed) by re-running OCR and inspecting
the **raw** PaddleOCR-VL output (before our schema discards
`block_polygon_points`). Two distinct causes found:

1. **Rotated-title bbox inflation (FIXED)**: some `paragraph_title` blocks
   are detected as slightly rotated quads (e.g. `poly=[[31,36],[178,91],
   [168,118],[21,63]]`), but `block_bbox` is just the axis-aligned min/max
   rectangle around that rotated quad -- which is always taller than the
   rotated rectangle itself. That inflated height was pushing titles down
   into the image/text block below them. Fixed by:
   - `schemas.py`: `Block` now has an optional `polygon` field, populated
     from `block_polygon_points` when present (`None` for legacy JSON).
   - `render_pdf.py`: new `_polygon_extent()` (true width/height of a
     4-point quad, measured along its own edges) and
     `_deskew_bbox_height()` (shrinks bbox height to the polygon's true
     height, keeping the top edge fixed, only when the polygon reveals
     genuine inflation). Applied once when blocks are built in
     `render_page_to_pdf()`, so everything downstream (columns, font
     fitting, positioning) sees the corrected bbox. Multi-point polygons
     (legitimate multi-line titles, e.g. a 24-point wrapped-title hull) are
     left untouched -- only simple 4-point rotated quads are affected.
     Verified on `vl1_6_3.jpeg`'s raw OCR output: rotated titles shrank
     from e.g. 74px -> 29px; the genuinely 2-line "Portrait of an Unknown
     Woman" title (24-point polygon) was correctly left at its original
     74px. Visual overlay confirms titles no longer balloon into the block
     below.
   - `output/vl1_6_3_deskewed.pdf` is a verification render with this fix.

Note (FIXED, see below): `output/vl1_6_3_res.json` (produced via `docir
analyze --json`) previously used our own clean `PageLayout` schema
(`blocks`/`bbox`), which was *not* the same shape as
`output/demo_img_res.json`/`output/vl1_6_10_res.json` in this repo (those
were saved directly from Paddle's raw `res.json`, with
`parsing_res_list`/`block_bbox`/`block_polygon_points`). `PageLayout.from_paddle_dict()`
only understands the raw `parsing_res_list` shape -- feeding it our own
clean-schema JSON silently produced zero blocks.

**Fix:** `docir analyze --json <path>` now saves PaddleOCR-VL's own raw
result to `<path>` unchanged (matching Paddle's own native `_res.json`
convention -- the same shape as the pre-existing `vl1_6_10_res.json`/
`demo_img_res.json` fixtures), so it's always loadable via
`from_paddle_dict()` and never needs OCR to be re-run just to get the raw
data again. Our own `PageLayout` schema is now saved to a separate,
automatically-derived path instead (`_res` suffix swapped for `_schema`,
e.g. `vl1_6_3_res.json` -> `vl1_6_3_schema.json`), so neither file
overwrites the other. Implemented via `PaddleOcrVlEngine.last_raw` (the raw
dict from the most recent `analyze()` call) and
`pipeline._schema_json_path()`; falls back to the old single-file/schema-
only behavior for engines that don't expose `last_raw` (e.g. `FakeOcrEngine`
in tests).

Also added `docir/tools/` (`overlay_bbox.py`, `overlay_polygon.py`, shared
`_overlay_common.py`) -- standalone read-only diagnostics that draw either
just `bbox` (red) or just `polygon` (blue, with a gray fallback outline for
blocks that have no polygon) over the source image, for visually comparing
what the detector actually reported. Usage:
```bash
.docir_env/bin/python -m docir.tools.overlay_bbox <json> <source_image> <out.png>
.docir_env/bin/python -m docir.tools.overlay_polygon <json> <source_image> <out.png>
```

## Font catalog: curated list -> full-directory scan (this session)

> Superseded: filename-based category guessing described here was replaced
> by rendered-glyph categorization -- see "Font type + size rewrite".


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

`48/48` passing as of the last change (`.docir_env/bin/python -m pytest -q`).
Covers `test_bold.py`, `test_fonts.py`, `test_font_size.py`, `test_ink.py`,
`test_italic.py`, `test_page_ratios.py`, `test_pipeline.py`, `test_schemas.py`.

## Font type + size rewrite (this session)

**Ask:** "check and correct both font type and font size -- make it almost
perfect". Baseline: every block on every fixture was classified
`sans_serif` (2 of 4 fixtures wrong), and sizes came from squeezing text
into (often inflated) bboxes, so body text was visibly too large and
wrapped differently from the source.

**Why font type was wrong (root causes found):**
1. Catalog pollution -- ~60 of 120 fonts were miscategorized by filename
   guessing (script/handwriting faces like Zapfino/Chalkduster/Marker Felt
   as sans, "Microsoft Sans Serif" as serif, real serifs like Iowan Old
   Style/Rockwell/New York as sans). These dominated the "best single
   match" ranking.
2. The whole block's text was rendered as one line and compared to only
   the observed first line -- different glyphs, different
   ascender/descender extent -- with strict pixel IoU.
3. Crisp high-res renders were compared to blurry low-res scans, which
   penalizes serif fonts (their detail doesn't survive the scan).

**Font type now (`fonts.py`):**
- `_classify_font_file()` renders I/l/x/H/a/A: serif if "I" has top+bottom
  feet and "l" has a foot; monospace from i/m/W advances; rejected if
  x/H ratio is atypical (caps/script), "a" looks like "A" (small caps),
  stems slant (script/italic) or are rough (distressed). Bold-only files
  are skipped (weight is handled separately).
- Per block: deskew, cut into word blobs (`close_horizontal` +
  `component_boxes`, chained into curved lines), align blobs to OCR words
  with a monotonic DP scored against 4 reference fonts, then compare each
  word to every catalog font -- renders degraded through the observed
  word's native pixel size and estimated blur -- by band-pass NCC with a
  stroke-weight search.
- `render_pdf.py` pools matched words per label (up to
  `FONT_POOL_WORDS_PER_LABEL`) and document-wide; a category scores the
  mean of its best 3 fonts. A label departs from the document's family only
  with `>= MIN_LABEL_FONT_WORDS` words and margin `>= 0.05`.
- Verified: vl1_6_10 sans (margin 0.27), demo_img sans (0.35), vl1_6_3
  serif (0.02 -- weakest; its DejaVu-Serif-like face has no close serif on
  macOS, Verdana-like sans compete), vl1_6_4 serif (0.04).

**Font size now (`render_pdf.py`):**
- `measure_text_lines()` measures line count, x-height and line pitch per
  block, using fixed-width vertical strips each deskewed independently
  (handles tilted *and* curved/warped pages; a short last line only needs
  to reach the left strips); robust-peak thresholds ignore underlines and
  decorative glyphs; slivers of neighboring text and drop caps are filtered.
- Size = measured x-height / the substitute font's AFM x-height
  (`X_HEIGHT_EM`), plus the size range that reproduces the measured line
  count at the block's width (`_line_count_size_range`, bisection; hard
  line breaks raise the minimum achievable count).
- Per label, sizes are clustered per visual role and one shared size is
  picked that keeps the most blocks' line counts (`_consensus_cluster_size`).
- The original font is often narrower/wider than Helvetica/Times, so
  instead of trading x-height for line breaks, text is drawn at the
  x-height size with horizontal glyph scaling (`MIN_HSCALE..MAX_HSCALE`,
  canvas `scale(hscale, 1)`), which keeps identical line breaks.
  Leading = measured pitch (floored at `MIN_LEADING_RATIO * size`);
  single-line blocks cap leading at their box height.
- Verified (rendered vs measured line counts): vl1_6_10 15/17, demo_img
  30/32, vl1_6_3 9/11, vl1_6_4 61/62, visually checked against sources.

**Follow-up fix: inconsistent sizes within a role (demo_img).** Per-block
horizontal scales and per-block size clamping made same-role paragraphs
look different sizes. Now every visual-role cluster gets exactly one point
size (median x-height size of its multi-line blocks) and one glyph width:
each block contributes the width interval that keeps its line breaks
(single-line blocks: box width / natural text width), the document-wide
width is the value inside the most intervals (`_best_shared_scale`), and
each role picks the value inside most of *its* intervals, tie-broken toward
the document's. A block that would wrap onto extra lines only condenses a
little more -- never a different size. `measure_text_lines()` also levels
the whole block before the per-strip pass (strongly rotated titles) and
anchors line height on the whole-block profile (a picture fragment clipped
into a title's bbox no longer inflates its x-height). demo_img: all body
text 11.89pt x0.73, footnote x0.74, section title x0.73; line matches
27/32 (vl1_6_10 16/17, vl1_6_3 9/11, vl1_6_4 61/62).
- Debug logging (config flag) prints per-block font, size, hscale, leading,
  rendered/measured lines, and the per-label font evidence.

**Known remaining gaps:** superscript prefixes like "[PDF]" can count as an
extra title line; very narrow/condensed sources hit `MIN_HSCALE` and fall
back to a smaller size; references render without hanging indents.

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
