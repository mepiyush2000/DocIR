# DocIR: Agent Handbook

> **Read this first.** This is the single source of truth for an AI agent (or human) picking up this
> project. It describes the goal, the architecture, how each stage works, the decisions and user
> preferences that shaped the code, the problems already hit (and how they were solved), the current
> scores, and the known open issues.
>
> Older docs in the repo are **historical**:
> - `docstyle_build_plan.md`: the original aspirational spec. It is not in sync with the code.
> - `RECONSTRUCTION_PLAN.md`: the six-phase plan this codebase was built from. All phases are implemented.
> - `PROGRESS.md`: an earlier progress log. It is partially outdated, and this file supersedes it.

---

## 1. Goal

Turn a **scanned or photographed document page** (JPEG/PNG) into a **near-identical, fully vector
PDF**:

- **Real, selectable text.** No image of text is pasted in. Pictures (photos, logos, QR codes, seals) are embedded as images.
- **Same geometry.** Every text line sits where it was printed, at the measured size, in a matching font category (serif, sans or monospace), weight, slant and colour.
- **Graphics.** Rules, table grids, shaded table bands and fill-in lines are drawn as vectors.
- **Levelled output.** Small scan tilts (≤ 5°) are levelled. Curved and photographed pages are flattened first.
- **Measured success.** A render-and-compare **ink F1 score** against the source image is the main metric.

Target documents range widely and must all work with the *same* code:
- magazine and article pages;
- books;
- Indian GST receipts and hospital bills (dense tables);
- rental agreements and e-stamp papers (security patterns, Devanagari);
- 1990s typewriter forms with fill-in blanks, checkboxes, underlines and sideways stamps.

---

## 2. Ground rules / user preferences (MUST follow)

These came from the user over many iterations. Violating them caused reverts.

1. **No document-specific or text-specific hard-coding.** Never key logic on particular words,
   strings, file names, or coordinates from a sample.
   - Tuning constants (ratios, thresholds) are fine, as named module-level constants with a one-line comment.
   - Logic must come from *measurements* (pixels, geometry, line lengths), not from what the text says.
2. **Keep everything modular.** One concern per module, small functions, no giant if-chains.
3. **Generalise, don't overfit.** Every change is checked on **all** pages (see §9). A fix for one page must not regress the others. Small dips (< ~0.01 F1) are tolerated when the visual result is clearly better.
4. **Focus on English documents, without breaking other scripts.** Devanagari is present in pk_rental and hospital receipts.
5. **Tables: simple and robust beats clever.**
   - Render the layout model's HTML grid with measured row and column positions.
   - Text must never overlap.
   - **Draw borders only if they are clearly dark in the original.** Faint or grey lines are ignored, and a full grid is drawn only when dark vertical rules exist.
6. **Straight text.** Small tilts are levelled.
7. **Consistent font sizes.** The same role and size class gets exactly one size per page, with no per-block jitter.
8. **Font sizes slightly smaller than measured.** Config `font_size_offset_pt = -1.0`, which the user explicitly chose.
9. **Page width is always 612 pt.** The height snaps to a standard ratio within 5%. The user said **no** to forcing true paper sizes.
10. **Coordinates are raw source-image pixels everywhere**, never normalised.
11. Comments: one short line, only for what the code can't show. Don't add markdown docs unless asked.
12. Ask before destructive actions (deleting files, git reset/push). The user deletes stale files themselves.

---

## 3. Environment & commands

**Environment**
- Python 3.13 venv `.docir_env`. Always run `.docir_env/bin/python -m ...`; the `docir` entry point also works when the venv is active.
- Core dependencies (pyproject): pydantic, reportlab, Pillow, numpy, opencv-contrib-python.
- Extras:
  - `ocr`: paddleocr ≥ 3.7, paddlepaddle, paddlex
  - `eval`: pypdfium2
  - `dev`: pytest, pypdfium2
- Fonts are bundled in `docir/assets/fonts/` and fetched by `scripts/fetch_fonts.sh`:
  - liberation (Sans/Serif/Mono)
  - liberation-narrow
  - dejavu (+ Condensed and Mono)
  - noto (Sans/Serif)
  - pt-sans-narrow
  - noto-devanagari

### CLI (`docir/cli.py`)
```
docir analyze <image> [out.pdf] [--json res.json] [--preprocess auto|off|force]
docir render  <res.json> [out.pdf] [--source-image img] [--mode lines|paragraph] [--score]
docir score   <res.json> <out.pdf> [--source-image img] [--diff d.png] [--report] [--worst]
docir lines   <res.json> [--source-image img] [--refresh]
docir flatten <image> [out] [--mode auto|force]
```
- `analyze` runs the full pipeline. It runs the models, which takes a few minutes per page on CPU, writes `<name>_res.json`, and also writes and caches line OCR to `<name>_lines.json`.
- Successful conversions log total, preprocessing, layout OCR, line OCR/JSON, and rendering times. These include lazy model initialization on the first page; compare subsequent pages to measure warm latency. The CLI and Streamlit use the same timing log.
- `render` re-renders from cached JSON. It is fast, and it's what you use when iterating on code.
- When `render` is given `--score`, it prints `page P=… R=… F1=… stray_ink=…`.

### Config (`docir_config.toml` / `docir/config.py`)
```toml
[logging]    debug = false
[preprocess] mode = "auto"            # auto | off | force
[lines]      enabled = true           # PP-OCRv5 line detection (cached)
[render]     mode = "lines"           # lines | paragraph (legacy)
             straighten_max_deg = 5.0
             font_size_offset_pt = -1.0
```

### Tests
- Run with `.docir_env/bin/python -m pytest -q`. Currently 105 tests pass, in about 75 s.
- Fixtures use `output/vl1_6_10_res.json` + `vl1_6_10.png` (`tests/conftest.py`).

### Re-render and score every page
```bash
for j in output/*_res.json; do n=$(basename $j _res.json); [ "$n" = "vl1_6_3" ] && continue
  src=$(.docir_env/bin/python -c "import json;print(json.load(open('$j'))['input_path'])")
  .docir_env/bin/python -m docir.cli render $j output/final/$n.pdf --source-image $src --score 2>&1 \
    | grep -E "^page|Error|Traceback" | head -1 | sed "s/^/$n: /"
done
```
`vl1_6_3` is skipped because its layout JSON belongs to the flattened image `vl1_6_3_flat.png`.

### Terminal quirks learned
- **Long inline `python -c` commands get mangled.** Write scratch scripts as `.tmp_*.py` and delete them afterwards.
- **New files sometimes flush late.** `sleep 1` before running a just-created script.
- **Editor buffers can clobber files.** Never `sed -i` a file that's open in the editor; the editor's buffer can overwrite it. This happened once and left a stray `pass` line in `table_layout.py`.
- **Use the browser tools to inspect images.** Open `file://…png?v=N`, bumping `N` to bust the cache, then take a screenshot.

---

## 4. Architecture

### 4.1 Data flow
```
image ──► preprocess (orientation, unwarp/dewarp, deskew) ──► *_flat.png (if changed)
      ──► PaddleOCR-VL 1.6 layout ──► <name>_res.json   (blocks: label, bbox, polygon, content/HTML)
      ──► PP-OCRv5 line OCR      ──► <name>_lines.json  (line quads + OCR text; cached)
      ──► schemas.PageLayout / PageLines   (adapter boundary; VL markup cleaned here)
      ──► page_builder.build_page  ──► page_model.Page   (everything measured, in source pixels)
      ──► render_model.render_page_model ──► vector PDF (612 pt wide)
      ──► fidelity.score (rasterise PDF back, compare ink per block) ──► P/R/F1
```

### 4.2 Models used
- **PaddleOCR-VL 1.6** produces the layout: block labels, reading order, the cleanest wording, and HTML for tables. Its labels include:
  - text roles: `text`, `paragraph_title`, `doc_title`, `figure_title`, `header`, `footer`, `number`, `vision_footnote`, `aside_text`;
  - picture roles: `image`, `seal`, `header_image`, `footer_image`;
  - `table`.
- **PP-OCRv5_server_det + server_rec** produce the exact geometry of each line. The image is resized to about 12 px x-height for OCR, then mapped back.
- **PP-LCNet_x1_0_doc_ori** detects orientation.
- **UVDoc** is the fallback unwarp model.
- **Division of labour:** wording comes from VL, geometry from the line detector, and size, colour, weight and font from the pixels.

### 4.3 Coordinates
- Everything is in **raw pixels of the analysed image** (the `_flat.png` if the page was flattened).
- `pipeline.match_layout_space` rescales a same-aspect image. An aspect mismatch beyond 1% raises a `ValueError`.
- The renderer's `_Geometry`:
  - `scale = 612 / img_w` and `y_pt = page_h − y·scale`, top-anchored.
  - The page height snaps to the closest standard ratio if within 5% (`page_ratios.py`); otherwise it equals the content height.

### 4.4 Module map (`docir/`)
| Module | Responsibility |
|---|---|
| `cli.py` | argparse CLI (§3). |
| `config.py` | Loads `docir_config.toml`, with defaults and getters (`straighten_max_deg()`, `font_size_offset_pt()`, …). |
| `pipeline.py` | Entry points: `process_image`, `render_json_file`, `render_pdf_from_layout`, `detect_lines` (cached), `score_pdf_file`, `lines_json_path`, `match_layout_space`. |
| `preprocess.py` | `measure_distortion` (3×4 tiles), then `flatten_page`: orientation, then curved → `dewarp` (fallback UVDoc), then skew → rotate. |
| `dewarp.py` | Text-line slope field, then a smooth vertical remap. |
| `ocr.py` | `PaddleLineOcrEngine`, `FakeLineOcrEngine`. |
| `schemas.py` | Pydantic `PageLayout`, `Block`, `PageLines`, `TextLine`. **Adapter boundary**: `clean_vl_text` resolves VL markup (§6.9). |
| `labels.py` | `PICTURE_LABELS`, `TABLE_LABELS`. |
| `reconcile.py` | Assigns detector lines to VL blocks and aligns VL wording onto OCR line breaks (§5.2). |
| `page_model.py` | Dataclasses `Page`, `Block`, `Line`, `TextStyle`, `Picture`, `Graphic`. |
| `page_builder.py` | **Orchestrator** `build_page` (§5). |
| `line_geometry.py` | `measure_line` (baseline, x-height, cap-top of one line quad), `pixel_lines`, `measure_glyph`, quad helpers. |
| `ink.py` | Binarisation, `line_cores`, `measure_text_lines` (line count and pitch of a block), `without_rules`, skew helpers. |
| `typography.py` | Role traits (bold, italic, category) → size clusters → font family fit → `harmonize_styles` → `_unify_sizes`; `offset_sizes` (§5.4). |
| `pitch.py` | **Typewriter detection** from line lengths (§6.7). |
| `fonts.py` | Word-shape font-category classifier (serif/sans/mono) by render-and-compare. |
| `font_catalog.py` | Bundled families, metrics (`x_height_em`, `cap_height_em`), `string_width`, the fallback chain (DejaVuSans, then NotoSansDevanagari), `split_by_coverage`. |
| `bold.py`, `italic.py` | Pixel stroke-width and slant measurement per block or role. |
| `line_fit.py` | DP re-flow of a block's words through its measured line slots, used when a line overflows by more than 15%. |
| `tables.py` | Row grouping for tabular blocks, `align_row_baselines`, HTML table parser (`parse_html_table` with colspan/rowspan → `TableGrid`). |
| `table_layout.py` | **HTML-grid table rendering** (§5.3). |
| `vertical.py` | **Sideways text** (page-number stamps) (§6.8). |
| `color.py` | Flat-field, page background, block fills, per-line ink colour, white balance. |
| `graphics.py` | Rules (horizontal/vertical, solid or dashed), shaded table bands (§5.5). |
| `straighten.py` | Levels blocks and rules tilted ≤ `straighten_max_deg`. |
| `render_model.py` | Draws the `Page` as a PDF: background, fills, rules, pictures, text. `fit_line` stretches or shrinks each line to its measured length. |
| `render_pdf.py` | Entry point `render_page_to_pdf`. `mode="lines"` calls `build_page`, then `render_page_model`. `mode="paragraph"` is the **legacy** reflow renderer, with fixed per-label styles. |
| `fidelity.py` | `rasterize_pdf` (pypdfium2), per-block tolerant ink precision/recall/F1, `diff_image`. |
| `page_ratios.py` | Standard paper and photo ratios. |
| `tools/` | Overlay debug tools (bbox/polygon drawing). |

**Stale files.** Not imported by anything; the user decides whether to delete them:
- `docir/_pipeline_stage3.py`
- `docir/line_refine.py`
- `docir/pipeline.py.new`
- `_legacy_*.py` at the repo root

---

## 5. How `build_page` works (stage by stage)

### 5.1 Overview (in `page_builder.py`)
1. **`reconcile`.** Each detector line is assigned to a VL block. VL wording is split onto the OCR line breaks.
2. **Per block, `_block_lines`** decides how to get the block's lines:
   1. **Sideways text.** If `vertical.is_vertical(bbox, content)`, the block becomes one rotated `Line`.
   2. **Drop cap.** `_split_drop_cap` picks out a tall glyph at the left edge as its own `Line(drop_cap=True)`.
   3. **HTML table.** If the role is `table` and the content is HTML, `_html_table_lines` calls `table_layout.analyze_table`. It stores `block.table`, records each fragment's measured x-height in `table.frag_xh`, and runs `detect_bold_rows`.
   4. **Tabular block.** If `tables.is_tabular(...)` (rows of separate cells), `_table_lines` emits one line per cell with `Line.row`.
   5. **Normal text.** `measure_text_lines(mask, ignore_rules=True)` counts the block's physical lines from its pixels. If the detector agrees (same line count, with no merged or crossing quads), the detector quads are used. Otherwise they are merged with `pixel_lines` quads. The VL wording is then split across the final lines.
   6. **Measurement.** Every line is measured with `line_geometry.measure_line`, giving the baseline, x-height and cap-top.
3. **Orphans.** OCR lines no VL block claimed (deduplicated, not inside pictures) become `text` blocks. Lines stacked left-aligned get merged into one block.
4. **`assign_styles`** (typography), then **`offset_sizes`** (config offset in pt, converted to px), then `refit_page` (line_fit), then `assign_colors`.
5. **HTML tables.** `table_layout.typeset` replaces the table's measured lines with typeset cell lines, and the colour is copied over.
6. **Graphics.** `_trim_pictures`, then `detect_graphics` (fills and rules), then `table_layout.grid_graphics` for each table.
7. **`straighten_page`**, then `align_row_baselines` for tabular blocks without `.table`.

### 5.2 Reconcile (`reconcile.py`)
- `_split_across_blocks` cuts a detector line that spans two VL blocks. `_clip_out_of_pictures` removes the parts that overlap pictures.
- `split_text_to_lines`:
  - aligns the VL text to the OCR line texts using normalised character matching;
  - restores hyphenation;
  - uses mid-word anchors;
  - trims an unseen tail with `_trim_unseen_tail`, so wording the detector never saw on that line is dropped.
- `vl_wording` lets an orphan OCR line take VL wording only if both of its ends match (`VL_WORDING_MIN_MATCH = 0.85`).

### 5.3 Tables (`table_layout.py`), simple grid rendering
1. **Grid.** `tables.parse_html_table` handles colspan and rowspan, giving a `TableGrid` of `GridCell(row, col, text, colspan, rowspan)`.
2. **Rows.**
   - `visual_rows`: detector fragments are grouped by vertical overlap.
   - `match_rows`: a monotone dynamic program assigns 0–4 visual rows to each HTML row (up to 10 for spanning cells), with text similarity as the score.
   - `absorb_wrapped_rows`: unmatched visual rows next to a row join it if they are part of its cells' text.
3. **Punctuation cells.** `absorb_punctuation` merges a punctuation-only cell (":", "☐ %") into the next cell, but **only when the detector read them as one printed fragment**.
4. **Row and column edges.**
   - Row edges are midpoints between rows' measured text centres. The outer edges always enclose the measured text.
   - Rows with no measurement get the font height plus padding. If that overflows the table box (a blank form with 20+ rows), the rows share the box evenly instead.
   - Column edges are the median measured left of each column's cells.
   - Edges snap to printed rules, but never into measured text.
5. **Printed rules.** `printed_rules` uses an adaptive threshold plus morphology, then `_is_dark`: a rule must be at least half as dark as the page's text ink relative to the paper (`RULE_MIN_CONTRAST`) along most of its length. `ruled = len(v_rules) >= 2`, so a full grid is drawn only with real dark verticals.
6. **Loose text.** Fragments inside the table area that aren't in the HTML (titles, captions) become `loose_lines`.
7. **`typeset`:**
   - **Size classes.** `_size_scales` clusters the table's fragment heights (ratio > 1.25 splits a class, and a class needs ≥ 3 lines). Small printed labels and larger typed entries in forms then each get their own size; a uniform table keeps one size.
   - **Multi-line cells.** A cell whose measured fragments span ≥ 2 lines keeps its printed line breaks (`_measured_lines`, sized per line).
   - **Other cells.** Text is wrapped inside the cell extent (`_cell_extent`: from its measured start to the next cell's start, or to the page margin for the last cell of an unruled row). It shrinks only if needed (`MIN_SHRINK = 0.6`), single lines always fit vertically, and text is clamped inside the cell.
   - Lines are emitted with `fit=False` (natural width), a per-line bold override for bold rows, and `size_px` set relative to the **style size**.
8. **`grid_graphics`.** A ruled table gets the full grid on its row and column edges, skipping spanning segments. Otherwise only its dark horizontal rules are drawn, **where they were printed**.

### 5.4 Typography (`typography.py`)
1. **`classify_roles`.**
   - Bold per role from stroke ratio, with the role holding the most text as the baseline.
   - Italic per role from slant.
   - Font category (serif/sans/mono) from word-shape matching (`fonts.py`); a role only departs from the document-wide category on clear evidence.
2. **Typewriter override** (`pitch.py`). A role whose line lengths follow the character count switches to `monospace`. If the role has too few lines to judge, the page-level verdict is used.
3. **`cluster_by_x_height`** per role.
   - A block's size is the median of its lines' x-height equivalents. A **clear outlier line** (> 1.45× spread inside the block) is ignored when most of the block's lines agree with the role's typical size; this keeps a two-line paragraph whose second line caught a photo at the body size.
   - Blocks are clustered at ratio 1.22. Short one-line blocks are "noisy" and join the nearest cluster.
4. **Family fit.** For each cluster, the bundled family in the category whose natural line widths best match the measured lengths. The default family wins ties within `FAMILY_TIE_LOG`. Size comes from `size_for_x_height(xh, font)`, which uses the font's x-height plus a calibrated measurement bias.
   - **`line_x_height` caps handling.** A line with < 50% short lowercase letters whose ink barely rises above its band is treated as all capitals, so its measured band is the cap height. It is converted by the font's x-height/cap-height ratio.
   - This uses an English letter list (`_X_ONLY`). It's a known text-level heuristic; see §8.
5. **`harmonize_styles`.**
   - Short-text styles adopt the slant of the nearest established style.
   - One face per category up to display size (1.6× body).
   - Same-look sizes within 1.12 are merged, weighted by amount of text.
   - **`_unify_sizes`** then snaps sizes within 1.12 across weights and slants, so a bold label printed at body size gets exactly the body size.
6. **`offset_sizes`** applies the user's −1 pt offset (never below 80% of the size).

### 5.5 Graphics (`graphics.py`)
1. **Mask.** An adaptive-threshold ink mask is built, then text lines are masked out. The mask polygon follows each line's direction, so sideways text works too.
2. **Find rules.** For each orientation:
   - close 3×3, then **close along the direction by `DASH_GAP_FRAC`** of the page, so dashed lines join;
   - open with a long kernel (6% of the page);
   - keep connected components with aspect ≥ 25 and limited thickness.
3. **`_plausible_rule`.** Not at the page edge, not a picture border, not striking through a text line, and not a short stroke inside a table. **Long rules inside text blocks are kept** (fill-in lines on forms).
4. **Darkness check.** The rule must be at least half as dark as the page's text ink relative to the paper (`RULE_MIN_DARKNESS`). The ink level is the 10th percentile of the page's dark pixels, so grey security patterns don't skew it.
5. **Shaded table bands.** `detect_table_fills` finds halftone bands by row percentile.

### 5.6 Rendering (`render_model.py`)
- **Draw order:** background, fills, rules, pictures, text in reading order.
- **`_draw_line`.**
  - `fit_line` stretches word gaps first, then letter spacing and glyph width, within the role's `width_scale` ± slack, so the text spans the measured length.
  - Per-line bold and `fit=False` (natural width) are honoured.
  - Lines are rotated by their baseline angle, so sideways text works.
  - **Page-edge guard:** a near-horizontal line that would run off the right edge is scaled down.
- **Font coverage.** Characters a face lacks fall back through DejaVuSans, then NotoSansDevanagari.

---

## 6. Key algorithms and decisions (with reasons)

### 6.1 Line measurement (`line_geometry.measure_line`)
The line quad is padded by 15%, rectified, and binarised with Otsu (ink is the minority class). Then:
- **Rule rows are dropped.** A row with an unbroken ink run ≥ 60% of the width, in the top or bottom 30% of the box, is a rule touching the line (underline, box edge). It is removed along with ±2 fringe rows (`RULE_ROW_FILL`, `RULE_EDGE_FRAC`, `RULE_FRINGE_ROWS`).
- **The core is the x-band.** `_core` takes the heaviest dense-row run.
- **Serif capitals** (typewriter "KENT") produce dense top and bottom bands with a sparse middle. If the core is thinner than 40% of the line box (`MIN_CORE_BOX_FRAC`), the dense runs *inside the box* are merged. That gives the cap height instead of the height of one stroke band.
  - Restricting this to thin cores **inside the box** was essential. Merging unconditionally grabbed neighbouring lines and inflated sizes on normal pages (vl1_6_4 dropped 0.918 → 0.824).

### 6.2 Line counting (`ink.measure_text_lines`)
- Deskew, then split into strips; count each strip's line cores, and take the maximum count.
- **The typical height is each strip's tallest core.** Taking the median of all cores let capitals' thin bars count as lines, so a checkbox next to text produced "3 lines".
- **`ignore_rules=True`** is used only by `page_builder`. It removes rows whose longest unbroken run is ≥ 30% of the block width, *after* deskew, so tilted underlines are caught too. It is opt-in because bold detection, font matching and the scorer also use this function, and the stripping changed their measurements.

### 6.3 Detector vs pixels agreement
`_detector_agrees` compares the detector's line count with the pixel line count. When they disagree, `_merge_with_pixels` takes over, which loses OCR text and wording alignment. Most "garbled heading" bugs came from a wrong pixel line count: underlines, checkboxes, and capitals' bars.

### 6.4 Font size
- Size = x-height ÷ the font's x-height in em ÷ a calibration bias, with caps lines converted (§5.4).
- **History.**
  - Sizes were originally about 20% too big on receipts, because all-caps lines had their cap height read as x-height.
  - The caps conversion fixed that.
  - Later, the user asked for −1 pt globally.
- A rejected alternative was measuring each line's own text rendered in the font (`band_per_px`). It was more general, but the user reverted it during a rollback; it could be re-introduced if desired.

### 6.5 Size consistency (demo_img)
- **The bug.** A 2-line block whose second line was mis-measured at 2.5× got its own 31.9 px style, against 17.6 px for the body.
- **The fix.** Outlier-robust block size, plus `_unify_sizes` across weights.
- **Attempt that was walked back.** "Always pull toward the role's median" broke a real heading on 94408603 (0.592 → 0.560). Only clear disagreements (> 1.45×) trigger it now.

### 6.6 Form handling
- **Fill-in blanks.** VL writes `___`; these are removed from the text in `schemas.clean_vl_text`, and the printed lines are drawn as rules (graphics now keeps long rules inside text blocks).
- **Dashed lines** are joined by directional closing.
- **Faint lines are never drawn.** E-stamp security patterns made bogus rules until the darkness check used the *dark end* of the page's ink, not the median.

### 6.7 Typewriter detection (`pitch.py`)
- Within each block (one size), compare two models of line length:
  - monospace: length ∝ character count;
  - proportional: length ∝ the rendered width in Liberation Sans.
- `ratio = median|log dev mono| / median|log dev prop|`, pooled over blocks. A ratio < 0.35 (with ≥ 6 lines of ≥ 12 chars) means typewriter.
- Observed values:

| Page group | Ratio |
|---|---|
| Typewriter pages (87452548, 93158159, 501713950, 83477011) | 0.12–0.22 |
| Proportional pages (most others) | ≥ 0.87 |
| hospital_receipt_Page_2 | 0.64 |

- A **page-level** pitch test, comparing lines across sizes, was too noisy and was abandoned.

### 6.8 Sideways text (`vertical.py`)
- **Trigger:** a block with height ≥ 2.5× width, ≥ 3 chars and a single line.
- **Orientation:** the crop is rotated both ways, and the result whose column ink profile best correlates with the wording rendered normally wins.
- **Placement:** the line is measured on the upright crop, and the baseline is mapped back to page coordinates. The result is a `Line` with an angle of ±90°.
- Straighten ignores it (> 5°). The renderer rotates it, and the graphics text mask follows its direction.

### 6.9 Layout-model text cleanup (`schemas.clean_vl_text`)
- A literal `\n` becomes a newline.
- `$ ^{*} $` becomes `*`, attached to the previous word.
- Other inline math holding TeX commands keeps just its text: `$ \underline{\text{PO Box 35090}} $` becomes `PO Box 35090`.
- Runs of `_` (2 or more) become a space.
- Whitespace is collapsed per line.
- A bare `$` is currency and is kept ("$100").

### 6.10 Colour (`color.py`)
- `_ink_split` treats ink as the minority class. Rows ≥ 85% dark across the region are excluded as rules.
- This fixed text inside a grey halftone box coming out **white**: the box's top border made dark pixels the majority, so the gaps were taken as ink.

---

## 7. Problem log (chronological highlights)

| Problem | Root cause | Fix |
|---|---|---|
| Table text overlapped (hospital receipts) | Per-fragment placement | HTML-grid typesetting with row DP + cell alignment |
| Rows drifted | Forced min height on measured rows | Min height only for unmeasured rows |
| Label columns squeezed | ":" cells as own columns | `absorb_punctuation` (later: only when printed together) |
| Spurious full grids | Grid inferred from h-rules alone | Full grid only with ≥ 2 dark vertical rules |
| Faint borders drawn (adobe_scan1) | No darkness check | `_is_dark` relative to page paper/ink |
| "BY Manual UPI" overlapped "INVOICE TOTAL" | Box border measured as letters; row edge inside text; text colour white | Rule rows dropped in `measure_line`; outer edges enclose text; rule rows excluded in colour |
| Fonts ~20% too large on receipts | Caps lines' cap height read as x-height | Caps conversion in `line_x_height` |
| Banner ran off page | Detector under-measured big glyph | Page-edge guard (horizontal lines only) |
| VL formula markup printed | `\underline{\text{…}}` in content | `clean_vl_text` |
| Checkbox caused "3 lines" → squashed row | Median of all strip cores | Tallest core per strip |
| Crash on 87052570 | `italic._shear_mask` shift ≥ width | Skip rows sheared out of a narrow crop |
| Fill-in lines missing | Rules inside text blocks rejected | Keep long rules inside text blocks |
| `___` drawn as text | VL blank tokens | Removed in `clean_vl_text` |
| Underlined/typewriter caps tiny | Underline or serif bands taken as x-band | Rule rows ≥ 60% run dropped; merge split bands when core is thin |
| Underlined heading → garbled pixel lines | Underline counted as a line | `without_rules` (opt-in) in line counting |
| "☐ % ☐ % % SLIM" chain | Punctuation-only cells merged blindly | Merge only when detector read them together |
| Form table headers huge (83477011) | One table size; size override compared to cell, not style | `_size_scales` + compare to style size |
| Blank-form grid overflowed into footer | 20+ empty rows forced to min height | Share the table box when overflowing |
| Security pattern drawn as lines (pk_rental_1) | Dash bridging + median ink level | Darkness check with 10th-percentile ink |
| Sideways stamps garbled | Read as horizontal lines | `vertical.py` |
| Typewriter text narrow | Classified sans | `pitch.py` |
| demo_img inconsistent sizes | Outlier line made a 2-line block its own size; bold footer kept own size | Robust block size + `_unify_sizes` |

---

## 8. Known open issues / next ideas

**Rendering gaps**
- **Bold is not reproduced on forms or receipts.** For example "INSERTION ORDER", and the condensed bold receipt body. Bold detection is per role relative to the body.
- **Handwritten entries are measured too large** in 83477011. Dashed row lines touch the digits, and dashes aren't unbroken runs, so rule-row removal misses them.
- **Entry boxes** render as the "☐" glyph rather than drawn rectangles. There is no rectangle detector.
- **hospital_receipt_Page_3** (a wide landscape receipt) has small body text in points. A −1 pt offset is about 20% there, which hits the 80% floor.

**Text-level heuristics still in code** (the user accepted them for now):
- `typography._X_ONLY` / `measures_caps`: an English letter list;
- `absorb_punctuation`: "no alphanumerics";
- hyphenation in `reconcile`/`line_fit`;
- the drop-cap first-letter check;
- the `fonts.py` probe glyphs.

A general alternative for caps sizing is to measure each line's own text rendered in the font.

**Other**
- **The legacy `paragraph` mode** (`render_pdf.py`) has fixed per-label styles and colours. It's only used with `--mode paragraph`.
- **Duplicate constant:** `RULE_ROW_FILL` is defined in both `color.py` and `line_geometry.py`.

---

## 9. Evaluation

**The score.** `fidelity.py` rasterises the PDF at source resolution and compares ink per block with a small tolerance.
- **P** (precision) is the share of drawn ink that matches source ink.
- **R** (recall) is the share of source ink that is reproduced.
- **F1** is their harmonic mean. `stray_ink` counts drawn ink outside every block.
- Bold originals rendered in regular weight lower recall. Text slightly smaller than measured (the −1 pt offset) also lowers recall, which the user accepted.

**Latest full run** (current code, `font_size_offset_pt = -1.0`):

| Page | Type | P | R | F1 |
|---|---|---|---|---|
| hospital_receipt_Page_2 | Indian hospital bill | 0.984 | 0.936 | 0.959 |
| pk_rental_Page_2 | rental agreement | 0.962 | 0.939 | 0.950 |
| vl1_6_10 | magazine article (test fixture) | 0.998 | 0.902 | 0.947 |
| book_20_sep | shoe-store GST receipt | 0.922 | 0.947 | 0.935 |
| pk_rental_Page_3 | rental agreement | 0.946 | 0.910 | 0.928 |
| vl1_6_4 | dense two-column article | 0.933 | 0.903 | 0.918 |
| 71225399 | typed letter | 0.953 | 0.883 | 0.916 |
| 10408675 | typed letter | 0.967 | 0.861 | 0.911 |
| pk_rental_Page_4 | rental agreement | 0.890 | 0.889 | 0.889 |
| demo_img | newsletter, 3 columns | 0.902 | 0.865 | 0.883 |
| vl1_6_3_flat | curved photo, flattened | 0.885 | 0.871 | 0.877 |
| adobe_scan1 | long retail receipt | 0.854 | 0.829 | 0.841 |
| 71375081 | ruled blank form | 0.798 | 0.887 | 0.840 |
| pk_rental_Page_1 | e-stamp, security pattern | 0.879 | 0.794 | 0.834 |
| 501713950_501713953 | typewriter memo with tables | 0.874 | 0.772 | 0.820 |
| 71360042 | form with typed fields | 0.746 | 0.862 | 0.800 |
| hospital_receipt_Page_1 | prescription table | 0.861 | 0.744 | 0.798 |
| 93158159 | typewriter certificate, fill-ins | 0.796 | 0.681 | 0.734 |
| 87052570 | insertion-order form | 0.672 | 0.788 | 0.725 |
| 91974564 | form | 0.721 | 0.712 | 0.717 |
| 83477011 | form + handwriting | 0.849 | 0.591 | 0.697 |
| hospital_receipt_Page_3 | wide pharmacy receipt | 0.781 | 0.557 | 0.650 |
| 94408603_94408605 | form with entry boxes | 0.599 | 0.587 | 0.593 |
| 87452548 | bordered typewriter form | 0.626 | 0.511 | 0.563 |

Average F1 is about 0.82.

**Recommended iteration loop for any change**
1. Score all pages and save the scores as a baseline.
2. Change the code.
3. Re-score all pages and diff against the baseline.
4. Visually inspect the source | render comparisons for the pages that moved.
5. Run `pytest`.
6. Accept only if regressions are under ~0.01 (or the user agrees).

**Comparison images.** `output/_eval/cmp/<name>.png` and `output/_eval/size_cmp/<name>.png` show the source on the left and the render on the right. Previews are in `output/final/<name>_preview.png`.

---

## 10. File layout

```
docir/                 package (see §4.4)
docir/assets/fonts/    bundled fonts
tests/                 pytest suite (105 tests); test_forms.py, test_table_layout.py cover the newest work
scripts/               fetch_fonts.sh, score_fixtures.py
output/                <name>_res.json (VL layout), <name>_lines.json (line OCR cache), <name>_schema.json
output/final/          rendered PDFs + *_preview.png
output/_eval/          comparison images
output/_logs/          analyze logs
*.jpg / *.png / *.jpeg sample inputs; *_flat.png = flattened versions written by preprocess
docir_config.toml      runtime config
```

---

## 11. Quick orientation checklist for a new agent

1. **Read §2 (rules) and §5 (pipeline).**
2. **Inspect a page.** Run `pytest` and the scoring loop (§3), then open a comparison PNG with the browser tools.
3. **Debugging a page:**
   - write a `.tmp_*.py` script that calls `build_page(load_layout_json(...), image, load_lines(lines_json_path(...)))`;
   - dump `page.blocks` (role, bbox, style size/family, each line's text, baseline, x_height, cap_top, source);
   - for line-measurement problems, print the rectified ink profile of `measure_line`'s padded quad;
   - monkeypatch functions (for example `typography._unify_sizes = lambda b: None`) to isolate which change moves a score.
4. **Keep fixes measurement-based and in the module that owns the concern.** Add a focused test, and re-score every page before reporting.
