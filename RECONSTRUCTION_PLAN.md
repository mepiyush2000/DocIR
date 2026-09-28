# Scan -> Vector PDF Reconstruction Plan

Goal: a **fully vector** PDF that looks almost identical to the scanned page.
Reading order alone does not change how a page looks once every element is
placed at its measured position; it matters for copy/paste order,
accessibility, and deciding which block owns overflowing text. Visual
fidelity comes from: a flat page before measuring, exact per-line geometry,
the *same* typeface (embedded open fonts, incl. condensed/wide variants),
and reproducing non-text visuals (colors, panels, rules, pictures).

Architecture: **analysis -> page model -> renderer**. All measurement lands
in one intermediate model; the renderer only draws it.

## Phase 1 -- Measurement first
1. `docir/fidelity.py`: rasterize the output PDF (pypdfium2), per-block ink
   precision/recall/F1 (tolerance ~0.3 x-height), red/blue diff image;
   `docir score` command.
2. Baseline scores for the 4 fixtures (vl1_6_10, demo_img, vl1_6_3,
   vl1_6_4). Every later phase must raise the page score without making any
   fixture visibly worse.

## Phase 2 -- Better analysis inputs (`docir/ocr.py`, `docir/preprocess.py`)
3. Page preprocessing: PaddleOCR document orientation + unwarping (UVDoc)
   -> flat page image `<name>_flat.png`; applied only when skew/curvature is
   detected.
4. Line OCR: PaddleOCR (PP-OCRv5 det+rec) -> per-line polygon + text,
   cached as `<name>_lines.json`.
5. Reconcile lines with PaddleOCR-VL blocks: each OCR line to exactly one
   block by overlap; unclaimed lines to the block they continue (VL
   `block_order`); VL's cleaner text aligned onto OCR lines by
   character-level sequence alignment (line breaks from pixels, wording from VL).

## Phase 3 -- Page model (`docir/page_model.py`, `docir/typography.py`)
6. Dataclasses: `Page` (size, background, graphics, pictures, blocks in
   reading order), `Block` (role, reading index, lines, style), `Line`
   (baseline, x0, x1, text, color, runs), `Graphic` (rule/box/fill).
7. Role typography (bold/italic/family/size clusters/width) moves out of
   `render_pdf.py`; one size + one glyph width per role.

## Phase 4 -- Fonts that actually match (`docir/fonts.py`, `docir/assets/fonts/`)
8. Bundle open-licence families (+ licences): Liberation Sans/Serif/Mono +
   Sans Narrow, DejaVu Sans/Serif (+Condensed), Noto Sans/Serif, a condensed
   sans.
9. Match against the bundled set only and embed the winner (reportlab
   `TTFont`), so real metrics are used and glyph-squeezing hacks disappear.

## Phase 5 -- Non-text visuals (`docir/color.py`, `docir/graphics.py`)
10. Text color per line (median ink), block background fill when not
    near-white, page background color.
11. Rules/underlines/boxes via OpenCV morphology on non-text residue ->
    vector lines/rects; leftover non-text regions -> cropped images.
12. Drop caps as their own glyph; superscript prefixes as small runs.

## Phase 6 -- Renderer (`docir/render_pdf.py` draws the page model)
13. Draw order: page background -> block fills -> graphics -> pictures -> text.
14. Absolute line placement; word/char spacing absorb width differences,
    glyph width within +/-5% of the role's; no column cursors.
15. Text written in reading order.
16. Per-block fallback to paragraph layout; `[render] mode` config switch.

## Verification
- After every phase: render + score all 4 fixtures, record page F1 in
  `PROGRESS.md`, visual comparison against the scan.
- Unit tests per new module (reconciliation, color, rules, fonts, model).
- Consistency: one size and one glyph width per role.
- Target: page F1 >= 0.95 on flat documents; large gain on vl1_6_3 after
  unwarping.

## Scope
Fully vector output (no scan background). Out of scope for now: tables,
multi-page PDFs, editing UI, rotated text (tilt is absorbed).
