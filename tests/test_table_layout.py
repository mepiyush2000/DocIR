import numpy as np

from docir.page_model import Block, TextStyle
from docir.reconcile import LinePlacement
from docir.schemas import clean_vl_text
from docir.table_layout import (_cell_extent, absorb_punctuation, absorb_wrapped_rows, analyze_table,
                                grid_graphics, match_rows, typeset, visual_rows)
from docir.tables import parse_html_table

XH = 10.0


def _place(x0, y0, x1, y1, text):
    return LinePlacement(polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)], text=text,
                         ocr_text=text, score=1.0)


def _html(rows):
    return "<table>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                               for r in rows) + "</table>"


def _block(bbox):
    return Block(id=0, role="table", reading_index=0, bbox=bbox, content="",
                 style=TextStyle(font_family="LiberationSans", category="sans_serif", size_px=20.0))


# ---------- VL markup ----------

def test_vl_markup_is_resolved_but_currency_dollars_stay():
    assert clean_vl_text("2.40 bil. $ ^{{*}} $\\n(66%)") == "2.40 bil.*\n(66%)"
    assert clean_vl_text("$ ^{*} $ Inflated by") == "* Inflated by"
    assert clean_vl_text("<td>$208 Million</td><td>$ 43</td>") == "<td>$208 Million</td><td>$ 43</td>"


# ---------- grid ----------

def test_spans_take_their_slots():
    g = parse_html_table('<table><tr><td rowspan="2">A</td><td colspan="2">B</td></tr>'
                         '<tr><td>C</td><td>D</td></tr></table>')
    assert (g.n_rows, g.n_cols) == (2, 3)
    assert [(c.row, c.col, c.text) for c in g.cells] == [(0, 0, "A"), (0, 1, "B"), (1, 1, "C"), (1, 2, "D")]


def test_colon_cells_join_the_value_after_them():
    g = absorb_punctuation(parse_html_table(_html([["Name", ":", "Piyush"]])))
    assert [c.text for c in g.at(0) if c.text] == ["Name", ": Piyush"]


# ---------- rows ----------

def test_rows_match_in_order_and_skip_text_around_the_table():
    grid = parse_html_table(_html([["Item", "Qty"], ["Soap", "2"], ["Rice", "5"]]))
    vrows = visual_rows([_place(0, 0, 200, 20, "Invoice"),
                         _place(0, 40, 60, 60, "Item"), _place(150, 40, 190, 60, "Qty"),
                         _place(0, 80, 60, 100, "Soap"), _place(150, 80, 170, 100, "2"),
                         _place(0, 120, 60, 140, "Rice"), _place(150, 120, 170, 140, "5")])
    assert match_rows(grid, vrows) == {0: [1], 1: [2], 2: [3]}


def test_wrapped_header_lines_join_their_row():
    grid = parse_html_table(_html([["", "Merit Year 1976"], ["Spending", "$ 43"]]))
    vrows = visual_rows([_place(300, 0, 360, 20, "Merit"), _place(280, 30, 400, 50, "Year 1976"),
                         _place(0, 80, 120, 100, "Spending"), _place(300, 80, 350, 100, "$ 43")])
    matched = match_rows(grid, vrows)
    absorb_wrapped_rows(grid, vrows, matched)
    assert matched[0] == [0, 1] and matched[1] == [2]


# ---------- typesetting ----------

def _layout(rows, placements, bbox, gray=None):
    grid = parse_html_table(_html(rows))
    gray = np.full((400, 800), 255, np.uint8) if gray is None else gray
    return analyze_table(grid, bbox, placements, gray, XH)


def _overlap(a, b):
    ax0, ay0 = a.baseline_start
    bx0, by0 = b.baseline_start
    ax1, bx1 = a.baseline_end[0], b.baseline_end[0]
    same_line = abs(ay0 - by0) < XH
    return same_line and min(ax1, bx1) > max(ax0, bx0)


def test_cells_never_overlap_their_neighbours():
    rows = [["Service Name", "Qty", "Amount"],
            ["Dr. Sheetal Yadav - Normal - Consultation for skin", "1", "1,000.00"]]
    placements = [_place(10, 10, 150, 30, "Service Name"), _place(300, 10, 340, 30, "Qty"),
                  _place(400, 10, 480, 30, "Amount"),
                  _place(10, 50, 290, 70, "Dr. Sheetal Yadav - Normal - Consultation for skin"),
                  _place(300, 50, 310, 70, "1"), _place(400, 50, 480, 70, "1,000.00")]
    bbox = (0, 0, 500, 80)
    layout = _layout(rows, placements, bbox)
    lines = typeset(_block(bbox), layout)
    assert {l.text for l in lines} >= {"Qty", "Amount", "1", "1,000.00"}
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            assert not _overlap(a, b), (a.text, b.text)


def test_a_value_merged_into_its_labels_box_starts_at_its_own_column():
    rows = [["Doctor", ": Dr. Yadav"], ["Name", ": Piyush"]]
    placements = [_place(10, 10, 300, 30, "Doctor: Dr. Yadav"),
                  _place(10, 50, 60, 70, "Name"), _place(200, 50, 280, 70, ": Piyush")]
    layout = _layout(rows, placements, (0, 0, 400, 80))
    cell = next(c for c in layout.grid.at(0) if c.col == 1)
    x0, _ = _cell_extent(layout, cell, 2.0)
    assert x0 >= layout.col_edges[1]


def test_grid_drawn_only_for_ruled_tables():
    rows = [["A", "1"], ["B", "2"], ["C", "3"]]
    placements = [_place(10, 10 + 40 * i, 30, 30 + 40 * i, r[0]) for i, r in enumerate(rows)] + \
                 [_place(200, 10 + 40 * i, 220, 30 + 40 * i, r[1]) for i, r in enumerate(rows)]
    bbox = (0, 0, 300, 120)
    plain = _layout(rows, placements, bbox)
    assert not plain.ruled
    assert grid_graphics(plain) == []

    gray = np.full((400, 800), 255, np.uint8)
    for y in (0, 40, 80, 119):
        gray[y:y + 2, 0:300] = 0
    ruled = _layout(rows, placements, bbox, gray)
    assert ruled.ruled
    assert len(grid_graphics(ruled)) >= 4
