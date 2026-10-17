from docir.page_model import Block, Line, TextStyle
from docir.tables import align_row_baselines, assign_cell_text, group_rows, is_tabular, vl_tokens
from docir.typography import size_table_cells


def _quad(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


RECEIPT = [_quad(36, 940, 198, 974), _quad(249, 942, 423, 974), _quad(495, 941, 620, 977),
           _quad(37, 974, 189, 1002), _quad(516, 973, 619, 1006),
           _quad(45, 1002, 206, 1030), _quad(516, 1004, 619, 1037)]
PARAGRAPH = [_quad(40, 100 + 30 * i, 600, 124 + 30 * i) for i in range(5)]


def test_rows_of_cells_are_tabular_but_paragraphs_are_not():
    assert [len(r) for r in group_rows(RECEIPT)] == [3, 2, 2]
    assert is_tabular("text", RECEIPT)
    assert not is_tabular("text", PARAGRAPH)
    assert is_tabular("table", PARAGRAPH)


def test_each_cell_gets_its_own_vl_words_and_duplicates_are_skipped():
    words = vl_tokens("Taxable value HSN:64029990 ₹3,811.02\nTax = 18.00% ₹685.98\n"
                      "CGST = 9.00% ₹342.99\nWhether RCM Payable: No")
    ocr = ["Taxable value", "HSN:64029990", "(3,811.02", "Tax=18.00%", "685.98", "CGST=9.00%", "342.99"]
    assert assign_cell_text(ocr, words) == [
        "Taxable value", "HSN:64029990", "₹3,811.02", "Tax = 18.00%", "₹685.98", "CGST = 9.00%", "₹342.99"]


def test_cell_without_vl_text_keeps_its_ocr_reading():
    assert assign_cell_text(["Subtotal", "*Consumer Details"], ["Subtotal"]) == ["Subtotal", "*Consumer Details"]


def test_a_cell_does_not_take_its_neighbours_vl_words():
    words = vl_tokens("<table><tr><td>MO</td><td>[]</td><td>[134358678:524838:-</td></tr></table>")
    assert assign_cell_text(["MO", "[134358678:524838:-"], words) == ["MO", "[134358678:524838:-"]


def test_a_box_holding_part_of_a_vl_word_gets_only_that_part():
    words = ["133548296:521673:-97.2745561:33.0227219:8:60.00:14;"]
    assert assign_cell_text(["2745561:33.0227219:8:69.90:141"], words) == ["2745561:33.0227219:8:60.00:14;"]


def test_a_box_spanning_several_rows_does_not_merge_them():
    rows = [_quad(40, 100 + 37 * i, 300, 137 + 37 * i) for i in range(4)]
    values = [_quad(500, 100 + 37 * i, 600, 137 + 37 * i) for i in range(4)]
    ticks = [_quad(400, 100, 420, 248)]          # a column of tick marks read as one box
    grouped = group_rows(rows + values + ticks)
    assert len(grouped) == 4
    assert sum(len(r) for r in grouped) == 9


def test_html_table_cells_are_tokenised_in_order():
    html = "<table><tr><td>Tax</td><td>₹685.98</td></tr><tr><td>Total</td><td>₹4,497.00</td></tr></table>"
    assert vl_tokens(html) == ["Tax", "₹685.98", "Total", "₹4,497.00"]


def test_row_cells_share_one_baseline():
    left = Line(text="Tax", baseline_start=(40, 1000), baseline_end=(180, 1000), x_height=12, cap_top=17)
    right = Line(text="₹685.98", baseline_start=(520, 992), baseline_end=(615, 992), x_height=12, cap_top=17)
    nxt = Line(text="CGST", baseline_start=(40, 1030), baseline_end=(200, 1030), x_height=12, cap_top=17)
    block = Block(id=0, role="text", reading_index=0, bbox=(0, 0, 1, 1), content="", lines=[left, right, nxt],
                  tabular=True)
    align_row_baselines(block)
    assert left.baseline_start[1] == right.baseline_start[1] == right.baseline_end[1]
    assert nxt.baseline_start[1] == 1030


def test_a_cell_printed_much_larger_than_its_table_keeps_its_own_size():
    def cell(text, x, xh):
        return Line(text=text, baseline_start=(x, 100), baseline_end=(x + 80, 100), x_height=xh, cap_top=1.4 * xh)
    labels = [cell("computer", 0, 10), cell("mixed office", 100, 10), cell("white office", 200, 10)]
    entry = cell("879", 300, 22)
    block = Block(id=0, role="table", reading_index=0, bbox=(0, 0, 1, 1), content="",
                  lines=labels + [entry], tabular=True, style=TextStyle(font_family="LiberationMono", category="monospace", size_px=20))
    size_table_cells([block])
    assert all(l.size_px is None for l in labels)
    assert entry.size_px is not None and entry.size_px > 1.5 * block.style.size_px


def test_similarity_pruning_preserves_all_qualifying_matches():
    from docir.tables import MIN_CELL_MATCH, _similarity

    words = ["", " A B ", "ababa", "babab"] + [
        "".join(chars) for size in range(1, 4) for chars in itertools.product("ab", repeat=size)]
    for a, b in itertools.product(words, repeat=2):
        expected = _similarity(a, b)
        actual = _similarity(a, b, minimum=MIN_CELL_MATCH)
        if expected >= MIN_CELL_MATCH:
            assert actual == expected
        else:
            assert actual < MIN_CELL_MATCH
import itertools
