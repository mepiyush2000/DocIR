from docir.page_model import Block, Line
from docir.tables import align_row_baselines, assign_cell_text, group_rows, is_tabular, vl_tokens


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
