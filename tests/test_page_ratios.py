from docir.page_ratios import STANDARD_PAGE_RATIOS, closest_standard_ratio


def test_letter_size_scan_snaps_to_us_letter():
    # a slightly noisy scan of an 8.5x11in page (not an exact pixel ratio)
    match = closest_standard_ratio(1699, 2201)
    assert match is not None
    assert match.name == "US Letter"


def test_landscape_orientation_is_also_matched():
    match = closest_standard_ratio(2200, 1700)
    assert match is not None
    assert match.name == "US Letter"


def test_a4_scan_snaps_to_iso_216():
    match = closest_standard_ratio(2480, 3508)  # 210x297mm at 300dpi
    assert match is not None
    assert "ISO 216" in match.name


def test_distinctively_different_ratio_returns_none():
    # a very wide banner image, far from every known standard ratio
    assert closest_standard_ratio(5000, 200) is None


def test_invalid_dimensions_return_none():
    assert closest_standard_ratio(0, 100) is None
    assert closest_standard_ratio(100, 0) is None


def test_no_duplicate_ratio_values():
    ratios = [round(r.ratio, 6) for r in STANDARD_PAGE_RATIOS]
    assert len(ratios) == len(set(ratios))
