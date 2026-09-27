"""Standard page/paper/photo aspect ratios.

Real-world documents (books, invoices, forms, photos, screenshots) almost
always come in one of a small number of well-known width:height ratios.
Source images fed into this pipeline can have slightly noisy dimensions
(cropping, scanning, compression) that don't exactly match any standard
ratio even though the original document clearly was one -- e.g. a scan of
a US Letter page might come out at 1699x2201 instead of exactly 8.5x11.

`closest_standard_ratio()` snaps a detected width/height to the nearest
known ratio (checking both portrait and landscape orientations of every
entry) as long as it's close enough to be a plausible match; otherwise it
returns None, meaning this page's shape is distinctively its own and
should be left alone rather than forced into a standard mold.
"""

from __future__ import annotations

from dataclasses import dataclass


def _r(width: float, height: float) -> float:
    return width / height


@dataclass(frozen=True)
class PageRatio:
    name: str
    ratio: float  # width / height, for this entry's conventional orientation


# One entry per *distinct* ratio -- e.g. every ISO A/B/C size shares the same
# 1:sqrt(2) ratio, so it's listed once rather than as A0..A10/B0..B10/C0..C10.
# Both this ratio and its reciprocal (landscape <-> portrait) are checked by
# `closest_standard_ratio()`, so only one conventional orientation is needed
# per entry.
STANDARD_PAGE_RATIOS: tuple[PageRatio, ...] = (
    # ISO 216 paper (A-series, B-series, C-series envelopes all share this)
    PageRatio("ISO 216 (A/B/C series, e.g. A4)", _r(1, 2 ** 0.5)),

    # North American paper
    PageRatio("US Letter", _r(8.5, 11)),
    PageRatio("US Legal", _r(8.5, 14)),
    PageRatio("US Ledger/Tabloid / Statement/Half Letter", _r(11, 17)),
    PageRatio("US Executive", _r(7.25, 10.5)),
    PageRatio("US Folio", _r(8.5, 13)),
    PageRatio("US Quarto / Photo print 8x10", _r(8, 10)),
    PageRatio("US Junior Legal", _r(5, 8)),
    PageRatio("US Foolscap", _r(8, 13)),

    # Common invoice/receipt/index-card/photo-print ratios
    PageRatio("Index card 3x5", _r(3, 5)),
    PageRatio("Index card 4x6", _r(4, 6)),
    PageRatio("Index card 5x7", _r(5, 7)),
    PageRatio("Photo print 11x14", _r(11, 14)),
    PageRatio("Panoramic/receipt strip", _r(1, 3)),

    # Business/credit cards
    PageRatio("Business card (3.5x2in)", _r(3.5, 2)),
    PageRatio("ISO/credit card (85.6x53.98mm)", _r(85.6, 53.98)),

    # Photo, screen, and video ratios
    PageRatio("Square (1:1)", _r(1, 1)),
    PageRatio("Classic photo/slide (4:3)", _r(4, 3)),
    PageRatio("35mm photo/APS-H (3:2)", _r(3, 2)),
    PageRatio("Medium format (5:4)", _r(5, 4)),
    PageRatio("Widescreen (16:9)", _r(16, 9)),
    PageRatio("Widescreen (16:10)", _r(16, 10)),
    PageRatio("Ultra-wide (21:9)", _r(21, 9)),
    PageRatio("Super 16 film (5:3)", _r(5, 3)),
)


def closest_standard_ratio(width: float, height: float, tolerance: float = 0.05) -> PageRatio | None:
    """Return the `PageRatio` (oriented to match `width`/`height`) closest to
    the observed shape, provided it's within `tolerance` relative
    difference -- else None if every standard ratio diverges too much."""
    if width <= 0 or height <= 0:
        return None
    observed = width / height
    best, best_diff = None, tolerance
    for candidate in STANDARD_PAGE_RATIOS:
        for oriented_ratio in (candidate.ratio, 1 / candidate.ratio):
            diff = abs(oriented_ratio - observed) / observed
            if diff < best_diff:
                best, best_diff = PageRatio(candidate.name, oriented_ratio), diff
    return best
