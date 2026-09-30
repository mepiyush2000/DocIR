"""Layout-label vocabulary shared by every stage (PaddleOCR-VL labels)."""

# Blocks whose pixels are placed as a cropped picture, never as text
PICTURE_LABELS = frozenset({"image", "figure", "chart", "seal", "header_image", "footer_image"})
TABLE_LABELS = frozenset({"table"})


def is_picture(label: str) -> bool:
    return label in PICTURE_LABELS
