import pytest

from docir.schemas import PageLayout


def test_from_paddle_dict_maps_fields():
    data = {
        "input_path": "page.png",
        "width": 100,
        "height": 200,
        "parsing_res_list": [
            {"block_id": 0, "block_label": "text", "block_content": " Hello ", "block_bbox": [1, 2, 3, 4]},
        ],
    }
    layout = PageLayout.from_paddle_dict(data)
    assert layout.input_path == "page.png"
    assert layout.width == 100
    assert layout.height == 200
    assert len(layout.blocks) == 1
    block = layout.blocks[0]
    assert block.id == 0
    assert block.label == "text"
    assert block.content == "Hello"  # whitespace stripped
    assert block.bbox == (1.0, 2.0, 3.0, 4.0)


def test_from_paddle_dict_defaults_missing_label_and_content():
    data = {
        "width": 10, "height": 10,
        "parsing_res_list": [{"block_bbox": [0, 0, 1, 1]}],
    }
    layout = PageLayout.from_paddle_dict(data)
    assert layout.blocks[0].label == "text"
    assert layout.blocks[0].content == ""
    assert layout.blocks[0].id == 0  # falls back to enumerate index


@pytest.mark.parametrize("width,height", [(0, 10), (10, 0), (None, 10), (10, None)])
def test_from_paddle_dict_rejects_invalid_dimensions(width, height):
    data = {"width": width, "height": height, "parsing_res_list": []}
    with pytest.raises(ValueError):
        PageLayout.from_paddle_dict(data)


def test_page_layout_round_trips_through_json():
    layout = PageLayout(width=10, height=20, blocks=[])
    restored = PageLayout.model_validate_json(layout.model_dump_json())
    assert restored == layout
