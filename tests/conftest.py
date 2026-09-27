import json
import os

import pytest

from docir.schemas import PageLayout

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_JSON = os.path.join(REPO_ROOT, "output", "vl1_6_10_res.json")
FIXTURE_IMAGE = os.path.join(REPO_ROOT, "vl1_6_10.png")


@pytest.fixture
def fixture_layout() -> PageLayout:
    with open(FIXTURE_JSON) as f:
        return PageLayout.from_paddle_dict(json.load(f))


@pytest.fixture
def fixture_image_path() -> str:
    return FIXTURE_IMAGE
