import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from docir import config
from docir.ocr import PaddleOcrVlEngine

from .conftest import FIXTURE_JSON


def _settings(monkeypatch, **overrides):
    vlm = {**config._DEFAULTS["vlm"], **overrides}
    monkeypatch.setattr(config, "load_config", lambda: {**config._DEFAULTS, "vlm": vlm})


def test_native_backend_does_not_pass_server_settings(monkeypatch):
    _settings(monkeypatch, server_url="", api_model_name="", max_concurrency=0)
    assert config.vlm_backend_kwargs() == {"vl_rec_backend": "native"}


def test_vllm_backend_maps_settings_to_paddleocr(monkeypatch):
    _settings(monkeypatch, backend="vllm-server", server_url="http://127.0.0.1:8001/v1/")
    assert config.vlm_backend_kwargs() == {
        "vl_rec_backend": "vllm-server",
        "vl_rec_server_url": "http://127.0.0.1:8001/v1",
        "vl_rec_api_model_name": "PaddleOCR-VL-1.6-0.9B",
        "vl_rec_max_concurrency": 8,
    }


@pytest.mark.parametrize("overrides", [
    {"backend": "unknown"},
    {"server_url": ""},
    {"server_url": "file:///tmp/model"},
    {"server_url": "http:///v1"},
    {"server_url": "http://localhost:8001/v1?model=x"},
    {"server_url": "http://localhost:8001/v1#fragment"},
    {"server_url": 123},
    {"api_model_name": ""},
    {"api_model_name": None},
    {"max_concurrency": 0},
    {"max_concurrency": -1},
    {"max_concurrency": True},
    {"max_concurrency": 1.5},
])
def test_invalid_backend_settings_fail_explicitly(monkeypatch, overrides):
    _settings(monkeypatch, **{"backend": "vllm-server", **overrides})
    with pytest.raises(ValueError, match=r"\[vlm\]"):
        config.vlm_backend_kwargs()


@pytest.mark.parametrize("backend", ["native", "vllm-server"])
def test_layout_engine_lazily_constructs_configured_backend(monkeypatch, fixture_layout, backend):
    _settings(monkeypatch, backend=backend)
    monkeypatch.setattr(config, "paddle_device_kwargs", lambda: {"device": "gpu:0"})
    with open(FIXTURE_JSON) as f:
        raw = json.load(f)
    pipeline = Mock()
    pipeline.predict.return_value = [SimpleNamespace(json={"res": raw})]
    constructor = Mock(return_value=pipeline)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCRVL=constructor))
    engine = PaddleOcrVlEngine()
    constructor.assert_not_called()
    assert engine.analyze("page.png") == fixture_layout
    assert engine.analyze("page.png") == fixture_layout
    constructor.assert_called_once_with(
        pipeline_version="v1.6", device="gpu:0", **config.vlm_backend_kwargs())
    assert engine.last_raw is raw


def test_server_failure_propagates_without_native_fallback(monkeypatch):
    _settings(monkeypatch, backend="vllm-server")
    constructor = Mock(side_effect=ConnectionError("server unavailable"))
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCRVL=constructor))
    with pytest.raises(ConnectionError, match="server unavailable"):
        PaddleOcrVlEngine().analyze("page.png")
    assert constructor.call_count == 1
