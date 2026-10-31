"""Project-wide runtime configuration, loaded from `docir_config.toml`.

Keeps tunables that a user wants to flip without editing source (currently
just the debug-logging switch) in one discoverable, human-editable file
rather than scattered environment variables or code constants.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

CONFIG_FILENAME = "docir_config.toml"

_DEFAULTS: dict = {
    "logging": {"debug": False},
    "preprocess": {"mode": "auto"},
    "lines": {"enabled": True},
    "render": {"mode": "lines", "straighten_max_deg": 5.0, "font_size_offset_pt": -1.0,
               "font_size_offset_pct": 0.0, "table_mode": "measured"},
    "runtime": {"device": "auto"},
    # public=true hides the server-folder tab and server details in the web app;
    # keep_runs_hours>0 deletes uploads/outputs older than that; 0 keeps everything
    "app": {"public": False, "keep_runs_hours": 0, "max_pages_per_document": 50},
    "vlm": {
        "backend": "native",
        "server_url": "http://127.0.0.1:8001/v1",
        "api_model_name": "PaddleOCR-VL-1.6-0.9B",
        "max_concurrency": 8,
    },
}


def _find_config_file() -> Path | None:
    # walk upward from the cwd looking for the config file, then fall back
    # to the repo root (one level above the `docir` package directory)
    for base in (Path.cwd(), *Path.cwd().parents):
        candidate = base / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    fallback = Path(__file__).resolve().parent.parent / CONFIG_FILENAME
    return fallback if fallback.is_file() else None


@lru_cache(maxsize=1)
def load_config() -> dict:
    path = _find_config_file()
    if path is None:
        return _DEFAULTS
    with path.open("rb") as f:
        data = tomllib.load(f)
    return {
        section: {**defaults, **data.get(section, {})}
        for section, defaults in _DEFAULTS.items()
    }


def is_debug_logging_enabled() -> bool:
    return bool(load_config()["logging"].get("debug", False))


def preprocess_mode() -> str:
    return str(load_config()["preprocess"]["mode"])


def line_ocr_enabled() -> bool:
    return bool(load_config()["lines"]["enabled"])


def render_mode() -> str:
    return str(load_config()["render"]["mode"])


def straighten_max_deg() -> float:
    return float(load_config()["render"]["straighten_max_deg"])


def font_size_offset_pt() -> float:
    return float(load_config()["render"]["font_size_offset_pt"])


def font_size_offset_pct() -> float:
    return float(load_config()["render"]["font_size_offset_pct"])


def table_mode() -> str:
    return str(load_config()["render"].get("table_mode", "measured"))


def app_public() -> bool:
    return bool(load_config()["app"]["public"])


def keep_runs_hours() -> float:
    return float(load_config()["app"]["keep_runs_hours"])


def max_pages_per_document() -> int:
    return int(load_config()["app"]["max_pages_per_document"])


def paddle_device_kwargs() -> dict:
    """`device=` for Paddle models ("gpu:0", "cpu", ...); empty for "auto",
    which lets Paddle use a GPU whenever its CUDA build finds one."""
    device = str(load_config()["runtime"].get("device", "auto")).strip()
    return {} if device in ("", "auto") else {"device": device}


def vlm_backend_kwargs() -> dict:
    """PaddleOCR-VL recognition backend; Paddle still handles page layout."""
    config = load_config()["vlm"]
    backend = config["backend"]
    if backend == "native":
        return {"vl_rec_backend": "native"}
    if backend != "vllm-server":
        raise ValueError("[vlm] backend must be 'native' or 'vllm-server'")
    url = config["server_url"]
    if not isinstance(url, str):
        raise ValueError("[vlm] server_url must be an HTTP(S) URL")
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment:
        raise ValueError("[vlm] server_url must be an HTTP(S) base URL without query or fragment")
    model = config["api_model_name"]
    if not isinstance(model, str) or not model.strip():
        raise ValueError("[vlm] api_model_name must be a nonempty string")
    concurrency = config["max_concurrency"]
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("[vlm] max_concurrency must be a positive integer")
    return {
        "vl_rec_backend": backend,
        "vl_rec_server_url": url.rstrip("/"),
        "vl_rec_api_model_name": model,
        "vl_rec_max_concurrency": concurrency,
    }
