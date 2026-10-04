"""Project-wide runtime configuration, loaded from `docir_config.toml`.

Keeps tunables that a user wants to flip without editing source (currently
just the debug-logging switch) in one discoverable, human-editable file
rather than scattered environment variables or code constants.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

CONFIG_FILENAME = "docir_config.toml"

_DEFAULTS: dict = {
    "logging": {"debug": False},
    "preprocess": {"mode": "auto"},
    "lines": {"enabled": True},
    "render": {"mode": "lines", "straighten_max_deg": 5.0, "font_size_offset_pt": -1.0,
               "font_size_offset_pct": 0.0},
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
