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
        "logging": {**_DEFAULTS["logging"], **data.get("logging", {})},
    }


def is_debug_logging_enabled() -> bool:
    return bool(load_config()["logging"].get("debug", False))
