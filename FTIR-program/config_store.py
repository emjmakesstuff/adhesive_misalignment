"""
Small JSON-backed config persistence. This is the "data storage" module:
kept separate from camera capture, UI, calibration, and detection so any
of those can read/write persisted settings without depending on each
other. Currently holds camera connection + exposure/gain values; later
calibration data (lens distortion, physical scale, brightness references)
belongs here too rather than inventing a second storage mechanism.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).parent / "app_config.json"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_config(config: dict[str, Any], path: Path = DEFAULT_CONFIG_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
