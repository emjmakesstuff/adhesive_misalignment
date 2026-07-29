"""
Detection threshold configuration: the USB/grayscale detector's three
difference thresholds, plus the VDO.Ninja/color detector's own settings
(kept here rather than a separate thresholds.py -- both are just
user-set detection configuration, not algorithm logic, so bundling them
keeps detection.py focused on its actual algorithm).

(Originally also held expected-circle-count/area/tolerance fields for
grouping detected fragments into circles -- removed along with
assignment.py/measurement.py, which were the only code that ever read
them; the Processing tab's per-pad ROI analysis replaced that whole
approach with user-drawn rectangular search regions instead. The
filename predates that removal.)

possible/probable/strong_threshold are 0-255 cut points applied to the
background-subtracted DIFFERENCE image (see background_reference.py),
not raw absolute brightness -- a real new contact spot might only be
10-60 counts brighter than the background, much smaller than typical
absolute-brightness values, hence the lower defaults below compared to
this module's earlier (pre background-subtraction) thresholds. Validated
elsewhere to stay non-decreasing (possible <= probable <= strong), since
detection.py's per-fragment area breakdown assumes that ordering. These
are the USB/grayscale detector's thresholds specifically -- the
vdo_*_threshold fields below are the separate, color-confidence-scale
equivalents used by color_detection.py, kept under distinctly-named keys
rather than reusing these three, since one profile's config could in
principle be inspected/compared without knowing which detector produced
it.

color_preset/hue_min/hue_max/weak_*/core_* are color_detection.py's
inputs (see that module for the confidence-scale formula they feed).
hue_min/hue_max are null unless explicitly overridden -- null means "use
color_preset's own range".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CIRCLE_CONFIG_PATH = Path(__file__).parent / "circle_config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "possible_threshold": 10,
    "probable_threshold": 25,
    "strong_threshold": 50,
    # VDO.Ninja / color_detection.py's own settings:
    "color_preset": "blue",
    "hue_min": None,
    "hue_max": None,
    "weak_sat_min": 90,
    "weak_val_min": 90,
    "core_sat_min": 100,
    "core_val_min": 100,
    "vdo_possible_threshold": 10,
    "vdo_probable_threshold": 80,
    "vdo_strong_threshold": 200,
    "show_masks": True,
    "vdo_processing_fps": 5.0,
}


def load_circle_config(path: Path = DEFAULT_CIRCLE_CONFIG_PATH) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        return dict(DEFAULT_CONFIG)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_CONFIG)

    merged = dict(DEFAULT_CONFIG)
    merged.update(data)
    return merged


def save_circle_config(data: dict[str, Any], path: Path = DEFAULT_CIRCLE_CONFIG_PATH) -> None:
    with open(Path(path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
