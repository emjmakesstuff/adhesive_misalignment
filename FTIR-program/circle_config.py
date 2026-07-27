"""
Stage 5: expected circle configuration, plus the three detection
brightness thresholds (kept here rather than a separate thresholds.py --
both are just user-set detection configuration, not algorithm logic, so
bundling them keeps detection.py/assignment.py/measurement.py focused on
their actual algorithms).

count_mode:
    "exact" -- expect exactly expected_count circles; a shortfall is
        reported as missing rather than an error.
    "max"   -- expect at most expected_count circles; fewer is normal.

area_tolerance_pct is a percentage of expected_area_mm2 (e.g. 25.0 means
a candidate group's total area must land within +/-25% of
expected_area_mm2 to be treated as a valid match, not "extra").

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
    "count_mode": "exact",
    "expected_count": 1,
    "expected_area_mm2": 78.5,
    "area_tolerance_pct": 25.0,
    "possible_threshold": 10,
    "probable_threshold": 25,
    "strong_threshold": 50,
    # VDO.Ninja / color_detection.py's own settings:
    "color_preset": "purple",
    "hue_min": None,
    "hue_max": None,
    "weak_sat_min": 30,
    "weak_val_min": 50,
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
