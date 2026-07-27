"""
Stage 6 (VDO.Ninja variant): HSV color-based fragment detection.

Reuses detection.detect_fragments() UNCHANGED -- that function was
already written to be agnostic to *what* single-channel 0-255 array it's
given (first exercised for background-subtraction's difference image).
This module's only job is to turn an HSV frame into one such array: a
per-pixel "color confidence" score, computed the same disciplined way
background_reference.compute_difference() computes its difference image
-- a pure pointwise formula, no blur, no smoothing.

Confidence formula, per pixel already matching the selected hue
range(s):
    confidence = 255 * min(
        normalize(saturation, weak_sat_min, core_sat_min),
        normalize(value,      weak_val_min, core_val_min),
    )
Pixels below either weak floor, or outside every hue range, get 0.
Pixels at/above both core thresholds saturate to 255.

possible/probable/strong then tier this confidence scale exactly like
the grayscale path tiers a difference image. strong_threshold doubles as
the hysteresis "core" cutoff: a fragment survives ONLY if at least one of
its pixels reaches the strong tier (detect_fragments() already tracks
this as strong_area_px) -- a broad, pale weak-mask blob with no core
pixel anywhere in it is dropped here, not silently merged or grown into
anything. This is the "strict core plus broader weak region" hysteresis
requirement 18 asks for, implemented with zero new connectivity logic:
detect_fragments()'s existing tiered-subset mechanism already does it.
"""

from __future__ import annotations

import cv2
import numpy as np

import detection

# Same named hue presets as ../test_code.py and vdo_ninja_source.py's own
# copy -- OpenCV hue range 0-179. A color needs two ranges only if it
# wraps around 0 (red).
HUE_PRESETS = {
    "red": [(0, 10), (170, 179)],
    "orange": [(10, 20)],
    "yellow": [(20, 35)],
    "green": [(35, 85)],
    "cyan": [(85, 100)],
    "blue": [(100, 130)],
    "purple": [(130, 160)],
    "pink": [(160, 170)],
}


def _hue_mask(hue_channel: np.ndarray, hue_ranges: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(hue_channel.shape, dtype=bool)

    for hue_min, hue_max in hue_ranges:
        mask |= (hue_channel >= hue_min) & (hue_channel <= hue_max)

    return mask


def _normalize(channel: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    0 at/below lo, 1 at/above hi, linear in between. hi <= lo degenerates
    to a hard step at lo (treated as "no gradient, just a floor") rather
    than dividing by zero.
    """
    if hi <= lo:
        return (channel >= lo).astype(np.float32)

    return np.clip((channel.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def build_confidence_map(
    frame_bgr: np.ndarray,
    hue_ranges: list[tuple[int, int]],
    weak_sat_min: int,
    weak_val_min: int,
    core_sat_min: int,
    core_val_min: int,
) -> np.ndarray:
    """
    Returns a new uint8 single-channel array -- never modifies frame_bgr.
    """
    if core_sat_min < weak_sat_min or core_val_min < weak_val_min:
        raise ValueError(
            f"Core thresholds must be >= the weak floor "
            f"(core_sat_min={core_sat_min} >= weak_sat_min={weak_sat_min}, "
            f"core_val_min={core_val_min} >= weak_val_min={weak_val_min})."
        )

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    hue_ok = _hue_mask(hue, hue_ranges)
    weak_floor_ok = (sat >= weak_sat_min) & (val >= weak_val_min)

    sat_score = _normalize(sat, weak_sat_min, core_sat_min)
    val_score = _normalize(val, weak_val_min, core_val_min)
    confidence = np.minimum(sat_score, val_score) * 255.0

    confidence = np.where(hue_ok & weak_floor_ok, confidence, 0.0)

    return confidence.astype(np.uint8)


def detect_color_fragments(
    frame_bgr: np.ndarray,
    hue_ranges: list[tuple[int, int]],
    weak_sat_min: int,
    weak_val_min: int,
    core_sat_min: int,
    core_val_min: int,
    possible_threshold: int,
    probable_threshold: int,
    strong_threshold: int,
) -> tuple[list[detection.Fragment], np.ndarray]:
    """
    Returns (fragments, confidence_map) -- the confidence map is also
    returned so the Detection tab can display it as one of its live
    panels, the same way the grayscale path displays its difference
    image.
    """
    confidence = build_confidence_map(
        frame_bgr, hue_ranges, weak_sat_min, weak_val_min, core_sat_min, core_val_min
    )

    fragments = detection.detect_fragments(confidence, possible_threshold, probable_threshold, strong_threshold)

    # Hysteresis: a weak/pale region only counts as a detected spot if it
    # contains at least one confident "core" pixel -- pulls in faint
    # edges of a real spot without also accepting unrelated pale regions
    # elsewhere in the frame that never touch a real core. Filters the
    # already-detected list; never merges/grows/re-labels anything.
    fragments = [f for f in fragments if f.strong_area_px > 0]

    return fragments, confidence
