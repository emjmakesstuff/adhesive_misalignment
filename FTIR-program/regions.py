"""
The common region structure both detectors (USB's grayscale/background-
subtraction detection.py path, VDO.Ninja's color_detection.py path)
produce, so downstream code (Detection tab reporting/rendering) is
written once against one type instead of duplicated per detector.

detection.Fragment (id, bbox, mask, contour, centroid, possible/probable/
strong pixel area, mean/median/p95/max/min value) already IS almost this
structure -- both detectors already return it unchanged, since
color_detection.py calls detection.detect_fragments() internally too.
Adding source-type/mm^2 info directly onto Fragment would mean teaching
that module about concepts (which source, physical scale) it has no
reason to know about, so this module wraps it instead of extending it.
"""

from __future__ import annotations

from dataclasses import dataclass

import detection


@dataclass
class DetectedRegion:
    fragment: detection.Fragment
    source_type: str  # "usb" | "vdo_ninja"
    possible_area_mm2: float | None
    probable_area_mm2: float | None
    strong_area_mm2: float | None


def from_fragment(fragment: detection.Fragment, source_type: str, mm_per_pixel: float | None) -> DetectedRegion:
    """
    mm_per_pixel=None (no matching calibration active) yields None for
    all three mm^2 fields rather than a fabricated number -- callers
    display "N/A" in that case, same as the Detection tab already does
    today when no scale is saved.
    """
    if mm_per_pixel is None:
        possible_mm2 = probable_mm2 = strong_mm2 = None
    else:
        mm2_per_px = mm_per_pixel**2
        possible_mm2 = fragment.possible_area_px * mm2_per_px
        probable_mm2 = fragment.probable_area_px * mm2_per_px
        strong_mm2 = fragment.strong_area_px * mm2_per_px

    return DetectedRegion(
        fragment=fragment,
        source_type=source_type,
        possible_area_mm2=possible_mm2,
        probable_area_mm2=probable_mm2,
        strong_area_mm2=strong_mm2,
    )
