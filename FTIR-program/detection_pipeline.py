"""
The actual fragment-detection dispatch, extracted from DetectionTab's
live loop so ProcessingTab (offline, working from a recording's own
calibration snapshot -- see recording_store.py) can run the identical
detector code against a saved recording instead of only ever against a
live frame. One function, run_detection(), used identically by both.

Deliberately does NOT know about calibration_profiles.py or
recording_store.py -- callers resolve detector_type/config/background
from wherever is appropriate for their context (the live active profile
for DetectionTab, a recording's own calibration_snapshot/ for
ProcessingTab) and pass the resolved values in. Keeping this module
profile-agnostic is what makes it safely reusable in both places without
blurring the "recordings are self-contained, never re-resolve the live
profile" rule the rest of this app is careful about.

detection.py/background_reference.py/color_detection.py are all
unmodified -- this module only orchestrates the same calls DetectionTab
already made, it adds no new detection logic of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

import background_reference
import color_detection
import detection


def active_hue_ranges(config: dict) -> list[tuple[int, int]]:
    """The same override-or-preset logic DetectionTab's hue controls
    already implement -- both live and offline color detection need it,
    so it lives here rather than only in tabs/detection_tab.py.

    config["color_presets"] (a list of preset names, e.g. ["cyan",
    "blue"]) lets multiple colors be targeted simultaneously -- their hue
    ranges are simply concatenated, since _hue_mask() already ORs every
    range together. Checked before the older single-preset key so a
    caller that writes both (or migrates later) prefers the list.
    color_preset (singular) is kept as the ONLY key DetectionTab's live
    controls ever write, so this stays fully backward compatible."""
    if config.get("hue_min") is not None and config.get("hue_max") is not None:
        return [(int(config["hue_min"]), int(config["hue_max"]))]
    presets = config.get("color_presets")
    if presets:
        ranges = []
        for name in presets:
            ranges.extend(color_detection.HUE_PRESETS[name])
        return ranges
    return color_detection.HUE_PRESETS[config["color_preset"]]


def run_detection(
    corrected_bgr: np.ndarray,
    detector_type: str,
    config: dict,
    background: np.ndarray | None = None,
) -> tuple[list[detection.Fragment], np.ndarray]:
    """
    Given an already undistorted+cropped BGR frame, runs the same
    detector DetectionTab's live loop always has:

      - "grayscale": background-subtracted difference + threshold
        (detection.py + background_reference.py). Requires `background`,
        already loaded and shape-matched by the caller -- this function
        does not load or validate it, since where a background reference
        comes from (a live profile's file vs. a recording's own
        calibration_snapshot/) and how a mismatch should be reported are
        both caller-specific, not shared detection logic.
      - "color": HSV confidence (color_detection.py, which itself calls
        detection.detect_fragments() internally -- see that module).
        `background` is ignored.

    Returns (fragments, display_map). Raises ValueError for a malformed
    config (non-monotonic thresholds, unknown color preset, ...) -- the
    same exception detection.py/color_detection.py already raise, not
    swallowed here, so each caller's own try/except keeps deciding how
    to surface it (matching what DetectionTab's live loop already did
    before this was extracted).
    """
    if detector_type == "grayscale":
        gray = detection.to_grayscale(corrected_bgr)
        display_map = background_reference.compute_difference(gray, background)
        fragments = detection.detect_fragments(
            display_map, config["possible_threshold"], config["probable_threshold"], config["strong_threshold"]
        )
        return fragments, display_map

    fragments, display_map = color_detection.detect_color_fragments(
        corrected_bgr,
        active_hue_ranges(config),
        config["weak_sat_min"],
        config["weak_val_min"],
        config["core_sat_min"],
        config["core_val_min"],
        config["vdo_possible_threshold"],
        config["vdo_probable_threshold"],
        config["vdo_strong_threshold"],
    )
    return fragments, display_map


# ============================================================
# ROI (per-pad) analysis -- built on top of run_detection() above,
# never a second/parallel detector implementation. See the module
# docstring's framing: DetectionTab and ProcessingTab's full-frame
# report both use run_detection() directly and are UNCHANGED by
# everything below; this section only adds an independent, ROI-cropped
# way to call the exact same function.
# ============================================================


def tier_thresholds(detector_type: str, config: dict) -> tuple[int, int, int]:
    """The three threshold values run_detection() itself already reads
    out of config, exposed here too since compute_tier_masks() and the
    ROI functions below need them independently of an actual detection
    call."""
    if detector_type == "grayscale":
        return config["possible_threshold"], config["probable_threshold"], config["strong_threshold"]
    return config["vdo_possible_threshold"], config["vdo_probable_threshold"], config["vdo_strong_threshold"]


def compute_tier_masks(
    display_map: np.ndarray,
    fragments: list[detection.Fragment],
    possible_threshold: int,
    probable_threshold: int,
    strong_threshold: int,
) -> dict[str, np.ndarray]:
    """
    Reconstructs boolean masks (possible/probable/strong), sized to
    display_map, from the fragments a detection run already accepted --
    NOT an independent re-threshold of display_map, which would disagree
    with the fragment list wherever hysteresis filtering applies (see
    color_detection.py: a whole connected component is discarded if none
    of its pixels reach the strong tier anywhere in it). Reconstructing
    from fragments guarantees these masks always match exactly what that
    run already decided counts as detected, pixel for pixel.

    Uses each fragment's own exact connected-component mask
    (fragment.mask, from cv2.connectedComponentsWithStats) -- NEVER fills
    its bounding rectangle. A fragment's real shape is frequently not
    rectangular, and filling bbox would silently invent detected pixels
    that were never actually thresholded as such.
    """
    h, w = display_map.shape[:2]
    possible_mask = np.zeros((h, w), dtype=bool)
    probable_mask = np.zeros((h, w), dtype=bool)
    strong_mask = np.zeros((h, w), dtype=bool)

    for fragment in fragments:
        x, y, fw, fh = fragment.bbox
        local_display = display_map[y : y + fh, x : x + fw]
        local_fragment_mask = fragment.mask  # exact per-pixel shape, local to bbox -- not a rectangle

        possible_mask[y : y + fh, x : x + fw] |= local_fragment_mask
        probable_mask[y : y + fh, x : x + fw] |= local_fragment_mask & (local_display >= probable_threshold)
        strong_mask[y : y + fh, x : x + fw] |= local_fragment_mask & (local_display >= strong_threshold)

    return {"possible": possible_mask, "probable": probable_mask, "strong": strong_mask}


@dataclass
class RoiDetectionResult:
    fragments: list  # detection.Fragment, translated to FULL corrected-frame coordinates (for overlay drawing)
    display_map: np.ndarray  # crop-local -- "detector signal": background difference (grayscale) or HSV
    #                           confidence score (color). NEVER physical brightness for color -- see
    #                           summarize_roi_result()'s detector_signal block.
    corrected_bgr_crop: np.ndarray  # crop-local -- the actual corrected-frame pixels (real camera signal)
    tier_masks: dict  # crop-local boolean masks, keys "possible"/"probable"/"strong" -- kept LOCAL
    #                    deliberately (see summarize_roi_result): numerical measurement always happens
    #                    in crop-local coordinates, only fragments (for display) get translated.
    rect_requested: tuple[int, int, int, int]  # x, y, w, h as originally requested
    rect_used: tuple[int, int, int, int]  # x, y, w, h actually used, after clipping to frame bounds
    clipped: bool


def run_roi_detection(
    corrected_bgr: np.ndarray,
    rect: dict,  # {"x", "y", "width", "height"}, corrected-frame pixel coordinates
    detector_type: str,
    config: dict,
    background: np.ndarray | None = None,
    min_area_mm2: float | None = None,
    mm_per_pixel: float | None = None,
) -> RoiDetectionResult | None:
    """
    Runs detection INDEPENDENTLY on a crop of the corrected frame
    restricted to `rect`, by cropping FIRST and then calling
    run_detection() on that crop directly -- not by running full-frame
    detection and intersecting the result with the ROI afterward.

    This distinction is the whole point: cropping first means pixels
    outside the ROI can never influence connected-components or the
    strong-tier hysteresis check inside it. Cropping AFTER a full-frame
    run would let a real bug through -- e.g. color detection's hysteresis
    keeps a whole connected component alive if ANY pixel in it reaches
    the strong tier, so a weak-only blob inside a pad's ROI could
    survive only because a strong pixel elsewhere in the frame (outside
    that ROI) happened to share its connected component. Cropping first
    makes that structurally impossible: the crop simply doesn't contain
    those outside pixels for connectedComponents to ever see.

    background (grayscale detector only) is cropped identically to the
    same rect -- it's already stored in the same corrected coordinate
    system a live/recorded frame is (see background_reference.py), so
    slicing it the same way keeps pixel alignment exact.

    Returns None if `rect` doesn't overlap the frame at all after
    clipping to its bounds.

    min_area_mm2, if given, drops any connected component smaller than it
    BEFORE tier_masks/area sums/coverage ever see it -- a per-ROI noise
    floor, filtering the already-detected fragment list exactly the way
    color_detection.py's own strong-tier hysteresis filter does (never a
    re-threshold or a merge/grow of what's left). Interpreted as mm2 when
    mm_per_pixel is available, else as the same number in raw px.
    """
    h, w = corrected_bgr.shape[:2]
    rx, ry, rw, rh = rect["x"], rect["y"], rect["width"], rect["height"]
    x0, y0 = max(0, rx), max(0, ry)
    x1, y1 = min(w, rx + rw), min(h, ry + rh)

    if x1 <= x0 or y1 <= y0:
        return None

    crop_bgr = corrected_bgr[y0:y1, x0:x1]
    crop_background = background[y0:y1, x0:x1] if background is not None else None

    fragments_local, display_map = run_detection(crop_bgr, detector_type, config, background=crop_background)

    if min_area_mm2 is not None:
        mm2_per_px = mm_per_pixel**2 if mm_per_pixel is not None else None
        min_area_px = min_area_mm2 / mm2_per_px if mm2_per_px is not None else min_area_mm2
        fragments_local = [f for f in fragments_local if f.possible_area_px >= min_area_px]

    possible_t, probable_t, strong_t = tier_thresholds(detector_type, config)
    tier_masks = compute_tier_masks(display_map, fragments_local, possible_t, probable_t, strong_t)

    # Translate EVERY coordinate-bearing field consistently for display
    # in full-frame coordinates -- bbox's (x, y) origin, the full contour
    # point array, and the centroid. The fragment's mask is deliberately
    # NOT translated/reprojected: it stays local to the (now-translated)
    # bbox, exactly as detection.py already defines it, and all of this
    # function's own numerical measurement (tier_masks above) stays in
    # crop-local coordinates too -- only what gets drawn on screen needs
    # full-frame coordinates.
    offset = np.array([x0, y0], dtype=np.int32)
    fragments_full = [
        replace(
            fragment,
            bbox=(fragment.bbox[0] + x0, fragment.bbox[1] + y0, fragment.bbox[2], fragment.bbox[3]),
            contour=fragment.contour + offset,
            centroid=(fragment.centroid[0] + x0, fragment.centroid[1] + y0),
        )
        for fragment in fragments_local
    ]

    return RoiDetectionResult(
        fragments=fragments_full,
        display_map=display_map,
        corrected_bgr_crop=crop_bgr,
        tier_masks=tier_masks,
        rect_requested=(rx, ry, rw, rh),
        rect_used=(x0, y0, x1 - x0, y1 - y0),
        clipped=(x0, y0, x1 - x0, y1 - y0) != (rx, ry, rw, rh),
    )


def _stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _corrected_frame_signal(corrected_bgr_crop: np.ndarray, mask: np.ndarray, detector_type: str) -> dict:
    """
    The REAL camera signal within the detected (possible-tier) pixels --
    always computed from the actual corrected-frame pixel values, never
    from display_map (which is a difference/confidence signal, not a
    direct measurement -- see the detector_signal block this is kept
    separate from in summarize_roi_result()).

    - monochrome (grayscale detector): a single set of stats, labeled
      "observed_monochrome_intensity" -- this recording's sensor reads
      brightness directly, so this genuinely is a physical intensity
      reading.
    - color: B/G/R channel stats PLUS the HSV Value channel, labeled
      "observed_color_channels" -- deliberately NOT called "brightness"
      or "intensity" on its own; a converted grayscale number from a
      color sensor conflates three channels in a way that isn't a
      physical measurement the way a monochrome sensor's raw reading is.
    """
    if detector_type == "grayscale":
        gray = detection.to_grayscale(corrected_bgr_crop)
        values = gray[mask] if mask.any() else np.array([])
        return {"source": "observed_monochrome_intensity", **_stats(values)}

    b = corrected_bgr_crop[:, :, 0]
    g = corrected_bgr_crop[:, :, 1]
    r = corrected_bgr_crop[:, :, 2]
    hsv_v = cv2.cvtColor(corrected_bgr_crop, cv2.COLOR_BGR2HSV)[:, :, 2]
    has_pixels = mask.any()

    return {
        "source": "observed_color_channels",
        "b": _stats(b[mask] if has_pixels else np.array([])),
        "g": _stats(g[mask] if has_pixels else np.array([])),
        "r": _stats(r[mask] if has_pixels else np.array([])),
        "hsv_v": _stats(hsv_v[mask] if has_pixels else np.array([])),
    }


def summarize_roi_result(
    result: RoiDetectionResult,
    detector_type: str,
    mm_per_pixel: float | None,
    expected_area_mm2: float | None,
) -> dict:
    """
    Turns one ROI's independent RoiDetectionResult into the actual
    reported numbers. All measurement happens in the result's crop-local
    coordinates -- nothing here needs full-frame coordinates.

    coverage_percent is computed against expected_area_mm2 (the pad's
    real physical contact area, entered by the user on the ROI -- see
    processing_project.add_roi), NEVER against the ROI rectangle's own
    area. The ROI is a search boundary, deliberately drawn larger than
    the expected pad so it never clips valid contact -- using its area
    as the coverage denominator would silently understate coverage by
    however much margin was drawn in. search_region_area_px is still
    reported, but only ever labeled as the search boundary's size, never
    as a coverage basis. If expected_area_mm2 (or mm_per_pixel) isn't
    available, coverage_percent is None -- never a fabricated 0 or a
    number computed against the wrong denominator.
    """
    possible_mask = result.tier_masks["possible"]
    probable_mask = result.tier_masks["probable"]
    strong_mask = result.tier_masks["strong"]

    possible_px = int(possible_mask.sum())
    probable_px = int(probable_mask.sum())
    strong_px = int(strong_mask.sum())

    search_region_area_px = result.rect_used[2] * result.rect_used[3]

    mm2_per_px = mm_per_pixel**2 if mm_per_pixel is not None else None
    detected_contact_area_mm2 = possible_px * mm2_per_px if mm2_per_px is not None else None

    coverage_percent = (
        (detected_contact_area_mm2 / expected_area_mm2 * 100.0)
        if detected_contact_area_mm2 is not None and expected_area_mm2 not in (None, 0)
        else None
    )

    # Never a fabricated 0 when there's nothing detected to take a ratio
    # of -- "no possible-tier pixels" is a different, more fundamental
    # statement than "0% of the detected pixels reached strong tier".
    strong_fraction = (strong_px / possible_px) if possible_px > 0 else None

    detector_signal_source = "background_difference" if detector_type == "grayscale" else "color_confidence_map"
    detector_signal_values = result.display_map[possible_mask] if possible_px > 0 else np.array([])
    detector_signal = {"source": detector_signal_source, **_stats(detector_signal_values)}

    corrected_frame_signal = _corrected_frame_signal(result.corrected_bgr_crop, possible_mask, detector_type)

    return {
        "possible_area_px": possible_px,
        "probable_area_px": probable_px,
        "strong_area_px": strong_px,
        "detected_contact_area_mm2": detected_contact_area_mm2,
        "expected_area_mm2": expected_area_mm2,
        "coverage_percent": coverage_percent,
        "search_region_area_px": search_region_area_px,  # the ROI's OWN area -- search boundary only, never a coverage basis
        "strong_fraction": strong_fraction,
        "detector_signal": detector_signal,
        "corrected_frame_signal": corrected_frame_signal,
        "clipped": result.clipped,
        "rect_used": result.rect_used,
    }
