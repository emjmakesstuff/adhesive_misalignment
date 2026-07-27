"""
Stage 6: multi-level bright-region ("fragment") detection.

Agnostic to what single-channel array it's given -- raw undistorted
grayscale, or (as the Detection tab now uses) a nonnegative background-
subtracted difference image from background_reference.py. Either way, a
fragment is a single connected component of pixels at or above the
*possible* (faintest) threshold, found once, directly on that array via
cv2.connectedComponentsWithStats, unblurred. Its mask, contour, centroid,
and bounding box are fixed at that point. No assumption anywhere in this
module that a fragment is circular or any particular shape -- contours
are whatever cv2.findContours actually traces.

The probable/strong thresholds do NOT run a second/third detection pass
-- they re-examine the exact same raw pixels already inside that
fragment's mask and count how many also clear the stricter cut point.
probable_area_px and strong_area_px are therefore always subsets of
possible_area_px (the fragment's own pixel count), never a grown or
merged region -- this is what guarantees no dark gap between two
separate fragments is ever bridged: connectivity is computed exactly
once, from real thresholded pixels, with no blur and no morphological
closing/dilation anywhere in this module.

Every fragment found is kept, including single-pixel ones -- filtering
"small" fragments is a job for the caller (grouping/reporting), not
detection, since discarding here would silently throw away real pixel
evidence the rest of the pipeline is supposed to account for.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Fragment:
    id: int
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in the detected frame
    mask: np.ndarray  # bool, shape (h, w), local to bbox -- the fragment's own pixels, untouched
    contour: np.ndarray  # Nx1x2 int32, full-frame coordinates
    centroid: tuple[float, float]  # full-frame (x, y)
    possible_area_px: int
    probable_area_px: int
    strong_area_px: int
    mean_brightness: float
    median_brightness: float
    p95_brightness: float
    max_brightness: float
    min_brightness: float


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """
    Returns a single-channel view for detection without ever modifying
    `frame` itself -- cv2.cvtColor always returns a new array, and a 2D
    input is returned as-is (still the same object, but nothing here
    writes into it).
    """
    if frame.ndim == 2:
        return frame

    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def detect_fragments(
    frame: np.ndarray,
    possible_threshold: int,
    probable_threshold: int,
    strong_threshold: int,
) -> list[Fragment]:
    if not (possible_threshold <= probable_threshold <= strong_threshold):
        raise ValueError(
            f"Thresholds must be non-decreasing (possible <= probable <= "
            f"strong), got {possible_threshold}, {probable_threshold}, "
            f"{strong_threshold}."
        )

    gray = to_grayscale(frame)

    # New array -- does not touch `gray`/`frame`. This is the one and
    # only place connectivity is decided; every fragment's identity comes
    # from this single thresholded mask.
    possible_mask = (gray >= possible_threshold).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        possible_mask, connectivity=8
    )

    fragments: list[Fragment] = []

    for label in range(1, num_labels):  # label 0 is background
        x, y, w, h, area = stats[label]

        local_labels = labels[y : y + h, x : x + w]
        local_mask = local_labels == label
        local_gray = gray[y : y + h, x : x + w]

        pixel_values = local_gray[local_mask]

        contours, _ = cv2.findContours(
            local_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # A single connected component can still produce more than one
        # external contour in rare cases (e.g. a one-pixel-wide neck); the
        # largest by point count is kept as this fragment's representative
        # outline for display -- it never affects the pixel-based area/
        # brightness stats above, which use the full mask regardless.
        contour = max(contours, key=len) if contours else np.empty((0, 1, 2), dtype=np.int32)
        contour = contour + np.array([x, y], dtype=np.int32)

        fragments.append(
            Fragment(
                id=label,
                bbox=(int(x), int(y), int(w), int(h)),
                mask=local_mask,
                contour=contour,
                centroid=(float(centroids[label][0]), float(centroids[label][1])),
                possible_area_px=int(area),
                probable_area_px=int(np.count_nonzero(pixel_values >= probable_threshold)),
                strong_area_px=int(np.count_nonzero(pixel_values >= strong_threshold)),
                mean_brightness=float(np.mean(pixel_values)),
                median_brightness=float(np.median(pixel_values)),
                p95_brightness=float(np.percentile(pixel_values, 95)),
                max_brightness=float(np.max(pixel_values)),
                min_brightness=float(np.min(pixel_values)),
            )
        )

    return fragments
