"""
Stage 4: physical scale calibration (mm-per-pixel).

Two-point-click + known-distance approach: the user marks two points a
known real-world distance apart (a ruler, or two marks of known
separation) on an *undistorted* frame, and this converts that into a
mm-per-pixel scale factor.

Must run on the undistorted frame -- distortion.py's
load_calibration()/undistort_with_maps() already exist for this. A scale
computed on the raw distorted image would be wrong, since lens distortion
changes apparent distances non-uniformly across the frame.

This is a simple linear scale (not a full homography/perspective
rectification) -- accurate as long as the measured surface is roughly
fronto-parallel to the camera and at the same distance/plane as whatever
is measured later using this scale, which matches the actual FTIR
contact-surface setup this is built for.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

DEFAULT_SCALE_PATH = Path(__file__).parent / "scale_calibration.json"


def pixel_distance(point1: tuple[float, float], point2: tuple[float, float]) -> float:
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])


def compute_scale(
    point1: tuple[float, float],
    point2: tuple[float, float],
    known_distance_mm: float,
    image_size: tuple[int, int],
) -> dict:
    """
    point1/point2 are pixel coordinates in the undistorted frame the
    points were clicked on. Raises ValueError if the two points coincide
    (zero pixel distance) or known_distance_mm isn't positive -- both
    would produce a meaningless or divide-by-zero scale.
    """
    if known_distance_mm <= 0:
        raise ValueError(f"known_distance_mm must be positive, got {known_distance_mm}")

    distance_px = pixel_distance(point1, point2)

    if distance_px <= 0:
        raise ValueError("The two points are identical (zero pixel distance) -- click two distinct points.")

    return {
        "mm_per_pixel": known_distance_mm / distance_px,
        "pixel_distance": distance_px,
        "known_distance_mm": known_distance_mm,
        "point1": list(point1),
        "point2": list(point2),
        "image_size": list(image_size),
    }


def measure_distance_mm(
    point1: tuple[float, float],
    point2: tuple[float, float],
    mm_per_pixel: float,
) -> float:
    """
    The inverse of compute_scale(): given an already-known mm_per_pixel,
    compute the real-world distance between two newly-clicked points.
    Used to verify a saved calibration -- click two points of a distance
    you separately know, and check the result matches.
    """
    return pixel_distance(point1, point2) * mm_per_pixel


def save_scale_calibration(data: dict[str, Any], path: Path = DEFAULT_SCALE_PATH) -> None:
    with open(Path(path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_scale_calibration(path: Path = DEFAULT_SCALE_PATH) -> dict | None:
    path = Path(path)

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
