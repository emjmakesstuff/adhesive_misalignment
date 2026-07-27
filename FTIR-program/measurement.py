"""
Stage 8 (partial): per-circle area coverage.

Deliberately does NOT include normalized 0-100% brightness intensity --
that requires dark-reference/max-reference calibration (Stage 9), not
built yet. What's here is purely geometric: for each circle group
produced by assignment.py, sum its assigned fragments' possible/probable/
strong pixel counts (already computed once in detection.py, never
recomputed or re-thresholded here), convert to mm^2 via the saved scale,
and compare against the expected area.
"""

from __future__ import annotations

from dataclasses import dataclass

from assignment import AssignmentResult, CircleGroup
from detection import Fragment


@dataclass
class CircleMeasurement:
    group_id: int
    status: str  # "accepted" | "extra"
    fragment_ids: list[int]
    center: tuple[float, float]
    radius_px: float
    expected_area_mm2: float
    possible_area_px: int
    probable_area_px: int
    strong_area_px: int
    possible_area_mm2: float
    probable_area_mm2: float
    strong_area_mm2: float
    possible_coverage_pct: float
    probable_coverage_pct: float
    strong_coverage_pct: float


def _measure_group(group: CircleGroup, fragments_by_id: dict[int, Fragment], expected_area_mm2: float, mm_per_pixel: float) -> CircleMeasurement:
    members = [fragments_by_id[fid] for fid in group.fragment_ids]

    possible_px = sum(f.possible_area_px for f in members)
    probable_px = sum(f.probable_area_px for f in members)
    strong_px = sum(f.strong_area_px for f in members)

    mm2_per_px = mm_per_pixel**2
    possible_mm2 = possible_px * mm2_per_px
    probable_mm2 = probable_px * mm2_per_px
    strong_mm2 = strong_px * mm2_per_px

    return CircleMeasurement(
        group_id=group.group_id,
        status=group.status,
        fragment_ids=group.fragment_ids,
        center=group.center,
        radius_px=group.radius_px,
        expected_area_mm2=expected_area_mm2,
        possible_area_px=possible_px,
        probable_area_px=probable_px,
        strong_area_px=strong_px,
        possible_area_mm2=possible_mm2,
        probable_area_mm2=probable_mm2,
        strong_area_mm2=strong_mm2,
        possible_coverage_pct=100.0 * possible_mm2 / expected_area_mm2,
        probable_coverage_pct=100.0 * probable_mm2 / expected_area_mm2,
        strong_coverage_pct=100.0 * strong_mm2 / expected_area_mm2,
    )


def measure_circles(
    fragments: list[Fragment],
    assignment_result: AssignmentResult,
    expected_area_mm2: float,
    mm_per_pixel: float,
) -> list[CircleMeasurement]:
    fragments_by_id = {f.id: f for f in fragments}

    all_groups = assignment_result.accepted + assignment_result.extra
    return [
        _measure_group(group, fragments_by_id, expected_area_mm2, mm_per_pixel)
        for group in all_groups
    ]
