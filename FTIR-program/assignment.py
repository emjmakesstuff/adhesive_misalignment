"""
Stage 7: grouping detection.py's fragments into physical circles.

Circle positions are not configured by the user -- they're located from
the fragments themselves, since the physical setup (how many contact
circles, exactly where) can shift between sessions. This module only
ever reads fragment metadata (centroids, pixel coordinates already
inside each fragment's own mask) to compute geometry -- it never writes
to a fragment's mask, dilates it, or creates a pixel that wasn't already
part of some fragment. The "expected circle" is a pure geometric ROI
(center + radius), used for scoring/visualization, never rasterized into
any mask.

Algorithm, per fragment set:
    1. Compute expected_radius_px from circle_config's expected_area_mm2
       and the saved mm_per_pixel scale.
    2. Cluster fragments whose centroids are within CLUSTER_DISTANCE_FACTOR
       * expected_radius_px of each other (union-find over a distance
       graph) -- a coarse first pass at "these probably belong together".
    3. For each raw cluster, compute the area-weighted centroid of its
       fragments as a candidate circle center (equivalent to the true
       combined centroid, since fragment masks never overlap), then drop
       any fragment whose own pixels are mostly (< MIN_ROI_OVERLAP) outside
       an ROI circle of that radius centered there, recomputing the
       centroid after each drop (bounded iterations) -- this is the
       "overlap with the expected circular ROI" refinement.
    4. Rank surviving candidate groups by total illuminated area (most
       real measured evidence first) and take the top expected_count as
       "accepted", the rest as "extra candidate groups". A shortfall in
       exact-count mode is reported as a missing count, never fabricated.

Deliberately NOT used as an accept/reject gate: how close a candidate's
illuminated area comes to expected_area_mm2. A physically real contact
circle can legitimately be illuminated anywhere from ~0% to 100% of its
expected footprint -- that fraction is exactly what measurement.py's
coverage percentage reports, so rejecting low-coverage groups here would
silently hide the very thing this whole pipeline exists to measure.
expected_area_mm2 is used for one thing only: sizing the ROI (ergo the
clustering/overlap radius) -- it represents the physical circle's
footprint, not how much of it is expected to be lit. area_tolerance_pct
is still recorded per group as an informational `within_tolerance` flag
(e.g. "this group is way bigger than one footprint -- maybe it absorbed
two circles' worth of fragments"), shown in the report but never used to
silently drop or reclassify a group.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from detection import Fragment

# How far apart (in multiples of the expected circle's radius) two
# fragment centroids can be and still be considered for the same circle.
# Generous on purpose: refinement (overlap-with-ROI) is what actually
# prunes bad members, this first pass just needs to not miss real ones.
CLUSTER_DISTANCE_FACTOR = 2.5

# A fragment stays in a candidate group only if at least this fraction of
# its own pixels fall inside the group's ROI circle.
MIN_ROI_OVERLAP = 0.3

MAX_REFINEMENT_ITERATIONS = 3


@dataclass
class CircleGroup:
    group_id: int
    fragment_ids: list[int]
    center: tuple[float, float]
    radius_px: float
    status: str  # "accepted" | "extra"
    total_area_mm2: float
    within_tolerance: bool  # informational only -- see module docstring


@dataclass
class AssignmentResult:
    accepted: list[CircleGroup]
    extra: list[CircleGroup]
    missing_count: int
    unassigned_fragment_ids: list[int]
    expected_radius_px: float


class _UnionFind:
    def __init__(self, ids: list[int]) -> None:
        self.parent = {i: i for i in ids}

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def expected_radius_px(expected_area_mm2: float, mm_per_pixel: float) -> float:
    radius_mm = math.sqrt(expected_area_mm2 / math.pi)
    return radius_mm / mm_per_pixel


def _weighted_centroid(fragments: list[Fragment]) -> tuple[float, float]:
    total_area = sum(f.possible_area_px for f in fragments)
    cx = sum(f.centroid[0] * f.possible_area_px for f in fragments) / total_area
    cy = sum(f.centroid[1] * f.possible_area_px for f in fragments) / total_area
    return (cx, cy)


def _fragment_overlap_fraction(fragment: Fragment, center: tuple[float, float], radius_px: float) -> float:
    x, y, w, h = fragment.bbox
    yy, xx = np.mgrid[y : y + h, x : x + w]
    inside_roi = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius_px**2
    inside_fragment_and_roi = np.count_nonzero(inside_roi & fragment.mask)
    return inside_fragment_and_roi / fragment.possible_area_px


def _cluster_by_distance(fragments: list[Fragment], max_distance: float) -> list[list[Fragment]]:
    by_id = {f.id: f for f in fragments}
    uf = _UnionFind([f.id for f in fragments])

    for i, a in enumerate(fragments):
        for b in fragments[i + 1 :]:
            dx = a.centroid[0] - b.centroid[0]
            dy = a.centroid[1] - b.centroid[1]
            if math.hypot(dx, dy) <= max_distance:
                uf.union(a.id, b.id)

    groups: dict[int, list[Fragment]] = {}
    for f in fragments:
        root = uf.find(f.id)
        groups.setdefault(root, []).append(by_id[f.id])

    return list(groups.values())


def _refine_cluster(cluster: list[Fragment], radius_px: float) -> tuple[list[Fragment], tuple[float, float]]:
    current = cluster

    for _ in range(MAX_REFINEMENT_ITERATIONS):
        if not current:
            break

        center = _weighted_centroid(current)
        kept = [f for f in current if _fragment_overlap_fraction(f, center, radius_px) >= MIN_ROI_OVERLAP]

        if len(kept) == len(current):
            return kept, center

        current = kept

    center = _weighted_centroid(current) if current else (0.0, 0.0)
    return current, center


def assign_fragments_to_circles(
    fragments: list[Fragment],
    circle_config: dict,
    mm_per_pixel: float,
) -> AssignmentResult:
    radius_px = expected_radius_px(circle_config["expected_area_mm2"], mm_per_pixel)

    if not fragments:
        return AssignmentResult(
            accepted=[], extra=[], missing_count=circle_config["expected_count"],
            unassigned_fragment_ids=[], expected_radius_px=radius_px,
        )

    raw_clusters = _cluster_by_distance(fragments, CLUSTER_DISTANCE_FACTOR * radius_px)

    expected_area_mm2 = circle_config["expected_area_mm2"]
    tolerance_fraction = circle_config["area_tolerance_pct"] / 100.0

    candidates: list[CircleGroup] = []
    next_group_id = 1

    for cluster in raw_clusters:
        kept, center = _refine_cluster(cluster, radius_px)

        if not kept:
            continue  # every member failed the ROI-overlap test; all fall through to unassigned below

        total_area_px = sum(f.possible_area_px for f in kept)
        total_area_mm2 = total_area_px * (mm_per_pixel**2)

        candidates.append(
            CircleGroup(
                group_id=next_group_id,
                fragment_ids=[f.id for f in kept],
                center=center,
                radius_px=radius_px,
                status="extra",  # provisional; finalized below
                total_area_mm2=total_area_mm2,
                within_tolerance=abs(total_area_mm2 - expected_area_mm2) <= expected_area_mm2 * tolerance_fraction,
            )
        )
        next_group_id += 1

    # Ranked by total illuminated area, most evidence first -- NOT by
    # closeness to expected_area_mm2 (see module docstring: a genuinely
    # real circle can be illuminated anywhere from ~0% to 100% of its
    # expected footprint, so low coverage must never disqualify a group).
    ranked = sorted(candidates, key=lambda c: c.total_area_mm2, reverse=True)

    expected_count = circle_config["expected_count"]
    accepted = ranked[:expected_count]
    extra = ranked[expected_count:]

    for group in accepted:
        group.status = "accepted"
    for group in extra:
        group.status = "extra"

    if circle_config["count_mode"] == "exact":
        missing_count = max(0, expected_count - len(accepted))
    else:
        missing_count = 0

    assigned_ids: set[int] = set()
    for group in accepted + extra:
        assigned_ids.update(group.fragment_ids)

    unassigned_ids = [f.id for f in fragments if f.id not in assigned_ids]

    return AssignmentResult(
        accepted=accepted,
        extra=extra,
        missing_count=missing_count,
        unassigned_fragment_ids=unassigned_ids,
        expected_radius_px=radius_px,
    )
