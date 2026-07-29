"""
Processing project storage: ROI definitions and their Range/Full
detection results, kept entirely separate from recordings/<id>/ (which
stays immutable -- see recording_store.py's own module docstring).

project_id is its own identifier, independent of recording_id, even
though the current UI only ever creates one project per recording (see
get_or_create_project_for_recording) -- multiple projects per recording
is a real possibility this schema doesn't foreclose, even if nothing
exercises it yet.

Layout:
    processing_projects/<project_id>/
        project.json                        -- ROI definitions, references recording_id
        analysis_runs/<analysis_run_id>/
            run.json                        -- one run's full context: requested/completed frame
                                                range, detector config snapshot, ROI definitions
                                                AS USED for this run, status, stop_reason
            roi_results.jsonl               -- one row per (frame, roi), references analysis_run_id
            .incomplete                     -- present only while the run is active; removed only
                                                after run.json's final write succeeds -- a run killed
                                                mid-flight is detectable exactly like a killed
                                                recording (recorder.py/recording_store.py's own
                                                .incomplete pattern)

Each Range/Full detection execution gets its OWN analysis_run_id and its
own roi_results.jsonl -- results from different detector configurations,
or different reruns, are never appended into one shared file (that would
make "which config produced this row" ambiguous without cross-
referencing every row). The full config is stored ONCE in run.json;
individual rows only reference analysis_run_id (+ a short config hash,
for a fast sanity check without opening run.json).

Current-frame preview results are NOT written here at all -- they're
ephemeral/in-memory only in the Processing tab.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).parent
PROCESSING_PROJECTS_DIR = PROJECT_ROOT / "processing_projects"

RUN_STATUSES = ("running", "completed", "failed", "cancelled")


def _new_project_id() -> str:
    now = datetime.datetime.now()
    return f"proj_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _new_roi_id() -> str:
    return f"roi_{uuid.uuid4().hex[:8]}"


def _new_analysis_run_id() -> str:
    now = datetime.datetime.now()
    return f"run_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def config_hash(config: dict) -> str:
    """Short, stable hash of a detection config dict -- embedded in each
    roi_results.jsonl row as a fast sanity check that it still matches
    run.json's stored config, without needing to reopen/parse that file
    on every row."""
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


# ---- project-level paths ----


def project_dir(project_id: str) -> Path:
    return PROCESSING_PROJECTS_DIR / project_id


def project_json_path(project_id: str) -> Path:
    return project_dir(project_id) / "project.json"


# ---- project CRUD ----


def create_project(recording_id: str) -> str:
    project_id = _new_project_id()
    project_dir(project_id).mkdir(parents=True, exist_ok=True)

    project = {
        "project_id": project_id,
        "recording_id": recording_id,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "rois": [],
    }
    _write_project(project_id, project)
    return project_id


def load_project(project_id: str) -> dict | None:
    path = project_json_path(project_id)

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_project(project_id: str, project: dict) -> None:
    with open(project_json_path(project_id), "w", encoding="utf-8") as f:
        json.dump(project, f, indent=2)


def list_projects() -> list[str]:
    """Every project_id on disk, newest first -- for UI that browses
    results across every recording at once (see ResultsTab), unlike
    find_projects_for_recording() below which is scoped to one."""
    if not PROCESSING_PROJECTS_DIR.exists():
        return []

    return sorted((p.name for p in PROCESSING_PROJECTS_DIR.iterdir() if p.is_dir()), reverse=True)


def find_projects_for_recording(recording_id: str) -> list[str]:
    """All project_ids referencing this recording_id, newest first."""
    if not PROCESSING_PROJECTS_DIR.exists():
        return []

    matches = []
    for entry in PROCESSING_PROJECTS_DIR.iterdir():
        if not entry.is_dir():
            continue
        project = load_project(entry.name)
        if project is not None and project.get("recording_id") == recording_id:
            matches.append(entry.name)

    return sorted(matches, reverse=True)


def get_or_create_project_for_recording(recording_id: str) -> str:
    """The UI's current one-project-per-recording policy -- reuses the
    newest existing project for this recording if one exists, else
    creates a fresh one. project_id is still a real, independent
    identifier (see module docstring); this just picks which one the
    single-project UI uses."""
    existing = find_projects_for_recording(recording_id)
    if existing:
        return existing[0]
    return create_project(recording_id)


# ---- ROI CRUD ----


def add_roi(
    project_id: str,
    name: str,
    rectangle: dict,  # {"x", "y", "width", "height"}, corrected-frame pixel coordinates
    reference_frame_number: int,
    start_frame: int,
    end_frame: int,
    corrected_width: int,
    corrected_height: int,
    calibration_snapshot_reference: dict,
    crop_info: dict,
    expected_area_mm2: float | None = None,
    expected_geometry: dict | None = None,
    min_area_mm2: float | None = None,
) -> str:
    """
    expected_area_mm2 is the pad's real physical contact area -- NOT the
    ROI rectangle's area. The ROI is deliberately drawn larger than the
    expected pad so it never clips valid contact (see the ROI-rules
    docstring in the Processing tab); coverage_percent is only ever
    computed against expected_area_mm2, never against the ROI's own
    (search-boundary) area. None here means "not configured" --
    coverage_percent will be null/N/A for this ROI until it is.

    min_area_mm2 is a per-ROI noise floor: a connected component smaller
    than this is dropped entirely (never counted toward possible/
    probable/strong area or coverage) before any other measurement runs
    -- see detection_pipeline.run_roi_detection(). The SAME number is
    interpreted as raw px instead of mm2 whenever this recording has no
    mm_per_pixel scale (mm2 can't be derived without one). None means no
    filtering.
    """
    project = load_project(project_id)
    if project is None:
        raise ValueError(f"No such project: {project_id!r}")

    roi_id = _new_roi_id()
    roi = {
        "roi_id": roi_id,
        "name": name,
        "rectangle": dict(rectangle),
        "reference_frame_number": reference_frame_number,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "corrected_width": corrected_width,
        "corrected_height": corrected_height,
        "calibration_snapshot_reference": dict(calibration_snapshot_reference),
        "crop_info": dict(crop_info),
        "expected_area_mm2": expected_area_mm2,
        "expected_geometry": dict(expected_geometry) if expected_geometry is not None else None,
        "min_area_mm2": min_area_mm2,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    project["rois"].append(roi)
    _write_project(project_id, project)
    return roi_id


def _mutate_roi(project_id: str, roi_id: str, mutate) -> None:
    project = load_project(project_id)
    if project is None:
        raise ValueError(f"No such project: {project_id!r}")

    for roi in project["rois"]:
        if roi["roi_id"] == roi_id:
            mutate(roi)
            _write_project(project_id, project)
            return

    raise ValueError(f"No such ROI: {roi_id!r}")


def rename_roi(project_id: str, roi_id: str, new_name: str) -> None:
    _mutate_roi(project_id, roi_id, lambda roi: roi.__setitem__("name", new_name))


def update_roi_rectangle(project_id: str, roi_id: str, rectangle: dict) -> None:
    _mutate_roi(project_id, roi_id, lambda roi: roi.__setitem__("rectangle", dict(rectangle)))


def update_roi_frame_range(project_id: str, roi_id: str, start_frame: int, end_frame: int) -> None:
    def _update(roi: dict) -> None:
        roi["start_frame"] = start_frame
        roi["end_frame"] = end_frame

    _mutate_roi(project_id, roi_id, _update)


def update_roi_expected_area(project_id: str, roi_id: str, expected_area_mm2: float | None, expected_geometry: dict | None = None) -> None:
    def _update(roi: dict) -> None:
        roi["expected_area_mm2"] = expected_area_mm2
        roi["expected_geometry"] = dict(expected_geometry) if expected_geometry is not None else None

    _mutate_roi(project_id, roi_id, _update)


def update_roi_min_area(project_id: str, roi_id: str, min_area_mm2: float | None) -> None:
    _mutate_roi(project_id, roi_id, lambda roi: roi.__setitem__("min_area_mm2", min_area_mm2))


def delete_roi(project_id: str, roi_id: str) -> None:
    project = load_project(project_id)
    if project is None:
        raise ValueError(f"No such project: {project_id!r}")

    project["rois"] = [r for r in project["rois"] if r["roi_id"] != roi_id]
    _write_project(project_id, project)


def list_rois(project_id: str) -> list[dict]:
    project = load_project(project_id)
    return list(project["rois"]) if project is not None else []


def find_overlapping_pairs(rois: list[dict]) -> list[tuple[str, str]]:
    """Simple axis-aligned rectangle overlap test over every pair. A
    non-empty result means the ROI set is invalid for Range/Full
    analysis (the Processing tab gates those buttons on this) -- current-
    frame preview may still run, but must show a prominent warning and
    must never be presented as a valid final measurement (see the
    Processing tab's ROI rules)."""
    pairs = []
    for i in range(len(rois)):
        for j in range(i + 1, len(rois)):
            a, b = rois[i]["rectangle"], rois[j]["rectangle"]
            ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["width"], a["y"] + a["height"]
            bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                pairs.append((rois[i]["roi_id"], rois[j]["roi_id"]))
    return pairs


# ---- analysis runs ----


def analysis_runs_dir(project_id: str) -> Path:
    return project_dir(project_id) / "analysis_runs"


def analysis_run_dir(project_id: str, analysis_run_id: str) -> Path:
    return analysis_runs_dir(project_id) / analysis_run_id


def run_json_path(project_id: str, analysis_run_id: str) -> Path:
    return analysis_run_dir(project_id, analysis_run_id) / "run.json"


def roi_results_path(project_id: str, analysis_run_id: str) -> Path:
    return analysis_run_dir(project_id, analysis_run_id) / "roi_results.jsonl"


def incomplete_marker_path(project_id: str, analysis_run_id: str) -> Path:
    return analysis_run_dir(project_id, analysis_run_id) / ".incomplete"


def start_analysis_run(
    project_id: str,
    recording_id: str,
    requested_start_frame: int,
    requested_end_frame: int,
    detector_type: str,
    config: dict,
    calibration_reference: dict,
    rois_snapshot: list[dict],
) -> str:
    """
    Creates a new analysis run directory, writes the initial run.json
    (status="running"), and touches .incomplete. rois_snapshot is a COPY
    of the ROI definitions AS THEY WERE at run start -- project.json's
    ROIs could change later (rename, rectangle edit), but a run's own
    record of what it actually used must never drift with them.
    """
    analysis_run_id = _new_analysis_run_id()
    run_dir = analysis_run_dir(project_id, analysis_run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    run = {
        "analysis_run_id": analysis_run_id,
        "project_id": project_id,
        "recording_id": recording_id,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "requested_frame_range": {"start": requested_start_frame, "end": requested_end_frame},
        "completed_frame_range": None,
        "detector_type": detector_type,
        "config": dict(config),
        "config_hash": config_hash(config),
        "calibration_reference": dict(calibration_reference),
        "rois": [dict(roi) for roi in rois_snapshot],
        "status": "running",
        "stop_reason": None,
        "result_row_count": 0,
    }
    _write_run(project_id, analysis_run_id, run)
    incomplete_marker_path(project_id, analysis_run_id).touch()

    return analysis_run_id


def _write_run(project_id: str, analysis_run_id: str, run: dict) -> None:
    with open(run_json_path(project_id, analysis_run_id), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)


def load_run(project_id: str, analysis_run_id: str) -> dict | None:
    path = run_json_path(project_id, analysis_run_id)

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def finalize_analysis_run(
    project_id: str,
    analysis_run_id: str,
    completed_start_frame: int | None,
    completed_end_frame: int | None,
    status: str,
    stop_reason: str,
    result_row_count: int,
) -> None:
    """
    Writes the final run.json and removes .incomplete -- but ONLY if
    status == "completed". A cancelled/failed/crashed run's .incomplete
    marker is left in place deliberately: finalize_analysis_run() itself
    is only ever called from the tab's finally-block (see
    _run_detection_over), so even a run that raised partway through
    still gets a real, honest run.json (status="failed", whatever
    completed_frame_range/result_row_count actually got reached) --
    it just never loses its "don't trust this" marker.
    """
    if status not in RUN_STATUSES:
        raise ValueError(f"Unknown status {status!r}, expected one of {RUN_STATUSES}")

    run = load_run(project_id, analysis_run_id)
    if run is None:
        raise ValueError(f"No such analysis run: {analysis_run_id!r}")

    run["completed_frame_range"] = (
        {"start": completed_start_frame, "end": completed_end_frame}
        if completed_start_frame is not None and completed_end_frame is not None
        else None
    )
    run["status"] = status
    run["stop_reason"] = stop_reason
    run["result_row_count"] = result_row_count
    _write_run(project_id, analysis_run_id, run)

    if status == "completed":
        marker = incomplete_marker_path(project_id, analysis_run_id)
        if marker.exists():
            marker.unlink()


def is_run_incomplete(project_id: str, analysis_run_id: str) -> bool:
    return incomplete_marker_path(project_id, analysis_run_id).exists()


def list_analysis_runs(project_id: str) -> list[str]:
    directory = analysis_runs_dir(project_id)
    if not directory.exists():
        return []
    return sorted((p.name for p in directory.iterdir() if p.is_dir()), reverse=True)


def delete_analysis_run(project_id: str, analysis_run_id: str) -> None:
    """Permanently removes one run's directory (run.json, roi_results.jsonl,
    and any exported files sitting alongside them, e.g. a CSV export) --
    never touches project.json/the ROI definitions, or any other run."""
    directory = analysis_run_dir(project_id, analysis_run_id)
    if directory.exists():
        shutil.rmtree(directory)


# ---- roi_results.jsonl (streamed, append-only, per analysis run) ----


class RoiResultsWriter:
    def __init__(self, project_id: str, analysis_run_id: str) -> None:
        self._path = roi_results_path(project_id, analysis_run_id)
        self._file = None
        self.rows_written = 0

    def __enter__(self) -> "RoiResultsWriter":
        self._file = open(self._path, "a", encoding="utf-8")
        return self

    def write_row(self, row: dict) -> None:
        self._file.write(json.dumps(row) + "\n")
        self.rows_written += 1

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def read_roi_results(project_id: str, analysis_run_id: str) -> Iterator[dict]:
    path = roi_results_path(project_id, analysis_run_id)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
