"""
On-disk recording folder structure: creation, metadata/frame-index
read-write, calibration snapshotting, and the disk-space estimator.
Schema read/write only, per Stage A's scope -- Stage B's recorder.py is
what will actually drive frame capture into these primitives; nothing
in this module talks to a camera, a dispatcher, or a VideoWriter.

Layout (recordings and processing projects are physically separate
trees -- no processing code path ever opens a file inside
recordings/<id>/ for writing, except the recorder itself during an
active session and this module's own narrow rename/notes/protect-flag
updates below -- never the video, frame index, or calibration
snapshot):

    recordings/<recording_id>/
        recording.<ext>            -- written by recorder.py, not this module
        metadata.json
        frame_index.jsonl
        preview_original.mp4       -- optional, generated on demand (later stage)
        calibration_snapshot/
            calibration_data.json
            scale_calibration.json
            circle_config.json
            background_reference.npy   -- only if present at recording time
        .incomplete                -- present only while open; removed on verified-clean finalize

A recording's calibration_snapshot/ is populated once, at record start,
by copying the ACTUAL files a profile has right now (see
snapshot_calibration) -- never a live re-resolve later. That's what
makes an old recording stay self-describing even if its source profile
is later edited or deleted (see the "Calibration snapshot" section of
the recording-workflow plan).
"""

from __future__ import annotations

import datetime
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterator

import cv2

import calibration_profiles
import scale as scale_module

PROJECT_ROOT = Path(__file__).parent
RECORDINGS_DIR = PROJECT_ROOT / "recordings"

SOURCES = ("captured", "imported")
QUALITY_MODES = ("lossless", "lossy_prototype")
STOP_REASONS = ("user_stop_button", "hotkey", "app_close", "camera_disconnected", "error", "disk_space")


def _new_recording_id() -> str:
    now = datetime.datetime.now()
    return f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def recording_dir(recording_id: str) -> Path:
    return RECORDINGS_DIR / recording_id


def recording_video_path(recording_id: str, ext: str) -> Path:
    ext = ext if ext.startswith(".") else f".{ext}"
    return recording_dir(recording_id) / f"recording{ext}"


def metadata_path(recording_id: str) -> Path:
    return recording_dir(recording_id) / "metadata.json"


def frame_index_path(recording_id: str) -> Path:
    return recording_dir(recording_id) / "frame_index.jsonl"


def calibration_snapshot_dir(recording_id: str) -> Path:
    return recording_dir(recording_id) / "calibration_snapshot"


def snapshot_distortion_path(recording_id: str) -> Path:
    return calibration_snapshot_dir(recording_id) / "calibration_data.json"


def snapshot_circle_config_path(recording_id: str) -> Path:
    return calibration_snapshot_dir(recording_id) / "circle_config.json"


def snapshot_background_reference_path(recording_id: str) -> Path:
    return calibration_snapshot_dir(recording_id) / "background_reference.npy"


def incomplete_marker_path(recording_id: str) -> Path:
    return recording_dir(recording_id) / ".incomplete"


def video_path_for(recording_id: str, metadata: dict | None = None) -> Path | None:
    """
    Resolves the actual recording.<ext> file for a recording, or None if
    it can't be found. Prefers metadata["camera"]["container_ext"]
    (always set by both recorder.py and import_video() going forward);
    falls back to globbing recording.* for a recording made before that
    field existed. Shared by recordings_tab.py and processing_tab.py so
    both resolve the video file identically.
    """
    if metadata is None:
        metadata = load_metadata(recording_id)
    if metadata is None:
        return None

    ext = metadata.get("camera", {}).get("container_ext")
    if ext:
        candidate = recording_video_path(recording_id, ext)
        if candidate.exists():
            return candidate

    matches = sorted(recording_dir(recording_id).glob("recording.*"))
    return matches[0] if matches else None


# ---- creation ----


def _empty_recording_block() -> dict[str, Any]:
    return {
        "frames_acquired_during_recording": 0,
        "frames_enqueue_attempted": 0,
        "recorder_queue_drops": 0,
        "frames_write_attempted": 0,
        "output_frames_verified": 0,
        "drop_ranges": [],
        "suspected_driver_gaps": [],
        "completed_normally": False,
        "stop_reason": None,
    }


def create_recording(
    camera_info: dict,
    calibration_info: dict,
    source: str = "captured",
    quality_mode: str = "lossless",
    experiment_name: str | None = None,
) -> str:
    """
    Allocates a new recording_id, creates recordings/<id>/ and its
    calibration_snapshot/ subfolder, writes an initial metadata.json
    (the "recording" block starts zeroed -- recorder.py fills in the
    real drop/gap accounting at finalize_recording()) and the
    .incomplete marker. Returns the new recording_id.

    camera_info/calibration_info are the "camera"/"calibration" blocks
    of the metadata schema -- callers build these from the live connect
    state (or from "unknown (imported)" placeholders for an imported
    video); this module only owns the folder/file mechanics.
    """
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}, expected one of {SOURCES}")
    if quality_mode not in QUALITY_MODES:
        raise ValueError(f"Unknown quality_mode {quality_mode!r}, expected one of {QUALITY_MODES}")

    recording_id = _new_recording_id()
    directory = recording_dir(recording_id)
    directory.mkdir(parents=True, exist_ok=False)
    calibration_snapshot_dir(recording_id).mkdir(parents=True, exist_ok=True)

    metadata = {
        "recording_id": recording_id,
        "experiment_name": experiment_name or recording_id,
        "notes": "",
        "protected": False,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "quality_mode": quality_mode,
        "camera": camera_info,
        "calibration": calibration_info,
        "recording": _empty_recording_block(),
    }
    write_metadata(recording_id, metadata)
    incomplete_marker_path(recording_id).touch()

    return recording_id


# ---- calibration snapshot ----


def snapshot_calibration(recording_id: str, profile: dict) -> None:
    """
    Copies the profile's actual calibration files -- as they exist RIGHT
    NOW, at record start -- into this recording's own
    calibration_snapshot/. Processing (a later stage) always loads from
    a recording's own snapshot, never by re-resolving the live profile,
    so a later edit to (or deletion of) the live profile can never
    change what an old recording measures against. Only copies files
    that actually exist -- e.g. background_reference.npy is legitimately
    absent for a color/HSV-detector profile.
    """
    profile_id = profile["id"]
    dest_dir = calibration_snapshot_dir(recording_id)
    dest_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "calibration_data.json": calibration_profiles.distortion_path(profile_id),
        "scale_calibration.json": calibration_profiles.scale_path(profile_id),
        "circle_config.json": calibration_profiles.circle_config_path(profile_id),
        "background_reference.npy": calibration_profiles.background_reference_path(profile_id),
    }
    for dest_name, src_path in sources.items():
        if src_path.exists():
            shutil.copyfile(src_path, dest_dir / dest_name)


# ---- metadata read/write ----


def write_metadata(recording_id: str, metadata: dict) -> None:
    with open(metadata_path(recording_id), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(recording_id: str) -> dict | None:
    path = metadata_path(recording_id)

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def update_recording_block(recording_id: str, recording_block: dict) -> None:
    """Called by recorder.py (during, and at the end of, an active
    session) with the real drop/gap accounting -- see the recording-
    workflow plan's "Drop/gap accounting" section for the field shapes.
    Replaces the whole "recording" block in one write."""
    metadata = load_metadata(recording_id)

    if metadata is None:
        raise ValueError(f"No such recording: {recording_id!r}")

    metadata["recording"] = recording_block
    write_metadata(recording_id, metadata)


def finalize_recording(recording_id: str, recording_block: dict) -> None:
    """
    Writes the final recording block and removes the .incomplete marker.
    Callers must only call this once the output file has already been
    verified-clean (output_frames_verified computed via a real
    sequential re-decode) -- never on a path that might still fail. A
    recording folder that still has an .incomplete marker after the app
    restarts is, by construction, one that was killed mid-session (see
    is_incomplete()).
    """
    update_recording_block(recording_id, recording_block)
    marker = incomplete_marker_path(recording_id)

    if marker.exists():
        marker.unlink()


def is_incomplete(recording_id: str) -> bool:
    return incomplete_marker_path(recording_id).exists()


def sequential_decode_count(video_path: Path) -> int:
    """Reopens a video file and sequentially DECODES every frame to get
    a real, trustworthy frame count -- never trust CAP_PROP_FRAME_COUNT
    alone, an unreliable container estimate for some codec/container
    combinations. Used both by recorder.py's post-hoc verification and
    by import_video() below."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


# ---- import external video (Stage C) ----


def import_video(source_path: Path, profile: dict | None, experiment_name: str | None = None) -> str:
    """
    Copies an existing video file into a new recordings/<id>/ folder --
    a real copy, not a path reference, keeping the same immutability/
    self-containment guarantee a captured recording has (a referenced-
    in-place mode can be added later if storage becomes a real
    constraint; not needed now).

    profile is the calibration_profiles profile dict the caller has
    already confirmed this footage corresponds to (resolution/mismatch
    warnings are the CALLER's job -- e.g. the Recordings tab's Import
    dialog -- this function only owns the folder/file mechanics), or
    None for "no profile / unknown".

    Every field that genuinely cannot be known for footage this app
    never captured (device_path, exposure, drop/gap accounting, ...) is
    stored as null, with source="imported" as the flag callers check to
    render "unknown (imported)" rather than a fabricated default (see
    the module docstring's schema notes). quality_mode is always
    "lossy_prototype" -- this app has no way to verify an externally-
    sourced file's lossless fidelity, so it is never silently treated as
    measurement-grade.
    """
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open {source_path} as a video file")

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = (
        "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)) if fourcc_int else None
    )
    cap.release()

    camera_info = {
        "device_path": None,
        "alias": profile.get("alias") if profile is not None else None,
        "camera_role": profile.get("camera_role") if profile is not None else None,
        "source_type": profile.get("source_type") if profile is not None else None,
        "opencv_index_at_session": None,
        "requested_width": None,
        "requested_height": None,
        "requested_fourcc": None,
        "actual_width": actual_width,
        "actual_height": actual_height,
        "requested_fps": None,
        "measured_fps": fps if fps > 0 else None,
        "exposure_ms": None,
        "gain": None,
        "other_controls": {},
        "codec_fourcc": fourcc_str,
        "container_ext": source_path.suffix,
    }

    mm_per_pixel = None
    if profile is not None:
        saved_scale = scale_module.load_scale_calibration(path=calibration_profiles.scale_path(profile["id"]))
        mm_per_pixel = saved_scale["mm_per_pixel"] if saved_scale is not None else None

    calibration_info = {
        "profile_id": profile["id"] if profile is not None else None,
        "profile_version_at_recording": profile.get("profile_version") if profile is not None else None,
        "calibration_model": profile.get("calibration_model") if profile is not None else None,
        "detector_type": profile.get("detector_type") if profile is not None else None,
        "usable_roi_crop_percentages": profile.get("crop_percentages", {}) if profile is not None else {},
        "mm_per_pixel": mm_per_pixel,
        "snapshot_dir": "calibration_snapshot/",
    }

    recording_id = create_recording(
        camera_info=camera_info,
        calibration_info=calibration_info,
        source="imported",
        quality_mode="lossy_prototype",
        experiment_name=experiment_name or source_path.stem,
    )

    if profile is not None:
        snapshot_calibration(recording_id, profile)

    dest_path = recording_video_path(recording_id, source_path.suffix)
    shutil.copyfile(source_path, dest_path)

    # frame_index.jsonl derived from the container's own frame timing --
    # host_receive_monotonic_ns has no real acquisition-time meaning for
    # footage this app never captured, so it's null on every row.
    # source_sequence_id has no real dispatcher sequence to reference
    # either -- the frame's own position is the most sensible analog.
    # One combined decode pass produces both the row data and the real
    # (sequentially-decoded, not container-estimate) frame count.
    frame_count = 0
    decode_cap = cv2.VideoCapture(str(dest_path))
    with FrameIndexWriter(recording_id) as writer:
        while True:
            ok, _ = decode_cap.read()
            if not ok:
                break
            recording_time_ns = int(round((frame_count / fps) * 1e9)) if fps and fps > 0 else None
            writer.write_row(
                output_frame_number=frame_count,
                source_sequence_id=frame_count,
                host_receive_monotonic_ns=None,
                recording_time_ns=recording_time_ns,
            )
            frame_count += 1
    decode_cap.release()

    recording_block = {
        "frames_acquired_during_recording": None,
        "frames_enqueue_attempted": None,
        "recorder_queue_drops": None,
        "frames_write_attempted": frame_count,
        "output_frames_verified": frame_count,
        "drop_ranges": [],
        "suspected_driver_gaps": [],
        "completed_normally": True,
        "stop_reason": None,
    }
    finalize_recording(recording_id, recording_block)

    return recording_id


# ---- frame_index.jsonl (streamed, append-only) ----


class FrameIndexWriter:
    """
    Streams frame_index rows to disk as they happen -- never held fully
    in memory, since a real recording can have tens of thousands of
    rows. Used as a context manager by recorder.py, one per active
    recording:

        with FrameIndexWriter(recording_id) as w:
            w.write_row(output_frame_number=0, source_sequence_id=142,
                        host_receive_monotonic_ns=..., recording_time_ns=0)
    """

    def __init__(self, recording_id: str) -> None:
        self._path = frame_index_path(recording_id)
        self._file = None

    def __enter__(self) -> "FrameIndexWriter":
        self._file = open(self._path, "a", encoding="utf-8")
        return self

    def write_row(
        self,
        output_frame_number: int,
        source_sequence_id: int,
        host_receive_monotonic_ns: int | None,
        recording_time_ns: int | None,
    ) -> None:
        row = {
            "output_frame_number": output_frame_number,
            "source_sequence_id": source_sequence_id,
            "host_receive_monotonic_ns": host_receive_monotonic_ns,
            "recording_time_ns": recording_time_ns,
        }
        self._file.write(json.dumps(row) + "\n")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def read_frame_index(recording_id: str) -> Iterator[dict]:
    """Yields frame_index.jsonl rows one at a time -- a real recording
    can have tens of thousands of rows, so this is a generator, not a
    list-returning function, matching the file's own streamed-write
    design. Downstream consumers (Results graphs, later stages) should
    read recording_time_ns directly for the x-axis, never
    frame_number / requested_fps."""
    path = frame_index_path(recording_id)

    if not path.exists():
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---- listing ----


def list_recordings() -> list[str]:
    """Recording ids present on disk, newest first -- recording_id's own
    timestamp prefix sorts chronologically as a plain string sort."""
    if not RECORDINGS_DIR.exists():
        return []

    return sorted(
        (p.name for p in RECORDINGS_DIR.iterdir() if p.is_dir() and metadata_path(p.name).exists()),
        reverse=True,
    )


# ---- narrow recording-folder mutations (see module docstring) ----


def rename_recording(recording_id: str, new_experiment_name: str) -> None:
    metadata = load_metadata(recording_id)

    if metadata is None:
        raise ValueError(f"No such recording: {recording_id!r}")

    metadata["experiment_name"] = new_experiment_name
    write_metadata(recording_id, metadata)


def update_recording_notes(recording_id: str, notes: str) -> None:
    metadata = load_metadata(recording_id)

    if metadata is None:
        raise ValueError(f"No such recording: {recording_id!r}")

    metadata["notes"] = notes
    write_metadata(recording_id, metadata)


def set_recording_protected(recording_id: str, protected: bool) -> None:
    metadata = load_metadata(recording_id)

    if metadata is None:
        raise ValueError(f"No such recording: {recording_id!r}")

    metadata["protected"] = protected
    write_metadata(recording_id, metadata)


def delete_recording(recording_id: str) -> None:
    """Mechanical removal only -- the confirmation dialog itself belongs
    to a later UI stage. Refuses to delete a protected recording."""
    metadata = load_metadata(recording_id)

    if metadata is not None and metadata.get("protected"):
        raise ValueError(f"Recording {recording_id!r} is protected -- unprotect it first.")

    directory = recording_dir(recording_id)
    if directory.exists():
        shutil.rmtree(directory)


# ---- disk-space protection ----


class DiskSpaceEstimator:
    """
    The dynamic safety-floor calculation:

        safety_floor_bytes = max(
            absolute_minimum_bytes,
            queued_unwritten_bytes + bytes_per_second * headroom_seconds + finalization_margin_bytes,
        )

    One instance per active recording -- bytes_per_second is a running
    measurement specific to that session's actual encode output, with a
    caller-supplied fallback for the first few seconds before a real
    rate is trustworthy.
    """

    ABSOLUTE_MINIMUM_BYTES = 100 * 1024 * 1024  # 100MB backstop for a degenerate early-session estimate
    DEFAULT_HEADROOM_SECONDS = 15.0  # within the 10-20s guidance
    FINALIZATION_MARGIN_BYTES = 20 * 1024 * 1024  # room for container index/trailer writes on close

    def __init__(
        self,
        fallback_bytes_per_second: float,
        headroom_seconds: float = DEFAULT_HEADROOM_SECONDS,
        finalization_margin_bytes: int = FINALIZATION_MARGIN_BYTES,
        absolute_minimum_bytes: int = ABSOLUTE_MINIMUM_BYTES,
    ) -> None:
        self._fallback_bytes_per_second = fallback_bytes_per_second
        self.headroom_seconds = headroom_seconds
        self.finalization_margin_bytes = finalization_margin_bytes
        self.absolute_minimum_bytes = absolute_minimum_bytes

    def bytes_per_second(self, measured_bytes_written: int, measured_seconds_elapsed: float) -> float:
        """The live estimate: real measured output so far this session,
        falling back to the caller's worst-case benchmark estimate for
        the first few seconds before a real rate is trustworthy."""
        if measured_seconds_elapsed < 3.0 or measured_bytes_written <= 0:
            return self._fallback_bytes_per_second

        return measured_bytes_written / measured_seconds_elapsed

    def safety_floor_bytes(self, queued_unwritten_bytes: int, bytes_per_second: float) -> int:
        dynamic = queued_unwritten_bytes + bytes_per_second * self.headroom_seconds + self.finalization_margin_bytes
        return int(max(self.absolute_minimum_bytes, dynamic))

    def check(self, path: Path, queued_unwritten_bytes: int, bytes_per_second: float) -> dict:
        """Returns free/floor/ok/estimated-remaining-seconds -- callers
        (the Stage B recorder, a pre-record warning dialog) decide what
        to do with `ok` being False; this makes no UI decision itself."""
        path.mkdir(parents=True, exist_ok=True)  # shutil.disk_usage needs an existing path -- harmless pre-first-recording
        free_bytes = shutil.disk_usage(path).free
        floor_bytes = self.safety_floor_bytes(queued_unwritten_bytes, bytes_per_second)
        estimated_seconds_remaining = (free_bytes / bytes_per_second) if bytes_per_second > 0 else float("inf")

        return {
            "free_bytes": free_bytes,
            "floor_bytes": floor_bytes,
            "ok": free_bytes >= floor_bytes,
            "estimated_seconds_remaining": estimated_seconds_remaining,
        }


def has_minimum_recording_headroom(
    path: Path, estimated_bytes_per_second: float, minimum_duration_seconds: float = 120.0
) -> bool:
    """
    Pre-record check: True unless less than minimum_duration_seconds of
    recording headroom remains at the estimated rate -- a warning
    condition, not by itself a hard block (the caller decides how to
    surface it). Distinct from DiskSpaceEstimator.check(), which runs
    DURING an active recording against the dynamic floor; this one is a
    simple go/no-go gate before Start Recording is even pressed, with no
    session state yet to draw a real bytes_per_second from.
    """
    path.mkdir(parents=True, exist_ok=True)  # shutil.disk_usage needs an existing path -- harmless pre-first-recording
    free_bytes = shutil.disk_usage(path).free

    if estimated_bytes_per_second <= 0:
        return True

    return free_bytes >= estimated_bytes_per_second * minimum_duration_seconds
