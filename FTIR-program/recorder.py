"""
Recorder: subscribes to main_window.frame_dispatcher and writes frames
to disk for the duration of one recording session. Built on Stage A's
frame_dispatcher.py / threaded_camera_source.py / recording_store.py --
this is the piece that actually drives a cv2.VideoWriter, which none of
those did.

Codec is chosen automatically from the active profile's camera_role,
never asked for directly -- see CODEC_BY_ROLE. This mapping is the
direct, evidence-based result of real-hardware validation against real
footage from both cameras (not synthetic benchmarks):
    monochrome_ftir (U20CAM) -> FFV1  -- HFYU corrupts grayscale content by +/-1 (round-trip tested)
    color_ftir (ELP)         -> HFYU  -- FFV1's color encoder can't keep up (67-71% frame-drop rate
                                          measured at both 1080p30 and 720p60); HFYU keeps up with
                                          zero drops and round-trips real ELP footage byte-identical.
"lossy_prototype" quality mode overrides this for EITHER role -> MJPG,
tagged quality_mode="lossy_prototype" in metadata.json and never treated
as measurement-grade.

Scoped strictly to the active recording period: this object subscribes
to the dispatcher only between start() and stop() -- frames the
dispatcher publishes while a camera is merely connected but not
recording are never seen by this module at all, so
frames_acquired_during_recording is honest by construction, not an
after-the-fact filtered count. The dispatcher's own session-safety check
(FrameDispatcher.publish() rejects stale-session frames before any
subscriber ever sees them) is the primary guard against a mid-recording
camera switch leaking frames into the wrong recording; this module's
own source_session_id check in _on_frame is defense in depth on top of
that, not the only thing preventing it.
"""

from __future__ import annotations

import queue
import shutil
import threading
import time

import cv2

import recording_store
import scale as scale_module
from calibration_profiles import scale_path
from frame_dispatcher import FrameDispatcher, FrameEvent

CODEC_BY_ROLE = {
    "monochrome_ftir": {"fourcc": "FFV1", "is_color": False, "ext": ".avi"},
    "color_ftir": {"fourcc": "HFYU", "is_color": True, "ext": ".avi"},
}
LOSSY_PROTOTYPE_CODEC = {"fourcc": "MJPG", "is_color": True, "ext": ".avi"}

# Real-hardware measured rates (see the Stage A recording-format
# validation) -- used only as DiskSpaceEstimator's fallback before a
# real per-session measurement exists, and to size the pre-record
# warning. Real recordings switch to their own live-measured rate
# within a few seconds (see DiskSpaceEstimator.bytes_per_second).
FALLBACK_BYTES_PER_SECOND_BY_ROLE = {
    "monochrome_ftir": 3_500_000,  # ~0.2 GB/min measured: FFV1, U20CAM, real footage
    "color_ftir": 47_500_000,  # ~2.85 GB/min measured: HFYU, ELP, real footage
}
DEFAULT_FALLBACK_BYTES_PER_SECOND = 5_000_000

QUEUE_MAXSIZE = 128
DISK_CHECK_INTERVAL_S = 5.0
GAP_HEURISTIC_MULTIPLIER = 3.0


def codec_for(camera_role: str | None, quality_mode: str) -> dict:
    """The single place codec selection happens -- never left to the
    caller. Returns a copy so callers can't accidentally mutate the
    shared constant dicts above."""
    if quality_mode == "lossy_prototype":
        return dict(LOSSY_PROTOTYPE_CODEC)

    config = CODEC_BY_ROLE.get(camera_role)
    if config is None:
        raise ValueError(
            f"No validated lossless codec for camera_role={camera_role!r} -- only "
            f"{list(CODEC_BY_ROLE)} have been round-trip/throughput validated against real hardware. "
            f"Use quality_mode='lossy_prototype' if this role genuinely has no lossless mapping yet."
        )
    return dict(config)


def estimated_gb_per_minute(camera_role: str | None, quality_mode: str) -> float:
    """Drives the UI's size warning -- see FALLBACK_BYTES_PER_SECOND_BY_ROLE."""
    if quality_mode == "lossy_prototype":
        bps = DEFAULT_FALLBACK_BYTES_PER_SECOND
    else:
        bps = FALLBACK_BYTES_PER_SECOND_BY_ROLE.get(camera_role, DEFAULT_FALLBACK_BYTES_PER_SECOND)
    return (bps * 60.0) / 1e9


class Recorder:
    def __init__(self, main_window) -> None:
        self.main_window = main_window
        self._dispatcher: FrameDispatcher = main_window.frame_dispatcher

        self._recording_id: str | None = None
        self._session_id: str | None = None
        self._codec: dict | None = None
        self._writer: cv2.VideoWriter | None = None
        self._frame_index_writer: recording_store.FrameIndexWriter | None = None

        self._queue: "queue.Queue[FrameEvent]" = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._writer_thread: threading.Thread | None = None
        self._stop_writer = threading.Event()

        self._counters_lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._drop_ranges: list[dict] = []
        self._current_drop_range: dict | None = None
        self._output_frame_number = 0

        self._recording_start_monotonic_ns: int | None = None
        self._disk_estimator: recording_store.DiskSpaceEstimator | None = None
        self._last_disk_check_time = 0.0

        # Checked as the very first thing in _on_frame, before it touches
        # any counter or the queue. Closes a real (if narrow) race: the
        # dispatcher snapshots its subscriber list under lock and calls
        # each callback AFTER releasing it (see frame_dispatcher.py), so a
        # publish() already in flight when stop() calls unsubscribe() can
        # still invoke _on_frame once more. Flipping this flag FIRST in
        # stop() -- before unsubscribe() even runs -- means that straggler
        # call becomes a no-op instead of incrementing
        # frames_enqueue_attempted for a frame nothing will ever drain.
        self._accepting_frames = False
        self._is_recording = False

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def recording_id(self) -> str | None:
        return self._recording_id

    # ---- pre-record check ----

    def check_headroom(self, camera_role: str | None, quality_mode: str, minimum_duration_seconds: float = 120.0) -> dict:
        """For a pre-record warning dialog -- informational, never a hard
        block (see the plan's disk-space-protection section)."""
        fallback_bps = (
            DEFAULT_FALLBACK_BYTES_PER_SECOND
            if quality_mode == "lossy_prototype"
            else FALLBACK_BYTES_PER_SECOND_BY_ROLE.get(camera_role, DEFAULT_FALLBACK_BYTES_PER_SECOND)
        )
        recording_store.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        ok = recording_store.has_minimum_recording_headroom(
            recording_store.RECORDINGS_DIR, fallback_bps, minimum_duration_seconds
        )
        free_bytes = shutil.disk_usage(recording_store.RECORDINGS_DIR).free
        estimated_minutes = (free_bytes / fallback_bps) / 60.0 if fallback_bps > 0 else float("inf")

        return {
            "ok": ok,
            "free_bytes": free_bytes,
            "estimated_minutes": estimated_minutes,
            "gb_per_minute": (fallback_bps * 60.0) / 1e9,
        }

    # ---- start ----

    def start(self, profile: dict, camera_info: dict, quality_mode: str, source_session_id: str) -> str:
        if self._is_recording:
            raise RuntimeError("Recorder.start() called while already recording -- call stop() first.")

        codec = codec_for(profile.get("camera_role"), quality_mode)

        camera_info = dict(camera_info)
        camera_info["codec_fourcc"] = codec["fourcc"]
        camera_info["container_ext"] = codec["ext"]

        saved_scale = scale_module.load_scale_calibration(path=scale_path(profile["id"]))
        calibration_info = {
            "profile_id": profile["id"],
            "profile_version_at_recording": profile.get("profile_version", 1),
            "calibration_model": profile.get("calibration_model"),
            "detector_type": profile.get("detector_type"),
            "usable_roi_crop_percentages": profile.get("crop_percentages", {}),
            "mm_per_pixel": saved_scale["mm_per_pixel"] if saved_scale is not None else None,
            "snapshot_dir": "calibration_snapshot/",
        }

        recording_id = recording_store.create_recording(
            camera_info=camera_info,
            calibration_info=calibration_info,
            source="captured",
            quality_mode=quality_mode,
        )
        recording_store.snapshot_calibration(recording_id, profile)

        actual_width = camera_info.get("actual_width")
        actual_height = camera_info.get("actual_height")
        requested_fps = camera_info.get("requested_fps") or 30

        video_path = recording_store.recording_video_path(recording_id, codec["ext"])
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*codec["fourcc"]),
            float(requested_fps),
            (int(actual_width), int(actual_height)),
            isColor=codec["is_color"],
        )
        if not writer.isOpened():
            # Never actually started -- nothing worth keeping.
            recording_store.delete_recording(recording_id)
            raise RuntimeError(f"Failed to open VideoWriter for {video_path} with codec {codec['fourcc']}")

        self._recording_id = recording_id
        self._session_id = source_session_id
        self._codec = codec
        self._writer = writer

        self._counters = {
            "frames_acquired_during_recording": 0,
            "frames_enqueue_attempted": 0,
            "recorder_queue_drops": 0,
            "frames_write_attempted": 0,
        }
        self._drop_ranges = []
        self._current_drop_range = None
        self._output_frame_number = 0
        self._recording_start_monotonic_ns = time.perf_counter_ns()

        fallback_bps = FALLBACK_BYTES_PER_SECOND_BY_ROLE.get(profile.get("camera_role"), DEFAULT_FALLBACK_BYTES_PER_SECOND)
        self._disk_estimator = recording_store.DiskSpaceEstimator(fallback_bytes_per_second=fallback_bps)
        self._last_disk_check_time = time.perf_counter()

        self._frame_index_writer = recording_store.FrameIndexWriter(recording_id)
        self._frame_index_writer.__enter__()

        self._stop_writer.clear()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        self._accepting_frames = True
        self._dispatcher.subscribe("recorder", self._on_frame)
        self._is_recording = True

        return recording_id

    # ---- dispatcher callback (runs on the CAPTURE thread -- must stay fast) ----

    def _on_frame(self, event: FrameEvent) -> None:
        if not self._accepting_frames:
            return  # stop() already began -- see the flag's own comment in __init__
        if event.source_session_id != self._session_id:
            return  # defense in depth -- see module docstring

        with self._counters_lock:
            self._counters["frames_acquired_during_recording"] += 1
            self._counters["frames_enqueue_attempted"] += 1

        try:
            self._queue.put_nowait(event)
            self._close_drop_range()
        except queue.Full:
            with self._counters_lock:
                self._counters["recorder_queue_drops"] += 1
            self._extend_drop_range(event.sequence_id, event.host_receive_monotonic_ns)

    def _extend_drop_range(self, sequence_id: int, host_receive_monotonic_ns: int) -> None:
        with self._counters_lock:
            if self._current_drop_range is not None and sequence_id == self._current_drop_range["last_sequence_id"] + 1:
                self._current_drop_range["last_sequence_id"] = sequence_id
                self._current_drop_range["count"] += 1
                return

            if self._current_drop_range is not None:
                self._drop_ranges.append(self._current_drop_range)

            approx_recording_time_ns = (
                host_receive_monotonic_ns - self._recording_start_monotonic_ns
                if self._recording_start_monotonic_ns is not None
                else None
            )
            self._current_drop_range = {
                "first_sequence_id": sequence_id,
                "last_sequence_id": sequence_id,
                "count": 1,
                "reason": "recorder_queue_full",
                "approx_recording_time_ns": approx_recording_time_ns,
            }

    def _close_drop_range(self) -> None:
        with self._counters_lock:
            if self._current_drop_range is not None:
                self._drop_ranges.append(self._current_drop_range)
                self._current_drop_range = None

    # ---- writer thread ----

    def _write_event(self, event: FrameEvent) -> None:
        """Writes one frame + its frame_index row. Shared by the writer
        thread's normal loop and stop()'s post-join safety-net drain
        below, so both paths account for a written frame identically."""
        frame = event.frame
        if not self._codec["is_color"]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self._writer.write(frame)

        recording_time_ns = (
            event.host_receive_monotonic_ns - self._recording_start_monotonic_ns
            if self._recording_start_monotonic_ns is not None
            else None
        )

        with self._counters_lock:
            self._counters["frames_write_attempted"] += 1
            output_frame_number = self._output_frame_number
            self._output_frame_number += 1

        self._frame_index_writer.write_row(
            output_frame_number=output_frame_number,
            source_sequence_id=event.sequence_id,
            host_receive_monotonic_ns=event.host_receive_monotonic_ns,
            recording_time_ns=recording_time_ns,
        )

    def _writer_loop(self) -> None:
        while not (self._stop_writer.is_set() and self._queue.empty()):
            try:
                event = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._write_event(event)

    # ---- live status / disk-space check (call from the GUI thread) ----

    def status(self) -> dict | None:
        if not self._is_recording:
            return None

        with self._counters_lock:
            counters = dict(self._counters)

        elapsed_s = (time.perf_counter_ns() - self._recording_start_monotonic_ns) / 1e9
        video_path = recording_store.recording_video_path(self._recording_id, self._codec["ext"])
        bytes_written = video_path.stat().st_size if video_path.exists() else 0

        return {
            "recording_id": self._recording_id,
            "elapsed_s": elapsed_s,
            "bytes_written": bytes_written,
            "queue_depth": self._queue.qsize(),
            **counters,
        }

    def check_disk_space(self) -> dict | None:
        """Call on an interval (e.g. every tick of LiveCameraTab's own
        status timer -- internally throttled to DISK_CHECK_INTERVAL_S, so
        calling it more often than that is harmless) while recording.
        Triggers an automatic stop(stop_reason="disk_space") if free
        space has dropped below the dynamic safety floor."""
        if not self._is_recording or self._disk_estimator is None:
            return None

        now = time.perf_counter()
        if now - self._last_disk_check_time < DISK_CHECK_INTERVAL_S:
            return None
        self._last_disk_check_time = now

        video_path = recording_store.recording_video_path(self._recording_id, self._codec["ext"])
        bytes_written = video_path.stat().st_size if video_path.exists() else 0
        elapsed_s = (time.perf_counter_ns() - self._recording_start_monotonic_ns) / 1e9
        bps = self._disk_estimator.bytes_per_second(bytes_written, elapsed_s)

        with self._counters_lock:
            frames_written = max(1, self._counters.get("frames_write_attempted", 1))
        avg_bytes_per_frame = bytes_written / frames_written
        queued_unwritten_bytes = int(self._queue.qsize() * avg_bytes_per_frame)

        result = self._disk_estimator.check(recording_store.RECORDINGS_DIR, queued_unwritten_bytes, bps)
        result["bytes_written"] = bytes_written
        result["bytes_per_second"] = bps

        if not result["ok"]:
            self.stop(stop_reason="disk_space")

        return result

    # ---- stop/finalize ----

    def stop(self, stop_reason: str = "user_stop_button") -> dict:
        if not self._is_recording:
            raise RuntimeError("Recorder.stop() called while not recording.")

        # Flipped FIRST, before unsubscribe() -- see the flag's own
        # comment in __init__ for the exact race this closes.
        self._accepting_frames = False
        self._dispatcher.unsubscribe("recorder")
        self._stop_writer.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=30.0)
            self._writer_thread = None

        # Safety net on top of the _accepting_frames guard above: drain
        # anything that still ended up in the queue (e.g. a callback
        # already past the guard check when it flipped) and write it for
        # real, rather than silently orphaning a frame that was already
        # counted in frames_enqueue_attempted. In the normal case this
        # loop runs zero times -- the writer thread's own drain-to-empty
        # condition already emptied the queue before join() returned.
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            self._write_event(event)

        self._close_drop_range()

        self._writer.release()
        self._writer = None
        self._frame_index_writer.__exit__(None, None, None)
        self._frame_index_writer = None

        with self._counters_lock:
            counters = dict(self._counters)
            drop_ranges = list(self._drop_ranges)

        recording_id = self._recording_id
        codec = self._codec

        suspected_gaps = self._scan_for_gaps(recording_id)

        # Post-hoc verification: reopen and SEQUENTIALLY DECODE every
        # frame -- never trust CAP_PROP_FRAME_COUNT alone (an unreliable
        # container estimate for some codec/container combinations, per
        # the Stage A recording-format validation).
        video_path = recording_store.recording_video_path(recording_id, codec["ext"])
        output_frames_verified = recording_store.sequential_decode_count(video_path)

        recording_block = {
            "frames_acquired_during_recording": counters["frames_acquired_during_recording"],
            "frames_enqueue_attempted": counters["frames_enqueue_attempted"],
            "recorder_queue_drops": counters["recorder_queue_drops"],
            "frames_write_attempted": counters["frames_write_attempted"],
            "output_frames_verified": output_frames_verified,
            "drop_ranges": drop_ranges,
            "suspected_driver_gaps": suspected_gaps,
            "completed_normally": stop_reason not in ("error", "camera_disconnected"),
            "stop_reason": stop_reason,
        }
        recording_store.finalize_recording(recording_id, recording_block)

        self._is_recording = False
        self._recording_id = None
        self._session_id = None
        self._codec = None

        return {"recording_id": recording_id, **recording_block}

    @staticmethod
    def _scan_for_gaps(recording_id: str) -> list[dict]:
        """Heuristic scan of this recording's own frame_index.jsonl for
        timing anomalies (delta > 3x the median inter-frame interval).
        Never presented as a measured drop count, only a suspicion -- the
        driver can silently drop frames before stream.read() ever returns
        them, and no software layer in this stack can measure that with
        certainty."""
        rows = list(recording_store.read_frame_index(recording_id))
        if len(rows) < 3:
            return []

        deltas = []
        for i in range(1, len(rows)):
            prev_ts = rows[i - 1]["host_receive_monotonic_ns"]
            cur_ts = rows[i]["host_receive_monotonic_ns"]
            if prev_ts is not None and cur_ts is not None:
                deltas.append((rows[i - 1]["source_sequence_id"], cur_ts - prev_ts))

        if not deltas:
            return []

        median = sorted(d for _, d in deltas)[len(deltas) // 2]
        if median <= 0:
            return []

        return [
            {"after_sequence_id": after_sequence_id, "gap_ns": gap_ns}
            for after_sequence_id, gap_ns in deltas
            if gap_ns > GAP_HEURISTIC_MULTIPLIER * median
        ]
