"""
Background-thread wrapper around camera.CameraStream, so USB capture
finally runs at its true delivered rate instead of being ceilinged by
LiveCameraTab's 15ms Qt GUI timer (~67fps in theory, less under GUI
load -- see the plan this module implements). Same precedent as
vdo_ninja_source.VdoNinjaSource: duck-types read()/get_info()/release()
(plus .cap/.index) so LiveCameraTab can hold this behind main_window.stream
exactly like a plain CameraStream, with zero changes to _update_frame().

camera.py is not modified -- this wraps an already-open CameraStream,
it doesn't change how one is built or opened.

Two separate locks, deliberately not one:
  - `lock` (public): guards actual cv2.VideoCapture access -- the
    capture loop's real_stream.read() call, and camera_controls_panel.py's
    direct stream.cap.get()/set() calls (wrapped there to match). Held
    only briefly around each individual call, never for the loop body.
  - `_state_lock` (private): guards self._latest, the last frame this
    object hands back from read(). LiveCameraTab's own 15ms timer calls
    read() far more often than camera_controls_panel.py touches cap, so
    keeping this separate means a slider drag never makes the preview
    stall waiting on the same lock as the capture loop's hot path.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from camera import CameraStream
from frame_dispatcher import FrameDispatcher


class ThreadedCameraSource:
    def __init__(
        self,
        real_stream: CameraStream,
        dispatcher: FrameDispatcher,
        source_key: str,
        source_session_id: str,
    ) -> None:
        self._real_stream = real_stream
        self._dispatcher = dispatcher
        self._source_key = source_key
        self._session_id = source_session_id

        self.lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._latest: np.ndarray | None = None

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- duck-typed CameraStream/VdoNinjaSource shape ----

    @property
    def cap(self):
        return self._real_stream.cap

    @property
    def index(self):
        return self._real_stream.index

    @property
    def session_id(self) -> str:
        """The source_session_id this instance publishes under -- read by
        recorder.py so a recording only ever accepts frames tagged with
        whichever session was active when it started (see
        frame_dispatcher.py's session-safety notes)."""
        return self._session_id

    def read(self) -> np.ndarray | None:
        with self._state_lock:
            return None if self._latest is None else self._latest.copy()

    def get_info(self) -> dict:
        return self._real_stream.get_info()

    def release(self) -> None:
        self.stop()
        self._real_stream.release()

    # ---- lifecycle ----

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("ThreadedCameraSource.start() called while already running -- call stop() first.")

        self._stop_event.clear()
        # Activated before the thread starts, so no frame this source
        # captures can ever be published before its session is the
        # active one.
        self._dispatcher.activate_session(self._session_id)

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signals the capture thread to stop and BLOCKS until it has
        actually exited (not merely signaled) -- callers (LiveCameraTab's
        source-switch handlers) rely on this to guarantee no in-flight
        read()/publish() from this source can race with a new source
        starting immediately after this returns."""
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

        self._dispatcher.invalidate_session(self._session_id)

    # ---- capture loop (background thread only) ----

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            with self.lock:
                frame = self._real_stream.read()

            if frame is None:
                # A real, connected camera's read() blocks until the next
                # frame arrives -- reaching here in steady state means a
                # transient failure (or the device just vanished). A tiny
                # sleep keeps that case from pegging a CPU core with a
                # busy-spin; it does not affect real throughput, since a
                # healthy read() call itself already blocks for pacing.
                time.sleep(0.001)
                continue

            with self._state_lock:
                self._latest = frame

            self._dispatcher.publish(frame, self._source_key, self._session_id)
