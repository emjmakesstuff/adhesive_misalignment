"""
Frame dispatcher: the single point every captured frame -- USB or
VDO.Ninja -- passes through after leaving its background capture thread,
before any GUI polling loop or recorder ever sees it. One instance,
owned by main_window (main_window.frame_dispatcher), created once at
startup and reused across every connect/disconnect.

publish() is called from whichever background thread just captured a
frame (threaded_camera_source.ThreadedCameraSource's capture loop, or
vdo_ninja_source.VdoNinjaSource's existing _capture_loop) -- never from
the GUI thread. It does three things, in order: (1) rejects the frame
outright if source_session_id doesn't match the currently active session
(see activate_session/invalidate_session below -- this is defense in
depth on top of the caller already having joined the old capture thread
before starting a new one, not the primary safety mechanism), (2)
updates main_window.latest_frame as a side effect for the existing GUI
polling consumers (CalibrationTab, DetectionTab, LiveCameraTab's own
preview -- none of these need a gapless sequence, just "something
recent"), and (3) invokes every subscribed callback synchronously, on
the calling (capture) thread, with the same FrameEvent.

Subscriber callbacks (the Stage B recorder will be the first) must
therefore be fast and non-blocking -- typically just a queue.put_nowait
into the subscriber's own bounded queue, with the subscriber counting
its own enqueue-attempts/drops (frames_enqueue_attempted/
recorder_queue_drops from the recording-metadata schema), not this
module. This dispatcher intentionally owns no per-subscriber queue
itself: what a subscriber does with an event, and how it accounts for
it, is entirely that subscriber's business. A callback that raises is
caught and printed, never allowed to take down the capture thread that's
calling publish().
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class FrameEvent:
    sequence_id: int  # monotonically increasing, one dispatcher instance per session
    source_session_id: str  # identifies WHICH connect this came from -- see activate_session/invalidate_session
    # time.perf_counter_ns() as close as possible to when stream.read() returned -- NOT the camera
    # sensor's true exposure instant (USB/DirectShow/driver buffering means these can differ). Named
    # deliberately honest: this codebase must never imply a precision it doesn't have.
    host_receive_monotonic_ns: int
    frame: np.ndarray  # an owned copy, read-only (.setflags(write=False)) -- see module docstring
    source_key: str  # active_profile_key at the moment of acquisition


class FrameDispatcher:
    def __init__(self, main_window) -> None:
        self._main_window = main_window
        self._lock = threading.Lock()
        self._sequence_id = 0
        self._active_session_id: str | None = None
        self._subscribers: dict[str, Callable[[FrameEvent], None]] = {}

    # ---- session safety (see the plan's "Camera-switch safety" section) ----

    def activate_session(self, source_session_id: str) -> None:
        """Called by a capture source right before it starts publishing
        (ThreadedCameraSource.start(), VdoNinjaSource.open()) -- makes
        this the only session publish() will accept frames from."""
        with self._lock:
            self._active_session_id = source_session_id

    def invalidate_session(self, source_session_id: str) -> None:
        """Called by a capture source once its background thread is
        confirmed joined (not merely signaled to stop). No-ops if a
        newer session has already taken over -- a stale/delayed
        invalidate call from an old capture object must never clobber a
        session that's already active."""
        with self._lock:
            if self._active_session_id == source_session_id:
                self._active_session_id = None

    def is_session_active(self, source_session_id: str) -> bool:
        with self._lock:
            return self._active_session_id == source_session_id

    # ---- subscribers ----

    def subscribe(self, subscriber_id: str, callback: Callable[[FrameEvent], None]) -> None:
        with self._lock:
            self._subscribers[subscriber_id] = callback

    def unsubscribe(self, subscriber_id: str) -> None:
        with self._lock:
            self._subscribers.pop(subscriber_id, None)

    # ---- publish ----

    def publish(self, raw_frame: np.ndarray, source_key: str, source_session_id: str) -> FrameEvent | None:
        """
        Called from a background capture thread with a freshly-read
        frame, immediately after read() returns. Returns the FrameEvent
        that was actually dispatched, or None if it was rejected (stale/
        inactive session -- the caller should simply drop the frame in
        that case, not treat it as an error).
        """
        host_receive_monotonic_ns = time.perf_counter_ns()

        with self._lock:
            if source_session_id != self._active_session_id:
                return None

            self._sequence_id += 1
            sequence_id = self._sequence_id
            callbacks = list(self._subscribers.values())

        frame = raw_frame.copy()
        frame.setflags(write=False)

        event = FrameEvent(
            sequence_id=sequence_id,
            source_session_id=source_session_id,
            host_receive_monotonic_ns=host_receive_monotonic_ns,
            frame=frame,
            source_key=source_key,
        )

        self._main_window.latest_frame = frame

        for callback in callbacks:
            try:
                callback(event)
            except Exception as error:
                print(f"[frame_dispatcher] subscriber callback raised: {error}")

        return event
