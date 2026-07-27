"""
Device I/O for the USB camera: index probing and a thin wrapper around
cv2.VideoCapture. No display/GUI code lives here -- that's preview.py.
Keeping this separate means the capture path can be reused as-is once
image-processing logic (processing.py) grows into a full detection stage.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

BACKENDS = {
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}


def list_device_names(max_index: int = 8) -> list[str] | None:
    """
    Real DirectShow device friendly names (e.g. "USB Camera"), via
    pygrabber/comtypes -- cv2.VideoCapture has no equivalent API. Returns
    None if pygrabber isn't available or enumeration fails for any reason,
    so callers can fall back to index-only reporting rather than crashing;
    this is a diagnostic nicety, not something capture should depend on.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph

        return FilterGraph().get_input_devices()[:max_index]
    except Exception:
        return None


def list_device_formats(index: int) -> list[dict] | None:
    """
    The real (resolution, pixel format, fps range) combinations this
    camera's driver actually advertises, via pygrabber's IAMStreamConfig
    enumeration -- the same DirectShow API AMCap's format dialog uses.
    This is how we know MJPG vs YUY2 support wildly different fps ranges
    at the same resolution, instead of guessing. Returns None on failure.

    Note: pygrabber names these fields after the underlying DirectShow
    struct fields (MinFrameInterval/MaxFrameInterval), which are inverted
    from what you'd expect -- 'min_framerate' is derived from the
    *shortest* interval, i.e. it's the fastest achievable fps, and
    'max_framerate' is the slowest. Kept as-is here to match the library;
    preview.py's printing accounts for this.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph

        graph = FilterGraph()
        graph.add_video_input_device(index)
        return graph.get_input_device().get_formats()
    except Exception:
        return None


@dataclass
class ProbeResult:
    index: int
    opened: bool
    can_read: bool
    width: int
    height: int
    backend_name: str


def _probe_one(index: int, backend: int, result_queue: "queue.Queue[ProbeResult]") -> None:
    cap = cv2.VideoCapture(index, backend)
    opened = cap.isOpened()
    can_read = False
    width = 0
    height = 0
    backend_name = ""

    if opened:
        backend_name = cap.getBackendName()
        ok, frame = cap.read()

        if ok and frame is not None:
            can_read = True
            height, width = frame.shape[:2]

    cap.release()
    result_queue.put(
        ProbeResult(
            index=index,
            opened=opened,
            can_read=can_read,
            width=width,
            height=height,
            backend_name=backend_name,
        )
    )


def probe_cameras(
    max_index: int = 5,
    backend: int = cv2.CAP_ANY,
    timeout_seconds: float = 3.0,
) -> list[ProbeResult]:
    """
    Windows/OpenCV has no camera-name enumeration API, so "detecting"
    cameras means trying sequential indices and seeing what responds.

    cv2.VideoCapture has no built-in open timeout and can block
    indefinitely on some index/backend combinations (observed directly
    while developing this: a plain call hung forever with zero feedback
    on a machine with no camera driver at all). Each index is probed in
    a daemon thread with a join timeout so a single stuck index is
    reported and skipped rather than freezing the whole probe -- and
    because the thread is a daemon, a permanently stuck native call
    can't hang process exit either.
    """
    results = []

    for index in range(max_index):
        print(f"  probing index {index} (timeout {timeout_seconds:.0f}s)...")
        result_queue: "queue.Queue[ProbeResult]" = queue.Queue()
        thread = threading.Thread(
            target=_probe_one,
            args=(index, backend, result_queue),
            daemon=True,
        )
        thread.start()
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            print(
                f"    index {index}: no response within {timeout_seconds:.0f}s, "
                f"skipping (likely no device at this index; if every index "
                f"times out, try a different --backend)"
            )
            results.append(
                ProbeResult(
                    index=index,
                    opened=False,
                    can_read=False,
                    width=0,
                    height=0,
                    backend_name="(timed out)",
                )
            )
            continue

        results.append(result_queue.get())

    return results


class CameraStream:
    """
    Thin wrapper around cv2.VideoCapture for one USB camera. Tracks what
    was requested vs what the driver actually granted, and detects when
    the camera stops delivering frames instead of failing silently.
    """

    def __init__(
        self,
        index: int,
        backend: int = cv2.CAP_ANY,
        requested_width: int | None = None,
        requested_height: int | None = None,
        requested_fps: float | None = None,
        requested_fourcc: str | None = None,
    ) -> None:
        self.index = index
        self.backend = backend
        self.requested_width = requested_width
        self.requested_height = requested_height
        self.requested_fps = requested_fps
        self.requested_fourcc = requested_fourcc
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.index, self.backend)

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            raise RuntimeError(
                f"Could not open camera index {self.index} "
                f"(backend={self._backend_name()}). "
                f"Things to check: is the camera plugged in, is another "
                f"program (Zoom, Teams, another OpenCV script) already "
                f"using it, is this the right index (run with --list), "
                f"or try a different --backend (any/dshow/msmf)."
            )

        if self.requested_width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)

        if self.requested_height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)

        if self.requested_fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, self.requested_fps)

        # FourCC MUST be set last. Empirically (tested against real
        # hardware): setting it before width/height/fps gets silently
        # reverted to the driver's default pixel format by those later
        # calls -- each cv2 DSHOW .set() appears to renegotiate the full
        # format, not just the one field. Confirmed the effect is large:
        # at 1280x800 on the reference camera, YUY2 (the default when
        # FourCC is never requested) is hard-capped at 10fps by the
        # driver itself, while MJPG at the same resolution supports the
        # full 10-120fps range -- so getting this order right is the
        # difference between ~9fps and 70+fps, not a minor tweak.
        if self.requested_fourcc is not None:
            self.cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*self.requested_fourcc),
            )

    def _backend_name(self) -> str:
        if self.cap is not None:
            return self.cap.getBackendName()

        return next(
            (name for name, value in BACKENDS.items() if value == self.backend),
            str(self.backend),
        )

    def read(self) -> np.ndarray | None:
        if self.cap is None:
            raise RuntimeError("CameraStream.open() must be called before read().")

        ok, frame = self.cap.read()

        if not ok or frame is None:
            return None

        return frame

    def get_info(self) -> dict:
        if self.cap is None:
            raise RuntimeError("CameraStream.open() must be called before get_info().")

        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_chars = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
        fourcc_str = (
            fourcc_chars
            if fourcc_int and fourcc_chars.isascii() and fourcc_chars.isprintable()
            else f"unknown (raw value {fourcc_int})"
        )

        return {
            "backend_name": self.cap.getBackendName(),
            "requested_width": self.requested_width,
            "requested_height": self.requested_height,
            "requested_fps": self.requested_fps,
            "requested_fourcc": self.requested_fourcc,
            "actual_width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "actual_height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "actual_fps_reported": self.cap.get(cv2.CAP_PROP_FPS),
            "fourcc": fourcc_str,
        }

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class FpsMeter:
    """Measures real frame-arrival rate; cap.get(CAP_PROP_FPS) is often
    unreliable or zero for USB cameras, so this is driven by wall-clock
    timestamps of frames actually received."""

    def __init__(self, window_seconds: float = 2.0) -> None:
        self.window_seconds = window_seconds
        self._timestamps: list[float] = []

    def tick(self) -> None:
        now = time.perf_counter()
        self._timestamps.append(now)
        cutoff = now - self.window_seconds

        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.pop(0)

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0

        span = self._timestamps[-1] - self._timestamps[0]

        if span <= 0:
            return 0.0

        return (len(self._timestamps) - 1) / span
