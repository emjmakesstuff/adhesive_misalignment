"""
VDO.Ninja viewer capture: adapted from ../test_code.py's Playwright-based
direct-video-element capture technique (draw the live <video> frame onto
a reusable <canvas>, read it back as PNG -- never a full-page
screenshot, since that was measurably slower and capped at the browser
viewport size rather than the stream's true native resolution;
confirmed in that file's own docstring, along with CDP
Page.startScreencast never delivering frames in this environment
either). Restructured here as a start/stop object (open/read/get_info/
release) instead of a blocking CLI loop, so LiveCameraTab can poll it
exactly like camera.CameraStream.

All Playwright/browser code lives here and only here -- camera.py never
imports this module and this module never imports camera.py, per the
project's module-separation rule.

Threading: Playwright's sync API is not safe to use from more than one
thread. The entire lifecycle -- launch, navigate, wait for video,
capture loop, close -- runs on one dedicated background thread created
in open() and torn down in release(); the GUI thread only ever touches
this object's read()/get_info()/release(), all of which are just
lock-protected attribute access or a threading.Event set, never
Playwright objects directly.
"""

from __future__ import annotations

import base64
import threading
import time

import cv2
import numpy as np
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright

# Same named hue presets as ../test_code.py, kept in sync manually --
# that file is a separate standalone CLI tool this module deliberately
# never imports from (this app stays "completely separate from the
# VDO.Ninja program in the parent folder", per README.md's existing
# framing; this module only reuses its *technique*, not its code).
HUE_PRESETS = {
    "red": [(0, 10), (170, 179)],
    "orange": [(10, 20)],
    "yellow": [(20, 35)],
    "green": [(35, 85)],
    "cyan": [(85, 100)],
    "blue": [(100, 130)],
    "purple": [(130, 160)],
    "pink": [(160, 170)],
}

_JS_FIND_LIVE_VIDEO = """
() => {
    const videos = Array.from(document.querySelectorAll("video"));
    return videos.some(v => v.readyState >= 2 && v.videoWidth > 0 && v.videoHeight > 0);
}
"""

_JS_UNMUTE_AND_PLAY = """
() => {
    const videos = Array.from(document.querySelectorAll("video"));
    for (const video of videos) {
        video.muted = true;
        video.play().catch(() => {});
    }
}
"""

# VDO.Ninja (like many WebRTC/simulcast platforms) selects which quality
# layer to SEND a given viewer based on how large that viewer's <video>
# element is actually displayed -- confirmed directly against a real
# live stream: with the default small viewport this plateaued around
# 450x800, and with the video element forced to a large box it ramped up
# to the stream's true 1080x1920. Applied as early as possible (right
# after the <video> element is found, before even waiting for it to
# start playing) so the size hint is in place before/during the
# negotiation ramp-up, not applied only after the fact.
_JS_MAXIMIZE_VIDEO_ELEMENTS = """
() => {
    const videos = Array.from(document.querySelectorAll("video"));
    for (const video of videos) {
        video.style.width = "1920px";
        video.style.height = "1920px";
        video.style.objectFit = "contain";
    }
}
"""

_JS_BIND_CANVAS = """
() => {
    const video = Array.from(document.querySelectorAll("video"))
        .find(v => v.readyState >= 2 && v.videoWidth > 0 && v.videoHeight > 0);
    if (!video) return null;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    window.__ftirCaptureVideo = video;
    window.__ftirCaptureCanvas = canvas;
    window.__ftirCaptureCtx = canvas.getContext("2d", { willReadFrequently: true });
    return { width: canvas.width, height: canvas.height };
}
"""

_JS_CAPTURE_FRAME = """
() => {
    const ctx = window.__ftirCaptureCtx;
    const video = window.__ftirCaptureVideo;
    const canvas = window.__ftirCaptureCanvas;
    if (!ctx || !video || !canvas) return null;

    // WebRTC senders commonly ramp up from a small placeholder frame to
    // the real negotiated resolution over the first few seconds -- the
    // canvas was originally sized once, right when the <video> first had
    // ANY nonzero frame (see _JS_BIND_CANVAS), which could still be that
    // low-res placeholder. Re-checking/resizing here on every capture
    // means a later resolution increase (e.g. up to the real 1080p) gets
    // picked up on the very next frame instead of staying locked to
    // whatever size happened to exist at bind time.
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    // JPEG, not PNG -- confirmed directly against a real 1080x1920
    // stream that PNG-encoding full-resolution frames capped achieved
    // fps around 12-13 even with requested_fps=30 (PNG encode time was
    // the bottleneck, not the poll rate). The source is WebRTC video --
    // already lossy-compressed before it ever reaches this canvas -- so
    // JPEG here doesn't add a first generation of loss, only a second
    // one on top of a signal that was never lossless to begin with.
    // Quality 0.85, not lower: tested 0.6 vs 0.9 directly against the
    // real stream and got the same ~19-20fps either way -- the
    // per-frame page.evaluate()/CDP round-trip is the real ceiling here,
    // not JPEG encode time or payload size, so there's no fps benefit to
    // trading away more fidelity than this.
    return { width: canvas.width, height: canvas.height, dataUrl: canvas.toDataURL("image/jpeg", 0.85) };
}
"""


class VdoNinjaSource:
    """
    Duck-types camera.CameraStream's read()/get_info()/release() so
    LiveCameraTab can hold either object behind main_window.stream
    without a formal shared base class -- nothing else in this project
    uses one, so this stays consistent with the existing style rather
    than introducing a new abstraction just for two implementations.
    """

    def __init__(self) -> None:
        # camera_controls_panel.py's every method checks `stream.cap is
        # None` before touching exposure/gain/video-proc-amp hardware
        # controls, since those are meaningless for a received video
        # stream (requirement: hide/disable USB-only controls while
        # VDO.Ninja is active). Rather than editing that already-working
        # file to add a source-type branch, this duck-types the one
        # attribute it checks -- every one of its existing guards then
        # already does the right thing (reports "not connected", no-ops)
        # against a VdoNinjaSource with zero changes there.
        self.cap = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._status = "idle"
        self._native_width = 0
        self._native_height = 0
        self._view_url = ""
        self._requested_fps = 15.0

        # Dispatcher wiring -- all optional (default None) so this class
        # stays usable standalone (e.g. ../test_code.py-style direct use)
        # without a main_window/FrameDispatcher in the picture. When
        # provided, _capture_loop publishes every captured frame through
        # the same dispatcher USB's ThreadedCameraSource uses -- see
        # frame_dispatcher.py and its "Camera-switch safety" notes.
        self._dispatcher = None
        self._source_key = None
        self._session_id = None

    @property
    def session_id(self) -> str | None:
        """The source_session_id this instance publishes under, or None
        if opened without dispatcher wiring -- same purpose as
        ThreadedCameraSource.session_id (see frame_dispatcher.py)."""
        return self._session_id

    def open(
        self,
        view_url: str,
        requested_fps: float = 15.0,
        dispatcher=None,
        source_key: str | None = None,
        source_session_id: str | None = None,
    ) -> None:
        if self._thread is not None:
            raise RuntimeError("VdoNinjaSource.open() called while already open -- call release() first.")

        self._view_url = view_url
        self._requested_fps = requested_fps
        self._dispatcher = dispatcher
        self._source_key = source_key
        self._session_id = source_session_id
        self._stop_event.clear()

        with self._lock:
            self._latest_frame = None
            self._status = "connecting"
            self._native_width = 0
            self._native_height = 0

        if self._dispatcher is not None and self._session_id is not None:
            # Activated before the thread starts, same ordering
            # ThreadedCameraSource.start() uses -- no frame this source
            # captures can be published before its session is the active
            # one.
            self._dispatcher.activate_session(self._session_id)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def get_info(self) -> dict:
        # Shaped like camera.CameraStream.get_info() where the concepts
        # overlap (backend_name/actual_width/actual_height/fourcc), plus
        # a VDO.Ninja-specific "status" the UI's connection-status label
        # reads directly -- a real cv2.VideoCapture has no equivalent
        # notion of "still connecting" vs "live", so this key simply
        # doesn't exist on CameraStream's dict; callers must not assume
        # every source provides it.
        with self._lock:
            return {
                "backend_name": "VDO.Ninja",
                "requested_fourcc": None,
                "fourcc": "N/A",
                "requested_width": None,
                "requested_height": None,
                "actual_width": self._native_width,
                "actual_height": self._native_height,
                "requested_fps": self._requested_fps,
                "actual_fps_reported": 0.0,
                "status": self._status,
            }

    def release(self) -> None:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

        if self._dispatcher is not None and self._session_id is not None:
            self._dispatcher.invalidate_session(self._session_id)

    # ---- worker thread body ----

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def _run(self) -> None:
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    channel="chrome",
                    headless=True,  # always -- no visible-browser option in this GUI path, confirmed with the user
                    args=["--autoplay-policy=no-user-gesture-required", "--disable-infobars"],
                )
                # Large and roughly square: big enough that neither
                # dimension becomes the constraining factor regardless of
                # whether the stream is landscape or portrait (confirmed
                # directly -- a 1280x800 landscape-shaped viewport capped
                # a real portrait 1080x1920 stream at 450x800, since the
                # video was letterboxed to fit the *shorter* dimension of
                # a differently-shaped box).
                context = browser.new_context(viewport={"width": 1920, "height": 1920})
                page = context.new_page()

                # Same diagnostic hooks ../test_code.py already has --
                # omitted when this was first adapted since this runs
                # headless/silent in a GUI, not a terminal CLI loop, but
                # that left no way to see *why* a connection failed
                # beyond a generic status string. Printed (visible if the
                # app was launched from a console) rather than routed
                # into the status label, since these can fire repeatedly/
                # verbosely -- unlike the status string, which is meant
                # to hold one current summary, not a scrolling log.
                page.on("console", lambda msg: print(f"[VDO.Ninja page] {msg.type.upper()}: {msg.text}"))
                page.on("pageerror", lambda error: print(f"[VDO.Ninja page] PAGE ERROR: {error}"))
                page.on(
                    "requestfailed",
                    lambda request: print(f"[VDO.Ninja page] FAILED REQUEST: {request.url} | {request.failure}"),
                )

                try:
                    self._capture_loop(page)
                finally:
                    try:
                        context.close()
                    except Exception:
                        pass
                    try:
                        browser.close()
                    except Exception:
                        pass
        except Exception as error:
            self._set_status(f"error: {error}")

    def _capture_loop(self, page) -> None:
        self._set_status("connecting")

        try:
            page.goto(self._view_url, wait_until="domcontentloaded", timeout=30_000)
        except PlaywrightTimeoutError:
            self._set_status("error: VDO.Ninja did not load within 30 seconds")
            return

        if self._stop_event.is_set():
            return

        self._set_status("waiting for video element")

        try:
            page.wait_for_selector("video", state="attached", timeout=30_000)
        except PlaywrightTimeoutError:
            self._set_status("error: no video element found -- check the viewer URL")
            return

        if self._stop_event.is_set():
            return

        # Applied before waiting for live video, not after -- the size
        # hint needs to be in place while VDO.Ninja is still deciding/
        # ramping up which quality layer to send, not applied only once
        # it's already settled on a low one.
        page.evaluate(_JS_MAXIMIZE_VIDEO_ELEMENTS)

        self._set_status("waiting for live video")

        try:
            page.wait_for_function(_JS_FIND_LIVE_VIDEO, timeout=60_000)
        except PlaywrightTimeoutError:
            self._set_status("error: page loaded but no live video arrived")
            return

        if self._stop_event.is_set():
            return

        page.evaluate(_JS_UNMUTE_AND_PLAY)
        capture_dims = page.evaluate(_JS_BIND_CANVAS)

        if capture_dims is None:
            self._set_status("error: could not bind to a live video element")
            return

        with self._lock:
            self._native_width = capture_dims["width"]
            self._native_height = capture_dims["height"]
            self._status = "live"

        frame_interval = 1.0 / self._requested_fps if self._requested_fps > 0 else 0.0
        next_frame_time = time.perf_counter()

        while not self._stop_event.is_set():
            if page.is_closed():
                self._set_status("error: browser tab closed unexpectedly")
                return

            now = time.perf_counter()

            if now < next_frame_time:
                # Waits on the stop event itself (not a plain sleep) so
                # release() is responsive even mid-interval, rather than
                # blocking up to a full frame_interval before noticing
                # the stop request.
                self._stop_event.wait(timeout=next_frame_time - now)
                continue

            next_frame_time = time.perf_counter() + frame_interval if frame_interval > 0 else time.perf_counter()

            try:
                captured = page.evaluate(_JS_CAPTURE_FRAME)
            except Exception as error:
                self._set_status(f"error: frame capture failed: {error}")
                return

            if captured is None:
                continue

            png_bytes = base64.b64decode(captured["dataUrl"].split(",", 1)[1])
            array = np.frombuffer(png_bytes, dtype=np.uint8)
            frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

            if frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    # Kept current every frame, not just once at bind time
                    # -- see _JS_CAPTURE_FRAME's own comment for why the
                    # resolution can legitimately change mid-stream.
                    self._native_width = captured["width"]
                    self._native_height = captured["height"]
                    self._status = "live"

                if self._dispatcher is not None and self._session_id is not None:
                    self._dispatcher.publish(frame, self._source_key, self._session_id)
