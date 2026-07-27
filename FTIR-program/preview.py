"""
Stage 1: reliable USB camera connection and live preview.

Examples:
    py preview.py --list
    py preview.py --index 1                          # defaults to 1280x800 @ 30fps
    py preview.py --index 1 --fps 120                 # camera's spec'd max; not yet reliably achieved, see README

Controls:
    q       quit
    space   pause/resume
    s       save the current raw (unprocessed) frame
    c       open the camera's native settings dialog (exposure, gain,
            brightness, etc.) in the background -- the live preview
            keeps running so you can see slider changes take effect
            immediately

See README.md for install/run/troubleshooting details.
"""

from __future__ import annotations

import argparse
import datetime
import threading
from pathlib import Path

import cv2
import numpy as np

from camera import BACKENDS, CameraStream, FpsMeter, list_device_formats, list_device_names, probe_cameras
from processing import process_frame
from settings import get_exposure_gain_info, open_properties_dialog, print_settings_report


def print_probe_table(results) -> None:
    names = list_device_names(max_index=len(results)) or []

    print(f"{'index':>5}  {'opened':>6}  {'got frame':>9}  {'resolution':>11}  {'backend':>8}  name")

    for r in results:
        resolution = f"{r.width}x{r.height}" if r.can_read else "-"
        name = names[r.index] if r.index < len(names) else "-"
        print(
            f"{r.index:>5}  {str(r.opened):>6}  {str(r.can_read):>9}  "
            f"{resolution:>11}  {r.backend_name or '-':>8}  {name}"
        )


def print_format_table(index: int) -> None:
    formats = list_device_formats(index)

    if formats is None:
        print(
            "Could not enumerate real driver capabilities for this index "
            "(pygrabber/comtypes issue, or an unsupported device). Falling "
            "back to trial-and-error with --width/--height/--fps/--fourcc "
            "is still possible, just without this upfront guidance."
        )
        return

    if not formats:
        print("No formats reported for this index.")
        return

    print(f"{'resolution':>11}  {'format':>6}  fastest fps  slowest fps")

    for f in formats:
        # pygrabber names these after the DirectShow struct fields directly:
        # 'min_framerate' comes from MinFrameInterval (shortest interval =
        # fastest fps), 'max_framerate' from MaxFrameInterval (slowest fps).
        # Printed here as fastest/slowest to avoid that confusion.
        resolution = f"{f['width']}x{f['height']}"
        print(
            f"{resolution:>11}  {f['media_type_str']:>6}  "
            f"{f['min_framerate']:>11.1f}  {f['max_framerate']:>11.1f}"
        )


def choose_index_interactively(max_probe_index: int, backend: int, probe_timeout: float) -> int:
    print(f"No --index given. Probing camera indices 0..{max_probe_index - 1}...")
    results = probe_cameras(max_probe_index, backend, probe_timeout)
    print_probe_table(results)

    workable = [r.index for r in results if r.can_read]

    if not workable:
        raise RuntimeError(
            "No camera index in the probed range delivered a frame. "
            "Try a larger --max-probe-index, a different --backend, "
            "or confirm the camera is plugged in and not in use by "
            "another program."
        )

    while True:
        choice = input(f"Enter camera index to use {workable}: ").strip()

        try:
            index = int(choice)
        except ValueError:
            print("Please enter a number.")
            continue

        if index not in workable:
            print(f"{index} did not deliver a frame during probing. Pick from {workable}.")
            continue

        return index


def describe_frame_format(frame: np.ndarray) -> str:
    if frame.ndim == 2:
        return f"true single-channel grayscale, shape {frame.shape}"

    if frame.ndim == 3:
        channels = frame.shape[2]

        if channels == 3:
            sample_equal = np.array_equal(frame[:, :, 0], frame[:, :, 1]) and np.array_equal(
                frame[:, :, 1], frame[:, :, 2]
            )
            note = (
                "R=G=B at every pixel -- grayscale data delivered inside a 3-channel frame"
                if sample_equal
                else "channels differ -- this looks like real color, not grayscale"
            )
            return f"3-channel, shape {frame.shape} ({note})"

        return f"{channels}-channel, shape {frame.shape}"

    return f"unexpected shape {frame.shape}"


def _open_settings_dialog_background(index: int) -> None:
    """
    Runs on a background thread so the main thread's preview loop is never
    blocked by the modal settings dialog. Uses its own separate pygrabber
    binding to the camera (confirmed to coexist fine with an already-open
    cv2.VideoCapture on the same device) and never touches stream.cap --
    that stays exclusively owned by the main thread.
    """
    try:
        open_properties_dialog(index)
    except RuntimeError as error:
        print(f"ERROR opening settings dialog: {error}")


def to_display_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert whatever shape the camera delivers into a BGR frame we can
    draw colored overlays on, without touching the original raw frame."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    return frame.copy()


def run(args: argparse.Namespace) -> None:
    backend = BACKENDS[args.backend]

    if args.list:
        results = probe_cameras(args.max_probe_index, backend, args.probe_timeout)
        print_probe_table(results)
        return

    index = args.index

    if index is None:
        index = choose_index_interactively(args.max_probe_index, backend, args.probe_timeout)

    if args.list_formats:
        print_format_table(index)
        return

    stream = CameraStream(
        index=index,
        backend=backend,
        requested_width=args.width,
        requested_height=args.height,
        requested_fps=args.fps,
        requested_fourcc=args.fourcc or None,  # "" means "don't request a format"
    )

    try:
        stream.open()
    except RuntimeError as error:
        print(f"ERROR: {error}")
        return

    info = stream.get_info()
    print("Camera opened.")
    print(f"  Backend:              {info['backend_name']}")
    print(f"  Requested FourCC:     {info['requested_fourcc'] or 'not set'}")
    print(f"  Actual FourCC:        {info['fourcc']}")
    print(
        f"  Requested resolution: "
        f"{info['requested_width'] or 'not set'} x {info['requested_height'] or 'not set'}"
    )
    print(f"  Actual resolution:    {info['actual_width']} x {info['actual_height']}")
    print(f"  Requested FPS:        {info['requested_fps'] or 'not set'}")
    print(f"  Driver-reported FPS:  {info['actual_fps_reported']:.2f} (often unreliable for USB cameras)")

    if args.fourcc and info["fourcc"] != args.fourcc:
        print(
            f"  NOTE: requested FourCC '{args.fourcc}' was not honored "
            f"(driver is using '{info['fourcc']}' instead) -- this format "
            f"may not be supported at this resolution. Run --list-formats "
            f"to see what this camera actually supports."
        )

    if args.report_settings:
        print("Exposure/gain settings (see settings.py to change these):")
        print_settings_report(get_exposure_gain_info(stream.cap))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    window_name = "USB Camera Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    max_preview_dim = 900
    preview_w, preview_h = info["actual_width"], info["actual_height"]

    if preview_w > 0 and preview_h > 0:
        scale = min(1.0, max_preview_dim / preview_w, max_preview_dim / preview_h)
        cv2.resizeWindow(window_name, max(1, int(preview_w * scale)), max(1, int(preview_h * scale)))

    fps_meter = FpsMeter()
    paused = False
    last_frame = None
    consecutive_failures = 0
    max_consecutive_failures = 60  # roughly 2-4s of no frames depending on true camera fps
    format_described = False
    save_count = 0
    settings_dialog_thread: threading.Thread | None = None

    print(
        "Preview started. Select the window and press q to quit, space to "
        "pause, s to save a frame, c to open the camera's settings dialog."
    )

    try:
        while True:
            if not paused:
                frame = stream.read()

                if frame is None:
                    consecutive_failures += 1

                    if consecutive_failures == 1:
                        print("Warning: camera did not return a frame this cycle.")

                    if consecutive_failures >= max_consecutive_failures:
                        print(
                            f"ERROR: no frame received for {consecutive_failures} consecutive "
                            f"reads. The camera may have been unplugged, gone to sleep, or been "
                            f"claimed by another program. Stopping."
                        )
                        break
                else:
                    if consecutive_failures > 0:
                        print(f"Recovered after {consecutive_failures} failed reads.")

                    consecutive_failures = 0
                    last_frame = frame
                    fps_meter.tick()

                    if not format_described:
                        print(f"Frame format: {describe_frame_format(frame)}")
                        format_described = True

            if last_frame is not None:
                processed = process_frame(last_frame)
                display = to_display_bgr(processed)

                h, w = display.shape[:2]
                cv2.putText(
                    display,
                    f"{w}x{h}  {fps_meter.fps:.1f} fps"
                    f"{'  [PAUSED]' if paused else ''}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                )

                cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                paused = not paused
                print("Paused" if paused else "Resumed")

            if key == ord("s"):
                if last_frame is None:
                    print("Nothing to save yet -- no frame received.")
                else:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    save_path = save_dir / f"frame_{timestamp}.png"
                    cv2.imwrite(str(save_path), last_frame)
                    save_count += 1
                    print(f"Saved raw frame to {save_path}")

            if key == ord("c"):
                if settings_dialog_thread is not None and settings_dialog_thread.is_alive():
                    print("Settings dialog is already open.")
                else:
                    print("Opening camera settings dialog (preview keeps running)...")
                    settings_dialog_thread = threading.Thread(
                        target=_open_settings_dialog_background,
                        args=(index,),
                        daemon=True,
                    )
                    settings_dialog_thread.start()

            # Runs on the main thread only -- stream.cap is read exclusively
            # here and in stream.read() above, never from the dialog thread,
            # to avoid concurrent access to the same cv2.VideoCapture object.
            if settings_dialog_thread is not None and not settings_dialog_thread.is_alive():
                settings_dialog_thread = None
                print("Settings dialog closed. Exposure/gain now:")
                print_settings_report(get_exposure_gain_info(stream.cap))

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        stream.release()
        cv2.destroyAllWindows()
        print(f"Done. {save_count} frame(s) saved to {save_dir}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: USB camera connection and live preview.")

    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Camera index to open. If omitted, probes and prompts interactively.",
    )

    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS.keys()),
        default="dshow",
        help=(
            "OpenCV capture backend. Defaults to dshow: for UVC industrial "
            "cameras like the U20CAM-9281M, the default 'any' backend often "
            "resolves to MSMF, which has poor UVC control support on "
            "Windows. Try 'any' or 'msmf' if dshow doesn't open the camera."
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help=(
            "Requested frame width. Defaults to the U20CAM-9281M's native "
            "1280x800 max resolution. Not guaranteed to be honored by the "
            "driver -- the program always prints the actual negotiated size."
        ),
    )

    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Requested frame height. Defaults to the U20CAM-9281M's native 800px.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=120,
        help=(
            "Requested capture FPS. Defaults to 120, this camera's spec'd "
            "max. Reaching it depends on --fourcc being set to a format "
            "that actually supports that range at this resolution (MJPG "
            "does; YUY2, the driver's default if no FourCC is requested, "
            "is hard-capped at 10fps at 1280x800 on this camera -- not a "
            "bandwidth estimate, the driver's own advertised capability). "
            "Check the measured fps shown in the preview window against "
            "this request; run --list-formats to see the real options."
        ),
    )

    parser.add_argument(
        "--fourcc",
        default="MJPG",
        help=(
            "Requested pixel format (4-char code, e.g. MJPG, YUY2). "
            "Defaults to MJPG -- empirically the format that unlocks high "
            "fps on this camera; YUY2 at 1280x800 is hard-capped at 10fps "
            "regardless of --fps. Must be set to something (not blank) to "
            "take effect -- pass '' to skip requesting a format entirely "
            "and let the driver pick its own default. Run --list-formats "
            "to see what your camera actually supports before assuming "
            "MJPG is right for it too."
        ),
    )

    parser.add_argument(
        "--list-formats",
        dest="list_formats",
        action="store_true",
        help=(
            "Print the real (resolution, pixel format, fps range) "
            "combinations this camera's driver advertises, then exit. "
            "Needs --index (or will prompt like --list does)."
        ),
    )

    parser.add_argument(
        "--report-settings",
        dest="report_settings",
        action="store_true",
        help=(
            "Also print exposure/gain/auto-exposure state on startup. "
            "Read-only here -- use settings.py to change these."
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="Probe camera indices and print what's available, then exit.",
    )

    parser.add_argument(
        "--max-probe-index",
        dest="max_probe_index",
        type=int,
        default=5,
        help="How many indices (0..N-1) to probe for --list or interactive selection.",
    )

    parser.add_argument(
        "--probe-timeout",
        dest="probe_timeout",
        type=float,
        default=3.0,
        help="Seconds to wait for each index during probing before giving up on it.",
    )

    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        default="captures",
        help="Directory to save frames into when pressing 's'.",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
