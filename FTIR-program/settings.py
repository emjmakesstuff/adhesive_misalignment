"""
Stage 2: camera settings inspection and locking.

Exposure and gain must be fixed (not auto) before any brightness
measurement means anything -- auto-exposure silently re-brightening or
darkening the image would invalidate every comparison a later stage makes.

Ported from the vendor's U20CAM-9281M Windows control script: DirectShow
expresses exposure as log2(seconds), not the UVC 100us units used on
Linux/macOS, and manual/auto exposure mode is a boolean-ish flag (0.25 =
manual, 0.75 = auto) rather than a real on/off property.
"""

from __future__ import annotations

import argparse
import math

import cv2

from camera import BACKENDS, CameraStream


def ms_to_log2(exposure_ms: float) -> int:
    """Convert milliseconds to DirectShow's log2-seconds exposure value,
    rounded to the nearest integer step (DirectShow camera filters
    typically only accept integer log2 steps)."""
    seconds = exposure_ms / 1000.0

    if seconds <= 0:
        return -13  # smallest practical exposure

    return round(math.log2(seconds))


def log2_to_ms(log2_value: float) -> float:
    """Convert a DirectShow log2-seconds exposure value back to milliseconds."""
    return (2**log2_value) * 1000.0


def get_exposure_gain_info(cap: cv2.VideoCapture) -> dict:
    """
    Read the current exposure/gain/auto-exposure state. Only meaningful
    if the backend/driver actually exposes these properties (DSHOW does
    for the U20CAM-9281M; other backends or cameras may return 0/-1,
    which is itself useful information -- it means these controls aren't
    available and can't be locked here).
    """
    auto_exposure_raw = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    exposure_log2 = cap.get(cv2.CAP_PROP_EXPOSURE)
    gain = cap.get(cv2.CAP_PROP_GAIN)

    if abs(auto_exposure_raw - 0.25) < 0.01:
        mode = "manual"
    elif abs(auto_exposure_raw - 0.75) < 0.01:
        mode = "auto"
    else:
        mode = f"unknown (raw value {auto_exposure_raw})"

    return {
        "auto_exposure_raw": auto_exposure_raw,
        "mode": mode,
        "exposure_log2": exposure_log2,
        "exposure_ms": log2_to_ms(exposure_log2) if exposure_log2 != 0 else 0.0,
        "gain": gain,
    }


def set_manual_exposure(cap: cv2.VideoCapture, exposure_ms: float, retries: int = 2) -> dict:
    """
    Switch to manual exposure mode and request a specific exposure time.
    Returns the verified readback (via get_exposure_gain_info), since the
    driver rounds to the nearest supported log2 step rather than granting
    the exact millisecond value requested.

    Retries on mismatch: observed directly against real hardware that the
    very first control-change request after opening a device can be
    silently dropped, with every subsequent identical request succeeding.
    Rather than trust the first attempt, this verifies the readback and
    retries before reporting a real failure.
    """
    target_log2 = ms_to_log2(exposure_ms)
    info = get_exposure_gain_info(cap)

    attempt = 0
    while info["exposure_log2"] != target_log2 and attempt <= retries:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # DirectShow: 0.25 = manual
        cap.set(cv2.CAP_PROP_EXPOSURE, target_log2)
        info = get_exposure_gain_info(cap)
        attempt += 1

    if info["exposure_log2"] != target_log2:
        print(
            f"  WARNING: requested exposure log2={target_log2} "
            f"({exposure_ms:.2f} ms) was not honored after {attempt} "
            f"attempt(s) -- driver reports {info['exposure_log2']} "
            f"({info['exposure_ms']:.2f} ms) instead. This may not be a "
            f"supported exposure step, or auto-exposure/gain controls may "
            f"not be exposed by this driver at all."
        )

    return info


def set_auto_exposure(cap: cv2.VideoCapture) -> dict:
    """Restore auto exposure (DirectShow: 0.75 = auto)."""
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    return get_exposure_gain_info(cap)


def set_gain(cap: cv2.VideoCapture, gain_value: float, retries: int = 2) -> dict:
    """
    Set gain. Only reliably meaningful once exposure is manual -- under
    auto-exposure the driver may ignore or override a requested gain.
    Retries on mismatch for the same reason as set_manual_exposure.
    """
    info = get_exposure_gain_info(cap)
    attempt = 0

    while info["gain"] != gain_value and attempt <= retries:
        cap.set(cv2.CAP_PROP_GAIN, gain_value)
        info = get_exposure_gain_info(cap)
        attempt += 1

    if info["gain"] != gain_value:
        print(
            f"  WARNING: requested gain {gain_value} was not honored after "
            f"{attempt} attempt(s) -- driver reports {info['gain']} instead. "
            f"This may be outside the camera's supported gain range."
        )

    return info


def open_properties_dialog(index: int) -> bool:
    """
    Open the camera's native DirectShow property dialog -- the same
    tabbed "Video Proc Amp" / "Camera Control" window AMCap shows, with
    sliders for every control the driver exposes (brightness, contrast,
    hue, saturation, sharpness, gamma, white balance, backlight
    compensation, gain, exposure, focus, etc). This is not a custom UI:
    it's the identical native dialog, invoked directly. Blocks until the
    user closes it. Returns True if the dialog was shown.

    Uses a hidden Tkinter window as the dialog's owner handle rather than
    pygrabber's own show_properties() helper, which resolves its owner
    via GetTopWindow(None). That returned no usable window in an
    automated/remote test session and made the dialog fail outright with
    a generic COM error; an explicit owner window fixed it (confirmed
    directly, including that the driver reports 2 property pages, i.e.
    both tabs, matching AMCap's dialog exactly).
    """
    try:
        import tkinter as tk
        from ctypes import byref, cast

        from pygrabber.dshow_core import ISpecifyPropertyPages
        from pygrabber.dshow_graph import FilterGraph
        from pygrabber.win_api_extra import LPUNKNOWN, OleCreatePropertyFrame
    except ImportError as error:
        raise RuntimeError(
            "pygrabber and comtypes are required for --configure "
            "(pip install pygrabber comtypes)"
        ) from error

    root = tk.Tk()
    root.withdraw()

    try:
        graph = FilterGraph()
        graph.add_video_input_device(index)
        instance = graph.get_input_device().instance

        spec_pages = instance.QueryInterface(ISpecifyPropertyPages)
        cauuid = spec_pages.GetPages()

        if cauuid.element_count == 0:
            print("This camera reports no configurable properties.")
            return False

        OleCreatePropertyFrame(
            root.winfo_id(),
            0,
            0,
            None,
            1,
            byref(cast(instance, LPUNKNOWN)),
            cauuid.element_count,
            cauuid.elements,
            0,
            0,
            None,
        )
        return True
    finally:
        root.destroy()


def print_settings_report(info: dict) -> None:
    print(f"  Auto-exposure mode:   {info['mode']}")
    print(
        f"  Exposure:             {info['exposure_log2']} (log2 sec) "
        f"= {info['exposure_ms']:.2f} ms"
    )
    print(f"  Gain:                 {info['gain']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: inspect or lock camera exposure/gain settings."
    )

    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Camera index to open (see preview.py --list).",
    )

    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS.keys()),
        default="dshow",
        help="OpenCV capture backend. dshow is required for exposure/gain control on Windows.",
    )

    parser.add_argument(
        "--exposure-ms",
        dest="exposure_ms",
        type=float,
        default=None,
        help="Switch to manual exposure and request this exposure time in milliseconds.",
    )

    parser.add_argument(
        "--auto-exposure",
        dest="auto_exposure",
        action="store_true",
        help="Restore auto exposure.",
    )

    parser.add_argument(
        "--gain",
        type=float,
        default=None,
        help="Set gain. Apply after --exposure-ms if setting both, since gain " "is only reliable under manual exposure.",
    )

    parser.add_argument(
        "--configure",
        action="store_true",
        help=(
            "Open the camera's native property dialog (the same one AMCap "
            "shows) with sliders for every control the driver exposes -- "
            "brightness, contrast, hue, saturation, sharpness, gamma, "
            "white balance, backlight compensation, gain, exposure, focus, "
            "etc. Blocks until you close it, then prints the resulting "
            "exposure/gain readback. Ignores --exposure-ms/--auto-exposure/"
            "--gain if also given."
        ),
    )

    args = parser.parse_args()

    if args.configure:
        try:
            shown = open_properties_dialog(args.index)
        except RuntimeError as error:
            print(f"ERROR: {error}")
            return

        if not shown:
            return

        stream = CameraStream(index=args.index, backend=BACKENDS[args.backend])

        try:
            stream.open()
        except RuntimeError as error:
            print(f"ERROR: {error}")
            return

        print("Exposure/gain settings after configuring:")
        print_settings_report(get_exposure_gain_info(stream.cap))
        stream.release()
        return

    stream = CameraStream(index=args.index, backend=BACKENDS[args.backend])

    try:
        stream.open()
    except RuntimeError as error:
        print(f"ERROR: {error}")
        return

    cam_info = stream.get_info()
    print("Camera info:")
    print(f"  Backend:    {cam_info['backend_name']}")
    print(f"  Resolution: {cam_info['actual_width']} x {cam_info['actual_height']}")
    print(f"  FourCC:     {cam_info['fourcc']}")

    cap = stream.cap

    if args.auto_exposure:
        print("Setting auto exposure...")
        info = set_auto_exposure(cap)
    elif args.exposure_ms is not None:
        print(f"Setting manual exposure: target {args.exposure_ms} ms...")
        info = set_manual_exposure(cap, args.exposure_ms)
    else:
        info = get_exposure_gain_info(cap)

    if args.gain is not None:
        print(f"Setting gain to {args.gain}...")
        info = set_gain(cap, args.gain)

    print("Exposure/gain settings:")
    print_settings_report(info)

    stream.release()


if __name__ == "__main__":
    main()
