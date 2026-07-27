"""
Direct control of the camera's Video-Proc-Amp-equivalent properties
(brightness, contrast, saturation, hue, gamma, sharpness, backlight
compensation, gain, white balance) via plain cv2.VideoCapture.get/set()
-- the exact mechanism settings.py already uses successfully for
exposure/gain, extended here to the rest of the properties shown in the
camera's native "Video Proc Amp" dialog.

An earlier version of this went through the raw DirectShow
IAMVideoProcAmp/IAMCameraControl COM interfaces directly, specifically to
get exact hardware min/max/step ranges the way the native dialog does.
That path is not used: QueryInterface reliably failed with E_NOINTERFACE
in this environment even though cv2's own equivalent CAP_PROP_* calls
succeeded on the same devices (confirmed directly) -- most likely because
the documented, correct way to locate these interfaces,
ICaptureGraphBuilder2::FindInterface, needs a connected/rendered graph,
which risks conflicting with an already-open capture stream. Since cv2
already reaches the same underlying properties successfully and is
proven throughout this project, that's what's used here.

Tradeoff: OpenCV doesn't expose real GetRange() min/max/step, so the
ranges below are reasonable generic defaults, not exact per-camera
hardware limits. Every set is verified via readback, retrying on
mismatch (same pattern as settings.py's exposure/gain functions, for the
same reason: the first control-change request after opening a device can
be silently dropped). A persistent mismatch after retries usually means
the driver clamped the request to its own real range, not that the call
failed -- what's displayed is always the verified ground truth either way.
"""

from __future__ import annotations

import cv2

# name -> {cv2 property id, generic default UI range}
CONTROLS = {
    "brightness": {"prop": cv2.CAP_PROP_BRIGHTNESS, "min": -64, "max": 64},
    "contrast": {"prop": cv2.CAP_PROP_CONTRAST, "min": 0, "max": 95},
    "saturation": {"prop": cv2.CAP_PROP_SATURATION, "min": 0, "max": 100},
    "hue": {"prop": cv2.CAP_PROP_HUE, "min": -180, "max": 180},
    "gamma": {"prop": cv2.CAP_PROP_GAMMA, "min": 1, "max": 500},
    "sharpness": {"prop": cv2.CAP_PROP_SHARPNESS, "min": 0, "max": 7},
    "backlight_compensation": {"prop": cv2.CAP_PROP_BACKLIGHT, "min": 0, "max": 2},
    "gain": {"prop": cv2.CAP_PROP_GAIN, "min": 0, "max": 255},
    "wb_temperature": {"prop": cv2.CAP_PROP_WB_TEMPERATURE, "min": 2000, "max": 10000},
}

AUTO_WB_PROP = cv2.CAP_PROP_AUTO_WB


def get_value(cap: cv2.VideoCapture, name: str) -> float:
    return cap.get(CONTROLS[name]["prop"])


def set_value(cap: cv2.VideoCapture, name: str, value: float, retries: int = 2) -> float:
    prop = CONTROLS[name]["prop"]
    cap.set(prop, value)
    result = cap.get(prop)

    attempt = 0
    while result != value and attempt < retries:
        cap.set(prop, value)
        result = cap.get(prop)
        attempt += 1

    return result


def get_auto_wb(cap: cv2.VideoCapture) -> bool:
    return cap.get(AUTO_WB_PROP) != 0


def set_auto_wb(cap: cv2.VideoCapture, enabled: bool) -> bool:
    cap.set(AUTO_WB_PROP, 1.0 if enabled else 0.0)
    return get_auto_wb(cap)


def get_all(cap: cv2.VideoCapture) -> dict:
    values = {name: get_value(cap, name) for name in CONTROLS}
    values["auto_wb"] = get_auto_wb(cap)
    return values
