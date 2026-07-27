"""
Background-reference capture for difference-based fragment detection.

Captured once, in the exact same corrected coordinate system used for
live detection (undistorted, then cropped to main_window.crop_percentages
-- see tabs/detection_tab.py), under normal LEDs with no intended
contact/light. Because it's already in that same coordinate system, every
live frame lines up with it pixel-for-pixel with no further registration
step.

Saved as a plain grayscale array via numpy, not JSON -- a full-resolution
image doesn't fit that format sensibly the way scale.py's/distortion.py's
small scalar results do.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_BACKGROUND_PATH = Path(__file__).parent / "background_reference.npy"


def save_background_reference(gray_frame: np.ndarray, path: Path = DEFAULT_BACKGROUND_PATH) -> None:
    np.save(path, gray_frame)


def load_background_reference(path: Path = DEFAULT_BACKGROUND_PATH) -> np.ndarray | None:
    path = Path(path)

    if not path.exists():
        return None

    try:
        return np.load(path)
    except (OSError, ValueError):
        return None


def compute_difference(current_gray: np.ndarray, background_gray: np.ndarray) -> np.ndarray:
    """
    Nonnegative difference: current - background, clipped at 0. Static
    structure present in both frames (fixture reflections, LED hot spots,
    ambient glare baked into the glass) subtracts to ~0 and disappears;
    genuinely new illumination (real contact) survives as a positive
    value. Pixels that got DARKER than the background (e.g. something now
    blocking existing ambient light) are clipped to 0, not reported as
    negative "light" -- this function never returns a negative value.

    Computed in int16 headroom (0-255 minus 0-255 can be -255..255) before
    clipping back to uint8, so the subtraction itself never wraps/
    overflows the way two uint8 arrays subtracted directly would.
    """
    diff = current_gray.astype(np.int16) - background_gray.astype(np.int16)
    return np.clip(diff, 0, 255).astype(np.uint8)
