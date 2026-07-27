"""
Stage 1 is capture + preview only. This is a deliberate passthrough so
bright-spot detection can be added here later without changing camera.py
or preview.py at all.
"""

import numpy as np


def process_frame(frame: np.ndarray) -> np.ndarray:
    return frame
