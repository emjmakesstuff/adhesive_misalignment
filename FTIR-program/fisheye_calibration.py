"""
Fisheye-lens ChArUco calibration -- the ELP 170-degree camera's model,
parallel to distortion.py's pinhole CharucoCalibrator but never touching
that file. Mirrors its API shape exactly (same constructor, detect()/
capture_frame()/clear()/capture_count/calibrate()) so CalibrationTab can
hold either calibrator behind the same call sites, branching only on
which class to instantiate.

A 170-degree lens is far outside the pinhole model's validity (the
standard model breaks down well before 180 degrees FOV) -- cv2.fisheye
implements Kannala-Brandt's equidistant model instead, with its own
4-coefficient distortion vector (k1-k4, not pinhole's 5) and its own
calibration/undistort functions (cv2.fisheye.calibrate(),
cv2.fisheye.initUndistortRectifyMap()) expecting differently-shaped
inputs than the plain cv2.* equivalents.

distortion.undistort_with_maps() (a plain cv2.remap() call) and
distortion.crop_edges() (pure array slicing) have no notion of which
calibration model produced their input and are reused unchanged --
only *building* the undistort maps differs between models.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np

from distortion import ARUCO_DICTIONARIES, undistort_with_maps  # noqa: F401 -- re-exported for callers that want one import surface

DEFAULT_CALIBRATION_PATH = Path(__file__).parent / "calibration_data_fisheye.json"

# Higher than distortion.MIN_CALIBRATION_VIEWS's 15 (kept as its own
# constant here rather than importing that one, to keep this module a
# genuinely independent parallel implementation, not coupled to
# pinhole's). Confirmed directly against synthetic ground-truth data:
# with a generic (non-truth) initial guess, 20 views was borderline --
# fx/fy converged reasonably but the higher-order k3/k4 terms sometimes
# came out wildly implausible (one run: k3=571, k4=-848 against a true
# ~0.01/-0.005) and correctly tripped the sanity check below. 25+ views
# converged cleanly and repeatably (fx within ~2%, no warning) across
# repeated runs. Fisheye's 4-parameter equidistant model is simply more
# sensitive to view count than pinhole's -- this is not a bug to fix,
# it's why the sanity check and this higher minimum both exist.
MIN_CALIBRATION_VIEWS = 25


class CharucoCalibratorFisheye:
    """
    Same accumulate-then-calibrate shape as distortion.CharucoCalibrator
    -- see that class's docstring. detect()/capture_frame() are
    byte-for-byte the same ChArUco corner-finding logic (a board is a
    board regardless of which lens model will consume its points), only
    calibrate() actually diverges.
    """

    def __init__(
        self,
        squares_x: int,
        squares_y: int,
        square_length_mm: float,
        marker_length_mm: float,
        dictionary_name: str = "DICT_5X5_100",
        legacy_pattern: bool = False,
    ) -> None:
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length_mm = square_length_mm
        self.marker_length_mm = marker_length_mm
        self.dictionary_name = dictionary_name
        self.legacy_pattern = legacy_pattern

        self.dictionary = aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary_name])
        self.board = aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length_mm,
            marker_length_mm,
            self.dictionary,
        )
        self.board.setLegacyPattern(legacy_pattern)
        self.detector = aruco.CharucoDetector(
            self.board,
            aruco.CharucoParameters(),
            aruco.DetectorParameters(),
        )

        self.object_points_list: list = []
        self.image_points_list: list = []
        self.capture_corner_counts: list[int] = []
        self.image_size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(gray)

        annotated = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if marker_ids is not None and len(marker_ids) > 0:
            try:
                aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)
            except cv2.error:
                pass

        corners_ok = charuco_corners is not None and len(charuco_corners) > 0

        if corners_ok:
            try:
                aruco.drawDetectedCornersCharuco(annotated, charuco_corners, charuco_ids)
            except cv2.error:
                pass

        corner_count = len(charuco_corners) if corners_ok else 0
        marker_count = 0 if marker_ids is None else len(marker_ids)

        return {
            "charuco_corners": charuco_corners,
            "charuco_ids": charuco_ids,
            "annotated": annotated,
            "corner_count": corner_count,
            "marker_count": marker_count,
        }

    def capture_frame(self, frame: np.ndarray, min_corners: int = 6) -> dict:
        result = self.detect(frame)
        added = False

        if result["corner_count"] >= min_corners:
            obj_pts, img_pts = self.board.matchImagePoints(
                result["charuco_corners"], result["charuco_ids"]
            )

            if obj_pts is not None and len(obj_pts) >= min_corners:
                # cv2.fisheye.calibrate() expects float64 points shaped
                # (1, N, 3)/(1, N, 2) per view -- matchImagePoints()
                # returns (N, 1, 3)/(N, 1, 2) float32 (the shape
                # cv2.calibrateCamera wants instead, confirmed working in
                # distortion.py). Reshaped once here, at capture time,
                # not deferred to calibrate() -- keeps calibrate() itself
                # a simple pass-through of already-correctly-shaped data.
                obj_pts = obj_pts.reshape(1, -1, 3).astype(np.float64)
                img_pts = img_pts.reshape(1, -1, 2).astype(np.float64)

                self.object_points_list.append(obj_pts)
                self.image_points_list.append(img_pts)
                self.capture_corner_counts.append(result["corner_count"])
                self.image_size = (frame.shape[1], frame.shape[0])
                added = True

        result["added"] = added
        result["total_captures"] = len(self.object_points_list)
        return result

    def clear(self) -> None:
        self.object_points_list = []
        self.image_points_list = []
        self.capture_corner_counts = []

    @property
    def capture_count(self) -> int:
        return len(self.object_points_list)

    def calibrate(self, min_views: int = MIN_CALIBRATION_VIEWS) -> dict:
        if self.capture_count < min_views:
            raise RuntimeError(
                f"Need at least {min_views} captured views, have "
                f"{self.capture_count}. Capture more frames from "
                f"different angles/positions/distances before calibrating."
            )

        if self.image_size is None:
            raise RuntimeError("No captured frames to determine image size.")

        width, height = self.image_size

        # NOT the pinhole path's fx=fy=width guess -- confirmed directly
        # against synthetic ground-truth data that this is badly wrong
        # for a wide fisheye lens (a 170-degree lens needs a MUCH smaller
        # focal length relative to image width than a normal lens, since
        # the same fx maps a far wider real-world angle per pixel; a
        # width-based guess was off by ~3.7x from a synthetic true fx and
        # the solver diverged completely, fx/fy even coming out negative
        # in one run). Used instead: max(width,height)/pi, OpenCV's own
        # documented rule-of-thumb fisheye focal-length convention
        # (referenced in cv2.fisheye.calibrate()'s own docstring under
        # CALIB_FIX_FOCAL_LENGTH) -- confirmed this recovers a
        # focal length within ~5% of a known synthetic ground truth,
        # versus the width-based guess's ~3.7x error and outright
        # divergence.
        f_init = max(width, height) / np.pi
        K_init = np.array(
            [
                [f_init, 0.0, width / 2.0],
                [0.0, f_init, height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        D_init = np.zeros((4, 1))

        # CALIB_RECOMPUTE_EXTRINSIC deliberately omitted -- confirmed
        # directly that it crashes in this OpenCV build (5.0.0) with
        # "Assertion failed: fabs(norm_u1) > 0 in InitExtrinsics",
        # reproducible even on clean, well-conditioned synthetic data,
        # so this isn't a data-quality problem to work around, it's this
        # build's implementation. Manually iterating (feeding one pass's
        # result back in as the next pass's initial guess, as a
        # replacement for what that flag would have done) was also tried
        # and made things WORSE, not better -- confirmed directly it
        # diverged from a reasonable first-pass result (reprojection
        # error ~17px) to a completely broken one (~8e8px) within three
        # more passes. A single calibrate() call with the good initial
        # guess above is the approach that actually works.
        #
        # CALIB_CHECK_COND also omitted: it raises a hard cv2.error on
        # any single ill-conditioned view instead of just fitting around
        # it, which would turn one so-so capture into a full restart --
        # _sanity_check_calibration_fisheye() below catches a genuinely
        # bad overall RESULT instead, the same "check the outcome, don't
        # reject the process" approach already used for the pinhole path.
        #
        # Flags live under the top-level cv2 namespace in this OpenCV
        # build, not cv2.fisheye -- confirmed directly
        # (cv2.fisheye.CALIB_FIX_SKEW raised AttributeError;
        # cv2.fisheye.calibrate()'s own docstring references these as
        # @ref cv::CALIB_*, i.e. the plain cv2.* names).
        flags = cv2.CALIB_FIX_SKEW | cv2.CALIB_USE_INTRINSIC_GUESS

        reprojection_error, K, D, _, _ = cv2.fisheye.calibrate(
            self.object_points_list,
            self.image_points_list,
            self.image_size,
            K_init,
            D_init,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        )

        warning = _sanity_check_calibration_fisheye(K, D, self.image_size)

        result = {
            "model": "fisheye",
            "reprojection_error": float(reprojection_error),
            "camera_matrix": K.tolist(),
            "dist_coeffs": D.flatten().tolist(),
            "image_size": list(self.image_size),
            "num_captures": self.capture_count,
            "board_params": {
                "squares_x": self.squares_x,
                "squares_y": self.squares_y,
                "square_length_mm": self.square_length_mm,
                "marker_length_mm": self.marker_length_mm,
                "dictionary_name": self.dictionary_name,
                "legacy_pattern": self.legacy_pattern,
            },
        }

        if warning is not None:
            result["warning"] = warning

        return result


def _sanity_check_calibration_fisheye(K: np.ndarray, D: np.ndarray, image_size: tuple[int, int]) -> str | None:
    """
    Same purpose as distortion._sanity_check_calibration -- reprojection
    error alone doesn't catch an overfit/underconstrained result.
    Fisheye's own coefficient scale differs from pinhole's (4 terms,
    generally smaller magnitude for a real equidistant lens), so this is
    its own bound set, not shared with the pinhole check.
    """
    width, height = image_size
    fx, fy = K[0, 0], K[1, 1]
    k1, k2, k3, k4 = (D.flatten().tolist() + [0, 0, 0, 0])[:4]

    reasons = []

    if not (0.15 * width <= fx <= 3.0 * width):
        reasons.append(f"focal length fx={fx:.0f} is implausible for a {width}px-wide fisheye image")

    if max(abs(k1), abs(k2), abs(k3), abs(k4)) > 5:
        reasons.append(
            f"distortion coefficients are far outside a normal fisheye lens's range "
            f"(k1={k1:.3f}, k2={k2:.3f}, k3={k3:.3f}, k4={k4:.3f})"
        )

    if not reasons:
        return None

    return (
        "This fisheye calibration looks overfit, not correct: " + "; ".join(reasons) + ". "
        "This usually means too few captured views, or views that are too "
        "similar to each other. Clear captures and redo with at least "
        f"{MIN_CALIBRATION_VIEWS} views spread across the whole frame -- "
        "close/far, tilted in different directions, touching each corner "
        "and edge, including the extreme wide-angle edges this lens sees -- "
        "before trusting this."
    )


def save_calibration(data: dict, path: Path = DEFAULT_CALIBRATION_PATH) -> None:
    with open(Path(path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_calibration(path: Path = DEFAULT_CALIBRATION_PATH) -> dict | None:
    path = Path(path)

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def build_undistort_maps(calibration: dict):
    """
    Fisheye equivalent of distortion.build_undistort_maps() -- applying
    the resulting maps is identical either way (distortion.
    undistort_with_maps() is plain cv2.remap(), reused unchanged, see
    this module's import at the top).
    """
    K = np.array(calibration["camera_matrix"])
    D = np.array(calibration["dist_coeffs"]).reshape(-1, 1)
    width, height = calibration["image_size"]

    # balance=1.0: keep the full field of view (accepting black corners),
    # the fisheye equivalent of the pinhole path's alpha=1 choice -- same
    # "maximize retained pixels" rationale.
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (width, height), np.eye(3), balance=1.0, new_size=(width, height)
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (width, height), cv2.CV_16SC2
    )

    return map1, map2
