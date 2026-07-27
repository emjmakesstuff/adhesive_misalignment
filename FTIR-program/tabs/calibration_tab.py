"""
Calibration tab: ChArUco-board lens distortion calibration (Stage 3) and
physical scale calibration (Stage 4) -- both scoped to a per-source
calibration profile (calibration_profiles.py) so the USB camera and a
VDO.Ninja stream never read or overwrite each other's lens/scale data.

Board parameters (grid size, square/marker length, ArUco dictionary) are
input fields, not hardcoded -- the physical board in use today is 8x8
squares/20mm squares/15mm markers, but this needs to keep working
unchanged if that's swapped for a 6x6 board or a different dictionary.
This part of the calibration process is identical for either source type
(a lens is a lens; distortion.py doesn't care what fed it the frame) --
only WHERE the result is saved differs, via the active profile's path.

Reads live frames from main_window.latest_frame (set by LiveCameraTab on
every successful read) rather than calling stream.read() itself, so this
tab's own live detection-preview timer never competes with LiveCameraTab
for the same capture stream.
"""

from __future__ import annotations

import cv2
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import calibration_profiles
import distortion
import scale as scale_module  # avoid shadowing the local "scale factor" variables below


class ClickableImageLabel(QLabel):
    """A QLabel that reports where it was clicked, in the label's own
    widget-pixel coordinates. Translating that to original-image pixel
    coordinates is the caller's job (it depends on how the caller scaled
    the image into the label), not this widget's."""

    clicked = Signal(int, int)

    def mousePressEvent(self, event) -> None:
        point = event.position().toPoint()
        self.clicked.emit(point.x(), point.y())
        super().mousePressEvent(event)


class CalibrationTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.calibrator: distortion.CharucoCalibrator | None = None

        # Stage 4 (physical scale) state. scale_display_info holds
        # (display_scale, offset_x, offset_y) for the currently-shown
        # frozen frame, computed by _render_scale_frame and consumed by
        # _on_scale_image_clicked to map a click back to image pixels --
        # kept explicit rather than relying on QPixmap.scaled()'s
        # internal size choice, so the inverse mapping is guaranteed
        # consistent with what's actually on screen.
        self.scale_frame = None
        self.scale_points: list[tuple[float, float]] = []
        self.scale_display_info: tuple[float, float, float] | None = None
        # "calibrate": the next 2 clicks + a known distance compute and
        # save a new scale. "verify": the next 2 clicks measure a
        # distance using the *already-saved* scale, to sanity-check it
        # against a distance you separately know -- doesn't touch the
        # saved file.
        self.scale_mode = "calibrate"

        # Everything below is built exactly as before -- the only change
        # is that `layout` is now parented to a scrollable content widget
        # instead of directly to the tab, since the ChArUco + physical
        # scale sections together are comfortably taller than most
        # windows (two preview images plus all their controls stacked in
        # one column) and were overlapping/clipping without this.
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(scroll_area)

        content = QWidget()
        scroll_area.setWidget(content)

        layout = QVBoxLayout(content)

        profile_box = QGroupBox("Calibration profile")
        profile_layout = QHBoxLayout(profile_box)
        profile_layout.addWidget(QLabel("Profile:"))
        self.profile_combo = QComboBox()
        profile_layout.addWidget(self.profile_combo, 1)
        self.new_profile_button = QPushButton("New Profile...")
        profile_layout.addWidget(self.new_profile_button)
        layout.addWidget(profile_box)

        self.profile_missing_label = QLabel(
            "No profile exists yet for the active source -- create one "
            "above before calibrating. Profiles keep each camera "
            "source's lens calibration, physical scale, and usable ROI "
            "completely separate; switching sources never reads or "
            "overwrites another source's profile."
        )
        self.profile_missing_label.setWordWrap(True)
        layout.addWidget(self.profile_missing_label)

        board_box = QGroupBox("Board parameters")
        self.board_box = board_box  # disabled as a whole while no profile is active
        board_form = QFormLayout(board_box)

        self.squares_x_spin = QSpinBox()
        self.squares_x_spin.setRange(2, 30)
        self.squares_x_spin.setValue(8)
        board_form.addRow("Squares (X):", self.squares_x_spin)

        self.squares_y_spin = QSpinBox()
        self.squares_y_spin.setRange(2, 30)
        self.squares_y_spin.setValue(8)
        board_form.addRow("Squares (Y):", self.squares_y_spin)

        self.square_length_spin = QDoubleSpinBox()
        self.square_length_spin.setRange(0.1, 500.0)
        self.square_length_spin.setDecimals(2)
        self.square_length_spin.setSuffix(" mm")
        self.square_length_spin.setValue(20.0)
        board_form.addRow("Square length:", self.square_length_spin)

        self.marker_length_spin = QDoubleSpinBox()
        self.marker_length_spin.setRange(0.1, 500.0)
        self.marker_length_spin.setDecimals(2)
        self.marker_length_spin.setSuffix(" mm")
        self.marker_length_spin.setValue(15.0)
        board_form.addRow("Marker length:", self.marker_length_spin)

        self.dictionary_combo = QComboBox()
        self.dictionary_combo.addItems(sorted(distortion.ARUCO_DICTIONARIES.keys()))
        self.dictionary_combo.setCurrentText("DICT_5X5_100")
        board_form.addRow("ArUco dictionary:", self.dictionary_combo)

        self.legacy_pattern_check = QCheckBox(
            "Legacy pattern (try this if markers detect but corners stay at 0)"
        )
        self.legacy_pattern_check.setToolTip(
            "OpenCV changed the ChArUco corner/marker layout convention "
            "around version 4.6. A board generated by another tool or an "
            "older OpenCV version may use the old layout -- markers still "
            "detect fine (dictionary-based, layout-independent), but "
            "corner interpolation silently returns zero corners against "
            "the wrong layout. If markers show but the corner count never "
            "leaves 0, toggle this and click Create/Update Board again."
        )
        board_form.addRow(self.legacy_pattern_check)

        self.create_board_button = QPushButton("Create / Update Board")
        board_form.addRow(self.create_board_button)

        layout.addWidget(board_box)

        self.preview_label = QLabel("Create a board, then point the camera at it.")
        self.preview_label.setMinimumSize(480, 360)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(self.preview_label, 1)

        self.detection_status_label = QLabel("No board created yet.")
        layout.addWidget(self.detection_status_label)

        capture_row = QHBoxLayout()
        self.capture_button = QPushButton("Capture Frame")
        self.capture_button.setEnabled(False)
        capture_row.addWidget(self.capture_button)
        self.clear_button = QPushButton("Clear Captures")
        self.clear_button.setEnabled(False)
        capture_row.addWidget(self.clear_button)
        self.calibrate_button = QPushButton("Run Calibration")
        self.calibrate_button.setEnabled(False)
        capture_row.addWidget(self.calibrate_button)
        layout.addLayout(capture_row)

        self.captures_label = QLabel(f"Captured views: 0 / {distortion.MIN_CALIBRATION_VIEWS} minimum")
        layout.addWidget(self.captures_label)

        capture_help_label = QLabel(
            "For each capture, move/tilt the board to a new position before "
            "capturing again -- vary distance (close and far), angle "
            "(tilted, not just flat-on), and position (cover the frame's "
            "edges and corners, not just the center). Too few or too "
            "similar captures produce a calibration that fits those views "
            "but distorts everything else -- worse than doing nothing."
        )
        capture_help_label.setWordWrap(True)
        layout.addWidget(capture_help_label)

        self.result_label = QLabel(
            f"No calibration saved yet "
            f"(will be written to {distortion.DEFAULT_CALIBRATION_PATH.name})."
        )
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        scale_box = QGroupBox("Physical scale (Stage 4)")
        self.scale_box = scale_box  # disabled as a whole while no profile is active
        scale_layout = QVBoxLayout(scale_box)

        self.scale_status_label = QLabel("Complete lens-distortion calibration above first.")
        self.scale_status_label.setWordWrap(True)
        scale_layout.addWidget(self.scale_status_label)

        self.scale_preview_label = ClickableImageLabel("No frame captured yet.")
        self.scale_preview_label.setMinimumSize(480, 360)
        self.scale_preview_label.setAlignment(Qt.AlignCenter)
        self.scale_preview_label.setStyleSheet("background-color: black; color: white;")
        scale_layout.addWidget(self.scale_preview_label, 1)

        scale_capture_row = QHBoxLayout()
        self.scale_capture_button = QPushButton("Capture Frame for Scale")
        self.scale_capture_button.setEnabled(False)
        scale_capture_row.addWidget(self.scale_capture_button)
        self.scale_retake_button = QPushButton("Retake")
        self.scale_retake_button.setEnabled(False)
        scale_capture_row.addWidget(self.scale_retake_button)
        scale_layout.addLayout(scale_capture_row)

        scale_input_row = QHBoxLayout()
        scale_input_row.addWidget(QLabel("Known distance between the two points:"))
        self.scale_distance_spin = QDoubleSpinBox()
        self.scale_distance_spin.setRange(0.01, 10000.0)
        self.scale_distance_spin.setDecimals(3)
        self.scale_distance_spin.setSuffix(" mm")
        self.scale_distance_spin.setValue(100.0)
        scale_input_row.addWidget(self.scale_distance_spin)
        self.scale_compute_button = QPushButton("Compute && Save Scale")
        self.scale_compute_button.setEnabled(False)
        scale_input_row.addWidget(self.scale_compute_button)
        scale_layout.addLayout(scale_input_row)

        self.scale_points_label = QLabel("Click two points a known distance apart.")
        scale_layout.addWidget(self.scale_points_label)

        self.scale_result_label = QLabel(
            f"No scale saved yet (will be written to {scale_module.DEFAULT_SCALE_PATH.name})."
        )
        self.scale_result_label.setWordWrap(True)
        scale_layout.addWidget(self.scale_result_label)

        verify_row = QHBoxLayout()
        self.verify_scale_button = QPushButton("Verify Scale (click 2 points)")
        self.verify_scale_button.setEnabled(False)
        self.verify_scale_button.setToolTip(
            "Click two points of a distance you separately know (e.g. two "
            "other marks on a ruler) and check the measured distance "
            "matches -- doesn't change the saved scale."
        )
        verify_row.addWidget(self.verify_scale_button)
        verify_row.addStretch(1)
        scale_layout.addLayout(verify_row)

        self.scale_verify_label = QLabel("")
        self.scale_verify_label.setWordWrap(True)
        scale_layout.addWidget(self.scale_verify_label)

        layout.addWidget(scale_box)

        layout.addStretch(1)

        self.profile_combo.currentIndexChanged.connect(self._on_profile_combo_changed)
        self.new_profile_button.clicked.connect(self._create_new_profile)

        self.create_board_button.clicked.connect(self.create_board)
        self.capture_button.clicked.connect(self.capture_frame)
        self.clear_button.clicked.connect(self.clear_captures)
        self.calibrate_button.clicked.connect(self.run_calibration)

        self.scale_capture_button.clicked.connect(self.capture_scale_frame)
        self.scale_retake_button.clicked.connect(self.retake_scale_frame)
        self.scale_preview_label.clicked.connect(self._on_scale_image_clicked)
        self.scale_compute_button.clicked.connect(self.compute_and_save_scale)
        self.verify_scale_button.clicked.connect(self.enter_verify_mode)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_preview)
        self.timer.start(150)  # live detection feedback; doesn't need to be fast

        # distortion.load_calibration()/scale_module.load_scale_calibration()
        # already read straight from disk wherever they're used (Live
        # Camera's undistort toggle, scale-frame capture) -- no
        # recalibration is actually required after a restart. Only the
        # labels below defaulted to "No calibration/scale saved yet"
        # regardless, which read as if a fresh calibration was needed.
        # This corrects that on open. _refresh_profile_combo() populates
        # the profile picker for the active source and calls both status
        # methods below once a profile is known.
        self._last_seen_profile_key = self.main_window.active_profile_key
        self._refresh_profile_combo()

    # ---- calibration profiles ----

    def _active_profile_id(self) -> str | None:
        profile = calibration_profiles.get_active_profile(self.main_window.active_profile_key)
        return None if profile is None else profile["id"]

    def _refresh_profile_combo(self) -> None:
        source_type = self.main_window.active_source_type
        key = self.main_window.active_profile_key

        if source_type == "usb":
            # "usb" now spans more than one physical camera -- only list
            # profiles that belong to the SAME physical device as the
            # currently active one (same profile_key()), never every USB
            # profile on file, so picking "Profile" here can't apply the
            # wrong camera's calibration to whichever one is actually
            # connected.
            profiles = (
                [p for p in calibration_profiles.list_profiles(source_type="usb") if calibration_profiles.profile_key(p) == key]
                if key is not None
                else []
            )
        else:
            profiles = calibration_profiles.list_profiles(source_type=source_type)

        active = calibration_profiles.get_active_profile(key) if key is not None else None

        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        for profile in profiles:
            self.profile_combo.addItem(profile["name"], profile["id"])
        if active is not None:
            index = self.profile_combo.findData(active["id"])
            if index >= 0:
                self.profile_combo.setCurrentIndex(index)
        self.profile_combo.blockSignals(False)

        has_profile = active is not None
        self.board_box.setEnabled(has_profile)
        self.scale_box.setEnabled(has_profile)
        self.profile_missing_label.setVisible(not has_profile)

        self._show_saved_calibration_status()
        self._show_saved_scale_status()

    def _on_profile_combo_changed(self, index: int) -> None:
        profile_id = self.profile_combo.itemData(index)

        if profile_id is None:
            return

        calibration_profiles.set_active_profile(self.main_window.active_profile_key, profile_id)
        self.main_window.on_active_profile_changed()
        self._refresh_profile_combo()

    def _create_new_profile(self) -> None:
        name, ok = QInputDialog.getText(self, "New Calibration Profile", "Profile name:")

        if not ok or not name.strip():
            return

        resolution = None
        frame = self.main_window.latest_frame
        if frame is not None:
            resolution = (frame.shape[1], frame.shape[0])

        source_type = self.main_window.active_source_type
        identity_kwargs = {}

        if source_type == "usb":
            # Carry over the currently active profile's physical-camera
            # identity (device_path/role/etc) onto the new one -- this is
            # an alternate calibration for the SAME already-identified
            # camera (e.g. a different board/lighting setup), not a new
            # unrecognized device, so it must never come up in the
            # first-connect naming dialog again.
            current = calibration_profiles.get_active_profile(self.main_window.active_profile_key)
            if current is not None:
                identity_kwargs = dict(
                    camera_role=current.get("camera_role"),
                    device_path=current.get("device_path"),
                    device_name=current.get("device_name"),
                    fourcc=current.get("fourcc"),
                    calibration_model=current.get("calibration_model", "pinhole"),
                    detector_type=current.get("detector_type"),
                )

        profile_id = calibration_profiles.create_profile(
            source_type, name.strip(), resolution=resolution, alias=name.strip(), **identity_kwargs
        )
        self.main_window.active_profile_key = calibration_profiles.profile_key(
            calibration_profiles.get_profile(profile_id)
        )
        self.main_window.on_active_profile_changed()
        self._refresh_profile_combo()

    def _show_saved_calibration_status(self) -> None:
        profile_id = self._active_profile_id()

        if profile_id is None:
            self.result_label.setText("No active profile.")
            return

        calibration = distortion.load_calibration(path=calibration_profiles.distortion_path(profile_id))

        if calibration is None:
            self.result_label.setText(
                f"No calibration saved yet for this profile "
                f"(will be written to {calibration_profiles.distortion_path(profile_id).name})."
            )
            return

        self.result_label.setText(
            f"Loaded saved calibration from {calibration_profiles.distortion_path(profile_id).name}: "
            f"reprojection error {calibration['reprojection_error']:.4f} px "
            f"from {calibration['num_captures']} captured views at "
            f"{calibration['image_size'][0]}x{calibration['image_size'][1]}. "
            f"Already in use -- no need to recalibrate unless the camera, "
            f"lens, or resolution has changed."
        )

    def _show_saved_scale_status(self) -> None:
        profile_id = self._active_profile_id()

        if profile_id is None:
            self.scale_result_label.setText("No active profile.")
            return

        saved = scale_module.load_scale_calibration(path=calibration_profiles.scale_path(profile_id))

        if saved is None:
            self.scale_result_label.setText(
                f"No scale saved yet for this profile "
                f"(will be written to {calibration_profiles.scale_path(profile_id).name})."
            )
            return

        self.scale_result_label.setText(
            f"Loaded saved scale from {calibration_profiles.scale_path(profile_id).name}: "
            f"{saved['mm_per_pixel']:.5f} mm/pixel "
            f"({1 / saved['mm_per_pixel']:.3f} px/mm). Already in use -- no "
            f"need to recalibrate unless the camera, lens, or resolution "
            f"has changed."
        )

    def create_board(self) -> None:
        if self.calibrator is not None and self.calibrator.capture_count > 0:
            self.calibrator = None  # discard: board params changed, old captures don't match

        self.calibrator = distortion.CharucoCalibrator(
            squares_x=self.squares_x_spin.value(),
            squares_y=self.squares_y_spin.value(),
            square_length_mm=self.square_length_spin.value(),
            marker_length_mm=self.marker_length_spin.value(),
            dictionary_name=self.dictionary_combo.currentText(),
            legacy_pattern=self.legacy_pattern_check.isChecked(),
        )

        self.captures_label.setText("Captured views: 0")
        self.detection_status_label.setText("Board created. Point the camera at it.")
        self.capture_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.calibrate_button.setEnabled(False)

    def _update_preview(self) -> None:
        # LiveCameraTab can switch active_profile_key at any time (source
        # type OR which physical USB camera is connected, its own combos,
        # not this tab's) -- polled here the same lightweight way this tab
        # already polls scale-capture readiness every tick, rather than
        # needing a cross-tab signal for it.
        if self.main_window.active_profile_key != self._last_seen_profile_key:
            self._last_seen_profile_key = self.main_window.active_profile_key
            self._refresh_profile_combo()

        self._update_scale_capture_availability()

        frame = self.main_window.latest_frame

        if frame is None:
            return

        if self.calibrator is None:
            self._show_preview(frame)
            return

        result = self.calibrator.detect(frame)
        self._show_preview(result["annotated"])

        expected = (self.calibrator.squares_x - 1) * (self.calibrator.squares_y - 1)

        if result["marker_count"] > 0 and result["corner_count"] == 0:
            # The exact symptom of a legacy/new pattern mismatch: markers
            # detect fine (dictionary lookup, layout-independent) but
            # corner interpolation fails because it's matching against
            # the wrong expected geometry. Confirmed directly this is
            # what that combination means, so say so instead of just
            # showing a bare "0 corners" with no explanation.
            self.detection_status_label.setText(
                f"{result['marker_count']} markers found but 0 corners -- "
                f"try toggling 'Legacy pattern' above and re-creating the "
                f"board (this usually means a legacy/new ChArUco layout "
                f"mismatch, not a detection-quality problem)."
            )
        else:
            self.detection_status_label.setText(
                f"Corners detected: {result['corner_count']} / {expected} possible "
                f"({result['marker_count']} markers found)"
            )

    def _show_preview(self, frame) -> None:
        display = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, channels = rgb.shape
        qimage = QImage(rgb.data, w, h, channels * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())

        scaled = pixmap.scaled(
            self.preview_label.width(),
            self.preview_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def capture_frame(self) -> None:
        if self.calibrator is None:
            return

        frame = self.main_window.latest_frame

        if frame is None:
            self.detection_status_label.setText("No live frame available yet -- connect the camera first.")
            return

        result = self.calibrator.capture_frame(frame)

        if result["added"]:
            self.detection_status_label.setText(
                f"Captured view with {result['corner_count']} corners."
            )
        else:
            self.detection_status_label.setText(
                f"Not enough corners detected ({result['corner_count']}) -- "
                f"not captured. Try a different angle/distance."
            )

        self.captures_label.setText(
            f"Captured views: {result['total_captures']} / {distortion.MIN_CALIBRATION_VIEWS} minimum"
        )
        self.calibrate_button.setEnabled(result["total_captures"] >= distortion.MIN_CALIBRATION_VIEWS)

    def clear_captures(self) -> None:
        if self.calibrator is None:
            return

        self.calibrator.clear()
        self.captures_label.setText(f"Captured views: 0 / {distortion.MIN_CALIBRATION_VIEWS} minimum")
        self.calibrate_button.setEnabled(False)
        self.detection_status_label.setText("Captures cleared.")

    def run_calibration(self) -> None:
        if self.calibrator is None:
            return

        profile_id = self._active_profile_id()

        if profile_id is None:
            self.result_label.setText("No active profile -- create one above first.")
            return

        try:
            result = self.calibrator.calibrate()
        except RuntimeError as error:
            self.result_label.setText(f"ERROR: {error}")
            return

        warning = result.get("warning")

        if warning:
            # Deliberately NOT saved -- a result flagged this far off
            # physically-plausible values (this is exactly what produced
            # the fx~5500/k3~-39896 calibration that made undistort worse
            # than doing nothing) is more likely to actively harm than help,
            # so it's shown but not written to disk. The previous good
            # calibration (if any) stays in place and in use.
            self.result_label.setText(
                f"NOT saved -- {warning} "
                f"(reprojection error was {result['reprojection_error']:.4f} px, "
                f"which looked fine -- that number alone doesn't catch this)."
            )
            return

        distortion.save_calibration(result, path=calibration_profiles.distortion_path(profile_id))

        self.result_label.setText(
            f"Calibration saved to this profile "
            f"({calibration_profiles.distortion_path(profile_id).name}). "
            f"Reprojection error: {result['reprojection_error']:.4f} px "
            f"(lower is better; well under 1.0 is good) -- "
            f"from {result['num_captures']} captured views at "
            f"{result['image_size'][0]}x{result['image_size'][1]}. "
            f"Toggle 'Show Undistorted' on the Live Camera tab to see it applied."
        )

    # ---- Stage 4: physical scale calibration ----

    def _update_scale_capture_availability(self) -> None:
        profile_id = self._active_profile_id()

        if profile_id is None:
            self.verify_scale_button.setEnabled(False)
            self.scale_capture_button.setEnabled(False)
            return  # profile_missing_label / the disabled scale_box already explain this

        saved_scale = scale_module.load_scale_calibration(path=calibration_profiles.scale_path(profile_id))
        self.verify_scale_button.setEnabled(saved_scale is not None and self.scale_frame is not None)

        if self.scale_frame is not None:
            return  # already frozen on a captured frame; capture-readiness below no longer applies

        calibration = distortion.load_calibration(path=calibration_profiles.distortion_path(profile_id))
        can_capture = calibration is not None and self.main_window.latest_frame is not None

        self.scale_capture_button.setEnabled(can_capture)

        if calibration is None:
            self.scale_status_label.setText("Complete lens-distortion calibration above first.")
        elif self.main_window.latest_frame is None:
            self.scale_status_label.setText("Connect the camera (Live Camera tab) first.")
        else:
            self.scale_status_label.setText("Ready -- point the camera, then Capture Frame for Scale.")

    def capture_scale_frame(self) -> None:
        profile_id = self._active_profile_id()

        if profile_id is None:
            self.scale_status_label.setText("No active profile -- create one above first.")
            return

        frame = self.main_window.latest_frame

        if frame is None:
            self.scale_status_label.setText("No live frame available yet.")
            return

        calibration = distortion.load_calibration(path=calibration_profiles.distortion_path(profile_id))

        if calibration is None:
            self.scale_status_label.setText("Complete lens-distortion calibration above first.")
            return

        # Same explicit-size-check pattern as LiveCameraTab's undistort
        # toggle: cv2.remap does not error on a map/frame size mismatch,
        # it silently samples wrong, so this has to be checked here too.
        frame_size = (frame.shape[1], frame.shape[0])
        calibration_size = tuple(calibration["image_size"])

        if frame_size != calibration_size:
            self.scale_status_label.setText(
                f"Live frame is {frame_size[0]}x{frame_size[1]} but the "
                f"saved distortion calibration was done at "
                f"{calibration_size[0]}x{calibration_size[1]} -- redo "
                f"Stage 3 at the current resolution first."
            )
            return

        maps = distortion.build_undistort_maps(calibration)
        undistorted = distortion.undistort_with_maps(frame, maps)

        # Same edge crop currently applied on Live Camera's "Show
        # Undistorted" preview (shared via main_window.crop_percentages,
        # updated live by LiveCameraTab's sliders) -- keeps the scale
        # reference consistent with what's actually being looked at, and
        # avoids the same least-trustworthy border pixels the crop
        # sliders exist to trim in the first place.
        crop = self.main_window.crop_percentages
        self.scale_frame = distortion.crop_edges(
            undistorted,
            top_pct=crop["top"],
            bottom_pct=crop["bottom"],
            left_pct=crop["left"],
            right_pct=crop["right"],
        )
        self.scale_points = []
        self.scale_mode = "calibrate"

        self._render_scale_frame()

        self.scale_capture_button.setEnabled(False)
        self.scale_retake_button.setEnabled(True)
        self.scale_compute_button.setEnabled(False)
        self.scale_points_label.setText("Click two points a known distance apart (0/2 selected).")
        self.scale_status_label.setText("Frame captured, undistorted, and cropped. Click two reference points below.")

    def retake_scale_frame(self) -> None:
        self.scale_frame = None
        self.scale_points = []
        self.scale_display_info = None
        self.scale_mode = "calibrate"

        self.scale_preview_label.setPixmap(QPixmap())
        self.scale_preview_label.setText("No frame captured yet.")
        self.scale_retake_button.setEnabled(False)
        self.scale_compute_button.setEnabled(False)
        self.verify_scale_button.setEnabled(False)
        self.scale_points_label.setText("Click two points a known distance apart.")
        self.scale_verify_label.setText("")

    def enter_verify_mode(self) -> None:
        if self.scale_frame is None:
            return

        self.scale_mode = "verify"
        self.scale_points = []
        self._render_scale_frame()

        self.scale_compute_button.setEnabled(False)
        self.scale_points_label.setText(
            "Verify mode: click two points of a distance you know separately "
            "(0/2 selected)."
        )
        self.scale_verify_label.setText("")

    def _render_scale_frame(self) -> None:
        if self.scale_frame is None:
            return

        display = self.scale_frame.copy()

        # Different marker color per mode so it's visually obvious which
        # one you're in: red crosses while calibrating (setting the
        # scale), orange while verifying (checking it) -- verify mode
        # never touches the saved file, and the color difference is a
        # cheap reminder of that.
        marker_color = (0, 0, 255) if self.scale_mode == "calibrate" else (0, 140, 255)

        for point in self.scale_points:
            center = (int(round(point[0])), int(round(point[1])))
            cv2.drawMarker(display, center, marker_color, markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)

        if len(self.scale_points) == 2:
            p1 = (int(round(self.scale_points[0][0])), int(round(self.scale_points[0][1])))
            p2 = (int(round(self.scale_points[1][0])), int(round(self.scale_points[1][1])))
            cv2.line(display, p1, p2, marker_color, 2)

        img_h, img_w = display.shape[:2]
        label_w = max(1, self.scale_preview_label.width())
        label_h = max(1, self.scale_preview_label.height())

        # Computed explicitly (not via QPixmap.scaled()'s own choice) so
        # the click-to-image inverse mapping in _on_scale_image_clicked
        # is guaranteed to match what's actually displayed.
        display_scale = min(label_w / img_w, label_h / img_h, 1.0)
        displayed_w = max(1, int(img_w * display_scale))
        displayed_h = max(1, int(img_h * display_scale))
        offset_x = (label_w - displayed_w) / 2
        offset_y = (label_h - displayed_h) / 2
        self.scale_display_info = (display_scale, offset_x, offset_y)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, channels = rgb.shape
        qimage = QImage(rgb.data, w, h, channels * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())
        scaled_pixmap = pixmap.scaled(displayed_w, displayed_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.scale_preview_label.setPixmap(scaled_pixmap)

    def _on_scale_image_clicked(self, click_x: int, click_y: int) -> None:
        if self.scale_frame is None or self.scale_display_info is None:
            return

        if len(self.scale_points) >= 2:
            self.scale_points = []  # third click starts a new pair

        display_scale, offset_x, offset_y = self.scale_display_info
        image_x = (click_x - offset_x) / display_scale
        image_y = (click_y - offset_y) / display_scale

        img_h, img_w = self.scale_frame.shape[:2]

        if not (0 <= image_x < img_w and 0 <= image_y < img_h):
            self.scale_points_label.setText(
                "Click landed outside the image (probably in the letterboxed "
                "margin) -- try again."
            )
            return

        self.scale_points.append((image_x, image_y))
        self._render_scale_frame()

        if self.scale_mode == "verify":
            if len(self.scale_points) < 2:
                self.scale_points_label.setText(
                    f"Verify mode: click two points of a distance you know "
                    f"separately ({len(self.scale_points)}/2 selected)."
                )
                return

            self.scale_points_label.setText(
                "Verify mode: click two points of a distance you know "
                "separately (2/2 selected)."
            )
            self._show_verify_result()
            return

        self.scale_points_label.setText(
            f"Click two points a known distance apart "
            f"({len(self.scale_points)}/2 selected)."
        )
        self.scale_compute_button.setEnabled(len(self.scale_points) == 2)

    def _show_verify_result(self) -> None:
        profile_id = self._active_profile_id()
        saved = None if profile_id is None else scale_module.load_scale_calibration(
            path=calibration_profiles.scale_path(profile_id)
        )

        if saved is None:
            self.scale_verify_label.setText("No saved scale to verify against.")
            return

        distance_px = scale_module.pixel_distance(self.scale_points[0], self.scale_points[1])
        measured_mm = scale_module.measure_distance_mm(
            self.scale_points[0], self.scale_points[1], saved["mm_per_pixel"]
        )

        self.scale_verify_label.setText(
            f"Measured distance: {measured_mm:.3f} mm ({distance_px:.1f}px, "
            f"using the saved {saved['mm_per_pixel']:.5f} mm/pixel scale). "
            f"Compare this to the real distance you know between those two "
            f"points -- if they're close, the scale checks out."
        )

    def compute_and_save_scale(self) -> None:
        profile_id = self._active_profile_id()

        if profile_id is None or self.scale_frame is None or len(self.scale_points) != 2:
            return

        img_h, img_w = self.scale_frame.shape[:2]

        try:
            result = scale_module.compute_scale(
                self.scale_points[0],
                self.scale_points[1],
                self.scale_distance_spin.value(),
                (img_w, img_h),
            )
        except ValueError as error:
            self.scale_result_label.setText(f"ERROR: {error}")
            return

        scale_module.save_scale_calibration(result, path=calibration_profiles.scale_path(profile_id))

        self.scale_result_label.setText(
            f"Scale saved to this profile ({calibration_profiles.scale_path(profile_id).name}): "
            f"{result['mm_per_pixel']:.5f} mm/pixel "
            f"({1 / result['mm_per_pixel']:.3f} px/mm) -- from a "
            f"{result['pixel_distance']:.1f}px reference measured as "
            f"{result['known_distance_mm']} mm."
        )

        # Otherwise this waits for the next 150ms timer tick to notice
        # the new file -- immediate is cheap and avoids a "why is Verify
        # still greyed out" moment right after saving.
        self.verify_scale_button.setEnabled(True)
