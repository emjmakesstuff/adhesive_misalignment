"""
Detection tab: continuous, live fragment detection that automatically
switches its whole detector -- controls, algorithm, and panels -- based
on the active calibration profile's detector_type (not on
main_window.active_source_type directly -- "usb" alone no longer implies
grayscale now that a color USB camera, the ELP, exists alongside the
monochrome U20CAM):

  - "grayscale": the existing background-subtracted detector
    (detection.py + background_reference.py), unchanged from before
    VDO.Ninja existed. Used by the U20CAM (source_type "usb").
  - "color": an HSV color detector (color_detection.py), which itself
    calls detection.detect_fragments() internally -- see that module for
    why no shape/connectivity code needed to be duplicated. Used by both
    VDO.Ninja profiles and the ELP (source_type "usb", but
    detector_type "color").

Both paths produce detection.Fragment lists, wrapped into the same
regions.DetectedRegion structure for reporting (see regions.py) --
downstream rendering code is written once, not duplicated per detector.

Nothing here assumes a detected region is circular; fragments are shown
exactly as cv2.findContours traces them. (An earlier version grouped
fragments into expected circles via assignment.py/measurement.py --
removed, since the Processing tab's per-pad ROI analysis
(processing_project.py/detection_pipeline.py) replaced that idea
entirely with user-drawn rectangular search regions instead of expected
circle counts/positions.)

Reads main_window.latest_frame on a timer -- the same shared reference
LiveCameraTab already writes to (from either a CameraStream or a
VdoNinjaSource) and CalibrationTab already reads -- never opens a second
camera connection. A plain repeating QTimer already gives "skip stale
frames, process the newest" for free: Qt does not queue up missed fires
for one timer, it just fires again once the event loop is free, and
every tick reads whatever main_window.latest_frame currently holds.
Processing is skipped entirely while the tab isn't the visible one.
"""

from __future__ import annotations

import time

import cv2
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import background_reference
import calibration_profiles
import circle_config as circle_config_module
import color_detection
import detection
import detection_pipeline
import distortion
import regions
import scale as scale_module

FRAGMENT_OUTLINE_COLOR = (0, 255, 0)  # BGR, lime -- one consistent color, no accept/extra/status coding
UPDATE_INTERVAL_MS = 400  # heavier per-tick work (diff/confidence + connected components + 6 panel renders) than other tabs' timers


class ClickableImageLabel(QLabel):
    """A QLabel that reports double-clicks -- used so any live panel can be
    expanded to fill the tab (double-click again, or Esc, to return)."""

    doubleClicked = Signal()

    def mouseDoubleClickEvent(self, event) -> None:
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)


def _make_panel(title: str, min_size: tuple[int, int] = (320, 220)) -> tuple[QWidget, ClickableImageLabel]:
    container = QWidget()
    v = QVBoxLayout(container)
    v.setContentsMargins(2, 2, 2, 2)

    title_label = QLabel(title)
    title_bold = QFont()
    title_bold.setBold(True)
    title_label.setFont(title_bold)
    v.addWidget(title_label)

    image_label = ClickableImageLabel("no data yet")
    image_label.setMinimumSize(*min_size)
    image_label.setAlignment(Qt.AlignCenter)
    image_label.setStyleSheet("background-color: black; color: white;")
    image_label.setToolTip("Double-click to expand full-size. Double-click again, or press Esc, to return.")
    v.addWidget(image_label)

    return container, image_label


class DetectionTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.last_fragments: list[detection.Fragment] = []
        self._last_seen_detector_type: str | None = None
        self._last_seen_profile_id: str | None = None
        self._last_vdo_process_time = 0.0

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(scroll_area)

        content = QWidget()
        scroll_area.setWidget(content)
        layout = QVBoxLayout(content)

        # ---- detector configuration (thresholds -- see threshold_stack below) ----

        config_box = QGroupBox("Detector configuration")
        config_form = QFormLayout(config_box)

        self.save_config_button = QPushButton("Save Config")

        layout.addWidget(config_box)

        # ---- detector-specific threshold controls (one page per source type) ----

        self.threshold_stack = QStackedWidget()

        usb_page = QWidget()
        usb_form = QFormLayout(usb_page)
        threshold_row = QHBoxLayout()
        self.possible_spin = QSpinBox()
        self.possible_spin.setRange(0, 255)
        self.probable_spin = QSpinBox()
        self.probable_spin.setRange(0, 255)
        self.strong_spin = QSpinBox()
        self.strong_spin.setRange(0, 255)
        for label_text, spin in (
            ("Possible:", self.possible_spin),
            ("Probable:", self.probable_spin),
            ("Strong:", self.strong_spin),
        ):
            threshold_row.addWidget(QLabel(label_text))
            threshold_row.addWidget(spin)
        threshold_row.addStretch(1)
        threshold_widget = QWidget()
        threshold_widget.setLayout(threshold_row)
        threshold_widget.setToolTip(
            "0-255 cut points applied to the BACKGROUND-SUBTRACTED "
            "DIFFERENCE image, not raw brightness -- a real new contact "
            "spot might only be 10-60 counts brighter than the background. "
            "Must be non-decreasing (possible <= probable <= strong)."
        )
        usb_form.addRow("Difference thresholds:", threshold_widget)
        self.threshold_stack.addWidget(usb_page)  # index 0

        vdo_page = QWidget()
        vdo_form = QFormLayout(vdo_page)

        self.color_preset_combo = QComboBox()
        self.color_preset_combo.addItems(sorted(color_detection.HUE_PRESETS.keys()))
        vdo_form.addRow("Color preset:", self.color_preset_combo)

        hue_row = QHBoxLayout()
        self.hue_override_check = QCheckBox("Override:")
        hue_row.addWidget(self.hue_override_check)
        hue_row.addWidget(QLabel("min"))
        self.hue_min_spin = QSpinBox()
        self.hue_min_spin.setRange(0, 179)
        hue_row.addWidget(self.hue_min_spin)
        hue_row.addWidget(QLabel("max"))
        self.hue_max_spin = QSpinBox()
        self.hue_max_spin.setRange(0, 179)
        hue_row.addWidget(self.hue_max_spin)
        hue_row.addStretch(1)
        hue_widget = QWidget()
        hue_widget.setLayout(hue_row)
        hue_widget.setToolTip("Replaces the color preset's own hue range (0-179) when checked.")
        vdo_form.addRow("Hue range:", hue_widget)

        weak_row = QHBoxLayout()
        weak_row.addWidget(QLabel("Sat min:"))
        self.weak_sat_spin = QSpinBox()
        self.weak_sat_spin.setRange(0, 255)
        weak_row.addWidget(self.weak_sat_spin)
        weak_row.addWidget(QLabel("Val min:"))
        self.weak_val_spin = QSpinBox()
        self.weak_val_spin.setRange(0, 255)
        weak_row.addWidget(self.weak_val_spin)
        weak_row.addStretch(1)
        weak_widget = QWidget()
        weak_widget.setLayout(weak_row)
        weak_widget.setToolTip(
            "The broad 'weak' floor -- pixels below either of these are 0 "
            "confidence regardless of hue. Lower this to catch paler edges "
            "of a real spot."
        )
        vdo_form.addRow("Weak (broad) floor:", weak_widget)

        core_row = QHBoxLayout()
        core_row.addWidget(QLabel("Sat min:"))
        self.core_sat_spin = QSpinBox()
        self.core_sat_spin.setRange(0, 255)
        core_row.addWidget(self.core_sat_spin)
        core_row.addWidget(QLabel("Val min:"))
        self.core_val_spin = QSpinBox()
        self.core_val_spin.setRange(0, 255)
        core_row.addWidget(self.core_val_spin)
        core_row.addStretch(1)
        core_widget = QWidget()
        core_widget.setLayout(core_row)
        core_widget.setToolTip(
            "The strict 'core' ceiling -- pixels at/above both saturate to "
            "full (255) confidence. A fragment is only kept if it contains "
            "at least one pixel at this level (hysteresis)."
        )
        vdo_form.addRow("Core (strict) ceiling:", core_widget)

        vdo_threshold_row = QHBoxLayout()
        self.vdo_possible_spin = QSpinBox()
        self.vdo_possible_spin.setRange(0, 255)
        self.vdo_probable_spin = QSpinBox()
        self.vdo_probable_spin.setRange(0, 255)
        self.vdo_strong_spin = QSpinBox()
        self.vdo_strong_spin.setRange(0, 255)
        for label_text, spin in (
            ("Possible:", self.vdo_possible_spin),
            ("Probable:", self.vdo_probable_spin),
            ("Strong:", self.vdo_strong_spin),
        ):
            vdo_threshold_row.addWidget(QLabel(label_text))
            vdo_threshold_row.addWidget(spin)
        vdo_threshold_row.addStretch(1)
        vdo_threshold_widget = QWidget()
        vdo_threshold_widget.setLayout(vdo_threshold_row)
        vdo_threshold_widget.setToolTip(
            "0-255 cut points on the color-confidence scale (see Weak/Core "
            "above). 'Strong' also sets the hysteresis cutoff: a fragment "
            "survives only if it reaches this tier somewhere."
        )
        vdo_form.addRow("Confidence thresholds:", vdo_threshold_widget)

        self.show_masks_check = QCheckBox("Show mask panels")
        vdo_form.addRow(self.show_masks_check)

        self.vdo_fps_spin = QDoubleSpinBox()
        self.vdo_fps_spin.setRange(0.5, 30.0)
        self.vdo_fps_spin.setDecimals(1)
        self.vdo_fps_spin.setSuffix(" fps")
        self.vdo_fps_spin.setToolTip("How often color detection actually runs -- independent of this tab's own panel refresh rate.")
        vdo_form.addRow("Processing rate:", self.vdo_fps_spin)

        self.threshold_stack.addWidget(vdo_page)  # index 1

        config_form.addRow(self.threshold_stack)
        config_form.addRow(self.save_config_button)

        # ---- background reference (USB only) ----

        self.background_box = QGroupBox("Background reference (USB only)")
        background_layout = QVBoxLayout(self.background_box)

        background_help = QLabel(
            "Capture with: camera in its fixed position, undistortion "
            "enabled (this always undistorts internally), normal LEDs on, "
            "exposure/gain locked (Settings tab), and no intended contact/"
            "light. Saved in the same undistorted + cropped coordinate "
            "system used for live detection below, so it lines up with "
            "every live frame with no further alignment step."
        )
        background_help.setWordWrap(True)
        background_layout.addWidget(background_help)

        self.capture_background_button = QPushButton("Capture Background Reference")
        self.capture_background_button.setEnabled(False)
        background_layout.addWidget(self.capture_background_button)

        self.background_status_label = QLabel("No background reference saved yet.")
        self.background_status_label.setWordWrap(True)
        background_layout.addWidget(self.background_status_label)

        layout.addWidget(self.background_box)

        # ---- live status ----

        self.status_label = QLabel("No active calibration profile -- create one on the Calibration tab first.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        roi_note = QLabel(
            "Processing runs only inside the current usable ROI -- the same "
            "undistort + edge-crop region controlled by Live Camera's "
            "\"Crop after undistort\" sliders."
        )
        roi_note.setWordWrap(True)
        layout.addWidget(roi_note)

        # ---- live panels ----

        panel_hint = QLabel("Double-click any panel below to expand it full-size; double-click again, or press Esc, to return to the grid.")
        panel_hint.setWordWrap(True)
        layout.addWidget(panel_hint)

        self._panel_titles = {
            "live": "Corrected live frame",
            "diff": "Difference from background / color confidence (auto-contrast for display only)",
            "possible": "Possible-threshold mask",
            "probable": "Probable-threshold mask",
            "strong": "Strong-threshold mask",
            "fragments": "Fragment contours (actual shape, no circles)",
        }

        panel_grid = QGridLayout()
        self._panel_labels: dict[str, ClickableImageLabel] = {}
        grid_positions = {
            "live": (0, 0), "diff": (0, 1),
            "possible": (1, 0), "probable": (1, 1),
            "strong": (2, 0), "fragments": (2, 1),
        }
        for key, title in self._panel_titles.items():
            panel_widget, image_label = _make_panel(title)
            image_label.doubleClicked.connect(lambda key=key: self._toggle_expand(key))
            self._panel_labels[key] = image_label
            row, col = grid_positions[key]
            panel_grid.addWidget(panel_widget, row, col)

        self.panel_grid_widget = QWidget()
        self.panel_grid_widget.setLayout(panel_grid)
        layout.addWidget(self.panel_grid_widget)

        # Expanded (fullscreen-in-tab) view of whichever panel was last
        # double-clicked -- hidden until then, shown in place of the grid
        # above (not a separate window, per how this was asked for).
        self.expanded_panel_key: str | None = None
        self.expanded_container = QWidget()
        expanded_layout = QVBoxLayout(self.expanded_container)
        expanded_layout.setContentsMargins(2, 2, 2, 2)

        self.expanded_title_label = QLabel("")
        expanded_title_font = QFont()
        expanded_title_font.setBold(True)
        expanded_title_font.setPointSize(expanded_title_font.pointSize() + 2)
        self.expanded_title_label.setFont(expanded_title_font)
        expanded_layout.addWidget(self.expanded_title_label)

        expanded_hint = QLabel("Double-click, or press Esc, to return to the grid view.")
        expanded_layout.addWidget(expanded_hint)

        self.expanded_image_label = ClickableImageLabel("no data yet")
        self.expanded_image_label.setMinimumSize(640, 480)
        self.expanded_image_label.setAlignment(Qt.AlignCenter)
        self.expanded_image_label.setStyleSheet("background-color: black; color: white;")
        self.expanded_image_label.doubleClicked.connect(self._collapse_expand)
        expanded_layout.addWidget(self.expanded_image_label, 1)

        self.expanded_container.setVisible(False)
        layout.addWidget(self.expanded_container, 1)

        self.setFocusPolicy(Qt.StrongFocus)  # so Escape reaches keyPressEvent below while expanded

        # ---- per-fragment report ----

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setFont(QFont("Consolas", 9))
        self.report_text.setMinimumHeight(200)
        layout.addWidget(self.report_text)

        layout.addStretch(1)

        self.save_config_button.clicked.connect(self._save_config)
        self.capture_background_button.clicked.connect(self._capture_background)

        self._last_seen_detector_type = self._active_detector_type()
        self._last_seen_profile_id = self._active_profile_id()
        self._load_config()
        self._update_gating()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(UPDATE_INTERVAL_MS)

    # ---- active profile ----

    def _active_profile_id(self) -> str | None:
        profile = calibration_profiles.get_active_profile(self.main_window.active_profile_key)
        return None if profile is None else profile["id"]

    def _active_detector_type(self) -> str:
        """
        "grayscale" or "color", from the active profile -- NOT derived
        from main_window.active_source_type, since "usb" alone is
        ambiguous once both the monochrome U20CAM and the color ELP are
        usb-sourced. Defaults to "grayscale" (today's only pre-existing
        behavior) when no profile is active yet.
        """
        profile = calibration_profiles.get_active_profile(self.main_window.active_profile_key)
        return profile["detector_type"] if profile is not None else "grayscale"

    # ---- configuration ----

    def _load_config(self) -> None:
        profile_id = self._active_profile_id()
        path = (
            calibration_profiles.circle_config_path(profile_id)
            if profile_id is not None
            else circle_config_module.DEFAULT_CIRCLE_CONFIG_PATH
        )
        config = circle_config_module.load_circle_config(path=path)

        self.possible_spin.setValue(config["possible_threshold"])
        self.probable_spin.setValue(config["probable_threshold"])
        self.strong_spin.setValue(config["strong_threshold"])

        self.color_preset_combo.setCurrentText(config["color_preset"])
        has_override = config["hue_min"] is not None and config["hue_max"] is not None
        self.hue_override_check.setChecked(has_override)
        self.hue_min_spin.setValue(config["hue_min"] if has_override else 0)
        self.hue_max_spin.setValue(config["hue_max"] if has_override else 179)
        self.weak_sat_spin.setValue(config["weak_sat_min"])
        self.weak_val_spin.setValue(config["weak_val_min"])
        self.core_sat_spin.setValue(config["core_sat_min"])
        self.core_val_spin.setValue(config["core_val_min"])
        self.vdo_possible_spin.setValue(config["vdo_possible_threshold"])
        self.vdo_probable_spin.setValue(config["vdo_probable_threshold"])
        self.vdo_strong_spin.setValue(config["vdo_strong_threshold"])
        self.show_masks_check.setChecked(config["show_masks"])
        self.vdo_fps_spin.setValue(config["vdo_processing_fps"])

    def _current_config(self) -> dict:
        return {
            "possible_threshold": self.possible_spin.value(),
            "probable_threshold": self.probable_spin.value(),
            "strong_threshold": self.strong_spin.value(),
            "color_preset": self.color_preset_combo.currentText(),
            "hue_min": self.hue_min_spin.value() if self.hue_override_check.isChecked() else None,
            "hue_max": self.hue_max_spin.value() if self.hue_override_check.isChecked() else None,
            "weak_sat_min": self.weak_sat_spin.value(),
            "weak_val_min": self.weak_val_spin.value(),
            "core_sat_min": self.core_sat_spin.value(),
            "core_val_min": self.core_val_spin.value(),
            "vdo_possible_threshold": self.vdo_possible_spin.value(),
            "vdo_probable_threshold": self.vdo_probable_spin.value(),
            "vdo_strong_threshold": self.vdo_strong_spin.value(),
            "show_masks": self.show_masks_check.isChecked(),
            "vdo_processing_fps": self.vdo_fps_spin.value(),
        }

    def _save_config(self) -> None:
        profile_id = self._active_profile_id()

        if profile_id is None:
            self.status_label.setText("No active profile -- create one on the Calibration tab first.")
            return

        circle_config_module.save_circle_config(self._current_config(), path=calibration_profiles.circle_config_path(profile_id))
        self.status_label.setText("Config saved.")

    # ---- panel expand/collapse ----

    def _toggle_expand(self, key: str) -> None:
        if self.expanded_panel_key == key:
            self._collapse_expand()
            return

        self.expanded_panel_key = key
        self.expanded_title_label.setText(self._panel_titles[key])
        self.panel_grid_widget.setVisible(False)
        self.expanded_container.setVisible(True)
        self.setFocus()  # so a stray Escape press (without clicking the image first) still works

    def _collapse_expand(self) -> None:
        self.expanded_panel_key = None
        self.expanded_container.setVisible(False)
        self.panel_grid_widget.setVisible(True)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape and self.expanded_panel_key is not None:
            self._collapse_expand()
            return
        super().keyPressEvent(event)

    # ---- shared corrected-frame helper ----

    def _get_corrected_frame(self, profile_id: str):
        """
        Returns (gray_corrected, bgr_corrected, None) for the current
        main_window.latest_frame using the given profile's lens
        calibration, or (None, None, reason) if unavailable -- same
        undistort + crop_percentages pipeline used everywhere else in
        this app (Live Camera's preview, Calibration's scale capture).
        """
        frame = self.main_window.latest_frame

        if frame is None:
            return None, None, "Connect the camera (Live Camera tab) first."

        calibration = distortion.load_calibration(path=calibration_profiles.distortion_path(profile_id))

        if calibration is None:
            return None, None, "Complete lens-distortion calibration for this profile on the Calibration tab first."

        frame_size = (frame.shape[1], frame.shape[0])
        calibration_size = tuple(calibration["image_size"])

        if frame_size != calibration_size:
            return None, None, (
                f"Live frame is {frame_size[0]}x{frame_size[1]} but this "
                f"profile's saved distortion calibration was done at "
                f"{calibration_size[0]}x{calibration_size[1]} -- redo "
                f"Stage 3 at the current resolution first."
            )

        maps = distortion.build_undistort_maps(calibration)
        undistorted = distortion.undistort_with_maps(frame, maps)

        crop = self.main_window.crop_percentages
        corrected_bgr = distortion.crop_edges(
            undistorted,
            top_pct=crop["top"],
            bottom_pct=crop["bottom"],
            left_pct=crop["left"],
            right_pct=crop["right"],
        )
        corrected_gray = detection.to_grayscale(corrected_bgr)

        return corrected_gray, corrected_bgr, None

    # ---- gating / source switching ----

    def _update_gating(self) -> None:
        detector_type = self._active_detector_type()
        profile_id = self._active_profile_id()

        # React to the active detector/profile changing (from
        # LiveCameraTab's combo(s) or CalibrationTab's profile picker) --
        # polled here the same lightweight way CalibrationTab already
        # polls this, rather than needing a cross-tab signal for it.
        if detector_type != self._last_seen_detector_type or profile_id != self._last_seen_profile_id:
            self._last_seen_detector_type = detector_type
            self._last_seen_profile_id = profile_id
            self._load_config()
            self.last_fragments = []
            self.report_text.clear()

        self.threshold_stack.setCurrentIndex(0 if detector_type == "grayscale" else 1)
        self.background_box.setVisible(detector_type == "grayscale")

        if profile_id is None:
            self.status_label.setText("No active calibration profile -- create one on the Calibration tab first.")
            self.capture_background_button.setEnabled(False)
            return

        _, _, reason = self._get_corrected_frame(profile_id)

        if detector_type == "grayscale":
            calibration_ready = distortion.load_calibration(path=calibration_profiles.distortion_path(profile_id)) is not None
            self.capture_background_button.setEnabled(calibration_ready and self.main_window.latest_frame is not None)

            saved_background = background_reference.load_background_reference(
                path=calibration_profiles.background_reference_path(profile_id)
            )
            if saved_background is not None:
                self.background_status_label.setText(
                    f"Background reference saved ({saved_background.shape[1]}x{saved_background.shape[0]})."
                )
            else:
                self.background_status_label.setText("No background reference saved yet.")

            if reason is not None:
                self.status_label.setText(reason)
            elif saved_background is None:
                self.status_label.setText("Capture a background reference above to start live detection.")
            else:
                self.status_label.setText("Live.")
        else:
            self.capture_background_button.setEnabled(False)
            self.status_label.setText(reason if reason is not None else "Live.")

    def _capture_background(self) -> None:
        profile_id = self._active_profile_id()

        if profile_id is None:
            return

        gray, _, reason = self._get_corrected_frame(profile_id)

        if reason is not None:
            self.status_label.setText(reason)
            return

        background_reference.save_background_reference(gray, path=calibration_profiles.background_reference_path(profile_id))
        self._update_gating()

    # ---- live loop ----

    def _tick(self) -> None:
        if not self.isVisible():
            return  # idle while this isn't the active tab -- no point spending CPU on it

        self._update_gating()

        profile_id = self._active_profile_id()
        if profile_id is None:
            return

        source_type = self.main_window.active_source_type  # transport label only, for the report -- see _render_report
        detector_type = self._active_detector_type()
        gray, bgr, reason = self._get_corrected_frame(profile_id)
        if reason is not None:
            return  # status_label already carries the reason via _update_gating()

        config = self._current_config()

        if detector_type == "grayscale":
            background = background_reference.load_background_reference(
                path=calibration_profiles.background_reference_path(profile_id)
            )
            if background is None:
                return

            if background.shape != gray.shape:
                self.status_label.setText(
                    f"Background reference is {background.shape[1]}x{background.shape[0]} but "
                    f"the current corrected frame is {gray.shape[1]}x{gray.shape[0]} -- crop "
                    f"settings or resolution changed since capture. Recapture the background."
                )
                return

            try:
                fragments, display_map = detection_pipeline.run_detection(
                    bgr, detector_type, config, background=background
                )
            except ValueError as error:
                self.status_label.setText(f"ERROR: {error}")
                return
        else:
            # Processing-rate throttle: color detection can be heavier
            # than the USB diff path, and unlike USB this is independent
            # of the panel refresh interval above -- skip the actual
            # detection work (leaving the last result on screen) until
            # the configured interval has elapsed.
            now = time.perf_counter()
            min_interval = 1.0 / config["vdo_processing_fps"] if config["vdo_processing_fps"] > 0 else 0.0
            if now - self._last_vdo_process_time < min_interval:
                return
            self._last_vdo_process_time = now

            try:
                fragments, display_map = detection_pipeline.run_detection(bgr, detector_type, config)
            except ValueError as error:
                self.status_label.setText(f"ERROR: {error}")
                return

        self.last_fragments = fragments

        self._render_panels(bgr, display_map, config, detector_type)
        self._render_report(fragments, source_type, detector_type, profile_id)

        self.status_label.setText(f"Live -- {len(fragments)} fragment(s) detected this frame.")

    def _render_panels(self, corrected_bgr, display_map, config: dict, detector_type: str) -> None:
        self._update_panel("live", corrected_bgr)

        display_stretched = cv2.normalize(display_map, None, 0, 255, cv2.NORM_MINMAX)
        self._update_panel("diff", cv2.cvtColor(display_stretched, cv2.COLOR_GRAY2BGR))

        show_masks = True if detector_type == "grayscale" else config["show_masks"]

        if show_masks:
            if detector_type == "grayscale":
                thresholds = (config["possible_threshold"], config["probable_threshold"], config["strong_threshold"])
            else:
                thresholds = (config["vdo_possible_threshold"], config["vdo_probable_threshold"], config["vdo_strong_threshold"])

            for key, threshold in zip(("possible", "probable", "strong"), thresholds):
                mask = ((display_map >= threshold).astype("uint8")) * 255
                self._update_panel(key, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        else:
            for key in ("possible", "probable", "strong"):
                self._panel_labels[key].setPixmap(QPixmap())
                self._panel_labels[key].setText("Masks hidden ('Show mask panels' unchecked)")

        fragments_display = corrected_bgr.copy()
        for fragment in self.last_fragments:
            cv2.drawContours(fragments_display, [fragment.contour], -1, FRAGMENT_OUTLINE_COLOR, 1)
        self._update_panel("fragments", fragments_display)

    def _update_panel(self, key: str, bgr_image) -> None:
        # Always keep the small grid panel current (cheap at this update
        # rate) so the grid is never stale if you collapse back to it --
        # and additionally mirror into the expanded view if this is the
        # panel currently expanded.
        self._show_image(self._panel_labels[key], bgr_image)
        if self.expanded_panel_key == key:
            self._show_image(self.expanded_image_label, bgr_image)

    @staticmethod
    def _show_image(label: QLabel, bgr_image) -> None:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h, w, channels = rgb.shape
        qimage = QImage(rgb.data, w, h, channels * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())
        scaled = pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)

    def _render_report(
        self, fragments: list[detection.Fragment], source_type: str, detector_type: str, profile_id: str
    ) -> None:
        saved_scale = scale_module.load_scale_calibration(path=calibration_profiles.scale_path(profile_id))
        mm_per_pixel = saved_scale["mm_per_pixel"] if saved_scale is not None else None

        # regions.DetectedRegion.source_type keeps its existing "usb" |
        # "vdo_ninja" contract unchanged (a true statement about
        # transport regardless of camera role) -- detector_type is only
        # used for the human-readable header below, where it's the more
        # useful distinction now that two differently-detected cameras
        # can both be "usb".
        detected_regions = [regions.from_fragment(f, source_type, mm_per_pixel) for f in fragments]
        detected_regions.sort(key=lambda r: r.fragment.possible_area_px, reverse=True)

        lines = [f"Fragments detected: {len(detected_regions)}   Detector: {detector_type} (source: {source_type})"]
        if mm_per_pixel is None:
            lines.append("(No saved physical scale for this profile -- areas shown in pixels only.)")
        lines.append(
            "Grouping/circle assignment is temporarily disabled while arbitrary-shape "
            "detection is being verified -- every fragment below is status=uncertain."
        )
        lines.append("")

        for region in detected_regions:
            f = region.fragment

            if mm_per_pixel is not None:
                area_str = (
                    f"possible={f.possible_area_px}px ({region.possible_area_mm2:.4f} mm2)  "
                    f"probable={f.probable_area_px}px ({region.probable_area_mm2:.4f} mm2)  "
                    f"strong={f.strong_area_px}px ({region.strong_area_mm2:.4f} mm2)"
                )
            else:
                area_str = (
                    f"possible={f.possible_area_px}px  "
                    f"probable={f.probable_area_px}px  "
                    f"strong={f.strong_area_px}px"
                )

            lines.append(f"[UNCERTAIN] Region {f.id} ({region.source_type}) -- centroid=({f.centroid[0]:.1f}, {f.centroid[1]:.1f})")
            lines.append(f"    {area_str}")
            lines.append(
                f"    intensity: mean={f.mean_brightness:.1f}  "
                f"median={f.median_brightness:.1f}  "
                f"p95={f.p95_brightness:.1f}  "
                f"max={f.max_brightness:.1f}"
            )
            lines.append("")

        self.report_text.setPlainText("\n".join(lines))
