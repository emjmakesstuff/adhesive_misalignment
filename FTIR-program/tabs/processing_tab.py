"""
Processing tab: basic playback of a recording -- raw and corrected
(undistort + crop) views side by side, a timeline, and frame
navigation. Always loads calibration from the RECORDING'S OWN
calibration_snapshot/ (see recording_store.py's module docstring),
never by re-resolving the live profile -- a later edit to (or deletion
of) the live profile can never silently change what an old recording
shows here, and a recording stays self-describing even if its source
profile is later deleted.

Opened via RecordingsTab's "Open in Processing" button (main_window
switches to this tab and calls open_recording()) -- this tab has no
recording list of its own, Stage C's Recordings tab already owns that.

Detection (Stage E) reuses detection_pipeline.run_detection() -- the
exact same detector code DetectionTab's live loop calls -- but resolves
detector_type/config/background from the recording's OWN
calibration_snapshot/ (circle_config.json, background_reference.npy),
never the live active profile, for the same self-containment reason
undistortion already does. Detection here is on-demand (a button press),
not continuous like the live tab: Current Frame, a Range, or the Full
Recording. Results are shown/reported only -- no mask persistence yet,
that's Stage F.
"""

from __future__ import annotations

import traceback

import cv2
import numpy as np
from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRubberBand,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import background_reference
import circle_config as circle_config_module
import color_detection
import detection_pipeline
import distortion
import processing_project as pp
import recording_store as rs
import regions

ROI_COLOR = (0, 165, 255)  # BGR, orange -- visually distinct from FRAGMENT_OUTLINE_COLOR's lime
ROI_OVERLAP_COLOR = (0, 0, 255)  # BGR, red -- an ROI involved in an overlap
ROI_COLUMNS = ["Name", "Rectangle", "Frame Range", "Expected Area (mm2)", "Min Area (noise filter)", "Overlap?"]

DEFAULT_PLAYBACK_FPS = 30.0
ZOOM_MIN_PCT, ZOOM_MAX_PCT, ZOOM_DEFAULT_PCT = 50, 400, 100
SPEED_MIN_PCT, SPEED_MAX_PCT, SPEED_DEFAULT_PCT = 10, 200, 100
FRAGMENT_OUTLINE_COLOR = (0, 255, 0)  # BGR, lime -- matches DetectionTab's live overlay color
DETECTION_PROGRESS_UPDATE_EVERY = 10  # frames between UI refresh/processEvents() during a range/full run


_OUTLINE_OFFSETS = [(-2, -2), (-2, 0), (-2, 2), (0, -2), (0, 2), (2, -2), (2, 0), (2, 2)]


def _draw_outlined_text(frame, text: str, pos: tuple[int, int], font_scale: float, thickness: int) -> None:
    """White text with a thick black outline behind it -- a single flat
    color (however bright) can still wash out against a background of a
    similar hue/brightness (e.g. yellow text on a pale/bright patch).
    Black-outline-plus-white-fill is the standard "readable over
    anything" overlay technique (subtitles use it for the same reason):
    for ANY background color, at least one of pure black or pure white
    has strong contrast against it, and the outline guarantees that one
    always borders the fill.

    NOT a single thicker black putText() call behind a thinner white
    one: cv2.putText's `thickness` does not reliably grow text outward
    from a fixed centerline the way it does for simple shapes -- verified
    directly (a thickness=5 black pass followed by a thickness=2 white
    pass at the same position left ZERO dark pixels behind; the "thinner"
    white pass fully covered the "thicker" black one instead of leaving a
    margin). Instead: draw the SAME thickness in black 8 times, offset a
    couple pixels in every direction, then the real text once in white on
    top -- the union of the 8 shifted copies is guaranteed to surround
    the centered white glyphs regardless of how any single putText call
    sizes its strokes."""
    x, y = pos
    for dx, dy in _OUTLINE_OFFSETS:
        cv2.putText(frame, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

FRAGMENT_OVERLAY_OFF = "Off"
FRAGMENT_OVERLAY_FULL = "Full-frame detection"
FRAGMENT_OVERLAY_ROI = "Per-ROI detection (independent)"


class _RoiLabel(QLabel):
    """A QLabel that reports double-clicks (same small pattern
    DetectionTab's ClickableImageLabel uses, for expand/collapse) and,
    when armed via `drawing_enabled`, supports click-drag rectangle
    drawing for ROI definition -- visual feedback via QRubberBand,
    emitting the drawn rectangle in LABEL-local pixel coordinates
    (converted to corrected-frame coordinates by the tab, which knows
    the current display scale -- see _label_rect_to_corrected_rect).
    Only ever armed on the Corrected panel; the Raw panel uses this same
    class but never sets drawing_enabled."""

    doubleClicked = Signal()
    roiDrawn = Signal(QRect)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drawing_enabled = False
        self._rubber_band: QRubberBand | None = None
        self._origin = None

    def mouseDoubleClickEvent(self, event) -> None:
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event) -> None:
        if self.drawing_enabled and event.button() == Qt.LeftButton:
            self._origin = event.pos()
            if self._rubber_band is None:
                self._rubber_band = QRubberBand(QRubberBand.Rectangle, self)
            self._rubber_band.setGeometry(QRect(self._origin, self._origin))
            self._rubber_band.show()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.drawing_enabled and self._origin is not None and self._rubber_band is not None:
            self._rubber_band.setGeometry(QRect(self._origin, event.pos()).normalized())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self.drawing_enabled and self._origin is not None:
            rect = QRect(self._origin, event.pos()).normalized()
            if self._rubber_band is not None:
                self._rubber_band.hide()
            self._origin = None
            if rect.width() >= 4 and rect.height() >= 4:  # ignore accidental clicks/tiny drags
                self.roiDrawn.emit(rect)
            return
        super().mouseReleaseEvent(event)


class _AutoRerenderScrollArea(QScrollArea):
    """A QScrollArea that emits `resized` whenever its VIEWPORT actually
    changes size. The corrected panel's displayed pixmap size (and
    _corrected_display_scale, which ROI mouse-coordinate mapping depends
    on) is only ever recomputed inside _show_image(), which only runs
    when a frame is explicitly (re)rendered -- a viewport resize that
    happens for any OTHER reason (the outer page scroll area settling
    into its final layout right after a recording is opened, the user
    resizing the window, ...) would otherwise leave the displayed pixmap
    at a stale size with no re-render ever triggered, silently
    desynchronizing "where the rubber band is drawn" from "what
    _corrected_display_scale thinks the scale is" -- exactly the "ROI
    box doesn't go where I draw it" bug this fixes. Connected to
    _rerender_current() in __init__ below."""

    resized = Signal()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.resized.emit()


class _ResettableSlider(QSlider):
    """A QSlider that snaps back to a given default value on double-click
    -- used below for Zoom/Speed so there's a quick way back to
    100%/1.0x without dragging precisely."""

    def __init__(self, orientation, default_value: int, parent=None):
        super().__init__(orientation, parent)
        self._default_value = default_value

    def mouseDoubleClickEvent(self, event) -> None:
        # Deliberately NOT forwarding to super() -- QSlider's own handler
        # would then treat the double-click as a normal click-to-position,
        # moving the handle to wherever the click landed and undoing the
        # reset just set above.
        self.setValue(self._default_value)
        event.accept()


class _NewRoiDialog(QDialog):
    """Shown right after a rectangle is drawn -- name (required) and the
    pad's real expected physical contact area (optional; 0 means "not
    set", NOT a real zero-area pad). expected_area_mm2 is what
    coverage_percent gets computed against later, never the ROI
    rectangle's own (deliberately larger, search-boundary-only) area."""

    def __init__(self, default_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New ROI")
        self.setModal(True)

        form = QFormLayout(self)
        self.name_edit = QLineEdit(default_name)
        form.addRow("Name:", self.name_edit)

        self.expected_area_spin = QDoubleSpinBox()
        self.expected_area_spin.setRange(0.0, 100000.0)
        self.expected_area_spin.setDecimals(4)
        self.expected_area_spin.setSuffix(" mm2 (0 = not set)")
        self.expected_area_spin.setToolTip(
            "The pad's real physical contact area -- NOT the rectangle you just drew (which should be "
            "somewhat larger, as a search boundary). Coverage % is computed against this value; leave at "
            "0 if unknown for now (coverage will show as N/A until it's set)."
        )
        form.addRow("Expected pad area:", self.expected_area_spin)

        self.min_area_spin = QDoubleSpinBox()
        self.min_area_spin.setRange(0.0, 100000.0)
        self.min_area_spin.setDecimals(4)
        self.min_area_spin.setSuffix(" mm2/px (0 = no filter)")
        self.min_area_spin.setToolTip(
            "Noise floor for THIS pad only -- any detected blob smaller than this is dropped entirely, "
            "before it counts toward area or coverage. mm2 if this recording has a physical scale, "
            "otherwise the same number is read as raw pixels. Leave at 0 to disable."
        )
        form.addRow("Min. area (noise filter):", self.min_area_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def result_values(self) -> tuple[str, float | None, float | None]:
        name = self.name_edit.text().strip() or "Pad"
        expected_area = self.expected_area_spin.value()
        min_area = self.min_area_spin.value()
        return name, (expected_area if expected_area > 0 else None), (min_area if min_area > 0 else None)


def _format_time_ns(recording_time_ns: int | None) -> str:
    if recording_time_ns is None:
        return "N/A"
    total_seconds = recording_time_ns / 1e9
    minutes, seconds = divmod(total_seconds, 60)
    return f"{int(minutes)}m {seconds:05.2f}s"


class ProcessingTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        self.recording_id: str | None = None
        self._metadata: dict | None = None
        self._cap: cv2.VideoCapture | None = None
        self._next_read_frame_number = 0
        self._total_frames = 0
        self._frame_index_rows: list[dict] = []
        self._undistort_maps = None
        self._undistort_target_size: tuple[int, int] | None = None
        self._crop: dict = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        self._last_raw_frame = None  # cached so a zoom change can re-render without re-reading the video
        self._zoom = ZOOM_DEFAULT_PCT / 100.0
        self._playback_speed = SPEED_DEFAULT_PCT / 100.0

        # ---- detection (Stage E) -- all resolved from the recording's
        #      own calibration_snapshot/, see _load_detection_inputs ----
        self._detector_type: str | None = None
        self._detection_config: dict | None = None
        self._detection_background = None
        self._detection_results: dict[int, list] = {}  # frame_number -> fragments, for the overlay + scrubbing

        # ---- ROI (per-pad) analysis -- see processing_project.py ----
        self.project_id: str | None = None
        self._rois: list[dict] = []
        self._roi_overlaps: list[tuple[str, str]] = []
        self._corrected_display_scale = 1.0  # set by _show_image() for the corrected panel -- see _label_rect_to_corrected_rect
        self._corrected_pixmap_offset = (0, 0)  # top-left of the pixmap WITHIN corrected_label -- see _show_image
        self._roi_current_frame_results: dict[str, dict] = {}  # roi_id -> summary, in-memory only (current-frame preview)
        self._roi_frame_fragments: dict[int, dict[str, list]] = {}  # frame_number -> {roi_id: Fragment list (full-frame coords)} -- see _render()

        self.expanded_panel: str | None = None  # "raw" | "corrected" | None -- see _toggle_expand

        # This tab has grown to have a lot of vertical content (two video
        # panels, playback controls, the Detection group, the ROI group)
        # -- wrapped in a scroll area, same pattern DetectionTab already
        # uses, so it's usable at any window size rather than getting
        # clipped. The video panels' OWN QScrollAreas (zoom/pan, see
        # raw_scroll/corrected_scroll below) nest inside this outer one
        # without conflict -- each manages its own viewport independently.
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(scroll_area)

        content = QWidget()
        scroll_area.setWidget(content)
        layout = QVBoxLayout(content)

        self.info_label = QLabel("No recording open -- use the Recordings tab's \"Open in Processing\" button.")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.calibration_status_label = QLabel("")
        self.calibration_status_label.setWordWrap(True)
        layout.addWidget(self.calibration_status_label)

        self.expand_hint = QLabel("Double-click either panel to expand it full-size; double-click again, or press Esc, to return.")
        self.expand_hint.setWordWrap(True)
        layout.addWidget(self.expand_hint)

        panels_row = QHBoxLayout()

        self.raw_group = QGroupBox("Raw")
        raw_layout = QVBoxLayout(self.raw_group)
        self.raw_label = _RoiLabel("No frame loaded.")
        self.raw_label.setMinimumSize(640, 480)
        self.raw_label.setAlignment(Qt.AlignCenter)
        self.raw_label.setStyleSheet("background-color: black; color: white;")
        self.raw_label.setToolTip("Double-click to expand full-size. Double-click again, or press Esc, to return.")
        self.raw_scroll = _AutoRerenderScrollArea()
        self.raw_scroll.setWidgetResizable(False)
        self.raw_scroll.setWidget(self.raw_label)
        raw_layout.addWidget(self.raw_scroll)
        panels_row.addWidget(self.raw_group)

        self.corrected_group = QGroupBox("Corrected (undistorted + cropped)")
        corrected_layout = QVBoxLayout(self.corrected_group)
        self.corrected_label = _RoiLabel("No frame loaded.")
        self.corrected_label.setMinimumSize(640, 480)
        self.corrected_label.setAlignment(Qt.AlignCenter)
        self.corrected_label.setStyleSheet("background-color: black; color: white;")
        self.corrected_label.setToolTip("Double-click to expand full-size. Double-click again, or press Esc, to return.")
        self.corrected_scroll = _AutoRerenderScrollArea()
        self.corrected_scroll.setWidgetResizable(False)
        self.corrected_scroll.setWidget(self.corrected_label)
        corrected_layout.addWidget(self.corrected_scroll)
        panels_row.addWidget(self.corrected_group)

        layout.addLayout(panels_row, 1)

        # ---- zoom / speed (apply to both panels/playback at once -- see
        #      _show_image and _toggle_play) ----

        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("Zoom:"))
        self.zoom_slider = _ResettableSlider(Qt.Horizontal, ZOOM_DEFAULT_PCT)
        self.zoom_slider.setRange(ZOOM_MIN_PCT, ZOOM_MAX_PCT)
        self.zoom_slider.setValue(ZOOM_DEFAULT_PCT)
        self.zoom_slider.setMaximumWidth(160)
        self.zoom_slider.setToolTip(
            "100% fits the panel exactly, same as before this control existed. Above 100%, the "
            "panel scrolls -- drag inside the image or use its scrollbars to pan around. "
            "Double-click to reset to 100%."
        )
        view_row.addWidget(self.zoom_slider)
        self.zoom_value_label = QLabel(f"{ZOOM_DEFAULT_PCT}%")
        self.zoom_value_label.setMinimumWidth(40)
        view_row.addWidget(self.zoom_value_label)

        view_row.addSpacing(20)
        view_row.addWidget(QLabel("Speed:"))
        self.speed_slider = _ResettableSlider(Qt.Horizontal, SPEED_DEFAULT_PCT)
        self.speed_slider.setRange(SPEED_MIN_PCT, SPEED_MAX_PCT)
        self.speed_slider.setValue(SPEED_DEFAULT_PCT)
        self.speed_slider.setMaximumWidth(160)
        self.speed_slider.setToolTip(
            "Playback speed relative to this recording's own real measured rate. "
            "Double-click to reset to 1.0x."
        )
        view_row.addWidget(self.speed_slider)
        self.speed_value_label = QLabel(f"{SPEED_DEFAULT_PCT / 100:.1f}x")
        self.speed_value_label.setMinimumWidth(40)
        view_row.addWidget(self.speed_value_label)

        view_row.addStretch(1)
        layout.addLayout(view_row)

        controls_row = QHBoxLayout()
        self.first_button = QPushButton("|<")
        self.prev_button = QPushButton("<")
        self.play_pause_button = QPushButton("Play")
        self.next_button = QPushButton(">")
        self.last_button = QPushButton(">|")
        for button in (self.first_button, self.prev_button, self.play_pause_button, self.next_button, self.last_button):
            controls_row.addWidget(button)

        self.frame_label = QLabel("Frame: -- / --")
        controls_row.addWidget(self.frame_label)
        self.time_label = QLabel("Time: --")
        controls_row.addWidget(self.time_label)
        controls_row.addStretch(1)
        layout.addLayout(controls_row)

        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setEnabled(False)
        layout.addWidget(self.timeline_slider)

        # ---- detection (Stage E) ----

        self.detection_group = QGroupBox("Detection (reuses the same detector as the Detection tab)")
        detection_layout = QVBoxLayout(self.detection_group)

        self.detection_status_label = QLabel("")
        self.detection_status_label.setWordWrap(True)
        detection_layout.addWidget(self.detection_status_label)

        # Thresholds/color controls -- same fields DetectionTab's live
        # loop exposes, initially populated from this recording's own
        # circle_config.json snapshot (see _populate_config_controls),
        # but editable here: changing them updates self._detection_config
        # in place for the NEXT Detect run, they're never written back
        # into the recording's snapshot (that stays immutable). Only one
        # page is shown -- a recording's detector_type is fixed, unlike
        # DetectionTab's live source-switchable stack.
        self.detector_config_stack = QStackedWidget()

        gray_page = QWidget()
        gray_form = QFormLayout(gray_page)
        gray_threshold_row = QHBoxLayout()
        self.possible_spin = QSpinBox()
        self.possible_spin.setRange(0, 255)
        self.probable_spin = QSpinBox()
        self.probable_spin.setRange(0, 255)
        self.strong_spin = QSpinBox()
        self.strong_spin.setRange(0, 255)
        for label_text, spin in (("Possible:", self.possible_spin), ("Probable:", self.probable_spin), ("Strong:", self.strong_spin)):
            gray_threshold_row.addWidget(QLabel(label_text))
            gray_threshold_row.addWidget(spin)
        gray_threshold_row.addStretch(1)
        gray_threshold_widget = QWidget()
        gray_threshold_widget.setLayout(gray_threshold_row)
        gray_threshold_widget.setToolTip(
            "0-255 cut points applied to the BACKGROUND-SUBTRACTED DIFFERENCE image, same meaning "
            "as the Detection tab's live thresholds. Must be non-decreasing."
        )
        gray_form.addRow("Difference thresholds:", gray_threshold_widget)
        self.detector_config_stack.addWidget(gray_page)  # index 0

        color_page = QWidget()
        color_form = QFormLayout(color_page)

        # Multiple presets can be checked at once (e.g. cyan + blue
        # together) -- active_hue_ranges() concatenates every checked
        # preset's hue range(s), so a pixel matching ANY of them counts.
        # This edits self._detection_config only (see
        # _update_config_from_controls), never the recording's own
        # immutable snapshot or the live Detection tab's config.
        color_preset_row = QHBoxLayout()
        self.color_preset_checks: dict[str, QCheckBox] = {}
        for name in sorted(color_detection.HUE_PRESETS.keys()):
            check = QCheckBox(name)
            self.color_preset_checks[name] = check
            color_preset_row.addWidget(check)
        color_preset_row.addStretch(1)
        color_preset_widget = QWidget()
        color_preset_widget.setLayout(color_preset_row)
        color_preset_widget.setToolTip("Check one or more colors -- a pixel counts if it matches ANY checked color's hue range.")
        color_form.addRow("Color preset(s):", color_preset_widget)

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
        color_form.addRow("Hue range:", hue_widget)

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
        weak_widget.setToolTip("The broad 'weak' floor -- pixels below either of these are 0 confidence regardless of hue.")
        color_form.addRow("Weak (broad) floor:", weak_widget)

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
            "The strict 'core' ceiling -- a fragment is only kept if it contains at least one pixel at this level."
        )
        color_form.addRow("Core (strict) ceiling:", core_widget)

        color_threshold_row = QHBoxLayout()
        self.vdo_possible_spin = QSpinBox()
        self.vdo_possible_spin.setRange(0, 255)
        self.vdo_probable_spin = QSpinBox()
        self.vdo_probable_spin.setRange(0, 255)
        self.vdo_strong_spin = QSpinBox()
        self.vdo_strong_spin.setRange(0, 255)
        for label_text, spin in (
            ("Possible:", self.vdo_possible_spin), ("Probable:", self.vdo_probable_spin), ("Strong:", self.vdo_strong_spin),
        ):
            color_threshold_row.addWidget(QLabel(label_text))
            color_threshold_row.addWidget(spin)
        color_threshold_row.addStretch(1)
        color_threshold_widget = QWidget()
        color_threshold_widget.setLayout(color_threshold_row)
        color_threshold_widget.setToolTip("0-255 cut points on the color-confidence scale (see Weak/Core above).")
        color_form.addRow("Confidence thresholds:", color_threshold_widget)

        self.detector_config_stack.addWidget(color_page)  # index 1

        detection_layout.addWidget(self.detector_config_stack)

        fragment_overlay_row = QHBoxLayout()
        fragment_overlay_row.addWidget(QLabel("Fragment overlay on Corrected panel:"))
        self.fragment_overlay_combo = QComboBox()
        self.fragment_overlay_combo.addItems([FRAGMENT_OVERLAY_OFF, FRAGMENT_OVERLAY_FULL, FRAGMENT_OVERLAY_ROI])
        self.fragment_overlay_combo.setToolTip(
            "Full-frame: contours from the whole-image detector. Per-ROI: contours from each ROI's own "
            "independent detection (see the ROI section below) -- switch freely, both come from the same "
            "'Detect current frame' run."
        )
        fragment_overlay_row.addWidget(self.fragment_overlay_combo)
        fragment_overlay_row.addStretch(1)
        detection_layout.addLayout(fragment_overlay_row)

        run_row = QHBoxLayout()
        self.detect_current_button = QPushButton("Detect: Current Frame")
        run_row.addWidget(self.detect_current_button)

        run_row.addSpacing(12)
        run_row.addWidget(QLabel("Range:"))
        self.range_start_spin = QSpinBox()
        self.range_start_spin.setRange(0, 0)
        run_row.addWidget(self.range_start_spin)
        run_row.addWidget(QLabel("to"))
        self.range_end_spin = QSpinBox()
        self.range_end_spin.setRange(0, 0)
        run_row.addWidget(self.range_end_spin)
        self.detect_range_button = QPushButton("Detect: Range")
        run_row.addWidget(self.detect_range_button)

        run_row.addSpacing(12)
        self.detect_full_button = QPushButton("Detect: Full Recording")
        run_row.addWidget(self.detect_full_button)
        run_row.addStretch(1)
        detection_layout.addLayout(run_row)

        self.detection_progress_label = QLabel("")
        detection_layout.addWidget(self.detection_progress_label)

        self.detection_report_text = QPlainTextEdit()
        self.detection_report_text.setReadOnly(True)
        self.detection_report_text.setMaximumHeight(200)
        detection_layout.addWidget(self.detection_report_text)

        layout.addWidget(self.detection_group)

        # ---- ROI (per-pad) analysis ----
        # ROIs are a SEARCH BOUNDARY, not a measurement -- they are never
        # counted as contact themselves. Draw each one slightly LARGER
        # than the expected pad contact area so it never clips valid
        # contact; enter the pad's real expected_area_mm2 below so
        # coverage is computed against that, not the ROI's own (larger,
        # arbitrary) rectangle area.
        self.roi_group = QGroupBox("ROIs (per-pad search boundaries -- independent detection per ROI, see below)")
        roi_layout = QVBoxLayout(self.roi_group)

        roi_help = QLabel(
            "Pause on a reference frame, click \"Define ROIs\", then click-drag a rectangle around each "
            "expected pad on the Corrected panel. Draw each ROI slightly LARGER than the pad's real contact "
            "area -- it is a search boundary only, never counted as contact itself. Detection runs "
            "INDEPENDENTLY within each ROI (pixels outside a pad's ROI can never affect that pad's result)."
        )
        roi_help.setWordWrap(True)
        roi_layout.addWidget(roi_help)

        self.roi_overlap_warning_label = QLabel("")
        self.roi_overlap_warning_label.setWordWrap(True)
        self.roi_overlap_warning_label.setStyleSheet("color: white; background-color: #b00000; padding: 4px;")
        self.roi_overlap_warning_label.setVisible(False)
        roi_layout.addWidget(self.roi_overlap_warning_label)

        roi_define_row = QHBoxLayout()
        self.define_rois_button = QPushButton("Define ROIs")
        self.define_rois_button.setCheckable(True)
        roi_define_row.addWidget(self.define_rois_button)

        roi_define_row.addSpacing(12)
        roi_define_row.addWidget(QLabel("New ROI frame range:"))
        self.roi_new_start_spin = QSpinBox()
        self.roi_new_start_spin.setRange(0, 0)
        roi_define_row.addWidget(self.roi_new_start_spin)
        roi_define_row.addWidget(QLabel("to"))
        self.roi_new_end_spin = QSpinBox()
        self.roi_new_end_spin.setRange(0, 0)
        roi_define_row.addWidget(self.roi_new_end_spin)
        roi_define_row.addStretch(1)
        roi_layout.addLayout(roi_define_row)

        self.roi_table = QTableWidget(0, len(ROI_COLUMNS))
        self.roi_table.setHorizontalHeaderLabels(ROI_COLUMNS)
        self.roi_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.roi_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.roi_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.roi_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.roi_table.setMaximumHeight(150)
        roi_layout.addWidget(self.roi_table)

        roi_action_row = QHBoxLayout()
        self.roi_rename_button = QPushButton("Rename Selected")
        roi_action_row.addWidget(self.roi_rename_button)
        self.roi_edit_area_button = QPushButton("Edit Expected Area")
        roi_action_row.addWidget(self.roi_edit_area_button)
        self.roi_edit_min_area_button = QPushButton("Edit Min Area (Noise Filter)")
        roi_action_row.addWidget(self.roi_edit_min_area_button)
        self.roi_delete_button = QPushButton("Delete Selected")
        roi_action_row.addWidget(self.roi_delete_button)
        roi_action_row.addStretch(1)
        roi_layout.addLayout(roi_action_row)

        self.show_rois_check = QCheckBox("Show ROIs (rect + name) on Corrected panel")
        roi_layout.addWidget(self.show_rois_check)

        self.combine_roi_fragments_check = QCheckBox(
            "Combine each ROI's fragments into one region (Per-ROI overlay only, display only)"
        )
        self.combine_roi_fragments_check.setToolTip(
            "When checked, the Per-ROI overlay draws ONE outline + ONE total area label per ROI instead "
            "of a separate outline/label for every disconnected detected blob inside it. Display only -- "
            "the underlying possible/probable/strong pixel totals used for coverage are already summed "
            "across all of an ROI's fragments either way; this only changes what gets drawn."
        )
        roi_layout.addWidget(self.combine_roi_fragments_check)

        self.roi_progress_label = QLabel("")
        roi_layout.addWidget(self.roi_progress_label)

        self.roi_report_text = QPlainTextEdit()
        self.roi_report_text.setReadOnly(True)
        self.roi_report_text.setMaximumHeight(200)
        roi_layout.addWidget(self.roi_report_text)

        layout.addWidget(self.roi_group)

        self._set_controls_enabled(False)
        self._set_detection_enabled(False)
        self._set_roi_defining_enabled(False)

        self.first_button.clicked.connect(lambda: self.seek_to(0))
        self.prev_button.clicked.connect(lambda: self.seek_to(self._current_frame_number() - 1))
        self.next_button.clicked.connect(lambda: self.seek_to(self._current_frame_number() + 1))
        self.last_button.clicked.connect(lambda: self.seek_to(self._total_frames - 1))
        self.play_pause_button.clicked.connect(self._toggle_play)
        self.timeline_slider.valueChanged.connect(self._on_slider_changed)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        self.raw_scroll.resized.connect(self._rerender_current)
        self.corrected_scroll.resized.connect(self._rerender_current)
        self.fragment_overlay_combo.currentIndexChanged.connect(lambda _index: self._rerender_current())
        self.detect_current_button.clicked.connect(self.detect_current_frame)
        self.detect_range_button.clicked.connect(self.detect_range)
        self.detect_full_button.clicked.connect(self.detect_full_recording)

        for widget in (self.possible_spin, self.probable_spin, self.strong_spin):
            widget.valueChanged.connect(self._update_config_from_controls)
        for check in self.color_preset_checks.values():
            check.toggled.connect(self._update_config_from_controls)
        self.hue_override_check.toggled.connect(self._update_config_from_controls)
        for widget in (
            self.hue_min_spin, self.hue_max_spin, self.weak_sat_spin, self.weak_val_spin,
            self.core_sat_spin, self.core_val_spin, self.vdo_possible_spin, self.vdo_probable_spin, self.vdo_strong_spin,
        ):
            widget.valueChanged.connect(self._update_config_from_controls)

        self.raw_label.doubleClicked.connect(lambda: self._toggle_expand("raw"))
        self.corrected_label.doubleClicked.connect(lambda: self._toggle_expand("corrected"))
        self.setFocusPolicy(Qt.StrongFocus)  # so Escape reaches keyPressEvent below while expanded

        self.define_rois_button.toggled.connect(self._on_define_rois_toggled)
        self.corrected_label.roiDrawn.connect(self._on_roi_drawn)
        self.roi_table.itemSelectionChanged.connect(self._on_roi_selection_changed)
        self.roi_rename_button.clicked.connect(self.rename_selected_roi)
        self.roi_edit_area_button.clicked.connect(self.edit_selected_roi_expected_area)
        self.roi_edit_min_area_button.clicked.connect(self.edit_selected_roi_min_area)
        self.roi_delete_button.clicked.connect(self.delete_selected_roi)
        self.show_rois_check.toggled.connect(lambda _checked: self._rerender_current())
        self.combine_roi_fragments_check.toggled.connect(lambda _checked: self._rerender_current())

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._on_play_tick)

    # ---- panel expand/collapse ----

    def _toggle_expand(self, panel: str) -> None:
        if self.expanded_panel == panel:
            self._collapse_expand()
            return

        self.expanded_panel = panel
        # Hides the sibling panel AND every other tall block on the page
        # (detection controls/report, the ROI group, the header labels)
        # -- playback controls, the timeline, and zoom/speed stay visible
        # so an expanded view is still fully navigable, not just a static
        # blown-up screenshot. Without hiding the ROI group specifically,
        # "expand" only freed up half the panels_row's WIDTH (the sibling
        # panel) while everything below it kept eating the same vertical
        # space -- not the actual "make it bigger" a user wants, since
        # the video's HEIGHT is usually the tighter constraint once
        # zoomed in.
        self.raw_group.setVisible(panel == "raw")
        self.corrected_group.setVisible(panel == "corrected")
        self.detection_group.setVisible(False)
        self.roi_group.setVisible(False)
        self.info_label.setVisible(False)
        self.calibration_status_label.setVisible(False)
        self.expand_hint.setVisible(False)
        self.setFocus()  # so a stray Escape press (without clicking the image first) still works

    def _collapse_expand(self) -> None:
        self.expanded_panel = None
        self.raw_group.setVisible(True)
        self.corrected_group.setVisible(True)
        self.detection_group.setVisible(True)
        self.roi_group.setVisible(True)
        self.info_label.setVisible(True)
        self.calibration_status_label.setVisible(True)
        self.expand_hint.setVisible(True)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape and self.expanded_panel is not None:
            self._collapse_expand()
            return
        super().keyPressEvent(event)

    # ---- opening a recording ----

    def open_recording(self, recording_id: str) -> None:
        self.close_recording()

        metadata = rs.load_metadata(recording_id)
        if metadata is None:
            self.info_label.setText(f"Could not load metadata for {recording_id!r}.")
            return

        video_path = rs.video_path_for(recording_id, metadata)
        if video_path is None:
            self.info_label.setText(f"No video file found for {recording_id!r}.")
            return

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.info_label.setText(f"Could not open {video_path.name} for playback.")
            return

        self.recording_id = recording_id
        self._metadata = metadata
        self._cap = cap
        self._next_read_frame_number = 0

        recording_block = metadata.get("recording", {})
        self._total_frames = (
            recording_block.get("output_frames_verified")
            or recording_block.get("frames_write_attempted")
            or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        self._frame_index_rows = list(rs.read_frame_index(recording_id))
        self._crop = metadata.get("calibration", {}).get("usable_roi_crop_percentages") or {
            "top": 0, "bottom": 0, "left": 0, "right": 0,
        }

        self._load_calibration_snapshot(metadata)
        self._load_detection_inputs(metadata)

        self.project_id = pp.get_or_create_project_for_recording(recording_id)
        self._refresh_roi_list()
        self._set_roi_defining_enabled(True)
        self.show_rois_check.setChecked(True)
        self.combine_roi_fragments_check.setChecked(True)

        self.info_label.setText(
            f"{metadata.get('experiment_name', recording_id)}  ({recording_id})   "
            f"source={metadata.get('source')}   quality_mode={metadata.get('quality_mode')}   "
            f"codec={metadata.get('camera', {}).get('codec_fourcc')}"
        )

        self.timeline_slider.setEnabled(True)
        self.timeline_slider.setRange(0, max(0, self._total_frames - 1))
        self._set_controls_enabled(self._total_frames > 0)

        self.range_start_spin.setRange(0, max(0, self._total_frames - 1))
        self.range_end_spin.setRange(0, max(0, self._total_frames - 1))
        self.range_start_spin.setValue(0)
        self.range_end_spin.setValue(max(0, self._total_frames - 1))

        self.roi_new_start_spin.setRange(0, max(0, self._total_frames - 1))
        self.roi_new_end_spin.setRange(0, max(0, self._total_frames - 1))
        self.roi_new_start_spin.setValue(0)
        self.roi_new_end_spin.setValue(max(0, self._total_frames - 1))

        self.seek_to(0)

    def close_recording(self) -> None:
        self.play_timer.stop()
        self.play_pause_button.setText("Play")

        if self.expanded_panel is not None:
            self._collapse_expand()

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self.recording_id = None
        self._metadata = None
        self._total_frames = 0
        self._frame_index_rows = []
        self._undistort_maps = None
        self._undistort_target_size = None
        self._last_raw_frame = None

        self._detector_type = None
        self._detection_config = None
        self._detection_background = None
        self._detection_results = {}

        if self.define_rois_button.isChecked():
            self.define_rois_button.setChecked(False)
        self.project_id = None
        self._rois = []
        self._roi_overlaps = []
        self._roi_current_frame_results = {}
        self._roi_frame_fragments = {}

        self.timeline_slider.setEnabled(False)
        self.timeline_slider.setRange(0, 0)
        self._set_controls_enabled(False)
        self._set_detection_enabled(False)
        self._set_roi_defining_enabled(False)

        self.range_start_spin.setRange(0, 0)
        self.range_end_spin.setRange(0, 0)
        self.roi_new_start_spin.setRange(0, 0)
        self.roi_new_end_spin.setRange(0, 0)
        self.roi_table.setRowCount(0)
        self.roi_overlap_warning_label.setVisible(False)

        self.raw_label.setPixmap(QPixmap())
        self.raw_label.setText("No frame loaded.")
        self.corrected_label.setPixmap(QPixmap())
        self.corrected_label.setText("No frame loaded.")
        self.frame_label.setText("Frame: -- / --")
        self.time_label.setText("Time: --")
        self.info_label.setText("No recording open -- use the Recordings tab's \"Open in Processing\" button.")
        self.calibration_status_label.setText("")
        self.detection_status_label.setText("")
        self.detection_progress_label.setText("")
        self.detection_report_text.setPlainText("")
        self.roi_progress_label.setText("")
        self.roi_report_text.setPlainText("")

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.first_button, self.prev_button, self.play_pause_button,
            self.next_button, self.last_button,
        ):
            widget.setEnabled(enabled)

    def _set_detection_enabled(self, enabled: bool) -> None:
        for widget in (
            self.detect_current_button, self.detect_range_button, self.detect_full_button,
            self.range_start_spin, self.range_end_spin,
        ):
            widget.setEnabled(enabled)

    def _set_roi_defining_enabled(self, enabled: bool) -> None:
        self.define_rois_button.setEnabled(enabled)
        self.roi_new_start_spin.setEnabled(enabled)
        self.roi_new_end_spin.setEnabled(enabled)

    def _load_calibration_snapshot(self, metadata: dict) -> None:
        calibration = distortion.load_calibration(path=rs.snapshot_distortion_path(self.recording_id))

        if calibration is None:
            self.calibration_status_label.setText(
                "No lens-distortion calibration in this recording's snapshot -- showing raw only "
                "(the Corrected panel will mirror Raw)."
            )
            self._undistort_maps = None
            self._undistort_target_size = None
            return

        actual_width = metadata.get("camera", {}).get("actual_width")
        actual_height = metadata.get("camera", {}).get("actual_height")
        cal_width, cal_height = calibration["image_size"]

        if actual_width and actual_height and (actual_width, actual_height) != (cal_width, cal_height):
            self.calibration_status_label.setText(
                f"WARNING: this recording is {actual_width}x{actual_height} but its OWN calibration "
                f"snapshot was done at {cal_width}x{cal_height} -- showing raw only (undistort maps "
                f"would sample the wrong region)."
            )
            self._undistort_maps = None
            self._undistort_target_size = None
            return

        self._undistort_maps = distortion.build_undistort_maps(calibration)
        self._undistort_target_size = (cal_width, cal_height)
        self.calibration_status_label.setText(
            "Using this recording's own calibration snapshot (never the live profile)."
        )

    def _load_detection_inputs(self, metadata: dict) -> None:
        """
        Resolves detector_type/config/background from THIS recording's
        own calibration_snapshot/ (circle_config.json,
        background_reference.npy) -- never the live active profile, same
        self-containment guarantee _load_calibration_snapshot already
        gives undistortion.
        """
        self._detector_type = metadata.get("calibration", {}).get("detector_type")
        self._detection_config = circle_config_module.load_circle_config(
            path=rs.snapshot_circle_config_path(self.recording_id)
        )
        self._detection_background = None
        self._detection_results = {}

        if self._detector_type == "grayscale":
            self._detection_background = background_reference.load_background_reference(
                path=rs.snapshot_background_reference_path(self.recording_id)
            )

        self.detector_config_stack.setCurrentIndex(0 if self._detector_type == "grayscale" else 1)
        self._populate_config_controls()

        if self._detector_type is None:
            self.detection_status_label.setText(
                "Detection unavailable: this recording has no known detector_type."
            )
            self._set_detection_enabled(False)
        elif self._detector_type == "grayscale" and self._detection_background is None:
            self.detection_status_label.setText(
                "Detection unavailable: this recording's calibration snapshot has no "
                "background_reference.npy (grayscale detection needs one)."
            )
            self._set_detection_enabled(False)
        elif self._undistort_maps is None:
            self.detection_status_label.setText(
                "Detection unavailable: no usable calibration for this recording (see the "
                "warning above) -- detection needs the same undistort+crop the Corrected panel uses."
            )
            self._set_detection_enabled(False)
        else:
            self.detection_status_label.setText(
                f"Detector: {self._detector_type} (thresholds + background from this recording's own snapshot)."
            )
            self._set_detection_enabled(True)

    def _populate_config_controls(self) -> None:
        """Reflects self._detection_config (just loaded from this
        recording's own snapshot) into the threshold/color widgets --
        blockSignals so this doesn't itself trigger
        _update_config_from_controls, same pattern DetectionTab's own
        _load_config() uses."""
        config = self._detection_config
        if config is None:
            return

        widgets = (
            self.possible_spin, self.probable_spin, self.strong_spin,
            *self.color_preset_checks.values(), self.hue_override_check, self.hue_min_spin, self.hue_max_spin,
            self.weak_sat_spin, self.weak_val_spin, self.core_sat_spin, self.core_val_spin,
            self.vdo_possible_spin, self.vdo_probable_spin, self.vdo_strong_spin,
        )
        for widget in widgets:
            widget.blockSignals(True)

        self.possible_spin.setValue(config.get("possible_threshold", 0))
        self.probable_spin.setValue(config.get("probable_threshold", 0))
        self.strong_spin.setValue(config.get("strong_threshold", 0))

        # color_presets (list) takes priority if present; otherwise -- a
        # recording saved before this multi-select feature existed, or
        # any recording at all -- default to blue+cyan (this project's
        # own material), not whichever single legacy color_preset
        # happened to be active when it was recorded. This only affects
        # the STARTING point of these in-memory, never-persisted
        # controls (see _update_config_from_controls); the recording's
        # own immutable snapshot is untouched either way. Written into
        # `config` (== self._detection_config) immediately, not just the
        # checkboxes -- blockSignals means _update_config_from_controls
        # never fires here, so without this line an untouched checkbox
        # default would show blue+cyan checked while detection quietly
        # kept using whatever single color_preset was actually loaded.
        selected_presets = config.get("color_presets") or ["blue", "cyan"]
        config["color_presets"] = selected_presets
        for name, check in self.color_preset_checks.items():
            check.setChecked(name in selected_presets)
        has_override = config.get("hue_min") is not None and config.get("hue_max") is not None
        self.hue_override_check.setChecked(has_override)
        self.hue_min_spin.setValue(config.get("hue_min") if has_override else 0)
        self.hue_max_spin.setValue(config.get("hue_max") if has_override else 179)
        # Same override as color_presets above: Processing tab always
        # starts the weak floor at 90/90 (this project's validated
        # working values -- unlike a genuinely-missing key, an old
        # recording's saved weak_sat_min/weak_val_min can't be
        # distinguished from "just whatever the default used to be", so
        # this deliberately does NOT defer to whatever the snapshot has
        # saved). Written into `config` immediately for the same reason
        # as color_presets -- otherwise the spinboxes would show 90 while
        # detection quietly still used the recording's old saved value.
        config["weak_sat_min"] = 90
        config["weak_val_min"] = 90
        self.weak_sat_spin.setValue(90)
        self.weak_val_spin.setValue(90)
        self.core_sat_spin.setValue(config.get("core_sat_min", 0))
        self.core_val_spin.setValue(config.get("core_val_min", 0))
        self.vdo_possible_spin.setValue(config.get("vdo_possible_threshold", 0))
        self.vdo_probable_spin.setValue(config.get("vdo_probable_threshold", 0))
        self.vdo_strong_spin.setValue(config.get("vdo_strong_threshold", 0))

        for widget in widgets:
            widget.blockSignals(False)

    def _update_config_from_controls(self, *_args) -> None:
        """Live-edits self._detection_config from whichever widgets are
        showing -- takes effect on the NEXT Detect run, never written
        back into the recording's own (immutable) circle_config.json
        snapshot. Invalidates cached per-frame results below: a fragment
        cached under the OLD thresholds must never be shown/trusted once
        the thresholds have changed."""
        if self._detection_config is None:
            return

        if self._detector_type == "grayscale":
            self._detection_config["possible_threshold"] = self.possible_spin.value()
            self._detection_config["probable_threshold"] = self.probable_spin.value()
            self._detection_config["strong_threshold"] = self.strong_spin.value()
        else:
            self._detection_config["color_presets"] = [
                name for name, check in self.color_preset_checks.items() if check.isChecked()
            ]
            has_override = self.hue_override_check.isChecked()
            self._detection_config["hue_min"] = self.hue_min_spin.value() if has_override else None
            self._detection_config["hue_max"] = self.hue_max_spin.value() if has_override else None
            self._detection_config["weak_sat_min"] = self.weak_sat_spin.value()
            self._detection_config["weak_val_min"] = self.weak_val_spin.value()
            self._detection_config["core_sat_min"] = self.core_sat_spin.value()
            self._detection_config["core_val_min"] = self.core_val_spin.value()
            self._detection_config["vdo_possible_threshold"] = self.vdo_possible_spin.value()
            self._detection_config["vdo_probable_threshold"] = self.vdo_probable_spin.value()
            self._detection_config["vdo_strong_threshold"] = self.vdo_strong_spin.value()

        self._detection_results = {}
        self._roi_frame_fragments = {}
        self._roi_current_frame_results = {}
        self._rerender_current()

    # ---- ROI definition ----

    def _on_define_rois_toggled(self, checked: bool) -> None:
        if checked and self.play_timer.isActive():
            self._toggle_play()  # pause -- drawing against a moving frame doesn't make sense
        self.corrected_label.drawing_enabled = checked

    def _label_rect_to_corrected_rect(self, qrect: QRect) -> dict:
        """Converts a rectangle in LABEL-local pixel coordinates (as
        QRubberBand/_RoiLabel report it) into corrected-frame pixel
        coordinates, using the scale _show_image() actually rendered the
        corrected panel at. _corrected_pixmap_offset accounts for the
        (usually zero) letterboxing that appears when the label's
        setMinimumSize floor forces it larger than the scaled pixmap --
        see the comment in _show_image."""
        scale = self._corrected_display_scale or 1.0
        offset_x, offset_y = self._corrected_pixmap_offset
        x = max(0, int(round((qrect.x() - offset_x) / scale)))
        y = max(0, int(round((qrect.y() - offset_y) / scale)))
        width = max(1, int(round(qrect.width() / scale)))
        height = max(1, int(round(qrect.height() / scale)))
        return {"x": x, "y": y, "width": width, "height": height}

    def _on_roi_drawn(self, qrect: QRect) -> None:
        if self.project_id is None or self._undistort_target_size is None:
            return

        rectangle = self._label_rect_to_corrected_rect(qrect)

        dialog = _NewRoiDialog(default_name=f"Pad {len(self._rois) + 1}", parent=self)
        if dialog.exec() != QDialog.Accepted:
            return

        name, expected_area_mm2, min_area_mm2 = dialog.result_values()
        corrected_width, corrected_height = self._undistort_target_size

        pp.add_roi(
            self.project_id,
            name,
            rectangle,
            reference_frame_number=self._current_frame_number(),
            start_frame=self.roi_new_start_spin.value(),
            end_frame=self.roi_new_end_spin.value(),
            corrected_width=corrected_width,
            corrected_height=corrected_height,
            calibration_snapshot_reference={"recording_id": self.recording_id, "snapshot_dir": "calibration_snapshot/"},
            crop_info=dict(self._crop),
            expected_area_mm2=expected_area_mm2,
            min_area_mm2=min_area_mm2,
        )

        self._refresh_roi_list()
        self.show_rois_check.setChecked(True)
        self._rerender_current()

    def _refresh_roi_list(self) -> None:
        if self.project_id is None:
            self._rois = []
            self._roi_overlaps = []
            self.roi_table.setRowCount(0)
            self.roi_overlap_warning_label.setVisible(False)
            return

        self._rois = pp.list_rois(self.project_id)
        self._roi_overlaps = pp.find_overlapping_pairs(self._rois)
        overlapping_ids = {roi_id for pair in self._roi_overlaps for roi_id in pair}

        self.roi_table.setRowCount(len(self._rois))
        for row, roi in enumerate(self._rois):
            rect = roi["rectangle"]
            values = [
                roi["name"],
                f"({rect['x']}, {rect['y']}) {rect['width']}x{rect['height']}",
                f"{roi['start_frame']} - {roi['end_frame']}",
                f"{roi['expected_area_mm2']:.4f}" if roi["expected_area_mm2"] is not None else "N/A",
                f"{roi['min_area_mm2']:.4f}" if roi.get("min_area_mm2") is not None else "off",
                "OVERLAP" if roi["roi_id"] in overlapping_ids else "",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, roi["roi_id"])
                if roi["roi_id"] in overlapping_ids:
                    item.setForeground(Qt.red)
                self.roi_table.setItem(row, col, item)

        if self._roi_overlaps:
            names_by_id = {roi["roi_id"]: roi["name"] for roi in self._rois}
            pair_text = ", ".join(f"{names_by_id[a]} / {names_by_id[b]}" for a, b in self._roi_overlaps)
            self.roi_overlap_warning_label.setText(
                f"⚠ INVALID ROI SET: overlapping ROIs ({pair_text}). Range/Full detection is disabled "
                f"until this is resolved -- current-frame preview may still be used to help fix it, but its "
                f"per-ROI numbers are NOT valid measurements while this warning is showing."
            )
            self.roi_overlap_warning_label.setVisible(True)
        else:
            self.roi_overlap_warning_label.setVisible(False)

        # Range/Full must never run ROI analysis against an invalid
        # (overlapping) ROI set -- gate the buttons themselves rather
        # than trying to run a partial/ambiguous analysis. Full-frame-
        # only detection (the pre-existing Stage E pipeline) is
        # unaffected when there are 0 or 1 ROIs, since overlap requires
        # at least 2.
        self.detect_range_button.setEnabled(not self._roi_overlaps)
        self.detect_full_button.setEnabled(not self._roi_overlaps)

    def _selected_roi_id(self) -> str | None:
        row = self.roi_table.currentRow()
        if row < 0 or row >= len(self._rois):
            return None
        return self._rois[row]["roi_id"]

    def _on_roi_selection_changed(self) -> None:
        pass  # reserved -- no per-selection UI reaction needed yet beyond what the action buttons already do on click

    def rename_selected_roi(self) -> None:
        roi_id = self._selected_roi_id()
        if roi_id is None or self.project_id is None:
            return

        current = next(r["name"] for r in self._rois if r["roi_id"] == roi_id)
        dialog = QDialog(self)
        dialog.setWindowTitle("Rename ROI")
        form = QFormLayout(dialog)
        name_edit = QLineEdit(current)
        form.addRow("Name:", name_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.Accepted or not name_edit.text().strip():
            return

        pp.rename_roi(self.project_id, roi_id, name_edit.text().strip())
        self._refresh_roi_list()
        self._rerender_current()

    def edit_selected_roi_expected_area(self) -> None:
        roi_id = self._selected_roi_id()
        if roi_id is None or self.project_id is None:
            return

        current = next(r for r in self._rois if r["roi_id"] == roi_id)
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Expected Area")
        form = QFormLayout(dialog)
        area_spin = QDoubleSpinBox()
        area_spin.setRange(0.0, 100000.0)
        area_spin.setDecimals(4)
        area_spin.setSuffix(" mm2 (0 = not set)")
        area_spin.setValue(current["expected_area_mm2"] or 0.0)
        form.addRow("Expected physical pad area:", area_spin)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        new_value = area_spin.value() if area_spin.value() > 0 else None
        pp.update_roi_expected_area(self.project_id, roi_id, new_value)
        self._refresh_roi_list()

    def edit_selected_roi_min_area(self) -> None:
        roi_id = self._selected_roi_id()
        if roi_id is None or self.project_id is None:
            return

        current = next(r for r in self._rois if r["roi_id"] == roi_id)
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Min Area (Noise Filter)")
        form = QFormLayout(dialog)
        area_spin = QDoubleSpinBox()
        area_spin.setRange(0.0, 100000.0)
        area_spin.setDecimals(4)
        area_spin.setSuffix(" mm2/px (0 = no filter)")
        area_spin.setValue(current.get("min_area_mm2") or 0.0)
        form.addRow("Min. blob area:", area_spin)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        new_value = area_spin.value() if area_spin.value() > 0 else None
        pp.update_roi_min_area(self.project_id, roi_id, new_value)
        self._refresh_roi_list()

    def delete_selected_roi(self) -> None:
        roi_id = self._selected_roi_id()
        if roi_id is None or self.project_id is None:
            return

        name = next(r["name"] for r in self._rois if r["roi_id"] == roi_id)
        confirm = QMessageBox.warning(
            self, "Delete ROI", f'Delete "{name}"?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if confirm != QMessageBox.Yes:
            return

        pp.delete_roi(self.project_id, roi_id)
        self._roi_current_frame_results.pop(roi_id, None)
        for frame_fragments in self._roi_frame_fragments.values():
            frame_fragments.pop(roi_id, None)
        self._refresh_roi_list()
        self._rerender_current()

    def _draw_roi_overlay(self, frame, frame_number: int):
        """Draws each applicable ROI's rectangle + name (+ an OVERLAP
        tag when relevant) directly on the corrected preview. "Applicable"
        means frame_number falls within that ROI's own start_frame/
        end_frame -- the table above still lists every ROI regardless."""
        overlapping_ids = {roi_id for pair in self._roi_overlaps for roi_id in pair}
        frame = frame.copy()

        for roi in self._rois:
            if not (roi["start_frame"] <= frame_number <= roi["end_frame"]):
                continue

            rect = roi["rectangle"]
            is_overlapping = roi["roi_id"] in overlapping_ids
            color = ROI_OVERLAP_COLOR if is_overlapping else ROI_COLOR
            x, y, w, h = rect["x"], rect["y"], rect["width"], rect["height"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label = f"{roi['name']}" + (" [OVERLAP]" if is_overlapping else "")
            cv2.putText(frame, label, (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        if self._roi_overlaps:
            cv2.putText(
                frame, "INVALID ROI SET -- OVERLAPPING ROIs, NOT VALID FOR MEASUREMENT",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ROI_OVERLAP_COLOR, 2, cv2.LINE_AA,
            )

        return frame

    # ---- frame access ----

    def _current_frame_number(self) -> int:
        return self._next_read_frame_number - 1 if self._next_read_frame_number > 0 else 0

    def _read_frame_at(self, frame_number: int):
        if self._cap is None:
            return None

        if frame_number != self._next_read_frame_number:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ok, frame = self._cap.read()
        if not ok:
            return None

        self._next_read_frame_number = frame_number + 1
        return frame

    def seek_to(self, frame_number: int) -> None:
        if self._cap is None or self._total_frames == 0:
            return

        frame_number = max(0, min(frame_number, self._total_frames - 1))
        frame = self._read_frame_at(frame_number)
        if frame is None:
            return

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(frame_number)
        self.timeline_slider.blockSignals(False)

        self._render(frame_number, frame)

        if frame_number >= self._total_frames - 1:
            self.play_timer.stop()
            self.play_pause_button.setText("Play")

    def _on_slider_changed(self, value: int) -> None:
        self.seek_to(value)

    # ---- rendering ----

    def _compute_corrected_frame(self, raw_frame):
        """Undistort + crop, exactly like the live preview's own pipeline
        (LiveCameraTab/DetectionTab) -- or None if this recording has no
        usable calibration for its actual frame size. Shared by
        rendering AND detection below, so both always agree on what
        "corrected" means for this recording."""
        if self._undistort_maps is None:
            return None

        frame_size = (raw_frame.shape[1], raw_frame.shape[0])
        if frame_size != self._undistort_target_size:
            return None

        corrected = distortion.undistort_with_maps(raw_frame, self._undistort_maps)
        return distortion.crop_edges(
            corrected,
            top_pct=self._crop.get("top", 0),
            bottom_pct=self._crop.get("bottom", 0),
            left_pct=self._crop.get("left", 0),
            right_pct=self._crop.get("right", 0),
        )

    def _draw_fragment_overlay(self, frame, fragments: list):
        """Draws each fragment's contour plus an area label (mm2 if this
        recording has a saved physical scale, else px) right next to it
        -- reuses regions.from_fragment() for the mm2 conversion so the
        label always matches the text report's own numbers exactly,
        never a second, potentially-inconsistent calculation."""
        mm_per_pixel = self._metadata.get("calibration", {}).get("mm_per_pixel") if self._metadata else None
        source_type = (self._metadata.get("camera", {}).get("source_type") or "unknown") if self._metadata else "unknown"

        frame = frame.copy()
        for fragment in fragments:
            cv2.drawContours(frame, [fragment.contour], -1, FRAGMENT_OUTLINE_COLOR, 1)

            if mm_per_pixel is not None:
                region = regions.from_fragment(fragment, source_type, mm_per_pixel)
                label = f"{region.possible_area_mm2:.2f}mm2"
            else:
                label = f"{fragment.possible_area_px}px"

            label_pos = (max(0, int(fragment.centroid[0]) - 20), max(12, int(fragment.centroid[1])))
            _draw_outlined_text(frame, label, label_pos, font_scale=0.4, thickness=1)

        return frame

    def _draw_combined_roi_fragment_overlay(self, frame, frame_roi_fragments: dict[str, list]):
        """Per-ROI overlay, "combine" mode: ONE (or, if the fragments are
        disjoint, a few) outline(s) tracing the TRUE union of every
        fragment's own exact mask, plus ONE total-area label per ROI --
        instead of one outline+label per disconnected fragment. NOT a
        convex hull: a hull straight-lines across any concave dip in the
        real shape (e.g. a crescent-shaped specular highlight), which
        visually looks like it's covering pixels that were never actually
        detected. Rebuilding the real union via each fragment's own mask
        (same technique as detection_pipeline.compute_tier_masks) and
        tracing ITS contour keeps the drawn boundary pixel-accurate.
        Display only -- the area summed here (sum of each fragment's
        possible_area_px) is the exact same number
        summarize_roi_result()/the text report already total for that
        ROI; this never re-detects or re-thresholds anything, only
        re-draws it.

        The total-area label is drawn BELOW the ROI's own rectangle
        (never inside it) in a third, distinct color -- placed over the
        detected region itself, it competed for contrast against
        whatever the pad's actual material color is (e.g. a bright blue
        specular highlight), which is exactly backwards for a label
        that's supposed to always be readable regardless of what's
        underneath."""
        mm_per_pixel = self._metadata.get("calibration", {}).get("mm_per_pixel") if self._metadata else None
        h, w = frame.shape[:2]
        rois_by_id = {roi["roi_id"]: roi for roi in self._rois}

        frame = frame.copy()
        for roi_id, fragments in frame_roi_fragments.items():
            if not fragments:
                continue

            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for fragment in fragments:
                x, y, fw, fh = fragment.bbox
                combined_mask[y : y + fh, x : x + fw] |= fragment.mask.astype(np.uint8)

            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, FRAGMENT_OUTLINE_COLOR, 1)

            total_px = sum(fragment.possible_area_px for fragment in fragments)
            actual_area_mm2 = total_px * mm_per_pixel ** 2 if mm_per_pixel is not None else None
            if actual_area_mm2 is not None:
                label = f"{actual_area_mm2:.2f}mm2"
            else:
                label = f"{total_px}px"

            # coverage_percent here is the SAME formula summarize_roi_result()
            # uses (actual / expected_area_mm2 * 100) -- never against the
            # ROI rectangle's own (deliberately larger, search-boundary-only)
            # area, and never fabricated when either side is unavailable.
            expected_area_mm2 = rois_by_id.get(roi_id, {}).get("expected_area_mm2")
            if actual_area_mm2 is not None and expected_area_mm2 not in (None, 0):
                coverage_percent = actual_area_mm2 / expected_area_mm2 * 100.0
                coverage_label = f"{actual_area_mm2:.2f} / {expected_area_mm2:.2f}mm2 ({coverage_percent:.1f}%)"
            elif expected_area_mm2 not in (None, 0):
                coverage_label = f"expected {expected_area_mm2:.2f}mm2 (no physical scale set -- coverage N/A)"
            else:
                coverage_label = "coverage N/A (no expected area set)"

            rect = rois_by_id.get(roi_id, {}).get("rectangle")
            if rect is not None:
                label_x = max(0, rect["x"])
                label_y = min(h - 24, rect["y"] + rect["height"] + 22)
            else:
                moments = cv2.moments(combined_mask)
                if moments["m00"]:
                    label_x, label_y = int(moments["m10"] / moments["m00"]) - 20, int(moments["m01"] / moments["m00"])
                else:
                    label_x, label_y = int(fragments[0].centroid[0]) - 20, int(fragments[0].centroid[1])
                label_x, label_y = max(0, label_x), max(12, label_y)

            _draw_outlined_text(frame, label, (label_x, label_y), font_scale=0.55, thickness=2)
            _draw_outlined_text(frame, coverage_label, (label_x, min(h - 6, label_y + 20)), font_scale=0.45, thickness=1)

        return frame

    def _render(self, frame_number: int, raw_frame) -> None:
        self._last_raw_frame = raw_frame  # cached so a zoom-only change can re-render without re-reading the video
        self._show_image(self.raw_label, self.raw_scroll, raw_frame)

        corrected = self._compute_corrected_frame(raw_frame)
        display_frame = raw_frame if corrected is None else corrected

        overlay_mode = self.fragment_overlay_combo.currentText()
        if corrected is not None and overlay_mode == FRAGMENT_OVERLAY_FULL:
            frame_fragments = self._detection_results.get(frame_number)
            if frame_fragments:
                display_frame = self._draw_fragment_overlay(display_frame, frame_fragments)
        elif corrected is not None and overlay_mode == FRAGMENT_OVERLAY_ROI:
            # Looked up by THIS frame_number, and filtered to ROIs that
            # actually apply to it -- fragments cached under a different
            # frame, or under an ROI whose start/end range no longer
            # covers this frame, must never be drawn here. Without the
            # frame_number lookup, fragments computed once (e.g. via
            # "Detect current frame") would keep getting redrawn,
            # unchanged, on every later frame the user scrubs to; without
            # the applicability filter, a fragment would still show up
            # floating with no ROI box around it once that ROI's range no
            # longer includes the current frame -- both looked exactly
            # like "detecting outside the ROI".
            applicable_ids = {roi["roi_id"] for roi in self._applicable_rois(frame_number)}
            frame_roi_fragments = self._roi_frame_fragments.get(frame_number, {})
            applicable_roi_fragments = {
                roi_id: fragments for roi_id, fragments in frame_roi_fragments.items() if roi_id in applicable_ids
            }
            if self.combine_roi_fragments_check.isChecked():
                if any(applicable_roi_fragments.values()):
                    display_frame = self._draw_combined_roi_fragment_overlay(display_frame, applicable_roi_fragments)
            else:
                roi_fragments = [fragment for fragments in applicable_roi_fragments.values() for fragment in fragments]
                if roi_fragments:
                    display_frame = self._draw_fragment_overlay(display_frame, roi_fragments)

        if corrected is not None and self.show_rois_check.isChecked() and self._rois:
            display_frame = self._draw_roi_overlay(display_frame, frame_number)

        self._show_image(self.corrected_label, self.corrected_scroll, display_frame)

        self.frame_label.setText(f"Frame: {frame_number} / {max(0, self._total_frames - 1)}")

        recording_time_ns = (
            self._frame_index_rows[frame_number]["recording_time_ns"]
            if frame_number < len(self._frame_index_rows)
            else None
        )
        self.time_label.setText(f"Time: {_format_time_ns(recording_time_ns)}")

    def _show_image(self, label: QLabel, scroll_area: QScrollArea, bgr_image) -> None:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h, w, channels = rgb.shape
        qimage = QImage(rgb.data, w, h, channels * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())

        # zoom=100% reproduces the old "always fits the panel" behavior
        # exactly (fit_scale alone); above 100% the pixmap grows past the
        # viewport and the QScrollArea (setWidgetResizable(False)) shows
        # scrollbars to pan around it -- see the zoom slider's tooltip.
        viewport = scroll_area.viewport().size()
        fit_scale = min(viewport.width() / w, viewport.height() / h) if w and h else 1.0
        effective_scale = max(fit_scale, 0.01) * self._zoom

        scaled = pixmap.scaled(
            max(1, int(w * effective_scale)),
            max(1, int(h * effective_scale)),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        label.setPixmap(scaled)
        label.resize(scaled.size())

        if label is self.corrected_label:
            # The real scale actually used (scaled.width()/w, not the
            # requested effective_scale) -- QPixmap.scaled() with
            # KeepAspectRatio can round to a slightly different final
            # size than requested; mouse->corrected-frame coordinate
            # mapping (_label_rect_to_corrected_rect) must use the SAME
            # number the pixmap was actually produced at, not the
            # pre-rounding target.
            self._corrected_display_scale = scaled.width() / w if w else 1.0

            # label.resize() above is a REQUEST -- Qt silently clamps it to
            # never go below the label's own setMinimumSize(640, 480). When
            # the computed scale would otherwise produce a smaller pixmap
            # (e.g. right after open, before the panel's real viewport size
            # has settled), the label ends up LARGER than the pixmap, and
            # AlignCenter centers the pixmap within it -- leaving a real
            # offset between "label-local click position" and "position
            # within the pixmap" that the scale alone can't account for.
            # Reading the label's actual post-clamp size here keeps mouse
            # coordinate mapping correct in that case instead of just when
            # the pixmap happens to fill the label exactly.
            actual_size = label.size()
            offset_x = max(0, (actual_size.width() - scaled.width()) // 2)
            offset_y = max(0, (actual_size.height() - scaled.height()) // 2)
            self._corrected_pixmap_offset = (offset_x, offset_y)

    def _rerender_current(self) -> None:
        if self._last_raw_frame is not None:
            self._render(self._current_frame_number(), self._last_raw_frame)

    # ---- zoom / speed ----

    def _on_zoom_changed(self, value: int) -> None:
        self._zoom = value / 100.0
        self.zoom_value_label.setText(f"{value}%")
        self._rerender_current()

    def _on_speed_changed(self, value: int) -> None:
        self._playback_speed = value / 100.0
        self.speed_value_label.setText(f"{self._playback_speed:.1f}x")
        if self.play_timer.isActive():
            self.play_timer.setInterval(self._playback_interval_ms())

    def _playback_interval_ms(self) -> int:
        fps = DEFAULT_PLAYBACK_FPS
        if self._metadata is not None:
            fps = (
                self._metadata.get("camera", {}).get("measured_fps")
                or self._metadata.get("camera", {}).get("requested_fps")
                or DEFAULT_PLAYBACK_FPS
            )
        return max(1, int(1000 / fps / self._playback_speed))

    # ---- playback ----

    def _toggle_play(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_pause_button.setText("Play")
            return

        if self._current_frame_number() >= self._total_frames - 1:
            self.seek_to(0)

        self.play_timer.start(self._playback_interval_ms())
        self.play_pause_button.setText("Pause")

    def _on_play_tick(self) -> None:
        self.seek_to(self._current_frame_number() + 1)

    # ---- detection (Stage E) ----

    def _read_corrected_frame(self, frame_number: int):
        """Returns (corrected_bgr, error) -- shared by full-frame and
        per-ROI detection below so a frame is only ever read/undistorted
        ONCE per frame_number, regardless of how many ROIs then run
        their own independent detection against that same array."""
        raw_frame = self._read_frame_at(frame_number)
        if raw_frame is None:
            return None, f"Could not read frame {frame_number}."

        corrected = self._compute_corrected_frame(raw_frame)
        if corrected is None:
            return None, "No usable calibration for this recording -- cannot run detection."

        return corrected, None

    def _run_detection_on_frame(self, frame_number: int) -> tuple[list, str | None]:
        """Returns (fragments, error) for one frame -- error is a
        human-readable string on failure (bad read, no calibration, or a
        ValueError from detection_pipeline.run_detection() for a
        malformed config), never raised past this point."""
        corrected, error = self._read_corrected_frame(frame_number)
        if error is not None:
            return [], error

        try:
            fragments, _display_map = detection_pipeline.run_detection(
                corrected, self._detector_type, self._detection_config, background=self._detection_background
            )
        except ValueError as error:
            return [], str(error)

        return fragments, None

    def _applicable_rois(self, frame_number: int) -> list[dict]:
        return [roi for roi in self._rois if roi["start_frame"] <= frame_number <= roi["end_frame"]]

    def detect_current_frame(self) -> None:
        frame_number = self._current_frame_number()
        corrected, error = self._read_corrected_frame(frame_number)

        if error is not None:
            self.detection_report_text.setPlainText(f"ERROR: {error}")
            return

        try:
            fragments, _display_map = detection_pipeline.run_detection(
                corrected, self._detector_type, self._detection_config, background=self._detection_background
            )
        except ValueError as detect_error:
            self.detection_report_text.setPlainText(f"ERROR: {detect_error}")
            return

        self._detection_results[frame_number] = fragments
        # so the result is actually visible without a separate step -- if
        # ROIs are defined, default to the Per-ROI overlay since that's
        # the more specific result; otherwise fall back to full-frame.
        self.fragment_overlay_combo.setCurrentText(FRAGMENT_OVERLAY_ROI if self._rois else FRAGMENT_OVERLAY_FULL)
        self.detection_report_text.setPlainText(self._format_single_frame_report(frame_number, fragments))

        # ROI (per-pad) preview -- independent detection per ROI (see
        # detection_pipeline.run_roi_detection), in-memory only, never
        # written to disk (current-frame preview is explicitly ephemeral
        # -- see the roi_group's own help text and PERSISTENCE notes).
        self._roi_current_frame_results = {}
        frame_fragments_by_roi: dict[str, list] = {}
        mm_per_pixel = self._metadata.get("calibration", {}).get("mm_per_pixel") if self._metadata else None
        for roi in self._applicable_rois(frame_number):
            roi_result = detection_pipeline.run_roi_detection(
                corrected, roi["rectangle"], self._detector_type, self._detection_config,
                background=self._detection_background,
                min_area_mm2=roi.get("min_area_mm2"), mm_per_pixel=mm_per_pixel,
            )
            if roi_result is None:
                continue
            frame_fragments_by_roi[roi["roi_id"]] = roi_result.fragments
            self._roi_current_frame_results[roi["roi_id"]] = detection_pipeline.summarize_roi_result(
                roi_result, self._detector_type, mm_per_pixel, roi["expected_area_mm2"]
            )
        self._roi_frame_fragments[frame_number] = frame_fragments_by_roi

        if self._rois:
            self.show_rois_check.setChecked(True)
        self.roi_report_text.setPlainText(self._format_roi_preview_report(frame_number))

        self._rerender_current()

    def _format_roi_preview_report(self, frame_number: int) -> str:
        applicable = self._applicable_rois(frame_number)
        if not applicable:
            return "No ROIs apply to this frame." if self._rois else "No ROIs defined for this recording yet."

        names_by_id = {roi["roi_id"]: roi["name"] for roi in self._rois}
        lines = [f"Frame {frame_number} -- PREVIEW ONLY (per-ROI results, not saved to disk):"]
        if self._roi_overlaps:
            lines.append("*** INVALID: overlapping ROIs -- these numbers are NOT valid measurements until resolved. ***")
        lines.append("")

        for roi_id, summary in self._roi_current_frame_results.items():
            name = names_by_id.get(roi_id, roi_id)
            contact_mm2 = (
                f"{summary['detected_contact_area_mm2']:.4f}mm2" if summary["detected_contact_area_mm2"] is not None else "N/A (no scale)"
            )
            coverage = f"{summary['coverage_percent']:.1f}%" if summary["coverage_percent"] is not None else "N/A (no expected area set)"
            strong_fraction = f"{summary['strong_fraction']:.3f}" if summary["strong_fraction"] is not None else "N/A"

            lines.append(f"  {name}:")
            lines.append(
                f"    possible={summary['possible_area_px']}px  probable={summary['probable_area_px']}px  "
                f"strong={summary['strong_area_px']}px   search_region={summary['search_region_area_px']}px (boundary only, not a coverage basis)"
            )
            lines.append(f"    detected_contact_area={contact_mm2}   coverage={coverage}   strong_fraction={strong_fraction}")

            ds = summary["detector_signal"]
            lines.append(f"    detector_signal [{ds['source']}]: mean={ds['mean']}  max={ds['max']}")
            cs = summary["corrected_frame_signal"]
            if cs["source"] == "observed_monochrome_intensity":
                lines.append(f"    corrected_frame_signal [{cs['source']}]: mean={cs['mean']}  max={cs['max']}")
            else:
                lines.append(
                    f"    corrected_frame_signal [{cs['source']}]: "
                    f"B_mean={cs['b']['mean']}  G_mean={cs['g']['mean']}  R_mean={cs['r']['mean']}  HSV_V_mean={cs['hsv_v']['mean']}"
                )
            if summary["clipped"]:
                lines.append("    WARNING: this ROI's rectangle was clipped to the frame bounds")
            lines.append("")

        return "\n".join(lines)

    def _format_single_frame_report(self, frame_number: int, fragments: list) -> str:
        mm_per_pixel = self._metadata.get("calibration", {}).get("mm_per_pixel") if self._metadata else None
        source_type = self._metadata.get("camera", {}).get("source_type") or "unknown" if self._metadata else "unknown"

        detected_regions = [regions.from_fragment(f, source_type, mm_per_pixel) for f in fragments]
        detected_regions.sort(key=lambda r: r.fragment.possible_area_px, reverse=True)

        lines = [f"Frame {frame_number}: {len(detected_regions)} fragment(s) detected."]
        if mm_per_pixel is None:
            lines.append("(No saved physical scale for this recording -- areas shown in pixels only.)")
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
                    f"possible={f.possible_area_px}px  probable={f.probable_area_px}px  strong={f.strong_area_px}px"
                )
            lines.append(f"  Region {f.id} -- centroid=({f.centroid[0]:.1f}, {f.centroid[1]:.1f})")
            lines.append(f"    {area_str}")
            lines.append(
                f"    intensity: mean={f.mean_brightness:.1f}  median={f.median_brightness:.1f}  "
                f"p95={f.p95_brightness:.1f}  max={f.max_brightness:.1f}"
            )

        return "\n".join(lines)

    def detect_range(self) -> None:
        self._run_detection_over(self.range_start_spin.value(), self.range_end_spin.value())

    def detect_full_recording(self) -> None:
        self._run_detection_over(0, self._total_frames - 1)

    def _run_detection_over(self, start: int, end: int) -> None:
        if self._cap is None or self._total_frames == 0:
            return

        start = max(0, min(start, self._total_frames - 1))
        end = max(start, min(end, self._total_frames - 1))
        saved_position = self._current_frame_number()

        # Full-frame detection (Stage E, unchanged) always runs. ROI
        # analysis additionally runs only when there ARE rois and the
        # set is valid -- this is checked again here, not just via the
        # button's enabled state, as a second real gate: an invalid ROI
        # set must never produce a saved analysis run, regardless of how
        # this method gets called.
        do_roi_analysis = bool(self._rois) and not self._roi_overlaps and self.project_id is not None

        total_fragments = 0
        frames_with_fragments = 0
        frames_with_errors = 0
        per_frame_counts: list[tuple[int, int]] = []
        span = end - start + 1

        analysis_run_id = None
        roi_results_writer = None
        roi_row_count = 0
        run_status = "completed"
        run_stop_reason = "completed"
        last_processed_frame = start - 1
        mm_per_pixel = self._metadata.get("calibration", {}).get("mm_per_pixel") if self._metadata else None

        if do_roi_analysis:
            calibration_reference = {
                "recording_id": self.recording_id,
                "snapshot_dir": "calibration_snapshot/",
                "corrected_width": self._undistort_target_size[0] if self._undistort_target_size else None,
                "corrected_height": self._undistort_target_size[1] if self._undistort_target_size else None,
                "crop_info": dict(self._crop),
            }
            analysis_run_id = pp.start_analysis_run(
                self.project_id, self.recording_id, start, end, self._detector_type,
                dict(self._detection_config), calibration_reference, self._rois,
            )
            roi_results_writer = pp.RoiResultsWriter(self.project_id, analysis_run_id)
            roi_results_writer.__enter__()
            self.roi_progress_label.setText(f"Started analysis run {analysis_run_id}...")

        try:
            for frame_number in range(start, end + 1):
                corrected, error = self._read_corrected_frame(frame_number)

                if error is not None:
                    frames_with_errors += 1
                else:
                    try:
                        fragments, _display_map = detection_pipeline.run_detection(
                            corrected, self._detector_type, self._detection_config, background=self._detection_background
                        )
                        self._detection_results[frame_number] = fragments  # cached so scrubbing shows the overlay
                        per_frame_counts.append((frame_number, len(fragments)))
                        total_fragments += len(fragments)
                        if fragments:
                            frames_with_fragments += 1
                    except ValueError:
                        frames_with_errors += 1

                    if do_roi_analysis:
                        recording_time_ns = (
                            self._frame_index_rows[frame_number]["recording_time_ns"]
                            if frame_number < len(self._frame_index_rows)
                            else None
                        )
                        frame_fragments_by_roi: dict[str, list] = {}
                        for roi in self._rois:
                            if not (roi["start_frame"] <= frame_number <= roi["end_frame"]):
                                continue
                            roi_result = detection_pipeline.run_roi_detection(
                                corrected, roi["rectangle"], self._detector_type, self._detection_config,
                                background=self._detection_background,
                                min_area_mm2=roi.get("min_area_mm2"), mm_per_pixel=mm_per_pixel,
                            )
                            if roi_result is None:
                                continue
                            frame_fragments_by_roi[roi["roi_id"]] = roi_result.fragments
                            summary = detection_pipeline.summarize_roi_result(
                                roi_result, self._detector_type, mm_per_pixel, roi["expected_area_mm2"]
                            )
                            roi_results_writer.write_row({
                                "frame_number": frame_number,
                                "recording_time_ns": recording_time_ns,
                                "analysis_run_id": analysis_run_id,
                                "config_hash": pp.config_hash(self._detection_config),
                                "roi_id": roi["roi_id"],
                                **summary,
                            })
                            roi_row_count += 1

                        self._roi_frame_fragments[frame_number] = frame_fragments_by_roi  # cached so scrubbing shows the Per-ROI overlay too

                last_processed_frame = frame_number

                if (frame_number - start) % DETECTION_PROGRESS_UPDATE_EVERY == 0 or frame_number == end:
                    self.detection_progress_label.setText(f"Processing frame {frame_number - start + 1} / {span}...")
                    if do_roi_analysis:
                        self.roi_progress_label.setText(
                            f"ROI analysis: frame {frame_number - start + 1} / {span}, {roi_row_count} row(s) written..."
                        )
                    QApplication.processEvents()

        except Exception as error:
            run_status = "failed"
            run_stop_reason = "error"
            traceback.print_exc()
            self.detection_progress_label.setText(f"Detection stopped early (error): {error}")
            if do_roi_analysis:
                self.roi_progress_label.setText(
                    f"ROI analysis run {analysis_run_id} FAILED and remains marked incomplete "
                    f"(processed frames {start}-{last_processed_frame} of {start}-{end})."
                )
        else:
            self.detection_progress_label.setText(f"Done: frames {start}-{end} ({span} total).")
        finally:
            if roi_results_writer is not None:
                roi_results_writer.__exit__(None, None, None)
            if analysis_run_id is not None:
                pp.finalize_analysis_run(
                    self.project_id, analysis_run_id,
                    completed_start_frame=start, completed_end_frame=last_processed_frame,
                    status=run_status, stop_reason=run_stop_reason, result_row_count=roi_row_count,
                )

        lines = [
            f"Detection over frames {start}-{end} ({span} frames):",
            f"  total fragments across all frames: {total_fragments}",
            f"  frames with at least one fragment: {frames_with_fragments}",
            f"  frames with a detection error: {frames_with_errors}",
            "",
        ]

        per_frame_counts.sort(key=lambda t: t[1], reverse=True)
        if per_frame_counts and per_frame_counts[0][1] > 0:
            lines.append("Frames with the most fragments:")
            for frame_number, count in per_frame_counts[:10]:
                if count == 0:
                    break
                lines.append(f"  frame {frame_number}: {count} fragment(s)")

        self.detection_report_text.setPlainText("\n".join(lines))

        if do_roi_analysis:
            status_word = "COMPLETED" if run_status == "completed" else "FAILED (left marked incomplete)"
            self.roi_report_text.setPlainText(
                f"ROI analysis run {analysis_run_id}: {status_word}\n"
                f"  {roi_row_count} result row(s) written across {len(self._rois)} ROI(s)\n"
                f"  saved to: {pp.roi_results_path(self.project_id, analysis_run_id)}\n"
                f"  run details: {pp.run_json_path(self.project_id, analysis_run_id)}"
            )
        elif self._rois:
            self.roi_report_text.setPlainText(
                "ROI analysis was NOT run: the ROI set is currently invalid (overlapping ROIs) -- "
                "resolve the overlap, then Range/Full detection will include per-pad results again."
                if self._roi_overlaps else
                "ROI analysis was not run."
            )

        # Per-ROI when ROI analysis actually ran this pass (fragments are
        # now cached per-frame right alongside the full-frame ones, so
        # scrubbing anywhere in the processed range shows real per-ROI
        # results) -- otherwise fall back to full-frame, matching
        # detect_current_frame()'s own default logic.
        self.fragment_overlay_combo.setCurrentText(FRAGMENT_OVERLAY_ROI if do_roi_analysis else FRAGMENT_OVERLAY_FULL)
        if do_roi_analysis:
            self.show_rois_check.setChecked(True)

        # Range/full runs are a survey, not a navigation action -- leave
        # the viewer back where it was rather than stranded at the end
        # of whatever range was just scanned. seek_to() re-renders too,
        # which is what actually shows the overlay for saved_position.
        self.seek_to(saved_position)
