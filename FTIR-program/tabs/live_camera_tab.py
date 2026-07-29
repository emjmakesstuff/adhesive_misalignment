"""
Live Camera tab: select a source type (USB camera or VDO.Ninja stream),
connect, and show its live preview plus basic info. Owns the app's
single active source instance (a camera.CameraStream or a
vdo_ninja_source.VdoNinjaSource, exposed via main_window.stream so other
tabs -- Settings, Calibration, Detection -- can read frames/apply
settings through the same instance rather than opening their own).

USB and VDO.Ninja each get their own control group (shown/hidden as a
unit via source_type_combo); switching source types always disconnects
whatever was active first (see _on_source_type_changed) so there is
never more than one stream alive at once. Nothing about the USB
connect/disconnect/undistort/crop logic below changed from before this
was added -- VDO.Ninja is additive, not a rewrite of the working path.

camera.py/settings.py/camera_controls.py are unmodified; VdoNinjaSource
duck-types CameraStream's read()/get_info()/release() (see
vdo_ninja_source.py) so the shared polling/preview code below and
CameraControlsPanel's existing `stream.cap is None` guards both already
do the right thing for either source without a formal interface.
"""

from __future__ import annotations

import threading
import uuid

import cv2
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import calibration_profiles
import camera_controls
import camera_identity
import distortion
import recorder as recorder_module
import vdo_ninja_source
from camera import BACKENDS, CameraStream, FpsMeter, list_device_formats, list_device_names, probe_cameras
from config_store import load_config, save_config
from preview import describe_frame_format, to_display_bgr
from processing import process_frame
from settings import get_exposure_gain_info
from threaded_camera_source import ThreadedCameraSource

USB_POLL_INTERVAL_MS = 15  # real display rate is limited by actual frame arrival, not this
# VDO.Ninja's poll interval is computed in connect_vdo_ninja() from the
# configured Processing FPS instead of a fixed constant -- see the
# comment there for why a fixed interval was actively throttling fps.

# Role -> (width, height, fps, fourcc) DEFAULT requested at connect time
# -- only used to preselect an entry in the Format dropdown below (or as
# a last-resort fallback if the driver's real format list can't be
# enumerated); the actual connect always uses whatever the dropdown has
# selected, so both of these are fully user-adjustable, never hardcoded
# in the sense of "can't be changed". monochrome_ftir matches the
# U20CAM's original hardcoded values exactly (byte-identical default
# behavior preserved). color_ftir defaults to 1280x720/60fps -- the
# ELP-USBFHD01M-BL170's spec sheet lists both 1920x1080@30fps and
# 1280x720@60fps as supported MJPEG modes; 720p60 was confirmed directly
# against the real camera to produce a proper live image (1080p30 also
# works -- the "black" video reported at first was the sensor's default
# Brightness being very low, not a broken capture; cranking Brightness
# in the controls panel fixes either resolution).
ROLE_CONNECT_PARAMS = {
    "monochrome_ftir": (1280, 800, 120, "MJPG"),
    "color_ftir": (1280, 720, 60, "MJPG"),
}


class _NewCameraDialog(QDialog):
    """
    Shown immediately the first time an unrecognized physical USB camera
    connects -- confirmed with the user that this happens automatically,
    not via a separate manual "New Profile" step. Role is constrained to
    the two fixed FTIR roles this app supports; both currently use the
    pinhole calibration model (see ROLE_CONNECT_PARAMS/calibration_tab.py
    -- the ELP's actual working area isn't distorted enough to need the
    fisheye model, per direct user decision).
    """

    def __init__(self, device_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Camera Detected")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f'Detected a camera not seen before: "{device_name}".\n'
            "Name it and choose its role before connecting:"
        ))

        form = QFormLayout()
        self.alias_edit = QLineEdit(device_name)
        form.addRow("Alias:", self.alias_edit)
        layout.addLayout(form)

        self.mono_radio = QRadioButton("Monochrome FTIR Camera")
        self.color_radio = QRadioButton("Color FTIR Camera")
        self.mono_radio.setChecked(True)
        layout.addWidget(self.mono_radio)
        layout.addWidget(self.color_radio)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def result_values(self) -> tuple[str, str]:
        role = "color_ftir" if self.color_radio.isChecked() else "monochrome_ftir"
        return self.alias_edit.text().strip(), role


class LiveCameraTab(QWidget):
    # Emitted from the background probe thread; Qt marshals delivery onto
    # this widget's own (GUI) thread automatically since it's a queued
    # cross-thread connection -- no manual locking needed.
    # formats_by_index (4th param) is declared as `object`, not `dict` --
    # confirmed directly that a concrete `dict` type here fails Shiboken's
    # C++ marshaling for this nested-list-of-dicts payload ("Cannot
    # copy-convert ... (dict) to C++"), even for a same-thread/direct
    # connection. `object` passes the real Python dict through untyped,
    # which works for arbitrary Python payloads like this one.
    probe_finished = Signal(list, list, list, object)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.stream: ThreadedCameraSource | vdo_ninja_source.VdoNinjaSource | None = None
        self.fps_meter = FpsMeter()
        self.consecutive_failures = 0
        self.max_consecutive_failures = 60
        self.format_described = False
        self.undistort_maps = None
        self.undistort_target_size = None
        # Populated by the last refresh_devices() probe -- the same
        # background pass that produced what's currently shown in
        # index_combo, so connect_camera() resolving identity from this
        # list is always consistent with what the user actually picked
        # (rather than a second, separately-timed COM enumeration).
        self._identities: list[camera_identity.CameraIdentity] = []
        # index -> camera.list_device_formats(index) result, gathered in
        # the same background probe pass (each call builds a DirectShow
        # filter graph -- noticeably slower than the identity/friendly-name
        # enumeration, not something to redo on the GUI thread every time
        # the Format dropdown needs repopulating).
        self._formats_by_index: dict[int, list[dict] | None] = {}

        layout = QVBoxLayout(self)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Source:"))
        self.source_type_combo = QComboBox()
        self.source_type_combo.addItem("USB Camera", "usb")
        self.source_type_combo.addItem("VDO.Ninja Stream", "vdo_ninja")
        source_row.addWidget(self.source_type_combo)
        source_row.addStretch(1)
        layout.addLayout(source_row)

        # ---- USB controls (unchanged from before VDO.Ninja existed) ----

        self.usb_group = QWidget()
        usb_layout = QVBoxLayout(self.usb_group)
        usb_layout.setContentsMargins(0, 0, 0, 0)

        conn_row = QHBoxLayout()
        conn_row.addWidget(QLabel("Camera:"))
        self.index_combo = QComboBox()
        conn_row.addWidget(self.index_combo, 1)
        self.refresh_button = QPushButton("Refresh")
        conn_row.addWidget(self.refresh_button)
        self.connect_button = QPushButton("Connect")
        conn_row.addWidget(self.connect_button)
        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.setEnabled(False)
        conn_row.addWidget(self.disconnect_button)
        self.controls_panel_button = QPushButton("Show Controls Panel")
        self.controls_panel_button.setCheckable(True)
        conn_row.addWidget(self.controls_panel_button)
        usb_layout.addLayout(conn_row)

        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.setToolTip(
            "Resolution/pixel format/fps requested at connect time -- "
            "populated from what this camera's driver actually advertises "
            "(the same list AMCap's format dialog would show), not a fixed "
            "guess. Remembered per camera after a successful connect."
        )
        format_row.addWidget(self.format_combo, 1)
        usb_layout.addLayout(format_row)

        layout.addWidget(self.usb_group)

        # ---- VDO.Ninja controls ----

        self.vdo_group = QWidget()
        vdo_layout = QVBoxLayout(self.vdo_group)
        vdo_layout.setContentsMargins(0, 0, 0, 0)

        vdo_url_row = QHBoxLayout()
        vdo_url_row.addWidget(QLabel("Viewer URL:"))
        self.vdo_url_edit = QLineEdit()
        self.vdo_url_edit.setPlaceholderText("https://vdo.ninja/?view=STREAM_ID")
        vdo_url_row.addWidget(self.vdo_url_edit, 1)
        vdo_layout.addLayout(vdo_url_row)

        vdo_conn_row = QHBoxLayout()
        vdo_conn_row.addWidget(QLabel("Processing FPS:"))
        self.vdo_fps_spin = QDoubleSpinBox()
        self.vdo_fps_spin.setRange(1.0, 60.0)
        self.vdo_fps_spin.setDecimals(1)
        self.vdo_fps_spin.setValue(30.0)
        self.vdo_fps_spin.setToolTip(
            "How often frames are pulled from the VDO.Ninja <video> element "
            "-- independent of this tab's own preview refresh rate. This is "
            "a ceiling, not a guarantee: each capture round-trips through "
            "the browser, so the real achieved rate may be lower, "
            "especially at high stream resolutions -- confirmed directly "
            "against a real 1080x1920 stream at ~19-20fps regardless of "
            "whether this is set to 30 or 60."
        )
        vdo_conn_row.addWidget(self.vdo_fps_spin)
        self.vdo_connect_button = QPushButton("Connect")
        vdo_conn_row.addWidget(self.vdo_connect_button)
        self.vdo_disconnect_button = QPushButton("Disconnect")
        self.vdo_disconnect_button.setEnabled(False)
        vdo_conn_row.addWidget(self.vdo_disconnect_button)
        vdo_conn_row.addStretch(1)
        vdo_layout.addLayout(vdo_conn_row)

        self.vdo_status_label = QLabel("Not connected.")
        self.vdo_status_label.setWordWrap(True)
        vdo_layout.addWidget(self.vdo_status_label)

        layout.addWidget(self.vdo_group)
        self.vdo_group.setVisible(False)  # USB is the starting/default source

        # ---- shared controls (apply to whichever source is active) ----

        # Both of these were previously nested inside usb_group by
        # mistake -- undistort/crop are source-agnostic (each profile,
        # USB or VDO.Ninja, has its own lens calibration; _on_undistort_
        # toggled() below already resolves the active profile correctly),
        # so neither belongs hidden away whenever VDO.Ninja is active.
        undistort_row = QHBoxLayout()
        self.undistort_check = QCheckBox("Show Undistorted")
        self.undistort_check.setToolTip(
            "Preview only -- applies Stage 3's lens-distortion correction "
            "to the displayed image so you can see calibration working. "
            "Does not affect the raw frame other tabs see."
        )
        undistort_row.addWidget(self.undistort_check)
        undistort_row.addStretch(1)
        layout.addLayout(undistort_row)

        crop_box = QGroupBox("Crop after undistort (trims the least-trustworthy edge pixels)")
        crop_form = QHBoxLayout(crop_box)
        self.crop_sliders: dict[str, QSlider] = {}
        self.crop_value_labels: dict[str, QLabel] = {}

        for side in ("top", "bottom", "left", "right"):
            side_layout = QVBoxLayout()
            side_layout.addWidget(QLabel(side.capitalize()))

            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 45)  # % of that dimension; see distortion.crop_edges for clamping
            side_layout.addWidget(slider)

            value_label = QLabel("0%")
            value_label.setAlignment(Qt.AlignCenter)
            side_layout.addWidget(value_label)

            self.crop_sliders[side] = slider
            self.crop_value_labels[side] = value_label
            crop_form.addLayout(side_layout)

        layout.addWidget(crop_box)

        # ---- recording (source-agnostic -- works for whichever of USB/VDO.Ninja is active) ----

        recording_box = QGroupBox("Recording")
        recording_layout = QVBoxLayout(recording_box)

        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Quality:"))
        self.quality_mode_combo = QComboBox()
        self.quality_mode_combo.addItem("Lossless (measurement)", "lossless")
        self.quality_mode_combo.addItem("Lossy prototype (framing/setup only)", "lossy_prototype")
        self.quality_mode_combo.setToolTip(
            "Lossless: codec is chosen automatically from this camera's role "
            "(FFV1 for monochrome, HFYU for color) -- see recorder.py. Lossy "
            "prototype (MJPG) is smaller/faster but NOT valid for calibrated "
            "intensity measurement -- every recording made in this mode is "
            "tagged accordingly and should only be used for quick framing/setup runs."
        )
        quality_row.addWidget(self.quality_mode_combo)
        quality_row.addStretch(1)
        recording_layout.addLayout(quality_row)

        self.recording_size_warning_label = QLabel("")
        self.recording_size_warning_label.setWordWrap(True)
        self.recording_size_warning_label.setStyleSheet("color: #b06000;")
        recording_layout.addWidget(self.recording_size_warning_label)

        record_button_row = QHBoxLayout()
        self.start_recording_button = QPushButton("Start Recording")
        self.start_recording_button.setEnabled(False)
        record_button_row.addWidget(self.start_recording_button)
        self.stop_recording_button = QPushButton("Stop Recording")
        self.stop_recording_button.setEnabled(False)
        record_button_row.addWidget(self.stop_recording_button)
        record_button_row.addStretch(1)
        recording_layout.addLayout(record_button_row)

        self.recording_status_label = QLabel("Not recording.")
        self.recording_status_label.setWordWrap(True)
        recording_layout.addWidget(self.recording_status_label)

        layout.addWidget(recording_box)

        # ---- shared preview (both sources render into these) ----

        self.info_label = QLabel("Not connected.")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.video_label = QLabel("No preview yet.")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(self.video_label, 1)

        self.source_type_combo.currentIndexChanged.connect(self._on_source_type_changed)
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.index_combo.currentIndexChanged.connect(self._on_index_combo_changed)
        self.connect_button.clicked.connect(self.connect_camera)
        # NOT `.connect(self.disconnect_camera)` directly: QPushButton.clicked
        # emits clicked(checked: bool), and PySide adapts the call to however
        # many arguments the connected callable accepts -- since
        # disconnect_camera()'s first (and only) parameter is
        # recorder_stop_reason, a direct connection silently passes the
        # button's checked state (False) into it instead of using the real
        # default "camera_disconnected". Confirmed directly: this exact bug
        # was live and had already corrupted a real recording's stop_reason
        # (and, worse, its completed_normally, since `False not in (...)` is
        # True) before being caught here. Same reasoning for the two
        # connections below.
        self.disconnect_button.clicked.connect(lambda: self.disconnect_camera())
        self.vdo_connect_button.clicked.connect(self.connect_vdo_ninja)
        self.vdo_disconnect_button.clicked.connect(lambda: self.disconnect_vdo_ninja())
        self.controls_panel_button.clicked.connect(self._toggle_controls_panel)
        self.undistort_check.toggled.connect(self._on_undistort_toggled)
        self.probe_finished.connect(self._on_probe_finished)
        self.quality_mode_combo.currentIndexChanged.connect(self._update_recording_size_warning)
        self.start_recording_button.clicked.connect(self.start_recording)
        self.stop_recording_button.clicked.connect(lambda: self.stop_recording())

        for side, slider in self.crop_sliders.items():
            slider.valueChanged.connect(lambda value, s=side: self._on_crop_value_changed(s, value))
            slider.sliderReleased.connect(self._save_crop_settings)

        self.reload_crop_from_profile()
        self._load_undistort_preference()
        self._update_recording_size_warning()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)

        # Separate from the frame-display timer above -- only runs while
        # actively recording, at a much coarser interval (this just
        # refreshes a status label, not the preview).
        self.recording_status_timer = QTimer(self)
        self.recording_status_timer.timeout.connect(self._refresh_recording_status)

        self.refresh_devices()

    # ---- source switching ----

    def _on_source_type_changed(self) -> None:
        new_source_type = self.source_type_combo.currentData()
        old_source_type = self.main_window.active_source_type

        # Never run two streams: whatever was connected under the OLD
        # source type is fully stopped before the new one's controls even
        # become active, regardless of which direction the switch goes.
        if self.stream is not None:
            if old_source_type == "usb":
                self.disconnect_camera()
            else:
                self.disconnect_vdo_ninja()

        self.main_window.active_source_type = new_source_type
        # "vdo_ninja" is its own plain key (matches active_source_type
        # exactly, see calibration_profiles.profile_key()); "usb" doesn't
        # resolve to a specific physical camera's key until Connect
        # actually identifies one, so it's left unresolved (None) here --
        # get_active_profile(None) safely returns None, same as "no
        # profile yet", which every caller below already handles.
        self.main_window.active_profile_key = "vdo_ninja" if new_source_type == "vdo_ninja" else None
        self.usb_group.setVisible(new_source_type == "usb")
        self.vdo_group.setVisible(new_source_type == "vdo_ninja")

        self.info_label.setText("Not connected.")
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("No preview yet.")

        self.reload_crop_from_profile()

    def _on_index_combo_changed(self) -> None:
        """
        Selecting a different physical USB camera while one is connected
        must disconnect first -- the same "never run two streams, release
        before switching" guarantee _on_source_type_changed already gives
        USB<->VDO.Ninja, applied one level down within "usb" now that the
        dropdown can hold more than one physical camera. _on_probe_finished
        blocks this signal while rebuilding the list, so this only fires
        for a genuine user selection change, never a refresh-triggered one.
        """
        if self.stream is not None and self.main_window.active_source_type == "usb":
            self.disconnect_camera()

        self._refresh_format_combo()

    # ---- crop persistence (per active profile, not a flat global) ----

    def reload_crop_from_profile(self) -> None:
        """
        Called on init, on source-type switch, and by CalibrationTab
        whenever the active profile changes (main_window.
        on_active_profile_changed) -- so the crop sliders always reflect
        whichever profile is actually active, not a stale flat setting
        shared across every source/profile.
        """
        profile = calibration_profiles.get_active_profile(self.main_window.active_profile_key)
        crop = profile["crop_percentages"] if profile is not None else {}

        for side in ("top", "bottom", "left", "right"):
            value = int(crop.get(side, 0))
            self.crop_sliders[side].blockSignals(True)
            self.crop_sliders[side].setValue(value)
            self.crop_value_labels[side].setText(f"{value}%")
            self.crop_sliders[side].blockSignals(False)
            # blockSignals() above means _on_crop_value_changed won't fire
            # to do this itself, so it's set here too.
            self.main_window.crop_percentages[side] = value

    def _save_crop_settings(self) -> None:
        profile = calibration_profiles.get_active_profile(self.main_window.active_profile_key)

        if profile is None:
            return  # no profile yet for this source -- nothing to persist into

        crop = {side: self.crop_sliders[side].value() for side in ("top", "bottom", "left", "right")}
        calibration_profiles.update_profile_crop(profile["id"], crop)

    def _on_crop_value_changed(self, side: str, value: int) -> None:
        self.crop_value_labels[side].setText(f"{value}%")
        # Shared so Calibration's scale-reference capture can apply the
        # same crop the live preview is currently showing.
        self.main_window.crop_percentages[side] = value

    # ---- USB (unchanged behavior) ----

    def _load_undistort_preference(self) -> None:
        config = load_config()

        if config.get("show_undistorted", False):
            # setChecked(True) here is a real state change (default is
            # unchecked), so it fires the already-connected toggled signal
            # -- reuses _on_undistort_toggled's own calibration-loading and
            # missing-calibration error handling rather than duplicating it.
            self.undistort_check.setChecked(True)

    def refresh_devices(self) -> None:
        self.index_combo.clear()
        self.index_combo.addItem("Probing...")
        self.index_combo.setEnabled(False)
        self.refresh_button.setEnabled(False)

        threading.Thread(target=self._probe_worker, daemon=True).start()

    def _probe_worker(self) -> None:
        results = probe_cameras(max_index=5, backend=BACKENDS["dshow"], timeout_seconds=3.0)
        names = list_device_names(max_index=len(results)) or []
        identities = camera_identity.list_camera_identities()
        formats_by_index = {r.index: list_device_formats(r.index) for r in results if r.can_read}
        self.probe_finished.emit(results, names, identities, formats_by_index)

    def _on_probe_finished(self, results, names, identities, formats_by_index) -> None:
        self._identities = identities
        self._formats_by_index = formats_by_index
        identities_by_index = {i.index: i for i in identities}
        usb_profiles = calibration_profiles.list_profiles(source_type="usb")

        # blockSignals: this rebuild fires currentIndexChanged just like a
        # real user selection would -- without this, _on_index_combo_changed
        # would spuriously disconnect a live camera every time the device
        # list is refreshed, not just when the user actually picks a
        # different one.
        self.index_combo.blockSignals(True)
        self.index_combo.clear()
        workable = [r for r in results if r.can_read]

        if not workable:
            self.index_combo.addItem("No camera found", None)
        else:
            for r in workable:
                name = names[r.index] if r.index < len(names) else "?"
                identity = identities_by_index.get(r.index)
                profile = self._find_profile_for_identity(identity, usb_profiles) if identity else None

                if profile is not None:
                    label = f"{r.index}: {profile['alias']} ({r.width}x{r.height})"
                else:
                    label = f"{r.index}: {name} ({r.width}x{r.height}) -- not yet set up"

                self.index_combo.addItem(label, r.index)

        self.index_combo.blockSignals(False)
        self.index_combo.setEnabled(True)
        self.refresh_button.setEnabled(True)

        self._refresh_format_combo()

    @staticmethod
    def _find_profile_for_identity(identity: camera_identity.CameraIdentity, usb_profiles: list[dict]) -> dict | None:
        if identity.device_path:
            for p in usb_profiles:
                if p.get("device_path") == identity.device_path:
                    return p

        for p in usb_profiles:
            if p.get("device_path") is None and p.get("device_name") == identity.name:
                return p

        return None

    def _refresh_format_combo(self) -> None:
        """
        Populates the Format dropdown from what the selected camera's
        driver actually advertises (camera.list_device_formats(), the
        same enumeration AMCap's own format dialog uses) rather than a
        fixed guess -- so resolution/pixel-format/fps are genuinely
        adjustable, not hardcoded per role. Preselects, in order: this
        camera's own previously-used format (persisted on its profile via
        update_profile_connect_format), else its role's ROLE_CONNECT_PARAMS
        default, else just the first (highest-resolution) entry.
        """
        index = self.index_combo.currentData()
        self.format_combo.blockSignals(True)
        self.format_combo.clear()

        identity = next((i for i in self._identities if i.index == index), None) if index is not None else None
        profile = (
            self._find_profile_for_identity(identity, calibration_profiles.list_profiles(source_type="usb"))
            if identity is not None
            else None
        )
        role_default = ROLE_CONNECT_PARAMS.get(
            profile.get("camera_role") if profile is not None else None, ROLE_CONNECT_PARAMS["monochrome_ftir"]
        )
        preferred = role_default
        if profile is not None and profile.get("connect_width") and profile.get("connect_height") and profile.get("fourcc"):
            preferred = (
                profile["connect_width"],
                profile["connect_height"],
                profile.get("connect_fps") or role_default[2],
                profile["fourcc"],
            )

        formats = self._formats_by_index.get(index) if index is not None else None

        if not formats:
            # Enumeration unavailable (pygrabber issue, or this index
            # wasn't in the last probe) -- still connectable, just not
            # adjustable beyond the one fallback entry.
            w, h, fps, fourcc = preferred
            self.format_combo.addItem(f"{w}x{h} {fourcc} ~{fps:.0f}fps (driver format list unavailable)", (w, h, fps, fourcc))
            self.format_combo.setEnabled(False)
            self.format_combo.blockSignals(False)
            return

        self.format_combo.setEnabled(True)

        # De-duplicate (a driver can list the same width/height/fourcc
        # more than once across near-identical frame intervals) keeping
        # the fastest fps seen for each -- min_framerate is, per
        # camera.list_device_formats()'s own docstring, the FASTEST
        # achievable fps for that entry despite the name.
        best_fps_by_key: dict[tuple[int, int, str], float] = {}
        for f in formats:
            key = (f["width"], f["height"], f["media_type_str"])
            fps = round(f["min_framerate"], 1)
            if key not in best_fps_by_key or fps > best_fps_by_key[key]:
                best_fps_by_key[key] = fps

        entries = sorted(best_fps_by_key.items(), key=lambda kv: (-(kv[0][0] * kv[0][1]), kv[0][2]))

        selected_index = 0
        for i, ((w, h, fourcc), fps) in enumerate(entries):
            self.format_combo.addItem(f"{w}x{h} {fourcc} (up to {fps:.0f}fps)", (w, h, fps, fourcc))
            if (w, h, fourcc) == (preferred[0], preferred[1], preferred[3]):
                selected_index = i

        self.format_combo.setCurrentIndex(selected_index)
        self.format_combo.blockSignals(False)

    def _prompt_new_camera(self, device_name: str) -> tuple[str, str] | None:
        dialog = _NewCameraDialog(device_name, parent=self)

        if dialog.exec() != QDialog.Accepted:
            return None

        alias, role = dialog.result_values()
        return (alias or device_name), role

    def connect_camera(self) -> None:
        index = self.index_combo.currentData()

        if index is None:
            self.info_label.setText("No camera selected.")
            return

        identity = next((i for i in self._identities if i.index == index), None)
        device_path = identity.device_path if identity is not None else None
        device_name = identity.name if identity is not None else None

        profile = None
        if device_path:
            profile = calibration_profiles.find_profile_by_device_path(device_path)
        if profile is None and device_name:
            profile = calibration_profiles.find_profile_by_name(device_name)

        if profile is None:
            # Unrecognized physical camera -- resolved BEFORE stream.open()
            # so the very first connection already uses the right
            # role-driven parameters, rather than connecting once with
            # generic defaults and reconnecting.
            prompted = self._prompt_new_camera(device_name or f"Camera {index}")

            if prompted is None:
                return  # cancelled -- never silently create/connect an unnamed profile

            alias, role = prompted
            detector_type = "grayscale" if role == "monochrome_ftir" else "color"
            profile_id = calibration_profiles.create_profile(
                "usb",
                device_name or f"Camera {index}",
                alias=alias,
                camera_role=role,
                device_path=device_path,
                device_name=device_name,
                fourcc="MJPG",
                calibration_model="pinhole",
                detector_type=detector_type,
            )
            profile = calibration_profiles.get_profile(profile_id)
        elif device_path and profile.get("device_path") != device_path:
            # Self-heal: this profile was previously matched by name only
            # (migrated before device_path existed, or a driver that took
            # a moment to expose one) -- now that a real device_path is
            # known, record it so future lookups use the more reliable
            # match instead of the name fallback.
            calibration_profiles.update_profile_device_path(profile["id"], device_path)
            profile = calibration_profiles.get_profile(profile["id"])

        # The Format dropdown -- not a fixed role lookup -- is the actual
        # source of truth for what gets requested: resolution/pixel
        # format/fps are meant to be fully user-adjustable (see
        # _refresh_format_combo), with ROLE_CONNECT_PARAMS only used to
        # PRESELECT a sensible entry in that dropdown. A freshly-created
        # profile hasn't had _refresh_format_combo() re-run against its
        # now-known role yet, so its dropdown selection above may still
        # reflect the pre-role-choice default -- harmless (the user can
        # always pick a different entry before clicking Connect again),
        # and rare in practice since a new camera's default role match is
        # only ever off by resolution/fps, never a wrong device.
        role = profile.get("camera_role") or "monochrome_ftir"
        format_data = self.format_combo.currentData()
        if format_data is not None:
            width, height, fps, fourcc = format_data
        else:
            width, height, fps, fourcc = ROLE_CONNECT_PARAMS.get(role, ROLE_CONNECT_PARAMS["monochrome_ftir"])

        stream = CameraStream(
            index=index,
            backend=BACKENDS["dshow"],
            requested_width=width,
            requested_height=height,
            requested_fps=fps,
            requested_fourcc=fourcc,
        )

        try:
            stream.open()
        except RuntimeError as error:
            self.info_label.setText(f"ERROR: {error}")
            return

        # Captured immediately, before anything else touches these
        # properties, so "Reset to Defaults" in the controls panel can
        # restore this camera's real as-connected/power-on state --
        # more accurate than a hardcoded guess, and doesn't depend on
        # OpenCV exposing real hardware default values (it doesn't).
        self.main_window.default_controls = camera_controls.get_all(stream.cap)
        self.main_window.default_exposure_info = get_exposure_gain_info(stream.cap)

        # Real acquisition moves off the GUI thread here: ThreadedCameraSource
        # spins a dedicated background thread that calls stream.read() in a
        # tight loop (paced only by cv2.VideoCapture's own blocking behavior,
        # not this tab's 15ms timer), publishing every frame through
        # main_window.frame_dispatcher. _update_frame() below is unchanged --
        # it still just calls self.stream.read(), which now returns whatever
        # the background thread last captured instead of driving capture
        # itself.
        source_key = calibration_profiles.profile_key(profile)
        threaded_stream = ThreadedCameraSource(
            stream,
            self.main_window.frame_dispatcher,
            source_key=source_key,
            source_session_id=uuid.uuid4().hex,
        )
        threaded_stream.start()

        self.stream = threaded_stream
        self.main_window.stream = threaded_stream
        self.format_described = False
        self.consecutive_failures = 0
        self.fps_meter = FpsMeter()

        self.main_window.active_profile_key = source_key
        calibration_profiles.set_active_profile(self.main_window.active_profile_key, profile["id"])
        # Remembers the format actually requested (matching what's shown
        # selected in the dropdown) so reconnecting later reselects the
        # same entry instead of falling back to the role default.
        calibration_profiles.update_profile_connect_format(profile["id"], width, height, fps, fourcc)
        self.reload_crop_from_profile()

        self._refresh_info_label()
        self.main_window.on_camera_connected()

        self.connect_button.setEnabled(False)
        self.disconnect_button.setEnabled(True)
        self.index_combo.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.format_combo.setEnabled(False)
        self.start_recording_button.setEnabled(True)
        self._update_recording_size_warning()

        self.timer.start(USB_POLL_INTERVAL_MS)

    def disconnect_camera(self, recorder_stop_reason: str = "camera_disconnected") -> None:
        self.timer.stop()

        if self.main_window.recorder.is_recording:
            self.stop_recording(stop_reason=recorder_stop_reason)

        if self.stream is not None:
            # ThreadedCameraSource.release() blocks until its capture
            # thread has actually exited (not merely signaled) before
            # returning -- see threaded_camera_source.py -- so by the time
            # this call returns, a subsequent connect_camera() cannot race
            # with any frame this source's worker might still be publishing.
            self.stream.release()
            self.stream = None
            self.main_window.stream = None
            # A stale frame from the just-released source must never be
            # mistaken for a newly-connected one's.
            self.main_window.latest_frame = None

        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
        self.index_combo.setEnabled(True)
        self.refresh_button.setEnabled(True)
        # Re-enables only if the driver's format list was actually
        # available for this camera -- _refresh_format_combo() itself
        # disables it again when falling back to the single "unavailable"
        # entry, so this doesn't need to duplicate that condition.
        self.format_combo.setEnabled(bool(self._formats_by_index.get(self.index_combo.currentData())))
        self.start_recording_button.setEnabled(False)
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("No preview yet.")
        self.info_label.setText("Not connected.")

    # ---- VDO.Ninja ----

    def connect_vdo_ninja(self) -> None:
        url = self.vdo_url_edit.text().strip()

        if not url:
            self.vdo_status_label.setText("Enter a VDO.Ninja viewer URL first.")
            return

        source = vdo_ninja_source.VdoNinjaSource()
        source.open(
            url,
            requested_fps=self.vdo_fps_spin.value(),
            dispatcher=self.main_window.frame_dispatcher,
            source_key="vdo_ninja",
            source_session_id=uuid.uuid4().hex,
        )

        self.stream = source
        self.main_window.stream = source
        self.format_described = False
        self.consecutive_failures = 0
        self.fps_meter = FpsMeter()

        self.main_window.active_profile_key = "vdo_ninja"

        self.vdo_connect_button.setEnabled(False)
        self.vdo_disconnect_button.setEnabled(True)
        self.vdo_url_edit.setEnabled(False)
        self.vdo_fps_spin.setEnabled(False)

        # Harmless for VDO.Ninja: CameraControlsPanel's refresh checks
        # `stream.cap is None` first (True here, see vdo_ninja_source.py)
        # and reports "not connected" for exposure/gain rather than
        # touching anything -- exactly the "hide/disable USB-only
        # controls" behavior requirement 4 asks for, with no changes
        # needed in that already-working file.
        self.main_window.on_camera_connected()

        # Tied to the configured Processing FPS, not a fixed constant --
        # a previous fixed 80ms (12.5Hz) poll interval was silently
        # throttling the display to ~12.5fps even after the capture
        # loop itself was fixed to achieve ~19-20fps: read() only ever
        # returns the LATEST frame (not a queue), so polling slower than
        # the real capture rate means real, distinct frames get skipped
        # before the GUI ever sees them, not just delayed. floor of 10ms
        # keeps this in the same ballpark as USB's own 15ms poll even if
        # Processing FPS is set very high.
        poll_interval_ms = max(10, int(1000 / self.vdo_fps_spin.value()))
        self.timer.start(poll_interval_ms)

        self.start_recording_button.setEnabled(True)
        self._update_recording_size_warning()

    def disconnect_vdo_ninja(self, recorder_stop_reason: str = "camera_disconnected") -> None:
        self.timer.stop()

        if self.main_window.recorder.is_recording:
            self.stop_recording(stop_reason=recorder_stop_reason)

        if self.stream is not None:
            self.stream.release()
            self.stream = None
            self.main_window.stream = None
            self.main_window.latest_frame = None

        self.vdo_connect_button.setEnabled(True)
        self.vdo_disconnect_button.setEnabled(False)
        self.vdo_url_edit.setEnabled(True)
        self.vdo_fps_spin.setEnabled(True)
        self.start_recording_button.setEnabled(False)
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("No preview yet.")
        self.info_label.setText("Not connected.")
        self.vdo_status_label.setText("Not connected.")

    # ---- recording ----

    def _update_recording_size_warning(self) -> None:
        """Refreshed on connect and whenever the quality-mode combo
        changes -- shows the real-hardware-measured GB/minute for
        whichever camera role is currently active (see recorder.py's
        FALLBACK_BYTES_PER_SECOND_BY_ROLE), so the ELP's much larger
        footprint is never a silent surprise."""
        profile = calibration_profiles.get_active_profile(self.main_window.active_profile_key)
        camera_role = profile.get("camera_role") if profile is not None else None
        quality_mode = self.quality_mode_combo.currentData()
        gb_per_min = recorder_module.estimated_gb_per_minute(camera_role, quality_mode)

        if quality_mode == "lossy_prototype":
            self.recording_size_warning_label.setText(
                f"Lossy prototype mode (MJPG, ~{gb_per_min:.2f} GB/min) -- NOT valid for calibrated "
                f"intensity measurement. Use only for quick framing/setup."
            )
        elif camera_role in recorder_module.CODEC_BY_ROLE:
            codec = recorder_module.CODEC_BY_ROLE[camera_role]
            self.recording_size_warning_label.setText(
                f"Lossless recording for this camera uses {codec['fourcc']}, ~{gb_per_min:.2f} GB/minute."
            )
        else:
            self.recording_size_warning_label.setText(
                "No validated lossless codec for this camera/role -- switch to Lossy prototype mode to record."
            )

    def _build_camera_info(self) -> dict:
        """The metadata schema's "camera" block -- see recording_store.py.
        opencv_index_at_session is only meaningful for USB (VdoNinjaSource
        has no .index at all, unlike ThreadedCameraSource); exposure/gain/
        other_controls only exist for a real cv2.VideoCapture."""
        info = self.stream.get_info()
        profile = calibration_profiles.get_active_profile(self.main_window.active_profile_key)
        source_type = self.main_window.active_source_type

        camera_info = {
            "device_path": profile.get("device_path") if profile is not None else None,
            "alias": profile.get("alias") if profile is not None else None,
            "camera_role": profile.get("camera_role") if profile is not None else None,
            "source_type": source_type,
            "opencv_index_at_session": self.stream.index if source_type == "usb" else None,
            "requested_width": info.get("requested_width"),
            "requested_height": info.get("requested_height"),
            "requested_fourcc": info.get("requested_fourcc"),
            "actual_width": info.get("actual_width"),
            "actual_height": info.get("actual_height"),
            "requested_fps": info.get("requested_fps"),
            "measured_fps": self.fps_meter.fps,
            "exposure_ms": None,
            "gain": None,
            "other_controls": {},
        }

        if source_type == "usb" and self.stream.cap is not None:
            with self.stream.lock:
                controls = camera_controls.get_all(self.stream.cap)
                exposure_info = get_exposure_gain_info(self.stream.cap)
            camera_info["exposure_ms"] = exposure_info["exposure_ms"]
            camera_info["gain"] = controls["gain"]
            camera_info["other_controls"] = controls

        return camera_info

    def start_recording(self) -> None:
        if self.stream is None:
            self.recording_status_label.setText("Not connected -- connect a camera first.")
            return

        profile = calibration_profiles.get_active_profile(self.main_window.active_profile_key)
        if profile is None:
            self.recording_status_label.setText("No active calibration profile -- set one up on the Calibration tab first.")
            return

        quality_mode = self.quality_mode_combo.currentData()

        try:
            recorder_module.codec_for(profile.get("camera_role"), quality_mode)
        except ValueError as error:
            QMessageBox.warning(self, "Cannot start recording", str(error))
            return

        headroom = self.main_window.recorder.check_headroom(profile.get("camera_role"), quality_mode)
        if not headroom["ok"]:
            proceed = QMessageBox.warning(
                self,
                "Low disk space",
                f"Only about {headroom['estimated_minutes']:.1f} minute(s) of recording headroom remain "
                f"at the expected ~{headroom['gb_per_minute']:.2f} GB/min for this camera/quality mode.\n\n"
                f"Start recording anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if proceed != QMessageBox.Yes:
                return

        camera_info = self._build_camera_info()

        try:
            recording_id = self.main_window.recorder.start(
                profile, camera_info, quality_mode=quality_mode, source_session_id=self.stream.session_id
            )
        except RuntimeError as error:
            QMessageBox.critical(self, "Recording failed to start", str(error))
            return

        self.start_recording_button.setEnabled(False)
        self.stop_recording_button.setEnabled(True)
        self.quality_mode_combo.setEnabled(False)
        self.recording_status_label.setText(f"Recording {recording_id}...")
        self.recording_status_timer.start(500)

    def stop_recording(self, stop_reason: str = "user_stop_button") -> None:
        if not self.main_window.recorder.is_recording:
            return

        self.recording_status_timer.stop()
        result = self.main_window.recorder.stop(stop_reason=stop_reason)

        self.start_recording_button.setEnabled(self.stream is not None)
        self.stop_recording_button.setEnabled(False)
        self.quality_mode_combo.setEnabled(True)

        self.recording_status_label.setText(
            f"Recording {result['recording_id']} finished ({stop_reason}): "
            f"{result['output_frames_verified']} frames verified, "
            f"{result['recorder_queue_drops']} queue drop(s), "
            f"{len(result['suspected_driver_gaps'])} suspected timing gap(s)."
        )

    def toggle_recording_hotkey(self) -> None:
        """Ctrl+R, registered app-wide in gui_app.py -- works regardless
        of which tab currently has focus."""
        if self.main_window.recorder.is_recording:
            self.stop_recording(stop_reason="hotkey")
        else:
            self.start_recording()

    def _refresh_recording_status(self) -> None:
        status = self.main_window.recorder.status()
        if status is None:
            return

        disk = self.main_window.recorder.check_disk_space()
        size_mb = status["bytes_written"] / 1e6

        text = (
            f"Recording {status['recording_id']}  elapsed={status['elapsed_s']:.0f}s  "
            f"acquired={status['frames_acquired_during_recording']}  "
            f"written={status['frames_write_attempted']}  "
            f"drops={status['recorder_queue_drops']}  queue={status['queue_depth']}  "
            f"size={size_mb:.1f}MB"
        )
        if disk is not None:
            text += f"  ~{disk['estimated_seconds_remaining'] / 60:.1f} min disk remaining"
        self.recording_status_label.setText(text)

    def _toggle_controls_panel(self) -> None:
        visible = self.main_window.toggle_controls_dock()
        self.controls_panel_button.setChecked(visible)

    def set_controls_button_checked(self, checked: bool) -> None:
        """Called by MainWindow when the dock is shown/hidden/closed by
        means other than this button (e.g. the dock's own close box),
        so the button's checked state stays in sync."""
        self.controls_panel_button.setChecked(checked)

    def _on_undistort_toggled(self, checked: bool) -> None:
        # Remembered across restarts so a saved calibration is applied
        # automatically next launch instead of requiring a manual re-check
        # every time, same idea as _save_crop_settings below.
        config = load_config()
        config["show_undistorted"] = checked
        save_config(config)

        if not checked:
            self.undistort_maps = None
            self.undistort_target_size = None
            return

        profile = calibration_profiles.get_active_profile(self.main_window.active_profile_key)
        calibration = None if profile is None else distortion.load_calibration(
            path=calibration_profiles.distortion_path(profile["id"])
        )

        if calibration is None:
            self.info_label.setText(
                "No calibration data found for the active profile -- run a "
                "calibration on the Calibration tab first."
            )
            self.undistort_check.setChecked(False)
            return

        # cv2.remap does NOT error on a map/frame size mismatch (confirmed
        # directly) -- it silently samples into whatever frame it's given
        # using the maps' own dimensions, producing a wrong/cropped result
        # rather than failing loudly. So the size has to be checked
        # explicitly, here and again per-frame in _update_frame, rather
        # than relying on an exception that doesn't actually occur.
        self.undistort_maps = distortion.build_undistort_maps(calibration)
        self.undistort_target_size = tuple(calibration["image_size"])

    def _refresh_info_label(self) -> None:
        if self.stream is None:
            return

        info = self.stream.get_info()
        text = (
            f"Backend: {info['backend_name']}   "
            f"Resolution: {info['actual_width']}x{info['actual_height']}   "
            f"FourCC: {info['fourcc']}   "
            f"Measured FPS: {self.fps_meter.fps:.1f}"
        )
        self.info_label.setText(text)

        # VDO.Ninja's dict carries an extra "status" key CameraStream's
        # doesn't (see vdo_ninja_source.py) -- mirrored into its own
        # dedicated status label rather than assuming every source
        # provides it.
        if "status" in info:
            self.vdo_status_label.setText(
                f"Status: {info['status']}   Native resolution: {info['actual_width']}x{info['actual_height']}"
            )

    def _update_frame(self) -> None:
        if self.stream is None:
            return

        frame = self.stream.read()

        if frame is None:
            if self.main_window.active_source_type == "vdo_ninja":
                # VdoNinjaSource legitimately returns None for as long as
                # it's still connecting -- page load, waiting for the
                # <video> element, waiting for the WebRTC handshake to
                # actually deliver frames. That can easily take several
                # seconds (much longer than USB's near-instant frame
                # delivery once connected), and VdoNinjaSource already
                # enforces its own generous internal timeouts (30s/30s/
                # 60s) before it ever reports an "error: ..." status. The
                # USB consecutive-failure counter below was being applied
                # here too -- at VDO_POLL_INTERVAL_MS=80 that's only 4.8s
                # before auto-disconnecting, so a real connection was
                # getting torn down mid-handshake before VDO.Ninja's own
                # WebRTC negotiation could ever finish (confirmed as the
                # cause of "connects for a moment, then disconnects").
                # Only ever auto-disconnect here on VdoNinjaSource's own
                # explicit error status, never on a bare None read.
                info = self.stream.get_info()
                if info["status"].startswith("error"):
                    self.info_label.setText(f"ERROR: {info['status']}. Disconnecting.")
                    self.disconnect_vdo_ninja(recorder_stop_reason="error")
                return

            self.consecutive_failures += 1

            if self.consecutive_failures >= self.max_consecutive_failures:
                self.info_label.setText(
                    "ERROR: camera stopped delivering frames. Disconnecting."
                )
                self.disconnect_camera(recorder_stop_reason="error")

            return

        self.consecutive_failures = 0
        self.fps_meter.tick()

        if not self.format_described:
            self.format_described = True
            print(f"Frame format: {describe_frame_format(frame)}")

        # Shared with other tabs (Calibration, Detection) so they can
        # look at live frames without a second, competing read() call.
        # Always the raw frame -- undistortion below is display-only and
        # must never feed back into this.
        self.main_window.latest_frame = frame

        processed = process_frame(frame)
        display = to_display_bgr(processed)

        if self.undistort_check.isChecked() and self.undistort_maps is not None:
            frame_size = (display.shape[1], display.shape[0])

            if frame_size != self.undistort_target_size:
                # info_label gets overwritten by _refresh_info_label() on
                # the very next tick, so this would otherwise flash and
                # vanish before anyone could read it -- printed too so
                # it's not silently lost.
                message = (
                    f"Undistort disabled: live frame is "
                    f"{frame_size[0]}x{frame_size[1]} but the saved "
                    f"calibration was done at "
                    f"{self.undistort_target_size[0]}x{self.undistort_target_size[1]}. "
                    f"Recalibrate at the current resolution to use this."
                )
                print(message)
                self.info_label.setText(message)
                self.undistort_check.setChecked(False)
                self.undistort_maps = None
                self.undistort_target_size = None
            else:
                display = distortion.undistort_with_maps(display, self.undistort_maps)
                display = distortion.crop_edges(
                    display,
                    top_pct=self.crop_sliders["top"].value(),
                    bottom_pct=self.crop_sliders["bottom"].value(),
                    left_pct=self.crop_sliders["left"].value(),
                    right_pct=self.crop_sliders["right"].value(),
                )

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, channels = rgb.shape
        qimage = QImage(rgb.data, w, h, channels * w, QImage.Format_RGB888)
        # .copy(): the underlying numpy buffer isn't kept alive by QImage
        # on its own once `rgb` goes out of scope at the end of this call.
        pixmap = QPixmap.fromImage(qimage.copy())

        scaled = pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

        self._refresh_info_label()

    def stop(self) -> None:
        """Called once by MainWindow.closeEvent on app shutdown, AFTER it
        has already stopped any active recording (stop_reason="app_close")
        -- this just tears down the frame timer/status timer/stream."""
        self.timer.stop()
        self.recording_status_timer.stop()

        if self.stream is not None:
            self.stream.release()
            self.stream = None
