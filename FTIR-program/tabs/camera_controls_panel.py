"""
Reusable camera-controls widget: every slider from the native "Video Proc
Amp" dialog (Brightness, Contrast, Hue, Saturation, Sharpness, Gamma,
White Balance + Auto, Backlight Compensation, Gain) via camera_controls.py,
plus Exposure + Auto Exposure via settings.py's existing, unmodified
functions, plus Load/Save and the native dialog button.

Instantiated twice: once embedded in the Settings tab, once inside a
dockable panel next to Live Camera (see gui_app.py) -- both instances
talk to the same main_window.stream independently, so either can be used
to tweak the live camera while watching the preview.

ColorEnable and PowerLine Frequency from the native dialog are not
included: ColorEnable is greyed out/inapplicable on this monochrome
sensor, and PowerLine Frequency has no cv2 property in this OpenCV build
(only reachable via the COM route that proved unreliable here -- see
camera_controls.py's docstring).
"""

from __future__ import annotations

import threading

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import camera_controls
from config_store import load_config, save_config
from settings import (
    get_exposure_gain_info,
    open_properties_dialog,
    set_auto_exposure,
    set_manual_exposure,
)

_LABELS = {
    "brightness": "Brightness",
    "contrast": "Contrast",
    "hue": "Hue",
    "saturation": "Saturation",
    "sharpness": "Sharpness",
    "gamma": "Gamma",
    "wb_temperature": "White Balance",
    "backlight_compensation": "Backlight Comp",
    "gain": "Gain",
}


class CameraControlsPanel(QWidget):
    dialog_closed = Signal()

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.dialog_thread: threading.Thread | None = None
        self.sliders: dict[str, QSlider] = {}
        self.value_labels: dict[str, QLabel] = {}

        layout = QVBoxLayout(self)

        exposure_box = QGroupBox("Exposure / Gain")
        exposure_layout = QHBoxLayout(exposure_box)
        exposure_layout.addWidget(QLabel("Exposure:"))
        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(0.1, 1000.0)
        self.exposure_spin.setDecimals(2)
        self.exposure_spin.setSuffix(" ms")
        exposure_layout.addWidget(self.exposure_spin)
        self.exposure_apply_button = QPushButton("Apply")
        exposure_layout.addWidget(self.exposure_apply_button)
        self.auto_exposure_check = QCheckBox("Auto Exposure")
        exposure_layout.addWidget(self.auto_exposure_check)
        exposure_layout.addStretch(1)
        layout.addWidget(exposure_box)

        video_box = QGroupBox("Video Proc Amp")
        video_layout = QVBoxLayout(video_box)

        for name in camera_controls.CONTROLS:
            video_layout.addLayout(self._make_slider_row(name))

        wb_row = QHBoxLayout()
        self.auto_wb_check = QCheckBox("Auto White Balance")
        wb_row.addWidget(self.auto_wb_check)
        wb_row.addStretch(1)
        video_layout.addLayout(wb_row)

        layout.addWidget(video_box)

        button_row = QHBoxLayout()
        self.load_button = QPushButton("Load")
        self.save_button = QPushButton("Save")
        self.reset_button = QPushButton("Reset to Defaults")
        self.configure_button = QPushButton("Open Native Dialog...")
        for button in (self.load_button, self.save_button, self.reset_button, self.configure_button):
            button_row.addWidget(button)
        layout.addLayout(button_row)

        self.status_label = QLabel("Not connected -- open Live Camera tab and connect first.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch(1)

        self.exposure_apply_button.clicked.connect(self.apply_exposure)
        self.auto_exposure_check.toggled.connect(self.apply_auto_exposure_toggle)
        self.auto_wb_check.toggled.connect(self.apply_auto_wb)
        self.load_button.clicked.connect(self.load_settings)
        self.save_button.clicked.connect(self.save_settings)
        self.reset_button.clicked.connect(self.reset_to_defaults)
        self.configure_button.clicked.connect(self.open_native_dialog)
        self.dialog_closed.connect(self._on_dialog_closed)

        self.load_settings()
        self.refresh_from_camera()

    def _make_slider_row(self, name: str) -> QHBoxLayout:
        bounds = camera_controls.CONTROLS[name]

        row = QHBoxLayout()
        row.addWidget(QLabel(f"{_LABELS[name]}:"))

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(bounds["min"]))
        slider.setMaximum(int(bounds["max"]))
        row.addWidget(slider, 1)

        value_label = QLabel("--")
        value_label.setMinimumWidth(50)
        row.addWidget(value_label)

        slider.valueChanged.connect(lambda value, n=name: self._on_slider_changed(n, value))

        self.sliders[name] = slider
        self.value_labels[name] = value_label

        return row

    def _on_slider_changed(self, name: str, value: int) -> None:
        stream = self.main_window.stream

        if stream is None or stream.cap is None:
            return

        result = camera_controls.set_value(stream.cap, name, float(value))
        self.value_labels[name].setText(f"{result:.0f}")

    def apply_auto_wb(self, checked: bool) -> None:
        stream = self.main_window.stream

        if stream is None or stream.cap is None:
            return

        result = camera_controls.set_auto_wb(stream.cap, checked)
        self.sliders["wb_temperature"].setEnabled(not result)

    def apply_exposure(self) -> None:
        stream = self._require_stream()

        if stream is None:
            return

        # Manual exposure always needs a specific value, unlike auto white
        # balance where "manual" can just mean "stop adjusting, keep the
        # current value" -- so Apply always (re)asserts manual mode too,
        # which _show_exposure_info reflects in the checkbox below.
        info = set_manual_exposure(stream.cap, self.exposure_spin.value())
        self._show_exposure_info(info)

    def apply_auto_exposure_toggle(self, checked: bool) -> None:
        stream = self.main_window.stream

        if stream is None or stream.cap is None:
            # Not connected yet -- revert the visual toggle rather than
            # leaving it showing a state nothing was actually applied to.
            self.auto_exposure_check.blockSignals(True)
            self.auto_exposure_check.setChecked(not checked)
            self.auto_exposure_check.blockSignals(False)
            self.status_label.setText("Not connected -- open Live Camera tab and connect first.")
            return

        if checked:
            info = set_auto_exposure(stream.cap)
        else:
            info = set_manual_exposure(stream.cap, self.exposure_spin.value())

        self._show_exposure_info(info)

    def _show_exposure_info(self, info: dict) -> None:
        # Single source of truth for the checkbox's visual on/off state --
        # every exposure-changing action (Apply, the Auto Exposure toggle,
        # Reset to Defaults, refresh_from_camera) routes through here, so
        # the checkbox can never silently drift out of sync with the
        # camera's real mode.
        is_auto = info["mode"] == "auto"

        self.auto_exposure_check.blockSignals(True)
        self.auto_exposure_check.setChecked(is_auto)
        self.auto_exposure_check.blockSignals(False)

        self.exposure_spin.setEnabled(not is_auto)
        self.exposure_apply_button.setEnabled(not is_auto)

        self.status_label.setText(
            f"Exposure mode: {info['mode']}   "
            f"Exposure: {info['exposure_log2']} log2 ({info['exposure_ms']:.2f} ms)"
        )

    def _require_stream(self):
        stream = self.main_window.stream

        if stream is None or stream.cap is None:
            self.status_label.setText("Not connected -- open Live Camera tab and connect first.")
            return None

        return stream

    def refresh_from_camera(self) -> None:
        """Pull current hardware values into the widgets. Called on
        connect and via the native dialog closing, so this panel and the
        dock's second instance (if open) both reflect real state rather
        than stale UI defaults."""
        stream = self.main_window.stream

        if stream is None or stream.cap is None:
            self.status_label.setText("Not connected -- open Live Camera tab and connect first.")
            return

        values = camera_controls.get_all(stream.cap)

        for name, slider in self.sliders.items():
            slider.blockSignals(True)
            slider.setValue(int(values[name]))
            slider.blockSignals(False)
            self.value_labels[name].setText(f"{values[name]:.0f}")

        self.auto_wb_check.blockSignals(True)
        self.auto_wb_check.setChecked(values["auto_wb"])
        self.auto_wb_check.blockSignals(False)
        self.sliders["wb_temperature"].setEnabled(not values["auto_wb"])

        self._show_exposure_info(get_exposure_gain_info(stream.cap))

    def reset_to_defaults(self) -> None:
        """
        Restores this camera's real as-connected/power-on values, not a
        hardcoded guess -- LiveCameraTab captures them once, right after
        connecting and before anything else touches these properties.
        Available even without the real camera plugged in right now:
        the capture happens automatically the next time you do connect,
        so this is ready to use whenever that happens.
        """
        stream = self._require_stream()

        if stream is None:
            return

        defaults = self.main_window.default_controls
        exposure_defaults = self.main_window.default_exposure_info

        if defaults is None:
            self.status_label.setText(
                "No defaults captured yet -- reconnect the camera first "
                "(defaults are captured automatically right after connecting)."
            )
            return

        for name, value in defaults.items():
            if name == "auto_wb":
                camera_controls.set_auto_wb(stream.cap, value)
            else:
                camera_controls.set_value(stream.cap, name, value)

        if exposure_defaults is not None:
            if exposure_defaults["mode"] == "manual":
                set_manual_exposure(stream.cap, exposure_defaults["exposure_ms"])
            elif exposure_defaults["mode"] == "auto":
                set_auto_exposure(stream.cap)
            # "unknown" mode: driver doesn't clearly support the DirectShow
            # manual/auto convention here (seen on this dev environment's
            # test camera) -- nothing meaningful to restore.

        self.refresh_from_camera()
        self.status_label.setText("Reset to as-connected defaults.")

    def load_settings(self) -> None:
        config = load_config()
        self.exposure_spin.setValue(config.get("exposure_ms", 5.0))

        for name, slider in self.sliders.items():
            if name in config:
                slider.blockSignals(True)
                slider.setValue(int(config[name]))
                slider.blockSignals(False)
                self.value_labels[name].setText(f"{config[name]:.0f}")

        self.status_label.setText("Loaded settings from disk (not yet applied to hardware).")

    def save_settings(self) -> None:
        config = load_config()
        config["exposure_ms"] = self.exposure_spin.value()

        for name, slider in self.sliders.items():
            config[name] = slider.value()

        save_config(config)
        self.status_label.setText("Saved settings to disk.")

    def open_native_dialog(self) -> None:
        stream = self._require_stream()

        if stream is None:
            return

        if self.dialog_thread is not None and self.dialog_thread.is_alive():
            self.status_label.setText("Settings dialog is already open.")
            return

        index = stream.index
        self.status_label.setText("Opening native settings dialog (preview keeps running)...")
        self.dialog_thread = threading.Thread(target=self._dialog_worker, args=(index,), daemon=True)
        self.dialog_thread.start()

    def _dialog_worker(self, index: int) -> None:
        try:
            open_properties_dialog(index)
        except RuntimeError as error:
            print(f"ERROR opening settings dialog: {error}")

        self.dialog_closed.emit()

    def _on_dialog_closed(self) -> None:
        self.refresh_from_camera()
