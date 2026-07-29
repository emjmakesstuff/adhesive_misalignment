"""
Desktop application entry point: tabbed UI (Live Camera, Settings,
Calibration, Detection, Results) wrapping the existing camera.py /
settings.py capture and control logic without modifying either --
preview.py stays available separately as the original CLI tool.

A single CameraStream is created by the Live Camera tab and shared (via
main_window.stream) with the Settings tab and the dockable controls
panel. Because all five tabs are constructed once up front and
QTabWidget only hides/shows them, switching tabs never restarts or
duplicates the camera connection -- only Connect/Disconnect on the Live
Camera tab does that.

Run:
    py gui_app.py
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QDockWidget, QMainWindow, QTabWidget

import calibration_profiles
from frame_dispatcher import FrameDispatcher
from recorder import Recorder
from tabs.calibration_tab import CalibrationTab
from tabs.camera_controls_panel import CameraControlsPanel
from tabs.detection_tab import DetectionTab
from tabs.live_camera_tab import LiveCameraTab
from tabs.processing_tab import ProcessingTab
from tabs.recordings_tab import RecordingsTab
from tabs.results_tab import ResultsTab
from tabs.settings_tab import SettingsTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FTIR Camera Measurement")

        # One-time: if a profile registry doesn't exist yet but the old
        # flat calibration/scale/circle_config/background_reference files
        # do (from before per-source profiles existed), migrate them into
        # a new "USB Camera" profile so nothing already captured is lost.
        # No-ops on every later startup once the registry exists.
        calibration_profiles.ensure_default_usb_profile()

        # "usb" or "vdo_ninja" -- which kind of object main_window.stream
        # currently holds (a camera.CameraStream or a
        # vdo_ninja_source.VdoNinjaSource), set by LiveCameraTab whenever
        # the source type is switched. Other tabs (Calibration, Detection)
        # read this to pick the right calibration profile/detector rather
        # than re-deriving it from type(main_window.stream).
        self.active_source_type = "usb"

        # The actual key used for calibration_profiles lookups
        # (get_active_profile/set_active_profile). Equal to
        # active_source_type for "vdo_ninja" ("vdo_ninja" both ways), but
        # for "usb" this is the more specific per-physical-camera key
        # calibration_profiles.profile_key() produces (so the U20CAM and
        # the ELP, both source_type "usb", resolve to different profiles).
        # Set by LiveCameraTab alongside active_source_type whenever the
        # source/camera changes; other tabs should read this, not
        # active_source_type, whenever looking up a profile.
        self.active_profile_key = "usb"

        self.stream = None  # owned by LiveCameraTab; shared here for other tabs -- a CameraStream or a VdoNinjaSource
        self.controls_dock: QDockWidget | None = None

        # Latest raw (unprocessed, undistorted) frame LiveCameraTab
        # successfully read, shared so other tabs -- Calibration now,
        # Detection later -- can look at live frames without a second
        # independent stream.read() call competing with LiveCameraTab's
        # own timer for the same cv2.VideoCapture.
        self.latest_frame = None

        # Current edge-crop percentages from Live Camera's "Crop after
        # undistort" sliders, kept here (not read tab-to-tab directly) so
        # Calibration's scale-reference capture can crop the same way the
        # live preview does, consistent with the pattern already used for
        # `stream`/`latest_frame`. Persisted per-profile now (via
        # calibration_profiles.update_profile_crop), loaded into this
        # runtime dict whenever the active source/profile changes.
        self.crop_percentages = {"top": 0, "bottom": 0, "left": 0, "right": 0}

        # As-connected camera_controls/exposure state, captured once by
        # LiveCameraTab right after connecting, before anything else
        # changes these properties. Used by CameraControlsPanel's
        # "Reset to Defaults" to restore this camera's real power-on
        # values rather than a hardcoded guess.
        self.default_controls: dict | None = None
        self.default_exposure_info: dict | None = None

        # Single dispatcher instance for the app's lifetime -- every
        # background capture thread (ThreadedCameraSource for USB,
        # VdoNinjaSource for VDO.Ninja) publishes into this one object
        # regardless of how many times a source connects/disconnects
        # across the session. See frame_dispatcher.py.
        self.frame_dispatcher = FrameDispatcher(self)

        # Single Recorder instance for the app's lifetime, same pattern as
        # frame_dispatcher above -- LiveCameraTab drives start()/stop() on
        # it (Start/Stop Recording buttons, the hotkey below, and the
        # finalize-on-disconnect/close/error paths it owns).
        self.recorder = Recorder(self)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.live_camera_tab = LiveCameraTab(self)
        self.settings_tab = SettingsTab(self)
        self.calibration_tab = CalibrationTab(self)
        self.detection_tab = DetectionTab(self)
        self.recordings_tab = RecordingsTab(self)
        self.processing_tab = ProcessingTab(self)
        self.results_tab = ResultsTab(self)

        self.tabs.addTab(self.live_camera_tab, "Live Camera")
        self.tabs.addTab(self.settings_tab, "Settings")
        self.tabs.addTab(self.calibration_tab, "Calibration")
        self.tabs.addTab(self.detection_tab, "Detection")
        self.tabs.addTab(self.recordings_tab, "Recordings")
        self.tabs.addTab(self.processing_tab, "Processing")
        self.tabs.addTab(self.results_tab, "Results")

        # Recordings created/finalized elsewhere (Live Camera's Start/Stop
        # Recording, the hotkey) don't otherwise notify this tab -- refresh
        # whenever it becomes the visible tab so its list is never stale
        # without requiring a manual Refresh click.
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self.recordings_tab:
            self.recordings_tab.refresh()
        elif self.tabs.widget(index) is self.results_tab:
            self.results_tab.refresh()

    def open_recording_in_processing(self, recording_id: str) -> None:
        """Called by RecordingsTab's "Open in Processing" button."""
        self.processing_tab.open_recording(recording_id)
        self.tabs.setCurrentWidget(self.processing_tab)

        # App-wide -- works regardless of which tab currently has focus,
        # not just while Live Camera is the visible tab.
        self.record_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        self.record_shortcut.activated.connect(self.live_camera_tab.toggle_recording_hotkey)

    def on_camera_connected(self) -> None:
        """Called by LiveCameraTab right after a successful connect, so
        any already-open settings surface (the Settings tab, and the
        dock panel if open) shows real hardware values instead of
        whatever was on screen (or loaded from disk) before connecting."""
        self.settings_tab.refresh_from_camera()

        if self.controls_dock is not None:
            self.controls_dock.widget().refresh_from_camera()

    def on_active_profile_changed(self) -> None:
        """Called by CalibrationTab whenever the active calibration
        profile (for the currently active source type) is created or
        switched, so LiveCameraTab's crop sliders -- persisted per-
        profile, not as a flat global -- reload from the newly-active
        profile instead of continuing to show/save into the old one."""
        self.live_camera_tab.reload_crop_from_profile()

    def toggle_controls_dock(self) -> bool:
        """Shows/hides a dockable camera-controls panel next to whatever
        tab is currently active (including Live Camera). Returns the new
        visibility state. This is a second, independent
        CameraControlsPanel instance -- not the same widget as the
        Settings tab's, since a widget can only have one parent at a
        time -- but both talk to the same main_window.stream, so either
        can be used to tweak the live camera."""
        if self.controls_dock is None:
            self.controls_dock = QDockWidget("Camera Controls", self)
            self.controls_dock.setWidget(CameraControlsPanel(self))
            self.controls_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
            self.addDockWidget(Qt.RightDockWidgetArea, self.controls_dock)
            self.controls_dock.visibilityChanged.connect(self._on_dock_visibility_changed)
            return True

        visible = not self.controls_dock.isVisible()
        self.controls_dock.setVisible(visible)
        return visible

    def _on_dock_visibility_changed(self, visible: bool) -> None:
        self.live_camera_tab.set_controls_button_checked(visible)

    def closeEvent(self, event) -> None:
        if self.recorder.is_recording:
            self.recorder.stop(stop_reason="app_close")
        self.live_camera_tab.stop()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1440, 900)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
