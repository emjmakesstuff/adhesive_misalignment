"""
Settings tab: embeds one CameraControlsPanel instance. The panel is also
available as a dockable window next to Live Camera (see gui_app.py) --
both are the same reusable widget class, just two separate instances.
"""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from tabs.camera_controls_panel import CameraControlsPanel


class SettingsTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        layout = QVBoxLayout(self)
        self.panel = CameraControlsPanel(main_window)
        layout.addWidget(self.panel)

    def refresh_from_camera(self) -> None:
        self.panel.refresh_from_camera()
