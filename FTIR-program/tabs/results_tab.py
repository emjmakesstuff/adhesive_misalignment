"""
Results tab placeholder. Planned later: per-circle area/brightness
measurements and how they change over time, backed by its own data-
storage module (separate from config_store.py's settings persistence).
No measurement or logging implemented yet.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class ResultsTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        layout = QVBoxLayout(self)
        label = QLabel(
            "Results -- not implemented yet.\n\n"
            "Planned: per-circle area/brightness measurements over time, "
            "with logging to disk."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch(1)
