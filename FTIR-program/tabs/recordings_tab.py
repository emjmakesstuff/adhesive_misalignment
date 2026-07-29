"""
Recordings tab: browse everything recording_store.py has on disk under
recordings/ -- list, Import Video, rename/notes/protect/delete, open the
folder in the OS file browser, and Open in Processing (switches
main_window to the Processing tab and loads the selected recording
there -- see gui_app.MainWindow.open_recording_in_processing).

Pure management UI over recording_store.py's existing primitives -- this
tab never writes video/frame_index/calibration_snapshot files itself,
only metadata.json's narrow rename/notes/protected fields (via
recording_store.rename_recording/update_recording_notes/
set_recording_protected) and, for Import Video, delegates the actual
file copy + schema construction to recording_store.import_video().
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import calibration_profiles
import distortion
import recording_store as rs

COLUMNS = [
    "Experiment Name", "Created", "Source", "Role / Alias", "Codec",
    "Quality Mode", "Frames", "Duration", "Size", "Notes", "Protected", "Status",
]


def _format_duration(meta: dict) -> str:
    recording = meta.get("recording", {})
    frames = recording.get("frames_write_attempted")
    fps = meta.get("camera", {}).get("measured_fps") or meta.get("camera", {}).get("requested_fps")

    if not frames or not fps:
        return "N/A"

    seconds = frames / fps
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s"


def _format_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "N/A"
    if size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.2f} GB"
    if size_bytes >= 1e6:
        return f"{size_bytes / 1e6:.1f} MB"
    return f"{size_bytes / 1e3:.1f} KB"


class ImportVideoDialog(QDialog):
    """
    Requires, before it completes (per the recording-workflow plan's
    Import section): camera-profile selection, resolution/orientation
    confirmation (read from the file itself, shown rather than trusted),
    and a calibration-mismatch warning if the selected profile's saved
    calibration resolution doesn't match the imported video's real
    resolution.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Video")
        self.setModal(True)
        self.setMinimumWidth(480)

        self.source_path: Path | None = None
        self._video_info: dict | None = None

        layout = QVBoxLayout(self)

        file_row = QHBoxLayout()
        self.file_label = QLabel("No file selected.")
        self.file_label.setWordWrap(True)
        file_row.addWidget(self.file_label, 1)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse)
        file_row.addWidget(self.browse_button)
        layout.addLayout(file_row)

        form = QFormLayout()
        self.resolution_label = QLabel("--")
        form.addRow("Real resolution (read from file):", self.resolution_label)
        self.fps_frames_label = QLabel("--")
        form.addRow("FPS / frame count:", self.fps_frames_label)

        self.profile_combo = QComboBox()
        self.profile_combo.addItem("None / unknown", None)
        for profile in calibration_profiles.list_profiles():
            label = f"{profile['alias']} ({profile['source_type']})"
            self.profile_combo.addItem(label, profile["id"])
        self.profile_combo.currentIndexChanged.connect(self._update_mismatch_warning)
        form.addRow("Camera profile:", self.profile_combo)

        self.name_edit = QLineEdit()
        form.addRow("Experiment name:", self.name_edit)

        layout.addLayout(form)

        self.mismatch_label = QLabel("")
        self.mismatch_label.setWordWrap(True)
        self.mismatch_label.setStyleSheet("color: #b00000;")
        layout.addWidget(self.mismatch_label)

        self.lossy_note_label = QLabel(
            "Imported footage is always tagged quality_mode=\"lossy_prototype\" -- this app "
            "cannot verify an externally-sourced file's lossless fidelity, so it is never "
            "silently treated as measurement-grade."
        )
        self.lossy_note_label.setWordWrap(True)
        layout.addWidget(self.lossy_note_label)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        layout.addWidget(self.buttons)

    def _browse(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Select a video file to import", "", "Video files (*.avi *.mp4 *.mkv *.mov);;All files (*)"
        )
        if not path_str:
            return

        path = Path(path_str)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            QMessageBox.warning(self, "Cannot import", f"Could not open {path.name} as a video file.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.source_path = path
        self._video_info = {"width": width, "height": height, "fps": fps, "frame_count": frame_count}

        self.file_label.setText(str(path))
        self.resolution_label.setText(f"{width} x {height}  (confirm this looks right -- not silently trusted)")
        self.fps_frames_label.setText(f"{fps:.2f} fps, ~{frame_count} frames (container estimate)")
        self.name_edit.setText(path.stem)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(True)

        self._update_mismatch_warning()

    def _update_mismatch_warning(self) -> None:
        self.mismatch_label.setText("")

        if self._video_info is None:
            return

        profile_id = self.profile_combo.currentData()
        if profile_id is None:
            return

        calibration = distortion.load_calibration(path=calibration_profiles.distortion_path(profile_id))
        if calibration is None or calibration.get("image_size") is None:
            return

        cal_width, cal_height = calibration["image_size"]
        if (cal_width, cal_height) != (self._video_info["width"], self._video_info["height"]):
            self.mismatch_label.setText(
                f"WARNING: the selected profile's saved calibration was done at "
                f"{cal_width}x{cal_height}, but this video is {self._video_info['width']}x"
                f"{self._video_info['height']}. Calibration/detection will not line up correctly "
                f"unless this profile is recalibrated at the video's real resolution."
            )

    def selected_profile(self) -> dict | None:
        profile_id = self.profile_combo.currentData()
        return calibration_profiles.get_profile(profile_id) if profile_id is not None else None

    def experiment_name(self) -> str:
        return self.name_edit.text().strip()


class RecordingsTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._recording_ids: list[str] = []

        layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        top_row.addWidget(self.refresh_button)
        self.import_button = QPushButton("Import Video...")
        top_row.addWidget(self.import_button)
        top_row.addStretch(1)
        layout.addLayout(top_row)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table, 1)

        action_row = QHBoxLayout()
        self.open_folder_button = QPushButton("Open Folder")
        action_row.addWidget(self.open_folder_button)
        self.rename_button = QPushButton("Rename")
        action_row.addWidget(self.rename_button)
        self.notes_button = QPushButton("Edit Notes")
        action_row.addWidget(self.notes_button)
        self.protect_button = QPushButton("Protect / Unprotect")
        action_row.addWidget(self.protect_button)
        self.delete_button = QPushButton("Delete...")
        action_row.addWidget(self.delete_button)
        self.open_processing_button = QPushButton("Open in Processing")
        action_row.addWidget(self.open_processing_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.refresh_button.clicked.connect(self.refresh)
        self.import_button.clicked.connect(self.import_video)
        self.open_folder_button.clicked.connect(self.open_folder)
        self.rename_button.clicked.connect(self.rename_selected)
        self.notes_button.clicked.connect(self.edit_notes_selected)
        self.protect_button.clicked.connect(self.toggle_protected_selected)
        self.delete_button.clicked.connect(self.delete_selected)
        self.open_processing_button.clicked.connect(self.open_in_processing)

        self.refresh()

    # ---- listing ----

    def refresh(self) -> None:
        self._recording_ids = rs.list_recordings()
        self.table.setRowCount(len(self._recording_ids))

        for row, recording_id in enumerate(self._recording_ids):
            meta = rs.load_metadata(recording_id)
            if meta is None:
                continue

            camera = meta.get("camera", {})
            recording = meta.get("recording", {})
            video_path = rs.video_path_for(recording_id, meta)
            size_bytes = video_path.stat().st_size if video_path is not None else None
            incomplete = rs.is_incomplete(recording_id)

            role_or_alias = camera.get("alias") or camera.get("camera_role") or "unknown (imported)"
            status = "INCOMPLETE (killed mid-recording)" if incomplete else meta.get("source", "")

            values = [
                meta.get("experiment_name", recording_id),
                meta.get("created_at", ""),
                meta.get("source", ""),
                role_or_alias,
                camera.get("codec_fourcc") or "N/A",
                meta.get("quality_mode", ""),
                str(recording.get("frames_write_attempted", "N/A")),
                _format_duration(meta),
                _format_size(size_bytes),
                "yes" if meta.get("notes") else "",
                "yes" if meta.get("protected") else "",
                status,
            ]

            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, recording_id)
                if incomplete:
                    item.setForeground(Qt.red)
                self.table.setItem(row, col, item)

        self.status_label.setText(f"{len(self._recording_ids)} recording(s) in {rs.RECORDINGS_DIR}")

    def _selected_recording_id(self) -> str | None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._recording_ids):
            return None
        return self._recording_ids[row]

    # ---- import ----

    def import_video(self) -> None:
        dialog = ImportVideoDialog(parent=self)
        if dialog.exec() != QDialog.Accepted or dialog.source_path is None:
            return

        try:
            recording_id = rs.import_video(
                dialog.source_path, dialog.selected_profile(), experiment_name=dialog.experiment_name()
            )
        except (FileNotFoundError, ValueError, OSError) as error:
            QMessageBox.critical(self, "Import failed", str(error))
            return

        self.status_label.setText(f"Imported as {recording_id}.")
        self.refresh()

    # ---- actions on the selected recording ----

    def open_folder(self) -> None:
        recording_id = self._selected_recording_id()
        if recording_id is None:
            return

        directory = rs.recording_dir(recording_id)
        if sys.platform.startswith("win"):
            os.startfile(directory)  # noqa: S606 -- opening a folder the user just picked from our own list
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(directory)])
        else:
            subprocess.Popen(["xdg-open", str(directory)])

    def rename_selected(self) -> None:
        recording_id = self._selected_recording_id()
        if recording_id is None:
            return

        meta = rs.load_metadata(recording_id)
        current = meta.get("experiment_name", recording_id) if meta is not None else recording_id

        new_name, ok = QInputDialog.getText(self, "Rename Recording", "Experiment name:", text=current)
        if not ok or not new_name.strip():
            return

        rs.rename_recording(recording_id, new_name.strip())
        self.refresh()

    def edit_notes_selected(self) -> None:
        recording_id = self._selected_recording_id()
        if recording_id is None:
            return

        meta = rs.load_metadata(recording_id)
        current_notes = meta.get("notes", "") if meta is not None else ""

        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Notes")
        dialog.setModal(True)
        dialog_layout = QVBoxLayout(dialog)
        text_edit = QPlainTextEdit(current_notes)
        dialog_layout.addWidget(text_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dialog_layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        rs.update_recording_notes(recording_id, text_edit.toPlainText())
        self.refresh()

    def toggle_protected_selected(self) -> None:
        recording_id = self._selected_recording_id()
        if recording_id is None:
            return

        meta = rs.load_metadata(recording_id)
        currently_protected = bool(meta.get("protected")) if meta is not None else False

        rs.set_recording_protected(recording_id, not currently_protected)
        self.refresh()

    def delete_selected(self) -> None:
        recording_id = self._selected_recording_id()
        if recording_id is None:
            return

        meta = rs.load_metadata(recording_id)
        name = meta.get("experiment_name", recording_id) if meta is not None else recording_id

        if meta is not None and meta.get("protected"):
            QMessageBox.information(
                self, "Protected", f'"{name}" is protected -- unprotect it first if you really want to delete it.'
            )
            return

        confirm = QMessageBox.warning(
            self,
            "Delete Recording",
            f'Permanently delete "{name}" ({recording_id})?\n\n'
            f"This removes the video, frame index, and calibration snapshot from disk. "
            f"This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        rs.delete_recording(recording_id)
        self.status_label.setText(f"Deleted {recording_id}.")
        self.refresh()

    def open_in_processing(self) -> None:
        recording_id = self._selected_recording_id()
        if recording_id is None:
            return

        self.main_window.open_recording_in_processing(recording_id)
