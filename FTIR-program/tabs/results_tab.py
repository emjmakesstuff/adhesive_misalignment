"""
Results tab: browse the actual SAVED, PROCESSED output files -- every
Range/Full analysis run, across every recording's processing project,
that processing_project.py has already written to disk under
processing_projects/<project_id>/analysis_runs/<analysis_run_id>/
(run.json + roi_results.jsonl). This is a pure read/export/manage layer
over files that already exist; it never runs detection itself -- that
only ever happens from the Processing tab.

(Originally planned as a live per-circle measurement view -- superseded
by the Processing tab's ROI analysis-run system, which already produces
and persists exactly that kind of per-frame/per-ROI measurement data for
recorded video. This tab is now where you go to find, inspect, export,
or clean up what that system has already saved, not to run anything new.)
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import processing_project as pp
import recording_store as rs

COLUMNS = [
    "Recording", "Analysis Run", "Detector", "Status", "Requested Frames",
    "Completed Frames", "ROIs", "Result Rows", "Created",
]


def _status_text(run: dict, incomplete: bool) -> str:
    status = run.get("status", "unknown")
    if status == "completed" and not incomplete:
        return "COMPLETED"
    return f"{status.upper()} (left marked incomplete)" if incomplete else status.upper()


def _frame_range_text(frame_range: dict | None) -> str:
    if frame_range is None:
        return "N/A"
    return f"{frame_range.get('start')}-{frame_range.get('end')}"


def _flatten(prefix: str, value, out: dict) -> None:
    """Turns an arbitrarily-nested dict (e.g. a roi_results.jsonl row's
    detector_signal/corrected_frame_signal blocks) into flat dot-joined
    columns for CSV export -- generic rather than hardcoded to the
    current schema's exact shape, so it stays correct if more fields
    (e.g. a future reference-normalization pass) get added later."""
    if isinstance(value, dict):
        for key, sub_value in value.items():
            _flatten(f"{prefix}_{key}" if prefix else key, sub_value, out)
    else:
        out[prefix] = value


class ResultsTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._rows: list[tuple[str, str, dict]] = []  # (project_id, analysis_run_id, run)

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Every Range/Full detection run that's been saved to disk, across every recording's "
            "processing project -- run.json (config/ROI snapshot) + roi_results.jsonl (per-frame, "
            "per-ROI measurements). A red row means the run is marked incomplete (cancelled, "
            "crashed, or still in progress) -- its numbers reflect only however far it actually got."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        top_row = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        top_row.addWidget(self.refresh_button)
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
        self.export_csv_button = QPushButton("Export CSV")
        action_row.addWidget(self.export_csv_button)
        self.open_processing_button = QPushButton("Open Recording in Processing")
        action_row.addWidget(self.open_processing_button)
        self.delete_button = QPushButton("Delete...")
        action_row.addWidget(self.delete_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.detail_text = QPlainTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(160)
        self.detail_text.setPlainText("Select a run above to see its detail.")
        layout.addWidget(self.detail_text)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.refresh_button.clicked.connect(self.refresh)
        self.open_folder_button.clicked.connect(self.open_folder)
        self.export_csv_button.clicked.connect(self.export_csv)
        self.open_processing_button.clicked.connect(self.open_recording_in_processing)
        self.delete_button.clicked.connect(self.delete_selected)
        self.table.currentCellChanged.connect(lambda *_args: self._update_detail())

        self.refresh()

    # ---- listing ----

    def refresh(self) -> None:
        self._rows = []
        for project_id in pp.list_projects():
            for analysis_run_id in pp.list_analysis_runs(project_id):
                run = pp.load_run(project_id, analysis_run_id)
                if run is not None:
                    self._rows.append((project_id, analysis_run_id, run))

        self.table.setRowCount(len(self._rows))
        for row_index, (project_id, analysis_run_id, run) in enumerate(self._rows):
            recording_id = run.get("recording_id", "")
            metadata = rs.load_metadata(recording_id) if recording_id else None
            recording_label = metadata.get("experiment_name", recording_id) if metadata is not None else f"{recording_id} (recording deleted)"
            incomplete = pp.is_run_incomplete(project_id, analysis_run_id)

            values = [
                recording_label,
                analysis_run_id,
                run.get("detector_type", ""),
                _status_text(run, incomplete),
                _frame_range_text(run.get("requested_frame_range")),
                _frame_range_text(run.get("completed_frame_range")),
                str(len(run.get("rois", []))),
                str(run.get("result_row_count", 0)),
                run.get("created_at", ""),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, (project_id, analysis_run_id))
                if incomplete:
                    item.setForeground(Qt.red)
                self.table.setItem(row_index, col, item)

        self.status_label.setText(f"{len(self._rows)} saved analysis run(s) across {len(pp.list_projects())} project(s)")
        self._update_detail()

    def _selected(self) -> tuple[str, str, dict] | None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._rows):
            return None
        return self._rows[row]

    def _update_detail(self) -> None:
        selected = self._selected()
        if selected is None:
            self.detail_text.setPlainText("Select a run above to see its detail.")
            return

        project_id, analysis_run_id, run = selected
        roi_names = ", ".join(roi.get("name", roi.get("roi_id", "?")) for roi in run.get("rois", []))
        lines = [
            f"analysis_run_id: {analysis_run_id}   project_id: {project_id}",
            f"recording_id: {run.get('recording_id')}",
            f"status: {run.get('status')}   stop_reason: {run.get('stop_reason')}",
            f"detector_type: {run.get('detector_type')}   config_hash: {run.get('config_hash')}",
            f"requested frames: {_frame_range_text(run.get('requested_frame_range'))}   "
            f"completed frames: {_frame_range_text(run.get('completed_frame_range'))}",
            f"result_row_count: {run.get('result_row_count')}",
            f"ROIs ({len(run.get('rois', []))}): {roi_names or 'none'}",
            f"saved at: {pp.analysis_run_dir(project_id, analysis_run_id)}",
        ]
        self.detail_text.setPlainText("\n".join(lines))

    # ---- actions ----

    def open_folder(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        project_id, analysis_run_id, _run = selected
        directory = pp.analysis_run_dir(project_id, analysis_run_id)

        if sys.platform.startswith("win"):
            os.startfile(directory)  # noqa: S606 -- opening a folder the user just picked from our own list
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(directory)])
        else:
            subprocess.Popen(["xdg-open", str(directory)])

    def export_csv(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        project_id, analysis_run_id, _run = selected

        rows = list(pp.read_roi_results(project_id, analysis_run_id))
        if not rows:
            QMessageBox.information(self, "Export CSV", "This run has no result rows to export.")
            return

        flat_rows = []
        fieldnames: list[str] = []
        for row in rows:
            flat: dict = {}
            _flatten("", row, flat)
            flat_rows.append(flat)
            for key in flat:
                if key not in fieldnames:
                    fieldnames.append(key)

        csv_path = pp.analysis_run_dir(project_id, analysis_run_id) / "roi_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)

        self.status_label.setText(f"Exported {len(flat_rows)} row(s) to {csv_path}")

    def open_recording_in_processing(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        _project_id, _analysis_run_id, run = selected
        recording_id = run.get("recording_id")
        if not recording_id or rs.load_metadata(recording_id) is None:
            QMessageBox.information(self, "Open in Processing", "This run's recording no longer exists on disk.")
            return

        self.main_window.open_recording_in_processing(recording_id)

    def delete_selected(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        project_id, analysis_run_id, run = selected

        confirm = QMessageBox.warning(
            self,
            "Delete Analysis Run",
            f'Permanently delete saved run "{analysis_run_id}" '
            f"({run.get('result_row_count', 0)} result row(s))?\n\n"
            f"This removes run.json, roi_results.jsonl, and any exported files from disk. "
            f"The recording and its ROI definitions are untouched. This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        pp.delete_analysis_run(project_id, analysis_run_id)
        self.status_label.setText(f"Deleted {analysis_run_id}.")
        self.refresh()
