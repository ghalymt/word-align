"""Main window for the native WordAlign GUI."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSpinBox, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
    QWidget,
)

from ..core.config import PipelineProfile
from ..core.events import CollectSink, PipelineCompleted, StageCompleted, StageMessage
from ..core.pipeline import PipelineRunner
from ..config import PipelineConfig
from .workers import PipelineWorker


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("WordAlign 2.0")
        self.resize(960, 640)

        self._worker: PipelineWorker | None = None

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # --- Input group ---
        input_group = QGroupBox("Pipeline inputs")
        form = QFormLayout(input_group)

        self.audio_edit = QLineEdit()
        audio_row = QHBoxLayout()
        audio_row.addWidget(self.audio_edit)
        audio_btn = QPushButton("Browse…")
        audio_btn.clicked.connect(self._pick_audio)
        audio_row.addWidget(audio_btn)
        form.addRow("Audio file:", audio_row)

        self.transcript_edit = QLineEdit()
        transcript_row = QHBoxLayout()
        transcript_row.addWidget(self.transcript_edit)
        transcript_btn = QPushButton("Browse…")
        transcript_btn.clicked.connect(self._pick_transcript)
        transcript_row.addWidget(transcript_btn)
        form.addRow("Transcript (optional):", transcript_row)

        self.language_edit = QLineEdit("en")
        form.addRow("Language (ISO 639-1):", self.language_edit)

        self.cpl_spin = QSpinBox()
        self.cpl_spin.setRange(20, 60)
        self.cpl_spin.setValue(42)
        form.addRow("Max chars per line:", self.cpl_spin)

        self.vosk_edit = QLineEdit(os.environ.get("WORDALIGN_VOSK_MODELS", ""))
        vosk_row = QHBoxLayout()
        vosk_row.addWidget(self.vosk_edit)
        vosk_btn = QPushButton("Browse…")
        vosk_btn.clicked.connect(self._pick_vosk)
        vosk_row.addWidget(vosk_btn)
        form.addRow("Vosk models dir:", vosk_row)

        root.addWidget(input_group)

        # --- Run controls ---
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run pipeline")
        self.run_btn.setDefault(True)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.cancel_btn)
        run_row.addStretch()
        root.addLayout(run_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        root.addWidget(self.progress)

        # --- Splitter: log + QA ---
        splitter = QSplitter(Qt.Vertical)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        splitter.addWidget(self.log_view)

        self.qa_tree = QTreeWidget()
        self.qa_tree.setHeaderLabels(["Severity", "Category", "Time", "Explanation"])
        splitter.addWidget(self.qa_tree)

        root.addWidget(splitter, stretch=1)

        # --- Output label ---
        self.output_label = QLabel("")
        self.output_label.setWordWrap(True)
        root.addWidget(self.output_label)

    def _connect_signals(self):
        self.run_btn.clicked.connect(self._start_run)
        self.cancel_btn.clicked.connect(self._cancel_run)

    # ------------------------------------------------------------ dialogs
    def _pick_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select audio/video", "",
            "Media (*.mp3 *.wav *.m4a *.mp4 *.mkv *.flac *.ogg);;All files (*.*)")
        if path:
            self.audio_edit.setText(path)

    def _pick_transcript(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select transcript", "", "Text (*.txt);;All files (*.*)")
        if path:
            self.transcript_edit.setText(path)

    def _pick_vosk(self):
        path = QFileDialog.getExistingDirectory(self, "Select Vosk models directory")
        if path:
            self.vosk_edit.setText(path)

    # -------------------------------------------------------------- runs
    def _start_run(self):
        audio = self.audio_edit.text().strip()
        if not audio or not Path(audio).exists():
            QMessageBox.warning(self, "WordAlign", "Please select a valid audio file.")
            return

        transcript = self.transcript_edit.text().strip() or None
        cfg = PipelineConfig(
            audio_path=audio,
            transcript_path=transcript,
            language=self.language_edit.text().strip() or None,
            vosk_models_dir=self.vosk_edit.text().strip() or None,
            use_vosk=True,
            use_mfa=False,
            max_cpl=self.cpl_spin.value(),
            max_lines=2,
            max_duration_ms=7000,
            min_cue_ms=700,
        )
        profile = PipelineProfile.preset("balanced")

        self.log_view.clear()
        self.qa_tree.clear()
        self.output_label.setText("")
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        sink = CollectSink()
        self._worker = PipelineWorker(cfg, profile, sink)
        self._worker.event_received.connect(self._on_event)
        self._worker.finished_run.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _cancel_run(self):
        if self._worker:
            self._worker.cancel()
            self.log_view.appendPlainText("[user] Cancellation requested…")

    def _on_event(self, text: str):
        self.log_view.appendPlainText(text)

    def _on_finished(self, output_files: list, events: list):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setValue(100)
        if output_files:
            self.output_label.setText("Output files:\n" + "\n".join(output_files))
        else:
            self.output_label.setText("Pipeline finished but produced no output.")
        self._populate_qa(events)

    def _on_failed(self, message: str):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setValue(0)
        QMessageBox.critical(self, "WordAlign", f"Pipeline failed:\n{message}")

    def _populate_qa(self, events: list):
        """Populate QA tree from pipeline events (if QA results are embedded)."""
        from ..qa.issues import TranscriptIssue
        issues = [e for e in events if isinstance(e, TranscriptIssue)]
        for issue in issues:
            item = QTreeWidgetItem([
                issue.severity, issue.category,
                f"{issue.start:.1f}s – {issue.end:.1f}s",
                issue.explanation or issue.original_text,
            ])
            self.qa_tree.addTopLevelItem(item)
