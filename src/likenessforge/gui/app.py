from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from likenessforge.config import ForgeConfig
from likenessforge.debug import DebugSink
from likenessforge.image_ops import annotate_faces
from likenessforge.model_store import ModelProgress, ModelStore
from likenessforge.registry import create, plugin_classes
from likenessforge.workflow import RepairWorkflow, TrainingWorkflow
from likenessforge.gui.workers import TaskThread


LOG = logging.getLogger("likenessforge")
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;All files (*)"


def _plugin_name(combo: QComboBox) -> str:
    return str(combo.currentData(Qt.UserRole) or combo.currentText())


def _set_combo_plugin(combo: QComboBox, name: str) -> None:
    for index in range(combo.count()):
        if combo.itemData(index, Qt.UserRole) == name:
            combo.setCurrentIndex(index)
            return


def _populate_plugin_combo(combo: QComboBox, kind: str) -> None:
    combo.clear()
    for name, cls in sorted(plugin_classes(kind).items()):
        info = cls.info
        low = " • low VRAM" if info.low_vram else ""
        combo.addItem(f"{name}{low}", name)
        combo.setItemData(combo.count() - 1, info.description, Qt.ToolTipRole)


def _format_bytes(value: float | None) -> str:
    if value is None:
        return "?"
    size = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or suffix == "TiB":
            return f"{size:.1f} {suffix}"
        size /= 1024.0
    return f"{size:.1f} TiB"



def _format_progress(event: ModelProgress) -> str:
    unit = (event.unit or "").lower()
    if unit in {"b", "byte", "bytes"}:
        return f"{_format_bytes(event.current)} / {_format_bytes(event.total)}"
    current = "?" if event.current is None else f"{event.current:g}"
    total = "?" if event.total is None else f"{event.total:g}"
    suffix = f" {event.unit}" if event.unit else ""
    return f"{current} / {total}{suffix}"


def _load_training_metadata(lora_path: str | Path) -> dict[str, Any] | None:
    path = Path(lora_path)
    candidates = [path.parent / "likenessforge-training.json", path.parent.parent / "likenessforge-training.json"]
    for candidate in candidates:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def _remove_selected_items(widget: QListWidget) -> None:
    rows = sorted({index.row() for index in widget.selectedIndexes()}, reverse=True)
    for row in rows:
        widget.takeItem(row)


class ImagePreview(QLabel):
    def __init__(self, text: str = "No image", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 260)
        self.setStyleSheet("QLabel { border: 1px solid palette(mid); background: palette(base); }")
        self._pixmap: QPixmap | None = None

    def set_pil(self, image: Image.Image | None) -> None:
        if image is None:
            self._pixmap = None
            self.setPixmap(QPixmap())
            self.setText("No image")
            return
        qimage = ImageQt(image.convert("RGBA"))
        self._pixmap = QPixmap.fromImage(qimage.copy())
        self._refresh()

    def _refresh(self) -> None:
        if self._pixmap is None:
            return
        self.setText("")
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()


class ModelSetupWidget(QGroupBox):
    download_requested = Signal()

    def __init__(self, cache_dir: str | Path | None = None, parent=None):
        super().__init__("Models", parent)
        self.store = ModelStore(cache_dir)
        self.detector = QComboBox()
        self.segmenter = QComboBox()
        self.modifier = QComboBox()
        self.trainer = QComboBox()
        self.low_vram = QComboBox()
        self.low_vram.addItems(["auto", "off", "cpu_offload", "bnb4"])
        _populate_plugin_combo(self.detector, "detector")
        _populate_plugin_combo(self.segmenter, "segmenter")
        _populate_plugin_combo(self.modifier, "modifier")
        _populate_plugin_combo(self.trainer, "trainer")

        defaults = ForgeConfig()
        _set_combo_plugin(self.detector, "mtcnn")
        _set_combo_plugin(self.segmenter, "sam2-tiny")
        _set_combo_plugin(self.modifier, defaults.modifier.name)
        _set_combo_plugin(self.trainer, "sdxl-lora")
        self.low_vram.setCurrentText(defaults.low_vram)

        form = QFormLayout()
        form.addRow("Face detector", self.detector)
        form.addRow("Segmentation model", self.segmenter)
        form.addRow("Image modifier / inpainting", self.modifier)
        form.addRow("LoRA trainer", self.trainer)
        form.addRow("Inference VRAM profile", self.low_vram)

        self.compatibility = QLabel()
        self.compatibility.setWordWrap(True)
        self.table = QTreeWidget()
        self.table.setHeaderLabels(["Role", "Plugin", "Model", "Status"])
        self.table.setRootIsDecorated(False)
        self.table.setMaximumHeight(130)
        self.download = QPushButton("Download / install selected model weights")
        self.refresh = QPushButton("Refresh model status")
        self.progress_label = QLabel("Ready")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        self.log.setMaximumHeight(100)

        buttons = QHBoxLayout()
        buttons.addWidget(self.download)
        buttons.addWidget(self.refresh)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.compatibility)
        layout.addWidget(self.table)
        layout.addLayout(buttons)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.log)

        for combo in (self.segmenter, self.modifier, self.trainer):
            combo.currentIndexChanged.connect(self.refresh_status)
        self.modifier.currentIndexChanged.connect(self._refresh_compatibility)
        self.trainer.currentIndexChanged.connect(self._refresh_compatibility)
        self.refresh.clicked.connect(self.refresh_status)
        self.download.clicked.connect(self.download_requested.emit)
        self.refresh_status()
        self._refresh_compatibility()

    def selected(self, kind: str) -> str:
        return _plugin_name(getattr(self, kind))

    def selected_model_plugins(self) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        for kind in ("segmenter", "modifier", "trainer"):
            name = self.selected(kind)
            info = plugin_classes(kind)[name].info
            if info.model_id:
                rows.append((kind, name, info.model_id))
        return rows

    def _refresh_compatibility(self) -> None:
        modifier_info = plugin_classes("modifier")[self.selected("modifier")].info
        trainer_info = plugin_classes("trainer")[self.selected("trainer")].info
        if modifier_info.lora_family and modifier_info.lora_family == trainer_info.lora_family:
            self.compatibility.setText(
                f"LoRA family: {trainer_info.lora_family} — selected trainer is compatible with the selected modifier."
            )
            self.compatibility.setStyleSheet("")
        else:
            self.compatibility.setText(
                f"LoRA family mismatch: trainer produces {trainer_info.lora_family or 'unknown'}, "
                f"modifier expects {modifier_info.lora_family or 'unknown'}. Train and repair with matching families."
            )
            self.compatibility.setStyleSheet("QLabel { color: #b45f06; }")

    def refresh_status(self) -> None:
        self.table.clear()
        for kind in ("segmenter", "modifier", "trainer"):
            name = self.selected(kind)
            info = plugin_classes(kind)[name].info
            if not info.model_id:
                status = "Built-in / runtime managed"
                model = "—"
            else:
                status = "Ready" if self.store.is_hf_snapshot_cached(info.model_id) else "Not downloaded"
                model = info.model_id
            self.table.addTopLevelItem(QTreeWidgetItem([kind.title(), name, model, status]))
        self.table.resizeColumnToContents(0)
        self.table.resizeColumnToContents(1)
        self.table.resizeColumnToContents(3)

    def append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def handle_model_event(self, event: ModelProgress) -> None:
        if event.stage == "model_start":
            self.progress.setRange(0, 0)
            self.progress_label.setText(f"Checking/downloading {event.repo_id}…")
            self.append_log(f"[{event.repo_id}] {event.description}")
        elif event.stage in {"file_progress", "file_complete"}:
            if event.total:
                self.progress.setRange(0, 100)
                self.progress.setValue(int((event.fraction or 0.0) * 100))
                self.progress_label.setText(
                    f"{event.repo_id}: {event.description} — {_format_progress(event)}"
                )
            else:
                self.progress.setRange(0, 0)
                self.progress_label.setText(f"{event.repo_id}: {event.description}")
        elif event.stage == "model_ready":
            self.progress.setRange(0, 100)
            self.progress.setValue(100)
            self.progress_label.setText(f"Ready: {event.repo_id}")
            self.append_log(f"Ready: {event.repo_id} -> {event.path}")
            self.refresh_status()
        elif event.stage == "model_error":
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress_label.setText(f"Failed: {event.repo_id}")
            self.append_log(f"ERROR {event.repo_id}: {event.error}")


class TrainingTab(QWidget):
    start_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.images = QListWidget()
        add = QPushButton("Add likeness references…")
        remove = QPushButton("Remove selected")
        clear = QPushButton("Clear")
        add.clicked.connect(self._add_images)
        remove.clicked.connect(lambda: _remove_selected_items(self.images))
        clear.clicked.connect(self.images.clear)
        image_buttons = QHBoxLayout()
        image_buttons.addWidget(add)
        image_buttons.addWidget(remove)
        image_buttons.addWidget(clear)
        image_buttons.addStretch(1)

        self.output = QLineEdit()
        browse_output = QPushButton("Browse…")
        browse_output.clicked.connect(self._choose_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output)
        output_row.addWidget(browse_output)
        output_widget = QWidget()
        output_widget.setLayout(output_row)

        self.trigger = QLineEdit("lf_person")
        self.face = QLineEdit("largest")
        self.background = QComboBox()
        self.background.addItems(["blur", "neutral", "black", "original"])
        self.steps = QSpinBox()
        self.steps.setRange(1, 100000)
        self.steps.setValue(800)
        self.rank = QSpinBox()
        self.rank.setRange(1, 512)
        self.rank.setValue(16)
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(8)
        self.lr.setRange(0.00000001, 1.0)
        self.lr.setSingleStep(0.00001)
        self.lr.setValue(0.0001)
        self.low_vram = QCheckBox("Use trainer low-VRAM/offload options")
        self.low_vram.setChecked(True)
        self.verbose = QCheckBox("Verbose debug: save every intermediate training crop/mask")
        self.verbose.setChecked(True)

        form = QFormLayout()
        form.addRow("Training output directory", output_widget)
        form.addRow("Unique trigger token", self.trigger)
        form.addRow("Face selector per reference", self.face)
        form.addRow("Background outside face", self.background)
        form.addRow("Training steps", self.steps)
        form.addRow("LoRA rank", self.rank)
        form.addRow("Learning rate", self.lr)
        form.addRow("", self.low_vram)
        form.addRow("", self.verbose)

        self.start = QPushButton("Prepare references and train LoRA")
        self.start.clicked.connect(self.start_requested.emit)
        self.status = QLabel("Select several clear likeness references of the same person.")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Likeness reference images"))
        layout.addWidget(self.images, 1)
        layout.addLayout(image_buttons)
        layout.addLayout(form)
        layout.addWidget(self.start)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        layout.addWidget(self.log, 1)

    def _add_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Select likeness reference images", "", IMAGE_FILTER)
        existing = {self.images.item(i).text() for i in range(self.images.count())}
        for path in files:
            if path not in existing:
                self.images.addItem(path)
                existing.add(path)

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Training output directory")
        if path:
            self.output.setText(path)

    def image_paths(self) -> list[Path]:
        return [Path(self.images.item(i).text()) for i in range(self.images.count())]


class RepairTab(QWidget):
    analyze_requested = Signal()
    repair_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = QLineEdit()
        source_browse = QPushButton("Browse…")
        source_browse.clicked.connect(self._choose_source)
        source_row = QHBoxLayout()
        source_row.addWidget(self.source)
        source_row.addWidget(source_browse)
        source_widget = QWidget()
        source_widget.setLayout(source_row)

        self.output = QLineEdit()
        output_browse = QPushButton("Browse…")
        output_browse.clicked.connect(self._choose_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output)
        output_row.addWidget(output_browse)
        output_widget = QWidget()
        output_widget.setLayout(output_row)

        self.lora = QLineEdit()
        lora_browse = QPushButton("Browse…")
        lora_browse.clicked.connect(self._choose_lora)
        lora_row = QHBoxLayout()
        lora_row.addWidget(self.lora)
        lora_row.addWidget(lora_browse)
        lora_widget = QWidget()
        lora_widget.setLayout(lora_row)

        self.references = QListWidget()
        self.references.setMaximumHeight(90)
        refs_add = QPushButton("Add identity reference…")
        refs_remove = QPushButton("Remove")
        refs_add.clicked.connect(self._add_references)
        refs_remove.clicked.connect(lambda: _remove_selected_items(self.references))
        refs_row = QHBoxLayout()
        refs_row.addWidget(refs_add)
        refs_row.addWidget(refs_remove)
        refs_row.addStretch(1)

        self.analyze = QPushButton("Detect faces")
        self.analyze.clicked.connect(self.analyze_requested.emit)
        self.face = QComboBox()
        self.face.addItem("Face 0", 0)
        face_row = QHBoxLayout()
        face_row.addWidget(self.analyze)
        face_row.addWidget(QLabel("Face to repair"))
        face_row.addWidget(self.face, 1)

        cfg = ForgeConfig()
        self.prompt = QTextEdit(cfg.prompt)
        self.prompt.setMaximumHeight(70)
        self.negative = QLineEdit(cfg.negative_prompt)
        self.lora_scale = QDoubleSpinBox()
        self.lora_scale.setRange(0.0, 3.0)
        self.lora_scale.setSingleStep(0.05)
        self.lora_scale.setValue(1.0)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(0)
        self.crop_margin = QDoubleSpinBox()
        self.crop_margin.setRange(0.05, 2.0)
        self.crop_margin.setSingleStep(0.05)
        self.crop_margin.setValue(cfg.crop_margin)
        self.feather = QSpinBox()
        self.feather.setRange(0, 512)
        self.feather.setValue(cfg.feather)
        self.steps = QSpinBox()
        self.steps.setRange(1, 500)
        self.steps.setValue(24)
        self.guidance = QDoubleSpinBox()
        self.guidance.setRange(0.0, 30.0)
        self.guidance.setSingleStep(0.1)
        self.guidance.setValue(6.0)
        self.strength = QDoubleSpinBox()
        self.strength.setRange(0.0, 1.0)
        self.strength.setSingleStep(0.01)
        self.strength.setValue(0.58)
        self.verbose = QCheckBox("Verbose debug: save every intermediate mask/crop/generated image")
        self.verbose.setChecked(True)

        form = QFormLayout()
        form.addRow("Original photograph", source_widget)
        form.addRow("Output photograph", output_widget)
        form.addRow("Identity LoRA", lora_widget)
        form.addRow("Prompt", self.prompt)
        form.addRow("Negative prompt", self.negative)
        form.addRow("LoRA scale", self.lora_scale)
        form.addRow("Seed", self.seed)
        form.addRow("Crop margin", self.crop_margin)
        form.addRow("Composite feather", self.feather)
        form.addRow("Inference steps", self.steps)
        form.addRow("Guidance", self.guidance)
        form.addRow("Inpaint strength", self.strength)
        form.addRow("", self.verbose)

        self.input_preview = ImagePreview("Select an original photograph")
        self.output_preview = ImagePreview("Repaired result")
        previews = QHBoxLayout()
        previews.addWidget(self.input_preview, 1)
        previews.addWidget(self.output_preview, 1)

        self.repair = QPushButton("Improve facial likeness")
        self.repair.clicked.connect(self.repair_requested.emit)
        self.status = QLabel("Choose an image, detect faces, select the target face, then repair it.")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1500)
        self.log.setMaximumHeight(160)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("Optional identity reference images (used by supporting modifier backends)"))
        layout.addWidget(self.references)
        layout.addLayout(refs_row)
        layout.addLayout(face_row)
        layout.addLayout(previews, 1)
        layout.addWidget(self.repair)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        layout.addWidget(self.log)

    def _choose_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Original photograph", "", IMAGE_FILTER)
        if path:
            self.source.setText(path)
            source = Path(path)
            self.output.setText(str(source.with_name(f"{source.stem}-repaired{source.suffix or '.png'}")))
            try:
                self.input_preview.set_pil(Image.open(path).convert("RGB"))
            except Exception:
                pass

    def _choose_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Output photograph", self.output.text(), IMAGE_FILTER)
        if path:
            self.output.setText(path)

    def _choose_lora(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Identity LoRA", "", "LoRA weights (*.safetensors);;All files (*)")
        if path:
            self.lora.setText(path)

    def _add_references(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Identity reference images", "", IMAGE_FILTER)
        existing = {self.references.item(i).text() for i in range(self.references.count())}
        for path in files:
            if path not in existing:
                self.references.addItem(path)
                existing.add(path)

    def reference_paths(self) -> list[Path]:
        return [Path(self.references.item(i).text()) for i in range(self.references.count())]


class LikenessForgeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LikenessForge")
        self.resize(1180, 920)
        self._workers: set[TaskThread] = set()

        root = QWidget()
        layout = QVBoxLayout(root)
        self.models = ModelSetupWidget()
        self.tabs = QTabWidget()
        self.training = TrainingTab()
        self.repair = RepairTab()
        self.tabs.addTab(self.training, "Train likeness LoRA")
        self.tabs.addTab(self.repair, "Repair photograph")
        layout.addWidget(self.models)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(root)

        self.models.download_requested.connect(self.download_selected_models)
        self.training.start_requested.connect(self.start_training)
        self.repair.analyze_requested.connect(self.analyze_faces)
        self.repair.repair_requested.connect(self.start_repair)

    def _start_worker(self, function, on_result=None, on_error=None, message_target=None) -> TaskThread:
        worker = TaskThread(function, self)
        self._workers.add(worker)
        worker.model_event.connect(self.models.handle_model_event)
        if message_target:
            worker.message.connect(message_target.appendPlainText)
        if on_result:
            worker.result.connect(on_result)
        if on_error:
            worker.failed.connect(on_error)
        else:
            worker.failed.connect(self._show_error)
        worker.finished.connect(lambda w=worker: self._workers.discard(w))
        worker.finished.connect(worker.deleteLater)
        worker.start()
        return worker

    def _show_error(self, traceback_text: str) -> None:
        LOG.error("GUI task failed:\n%s", traceback_text)
        QMessageBox.critical(self, "LikenessForge error", traceback_text)

    def download_selected_models(self) -> None:
        models = self.models.selected_model_plugins()
        if not models:
            self.models.append_log("Selected plugins do not require Hugging Face model downloads.")
            return
        self.models.download.setEnabled(False)

        def task(worker: TaskThread):
            store = ModelStore(progress_callback=worker.model_event.emit)
            ready = []
            for kind, name, repo_id in models:
                worker.report(f"Preparing {kind} {name}: {repo_id}")
                store.ensure_hf_snapshot(repo_id)
                ready.append(repo_id)
            return ready

        def done(ready):
            self.models.download.setEnabled(True)
            self.models.append_log(f"Selected model setup complete ({len(ready)} repositories).")
            self.models.refresh_status()

        def failed(text):
            self.models.download.setEnabled(True)
            self._show_error(text)

        self._start_worker(task, done, failed, self.models.log)

    def _debug_callback(self, worker: TaskThread):
        def callback(record: dict[str, Any]) -> None:
            name = record.get("step", "step")
            details = {k: v for k, v in record.items() if k != "step"}
            worker.report(f"{name}: {json.dumps(details, default=str)}")
        return callback

    def start_training(self) -> None:
        sources = self.training.image_paths()
        output = Path(self.training.output.text()).expanduser() if self.training.output.text() else None
        trigger = self.training.trigger.text().strip()
        if not sources:
            QMessageBox.warning(self, "Training references", "Add at least one likeness reference image.")
            return
        if output is None:
            QMessageBox.warning(self, "Training output", "Choose an output directory.")
            return
        if not trigger:
            QMessageBox.warning(self, "Trigger token", "Enter a unique LoRA trigger token.")
            return

        detector_name = self.models.selected("detector")
        segmenter_name = self.models.selected("segmenter")
        trainer_name = self.models.selected("trainer")
        selector = self.training.face.text().strip() or "largest"
        background = self.training.background.currentText()
        steps = self.training.steps.value()
        rank = self.training.rank.value()
        learning_rate = self.training.lr.value()
        low_vram_training = self.training.low_vram.isChecked()
        verbose = self.training.verbose.isChecked()
        self.training.start.setEnabled(False)
        self.training.progress.setRange(0, 0)
        self.training.status.setText("Preparing training references…")
        self.training.log.clear()

        def task(worker: TaskThread):
            store = ModelStore(progress_callback=worker.model_event.emit)
            detector = create("detector", detector_name, store)
            segmenter = create("segmenter", segmenter_name, store)
            trainer = create("trainer", trainer_name, store, progress_callback=worker.report)
            debug_root = output / "debug"
            debug = DebugSink(
                verbose,
                debug_root,
                event_callback=self._debug_callback(worker),
            )
            workflow = TrainingWorkflow(detector, segmenter, trainer, debug)
            return workflow.train(
                sources,
                output,
                trigger,
                selector=selector,
                background=background,
                steps=steps,
                rank=rank,
                lr=learning_rate,
                low_vram=low_vram_training,
            )

        def done(weights):
            self.training.start.setEnabled(True)
            self.training.progress.setRange(0, 100)
            self.training.progress.setValue(100)
            self.training.status.setText(f"Training complete: {weights}")
            self.training.log.appendPlainText(f"LoRA ready: {weights}")
            self.repair.lora.setText(str(weights))
            self.tabs.setCurrentWidget(self.repair)

        def failed(text):
            self.training.start.setEnabled(True)
            self.training.progress.setRange(0, 100)
            self.training.progress.setValue(0)
            self.training.status.setText("Training failed")
            self.training.log.appendPlainText(text)
            self._show_error(text)

        self._start_worker(task, done, failed, self.training.log)

    def analyze_faces(self) -> None:
        source = Path(self.repair.source.text()).expanduser() if self.repair.source.text() else None
        if source is None or not source.is_file():
            QMessageBox.warning(self, "Original photograph", "Choose an existing photograph first.")
            return
        detector_name = self.models.selected("detector")
        self.repair.analyze.setEnabled(False)
        self.repair.status.setText("Detecting faces…")

        def task(worker: TaskThread):
            store = ModelStore(progress_callback=worker.model_event.emit)
            detector = create("detector", detector_name, store)
            image = Image.open(source).convert("RGB")
            workflow = RepairWorkflow(detector, None, None, DebugSink(False, event_callback=self._debug_callback(worker)))
            faces = workflow.inspect_faces(image)
            return faces, annotate_faces(image, faces)

        def done(result):
            faces, preview = result
            self.repair.analyze.setEnabled(True)
            self.repair.face.clear()
            for face in faces:
                self.repair.face.addItem(
                    f"Face {face.index} — score {face.score:.3f} — {face.box.width}×{face.box.height}",
                    face.index,
                )
            self.repair.input_preview.set_pil(preview)
            if faces:
                self.repair.status.setText(f"Found {len(faces)} face(s). Select the one to repair.")
            else:
                self.repair.status.setText("No faces detected.")

        def failed(text):
            self.repair.analyze.setEnabled(True)
            self.repair.status.setText("Face detection failed")
            self._show_error(text)

        self._start_worker(task, done, failed, self.repair.log)

    def _warn_lora_mismatch(self) -> bool:
        path = self.repair.lora.text().strip()
        if not path:
            return True
        metadata = _load_training_metadata(path)
        if not metadata:
            return True
        expected = plugin_classes("modifier")[self.models.selected("modifier")].info.lora_family
        actual = metadata.get("lora_family")
        if expected and actual and expected != actual:
            answer = QMessageBox.warning(
                self,
                "LoRA family mismatch",
                f"This LoRA was trained for {actual}, but the selected modifier expects {expected}. "
                "The weights are not architecture-compatible. Continue anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            return answer == QMessageBox.Yes
        return True

    def start_repair(self) -> None:
        source = Path(self.repair.source.text()).expanduser() if self.repair.source.text() else None
        output = Path(self.repair.output.text()).expanduser() if self.repair.output.text() else None
        if source is None or not source.is_file():
            QMessageBox.warning(self, "Original photograph", "Choose an existing photograph first.")
            return
        if output is None:
            QMessageBox.warning(self, "Output photograph", "Choose an output path.")
            return
        if not self._warn_lora_mismatch():
            return

        detector_name = self.models.selected("detector")
        segmenter_name = self.models.selected("segmenter")
        modifier_name = self.models.selected("modifier")
        low_vram = self.models.low_vram.currentText()
        face_selector = self.repair.face.currentData(Qt.UserRole)
        if face_selector is None:
            face_selector = 0
        references = self.repair.reference_paths()
        lora = self.repair.lora.text().strip() or None
        prompt = self.repair.prompt.toPlainText().strip()
        negative_prompt = self.repair.negative.text().strip() or None
        crop_margin = self.repair.crop_margin.value()
        feather = self.repair.feather.value()
        seed = self.repair.seed.value()
        lora_scale = self.repair.lora_scale.value()
        steps = self.repair.steps.value()
        guidance = self.repair.guidance.value()
        strength = self.repair.strength.value()
        verbose = self.repair.verbose.isChecked()

        self.repair.repair.setEnabled(False)
        self.repair.progress.setRange(0, 0)
        self.repair.status.setText("Repairing likeness…")
        self.repair.log.clear()

        def task(worker: TaskThread):
            store = ModelStore(progress_callback=worker.model_event.emit)
            detector = create("detector", detector_name, store)
            segmenter = create("segmenter", segmenter_name, store)
            modifier = create("modifier", modifier_name, store, low_vram=low_vram)
            image = Image.open(source).convert("RGB")
            refs = [Image.open(path).convert("RGB") for path in references]
            debug_root = output.parent / f"{output.stem}-debug"
            debug = DebugSink(
                verbose,
                debug_root,
                event_callback=self._debug_callback(worker),
            )
            workflow = RepairWorkflow(detector, segmenter, modifier, debug)
            result = workflow.repair(
                image=image,
                face_selector=face_selector,
                lora=lora,
                prompt=prompt,
                negative_prompt=negative_prompt,
                crop_margin=crop_margin,
                feather=feather,
                seed=seed,
                lora_scale=lora_scale,
                reference_images=refs,
                modifier_options={
                    "steps": steps,
                    "guidance": guidance,
                    "strength": strength,
                },
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            result.save(output)
            return output, result

        def done(result):
            path, image = result
            self.repair.repair.setEnabled(True)
            self.repair.progress.setRange(0, 100)
            self.repair.progress.setValue(100)
            self.repair.status.setText(f"Repair complete: {path}")
            self.repair.output_preview.set_pil(image)
            self.repair.log.appendPlainText(f"Saved: {path}")

        def failed(text):
            self.repair.repair.setEnabled(True)
            self.repair.progress.setRange(0, 100)
            self.repair.progress.setValue(0)
            self.repair.status.setText("Repair failed")
            self.repair.log.appendPlainText(text)
            self._show_error(text)

        self._start_worker(task, done, failed, self.repair.log)


def main(argv=None) -> int:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = QApplication(list(sys.argv if argv is None else [sys.argv[0], *argv]))
    app.setApplicationName("LikenessForge")
    app.setOrganizationName("Kieran Simkin")
    window = LikenessForgeWindow()
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
