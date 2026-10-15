import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image
from PySide6.QtWidgets import QApplication

from likenessforge.gui.app import (
    DropFileLineEdit,
    DropFileListWidget,
    LikenessForgeWindow,
)


def test_gui_exposes_model_and_workflow_controls():
    app = QApplication.instance() or QApplication([])
    window = LikenessForgeWindow()
    try:
        assert window.models.segmenter.count() >= 2
        assert window.models.modifier.count() >= 3
        assert window.models.trainer.count() >= 3
        assert window.tabs.count() == 4
        assert window.tabs.tabText(0) == "Train likeness LoRA"
        assert window.tabs.tabText(1) == "Training wizard"
        assert window.tabs.tabText(2) == "Repair photograph"
        assert window.tabs.tabText(3) == "Repair wizard"
        assert window.training_wizard.preset.count() == 3
        assert window.wizard.preset.count() == 3
    finally:
        window.close()
        app.processEvents()


def test_all_four_workflows_support_drop_targets_and_routing(tmp_path):
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "source.png"
    reference = tmp_path / "reference.jpg"
    validation = tmp_path / "validation.webp"
    lora = tmp_path / "identity.safetensors"
    Image.new("RGB", (64, 64), (20, 30, 40)).save(source)
    Image.new("RGB", (64, 64), (40, 50, 60)).save(reference)
    Image.new("RGB", (64, 64), (60, 70, 80)).save(validation)
    lora.write_bytes(b"test")

    window = LikenessForgeWindow()
    try:
        assert window.acceptDrops()
        assert isinstance(window.training.images, DropFileListWidget)
        assert window.training.images.acceptDrops()
        assert isinstance(window.training_wizard.images, DropFileListWidget)
        assert isinstance(window.training_wizard.validation, DropFileLineEdit)
        assert window.training_wizard.validation.acceptDrops()
        assert isinstance(window.repair.source, DropFileLineEdit)
        assert isinstance(window.repair.lora, DropFileLineEdit)
        assert isinstance(window.repair.references, DropFileListWidget)
        assert isinstance(window.wizard.source, DropFileLineEdit)
        assert isinstance(window.wizard.lora, DropFileLineEdit)
        assert isinstance(window.wizard.references, DropFileListWidget)

        assert window.training.handle_dropped_files([source, reference])
        assert window.training.images.count() == 2

        assert window.training_wizard.handle_dropped_files([validation])
        assert Path(window.training_wizard.validation.text()) == validation.resolve()
        window.training_wizard.add_image_paths([source, reference])
        assert window.training_wizard.images.count() == 2

        assert window.repair.handle_dropped_files([source, reference, lora])
        assert Path(window.repair.source.text()) == source.resolve()
        assert Path(window.repair.lora.text()) == lora.resolve()
        assert window.repair.references.count() == 1
        assert Path(window.repair.references.item(0).text()) == reference.resolve()

        assert window.wizard.handle_dropped_files([source, reference, lora])
        assert Path(window.wizard.source.text()) == source.resolve()
        assert Path(window.wizard.lora.text()) == lora.resolve()
        assert window.wizard.references.count() == 1
        assert Path(window.wizard.references.item(0).text()) == reference.resolve()
    finally:
        window.close()
        app.processEvents()
