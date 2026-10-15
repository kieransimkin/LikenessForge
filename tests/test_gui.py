import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from likenessforge.gui.app import LikenessForgeWindow


def test_gui_exposes_model_and_workflow_controls():
    app = QApplication.instance() or QApplication([])
    window = LikenessForgeWindow()
    try:
        assert window.models.segmenter.count() >= 2
        assert window.models.modifier.count() >= 3
        assert window.models.trainer.count() >= 3
        assert window.tabs.count() == 2
        assert window.tabs.tabText(0) == "Train likeness LoRA"
        assert window.tabs.tabText(1) == "Repair photograph"
    finally:
        window.close()
        app.processEvents()
