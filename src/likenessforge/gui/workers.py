from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QThread, Signal


class TaskThread(QThread):
    """Run one blocking LikenessForge task without freezing the Qt event loop."""

    result = Signal(object)
    failed = Signal(str)
    message = Signal(str)
    model_event = Signal(object)

    def __init__(self, function: Callable[["TaskThread"], Any], parent=None):
        super().__init__(parent)
        self.function = function

    def report(self, message: str) -> None:
        self.message.emit(str(message))

    def run(self) -> None:
        try:
            self.result.emit(self.function(self))
        except Exception:
            self.failed.emit(traceback.format_exc())
