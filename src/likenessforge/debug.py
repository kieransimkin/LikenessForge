from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


LOG = logging.getLogger("likenessforge")


class DebugSink:
    def __init__(self, enabled: bool = False, root: str | Path | None = None):
        self.enabled = enabled
        self.counter = 0
        self.records: list[dict[str, Any]] = []
        if enabled:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.root = Path(root or "likenessforge-debug") / stamp
            self.root.mkdir(parents=True, exist_ok=True)
        else:
            self.root = None

    def step(self, name: str, **details: Any) -> None:
        record = {"step": name, **details}
        self.records.append(record)
        LOG.debug("%s | %s", name, json.dumps(details, default=str, sort_keys=True))
        if self.enabled and self.root:
            (self.root / "trace.json").write_text(
                json.dumps(self.records, indent=2, default=str), encoding="utf-8"
            )

    def image(self, name: str, image: Image.Image) -> Path | None:
        if not self.enabled or not self.root:
            return None
        self.counter += 1
        path = self.root / f"{self.counter:03d}-{name}.png"
        image.save(path)
        LOG.debug("saved intermediate image: %s", path)
        return path
