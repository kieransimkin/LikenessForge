from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from likenessforge.model_store import ModelStore
from likenessforge.types import FaceCandidate, ModifyRequest, TrainingRequest


@dataclass(frozen=True)
class PluginInfo:
    name: str
    description: str
    model_id: str | None = None
    preferred_size: int | None = None
    low_vram: bool = False
    license_note: str | None = None
    lora_family: str | None = None


class Plugin(ABC):
    info: PluginInfo

    def __init__(self, store: ModelStore, **options: Any):
        self.store = store
        self.options = options


class FaceDetector(Plugin):
    @abstractmethod
    def detect(self, image: Image.Image) -> list[FaceCandidate]: ...


class FaceSegmenter(Plugin):
    @abstractmethod
    def segment(self, image: Image.Image, face: FaceCandidate) -> Image.Image: ...


class ImageModifier(Plugin):
    @property
    def preferred_size(self) -> int:
        return self.info.preferred_size or 512

    @abstractmethod
    def modify(self, request: ModifyRequest) -> Image.Image: ...


class LoraTrainer(Plugin):
    @property
    def preferred_size(self) -> int:
        return self.info.preferred_size or 512

    @abstractmethod
    def train(self, request: TrainingRequest) -> Path: ...
