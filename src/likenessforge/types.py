from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def clamp(self, width: int, height: int) -> "Box":
        return Box(
            max(0, min(width, self.x1)),
            max(0, min(height, self.y1)),
            max(0, min(width, self.x2)),
            max(0, min(height, self.y2)),
        )

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class FaceCandidate:
    index: int
    box: Box
    score: float = 1.0
    landmarks: list[tuple[float, float]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedFace:
    candidate: FaceCandidate
    crop_box: Box
    crop_image: Image.Image
    crop_mask: Image.Image
    target_size: int


@dataclass
class ModifyRequest:
    image: Image.Image
    mask: Image.Image
    prompt: str
    negative_prompt: str | None = None
    lora: str | Path | None = None
    lora_scale: float = 1.0
    seed: int = 0
    reference_images: list[Image.Image] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingRequest:
    dataset_dir: Path
    output_dir: Path
    instance_prompt: str
    resolution: int
    max_train_steps: int = 800
    rank: int = 16
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "fp16"
    low_vram: bool = False
    extra_args: list[str] = field(default_factory=list)


MaskArray = np.ndarray
