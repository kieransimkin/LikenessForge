from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from typing import Any

from likenessforge.model_store import ModelStore


KINDS = {
    "detector": "likenessforge.detectors",
    "segmenter": "likenessforge.segmenters",
    "modifier": "likenessforge.modifiers",
    "trainer": "likenessforge.trainers",
}


def _builtins() -> dict[str, dict[str, type]]:
    from likenessforge.plugins.detectors.mtcnn import MTCNNDetector
    from likenessforge.plugins.detectors.opencv_haar import OpenCVHaarDetector
    from likenessforge.plugins.modifiers.diffusers_inpaint import SD15InpaintModifier, SDXLInpaintModifier
    from likenessforge.plugins.modifiers.flux2_klein import Flux2KleinModifier, Flux2KleinLowVRAMModifier
    from likenessforge.plugins.segmenters.sam2 import SAM2TinySegmenter
    from likenessforge.plugins.segmenters.soft_face import SoftFaceSegmenter
    from likenessforge.plugins.trainers.diffusers_scripts import (
        Flux2KleinLoraTrainer,
        SD15LoraTrainer,
        SDXLLoraTrainer,
    )

    return {
        "detector": {
            OpenCVHaarDetector.info.name: OpenCVHaarDetector,
            MTCNNDetector.info.name: MTCNNDetector,
        },
        "segmenter": {
            SoftFaceSegmenter.info.name: SoftFaceSegmenter,
            SAM2TinySegmenter.info.name: SAM2TinySegmenter,
        },
        "modifier": {
            SD15InpaintModifier.info.name: SD15InpaintModifier,
            SDXLInpaintModifier.info.name: SDXLInpaintModifier,
            Flux2KleinModifier.info.name: Flux2KleinModifier,
            Flux2KleinLowVRAMModifier.info.name: Flux2KleinLowVRAMModifier,
        },
        "trainer": {
            SD15LoraTrainer.info.name: SD15LoraTrainer,
            SDXLLoraTrainer.info.name: SDXLLoraTrainer,
            Flux2KleinLoraTrainer.info.name: Flux2KleinLoraTrainer,
        },
    }


def plugin_classes(kind: str) -> dict[str, type]:
    result = dict(_builtins()[kind])
    group = KINDS[kind]
    try:
        eps = entry_points(group=group)
    except TypeError:
        eps = entry_points().get(group, [])
    for ep in eps:
        result[ep.name] = ep.load()
    return result


def resolve_class(kind: str, name: str) -> type:
    classes = plugin_classes(kind)
    if name in classes:
        return classes[name]
    if ":" in name:
        module_name, class_name = name.split(":", 1)
        return getattr(importlib.import_module(module_name), class_name)
    raise KeyError(f"Unknown {kind} plugin {name!r}. Available: {', '.join(sorted(classes))}")


def create(kind: str, name: str, store: ModelStore, **options: Any):
    cls = resolve_class(kind, name)
    return cls(store=store, **options)


def describe_plugins() -> list[dict[str, Any]]:
    rows = []
    for kind in KINDS:
        for name, cls in sorted(plugin_classes(kind).items()):
            info = cls.info
            rows.append(
                {
                    "kind": kind,
                    "name": name,
                    "description": info.description,
                    "model_id": info.model_id,
                    "preferred_size": info.preferred_size,
                    "low_vram": info.low_vram,
                    "lora_family": info.lora_family,
                }
            )
    return rows
