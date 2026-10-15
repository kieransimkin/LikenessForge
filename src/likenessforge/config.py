from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PluginSelection:
    name: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForgeConfig:
    detector: PluginSelection = field(default_factory=lambda: PluginSelection("opencv-haar"))
    segmenter: PluginSelection = field(default_factory=lambda: PluginSelection("soft-face"))
    modifier: PluginSelection = field(default_factory=lambda: PluginSelection("sd15-inpaint-lowvram"))
    trainer: PluginSelection = field(default_factory=lambda: PluginSelection("sd15-lora"))
    cache_dir: str | None = None
    low_vram: str = "auto"
    crop_margin: float = 0.55
    feather: int = 32
    blend: str = "alpha"
    prompt: str = (
        "repair only the face so it matches the trained identity; preserve expression, pose, age, "
        "lighting, skin texture, hair, camera perspective and surrounding image"
    )
    negative_prompt: str = "different person, distorted face, asymmetry, extra eyes, waxy skin"


def _selection(value: Any, default: str) -> PluginSelection:
    if value is None:
        return PluginSelection(default)
    if isinstance(value, str):
        return PluginSelection(value)
    if isinstance(value, dict):
        return PluginSelection(value.get("name", default), dict(value.get("options", {})))
    raise TypeError(f"Invalid plugin selection: {value!r}")


def load_config(path: str | Path | None) -> ForgeConfig:
    cfg = ForgeConfig()
    if not path:
        return cfg
    p = Path(path)
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    elif p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
    elif p.suffix.lower() == ".toml":
        import tomllib

        data = tomllib.loads(p.read_text(encoding="utf-8"))
    else:
        raise ValueError("Config must be .yaml, .yml, .json or .toml")
    plugins = data.get("plugins", data)
    cfg.detector = _selection(plugins.get("detector"), cfg.detector.name)
    cfg.segmenter = _selection(plugins.get("segmenter"), cfg.segmenter.name)
    cfg.modifier = _selection(plugins.get("modifier"), cfg.modifier.name)
    cfg.trainer = _selection(plugins.get("trainer"), cfg.trainer.name)
    for key in ("cache_dir", "low_vram", "crop_margin", "feather", "blend", "prompt", "negative_prompt"):
        if key in data:
            setattr(cfg, key, data[key])
    return cfg
