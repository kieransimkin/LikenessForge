from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import requests

from likenessforge.plugins.base import LoraTrainer, PluginInfo


LOG = logging.getLogger("likenessforge")
DIFFUSERS_REF = "v0.41.0"
RAW_BASE = f"https://raw.githubusercontent.com/huggingface/diffusers/{DIFFUSERS_REF}/examples/dreambooth"


class _DiffusersScriptTrainer(LoraTrainer):
    script_name: str
    base_model_id: str

    @property
    def _progress_callback(self) -> Callable[[str], None] | None:
        callback = self.options.get("progress_callback")
        return callback if callable(callback) else None

    def _emit(self, message: str) -> None:
        LOG.info("%s", message)
        if self._progress_callback:
            self._progress_callback(message)

    def _ensure_script(self) -> Path:
        dest = self.store.tools / "diffusers" / DIFFUSERS_REF / self.script_name
        if dest.exists():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{RAW_BASE}/{self.script_name}"
        self._emit(f"Downloading official Diffusers trainer: {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest.write_bytes(response.content)
        self._emit(f"Trainer script ready: {dest}")
        return dest

    def _base_args(self, request):
        return [
            "--pretrained_model_name_or_path", self.base_model_id,
            "--cache_dir", str(self.store.hf),
            "--instance_data_dir", str(request.dataset_dir),
            "--output_dir", str(request.output_dir),
            "--instance_prompt", request.instance_prompt,
            "--resolution", str(request.resolution),
            "--train_batch_size", "1",
            "--gradient_accumulation_steps", str(request.gradient_accumulation_steps),
            "--learning_rate", str(request.learning_rate),
            "--lr_scheduler", "constant",
            "--lr_warmup_steps", "0",
            "--max_train_steps", str(request.max_train_steps),
            "--rank", str(request.rank),
            "--mixed_precision", request.mixed_precision,
            "--checkpointing_steps", str(max(100, request.max_train_steps // 4)),
            "--gradient_checkpointing",
        ]

    def train(self, request):
        request.output_dir.mkdir(parents=True, exist_ok=True)
        self.store.ensure_hf_snapshot(self.base_model_id)
        script = self._ensure_script()
        cmd = [sys.executable, "-m", "accelerate.commands.launch", str(script), *self._base_args(request)]
        if request.low_vram:
            cmd.extend(self.low_vram_args())
        cmd.extend(request.extra_args)
        self._emit("Trainer command: " + " ".join(map(str, cmd)))

        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if line:
                self._emit(line)
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)

        weights = list(request.output_dir.glob("*.safetensors"))
        if not weights:
            weights = list(request.output_dir.rglob("*.safetensors"))
        if not weights:
            raise RuntimeError(f"Training completed but no .safetensors weights were found in {request.output_dir}")
        result = max(weights, key=lambda p: p.stat().st_mtime)
        self._emit(f"Training complete: {result}")
        return result

    def low_vram_args(self):
        return ["--use_8bit_adam"]


class SD15LoraTrainer(_DiffusersScriptTrainer):
    info = PluginInfo(
        name="sd15-lora",
        description="Official Diffusers DreamBooth-LoRA trainer for Stable Diffusion 1.x.",
        model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        preferred_size=512,
        low_vram=True,
        lora_family="sd15",
    )
    script_name = "train_dreambooth_lora.py"
    base_model_id = info.model_id


class SDXLLoraTrainer(_DiffusersScriptTrainer):
    info = PluginInfo(
        name="sdxl-lora",
        description="Official Diffusers DreamBooth-LoRA trainer for SDXL.",
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        preferred_size=768,
        low_vram=True,
        lora_family="sdxl",
    )
    script_name = "train_dreambooth_lora_sdxl.py"
    base_model_id = info.model_id

    def low_vram_args(self):
        return ["--use_8bit_adam", "--pre_compute_text_embeddings"]


class Flux2KleinLoraTrainer(_DiffusersScriptTrainer):
    info = PluginInfo(
        name="flux2-klein-lora",
        description="Official Diffusers DreamBooth-LoRA trainer for FLUX.2 Klein 4B Base.",
        model_id="black-forest-labs/FLUX.2-klein-base-4B",
        preferred_size=768,
        low_vram=False,
        license_note="BFL documents ~24 GB VRAM for typical 4B LoRA training; quantized training is available.",
        lora_family="flux2-klein",
    )
    script_name = "train_dreambooth_lora_flux2_klein.py"
    base_model_id = info.model_id

    def low_vram_args(self):
        # The official trainer supports a bitsandbytes DiT quantization config; users may supply a
        # custom JSON through --extra-arg. FP8 is hardware-specific, so it is not forced here.
        return ["--offload"]
