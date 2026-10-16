from __future__ import annotations

import logging

from likenessforge.plugins.base import ImageModifier, PluginInfo

LOG = logging.getLogger("likenessforge")


class _DiffusersInpaint(ImageModifier):
    def _load(self):
        cached = getattr(self, "_pipeline_cache", None)
        if cached is not None:
            return cached
        try:
            import torch
            from diffusers import AutoPipelineForInpainting
        except ImportError as exc:
            raise RuntimeError("Diffusion modifiers require `pip install likenessforge[diffusion]`") from exc
        device = self.options.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        variant = self.options.get("variant", "fp16" if device == "cuda" else None)
        model_dir = self.store.ensure_diffusers_snapshot(
            self.info.model_id, revision=self.options.get("revision"), variant=variant,
        )
        pipe = AutoPipelineForInpainting.from_pretrained(model_dir, torch_dtype=dtype, variant=variant)
        low_vram = self.options.get("low_vram", "auto")
        if device == "cuda" and low_vram in {True, "auto", "cpu_offload"}:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        self._pipeline_cache = (pipe, torch, device)
        return self._pipeline_cache

    def _configure_lora(self, pipe, request) -> None:
        if not request.lora:
            return
        lora_path = self.store.resolve_local_or_hf(request.lora)
        lora_key = str(lora_path)
        if getattr(self, "_loaded_lora", None) != lora_key:
            LOG.info("loading LoRA: %s", request.lora)
            if getattr(self, "_loaded_lora", None) and hasattr(pipe, "unload_lora_weights"):
                pipe.unload_lora_weights()
            pipe.load_lora_weights(lora_key, adapter_name="identity")
            self._loaded_lora = lora_key
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters(["identity"], adapter_weights=[float(request.lora_scale)])

    def modify(self, request):
        pipe, torch, device = self._load()
        self._configure_lora(pipe, request)
        gen_device = device if device != "cuda" or not hasattr(pipe, "_execution_device") else pipe._execution_device
        generator = torch.Generator(device=str(gen_device)).manual_seed(int(request.seed))
        kwargs = {
            "prompt": request.prompt,
            "image": request.image,
            "mask_image": request.mask,
            "generator": generator,
            "num_inference_steps": int(request.options.get("steps", self.options.get("steps", 24))),
            "guidance_scale": float(request.options.get("guidance", self.options.get("guidance", 6.0))),
            "strength": float(request.options.get("strength", self.options.get("strength", 0.58))),
        }
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
        result = pipe(**kwargs).images[0]
        return result.convert("RGB")


class SD15InpaintModifier(_DiffusersInpaint):
    info = PluginInfo(
        name="sd15-inpaint-lowvram",
        description="Stable Diffusion 1.5 inpainting with CPU offload; a hardware-dependent lower-memory starting point.",
        model_id="stable-diffusion-v1-5/stable-diffusion-inpainting",
        preferred_size=512,
        low_vram=True,
        license_note="CreativeML OpenRAIL-M; see model card.",
        lora_family="sd15",
    )


class SDXLInpaintModifier(_DiffusersInpaint):
    info = PluginInfo(
        name="sdxl-inpaint",
        description="SDXL inpainting with CPU offload/tiling support for constrained GPUs.",
        model_id="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        preferred_size=1024,
        low_vram=True,
        license_note="OpenRAIL++; see model card.",
        lora_family="sdxl",
    )
