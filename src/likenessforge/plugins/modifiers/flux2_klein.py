from __future__ import annotations

import logging

from likenessforge.plugins.base import ImageModifier, PluginInfo

LOG = logging.getLogger("likenessforge")


class Flux2KleinModifier(ImageModifier):
    info = PluginInfo(
        name="flux2-klein-4b",
        description="FLUX.2 Klein 4B unified image editor with LoRA support.",
        model_id="black-forest-labs/FLUX.2-klein-4B",
        preferred_size=1024,
        low_vram=False,
        license_note="Apache-2.0 for the 4B checkpoint.",
        lora_family="flux2-klein",
    )

    def _load(self, low_vram: bool = False):
        cache_key = bool(low_vram)
        cache = getattr(self, "_pipeline_cache", {})
        if cache_key in cache:
            return cache[cache_key]
        try:
            import torch
            from diffusers import Flux2KleinPipeline
        except ImportError as exc:
            raise RuntimeError("FLUX.2 requires `pip install likenessforge[diffusion]`") from exc
        model_dir = self.store.ensure_hf_snapshot(self.info.model_id)
        device = self.options.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        kwargs = {"torch_dtype": dtype}
        quantized = False
        if low_vram and device == "cuda":
            try:
                from diffusers import PipelineQuantizationConfig

                kwargs["quantization_config"] = PipelineQuantizationConfig(
                    quant_backend="bitsandbytes_4bit",
                    components_to_quantize=["transformer", "text_encoder"],
                )
                quantized = True
                LOG.info("using bitsandbytes 4-bit quantization for FLUX.2 Klein")
            except Exception as exc:  # noqa: BLE001 - optional quantization backend failures must fall back
                LOG.warning("4-bit pipeline quantization unavailable (%s); falling back to CPU offload", exc)
        try:
            pipe = Flux2KleinPipeline.from_pretrained(model_dir, **kwargs)
        except Exception as exc:
            if not quantized:
                raise
            LOG.warning("quantized Klein load failed (%s); retrying unquantized with CPU offload", exc)
            pipe = Flux2KleinPipeline.from_pretrained(model_dir, torch_dtype=dtype)
        if device == "cuda" and low_vram:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        cache[cache_key] = (pipe, torch, device)
        self._pipeline_cache = cache
        return cache[cache_key]

    def _wants_low_vram(self):
        mode = self.options.get("low_vram", "off")
        if mode in {True, "bnb4", "cpu_offload"}:
            return True
        if mode in {False, None, "off"}:
            return False
        if mode == "auto":
            try:
                import torch
                if not torch.cuda.is_available():
                    return False
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                return total_gb < 13.0
            except Exception as exc:  # noqa: BLE001 - GPU capability probing is best-effort
                LOG.debug("unable to determine GPU VRAM for auto mode: %s", exc)
                return False
        return bool(mode)

    def _configure_lora(self, pipe, request) -> None:
        if not request.lora:
            return
        lora_path = self.store.resolve_local_or_hf(request.lora)
        lora_key = str(lora_path)
        loaded = getattr(self, "_loaded_lora", {})
        pipe_key = id(pipe)
        if loaded.get(pipe_key) != lora_key:
            LOG.info("loading LoRA: %s", request.lora)
            if loaded.get(pipe_key) and hasattr(pipe, "unload_lora_weights"):
                pipe.unload_lora_weights()
            pipe.load_lora_weights(lora_key, adapter_name="identity")
            loaded[pipe_key] = lora_key
            self._loaded_lora = loaded
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters(["identity"], adapter_weights=[float(request.lora_scale)])

    def modify(self, request):
        pipe, torch, _device = self._load(low_vram=self._wants_low_vram())
        self._configure_lora(pipe, request)
        images = [request.image] + list(request.reference_images)
        image_arg = images[0] if len(images) == 1 else images
        generator = torch.Generator(device="cpu").manual_seed(int(request.seed))
        result = pipe(
            prompt=request.prompt,
            image=image_arg,
            generator=generator,
            num_inference_steps=int(request.options.get("steps", self.options.get("steps", 4))),
            guidance_scale=float(request.options.get("guidance", self.options.get("guidance", 4.0))),
            height=request.image.height,
            width=request.image.width,
        ).images[0]
        return result.convert("RGB")


class Flux2KleinLowVRAMModifier(Flux2KleinModifier):
    info = PluginInfo(
        name="flux2-klein-4b-lowvram",
        description="FLUX.2 Klein 4B with 4-bit quantization where supported plus model CPU offload.",
        model_id="black-forest-labs/FLUX.2-klein-4B",
        preferred_size=768,
        low_vram=True,
        license_note="Apache-2.0 for the 4B checkpoint.",
        lora_family="flux2-klein",
    )

    def __init__(self, store, **options):
        options.setdefault("low_vram", "bnb4")
        super().__init__(store, **options)
