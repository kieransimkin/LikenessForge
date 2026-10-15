from __future__ import annotations

import numpy as np
from PIL import Image

from likenessforge.plugins.base import FaceSegmenter, PluginInfo


class SAM2TinySegmenter(FaceSegmenter):
    info = PluginInfo(
        name="sam2-tiny",
        description="SAM 2.1 Hiera Tiny prompted with the detected face box.",
        model_id="facebook/sam2.1-hiera-tiny",
        low_vram=True,
        license_note="See the SAM 2.1 model card/license on Hugging Face.",
    )

    def segment(self, image, face):
        try:
            import torch
            from transformers import Sam2Model, Sam2Processor
        except ImportError as exc:
            raise RuntimeError("sam2-tiny requires `pip install likenessforge[sam]`") from exc
        model_dir = self.store.ensure_hf_snapshot(self.info.model_id)
        device = self.options.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        processor = Sam2Processor.from_pretrained(model_dir)
        model = Sam2Model.from_pretrained(model_dir, torch_dtype=dtype).to(device)
        box = [list(face.box.as_tuple())]
        inputs = processor(images=image.convert("RGB"), input_boxes=[box], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)
        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        arr = masks[0, 0].numpy().astype(np.uint8) * 255
        return Image.fromarray(arr, mode="L")
