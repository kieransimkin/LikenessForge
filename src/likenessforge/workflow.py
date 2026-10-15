from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

from PIL import Image

from likenessforge.debug import DebugSink
from likenessforge.image_ops import (
    annotate_faces,
    composite_crop,
    iter_images,
    masked_training_image,
    prepare_face_crop,
    select_face,
    tighten_to_mask,
)
from likenessforge.types import ModifyRequest, TrainingRequest

LOG = logging.getLogger("likenessforge")


def _training_sources(source: Path | str | Iterable[Path | str]):
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            yield from iter_images(path)
            return
        if path.is_file():
            yield path
            return
        raise FileNotFoundError(path)
    for item in source:
        path = Path(item)
        if not path.is_file():
            raise FileNotFoundError(path)
        yield path


class RepairWorkflow:
    def __init__(self, detector, segmenter, modifier, debug: DebugSink | None = None):
        self.detector = detector
        self.segmenter = segmenter
        self.modifier = modifier
        self.debug = debug or DebugSink(False)

    def inspect_faces(self, image: Image.Image):
        self.debug.step("detect_faces", detector=self.detector.info.name, image_size=image.size)
        faces = self.detector.detect(image)
        self.debug.step(
            "faces_detected",
            count=len(faces),
            faces=[{"index": f.index, "box": f.box.as_tuple(), "score": f.score} for f in faces],
        )
        self.debug.image("faces", annotate_faces(image, faces))
        return faces

    def repair(
        self,
        image: Image.Image,
        face_selector: str | int,
        lora: str | Path | None,
        prompt: str,
        negative_prompt: str | None,
        crop_margin: float = 0.55,
        feather: int = 32,
        seed: int = 0,
        lora_scale: float = 1.0,
        reference_images: list[Image.Image] | None = None,
        modifier_options: dict | None = None,
    ) -> Image.Image:
        image = image.convert("RGB")
        self.debug.image("original", image)
        faces = self.inspect_faces(image)
        face = select_face(faces, face_selector)
        self.debug.step("face_selected", index=face.index, box=face.box.as_tuple(), score=face.score)

        self.debug.step("segment_face", segmenter=self.segmenter.info.name)
        full_mask = self.segmenter.segment(image, face).convert("L")
        self.debug.image("full-face-mask", full_mask)

        target_size = self.modifier.preferred_size
        prepared = prepare_face_crop(image, full_mask, face, target_size, crop_margin)
        self.debug.step(
            "crop_face",
            crop_box=prepared.crop_box.as_tuple(),
            target_size=target_size,
            margin=crop_margin,
        )
        self.debug.image("face-crop", prepared.crop_image)
        self.debug.image("face-crop-mask", prepared.crop_mask)

        request = ModifyRequest(
            image=prepared.crop_image,
            mask=prepared.crop_mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            lora=lora,
            lora_scale=lora_scale,
            seed=seed,
            reference_images=reference_images or [],
            options=modifier_options or {},
        )
        self.debug.step(
            "modify_face",
            modifier=self.modifier.info.name,
            model_id=self.modifier.info.model_id,
            lora=str(lora) if lora else None,
            lora_scale=lora_scale,
            seed=seed,
            options=request.options,
        )
        generated = self.modifier.modify(request).convert("RGB")
        self.debug.image("generated-face", generated)

        self.debug.step("composite", feather=feather, blend="alpha-mask")
        output = composite_crop(
            image,
            generated,
            prepared.crop_mask,
            prepared.crop_box,
            feather=feather,
        )
        self.debug.image("composited", output)
        return output


class TrainingWorkflow:
    def __init__(self, detector, segmenter, trainer, debug: DebugSink | None = None):
        self.detector = detector
        self.segmenter = segmenter
        self.trainer = trainer
        self.debug = debug or DebugSink(False)

    def prepare_dataset(
        self,
        source_dir: Path | str | Iterable[Path | str],
        prepared_dir: Path,
        trigger: str,
        selector: str | int = "largest",
        background: str = "blur",
    ) -> int:
        prepared_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for source in _training_sources(source_dir):
            image = Image.open(source).convert("RGB")
            faces = self.detector.detect(image)
            if not faces:
                LOG.warning("skipping image with no detected face: %s", source)
                continue
            face = select_face(faces, selector)
            mask = self.segmenter.segment(image, face).convert("L")
            size = self.trainer.preferred_size
            crop, crop_mask = tighten_to_mask(image, mask, size, margin=0.14)
            train_image = masked_training_image(crop, crop_mask, background=background)
            stem = f"{count:04d}-{source.stem}"
            train_image.save(prepared_dir / f"{stem}.png")
            if self.debug.enabled:
                self.debug.image(f"train-{stem}-crop", crop)
                self.debug.image(f"train-{stem}-mask", crop_mask)
                self.debug.image(f"train-{stem}-final", train_image)
            self.debug.step(
                "prepare_training_image",
                source=str(source),
                face_index=face.index,
                trainer=self.trainer.info.name,
                resolution=size,
            )
            count += 1
        if count == 0:
            raise RuntimeError("No usable training images were produced")
        return count

    def train(
        self,
        source_dir: Path | str | Iterable[Path | str],
        output_dir: Path,
        trigger: str,
        selector: str | int = "largest",
        background: str = "blur",
        steps: int = 800,
        rank: int = 16,
        lr: float = 1e-4,
        low_vram: bool = False,
        extra_args: list[str] | None = None,
    ) -> Path:
        prepared = output_dir / "prepared-dataset"
        count = self.prepare_dataset(source_dir, prepared, trigger, selector, background)
        self.debug.step("training_dataset_ready", count=count, path=str(prepared))
        req = TrainingRequest(
            dataset_dir=prepared,
            output_dir=output_dir / "lora",
            instance_prompt=f"photo of {trigger} person",
            resolution=self.trainer.preferred_size,
            max_train_steps=steps,
            rank=rank,
            learning_rate=lr,
            low_vram=low_vram,
            extra_args=extra_args or [],
        )
        self.debug.step(
            "train_lora",
            trainer=self.trainer.info.name,
            model_id=self.trainer.info.model_id,
            resolution=req.resolution,
            steps=steps,
            rank=rank,
            low_vram=low_vram,
        )
        weights = self.trainer.train(req)
        metadata = {
            "trigger": trigger,
            "trainer": self.trainer.info.name,
            "model_id": self.trainer.info.model_id,
            "lora_family": self.trainer.info.lora_family,
            "weights": str(weights),
            "prepared_images": count,
            "resolution": req.resolution,
            "steps": steps,
            "rank": rank,
            "learning_rate": lr,
            "low_vram": low_vram,
        }
        metadata_path = output_dir / "likenessforge-training.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.debug.step("training_complete", weights=str(weights), metadata=str(metadata_path))
        return weights
