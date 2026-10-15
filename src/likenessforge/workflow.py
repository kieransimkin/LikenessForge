from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from likenessforge.debug import DebugSink
from likenessforge.image_ops import (
    annotate_faces,
    comparison_sheet,
    composite_crop,
    iter_images,
    masked_training_image,
    prepare_face_crop,
    select_face,
    square_crop_box,
    tighten_to_mask,
)
from likenessforge.types import ModifyRequest, TrainingRequest

LOG = logging.getLogger("likenessforge")
WizardProgress = Callable[[int, int, str], None]


@dataclass(frozen=True)
class WizardVariant:
    name: str
    crop_margin: float
    lora_scale: float
    seed_offset: int
    feather: int
    dilate: int
    merge_opacity: float
    options: dict[str, float | int] = field(default_factory=dict)

    def label(self) -> str:
        option_bits = ", ".join(f"{key}={value:g}" for key, value in self.options.items())
        if option_bits:
            option_bits = f", {option_bits}"
        return (
            f"{self.name}: crop={self.crop_margin:.2f}, LoRA={self.lora_scale:.2f}, "
            f"merge={self.merge_opacity:.2f}, feather={self.feather}, dilate={self.dilate}{option_bits}"
        )


@dataclass
class WizardCandidate:
    index: int
    variant: WizardVariant
    image: Image.Image
    path: Path


@dataclass
class WizardRun:
    output_dir: Path
    candidates: list[WizardCandidate]
    comparison_face_path: Path
    comparison_full_path: Path
    manifest_path: Path
    comparison_face: Image.Image
    comparison_full: Image.Image


def wizard_variants(modifier_name: str, preset: str = "balanced") -> list[WizardVariant]:
    """Return a bounded set of useful repair experiments for a modifier family."""
    preset = preset.lower()
    if preset not in {"quick", "balanced", "thorough"}:
        raise ValueError("wizard preset must be quick, balanced or thorough")

    is_flux = modifier_name.startswith("flux2-")
    if is_flux:
        variants = [
            WizardVariant("tight-natural", 0.34, 0.85, 0, 18, 2, 0.82, {"steps": 4, "guidance": 3.2}),
            WizardVariant("balanced", 0.52, 0.95, 0, 28, 4, 0.92, {"steps": 4, "guidance": 4.0}),
            WizardVariant("wide-context", 0.78, 1.00, 0, 36, 5, 0.96, {"steps": 4, "guidance": 4.0}),
            WizardVariant("identity-strong", 0.55, 1.15, 1, 30, 4, 1.00, {"steps": 5, "guidance": 4.6}),
            WizardVariant("original-preserving", 0.44, 1.05, 2, 46, 3, 0.72, {"steps": 4, "guidance": 3.6}),
            WizardVariant("soft-wide", 0.88, 0.90, 3, 54, 7, 0.80, {"steps": 5, "guidance": 3.8}),
            WizardVariant("tight-strong", 0.28, 1.20, 4, 22, 2, 0.88, {"steps": 5, "guidance": 4.8}),
            WizardVariant("wide-strong", 0.95, 1.10, 5, 40, 6, 0.90, {"steps": 6, "guidance": 4.4}),
            WizardVariant("soft-identity", 0.64, 1.25, 6, 62, 8, 0.68, {"steps": 6, "guidance": 4.2}),
            WizardVariant("alternate-seed", 0.52, 1.00, 7, 34, 4, 0.94, {"steps": 6, "guidance": 4.0}),
        ]
    else:
        variants = [
            WizardVariant(
                "tight-natural", 0.34, 0.85, 0, 18, 2, 0.82, {"steps": 20, "guidance": 5.0, "strength": 0.44}
            ),
            WizardVariant(
                "balanced", 0.52, 0.95, 0, 28, 4, 0.92, {"steps": 24, "guidance": 5.8, "strength": 0.52}
            ),
            WizardVariant(
                "wide-context", 0.78, 1.00, 0, 36, 5, 0.96, {"steps": 24, "guidance": 6.0, "strength": 0.56}
            ),
            WizardVariant(
                "identity-strong", 0.55, 1.15, 1, 30, 4, 1.00, {"steps": 28, "guidance": 6.4, "strength": 0.64}
            ),
            WizardVariant(
                "original-preserving", 0.44, 1.05, 2, 46, 3, 0.70, {"steps": 24, "guidance": 5.4, "strength": 0.56}
            ),
            WizardVariant(
                "soft-wide", 0.88, 0.90, 3, 54, 7, 0.80, {"steps": 28, "guidance": 5.2, "strength": 0.50}
            ),
            WizardVariant(
                "tight-strong", 0.28, 1.20, 4, 22, 2, 0.88, {"steps": 30, "guidance": 6.8, "strength": 0.70}
            ),
            WizardVariant(
                "wide-strong", 0.95, 1.10, 5, 40, 6, 0.90, {"steps": 30, "guidance": 6.2, "strength": 0.62}
            ),
            WizardVariant(
                "soft-identity", 0.64, 1.25, 6, 62, 8, 0.66, {"steps": 32, "guidance": 6.0, "strength": 0.66}
            ),
            WizardVariant(
                "alternate-seed", 0.52, 1.00, 7, 34, 4, 0.94, {"steps": 32, "guidance": 5.8, "strength": 0.58}
            ),
        ]

    counts = {"quick": 4, "balanced": 6, "thorough": 10}
    return variants[: counts[preset]]


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

    def _prepare_subject(self, image: Image.Image, face_selector: str | int):
        image = image.convert("RGB")
        self.debug.image("original", image)
        faces = self.inspect_faces(image)
        face = select_face(faces, face_selector)
        self.debug.step("face_selected", index=face.index, box=face.box.as_tuple(), score=face.score)
        self.debug.step("segment_face", segmenter=self.segmenter.info.name)
        full_mask = self.segmenter.segment(image, face).convert("L")
        self.debug.image("full-face-mask", full_mask)
        return image, face, full_mask

    def _repair_prepared(
        self,
        image: Image.Image,
        face,
        full_mask: Image.Image,
        lora: str | Path | None,
        prompt: str,
        negative_prompt: str | None,
        *,
        crop_margin: float,
        feather: int,
        dilate: int = 4,
        merge_opacity: float = 1.0,
        seed: int,
        lora_scale: float,
        reference_images: list[Image.Image] | None,
        modifier_options: dict | None,
        debug_prefix: str = "",
    ) -> Image.Image:
        target_size = self.modifier.preferred_size
        prepared = prepare_face_crop(image, full_mask, face, target_size, crop_margin)
        self.debug.step(
            "crop_face",
            crop_box=prepared.crop_box.as_tuple(),
            target_size=target_size,
            margin=crop_margin,
            merge_opacity=merge_opacity,
            feather=feather,
            dilate=dilate,
        )
        if debug_prefix:
            self.debug.image(f"{debug_prefix}-face-crop", prepared.crop_image)
            self.debug.image(f"{debug_prefix}-face-crop-mask", prepared.crop_mask)
        else:
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
        self.debug.image(f"{debug_prefix}-generated-face" if debug_prefix else "generated-face", generated)

        self.debug.step(
            "composite",
            feather=feather,
            dilate=dilate,
            merge_opacity=merge_opacity,
            blend="alpha-mask",
        )
        output = composite_crop(
            image,
            generated,
            prepared.crop_mask,
            prepared.crop_box,
            feather=feather,
            dilate=dilate,
            opacity=merge_opacity,
        )
        self.debug.image(f"{debug_prefix}-composited" if debug_prefix else "composited", output)
        return output

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
        image, face, full_mask = self._prepare_subject(image, face_selector)
        return self._repair_prepared(
            image,
            face,
            full_mask,
            lora,
            prompt,
            negative_prompt,
            crop_margin=crop_margin,
            feather=feather,
            seed=seed,
            lora_scale=lora_scale,
            reference_images=reference_images,
            modifier_options=modifier_options,
        )

    def wizard(
        self,
        image: Image.Image,
        output_dir: Path,
        face_selector: str | int,
        lora: str | Path | None,
        prompt: str,
        negative_prompt: str | None,
        *,
        preset: str = "balanced",
        seed: int = 0,
        reference_images: list[Image.Image] | None = None,
        progress: WizardProgress | None = None,
    ) -> WizardRun:
        """Run a bounded repair sweep and emit original-vs-result comparison sheets."""
        output_dir.mkdir(parents=True, exist_ok=True)
        image, face, full_mask = self._prepare_subject(image, face_selector)
        variants = wizard_variants(self.modifier.info.name, preset)
        candidates: list[WizardCandidate] = []
        images: list[Image.Image] = []
        labels: list[str] = []

        for index, variant in enumerate(variants, start=1):
            if progress:
                progress(index, len(variants), variant.label())
            result = self._repair_prepared(
                image,
                face,
                full_mask,
                lora,
                prompt,
                negative_prompt,
                crop_margin=variant.crop_margin,
                feather=variant.feather,
                dilate=variant.dilate,
                merge_opacity=variant.merge_opacity,
                seed=seed + variant.seed_offset,
                lora_scale=variant.lora_scale,
                reference_images=reference_images,
                modifier_options=dict(variant.options),
                debug_prefix=f"wizard-{index:02d}",
            )
            path = output_dir / f"candidate-{index:02d}-{variant.name}.png"
            result.save(path)
            candidate = WizardCandidate(index, variant, result, path)
            candidates.append(candidate)
            images.append(result)
            labels.append(f"#{index} {variant.label()}")

        focus_margin = min(1.25, max(variant.crop_margin for variant in variants) + 0.12)
        focus_box = square_crop_box(face.box, image.size, margin=focus_margin)
        comparison_face = comparison_sheet(image, images, labels, focus_box=focus_box, columns=3)
        comparison_full = comparison_sheet(image, images, labels, columns=3, tile_size=(420, 300))
        comparison_face_path = output_dir / "comparison-face.png"
        comparison_full_path = output_dir / "comparison-full.png"
        comparison_face.save(comparison_face_path)
        comparison_full.save(comparison_full_path)

        manifest = {
            "modifier": self.modifier.info.name,
            "model_id": self.modifier.info.model_id,
            "preset": preset,
            "base_seed": seed,
            "face_index": face.index,
            "face_box": face.box.as_tuple(),
            "comparison_face": str(comparison_face_path),
            "comparison_full": str(comparison_full_path),
            "candidates": [
                {
                    "index": candidate.index,
                    "path": str(candidate.path),
                    "settings": asdict(candidate.variant),
                    "effective_seed": seed + candidate.variant.seed_offset,
                }
                for candidate in candidates
            ],
        }
        manifest_path = output_dir / "wizard-results.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.debug.step(
            "wizard_complete",
            preset=preset,
            candidates=len(candidates),
            comparison_face=str(comparison_face_path),
            comparison_full=str(comparison_full_path),
            manifest=str(manifest_path),
        )
        return WizardRun(
            output_dir,
            candidates,
            comparison_face_path,
            comparison_full_path,
            manifest_path,
            comparison_face,
            comparison_full,
        )


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
