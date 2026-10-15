from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from likenessforge.config import load_config
from likenessforge.debug import DebugSink
from likenessforge.image_ops import composite_crop, select_face, stable_face_order
from likenessforge.types import Box, FaceCandidate
from likenessforge.workflow import RepairWorkflow, TrainingWorkflow, wizard_variants


class DummyInfo:
    def __init__(self, name, model_id=None, preferred_size=64, lora_family=None):
        self.name = name
        self.model_id = model_id
        self.preferred_size = preferred_size
        self.lora_family = lora_family


class DummyDetector:
    info = DummyInfo("dummy-detector")

    def detect(self, image):
        return stable_face_order([
            FaceCandidate(0, Box(70, 20, 110, 70), 0.9),
            FaceCandidate(1, Box(10, 20, 50, 70), 0.8),
        ])


class DummySegmenter:
    info = DummyInfo("dummy-segmenter")

    def segment(self, image, face):
        m = Image.new("L", image.size, 0)
        ImageDraw.Draw(m).ellipse(face.box.as_tuple(), fill=255)
        return m


class DummyModifier:
    info = DummyInfo("dummy-modifier", "dummy/model", 64)

    @property
    def preferred_size(self):
        return 64

    def modify(self, request):
        return Image.new("RGB", request.image.size, (255, 0, 0))


def test_face_order_and_select():
    faces = DummyDetector().detect(Image.new("RGB", (128, 128)))
    assert faces[0].box.x1 == 10
    assert select_face(faces, "1").box.x1 == 70
    assert select_face(faces, "largest").box.area == 2000


def test_composite_does_not_replace_square_corners():
    original = Image.new("RGB", (100, 100), (0, 0, 0))
    generated = Image.new("RGB", (50, 50), (255, 0, 0))
    mask = Image.new("L", (50, 50), 0)
    ImageDraw.Draw(mask).ellipse((10, 10, 40, 40), fill=255)
    out = composite_crop(original, generated, mask, Box(25, 25, 75, 75), feather=2, dilate=0)
    arr = np.asarray(out)
    assert tuple(arr[25, 25]) == (0, 0, 0)
    assert arr[50, 50, 0] > 200


def test_repair_workflow_only_changes_face_region():
    image = Image.new("RGB", (128, 96), (20, 40, 60))
    wf = RepairWorkflow(DummyDetector(), DummySegmenter(), DummyModifier(), DebugSink(False))
    out = wf.repair(image, 0, None, "repair", None, crop_margin=0.2, feather=1)
    arr = np.asarray(out)
    assert tuple(arr[0, 0]) == (20, 40, 60)
    assert arr[45, 30, 0] > 20


def test_yaml_config(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text("plugins:\n  detector: mtcnn\n  modifier:\n    name: sdxl-inpaint\n    options:\n      steps: 12\nfeather: 24\n")
    cfg = load_config(p)
    assert cfg.detector.name == "mtcnn"
    assert cfg.modifier.name == "sdxl-inpaint"
    assert cfg.modifier.options["steps"] == 12
    assert cfg.feather == 24


class DummyTrainer:
    info = DummyInfo("dummy-trainer", "dummy/base", 64, "dummy")

    @property
    def preferred_size(self):
        return 64

    def train(self, request):
        request.output_dir.mkdir(parents=True, exist_ok=True)
        weights = request.output_dir / "identity.safetensors"
        weights.write_bytes(b"dummy")
        return weights


def test_training_workflow_accepts_selected_reference_files(tmp_path: Path):
    refs = []
    for index in range(2):
        path = tmp_path / f"reference-{index}.png"
        Image.new("RGB", (128, 96), (40 + index, 50, 60)).save(path)
        refs.append(path)

    output = tmp_path / "training"
    workflow = TrainingWorkflow(DummyDetector(), DummySegmenter(), DummyTrainer(), DebugSink(False))
    weights = workflow.train(refs, output, "lf_subject", steps=2, rank=4)

    assert weights.exists()
    prepared = sorted((output / "prepared-dataset").glob("*.png"))
    assert len(prepared) == 2
    metadata = (output / "likenessforge-training.json").read_text(encoding="utf-8")
    assert '"lora_family": "dummy"' in metadata
    assert '"prepared_images": 2' in metadata


def test_composite_opacity_preserves_original_detail():
    original = Image.new("RGB", (100, 100), (0, 0, 0))
    generated = Image.new("RGB", (50, 50), (255, 0, 0))
    mask = Image.new("L", (50, 50), 255)
    out = composite_crop(
        original, generated, mask, Box(25, 25, 75, 75), feather=0, dilate=0, opacity=0.5
    )
    pixel = np.asarray(out)[50, 50]
    assert 120 <= int(pixel[0]) <= 135
    assert int(pixel[1]) == 0


def test_wizard_presets_vary_crop_and_merge_settings():
    variants = wizard_variants("sdxl-inpaint", "balanced")
    assert len(variants) == 6
    assert len({variant.crop_margin for variant in variants}) >= 4
    assert len({variant.merge_opacity for variant in variants}) >= 4
    assert len({variant.feather for variant in variants}) >= 4
    assert all("strength" in variant.options for variant in variants)

    flux = wizard_variants("flux2-klein-4b-lowvram", "quick")
    assert len(flux) == 4
    assert all("strength" not in variant.options for variant in flux)


def test_repair_wizard_saves_candidates_comparisons_and_manifest(tmp_path: Path):
    image = Image.new("RGB", (128, 96), (20, 40, 60))
    workflow = RepairWorkflow(DummyDetector(), DummySegmenter(), DummyModifier(), DebugSink(False))
    output = tmp_path / "wizard"
    run = workflow.wizard(
        image=image,
        output_dir=output,
        face_selector=0,
        lora=None,
        prompt="repair",
        negative_prompt=None,
        preset="quick",
        seed=10,
    )

    assert len(run.candidates) == 4
    assert all(candidate.path.exists() for candidate in run.candidates)
    assert run.comparison_face_path.exists()
    assert run.comparison_full_path.exists()
    assert run.manifest_path.exists()
    manifest = run.manifest_path.read_text(encoding="utf-8")
    assert '"crop_margin"' in manifest
    assert '"merge_opacity"' in manifest
    assert '"effective_seed"' in manifest
