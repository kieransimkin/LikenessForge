from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from likenessforge.config import load_config
from likenessforge.debug import DebugSink
from likenessforge.image_ops import composite_crop, select_face, stable_face_order
from likenessforge.types import Box, FaceCandidate
from likenessforge.workflow import RepairWorkflow


class DummyInfo:
    def __init__(self, name, model_id=None, preferred_size=64):
        self.name = name
        self.model_id = model_id
        self.preferred_size = preferred_size


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
