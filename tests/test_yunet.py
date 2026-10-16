import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from likenessforge.plugins.detectors.yunet import OpenCVYuNetDetector


class DummyStore:
    def ensure_hf_file(self, repo_id, filename, revision=None):
        assert repo_id == "opencv/face_detection_yunet"
        assert filename == "face_detection_yunet_2023mar.onnx"
        return "/tmp/yunet.onnx"


def test_yunet_maps_detection_box_landmarks_and_score(monkeypatch):
    rows = np.array([[10, 20, 40, 50, 18, 35, 40, 35, 30, 47, 20, 58, 39, 58, 0.93]], dtype=np.float32)

    class FakeDetector:
        def setInputSize(self, size):
            assert size == (100, 80)

        def detect(self, image):
            assert image.shape == (80, 100, 3)
            return 1, rows

    fake_cv2 = SimpleNamespace(
        FaceDetectorYN=SimpleNamespace(create=lambda *args, **kwargs: FakeDetector())
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    detector = OpenCVYuNetDetector(DummyStore(), min_face_size=20)
    faces = detector.detect(Image.new("RGB", (100, 80)))

    assert len(faces) == 1
    assert faces[0].box.as_tuple() == (10, 20, 50, 70)
    assert abs(faces[0].score - 0.93) < 1e-5
    assert len(faces[0].landmarks) == 5


def test_yunet_bounds_large_input_and_returns_original_coordinates(monkeypatch):
    rows = np.array([[10, 20, 100, 80, 20, 30, 90, 30, 55, 50, 25, 80, 85, 80, .95]], dtype=np.float32)
    sizes = []

    class FakeDetector:
        def setInputSize(self, size):
            assert size == (640, 480)

        def detect(self, image):
            assert image.shape == (480, 640, 3)
            return 1, rows

    def resize(image, size, interpolation):
        sizes.append((image.shape, size))
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    fake_cv2 = SimpleNamespace(
        FaceDetectorYN=SimpleNamespace(create=lambda *args, **kwargs: FakeDetector()),
        resize=resize, INTER_AREA=3,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    faces = OpenCVYuNetDetector(DummyStore()).detect(Image.new("RGB", (1600, 1200)))
    assert sizes == [((1200, 1600, 3), (640, 480))]
    assert faces[0].box.as_tuple() == (25, 50, 275, 250)
    assert faces[0].landmarks[0] == (50, 75)
    assert faces[0].landmarks[-1] == (212.5, 200)
