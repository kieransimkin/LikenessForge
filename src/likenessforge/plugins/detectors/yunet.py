from __future__ import annotations

import numpy as np

from likenessforge.image_ops import stable_face_order
from likenessforge.plugins.base import FaceDetector, PluginInfo
from likenessforge.types import Box, FaceCandidate


MODEL_ID = "opencv/face_detection_yunet"
MODEL_FILE = "face_detection_yunet_2023mar.onnx"


class OpenCVYuNetDetector(FaceDetector):
    info = PluginInfo(
        name="opencv-yunet",
        description=(
            "Lightweight OpenCV YuNet ONNX face detector with five landmarks; "
            "CPU-friendly and independent of the PyTorch diffusion stack."
        ),
        model_id=MODEL_ID,
        low_vram=True,
        license_note="OpenCV YuNet model licence applies; see the upstream Hugging Face repository.",
    )

    def detect(self, image):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-yunet requires `pip install likenessforge[faces]`") from exc

        model_path = self.store.ensure_hf_file(MODEL_ID, MODEL_FILE)
        rgb = np.asarray(image.convert("RGB"))
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        height, width = bgr.shape[:2]

        score_threshold = float(self.options.get("score_threshold", 0.75))
        nms_threshold = float(self.options.get("nms_threshold", 0.3))
        top_k = int(self.options.get("top_k", 5000))
        min_face_size = int(self.options.get("min_face_size", 20))

        detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (width, height),
            score_threshold,
            nms_threshold,
            top_k,
        )
        detector.setInputSize((width, height))
        _, detections = detector.detect(bgr)
        if detections is None:
            return []

        faces = []
        for raw in detections:
            x, y, w, h = [float(v) for v in raw[:4]]
            if w < min_face_size or h < min_face_size:
                continue
            x1, y1 = int(round(x)), int(round(y))
            x2, y2 = int(round(x + w)), int(round(y + h))
            landmarks = [(float(raw[i]), float(raw[i + 1])) for i in range(4, 14, 2)]
            score = float(raw[14])
            faces.append(
                FaceCandidate(
                    len(faces),
                    Box(x1, y1, x2, y2).clamp(*image.size),
                    score,
                    landmarks,
                )
            )
        return stable_face_order(faces)
