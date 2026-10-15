from __future__ import annotations

from likenessforge.image_ops import stable_face_order
from likenessforge.plugins.base import FaceDetector, PluginInfo
from likenessforge.types import Box, FaceCandidate


class MTCNNDetector(FaceDetector):
    info = PluginInfo(
        name="mtcnn",
        description="facenet-pytorch MTCNN face detector with landmarks; compact and GPU/CPU capable.",
        low_vram=True,
    )

    def detect(self, image):
        try:
            import torch
            from facenet_pytorch import MTCNN
        except ImportError as exc:
            raise RuntimeError("mtcnn requires `pip install likenessforge[faces]`") from exc
        device = self.options.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        detector = MTCNN(
            keep_all=True,
            device=device,
            min_face_size=int(self.options.get("min_face_size", 32)),
            thresholds=tuple(self.options.get("thresholds", (0.6, 0.7, 0.7))),
        )
        boxes, probs, landmarks = detector.detect(image.convert("RGB"), landmarks=True)
        if boxes is None:
            return []
        faces = []
        for i, raw in enumerate(boxes):
            x1, y1, x2, y2 = [int(round(v)) for v in raw]
            lm = None
            if landmarks is not None and landmarks[i] is not None:
                lm = [(float(x), float(y)) for x, y in landmarks[i]]
            faces.append(FaceCandidate(i, Box(x1, y1, x2, y2).clamp(*image.size), float(probs[i]), lm))
        return stable_face_order(faces)
