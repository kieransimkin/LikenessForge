from __future__ import annotations

import numpy as np

from likenessforge.image_ops import stable_face_order
from likenessforge.plugins.base import FaceDetector, PluginInfo
from likenessforge.types import Box, FaceCandidate


class OpenCVHaarDetector(FaceDetector):
    info = PluginInfo(
        name="opencv-haar",
        description="Very small CPU fallback using OpenCV's bundled frontal-face cascade.",
        low_vram=True,
        license_note="OpenCV license applies to the cascade data.",
    )

    def detect(self, image):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-haar requires `pip install likenessforge[faces]`") from exc
        gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        rects = cascade.detectMultiScale(
            gray,
            scaleFactor=float(self.options.get("scale_factor", 1.08)),
            minNeighbors=int(self.options.get("min_neighbors", 5)),
            minSize=tuple(self.options.get("min_size", (36, 36))),
        )
        faces = [FaceCandidate(i, Box(int(x), int(y), int(x + w), int(y + h)), 1.0) for i, (x, y, w, h) in enumerate(rects)]
        return stable_face_order(faces)
