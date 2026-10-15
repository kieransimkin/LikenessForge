from __future__ import annotations

from PIL import Image, ImageDraw, ImageFilter

from likenessforge.plugins.base import FaceSegmenter, PluginInfo


class SoftFaceSegmenter(FaceSegmenter):
    info = PluginInfo(
        name="soft-face",
        description="Zero-download elliptical face mask generated from the detector bounding box.",
        low_vram=True,
    )

    def segment(self, image, face):
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        x1, y1, x2, y2 = face.box.as_tuple()
        w, h = face.box.width, face.box.height
        # Include forehead/chin while remaining comfortably inside the crop.
        pad_x = int(w * float(self.options.get("pad_x", 0.04)))
        pad_top = int(h * float(self.options.get("pad_top", 0.12)))
        pad_bottom = int(h * float(self.options.get("pad_bottom", 0.08)))
        draw.ellipse((x1 - pad_x, y1 - pad_top, x2 + pad_x, y2 + pad_bottom), fill=255)
        blur = int(self.options.get("edge_blur", max(2, min(w, h) // 24)))
        return mask.filter(ImageFilter.GaussianBlur(blur))
