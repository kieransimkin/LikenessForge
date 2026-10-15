from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from likenessforge.types import Box, FaceCandidate, PreparedFace


def stable_face_order(faces: list[FaceCandidate]) -> list[FaceCandidate]:
    ordered = sorted(faces, key=lambda f: (round(f.box.center[1] / 64), f.box.center[0]))
    for i, face in enumerate(ordered):
        face.index = i
    return ordered


def select_face(faces: list[FaceCandidate], selector: str | int) -> FaceCandidate:
    if not faces:
        raise RuntimeError("No faces were detected")
    if selector == "largest":
        return max(faces, key=lambda f: f.box.area)
    if isinstance(selector, str) and selector.isdigit():
        selector = int(selector)
    if isinstance(selector, int):
        for face in faces:
            if face.index == selector:
                return face
        raise IndexError(f"Face index {selector} not present; available: {[f.index for f in faces]}")
    raise ValueError("Face selector must be an integer index or 'largest'")


def square_crop_box(box: Box, image_size: tuple[int, int], margin: float = 0.55) -> Box:
    iw, ih = image_size
    cx, cy = box.center
    side = max(box.width, box.height) * (1.0 + 2.0 * margin)
    side = max(16.0, side)
    x1 = round(cx - side / 2)
    y1 = round(cy - side / 2)
    x2 = round(cx + side / 2)
    y2 = round(cy + side / 2)
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > iw:
        shift = x2 - iw
        x1 = max(0, x1 - shift)
        x2 = iw
    if y2 > ih:
        shift = y2 - ih
        y1 = max(0, y1 - shift)
        y2 = ih
    return Box(x1, y1, x2, y2)


def prepare_face_crop(
    image: Image.Image,
    full_mask: Image.Image,
    face: FaceCandidate,
    target_size: int,
    margin: float,
) -> PreparedFace:
    crop_box = square_crop_box(face.box, image.size, margin=margin)
    crop = image.crop(crop_box.as_tuple()).convert("RGB")
    mask = full_mask.crop(crop_box.as_tuple()).convert("L")
    crop = ImageOps.fit(crop, (target_size, target_size), method=Image.Resampling.LANCZOS)
    mask = ImageOps.fit(mask, (target_size, target_size), method=Image.Resampling.LANCZOS)
    return PreparedFace(face, crop_box, crop, mask, target_size)


def tighten_to_mask(image: Image.Image, mask: Image.Image, target_size: int, margin: float = 0.12) -> tuple[Image.Image, Image.Image]:
    arr = np.asarray(mask.convert("L"))
    ys, xs = np.where(arr > 32)
    if len(xs) == 0:
        box = Box(0, 0, image.width, image.height)
    else:
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        face_box = Box(x1, y1, x2, y2)
        box = square_crop_box(face_box, image.size, margin=margin)
    crop = ImageOps.fit(image.crop(box.as_tuple()).convert("RGB"), (target_size, target_size), Image.Resampling.LANCZOS)
    crop_mask = ImageOps.fit(mask.crop(box.as_tuple()).convert("L"), (target_size, target_size), Image.Resampling.LANCZOS)
    return crop, crop_mask


def masked_training_image(image: Image.Image, mask: Image.Image, background: str = "blur") -> Image.Image:
    image = image.convert("RGB")
    mask = mask.convert("L").filter(ImageFilter.GaussianBlur(radius=max(2, image.width // 128)))
    if background == "original":
        return image
    if background == "neutral":
        bg = Image.new("RGB", image.size, (127, 127, 127))
    elif background == "black":
        bg = Image.new("RGB", image.size, (0, 0, 0))
    else:
        bg = image.filter(ImageFilter.GaussianBlur(radius=max(8, image.width // 32)))
    return Image.composite(image, bg, mask)


def feather_mask(mask: Image.Image, feather: int, dilate: int = 0) -> Image.Image:
    m = mask.convert("L")
    if dilate > 0:
        size = max(3, dilate * 2 + 1)
        if size % 2 == 0:
            size += 1
        m = m.filter(ImageFilter.MaxFilter(size=size))
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=feather))
    return m


def composite_crop(
    original: Image.Image,
    generated_crop: Image.Image,
    crop_mask: Image.Image,
    crop_box: Box,
    feather: int = 32,
    dilate: int = 4,
    opacity: float = 1.0,
) -> Image.Image:
    """Merge a generated face crop back into the original image.

    ``opacity`` controls how much of the generated result is retained before the
    feathered face mask is applied. Lower values deliberately preserve more of
    the original photograph's expression, lighting and skin texture.
    """
    target_size = (crop_box.width, crop_box.height)
    generated = generated_crop.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
    mask = crop_mask.convert("L").resize(target_size, Image.Resampling.LANCZOS)
    scale = max(target_size) / max(1, max(crop_mask.size))
    mask = feather_mask(mask, feather=max(1, int(feather * scale)), dilate=max(0, int(dilate * scale)))
    out = original.convert("RGB").copy()
    base_crop = out.crop(crop_box.as_tuple())
    opacity = max(0.0, min(1.0, float(opacity)))
    if opacity < 1.0:
        generated = Image.blend(base_crop, generated, opacity)
    blended = Image.composite(generated, base_crop, mask)
    out.paste(blended, (crop_box.x1, crop_box.y1))
    return out


def comparison_sheet(
    original: Image.Image,
    candidates: list[Image.Image],
    labels: list[str],
    *,
    focus_box: Box | None = None,
    columns: int = 3,
    tile_size: tuple[int, int] = (360, 360),
) -> Image.Image:
    """Build a labelled original-vs-results contact sheet.

    When ``focus_box`` is supplied, each tile shows the same crop around the
    target face. This makes likeness differences much easier to compare than a
    full-frame contact sheet for photographs where the face is small.
    """
    images = [original.convert("RGB"), *[image.convert("RGB") for image in candidates]]
    captions = ["ORIGINAL", *labels]
    if len(captions) != len(images):
        raise ValueError("labels must contain exactly one entry per candidate")

    tile_w, tile_h = tile_size
    caption_h = 74
    columns = max(1, min(int(columns), len(images)))
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, rows * (tile_h + caption_h)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)

    for index, (image, caption) in enumerate(zip(images, captions, strict=True)):
        if focus_box is not None:
            image = image.crop(focus_box.as_tuple())
        fitted = ImageOps.contain(image, (tile_w - 12, tile_h - 12), method=Image.Resampling.LANCZOS)
        x0 = (index % columns) * tile_w
        y0 = (index // columns) * (tile_h + caption_h)
        px = x0 + (tile_w - fitted.width) // 2
        py = y0 + (tile_h - fitted.height) // 2
        sheet.paste(fitted, (px, py))
        draw.rectangle((x0, y0 + tile_h, x0 + tile_w, y0 + tile_h + caption_h), fill=(12, 12, 12))
        text = caption[:150]
        # Pillow's default font has no wrapping helper; split approximately by width.
        line_len = max(24, tile_w // 7)
        lines = [text[i : i + line_len] for i in range(0, len(text), line_len)][:3]
        draw.multiline_text((x0 + 8, y0 + tile_h + 7), "\n".join(lines), fill="white", spacing=3)
    return sheet


def annotate_faces(image: Image.Image, faces: list[FaceCandidate]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for face in faces:
        draw.rectangle(face.box.as_tuple(), outline="white", width=max(2, image.width // 300))
        label = f"{face.index} ({face.score:.2f})"
        tx, ty = face.box.x1, max(0, face.box.y1 - 18)
        draw.rectangle((tx, ty, tx + 90, ty + 18), fill="black")
        draw.text((tx + 3, ty + 2), label, fill="white")
    return out


def iter_images(path: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p
