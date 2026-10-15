from __future__ import annotations

from likenessforge.plugins.base import PluginInfo
from likenessforge.plugins.detectors.yunet import MODEL_ID, OpenCVYuNetDetector


class MTCNNDetector(OpenCVYuNetDetector):
    """Backward-compatible alias for configurations that selected ``mtcnn`` before v0.2.1.

    facenet-pytorch's MTCNN pins an obsolete Torch/Torchvision pair that conflicts with
    LikenessForge's diffusion backends. The legacy plugin name therefore delegates to
    the lightweight YuNet backend instead of pulling a second incompatible Torch stack.
    """

    info = PluginInfo(
        name="mtcnn",
        description=(
            "Deprecated compatibility alias for opencv-yunet. Existing configs keep working "
            "without the old facenet-pytorch Torch/Torchvision dependency conflict."
        ),
        model_id=MODEL_ID,
        low_vram=True,
        license_note="OpenCV YuNet model licence applies; see the upstream Hugging Face repository.",
    )
