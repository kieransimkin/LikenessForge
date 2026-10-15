from likenessforge.registry import plugin_classes


def test_builtin_plugins_are_registered():
    assert "opencv-haar" in plugin_classes("detector")
    assert "opencv-yunet" in plugin_classes("detector")
    assert "mtcnn" in plugin_classes("detector")
    assert "sam2-tiny" in plugin_classes("segmenter")
    assert "sd15-inpaint-lowvram" in plugin_classes("modifier")
    assert "flux2-klein-4b-lowvram" in plugin_classes("modifier")
    assert "sdxl-lora" in plugin_classes("trainer")
