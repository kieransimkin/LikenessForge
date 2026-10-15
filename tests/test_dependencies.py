from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_default_dependency_graph_does_not_include_facenet_pytorch():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "facenet-pytorch" not in pyproject
    assert '"opencv-python-headless>=4.10"' in pyproject


def test_yunet_is_documented_as_default_detector():
    from likenessforge.config import ForgeConfig

    assert ForgeConfig().detector.name == "opencv-yunet"


def test_platform_cache_dependency_removed():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "platformdirs" not in pyproject
