from fnmatch import fnmatch
from pathlib import Path

import pytest

from likenessforge import model_store
from likenessforge.model_store import ModelStore, diffusers_snapshot_files


def test_diffusers_snapshot_selects_one_pipeline_representation(monkeypatch, tmp_path):
    captured = {}
    store = ModelStore(project_root=tmp_path)

    def download(repo_id, **kwargs):
        captured.update(repo_id=repo_id, **kwargs)
        return tmp_path / "snapshot"

    monkeypatch.setattr(store, "ensure_hf_snapshot", download)
    monkeypatch.setattr(model_store, "list_repo_files", lambda *args, **kwargs: [
        "model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
        "text_encoder/model-00001-of-00002.safetensors", "tokenizer/vocab.json", "LICENSE",
        "model.ckpt", "v1-5.safetensors", "unet/diffusion_pytorch_model.bin",
        "unet/diffusion_pytorch_model.fp16.safetensors", "onnx/model.onnx",
    ])
    assert store.ensure_diffusers_snapshot("example/model", revision="immutable") == tmp_path / "snapshot"
    assert captured["revision"] == "immutable"
    patterns = captured["allow_patterns"]
    for path in ("model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
                 "text_encoder/model-00001-of-00002.safetensors", "tokenizer/vocab.json", "LICENSE"):
        assert any(fnmatch(path, pattern) for pattern in patterns), path
    for path in ("model.ckpt", "v1-5.safetensors", "unet/diffusion_pytorch_model.bin",
                 "unet/diffusion_pytorch_model.fp16.safetensors", "onnx/model.onnx"):
        assert not any(fnmatch(path, pattern) for pattern in patterns), path


def test_snapshot_selects_safetensors_or_binary_once_per_component():
    files = ["unet/diffusion_pytorch_model.safetensors", "unet/diffusion_pytorch_model.bin",
             "vae/diffusion_pytorch_model.bin", "vae/diffusion_pytorch_model.fp16.bin"]
    assert diffusers_snapshot_files(files) == [files[0], files[2]]


def test_snapshot_precision_variant_is_explicit_and_keeps_shard_index():
    files = ["unet/diffusion_pytorch_model.fp16.safetensors",
             "unet/diffusion_pytorch_model.bin", "text_encoder/model.fp16.safetensors.index.json",
             "text_encoder/model.fp16-00001-of-00002.safetensors"]
    assert diffusers_snapshot_files(files, "fp16") == sorted([files[0], files[2], files[3]])
    with pytest.raises(ValueError, match="No compatible"):
        diffusers_snapshot_files(["model_index.json", "unet/config.json"])


def test_model_store_emits_download_progress(monkeypatch, tmp_path: Path):
    events = []

    def fake_snapshot_download(**kwargs):
        target = tmp_path / "snapshot"
        target.mkdir(exist_ok=True)
        tqdm_class = kwargs.get("tqdm_class")
        if tqdm_class:
            bar = tqdm_class(total=100, desc="weights.safetensors", unit="B", disable=True)
            bar.update(40)
            bar.update(60)
            bar.close()
        return str(target)

    monkeypatch.setattr(model_store, "snapshot_download", fake_snapshot_download)
    store = ModelStore(project_root=tmp_path, progress_callback=events.append)
    result = store.ensure_hf_snapshot("example/model")

    assert result == tmp_path / "snapshot"
    stages = [event.stage for event in events]
    assert stages[0] == "model_start"
    assert "file_progress" in stages
    assert stages[-1] == "model_ready"
    assert events[-1].path == str(result)


def test_model_store_is_always_project_local(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("LIKENESSFORGE_PROJECT_ROOT", str(tmp_path))
    store = ModelStore()

    assert store.project_root == tmp_path.resolve()
    assert store.root == tmp_path.resolve() / "models"
    assert store.hf == tmp_path.resolve() / "models" / "huggingface"
    assert store.tools == tmp_path.resolve() / "models" / "tools"
    assert store.hf.is_dir()
    assert store.tools.is_dir()


def test_project_models_directory_is_gitignored():
    root = Path(__file__).resolve().parents[1]
    patterns = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "models/" in patterns


def test_remote_asset_resolution_uses_project_huggingface_cache(monkeypatch, tmp_path: Path):
    store = ModelStore(project_root=tmp_path)
    expected = tmp_path / "models" / "huggingface" / "snapshot"

    def fake_ensure(repo_id, revision=None, allow_patterns=None, ignore_patterns=None):
        assert repo_id == "example/identity-lora"
        expected.mkdir(parents=True, exist_ok=True)
        return expected

    monkeypatch.setattr(store, "ensure_hf_snapshot", fake_ensure)
    assert store.resolve_local_or_hf("example/identity-lora") == expected
