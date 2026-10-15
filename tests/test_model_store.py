from pathlib import Path

from likenessforge import model_store
from likenessforge.model_store import ModelStore


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
    store = ModelStore(tmp_path / "cache", progress_callback=events.append)
    result = store.ensure_hf_snapshot("example/model")

    assert result == tmp_path / "snapshot"
    stages = [event.stage for event in events]
    assert stages[0] == "model_start"
    assert "file_progress" in stages
    assert stages[-1] == "model_ready"
    assert events[-1].path == str(result)
