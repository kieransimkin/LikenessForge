from types import SimpleNamespace

from likenessforge import cli
from likenessforge.model_store import ModelStore
from likenessforge.plugins.modifiers.diffusers_inpaint import SD15InpaintModifier
from likenessforge.plugins.trainers.diffusers_scripts import SD15LoraTrainer


def test_setup_and_readiness_share_plugin_variant_and_revision(monkeypatch, tmp_path):
    store = ModelStore(tmp_path)
    calls = []

    def acquire(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return tmp_path

    monkeypatch.setattr(store, "ensure_diffusers_snapshot", acquire)
    monkeypatch.setattr(store, "is_diffusers_snapshot_cached", acquire)
    modifier = SD15InpaintModifier(store, revision="fixed", device="cuda")
    assert modifier.prepare_model() == tmp_path
    assert modifier.is_model_cached()
    assert calls[0] == calls[1] == (modifier.info.model_id, {"revision": "fixed", "variant": "fp16"})
    calls.clear()
    trainer = SD15LoraTrainer(store, revision="base-fixed")
    trainer.prepare_model()
    trainer.is_model_cached()
    assert calls[0] == calls[1] == (trainer.info.model_id, {"revision": "base-fixed"})


def test_cli_model_download_uses_plugin_setup_and_config(monkeypatch, tmp_path, capsys):
    seen = {}

    def create(kind, name, store, **options):
        seen.update(kind=kind, name=name, options=options)
        return SimpleNamespace(prepare_model=lambda: tmp_path)

    config = tmp_path / "config.yaml"
    config.write_text("modifier:\n  name: sd15-inpaint-lowvram\n  options:\n    revision: fixed\n    device: cpu\n")
    monkeypatch.setattr(cli, "create", create)
    monkeypatch.setenv("LIKENESSFORGE_PROJECT_ROOT", str(tmp_path))
    assert cli.main(["models", "download", "sd15-inpaint-lowvram", "--config", str(config)]) == 0
    assert seen == {"kind": "modifier", "name": "sd15-inpaint-lowvram",
                    "options": {"revision": "fixed", "device": "cpu"}}
    assert str(tmp_path) in capsys.readouterr().out


def test_cache_status_does_not_load_torch_for_automatic_device(monkeypatch, tmp_path):
    store = ModelStore(tmp_path)
    modifier = SD15InpaintModifier(store)
    monkeypatch.setattr(modifier, "_model_variant", lambda: (_ for _ in ()).throw(AssertionError("device probe")))
    monkeypatch.setattr(store, "is_diffusers_snapshot_cached", lambda repo, **options: options["variant"] == "fp16")
    assert modifier.is_model_cached()
