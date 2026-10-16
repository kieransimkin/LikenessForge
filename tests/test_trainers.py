import json
from pathlib import Path
from types import SimpleNamespace

from likenessforge.plugins.trainers.diffusers_scripts import SD15LoraTrainer
from likenessforge.types import TrainingRequest


def test_trainer_uses_resolved_snapshot_and_local_bounded_logging(monkeypatch, tmp_path):
    requested = []
    snapshot = tmp_path / "models" / "snapshot"
    store = SimpleNamespace(
        root=tmp_path / "models", hf=tmp_path / "models" / "huggingface", tools=tmp_path / "tools",
        ensure_diffusers_snapshot=lambda repo, **kwargs: requested.append((repo, kwargs)) or snapshot,
    )
    trainer = SD15LoraTrainer(store, revision="pinned-sha")
    script = tmp_path / "trainer.py"
    monkeypatch.setattr(trainer, "_ensure_script", lambda: script)
    captured = {}
    request = TrainingRequest(tmp_path / "dataset", tmp_path / "output", "photo of lf_subject person", 512)

    def launch(cmd, **kwargs):
        captured.update(cmd=cmd, **kwargs)
        (request.output_dir / "pytorch_lora_weights.safetensors").write_bytes(b"synthetic")
        return SimpleNamespace(stdout=iter(["completed\n"]), wait=lambda: 0)

    monkeypatch.setattr("likenessforge.plugins.trainers.diffusers_scripts.subprocess.Popen", launch)
    result = trainer.train(request)
    cmd = captured["cmd"]
    assert requested == [(trainer.base_model_id, {"revision": "pinned-sha"})]
    assert cmd[cmd.index("--pretrained_model_name_or_path") + 1] == str(snapshot)
    assert cmd[cmd.index("--report_to") + 1] == "tensorboard"
    assert cmd[cmd.index("--checkpoints_total_limit") + 1] == "1"
    assert "--push_to_hub" not in cmd
    assert "--cache_dir" not in cmd
    assert captured["env"]["HF_HUB_CACHE"] == str(store.hf)
    assert captured["env"]["HF_HOME"] == str(store.root / "hf-home")
    contract = json.loads((Path(__file__).parent / "fixtures" / "sd15-trainer-arguments.json").read_text("utf-8"))
    assert {arg.split("=")[0] for arg in cmd[4:] if arg.startswith("--")} <= set(contract["arguments"])
    assert result == request.output_dir / "pytorch_lora_weights.safetensors"
