from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = spec_from_file_location("runtime_setup", Path(__file__).parents[1] / "scripts/prepare_windows_runtime.py")
setup = module_from_spec(spec)
spec.loader.exec_module(setup)


def test_module_metadata_is_derived_and_existing_package_bytes_are_preserved(tmp_path):
    site = tmp_path / "site-packages"
    metadata = site / "torch-1.dist-info"
    metadata.mkdir(parents=True)
    (site / "torch").mkdir()
    (site / "torchgen").mkdir()
    record = "torch/__init__.py,,\ntorchgen/__init__.py,,\ntorch-1.dist-info/METADATA,,\n"
    (metadata / "RECORD").write_text(record)
    (metadata / "METADATA").write_text("Name: torch\n")
    dist = SimpleNamespace(read_text=lambda name: (metadata / name).read_text(), locate_file=lambda p: site / p)
    original = {p: p.read_bytes() for p in metadata.iterdir()}
    result = setup.prepare_torch_metadata(dist, tmp_path)
    assert result["state"] == "dry_run" and not (metadata / "top_level.txt").exists()
    result = setup.prepare_torch_metadata(dist, tmp_path, apply=True)
    assert result["modules"] == ["torch", "torchgen"]
    assert (metadata / "top_level.txt").read_text() == "torch\ntorchgen\n"
    assert all(p.read_bytes() == content for p, content in original.items())
    assert setup.prepare_torch_metadata(dist, tmp_path, apply=True)["state"] == "already_present"
    with pytest.raises(ValueError, match="outside"):
        setup.prepare_torch_metadata(dist, tmp_path / "other", apply=True)
