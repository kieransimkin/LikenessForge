from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

spec = spec_from_file_location("validate_distribution", Path(__file__).parents[1] / "scripts/validate_distribution.py")
validator = module_from_spec(spec)
spec.loader.exec_module(validator)


@pytest.mark.parametrize("name", ["models/model.bin", "outputs/portrait.png", "identity.safetensors",
                                 "private/photo.jpg", "docs/portrait.png", "../escape.py", "C:/private.txt"])
def test_distribution_rejects_private_or_unsafe_members(name):
    with pytest.raises(AssertionError):
        validator.check_paths([name])


def test_distribution_accepts_public_package_assets():
    validator.check_paths(["likenessforge/assets/logo.svg", "docs/branding/logo.png", "tests/test_core.py"])


def test_metadata_rejects_stale_version_and_missing_website():
    with pytest.raises(AssertionError, match="version"):
        validator.check_metadata("Name: likenessforge\nVersion: 0.1.0\n", "0.4.2")
    with pytest.raises(AssertionError, match="website"):
        validator.check_metadata("Name: likenessforge\nVersion: 0.4.2\nSummary: Face repair\n", "0.4.2")
