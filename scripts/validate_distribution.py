"""Validate released metadata and private-file exclusions in built distributions."""
from __future__ import annotations

import ast
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

SITE = "https://kieransimkin.co.uk/danceflow/"
REPO = "https://github.com/kieransimkin/LikenessForge"
ROOT = Path(__file__).resolve().parents[1]


def check_metadata(text: str, version: str) -> None:
    data = Parser().parsestr(text)
    assert data["Name"] == "likenessforge", "Unexpected package identity"
    assert data["Version"] == version, "Distribution/runtime version mismatch"
    assert SITE in data["Summary"], "Summary must retain the website"
    urls = dict(item.split(", ", 1) for item in data.get_all("Project-URL", []))
    assert urls["DanceFlow ecosystem"] == SITE, "Missing ecosystem URL"
    assert urls["Repository"] == REPO, "Wrong canonical repository"
    assert urls["Documentation"] == REPO + "#readme", "Missing documentation URL"
    assert "https://kieransimkin.co.uk/" in data.get_payload(), "README author website missing"
    assert REPO + "/pulls" in data.get_payload(), "README contribution route missing"


def check_paths(names: list[str]) -> None:
    for name in names:
        path = PurePosixPath(name)
        assert not path.is_absolute() and ".." not in path.parts, "Unsafe archive member"
        assert "\\" not in name and ":" not in name, "Nonportable archive member"
        blocked = {"models", "outputs", "loras", "likenessforge-debug", ".venv", ".env", ".git"}
        assert not blocked.intersection(part.lower() for part in path.parts), "Private/cache path packaged"
        assert path.suffix.lower() not in {".safetensors", ".ckpt", ".pt", ".pth", ".jpg", ".jpeg"}, "Private image or weights packaged"
        if path.suffix.lower() in {".png", ".webp", ".bmp", ".tif", ".tiff"}:
            assert path.name in {"logo.png", "logo-monochrome.png"} and path.parent.parts[-2:] == ("docs", "branding"), "Unreviewed image packaged"


def validate(folder: Path) -> str:
    module = ast.parse((ROOT / "src/likenessforge/__init__.py").read_text("utf-8"))
    version = next(ast.literal_eval(node.value) for node in module.body
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__"
                                                          for t in node.targets))
    wheel = folder / f"likenessforge-{version}-py3-none-any.whl"
    source = folder / f"likenessforge-{version}.tar.gz"
    logo = (ROOT / "src/likenessforge/assets/logo.svg").read_bytes()
    with zipfile.ZipFile(wheel) as package:
        check_paths(package.namelist())
        metadata = package.read(f"likenessforge-{version}.dist-info/METADATA").decode("utf-8")
        check_metadata(metadata, version)
        assert package.read("likenessforge/assets/logo.svg") == logo, "Packaged logo differs"
    with tarfile.open(source) as package:
        check_paths(package.getnames())
        assert not any(item.issym() or item.islnk() for item in package), "Unexpected archive link"
        prefix = f"likenessforge-{version}/"
        check_metadata(package.extractfile(prefix + "PKG-INFO").read().decode("utf-8"), version)
        assert package.extractfile(prefix + "docs/branding/logo.svg").read() == logo, "Source logo differs"
    return version


if __name__ == "__main__":
    checked = validate(Path(sys.argv[1] if len(sys.argv) > 1 else "dist"))
    print(f"Validated LikenessForge {checked} wheel/source metadata, logo and private-file exclusions")
