"""Identify tested release distributions without modifying their bytes."""
import hashlib
import json
import subprocess
from pathlib import Path

from validate_distribution import validate

root = Path(__file__).resolve().parents[1]
dist = root / "dist"
version = validate(dist)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
assets = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
          for p in sorted(dist.iterdir()) if p.name.endswith((".whl", ".tar.gz"))}
assert len(assets) == 2, "Release requires exactly its tested wheel and source distribution"
(dist / "SHA256SUMS.txt").write_text("".join(f'{v["sha256"]}  {k}\n' for k, v in assets.items()), encoding="utf-8")
(dist / "package-manifest.json").write_text(json.dumps({"version": version, "sourceCommit": commit,
    "website": "https://kieransimkin.co.uk/danceflow/", "assets": assets}, indent=2) + "\n", encoding="utf-8")
