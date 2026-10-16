"""Supply missing Torch module-name metadata without altering package code.

Python 3.12 importlib.metadata otherwise stats every Torch wheel file during
Transformers package discovery. Run in the intended venv; dry run is the default.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from importlib.metadata import distribution
from pathlib import Path, PurePosixPath


def prepare_torch_metadata(dist, environment: Path, *, apply: bool = False) -> dict:
    record = dist.read_text("RECORD")
    if not record:
        raise ValueError("Torch distribution has no wheel RECORD")
    files = [row[0] for row in csv.reader(io.StringIO(record))]
    metadata = next(PurePosixPath(x) for x in files if x.endswith(".dist-info/METADATA"))
    target = Path(dist.locate_file(metadata.parent / "top_level.txt")).resolve()
    if not target.is_relative_to(environment.resolve()):
        raise ValueError("Refusing to modify metadata outside the selected Python environment")
    if target.exists():
        return {"state": "already_present", "file": str(target)}
    modules = sorted({p.parts[0] for name in files for p in [PurePosixPath(name)]
                      if len(p.parts) > 1 and p.parts[0].isidentifier()
                      and Path(dist.locate_file(p.parts[0])).is_dir()})
    if "torch" not in modules:
        raise ValueError("Installed Torch package directory was not found")
    if apply:
        target.write_text("\n".join(modules) + "\n", encoding="utf-8")
    return {"state": "prepared" if apply else "dry_run", "file": str(target), "modules": modules}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Add the missing metadata in this venv")
    args = parser.parse_args()
    print(json.dumps(prepare_torch_metadata(distribution("torch"), Path(sys.prefix), apply=args.apply)))
