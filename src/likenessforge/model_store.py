from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from platformdirs import user_cache_dir


LOG = logging.getLogger("likenessforge")


class ModelStore:
    def __init__(self, cache_dir: str | Path | None = None):
        self.root = Path(cache_dir or user_cache_dir("LikenessForge", "Kieran Simkin"))
        self.hf = self.root / "huggingface"
        self.tools = self.root / "tools"
        self.hf.mkdir(parents=True, exist_ok=True)
        self.tools.mkdir(parents=True, exist_ok=True)

    def ensure_hf_snapshot(
        self,
        repo_id: str,
        revision: str | None = None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
    ) -> Path:
        LOG.info("ensuring Hugging Face model is cached: %s", repo_id)
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(self.hf),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        return Path(path)

    def ensure_hf_file(self, repo_id: str, filename: str, revision: str | None = None) -> Path:
        LOG.info("ensuring Hugging Face model file is cached: %s/%s", repo_id, filename)
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=str(self.hf),
        )
        return Path(path)
