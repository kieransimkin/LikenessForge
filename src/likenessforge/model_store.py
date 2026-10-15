from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from platformdirs import user_cache_dir

LOG = logging.getLogger("likenessforge")


@dataclass(frozen=True)
class ModelProgress:
    """Structured model-download status suitable for CLI, GUI and future service frontends."""

    stage: str
    repo_id: str
    description: str = ""
    current: float | None = None
    total: float | None = None
    unit: str = "B"
    path: str | None = None
    error: str | None = None

    @property
    def fraction(self) -> float | None:
        if self.current is None or not self.total:
            return None
        return max(0.0, min(1.0, float(self.current) / float(self.total)))


ProgressCallback = Callable[[ModelProgress], None]


class ModelStore:
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ):
        self.root = Path(cache_dir or user_cache_dir("LikenessForge", "Kieran Simkin"))
        self.hf = self.root / "huggingface"
        self.tools = self.root / "tools"
        self.hf.mkdir(parents=True, exist_ok=True)
        self.tools.mkdir(parents=True, exist_ok=True)
        self.progress_callback = progress_callback

    def set_progress_callback(self, callback: ProgressCallback | None) -> None:
        self.progress_callback = callback

    def _emit(self, stage: str, repo_id: str, **kwargs) -> None:
        if self.progress_callback:
            self.progress_callback(ModelProgress(stage=stage, repo_id=repo_id, **kwargs))

    def _reporting_tqdm(self, repo_id: str):
        if self.progress_callback is None:
            return None

        from tqdm.auto import tqdm as BaseTqdm

        store = self

        class ReportingTqdm(BaseTqdm):
            """tqdm adapter that mirrors Hugging Face transfer progress to ModelStore listeners."""

            def __init__(self, *args, **kwargs):
                self._lf_last_emit = 0.0
                super().__init__(*args, **kwargs)
                self._lf_emit(force=True)

            def _lf_emit(self, force: bool = False, stage: str = "file_progress") -> None:
                now = time.monotonic()
                if not force and now - self._lf_last_emit < 0.1:
                    return
                self._lf_last_emit = now
                store._emit(
                    stage,
                    repo_id,
                    description=str(getattr(self, "desc", "") or "Downloading"),
                    current=float(getattr(self, "n", 0) or 0),
                    total=float(self.total) if getattr(self, "total", None) else None,
                    unit=str(getattr(self, "unit", "B") or "B"),
                )

            def update(self, n=1):
                result = super().update(n)
                self._lf_emit()
                return result

            def close(self):
                self._lf_emit(force=True, stage="file_complete")
                return super().close()

        return ReportingTqdm

    def is_hf_snapshot_cached(self, repo_id: str, revision: str | None = None) -> bool:
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=str(self.hf),
                local_files_only=True,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - cache probes may fail through multiple HF backends
            LOG.debug("Hugging Face snapshot not cached for %s: %s", repo_id, exc)
            return False

    def ensure_hf_snapshot(
        self,
        repo_id: str,
        revision: str | None = None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
    ) -> Path:
        LOG.info("ensuring Hugging Face model is cached: %s", repo_id)
        self._emit("model_start", repo_id, description="Checking local cache / Hugging Face")
        kwargs = {
            "repo_id": repo_id,
            "revision": revision,
            "cache_dir": str(self.hf),
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        tqdm_class = self._reporting_tqdm(repo_id)
        if tqdm_class is not None:
            kwargs["tqdm_class"] = tqdm_class
        try:
            path = snapshot_download(**kwargs)
        except Exception as exc:
            self._emit("model_error", repo_id, description="Download failed", error=str(exc))
            raise
        result = Path(path)
        self._emit("model_ready", repo_id, description="Model ready", path=str(result))
        return result

    def ensure_hf_file(self, repo_id: str, filename: str, revision: str | None = None) -> Path:
        LOG.info("ensuring Hugging Face model file is cached: %s/%s", repo_id, filename)
        self._emit("model_start", repo_id, description=filename)
        kwargs = {
            "repo_id": repo_id,
            "filename": filename,
            "revision": revision,
            "cache_dir": str(self.hf),
        }
        tqdm_class = self._reporting_tqdm(repo_id)
        if tqdm_class is not None:
            kwargs["tqdm_class"] = tqdm_class
        try:
            path = hf_hub_download(**kwargs)
        except Exception as exc:
            self._emit("model_error", repo_id, description=filename, error=str(exc))
            raise
        result = Path(path)
        self._emit("model_ready", repo_id, description=filename, path=str(result))
        return result
