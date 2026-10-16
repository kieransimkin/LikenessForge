from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

LOG = logging.getLogger("likenessforge")
PROJECT_ROOT_ENV = "LIKENESSFORGE_PROJECT_ROOT"

# Load one standard Diffusers safetensors representation. Repository snapshots
# may also contain full checkpoints, Flax/ONNX exports and precision variants.
DIFFUSERS_ALLOW_PATTERNS = [
    "model_index.json", "README.md", "LICENSE*",
    "scheduler/*", "tokenizer/*", "tokenizer_2/*", "tokenizer_3/*", "feature_extractor/*",
    "*/config.json", "*/model.safetensors", "*/model.safetensors.index.json",
    "*/model-*.safetensors", "*/diffusion_pytorch_model.safetensors",
    "*/diffusion_pytorch_model.safetensors.index.json", "*/diffusion_pytorch_model-*.safetensors",
]


def find_project_root() -> Path:
    """Locate the LikenessForge project root used for project-local model storage."""
    explicit = os.environ.get(PROJECT_ROOT_ENV)
    if explicit:
        return Path(explicit).expanduser().resolve()

    starts = (Path.cwd(), Path(__file__).resolve().parent)
    for start in starts:
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "likenessforge").is_dir():
                return candidate.resolve()

    # A wheel may be executed outside a source checkout. In that case the caller's
    # working directory is the only meaningful project directory available.
    return Path.cwd().resolve()


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
        project_root: str | Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ):
        self.project_root = (
            Path(project_root).expanduser().resolve() if project_root is not None else find_project_root()
        )
        self.root = self.project_root / "models"
        self.hf = self.root / "huggingface"
        self.tools = self.root / "tools"
        self.hf.mkdir(parents=True, exist_ok=True)
        self.tools.mkdir(parents=True, exist_ok=True)
        self.progress_callback = progress_callback

    def set_progress_callback(self, callback: ProgressCallback | None) -> None:
        self.progress_callback = callback

    def resolve_local_or_hf(self, value: str | Path) -> Path:
        """Resolve an existing local asset or cache a Hugging Face repository project-locally."""
        local = Path(value).expanduser()
        if local.exists():
            return local.resolve()
        return self.ensure_hf_snapshot(str(value))

    def ensure_diffusers_snapshot(self, repo_id: str, revision: str | None = None) -> Path:
        """Cache the supported safetensors pipeline without unrelated weight formats."""
        return self.ensure_hf_snapshot(repo_id, revision=revision, allow_patterns=DIFFUSERS_ALLOW_PATTERNS)

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
