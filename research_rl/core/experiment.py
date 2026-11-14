"""Unified experiment records.

Every run writes ``<output_dir>/<study_id>/<run_name>/``::

    config.json          the ExperimentConfig (versioned)
    metrics.jsonl        one JSON object per TRL log event (loss, reward, grad_norm, tokens, ...)
    record.json          the run summary (identity, counters, final metrics, checkpoints, evals)

Honesty rules enforced here:

* ``hardware.gpu_metrics`` is ``"NOT_TESTED"`` unless the values were *measured* on a CUDA
  device in this process (``record_gpu_metrics`` refuses to store numbers otherwise).
* ``hardware.device`` records what the run actually used (``cpu`` in CI).
* Step counters distinguish *nominal* optimizer steps (what the HF Trainer counted) from
  *applied* parameter updates (what really changed the weights).
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research_rl.core.config import ExperimentConfig

NOT_TESTED = "NOT_TESTED"
RECORD_VERSION = "1.0"


def _jsonable(value: Any) -> Any:
    """Best-effort JSON conversion for logs/records (NaN/inf -> strings, tensors/numpy -> numbers)."""
    if isinstance(value, torch.Tensor):
        return _jsonable(value.item() if value.numel() == 1 else value.tolist())
    if isinstance(value, (np.generic, np.ndarray)):
        return _jsonable(value.tolist())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class _NotCanonical(TypeError):
    pass


def _canonical(value: Any) -> Any:
    """Strict, deterministic conversion used for content hashes (no repr() of arbitrary objects)."""
    if isinstance(value, torch.Tensor):
        return {"__tensor__": str(value.dtype), "data": _canonical(value.tolist())}
    if isinstance(value, (np.generic, np.ndarray)):
        return _canonical(value.tolist())
    if isinstance(value, float):
        return repr(value) if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return {"__bytes__": hashlib.sha256(value).hexdigest()}
    raise _NotCanonical(f"cannot hash a {type(value).__qualname__} deterministically")


def dataset_identity(
    dataset: Any, name: str | None = None, max_rows_for_hash: int = 200_000
) -> dict[str, Any]:
    """Stable identity of a dataset: row count, column names and a content hash.

    Works for ``datasets.Dataset`` and any sequence of JSON-serialisable rows. For datasets larger
    than ``max_rows_for_hash`` only the HF fingerprint is recorded.
    """
    ident: dict[str, Any] = {"name": name}
    try:
        n = len(dataset)
    except TypeError:
        return {**ident, "num_rows": None, "content_sha256": None, "note": "unsized dataset"}
    ident["num_rows"] = n
    ident["columns"] = list(getattr(dataset, "column_names", []) or [])
    hf_fp = getattr(dataset, "_fingerprint", None)
    if hf_fp is not None:
        ident["hf_fingerprint"] = hf_fp
    if n <= max_rows_for_hash:
        h = hashlib.sha256()
        try:
            for i in range(n):
                h.update(json.dumps(_canonical(dataset[i]), sort_keys=True, ensure_ascii=False).encode())
                h.update(b"\n")
            ident["content_sha256"] = h.hexdigest()
        except _NotCanonical as exc:
            ident["content_sha256"] = None
            ident["note"] = f"no content hash: {exc}"
    else:
        ident["content_sha256"] = None
    return ident


def model_identity(
    model: Any, revision: str | None = None, hash_weights: bool | None = None
) -> dict[str, Any]:
    """Model revision info: HF commit hash if known, and (for small models) a weights hash."""
    from research_rl.models.model_interface import count_parameters, fingerprint

    config = getattr(model, "config", None)
    n_params = count_parameters(model)
    if hash_weights is None:
        hash_weights = n_params <= 50_000_000  # hashing multi-GB weights on every run is not free
    return {
        "class": type(model).__name__,
        "name_or_path": getattr(config, "_name_or_path", None),
        "commit_hash": getattr(config, "_commit_hash", None),
        "requested_revision": revision,
        "num_parameters": n_params,
        "num_trainable_parameters": count_parameters(model, trainable_only=True),
        "weights_sha256": fingerprint(model) if hash_weights else None,
    }


def environment_info() -> dict[str, Any]:
    import importlib.metadata as md

    versions: dict[str, str | None] = {}
    for pkg in ("torch", "transformers", "trl", "accelerate", "peft", "datasets", "safetensors"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "device": "cuda" if cuda else "cpu",
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda else None,
        "cuda_version": torch.version.cuda,
    }


class ExperimentRecorder:
    """Collects identity, metrics, checkpoints and evaluations for one run."""

    METRIC_KEYS = ("loss", "reward", "grad_norm", "num_tokens", "learning_rate")

    def __init__(self, config: ExperimentConfig, run_dir: str | Path | None = None) -> None:
        self.config = config
        self.run_dir = Path(run_dir) if run_dir is not None else config.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        config.save(self.run_dir / "config.json")
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.record: dict[str, Any] = {
            "record_version": RECORD_VERSION,
            "study_id": config.study_id,
            "run_name": config.resolved_run_name,
            "algorithm": config.trainer.algorithm,
            "seed": config.seed,
            "config_version": config.config_version,
            "config": config.to_dict(),
            "config_sha256": hashlib.sha256(
                json.dumps(_jsonable(config.to_dict()), sort_keys=True).encode()
            ).hexdigest(),
            "status": "created",
            "created_unix": time.time(),
            "model": None,
            "dataset": None,
            "trainable_parameters": None,
            "steps": {
                "nominal_optimizer_steps": 0,
                "applied_updates": 0,
                "skipped_steps": 0,
                "global_step": 0,
            },
            "training_tokens": None,
            "final_metrics": {},
            "checkpoints": [],
            "evaluations": [],
            "environment": environment_info(),
            "hardware": {"gpu_metrics": NOT_TESTED},
        }
        self._history: dict[str, list[float]] = {}
        # A previous run in the same directory: keep its rows/record aside. They are restored up to the
        # checkpoint step by resume_from(); a fresh (non-resumed) run starts with clean files.
        self._prior_rows: list[dict[str, Any]] = []
        self._prior_record: dict[str, Any] = {}
        if self.metrics_path.exists():
            for line in self.metrics_path.read_text().splitlines():
                if line.strip():
                    self._prior_rows.append(json.loads(line))
        if (self.run_dir / "record.json").exists():
            try:
                self._prior_record = json.loads((self.run_dir / "record.json").read_text())
            except json.JSONDecodeError:
                self._prior_record = {}
        self.metrics_path.write_text("")
        self._flush()

    def resume_from(self, global_step: int) -> None:
        """Continue the record of an interrupted run from checkpoint ``global_step``.

        Metric rows, checkpoints and evaluations of the previous run up to that step are kept (later
        ones are discarded, they will be re-produced), and the metric history is rebuilt from them.
        """
        rows = [r for r in self._prior_rows if int(r.get("step", 0)) <= global_step]
        with self.metrics_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        self._history = {}
        for r in rows:
            for k, v in r.items():
                if k != "step" and isinstance(v, (int, float)) and not isinstance(v, bool):
                    self._history.setdefault(k, []).append(float(v))
            if "num_tokens" in r:
                self.record["training_tokens"] = r["num_tokens"]
        self.record["checkpoints"] = [
            c
            for c in self._prior_record.get("checkpoints", [])
            if int(c.get("global_step", 0)) <= global_step
        ]
        self.record["evaluations"] = [
            e
            for e in self._prior_record.get("evaluations", [])
            if e.get("step") is None or int(e["step"]) <= global_step
        ]
        self.record["resumed_from_step"] = int(global_step)
        self._flush()

    # ------------------------------------------------------------------ identity
    def set_model(self, model: Any, revision: str | None = None, hash_weights: bool | None = None) -> None:
        self.record["model"] = model_identity(model, revision or self.config.model.revision, hash_weights)
        self.record["trainable_parameters"] = self.record["model"]["num_trainable_parameters"]
        self._flush()

    def set_dataset(self, dataset: Any, name: str | None = None) -> None:
        self.record["dataset"] = dataset_identity(dataset, name or self.config.data.name)
        self._flush()

    # ------------------------------------------------------------------ metrics
    def log_metrics(self, step: int, logs: Mapping[str, Any]) -> None:
        row = {"step": step, **{k: _jsonable(v) for k, v in logs.items()}}
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        for k, v in logs.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                v = v.item()
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                self._history.setdefault(k, []).append(float(v))
        if "num_tokens" in logs:
            self.record["training_tokens"] = _jsonable(logs["num_tokens"])

    def history(self, key: str) -> list[float]:
        return list(self._history.get(key, []))

    def set_steps(self, *, nominal: int, applied: int, skipped: int, global_step: int) -> None:
        self.record["steps"] = {
            "nominal_optimizer_steps": int(nominal),
            "applied_updates": int(applied),
            "skipped_steps": int(skipped),
            "global_step": int(global_step),
        }

    def set_extra(self, key: str, value: Any) -> None:
        """Store an algorithm-specific top-level field (e.g. reference/teacher model identity)."""
        reserved = {"record_version", "study_id", "run_name", "config", "config_sha256", "steps", "hardware"}
        if key in reserved:
            raise KeyError(f"{key!r} is a reserved record field")
        self.record[key] = _jsonable(value)
        self._flush()

    def add_checkpoint(self, path: str | Path, step: int) -> None:
        entry = {"path": str(path), "global_step": int(step)}
        if entry not in self.record["checkpoints"]:
            self.record["checkpoints"].append(entry)
        self._flush()

    def add_evaluation(self, name: str, results: Mapping[str, Any], step: int | None = None) -> None:
        self.record["evaluations"].append({"name": name, "step": step, "results": _jsonable(results)})
        self._flush()

    def record_gpu_metrics(self, metrics: Mapping[str, Any]) -> None:
        """Store measured GPU metrics. Refuses unless running on CUDA right now."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU metrics can only be recorded from a CUDA run; they stay NOT_TESTED on CPU"
            )
        self.record["hardware"]["gpu_metrics"] = {
            "measured": True,
            "device": torch.cuda.get_device_name(0),
            **_jsonable(dict(metrics)),
        }
        self._flush()

    def finalize(self, status: str = "completed") -> dict[str, Any]:
        final = {}
        for key in self.METRIC_KEYS:
            if self._history.get(key):
                final[key] = self._history[key][-1]
        for key, values in self._history.items():
            if key.startswith(("rewards/", "eval_")) and values:
                final[key] = values[-1]
        self.record["final_metrics"] = final
        self.record["status"] = status
        self.record["finished_unix"] = time.time()
        self._flush()
        return _jsonable(self.record)

    def _flush(self) -> None:
        (self.run_dir / "record.json").write_text(json.dumps(_jsonable(self.record), indent=2))

    @staticmethod
    def load(run_dir: str | Path) -> dict[str, Any]:
        return json.loads((Path(run_dir) / "record.json").read_text())
