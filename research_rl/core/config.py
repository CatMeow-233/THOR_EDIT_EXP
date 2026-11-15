"""Versioned experiment configuration.

A single :class:`ExperimentConfig` describes one run of one study: which model, which data,
which (TRL) trainer, which ResearchRL extensions. It round-trips through JSON/YAML and carries a
``config_version`` so that stored runs can be re-read after the schema evolves.

The ``trainer.args`` mapping is passed *verbatim* to the corresponding TRL ``*Config`` class
(``SFTConfig``, ``GRPOConfig``, ...); ResearchRL does not re-declare TRL's hundreds of options.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_VERSION = "1.0"

_SUPPORTED_DTYPES = ("float32", "bfloat16", "float16", "auto")


class ConfigError(ValueError):
    pass


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,127}$")


def _is_safe_path_component(value: str) -> bool:
    """study_id / run_name become directory names: no separators, no '..', no leading dot."""
    return bool(_SAFE_COMPONENT.match(value)) and ".." not in value


@dataclass
class ModelSpec:
    """Where the policy weights come from."""

    name_or_path: str
    revision: str | None = None  # HF commit / tag; recorded in the experiment record
    dtype: str = "float32"
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in _SUPPORTED_DTYPES:
            raise ConfigError(f"model.dtype must be one of {_SUPPORTED_DTYPES}, got {self.dtype!r}")


@dataclass
class DataSpec:
    """Identity of the training data (the dataset object itself is passed at run time)."""

    name: str
    split: str = "train"
    revision: str | None = None
    notes: str = ""


@dataclass
class TrainerSpec:
    """Which registered trainer adapter to use and the raw TRL config arguments."""

    algorithm: str  # registry key: "sft", "dpo", "grpo", "rloo", "distillation", ...
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimSpec:
    """ResearchRL optimizer-extension settings (see :mod:`research_rl.optim`)."""

    hooked: bool = True  # wrap the TRL-created optimizer in HookedOptimizer
    # If set, gradient clipping is performed *inside* HookedOptimizer, after post-grad hooks/masks,
    # and the HF/TRL ``max_grad_norm`` is forced to 0 so the norm is never computed on masked grads.
    clip_in_wrapper: float | None = None
    # Skip the optimizer step entirely when every advantage in the accumulation window is zero
    # (RL trainers only). Counted separately from nominal steps.
    skip_zero_advantage_steps: bool = False
    # Precision of low-precision (bf16/fp16) trainable parameters, see HookedOptimizer: keep fp32
    # master weights, how the served weights are rounded ("nearest" | "stochastic"), and what to do
    # when low-precision trainable parameters have no masters ("warn" | "error" | "off").
    # Stochastic rounding is seeded from the experiment seed.
    master_weights: bool = False
    write_rounding: str = "nearest"
    precision_check: str = "warn"

    def __post_init__(self) -> None:
        if self.write_rounding not in ("nearest", "stochastic"):
            raise ConfigError(
                f"optim.write_rounding must be 'nearest' or 'stochastic', got {self.write_rounding!r}"
            )
        if self.precision_check not in ("warn", "error", "off"):
            raise ConfigError(
                f"optim.precision_check must be 'warn', 'error' or 'off', got {self.precision_check!r}"
            )
        if not isinstance(self.master_weights, bool):
            raise ConfigError(f"optim.master_weights must be a bool, got {self.master_weights!r}")
        if self.write_rounding == "stochastic" and not self.hooked:
            raise ConfigError("optim.write_rounding='stochastic' needs optim.hooked=True")
        if self.master_weights and not self.hooked:
            raise ConfigError("optim.master_weights=True needs optim.hooked=True")


@dataclass
class ExperimentConfig:
    study_id: str
    model: ModelSpec
    data: DataSpec
    trainer: TrainerSpec
    seed: int = 0
    output_dir: str = "runs"
    run_name: str | None = None
    optim: OptimSpec = field(default_factory=OptimSpec)
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)  # free-form per-study settings
    config_version: str = CONFIG_VERSION

    def __post_init__(self) -> None:
        if not _is_safe_path_component(self.study_id):
            raise ConfigError(f"invalid study_id {self.study_id!r} (letters, digits, '-', '_', '.' only)")
        if self.run_name is not None and not _is_safe_path_component(self.run_name):
            raise ConfigError(f"invalid run_name {self.run_name!r} (letters, digits, '-', '_', '.' only)")
        if self.config_version != CONFIG_VERSION:
            raise ConfigError(
                f"config_version {self.config_version!r} is not supported (expected {CONFIG_VERSION!r})"
            )

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        raw = dict(raw)
        if "config_version" not in raw:
            raise ConfigError(
                f"config has no 'config_version' (expected {CONFIG_VERSION!r}); pre-versioning configs are rejected"
            )
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ConfigError(f"unknown top-level config keys: {sorted(unknown)}")
        try:
            raw["model"] = ModelSpec(**raw["model"])
            raw["data"] = DataSpec(**raw["data"])
            raw["trainer"] = TrainerSpec(**raw["trainer"])
            if "optim" in raw:
                raw["optim"] = OptimSpec(**raw["optim"])
        except TypeError as exc:  # unknown nested keys
            raise ConfigError(str(exc)) from exc
        except KeyError as exc:
            raise ConfigError(f"missing required config section {exc}") from exc
        return cls(**raw)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in (".yaml", ".yml"):
            import yaml

            path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False))
        return path

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        path = Path(path)
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            import yaml

            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} does not contain a mapping")
        return cls.from_dict(raw)

    @property
    def resolved_run_name(self) -> str:
        return self.run_name or f"{self.trainer.algorithm}-seed{self.seed}"

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.study_id / self.resolved_run_name
