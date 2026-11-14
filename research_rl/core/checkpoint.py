"""Checkpoint contents and ResearchRL-specific resume state.

The HF ``Trainer`` (and therefore every TRL trainer) already saves, per ``checkpoint-<step>/``:

* model weights (``model.safetensors``),
* optimizer state (``optimizer.pt`` – for a ``HookedOptimizer`` this also holds the update-rule
  state and the nominal/applied/skipped step counters),
* LR scheduler (``scheduler.pt``), RNG states (``rng_state.pth``),
* ``trainer_state.json`` (global step, epoch, log history) from which the data-loader position is
  restored on resume.

ResearchRL adds ``research_rl_state.json`` (config version, policy version, step counters, data
progress, JSON algorithm state) and ``research_rl_components.pt`` (tensor state of registered
stateful components such as recurrent states or expert routers).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

from research_rl.core.config import CONFIG_VERSION

RESEARCH_STATE_FILE = "research_rl_state.json"
COMPONENTS_FILE = "research_rl_components.pt"

# Files the HF Trainer writes that a full resume relies on.
REQUIRED_HF_FILES = {
    "optimizer": "optimizer.pt",
    "scheduler": "scheduler.pt",
    "rng_state": "rng_state.pth",
    "trainer_state": "trainer_state.json",
}
WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "adapter_model.safetensors",
)


@runtime_checkable
class StatefulComponent(Protocol):
    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


def save_research_state(checkpoint_dir: str | Path, state: Mapping[str, Any]) -> Path:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload = {"config_version": CONFIG_VERSION, **state}
    out = path / RESEARCH_STATE_FILE
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out


def load_research_state(checkpoint_dir: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_dir) / RESEARCH_STATE_FILE
    if not path.exists():
        return {}
    state = json.loads(path.read_text())
    if state.get("config_version") != CONFIG_VERSION:
        raise ValueError(
            f"checkpoint {checkpoint_dir} has config_version {state.get('config_version')!r}, "
            f"this ResearchRL expects {CONFIG_VERSION!r}"
        )
    return state


def save_components(checkpoint_dir: str | Path, components: Mapping[str, StatefulComponent]) -> Path | None:
    if not components:
        return None
    out = Path(checkpoint_dir) / COMPONENTS_FILE
    torch.save({name: comp.state_dict() for name, comp in components.items()}, out)
    return out


def load_components(checkpoint_dir: str | Path, components: Mapping[str, StatefulComponent]) -> list[str]:
    """Restore registered components; returns the names that were restored."""
    path = Path(checkpoint_dir) / COMPONENTS_FILE
    if not components or not path.exists():
        return []
    saved = torch.load(path, map_location="cpu", weights_only=True)
    restored = []
    for name, comp in components.items():
        if name in saved:
            comp.load_state_dict(saved[name])
            restored.append(name)
    return restored


def checkpoint_inventory(checkpoint_dir: str | Path) -> dict[str, bool]:
    """Which resume-relevant artefacts exist in a checkpoint directory."""
    path = Path(checkpoint_dir)
    inv = {key: (path / fname).exists() for key, fname in REQUIRED_HF_FILES.items()}
    inv["model_weights"] = any((path / f).exists() for f in WEIGHT_FILES)
    inv["research_state"] = (path / RESEARCH_STATE_FILE).exists()
    inv["components"] = (path / COMPONENTS_FILE).exists()
    return inv


def latest_checkpoint(output_dir: str | Path) -> str | None:
    from transformers.trainer_utils import get_last_checkpoint

    if not os.path.isdir(output_dir):
        return None
    return get_last_checkpoint(str(output_dir))
