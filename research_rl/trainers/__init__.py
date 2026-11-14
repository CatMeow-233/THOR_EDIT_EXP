"""TRL trainer adapters.

``get_adapter("grpo")`` returns the registered adapter; ``build_run(config, ...)`` builds a
:class:`~research_rl.trainers.base.ResearchRun` (TRL trainer + experiment recorder).
Adapter modules register themselves in :data:`research_rl.core.registry.TRAINERS` on import.
"""

from __future__ import annotations

import importlib
from typing import Any

from research_rl.core.config import ExperimentConfig
from research_rl.core.registry import TRAINERS
from research_rl.trainers.base import (
    ResearchCallback,
    ResearchRun,
    ResearchTrainerMixin,
    TrainerAdapter,
    research_trainer_class,
)

ADAPTER_MODULES = (
    "research_rl.trainers.sft_adapter",
    "research_rl.trainers.dpo_adapter",
    "research_rl.trainers.grpo_adapter",
    "research_rl.trainers.rloo_adapter",
    "research_rl.trainers.distillation_adapter",
)


def load_builtin_adapters() -> None:
    for module in ADAPTER_MODULES:
        importlib.import_module(module)


def get_adapter(algorithm: str) -> TrainerAdapter:
    if algorithm not in TRAINERS:
        load_builtin_adapters()
    adapter = TRAINERS.build(algorithm)
    assert isinstance(adapter, TrainerAdapter)
    return adapter


def build_run(config: ExperimentConfig, **kwargs: Any) -> ResearchRun:
    """Build the TRL trainer + recorder for ``config`` (see ``TrainerAdapter.build`` for kwargs)."""
    return get_adapter(config.trainer.algorithm).build(config, **kwargs)


__all__ = [
    "ResearchCallback",
    "ResearchRun",
    "ResearchTrainerMixin",
    "TrainerAdapter",
    "build_run",
    "get_adapter",
    "load_builtin_adapters",
    "research_trainer_class",
]
