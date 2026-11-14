"""Policy versions: which parameters generated a rollout.

The *policy version* is the number of parameter updates that were actually applied before a
rollout was generated, i.e. :attr:`HookedOptimizer.applied_updates`. It differs from the HF
``global_step`` (which counts *nominal* optimizer steps) as soon as a step is skipped, e.g. by
``OptimSpec.skip_zero_advantage_steps``.

TRL's GRPO/RLOO trainers generate one *generation batch* and reuse it for
``steps_per_generation × num_iterations`` micro-batches. The version is stamped on the batch as a
0-dim tensor under :data:`POLICY_VERSION_KEY`; TRL's ``shuffle_sequence_dict`` and
``split_tensor_dict`` pass 0-dim tensors through unchanged, so every micro-batch carries the
version of the parameters that produced it and the *policy lag* (current version − stamp) can be
measured where the loss is computed.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import torch

from research_rl.optim.optimizer_wrapper import unwrap_hooked

POLICY_VERSION_KEY = "policy_version"


def policy_version_of(obj: Any) -> int:
    """Current policy version of a trainer or optimizer.

    * a :class:`~research_rl.optim.HookedOptimizer` (possibly inside accelerate/deepspeed wrappers): its
      ``applied_updates``;
    * a trainer with a hooked optimizer: the same, read from ``trainer.optimizer``;
    * a trainer without one: ``trainer.state.global_step`` (every nominal step is assumed to be
      applied, which is only true when nothing skips steps), and ``0`` before training starts.
    """
    hooked = unwrap_hooked(obj)
    if hooked is None:
        hooked = unwrap_hooked(getattr(obj, "optimizer", None))
    if hooked is not None:
        return int(hooked.applied_updates)
    state = getattr(obj, "state", None)
    return int(getattr(state, "global_step", 0) or 0)


def stamp_policy_version(batch: MutableMapping[str, Any], version: int) -> MutableMapping[str, Any]:
    """Store ``version`` in ``batch`` (in place) as a 0-dim ``long`` tensor; returns ``batch``.

    Insert the stamp *after* the tensors that define the batch size: TRL sizes a batch from the
    first value of the dict.
    """
    device = None
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            device = value.device
            break
    batch[POLICY_VERSION_KEY] = torch.tensor(int(version), dtype=torch.long, device=device)
    return batch


def read_policy_version(batch: Mapping[str, Any]) -> int | None:
    """The stamped version of ``batch`` (``None`` if it was never stamped)."""
    value = batch.get(POLICY_VERSION_KEY)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{POLICY_VERSION_KEY} must be a scalar, got shape {tuple(value.shape)}")
        return int(value.item())
    return int(value)


def policy_lag(current_version: int, batch: Mapping[str, Any]) -> int | None:
    """Number of applied updates between generating ``batch`` and now (``None`` if unstamped)."""
    stamped = read_policy_version(batch)
    return None if stamped is None else int(current_version) - stamped


__all__ = [
    "POLICY_VERSION_KEY",
    "policy_lag",
    "policy_version_of",
    "read_policy_version",
    "stamp_policy_version",
]
