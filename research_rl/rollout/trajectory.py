"""Rollout bookkeeping: what was generated, how it was scored, and by which policy version.

* :class:`Trajectory` – one prompt/completion sample (unbatched, Python lists).
* :class:`RolloutBatch` – one generation batch as the trainer saw it (padded tensors on CPU),
  with per-batch statistics (mean reward, fraction of zero-advantage samples, whether *any*
  non-zero advantage exists).
* :class:`TrajectoryBuffer` – a bounded FIFO of recent :class:`RolloutBatch` objects plus running
  counters. Only the counters are checkpointed (``state_dict`` holds plain ints so it survives
  ``torch.load(weights_only=True)``); the stored tensors are a diagnostic window and are *not*
  restored on resume.

Conventions (TRL GRPO/RLOO): prompts are left-padded (``prompt_mask`` marks real tokens),
completions are right-padded, ``completion_mask`` is 1 on tokens that enter the loss.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Trajectory:
    """A single sample: prompt tokens (padding removed), completion row and its mask."""

    prompt_ids: list[int]
    completion_ids: list[int]  # the padded completion row exactly as trained on
    completion_mask: list[int]
    reward: float
    advantage: float
    policy_version: int

    @property
    def num_loss_tokens(self) -> int:
        return int(sum(self.completion_mask))


def _to_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu").clone()


@dataclass
class RolloutBatch:
    """One generation batch (the local, per-process part) together with its scores."""

    prompt_ids: torch.Tensor  # (B, P), left-padded
    prompt_mask: torch.Tensor  # (B, P)
    completion_ids: torch.Tensor  # (B, C), right-padded
    completion_mask: torch.Tensor  # (B, C)
    rewards: torch.Tensor  # (B,) weighted sum over reward functions; NaN = unscorable
    advantages: torch.Tensor  # (B,) or (B, C)
    policy_version: int
    global_step: int = 0  # HF global_step when the batch was generated
    mode: str = "train"
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.prompt_ids.shape[0]
        for name in ("prompt_mask", "completion_ids", "completion_mask", "rewards", "advantages"):
            value = getattr(self, name)
            if value.shape[0] != n:
                raise ValueError(f"RolloutBatch.{name} has {value.shape[0]} rows, expected {n}")
        if self.completion_ids.shape != self.completion_mask.shape:
            raise ValueError("completion_ids and completion_mask must have the same shape")

    @classmethod
    def from_tensors(
        cls,
        *,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        policy_version: int,
        global_step: int = 0,
        mode: str = "train",
    ) -> RolloutBatch:
        """Build a batch from (possibly GPU, possibly grad-carrying) tensors; stores detached CPU copies."""
        return cls(
            prompt_ids=_to_cpu(prompt_ids),
            prompt_mask=_to_cpu(prompt_mask),
            completion_ids=_to_cpu(completion_ids),
            completion_mask=_to_cpu(completion_mask),
            rewards=_to_cpu(rewards).float(),
            advantages=_to_cpu(advantages).float(),
            policy_version=int(policy_version),
            global_step=int(global_step),
            mode=mode,
        )

    # ------------------------------------------------------------------ statistics
    def __len__(self) -> int:
        return int(self.prompt_ids.shape[0])

    def _per_sample_zero_adv(self) -> torch.Tensor:
        adv = self.advantages
        if adv.dim() > 1:  # token-level advantages: a sample is "zero" only if every token is
            return (adv == 0).flatten(1).all(dim=1)
        return adv == 0

    @property
    def mean_reward(self) -> float:
        """Mean over scorable samples (NaN if none is scorable)."""
        valid = self.rewards[~torch.isnan(self.rewards)]
        return float(valid.mean()) if valid.numel() else math.nan

    @property
    def frac_zero_advantage(self) -> float:
        return float(self._per_sample_zero_adv().float().mean()) if len(self) else 0.0

    @property
    def has_nonzero_advantage(self) -> bool:
        return bool((self.advantages != 0).any())

    @property
    def num_loss_tokens(self) -> int:
        return int(self.completion_mask.sum())

    def stats(self) -> dict[str, float | int | bool]:
        return {
            "policy_version": self.policy_version,
            "global_step": self.global_step,
            "num_samples": len(self),
            "num_loss_tokens": self.num_loss_tokens,
            "mean_reward": self.mean_reward,
            "frac_zero_advantage": self.frac_zero_advantage,
            "has_nonzero_advantage": self.has_nonzero_advantage,
        }

    def trajectories(self) -> list[Trajectory]:
        """Unbatch into :class:`Trajectory` objects (prompt left-padding removed via ``prompt_mask``)."""
        out = []
        adv = self.advantages
        for i in range(len(self)):
            a = adv[i]
            out.append(
                Trajectory(
                    prompt_ids=self.prompt_ids[i][self.prompt_mask[i].bool()].tolist(),
                    completion_ids=self.completion_ids[i].tolist(),
                    completion_mask=self.completion_mask[i].tolist(),
                    reward=float(self.rewards[i]),
                    advantage=float(a) if a.numel() == 1 else float(a.float().mean()),
                    policy_version=self.policy_version,
                )
            )
        return out


class TrajectoryBuffer:
    """Bounded log of recent rollout batches plus running counters (the counters are checkpointed).

    Also counts optimizer *windows* (one gradient-accumulation window = one nominal optimizer step):
    how many had no non-zero advantage in any micro-batch and how many of those were skipped.
    """

    def __init__(self, capacity: int = 16) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self._batches: deque[RolloutBatch] = deque(maxlen=self.capacity)
        self.total_batches = 0
        self.total_samples = 0
        self.zero_signal_batches = 0  # generation batches whose advantages were all zero
        self.windows = 0
        self.zero_signal_windows = 0
        self.skipped_windows = 0
        self.last_policy_version: int | None = None

    # ------------------------------------------------------------------ batches
    def append(self, batch: RolloutBatch) -> None:
        self._batches.append(batch)
        self.total_batches += 1
        self.total_samples += len(batch)
        if not batch.has_nonzero_advantage:
            self.zero_signal_batches += 1
        self.last_policy_version = batch.policy_version

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self) -> Iterator[RolloutBatch]:
        return iter(list(self._batches))

    def latest(self) -> RolloutBatch | None:
        return self._batches[-1] if self._batches else None

    def policy_versions(self) -> list[int]:
        return [b.policy_version for b in self._batches]

    def stats_history(self) -> list[dict[str, float | int | bool]]:
        return [b.stats() for b in self._batches]

    def clear(self) -> None:
        self._batches.clear()

    # ------------------------------------------------------------------ optimizer windows
    def record_window(self, *, has_signal: bool, skipped: bool) -> None:
        self.windows += 1
        if not has_signal:
            self.zero_signal_windows += 1
        if skipped:
            self.skipped_windows += 1

    # ------------------------------------------------------------------ persistence
    def counters(self) -> dict[str, int | None]:
        return {
            "total_batches": self.total_batches,
            "total_samples": self.total_samples,
            "zero_signal_batches": self.zero_signal_batches,
            "windows": self.windows,
            "zero_signal_windows": self.zero_signal_windows,
            "skipped_windows": self.skipped_windows,
            "last_policy_version": self.last_policy_version,
        }

    def state_dict(self) -> dict[str, Any]:
        return {"capacity": self.capacity, **self.counters()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for key in (
            "total_batches",
            "total_samples",
            "zero_signal_batches",
            "windows",
            "zero_signal_windows",
            "skipped_windows",
        ):
            setattr(self, key, int(state.get(key, 0)))
        last = state.get("last_policy_version")
        self.last_policy_version = None if last is None else int(last)
