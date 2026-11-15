"""Rollout bookkeeping: what was generated, how it was scored, and by which policy version.

* :class:`Trajectory` – one prompt/completion sample (unbatched, Python lists).
* :class:`RolloutBatch` – one generation batch as the trainer saw it (padded tensors on CPU),
  with per-batch statistics (mean reward, fraction of samples without learning signal, whether
  *any* sample carries one). A sample's signal is the trainer-supplied ``signal`` flag when given
  (the RL trainers derive it from the rewards, robust to float32 round-off in the advantages,
  see :mod:`research_rl.trainers.rl_common`), otherwise "advantage exactly non-zero".
* :class:`TrajectoryBuffer` – a bounded FIFO of recent :class:`RolloutBatch` objects plus running
  counters. Only the counters are checkpointed (``state_dict`` holds plain ints so it survives
  ``torch.load(weights_only=True)``); the stored tensors are a diagnostic window and are *not*
  restored on resume.

Experts (TLC, :mod:`research_rl.models.expert_manager`): when the RL trainer routes rows through
response-level experts, :attr:`RolloutBatch.expert_ids` names the expert that produced each row,
and the buffer counts samples per expert (``expert_samples``, a ``{name: int}`` dict, present in
``counters()`` / ``state_dict()`` only once an expert-tagged batch was appended; checkpoints
without it load as "no expert samples").

Conventions (TRL GRPO/RLOO): prompts are left-padded (``prompt_mask`` marks real tokens),
completions are right-padded, ``completion_mask`` is 1 on tokens that enter the loss.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
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
    expert: str | None = None  # expert that produced the sample (None = no experts)

    @property
    def num_loss_tokens(self) -> int:
        return int(sum(self.completion_mask))


def _to_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu").clone()


def _expert_id_tuple(expert_ids: Any) -> tuple[str, ...]:
    """``expert_ids`` as a tuple of names; a bare ``str`` or any non-``str`` entry is a ``TypeError``.

    No coercion: ``"ab"`` would otherwise become ``("a", "b")`` and ``None`` / ints the names
    ``"None"`` / ``"1"``, which the buffer's per-expert counters would then record and checkpoint.
    """
    if isinstance(expert_ids, (str, bytes)) or not isinstance(expert_ids, Sequence):
        raise TypeError(
            f"RolloutBatch.expert_ids must be a sequence of expert names (one per row), got "
            f"{type(expert_ids).__name__} {expert_ids!r}"
        )
    bad = [(i, e) for i, e in enumerate(expert_ids) if not isinstance(e, str)]
    if bad:
        i, e = bad[0]
        raise TypeError(
            f"RolloutBatch.expert_ids[{i}] must be a str expert name, got {type(e).__name__} {e!r}"
        )
    return tuple(expert_ids)


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
    signal: torch.Tensor | None = None  # (B,) bool: sample carries a learning signal (None = derive)
    expert_ids: tuple[str, ...] | None = None  # expert that produced each row (None = no experts)

    def __post_init__(self) -> None:
        n = self.prompt_ids.shape[0]
        for name in ("prompt_mask", "completion_ids", "completion_mask", "rewards", "advantages"):
            value = getattr(self, name)
            if value.shape[0] != n:
                raise ValueError(f"RolloutBatch.{name} has {value.shape[0]} rows, expected {n}")
        if self.completion_ids.shape != self.completion_mask.shape:
            raise ValueError("completion_ids and completion_mask must have the same shape")
        if self.signal is not None and tuple(self.signal.shape) != (n,):
            raise ValueError(f"RolloutBatch.signal must have shape ({n},), got {tuple(self.signal.shape)}")
        if self.expert_ids is not None:
            self.expert_ids = _expert_id_tuple(self.expert_ids)
            if len(self.expert_ids) != n:
                raise ValueError(f"RolloutBatch.expert_ids has {len(self.expert_ids)} entries, expected {n}")

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
        signal: torch.Tensor | None = None,
        expert_ids: Sequence[str] | None = None,
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
            signal=None if signal is None else _to_cpu(signal).bool(),
            expert_ids=None if expert_ids is None else _expert_id_tuple(expert_ids),
        )

    # ------------------------------------------------------------------ statistics
    def __len__(self) -> int:
        return int(self.prompt_ids.shape[0])

    def _per_sample_zero_adv(self) -> torch.Tensor:
        """Per sample: no learning signal (``~signal`` if given, else every advantage exactly 0)."""
        if self.signal is not None:
            return ~self.signal
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
        """Fraction of samples without learning signal (see :meth:`_per_sample_zero_adv`)."""
        return float(self._per_sample_zero_adv().float().mean()) if len(self) else 0.0

    @property
    def has_nonzero_advantage(self) -> bool:
        """Does any sample carry a learning signal?"""
        return bool((~self._per_sample_zero_adv()).any())

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
                    expert=None if self.expert_ids is None else self.expert_ids[i],
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
        self.expert_samples: dict[str, int] = {}  # samples per expert (expert-tagged batches only)

    # ------------------------------------------------------------------ batches
    def append(self, batch: RolloutBatch) -> None:
        self._batches.append(batch)
        self.total_batches += 1
        self.total_samples += len(batch)
        if not batch.has_nonzero_advantage:
            self.zero_signal_batches += 1
        self.last_policy_version = batch.policy_version
        if batch.expert_ids is not None:
            for name in batch.expert_ids:
                self.expert_samples[name] = self.expert_samples.get(name, 0) + 1

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
    def counters(self) -> dict[str, Any]:
        """Running counters (JSON- and ``weights_only``-safe: ints, ``None`` and a ``{str: int}`` dict).

        ``expert_samples`` is included only once an expert-tagged batch has been appended.
        """
        out: dict[str, Any] = {
            "total_batches": self.total_batches,
            "total_samples": self.total_samples,
            "zero_signal_batches": self.zero_signal_batches,
            "windows": self.windows,
            "zero_signal_windows": self.zero_signal_windows,
            "skipped_windows": self.skipped_windows,
            "last_policy_version": self.last_policy_version,
        }
        if self.expert_samples:
            out["expert_samples"] = dict(self.expert_samples)
        return out

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
        # Older checkpoints have no per-expert counts.
        self.expert_samples = {str(k): int(v) for k, v in (state.get("expert_samples") or {}).items()}
