"""Capability curves over training (``docs/research_rl/CAPABILITY.md`` 1.8).

:class:`CapabilityTracker` measures, at a training step, the model's current weights on fresh
items of every non-AUDIT slice and returns a :class:`CapabilityPoint`:

* ``A`` (acquisition): mean pass rate on the TRAIN slices (fresh items of trained rules);
* ``R``: pass rate per RETENTION slice;
* ``T_task`` / ``T_rule``: unweighted mean over the T_TASK / T_RULE slices of their pass rates;
* ``D(t) = [reward(t) - reward(0)] - [T_rule(t) - T_rule(0)]`` - the reward-vs-capability
  divergence ("reward hallucination" when the training reward rises while T_rule does not).

Baselines. ``T_rule(0)`` is taken from the :class:`~research_rl.capability.probe.StartMap` of the
start checkpoint, aggregated exactly as ``T_rule(t)`` (unweighted mean of the per-slice greedy
pass rates of the T_RULE slices; the start map's item count does not enter). ``reward(0)`` is the
**first non-None ``reward_v1`` passed to** :meth:`CapabilityTracker.measure` (the start map holds
no reward); ``D`` is ``None`` until one has been seen, and also for every later point measured
with ``reward_v1=None``. The two baselines can therefore refer to
different times (the start checkpoint vs the first measured block's mean reward), and ``T_rule(0)``
was measured on other items than ``T_rule(t)``, so ``D(0)`` is not exactly 0 but noise.

Items: :func:`~research_rl.capability.probe.draw_slice` with ``purpose="track"``,
``index=step`` and ``pool=TRACK_POOL`` - a stream separate from the gate's (``"gate/<attempt>"``)
and the start map's (``"start"``). Measuring the same step twice reuses the same items. Tracker
items **never coincide with gate items**: every uid belongs to exactly one pool
(:func:`~research_rl.capability.probe.item_pool`), the gate draws only from ``GATE_POOL``, the
tracker and the start map only from ``TRACK_POOL`` (1 in
:data:`~research_rl.capability.probe.TRACK_POOL_DENOMINATOR` uids). Tracker items of different
steps are independent draws from the track pool and repeat by chance (small length-3 domains).
Items drawn with ``pool=None`` elsewhere (e.g. a subset search) are not partitioned.

Truncation. When ``score_fn`` exposes an integer ``max_new_tokens``, it must equal the start
map's ``max_new_tokens`` (``ValueError`` otherwise), so ``T_rule(0)`` and ``T_rule(t)`` are scored
under the same truncation budget; the verifier itself is not recorded in the start map and is not
compared.

The tracker never touches parameters or optimizer state; every submodule's training mode is
restored.
CPU-verified with tiny models only; GPU behaviour is NOT_TESTED.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from torch import nn

from research_rl.capability.probe import (
    TRACK_POOL,
    SliceScores,
    SliceStart,
    StartMap,
    _budget,
    check_truncation_budget,
    draw_slice,
    score_items,
)
from research_rl.capability.tasks import ItemSampler
from research_rl.capability.types import Item, ScoreFn, Split

__all__ = ["TRACK_PURPOSE", "CapabilityPoint", "CapabilityTracker", "SliceStart", "StartMap"]

TRACK_PURPOSE = "track"
"""``ItemSampler`` purpose of every tracker draw (index = training step)."""


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _finite_or_none(name: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number or None, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class CapabilityPoint:
    """One capability measurement (see the module docstring for the definitions).

    ``slices`` holds every measured slice's pass rate and ``n`` its item count; ``reward0`` /
    ``t_rule0`` are the baselines ``D`` was computed against (``reward0`` is ``None`` and ``D`` is
    ``None`` until a reward has been seen; ``D`` is also ``None`` for a point measured with
    ``reward_v1=None``).
    """

    step: int
    reward_v1: float | None
    acquisition: float
    retention: Mapping[str, float]
    t_task: float
    t_rule: float
    divergence: float | None
    reward0: float | None
    t_rule0: float
    slices: Mapping[str, float] = field(default_factory=dict)
    n: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able with the paper's names: ``A``, ``R``, ``T_task``, ``T_rule``, ``D``."""
        return {
            "step": self.step,
            "reward_v1": self.reward_v1,
            "A": self.acquisition,
            "R": dict(self.retention),
            "T_task": self.t_task,
            "T_rule": self.t_rule,
            "D": self.divergence,
            "reward0": self.reward0,
            "T_rule0": self.t_rule0,
            "slices": dict(self.slices),
            "n": dict(self.n),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CapabilityPoint:
        """Inverse of :meth:`to_dict`."""
        return cls(
            step=int(data["step"]),
            reward_v1=_finite_or_none("reward_v1", data["reward_v1"]),
            acquisition=float(data["A"]),
            retention={str(k): float(v) for k, v in data["R"].items()},
            t_task=float(data["T_task"]),
            t_rule=float(data["T_rule"]),
            divergence=_finite_or_none("D", data["D"]),
            reward0=_finite_or_none("reward0", data["reward0"]),
            t_rule0=float(data["T_rule0"]),
            slices={str(k): float(v) for k, v in data.get("slices", {}).items()},
            n={str(k): int(v) for k, v in data.get("n", {}).items()},
        )


class CapabilityTracker:
    """Measures :class:`CapabilityPoint` s of a training run.

    ``start`` must come from the same splits (``splits_fingerprint``) and contain every T_RULE
    slice of ``sampler.splits`` (``ValueError`` otherwise). Every measurement scores
    ``items_per_slice`` fresh items of each non-AUDIT slice (TRAIN, T_RULE, T_TASK, RETENTION, in
    :meth:`~research_rl.capability.tasks.CompositionSplits.slices` order) with ``score_fn`` in one
    call. The truncation budget of ``score_fn`` is checked at construction, and a ``score_fn``
    with an integer ``max_new_tokens`` must match ``start.max_new_tokens`` (``ValueError``).
    """

    def __init__(
        self, sampler: ItemSampler, score_fn: ScoreFn, start: StartMap, *, items_per_slice: int
    ) -> None:
        if not isinstance(sampler, ItemSampler):
            raise TypeError(f"sampler must be an ItemSampler, got {type(sampler).__name__}")
        if not isinstance(start, StartMap):
            raise TypeError(f"start must be a StartMap, got {type(start).__name__}")
        if isinstance(items_per_slice, bool) or not isinstance(items_per_slice, int) or items_per_slice < 1:
            raise ValueError(f"items_per_slice must be an int >= 1, got {items_per_slice!r}")
        if start.splits_fingerprint != sampler.splits.fingerprint():
            raise ValueError("the start map was measured on different splits than the sampler's")
        rule_slices = sampler.splits.slices_of(Split.T_RULE)
        missing = [s for s in rule_slices if s not in start.slices]
        if missing:
            raise ValueError(f"the start map lacks the T_RULE slice(s) {missing} needed for T_rule(0)")
        check_truncation_budget(sampler, score_fn)
        budget = _budget(score_fn)
        if budget is not None and budget != start.max_new_tokens:
            raise ValueError(
                f"score_fn.max_new_tokens={budget} differs from the start map's "
                f"max_new_tokens={start.max_new_tokens}: T_rule(0) and T_rule(t) would be scored "
                "under different truncation budgets"
            )
        self.sampler = sampler
        self.score_fn = score_fn
        self.start = start
        self.items_per_slice = int(items_per_slice)
        self.slices: dict[str, Split] = {
            name: split for name, split in sampler.splits.slices().items() if split is not Split.AUDIT
        }
        self.t_rule0 = _mean([start.slices[s].pass1 for s in rule_slices])
        self.reward0: float | None = None
        self.history: list[CapabilityPoint] = []

    def items(self, step: int) -> dict[str, list[Item]]:
        """The items of the measurement at ``step`` (``purpose="track"``, ``index=step``,
        ``pool=TRACK_POOL``), per slice."""
        return {
            name: draw_slice(
                self.sampler, name, self.items_per_slice, purpose=TRACK_PURPOSE, index=step, pool=TRACK_POOL
            )
            for name in self.slices
        }

    def score(self, model: nn.Module, step: int) -> dict[str, SliceScores]:
        """Per-slice :class:`SliceScores` of the model's current weights at ``step`` (one
        ``score_fn`` call over all items; no baseline or history update)."""
        per_slice = self.items(step)
        flat = [it for group in per_slice.values() for it in group]
        scores = score_items(self.score_fn, model, flat)
        out: dict[str, SliceScores] = {}
        pos = 0
        for name, group in per_slice.items():
            out[name] = SliceScores(
                slice=name,
                split=self.slices[name],
                purpose=TRACK_PURPOSE,
                index=step,
                items=tuple(group),
                scores=tuple(scores[pos : pos + len(group)]),
                pool=TRACK_POOL,
            )
            pos += len(group)
        return out

    def measure(self, model: nn.Module, step: int, reward_v1: float | None) -> CapabilityPoint:
        """Measure the current weights at training ``step`` (int >= 0) and append the point to
        :attr:`history`.

        ``reward_v1`` is the mean training reward (V1) of the block that produced the weights, or
        ``None`` (e.g. before any update). The first non-None reward becomes ``reward(0)``; ``D``
        is ``None`` until then and whenever ``reward_v1`` is ``None`` (the baseline is kept).
        Non-finite rewards raise ``ValueError``.
        """
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"step must be an int >= 0, got {step!r}")
        reward = _finite_or_none("reward_v1", reward_v1)
        scored = self.score(model, step)
        rates = {name: s.mean for name, s in scored.items()}

        def split_mean(split: Split) -> float:
            return _mean([rates[name] for name, sp in self.slices.items() if sp is split])

        t_rule = split_mean(Split.T_RULE)
        if reward is not None and self.reward0 is None:
            self.reward0 = reward
        divergence = None
        if reward is not None and self.reward0 is not None:
            divergence = (reward - self.reward0) - (t_rule - self.t_rule0)
        point = CapabilityPoint(
            step=step,
            reward_v1=reward,
            acquisition=split_mean(Split.TRAIN),
            retention={name: rates[name] for name, sp in self.slices.items() if sp is Split.RETENTION},
            t_task=split_mean(Split.T_TASK),
            t_rule=t_rule,
            divergence=divergence,
            reward0=self.reward0,
            t_rule0=self.t_rule0,
            slices=rates,
            n={name: s.n for name, s in scored.items()},
        )
        self.history.append(point)
        return point

    def to_dict(self) -> dict[str, Any]:
        """JSON-able: configuration, start map, baselines and every measured point."""
        return {
            "items_per_slice": self.items_per_slice,
            "slices": {name: split.value for name, split in self.slices.items()},
            "sampler": self.sampler.state_dict(),
            "start": self.start.to_dict(),
            "t_rule0": self.t_rule0,
            "reward0": self.reward0,
            "points": [p.to_dict() for p in self.history],
        }

    def state_dict(self) -> dict[str, Any]:
        """Resume state: ``reward0``, the history, and identity fields checked on load."""
        return {
            "items_per_slice": self.items_per_slice,
            "sampler": self.sampler.state_dict(),
            "start": self.start.to_dict(),
            "reward0": self.reward0,
            "points": [p.to_dict() for p in self.history],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore ``reward0`` and the history; atomic. ``ValueError`` if the state came from a
        tracker with another sampler, start map or ``items_per_slice``."""
        if int(state["items_per_slice"]) != self.items_per_slice:
            raise ValueError(
                f"items_per_slice mismatch: state {state['items_per_slice']}, tracker {self.items_per_slice}"
            )
        self.sampler.load_state_dict(state["sampler"])
        if StartMap.from_dict(state["start"]) != self.start:
            raise ValueError("the state was recorded against a different start map")
        reward0 = _finite_or_none("reward0", state["reward0"])
        points = [CapabilityPoint.from_dict(p) for p in state["points"]]
        self.reward0 = reward0
        self.history = points
