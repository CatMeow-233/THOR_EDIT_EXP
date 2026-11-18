"""Shared contracts of the capability-gated training package.

This module defines only data types and protocols. It has no behaviour beyond validation, so
every other module of :mod:`research_rl.capability` can depend on it without import cycles.

The paradigm (see ``docs/research_rl/CAPABILITY.md``): training reward is not the measure of
learning. A block of updates is a *candidate*; it replaces the *incumbent* (the last accepted
weights) only if a paired statistical test on freshly generated held-out items shows that
transfer improved and that no retention slice regressed by more than a margin. Every score that
enters the gate comes from a strict verifier (V2) that never produces training reward.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn

from research_rl.models.ffn_channels import ChannelSelection


# ============================================================================== splits and items
class Split(str, Enum):
    """Where an item may be used. Rule-level disjointness between splits is enforced by the task
    generators (:mod:`research_rl.capability.tasks`), not by this enum."""

    TRAIN = "train"  # trained rule combinations; the only split that produces reward
    T_RULE = "t_rule"  # unseen combinations / deeper nesting of trained rules (gate: transfer)
    T_TASK = "t_task"  # held-out task families (gate: transfer)
    RETENTION = "retention"  # prior-capability slices (gate: non-inferiority)
    AUDIT = "audit"  # final report only; never reachable from the training loop or the UI API


@dataclass(frozen=True)
class Item:
    """One verifiable task instance.

    ``uid`` is unique and deterministic (same generator seed -> same uid). ``rule`` lists the
    operator names, outermost first (``("zib", "mok")`` means ``zib(mok(x))``); it is ``()`` for
    families that are not operator compositions. ``slice`` names the evaluation slice the item
    belongs to (e.g. ``"t_rule/unseen_pairs"``, ``"retention/atomic"``); the gate tests retention
    per slice.
    """

    uid: str
    split: Split
    family: str
    slice: str
    rule: tuple[str, ...]
    prompt: str
    answer: str
    meta: Mapping[str, Any] = field(default_factory=dict)


# A score function maps (model, items) to one score per item in [0, 1]; 1.0 = solved under the
# strict verifier V2. It must be deterministic for a fixed model (greedy decoding) and must not
# modify the model.
class ScoreFn(Protocol):
    def __call__(self, model: nn.Module, items: Sequence[Item]) -> list[float]: ...


# ============================================================================== gate evidence
@dataclass(frozen=True)
class PairedScores:
    """Scores of the incumbent and the candidate on the *same* items of one slice."""

    slice: str
    split: Split
    uids: tuple[str, ...]
    incumbent: tuple[float, ...]
    candidate: tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.uids)
        if len(self.incumbent) != n or len(self.candidate) != n:
            raise ValueError(
                f"slice {self.slice!r}: {n} uids but {len(self.incumbent)} incumbent and "
                f"{len(self.candidate)} candidate scores"
            )
        if len(set(self.uids)) != n:
            raise ValueError(f"slice {self.slice!r}: duplicate item uids")
        for v in (*self.incumbent, *self.candidate):
            if not (math.isfinite(v) and 0.0 <= v <= 1.0):
                raise ValueError(f"slice {self.slice!r}: score {v!r} is not in [0, 1]")

    @property
    def n(self) -> int:
        return len(self.uids)

    @property
    def is_binary(self) -> bool:
        return all(v in (0.0, 1.0) for v in (*self.incumbent, *self.candidate))

    def discordant(self) -> tuple[int, int]:
        """``(n01, n10)`` for binary scores: candidate-only wins, incumbent-only wins."""
        if not self.is_binary:
            raise ValueError(f"slice {self.slice!r}: discordant counts need binary scores")
        n01 = sum(1 for a, b in zip(self.incumbent, self.candidate) if a == 0.0 and b == 1.0)
        n10 = sum(1 for a, b in zip(self.incumbent, self.candidate) if a == 1.0 and b == 0.0)
        return n01, n10

    def mean_difference(self) -> float:
        """Mean of candidate - incumbent (positive = candidate better)."""
        if self.n == 0:
            return 0.0
        return sum(b - a for a, b in zip(self.incumbent, self.candidate)) / self.n


@dataclass(frozen=True)
class GateEvidence:
    """Everything the gate decides on for one decision.

    ``transfer`` slices are pooled for the transfer test (T_RULE and T_TASK items); every
    ``retention`` slice is tested separately (intersection-union). ``alarm`` is the tier-1
    verifier-disagreement alarm state at the time of the decision.
    """

    decision_index: int
    transfer: tuple[PairedScores, ...]
    retention: tuple[PairedScores, ...]
    alarm: bool = False

    def __post_init__(self) -> None:
        if self.decision_index < 0:
            raise ValueError("decision_index must be >= 0")
        names = [p.slice for p in self.retention]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate retention slices: {names}")


class GateDecision(str, Enum):
    """Outcome of one gate decision (the loop and operators only ever see this value)."""

    ACCEPT = "accept"  # candidate becomes the incumbent
    HOLD = "hold"  # retention passed, transfer not yet confirmed: keep training, do not promote
    SHRINK = "shrink"  # retention failed; an interpolation incumbent->candidate passed and is kept
    GROUP_ROLLBACK = "group_rollback"  # retention failed; reverting some channel groups passed
    ROLLBACK = "rollback"  # back to the incumbent (weights and optimizer state)


@dataclass(frozen=True)
class SliceTest:
    """One statistical test inside a verdict.

    For the transfer test ``point`` is the mean gain and ``bound`` its one-sided lower confidence
    bound; ``passed`` = bound > 0 (or the exact test's p-value <= alpha). For a retention slice
    ``point`` is the mean drop (incumbent - candidate) and ``bound`` its one-sided upper bound;
    ``passed`` = bound < margin.
    """

    slice: str
    n: int
    point: float
    bound: float
    passed: bool
    method: str
    alpha: float
    p_value: float | None = None
    margin: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice": self.slice,
            "n": self.n,
            "point": self.point,
            "bound": self.bound,
            "passed": self.passed,
            "method": self.method,
            "alpha": self.alpha,
            "p_value": self.p_value,
            "margin": self.margin,
        }


@dataclass(frozen=True)
class GateVerdict:
    """Result of evaluating one :class:`GateEvidence` (pure statistics, no weight changes)."""

    decision_index: int
    transfer: SliceTest
    retention: tuple[SliceTest, ...]
    alarm: bool

    @property
    def transfer_ok(self) -> bool:
        return self.transfer.passed

    @property
    def retention_ok(self) -> bool:
        return all(t.passed for t in self.retention)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_index": self.decision_index,
            "transfer": self.transfer.to_dict(),
            "retention": [t.to_dict() for t in self.retention],
            "alarm": self.alarm,
            "transfer_ok": self.transfer_ok,
            "retention_ok": self.retention_ok,
        }


@dataclass
class DecisionRecord:
    """Audit record of one full gate procedure (accept / hold / shrink / group rollback / rollback).

    ``verdicts`` holds one verdict per evaluation the procedure ran (the first on the raw
    candidate, then one per shrink factor or group revert tried), each on fresh items.
    ``reward_v1`` is the mean training reward of the block (the number a reward-only trainer
    would look at), recorded beside capability so the divergence is visible.
    """

    decision_index: int
    update_index: int
    decision: GateDecision
    verdicts: list[GateVerdict]
    reason: str
    shrink_lambda: float | None = None
    groups_reverted: list[str] = field(default_factory=list)
    reward_v1: float | None = None
    reward_v2: float | None = None
    alarm: bool = False
    override: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_index": self.decision_index,
            "update_index": self.update_index,
            "decision": self.decision.value,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "reason": self.reason,
            "shrink_lambda": self.shrink_lambda,
            "groups_reverted": list(self.groups_reverted),
            "reward_v1": self.reward_v1,
            "reward_v2": self.reward_v2,
            "alarm": self.alarm,
            "override": self.override,
        }


# ============================================================================== update operators
@dataclass
class OperatorReport:
    """What one operator update did. ``v1_pass`` / ``v2_pass`` are per-rollout verifier outcomes
    on the training rollouts of this update (same order, same length), used by the tier-1 alarm.
    ``screened`` counts rollouts excluded from the update because V1 and V2 disagreed."""

    update_index: int
    reward_v1: float
    reward_v2: float | None
    v1_pass: tuple[bool, ...]
    v2_pass: tuple[bool, ...]
    completion_lengths: tuple[int, ...] = ()
    screened: int = 0
    forward_passes: int = 0
    backward_passes: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.v1_pass) != len(self.v2_pass):
            raise ValueError("v1_pass and v2_pass must have the same length")
        if self.completion_lengths and len(self.completion_lengths) != len(self.v1_pass):
            raise ValueError("completion_lengths must match v1_pass")


@runtime_checkable
class UpdateOperator(Protocol):
    """Produces one parameter update per :meth:`step`, in place on ``model``.

    ``parameter_names()`` is the complete set of parameters the operator may change (the rollback
    scope). ``optimizer`` is the optimizer whose state must be snapshotted with the incumbent and
    restored on rollback (``None`` for forward-only operators). ``on_decision`` is told the gate's
    decision after each procedure (e.g. a forward-only operator re-bases its stored base weights
    after ROLLBACK); it never sees scores.
    """

    name: str
    optimizer: torch.optim.Optimizer | None

    def parameter_names(self) -> list[str]: ...

    def step(self, model: nn.Module, update_index: int) -> OperatorReport: ...

    def on_decision(self, decision: GateDecision) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class RollbackUnit(str, Enum):
    """Finest unit a failed retention test may revert.

    ``CHANNELS``: FFN channels are what is trained (channel masks, gain vectors, channel ES):
    shrink, channel-group rollback and full rollback are available. ``PARAMETERS``: anything else
    (LoRA adapters, full fine-tuning): only shrink and full rollback apply.
    """

    CHANNELS = "channels"
    PARAMETERS = "parameters"


@dataclass(frozen=True)
class ChannelGroup:
    """A named set of FFN channels: the unit of subset search and of group rollback."""

    name: str
    selection: ChannelSelection

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "selection": self.selection.to_dict()}


# ============================================================================== forward-only spaces
@runtime_checkable
class ParameterSpace(Protocol):
    """A flat, ``dim``-dimensional coordinate system over some model parameters.

    The space stores a *base* (a detached copy of the parameters it covers). ``write(v)`` sets the
    parameters to ``base (+) v`` computed from the stored base, never cumulatively
    (``None`` writes the base itself). ``rebase()`` makes the current parameters the new base.
    """

    dim: int

    def parameter_names(self) -> list[str]: ...

    def write(self, vector: torch.Tensor | None) -> None: ...

    def rebase(self) -> None: ...
