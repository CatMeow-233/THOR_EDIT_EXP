"""The pure gate verdict: paired statistics on one :class:`GateEvidence`, no weight changes.

:class:`CapabilityGate` turns the incumbent-vs-candidate scores of one decision into a
:class:`GateVerdict`:

* **transfer**: every transfer slice (T_RULE / T_TASK) is pooled into one sample (items are
  concatenated; uids must be unique across slices) and tested with
  :func:`~research_rl.capability.stats.paired_gain_lower_bound` (H0 "candidate not better").
* **retention**: every retention slice is tested separately with
  :func:`~research_rl.capability.stats.noninferiority_upper_bound` (H0 "drop >= margin") at its
  own margin (``GateConfig.retention_margins`` override, else ``GateConfig.retention_margin``).

Each call of :meth:`CapabilityGate.verdict` consumes exactly one alpha from the
:class:`~research_rl.capability.stats.AlphaLedger`; the transfer test and all retention tests of
that call share it (intersection-union: an accept requires every test to reject, so no split is
needed). Re-tests inside one gate procedure (after a shrink or a group rollback) are separate
calls and each consumes its own alpha.

What to *do* with a verdict (ACCEPT / HOLD / SHRINK / GROUP_ROLLBACK / ROLLBACK, the hold counter,
``require_transfer_gain``) is decided by the controller (``loop.GateController``), not here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from research_rl.capability.stats import (
    AlphaLedger,
    noninferiority_upper_bound,
    paired_gain_lower_bound,
    pool_paired,
)
from research_rl.capability.types import (
    GateEvidence,
    GateVerdict,
    PairedScores,
    RollbackUnit,
    SliceTest,
    Split,
)

__all__ = ["CapabilityGate", "GateConfig"]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_margin(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not (math.isfinite(value) and 0.0 <= value < 1.0):
        raise ValueError(f"{name} must be in [0, 1), got {value!r}")
    return float(value)


@dataclass(frozen=True)
class GateConfig:
    """Configuration of the capability gate and of the gated loop around it (JSON-able).

    Fields (validated in ``__post_init__``; ``ValueError`` on any violation):

    * ``enabled``: ``False`` gives the ungated control arm. :meth:`CapabilityGate.verdict` still
      computes every statistic **and still consumes alpha from the ledger** (so the logged
      verdicts of a gated and an ungated arm are computed identically); only the controller
      ignores the verdict (it never rolls back).
    * ``alpha`` in (0, 0.5): total significance level handed to the ledger.
    * ``alpha_schedule``: ``"spending"`` (run-level family-wise guarantee for accepts) or
      ``"per_decision"`` (guarantee per decision only); see :class:`AlphaLedger`.
    * ``retention_margin`` in [0, 1): allowed drop in absolute pass-rate points (0.02 = 2 points).
    * ``retention_margins``: per-slice overrides ``{slice_name: margin}``, each in [0, 1).
    * ``require_transfer_gain``: if True the controller accepts only with a confirmed transfer
      gain; if False retention passing (and no alarm) suffices. The verdict always reports the
      raw transfer test; this flag is applied by :meth:`CapabilityGate.accepts` / the controller.
    * ``hold_limit`` >= 1: consecutive HOLDs before a forced ROLLBACK.
    * ``shrink_lambdas``: interpolation factors tried after a retention failure, each in (0, 1);
      may be empty (no shrink).
    * ``max_group_rollbacks`` >= 0: channel groups reverted at most (``RollbackUnit.CHANNELS``).
    * ``rollback_unit``: :class:`RollbackUnit` (a string value is converted).
    * ``block_updates`` >= 1: operator updates per decision.
    * ``items_per_transfer_slice`` / ``items_per_retention_slice`` >= 1: fresh items per slice
      and evaluation. Power floor: with margin 0.02 a retention slice needs at least
      :func:`~research_rl.capability.stats.retention_power_floor_n` items to pass even with
      zero discordant pairs (173 at the first spending alpha with ``alpha=0.05``, 512 at the seventh, 576 at the
      tenth), so the default of 512 lets even a perfect slice pass only for the first seven
      decisions; long runs need larger slices, a larger margin or ``"per_decision"``.
    * ``alarm_triggers_decision``: run a decision immediately when the tier-1 alarm fires.
    """

    enabled: bool = True
    alpha: float = 0.05
    alpha_schedule: str = "spending"
    retention_margin: float = 0.02
    retention_margins: dict[str, float] = field(default_factory=dict)
    require_transfer_gain: bool = True
    hold_limit: int = 3
    shrink_lambdas: tuple[float, ...] = (0.5, 0.25)
    max_group_rollbacks: int = 4
    rollback_unit: RollbackUnit = RollbackUnit.PARAMETERS
    block_updates: int = 8
    items_per_transfer_slice: int = 256
    items_per_retention_slice: int = 512
    alarm_triggers_decision: bool = True

    def __post_init__(self) -> None:
        for name in ("enabled", "require_transfer_gain", "alarm_triggers_decision"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}")
        alpha = self.alpha
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not (0.0 < alpha < 0.5):
            raise ValueError(f"alpha must be in (0, 0.5), got {alpha!r}")
        object.__setattr__(self, "alpha", float(alpha))
        if self.alpha_schedule not in AlphaLedger.SCHEDULES:
            raise ValueError(f"alpha_schedule must be one of {AlphaLedger.SCHEDULES}, got {self.alpha_schedule!r}")
        object.__setattr__(self, "retention_margin", _check_margin("retention_margin", self.retention_margin))
        if not isinstance(self.retention_margins, Mapping):
            raise ValueError(f"retention_margins must be a mapping, got {self.retention_margins!r}")
        margins: dict[str, float] = {}
        for key, value in self.retention_margins.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"retention_margins keys must be non-empty slice names, got {key!r}")
            margins[key] = _check_margin(f"retention_margins[{key!r}]", value)
        object.__setattr__(self, "retention_margins", margins)
        if not _is_int(self.hold_limit) or self.hold_limit < 1:
            raise ValueError(f"hold_limit must be an int >= 1, got {self.hold_limit!r}")
        try:
            lambdas = tuple(self.shrink_lambdas)
        except TypeError as exc:
            raise ValueError(f"shrink_lambdas must be a sequence, got {self.shrink_lambdas!r}") from exc
        for lam in lambdas:
            if isinstance(lam, bool) or not isinstance(lam, (int, float)) or not (0.0 < lam < 1.0):
                raise ValueError(f"every shrink lambda must be in (0, 1), got {lam!r}")
        object.__setattr__(self, "shrink_lambdas", tuple(float(lam) for lam in lambdas))
        if not _is_int(self.max_group_rollbacks) or self.max_group_rollbacks < 0:
            raise ValueError(f"max_group_rollbacks must be an int >= 0, got {self.max_group_rollbacks!r}")
        try:
            object.__setattr__(self, "rollback_unit", RollbackUnit(self.rollback_unit))
        except ValueError as exc:
            raise ValueError(f"unknown rollback_unit {self.rollback_unit!r}") from exc
        for name in ("block_updates", "items_per_transfer_slice", "items_per_retention_slice"):
            value = getattr(self, name)
            if not _is_int(value) or value < 1:
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")

    def margin_for(self, slice_name: str) -> float:
        """Retention margin of ``slice_name`` (override if present, else the default margin)."""
        return self.retention_margins.get(slice_name, self.retention_margin)

    def make_ledger(self) -> AlphaLedger:
        """A fresh :class:`AlphaLedger` with this config's ``alpha`` and ``alpha_schedule``."""
        return AlphaLedger(self.alpha, self.alpha_schedule)

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-able dict (enum as its value, tuples as lists)."""
        out = asdict(self)
        out["rollback_unit"] = self.rollback_unit.value
        out["shrink_lambdas"] = list(self.shrink_lambdas)
        out["retention_margins"] = dict(self.retention_margins)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GateConfig:
        """Inverse of :meth:`to_dict`. Unknown keys raise ``ValueError``; missing keys take defaults."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown GateConfig keys: {unknown}")
        kwargs = dict(data)
        if "shrink_lambdas" in kwargs and isinstance(kwargs["shrink_lambdas"], list):
            kwargs["shrink_lambdas"] = tuple(kwargs["shrink_lambdas"])
        return cls(**kwargs)


class CapabilityGate:
    """Evaluates :class:`GateEvidence` into a :class:`GateVerdict` (statistics only).

    ``ledger`` defaults to ``config.make_ledger()``; a ledger passed explicitly must have the
    config's ``alpha`` and ``alpha_schedule`` (``ValueError`` otherwise), so the recorded config
    always describes the alphas actually used.
    """

    def __init__(self, config: GateConfig, ledger: AlphaLedger | None = None) -> None:
        if not isinstance(config, GateConfig):
            raise TypeError(f"config must be a GateConfig, got {type(config).__name__}")
        if ledger is None:
            ledger = config.make_ledger()
        if ledger.total_alpha != config.alpha or ledger.schedule != config.alpha_schedule:
            raise ValueError(
                f"ledger (total_alpha={ledger.total_alpha}, schedule={ledger.schedule!r}) does not match "
                f"config (alpha={config.alpha}, alpha_schedule={config.alpha_schedule!r})"
            )
        self.config = config
        self.ledger = ledger
        self._verdicts = 0

    @property
    def verdict_count(self) -> int:
        """Number of verdicts computed so far (each consumed one alpha)."""
        return self._verdicts

    def verdict(self, evidence: GateEvidence) -> GateVerdict:
        """Test one decision's evidence; consumes exactly one alpha from the ledger.

        Transfer slices are pooled (uids must be unique across them, else ``ValueError``); no
        transfer slice at all yields an ``n=0`` transfer test that does not pass. Each retention
        slice is tested at its margin. Every resulting :class:`SliceTest` records the alpha used.
        Validation happens before the alpha is drawn, so invalid evidence does not consume alpha.
        The verdict is computed identically whether or not ``config.enabled`` is set (see
        :class:`GateConfig`).
        """
        if not isinstance(evidence, GateEvidence):
            raise TypeError(f"evidence must be a GateEvidence, got {type(evidence).__name__}")
        if evidence.transfer:
            pooled = pool_paired(evidence.transfer)
        else:
            pooled = PairedScores(slice="transfer", split=Split.T_RULE, uids=(), incumbent=(), candidate=())
        margins = [self.config.margin_for(p.slice) for p in evidence.retention]
        alpha = self.ledger.next_alpha()
        transfer = paired_gain_lower_bound(pooled, alpha)
        retention: tuple[SliceTest, ...] = tuple(
            noninferiority_upper_bound(p, margin, alpha) for p, margin in zip(evidence.retention, margins)
        )
        self._verdicts += 1
        return GateVerdict(
            decision_index=evidence.decision_index,
            transfer=transfer,
            retention=retention,
            alarm=evidence.alarm,
        )

    def accepts(self, verdict: GateVerdict) -> bool:
        """Whether ``verdict`` licenses promoting the candidate under this config's statistics.

        True iff every retention slice passed, there is no alarm, and (if
        ``require_transfer_gain``) the transfer test passed. ``enabled`` is not consulted: an
        ungated controller ignores the answer rather than changing it.
        """
        if verdict.alarm or not verdict.retention_ok:
            return False
        return verdict.transfer_ok or not self.config.require_transfer_gain

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "ledger": self.ledger.state_dict(),
            "verdicts": self._verdicts,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore ledger position and verdict count; the stored config must equal this gate's."""
        stored = GateConfig.from_dict(state["config"])
        if stored != self.config:
            raise ValueError(f"gate config mismatch: state has {stored}, this gate has {self.config}")
        verdicts = state["verdicts"]
        if not _is_int(verdicts) or verdicts < 0:
            raise ValueError(f"invalid verdict count {verdicts!r}")
        self.ledger.load_state_dict(state["ledger"])
        self._verdicts = verdicts
