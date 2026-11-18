"""The gated training loop (``docs/research_rl/CAPABILITY.md`` 1.9 with the pinned semantics of 1.16).

* :class:`GateController` runs **one gate procedure** on the current weights (the *candidate*)
  against the *incumbent* (the last accepted :class:`~research_rl.capability.weights.StateSnapshot`)
  and applies the outcome to the model and the optimizer.
* :class:`GatedLoop` drives an :class:`~research_rl.capability.types.UpdateOperator` (GRPO, ES,
  ...), feeds every :class:`~research_rl.capability.types.OperatorReport` to the tier-1
  :class:`~research_rl.capability.alarm.DisagreementAlarm`, runs a decision every
  ``block_updates`` updates (and immediately on an alarm when ``alarm_triggers_decision``), writes
  a JSONL decision log and checkpoints everything needed for an exact resume.

The procedure (gate enabled)
----------------------------
Every evaluation ("verdict") draws **fresh** items: evaluation ``attempt`` of decision ``d`` uses
``probe.evidence(model, incumbent, d, attempt=attempt, alarm=alarm)`` (``attempt`` 0 is the raw
candidate, then 1, 2, ... for each re-test) and consumes one alpha from the ledger (each re-test is
a separate verdict). One decision therefore uses at most ``1 + len(shrink_lambdas) +
max_group_rollbacks`` verdicts (the group stage only for ``RollbackUnit.CHANNELS`` with groups).

"Transfer passes" below means ``verdict.transfer_ok or not config.require_transfer_gain`` (with
``require_transfer_gain=False`` the transfer test is reported but not required, also for the alarm
rules). The first verdict is classified into exactly one reason (:func:`classify_verdict`):

=====================  ======================================================================
reason                 condition on the first verdict -> what the controller does
=====================  ======================================================================
``horizon_exhausted``  retention failed and some failing retention slice could not pass at this
                       verdict's alpha **even with zero discordant pairs** (its size is below the
                       power floor, :func:`retention_unpassable`) -> ROLLBACK; no re-test is run
                       (later alphas are never larger). Takes precedence over every other reason.
``confirmed_alarm``    alarm and (retention failed or transfer did not pass) -> ROLLBACK directly.
``alarm_hold``         alarm, retention and transfer passed -> HOLD (never promoted).
``accept``             no alarm, retention and transfer passed -> ACCEPT (candidate promoted).
``no_transfer_gain``   no alarm, retention passed, transfer did not -> HOLD.
``retention_failed``   no alarm, retention failed -> the remedy stage below.
=====================  ======================================================================

Hold streak (``hold_limit``, kept by the controller and checkpointed): the number of consecutive
decisions that kept unpromoted weights (HOLD for ``no_transfer_gain`` / ``alarm_hold``, and an
unpromoted SHRINK / GROUP_ROLLBACK, see below), capped at ``hold_limit``. ``hold_limit`` HOLDs are
allowed in a row; the next would-be HOLD becomes a ROLLBACK with reason ``hold_limit`` instead
(so ``hold_limit=1`` allows one HOLD, then rolls back). An unpromoted SHRINK / GROUP_ROLLBACK is
never converted (the pinned rule keeps weights that passed retention); it only extends the
streak, so a HOLD right after it may be forced. ACCEPT, a promoted SHRINK / GROUP_ROLLBACK and
every ROLLBACK reset the streak to 0.

Remedy stage (only after ``retention_failed``; the candidate is captured first, with optimizer
state and masters):

1. **SHRINK**, for each ``lam`` in ``shrink_lambdas`` in order:
   :func:`~research_rl.capability.weights.interpolate` writes ``incumbent + lam (candidate -
   incumbent)`` (always from the two endpoints, never cumulatively; masters of a
   ``HookedOptimizer`` updated consistently) and the next attempt is evaluated. It *passes* when
   retention passes. The shrunk weights then stay as the current weights; the incumbent is
   promoted to them only if transfer also passes and there is no alarm (reason
   ``"shrink_passed;promoted=true"`` / ``"...;promoted=false"``, ``shrink_lambda`` set). **Optimizer
   state is kept from the candidate** (Adam moments now sit on interpolated weights; this is the
   pinned behaviour, not an oversight).
2. **GROUP_ROLLBACK** (``RollbackUnit.CHANNELS`` with non-empty ``channel_groups`` and
   ``max_group_rollbacks > 0``): the raw candidate's parameters (and masters, not moments) are
   written back, then the groups are ordered by
   :func:`~research_rl.capability.weights.channel_groups_by_damage` with activation scores = the
   per-channel mean ``|h|`` of :func:`~research_rl.capability.scores.collect_channel_stats` on the
   texts ``damage_texts_fn(items)`` (default: the prompts) of the first verdict's **failing**
   retention slices (their attempt-0 items), computed on the raw candidate weights. Groups whose
   damage is exactly 0 (no displacement or no activation) are skipped: reverting them cannot
   change the failing slice and would only spend alpha. The first ``max_group_rollbacks`` of the
   rest are reverted **cumulatively** with :func:`~research_rl.capability.weights.revert_groups`,
   one more group before each re-test; the pass / promotion rule is that of SHRINK (reason
   ``"group_rollback_passed;promoted=..."``, ``groups_reverted`` = every reverted group, in order).
   Whether every group can be reverted to the incumbent (e.g. a group on a layer outside
   ``operator.parameter_names()``) is checked **before the first verdict** of every gated
   decision (:meth:`GateController.check_group_scope`), so such a setup fails without spending
   alpha or writing anything.
3. Nothing passed -> ROLLBACK (reason ``retention_failed``). A re-test whose retention is
   unpassable at its alpha (see above) stops the stage at once: ROLLBACK with reason
   ``horizon_exhausted``.

ROLLBACK is ``weights.restore(model, incumbent, optimizer=operator.optimizer)`` (parameters,
optimizer state and ``HookedOptimizer`` masters bit-identical to the incumbent; parameters only
when the operator has no optimizer). After every procedure the operator is told the decision with
``operator.on_decision(decision)`` (after the weights were written; it never sees scores). A
promotion captures the new incumbent with :func:`~research_rl.capability.weights.capture` (same
parameter names, the operator's optimizer, the same channel-state layers as the old incumbent).

Exceptions during the procedure: the first verdict writes nothing. If anything raises during the
remedy stage's trial writes and re-tests (a re-test OOM, ``KeyboardInterrupt``, a scorer error),
the raw candidate's parameters and masters are written back with ``weights.restore(...,
restore_optimizer=False)`` (the stage never touches the moments, so the optimizer state is the
candidate's too) before the exception propagates: the model is left at the pre-decision
candidate, never at interpolated or partly reverted weights. A failure while the chosen outcome
is applied (``_passed`` / ``_rollback``: the promotion's capture, ``operator.on_decision``) keeps
that outcome's weights. The verdicts already computed have spent their alpha; no record is returned and the hold streak is unchanged. ``GatedLoop`` keeps
the decision pending (:attr:`GatedLoop.decision_pending`) and takes it at its next ``run``.

Gate disabled (the ungated control arm, ``GateConfig.enabled=False``)
---------------------------------------------------------------------
Only the first verdict is computed (same fresh items as the gated arm would use) and classified as
above; the applied decision is always ACCEPT (the candidate becomes the incumbent, nothing is ever
rolled back, the hold streak stays 0), recorded with ``override="gate_disabled"`` and
``reason="would=<reason>"`` where ``<reason>`` is the first-verdict reason from the closed set
:data:`WOULD_REASONS` (``accept``, ``no_transfer_gain``, ``alarm_hold``, ``confirmed_alarm``,
``retention_failed``, ``horizon_exhausted``). ``would=accept`` is the gate's accept; every other
token is a non-accept, and ``horizon_exhausted`` means the verdict carried no retention evidence.
The disabled arm draws its alphas from a **private ledger** owned by the controller (a copy of the
passed gate's ledger position taken at construction): the passed gate and its ledger are never
used or advanced, so a ledger shared with a gated arm is not changed. :attr:`GateController.verdict_gate`
is the gate that actually computes the verdicts in either mode.

Limits: every statement is verified on CPU with tiny models only (``tests/research_rl/
test_capability_loop.py``); nothing here has run on a GPU (NOT_TESTED).
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import os
import warnings
import weakref
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeGuard

import torch
from torch import nn

from research_rl.capability.alarm import AlarmState, DisagreementAlarm
from research_rl.capability.gate import CapabilityGate, GateConfig
from research_rl.capability.probe import CapabilityProbe
from research_rl.capability.scores import collect_channel_stats
from research_rl.capability.stats import AlphaLedger, normal_cdf, tango_z
from research_rl.capability.types import (
    ChannelGroup,
    DecisionRecord,
    GateDecision,
    GateVerdict,
    Item,
    OperatorReport,
    RollbackUnit,
    UpdateOperator,
)
from research_rl.capability.weights import (
    StateSnapshot,
    _check_incumbent,
    _check_reference,
    _reference_state,
    capture,
    channel_groups_by_damage,
    group_damage,
    interpolate,
    restore,
    revert_groups,
)

__all__ = [
    "OVERRIDE_GATE_DISABLED",
    "REASON_ACCEPT",
    "REASON_ALARM_HOLD",
    "REASON_CONFIRMED_ALARM",
    "REASON_HOLD_LIMIT",
    "REASON_HORIZON_EXHAUSTED",
    "REASON_NO_TRANSFER_GAIN",
    "REASON_RETENTION_FAILED",
    "WOULD_PREFIX",
    "WOULD_REASONS",
    "GateController",
    "GatedLoop",
    "MetricsRecorder",
    "classify_verdict",
    "retention_unpassable",
    "snapshot_from_state",
    "snapshot_to_state",
    "worst_case_verdicts",
]

REASON_ACCEPT = "accept"
REASON_NO_TRANSFER_GAIN = "no_transfer_gain"
REASON_ALARM_HOLD = "alarm_hold"
REASON_HOLD_LIMIT = "hold_limit"
REASON_CONFIRMED_ALARM = "confirmed_alarm"
REASON_RETENTION_FAILED = "retention_failed"
REASON_HORIZON_EXHAUSTED = "horizon_exhausted"
OVERRIDE_GATE_DISABLED = "gate_disabled"
WOULD_PREFIX = "would="
WOULD_REASONS = (
    REASON_ACCEPT,
    REASON_NO_TRANSFER_GAIN,
    REASON_ALARM_HOLD,
    REASON_CONFIRMED_ALARM,
    REASON_RETENTION_FAILED,
    REASON_HORIZON_EXHAUSTED,
)
"""The closed set of first-verdict reasons recorded as ``"would=<reason>"`` by the disabled arm."""

_CONTROLLER_STATE_VERSION = 1
_LOOP_STATE_VERSION = 1


def _is_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_index(name: str, value: Any) -> int:
    if not _is_int(value) or value < 0:
        raise ValueError(f"{name} must be an int >= 0, got {value!r}")
    return int(value)


def _promoted_reason(stage: str, promoted: bool) -> str:
    return f"{stage}_passed;promoted={'true' if promoted else 'false'}"


# ============================================================================== verdict rules
def retention_unpassable(verdict: GateVerdict) -> bool:
    """True iff some **failing** retention slice of ``verdict`` could not have passed at the
    verdict's alpha even with zero discordant pairs.

    For each failing :class:`~research_rl.capability.types.SliceTest` with ``n`` items, margin ``m``
    and alpha ``a`` the best possible evidence (every pair tied) has Tango p-value
    ``Phi(tango_z(0, 0, n, m))`` (the statistic of
    :func:`~research_rl.capability.stats.noninferiority_upper_bound`, binary and fractional alike);
    the slice is unpassable iff that p-value exceeds ``a`` (always for ``m == 0`` or ``n == 0``).
    Passing slices are never "unpassable" (candidate-only wins can pass a slice whose all-tied
    version would not). Pure arithmetic on the recorded tests.
    """
    for test in verdict.retention:
        if test.passed:
            continue
        margin = test.margin if test.margin is not None else 0.0
        if test.n < 1 or margin <= 0.0:
            return True
        if normal_cdf(tango_z(0.0, 0.0, test.n, margin)) > test.alpha:
            return True
    return False


def classify_verdict(verdict: GateVerdict, *, require_transfer_gain: bool) -> str:
    """The first-verdict reason of the module docstring's table (one of :data:`WOULD_REASONS`).

    Precedence: ``horizon_exhausted`` > alarm rules > retention > transfer. "Transfer passes" is
    ``verdict.transfer_ok or not require_transfer_gain``.
    """
    if not verdict.retention_ok and retention_unpassable(verdict):
        return REASON_HORIZON_EXHAUSTED
    transfer = verdict.transfer_ok or not require_transfer_gain
    if verdict.alarm:
        return REASON_ALARM_HOLD if (verdict.retention_ok and transfer) else REASON_CONFIRMED_ALARM
    if not verdict.retention_ok:
        return REASON_RETENTION_FAILED
    return REASON_ACCEPT if transfer else REASON_NO_TRANSFER_GAIN


def _default_damage_texts(items: Sequence[Item]) -> list[str]:
    return [it.prompt for it in items]


# ============================================================================== the controller
class GateController:
    """One gate procedure per :meth:`run` (see the module docstring for the exact semantics).

    Args:
        gate: the :class:`~research_rl.capability.gate.CapabilityGate`; its ``config`` drives the
            procedure (``enabled``, ``require_transfer_gain``, ``hold_limit``, ``shrink_lambdas``,
            ``max_group_rollbacks``). With ``config.enabled=False`` its ledger is never used (a
            private copy is, see :attr:`verdict_gate`).
        probe: a :class:`~research_rl.capability.probe.CapabilityProbe` (``evidence`` for every
            verdict, ``gate_items`` for the damage texts of the group stage).
        rollback_unit: the finest revert unit. **Authoritative** for the controller (it is used
            even when ``gate.config.rollback_unit`` differs). ``channel_groups`` require
            ``RollbackUnit.CHANNELS`` (``ValueError`` otherwise).
        channel_groups: candidate groups for the group-rollback stage (unique names); ``None`` or
            empty disables the stage.
        damage_texts_fn: maps the failing slices' items to the texts whose activations rank the
            groups (default: ``[item.prompt for item in items]``).
        tokenizer: tokenizer for :func:`~research_rl.capability.scores.collect_channel_stats`
            (default: ``probe.score_fn.tokenizer``). Required (``ValueError`` at construction)
            when the group stage is enabled. Extension beyond the 1.9 signature.
        damage_batch_size: batch size of that activation pass (>= 1).
    """

    def __init__(
        self,
        gate: CapabilityGate,
        probe: CapabilityProbe,
        *,
        rollback_unit: RollbackUnit,
        channel_groups: Sequence[ChannelGroup] | None = None,
        damage_texts_fn: Callable[[Sequence[Item]], Sequence[str]] | None = None,
        tokenizer: Any | None = None,
        damage_batch_size: int = 8,
    ) -> None:
        if not isinstance(gate, CapabilityGate):
            raise TypeError(f"gate must be a CapabilityGate, got {type(gate).__name__}")
        if not callable(getattr(probe, "evidence", None)):
            raise TypeError(f"probe must provide evidence(), got {type(probe).__name__}")
        try:
            unit = RollbackUnit(rollback_unit)
        except ValueError as exc:
            raise ValueError(f"unknown rollback_unit {rollback_unit!r}") from exc
        groups = tuple(channel_groups or ())
        for g in groups:
            if not isinstance(g, ChannelGroup):
                raise TypeError(f"channel_groups entries must be ChannelGroup, got {type(g).__name__}")
        names = [g.name for g in groups]
        if len(set(names)) != len(names):
            raise ValueError(
                f"duplicate channel group names: {sorted({n for n in names if names.count(n) > 1})}"
            )
        if groups and unit is not RollbackUnit.CHANNELS:
            raise ValueError("channel_groups require rollback_unit=RollbackUnit.CHANNELS")
        if not _is_int(damage_batch_size) or damage_batch_size < 1:
            raise ValueError(f"damage_batch_size must be an int >= 1, got {damage_batch_size!r}")
        self.gate = gate
        self.probe = probe
        self.rollback_unit = unit
        self.channel_groups: tuple[ChannelGroup, ...] = groups
        self.damage_texts_fn: Callable[[Sequence[Item]], Sequence[str]] = (
            damage_texts_fn if damage_texts_fn is not None else _default_damage_texts
        )
        if tokenizer is None:
            tokenizer = getattr(getattr(probe, "score_fn", None), "tokenizer", None)
        self.tokenizer = tokenizer
        self.damage_batch_size = damage_batch_size
        if self.group_stage_enabled:
            if self.tokenizer is None:
                raise ValueError(
                    "the group-rollback stage needs a tokenizer (pass tokenizer= or use a probe whose "
                    "score_fn has a .tokenizer)"
                )
            if not callable(getattr(probe, "gate_items", None)):
                raise TypeError("the group-rollback stage needs probe.gate_items()")
        if gate.config.enabled:
            self._verdict_gate = gate
        else:
            ledger = AlphaLedger(gate.ledger.total_alpha, gate.ledger.schedule)
            ledger.load_state_dict(gate.ledger.state_dict())
            self._verdict_gate = CapabilityGate(gate.config, ledger)
        self.holds = 0
        self._scope_checked: tuple[weakref.ref[nn.Module], weakref.ref[StateSnapshot]] | None = None

    # ---------------------------------------------------------------- properties
    @property
    def config(self) -> GateConfig:
        return self.gate.config

    @property
    def verdict_gate(self) -> CapabilityGate:
        """The gate that computes (and spends alpha for) every verdict: ``gate`` itself when
        enabled, a private copy with its own ledger when disabled."""
        return self._verdict_gate

    @property
    def group_stage_enabled(self) -> bool:
        return (
            self.rollback_unit is RollbackUnit.CHANNELS
            and bool(self.channel_groups)
            and self.gate.config.max_group_rollbacks > 0
        )

    @property
    def max_verdicts_per_decision(self) -> int:
        """Worst-case verdicts of one decision: 1 when disabled, else ``1 + len(shrink_lambdas)``
        plus ``min(max_group_rollbacks, len(channel_groups))`` when the group stage is enabled."""
        cfg = self.gate.config
        if not cfg.enabled:
            return 1
        groups = min(cfg.max_group_rollbacks, len(self.channel_groups)) if self.group_stage_enabled else 0
        return 1 + len(cfg.shrink_lambdas) + groups

    # ---------------------------------------------------------------- the procedure
    def run(
        self,
        model: nn.Module,
        operator: UpdateOperator,
        incumbent: StateSnapshot,
        decision_index: int,
        update_index: int,
        alarm: bool | AlarmState,
        reward_v1: float | None,
        reward_v2: float | None = None,
    ) -> tuple[DecisionRecord, StateSnapshot]:
        """Run one gate procedure on ``model``'s current weights; returns the record and the new
        incumbent (the same object unless the decision promoted the weights).

        ``alarm`` is a bool or an :class:`~research_rl.capability.alarm.AlarmState` (its
        ``alarm`` flag). ``update_index`` is recorded as given (the loop passes the index of the
        block's last update). Preconditions checked before any verdict (``ValueError``; no alpha
        is spent and nothing is written then): ``decision_index`` / ``update_index`` ints >= 0, an
        incumbent that holds optimizer state whenever ``operator.optimizer`` is not None (a
        ROLLBACK must restore it), and, when the gate is enabled and the group stage is enabled,
        that every channel group can be reverted to this incumbent (:meth:`check_group_scope`).
        An exception raised later leaves the model at the candidate (see the module docstring,
        "Exceptions during the procedure").
        """
        if not isinstance(incumbent, StateSnapshot):
            raise TypeError(f"incumbent must be a StateSnapshot, got {type(incumbent).__name__}")
        decision_index = _check_index("decision_index", decision_index)
        update_index = _check_index("update_index", update_index)
        if isinstance(alarm, AlarmState):
            alarm_flag = alarm.alarm
        elif isinstance(alarm, bool):
            alarm_flag = alarm
        else:
            raise TypeError(f"alarm must be a bool or an AlarmState, got {type(alarm).__name__}")
        optimizer = operator.optimizer
        if optimizer is not None and incumbent.optimizer_state is None:
            raise ValueError(
                "the operator has an optimizer but the incumbent holds no optimizer state; capture "
                "it with weights.capture(..., optimizer=operator.optimizer)"
            )
        cfg = self.gate.config
        if cfg.enabled and self.group_stage_enabled:
            self.check_group_scope(model, incumbent)
        verdicts: list[GateVerdict] = []

        def evaluate(attempt: int) -> GateVerdict:
            evidence = self.probe.evidence(
                model, incumbent, decision_index, attempt=attempt, alarm=alarm_flag
            )
            verdict = self._verdict_gate.verdict(evidence)
            verdicts.append(verdict)
            return verdict

        def record(
            decision: GateDecision,
            reason: str,
            *,
            shrink_lambda: float | None = None,
            groups: Sequence[str] = (),
            override: str | None = None,
        ) -> DecisionRecord:
            return DecisionRecord(
                decision_index=decision_index,
                update_index=update_index,
                decision=decision,
                verdicts=verdicts,
                reason=reason,
                shrink_lambda=shrink_lambda,
                groups_reverted=list(groups),
                reward_v1=reward_v1,
                reward_v2=reward_v2,
                alarm=alarm_flag,
                override=override,
            )

        first = evaluate(0)
        why = classify_verdict(first, require_transfer_gain=cfg.require_transfer_gain)

        if not cfg.enabled:
            new = self._promote(model, incumbent, optimizer)
            self.holds = 0
            operator.on_decision(GateDecision.ACCEPT)
            return record(GateDecision.ACCEPT, WOULD_PREFIX + why, override=OVERRIDE_GATE_DISABLED), new

        if why == REASON_ACCEPT:
            new = self._promote(model, incumbent, optimizer)
            self.holds = 0
            operator.on_decision(GateDecision.ACCEPT)
            return record(GateDecision.ACCEPT, why), new
        if why in (REASON_NO_TRANSFER_GAIN, REASON_ALARM_HOLD):
            if self.holds >= cfg.hold_limit:
                return self._rollback(model, operator, incumbent, record, REASON_HOLD_LIMIT)
            self.holds += 1
            operator.on_decision(GateDecision.HOLD)
            return record(GateDecision.HOLD, why), incumbent
        if why in (REASON_CONFIRMED_ALARM, REASON_HORIZON_EXHAUSTED):
            return self._rollback(model, operator, incumbent, record, why)

        # ---- retention failed: SHRINK, then GROUP_ROLLBACK, then ROLLBACK
        candidate = capture(model, incumbent.names(), optimizer=optimizer)
        attempt = 0
        finish: Callable[[], tuple[DecisionRecord, StateSnapshot]] | None = None
        # The trial writes and re-tests are exception-safe: if anything raises before the outcome
        # is chosen (a re-test OOM, KeyboardInterrupt, a scorer error), the raw candidate's
        # parameters and masters are written back (the remedy stage never touches the moments)
        # and the exception propagates; the outcome itself (_passed / _rollback, which also tell
        # the operator) is applied outside this block.
        try:
            for lam in cfg.shrink_lambdas:
                interpolate(model, incumbent, candidate, lam, optimizer=optimizer)
                attempt += 1
                verdict = evaluate(attempt)
                if verdict.retention_ok:
                    finish = functools.partial(
                        self._passed,
                        model,
                        operator,
                        incumbent,
                        record,
                        verdict,
                        GateDecision.SHRINK,
                        shrink_lambda=lam,
                    )
                    break
                if retention_unpassable(verdict):
                    finish = functools.partial(
                        self._rollback, model, operator, incumbent, record, REASON_HORIZON_EXHAUSTED
                    )
                    break
            if finish is None and self.group_stage_enabled:
                restore(model, candidate, optimizer=optimizer, restore_optimizer=False)
                order = self._damage_order(model, incumbent, decision_index, first)
                reverted: list[str] = []
                for group in order[: cfg.max_group_rollbacks]:
                    revert_groups(model, incumbent, [group], optimizer=optimizer)
                    reverted.append(group.name)
                    attempt += 1
                    verdict = evaluate(attempt)
                    if verdict.retention_ok:
                        finish = functools.partial(
                            self._passed,
                            model,
                            operator,
                            incumbent,
                            record,
                            verdict,
                            GateDecision.GROUP_ROLLBACK,
                            groups=list(reverted),
                        )
                        break
                    if retention_unpassable(verdict):
                        finish = functools.partial(
                            self._rollback, model, operator, incumbent, record, REASON_HORIZON_EXHAUSTED
                        )
                        break
        except BaseException:
            restore(model, candidate, optimizer=optimizer, restore_optimizer=False)
            raise
        if finish is None:
            finish = functools.partial(
                self._rollback, model, operator, incumbent, record, REASON_RETENTION_FAILED
            )
        return finish()

    def check_group_scope(self, model: nn.Module, incumbent: StateSnapshot) -> None:
        """Raise :class:`~research_rl.capability.weights.SnapshotError` (a ``ValueError``) unless
        every channel group can be ranked and reverted against ``incumbent`` on ``model``.

        Runs exactly the pure checks that :func:`~research_rl.capability.weights.group_damage` and
        :func:`~research_rl.capability.weights.revert_groups` run before they compute or write:
        the groups' selections are valid for ``model``, the incumbent's ``params`` / ``masters``
        fit ``model``, the incumbent covers every group layer (its ``channel_state`` or the
        layer's FFN parameters, i.e. the layer lies inside ``operator.parameter_names()``) and its
        slices have the model's shapes, dtypes, bias presence and module paths. Writes nothing.
        The result is cached for the last ``(model, incumbent)`` pair (by identity, through weak
        references, so the cache keeps no snapshot alive), so a run of
        HOLDs pays it once; the check copies the incumbent's slices of the groups once per new
        incumbent.
        """
        cached = self._scope_checked
        if cached is not None and cached[0]() is model and cached[1]() is incumbent:
            return
        union = self.channel_groups[0].selection
        for g in self.channel_groups[1:]:
            union = union.union(g.selection)
        for g in self.channel_groups:
            g.selection.validate(model)
        _check_incumbent(model, incumbent, "group stage")
        if union:
            _check_reference(model, _reference_state(model, incumbent, union))
        self._scope_checked = (weakref.ref(model), weakref.ref(incumbent))

    # ---------------------------------------------------------------- helpers
    def _promote(
        self, model: nn.Module, incumbent: StateSnapshot, optimizer: torch.optim.Optimizer | None
    ) -> StateSnapshot:
        layers = list(incumbent.channel_state.layers) if incumbent.channel_state is not None else None
        return capture(model, incumbent.names(), optimizer=optimizer, channel_layers=layers)

    def _rollback(
        self,
        model: nn.Module,
        operator: UpdateOperator,
        incumbent: StateSnapshot,
        record: Callable[..., DecisionRecord],
        reason: str,
    ) -> tuple[DecisionRecord, StateSnapshot]:
        restore(model, incumbent, optimizer=operator.optimizer)
        self.holds = 0
        operator.on_decision(GateDecision.ROLLBACK)
        return record(GateDecision.ROLLBACK, reason), incumbent

    def _passed(
        self,
        model: nn.Module,
        operator: UpdateOperator,
        incumbent: StateSnapshot,
        record: Callable[..., DecisionRecord],
        verdict: GateVerdict,
        decision: GateDecision,
        *,
        shrink_lambda: float | None = None,
        groups: Sequence[str] = (),
    ) -> tuple[DecisionRecord, StateSnapshot]:
        """A re-test whose retention passed: promote iff transfer passes and there is no alarm."""
        cfg = self.gate.config
        promoted = (verdict.transfer_ok or not cfg.require_transfer_gain) and not verdict.alarm
        stage = "shrink" if decision is GateDecision.SHRINK else "group_rollback"
        reason = _promoted_reason(stage, promoted)
        if promoted:
            new = self._promote(model, incumbent, operator.optimizer)
            self.holds = 0
        else:
            # Pinned 1.16: a passing shrink / group revert is always kept; it extends the hold
            # streak (capped at hold_limit) but is never itself converted into a ROLLBACK.
            self.holds = min(self.holds + 1, cfg.hold_limit)
            new = incumbent
        operator.on_decision(decision)
        return record(decision, reason, shrink_lambda=shrink_lambda, groups=groups), new

    def _damage_order(
        self, model: nn.Module, incumbent: StateSnapshot, decision_index: int, first: GateVerdict
    ) -> list[ChannelGroup]:
        """Groups with non-zero damage, largest first (see the module docstring)."""
        failing = [t.slice for t in first.retention if not t.passed]
        per_slice = self.probe.gate_items(decision_index, 0)
        items = [it for name in failing for it in per_slice[name]]
        texts = list(self.damage_texts_fn(items))
        layers = sorted({layer for g in self.channel_groups for layer in g.selection.layers()})
        stats = collect_channel_stats(
            model, self.tokenizer, texts, layers=layers, batch_size=self.damage_batch_size
        )
        activations = stats.mean_abs
        damage = group_damage(incumbent, model, activations, self.channel_groups)
        ordered = channel_groups_by_damage(incumbent, model, activations, self.channel_groups)
        return [g for g in ordered if math.isnan(damage[g.name]) or damage[g.name] != 0.0]

    # ---------------------------------------------------------------- state
    def state_dict(self) -> dict[str, Any]:
        """JSON-able: hold streak, the verdict gate's state (ledger position, verdict count),
        ``enabled``, the rollback unit and the group names (checked on load)."""
        return {
            "version": _CONTROLLER_STATE_VERSION,
            "enabled": self.gate.config.enabled,
            "rollback_unit": self.rollback_unit.value,
            "channel_groups": [g.name for g in self.channel_groups],
            "holds": self.holds,
            "verdict_gate": self._verdict_gate.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a :meth:`state_dict`; ``ValueError`` on any mismatch (nothing changes then)."""
        if state.get("version") != _CONTROLLER_STATE_VERSION:
            raise ValueError(f"unsupported controller state version {state.get('version')!r}")
        mine = self.state_dict()
        for key in ("enabled", "rollback_unit", "channel_groups"):
            if state.get(key) != mine[key]:
                raise ValueError(f"controller state mismatch on {key!r}: {state.get(key)!r} != {mine[key]!r}")
        holds = state.get("holds")
        if not _is_int(holds) or not 0 <= holds <= self.gate.config.hold_limit:
            raise ValueError(f"invalid hold streak {holds!r}")
        self._verdict_gate.load_state_dict(state["verdict_gate"])
        self.holds = int(holds)


# ============================================================================== snapshots as state
def snapshot_to_state(snapshot: StateSnapshot) -> dict[str, Any]:
    """A :class:`StateSnapshot` as a plain dict loadable with ``torch.load(weights_only=True)``.

    Format: ``{"params": {name: CPU tensor}, "optimizer_state": optimizer state_dict tree or None,
    "masters": {name: fp32 CPU tensor} or None, "fingerprint": str}``. The tensors are the
    snapshot's own (not cloned; treat them as read-only). A snapshot with a ``channel_state``
    raises ``ValueError`` (the loop never captures one).
    """
    if snapshot.channel_state is not None:
        raise ValueError("snapshots with a channel_state cannot be serialised by the gated loop")
    return {
        "params": dict(snapshot.params),
        "optimizer_state": snapshot.optimizer_state,
        "masters": None if snapshot.masters is None else dict(snapshot.masters),
        "fingerprint": snapshot.fingerprint,
    }


def snapshot_from_state(state: Mapping[str, Any]) -> StateSnapshot:
    """Inverse of :func:`snapshot_to_state`; the fingerprint is re-verified
    (:class:`~research_rl.capability.weights.SnapshotError` on mismatch)."""
    params = state["params"]
    if not isinstance(params, Mapping) or not all(isinstance(v, torch.Tensor) for v in params.values()):
        raise ValueError("snapshot state 'params' must map names to tensors")
    masters = state.get("masters")
    snap = StateSnapshot(
        params=dict(params),
        optimizer_state=state.get("optimizer_state"),
        masters=None if masters is None else dict(masters),
        channel_state=None,
        fingerprint=str(state["fingerprint"]),
    )
    snap.verify()
    return snap


# ============================================================================== the loop
class MetricsRecorder(Protocol):
    """Anything with ``log_metrics(step, logs)``, e.g.
    :class:`~research_rl.core.experiment.ExperimentRecorder`."""

    def log_metrics(self, step: int, logs: Mapping[str, Any]) -> None: ...


def worst_case_verdicts(
    num_updates: int,
    *,
    block_updates: int,
    verdicts_per_decision: int,
    alarm_triggers_decision: bool,
    since: int = 0,
) -> int:
    """Upper bound on the verdicts of ``num_updates`` more updates of a :class:`GatedLoop`.

    ``since`` updates of the running block are already done. Block decisions happen only when the
    running block reaches ``block_updates`` updates, so there are at most
    ``b = min(num_updates, floor((since + num_updates) / block_updates))`` of them, each costing
    at most ``verdicts_per_decision`` verdicts. With ``alarm_triggers_decision`` every other update
    can end in an alarm decision, which costs exactly one verdict (an alarm never reaches the
    remedy stage). Returns ``b * verdicts_per_decision + (num_updates - b if alarm_triggers_decision
    else 0)``.
    """
    for name, value in (("num_updates", num_updates), ("since", since)):
        _check_index(name, value)
    if not _is_int(block_updates) or block_updates < 1:
        raise ValueError(f"block_updates must be an int >= 1, got {block_updates!r}")
    if not _is_int(verdicts_per_decision) or verdicts_per_decision < 1:
        raise ValueError(f"verdicts_per_decision must be an int >= 1, got {verdicts_per_decision!r}")
    blocks = min(num_updates, (since + num_updates) // block_updates)
    extra = num_updates - blocks if alarm_triggers_decision else 0
    return blocks * verdicts_per_decision + extra


class GatedLoop:
    """Operator updates, alarm, gate decisions, decision log and resume state.

    Args:
        model: the trained model (the one ``operator.step`` writes).
        operator: any :class:`~research_rl.capability.types.UpdateOperator`. The loop only calls
            ``step``, ``parameter_names``, ``optimizer``, ``state_dict`` / ``load_state_dict``; the
            controller calls ``on_decision``.
        controller: the :class:`GateController`; its ``gate.config`` must equal ``config``.
        alarm: the :class:`~research_rl.capability.alarm.DisagreementAlarm` fed with every report
            (build it with ``rollouts_per_unit=population`` for ES, or rely on GRPO's
            ``report.extra["alarm_units"]``).
        config: the :class:`~research_rl.capability.gate.GateConfig` (``block_updates``,
            ``alarm_triggers_decision``).
        recorder: optional :class:`MetricsRecorder`; after every decision it receives
            ``log_metrics(record.update_index, {...})`` with the decision, reason, rewards, alarm,
            verdict count and the first verdict's transfer point / bound.
        log_path: optional JSONL file; one ``DecisionRecord.to_dict()`` per line, appended and
            flushed after every decision (the file is never truncated except by
            :meth:`load_state_dict`, see there).
        allow_short_horizon: turn the start-of-run horizon check's ``ValueError`` into a
            ``UserWarning`` (see :meth:`check_horizon`).

    Schedule: update ``t`` (0, 1, ...) runs ``operator.step(model, t)`` and ``alarm.update(report)``.
    A decision follows update ``t`` when the running block holds ``block_updates`` updates, or at
    once when the alarm fired at ``t`` and ``alarm_triggers_decision``; the block restarts after
    every decision. The decision's ``alarm`` is True iff the alarm fired at any update of the block
    (latched, so a one-update pulse is not lost when alarms do not trigger decisions); its
    ``reward_v1`` / ``reward_v2`` are the unweighted means over the block's reports
    (``reward_v2`` None if any report had None); ``update_index`` is ``t``.

    The incumbent at construction is ``weights.capture(model, operator.parameter_names(),
    optimizer=operator.optimizer)`` (no channel state).
    """

    def __init__(
        self,
        model: nn.Module,
        operator: UpdateOperator,
        controller: GateController,
        alarm: DisagreementAlarm,
        *,
        config: GateConfig,
        recorder: MetricsRecorder | None = None,
        log_path: str | Path | None = None,
        allow_short_horizon: bool = False,
    ) -> None:
        if not isinstance(operator, UpdateOperator):
            raise TypeError(f"operator must implement UpdateOperator, got {type(operator).__name__}")
        if not isinstance(controller, GateController):
            raise TypeError(f"controller must be a GateController, got {type(controller).__name__}")
        if not isinstance(alarm, DisagreementAlarm):
            raise TypeError(f"alarm must be a DisagreementAlarm, got {type(alarm).__name__}")
        if not isinstance(config, GateConfig):
            raise TypeError(f"config must be a GateConfig, got {type(config).__name__}")
        if config != controller.gate.config:
            raise ValueError("config must equal controller.gate.config")
        if recorder is not None and not callable(getattr(recorder, "log_metrics", None)):
            raise TypeError("recorder must provide log_metrics(step, logs)")
        self.model = model
        self.operator = operator
        self.controller = controller
        self.alarm = alarm
        self.config = config
        self.recorder = recorder
        self.log_path = Path(log_path) if log_path is not None else None
        self.allow_short_horizon = bool(allow_short_horizon)
        self.incumbent: StateSnapshot = capture(
            model, operator.parameter_names(), optimizer=operator.optimizer
        )
        self.update_index = 0
        self.decision_index = 0
        self.records: list[DecisionRecord] = []
        self.last_alarm: AlarmState | None = None
        self._reset_block()

    def _reset_block(self) -> None:
        self._block_v1: list[float] = []
        self._block_v2: list[float | None] = []
        self._block_alarm = False

    @property
    def updates_since_decision(self) -> int:
        return len(self._block_v1)

    @property
    def decision_pending(self) -> bool:
        """True when the running block is owed a decision that has not been recorded: it holds
        ``block_updates`` updates, or its latched alarm fired with ``alarm_triggers_decision``.
        This happens only when a decision raised (see :meth:`run`); the next :meth:`run` takes
        that decision before its first update."""
        n = len(self._block_v1)
        return n >= self.config.block_updates or (
            n > 0 and self._block_alarm and self.config.alarm_triggers_decision
        )

    # ---------------------------------------------------------------- horizon
    def _horizon_config(self) -> GateConfig:
        """``config`` with ``alpha`` / ``alpha_schedule`` of the ledger the verdicts spend from
        (``controller.verdict_gate.ledger``; an explicit gate ledger is authoritative)."""
        ledger = self.controller.verdict_gate.ledger
        return dataclasses.replace(self.config, alpha=ledger.total_alpha, alpha_schedule=ledger.schedule)

    def horizon(self) -> int | None:
        """Last verdict index (1-based, ledger count) at which every retention slice of the probe
        can still pass with ``n10 = round(0.005 n)`` incumbent-only wins and no candidate-only win
        (``GateConfig.retention_horizon`` per slice, ``n = probe.items_per_retention_slice``, the
        size the probe actually tests, falling back to ``config.items_per_retention_slice``);
        ``None`` = no practical limit. The minimum over the probe's retention slices. The alpha
        and schedule are those of ``controller.verdict_gate.ledger`` (the ledger the verdicts
        actually spend from), which may differ from ``config.alpha`` / ``config.alpha_schedule``
        when the gate was built with an explicit ledger."""
        probe = self.controller.probe
        n = getattr(probe, "items_per_retention_slice", self.config.items_per_retention_slice)
        slices = list(getattr(probe, "retention_slices", ())) or [None]
        horizon_config = self._horizon_config()
        best: int | None = None
        for name in slices:
            k = horizon_config.retention_horizon(name, n=n, n10=round(0.005 * n), n01=0)
            if k is not None and (best is None or k < best):
                best = k
        return best

    def required_verdicts(self, num_updates: int) -> int:
        """:func:`worst_case_verdicts` of ``num_updates`` more updates from the current position
        (with a :attr:`decision_pending`: that decision's ``max_verdicts_per_decision`` plus the
        worst case of ``num_updates`` updates from an empty block)."""
        num_updates = _check_index("num_updates", num_updates)
        per_decision = self.controller.max_verdicts_per_decision
        pending = self.decision_pending
        return (per_decision if pending else 0) + worst_case_verdicts(
            num_updates,
            block_updates=self.config.block_updates,
            verdicts_per_decision=per_decision,
            alarm_triggers_decision=self.config.alarm_triggers_decision,
            since=0 if pending else self.updates_since_decision,
        )

    def check_horizon(self, num_updates: int) -> None:
        """Start-of-run check: ``verdict_gate.ledger.count + required_verdicts(num_updates)`` must
        not exceed :meth:`horizon`. Otherwise ``ValueError`` (``UserWarning`` with
        ``allow_short_horizon=True``). Run by :meth:`run` before the first update."""
        horizon = self.horizon()
        if horizon is None:
            return
        used = self.controller.verdict_gate.ledger.count
        need = used + self.required_verdicts(num_updates)
        if need <= horizon:
            return
        n = getattr(self.controller.probe, "items_per_retention_slice", self.config.items_per_retention_slice)
        ledger = self.controller.verdict_gate.ledger
        message = (
            f"retention horizon too short: the run may need up to {need} verdicts ({used} already used), "
            f"but a retention slice of {n} items with {round(0.005 * n)} incumbent-only wins can pass only "
            f"up to verdict {horizon} under alpha={ledger.total_alpha} ({ledger.schedule}, the verdict ledger); later "
            "verdicts would be 'horizon_exhausted'. Increase items_per_retention_slice, the margin or alpha, "
            "or plan fewer decisions"
        )
        if not self.allow_short_horizon:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=3)

    # ---------------------------------------------------------------- running
    def run(self, num_updates: int) -> list[DecisionRecord]:
        """Run ``num_updates`` more operator updates (with their decisions); returns the decision
        records of this call (also appended to :attr:`records`). The horizon check runs first.

        Exceptions: if a decision raises (a probe / scorer error, OOM, ``KeyboardInterrupt``), the
        controller has written the raw candidate back (see :meth:`GateController.run`), no record
        is written, ``decision_index`` is not advanced and the running block (rewards, latched
        alarm) is kept, so :attr:`decision_pending` is True; the verdicts already computed have
        spent their alpha. :meth:`state_dict` / :meth:`load_state_dict` keep that state, and the
        next :meth:`run` (also ``run(0)``) first takes the pending decision, on the same weights,
        with the same ``decision_index`` (so the same attempt-0 items; the ledger moves on) and
        ``update_index`` = the block's last update."""
        num_updates = _check_index("num_updates", num_updates)
        self.check_horizon(num_updates)
        out: list[DecisionRecord] = []
        cfg = self.config
        if self.decision_pending:
            out.append(self._decide(self.update_index - 1))
        for _ in range(num_updates):
            t = self.update_index
            report = self.operator.step(self.model, t)
            if not isinstance(report, OperatorReport):
                raise TypeError(f"operator.step must return an OperatorReport, got {type(report).__name__}")
            state = self.alarm.update(report)
            self.last_alarm = state
            self._block_v1.append(float(report.reward_v1))
            self._block_v2.append(None if report.reward_v2 is None else float(report.reward_v2))
            self._block_alarm = self._block_alarm or state.alarm
            self.update_index = t + 1
            if len(self._block_v1) >= cfg.block_updates or (state.alarm and cfg.alarm_triggers_decision):
                out.append(self._decide(t))
        return out

    def _decide(self, t: int) -> DecisionRecord:
        n = len(self._block_v1)
        reward_v1 = math.fsum(self._block_v1) / n
        reward_v2 = (
            None
            if any(v is None for v in self._block_v2)
            else math.fsum(v for v in self._block_v2 if v is not None) / n
        )
        record, incumbent = self.controller.run(
            self.model,
            self.operator,
            self.incumbent,
            self.decision_index,
            t,
            self._block_alarm,
            reward_v1,
            reward_v2,
        )
        self.incumbent = incumbent
        self.decision_index += 1
        self._reset_block()
        self.records.append(record)
        self._write_log(record)
        if self.recorder is not None:
            first = record.verdicts[0] if record.verdicts else None
            logs: dict[str, Any] = {
                "gate/decision_index": record.decision_index,
                "gate/decision": record.decision.value,
                "gate/reason": record.reason,
                "gate/alarm": record.alarm,
                "gate/verdicts": len(record.verdicts),
                "gate/reward_v1": record.reward_v1,
                "gate/reward_v2": record.reward_v2,
                "gate/override": record.override,
            }
            if first is not None:
                logs["gate/transfer_point"] = first.transfer.point
                logs["gate/transfer_bound"] = first.transfer.bound
            self.recorder.log_metrics(record.update_index, logs)
        return record

    def _write_log(self, record: DecisionRecord) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict()) + "\n")
            fh.flush()

    # ---------------------------------------------------------------- state
    def state_dict(self) -> dict[str, Any]:
        """Everything needed to continue the run exactly; loadable with
        ``torch.save`` / ``torch.load(weights_only=True)`` (tensors, numbers, strings, lists, dicts).

        Keys: ``version``, ``config`` (``GateConfig.to_dict``), ``update_index`` (next update),
        ``decision_index`` (next decision), ``block`` (``reward_v1`` / ``reward_v2`` lists of the
        running block's reports and the latched ``alarm``), ``incumbent`` and ``current``
        (:func:`snapshot_to_state` of the incumbent and of ``capture(model,
        operator.parameter_names(), optimizer=operator.optimizer)`` - the current trained weights,
        optimizer state and masters), ``controller`` (hold streak, verdict gate and its ledger),
        ``sampler`` (the probe sampler's identity; it has no position: items are keyed by
        ``(purpose, index)``), ``alarm`` and ``operator`` state dicts. Parameters outside
        ``operator.parameter_names()`` are not saved: rebuild the model identically before loading.
        After a decision raised (:attr:`decision_pending`), the owed block (up to
        ``block_updates`` updates) is saved as it is; the restored loop takes that decision at its
        next :meth:`run`.
        """
        current = capture(self.model, self.operator.parameter_names(), optimizer=self.operator.optimizer)
        return {
            "version": _LOOP_STATE_VERSION,
            "config": self.config.to_dict(),
            "update_index": self.update_index,
            "decision_index": self.decision_index,
            "block": {
                "reward_v1": list(self._block_v1),
                "reward_v2": list(self._block_v2),
                "alarm": self._block_alarm,
            },
            "incumbent": snapshot_to_state(self.incumbent),
            "current": snapshot_to_state(current),
            "controller": self.controller.state_dict(),
            "sampler": self.controller.probe.sampler.state_dict(),
            "alarm": self.alarm.state_dict(),
            "operator": self.operator.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Continue from a :meth:`state_dict` (build every object exactly as the saved run did).

        Order: check version, config and the running block, read and validate the decision log
        (with ``log_path``, see below), check the sampler identity, rebuild and verify both
        snapshots, validate the alarm state on a scratch alarm, then write the current weights /
        optimizer state / masters with ``weights.restore`` (all of its checks run before it
        writes), load the controller, the alarm and finally the operator (after the weights,
        because an ES operator re-bases its space there), and last rewrite the log.

        Decision log: every line is parsed before anything is loaded. Only the **last** line may
        be unparsable JSON: it is a record torn by a crash during its write and is dropped (it
        was written after the checkpoint, since a checkpoint is taken after its records were
        flushed, so it would be dropped anyway). Any other unparsable line, or a record without
        an int ``decision_index``, raises ``ValueError`` before anything is changed (loop and
        file untouched). The file is then rewritten (via a temporary file and ``os.replace``) to
        keep only the records with ``decision_index`` below the loaded one, in their order
        (records written after the checkpoint will be produced again). ``ValueError`` on
        mismatches; a failure after the weights were written (an operator / controller state
        that does not fit) leaves the loop partially loaded and the log unchanged.
        """
        if state.get("version") != _LOOP_STATE_VERSION:
            raise ValueError(f"unsupported loop state version {state.get('version')!r}")
        if GateConfig.from_dict(state["config"]) != self.config:
            raise ValueError("loop state was saved with a different GateConfig")
        update_index = _check_index("update_index", state["update_index"])
        decision_index = _check_index("decision_index", state["decision_index"])
        block = state["block"]
        v1 = [float(v) for v in block["reward_v1"]]
        v2 = [None if v is None else float(v) for v in block["reward_v2"]]
        if len(v1) != len(v2) or len(v1) > self.config.block_updates:
            raise ValueError(f"invalid running block of {len(v1)} updates")
        if not isinstance(block["alarm"], bool):
            raise ValueError(f"invalid block alarm flag {block['alarm']!r}")
        kept_log = self._read_log(decision_index)
        self.controller.probe.sampler.load_state_dict(state["sampler"])
        incumbent = snapshot_from_state(state["incumbent"])
        current = snapshot_from_state(state["current"])
        names = set(self.operator.parameter_names())
        if set(incumbent.params) != names or set(current.params) != names:
            raise ValueError("saved snapshots do not cover exactly the operator's parameters")
        scratch = DisagreementAlarm(**self.alarm.config())
        scratch.load_state_dict(state["alarm"])
        restore(self.model, current, optimizer=self.operator.optimizer)
        self.controller.load_state_dict(state["controller"])
        self.alarm.load_state_dict(state["alarm"])
        self.operator.load_state_dict(state["operator"])
        self.incumbent = incumbent
        self.update_index = update_index
        self.decision_index = decision_index
        self._block_v1 = v1
        self._block_v2 = v2
        self._block_alarm = block["alarm"]
        self.last_alarm = None
        if kept_log is not None:
            self._rewrite_log(kept_log)

    def _read_log(self, decision_index: int) -> list[str] | None:
        """The log lines to keep on resume (see :meth:`load_state_dict`); ``None`` without a log
        file. Pure: reads only."""
        if self.log_path is None or not self.log_path.exists():
            return None
        lines = [ln for ln in self.log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        kept: list[str] = []
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if i == len(lines) - 1:
                    break  # torn last record of a crash mid-write: dropped
                raise ValueError(
                    f"decision log {self.log_path} line {i + 1} is not valid JSON (only a torn last "
                    f"line is tolerated): {exc}"
                ) from exc
            index = row.get("decision_index") if isinstance(row, dict) else None
            if not _is_int(index):
                raise ValueError(
                    f"decision log {self.log_path} line {i + 1} has no int 'decision_index': {line[:80]!r}"
                )
            if index < decision_index:
                kept.append(line)
        return kept

    def _rewrite_log(self, kept: Sequence[str]) -> None:
        assert self.log_path is not None
        tmp = self.log_path.with_name(self.log_path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write("".join(ln + "\n" for ln in kept))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.log_path)
