"""The capability gate inside the existing TRL GRPO path (``docs/research_rl/CAPABILITY.md`` 1.10).

* :class:`CapabilityGateCallback` - a ``transformers.TrainerCallback`` and a
  :class:`~research_rl.core.checkpoint.StatefulComponent` that runs the framework-agnostic
  :class:`~research_rl.capability.loop.GateController` on the trainer's model and optimizer every
  ``block_updates`` optimizer steps (and on a tier-1 alarm).
* :class:`DisagreementScreen` - an :class:`~research_rl.trainers.rl_common.AdvantageShaper` that
  scores every training rollout with V1 and V2 under the gate's truncation rule, gives the
  V1-pass / V2-fail rows advantage 0 (signal ``False``) and feeds the
  :class:`~research_rl.capability.alarm.DisagreementAlarm` with prompt groups as units.

TRL 1.18 facts this module relies on (read in ``trl/trainer/grpo_trainer.py``)
--------------------------------------------------------------------------------
* ``GRPOTrainer.training_step`` increments ``self._step`` once per **micro-batch**;
  ``_prepare_inputs`` generates a new batch iff ``self._step % (steps_per_generation *
  num_iterations) == 0`` or ``self._buffered_inputs is None`` and otherwise trains on a slice of the
  buffered batch. A decision taken at ``on_step_end`` when ``trainer._step`` is a multiple of
  ``generate_every = steps_per_generation * num_iterations`` is therefore followed by a fresh
  generation from the (possibly restored) weights.
* ``old_per_token_logps`` are computed at generation time iff ``gradient_accumulation_steps %
  generate_every != 0`` (or vLLM importance sampling); otherwise they are ``None`` and the loss uses
  the current log-probs detached. Either way they belong to the generation batch, so a fresh
  generation also brings fresh old log-probs: nothing computed from pre-decision weights survives a
  decision on a generation boundary.
* HF calls ``lr_scheduler.step()`` **before** ``on_step_end``, hands the prepared optimizer and the
  model to every callback event and has already loaded checkpoint weights / optimizer state when
  ``on_train_begin`` runs. ``ResearchTrainerMixin.train`` loads the research components
  (:meth:`CapabilityGateCallback.load_state_dict`) *before* HF loads the checkpoint, so the
  callback only stashes its state there and applies it in ``on_train_begin``.

Schedule
--------
* **Alignment** (checked in ``on_train_begin``, ``ValueError``): ``block_updates *
  gradient_accumulation_steps`` must be a multiple of ``steps_per_generation * num_iterations``, so
  block decisions always land on a generation boundary.
* **Block decisions** run at ``on_step_end`` whenever ``state.global_step`` is a multiple of
  ``block_updates`` (fixed multiples of the global step, also after a resume and after an alarm
  decision; unlike :class:`~research_rl.capability.loop.GatedLoop`, the block does *not* restart
  after an alarm decision, because that would push every later decision off the generation
  boundary).
* **Alarm decisions**: with an ``alarm`` and ``gate.config.alarm_triggers_decision``, a decision
  also runs at the ``on_step_end`` of any step during which the alarm fired (the alarm's
  ``gap_alarms + length_alarms`` counter grew). The decision's ``alarm`` flag is latched over all
  steps since the previous decision (so with ``alarm_triggers_decision=False`` a pulse reaches the
  next block decision).
* **Off-boundary decisions** (an alarm decision, or any decision when ``trainer._step`` is not a
  multiple of ``generate_every``, e.g. after a resume from an unaligned checkpoint): if the decision
  changed the weights (ROLLBACK, SHRINK, GROUP_ROLLBACK) the callback sets
  ``trainer._buffered_inputs = None``, so TRL regenerates at the very next micro-batch from the new
  weights (with old log-probs from those weights). The rest of that generation block then trains on
  the regenerated batch (TRL picks slice ``_step % steps_per_generation`` of it), and the next
  regular generation happens at the next multiple of ``generate_every``. ACCEPT and HOLD leave the
  weights untouched and never invalidate. :attr:`CapabilityGateCallback.buffer_invalidations`
  counts invalidations; every decision-log line records ``buffer_invalidated``.

What a decision writes (beyond :class:`~research_rl.capability.loop.GateController`)
----------------------------------------------------------------------------------------
The incumbent covers the model's **trainable** parameters (``requires_grad``, by
``named_parameters`` name, no channel state) with the optimizer state (and
``HookedOptimizer`` masters and rule state) of the **unwrapped** optimizer
(:func:`~research_rl.optim.optimizer_wrapper.unwrap_hooked`, else the innermost
``torch.optim.Optimizer`` below accelerate's wrapper). A ROLLBACK restores the parameters, the
per-parameter optimizer state (moments, Adam step tensors), masters and update-rule state
bit-identically to the incumbent, with two deliberate exceptions kept at their pre-decision values:

* the ``HookedOptimizer`` step counters ``nominal_steps`` / ``applied_updates`` /
  ``skipped_steps``. ``policy_version`` (``applied_updates``, stamped on every rollout by
  :mod:`research_rl.trainers.rl_common`) therefore counts **every applied optimizer step, including
  rolled-back ones**: it stays monotonic, ``nominal_steps`` keeps tracking HF's ``global_step``
  (the recorder and the stochastic-rounding seeds rely on it), and the policy lag of a micro-batch
  is never negative. It is not an identifier of the weights (after a rollback the weights equal an
  earlier version's).
* every ``param_groups`` hyper-parameter (``lr``, ``initial_lr``, ``betas``, ...): HF's LR scheduler
  owns them and has already stepped for the next update when ``on_step_end`` runs, so restoring the
  incumbent's ``lr`` would train the next step at a stale learning rate. LR-scheduler and RNG state
  are not rolled back either (as documented in :mod:`research_rl.capability.weights`).

Nothing outside the trainable parameters and the optimizer is written, in particular not TRL's
``ref_model``. TRL's reference-model sync (``sync_ref_model=True``) mixes the policy into
``ref_model`` at its sync steps and a decision could not undo that, so an enabled gate refuses
it in ``on_train_begin`` (``ValueError``); the disabled gate (``enabled=False``) never changes the
weights and allows it.

Exceptions: if the gate procedure raises (a probe / scorer error, OOM, ``KeyboardInterrupt``),
the controller has written the raw candidate's parameters and masters back (see
:mod:`research_rl.capability.loop`, "Exceptions during the procedure"), the callback puts
the step counters and ``param_groups`` back, and the exception propagates out of ``train()``. No
record is written, ``decision_index`` is not advanced and the running block (rewards, latched
alarm) is kept; the verdicts already computed have spent their alpha. A further ``train()``
continues the gate (see "Repeated train()"), so the block's rewards and alarm reach its next
decision.

State and resume
----------------
:meth:`CapabilityGateCallback.bind` registers the callback in ``trainer.research_components``
(default key :data:`GATE_COMPONENT`), so it is saved in ``research_rl_components.pt`` with every
checkpoint: GateConfig, incumbent snapshot, controller (hold streak, verdict gate, ledger), decision
index, the running block, the alarm / rollout counters it has consumed and the probe sampler's
identity. A resumed run continues the same gate: the next decision uses the next decision index
(fresh items), the restored ledger position and hold streak, and the original incumbent. The alarm
itself is checkpointed by whoever feeds it (a :class:`DisagreementScreen` is registered
automatically as an advantage-shaper component); a resumed alarm whose counters are behind the
callback's is refused (``ValueError``). The decision log is rewritten on resume to the records with
``decision_index`` below the loaded one (a torn last line is dropped), as in
:meth:`~research_rl.capability.loop.GatedLoop.load_state_dict`.

Repeated ``train()``: a further ``trainer.train()`` without ``resume_from_checkpoint`` on a trainer
whose callback already started **continues the same gate**: the incumbent (the last accepted
snapshot, never the weights the previous ``train()`` ended with), the hold streak, the ledger, the
decision index, the running block (rewards and latched alarm) and the consumed counters are kept;
only the optimizer handle is re-read (``ValueError`` if the trainable parameters no longer match
the incumbent). Weights left unaccepted by the previous ``train()`` (after a HOLD, or a partial
block) are therefore still a candidate judged against the incumbent at the next decision. HF
restarts ``state.global_step`` at 0 in every ``train()``, so block decisions fall on multiples of
``block_updates`` of the new run's steps and ``update_index`` / the logged ``global_step`` restart
too (``decision_index`` keeps counting). A fresh gate needs a new callback.

Limits: single process only (``ValueError`` for ``world_size > 1``: every rank would run its own
gate on its own alarm); no ``sync_ref_model`` with an enabled gate (see above). The retention-horizon check of ``GatedLoop`` is not run here (a run whose
retention slices are too small gets ``horizon_exhausted`` rollbacks, labelled by the controller).
Verified on CPU with tiny models only (``tests/research_rl/test_capability_trl_gate.py``); GPU,
vLLM, PEFT and DeepSpeed / FSDP runs are NOT_TESTED.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeGuard

import torch
from torch import nn
from transformers import TrainerCallback

from research_rl.capability.alarm import UNITS_KEY, AlarmState, DisagreementAlarm, screen_mask
from research_rl.capability.gate import CapabilityGate, GateConfig
from research_rl.capability.loop import GateController, snapshot_from_state, snapshot_to_state
from research_rl.capability.tasks import VerifierScoreFn
from research_rl.capability.types import (
    ChannelGroup,
    DecisionRecord,
    GateDecision,
    Item,
    OperatorReport,
    RollbackUnit,
    Split,
)
from research_rl.capability.weights import StateSnapshot, capture
from research_rl.optim.optimizer_wrapper import unwrap_hooked
from research_rl.rewards.verifier_interface import Verifier
from research_rl.trainers.rl_common import AdvantageShaper, ShapedAdvantages

__all__ = [
    "GATE_COMPONENT",
    "CapabilityGateCallback",
    "DisagreementScreen",
    "base_optimizer",
    "generate_every",
]

GATE_COMPONENT = "capability_gate"
"""Default ``trainer.research_components`` key of :class:`CapabilityGateCallback`."""

_CALLBACK_STATE_VERSION = 1
_SCREEN_STATE_VERSION = 1
_WEIGHT_CHANGING = (GateDecision.ROLLBACK, GateDecision.SHRINK, GateDecision.GROUP_ROLLBACK)
_COUNTERS = ("nominal_steps", "applied_updates", "skipped_steps")


def _is_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_count(name: str, value: Any) -> int:
    if not _is_int(value) or value < 0:
        raise ValueError(f"{name} must be an int >= 0, got {value!r}")
    return int(value)


# ============================================================================== trainer helpers
def base_optimizer(optimizer: Any) -> torch.optim.Optimizer | None:
    """The optimizer the gate snapshots and restores: the ``HookedOptimizer`` inside any wrapper
    (:func:`~research_rl.optim.optimizer_wrapper.unwrap_hooked`), else the innermost
    ``torch.optim.Optimizer`` reachable through ``.optimizer`` attributes (accelerate's
    ``AcceleratedOptimizer`` wraps the real one), else ``None`` for ``None``."""
    if optimizer is None:
        return None
    hooked = unwrap_hooked(optimizer)
    if hooked is not None:
        return hooked
    current = optimizer
    for _ in range(8):
        inner = getattr(current, "optimizer", None)
        if not isinstance(inner, torch.optim.Optimizer):
            break
        current = inner
    if not isinstance(current, torch.optim.Optimizer):
        raise TypeError(f"expected a torch.optim.Optimizer, got {type(optimizer).__name__}")
    return current


def generate_every(args: Any, trainer: Any) -> int:
    """TRL's generation period in micro-batches: ``steps_per_generation * num_iterations``
    (``steps_per_generation`` defaults to ``gradient_accumulation_steps`` as in ``GRPOConfig``;
    ``num_iterations`` to 1)."""
    ga = int(args.gradient_accumulation_steps)
    spg = int(getattr(args, "steps_per_generation", None) or ga)
    iterations = int(getattr(trainer, "num_iterations", None) or getattr(args, "num_iterations", 1) or 1)
    return spg * iterations


def _world_size(args: Any) -> int:
    try:
        return int(args.world_size)
    except (AttributeError, TypeError, ValueError):
        return 1


class _TrainerOperator:
    """The :class:`~research_rl.capability.types.UpdateOperator` view of a TRL trainer for the
    controller: TRL makes the updates, so :meth:`step` is never called; the controller only reads
    ``optimizer`` / ``parameter_names`` and reports the decision."""

    name = "trl"

    def __init__(self, names: Sequence[str], optimizer: torch.optim.Optimizer | None) -> None:
        self._names = list(names)
        self.optimizer = optimizer
        self.last_decision: GateDecision | None = None

    def parameter_names(self) -> list[str]:
        return list(self._names)

    def step(self, model: nn.Module, update_index: int) -> OperatorReport:
        raise RuntimeError("the TRL trainer performs the updates; the gate callback never steps")

    def on_decision(self, decision: GateDecision) -> None:
        self.last_decision = decision

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        return None


def _trainable_names(model: nn.Module) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


# ============================================================================== the callback
class CapabilityGateCallback(TrainerCallback):
    """Run the capability gate on a TRL trainer (see the module docstring for the exact schedule,
    what a decision writes and how state travels with checkpoints).

    Args:
        probe: the :class:`~research_rl.capability.probe.CapabilityProbe` (anything with
            ``evidence(model, incumbent, decision_index, *, attempt, alarm)``; ``gate_items`` and
            ``score_fn.tokenizer`` are needed only for the group-rollback stage).
        gate: the :class:`~research_rl.capability.gate.CapabilityGate` (its ledger is spent by the
            verdicts; with ``config.enabled=False`` the controller uses a private ledger copy).
        block_updates: optimizer steps per block decision. **Authoritative** for the callback (used
            even when ``gate.config.block_updates`` differs, as ``rollback_unit`` is for the
            controller); it is stored and checked on resume.
        rollback_unit: the finest revert unit, authoritative for the controller.
        channel_groups: groups for the group-rollback stage (``RollbackUnit.CHANNELS`` only).
        log_path: optional JSONL decision log: one ``DecisionRecord.to_dict()`` per decision plus
            ``global_step``, ``trigger`` (``"block"`` / ``"alarm"``) and ``buffer_invalidated``;
            appended and flushed after every decision.
        alarm: optional :class:`~research_rl.capability.alarm.DisagreementAlarm` (normally the
            one a :class:`DisagreementScreen` feeds). The callback only reads its counters.

    Call :meth:`bind` with the trainer before ``train()`` (``RuntimeError`` in ``on_train_begin``
    otherwise). After training, :attr:`records` holds this process's decision records,
    :attr:`incumbent` the last accepted snapshot.
    """

    def __init__(
        self,
        probe: Any,
        gate: CapabilityGate,
        *,
        block_updates: int,
        rollback_unit: RollbackUnit,
        channel_groups: Sequence[ChannelGroup] | None = None,
        log_path: str | Path | None = None,
        alarm: DisagreementAlarm | None = None,
    ) -> None:
        if not isinstance(gate, CapabilityGate):
            raise TypeError(f"gate must be a CapabilityGate, got {type(gate).__name__}")
        if not _is_int(block_updates) or block_updates < 1:
            raise ValueError(f"block_updates must be an int >= 1, got {block_updates!r}")
        if alarm is not None and not isinstance(alarm, DisagreementAlarm):
            raise TypeError(f"alarm must be a DisagreementAlarm or None, got {type(alarm).__name__}")
        self.controller = GateController(
            gate, probe, rollback_unit=rollback_unit, channel_groups=channel_groups
        )
        self.gate = gate
        self.probe = probe
        self.block_updates = int(block_updates)
        self.alarm = alarm
        self.log_path = Path(log_path) if log_path is not None else None
        self.trainer: Any = None
        self.incumbent: StateSnapshot | None = None
        self.decision_index = 0
        self.records: list[DecisionRecord] = []
        self.buffer_invalidations = 0
        self.last_alarm_count = 0
        self._batches_seen = 0
        self._screen_seen = 0
        self._operator: _TrainerOperator | None = None
        self._pending: dict[str, Any] | None = None
        self._reset_block()

    @property
    def config(self) -> GateConfig:
        return self.gate.config

    def _reset_block(self) -> None:
        self._block_v1: list[float] = []
        self._block_v2: list[float] = []
        self._block_alarm = False

    # ---------------------------------------------------------------- wiring
    def bind(self, trainer: Any, *, name: str = GATE_COMPONENT) -> CapabilityGateCallback:
        """Attach to ``trainer``: keep a reference (for ``_step`` / ``_buffered_inputs`` /
        ``trajectory_buffer`` / ``advantage_shapers``), register ``self`` as
        ``trainer.research_components[name]`` (``ValueError`` if another object holds that key) and
        add ``self`` to the trainer's callbacks unless it is already there. Returns ``self``."""
        components = getattr(trainer, "research_components", None)
        if not isinstance(components, dict):
            raise TypeError("trainer has no research_components dict (use a ResearchRL trainer)")
        existing = components.get(name)
        if existing is not None and existing is not self:
            raise ValueError(f"research component {name!r} is already registered to another object")
        components[name] = self
        handler = getattr(trainer, "callback_handler", None)
        if handler is not None and not any(cb is self for cb in handler.callbacks):
            trainer.add_callback(self)
        self.trainer = trainer
        return self

    def _screen(self) -> DisagreementScreen | None:
        shapers = getattr(self.trainer, "advantage_shapers", None) or ()
        for shaper in shapers:
            if isinstance(shaper, DisagreementScreen):
                return shaper
        return None

    def _alarm_count(self) -> int:
        if self.alarm is None:
            return 0
        return int(self.alarm.gap_alarms + self.alarm.length_alarms)

    def _rollout_total(self) -> int:
        buffer = getattr(self.trainer, "trajectory_buffer", None)
        return 0 if buffer is None else int(buffer.total_batches)

    # ---------------------------------------------------------------- events
    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Check the setup, then start the gate: apply the state of a resumed checkpoint (after
        :meth:`load_state_dict`); else, on a further ``train()`` of the same callback, continue the
        gate as it is (see the module docstring, "Repeated train()"); else capture the incumbent.

        ``RuntimeError`` without :meth:`bind`; ``ValueError`` for ``world_size > 1``,
        ``sync_ref_model=True`` with an enabled gate, a ``block_updates`` whose micro-batch count is not a multiple of the generation period, a
        model without trainable parameters, a missing optimizer, a continued gate whose incumbent
        does not cover exactly the trainable parameters, or a resumed state that does not fit (see
        :meth:`load_state_dict`)."""
        trainer = self.trainer
        if trainer is None:
            raise RuntimeError("CapabilityGateCallback.bind(trainer) must be called before train()")
        if _world_size(args) > 1:
            raise ValueError(
                "CapabilityGateCallback supports single-process training only (NOT_TESTED otherwise)"
            )
        if self.config.enabled and getattr(args, "sync_ref_model", False):
            raise ValueError(
                "sync_ref_model=True is not supported with an enabled CapabilityGateCallback: TRL "
                "mixes the policy into the reference model (the KL anchor) at its sync steps, and "
                "a ROLLBACK / SHRINK / GROUP_ROLLBACK does not undo that, so weights the gate "
                "rejected would stay in ref_model"
            )
        period = generate_every(args, trainer)
        micro = self.block_updates * int(args.gradient_accumulation_steps)
        if micro % period:
            raise ValueError(
                f"block_updates={self.block_updates} x gradient_accumulation_steps="
                f"{args.gradient_accumulation_steps} = {micro} micro-batches is not a multiple of "
                f"steps_per_generation x num_iterations = {period}: block decisions would land inside "
                "TRL's buffered generation batch"
            )
        model = trainer.model
        optimizer = base_optimizer(getattr(trainer, "optimizer", None) or kwargs.get("optimizer"))
        if optimizer is None:
            raise ValueError("the trainer has no optimizer at on_train_begin")
        names = _trainable_names(model)
        if not names:
            raise ValueError("the model has no trainable parameters")
        if self._pending is not None:
            self._apply_pending(model, names)
            self._operator = _TrainerOperator(names, optimizer)
            return
        if self.incumbent is not None:  # a further train() on the same trainer: continue the gate
            if set(self.incumbent.params) != set(names):
                raise ValueError(
                    "the model's trainable parameters changed since the gate's incumbent was captured"
                )
            self._operator = _TrainerOperator(names, optimizer)
            return
        self._operator = _TrainerOperator(names, optimizer)
        self.incumbent = capture(model, names, optimizer=optimizer)
        self.controller.holds = 0
        self._reset_block()
        self.last_alarm_count = self._alarm_count()
        self._batches_seen = self._rollout_total()
        screen = self._screen()
        self._screen_seen = 0 if screen is None else screen.train_batches

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Collect the block's rewards and alarm, then decide at block boundaries / on alarms."""
        if self.incumbent is None or self._operator is None:
            raise RuntimeError("on_step_end before on_train_begin")
        self._collect()
        count = self._alarm_count()
        fired = count > self.last_alarm_count
        self.last_alarm_count = count
        self._block_alarm = self._block_alarm or fired
        step = int(state.global_step)
        if step % self.block_updates == 0:
            self._decide(args, step, "block")
        elif fired and self.config.alarm_triggers_decision:
            self._decide(args, step, "alarm")

    def _collect(self) -> None:
        buffer = getattr(self.trainer, "trajectory_buffer", None)
        if buffer is not None:
            total = int(buffer.total_batches)
            new = total - self._batches_seen
            if new > 0:
                batches = list(buffer)[-min(new, len(buffer)) :] if len(buffer) else []
                self._block_v1.extend(float(b.mean_reward) for b in batches)
            self._batches_seen = total
        screen = self._screen()
        if screen is not None:
            new = screen.train_batches - self._screen_seen
            if new > 0:
                self._block_v2.extend(v2 for _, v2 in screen.recent(new))
            self._screen_seen = screen.train_batches

    def _decide(self, args: Any, step: int, trigger: str) -> DecisionRecord:
        trainer = self.trainer
        operator = self._operator
        incumbent = self.incumbent
        assert operator is not None and incumbent is not None
        model = trainer.model
        optimizer = operator.optimizer
        hooked = unwrap_hooked(optimizer)
        counters = {k: getattr(hooked, k) for k in _COUNTERS} if hooked is not None else None
        hyper = (
            [{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups]
            if optimizer is not None
            else None
        )
        finite_v1 = [v for v in self._block_v1 if math.isfinite(v)]
        reward_v1 = math.fsum(finite_v1) / len(finite_v1) if finite_v1 else None
        reward_v2 = math.fsum(self._block_v2) / len(self._block_v2) if self._block_v2 else None
        try:
            record, new_incumbent = self.controller.run(
                model,
                operator,  # type: ignore[arg-type]  # structural UpdateOperator
                incumbent,
                self.decision_index,
                max(step - 1, 0),
                self._block_alarm,
                reward_v1,
                reward_v2,
            )
        finally:  # also when the procedure raised after writing: counters / schedule stay the HF ones
            if hooked is not None and counters is not None:
                for key, value in counters.items():
                    setattr(hooked, key, value)
            if optimizer is not None and hyper is not None:
                for group, saved in zip(optimizer.param_groups, hyper):
                    group.update(saved)
        invalidated = False
        if record.decision in _WEIGHT_CHANGING:
            micro = getattr(trainer, "_step", None)
            aligned = _is_int(micro) and micro % generate_every(args, trainer) == 0
            if not aligned and hasattr(trainer, "_buffered_inputs"):
                trainer._buffered_inputs = None
                self.buffer_invalidations += 1
                invalidated = True
        self.incumbent = new_incumbent
        self.decision_index += 1
        self._reset_block()
        self.records.append(record)
        self._write_log(record, step, trigger, invalidated)
        return record

    def _write_log(self, record: DecisionRecord, step: int, trigger: str, invalidated: bool) -> None:
        if self.log_path is None:
            return
        row = {**record.to_dict(), "global_step": step, "trigger": trigger, "buffer_invalidated": invalidated}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()

    # ---------------------------------------------------------------- state
    def state_dict(self) -> dict[str, Any]:
        """Checkpoint state (loadable with ``torch.load(weights_only=True)``): ``version``,
        ``config`` (``GateConfig.to_dict``), ``block_updates``, ``rollback_unit``,
        ``decision_index`` (next decision), ``incumbent`` (:func:`~research_rl.capability.loop.snapshot_to_state`
        or None before training), ``controller`` (hold streak, verdict gate, ledger), ``block``
        (TRL batch rewards / screen V2 rates of the running block and the latched alarm),
        ``alarm_count`` / ``batches_seen`` / ``screen_seen`` (counters consumed so far),
        ``buffer_invalidations`` and ``sampler`` (the probe sampler's identity, or None)."""
        sampler = getattr(self.probe, "sampler", None)
        return {
            "version": _CALLBACK_STATE_VERSION,
            "config": self.config.to_dict(),
            "block_updates": self.block_updates,
            "rollback_unit": self.controller.rollback_unit.value,
            "decision_index": self.decision_index,
            "incumbent": None if self.incumbent is None else snapshot_to_state(self.incumbent),
            "controller": self.controller.state_dict(),
            "block": {
                "reward_v1": list(self._block_v1),
                "reward_v2": list(self._block_v2),
                "alarm": self._block_alarm,
            },
            "alarm_count": self.last_alarm_count,
            "batches_seen": self._batches_seen,
            "screen_seen": self._screen_seen,
            "buffer_invalidations": self.buffer_invalidations,
            "sampler": None if sampler is None else sampler.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate ``state`` and stash it; it is applied in the next ``on_train_begin`` (after HF
        loaded the checkpoint's weights and optimizer state).

        Checked here (``ValueError``, nothing changes): version, GateConfig, ``block_updates``,
        rollback unit, counters, the running block, the sampler identity and the incumbent snapshot
        (fingerprint re-verified). Checked when applied: the incumbent covers exactly the model's
        trainable parameters, the controller state fits (atomic), and the alarm / rollout / screen
        counters are not behind the consumed ones (their own state must have been restored)."""
        if state.get("version") != _CALLBACK_STATE_VERSION:
            raise ValueError(f"unsupported gate callback state version {state.get('version')!r}")
        if GateConfig.from_dict(state["config"]) != self.config:
            raise ValueError("gate callback state was saved with a different GateConfig")
        if state.get("block_updates") != self.block_updates:
            raise ValueError(
                f"block_updates mismatch: {state.get('block_updates')!r} != {self.block_updates}"
            )
        if state.get("rollback_unit") != self.controller.rollback_unit.value:
            raise ValueError(f"rollback_unit mismatch: {state.get('rollback_unit')!r}")
        for key in ("decision_index", "alarm_count", "batches_seen", "screen_seen", "buffer_invalidations"):
            _check_count(key, state.get(key))
        block = state["block"]
        v1 = [float(v) for v in block["reward_v1"]]
        v2 = [float(v) for v in block["reward_v2"]]
        if not isinstance(block["alarm"], bool):
            raise ValueError(f"invalid block alarm flag {block['alarm']!r}")
        sampler = getattr(self.probe, "sampler", None)
        if sampler is not None and state.get("sampler") is not None:
            sampler.load_state_dict(state["sampler"])
        incumbent = None if state.get("incumbent") is None else snapshot_from_state(state["incumbent"])
        self._pending = {
            "decision_index": int(state["decision_index"]),
            "incumbent": incumbent,
            "controller": state["controller"],
            "block": (v1, v2, block["alarm"]),
            "alarm_count": int(state["alarm_count"]),
            "batches_seen": int(state["batches_seen"]),
            "screen_seen": int(state["screen_seen"]),
            "buffer_invalidations": int(state["buffer_invalidations"]),
        }

    def _apply_pending(self, model: nn.Module, names: Sequence[str]) -> None:
        pending = self._pending
        assert pending is not None
        incumbent: StateSnapshot | None = pending["incumbent"]
        if incumbent is None:
            raise ValueError("the resumed gate state has no incumbent (saved before training began)")
        if set(incumbent.params) != set(names):
            raise ValueError("the resumed incumbent does not cover exactly the model's trainable parameters")
        if self._alarm_count() < pending["alarm_count"]:
            raise ValueError(
                "the alarm's counters are behind the gate state: restore the alarm's state too "
                "(e.g. through a DisagreementScreen registered as an advantage shaper)"
            )
        if self._rollout_total() < pending["batches_seen"]:
            raise ValueError("the trajectory buffer's counters are behind the gate state")
        screen = self._screen()
        if screen is not None and screen.train_batches < pending["screen_seen"]:
            raise ValueError("the DisagreementScreen's counters are behind the gate state")
        kept = self._read_log(pending["decision_index"])
        self.controller.load_state_dict(pending["controller"])
        self.incumbent = incumbent
        self.decision_index = pending["decision_index"]
        v1, v2, alarm = pending["block"]
        self._block_v1, self._block_v2, self._block_alarm = list(v1), list(v2), alarm
        self.last_alarm_count = pending["alarm_count"]
        self._batches_seen = pending["batches_seen"]
        self._screen_seen = pending["screen_seen"]
        self.buffer_invalidations = pending["buffer_invalidations"]
        self._pending = None
        if kept is not None:
            self._rewrite_log(kept)

    def _read_log(self, decision_index: int) -> list[str] | None:
        """Log lines to keep on resume (``decision_index`` below the loaded one); a torn last line
        is dropped, any other unparsable line raises ``ValueError``. ``None`` without a log."""
        if self.log_path is None or not self.log_path.exists():
            return None
        lines = [ln for ln in self.log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        kept: list[str] = []
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if i == len(lines) - 1:
                    break
                raise ValueError(f"decision log {self.log_path} line {i + 1} is not valid JSON") from exc
            index = row.get("decision_index") if isinstance(row, dict) else None
            if not _is_int(index):
                raise ValueError(f"decision log {self.log_path} line {i + 1} has no int 'decision_index'")
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


# ============================================================================== the screen
AnswerSource = Mapping[str, str] | Callable[[str], str]


class DisagreementScreen(AdvantageShaper):
    """Screen V1-pass / V2-fail rollouts out of the GRPO advantage and feed the tier-1 alarm.

    For every **training** generation batch (``output`` of TRL's ``_generate_and_score_completions``
    while ``trainer.model.training``):

    1. **Decode.** Row ``i``'s prompt text is ``tokenizer.decode`` of its prompt ids where
       ``prompt_mask`` is 1 (``skip_special_tokens=True``). Its completion is the tokens before the
       first EOS (``trainer.eos_token_ids``, else ``tokenizer.eos_token_id``) inside
       ``completion_mask``, decoded with ``skip_special_tokens=True``; a row with no EOS uses all of
       its masked tokens. With ``trainer.mask_truncated_completions`` TRL zeroes the whole mask of
       every row whose last token is neither EOS nor pad (a budget-filling completion); such an
       all-zero row is recovered as its ``completion_ids`` with the trailing pad ids
       (``trainer._tokenizer.pad_token_id``, else ``tokenizer.pad_token_id``) stripped, which is
       exactly the generated row, so the screen sees the same completions, flags, scores and
       lengths whether the flag is on or off. Without the flag an all-zero mask row is an empty
       completion.
    2. **Truncation** (the gate's rule, as in :mod:`research_rl.capability.operators`): a row is
       truncated iff it has no EOS **or** :meth:`VerifierScoreFn.truncated` (``max_new_tokens``)
       flags its text; both verifiers receive the same flags through
       :meth:`VerifierScoreFn.score_completions` (never raw ``verify()``), so a truncated row scores
       0 under V1 and V2.
    3. **Reference answers**: ``answers`` maps the decoded prompt text to the reference (a mapping
       or a callable); default (``None``): built once from ``trainer.train_dataset`` columns
       ``"prompt"`` and ``answer_column`` (string prompts only; conflicting duplicates raise
       ``ValueError``). A prompt without an answer raises ``KeyError`` (it is never scored as a
       failure).
    4. **Screen.** :func:`~research_rl.capability.alarm.screen_mask`: rows that pass V1 and fail V2
       get advantage 0 (every entry of their row for ``(B, T)`` advantages); other rows keep the
       incoming advantage. ``signal`` = TRL's reward-based flags
       (``trainer.trl_advantage_signal``) when this screen is the first shaper, else "incoming
       advantage row exactly non-zero and completion mask non-empty", AND the keep mask.
    5. **Alarm**: one :class:`~research_rl.capability.types.OperatorReport` per
       generation batch with ``update_index = trainer.state.global_step``, V1 / V2 mean rewards,
       pass flags, completion lengths (tokens before EOS; all generated tokens for rows without
       EOS, i.e. the budget for budget-filling rows, also when ``mask_truncated_completions``
       zeroed their mask, as :mod:`research_rl.capability.operators` reports them),
       ``screened`` and ``extra["alarm_units"]`` = the prompt group of each row
       (``(process_index * B + i) // trainer.num_generations``). Rows of one group must share their
       prompt ids (``ValueError`` otherwise).

    **Eval batches** (``trainer.model.training`` False) are returned untouched (the incoming
    advantages and the step-4 base signal): nothing is decoded, looked up or scored, and the alarm
    and counters are not changed (eval advantages never reach a backward pass, and eval prompts
    need not be in the answer table).

    State (``state_dict`` / ``load_state_dict``, restored automatically as a research component):
    the alarm state, counters (``train_batches``, ``rows``, ``screened``) and the recent per-batch
    ``(reward_v1, reward_v2)`` means read by :class:`CapabilityGateCallback`. Single process only
    is tested (NOT_TESTED otherwise).
    """

    name = "disagreement_screen"
    RECENT = 256

    def __init__(
        self,
        v1: Verifier,
        v2: Verifier,
        alarm: DisagreementAlarm | None,
        tokenizer: Any,
        *,
        max_new_tokens: int,
        answers: AnswerSource | None = None,
        answer_column: str = "answer",
    ) -> None:
        for label, verifier in (("v1", v1), ("v2", v2)):
            if not callable(getattr(verifier, "verify", None)):
                raise TypeError(f"{label} must be a verifier with verify(), got {type(verifier).__name__}")
        if alarm is not None and not isinstance(alarm, DisagreementAlarm):
            raise TypeError(f"alarm must be a DisagreementAlarm or None, got {type(alarm).__name__}")
        if not _is_int(max_new_tokens) or max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be an int >= 1, got {max_new_tokens!r}")
        if answers is not None and not (isinstance(answers, Mapping) or callable(answers)):
            raise TypeError("answers must be a mapping prompt -> answer, a callable or None")
        self.v1 = v1
        self.v2 = v2
        self.alarm = alarm
        self.tokenizer = tokenizer
        self.max_new_tokens = int(max_new_tokens)
        self.v1_score = VerifierScoreFn(tokenizer, v1, max_new_tokens=self.max_new_tokens, batch_size=1)
        self.v2_score = VerifierScoreFn(tokenizer, v2, max_new_tokens=self.max_new_tokens, batch_size=1)
        self.answers = answers
        self.answer_column = str(answer_column)
        self.train_batches = 0
        self.rows = 0
        self.screened = 0
        self.last_state: AlarmState | None = None
        self.last_keep: list[bool] = []
        self._recent: deque[tuple[float, float]] = deque(maxlen=self.RECENT)

    # ---------------------------------------------------------------- helpers
    def recent(self, n: int) -> list[tuple[float, float]]:
        """The ``(reward_v1, reward_v2)`` means of the last ``n`` training batches (at most
        :attr:`RECENT`), oldest first."""
        if n <= 0:
            return []
        items = list(self._recent)
        return items[-min(n, len(items)) :]

    def _answer_source(self, trainer: Any) -> AnswerSource:
        if self.answers is not None:
            return self.answers
        dataset = getattr(trainer, "train_dataset", None)
        columns = getattr(dataset, "column_names", None)
        if dataset is None or columns is None or "prompt" not in columns or self.answer_column not in columns:
            raise ValueError(
                f"DisagreementScreen needs answers=... or a train_dataset with 'prompt' and "
                f"{self.answer_column!r} columns"
            )
        table: dict[str, str] = {}
        for prompt, answer in zip(dataset["prompt"], dataset[self.answer_column]):
            if not isinstance(prompt, str):
                raise TypeError("the default answer table needs string prompts; pass answers=...")
            answer = str(answer)
            if table.get(prompt, answer) != answer:
                raise ValueError(f"prompt {prompt!r} has two different answers in the dataset")
            table[prompt] = answer
        self.answers = table
        return table

    @staticmethod
    def _lookup(source: AnswerSource, prompt: str) -> str:
        if isinstance(source, Mapping):
            if prompt not in source:
                raise KeyError(f"no reference answer for prompt {prompt!r}")
            return str(source[prompt])
        return str(source(prompt))

    def _eos_ids(self, trainer: Any) -> set[int]:
        ids = getattr(trainer, "eos_token_ids", None)
        if ids:
            return {int(i) for i in ids}
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is None:
            raise ValueError("no EOS token id (trainer.eos_token_ids / tokenizer.eos_token_id)")
        return {int(eos)}

    def _pad_id(self, trainer: Any) -> int:
        """The id TRL right-padded ``completion_ids`` with (``trainer._tokenizer``, else
        ``self.tokenizer``); ``ValueError`` if neither has one."""
        for tok in (getattr(trainer, "_tokenizer", None), self.tokenizer):
            pad = getattr(tok, "pad_token_id", None)
            if pad is not None:
                return int(pad)
        raise ValueError("mask_truncated_completions needs a pad token id to recover truncated rows")

    @staticmethod
    def _unmasked_row(ids: torch.Tensor, pad: int) -> list[int]:
        """A row whose completion mask TRL zeroed (``mask_truncated_completions``): its ids with the
        trailing ``pad`` ids stripped. Exact, because TRL zeroes a row only when its last generated
        token is neither EOS nor pad, so every trailing pad is padding."""
        row = [int(t) for t in ids.tolist()]
        end = len(row)
        while end and row[end - 1] == pad:
            end -= 1
        return row[:end]

    # ---------------------------------------------------------------- the shaper
    def __call__(self, output: Mapping[str, Any], trainer: Any) -> ShapedAdvantages:
        advantages: torch.Tensor = output["advantages"]
        completion_mask = output["completion_mask"].detach().cpu().bool()
        n = int(advantages.shape[0])
        shapers = getattr(trainer, "advantage_shapers", None) or ()
        base = getattr(trainer, "trl_advantage_signal", None)
        if not (shapers and shapers[0] is self and isinstance(base, torch.Tensor)):
            nonzero = (advantages != 0).reshape(n, -1).any(dim=1)
            base = nonzero & completion_mask.reshape(n, -1).any(dim=1).to(nonzero.device)
        base = base.detach().to(advantages.device).bool()
        if not bool(trainer.model.training):  # eval batches: untouched (they never reach a backward)
            return ShapedAdvantages(advantages=advantages.detach().clone(), signal=base)
        completion_ids = output["completion_ids"].detach().cpu()
        prompt_ids = output["prompt_ids"].detach().cpu()
        prompt_mask = output["prompt_mask"].detach().cpu().bool()
        eos = self._eos_ids(trainer)
        masks_truncated = bool(getattr(trainer, "mask_truncated_completions", False))
        pad = self._pad_id(trainer) if masks_truncated else None
        source = self._answer_source(trainer)
        prompts: list[str] = []
        prompt_rows: list[tuple[int, ...]] = []
        texts: list[str] = []
        no_eos: list[bool] = []
        lengths: list[int] = []
        for i in range(n):
            p_ids = tuple(int(t) for t in prompt_ids[i][prompt_mask[i]].tolist())
            prompt_rows.append(p_ids)
            prompts.append(self.tokenizer.decode(list(p_ids), skip_special_tokens=True))
            if pad is not None and not bool(completion_mask[i].any()):
                row = self._unmasked_row(completion_ids[i], pad)
            else:
                row = [int(t) for t in completion_ids[i][completion_mask[i]].tolist()]
            cut = next((j for j, t in enumerate(row) if t in eos), None)
            body = row if cut is None else row[:cut]
            no_eos.append(cut is None)
            lengths.append(len(body))
            texts.append(self.tokenizer.decode(body, skip_special_tokens=True))
        truncated = [a or b for a, b in zip(no_eos, self.v2_score.truncated(texts))]
        items = [
            Item(
                uid=f"trl/{i}",
                split=Split.TRAIN,
                family="trl",
                slice="trl/train",
                rule=(),
                prompt=prompts[i],
                answer=self._lookup(source, prompts[i]),
            )
            for i in range(n)
        ]
        v1 = self.v1_score.score_completions(texts, items, truncated)
        v2 = self.v2_score.score_completions(texts, items, truncated)
        v1_pass = [s == 1.0 for s in v1]
        v2_pass = [s == 1.0 for s in v2]
        keep = screen_mask(v1_pass, v2_pass)
        keep_t = torch.tensor(keep, dtype=torch.bool, device=advantages.device)
        shaped = advantages.detach().clone()
        shaped[~keep_t] = 0
        signal = base & keep_t
        screened = sum(1 for k in keep if not k)
        self.last_keep = list(keep)
        group = int(trainer.num_generations)
        offset = int(getattr(getattr(trainer, "accelerator", None), "process_index", 0) or 0) * n
        units = [(offset + i) // group for i in range(n)]
        first: dict[int, tuple[int, ...]] = {}
        for unit, ids in zip(units, prompt_rows):
            if first.setdefault(unit, ids) != ids:
                raise ValueError(
                    f"rows of prompt group {unit} have different prompts; the alarm units assume "
                    f"num_generations={group} consecutive rows per prompt"
                )
        reward_v1 = math.fsum(v1) / n if n else 0.0
        reward_v2 = math.fsum(v2) / n if n else 0.0
        report = OperatorReport(
            update_index=int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0),
            reward_v1=reward_v1,
            reward_v2=reward_v2,
            v1_pass=tuple(v1_pass),
            v2_pass=tuple(v2_pass),
            completion_lengths=tuple(lengths),
            screened=screened,
            extra={UNITS_KEY: units, "truncated": sum(truncated)},
        )
        if self.alarm is not None:
            self.last_state = self.alarm.update(report)
        self.train_batches += 1
        self.rows += n
        self.screened += screened
        self._recent.append((reward_v1, reward_v2))
        return ShapedAdvantages(advantages=shaped, signal=signal)

    # ---------------------------------------------------------------- state
    def state_dict(self) -> dict[str, Any]:
        """``version``, ``alarm`` (its ``state_dict`` or None), ``train_batches``, ``rows``,
        ``screened`` and ``recent`` (list of ``[reward_v1, reward_v2]``)."""
        return {
            "version": _SCREEN_STATE_VERSION,
            "alarm": None if self.alarm is None else self.alarm.state_dict(),
            "train_batches": self.train_batches,
            "rows": self.rows,
            "screened": self.screened,
            "recent": [[a, b] for a, b in self._recent],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a :meth:`state_dict` (``ValueError`` on a version / alarm presence mismatch or
        invalid counters; the alarm's own ``load_state_dict`` checks its config)."""
        if state.get("version") != _SCREEN_STATE_VERSION:
            raise ValueError(f"unsupported screen state version {state.get('version')!r}")
        if (state.get("alarm") is None) != (self.alarm is None):
            raise ValueError("screen state and screen disagree on having an alarm")
        counts = {k: _check_count(k, state.get(k)) for k in ("train_batches", "rows", "screened")}
        recent = [(float(a), float(b)) for a, b in state["recent"]]
        if self.alarm is not None:
            self.alarm.load_state_dict(state["alarm"])
        self.train_batches = counts["train_batches"]
        self.rows = counts["rows"]
        self.screened = counts["screened"]
        self._recent = deque(recent, maxlen=self.RECENT)
