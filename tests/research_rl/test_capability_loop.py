"""Tests of research_rl.capability.loop: the gate procedure (GateController) and GatedLoop.

CPU only, tiny Llama built in-process. The decision-path tests use a *scripted* score function
(each gate evaluation's outcome is fixed in advance: candidate-only wins on transfer items and/or
incumbent-only wins on retention items) so that every path is forced deterministically; the
weights are moved by a plain LM-loss step operator. Bit-exactness is asserted only for copies,
swaps and restores (rollback, promotion captures, untouched optimizer state) and, in the two
resume tests (with and without dropout), for two trajectories that run identical ops on identical
inputs in one process (explained in test_resume_equals_uninterrupted_run); interpolated values
are compared with allclose.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from research_rl.capability.alarm import DisagreementAlarm
from research_rl.capability.es import ESConfig, ESOperator, FitnessResult
from research_rl.capability.gate import CapabilityGate, GateConfig
from research_rl.capability.loop import (
    OVERRIDE_GATE_DISABLED,
    REASON_ACCEPT,
    REASON_ALARM_HOLD,
    REASON_CONFIRMED_ALARM,
    REASON_HOLD_LIMIT,
    REASON_HORIZON_EXHAUSTED,
    REASON_NO_TRANSFER_GAIN,
    REASON_RETENTION_FAILED,
    GateController,
    GatedLoop,
    classify_verdict,
    retention_unpassable,
    snapshot_from_state,
    snapshot_to_state,
    worst_case_verdicts,
)
from research_rl.capability.operators import TorchGRPOOperator
from research_rl.capability.probe import CapabilityProbe
from research_rl.capability.scores import collect_channel_stats, group_channels, random_scores
from research_rl.capability.spaces import ChannelGainSpace
from research_rl.capability.stats import AlphaLedger
from research_rl.capability.tasks import (
    CompositionSplits,
    ExactAnswerVerifier,
    ItemSampler,
    OperatorSet,
    VerifierScoreFn,
)
from research_rl.capability.types import (
    ChannelGroup,
    GateDecision,
    Item,
    OperatorReport,
    RollbackUnit,
    Split,
)
from research_rl.capability.weights import StateSnapshot, capture, channel_groups_by_damage
from research_rl.models.ffn_channels import ChannelSelection, channel_mask
from research_rl.models.tiny import build_char_tokenizer, build_tiny_llama
from research_rl.optim.optimizer_wrapper import HookedOptimizer
from research_rl.rewards.verifier_interface import BaseVerifier, Status

TRANSFER = ("t_rule/unseen_pairs",)
RETENTION = ("retention/atomic", "retention/copy")


# ============================================================================== fixtures / helpers
@pytest.fixture(scope="module")
def splits() -> CompositionSplits:
    return CompositionSplits.build(OperatorSet(seed=0, num_operators=10), seed=0)


@pytest.fixture(scope="module")
def sampler(splits: CompositionSplits) -> ItemSampler:
    return ItemSampler(splits, root_seed=5)


@pytest.fixture(scope="module")
def tok() -> Any:
    return build_char_tokenizer()


# Scripted outcomes: (transfer gain, retention ok).
OUTCOMES = {"pass": (True, True), "hold": (False, True), "fail": (False, False), "fail_gain": (True, False)}


class ScriptedScore:
    """Score function whose k-th evidence (two calls: candidate first, then the incumbent inside
    ``swapped``) follows ``plan[k]``: transfer items 0 for the incumbent and 1/0 for the candidate
    (gain or not); retention items 1 for the incumbent and 1/0 for the candidate (ok or not)."""

    def __init__(self, plan: Sequence[str]) -> None:
        self.plan = list(plan)
        self.calls = 0
        self.uids: list[list[str]] = []

    @property
    def evaluations(self) -> int:
        return self.calls // 2

    def __call__(self, model: nn.Module, items: Sequence[Item]) -> list[float]:
        k, role = divmod(self.calls, 2)
        self.calls += 1
        gain, ok = OUTCOMES[self.plan[k]]
        if role == 0:
            self.uids.append([it.uid for it in items])
        out = []
        for it in items:
            if it.split is Split.RETENTION:
                out.append(1.0 if role == 1 or ok else 0.0)
            else:
                out.append(1.0 if role == 0 and gain else 0.0)
        return out


def vocab(tok: Any) -> int:
    return len(tok)


class LMStepOperator:
    """UpdateOperator doing one LM-loss step on a fixed random batch per update index.

    ``feed(update_index) -> (v1_pass, v2_pass)`` supplies the rollouts reported to the alarm."""

    name = "lm_step"

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        vocab_size: int,
        feed: Callable[[int], tuple[list[bool], list[bool]]] | None = None,
    ) -> None:
        self.model = model
        self.optimizer: torch.optim.Optimizer | None = optimizer
        self.vocab_size = vocab_size
        self.feed = feed or (lambda t: ([False] * 32, [False] * 32))
        ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        self._names = [n for n, p in model.named_parameters() if id(p) in ids]
        self.decisions: list[GateDecision] = []
        self.steps = 0

    def parameter_names(self) -> list[str]:
        return list(self._names)

    def step(self, model: nn.Module, update_index: int) -> OperatorReport:
        g = torch.Generator().manual_seed(1000 + update_index)
        ids = torch.randint(3, self.vocab_size, (4, 10), generator=g)
        assert self.optimizer is not None
        self.optimizer.zero_grad(set_to_none=True)
        model(input_ids=ids, labels=ids).loss.backward()
        self.optimizer.step()
        self.steps += 1
        v1, v2 = self.feed(update_index)
        n = len(v1)
        return OperatorReport(
            update_index=update_index,
            reward_v1=sum(v1) / n if n else 0.0,
            reward_v2=sum(v2) / n if n else 0.0,
            v1_pass=tuple(v1),
            v2_pass=tuple(v2),
            completion_lengths=(5,) * n,
            forward_passes=1,
            backward_passes=1,
        )

    def on_decision(self, decision: GateDecision) -> None:
        self.decisions.append(decision)

    def state_dict(self) -> dict[str, Any]:
        return {"steps": self.steps}

    def load_state_dict(self, state: Any) -> None:
        self.steps = int(state["steps"])


def base_config(**overrides: Any) -> GateConfig:
    kwargs: dict[str, Any] = {
        "alpha": 0.05,
        "alpha_schedule": "per_decision",
        "retention_margin": 0.2,
        "hold_limit": 2,
        "shrink_lambdas": (0.5, 0.25),
        "max_group_rollbacks": 2,
        "block_updates": 2,
        "items_per_transfer_slice": 12,
        "items_per_retention_slice": 16,
    }
    kwargs.update(overrides)
    return GateConfig(**kwargs)


def make_probe(sampler: ItemSampler, score: Any, config: GateConfig) -> CapabilityProbe:
    return CapabilityProbe(
        sampler,
        score,
        transfer_slices=TRANSFER,
        retention_slices=RETENTION,
        items_per_transfer_slice=config.items_per_transfer_slice,
        items_per_retention_slice=config.items_per_retention_slice,
    )


def params_of(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def assert_params(model: nn.Module, expected: dict[str, torch.Tensor]) -> None:
    named = dict(model.named_parameters())
    for n, v in expected.items():
        assert named[n].dtype == v.dtype, n
        assert torch.equal(named[n].detach().cpu(), v.cpu()), n


def assert_tree_equal(a: Any, b: Any, path: str = "") -> None:
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), path
        assert a.dtype == b.dtype and a.shape == b.shape, path
        assert torch.equal(a.cpu(), b.cpu()), path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for k in a:
            assert_tree_equal(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_tree_equal(x, y, f"{path}[{i}]")
    else:
        assert a == b, (path, a, b)


def masters_of(model: nn.Module, opt: HookedOptimizer) -> dict[str, torch.Tensor]:
    out = {}
    for n, p in model.named_parameters():
        m = opt.master_of(p)
        if m is not None:
            out[n] = m.detach().clone()
    return out


class Setup:
    """Model + optimizer + operator, one step taken before the incumbent and one after."""

    def __init__(self, tok: Any, *, dtype: torch.dtype = torch.float32, hooked: bool = False) -> None:
        self.model = build_tiny_llama(vocab(tok), seed=0).to(dtype)  # type: ignore[arg-type]
        self.model.train()
        if hooked:
            self.opt: torch.optim.Optimizer = HookedOptimizer(
                torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.1),
                self.model.named_parameters(),
                master_weights=True,
            )
        else:
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-2, weight_decay=0.1)
        self.op = LMStepOperator(self.model, self.opt, vocab(tok))
        self.op.step(self.model, 0)
        self.incumbent = capture(self.model, self.op.parameter_names(), optimizer=self.opt)
        self.op.step(self.model, 1)
        self.candidate = capture(self.model, self.op.parameter_names(), optimizer=self.opt)

    def assert_is_incumbent(self) -> None:
        assert_params(self.model, self.incumbent.params)
        assert_tree_equal(self.opt.state_dict(), self.incumbent.optimizer_state)

    def assert_is_candidate(self) -> None:
        assert_params(self.model, self.candidate.params)
        assert_tree_equal(self.opt.state_dict(), self.candidate.optimizer_state)


def controller_for(
    sampler: ItemSampler,
    plan: Sequence[str],
    config: GateConfig,
    *,
    tok: Any = None,
    rollback_unit: RollbackUnit = RollbackUnit.PARAMETERS,
    groups: Sequence[ChannelGroup] | None = None,
) -> tuple[GateController, ScriptedScore, CapabilityGate]:
    score = ScriptedScore(plan)
    probe = make_probe(sampler, score, config)
    gate = CapabilityGate(config)
    ctl = GateController(gate, probe, rollback_unit=rollback_unit, channel_groups=groups, tokenizer=tok)
    return ctl, score, gate


def assert_fresh_attempts(score: ScriptedScore) -> None:
    sets = [set(u) for u in score.uids]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert sets[i] != sets[j]


# ============================================================================== decision paths
def test_accept_promotes_candidate(sampler: ItemSampler, tok: Any) -> None:
    s = Setup(tok)
    ctl, _score, gate = controller_for(sampler, ["pass"], base_config())
    record, new = ctl.run(s.model, s.op, s.incumbent, 0, 1, False, 0.5, 0.25)
    assert record.decision is GateDecision.ACCEPT and record.reason == REASON_ACCEPT
    assert len(record.verdicts) == 1 == gate.ledger.count
    assert new is not s.incumbent
    assert_tree_equal(new.params, s.candidate.params)
    assert_tree_equal(new.optimizer_state, s.candidate.optimizer_state)
    s.assert_is_candidate()  # nothing written
    assert s.op.decisions == [GateDecision.ACCEPT]
    assert record.reward_v1 == 0.5 and record.reward_v2 == 0.25 and record.override is None
    assert ctl.holds == 0


def test_hold_keeps_weights_and_counts(sampler: ItemSampler, tok: Any) -> None:
    s = Setup(tok)
    ctl, _, gate = controller_for(sampler, ["hold"], base_config())
    record, new = ctl.run(s.model, s.op, s.incumbent, 0, 1, False, None)
    assert record.decision is GateDecision.HOLD and record.reason == REASON_NO_TRANSFER_GAIN
    assert new is s.incumbent and ctl.holds == 1 and gate.ledger.count == 1
    s.assert_is_candidate()


def test_hold_limit_forces_rollback(sampler: ItemSampler, tok: Any) -> None:
    s = Setup(tok)
    ctl, score, gate = controller_for(sampler, ["hold"] * 3, base_config(hold_limit=2))
    decisions = []
    for d in range(3):
        record, inc = ctl.run(s.model, s.op, s.incumbent, d, d + 1, False, None)
        assert inc is s.incumbent
        decisions.append((record.decision, record.reason))
    assert decisions == [
        (GateDecision.HOLD, REASON_NO_TRANSFER_GAIN),
        (GateDecision.HOLD, REASON_NO_TRANSFER_GAIN),
        (GateDecision.ROLLBACK, REASON_HOLD_LIMIT),
    ]
    assert ctl.holds == 0 and gate.ledger.count == 3
    s.assert_is_incumbent()
    assert_fresh_attempts(score)  # different decisions -> different items


def test_alarm_hold_counts_toward_hold_limit(sampler: ItemSampler, tok: Any) -> None:
    s = Setup(tok)
    ctl, _, _ = controller_for(sampler, ["pass", "pass"], base_config(hold_limit=1))
    record, inc = ctl.run(s.model, s.op, s.incumbent, 0, 1, True, None)
    assert record.decision is GateDecision.HOLD and record.reason == REASON_ALARM_HOLD and record.alarm
    assert inc is s.incumbent and ctl.holds == 1
    s.assert_is_candidate()
    record, inc = ctl.run(s.model, s.op, s.incumbent, 1, 2, True, None)
    assert record.decision is GateDecision.ROLLBACK and record.reason == REASON_HOLD_LIMIT
    s.assert_is_incumbent()


@pytest.mark.parametrize("plan", ["fail", "hold", "fail_gain"])
def test_confirmed_alarm_rolls_back_directly(sampler: ItemSampler, tok: Any, plan: str) -> None:
    s = Setup(tok)
    ctl, _, gate = controller_for(sampler, [plan], base_config())
    record, inc = ctl.run(s.model, s.op, s.incumbent, 0, 1, True, None)
    assert record.decision is GateDecision.ROLLBACK and record.reason == REASON_CONFIRMED_ALARM
    assert len(record.verdicts) == 1 == gate.ledger.count  # no shrink re-test under an alarm
    assert inc is s.incumbent
    s.assert_is_incumbent()
    assert s.op.decisions == [GateDecision.ROLLBACK]


def test_shrink_promoted(sampler: ItemSampler, tok: Any) -> None:
    s = Setup(tok)
    ctl, score, gate = controller_for(sampler, ["fail", "pass"], base_config())
    record, new = ctl.run(s.model, s.op, s.incumbent, 0, 1, False, None)
    assert record.decision is GateDecision.SHRINK and record.shrink_lambda == 0.5
    assert record.reason == "shrink_passed;promoted=true"
    assert len(record.verdicts) == 2 == gate.ledger.count
    assert_fresh_attempts(score)
    named = dict(s.model.named_parameters())
    for n in s.incumbent.params:
        a, b = s.incumbent.params[n], s.candidate.params[n]
        torch.testing.assert_close(named[n].detach(), a + 0.5 * (b - a), rtol=0, atol=1e-6)
    assert_tree_equal(s.opt.state_dict(), s.candidate.optimizer_state)  # moments kept
    assert new is not s.incumbent
    assert_tree_equal(new.params, params_of(s.model))
    assert ctl.holds == 0 and s.op.decisions == [GateDecision.SHRINK]


def test_shrink_not_promoted_counts_as_hold(sampler: ItemSampler, tok: Any) -> None:
    s = Setup(tok)
    ctl, _, gate = controller_for(sampler, ["fail", "fail", "hold"], base_config())
    record, new = ctl.run(s.model, s.op, s.incumbent, 0, 1, False, None)
    assert record.decision is GateDecision.SHRINK and record.shrink_lambda == 0.25
    assert record.reason == "shrink_passed;promoted=false"
    assert len(record.verdicts) == 3 == gate.ledger.count
    assert new is s.incumbent and ctl.holds == 1
    named = dict(s.model.named_parameters())
    for n in s.incumbent.params:
        a, b = s.incumbent.params[n], s.candidate.params[n]
        torch.testing.assert_close(named[n].detach(), a + 0.25 * (b - a), rtol=0, atol=1e-6)
    assert_tree_equal(s.opt.state_dict(), s.candidate.optimizer_state)


def test_unpromoted_shrink_at_hold_limit_is_kept(sampler: ItemSampler, tok: Any) -> None:
    """A passing shrink is never converted into a ROLLBACK; it extends the streak (capped), so the
    next would-be HOLD is forced to ROLLBACK."""
    s = Setup(tok)
    ctl, _, gate = controller_for(sampler, ["hold", "fail", "hold", "hold"], base_config(hold_limit=1))
    first, _ = ctl.run(s.model, s.op, s.incumbent, 0, 1, False, None)
    assert first.decision is GateDecision.HOLD and ctl.holds == 1
    record, inc = ctl.run(s.model, s.op, s.incumbent, 1, 2, False, None)
    assert record.decision is GateDecision.SHRINK and record.reason == "shrink_passed;promoted=false"
    assert len(record.verdicts) == 2 and gate.ledger.count == 3 and ctl.holds == 1
    assert inc is s.incumbent
    named = dict(s.model.named_parameters())
    for n in s.incumbent.params:
        a, b = s.incumbent.params[n], s.candidate.params[n]
        torch.testing.assert_close(named[n].detach(), a + 0.5 * (b - a), rtol=0, atol=1e-6)
    third, _ = ctl.run(s.model, s.op, s.incumbent, 2, 3, False, None)
    assert third.decision is GateDecision.ROLLBACK and third.reason == REASON_HOLD_LIMIT
    s.assert_is_incumbent()


def test_rollback_after_everything_fails(sampler: ItemSampler, tok: Any) -> None:
    s = Setup(tok)
    ctl, score, gate = controller_for(sampler, ["fail"] * 3, base_config())
    record, inc = ctl.run(s.model, s.op, s.incumbent, 0, 1, False, None)
    assert record.decision is GateDecision.ROLLBACK and record.reason == REASON_RETENTION_FAILED
    assert len(record.verdicts) == 3 == gate.ledger.count == score.evaluations
    assert_fresh_attempts(score)
    assert inc is s.incumbent
    s.assert_is_incumbent()


def test_rollback_restores_hooked_masters_bitwise(sampler: ItemSampler, tok: Any) -> None:
    s = Setup(tok, dtype=torch.bfloat16, hooked=True)
    assert isinstance(s.opt, HookedOptimizer)
    assert s.incumbent.masters
    assert masters_of(s.model, s.opt).keys() == s.incumbent.masters.keys()
    ctl, _, _ = controller_for(sampler, ["fail"] * 3, base_config())
    record, _ = ctl.run(s.model, s.op, s.incumbent, 0, 1, False, None)
    assert record.decision is GateDecision.ROLLBACK
    assert_params(s.model, s.incumbent.params)
    assert_tree_equal(masters_of(s.model, s.opt), s.incumbent.masters)
    assert_tree_equal(s.opt.state_dict(), s.incumbent.optimizer_state)
    s.op.step(s.model, 2)  # training continues from the restored state without resync
    assert s.opt.master_resyncs == 0


def test_shrink_with_hooked_masters_keeps_masters_consistent(sampler: ItemSampler, tok: Any) -> None:
    s = Setup(tok, dtype=torch.bfloat16, hooked=True)
    assert isinstance(s.opt, HookedOptimizer)
    ctl, _, _ = controller_for(sampler, ["fail", "pass"], base_config())
    record, new = ctl.run(s.model, s.op, s.incumbent, 0, 1, False, None)
    assert record.decision is GateDecision.SHRINK
    masters = masters_of(s.model, s.opt)
    for n, p in s.model.named_parameters():
        assert torch.equal(p.detach(), masters[n].to(p.dtype)), n
    assert new.masters is not None
    assert_tree_equal(new.masters, masters)


def test_group_rollback_in_damage_order(sampler: ItemSampler, tok: Any) -> None:
    s = Setup(tok)
    groups = group_channels(random_scores(s.model, 3), bands=2)
    assert len(groups) == 4
    config = base_config(rollback_unit=RollbackUnit.CHANNELS)
    ctl, score, gate = controller_for(
        sampler, ["fail"] * 4 + ["pass"], config, tok=tok, rollback_unit=RollbackUnit.CHANNELS, groups=groups
    )
    # expected damage order: raw candidate, prompts of the first verdict's failing slices (both)
    items = ctl.probe.gate_items(0, 0)
    texts = [it.prompt for name in RETENTION for it in items[name]]
    stats = collect_channel_stats(s.model, tok, texts, layers=[0, 1])
    order = channel_groups_by_damage(s.incumbent, s.model, stats.mean_abs, groups)
    assert [g.name for g in order[:2]] != [g.name for g in groups[:2]]  # the order matters here
    record, new = ctl.run(s.model, s.op, s.incumbent, 0, 1, False, None)
    assert record.decision is GateDecision.GROUP_ROLLBACK
    assert record.reason == "group_rollback_passed;promoted=true"
    assert record.groups_reverted == [order[0].name, order[1].name]
    assert len(record.verdicts) == 5 == gate.ledger.count
    assert_fresh_attempts(score)
    # reverted channel slices == incumbent, every other element == raw candidate (shrink undone)
    union = order[0].selection.union(order[1].selection)
    mask = channel_mask(s.model, union)
    for n, p in s.model.named_parameters():
        now = p.detach()
        owned = mask.get(n)
        if owned is None:
            assert torch.equal(now, s.candidate.params[n]), n
        else:
            assert torch.equal(now[owned], s.incumbent.params[n][owned]), n
            assert torch.equal(now[~owned], s.candidate.params[n][~owned]), n
    assert_tree_equal(s.opt.state_dict(), s.candidate.optimizer_state)
    assert new is not s.incumbent


def test_group_stage_then_rollback(sampler: ItemSampler, tok: Any) -> None:
    s = Setup(tok)
    groups = group_channels(random_scores(s.model, 3), bands=2)
    config = base_config(rollback_unit=RollbackUnit.CHANNELS)
    ctl, _, gate = controller_for(
        sampler, ["fail"] * 5, config, tok=tok, rollback_unit=RollbackUnit.CHANNELS, groups=groups
    )
    assert ctl.max_verdicts_per_decision == 5
    record, _ = ctl.run(s.model, s.op, s.incumbent, 0, 1, False, None)
    assert record.decision is GateDecision.ROLLBACK and record.reason == REASON_RETENTION_FAILED
    assert len(record.verdicts) == 5 == gate.ledger.count
    s.assert_is_incumbent()


def test_parameters_unit_never_group_rollback(sampler: ItemSampler, tok: Any) -> None:
    s = Setup(tok)
    groups = group_channels(random_scores(s.model, 3), bands=2)
    with pytest.raises(ValueError, match="CHANNELS"):
        controller_for(sampler, ["fail"], base_config(), tok=tok, groups=groups)


def test_gate_disabled_records_would_and_separate_ledger(sampler: ItemSampler, tok: Any) -> None:
    s = Setup(tok)
    config = base_config(enabled=False)
    ctl, score, gate = controller_for(sampler, ["fail", "pass", "hold"], config)
    assert ctl.verdict_gate is not gate
    record, new = ctl.run(s.model, s.op, s.incumbent, 0, 1, False, None)
    assert record.decision is GateDecision.ACCEPT and record.override == OVERRIDE_GATE_DISABLED
    assert record.reason == "would=" + REASON_RETENTION_FAILED
    assert len(record.verdicts) == 1 and score.evaluations == 1  # only the first verdict
    assert gate.ledger.count == 0 and ctl.verdict_gate.ledger.count == 1
    s.assert_is_candidate()  # never rolled back
    assert_tree_equal(new.params, s.candidate.params)
    reasons = [record.reason]
    for d in (1, 2):
        rec, new = ctl.run(s.model, s.op, new, d, d + 1, False, None)
        reasons.append(rec.reason)
    assert reasons == ["would=retention_failed", "would=accept", "would=no_transfer_gain"]
    assert gate.ledger.count == 0 and ctl.verdict_gate.ledger.count == 3


def test_gate_disabled_alarm_would_reasons(sampler: ItemSampler, tok: Any) -> None:
    s = Setup(tok)
    ctl, _, _ = controller_for(sampler, ["fail", "pass"], base_config(enabled=False))
    r1, inc = ctl.run(s.model, s.op, s.incumbent, 0, 1, True, None)
    r2, _ = ctl.run(s.model, s.op, inc, 1, 2, True, None)
    assert (r1.reason, r2.reason) == ("would=confirmed_alarm", "would=alarm_hold")
    assert r1.decision is r2.decision is GateDecision.ACCEPT


def test_horizon_exhausted(sampler: ItemSampler, tok: Any) -> None:
    s = Setup(tok)
    config = base_config(retention_margin=0.02, items_per_retention_slice=4)
    ctl, _, gate = controller_for(sampler, ["pass"], config)
    record, inc = ctl.run(s.model, s.op, s.incumbent, 0, 1, False, None)
    assert record.decision is GateDecision.ROLLBACK and record.reason == REASON_HORIZON_EXHAUSTED
    assert len(record.verdicts) == 1 == gate.ledger.count  # no shrink: later alphas never larger
    assert retention_unpassable(record.verdicts[0])
    s.assert_is_incumbent()
    ctl2, _, _ = controller_for(
        sampler, ["fail"], base_config(retention_margin=0.02, items_per_retention_slice=4, enabled=False)
    )
    rec2, _ = ctl2.run(s.model, s.op, inc, 0, 1, False, None)
    assert rec2.reason == "would=horizon_exhausted"


def test_horizon_exhausted_during_shrink(sampler: ItemSampler, tok: Any) -> None:
    """Spending schedule: the first verdict can pass retention, the second (smaller alpha) not."""
    config = base_config(alpha_schedule="spending", retention_margin=0.2, items_per_retention_slice=16)
    gate = CapabilityGate(config)
    assert config.retention_horizon(n=16) == 1
    s = Setup(tok)
    score = ScriptedScore(["fail", "pass"])
    ctl = GateController(gate, make_probe(sampler, score, config), rollback_unit=RollbackUnit.PARAMETERS)
    record, _ = ctl.run(s.model, s.op, s.incumbent, 0, 1, False, None)
    assert record.reason == REASON_HORIZON_EXHAUSTED and len(record.verdicts) == 2
    assert not retention_unpassable(record.verdicts[0]) and retention_unpassable(record.verdicts[1])
    s.assert_is_incumbent()


def test_classify_require_transfer_gain_false(sampler: ItemSampler, tok: Any) -> None:
    s = Setup(tok)
    ctl, _, _ = controller_for(sampler, ["hold", "hold"], base_config(require_transfer_gain=False))
    record, _ = ctl.run(s.model, s.op, s.incumbent, 0, 1, False, None)
    assert record.decision is GateDecision.ACCEPT
    record, _ = ctl.run(s.model, s.op, s.incumbent, 1, 2, True, None)
    assert record.reason == REASON_ALARM_HOLD
    assert classify_verdict(record.verdicts[0], require_transfer_gain=True) == REASON_CONFIRMED_ALARM


def test_incumbent_without_optimizer_state_refused(sampler: ItemSampler, tok: Any) -> None:
    s = Setup(tok)
    bare = capture(s.model, s.op.parameter_names())
    ctl, score, gate = controller_for(sampler, ["fail"], base_config())
    with pytest.raises(ValueError, match="optimizer state"):
        ctl.run(s.model, s.op, bare, 0, 1, False, None)
    assert gate.ledger.count == 0 and score.calls == 0
    s.assert_is_candidate()
    with pytest.raises(TypeError):
        ctl.run(s.model, s.op, s.incumbent, 0, 1, 1, None)  # type: ignore[arg-type]


def test_controller_state_dict_roundtrip(sampler: ItemSampler, tok: Any) -> None:
    s = Setup(tok)
    ctl, _, _gate = controller_for(sampler, ["hold"], base_config())
    ctl.run(s.model, s.op, s.incumbent, 0, 1, False, None)
    state = json.loads(json.dumps(ctl.state_dict()))
    other, _, gate2 = controller_for(sampler, [], base_config())
    other.load_state_dict(state)
    assert other.holds == 1 and gate2.ledger.count == 1
    state["holds"] = 7
    with pytest.raises(ValueError):
        other.load_state_dict(state)


def test_snapshot_state_roundtrip(tmp_path: Path, tok: Any) -> None:
    s = Setup(tok, dtype=torch.bfloat16, hooked=True)
    path = tmp_path / "snap.pt"
    torch.save(snapshot_to_state(s.incumbent), path)
    back = snapshot_from_state(torch.load(path, weights_only=True))
    assert isinstance(back, StateSnapshot)
    assert_tree_equal(back.params, s.incumbent.params)
    assert_tree_equal(back.optimizer_state, s.incumbent.optimizer_state)
    assert_tree_equal(back.masters, s.incumbent.masters)


def test_worst_case_verdicts() -> None:
    assert (
        worst_case_verdicts(10, block_updates=4, verdicts_per_decision=3, alarm_triggers_decision=False) == 6
    )
    assert (
        worst_case_verdicts(10, block_updates=4, verdicts_per_decision=3, alarm_triggers_decision=True) == 14
    )
    assert (
        worst_case_verdicts(
            3, block_updates=4, verdicts_per_decision=3, alarm_triggers_decision=False, since=1
        )
        == 3
    )
    assert worst_case_verdicts(0, block_updates=4, verdicts_per_decision=3, alarm_triggers_decision=True) == 0


# ============================================================================== GatedLoop
class ListRecorder:
    def __init__(self) -> None:
        self.rows: list[tuple[int, dict[str, Any]]] = []

    def log_metrics(self, step: int, logs: Any) -> None:
        self.rows.append((step, dict(logs)))


def lm_loop(
    sampler: ItemSampler,
    tok: Any,
    plan: Sequence[str],
    config: GateConfig,
    *,
    feed: Callable[[int], tuple[list[bool], list[bool]]] | None = None,
    alarm: DisagreementAlarm | None = None,
    log_path: Path | None = None,
    recorder: Any = None,
    allow_short_horizon: bool = False,
) -> tuple[GatedLoop, LMStepOperator]:
    model = build_tiny_llama(vocab(tok), seed=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    op = LMStepOperator(model, opt, vocab(tok), feed)
    ctl, _, _ = controller_for(sampler, plan, config)
    loop = GatedLoop(
        model,
        op,
        ctl,
        alarm or DisagreementAlarm(min_rollouts=8),
        config=config,
        recorder=recorder,
        log_path=log_path,
        allow_short_horizon=allow_short_horizon,
    )
    return loop, op


def test_loop_block_schedule_log_and_recorder(sampler: ItemSampler, tok: Any, tmp_path: Path) -> None:
    log = tmp_path / "decisions.jsonl"
    rec = ListRecorder()
    config = base_config(block_updates=3, hold_limit=5)
    loop, op = lm_loop(sampler, tok, ["hold", "pass"], config, log_path=log, recorder=rec)
    records = loop.run(7)
    assert [r.update_index for r in records] == [2, 5]
    assert [r.decision for r in records] == [GateDecision.HOLD, GateDecision.ACCEPT]
    assert loop.decision_index == 2 and loop.update_index == 7 and loop.updates_since_decision == 1
    lines = log.read_text().splitlines()
    assert [json.loads(ln) for ln in lines] == [r.to_dict() for r in records]
    assert [step for step, _ in rec.rows] == [2, 5]
    assert rec.rows[1][1]["gate/decision"] == "accept"
    assert op.decisions == [GateDecision.HOLD, GateDecision.ACCEPT]
    # incumbent after ACCEPT at update 5 == weights after update 5; update 6 moved them since
    assert not all(
        torch.equal(loop.incumbent.params[n], p.detach()) for n, p in loop.model.named_parameters()
    )


def _leak_feed(start: int) -> Callable[[int], tuple[list[bool], list[bool]]]:
    def feed(t: int) -> tuple[list[bool], list[bool]]:
        if t < start:
            return [False] * 32, [False] * 32
        return [True] * 32, [False] * 32

    return feed


def test_alarm_triggers_immediate_decision(sampler: ItemSampler, tok: Any) -> None:
    config = base_config(block_updates=4)
    alarm = DisagreementAlarm(min_rollouts=8, baseline_updates=1)
    loop, _op = lm_loop(sampler, tok, ["fail"], config, feed=_leak_feed(1), alarm=alarm)
    records = loop.run(2)
    assert len(records) == 1
    rec = records[0]
    assert rec.update_index == 1 and rec.alarm
    assert rec.decision is GateDecision.ROLLBACK and rec.reason == REASON_CONFIRMED_ALARM
    assert_params(loop.model, loop.incumbent.params)
    assert loop.updates_since_decision == 0


def test_alarm_latched_when_not_triggering(sampler: ItemSampler, tok: Any) -> None:
    config = base_config(block_updates=4, alarm_triggers_decision=False)
    alarm = DisagreementAlarm(min_rollouts=8, baseline_updates=1)
    feed = _leak_feed(1)

    def pulse(t: int) -> tuple[list[bool], list[bool]]:  # leak only at update 1: one alarm pulse
        return feed(t) if t == 1 else ([False] * 32, [False] * 32)

    loop, _ = lm_loop(sampler, tok, ["fail"], config, feed=pulse, alarm=alarm)
    records = loop.run(4)
    assert [r.update_index for r in records] == [3]
    assert records[0].alarm and records[0].reason == REASON_CONFIRMED_ALARM


def test_start_of_run_horizon_check(sampler: ItemSampler, tok: Any) -> None:
    config = base_config(alpha_schedule="spending", block_updates=2)
    loop, op = lm_loop(sampler, tok, ["hold"] * 4, config)
    assert loop.horizon() == 1
    assert loop.required_verdicts(2) == 3 + 1  # one block decision (3 verdicts) + one alarm update
    with pytest.raises(ValueError, match="retention horizon"):
        loop.run(2)
    assert op.steps == 0  # refused before the first update
    loop.run(0)  # nothing planned: fine
    warned, op2 = lm_loop(sampler, tok, ["hold"] * 4, config, allow_short_horizon=True)
    with pytest.warns(UserWarning, match="retention horizon"):
        warned.run(2)
    assert op2.steps == 2
    ok, _ = lm_loop(sampler, tok, ["hold"] * 4, base_config(block_updates=2))
    assert ok.horizon() is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ok.run(2)


def test_loop_rejects_config_mismatch(sampler: ItemSampler, tok: Any) -> None:
    model = build_tiny_llama(vocab(tok), seed=0)
    op = LMStepOperator(model, torch.optim.AdamW(model.parameters()), vocab(tok))
    ctl, _, _ = controller_for(sampler, [], base_config())
    with pytest.raises(ValueError, match="config"):
        GatedLoop(model, op, ctl, DisagreementAlarm(), config=base_config(block_updates=5))


# ============================================================================== real operators
class DigitStartVerifier(BaseVerifier):
    """Lenient test verifier: correct iff the stripped completion starts with a digit 0-4 (about
    half of the warm model's samples, so groups have reward variance)."""

    name = "low_digit_start"

    def _verify(
        self, completion: str | None, reference: Any, **meta: Any
    ) -> tuple[Status, str | None, str | None, dict[str, Any]]:
        if meta.get("truncated"):
            return "truncated", None, str(reference), {}
        text = (completion or "").strip()
        if not text:
            return "no_answer", None, str(reference), {}
        return ("correct" if text[0] in "01234" else "incorrect"), text, str(reference), {}


@pytest.fixture(scope="module")
def warm_state(sampler: ItemSampler, tok: Any) -> dict[str, torch.Tensor]:
    """A tiny model trained for 40 steps to answer ``prompt -> one random digit, EOS`` (so sampled
    completions end with EOS and the lenient verifier sees varied rewards)."""
    model = build_tiny_llama(vocab(tok), seed=0)
    items = sampler.draw(Split.TRAIN, 64, purpose="warm", index=0)
    g = torch.Generator().manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for step in range(40):
        seqs = []
        for i in range(16):
            item = items[(step * 16 + i) % 64]
            digit = str(int(torch.randint(0, 10, (1,), generator=g)))
            seqs.append(
                [*list(tok(item.prompt + digit, add_special_tokens=False)["input_ids"]), tok.eos_token_id]
            )
        width = max(len(q) for q in seqs)
        ids = torch.zeros(16, width, dtype=torch.long)
        labels = torch.full((16, width), -100, dtype=torch.long)
        for r, q in enumerate(seqs):
            ids[r, : len(q)] = torch.tensor(q)
            labels[r, len(q) - 2 : len(q)] = torch.tensor(q[-2:])
        opt.zero_grad(set_to_none=True)
        model(input_ids=ids, labels=labels).loss.backward()
        opt.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def real_probe(sampler: ItemSampler, tok: Any, config: GateConfig) -> CapabilityProbe:
    score = VerifierScoreFn(tok, ExactAnswerVerifier(), max_new_tokens=8, batch_size=32)
    return CapabilityProbe(
        sampler,
        score,
        transfer_slices=TRANSFER,
        retention_slices=RETENTION[:1],
        items_per_transfer_slice=config.items_per_transfer_slice,
        items_per_retention_slice=config.items_per_retention_slice,
    )


def grpo_loop(
    sampler: ItemSampler,
    tok: Any,
    config: GateConfig,
    warm: dict[str, torch.Tensor],
    *,
    prompts: int = 4,
    log_path: Path | None = None,
    min_rollouts: int = 4,
    alarm: DisagreementAlarm | None = None,
    dropout: float = 0.0,
    v2: Any = None,
) -> GatedLoop:
    model = build_tiny_llama(vocab(tok), seed=0)
    model.load_state_dict(warm)
    for layer in model.model.layers:
        layer.self_attn.attention_dropout = dropout
    model.train()
    op = TorchGRPOOperator(
        model,
        tok,
        sampler,
        DigitStartVerifier(),
        v2 if v2 is not None else DigitStartVerifier(),
        group_size=4,
        prompts_per_update=prompts,
        lr=1e-2,
        max_new_tokens=8,
        temperature=1.0,
        seed=11,
    )
    ctl = GateController(
        CapabilityGate(config), real_probe(sampler, tok, config), rollback_unit=RollbackUnit.PARAMETERS
    )
    return GatedLoop(
        model,
        op,
        ctl,
        alarm if alarm is not None else DisagreementAlarm(min_rollouts=min_rollouts),
        config=config,
        log_path=log_path,
    )


class RecordingAlarm(DisagreementAlarm):
    """The real alarm, also recording the Kish effective size of every report it is fed."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sizes: list[float] = []

    def update(self, report: OperatorReport) -> Any:
        self.sizes.append(self.effective_size(report))
        return super().update(report)


def test_clean_grpo_stream_does_not_alarm(
    sampler: ItemSampler, tok: Any, warm_state: dict[str, torch.Tensor]
) -> None:
    config = base_config(
        block_updates=4, items_per_transfer_slice=8, items_per_retention_slice=16, hold_limit=5
    )
    recording = RecordingAlarm(min_rollouts=8)
    loop = grpo_loop(sampler, tok, config, warm_state, prompts=8, alarm=recording)
    records = loop.run(8)
    alarm = loop.alarm
    # GRPO's alarm units reach the alarm: 8 prompts x 4 correlated rollouts count as 8 units (a
    # per-rollout unit assignment would give 32 and pass every check below on this V1 == V2 stream)
    assert recording.sizes == [8.0] * 8
    assert alarm.updates == 8 and alarm.skipped_updates == 0 and alarm.baseline_ready
    assert alarm.gap_alarms == 0 and alarm.length_alarms == 0
    assert loop.last_alarm is not None and loop.last_alarm.gap == 0.0
    assert len(records) == 2 and not any(r.alarm for r in records)
    assert all(r.reward_v1 is not None and 0.0 < r.reward_v1 < 1.0 for r in records)  # real signal
    assert all(r.reward_v1 == r.reward_v2 for r in records)


def test_resume_equals_uninterrupted_run(
    sampler: ItemSampler, tok: Any, tmp_path: Path, warm_state: dict[str, torch.Tensor]
) -> None:
    """run(3); save; rebuild; load; run(3)  ==  run(6): decisions, log and weights.

    The cut lies mid-block (block 2: decisions after updates 1, 3, 5) and after a HOLD, so the
    incumbent and the current weights differ at the checkpoint. Both trajectories execute the same
    ops on bit-identical inputs in one process (weights / AdamW state restored by copy, sampling
    seeded per update index, items keyed by index), so the weights are compared bitwise; a
    different thread count or kernel between the two runs would require allclose instead.
    """
    config = base_config(
        block_updates=2, hold_limit=1, items_per_transfer_slice=8, items_per_retention_slice=16
    )
    ref = grpo_loop(sampler, tok, config, warm_state, log_path=tmp_path / "ref.jsonl")
    ref_records = ref.run(6)
    assert [r.decision for r in ref_records] == [GateDecision.HOLD, GateDecision.ROLLBACK, GateDecision.HOLD]

    first = grpo_loop(sampler, tok, config, warm_state, log_path=tmp_path / "cut.jsonl")
    first.run(3)
    assert first.updates_since_decision == 1
    assert any(
        not torch.equal(first.incumbent.params[n], p.detach()) for n, p in first.model.named_parameters()
    )
    path = tmp_path / "state.pt"
    torch.save(first.state_dict(), path)
    state = torch.load(path, weights_only=True)
    resumed = grpo_loop(sampler, tok, config, warm_state, log_path=tmp_path / "cut.jsonl")
    resumed.load_state_dict(state)
    resumed.run(3)

    assert (tmp_path / "cut.jsonl").read_text() == (tmp_path / "ref.jsonl").read_text()
    assert [r.to_dict() for r in first.records + resumed.records] == [r.to_dict() for r in ref_records]
    assert_params(resumed.model, params_of(ref.model))
    assert resumed.operator.optimizer is not None and ref.operator.optimizer is not None
    assert_tree_equal(resumed.operator.optimizer.state_dict(), ref.operator.optimizer.state_dict())
    assert_tree_equal(resumed.incumbent.params, ref.incumbent.params)
    assert resumed.alarm.state_dict() == ref.alarm.state_dict()
    assert resumed.controller.state_dict() == ref.controller.state_dict()
    assert resumed.operator.state_dict() == ref.operator.state_dict()


def test_gate_disabled_loop_never_rolls_back_but_logs(sampler: ItemSampler, tok: Any, tmp_path: Path) -> None:
    log = tmp_path / "ungated.jsonl"
    config = base_config(enabled=False, block_updates=2)
    loop, op = lm_loop(sampler, tok, ["fail", "hold", "pass"], config, log_path=log)
    records = loop.run(6)
    assert [r.decision for r in records] == [GateDecision.ACCEPT] * 3
    rows = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert [r["override"] for r in rows] == [OVERRIDE_GATE_DISABLED] * 3
    assert [r["reason"] for r in rows] == ["would=retention_failed", "would=no_transfer_gain", "would=accept"]
    assert all(len(r["verdicts"]) == 1 for r in rows)
    assert loop.controller.gate.ledger.count == 0 and loop.controller.verdict_gate.ledger.count == 3
    assert_params(loop.model, loop.incumbent.params)  # every block promoted, nothing restored
    assert op.decisions == [GateDecision.ACCEPT] * 3


def test_load_state_trims_log(sampler: ItemSampler, tok: Any, tmp_path: Path) -> None:
    log = tmp_path / "d.jsonl"
    config = base_config(block_updates=2, hold_limit=5)
    loop, _ = lm_loop(sampler, tok, ["hold"] * 4, config, log_path=log)
    loop.run(2)
    state = loop.state_dict()
    loop.run(2)
    assert len(log.read_text().splitlines()) == 2
    again, _ = lm_loop(sampler, tok, ["hold"] * 4, config, log_path=log)
    again.load_state_dict(state)
    assert len(log.read_text().splitlines()) == 1
    bad = dict(state)
    bad["config"] = base_config(block_updates=3).to_dict()
    with pytest.raises(ValueError, match="GateConfig"):
        again.load_state_dict(bad)


def test_es_operator_inside_gated_loop(sampler: ItemSampler, tok: Any) -> None:
    config = base_config(
        block_updates=2, hold_limit=1, items_per_transfer_slice=8, items_per_retention_slice=16
    )
    model = build_tiny_llama(vocab(tok), seed=0)
    space = ChannelGainSpace(model, ChannelSelection({0: list(range(16)), 1: list(range(16))}))
    v1 = VerifierScoreFn(tok, DigitStartVerifier(), max_new_tokens=8, batch_size=8)
    v2 = VerifierScoreFn(tok, ExactAnswerVerifier(), max_new_tokens=8, batch_size=8)

    def fitness(m: nn.Module, update_index: int, member_index: int) -> FitnessResult:
        """Smooth fitness (the warm model's greedy output is the same digit for every prompt, so a
        greedy pass rate would be flat): mean probability of the answer's first digit after the
        prompt; the rollouts reported to the alarm are greedy completions under V1 / V2."""
        items = sampler.draw(Split.TRAIN, 8, purpose="es", index=update_index)
        texts = v1.generate(m, items)
        trunc = v1.truncated(texts)
        s1 = v1.score_completions(texts, items, trunc)
        s2 = v2.score_completions(texts, items, trunc)
        probs = []
        for it in items:
            ids = torch.tensor([tok(it.prompt, add_special_tokens=False)["input_ids"]])
            target = tok(it.answer[0], add_special_tokens=False)["input_ids"][0]
            probs.append(float(torch.softmax(m(input_ids=ids).logits[0, -1].float(), -1)[target]))
        return FitnessResult(
            reward_v1=sum(probs) / len(probs),
            reward_v2=sum(s2) / len(s2),
            v1_pass=tuple(x == 1.0 for x in s1),
            v2_pass=tuple(x == 1.0 for x in s2),
            lengths=tuple(len(t) for t in texts),
        )

    es = ESOperator(space, fitness, ESConfig(population=4, sigma=0.1, lr=0.01, seed=0))
    alarm = DisagreementAlarm(min_rollouts=8, rollouts_per_unit=4)  # 4 members x 8 items -> n_eff 8
    ctl = GateController(
        CapabilityGate(config), real_probe(sampler, tok, config), rollback_unit=RollbackUnit.CHANNELS
    )
    loop = GatedLoop(model, es, ctl, alarm, config=config)
    assert set(loop.incumbent.params) == set(es.parameter_names())
    records = loop.run(2)
    assert [r.decision for r in records] == [GateDecision.HOLD]
    assert es.steps_done == 2 and es.last_update_index == 1
    moved = [
        n
        for n in es.parameter_names()
        if not torch.equal(dict(model.named_parameters())[n].detach(), loop.incumbent.params[n])
    ]
    assert moved  # the ES steps changed the gains (non-constant fitness)
    records += loop.run(2)
    assert [r.decision for r in records] == [GateDecision.HOLD, GateDecision.ROLLBACK]
    assert alarm.skipped_updates == 0 and alarm.updates == 4
    # after the rollback the model is the incumbent and the ES space was re-based on it
    assert_params(model, loop.incumbent.params)
    before = params_of(model)
    space.write(None)
    assert_params(model, before)
    assert es.forward_passes_total == 4 * 4
    assert records[0].reward_v1 is not None and records[0].reward_v1 > 0.0
    assert math.isfinite(records[0].reward_v1 or 0.0)


# ============================================================================== review regressions
def test_resume_with_dropout_equals_uninterrupted_run(
    sampler: ItemSampler, tok: Any, warm_state: dict[str, torch.Tensor]
) -> None:
    """Regression: with attention dropout active in the GRPO training forward (model in train
    mode), the dropout masks are seeded per update, so a resumed run equals the uninterrupted one
    even though the global RNG differs between them (bitwise: identical ops on identical inputs in
    one process, see test_resume_equals_uninterrupted_run)."""
    config = base_config(
        block_updates=2, hold_limit=1, items_per_transfer_slice=8, items_per_retention_slice=16
    )
    torch.manual_seed(0)
    ref = grpo_loop(sampler, tok, config, warm_state, dropout=0.2)
    ref_records = ref.run(5)
    torch.manual_seed(1)
    first = grpo_loop(sampler, tok, config, warm_state, dropout=0.2)
    first.run(3)
    state = first.state_dict()
    resumed = grpo_loop(sampler, tok, config, warm_state, dropout=0.2)
    torch.manual_seed(2)
    resumed.load_state_dict(state)
    resumed.run(2)
    assert [r.to_dict() for r in first.records + resumed.records] == [r.to_dict() for r in ref_records]
    assert_params(resumed.model, params_of(ref.model))


def test_group_scope_checked_before_any_verdict(sampler: ItemSampler, tok: Any) -> None:
    """Regression: channel groups the incumbent cannot revert (a layer outside the operator's
    parameters) are refused before the first verdict: no alpha spent, nothing written."""
    model = build_tiny_llama(vocab(tok), seed=0)
    prefix = "model.layers.0.mlp"
    named = dict(model.named_parameters())
    opt = torch.optim.AdamW(
        [named[f"{prefix}.{p}.weight"] for p in ("gate_proj", "up_proj", "down_proj")], lr=1e-2
    )
    op = LMStepOperator(model, opt, vocab(tok))
    op.step(model, 0)
    incumbent = capture(model, op.parameter_names(), optimizer=opt)
    op.step(model, 1)
    groups = [
        ChannelGroup("g0", ChannelSelection({0: [0, 1]})),
        ChannelGroup("g1", ChannelSelection({1: [0, 1, 2]})),
    ]
    config = base_config(rollback_unit=RollbackUnit.CHANNELS)
    ctl, score, gate = controller_for(
        sampler, ["fail"] * 6, config, tok=tok, rollback_unit=RollbackUnit.CHANNELS, groups=groups
    )
    before = params_of(model)
    with pytest.raises(ValueError, match="layer 1"):
        ctl.run(model, op, incumbent, 0, 1, False, None)
    assert gate.ledger.count == 0 and score.calls == 0 and ctl.holds == 0
    assert_params(model, before)
    assert op.decisions == []
    # in-scope groups pass the check and the procedure runs (here: group stage, then ROLLBACK)
    ok, _, gate_ok = controller_for(
        sampler, ["fail"] * 6, config, tok=tok, rollback_unit=RollbackUnit.CHANNELS, groups=groups[:1]
    )
    record, _ = ok.run(model, op, incumbent, 0, 1, False, None)
    assert record.decision is GateDecision.ROLLBACK and gate_ok.ledger.count == 4
    # the disabled arm never reaches the group stage and is not refused
    off, _, _ = controller_for(
        sampler,
        ["fail"],
        base_config(rollback_unit=RollbackUnit.CHANNELS, enabled=False),
        tok=tok,
        rollback_unit=RollbackUnit.CHANNELS,
        groups=groups,
    )
    assert off.run(model, op, incumbent, 0, 1, False, None)[0].decision is GateDecision.ACCEPT


def test_load_state_drops_torn_last_log_line(sampler: ItemSampler, tok: Any, tmp_path: Path) -> None:
    """Regression: a crash mid-write leaves a torn last line; resuming drops it (it can only be a
    record at or after the checkpoint) instead of failing."""
    log = tmp_path / "d.jsonl"
    config = base_config(block_updates=2, hold_limit=5)
    loop, _ = lm_loop(sampler, tok, ["hold"] * 4, config, log_path=log)
    loop.run(2)
    state = loop.state_dict()
    loop.run(2)
    with log.open("a", encoding="utf-8") as fh:
        fh.write('{"decision_index": 2, "update_ind')
    again, _ = lm_loop(sampler, tok, ["hold"] * 4, config, log_path=log)
    again.load_state_dict(state)
    rows = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()]
    assert [r["decision_index"] for r in rows] == [0]
    assert again.decision_index == 1 and again.controller.holds == 1


@pytest.mark.parametrize("bad", ["{not json\n", '{"update_index": 1}\n'])
def test_load_state_refuses_corrupt_log_before_writing(
    sampler: ItemSampler, tok: Any, tmp_path: Path, bad: str
) -> None:
    """Regression: an unparsable line that is not the last one (or a record without a
    decision_index) is refused before anything is loaded; the loop and the file are unchanged."""
    log = tmp_path / "d.jsonl"
    config = base_config(block_updates=2, hold_limit=5)
    loop, _ = lm_loop(sampler, tok, ["hold"] * 4, config, log_path=log)
    loop.run(2)
    state = loop.state_dict()
    loop.run(2)
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    log.write_text(lines[0] + bad + lines[1], encoding="utf-8")
    content = log.read_text(encoding="utf-8")
    again, _ = lm_loop(sampler, tok, ["hold"] * 4, config, log_path=log)
    before = params_of(again.model)
    with pytest.raises(ValueError, match="decision log"):
        again.load_state_dict(state)
    assert_params(again.model, before)
    assert again.decision_index == 0 and again.update_index == 0 and again.controller.holds == 0
    assert log.read_text(encoding="utf-8") == content


def test_horizon_check_uses_the_verdict_ledger(sampler: ItemSampler, tok: Any) -> None:
    """Regression: the start-of-run horizon is computed with the alpha of the ledger the verdicts
    actually spend from (an explicit gate ledger is authoritative), not ``config.alpha``."""
    config = base_config(
        alpha_schedule="spending", shrink_lambdas=(), max_group_rollbacks=0, alarm_triggers_decision=False
    )
    for ledger, horizon in ((None, 1), (AlphaLedger(0.001, "spending"), 0)):
        model = build_tiny_llama(vocab(tok), seed=0)
        op = LMStepOperator(model, torch.optim.AdamW(model.parameters(), lr=1e-2), vocab(tok))
        gate = CapabilityGate(config, ledger)
        probe = make_probe(sampler, ScriptedScore(["pass"]), config)
        ctl = GateController(gate, probe, rollback_unit=RollbackUnit.PARAMETERS)
        loop = GatedLoop(model, op, ctl, DisagreementAlarm(min_rollouts=8), config=config)
        assert loop.horizon() == horizon and loop.required_verdicts(2) == 1
        if horizon:
            assert loop.run(2)[0].reason == REASON_ACCEPT
        else:
            with pytest.raises(ValueError, match=r"alpha=0\.001"):
                loop.run(2)
            assert op.steps == 0
