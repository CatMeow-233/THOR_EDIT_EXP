"""Tests of research_rl.capability.gate: config validation / serialisation and the pure verdict."""

from __future__ import annotations

import dataclasses
import json
import math
import time

import pytest

from research_rl.capability.gate import CapabilityGate, GateConfig
from research_rl.capability.stats import AlphaLedger, normal_cdf, sign_test_p_value, tango_z
from research_rl.capability.types import GateEvidence, PairedScores, RollbackUnit, Split


def paired(name: str, split: Split, *, n01: int, n10: int, n11: int, n00: int) -> PairedScores:
    """Binary paired scores with the given cells; uids are prefixed with the slice name."""
    inc = [0.0] * n01 + [1.0] * n10 + [1.0] * n11 + [0.0] * n00
    cand = [1.0] * n01 + [0.0] * n10 + [1.0] * n11 + [0.0] * n00
    return PairedScores(name, split, tuple(f"{name}#{i}" for i in range(len(inc))), tuple(inc), tuple(cand))


def clear_gain_transfer() -> tuple[PairedScores, ...]:
    return (
        paired("t_rule/unseen_pairs", Split.T_RULE, n01=40, n10=4, n11=60, n00=96),
        paired("t_task/argmax", Split.T_TASK, n01=25, n10=3, n11=40, n00=132),
    )


def clean_retention(n: int = 400) -> tuple[PairedScores, ...]:
    return (
        paired("retention/atomic", Split.RETENTION, n01=1, n10=1, n11=n - 50, n00=48),
        paired("retention/copy", Split.RETENTION, n01=0, n10=0, n11=n - 10, n00=10),
    )


# ============================================================================== GateConfig
def test_gate_config_defaults_match_contract() -> None:
    cfg = GateConfig()
    assert cfg.enabled is True
    assert cfg.alpha == 0.05
    assert cfg.alpha_schedule == "spending"
    assert cfg.retention_margin == 0.02
    assert cfg.retention_margins == {}
    assert cfg.require_transfer_gain is True
    assert cfg.hold_limit == 3
    assert cfg.shrink_lambdas == (0.5, 0.25)
    assert cfg.max_group_rollbacks == 4
    assert cfg.rollback_unit is RollbackUnit.PARAMETERS
    assert cfg.block_updates >= 1
    assert cfg.items_per_transfer_slice >= 1 and cfg.items_per_retention_slice >= 1
    assert cfg.alarm_triggers_decision is True


def test_gate_config_json_round_trip() -> None:
    cfg = GateConfig(
        enabled=False,
        alpha=0.1,
        alpha_schedule="per_decision",
        retention_margins={"retention/copy": 0.0, "retention/atomic": 0.05},
        shrink_lambdas=[0.75, 0.5, 0.125],  # type: ignore[arg-type]  # lists are normalised
        rollback_unit="channels",  # type: ignore[arg-type]  # strings are normalised
        block_updates=3,
        items_per_transfer_slice=64,
        items_per_retention_slice=900,
    )
    assert cfg.shrink_lambdas == (0.75, 0.5, 0.125)
    assert cfg.rollback_unit is RollbackUnit.CHANNELS
    data = json.loads(json.dumps(cfg.to_dict()))
    assert data["rollback_unit"] == "channels"
    assert GateConfig.from_dict(data) == cfg
    assert GateConfig.from_dict({}) == GateConfig()
    revalidated = dataclasses.replace(cfg, alpha=0.01)
    assert revalidated.alpha == 0.01
    with pytest.raises(ValueError):
        dataclasses.replace(cfg, alpha=0.7)
    with pytest.raises(ValueError, match="unknown GateConfig keys"):
        GateConfig.from_dict({"alpha": 0.05, "margin": 0.02})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.0},
        {"alpha": 0.5},
        {"alpha": math.nan},
        {"alpha": True},
        {"alpha_schedule": "bonferroni"},
        {"retention_margin": -0.01},
        {"retention_margin": 1.0},
        {"retention_margins": {"retention/copy": 1.5}},
        {"retention_margins": {"": 0.01}},
        {"hold_limit": 0},
        {"hold_limit": 2.0},
        {"shrink_lambdas": (0.5, 1.0)},
        {"shrink_lambdas": (0.0,)},
        {"max_group_rollbacks": -1},
        {"rollback_unit": "layers"},
        {"block_updates": 0},
        {"items_per_transfer_slice": 0},
        {"items_per_retention_slice": -5},
        {"enabled": 1},
        {"require_transfer_gain": "yes"},
    ],
)
def test_gate_config_rejects_out_of_range(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GateConfig(**kwargs)  # type: ignore[arg-type]


def test_gate_config_margin_override_and_empty_shrink() -> None:
    cfg = GateConfig(retention_margins={"retention/copy": 0.0}, shrink_lambdas=())
    assert cfg.margin_for("retention/copy") == 0.0
    assert cfg.margin_for("retention/atomic") == 0.02
    assert cfg.shrink_lambdas == ()
    ledger = cfg.make_ledger()
    assert (ledger.total_alpha, ledger.schedule) == (0.05, "spending")


# ============================================================================== verdict
def test_clear_gain_and_no_drop_passes_both() -> None:
    gate = CapabilityGate(GateConfig(), AlphaLedger(0.05, "spending"))
    verdict = gate.verdict(GateEvidence(0, clear_gain_transfer(), clean_retention()))
    assert verdict.transfer_ok
    assert verdict.retention_ok
    assert verdict.transfer.n == 400
    assert verdict.transfer.slice == "t_rule/unseen_pairs+t_task/argmax"
    assert verdict.transfer.p_value == sign_test_p_value(65, 7)
    assert verdict.transfer.bound > 0.0
    assert [t.slice for t in verdict.retention] == ["retention/atomic", "retention/copy"]
    assert all(t.bound < t.margin for t in verdict.retention if t.margin is not None)
    assert gate.accepts(verdict)
    json.dumps(verdict.to_dict())  # JSON-able


def test_large_drop_on_one_slice_fails_retention() -> None:
    gate = CapabilityGate(GateConfig())
    retention = (
        clean_retention()[0],
        paired("retention/copy", Split.RETENTION, n01=2, n10=40, n11=300, n00=58),  # ~9.5 point drop
    )
    verdict = gate.verdict(GateEvidence(0, clear_gain_transfer(), retention))
    assert verdict.transfer_ok
    assert not verdict.retention_ok
    by_slice = {t.slice: t for t in verdict.retention}
    assert by_slice["retention/atomic"].passed
    assert not by_slice["retention/copy"].passed
    assert by_slice["retention/copy"].point == pytest.approx(38 / 400)
    assert by_slice["retention/copy"].bound > 0.02
    assert not gate.accepts(verdict)


def test_no_gain_fails_transfer() -> None:
    gate = CapabilityGate(GateConfig())
    transfer = (
        paired("t_rule/unseen_pairs", Split.T_RULE, n01=10, n10=10, n11=50, n00=130),
        paired("t_rule/depth3", Split.T_RULE, n01=4, n10=6, n11=20, n00=170),
    )
    verdict = gate.verdict(GateEvidence(0, transfer, clean_retention()))
    assert not verdict.transfer_ok
    assert verdict.retention_ok
    assert verdict.transfer.bound <= 0.0
    assert not gate.accepts(verdict)
    relaxed = CapabilityGate(GateConfig(require_transfer_gain=False))
    assert relaxed.accepts(relaxed.verdict(GateEvidence(0, transfer, clean_retention())))


def test_alarm_blocks_accept_but_not_statistics() -> None:
    gate = CapabilityGate(GateConfig())
    verdict = gate.verdict(GateEvidence(2, clear_gain_transfer(), clean_retention(), alarm=True))
    assert verdict.alarm and verdict.transfer_ok and verdict.retention_ok
    assert verdict.decision_index == 2
    assert not gate.accepts(verdict)


def test_one_alpha_per_verdict_shared_by_all_tests() -> None:
    ledger = AlphaLedger(0.05, "spending")
    gate = CapabilityGate(GateConfig(), ledger)
    evidence = GateEvidence(0, clear_gain_transfer(), clean_retention())
    first = gate.verdict(evidence)
    second = gate.verdict(evidence)
    expected = [AlphaLedger(0.05, "spending").alpha_at(k) for k in (1, 2)]
    for verdict, alpha in zip((first, second), expected):
        assert verdict.transfer.alpha == alpha
        assert all(t.alpha == alpha for t in verdict.retention)
    assert ledger.count == 2 and gate.verdict_count == 2
    assert ledger.spent == pytest.approx(sum(expected))


def test_per_slice_margin_override_applied() -> None:
    cfg = GateConfig(retention_margins={"retention/copy": 0.08})
    gate = CapabilityGate(cfg)
    retention = (paired("retention/copy", Split.RETENTION, n01=0, n10=12, n11=380, n00=8),)  # 3-point drop
    verdict = gate.verdict(GateEvidence(0, clear_gain_transfer(), retention))
    (copy,) = verdict.retention
    assert copy.margin == 0.08 and copy.passed
    strict = CapabilityGate(GateConfig()).verdict(GateEvidence(0, clear_gain_transfer(), retention))
    assert strict.retention[0].margin == 0.02 and not strict.retention[0].passed


def test_duplicate_uids_across_transfer_slices_rejected_without_spending_alpha() -> None:
    gate = CapabilityGate(GateConfig())
    a = paired("t_rule/unseen_pairs", Split.T_RULE, n01=5, n10=0, n11=0, n00=5)
    b = PairedScores("t_task/argmax", Split.T_TASK, a.uids[:3], (0.0,) * 3, (1.0,) * 3)
    with pytest.raises(ValueError, match="appears in slices"):
        gate.verdict(GateEvidence(0, (a, b), clean_retention()))
    assert gate.ledger.count == 0 and gate.verdict_count == 0


def test_no_transfer_slices_gives_failing_empty_test() -> None:
    gate = CapabilityGate(GateConfig())
    verdict = gate.verdict(GateEvidence(0, (), clean_retention()))
    assert verdict.transfer.n == 0 and not verdict.transfer_ok and verdict.transfer.method == "empty"
    assert gate.ledger.count == 1


def test_empty_retention_slice_fails() -> None:
    gate = CapabilityGate(GateConfig())
    empty = PairedScores("retention/atomic", Split.RETENTION, (), (), ())
    verdict = gate.verdict(GateEvidence(0, clear_gain_transfer(), (empty,)))
    assert not verdict.retention_ok


def test_disabled_gate_still_computes_and_consumes_alpha() -> None:
    evidence = GateEvidence(0, clear_gain_transfer(), clean_retention())
    on = CapabilityGate(GateConfig())
    off = CapabilityGate(GateConfig(enabled=False))
    v_on, v_off = on.verdict(evidence), off.verdict(evidence)
    assert v_on == v_off
    assert off.ledger.count == 1


def test_explicit_ledger_is_authoritative() -> None:
    gate = CapabilityGate(GateConfig(), AlphaLedger(0.1, "per_decision"))
    verdict = gate.verdict(GateEvidence(0, clear_gain_transfer(), clean_retention()))
    assert verdict.transfer.alpha == 0.1
    assert all(t.alpha == 0.1 for t in verdict.retention)
    assert gate.state_dict()["ledger"]["schedule"] == "per_decision"
    with pytest.raises(TypeError):
        CapabilityGate({"alpha": 0.05})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CapabilityGate(GateConfig(), 0.05)  # type: ignore[arg-type]


def test_no_retention_slices_is_vacuously_ok() -> None:
    gate = CapabilityGate(GateConfig())
    verdict = gate.verdict(GateEvidence(0, clear_gain_transfer(), ()))
    assert verdict.retention == () and verdict.retention_ok


def test_gate_state_dict_round_trip() -> None:
    evidence = GateEvidence(0, clear_gain_transfer(), clean_retention())
    gate = CapabilityGate(GateConfig(retention_margins={"retention/copy": 0.03}))
    for _ in range(3):
        gate.verdict(evidence)
    state = json.loads(json.dumps(gate.state_dict()))
    restored = CapabilityGate(GateConfig(retention_margins={"retention/copy": 0.03}))
    restored.load_state_dict(state)
    assert restored.verdict_count == 3 and restored.ledger.count == 3
    assert restored.verdict(evidence) == gate.verdict(evidence)
    other = CapabilityGate(dataclasses.replace(GateConfig(), retention_margin=0.05))
    with pytest.raises(ValueError, match="config mismatch"):
        other.load_state_dict(state)


def test_default_config_keeps_retention_passable_over_many_verdicts() -> None:
    """Regression: with 512 default items a clean slice failed every verdict from k = 5..8 on."""
    cfg = GateConfig()
    assert cfg.items_per_retention_slice == 2048
    n = cfg.items_per_retention_slice
    gate = CapabilityGate(cfg)
    retention = (
        paired("retention/atomic", Split.RETENTION, n01=1, n10=1, n11=n - 102, n00=100),
        paired("retention/copy", Split.RETENTION, n01=0, n10=1, n11=n - 100, n00=99),
    )
    evidence = GateEvidence(0, clear_gain_transfer(), retention)
    assert all(gate.verdict(evidence).retention_ok for _ in range(40))
    # The old 512-item default fails from the 5th verdict with one incumbent win.
    small = CapabilityGate(dataclasses.replace(cfg, items_per_retention_slice=512))
    slice_512 = (paired("retention/atomic", Split.RETENTION, n01=0, n10=1, n11=411, n00=100),)
    oks = [small.verdict(GateEvidence(0, clear_gain_transfer(), slice_512)).retention_ok for _ in range(6)]
    assert oks == [True] * 4 + [False] * 2


def test_retention_horizon_matches_docstring_and_verdicts() -> None:
    cfg = GateConfig()
    assert [cfg.retention_horizon(n10=k) for k in (0, 1, 2, 4)] == [24495, 14624, 8838, 3348]
    assert [cfg.retention_horizon(n=512, n10=k) for k in (0, 1, 2, 4)] == [7, 4, 2, 1]
    assert cfg.retention_horizon(n=512, n01=1, n10=1) == 5
    assert cfg.retention_horizon(n=50) == 0  # below the power floor even at the first alpha
    assert cfg.retention_horizon(n=10**7) is None  # p-value underflows: no practical limit
    # The horizon is exactly where real verdicts start failing.
    for n10 in (0, 1):
        k_max = cfg.retention_horizon(n=512, n10=n10)
        assert k_max is not None
        gate = CapabilityGate(cfg)
        ev = GateEvidence(
            0, clear_gain_transfer(), (paired("r", Split.RETENTION, n01=0, n10=n10, n11=412, n00=100 - n10),)
        )
        oks = [gate.verdict(ev).retention_ok for _ in range(k_max + 2)]
        assert oks == [True] * k_max + [False] * 2
    override = GateConfig(retention_margins={"retention/copy": 0.05})
    wide = override.retention_horizon("retention/copy", n=512)
    assert wide is not None and wide > 500  # margin 0.05 instead of 0.02
    assert override.retention_horizon("retention/atomic", n=512) == 7
    # Regression: huge-but-finite horizons used to loop for ~1e9 steps (k beyond float precision).
    start = time.perf_counter()
    assert GateConfig(retention_margin=0.1).retention_horizon() is None
    assert cfg.retention_horizon(n=16384) is None
    assert time.perf_counter() - start < 1.0
    mid = GateConfig(retention_margin=0.05)
    k = mid.retention_horizon()
    assert k is not None and k > 10**6
    ledger = mid.make_ledger()
    p = normal_cdf(tango_z(0, 0, mid.items_per_retention_slice, 0.05))
    assert ledger.alpha_at(k) >= p > ledger.alpha_at(k + 1)
    per = GateConfig(alpha_schedule="per_decision")
    assert per.retention_horizon(n=512) is None and per.retention_horizon(n=50) == 0
    for bad in ({"n": 0}, {"n10": -1}, {"n": 10, "n10": 6, "n01": 5}, {"n01": 1.5}):
        with pytest.raises(ValueError):
            cfg.retention_horizon(**bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("transfer_split", "retention_split"),
    [
        (Split.AUDIT, Split.RETENTION),
        (Split.TRAIN, Split.RETENTION),
        (Split.RETENTION, Split.RETENTION),
        (Split.T_RULE, Split.T_RULE),
        (Split.T_TASK, Split.AUDIT),
    ],
)
def test_verdict_rejects_wrong_splits_without_spending_alpha(
    transfer_split: Split, retention_split: Split
) -> None:
    """Regression: AUDIT / TRAIN items used to be accepted silently as transfer evidence."""
    gate = CapabilityGate(GateConfig())
    transfer = (paired("x/transfer", transfer_split, n01=40, n10=2, n11=50, n00=108),)
    retention = (paired("x/retention", retention_split, n01=0, n10=0, n11=300, n00=100),)
    with pytest.raises(ValueError, match="split"):
        gate.verdict(GateEvidence(0, transfer, retention))
    assert gate.ledger.count == 0 and gate.verdict_count == 0


def test_verdict_rejects_uid_overlap_between_transfer_and_retention() -> None:
    gate = CapabilityGate(GateConfig())
    shared = paired("t_rule/unseen_pairs", Split.T_RULE, n01=0, n10=0, n11=300, n00=100)
    as_retention = PairedScores(
        "retention/atomic", Split.RETENTION, shared.uids, shared.incumbent, shared.candidate
    )
    with pytest.raises(ValueError, match="appears in slices"):
        gate.verdict(GateEvidence(0, (shared,), (as_retention,)))
    a = paired("retention/atomic", Split.RETENTION, n01=0, n10=0, n11=5, n00=0)
    b = PairedScores("retention/copy", Split.RETENTION, a.uids, a.incumbent, a.candidate)
    with pytest.raises(ValueError, match="appears in slices"):
        gate.verdict(GateEvidence(0, clear_gain_transfer(), (a, b)))
    assert gate.ledger.count == 0


def test_gate_load_state_dict_is_atomic_and_checks_verdict_count() -> None:
    """Regression: verdicts > ledger count was accepted; a failed load could half-apply."""
    gate = CapabilityGate(GateConfig())
    evidence = GateEvidence(0, clear_gain_transfer(), clean_retention())
    gate.verdict(evidence)
    before = gate.state_dict()
    too_many = json.loads(json.dumps(before))
    too_many["verdicts"] = 5
    bad_ledger = json.loads(json.dumps(before))
    bad_ledger["verdicts"] = 0
    bad_ledger["ledger"]["count"] = 3  # spent no longer matches the count
    for state in (too_many, bad_ledger):
        with pytest.raises(ValueError):
            gate.load_state_dict(state)
        assert gate.state_dict() == before
    # A pre-used explicit ledger: fewer verdicts than ledger alphas is legitimate.
    ledger = AlphaLedger(0.05, "spending")
    ledger.next_alpha()
    shared = CapabilityGate(GateConfig(), ledger)
    shared.verdict(evidence)
    restored = CapabilityGate(GateConfig())
    restored.load_state_dict(shared.state_dict())
    assert (restored.verdict_count, restored.ledger.count) == (1, 2)


def test_verdict_non_binary_scores_use_fractional_methods() -> None:
    gate = CapabilityGate(GateConfig(alpha_schedule="per_decision"))
    n = 400
    inc = tuple([0.5] * n)
    transfer = (
        PairedScores(
            "t_rule/unseen_pairs",
            Split.T_RULE,
            tuple(f"t{i}" for i in range(n)),
            inc,
            tuple([0.5] * 300 + [1.0] * 100),
        ),
    )
    retention = (
        PairedScores("retention/atomic", Split.RETENTION, tuple(f"r{i}" for i in range(n)), inc, inc),
    )
    verdict = gate.verdict(GateEvidence(0, transfer, retention))
    assert verdict.transfer.method == "sign_test_fractional" and verdict.transfer_ok
    assert verdict.retention[0].method == "tango_score_fractional" and verdict.retention_ok


def test_per_decision_schedule_uses_constant_alpha() -> None:
    gate = CapabilityGate(GateConfig(alpha_schedule="per_decision"))
    evidence = GateEvidence(0, clear_gain_transfer(), clean_retention())
    alphas = [gate.verdict(evidence).transfer.alpha for _ in range(4)]
    assert alphas == [0.05] * 4
