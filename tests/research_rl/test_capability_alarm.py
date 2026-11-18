"""Tests of research_rl.capability.alarm: CUSUM formula, simulated false-alarm rate and detection
delay, length collapse, persistence, screening and the static leak audit.

The simulations draw binomial counts from seeded ``torch.Generator`` streams (no models): per
update, V2 passes ~ Bin(n, p) and V2-failing rollouts pass V1 with probability ``c / (1 - p)``, so
the expected V1-only share (the gap) is ``c``. The alarm's defaults were chosen on other seeds than
the ones used here.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any

import pytest
import torch

from research_rl.capability.alarm import (
    UNITS_KEY,
    AlarmState,
    DisagreementAlarm,
    LeakAudit,
    screen_mask,
    static_leak_audit,
    strict_math_verifier,
)
from research_rl.capability.types import OperatorReport
from research_rl.rewards.math_verifier import MathVerifier
from research_rl.rewards.verifier_interface import BaseVerifier, Status

VerifyOut = tuple[Status, str | None, str | None, dict[str, Any]]

# Documented behaviour of the default parameters (module docstring of alarm.py).
CLEAN_LEVEL = 0.02  # share of 200-update clean streams with any gap alarm
DELAY_BOUND = 40  # the first alarm on a 0 -> 0.5 linear leak fires at update index <= this ...
DELAY_SHARE = 0.99  # ... in at least this share of streams
CLEAN_REPS = 1000
LEAK_REPS = 1000


def mc_tol(level: float, reps: int) -> float:
    return 3.0 * math.sqrt(level * (1.0 - level) / reps)


def make_report(
    t: int,
    n: int,
    v1_only: int,
    v2_pass: int,
    lengths: tuple[int, ...] = (),
    *,
    v2_only: int = 0,
    units: tuple[int | str, ...] | None = None,
) -> OperatorReport:
    """Report with ``v2_pass`` rollouts passing both, ``v1_only`` passing V1 only, ``v2_only``
    passing V2 only, rest failing both; ``units`` (if given) goes to ``extra[UNITS_KEY]``."""
    rest = n - v1_only - v2_pass - v2_only
    v1 = (True,) * v2_pass + (True,) * v1_only + (False,) * v2_only + (False,) * rest
    v2 = (True,) * v2_pass + (False,) * v1_only + (True,) * v2_only + (False,) * rest
    reward = sum(v1) / max(n, 1)
    extra: dict[str, Any] = {} if units is None else {UNITS_KEY: units}
    return OperatorReport(t, reward, None, v1, v2, completion_lengths=lengths, extra=extra)


def simulate_counts(seed: int, reps: int, n: int, gaps: list[float], p: float = 0.3) -> torch.Tensor:
    """``(reps, T, 2)`` int tensor of (V1-only count, V2-pass count) per update."""
    g = torch.Generator().manual_seed(seed)
    steps = len(gaps)
    v2 = torch.binomial(torch.full((reps, steps), float(n)), torch.full((reps, steps), p), generator=g)
    q = torch.tensor([c / (1.0 - p) for c in gaps]).expand(reps, steps).contiguous()
    v1_only = torch.binomial(float(n) - v2, q, generator=g)
    return torch.stack([v1_only, v2], dim=-1).long()


def first_alarm(alarm: DisagreementAlarm, counts: torch.Tensor, n: int) -> int | None:
    for t, (v1_only, v2) in enumerate(counts.tolist()):
        if alarm.update(make_report(t, n, v1_only, v2)).alarm:
            return t
    return None


def reference_cusum(gaps: list[float], b: float, k: float, h: float, start: int, reset: bool) -> list[bool]:
    """Independent implementation of S(t) = max(0, S + g - b - k), fire if S > h."""
    s, fired = 0.0, []
    for t, g in enumerate(gaps):
        if t < start:
            fired.append(False)
            continue
        s = max(0.0, s + g - b - k)
        fired.append(s > h)
        if s > h and reset:
            s = 0.0
    return fired


# ============================================================================== construction
def test_defaults_and_config_roundtrip() -> None:
    alarm = DisagreementAlarm()
    cfg = alarm.config()
    assert cfg["window"] == 5 and cfg["threshold"] == 0.3 and cfg["min_rollouts"] == 32
    assert cfg["drift"] == 0.1 and cfg["baseline_updates"] == 3 and cfg["length_fraction"] == 0.5
    assert cfg["baseline_gap"] is None and cfg["baseline_length"] is None
    assert cfg["length_collapse_alarm"] is True and cfg["reset_on_alarm"] is True
    assert cfg["rollouts_per_unit"] == 1
    assert DisagreementAlarm(**json.loads(json.dumps(cfg))).config() == cfg
    assert not alarm.baseline_ready and alarm.baseline_gap == 0.0 and alarm.baseline_length is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 0},
        {"window": 2.0},
        {"threshold": 0.0},
        {"threshold": math.inf},
        {"min_rollouts": 0},
        {"min_rollouts": True},
        {"drift": -0.1},
        {"baseline_updates": 0},
        {"baseline_gap": 1.5},
        {"baseline_gap": math.nan},
        {"baseline_length": 0.0},
        {"length_fraction": 1.0},
        {"length_fraction": 0.0},
        {"length_collapse_alarm": 1},
        {"reset_on_alarm": "yes"},
        {"rollouts_per_unit": 0},
        {"rollouts_per_unit": True},
    ],
)
def test_invalid_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        DisagreementAlarm(**kwargs)


# ============================================================================== CUSUM semantics
@pytest.mark.parametrize("reset", [True, False])
def test_cusum_matches_reference_formula(reset: bool) -> None:
    rng = random.Random(7)
    n = 50
    counts = [(rng.randint(0, 3) if t < 20 else rng.randint(5, 20), rng.randint(5, 20)) for t in range(80)]
    alarm = DisagreementAlarm(min_rollouts=10, reset_on_alarm=reset)
    states = [alarm.update(make_report(t, n, a, b)) for t, (a, b) in enumerate(counts)]
    gaps = [a / n for a, _ in counts]
    b = sum(a for a, _ in counts[:3]) / (3 * n)  # pooled baseline of the first 3 reports
    expected = reference_cusum(gaps, b, 0.1, 0.3, start=3, reset=reset)
    assert [s.alarm for s in states] == expected
    assert any(expected) and not any(expected[:3])
    for t, s in enumerate(states):
        assert s.gap == pytest.approx(gaps[t])
        assert s.reasons == (("gap_cusum",) if expected[t] else ())
        assert s.length_ratio == 1.0  # no lengths in these reports
        if t >= 3:
            assert s.baseline_gap == pytest.approx(b)
    assert alarm.gap_alarms == sum(expected)  # one count per state carrying "gap_cusum"
    if reset:
        assert sum(expected) >= 2  # a persistent leak re-fires
        fire = expected.index(True)
        assert states[fire].cusum > 0.3  # shown before the reset


def test_baseline_is_pooled_over_configurable_updates_and_cusum_waits() -> None:
    alarm = DisagreementAlarm(min_rollouts=1, baseline_updates=5)
    sizes = [40, 60, 40, 60, 100]
    v1_only = [4, 0, 8, 6, 2]
    for t, (n, a) in enumerate(zip(sizes, v1_only)):
        state = alarm.update(make_report(t, n, a, 0))
        assert state.cusum == 0.0 and not state.alarm
        assert alarm.baseline_ready == (t == 4)
        assert state.baseline_gap == pytest.approx(sum(v1_only[: t + 1]) / sum(sizes[: t + 1]))
    assert alarm.baseline_gap == pytest.approx(20 / 300)
    state = alarm.update(make_report(5, 100, 50, 0))
    assert state.cusum == pytest.approx(0.5 - 20 / 300 - 0.1)
    assert state.alarm and state.reasons == ("gap_cusum",)


def test_explicit_static_baseline_from_leak_audit() -> None:
    audit = LeakAudit(n=100, v1_pass_rate=0.5, v2_pass_rate=0.4, leak_rate=0.12)
    assert audit.gap == pytest.approx(0.1)
    alarm = DisagreementAlarm(min_rollouts=1, baseline_gap=audit.gap)
    assert alarm.baseline_ready
    state = alarm.update(make_report(0, 100, 30, 10))  # gap 0.3
    assert state.baseline_gap == pytest.approx(0.1)
    assert state.cusum == pytest.approx(0.3 - 0.1 - 0.1)
    state = alarm.update(make_report(1, 100, 35, 10))  # gap 0.35: S = 0.1 + 0.15 = 0.25
    assert not state.alarm and state.cusum == pytest.approx(0.25)
    state = alarm.update(make_report(2, 100, 30, 10))  # S = 0.35 > 0.3
    assert state.alarm and alarm.cusum == 0.0


def test_skipped_reports_change_no_statistic() -> None:
    alarm = DisagreementAlarm(min_rollouts=10)
    for t in range(5):
        alarm.update(make_report(t, 20, 1, 5, lengths=(10,) * 20))
    before = alarm.state_dict()
    small = alarm.update(make_report(5, 9, 9, 0, lengths=(1,) * 9))
    empty = alarm.update(OperatorReport(6, 0.0, None, (), ()))
    after = alarm.state_dict()
    assert small.gap == 1.0 and empty.gap == 0.0
    assert not small.alarm and not empty.alarm
    assert small.cusum == empty.cusum == before["cusum"]
    assert after["updates"] == before["updates"] + 2 and after["skipped"] == before["skipped"] + 2
    for key in set(before) - {"updates", "skipped"}:
        assert after[key] == before[key], key
    assert alarm.skipped_updates == 2 and alarm.updates == 7


# ============================================================================== simulation
@pytest.mark.parametrize("n", [32, 64])
@pytest.mark.parametrize("c", [0.0, 0.05, 0.1])
def test_clean_stream_false_alarm_rate(n: int, c: float) -> None:
    counts = simulate_counts(seed=20_000 + int(1000 * c) + n, reps=CLEAN_REPS, n=n, gaps=[c] * 200)
    alarms = sum(first_alarm(DisagreementAlarm(), counts[r], n) is not None for r in range(CLEAN_REPS))
    rate = alarms / CLEAN_REPS
    assert rate <= CLEAN_LEVEL + mc_tol(CLEAN_LEVEL, CLEAN_REPS), (n, c, rate)


@pytest.mark.parametrize("n", [32, 64])
def test_leaking_stream_detection_delay(n: int) -> None:
    gaps = [0.5 * t / 99 for t in range(100)]
    counts = simulate_counts(seed=30_000 + n, reps=LEAK_REPS, n=n, gaps=gaps)
    delays = [first_alarm(DisagreementAlarm(), counts[r], n) for r in range(LEAK_REPS)]
    within = sum(d is not None and d <= DELAY_BOUND for d in delays) / LEAK_REPS
    assert within >= DELAY_SHARE - mc_tol(DELAY_SHARE, LEAK_REPS), (n, within)
    assert all(d is None or d >= 3 for d in delays)  # never during the baseline period


def simulate_two_sided(seed: int, reps: int, n: int, p10: float, p01: float, steps: int) -> torch.Tensor:
    """``(reps, T, 2)`` multinomial (V1-only count, V2-only count) per update."""
    g = torch.Generator().manual_seed(seed)
    probs = torch.tensor([p10, p01, 1.0 - p10 - p01])
    draws = torch.multinomial(probs, reps * steps * n, replacement=True, generator=g)
    draws = draws.view(reps, steps, n)
    return torch.stack([(draws == 0).sum(-1), (draws == 1).sum(-1)], dim=-1)


@pytest.mark.parametrize(("n", "p10", "p01"), [(32, 0.05, 0.05), (64, 0.1, 0.1), (64, 0.15, 0.05)])
def test_two_sided_clean_stream_within_documented_scope(n: int, p10: float, p01: float) -> None:
    sigma2 = p10 + p01 - (p10 - p01) ** 2
    assert sigma2 / n <= 0.1 / 32 + 1e-12  # inside the documented scope
    reps = 500
    counts = simulate_two_sided(50_000 + n + int(100 * p10), reps, n, p10, p01, 200)
    alarms = 0
    for r in range(reps):
        alarm = DisagreementAlarm()
        for t, (a, b) in enumerate(counts[r].tolist()):
            if alarm.update(make_report(t, n, a, 0, v2_only=b)).alarm:
                alarms += 1
                break
    assert alarms / reps <= CLEAN_LEVEL + mc_tol(CLEAN_LEVEL, reps), (n, p10, p01, alarms)


# ---------------------------------------------------------------- clustered (correlated) rollouts
def simulate_replicated(
    seed: int, reps: int, members: int, items: int, gaps: list[float], flip: float
) -> torch.Tensor:
    """ES-like ``(reps, T, members, items)`` V1-only outcomes: every member scores the same items;
    a member reproduces the item's outcome exactly, except that with probability ``flip`` it
    re-draws it independently (V2 always fails)."""
    g = torch.Generator().manual_seed(seed)
    steps = len(gaps)
    c = torch.tensor(gaps).view(1, steps, 1, 1)
    item = torch.rand(reps, steps, 1, items, generator=g) < c
    redraw = torch.rand(reps, steps, members, items, generator=g) < flip
    fresh = torch.rand(reps, steps, members, items, generator=g) < c
    return torch.where(redraw, fresh, item.expand(reps, steps, members, items))


def replicated_report(t: int, outcomes: torch.Tensor, with_ids: bool) -> OperatorReport:
    """Member-major concatenation (as ESOperator reports); ids = item index if ``with_ids``."""
    members, items = outcomes.shape
    v1 = tuple(bool(x) for x in outcomes.reshape(-1).tolist())
    extra: dict[str, Any] = {UNITS_KEY: tuple(range(items)) * members} if with_ids else {}
    return OperatorReport(t, 0.0, None, v1, (False,) * len(v1), extra=extra)


def test_replicated_rollouts_are_not_counted_as_independent() -> None:
    """Regression (review): 8 ES members x 4 shared items look like 32 rollouts but are 4 units."""
    reps, members, items = 100, 8, 4
    outcomes = simulate_replicated(61, reps, members, items, [0.1] * 200, flip=0.0)
    naive = by_ids = by_count = 0
    for r in range(reps):
        alarms = [DisagreementAlarm(), DisagreementAlarm(), DisagreementAlarm(rollouts_per_unit=members)]
        hit = [False, False, False]
        for t in range(200):
            reports = (
                replicated_report(t, outcomes[r, t], False),
                replicated_report(t, outcomes[r, t], True),
                replicated_report(t, outcomes[r, t], False),
            )
            for i, (alarm, report) in enumerate(zip(alarms, reports)):
                hit[i] |= alarm.update(report).alarm
        naive += hit[0]
        by_ids += hit[1]
        by_count += hit[2]
        assert alarms[1].skipped_updates == alarms[2].skipped_updates == 200  # n_eff = 4 < 32
        assert alarms[1].state_dict()["eligible"] == 0
    assert naive / reps > 0.5  # without unit information the level is badly violated
    assert by_ids == by_count == 0


@pytest.mark.parametrize("flip", [0.0, 0.5])
def test_clustered_clean_stream_with_32_units_keeps_level(flip: float) -> None:
    reps, members, items = 500, 4, 32
    outcomes = simulate_replicated(70 + int(10 * flip), reps, members, items, [0.1] * 200, flip)
    counts = outcomes.sum(dim=(-1, -2)).tolist()  # rollouts_per_unit needs only the counts
    n = members * items
    alarms = 0
    for r in range(reps):
        alarm = DisagreementAlarm(rollouts_per_unit=members)
        if first_alarm_counts(alarm, counts[r], n) is not None:
            alarms += 1
        assert alarm.skipped_updates == 0  # n_eff = 32
    assert alarms / reps <= CLEAN_LEVEL + mc_tol(CLEAN_LEVEL, reps), (flip, alarms)


@pytest.mark.parametrize("flip", [0.0, 0.5])
def test_clustered_leak_detected_within_documented_delay(flip: float) -> None:
    reps, members, items = 500, 4, 32
    gaps = [0.5 * t / 99 for t in range(100)]
    outcomes = simulate_replicated(80 + int(10 * flip), reps, members, items, gaps, flip)
    counts = outcomes.sum(dim=(-1, -2)).tolist()
    n = members * items
    delays = [
        first_alarm_counts(DisagreementAlarm(rollouts_per_unit=members), counts[r], n) for r in range(reps)
    ]
    within = sum(d is not None and d <= DELAY_BOUND for d in delays) / reps
    assert within >= DELAY_SHARE - mc_tol(DELAY_SHARE, reps), within
    assert all(d is None or d >= 3 for d in delays)


def first_alarm_counts(alarm: DisagreementAlarm, counts: list[int], n: int) -> int | None:
    for t, v1_only in enumerate(counts):
        if alarm.update(make_report(t, n, v1_only, 0)).alarm:
            return t
    return None


def test_effective_size_is_kish_and_gates_eligibility() -> None:
    alarm = DisagreementAlarm(min_rollouts=4)
    iid = make_report(0, 6, 1, 1)
    assert alarm.effective_size(iid) == 6.0
    unequal = make_report(0, 6, 1, 1, units=("a", "a", "a", 1, 2, 2))  # sizes 3, 1, 2
    assert alarm.effective_size(unequal) == pytest.approx(36 / 14)
    assert alarm.effective_size(OperatorReport(0, 0.0, None, (), ())) == 0.0
    assert DisagreementAlarm(rollouts_per_unit=3).effective_size(iid) == 2.0
    # ids take precedence over rollouts_per_unit
    assert (
        DisagreementAlarm(rollouts_per_unit=3).effective_size(make_report(0, 6, 0, 0, units=tuple(range(6))))
        == 6.0
    )
    # exact boundary: n_eff == min_rollouts is eligible, just below is skipped
    alarm.update(make_report(0, 8, 0, 0, units=(0, 0, 1, 1, 2, 2, 3, 3)))  # n_eff = 4
    assert alarm.skipped_updates == 0
    alarm.update(make_report(1, 8, 0, 0, units=(0, 0, 0, 1, 1, 2, 2, 3)))  # 64 / 18 < 4
    assert alarm.skipped_updates == 1
    gap_only = make_report(2, 8, 4, 0, units=(0,) * 8)  # one unit: skipped, gap still reported
    assert alarm.update(gap_only).gap == 0.5 and alarm.skipped_updates == 2


def test_unit_information_does_not_change_the_statistic() -> None:
    """Same counts with and without ids: identical states while every report is eligible."""
    rng = random.Random(5)
    plain, with_ids = DisagreementAlarm(min_rollouts=8), DisagreementAlarm(min_rollouts=8)
    for t in range(40):
        a = rng.randint(0, 6 if t < 20 else 30)
        plain_state = plain.update(make_report(t, 64, a, 10))
        ids_state = with_ids.update(make_report(t, 64, a, 10, units=tuple(i // 8 for i in range(64))))
        assert plain_state == ids_state


@pytest.mark.parametrize(
    ("units", "n"),
    [(("a",) * 3, 4), ((True, False, True, False), 4), ((1.0, 2.0, 3.0, 4.0), 4), ("abcd", 4)],
)
def test_malformed_unit_ids_rejected_without_state_change(units: Any, n: int) -> None:
    alarm = DisagreementAlarm(min_rollouts=1)
    report = OperatorReport(0, 0.0, None, (True,) * n, (False,) * n, extra={UNITS_KEY: units})
    with pytest.raises(ValueError):
        alarm.update(report)
    assert alarm.updates == 0


def test_rollouts_per_unit_must_divide_the_report() -> None:
    alarm = DisagreementAlarm(rollouts_per_unit=4, min_rollouts=1)
    with pytest.raises(ValueError, match="multiple"):
        alarm.update(make_report(0, 6, 1, 1))
    assert alarm.updates == 0
    alarm.update(make_report(0, 8, 1, 1))  # 8 rollouts = 2 units of 4: eligible for min_rollouts=2
    assert alarm.updates == 1 and alarm.skipped_updates == 0
    strict = DisagreementAlarm(rollouts_per_unit=4, min_rollouts=3)
    assert strict.update(make_report(0, 8, 1, 1)).gap == pytest.approx(1 / 8)
    assert strict.skipped_updates == 1  # n_eff = 2 < 3


def test_noise_free_leak_fires_at_documented_update() -> None:
    n = 200
    alarm = DisagreementAlarm()
    fired = None
    for t in range(100):
        v1_only = round(0.5 * t / 99 * n)
        if alarm.update(make_report(t, n, v1_only, 60)).alarm:
            fired = t
            break
    assert fired == 32  # documented in the module docstring (gap 0.16)


def test_recovery_after_fire_and_refire_on_persistent_leak() -> None:
    n = 64
    leak = simulate_counts(seed=41, reps=1, n=n, gaps=[0.02] * 10 + [0.4] * 10)[0]
    clean = simulate_counts(seed=42, reps=1, n=n, gaps=[0.02] * 100)[0]
    alarm = DisagreementAlarm()
    leak_states = [alarm.update(make_report(t, n, a, b)) for t, (a, b) in enumerate(leak.tolist())]
    assert sum(s.alarm for s in leak_states) >= 2  # persistent leak re-fires after re-accumulating
    clean_states = [alarm.update(make_report(20 + t, n, a, b)) for t, (a, b) in enumerate(clean.tolist())]
    assert not any(s.alarm for s in clean_states)  # pulse mode: off once back at baseline


def test_latched_mode_stays_on_then_decays() -> None:
    # Dyadic values keep the arithmetic exact: drift 1/8, threshold 1/4, gap 1/2.
    alarm = DisagreementAlarm(
        min_rollouts=1, drift=0.125, threshold=0.25, baseline_gap=0.0, reset_on_alarm=False
    )
    states = [alarm.update(make_report(t, 8, 4, 0)) for t in range(2)]  # S = 0.375, 0.75
    assert [s.alarm for s in states] == [True, True]
    skipped = alarm.update(OperatorReport(2, 0.0, None, (), ()))
    assert skipped.alarm and skipped.cusum == 0.75  # latched: on while S > h, even when skipped
    on = []
    for t in range(3, 10):
        state = alarm.update(make_report(t, 8, 0, 0))  # S decreases by 0.125 per update
        on.append(state.alarm)
    assert on == [True] * 3 + [False] * 4  # 0.625, 0.5, 0.375 > 0.25; 0.25 is not
    assert alarm.gap_alarms == 6  # 2 + the skipped report + 3: every state with "gap_cusum"


def test_gap_alarm_counter_matches_states_in_latched_mode_with_skips() -> None:
    alarm = DisagreementAlarm(min_rollouts=4, baseline_gap=0.0, reset_on_alarm=False)
    stream = [(8, 8), (8, 8), (2, 2), (8, 0), (2, 0), (8, 0), (8, 0), (8, 0), (2, 0)]
    states = [alarm.update(make_report(t, n, a, 0)) for t, (n, a) in enumerate(stream)]
    assert sum(s.alarm for s in states[:3]) == 3  # S = 0.9, 1.8 (> 0.3); the skipped one latched
    assert alarm.gap_alarms == sum("gap_cusum" in s.reasons for s in states)


# ============================================================================== length collapse
def lengths_report(t: int, length: int, n: int = 4) -> OperatorReport:
    return make_report(t, n, 0, 0, lengths=(length,) * n)


def test_length_collapse_edge_alarm_and_level_reason() -> None:
    alarm = DisagreementAlarm(window=1, min_rollouts=1)
    for t, length in enumerate([10, 20, 30]):  # baseline = 20
        state = alarm.update(lengths_report(t, length))
        assert state.length_ratio == 1.0 and not state.alarm
    assert alarm.baseline_length == pytest.approx(20.0)
    seq = [12, 10, 9, 8, 15, 9, 9]
    states = [alarm.update(lengths_report(3 + i, length)) for i, length in enumerate(seq)]
    assert [s.length_ratio for s in states] == pytest.approx([x / 20 for x in seq])
    collapsed = [x / 20 < 0.5 for x in seq]
    assert [("length_collapse" in s.reasons) for s in states] == collapsed
    assert [s.alarm for s in states] == [False, False, True, False, False, True, False]
    assert alarm.length_alarms == 2


def test_length_collapse_without_alarm_switch() -> None:
    alarm = DisagreementAlarm(window=1, min_rollouts=1, length_collapse_alarm=False)
    for t in range(3):
        alarm.update(lengths_report(t, 20))
    state = alarm.update(lengths_report(3, 5))
    assert state.reasons == ("length_collapse",) and not state.alarm


def test_length_window_and_explicit_baseline() -> None:
    narrow = DisagreementAlarm(window=1, min_rollouts=1, baseline_length=20.0)
    wide = DisagreementAlarm(window=5, min_rollouts=1, baseline_length=20.0)
    seq = [20, 20, 20, 20, 2]
    s_narrow = [narrow.update(lengths_report(t, x)) for t, x in enumerate(seq)]
    s_wide = [wide.update(lengths_report(t, x)) for t, x in enumerate(seq)]
    assert s_narrow[0].length_ratio == 1.0  # explicit baseline: tracked from the first report
    assert s_narrow[-1].alarm and s_narrow[-1].length_ratio == pytest.approx(0.1)
    assert not s_wide[-1].alarm and s_wide[-1].length_ratio == pytest.approx((80 + 2) / 5 / 20)
    for t in range(5, 9):
        state = wide.update(lengths_report(t, 2))
    assert state.alarm is False and "length_collapse" in state.reasons  # rose earlier, level now
    assert wide.length_alarms == 1


def test_length_window_is_pooled_over_completions_not_reports() -> None:
    """Regression (review): total tokens / total completions, not a mean of per-report means."""
    alarm = DisagreementAlarm(window=2, min_rollouts=1, baseline_length=20.0)
    alarm.update(make_report(0, 99, 0, 0, lengths=(20,) * 99))
    state = alarm.update(make_report(1, 1, 0, 0, lengths=(2,)))
    assert state.length_ratio == pytest.approx((99 * 20 + 2) / 100 / 20)  # 0.991, not 0.55
    assert "length_collapse" not in state.reasons


def test_reports_without_lengths_leave_length_statistics() -> None:
    alarm = DisagreementAlarm(window=2, min_rollouts=1)
    for t in range(3):
        alarm.update(lengths_report(t, 10))
    alarm.update(lengths_report(3, 4))  # ratio 0.4 -> collapse
    before = alarm.state_dict()
    state = alarm.update(make_report(4, 4, 0, 0))  # no lengths
    assert state.length_ratio == pytest.approx(0.4) and "length_collapse" in state.reasons
    assert not state.alarm  # still the same episode
    after = alarm.state_dict()
    for key in ("len_sum", "len_count", "len_reports", "window", "collapsed"):
        assert after[key] == before[key]


def test_zero_length_baseline_never_flags_collapse() -> None:
    alarm = DisagreementAlarm(window=1, min_rollouts=1)
    for t in range(3):
        alarm.update(lengths_report(t, 0))
    assert alarm.length_baseline_ready and alarm.baseline_length is None  # ratio undefined
    state = alarm.update(lengths_report(3, 0))
    assert state.length_ratio == 1.0 and state.reasons == () and not state.alarm


def test_update_rejects_non_bool_pass_entries() -> None:
    report = OperatorReport(0, 0.0, None, ("yes", True), (True, True))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DisagreementAlarm(min_rollouts=1).update(report)


def test_negative_length_rejected() -> None:
    with pytest.raises(ValueError):
        DisagreementAlarm(min_rollouts=1).update(make_report(0, 2, 0, 0, lengths=(3, -1)))


# ============================================================================== persistence
def mixed_stream(seed: int, steps: int) -> list[OperatorReport]:
    rng = random.Random(seed)
    reports = []
    for t in range(steps):
        n = rng.choice([8, 40, 40, 64])
        v1_only = rng.randint(0, n // 2 if t > 15 else n // 10)
        v2 = rng.randint(0, n - v1_only)
        lengths = tuple(rng.randint(1, 30 if t < 25 else 8) for _ in range(n)) if rng.random() < 0.8 else ()
        reports.append(make_report(t, n, v1_only, v2, lengths))
    return reports


@pytest.mark.parametrize("cut", [1, 6, None])
def test_state_dict_roundtrip_reproduces_decisions(cut: int | None) -> None:
    reports = mixed_stream(3, 80)
    kwargs: dict[str, Any] = {"window": 3, "baseline_updates": 3}
    reference = DisagreementAlarm(**kwargs)
    ref_states = [reference.update(r) for r in reports]
    if cut is None:  # right after the first gap alarm, window partly filled
        cut = next(t for t, s in enumerate(ref_states) if "gap_cusum" in s.reasons) + 1
    assert any(s.alarm for s in ref_states[cut:])
    first = DisagreementAlarm(**kwargs)
    for r in reports[:cut]:
        first.update(r)
    state = json.loads(json.dumps(first.state_dict()))
    resumed = DisagreementAlarm(**kwargs)
    resumed.load_state_dict(state)
    assert resumed.state_dict() == first.state_dict()
    assert [resumed.update(r) for r in reports[cut:]] == ref_states[cut:]


def test_state_dict_roundtrip_latched_mode_above_threshold() -> None:
    kwargs: dict[str, Any] = {"min_rollouts": 1, "reset_on_alarm": False, "window": 2}
    reports = [make_report(t, 10, 5 if 4 <= t < 8 else 0, 2, lengths=(6,) * 10) for t in range(30)]
    reference = DisagreementAlarm(**kwargs)
    ref_states = [reference.update(r) for r in reports]
    first = DisagreementAlarm(**kwargs)
    for r in reports[:6]:
        first.update(r)
    assert first.cusum > first.threshold
    resumed = DisagreementAlarm(**kwargs)
    resumed.load_state_dict(json.loads(json.dumps(first.state_dict())))
    resumed_states = [resumed.update(r) for r in reports[6:]]
    assert resumed_states == ref_states[6:]
    assert resumed_states[0].alarm and not resumed_states[-1].alarm


def test_state_dict_roundtrip_inside_length_collapse_episode() -> None:
    """Regression (review): resuming mid-collapse must not raise a second rising-edge alarm."""
    kwargs: dict[str, Any] = {"window": 1, "min_rollouts": 1}
    seq = [20, 20, 20, 20, 5, 5, 5, 20, 5]
    reports = [lengths_report(t, x) for t, x in enumerate(seq)]
    reference = DisagreementAlarm(**kwargs)
    ref_states = [reference.update(r) for r in reports]
    assert [s.alarm for s in ref_states] == [False] * 4 + [True, False, False, False, True]
    first = DisagreementAlarm(**kwargs)
    for r in reports[:5]:  # cut right after the rising edge, collapse ongoing
        first.update(r)
    assert first.state_dict()["collapsed"] is True
    resumed = DisagreementAlarm(**kwargs)
    resumed.load_state_dict(json.loads(json.dumps(first.state_dict())))
    resumed_states = [resumed.update(r) for r in reports[5:]]
    assert resumed_states == ref_states[5:]
    assert not resumed_states[0].alarm and "length_collapse" in resumed_states[0].reasons
    assert resumed.length_alarms == reference.length_alarms == 2


def test_state_dict_roundtrip_with_rollouts_per_unit() -> None:
    kwargs: dict[str, Any] = {"rollouts_per_unit": 2, "min_rollouts": 16}
    rng = random.Random(9)
    reports = [make_report(t, 32 if t % 5 else 16, rng.randint(0, 3 if t < 10 else 12), 4) for t in range(40)]
    reference = DisagreementAlarm(**kwargs)
    ref_states = [reference.update(r) for r in reports]
    assert reference.skipped_updates == 8 and any(s.alarm for s in ref_states[12:])
    first = DisagreementAlarm(**kwargs)
    for r in reports[:12]:
        first.update(r)
    resumed = DisagreementAlarm(**kwargs)
    resumed.load_state_dict(json.loads(json.dumps(first.state_dict())))
    assert [resumed.update(r) for r in reports[12:]] == ref_states[12:]
    with pytest.raises(ValueError, match="config mismatch"):
        DisagreementAlarm(min_rollouts=16).load_state_dict(first.state_dict())


def test_load_state_dict_rejects_mismatch_and_garbage() -> None:
    alarm = DisagreementAlarm()
    alarm.update(make_report(0, 40, 2, 3, lengths=(5,) * 40))
    state = alarm.state_dict()
    with pytest.raises(ValueError, match="config mismatch"):
        DisagreementAlarm(threshold=0.5).load_state_dict(state)
    for key, bad in (("cusum", -1.0), ("gap_rollouts", -3), ("collapsed", 1), ("window", [[1, 2, 3]])):
        broken = dict(state, **{key: bad})
        with pytest.raises(ValueError):
            DisagreementAlarm().load_state_dict(broken)
    too_long = dict(state, window=[[1, 1]] * 6)
    with pytest.raises(ValueError):
        DisagreementAlarm().load_state_dict(too_long)


def test_alarm_state_to_dict_is_json() -> None:
    state = AlarmState(True, 0.25, 0.01, 0.4, 0.3, ("gap_cusum", "length_collapse"))
    assert json.loads(json.dumps(state.to_dict())) == {
        "alarm": True,
        "gap": 0.25,
        "baseline_gap": 0.01,
        "cusum": 0.4,
        "length_ratio": 0.3,
        "reasons": ["gap_cusum", "length_collapse"],
    }


# ============================================================================== screening
def test_screen_mask_semantics() -> None:
    v1 = [True, True, False, False]
    v2 = [True, False, True, False]
    assert screen_mask(v1, v2) == [True, False, True, True]
    assert screen_mask([1, 1], [0, 1]) == [False, True]
    assert screen_mask([], []) == []
    with pytest.raises(ValueError):
        screen_mask([True], [True, False])
    with pytest.raises(TypeError):
        screen_mask(["yes"], [True])


# ============================================================================== static audit
def test_static_leak_audit_math_lenient_vs_strict() -> None:
    v1 = MathVerifier()
    v2 = strict_math_verifier()
    completions = [
        "First \\boxed{3}, no wait: \\boxed{4}",  # two boxes, last one right: V1 pass, V2 fail
        "So the result is \\boxed{4} and I am confident about it",  # trailing text: V1 pass, V2 fail
        "The answer is 4",  # no box: V1 pass, V2 fail
        "Therefore \\boxed{4}.",  # both pass
        "Therefore \\boxed{5}",  # both fail
        "nothing here",  # both fail
    ]
    references = ["4"] * len(completions)
    for c in completions[:3]:
        assert v1.verify(c, "4").correct and not v2.verify(c, "4").correct
    assert v1.verify(completions[3], "4").correct and v2.verify(completions[3], "4").correct
    assert v2.verify(completions[0], "4").details["reason"] == "conflicting_boxes"
    assert v2.verify(completions[1], "4").details["reason"] == "trailing_text"
    audit = static_leak_audit(completions, references, v1, v2)
    assert audit.n == 6
    assert audit.v1_pass_rate == pytest.approx(4 / 6)
    assert audit.v2_pass_rate == pytest.approx(1 / 6)
    assert audit.leak_rate == pytest.approx(3 / 6)
    assert audit.gap == pytest.approx(3 / 6)
    assert json.loads(json.dumps(audit.to_dict()))["leak_rate"] == pytest.approx(0.5)


class _LastCharVerifier(BaseVerifier):
    """Toy verifier: passes when the completion's last character equals the reference's."""

    name = "last_char"

    def _verify(self, completion: str | None, reference: Any, **meta: Any) -> VerifyOut:
        text = (completion or "").strip()
        ok = bool(text) and text[-1] == str(reference)[-1]
        return ("correct" if ok else "incorrect"), text, str(reference), {}


class _ExactVerifier(BaseVerifier):
    name = "exact"

    def _verify(self, completion: str | None, reference: Any, **meta: Any) -> VerifyOut:
        text = (completion or "").strip()
        return ("correct" if text == str(reference) else "incorrect"), text, str(reference), {}


def test_static_leak_audit_any_verifier_pair_and_reverse_disagreement() -> None:
    completions = ["1234", "9934", "1235", None]
    references = ["1234", "1234", "1234", "1234"]
    audit = static_leak_audit(completions, references, _LastCharVerifier(), _ExactVerifier())
    assert (audit.v1_pass_rate, audit.v2_pass_rate, audit.leak_rate) == (0.5, 0.25, 0.25)
    # Reverse pair: V2-only passes are not leaks, and the gap becomes negative.
    reverse = static_leak_audit(completions, references, _ExactVerifier(), _LastCharVerifier())
    assert reverse.leak_rate == 0.0 and reverse.gap == pytest.approx(-0.25)
    with pytest.raises(ValueError):
        static_leak_audit([], [], _ExactVerifier(), _ExactVerifier())
    with pytest.raises(ValueError):
        static_leak_audit(["1"], [], _ExactVerifier(), _ExactVerifier())


def test_strict_math_verifier_configuration() -> None:
    v = strict_math_verifier()
    assert isinstance(v, MathVerifier)
    assert v.extractors == ("boxed",) and v.strict_boxed and v.strict_trailing and not v.strip_percent
    assert not v.verify("\\boxed{50}", "50%").correct  # percent is not stripped
    custom = strict_math_verifier(strict_trailing=False, max_answer_chars=64)
    assert not custom.strict_trailing and custom.strict_boxed and custom.max_answer_chars == 64
    assert custom.verify("\\boxed{4} trailing words", "4").correct
