"""Tests for research_rl.capability.tracker: capability points, D(t), item streams, resume.

Tiny CPU models only. The arithmetic tests use a stub score function whose pass rate per slice
is set by the test, so A / R / T_task / T_rule / D have known values.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence

import pytest
import torch
from torch import nn

from research_rl.capability.gate import GateConfig
from research_rl.capability.probe import TRACK_POOL, CapabilityProbe, SliceStart, StartMap, item_pool
from research_rl.capability.tasks import (
    CompositionSplits,
    ExactAnswerVerifier,
    ItemSampler,
    OperatorSet,
    make_score_fn,
)
from research_rl.capability.tracker import TRACK_PURPOSE, CapabilityPoint, CapabilityTracker
from research_rl.capability.types import Item, Split
from research_rl.models.tiny import build_char_tokenizer, build_tiny_llama


@pytest.fixture(scope="module")
def splits() -> CompositionSplits:
    return CompositionSplits.build(OperatorSet(seed=1, num_operators=10), seed=1)


@pytest.fixture(scope="module")
def sampler(splits: CompositionSplits) -> ItemSampler:
    return ItemSampler(splits, root_seed=0)


class SliceScore:
    """Scores 1.0 for items of the slices in ``solved``, 0.0 otherwise; records the items seen."""

    def __init__(self) -> None:
        self.solved: set[str] = set()
        self.seen: list[Item] = []

    def __call__(self, model: nn.Module, items: Sequence[Item]) -> list[float]:
        self.seen.extend(items)
        return [1.0 if it.slice in self.solved else 0.0 for it in items]


def _manual_start(sampler: ItemSampler, greedy: dict[str, tuple[float, ...]]) -> StartMap:
    slices = {
        name: SliceStart(
            slice=name,
            split=sampler.splits.slices()[name],
            uids=tuple(f"{name}|u{i}" for i in range(len(g))),
            greedy=g,
            correct_counts=tuple(int(v) for v in g),
            num_samples=1,
            k=1,
        )
        for name, g in greedy.items()
    }
    return StartMap(
        slices=slices,
        k=1,
        num_samples=1,
        temperature=1.0,
        seed=0,
        items_per_slice=2,
        max_new_tokens=8,
        splits_fingerprint=sampler.splits.fingerprint(),
        root_seed=sampler.root_seed,
    )


@pytest.fixture
def start(sampler: ItemSampler) -> StartMap:
    # T_rule(0) = mean(0.5, 0.0) = 0.25 (unweighted over slices, whatever their n)
    return _manual_start(sampler, {"t_rule/unseen_pairs": (1.0, 0.0), "t_rule/depth3": (0.0, 0.0, 0.0, 0.0)})


def test_measure_arithmetic_and_divergence(sampler: ItemSampler, start: StartMap) -> None:
    stub = SliceScore()
    tracker = CapabilityTracker(sampler, stub, start, items_per_slice=5)
    assert tracker.t_rule0 == pytest.approx(0.25)
    model = nn.Linear(2, 2)

    p0 = tracker.measure(model, 0, None)  # before any update: no reward baseline yet
    assert p0.divergence is None and p0.reward0 is None
    assert p0.t_rule == 0.0 and p0.acquisition == 0.0

    stub.solved = {"train/pairs", "retention/copy"}
    p1 = tracker.measure(model, 10, 0.2)
    assert p1.reward0 == 0.2
    assert p1.acquisition == 1.0
    assert p1.retention == {"retention/atomic": 0.0, "retention/copy": 1.0, "retention/prereq": 0.0}
    assert p1.t_task == 0.0
    # D = (0.2 - 0.2) - (0.0 - 0.25)
    assert p1.divergence == pytest.approx(0.25)

    stub.solved = {"train/pairs", "t_rule/unseen_pairs", "t_task/argmax"}
    p2 = tracker.measure(model, 20, 0.9)
    assert p2.t_rule == pytest.approx(0.5)
    assert p2.t_task == pytest.approx(0.5)  # unweighted over the two T_TASK slices
    # D = (0.9 - 0.2) - (0.5 - 0.25): reward rose more than unseen-rule capability
    assert p2.divergence == pytest.approx(0.45)
    assert p2.reward0 == 0.2 and p2.t_rule0 == pytest.approx(0.25)
    p_none = tracker.measure(model, 25, None)  # no reward for this point: no D, baseline kept
    assert p_none.divergence is None and p_none.reward0 == 0.2 and tracker.reward0 == 0.2
    assert set(p2.slices) == {s for s, sp in sampler.splits.slices().items() if sp is not Split.AUDIT}
    assert all(v == 5 for v in p2.n.values())
    assert tracker.history == [p0, p1, p2, p_none]

    d = p2.to_dict()
    assert {"A", "R", "T_task", "T_rule", "D", "step", "reward_v1"} <= set(d)
    assert CapabilityPoint.from_dict(json.loads(json.dumps(d))) == p2
    json.dumps(tracker.to_dict())
    # AUDIT is never drawn; every item came from the "track" stream of its step
    assert all(it.split is not Split.AUDIT for it in stub.seen)
    assert all(it.meta["source"] == TRACK_PURPOSE for it in stub.seen)
    for bad_step in (-1, True, 1.5):
        with pytest.raises(ValueError):
            tracker.measure(model, bad_step, 0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tracker.measure(model, 3, float("nan"))


def test_items_fresh_per_step_and_separate_from_gate(sampler: ItemSampler, start: StartMap) -> None:
    tracker = CapabilityTracker(sampler, SliceScore(), start, items_per_slice=16)
    probe = CapabilityProbe.from_config(
        sampler,
        SliceScore(),
        GateConfig(items_per_transfer_slice=16, items_per_retention_slice=16),
    )

    def track(step: int) -> set[str]:
        return {it.uid for group in tracker.items(step).values() for it in group}

    assert track(3) == track(3)  # same step -> same items
    overlap_steps = len(track(0) & track(1))
    total = len(track(0))
    assert total == 16 * len(tracker.slices)
    assert overlap_steps <= total // 10
    # tracker items come from the track pool, gate items from the gate pool: never the same item
    assert all(item_pool(sampler, u) == TRACK_POOL for u in track(0))
    overlap = checked = 0
    for index in range(4):
        tracked = track(index)
        for attempt in range(3):
            gate = {it.uid for group in probe.gate_items(index, attempt).values() for it in group}
            assert len(gate) == 16 * 7
            overlap += len(tracked & gate)
            checked += len(gate)
            # the tracker's draw of a gate slice is a different sample than the gate's
            for name in probe.transfer_slices:
                t_uids = [it.uid for it in tracker.items(index)[name]]
                g_uids = [it.uid for it in probe.gate_items(index, attempt)[name]]
                assert t_uids != g_uids
    assert checked == 4 * 3 * 16 * 7
    assert overlap == 0


def test_construction_checks(sampler: ItemSampler, splits: CompositionSplits, start: StartMap) -> None:
    tok = build_char_tokenizer()
    max_len = splits.operators.max_len
    with pytest.raises(ValueError, match="truncation contract"):
        CapabilityTracker(
            sampler,
            make_score_fn(tok, ExactAnswerVerifier(), max_new_tokens=max_len, batch_size=2),
            start,
            items_per_slice=2,
        )
    missing = _manual_start(sampler, {"t_rule/unseen_pairs": (1.0,)})
    with pytest.raises(ValueError, match="T_RULE"):
        CapabilityTracker(sampler, SliceScore(), missing, items_per_slice=2)
    other_splits = CompositionSplits.build(OperatorSet(seed=2, num_operators=10), seed=2)
    foreign = dataclasses.replace(start, splits_fingerprint=other_splits.fingerprint())
    with pytest.raises(ValueError, match="different splits"):
        CapabilityTracker(sampler, SliceScore(), foreign, items_per_slice=2)
    for bad in (0, True, 2.0):
        with pytest.raises(ValueError):
            CapabilityTracker(sampler, SliceScore(), start, items_per_slice=bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CapabilityTracker(sampler, SliceScore(), start.to_dict(), items_per_slice=2)  # type: ignore[arg-type]


def test_state_dict_round_trip(sampler: ItemSampler, start: StartMap) -> None:
    stub = SliceScore()
    tracker = CapabilityTracker(sampler, stub, start, items_per_slice=3)
    model = nn.Linear(1, 1)
    tracker.measure(model, 0, 0.4)
    stub.solved = {"t_rule/depth3"}
    tracker.measure(model, 5, 0.6)
    state = json.loads(json.dumps(tracker.state_dict()))

    resumed = CapabilityTracker(sampler, stub, start, items_per_slice=3)
    resumed.load_state_dict(state)
    assert resumed.reward0 == 0.4
    assert resumed.history == tracker.history
    nxt = resumed.measure(model, 6, 0.4)
    assert nxt.reward0 == 0.4  # baseline survived the resume
    assert nxt.divergence == pytest.approx((0.4 - 0.4) - (0.5 - 0.25))

    with pytest.raises(ValueError):
        CapabilityTracker(sampler, stub, start, items_per_slice=4).load_state_dict(state)
    other_start = _manual_start(sampler, {"t_rule/unseen_pairs": (0.0,), "t_rule/depth3": (0.0,)})
    fresh = CapabilityTracker(sampler, stub, other_start, items_per_slice=3)
    with pytest.raises(ValueError, match="start map"):
        fresh.load_state_dict(state)
    assert fresh.history == [] and fresh.reward0 is None  # atomic: nothing changed
    other_sampler = ItemSampler(sampler.splits, root_seed=99)
    with pytest.raises(ValueError):
        CapabilityTracker(other_sampler, stub, start, items_per_slice=3).load_state_dict(state)


def test_real_model_with_measured_start_map(splits: CompositionSplits, sampler: ItemSampler) -> None:
    tok = build_char_tokenizer()
    model = build_tiny_llama(len(tok), seed=0)
    score_fn = make_score_fn(
        tok, ExactAnswerVerifier(), max_new_tokens=splits.operators.max_len + 1, batch_size=16
    )
    start = StartMap.measure(
        model, sampler, score_fn, items_per_slice=2, slices=splits.slices_of(Split.T_RULE), k=2, num_samples=2
    )
    tracker = CapabilityTracker(sampler, score_fn, start, items_per_slice=2)
    assert tracker.t_rule0 == pytest.approx(start.split_mean_pass1(Split.T_RULE))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.train()
    point = tracker.measure(model, 1, 0.3)
    assert model.training
    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), before[name]), name
    assert 0.0 <= point.acquisition <= 1.0 and 0.0 <= point.t_rule <= 1.0
    assert point.divergence == pytest.approx(0.0 - (point.t_rule - tracker.t_rule0))
    scored = tracker.score(model, 1)
    assert {name: s.mean for name, s in scored.items()} == dict(point.slices)
    json.dumps(point.to_dict())


# ============================================================================== regressions (review wave 2)
def test_tracker_and_gate_items_never_coincide_at_default_sizes(
    sampler: ItemSampler, start: StartMap
) -> None:
    config = GateConfig()  # default sizes: 256 per transfer slice, 2048 per retention slice
    probe = CapabilityProbe.from_config(sampler, SliceScore(), config)
    tracker = CapabilityTracker(sampler, SliceScore(), start, items_per_slice=256)
    gate_used: dict[str, set[str]] = {
        name: set() for name in (*probe.transfer_slices, *probe.retention_slices)
    }
    for decision in range(4):
        for attempt in range(3):
            for name, group in probe.gate_items(decision, attempt).items():
                assert all(item_pool(sampler, it.uid) != TRACK_POOL for it in group)
                gate_used[name].update(it.uid for it in group)
    for step in (0, 1, 2, 3, 20):
        tracked = tracker.items(step)
        for name, used in gate_used.items():
            overlap = [it.uid for it in tracked[name] if it.uid in used]
            assert overlap == [], (step, name, overlap[:3])
    scored = tracker.score(nn.Linear(1, 1), 1)
    assert all(sc.pool == TRACK_POOL and sc.to_dict()["pool"] == TRACK_POOL for sc in scored.values())


def test_start_budget_must_match_tracker_score_fn(splits: CompositionSplits, sampler: ItemSampler) -> None:
    tok = build_char_tokenizer()
    max_len = splits.operators.max_len
    start = _manual_start(sampler, {"t_rule/unseen_pairs": (1.0,), "t_rule/depth3": (0.0,)})
    assert start.max_new_tokens == max_len + 1
    wider = make_score_fn(tok, ExactAnswerVerifier(), max_new_tokens=4 * max_len, batch_size=2)
    with pytest.raises(ValueError, match="max_new_tokens"):
        CapabilityTracker(sampler, wider, start, items_per_slice=2)
    same = make_score_fn(tok, ExactAnswerVerifier(), max_new_tokens=max_len + 1, batch_size=2)
    CapabilityTracker(sampler, same, start, items_per_slice=2)
    CapabilityTracker(sampler, SliceScore(), start, items_per_slice=2)  # no budget: not compared
