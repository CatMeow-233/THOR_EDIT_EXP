"""Tests for research_rl.capability.search (subset search with capability as the objective).

Tiny CPU models only. A random tiny Llama scores ~0 under the exact verifier, so the search is
tested with a constructed, weight-sensitive score function: transfer improves with the gain of
one *planted* channel group's ``down_proj`` columns, and a *harmful* group raises transfer a
little but costs retention. The ES operator's fitness rewards both groups (a reward that the
capability objective must see through). Scores are exact functions of the weights and of a
uid hash, so every outcome below is deterministic.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest
import torch
from torch import nn

from research_rl.capability.es import ESConfig, ESOperator, FitnessResult
from research_rl.capability.gate import GateConfig
from research_rl.capability.operators import TorchGRPOOperator
from research_rl.capability.probe import GATE_POOL, ITEM_POOLS, TRACK_POOL, CapabilityProbe, item_pool
from research_rl.capability.search import (
    CONTROL_KINDS,
    DEFAULT_MARGIN,
    STATEMENT,
    SearchReport,
    SubsetSearch,
    burst_seed,
    combine_groups,
    control_seed,
    final_purpose,
    objective,
    score_seed,
    search_purpose,
    whole_layer_selection,
)
from research_rl.capability.spaces import ChannelGainSpace
from research_rl.capability.tasks import CompositionSplits, ExactAnswerVerifier, ItemSampler, OperatorSet
from research_rl.capability.types import ChannelGroup, GateDecision, Item, OperatorReport, Split
from research_rl.models.ffn_channels import ChannelSelection, channel_mask
from research_rl.models.model_interface import find_swiglu_ffns, fingerprint
from research_rl.models.tiny import build_char_tokenizer, build_tiny_llama
from research_rl.optim.optimizer_wrapper import HookedOptimizer
from research_rl.optim.update_interface import MaskRule
from research_rl.rewards.verifier_interface import BaseVerifier, Status

TRANSFER = ("t_rule/unseen_pairs", "t_rule/depth3")
RETENTION = ("retention/atomic", "retention/copy")
PLANTED = ChannelSelection({0: range(8, 16)})
HARMFUL = ChannelSelection({1: range(0, 8)})


# ============================================================================== fixtures
@pytest.fixture(scope="module")
def splits() -> CompositionSplits:
    return CompositionSplits.build(OperatorSet(seed=0, num_operators=10), seed=0)


@pytest.fixture(scope="module")
def sampler(splits: CompositionSplits) -> ItemSampler:
    return ItemSampler(splits, root_seed=3)


@pytest.fixture(scope="module")
def tok() -> Any:
    return build_char_tokenizer()


@pytest.fixture
def model(tok: Any) -> nn.Module:
    return build_tiny_llama(len(tok), seed=0)


def _groups() -> list[ChannelGroup]:
    # the planted group is deliberately not first, so a tie-break by index cannot find it
    return [
        ChannelGroup("L0/a", ChannelSelection({0: range(0, 8)})),
        ChannelGroup("planted", PLANTED),
        ChannelGroup("harmful", HARMFUL),
        ChannelGroup("L1/b", ChannelSelection({1: range(8, 16)})),
    ]


def _params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _assert_bitwise(model: nn.Module, before: Mapping[str, torch.Tensor]) -> None:
    now = dict(model.named_parameters())
    assert set(now) == set(before)
    for name, value in before.items():
        assert now[name].dtype == value.dtype
        assert torch.equal(now[name].detach(), value), name


def _sel(d: Mapping[str, Sequence[int]]) -> ChannelSelection:
    return ChannelSelection({int(k): v for k, v in d.items()})


def _u(uid: str) -> float:
    return int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8], "big") / 2.0**64


def _down(model: nn.Module, layer: int) -> torch.Tensor:
    return find_swiglu_ffns(model, [layer])[0].down.weight


def _gain(model: nn.Module, selection: ChannelSelection, ref: Mapping[int, torch.Tensor]) -> float:
    """Mean projection ratio ``<W[:, c], ref[:, c]> / <ref[:, c], ref[:, c]>`` over the selection
    (exactly ``mean(1 + v_c)`` for a ChannelGainSpace write ``v``)."""
    total, count = 0.0, 0
    for layer, chans in selection.items():
        w = _down(model, layer).detach().double()
        r = ref[layer].double()
        for c in chans:
            total += float((w[:, c] @ r[:, c]) / (r[:, c] @ r[:, c]))
            count += 1
    return total / count


class PlantedScore:
    """Score 1.0 iff ``u(uid) < p``: transfer ``p = 0.3 + 2 x + 1 h``, retention ``p = 0.7 - 3 h``
    (clamped to [0, 1]) with ``x`` / ``h`` the planted / harmful gain excess over the start."""

    def __init__(self, model: nn.Module, *, fail_after: int | None = None) -> None:
        self.ref = {layer: _down(model, layer).detach().clone() for layer in (0, 1)}
        self.calls = 0
        self.fail_after = fail_after
        self.purposes_seen: list[str] = []

    def excess(self, model: nn.Module) -> tuple[float, float]:
        return _gain(model, PLANTED, self.ref) - 1.0, _gain(model, HARMFUL, self.ref) - 1.0

    def __call__(self, model: nn.Module, items: Sequence[Item]) -> list[float]:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("score function failure")
        x, h = self.excess(model)
        p_t = min(1.0, max(0.0, 0.3 + 2.0 * x + 1.0 * h))
        p_r = min(1.0, max(0.0, 0.7 - 3.0 * h))
        out = []
        for it in items:
            p = p_r if it.split is Split.RETENTION else p_t
            out.append(1.0 if _u(it.uid) < p else 0.0)
        return out


def _probe(sampler: ItemSampler, score: Any, n: int = 40) -> CapabilityProbe:
    return CapabilityProbe(
        sampler,
        score,
        transfer_slices=TRANSFER,
        retention_slices=RETENTION,
        items_per_transfer_slice=n,
        items_per_retention_slice=n,
    )


def _es_factory(
    model: nn.Module, score: PlantedScore, *, calls: list[ChannelSelection | None] | None = None
) -> Callable[[ChannelSelection | None], ESOperator]:
    def fitness(m: nn.Module, update_index: int, member_index: int) -> FitnessResult:
        x, h = score.excess(m)
        return FitnessResult(reward_v1=x + h, reward_v2=x, v1_pass=(), v2_pass=())

    def make(selection: ChannelSelection | None) -> ESOperator:
        if calls is not None:
            calls.append(selection)
        sel = ChannelSelection.all_channels(model) if selection is None else selection
        space = ChannelGainSpace(model, sel)
        config = ESConfig(population=4, sigma=0.05, lr=0.05, max_step_norm=0.5, seed=7)
        return ESOperator(space, fitness, config)

    return make


def _search(model: nn.Module, sampler: ItemSampler, **kw: Any) -> tuple[SubsetSearch, PlantedScore]:
    score = PlantedScore(model)
    opts: dict[str, Any] = {"rounds": 1, "burst_updates": 2, "repeats_final": 2, "seed": 0}
    opts.update(kw)
    search = SubsetSearch(_groups(), _es_factory(model, score), _probe(sampler, score), **opts)
    return search, score


# ============================================================================== helpers
def test_purposes_and_seeds() -> None:
    assert search_purpose(2, 5) == "search/2/5"
    assert final_purpose(1) == "search/final/1"
    with pytest.raises(ValueError):
        search_purpose(-1, 0)
    with pytest.raises(ValueError):
        final_purpose(True)  # type: ignore[arg-type]
    assert burst_seed(0) == burst_seed(0) != burst_seed(1)
    assert control_seed(0) != burst_seed(0)
    assert 0 <= burst_seed(123) < 2**63
    assert objective(0.3, 0.01, penalty=2.0, margin=0.02) == pytest.approx(0.3)
    assert objective(0.3, 0.12, penalty=2.0, margin=0.02) == pytest.approx(0.1)
    assert GateConfig().retention_margin == DEFAULT_MARGIN


def test_combine_groups() -> None:
    a = ChannelGroup("a", ChannelSelection({0: [1, 2]}))
    b = ChannelGroup("b", ChannelSelection({0: [2, 3], 1: [0]}))
    c = combine_groups([a, b])
    assert c.name == "a+b"
    assert c.selection == ChannelSelection({0: [1, 2, 3], 1: [0]})
    assert combine_groups([a], name="x").name == "x"
    with pytest.raises(ValueError):
        combine_groups([])


def test_whole_layer_selection(model: nn.Module) -> None:
    sel = ChannelSelection({0: range(0, 6), 1: range(0, 64, 2)})  # 6 + 32 = 38 channels
    out = whole_layer_selection(sel, model)
    assert out.count() == sel.count()
    assert out == ChannelSelection({1: range(38)})  # largest layer first, contiguous from 0
    big = ChannelSelection({0: range(64), 1: range(0, 64, 2)})
    # 64 in layer 0, 32 in layer 1 -> layer 0 whole, then 32 contiguous of layer 1
    out = whole_layer_selection(big, model)
    assert out == ChannelSelection({0: range(64), 1: range(32)})
    # a whole layer stays that layer
    out = whole_layer_selection(ChannelSelection({1: range(64)}), model)
    assert out == ChannelSelection({1: range(64)})
    with pytest.raises(ValueError):
        whole_layer_selection(ChannelSelection(), model)


def test_whole_layer_selection_avoids_the_winner(model: nn.Module) -> None:
    # regression: a contiguous block from channel 0 used to be returned unchanged (vacuous control)
    first8 = ChannelSelection({0: range(0, 8)})
    assert whole_layer_selection(first8, model) == ChannelSelection({0: range(8, 16)})
    assert whole_layer_selection(ChannelSelection({0: range(8, 16)}), model) == ChannelSelection(
        {0: range(0, 8)}
    )
    # the lowest start with the least overlap: [24, 40) misses both {0..7} and {16..23}
    split = ChannelSelection({0: [*range(0, 8), *range(16, 24)]})
    assert whole_layer_selection(split, model) == ChannelSelection({0: range(24, 40)})
    # whole layer 0 plus a 4-block of layer 1 placed off the winner's {1: 0..3}
    mixed = ChannelSelection({0: range(64), 1: range(0, 4)})
    assert whole_layer_selection(mixed, model) == ChannelSelection({0: range(64), 1: range(4, 8)})


# ============================================================================== validation
def test_constructor_validation(model: nn.Module, sampler: ItemSampler) -> None:
    score = PlantedScore(model)
    probe = _probe(sampler, score)
    make = _es_factory(model, score)
    ok: dict[str, Any] = {"rounds": 1, "burst_updates": 1}
    with pytest.raises(ValueError, match="at least one"):
        SubsetSearch([], make, probe, **ok)
    g = _groups()
    with pytest.raises(ValueError, match="duplicate candidate"):
        SubsetSearch([g[0], g[0]], make, probe, **ok)
    with pytest.raises(ValueError, match="empty selection"):
        SubsetSearch([ChannelGroup("e", ChannelSelection())], make, probe, **ok)
    with pytest.raises(ValueError, match="control names"):
        SubsetSearch([ChannelGroup("random", PLANTED)], make, probe, **ok)
    with pytest.raises(ValueError, match="unknown controls"):
        SubsetSearch(g, make, probe, controls=("magic",), **ok)
    with pytest.raises(ValueError, match="duplicate controls"):
        SubsetSearch(g, make, probe, controls=("random", "random"), **ok)
    for bad in (0.0, 1.5, True):
        with pytest.raises(ValueError, match="survivors_fraction"):
            SubsetSearch(g, make, probe, survivors_fraction=bad, **ok)
    with pytest.raises(ValueError, match="rounds"):
        SubsetSearch(g, make, probe, rounds=0, burst_updates=1)
    with pytest.raises(ValueError, match="burst_updates"):
        SubsetSearch(g, make, probe, rounds=1, burst_updates=0)
    with pytest.raises(ValueError, match="penalty"):
        SubsetSearch(g, make, probe, penalty=-1.0, **ok)
    with pytest.raises(ValueError, match="margin"):
        SubsetSearch(g, make, probe, margin=float("nan"), **ok)
    with pytest.raises(ValueError, match="pool"):
        SubsetSearch(g, make, probe, pool="other", **ok)
    with pytest.raises(TypeError):
        SubsetSearch(g, make, object(), **ok)  # type: ignore[arg-type]
    s = SubsetSearch(g, make, probe, **ok)
    assert s.items_per_transfer_slice == 40 and s.items_per_retention_slice == 40
    assert s.controls == CONTROL_KINDS
    assert [s.survivors_after(n) for n in (1, 2, 3, 4, 5)] == [1, 1, 2, 2, 3]


def test_run_validates_selection_before_training(tok: Any, sampler: ItemSampler) -> None:
    model = build_tiny_llama(len(tok), seed=0)
    score = PlantedScore(model)
    calls: list[ChannelSelection | None] = []
    bad = [*_groups(), ChannelGroup("too_big", ChannelSelection({0: [999]}))]
    search = SubsetSearch(
        bad, _es_factory(model, score, calls=calls), _probe(sampler, score), rounds=1, burst_updates=1
    )
    with pytest.raises(Exception, match="intermediate_size"):
        search.run(model)
    assert calls == []


# ============================================================================== the search
def test_finds_planted_group(model: nn.Module, sampler: ItemSampler) -> None:
    before = _params(model)
    search, _ = _search(model, sampler)
    report = search.run(model)
    _assert_bitwise(model, before)
    assert isinstance(report, SearchReport)

    rounds = [e for e in report.evaluations if e.stage == "round"]
    assert [e.candidate for e in rounds] == [g.name for g in _groups()]
    by_name = {e.candidate: e for e in rounds}
    assert by_name["planted"].gain > 0.1 and by_name["planted"].objective > 0.1
    assert by_name["planted"].worst_drop == 0.0
    # zero-fitness candidates take an exactly zero ES step: nothing changes
    for name in ("L0/a", "L1/b"):
        assert by_name[name].gain == 0.0 and by_name[name].worst_drop == 0.0
    # the harmful group raises the training reward and transfer, but costs retention
    harm = by_name["harmful"]
    assert harm.gain > 0.0
    assert harm.worst_drop > DEFAULT_MARGIN and harm.objective < 0.0
    burst_of = {report.bursts[e.burst].candidate: report.bursts[e.burst] for e in rounds}
    assert burst_of["harmful"].reward_v1[-1] > 0.0
    assert by_name["harmful"].objective == pytest.approx(
        harm.gain - search.penalty * (harm.worst_drop - search.margin)
    )

    assert report.rounds == [[g.name for g in _groups()]]
    assert report.finalists[0] == "planted" == report.winner
    assert set(report.finalists) == {"planted", "L0/a"}
    assert report.winner_selection == {"0": list(range(8, 16))}
    best = report.final[0]
    assert best.candidate == "planted" and best.mean_objective > report.final[1].mean_objective
    assert len(best.objectives) == 2
    assert report.statement == STATEMENT and "no optimality claim" in report.statement


def test_successive_halving_counts(model: nn.Module, sampler: ItemSampler) -> None:
    search, _ = _search(model, sampler, rounds=3, repeats_final=1, controls=())
    report = search.run(model)
    # 4 -> 2 -> 1; the third round is skipped (a single candidate is left)
    assert [len(r) for r in report.rounds] == [4, 2]
    assert [e.round for e in report.evaluations if e.stage == "round"] == [0, 0, 0, 0, 1, 1]
    assert report.rounds[1][0] == "planted"
    assert report.finalists == ["planted"]
    assert report.controls == {}
    assert report.totals["bursts"] == 4 + 2 + 1
    assert report.totals["updates"] == 7 * search.burst_updates
    # ES: population forward passes per update, no backward
    assert report.totals["forward_passes"] == 7 * search.burst_updates * 4
    assert report.totals["backward_passes"] == 0
    # one score call per slice: start + candidate for each round evaluation, then the final
    assert report.totals["score_calls"] == 6 * 2 * 4 + 1 * 4 + 1 * 4


def test_items_are_fresh_search_items(model: nn.Module, sampler: ItemSampler) -> None:
    search, _ = _search(model, sampler, rounds=2)
    report = search.run(model)
    rounds = [e for e in report.evaluations if e.stage == "round"]
    names = [g.name for g in _groups()]
    assert [e.purpose for e in rounds[:4]] == [search_purpose(0, c) for c in range(4)]
    for e in rounds[4:]:
        assert e.purpose == search_purpose(1, names.index(e.candidate))
    assert all(e.purpose.startswith("search/") for e in report.evaluations)
    assert len({e.uid_digest for e in rounds}) == len(rounds)  # fresh items per evaluation
    finals = [e for e in report.evaluations if e.stage == "final"]
    assert {e.purpose for e in finals} == {final_purpose(0), final_purpose(1)}
    digests = {j: {e.uid_digest for e in finals if e.repeat == j} for j in (0, 1)}
    assert all(len(d) == 1 for d in digests.values())  # same items for every finalist / control
    assert digests[0] != digests[1]
    assert not ({e.uid_digest for e in finals} & {e.uid_digest for e in rounds})
    for e in report.evaluations:
        assert e.items == 4 * 40 and [s.n for s in e.slices] == [40] * 4


def test_update_offset_and_index(model: nn.Module, sampler: ItemSampler) -> None:
    before = _params(model)
    base, _ = _search(model, sampler, controls=("unrestricted",), repeats_final=1)
    first = base.run(model)
    moved, _ = _search(model, sampler, controls=("unrestricted",), repeats_final=1, index=1)
    second = moved.run(model, update_offset=5)
    _assert_bitwise(model, before)
    assert all(b.update_indices == (0, 1) for b in first.bursts)
    assert all(b.update_indices == (5, 6) for b in second.bursts)
    assert all(e.index == 0 for e in first.evaluations)
    assert all(e.index == 1 for e in second.evaluations)
    assert second.config["index"] == 1
    old = {e.uid_digest for e in first.evaluations}
    assert not old & {e.uid_digest for e in second.evaluations}  # fresh items for index=1
    with pytest.raises(ValueError, match="update_offset"):
        base.run(model, update_offset=-1)
    with pytest.raises(ValueError, match="index"):
        _search(model, sampler, index=-1)


@pytest.mark.parametrize("pool", ["default", TRACK_POOL, GATE_POOL, None])
def test_pool_passthrough(model: nn.Module, sampler: ItemSampler, pool: str | None) -> None:
    seen: list[str] = []

    class Recording(PlantedScore):
        def __call__(self, model: nn.Module, items: Sequence[Item]) -> list[float]:
            seen.extend(it.uid for it in items)
            return super().__call__(model, items)

    score = Recording(model)
    kw: dict[str, Any] = {} if pool == "default" else {"pool": pool}
    search = SubsetSearch(
        _groups()[:2],
        _es_factory(model, score),
        _probe(sampler, score, n=12),
        rounds=1,
        burst_updates=1,
        repeats_final=1,
        controls=(),
        **kw,
    )
    report = search.run(model)
    pools = {item_pool(sampler, uid) for uid in seen}
    assert seen and set(ITEM_POOLS) == {TRACK_POOL, GATE_POOL}
    if pool in ("default", TRACK_POOL):
        # regression: the default used to be pool=None, which reuses the gate's evidence items
        assert pools == {TRACK_POOL}  # never a gate item
        assert search.pool == TRACK_POOL == report.config["pool"]
        assert "no search item is a gate item" in report.extra["item_pool_note"]
    elif pool == GATE_POOL:
        assert pools == {GATE_POOL}
        assert "coincide by chance with gate items" in report.extra["item_pool_note"]
    else:
        assert pools == {TRACK_POOL, GATE_POOL}  # unpartitioned
        assert report.config["pool"] is None
        assert "coincide by chance with gate items" in report.extra["item_pool_note"]


def test_restores_model_rng_and_modes(model: nn.Module, sampler: ItemSampler) -> None:
    model.train()
    model.model.layers[0].eval()  # type: ignore[union-attr,index]
    modes = [m.training for m in model.modules()]
    before = _params(model)
    rng = torch.get_rng_state()
    search, _ = _search(model, sampler)
    report = search.run(model)
    _assert_bitwise(model, before)
    assert torch.equal(torch.get_rng_state(), rng)
    assert [m.training for m in model.modules()] == modes
    assert all(p.grad is None for p in model.parameters())
    assert report.start_fingerprint == fingerprint(model)


class NoisyScore(PlantedScore):
    """A sampled score function: :class:`PlantedScore` with each score flipped with probability
    0.1 by a draw from the global torch RNG (what a pass@k-style scorer may do)."""

    def __call__(self, model: nn.Module, items: Sequence[Item]) -> list[float]:
        out = super().__call__(model, items)
        flips = (torch.rand(len(items)) < 0.1).tolist()
        return [1.0 - v if f else v for v, f in zip(out, flips, strict=True)]


class RngRecordingOperator:
    """Wraps an operator and records one global-RNG draw per step (as a sampling operator would
    consume the global RNG)."""

    def __init__(self, inner: ESOperator, log: list[float]) -> None:
        self.inner = inner
        self.name = inner.name
        self.optimizer: torch.optim.Optimizer | None = inner.optimizer
        self.log = log

    def parameter_names(self) -> list[str]:
        return self.inner.parameter_names()

    def step(self, model: nn.Module, update_index: int) -> OperatorReport:
        self.log.append(float(torch.rand(()).item()))
        return self.inner.step(model, update_index)

    def on_decision(self, decision: GateDecision) -> None:
        self.inner.on_decision(decision)

    def state_dict(self) -> dict[str, Any]:
        return self.inner.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.inner.load_state_dict(state)


def _noisy_run(
    tok: Any, sampler: ItemSampler, *, seed: int, pre_seed: int
) -> tuple[dict[str, Any], list[float]]:
    model = build_tiny_llama(len(tok), seed=0)
    score = NoisyScore(model)
    inner = _es_factory(model, score)
    draws: list[float] = []
    search = SubsetSearch(
        _groups(),
        lambda sel: RngRecordingOperator(inner(sel), draws),
        _probe(sampler, score, n=20),
        rounds=1,
        burst_updates=2,
        repeats_final=1,
        seed=seed,
    )
    with torch.random.fork_rng():  # the caller's RNG state, without leaking into other tests
        torch.manual_seed(pre_seed)
        rng = torch.get_rng_state()
        report = search.run(model)
        assert torch.equal(torch.get_rng_state(), rng)  # also with a sampled scorer / operator
    return report.to_dict(), draws


def test_rng_seeded_and_independent_of_caller(tok: Any, sampler: ItemSampler) -> None:
    """Regression: start weights used to be scored outside any fork, so a sampled score function
    advanced the caller's RNG and every gain depended on the caller's RNG state."""
    first, draws0 = _noisy_run(tok, sampler, seed=0, pre_seed=0)
    second, draws1 = _noisy_run(tok, sampler, seed=0, pre_seed=1)
    _assert_same_report(first, second)  # depends on ``seed``, not on the caller's RNG
    assert draws0 == draws1
    # every burst runs on the same seeded RNG stream: identical draws per update in every burst
    per_burst = [tuple(draws0[i : i + 2]) for i in range(0, len(draws0), 2)]
    assert len(per_burst) == len(first["bursts"]) and len(set(per_burst)) == 1
    with torch.random.fork_rng():
        torch.random.default_generator.manual_seed(burst_seed(0))
        assert per_burst[0][0] == float(torch.rand(()).item())
    # common random numbers: start and trained weights of a slice see the same flips, so the
    # zero-step candidates (exactly unchanged weights) still have exactly zero gain and drop
    rounds = {e["candidate"]: e for e in first["evaluations"] if e["stage"] == "round"}
    for name in ("L0/a", "L1/b"):
        assert rounds[name]["gain"] == 0.0 and rounds[name]["worst_drop"] == 0.0
        for sl in rounds[name]["slices"]:
            assert sl["start_mean"] == sl["candidate_mean"]
    # the noise is live: another search seed gives other scores on the same items
    other, _ = _noisy_run(tok, sampler, seed=1, pre_seed=0)
    starts = [[sl["start_mean"] for sl in e["slices"]] for e in first["evaluations"]]
    other_starts = [[sl["start_mean"] for sl in e["slices"]] for e in other["evaluations"]]
    assert [e["uid_digest"] for e in first["evaluations"][:4]] == [
        e["uid_digest"] for e in other["evaluations"][:4]
    ]
    assert starts != other_starts
    assert score_seed(0, "search/0/0", 0, "a") != score_seed(1, "search/0/0", 0, "a")
    assert score_seed(0, "search/0/0", 0, "a") != score_seed(0, "search/0/0", 0, "b")


def test_control_overlap_flags(tok: Any, sampler: ItemSampler) -> None:
    model = build_tiny_llama(len(tok), seed=0)
    score = PlantedScore(model)
    first8 = ChannelGroup("L0/first8", ChannelSelection({0: range(0, 8)}))
    search = SubsetSearch(
        [first8],
        _es_factory(model, score),
        _probe(sampler, score, n=8),
        rounds=1,
        burst_updates=1,
        repeats_final=1,
    )
    report = search.run(model)
    assert report.winner == "L0/first8"
    wl = report.controls["whole_layer"]
    assert wl.selection == {"0": list(range(8, 16))} and wl.channels == 8
    assert wl.winner_overlap == 0 and wl.same_as_winner is False
    unr = report.controls["unrestricted"]
    assert unr.winner_overlap == 8 and unr.same_as_winner is False
    rnd = report.controls["random"]
    assert rnd.winner_overlap == len(set(rnd.selection["0"]) & set(range(8)))  # type: ignore[index]
    assert all(f.winner_overlap is None and f.same_as_winner is None for f in report.final)
    d = report.to_dict()["controls"]["whole_layer"]
    assert d["winner_overlap"] == 0 and d["same_as_winner"] is False
    # a winner that already is a whole layer: the control cannot differ, and the report says so
    layer0 = ChannelGroup("L0/all", ChannelSelection.all_channels(model, [0]))
    search = SubsetSearch(
        [layer0],
        _es_factory(model, score),
        _probe(sampler, score, n=8),
        rounds=1,
        burst_updates=1,
        repeats_final=1,
        controls=("whole_layer",),
    )
    wl = search.run(model).controls["whole_layer"]
    assert wl.same_as_winner is True and wl.winner_overlap == 64


def _assert_same_report(a: Any, b: Any, path: str = "") -> None:
    """Structural equality: exact for everything but floats (trained values), which must agree to
    ``pytest.approx`` (bit-exactness is asserted only for copies / restores)."""
    if isinstance(a, float) and not isinstance(a, bool):
        assert isinstance(b, float) and b == pytest.approx(a, rel=1e-9, abs=1e-12), path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), path
        for k in a:
            _assert_same_report(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same_report(x, y, f"{path}[{i}]")
    else:
        assert a == b, path


def test_deterministic_and_json(tok: Any, sampler: ItemSampler) -> None:
    reports = []
    for _ in range(2):
        model = build_tiny_llama(len(tok), seed=0)
        search, _ = _search(model, sampler, seed=5)
        reports.append(search.run(model).to_dict())
    _assert_same_report(reports[0], reports[1])
    text = json.dumps(reports[0], allow_nan=False, sort_keys=True)
    assert json.loads(text)["winner"] == "planted"
    model = build_tiny_llama(len(tok), seed=0)
    other, _ = _search(model, sampler, seed=6)
    third = other.run(model).to_dict()
    assert third["controls"]["random"]["selection"] != reports[0]["controls"]["random"]["selection"]
    assert third["config"]["burst_seed"] == burst_seed(6)


def test_controls_equal_budget_and_items(model: nn.Module, sampler: ItemSampler) -> None:
    calls: list[ChannelSelection | None] = []
    score = PlantedScore(model)
    search = SubsetSearch(
        _groups(),
        _es_factory(model, score, calls=calls),
        _probe(sampler, score),
        rounds=1,
        burst_updates=2,
        seed=3,
    )
    report = search.run(model)
    assert list(report.controls) == list(CONTROL_KINDS)
    winner_sel = _sel(report.winner_selection)
    rnd = report.controls["random"]
    rnd_sel = _sel(rnd.selection or {})
    assert {layer: len(c) for layer, c in rnd_sel.items()} == {
        layer: len(c) for layer, c in winner_sel.items()
    }
    wl = report.controls["whole_layer"]
    assert wl.channels == winner_sel.count() == rnd.channels
    assert _sel(wl.selection or {}) == whole_layer_selection(winner_sel, model)
    assert (
        report.controls["unrestricted"].selection is None and report.controls["unrestricted"].channels is None
    )
    assert calls[-1] is None  # make_operator(None) for the unrestricted control

    finals = [*report.final, *report.controls.values()]
    for f in finals:
        b = report.bursts[f.burst]
        assert b.stage == "final" and len(b.update_indices) == search.burst_updates
        assert b.update_indices == report.bursts[report.final[0].burst].update_indices
        assert f.forward_passes == b.forward_passes == search.burst_updates * 4
        assert len(f.objectives) == search.repeats_final
    for j in range(search.repeats_final):
        evs = [e for e in report.evaluations if e.stage == "final" and e.repeat == j]
        assert {e.candidate for e in evs} == {*report.finalists, *CONTROL_KINDS}
        assert len({e.uid_digest for e in evs}) == 1
    # the unrestricted ES control also trains the harmful channels: it loses retention
    unr = report.controls["unrestricted"]
    assert unr.mean_worst_drop > 0.0


# ============================================================================== exceptions
class ShiftOperator:
    """Minimal backprop operator: loss = -sum of the selected ``down_proj`` columns; one step of
    a (possibly shared) AdamW per update. Raises on the ``fail_on``-th step overall."""

    name = "shift"

    def __init__(
        self,
        model: nn.Module,
        selection: ChannelSelection | None,
        optimizer: torch.optim.Optimizer,
        counter: list[int],
        fail_on: int | None,
    ) -> None:
        self.model = model
        self.selection = selection
        self.optimizer: torch.optim.Optimizer | None = optimizer
        self.counter = counter
        self.fail_on = fail_on

    def parameter_names(self) -> list[str]:
        return [n for n, _ in self.model.named_parameters()]

    def step(self, model: nn.Module, update_index: int) -> OperatorReport:
        self.counter[0] += 1
        sel = self.selection if self.selection is not None else ChannelSelection.all_channels(model)
        loss = sum(-_down(model, layer)[:, list(ch)].sum() for layer, ch in sel.items())
        assert isinstance(loss, torch.Tensor)
        assert self.optimizer is not None
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.fail_on is not None and self.counter[0] == self.fail_on:
            raise RuntimeError("operator failure")
        return OperatorReport(update_index, 0.0, None, (), (), forward_passes=1, backward_passes=1)

    def on_decision(self, decision: GateDecision) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        pass


def _shared_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    # give it non-trivial state, then put the weights back
    saved = _params(model)
    loss = sum(p.float().pow(2).sum() for p in model.parameters())
    assert isinstance(loss, torch.Tensor)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(saved[n])
    return opt


def _opt_state(opt: torch.optim.Optimizer) -> dict[str, Any]:
    sd = opt.state_dict()
    return {
        "param_groups": json.dumps(sd["param_groups"], sort_keys=True, default=str),
        "state": {
            k: {kk: vv.clone() if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
            for k, v in sd["state"].items()
        },
    }


def _assert_same_opt(a: dict[str, Any], b: dict[str, Any]) -> None:
    assert a["param_groups"] == b["param_groups"]
    assert set(a["state"]) == set(b["state"])
    for k in a["state"]:
        for kk, vv in a["state"][k].items():
            other = b["state"][k][kk]
            if isinstance(vv, torch.Tensor):
                assert torch.equal(vv, other)
            else:
                assert vv == other


@pytest.mark.parametrize("fail_on", [None, 3])
def test_shared_optimizer_restored(model: nn.Module, sampler: ItemSampler, fail_on: int | None) -> None:
    opt = _shared_optimizer(model)
    before_opt = _opt_state(opt)
    before = _params(model)
    score = PlantedScore(model)
    counter = [0]
    search = SubsetSearch(
        _groups(),
        lambda sel: ShiftOperator(model, sel, opt, counter, fail_on),
        _probe(sampler, score, n=10),
        rounds=1,
        burst_updates=2,
        repeats_final=1,
    )
    if fail_on is None:
        report = search.run(model)
        assert report.totals["backward_passes"] == report.totals["updates"] > 0
        assert report.winner in {g.name for g in _groups()}
    else:
        with pytest.raises(RuntimeError, match="operator failure"):
            search.run(model)
    _assert_bitwise(model, before)
    _assert_same_opt(_opt_state(opt), before_opt)
    assert all(p.grad is None for p in model.parameters())


def test_restores_on_make_operator_and_score_failures(tok: Any, sampler: ItemSampler) -> None:
    model = build_tiny_llama(len(tok), seed=0)
    before = _params(model)
    score = PlantedScore(model)
    make = _es_factory(model, score)
    n_calls = [0]

    def failing_make(sel: ChannelSelection | None) -> ESOperator:
        n_calls[0] += 1
        if n_calls[0] == 3:
            raise RuntimeError("make_operator failure")
        return make(sel)

    search = SubsetSearch(_groups(), failing_make, _probe(sampler, score, n=8), rounds=1, burst_updates=2)
    with pytest.raises(RuntimeError, match="make_operator failure"):
        search.run(model)
    _assert_bitwise(model, before)

    # fail while scoring a trained candidate (after its burst moved the planted weights)
    # score calls (one per slice): candidate 0 start 1-4, trained 5-8; candidate 1 (planted)
    # start 9-12, trained 13-16 -> call 16 fails inside the planted candidate's trained scoring
    failing = PlantedScore(model, fail_after=15)
    search = SubsetSearch(
        _groups(), _es_factory(model, failing), _probe(sampler, failing, n=8), rounds=1, burst_updates=2
    )
    with pytest.raises(RuntimeError, match="score function failure"):
        search.run(model)
    _assert_bitwise(model, before)

    def not_an_operator(sel: ChannelSelection | None) -> Any:
        return object()

    search = SubsetSearch(_groups(), not_an_operator, _probe(sampler, score, n=8), rounds=1, burst_updates=1)
    with pytest.raises(TypeError, match="UpdateOperator"):
        search.run(model)
    _assert_bitwise(model, before)


# ============================================================================== GRPO + HookedOptimizer MaskRule
class AnyDigitVerifier(BaseVerifier):
    """Lenient training verifier V1: correct iff the completion contains a digit (truncation
    ignored), so a random tiny model gets non-constant group rewards and a non-zero gradient."""

    name = "any_digit"

    def _verify(
        self, completion: str | None, reference: Any, **meta: Any
    ) -> tuple[Status, str | None, str | None, dict[str, Any]]:
        ok = completion is not None and any(c.isdigit() for c in completion)
        return ("correct" if ok else "incorrect"), completion, str(reference), {}


def _ffn_named(model: nn.Module, layers: Sequence[int]) -> list[tuple[str, nn.Parameter]]:
    wanted: set[int] = set()
    for ref in find_swiglu_ffns(model, layers):
        for proj in (ref.gate, ref.up, ref.down):
            wanted.update(id(p) for p in proj.parameters())
    return [(n, p) for n, p in model.named_parameters() if id(p) in wanted]


def _grpo_search(
    model: nn.Module, tok: Any, splits: CompositionSplits, sampler: ItemSampler
) -> tuple[SubsetSearch, list[tuple[ChannelSelection | None, bool]]]:
    """A search whose operators are TorchGRPOOperators; masked ones use a HookedOptimizer with a
    MaskRule over the selection's FFN parameters. The score function asserts, whenever it scores
    trained weights, that every change lies inside the current selection's channel mask, and
    records whether anything changed."""
    ref = _params(model)
    current: dict[str, ChannelSelection | None] = {}
    observed: list[tuple[ChannelSelection | None, bool]] = []

    class ScopeScore(PlantedScore):
        def __call__(self, m: nn.Module, items: Sequence[Item]) -> list[float]:
            if "sel" in current:
                sel = current["sel"]
                named = dict(m.named_parameters())
                changed = any(not torch.equal(p.detach(), ref[n]) for n, p in named.items())
                if sel is not None:
                    mask = channel_mask(m, sel)
                    for n, p in named.items():
                        diff = p.detach() != ref[n]
                        allowed = mask.get(n)
                        if allowed is None:
                            assert not diff.any(), f"{n} changed outside the selection's FFNs"
                        else:
                            assert not (diff & ~allowed).any(), f"{n} changed outside the mask"
                observed.append((sel, changed))
            return super().__call__(m, items)

    scope = ScopeScore(model)
    max_new = splits.operators.max_len + 1

    def make(sel: ChannelSelection | None) -> TorchGRPOOperator:
        current["sel"] = sel
        optimizer: torch.optim.Optimizer
        if sel is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
        else:
            named = _ffn_named(model, sel.layers())
            inner = torch.optim.AdamW([p for _, p in named], lr=0.05, weight_decay=0.0)
            optimizer = HookedOptimizer(inner, named, [MaskRule(channel_mask(model, sel))])
        return TorchGRPOOperator(
            model,
            tok,
            sampler,
            AnyDigitVerifier(),
            ExactAnswerVerifier(),
            group_size=4,
            prompts_per_update=2,
            lr=0.05,
            max_new_tokens=max_new,
            temperature=1.0,
            optimizer=optimizer,
            seed=11,
        )

    groups = [ChannelGroup("a", ChannelSelection({0: range(0, 8)})), ChannelGroup("planted", PLANTED)]
    search = SubsetSearch(
        groups, make, _probe(sampler, scope, n=8), rounds=1, burst_updates=2, repeats_final=1, seed=1
    )
    return search, observed


def test_grpo_with_hooked_mask_rule(tok: Any, splits: CompositionSplits, sampler: ItemSampler) -> None:
    model = build_tiny_llama(len(tok), seed=0)
    before = _params(model)
    rng = torch.get_rng_state()
    search, observed = _grpo_search(model, tok, splits, sampler)
    report = search.run(model)
    _assert_bitwise(model, before)
    assert torch.equal(torch.get_rng_state(), rng)
    assert report.totals["backward_passes"] == report.totals["updates"] == report.totals["bursts"] * 2
    assert report.totals["forward_passes"] == 2 * report.totals["updates"]
    assert set(report.controls) == set(CONTROL_KINDS)
    # the bursts actually moved weights (the lenient V1 gives a gradient), within the masks
    assert any(changed for sel, changed in observed if sel is not None)
    assert any(changed for sel, changed in observed if sel is None)
    for f in [*report.final, *report.controls.values()]:
        assert report.bursts[f.burst].backward_passes == search.burst_updates
    assert json.dumps(report.to_dict(), allow_nan=False)
    # deterministic: the same search on a fresh identical model gives the same report
    model2 = build_tiny_llama(len(tok), seed=0)
    search2, _ = _grpo_search(model2, tok, splits, sampler)
    _assert_same_report(search2.run(model2).to_dict(), report.to_dict())
