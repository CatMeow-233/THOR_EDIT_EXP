"""Tests for research_rl.capability.probe: gate evidence, score_slice and the start map.

Tiny CPU models only. A random tiny Llama scores ~0 under the exact verifier, so the pairing
tests use a weight-sensitive stub score function (its scores depend on one weight), and the
pass@k tests use a lenient verifier so that the counts are not all zero.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import pytest
import torch
from torch import nn

from research_rl.capability.gate import CapabilityGate, GateConfig
from research_rl.capability.probe import (
    GATE_POOL,
    ITEM_POOLS,
    START_PURPOSE,
    TRACK_POOL,
    TRACK_POOL_DENOMINATOR,
    CapabilityProbe,
    SliceScores,
    SliceStart,
    StartMap,
    check_truncation_budget,
    draw_slice,
    gate_purpose,
    item_pool,
    sample_completions,
    sample_completions_with_truncation,
    score_items,
    score_slice,
)
from research_rl.capability.tasks import (
    AuditAccessError,
    CompositionSplits,
    ExactAnswerVerifier,
    ItemSampler,
    OperatorSet,
    VerifierScoreFn,
    make_score_fn,
)
from research_rl.capability.types import GateEvidence, Item, Split
from research_rl.capability.weights import SnapshotError, capture
from research_rl.evaluation.metrics import pass_at_k
from research_rl.models.model_interface import fingerprint
from research_rl.models.tiny import build_char_tokenizer, build_tiny_llama
from research_rl.rewards.verifier_interface import BaseVerifier, Status

TRANSFER = ("t_rule/unseen_pairs", "t_rule/depth3", "t_task/argmax", "t_task/countbig")
RETENTION = ("retention/atomic", "retention/copy", "retention/prereq")


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


def _params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _assert_bitwise(model: nn.Module, before: dict[str, torch.Tensor]) -> None:
    now = dict(model.named_parameters())
    assert set(now) == set(before)
    for name, value in before.items():
        assert now[name].dtype == value.dtype
        assert torch.equal(now[name].detach(), value), name


def _h(uid: str) -> int:
    return int.from_bytes(hashlib.sha256(uid.encode()).digest()[:4], "big")


class MarkerScore:
    """Score depends on the sign of ``lm_head.weight[0, 0]``: 1.0 iff (hash(uid) even) == (w > 0).

    Records, per call, the fingerprint of the weights it saw and the uids in order.
    """

    def __init__(self, *, max_new_tokens: int | None = None, fail_on_call: int | None = None) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.fail_on_call = fail_on_call
        if max_new_tokens is not None:
            self.max_new_tokens = max_new_tokens

    @staticmethod
    def expected(uid: str, positive: bool) -> float:
        return 1.0 if (_h(uid) % 2 == 0) == positive else 0.0

    def __call__(self, model: nn.Module, items: Sequence[Item]) -> list[float]:
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            self.calls.append(("failed", []))
            raise RuntimeError("score function failure")
        positive = float(model.lm_head.weight[0, 0]) > 0  # type: ignore[index]
        self.calls.append((fingerprint(model), [it.uid for it in items]))
        model.train(not model.training)  # a careless score function that flips the mode
        return [self.expected(it.uid, positive) for it in items]


def _set_marker(model: nn.Module, value: float) -> None:
    with torch.no_grad():
        model.lm_head.weight[0, 0] = value  # type: ignore[index]


def _probe(sampler: ItemSampler, score_fn: Any, n_t: int = 6, n_r: int = 5) -> CapabilityProbe:
    return CapabilityProbe(
        sampler,
        score_fn,
        transfer_slices=TRANSFER,
        retention_slices=RETENTION,
        items_per_transfer_slice=n_t,
        items_per_retention_slice=n_r,
    )


# ============================================================================== construction
def test_construction_validates_slices(sampler: ItemSampler) -> None:
    stub = MarkerScore()
    with pytest.raises(AuditAccessError):
        CapabilityProbe(
            sampler,
            stub,
            transfer_slices=["audit/pairs"],
            retention_slices=[],
            items_per_transfer_slice=2,
            items_per_retention_slice=2,
        )
    with pytest.raises(AuditAccessError):
        CapabilityProbe(
            sampler,
            stub,
            transfer_slices=[],
            retention_slices=["audit/depth3"],
            items_per_transfer_slice=2,
            items_per_retention_slice=2,
        )
    for transfer, retention in (
        (["train/pairs"], []),  # TRAIN is not held out
        (["retention/atomic"], []),  # retention slice as transfer
        ([], ["t_rule/depth3"]),  # transfer slice as retention
        (["t_rule/depth3", "t_rule/depth3"], []),  # duplicate
        (["t_rule/depth3"], ["t_rule/depth3"]),  # both lists
        ([], []),  # nothing
    ):
        with pytest.raises(ValueError):
            CapabilityProbe(
                sampler,
                stub,
                transfer_slices=transfer,
                retention_slices=retention,
                items_per_transfer_slice=2,
                items_per_retention_slice=2,
            )
    with pytest.raises(KeyError):
        CapabilityProbe(
            sampler,
            stub,
            transfer_slices=["t_rule/nope"],
            retention_slices=[],
            items_per_transfer_slice=2,
            items_per_retention_slice=2,
        )
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            CapabilityProbe(
                sampler,
                stub,
                transfer_slices=TRANSFER,
                retention_slices=RETENTION,
                items_per_transfer_slice=bad,
                items_per_retention_slice=2,  # type: ignore[arg-type]
            )


def test_audit_refused_even_with_allow_audit_sampler(splits: CompositionSplits) -> None:
    open_sampler = ItemSampler(splits, root_seed=0, allow_audit=True)
    with pytest.raises(AuditAccessError):
        draw_slice(open_sampler, "audit/pairs", 3, purpose="gate/0", index=0)
    with pytest.raises(AuditAccessError):
        CapabilityProbe(
            open_sampler,
            MarkerScore(),
            transfer_slices=["audit/pairs"],
            retention_slices=[],
            items_per_transfer_slice=2,
            items_per_retention_slice=2,
        )
    probe = _probe(open_sampler, MarkerScore())
    with pytest.raises(AuditAccessError):
        probe.score_slice(nn.Linear(1, 1), "audit/depth3", 2, purpose="search", index=0)


def test_from_config_defaults(sampler: ItemSampler) -> None:
    config = GateConfig(items_per_transfer_slice=7, items_per_retention_slice=9)
    probe = CapabilityProbe.from_config(sampler, MarkerScore(), config)
    assert probe.transfer_slices == TRANSFER
    assert probe.retention_slices == RETENTION
    assert probe.items_per_transfer_slice == 7
    assert probe.items_per_retention_slice == 9
    custom = CapabilityProbe.from_config(
        sampler, MarkerScore(), config, transfer_slices=["t_rule/depth3"], retention_slices=["retention/copy"]
    )
    assert custom.transfer_slices == ("t_rule/depth3",)
    assert custom.retention_slices == ("retention/copy",)
    json.dumps(custom.to_dict())
    with pytest.raises(TypeError):
        CapabilityProbe.from_config(sampler, MarkerScore(), {"alpha": 0.05})  # type: ignore[arg-type]


def test_truncation_budget_is_validated(sampler: ItemSampler, tok: Any) -> None:
    max_len = sampler.splits.operators.max_len
    verifier = ExactAnswerVerifier()
    short = make_score_fn(tok, verifier, max_new_tokens=max_len, batch_size=4)
    with pytest.raises(ValueError, match="truncation contract"):
        _probe(sampler, short)
    with pytest.raises(ValueError, match="truncation contract"):
        check_truncation_budget(sampler, short)
    _probe(sampler, make_score_fn(tok, verifier, max_new_tokens=max_len + 1, batch_size=4))
    _probe(sampler, MarkerScore())  # no max_new_tokens: not checked
    # exact per-item check in the scoring path
    items = draw_slice(sampler, "retention/copy", 6, purpose="x", index=0)
    longest = max(len(it.answer) for it in items)
    with pytest.raises(ValueError, match="too long"):
        score_items(MarkerScore(max_new_tokens=longest), nn.Linear(1, 1), items)


# ============================================================================== evidence
def test_evidence_pairs_candidate_and_incumbent_on_same_items(sampler: ItemSampler, model: nn.Module) -> None:
    stub = MarkerScore()
    probe = _probe(sampler, stub)
    _set_marker(model, 1.0)
    incumbent = capture(model, None)
    _set_marker(model, -1.0)
    candidate_fp = fingerprint(model)
    before = _params(model)
    model.train()

    ev = probe.evidence(model, incumbent, 4, attempt=1, alarm=True)

    _assert_bitwise(model, before)
    assert model.training  # restored although the stub flipped it
    assert isinstance(ev, GateEvidence)
    assert ev.decision_index == 4 and ev.alarm is True
    assert [p.slice for p in ev.transfer] == list(TRANSFER)
    assert [p.slice for p in ev.retention] == list(RETENTION)
    want_split = {name: sampler.splits.slices()[name] for name in (*TRANSFER, *RETENTION)}
    for p in (*ev.transfer, *ev.retention):
        assert p.split is want_split[p.slice]
        assert p.n == (6 if p.slice in TRANSFER else 5)
        assert all(uid.startswith(p.slice + "|") for uid in p.uids)
        assert p.candidate == tuple(MarkerScore.expected(u, False) for u in p.uids)
        assert p.incumbent == tuple(MarkerScore.expected(u, True) for u in p.uids)
    # two calls: candidate (current weights) first, then incumbent (snapshot), same uids, same order
    assert len(stub.calls) == 2
    (fp_c, uids_c), (fp_i, uids_i) = stub.calls
    assert fp_c == candidate_fp
    assert fp_i == incumbent.fingerprint
    assert uids_c == uids_i
    assert uids_c == [u for p in (*ev.transfer, *ev.retention) for u in p.uids]
    # items are the gate-pool part of the sampler's gate stream ("gate/1", index 4), in order
    expected = probe.gate_items(4, 1)
    for p in (*ev.transfer, *ev.retention):
        assert p.uids == tuple(it.uid for it in expected[p.slice])
        stream = sampler.draw(p.split, 200, purpose="gate/1", index=4, slice=p.slice)
        direct = [it.uid for it in stream if item_pool(sampler, it.uid) == GATE_POOL][: p.n]
        assert p.uids == tuple(direct)


def test_evidence_mode_preserved_in_eval_and_verdict_accepts(sampler: ItemSampler, model: nn.Module) -> None:
    probe = _probe(sampler, MarkerScore(), n_t=8, n_r=8)
    incumbent = capture(model, None)
    model.eval()
    ev = probe.evidence(model, incumbent, 0)
    assert not model.training
    assert ev.alarm is False
    gate = CapabilityGate(
        GateConfig(alpha_schedule="per_decision", items_per_transfer_slice=8, items_per_retention_slice=8)
    )
    verdict = gate.verdict(ev)
    assert verdict.transfer.n == 8 * len(TRANSFER)
    assert [t.slice for t in verdict.retention] == list(RETENTION)
    # identical weights -> identical scores -> nothing to gain, nothing lost
    for p in (*ev.transfer, *ev.retention):
        assert p.candidate == p.incumbent
    assert not verdict.transfer_ok


def test_fresh_items_per_decision_and_attempt(sampler: ItemSampler) -> None:
    probe = _probe(sampler, MarkerScore(), n_t=12, n_r=12)

    def uids(decision: int, attempt: int) -> set[str]:
        return {it.uid for group in probe.gate_items(decision, attempt).values() for it in group}

    base = uids(0, 0)
    assert base == uids(0, 0)  # deterministic
    total = len(base)
    for other in (uids(0, 1), uids(1, 0), uids(2, 3)):
        assert len(other) == total
        # independent streams: overlap only by chance (short inputs have small domains)
        assert len(base & other) <= total // 10
    assert gate_purpose(2) == "gate/2"
    with pytest.raises(ValueError):
        gate_purpose(-1)
    with pytest.raises(ValueError):
        probe.gate_items(-1)
    # a namespace (one per version store) gives an independent stream of the same shape
    assert gate_purpose(0, "edit/0f") == "edit/0f/gate/0"
    for bad in ("", "a b", "/x", "x/", "a//b", 3):
        with pytest.raises(ValueError, match="namespace"):
            gate_purpose(0, bad)  # type: ignore[arg-type]
    named = probe.gate_items(1, 0, namespace="edit/0f")
    assert named == probe.gate_items(1, 0, namespace="edit/0f")
    assert named != probe.gate_items(1, 0) and named != probe.gate_items(1, 0, namespace="edit/1f")
    assert [len(g) for g in named.values()] == [len(g) for g in probe.gate_items(1, 0).values()]


def test_evidence_restores_weights_on_score_failure(sampler: ItemSampler, model: nn.Module) -> None:
    _set_marker(model, 1.0)
    incumbent = capture(model, None)
    _set_marker(model, -1.0)
    before = _params(model)
    model.eval()
    probe = _probe(sampler, MarkerScore(fail_on_call=1))  # fails inside the incumbent swap
    with pytest.raises(RuntimeError, match="score function failure"):
        probe.evidence(model, incumbent, 0)
    _assert_bitwise(model, before)
    assert not model.training
    # a snapshot that does not fit the model is refused before any write
    bad = capture(model, None)
    bad.params["lm_head.weight"] = bad.params["lm_head.weight"][:, :3].clone()
    with pytest.raises(SnapshotError):
        _probe(sampler, MarkerScore()).evidence(model, bad, 0)
    _assert_bitwise(model, before)
    with pytest.raises(TypeError):
        _probe(sampler, MarkerScore()).evidence(model, before, 0)  # type: ignore[arg-type]


def test_evidence_with_real_score_fn(sampler: ItemSampler, model: nn.Module, tok: Any) -> None:
    max_len = sampler.splits.operators.max_len
    score_fn = make_score_fn(tok, ExactAnswerVerifier(), max_new_tokens=max_len + 1, batch_size=16)
    probe = _probe(sampler, score_fn, n_t=3, n_r=3)
    incumbent = capture(model, None)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.sign(p))  # deterministic perturbation: the candidate differs
    before = _params(model)
    model.train()
    ev = probe.evidence(model, incumbent, 2)
    _assert_bitwise(model, before)
    assert model.training
    gate = CapabilityGate(
        GateConfig(alpha_schedule="per_decision", items_per_transfer_slice=3, items_per_retention_slice=3)
    )
    verdict = gate.verdict(ev)
    assert verdict.transfer.n == 3 * len(TRANSFER)
    # the incumbent's scores equal a direct evaluation of the incumbent weights
    restored = build_tiny_llama(len(tok), seed=0)
    restored.load_state_dict({**restored.state_dict(), **incumbent.params})
    for p in (*ev.transfer, *ev.retention):
        items = probe.gate_items(2)[p.slice]
        assert tuple(score_fn(restored, items)) == p.incumbent
        assert tuple(score_fn(model, items)) == p.candidate


# ============================================================================== score_slice
def test_score_slice(sampler: ItemSampler, model: nn.Module) -> None:
    stub = MarkerScore()
    probe = _probe(sampler, stub)
    _set_marker(model, 1.0)
    res = probe.score_slice(model, "train/pairs", 9, purpose="search/r0", index=5)
    assert isinstance(res, SliceScores)
    assert res.slice == "train/pairs" and res.split is Split.TRAIN
    assert res.purpose == "search/r0" and res.index == 5 and res.n == 9
    assert res.uids == tuple(
        it.uid for it in sampler.draw(Split.TRAIN, 9, purpose="search/r0", index=5, slice="train/pairs")
    )
    assert res.scores == tuple(MarkerScore.expected(u, True) for u in res.uids)
    assert res.mean == pytest.approx(sum(res.scores) / 9)
    again = score_slice(sampler, stub, model, "train/pairs", 9, purpose="search/r0", index=5)
    assert again == res
    d = res.to_dict()
    json.dumps(d)
    assert d["split"] == "train" and d["uids"] == list(res.uids)
    with pytest.raises(ValueError):
        probe.score_slice(model, "train/pairs", 0, purpose="x", index=0)
    with pytest.raises(KeyError):
        probe.score_slice(model, "nope", 2, purpose="x", index=0)

    class Bad:
        def __call__(self, model: nn.Module, items: Sequence[Item]) -> list[float]:
            return [2.0] * len(items)

    with pytest.raises(ValueError, match="outside"):
        score_slice(sampler, Bad(), model, "train/pairs", 2, purpose="x", index=0)

    class Short:
        def __call__(self, model: nn.Module, items: Sequence[Item]) -> list[float]:
            return [0.0]

    with pytest.raises(ValueError, match="scores for"):
        score_slice(sampler, Short(), model, "train/pairs", 2, purpose="x", index=0)
    assert score_items(stub, model, []) == []


# ============================================================================== sampling / start map
class DigitVerifier(BaseVerifier):
    """Lenient test verifier: correct iff the completion contains a digit (truncation ignored),
    so random tiny models have non-zero pass@k. Records the truncation flags it was given."""

    name = "digit"

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[str | None, bool]] = []

    def _verify(
        self, completion: str | None, reference: Any, **meta: Any
    ) -> tuple[Status, str | None, str | None, dict[str, Any]]:
        self.seen.append((completion, bool(meta.get("truncated"))))
        ok = completion is not None and any(c.isdigit() for c in completion)
        return ("correct" if ok else "incorrect"), completion, str(reference), {}


def test_sample_completions_seeded_and_rng_neutral(model: nn.Module, tok: Any) -> None:
    model.train()
    torch.manual_seed(123)
    state = torch.get_rng_state()
    a = sample_completions(model, tok, "abc(123)=", num_samples=6, max_new_tokens=8, seed=7, batch_size=4)
    assert torch.equal(torch.get_rng_state(), state)
    texts, flags = sample_completions_with_truncation(
        model, tok, "abc(123)=", num_samples=6, max_new_tokens=8, seed=7, batch_size=4
    )
    assert texts == a and len(flags) == 6 and torch.equal(torch.get_rng_state(), state)
    assert model.training
    b = sample_completions(model, tok, "abc(123)=", num_samples=6, max_new_tokens=8, seed=7, batch_size=4)
    c = sample_completions(model, tok, "abc(123)=", num_samples=6, max_new_tokens=8, seed=8, batch_size=4)
    assert a == b
    assert a != c
    assert len(a) == 6 and len(set(a)) > 1  # sampled, not greedy
    assert all(len(tok(s, add_special_tokens=False)["input_ids"]) <= 8 for s in a)
    greedy_like = sample_completions(
        model, tok, "abc(123)=", num_samples=3, max_new_tokens=8, seed=7, temperature=1e-4
    )
    assert len(set(greedy_like)) == 1
    for kwargs in ({"num_samples": 0}, {"temperature": 0.0}, {"seed": -1}, {"max_new_tokens": 0}):
        args: dict[str, Any] = {"num_samples": 2, "max_new_tokens": 4, "seed": 0, **kwargs}
        with pytest.raises(ValueError):
            sample_completions(model, tok, "x=", **args)


def test_start_map(sampler: ItemSampler, model: nn.Module, tok: Any) -> None:
    max_len = sampler.splits.operators.max_len
    verifier = DigitVerifier()
    score_fn = VerifierScoreFn(tok, verifier, max_new_tokens=max_len + 1, batch_size=8)
    slices = ["t_rule/unseen_pairs", "retention/copy"]
    before = _params(model)
    model.eval()
    torch.manual_seed(5)
    rng = torch.get_rng_state()
    # spies: raw sampled rows (do_sample generate calls) and the flags the sampled path scored with
    raw_rows: list[list[int]] = []
    sampled_flags: list[bool] = []
    generate = model.generate
    score_completions = score_fn.score_completions

    def spy_generate(**kwargs: Any) -> torch.Tensor:
        out = generate(**kwargs)
        if kwargs["generation_config"].do_sample:
            raw_rows.extend(out[:, kwargs["input_ids"].shape[1] :].tolist())
        return out

    def spy_score(completions: Any, items: Any, truncated: Any = None) -> list[float]:
        if len(completions) == 6:  # the sampled path (num_samples=6); greedy calls score 4 items
            sampled_flags.extend(bool(t) for t in truncated)
        return score_completions(completions, items, truncated)

    model.generate = spy_generate  # type: ignore[method-assign,assignment]
    score_fn.score_completions = spy_score  # type: ignore[method-assign]
    sm = StartMap.measure(
        model,
        sampler,
        score_fn,
        items_per_slice=4,
        slices=slices,
        k=4,
        num_samples=6,
        seed=11,
        sample_batch_size=3,
    )
    del model.generate, score_fn.score_completions
    assert torch.equal(torch.get_rng_state(), rng)
    _assert_bitwise(model, before)
    assert not model.training
    assert list(sm.slices) == slices
    assert sm.k == 4 and sm.num_samples == 6 and sm.items_per_slice == 4
    assert sm.splits_fingerprint == sampler.splits.fingerprint()
    total_correct = 0
    for name in slices:
        s = sm.slices[name]
        items = draw_slice(sampler, name, 4, purpose=START_PURPOSE, index=0, pool=TRACK_POOL)
        assert s.uids == tuple(it.uid for it in items)
        assert all(item_pool(sampler, u) == TRACK_POOL for u in s.uids)
        assert s.split is sampler.splits.slices()[name]
        assert s.greedy == tuple(score_fn(model, items))
        assert sm.pass1(name) == pytest.approx(sum(s.greedy) / 4)
        assert sm.pass_k(name) == pytest.approx(sum(pass_at_k(6, c, 4) for c in s.correct_counts) / 4)
        assert s.sampled_mean == pytest.approx(sum(s.correct_counts) / 24)
        total_correct += sum(s.correct_counts)
    assert 0 < total_correct < 2 * 4 * 6  # the lenient verifier gives informative counts
    # every sampled completion was scored as truncated iff its raw row has no EOS (it used the
    # whole budget) - checked against the raw generated ids, not against the implementation's rule
    assert len(raw_rows) == len(sampled_flags) == 2 * 4 * 6
    assert all(len(r) <= max_len + 1 for r in raw_rows)
    want = [tok.eos_token_id not in r for r in raw_rows]
    assert sampled_flags == want
    assert any(want) and not all(want)
    assert [t for _, t in verifier.seen if t] != []  # the flags reached the verifier
    # reproducible; another seed changes the samples
    again = StartMap.measure(
        model,
        sampler,
        score_fn,
        items_per_slice=4,
        slices=slices,
        k=4,
        num_samples=6,
        seed=11,
        sample_batch_size=3,
    )
    assert again == sm
    other = StartMap.measure(
        model,
        sampler,
        score_fn,
        items_per_slice=4,
        slices=slices,
        k=4,
        num_samples=6,
        seed=12,
        sample_batch_size=3,
    )
    assert [other.slices[n].correct_counts for n in slices] != [sm.slices[n].correct_counts for n in slices]
    # JSON round trip
    blob = json.dumps(sm.to_dict())
    assert StartMap.from_dict(json.loads(blob)) == sm
    assert set(sm.summary()) == set(slices)
    assert sm.split_mean_pass1(Split.T_RULE) == sm.pass1("t_rule/unseen_pairs")
    with pytest.raises(ValueError):
        sm.split_mean_pass1(Split.T_TASK)


def test_start_map_validation(sampler: ItemSampler, model: nn.Module, tok: Any) -> None:
    max_len = sampler.splits.operators.max_len
    score_fn = VerifierScoreFn(tok, ExactAnswerVerifier(), max_new_tokens=max_len + 1, batch_size=8)
    common: dict[str, Any] = {"items_per_slice": 1, "slices": ["retention/copy"]}
    with pytest.raises(ValueError, match="num_samples"):
        StartMap.measure(model, sampler, score_fn, k=4, num_samples=3, **common)
    with pytest.raises(TypeError):
        StartMap.measure(model, sampler, MarkerScore(), k=1, **common)
    with pytest.raises(AuditAccessError):
        StartMap.measure(model, sampler, score_fn, items_per_slice=1, slices=["audit/pairs"], k=1)
    with pytest.raises(ValueError, match="truncation contract"):
        StartMap.measure(
            model,
            sampler,
            VerifierScoreFn(tok, ExactAnswerVerifier(), max_new_tokens=max_len, batch_size=2),
            k=1,
            **common,
        )
    with pytest.raises(ValueError):
        SliceStart(
            slice="s", split=Split.T_RULE, uids=("a",), greedy=(1.0,), correct_counts=(3,), num_samples=2, k=1
        )
    with pytest.raises(ValueError):
        SliceStart(
            slice="s", split=Split.T_RULE, uids=("a",), greedy=(1.0,), correct_counts=(1,), num_samples=2, k=3
        )
    # default slices: every non-AUDIT slice, k defaults to 16 with num_samples = k
    sm = StartMap.measure(model, sampler, score_fn, items_per_slice=1, k=2)
    assert set(sm.slices) == {s for s, sp in sampler.splits.slices().items() if sp is not Split.AUDIT}
    assert sm.num_samples == 2


# ============================================================================== regressions (review wave 2)
class ModeFlipScore:
    """Never looks at the weights; flips the root's mode recursively (a careless score function)."""

    def __call__(self, model: nn.Module, items: Sequence[Item]) -> list[float]:
        model.train(not model.training)
        return [0.0] * len(items)


def _modes(model: nn.Module) -> dict[str, bool]:
    return {name: m.training for name, m in model.named_modules()}


def test_evidence_preserves_mixed_submodule_modes(sampler: ItemSampler, model: nn.Module, tok: Any) -> None:
    small = nn.Sequential(nn.Linear(1, 1), nn.Dropout(0.1))
    small.train()
    small[1].eval()  # a frozen dropout inside a training model
    before = _modes(small)
    for stub in (ModeFlipScore(), lambda m, items: [0.0] * len(items)):
        probe = CapabilityProbe(
            sampler,
            stub,
            transfer_slices=["t_rule/depth3"],
            retention_slices=["retention/copy"],
            items_per_transfer_slice=2,
            items_per_retention_slice=2,
        )
        probe.evidence(small, capture(small, None), 0)
        assert _modes(small) == before
        probe.score_slice(small, "train/pairs", 2, purpose="search", index=0)
        assert _modes(small) == before
    # the real score function (greedy_generate restores only the root mode recursively)
    max_len = sampler.splits.operators.max_len
    score_fn = make_score_fn(tok, ExactAnswerVerifier(), max_new_tokens=max_len + 1, batch_size=4)
    model.train()
    model.model.layers[0].mlp.eval()  # type: ignore[union-attr,index]
    before = _modes(model)
    _probe(sampler, score_fn, n_t=1, n_r=1).evidence(model, capture(model, None), 1)
    assert _modes(model) == before
    StartMap.measure(model, sampler, score_fn, items_per_slice=1, slices=["retention/copy"], k=1)
    assert _modes(model) == before


def test_probe_requires_a_retention_slice(sampler: ItemSampler) -> None:
    # transfer-only evidence would make the gate's retention test vacuously pass
    with pytest.raises(ValueError, match="retention slice"):
        CapabilityProbe(
            sampler,
            MarkerScore(),
            transfer_slices=["t_rule/unseen_pairs"],
            retention_slices=[],
            items_per_transfer_slice=4,
            items_per_retention_slice=4,
        )
    with pytest.raises(ValueError, match="retention slice"):
        CapabilityProbe.from_config(
            sampler, MarkerScore(), GateConfig(), transfer_slices=["t_rule/depth3"], retention_slices=[]
        )
    # retention-only is allowed: the gate's n=0 transfer test never passes (fails safe)
    probe = CapabilityProbe(
        sampler,
        MarkerScore(),
        transfer_slices=[],
        retention_slices=["retention/copy"],
        items_per_transfer_slice=4,
        items_per_retention_slice=4,
    )
    assert probe.transfer_slices == ()


def _scripted_generate(
    model: nn.Module, tok: Any, answers: dict[str, str], *, with_eos: bool
) -> list[list[int]]:
    """Replace ``model.generate`` by a stub that emits the item's answer, then (optionally) EOS,
    then pad up to ``max_new_tokens``; returns the list the raw generated rows are appended to."""
    rows: list[list[int]] = []

    def generate(
        *, input_ids: torch.Tensor, attention_mask: torch.Tensor, generation_config: Any
    ) -> torch.Tensor:
        budget = int(generation_config.max_new_tokens)
        out = []
        for prompt_ids in input_ids.tolist():
            prompt = tok.decode(prompt_ids, skip_special_tokens=True)
            body = list(tok(answers[prompt], add_special_tokens=False)["input_ids"])
            if with_eos:
                body.append(tok.eos_token_id)
            body += [tok.pad_token_id] * (budget - len(body))
            assert len(body) == budget
            rows.append(body)
            out.append(prompt_ids + body)
        return torch.tensor(out, dtype=torch.long)

    model.generate = generate  # type: ignore[method-assign,assignment]
    return rows


def test_start_map_sampled_budget_filling_completion_is_truncated(
    sampler: ItemSampler, model: nn.Module, tok: Any
) -> None:
    max_len = sampler.splits.operators.max_len
    score_fn = VerifierScoreFn(tok, ExactAnswerVerifier(), max_new_tokens=max_len + 1, batch_size=4)
    stream = draw_slice(sampler, "retention/copy", 64, purpose=START_PURPOSE, index=0)
    answers = {it.prompt: it.answer for it in stream}
    # correct answer, no EOS, pad fill: the row used the whole budget -> truncated -> 0
    rows = _scripted_generate(model, tok, answers, with_eos=False)
    sm = StartMap.measure(
        model, sampler, score_fn, items_per_slice=1, slices=["retention/copy"], k=2, num_samples=4
    )
    assert rows and all(tok.eos_token_id not in r and len(r) == max_len + 1 for r in rows)
    assert sm.slices["retention/copy"].correct_counts == (0,)
    # positive control: the same answer followed by EOS is correct
    _scripted_generate(model, tok, answers, with_eos=True)
    sm = StartMap.measure(
        model, sampler, score_fn, items_per_slice=1, slices=["retention/copy"], k=2, num_samples=4
    )
    assert sm.slices["retention/copy"].correct_counts == (4,)


def test_item_pools_partition_the_draws(sampler: ItemSampler) -> None:
    uids = [it.uid for it in draw_slice(sampler, "retention/copy", 4000, purpose="pools", index=0)]
    pools = [item_pool(sampler, u) for u in uids]
    assert set(pools) == {GATE_POOL, TRACK_POOL}
    share = pools.count(TRACK_POOL) / len(pools)
    assert abs(share - 1 / TRACK_POOL_DENOMINATOR) < 0.03
    assert pools == [item_pool(sampler, u) for u in uids]  # a fixed function of the uid
    # pool=None is the plain sampler draw; a pool keeps the stream order of its members
    plain = draw_slice(sampler, "t_rule/depth3", 50, purpose="p", index=2)
    assert plain == sampler.draw(Split.T_RULE, 50, purpose="p", index=2, slice="t_rule/depth3")
    for pool in ITEM_POOLS:
        got = draw_slice(sampler, "t_rule/depth3", 30, purpose="p", index=2, pool=pool)
        assert len(got) == 30 and len({it.uid for it in got}) == 30
        assert all(item_pool(sampler, it.uid) == pool for it in got)
        assert got == draw_slice(sampler, "t_rule/depth3", 30, purpose="p", index=2, pool=pool)
        stream = sampler.draw(Split.T_RULE, 2000, purpose="p", index=2, slice="t_rule/depth3")
        assert [it.uid for it in got] == [it.uid for it in stream if item_pool(sampler, it.uid) == pool][:30]
    gate = {
        it.uid for it in draw_slice(sampler, "retention/copy", 3000, purpose="p", index=0, pool=GATE_POOL)
    }
    track = {
        it.uid for it in draw_slice(sampler, "retention/copy", 400, purpose="p", index=0, pool=TRACK_POOL)
    }
    assert not gate & track
    with pytest.raises(ValueError, match="pool"):
        draw_slice(sampler, "t_rule/depth3", 3, purpose="p", index=0, pool="audit")
    other = ItemSampler(sampler.splits, root_seed=sampler.root_seed + 1)
    assert [item_pool(other, u) for u in uids] != pools  # the partition depends on root_seed
    res = score_slice(
        sampler, ModeFlipScore(), nn.Linear(1, 1), "retention/copy", 3, purpose="p", index=0, pool=TRACK_POOL
    )
    assert res.pool == TRACK_POOL and res.to_dict()["pool"] == TRACK_POOL
    assert all(item_pool(sampler, u) == TRACK_POOL for u in res.uids)


def test_pool_draw_reports_an_exhausted_domain(splits: CompositionSplits) -> None:
    class TinyDomain(ItemSampler):
        """Raises like ItemSampler.draw does when a slice cannot supply enough distinct items."""

        def draw(
            self, split: Split, n: int, *, purpose: str, index: int, slice: str | None = None
        ) -> list[Item]:
            if n > 40:
                raise ValueError(f"slice {slice!r}: could not draw {n} distinct items (domain too small)")
            return super().draw(split, n, purpose=purpose, index=index, slice=slice)

    small = TinyDomain(splits, root_seed=3)
    assert len(draw_slice(small, "retention/copy", 40, purpose="p", index=0)) == 40
    with pytest.raises(ValueError, match="'track' pool cannot supply 3"):
        draw_slice(small, "retention/copy", 3, purpose="p", index=0, pool=TRACK_POOL)
