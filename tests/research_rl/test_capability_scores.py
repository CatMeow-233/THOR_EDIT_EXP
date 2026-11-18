"""Tests of research_rl.capability.scores: forward-only channel statistics, scorers and grouping.

CPU only, tiny Llama built in-process. Activation statistics are compared with tolerances (they
are floating-point results of forward passes); only counts, partitions and seeded draws are
compared exactly.
"""

from __future__ import annotations

import copy
import itertools
import json
from typing import Any

import pytest
import torch
from torch import nn

from research_rl.capability.scores import (
    ChannelStats,
    collect_channel_stats,
    contrast_scores,
    density_matched_random,
    group_channels,
    layer_prior,
    random_scores,
    usage_ratio_scores,
    wanda_scores,
)
from research_rl.capability.types import ChannelGroup
from research_rl.models.ffn_channels import ChannelSelection, ChannelSelectionError
from research_rl.models.model_interface import find_swiglu_ffns, fingerprint
from research_rl.models.tiny import build_char_tokenizer, build_tiny_llama

TEXTS = [
    "zib(mok(4721))=1274",
    "x=1",
    "abc(305)=503 and some more text",
    "42",
    "rev(98765)=56789",
]
INTERMEDIATE = 64
NUM_LAYERS = 2


@pytest.fixture(scope="module")
def tok() -> Any:
    return build_char_tokenizer()


@pytest.fixture
def model(tok: Any) -> nn.Module:
    return build_tiny_llama(len(tok), seed=0)


def _manual_stats(
    model: nn.Module, tok: Any, texts: list[str], threshold: float
) -> dict[int, dict[str, Any]]:
    """Reference: per text alone, read the MLP *input* x and compute act(gate(x)) * up(x) by hand."""
    refs = find_swiglu_ffns(model)
    captured: dict[int, list[torch.Tensor]] = {r.layer_index: [] for r in refs}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for r in refs:

        def hook(mod: nn.Module, args: tuple[Any, ...], out: Any, layer: int = r.layer_index) -> None:
            x = args[0]
            h = mod.act_fn(mod.gate_proj(x)) * mod.up_proj(x)  # type: ignore[operator]
            captured[layer].append(h.detach()[0].double())

        handles.append(r.module.register_forward_hook(hook))
    model.eval()
    try:
        with torch.no_grad():
            for t in texts:
                ids = torch.tensor([tok(t, add_special_tokens=False)["input_ids"]])
                model(input_ids=ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    out: dict[int, dict[str, Any]] = {}
    for layer, chunks in captured.items():
        acts = torch.cat(chunks, dim=0)
        out[layer] = {
            "count": acts.shape[0],
            "mean_abs": acts.abs().mean(0),
            "mean_sq": (acts * acts).mean(0),
            "active_rate": (acts.abs() > threshold).double().mean(0),
            "all": acts,
        }
    return out


def _safe_threshold(model: nn.Module, tok: Any, texts: list[str]) -> float:
    """A threshold far (> 1e-6) from every |h| value of ``texts``, near the middle of the range.

    Rates compared across two collection paths (alone vs padded, merged vs concatenated, manual
    vs hooked) must not hinge on a value that sits on the strict ``>`` boundary, where float
    noise of a different batch layout or torch wheel would move a rate by 1/N.
    """
    ref = _manual_stats(model, tok, texts, threshold=0.0)
    vals = torch.cat([r["all"].abs().flatten() for r in ref.values()]).unique().sort().values
    n = vals.numel()
    lo, hi = vals[n // 4 : 3 * n // 4], vals[n // 4 + 1 : 3 * n // 4 + 1]
    i = int((hi - lo).argmax())
    thr = float((lo[i] + hi[i]) / 2)
    assert float((vals - thr).abs().min()) > 1e-6
    return thr


def _token_count(tok: Any, texts: list[str]) -> int:
    return sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts)


def _assert_stats_close(a: ChannelStats, b: ChannelStats, atol: float = 1e-6) -> None:
    assert a.layers() == b.layers()
    for layer in a.layers():
        assert int(a.count[layer]) == int(b.count[layer])
        for name in ("mean_abs", "mean_sq", "active_rate"):
            torch.testing.assert_close(getattr(a, name)[layer], getattr(b, name)[layer], atol=atol, rtol=1e-5)


def _no_down_hooks(model: nn.Module) -> bool:
    return all(len(r.down._forward_pre_hooks) == 0 for r in find_swiglu_ffns(model))


# ============================================================================== collection
def test_stats_match_manual_down_proj_input(model: nn.Module, tok: Any) -> None:
    # A positive threshold from the data: with threshold 0 the rate is ~1 for every SiLU channel.
    thr = _safe_threshold(model, tok, TEXTS)
    ref = _manual_stats(model, tok, TEXTS, threshold=thr)
    stats = collect_channel_stats(model, tok, TEXTS, batch_size=1, threshold=thr)
    assert stats.layers() == [0, 1]
    assert stats.num_texts == len(TEXTS)
    assert stats.threshold == thr
    for layer in stats.layers():
        assert int(stats.count[layer]) == ref[layer]["count"] == _token_count(tok, TEXTS)
        assert stats.count[layer].dim() == 0
        for name in ("mean_abs", "mean_sq", "active_rate"):
            got = getattr(stats, name)[layer]
            assert got.dtype == torch.float64 and got.shape == (INTERMEDIATE,)
            torch.testing.assert_close(got, ref[layer][name], atol=1e-7, rtol=1e-6)
    rate = stats.active_rate[1]
    assert float(rate.min()) > 0.0 and float(rate.max()) < 1.0  # the threshold is not vacuous
    assert bool((stats.active_rate[0] <= 1.0).all()) and bool((stats.active_rate[0] >= 0.0).all())


def test_padding_tokens_are_excluded(model: nn.Module, tok: Any) -> None:
    lengths = {len(tok(t, add_special_tokens=False)["input_ids"]) for t in TEXTS}
    assert len(lengths) > 1  # the batch really is padded
    thr = _safe_threshold(model, tok, TEXTS)
    alone = collect_channel_stats(model, tok, TEXTS, batch_size=1, threshold=thr)
    padded = collect_channel_stats(model, tok, TEXTS, batch_size=len(TEXTS), threshold=thr)
    uneven = collect_channel_stats(model, tok, TEXTS, batch_size=2, threshold=thr)
    assert int(padded.count[0]) == _token_count(tok, TEXTS)
    _assert_stats_close(alone, padded)
    _assert_stats_close(alone, uneven)
    assert 0.0 < float(padded.active_rate[1].mean()) < 1.0  # rates are not vacuous
    assert tok.padding_side == "left"  # untouched


class _EosPadTokenizer:
    """Proxy of the char tokenizer with no pad id: collection must fall back to EOS for padding.

    Unlike ``<pad>`` (id 0, the tiny Llama's ``padding_idx``, a zero embedding row) the EOS token
    has a non-zero embedding, so pad positions carry non-zero ``down_proj`` inputs and summing
    them into the statistics would be visible.
    """

    pad_token_id = None

    def __init__(self, tok: Any) -> None:
        self._tok = tok
        self.eos_token_id = tok.eos_token_id

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return self._tok(text, add_special_tokens=add_special_tokens)


def test_padding_excluded_with_nonzero_pad_embedding(model: nn.Module, tok: Any) -> None:
    eos_tok = _EosPadTokenizer(tok)
    eos = int(eos_tok.eos_token_id)
    assert eos != tok.pad_token_id
    embed = dict(model.named_parameters())["model.embed_tokens.weight"]
    assert float(embed[eos].norm()) > 1e-3  # pad rows are not zero (unlike padding_idx 0)
    thr = _safe_threshold(model, tok, TEXTS)
    # record the padded batch layout and the largest |h| on pad rows of every down_proj input
    seen_ids: list[torch.Tensor] = []
    pad_max: list[float] = []
    top = model.register_forward_pre_hook(
        lambda _m, _a, kw: seen_ids.append(kw["input_ids"].clone()), with_kwargs=True
    )
    probes = [
        r.down.register_forward_pre_hook(
            lambda _m, args: (
                pad_max.append(float(args[0][seen_ids[-1].eq(eos)].abs().max()))
                if bool(seen_ids[-1].eq(eos).any())
                else None
            )
        )
        for r in find_swiglu_ffns(model)
    ]
    try:
        padded = collect_channel_stats(model, eos_tok, TEXTS, batch_size=len(TEXTS), threshold=thr)
    finally:
        top.remove()
        for handle in probes:
            handle.remove()
    assert len(seen_ids) == 1 and bool(seen_ids[0].eq(eos).any())  # padded with EOS
    assert eos not in {i for t in TEXTS for i in tok(t, add_special_tokens=False)["input_ids"]}
    assert len(pad_max) == NUM_LAYERS and min(pad_max) > 1e-4  # pad rows really are non-zero
    alone = collect_channel_stats(model, eos_tok, TEXTS, batch_size=1, threshold=thr)
    uneven = collect_channel_stats(model, eos_tok, TEXTS, batch_size=2, threshold=thr)
    assert int(padded.count[0]) == _token_count(tok, TEXTS)
    _assert_stats_close(alone, padded)
    _assert_stats_close(alone, uneven)
    assert all(bool((padded.active_rate[k] <= 1.0).all()) for k in padded.layers())


def test_forward_gets_left_padded_mask_and_positions(model: nn.Module, tok: Any) -> None:
    texts = TEXTS[:3]
    calls: list[dict[str, Any]] = []
    handle = model.register_forward_pre_hook(
        lambda _m, _a, kw: calls.append({k: v for k, v in kw.items()}), with_kwargs=True
    )
    try:
        collect_channel_stats(model, tok, texts, batch_size=len(texts))
    finally:
        handle.remove()
    assert len(calls) == 1
    kw = calls[0]
    encoded = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    width = max(len(ids) for ids in encoded)
    ids = torch.full((len(texts), width), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros(len(texts), width, dtype=torch.long)
    positions = torch.zeros(len(texts), width, dtype=torch.long)
    for row, e in enumerate(encoded):
        ids[row, width - len(e) :] = torch.tensor(e)
        mask[row, width - len(e) :] = 1
        positions[row, width - len(e) :] = torch.arange(len(e))
    assert torch.equal(kw["input_ids"], ids)
    assert torch.equal(kw["attention_mask"], mask)
    assert torch.equal(kw["position_ids"], positions)
    assert kw["use_cache"] is False


def test_tokenizer_without_pad_or_eos(model: nn.Module, tok: Any) -> None:
    class NoPad:
        pad_token_id = None
        eos_token_id = None

        def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
            return tok(text, add_special_tokens=add_special_tokens)

    with pytest.raises(ValueError, match="pad_token_id"):
        collect_channel_stats(model, NoPad(), TEXTS)


def test_layers_subset(model: nn.Module, tok: Any) -> None:
    full = collect_channel_stats(model, tok, TEXTS[:2])
    only1 = collect_channel_stats(model, tok, TEXTS[:2], layers=[1])
    assert only1.layers() == [1]
    torch.testing.assert_close(only1.mean_abs[1], full.mean_abs[1])


@pytest.mark.parametrize("training", [True, False])
def test_hooks_removed_mode_restored_no_grad(model: nn.Module, tok: Any, training: bool) -> None:
    model.train(training)
    before = fingerprint(model)
    seen_grad: list[bool] = []
    probe = find_swiglu_ffns(model, [0])[0].down.register_forward_pre_hook(
        lambda _m, args: seen_grad.append(bool(args[0].requires_grad) or args[0].grad_fn is not None)
    )
    try:
        stats = collect_channel_stats(model, tok, TEXTS, batch_size=2)
    finally:
        probe.remove()
    assert seen_grad and not any(seen_grad)
    assert _no_down_hooks(model)
    assert model.training is training
    assert fingerprint(model) == before
    assert all(p.grad is None for p in model.parameters())
    for name in ("mean_abs", "mean_sq", "active_rate"):
        for t in getattr(stats, name).values():
            assert not t.requires_grad and t.grad_fn is None and t.device.type == "cpu"


@pytest.mark.parametrize("training", [True, False])
def test_hooks_removed_after_exception_mid_forward(model: nn.Module, tok: Any, training: bool) -> None:
    model.train(training)

    def boom(*_: Any) -> None:
        raise RuntimeError("boom")

    # layer 0's down_proj hook has already run when layer 1's MLP raises
    handle = find_swiglu_ffns(model, [1])[0].module.register_forward_hook(boom)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            collect_channel_stats(model, tok, TEXTS)
    finally:
        handle.remove()
    assert _no_down_hooks(model)
    assert model.training is training
    # the model still works and collecting again succeeds
    assert collect_channel_stats(model, tok, TEXTS[:1]).num_texts == 1


def _modes(model: nn.Module) -> dict[str, bool]:
    return {name: m.training for name, m in model.named_modules()}


@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("training", [True, False])
def test_mixed_submodule_modes_restored(model: nn.Module, tok: Any, training: bool, raises: bool) -> None:
    model.train(training)
    inner = model.get_submodule("model.layers.0")
    inner.train(not training)  # a submodule in the opposite mode of its parent
    model.get_submodule("model.layers.1.mlp").train(not training)
    before = _modes(model)
    assert len(set(before.values())) == 2
    seen_eval: list[bool] = []
    probe = find_swiglu_ffns(model, [0])[0].down.register_forward_pre_hook(
        lambda m, _a: seen_eval.append(not m.training and not inner.training)
    )
    boom = None
    if raises:

        def fail(*_: Any) -> None:
            raise RuntimeError("boom")

        boom = find_swiglu_ffns(model, [1])[0].module.register_forward_hook(fail)
    try:
        if raises:
            with pytest.raises(RuntimeError, match="boom"):
                collect_channel_stats(model, tok, TEXTS[:2])
        else:
            collect_channel_stats(model, tok, TEXTS[:2])
    finally:
        probe.remove()
        if boom is not None:
            boom.remove()
    assert seen_eval and all(seen_eval)  # eval mode during the forward
    assert _modes(model) == before
    assert _no_down_hooks(model)


def test_channel_stats_equality(model: nn.Module, tok: Any) -> None:
    a = collect_channel_stats(model, tok, TEXTS[:2], threshold=0.02)
    same = copy.deepcopy(a)
    assert a == same
    assert (a != same) is False
    assert [a, {"s": a}] == [same, {"s": same}]
    assert a == ChannelStats.from_dict(json.loads(json.dumps(a.to_dict())))
    changed = copy.deepcopy(a)
    changed.mean_sq[1][5] += 1e-12
    assert a != changed
    assert a != ChannelStats.from_dict({**a.to_dict(), "num_texts": 7})
    assert a != ChannelStats.from_dict({**a.to_dict(), "threshold": 0.5})
    as_float32 = copy.deepcopy(a)
    as_float32.mean_abs[0] = as_float32.mean_abs[0].float()
    assert a != as_float32  # dtype is part of equality
    assert a != collect_channel_stats(model, tok, TEXTS[:2], layers=[0], threshold=0.02)
    assert a != "not stats"
    with pytest.raises(TypeError):
        hash(a)


def test_bad_arguments(model: nn.Module, tok: Any) -> None:
    with pytest.raises(ValueError, match="texts"):
        collect_channel_stats(model, tok, [])
    with pytest.raises(ValueError, match="no token"):
        collect_channel_stats(model, tok, ["ab", ""])
    with pytest.raises(ValueError, match="batch_size"):
        collect_channel_stats(model, tok, TEXTS, batch_size=0)
    with pytest.raises(ValueError, match="threshold"):
        collect_channel_stats(model, tok, TEXTS, threshold=-1.0)
    model.train()
    with pytest.raises(ValueError, match="out of range"):  # failure while resolving layers
        collect_channel_stats(model, tok, TEXTS, layers=[0, 7])
    assert _no_down_hooks(model)
    assert model.training


def test_to_dict_json_roundtrip_and_merge(model: nn.Module, tok: Any) -> None:
    thr = _safe_threshold(model, tok, TEXTS)
    a = collect_channel_stats(model, tok, TEXTS[:2], threshold=thr)
    b = collect_channel_stats(model, tok, TEXTS[2:], threshold=thr)
    both = collect_channel_stats(model, tok, TEXTS, threshold=thr)
    d = json.loads(json.dumps(a.to_dict()))
    assert set(d) == {"num_texts", "threshold", "count", "mean_abs", "mean_sq", "active_rate"}
    assert isinstance(d["mean_abs"]["0"], list) and len(d["mean_abs"]["0"]) == INTERMEDIATE
    back = ChannelStats.from_dict(d)
    assert back.num_texts == a.num_texts and back.threshold == a.threshold
    for layer in a.layers():
        assert torch.equal(back.count[layer], a.count[layer])
        for name in ("mean_abs", "mean_sq", "active_rate"):
            assert torch.equal(getattr(back, name)[layer], getattr(a, name)[layer])
    merged = a.merge(b)
    assert merged.num_texts == len(TEXTS)
    _assert_stats_close(merged, both, atol=1e-9)
    with pytest.raises(ValueError, match="threshold"):
        a.merge(collect_channel_stats(model, tok, TEXTS[:1], threshold=0.5))
    with pytest.raises(ValueError, match="layers"):
        a.merge(collect_channel_stats(model, tok, TEXTS[:1], layers=[0], threshold=thr))


def test_channel_stats_validation() -> None:
    z = torch.zeros(4, dtype=torch.float64)
    c = torch.tensor(3, dtype=torch.int64)
    with pytest.raises(ValueError, match="layers"):
        ChannelStats(count={0: c}, mean_abs={1: z}, mean_sq={0: z}, active_rate={0: z}, num_texts=1)
    with pytest.raises(ValueError, match="equal length"):
        ChannelStats(
            count={0: c}, mean_abs={0: z}, mean_sq={0: torch.zeros(3)}, active_rate={0: z}, num_texts=1
        )
    with pytest.raises(ValueError, match="count"):
        ChannelStats(
            count={0: torch.tensor(1.0)}, mean_abs={0: z}, mean_sq={0: z}, active_rate={0: z}, num_texts=1
        )
    empty = ChannelStats(
        count={0: torch.tensor(0)}, mean_abs={0: z}, mean_sq={0: z}, active_rate={0: z}, num_texts=0
    )
    merged = empty.merge(empty)
    assert int(merged.count[0]) == 0 and torch.equal(merged.mean_abs[0], z)
    with pytest.raises(ValueError, match="no tokens"):
        contrast_scores(empty, empty)
    with pytest.raises(ValueError, match="no tokens"):
        usage_ratio_scores(empty, empty)
    with pytest.raises(ValueError, match="num_texts"):
        ChannelStats(count={0: c}, mean_abs={0: z}, mean_sq={0: z}, active_rate={0: z}, num_texts=-1)
    with pytest.raises(ValueError, match="threshold"):
        ChannelStats(
            count={0: c}, mean_abs={0: z}, mean_sq={0: z}, active_rate={0: z}, num_texts=1, threshold=-0.5
        )
    z5 = torch.zeros(5, dtype=torch.float64)
    wider = ChannelStats(count={0: c}, mean_abs={0: z5}, mean_sq={0: z5}, active_rate={0: z5}, num_texts=1)
    with pytest.raises(ValueError, match="intermediate sizes"):
        empty.merge(wider)


def test_wanda_rejects_empty_layer(model: nn.Module) -> None:
    z = torch.zeros(INTERMEDIATE, dtype=torch.float64)
    empty = ChannelStats(
        count={0: torch.tensor(0)}, mean_abs={0: z}, mean_sq={0: z}, active_rate={0: z}, num_texts=0
    )
    with pytest.raises(ValueError, match="no tokens"):
        wanda_scores(model, empty)


# ============================================================================== scorers
@pytest.fixture
def stats_pair(model: nn.Module, tok: Any) -> tuple[ChannelStats, ChannelStats]:
    return (
        collect_channel_stats(model, tok, TEXTS[:3], threshold=0.02),
        collect_channel_stats(model, tok, TEXTS[3:], threshold=0.02),
    )


def _check_shapes(scores: dict[int, torch.Tensor]) -> None:
    assert sorted(scores) == list(range(NUM_LAYERS))
    for s in scores.values():
        assert s.dim() == 1 and s.numel() == INTERMEDIATE and s.dtype == torch.float64
        assert bool(torch.isfinite(s).all())


def test_wanda_scores(model: nn.Module, tok: Any, stats_pair: tuple[ChannelStats, ChannelStats]) -> None:
    stats = stats_pair[0]
    s = wanda_scores(model, stats)
    _check_shapes(s)
    for r in find_swiglu_ffns(model):
        norm = r.down.weight.detach().double().norm(dim=0)
        torch.testing.assert_close(s[r.layer_index], norm * stats.mean_sq[r.layer_index].sqrt())
    assert all(bool((v >= 0).all()) for v in s.values())
    s2 = wanda_scores(model, collect_channel_stats(model, tok, TEXTS[:3], threshold=0.02))
    for layer in s:
        assert torch.equal(s[layer], s2[layer])  # deterministic
    other = build_tiny_llama(len(tok), seed=0, intermediate_size=32)
    with pytest.raises(ValueError, match="intermediate size"):
        wanda_scores(other, stats)


def test_contrast_and_usage_ratio(stats_pair: tuple[ChannelStats, ChannelStats]) -> None:
    a, b = stats_pair
    c = contrast_scores(a, b)
    _check_shapes(c)
    for layer in c:
        torch.testing.assert_close(c[layer], a.mean_abs[layer] - b.mean_abs[layer])
    assert all(torch.equal(v, torch.zeros_like(v)) for v in contrast_scores(a, a).values())
    u = usage_ratio_scores(a, b, eps=1e-2)
    _check_shapes(u)
    for layer in u:
        torch.testing.assert_close(u[layer], a.active_rate[layer] / (b.active_rate[layer] + 1e-2))
        assert float(u[layer].max()) <= 1e2
    assert float(u[1].std()) > 0  # the ratio actually varies with a positive threshold
    u2 = usage_ratio_scores(a, b, eps=1e-2)
    assert all(torch.equal(u[k], u2[k]) for k in u)
    with pytest.raises(ValueError, match="eps"):
        usage_ratio_scores(a, b, eps=0.0)
    other_thr = ChannelStats.from_dict({**b.to_dict(), "threshold": 0.5})
    with pytest.raises(ValueError, match="thresholds"):
        usage_ratio_scores(a, other_thr)
    one_layer = ChannelStats.from_dict(
        {k: ({"0": v["0"]} if isinstance(v, dict) else v) for k, v in b.to_dict().items()}
    )
    with pytest.raises(ValueError, match="layers"):
        contrast_scores(a, one_layer)


def test_random_scores(model: nn.Module) -> None:
    rng_before = torch.get_rng_state()
    s = random_scores(model, seed=3)
    assert torch.equal(torch.get_rng_state(), rng_before)
    _check_shapes(s)
    assert all(bool(((v >= 0) & (v < 1)).all()) for v in s.values())
    again = random_scores(model, seed=3)
    other = random_scores(model, seed=4)
    assert all(torch.equal(s[k], again[k]) for k in s)
    assert not torch.equal(s[0], other[0]) and not torch.equal(s[1], other[1])
    assert not torch.equal(s[0], s[1])
    only1 = random_scores(model, seed=3, layers=[1])
    assert list(only1) == [1] and torch.equal(only1[1], s[1])
    with pytest.raises(ValueError, match="not FFN layers"):
        random_scores(model, seed=3, layers=[5])


def test_layer_prior(model: nn.Module) -> None:
    s = random_scores(model, seed=0)
    original = {k: v.clone() for k, v in s.items()}
    p = layer_prior(s, {1: 2.5})
    assert torch.equal(p[0], s[0])
    torch.testing.assert_close(p[1], s[1] * 2.5)
    assert all(torch.equal(s[k], original[k]) for k in s)
    zeroed = layer_prior(s, {0: 0.0})
    assert torch.equal(zeroed[0], torch.zeros(INTERMEDIATE, dtype=torch.float64))
    sel = ChannelSelection.top_k_from_scores(zeroed, 10)
    assert sel.layers() == [1]
    with pytest.raises(ValueError, match="no scores"):
        layer_prior(s, {9: 1.0})
    with pytest.raises(ValueError, match="finite"):
        layer_prior(s, {0: -1.0})
    with pytest.raises(ValueError, match="finite"):
        layer_prior(s, {0: float("nan")})


def test_scores_integrate_with_top_k(model: nn.Module, stats_pair: tuple[ChannelStats, ChannelStats]) -> None:
    a, b = stats_pair
    for scores in (
        wanda_scores(model, a),
        contrast_scores(a, b),
        usage_ratio_scores(a, b),
        random_scores(model, seed=1),
    ):
        sel = ChannelSelection.top_k_from_scores(scores, 12)
        assert sel.count() == 12
        sel.validate(model)
        per = ChannelSelection.top_k_from_scores(scores, 5, per_layer=True)
        assert per.to_dict().keys() == {0, 1} and all(len(v) == 5 for v in per.to_dict().values())
        flat = sorted(
            ((float(scores[k][c]), k, c) for k in scores for c in range(INTERMEDIATE)), reverse=True
        )
        cutoff = flat[11][0]
        assert all(float(scores[k][c]) >= cutoff for k, chans in sel.items() for c in chans)


# ============================================================================== grouping
def _check_partition(groups: list[ChannelGroup], model: nn.Module, layers: list[int]) -> None:
    names = [g.name for g in groups]
    assert len(set(names)) == len(names)
    seen: set[tuple[int, int]] = set()
    for g in groups:
        pairs = {(lay, c) for lay, chans in g.selection.items() for c in chans}
        assert pairs and not (pairs & seen)
        seen |= pairs
    full = ChannelSelection.all_channels(model, layers)
    assert seen == {(lay, c) for lay, chans in full.items() for c in chans}


@pytest.mark.parametrize("bands", [1, 3, 4, 5])
def test_group_channels_partitions(model: nn.Module, bands: int) -> None:
    s = random_scores(model, seed=2)
    groups = group_channels(s, bands=bands)
    _check_partition(groups, model, [0, 1])
    assert [g.name for g in groups] == [f"L{lay}/b{b}" for lay in (0, 1) for b in range(bands)]
    sizes = [g.selection.count() for g in groups if g.name.startswith("L0/")]
    assert max(sizes) - min(sizes) <= 1
    # b0 holds the highest scores: every score in band b >= every score in band b+1
    for lay in (0, 1):
        band_vals = [s[lay][list(g.selection.channels(lay))] for g in groups if g.name.startswith(f"L{lay}/")]
        for hi, lo in itertools.pairwise(band_vals):
            assert float(hi.min()) >= float(lo.max())
    again = group_channels(s, bands=bands)
    assert [(g.name, g.selection) for g in again] == [(g.name, g.selection) for g in groups]


def test_group_channels_ties_small_layers_and_filter(model: nn.Module) -> None:
    ties = {0: torch.zeros(INTERMEDIATE), 1: torch.ones(INTERMEDIATE)}
    groups = group_channels(ties, bands=4)
    _check_partition(groups, model, [0, 1])
    assert groups[0].selection.channels(0) == tuple(range(16))  # ties broken by channel index
    tiny = {3: torch.tensor([0.1, 0.9, 0.5])}
    small = group_channels(tiny, bands=5)
    assert [(g.name, g.selection.to_dict()) for g in small] == [
        ("L3/b0", {3: [1]}),
        ("L3/b1", {3: [2]}),
        ("L3/b3", {3: [0]}),
    ]
    only = group_channels(random_scores(model, seed=0), bands=2, layers=[1])
    assert [g.name for g in only] == ["L1/b0", "L1/b1"]
    _check_partition(only, model, [1])
    with pytest.raises(ValueError, match="no scores"):
        group_channels(tiny, layers=[0])
    with pytest.raises(ValueError, match="bands"):
        group_channels(tiny, bands=0)
    with pytest.raises(ValueError, match="NaN"):
        group_channels({0: torch.tensor([1.0, float("nan")])})
    with pytest.raises(ValueError, match="1-D"):
        group_channels({0: torch.zeros(2, 2)})


def test_density_matched_random(model: nn.Module) -> None:
    sel = ChannelSelection({0: range(10), 1: [3, 7, 40]})
    rng_before = torch.get_rng_state()
    r = density_matched_random(sel, model, seed=0)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert r.layers() == sel.layers()
    assert {k: len(v) for k, v in r.items()} == {k: len(v) for k, v in sel.items()}
    r.validate(model)
    assert density_matched_random(sel, model, seed=0) == r
    draws = [density_matched_random(sel, model, seed=s) for s in range(20)]
    assert sum(d != sel for d in draws) >= 19  # P(identical) per draw is ~1e-17
    assert len(set(draws)) >= 19  # different seeds give different selections
    assert density_matched_random(ChannelSelection(), model, seed=0) == ChannelSelection()
    with pytest.raises(ChannelSelectionError):
        density_matched_random(ChannelSelection({0: [INTERMEDIATE]}), model, seed=0)


def test_ffn_bypassed_by_forward_raises(model: nn.Module, tok: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    mlp = find_swiglu_ffns(model, [1])[0].module
    monkeypatch.setattr(mlp, "forward", lambda x: torch.zeros_like(x))
    with pytest.raises(RuntimeError, match="never called"):
        collect_channel_stats(model, tok, TEXTS[:1])
    assert _no_down_hooks(model)
