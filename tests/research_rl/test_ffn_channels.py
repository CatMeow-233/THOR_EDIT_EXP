"""SwiGLU FFN channel operations, checked against independent plain-torch references."""

from __future__ import annotations

import copy
import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from research_rl.models.expert_manager import ExpertManager
from research_rl.models.ffn_channels import (
    ChannelDiff,
    ChannelSelection,
    ChannelSelectionError,
    ChannelState,
    ScopeViolationError,
    ablate_channels,
    channel_diff,
    channel_mask,
    channel_state,
    extract_channels,
    freeze_mask,
    ranked_channels,
    rollback_channels,
    setup_channel_training,
    top_changed_channels,
    verify_modification_scope,
)
from research_rl.models.model_interface import snapshot
from research_rl.models.parameter_mask import ParameterMask
from research_rl.models.tiny import build_tiny_llama
from research_rl.optim import HookedOptimizer
from research_rl.trainers.base import research_trainer_class

HIDDEN, INTER, LAYERS = 32, 64, 2
PROJ = ("gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------------------------- independent helpers
def wname(layer: int, proj: str) -> str:
    return f"model.layers.{layer}.mlp.{proj}.weight"


def raw(model, layer: int) -> dict[str, torch.Tensor]:
    params = dict(model.named_parameters())
    return {p: params[wname(layer, p)].detach().clone() for p in PROJ}


def manual_ffn(w: dict[str, torch.Tensor], x: torch.Tensor, keep: list[int] | None = None) -> torch.Tensor:
    """y = W_down[:, keep] @ (silu(W_gate[keep] x) * (W_up[keep] x)), written with plain matmuls."""
    keep = list(range(w["gate_proj"].shape[0])) if keep is None else keep
    g, u, d = w["gate_proj"][keep], w["up_proj"][keep], w["down_proj"][:, keep]
    return (F.silu(x @ g.T) * (x @ u.T)) @ d.T


def manual_contributions(w: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """[..., channel, hidden]: channel j's additive contribution h_j * W_down[:, j]."""
    h = F.silu(x @ w["gate_proj"].T) * (x @ w["up_proj"].T)
    return h.unsqueeze(-1) * w["down_proj"].T


def manual_allowed(selection: dict[int, list[int]], model) -> dict[str, torch.Tensor]:
    """Bool masks for every parameter: True exactly on selected gate/up rows and down columns."""
    out = {n: torch.zeros(p.shape, dtype=torch.bool) for n, p in model.named_parameters()}
    for layer, chans in selection.items():
        out[wname(layer, "gate_proj")][chans, :] = True
        out[wname(layer, "up_proj")][chans, :] = True
        out[wname(layer, "down_proj")][:, chans] = True
    return out


def make_model(tokenizer, dtype=torch.float32, seed: int = 0):
    return build_tiny_llama(len(tokenizer), seed=seed).to(dtype)


def random_ids(tokenizer, step: int, batch: int = 4, length: int = 12) -> torch.Tensor:
    g = torch.Generator().manual_seed(500 + step)
    return torch.randint(4, len(tokenizer), (batch, length), generator=g)


# ---------------------------------------------------------------------------------- ChannelSelection
def test_selection_normalises_validates_and_combines(tiny_model):
    s = ChannelSelection({1: [5, 3, 3, 0], 0: torch.tensor([7]), 2: []})
    assert s.to_dict() == {0: [7], 1: [0, 3, 5]} and s.layers() == [0, 1] and s.count() == 4
    assert (1, 3) in s and (1, 4) not in s and 2 not in s
    assert ChannelSelection.from_dict({"1": [2], "0": [1]}) == ChannelSelection({0: [1], 1: [2]})
    t = ChannelSelection({1: [5, 6], 0: [8]})
    assert s.union(t).to_dict() == {0: [7, 8], 1: [0, 3, 5, 6]}
    assert s.intersection(t).to_dict() == {1: [5]}
    assert s.difference(t).to_dict() == {0: [7], 1: [0, 3]}
    assert s.complement(tiny_model).to_dict() == {
        0: [c for c in range(INTER) if c != 7],
        1: [c for c in range(INTER) if c not in (0, 3, 5)],
    }
    s.validate(tiny_model)
    for bad in ({0: [INTER]}, {LAYERS: [0]}):
        with pytest.raises(ChannelSelectionError):
            ChannelSelection(bad).validate(tiny_model)
    for bad in ({0: [-1]}, {0: [1.0]}, {0: [True]}):
        with pytest.raises(ChannelSelectionError):
            ChannelSelection(bad)
    with pytest.raises(ChannelSelectionError):
        ChannelSelection({0: [64]}, intermediate_size=64)


def test_from_dict_rejects_non_integer_layer_keys():
    assert ChannelSelection.from_dict({2: [1], "0": [4]}).to_dict() == {0: [4], 2: [1]}
    for bad in ({1.9: [0]}, {True: [0]}, {"x": [0]}, {"1.0": [0]}, {"-1": [0]}, {"1": [0], 1: [1]}):
        with pytest.raises(ChannelSelectionError):
            ChannelSelection.from_dict(bad)


def test_top_k_from_scores_matches_sorted_reference():
    g = torch.Generator().manual_seed(0)
    scores = {0: torch.rand(10, generator=g), 3: torch.rand(10, generator=g)}
    scores[3][4] = scores[0][2] = 5.0  # tie: broken by (layer, channel)
    flat = sorted(
        ((float(v), layer, c) for layer, s in scores.items() for c, v in enumerate(s)),
        key=lambda e: (-e[0], e[1], e[2]),
    )
    ref: dict[int, list[int]] = {}
    for _, layer, c in flat[:5]:
        ref.setdefault(layer, []).append(c)
    top = ChannelSelection.top_k_from_scores(scores, 5)
    assert top == ChannelSelection(ref)
    assert ChannelSelection.top_k_from_scores(scores, 1).to_dict() == {0: [2]}
    per = ChannelSelection.top_k_from_scores(scores, 2, per_layer=True)
    assert per.to_dict() == {k: sorted(torch.topk(v, 2).indices.tolist()) for k, v in scores.items()}
    low: dict[int, list[int]] = {}
    for _, layer, c in sorted(flat, key=lambda e: (e[0], e[1], e[2]))[:4]:
        low.setdefault(layer, []).append(c)
    assert ChannelSelection.top_k_from_scores(scores, 4, largest=False) == ChannelSelection(low)
    with pytest.raises(ChannelSelectionError):
        ChannelSelection.top_k_from_scores({0: torch.tensor([1.0, float("nan")])}, 1)


# ---------------------------------------------------------------------------------- masks
def test_channel_mask_and_freeze_mask_match_manual(tiny_model):
    sel = {0: [1, 7, 63], 1: [0]}
    mask = channel_mask(tiny_model, ChannelSelection(sel))
    ref = manual_allowed(sel, tiny_model)
    assert sorted(mask.names()) == sorted(wname(layer, p) for layer in sel for p in PROJ)
    for name in mask.names():
        assert torch.equal(mask.get(name), ref[name]), name
    assert mask.num_trainable() == 4 * 3 * HIDDEN
    frozen = freeze_mask(tiny_model, ChannelSelection(sel))
    for name in mask.names():
        assert torch.equal(frozen.get(name), ~ref[name]), name


def biased_model(seed: int = 1):
    """Tiny Llama whose layer-0 FFN projections all carry biases."""
    model = build_tiny_llama(40, seed=seed).double()
    mlp = model.model.layers[0].mlp
    g = torch.Generator().manual_seed(seed)
    for proj in PROJ:
        old = getattr(mlp, proj)
        new = torch.nn.Linear(old.in_features, old.out_features, bias=True, dtype=torch.float64)
        with torch.no_grad():
            new.weight.copy_(old.weight)
            new.bias.copy_(torch.randn(old.out_features, generator=g, dtype=torch.float64))
        setattr(mlp, proj, new)
    return model


def test_bias_entries_are_owned_by_channels():
    model = biased_model()
    chans = [2, 5]
    sel = ChannelSelection({0: chans})
    owned = torch.tensor([i in chans for i in range(INTER)])
    mask = channel_mask(model, sel)
    bname = "model.layers.0.mlp.{}.bias".format
    assert torch.equal(mask.get(bname("gate_proj")), owned)
    assert torch.equal(mask.get(bname("up_proj")), owned)
    assert not mask.get(bname("down_proj")).any()  # hidden-space bias: no owner
    frozen = freeze_mask(model, sel)
    assert torch.equal(frozen.get(bname("up_proj")), ~owned) and frozen.get(bname("down_proj")).all()

    base_sd, base = snapshot(model), channel_state(model, [0])
    assert torch.equal(base[0].gate_bias, base_sd[bname("gate_proj")])
    trained = perturbed_copy(model, seed=13)  # every weight AND bias element moves
    trained_sd = snapshot(trained)
    diff = channel_diff(base, trained)[0]
    manual_bias = torch.stack(
        [trained_sd[bname(p)] - base_sd[bname(p)] for p in ("gate_proj", "up_proj")], dim=1
    )
    torch.testing.assert_close(
        diff.per_projection_l2["bias"], torch.linalg.vector_norm(manual_bias, dim=1), rtol=1e-12, atol=0
    )
    report = rollback_channels(trained, base, sel)
    # hybrid reference: owned weight slices AND owned gate/up bias entries back to base; down bias stays
    allowed = manual_allowed({0: chans}, model)
    allowed[bname("gate_proj")] = owned.clone()
    allowed[bname("up_proj")] = owned.clone()
    after = snapshot(trained)
    for n in base_sd:
        assert torch.equal(after[n], torch.where(allowed[n], base_sd[n], trained_sd[n])), n
    assert report.elements_written == len(chans) * (3 * HIDDEN + 2)
    assert report.total_elements_changed == len(chans) * (3 * HIDDEN + 2)


def test_rollback_rejects_bias_presence_mismatch_without_writing():
    with_bias, plain = biased_model(seed=2), build_tiny_llama(40, seed=2).double()
    sel = ChannelSelection({0: [1]})
    for model, ref in ((with_bias, channel_state(plain)), (plain, channel_state(with_bias))):
        before = snapshot(model)
        with pytest.raises(ChannelSelectionError, match="bias presence"):
            rollback_channels(model, ref, sel)
        assert all(torch.equal(before[n], t) for n, t in snapshot(model).items())


# ---------------------------------------------------------------------------------- (a) localisation
def test_ablation_equals_manual_ffn_without_those_channels(tiny_model):
    model = tiny_model.double()
    sel = {0: [2], 1: [0, 5, 17, 63]}
    x = torch.randn(3, 5, HIDDEN, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    w = {layer: raw(model, layer) for layer in range(LAYERS)}
    out_full = {layer: model.model.layers[layer].mlp(x) for layer in range(LAYERS)}
    before = snapshot(model)

    previous = ablate_channels(model, ChannelSelection(sel))
    for layer, chans in sel.items():
        keep = [c for c in range(INTER) if c not in chans]
        got = model.model.layers[layer].mlp(x)
        torch.testing.assert_close(got, manual_ffn(w[layer], x, keep), rtol=0, atol=1e-12)
        assert (got - out_full[layer]).abs().max() > 1e-5  # the ablation really mattered
    report = verify_modification_scope(before, model, ChannelSelection(sel), model)
    assert report.ok and report.changed_allowed > 0

    rollback_channels(model, previous)
    for layer in range(LAYERS):
        assert torch.equal(model.model.layers[layer].mlp(x), out_full[layer])
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())


@pytest.mark.parametrize("proj", PROJ)
def test_perturbing_one_channel_slice_acts_only_through_that_channel(tiny_model, proj):
    model = tiny_model.double()
    layer, c = 1, 7
    x = torch.randn(4, HIDDEN, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    mlp = model.model.layers[layer].mlp
    w0 = raw(model, layer)
    y0, contrib0 = mlp(x), manual_contributions(w0, x)
    reference, before = channel_state(model), snapshot(model)

    # Perturb ONLY `proj`'s slice of channel c, writing it through rollback_channels.
    target = extract_channels(model, ChannelSelection({layer: [c]}))
    noise = torch.randn(HIDDEN, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    if proj == "down_proj":
        target[layer].down[:, 0] += noise
    else:
        getattr(target[layer], proj.split("_")[0])[0] += noise
    rollback_channels(model, target)

    w1 = raw(model, layer)
    contrib1 = manual_contributions(w1, x)
    others = [j for j in range(INTER) if j != c]
    assert torch.equal(contrib0[:, others], contrib1[:, others])  # other channels untouched bit-for-bit
    assert not torch.equal(contrib0[:, c], contrib1[:, c])
    torch.testing.assert_close(mlp(x) - y0, contrib1[:, c] - contrib0[:, c], rtol=0, atol=1e-12)

    diff = channel_diff(reference, model)
    assert diff[layer].changed_channels() == [c] and diff[1 - layer].changed_channels() == []
    torch.testing.assert_close(diff[layer].l2[c], torch.linalg.vector_norm(noise), rtol=1e-12, atol=0)
    for p in PROJ:
        assert (diff[layer].per_projection_l2[p][c] > 0) == (p == proj)
    assert top_changed_channels(diff, 1) == ChannelSelection({layer: [c]})
    assert top_changed_channels(diff, 5, min_score=0.0) == ChannelSelection({layer: [c]})

    assert verify_modification_scope(before, model, ChannelSelection({layer: [c]}), model).ok
    report = verify_modification_scope(before, model, ChannelSelection({layer: [c + 1]}), model)
    assert list(report.violations) == [wname(layer, proj)]
    v = report.violations[wname(layer, proj)]
    expected = [[c, j] for j in range(HIDDEN)] if proj != "down_proj" else [[j, c] for j in range(HIDDEN)]
    assert v.count == HIDDEN and v.indices.tolist() == expected
    with pytest.raises(ScopeViolationError):
        verify_modification_scope(before, model, ChannelSelection({layer: [c + 1]}), model, strict=True)


# ---------------------------------------------------------------------------------- (b) rollback
def perturbed_copy(model, seed: int):
    trained = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in trained.parameters():
            p.add_((torch.randn(p.shape, generator=g) * 0.05).to(p.dtype))
    return trained


def test_rollback_restores_only_selected_channels_and_matches_hybrid_reference(tiny_model):
    base = tiny_model.double()
    trained = perturbed_copy(base, seed=11)
    sel = {0: [1, 4, 9, 63], 1: [0, 32]}
    base_sd, trained_sd = snapshot(base), snapshot(trained)
    assert all((base_sd[n] != trained_sd[n]).all() for n in base_sd)  # every element differs

    report = rollback_channels(trained, channel_state(base), ChannelSelection(sel))
    after = snapshot(trained)

    allowed = manual_allowed(sel, base)
    hybrid = {n: torch.where(allowed[n], base_sd[n], trained_sd[n]) for n in base_sd}
    for n in base_sd:
        assert torch.equal(after[n], hybrid[n]), n
        assert after[n].dtype == trained_sd[n].dtype
    assert report.channels == sel
    assert report.elements_written == 6 * 3 * HIDDEN
    assert sum(report.elements_changed.values()) == 6 * 3 * HIDDEN
    for layer, chans in sel.items():
        d = torch.cat(
            [
                trained_sd[wname(layer, "gate_proj")][chans] - base_sd[wname(layer, "gate_proj")][chans],
                trained_sd[wname(layer, "up_proj")][chans] - base_sd[wname(layer, "up_proj")][chans],
                (
                    trained_sd[wname(layer, "down_proj")][:, chans]
                    - base_sd[wname(layer, "down_proj")][:, chans]
                ).T,
            ],
            dim=1,
        )
        torch.testing.assert_close(
            report.delta_l2[layer], torch.linalg.vector_norm(d, dim=1), rtol=1e-12, atol=0
        )

    # FFN outputs equal the manual formula on the hand-built hybrid weights...
    x = torch.randn(3, HIDDEN, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    for layer in range(LAYERS):
        hw = {p: hybrid[wname(layer, p)] for p in PROJ}
        torch.testing.assert_close(trained.model.layers[layer].mlp(x), manual_ffn(hw, x), rtol=0, atol=1e-12)
    # ...and the whole model is bit-identical to one loaded from the hybrid state dict.
    ref_model = copy.deepcopy(base)
    ref_model.load_state_dict(hybrid, strict=False)
    ids = torch.randint(4, 90, (2, 9), generator=torch.Generator().manual_seed(5))
    assert torch.equal(trained(input_ids=ids).logits, ref_model(input_ids=ids).logits)
    # diff after rollback: selected channels are back at zero distance, the rest still differ
    diff = channel_diff(channel_state(base), trained)
    for layer in range(LAYERS):
        assert diff[layer].changed_channels() == [c for c in range(INTER) if c not in sel.get(layer, [])]


def test_rollback_is_dtype_preserving_and_exact_in_bf16(tokenizer):
    base = make_model(tokenizer, torch.bfloat16)
    trained = perturbed_copy(base, seed=12)
    sel = {1: [3, 30]}
    ref_state = channel_state(base)
    assert ref_state[1].gate.dtype == torch.bfloat16
    trained_sd = snapshot(trained)
    rollback_channels(trained, ref_state, ChannelSelection(sel))
    allowed, base_sd = manual_allowed(sel, base), snapshot(base)
    for n, p in trained.named_parameters():
        assert p.dtype == torch.bfloat16
        assert torch.equal(p.detach(), torch.where(allowed[n], base_sd[n], trained_sd[n])), n

    f32_state = channel_state(base.float())
    with pytest.raises(ChannelSelectionError, match="dtype"):
        rollback_channels(trained, f32_state, ChannelSelection(sel))
    rollback_channels(trained, f32_state, ChannelSelection(sel), allow_cast=True)
    assert all(p.dtype == torch.bfloat16 for p in trained.parameters())


def test_failed_rollback_leaves_model_untouched(tiny_model):
    """Every validation runs for all layers before the first write (layer 0 valid, layer 1 bad)."""
    full = channel_state(tiny_model)
    with torch.no_grad():
        for p in tiny_model.parameters():
            p.add_(1.0)  # the valid layer-0 part of each rollback below WOULD change the model
    sel = ChannelSelection({0: [1, 2], 1: [5]})
    l1 = full[1]
    bad_refs = {
        "missing layer": ChannelState({0: full[0]}),
        "missing channel": ChannelState({0: full[0], 1: l1.select([0, 1])}),
        "dtype": ChannelState({0: full[0], 1: dataclasses.replace(l1, gate=l1.gate.double())}),
        "bias presence": ChannelState({0: full[0], 1: dataclasses.replace(l1, up_bias=torch.zeros(INTER))}),
        "shape": ChannelState({0: full[0], 1: dataclasses.replace(l1, down=l1.down[:-1])}),
        "module path": ChannelState({0: full[0], 1: dataclasses.replace(l1, module_path="other.mlp")}),
    }
    before = snapshot(tiny_model)
    for what, ref in bad_refs.items():
        with pytest.raises(ChannelSelectionError):
            rollback_channels(tiny_model, ref, sel)
        after = snapshot(tiny_model)
        assert all(torch.equal(before[n], after[n]) for n in before), what
    # positive control: the same call with a valid reference does write layer 0
    rollback_channels(tiny_model, full, sel)
    assert not torch.equal(before[wname(0, "gate_proj")], tiny_model.model.layers[0].mlp.gate_proj.weight)


def test_rollback_report_counts_only_elements_that_really_change(tokenizer):
    base = make_model(tokenizer, torch.float64)
    ref = channel_state(base)
    model = copy.deepcopy(base)
    sel = ChannelSelection({0: [3, 9], 1: [7]})
    with torch.no_grad():  # only some selected elements differ from the reference (+ unselected ones)
        model.model.layers[0].mlp.gate_proj.weight[3, :5] += 1.0
        model.model.layers[0].mlp.down_proj.weight[2, 9] -= 3.0
        model.model.layers[1].mlp.up_proj.weight[7, 0] += 0.5
        model.model.layers[1].mlp.up_proj.weight[8] += 1.0  # unselected channel
    pre = snapshot(model)
    report = rollback_channels(model, ref, sel)
    post = snapshot(model)
    actual = sum(int((pre[n] != post[n]).sum()) for n in pre)
    assert actual == 7 and report.total_elements_changed == actual
    assert report.elements_changed == {0: 6, 1: 1}
    torch.testing.assert_close(
        report.delta_l2[0], torch.tensor([math.sqrt(5.0), 3.0], dtype=torch.float64), rtol=1e-12, atol=0
    )
    assert report.elements_written == 3 * 3 * HIDDEN
    assert rollback_channels(model, ref, sel).total_elements_changed == 0  # idempotent

    # lossy cast: a second identical rollback changes nothing and must report 0
    bf = make_model(tokenizer, torch.bfloat16)
    f32_ref = channel_state(make_model(tokenizer, torch.float32, seed=3))
    one = ChannelSelection({0: [0]})
    first = rollback_channels(bf, f32_ref, one, allow_cast=True)
    assert first.total_elements_changed > 0
    mid = snapshot(bf)
    second = rollback_channels(bf, f32_ref, one, allow_cast=True)
    assert all(torch.equal(mid[n], t) for n, t in snapshot(bf).items())
    assert second.total_elements_changed == 0 and not second.delta_l2[0].any()

    # NaN -> NaN is not a change (same convention as verify_modification_scope)
    with torch.no_grad():
        model.model.layers[0].mlp.gate_proj.weight[3, 0] = float("nan")
    nan_ref = channel_state(model)
    nan_report = rollback_channels(model, nan_ref, ChannelSelection({0: [3]}))
    assert nan_report.total_elements_changed == 0 and not nan_report.delta_l2[0].isnan().any()


# ---------------------------------------------------------------------------------- (c) channel-only training
SEL = {0: [3, 10, 40], 1: [0, 63]}


def lm_loss(model, tokenizer, step: int) -> torch.Tensor:
    ids = random_ids(tokenizer, step)
    return model(input_ids=ids, labels=ids).loss


def warmed_up(tokenizer, wd: float):
    """Model + plain AdamW after 3 unmasked steps (non-zero Adam moments everywhere)."""
    model = make_model(tokenizer)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=wd)
    for step in range(3):
        opt.zero_grad()
        lm_loss(model, tokenizer, step).backward()
        opt.step()
    return model, opt


@pytest.mark.parametrize("sel", [SEL, {0: [3, 10, 40]}], ids=["both_layers", "layer1_untouched"])
@pytest.mark.parametrize("disable_grad_outside", [False, True])
def test_channel_only_training_after_warmup_matches_reference(tokenizer, disable_grad_outside, sel):
    model, inner = warmed_up(tokenizer, wd=0.1)
    ref_model = copy.deepcopy(model)
    ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=1e-2, weight_decay=0.1)
    ref_opt.load_state_dict(copy.deepcopy(inner.state_dict()))  # load_state_dict would alias tensors
    free_model = copy.deepcopy(model)  # unmasked control run
    free_opt = torch.optim.AdamW(free_model.parameters(), lr=1e-2, weight_decay=0.1)
    free_opt.load_state_dict(copy.deepcopy(inner.state_dict()))
    params = dict(model.named_parameters())
    assert inner.state[params[wname(0, "gate_proj")]]["exp_avg"][11].abs().sum() > 0  # warm moments

    opt = HookedOptimizer(inner, model.named_parameters())
    handle = setup_channel_training(model, ChannelSelection(sel), disable_grad_outside=disable_grad_outside)
    for rule in handle.rules:
        opt.add_rule(rule)
    before = snapshot(model)
    allowed = manual_allowed(sel, model)
    for step in range(3, 7):
        opt.zero_grad(set_to_none=False)  # zero (not None) grads: stale moments would act on them
        lm_loss(model, tokenizer, step).backward()
        opt.step()
        # Independent reference: unmasked AdamW, then hand-written restore of every frozen element.
        ref_opt.zero_grad()
        lm_loss(ref_model, tokenizer, step).backward()
        ref_opt.step()
        with torch.no_grad():
            for n, p in ref_model.named_parameters():
                p.copy_(torch.where(allowed[n], p, before[n]))
        free_opt.zero_grad()
        lm_loss(free_model, tokenizer, step).backward()
        free_opt.step()
    unmasked = snapshot(free_model)
    after = snapshot(model)
    ref_after = snapshot(ref_model)
    for n in after:
        assert torch.equal(after[n], ref_after[n]), n
    verify_modification_scope(before, after, ChannelSelection(sel), model, strict=True)
    for layer in range(LAYERS):
        if layer not in sel:  # an FFN layer with no selected channel is entirely frozen
            assert all(torch.equal(before[wname(layer, p)], after[wname(layer, p)]) for p in PROJ)
            # ...although unmasked AdamW (warm moments + weight decay) would have moved it
            assert not torch.equal(before[wname(layer, "gate_proj")], unmasked[wname(layer, "gate_proj")])
    for layer, chans in sel.items():
        for c in chans:
            assert not torch.equal(before[wname(layer, "gate_proj")][c], after[wname(layer, "gate_proj")][c])
            assert not torch.equal(before[wname(layer, "up_proj")][c], after[wname(layer, "up_proj")][c])
            assert not torch.equal(
                before[wname(layer, "down_proj")][:, c], after[wname(layer, "down_proj")][:, c]
            )
    if disable_grad_outside:
        assert not params["model.embed_tokens.weight"].requires_grad
    handle.remove()
    assert all(p.requires_grad for p in model.parameters())
    assert handle.gradient_hooks is None


def test_disable_grad_outside_also_drops_stale_grads_of_already_frozen_params(tokenizer):
    model, inner = warmed_up(tokenizer, wd=0.1)
    params = dict(model.named_parameters())
    embed = params["model.embed_tokens.weight"]
    embed.requires_grad_(False)  # frozen by the user earlier: still holds a stale grad + warm moments
    assert embed.grad is not None and embed.grad.abs().sum() > 0
    assert inner.state[embed]["exp_avg"].abs().sum() > 0
    opt = HookedOptimizer(inner, model.named_parameters())
    sel = ChannelSelection({0: [3], 1: [1]})
    handle = setup_channel_training(model, sel, disable_grad_outside=True)
    for rule in handle.rules:
        opt.add_rule(rule)
    assert embed.grad is None and "model.embed_tokens.weight" not in handle.disabled_grad
    before = snapshot(model)
    for step in range(3, 5):
        opt.zero_grad(set_to_none=False)
        lm_loss(model, tokenizer, step).backward()
        opt.step()
    report = verify_modification_scope(before, model, sel, model, strict=True)
    assert report.changed_allowed > 0 and torch.equal(before["model.embed_tokens.weight"], embed.detach())
    handle.remove()
    assert not embed.requires_grad  # stays frozen: it was not frozen by the handle
    assert all(p.requires_grad for n, p in params.items() if n != "model.embed_tokens.weight")


@pytest.mark.parametrize("wd", [0.0, 0.1])
def test_gradient_only_mask_would_move_frozen_channels(tokenizer, wd):
    """Sensitivity: masking only gradients lets warm Adam moments / weight decay move frozen weights."""
    model, opt = warmed_up(tokenizer, wd=wd)
    allowed = manual_allowed(SEL, model)
    before = snapshot(model)
    opt.zero_grad()
    lm_loss(model, tokenizer, 3).backward()
    for n, p in model.named_parameters():
        p.grad.mul_(allowed[n])
    opt.step()
    after = snapshot(model)
    moved = {n for n in before if not torch.equal(before[n][~allowed[n]], after[n][~allowed[n]])}
    assert {wname(0, "gate_proj"), wname(1, "down_proj"), "model.embed_tokens.weight"} <= moved
    report = verify_modification_scope(before, after, ChannelSelection(SEL), model)
    assert set(report.violations) == moved
    for n in moved:
        assert report.violations[n].count == int(((before[n] != after[n]) & ~allowed[n]).sum())


def test_backward_masks_zero_frozen_grads_and_remove_detaches(tokenizer):
    model = make_model(tokenizer)
    ref = copy.deepcopy(model)
    handle = setup_channel_training(model, ChannelSelection(SEL))
    lm_loss(model, tokenizer, 0).backward()
    lm_loss(ref, tokenizer, 0).backward()
    allowed = manual_allowed(SEL, model)
    ref_params = dict(ref.named_parameters())
    for n, p in model.named_parameters():
        assert torch.equal(p.grad, ref_params[n].grad * allowed[n]), n
    handle.remove()
    model.zero_grad()
    ref.zero_grad()
    lm_loss(model, tokenizer, 0).backward()
    lm_loss(ref, tokenizer, 0).backward()
    assert all(torch.equal(p.grad, ref_params[n].grad) for n, p in model.named_parameters())


# ---------------------------------------------------------------------------------- (d) real TRL SFTTrainer
class GradCapture(TrainerCallback):
    """Records unmasked micro-batch grads (hooks registered *before* the masks) per optimizer step."""

    def __init__(self, model, allowed: dict[str, torch.Tensor]) -> None:
        self.allowed = allowed
        self.acc: dict[str, torch.Tensor] = {}
        self.masked_norms: list[float] = []
        self.full_norms: list[float] = []
        self.handles = [
            p.register_hook(lambda g, n=n: self._capture(n, g)) for n, p in model.named_parameters()
        ]

    def _capture(self, name: str, grad: torch.Tensor) -> None:
        g = grad.detach().double()
        self.acc[name] = self.acc[name] + g if name in self.acc else g.clone()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        masked = torch.cat([(g * self.allowed[n]).flatten() for n, g in self.acc.items()])
        full = torch.cat([g.flatten() for g in self.acc.values()])
        self.masked_norms.append(float(torch.linalg.vector_norm(masked)))
        self.full_norms.append(float(torch.linalg.vector_norm(full)))
        self.acc = {}


@pytest.mark.integration
@pytest.mark.parametrize("disable_grad_outside", [False, True])
def test_sft_trainer_channel_only_scope_and_grad_norm(tokenizer, tmp_path, disable_grad_outside):
    model = make_model(tokenizer)
    sel = ChannelSelection(SEL)
    allowed = manual_allowed(SEL, model)
    capture = GradCapture(model, allowed)  # registered first => sees grads before the masks
    handle = setup_channel_training(model, sel, disable_grad_outside=disable_grad_outside)
    rows = [{"prompt": f"{a}+{b}=", "completion": str(a + b)} for a in range(4) for b in range(4)]
    args = SFTConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        max_steps=3,
        learning_rate=5e-3,
        weight_decay=0.1,
        max_grad_norm=0.02,
        logging_steps=1,
        report_to=[],
        use_cpu=True,
        save_strategy="no",
        seed=7,
    )
    trainer = research_trainer_class(SFTTrainer)(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        update_rules=handle.rules,
        callbacks=[capture],
    )
    before, before_channels = snapshot(model), channel_state(model)
    trainer.train()
    after = snapshot(model)

    report = verify_modification_scope(before, after, sel, model, strict=True)
    assert report.changed_allowed > 0
    for n in before:  # plain check, independent of the verifier
        assert torch.equal(before[n][~allowed[n]], after[n][~allowed[n]]), n
    optimised = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    params = dict(model.named_parameters())
    if disable_grad_outside:
        assert handle.disabled_grad and not any(id(params[n]) in optimised for n in handle.disabled_grad)
        assert len(handle.mask) == 2 * 3  # only the affected FFN weights need masks
    else:
        assert all(id(p) in optimised for p in params.values()) and len(handle.mask) == len(params)
    diff = channel_diff(before_channels, model)
    assert top_changed_channels(diff, 100, min_score=0.0) == sel  # exactly the selected channels moved
    logged = [h["grad_norm"] for h in trainer.state.log_history if "grad_norm" in h]
    assert len(logged) == 3 == len(capture.masked_norms)
    for got, want, full in zip(logged, capture.masked_norms, capture.full_norms):
        assert got == pytest.approx(want, rel=1e-5)
        assert full > 1.5 * want  # unmasked grads would have produced a clearly different norm
        assert want > args.max_grad_norm  # clipping was active, so the norm actually mattered
    handle.remove()
    for h in capture.handles:
        h.remove()


# ---------------------------------------------------------------------------------- remaining branches
def test_diff_metrics_and_ranking_match_manual(tiny_model):
    base = tiny_model.double()
    trained = copy.deepcopy(base)
    g = torch.Generator().manual_seed(21)
    scale = {
        0: torch.rand(INTER, generator=g, dtype=torch.float64),
        1: torch.rand(INTER, generator=g, dtype=torch.float64),
    }
    with torch.no_grad():
        for layer in range(LAYERS):
            mlp = trained.model.layers[layer].mlp
            mlp.gate_proj.weight.add_(scale[layer][:, None] * 0.1)
            mlp.down_proj.weight[:, ::2].mul_(1.5)  # even channels also change in down
    diff = channel_diff(channel_state(base), trained)
    scores: dict[str, dict[int, torch.Tensor]] = {"l2": {}, "max_abs": {}, "relative": {}}
    for layer in range(LAYERS):
        wb, wt = raw(base, layer), raw(trained, layer)
        vb = torch.cat([wb["gate_proj"], wb["up_proj"], wb["down_proj"].T], dim=1)
        d = torch.cat([wt["gate_proj"], wt["up_proj"], wt["down_proj"].T], dim=1) - vb
        scores["l2"][layer] = torch.linalg.vector_norm(d, dim=1)
        scores["max_abs"][layer] = d.abs().amax(dim=1)
        scores["relative"][layer] = scores["l2"][layer] / torch.linalg.vector_norm(vb, dim=1)
        for metric in scores:
            torch.testing.assert_close(diff[layer].score(metric), scores[metric][layer], rtol=1e-12, atol=0)
        parts = sum(v**2 for v in diff[layer].per_projection_l2.values())
        torch.testing.assert_close(parts, diff[layer].l2 ** 2, rtol=1e-12, atol=0)
    for metric in scores:
        manual = sorted(
            ((float(v), layer, c) for layer, s in scores[metric].items() for c, v in enumerate(s)),
            key=lambda e: (-e[0], e[1], e[2]),
        )
        ranked = ranked_channels(diff, metric)
        assert [(layer, c) for layer, c, _ in ranked] == [(layer, c) for _, layer, c in manual]
        top = top_changed_channels(diff, 3, metric=metric, per_layer=True)
        assert top == ChannelSelection.top_k_from_scores(scores[metric], 3, per_layer=True)
        assert top.to_dict() == {
            k: sorted(torch.topk(v, 3).indices.tolist()) for k, v in scores[metric].items()
        }
    only1 = channel_diff(channel_state(base), trained, layers=[1])
    assert list(only1) == [1]
    torch.testing.assert_close(only1[1].l2, scores["l2"][1], rtol=1e-12, atol=0)


def test_nan_scores_rank_first_and_are_rejected_by_top_k():
    l2 = torch.tensor([float("nan"), 1.0, 2.0, float("nan")], dtype=torch.float64)
    zeros = torch.zeros(4, dtype=torch.float64)
    d = ChannelDiff(
        layer=0, channels=torch.arange(4), l2=l2, max_abs=zeros, ref_l2=zeros + 1, per_projection_l2={}
    )
    ranked = ranked_channels({0: d})
    assert [(layer, c) for layer, c, _ in ranked] == [(0, 0), (0, 3), (0, 2), (0, 1)]
    assert math.isnan(ranked[0][2]) and ranked[2][2] == 2.0
    for kwargs in ({}, {"min_score": 0.0}, {"min_score": 5.0}):
        with pytest.raises(ChannelSelectionError, match="NaN"):
            top_changed_channels({0: d}, 2, **kwargs)


def test_verify_scope_with_parameter_mask_and_unlisted_policy(tiny_model):
    before = snapshot(tiny_model)
    after = {n: t.clone() for n, t in before.items()}
    q = "model.layers.0.self_attn.q_proj.weight"
    after[q][0, :3] += 1.0  # allowed below
    after[q][5, 1] -= 2.0  # not allowed (negative delta: max_abs must use |Δ|)
    after["lm_head.weight"][2, 2] += 1.0  # unlisted parameter
    after["model.norm.weight"][0] = float("nan")
    before["model.norm.weight"][0] = after["model.norm.weight"][0] = float("nan")  # NaN == NaN: unchanged
    allowed = torch.zeros_like(before[q], dtype=torch.bool)
    allowed[0] = True
    mask = ParameterMask({q: allowed})
    frozen = verify_modification_scope(before, after, mask, tiny_model)
    assert {n: v.indices.tolist() for n, v in frozen.violations.items()} == {
        q: [[5, 1]],
        "lm_head.weight": [[2, 2]],
    }
    assert frozen.changed_allowed == 3 and frozen.violations[q].max_abs == pytest.approx(2.0)
    free = verify_modification_scope(before, after, mask, tiny_model, unlisted="free")
    assert list(free.violations) == [q] and free.params_checked == 1
    with pytest.raises(ScopeViolationError, match="lm_head"):
        verify_modification_scope(before, after, mask, tiny_model, strict=True)
    # malformed inputs are rejected, not silently skipped
    with pytest.raises(KeyError, match="not in the snapshots"):
        verify_modification_scope(before, after, ParameterMask({"no.such.weight": allowed}), tiny_model)
    with pytest.raises(KeyError, match="disagree"):
        verify_modification_scope(before, {n: t for n, t in after.items() if n != q}, mask, tiny_model)
    for changed in (after[q].double(), after[q][:-1]):
        with pytest.raises(ValueError, match="shape/dtype"):
            verify_modification_scope(before, {**after, q: changed}, mask, tiny_model)


def test_setup_without_freeze_rest_or_gradient_masks(tokenizer):
    model, inner = warmed_up(tokenizer, wd=0.1)
    handle = setup_channel_training(
        model, ChannelSelection({0: [3]}), freeze_rest=False, gradient_masks=False
    )
    assert handle.gradient_hooks is None
    assert sorted(handle.mask.names()) == sorted(wname(0, p) for p in PROJ)
    ref = copy.deepcopy(model)
    opt = HookedOptimizer(inner, model.named_parameters(), handle.rules)
    before = snapshot(model)
    opt.zero_grad()
    lm_loss(model, tokenizer, 3).backward()
    lm_loss(ref, tokenizer, 3).backward()
    ref_params = dict(ref.named_parameters())
    assert all(torch.equal(p.grad, ref_params[n].grad) for n, p in model.named_parameters())  # no hooks
    opt.step()
    after = snapshot(model)
    allowed = manual_allowed({0: [3]}, model)
    for p in PROJ:  # layer-0 FFN: only channel 3 moved (MaskRule still masks grads + restores)
        n = wname(0, p)
        assert torch.equal(before[n][~allowed[n]], after[n][~allowed[n]])
        assert not torch.equal(before[n][allowed[n]], after[n][allowed[n]])
    # parameters outside the affected FFN are unconstrained with freeze_rest=False
    assert not torch.equal(before["model.embed_tokens.weight"], after["model.embed_tokens.weight"])
    assert not torch.equal(before[wname(1, "gate_proj")], after[wname(1, "gate_proj")])


# ---------------------------------------------------------------------------------- expert containers
def test_channel_ops_target_one_expert_and_refuse_cross_expert_references(tiny_model):
    model = tiny_model
    mgr = ExpertManager(model)
    mgr.attach()
    mgr.create_expert("a")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".experts.a." in n:
                p.add_(1.0)
    ename = "model.layers.{}.mlp.experts.{}.{}.weight".format
    sel = ChannelSelection({0: [0, 4]})
    assert set(channel_mask(model, sel, expert="a").names()) == {ename(0, "a", p) for p in PROJ}
    ref_base = channel_state(model)  # active expert: base
    assert ref_base[0].module_path == "model.layers.0.mlp.experts.base"
    mgr.activate("a")
    before = snapshot(model)
    with pytest.raises(ChannelSelectionError, match="allow_path_mismatch"):
        rollback_channels(model, ref_base, sel)
    with pytest.raises(ChannelSelectionError, match="allow_path_mismatch"):
        channel_diff(ref_base, model)
    assert all(torch.equal(before[n], t) for n, t in snapshot(model).items())

    # explicit expert= addresses base although "a" is active
    st_base = channel_state(model, expert="base")
    assert st_base[1].module_path == "model.layers.1.mlp.experts.base"
    assert torch.equal(st_base[0].gate, ref_base[0].gate) and torch.equal(st_base[1].down, ref_base[1].down)
    assert channel_diff(ref_base, model, expert="base")[0].changed_channels() == []
    assert set(freeze_mask(model, sel, expert="base").names()) == {ename(0, "base", p) for p in PROJ}
    assert sel.complement(model, expert="a").to_dict() == {0: [c for c in range(INTER) if c not in (0, 4)]}
    previous = ablate_channels(model, sel, expert="base")
    ablated = snapshot(model)
    zeroed = {n: torch.zeros(t.shape, dtype=torch.bool) for n, t in before.items()}
    zeroed[ename(0, "base", "gate_proj")][[0, 4]] = True
    zeroed[ename(0, "base", "up_proj")][[0, 4]] = True
    zeroed[ename(0, "base", "down_proj")][:, [0, 4]] = True
    for n in before:
        assert torch.equal(ablated[n], torch.where(zeroed[n], 0.0, before[n])), n
    rollback_channels(model, previous, expert="base")
    assert all(torch.equal(before[n], t) for n, t in snapshot(model).items())
    # nothing changes (same values)
    assert rollback_channels(model, ref_base, sel, expert="base").total_elements_changed == 0
    diff_a = channel_diff(ref_base, model, allow_path_mismatch=True)
    torch.testing.assert_close(diff_a[0].max_abs, torch.ones(INTER, dtype=torch.float64), rtol=0, atol=1e-6)
    # deliberate cross-expert reset: only expert a's selected slices change, to base's values
    rollback_channels(model, ref_base, sel, allow_path_mismatch=True)
    after = snapshot(model)
    allowed = {n: torch.zeros(t.shape, dtype=torch.bool) for n, t in before.items()}
    allowed[ename(0, "a", "gate_proj")][[0, 4]] = True
    allowed[ename(0, "a", "up_proj")][[0, 4]] = True
    allowed[ename(0, "a", "down_proj")][:, [0, 4]] = True
    for n in before:
        base_n = n.replace(".experts.a.", ".experts.base.")
        assert torch.equal(after[n], torch.where(allowed[n], before[base_n], before[n])), n
    assert verify_modification_scope(before, after, sel, model, expert="a").ok
    assert not verify_modification_scope(before, after, sel, model, expert="base").ok
    handle = setup_channel_training(model, sel, expert="base", disable_grad_outside=True)
    assert set(handle.mask.names()) == {ename(0, "base", p) for p in PROJ}
    assert ename(0, "a", "gate_proj") in handle.disabled_grad
    handle.remove()
