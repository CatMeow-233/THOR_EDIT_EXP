"""SwiGLU FFN channel operations, checked against independent plain-torch references."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from research_rl.models.ffn_channels import (
    ChannelSelection,
    ChannelSelectionError,
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
    assert s.complement(tiny_model).count() == 2 * INTER - 4
    s.validate(tiny_model)
    for bad in ({0: [INTER]}, {LAYERS: [0]}):
        with pytest.raises(ChannelSelectionError):
            ChannelSelection(bad).validate(tiny_model)
    for bad in ({0: [-1]}, {0: [1.0]}, {0: [True]}):
        with pytest.raises(ChannelSelectionError):
            ChannelSelection(bad)
    with pytest.raises(ChannelSelectionError):
        ChannelSelection({0: [64]}, intermediate_size=64)


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


def test_bias_entries_are_owned_by_channels():
    lin_model = build_tiny_llama(40, seed=1)
    mlp = lin_model.model.layers[0].mlp
    for proj in PROJ:
        old = getattr(mlp, proj)
        new = torch.nn.Linear(old.in_features, old.out_features, bias=True)
        setattr(mlp, proj, new)
    sel = ChannelSelection({0: [2, 5]})
    mask = channel_mask(lin_model, sel)
    gb = mask.get("model.layers.0.mlp.gate_proj.bias")
    assert gb.tolist() == [i in (2, 5) for i in range(INTER)]
    assert not mask.get("model.layers.0.mlp.down_proj.bias").any()  # hidden-space bias: no owner
    assert freeze_mask(lin_model, sel).get("model.layers.0.mlp.down_proj.bias").all()
    # rollback restores bias entries too
    base = channel_state(lin_model, [0])
    with torch.no_grad():
        mlp.up_proj.bias.add_(1.0)
    diff = channel_diff(base, lin_model)[0]
    torch.testing.assert_close(diff.per_projection_l2["bias"], torch.ones(INTER, dtype=torch.float64))
    assert all(not diff.per_projection_l2[p].any() for p in PROJ)
    torch.testing.assert_close(diff.l2, diff.max_abs, rtol=1e-15, atol=0)
    before = mlp.up_proj.bias.detach().clone()
    rollback_channels(lin_model, base, sel)
    expected = before.clone()
    expected[[2, 5]] = base[0].up_bias[[2, 5]]
    assert torch.equal(mlp.up_proj.bias, expected)


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


def test_rollback_requires_reference_channels(tiny_model):
    partial = extract_channels(tiny_model, ChannelSelection({0: [1, 2]}))
    with pytest.raises(ChannelSelectionError):
        rollback_channels(tiny_model, partial, ChannelSelection({0: [3]}))
    with pytest.raises(ChannelSelectionError):
        rollback_channels(tiny_model, partial, ChannelSelection({1: [1]}))


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


@pytest.mark.parametrize("disable_grad_outside", [False, True])
def test_channel_only_training_after_warmup_matches_reference(tokenizer, disable_grad_outside):
    model, inner = warmed_up(tokenizer, wd=0.1)
    ref_model = copy.deepcopy(model)
    ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=1e-2, weight_decay=0.1)
    ref_opt.load_state_dict(copy.deepcopy(inner.state_dict()))  # load_state_dict would alias tensors
    params = dict(model.named_parameters())
    assert inner.state[params[wname(0, "gate_proj")]]["exp_avg"][11].abs().sum() > 0  # warm moments

    opt = HookedOptimizer(inner, model.named_parameters())
    handle = setup_channel_training(model, ChannelSelection(SEL), disable_grad_outside=disable_grad_outside)
    for rule in handle.rules:
        opt.add_rule(rule)
    before = snapshot(model)
    allowed = manual_allowed(SEL, model)
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
    after = snapshot(model)
    ref_after = snapshot(ref_model)
    for n in after:
        assert torch.equal(after[n], ref_after[n]), n
    verify_modification_scope(before, after, ChannelSelection(SEL), model, strict=True)
    for layer, chans in SEL.items():
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


def test_verify_scope_with_parameter_mask_and_unlisted_policy(tiny_model):
    before = snapshot(tiny_model)
    after = {n: t.clone() for n, t in before.items()}
    q = "model.layers.0.self_attn.q_proj.weight"
    after[q][0, :3] += 1.0  # allowed below
    after[q][5, 1] += 1.0  # not allowed
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
    assert frozen.changed_allowed == 3 and frozen.violations[q].max_abs == pytest.approx(1.0)
    free = verify_modification_scope(before, after, mask, tiny_model, unlisted="free")
    assert list(free.violations) == [q] and free.params_checked == 1
    with pytest.raises(ScopeViolationError, match="lm_head"):
        verify_modification_scope(before, after, mask, tiny_model, strict=True)


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
