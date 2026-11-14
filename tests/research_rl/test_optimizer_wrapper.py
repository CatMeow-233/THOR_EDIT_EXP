"""Numerical tests for HookedOptimizer against independent reference computations."""

from __future__ import annotations

import io

import pytest
import torch
from torch import nn

from research_rl.models.parameter_mask import ParameterMask
from research_rl.optim import (
    DisplacementLedger,
    HookedOptimizer,
    MaskRule,
    UpdateRule,
    register_gradient_masks,
)
from research_rl.optim.gradient_hooks import global_grad_norm


def make_model(seed: int = 0, dtype: torch.dtype = torch.float32) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 3)).to(dtype)


def batch(seed: int, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1000 + seed)
    return torch.randn(5, 6, generator=g).to(dtype), torch.randn(5, 3, generator=g).to(dtype)


def backward(model: nn.Module, seed: int, scale: float = 1.0) -> None:
    x, y = batch(seed, next(model.parameters()).dtype)
    (scale * ((model(x) - y) ** 2).mean()).backward()


def adamw(model: nn.Module, wd: float = 0.1) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=wd)


def params_equal(a: nn.Module, b: nn.Module) -> bool:
    return all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def first_layer_mask(model: nn.Module) -> ParameterMask:
    w = dict(model.named_parameters())["0.weight"]
    m = torch.zeros_like(w, dtype=torch.bool)
    m[:3] = True  # rows 0..2 trainable, rows 3..7 frozen
    b = torch.zeros(8, dtype=torch.bool)
    return ParameterMask({"0.weight": m, "0.bias": b})


# ---------------------------------------------------------------------------------- passthrough
@pytest.mark.parametrize("rules", [(), ("ledger",)])
def test_passthrough_is_bit_identical(rules):
    ref, hooked = make_model(), make_model()
    opt_ref = adamw(ref)
    ledger = DisplacementLedger()
    opt = HookedOptimizer(adamw(hooked), hooked.named_parameters(), [ledger] if rules else [])
    for step in range(5):
        for m, o in ((ref, opt_ref), (hooked, opt)):
            o.zero_grad()
            backward(m, step)
            o.step()
    assert params_equal(ref, hooked)
    for (_, s1), (_, s2) in zip(opt_ref.state.items(), opt.state.items()):
        assert torch.equal(s1["exp_avg"], s2["exp_avg"]) and torch.equal(s1["exp_avg_sq"], s2["exp_avg_sq"])
    assert opt.applied_updates == 5 and opt.nominal_steps == 5


def test_ledger_reports_actual_displacement():
    model = make_model()
    ledger = DisplacementLedger()
    opt = HookedOptimizer(adamw(model), model.named_parameters(), [ledger])
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    backward(model, 0)
    opt.step()
    for n, p in model.named_parameters():
        assert torch.equal(ledger.last[n], p.detach().float() - before[n].float())


# ---------------------------------------------------------------------------------- masks + momentum
def test_mask_after_warmup_keeps_frozen_bits_with_momentum_and_weight_decay():
    """Warm Adam moments first, then mask: frozen elements must not move by a single bit."""
    model = make_model()
    inner = adamw(model, wd=0.1)
    opt = HookedOptimizer(inner, model.named_parameters())
    for step in range(3):  # unmasked warm-up builds non-zero exp_avg everywhere
        opt.zero_grad()
        backward(model, step)
        opt.step()
    mask = first_layer_mask(model)
    opt.add_rule(MaskRule(mask))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for step in range(3, 8):
        opt.zero_grad()
        backward(model, step)
        opt.step()
    after = {n: p.detach().clone() for n, p in model.named_parameters()}
    assert mask.changed_outside(before, after) == {"0.weight": 0, "0.bias": 0}
    # selected rows and unmasked layers did move
    assert not torch.equal(before["0.weight"][:3], after["0.weight"][:3])
    assert not torch.equal(before["2.weight"], after["2.weight"])


def test_gradient_only_masking_is_insufficient_reference():
    """Sanity check that the previous test is sensitive: grad-only masking DOES move frozen rows."""
    model = make_model()
    opt = adamw(model, wd=0.0)
    for step in range(3):
        opt.zero_grad()
        backward(model, step)
        opt.step()
    mask = first_layer_mask(model)
    before = model[0].weight.detach().clone()
    opt.zero_grad()
    backward(model, 3)
    for n, p in model.named_parameters():
        if n in mask:
            p.grad.mul_(mask.get(n))
    opt.step()
    assert not torch.equal(before[3:], model[0].weight.detach()[3:])


def test_mask_bf16_exact():
    model = make_model(dtype=torch.bfloat16)
    opt = HookedOptimizer(adamw(model), model.named_parameters())
    opt.zero_grad()
    backward(model, 0)
    opt.step()
    mask = first_layer_mask(model)
    opt.add_rule(MaskRule(mask))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for step in range(1, 4):
        opt.zero_grad()
        backward(model, step)
        opt.step()
    after = {n: p.detach().clone() for n, p in model.named_parameters()}
    assert mask.changed_outside(before, after) == {"0.weight": 0, "0.bias": 0}


# ---------------------------------------------------------------------------------- custom update rules
class ScaleUpdate(UpdateRule):
    name = "scale"

    def __init__(self, factor: float) -> None:
        self.factor = factor

    def tracks(self, name, param):
        return True

    def transform_update(self, ctx, name, param, old, delta):
        return delta * self.factor


class ClampUpdate(UpdateRule):
    """Element-wise trust region |Δ| <= tau (DeltaHippo-style constraint on the Adam update)."""

    name = "clamp"

    def __init__(self, tau: float) -> None:
        self.tau = tau

    def tracks(self, name, param):
        return True

    def transform_update(self, ctx, name, param, old, delta):
        return delta.clamp(-self.tau, self.tau)


@pytest.mark.parametrize("rule_factory", [lambda: ScaleUpdate(0.5), lambda: ClampUpdate(3e-3)])
def test_transform_update_matches_reference(rule_factory):
    """Reference: run plain AdamW on a copy to get the candidate, apply the transform by hand."""
    hooked = make_model()
    rule = rule_factory()
    opt = HookedOptimizer(adamw(hooked), hooked.named_parameters(), [rule])
    shadow = make_model()  # independent reference trajectory
    shadow_opt = adamw(shadow)
    for step in range(4):
        # reference: candidate = AdamW step from the reference parameters, then transform
        old = {n: p.detach().clone() for n, p in shadow.named_parameters()}
        shadow_opt.zero_grad()
        backward(shadow, step)
        shadow_opt.step()
        with torch.no_grad():
            for n, p in shadow.named_parameters():
                delta = p.float() - old[n].float()
                delta = (
                    delta * rule.factor if isinstance(rule, ScaleUpdate) else delta.clamp(-rule.tau, rule.tau)
                )
                p.copy_(old[n].float() + delta)
        opt.zero_grad()
        backward(hooked, step)
        opt.step()
    for p, q in zip(hooked.parameters(), shadow.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=1e-7)


def test_rules_compose_in_order_and_mask_wins():
    model = make_model()
    mask = first_layer_mask(model)
    opt = HookedOptimizer(adamw(model), model.named_parameters(), [ScaleUpdate(2.0), MaskRule(mask)])
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    backward(model, 0)
    opt.step()
    after = {n: p.detach().clone() for n, p in model.named_parameters()}
    assert mask.changed_outside(before, after) == {"0.weight": 0, "0.bias": 0}


# ---------------------------------------------------------------------------------- accumulation / clipping
class CountingRule(UpdateRule):
    name = "count"

    def __init__(self):
        self.post_grad_calls = 0
        self.grads: list[torch.Tensor] = []

    def post_grad(self, ctx, named_params):
        self.post_grad_calls += 1
        self.grads.append(dict(named_params)["2.weight"].grad.detach().clone())


def test_gradient_accumulation_hooks_fire_once_on_accumulated_grad():
    model = make_model()
    rule = CountingRule()
    opt = HookedOptimizer(adamw(model), model.named_parameters(), [rule])
    backward(model, 0, scale=0.5)
    backward(model, 1, scale=0.5)
    opt.step()
    assert rule.post_grad_calls == 1
    ref = make_model()
    backward(ref, 0, scale=0.5)
    backward(ref, 1, scale=0.5)
    assert torch.equal(rule.grads[0], ref[2].weight.grad)


def test_clip_in_wrapper_uses_post_mask_norm():
    model = make_model()
    mask = first_layer_mask(model)
    max_norm = 0.05
    opt = HookedOptimizer(
        adamw(model, wd=0.0), model.named_parameters(), [MaskRule(mask)], clip_norm=max_norm
    )
    ref = make_model()
    ref_opt = adamw(ref, wd=0.0)

    backward(model, 0)
    opt.step()

    backward(ref, 0)
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if n in mask:
                p.grad.mul_(mask.get(n))
        norm = global_grad_norm(ref.parameters())
        coef = min(1.0, max_norm / (norm + 1e-6))
        for p in ref.parameters():
            p.grad.mul_(coef)
    ref_opt.step()
    assert opt.last_grad_norm == pytest.approx(norm, rel=1e-6)
    for p, q in zip(model.parameters(), ref.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=1e-7)


def test_backward_time_gradient_masks_exclude_frozen_from_norm():
    model = make_model()
    mask = first_layer_mask(model)
    handle = register_gradient_masks(model, mask)
    backward(model, 0)
    ref = make_model()
    backward(ref, 0)
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if n in mask:
                p.grad.mul_(mask.get(n))
    assert global_grad_norm(model.parameters()) == pytest.approx(global_grad_norm(ref.parameters()), rel=1e-6)
    assert torch.count_nonzero(model[0].weight.grad[3:]) == 0
    handle.remove()
    model.zero_grad()
    backward(model, 0)
    assert torch.count_nonzero(model[0].weight.grad[3:]) > 0


# ---------------------------------------------------------------------------------- skip / state / scheduler
def test_request_skip_changes_nothing():
    model = make_model()
    opt = HookedOptimizer(adamw(model), model.named_parameters())
    backward(model, 0)
    opt.step()
    params_before = [p.detach().clone() for p in model.parameters()]
    state_before = {k: {kk: vv.clone() for kk, vv in v.items()} for k, v in opt.state.items()}
    opt.zero_grad()
    backward(model, 1)
    opt.request_skip()
    opt.step()
    assert all(torch.equal(a, b) for a, b in zip(params_before, model.parameters()))
    for k, v in opt.state.items():
        for kk, vv in v.items():
            assert torch.equal(vv, state_before[k][kk])
    assert (opt.nominal_steps, opt.applied_updates, opt.skipped_steps) == (2, 1, 1)


def test_state_dict_roundtrip_weights_only_and_bit_exact_resume():
    def run(steps, model, opt):
        for step in steps:
            opt.zero_grad()
            backward(model, step)
            opt.step()

    # uninterrupted
    m_full = make_model()
    mask_full = first_layer_mask(m_full)
    ledger_full = DisplacementLedger()
    o_full = HookedOptimizer(adamw(m_full), m_full.named_parameters(), [MaskRule(mask_full), ledger_full])
    run(range(6), m_full, o_full)

    # interrupted after 3 steps, saved with torch.save and reloaded weights_only like HF does
    m_a = make_model()
    o_a = HookedOptimizer(
        adamw(m_a), m_a.named_parameters(), [MaskRule(first_layer_mask(m_a)), DisplacementLedger()]
    )
    run(range(3), m_a, o_a)
    buf, mbuf = io.BytesIO(), io.BytesIO()
    torch.save(o_a.state_dict(), buf)
    torch.save(m_a.state_dict(), mbuf)
    buf.seek(0)
    mbuf.seek(0)

    m_b = make_model(seed=123)  # different init, must be overwritten by the checkpoint
    m_b.load_state_dict(torch.load(mbuf, weights_only=True))
    ledger_b = DisplacementLedger()
    o_b = HookedOptimizer(adamw(m_b), m_b.named_parameters(), [MaskRule(first_layer_mask(m_b)), ledger_b])
    o_b.load_state_dict(torch.load(buf, weights_only=True))
    assert (o_b.nominal_steps, o_b.applied_updates) == (3, 3)
    run(range(3, 6), m_b, o_b)

    assert params_equal(m_full, m_b)
    assert o_b.applied_updates == 6
    assert ledger_b.total_sq_norm == pytest.approx(ledger_full.total_sq_norm, rel=1e-12)


def test_lr_scheduler_drives_inner_optimizer():
    model = make_model()
    opt = HookedOptimizer(adamw(model), model.named_parameters())
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 0.5**step)
    for step in range(3):
        backward(model, step)
        opt.step()
        sched.step()
    assert opt.inner.param_groups[0]["lr"] == pytest.approx(1e-2 * 0.5**3)
    assert opt.param_groups is opt.inner.param_groups


def test_mask_covers_requires_grad_false_param_with_stale_grad():
    """A param disabled via requires_grad=False but holding a stale grad must still obey the mask."""
    model = make_model()
    opt = HookedOptimizer(adamw(model), model.named_parameters())
    backward(model, 0)
    opt.step()  # warm moments everywhere
    mask = first_layer_mask(model)
    opt.add_rule(MaskRule(mask))
    backward(model, 1)  # leaves grads on layer 0
    model[0].weight.requires_grad_(False)  # disabled, stale grad still present
    before = model[0].weight.detach().clone()
    opt.step()
    assert torch.equal(before[3:], model[0].weight.detach()[3:])


def test_mask_state_is_digest_only_and_verified_on_resume():
    model = make_model()
    opt = HookedOptimizer(adamw(model), model.named_parameters(), [MaskRule(first_layer_mask(model))])
    backward(model, 0)
    opt.step()
    buf = io.BytesIO()
    torch.save(opt.state_dict(), buf)
    rule_state = opt.state_dict()["research_rl"]["rules"]["0:mask"]
    assert set(rule_state) == {"mask_sha256"}  # no full bool tensors in the checkpoint
    buf.seek(0)
    sd = torch.load(buf, weights_only=True)
    other = first_layer_mask(model)
    other.set("0.bias", torch.ones(8, dtype=torch.bool))  # a different mask
    opt2 = HookedOptimizer(adamw(make_model()), make_model().named_parameters(), [MaskRule(other)])
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        opt2.load_state_dict(sd)


class VetoRule(UpdateRule):
    name = "veto"

    def __init__(self, opt_ref):
        self.opt_ref = opt_ref
        self.skips = 0

    def post_grad(self, ctx, named_params):
        self.opt_ref[0].request_skip()

    def on_skip(self, ctx):
        self.skips += 1


def test_post_grad_hook_can_skip_the_current_step():
    model = make_model()
    ref = [None]
    rule = VetoRule(ref)
    opt = HookedOptimizer(adamw(model), model.named_parameters())
    ref[0] = opt
    backward(model, 0)
    opt.step()  # warm state
    opt.add_rule(rule)
    params_before = [p.detach().clone() for p in model.parameters()]
    state_before = {k: {kk: vv.clone() for kk, vv in v.items()} for k, v in opt.state.items()}
    backward(model, 1)
    opt.step()
    assert all(torch.equal(a, b) for a, b in zip(params_before, model.parameters()))
    for k, v in opt.state.items():
        for kk, vv in v.items():
            assert torch.equal(vv, state_before[k][kk])
    assert (opt.nominal_steps, opt.applied_updates, opt.skipped_steps, rule.skips) == (2, 1, 1, 1)
