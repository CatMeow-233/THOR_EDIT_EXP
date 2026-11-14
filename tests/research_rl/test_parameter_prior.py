"""Numerical tests for task vectors, the task-vector prior and the P²-MOPD parameter-prior rules.

Every expectation is computed by an independent reference inside the test (plain torch autograd,
``torch.optim.Adam``/``AdamW``, ``torch.linalg.lstsq``, finite differences in float64).
"""

from __future__ import annotations

import copy
import io
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from research_rl.core.registry import PARAMETER_PRIORS, UPDATE_RULES
from research_rl.models.tiny import build_tiny_llama
from research_rl.optim import HookedOptimizer
from research_rl.optim.parameter_prior import (
    P2PriorAttachCallback,
    P2PriorRule,
    PriorSubspaceRule,
    TaskVector,
    TaskVectorPrior,
    default_prior_modules,
    project_onto_span,
)

V = 64  # vocab of the tiny Llama used here (only the token range matters for these tests)


# ------------------------------------------------------------------------------------------- helpers
def tiny(seed: int = 0) -> nn.Module:
    return build_tiny_llama(V, seed=seed)


def random_task_vectors(model: nn.Module, names: list[str], k: int, seed: int, scale: float = 0.05):
    g = torch.Generator().manual_seed(seed)
    params = dict(model.named_parameters())
    return [
        TaskVector({n: scale * torch.randn(params[n].shape, generator=g) for n in names}) for _ in range(k)
    ]


def token_batch(step: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(100 + step)
    return torch.randint(3, V, (2, 10), generator=g)


def ref_group_index(names: list[str], granularity: str) -> dict[str, int]:
    """Independent re-implementation of the coefficient tying (module / layer / model)."""
    keys = []
    for n in names:
        if granularity == "module":
            keys.append(n)
        elif granularity == "layer":
            keys.append(n.split("layers.")[1].split(".")[0])
        else:
            keys.append("all")
    order = list(dict.fromkeys(keys))
    return {n: order.index(key) for n, key in zip(names, keys)}


def explicit_weights(base, alpha, tvs, names, gidx):
    """W_m = B_m + Σ_k α[g(m), k] τ_k[m] built with plain torch (differentiable in B and α)."""
    return {n: base[n] + sum(alpha[gidx[n], k] * tvs[k][n] for k in range(len(tvs))) for n in names}


# ------------------------------------------------------------------------------------------- task vectors
def test_task_vector_arithmetic_norms_and_cosine():
    base, e1, e2 = tiny(0), tiny(1), tiny(2)
    names = [n for n, _ in base.named_parameters()][:5]
    t1 = TaskVector.from_models(e1, base, names)
    t2 = TaskVector.from_models(e2, base, names)
    pb, p1, p2 = (dict(m.named_parameters()) for m in (base, e1, e2))
    for n in names:
        assert torch.equal(t1[n], p1[n].detach() - pb[n].detach())
        assert torch.equal(t2[n], p2[n].detach() - pb[n].detach())
    s = 2.0 * t1 - t2 + (-t1)
    for n in names:
        torch.testing.assert_close(s[n], t1[n] - t2[n])
    flat1 = torch.cat([t1[n].double().flatten() for n in names])
    flat2 = torch.cat([t2[n].double().flatten() for n in names])
    assert t1.norm() == pytest.approx(float(flat1.norm()), rel=1e-12)
    assert t1.dot(t2) == pytest.approx(float(flat1 @ flat2), rel=1e-10)
    ref_cos = float(torch.nn.functional.cosine_similarity(flat1, flat2, dim=0))
    assert t1.cosine_similarity(t2) == pytest.approx(ref_cos, rel=1e-9)
    assert t1.cosine_similarity(3.0 * t1) == pytest.approx(1.0, rel=1e-12)
    pc = t1.param_cosine_similarity(t2)
    for n in names:
        ref = float(
            torch.nn.functional.cosine_similarity(t1[n].double().flatten(), t2[n].double().flatten(), 0)
        )
        assert pc[n] == pytest.approx(ref, rel=1e-9)
    with pytest.raises(KeyError):
        _ = t1 + TaskVector({names[0]: t1[names[0]]})


def test_prior_combine_and_apply_to_are_exact():
    base, e1, e2 = tiny(0), tiny(1), tiny(2)
    names = default_prior_modules(n for n, _ in base.named_parameters())
    assert len(names) == 14 and all(n.endswith("_proj.weight") for n in names)
    prior = TaskVectorPrior(base, names)
    prior.register("math", TaskVector.from_models(e1, base, names))
    prior.register("code", TaskVector.from_models(e2, base, names))
    target = tiny(5)
    written = prior.apply_to(target, {"math": 0.3, "code": -0.7})
    params, pb, p1, p2 = (dict(m.named_parameters()) for m in (target, base, e1, e2))
    for n in names:
        assert torch.equal(params[n], written[n])  # bit-exact write
        ref = pb[n] + 0.3 * (p1[n] - pb[n]) - 0.7 * (p2[n] - pb[n])
        torch.testing.assert_close(params[n], ref.detach(), rtol=0, atol=1e-6)
    prior.apply_to(target, [0.0, 0.0])
    for n in names:
        assert torch.equal(params[n], pb[n])  # zero coefficients restore the base bit-for-bit
    with pytest.raises(KeyError):
        prior.combine({"unknown": 1.0})


def test_prior_functional_call_gradient_matches_chain_rule_and_finite_differences():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3)).double()
    names = ["0.weight", "2.weight"]
    prior = TaskVectorPrior(net, names)
    g = torch.Generator().manual_seed(3)
    for i in range(3):
        prior.register(
            f"t{i}",
            TaskVector(
                {
                    n: 0.3
                    * torch.randn(dict(net.named_parameters())[n].shape, generator=g, dtype=torch.float64)
                    for n in names
                }
            ),
        )
    # combine() accumulates in fp32; use a float64 copy of the formula for the FD check via functional_call.
    x = torch.randn(4, 5, generator=g, dtype=torch.float64)
    y = torch.randn(4, 3, generator=g, dtype=torch.float64)

    def loss_at(alpha: torch.Tensor) -> torch.Tensor:
        params = {
            n: prior.base[n] + sum(alpha[i] * prior.task_vectors[f"t{i}"][n] for i in range(3)) for n in names
        }
        out = torch.func.functional_call(net, params, (x,))
        return ((out - y) ** 2).mean()

    alpha = torch.tensor([0.2, -0.5, 0.9], dtype=torch.float64, requires_grad=True)
    # 1) the prior's differentiable path vs autograd of an explicit float64 formula
    out = prior.functional_call(net, alpha, (x,))
    loss = ((out - y) ** 2).mean()
    (grad_prior,) = torch.autograd.grad(loss, alpha)
    (grad_ref,) = torch.autograd.grad(loss_at(alpha), alpha)
    torch.testing.assert_close(grad_prior, grad_ref, rtol=1e-5, atol=1e-7)  # combine() runs in fp32
    # 2) manual chain rule: dL/dα_i = Σ_p <∂L/∂θ_p, τ_i,p>
    theta = {
        n: (prior.base[n] + sum(float(alpha[i]) * prior.task_vectors[f"t{i}"][n] for i in range(3)))
        .clone()
        .requires_grad_()
        for n in names
    }
    lt = ((torch.func.functional_call(net, theta, (x,)) - y) ** 2).mean()
    gt = torch.autograd.grad(lt, [theta[n] for n in names])
    chain = torch.stack(
        [sum((gp * prior.task_vectors[f"t{i}"][n]).sum() for gp, n in zip(gt, names)) for i in range(3)]
    )
    torch.testing.assert_close(grad_ref, chain, rtol=1e-10, atol=1e-12)
    # 3) central finite differences in float64
    h = 1e-6
    fd = torch.stack(
        [
            (loss_at(alpha.detach() + h * e) - loss_at(alpha.detach() - h * e)) / (2 * h)
            for e in torch.eye(3, dtype=torch.float64)
        ]
    )
    torch.testing.assert_close(grad_ref, fd, rtol=1e-6, atol=1e-9)


# ------------------------------------------------------------------------------------------- projection
def test_project_onto_span_matches_lstsq_including_rank_deficient_basis():
    g = torch.Generator().manual_seed(0)
    basis = [torch.randn(6, 5, generator=g, dtype=torch.float64) for _ in range(3)]
    d = torch.randn(6, 5, generator=g, dtype=torch.float64)
    res = project_onto_span(d, basis)
    t = torch.stack([b.flatten() for b in basis], 1)
    ref = torch.linalg.lstsq(t, d.flatten().unsqueeze(1), driver="gelsd").solution.squeeze(1)
    torch.testing.assert_close(res.coefficients, ref)
    torch.testing.assert_close(res.projection.flatten(), t @ ref)
    torch.testing.assert_close(res.residual, d - res.projection)
    assert abs(float((t.T @ res.residual.flatten()).abs().max())) < 1e-10  # residual ⟂ span
    assert res.relative_residual == pytest.approx(float(res.residual.norm() / d.norm()))
    # duplicated direction: projection is unique, coefficients are the minimum-norm solution
    dup = [*basis, 2.0 * basis[0]]
    res2 = project_onto_span(d, dup)
    t2 = torch.stack([b.flatten() for b in dup], 1)
    torch.testing.assert_close(res2.projection, res.projection)
    torch.testing.assert_close(res2.coefficients, torch.linalg.pinv(t2) @ d.flatten())
    # a displacement inside the span is reproduced exactly
    inside = 0.5 * basis[0] - 2.0 * basis[2]
    r3 = project_onto_span(inside, basis)
    torch.testing.assert_close(r3.coefficients, torch.tensor([0.5, 0.0, -2.0], dtype=torch.float64))
    assert r3.relative_residual < 1e-12


@pytest.mark.parametrize("mode", ["onto", "away"])
def test_prior_subspace_rule_projects_adam_update(mode):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 3))
    g = torch.Generator().manual_seed(1)
    tvs = [TaskVector({"0.weight": torch.randn(8, 6, generator=g)}) for _ in range(3)]
    rule = PriorSubspaceRule(tvs, mode=mode)
    opt = HookedOptimizer(
        torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=0.1), net.named_parameters(), [rule]
    )
    x, y = torch.randn(5, 6, generator=g), torch.randn(5, 3, generator=g)
    for step in range(4):
        # reference: the same AdamW state on a clone, unconstrained
        ref = copy.deepcopy(net)
        ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-2, weight_decay=0.1)
        ref_opt.load_state_dict(copy.deepcopy(opt.inner.state_dict()))
        old = {n: p.detach().clone() for n, p in net.named_parameters()}
        for m, o in ((net, opt), (ref, ref_opt)):
            o.zero_grad()
            ((m(x * (step + 1)) - y) ** 2).mean().backward()
            o.step()
        delta = ref[0].weight.detach().double() - old["0.weight"].double()
        t = torch.stack([tv["0.weight"].double().flatten() for tv in tvs], 1)
        coef = torch.linalg.lstsq(t, delta.flatten().unsqueeze(1)).solution.squeeze(1)
        proj = (t @ coef).reshape(8, 6)
        expected = proj if mode == "onto" else delta - proj
        actual = net[0].weight.detach().double() - old["0.weight"].double()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
        assert float(delta.norm()) > 1e-4 and float((delta - expected).norm()) > 1e-4  # not vacuous
        for n in ("0.bias", "2.weight", "2.bias"):  # uncovered parameters: plain AdamW bit-for-bit
            assert torch.equal(dict(net.named_parameters())[n], dict(ref.named_parameters())[n])


# ------------------------------------------------------------------------------------------- P2 prior rule
PRIOR_NAMES = default_prior_modules(n for n, _ in tiny().named_parameters())


def make_rule_setup(
    granularity: str,
    k: int,
    *,
    lr_alpha: float,
    clip_w: float | None,
    clip_alpha,
    lam: float | None,
    wd: float = 0.0,
    lr: float = 3e-3,
):
    base_model = tiny(0)
    base = {n: p.detach().clone() for n, p in base_model.named_parameters()}
    tvs = random_task_vectors(base_model, PRIOR_NAMES, k, seed=11)
    model = tiny(0)
    rule = P2PriorRule(
        tvs,
        PRIOR_NAMES,
        granularity=granularity,
        init=lam,
        lr=lr_alpha,
        betas=(0.8, 0.95),
        eps=1e-8,
        clip_norm=clip_alpha,
    )
    rule.materialize(model, base)
    inner = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=wd)
    opt = HookedOptimizer(inner, model.named_parameters(), [rule], clip_norm=clip_w)
    return SimpleNamespace(base=base, tvs=tvs, model=model, rule=rule, opt=opt)


LOSS_SCALES = [1.0, 0.02, 4.0, 0.1, 2.0, 0.5]


def lm_loss(model: nn.Module, step: int, params: dict | None = None) -> torch.Tensor:
    ids = token_batch(step)
    if params is None:
        out = model(input_ids=ids, labels=ids)
    else:
        out = torch.func.functional_call(model, params, (), {"input_ids": ids, "labels": ids})
    return LOSS_SCALES[step % len(LOSS_SCALES)] * out.loss


@pytest.mark.parametrize("granularity,k", [("module", 2), ("layer", 3), ("model", 3)])
def test_p2_rule_equivalent_to_explicit_b_alpha_optimisation(granularity, k):
    c_w, c_a, lr_a, lam = 1.0, 0.1, 2e-2, 1.0 / k
    s = make_rule_setup(granularity, k, lr_alpha=lr_a, clip_w=c_w, clip_alpha=c_a, lam=lam)
    s.rule.attach(s.opt)
    gidx = ref_group_index(PRIOR_NAMES, granularity)
    n_groups = len(set(gidx.values()))
    assert s.rule.num_free == n_groups * k

    # ---- independent reference: explicit B (AdamW, wd=0) and α (Adam on clipped autograd grad)
    ref_model = tiny(0)
    others = {n: p for n, p in ref_model.named_parameters() if n not in PRIOR_NAMES}
    b = {n: s.base[n].clone().requires_grad_() for n in PRIOR_NAMES}
    alpha = torch.full((n_groups, k), lam, requires_grad=True)
    w_params = [*b.values(), *others.values()]
    ref_w_opt = torch.optim.AdamW(w_params, lr=3e-3, betas=(0.9, 0.99), weight_decay=0.0)
    ref_a_opt = torch.optim.Adam([alpha], lr=lr_a, betas=(0.8, 0.95), eps=1e-8)

    # initial materialised weights agree
    for n, p in s.model.named_parameters():
        if n in PRIOR_NAMES:
            ref_w = explicit_weights(b, alpha, s.tvs, [n], gidx)[n]
            torch.testing.assert_close(p.detach(), ref_w.detach(), rtol=0, atol=1e-7)

    regimes = {"w_binds_a_free": 0, "a_binds": 0}
    for step in range(len(LOSS_SCALES)):
        s.opt.zero_grad()
        lm_loss(s.model, step).backward()
        s.opt.step()

        ref_w_opt.zero_grad()
        ref_a_opt.zero_grad()
        weights = explicit_weights(b, alpha, s.tvs, PRIOR_NAMES, gidx)
        lm_loss(ref_model, step, weights).backward()
        g_alpha = alpha.grad.detach().reshape(-1).clone()
        g_norm = float(g_alpha.norm())
        alpha.grad.mul_(min(1.0, c_a / (g_norm + 1e-6)))
        w_norm = float(torch.nn.utils.clip_grad_norm_(w_params, c_w))
        regimes["w_binds_a_free"] += int(w_norm > c_w and g_norm < c_a)
        regimes["a_binds"] += int(g_norm > c_a)
        ref_w_opt.step()
        ref_a_opt.step()

        # gradients themselves (Adam is ~scale-invariant, so check them directly)
        torch.testing.assert_close(s.rule.last_grad, g_alpha, rtol=2e-4, atol=1e-7)
        torch.testing.assert_close(s.rule.last_clipped_grad, alpha.grad.reshape(-1), rtol=2e-4, atol=1e-7)
        torch.testing.assert_close(s.rule.coefficients, alpha.detach().reshape(-1), rtol=1e-5, atol=1e-6)
        weights = explicit_weights(b, alpha, s.tvs, PRIOR_NAMES, gidx)
        for n, p in s.model.named_parameters():
            ref = weights[n] if n in PRIOR_NAMES else others[n]
            torch.testing.assert_close(p.detach(), ref.detach(), rtol=1e-4, atol=2e-5)
    # both clip regimes were exercised and α actually moved
    assert regimes["w_binds_a_free"] >= 1 and regimes["a_binds"] >= 1, regimes
    traj = s.rule.trajectory_tensor()
    assert traj.shape == (len(LOSS_SCALES) + 1, s.rule.num_free)
    assert float((traj[1:] - traj[:-1]).abs().max()) > 0.5 * lr_a  # Adam steps are ~lr in size
    torch.testing.assert_close(traj[0], torch.full((s.rule.num_free,), lam))
    torch.testing.assert_close(traj[-1], s.rule.coefficients)


@pytest.mark.parametrize("granularity", ["module", "layer", "model"])
def test_p2_coefficient_gradient_equals_autograd_and_tying_sums(granularity):
    k = 3
    base_model = tiny(0)
    base = {n: p.detach().clone() for n, p in base_model.named_parameters()}
    tvs = random_task_vectors(base_model, PRIOR_NAMES, k, seed=5)
    gidx = ref_group_index(PRIOR_NAMES, granularity)
    n_groups = len(set(gidx.values()))
    g = torch.Generator().manual_seed(9)
    alpha = (0.3 * torch.randn(n_groups, k, generator=g)).requires_grad_()
    # autograd through W(B, α) via functional_call
    weights = explicit_weights(base, alpha, tvs, PRIOR_NAMES, gidx)
    lm_loss(base_model, 0, weights).backward()
    # the rule on a materialised model with the same α
    rule = P2PriorRule(tvs, PRIOR_NAMES, granularity=granularity)
    rule.coefficients = alpha.detach().reshape(-1).clone()
    model = tiny(0)
    rule.materialize(model, base)
    lm_loss(model, 0).backward()
    got = rule.coefficient_gradient(model.named_parameters())
    torch.testing.assert_close(got, alpha.grad.reshape(-1), rtol=2e-4, atol=1e-7)
    assert float(got.abs().min()) > 0
    # tying = sum of the per-module gradients over the covered modules
    per_module = P2PriorRule(tvs, PRIOR_NAMES, granularity="module").coefficient_gradient(
        model.named_parameters()
    )
    per_module = per_module.reshape(len(PRIOR_NAMES), k)
    expected = torch.zeros(n_groups, k)
    for i, n in enumerate(PRIOR_NAMES):
        expected[gidx[n]] += per_module[i]
    torch.testing.assert_close(got, expected.reshape(-1), rtol=1e-5, atol=1e-7)
    table = rule.coefficient_table()
    for n in PRIOR_NAMES:
        torch.testing.assert_close(table[n], alpha.detach()[gidx[n]])


def test_p2_zero_coefficient_lr_is_bit_identical_to_plain_adamw_and_rule_is_live():
    def run(lr_alpha: float | None):
        s = make_rule_setup("layer", 2, lr_alpha=lr_alpha or 0.0, clip_w=1.0, clip_alpha=None, lam=0.5)
        if lr_alpha is None:  # plain AdamW (same wrapper clip, no rule)
            s.opt = HookedOptimizer(s.opt.inner, s.model.named_parameters(), [], clip_norm=1.0)
        else:
            s.rule.attach(s.opt)
        for step in range(4):
            s.opt.zero_grad()
            lm_loss(s.model, step).backward()
            s.opt.step()
        return s

    plain, zero, live = run(None), run(0.0), run(1e-2)
    for (n, p), q in zip(plain.model.named_parameters(), zero.model.parameters()):
        assert torch.equal(p, q), n
    assert zero.rule.last_grad is not None and float(zero.rule.last_grad.abs().max()) > 0
    torch.testing.assert_close(
        zero.rule.coefficients, torch.full_like(zero.rule.coefficients, 0.5), rtol=0, atol=0
    )
    changed = [
        n
        for (n, p), q in zip(plain.model.named_parameters(), live.model.parameters())
        if not torch.equal(p, q)
    ]
    assert set(PRIOR_NAMES) <= set(changed)


def test_p2_state_dict_round_trip_resumes_bit_exactly():
    def fresh():
        s = make_rule_setup("module", 2, lr_alpha=1e-2, clip_w=1.0, clip_alpha=0.5, lam=0.5)
        s.rule.attach(s.opt)
        return s

    def steps(s, start, n):
        for step in range(start, start + n):
            s.opt.zero_grad()
            lm_loss(s.model, step).backward()
            s.opt.step()

    a = fresh()
    steps(a, 0, 3)
    buf = io.BytesIO()
    torch.save({"model": a.model.state_dict(), "opt": a.opt.state_dict()}, buf)
    steps(a, 3, 3)

    buf.seek(0)
    saved = torch.load(buf, weights_only=True)
    assert "research_rl" in saved["opt"]
    b = fresh()
    b.model.load_state_dict(saved["model"])
    b.opt.load_state_dict(saved["opt"])
    assert b.rule.step_count == 3 and b.rule.trajectory_tensor().shape[0] == 4
    steps(b, 3, 3)
    for (n, p), q in zip(a.model.named_parameters(), b.model.parameters()):
        assert torch.equal(p, q), n
    assert torch.equal(a.rule.coefficients, b.rule.coefficients)
    assert torch.equal(a.rule.exp_avg_sq, b.rule.exp_avg_sq)
    assert torch.equal(a.rule.trajectory_tensor(), b.rule.trajectory_tensor())

    # sensitivity: restoring weights + AdamW state but NOT the rule state diverges
    c = fresh()
    c.model.load_state_dict(saved["model"])
    opt_state = {k: v for k, v in saved["opt"].items() if k != "research_rl"}
    c.opt.inner.load_state_dict(opt_state)
    steps(c, 3, 3)
    assert not all(torch.equal(p, q) for p, q in zip(a.model.parameters(), c.model.parameters()))


def test_p2_validation_weight_decay_attach_and_callback():
    # HookedOptimizer calls rule.on_attach() at registration: decay on prior modules fails immediately
    with pytest.raises(ValueError, match="weight_decay"):
        make_rule_setup("model", 2, lr_alpha=1e-2, clip_w=1.0, clip_alpha=None, lam=None, wd=0.01)
    # decay only on non-prior parameters is allowed
    s = make_rule_setup("model", 2, lr_alpha=1e-2, clip_w=1.0, clip_alpha=None, lam=None)
    params = dict(s.model.named_parameters())
    groups = [
        {"params": [params[n] for n in PRIOR_NAMES], "weight_decay": 0.0},
        {"params": [p for n, p in params.items() if n not in PRIOR_NAMES], "weight_decay": 0.1},
    ]
    opt = HookedOptimizer(torch.optim.AdamW(groups, lr=1e-3), s.model.named_parameters(), [s.rule])
    # the explicit callback path still works (idempotent re-attach)
    P2PriorAttachCallback(s.rule).on_train_begin(None, None, None, optimizer=SimpleNamespace(optimizer=opt))
    opt.zero_grad()
    lm_loss(s.model, 0).backward()
    opt.step()
    assert s.rule.step_count == 1
    # decay switched on later is caught at the next step
    opt.param_groups[0]["weight_decay"] = 0.05
    with pytest.raises(ValueError, match="weight_decay"):
        opt.step()
    # a frozen prior module (even with a stale .grad) is rejected instead of being moved by dα·τ
    opt.param_groups[0]["weight_decay"] = 0.0
    frozen = params[PRIOR_NAMES[0]]
    frozen.requires_grad_(False)
    assert frozen.grad is not None  # stale gradient from the previous backward
    with pytest.raises(ValueError, match="requires_grad=False"):
        opt.step()
    with pytest.raises(ValueError, match="requires_grad=False"):
        s.rule.attach(opt)
    frozen.requires_grad_(True)
    # a rule not registered in the optimizer cannot attach
    other = P2PriorRule(s.tvs, PRIOR_NAMES)
    with pytest.raises(ValueError, match="not registered"):
        other.attach(opt)
    with pytest.raises(ValueError, match="layout"):
        P2PriorRule(s.tvs, PRIOR_NAMES, granularity="layer").load_state_dict(s.rule.state_dict())


def test_registries():
    assert PARAMETER_PRIORS.get("task_vector_prior") is TaskVectorPrior
    assert PARAMETER_PRIORS.get("task_vector") is TaskVector
    assert UPDATE_RULES.get("p2_prior") is P2PriorRule
    assert UPDATE_RULES.get("prior_subspace") is PriorSubspaceRule


def test_p2_refuses_when_trainer_clips_before_step():
    """HF/TRL clipping (max_grad_norm>0) happens before optimizer.step(): the rule must refuse."""
    s = make_rule_setup("model", 2, lr_alpha=1e-2, clip_w=1.0, clip_alpha=None, lam=None)
    s.opt.step_extras["hf_max_grad_norm"] = 1.0
    s.opt.zero_grad()
    lm_loss(s.model, 0).backward()
    with pytest.raises(RuntimeError, match="UNCLIPPED"):
        s.opt.step()
