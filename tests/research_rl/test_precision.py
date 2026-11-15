"""Low-precision updates: stochastic rounding and HookedOptimizer fp32 master weights.

Every numerical claim is checked against an independent reference (an fp64 enumeration of all bf16
values, plain AdamW on free fp32 tensors, closed-form P2 corrections).
"""

from __future__ import annotations

import copy
import io
import math
import warnings

import pytest
import torch
from torch import nn

from research_rl.core.config import ConfigError, DataSpec, ExperimentConfig, ModelSpec, OptimSpec, TrainerSpec
from research_rl.models.parameter_mask import ParameterMask
from research_rl.optim import (
    DisplacementLedger,
    HookedOptimizer,
    MaskRule,
    P2PriorRule,
    TaskVector,
    UpdateRule,
    is_low_precision,
    low_precision_trainable,
    round_to,
    rounding_generator,
    stochastic_round,
)
from research_rl.optim.precision import rounding_seed_for

BF16_MAX = torch.finfo(torch.bfloat16).max


# ------------------------------------------------------------------------------------------ helpers
def all_finite_bf16() -> torch.Tensor:
    """Every finite bf16 value (both zeros included), as bf16."""
    bits = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16)
    vals = bits.view(torch.bfloat16)
    return vals[torch.isfinite(vals)]


def bf16_brackets(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent reference: (lo, hi, P[hi]) in fp64 from the sorted list of all bf16 values."""
    grid = torch.unique(all_finite_bf16().double())  # sorted, -0.0 merged into 0.0
    xd = x.double()
    hi_i = torch.searchsorted(grid, xd, right=False).clamp(max=grid.numel() - 1)
    hi = grid[hi_i]
    lo = torch.where(hi == xd, hi, grid[(hi_i - 1).clamp(min=0)])
    p = torch.where(hi == lo, torch.zeros_like(xd), (xd - lo) / (hi - lo))
    return lo, hi, p


def sample_inputs() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    mags = torch.exp2(torch.empty(120).uniform_(-30, 30, generator=g))
    signs = torch.where(torch.rand(120, generator=g) < 0.5, -1.0, 1.0)
    special = torch.tensor(
        [
            1.0 + 2**-10,  # P[hi] = 1/8
            1.0 + 3 * 2**-9,  # P[hi] = 3/4
            -(1.0 + 2**-10),
            2.0 - 2**-20,  # just below a binade boundary
            -(2.0 - 2**-20),
            1e-40,  # fp32 subnormal -> bf16 subnormal neighbours
            -3.3e-39,
            float(BF16_MAX) * (1 - 2**-12),  # below the bf16 max
            1.17549435e-38,  # smallest normal fp32
        ]
    )
    return torch.cat([signs * mags, special]).float()


def make_model(seed: int = 0, dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 3)).to(dtype)


def backward(model: nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(1000 + seed)
    dtype = next(model.parameters()).dtype
    x, y = torch.randn(5, 6, generator=g).to(dtype), torch.randn(5, 3, generator=g).to(dtype)
    ((model(x) - y) ** 2).mean().backward()


class Plate(nn.Module):
    """One bf16 weight with |w| in [1, 2) (ulp 2**-7): Adam steps of ~lr are far below half an ulp."""

    def __init__(self, seed: int = 0, shape: tuple[int, int] = (64, 64)) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w = 1.0 + torch.rand(shape, generator=g)
        self.weight = nn.Parameter(w.bfloat16())

    def loss(self, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        c = 0.5 + torch.rand(self.weight.shape, generator=g)  # positive: every element is pushed down
        return (self.weight.float() * c).sum()


def quiet_hooked(*args, **kwargs) -> HookedOptimizer:
    kwargs.setdefault("precision_check", "off")
    return HookedOptimizer(*args, **kwargs)


# ------------------------------------------------------------------------------- precision helpers
def test_dtype_helpers():
    assert is_low_precision(torch.bfloat16) and is_low_precision(torch.float16)
    assert not is_low_precision(torch.float32) and not is_low_precision(torch.float64)
    model = make_model(dtype=torch.float32)
    model[2].to(torch.bfloat16)
    model[2].bias.requires_grad_(False)
    assert low_precision_trainable(model.named_parameters()) == ["2.weight"]


def test_stochastic_round_draws_only_neighbours_with_exact_probabilities():
    x = sample_inputs()
    lo, hi, p = bf16_brackets(x)
    # The bit trick rounds away from zero with probability low_bits / 2**16; that equals the fp64
    # linear-interpolation weight of the neighbour away from zero exactly (for both signs).
    low_bits = (x.view(torch.int32) & 0xFFFF).double() / 2**16
    p_away = torch.where(x >= 0, p, torch.where(hi == lo, p, 1 - p))
    assert torch.equal(p_away, low_bits)

    n = 4000
    gen = torch.Generator().manual_seed(123)
    draws = stochastic_round(x.repeat(n, 1), torch.bfloat16, gen).double()
    assert bool(((draws == lo) | (draws == hi)).all())
    freq = (draws == hi).double().mean(0)
    freq = torch.where(hi == lo, torch.zeros_like(freq), freq)
    bound = 5 * torch.sqrt(p * (1 - p) / n) + 1.0 / n
    assert bool(((freq - p).abs() <= bound).all()), float(((freq - p).abs() - bound).max())
    # Unbiased: the mean over draws is within the statistical bound of x itself.
    err = (draws.mean(0) - x.double()).abs()
    assert bool((err <= 5 * (hi - lo) * torch.sqrt(p * (1 - p) / n) + (hi - lo) / n).all())


def test_stochastic_round_mean_over_many_draws():
    x = torch.full((400_000,), 1.0 + 2**-10)  # 1/8 of the way from 1 to 1 + 2**-7
    out = stochastic_round(x, torch.bfloat16, torch.Generator().manual_seed(7)).double()
    sigma = 2**-7 * math.sqrt(0.125 * 0.875 / x.numel())
    assert abs(float(out.mean()) - (1.0 + 2**-10)) < 5 * sigma
    assert float(x.bfloat16().double().mean()) == 1.0  # nearest rounding is biased (never moves)


def test_stochastic_round_is_identity_on_representable_values_and_keeps_specials():
    vals = all_finite_bf16()
    out = stochastic_round(vals.float(), torch.bfloat16, torch.Generator().manual_seed(1))
    assert torch.equal(out.view(torch.int16), vals.view(torch.int16))  # bit-exact, incl. -0.0
    specials = torch.tensor([float("nan"), float("inf"), float("-inf"), -float("nan")])
    out = stochastic_round(specials, torch.bfloat16, torch.Generator().manual_seed(2))
    assert out.dtype == torch.bfloat16
    assert torch.isnan(out[0]) and torch.isnan(out[3]) and out[1] == float("inf") and out[2] == float("-inf")
    # Beyond the bf16 max: either the max or +inf (round-to-nearest overflows there too).
    big = torch.full((2000,), float(torch.finfo(torch.float32).max))
    out = stochastic_round(big, torch.bfloat16, torch.Generator().manual_seed(3)).float()
    assert set(out.unique().tolist()) <= {float(BF16_MAX), float("inf")}


def test_round_to_dispatch_and_unsupported_targets():
    x = sample_inputs()
    assert torch.equal(round_to(x, torch.bfloat16, "nearest"), x.to(torch.bfloat16))
    assert torch.equal(stochastic_round(x, torch.float32), x)
    assert stochastic_round(x, torch.float64).dtype == torch.float64
    with pytest.raises(NotImplementedError, match="float16"):
        stochastic_round(x, torch.float16)
    with pytest.raises(NotImplementedError, match="float16"):
        round_to(x, torch.float16, "stochastic")
    with pytest.raises(ValueError, match="float32 input"):
        stochastic_round(x.double(), torch.bfloat16)
    with pytest.raises(ValueError, match="rounding mode"):
        round_to(x, torch.bfloat16, "up")


def test_rounding_generator_is_a_pure_function_of_seed_step_index():
    def draw(seed, step, index):
        g = rounding_generator(seed, step, index, "cpu")
        return torch.randint(0, 2**16, (8,), generator=g)

    assert torch.equal(draw(0, 3, 5), draw(0, 3, 5))
    reused = rounding_generator(0, 3, 5, generator=torch.Generator())
    assert torch.equal(torch.randint(0, 2**16, (8,), generator=reused), draw(0, 3, 5))
    assert len({tuple(draw(*k).tolist()) for k in [(0, 3, 5), (1, 3, 5), (0, 4, 5), (0, 3, 6)]}) == 4
    assert 0 <= rounding_seed_for(2**70, -1, 3) < 2**63


# ------------------------------------------------------------------------------- the dead zone
@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_small_updates_vanish_without_masters_and_accumulate_with_them(rounding):
    lr, steps = 1e-4, 30  # 30 * 1e-4 = 3e-3 < half an ulp (3.9e-3) of |w| in [1, 2)
    plain, master = Plate(), Plate()
    w0 = plain.weight.detach().clone()
    o_plain = quiet_hooked(
        torch.optim.AdamW(plain.parameters(), lr=lr, weight_decay=0.0), plain.named_parameters()
    )
    o_master = HookedOptimizer(
        torch.optim.AdamW(master.parameters(), lr=lr, weight_decay=0.0),
        master.named_parameters(),
        master_weights=True,
        write_rounding=rounding,
    )
    ref = w0.float().clone().requires_grad_(True)  # independent fp32 reference
    o_ref = torch.optim.AdamW([ref], lr=lr, weight_decay=0.0)
    for step in range(steps):
        for net, opt in ((plain, o_plain), (master, o_master)):
            opt.zero_grad()
            net.loss(step).backward()
            opt.step()
        o_ref.zero_grad()
        ref.grad = master.weight.grad.float()
        o_ref.step()
    assert torch.equal(plain.weight, w0)  # every single update was rounded away
    m = o_master.master_of(master.weight)
    assert m is not None and m.dtype == torch.float32
    assert torch.equal(m, ref.detach())  # the master is exactly AdamW on fp32
    moved = float((w0.float() - m).mean())
    assert moved == pytest.approx(lr * steps, rel=0.05)
    served = master.weight.detach().float()
    if rounding == "nearest":
        assert torch.equal(master.weight, m.bfloat16())
    else:
        # Unbiased: the served weights moved by the master displacement on average.
        ulp = 2**-7
        sigma = ulp * 0.5 / math.sqrt(served.numel())
        assert abs(float((served - m).mean())) < 5 * sigma
        assert float((w0.float() - served).mean()) == pytest.approx(moved, rel=0.15)
    assert o_master.master_resyncs == 0


def test_tiny_llama_bf16_dead_zone_and_master_accumulation(tokenizer):
    """The real model: with lr*update << 2**-8*|w| plain bf16 AdamW cannot move large weights; masters do."""
    from research_rl.models.tiny import build_tiny_llama

    def make(master: bool):
        model = build_tiny_llama(len(tokenizer), seed=0).to(torch.bfloat16)
        opt = HookedOptimizer(
            torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0),
            model.named_parameters(),
            master_weights=master,
            precision_check="off",
        )
        return model, opt

    plain, o_plain = make(False)
    model, opt = make(True)
    w0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    ids = torch.tensor([[1, 5, 6, 7, 8, 9, 10, 2]])
    for _ in range(10):
        for net, o in ((plain, o_plain), (model, opt)):
            o.zero_grad()
            net(input_ids=ids, labels=ids).loss.backward()
            o.step()
    big_total, master_moved = 0, 0
    for (n, p), (_, q) in zip(plain.named_parameters(), model.named_parameters()):
        big = w0[n].float().abs() >= 2**-6  # half an ulp >= 2**-14 ≈ 6.1e-5 >> one Adam step (~lr = 1e-5)
        assert torch.equal(p.detach()[big], w0[n][big]), n  # every update rounded away
        m = opt.master_of(q)
        assert m is not None
        master_moved += int((m[big] != w0[n].float()[big]).sum())
        big_total += int(big.sum())
    assert big_total > 1000 and master_moved > 0.5 * big_total  # most large weights moved in the master
    assert opt.master_resyncs == 0


# ------------------------------------------------------------------------- bit-exact reference
@pytest.mark.parametrize("clip", [None, 0.05])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_master_mode_equals_adamw_on_fp32_copy_then_nearest(clip, dtype):
    model = make_model(dtype=dtype)
    opt = HookedOptimizer(
        torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1),
        model.named_parameters(),
        clip_norm=clip,
        master_weights=True,
    )
    refs = [p.detach().float().clone().requires_grad_(True) for p in model.parameters()]
    o_ref = torch.optim.AdamW(refs, lr=1e-2, weight_decay=0.1)
    for step in range(6):
        opt.zero_grad()
        backward(model, step)
        grads = [p.grad.detach().clone() for p in model.parameters()]
        if clip is not None:  # the wrapper clips the low-precision grads, before the fp32 conversion
            holders = [torch.zeros_like(g, requires_grad=True) for g in grads]
            for h, g in zip(holders, grads):
                h.grad = g
            torch.nn.utils.clip_grad_norm_(holders, clip)
            grads = [h.grad for h in holders]
        for r, g in zip(refs, grads):
            r.grad = g.float()
        opt.step()
        o_ref.step()
        for p, r in zip(model.parameters(), refs):
            assert p.dtype == dtype and p.grad.dtype == dtype
            assert torch.equal(opt.master_of(p), r.detach())
            assert torch.equal(p.detach(), r.detach().to(dtype))
    for p in model.parameters():
        assert all(v.dtype == torch.float32 for v in opt.inner.state[p].values())


def test_master_weights_on_fp32_model_is_plain_adamw():
    a, b = make_model(dtype=torch.float32), make_model(dtype=torch.float32)
    o_a = torch.optim.AdamW(a.parameters(), lr=1e-2, weight_decay=0.1)
    o_b = HookedOptimizer(
        torch.optim.AdamW(b.parameters(), lr=1e-2, weight_decay=0.1),
        b.named_parameters(),
        master_weights=True,
    )
    assert not o_b._masters
    for step in range(4):
        for net, opt in ((a, o_a), (b, o_b)):
            opt.zero_grad()
            backward(net, step)
            opt.step()
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_post_step_sees_fp32_master_displacements_and_rules_see_fp32():
    seen: list[torch.dtype] = []

    class Spy(UpdateRule):
        name = "spy"

        def tracks(self, name, param):
            return True

        def transform_update(self, ctx, name, param, old, delta):
            seen.extend([param.dtype, old.dtype])
            return None

    model = make_model()
    ledger = DisplacementLedger()
    opt = HookedOptimizer(
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        model.named_parameters(),
        [Spy(), ledger],
        master_weights=True,
    )
    before = {n: opt.master_of(p).clone() for n, p in model.named_parameters()}
    opt.zero_grad()
    backward(model, 0)
    opt.step()
    assert set(seen) == {torch.float32}
    for n, p in model.named_parameters():
        assert torch.equal(ledger.last[n], opt.master_of(p) - before[n])


# -------------------------------------------------------------------------------- masks / frozen
def first_layer_mask(model: nn.Module) -> ParameterMask:
    w = dict(model.named_parameters())["0.weight"]
    m = torch.zeros_like(w, dtype=torch.bool)
    m[:3] = True
    return ParameterMask({"0.weight": m, "0.bias": torch.zeros(8, dtype=torch.bool)})


@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_mask_installed_after_warmup_is_bit_exact_in_master_and_served(rounding):
    model = make_model()
    opt = HookedOptimizer(
        torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1),
        model.named_parameters(),
        master_weights=True,
        write_rounding=rounding,
    )
    for step in range(3):  # warm-up: moments and (stochastic) non-representable masters
        opt.zero_grad()
        backward(model, step)
        opt.step()
    w = model[0].weight
    master = opt.master_of(w)
    assert not torch.equal(master, w.detach().float())  # masters hold sub-ulp information (not vacuous)
    mask = first_layer_mask(model)
    opt.add_rule(MaskRule(mask))
    served_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    masters_before = {n: opt.master_of(p).clone() for n, p in model.named_parameters()}
    for step in range(3, 9):
        opt.zero_grad()
        backward(model, step)
        opt.step()
    served_after = {n: p.detach().clone() for n, p in model.named_parameters()}
    masters_after = {n: opt.master_of(p).clone() for n, p in model.named_parameters()}
    assert mask.changed_outside(served_before, served_after) == {"0.weight": 0, "0.bias": 0}
    assert mask.changed_outside(masters_before, masters_after) == {"0.weight": 0, "0.bias": 0}
    assert not torch.equal(masters_before["0.weight"][:3], masters_after["0.weight"][:3])


def test_frozen_parameter_without_grad_is_never_rewritten():
    model = make_model()
    opt = HookedOptimizer(
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        model.named_parameters(),
        master_weights=True,
        write_rounding="stochastic",
    )
    model[2].weight.requires_grad_(False)
    frozen = model[2].weight.detach().clone()
    for step in range(3):
        opt.zero_grad()
        backward(model, step)
        opt.step()
    assert torch.equal(model[2].weight, frozen)


def test_parameter_frozen_after_stochastic_steps_stays_bit_exact():
    """Frozen once its master is no longer representable: re-rounding it would jitter the served weight."""
    model = make_model()
    opt = HookedOptimizer(
        torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0),
        model.named_parameters(),
        master_weights=True,
        write_rounding="stochastic",
    )
    w = model[2].weight
    run_steps(model, opt, range(3))
    master = opt.master_of(w).clone()
    assert not torch.equal(master, w.detach().float())  # not vacuous: re-rounding could change w
    w.requires_grad_(False)
    served = w.detach().clone()
    for step in range(3, 8):
        opt.zero_grad(set_to_none=True)
        backward(model, step)
        opt.step()
    assert w.grad is None
    assert torch.equal(w.detach(), served) and torch.equal(opt.master_of(w), master)
    assert not torch.equal(opt.master_of(model[0].weight), model[0].weight.detach().float())


def test_stochastic_noise_is_fresh_at_every_step():
    """Time average of the served weight -> master (a step-constant draw would pin each element)."""
    p = nn.Parameter(torch.ones(64, dtype=torch.bfloat16))
    opt = HookedOptimizer(
        torch.optim.SGD([p], lr=0.1), [("w", p)], master_weights=True, write_rounding="stochastic"
    )
    ulp = 2**-7
    target = torch.full((64,), 1.0 + 2**-10)  # P[next bf16 up] = 1/8
    opt.set_master(p, target)
    total, flips, steps = torch.zeros(64, dtype=torch.float64), 0, 400
    previous = p.detach().clone()
    for _ in range(steps):
        p.grad = torch.zeros_like(p)  # a zero update: the master stays put, the served value is re-drawn
        opt.step()
        total += p.detach().double()
        flips += int((p.detach() != previous).sum())
        previous = p.detach().clone()
    assert torch.equal(opt.master_of(p), target)
    sigma = math.sqrt(1 / 8 * 7 / 8 / steps)  # binomial std of the time-average, in ulps
    assert float(((total / steps - target.double()) / ulp).abs().max()) < 5 * sigma
    assert flips > 0


def test_stochastic_noise_is_fresh_at_every_step_without_masters():
    """Same for the wrapper's own cast old + Δ' (transform_update) on a plain bf16 parameter."""
    target = 1.0 + 2**-10

    class Pin(UpdateRule):
        name = "pin"

        def tracks(self, name, param):
            return True

        def transform_update(self, ctx, name, param, old, delta):
            return torch.full_like(delta, target) - old.float()

    p = nn.Parameter(torch.ones(64, dtype=torch.bfloat16))
    opt = quiet_hooked(torch.optim.SGD([p], lr=0.0), [("w", p)], [Pin()], write_rounding="stochastic")
    total, steps = torch.zeros(64, dtype=torch.float64), 400
    for _ in range(steps):
        p.grad = torch.zeros_like(p)
        opt.step()
        total += p.detach().double()
    sigma = math.sqrt(1 / 8 * 7 / 8 / steps)
    assert float(((total / steps - target) / 2**-7).abs().max()) < 5 * sigma


def test_stochastic_noise_is_independent_across_parameters():
    a = nn.Parameter(torch.ones(256, dtype=torch.bfloat16))
    b = nn.Parameter(torch.ones(256, dtype=torch.bfloat16))
    opt = HookedOptimizer(
        torch.optim.SGD([a, b], lr=0.1),
        [("a", a), ("b", b)],
        master_weights=True,
        write_rounding="stochastic",
        rounding_seed=5,
    )
    master = torch.full((256,), 1.0 + 2**-8)  # exactly between two bf16 values: P = 1/2
    for p in (a, b):
        opt.set_master(p, master)
        p.grad = torch.zeros_like(p)
    opt.step()
    assert not torch.equal(a.detach(), b.detach())
    for index, p in enumerate((a, b)):  # nominal step 1, flat parameter index
        expected = stochastic_round(master, torch.bfloat16, rounding_generator(5, 1, index))
        assert torch.equal(p.detach(), expected)


def test_master_write_is_in_place_and_aliases_stay_valid():
    model = make_model()
    opt = HookedOptimizer(
        torch.optim.AdamW(model.parameters(), lr=1e-2), model.named_parameters(), master_weights=True
    )
    w = model[0].weight
    alias, ptr, before = w.detach(), w.data_ptr(), w.detach().clone()
    run_steps(model, opt, [0])
    assert not torch.equal(w.detach(), before)
    assert w.data_ptr() == ptr and alias.data_ptr() == ptr and torch.equal(alias, w.detach())


def test_stochastic_rounding_is_reproducible_and_seed_dependent():
    def run(seed):
        model = make_model()
        opt = HookedOptimizer(
            torch.optim.AdamW(model.parameters(), lr=1e-3),
            model.named_parameters(),
            master_weights=True,
            write_rounding="stochastic",
            rounding_seed=seed,
        )
        for step in range(3):
            opt.zero_grad()
            backward(model, step)
            opt.step()
        return [p.detach().clone() for p in model.parameters()]

    a, b, c = run(0), run(0), run(1)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not all(torch.equal(x, y) for x, y in zip(a, c))


# ---------------------------------------------------------------------------- external writes
def test_external_write_resyncs_master_with_warning_and_set_master_is_exact():
    model = make_model()
    opt = HookedOptimizer(
        torch.optim.AdamW(model.parameters(), lr=1e-3), model.named_parameters(), master_weights=True
    )
    opt.zero_grad()
    backward(model, 0)
    opt.step()
    w = model[0].weight
    with torch.no_grad():
        w[0, 0] = 5.0  # an external write far from the master
    opt.zero_grad()
    backward(model, 1)
    with pytest.warns(UserWarning, match="modified outside the optimizer"):
        opt.step()
    assert opt.master_resyncs == 1
    value = torch.full_like(w, 0.3, dtype=torch.float32)
    opt.set_master(w, value)
    assert torch.equal(opt.master_of(w), value) and torch.equal(w.detach(), value.bfloat16())
    opt.resync_masters()
    assert torch.equal(opt.master_of(w), w.detach().float())


def _mlp_master(rounding: str, lr: float = 1e-4):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 64), nn.Tanh(), nn.Linear(64, 8)).to(torch.bfloat16)
    opt = HookedOptimizer(
        torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0),
        model.named_parameters(),
        master_weights=True,
        write_rounding=rounding,
    )
    g = torch.Generator().manual_seed(1)

    def step():
        x = torch.randn(16, 32, generator=g).bfloat16()
        opt.zero_grad()
        model(x).float().pow(2).mean().backward()
        opt.step()

    return model, opt, step


def zero_update_step(model: nn.Module, opt: HookedOptimizer) -> None:
    """A step that writes every master parameter but changes no master (lr 0, zero gradients)."""
    lrs = [g["lr"] for g in opt.param_groups]
    for g in opt.param_groups:
        g["lr"] = 0.0
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    for g, lr in zip(opt.param_groups, lrs):
        g["lr"] = lr


@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_rollback_to_an_earlier_snapshot_reaches_the_masters(rounding):
    """model.load_state_dict(snapshot) after sub-ulp steps: every rewritten element's master follows."""
    model, opt, step = _mlp_master(rounding)
    for _ in range(3):
        step()
    snap = {k: v.clone() for k, v in model.state_dict().items()}
    for _ in range(5):
        step()
    params = dict(model.named_parameters())
    changed = {n: params[n].detach() != snap[n] for n in snap}
    n_changed = sum(int(c.sum()) for c in changed.values())
    assert n_changed > 100  # not vacuous: the 5 steps moved many served weights by an ulp
    model.load_state_dict(snap)
    with pytest.warns(UserWarning, match="modified outside the optimizer"):
        opt.state_dict()  # reconciles like the next step would
    assert opt.master_resyncs == n_changed
    for n, p in params.items():
        m = opt.master_of(p)
        assert torch.equal(m[changed[n]], snap[n].float()[changed[n]]), n
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        zero_update_step(model, opt)
    if rounding == "nearest":
        # served == master.to(bf16) == the snapshot everywhere, also for elements that did not move
        assert all(torch.equal(p.detach(), snap[n]) for n, p in params.items())
    else:
        assert all(torch.equal(p.detach()[changed[n]], snap[n][changed[n]]) for n, p in params.items())
    model.load_state_dict(snap)
    opt.resync_masters()  # the documented exact rollback: masters = the restored served values
    zero_update_step(model, opt)
    for n, p in params.items():
        assert torch.equal(p.detach(), snap[n]) and torch.equal(opt.master_of(p), snap[n].float()), n


def test_rollback_after_a_sub_ulp_step_is_not_undone_by_the_next_write():
    w0 = (1 + torch.rand(4096, generator=torch.Generator().manual_seed(0))).to(torch.bfloat16)
    p = nn.Parameter(w0.clone())
    opt = HookedOptimizer(torch.optim.SGD([p], lr=1.0), [("w", p)], master_weights=True)
    ulp = 2**-7
    p.grad = torch.full_like(p, -0.7 * ulp)  # every master moves by 0.7 ulp; served rounds up one ulp
    opt.step()
    moved = p.detach() != w0
    assert int(moved.sum()) > 4000
    with torch.no_grad():
        p.copy_(w0)  # how restore_frozen / rollback_channels write
    p.grad = torch.zeros_like(p)
    with pytest.warns(UserWarning, match="modified outside the optimizer"):
        opt.step()
    assert opt.master_resyncs == int(moved.sum())
    assert torch.equal(p.detach(), w0)
    assert torch.equal(opt.master_of(p)[moved], w0.float()[moved])


def test_repo_write_tools_reach_the_masters_and_temporary_edits_keep_them(tokenizer):
    from research_rl.models.ffn_channels import (
        ChannelSelection,
        ablate_channels,
        channel_state,
        rollback_channels,
    )
    from research_rl.models.model_interface import snapshot
    from research_rl.models.tiny import build_tiny_llama
    from research_rl.optim import TaskVectorPrior

    model = build_tiny_llama(len(tokenizer), seed=0).to(torch.bfloat16)
    named = dict(model.named_parameters())
    reference = {k: v.clone() for k, v in snapshot(model).items()}
    channels_ref = channel_state(model)
    opt = HookedOptimizer(
        torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0),
        model.named_parameters(),
        master_weights=True,
    )
    ids = torch.randint(3, len(tokenizer), (4, 16), generator=torch.Generator().manual_seed(0))

    def train(steps):
        for _ in range(steps):
            opt.zero_grad()
            model(input_ids=ids, labels=ids).loss.backward()
            opt.step()

    def served_equal_reference(names) -> bool:
        return all(torch.equal(named[n].detach(), reference[n]) for n in names)

    def check_restored(write, names):
        train(3)
        assert not served_equal_reference(names)
        write()
        assert served_equal_reference(names)
        with pytest.warns(UserWarning, match="modified outside the optimizer"):
            zero_update_step(model, opt)  # the next write must not undo the restore
        assert served_equal_reference(names)

    everything = list(named)
    frozen_all = ParameterMask({n: torch.zeros(p.shape, dtype=torch.bool) for n, p in named.items()})
    check_restored(lambda: frozen_all.restore_frozen(model, reference), everything)
    ffn = [n for n in named if ".mlp." in n]
    check_restored(lambda: rollback_channels(model, channels_ref), ffn)
    prior = TaskVectorPrior(reference)
    prior.register("t", TaskVector({n: torch.zeros(reference[n].shape) for n in ffn}))
    check_restored(lambda: prior.apply_to(model, [0.0]), everything)

    # ablate_channels is a detected write too: the next step keeps the ablated slices at zero.
    train(2)
    ablation = ChannelSelection({0: [2], 1: [3]})
    ablate_channels(model, ablation)
    with pytest.warns(UserWarning, match="modified outside the optimizer"):
        zero_update_step(model, opt)
    for layer, chan in ((0, 2), (1, 3)):
        mlp = model.model.layers[layer].mlp
        for t in (mlp.gate_proj.weight[chan], mlp.up_proj.weight[chan], mlp.down_proj.weight[:, chan]):
            assert torch.equal(t.detach(), torch.zeros_like(t))
        assert torch.equal(opt.master_of(mlp.up_proj.weight)[chan], torch.zeros(mlp.up_proj.weight.shape[1]))

    # A temporary edit undone bit-for-bit before the next step leaves the masters untouched.
    train(2)
    masters = {n: opt.master_of(p).clone() for n, p in named.items()}
    assert any(not torch.equal(masters[n], p.detach().float()) for n, p in named.items())
    resyncs = opt.master_resyncs
    previous = ablate_channels(model, ChannelSelection({0: [1, 4], 1: [0, 7]}))
    rollback_channels(model, previous)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        opt.state_dict()
        zero_update_step(model, opt)
    assert opt.master_resyncs == resyncs
    assert all(torch.equal(opt.master_of(p), masters[n]) for n, p in named.items())


def test_external_write_detection_treats_nan_as_unchanged():
    model = make_model()
    opt = HookedOptimizer(
        torch.optim.SGD(model.parameters(), lr=0.0), model.named_parameters(), master_weights=True
    )
    w = model[0].weight
    with torch.no_grad():
        w[0, 0] = float("nan")
    opt.state_dict()
    assert opt.master_resyncs == 1 and torch.isnan(opt.master_of(w)[0, 0])
    with torch.no_grad():
        w[1, 1] = 3.0
    with pytest.warns(UserWarning, match="1 elements"):
        opt.state_dict()  # only the new write is counted, the NaN is not re-detected
    assert opt.master_resyncs == 2


def test_stochastic_checkpoint_loaded_with_nearest_rounding_detects_only_real_writes():
    """The loaded weights are stochastic roundings (!= master.to(bf16)): they must not count as external writes."""
    model, opt, step = _mlp_master("stochastic")
    for _ in range(4):
        step()
    sd, weights = copy.deepcopy(opt.state_dict()), copy.deepcopy(model.state_dict())
    masters = sd["research_rl"]["precision"]["masters"]
    w_index = 0  # flat index of model[0].weight
    assert not torch.equal(model[0].weight.detach(), masters[w_index].bfloat16())  # not vacuous

    class Watch(
        UpdateRule
    ):  # tracks every parameter, changes nothing: served bits of unchanged masters are kept
        name = "watch"

        def tracks(self, name, param):
            return True

    resumed, _, _ = _mlp_master("nearest")
    opt2 = HookedOptimizer(
        torch.optim.AdamW(resumed.parameters(), lr=1e-4, weight_decay=0.0),
        resumed.named_parameters(),
        [Watch()],
        master_weights=True,
    )
    resumed.load_state_dict(weights)
    with pytest.warns(UserWarning, match="write_rounding"):
        opt2.load_state_dict(sd)
    w = resumed[0].weight
    served = w.detach().clone()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        zero_update_step(resumed, opt2)  # masters unchanged -> the stochastic served bits are kept
    assert torch.equal(w.detach(), served) and opt2.master_resyncs == 0
    expected = masters[w_index].clone()
    for count, (i, j) in enumerate([(0, 0), (3, 5)], start=1):
        with torch.no_grad():
            w[i, j] = 7.0  # one real external write
        with pytest.warns(UserWarning, match="1 elements"):
            opt2.state_dict()
        assert opt2.master_resyncs == count
        expected[i, j] = 7.0
        assert torch.equal(opt2.master_of(w), expected)


def test_has_master_only_for_optimised_parameters():
    model = make_model()
    extra = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    opt = HookedOptimizer(
        torch.optim.SGD(model.parameters(), lr=0.1), model.named_parameters(), master_weights=True
    )
    assert opt.has_master(model[0].weight)
    assert not opt.has_master(extra)
    opt.set_master(extra, torch.full((4,), 1.0 + 2**-10))  # not optimised: a plain copy, no master kept
    assert opt.master_of(extra) is None and torch.equal(extra.detach(), torch.ones(4, dtype=torch.bfloat16))
    opt.add_param_group({"params": [extra]})
    assert opt.has_master(extra) and opt.master_of(extra) is None  # created at the next step
    extra.grad = torch.zeros_like(extra)
    opt.step()
    assert opt.master_of(extra) is not None


def test_closure_is_refused_in_master_mode():
    model = make_model()
    opt = HookedOptimizer(
        torch.optim.AdamW(model.parameters()), model.named_parameters(), master_weights=True
    )
    with pytest.raises(ValueError, match="closure"):
        opt.step(lambda: 0.0)
    assert opt.nominal_steps == 0


# ---------------------------------------------------------------------------- precision_check
def test_precision_check_warn_error_off():
    model = make_model()
    with pytest.warns(UserWarning, match=r"2\*\*-8.*master_weights=True") as rec:
        HookedOptimizer(torch.optim.AdamW(model.parameters()), model.named_parameters())
    assert sum("low precision" in str(r.message) for r in rec) == 1
    assert "4 trainable parameters" in str(rec[0].message) and "'0.weight'" in str(rec[0].message)
    with pytest.raises(ValueError, match="low precision"):
        HookedOptimizer(
            torch.optim.AdamW(model.parameters()), model.named_parameters(), precision_check="error"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        HookedOptimizer(
            torch.optim.AdamW(model.parameters()), model.named_parameters(), precision_check="off"
        )
        HookedOptimizer(torch.optim.AdamW(model.parameters()), model.named_parameters(), master_weights=True)
        fp32 = make_model(dtype=torch.float32)
        HookedOptimizer(
            torch.optim.AdamW(fp32.parameters()), fp32.named_parameters(), precision_check="error"
        )
        # A frozen (requires_grad=False) bf16 tensor in the optimizer is not a trainable parameter.
        frozen_bf16 = nn.Parameter(torch.ones(3, dtype=torch.bfloat16), requires_grad=False)
        HookedOptimizer(
            torch.optim.AdamW([*fp32.parameters(), frozen_bf16]),
            [*fp32.named_parameters(), ("frozen", frozen_bf16)],
            precision_check="error",
        )
    with pytest.raises(ValueError, match="precision_check"):
        HookedOptimizer(
            torch.optim.AdamW(model.parameters()), model.named_parameters(), precision_check="loud"
        )
    with pytest.raises(ValueError, match="write_rounding"):
        HookedOptimizer(torch.optim.AdamW(model.parameters()), model.named_parameters(), write_rounding="up")
    half = make_model(dtype=torch.float16)
    with pytest.raises(ValueError, match="only implemented for bfloat16"):
        HookedOptimizer(
            torch.optim.AdamW(half.parameters()),
            half.named_parameters(),
            master_weights=True,
            write_rounding="stochastic",
        )


def test_stochastic_without_masters_only_rounds_the_wrappers_own_cast():
    """Without masters, write_rounding applies to old + Δ' after a transform_update rule (and nothing else)."""

    class Shrink(UpdateRule):
        name = "shrink"

        def tracks(self, name, param):
            return True

        def transform_update(self, ctx, name, param, old, delta):
            return delta * 0.25

    def run(rounding, rules):
        net = Plate(shape=(32, 32))
        opt = quiet_hooked(
            torch.optim.AdamW(net.parameters(), lr=6e-3, weight_decay=0.0),
            net.named_parameters(),
            rules,
            write_rounding=rounding,
        )
        w0 = net.weight.detach().clone()
        net.loss(0).backward()
        opt.step()
        return w0, net.weight.detach().clone()

    w0, near = run("nearest", [Shrink()])
    _, sto = run("stochastic", [Shrink()])
    # AdamW's own nearest write moves w by one ulp (2**-7); a quarter of that is below half an ulp.
    assert torch.equal(near, w0)
    assert not torch.equal(sto, w0)
    assert float((w0.float() - sto.float()).mean()) == pytest.approx(0.25 * 2**-7, rel=0.3)
    # No transform_update rule: the inner optimizer's own (nearest) write is untouched.
    _, a = run("nearest", [])
    _, b = run("stochastic", [])
    assert torch.equal(a, b)


# ------------------------------------------------------------------------------------ P2 prior
def p2_setup(master: bool, *, lr_w: float = 0.0, allow_low_precision: bool = False):
    g = torch.Generator().manual_seed(0)
    net = nn.Linear(16, 12, bias=False)
    with torch.no_grad():
        net.weight.copy_(1.0 + torch.rand(12, 16, generator=g))
    net.bfloat16()
    tvs = [TaskVector({"weight": 0.5 * torch.randn(12, 16, generator=g)}) for _ in range(2)]
    rule = P2PriorRule(tvs, ["weight"], lr=1e-4, init=0.5, allow_low_precision=allow_low_precision)
    opt = HookedOptimizer(
        torch.optim.AdamW(net.parameters(), lr=lr_w, weight_decay=0.0),
        net.named_parameters(),
        [rule],
        master_weights=master,
        precision_check="off",
    )
    return net, tvs, rule, opt, g


def test_p2_refuses_bf16_without_master_weights():
    with pytest.raises(ValueError, match="master weights"):
        p2_setup(master=False)
    p2_setup(master=False, allow_low_precision=True)
    p2_setup(master=True)


def test_p2_corrections_vanish_in_bf16_and_accumulate_in_masters():
    plain, tvs, rule_p, opt_p, g = p2_setup(master=False, allow_low_precision=True)
    net, _, rule, opt, _ = p2_setup(master=True)
    w0 = net.weight.detach().clone()
    a0 = rule.coefficients.clone()
    for _ in range(20):
        grad = torch.randn(12, 16, generator=g).bfloat16()
        for n, o in ((plain, opt_p), (net, opt)):
            n.weight.grad = grad.clone()
            o.step()
    assert torch.equal(rule.coefficients, rule_p.coefficients)  # same coefficient channel
    da = rule.coefficients - a0
    assert float(da.abs().max()) > 2e-4  # α really moved
    expected = w0.float() + sum(da[i] * tvs[i]["weight"] for i in range(2))
    assert torch.equal(plain.weight, w0)  # the bf16 write rounded every correction away
    master = opt.master_of(net.weight)
    torch.testing.assert_close(master, expected, rtol=0, atol=2e-6)
    assert not torch.equal(master, w0.float())
    assert opt.master_resyncs == 0


def test_p2_materialize_writes_the_master_exactly():
    net, tvs, rule, opt, _ = p2_setup(master=True)
    base = {"weight": torch.ones(12, 16)}
    rule.materialize(net, base)
    expected = 1.0 + sum(rule.coefficients[i] * tvs[i]["weight"] for i in range(2))
    assert torch.equal(opt.master_of(net.weight), expected)
    assert torch.equal(net.weight.detach(), expected.bfloat16())


# --------------------------------------------------------------------------------- persistence
def run_steps(model, opt, steps):
    for step in steps:
        opt.zero_grad()
        backward(model, step)
        opt.step()


@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_state_dict_weights_only_roundtrip_is_bit_exact(rounding):
    def make(seed=0):
        m = make_model(seed)
        o = HookedOptimizer(
            torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1),
            m.named_parameters(),
            [MaskRule(first_layer_mask(m))],
            master_weights=True,
            write_rounding=rounding,
        )
        return m, o

    m_full, o_full = make()
    run_steps(m_full, o_full, range(6))

    m_a, o_a = make()
    run_steps(m_a, o_a, range(3))
    sd = o_a.state_dict()
    masters = sd["research_rl"]["precision"]["masters"]
    assert sorted(masters) == [0, 1, 2, 3]
    assert any(not torch.equal(masters[i], p.detach().float()) for i, p in enumerate(m_a.parameters()))
    buf, mbuf = io.BytesIO(), io.BytesIO()
    torch.save(sd, buf)
    torch.save(m_a.state_dict(), mbuf)
    buf.seek(0)
    mbuf.seek(0)

    m_b, o_b = make(seed=123)
    m_b.load_state_dict(torch.load(mbuf, weights_only=True))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        o_b.load_state_dict(torch.load(buf, weights_only=True))
    for p in m_b.parameters():
        assert all(v.dtype == torch.float32 for k, v in o_b.inner.state[p].items() if k != "step")
    for p, q in zip(m_a.parameters(), m_b.parameters()):
        assert torch.equal(o_a.master_of(p), o_b.master_of(q))
        for k in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(o_a.inner.state[p][k], o_b.inner.state[q][k])
    run_steps(m_b, o_b, range(3, 6))
    for p, q in zip(m_full.parameters(), m_b.parameters()):
        assert torch.equal(p, q)
        assert torch.equal(o_full.master_of(p), o_b.master_of(q))
    assert o_b.master_resyncs == 0


def test_accelerated_optimizer_prepare_roundtrip_keeps_fp32_state():
    from accelerate import Accelerator
    from accelerate.optimizer import AcceleratedOptimizer

    Accelerator(cpu=True)
    model = make_model()
    opt = HookedOptimizer(
        torch.optim.AdamW(model.parameters(), lr=1e-3), model.named_parameters(), master_weights=True
    )
    run_steps(model, opt, range(3))
    moments = {id(p): {k: v.clone() for k, v in opt.inner.state[p].items()} for p in model.parameters()}
    masters = {id(p): opt.master_of(p) for p in model.parameters()}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wrapped = AcceleratedOptimizer(opt, device_placement=True)  # state_dict -> move -> load_state_dict
    for p in model.parameters():
        assert opt.master_of(p) is masters[id(p)]  # same objects handed back: kept, not re-created
        for k, v in opt.inner.state[p].items():
            assert v.dtype == moments[id(p)][k].dtype and torch.equal(v, moments[id(p)][k])
        assert opt.inner.state[p]["exp_avg"].dtype == torch.float32
    model_ref = copy.deepcopy(model)
    opt_ref = HookedOptimizer(
        torch.optim.AdamW(model_ref.parameters(), lr=1e-3), model_ref.named_parameters(), master_weights=True
    )
    opt_ref.load_state_dict(copy.deepcopy(opt.state_dict()))
    wrapped.zero_grad()
    backward(model, 3)
    wrapped.step()
    run_steps(model_ref, opt_ref, [3])
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), model_ref.parameters()))


def test_loading_without_masters_reinitialises_and_masters_into_plain_optimizer_warn():
    model = make_model()
    plain = quiet_hooked(torch.optim.AdamW(model.parameters(), lr=1e-3), model.named_parameters())
    run_steps(model, plain, range(2))
    sd = plain.state_dict()
    assert sd["research_rl"]["precision"]["masters"] == {}
    target = HookedOptimizer(
        torch.optim.AdamW(model.parameters(), lr=1e-3), model.named_parameters(), master_weights=True
    )
    with pytest.warns(UserWarning, match="no fp32 master weights for 4"):
        target.load_state_dict(sd)
    for p in model.parameters():
        assert torch.equal(target.master_of(p), p.detach().float())
        assert target.inner.state[p]["exp_avg"].dtype == torch.float32  # upcast, not left in bf16
    with pytest.warns(UserWarning, match="dropped"):
        quiet_hooked(
            torch.optim.AdamW(model.parameters(), lr=1e-3), model.named_parameters()
        ).load_state_dict(target.state_dict())
    other = HookedOptimizer(
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        model.named_parameters(),
        master_weights=True,
        write_rounding="stochastic",
    )
    with pytest.warns(UserWarning, match="write_rounding"):
        other.load_state_dict(target.state_dict())
    reseeded = HookedOptimizer(
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        model.named_parameters(),
        master_weights=True,
        rounding_seed=1,
    )
    with pytest.warns(UserWarning, match="rounding_seed=0") as rec:
        reseeded.load_state_dict(target.state_dict())
    assert not any("write_rounding" in str(r.message) for r in rec)


@pytest.mark.parametrize("master", [True, False])
def test_masters_saved_for_parameters_that_are_now_fp32_warn_and_are_never_applied(master):
    model = make_model()
    opt = HookedOptimizer(
        torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0),
        model.named_parameters(),
        master_weights=True,
    )
    run_steps(model, opt, range(2))
    buf = io.BytesIO()
    torch.save(opt.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=True)
    masters = sd["research_rl"]["precision"]["masters"]
    fp32 = copy.deepcopy(model).float()  # the served bf16 weights, upcast: the sub-ulp progress is missing
    before = [p.detach().clone() for p in fp32.parameters()]
    resumed = HookedOptimizer(
        torch.optim.AdamW(fp32.parameters(), lr=1e-4, weight_decay=0.0),
        fp32.named_parameters(),
        master_weights=master,
    )
    with pytest.warns(UserWarning, match="now full precision") as rec:
        resumed.load_state_dict(copy.deepcopy(sd))
    assert "4 parameters" in str(next(r.message for r in rec if "now full precision" in str(r.message)))
    assert all(
        torch.equal(p, q) for p, q in zip(fp32.parameters(), before)
    )  # an optimizer load never writes weights
    # The documented remedy: copy the masters in first; then nothing is lost and nothing warns.
    with torch.no_grad():
        for i, p in enumerate(fp32.parameters()):
            p.copy_(masters[i])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        HookedOptimizer(
            torch.optim.AdamW(fp32.parameters(), lr=1e-4, weight_decay=0.0),
            fp32.named_parameters(),
            master_weights=master,
        ).load_state_dict(copy.deepcopy(sd))


def test_existing_low_precision_inner_state_is_upcast():
    model = make_model()
    inner = torch.optim.AdamW(model.parameters(), lr=1e-3)
    run_steps(model, inner, range(2))
    assert inner.state[model[0].weight]["exp_avg"].dtype == torch.bfloat16
    opt = HookedOptimizer(inner, model.named_parameters(), master_weights=True)
    assert inner.state[model[0].weight]["exp_avg"].dtype == torch.float32
    run_steps(model, opt, [2])


# ----------------------------------------------------------------------------------- config
def test_optim_spec_precision_fields_and_back_compat():
    raw = ExperimentConfig(
        study_id="s", model=ModelSpec("m"), data=DataSpec("d"), trainer=TrainerSpec("sft")
    ).to_dict()
    for key in ("master_weights", "write_rounding", "precision_check"):
        del raw["optim"][key]
    cfg = ExperimentConfig.from_dict(raw)  # configs written before these fields still load
    assert (cfg.optim.master_weights, cfg.optim.write_rounding, cfg.optim.precision_check) == (
        False,
        "nearest",
        "warn",
    )
    assert cfg.config_version == "1.0"
    spec = OptimSpec(master_weights=True, write_rounding="stochastic", precision_check="error")
    raw["optim"] = {**raw["optim"], "master_weights": True, "write_rounding": "stochastic"}
    assert ExperimentConfig.from_dict(raw).optim.write_rounding == spec.write_rounding
    with pytest.raises(ConfigError, match="write_rounding"):
        OptimSpec(write_rounding="up")
    with pytest.raises(ConfigError, match="precision_check"):
        OptimSpec(precision_check="never")
    with pytest.raises(ConfigError, match="hooked"):
        OptimSpec(hooked=False, master_weights=True)
    with pytest.raises(ConfigError, match="hooked"):
        OptimSpec(hooked=False, write_rounding="stochastic")
    with pytest.raises(ConfigError, match="master_weights must be a bool"):
        OptimSpec(master_weights="yes")  # type: ignore[arg-type]


# ------------------------------------------------------------------------------- TRL / HF Trainer
def _sft_trainer(tokenizer, out_dir, *, max_steps, rounding, save_steps=None, model_seed=0, **precision):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    from research_rl.models.tiny import build_tiny_llama
    from research_rl.trainers.base import ResearchCallback, research_trainer_class

    model = build_tiny_llama(len(tokenizer), seed=model_seed).to(torch.bfloat16)
    rows = [{"prompt": f"{a}+{b}=", "completion": str(a + b)} for a in range(4) for b in range(4)]
    args = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        max_steps=max_steps,
        learning_rate=1e-4,  # Adam steps ~1e-4: mostly below half a bf16 ulp
        weight_decay=0.0,
        lr_scheduler_type="constant",
        logging_steps=1,
        report_to=[],
        use_cpu=True,
        bf16=False,  # the parameters themselves are bf16; no autocast
        seed=7,
        save_strategy="steps" if save_steps else "no",
        save_steps=save_steps or 500,
        save_only_model=False,
    )
    callback = ResearchCallback()
    trainer = research_trainer_class(SFTTrainer)(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        callbacks=[callback],
        **{"master_weights": True, "write_rounding": rounding, "precision_check": "error", **precision},
    )
    callback.bind(trainer)
    return trainer


@pytest.mark.integration
@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_trl_sft_bf16_master_weights_resume_is_bit_identical(tokenizer, tmp_path, rounding):
    from research_rl.models.model_interface import snapshot

    full = _sft_trainer(tokenizer, tmp_path / "full", max_steps=4, rounding=rounding)
    full.train()
    part = _sft_trainer(tokenizer, tmp_path / "part", max_steps=4, rounding=rounding, save_steps=2)
    part.train()
    ckpt = tmp_path / "part" / "checkpoint-2"
    saved = torch.load(ckpt / "optimizer.pt", weights_only=True)
    masters = saved["research_rl"]["precision"]["masters"]
    assert saved["research_rl"]["precision"]["master_weights"] is True and masters
    assert all(v.dtype == torch.float32 for v in masters.values())
    assert all(v["exp_avg"].dtype == torch.float32 for v in saved["state"].values())

    # Different random init: only a genuine resume (weights + masters + fp32 moments) reproduces the run.
    resumed = _sft_trainer(
        tokenizer, tmp_path / "part", max_steps=4, rounding=rounding, save_steps=2, model_seed=99
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resumed.train(resume_from_checkpoint=str(ckpt))
    assert not [w for w in caught if "HookedOptimizer" in str(w.message)]
    opt_full, opt_res = full.hooked_optimizer, resumed.hooked_optimizer
    assert opt_res is not None and opt_res.applied_updates == 4 and opt_res.master_resyncs == 0
    assert opt_full is not None and opt_full.master_resyncs == 0
    full_params, res_params = snapshot(full.model), snapshot(resumed.model)
    named_full, named_res = dict(full.model.named_parameters()), dict(resumed.model.named_parameters())
    not_vacuous = False
    for name, tensor in full_params.items():
        assert tensor.dtype == torch.bfloat16
        assert torch.equal(tensor, res_params[name]), name
        m_full, m_res = opt_full.master_of(named_full[name]), opt_res.master_of(named_res[name])
        assert m_full is not None and m_res is not None and torch.equal(m_full, m_res), name
        not_vacuous |= not torch.equal(m_full, tensor.float())
        for k in ("exp_avg", "exp_avg_sq"):
            a, b = opt_full.inner.state[named_full[name]][k], opt_res.inner.state[named_res[name]][k]
            assert a.dtype == torch.float32 and torch.equal(a, b), (name, k)
    assert not_vacuous  # the masters carry information the bf16 weights do not


@pytest.mark.integration
def test_second_train_call_reuses_the_hooked_optimizer_and_its_masters(tokenizer, tmp_path):
    from accelerate.optimizer import AcceleratedOptimizer

    from research_rl.optim import unwrap_hooked

    trainer = _sft_trainer(tokenizer, tmp_path / "twice", max_steps=2, rounding="nearest")
    trainer.train()
    hooked = trainer.hooked_optimizer
    assert hooked is not None and hooked.applied_updates == 2
    named = dict(trainer.model.named_parameters())
    masters = {n: hooked.master_of(p) for n, p in named.items()}
    assert all(m is not None for m in masters.values())
    assert any(not torch.equal(m, named[n].detach().float()) for n, m in masters.items())  # sub-ulp progress
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer.train()
    assert not [w for w in caught if "HookedOptimizer" in str(w.message)]
    assert unwrap_hooked(trainer.optimizer) is hooked and hooked.applied_updates == 4
    assert isinstance(trainer.optimizer, AcceleratedOptimizer) and trainer.optimizer.optimizer is hooked
    assert not isinstance(hooked.inner, HookedOptimizer)  # wrapped once, not twice
    assert hooked.master_resyncs == 0 and all(hooked.master_of(p) is not None for p in named.values())


@pytest.mark.integration
@pytest.mark.parametrize(
    "precision",
    [
        {"master_weights": True},
        {"master_weights": False, "write_rounding": "stochastic", "precision_check": "off"},
    ],
    ids=["master_weights", "stochastic"],
)
def test_precision_options_imply_the_hooked_optimizer(tokenizer, tmp_path, precision):
    trainer = _sft_trainer(
        tokenizer, tmp_path / "implied", max_steps=1, rounding="nearest", hooked_optimizer=False, **precision
    )
    trainer.create_optimizer()
    hooked = trainer.hooked_optimizer
    assert hooked is not None
    assert hooked.master_weights == precision["master_weights"]
    assert hooked.write_rounding == precision.get("write_rounding", "nearest")


def test_precision_info_counts_trainable_parameters_only():
    from research_rl.trainers.base import precision_info

    model = make_model()
    model[2].float()
    model[2].bias.requires_grad_(False)
    assert precision_info(model, OptimSpec(master_weights=True)) == {
        "param_dtypes": {"bfloat16": 2, "float32": 1},
        "master_weights": True,
        "write_rounding": "nearest",
    }


@pytest.mark.integration
def test_build_passes_precision_options_and_records_them(tokenizer, tmp_path):
    import json

    from datasets import Dataset

    import research_rl.trainers.sft_adapter  # noqa: F401  (registers the adapter)
    from research_rl.models.tiny import build_tiny_llama
    from research_rl.trainers import build_run

    exp = ExperimentConfig(
        study_id="precision",
        model=ModelSpec(name_or_path="tiny-llama-offline", dtype="bfloat16"),
        data=DataSpec(name="toy"),
        trainer=TrainerSpec(algorithm="sft", args={"max_steps": 2, "per_device_train_batch_size": 2}),
        optim=OptimSpec(master_weights=True, write_rounding="stochastic", precision_check="error"),
        output_dir=str(tmp_path),
        run_name="r",
        seed=3,
    )
    model = build_tiny_llama(len(tokenizer), seed=0).to(torch.bfloat16)
    rows = [{"prompt": "1+1=", "completion": "2"}, {"prompt": "2+3=", "completion": "5"}]
    run = build_run(exp, train_dataset=Dataset.from_list(rows), model=model, processing_class=tokenizer)
    record = run.train()
    opt = run.hooked_optimizer
    assert opt is not None and opt.master_weights and opt.write_rounding == "stochastic"
    assert opt.rounding_seed == 3 and opt.precision_check == "error" and opt.applied_updates == 2
    n_params = sum(1 for p in model.parameters() if p.requires_grad)
    expected = {
        "param_dtypes": {"bfloat16": n_params},
        "master_weights": True,
        "write_rounding": "stochastic",
    }
    assert record["precision"] == expected
    assert json.loads((exp.run_dir / "record.json").read_text())["precision"] == expected

    exp_err = ExperimentConfig.from_dict(
        {**exp.to_dict(), "run_name": "e", "optim": {"precision_check": "error"}}
    )
    with pytest.raises(ValueError, match="low precision"):
        build_run(
            exp_err,
            train_dataset=Dataset.from_list(rows),
            model=build_tiny_llama(len(tokenizer), seed=0).to(torch.bfloat16),
            processing_class=tokenizer,
        ).train()
