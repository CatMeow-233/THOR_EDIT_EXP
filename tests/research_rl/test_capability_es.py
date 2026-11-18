"""Tests of research_rl.capability.es: the forward-only ES update operator.

CPU only; tiny Llama / small ``nn.Linear`` built in-process. Bit-exactness is asserted only for
restores, copies and same-seed random draws; updated weights are compared with tolerances and the
functional tests assert improvement, never exact trained values.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

from research_rl.capability.es import ESConfig, ESError, ESOperator, FitnessResult, shape_fitness
from research_rl.capability.spaces import ChannelGainSpace, FullParameterSpace
from research_rl.capability.types import GateDecision, OperatorReport, ParameterSpace, UpdateOperator
from research_rl.models.ffn_channels import ChannelSelection, verify_modification_scope
from research_rl.models.model_interface import fingerprint, snapshot
from research_rl.models.tiny import build_char_tokenizer, build_tiny_llama

VOCAB = len(build_char_tokenizer())


# ------------------------------------------------------------------ helpers
def _linear(seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(4, 3)


def _flat(m: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()]).clone()


def _quadratic(target: torch.Tensor) -> Callable[[nn.Module, int, int], FitnessResult]:
    def fitness(model: nn.Module, update_index: int, member_index: int) -> FitnessResult:
        f = -float(((_flat(model) - target) ** 2).sum())
        return FitnessResult(f, f, (f > -0.5,), (f > -1.0,), (3,))

    return fitness


def _op(
    model: nn.Module,
    fitness: Callable[[nn.Module, int, int], FitnessResult] | None = None,
    **cfg: Any,
) -> ESOperator:
    space = FullParameterSpace(model, [n for n, _ in model.named_parameters()])
    target = torch.zeros(space.dim)
    return ESOperator(
        space, fitness or _quadratic(target), ESConfig(**{"population": 6, "sigma": 0.1, "lr": 0.05, **cfg})
    )


class SpySpace:
    """Delegating ParameterSpace that records every call."""

    def __init__(self, inner: ParameterSpace, model: nn.Module) -> None:
        self.inner = inner
        self.dim = inner.dim
        self.calls: list[tuple[str, torch.Tensor | None]] = []
        self.after_restore: list[str] = []  # model fingerprint right after every write(None)
        self._model = model

    def parameter_names(self) -> list[str]:
        return self.inner.parameter_names()

    def write(self, vector: torch.Tensor | None) -> None:
        self.calls.append(("write", None if vector is None else vector.detach().clone()))
        self.inner.write(vector)
        if vector is None:
            self.after_restore.append(fingerprint(self._model))

    def rebase(self) -> None:
        self.calls.append(("rebase", None))
        self.inner.rebase()


# ------------------------------------------------------------------ protocol
def test_implements_update_operator() -> None:
    lin = _linear()
    op: UpdateOperator = _op(lin)  # static conformance (mypy) ...
    assert isinstance(op, UpdateOperator)  # ... and runtime conformance
    assert op.optimizer is None
    assert op.name == "es"
    assert op.parameter_names() == ["bias", "weight"]


def test_config_validation_and_roundtrip() -> None:
    cfg = ESConfig(population=4, sigma=0.1, lr=0.2, fitness_shaping="rank", max_step_norm=1.0, seed=7)
    assert ESConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    for bad in (
        {"population": 3},
        {"population": 1, "antithetic": False},
        {"population": True},
        {"sigma": 0.0},
        {"lr": -1.0},
        {"sigma": math.nan},
        {"fitness_shaping": "centered"},
        {"max_step_norm": 0.0},
        {"seed": 1.5},
    ):
        with pytest.raises(ESError):
            ESConfig(**bad)
    ESConfig(population=3, antithetic=False)  # odd is fine without antithetic pairs


def test_fitness_result_validation() -> None:
    r = FitnessResult(1, None, [True, False], [1, 0], [2, 3], {"extra_forward_passes": 2})  # type: ignore[arg-type]
    assert r.v1_pass == (True, False) and r.v2_pass == (True, False) and r.lengths == (2, 3)
    assert r.reward_v1 == 1.0 and isinstance(r.reward_v1, float)
    assert r.forward_passes == 3
    with pytest.raises(ESError):
        FitnessResult(0.0, 0.0, (True,), (True, False))
    with pytest.raises(ESError):
        FitnessResult(0.0, 0.0, (True,), (True,), (1, 2))
    with pytest.raises(ESError):
        FitnessResult(0.0, 0.0, (2,), (True,))  # type: ignore[arg-type]
    with pytest.raises(ESError):
        FitnessResult(0.0, 0.0, (), (), extra={"extra_forward_passes": -1})
    with pytest.raises(ESError):
        FitnessResult("x", 0.0, (), ())  # type: ignore[arg-type]


def test_shape_fitness() -> None:
    assert shape_fitness([1.0, 1.0, 1.0], "zscore") == [0.0, 0.0, 0.0]
    z = shape_fitness([1.0, 2.0, 3.0, 6.0], "zscore")
    assert abs(sum(z)) < 1e-12 and abs(sum(x * x for x in z) / 4 - 1.0) < 1e-12
    assert shape_fitness([5.0, -1.0, 2.0], "rank") == [0.5, -0.5, 0.0]
    assert shape_fitness([0.0, 1.0, 1.0, 2.0], "rank") == [-0.5, 0.0, 0.0, 0.5]  # ties averaged
    assert shape_fitness([3.0, 3.0], "rank") == [0.0, 0.0]


# ------------------------------------------------------------------ perturbations
def test_perturbations_reproducible_and_antithetic() -> None:
    a, b = _op(_linear()), _op(_linear(1))
    for t in (0, 3):
        for i in range(6):
            assert torch.equal(a.perturbation(t, i), b.perturbation(t, i))
    for j in range(3):
        assert torch.equal(a.perturbation(2, 2 * j + 1), -a.perturbation(2, 2 * j))
        assert a.noise_index(2 * j) == (j, 1.0) and a.noise_index(2 * j + 1) == (j, -1.0)
    assert not torch.equal(a.perturbation(0, 0), a.perturbation(1, 0))
    assert not torch.equal(a.perturbation(0, 0), a.perturbation(0, 2))
    other = _op(_linear(), seed=1)
    assert not torch.equal(a.perturbation(0, 0), other.perturbation(0, 0))
    plain = _op(_linear(), antithetic=False, population=3)
    assert plain.noise_index(1) == (1, 1.0)
    assert not torch.equal(plain.perturbation(0, 1), -plain.perturbation(0, 0))
    assert a.perturbation(0, 0).dtype == torch.float32 and a.perturbation(0, 0).shape == (a.dim,)
    with pytest.raises(ESError):
        a.perturbation(0, 6)
    with pytest.raises(ESError):
        a.perturbation(-1, 0)


# ------------------------------------------------------------------ member evaluation
def test_call_sequence_and_base_restored_after_every_member() -> None:
    lin = _linear()
    inner = FullParameterSpace(lin, ["weight", "bias"])
    spy = SpySpace(inner, lin)
    base = _flat(lin)
    base_fp = fingerprint(lin)
    seen: list[tuple[int, int]] = []
    cfg = ESConfig(population=4, sigma=0.1, lr=0.05, seed=5)
    op_ref = ESOperator(FullParameterSpace(_linear(), ["weight", "bias"]), _quadratic(torch.zeros(15)), cfg)

    def fitness(model: nn.Module, t: int, i: int) -> FitnessResult:
        assert model is lin
        assert not torch.is_grad_enabled()
        seen.append((t, i))
        # the model is the base plus exactly this member's perturbation
        torch.testing.assert_close(_flat(model), base + cfg.sigma * op_ref.perturbation(t, i))
        return FitnessResult(float(i), None, (), ())

    op = ESOperator(spy, fitness, cfg)
    op.step(lin, 7)
    assert spy.after_restore == [base_fp] * 5  # bitwise base after every member (+ final restore)
    assert seen == [(7, i) for i in range(4)]
    kinds = ["rebase" if k == "rebase" else ("write_none" if v is None else "write_v") for k, v in spy.calls]
    assert kinds == ["rebase", *(["write_v", "write_none"] * 4), "write_none", "write_v", "rebase"]
    for i in range(4):
        _, v = spy.calls[1 + 2 * i]
        assert v is not None
        assert torch.equal(v, cfg.sigma * op.perturbation(7, i))


def test_fitness_exception_restores_base() -> None:
    lin = _linear()
    op = _op(lin)
    before = fingerprint(lin)

    def boom(model: nn.Module, t: int, i: int) -> FitnessResult:
        if i == 3:
            raise RuntimeError("fitness failed")
        return FitnessResult(0.0, 0.0, (), ())

    op.fitness = boom
    with pytest.raises(RuntimeError, match="fitness failed"):
        op.step(lin, 0)
    assert fingerprint(lin) == before
    assert op.steps_done == 0


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_non_finite_fitness_raises_after_restore(value: float) -> None:
    lin = _linear()
    op = _op(lin, fitness=lambda m, t, i: FitnessResult(value if i == 1 else 0.0, 0.0, (), ()))
    before = fingerprint(lin)
    with pytest.raises(ESError, match="non-finite"):
        op.step(lin, 0)
    assert fingerprint(lin) == before
    op2 = _op(lin, fitness=lambda m, t, i: FitnessResult(0.0, value, (), ()))
    with pytest.raises(ESError, match="non-finite"):
        op2.step(lin, 0)
    assert fingerprint(lin) == before


def test_wrong_return_type_and_wrong_model() -> None:
    lin = _linear()
    op = _op(lin, fitness=lambda m, t, i: 1.0)  # type: ignore[arg-type,return-value]
    before = fingerprint(lin)
    with pytest.raises(ESError, match="FitnessResult"):
        op.step(lin, 0)
    assert fingerprint(lin) == before
    with pytest.raises(ESError, match="not the model"):
        _op(lin).step(_linear(), 0)
    with pytest.raises(ESError):
        _op(lin).step(lin, -1)


# ------------------------------------------------------------------ the update
def _independent_shaping(fits: list[float], method: str) -> list[float]:
    """The documented shaping formulas, recomputed here (no ties in the tests that use it)."""
    n = len(fits)
    if method == "zscore":
        mean = sum(fits) / n
        std = math.sqrt(sum((f - mean) ** 2 for f in fits) / n)
        return [(f - mean) / std for f in fits]
    assert len(set(fits)) == n
    order = sorted(fits)
    return [order.index(f) / (n - 1) - 0.5 for f in fits]


@pytest.mark.parametrize("shaping", ["zscore", "rank"])
def test_update_equals_documented_formula(shaping: str) -> None:
    """step() applies the *configured* shaping (a step that always used zscore fails the rank case:
    the shaped values and the step size differ)."""
    lin = _linear()
    target = torch.zeros(15)
    op = _op(lin, fitness=_quadratic(target), fitness_shaping=shaping)
    base = _flat(lin).double()
    fits = []
    for i in range(op.config.population):
        x = base + op.config.sigma * op.perturbation(4, i).double()
        fits.append(-float(((x - target.double()) ** 2).sum()))
    shaped = _independent_shaping(fits, shaping)
    assert shaped == pytest.approx(shape_fitness(fits, shaping), abs=1e-12)  # type: ignore[arg-type]
    d = sum(s * op.perturbation(4, i).double() for i, s in enumerate(shaped))
    expected = base + op.config.lr * d / (op.config.population * op.config.sigma)
    report = op.step(lin, 4)
    torch.testing.assert_close(_flat(lin).double(), expected, atol=1e-6, rtol=1e-5)
    assert report.extra["shaped"] == pytest.approx(shaped, abs=1e-5)
    # the accepted direction became the base
    after = fingerprint(lin)
    op.space.write(None)
    assert fingerprint(lin) == after


def test_report_contents() -> None:
    lin = _linear()

    def fitness(model: nn.Module, t: int, i: int) -> FitnessResult:
        k = i + 1  # member i has i + 1 rollouts
        return FitnessResult(
            float(i),
            0.5,
            tuple(j % 2 == 0 for j in range(k)),
            tuple(j == 0 for j in range(k)),
            tuple(10 * i + j for j in range(k)),
            {"extra_forward_passes": i},
        )

    op = _op(lin, fitness=fitness, population=4)
    report = op.step(lin, 2)
    assert isinstance(report, OperatorReport)
    assert report.update_index == 2
    assert report.backward_passes == 0
    assert report.forward_passes == 4 + (0 + 1 + 2 + 3)
    assert report.v1_pass == tuple(j % 2 == 0 for i in range(4) for j in range(i + 1))
    assert report.v2_pass == tuple(j == 0 for i in range(4) for j in range(i + 1))
    assert report.completion_lengths == tuple(10 * i + j for i in range(4) for j in range(i + 1))
    assert report.reward_v1 == pytest.approx(1.5)
    assert report.reward_v2 == pytest.approx(0.5)
    assert report.screened == 0
    assert report.extra["rollouts_per_member"] == [1, 2, 3, 4]
    json.dumps(report.extra)
    assert op.forward_passes_total == report.forward_passes and op.steps_done == 1

    # reward_v2 None if any member has none; lengths dropped unless every member has them
    def partial(model: nn.Module, t: int, i: int) -> FitnessResult:
        return FitnessResult(float(i), None if i == 0 else 1.0, (True,), (True,), (5,) if i else ())

    r2 = _op(lin, fitness=partial, population=4).step(lin, 0)
    assert r2.reward_v2 is None and r2.completion_lengths == () and len(r2.v1_pass) == 4


def test_constant_fitness_is_a_zero_step() -> None:
    lin = _linear()
    before = fingerprint(lin)
    report = _op(lin, fitness=lambda m, t, i: FitnessResult(1.0, 1.0, (True,), (True,))).step(lin, 0)
    assert fingerprint(lin) == before
    assert report.extra["step_norm"] == 0.0 and report.extra["shaped"] == [0.0] * 6


def test_max_step_norm_clips() -> None:
    lin, lin2 = _linear(), _linear()
    free = _op(lin, lr=1.0).step(lin, 0)
    assert free.extra["step_norm"] > 0.05 and not free.extra["clipped"]
    base = _flat(lin2)
    op = _op(lin2, lr=1.0, max_step_norm=0.05)
    report = op.step(lin2, 0)
    assert report.extra["clipped"]
    assert report.extra["step_norm_unclipped"] == pytest.approx(free.extra["step_norm"], rel=1e-5)
    assert report.extra["step_norm"] == pytest.approx(0.05, rel=1e-5)
    assert float(torch.linalg.vector_norm(_flat(lin2) - base)) == pytest.approx(0.05, rel=1e-3)
    # same direction as the unclipped step
    d_free = _flat(lin) - base
    d_clip = _flat(lin2) - base
    cos = float(torch.dot(d_free, d_clip) / (d_free.norm() * d_clip.norm()))
    assert cos > 0.999


def test_consolidate_reweighting() -> None:
    lin_plain, lin_cons = _linear(), _linear()
    base = _flat(lin_plain).double()
    target = torch.randn(15, generator=torch.Generator().manual_seed(9))
    plain = _op(lin_plain, fitness=_quadratic(target), population=8, antithetic=False)
    cons = _op(lin_cons, fitness=_quadratic(target), population=8, antithetic=False, consolidate=True)
    rp = plain.step(lin_plain, 1)
    rc = cons.step(lin_cons, 1)
    d = _flat(lin_plain).double() - base
    dc = _flat(lin_cons).double() - base
    # renormalised to the plain step's norm, never opposed to it
    assert rc.extra["step_norm"] == pytest.approx(rp.extra["step_norm"], rel=1e-6)
    assert float(torch.dot(d, dc)) > 0
    # manual formula: w_i = s_i * max(0, cos(s_i eps_i, d)), rescaled to ||d||
    shaped = rp.extra["shaped"]
    eps = [plain.perturbation(1, i).double() for i in range(8)]
    dir_plain = sum(s * e for s, e in zip(shaped, eps))
    agree = []
    for s, e in zip(shaped, eps):
        c = s * e
        agree.append(max(0.0, float(torch.dot(c, dir_plain) / (c.norm() * dir_plain.norm()))) if s else 0.0)
    assert rc.extra["agreement"] == pytest.approx(agree, abs=1e-6)
    assert all(0.0 <= a <= 1.0 for a in agree) and len(set(agree)) > 1
    dir_cons = sum(s * a * e for s, a, e in zip(shaped, agree, eps))
    expected = dir_cons * (d.norm() / dir_cons.norm())
    torch.testing.assert_close(dc, expected, atol=1e-6, rtol=1e-4)


def test_consolidate_zeroes_disagreeing_members() -> None:
    """In one dimension cos(c_i, d) is +-1: members whose contribution opposes d get weight 0."""
    torch.manual_seed(0)
    lin = nn.Linear(1, 1, bias=False)
    fits = [3.0, -1.0, 0.5, 2.0]
    fixed = [1.0, 1.0, -1.0, 0.5]  # fixed draws so the sign pattern does not depend on the RNG

    class FixedNoiseES(ESOperator):
        def noise(self, update_index: int, noise_index: int) -> torch.Tensor:
            return torch.tensor([fixed[noise_index]])

    op = FixedNoiseES(
        FullParameterSpace(lin, ["weight"]),
        lambda m, t, i: FitnessResult(fits[i], None, (), ()),
        ESConfig(population=4, sigma=0.1, lr=0.05, antithetic=False, consolidate=True),
    )
    base = float(lin.weight)
    report = op.step(lin, 0)
    shaped = report.extra["shaped"]
    eps = [float(op.perturbation(0, i)[0]) for i in range(4)]
    d = sum(s * e for s, e in zip(shaped, eps))
    expected_agree = [1.0 if s * e * d > 0 else 0.0 for s, e in zip(shaped, eps)]
    assert expected_agree == [1.0, 0.0, 1.0, 1.0]  # member 1's contribution opposes d
    assert report.extra["agreement"] == pytest.approx(expected_agree)
    # renormalised to ||d||: in 1-D the consolidated step equals the plain step
    assert float(lin.weight) - base == pytest.approx(op.config.lr * d / (4 * op.config.sigma), rel=1e-5)


# ------------------------------------------------------------------ decisions and state
@pytest.mark.parametrize("decision", list(GateDecision))
def test_on_decision_rebases_from_current_weights(decision: GateDecision) -> None:
    lin = _linear()
    op = _op(lin)
    op.step(lin, 0)
    with torch.no_grad():  # the loop restores / shrinks / reverts weights before on_decision
        lin.weight.mul_(0.5)
        lin.bias.add_(1.0)
    current = fingerprint(lin)
    op.on_decision(decision)
    op.space.write(None)
    assert fingerprint(lin) == current
    with pytest.raises(ESError):
        op.on_decision("rollback")  # type: ignore[arg-type]


def test_step_perturbs_around_current_weights() -> None:
    lin = _linear()
    op = _op(lin)
    op.step(lin, 0)
    with torch.no_grad():
        lin.weight.add_(0.25)  # external write without on_decision
    current = _flat(lin)
    seen: list[torch.Tensor] = []

    def fitness(model: nn.Module, t: int, i: int) -> FitnessResult:
        seen.append(_flat(model))
        return FitnessResult(0.0, 0.0, (), ())

    op.fitness = fitness
    op.step(lin, 1)
    torch.testing.assert_close(seen[0], current + op.config.sigma * op.perturbation(1, 0))


def test_state_dict_roundtrip_continues_identically() -> None:
    target = torch.linspace(-1, 1, 15)
    lin_a = _linear()
    op_a = _op(lin_a, fitness=_quadratic(target), seed=11)
    for t in range(3):
        op_a.step(lin_a, t)
    state = json.loads(json.dumps(op_a.state_dict()))
    assert state["steps_done"] == 3 and state["last_update_index"] == 2

    lin_b = _linear(5)  # different initial weights: resume loads the model first
    lin_b.load_state_dict(lin_a.state_dict())
    op_b = _op(lin_b, fitness=_quadratic(target), seed=11)
    op_b.load_state_dict(state)
    assert op_b.state_dict() == op_a.state_dict()
    for t in (3, 4):
        for i in range(op_a.config.population):
            assert torch.equal(op_a.perturbation(t, i), op_b.perturbation(t, i))
        ra, rb = op_a.step(lin_a, t), op_b.step(lin_b, t)
        assert ra.extra["fitness"] == pytest.approx(rb.extra["fitness"])
        for pa, pb in zip(lin_a.parameters(), lin_b.parameters()):
            torch.testing.assert_close(pa, pb)
    assert op_b.state_dict() == op_a.state_dict()

    with pytest.raises(ESError, match="config differs"):
        _op(_linear(), seed=12).load_state_dict(state)
    with pytest.raises(ESError, match="dim"):
        ESOperator(
            FullParameterSpace(_linear(), ["weight"]),
            _quadratic(target),
            ESConfig(population=6, sigma=0.1, lr=0.05, seed=11),
        ).load_state_dict(state)
    with pytest.raises(ESError, match="version"):
        op_b.load_state_dict({**state, "version": 99})


@pytest.mark.parametrize(
    "corrupt",
    [
        {"drop": "forward_passes_total"},
        {"drop": "steps_done"},
        {"drop": "last_update_index"},
        {"drop": "config"},
        {"drop": "dim"},
        {"drop": "version"},
        {"set": ("steps_done", -1)},
        {"set": ("steps_done", True)},
        {"set": ("steps_done", "3")},
        {"set": ("last_update_index", 2.5)},
        {"set": ("forward_passes_total", None)},
        {"set": ("config", [1, 2])},
        {"set": ("dim", "15")},
    ],
    ids=lambda c: f"drop-{c['drop']}" if "drop" in c else f"set-{c['set'][0]}-{c['set'][1]!r}",
)
def test_load_state_dict_malformed_raises_eserror_and_changes_nothing(corrupt: dict[str, Any]) -> None:
    lin_a = _linear()
    op_a = _op(lin_a)
    for t in range(3):
        op_a.step(lin_a, t)
    state: dict[str, Any] = json.loads(json.dumps(op_a.state_dict()))
    if "drop" in corrupt:
        del state[corrupt["drop"]]
    else:
        key, value = corrupt["set"]
        state[key] = value
    lin_b = _linear()
    op_b = _op(lin_b)
    op_b.step(lin_b, 0)
    before = op_b.state_dict()
    with pytest.raises(ESError):
        op_b.load_state_dict(state)
    assert op_b.state_dict() == before  # not half-loaded
    with pytest.raises(ESError):
        op_b.load_state_dict([("version", 1)])  # type: ignore[arg-type]


def test_load_state_dict_rebases() -> None:
    lin = _linear()
    op = _op(lin)
    state = op.state_dict()
    with torch.no_grad():
        lin.weight.add_(1.0)  # e.g. weights loaded from a checkpoint
    current = fingerprint(lin)
    op.load_state_dict(state)
    op.space.write(None)
    assert fingerprint(lin) == current


# ------------------------------------------------------------------ functional
def _gain_problem() -> tuple[nn.Module, ChannelGainSpace, Callable[[nn.Module], float]]:
    model = build_tiny_llama(VOCAB, seed=0)
    sel = ChannelSelection({0: range(8), 1: range(8)})
    layers = model.model.layers
    orig = {
        layer: layers[layer].mlp.down_proj.weight.detach()[0, list(c)].clone() for layer, c in sel.items()
    }
    target = torch.linspace(-0.3, 0.3, sel.count())

    def score(m: nn.Module) -> float:
        g = torch.cat(
            [
                m.model.layers[layer].mlp.down_proj.weight.detach()[0, list(c)] / orig[layer] - 1  # type: ignore[union-attr,index]
                for layer, c in sel.items()
            ]
        )
        return -float(((g - target) ** 2).sum())

    return model, ChannelGainSpace(model, sel), score


@pytest.mark.parametrize(
    "extra_cfg",
    [{}, {"fitness_shaping": "rank"}, {"consolidate": True}],
    ids=["zscore", "rank", "consolidate"],
)
def test_es_improves_gain_objective_on_tiny_model(extra_cfg: dict[str, Any]) -> None:
    model, space, score = _gain_problem()
    before_out = snapshot(model)

    def fitness(m: nn.Module, t: int, i: int) -> FitnessResult:
        f = score(m)
        return FitnessResult(f, f, (f > -0.05,), (f > -0.05,))

    op = ESOperator(space, fitness, ESConfig(population=16, sigma=0.05, lr=0.005, seed=3, **extra_cfg))
    start = score(model)  # unperturbed model, not the population mean
    reports = [op.step(model, t) for t in range(30)]
    end = score(model)
    assert end > start
    assert end > 0.25 * start  # squared error reduced at least 4x (start < 0)
    assert all(r.backward_passes == 0 and r.forward_passes == 16 for r in reports)
    assert all(len(r.v1_pass) == 16 for r in reports)
    # only the selected down_proj columns moved
    report = verify_modification_scope(before_out, model, space.mask(), model)
    assert report.ok, report.summary()


def test_es_improves_quadratic_full_space() -> None:
    lin = _linear()
    target = torch.linspace(-1, 1, 15)
    op = _op(lin, fitness=_quadratic(target), population=10, sigma=0.1, lr=0.02, seed=4)
    start = -float(((_flat(lin) - target) ** 2).sum())
    for t in range(30):
        op.step(lin, t)
    end = -float(((_flat(lin) - target) ** 2).sum())
    assert end > 0.5 * start


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_default_config_improves_gain_objective(seed: int) -> None:
    """The shipped defaults (only the seed is set) must not overshoot on the gain toy problem.

    Regression: the former defaults (sigma=0.02, lr=0.01) made every zscore step about 12x the
    perturbation norm, and this objective got 4-6x worse over 30 updates.
    """
    model, space, score = _gain_problem()

    def fitness(m: nn.Module, t: int, i: int) -> FitnessResult:
        f = score(m)
        return FitnessResult(f, f, (), ())

    cfg = ESConfig(seed=seed)
    assert cfg.max_step_norm is None and cfg.fitness_shaping == "zscore"  # defaults, unclipped
    op = ESOperator(space, fitness, cfg)
    start = score(model)
    reports = [op.step(model, t) for t in range(30)]
    end = score(model)
    assert end > 0.5 * start  # squared error at least halved (start < 0)
    perturbation_norm = cfg.sigma * math.sqrt(space.dim)
    assert max(r.extra["step_norm"] for r in reports) < 4 * perturbation_norm
