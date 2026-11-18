"""Tests of research_rl.capability.spaces: flat parameter spaces written from a stored base.

CPU only, tiny Llama built in-process. Bit-exactness is asserted only for copies / restores and
for repeated identical writes from the same base; computed values are compared with tolerances.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from research_rl.capability.spaces import (
    ChannelGainSpace,
    ChannelWeightSpace,
    FullParameterSpace,
    SpaceError,
)
from research_rl.capability.types import ParameterSpace
from research_rl.models.ffn_channels import (
    ChannelSelection,
    ChannelSelectionError,
    channel_mask,
    extract_channels,
    verify_modification_scope,
)
from research_rl.models.model_interface import fingerprint, snapshot
from research_rl.models.tiny import build_char_tokenizer, build_tiny_llama, tiny_llama_config

VOCAB = len(build_char_tokenizer())
HIDDEN = 32
SEL = ChannelSelection({0: [1, 5, 9], 1: [0, 2, 3, 63]})

SpaceFactory = Callable[[nn.Module], ChannelGainSpace | ChannelWeightSpace | FullParameterSpace]


def _model(bias: bool = False) -> nn.Module:
    if not bias:
        return build_tiny_llama(VOCAB, seed=0)
    from transformers import LlamaForCausalLM

    torch.manual_seed(0)
    cfg = tiny_llama_config(VOCAB)
    cfg.mlp_bias = True
    model = LlamaForCausalLM(cfg)
    with torch.no_grad():  # HF initialises biases to 0; make them non-zero so changes are visible
        for name, p in model.named_parameters():
            if name.endswith("bias"):
                p.normal_(0.0, 0.1)
    return model


def _full_names(model: nn.Module) -> list[str]:
    return ["model.layers.1.mlp.down_proj.weight", "lm_head.weight", "model.norm.weight"]


FACTORIES: dict[str, SpaceFactory] = {
    "gain": lambda m: ChannelGainSpace(m, SEL),
    "gain_up": lambda m: ChannelGainSpace(m, SEL, up=True),
    "weight": lambda m: ChannelWeightSpace(m, SEL),
    "full": lambda m: FullParameterSpace(m, _full_names(m)),
}


def _vector(dim: int, seed: int = 1, scale: float = 0.1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(dim, generator=g) * scale


@pytest.fixture(params=sorted(FACTORIES))
def case(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(params=[False, True], ids=["nobias", "bias"])
def bias(request: pytest.FixtureRequest) -> bool:
    return bool(request.param)


# ------------------------------------------------------------------ protocol and dims
def test_protocol_and_dims(bias: bool) -> None:
    model = _model(bias)
    k = SEL.count()
    gain = ChannelGainSpace(model, SEL)
    gain_up = ChannelGainSpace(model, SEL, up=True)
    weight = ChannelWeightSpace(model, SEL)
    names = _full_names(model)
    full = FullParameterSpace(model, names)
    for space in (gain, gain_up, weight, full):
        assert isinstance(space, ParameterSpace)
        assert isinstance(space.dim, int)
    assert gain.dim == k and gain_up.dim == k  # one shared gain per channel
    assert weight.dim == k * (3 * HIDDEN + (2 if bias else 0))
    params = dict(model.named_parameters())
    assert full.dim == sum(params[n].numel() for n in names)
    assert gain.parameter_names() == sorted(f"model.layers.{i}.mlp.down_proj.weight" for i in (0, 1))
    up_names = {f"model.layers.{i}.mlp.up_proj.weight" for i in (0, 1)}
    if bias:
        up_names |= {f"model.layers.{i}.mlp.up_proj.bias" for i in (0, 1)}
    assert set(gain_up.parameter_names()) == set(gain.parameter_names()) | up_names
    assert full.parameter_names() == sorted(names)


# ------------------------------------------------------------------ base semantics
def test_write_is_from_base_not_cumulative(case: str, bias: bool) -> None:
    model = _model(bias)
    space = FACTORIES[case](model)
    v = _vector(space.dim)
    space.write(v)
    once = snapshot(model)
    space.write(v)
    twice = snapshot(model)
    assert all(torch.equal(once[n], twice[n]) for n in once)
    space.write(-v)
    space.write(v)
    again = snapshot(model)
    assert all(torch.equal(once[n], again[n]) for n in once)


def test_write_none_restores_base_bitwise(case: str, bias: bool) -> None:
    model = _model(bias)
    before = fingerprint(model)
    space = FACTORIES[case](model)
    space.write(_vector(space.dim, scale=0.5))
    assert fingerprint(model) != before
    space.write(None)
    assert fingerprint(model) == before


def test_rebase_then_write_none_keeps_current(case: str, bias: bool) -> None:
    model = _model(bias)
    space = FACTORIES[case](model)
    v = _vector(space.dim)
    space.write(v)
    current = fingerprint(model)
    space.rebase()
    space.write(None)
    assert fingerprint(model) == current
    # writes are now relative to the new base
    space.write(_vector(space.dim, seed=2))
    assert fingerprint(model) != current
    space.write(None)
    assert fingerprint(model) == current


def test_elements_outside_space_unchanged(case: str, bias: bool) -> None:
    model = _model(bias)
    before = snapshot(model)
    space = FACTORIES[case](model)
    space.write(_vector(space.dim, scale=0.3))
    mask = space.mask()
    report = verify_modification_scope(before, model, mask, model)  # unlisted params: frozen
    assert report.ok, report.summary()
    assert report.params_checked == len(before)
    # every element inside the space changed (random non-zero weights, non-zero vector)
    assert report.changed_allowed == mask.num_trainable()
    assert set(mask.names()) == set(space.parameter_names())


def test_gain_scope_is_tight(bias: bool) -> None:
    """up=False must touch only the down_proj columns: gate and up rows are out of scope."""
    model = _model(bias)
    before = snapshot(model)
    space = ChannelGainSpace(model, SEL)
    space.write(_vector(space.dim, scale=0.3))
    mask = space.mask()
    for layer, chans in SEL.items():
        m = mask.get(f"model.layers.{layer}.mlp.down_proj.weight")
        assert m is not None
        expected = torch.zeros_like(m)
        expected[:, list(chans)] = True
        assert torch.equal(m, expected)
    assert mask.num_trainable() == SEL.count() * HIDDEN
    report = verify_modification_scope(before, model, mask, model)
    assert report.ok and report.changed_allowed == SEL.count() * HIDDEN


def test_weight_space_scope_equals_channel_mask(bias: bool) -> None:
    model = _model(bias)
    before = snapshot(model)
    space = ChannelWeightSpace(model, SEL)
    space.write(_vector(space.dim, scale=0.3))
    report = verify_modification_scope(before, model, SEL, model)  # exactly the owned slices
    assert report.ok, report.summary()
    cm = channel_mask(model, SEL)
    mine = space.mask()
    assert report.changed_allowed == cm.num_trainable() == mine.num_trainable()
    for name in mine.names():
        expected = cm.get(name)
        assert expected is not None
        assert torch.equal(mine.get(name), expected)  # type: ignore[arg-type]


# ------------------------------------------------------------------ exact semantics
def test_gain_values(bias: bool) -> None:
    model = _model(bias)
    base = snapshot(model)
    space = ChannelGainSpace(model, SEL, up=True)
    v = _vector(space.dim, scale=0.2)
    space.write(v)
    now = snapshot(model)
    pos = 0
    for layer, chans in SEL.items():
        g = v[pos : pos + len(chans)]
        pos += len(chans)
        idx = list(chans)
        down = f"model.layers.{layer}.mlp.down_proj.weight"
        up = f"model.layers.{layer}.mlp.up_proj.weight"
        torch.testing.assert_close(now[down][:, idx], base[down][:, idx] * (1 + g)[None, :])
        torch.testing.assert_close(now[up][idx], base[up][idx] * (1 + g)[:, None])
        if bias:
            ub = f"model.layers.{layer}.mlp.up_proj.bias"
            torch.testing.assert_close(now[ub][idx], base[ub][idx] * (1 + g))
            gb = f"model.layers.{layer}.mlp.gate_proj.bias"
            assert torch.equal(now[gb], base[gb])  # gate is never scaled


def test_gain_zero_vector_is_identity() -> None:
    model = _model().to(torch.bfloat16)
    before = fingerprint(model)
    space = ChannelGainSpace(model, SEL, up=True)
    space.write(torch.zeros(space.dim))
    assert fingerprint(model) == before


def test_weight_layout_matches_layer_channels_vectors(bias: bool) -> None:
    model = _model(bias)
    space = ChannelWeightSpace(model, SEL)
    before = extract_channels(model, SEL)
    v = _vector(space.dim, scale=0.05).double()
    space.write(v)
    after = extract_channels(model, SEL)
    pos = 0
    for layer in SEL.layers():
        delta = after[layer].vectors() - before[layer].vectors()  # [k, D] float64
        n = delta.numel()
        torch.testing.assert_close(delta, v[pos : pos + n].reshape(delta.shape), atol=1e-6, rtol=0)
        pos += n
    assert pos == space.dim


def test_full_layout_is_names_order_row_major() -> None:
    model = _model()
    names = _full_names(model)
    before = snapshot(model)
    space = FullParameterSpace(model, names)
    v = _vector(space.dim, scale=0.05)
    space.write(v)
    after = snapshot(model)
    flat = torch.cat([(after[n] - before[n]).reshape(-1) for n in names])
    torch.testing.assert_close(flat, v, atol=1e-6, rtol=0)


def test_full_space_on_plain_module_and_bf16() -> None:
    torch.manual_seed(0)
    lin = nn.Linear(4, 3).to(torch.bfloat16)
    before = fingerprint(lin)
    space = FullParameterSpace(lin, ["weight", "bias"])
    assert space.dim == 15
    space.write(_vector(15))
    assert lin.weight.dtype == torch.bfloat16
    assert fingerprint(lin) != before
    space.write(None)
    assert fingerprint(lin) == before


def test_vector_on_other_dtype_is_accepted() -> None:
    model = _model()
    space = ChannelGainSpace(model, SEL)
    v = _vector(space.dim)
    space.write(v.double())
    a = snapshot(model)
    space.write(v)
    b = snapshot(model)
    for n in a:
        torch.testing.assert_close(a[n], b[n])


def test_parameter_replaced_after_construction_is_followed() -> None:
    model = _model()
    space = ChannelGainSpace(model, ChannelSelection({0: [1]}))
    mlp = model.model.layers[0].mlp  # type: ignore[union-attr,index]
    old = mlp.down_proj.weight
    mlp.down_proj.weight = nn.Parameter(old.detach().clone() + 1.0)
    space.write(None)  # writes into the new Parameter object
    assert torch.equal(mlp.down_proj.weight[:, 1], old.detach()[:, 1])
    assert torch.equal(mlp.down_proj.weight[:, 0], old.detach()[:, 0] + 1.0)


# ------------------------------------------------------------------ errors
@pytest.mark.parametrize(
    "bad", [torch.zeros(3), torch.zeros(SEL.count(), 1), torch.zeros(SEL.count(), dtype=torch.long)]
)
def test_bad_vector_rejected_before_write(bad: torch.Tensor) -> None:
    model = _model()
    space = ChannelGainSpace(model, SEL)
    space.write(_vector(space.dim))
    current = fingerprint(model)
    with pytest.raises(SpaceError):
        space.write(bad)
    with pytest.raises(SpaceError):
        space.write([0.0] * space.dim)  # type: ignore[arg-type]
    assert fingerprint(model) == current


def test_construction_errors() -> None:
    model = _model()
    with pytest.raises(SpaceError):
        ChannelGainSpace(model, ChannelSelection())
    with pytest.raises(SpaceError):
        ChannelWeightSpace(model, ChannelSelection())
    with pytest.raises(SpaceError):
        ChannelGainSpace(model, {0: [1]})  # type: ignore[arg-type]
    with pytest.raises(ChannelSelectionError):
        ChannelGainSpace(model, ChannelSelection({0: [64]}))
    with pytest.raises(ChannelSelectionError):
        ChannelWeightSpace(model, ChannelSelection({5: [0]}))
    with pytest.raises(SpaceError):
        FullParameterSpace(model, [])
    with pytest.raises(SpaceError):
        FullParameterSpace(model, ["lm_head.weight", "lm_head.weight"])
    with pytest.raises(SpaceError):
        FullParameterSpace(model, ["nope.weight"])
    with pytest.raises(SpaceError):
        FullParameterSpace(model, "lm_head.weight")


def test_dtype_change_after_construction_rejected() -> None:
    model = _model()
    space = ChannelGainSpace(model, SEL)
    model.to(torch.float64)
    current = fingerprint(model)
    with pytest.raises(SpaceError):
        space.rebase()
    with pytest.raises(SpaceError):
        space.write(None)
    with pytest.raises(SpaceError):
        space.write(_vector(space.dim))
    assert fingerprint(model) == current


def test_shape_change_after_construction_rejected_by_write_and_rebase() -> None:
    """A replaced parameter of another (broadcastable / index-compatible) shape is rejected by
    ``write(None)``, ``write(v)`` and ``rebase()`` alike, and is left untouched."""
    torch.manual_seed(0)
    lin = nn.Linear(4, 1)
    full = FullParameterSpace(lin, ["bias"])
    lin.bias = nn.Parameter(torch.arange(5, dtype=torch.float32))  # (1,) -> (5,): broadcastable
    model = _model()
    gain = ChannelGainSpace(model, ChannelSelection({0: [1]}))
    mlp = model.model.layers[0].mlp  # type: ignore[union-attr,index]
    wide = torch.randn(HIDDEN, 2 * mlp.down_proj.weight.shape[1], generator=torch.Generator().manual_seed(3))
    mlp.down_proj.weight = nn.Parameter(wide.clone())  # column 1 still exists: index_copy_ would work
    for space, module, param in ((full, lin, "bias"), (gain, mlp.down_proj, "weight")):
        before = getattr(module, param).detach().clone()
        for arg in ("none", "vector", "rebase"):
            with pytest.raises(SpaceError, match="shape changed"):
                if arg == "rebase":
                    space.rebase()
                else:
                    space.write(None if arg == "none" else _vector(space.dim))
            assert torch.equal(getattr(module, param).detach(), before)


def test_rebase_is_all_or_nothing() -> None:
    """If one covered parameter fails validation, ``rebase`` changes no stored base."""
    torch.manual_seed(0)
    lin = nn.Linear(4, 3)
    space = FullParameterSpace(lin, ["weight", "bias"])
    base_weight = lin.weight.detach().clone()
    with torch.no_grad():
        lin.weight.add_(1.0)
    lin.bias = nn.Parameter(torch.zeros(3, dtype=torch.float64))  # dtype change: bias fails
    with pytest.raises(SpaceError):
        space.rebase()
    lin.bias = nn.Parameter(torch.zeros(3))
    space.write(None)
    assert torch.equal(lin.weight.detach(), base_weight)  # the weight base was not replaced
