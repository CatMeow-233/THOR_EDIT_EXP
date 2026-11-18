"""Tests of research_rl.capability.weights: exact snapshots, swaps, interpolation, group rollback.

CPU only, tiny Llama built in-process. Bit-exactness is asserted only for copies / swaps /
restores and for trajectories that execute identical ops on identical inputs in one process
(a restored run vs. a run that never left the snapshot); trained values are never compared
against hard-coded numbers.
"""

from __future__ import annotations

import copy
import dataclasses
import warnings
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn
from transformers import LlamaForCausalLM

from research_rl.capability.types import ChannelGroup
from research_rl.capability.weights import (
    SnapshotError,
    StateSnapshot,
    capture,
    channel_groups_by_damage,
    group_damage,
    interpolate,
    restore,
    revert_groups,
    swapped,
)
from research_rl.models.ffn_channels import (
    ChannelSelection,
    ChannelSelectionError,
    channel_mask,
    extract_channels,
    verify_modification_scope,
)
from research_rl.models.model_interface import fingerprint, snapshot
from research_rl.models.tiny import build_tiny_llama, tiny_llama_config
from research_rl.optim.optimizer_wrapper import HookedOptimizer

VOCAB = 100  # >= len(build_char_tokenizer()); token ids are drawn from [3, VOCAB)
HIDDEN = 32
INTER = 64


# ============================================================================== helpers
def make_model(dtype: torch.dtype = torch.float32, seed: int = 0, **overrides: int) -> nn.Module:
    model = build_tiny_llama(VOCAB, seed=seed, **overrides).to(dtype)
    model.train()
    return model


def adamw(model: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)


def hooked(model: nn.Module, write_rounding: str = "nearest") -> HookedOptimizer:
    return HookedOptimizer(
        torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1),
        model.named_parameters(),
        master_weights=True,
        write_rounding=write_rounding,
        rounding_seed=7,
    )


def batch(step: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + step)
    return torch.randint(3, VOCAB, (4, 10), generator=g)


def train_step(model: nn.Module, opt: torch.optim.Optimizer, step: int) -> None:
    opt.zero_grad(set_to_none=True)
    ids = batch(step)
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    opt.step()


def train(model: nn.Module, opt: torch.optim.Optimizer, steps: range) -> None:
    for s in steps:
        train_step(model, opt, s)


def params_of(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def masters_of(model: nn.Module, opt: HookedOptimizer) -> dict[str, torch.Tensor]:
    out = {}
    for n, p in model.named_parameters():
        m = opt.master_of(p)
        if m is not None:
            out[n] = m.detach().clone()
    return out


def assert_tensors_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert set(a) == set(b)
    for n in a:
        assert a[n].dtype == b[n].dtype, n
        assert torch.equal(a[n].cpu(), b[n].cpu()), n


def assert_tree_equal(a: Any, b: Any, path: str = "") -> None:
    """Bitwise equality of two state-dict trees (dtype and shape included)."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), path
        assert a.dtype == b.dtype and a.shape == b.shape, path
        assert torch.equal(a.cpu(), b.cpu()), path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), (path, sorted(map(str, a)), sorted(map(str, b)))
        for k in a:
            assert_tree_equal(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_tree_equal(x, y, f"{path}[{i}]")
    else:
        assert a == b, (path, a, b)


def changed_anywhere(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return any(not torch.equal(a[n], b[n]) for n in a)


def no_resync_step(model: nn.Module, opt: HookedOptimizer, step: int) -> None:
    """One step that must not see any 'modified outside the optimizer' reconciliation."""
    before = opt.master_resyncs
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train_step(model, opt, step)
    assert opt.master_resyncs == before
    assert not [w for w in caught if "outside the optimizer" in str(w.message)]


# ============================================================================== capture
def test_capture_copies_are_independent_of_later_mutation() -> None:
    model = make_model()
    opt = adamw(model)
    train(model, opt, range(2))
    snap = capture(model, None, optimizer=opt, channel_layers=[0])
    oracle_params = params_of(model)
    oracle_state = copy.deepcopy(opt.state_dict())
    assert isinstance(snap, StateSnapshot)
    assert snap.names() == [n for n, _ in model.named_parameters()]
    assert snap.masters is None
    assert snap.fingerprint == fingerprint(model)
    assert all(t.device.type == "cpu" for t in snap.params.values())
    # In-place mutation of the live optimizer state / params must not reach the snapshot.
    with torch.no_grad():
        for st in opt.state.values():
            st["exp_avg"].add_(1.0)
            st["step"].add_(5.0)
        for p in model.parameters():
            p.add_(1.0)
    train(model, opt, range(2, 4))
    snap.verify()
    assert_tensors_equal(snap.params, oracle_params)
    assert snap.optimizer_state is not None
    assert_tree_equal(snap.optimizer_state, oracle_state)
    assert snap.channel_state is not None
    assert torch.equal(snap.channel_state[0].down, oracle_params["model.layers.0.mlp.down_proj.weight"])
    # verify() detects tampering with the snapshot itself.
    first = next(iter(snap.params))
    snap.params[first].add_(1.0)
    with pytest.raises(SnapshotError):
        snap.verify()


def test_capture_rejects_unknown_names_and_keeps_order() -> None:
    model = make_model()
    with pytest.raises(SnapshotError, match="unknown"):
        capture(model, ["model.layers.0.mlp.up_proj.weight", "nope.weight"])
    names = ["model.layers.1.mlp.up_proj.weight", "model.embed_tokens.weight", "model.embed_tokens.weight"]
    snap = capture(model, names)
    assert snap.names() == ["model.embed_tokens.weight", "model.layers.1.mlp.up_proj.weight"]


# ============================================================================== restore (AdamW)
def test_restore_adamw_bit_identical_and_trajectory_matches_reference() -> None:
    model = make_model()
    opt = adamw(model)
    train(model, opt, range(3))
    snap = capture(model, None, optimizer=opt)
    oracle_params = params_of(model)
    oracle_state = copy.deepcopy(opt.state_dict())
    train(model, opt, range(3, 7))  # diverge
    assert changed_anywhere(oracle_params, params_of(model))

    restore(model, snap, optimizer=opt)
    assert_tensors_equal(params_of(model), oracle_params)
    assert_tree_equal(opt.state_dict(), oracle_state)  # exp_avg, exp_avg_sq, step, param_groups

    # Continue training: identical ops on identical inputs as a run that never left the snapshot.
    train(model, opt, range(100, 103))
    reference = make_model()
    ref_opt = adamw(reference)
    train(reference, ref_opt, range(3))
    train(reference, ref_opt, range(100, 103))
    assert_tensors_equal(params_of(model), params_of(reference))
    assert_tree_equal(opt.state_dict(), ref_opt.state_dict())

    # Training after a restore must not have mutated the snapshot (restore loads a fresh clone).
    restore(model, snap, optimizer=opt)
    assert_tensors_equal(params_of(model), oracle_params)
    assert_tree_equal(opt.state_dict(), oracle_state)
    snap.verify()


def test_restore_without_optimizer_state_restores_params_only() -> None:
    model = make_model()
    opt = adamw(model)
    train(model, opt, range(2))
    snap = capture(model, None, optimizer=opt)
    oracle_params = params_of(model)
    train(model, opt, range(2, 4))
    state_now = copy.deepcopy(opt.state_dict())
    restore(model, snap, optimizer=opt, restore_optimizer=False)
    assert_tensors_equal(params_of(model), oracle_params)
    assert_tree_equal(opt.state_dict(), state_now)


def test_restore_subset_of_names_leaves_other_params() -> None:
    model = make_model()
    opt = adamw(model)
    train(model, opt, range(2))
    names = [n for n, _ in model.named_parameters() if ".mlp." in n]
    snap = capture(model, names, optimizer=opt)
    oracle = params_of(model)
    oracle_state = copy.deepcopy(opt.state_dict())
    train(model, opt, range(2, 4))
    current = params_of(model)
    restore(model, snap, optimizer=opt)
    now = params_of(model)
    for n in now:
        assert torch.equal(now[n], oracle[n] if n in names else current[n]), n
    assert_tree_equal(opt.state_dict(), oracle_state)  # the optimizer state is global


# ============================================================================== restore (bf16 masters)
@pytest.mark.parametrize("write_rounding", ["nearest", "stochastic"])
def test_restore_hooked_bf16_masters_bit_identical(write_rounding: str) -> None:
    model = make_model(torch.bfloat16)
    opt = hooked(model, write_rounding)
    train(model, opt, range(3))
    snap = capture(model, None, optimizer=opt)
    oracle_params = params_of(model)
    oracle_masters = masters_of(model, opt)
    oracle_state = copy.deepcopy(opt.state_dict())
    assert snap.masters is not None and set(snap.masters) == set(oracle_masters)
    # Non-trivial masters: they hold sub-ulp information a resync would destroy.
    assert any(not torch.equal(m, m.to(torch.bfloat16).float()) for m in oracle_masters.values())
    train(model, opt, range(3, 7))
    assert changed_anywhere(oracle_masters, masters_of(model, opt))

    restore(model, snap, optimizer=opt)
    assert_tensors_equal(params_of(model), oracle_params)
    assert_tensors_equal(masters_of(model, opt), oracle_masters)
    assert_tree_equal(opt.state_dict(), oracle_state)  # moments (fp32), counters, masters

    no_resync_step(model, opt, 100)
    train(model, opt, range(101, 103))
    reference = make_model(torch.bfloat16)
    ref_opt = hooked(reference, write_rounding)
    train(reference, ref_opt, range(3))
    train(reference, ref_opt, range(100, 103))
    assert_tensors_equal(params_of(model), params_of(reference))
    assert_tensors_equal(masters_of(model, opt), masters_of(reference, ref_opt))
    assert_tree_equal(opt.state_dict(), ref_opt.state_dict())

    restore(model, snap, optimizer=opt)  # snapshot untouched by the training after the restore
    assert_tensors_equal(masters_of(model, opt), oracle_masters)
    assert_tree_equal(opt.state_dict(), oracle_state)


def test_restore_hooked_without_optimizer_state_fixes_masters_only() -> None:
    model = make_model(torch.bfloat16)
    opt = hooked(model)
    train(model, opt, range(2))
    snap = capture(model, None, optimizer=opt)
    oracle_params = params_of(model)
    oracle_masters = masters_of(model, opt)
    train(model, opt, range(2, 4))
    state_now = copy.deepcopy(opt.state_dict())
    restore(model, snap, optimizer=opt, restore_optimizer=False)
    assert_tensors_equal(params_of(model), oracle_params)
    assert_tensors_equal(masters_of(model, opt), oracle_masters)
    after = opt.state_dict()
    assert_tree_equal(after["state"], state_now["state"])  # moments untouched
    assert after["research_rl"]["nominal_steps"] == state_now["research_rl"]["nominal_steps"]
    no_resync_step(model, opt, 50)


def test_restore_hooked_from_snapshot_without_masters_uses_param_values() -> None:
    model = make_model(torch.bfloat16)
    opt = hooked(model)
    train(model, opt, range(2))
    snap = capture(model, None)  # no optimizer: no masters
    oracle = params_of(model)
    train(model, opt, range(2, 4))
    restore(model, snap, optimizer=opt, restore_optimizer=False)
    assert_tensors_equal(params_of(model), oracle)
    assert_tensors_equal(masters_of(model, opt), {n: t.float() for n, t in oracle.items()})
    no_resync_step(model, opt, 9)


# ============================================================================== swapped
def test_swapped_loads_snapshot_and_restores_bitwise_also_on_exception() -> None:
    model = make_model()
    opt = adamw(model)
    train(model, opt, range(2))
    snap = capture(model, None, optimizer=opt)
    oracle = params_of(model)
    train(model, opt, range(2, 4))
    pre = params_of(model)
    pre_state = copy.deepcopy(opt.state_dict())
    with swapped(model, snap) as s:
        assert s is snap
        assert_tensors_equal(params_of(model), oracle)
    assert_tensors_equal(params_of(model), pre)
    assert_tree_equal(opt.state_dict(), pre_state)
    with pytest.raises(RuntimeError, match="boom"), swapped(model, snap):
        assert_tensors_equal(params_of(model), oracle)
        raise RuntimeError("boom")
    assert_tensors_equal(params_of(model), pre)
    assert_tree_equal(opt.state_dict(), pre_state)
    # A partial snapshot swaps only its own parameters.
    part = capture(model, ["model.layers.0.mlp.gate_proj.weight"])
    with torch.no_grad():
        model.model.layers[0].mlp.gate_proj.weight.add_(1.0)
    changed = params_of(model)
    with swapped(model, part):
        now = params_of(model)
        for n in now:
            expect = part.params[n] if n in part.params else changed[n]
            assert torch.equal(now[n], expect), n
    assert_tensors_equal(params_of(model), changed)


def test_swapped_does_not_disturb_hooked_masters_or_trajectory() -> None:
    model, reference = make_model(torch.bfloat16), make_model(torch.bfloat16)
    opt, ref_opt = hooked(model, "stochastic"), hooked(reference, "stochastic")
    train(model, opt, range(2))
    train(reference, ref_opt, range(2))
    snap = capture(model, None, optimizer=opt)
    train(model, opt, range(2, 4))
    train(reference, ref_opt, range(2, 4))
    masters_before = masters_of(model, opt)
    with swapped(model, snap):
        assert_tensors_equal(params_of(model), snap.params)
    assert_tensors_equal(masters_of(model, opt), masters_before)
    no_resync_step(model, opt, 4)
    train_step(reference, ref_opt, 4)
    assert_tensors_equal(params_of(model), params_of(reference))
    assert_tensors_equal(masters_of(model, opt), masters_of(reference, ref_opt))


# ============================================================================== interpolate
def test_interpolate_fp32_endpoints_exact_and_midpoint() -> None:
    model = make_model()
    opt = adamw(model)
    train(model, opt, range(2))
    inc = capture(model, None)
    train(model, opt, range(2, 5))
    cand = capture(model, None)
    state = copy.deepcopy(opt.state_dict())
    interpolate(model, inc, cand, 0.0)
    assert_tensors_equal(params_of(model), inc.params)
    interpolate(model, inc, cand, 1.0)
    assert_tensors_equal(params_of(model), cand.params)
    interpolate(model, inc, cand, 0.5)
    now = params_of(model)
    for n in now:
        mid = (inc.params[n].double() + cand.params[n].double()) / 2
        torch.testing.assert_close(now[n].double(), mid, rtol=1e-6, atol=1e-7)
    assert changed_anywhere(now, inc.params) and changed_anywhere(now, cand.params)
    assert_tree_equal(opt.state_dict(), state)  # no optimizer passed: untouched
    for bad in (-0.1, 1.5, float("nan"), True):
        with pytest.raises(ValueError):
            interpolate(model, inc, cand, bad)
    assert_tensors_equal(params_of(model), now)


def test_interpolate_bf16_without_masters_rounds_to_nearest() -> None:
    model = make_model(torch.bfloat16)
    opt = hooked(model)
    train(model, opt, range(2))
    inc = capture(model, None)
    train(model, opt, range(2, 6))
    cand = capture(model, None)
    interpolate(model, inc, cand, 0.0)
    assert_tensors_equal(params_of(model), inc.params)
    interpolate(model, inc, cand, 1.0)
    assert_tensors_equal(params_of(model), cand.params)
    interpolate(model, inc, cand, 0.5)
    for n, p in model.named_parameters():
        assert p.dtype == torch.bfloat16
        mid = (inc.params[n].float() + cand.params[n].float()) / 2
        # one bf16 rounding of the fp32 midpoint: relative error <= 2**-8
        torch.testing.assert_close(p.detach().float(), mid, rtol=2**-8, atol=1e-30)


def test_interpolate_hooked_masters_consistent() -> None:
    model = make_model(torch.bfloat16)
    opt = hooked(model)
    train(model, opt, range(2))
    inc = capture(model, None, optimizer=opt)
    train(model, opt, range(2, 5))
    cand = capture(model, None, optimizer=opt)
    assert inc.masters is not None and cand.masters is not None
    moments = copy.deepcopy(opt.state_dict()["state"])
    interpolate(model, inc, cand, 0.0, optimizer=opt)
    assert_tensors_equal(masters_of(model, opt), inc.masters)
    assert_tensors_equal(params_of(model), inc.params)  # nearest rounding: served == master.to(bf16)
    interpolate(model, inc, cand, 1.0, optimizer=opt)
    assert_tensors_equal(masters_of(model, opt), cand.masters)
    assert_tensors_equal(params_of(model), cand.params)
    interpolate(model, inc, cand, 0.5, optimizer=opt)
    masters = masters_of(model, opt)
    for n, p in model.named_parameters():
        mid = (inc.masters[n].double() + cand.masters[n].double()) / 2
        torch.testing.assert_close(masters[n].double(), mid, rtol=1e-6, atol=1e-9)
        assert torch.equal(p.detach(), masters[n].to(torch.bfloat16)), n  # served = round(master)
    assert_tree_equal(opt.state_dict()["state"], moments)  # moments untouched
    no_resync_step(model, opt, 10)


# ============================================================================== channel groups
L0_DOWN = "model.layers.0.mlp.down_proj.weight"
L1_GATE = "model.layers.1.mlp.gate_proj.weight"
L1_DOWN = "model.layers.1.mlp.down_proj.weight"


def groups_abcd() -> list[ChannelGroup]:
    return [
        ChannelGroup("A", ChannelSelection({0: [1, 2]})),
        ChannelGroup("B", ChannelSelection({1: [5]})),
        ChannelGroup("C", ChannelSelection({0: [10]})),
        ChannelGroup("D", ChannelSelection({1: [7]})),
    ]


def displace(model: nn.Module) -> None:
    """Known displacement: A: L0 ch1 down col +1; B: L1 ch5 gate row +3; C: L0 ch10 down col +10."""
    with torch.no_grad():
        model.model.layers[0].mlp.down_proj.weight[:, 1] += 1.0
        model.model.layers[1].mlp.gate_proj.weight[5, :] += 3.0
        model.model.layers[0].mlp.down_proj.weight[:, 10] += 10.0


def activations() -> dict[int, torch.Tensor]:
    a0 = torch.ones(INTER)
    a0[10] = 0.0  # large displacement on a channel the failing slice never uses -> no damage
    return {0: a0, 1: torch.full((INTER,), 0.5)}


@pytest.mark.parametrize("source", ["params", "channel_state", "channel_state_only"])
def test_channel_groups_by_damage_ordering(source: str) -> None:
    model = make_model()
    if source == "params":
        inc = capture(model, None)
    elif source == "channel_state":
        inc = capture(model, None, channel_layers=[0, 1])
    else:
        inc = capture(model, [], channel_layers=[0, 1])
    displace(model)
    before = params_of(model)
    damage = group_damage(inc, model, activations(), groups_abcd())
    assert damage["A"] == pytest.approx(HIDDEN**0.5 * 1.0 * 1.0, rel=1e-5)
    assert damage["B"] == pytest.approx(3.0 * HIDDEN**0.5 * 0.5, rel=1e-5)
    assert damage["C"] == 0.0 and damage["D"] == 0.0
    order = channel_groups_by_damage(inc, model, activations(), groups_abcd())
    assert [g.name for g in order] == ["B", "A", "C", "D"]  # C before D: stable tie
    assert_tensors_equal(params_of(model), before)  # pure


def test_group_damage_validation() -> None:
    model = make_model()
    inc = capture(model, None)
    groups = groups_abcd()
    with pytest.raises(ValueError, match="no entry for layer 1"):
        group_damage(inc, model, {0: torch.ones(INTER)}, groups)
    with pytest.raises(ValueError, match="length"):
        group_damage(inc, model, {0: torch.ones(3), 1: torch.ones(INTER)}, groups)
    with pytest.raises(ValueError, match=">= 0"):
        group_damage(inc, model, {0: -torch.ones(INTER), 1: torch.ones(INTER)}, groups)
    with pytest.raises(ValueError, match="duplicate"):
        group_damage(inc, model, activations(), [groups[0], groups[0]])
    with pytest.raises(SnapshotError, match="neither"):
        group_damage(capture(model, ["model.embed_tokens.weight"]), model, activations(), groups)


def test_revert_groups_restores_only_group_channels() -> None:
    model = make_model()
    inc = capture(model, None, channel_layers=[0, 1])
    displace(model)
    before = snapshot(model)
    groups = groups_abcd()
    chosen = [groups[1], groups[2]]  # B, C
    reports = revert_groups(model, inc, chosen)
    assert len(reports) == 2
    assert reports[0].elements_changed == {1: HIDDEN}
    assert reports[1].elements_changed == {0: HIDDEN}
    sel = ChannelSelection({1: [5], 0: [10]})
    scope = verify_modification_scope(before, model, sel, model)
    assert scope.ok and scope.changed_allowed == 2 * HIDDEN
    now = extract_channels(model, sel)
    assert inc.channel_state is not None
    for layer in (0, 1):
        ref = inc.channel_state[layer].select(sel.channels(layer))
        assert torch.equal(now[layer].gate, ref.gate) and torch.equal(now[layer].down, ref.down)
    # group A was not reverted: still displaced
    assert torch.equal(model.model.layers[0].mlp.down_proj.weight[:, 1], before[L0_DOWN][:, 1])
    assert not torch.equal(before[L0_DOWN][:, 1], inc.params[L0_DOWN][:, 1])


@pytest.mark.parametrize("write_rounding", ["nearest", "stochastic"])
def test_revert_groups_fixes_hooked_masters(write_rounding: str) -> None:
    model = make_model(torch.bfloat16)
    opt = hooked(model, write_rounding)
    train(model, opt, range(2))
    inc = capture(model, None, optimizer=opt)
    assert inc.masters is not None
    train(model, opt, range(2, 5))
    pre_params = snapshot(model)
    pre_masters = masters_of(model, opt)
    moments = copy.deepcopy(opt.state_dict()["state"])
    group = ChannelGroup("g", ChannelSelection({0: [0, 3, 9], 1: [4]}))
    revert_groups(model, inc, [group], optimizer=opt)
    assert verify_modification_scope(pre_params, model, group.selection, model).ok
    mask = channel_mask(model, group.selection)
    masters = masters_of(model, opt)
    for n, p in model.named_parameters():
        owned = mask.get(n)
        if owned is None:
            assert torch.equal(masters[n], pre_masters[n]), n
            continue
        assert torch.equal(masters[n], torch.where(owned, inc.masters[n], pre_masters[n])), n
        assert torch.equal(p.detach()[owned], inc.params[n][owned]), n
        assert torch.equal(p.detach()[~owned], pre_params[n][~owned]), n
    assert_tree_equal(opt.state_dict()["state"], moments)
    no_resync_step(model, opt, 20)


# ============================================================================== atomicity
def _guarded(model: nn.Module, opt: torch.optim.Optimizer | None, fn: Callable[[], Any], exc: type) -> None:
    """``fn`` must raise ``exc`` and leave params and optimizer state bit-identical."""
    before = params_of(model)
    state = copy.deepcopy(opt.state_dict()) if opt is not None else None
    with pytest.raises(exc):
        fn()
    assert_tensors_equal(params_of(model), before)
    if opt is not None:
        assert_tree_equal(opt.state_dict(), state)


def test_restore_rejects_foreign_snapshots_before_writing() -> None:
    model = make_model()
    opt = adamw(model)
    train(model, opt, range(2))
    wider = make_model(hidden_size=48)
    deeper = make_model(num_hidden_layers=3)
    w_snap = capture(wider, None, optimizer=adamw(wider))
    d_snap = capture(deeper, None)
    _guarded(model, opt, lambda: restore(model, w_snap, optimizer=opt), SnapshotError)
    _guarded(model, opt, lambda: restore(model, d_snap), SnapshotError)  # layer 2 names missing
    _guarded(model, opt, lambda: swapped(model, w_snap).__enter__(), SnapshotError)
    _guarded(model, opt, lambda: interpolate(model, w_snap, w_snap, 0.5), SnapshotError)
    bf = capture(make_model(torch.bfloat16), None)
    _guarded(model, opt, lambda: restore(model, bf), SnapshotError)  # dtype mismatch

    own = capture(model, None)  # no optimizer state
    _guarded(model, opt, lambda: restore(model, own, optimizer=opt), SnapshotError)
    # Optimizer over a different parameter set: group size mismatch, detected before the write.
    partial = capture(model, None, optimizer=torch.optim.AdamW(list(model.parameters())[:3]))
    train(model, opt, range(2, 3))
    _guarded(model, opt, lambda: restore(model, partial, optimizer=opt), SnapshotError)
    # Hooked snapshot into a plain optimizer.
    bf_model = make_model(torch.bfloat16)
    bf_opt = hooked(bf_model)
    train(bf_model, bf_opt, range(1))
    hooked_snap = capture(bf_model, None, optimizer=bf_opt)
    plain_bf = torch.optim.AdamW(bf_model.parameters())
    _guarded(bf_model, plain_bf, lambda: restore(bf_model, hooked_snap, optimizer=plain_bf), SnapshotError)
    # Mismatched name sets in interpolate.
    sub = capture(model, ["model.embed_tokens.weight"])
    _guarded(model, opt, lambda: interpolate(model, own, sub, 0.5), SnapshotError)


def test_revert_groups_atomic_across_groups() -> None:
    model = make_model()
    inc = capture(model, None)
    displace(model)
    good = ChannelGroup("good", ChannelSelection({0: [1]}))
    bad = ChannelGroup("bad", ChannelSelection({1: [INTER + 5]}))
    _guarded(model, None, lambda: revert_groups(model, inc, [good, bad]), ChannelSelectionError)
    no_ref = capture(model, ["model.embed_tokens.weight"])
    _guarded(model, None, lambda: revert_groups(model, no_ref, [good]), SnapshotError)
    bf_inc = capture(make_model(torch.bfloat16), None)
    _guarded(model, None, lambda: revert_groups(model, bf_inc, [good]), SnapshotError)
    assert revert_groups(model, inc, []) == []


# ============================================================================== further paths
def test_restore_hooked_subset_names_keeps_out_of_scope_masters() -> None:
    model = make_model(torch.bfloat16)
    opt = hooked(model, "stochastic")
    train(model, opt, range(2))
    names = [n for n, _ in model.named_parameters() if ".mlp." in n]
    snap = capture(model, names, optimizer=opt)
    assert snap.masters is not None and set(snap.masters) == set(names)
    oracle_params = params_of(model)
    oracle_masters = masters_of(model, opt)
    oracle_state = copy.deepcopy(opt.state_dict())
    train(model, opt, range(2, 5))
    cur_params = params_of(model)
    cur_masters = masters_of(model, opt)
    restore(model, snap, optimizer=opt)
    params, masters = params_of(model), masters_of(model, opt)
    for n in params:
        inside = n in names
        assert torch.equal(params[n], oracle_params[n] if inside else cur_params[n]), n
        assert torch.equal(masters[n], oracle_masters[n] if inside else cur_masters[n]), n
    after = opt.state_dict()
    assert_tree_equal(after["state"], oracle_state["state"])  # the optimizer state is global
    assert after["research_rl"]["nominal_steps"] == oracle_state["research_rl"]["nominal_steps"]
    no_resync_step(model, opt, 30)


class _Wrapper:
    """Minimal Accelerate-style wrapper: the HookedOptimizer is reachable as ``.optimizer``."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.optimizer.param_groups

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state)


def test_restore_through_optimizer_wrapper() -> None:
    model = make_model(torch.bfloat16)
    opt = hooked(model)
    wrapped: Any = _Wrapper(opt)
    train(model, opt, range(2))
    snap = capture(model, None, optimizer=wrapped)
    assert snap.masters is not None
    oracle_params, oracle_masters = params_of(model), masters_of(model, opt)
    oracle_state = copy.deepcopy(opt.state_dict())
    train(model, opt, range(2, 4))
    restore(model, snap, optimizer=wrapped)
    assert_tensors_equal(params_of(model), oracle_params)
    assert_tensors_equal(masters_of(model, opt), oracle_masters)
    assert_tree_equal(opt.state_dict(), oracle_state)


def test_nan_damage_sorts_first() -> None:
    model = make_model()
    inc = capture(model, None)
    displace(model)
    with torch.no_grad():
        model.model.layers[1].mlp.up_proj.weight[7, 0] = float("nan")  # group D
    order = channel_groups_by_damage(inc, model, activations(), groups_abcd())
    assert [g.name for g in order] == ["D", "B", "A", "C"]


def test_revert_groups_master_fallback_without_incumbent_masters() -> None:
    model = make_model(torch.bfloat16)
    opt = hooked(model)
    train(model, opt, range(2))
    inc = capture(model, None)  # no optimizer: no masters in the incumbent
    train(model, opt, range(2, 4))
    pre_masters = masters_of(model, opt)
    group = ChannelGroup("g", ChannelSelection({0: [2, 5]}))
    revert_groups(model, inc, [group], optimizer=opt)
    mask = channel_mask(model, group.selection)
    masters = masters_of(model, opt)
    for n in masters:
        owned = mask.get(n)
        expect = (
            pre_masters[n] if owned is None else torch.where(owned, inc.params[n].float(), pre_masters[n])
        )
        assert torch.equal(masters[n], expect), n
    no_resync_step(model, opt, 40)


# ============================================================================== regression tests (review findings)
def test_restore_rejects_foreign_optimizer_state_before_writing() -> None:
    """Same names/shapes for the captured params, but the optimizer state belongs to another model."""
    model = make_model()
    opt = adamw(model)
    train(model, opt, range(2))
    narrow = make_model(seed=1, intermediate_size=48)  # MLP params (48, 32) instead of (64, 32)
    n_opt = adamw(narrow)
    train(narrow, n_opt, range(2))
    attn = [n for n, _ in narrow.named_parameters() if "self_attn" in n]
    snap = capture(narrow, attn, optimizer=n_opt)  # params pass _check_params, the moments do not fit
    _guarded(model, opt, lambda: restore(model, snap, optimizer=opt), SnapshotError)
    # Into a fresh optimizer (no state yet): the parameter shape is the reference.
    fresh = adamw(model)
    _guarded(model, fresh, lambda: restore(model, snap, optimizer=fresh), SnapshotError)
    # Same model, optimizer built over the parameters in another order (equal group size).
    rev_opt = torch.optim.AdamW(list(model.parameters())[::-1], lr=1e-2)
    train(model, rev_opt, range(2, 3))
    rev_snap = capture(model, None, optimizer=rev_opt)
    _guarded(model, opt, lambda: restore(model, rev_snap, optimizer=opt), SnapshotError)
    # restore_optimizer=False ignores the snapshot's optimizer state entirely.
    restore(model, snap, optimizer=opt, restore_optimizer=False)
    # Factored (non-parameter-shaped) state: accepted into a fresh optimizer, foreign one rejected.
    fac_model = make_model()
    fac_opt = torch.optim.Adafactor(fac_model.parameters(), lr=1e-2)
    train(fac_model, fac_opt, range(1))
    fac_snap = capture(fac_model, None, optimizer=fac_opt)
    fresh_fac = torch.optim.Adafactor(fac_model.parameters(), lr=1e-2)
    restore(fac_model, fac_snap, optimizer=fresh_fac)
    assert_tree_equal(fresh_fac.state_dict(), fac_snap.optimizer_state)
    fac_narrow = make_model(seed=1, intermediate_size=48)
    fac_n_opt = torch.optim.Adafactor(fac_narrow.parameters(), lr=1e-2)
    train(fac_narrow, fac_n_opt, range(1))
    fac_foreign = capture(fac_narrow, attn, optimizer=fac_n_opt)
    for target in (fac_opt, torch.optim.Adafactor(fac_model.parameters(), lr=1e-2)):
        _guarded(
            fac_model, target, lambda t=target: restore(fac_model, fac_foreign, optimizer=t), SnapshotError
        )
    # The same check through a HookedOptimizer (its inner state is the snapshot's "state").
    bf_model = make_model(torch.bfloat16)
    bf_opt = hooked(bf_model)
    train(bf_model, bf_opt, range(1))
    bf_narrow = make_model(torch.bfloat16, seed=1, intermediate_size=48)
    bf_n_opt = hooked(bf_narrow)
    train(bf_narrow, bf_n_opt, range(1))
    bf_snap = capture(bf_narrow, attn, optimizer=bf_n_opt)
    _guarded(bf_model, bf_opt, lambda: restore(bf_model, bf_snap, optimizer=bf_opt), SnapshotError)


def test_revert_groups_and_damage_reject_foreign_incumbent_before_writing() -> None:
    """Incumbent from a model with the same hidden size but another intermediate size."""
    model = make_model(torch.bfloat16)
    opt = hooked(model)
    train(model, opt, range(1))
    narrow = make_model(torch.bfloat16, seed=1, intermediate_size=48)
    n_opt = hooked(narrow)
    train(narrow, n_opt, range(1))
    foreign = capture(narrow, None, optimizer=n_opt)
    group = ChannelGroup("g", ChannelSelection({0: [1, 2]}))
    _guarded(model, opt, lambda: revert_groups(model, foreign, [group], optimizer=opt), SnapshotError)
    _guarded(model, None, lambda: revert_groups(model, foreign, [group]), SnapshotError)
    with pytest.raises(SnapshotError):
        group_damage(foreign, model, {0: torch.ones(INTER)}, [group])
    # Own params but a foreign master: rejected before the channels are written.
    own = capture(model, None, optimizer=opt)
    assert own.masters is not None
    name = "model.layers.0.mlp.up_proj.weight"
    bad = dataclasses.replace(own, masters={**own.masters, name: foreign.masters[name]})  # type: ignore[index]
    with torch.no_grad():
        model.model.layers[0].mlp.up_proj.weight[1, :] += 1.0
    _guarded(model, opt, lambda: revert_groups(model, bad, [group], optimizer=opt), SnapshotError)


def test_interpolate_endpoints_exact_with_non_finite_other_endpoint() -> None:
    model = make_model()
    inc = capture(model, None)
    with torch.no_grad():
        up = model.model.layers[0].mlp.up_proj.weight
        up[0, 0], up[0, 1], up[0, 2] = float("nan"), float("inf"), float("-inf")
    cand = capture(model, None)  # a diverged candidate
    interpolate(model, inc, cand, 0.0)
    assert_tensors_equal(params_of(model), inc.params)
    interpolate(model, inc, cand, 1.0)
    assert fingerprint(params_of(model)) == cand.fingerprint  # bitwise, NaN included
    interpolate(model, cand, inc, 1.0)  # non-finite incumbent, lam=1 -> the (finite) candidate
    assert_tensors_equal(params_of(model), inc.params)
    interpolate(model, inc, cand, 0.5)  # interior: the formula propagates non-finite values
    assert not bool(torch.isfinite(model.model.layers[0].mlp.up_proj.weight[0, :3]).any())


def test_interpolate_hooked_endpoints_exact_with_non_finite_candidate_master() -> None:
    model = make_model(torch.bfloat16)
    opt = hooked(model)
    train(model, opt, range(1))
    inc = capture(model, None, optimizer=opt)
    with torch.no_grad():
        model.model.layers[1].mlp.down_proj.weight[0, 0] = float("nan")
        model.model.layers[1].mlp.down_proj.weight[0, 1] = float("inf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cand = capture(model, None, optimizer=opt)  # reconciles the external write into the masters
    assert cand.masters is not None and not bool(torch.isfinite(cand.masters[L1_DOWN]).all())
    interpolate(model, inc, cand, 0.0, optimizer=opt)
    assert inc.masters is not None
    assert_tensors_equal(masters_of(model, opt), inc.masters)
    assert_tensors_equal(params_of(model), inc.params)
    no_resync_step(model, opt, 5)


def test_interpolate_fp64_computes_in_fp64() -> None:
    model = make_model(torch.float64)
    gen = torch.Generator().manual_seed(3)

    def perturb() -> None:
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn(p.shape, generator=gen, dtype=torch.float64) * 1e-3)

    perturb()  # values not representable in fp32: an fp32 computation would be visibly off
    inc = capture(model, None)
    perturb()
    cand = capture(model, None)
    interpolate(model, inc, cand, 0.0)
    assert_tensors_equal(params_of(model), inc.params)
    interpolate(model, inc, cand, 1.0)
    assert_tensors_equal(params_of(model), cand.params)
    interpolate(model, inc, cand, 0.5)
    for n, p in model.named_parameters():
        assert p.dtype == torch.float64
        mid = (inc.params[n] + cand.params[n]) / 2
        torch.testing.assert_close(p.detach(), mid, rtol=1e-12, atol=1e-15)
        # fp32 compute would miss by ~1e-9 here: the tolerance above rules it out.
        fp32 = torch.lerp(inc.params[n].float(), cand.params[n].float(), 0.5).double()
        assert float((fp32 - mid).abs().max()) > 1e-12, n


def make_bias_model(dtype: torch.dtype = torch.float32) -> nn.Module:
    """Tiny Llama with ``mlp_bias=True`` (gate/up/down biases, zero-initialised by transformers)."""
    cfg = tiny_llama_config(VOCAB)
    cfg.mlp_bias = True
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to(dtype)
    model.train()
    return model


@pytest.mark.parametrize("source", ["params", "channel_state"])
def test_mlp_bias_damage_and_revert_with_masters(source: str) -> None:
    model = make_bias_model(torch.bfloat16)
    mlp = model.model.layers[0].mlp
    assert mlp.gate_proj.bias is not None and mlp.up_proj.bias is not None and mlp.down_proj.bias is not None
    opt = hooked(model)
    inc = capture(model, None, optimizer=opt, channel_layers=[0] if source == "channel_state" else None)
    with torch.no_grad():  # constructed displacement: only bias entries of channel 3 (and down bias)
        mlp.gate_proj.bias[3] += 3.0
        mlp.up_proj.bias[3] += 4.0
        mlp.down_proj.bias[0] += 2.0  # owned by no channel: never reverted
    acts = {0: torch.full((INTER,), 2.0)}
    hit, quiet = (
        ChannelGroup("hit", ChannelSelection({0: [3]})),
        ChannelGroup("quiet", ChannelSelection({0: [5]})),
    )
    damage = group_damage(inc, model, acts, [quiet, hit])
    assert damage == {"quiet": 0.0, "hit": pytest.approx(5.0 * 2.0, rel=1e-6)}  # ||(3, 4)|| * 2
    assert [g.name for g in channel_groups_by_damage(inc, model, acts, [quiet, hit])] == ["hit", "quiet"]
    pre_params = snapshot(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt.state_dict()  # reconcile the external bias writes, as revert_groups does
        pre_masters = masters_of(model, opt)
    reports = revert_groups(model, inc, [hit], optimizer=opt)
    assert reports[0].elements_changed == {0: 2}  # the two bias entries
    assert verify_modification_scope(pre_params, model, hit.selection, model).ok
    assert inc.masters is not None
    mask = channel_mask(model, hit.selection)
    masters = masters_of(model, opt)
    for n, p in model.named_parameters():
        owned = mask.get(n)
        if owned is None:
            assert torch.equal(masters[n], pre_masters[n]), n
            continue
        assert torch.equal(masters[n], torch.where(owned, inc.masters[n], pre_masters[n])), n
        assert torch.equal(p.detach()[owned], inc.params[n][owned]), n
        assert torch.equal(p.detach()[~owned], pre_params[n][~owned]), n
    assert float(mlp.gate_proj.bias[3]) == float(inc.params["model.layers.0.mlp.gate_proj.bias"][3])
    assert float(mlp.down_proj.bias[0]) == float(pre_params["model.layers.0.mlp.down_proj.bias"][0])
    no_resync_step(model, opt, 60)
