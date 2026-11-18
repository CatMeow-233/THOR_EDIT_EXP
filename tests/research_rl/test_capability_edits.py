"""Tests for research_rl.capability.edits: declarative edits, the version store and its safety.

Tiny CPU Llama models only (``research_rl.models.tiny``). The gate tests use a real
:class:`CapabilityProbe` with a cheap weight-sensitive stub score function (no generation): transfer
items score 1 iff channel 0 of layer 0 is ablated (and the stub's ``transfer_ok`` switch is on),
retention items always score 1, so a verdict's outcome is known in advance.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from transformers import LlamaForCausalLM

from research_rl.capability import edits as edits_mod
from research_rl.capability.edits import (
    CommitRefusedError,
    EditError,
    EditOp,
    EditRecord,
    StoreError,
    Version,
    VersionStore,
)
from research_rl.capability.gate import GateConfig
from research_rl.capability.probe import CapabilityProbe
from research_rl.capability.stats import AlphaLedger
from research_rl.capability.tasks import CompositionSplits, ItemSampler, OperatorSet
from research_rl.capability.types import GateVerdict, Item, SliceTest, Split
from research_rl.models.ffn_channels import ChannelSelection, ChannelState, LayerChannels, extract_channels
from research_rl.models.model_interface import fingerprint
from research_rl.models.tiny import build_char_tokenizer, build_tiny_llama, tiny_llama_config

T_SLICE = "t_rule/unseen_pairs"
R_SLICE = "retention/atomic"
N_T, N_R = 20, 150


# ============================================================================== fixtures / helpers
@pytest.fixture(scope="module")
def vocab() -> int:
    return len(build_char_tokenizer())


@pytest.fixture(scope="module")
def sampler() -> ItemSampler:
    splits = CompositionSplits.build(OperatorSet(seed=0, num_operators=10), seed=0)
    return ItemSampler(splits, root_seed=11)


@pytest.fixture
def model(vocab: int) -> nn.Module:
    return build_tiny_llama(vocab, seed=0)


def _params(m: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in m.named_parameters()}


def _assert_bitwise(m: nn.Module, ref: dict[str, torch.Tensor]) -> None:
    now = dict(m.named_parameters())
    assert set(now) == set(ref)
    for n, v in ref.items():
        assert now[n].dtype == v.dtype
        assert torch.equal(now[n].detach(), v), n


def _ffn(m: nn.Module, layer: int) -> Any:
    return m.model.layers[layer].mlp  # type: ignore[index, union-attr]


class SwitchScore:
    """Transfer items: 1.0 iff layer-0 channel-0 down column is zero and ``transfer_ok``; retention
    items: always 1.0. Records the model fingerprint of every call."""

    def __init__(self) -> None:
        self.transfer_ok = True
        self.calls: list[str] = []

    def __call__(self, m: nn.Module, items: Sequence[Item]) -> list[float]:
        ablated = bool((_ffn(m, 0).down_proj.weight[:, 0] == 0).all())
        self.calls.append(fingerprint(m))
        return [1.0 if it.split is Split.RETENTION or (ablated and self.transfer_ok) else 0.0 for it in items]


def _probe(sampler: ItemSampler, score: SwitchScore) -> CapabilityProbe:
    return CapabilityProbe(
        sampler,
        score,
        transfer_slices=(T_SLICE,),
        retention_slices=(R_SLICE,),
        items_per_transfer_slice=N_T,
        items_per_retention_slice=N_R,
    )


def _config(schedule: str = "per_decision", alpha: float = 0.05) -> GateConfig:
    cfg = GateConfig(
        alpha=alpha,
        alpha_schedule=schedule,
        retention_margin=0.1,
        items_per_transfer_slice=N_T,
        items_per_retention_slice=N_R,
    )
    horizon = cfg.retention_horizon(n=N_R)
    assert horizon is None or horizon >= 4  # a clean retention slice passes the verdicts tested here
    return cfg


def _edit(op: str, selection: dict[int, list[int]], *, note: str = "", **params: Any) -> EditRecord:
    return EditRecord(op=op, selection=selection, params=params, note=note)  # type: ignore[arg-type]


def _store(tmp_path: Path, m: nn.Module, **kw: Any) -> VersionStore:
    return VersionStore(tmp_path / "store", m, **kw)


# ============================================================================== EditRecord
def test_edit_record_normalises_and_round_trips() -> None:
    e = EditRecord(op="scale", selection={"1": [5, 2], 0: (3,)}, params={"factor": 2}, note="n")
    assert e.op is EditOp.SCALE
    assert e.selection == {0: [3], 1: [2, 5]}
    assert e.params == {"factor": 2.0, "up": False}
    assert isinstance(e.params["factor"], float)
    assert EditRecord.from_json(e.to_json()) == e
    assert EditRecord.from_dict(json.loads(json.dumps(e.to_dict()))) == e
    for op, params in (
        ("ablate", {}),
        ("revert", {"version": "a" * 64}),
        ("interpolate", {"version": "b" * 64, "lam": 1}),
        ("apply_delta", {"delta": "c" * 64}),
    ):
        r = _edit(op, {0: [1]}, **params)
        assert EditRecord.from_json(r.to_json()) == r
    assert _edit("apply_delta", {0: [1]}, delta="c" * 64).params == {"delta": "c" * 64, "factor": 1.0}
    assert _edit("interpolate", {0: [1]}, version="b" * 64, lam=1).params["lam"] == 1.0


@pytest.mark.parametrize(
    ("op", "selection", "params"),
    [
        ("exec", {0: [1]}, {}),  # no free-form ops
        ("ablate", {0: [1]}, {"code": "print(1)"}),  # no extra params
        ("ablate", {}, {}),  # empty selection
        ("ablate", {0: []}, {}),
        ("ablate", {0: [1, 1]}, {}),  # duplicates
        ("ablate", {0: [-1]}, {}),
        ("ablate", {0: [True]}, {}),
        ("ablate", {0: "12"}, {}),
        ("ablate", {"x": [1]}, {}),
        ("ablate", {0: [1.5]}, {}),
        ("scale", {0: [1]}, {}),  # missing factor
        ("scale", {0: [1]}, {"factor": float("nan")}),
        ("scale", {0: [1]}, {"factor": float("inf")}),
        ("scale", {0: [1]}, {"factor": True}),
        ("scale", {0: [1]}, {"factor": "2"}),
        ("scale", {0: [1]}, {"factor": 2.0, "up": 1}),
        ("revert", {0: [1]}, {"version": "../../etc/passwd"}),
        ("revert", {0: [1]}, {"version": "A" * 64}),
        ("revert", {0: [1]}, {"version": "a" * 63}),
        ("revert", {0: [1]}, {"version": "a" * 64, "lam": 0.5}),
        ("interpolate", {0: [1]}, {"version": "a" * 64}),
        ("interpolate", {0: [1]}, {"version": "a" * 64, "lam": 1.5}),
        ("interpolate", {0: [1]}, {"version": "a" * 64, "lam": -0.1}),
        ("apply_delta", {0: [1]}, {"delta": "x/y"}),
        ("apply_delta", {0: [1]}, {"delta": "a" * 64, "factor": None}),
    ],
)
def test_edit_record_rejects_invalid(op: str, selection: Any, params: Any) -> None:
    with pytest.raises(EditError):
        EditRecord(op=op, selection=selection, params=params)  # type: ignore[arg-type]


def test_edit_record_rejects_bad_fields() -> None:
    good = _edit("ablate", {0: [1]}).to_dict()
    for bad in (
        {**good, "extra": 1},
        {k: v for k, v in good.items() if k != "params"},
        {**good, "note": 3},
        {**good, "note": "x" * (edits_mod.MAX_TEXT + 1)},
        {**good, "created": "yesterday"},
        {**good, "created": "2026-01-01T00:00:00"},  # no timezone
        {**good, "params": [1]},
        [good],
    ):
        with pytest.raises(EditError):
            EditRecord.from_dict(bad)
    with pytest.raises(EditError):
        EditRecord.from_json("{not json")


# ============================================================================== each op on a tiny Llama
def test_ablate_effect_scope_and_diff(tmp_path: Path, model: nn.Module) -> None:
    store = _store(tmp_path, model)
    root = store.init_from_model("start")
    before = _params(model)
    staged = store.stage(_edit("ablate", {0: [1, 4], 1: [7]}))
    for layer, chans in ((0, [1, 4]), (1, [7])):
        mlp = _ffn(model, layer)
        assert (mlp.gate_proj.weight[chans] == 0).all()
        assert (mlp.up_proj.weight[chans] == 0).all()
        assert (mlp.down_proj.weight[:, chans] == 0).all()
    v1 = store.commit(staged, override="manual ablation")
    assert v1.parent == root.id and v1.override == "manual ablation" and v1.verdict is None
    assert v1.scope is not None and v1.scope["ok"] and v1.scope["outside_unchanged"]
    hidden = model.config.hidden_size
    assert v1.scope["changed_allowed"] == 3 * 3 * hidden  # gate row + up row + down col per channel
    assert store.diff(root.id, v1.id) == {0: [1, 4], 1: [7]}
    assert store.diff(v1.id, v1.id) == {}
    # every element outside the selected slices kept its bits
    now = dict(model.named_parameters())
    for name, value in before.items():
        if ".mlp." not in name:
            assert torch.equal(now[name], value), name
    keep = [c for c in range(model.config.intermediate_size) if c not in (1, 4)]
    assert torch.equal(
        _ffn(model, 0).down_proj.weight[:, keep], before["model.layers.0.mlp.down_proj.weight"][:, keep]
    )


@pytest.mark.parametrize("up", [False, True])
def test_scale_effect_and_scope(tmp_path: Path, model: nn.Module, up: bool) -> None:
    store = _store(tmp_path, model)
    store.init_from_model()
    before = _params(model)
    v = store.commit(store.stage(_edit("scale", {1: [0, 9]}, factor=-1.5, up=up)), override="test")
    p = "model.layers.1.mlp."
    w = dict(model.named_parameters())
    torch.testing.assert_close(
        w[p + "down_proj.weight"][:, [0, 9]], before[p + "down_proj.weight"][:, [0, 9]] * -1.5
    )
    if up:
        torch.testing.assert_close(
            w[p + "up_proj.weight"][[0, 9]], before[p + "up_proj.weight"][[0, 9]] * -1.5
        )
    else:
        assert torch.equal(w[p + "up_proj.weight"], before[p + "up_proj.weight"])
    assert torch.equal(w[p + "gate_proj.weight"], before[p + "gate_proj.weight"])
    others = [c for c in range(model.config.intermediate_size) if c not in (0, 9)]
    assert torch.equal(w[p + "down_proj.weight"][:, others], before[p + "down_proj.weight"][:, others])
    for name, value in before.items():
        if not name.startswith(p):
            assert torch.equal(w[name], value), name
    hidden = model.config.hidden_size
    assert v.scope is not None and v.scope["changed_allowed"] == 2 * hidden * (2 if up else 1)
    # factor 1 is a bitwise no-op
    fp = fingerprint(model)
    staged = store.stage(_edit("scale", {0: [3]}, factor=1.0, up=True))
    assert staged.fingerprint == fp
    store.discard(staged)


def test_revert_and_interpolate(tmp_path: Path, model: nn.Module) -> None:
    store = _store(tmp_path, model)
    root = store.init_from_model()
    root_w = _params(model)
    v1 = store.commit(store.stage(_edit("scale", {0: [2, 3, 5]}, factor=3.0, up=True)), override="t")
    v1_w = _params(model)
    name = "model.layers.0.mlp.down_proj.weight"

    # revert a subset: copies (bitwise) of the root's slices, the rest stays scaled
    staged = store.stage(_edit("revert", {0: [2, 5]}, version=root.id))
    w = dict(model.named_parameters())
    assert torch.equal(w[name][:, [2, 5]], root_w[name][:, [2, 5]])
    assert torch.equal(
        w["model.layers.0.mlp.up_proj.weight"][[2, 5]], root_w["model.layers.0.mlp.up_proj.weight"][[2, 5]]
    )
    assert torch.equal(w[name][:, 3], v1_w[name][:, 3])
    reverted = fingerprint(model)
    store.discard(staged)
    _assert_bitwise(model, v1_w)

    # lam = 0 is exactly revert, lam = 1 a bitwise no-op, lam = 0.5 the midpoint
    staged = store.stage(_edit("interpolate", {0: [2, 5]}, version=root.id, lam=0.0))
    assert staged.fingerprint == reverted
    store.discard(staged)
    staged = store.stage(_edit("interpolate", {0: [2, 5]}, version=root.id, lam=1.0))
    assert staged.fingerprint == v1.fingerprint
    store.discard(staged)
    staged = store.stage(_edit("interpolate", {0: [2, 5]}, version=root.id, lam=0.5))
    w = dict(model.named_parameters())
    torch.testing.assert_close(w[name][:, [2, 5]], 0.5 * (root_w[name][:, [2, 5]] + v1_w[name][:, [2, 5]]))
    assert torch.equal(w[name][:, 3], v1_w[name][:, 3])
    v2 = store.commit(staged, override="t")
    assert store.diff(v1.id, v2.id) == {0: [2, 5]}
    with pytest.raises(EditError):  # unknown reference version
        store.stage(_edit("revert", {0: [1]}, version="f" * 64))


def test_apply_delta(tmp_path: Path, model: nn.Module) -> None:
    store = _store(tmp_path, model)
    root = store.init_from_model()
    root_w = _params(model)
    v1 = store.commit(store.stage(_edit("scale", {0: [1, 2], 1: [4]}, factor=2.0, up=True)), override="t")
    v1_w = _params(model)
    delta_id = store.delta_between(root.id, v1.id)
    assert store.deltas() == [delta_id]
    assert store.delta_between(root.id, v1.id) == delta_id  # content-addressed, idempotent
    delta = store.load_delta(delta_id)
    assert delta.selection().to_dict() == {0: [1, 2], 1: [4]}
    store.restore(root.id)
    _assert_bitwise(model, root_w)

    staged = store.stage(_edit("apply_delta", {0: [1, 2], 1: [4]}, delta=delta_id))
    for name in v1_w:
        torch.testing.assert_close(dict(model.named_parameters())[name], v1_w[name])
    store.discard(staged)
    staged = store.stage(_edit("apply_delta", {0: [2]}, delta=delta_id, factor=0.5))
    d = "model.layers.0.mlp.down_proj.weight"
    w = dict(model.named_parameters())
    torch.testing.assert_close(w[d][:, 2], 0.5 * (root_w[d][:, 2] + v1_w[d][:, 2]))
    assert torch.equal(w[d][:, 1], root_w[d][:, 1])  # not selected: untouched
    store.discard(staged)
    with pytest.raises(EditError):  # channel not in the delta
        store.stage(_edit("apply_delta", {0: [7]}, delta=delta_id))
    with pytest.raises(EditError):  # unknown delta
        store.stage(_edit("apply_delta", {0: [1]}, delta="e" * 64))
    with pytest.raises(EditError):
        store.delta_between(root.id, root.id)
    hidden = model.config.hidden_size
    good = LayerChannels(
        channels=torch.tensor([0]),
        gate=torch.zeros(1, hidden),
        up=torch.zeros(1, hidden),
        down=torch.zeros(hidden, 1),
    )
    assert len(store.add_delta(ChannelState({0: good}))) == 64
    for bad in (
        ChannelState({}),
        ChannelState({5: good}),  # no such layer
        ChannelState({0: LayerChannels(torch.tensor([0]), good.gate, good.up, torch.zeros(hidden, 2))}),
        ChannelState({0: LayerChannels(torch.tensor([0]), good.gate.long(), good.up, good.down)}),
        ChannelState({0: LayerChannels(torch.tensor([999]), good.gate, good.up, good.down)}),
    ):
        with pytest.raises(EditError):
            store.add_delta(bad)


# ============================================================================== stage / discard / restore
def test_discard_restore_bitwise_and_stage_rules(tmp_path: Path, model: nn.Module) -> None:
    store = _store(tmp_path, model)
    with pytest.raises(StoreError):  # not initialised
        store.stage(_edit("ablate", {0: [1]}))
    root = store.init_from_model()
    with pytest.raises(StoreError):
        store.init_from_model()
    root_w = _params(model)
    staged = store.stage([_edit("ablate", {0: [1]}), _edit("scale", {1: [2]}, factor=0.5)])
    assert staged.selection.to_dict() == {0: [1], 1: [2]}
    with pytest.raises(StoreError):  # one staged edit at a time
        store.stage(_edit("ablate", {0: [2]}))
    with pytest.raises(StoreError):  # restore while staged
        store.restore(root.id)
    store.discard(staged)
    _assert_bitwise(model, root_w)
    assert staged.status == "discarded"
    with pytest.raises(StoreError):
        store.discard(staged)
    with pytest.raises(StoreError):
        store.commit(staged, override="x")

    v1 = store.commit(store.stage(_edit("ablate", {1: [3, 4]})), override="t")
    v1_w = _params(model)
    v2 = store.commit(store.stage(_edit("scale", {0: [0]}, factor=4.0)), override="t")
    assert store.history() == [root, v1, v2]
    store.restore(root.id)
    _assert_bitwise(model, root_w)
    assert store.head == root.id and fingerprint(model) == root.fingerprint
    store.restore(v1.id)
    _assert_bitwise(model, v1_w)
    assert store.history() == [root, v1]
    # branch from v1
    v3 = store.commit(store.stage(_edit("ablate", {0: [9]})), override="branch")
    assert v3.parent == v1.id and store.history(v2.id) == [root, v1, v2]
    assert [v.id for v in store.versions()] == [root.id, v1.id, v2.id, v3.id]

    # the model must be at the head to stage
    with torch.no_grad():
        model.lm_head.weight[0, 0] += 1.0  # type: ignore[index, operator]
    with pytest.raises(StoreError):
        store.stage(_edit("ablate", {0: [1]}))
    # ... and restore cannot repair the frame (outside the channel layers): refused, nothing written
    drifted = _params(model)
    with pytest.raises(StoreError, match="outside the channel layers"):
        store.restore(root.id)
    _assert_bitwise(model, drifted)


def test_stage_is_atomic_on_failure(
    tmp_path: Path, model: nn.Module, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, model)
    store.init_from_model()
    root_w = _params(model)
    with pytest.raises(EditError):  # validation failure: channel out of range, nothing written
        store.stage([_edit("ablate", {0: [1]}), _edit("ablate", {0: [10_000]})])
    _assert_bitwise(model, root_w)
    original = VersionStore._apply
    calls = {"n": 0}

    def flaky(self: VersionStore, edit: EditRecord) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        original(self, edit)

    monkeypatch.setattr(VersionStore, "_apply", flaky)
    with pytest.raises(RuntimeError, match="boom"):
        store.stage([_edit("ablate", {0: [1]}), _edit("ablate", {1: [2]})])
    _assert_bitwise(model, root_w)
    assert store.staged is None


def test_bf16_scale_discard_restore_bitwise(tmp_path: Path, vocab: int) -> None:
    m = build_tiny_llama(vocab, seed=1).to(torch.bfloat16)
    store = _store(tmp_path, m)
    root = store.init_from_model()
    root_w = _params(m)
    staged = store.stage(_edit("scale", {0: [1]}, factor=0.3, up=True))
    assert _ffn(m, 0).down_proj.weight.dtype == torch.bfloat16
    store.discard(staged)
    _assert_bitwise(m, root_w)
    v1 = store.commit(store.stage(_edit("scale", {0: [1]}, factor=0.3)), override="t")
    v1_w = _params(m)
    store.restore(root.id)
    _assert_bitwise(m, root_w)
    store.restore(v1.id)
    _assert_bitwise(m, v1_w)


# ============================================================================== evaluate / commit
def test_evaluate_uses_parent_as_incumbent_and_own_ledger(
    tmp_path: Path, model: nn.Module, sampler: ItemSampler
) -> None:
    score = SwitchScore()
    probe = _probe(sampler, score)
    store = _store(tmp_path, model)
    root = store.init_from_model()
    staged = store.stage(_edit("ablate", {0: [0]}))
    verdict = store.evaluate(staged, _config("spending"), probe)
    assert verdict.transfer_ok and verdict.retention_ok
    assert score.calls == [staged.fingerprint, root.fingerprint]  # candidate, then parent (swapped)
    assert fingerprint(model) == staged.fingerprint
    ledger = store.ledger
    assert ledger is not None and ledger.count == 1 and store.evaluation_count == 1
    # ``store.ledger`` is a copy: advancing it (as a training run would advance its own ledger)
    # neither spends the store's alpha nor changes what is persisted
    ledger.next_alpha()
    ledger.next_alpha()
    own = store.ledger
    assert own is not None and own.count == 1 and own is not ledger
    assert json.loads((tmp_path / "store" / "ledger.json").read_text())["ledger"]["count"] == 1
    assert verdict.transfer.alpha == AlphaLedger(0.05, "spending").alpha_at(1)
    assert json.loads((tmp_path / "store" / "ledger.json").read_text())["ledger"]["count"] == 1
    with pytest.raises(ValueError):  # the ledger is pinned to (0.05, spending)
        store.evaluate(staged, _config("spending", alpha=0.1), probe)
    with pytest.raises(ValueError):
        store.evaluate(staged, _config("per_decision"), probe)
    with pytest.raises(ValueError):
        store.evaluate(staged, GateConfig(enabled=False, alpha_schedule="spending"), probe)
    v1 = store.commit(staged)
    assert v1.passed and v1.override is None and v1.verdict == json.loads(json.dumps(verdict.to_dict()))
    assert v1.evaluation is not None and v1.evaluation["evaluation_index"] == 0

    # the ledger and the evaluation counter survive a reopen: the next verdict spends alpha_2
    reopened = VersionStore(tmp_path / "store", model)
    staged2 = reopened.stage(_edit("ablate", {0: [1]}))
    verdict2 = reopened.evaluate(staged2, _config("spending"), probe)
    assert verdict2.transfer.alpha == AlphaLedger(0.05, "spending").alpha_at(2)
    assert verdict2.decision_index == 1
    assert reopened.evaluation_count == 2
    assert score.calls[-1] == v1.fingerprint  # incumbent = parent v1
    reopened.discard(staged2)


class ItemLog(SwitchScore):
    """:class:`SwitchScore` that also records the uids of every scored batch."""

    def __init__(self) -> None:
        super().__init__()
        self.uids: list[tuple[str, ...]] = []

    def __call__(self, m: nn.Module, items: Sequence[Item]) -> list[float]:
        self.uids.append(tuple(it.uid for it in items))
        return super().__call__(m, items)


def _evaluated_uids(
    store: VersionStore, probe: CapabilityProbe, score: ItemLog, n: int
) -> list[tuple[str, ...]]:
    out = []
    for k in range(n):
        score.uids.clear()
        staged = store.stage(_edit("scale", {0: [k]}, factor=0.5))
        store.evaluate(staged, _config("spending"), probe)
        assert score.uids[0] == score.uids[1]  # candidate and incumbent on the same items
        out.append(score.uids[0])
        store.discard(staged)
    return out


def test_each_store_draws_its_own_gate_items(tmp_path: Path, vocab: int, sampler: ItemSampler) -> None:
    # regression: a new store (the remedy CAPABILITY.md 4.3 names) must not replay the gate items
    # of another store - or of a training run's decisions - at the same evaluation index
    score = ItemLog()
    probe = _probe(sampler, score)
    a = VersionStore(tmp_path / "a", build_tiny_llama(vocab, seed=0))
    b = VersionStore(tmp_path / "b", build_tiny_llama(vocab, seed=0))
    assert a.gate_namespace is not None and b.gate_namespace is not None
    assert a.gate_namespace != b.gate_namespace and a.gate_namespace.startswith("edit/")
    a.init_from_model()
    b.init_from_model()
    uids_a = _evaluated_uids(a, probe, score, 3)
    uids_b = _evaluated_uids(b, probe, score, 3)
    run_items = [tuple(it.uid for g in probe.gate_items(k, 0).values() for it in g) for k in range(3)]
    for k in range(3):
        assert uids_a[k] != uids_b[k]
        assert uids_a[k] != run_items[k] and uids_b[k] != run_items[k]
        assert uids_a[k] == tuple(
            it.uid for g in probe.gate_items(k, 0, namespace=a.gate_namespace).values() for it in g
        )
    assert len(set(uids_a)) == 3  # fresh per evaluation within a store
    # the namespace is persisted: a reopened store continues its own stream
    reopened = VersionStore(tmp_path / "a", a.model)
    assert reopened.gate_namespace == a.gate_namespace
    meta = json.loads((tmp_path / "a" / "store.json").read_text())
    assert reopened.gate_namespace == f"edit/{meta['gate_nonce']}"
    score.uids.clear()
    staged = reopened.stage(_edit("ablate", {0: [0]}))
    reopened.evaluate(staged, _config("spending"), probe)
    assert score.uids[0] == tuple(
        it.uid for g in probe.gate_items(3, 0, namespace=a.gate_namespace).values() for it in g
    )
    version = reopened.commit(staged)
    assert version.evaluation is not None and version.evaluation["gate_namespace"] == a.gate_namespace


def test_store_gate_nonce_validation_and_legacy_store(
    tmp_path: Path, model: nn.Module, sampler: ItemSampler, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, model)
    store.init_from_model()
    path = tmp_path / "store" / "store.json"
    meta = json.loads(path.read_text())
    assert set(meta) == {"format", "channel_layers", "signature", "created", "gate_nonce"}
    for bad in ("", "XYZ", meta["gate_nonce"].upper(), None, 7):
        path.write_text(json.dumps({**meta, "gate_nonce": bad}))
        with pytest.raises(StoreError, match="gate_nonce"):
            VersionStore(tmp_path / "store", model)
    # a legacy store (store.json without a nonce) still opens and keeps the training-run key gate/0
    legacy = {k: v for k, v in meta.items() if k != "gate_nonce"}
    path.write_text(json.dumps(legacy))
    old = VersionStore(tmp_path / "store", model)
    assert old.gate_namespace is None
    score = ItemLog()
    probe = _probe(sampler, score)
    staged = old.stage(_edit("ablate", {0: [0]}))
    old.evaluate(staged, _config("spending"), probe)
    assert score.uids[0] == tuple(it.uid for g in probe.gate_items(0, 0).values() for it in g)
    version = old.commit(staged)
    assert version.evaluation is not None and version.evaluation["gate_namespace"] is None
    # a version evaluated on another namespace is inconsistent with the store
    path.write_text(json.dumps({**legacy, "gate_nonce": "0" * 32}))
    with pytest.raises(StoreError, match="gate namespace"):
        VersionStore(tmp_path / "store", model)
    # an invalid nonce from the factory is refused before anything is written
    monkeypatch.setattr(edits_mod, "_new_gate_nonce", lambda: "not-hex")
    with pytest.raises(StoreError, match="gate nonce"):
        VersionStore(tmp_path / "fresh", model)
    assert not (tmp_path / "fresh").exists()


def test_commit_refused_without_passing_verdict_or_override(
    tmp_path: Path, model: nn.Module, sampler: ItemSampler
) -> None:
    score = SwitchScore()
    probe = _probe(sampler, score)
    store = _store(tmp_path, model)
    root = store.init_from_model()
    cfg = _config()

    staged = store.stage(_edit("scale", {0: [0]}, factor=0.5))
    with pytest.raises(CommitRefusedError, match="never evaluated"):
        store.commit(staged)
    for empty in ("", "   "):
        with pytest.raises(ValueError):
            store.commit(staged, override=empty)
    verdict = store.evaluate(staged, cfg, probe)  # channel not ablated: no transfer gain
    assert not verdict.transfer_ok and verdict.retention_ok
    with pytest.raises(CommitRefusedError, match="did not pass"):
        store.commit(staged)
    assert store.staged is staged and store.head == root.id
    store.discard(staged)

    # pass, then fail on the same staged edit: the LAST evaluation decides
    staged = store.stage(_edit("ablate", {0: [0]}))
    assert store.evaluate(staged, cfg, probe).transfer_ok
    score.transfer_ok = False
    assert not store.evaluate(staged, cfg, probe).transfer_ok
    with pytest.raises(CommitRefusedError):
        store.commit(staged)
    v = store.commit(staged, override="reviewed by hand")
    assert v.override == "reviewed by hand" and not v.passed and v.verdict is not None
    assert len(store.versions()) == 2
    # only versions with a passing verdict or an override can exist
    with pytest.raises(EditError):
        Version.build(
            parent=v.parent,
            edits=v.edits,
            verdict=v.verdict,
            evaluation=v.evaluation,
            override=None,
            fingerprint=v.fingerprint,
            frame_fingerprint=v.frame_fingerprint,
            state=v.state,
            scope=v.scope,
            note=v.note,
            created=v.created,
        )

    # a staged edit whose weights were changed behind the store's back cannot be evaluated/committed
    staged = store.stage(_edit("ablate", {1: [1]}))
    with torch.no_grad():
        _ffn(model, 1).down_proj.weight[0, 5] += 1.0
    with pytest.raises(StoreError):
        store.evaluate(staged, cfg, probe)
    with pytest.raises(StoreError):
        store.commit(staged, override="x")


def test_commit_refused_on_scope_violation(
    tmp_path: Path, model: nn.Module, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, model)
    store.init_from_model()
    original = VersionStore._apply

    def leaky(self: VersionStore, edit: EditRecord) -> None:  # a buggy op that also writes channel 9
        original(self, edit)
        with torch.no_grad():
            _ffn(self.model, 0).down_proj.weight[0, 9] += 1.0

    monkeypatch.setattr(VersionStore, "_apply", leaky)
    staged = store.stage(_edit("ablate", {0: [1]}))
    with pytest.raises(CommitRefusedError, match="scope"):
        store.commit(staged, override="x")
    assert len(store.versions()) == 1


# ============================================================================== persistence and safety
def test_reopen_reproduces_history_and_restore(tmp_path: Path, model: nn.Module, vocab: int) -> None:
    store = _store(tmp_path, model)
    root = store.init_from_model("root note")
    v1 = store.commit(store.stage(_edit("ablate", {0: [3]}, note="first")), override="t", note="v1")
    v2 = store.commit(store.stage(_edit("scale", {1: [6]}, factor=2.5, up=True)), override="t")
    v2_w = _params(model)
    root_w = _params(build_tiny_llama(vocab, seed=0))
    assert not list((tmp_path / "store" / "tmp").iterdir())  # no temporary files left behind

    fresh = build_tiny_llama(vocab, seed=0)  # same frame, at the root
    again = VersionStore(tmp_path / "store", fresh)
    assert again.head == v2.id
    assert again.history() == [root, v1, v2] == store.history()
    assert again.versions() == store.versions()
    assert again.get(v1.id).edits[0].note == "first" and again.get(v1.id).note == "v1"
    with pytest.raises(StoreError):  # the fresh model is at the root, not at the head
        again.stage(_edit("ablate", {0: [1]}))
    again.restore(v2.id)
    _assert_bitwise(fresh, v2_w)
    again.restore(root.id)
    _assert_bitwise(fresh, root_w)
    assert VersionStore(tmp_path / "store", fresh).head == root.id

    # a model with another shape is rejected, and a different frame cannot be restored into
    with pytest.raises(StoreError):
        VersionStore(tmp_path / "store", build_tiny_llama(vocab, seed=0, intermediate_size=48))
    other = build_tiny_llama(vocab, seed=5)
    other_w = _params(other)
    with pytest.raises(StoreError):
        VersionStore(tmp_path / "store", other).restore(v2.id)
    _assert_bitwise(other, other_w)


def test_layer_keys_hash_consistently(tmp_path: Path, vocab: int) -> None:
    m = build_tiny_llama(
        vocab,
        seed=0,
        num_hidden_layers=11,
        hidden_size=8,
        intermediate_size=12,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    store = _store(tmp_path, m, channel_layers=[10, 2])
    assert store.channel_layers == [2, 10]
    store.init_from_model()
    v = store.commit(store.stage(_edit("ablate", {10: [1], 2: [3]})), override="t")
    with pytest.raises(EditError):  # layer 3 is not versioned by this store
        store.stage(_edit("ablate", {3: [1]}))
    again = VersionStore(tmp_path / "store", m)  # channel layers adopted from store.json
    assert again.channel_layers == [2, 10] and again.get(v.id) == v
    assert again.diff(v.parent or "", v.id) == {2: [3], 10: [1]}
    with pytest.raises(StoreError):
        VersionStore(tmp_path / "store", m, channel_layers=[2])


def test_only_safetensors_are_loaded(
    tmp_path: Path, model: nn.Module, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, model)
    root = store.init_from_model()
    v1 = store.commit(store.stage(_edit("ablate", {0: [2]})), override="t")
    sdir = tmp_path / "store"

    def no_torch_load(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("torch.load must never be called by the store")

    monkeypatch.setattr(torch, "load", no_torch_load)
    monkeypatch.setattr(pickle, "load", no_torch_load)
    VersionStore(sdir, model).restore(root.id)  # normal open/restore never unpickles

    evil = sdir / "weights.pt"
    evil.write_bytes(pickle.dumps({"x": 1}))
    with pytest.raises(StoreError, match="safetensors"):
        VersionStore(sdir, model)
    evil.unlink()

    disguised = sdir / "states" / ("d" * 64 + ".safetensors")
    disguised.write_bytes(pickle.dumps({"x": 1}))
    with pytest.raises(StoreError, match="not a valid safetensors"):
        VersionStore(sdir, model)
    disguised.unlink()

    stray = sdir / "deltas" / ("e" * 64 + ".pt")
    stray.write_bytes(b"\x80\x04")
    with pytest.raises(StoreError):
        VersionStore(sdir, model)
    stray.unlink()

    link = sdir / "versions" / ("f" * 64 + ".json")
    link.symlink_to(sdir / "versions" / f"{v1.id}.json")
    with pytest.raises(StoreError, match=r"symlink|unexpected"):
        VersionStore(sdir, model)
    link.unlink()

    # tampering with the tensor bytes of a state file: content hash mismatch at use
    state_file = sdir / "states" / f"{v1.state}.safetensors"
    raw = bytearray(state_file.read_bytes())
    raw[-1] ^= 0xFF
    state_file.write_bytes(bytes(raw))
    reopened = VersionStore(sdir, model)
    before = _params(model)
    with pytest.raises(StoreError, match="content hash"):
        reopened.restore(v1.id)
    _assert_bitwise(model, before)


def test_tampered_fingerprint_rejected(tmp_path: Path, model: nn.Module) -> None:
    store = _store(tmp_path, model)
    root = store.init_from_model()
    v1 = store.commit(store.stage(_edit("ablate", {0: [2]})), override="t")
    sdir = tmp_path / "store"
    path = sdir / "versions" / f"{v1.id}.json"
    data = json.loads(path.read_text())
    good_text = path.read_text()

    # 1. naive tampering: the id no longer matches the content
    data["fingerprint"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(StoreError, match="hash mismatch"):
        VersionStore(sdir, model)
    path.write_text(good_text)

    # 2. consistent tampering (id recomputed, file renamed, head moved): restore refuses before writing
    forged = {k: v for k, v in data.items() if k != "id"}
    forged_id = Version.compute_id(forged)
    path.unlink()
    (sdir / "versions" / f"{forged_id}.json").write_text(json.dumps({"id": forged_id, **forged}))
    (sdir / "head.json").write_text(json.dumps({"format": edits_mod.STORE_FORMAT, "head": forged_id}))
    forged_store = VersionStore(sdir, model)
    forged_store.restore(root.id)
    at_root = _params(model)
    with pytest.raises(StoreError, match="fingerprint"):
        forged_store.restore(forged_id)
    _assert_bitwise(model, at_root)


def test_paths_stay_inside_root(tmp_path: Path, model: nn.Module) -> None:
    store = _store(tmp_path, model)
    store.init_from_model()
    for bad in ("../x", "a/b", "..", "/etc/passwd", "A" * 64, "a" * 65, ""):
        with pytest.raises(EditError):
            store.restore(bad)
        with pytest.raises(EditError):
            store.get(bad)
        with pytest.raises(EditError):
            store.load_delta(bad)
    with pytest.raises(KeyError):
        store.restore("a" * 64)
    with pytest.raises(StoreError):
        store._atomic_write(tmp_path / "outside.json", b"{}")
    assert not (tmp_path / "outside.json").exists()
    # a non-empty directory that is not a store is never adopted
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "file.txt").write_text("x")
    with pytest.raises(StoreError):
        VersionStore(tmp_path / "busy", model)


def test_atomic_write_failure_leaves_store_consistent(
    tmp_path: Path, model: nn.Module, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, model)
    root = store.init_from_model()
    staged = store.stage(_edit("ablate", {0: [2]}))

    def failing_replace(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(edits_mod.os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        store.commit(staged, override="t")
    monkeypatch.undo()
    sdir = tmp_path / "store"
    assert not list((sdir / "tmp").iterdir())
    assert not list((sdir / "versions").glob("*.json"))[1:]  # only the root record
    reopened = VersionStore(sdir, build_tiny_llama(model.config.vocab_size, seed=0))
    assert reopened.head == root.id and [v.id for v in reopened.versions()] == [root.id]
    store.discard(staged)
    assert fingerprint(model) == root.fingerprint
    assert extract_channels(model, ChannelSelection({0: [2]}))[0].down.abs().sum() > 0


def _biased_model(vocab: int) -> nn.Module:
    cfg = tiny_llama_config(vocab)
    cfg.mlp_bias = True
    with torch.random.fork_rng():
        torch.manual_seed(3)
        m = LlamaForCausalLM(cfg)
        with torch.no_grad():
            for name, p in m.named_parameters():
                if name.endswith("bias"):
                    p.normal_()  # biases initialise to zero: make ablate / scale on them visible
    return m


def test_ffn_biases_and_reference_ops_survive_reopen(tmp_path: Path, vocab: int) -> None:
    m = _biased_model(vocab)
    mlp = _ffn(m, 0)
    assert mlp.gate_proj.bias is not None and mlp.down_proj.bias is not None
    store = _store(tmp_path, m)
    root = store.init_from_model()
    root_w = _params(m)
    v1 = store.commit(store.stage(_edit("ablate", {0: [1]})), override="t")
    assert float(mlp.gate_proj.bias[1]) == 0.0 and float(mlp.up_proj.bias[1]) == 0.0
    assert torch.equal(mlp.down_proj.bias, root_w["model.layers.0.mlp.down_proj.bias"])  # owned by no channel
    v2 = store.commit(store.stage(_edit("scale", {0: [2]}, factor=2.0, up=True)), override="t")
    torch.testing.assert_close(mlp.up_proj.bias[2], 2.0 * root_w["model.layers.0.mlp.up_proj.bias"][2])
    assert torch.equal(mlp.gate_proj.bias[2], root_w["model.layers.0.mlp.gate_proj.bias"][2])
    v2_w = _params(m)
    assert v2.scope is not None and v2.scope["ok"]
    # explicit selection for delta_between: only channel 2 of the two changed channels
    delta_id = store.delta_between(root.id, v2.id, selection={0: [2]})
    assert store.load_delta(delta_id).selection().to_dict() == {0: [2]}
    v3 = store.commit(store.stage(_edit("revert", {0: [1]}, version=root.id)), override="t")
    v4 = store.commit(store.stage(_edit("apply_delta", {0: [2]}, delta=delta_id, factor=-1.0)), override="t")
    v4_w = _params(m)
    for name, value in root_w.items():  # reverted + delta undone: back to the root up to rounding
        torch.testing.assert_close(v4_w[name], value)
    fresh = _biased_model(vocab)
    again = VersionStore(tmp_path / "store", fresh)
    assert [v.id for v in again.history()] == [root.id, v1.id, v2.id, v3.id, v4.id]
    again.restore(v2.id)
    _assert_bitwise(fresh, v2_w)
    again.restore(v4.id)
    _assert_bitwise(fresh, v4_w)
    # a store whose reference target is gone is rejected on open
    (tmp_path / "store" / "deltas" / f"{delta_id}.safetensors").unlink()
    with pytest.raises(StoreError, match="missing delta"):
        VersionStore(tmp_path / "store", fresh)


# ============================================================================== review regressions
def test_mutating_records_after_stage_or_evaluate_changes_nothing(
    tmp_path: Path, model: nn.Module, sampler: ItemSampler
) -> None:
    store = _store(tmp_path, model)
    root = store.init_from_model()
    w0 = _ffn(model, 0).down_proj.weight.detach().clone()
    e = _edit("scale", {0: [1]}, factor=2.0)
    expected = e.to_dict()
    staged = store.stage(e)
    assert staged.edits[0] is not e
    # the caller's record and the StagedEdit's public copy are both mutated to other valid values
    e.params["factor"] = 5.0
    e.selection[0].append(7)
    staged.edits[0].params["factor"] = 7.0
    staged.edits[0].selection[0].append(8)
    v1 = store.commit(staged, override="t")
    assert [x.to_dict() for x in v1.edits] == [expected]
    w = _ffn(model, 0).down_proj.weight.detach()
    torch.testing.assert_close(w[:, 1], w0[:, 1] * 2.0)  # factor 2.0 was applied, not 5.0 / 7.0
    assert torch.equal(w[:, 7], w0[:, 7])  # channel 7 untouched (a copy, so bitwise)
    # returned records are copies: mutating them changes neither the store nor its history
    v1.edits[0].params["factor"] = 9.0
    assert store.get(v1.id).edits[0].params["factor"] == 2.0
    store.history()[-1].edits[0].selection[0].append(3)
    assert store.history()[-1].edits[0].selection == {0: [1]}
    reopened = VersionStore(tmp_path / "store", model)
    assert [x.to_dict() for x in reopened.get(v1.id).edits] == [expected]

    # after evaluate: mutating the public verdict / evaluation lists changes neither the commit
    # decision nor the recorded context
    score = SwitchScore()
    score.transfer_ok = False
    staged = reopened.stage(_edit("ablate", {0: [0]}))
    verdict = reopened.evaluate(staged, _config(), _probe(sampler, score))
    assert not verdict.transfer_ok
    staged.evaluations[-1]["gate_config"]["alpha"] = 0.4
    staged.evaluations[-1]["evaluation_index"] = 99
    staged.edits[0].selection[0].append(5)
    passing = GateVerdict(
        decision_index=0,
        transfer=SliceTest(T_SLICE, 1, 1.0, 1.0, True, "x", 0.05, 0.0),
        retention=(),
        alarm=False,
    )
    staged.verdicts.append(passing)  # the public list says "passed"; the store does not care
    assert staged.passed
    with pytest.raises(CommitRefusedError, match="did not pass"):
        reopened.commit(staged)
    v2 = reopened.commit(staged, override="t")
    assert v2.evaluation is not None and v2.evaluation["evaluation_index"] == 0
    assert v2.evaluation["gate_config"]["alpha"] == 0.05
    assert v2.verdict == json.loads(json.dumps(verdict.to_dict()))
    assert v2.edits[0].selection == {0: [0]}
    assert [v.id for v in VersionStore(tmp_path / "store", model).history()] == [root.id, v1.id, v2.id]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e.params.__setitem__("code", "__import__('os').system('id')"),
        lambda e: e.selection[0].append(1),
        lambda e: e.params.__setitem__("factor", float("inf")),
        lambda e: e.params.__setitem__("factor", "2"),
        lambda e: e.selection.__setitem__(-1, [0]),
    ],
)
def test_mutated_invalid_record_is_refused_before_any_write(
    tmp_path: Path, model: nn.Module, mutate: Any
) -> None:
    store = _store(tmp_path, model)
    root = store.init_from_model()
    before = _params(model)
    e = _edit("scale", {0: [1]}, factor=2.0)
    mutate(e)
    with pytest.raises(EditError):
        store.stage(e)
    with pytest.raises(EditError):
        e.copy()
    _assert_bitwise(model, before)
    assert store.staged is None
    # a non-normalised (unsorted) mutation is refused as well: the record would not round-trip
    e2 = _edit("ablate", {0: [1, 2]})
    e2.selection[0].reverse()
    with pytest.raises(EditError, match="normal form"):
        store.stage(e2)
    reopened = VersionStore(tmp_path / "store", model)
    assert [v.id for v in reopened.versions()] == [root.id]


@pytest.mark.parametrize(
    ("op", "params"),
    [
        ("scale", '{"factor": HUGE}'),
        ("interpolate", '{"version": "' + "a" * 64 + '", "lam": HUGE}'),
        ("apply_delta", '{"delta": "' + "b" * 64 + '", "factor": HUGE}'),
    ],
)
def test_huge_integers_raise_edit_error(op: str, params: str) -> None:
    huge = "1" + "0" * 400
    text = json.dumps({"op": op, "selection": {"0": [1]}, "params": "PARAMS"}).replace(
        '"PARAMS"', params.replace("HUGE", huge)
    )
    with pytest.raises(EditError, match="finite"):
        EditRecord.from_json(text)
    with pytest.raises(EditError, match="finite"):
        EditRecord.from_json(text.replace(huge, "-" + huge))


def _ledger_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((path / "ledger.json").read_text())
    return data


def test_ledger_file_is_cross_checked_against_version_records(
    tmp_path: Path, model: nn.Module, sampler: ItemSampler
) -> None:
    probe = _probe(sampler, SwitchScore())
    store = _store(tmp_path, model)
    store.init_from_model()
    cfg = _config("spending")
    # evaluate + discard, then evaluate + commit: a legitimate store reopens fine
    staged = store.stage(_edit("ablate", {0: [1]}))
    store.evaluate(staged, cfg, probe)
    store.discard(staged)
    staged = store.stage(_edit("ablate", {0: [0]}))
    assert store.evaluate(staged, cfg, probe).transfer_ok
    v1 = store.commit(staged)
    assert v1.evaluation is not None and v1.evaluation["evaluation_index"] == 1
    sdir = tmp_path / "store"
    good = _ledger_json(sdir)
    assert VersionStore(sdir, model).evaluation_count == 2

    def reopen_with(ledger: Any, evaluations: int) -> VersionStore:
        (sdir / "ledger.json").write_text(
            json.dumps({"format": edits_mod.STORE_FORMAT, "ledger": ledger, "evaluations": evaluations})
        )
        return VersionStore(sdir, model)

    def state(alpha: float, schedule: str, count: int) -> dict[str, Any]:
        led = AlphaLedger(alpha, schedule)
        for _ in range(count):
            led.next_alpha()
        return led.state_dict()

    with pytest.raises(StoreError, match="inconsistent"):  # reset to empty
        reopen_with(None, 0)
    with pytest.raises(StoreError, match="inconsistent"):  # rolled back below the recorded index
        reopen_with(state(0.05, "spending", 1), 1)
    with pytest.raises(StoreError, match="inconsistent"):  # counter and ledger disagree
        reopen_with(state(0.05, "spending", 2), 5)
    with pytest.raises(StoreError, match="inconsistent"):  # the alpha / schedule pin was changed
        reopen_with(state(0.2, "per_decision", 2), 2)
    with pytest.raises(StoreError, match="inconsistent"):
        reopen_with(state(0.05, "spending", 1), 0)
    again = reopen_with(good["ledger"], good["evaluations"])
    assert again.evaluation_count == 2
    # ahead of the records (undetectable lower bound, but harmless): accepted
    assert reopen_with(state(0.05, "spending", 4), 4).evaluation_count == 4


def test_ledger_unchanged_when_the_gate_raises(
    tmp_path: Path, model: nn.Module, sampler: ItemSampler, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, model)
    store.init_from_model()
    probe = _probe(sampler, SwitchScore())
    staged = store.stage(_edit("ablate", {0: [0]}))
    store.evaluate(staged, _config("spending"), probe)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("statistics failed")

    monkeypatch.setattr(
        edits_mod.CapabilityGate, "verdict", lambda self, ev: (self.ledger.next_alpha(), boom())[1]
    )
    with pytest.raises(RuntimeError, match="statistics failed"):
        store.evaluate(staged, _config("spending"), probe)
    monkeypatch.undo()
    led = store.ledger
    assert led is not None and led.count == 1 and store.evaluation_count == 1
    assert _ledger_json(tmp_path / "store")["ledger"]["count"] == 1
    store.commit(staged)
    assert VersionStore(tmp_path / "store", model).evaluation_count == 1


def test_head_write_failure_rolls_back_commit_and_init(
    tmp_path: Path, model: nn.Module, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_replace = edits_mod.os.replace

    def fail_head(src: Any, dst: Any) -> None:
        if Path(dst).name == "head.json":
            raise OSError("disk full at head.json")
        real_replace(src, dst)

    store = _store(tmp_path, model)
    root = store.init_from_model()
    sdir = tmp_path / "store"
    staged = store.stage(_edit("ablate", {0: [2]}))
    monkeypatch.setattr(edits_mod.os, "replace", fail_head)
    with pytest.raises(OSError, match=r"head\.json"):
        store.commit(staged, override="t")
    monkeypatch.undo()
    assert [v.id for v in store.versions()] == [root.id] and store.head == root.id
    assert store.staged is staged and staged.status == "staged"
    assert [p.name for p in (sdir / "versions").iterdir()] == [f"{root.id}.json"]
    v1 = store.commit(staged, override="t")  # the retry leaves no orphan sibling
    assert [v.id for v in store.versions()] == [root.id, v1.id]
    assert [v.id for v in VersionStore(sdir, model).versions()] == [root.id, v1.id]

    # init: a failed head write leaves an empty store that init_from_model can retry
    m2 = build_tiny_llama(model.config.vocab_size, seed=0)
    store2 = VersionStore(tmp_path / "s2", m2)
    monkeypatch.setattr(edits_mod.os, "replace", fail_head)
    with pytest.raises(OSError, match=r"head\.json"):
        store2.init_from_model()
    monkeypatch.undo()
    assert store2.versions() == [] and store2.head is None
    assert not list((tmp_path / "s2" / "versions").iterdir())
    r2 = store2.init_from_model()
    assert store2.head == r2.id

    # a crash-shaped store (a root record, head unset) names the recovering call
    (tmp_path / "s2" / "head.json").write_text(json.dumps({"format": edits_mod.STORE_FORMAT, "head": None}))
    crashed = VersionStore(tmp_path / "s2", m2)
    with pytest.raises(StoreError, match=r"restore\('" + r2.id):
        crashed.stage(_edit("ablate", {0: [2]}))
    with pytest.raises(StoreError, match=r"restore\('" + r2.id):
        crashed.init_from_model()
    crashed.restore(r2.id)
    crashed.discard(crashed.stage(_edit("ablate", {0: [2]})))


def test_corrupted_existing_state_file_is_rewritten(tmp_path: Path, model: nn.Module) -> None:
    store = _store(tmp_path, model)
    root = store.init_from_model()
    root_w = _params(model)
    store.commit(store.stage(_edit("scale", {0: [2]}, factor=2.0)), override="t")
    sfile = tmp_path / "store" / "states" / f"{root.state}.safetensors"
    data = bytearray(sfile.read_bytes())
    data[-1] ^= 0x01  # bit rot in the tensor data; the safetensors header still parses
    sfile.write_bytes(bytes(data))
    reopened = VersionStore(tmp_path / "store", model)
    with pytest.raises(StoreError, match="content hash"):
        reopened.load_state(root.id)
    # x2 then x0.5 is exact in float32 (power-of-two scaling), so v2's state file IS the root's:
    # the reviewer's scenario of a commit hitting an existing, corrupted content-addressed file
    v2 = reopened.commit(reopened.stage(_edit("scale", {0: [2]}, factor=0.5)), override="t")
    assert v2.state == root.state
    _assert_bitwise(model, root_w)
    reopened.restore(v2.id)
    reopened.restore(root.id)  # the shared file was repaired
    _assert_bitwise(model, root_w)


def test_ids_with_trailing_newline_are_rejected(tmp_path: Path, model: nn.Module) -> None:
    store = _store(tmp_path, model)
    root = store.init_from_model()
    for bad in (root.id + "\n", "a" * 64 + "\n"):
        with pytest.raises(EditError):
            _edit("revert", {0: [1]}, version=bad)
        with pytest.raises(EditError):
            store.get(bad)
        with pytest.raises(EditError):
            store.restore(bad)
        with pytest.raises(EditError):
            store.load_delta("b" * 64 + "\n")
    state_file = tmp_path / "store" / "states" / f"{root.state}.safetensors"
    (state_file.parent / (state_file.name + "\n")).write_bytes(state_file.read_bytes())
    with pytest.raises(StoreError, match="unexpected entry"):
        VersionStore(tmp_path / "store", model)


def test_ledger_counter_pair_checked_without_versions(tmp_path: Path, model: nn.Module) -> None:
    _store(tmp_path, model)  # empty store, no root version
    sdir = tmp_path / "store"
    (sdir / "ledger.json").write_text(
        json.dumps({"format": edits_mod.STORE_FORMAT, "ledger": None, "evaluations": 3})
    )
    with pytest.raises(StoreError, match="inconsistent"):
        VersionStore(sdir, model)


def test_version_evaluation_invariants_and_edit_copy(
    tmp_path: Path, model: nn.Module, sampler: ItemSampler
) -> None:
    e = _edit("scale", {0: [2, 1]}, factor=2)
    c = e.copy()
    assert c == e and c is not e and c.params is not e.params and c.selection is not e.selection
    assert c.params["factor"] == 2.0 and isinstance(c.params["factor"], float)
    store = _store(tmp_path, model)
    root = store.init_from_model()
    assert root.evaluation_pin() is None
    v_override = store.commit(store.stage(_edit("ablate", {0: [3]})), override="t")
    assert v_override.evaluation_pin() is None
    staged = store.stage(_edit("ablate", {0: [0]}))
    store.evaluate(staged, _config("spending"), _probe(sampler, SwitchScore()))
    v = store.commit(staged)
    assert v.evaluation_pin() == (0, 0.05, "spending")
    fields_ = {k: getattr(v, k) for k in edits_mod._ROOT_TEMPLATE}
    with pytest.raises(EditError, match="without a verdict"):
        Version.build(**{**fields_, "verdict": None, "override": "t"})
    assert v.verdict is not None and v.evaluation is not None
    with pytest.raises(EditError, match="decision_index"):
        Version.build(**{**fields_, "verdict": {**v.verdict, "decision_index": 5}})
    with pytest.raises(EditError, match="evaluation context"):
        Version.build(**{**fields_, "evaluation": None})
    with pytest.raises(EditError, match="gate_config"):
        Version.build(**{**fields_, "evaluation": {**v.evaluation, "gate_config": None}})
    # a Version copies its inputs: mutating them later changes nothing
    verdict = dict(v.verdict)
    built = Version.build(**{**fields_, "verdict": verdict})
    verdict["alarm"] = True
    assert built.passed and built.id == v.id


def test_nan_in_a_verdict_commits_and_reopens(
    tmp_path: Path, model: nn.Module, sampler: ItemSampler, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, model)
    store.init_from_model()
    real = edits_mod.CapabilityGate.verdict

    def nan_verdict(self: Any, evidence: Any) -> GateVerdict:
        out = real(self, evidence)
        return GateVerdict(
            decision_index=out.decision_index,
            transfer=SliceTest(T_SLICE, 0, float("nan"), float("nan"), False, "x", 0.05, float("nan")),
            retention=out.retention,
            alarm=out.alarm,
        )

    monkeypatch.setattr(edits_mod.CapabilityGate, "verdict", nan_verdict)
    staged = store.stage(_edit("ablate", {0: [0]}))
    store.evaluate(staged, _config(), _probe(sampler, SwitchScore()))
    v = store.commit(staged, override="t")
    again = VersionStore(tmp_path / "store", model)
    got = again.get(v.id)
    assert (
        got.verdict is not None and got.verdict["transfer"]["p_value"] != got.verdict["transfer"]["p_value"]
    )
    assert [x.id for x in again.history()] == [x.id for x in store.history()]


def test_interrupt_after_head_replace_keeps_the_version(
    tmp_path: Path, model: nn.Module, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, model)
    root = store.init_from_model()
    real = VersionStore._write_head

    def write_then_interrupt(self: VersionStore, head: str | None) -> None:
        real(self, head)
        raise KeyboardInterrupt  # after head.json was replaced

    staged = store.stage(_edit("ablate", {0: [4]}))
    monkeypatch.setattr(VersionStore, "_write_head", write_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        store.commit(staged, override="t")
    monkeypatch.undo()
    reopened = VersionStore(tmp_path / "store", model)
    assert store.head == reopened.head != root.id
    assert [v.id for v in store.versions()] == [v.id for v in reopened.versions()]
    assert len(reopened.versions()) == 2
