"""FastSlowMemory / StateStore vs a manual per-row reference loop."""

from __future__ import annotations

import pytest
import torch

from research_rl.core.checkpoint import StatefulComponent, load_components, save_components
from research_rl.core.registry import RECURRENT_STATES
from research_rl.state import FastSlowMemory, StateStore

D, K, ETA_F, ETA_S = 3, 3, 0.6, 0.2


def manual_row(xs: list[torch.Tensor], resets: set[int] = frozenset()):
    """Independent reference for one row: plain python loop over the documented equations."""
    f, s, n = torch.zeros(D, dtype=torch.float64), torch.zeros(D, dtype=torch.float64), 0
    for t, x in enumerate(xs):
        if t in resets:  # reset happens *before* the input of step t
            f, s, n = torch.zeros(D, dtype=torch.float64), torch.zeros(D, dtype=torch.float64), 0
        f = (1 - ETA_F) * f + ETA_F * x
        n += 1
        if n % K == 0:
            s = (1 - ETA_S) * s + ETA_S * f
    return f, s, n


def inputs(b: int, t: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(t, b, D, generator=g, dtype=torch.float64)


def test_fast_slow_memory_matches_manual_loop_with_staggered_resets():
    mem = FastSlowMemory(D, ETA_F, ETA_S, K)
    b, steps = 3, 2 * K + 3  # > 2k+1 so consolidation happens several times
    xs = inputs(b, steps, 0)
    resets = {0: set(), 1: {4}, 2: {2, 7}}  # per-row reset times
    state = mem.init(b, dtype=torch.float64)
    for t in range(steps):
        mask = torch.tensor([t in resets[i] for i in range(b)])
        if mask.any():
            state = mem.reset(state, mask)
        state = mem.update(state, xs[t])
    for i in range(b):
        f, s, n = manual_row([xs[t, i] for t in range(steps)], resets[i])
        torch.testing.assert_close(state["fast"][i], f, rtol=0, atol=1e-14)
        torch.testing.assert_close(state["slow"][i], s, rtol=0, atol=1e-14)
        assert int(state["steps"][i]) == n
    assert state["slow"].abs().sum() > 0  # consolidation happened
    # rows have different consolidation phases -> slow memories really differ in schedule
    assert [int(n) for n in state["steps"]] == [9, 5, 2]
    torch.testing.assert_close(mem.read(state), torch.cat([state["fast"], state["slow"]], -1))


def test_rows_are_independent_and_reset_only_touches_masked_rows():
    mem = FastSlowMemory(D, ETA_F, ETA_S, K)
    xs = inputs(4, 7, 1)
    xs2 = xs.clone()
    xs2[:, 2] += 5.0  # change only row 2's inputs
    a, b = mem.init(4, dtype=torch.float64), mem.init(4, dtype=torch.float64)
    for t in range(7):
        a, b = mem.update(a, xs[t]), mem.update(b, xs2[t])
    for key in a:
        for i in (0, 1, 3):
            assert torch.equal(a[key][i], b[key][i])
    assert not torch.equal(a["fast"][2], b["fast"][2])
    mask = torch.tensor([False, True, False, True])
    r = mem.reset(a, mask)
    for key in a:
        assert torch.equal(r[key][0], a[key][0]) and torch.equal(r[key][2], a[key][2])
        assert torch.equal(r[key][1], torch.zeros_like(a[key][1])) and torch.equal(
            r[key][3], torch.zeros_like(a[key][3])
        )
    # inactive rows are frozen
    active = torch.tensor([True, False, True, False])
    u = mem.update(a, xs[0], active=active)
    assert torch.equal(u["fast"][1], a["fast"][1]) and int(u["steps"][1]) == int(a["steps"][1])
    assert not torch.equal(u["fast"][0], a["fast"][0]) and int(u["steps"][0]) == int(a["steps"][0]) + 1


def test_state_store_is_keyed_by_sample_id_and_never_shares_state():
    mem = FastSlowMemory(D, ETA_F, ETA_S, K)
    store = StateStore(mem, dtype=torch.float64)
    xs = inputs(3, 6, 2)
    history: dict[object, list[torch.Tensor]] = {"a": [], 7: [], "b": []}
    # batches with changing composition and order
    plan = [["a", 7], [7, "b", "a"], ["b"], ["a", "b", 7], [7], ["b", "a"]]
    for t, ids in enumerate(plan):
        x = torch.stack([xs[t, j] for j in range(len(ids))])
        for j, sid in enumerate(ids):
            history[sid].append(x[j])
        out = store.step(ids, x)
        for j, sid in enumerate(ids):
            f, s, _ = manual_row(history[sid])
            torch.testing.assert_close(out[j], torch.cat([f, s]), rtol=0, atol=1e-14)
    assert sorted(map(str, store.ids())) == ["7", "a", "b"]
    # a brand-new id starts fresh, even right after other ids were updated
    state = store.gather(["new", "a"])
    assert torch.equal(state["fast"][0], torch.zeros(D, dtype=torch.float64)) and int(state["steps"][0]) == 0
    # composition mismatch, reordering and broadcasting are rejected
    with pytest.raises(ValueError, match="mismatch"):
        store.scatter(["a", "new"], state)
    with pytest.raises(ValueError, match="batch size"):
        store.scatter(["new", "a"], {k: v[:1] for k, v in state.items()})
    with pytest.raises(RuntimeError):
        store.gather(["a"])
    store.release()
    with pytest.raises(ValueError, match="duplicate"):
        store.gather(["a", "a"])
    # stored rows do not alias each other or the gathered batch
    st = store.gather(["a", 7])
    st["fast"].add_(100.0)
    store.release()
    assert float(store.get("a")["fast"].abs().max()) < 50
    assert store.get("a")["fast"].data_ptr() != store.get(7)["fast"].data_ptr()


def test_store_serialization_round_trip_through_checkpoint_components(tmp_path):
    mem = FastSlowMemory(D, ETA_F, ETA_S, K)
    store = StateStore(mem, dtype=torch.float64)
    assert isinstance(store, StatefulComponent)
    xs = inputs(2, 5, 3)
    for t in range(5):
        store.step(["x", 3], xs[t])
    save_components(tmp_path, {"memory": store})
    restored = StateStore(FastSlowMemory(D, 0.1, 0.1, 2), dtype=torch.float64)
    assert load_components(tmp_path, {"memory": restored}) == ["memory"]  # torch.load(weights_only=True)
    assert restored.module.state_dict() == mem.state_dict()
    for sid in ("x", 3):
        for key, v in store.get(sid).items():
            assert torch.equal(restored.get(sid)[key], v)
    # continuing from the restored store equals continuing the original
    nxt = inputs(2, 1, 4)[0]
    assert torch.equal(store.step(["x", 3], nxt), restored.step(["x", 3], nxt))
    with pytest.raises(ValueError):
        FastSlowMemory(D + 1).load_state_dict(mem.state_dict())
    assert RECURRENT_STATES.get("fast_slow_memory") is FastSlowMemory
