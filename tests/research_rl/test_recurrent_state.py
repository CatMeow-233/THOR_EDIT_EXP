"""FastSlowMemory / StateStore vs a manual per-row reference loop."""

from __future__ import annotations

import warnings

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
    # an aborted gather does not leak unseen ids into the store (len / ids / checkpoints)
    known = sorted(map(str, store.ids()))
    store.gather(["ghost", "a"])
    store.release()
    assert "ghost" not in store and sorted(map(str, store.ids())) == known and len(store) == len(known)
    assert "ghost" not in {k.partition(":")[2] for k in store.state_dict()["states"]}
    # ...while a completed gather/scatter does store the new id
    st = store.gather(["ghost"])
    store.scatter(["ghost"], st)
    assert "ghost" in store and len(store) == len(known) + 1


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


# ---------------------------------------------------------------------- precision (slow_dtype)


def old_update(mem: FastSlowMemory, state, inputs, active=None):
    """Verbatim copy of the pre-fix ``FastSlowMemory.update`` (everything in the state dtype)."""
    f, s, n = state["fast"], state["slow"], state["steps"]
    x = inputs.to(f.dtype)
    f_new = (1.0 - mem.fast_rate) * f + mem.fast_rate * x
    n_new = n + 1
    consolidate = (n_new % mem.consolidate_every == 0).unsqueeze(-1)
    s_new = torch.where(consolidate, (1.0 - mem.slow_rate) * s + mem.slow_rate * f_new, s)
    new = {"fast": f_new, "slow": s_new, "steps": n_new}
    if active is not None:
        new = {
            k: torch.where(active.reshape(-1, *([1] * (v.dim() - 1))), v, state[k]) for k, v in new.items()
        }
    return new


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64])
def test_fp32_state_is_bit_identical_to_the_old_formula(input_dtype):
    mem = FastSlowMemory(D, ETA_F, ETA_S, K)  # default slow_dtype=float32
    b, steps = 4, 4 * K + 1
    xs = inputs(b, steps, 5).to(input_dtype) * 3.0
    g = torch.Generator().manual_seed(6)
    new = mem.init(b)
    assert {k: v.dtype for k, v in new.items()} == {
        "fast": torch.float32,
        "slow": torch.float32,
        "steps": torch.long,
    }
    old = {k: v.clone() for k, v in new.items()}
    for t in range(steps):
        active = torch.rand(b, generator=g) < 0.7 if t % 2 else None
        new = mem.update(new, xs[t], active=active)
        old = old_update(mem, old, xs[t], active=active)
        for k in old:
            assert new[k].dtype == old[k].dtype and torch.equal(new[k], old[k]), (t, k)
        assert torch.equal(mem.read(new), torch.cat([old["fast"], old["slow"]], dim=-1))
    assert old["slow"].abs().sum() > 0
    # same through the StateStore path
    store = StateStore(mem)
    ref = mem.init(2)
    for t in range(steps):
        out = store.step(["p", "q"], xs[t, :2])
        ref = old_update(mem, ref, xs[t, :2])
        assert torch.equal(out, torch.cat([ref["fast"], ref["slow"]], dim=-1))


def _decay_run(mem: FastSlowMemory, dtype: torch.dtype, steps: int = 200):
    state = mem.init(1, dtype=dtype)
    state["slow"] = torch.ones_like(state["slow"])
    zeros = torch.zeros(1, D, dtype=dtype)
    for _ in range(steps):
        state = mem.update(state, zeros)
    return state


def test_bf16_state_slow_memory_does_not_freeze():
    """bf16 state, slow_rate=1e-3, k=1, s_0=1, f=x=0: s_200 must follow 0.999**200 ≈ 0.8186."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the default configuration must not warn
        mem = FastSlowMemory(D, fast_rate=0.5, slow_rate=1e-3, consolidate_every=1)
        state = _decay_run(mem, torch.bfloat16)
    assert state["fast"].dtype == torch.bfloat16 and state["slow"].dtype == torch.float32
    assert mem.read(state).dtype == torch.float32
    reference = torch.full((1, D), (1.0 - 1e-3) ** 200, dtype=torch.float64)
    # Tolerance 1e-4: the slow EMA is computed and stored in fp32 (f stays exactly 0), so each of the
    # 200 steps adds at most ~3 fp32 roundings (rate conversion, mul, add) of relative size 2**-24
    # => worst-case |error| <= 200 * 3 * 2**-24 ≈ 3.6e-5 (observed ≈ 2e-6). It is ~40x tighter than
    # the bf16 spacing near 0.82 (2**-8 ≈ 3.9e-3), so it would catch slow storage rounded to bf16.
    torch.testing.assert_close(state["slow"].double(), reference, rtol=0, atol=1e-4)

    # negative control: half-precision slow storage reproduces the freeze and warns at init()
    legacy = FastSlowMemory(D, fast_rate=0.5, slow_rate=1e-3, consolidate_every=1, slow_dtype=None)
    with pytest.warns(UserWarning, match="rounded away"):
        frozen = _decay_run(legacy, torch.bfloat16)
    assert frozen["slow"].dtype == torch.bfloat16 and bool((frozen["slow"] == 1.0).all())

    # a hand-built all-bf16 state (e.g. from an older checkpoint) is promoted on its first update
    hand = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in mem.init(1).items()}
    hand["slow"] = torch.ones_like(hand["slow"])
    for _ in range(200):
        hand = mem.update(hand, torch.zeros(1, D))
    assert hand["slow"].dtype == torch.float32
    torch.testing.assert_close(hand["slow"].double(), reference, rtol=0, atol=1e-4)


def test_bf16_with_random_inputs_tracks_float64_reference():
    eta_f, eta_s, k, steps = 0.5, 2e-3, 1, 300
    mem = FastSlowMemory(D, eta_f, eta_s, k)
    xs = inputs(2, steps, 7) + 1.0  # float64, N(1, 1): the slow EMA climbs towards ~1
    state = mem.init(2, dtype=torch.bfloat16)
    f64 = torch.zeros(2, D, dtype=torch.float64)
    s64 = torch.zeros(2, D, dtype=torch.float64)
    f_max = 0.0
    for t in range(steps):
        state = mem.update(state, xs[t])
        f64 = (1 - eta_f) * f64 + eta_f * xs[t]
        s64 = (1 - eta_s) * s64 + eta_s * f64
        f_max = max(f_max, float(f64.abs().max()))
    # fast is rounded to bf16 every step: e_t = (1-eta_f) e_{t-1} + delta_t with |delta_t| <= u|f_t|,
    # u = 2**-8, so |e_f| <= u * max_t|f_t| / eta_f (+ fp32 terms). The slow EMA is a convex average of
    # the unrounded fast values computed from that state, so |e_s| obeys the same worst-case bound; the
    # 1e-4 slack covers the fp32 compute roundings (cf. the decay test above).
    bound = 2**-8 * f_max / eta_f + 1e-4
    assert float((state["fast"].double() - f64).abs().max()) <= bound
    assert float((state["slow"].double() - s64).abs().max()) <= bound
    assert float(s64.min()) > 0.3  # the slow state really moved, although eta_s < 2**-8


def test_mixed_dtype_state_store_round_trip_and_reset(tmp_path):
    mem = FastSlowMemory(D, ETA_F, ETA_S, K)
    store = StateStore(mem, dtype=torch.bfloat16)
    xs = inputs(2, 5, 8)
    for t in range(5):
        out = store.step(["x", 3], xs[t])
        assert out.dtype == torch.float32
    row = store.get("x")
    assert (row["fast"].dtype, row["slow"].dtype, row["steps"].dtype) == (
        torch.bfloat16,
        torch.float32,
        torch.long,
    )
    # gather mixes known and unseen ids with consistent per-key dtypes
    st = store.gather(["new", "x"])
    assert (st["fast"].dtype, st["slow"].dtype) == (torch.bfloat16, torch.float32)
    store.release()
    # rows stored at another precision (e.g. an older all-bf16 checkpoint) are cast to the module layout
    store._states["str:old"] = {k: v.to(torch.bfloat16) if k == "slow" else v for k, v in row.items()}
    st = store.gather(["old", "x"])
    assert st["slow"].dtype == torch.float32
    store.release()
    del store._states["str:old"]

    save_components(tmp_path, {"memory": store})
    restored_mem = FastSlowMemory(D, 0.1, 0.1, 2, slow_dtype=None)
    restored = StateStore(restored_mem, dtype=torch.bfloat16)
    assert load_components(tmp_path, {"memory": restored}) == ["memory"]  # torch.load(weights_only=True)
    assert restored_mem.state_dict() == mem.state_dict() and restored_mem.slow_dtype == torch.float32
    for sid in ("x", 3):
        for key, v in store.get(sid).items():
            got = restored.get(sid)[key]
            assert got.dtype == v.dtype and torch.equal(got, v)
    nxt = inputs(2, 1, 9)[0]
    assert torch.equal(store.step(["x", 3], nxt), restored.step(["x", 3], nxt))

    # reset() keeps every key's dtype and only zeroes the masked rows
    state = store.gather(["x", 3])
    store.release()
    r = mem.reset(state, torch.tensor([True, False]))
    for key in state:
        assert r[key].dtype == state[key].dtype
        assert torch.equal(r[key][1], state[key][1]) and not r[key][0].any()


def test_slow_dtype_options_and_checkpoint_compat():
    assert FastSlowMemory(D).init(1, dtype=torch.float64)["slow"].dtype == torch.float64
    assert FastSlowMemory(D).init(1, dtype=torch.float16)["slow"].dtype == torch.float32
    assert FastSlowMemory(D, slow_dtype="float64").init(1)["slow"].dtype == torch.float64
    assert FastSlowMemory(D, slow_dtype=None).init(1, dtype=torch.float16)["slow"].dtype == torch.float16
    with pytest.raises(ValueError, match="slow_dtype"):
        FastSlowMemory(D, slow_dtype=torch.long)
    with pytest.raises(ValueError, match="slow_dtype"):
        FastSlowMemory(D, slow_dtype="not_a_dtype")
    # checkpoints written before slow_dtype existed keep the constructor's value
    mem = FastSlowMemory(D, slow_dtype=None)
    legacy = {k: v for k, v in FastSlowMemory(D).state_dict().items() if k != "slow_dtype"}
    mem.load_state_dict(legacy)
    assert mem.slow_dtype is None
    mem.load_state_dict(FastSlowMemory(D, slow_dtype="bfloat16").state_dict())
    assert mem.slow_dtype == torch.bfloat16


@pytest.mark.parametrize(
    ("dtype", "slow_rate", "fast_rate", "warns"),
    [
        (torch.bfloat16, 1e-3, 0.5, True),  # slow_rate < 2**-8
        (torch.bfloat16, 2**-7, 0.5, False),
        (torch.bfloat16, 0.0, 0.5, False),  # slow memory disabled: nothing to round away
        (torch.bfloat16, 0.1, 2e-3, True),  # fast_rate < 2**-8
        (torch.float16, 1e-3, 0.5, False),  # 1e-3 > 2**-11 ≈ 4.9e-4
        (torch.float16, 2e-4, 0.5, True),
        (torch.float32, 1e-6, 0.5, False),
    ],
)
def test_half_precision_slow_storage_warns_when_rate_is_rounded_away(dtype, slow_rate, fast_rate, warns):
    mem = FastSlowMemory(D, fast_rate, slow_rate, consolidate_every=1, slow_dtype=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        state = mem.init(2, dtype=dtype)
        mem.init(1, dtype=dtype)  # every init() checks
        mem.reset(state, torch.tensor([True, False]))  # reset re-initialises without re-warning
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert len(msgs) == (2 if warns else 0), msgs
    if warns:
        assert "rounded away" in msgs[0] and str(dtype) in msgs[0]
        assert all(w.filename == __file__ for w in caught)  # stacklevel points at the caller
    # the default slow_dtype never warns, whatever the requested dtype
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FastSlowMemory(D, fast_rate, slow_rate, consolidate_every=1).init(2, dtype=dtype)
