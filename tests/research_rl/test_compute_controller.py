"""run_looped / controllers / ledger / best_of_depth vs manual per-row reference loops."""

from __future__ import annotations

import io

import pytest
import torch

from research_rl.core.registry import COMPUTE_CONTROLLERS
from research_rl.state import (
    ComputeLedger,
    FixedDepth,
    ThresholdHalting,
    best_of_depth,
    collect_depth_outputs,
    run_looped,
)


class ContractingBlock:
    """Row-wise block: h[:, :-1] ← r·h[:, :-1] + c, with the per-row rate r stored in h[:, -1].

    Rows contract at different rates, so they halt at different depths. Counts processed rows.
    """

    def __init__(self) -> None:
        self.rows_processed = 0

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        self.rows_processed += h.shape[0]
        rate = h[:, -1:]
        return torch.cat([rate * h[:, :-1] + 0.1, rate], dim=-1)


def make_h0() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(5, 3, generator=g, dtype=torch.float64)
    rates = torch.tensor([[0.1], [0.5], [0.8], [0.95], [0.3]], dtype=torch.float64)
    return torch.cat([x, rates], dim=-1)


def manual_threshold(h0: torch.Tensor, eps: float, max_depth: int):
    """Independent per-row reference: apply the block row by row until the change is below eps."""
    outs, depths, early = [], [], []
    for i in range(h0.shape[0]):
        h = h0[i].clone()
        d, stopped = 0, False
        while d < max_depth:
            new = torch.cat([h[-1] * h[:-1] + 0.1, h[-1:]])
            d += 1
            delta = float(torch.sqrt(((new - h) ** 2).sum()))
            h = new
            if delta < eps:
                stopped = True
                break
        outs.append(h)
        depths.append(d)
        early.append(stopped)
    return torch.stack(outs), depths, early


def test_threshold_halting_matches_manual_loop_and_freezes_halted_rows():
    h0 = make_h0()
    block = ContractingBlock()
    ledger = ComputeLedger()
    res = run_looped(
        block,
        h0,
        ThresholdHalting(1e-3),
        max_depth=40,
        ledger=ledger,
        cost_per_iteration=[2.0, 2.0, 3.0, 3.0, 1.0],
        call_id="q1",
        sample_ids=list("abcde"),
    )
    ref_out, ref_depths, ref_early = manual_threshold(h0, 1e-3, 40)
    assert res.depths.tolist() == ref_depths
    assert len(set(ref_depths)) >= 4  # genuinely per-sample depths
    assert ref_depths[3] == 40 and not ref_early[3]  # slow row hit max_depth
    assert res.halted_by_controller.tolist() == ref_early
    assert torch.equal(res.output, ref_out)  # frozen rows: bit-exact equality with the per-row loop
    assert block.rows_processed == sum(ref_depths)  # halted rows were not computed again
    e = res.entry
    assert e is not None and e.call_id == "q1" and e.depths == ref_depths
    assert e.costs == [d * c for d, c in zip(ref_depths, [2.0, 2.0, 3.0, 3.0, 1.0])]
    assert ledger.total_iterations() == sum(ref_depths)
    assert ledger.total_cost() == pytest.approx(sum(e.costs))


def test_fixed_depth_and_ledger_accumulates_per_sample():
    h0 = make_h0()
    ledger = ComputeLedger()
    r1 = run_looped(
        ContractingBlock(), h0, FixedDepth(3), max_depth=10, ledger=ledger, sample_ids=list("abcde")
    )
    h = h0.clone()
    for _ in range(3):
        h = ContractingBlock()(h)
    assert torch.equal(r1.output, h) and r1.depths.tolist() == [3] * 5
    r2 = run_looped(
        ContractingBlock(),
        h0[:2],
        FixedDepth(20),
        max_depth=4,
        ledger=ledger,
        cost_per_iteration=10.0,
        sample_ids=["a", "z"],
    )
    assert r2.depths.tolist() == [4, 4] and not r2.halted_by_controller.any()
    per = ledger.per_sample()
    assert per["a"] == {"iterations": 7.0, "cost": 3.0 + 40.0, "calls": 2.0}
    assert per["z"]["iterations"] == 4.0 and per["e"]["calls"] == 1.0
    assert ledger.mean_depth() == pytest.approx((15 + 8) / 7)
    buf = io.BytesIO()
    torch.save(ledger.state_dict(), buf)
    buf.seek(0)
    restored = ComputeLedger()
    restored.load_state_dict(torch.load(buf, weights_only=True))
    assert restored.per_sample() == per and restored.total_cost() == ledger.total_cost()


def test_best_of_depth_selects_per_sample_and_records_full_cost():
    h0 = make_h0()
    outs = collect_depth_outputs(ContractingBlock(), h0, 6)
    target = torch.tensor([0.12, 0.5, 0.3, 1.0, 0.14], dtype=torch.float64)

    def scorer(h: torch.Tensor) -> torch.Tensor:
        return -(h[:, 0] - target).abs()

    ledger = ComputeLedger()
    res = best_of_depth(outs, scorer, ledger=ledger, cost_per_iteration=2.5)
    # manual reference
    h = h0.clone()
    manual = []
    for _ in range(6):
        h = torch.cat([h[:, -1:] * h[:, :-1] + 0.1, h[:, -1:]], -1)
        manual.append(h)
    for i in range(5):
        sc = [float(-(manual[d][i, 0] - target[i]).abs()) for d in range(6)]
        best = max(range(6), key=lambda d: (sc[d], -d))  # earliest on ties
        assert int(res.best_depth[i]) == best + 1
        assert torch.equal(res.output[i], manual[best][i])
    assert len(set(res.best_depth.tolist())) >= 2
    assert res.entry is not None and res.entry.depths == [6] * 5 and res.entry.kind == "best_of_depth"
    assert ledger.total_cost() == pytest.approx(6 * 5 * 2.5)
    # ties resolve to the earliest depth
    tie = best_of_depth(torch.ones(4, 2, 3), lambda x: x.sum(-1))
    assert tie.best_depth.tolist() == [1, 1]


def test_controller_contract_errors_and_registry():
    with pytest.raises(ValueError):
        run_looped(lambda h: h[:, :1], make_h0(), FixedDepth(2), max_depth=3)
    with pytest.raises(ValueError):
        FixedDepth(0)
    assert COMPUTE_CONTROLLERS.get("fixed_depth") is FixedDepth
    assert COMPUTE_CONTROLLERS.get("threshold_halting") is ThresholdHalting
    th = ThresholdHalting(0.5, min_depth=3)
    th.load_state_dict({"eps": 0.25, "min_depth": 2})
    assert th.state_dict() == {"eps": 0.25, "min_depth": 2}
    same = torch.zeros(2, 3)
    assert th.halt(1, same, same).tolist() == [False, False] and th.halt(2, same, same).tolist() == [
        True,
        True,
    ]


def test_threshold_halting_boundary_is_strict():
    """‖Δh‖ exactly equal to eps does NOT halt (strict ``<``); just below does; min_depth gates both."""
    eps = 0.5  # exactly representable, as are the displacements below
    ctrl = ThresholdHalting(eps, min_depth=2)
    prev = torch.zeros(4, 2, 3)
    cur = torch.zeros(4, 2, 3)
    cur[0, 0, 0] = 0.5  # norm == eps
    cur[1, 1, 2] = -0.25  # norm < eps
    cur[2, 0, 1] = 0.75  # norm > eps
    # row 3 unchanged (norm 0)
    assert ctrl.halt(2, prev, cur).tolist() == [False, True, False, True]
    assert ctrl.halt(1, prev, cur).tolist() == [False, False, False, False]  # below min_depth
