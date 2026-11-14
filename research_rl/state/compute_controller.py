"""Adaptive-compute contract for looped / recurrent-depth models (AIC, Best-of-Depth, ...).

* :class:`ComputeController` – per-sample halting decisions for a weight-shared block applied
  repeatedly. ``halt(depth, previous, current)`` is called with the *active* rows only and returns
  a bool per row.
* :func:`run_looped` – applies ``block`` up to ``max_depth`` times; halted rows are frozen (never
  passed to ``block`` again), so the recorded depth is the number of block applications a row
  actually received. ``block`` must be row-wise (no cross-row interaction), because it sees a
  shrinking subset of rows.
* :class:`ComputeLedger` – per inference call and per sample: actual iteration depth and an
  estimated cost = depth × cost-per-iteration (e.g. FLOPs/iteration supplied by the caller; the
  ledger does not measure FLOPs itself).
* :func:`best_of_depth` – scores the outputs after each depth and picks the best per sample
  (ties → earliest depth), recording the cost of *all* depths that were computed.

Reference controllers: :class:`FixedDepth`, :class:`ThresholdHalting`. Learned halting policies
(e.g. ACT/PonderNet-style heads) are NOT provided; they plug in by subclassing the contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from research_rl.core.registry import COMPUTE_CONTROLLERS


class ComputeController(ABC):
    """Base class. Override :meth:`halt`; :meth:`start` is called once per inference call."""

    name: str = "compute_controller"

    def start(self, batch_size: int, device: torch.device) -> None:
        return None

    @abstractmethod
    def halt(self, depth: int, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        """Bool ``[rows]``: True = stop after ``depth`` applications (rows = currently active rows)."""

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        return None


@COMPUTE_CONTROLLERS.register("fixed_depth")
class FixedDepth(ComputeController):
    """Every sample receives exactly ``depth`` iterations (capped by ``max_depth``)."""

    name = "fixed_depth"

    def __init__(self, depth: int) -> None:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = int(depth)

    def halt(self, depth: int, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        return torch.full((current.shape[0],), depth >= self.depth, dtype=torch.bool, device=current.device)

    def state_dict(self) -> dict[str, Any]:
        return {"depth": self.depth}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.depth = int(state.get("depth", self.depth))


@COMPUTE_CONTROLLERS.register("threshold_halting")
class ThresholdHalting(ComputeController):
    """Halt a row once ‖h_t − h_{t−1}‖₂ (over all non-batch dims) < ``eps`` and depth ≥ ``min_depth``.

    ``run_looped`` additionally stops every row at ``max_depth``.
    """

    name = "threshold_halting"

    def __init__(self, eps: float, min_depth: int = 1) -> None:
        self.eps = float(eps)
        self.min_depth = int(min_depth)

    def halt(self, depth: int, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        diff = (current - previous).reshape(current.shape[0], -1).float()
        small = torch.linalg.vector_norm(diff, dim=1) < self.eps
        return small & (depth >= self.min_depth)

    def state_dict(self) -> dict[str, Any]:
        return {"eps": self.eps, "min_depth": self.min_depth}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.eps = float(state.get("eps", self.eps))
        self.min_depth = int(state.get("min_depth", self.min_depth))


# ===================================================================================== ledger
@dataclass
class LedgerEntry:
    call_id: str
    kind: str
    sample_ids: list[str]
    depths: list[int]  # actual iterations per sample
    cost_per_iteration: list[float]
    costs: list[float]  # depth × cost_per_iteration

    @property
    def total_iterations(self) -> int:
        return sum(self.depths)

    @property
    def total_cost(self) -> float:
        return sum(self.costs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "kind": self.kind,
            "sample_ids": list(self.sample_ids),
            "depths": list(self.depths),
            "cost_per_iteration": list(self.cost_per_iteration),
            "costs": list(self.costs),
        }


class ComputeLedger:
    """Records actual iteration depth and estimated cost per call and per sample."""

    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []

    def record(
        self,
        depths: torch.Tensor | Sequence[int],
        cost_per_iteration: float | Sequence[float] | torch.Tensor = 1.0,
        *,
        call_id: str | None = None,
        sample_ids: Sequence[Hashable] | None = None,
        kind: str = "looped",
    ) -> LedgerEntry:
        d = [int(x) for x in (depths.tolist() if isinstance(depths, torch.Tensor) else depths)]
        if any(x < 0 for x in d):
            raise ValueError("depths must be non-negative")
        if isinstance(cost_per_iteration, torch.Tensor):
            cpi = [float(x) for x in cost_per_iteration.reshape(-1).tolist()]
        elif isinstance(cost_per_iteration, int | float):
            cpi = [float(cost_per_iteration)] * len(d)
        else:
            cpi = [float(x) for x in cost_per_iteration]
        if len(cpi) != len(d):
            raise ValueError(f"{len(cpi)} per-iteration costs for {len(d)} samples")
        ids = [str(s) for s in sample_ids] if sample_ids is not None else [str(i) for i in range(len(d))]
        if len(ids) != len(d):
            raise ValueError(f"{len(ids)} sample ids for {len(d)} samples")
        entry = LedgerEntry(
            call_id=call_id if call_id is not None else f"call{len(self.entries)}",
            kind=kind,
            sample_ids=ids,
            depths=d,
            cost_per_iteration=cpi,
            costs=[x * c for x, c in zip(d, cpi)],
        )
        self.entries.append(entry)
        return entry

    def total_iterations(self) -> int:
        return sum(e.total_iterations for e in self.entries)

    def total_cost(self) -> float:
        return sum(e.total_cost for e in self.entries)

    def mean_depth(self) -> float:
        n = sum(len(e.depths) for e in self.entries)
        return self.total_iterations() / n if n else 0.0

    def per_sample(self) -> dict[str, dict[str, float]]:
        """``{sample_id: {"iterations": Σ depth, "cost": Σ cost, "calls": #calls}}`` over all calls."""
        out: dict[str, dict[str, float]] = {}
        for e in self.entries:
            for sid, d, c in zip(e.sample_ids, e.depths, e.costs):
                row = out.setdefault(sid, {"iterations": 0.0, "cost": 0.0, "calls": 0.0})
                row["iterations"] += d
                row["cost"] += c
                row["calls"] += 1
        return out

    def state_dict(self) -> dict[str, Any]:
        return {"entries": [e.to_dict() for e in self.entries]}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.entries = [LedgerEntry(**dict(e)) for e in state.get("entries", [])]


# ===================================================================================== looped execution
@dataclass
class LoopResult:
    output: torch.Tensor  # final hidden state per sample (frozen at its halting depth)
    depths: torch.Tensor  # long [B]: block applications each sample received
    halted_by_controller: torch.Tensor  # bool [B]: False = stopped by max_depth
    entry: LedgerEntry | None = None


def run_looped(
    block: Callable[[torch.Tensor], torch.Tensor],
    h0: torch.Tensor,
    controller: ComputeController,
    max_depth: int,
    *,
    ledger: ComputeLedger | None = None,
    cost_per_iteration: float | Sequence[float] | torch.Tensor = 1.0,
    call_id: str | None = None,
    sample_ids: Sequence[Hashable] | None = None,
) -> LoopResult:
    """Apply the weight-shared ``block`` repeatedly with per-sample halting (see module docstring)."""
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")
    b = h0.shape[0]
    h = h0
    depths = torch.zeros(b, dtype=torch.long, device=h0.device)
    active = torch.ones(b, dtype=torch.bool, device=h0.device)
    by_controller = torch.zeros(b, dtype=torch.bool, device=h0.device)
    controller.start(b, h0.device)
    for t in range(1, max_depth + 1):
        idx = active.nonzero().squeeze(1)
        if idx.numel() == 0:
            break
        prev = h.index_select(0, idx)
        cur = block(prev)
        if cur.shape != prev.shape:
            raise ValueError(f"block must preserve the shape {tuple(prev.shape)}, got {tuple(cur.shape)}")
        h = h.index_copy(0, idx, cur)
        depths[idx] += 1
        stop = controller.halt(t, prev, cur).to(device=h0.device, dtype=torch.bool)
        if stop.shape != (idx.numel(),):
            raise ValueError(f"controller returned shape {tuple(stop.shape)} for {idx.numel()} active rows")
        by_controller[idx[stop]] = True
        active[idx[stop]] = False
    entry = None
    if ledger is not None:
        entry = ledger.record(
            depths, cost_per_iteration, call_id=call_id, sample_ids=sample_ids, kind="looped"
        )
    return LoopResult(output=h, depths=depths, halted_by_controller=by_controller, entry=entry)


def collect_depth_outputs(
    block: Callable[[torch.Tensor], torch.Tensor], h0: torch.Tensor, max_depth: int
) -> torch.Tensor:
    """``[D, B, ...]`` stack of h_1..h_D (all rows run the full ``max_depth``)."""
    outs = []
    h = h0
    for _ in range(max_depth):
        h = block(h)
        outs.append(h)
    return torch.stack(outs)


@dataclass
class BestOfDepthResult:
    output: torch.Tensor  # [B, ...] selected output
    best_depth: torch.Tensor  # long [B], 1-based depth of the selected output
    scores: torch.Tensor  # [D, B]
    entry: LedgerEntry | None = field(default=None)


def best_of_depth(
    outputs_per_depth: torch.Tensor | Sequence[torch.Tensor],
    scorer: Callable[[torch.Tensor], torch.Tensor],
    *,
    ledger: ComputeLedger | None = None,
    cost_per_iteration: float | Sequence[float] | torch.Tensor = 1.0,
    call_id: str | None = None,
    sample_ids: Sequence[Hashable] | None = None,
    iterations_computed: torch.Tensor | Sequence[int] | None = None,
) -> BestOfDepthResult:
    """Pick, per sample, the depth whose output scores highest (ties → earliest depth).

    The recorded cost uses the iterations actually computed (default: all D depths per sample),
    not the selected depth — selecting depth 2 of 8 still paid for 8 iterations.
    """
    outs = (
        outputs_per_depth
        if isinstance(outputs_per_depth, torch.Tensor)
        else torch.stack(list(outputs_per_depth))
    )
    d, b = int(outs.shape[0]), int(outs.shape[1])
    scores = torch.stack([scorer(outs[i]).reshape(-1) for i in range(d)])
    if scores.shape != (d, b):
        raise ValueError(f"scorer must return one score per sample, got shape {tuple(scores.shape)}")
    best = scores.argmax(dim=0)  # first maximal index on ties
    gather_idx = best.reshape(1, b, *([1] * (outs.dim() - 2))).expand(1, *outs.shape[1:])
    selected = outs.gather(0, gather_idx).squeeze(0)
    entry = None
    if ledger is not None:
        iters = iterations_computed if iterations_computed is not None else [d] * b
        entry = ledger.record(
            iters, cost_per_iteration, call_id=call_id, sample_ids=sample_ids, kind="best_of_depth"
        )
    return BestOfDepthResult(output=selected, best_depth=best + 1, scores=scores, entry=entry)
