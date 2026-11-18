"""Subset search with capability as the objective (``docs/research_rl/CAPABILITY.md`` 1.11).

:class:`SubsetSearch` ranks candidate FFN-channel subsets (:class:`~research_rl.capability.types.ChannelGroup`)
by what a short training burst restricted to each subset does to **held-out capability**, not to
training reward. ``run(model)`` does exactly this:

1. **Start snapshot.** ``weights.capture(model, None)`` (every parameter). Every candidate, every
   finalist and every control starts from it; it is restored bit-identically after the search,
   also when anything raises (outer ``finally``).
2. **One evaluation** of a selection (``None`` = unrestricted) on an item key ``(purpose, index)``:

   a. ``weights.restore(model, start)``; for every probe slice draw the items of the key and score
      the **start weights** on them (:meth:`~research_rl.capability.probe.CapabilityProbe.score_slice`,
      one ``score_fn`` call per slice).
   b. ``op = make_operator(selection)``. It is built *after* the restore because forward-only
      spaces (e.g. :class:`~research_rl.capability.spaces.ChannelGainSpace`) take their base at
      construction. ``eval_snapshot = weights.capture(model, op.parameter_names(),
      optimizer=op.optimizer)`` records the operator's parameters and, if the operator has an
      optimizer, its state (and ``HookedOptimizer`` masters).
   c. **Burst**: ``op.step(model, update_offset + k)`` for ``k = 0 .. burst_updates - 1``.
      Steps b, c and d run inside ``torch.random.fork_rng`` with the CPU generator (and a CUDA
      model's device generator; NOT_TESTED) seeded with :func:`burst_seed` ``(seed)``. Every burst
      of the search uses the same update indices and the same RNG seed (common random numbers:
      candidates differ only in their subset), and the global RNG state is unchanged afterwards.
   d. Score the trained weights on the **same** items (one ``score_fn`` call per slice, through
      :func:`~research_rl.capability.probe.score_items`).

   **Scoring RNG.** Every ``score_fn`` call (start weights in a, trained weights in d) runs inside
   its own ``torch.random.fork_rng`` seeded with :func:`score_seed` ``(seed, purpose, index,
   slice)``: a score function that draws from the global torch RNG (a sampled / pass@k scorer)
   sees the **same** random stream for the start and the trained weights of a slice (common
   random numbers for the paired difference), and its draws depend on ``seed`` and the item key
   only, never on the caller's RNG state or on what the burst consumed. Greedy score functions
   are unaffected.
   e. ``finally``: ``weights.restore(model, eval_snapshot, optimizer=op.optimizer)`` (parameters,
      optimizer state and masters back to step b), then ``weights.restore(model, start)``, every
      parameter's ``.grad`` set to ``None`` and every submodule's training mode written back. On
      the normal path ``op.on_decision(GateDecision.ROLLBACK)`` is then called (the operator is
      told that its burst was undone; a forward-only operator re-bases from the restored weights).

3. **Objective** of an evaluation (point estimates, no test):
   ``gain`` = pooled mean of ``candidate - start`` over the items of every transfer slice
   (0.0 when the probe has no transfer slice); ``worst_drop`` = max over retention slices of the
   slice mean of ``start - candidate``; ``objective = gain - penalty * max(0, worst_drop - margin)``.
4. **Successive halving.** Round ``r = 0 .. rounds - 1``: every surviving candidate ``c`` (index in
   ``groups``) is evaluated on its own fresh items, key ``(f"search/{r}/{c}", index)``
   (:func:`search_purpose`); the best ``max(1, ceil(survivors_fraction * n))`` by that round's
   objective survive (ties: lower candidate index first). A round is skipped when a single
   candidate is left. Each round ranks on its own items only (earlier rounds are not averaged in).
   Round-stage candidates are scored on **different** item sets, so the within-round ranking is
   unpaired across candidates: it carries item-sampling noise on top of each candidate's paired
   gain vs the start weights (the final stage shares items across finalists and controls).
5. **Finalists** (the survivors) are re-evaluated on ``repeats_final`` new item sets, keys
   ``(f"search/final/{j}", index)`` (:func:`final_purpose`), shared by every finalist and every
   control: one burst per finalist, scored on all ``repeats_final`` sets (the start weights are
   scored once per set). Finalists are ranked by the mean objective over the sets (ties: earlier
   finalist first); the first is the **winner**. The burst is deterministic (seeded operator and
   :func:`burst_seed`), so the repeats average **item** noise only, not training noise.
6. **Controls**, each one burst of ``burst_updates`` updates (the same per-candidate burst budget
   as each finalist) scored on the same final item sets:

   * ``"random"``: :func:`~research_rl.capability.scores.density_matched_random` of the winner's
     selection with seed :func:`control_seed` ``(seed)`` (same per-layer counts);
   * ``"whole_layer"``: :func:`whole_layer_selection` of the winner (same number of channels in
     whole layers plus at most one contiguous partial block per selection, placed to overlap the
     winner as little as possible; a partial block is forced whenever the winner's count is not a
     sum of whole layers, e.g. any winner smaller than one layer);
   * ``"unrestricted"``: ``make_operator(None)``.

   Every control's :class:`FinalResult` records ``winner_overlap`` (channels shared with the
   winner; the winner's count for ``"unrestricted"``) and ``same_as_winner`` (the control trains
   exactly the winner's channels, e.g. ``"whole_layer"`` of a winner that already is whole
   layers, or ``"unrestricted"`` of a winner that is every channel: that comparison is then
   vacuous, and the report says so instead of hiding it).

   "Same budget" is per candidate: a control gets exactly one burst, as each finalist does in the
   final stage. The search as a whole spent many more bursts to *select* the winner; that cost is
   in :attr:`SearchReport.totals` and is not matched by the controls.

Items. Every draw goes through the probe's sampler with a ``search/...`` purpose, never the gate's
(``gate/<attempt>``) or the tracker's purposes, so search items are independent streams. They
come from ``pool`` (see :func:`~research_rl.capability.probe.draw_slice`), default
``"track"`` (:data:`~research_rl.capability.probe.TRACK_POOL`): the gate draws its evidence only
from the gate pool, so with the default **no search item is ever a gate item** (by construction)
and the subset is never selected on items the gate later accepts or rejects on. The probe's two
pools partition every uid, so search items cannot also be disjoint from the tracker's and the
start map's (track-pool) items: they coincide **by chance**. Measured with the test fixture's
sampler (``OperatorSet(seed=0, num_operators=10)``, ``root_seed=3``) at the gate's default sizes
(256 items per transfer and 2048 per retention slice, 4 candidates, 2 final repeats): 8-13% of
the search's uids per slice were among 30 tracker steps x 256 items, 0.1-1.1% among the 256
start-map items, and 1-21% of the final-set uids had already appeared in a round set (the track
pool is 1/8 of the domain; the small length-3 retention domains repeat most). With ``pool=None``
the same draws put 1.5-25% of the search's uids among 30 gate decisions' items (that is why it
is not the default); ``pool="gate"`` makes them coincide with gate items and never with
tracker items. The pool is in :meth:`SubsetSearch.config` and its consequence in
``SearchReport.extra["item_pool_note"]``. Limits of the default: the guarantee holds for a gate
on the **same** sampler (the pool hash includes ``root_seed``; a gate on another sampler has
another partition); and the track pool is 1/8 of each slice's domain, so a slice whose track
part cannot supply the requested items makes the draw raise ``ValueError`` (only the fixture
sampler's capacity was checked: 2048 items per slice worked) - pass ``pool=None`` or smaller
``items_per_*_slice`` then. AUDIT is never reachable (the probe refuses it).
``seed`` does not change the items (they depend on the sampler's ``root_seed``, the purpose and
``index``); use ``index`` to give a second search fresh items.

Candidates. A candidate is any :class:`ChannelGroup`; a union of several groups is one candidate
built with :func:`combine_groups`. Names must be distinct and selections non-empty.

Counting. ``forward_passes`` / ``backward_passes`` of a burst are the sums of the operators'
:class:`~research_rl.capability.types.OperatorReport` counts (the operators' own conventions:
generation calls for ES, generation + training forward for GRPO). Scoring is counted separately
as ``score_fn`` calls (``score_calls``) and items scored, start-weight scoring included in
:attr:`SearchReport.totals`.

RNG. ``run`` executes inside an outer ``torch.random.fork_rng`` (CPU, plus a CUDA model's device
generator; NOT_TESTED), so the caller's global RNG state is unchanged on return and on
exceptions, whatever ``make_operator``, the operators or the score function draw; every draw
the search causes comes from :func:`burst_seed` or :func:`score_seed`, so the report depends on
``seed`` (and the items), not on the caller's RNG state.

Limits. The ranking is a noisy point estimate on finitely many items after a short burst; it is
not a statistical test and makes no optimality claim (the :attr:`SearchReport.statement` says
so). ``.grad`` buffers, LR-scheduler state and buffers are not snapshotted (``.grad`` is cleared
after every evaluation). Verified on CPU with tiny models only
(``tests/research_rl/test_capability_search.py``); nothing here has run on a GPU (NOT_TESTED).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import nn

from research_rl.capability.gate import GateConfig
from research_rl.capability.probe import ITEM_POOLS, TRACK_POOL, CapabilityProbe, score_items
from research_rl.capability.scores import density_matched_random
from research_rl.capability.types import ChannelGroup, GateDecision, Item, Split, UpdateOperator
from research_rl.capability.weights import StateSnapshot, capture, restore
from research_rl.models.ffn_channels import ChannelSelection
from research_rl.models.model_interface import find_swiglu_ffns

__all__ = [
    "CONTROL_KINDS",
    "DEFAULT_MARGIN",
    "DEFAULT_PENALTY",
    "SEARCH_PURPOSE",
    "STATEMENT",
    "BurstRecord",
    "FinalResult",
    "MakeOperator",
    "SearchEvaluation",
    "SearchReport",
    "SliceResult",
    "SubsetSearch",
    "burst_seed",
    "combine_groups",
    "control_seed",
    "final_purpose",
    "objective",
    "score_seed",
    "search_purpose",
    "whole_layer_selection",
]

SEARCH_PURPOSE = "search"
"""Prefix of every sampler purpose used by the search."""
CONTROL_KINDS = ("random", "whole_layer", "unrestricted")
"""The controls :class:`SubsetSearch` can run (default: all three)."""
DEFAULT_PENALTY = 1.0
"""Default weight of the retention-drop excess in :func:`objective` (one point of drop beyond the
margin costs one point of transfer gain). A choice, not a tuned value."""
DEFAULT_MARGIN = GateConfig().retention_margin
"""Default allowed retention drop (the gate's default ``retention_margin``, 0.02)."""
STATEMENT = (
    "The ranking is a noisy point estimate: each candidate's objective is measured on a finite set "
    "of fresh items after one short, seeded training burst from the start weights. It is not a "
    "statistical test and makes no optimality claim; the winner is not shown to be the best "
    "subset, and finalist repeats average item noise only, not training noise. Controls match the "
    "per-candidate burst budget, not the total compute spent selecting the winner."
)

MakeOperator = Callable[[ChannelSelection | None], UpdateOperator]
"""``make_operator(selection)``: an :class:`UpdateOperator` restricted to ``selection``
(``None`` = the unrestricted control). Called with the model at the start weights; it must not
modify the weights."""


# ============================================================================== helpers
def _seed_of(*parts: object) -> int:
    text = "\x1f".join(repr(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


def burst_seed(seed: int) -> int:
    """Seed of the global torch RNG inside every burst of a search with ``seed`` (a 63-bit int
    from SHA-256 of ``("research_rl.capability.search", "burst", seed)``)."""
    return _seed_of("research_rl.capability.search", "burst", int(seed))


def control_seed(seed: int) -> int:
    """Seed passed to :func:`~research_rl.capability.scores.density_matched_random` for the
    ``"random"`` control of a search with ``seed``."""
    return _seed_of("research_rl.capability.search", "random_control", int(seed))


def score_seed(seed: int, purpose: str, index: int, slice_name: str) -> int:
    """Seed of the global torch RNG inside every ``score_fn`` call on slice ``slice_name`` of the
    item set ``(purpose, index)`` of a search with ``seed`` (a 63-bit int from SHA-256 of
    ``("research_rl.capability.search", "score", seed, purpose, index, slice_name)``). The start
    and the trained weights of one evaluation use the same seed for the same slice."""
    return _seed_of(
        "research_rl.capability.search", "score", int(seed), str(purpose), int(index), str(slice_name)
    )


def search_purpose(round_index: int, candidate_index: int) -> str:
    """Sampler purpose of candidate ``candidate_index`` in round ``round_index``:
    ``f"search/{round_index}/{candidate_index}"``."""
    return f"{SEARCH_PURPOSE}/{_check_index('round_index', round_index)}/{_check_index('candidate_index', candidate_index)}"


def final_purpose(repeat: int) -> str:
    """Sampler purpose of final repeat ``repeat`` (shared by finalists and controls):
    ``f"search/final/{repeat}"``."""
    return f"{SEARCH_PURPOSE}/final/{_check_index('repeat', repeat)}"


def _check_index(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an int >= 0, got {value!r}")
    return int(value)


def _check_count(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    return int(value)


def _check_number(name: str, value: Any, *, low: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < low
    ):
        raise ValueError(f"{name} must be a finite number >= {low}, got {value!r}")
    return float(value)


def objective(gain: float, worst_drop: float, *, penalty: float, margin: float) -> float:
    """``gain - penalty * max(0, worst_drop - margin)``."""
    return float(gain) - float(penalty) * max(0.0, float(worst_drop) - float(margin))


def combine_groups(groups: Sequence[ChannelGroup], *, name: str | None = None) -> ChannelGroup:
    """One candidate made of the union of ``groups`` (``ValueError`` for no groups). The default
    name joins the group names with ``"+"``."""
    groups = list(groups)
    if not groups:
        raise ValueError("combine_groups needs at least one group")
    selection = ChannelSelection()
    for g in groups:
        selection = selection.union(g.selection)
    return ChannelGroup(
        name=name if name is not None else "+".join(g.name for g in groups), selection=selection
    )


def _least_overlap_start(size: int, take: int, chosen: Sequence[int]) -> int:
    """Lowest ``s`` in ``0 .. size - take`` minimising ``|range(s, s + take) & chosen|``."""
    marks = [0] * (size + 1)
    for c in chosen:
        marks[c + 1] = 1
    for i in range(size):
        marks[i + 1] += marks[i]
    best_start, best = 0, marks[take] - marks[0]
    for start in range(1, size - take + 1):
        overlap = marks[start + take] - marks[start]
        if overlap < best:
            best_start, best = start, overlap
    return best_start


def whole_layer_selection(selection: ChannelSelection, model: nn.Module) -> ChannelSelection:
    """``selection.count()`` channels as whole layers plus at most one contiguous partial block.

    Only the selection's own layers are used, visited by descending selected count (ties: lower
    layer first); each visited layer contributes ``min(remaining, intermediate_size)`` channels,
    so every visited layer but the last is taken whole (the own layers always suffice, since each
    holds at most its size). When the last one is not taken whole it gets **one contiguous block**
    ``range(s, s + take)`` whose start ``s`` minimises the number of the selection's own channels
    of that layer inside the block (ties: lowest ``s``), so a contiguous winner such as
    ``{0: range(0, 8)}`` gets a different block (``{0: range(8, 16)}``) instead of itself. Layers
    left over once the count is reached are not used. Deterministic (no randomness).

    Limits: a partial block is unavoidable when the count is not a sum of whole layers (e.g. any
    selection smaller than one layer becomes a single block of its largest layer); and a selection
    that already is whole layers of its own layers is returned unchanged (identity cannot be
    avoided within its own layers) - :class:`SubsetSearch` flags that case in
    :attr:`FinalResult.same_as_winner`. Validates ``selection`` against ``model``
    (``ChannelSelectionError``); ``ValueError`` for an empty selection.
    """
    selection.validate(model)
    total = selection.count()
    if total == 0:
        raise ValueError("whole_layer_selection needs a non-empty selection")
    sizes = {r.layer_index: r.intermediate_size for r in find_swiglu_ffns(model, selection.layers())}
    order = sorted(selection.layers(), key=lambda layer: (-len(selection.channels(layer)), layer))
    out: dict[int, range] = {}
    remaining = total
    for layer in order:
        if remaining == 0:
            break
        take = min(remaining, sizes[layer])
        start = _least_overlap_start(sizes[layer], take, selection.channels(layer))
        out[layer] = range(start, start + take)
        remaining -= take
    return ChannelSelection(out)


def _uid_digest(items: Iterable[Item]) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(it.uid.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


@contextmanager
def _seeded_rng(devices: Sequence[int], rng_seed: int) -> Iterator[None]:
    """Fork the global CPU RNG (and the CUDA generators of ``devices``; NOT_TESTED), seed it with
    ``rng_seed`` and restore the previous state on exit."""
    with torch.random.fork_rng(devices=list(devices)):
        torch.random.default_generator.manual_seed(rng_seed)
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(rng_seed)
        yield


@contextmanager
def _keep_modes(model: nn.Module) -> Iterator[None]:
    modes = [(m, m.training) for m in model.modules()]
    try:
        yield
    finally:
        for m, was_training in modes:
            m.training = was_training


def _rng_devices(model: nn.Module) -> list[int]:
    device = next(model.parameters()).device
    if device.type == "cuda":
        return [device.index if device.index is not None else torch.cuda.current_device()]
    return []


def _pool_note(pool: str | None) -> str:
    if pool == TRACK_POOL:
        return (
            "pool='track': no search item is a gate item (by construction); search items can "
            "coincide by chance with tracker and start-map items (same pool)."
        )
    if pool is None:
        return (
            "pool=None (unpartitioned): search items can coincide by chance with gate items, so "
            "the subset may have been selected on items the gate later decides on, and with "
            "tracker items."
        )
    return (
        f"pool={pool!r}: search items can coincide by chance with gate items (the subset may "
        "have been selected on items the gate later decides on); never with tracker items."
    )


def _selection_dict(selection: ChannelSelection | None) -> dict[str, list[int]] | None:
    if selection is None:
        return None
    return {str(layer): list(chans) for layer, chans in selection.items()}


# ============================================================================== records
@dataclass(frozen=True)
class SliceResult:
    """Start and trained-candidate mean scores of one slice in one evaluation."""

    slice: str
    split: Split
    n: int
    start_mean: float
    candidate_mean: float

    @property
    def difference(self) -> float:
        """``candidate_mean - start_mean`` (positive = the burst helped)."""
        return self.candidate_mean - self.start_mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice": self.slice,
            "split": self.split.value,
            "n": self.n,
            "start_mean": self.start_mean,
            "candidate_mean": self.candidate_mean,
            "difference": self.difference,
        }


@dataclass(frozen=True)
class BurstRecord:
    """One training burst from the start weights.

    ``kind`` is ``"candidate"`` or the control name; ``stage`` is ``"round"`` or ``"final"``;
    ``update_indices`` are the operator update indices used; ``forward_passes`` /
    ``backward_passes`` are the sums of the operator reports; ``reward_v1`` / ``reward_v2`` the
    per-update training rewards (``reward_v2`` entries may be ``None``); ``rng_seed`` the global
    RNG seed of the burst.
    """

    candidate: str
    kind: str
    stage: str
    round: int | None
    channels: int | None
    update_indices: tuple[int, ...]
    forward_passes: int
    backward_passes: int
    reward_v1: tuple[float, ...]
    reward_v2: tuple[float | None, ...]
    rng_seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "kind": self.kind,
            "stage": self.stage,
            "round": self.round,
            "channels": self.channels,
            "updates": len(self.update_indices),
            "update_indices": list(self.update_indices),
            "forward_passes": self.forward_passes,
            "backward_passes": self.backward_passes,
            "reward_v1": list(self.reward_v1),
            "reward_v2": list(self.reward_v2),
            "rng_seed": self.rng_seed,
        }


@dataclass(frozen=True)
class SearchEvaluation:
    """One scoring of trained weights on one item set ``(purpose, index)``.

    ``burst`` indexes :attr:`SearchReport.bursts` (the burst that produced the weights; the
    ``repeats_final`` evaluations of a finalist or control share one burst, so their
    ``forward_passes`` / ``backward_passes``, copied from it, are not additive). ``score_calls``
    counts the ``score_fn`` calls that scored the trained weights here (the start-weight scoring
    is counted in :attr:`SearchReport.totals`). ``uid_digest`` is SHA-256 over the item uids in
    order (equal digests = the same items).
    """

    candidate: str
    kind: str
    stage: str
    round: int | None
    repeat: int | None
    purpose: str
    index: int
    burst: int
    gain: float
    worst_drop: float
    worst_slice: str
    objective: float
    slices: tuple[SliceResult, ...]
    uid_digest: str
    items: int
    forward_passes: int
    backward_passes: int
    score_calls: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "kind": self.kind,
            "stage": self.stage,
            "round": self.round,
            "repeat": self.repeat,
            "purpose": self.purpose,
            "index": self.index,
            "burst": self.burst,
            "gain": self.gain,
            "worst_drop": self.worst_drop,
            "worst_slice": self.worst_slice,
            "objective": self.objective,
            "slices": [s.to_dict() for s in self.slices],
            "uid_digest": self.uid_digest,
            "items": self.items,
            "forward_passes": self.forward_passes,
            "backward_passes": self.backward_passes,
            "score_calls": self.score_calls,
        }


@dataclass(frozen=True)
class FinalResult:
    """A finalist or control over the final item sets: per-repeat objectives and their means.

    ``selection`` is ``None`` for the unrestricted control; ``channels`` its channel count
    (``None`` when unrestricted). ``forward_passes`` / ``backward_passes`` are the counts of its
    single final burst. For a control, ``winner_overlap`` is the number of channels it shares
    with the winner (the winner's count for the unrestricted control) and ``same_as_winner`` is
    True iff it trains exactly the winner's channels (the comparison is then vacuous); both are
    ``None`` for finalists.
    """

    candidate: str
    kind: str
    selection: dict[str, list[int]] | None
    channels: int | None
    objectives: tuple[float, ...]
    gains: tuple[float, ...]
    worst_drops: tuple[float, ...]
    burst: int
    forward_passes: int
    backward_passes: int
    winner_overlap: int | None = None
    same_as_winner: bool | None = None

    @property
    def mean_objective(self) -> float:
        return sum(self.objectives) / len(self.objectives)

    @property
    def mean_gain(self) -> float:
        return sum(self.gains) / len(self.gains)

    @property
    def mean_worst_drop(self) -> float:
        return sum(self.worst_drops) / len(self.worst_drops)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "kind": self.kind,
            "selection": self.selection,
            "channels": self.channels,
            "objectives": list(self.objectives),
            "gains": list(self.gains),
            "worst_drops": list(self.worst_drops),
            "mean_objective": self.mean_objective,
            "mean_gain": self.mean_gain,
            "mean_worst_drop": self.mean_worst_drop,
            "burst": self.burst,
            "forward_passes": self.forward_passes,
            "backward_passes": self.backward_passes,
            "winner_overlap": self.winner_overlap,
            "same_as_winner": self.same_as_winner,
        }


@dataclass
class SearchReport:
    """Everything one :meth:`SubsetSearch.run` did (JSON-able via :meth:`to_dict`; no timestamps,
    so equal runs give equal reports).

    * ``config``: the search settings; ``candidates``: ``{name: selection}`` in input order.
    * ``rounds``: the candidate names entering each round that was run; ``evaluations``: every
      evaluation (rounds, then finalists, then controls); ``bursts``: every burst.
    * ``finalists``: finalist names ranked by mean final objective (best first); ``final``: their
      :class:`FinalResult` in that order; ``controls``: ``{control: FinalResult}``.
    * ``winner`` / ``winner_selection``: the top finalist.
    * ``totals``: bursts, operator updates, forward / backward passes, ``score_fn`` calls and items
      scored over the whole search (start-weight scoring included).
    * ``start_fingerprint``: fingerprint of the start snapshot (the weights after the search).
    * ``statement``: :data:`STATEMENT` (the ranking is a noisy estimate; no optimality claim).
    * ``extra["item_pool_note"]``: what the item ``pool`` guarantees about overlap with gate and
      tracker items (module docstring, "Items").
    """

    config: dict[str, Any]
    candidates: dict[str, dict[str, list[int]]]
    rounds: list[list[str]]
    evaluations: list[SearchEvaluation]
    bursts: list[BurstRecord]
    finalists: list[str]
    final: list[FinalResult]
    controls: dict[str, FinalResult]
    winner: str
    winner_selection: dict[str, list[int]]
    totals: dict[str, int]
    start_fingerprint: str
    statement: str = STATEMENT
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": dict(self.config),
            "candidates": {k: dict(v) for k, v in self.candidates.items()},
            "rounds": [list(r) for r in self.rounds],
            "evaluations": [e.to_dict() for e in self.evaluations],
            "bursts": [b.to_dict() for b in self.bursts],
            "finalists": list(self.finalists),
            "final": [f.to_dict() for f in self.final],
            "controls": {k: v.to_dict() for k, v in self.controls.items()},
            "winner": self.winner,
            "winner_selection": dict(self.winner_selection),
            "totals": dict(self.totals),
            "start_fingerprint": self.start_fingerprint,
            "statement": self.statement,
            "extra": dict(self.extra),
        }


# ============================================================================== the search
@dataclass(frozen=True)
class _ItemSet:
    purpose: str
    index: int
    slices: tuple[tuple[str, Split, tuple[Item, ...], tuple[float, ...]], ...]

    @property
    def items(self) -> list[Item]:
        return [it for _, _, items, _ in self.slices for it in items]


@dataclass(frozen=True)
class _Trained:
    burst: int
    scores: tuple[tuple[float, ...], ...]  # per item set, all slices concatenated in slice order


class SubsetSearch:
    """Successive-halving search over channel subsets (module docstring for exact semantics).

    Args:
        groups: the candidates (distinct names, non-empty selections; at least one).
        make_operator: ``selection -> UpdateOperator`` (``None`` = unrestricted control).
        probe: supplies the sampler, the score function and the transfer / retention slices.
        rounds: halving rounds (>= 1).
        burst_updates: operator updates per burst (>= 1).
        survivors_fraction: share kept per round, in (0, 1]; at least one survives.
        repeats_final: final item sets (>= 1).
        controls: subset of :data:`CONTROL_KINDS`, no duplicates (may be empty).
        penalty: weight of the retention-drop excess (finite, >= 0).
        margin: allowed retention drop before the penalty applies (finite, >= 0).
        seed: root of the burst RNG seed and the random control (not of the items).
        items_per_transfer_slice / items_per_retention_slice: items per slice per evaluation
            (default: the probe's).
        index: sampler ``index`` of every draw (give a second search on the same sampler another
            value to get fresh items).
        pool: ``"track"`` (default: never a gate item), ``"gate"`` or ``None`` (unpartitioned; can
            coincide with gate items) - module docstring, "Items".
    """

    def __init__(
        self,
        groups: Sequence[ChannelGroup],
        make_operator: MakeOperator,
        probe: CapabilityProbe,
        *,
        rounds: int,
        burst_updates: int,
        survivors_fraction: float = 0.5,
        repeats_final: int = 2,
        controls: Sequence[str] = CONTROL_KINDS,
        penalty: float = DEFAULT_PENALTY,
        margin: float = DEFAULT_MARGIN,
        seed: int = 0,
        items_per_transfer_slice: int | None = None,
        items_per_retention_slice: int | None = None,
        index: int = 0,
        pool: str | None = TRACK_POOL,
    ) -> None:
        groups = list(groups)
        if not groups:
            raise ValueError("SubsetSearch needs at least one candidate group")
        for g in groups:
            if not isinstance(g, ChannelGroup):
                raise TypeError(f"groups must be ChannelGroups, got {type(g).__name__}")
            if not g.selection:
                raise ValueError(f"candidate {g.name!r} has an empty selection")
        names = [g.name for g in groups]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"duplicate candidate names: {dup}")
        clash = sorted(set(names) & set(CONTROL_KINDS))
        if clash:
            raise ValueError(f"candidate names may not be control names: {clash}")
        if not callable(make_operator):
            raise TypeError("make_operator must be callable")
        if not isinstance(probe, CapabilityProbe):
            raise TypeError(f"probe must be a CapabilityProbe, got {type(probe).__name__}")
        controls = tuple(controls)
        unknown = [c for c in controls if c not in CONTROL_KINDS]
        if unknown:
            raise ValueError(f"unknown controls {unknown}; known: {CONTROL_KINDS}")
        if len(set(controls)) != len(controls):
            raise ValueError(f"duplicate controls: {controls}")
        frac = survivors_fraction
        if isinstance(frac, bool) or not isinstance(frac, (int, float)) or not (0.0 < frac <= 1.0):
            raise ValueError(f"survivors_fraction must be in (0, 1], got {frac!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an int, got {seed!r}")
        if pool is not None and pool not in ITEM_POOLS:
            raise ValueError(f"pool must be None or one of {ITEM_POOLS}, got {pool!r}")
        self.groups: tuple[ChannelGroup, ...] = tuple(groups)
        self.make_operator = make_operator
        self.probe = probe
        self.rounds = _check_count("rounds", rounds)
        self.burst_updates = _check_count("burst_updates", burst_updates)
        self.survivors_fraction = float(frac)
        self.repeats_final = _check_count("repeats_final", repeats_final)
        self.controls: tuple[str, ...] = controls
        self.penalty = _check_number("penalty", penalty)
        self.margin = _check_number("margin", margin)
        self.seed = int(seed)
        self.items_per_transfer_slice = _check_count(
            "items_per_transfer_slice",
            probe.items_per_transfer_slice if items_per_transfer_slice is None else items_per_transfer_slice,
        )
        self.items_per_retention_slice = _check_count(
            "items_per_retention_slice",
            probe.items_per_retention_slice
            if items_per_retention_slice is None
            else items_per_retention_slice,
        )
        self.index = _check_index("index", index)
        self.pool = pool

    # ---------------------------------------------------------------- configuration
    def config(self) -> dict[str, Any]:
        """JSON-able settings (the probe's slices and sampler identity included)."""
        return {
            "rounds": self.rounds,
            "burst_updates": self.burst_updates,
            "survivors_fraction": self.survivors_fraction,
            "repeats_final": self.repeats_final,
            "controls": list(self.controls),
            "penalty": self.penalty,
            "margin": self.margin,
            "seed": self.seed,
            "burst_seed": burst_seed(self.seed),
            "items_per_transfer_slice": self.items_per_transfer_slice,
            "items_per_retention_slice": self.items_per_retention_slice,
            "index": self.index,
            "pool": self.pool,
            "transfer_slices": list(self.probe.transfer_slices),
            "retention_slices": list(self.probe.retention_slices),
            "sampler": self.probe.sampler.state_dict(),
        }

    def survivors_after(self, n: int) -> int:
        """Candidates kept from a round of ``n``: ``max(1, ceil(survivors_fraction * n))``."""
        return max(1, math.ceil(self.survivors_fraction * _check_count("n", n) - 1e-12))

    # ---------------------------------------------------------------- internals
    def _slice_names(self) -> list[tuple[str, int]]:
        return [
            *((s, self.items_per_transfer_slice) for s in self.probe.transfer_slices),
            *((s, self.items_per_retention_slice) for s in self.probe.retention_slices),
        ]

    def _draw_and_score_start(self, model: nn.Module, purpose: str, totals: dict[str, int]) -> _ItemSet:
        """Model must hold the start weights. Each slice is drawn and scored inside
        ``_seeded_rng(score_seed(seed, purpose, index, slice))``."""
        devices = _rng_devices(model)
        out = []
        for name, n in self._slice_names():
            with _seeded_rng(devices, score_seed(self.seed, purpose, self.index, name)):
                s = self.probe.score_slice(model, name, n, purpose=purpose, index=self.index, pool=self.pool)
            out.append((name, s.split, s.items, s.scores))
            totals["score_calls"] += 1
            totals["items_scored"] += s.n
        return _ItemSet(purpose=purpose, index=self.index, slices=tuple(out))

    def _measure(
        self, item_set: _ItemSet, trained: Sequence[float]
    ) -> tuple[float, float, str, list[SliceResult]]:
        pos = 0
        results: list[SliceResult] = []
        gain_sum = 0.0
        gain_n = 0
        worst_drop = -math.inf
        worst_slice = ""
        transfer = set(self.probe.transfer_slices)
        for name, split, items, start in item_set.slices:
            cand = trained[pos : pos + len(items)]
            pos += len(items)
            start_mean = sum(start) / len(start)
            cand_mean = sum(cand) / len(cand)
            results.append(SliceResult(name, split, len(items), start_mean, cand_mean))
            if name in transfer:
                gain_sum += sum(c - s for c, s in zip(cand, start, strict=True))
                gain_n += len(items)
            else:
                drop = start_mean - cand_mean
                if drop > worst_drop:
                    worst_drop, worst_slice = drop, name
        gain = gain_sum / gain_n if gain_n else 0.0
        return gain, worst_drop, worst_slice, results

    def _train_and_score(
        self,
        model: nn.Module,
        start: StateSnapshot,
        selection: ChannelSelection | None,
        item_sets: Sequence[_ItemSet],
        *,
        candidate: str,
        kind: str,
        stage: str,
        round_index: int | None,
        update_offset: int,
        bursts: list[BurstRecord],
        totals: dict[str, int],
    ) -> _Trained:
        restore(model, start)
        rng_seed = burst_seed(self.seed)
        indices = tuple(update_offset + k for k in range(self.burst_updates))
        op: UpdateOperator | None = None
        eval_snapshot: StateSnapshot | None = None
        scores: list[tuple[float, ...]] = []
        forward = backward = 0
        rewards_v1: list[float] = []
        rewards_v2: list[float | None] = []
        try:
            devices = _rng_devices(model)
            with _keep_modes(model), _seeded_rng(devices, rng_seed):
                op = self.make_operator(selection)
                if not isinstance(op, UpdateOperator):
                    raise TypeError(f"make_operator returned {type(op).__name__}, not an UpdateOperator")
                eval_snapshot = capture(model, op.parameter_names(), optimizer=op.optimizer)
                for t in indices:
                    report = op.step(model, t)
                    forward += int(report.forward_passes)
                    backward += int(report.backward_passes)
                    rewards_v1.append(float(report.reward_v1))
                    rewards_v2.append(None if report.reward_v2 is None else float(report.reward_v2))
                for item_set in item_sets:
                    row: list[float] = []
                    for name, _, items, _ in item_set.slices:
                        seed = score_seed(self.seed, item_set.purpose, item_set.index, name)
                        with _seeded_rng(devices, seed):
                            row.extend(score_items(self.probe.score_fn, model, items))
                        totals["score_calls"] += 1
                        totals["items_scored"] += len(items)
                    scores.append(tuple(row))
        finally:
            try:
                if op is not None and eval_snapshot is not None:
                    restore(model, eval_snapshot, optimizer=op.optimizer)
            finally:
                restore(model, start)
                for p in model.parameters():
                    p.grad = None
        assert op is not None
        op.on_decision(GateDecision.ROLLBACK)
        bursts.append(
            BurstRecord(
                candidate=candidate,
                kind=kind,
                stage=stage,
                round=round_index,
                channels=None if selection is None else selection.count(),
                update_indices=indices,
                forward_passes=forward,
                backward_passes=backward,
                reward_v1=tuple(rewards_v1),
                reward_v2=tuple(rewards_v2),
                rng_seed=rng_seed,
            )
        )
        totals["bursts"] += 1
        totals["updates"] += len(indices)
        totals["forward_passes"] += forward
        totals["backward_passes"] += backward
        return _Trained(burst=len(bursts) - 1, scores=tuple(scores))

    def _evaluation(
        self,
        item_set: _ItemSet,
        scores: Sequence[float],
        burst: BurstRecord,
        burst_index: int,
        *,
        candidate: str,
        kind: str,
        stage: str,
        round_index: int | None,
        repeat: int | None,
    ) -> SearchEvaluation:
        gain, worst_drop, worst_slice, results = self._measure(item_set, scores)
        return SearchEvaluation(
            candidate=candidate,
            kind=kind,
            stage=stage,
            round=round_index,
            repeat=repeat,
            purpose=item_set.purpose,
            index=item_set.index,
            burst=burst_index,
            gain=gain,
            worst_drop=worst_drop,
            worst_slice=worst_slice,
            objective=objective(gain, worst_drop, penalty=self.penalty, margin=self.margin),
            slices=tuple(results),
            uid_digest=_uid_digest(item_set.items),
            items=len(item_set.items),
            forward_passes=burst.forward_passes,
            backward_passes=burst.backward_passes,
            score_calls=len(item_set.slices),
        )

    def _final(
        self,
        model: nn.Module,
        start: StateSnapshot,
        selection: ChannelSelection | None,
        final_sets: Sequence[_ItemSet],
        *,
        candidate: str,
        kind: str,
        update_offset: int,
        bursts: list[BurstRecord],
        evaluations: list[SearchEvaluation],
        totals: dict[str, int],
    ) -> FinalResult:
        trained = self._train_and_score(
            model,
            start,
            selection,
            final_sets,
            candidate=candidate,
            kind=kind,
            stage="final",
            round_index=None,
            update_offset=update_offset,
            bursts=bursts,
            totals=totals,
        )
        burst = bursts[trained.burst]
        evs = [
            self._evaluation(
                item_set,
                scores,
                burst,
                trained.burst,
                candidate=candidate,
                kind=kind,
                stage="final",
                round_index=None,
                repeat=j,
            )
            for j, (item_set, scores) in enumerate(zip(final_sets, trained.scores, strict=True))
        ]
        evaluations.extend(evs)
        return FinalResult(
            candidate=candidate,
            kind=kind,
            selection=_selection_dict(selection),
            channels=None if selection is None else selection.count(),
            objectives=tuple(e.objective for e in evs),
            gains=tuple(e.gain for e in evs),
            worst_drops=tuple(e.worst_drop for e in evs),
            burst=trained.burst,
            forward_passes=burst.forward_passes,
            backward_passes=burst.backward_passes,
        )

    # ---------------------------------------------------------------- run
    def run(self, model: nn.Module, *, update_offset: int = 0) -> SearchReport:
        """Run the search on ``model`` and return the :class:`SearchReport`.

        ``update_offset`` (>= 0) is the first operator update index of every burst. On return, and
        when anything raises, every parameter of ``model`` is bit-identical to its value at the
        call, every submodule's training mode is restored, ``.grad`` is ``None`` for every
        parameter, the global RNG state (CPU, and a CUDA model's device; the latter NOT_TESTED) is
        unchanged (outer ``fork_rng``; the report does not depend on it), and the optimizer of
        every operator built by ``make_operator`` is back to its state at construction (so an
        optimizer shared across operators ends where it started). Candidate selections are validated against ``model``
        before any training (``ChannelSelectionError``).
        """
        update_offset = _check_index("update_offset", update_offset)
        for g in self.groups:
            g.selection.validate(model)
        totals = {
            "bursts": 0,
            "updates": 0,
            "forward_passes": 0,
            "backward_passes": 0,
            "score_calls": 0,
            "items_scored": 0,
        }
        bursts: list[BurstRecord] = []
        evaluations: list[SearchEvaluation] = []
        rounds_run: list[list[str]] = []
        start = capture(model, None)
        try:
            with _keep_modes(model), torch.random.fork_rng(devices=_rng_devices(model)):
                survivors = list(range(len(self.groups)))
                for r in range(self.rounds):
                    if len(survivors) <= 1:
                        break
                    rounds_run.append([self.groups[c].name for c in survivors])
                    scored: list[tuple[float, int]] = []
                    for c in survivors:
                        group = self.groups[c]
                        restore(model, start)
                        item_set = self._draw_and_score_start(model, search_purpose(r, c), totals)
                        trained = self._train_and_score(
                            model,
                            start,
                            group.selection,
                            [item_set],
                            candidate=group.name,
                            kind="candidate",
                            stage="round",
                            round_index=r,
                            update_offset=update_offset,
                            bursts=bursts,
                            totals=totals,
                        )
                        ev = self._evaluation(
                            item_set,
                            trained.scores[0],
                            bursts[trained.burst],
                            trained.burst,
                            candidate=group.name,
                            kind="candidate",
                            stage="round",
                            round_index=r,
                            repeat=None,
                        )
                        evaluations.append(ev)
                        scored.append((ev.objective, c))
                    keep = self.survivors_after(len(survivors))
                    ranked = sorted(scored, key=lambda oc: (-oc[0], oc[1]))
                    survivors = [c for _, c in ranked[:keep]]

                restore(model, start)
                final_sets = [
                    self._draw_and_score_start(model, final_purpose(j), totals)
                    for j in range(self.repeats_final)
                ]
                finals: list[tuple[FinalResult, int]] = []
                for order, c in enumerate(survivors):
                    group = self.groups[c]
                    res = self._final(
                        model,
                        start,
                        group.selection,
                        final_sets,
                        candidate=group.name,
                        kind="candidate",
                        update_offset=update_offset,
                        bursts=bursts,
                        evaluations=evaluations,
                        totals=totals,
                    )
                    finals.append((res, order))
                finals.sort(key=lambda fo: (-fo[0].mean_objective, fo[1]))
                ranked_final = [f for f, _ in finals]
                winner = next(g for g in self.groups if g.name == ranked_final[0].candidate)

                control_results: dict[str, FinalResult] = {}
                for kind in self.controls:
                    selection: ChannelSelection | None
                    if kind == "random":
                        selection = density_matched_random(winner.selection, model, control_seed(self.seed))
                    elif kind == "whole_layer":
                        selection = whole_layer_selection(winner.selection, model)
                    else:
                        selection = None
                    result = self._final(
                        model,
                        start,
                        selection,
                        final_sets,
                        candidate=kind,
                        kind=kind,
                        update_offset=update_offset,
                        bursts=bursts,
                        evaluations=evaluations,
                        totals=totals,
                    )
                    if selection is None:
                        overlap = winner.selection.count()
                        same = winner.selection == ChannelSelection.all_channels(model)
                    else:
                        overlap = selection.intersection(winner.selection).count()
                        same = selection == winner.selection
                    control_results[kind] = replace(result, winner_overlap=overlap, same_as_winner=same)
        finally:
            restore(model, start)
        winner_selection = _selection_dict(winner.selection)
        assert winner_selection is not None
        return SearchReport(
            config=self.config(),
            candidates={g.name: _selection_dict(g.selection) or {} for g in self.groups},
            rounds=rounds_run,
            evaluations=evaluations,
            bursts=bursts,
            finalists=[f.candidate for f in ranked_final],
            final=ranked_final,
            controls=control_results,
            winner=winner.name,
            winner_selection=winner_selection,
            totals=totals,
            start_fingerprint=start.fingerprint,
            extra={"item_pool_note": _pool_note(self.pool)},
        )
