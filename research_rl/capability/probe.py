"""Gate evidence and start-of-run capability maps (``docs/research_rl/CAPABILITY.md`` 1.8).

* :class:`CapabilityProbe` - draws **fresh** held-out items for one gate evaluation
  ``(decision_index, attempt)``, scores the *candidate* (the model's current weights) and the
  *incumbent* (a :class:`~research_rl.capability.weights.StateSnapshot`, loaded temporarily with
  :func:`~research_rl.capability.weights.swapped`) on the **same** items and returns
  :class:`~research_rl.capability.types.GateEvidence` for
  :meth:`~research_rl.capability.gate.CapabilityGate.verdict`.
* :func:`score_slice` / :meth:`CapabilityProbe.score_slice` - the lower-level "draw ``n`` items of
  one slice for ``(purpose, index)`` and score them" path, shared with the capability tracker and
  the subset search. It returns a :class:`SliceScores`.
* :class:`StartMap` - pass@1 (greedy, through the score function) and pass@k (sampled, seeded,
  unbiased estimator :func:`~research_rl.evaluation.metrics.pass_at_k`) of the start checkpoint
  per slice; JSON-able.

Item streams. Every draw goes through :meth:`~research_rl.capability.tasks.ItemSampler.draw`
with an explicit ``(purpose, index)`` key: gate evidence uses ``purpose=f"gate/{attempt}"`` and
``index=decision_index`` (:func:`gate_purpose`), the start map ``purpose="start"``, ``index=0``,
and the tracker ``purpose="track"``, ``index=step``. Different keys are independent random
streams.

Item pools (gate vs tracker disjointness). Independent streams alone can contain the same item
by chance (an item is a uniformly drawn rule and input; length-3 inputs have a domain of 1000
per rule / family): at the default sizes 6-8% of the tracker's retention items of a step were
gate items of the same decision. So every item uid is assigned to exactly one **pool** by a fixed
hash (:func:`item_pool`: ``TRACK_POOL`` for 1 in :data:`TRACK_POOL_DENOMINATOR` uids, otherwise
``GATE_POOL``; the hash includes ``sampler.root_seed``). Gate items are drawn only from the gate
pool, tracker and start-map items only from the track pool (:func:`draw_slice` with ``pool=``),
so **tracker and gate items never coincide**, by construction. Limits, stated exactly:

* the pool split is by uid hash, independent of content, so each pool is a pseudo-random
  subset of each slice's item domain of the stated share; the within-pool distribution can
  deviate slightly from the unpartitioned one where the domain is small (e.g. the share of
  length-3 inputs, already reduced by the sampler's within-draw distinctness);
* gate items of different decisions / attempts are independent draws from the same gate pool
  and still repeat by chance (about 6% of a default-size retention/copy draw were items of the
  previous decision, as without pools): excluding earlier gate items would exhaust the
  length-3 domain within a few decisions at 2048 items per retention slice and change the item
  distribution, so it is not done (a limit of the wave-1 sampler's domains, not fixed here);
* draws with ``pool=None`` (the default of :func:`draw_slice` / :func:`score_slice`, e.g. for a
  subset search) are unpartitioned and may coincide with gate and tracker items.

AUDIT. The draw-and-score path of this module refuses every AUDIT slice with
:class:`~research_rl.capability.tasks.AuditAccessError`, also when the sampler itself was built
with ``allow_audit=True`` (only the final report may touch AUDIT, and it does not go through here).

Truncation contract (``CAPABILITY.md`` 1.15). A score function that exposes an integer
``max_new_tokens`` (e.g. :class:`~research_rl.capability.tasks.VerifierScoreFn`) scores a
completion that fills the whole budget as 0, so the budget must exceed the longest answer:
construction checks ``max_new_tokens >= operators.max_len + 1`` (every answer of every slice of
:mod:`research_rl.capability.tasks` has at most ``max_len`` characters), and every scored item is
checked exactly, ``len(item.answer) + 1 <= max_new_tokens`` (characters: exact for the character
tokenizer of :mod:`research_rl.models.tiny`, an upper bound on tokens for tokenizers that never
split one digit into several tokens). Score functions without ``max_new_tokens`` are not checked.

Model state. :meth:`CapabilityProbe.evidence`, :func:`score_slice` and :meth:`StartMap.measure`
leave the parameters bit-identical (the incumbent swap is undone also on exceptions) and restore
the training mode of **every** submodule in a ``finally`` block (each module's own
``training`` flag is recorded and written back, so mixed modes such as a frozen dropout in a
training model survive, also when the score function flips modes); they never touch optimizer
state. They do not check that the score function itself does not modify weights (the
:class:`~research_rl.capability.types.ScoreFn` contract forbids it). Submodules added or removed
by the score function are not tracked.

Sampled truncation. :func:`sample_completions_with_truncation` flags a sampled completion as
truncated when its raw generated row contains no EOS (it used the whole ``max_new_tokens``
budget), which also catches budget-filling rows whose sampled special tokens (pad, bos, unk)
are dropped by decoding; :class:`StartMap` ORs this flag with ``score_fn.truncated`` (the
``CAPABILITY.md`` 1.16 rule). The greedy pass@1 path goes through ``score_fn`` itself, whose
re-encoding rule (``VerifierScoreFn.truncated``) can miss such rows; for the greedy path that
is a documented limit of the wave-1 score function, so pass@1 and pass@k can differ on this edge.

Everything here is verified on CPU with tiny models only; GPU behaviour (including the CUDA RNG
handling of :func:`sample_completions`) is NOT_TESTED.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from transformers import GenerationConfig

from research_rl.capability.gate import GateConfig
from research_rl.capability.tasks import AuditAccessError, ItemSampler, _derive_seed
from research_rl.capability.types import GateEvidence, Item, PairedScores, ScoreFn, Split
from research_rl.capability.weights import StateSnapshot, swapped
from research_rl.evaluation.capability_retention import _neutralise_generation_configs
from research_rl.evaluation.metrics import pass_at_k

__all__ = [
    "GATE_POOL",
    "ITEM_POOLS",
    "START_PURPOSE",
    "TRACK_POOL",
    "TRACK_POOL_DENOMINATOR",
    "CapabilityProbe",
    "SliceScores",
    "SliceStart",
    "StartMap",
    "check_truncation_budget",
    "draw_slice",
    "gate_purpose",
    "item_pool",
    "sample_completions",
    "sample_completions_with_truncation",
    "score_items",
    "score_slice",
]

START_PURPOSE = "start"
"""``ItemSampler`` purpose of the :class:`StartMap` items (index 0)."""

GATE_POOL = "gate"
"""Pool of gate-evidence items (:func:`item_pool`)."""
TRACK_POOL = "track"
"""Pool of tracker and start-map items (:func:`item_pool`); disjoint from :data:`GATE_POOL`."""
ITEM_POOLS = (GATE_POOL, TRACK_POOL)
TRACK_POOL_DENOMINATOR = 8
"""1 in this many uids belongs to :data:`TRACK_POOL` (the tracker draws ~8x fewer items per
slice than the gate's retention test, so it gets the minority share; a larger gate pool keeps
gate-vs-gate repetition across decisions at its unpartitioned level)."""

_TRANSFER_SPLITS = (Split.T_RULE, Split.T_TASK)


def gate_purpose(attempt: int) -> str:
    """Sampler purpose of gate evaluation ``attempt`` (0 = the raw candidate, then one per shrink
    factor / group revert of the same decision): ``f"gate/{attempt}"``."""
    _check_index("attempt", attempt)
    return f"gate/{attempt}"


def _check_index(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an int >= 0, got {value!r}")
    return int(value)


def _check_count(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    return int(value)


@contextmanager
def _keep_mode(model: nn.Module) -> Iterator[None]:
    """Restore the ``training`` flag of ``model`` and of every submodule present on entry, each
    to its own value, on exit (normal or exception)."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        yield
    finally:
        for module, was_training in modes:
            module.training = was_training


def item_pool(sampler: ItemSampler, uid: str) -> str:
    """The pool of item ``uid`` for ``sampler``: :data:`TRACK_POOL` iff
    ``_derive_seed("item_pool", sampler.root_seed, uid) % TRACK_POOL_DENOMINATOR == 0``, else
    :data:`GATE_POOL`. A fixed function of ``(root_seed, uid)``; every uid is in exactly one pool."""
    if _derive_seed("item_pool", sampler.root_seed, uid) % TRACK_POOL_DENOMINATOR == 0:
        return TRACK_POOL
    return GATE_POOL


def _check_pool(pool: Any) -> str | None:
    if pool is None:
        return None
    if pool not in ITEM_POOLS:
        raise ValueError(f"pool must be None or one of {ITEM_POOLS}, got {pool!r}")
    return str(pool)


# ============================================================================== drawing and scoring
def _slice_split(sampler: ItemSampler, slice_name: str) -> Split:
    """Split of ``slice_name``; ``KeyError`` if unknown, :class:`AuditAccessError` for AUDIT."""
    slices = sampler.splits.slices()
    if slice_name not in slices:
        raise KeyError(f"unknown slice {slice_name!r}; known: {sorted(slices)}")
    split = slices[slice_name]
    if split is Split.AUDIT:
        raise AuditAccessError(
            f"slice {slice_name!r} belongs to the AUDIT split; probes, trackers and searches never draw AUDIT"
        )
    return split


def draw_slice(
    sampler: ItemSampler,
    slice_name: str,
    n: int,
    *,
    purpose: str,
    index: int,
    pool: str | None = None,
) -> list[Item]:
    """``n`` >= 1 distinct items of a non-AUDIT slice for the key ``(purpose, index)``.

    * ``pool=None``: exactly ``sampler.draw(split, n, purpose=purpose, index=index, slice=slice_name)``.
    * ``pool`` in :data:`ITEM_POOLS`: the first ``n`` items, in stream order, of
      ``sampler.draw(split, N, purpose=purpose, index=index, slice=slice_name)`` whose
      :func:`item_pool` is ``pool``, for the first ``N`` of the schedule
      ``N_0 = ceil(1.25 n / share) + 16, 2 N_0, 4 N_0, ...`` that yields ``n`` of them
      (``share`` = 1/:data:`TRACK_POOL_DENOMINATOR` for the track pool, the rest for the gate pool).

    Deterministic in the key either way. Raises :class:`AuditAccessError` for an AUDIT slice even
    when ``sampler.allow_audit`` is True, ``KeyError`` for an unknown slice and ``ValueError`` for
    ``n < 1``, ``index < 0``, an empty ``purpose``, an unknown pool, or a pool whose part of the
    slice's domain cannot supply ``n`` distinct items.
    """
    split = _slice_split(sampler, slice_name)
    n = _check_count("n", n)
    index = _check_index("index", index)
    pool = _check_pool(pool)
    if pool is None:
        return sampler.draw(split, n, purpose=purpose, index=index, slice=slice_name)
    share = 1.0 / TRACK_POOL_DENOMINATOR if pool == TRACK_POOL else 1.0 - 1.0 / TRACK_POOL_DENOMINATOR
    size = math.ceil(1.25 * n / share) + 16
    while True:
        try:
            stream = sampler.draw(split, size, purpose=purpose, index=index, slice=slice_name)
        except ValueError as err:
            raise ValueError(
                f"slice {slice_name!r}: the {pool!r} pool cannot supply {n} distinct items "
                f"(drawing {size} candidates failed: {err})"
            ) from err
        picked = [it for it in stream if item_pool(sampler, it.uid) == pool]
        if len(picked) >= n:
            return picked[:n]
        size *= 2


def _budget(score_fn: Any) -> int | None:
    value = getattr(score_fn, "max_new_tokens", None)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def check_truncation_budget(sampler: ItemSampler, score_fn: Any) -> None:
    """Raise ``ValueError`` if ``score_fn`` exposes an integer ``max_new_tokens`` smaller than
    ``sampler.splits.operators.max_len + 1`` (the longest possible answer plus the EOS token);
    no-op for score functions without ``max_new_tokens``."""
    budget = _budget(score_fn)
    if budget is None:
        return
    need = sampler.splits.operators.max_len + 1
    if budget < need:
        raise ValueError(
            f"score_fn.max_new_tokens={budget} violates the truncation contract: answers can have "
            f"{need - 1} characters, so max_new_tokens must be >= {need} (a completion that fills the "
            "budget is scored as truncated = 0)"
        )


def score_items(score_fn: ScoreFn, model: nn.Module, items: Sequence[Item]) -> list[float]:
    """``score_fn(model, items)`` with validation: one finite score in [0, 1] per item, every item
    within the truncation budget (``len(answer) + 1 <= score_fn.max_new_tokens`` when the score
    function exposes it), training mode restored afterwards. ``[]`` for no items (no call)."""
    if not items:
        return []
    budget = _budget(score_fn)
    if budget is not None:
        long = [it.uid for it in items if len(it.answer) + 1 > budget]
        if long:
            raise ValueError(
                f"{len(long)} item(s) have answers too long for max_new_tokens={budget} "
                f"(need len(answer) + 1 <= max_new_tokens), e.g. {long[:3]}"
            )
    with _keep_mode(model):
        scores = [float(s) for s in score_fn(model, items)]
    if len(scores) != len(items):
        raise ValueError(f"score_fn returned {len(scores)} scores for {len(items)} items")
    bad = [s for s in scores if not (math.isfinite(s) and 0.0 <= s <= 1.0)]
    if bad:
        raise ValueError(f"score_fn returned scores outside [0, 1]: {bad[:5]}")
    return scores


@dataclass(frozen=True)
class SliceScores:
    """Scores of one model state on the items of one slice drawn for ``(purpose, index)`` (from
    ``pool``, ``None`` = unpartitioned; see :func:`draw_slice`).

    ``items[i]`` was scored ``scores[i]``; ``uids`` are the item uids in the same order.
    """

    slice: str
    split: Split
    purpose: str
    index: int
    items: tuple[Item, ...]
    scores: tuple[float, ...]
    pool: str | None = None

    def __post_init__(self) -> None:
        if len(self.items) != len(self.scores):
            raise ValueError(f"slice {self.slice!r}: {len(self.items)} items but {len(self.scores)} scores")

    @property
    def uids(self) -> tuple[str, ...]:
        return tuple(it.uid for it in self.items)

    @property
    def n(self) -> int:
        return len(self.items)

    @property
    def mean(self) -> float:
        """Mean score (pass rate for binary scores); ``nan`` for no items."""
        return sum(self.scores) / len(self.scores) if self.scores else math.nan

    def to_dict(self) -> dict[str, Any]:
        """JSON-able: slice, split value, purpose, index, pool, n, mean, uids, scores (items by uid
        only)."""
        return {
            "slice": self.slice,
            "split": self.split.value,
            "purpose": self.purpose,
            "index": self.index,
            "pool": self.pool,
            "n": self.n,
            "mean": self.mean,
            "uids": list(self.uids),
            "scores": list(self.scores),
        }


def score_slice(
    sampler: ItemSampler,
    score_fn: ScoreFn,
    model: nn.Module,
    slice_name: str,
    n: int,
    *,
    purpose: str,
    index: int,
    pool: str | None = None,
) -> SliceScores:
    """Draw ``n`` items of ``slice_name`` for ``(purpose, index)`` from ``pool``
    (:func:`draw_slice`; ``None`` = unpartitioned, may coincide with gate and tracker items) and
    score the model's **current** weights on them (:func:`score_items`). Parameters are not
    touched here; every submodule's training mode is restored. AUDIT slices raise
    :class:`AuditAccessError`."""
    split = _slice_split(sampler, slice_name)
    items = draw_slice(sampler, slice_name, n, purpose=purpose, index=index, pool=pool)
    scores = score_items(score_fn, model, items)
    return SliceScores(
        slice=slice_name,
        split=split,
        purpose=purpose,
        index=index,
        items=tuple(items),
        scores=tuple(scores),
        pool=pool,
    )


# ============================================================================== the probe
class CapabilityProbe:
    """Builds :class:`GateEvidence` for one gate evaluation.

    ``transfer_slices`` must be T_RULE / T_TASK slices (pooled by the gate), ``retention_slices``
    RETENTION slices (tested one by one); each name may appear once in total, and AUDIT slices
    raise :class:`AuditAccessError` (``ValueError`` for any other wrong split, ``KeyError`` for
    unknown names). At least one **retention** slice is required (``ValueError``): the gate treats
    evidence without retention slices as retention-OK (vacuously), so a transfer-only probe could
    ACCEPT with no retention test. ``transfer_slices`` may be empty (the gate's ``n=0`` transfer
    test never passes, so such a probe can never ACCEPT). Every transfer slice gets
    ``items_per_transfer_slice`` items per evaluation, every retention slice
    ``items_per_retention_slice``. The truncation budget of ``score_fn`` is checked at
    construction (:func:`check_truncation_budget`).
    """

    def __init__(
        self,
        sampler: ItemSampler,
        score_fn: ScoreFn,
        *,
        transfer_slices: Sequence[str],
        retention_slices: Sequence[str],
        items_per_transfer_slice: int,
        items_per_retention_slice: int,
    ) -> None:
        if not isinstance(sampler, ItemSampler):
            raise TypeError(f"sampler must be an ItemSampler, got {type(sampler).__name__}")
        transfer = tuple(transfer_slices)
        retention = tuple(retention_slices)
        names = [*transfer, *retention]
        if not names:
            raise ValueError("a probe needs at least one transfer or retention slice")
        if len(set(names)) != len(names):
            dup = sorted({s for s in names if names.count(s) > 1})
            raise ValueError(f"slices may appear only once across transfer and retention: {dup}")
        splits: dict[str, Split] = {}
        for name in transfer:
            split = _slice_split(sampler, name)
            if split not in _TRANSFER_SPLITS:
                raise ValueError(f"transfer slice {name!r} has split {split.value!r}; need t_rule or t_task")
            splits[name] = split
        for name in retention:
            split = _slice_split(sampler, name)
            if split is not Split.RETENTION:
                raise ValueError(f"retention slice {name!r} has split {split.value!r}, not 'retention'")
            splits[name] = split
        if not retention:
            raise ValueError(
                "a probe needs at least one retention slice (evidence without retention slices "
                "passes the gate's retention test vacuously)"
            )
        check_truncation_budget(sampler, score_fn)
        self.sampler = sampler
        self.score_fn = score_fn
        self.transfer_slices: tuple[str, ...] = transfer
        self.retention_slices: tuple[str, ...] = retention
        self.items_per_transfer_slice = _check_count("items_per_transfer_slice", items_per_transfer_slice)
        self.items_per_retention_slice = _check_count("items_per_retention_slice", items_per_retention_slice)
        self._splits = splits

    @classmethod
    def from_config(
        cls,
        sampler: ItemSampler,
        score_fn: ScoreFn,
        config: GateConfig,
        *,
        transfer_slices: Sequence[str] | None = None,
        retention_slices: Sequence[str] | None = None,
    ) -> CapabilityProbe:
        """Probe with ``config.items_per_transfer_slice`` / ``config.items_per_retention_slice``.

        Defaults: every T_RULE and T_TASK slice of ``sampler.splits`` (in
        :meth:`~research_rl.capability.tasks.CompositionSplits.slices` order) for transfer and
        every RETENTION slice for retention.
        """
        if not isinstance(config, GateConfig):
            raise TypeError(f"config must be a GateConfig, got {type(config).__name__}")
        splits = sampler.splits
        if transfer_slices is None:
            transfer_slices = (*splits.slices_of(Split.T_RULE), *splits.slices_of(Split.T_TASK))
        if retention_slices is None:
            retention_slices = splits.slices_of(Split.RETENTION)
        return cls(
            sampler,
            score_fn,
            transfer_slices=transfer_slices,
            retention_slices=retention_slices,
            items_per_transfer_slice=config.items_per_transfer_slice,
            items_per_retention_slice=config.items_per_retention_slice,
        )

    # ---------------------------------------------------------------- items
    def items_per(self, slice_name: str) -> int:
        """Items per evaluation of one of this probe's slices (``KeyError`` for other slices)."""
        if slice_name not in self._splits:
            raise KeyError(f"{slice_name!r} is not a slice of this probe")
        if slice_name in self.transfer_slices:
            return self.items_per_transfer_slice
        return self.items_per_retention_slice

    def gate_items(self, decision_index: int, attempt: int = 0) -> dict[str, list[Item]]:
        """The items of evaluation ``(decision_index, attempt)``, per slice (transfer slices first,
        then retention, in construction order): :func:`draw_slice` with
        ``purpose=gate_purpose(attempt)``, ``index=decision_index`` and ``pool=GATE_POOL`` (so they
        never coincide with tracker or start-map items). Deterministic; a pure function of the key."""
        purpose = gate_purpose(attempt)
        index = _check_index("decision_index", decision_index)
        return {
            name: draw_slice(
                self.sampler, name, self.items_per(name), purpose=purpose, index=index, pool=GATE_POOL
            )
            for name in (*self.transfer_slices, *self.retention_slices)
        }

    def score_slice(
        self, model: nn.Module, slice: str, n: int, *, purpose: str, index: int, pool: str | None = None
    ) -> SliceScores:
        """:func:`score_slice` with this probe's sampler and score function. ``slice`` may be any
        non-AUDIT slice of the sampler (TRAIN included), not only this probe's slices."""
        return score_slice(
            self.sampler, self.score_fn, model, slice, n, purpose=purpose, index=index, pool=pool
        )

    # ---------------------------------------------------------------- evidence
    def evidence(
        self,
        model: nn.Module,
        incumbent: StateSnapshot,
        decision_index: int,
        *,
        attempt: int = 0,
        alarm: bool = False,
    ) -> GateEvidence:
        """Paired evidence of the candidate (current weights) against ``incumbent``.

        Items: :meth:`gate_items` for ``(decision_index, attempt)`` - fresh for each decision and
        each attempt of a decision. All items are scored on the current weights in one
        ``score_fn`` call, then once more, in the same order, inside
        ``weights.swapped(model, incumbent)``. Each slice becomes one :class:`PairedScores` with
        its own split (``Split.T_RULE`` / ``Split.T_TASK`` / ``Split.RETENTION``). On return (or on
        an exception) the parameters are bit-identical to their values before the call and every
        submodule's training mode is restored; optimizer state is never touched. ``alarm`` is copied into the
        evidence. ``incumbent`` must name parameters of ``model`` with matching shapes and dtypes
        (:class:`~research_rl.capability.weights.SnapshotError` otherwise, before any write).
        """
        if not isinstance(incumbent, StateSnapshot):
            raise TypeError(f"incumbent must be a StateSnapshot, got {type(incumbent).__name__}")
        per_slice = self.gate_items(decision_index, attempt)
        items = [it for group in per_slice.values() for it in group]
        with _keep_mode(model):
            candidate = score_items(self.score_fn, model, items)
            with swapped(model, incumbent):
                base = score_items(self.score_fn, model, items)
        transfer: list[PairedScores] = []
        retention: list[PairedScores] = []
        pos = 0
        for name, group in per_slice.items():
            end = pos + len(group)
            paired = PairedScores(
                slice=name,
                split=self._splits[name],
                uids=tuple(it.uid for it in group),
                incumbent=tuple(base[pos:end]),
                candidate=tuple(candidate[pos:end]),
            )
            (transfer if name in self.transfer_slices else retention).append(paired)
            pos = end
        return GateEvidence(
            decision_index=int(decision_index),
            transfer=tuple(transfer),
            retention=tuple(retention),
            alarm=bool(alarm),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-able configuration (slices, item counts, sampler identity)."""
        return {
            "transfer_slices": list(self.transfer_slices),
            "retention_slices": list(self.retention_slices),
            "items_per_transfer_slice": self.items_per_transfer_slice,
            "items_per_retention_slice": self.items_per_retention_slice,
            "sampler": self.sampler.state_dict(),
        }


# ============================================================================== sampling and the start map
def _rng_devices(model: nn.Module) -> tuple[torch.device, list[int]]:
    device = next(model.parameters()).device
    if device.type == "cuda":
        return device, [device.index if device.index is not None else torch.cuda.current_device()]
    return device, []


@torch.no_grad()
def sample_completions_with_truncation(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    num_samples: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    seed: int,
    batch_size: int = 16,
) -> tuple[list[str], list[bool]]:
    """:func:`sample_completions` plus one exact truncation flag per sample.

    ``truncated[i]`` is True iff the raw generated row ``i`` contains no EOS token, i.e. the
    sample used the whole ``max_new_tokens`` budget (a tokenizer without ``eos_token_id`` flags
    every sample). The flag is read from the token ids before decoding, so budget-filling rows
    whose special tokens (pad, bos, unk) are removed by ``decode(skip_special_tokens=True)`` are
    flagged too. Same sampling, seeding and restoration semantics as :func:`sample_completions`
    (which returns ``texts`` of this function).
    """
    num_samples = _check_count("num_samples", num_samples)
    max_new_tokens = _check_count("max_new_tokens", max_new_tokens)
    batch_size = _check_count("batch_size", batch_size)
    if isinstance(temperature, bool) or not (math.isfinite(float(temperature)) and temperature > 0.0):
        raise ValueError(f"temperature must be a finite number > 0, got {temperature!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError(f"seed must be an int in [0, 2**64), got {seed!r}")
    ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if not ids:
        raise ValueError("cannot sample from an empty prompt")
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    if pad is None:
        raise ValueError("tokenizer needs a pad_token_id or eos_token_id for generation")
    device, devices = _rng_devices(model)
    config = GenerationConfig(
        do_sample=True,
        temperature=float(temperature),
        top_k=0,
        top_p=1.0,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos,
        pad_token_id=int(pad),
    )
    saved = _neutralise_generation_configs(model)
    texts: list[str] = []
    truncated: list[bool] = []
    try:
        with _keep_mode(model), torch.random.fork_rng(devices=devices):
            model.eval()
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            for start in range(0, num_samples, batch_size):
                rows = min(batch_size, num_samples - start)
                input_ids = torch.tensor([ids] * rows, dtype=torch.long, device=device)
                generated = model.generate(  # type: ignore[operator]
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    generation_config=config,
                )
                for row in generated[:, len(ids) :].tolist():
                    hit_eos = eos is not None and eos in row
                    if hit_eos:
                        row = row[: row.index(eos)]
                    truncated.append(not hit_eos)
                    texts.append(tokenizer.decode(row, skip_special_tokens=True))
    finally:
        for owner, original in saved:
            setattr(owner, "generation_config", original)  # noqa: B010
    return texts, truncated


def sample_completions(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    num_samples: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    seed: int,
    batch_size: int = 16,
) -> list[str]:
    """``num_samples`` sampled continuations of one ``prompt`` (decoded, EOS and after removed).

    Pure ancestral sampling from ``softmax(logits / temperature)``: ``do_sample=True``,
    ``top_k=0``, ``top_p=1.0``, one beam; every ``GenerationConfig`` stored on the model is
    replaced by a default one for the call (as in
    :func:`~research_rl.evaluation.capability_retention.greedy_generate`), so no warper or penalty
    of the model's own config applies. Samples are generated ``batch_size`` at a time from the
    repeated prompt (no padding). Reproducible: the whole call runs inside
    ``torch.random.fork_rng`` with the CPU generator (and, for a CUDA model, that device's
    generator; NOT_TESTED) seeded with ``seed``, so the global RNG state is unchanged afterwards
    and the same ``(model, prompt, seed, batch_size)`` gives the same samples on the same
    hardware and library versions. Every submodule's training mode and the generation configs are
    restored. For exact truncation flags use :func:`sample_completions_with_truncation`.
    """
    texts, _ = sample_completions_with_truncation(
        model,
        tokenizer,
        prompt,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
        batch_size=batch_size,
    )
    return texts


def _require_completion_scorer(score_fn: Any) -> None:
    missing = [
        name
        for name in ("tokenizer", "max_new_tokens", "truncated", "score_completions")
        if not hasattr(score_fn, name)
    ]
    if missing or _budget(score_fn) is None:
        raise TypeError(
            "StartMap needs a completion scorer such as tasks.VerifierScoreFn (attributes tokenizer, "
            f"int max_new_tokens, truncated(), score_completions()); missing: {missing or ['int max_new_tokens']}"
        )


@dataclass(frozen=True)
class SliceStart:
    """Start-checkpoint measurement of one slice.

    ``greedy[i]`` is the greedy (score-function) score of item ``uids[i]``; ``correct_counts[i]``
    of its ``num_samples`` sampled completions were scored 1.0. :attr:`pass1` = mean greedy score,
    :attr:`pass_k` = mean over items of the unbiased ``pass_at_k(num_samples, c, k)``,
    :attr:`sampled_mean` = fraction of correct samples (sampled pass@1).
    """

    slice: str
    split: Split
    uids: tuple[str, ...]
    greedy: tuple[float, ...]
    correct_counts: tuple[int, ...]
    num_samples: int
    k: int

    def __post_init__(self) -> None:
        n = len(self.uids)
        if n < 1 or len(self.greedy) != n or len(self.correct_counts) != n:
            raise ValueError(f"slice {self.slice!r}: need >= 1 item and one greedy score / count per uid")
        if not 1 <= self.k <= self.num_samples:
            raise ValueError(
                f"slice {self.slice!r}: need 1 <= k <= num_samples, got {self.k}, {self.num_samples}"
            )
        if any(not 0 <= c <= self.num_samples for c in self.correct_counts):
            raise ValueError(f"slice {self.slice!r}: correct counts must be in [0, {self.num_samples}]")

    @property
    def n(self) -> int:
        return len(self.uids)

    @property
    def pass1(self) -> float:
        return sum(self.greedy) / self.n

    @property
    def pass_k(self) -> float:
        return sum(pass_at_k(self.num_samples, c, self.k) for c in self.correct_counts) / self.n

    @property
    def sampled_mean(self) -> float:
        return sum(self.correct_counts) / (self.n * self.num_samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice": self.slice,
            "split": self.split.value,
            "n": self.n,
            "k": self.k,
            "num_samples": self.num_samples,
            "pass1": self.pass1,
            "pass_k": self.pass_k,
            "sampled_mean": self.sampled_mean,
            "uids": list(self.uids),
            "greedy": list(self.greedy),
            "correct_counts": list(self.correct_counts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SliceStart:
        """Inverse of :meth:`to_dict` (derived fields are recomputed, not read)."""
        return cls(
            slice=str(data["slice"]),
            split=Split(data["split"]),
            uids=tuple(str(u) for u in data["uids"]),
            greedy=tuple(float(v) for v in data["greedy"]),
            correct_counts=tuple(int(c) for c in data["correct_counts"]),
            num_samples=int(data["num_samples"]),
            k=int(data["k"]),
        )


@dataclass(frozen=True)
class StartMap:
    """pass@1 (greedy) and pass@k (sampled) of the start checkpoint, per slice (JSON-able).

    Build with :meth:`measure`. ``slices`` maps slice name -> :class:`SliceStart`;
    ``splits_fingerprint`` / ``root_seed`` identify the sampler the items came from
    (``purpose="start"``, ``index=0``, :data:`TRACK_POOL`).
    """

    slices: Mapping[str, SliceStart]
    k: int
    num_samples: int
    temperature: float
    seed: int
    items_per_slice: int
    max_new_tokens: int
    splits_fingerprint: str
    root_seed: int

    @classmethod
    def measure(
        cls,
        model: nn.Module,
        sampler: ItemSampler,
        score_fn: Any,
        *,
        items_per_slice: int,
        slices: Sequence[str] | None = None,
        k: int = 16,
        num_samples: int | None = None,
        temperature: float = 1.0,
        seed: int = 0,
        sample_batch_size: int = 16,
    ) -> StartMap:
        """Measure the start map of ``model`` (its current weights).

        * ``slices``: default every non-AUDIT slice of ``sampler.splits``; AUDIT slices raise
          :class:`AuditAccessError`, unknown ones ``KeyError``.
        * Items: ``items_per_slice`` per slice, ``purpose="start"``, ``index=0``, from the
          :data:`TRACK_POOL` (the tracker's population, so ``T_rule(0)`` and ``T_rule(t)`` are
          measured on the same item distribution, never on gate items).
        * pass@1: ``score_fn(model, items)`` (greedy; :func:`score_items`).
        * pass@k: ``num_samples`` (default ``k``; must be >= ``k``) completions per item from
          :func:`sample_completions_with_truncation` at ``temperature`` with the per-item seed
          derived from ``(seed, uid)`` (so an item's samples do not depend on the other items),
          scored with ``score_fn.score_completions``; a sample is passed as truncated when its
          raw row has no EOS (it used the whole budget) **or** ``score_fn.truncated`` flags it
          (the ``CAPABILITY.md`` 1.16 rule; the raw-row test additionally catches rows whose
          special tokens decoding removed). ``score_fn`` must therefore be a completion scorer
          such as :class:`~research_rl.capability.tasks.VerifierScoreFn` (``TypeError``
          otherwise) with binary scores. The greedy pass@1 relies on ``score_fn``'s own
          truncation rule (see the module docstring for the edge where the two differ).

        The global RNG state, parameters and every submodule's training mode are unchanged
        afterwards.
        """
        _require_completion_scorer(score_fn)
        check_truncation_budget(sampler, score_fn)
        k = _check_count("k", k)
        n_samples = k if num_samples is None else _check_count("num_samples", num_samples)
        if n_samples < k:
            raise ValueError(f"num_samples={n_samples} must be >= k={k}")
        items_per_slice = _check_count("items_per_slice", items_per_slice)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an int, got {seed!r}")
        names = (
            tuple(s for s, split in sampler.splits.slices().items() if split is not Split.AUDIT)
            if slices is None
            else tuple(slices)
        )
        if not names or len(set(names)) != len(names):
            raise ValueError(f"slices must be non-empty and distinct, got {names}")
        out: dict[str, SliceStart] = {}
        with _keep_mode(model):
            for name in names:
                greedy = score_slice(
                    sampler,
                    score_fn,
                    model,
                    name,
                    items_per_slice,
                    purpose=START_PURPOSE,
                    index=0,
                    pool=TRACK_POOL,
                )
                counts: list[int] = []
                for item in greedy.items:
                    completions, raw_truncated = sample_completions_with_truncation(
                        model,
                        score_fn.tokenizer,
                        item.prompt,
                        num_samples=n_samples,
                        max_new_tokens=int(score_fn.max_new_tokens),
                        temperature=temperature,
                        seed=_derive_seed("start_map", seed, item.uid),
                        batch_size=sample_batch_size,
                    )
                    flags = [
                        raw or rule
                        for raw, rule in zip(raw_truncated, score_fn.truncated(completions), strict=True)
                    ]
                    scores = score_fn.score_completions(completions, [item] * n_samples, flags)
                    if any(s not in (0.0, 1.0) for s in scores):
                        raise ValueError(f"pass@k needs binary scores; got {sorted(set(scores))[:5]}")
                    counts.append(sum(1 for s in scores if s == 1.0))
                out[name] = SliceStart(
                    slice=name,
                    split=greedy.split,
                    uids=greedy.uids,
                    greedy=greedy.scores,
                    correct_counts=tuple(counts),
                    num_samples=n_samples,
                    k=k,
                )
        return cls(
            slices=out,
            k=k,
            num_samples=n_samples,
            temperature=float(temperature),
            seed=int(seed),
            items_per_slice=items_per_slice,
            max_new_tokens=int(score_fn.max_new_tokens),
            splits_fingerprint=sampler.splits.fingerprint(),
            root_seed=sampler.root_seed,
        )

    def pass1(self, slice_name: str) -> float:
        """Greedy pass rate of ``slice_name`` (``KeyError`` if not measured)."""
        return self.slices[slice_name].pass1

    def pass_k(self, slice_name: str) -> float:
        """Unbiased pass@k of ``slice_name`` (``KeyError`` if not measured)."""
        return self.slices[slice_name].pass_k

    def split_mean_pass1(self, split: Split) -> float:
        """Unweighted mean over the measured slices of ``split`` of their greedy pass rates
        (``ValueError`` if no slice of that split was measured)."""
        split = Split(split)
        values = [s.pass1 for s in self.slices.values() if s.split is split]
        if not values:
            raise ValueError(f"the start map has no {split.value!r} slice")
        return sum(values) / len(values)

    def summary(self) -> dict[str, dict[str, float]]:
        """``{slice: {"pass1": ..., "pass_k": ..., "sampled_mean": ...}}``."""
        return {
            name: {"pass1": s.pass1, "pass_k": s.pass_k, "sampled_mean": s.sampled_mean}
            for name, s in self.slices.items()
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-able (``json.dumps`` works); :meth:`from_dict` inverts it."""
        return {
            "k": self.k,
            "num_samples": self.num_samples,
            "temperature": self.temperature,
            "seed": self.seed,
            "items_per_slice": self.items_per_slice,
            "max_new_tokens": self.max_new_tokens,
            "splits_fingerprint": self.splits_fingerprint,
            "root_seed": self.root_seed,
            "slices": {name: s.to_dict() for name, s in self.slices.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StartMap:
        slices = cast(Mapping[str, Mapping[str, Any]], data["slices"])
        return cls(
            slices={str(name): SliceStart.from_dict(s) for name, s in slices.items()},
            k=int(data["k"]),
            num_samples=int(data["num_samples"]),
            temperature=float(data["temperature"]),
            seed=int(data["seed"]),
            items_per_slice=int(data["items_per_slice"]),
            max_new_tokens=int(data["max_new_tokens"]),
            splits_fingerprint=str(data["splits_fingerprint"]),
            root_seed=int(data["root_seed"]),
        )
