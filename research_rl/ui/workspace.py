"""The model-side state of the UI: model, tasks, probe, version store, stored statistics and scores.

Thread model. Every method that reads or writes weights, generates, or calls the
:class:`~research_rl.capability.edits.VersionStore` (which is single-threaded) must run on the
:class:`~research_rl.ui.jobs.ModelWorker` thread; they check it (:meth:`Workspace._require_worker`)
and raise ``RuntimeError`` otherwise. After every such operation the worker *publishes* a
read-only view (JSON dicts and CPU score tensors that are never mutated, only replaced) under a
lock; the HTTP threads read only that view (:meth:`public_status`, :meth:`public_model`,
:meth:`public_history`, :meth:`rank_channels`, :meth:`public_scores`), never the model.

AUDIT. The sampler is built without ``allow_audit``; statistics are collected only for TRAIN,
T_RULE, T_TASK and RETENTION (:data:`STATS_SPLITS`), and every draw goes through
:func:`~research_rl.capability.probe.draw_slice`, which refuses AUDIT slices.

Item streams (all keyed ``(purpose, index)`` on the workspace's sampler):

* statistics: ``purpose="ui/stats/<split>"``, ``index`` = number of earlier collections of that
  split, items from the **track** pool (never gate items, so channel selection is not done on
  the items an edit's gate verdict is decided on);
* edit evaluation: the gate items of :meth:`VersionStore.evaluate` (gate pool, key
  ``("gate/0", evaluation counter)``); the training reward beside it on ``train/pairs`` items
  ``("ui/reward", evaluation counter)``;
* subset search number ``k`` of this process (``k`` = 0, 1, ... counts the searches of this
  process; reported as ``search_index`` and ``report.config.index``):
  :class:`~research_rl.capability.search.SubsetSearch` with ``index=k`` (track pool), so its
  round and final item sets ``(f"search/{r}/{c}", k)`` / ``(f"search/final/{j}", k)`` are drawn
  independently for every search (not guaranteed disjoint) and a user comparing several
  searches' winners does not select on one fixed item set; the ES fitness of update ``u`` on ``train/pairs`` items
  ``("ui/search/fitness/<k>", u)``. Like the statistics counters, ``k`` is not persisted: after
  a restart of the UI the first search again uses ``index=0`` (the same items as the first
  search of an earlier process on the same sampler).

Every number this module returns was computed by the model loaded in this process on this
machine. With ``--tiny-demo`` that is a randomly initialised tiny model: the numbers exercise the
code paths and say nothing about real large models.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from research_rl.capability.edits import EditRecord, VersionStore
from research_rl.capability.es import ESConfig, ESOperator, FitnessResult
from research_rl.capability.gate import GateConfig
from research_rl.capability.probe import TRACK_POOL, CapabilityProbe, draw_slice
from research_rl.capability.scores import (
    ChannelStats,
    collect_channel_stats,
    contrast_scores,
    group_channels,
    random_scores,
    usage_ratio_scores,
    wanda_scores,
)
from research_rl.capability.search import CONTROL_KINDS, SubsetSearch
from research_rl.capability.spaces import ChannelGainSpace
from research_rl.capability.tasks import (
    AuditAccessError,
    CompositionSplits,
    ExactAnswerVerifier,
    ItemSampler,
    OperatorSet,
    SuffixLeakVerifier,
    VerifierScoreFn,
)
from research_rl.capability.types import ChannelGroup, GateVerdict, Item, ScoreFn, Split
from research_rl.capability.weights import swapped
from research_rl.models.ffn_channels import ChannelSelection
from research_rl.models.model_interface import find_swiglu_ffns, fingerprint
from research_rl.ui.jobs import JobContext, ModelWorker
from research_rl.ui.loading import LoadedModel

__all__ = [
    "HONESTY_NOTE",
    "SCORE_KINDS",
    "STATS_SPLITS",
    "SearchParams",
    "StoredScore",
    "StoredStats",
    "Workspace",
    "WorkspaceConfig",
]

logger = logging.getLogger("research_rl.ui")

STATS_SPLITS: tuple[Split, ...] = (Split.TRAIN, Split.T_RULE, Split.T_TASK, Split.RETENTION)
"""Splits whose prompts the UI may collect statistics on (never AUDIT)."""

SCORE_KINDS: tuple[str, ...] = ("wanda", "contrast", "usage_ratio", "random")
"""Channel scorers the UI ranks by (:mod:`research_rl.capability.scores`)."""

HONESTY_NOTE = (
    "All numbers are computed by the model loaded in this server process on this machine. "
    "Nothing here is a measurement on a GPU or on a real large model unless that model was "
    "loaded here; GPU metrics that were not measured are NOT_TESTED."
)
TINY_NOTE = (
    "Tiny demo: a randomly initialised tiny Llama built in-process. Its numbers exercise the code "
    "paths only and say nothing about real large-model training."
)


# ============================================================================== configuration
@dataclass(frozen=True)
class WorkspaceConfig:
    """Settings of the task testbed, the gate used for edits and the reward shown beside it.

    ``max_new_tokens`` ``None`` means ``operators.max_len + 1`` (the truncation contract of
    ``CAPABILITY.md`` 1.15). ``reward_verifier`` names the training-reward verifier V1 shown beside
    capability: ``"exact"`` (the strict verifier) or ``"suffix_leak"`` (the planted leak).
    """

    seed: int = 0
    num_operators: int = 10
    sampler_seed: int = 1
    max_new_tokens: int | None = None
    batch_size: int = 64
    items_per_transfer_slice: int = 128
    items_per_retention_slice: int = 512
    alpha: float = 0.05
    alpha_schedule: str = "spending"
    retention_margin: float = 0.02
    reward_verifier: str = "exact"
    reward_items: int = 64
    random_score_seed: int = 0

    def gate_config(self) -> GateConfig:
        """The (enabled) :class:`GateConfig` used for every edit evaluation of the store."""
        return GateConfig(
            enabled=True,
            alpha=self.alpha,
            alpha_schedule=self.alpha_schedule,
            retention_margin=self.retention_margin,
            items_per_transfer_slice=self.items_per_transfer_slice,
            items_per_retention_slice=self.items_per_retention_slice,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-able copy of every field."""
        return {
            "seed": self.seed,
            "num_operators": self.num_operators,
            "sampler_seed": self.sampler_seed,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "items_per_transfer_slice": self.items_per_transfer_slice,
            "items_per_retention_slice": self.items_per_retention_slice,
            "alpha": self.alpha,
            "alpha_schedule": self.alpha_schedule,
            "retention_margin": self.retention_margin,
            "reward_verifier": self.reward_verifier,
            "reward_items": self.reward_items,
            "random_score_seed": self.random_score_seed,
        }


@dataclass(frozen=True)
class SearchParams:
    """Bounded parameters of one UI subset search (validated by the API layer)."""

    score: str
    split: str | None
    bands: int
    layers: tuple[int, ...] | None
    groups: tuple[str, ...] | None
    rounds: int
    burst_updates: int
    survivors_fraction: float
    repeats_final: int
    controls: tuple[str, ...]
    items_per_transfer_slice: int
    items_per_retention_slice: int
    population: int
    sigma: float
    lr: float
    max_step_norm: float | None
    fitness_items: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-able copy of every field."""
        return {
            "score": self.score,
            "split": self.split,
            "bands": self.bands,
            "layers": None if self.layers is None else list(self.layers),
            "groups": None if self.groups is None else list(self.groups),
            "rounds": self.rounds,
            "burst_updates": self.burst_updates,
            "survivors_fraction": self.survivors_fraction,
            "repeats_final": self.repeats_final,
            "controls": list(self.controls),
            "items_per_transfer_slice": self.items_per_transfer_slice,
            "items_per_retention_slice": self.items_per_retention_slice,
            "population": self.population,
            "sigma": self.sigma,
            "lr": self.lr,
            "max_step_norm": self.max_step_norm,
            "fitness_items": self.fitness_items,
            "seed": self.seed,
        }


# ============================================================================== stored data
@dataclass(frozen=True)
class StoredStats:
    """Channel statistics collected on the prompts of one split (and, with ``contrast``, on the
    prompt + greedy completion of V2-correct and V2-incorrect items separately)."""

    split: Split
    slices: tuple[str, ...]
    n: int
    index: int
    threshold: float
    stats: ChannelStats
    correct: ChannelStats | None
    incorrect: ChannelStats | None
    n_correct: int | None
    n_incorrect: int | None
    fingerprint: str
    version: str | None
    created: float

    def summary(self, current_fingerprint: str) -> dict[str, Any]:
        """JSON-able description; ``stale`` = collected on other weights than ``current_fingerprint``."""
        return {
            "split": self.split.value,
            "slices": list(self.slices),
            "n": self.n,
            "index": self.index,
            "threshold": self.threshold,
            "contrast": self.correct is not None or self.incorrect is not None or self.n_correct is not None,
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
            "fingerprint": self.fingerprint,
            "version": self.version,
            "stale": self.fingerprint != current_fingerprint,
            "created": self.created,
        }


@dataclass(frozen=True)
class StoredScore:
    """One stored channel score (``scores`` is ``None`` when unavailable, with ``reason``)."""

    name: str
    kind: str
    split: str | None
    scores: Mapping[int, torch.Tensor] | None
    source: str
    fingerprint: str
    version: str | None
    reason: str = ""
    depends_on_weights: bool = True

    def summary(self, current_fingerprint: str) -> dict[str, Any]:
        """JSON-able description (without the tensors); ``stale`` for weight-dependent scores
        whose fingerprint differs from ``current_fingerprint``."""
        return {
            "name": self.name,
            "kind": self.kind,
            "split": self.split,
            "available": self.scores is not None,
            "reason": self.reason,
            "source": self.source,
            "fingerprint": self.fingerprint,
            "version": self.version,
            "stale": self.depends_on_weights and self.fingerprint != current_fingerprint,
        }


def score_name(kind: str, split: str | None) -> str:
    """Key of a stored score: ``"wanda:train"``, ``"contrast:retention"``, ``"usage_ratio"``, ``"random"``."""
    return kind if split is None else f"{kind}:{split}"


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _unembedding(model: nn.Module) -> torch.Tensor | None:
    """Weight ``[vocab, hidden]`` of the model's output embedding (``get_output_embeddings()``)."""
    getter = getattr(model, "get_output_embeddings", None)
    module = getter() if callable(getter) else None
    weight = getattr(module, "weight", None)
    return weight if isinstance(weight, torch.Tensor) and weight.dim() == 2 else None


def _allocate(n: int, k: int) -> list[int]:
    return [n // k + (1 if i < n % k else 0) for i in range(k)]


@dataclass
class _Published:
    status: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    history: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, StoredStats] = field(default_factory=dict)
    scores: dict[str, StoredScore] = field(default_factory=dict)
    fingerprint: str = ""
    last_search: dict[str, Any] | None = None
    last_evaluation: dict[str, Any] | None = None


# ============================================================================== the workspace
class Workspace:
    """Model, testbed, edit store and stored statistics; see the module docstring.

    Construct it on the worker thread (e.g. ``worker.call("open", lambda ctx: Workspace(...))``).
    ``score_fn`` replaces the gate's score function (default: greedy decoding scored by the strict
    verifier); tests inject a cheap weight-sensitive one.
    """

    def __init__(
        self,
        loaded: LoadedModel,
        store_root: str | Path,
        worker: ModelWorker,
        config: WorkspaceConfig | None = None,
        *,
        score_fn: ScoreFn | None = None,
    ) -> None:
        self._worker = worker
        self._require_worker()
        self.config = config or WorkspaceConfig()
        if self.config.reward_verifier not in ("exact", "suffix_leak"):
            raise ValueError(
                f"reward_verifier must be 'exact' or 'suffix_leak', got {self.config.reward_verifier!r}"
            )
        self.loaded = loaded
        self.model: nn.Module = loaded.model
        self.tokenizer = loaded.tokenizer
        self.model.eval()
        self.operators = OperatorSet(seed=self.config.seed, num_operators=self.config.num_operators)
        self.splits = CompositionSplits.build(self.operators, seed=self.config.seed)
        self.sampler = ItemSampler(self.splits, root_seed=self.config.sampler_seed)  # never allow_audit
        budget = self.config.max_new_tokens or self.operators.max_len + 1
        self.v2_fn = VerifierScoreFn(
            self.tokenizer, ExactAnswerVerifier(), max_new_tokens=budget, batch_size=self.config.batch_size
        )
        v1 = ExactAnswerVerifier() if self.config.reward_verifier == "exact" else SuffixLeakVerifier(k=2)
        self.v1_fn = VerifierScoreFn(
            self.tokenizer, v1, max_new_tokens=budget, batch_size=self.config.batch_size
        )
        self.score_fn: ScoreFn = score_fn if score_fn is not None else self.v2_fn
        self.gate_config = self.config.gate_config()
        self.probe = CapabilityProbe.from_config(self.sampler, self.score_fn, self.gate_config)
        self.store = VersionStore(Path(store_root), self.model)
        ledger = self.store.ledger
        if ledger is not None and (
            ledger.total_alpha != self.gate_config.alpha or ledger.schedule != self.gate_config.alpha_schedule
        ):
            raise ValueError(
                f"the store's alpha ledger is pinned to alpha={ledger.total_alpha}, schedule={ledger.schedule!r}; "
                f"start the UI with --alpha {ledger.total_alpha} --alpha-schedule {ledger.schedule} "
                "or use a new --store directory"
            )
        if not self.store.versions():
            self.store.init_from_model(note=f"initial weights: {loaded.source}")
        else:
            head = self.store.head
            if head is None:
                roots = [v.id for v in self.store.versions() if v.parent is None]
                head = roots[0]
            self.store.restore(head)
        self._stats_draws: dict[Split, int] = {}
        self._searches = 0
        self._lock = threading.Lock()
        self._pub = _Published()
        self._stats: dict[str, StoredStats] = {}
        self._scores: dict[str, StoredScore] = {}
        self._last_search: dict[str, Any] | None = None
        self._last_evaluation: dict[str, Any] | None = None
        self._refs = {r.layer_index: r for r in find_swiglu_ffns(self.model)}
        self._model_info = self._describe_model()
        fp = fingerprint(self.model)
        self._scores["random"] = StoredScore(
            name="random",
            kind="random",
            split=None,
            scores=random_scores(self.model, self.config.random_score_seed),
            source=f"uniform random scores, seed {self.config.random_score_seed} (control)",
            fingerprint=fp,
            version=self.store.head,
            depends_on_weights=False,
        )
        self._publish()

    # ---------------------------------------------------------------- threading
    def _require_worker(self) -> None:
        if not self._worker.on_worker():
            raise RuntimeError("model and store operations must run on the model worker thread")

    # ---------------------------------------------------------------- publishing (worker side)
    def _describe_model(self) -> dict[str, Any]:
        params = list(self.model.parameters())
        cfg = getattr(self.model, "config", None)
        w_unembed = _unembedding(self.model)
        return {
            "source": self.loaded.source,
            "tiny_demo": self.loaded.tiny_demo,
            "class": type(self.model).__name__,
            "parameters": int(sum(p.numel() for p in params)),
            "dtype": str(params[0].dtype) if params else None,
            "device": str(params[0].device) if params else None,
            "hidden_size": getattr(cfg, "hidden_size", None),
            "vocab_size": None if w_unembed is None else int(w_unembed.shape[0]),
            "num_layers": len(self._refs),
            "layers": [
                {
                    "layer": r.layer_index,
                    "module_path": r.module_path,
                    "intermediate_size": r.intermediate_size,
                }
                for r in sorted(self._refs.values(), key=lambda r: r.layer_index)
            ],
            "channel_layers": list(self.store.channel_layers),
            "max_new_tokens": self.v2_fn.max_new_tokens,
            "operators": self.operators.describe(),
            "slices": {
                name: split.value for name, split in self.splits.slices().items() if split is not Split.AUDIT
            },
            "transfer_slices": list(self.probe.transfer_slices),
            "retention_slices": list(self.probe.retention_slices),
            "gate_config": self.gate_config.to_dict(),
            "workspace_config": self.config.to_dict(),
            "honesty": HONESTY_NOTE,
            "tiny_note": TINY_NOTE if self.loaded.tiny_demo else None,
        }

    def _publish(self) -> None:
        """Recompute the read-only view (worker only)."""
        self._require_worker()
        fp = fingerprint(self.model)
        staged = self.store.staged
        versions = self.store.versions()
        lineage = self.store.history()
        history = {
            "head": self.store.head,
            "lineage": [v.id for v in lineage],
            "versions": [v.to_dict() | {"passed": v.passed} for v in versions],
            "staged": None
            if staged is None
            else {
                "parent": staged.parent,
                "edits": [e.to_dict() for e in staged.edits],
                "selection": {str(k): list(v) for k, v in staged.selection.to_dict().items()},
                "fingerprint": staged.fingerprint,
                "evaluations": len(staged.verdicts),
                "passed": staged.passed,
                "last_verdict": None if staged.last_verdict is None else staged.last_verdict.to_dict(),
            },
            "evaluation_count": self.store.evaluation_count,
        }
        ledger = self.store.ledger
        status = {
            "source": self.loaded.source,
            "tiny_demo": self.loaded.tiny_demo,
            "fingerprint": fp,
            "head": self.store.head,
            "at_head": staged is None
            and self.store.head is not None
            and any(v.id == self.store.head and v.fingerprint == fp for v in versions),
            "staged": staged is not None,
            "versions": len(versions),
            "evaluation_count": self.store.evaluation_count,
            "ledger": None
            if ledger is None
            else {"total_alpha": ledger.total_alpha, "schedule": ledger.schedule},
            "stats": {k: s.summary(fp) for k, s in sorted(self._stats.items())},
            "scores": {k: s.summary(fp) for k, s in sorted(self._scores.items())},
            "honesty": HONESTY_NOTE,
            "tiny_note": TINY_NOTE if self.loaded.tiny_demo else None,
        }
        with self._lock:
            self._pub = _Published(
                status=status,
                model=self._model_info,
                history=history,
                stats=dict(self._stats),
                scores=dict(self._scores),
                fingerprint=fp,
                last_search=self._last_search,
                last_evaluation=self._last_evaluation,
            )

    # ---------------------------------------------------------------- published reads (any thread)
    def public_status(self) -> dict[str, Any]:
        """Published status (source, fingerprint, head, stats and score summaries, last evaluation)."""
        with self._lock:
            out = dict(self._pub.status)
            out["last_search"] = None if self._pub.last_search is None else {"available": True}
            out["last_evaluation"] = self._pub.last_evaluation
            return out

    def public_model(self) -> dict[str, Any]:
        """Published model description (layers, intermediate sizes, slices, gate config)."""
        with self._lock:
            return dict(self._pub.model)

    def public_history(self) -> dict[str, Any]:
        """Published version graph: head, lineage, every version record and the staged edit."""
        with self._lock:
            return dict(self._pub.history)

    def public_last_search(self) -> dict[str, Any] | None:
        """Result of the last subset search of this process, or ``None``."""
        with self._lock:
            return self._pub.last_search

    def public_scores(self) -> dict[str, dict[str, Any]]:
        """Summaries of every stored score (availability, reason, staleness)."""
        with self._lock:
            fp = self._pub.fingerprint
            return {k: s.summary(fp) for k, s in sorted(self._pub.scores.items())}

    def _published_score(self, kind: str, split: str | None) -> tuple[StoredScore, str]:
        name = score_name(kind, split)
        with self._lock:
            stored = self._pub.scores.get(name)
            fp = self._pub.fingerprint
        if stored is None:
            hint = "POST /api/stats first" if kind != "random" else ""
            raise KeyError(f"no stored score {name!r} {hint}".strip())
        if stored.scores is None:
            raise LookupError(f"score {name!r} is unavailable: {stored.reason}")
        return stored, fp

    def rank_channels(
        self, *, kind: str, split: str | None, layer: int | None, group: str | None, bands: int, top: int
    ) -> dict[str, Any]:
        """Rank channels by a stored score (published data only; no weights are read).

        ``layer`` keeps one layer; ``group`` keeps the channels of one ``L{layer}/b{band}`` group
        of :func:`~research_rl.capability.scores.group_channels` with ``bands`` bands. Ties are
        broken by (layer, channel) ascending. Non-finite scores are returned as ``null``.
        """
        stored, fp = self._published_score(kind, split)
        assert stored.scores is not None
        scores = stored.scores
        if layer is not None and layer not in scores:
            raise KeyError(f"layer {layer} has no {stored.name} scores; layers: {sorted(scores)}")
        groups = group_channels(scores, bands=bands)
        group_of: dict[tuple[int, int], str] = {}
        for g in groups:
            for lay, chans in g.selection.items():
                for c in chans:
                    group_of[(lay, c)] = g.name
        if group is not None and group not in {g.name for g in groups}:
            raise KeyError(f"unknown group {group!r}; groups: {[g.name for g in groups]}")
        rows: list[tuple[float, int, int]] = []
        for lay in sorted(scores):
            if layer is not None and lay != layer:
                continue
            for c, v in enumerate(scores[lay].double().tolist()):
                if group is not None and group_of[(lay, c)] != group:
                    continue
                rows.append((v, lay, c))
        rows.sort(key=lambda r: (-(r[0] if math.isfinite(r[0]) else -math.inf), r[1], r[2]))
        total = len(rows)
        out = [
            {"rank": i + 1, "layer": lay, "channel": c, "score": _finite(v), "group": group_of[(lay, c)]}
            for i, (v, lay, c) in enumerate(rows[:top])
        ]
        return {
            "score": stored.summary(fp),
            "bands": bands,
            "groups": [{"name": g.name, "channels": g.selection.count()} for g in groups],
            "total": total,
            "channels": out,
        }

    # ---------------------------------------------------------------- statistics (worker)
    def collect_stats(
        self,
        split: Split,
        n: int,
        *,
        slice_name: str | None = None,
        contrast: bool = False,
        threshold: float = 0.0,
        ctx: JobContext | None = None,
    ) -> dict[str, Any]:
        """Collect :class:`ChannelStats` on ``n`` prompts of ``split`` (or of one of its slices),
        store them (replacing earlier stats of the split) and recompute the stored scores."""
        self._require_worker()
        split = Split(split)
        if split is Split.AUDIT or split not in STATS_SPLITS:
            raise AuditAccessError("statistics are never collected on the AUDIT split")
        slices = self.splits.slices_of(split)
        if slice_name is not None:
            if slice_name not in slices:
                raise ValueError(
                    f"slice {slice_name!r} is not a slice of split {split.value!r}: {list(slices)}"
                )
            slices = (slice_name,)
        if n < len(slices):
            raise ValueError(f"n must be >= {len(slices)} (one item per slice)")
        index = self._stats_draws.get(split, 0)
        self._stats_draws[split] = index + 1
        items: list[Item] = []
        for name, count in zip(slices, _allocate(n, len(slices))):
            items.extend(
                draw_slice(
                    self.sampler, name, count, purpose=f"ui/stats/{split.value}", index=index, pool=TRACK_POOL
                )
            )
        if ctx is not None:
            ctx.progress(0.1, f"collecting statistics on {len(items)} prompts")
        prompts = [it.prompt for it in items]
        stats = collect_channel_stats(
            self.model, self.tokenizer, prompts, batch_size=self.config.batch_size, threshold=threshold
        )
        correct = incorrect = None
        n_correct = n_incorrect = None
        if contrast:
            if ctx is not None:
                ctx.progress(0.4, "greedy completions for the V2-correct / V2-incorrect contrast")
            completions = self.v2_fn.generate(self.model, items)
            flags = self.v2_fn.truncated(completions)
            ok = self.v2_fn.score_completions(completions, items, flags)
            good = [it.prompt + c for it, c, s in zip(items, completions, ok) if s == 1.0]
            bad = [it.prompt + c for it, c, s in zip(items, completions, ok) if s != 1.0]
            n_correct, n_incorrect = len(good), len(bad)
            if good:
                correct = collect_channel_stats(
                    self.model, self.tokenizer, good, batch_size=self.config.batch_size, threshold=threshold
                )
            if bad:
                incorrect = collect_channel_stats(
                    self.model, self.tokenizer, bad, batch_size=self.config.batch_size, threshold=threshold
                )
        stored = StoredStats(
            split=split,
            slices=tuple(slices),
            n=len(items),
            index=index,
            threshold=float(threshold),
            stats=stats,
            correct=correct,
            incorrect=incorrect,
            n_correct=n_correct,
            n_incorrect=n_incorrect,
            fingerprint=fingerprint(self.model),
            version=self.store.head,
            created=time.time(),
        )
        self._stats[split.value] = stored
        if ctx is not None:
            ctx.progress(0.9, "scoring channels")
        self._recompute_scores()
        self._publish()
        fp = fingerprint(self.model)
        return {
            "stats": stored.summary(fp),
            "scores": {k: s.summary(fp) for k, s in sorted(self._scores.items())},
        }

    def _recompute_scores(self) -> None:
        fp = fingerprint(self.model)
        head = self.store.head
        keep = {"random": self._scores["random"]}
        for key, st in sorted(self._stats.items()):
            keep[score_name("wanda", key)] = StoredScore(
                name=score_name("wanda", key),
                kind="wanda",
                split=key,
                scores=wanda_scores(self.model, st.stats),
                source=f"||W_down[:, c]|| * sqrt(mean h_c^2) on {st.n} {key} prompts (collection {st.index})",
                # current W_down with activations of the weights the stats were collected on:
                # "mixed" (always shown as stale) when those weights differ from the current ones
                fingerprint=fp if st.fingerprint == fp else "mixed",
                version=head,
            )
            if st.n_correct is not None:
                name = score_name("contrast", key)
                if st.correct is not None and st.incorrect is not None:
                    keep[name] = StoredScore(
                        name=name,
                        kind="contrast",
                        split=key,
                        scores=contrast_scores(st.correct, st.incorrect),
                        source=(
                            f"mean |h| on V2-correct minus V2-incorrect completions of {key} prompts "
                            f"({st.n_correct} correct, {st.n_incorrect} incorrect)"
                        ),
                        fingerprint=st.fingerprint,
                        version=st.version,
                    )
                else:
                    keep[name] = StoredScore(
                        name=name,
                        kind="contrast",
                        split=key,
                        scores=None,
                        source="contrast of V2-correct and V2-incorrect completions",
                        fingerprint=st.fingerprint,
                        version=st.version,
                        reason=(
                            f"needs both V2-correct and V2-incorrect completions; got {st.n_correct} correct "
                            f"and {st.n_incorrect} incorrect (a model that solves nothing has no contrast)"
                        ),
                    )
        train, retention = self._stats.get(Split.TRAIN.value), self._stats.get(Split.RETENTION.value)
        if train is not None and retention is not None:
            if train.threshold == retention.threshold:
                keep["usage_ratio"] = StoredScore(
                    name="usage_ratio",
                    kind="usage_ratio",
                    split=None,
                    scores=usage_ratio_scores(train.stats, retention.stats),
                    source=(
                        f"activation rate on train / (rate on retention + 1e-3), threshold {train.threshold}"
                    ),
                    fingerprint=train.fingerprint if train.fingerprint == retention.fingerprint else "mixed",
                    version=train.version,
                )
            else:
                keep["usage_ratio"] = StoredScore(
                    name="usage_ratio",
                    kind="usage_ratio",
                    split=None,
                    scores=None,
                    source="activation-rate ratio train / retention",
                    fingerprint=fp,
                    version=head,
                    reason=(
                        f"train and retention statistics use different thresholds "
                        f"({train.threshold} vs {retention.threshold}); collect both with the same threshold"
                    ),
                )
        else:
            keep["usage_ratio"] = StoredScore(
                name="usage_ratio",
                kind="usage_ratio",
                split=None,
                scores=None,
                source="activation-rate ratio train / retention",
                fingerprint=fp,
                version=head,
                reason="needs statistics of both the train and the retention split",
            )
        self._scores = keep

    # ---------------------------------------------------------------- inspect (worker)
    def channel_detail(
        self, layer: int, channel: int, *, k: int = 10, max_versions: int = 50
    ) -> dict[str, Any]:
        """Vocabulary projection, stored statistics and version history of one FFN channel."""
        self._require_worker()
        if layer not in self._refs:
            raise KeyError(f"no FFN layer {layer}; layers: {sorted(self._refs)}")
        ref = self._refs[layer]
        if not 0 <= channel < ref.intermediate_size:
            raise KeyError(f"layer {layer} has channels 0..{ref.intermediate_size - 1}, not {channel}")
        with torch.no_grad():
            col = ref.down.weight[:, channel].detach().float().cpu()
            w_unembed = _unembedding(self.model)
            if w_unembed is None:
                raise ValueError("the model has no output embedding (unembedding) matrix")
            w_u = w_unembed.detach().float().cpu()
            norm = getattr(getattr(self.model, "model", None), "norm", None)
            gain = getattr(norm, "weight", None)
            used_norm = isinstance(gain, torch.Tensor) and tuple(gain.shape) == tuple(col.shape)
            direction = col * gain.detach().float().cpu() if used_norm and gain is not None else col
            logits = w_u @ direction
        kk = max(1, min(k, int(logits.numel())))
        top = torch.topk(logits, kk)
        bottom = torch.topk(-logits, kk)

        def tokens(ids: Sequence[int], values: Sequence[float]) -> list[dict[str, Any]]:
            rows = []
            for i, v in zip(ids, values):
                raw = self.tokenizer.convert_ids_to_tokens(int(i))
                rows.append(
                    {
                        "id": int(i),
                        "token": raw if isinstance(raw, str) else str(raw),
                        "text": self.tokenizer.decode([int(i)]),
                        "logit": _finite(float(v)),
                    }
                )
            return rows

        projection = {
            "top": tokens(top.indices.tolist(), top.values.tolist()),
            "bottom": tokens(bottom.indices.tolist(), (-bottom.values).tolist()),
            "column_norm": _finite(float(col.norm())),
            "method": (
                "logits = W_unembed @ (final_norm.weight * down_proj.weight[:, c])"
                if used_norm
                else "logits = W_unembed @ down_proj.weight[:, c]"
            ),
        }
        fp = fingerprint(self.model)
        per_split: list[dict[str, Any]] = []
        for key, st in sorted(self._stats.items()):
            if layer not in st.stats.count:
                continue
            values = st.stats.mean_abs[layer].double()
            v = float(values[channel])
            edges = torch.linspace(float(values.min()), float(values.max()) + 1e-12, 21, dtype=torch.float64)
            counts = torch.histc(values, bins=20, min=float(edges[0]), max=float(edges[-1]))
            bin_index = int(
                min(19, max(0, int(torch.bucketize(torch.tensor([v], dtype=torch.float64), edges)) - 1))
            )
            per_split.append(
                {
                    "split": key,
                    "n_prompts": st.n,
                    "tokens": int(st.stats.count[layer]),
                    "threshold": st.threshold,
                    "mean_abs": _finite(v),
                    "rms": _finite(math.sqrt(max(0.0, float(st.stats.mean_sq[layer][channel])))),
                    "active_rate": _finite(float(st.stats.active_rate[layer][channel])),
                    "percentile_in_layer": _finite(float((values < v).double().mean()) * 100.0),
                    "layer_mean_abs_histogram": {
                        "edges": [_finite(float(e)) for e in edges.tolist()],
                        "counts": [int(c) for c in counts.tolist()],
                        "channel_bin": bin_index,
                    },
                    "stale": st.fingerprint != fp,
                }
            )
        channel_scores: dict[str, Any] = {}
        for name, s in sorted(self._scores.items()):
            if s.scores is None or layer not in s.scores:
                continue
            vals = s.scores[layer].double()
            val = float(vals[channel])
            channel_scores[name] = {
                "score": _finite(val),
                "rank_in_layer": int((vals > val).sum()) + 1,
                "stale": s.depends_on_weights and s.fingerprint != fp,
            }
        return {
            "layer": layer,
            "channel": channel,
            "module_path": ref.module_path,
            "intermediate_size": ref.intermediate_size,
            "projection": projection,
            "activation": {
                "note": (
                    "stored statistics keep per-channel means over tokens (mean |h|, RMS, activation rate), "
                    "not per-token values; the histogram is over the channels of this layer (mean |h|), with "
                    "this channel's bin marked"
                ),
                "splits": per_split,
            },
            "scores": channel_scores,
            "versions": self._channel_versions(layer, channel, max_versions),
            "fingerprint": fp,
        }

    def _channel_versions(self, layer: int, channel: int, max_versions: int) -> list[dict[str, Any]]:
        ref = self._refs[layer]
        live_gate = ref.gate.weight[channel].detach().double().cpu()
        live_up = ref.up.weight[channel].detach().double().cpu()
        live_down = ref.down.weight[:, channel].detach().double().cpu()
        rows: list[dict[str, Any]] = []
        prev: torch.Tensor | None = None
        lineage = self.store.history()[-max_versions:]
        if layer not in self.store.channel_layers:
            return rows
        for v in lineage:
            lc = self.store.load_state(v.id)[layer]
            pos = (lc.channels == channel).nonzero()
            if pos.numel() == 0:
                continue
            j = int(pos[0, 0])
            g, u, d = lc.gate[j].double(), lc.up[j].double(), lc.down[:, j].double()
            vec = torch.cat([g, u, d])
            live = torch.cat([live_gate, live_up, live_down])
            rows.append(
                {
                    "version": v.id,
                    "parent": v.parent,
                    "created": v.created,
                    "edits": [e.op.value for e in v.edits],
                    "gate_norm": _finite(float(g.norm())),
                    "up_norm": _finite(float(u.norm())),
                    "down_norm": _finite(float(d.norm())),
                    "change_from_parent": None if prev is None else _finite(float((vec - prev).norm())),
                    "distance_to_live": _finite(float((vec - live).norm())),
                    "head": v.id == self.store.head,
                }
            )
            prev = vec
        rows.append(
            {
                "version": "live",
                "parent": self.store.head,
                "created": None,
                "edits": [] if self.store.staged is None else [e.op.value for e in self.store.staged.edits],
                "gate_norm": _finite(float(live_gate.norm())),
                "up_norm": _finite(float(live_up.norm())),
                "down_norm": _finite(float(live_down.norm())),
                "change_from_parent": None,
                "distance_to_live": 0.0,
                "head": False,
            }
        )
        return rows

    # ---------------------------------------------------------------- edits (worker)
    def stage(self, edits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Stage declarative edits (``EditRecord.from_dict`` of each JSON object; no code)."""
        self._require_worker()
        records = [EditRecord.from_dict(e) for e in edits]
        staged = self.store.stage(records)
        self._last_evaluation = None
        self._publish()
        return {
            "parent": staged.parent,
            "edits": [e.to_dict() for e in staged.edits],
            "selection": {str(k): list(v) for k, v in staged.selection.to_dict().items()},
            "fingerprint": staged.fingerprint,
        }

    def _require_staged(self) -> Any:
        """The active staged edit (``LookupError`` if none)."""
        staged = self.store.staged
        if staged is None:
            raise LookupError("no edit is staged; POST /api/stage first")
        return staged

    def evaluate(self, ctx: JobContext | None = None) -> dict[str, Any]:
        """Gate verdict of the staged edit (store ledger) and the training reward beside it.

        The verdict is recorded (and its alpha spent) by :meth:`VersionStore.evaluate` first; the
        published view is refreshed whatever happens after that, so the UI always shows a verdict
        the store holds. If the training-reward step fails (e.g. out of memory while generating),
        the result still carries the verdict, with ``reward`` and ``divergence`` ``None`` and the
        failure in ``reward_error`` (the traceback goes to the server log).
        """
        self._require_worker()
        staged = self._require_staged()
        index = self.store.evaluation_count
        self._last_evaluation = None
        try:
            if ctx is not None:
                ctx.progress(0.05, "scoring gate items (candidate and parent)")
            verdict = self.store.evaluate(staged, self.gate_config, self.probe)
            result: dict[str, Any] = {
                "evaluation_index": index,
                "verdict": verdict.to_dict(),
                "passed": staged.passed,
                "commit_allowed_without_override": staged.passed,
                "reward": None,
                "reward_error": None,
                "divergence": None,
                "gate_config": self.gate_config.to_dict(),
                "items": {
                    "transfer_slices": list(self.probe.transfer_slices),
                    "retention_slices": list(self.probe.retention_slices),
                    "items_per_transfer_slice": self.probe.items_per_transfer_slice,
                    "items_per_retention_slice": self.probe.items_per_retention_slice,
                },
                "honesty": HONESTY_NOTE,
            }
            self._last_evaluation = result
            if ctx is not None:
                ctx.progress(0.8, "training reward on train items")
            try:
                reward = self._reward(staged, index)
            except Exception as exc:
                logger.exception("training reward of evaluation %d failed; the gate verdict stands", index)
                result["reward_error"] = (
                    f"the training reward could not be computed ({type(exc).__name__}; see the server "
                    "log); the gate verdict above is recorded and its alpha is spent"
                )
            else:
                result["reward"] = reward
                result["divergence"] = self._divergence(verdict, reward)
            return result
        finally:
            self._publish()

    def _reward(self, staged: Any, index: int) -> dict[str, Any]:
        n = self.config.reward_items
        items = draw_slice(self.sampler, "train/pairs", n, purpose="ui/reward", index=index)

        def score(model: nn.Module) -> tuple[float, float]:
            completions = self.v1_fn.generate(model, items)
            flags = self.v1_fn.truncated(completions)
            v1 = self.v1_fn.score_completions(completions, items, flags)
            v2 = self.v2_fn.score_completions(completions, items, flags)
            return sum(v1) / len(v1), sum(v2) / len(v2)

        cand_v1, cand_v2 = score(self.model)
        with swapped(self.model, staged.incumbent):
            inc_v1, inc_v2 = score(self.model)
        return {
            "verifier_v1": self.config.reward_verifier,
            "items": n,
            "slice": "train/pairs",
            "candidate_v1": cand_v1,
            "incumbent_v1": inc_v1,
            "change_v1": cand_v1 - inc_v1,
            "candidate_v2": cand_v2,
            "incumbent_v2": inc_v2,
            "change_v2": cand_v2 - inc_v2,
        }

    @staticmethod
    def _divergence(verdict: GateVerdict, reward: Mapping[str, Any]) -> dict[str, Any]:
        gain = verdict.transfer.point
        change = float(reward["change_v1"])
        if change > 0 and not verdict.transfer_ok:
            flag = "reward rose while held-out transfer was not confirmed"
        elif change <= 0 and verdict.transfer_ok:
            flag = "transfer confirmed although training reward did not rise"
        else:
            flag = "reward and capability agree in direction"
        return {
            "reward_change_v1": change,
            "transfer_gain_point": _finite(gain),
            "transfer_lower_bound": _finite(verdict.transfer.bound),
            "d": _finite(change - gain),
            "flag": flag,
            "definition": "d = (reward change on train items) - (point gain on held-out transfer items)",
        }

    def commit(self, override: str | None, note: str = "") -> dict[str, Any]:
        """Commit the staged edit (refused without a passing verdict unless ``override``)."""
        self._require_worker()
        staged = self._require_staged()
        version = self.store.commit(staged, override=override, note=note)
        self._last_evaluation = None
        self._publish()
        return {"version": version.to_dict() | {"passed": version.passed}}

    def discard(self) -> dict[str, Any]:
        """Discard the staged edit (the parent's weights are restored bitwise)."""
        self._require_worker()
        staged = self._require_staged()
        self.store.discard(staged)
        self._last_evaluation = None
        self._publish()
        return {"discarded": True, "head": self.store.head}

    def restore(self, version_id: str) -> dict[str, Any]:
        """Put the live model at ``version_id`` bitwise (:meth:`VersionStore.restore`)."""
        self._require_worker()
        version = self.store.restore(version_id)
        self._publish()
        return {
            "version": version.to_dict() | {"passed": version.passed},
            "fingerprint": fingerprint(self.model),
        }

    # ---------------------------------------------------------------- search (worker)
    def search_groups(self, params: SearchParams) -> list[ChannelGroup]:
        """Candidate groups of a search: ``group_channels`` of the stored score, optionally filtered."""
        stored, _ = self._published_score(params.score, params.split)
        assert stored.scores is not None
        groups = group_channels(stored.scores, bands=params.bands, layers=params.layers)
        if params.groups is not None:
            by_name = {g.name: g for g in groups}
            unknown = [g for g in params.groups if g not in by_name]
            if unknown:
                raise KeyError(f"unknown groups {unknown}; groups: {sorted(by_name)}")
            groups = [by_name[g] for g in params.groups]
        return groups

    def run_search(
        self, params: SearchParams, *, max_candidates: int, ctx: JobContext | None = None
    ) -> dict[str, Any]:
        """Run :class:`SubsetSearch` over the groups of a stored score with ES bursts on a
        :class:`ChannelGainSpace`; the model is back at its start weights afterwards (checked)."""
        self._require_worker()
        groups = self.search_groups(params)
        if not groups:
            raise ValueError("the chosen score and filters give no channel group")
        if len(groups) > max_candidates:
            raise ValueError(
                f"{len(groups)} candidate groups exceed the limit of {max_candidates}; "
                "restrict layers / groups or use fewer bands"
            )
        unknown = [c for c in params.controls if c not in CONTROL_KINDS]
        if unknown:
            raise ValueError(f"unknown controls {unknown}")
        k = self._searches
        self._searches += 1
        cache: dict[int, list[Item]] = {}

        def fitness(model: nn.Module, update_index: int, member_index: int) -> FitnessResult:
            items = cache.get(update_index)
            if items is None:
                items = draw_slice(
                    self.sampler,
                    "train/pairs",
                    params.fitness_items,
                    purpose=f"ui/search/fitness/{k}",
                    index=update_index,
                )
                cache[update_index] = items
            completions = self.v1_fn.generate(model, items)
            flags = self.v1_fn.truncated(completions)
            v1 = self.v1_fn.score_completions(completions, items, flags)
            v2 = self.v2_fn.score_completions(completions, items, flags)
            lengths = tuple(
                len(self.tokenizer(c, add_special_tokens=False)["input_ids"]) for c in completions
            )
            return FitnessResult(
                reward_v1=sum(v1) / len(v1),
                reward_v2=sum(v2) / len(v2),
                v1_pass=tuple(x == 1.0 for x in v1),
                v2_pass=tuple(x == 1.0 for x in v2),
                lengths=lengths,
            )

        es_config = ESConfig(
            population=params.population,
            sigma=params.sigma,
            lr=params.lr,
            max_step_norm=params.max_step_norm,
            seed=params.seed,
        )
        built = [0]
        total = max(1, self._expected_bursts(params, len(groups)))

        def make_operator(selection: ChannelSelection | None) -> ESOperator:
            built[0] += 1
            if ctx is not None:
                ctx.progress(min(0.99, (built[0] - 1) / total), f"burst {built[0]} of about {total}")
            sel = ChannelSelection.all_channels(self.model) if selection is None else selection
            return ESOperator(ChannelGainSpace(self.model, sel), fitness, es_config)

        search = SubsetSearch(
            groups,
            make_operator,
            self.probe,
            rounds=params.rounds,
            burst_updates=params.burst_updates,
            survivors_fraction=params.survivors_fraction,
            repeats_final=params.repeats_final,
            controls=params.controls,
            seed=params.seed,
            items_per_transfer_slice=params.items_per_transfer_slice,
            items_per_retention_slice=params.items_per_retention_slice,
            index=k,
        )
        before = fingerprint(self.model)
        report = search.run(self.model)
        after = fingerprint(self.model)
        if after != before:
            raise RuntimeError("the search did not restore the start weights")
        result = {
            "params": params.to_dict(),
            "search_index": k,
            "report": report.to_dict(),
            "fingerprint": after,
            "honesty": HONESTY_NOTE,
        }
        self._last_search = result
        self._publish()
        return result

    def _expected_bursts(self, params: SearchParams, n: int) -> int:
        total = 0
        survivors = n
        for r in range(params.rounds):
            if survivors <= 1 and r > 0:
                break
            if survivors <= 1:
                break
            total += survivors
            survivors = max(1, math.ceil(params.survivors_fraction * survivors - 1e-12))
        return total + survivors + len(params.controls)


def call_on_worker(worker: ModelWorker, fn: Callable[[], Any], *, timeout: float = 600.0) -> Any:
    """Run ``fn`` on the worker and return its result (re-raising its exception); for startup."""
    job = worker.call("startup", lambda _ctx: fn(), wait=timeout)
    if not job.done_event.is_set():
        raise TimeoutError("the model worker did not finish the startup job in time")
    if job.exception is not None:
        raise job.exception
    return job.result
