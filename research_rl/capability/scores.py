"""Forward-only channel scoring and grouping (``docs/research_rl/CAPABILITY.md`` section 1.6).

Everything here runs forward passes only (no backward, no gradient graph):

* :class:`ChannelStats` / :func:`collect_channel_stats` - per layer and intermediate channel of
  every SwiGLU FFN: token count, mean ``|h|``, mean ``h^2`` and activation rate ``P(|h| > t)`` of
  the ``down_proj`` input ``h = act(gate_proj(x)) * up_proj(x)``, over the real (non-padding)
  tokens of a list of texts.
* Scorers returning ``dict[layer, 1-D float64 CPU tensor of length intermediate_size]``, directly
  usable with :meth:`~research_rl.models.ffn_channels.ChannelSelection.top_k_from_scores`:
  :func:`wanda_scores`, :func:`contrast_scores`, :func:`usage_ratio_scores`, :func:`random_scores`
  and the re-weighting :func:`layer_prior`.
* :func:`group_channels` - partition of every channel into ``layer x score-quantile band`` groups
  (:class:`~research_rl.capability.types.ChannelGroup`), the unit of subset search and of group
  rollback.
* :func:`density_matched_random` - a random selection with the same per-layer channel counts as a
  given one (the control arm of the subset experiments).

Every random draw comes from a local CPU ``torch.Generator`` seeded by the caller; the global torch
RNG is never touched. Verified on CPU with tiny Llama models only
(``tests/research_rl/test_capability_scores.py``); nothing has run on a GPU (NOT_TESTED).
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from research_rl.capability.types import ChannelGroup
from research_rl.models.ffn_channels import ChannelSelection
from research_rl.models.model_interface import find_swiglu_ffns

_STAT_FIELDS = ("mean_abs", "mean_sq", "active_rate")


# ============================================================================== statistics
@dataclass(eq=False)
class ChannelStats:
    """Per-layer, per-channel statistics of the SwiGLU ``down_proj`` input over real tokens.

    For every layer ``L`` (keys of the dicts, identical across the four dicts):

    * ``count[L]`` - 0-d ``int64`` tensor: number of real (attention-mask = 1) tokens seen.
    * ``mean_abs[L]`` - ``float64 [intermediate_size]``: mean of ``|h_c|`` over those tokens.
    * ``mean_sq[L]`` - ``float64 [intermediate_size]``: mean of ``h_c^2``.
    * ``active_rate[L]`` - ``float64 [intermediate_size]``: fraction of tokens with
      ``|h_c| > threshold`` (strict inequality).

    ``num_texts`` is the number of texts the statistics were collected on and ``threshold`` the
    activation threshold used for ``active_rate`` (a field beyond the spec, with a default, so
    that rates collected at different thresholds are never compared or merged). A layer with
    ``count == 0`` carries all-zero means by convention; the scorers reject it. All tensors live
    on the CPU and carry no gradient graph. Validation runs in ``__post_init__``
    (``ValueError``).

    Equality (``==``) is exact value equality: same ``num_texts``, ``threshold`` and layers, and
    every tensor equal in dtype, device, shape and value (``torch.equal``; NaN != NaN). The dataclass
    default ``__eq__`` is disabled because it would compare dicts of multi-element tensors and
    raise. Instances are mutable and therefore unhashable.
    """

    count: dict[int, torch.Tensor]
    mean_abs: dict[int, torch.Tensor]
    mean_sq: dict[int, torch.Tensor]
    active_rate: dict[int, torch.Tensor]
    num_texts: int
    threshold: float = 0.0

    def __post_init__(self) -> None:
        layers = set(self.count)
        for name in _STAT_FIELDS:
            if set(getattr(self, name)) != layers:
                raise ValueError(
                    f"ChannelStats.{name} has layers {sorted(getattr(self, name))}, count has {sorted(layers)}"
                )
        for layer in layers:
            c = self.count[layer]
            if c.dim() != 0 or c.dtype.is_floating_point or int(c) < 0:
                raise ValueError(f"layer {layer}: count must be a non-negative 0-d integer tensor")
            sizes = {getattr(self, name)[layer].shape for name in _STAT_FIELDS}
            if len(sizes) != 1 or len(next(iter(sizes))) != 1:
                raise ValueError(
                    f"layer {layer}: statistics must be 1-D tensors of equal length, got {sizes}"
                )
        if self.num_texts < 0:
            raise ValueError("num_texts must be >= 0")
        if not (math.isfinite(self.threshold) and self.threshold >= 0.0):
            raise ValueError(f"threshold must be finite and >= 0, got {self.threshold!r}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChannelStats):
            return NotImplemented
        if (
            self.num_texts != other.num_texts
            or self.threshold != other.threshold
            or self.layers() != other.layers()
        ):
            return False
        for name in ("count", *_STAT_FIELDS):
            mine, theirs = getattr(self, name), getattr(other, name)
            for layer in self.layers():
                a, b = mine[layer], theirs[layer]
                if a.dtype != b.dtype or a.device != b.device or not torch.equal(a, b):
                    return False
        return True

    def layers(self) -> list[int]:
        """Layer indices, ascending."""
        return sorted(self.count)

    def intermediate_size(self, layer: int) -> int:
        return int(self.mean_abs[layer].numel())

    def to_dict(self) -> dict[str, Any]:
        """JSON-able dict (string layer keys, Python ints / float lists); see :meth:`from_dict`."""
        return {
            "num_texts": self.num_texts,
            "threshold": self.threshold,
            "count": {str(k): int(self.count[k]) for k in self.layers()},
            **{
                name: {str(k): getattr(self, name)[k].double().tolist() for k in self.layers()}
                for name in _STAT_FIELDS
            },
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ChannelStats:
        """Inverse of :meth:`to_dict` (float64 values round-trip exactly through JSON)."""

        def vec(name: str) -> dict[int, torch.Tensor]:
            return {int(k): torch.tensor(v, dtype=torch.float64) for k, v in d[name].items()}

        return cls(
            count={int(k): torch.tensor(int(v), dtype=torch.int64) for k, v in d["count"].items()},
            mean_abs=vec("mean_abs"),
            mean_sq=vec("mean_sq"),
            active_rate=vec("active_rate"),
            num_texts=int(d["num_texts"]),
            threshold=float(d.get("threshold", 0.0)),
        )

    def merge(self, other: ChannelStats) -> ChannelStats:
        """Statistics of the union of both text sets: means weighted by token counts (float64).

        Up to float rounding this equals collecting on the concatenated texts. Both must cover the
        same layers with the same sizes and use the same threshold (``ValueError`` otherwise).
        """
        if self.layers() != other.layers():
            raise ValueError(f"cannot merge stats over layers {self.layers()} and {other.layers()}")
        if self.threshold != other.threshold:
            raise ValueError(f"cannot merge stats with thresholds {self.threshold} and {other.threshold}")
        out: dict[str, dict[int, torch.Tensor]] = {name: {} for name in _STAT_FIELDS}
        count: dict[int, torch.Tensor] = {}
        for layer in self.layers():
            if self.intermediate_size(layer) != other.intermediate_size(layer):
                raise ValueError(f"layer {layer}: intermediate sizes differ")
            na, nb = int(self.count[layer]), int(other.count[layer])
            n = na + nb
            count[layer] = torch.tensor(n, dtype=torch.int64)
            for name in _STAT_FIELDS:
                a, b = getattr(self, name)[layer].double(), getattr(other, name)[layer].double()
                out[name][layer] = (a * na + b * nb) / n if n else torch.zeros_like(a)
        return ChannelStats(
            count=count,
            mean_abs=out["mean_abs"],
            mean_sq=out["mean_sq"],
            active_rate=out["active_rate"],
            num_texts=self.num_texts + other.num_texts,
            threshold=self.threshold,
        )


def _restore_modes(modes: Sequence[tuple[nn.Module, bool]]) -> None:
    """Call ``module.train(training)`` for every recorded module, in the recorded order.

    ``modes`` comes from ``model.modules()`` (parents before children), so a parent's recursive
    ``train()`` is overridden by its children's own later calls: every module ends in exactly its
    recorded mode, and any ``train()`` override a module defines is invoked.
    """
    for module, training in modes:
        module.train(training)


def _accepts_position_ids(model: nn.Module) -> bool:
    try:
        return "position_ids" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def collect_channel_stats(
    model: nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    layers: Iterable[int] | None = None,
    batch_size: int = 8,
    threshold: float = 0.0,
) -> ChannelStats:
    """Run ``texts`` through ``model`` and collect :class:`ChannelStats` of every ``down_proj`` input.

    * One forward pre-hook per selected SwiGLU FFN (``layers``; default: every decoder layer; the
      active expert of an expert container, as :func:`find_swiglu_ffns` resolves it) reads the
      ``down_proj`` input ``h = act(gate_proj(x)) * up_proj(x)`` of shape ``[batch, seq, I]``.
    * Texts are encoded like :func:`~research_rl.evaluation.capability_retention.greedy_generate`
      (``tokenizer(text, add_special_tokens=False)``, no BOS/EOS added), in input order, in batches
      of ``batch_size``, **left-padded here** with the tokenizer's ``pad_token_id`` (or EOS) and an
      attention mask; the tokenizer's ``padding_side`` is not touched. When the model's ``forward``
      takes ``position_ids`` they are set from the mask (``cumsum - 1``), so every text gets the
      same positions as when run alone. Only positions with mask 1 enter the statistics, so a
      padded batch gives the statistics of running each text alone up to float rounding.
    * The forward is the plain model call with ``use_cache=False`` under ``torch.no_grad()``, with
      the model in eval mode (dropout off). In a ``finally`` block (also when the forward raises)
      every hook is removed and the ``training`` flag of *every* submodule is restored to its own
      previous value, so a submodule the caller kept in eval mode inside a training-mode model
      stays in eval mode. No gradient graph is built and no parameter is changed.
    * Sums are accumulated in float64 on the activation's device and returned on the CPU.

    Raises ``ValueError`` for an empty ``texts``, a text that encodes to no token,
    ``batch_size < 1``, a negative / non-finite ``threshold`` or a tokenizer with neither a pad
    nor an EOS id; ``ModelInterfaceError`` (a ``ValueError``) for unknown ``layers``;
    ``RuntimeError`` if a selected ``down_proj`` is never called by the forward pass (a model
    whose forward bypasses the FFN). Texts longer than the model's position range are not
    truncated (the model's own error surfaces). Not thread-safe with concurrent use of the same
    model.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not (math.isfinite(threshold) and threshold >= 0.0):
        raise ValueError(f"threshold must be finite and >= 0, got {threshold!r}")
    if len(texts) == 0:
        raise ValueError("texts must not be empty")
    encoded = [list(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
    for i, ids in enumerate(encoded):
        if not ids:
            raise ValueError(f"text {i} encodes to no token")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("tokenizer needs a pad_token_id or eos_token_id for padding")
    refs = find_swiglu_ffns(model, layers)
    device = next(model.parameters()).device
    pass_positions = _accepts_position_ids(model)

    sums: dict[int, dict[str, torch.Tensor]] = {}
    counts: dict[int, int] = {r.layer_index: 0 for r in refs}
    current_mask: list[torch.Tensor] = []  # the real-token mask of the running batch

    def make_hook(layer: int) -> Any:
        def hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
            h = args[0].detach()
            mask = current_mask[0].to(h.device)
            real = h[mask].to(torch.float64)  # [n_real, I]; boolean indexing never touches pad rows
            a = real.abs()
            acc = sums.get(layer)
            if acc is None:
                zeros = torch.zeros(real.shape[-1], dtype=torch.float64, device=real.device)
                acc = sums[layer] = {name: zeros.clone() for name in _STAT_FIELDS}
            acc["mean_abs"] += a.sum(dim=0)
            acc["mean_sq"] += (real * real).sum(dim=0)
            acc["active_rate"] += (a > threshold).to(torch.float64).sum(dim=0)
            counts[layer] += int(real.shape[0])

        return hook

    modes = [(module, module.training) for module in model.modules()]
    handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        for ref in refs:
            handles.append(ref.down.register_forward_pre_hook(make_hook(ref.layer_index)))
        model.eval()
        with torch.no_grad():
            for start in range(0, len(encoded), batch_size):
                batch = encoded[start : start + batch_size]
                width = max(len(ids) for ids in batch)
                input_ids = torch.full((len(batch), width), int(pad), dtype=torch.long)
                attention = torch.zeros(len(batch), width, dtype=torch.long)
                for row, ids in enumerate(batch):
                    input_ids[row, width - len(ids) :] = torch.tensor(ids, dtype=torch.long)
                    attention[row, width - len(ids) :] = 1
                current_mask[:] = [attention.bool()]
                kwargs: dict[str, Any] = {
                    "input_ids": input_ids.to(device),
                    "attention_mask": attention.to(device),
                    "use_cache": False,
                }
                if pass_positions:
                    kwargs["position_ids"] = (attention.cumsum(-1) - 1).clamp_min(0).to(device)
                model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
        _restore_modes(modes)

    out: dict[str, dict[int, torch.Tensor]] = {name: {} for name in _STAT_FIELDS}
    count: dict[int, torch.Tensor] = {}
    for ref in refs:
        layer, n = ref.layer_index, counts[ref.layer_index]
        if layer not in sums:
            raise RuntimeError(f"layer {layer}: down_proj was never called during the forward pass")
        count[layer] = torch.tensor(n, dtype=torch.int64)
        for name in _STAT_FIELDS:
            out[name][layer] = (sums[layer][name] / n).cpu()
    return ChannelStats(
        count=count,
        mean_abs=out["mean_abs"],
        mean_sq=out["mean_sq"],
        active_rate=out["active_rate"],
        num_texts=len(texts),
        threshold=float(threshold),
    )


# ============================================================================== scorers
def _require_counts(stats: ChannelStats, what: str) -> None:
    empty = [layer for layer in stats.layers() if int(stats.count[layer]) == 0]
    if empty:
        raise ValueError(f"{what}: layers {empty} have no tokens")


def _require_same_layers(a: ChannelStats, b: ChannelStats, what: str) -> None:
    if a.layers() != b.layers():
        raise ValueError(f"{what}: stats cover layers {a.layers()} and {b.layers()}")
    for layer in a.layers():
        if a.intermediate_size(layer) != b.intermediate_size(layer):
            raise ValueError(f"{what}: layer {layer} intermediate sizes differ")


@torch.no_grad()
def wanda_scores(model: nn.Module, stats: ChannelStats) -> dict[int, torch.Tensor]:
    """Wanda-style channel importance ``||W_down[:, c]||_2 * sqrt(mean h_c^2)`` per layer of ``stats``.

    The column norm is computed in float64 from the model's current ``down_proj.weight`` (of the
    FFN :func:`find_swiglu_ffns` resolves for that layer). Returns float64 CPU tensors, all
    ``>= 0``. ``ValueError`` if a layer has no tokens or its size differs from the model's.
    """
    _require_counts(stats, "wanda_scores")
    out: dict[int, torch.Tensor] = {}
    for ref in find_swiglu_ffns(model, stats.layers()):
        layer = ref.layer_index
        if ref.intermediate_size != stats.intermediate_size(layer):
            raise ValueError(
                f"layer {layer}: model intermediate size {ref.intermediate_size} != "
                f"stats size {stats.intermediate_size(layer)}"
            )
        norm = torch.linalg.vector_norm(ref.down.weight.detach().to(torch.float64), dim=0).cpu()
        out[layer] = norm * stats.mean_sq[layer].double().clamp_min(0.0).sqrt()
    return out


def contrast_scores(stats_correct: ChannelStats, stats_incorrect: ChannelStats) -> dict[int, torch.Tensor]:
    """``mean|h|`` on V2-correct rollouts minus ``mean|h|`` on V2-incorrect rollouts, per channel.

    Signed: positive = the channel is more active when the strict verifier passes. The caller is
    responsible for both stats coming from rollouts of the same prompts (that is what makes the
    difference a contrast and not a prompt effect). ``ValueError`` if the layer sets or sizes
    differ or a layer has no tokens.
    """
    _require_same_layers(stats_correct, stats_incorrect, "contrast_scores")
    _require_counts(stats_correct, "contrast_scores (correct)")
    _require_counts(stats_incorrect, "contrast_scores (incorrect)")
    return {
        layer: (stats_correct.mean_abs[layer].double() - stats_incorrect.mean_abs[layer].double()).cpu()
        for layer in stats_correct.layers()
    }


def usage_ratio_scores(
    stats_train: ChannelStats, stats_retention: ChannelStats, eps: float = 1e-3
) -> dict[int, torch.Tensor]:
    """Activation-rate ratio ``active_rate_train / (active_rate_retention + eps)`` per channel.

    High = the channel fires on the task and rarely on the retention texts. ``eps > 0`` keeps the
    score finite (``<= 1 / eps``); a channel inactive on both gets 0. Both stats must share
    layers, sizes and ``threshold`` (``ValueError`` otherwise; with ``threshold=0`` almost every
    SwiGLU channel is "active" on every token, so pick a positive threshold for a useful ratio).
    """
    if not (math.isfinite(eps) and eps > 0.0):
        raise ValueError(f"eps must be finite and > 0, got {eps!r}")
    _require_same_layers(stats_train, stats_retention, "usage_ratio_scores")
    if stats_train.threshold != stats_retention.threshold:
        raise ValueError(
            f"usage_ratio_scores: thresholds differ ({stats_train.threshold} vs {stats_retention.threshold})"
        )
    _require_counts(stats_train, "usage_ratio_scores (train)")
    _require_counts(stats_retention, "usage_ratio_scores (retention)")
    return {
        layer: (
            stats_train.active_rate[layer].double() / (stats_retention.active_rate[layer].double() + eps)
        ).cpu()
        for layer in stats_train.layers()
    }


def random_scores(
    model: nn.Module, seed: int, *, layers: Iterable[int] | None = None
) -> dict[int, torch.Tensor]:
    """Uniform ``[0, 1)`` float64 scores per channel from ``torch.Generator().manual_seed(seed)``.

    Draws are made for *every* FFN layer of the model in ascending order and then filtered to
    ``layers`` (default: all), so a layer's scores do not depend on which other layers are asked
    for. With :meth:`ChannelSelection.top_k_from_scores` and the same ``k`` as an informed
    selection this gives a random control of the same total size; for the same *per-layer* counts
    use :func:`density_matched_random`. The global torch RNG is not touched.
    """
    refs = find_swiglu_ffns(model)
    wanted = None if layers is None else set(layers)
    if wanted is not None:
        missing = sorted(wanted - {r.layer_index for r in refs})
        if missing:
            raise ValueError(f"random_scores: layers {missing} are not FFN layers of the model")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    out: dict[int, torch.Tensor] = {}
    for ref in refs:
        values = torch.rand(ref.intermediate_size, generator=gen, dtype=torch.float64)
        if wanted is None or ref.layer_index in wanted:
            out[ref.layer_index] = values
    return out


def layer_prior(scores: Mapping[int, torch.Tensor], weights: Mapping[int, float]) -> dict[int, torch.Tensor]:
    """``scores[layer] * weights[layer]`` (float64); layers absent from ``weights`` keep weight 1.0.

    Weights must be finite and ``>= 0`` and name only layers present in ``scores``
    (``ValueError`` otherwise). For signed scores (e.g. :func:`contrast_scores`) the weight scales
    both signs. The input tensors are not modified.
    """
    unknown = sorted(set(weights) - set(scores))
    if unknown:
        raise ValueError(f"layer_prior: weights for layers {unknown} that have no scores")
    out: dict[int, torch.Tensor] = {}
    for layer, s in scores.items():
        w = float(weights.get(layer, 1.0))
        if not (math.isfinite(w) and w >= 0.0):
            raise ValueError(f"layer_prior: weight for layer {layer} must be finite and >= 0, got {w!r}")
        out[layer] = s.detach().double().cpu() * w
    return out


# ============================================================================== grouping
def group_channels(
    scores: Mapping[int, torch.Tensor], *, bands: int = 4, layers: Iterable[int] | None = None
) -> list[ChannelGroup]:
    """Partition every channel of the scored layers into ``layer x score-quantile band`` groups.

    Per layer (ascending; restricted to ``layers`` when given, each of which must be scored) the
    ``n`` channels are ranked by score **descending**, ties by channel index ascending, and rank
    ``r`` goes to band ``floor(r * bands / n)``; band sizes therefore differ by at most one and
    ``b0`` holds the highest scores. Groups are named ``"L{layer}/b{band}"`` and returned in
    ``(layer, band)`` order. When ``n < bands`` the empty bands are omitted (names keep their band
    index). Every channel of every chosen layer is in exactly one group. ``ValueError`` for
    ``bands < 1``, NaN scores, non-1-D tensors or unknown layers.
    """
    if bands < 1:
        raise ValueError("bands must be >= 1")
    chosen = sorted(scores) if layers is None else sorted(set(layers))
    missing = [layer for layer in chosen if layer not in scores]
    if missing:
        raise ValueError(f"group_channels: no scores for layers {missing}")
    groups: list[ChannelGroup] = []
    for layer in chosen:
        s = scores[layer].detach()
        if s.dim() != 1:
            raise ValueError(f"group_channels: scores for layer {layer} must be 1-D, got {tuple(s.shape)}")
        if bool(torch.isnan(s).any()):
            raise ValueError(f"group_channels: scores for layer {layer} contain NaN")
        vals = s.double().cpu().tolist()
        n = len(vals)
        order = sorted(range(n), key=lambda c: (-vals[c], c))
        members: list[list[int]] = [[] for _ in range(bands)]
        for rank, channel in enumerate(order):
            members[rank * bands // n].append(channel)
        for band, chans in enumerate(members):
            if chans:
                groups.append(
                    ChannelGroup(name=f"L{layer}/b{band}", selection=ChannelSelection({layer: chans}))
                )
    return groups


def density_matched_random(selection: ChannelSelection, model: nn.Module, seed: int) -> ChannelSelection:
    """A uniformly random selection with exactly ``selection``'s per-layer channel counts.

    For each selected layer (ascending) ``count`` distinct channels are drawn uniformly from
    ``range(intermediate_size)`` with ``torch.randperm`` from one CPU generator seeded by ``seed``.
    The draw may overlap the original selection (it is an independent uniform draw, not its
    complement). Validates ``selection`` against ``model`` (``ChannelSelectionError``). The
    global torch RNG is not touched.
    """
    selection.validate(model)
    sizes = {r.layer_index: r.intermediate_size for r in find_swiglu_ffns(model, selection.layers())}
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    out: dict[int, list[int]] = {}
    for layer, chans in selection.items():
        perm = torch.randperm(sizes[layer], generator=gen)
        out[layer] = perm[: len(chans)].tolist()
    return ChannelSelection(out)


__all__ = [
    "ChannelStats",
    "collect_channel_stats",
    "contrast_scores",
    "density_matched_random",
    "group_channels",
    "layer_prior",
    "random_scores",
    "usage_ratio_scores",
    "wanda_scores",
]
