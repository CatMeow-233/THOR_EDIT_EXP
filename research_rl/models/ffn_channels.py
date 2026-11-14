"""Channel-level parameter operations on SwiGLU FFNs (selective fine-tuning, diffing, rollback).

For a SwiGLU FFN ``y = W_down (act(W_gate x + b_gate) * (W_up x + b_up)) + b_down`` the intermediate
channel ``c`` *owns* row ``c`` of ``gate_proj.weight``/``up_proj.weight``, column ``c`` of
``down_proj.weight`` and, when biases exist, entry ``c`` of ``gate_proj.bias``/``up_proj.bias``.
``down_proj.bias`` lives in hidden space and is owned by no channel. Changing only the slices owned
by channel ``c`` changes the FFN output only through channel ``c``'s additive contribution
``W_down[:, c] * act(g_c) * u_c``.

Every function below is implemented (no contract-only parts) and exercised by the numerical tests in
``tests/research_rl/test_ffn_channels.py``:

* :class:`ChannelSelection` – validated ``layer -> sorted unique channels`` sets with algebra and
  top-k selection from scores.
* :func:`channel_mask` / :func:`freeze_mask` – :class:`ParameterMask` for "only these channels may
  change" / "these channels are frozen, the rest of the FFN may change".
* :func:`setup_channel_training` – the :class:`MaskRule` to pass as ``update_rules`` plus
  backward-time gradient masks (so frozen gradients never enter HF clipping / ``grad_norm``).
* :func:`extract_channels` / :func:`channel_state` – detached copies of channel slices.
* :func:`channel_diff` / :func:`top_changed_channels` – per-channel change statistics and ranking.
* :func:`rollback_channels` / :func:`ablate_channels` – in-place, dtype-preserving writes of only
  the selected channel slices.
* :func:`verify_modification_scope` – element-exact report of every change outside an allowed region.

Expert containers (:mod:`research_rl.models.expert_manager`): every op takes ``expert=`` and, like
:func:`find_swiglu_ffns`, defaults to the container's *active* expert (``routing`` only affects the
forward pass, not which expert these ops address). :class:`LayerChannels` records the module path it
was taken from; :func:`rollback_channels` / :func:`channel_diff` refuse a reference taken from a
different module path (e.g. another expert) unless ``allow_path_mismatch=True``.

Not handled: tensor-parallel / FSDP-sharded or quantised weights (the ops index full local
``nn.Linear`` weights), and optimizer state (rollback restores weights, not Adam moments).
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from research_rl.models.model_interface import SwiGLURef, find_swiglu_ffns, snapshot
from research_rl.models.parameter_mask import ParameterMask

if TYPE_CHECKING:  # research_rl.optim imports research_rl.models: import it lazily to avoid a cycle
    from research_rl.optim.gradient_hooks import GradientHookHandle
    from research_rl.optim.update_interface import MaskRule

Metric = Literal["l2", "max_abs", "relative"]


class ChannelSelectionError(ValueError):
    pass


class ScopeViolationError(AssertionError):
    """Raised by :func:`verify_modification_scope` in strict mode."""

    def __init__(self, report: ScopeReport) -> None:
        super().__init__(report.summary())
        self.report = report


# ============================================================================== selection
class ChannelSelection:
    """Immutable mapping ``layer_index -> sorted unique channel indices``.

    Layers with no selected channel are dropped, so ``layers()`` are exactly the affected layers.
    Construction checks types and non-negativity; :meth:`validate` checks layer existence and
    ``channel < intermediate_size`` against a model.
    """

    __slots__ = ("_channels",)

    def __init__(
        self, channels: Mapping[int, Iterable[int]] | None = None, *, intermediate_size: int | None = None
    ) -> None:
        clean: dict[int, tuple[int, ...]] = {}
        for layer, chans in (channels or {}).items():
            layer_i = _as_index(layer, "layer index")
            vals = sorted({_as_index(c, f"channel index (layer {layer_i})") for c in _iter_ints(chans)})
            if intermediate_size is not None and vals and vals[-1] >= intermediate_size:
                raise ChannelSelectionError(
                    f"layer {layer_i}: channel {vals[-1]} >= intermediate_size {intermediate_size}"
                )
            if vals:
                clean[layer_i] = tuple(vals)
        self._channels = dict(sorted(clean.items()))

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dict(
        cls, d: Mapping[int | str, Iterable[int]], *, intermediate_size: int | None = None
    ) -> ChannelSelection:
        """Build from a (possibly JSON-loaded, string-keyed) dict.

        ``str`` keys must be integer literals (``"3"``); every other key goes through the same
        int-only check as the constructor (bools and floats are rejected, not truncated).
        """
        clean: dict[int, Iterable[int]] = {}
        for k, v in d.items():
            if isinstance(k, str):
                try:
                    key = _as_index(int(k), "layer index")
                except ValueError as e:
                    raise ChannelSelectionError(f"layer key {k!r} is not an integer literal") from e
            else:
                key = _as_index(k, "layer index")
            if key in clean:
                raise ChannelSelectionError(f"layer {key} appears more than once in {list(d)!r}")
            clean[key] = v
        return cls(clean, intermediate_size=intermediate_size)

    @classmethod
    def all_channels(
        cls, model: nn.Module, layers: Iterable[int] | None = None, *, expert: str | None = None
    ) -> ChannelSelection:
        refs = find_swiglu_ffns(model, layers, expert=expert)
        return cls({r.layer_index: range(r.intermediate_size) for r in refs})

    @classmethod
    def top_k_from_scores(
        cls, scores: Mapping[int, torch.Tensor], k: int, *, per_layer: bool = False, largest: bool = True
    ) -> ChannelSelection:
        """Select the ``k`` best channels globally (or ``k`` per layer).

        ``scores[layer]`` is a 1-D tensor indexed by channel. Ties break by ``(layer, channel)``
        ascending, so the result is deterministic. NaN scores are rejected.
        """
        if k < 0:
            raise ChannelSelectionError(f"k must be >= 0, got {k}")
        entries: list[tuple[float, int, int]] = []
        for layer, s in scores.items():
            s = s.detach()
            if s.dim() != 1:
                raise ChannelSelectionError(
                    f"scores for layer {layer} must be 1-D, got shape {tuple(s.shape)}"
                )
            if torch.isnan(s).any():
                raise ChannelSelectionError(f"scores for layer {layer} contain NaN")
            vals = s.double().cpu().tolist()
            entries.extend((v, int(layer), c) for c, v in enumerate(vals))
        return cls(_rank_entries(entries, k, per_layer=per_layer, largest=largest))

    # ------------------------------------------------------------------ queries
    def layers(self) -> list[int]:
        return list(self._channels)

    def channels(self, layer: int) -> tuple[int, ...]:
        return self._channels.get(layer, ())

    def items(self) -> Iterator[tuple[int, tuple[int, ...]]]:
        return iter(self._channels.items())

    def count(self) -> int:
        return sum(len(c) for c in self._channels.values())

    def to_dict(self) -> dict[int, list[int]]:
        return {k: list(v) for k, v in self._channels.items()}

    def index_tensor(self, layer: int, device: torch.device | str | None = None) -> torch.Tensor:
        return torch.tensor(self.channels(layer), dtype=torch.long, device=device)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, tuple) and len(item) == 2:
            return item[1] in self._channels.get(item[0], ())
        return item in self._channels

    def __len__(self) -> int:
        return len(self._channels)

    def __bool__(self) -> bool:
        return bool(self._channels)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ChannelSelection) and self._channels == other._channels

    def __hash__(self) -> int:
        return hash(tuple(self._channels.items()))

    def __repr__(self) -> str:
        return f"ChannelSelection({self.to_dict()})"

    # ------------------------------------------------------------------ algebra
    def union(self, other: ChannelSelection) -> ChannelSelection:
        keys = set(self._channels) | set(other._channels)
        return ChannelSelection({k: set(self.channels(k)) | set(other.channels(k)) for k in keys})

    def intersection(self, other: ChannelSelection) -> ChannelSelection:
        keys = set(self._channels) & set(other._channels)
        return ChannelSelection({k: set(self.channels(k)) & set(other.channels(k)) for k in keys})

    def difference(self, other: ChannelSelection) -> ChannelSelection:
        return ChannelSelection({k: set(v) - set(other.channels(k)) for k, v in self._channels.items()})

    def complement(self, model: nn.Module, *, expert: str | None = None) -> ChannelSelection:
        """All channels of the *affected* layers that are not selected."""
        sizes = _intermediate_sizes(model, self.layers(), expert)
        return ChannelSelection({k: set(range(sizes[k])) - set(v) for k, v in self._channels.items()})

    # ------------------------------------------------------------------ validation
    def validate(self, model: nn.Module, *, expert: str | None = None) -> None:
        sizes = _intermediate_sizes(model, self.layers(), expert)
        for layer, chans in self._channels.items():
            if chans[-1] >= sizes[layer]:
                raise ChannelSelectionError(
                    f"layer {layer}: channel {chans[-1]} >= intermediate_size {sizes[layer]}"
                )


def _iter_ints(chans: Iterable[int] | torch.Tensor) -> Iterable[int]:
    if isinstance(chans, torch.Tensor):
        if chans.dtype.is_floating_point or chans.dtype == torch.bool:
            raise ChannelSelectionError(f"channel tensor must be integer, got {chans.dtype}")
        return [int(c) for c in chans.flatten().tolist()]
    return chans


def _as_index(value: object, what: str) -> int:
    if isinstance(value, bool):
        raise ChannelSelectionError(f"{what} must be an int, got {value!r}")
    try:
        out = operator.index(value)  # type: ignore[arg-type]
    except TypeError as e:
        raise ChannelSelectionError(f"{what} must be an int, got {value!r}") from e
    if out < 0:
        raise ChannelSelectionError(f"{what} must be >= 0, got {out}")
    return out


def _intermediate_sizes(model: nn.Module, layers: Iterable[int], expert: str | None) -> dict[int, int]:
    layers = list(layers)
    if not layers:
        return {}
    try:
        refs = find_swiglu_ffns(model, layers, expert=expert)
    except ValueError as e:
        raise ChannelSelectionError(str(e)) from e
    return {r.layer_index: r.intermediate_size for r in refs}


def _rank_entries(
    entries: list[tuple[float, int, int]], k: int, *, per_layer: bool, largest: bool
) -> dict[int, list[int]]:
    sign = -1.0 if largest else 1.0
    ordered = sorted(entries, key=lambda e: (sign * e[0], e[1], e[2]))
    out: dict[int, list[int]] = {}
    taken = 0
    for _, layer, ch in ordered:
        bucket = out.setdefault(layer, [])
        if per_layer:
            if len(bucket) < k:
                bucket.append(ch)
        elif taken < k:
            bucket.append(ch)
            taken += 1
    return out


# ============================================================================== masks
def _owned_slice(proj: str, idx: torch.Tensor, shape: torch.Size, *, bias: bool) -> torch.Tensor:
    """Boolean tensor of ``shape`` that is True on the elements owned by channels ``idx``."""
    m = torch.zeros(shape, dtype=torch.bool)
    if bias:
        if proj != "down_proj":
            m[idx] = True
    elif proj == "down_proj":
        m[:, idx] = True
    else:
        m[idx] = True
    return m


def _ffn_params(ref: SwiGLURef) -> list[tuple[str, str, torch.Tensor, bool]]:
    """``(qualified_name, proj, param, is_bias)`` for every weight/bias of the FFN."""
    out: list[tuple[str, str, torch.Tensor, bool]] = []
    for proj, lin in (("gate_proj", ref.gate), ("up_proj", ref.up), ("down_proj", ref.down)):
        out.append((f"{ref.module_path}.{proj}.weight", proj, lin.weight, False))
        if lin.bias is not None:
            out.append((f"{ref.module_path}.{proj}.bias", proj, lin.bias, True))
    return out


def channel_mask(
    model: nn.Module, selection: ChannelSelection, *, trainable: bool = True, expert: str | None = None
) -> ParameterMask:
    """Mask over the gate/up/down weights (and biases) of every affected layer (of ``expert``).

    ``trainable=True``: selected channel slices True, every other element of those tensors False
    (``down_proj.bias`` is owned by no channel and is therefore False).
    ``trainable=False``: the complement – selected slices False, the rest of those FFNs True.
    Parameters outside the affected FFNs are not in the mask (unconstrained, see ParameterMask).
    """
    selection.validate(model, expert=expert)
    mask = ParameterMask()
    for ref in find_swiglu_ffns(model, selection.layers(), expert=expert):
        idx = selection.index_tensor(ref.layer_index)
        for name, proj, p, is_bias in _ffn_params(ref):
            owned = _owned_slice(proj, idx, p.shape, bias=is_bias)
            mask.set(name, owned if trainable else ~owned)
    return mask


def freeze_mask(model: nn.Module, selection: ChannelSelection, *, expert: str | None = None) -> ParameterMask:
    """Selected channels frozen; everything else in the affected FFNs may change."""
    return channel_mask(model, selection, trainable=False, expert=expert)


# ============================================================================== training setup
@dataclass
class ChannelTrainingHandle:
    """Result of :func:`setup_channel_training`.

    ``rules`` go into ``update_rules=`` (or ``HookedOptimizer.add_rule``); ``remove()`` detaches
    the backward hooks and restores ``requires_grad`` flags. It does not remove the rules from an
    optimizer that already holds them.
    """

    selection: ChannelSelection
    mask: ParameterMask  # what the MaskRule enforces
    rules: list[MaskRule]
    gradient_hooks: GradientHookHandle | None
    disabled_grad: list[str] = field(default_factory=list)
    _params: dict[str, nn.Parameter] = field(default_factory=dict, repr=False)

    def remove(self) -> None:
        if self.gradient_hooks is not None:
            self.gradient_hooks.remove()
            self.gradient_hooks = None
        for name in self.disabled_grad:
            self._params[name].requires_grad_(True)
        self.disabled_grad = []


def setup_channel_training(
    model: nn.Module,
    selection: ChannelSelection,
    *,
    freeze_rest: bool = True,
    disable_grad_outside: bool = False,
    gradient_masks: bool = True,
    expert: str | None = None,
) -> ChannelTrainingHandle:
    """Prepare ``model`` so that only the selected channel slices (of ``expert``) can change.

    * The returned :class:`MaskRule` restores every frozen element after each optimizer step with
      ``torch.where`` (exact under Adam momentum and decoupled weight decay).
    * ``gradient_masks``: backward-time hooks zero frozen gradient elements so they never enter
      HF gradient clipping or the logged ``grad_norm``.
    * ``freeze_rest``: every parameter outside the affected FFNs gets an all-False mask as well.
      Cost: the masks themselves (one bool per model element, in memory), and, because every
      masked parameter is *tracked* by the rule, :class:`HookedOptimizer` snapshots the whole model
      (one clone in its working dtype) on every step. Only a SHA-256 of the mask is checkpointed
      (``MaskRule(store_mask_in_checkpoint=False)``), so at resume the caller must re-create the
      same selection; a different one is rejected by the digest check. For large models prefer
      ``disable_grad_outside=True``. With ``freeze_rest=False`` those parameters are unconstrained.
    * ``disable_grad_outside``: every parameter outside the affected FFNs (non-FFN parameters
      *and* the FFN weights of unselected layers) gets ``requires_grad=False`` and ``grad=None``
      (torch optimizers skip ``grad is None`` parameters, so stale Adam moments / weight decay
      cannot move them); they are then left out of the MaskRule's mask, saving memory and the
      per-step snapshot. ``grad=None`` is also applied to parameters that were already
      ``requires_grad=False`` (a stale ``.grad`` would otherwise let the optimizer move them);
      those are not listed in ``disabled_grad`` and stay frozen after ``remove()``. Must be
      called before the trainer builds its optimizer to also keep them out of its param groups.
      If gradient checkpointing is used with frozen embeddings, call
      ``model.enable_input_require_grads()`` yourself.
    """
    from research_rl.optim.gradient_hooks import register_gradient_masks
    from research_rl.optim.update_interface import MaskRule

    rule_mask = channel_mask(model, selection, trainable=True, expert=expert)
    params = dict(model.named_parameters())
    disabled: list[str] = []
    for name, p in params.items():
        if name in rule_mask:
            continue
        if disable_grad_outside:
            p.grad = None  # also for parameters frozen before this call (stale grads)
            if p.requires_grad:
                p.requires_grad_(False)
                disabled.append(name)
        elif freeze_rest:
            rule_mask.set(name, torch.zeros(p.shape, dtype=torch.bool))
    hooks = register_gradient_masks(model, rule_mask) if gradient_masks else None
    return ChannelTrainingHandle(
        selection=selection,
        mask=rule_mask,
        rules=[MaskRule(rule_mask)],
        gradient_hooks=hooks,
        disabled_grad=disabled,
        _params=params,
    )


# ============================================================================== channel state
@dataclass
class LayerChannels:
    """Detached copies of the slices owned by ``channels`` in one layer (original dtype).

    ``module_path`` is the FFN module the slices were read from (``None`` = unknown, not checked).
    """

    channels: torch.Tensor  # long [k]
    gate: torch.Tensor  # [k, hidden]
    up: torch.Tensor  # [k, hidden]
    down: torch.Tensor  # [hidden, k]
    gate_bias: torch.Tensor | None = None  # [k]
    up_bias: torch.Tensor | None = None  # [k]
    module_path: str | None = None

    def vectors(self, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """``[k, D]``: per channel the concatenation [gate row, up row, down col, (biases)]."""
        parts = [self.gate.to(dtype), self.up.to(dtype), self.down.t().to(dtype)]
        parts += [b.to(dtype).unsqueeze(1) for b in (self.gate_bias, self.up_bias) if b is not None]
        return torch.cat(parts, dim=1)

    def select(self, channels: Iterable[int]) -> LayerChannels:
        """Sub-state for ``channels`` (each must be present)."""
        pos = {int(c): i for i, c in enumerate(self.channels.tolist())}
        wanted = list(channels)
        missing = [c for c in wanted if c not in pos]
        if missing:
            raise ChannelSelectionError(f"channels {missing} are not in this state")
        rows = torch.tensor([pos[c] for c in wanted], dtype=torch.long)
        return LayerChannels(
            channels=self.channels[rows],
            gate=self.gate[rows],
            up=self.up[rows],
            down=self.down[:, rows],
            gate_bias=None if self.gate_bias is None else self.gate_bias[rows],
            up_bias=None if self.up_bias is None else self.up_bias[rows],
            module_path=self.module_path,
        )


@dataclass
class ChannelState:
    layers: dict[int, LayerChannels]

    def __getitem__(self, layer: int) -> LayerChannels:
        return self.layers[layer]

    def __contains__(self, layer: object) -> bool:
        return layer in self.layers

    def __iter__(self) -> Iterator[int]:
        return iter(self.layers)

    def selection(self) -> ChannelSelection:
        return ChannelSelection({k: v.channels.tolist() for k, v in self.layers.items()})


def _check_path(stored: str | None, target: str | None, layer: int, allow: bool) -> None:
    if allow or stored is None or target is None or stored == target:
        return
    raise ChannelSelectionError(
        f"layer {layer}: reference was taken from {stored!r} but the target is {target!r} "
        "(another expert?); pass allow_path_mismatch=True if this is intended"
    )


@torch.no_grad()
def extract_channels(
    model: nn.Module, selection: ChannelSelection, *, expert: str | None = None
) -> ChannelState:
    """Detached CPU copies (original dtype) of the slices owned by the selected channels."""
    selection.validate(model, expert=expert)
    out: dict[int, LayerChannels] = {}
    for ref in find_swiglu_ffns(model, selection.layers(), expert=expert):
        idx = selection.index_tensor(ref.layer_index, device=ref.gate.weight.device)

        def take(t: torch.Tensor, dim: int, idx: torch.Tensor = idx) -> torch.Tensor:
            return t.detach().index_select(dim, idx.to(t.device)).cpu().clone()

        out[ref.layer_index] = LayerChannels(
            channels=idx.cpu().clone(),
            gate=take(ref.gate.weight, 0),
            up=take(ref.up.weight, 0),
            down=take(ref.down.weight, 1),
            gate_bias=None if ref.gate.bias is None else take(ref.gate.bias, 0),
            up_bias=None if ref.up.bias is None else take(ref.up.bias, 0),
            module_path=ref.module_path,
        )
    return ChannelState(out)


def channel_state(
    model: nn.Module, layers: Iterable[int] | None = None, *, expert: str | None = None
) -> ChannelState:
    """Every channel of ``layers`` (default: all FFN layers). Use as a rollback/diff reference."""
    return extract_channels(model, ChannelSelection.all_channels(model, layers, expert=expert), expert=expert)


# ============================================================================== diff + ranking
@dataclass
class ChannelDiff:
    """Per-channel change statistics for one layer (float64)."""

    layer: int
    channels: torch.Tensor  # long [k]
    l2: torch.Tensor  # ||Δ[gate row, up row, down col, (biases)]||_2 per channel
    max_abs: torch.Tensor  # max |Δ| per channel
    ref_l2: torch.Tensor  # ||reference vector||_2 per channel
    # "gate_proj"/"up_proj"/"down_proj" (weights) and "bias" (gate/up bias entries, if any) -> [k];
    # the squares of these parts sum to l2**2.
    per_projection_l2: dict[str, torch.Tensor]

    @property
    def relative(self) -> torch.Tensor:
        return self.l2 / self.ref_l2.clamp_min(torch.finfo(torch.float64).tiny)

    def score(self, metric: Metric) -> torch.Tensor:
        if metric == "l2":
            return self.l2
        if metric == "max_abs":
            return self.max_abs
        if metric == "relative":
            return self.relative
        raise ValueError(f"unknown metric {metric!r}")

    def changed_channels(self) -> list[int]:
        return [int(c) for c, m in zip(self.channels.tolist(), self.max_abs.tolist()) if m != 0]


def channel_diff(
    reference: ChannelState,
    current: ChannelState | nn.Module,
    layers: Iterable[int] | None = None,
    *,
    expert: str | None = None,
    allow_path_mismatch: bool = False,
) -> dict[int, ChannelDiff]:
    """Compare two channel states (or a state against a live model) channel by channel.

    ``expert`` selects the expert read from a live model (default: the active one). Comparing slices
    taken from different module paths (e.g. expert ``a`` against ``base``) needs
    ``allow_path_mismatch=True``.
    """
    wanted = list(reference.layers) if layers is None else list(layers)
    for layer in wanted:
        if layer not in reference:
            raise ChannelSelectionError(f"layer {layer} missing from reference state")
    if isinstance(current, nn.Module):
        cur_state = extract_channels(
            current, ChannelSelection({k: reference[k].channels.tolist() for k in wanted}), expert=expert
        )
    else:
        cur_state = current
    out: dict[int, ChannelDiff] = {}
    for layer in wanted:
        if layer not in cur_state:
            raise ChannelSelectionError(f"layer {layer} missing from current state")
        ref = reference[layer]
        _check_path(ref.module_path, cur_state[layer].module_path, layer, allow_path_mismatch)
        cur = cur_state[layer].select(ref.channels.tolist())
        rv, cv = ref.vectors(), cur.vectors()
        if rv.shape != cv.shape:
            raise ChannelSelectionError(
                f"layer {layer}: shape mismatch {tuple(rv.shape)} vs {tuple(cv.shape)}"
            )
        delta = cv - rv
        h_in, h_out = ref.gate.shape[1], ref.down.shape[0]
        bounds = {
            "gate_proj": (0, h_in),
            "up_proj": (h_in, 2 * h_in),
            "down_proj": (2 * h_in, 2 * h_in + h_out),
        }
        if delta.shape[1] > 2 * h_in + h_out:
            bounds["bias"] = (2 * h_in + h_out, delta.shape[1])
        out[layer] = ChannelDiff(
            layer=layer,
            channels=ref.channels.clone(),
            l2=torch.linalg.vector_norm(delta, dim=1),
            max_abs=delta.abs().amax(dim=1),
            ref_l2=torch.linalg.vector_norm(rv, dim=1),
            per_projection_l2={
                k: torch.linalg.vector_norm(delta[:, i:j], dim=1) for k, (i, j) in bounds.items()
            },
        )
    return out


def ranked_channels(diff: Mapping[int, ChannelDiff], metric: Metric = "l2") -> list[tuple[int, int, float]]:
    """All ``(layer, channel, score)`` sorted by score descending, ties by (layer, channel).

    NaN scores (e.g. a weight that became NaN) are placed first, ordered by (layer, channel).
    """
    entries = [
        (float(s), layer, int(c))
        for layer, d in diff.items()
        for c, s in zip(d.channels.tolist(), d.score(metric).tolist())
    ]

    def key(e: tuple[float, int, int]) -> tuple[int, float, int, int]:
        return (0, 0.0, e[1], e[2]) if math.isnan(e[0]) else (1, -e[0], e[1], e[2])

    return [(layer, c, s) for s, layer, c in sorted(entries, key=key)]


def top_changed_channels(
    diff: Mapping[int, ChannelDiff],
    k: int,
    *,
    metric: Metric = "l2",
    per_layer: bool = False,
    min_score: float | None = None,
) -> ChannelSelection:
    """The ``k`` most-changed channels (globally or per layer); ``min_score`` drops scores <= it.

    Raises :class:`ChannelSelectionError` if any score is NaN (checked before ``min_score`` filtering).
    """
    ranked = ranked_channels(diff, metric)
    nan = [(layer, c) for layer, c, s in ranked if math.isnan(s)]
    if nan:
        raise ChannelSelectionError(f"NaN change scores for (layer, channel) {nan[:10]}")
    entries = [(s, layer, c) for layer, c, s in ranked if min_score is None or s > min_score]
    return ChannelSelection(_rank_entries(entries, k, per_layer=per_layer, largest=True))


# ============================================================================== rollback / ablation
@dataclass
class RollbackReport:
    """Per layer: rolled-back channels, how far they were from the reference, elements that changed.

    Both statistics compare the pre-write values with the values actually written (the reference
    cast to the parameter dtype); NaN -> NaN counts as unchanged.
    """

    channels: dict[int, list[int]]
    delta_l2: dict[int, torch.Tensor]  # per channel ||current - written|| before the write (float64)
    elements_changed: dict[int, int]
    elements_written: int

    @property
    def total_elements_changed(self) -> int:
        return sum(self.elements_changed.values())


@torch.no_grad()
def rollback_channels(
    model: nn.Module,
    reference: ChannelState,
    selection: ChannelSelection | None = None,
    *,
    allow_cast: bool = False,
    expert: str | None = None,
    allow_path_mismatch: bool = False,
) -> RollbackReport:
    """Restore ONLY the selected channels' slices (gate row, up row, down col, biases) to ``reference``.

    In place, under ``no_grad``, parameter dtype/device preserved. ``selection`` defaults to every
    channel stored in ``reference``. A reference stored in another dtype is rejected unless
    ``allow_cast`` (the cast may be lossy). ``expert`` picks the target expert (default: active);
    a reference taken from another module path is rejected unless ``allow_path_mismatch``.
    Atomic with respect to validation: every check (missing layers/channels, module path, bias
    presence, slice shapes, dtypes) runs for all layers before the first write, so a failing call
    leaves the model untouched. Optimizer state is not touched.
    """
    selection = reference.selection() if selection is None else selection
    selection.validate(model, expert=expert)
    channels: dict[int, list[int]] = {}
    delta_l2: dict[int, torch.Tensor] = {}
    changed: dict[int, int] = {}
    writes: list[tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]] = []
    # Pass 1: validate everything and stage the casted values; no parameter is written here.
    for ref_ffn in find_swiglu_ffns(model, selection.layers(), expert=expert):
        layer = ref_ffn.layer_index
        if layer not in reference:
            raise ChannelSelectionError(f"reference has no channels for layer {layer}")
        _check_path(reference[layer].module_path, ref_ffn.module_path, layer, allow_path_mismatch)
        chans = list(selection.channels(layer))
        src = reference[layer].select(chans)
        pairs: list[tuple[str, torch.Tensor, torch.Tensor | None, int]] = [
            ("gate_proj.weight", ref_ffn.gate.weight, src.gate, 0),
            ("up_proj.weight", ref_ffn.up.weight, src.up, 0),
            ("down_proj.weight", ref_ffn.down.weight, src.down, 1),
        ]
        for proj, lin, b in (
            ("gate_proj", ref_ffn.gate, src.gate_bias),
            ("up_proj", ref_ffn.up, src.up_bias),
        ):
            if (lin.bias is None) != (b is None):
                raise ChannelSelectionError(
                    f"layer {layer}: {proj} bias presence differs between model and reference"
                )
            if lin.bias is not None:
                pairs.append((f"{proj}.bias", lin.bias, b, 0))
        staged: dict[str, torch.Tensor] = {}
        for what, p, value, dim in pairs:
            assert value is not None
            want = list(p.shape)
            want[dim] = len(chans)
            if list(value.shape) != want:
                raise ChannelSelectionError(
                    f"layer {layer} {what}: reference slice shape {tuple(value.shape)} != {tuple(want)}"
                )
            if value.dtype != p.dtype and not allow_cast:
                raise ChannelSelectionError(
                    f"layer {layer}: reference dtype {value.dtype} != parameter dtype {p.dtype} "
                    "(pass allow_cast=True to cast)"
                )
            idx = torch.tensor(chans, dtype=torch.long, device=p.device)
            v = value.to(device=p.device, dtype=p.dtype)
            writes.append((p, dim, idx, v))
            staged[what] = v.detach().cpu()
        before = extract_channels(model, ChannelSelection({layer: chans}), expert=expert)[layer]
        target = LayerChannels(
            channels=src.channels,
            gate=staged["gate_proj.weight"],
            up=staged["up_proj.weight"],
            down=staged["down_proj.weight"],
            gate_bias=staged.get("gate_proj.bias"),
            up_bias=staged.get("up_proj.bias"),
        )
        bv, tv = before.vectors(), target.vectors()
        diff = _changed(bv, tv)
        delta_l2[layer] = torch.linalg.vector_norm(torch.where(diff, bv - tv, 0.0), dim=1)
        changed[layer] = int(diff.sum())
        channels[layer] = chans
    # Pass 2: write.
    written = 0
    for p, dim, idx, v in writes:
        p.index_copy_(dim, idx, v)
        written += v.numel()
    return RollbackReport(
        channels=channels, delta_l2=delta_l2, elements_changed=changed, elements_written=written
    )


@torch.no_grad()
def ablate_channels(
    model: nn.Module, selection: ChannelSelection, *, expert: str | None = None
) -> ChannelState:
    """Zero the selected channels' slices in place; returns their previous values (for rollback)."""
    previous = extract_channels(model, selection, expert=expert)
    zeros = ChannelState(
        {
            layer: LayerChannels(
                channels=lc.channels,
                gate=torch.zeros_like(lc.gate),
                up=torch.zeros_like(lc.up),
                down=torch.zeros_like(lc.down),
                gate_bias=None if lc.gate_bias is None else torch.zeros_like(lc.gate_bias),
                up_bias=None if lc.up_bias is None else torch.zeros_like(lc.up_bias),
                module_path=lc.module_path,
            )
            for layer, lc in previous.layers.items()
        }
    )
    rollback_channels(model, zeros, selection, expert=expert)
    return previous


# ============================================================================== scope verification
@dataclass
class ScopeViolation:
    count: int
    max_abs: float
    indices: torch.Tensor  # long [count, ndim]: every changed element outside the allowed region


@dataclass
class ScopeReport:
    violations: dict[str, ScopeViolation]
    params_checked: int
    elements_checked: int
    changed_allowed: int  # changed elements inside the allowed region

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def total_violations(self) -> int:
        return sum(v.count for v in self.violations.values())

    def summary(self) -> str:
        if self.ok:
            return (
                f"scope ok: {self.params_checked} params / {self.elements_checked} elements checked, "
                f"{self.changed_allowed} allowed elements changed"
            )
        lines = [f"{self.total_violations} element(s) changed outside the allowed scope:"]
        lines += [f"  {n}: {v.count} (max |Δ|={v.max_abs:.3e})" for n, v in sorted(self.violations.items())]
        return "\n".join(lines)


def _changed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a != b
    if a.dtype.is_floating_point or a.dtype.is_complex:
        diff &= ~(torch.isnan(a) & torch.isnan(b))
    return diff


@torch.no_grad()
def verify_modification_scope(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor] | nn.Module,
    allowed: ParameterMask | ChannelSelection,
    model: nn.Module,
    *,
    unlisted: Literal["frozen", "free"] = "frozen",
    strict: bool = False,
    expert: str | None = None,
) -> ScopeReport:
    """Report every element of ``before`` that differs in ``after`` outside ``allowed``.

    ``allowed`` as a :class:`ChannelSelection` means exactly the selected channel slices may change.
    Parameters absent from the allowed mask are treated as frozen (``unlisted="frozen"``, the
    default for scope checks) or ignored (``"free"``). NaN==NaN counts as unchanged. ``strict``
    raises :class:`ScopeViolationError` on any violation. ``expert`` resolves a ChannelSelection
    against that expert's FFNs (default: the active one).
    """
    after_t = snapshot(after) if isinstance(after, nn.Module) else after
    mask = channel_mask(model, allowed, expert=expert) if isinstance(allowed, ChannelSelection) else allowed
    missing = sorted(set(before) ^ set(after_t))
    if missing:
        raise KeyError(f"snapshots disagree on parameter names: {missing[:10]}")
    unknown = sorted(set(mask.names()) - set(before))
    if unknown:
        raise KeyError(f"allowed mask refers to parameters not in the snapshots: {unknown[:10]}")
    violations: dict[str, ScopeViolation] = {}
    n_params = n_elems = changed_allowed = 0
    for name in sorted(before):
        a, b = before[name].detach().cpu(), after_t[name].detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(
                f"{name}: shape/dtype changed {tuple(a.shape)}/{a.dtype} -> {tuple(b.shape)}/{b.dtype}"
            )
        m = mask.get(name)
        if m is None and unlisted == "free":
            continue
        m = torch.zeros(a.shape, dtype=torch.bool) if m is None else m.cpu()
        if m.shape != a.shape:
            raise ValueError(f"{name}: mask shape {tuple(m.shape)} != parameter shape {tuple(a.shape)}")
        n_params += 1
        n_elems += a.numel()
        diff = _changed(a, b)
        changed_allowed += int((diff & m).sum())
        bad = diff & ~m
        count = int(bad.sum())
        if count:
            delta = (b[bad].double() - a[bad].double()).abs()
            violations[name] = ScopeViolation(
                count=count, max_abs=float(delta.max()), indices=torch.nonzero(bad, as_tuple=False)
            )
    report = ScopeReport(violations, n_params, n_elems, changed_allowed)
    if strict and not report.ok:
        raise ScopeViolationError(report)
    return report


__all__ = [
    "ChannelDiff",
    "ChannelSelection",
    "ChannelSelectionError",
    "ChannelState",
    "ChannelTrainingHandle",
    "LayerChannels",
    "RollbackReport",
    "ScopeReport",
    "ScopeViolation",
    "ScopeViolationError",
    "ablate_channels",
    "channel_diff",
    "channel_mask",
    "channel_state",
    "extract_channels",
    "freeze_mask",
    "ranked_channels",
    "rollback_channels",
    "setup_channel_training",
    "top_changed_channels",
    "verify_modification_scope",
]
