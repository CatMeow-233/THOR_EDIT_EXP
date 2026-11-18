"""Exact weight snapshots, temporary swaps, interpolation and channel-group rollback.

The gate (``docs/research_rl/CAPABILITY.md`` section 1.5) compares a *candidate* (the current
weights) with the *incumbent* (the last accepted weights) and may then roll back, shrink the step
or revert channel groups. Every one of those operations is a copy, so this module makes them
**bit-exact** and **atomic with respect to validation** (every check runs before the first write;
a failing check leaves the model and the optimizer untouched). One exception lies outside that
guarantee: a ``HookedOptimizer`` reloads its update rules' state last inside its own
``load_state_dict`` (e.g. :class:`~research_rl.optim.update_interface.MaskRule` raises on a mask
digest mismatch), i.e. after the parameters were written; such an error is not pre-checked here.

* :class:`StateSnapshot` / :func:`capture` - CPU clones of named parameters (original dtype), a
  deep copy of ``optimizer.state_dict()`` (every tensor cloned, so later optimizer steps never leak
  into the snapshot), the fp32 master weights of a
  :class:`~research_rl.optim.optimizer_wrapper.HookedOptimizer` and optionally a
  :class:`~research_rl.models.ffn_channels.ChannelState`.
* :func:`restore` - parameters, optimizer state and masters back to the snapshot, bit-identical.
* :func:`swapped` - context manager that temporarily loads the snapshot's parameters (to evaluate
  the incumbent) and puts the current values back bit-identically on exit, also on exceptions.
* :func:`interpolate` - ``incumbent + lam * (candidate - incumbent)`` in float32 (float64 for
  float64 parameters), exact copies at ``lam`` 0 and 1, masters updated consistently.
* :func:`group_damage` / :func:`channel_groups_by_damage` - rank channel groups by
  displacement x activation.
* :func:`revert_groups` - revert the channels of some groups with
  :func:`~research_rl.models.ffn_channels.rollback_channels` and fix the masters.

Master weights
--------------
A ``HookedOptimizer(master_weights=True)`` keeps an fp32 master of every bf16/fp16 parameter and
detects external writes only element-wise (see its docstring). Writing the served weights back is
therefore not enough for an exact rollback, and ``HookedOptimizer.set_master`` rewrites the whole
served tensor with round-to-nearest (which changes served bits under ``write_rounding=
"stochastic"``). Every function here installs masters through the wrapper's own public
``state_dict()`` / ``load_state_dict()`` instead: ``state_dict()`` is read *before* any parameter
is written (so no spurious "modified outside the optimizer" resync happens), the parameters are
written, then the state dict - with the ``["research_rl"]["precision"]["masters"]`` entries (keyed
by the flat parameter index in ``param_groups`` order) replaced - is loaded. ``load_state_dict``
leaves the served weights alone and records for every parameter whether ``served ==
master.to(dtype)``, so later external-write detection stays exact. The wrapper is found inside
Accelerate-style wrappers with :func:`~research_rl.optim.optimizer_wrapper.unwrap_hooked`.

Limits (all documented at the functions):

* ``.grad`` buffers, LR-scheduler state and RNG state are not captured. The optimizer's
  ``param_groups`` hyper-parameters (e.g. ``lr``) are part of ``optimizer.state_dict()`` and are
  restored with it; a scheduler that sets ``lr`` from its own counter keeps doing so.
* Optimizer state is global: :func:`restore` with ``restore_optimizer=True`` restores the moments
  of *every* parameter the optimizer holds, also of parameters outside the snapshot's names (their
  weights and masters keep their current values).
* :func:`capture` calls ``optimizer.state_dict()``, which makes a ``HookedOptimizer`` reconcile
  pending external writes into its masters (its documented behaviour).
* Optimizer state is mapped to parameters positionally (torch's ``param_groups`` order); the
  snapshot records no parameter identity, so :func:`restore` rejects state whose tensors do not
  fit the target parameters' shapes, but cannot detect a reordering with identical shapes.
* Only full local tensors (no FSDP / tensor-parallel shards, no quantised weights).

Everything here is verified on CPU with tiny models (``tests/research_rl/test_capability_weights.py``);
nothing has run on a GPU (NOT_TESTED).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import reduce
from typing import Any

import torch
from torch import nn

from research_rl.capability.types import ChannelGroup
from research_rl.models.ffn_channels import (
    ChannelSelection,
    ChannelState,
    LayerChannels,
    RollbackReport,
    channel_diff,
    channel_mask,
    channel_state,
    rollback_channels,
)
from research_rl.models.model_interface import find_swiglu_ffns, fingerprint, snapshot
from research_rl.optim.optimizer_wrapper import HookedOptimizer, unwrap_hooked

__all__ = [
    "SnapshotError",
    "StateSnapshot",
    "capture",
    "channel_groups_by_damage",
    "group_damage",
    "interpolate",
    "restore",
    "revert_groups",
    "swapped",
]

_RR_KEY = "research_rl"  # HookedOptimizer's key inside its state_dict


class SnapshotError(ValueError):
    """A snapshot does not fit the model / optimizer it is applied to (raised before any write)."""


# ============================================================================== the snapshot
@dataclass
class StateSnapshot:
    """An exact copy of (part of) a model's training state. Treat it as immutable.

    * ``params``: ``{named_parameters name: tensor}``, detached CPU clones in the original dtype.
    * ``optimizer_state``: deep copy of ``optimizer.state_dict()`` (every tensor cloned to CPU, the
      containers rebuilt), or ``None`` when captured without an optimizer. For a
      ``HookedOptimizer`` it includes the wrapper's ``"research_rl"`` entry (step counters, rule
      state, masters by flat index).
    * ``masters``: ``{name: fp32 CPU tensor}`` - the fp32 masters of the captured parameters held
      by a ``HookedOptimizer(master_weights=True)``; ``None`` without such an optimizer. The same
      tensor objects are referenced from ``optimizer_state`` (no duplicate memory).
    * ``channel_state``: :func:`~research_rl.models.ffn_channels.channel_state` of the requested
      layers, or ``None``.
    * ``fingerprint``: :func:`~research_rl.models.model_interface.fingerprint` of ``params``
      (SHA-256 over names, shapes, dtypes and bytes); :meth:`verify` recomputes it.
    """

    params: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any] | None
    masters: dict[str, torch.Tensor] | None
    channel_state: ChannelState | None
    fingerprint: str

    def names(self) -> list[str]:
        """Captured parameter names, in ``named_parameters`` order."""
        return list(self.params)

    def verify(self) -> None:
        """Raise :class:`SnapshotError` if ``params`` no longer match ``fingerprint`` (mutated)."""
        now = fingerprint(self.params)
        if now != self.fingerprint:
            raise SnapshotError(
                f"snapshot parameters were modified after capture ({now} != {self.fingerprint})"
            )


# ============================================================================== helpers
def _clone_tensor(t: torch.Tensor, memo: dict[int, torch.Tensor]) -> torch.Tensor:
    key = id(t)
    out = memo.get(key)
    if out is None:
        out = t.detach().to("cpu", copy=True)
        memo[key] = out
    return out


def _clone_tree(obj: Any, memo: dict[int, torch.Tensor] | None = None) -> Any:
    """Deep copy of a state-dict tree: tensors cloned to CPU, dicts/lists/tuples rebuilt.

    A tensor object reached twice is cloned once (``memo``), so aliasing inside the tree is kept.
    Other leaves (numbers, strings, ``None``, ...) are immutable in state dicts and shared.
    """
    memo = {} if memo is None else memo
    if isinstance(obj, torch.Tensor):
        return _clone_tensor(obj, memo)
    if isinstance(obj, dict):
        return {k: _clone_tree(v, memo) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone_tree(v, memo) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_tree(v, memo) for v in obj)
    return obj


def _check_params(
    model: nn.Module, tensors: Mapping[str, torch.Tensor], what: str
) -> dict[str, nn.Parameter]:
    """Map every name of ``tensors`` to the model parameter; raise on missing names / shape / dtype."""
    named = dict(model.named_parameters())
    missing = [n for n in tensors if n not in named]
    if missing:
        raise SnapshotError(
            f"{what}: {len(missing)} parameter name(s) not in the model (different model?): {missing[:5]}"
        )
    bad: list[str] = []
    for n, t in tensors.items():
        p = named[n]
        if tuple(t.shape) != tuple(p.shape) or t.dtype != p.dtype:
            bad.append(f"{n}: {tuple(t.shape)}/{t.dtype} vs model {tuple(p.shape)}/{p.dtype}")
    if bad:
        raise SnapshotError(f"{what}: shape/dtype mismatch for {len(bad)} parameter(s): {bad[:5]}")
    return {n: named[n] for n in tensors}


def _flat_params(optimizer: Any) -> list[torch.Tensor]:
    """Optimised parameters in torch's state_dict numbering (``param_groups`` order, deduplicated)."""
    out: list[torch.Tensor] = []
    seen: set[int] = set()
    for g in optimizer.param_groups:
        for p in g["params"]:
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
    return out


def _check_optimizer_structure(optimizer: Any, state: Mapping[str, Any]) -> None:
    """The checks torch's ``load_state_dict`` would run, done before anything is written."""
    groups = optimizer.param_groups
    saved = state.get("param_groups")
    if not isinstance(saved, list) or "state" not in state:
        raise SnapshotError("snapshot optimizer state is not an optimizer state_dict")
    if len(groups) != len(saved):
        raise SnapshotError(
            f"snapshot optimizer state has {len(saved)} parameter groups, the optimizer has {len(groups)}"
        )
    for i, (g, s) in enumerate(zip(groups, saved)):
        if len(g["params"]) != len(s["params"]):
            raise SnapshotError(
                f"parameter group {i}: snapshot has {len(s['params'])} parameters, optimizer has {len(g['params'])}"
            )
    hooked = unwrap_hooked(optimizer)
    if (hooked is not None) != (_RR_KEY in state):
        raise SnapshotError(
            "snapshot optimizer state and optimizer disagree on HookedOptimizer "
            f"(snapshot has {_RR_KEY!r} entry: {_RR_KEY in state}, optimizer is hooked: {hooked is not None})"
        )
    _check_optimizer_state_shapes(optimizer, state)


def _check_optimizer_state_shapes(optimizer: Any, state: Mapping[str, Any]) -> None:
    """Every tensor of the snapshot's per-parameter state must fit the parameter it will land on.

    torch's ``load_state_dict`` maps saved state to parameters positionally (``zip`` of the saved
    and the current ``param_groups[*]["params"]`` lists) and never checks shapes, so moments of
    another model would load silently and fail (or broadcast) only at the next ``step()``. For
    each mapped parameter ``p`` and each tensor entry ``k`` of its saved state except ``"step"``:

    * if the optimizer currently holds a tensor under ``k`` for ``p``, the shapes must be equal
      (this covers state that is not parameter-shaped, e.g. factored second moments);
    * otherwise (fresh optimizer / no state for ``p`` yet) an entry with ``p.dim() > 0`` and the
      same number of dimensions as ``p`` must match ``p``'s shape in every dimension, where a size
      of 1 is also accepted (factored moments such as ``torch.optim.Adafactor``'s ``row_var``
      ``(m, 1)`` / ``col_var`` ``(1, n)``); entries of another rank (and 0-d entries) are not
      checked, since their layout is optimizer-specific.

    Limit: a different parameter order whose shapes happen to match cannot be detected (the state
    dict records no parameter identity).
    """
    current = optimizer.state_dict()
    cur_state = current.get("state", {})
    saved_ids = [i for g in state["param_groups"] for i in g["params"]]
    cur_ids = [i for g in current["param_groups"] for i in g["params"]]
    params = [p for g in optimizer.param_groups for p in g["params"]]
    saved_state = state["state"]
    bad: list[str] = []
    for saved_id, cur_id, p in zip(saved_ids, cur_ids, params):
        entry = saved_state.get(saved_id)
        if not isinstance(entry, Mapping):
            continue
        cur_entry = cur_state.get(cur_id)
        for key, value in entry.items():
            if key == "step" or not isinstance(value, torch.Tensor):
                continue
            ref = cur_entry.get(key) if isinstance(cur_entry, Mapping) else None
            shape = tuple(value.shape)
            if isinstance(ref, torch.Tensor):
                if shape != tuple(ref.shape):
                    bad.append(f"param {cur_id} state[{key!r}]: {shape}, current {tuple(ref.shape)}")
            elif (
                p.dim() > 0
                and value.dim() == p.dim()
                and any(s not in (d, 1) for s, d in zip(shape, p.shape))
            ):
                bad.append(f"param {cur_id} state[{key!r}]: {shape}, parameter {tuple(p.shape)}")
    if bad:
        raise SnapshotError(
            f"snapshot optimizer state does not fit the optimizer's parameters (different model or "
            f"parameter order?) for {len(bad)} entr(y/ies): {bad[:5]}"
        )


def _check_incumbent(model: nn.Module, incumbent: StateSnapshot, what: str) -> None:
    """Full-shape checks of an incumbent's ``params`` and ``masters`` against ``model``."""
    _check_params(model, incumbent.params, what)
    if incumbent.masters is None:
        return
    named = dict(model.named_parameters())
    bad: list[str] = []
    for n, m in incumbent.masters.items():
        p = named.get(n)
        if p is None:
            bad.append(f"{n}: not a parameter of the model")
        elif tuple(m.shape) != tuple(p.shape):
            bad.append(f"{n}: master {tuple(m.shape)} vs model {tuple(p.shape)}")
    if bad:
        raise SnapshotError(f"{what}: incumbent masters do not fit the model: {bad[:5]}")


@dataclass
class _MasterPlan:
    """A ``HookedOptimizer(master_weights=True)`` and its state dict read before any write."""

    hooked: HookedOptimizer
    state: dict[str, Any]  # live (not cloned) hooked.state_dict()
    entries: list[tuple[int, str, nn.Parameter]]  # (flat index, model name, param) of master params

    def current(self, index: int) -> torch.Tensor:
        masters: dict[int, torch.Tensor] = self.state[_RR_KEY]["precision"]["masters"]
        return masters[index]

    def load(self, base: dict[str, Any], masters: Mapping[int, torch.Tensor]) -> None:
        """``hooked.load_state_dict(base)`` with the masters of ``masters`` replaced (others kept)."""
        rr = dict(base[_RR_KEY])
        precision = dict(rr["precision"])
        merged = dict(precision.get("masters") or {})
        merged.update(masters)
        precision["masters"] = merged
        rr["precision"] = precision
        state = dict(base)
        state[_RR_KEY] = rr
        self.hooked.load_state_dict(state)


def _master_plan(optimizer: Any, model: nn.Module) -> _MasterPlan | None:
    hooked = unwrap_hooked(optimizer) if optimizer is not None else None
    if hooked is None or not hooked.master_weights:
        return None
    state = hooked.state_dict()  # reconciles pending external writes *before* we write anything
    masters = state[_RR_KEY]["precision"]["masters"]
    names = {id(p): n for n, p in model.named_parameters()}
    entries: list[tuple[int, str, nn.Parameter]] = []
    for index, p in enumerate(_flat_params(hooked)):
        if index not in masters:
            continue
        name = names.get(id(p))
        if name is None:
            raise SnapshotError(
                f"optimizer parameter {index} has a master weight but is not a parameter of the model"
            )
        entries.append((index, name, p))  # type: ignore[arg-type]
    return _MasterPlan(hooked, state, entries)


def _union(selections: Iterable[ChannelSelection]) -> ChannelSelection:
    return reduce(lambda a, b: a.union(b), selections, ChannelSelection())


def _check_groups(groups: Sequence[ChannelGroup]) -> None:
    names = [g.name for g in groups]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"duplicate channel group names: {dup}")


def _reference_state(model: nn.Module, incumbent: StateSnapshot, selection: ChannelSelection) -> ChannelState:
    """The incumbent's slices of ``selection``: from ``channel_state`` if it covers them, else ``params``."""
    cs = incumbent.channel_state
    if cs is not None and all(
        layer in cs and set(chans) <= set(cs[layer].channels.tolist()) for layer, chans in selection.items()
    ):
        return ChannelState({layer: cs[layer].select(chans) for layer, chans in selection.items()})
    out: dict[int, LayerChannels] = {}
    for ref in find_swiglu_ffns(model, selection.layers()):
        layer = ref.layer_index
        idx = selection.index_tensor(layer)

        def take(
            proj: str,
            kind: str,
            dim: int,
            idx: torch.Tensor = idx,
            path: str = ref.module_path,
            layer: int = layer,
        ) -> torch.Tensor:
            name = f"{path}.{proj}.{kind}"
            if name not in incumbent.params:
                raise SnapshotError(
                    f"incumbent has neither a channel_state covering layer {layer} nor parameter {name!r}"
                )
            return incumbent.params[name].index_select(dim, idx).clone()

        out[layer] = LayerChannels(
            channels=idx.clone(),
            gate=take("gate_proj", "weight", 0),
            up=take("up_proj", "weight", 0),
            down=take("down_proj", "weight", 1),
            gate_bias=None if ref.gate.bias is None else take("gate_proj", "bias", 0),
            up_bias=None if ref.up.bias is None else take("up_proj", "bias", 0),
            module_path=ref.module_path,
        )
    return ChannelState(out)


def _check_reference(model: nn.Module, reference: ChannelState) -> None:
    """Everything :func:`rollback_channels` checks, for all layers at once (cross-group atomicity)."""
    for ref in find_swiglu_ffns(model, list(reference.layers)):
        layer = ref.layer_index
        lc = reference[layer]
        if lc.module_path is not None and lc.module_path != ref.module_path:
            raise SnapshotError(
                f"layer {layer}: incumbent slices come from {lc.module_path!r}, model has {ref.module_path!r}"
            )
        k = int(lc.channels.numel())
        pairs: list[tuple[str, torch.Tensor | None, torch.Tensor | None, tuple[int, ...]]] = [
            ("gate_proj.weight", ref.gate.weight, lc.gate, (k, ref.gate.weight.shape[1])),
            ("up_proj.weight", ref.up.weight, lc.up, (k, ref.up.weight.shape[1])),
            ("down_proj.weight", ref.down.weight, lc.down, (ref.down.weight.shape[0], k)),
            ("gate_proj.bias", ref.gate.bias, lc.gate_bias, (k,)),
            ("up_proj.bias", ref.up.bias, lc.up_bias, (k,)),
        ]
        for what, p, v, shape in pairs:
            if (p is None) != (v is None):
                raise SnapshotError(f"layer {layer}: {what} presence differs between model and incumbent")
            if p is None or v is None:
                continue
            if tuple(v.shape) != shape or v.dtype != p.dtype:
                raise SnapshotError(
                    f"layer {layer} {what}: incumbent slice {tuple(v.shape)}/{v.dtype}, expected {shape}/{p.dtype}"
                )


# ============================================================================== capture / restore
@torch.no_grad()
def capture(
    model: nn.Module,
    names: Iterable[str] | None,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    channel_layers: Iterable[int] | None = None,
) -> StateSnapshot:
    """Snapshot the parameters ``names`` (``None`` = every parameter), the optimizer and masters.

    * Unknown names raise :class:`SnapshotError`; duplicates are ignored; the stored order is the
      model's ``named_parameters`` order.
    * ``optimizer``: ``optimizer.state_dict()`` is deep-copied (tensors cloned to CPU). For a
      ``HookedOptimizer(master_weights=True)`` (also inside wrappers, via ``unwrap_hooked``) the
      fp32 masters of the captured parameters are stored by name in ``masters``. The optimizer may
      hold parameters outside ``names`` (see :func:`restore` for what happens to them).
    * ``channel_layers``: if given, ``channel_state(model, channel_layers)`` is stored as well.

    The model and the optimizer are not modified (apart from the master reconciliation that
    ``HookedOptimizer.state_dict()`` itself performs).
    """
    named = dict(model.named_parameters())
    wanted = list(named) if names is None else list(dict.fromkeys(names))
    unknown = [n for n in wanted if n not in named]
    if unknown:
        raise SnapshotError(f"capture: {len(unknown)} unknown parameter name(s): {unknown[:5]}")
    keep = set(wanted)
    optimizer_state: dict[str, Any] | None = None
    masters: dict[str, torch.Tensor] | None = None
    if optimizer is not None:
        memo: dict[int, torch.Tensor] = {}
        optimizer_state = _clone_tree(optimizer.state_dict(), memo)
        hooked = unwrap_hooked(optimizer)
        if hooked is not None and hooked.master_weights:
            id_to_name = {id(p): n for n, p in named.items()}
            masters = {}
            for p in _flat_params(hooked):
                m = hooked.master_of(p)
                name = id_to_name.get(id(p))
                if m is not None and name in keep:
                    masters[name] = _clone_tensor(m, memo)  # same object as in optimizer_state
    params = snapshot(model, wanted)
    params = {n: params[n] for n in named if n in keep}
    cs = channel_state(model, channel_layers) if channel_layers is not None else None
    return StateSnapshot(
        params=params,
        optimizer_state=optimizer_state,
        masters=masters,
        channel_state=cs,
        fingerprint=fingerprint(params),
    )


@torch.no_grad()
def restore(
    model: nn.Module,
    snapshot: StateSnapshot,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    restore_optimizer: bool = True,
) -> None:
    """Write the snapshot back: parameters, optimizer state and masters bit-identical to it.

    Order: validate everything, write the parameters in place (``copy_``, so aliases and tied
    weights stay valid), then load the optimizer state. Validation (all before the first write;
    :class:`SnapshotError`): every snapshot name exists in the model with the same shape and
    dtype; with ``optimizer`` and ``restore_optimizer`` the snapshot has optimizer state, the same
    number and sizes of parameter groups, the same kind of optimizer (``HookedOptimizer`` or
    not), and every tensor of its per-parameter state (``exp_avg``, ``exp_avg_sq``, ... - not
    ``step``) fits the parameter it is mapped to (same shape as the optimizer's current entry, or,
    without one, the parameter's sizes or 1 per dimension for entries of the parameter's rank; see
    ``_check_optimizer_state_shapes``); snapshot masters belong to master-backed parameters of the
    optimizer with matching shapes. Optimizer state is mapped positionally (as torch does): an
    optimizer over the same parameters in another order is caught only where shapes differ.

    * ``optimizer=None``: parameters only.
    * ``restore_optimizer=True``: ``optimizer.load_state_dict`` of a fresh clone of the snapshot's
      state (training after the restore never mutates the snapshot), so the moments, step counts,
      ``param_groups`` hyper-parameters and a ``HookedOptimizer``'s counters / rule state are
      exactly those at capture. The state is global: parameters outside the snapshot's names get
      the snapshot's moments but keep their current weights and masters.
    * ``restore_optimizer=False``: the optimizer state is not restored; only the masters of a
      ``HookedOptimizer(master_weights=True)`` are made consistent with the restored weights.

    Masters (``HookedOptimizer(master_weights=True)``): a restored parameter's master becomes the
    snapshot's master; if the snapshot has none for it (captured without that optimizer) it
    becomes ``snapshot_param.float()`` (the sub-ulp information is then lost, by necessity).
    Not restored: ``.grad``, LR-scheduler state, RNG state.
    """
    targets = _check_params(model, snapshot.params, "restore")
    if snapshot.masters is not None:
        extra = [n for n in snapshot.masters if n not in snapshot.params]
        if extra:
            raise SnapshotError(f"restore: snapshot masters without parameters: {extra[:5]}")
    if optimizer is not None and restore_optimizer:
        if snapshot.optimizer_state is None:
            raise SnapshotError("restore: the snapshot holds no optimizer state (captured without optimizer)")
        _check_optimizer_structure(optimizer, snapshot.optimizer_state)
    plan = _master_plan(optimizer, model)
    new_masters: dict[int, torch.Tensor] = {}
    if plan is not None:
        master_names = {name for _, name, _ in plan.entries}
        if snapshot.masters is not None:
            stray = [n for n in snapshot.masters if n not in master_names]
            if stray:
                raise SnapshotError(f"restore: snapshot masters for parameters without a master: {stray[:5]}")
        for index, name, p in plan.entries:
            if snapshot.masters is not None and name in snapshot.masters:
                value = snapshot.masters[name]
                if tuple(value.shape) != tuple(p.shape):
                    raise SnapshotError(
                        f"restore: master {name!r} has shape {tuple(value.shape)}, param {tuple(p.shape)}"
                    )
                new_masters[index] = value  # HookedOptimizer copies it (copy=True)
            elif name in snapshot.params:
                new_masters[index] = snapshot.params[name].float()
            else:
                new_masters[index] = plan.current(index)  # outside the snapshot: keep
    # ---- writes (validated above; only a HookedOptimizer rule's load_state_dict can still raise)
    for name, p in targets.items():
        p.copy_(snapshot.params[name])
    if optimizer is None:
        return
    if plan is not None:
        base = _clone_tree(snapshot.optimizer_state) if restore_optimizer else plan.state
        plan.load(base, new_masters)
    elif restore_optimizer:
        optimizer.load_state_dict(_clone_tree(snapshot.optimizer_state))


@contextmanager
def swapped(model: nn.Module, snapshot: StateSnapshot) -> Iterator[StateSnapshot]:
    """Temporarily load the snapshot's parameters (e.g. to evaluate the incumbent).

    On entry the current values of the snapshot's parameters are cloned (on their device) and the
    snapshot is written in place; on exit - normal or exception - the clones are written back, so
    the parameters are bit-identical to their values before the ``with``. Validation (names,
    shapes, dtypes) happens before anything is written. Optimizer state and masters are never
    touched: a ``HookedOptimizer`` sees the swap only as an external write whose bits were undone,
    which it ignores element-wise. Do not call ``optimizer.step()`` or ``optimizer.state_dict()``
    inside the block (a master-weight optimizer would reconcile the swapped values into its
    masters). ``.grad``, training mode and buffers are not touched. Memory: one extra copy of the
    swapped parameters on their device. Yields the snapshot.
    """
    targets = _check_params(model, snapshot.params, "swapped")
    with torch.no_grad():
        saved = {n: p.detach().clone() for n, p in targets.items()}
    try:
        with torch.no_grad():
            for n, p in targets.items():
                p.copy_(snapshot.params[n])
        yield snapshot
    finally:
        with torch.no_grad():
            for n, p in targets.items():
                p.copy_(saved[n])


@torch.no_grad()
def interpolate(
    model: nn.Module,
    incumbent: StateSnapshot,
    candidate: StateSnapshot,
    lam: float,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Write ``incumbent + lam * (candidate - incumbent)`` into the snapshots' parameters.

    ``lam`` must be a finite number in ``[0, 1]`` (0 = incumbent, 1 = candidate). Both snapshots
    must hold the same parameter names; names, shapes and dtypes are validated against the model
    before anything is written (:class:`SnapshotError`). For ``0 < lam < 1`` the value is computed
    on CPU with ``torch.lerp`` in float32 (float64 for float64 parameters) and cast
    (round-to-nearest) to the parameter dtype; non-finite elements in either endpoint then give
    non-finite results (the formula: ``0 * inf`` / ``inf - inf`` are NaN). ``lam == 0`` and
    ``lam == 1`` are exact copies of the interpolated values (no arithmetic), even if the *other*
    endpoint holds NaN/inf (e.g. a diverged candidate). Without masters (see below) the
    interpolated values are the snapshots' parameters, so ``lam=0`` / ``lam=1`` give the
    incumbent / candidate parameters bit-exactly for every dtype; bf16/fp16 interior values are
    rounded to the nearest representable value.

    Masters (``HookedOptimizer(master_weights=True)``): for a master-backed parameter whose master
    is present in *both* snapshots, the fp32 masters are interpolated instead (at the endpoints the
    master becomes the snapshot's master bit-exactly) and the served weight is
    ``round_to_nearest(new master)`` (the masters are the more precise state). Under
    ``write_rounding="nearest"`` the endpoints then equal the snapshots' served weights; under
    ``"stochastic"`` the snapshot's served weight was a stochastic rounding of its master and may
    differ from ``round_to_nearest(master)`` by one ulp per element. Otherwise the interpolated
    fp32 value of the served weights becomes the master. Masters of parameters outside the
    snapshots are unchanged. The optimizer's moments are **not** touched (they stay the
    candidate's); restore them yourself if the procedure needs it.
    """
    if (
        isinstance(lam, bool)
        or not isinstance(lam, (int, float))
        or not math.isfinite(lam)
        or not 0.0 <= lam <= 1.0
    ):
        raise ValueError(f"lam must be a finite number in [0, 1], got {lam!r}")
    if set(incumbent.params) != set(candidate.params):
        diff = sorted(set(incumbent.params) ^ set(candidate.params))
        raise SnapshotError(f"interpolate: incumbent and candidate hold different parameters: {diff[:5]}")
    targets = _check_params(model, incumbent.params, "interpolate (incumbent)")
    _check_params(model, candidate.params, "interpolate (candidate)")
    plan = _master_plan(optimizer, model)
    master_index = {name: index for index, name, _ in plan.entries} if plan is not None else {}
    new_masters: dict[int, torch.Tensor] = {}
    weight = float(lam)
    for name, p in targets.items():
        dt = torch.float64 if p.dtype == torch.float64 else torch.float32
        index = master_index.get(name)
        if (
            index is not None
            and incumbent.masters is not None
            and candidate.masters is not None
            and name in incumbent.masters
            and name in candidate.masters
        ):
            a, b = incumbent.masters[name], candidate.masters[name]
        else:
            a, b = incumbent.params[name], candidate.params[name]
        if weight == 0.0:  # exact copies at the endpoints, also when the other side is non-finite
            value = a.to(dt)
        elif weight == 1.0:
            value = b.to(dt)
        else:
            value = torch.lerp(a.to(dt), b.to(dt), weight)
        p.copy_(value.to(p.dtype))
        if index is not None:
            new_masters[index] = value.to(torch.float32)
    if plan is not None:
        plan.load(plan.state, new_masters)


# ============================================================================== channel groups
def group_damage(
    incumbent: StateSnapshot,
    candidate_model: nn.Module,
    activation_scores: Mapping[int, torch.Tensor],
    groups: Sequence[ChannelGroup],
) -> dict[str, float]:
    """Damage of each group: ``sum_{(layer, c) in group} displacement[layer, c] * activation[layer][c]``.

    ``displacement`` is the L2 norm of the change of the channel's slices (gate row, up row, down
    column, gate/up bias entries) from the incumbent to ``candidate_model``, computed in float64 by
    :func:`~research_rl.models.ffn_channels.channel_diff`. The incumbent's slices come from its
    ``channel_state`` when it covers the groups, else from its ``params`` (:class:`SnapshotError`
    if neither does). ``activation_scores[layer]`` is a 1-D tensor of length ``intermediate_size``
    (e.g. mean |h| or sqrt(mean h^2) of the channel on the failing slice); every value must be
    finite and >= 0 (``ValueError`` otherwise, also for a missing layer). A channel in several
    groups counts in each. A NaN displacement (a weight became NaN) gives a NaN damage. The
    incumbent's ``params`` and ``masters`` must exist in ``candidate_model`` with its full shapes
    (and ``params`` its dtypes), else :class:`SnapshotError` (an incumbent of another model);
    an incumbent holding only a ``channel_state`` is checked by slice shapes only.
    Pure: the model is not modified.
    """
    _check_groups(groups)
    union = _union(g.selection for g in groups)
    if not union:
        return {g.name: 0.0 for g in groups}
    union.validate(candidate_model)
    _check_incumbent(candidate_model, incumbent, "group_damage")
    sizes = {r.layer_index: r.intermediate_size for r in find_swiglu_ffns(candidate_model, union.layers())}
    acts: dict[int, list[float]] = {}
    for layer in union.layers():
        if layer not in activation_scores:
            raise ValueError(f"activation_scores has no entry for layer {layer}")
        s = activation_scores[layer].detach()
        if s.dim() != 1 or s.numel() != sizes[layer]:
            raise ValueError(
                f"activation_scores[{layer}] must be 1-D of length {sizes[layer]}, got shape {tuple(s.shape)}"
            )
        vals = s.double().cpu()
        if not bool(torch.isfinite(vals).all()) or bool((vals < 0).any()):
            raise ValueError(f"activation_scores[{layer}] must be finite and >= 0")
        acts[layer] = vals.tolist()
    reference = _reference_state(candidate_model, incumbent, union)
    diff = channel_diff(reference, candidate_model)
    per_channel: dict[tuple[int, int], float] = {}
    for layer, d in diff.items():
        for c, disp in zip(d.channels.tolist(), d.l2.tolist()):
            per_channel[(layer, int(c))] = float(disp) * acts[layer][int(c)]
    return {
        g.name: math.fsum(per_channel[(lay, c)] for lay, chans in g.selection.items() for c in chans)
        for g in groups
    }


def channel_groups_by_damage(
    incumbent: StateSnapshot,
    candidate_model: nn.Module,
    activation_scores: Mapping[int, torch.Tensor],
    groups: Sequence[ChannelGroup],
) -> list[ChannelGroup]:
    """``groups`` ordered by :func:`group_damage`, largest first.

    Ties keep the input order (stable sort); NaN damages come first (a NaN weight is treated as
    maximal damage, like :func:`~research_rl.models.ffn_channels.ranked_channels`). Group names
    must be unique (``ValueError``).
    """
    groups = list(groups)
    damage = group_damage(incumbent, candidate_model, activation_scores, groups)

    def key(i: int) -> tuple[int, float, int]:
        d = damage[groups[i].name]
        return (0, 0.0, i) if math.isnan(d) else (1, -d, i)

    return [groups[i] for i in sorted(range(len(groups)), key=key)]


@torch.no_grad()
def revert_groups(
    model: nn.Module,
    incumbent: StateSnapshot,
    groups: Sequence[ChannelGroup],
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> list[RollbackReport]:
    """Revert the channels of ``groups`` to the incumbent; returns one ``RollbackReport`` per group.

    Each group is reverted with :func:`~research_rl.models.ffn_channels.rollback_channels` (only
    the slices owned by its channels are written; every other element keeps its bits). The
    incumbent's slices come from its ``channel_state`` if it covers the groups, else from its
    ``params``. All checks (selection validity; the incumbent's ``params`` and ``masters`` exist in
    the model with its full shapes, ``params`` also with its dtypes; coverage, module paths, slice
    shapes, dtypes, bias presence) run for all groups before the first write, so a failing call
    changes nothing. An incumbent that holds only a ``channel_state`` (no ``params``) can be
    checked by slice shapes only: one from another model with the same hidden size is not
    detected. A
    channel in several groups is written by each (later reports then show 0 changed elements).

    Masters (``HookedOptimizer(master_weights=True)``): on the reverted elements the master becomes
    the incumbent's master (or, if the incumbent has none for that parameter, the reverted served
    value as fp32); every other element keeps its master. The served weights outside the groups are
    not rewritten. The optimizer's moments are not touched; other optimizers are not touched at all.
    """
    groups = list(groups)
    _check_groups(groups)
    if not groups:
        return []
    for g in groups:
        g.selection.validate(model)
    _check_incumbent(model, incumbent, "revert_groups")
    union = _union(g.selection for g in groups)
    if not union:
        return [rollback_channels(model, ChannelState({}), g.selection) for g in groups]
    reference = _reference_state(model, incumbent, union)
    _check_reference(model, reference)
    plan = _master_plan(optimizer, model)
    reports = [rollback_channels(model, reference, g.selection) for g in groups]
    if plan is None:
        return reports
    mask = channel_mask(model, union)
    new_masters: dict[int, torch.Tensor] = {}
    for index, name, p in plan.entries:
        owned = mask.get(name)
        if owned is None:
            continue
        current = plan.current(index)
        if incumbent.masters is not None and name in incumbent.masters:
            source = incumbent.masters[name].to(device=current.device, dtype=torch.float32)
        else:
            source = p.detach().to(device=current.device, dtype=torch.float32)
        new_masters[index] = torch.where(owned.to(current.device), source, current)
    plan.load(plan.state, new_masters)
    return reports
