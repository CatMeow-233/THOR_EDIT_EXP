"""Flat coordinate systems over model parameters for forward-only operators (``CAPABILITY.md`` 1.7).

Each class here implements :class:`research_rl.capability.types.ParameterSpace`: it stores a
detached *base* copy of the parameter elements it covers and maps a flat vector ``v`` of length
``dim`` onto those elements **from the base**, never cumulatively:

* :class:`ChannelGainSpace` (P1): one multiplicative gain per selected FFN channel,
  ``W_down[:, c] = base[:, c] * (1 + v_c)``; with ``up=True`` the *same* gain also multiplies the
  channel's ``up_proj`` row (and its ``up_proj`` bias entry, if the layer has one).
* :class:`ChannelWeightSpace` (P2): every element owned by the selected channels (gate row, up row,
  down column and, when present, gate/up bias entries), ``W = base + v``.
* :class:`FullParameterSpace`: every element of the named parameters, ``W = base + v``.

Common semantics (all three classes):

* ``write(v)`` sets the covered elements to ``base (+) v`` and nothing else; ``write(None)``
  copies the stored base back, bit-identically. ``write(v); write(v)`` equals one ``write(v)``.
* ``rebase()`` copies the current values of the covered elements into the base (detached, same
  dtype and device as the parameters).
* Arithmetic runs in float32 (float64 for float64 parameters) and is cast to the parameter dtype
  with ``copy_``/``index_copy_`` under ``torch.no_grad()``; parameter dtype and device are kept.
  ``v`` may live on any device and have any floating dtype; it is moved/cast per write. A ``v``
  that is not 1-D of length ``dim`` (or not floating point) is rejected before anything is written.
* Elements outside the space are never written (see :meth:`mask`, checked in the tests with
  :func:`research_rl.models.ffn_channels.verify_modification_scope`).
* Parameters are looked up by qualified name on every write/rebase (``model.get_parameter``), so
  an in-place restore or a replaced ``Parameter`` object of the same shape/dtype is handled. A
  parameter whose full shape or dtype differs from construction time is rejected with
  :class:`SpaceError` by ``write(None)``, ``write(v)`` and ``rebase()`` alike, before anything is
  written (no silent broadcasting of the base into a resized tensor); ``rebase()`` is
  all-or-nothing (no base is replaced if any covered parameter fails that check). The
  FFN of each selected layer is resolved once at construction with
  :func:`~research_rl.models.model_interface.find_swiglu_ffns` (its default module path); the
  resolved names are fixed afterwards. Expert containers are not exercised by the tests.
* Memory: the base is a full copy of the covered elements on the parameters' device
  (:class:`FullParameterSpace` therefore doubles the memory of the named parameters).
* Not supported: sharded (FSDP / tensor-parallel) or quantised parameters.

Verified on CPU with tiny models only (``tests/research_rl/test_capability_spaces.py``); nothing
here has run on a GPU (NOT_TESTED).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from research_rl.models.ffn_channels import ChannelSelection
from research_rl.models.model_interface import find_swiglu_ffns
from research_rl.models.parameter_mask import ParameterMask


class SpaceError(ValueError):
    """Invalid space construction or an invalid vector passed to ``write``."""


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def _check_vector(vector: torch.Tensor, dim: int) -> torch.Tensor:
    if not isinstance(vector, torch.Tensor):
        raise SpaceError(f"vector must be a torch.Tensor or None, got {type(vector).__name__}")
    if not vector.dtype.is_floating_point:
        raise SpaceError(f"vector must be floating point, got {vector.dtype}")
    if vector.dim() != 1 or vector.numel() != dim:
        raise SpaceError(f"vector must be 1-D of length {dim}, got shape {tuple(vector.shape)}")
    return vector.detach()


def _check_float_param(name: str, p: torch.Tensor) -> None:
    if not p.dtype.is_floating_point:
        raise SpaceError(f"parameter {name!r} has non-floating dtype {p.dtype}")


@dataclass
class _Slice:
    """Elements of one parameter: rows (``dim=0``) or columns (``dim=1``) ``index``, or the whole
    tensor (``index is None``). ``base`` holds exactly those elements (same dtype/device);
    ``param_shape`` is the full parameter shape at construction (checked on every access)."""

    name: str
    dim: int
    index: torch.Tensor | None
    base: torch.Tensor
    param_shape: tuple[int, ...]

    def read(self, p: torch.Tensor) -> torch.Tensor:
        if self.index is None:
            return p.detach().clone()
        return p.detach().index_select(self.dim, self.index.to(p.device)).clone()

    def write(self, p: torch.Tensor, value: torch.Tensor) -> None:
        value = value.to(device=p.device)
        if self.index is None:
            p.copy_(value)
        else:
            p.index_copy_(self.dim, self.index.to(p.device), value)


class _SpaceBase(ABC):
    """Shared bookkeeping: slices, lookup by name, write(None), rebase, mask."""

    dim: int

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._slices: list[_Slice] = []

    # ------------------------------------------------------------------ helpers
    def _param(self, name: str) -> nn.Parameter:
        return self.model.get_parameter(name)

    def _target(self, s: _Slice) -> nn.Parameter:
        """The parameter of slice ``s``, checked to still have its construction shape and dtype."""
        p = self._param(s.name)
        if p.dtype != s.base.dtype:
            raise SpaceError(f"{s.name}: dtype changed {s.base.dtype} -> {p.dtype} since construction")
        if tuple(p.shape) != s.param_shape:
            raise SpaceError(
                f"{s.name}: shape changed {s.param_shape} -> {tuple(p.shape)} since construction"
            )
        return p

    def _add_slice(self, name: str, dim: int, index: torch.Tensor | None) -> _Slice:
        p = self._param(name)
        _check_float_param(name, p)
        idx = None if index is None else index.to(device=p.device, dtype=torch.long)
        s = _Slice(name=name, dim=dim, index=idx, base=torch.empty(0), param_shape=tuple(p.shape))
        s.base = s.read(p)
        self._slices.append(s)
        return s

    # ------------------------------------------------------------------ ParameterSpace
    def parameter_names(self) -> list[str]:
        """Sorted qualified names of every parameter with at least one element in the space."""
        return sorted({s.name for s in self._slices})

    @torch.no_grad()
    def rebase(self) -> None:
        """Make the current values of the covered elements the new base (detached copies).

        All-or-nothing: every covered parameter is validated and read before any base is replaced.
        """
        new = [s.read(self._target(s)) for s in self._slices]
        for s, b in zip(self._slices, new):
            s.base = b

    @torch.no_grad()
    def write(self, vector: torch.Tensor | None) -> None:
        """``None``: copy the base back (bit-identical); otherwise write ``base (+) vector``."""
        if vector is None:
            targets = [self._target(s) for s in self._slices]  # validate before the first write
            for s, p in zip(self._slices, targets):
                s.write(p, s.base)
            return
        v = _check_vector(vector, self.dim)
        for s in self._slices:  # validate before the first write
            self._target(s)
        self._write_vector(v)

    @abstractmethod
    def _write_vector(self, v: torch.Tensor) -> None:
        """Write ``base (+) v`` for a validated 1-D vector of length ``dim``."""

    def mask(self) -> ParameterMask:
        """Boolean mask (CPU) of exactly the elements :meth:`write` may change, keyed by name.

        Parameters of the model that are not listed are outside the space; use it as ``allowed``
        of :func:`~research_rl.models.ffn_channels.verify_modification_scope` (``unlisted="frozen"``).
        """
        out = ParameterMask()
        for s in self._slices:
            p = self._param(s.name)
            m = out.get(s.name)
            if m is None:
                m = torch.zeros(p.shape, dtype=torch.bool)
            else:
                m = m.clone()
            if s.index is None:
                m[...] = True
            elif s.dim == 0:
                m[s.index.cpu()] = True
            else:
                m[:, s.index.cpu()] = True
            out.set(s.name, m)
        return out


# ============================================================================== channel spaces
def _check_selection(model: nn.Module, selection: ChannelSelection) -> None:
    if not isinstance(selection, ChannelSelection):
        raise SpaceError(f"selection must be a ChannelSelection, got {type(selection).__name__}")
    if selection.count() == 0:
        raise SpaceError("selection is empty: a space needs at least one channel")
    selection.validate(model)


class ChannelGainSpace(_SpaceBase):
    """P1: one multiplicative gain per selected channel (``dim == selection.count()``).

    Coordinates are ordered layer-ascending, then channel-ascending (the order of
    ``selection.items()``). ``write(v)`` sets, for every selected channel ``c`` with gain
    ``g = v[pos(c)]``:

    * ``down_proj.weight[:, c] = base[:, c] * (1 + g)``;
    * with ``up=True`` additionally ``up_proj.weight[c, :] = base[c, :] * (1 + g)`` and, if the
      layer has an ``up_proj`` bias, ``up_proj.bias[c] = base[c] * (1 + g)``. The up activation
      ``u_c`` is then scaled by exactly ``1 + g``, so the channel's additive FFN contribution
      ``W_down[:, c] * act(gate_c) * u_c`` is scaled by ``(1 + g)^2`` (``(1 + g)`` with ``up=False``).

    ``v = 0`` reproduces the base values exactly (``x * 1`` is exact in float32/float64 and the
    cast back is exact), but :meth:`write` with ``None`` is the canonical restore.
    """

    def __init__(self, model: nn.Module, selection: ChannelSelection, *, up: bool = False) -> None:
        super().__init__(model)
        _check_selection(model, selection)
        self.selection = selection
        self.up = bool(up)
        self.dim = selection.count()
        # Per layer: (slice offset, count, down slice, up-weight slice | None, up-bias slice | None)
        self._layers: list[tuple[int, int, _Slice, _Slice | None, _Slice | None]] = []
        offset = 0
        for ref in find_swiglu_ffns(model, selection.layers()):
            idx = selection.index_tensor(ref.layer_index)
            down = self._add_slice(f"{ref.module_path}.down_proj.weight", 1, idx)
            up_w = up_b = None
            if self.up:
                up_w = self._add_slice(f"{ref.module_path}.up_proj.weight", 0, idx)
                if ref.up.bias is not None:
                    up_b = self._add_slice(f"{ref.module_path}.up_proj.bias", 0, idx)
            self._layers.append((offset, len(idx), down, up_w, up_b))
            offset += len(idx)

    def _write_vector(self, v: torch.Tensor) -> None:
        for offset, k, down, up_w, up_b in self._layers:
            seg = v[offset : offset + k]
            for s, shape in ((down, (1, k)), (up_w, (k, 1)), (up_b, (k,))):
                if s is None:
                    continue
                p = self._param(s.name)
                cd = _compute_dtype(p.dtype)
                factor = (1.0 + seg.to(device=p.device, dtype=cd)).reshape(shape)
                s.write(p, (s.base.to(cd) * factor).to(p.dtype))


class ChannelWeightSpace(_SpaceBase):
    """P2: every element owned by the selected channels, ``W = base + v``.

    Layout: layers ascending; within a layer of ``k`` selected channels a block of ``k * D``
    coordinates that reshapes (row-major) to ``[k, D]``, one row per channel (channel-ascending)
    with columns ``[gate row (hidden), up row (hidden), down column (hidden), gate bias (1, if
    present), up bias (1, if present)]`` - the layout of
    :meth:`research_rl.models.ffn_channels.LayerChannels.vectors`. ``down_proj.bias`` belongs to no
    channel and is never written. ``dim = sum_layers k * D``. The written region equals
    :func:`~research_rl.models.ffn_channels.channel_mask` of ``selection`` restricted to the
    selected channels (so ``verify_modification_scope(..., allowed=selection)`` applies).
    """

    def __init__(self, model: nn.Module, selection: ChannelSelection) -> None:
        super().__init__(model)
        _check_selection(model, selection)
        self.selection = selection
        # Per layer: (offset, k, [(slice, width, transpose)])
        self._layers: list[tuple[int, int, list[tuple[_Slice, int, bool]]]] = []
        offset = 0
        for ref in find_swiglu_ffns(model, selection.layers()):
            idx = selection.index_tensor(ref.layer_index)
            k = len(idx)
            hidden = int(ref.down.weight.shape[0])
            parts: list[tuple[_Slice, int, bool]] = [
                (
                    self._add_slice(f"{ref.module_path}.gate_proj.weight", 0, idx),
                    int(ref.gate.weight.shape[1]),
                    False,
                ),
                (
                    self._add_slice(f"{ref.module_path}.up_proj.weight", 0, idx),
                    int(ref.up.weight.shape[1]),
                    False,
                ),
                (self._add_slice(f"{ref.module_path}.down_proj.weight", 1, idx), hidden, True),
            ]
            for proj, lin in (("gate_proj", ref.gate), ("up_proj", ref.up)):
                if lin.bias is not None:
                    parts.append((self._add_slice(f"{ref.module_path}.{proj}.bias", 0, idx), 1, False))
            width = sum(w for _, w, _ in parts)
            self._layers.append((offset, k, parts))
            offset += k * width
        self.dim = offset

    def _write_vector(self, v: torch.Tensor) -> None:
        for offset, k, parts in self._layers:
            width = sum(w for _, w, _ in parts)
            block = v[offset : offset + k * width].reshape(k, width)
            col = 0
            for s, w, transpose in parts:
                p = self._param(s.name)
                cd = _compute_dtype(p.dtype)
                delta = block[:, col : col + w].to(device=p.device, dtype=cd)
                col += w
                if transpose:  # down column: base is [hidden, k]
                    delta = delta.t()
                elif s.base.dim() == 1:  # bias entry: base is [k]
                    delta = delta.reshape(k)
                s.write(p, (s.base.to(cd) + delta).to(p.dtype))


# ============================================================================== full parameters
class FullParameterSpace(_SpaceBase):
    """Every element of the named parameters, ``W = base + v``.

    Coordinates are the parameters in the order of ``names`` (as given), each flattened
    row-major (``tensor.reshape(-1)``). ``dim`` is the total element count. Unknown, duplicate or
    non-floating-point names and an empty ``names`` are rejected. Works on any ``nn.Module``.
    """

    def __init__(self, model: nn.Module, names: Iterable[str]) -> None:
        super().__init__(model)
        if isinstance(names, str):
            raise SpaceError("names must be an iterable of parameter names, not a single string")
        name_list = list(names)
        if not name_list:
            raise SpaceError("names is empty: a space needs at least one parameter")
        if len(set(name_list)) != len(name_list):
            dup = sorted({n for n in name_list if name_list.count(n) > 1})
            raise SpaceError(f"duplicate parameter names: {dup}")
        known = dict(model.named_parameters())
        unknown = [n for n in name_list if n not in known]
        if unknown:
            raise SpaceError(f"unknown parameter names: {unknown[:10]}")
        self._order: list[tuple[int, int, _Slice]] = []
        offset = 0
        for name in name_list:
            s = self._add_slice(name, 0, None)
            n = s.base.numel()
            self._order.append((offset, n, s))
            offset += n
        self.dim = offset

    def _write_vector(self, v: torch.Tensor) -> None:
        for offset, n, s in self._order:
            p = self._param(s.name)
            cd = _compute_dtype(p.dtype)
            delta = v[offset : offset + n].to(device=p.device, dtype=cd).reshape(s.base.shape)
            s.write(p, (s.base.to(cd) + delta).to(p.dtype))


__all__ = [
    "ChannelGainSpace",
    "ChannelWeightSpace",
    "FullParameterSpace",
    "SpaceError",
]
