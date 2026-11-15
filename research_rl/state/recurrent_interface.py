"""Per-sample recurrent state contract (memory carried across steps / turns / chunks).

Contract (:class:`RecurrentState`): a state is a ``dict[str, Tensor]`` whose tensors all have the
batch dimension first. Implementations must be **row-wise**: row i of ``update(state, inputs)``
depends only on row i of ``state`` and ``inputs`` (no batch statistics, no cross-row mixing).

* ``init(batch_size, device, dtype)`` – fresh state; ``dtype`` is the *requested* state dtype, an
  implementation may store individual keys at a different (e.g. higher) precision, so the tensors of
  one state can have different dtypes per key,
* ``read(state)`` – the readout tensor ``[B, ...]`` (one dtype),
* ``update(state, inputs, active=None)`` – functional update (rows with ``active=False`` keep
  their state),
* ``reset(state, row_mask)`` – re-initialise only the rows where ``row_mask`` is True,
* ``state_dict()`` / ``load_state_dict()`` – hyper-parameters (tensors/primitives only).

:class:`StateStore` keeps one state per *sample id* and is the only sanctioned way to carry
state across batches: unknown ids get a fresh state, batches are gathered/scattered by id, a
scatter must use exactly the ids of the preceding gather (otherwise it raises), and a single
row is never broadcast to several ids. It implements the checkpoint ``StatefulComponent``
protocol (:mod:`research_rl.core.checkpoint`) so it can be passed as a trainer component.
Stored rows and checkpoints keep each key's own dtype (mixed per-key dtypes round-trip through
``torch.save`` / ``torch.load(weights_only=True)`` unchanged).

Precision: :class:`FastSlowMemory` computes every update in at least fp32 and by default stores its
slow (low-rate) state in at least fp32 even for a bf16/fp16 state, because a slow EMA increment
``η_s · (f − s)`` is easily below half an ulp of a half-precision ``s`` and would be rounded away.

Reference implementation: :class:`FastSlowMemory`. What is NOT provided: any learned memory
architecture or integration into a model's forward pass; those are research code on top.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import torch

from research_rl.core.registry import RECURRENT_STATES

State = dict[str, torch.Tensor]


def _expand_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return mask.to(device=like.device, dtype=torch.bool).reshape(-1, *([1] * (like.dim() - 1)))


class RecurrentState(ABC):
    """Abstract per-sample recurrent state (see module docstring for the contract)."""

    name: str = "recurrent_state"

    @abstractmethod
    def init(
        self, batch_size: int, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
    ) -> State: ...

    @abstractmethod
    def read(self, state: State) -> torch.Tensor: ...

    @abstractmethod
    def update(self, state: State, inputs: torch.Tensor, active: torch.Tensor | None = None) -> State: ...

    def reset(self, state: State, row_mask: torch.Tensor) -> State:
        """Re-initialise rows where ``row_mask`` is True; other rows are returned unchanged.

        Every key keeps its incoming dtype (fresh values are cast to it), so mixed per-key dtypes survive.
        """
        b = batch_size_of(state)
        if row_mask.shape != (b,):
            raise ValueError(f"row_mask must have shape ({b},), got {tuple(row_mask.shape)}")
        any_t = next(iter(state.values()))
        fresh = self._fresh(b, any_t.device, self.state_dtype(state))
        return {k: torch.where(_expand_mask(row_mask, v), fresh[k].to(v.dtype), v) for k, v in state.items()}

    def state_dtype(self, state: State) -> torch.dtype:
        """The dtype that was requested from :meth:`init` for ``state`` (used by :meth:`reset`).

        Default: the dtype of the first floating-point tensor. Implementations that store keys at
        different precisions override this.
        """
        return _float_dtype(state)

    def _fresh(self, batch_size: int, device: torch.device | str | None, dtype: torch.dtype) -> State:
        """Fresh state for internal use (:meth:`reset`); implementations may skip init-time warnings here."""
        return self.init(batch_size, device, dtype)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        return None


def batch_size_of(state: State) -> int:
    sizes = {int(v.shape[0]) for v in state.values()}
    if len(sizes) != 1:
        raise ValueError(f"state tensors disagree on the batch size: {sorted(sizes)}")
    return sizes.pop()


def _float_dtype(state: State) -> torch.dtype:
    for v in state.values():
        if v.is_floating_point():
            return v.dtype
    return torch.float32


@RECURRENT_STATES.register("fast_slow_memory")
class FastSlowMemory(RecurrentState):
    """Two-timescale memory. Per row, with input x_t ∈ R^d and step counter n::

        f_t = (1 − η_f) · f_{t−1} + η_f · x_t                       (fast: every step, high rate η_f)
        n_t = n_{t−1} + 1
        s_t = (1 − η_s) · s_{t−1} + η_s · f_t   if n_t mod k == 0    (slow: consolidated from fast
        s_t = s_{t−1}                            otherwise              every k steps, low rate η_s)

    ``read(state) = concat(f, s)`` of shape ``[B, 2d]``. Initial state: f = s = 0, n = 0. Step counters
    are per row, so rows reset at different times keep independent consolidation schedules.

    Precision (``init(..., dtype)`` is the *requested* state dtype):

    * ``fast`` is stored in ``dtype``; ``slow`` is stored in ``promote_types(dtype, slow_dtype)``
      (default ``slow_dtype=torch.float32``: at least fp32, i.e. fp32 for bf16/fp16/fp32 and float64
      for float64); ``slow_dtype=None`` stores ``slow`` in ``dtype`` (the pre-fix behaviour);
      ``steps`` is ``torch.long``.
    * :meth:`update` computes both EMAs in ``promote_types(promote_types(fast, slow), float32)``
      (inputs are cast straight to that dtype) and feeds the *unrounded* ``f_t`` into the slow EMA;
      it then rounds ``f_t`` to the fast dtype and ``s_t`` to ``promote_types(incoming slow dtype,
      slow storage dtype for the fast dtype)``, so a hand-built or pre-fix all-bf16 state gets fp32
      slow storage on its first update. For an fp32 state all casts are no-ops and the results are
      bit-identical to the original fp32 implementation.
    * :meth:`read` returns ``promote_types(fast dtype, slow dtype)`` (fp32 for a bf16 state with the
      default ``slow_dtype``; ``dtype`` itself whenever both keys share it).
    * With half-precision slow storage (``slow_dtype=None`` and a bf16/fp16 state), :meth:`init` emits
      a ``UserWarning`` when ``slow_rate`` or ``fast_rate`` is positive but below that dtype's relative
      precision ``finfo(dtype).eps / 2`` (2**-8 for bf16, 2**-11 for fp16): the EMA increment is then
      below half an ulp of the state and is rounded away, so the state never moves. The fast state is
      always stored in ``dtype``, so a ``fast_rate`` that small also stalls ``fast`` under the default
      ``slow_dtype`` (not warned about: fast rates are large by design).
    """

    name = "fast_slow_memory"

    def __init__(
        self,
        dim: int,
        fast_rate: float = 0.5,
        slow_rate: float = 0.05,
        consolidate_every: int = 4,
        slow_dtype: torch.dtype | str | None = torch.float32,
    ) -> None:
        """``slow_dtype``: minimum storage precision of the slow state, a floating ``torch.dtype`` or its
        name (e.g. ``"float32"``); ``None`` stores it in the requested state dtype. See the class docs."""
        if not (0.0 < fast_rate <= 1.0 and 0.0 <= slow_rate <= 1.0):
            raise ValueError("rates must lie in (0, 1] (fast) and [0, 1] (slow)")
        if consolidate_every < 1:
            raise ValueError("consolidate_every must be >= 1")
        self.dim = dim
        self.fast_rate = float(fast_rate)
        self.slow_rate = float(slow_rate)
        self.consolidate_every = int(consolidate_every)
        self.slow_dtype = _resolve_dtype(slow_dtype)

    def slow_storage_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """Dtype in which the slow state is stored for a requested state ``dtype``."""
        return dtype if self.slow_dtype is None else torch.promote_types(dtype, self.slow_dtype)

    def _fresh(self, batch_size: int, device: torch.device | str | None, dtype: torch.dtype) -> State:
        return {
            "fast": torch.zeros(batch_size, self.dim, device=device, dtype=dtype),
            "slow": torch.zeros(batch_size, self.dim, device=device, dtype=self.slow_storage_dtype(dtype)),
            "steps": torch.zeros(batch_size, dtype=torch.long, device=device),
        }

    def init(
        self, batch_size: int, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
    ) -> State:
        """Fresh state: ``fast`` in ``dtype``, ``slow`` in :meth:`slow_storage_dtype`, ``steps`` long."""
        self._warn_if_rounded_away(self.slow_storage_dtype(dtype))
        return self._fresh(batch_size, device, dtype)

    def _warn_if_rounded_away(self, slow_storage: torch.dtype) -> None:
        if slow_storage not in (torch.bfloat16, torch.float16):
            return
        precision = torch.finfo(slow_storage).eps / 2  # unit roundoff: 2**-8 (bf16), 2**-11 (fp16)
        small = [
            f"{name}={rate:g}"
            for name, rate in (("slow_rate", self.slow_rate), ("fast_rate", self.fast_rate))
            if 0.0 < rate < precision
        ]
        if small:
            warnings.warn(
                f"FastSlowMemory stores its slow state in {slow_storage} and {', '.join(small)} is below "
                f"that dtype's relative precision ({precision:g}): the EMA increment rate * (target - state) "
                "is smaller than half an ulp of the state and is rounded away, so the state stops moving. "
                "Use slow_dtype=torch.float32 (the default) or a larger rate.",
                UserWarning,
                stacklevel=3,
            )

    def state_dtype(self, state: State) -> torch.dtype:
        return state["fast"].dtype

    def read(self, state: State) -> torch.Tensor:
        """``concat(fast, slow)`` in ``promote_types(fast dtype, slow dtype)``."""
        f, s = state["fast"], state["slow"]
        out = torch.promote_types(f.dtype, s.dtype)
        return torch.cat([f.to(out), s.to(out)], dim=-1)

    def update(self, state: State, inputs: torch.Tensor, active: torch.Tensor | None = None) -> State:
        f, s, n = state["fast"], state["slow"], state["steps"]
        if inputs.shape != f.shape:
            raise ValueError(f"inputs must have shape {tuple(f.shape)}, got {tuple(inputs.shape)}")
        compute = torch.promote_types(torch.promote_types(f.dtype, s.dtype), torch.float32)
        slow_out = torch.promote_types(s.dtype, self.slow_storage_dtype(f.dtype))
        fc, sc, x = f.to(compute), s.to(compute), inputs.to(compute)
        f_new = (1.0 - self.fast_rate) * fc + self.fast_rate * x
        n_new = n + 1
        consolidate = (n_new % self.consolidate_every == 0).unsqueeze(-1)
        s_new = torch.where(consolidate, (1.0 - self.slow_rate) * sc + self.slow_rate * f_new, sc)
        new = {"fast": f_new.to(f.dtype), "slow": s_new.to(slow_out), "steps": n_new}
        if active is not None:
            new = {k: torch.where(_expand_mask(active, v), v, state[k].to(v.dtype)) for k, v in new.items()}
        return new

    def state_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "fast_rate": self.fast_rate,
            "slow_rate": self.slow_rate,
            "consolidate_every": self.consolidate_every,
            "slow_dtype": None if self.slow_dtype is None else str(self.slow_dtype).removeprefix("torch."),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("dim", self.dim)) != self.dim:
            raise ValueError(f"checkpoint memory dim {state.get('dim')} != {self.dim}")
        self.fast_rate = float(state.get("fast_rate", self.fast_rate))
        self.slow_rate = float(state.get("slow_rate", self.slow_rate))
        self.consolidate_every = int(state.get("consolidate_every", self.consolidate_every))
        if "slow_dtype" in state:  # absent in older checkpoints: keep the constructor's value
            self.slow_dtype = _resolve_dtype(state["slow_dtype"])


def _resolve_dtype(dtype: torch.dtype | str | None) -> torch.dtype | None:
    if dtype is None:
        return None
    resolved = getattr(torch, dtype.removeprefix("torch."), None) if isinstance(dtype, str) else dtype
    if not isinstance(resolved, torch.dtype) or not resolved.is_floating_point:
        raise ValueError(f"slow_dtype must be None or a floating-point torch dtype, got {dtype!r}")
    return resolved


def _encode_id(sample_id: Hashable) -> str:
    if isinstance(sample_id, bool) or not isinstance(sample_id, int | str):
        raise TypeError(f"sample ids must be int or str, got {type(sample_id).__name__}")
    return f"int:{sample_id}" if isinstance(sample_id, int) else f"str:{sample_id}"


def _decode_id(key: str) -> int | str:
    kind, _, value = key.partition(":")
    return int(value) if kind == "int" else value


class StateStore:
    """Per-sample-id storage of :class:`RecurrentState` rows (no implicit cross-sample sharing).

    Usage per batch: ``state = store.gather(ids)`` → model code updates ``state`` →
    ``store.scatter(ids, state)``; or simply ``store.step(ids, inputs)``.

    ``dtype`` is the requested state dtype passed to ``module.init``; the module may store keys at other
    precisions (e.g. :class:`FastSlowMemory` keeps ``slow`` in fp32 for a bf16 store). ``gather`` casts
    each key to the dtype of a fresh row; ``scatter``, ``state_dict`` and ``load_state_dict`` keep each
    stored key's dtype unchanged.
    """

    def __init__(
        self,
        module: RecurrentState,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.module = module
        self.device = device
        self.dtype = dtype
        self._states: dict[str, State] = {}
        self._checked_out: tuple[str, ...] | None = None

    def __contains__(self, sample_id: object) -> bool:
        try:
            return _encode_id(sample_id) in self._states  # type: ignore[arg-type]
        except TypeError:
            return False

    def __len__(self) -> int:
        return len(self._states)

    def ids(self) -> list[int | str]:
        return [_decode_id(k) for k in self._states]

    def _keys(self, ids: Sequence[Hashable]) -> tuple[str, ...]:
        keys = tuple(_encode_id(i) for i in ids)
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate sample ids in one batch would share a state row")
        return keys

    def gather(self, ids: Sequence[Hashable]) -> State:
        """Batch the states of ``ids`` (fresh state for unseen ids). Must be followed by ``scatter``.

        Unseen ids are *not* inserted into the store here; they are stored by :meth:`scatter`, so an
        aborted batch (:meth:`release`) leaves no trace in ``len()``, ``ids()`` or checkpoints.
        """
        if self._checked_out is not None:
            raise RuntimeError("a batch is already gathered; scatter() or release() it first")
        keys = self._keys(ids)
        if not keys:
            raise ValueError("empty batch")
        fresh = self.module.init(1, self.device, self.dtype)
        rows = [self._states.get(k, fresh) for k in keys]
        self._checked_out = keys
        # every key is batched in the module's per-key dtype (that of ``fresh``, e.g. fast bf16 / slow
        # fp32), so rows stored at another precision (older checkpoints) cannot change the batch layout
        return {name: torch.cat([r[name].to(t.dtype) for r in rows], dim=0) for name, t in fresh.items()}

    def scatter(self, ids: Sequence[Hashable], state: State) -> None:
        """Store row i of ``state`` under ``ids[i]``; ids must equal those of the last ``gather``."""
        keys = self._keys(ids)
        if self._checked_out is None:
            raise RuntimeError("scatter() without a preceding gather()")
        if keys != self._checked_out:
            raise ValueError(
                f"batch composition mismatch: gathered {len(self._checked_out)} ids "
                f"{list(self._checked_out)[:4]}..., scattering {list(keys)[:4]}..."
            )
        if batch_size_of(state) != len(keys):
            raise ValueError(
                f"state has batch size {batch_size_of(state)} for {len(keys)} ids (no broadcasting)"
            )
        for i, k in enumerate(keys):
            self._states[k] = {name: v[i : i + 1].detach().clone() for name, v in state.items()}
        self._checked_out = None

    def release(self) -> None:
        """Abandon the gathered batch: nothing is stored, and ids first seen in that ``gather`` stay unknown."""
        self._checked_out = None

    def step(self, ids: Sequence[Hashable], inputs: torch.Tensor) -> torch.Tensor:
        """gather → update → scatter; returns the readout after the update."""
        state = self.gather(ids)
        try:
            new = self.module.update(state, inputs)
        except Exception:
            self.release()
            raise
        self.scatter(ids, new)
        return self.module.read(new)

    def forget(self, ids: Sequence[Hashable]) -> None:
        """Drop the state of ``ids`` (they start fresh next time)."""
        for k in self._keys(ids):
            self._states.pop(k, None)

    def get(self, sample_id: Hashable) -> State:
        return {n: v.clone() for n, v in self._states[_encode_id(sample_id)].items()}

    # ------------------------------------------------------------------ StatefulComponent
    def state_dict(self) -> dict[str, Any]:
        return {
            "module": self.module.state_dict(),
            "states": {k: {n: v.cpu() for n, v in s.items()} for k, s in self._states.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.module.load_state_dict(state.get("module", {}))
        self._states = {
            k: {n: v.to(self.device) if self.device is not None else v.clone() for n, v in s.items()}
            for k, s in state.get("states", {}).items()
        }
        self._checked_out = None
