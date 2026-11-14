"""Per-sample recurrent state contract (memory carried across steps / turns / chunks).

Contract (:class:`RecurrentState`): a state is a ``dict[str, Tensor]`` whose tensors all have the
batch dimension first. Implementations must be **row-wise**: row i of ``update(state, inputs)``
depends only on row i of ``state`` and ``inputs`` (no batch statistics, no cross-row mixing).

* ``init(batch_size, device, dtype)`` – fresh state,
* ``read(state)`` – the readout tensor ``[B, ...]``,
* ``update(state, inputs, active=None)`` – functional update (rows with ``active=False`` keep
  their state),
* ``reset(state, row_mask)`` – re-initialise only the rows where ``row_mask`` is True,
* ``state_dict()`` / ``load_state_dict()`` – hyper-parameters (tensors/primitives only).

:class:`StateStore` keeps one state per *sample id* and is the only sanctioned way to carry
state across batches: unknown ids get a fresh state, batches are gathered/scattered by id, a
scatter must use exactly the ids of the preceding gather (otherwise it raises), and a single
row is never broadcast to several ids. It implements the checkpoint ``StatefulComponent``
protocol (:mod:`research_rl.core.checkpoint`) so it can be passed as a trainer component.

Reference implementation: :class:`FastSlowMemory`. What is NOT provided: any learned memory
architecture or integration into a model's forward pass; those are research code on top.
"""

from __future__ import annotations

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
        """Re-initialise rows where ``row_mask`` is True; other rows are returned unchanged."""
        b = batch_size_of(state)
        if row_mask.shape != (b,):
            raise ValueError(f"row_mask must have shape ({b},), got {tuple(row_mask.shape)}")
        any_t = next(iter(state.values()))
        fresh = self.init(b, any_t.device, _float_dtype(state))
        return {k: torch.where(_expand_mask(row_mask, v), fresh[k].to(v.dtype), v) for k, v in state.items()}

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
    """

    name = "fast_slow_memory"

    def __init__(
        self, dim: int, fast_rate: float = 0.5, slow_rate: float = 0.05, consolidate_every: int = 4
    ) -> None:
        if not (0.0 < fast_rate <= 1.0 and 0.0 <= slow_rate <= 1.0):
            raise ValueError("rates must lie in (0, 1] (fast) and [0, 1] (slow)")
        if consolidate_every < 1:
            raise ValueError("consolidate_every must be >= 1")
        self.dim = dim
        self.fast_rate = float(fast_rate)
        self.slow_rate = float(slow_rate)
        self.consolidate_every = int(consolidate_every)

    def init(
        self, batch_size: int, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
    ) -> State:
        return {
            "fast": torch.zeros(batch_size, self.dim, device=device, dtype=dtype),
            "slow": torch.zeros(batch_size, self.dim, device=device, dtype=dtype),
            "steps": torch.zeros(batch_size, dtype=torch.long, device=device),
        }

    def read(self, state: State) -> torch.Tensor:
        return torch.cat([state["fast"], state["slow"]], dim=-1)

    def update(self, state: State, inputs: torch.Tensor, active: torch.Tensor | None = None) -> State:
        f, s, n = state["fast"], state["slow"], state["steps"]
        if inputs.shape != f.shape:
            raise ValueError(f"inputs must have shape {tuple(f.shape)}, got {tuple(inputs.shape)}")
        x = inputs.to(f.dtype)
        f_new = (1.0 - self.fast_rate) * f + self.fast_rate * x
        n_new = n + 1
        consolidate = (n_new % self.consolidate_every == 0).unsqueeze(-1)
        s_new = torch.where(consolidate, (1.0 - self.slow_rate) * s + self.slow_rate * f_new, s)
        new = {"fast": f_new, "slow": s_new, "steps": n_new}
        if active is not None:
            new = {k: torch.where(_expand_mask(active, v), v, state[k]) for k, v in new.items()}
        return new

    def state_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "fast_rate": self.fast_rate,
            "slow_rate": self.slow_rate,
            "consolidate_every": self.consolidate_every,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("dim", self.dim)) != self.dim:
            raise ValueError(f"checkpoint memory dim {state.get('dim')} != {self.dim}")
        self.fast_rate = float(state.get("fast_rate", self.fast_rate))
        self.slow_rate = float(state.get("slow_rate", self.slow_rate))
        self.consolidate_every = int(state.get("consolidate_every", self.consolidate_every))


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
        rows = [
            self._states[k] if k in self._states else self.module.init(1, self.device, self.dtype)
            for k in keys
        ]
        self._checked_out = keys
        return {name: torch.cat([r[name] for r in rows], dim=0) for name in rows[0]}

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
