"""The update-rule contract of :class:`~research_rl.optim.optimizer_wrapper.HookedOptimizer`.

An optimizer step is decomposed into four observable/controllable stages::

    grads ready ──► (1) post_grad ──► [optional clip] ──► inner optimizer computes candidate
                                                            (Adam preconditioning, weight decay,
                                                             optimizer-state update)
              ──► (2) transform_update(candidate Δ) ──► (3) pre_write(final value)
              ──► parameters hold final value ──► (4) post_step(actual displacement)

* ``post_grad``      – edit ``p.grad`` in place (projection, masking, noise, ...).
* ``transform_update`` – receives the *preconditioned candidate update* Δ = θ_cand − θ_old
  (fp32) and may return a replacement Δ'. This is where STABLE / DeltaHippo-style constraints
  that act on the Adam update (not on the gradient) plug in.
* ``pre_write``      – receives θ_old and the proposed θ_new and may return the value that is
  actually written. Masks (``MaskRule``) live here.
* ``post_step``      – receives the *actual* displacement θ_final − θ_old for every tracked
  parameter, after everything else. Use it for statistics, ledgers, protection state.

Rules are applied in registration order at every stage. A rule only pays for snapshots of the
parameters it declares via :meth:`UpdateRule.tracks`. All per-rule state that must survive a
checkpoint goes through :meth:`state_dict` / :meth:`load_state_dict`; it is stored inside the
optimizer state file, which Transformers reloads with ``torch.load(weights_only=True)``, so only
tensors, numbers, strings, lists and dicts are allowed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from research_rl.core.registry import UPDATE_RULES
from research_rl.models.parameter_mask import ParameterMask


@dataclass
class StepContext:
    """Information handed to every rule hook."""

    nominal_step: int  # number of times optimizer.step() has been called (incl. this one, 1-based)
    update_step: int  # number of *applied* updates before this one (= policy version before the step)
    lrs: list[float]  # learning rate of each param group
    grad_norm: float | None = None  # post-hook norm when clipping happens inside the wrapper
    extras: dict[str, Any] = field(default_factory=dict)


class UpdateRule:
    """Base class; override any subset of the hooks."""

    name: str = "update_rule"

    # Which parameters need a pre-step snapshot (only those reach transform_update/pre_write/post_step).
    def tracks(self, name: str, param: torch.nn.Parameter) -> bool:
        return False

    def post_grad(self, ctx: StepContext, named_params: list[tuple[str, torch.nn.Parameter]]) -> None:
        return None

    def transform_update(
        self, ctx: StepContext, name: str, param: torch.nn.Parameter, old: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor | None:
        """Return a replacement fp32 Δ, or ``None`` to keep it."""
        return None

    def pre_write(
        self, ctx: StepContext, name: str, param: torch.nn.Parameter, old: torch.Tensor, new: torch.Tensor
    ) -> torch.Tensor | None:
        """Return the tensor to write (same dtype as the parameter), or ``None`` to keep ``new``."""
        return None

    def post_step(self, ctx: StepContext, displacements: Mapping[str, torch.Tensor]) -> None:
        return None

    def on_skip(self, ctx: StepContext) -> None:
        """Called instead of the other hooks when the step is skipped."""
        return None

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        return None


@UPDATE_RULES.register("mask")
class MaskRule(UpdateRule):
    """Hard element-wise mask: masked-out elements are bit-identical before and after every step.

    Two stages are used on purpose:

    1. ``post_grad`` zeroes masked gradients so they do not accumulate into optimizer moments
       (keeps the optimizer state of frozen elements from drifting),
    2. ``pre_write`` restores ``old`` for masked elements with ``torch.where`` (not ``old + Δ``),
       which removes momentum- and weight-decay-driven motion and is exact in every dtype.

    Gradient clipping in the HF Trainer happens *before* ``optimizer.step()``; to keep masked
    gradients out of the clip norm, also register backward-time masks
    (:func:`research_rl.optim.gradient_hooks.register_gradient_masks`) or let the wrapper clip
    (``OptimSpec.clip_in_wrapper``).
    """

    name = "mask"

    def __init__(self, mask: ParameterMask, *, mask_gradients: bool = True) -> None:
        self.mask = mask
        self.mask_gradients = mask_gradients
        self._device_cache: dict[tuple[str, torch.device], torch.Tensor] = {}

    def _m(self, name: str, device: torch.device) -> torch.Tensor:
        key = (name, device)
        if key not in self._device_cache:
            m = self.mask.get(name)
            assert m is not None
            self._device_cache[key] = m.to(device)
        return self._device_cache[key]

    def tracks(self, name: str, param: torch.nn.Parameter) -> bool:
        return name in self.mask

    def post_grad(self, ctx: StepContext, named_params: list[tuple[str, torch.nn.Parameter]]) -> None:
        if not self.mask_gradients:
            return
        for name, p in named_params:
            if p.grad is not None and name in self.mask:
                p.grad.mul_(self._m(name, p.grad.device).to(p.grad.dtype))

    def pre_write(
        self, ctx: StepContext, name: str, param: torch.nn.Parameter, old: torch.Tensor, new: torch.Tensor
    ) -> torch.Tensor | None:
        return torch.where(self._m(name, new.device), new, old)

    def state_dict(self) -> dict[str, Any]:
        return {"mask": self.mask.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if "mask" in state:
            self.mask = ParameterMask.from_state_dict(state["mask"])
            self._device_cache.clear()


@UPDATE_RULES.register("displacement_ledger")
class DisplacementLedger(UpdateRule):
    """Accumulates the *actual* per-parameter displacement norm. Useful for diagnostics/tests."""

    name = "displacement_ledger"

    def __init__(self, names: list[str] | None = None) -> None:
        self.names = set(names) if names is not None else None
        self.total_sq_norm: dict[str, float] = {}
        self.last: dict[str, torch.Tensor] = {}

    def tracks(self, name: str, param: torch.nn.Parameter) -> bool:
        return self.names is None or name in self.names

    def post_step(self, ctx: StepContext, displacements: Mapping[str, torch.Tensor]) -> None:
        for name, d in displacements.items():
            if not self.tracks(name, None):  # type: ignore[arg-type]
                continue
            self.last[name] = d.detach().clone()
            self.total_sq_norm[name] = self.total_sq_norm.get(name, 0.0) + float(d.float().pow(2).sum())

    def state_dict(self) -> dict[str, Any]:
        return {"total_sq_norm": dict(self.total_sq_norm)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.total_sq_norm = dict(state.get("total_sq_norm", {}))
