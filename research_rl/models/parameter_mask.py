"""Element-wise parameter masks.

A :class:`ParameterMask` maps parameter names to boolean tensors of the same shape. ``True``
means *this element is allowed to change*; ``False`` means it must stay bit-identical.
Parameters that do not appear in the mask are unconstrained.

Masks are consumed in two places:

* backward time (:func:`research_rl.optim.gradient_hooks.register_gradient_masks`) so that
  masked-out gradients never enter gradient clipping or grad-norm statistics, and
* update time (:class:`research_rl.optim.update_interface.MaskRule`) which restores masked-out
  elements *after* the optimizer step. This second stage is what makes masks correct for
  optimizers with momentum / weight decay: masking only the gradient is not enough, because
  Adam keeps moving an element with stale first/second moments and AdamW decays it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping

import torch
from torch import nn


class MaskError(ValueError):
    pass


class ParameterMask:
    def __init__(self, masks: Mapping[str, torch.Tensor] | None = None) -> None:
        self._masks: dict[str, torch.Tensor] = {}
        for name, mask in (masks or {}).items():
            self.set(name, mask)

    # ----------------------------------------------------------------- construction
    @classmethod
    def all_frozen(cls, model: nn.Module, names: Iterable[str]) -> ParameterMask:
        params = dict(model.named_parameters())
        return cls({n: torch.zeros_like(params[n], dtype=torch.bool) for n in names})

    @classmethod
    def all_trainable(cls, model: nn.Module, names: Iterable[str]) -> ParameterMask:
        params = dict(model.named_parameters())
        return cls({n: torch.ones_like(params[n], dtype=torch.bool) for n in names})

    def set(self, name: str, mask: torch.Tensor) -> None:
        if mask.dtype != torch.bool:
            raise MaskError(f"mask for {name!r} must be bool, got {mask.dtype}")
        self._masks[name] = mask.detach()

    # ----------------------------------------------------------------- algebra
    def union(self, other: ParameterMask) -> ParameterMask:
        """Element may change if it may change in either mask.

        A name missing from one operand is unconstrained there, so it is unconstrained (absent)
        in the union as well — consistent with :meth:`intersection`.
        """
        out = ParameterMask()
        for name in set(self._masks) & set(other._masks):
            out._masks[name] = self._masks[name] | other._masks[name]
        return out

    def intersection(self, other: ParameterMask) -> ParameterMask:
        out = ParameterMask()
        for name in set(self._masks) | set(other._masks):
            a, b = self._masks.get(name), other._masks.get(name)
            if a is None:
                assert b is not None
                out._masks[name] = b
            elif b is None:
                out._masks[name] = a
            else:
                out._masks[name] = a & b
        return out

    # ----------------------------------------------------------------- queries
    def names(self) -> list[str]:
        return list(self._masks)

    def items(self) -> Iterator[tuple[str, torch.Tensor]]:
        return iter(self._masks.items())

    def get(self, name: str) -> torch.Tensor | None:
        return self._masks.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._masks

    def __len__(self) -> int:
        return len(self._masks)

    def num_trainable(self) -> int:
        return int(sum(int(m.sum().item()) for m in self._masks.values()))

    def num_frozen(self) -> int:
        return int(sum(int((~m).sum().item()) for m in self._masks.values()))

    def validate(self, model: nn.Module) -> None:
        params = dict(model.named_parameters())
        for name, m in self._masks.items():
            if name not in params:
                raise MaskError(f"mask refers to unknown parameter {name!r}")
            if m.shape != params[name].shape:
                raise MaskError(f"mask shape {tuple(m.shape)} != parameter shape {tuple(params[name].shape)}")

    def to(self, device: torch.device | str) -> ParameterMask:
        return ParameterMask({n: m.to(device) for n, m in self._masks.items()})

    # ----------------------------------------------------------------- application
    @torch.no_grad()
    def restore_frozen(self, model: nn.Module, reference: Mapping[str, torch.Tensor]) -> None:
        """Write ``reference`` back into every masked-out element of ``model``."""
        params = dict(model.named_parameters())
        for name, m in self._masks.items():
            p = params[name]
            ref = reference[name].to(device=p.device, dtype=p.dtype)
            p.copy_(torch.where(m.to(p.device), p, ref))

    def changed_outside(
        self, before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
    ) -> dict[str, int]:
        """Number of masked-out elements that differ between two snapshots (should be all 0)."""
        out: dict[str, int] = {}
        for name, m in self._masks.items():
            diff = before[name].cpu() != after[name].cpu()
            out[name] = int((diff & ~m.cpu()).sum().item())
        return out

    # ----------------------------------------------------------------- persistence
    def state_dict(self) -> dict[str, torch.Tensor]:
        return {n: m.cpu() for n, m in self._masks.items()}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor]) -> ParameterMask:
        return cls(dict(state))
