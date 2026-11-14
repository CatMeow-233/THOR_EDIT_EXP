"""Backward-time gradient hooks.

These run inside ``loss.backward()`` (via ``Tensor.register_hook``), i.e. *before* the HF
Trainer computes the gradient norm and clips. Use them when a transformation must also affect
clipping and the logged ``grad_norm`` (e.g. channel masks: masked-out gradients must not
inflate the clip scale of the selected channels).

Hooks compose with gradient accumulation: every micro-batch gradient is transformed before
it is summed into ``.grad``. For linear transforms (masks, scaling) this is identical to
transforming the accumulated gradient.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn

from research_rl.models.parameter_mask import ParameterMask

GradTransform = Callable[[str, torch.Tensor], torch.Tensor]


class GradientHookHandle:
    """Holds the underlying torch handles; call :meth:`remove` to detach all hooks."""

    def __init__(self, handles: list[torch.utils.hooks.RemovableHandle]) -> None:
        self._handles = handles

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __len__(self) -> int:
        return len(self._handles)


def register_gradient_transform(
    model: nn.Module, transform: GradTransform, names: Iterable[str] | None = None
) -> GradientHookHandle:
    """Apply ``transform(name, grad) -> grad`` to the gradients of ``names`` (default: all trainable)."""
    wanted = set(names) if names is not None else None
    handles = []
    for name, p in model.named_parameters():
        if not p.requires_grad or (wanted is not None and name not in wanted):
            continue

        def hook(grad: torch.Tensor, _name: str = name) -> torch.Tensor:
            return transform(_name, grad)

        handles.append(p.register_hook(hook))
    return GradientHookHandle(handles)


def register_gradient_masks(model: nn.Module, mask: ParameterMask) -> GradientHookHandle:
    """Zero gradient elements that ``mask`` marks as frozen, at backward time."""
    mask.validate(model)
    device_masks: dict[str, torch.Tensor] = {}
    params = dict(model.named_parameters())
    for name, m in mask.items():
        device_masks[name] = m.to(params[name].device)

    def transform(name: str, grad: torch.Tensor) -> torch.Tensor:
        return grad * device_masks[name].to(grad.dtype)

    return register_gradient_transform(model, transform, names=mask.names())


def global_grad_norm(params: Iterable[torch.nn.Parameter], norm_type: float = 2.0) -> float:
    """Reference global gradient norm (independent of ``clip_grad_norm_``)."""
    grads = [p.grad.detach().float().flatten() for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.linalg.vector_norm(torch.cat(grads), ord=norm_type))
