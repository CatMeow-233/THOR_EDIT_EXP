"""Optimizer extension: update rules, HookedOptimizer and backward-time gradient hooks."""

from research_rl.optim.gradient_hooks import register_gradient_masks, register_gradient_transform
from research_rl.optim.optimizer_wrapper import HookedOptimizer, unwrap_hooked
from research_rl.optim.update_interface import DisplacementLedger, MaskRule, StepContext, UpdateRule

__all__ = [
    "DisplacementLedger",
    "HookedOptimizer",
    "MaskRule",
    "StepContext",
    "UpdateRule",
    "register_gradient_masks",
    "register_gradient_transform",
    "unwrap_hooked",
]
