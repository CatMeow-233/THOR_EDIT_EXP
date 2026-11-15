"""Optimizer extension: update rules, HookedOptimizer and backward-time gradient hooks."""

from research_rl.optim.gradient_hooks import register_gradient_masks, register_gradient_transform
from research_rl.optim.optimizer_wrapper import HookedOptimizer, unwrap_hooked
from research_rl.optim.parameter_prior import (
    P2PriorRule,
    PriorSubspaceRule,
    TaskVector,
    TaskVectorPrior,
    project_onto_span,
)
from research_rl.optim.precision import (
    is_low_precision,
    low_precision_trainable,
    round_to,
    rounding_generator,
    stochastic_round,
)
from research_rl.optim.update_interface import DisplacementLedger, MaskRule, StepContext, UpdateRule

__all__ = [
    "DisplacementLedger",
    "HookedOptimizer",
    "MaskRule",
    "P2PriorRule",
    "PriorSubspaceRule",
    "StepContext",
    "TaskVector",
    "TaskVectorPrior",
    "UpdateRule",
    "is_low_precision",
    "low_precision_trainable",
    "project_onto_span",
    "register_gradient_masks",
    "register_gradient_transform",
    "round_to",
    "rounding_generator",
    "stochastic_round",
    "unwrap_hooked",
]
