"""Per-sample recurrent state and adaptive-compute contracts (extension points for future research)."""

from research_rl.state.compute_controller import (
    BestOfDepthResult,
    ComputeController,
    ComputeLedger,
    FixedDepth,
    LedgerEntry,
    LoopResult,
    ThresholdHalting,
    best_of_depth,
    collect_depth_outputs,
    run_looped,
)
from research_rl.state.recurrent_interface import FastSlowMemory, RecurrentState, StateStore, batch_size_of

__all__ = [
    "BestOfDepthResult",
    "ComputeController",
    "ComputeLedger",
    "FastSlowMemory",
    "FixedDepth",
    "LedgerEntry",
    "LoopResult",
    "RecurrentState",
    "StateStore",
    "ThresholdHalting",
    "batch_size_of",
    "best_of_depth",
    "collect_depth_outputs",
    "run_looped",
]
