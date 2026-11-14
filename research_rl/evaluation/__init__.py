"""Evaluation: exact match, pass@k, bootstrap CIs, token NLL, and capability-retention probes."""

from __future__ import annotations

from research_rl.evaluation.capability_retention import (
    CapabilityRetention,
    KLResult,
    ProbeTask,
    RetentionReport,
    RetentionSuite,
    SuiteResult,
    exact_match_scorer,
    greedy_generate,
    logit_kl,
    prefix_match_scorer,
)
from research_rl.evaluation.metrics import (
    MeanCI,
    NLLResult,
    exact_match,
    exact_match_accuracy,
    mean_pass_at_k,
    mean_with_bootstrap_ci,
    normalize_answer,
    pass_at_k,
    sequence_nll,
)

__all__ = [
    "CapabilityRetention",
    "KLResult",
    "MeanCI",
    "NLLResult",
    "ProbeTask",
    "RetentionReport",
    "RetentionSuite",
    "SuiteResult",
    "exact_match",
    "exact_match_accuracy",
    "exact_match_scorer",
    "greedy_generate",
    "logit_kl",
    "mean_pass_at_k",
    "mean_with_bootstrap_ci",
    "normalize_answer",
    "pass_at_k",
    "prefix_match_scorer",
    "sequence_nll",
]
