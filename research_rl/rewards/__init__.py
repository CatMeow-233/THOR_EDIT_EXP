"""Verifiable rewards for RLVR: verifier contract, deterministic math verifier, disabled-by-default code sandbox.

Importing this package registers ``"math"`` and ``"code_execution"`` in
:data:`research_rl.core.registry.VERIFIERS`.
"""

from __future__ import annotations

from research_rl.rewards.code_sandbox import (
    CodeExecutionVerifier,
    ExecutionRequest,
    ExecutionResult,
    IsolationGuarantees,
    SandboxBackend,
    SandboxDisabledError,
    SandboxFusionBackend,
    SandboxPolicy,
    check_sandbox,
    extract_code,
)
from research_rl.rewards.math_verifier import MathVerifier, normalize_answer, parse_number
from research_rl.rewards.verifier_interface import (
    STATUSES,
    BaseVerifier,
    FatalVerifierError,
    RewardScheme,
    TRLRewardFunction,
    VerificationResult,
    Verifier,
    as_trl_reward_func,
    completion_to_text,
)

__all__ = [
    "STATUSES",
    "BaseVerifier",
    "CodeExecutionVerifier",
    "ExecutionRequest",
    "ExecutionResult",
    "FatalVerifierError",
    "IsolationGuarantees",
    "MathVerifier",
    "RewardScheme",
    "SandboxBackend",
    "SandboxDisabledError",
    "SandboxFusionBackend",
    "SandboxPolicy",
    "TRLRewardFunction",
    "VerificationResult",
    "Verifier",
    "as_trl_reward_func",
    "check_sandbox",
    "completion_to_text",
    "extract_code",
    "normalize_answer",
    "parse_number",
]
