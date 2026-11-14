"""GRPO adapter on top of TRL's ``GRPOTrainer`` (reinforcement learning from verifiable/learned rewards).

Semantics of the installed TRL (1.18), checked numerically in ``tests/research_rl``:

* **Advantages** (``scale_rewards``): ``A_i = r_i - mean(group)``. For ``"group"`` (default) it is
  divided by ``std(group) + 1e-4``. For ``"batch"`` it is divided by ``std(generation batch) + 1e-4``,
  while the mean is still the *group* mean. ``"none"`` does not divide. ``std`` is the
  Bessel-corrected (unbiased) NaN-aware ``trl.trainer.utils.nanstd``. A group with equal rewards
  gets exactly zero advantages.
* **Loss** (default ``loss_type="dapo"``, ``beta=0``): the per-token clipped surrogate
  ``-min(ρ A, clip(ρ, 1-ε, 1+ε) A)``. Its token sum over the micro-batch is divided by the number
  of loss tokens of the whole generation batch, rescaled to one accumulation window, so the
  gradient of a full window is ``-Σ_{i,t} m_it A_i ∇ρ_it / Σ m``. On-policy (``num_iterations=1``,
  ``steps_per_generation <= gradient_accumulation_steps``) TRL uses ``ρ = exp(logp - logp.detach())``.
* **Completion mask**: with ``transformers`` generation, it is 1 up to and including the first EOS.
  With a custom ``rollout_func``, it is 1 on every returned token (see
  :func:`research_rl.trainers.rl_common.eos_truncating_rollout`). ``mask_truncated_completions``
  zeroes rows whose last token is neither EOS nor pad.

ResearchRL additions (policy-version stamps, trajectory log, zero-advantage skipping) are
described in :mod:`research_rl.trainers.rl_common`.
"""

from __future__ import annotations

from typing import Any, ClassVar

from trl import GRPOConfig, GRPOTrainer

from research_rl.core.registry import TRAINERS
from research_rl.trainers.base import ResearchTrainerMixin
from research_rl.trainers.rl_common import RLRolloutMixin, RLTrainerAdapter


class ResearchGRPOTrainer(RLRolloutMixin, ResearchTrainerMixin, GRPOTrainer):
    """``GRPOTrainer`` + HookedOptimizer + policy-version stamps, trajectory log, zero-advantage skip."""


@TRAINERS.register("grpo")
class GRPOAdapter(RLTrainerAdapter):
    """``TRAINERS.build("grpo")``: maps an ``ExperimentConfig`` onto :class:`ResearchGRPOTrainer`.

    Pass ``reward_funcs=...`` (and optionally ``rollout_func``, ``peft_config``,
    ``trajectory_capacity``) to ``build``. ``trainer.args`` go verbatim to ``trl.GRPOConfig``.
    """

    algorithm: ClassVar[str] = "grpo"
    trl_trainer_cls: ClassVar[type] = GRPOTrainer
    trl_config_cls: ClassVar[type] = GRPOConfig
    research_trainer_cls: ClassVar[type] = ResearchGRPOTrainer
    default_args: ClassVar[dict[str, Any]] = {"bf16": False}
