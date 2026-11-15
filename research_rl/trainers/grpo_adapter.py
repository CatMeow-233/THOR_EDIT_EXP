"""GRPO adapter on top of TRL's ``GRPOTrainer`` (reinforcement learning from verifiable/learned rewards).

Semantics of the installed TRL (1.18), checked numerically in ``tests/research_rl``:

* **Advantages** (``scale_rewards``): ``A_i = r_i - mean(group)``. For ``"group"`` (default) it is
  divided by ``std(group) + 1e-4``. For ``"batch"`` it is divided by ``std(generation batch) + 1e-4``,
  while the mean is still the *group* mean. ``"none"`` does not divide. ``std`` is the
  Bessel-corrected (unbiased) NaN-aware ``trl.trainer.utils.nanstd``. A group with equal rewards
  has *mathematically* zero advantages. In TRL's float32 arithmetic they are exactly zero only
  for some values (e.g. 1.0; it depends on the value and on ``G``). Eight rewards of 0.1 give
  round-off advantages of ~7e-5, three of 0.9 give ~6e-4 (``1e-4`` in the divisor amplifies the ulp-level
  ``r - mean``). ResearchRL's learning-signal flags and step skipping recognise such groups from
  the rewards (see :meth:`ResearchGRPOTrainer._rr_reward_signal`). TRL still trains on the
  round-off when steps are not skipped.
* **Loss** (default ``loss_type="dapo"``, ``beta=0``): the per-token clipped surrogate
  ``-min(ρ A, clip(ρ, 1-ε, 1+ε) A)``. Its token sum over the micro-batch is divided by the number
  of loss tokens of the whole generation batch, rescaled to one accumulation window, so the
  gradient of a full window is ``-Σ_{i,t} m_it A_i ∇ρ_it / Σ m``. On-policy (``num_iterations=1``,
  ``steps_per_generation <= gradient_accumulation_steps``) TRL uses ``ρ = exp(logp - logp.detach())``.
* **Completion mask**: with ``transformers`` generation, it is 1 up to and including the first EOS.
  With a custom ``rollout_func``, it is 1 on every returned token (see
  :func:`research_rl.trainers.rl_common.eos_truncating_rollout`). ``mask_truncated_completions``
  zeroes rows whose last token is neither EOS nor pad.

ResearchRL additions (policy-version stamps, trajectory log, advantage shaping, zero-advantage
skipping, per-row expert routing with ``expert_manager=``) are described in :mod:`research_rl.trainers.rl_common`. Shaped advantages may be
``(B,)`` or per token ``(B, T)``. ``skip_zero_advantage_steps=True`` is refused with ``beta != 0``
(the KL term is a separate loss term), with the entropy bonus and with the MoE aux loss.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from trl import GRPOConfig, GRPOTrainer

from research_rl.core.registry import TRAINERS
from research_rl.trainers.base import ResearchTrainerMixin
from research_rl.trainers.rl_common import RLRolloutMixin, RLTrainerAdapter, group_deviation_signal


class ResearchGRPOTrainer(RLRolloutMixin, ResearchTrainerMixin, GRPOTrainer):  # type: ignore[misc]  # cooperative mixins refine TRL attributes
    """``GRPOTrainer`` + HookedOptimizer + policy-version stamps, trajectory log, zero-advantage skip."""

    multi_objective_aggregation: str
    beta: float
    _rr_advantage_ndims: ClassVar[tuple[int, ...]] = (1, 2)  # TRL's GRPO loss accepts (B,) and (B, T)

    def _rr_skip_conflicts(self) -> list[str]:
        """TRL 1.18 GRPO terms that do not scale with the advantage, plus the MoE aux loss (base class).

        * ``beta != 0``: ``per_token_loss + beta * per_token_kl`` is added to the per-token loss.
        * entropy bonus (``_entropy_bonus_enabled``: ``entropy_coef != 0`` or ``use_adaptive_entropy``):
          ``loss - coef * mean entropy``.
        """
        conflicts = super()._rr_skip_conflicts()
        if self.beta != 0.0:
            conflicts.append(f"GRPO beta={self.beta} adds a KL loss term (set beta=0)")
        if getattr(self, "_entropy_bonus_enabled", False):
            conflicts.append(
                f"the GRPO entropy bonus is enabled (entropy_coef={getattr(self, 'entropy_coef', None)}, "
                f"use_adaptive_entropy={getattr(self, 'use_adaptive_entropy', None)})"
            )
        return conflicts

    def _rr_reward_signal(self, mode: str, completion_mask: torch.Tensor) -> torch.Tensor | None:
        """Mathematically non-zero GRPO advantage, per sample (gathered).

        * ``sum_then_normalize`` (default): ``A_i ∝ r_i - mean(group)`` for every ``scale_rewards``
          (the divisor is positive), so the signal is "weighted reward differs from its group mean".
        * ``normalize_then_sum``: ``A_i ∝ R_i - mean(R)`` with ``R_i = Σ_k w_k z_ik`` and per-function
          group z-scores ``z``. Every group's z-scores sum to 0, so ``mean(R) = 0`` and ``A_i = 0``
          iff ``R_i = 0``. The flag is "some weighted reward function deviates from its group
          mean". This is conservative: a cancellation between functions still counts as signal.

        ``beta`` does not enter: GRPO's KL term is part of the loss, not of the advantage.
        """
        rpf = self._rr_rewards_per_func
        if rpf is None:
            return None
        num_generations = self._rr_num_generations(mode)
        if self.multi_objective_aggregation == "sum_then_normalize":
            rewards = self._rr_weighted_rewards()
            assert rewards is not None
            return group_deviation_signal(rewards, num_generations)
        if self.multi_objective_aggregation == "normalize_then_sum":
            weights = self.reward_weights.to(rpf.device)
            unscorable = torch.isnan(rpf).all(dim=1)
            signal = torch.zeros(rpf.shape[0], dtype=torch.bool, device=rpf.device)
            for k in range(rpf.shape[1]):
                if float(weights[k]) != 0.0:
                    signal |= group_deviation_signal(rpf[:, k], num_generations)
            return signal & ~unscorable
        return None


@TRAINERS.register("grpo")
class GRPOAdapter(RLTrainerAdapter):
    """``TRAINERS.build("grpo")``: maps an ``ExperimentConfig`` onto :class:`ResearchGRPOTrainer`.

    Pass ``reward_funcs=...`` (and optionally ``rollout_func``, ``peft_config``,
    ``trajectory_capacity``, ``expert_manager``, ``expert_column``) to ``build``. ``trainer.args`` go verbatim to ``trl.GRPOConfig``.
    """

    algorithm: ClassVar[str] = "grpo"
    trl_trainer_cls: ClassVar[type] = GRPOTrainer
    trl_config_cls: ClassVar[type] = GRPOConfig
    research_trainer_cls: ClassVar[type] = ResearchGRPOTrainer
    default_args: ClassVar[dict[str, Any]] = {"bf16": False}
