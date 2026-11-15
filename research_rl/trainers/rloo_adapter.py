"""RLOO adapter on top of TRL's ``RLOOTrainer`` (REINFORCE with a leave-one-out baseline).

Semantics of the installed TRL (1.18), checked numerically in ``tests/research_rl``:

* **Advantages**: ``A_i = r_i - (Σ_{j in group} r_j - r_i) / (k - 1)`` (leave-one-out baseline over the
  ``k = num_generations`` completions of a prompt; NaN/unscorable rewards are excluded). If
  ``normalize_advantages=True`` (default ``False``), the advantages are standardised over the
  generation batch: ``(A - mean(A)) / (std(A) + 1e-4)`` with the unbiased ``std``.
* **KL**: default ``beta=0.05``. The KL penalty ``β Σ_t (logp_old - logp_ref)`` is subtracted
  from the *reward* before the baseline. The reference model is re-loaded from
  ``model.config._name_or_path``, so in-memory models need ``beta=0`` or a model saved to disk.
* **Loss**: per *sequence*, ``-min(ρ A, clip(ρ, 1-ε, 1+ε) A)`` with
  ``ρ = exp(Σ_t m_t logp_t - Σ_t m_t logp_old_t)``, averaged over the micro-batch. HF divides it by
  the number of accumulation steps.
* **Round-off**: equal (shaped) rewards give mathematically zero advantages, but TRL's float32
  ``(sum - r_i) / (k - 1)`` leaves a few ulp for many values (e.g. four rewards of 0.9 give
  ~6e-8; with ``normalize_advantages`` this is amplified by ``1 / (std + 1e-4)``). ResearchRL's
  learning-signal flags recognise such groups from the shaped rewards
  (see :meth:`ResearchRLOOTrainer._rr_reward_signal`).
* **Completion mask**: 1 up to and including the first EOS (transformers generation path).
  ``mask_truncated_completions`` zeroes rows whose last token is neither EOS nor pad. RLOO has no
  ``rollout_func`` in this TRL version.

ResearchRL additions (policy-version stamps, trajectory log, advantage shaping, zero-advantage
skipping, per-row expert routing with ``expert_manager=``) are described in :mod:`research_rl.trainers.rl_common`. Shaped advantages must be
``(B,)`` (the loss is per sequence). ``skip_zero_advantage_steps=True`` is compatible with
``beta != 0``: the KL penalty is folded into the reward and hence into the advantage, so a window
whose advantages are all zero has no KL gradient either. It is refused with the MoE aux loss.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from trl import RLOOConfig, RLOOTrainer

from research_rl.core.registry import TRAINERS
from research_rl.trainers.base import ResearchTrainerMixin
from research_rl.trainers.rl_common import RLRolloutMixin, RLTrainerAdapter, group_deviation_signal


class ResearchRLOOTrainer(RLRolloutMixin, ResearchTrainerMixin, RLOOTrainer):  # type: ignore[misc]  # cooperative mixins refine TRL attributes
    """``RLOOTrainer`` + HookedOptimizer + policy-version stamps, trajectory log, zero-advantage skip."""

    beta: float
    reward_clip_range: tuple[float, float] | None
    _rr_logp_capture: list[torch.Tensor] | None = None

    def _rr_begin_generation(self) -> None:
        # With beta != 0 TRL computes the policy's and the reference's per-token logps (in this
        # order) inside _generate_and_score_completions; capture them to rebuild the KL-shaped reward.
        self._rr_logp_capture = [] if self.beta != 0.0 else None

    def _get_per_token_logps_and_entropies(self, *args: Any, **kwargs: Any) -> Any:
        out = super()._get_per_token_logps_and_entropies(*args, **kwargs)
        if self._rr_logp_capture is not None:
            self._rr_logp_capture.append(out[0].detach())
        return out

    def _rr_reward_signal(self, mode: str, completion_mask: torch.Tensor) -> torch.Tensor | None:
        """Mathematically non-zero RLOO advantage, per sample (gathered).

        ``A_i = r_i - (Σ_j r_j - r_i) / (n - 1) = n / (n - 1) · (r_i - mean)`` over the ``n`` scorable
        members of the group, where ``r`` is TRL's reward after weighting, ``reward_clip_range``
        and (``beta != 0``) the KL penalty ``β Σ_t m_t (logp_old - logp_ref)``. The KL is rebuilt from
        the captured log-probs with TRL's float32 operations. So ``A_i = 0`` iff ``r_i`` equals the
        group mean. ``normalize_advantages`` does not change this: the leave-one-out advantages of
        a group sum to 0, so their batch mean is 0. Returns ``None`` (exact-zero fallback) if the
        two log-prob passes were not observed.
        """
        rewards = self._rr_weighted_rewards()
        capture, self._rr_logp_capture = self._rr_logp_capture, None
        if rewards is None:
            return None
        if self.reward_clip_range:
            rewards = rewards.clamp(min=self.reward_clip_range[0], max=self.reward_clip_range[1])
        if self.beta != 0.0:
            if capture is None or len(capture) != 2:
                return None
            old, ref = capture
            kl = ((old - ref) * completion_mask).sum(-1)
            kl = self.accelerator.gather(kl)
            rewards = rewards - self.beta * kl
        return group_deviation_signal(rewards, self._rr_num_generations(mode))


@TRAINERS.register("rloo")
class RLOOAdapter(RLTrainerAdapter):
    """``TRAINERS.build("rloo")``: maps an ``ExperimentConfig`` onto :class:`ResearchRLOOTrainer`.

    Pass ``reward_funcs=...`` (and optionally ``peft_config``, ``trajectory_capacity``,
    ``expert_manager``, ``expert_column``) to ``build``. ``trainer.args`` go verbatim to ``trl.RLOOConfig``.
    """

    algorithm: ClassVar[str] = "rloo"
    trl_trainer_cls: ClassVar[type] = RLOOTrainer
    trl_config_cls: ClassVar[type] = RLOOConfig
    research_trainer_cls: ClassVar[type] = ResearchRLOOTrainer
    default_args: ClassVar[dict[str, Any]] = {"bf16": False}
