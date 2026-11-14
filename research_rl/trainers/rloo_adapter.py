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
* **Completion mask**: 1 up to and including the first EOS (transformers generation path).
  ``mask_truncated_completions`` zeroes rows whose last token is neither EOS nor pad. RLOO has no
  ``rollout_func`` in this TRL version.

ResearchRL additions (policy-version stamps, trajectory log, zero-advantage skipping) are
described in :mod:`research_rl.trainers.rl_common`.
"""

from __future__ import annotations

from typing import Any, ClassVar

from trl import RLOOConfig, RLOOTrainer

from research_rl.core.registry import TRAINERS
from research_rl.trainers.base import ResearchTrainerMixin
from research_rl.trainers.rl_common import RLRolloutMixin, RLTrainerAdapter


class ResearchRLOOTrainer(RLRolloutMixin, ResearchTrainerMixin, RLOOTrainer):
    """``RLOOTrainer`` + HookedOptimizer + policy-version stamps, trajectory log, zero-advantage skip."""


@TRAINERS.register("rloo")
class RLOOAdapter(RLTrainerAdapter):
    """``TRAINERS.build("rloo")``: maps an ``ExperimentConfig`` onto :class:`ResearchRLOOTrainer`.

    Pass ``reward_funcs=...`` (and optionally ``peft_config``, ``trajectory_capacity``) to
    ``build``. ``trainer.args`` go verbatim to ``trl.RLOOConfig``.
    """

    algorithm: ClassVar[str] = "rloo"
    trl_trainer_cls: ClassVar[type] = RLOOTrainer
    trl_config_cls: ClassVar[type] = RLOOConfig
    research_trainer_cls: ClassVar[type] = ResearchRLOOTrainer
    default_args: ClassVar[dict[str, Any]] = {"bf16": False}
