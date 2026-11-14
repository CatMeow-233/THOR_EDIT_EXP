"""Direct Preference Optimisation (DPO) adapter around TRL's ``DPOTrainer``.

DPO is **offline preference optimisation**, not reinforcement learning: it fits a fixed dataset
of (prompt, chosen, rejected) pairs, never samples from the policy and uses no reward model.

Loss semantics (TRL 1.18 defaults: ``loss_type=["sigmoid"]``, ``beta=0.1``,
``f_divergence_type="reverse_kl"``, ``ld_alpha=None``)::

    logp(y) = sum over completion tokens of log pi(y_t | x, y_<t)   (prompt and padding excluded)
    loss    = mean_pairs  -logsigmoid(beta * ((logp_pi(c) - logp_ref(c)) - (logp_pi(r) - logp_ref(r))))

For non-conversational data TRL appends the tokenizer's EOS to ``chosen`` and ``rejected``.

Reference model handling (the only behaviour this adapter adds on top of TRL):

* ``ref_model=<PreTrainedModel>`` is passed through.
* ``ref_model="<hf id or path>"`` is loaded with the experiment's dtype / trust settings.
* ``ref_model=None`` without PEFT: a frozen deep copy of the *initial* policy is used. This is
  the same reference TRL would load from ``model.config._name_or_path``, but it also works for
  in-memory models that were never saved and avoids a second disk read.
* ``ref_model=None`` with ``peft_config`` (or an existing ``PeftModel``): left to TRL, which uses
  the adapter-disabled base model as the reference.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, ClassVar

import torch
from transformers import PreTrainedModel
from trl import DPOConfig, DPOTrainer

from research_rl.core.config import ExperimentConfig
from research_rl.core.experiment import model_identity
from research_rl.core.registry import TRAINERS
from research_rl.trainers.base import ResearchRun, TrainerAdapter, load_causal_lm


def _is_peft_model(model: Any) -> bool:
    try:
        from peft import PeftModel
    except ImportError:  # pragma: no cover - peft is a pinned dependency
        return False
    return isinstance(model, PeftModel)


def frozen_copy(model: PreTrainedModel) -> PreTrainedModel:
    """Deep copy with gradients disabled and eval mode (used as a fixed reference policy)."""
    ref = copy.deepcopy(model)
    ref.requires_grad_(False)
    ref.eval()
    return ref


@TRAINERS.register("dpo")
class DPOAdapter(TrainerAdapter):
    """``TRAINERS.build("dpo")`` -> adapter for offline preference optimisation with ``DPOTrainer``.

    ``bf16`` defaults to ``False`` (see :class:`~research_rl.trainers.sft_adapter.SFTAdapter`).
    The reference model's identity is written to the experiment record as ``reference_model``.
    """

    algorithm: ClassVar[str] = "dpo"
    trl_trainer_cls: ClassVar[type] = DPOTrainer
    trl_config_cls: ClassVar[type] = DPOConfig
    is_reinforcement_learning: ClassVar[bool] = False
    default_args: ClassVar[dict[str, Any]] = {"bf16": False}

    def build(  # type: ignore[override]
        self,
        exp: ExperimentConfig,
        *,
        train_dataset: Any,
        model: PreTrainedModel | str | None = None,
        ref_model: PreTrainedModel | str | None = None,
        **kwargs: Any,
    ) -> ResearchRun:
        if exp.trainer.algorithm != self.algorithm:
            raise ValueError(f"config asks for {exp.trainer.algorithm!r}, adapter is {self.algorithm!r}")
        torch.manual_seed(exp.seed)
        if model is None or isinstance(model, str):
            spec = exp.model if model is None else dataclasses.replace(exp.model, name_or_path=model)
            model = load_causal_lm(spec)
        if isinstance(ref_model, str):
            ref_model = load_causal_lm(dataclasses.replace(exp.model, name_or_path=ref_model, revision=None))
        elif ref_model is None and kwargs.get("peft_config") is None and not _is_peft_model(model):
            ref_model = frozen_copy(model)
        run = super().build(exp, train_dataset=train_dataset, model=model, ref_model=ref_model, **kwargs)
        ref = getattr(run.trainer, "ref_model", None)
        if run.recorder is not None:
            run.recorder.record["reference_model"] = (
                model_identity(run.trainer.accelerator.unwrap_model(ref))
                if ref is not None
                else "policy with PEFT adapters disabled"
            )
            run.recorder._flush()
        return run
