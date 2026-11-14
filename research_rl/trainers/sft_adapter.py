"""Supervised fine-tuning (SFT) adapter around TRL's ``SFTTrainer``.

SFT is maximum-likelihood training on fixed target text; it is **not** reinforcement learning.

Loss semantics (TRL 1.18, ``loss_type="nll"``): token-level cross-entropy of the shifted labels,
summed over every non-ignored token of the micro-batch and divided by the number of such tokens
(or by ``num_items_in_batch`` across the whole gradient-accumulation window when HF passes it).
For prompt/completion datasets TRL's default ``completion_only_loss=None`` resolves to ``True``:
prompt tokens and padding are excluded and an EOS token is appended to every completion.

Everything in ``ExperimentConfig.trainer.args`` is forwarded verbatim to ``trl.SFTConfig``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from trl import SFTConfig, SFTTrainer

from research_rl.core.registry import TRAINERS
from research_rl.trainers.base import TrainerAdapter


@TRAINERS.register("sft")
class SFTAdapter(TrainerAdapter):
    """``TRAINERS.build("sft")`` -> adapter mapping an ``ExperimentConfig`` onto ``SFTTrainer``.

    Extra ``algo_kwargs`` (e.g. ``peft_config``, ``formatting_func``, ``data_collator``) are passed
    unchanged to ``SFTTrainer``.

    Default difference from TRL: ``bf16`` defaults to ``False`` (TRL's base config turns bf16
    autocast on whenever ``fp16`` is unset, also on CPU). Precision must be requested explicitly
    via ``trainer.args.bf16`` so that the experiment record states what actually ran.
    """

    algorithm: ClassVar[str] = "sft"
    trl_trainer_cls: ClassVar[type] = SFTTrainer
    trl_config_cls: ClassVar[type] = SFTConfig
    is_reinforcement_learning: ClassVar[bool] = False
    default_args: ClassVar[dict[str, Any]] = {"bf16": False}
