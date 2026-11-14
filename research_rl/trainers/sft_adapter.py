"""Supervised fine-tuning (SFT) adapter around TRL's ``SFTTrainer``.

SFT is maximum-likelihood training on fixed target text; it is **not** reinforcement learning.

Loss semantics (TRL 1.18, ``loss_type="nll"``): token-level cross-entropy of the shifted labels,
summed over every non-ignored token of the micro-batch and divided by the number of such tokens
(or by ``num_items_in_batch`` across the whole gradient-accumulation window when HF passes it).
For prompt/completion datasets TRL's default ``completion_only_loss=None`` resolves to ``True``:
prompt tokens and padding are excluded and an EOS token is appended to every completion.

Everything in ``ExperimentConfig.trainer.args`` is forwarded verbatim to ``trl.SFTConfig``.

Token counter on resume: TRL's ``SFTTrainer`` / ``DPOTrainer`` keep the cumulative ``num_tokens``
metric in ``trainer._total_train_tokens``, which starts at 0 in ``__init__`` and is not part of any
HF checkpoint, so after a resume TRL alone would report only post-resume tokens. These adapters save
the counter in each checkpoint's ResearchRL state (``algorithm_state["trl_total_train_tokens"]``,
written at the optimizer step that is being checkpointed) and restore it in ``train()``. A
checkpoint without that key (e.g. not written through these adapters) leaves TRL's behaviour as is.
"""

from __future__ import annotations

from typing import Any, ClassVar

from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from research_rl.core.checkpoint import load_research_state
from research_rl.core.config import ExperimentConfig
from research_rl.core.registry import TRAINERS
from research_rl.trainers.base import ResearchCallback, ResearchRun, TrainerAdapter, research_trainer_class

TRAIN_TOKENS_KEY = "trl_total_train_tokens"


class TrainTokensResumeMixin:
    """Mixed in directly above a TRL trainer that counts tokens in ``_total_train_tokens``.

    Sits *below* :class:`~research_rl.trainers.base.ResearchTrainerMixin` in the MRO, so
    ``resume_from_checkpoint=True`` has already been resolved to a path when :meth:`train` runs.
    """

    def train(self, resume_from_checkpoint: str | bool | None = None, **kwargs: Any) -> Any:
        if isinstance(resume_from_checkpoint, str):
            algo_state = load_research_state(resume_from_checkpoint).get("algorithm_state") or {}
            saved = algo_state.get(TRAIN_TOKENS_KEY)
            if saved is not None:
                self._total_train_tokens = saved
        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)  # type: ignore[misc]


class TrainTokensCallback(TrainerCallback):
    """Copies ``trainer._total_train_tokens`` into the ResearchRL checkpoint state after every step.

    ``on_step_end`` runs before ``on_save`` for the same step, so the saved value is exact even when
    ``logging_steps`` and ``save_steps`` are not aligned.
    """

    def __init__(self, trainer: Any, research_callback: ResearchCallback) -> None:
        self.trainer = trainer
        self.research_callback = research_callback

    def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        self.research_callback.extra_state[TRAIN_TOKENS_KEY] = self.trainer._total_train_tokens


_TOKEN_RESUME_CLASSES: dict[type, type] = {}


def token_resume_trainer_class(trl_cls: type) -> type:
    """``research_trainer_class`` of ``trl_cls`` with :class:`TrainTokensResumeMixin` (cached)."""
    if trl_cls not in _TOKEN_RESUME_CLASSES:
        inner = type(trl_cls.__name__, (TrainTokensResumeMixin, trl_cls), {"__module__": __name__})
        _TOKEN_RESUME_CLASSES[trl_cls] = research_trainer_class(inner)
    return _TOKEN_RESUME_CLASSES[trl_cls]


def track_train_tokens(run: ResearchRun) -> ResearchRun:
    run.trainer.add_callback(TrainTokensCallback(run.trainer, run.callback))
    return run


@TRAINERS.register("sft")
class SFTAdapter(TrainerAdapter):
    """``TRAINERS.build("sft")`` -> adapter mapping an ``ExperimentConfig`` onto ``SFTTrainer``.

    Extra ``algo_kwargs`` (e.g. ``peft_config``, ``formatting_func``, ``data_collator``) are passed
    unchanged to ``SFTTrainer``.

    Default difference from TRL: ``bf16`` defaults to ``False`` (TRL's base config turns bf16
    autocast on whenever ``fp16`` is unset, also on CPU). Precision must be requested explicitly
    via ``trainer.args.bf16`` so that the experiment record states what actually ran.
    The ``num_tokens`` / ``training_tokens`` counter survives checkpoint resume (see module docs).
    """

    algorithm: ClassVar[str] = "sft"
    trl_trainer_cls: ClassVar[type] = SFTTrainer
    trl_config_cls: ClassVar[type] = SFTConfig
    is_reinforcement_learning: ClassVar[bool] = False
    default_args: ClassVar[dict[str, Any]] = {"bf16": False}

    def trainer_class(self) -> type:
        return token_resume_trainer_class(self.trl_trainer_cls)

    def build(self, exp: ExperimentConfig, **kwargs: Any) -> ResearchRun:  # type: ignore[override]
        return track_train_tokens(super().build(exp, **kwargs))
