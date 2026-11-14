"""On-policy knowledge distillation adapter around TRL's ``DistillationTrainer``.

This is **distillation, not reinforcement learning**: there is no reward and no policy-gradient
estimator. The student samples completions for the dataset prompts (on-policy rollouts, regenerated
once per gradient-accumulation window) and is trained to match the teacher's full next-token
distribution on those completion tokens.

Loss semantics (read from ``trl/trainer/distillation_trainer.py`` and
``trl/kernels/chunked_divergence.py`` in TRL 1.18). With ``T = temperature``, per completion token::

    p_s = softmax(student_logits / T),  p_t = softmax(teacher_logits / T)
    beta == 0     : KL(p_t || p_s)                         (forward KL)
    beta == 1     : KL(p_s || p_t)                         (reverse KL, TRL default)
    0 < beta < 1  : beta * KL(p_t || m) + (1 - beta) * KL(p_s || m),  m = beta * p_t + (1 - beta) * p_s

The per-token divergences are summed over completion tokens (prompt and right padding excluded)
and divided by the number of completion tokens in the whole generation batch (so gradient
accumulation is token-weighted); there is no ``T**2`` rescaling. The teacher gets no gradient.
``temperature`` also sets the student's sampling temperature.

Note on TRL versions: TRL 1.18's ``DistillationTrainer`` is purely on-policy — its config has no
``lmbda`` (mixing of dataset completions and student samples) and no ``seq_kd``. In TRL 1.18
``lmbda`` exists only in experimental trainers (``trl.experimental.gkd``, ``.gold``, ``.iw_opd``;
``seq_kd`` in the first two), which this adapter does not wrap.
"""

from __future__ import annotations

from typing import Any, ClassVar

from transformers import PreTrainedModel
from trl import DistillationConfig, DistillationTrainer

from research_rl.core.config import ExperimentConfig
from research_rl.core.experiment import model_identity
from research_rl.core.registry import TRAINERS
from research_rl.trainers.base import ResearchRun, TrainerAdapter


@TRAINERS.register("distillation")
class DistillationAdapter(TrainerAdapter):
    """``TRAINERS.build("distillation")`` -> adapter for on-policy distillation.

    The teacher is given as ``teacher_model=`` (a ``PreTrainedModel`` or an HF id / local path,
    which TRL loads with ``trainer.args.teacher_model_init_kwargs``). If it is omitted, the
    config's ``trainer.args.teacher_model_name_or_path`` is used. Teacher and student must share
    the vocabulary. The teacher's identity is written to the experiment record as ``teacher_model``.

    ``bf16`` defaults to ``False`` (see :class:`~research_rl.trainers.sft_adapter.SFTAdapter`).
    """

    algorithm: ClassVar[str] = "distillation"
    trl_trainer_cls: ClassVar[type] = DistillationTrainer
    trl_config_cls: ClassVar[type] = DistillationConfig
    is_reinforcement_learning: ClassVar[bool] = False
    default_args: ClassVar[dict[str, Any]] = {"bf16": False}

    def trainer_kwargs(self, exp: ExperimentConfig, **algo_kwargs: Any) -> dict[str, Any]:
        kwargs = dict(algo_kwargs)
        teacher = kwargs.pop("teacher_model", None)
        if teacher is None:
            teacher = exp.trainer.args.get("teacher_model_name_or_path")
        if teacher is None:
            raise ValueError(
                "distillation needs a teacher: pass teacher_model=<model or path> to build_run, or set "
                "trainer.args.teacher_model_name_or_path"
            )
        if not isinstance(teacher, (str, PreTrainedModel)):
            raise TypeError(f"teacher_model must be a path/hub id or a PreTrainedModel, got {type(teacher)}")
        kwargs["teacher_model"] = teacher
        return kwargs

    def build(self, exp: ExperimentConfig, **kwargs: Any) -> ResearchRun:  # type: ignore[override]
        run = super().build(exp, **kwargs)
        teacher = getattr(run.trainer, "teacher_model", None)
        if run.recorder is not None and teacher is not None:
            run.recorder.record["teacher_model"] = model_identity(
                run.trainer.accelerator.unwrap_model(teacher)
            )
            run.recorder._flush()
        return run
