"""Glue between ResearchRL and TRL trainers.

* :class:`ResearchTrainerMixin` – mixed into any TRL trainer class; replaces the optimizer that
  TRL/Transformers creates with a :class:`HookedOptimizer` (works for every TRL trainer because
  it overrides ``create_optimizer`` instead of relying on ``optimizer_cls_and_kwargs``, which
  only ``SFTTrainer`` forwards), restores registered components on resume.
* :class:`ResearchCallback` – writes TRL log events, step counters and checkpoint extras to the
  :class:`ExperimentRecorder`.
* :class:`TrainerAdapter` – maps an :class:`ExperimentConfig` to a TRL ``*Config`` + trainer.
  Concrete adapters (``sft``, ``dpo``, ``grpo``, ``rloo``, ``distillation``) live next to this
  file and register themselves in :data:`research_rl.core.registry.TRAINERS`.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, TrainerCallback

from research_rl.core.checkpoint import (
    StatefulComponent,
    latest_checkpoint,
    load_components,
    load_research_state,
    save_components,
    save_research_state,
)
from research_rl.core.config import ExperimentConfig, ModelSpec
from research_rl.core.experiment import ExperimentRecorder
from research_rl.optim.optimizer_wrapper import HookedOptimizer, unwrap_hooked
from research_rl.optim.update_interface import UpdateRule

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def resolve_dtype(name: str) -> torch.dtype | str:
    return "auto" if name == "auto" else _DTYPES[name]


def load_causal_lm(spec: ModelSpec) -> PreTrainedModel:
    return AutoModelForCausalLM.from_pretrained(
        spec.name_or_path,
        revision=spec.revision,
        dtype=resolve_dtype(spec.dtype),
        trust_remote_code=spec.trust_remote_code,
    )


def load_tokenizer(spec: ModelSpec) -> Any:
    return AutoTokenizer.from_pretrained(
        spec.name_or_path, revision=spec.revision, trust_remote_code=spec.trust_remote_code
    )


class ResearchTrainerMixin:
    """Mix *before* a TRL trainer class: ``class X(ResearchTrainerMixin, GRPOTrainer)``."""

    def __init__(
        self,
        *args: Any,
        update_rules: Sequence[UpdateRule] = (),
        clip_in_wrapper: float | None = None,
        hooked_optimizer: bool = True,
        components: Mapping[str, StatefulComponent] | None = None,
        **kwargs: Any,
    ) -> None:
        self._rr_rules = list(update_rules)
        self._rr_clip = clip_in_wrapper
        self._rr_hooked = hooked_optimizer or bool(update_rules) or clip_in_wrapper is not None
        self.research_components: dict[str, StatefulComponent] = dict(components or {})
        if clip_in_wrapper is not None:
            training_args = kwargs.get("args")
            if training_args is not None and training_args.max_grad_norm not in (0, 0.0, None):
                raise ValueError(
                    "clip_in_wrapper is set: set the TRL config's max_grad_norm=0 so gradients are not "
                    "clipped twice (once by HF on unmasked grads, once inside HookedOptimizer)"
                )
        super().__init__(*args, **kwargs)

    def create_optimizer(self, model: Any = None) -> Any:
        super().create_optimizer(model)  # type: ignore[misc]
        opt = self.optimizer  # type: ignore[has-type]
        if self._rr_hooked and not isinstance(opt, HookedOptimizer):
            target = self.model if model is None else model  # type: ignore[attr-defined]
            hooked = HookedOptimizer(opt, target.named_parameters(), self._rr_rules, clip_norm=self._rr_clip)
            # HF clips *before* optimizer.step(); rules that need unclipped gradients check this flag.
            hooked.step_extras["hf_max_grad_norm"] = float(self.args.max_grad_norm or 0.0)  # type: ignore[attr-defined]
            self.optimizer = hooked
        return self.optimizer

    @property
    def hooked_optimizer(self) -> HookedOptimizer | None:
        return unwrap_hooked(getattr(self, "optimizer", None))

    def train(self, resume_from_checkpoint: str | bool | None = None, **kwargs: Any) -> Any:
        if resume_from_checkpoint is True:
            resume_from_checkpoint = latest_checkpoint(self.args.output_dir)  # type: ignore[attr-defined]
            if resume_from_checkpoint is None:
                raise ValueError(f"no checkpoint found in {self.args.output_dir}")  # type: ignore[attr-defined]
        if isinstance(resume_from_checkpoint, str):
            load_research_state(resume_from_checkpoint)  # validates config_version
            load_components(resume_from_checkpoint, self.research_components)
        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)  # type: ignore[misc]


_CLASS_CACHE: dict[type, type] = {}


def research_trainer_class(trl_cls: type) -> type:
    """Return ``ResearchTrainerMixin`` combined with ``trl_cls`` (cached)."""
    if issubclass(trl_cls, ResearchTrainerMixin):
        return trl_cls
    if trl_cls not in _CLASS_CACHE:
        _CLASS_CACHE[trl_cls] = type(f"Research{trl_cls.__name__}", (ResearchTrainerMixin, trl_cls), {})
    return _CLASS_CACHE[trl_cls]


class ResearchCallback(TrainerCallback):
    """Feeds the experiment recorder and writes ResearchRL extras into each checkpoint."""

    def __init__(self, recorder: ExperimentRecorder | None = None) -> None:
        self.recorder = recorder
        self.trainer: Any = None
        self.extra_state: dict[str, Any] = {}  # algorithm-specific JSON state (set by adapters)

    def bind(self, trainer: Any) -> None:
        self.trainer = trainer

    def _counters(self, state: Any) -> dict[str, int]:
        opt = unwrap_hooked(getattr(self.trainer, "optimizer", None)) if self.trainer is not None else None
        if opt is None:
            return {
                "nominal": int(state.global_step),
                "applied": int(state.global_step),
                "skipped": 0,
                "global_step": int(state.global_step),
            }
        return {
            "nominal": opt.nominal_steps,
            "applied": opt.applied_updates,
            "skipped": opt.skipped_steps,
            "global_step": int(state.global_step),
        }

    def on_train_begin(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        # HF has already restored trainer_state.json here when resuming from a checkpoint.
        if self.recorder is not None and int(state.global_step) > 0:
            self.recorder.resume_from(int(state.global_step))

    def on_log(
        self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kw: Any
    ) -> None:
        if self.recorder is None or not logs:
            return
        counters = self._counters(state)
        self.recorder.log_metrics(
            state.global_step,
            {**logs, "rr/applied_updates": counters["applied"], "rr/skipped_steps": counters["skipped"]},
        )
        self.recorder.set_steps(**counters)

    def on_save(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        counters = self._counters(state)
        save_research_state(
            ckpt,
            {
                "global_step": int(state.global_step),
                "epoch": float(state.epoch or 0.0),
                "policy_version": counters["applied"],
                "steps": counters,
                "algorithm_state": dict(self.extra_state),
            },
        )
        if self.trainer is not None:
            save_components(ckpt, getattr(self.trainer, "research_components", {}))
        if self.recorder is not None:
            self.recorder.set_steps(**counters)
            self.recorder.add_checkpoint(ckpt, state.global_step)

    def on_train_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        if self.recorder is not None:
            self.recorder.set_steps(**self._counters(state))
            self.recorder.finalize()


@dataclass
class ResearchRun:
    """A configured run: the TRL trainer plus its recorder."""

    config: ExperimentConfig
    trainer: Any
    recorder: ExperimentRecorder | None
    callback: ResearchCallback
    info: dict[str, Any] = field(default_factory=dict)

    def train(self, resume_from_checkpoint: str | bool | None = None) -> dict[str, Any]:
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        if self.recorder is None:
            return {}
        return ExperimentRecorder.load(self.recorder.run_dir)

    @property
    def hooked_optimizer(self) -> HookedOptimizer | None:
        return unwrap_hooked(getattr(self.trainer, "optimizer", None))


class TrainerAdapter:
    """Base adapter. Subclasses set the class variables and may override ``trainer_kwargs``."""

    algorithm: ClassVar[str]
    trl_trainer_cls: ClassVar[type]
    trl_config_cls: ClassVar[type]
    # Honest labelling: SFT, DPO and distillation are *not* reinforcement learning.
    is_reinforcement_learning: ClassVar[bool] = False
    default_args: ClassVar[dict[str, Any]] = {}

    def build_trl_config(self, exp: ExperimentConfig) -> Any:
        args: dict[str, Any] = {
            "report_to": [],
            "save_strategy": "no",
            "logging_steps": 1,
            "use_cpu": not torch.cuda.is_available(),
            **self.default_args,
            **exp.trainer.args,
        }
        args["output_dir"] = str(exp.run_dir / "trainer")
        args["seed"] = exp.seed
        args.setdefault("data_seed", exp.seed)
        if exp.optim.clip_in_wrapper is not None:
            args["max_grad_norm"] = 0.0
        valid = {f.name for f in dataclasses.fields(self.trl_config_cls)}
        unknown = set(args) - valid
        if unknown:
            raise ValueError(f"unknown {self.trl_config_cls.__name__} arguments: {sorted(unknown)}")
        return self.trl_config_cls(**args)

    def trainer_kwargs(self, exp: ExperimentConfig, **algo_kwargs: Any) -> dict[str, Any]:
        """Algorithm-specific keyword arguments for the TRL trainer constructor."""
        return dict(algo_kwargs)

    def trainer_class(self) -> type:
        return research_trainer_class(self.trl_trainer_cls)

    def build(
        self,
        exp: ExperimentConfig,
        *,
        train_dataset: Any,
        model: PreTrainedModel | str | None = None,
        processing_class: Any = None,
        eval_dataset: Any = None,
        update_rules: Sequence[UpdateRule] = (),
        components: Mapping[str, StatefulComponent] | None = None,
        callbacks: Sequence[TrainerCallback] = (),
        record: bool = True,
        **algo_kwargs: Any,
    ) -> ResearchRun:
        if exp.trainer.algorithm != self.algorithm:
            raise ValueError(f"config asks for {exp.trainer.algorithm!r}, adapter is {self.algorithm!r}")
        torch.manual_seed(exp.seed)
        # A model path passed here overrides exp.model for weights, tokenizer and the recorded identity.
        spec = exp.model
        if isinstance(model, str):
            spec = dataclasses.replace(exp.model, name_or_path=model, revision=None)
        if model is None or isinstance(model, str):
            model = load_causal_lm(spec)
        if processing_class is None:
            processing_class = load_tokenizer(spec)
        recorder = ExperimentRecorder(exp) if record else None
        callback = ResearchCallback(recorder)
        trainer = self.trainer_class()(
            model=model,
            args=self.build_trl_config(exp),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=[callback, *callbacks],
            update_rules=update_rules,
            clip_in_wrapper=exp.optim.clip_in_wrapper,
            hooked_optimizer=exp.optim.hooked,
            components=components,
            **self.trainer_kwargs(exp, **algo_kwargs),
        )
        callback.bind(trainer)
        if recorder is not None:
            recorder.set_model(trainer.model, spec.revision)
            recorder.set_dataset(train_dataset, exp.data.name)
        return ResearchRun(config=exp, trainer=trainer, recorder=recorder, callback=callback)
