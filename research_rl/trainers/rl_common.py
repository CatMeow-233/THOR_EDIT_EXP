"""Shared machinery of the RL adapters (GRPO, RLOO).

:class:`RLRolloutMixin` is mixed *before* :class:`~research_rl.trainers.base.ResearchTrainerMixin`
into a TRL trainer class (``GRPOTrainer`` / ``RLOOTrainer``)::

    class ResearchGRPOTrainer(RLRolloutMixin, ResearchTrainerMixin, GRPOTrainer): ...

It changes nothing in how TRL generates, scores or optimises. It adds:

* **Policy-version stamps.** ``_generate_and_score_completions`` (TRL's, called via ``super``)
  is wrapped. The returned generation batch gets the number of *applied* parameter updates
  (``HookedOptimizer.applied_updates``) under ``"policy_version"``. ``_prepare_inputs`` logs the
  policy lag (current version − stamp) of every micro-batch actually trained on. The lag is
  non-zero with ``steps_per_generation > gradient_accumulation_steps`` or ``num_iterations > 1``.
* **A bounded trajectory log** (:class:`~research_rl.rollout.trajectory.TrajectoryBuffer`) of
  every training generation batch: ids, masks, rewards, advantages, version, and the per-batch
  statistics mean reward, fraction of zero-advantage samples, and whether any advantage is
  non-zero. The same statistics are logged as ``rollout/*`` metrics. The buffer's counters are
  registered as a research component and so travel with checkpoints.
* **Learning-signal flags.** Every sample of a generation batch gets a boolean under
  :data:`SIGNAL_KEY`: does it carry a learning signal? A sample has *no* signal if its advantage
  is mathematically zero, or exactly zero in TRL's tensor, or its completion mask is all zero.
  "Mathematically zero" is decided from the rewards TRL normalised, not from the float32
  advantage: TRL's float32 arithmetic (``(r - mean) / (std + 1e-4)`` for GRPO, ``(sum - r) / (k-1)``
  for RLOO) turns a constant group such as eight rewards of 0.1 into round-off advantages of up to
  ~1e-3, and Adam (scale-invariant) would turn that noise into full learning-rate updates. The
  per-algorithm rules live in the trainer classes (``_rr_reward_signal``). In short, a sample
  has signal iff its (weighted, clipped, KL-shaped where applicable) reward differs from its
  group's mean, recomputed in float64, where a constant group's mean is exact. If the rule cannot
  be evaluated, the flag falls back to "advantage exactly non-zero". The flags feed the
  ``rollout/*`` statistics and the step skipping below.
* **Zero-advantage step skipping** (``OptimSpec.skip_zero_advantage_steps``). TRL generates once
  per ``steps_per_generation × num_iterations`` micro-batches, shuffles the generation batch and
  slices it into micro-batches. So the decision cannot be made at generation time. It is made per
  optimizer window instead. The flags travel with the samples through TRL's shuffle and split.
  ``_prepare_inputs`` ORs them over the micro-batches of the current gradient-accumulation window,
  i.e. exactly the samples that reach ``compute_loss``. At ``on_pre_optimizer_step`` (after HF
  clipping, before ``optimizer.step()``), a window with no signal on any process leads to
  ``HookedOptimizer.request_skip()``. The step still counts as nominal, so the LR schedule
  advances and ``global_step`` increments, but parameters and optimizer state stay bit-identical.
  Caveat: the skip also drops gradient from loss terms that do not depend on the advantage
  (GRPO ``beta > 0`` KL term, entropy bonus, MoE aux loss).
* **Resume alignment.** TRL keeps the current generation batch (``_buffered_inputs``) and its
  micro-batch counter only in memory. After ``train(resume_from_checkpoint=...)`` it generates a
  fresh batch at once. The resumed run therefore matches the uninterrupted one only if the
  checkpoint lies on a generation boundary, i.e.
  ``(global_step × gradient_accumulation_steps) % (steps_per_generation × num_iterations) == 0``
  (:func:`resume_is_generation_aligned`). For example, with ``gradient_accumulation_steps=1``,
  ``save_steps`` must be a multiple of ``steps_per_generation × num_iterations``. Otherwise a
  ``UserWarning`` is emitted at the start of the resumed training. ResearchRL's own state
  (policy version, counters, HookedOptimizer) is restored either way.
* :func:`eos_truncating_rollout`, an *opt-in* wrapper for GRPO ``rollout_func``. TRL takes
  rollout token lists verbatim: the completion mask is 1 on every returned token, including
  tokens after an EOS. It treats a row whose last token is EOS **or pad** as terminated (see the
  tests). The wrapper cuts each completion after its first EOS and strips trailing pad tokens.
  The mask is then the usual "1 up to and including the first EOS" mask, and
  ``mask_truncated_completions`` behaves as documented.

Not tested here (NOT_TESTED): multi-process / DeepSpeed / FSDP runs. The skip decision is
all-gathered across processes, but HookedOptimizer itself is not DeepSpeed-aware. vLLM
generation is also not tested.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

import torch
from transformers import TrainerCallback

from research_rl.core.config import ExperimentConfig
from research_rl.core.experiment import ExperimentRecorder
from research_rl.rollout.policy_version import policy_lag, policy_version_of, stamp_policy_version
from research_rl.rollout.trajectory import RolloutBatch, TrajectoryBuffer
from research_rl.trainers.base import ResearchRun, TrainerAdapter

ROLLOUT_COMPONENT = "rl_rollouts"
# Per-sample bool (B,) stamped on generation batches: does the sample carry a learning signal?
SIGNAL_KEY = "rr_signal"
# Per-token fields a GRPO rollout_func may return next to ``completion_ids``.
PER_TOKEN_ROLLOUT_KEYS = ("logprobs", "logprob_token_ids", "env_mask")

RolloutFunc = Callable[[list[Any], Any], dict[str, Any]]


# ---------------------------------------------------------------------------------- EOS handling
def eos_cut_length(ids: Sequence[int], eos_token_ids: Sequence[int], pad_token_id: int | None) -> int:
    """Length of ``ids`` after cutting at the first EOS (kept) or stripping trailing pad tokens."""
    eos = set(int(e) for e in eos_token_ids)
    for i, tok in enumerate(ids):
        if int(tok) in eos:
            return i + 1
    n = len(ids)
    if pad_token_id is not None:
        while n > 0 and int(ids[n - 1]) == int(pad_token_id):
            n -= 1
    return n


def eos_truncating_rollout(
    rollout_func: RolloutFunc,
    eos_token_ids: Sequence[int] | None = None,
    pad_token_id: int | None = None,
) -> RolloutFunc:
    """Wrap a GRPO ``rollout_func`` so each completion ends at its first EOS (see module docstring).

    ``eos_token_ids``/``pad_token_id`` default to the trainer's (``trainer.eos_token_ids`` and the
    tokenizer's pad id) at call time. Per-token fields in :data:`PER_TOKEN_ROLLOUT_KEYS` are cut
    to the same length. A completion that would become empty (only pad tokens, no EOS) raises
    ``ValueError``: TRL cannot represent an empty completion (it indexes ``ids[-1]``).
    """

    def wrapped(prompts: list[Any], trainer: Any) -> dict[str, Any]:
        output = dict(rollout_func(prompts, trainer))
        eos = list(eos_token_ids) if eos_token_ids is not None else list(trainer.eos_token_ids)
        pad = pad_token_id
        if pad is None:
            tok = getattr(trainer, "_tokenizer", None) or trainer.processing_class
            pad = getattr(tok, "pad_token_id", None)
        rows = output["completion_ids"]
        lengths = [eos_cut_length(row, eos, pad) for row in rows]
        for i, n in enumerate(lengths):
            if n == 0:
                raise ValueError(f"rollout completion {i} is empty after removing padding (no EOS, only pad)")
        output["completion_ids"] = [list(row[:n]) for row, n in zip(rows, lengths)]
        for key in PER_TOKEN_ROLLOUT_KEYS:
            values = output.get(key)
            if values is None:
                continue
            if len(values) != len(rows):
                raise ValueError(
                    f"rollout field {key!r} has {len(values)} rows, completion_ids has {len(rows)}"
                )
            output[key] = [list(v[:n]) for v, n in zip(values, lengths)]
        return output

    wrapped.__name__ = f"eos_truncating_{getattr(rollout_func, '__name__', 'rollout')}"
    return wrapped


# ---------------------------------------------------------------------------------- signal / resume
def group_deviation_signal(rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
    """Per sample: does its reward differ from the mean of its group? (NaN = unscorable -> ``False``).

    ``rewards`` is the flat (gathered) reward vector, ``num_generations`` consecutive samples per
    group. The comparison is done in float64. The float64 sum of up to 2**29 equal float32 values
    is exact, so the mean of a constant group equals its members exactly and the result is
    ``False`` for every member. TRL's float32 ``r - mean`` can be a few ulp instead.
    """
    if num_generations < 1 or rewards.numel() % num_generations:
        raise ValueError(f"{rewards.numel()} rewards cannot be grouped by num_generations={num_generations}")
    grouped = rewards.detach().to(torch.float64).view(-1, num_generations)
    valid = ~torch.isnan(grouped)
    counts = valid.sum(dim=1, keepdim=True).clamp(min=1)
    mean = torch.nansum(grouped, dim=1, keepdim=True) / counts
    return ((grouped != mean) & valid).view(-1)


def resume_is_generation_aligned(
    global_step: int, gradient_accumulation_steps: int, steps_per_generation: int, num_iterations: int
) -> bool:
    """Is ``global_step`` a generation boundary of TRL's GRPO/RLOO buffering?

    TRL counts micro-batches and generates when the count is a multiple of
    ``steps_per_generation * num_iterations``. After ``global_step`` optimizer steps the count is
    ``global_step * gradient_accumulation_steps``. Only a resume from such a step reproduces the
    uninterrupted run (see the module docstring).
    """
    return (int(global_step) * int(gradient_accumulation_steps)) % (
        int(steps_per_generation) * int(num_iterations)
    ) == 0


# ---------------------------------------------------------------------------------- trainer mixin
class _WindowCallback(TrainerCallback):
    """Calls the trainer's end-of-window decision right before ``optimizer.step()``.

    Also warns at ``on_train_begin`` when a resumed run starts off a generation boundary.
    """

    def __init__(self, trainer: RLRolloutMixin) -> None:
        self._trainer = trainer

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        step = int(state.global_step or 0)
        spg = int(getattr(args, "steps_per_generation", None) or args.gradient_accumulation_steps)
        num_iterations = int(getattr(self._trainer, "num_iterations", 1) or 1)
        if step > 0 and not resume_is_generation_aligned(
            step, args.gradient_accumulation_steps, spg, num_iterations
        ):
            warnings.warn(
                f"resuming at global_step {step}, which is not a generation boundary "
                f"(gradient_accumulation_steps={args.gradient_accumulation_steps}, "
                f"steps_per_generation={spg}, num_iterations={num_iterations}): TRL does not checkpoint "
                "its buffered generation batch and regenerates now, so the resumed run will differ from "
                "an uninterrupted one. Save at multiples of the generation block to resume exactly.",
                UserWarning,
                stacklevel=2,
            )

    def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._trainer._rr_end_of_window()


class RLRolloutMixin:
    """Policy-version stamps, trajectory log and zero-advantage skipping for TRL GRPO/RLOO trainers.

    Extra constructor keywords: ``skip_zero_advantage_steps`` (requires the hooked optimizer) and
    ``trajectory_capacity`` (max generation batches kept in :attr:`trajectory_buffer`).
    """

    # Attributes provided by ResearchTrainerMixin / the TRL trainer (declared for type checkers).
    _rr_hooked: bool
    research_components: dict[str, Any]
    _metrics: dict[str, Any]
    accelerator: Any
    model: Any
    state: Any
    reward_weights: torch.Tensor
    num_generations: int
    num_generations_eval: int

    def __init__(
        self,
        *args: Any,
        skip_zero_advantage_steps: bool = False,
        trajectory_capacity: int = 16,
        **kwargs: Any,
    ) -> None:
        self.skip_zero_advantage_steps = bool(skip_zero_advantage_steps)
        self.trajectory_buffer = TrajectoryBuffer(trajectory_capacity)
        self._rr_window_signal = False
        self._rr_window_microbatches = 0
        self._rr_rewards_per_func: torch.Tensor | None = None
        super().__init__(*args, **kwargs)
        if self.skip_zero_advantage_steps and not self._rr_hooked:
            raise ValueError(
                "skip_zero_advantage_steps needs the HookedOptimizer (OptimSpec.hooked=True): only it can "
                "turn an optimizer step into an exact no-op"
            )
        self.research_components.setdefault(ROLLOUT_COMPONENT, self.trajectory_buffer)
        self.add_callback(_WindowCallback(self))  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ helpers
    @property
    def policy_version(self) -> int:
        """Applied parameter updates so far (see :func:`research_rl.rollout.policy_version_of`)."""
        return policy_version_of(self)

    def _rr_log(self, mode: str, key: str, value: float) -> None:
        self._metrics[mode][key].append(float(value))

    def _rr_num_generations(self, mode: str) -> int:
        return int(self.num_generations if mode == "train" else self.num_generations_eval)

    def _rr_weighted_rewards(self) -> torch.Tensor | None:
        """Gathered weighted reward sum per sample (as TRL's ``sum_then_normalize``); NaN = unscorable."""
        rpf = self._rr_rewards_per_func
        if rpf is None:
            return None
        rewards = (rpf * self.reward_weights.to(rpf.device).unsqueeze(0)).nansum(dim=1)
        rewards[torch.isnan(rpf).all(dim=1)] = float("nan")
        return rewards

    def _rr_begin_generation(self) -> None:
        """Hook run before TRL's ``_generate_and_score_completions`` (for capturing intermediates)."""

    def _rr_reward_signal(self, mode: str, completion_mask: torch.Tensor) -> torch.Tensor | None:
        """Gathered per-sample bool: is the advantage *mathematically* non-zero? ``None`` = unknown.

        Overridden by the GRPO/RLOO trainers with the rule matching their advantage estimator.
        ``completion_mask`` is the local mask. ``None`` makes the flag fall back to ``advantages != 0``.
        """
        return None

    def _rr_local_signal(
        self, mode: str, advantages: torch.Tensor, completion_mask: torch.Tensor
    ) -> torch.Tensor:
        """Local per-sample learning-signal flags (see the module docstring)."""
        n_local = advantages.shape[0]
        exact_nonzero = (advantages != 0).reshape(n_local, -1).any(dim=1)
        signal = exact_nonzero & completion_mask.reshape(n_local, -1).bool().any(dim=1)
        reward_signal = self._rr_reward_signal(mode, completion_mask)
        if reward_signal is not None:
            start = self.accelerator.process_index * n_local
            signal = signal & reward_signal[start : start + n_local].to(signal.device)
        return signal

    def _rr_local_rewards(self, n_local: int, device: torch.device) -> torch.Tensor:
        """Weighted reward per local sample, NaN where every reward function returned None."""
        rewards = self._rr_weighted_rewards()
        if rewards is None:
            return torch.full((n_local,), float("nan"), device=device)
        start = self.accelerator.process_index * n_local
        return rewards[start : start + n_local]

    # ------------------------------------------------------------------ TRL overrides
    def _calculate_rewards(
        self, inputs: Any, prompts: Any, completions: Any, completion_ids_list: Any
    ) -> Any:
        rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)  # type: ignore[misc]
        self._rr_rewards_per_func = rewards_per_func.detach()
        return rewards_per_func

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        version = self.policy_version
        self._rr_rewards_per_func = None
        self._rr_begin_generation()
        output: dict[str, Any] = super()._generate_and_score_completions(inputs)  # type: ignore[misc]
        mode = "train" if self.model.training else "eval"
        advantages = output["advantages"]
        signal = self._rr_local_signal(mode, advantages, output["completion_mask"])
        batch = RolloutBatch.from_tensors(
            prompt_ids=output["prompt_ids"],
            prompt_mask=output["prompt_mask"],
            completion_ids=output["completion_ids"],
            completion_mask=output["completion_mask"],
            rewards=self._rr_local_rewards(len(inputs), advantages.device),
            advantages=advantages,
            policy_version=version,
            global_step=int(self.state.global_step),
            mode=mode,
            signal=signal,
        )
        if mode == "train":
            self.trajectory_buffer.append(batch)
        self._rr_log(mode, "rollout/policy_version", version)
        self._rr_log(mode, "rollout/frac_zero_advantage", batch.frac_zero_advantage)
        self._rr_log(mode, "rollout/has_nonzero_advantage", float(batch.has_nonzero_advantage))
        # Appended last: TRL sizes a batch dict from its first value.
        output[SIGNAL_KEY] = signal
        stamp_policy_version(output, version)
        return output

    def _prepare_inputs(self, generation_batch: Any) -> Any:
        inputs = super()._prepare_inputs(generation_batch)  # type: ignore[misc]
        if self.model.training and isinstance(inputs, Mapping) and "advantages" in inputs:
            self._rr_window_microbatches += 1
            flags = inputs.get(SIGNAL_KEY)
            if flags is None:  # batch not produced by this mixin: exact test on the advantages
                flags = inputs["advantages"] != 0
            if not self._rr_window_signal and bool(flags.any()):
                self._rr_window_signal = True
            lag = policy_lag(self.policy_version, inputs)
            if lag is not None:
                self._rr_log("train", "rollout/policy_lag", lag)
        return inputs

    # ------------------------------------------------------------------ window decision
    def _rr_end_of_window(self) -> None:
        """Called right before ``optimizer.step()``: decide whether this step is skipped."""
        has_signal = self._rr_window_signal
        if self.accelerator.num_processes > 1:  # every rank must take the same decision
            flag = torch.tensor([1.0 if has_signal else 0.0], device=self.accelerator.device)
            has_signal = bool(self.accelerator.gather(flag).max() > 0)
        skip = self.skip_zero_advantage_steps and not has_signal and self._rr_window_microbatches > 0
        if skip:
            hooked = self.hooked_optimizer  # type: ignore[attr-defined]
            assert hooked is not None  # guaranteed by the constructor check
            hooked.request_skip()
        self.trajectory_buffer.record_window(has_signal=has_signal, skipped=skip)
        self._rr_log("train", "rollout/window_has_signal", float(has_signal))
        self._rr_window_signal = False
        self._rr_window_microbatches = 0


# ---------------------------------------------------------------------------------- recording
class RolloutRecordCallback(TrainerCallback):
    """Copies the trajectory-buffer counters into ``record.json`` under ``"rollout"``."""

    def __init__(self, recorder: ExperimentRecorder, buffer: TrajectoryBuffer) -> None:
        self.recorder = recorder
        self.buffer = buffer

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.recorder.set_extra("rollout", self.buffer.counters())


class RLTrainerAdapter(TrainerAdapter):
    """Base for RL adapters: ``trainer_class()`` returns :attr:`research_trainer_cls`.

    ``OptimSpec.skip_zero_advantage_steps`` is forwarded to the trainer. Extra ``algo_kwargs``
    (``reward_funcs``, ``rollout_func``, ``peft_config``, ``trajectory_capacity``, ...) go to the
    trainer unchanged. ``bf16`` defaults to ``False``: TRL's configs enable bf16 autocast
    whenever ``fp16`` is unset, also on CPU. Precision must be requested explicitly.
    """

    is_reinforcement_learning: ClassVar[bool] = True
    default_args: ClassVar[dict[str, Any]] = {"bf16": False}
    research_trainer_cls: ClassVar[type]

    def trainer_class(self) -> type:
        return self.research_trainer_cls

    def trainer_kwargs(self, exp: ExperimentConfig, **algo_kwargs: Any) -> dict[str, Any]:
        if "skip_zero_advantage_steps" in algo_kwargs:
            raise ValueError("set skip_zero_advantage_steps in ExperimentConfig.optim, not as a keyword")
        return {**algo_kwargs, "skip_zero_advantage_steps": exp.optim.skip_zero_advantage_steps}

    def build(self, exp: ExperimentConfig, **kwargs: Any) -> ResearchRun:
        run = super().build(exp, **kwargs)
        run.info["is_reinforcement_learning"] = True
        if run.recorder is not None:
            run.recorder.set_extra("rollout", run.trainer.trajectory_buffer.counters())
            run.trainer.add_callback(RolloutRecordCallback(run.recorder, run.trainer.trajectory_buffer))
        return run
