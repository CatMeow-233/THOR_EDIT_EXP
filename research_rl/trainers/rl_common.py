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
  ``rollout/*`` statistics and the step skipping below. The reward rule is only valid for TRL's
  own advantages; it is not used once advantages were shaped (next item).
* **Advantage shaping** (``advantage_shapers=[...]`` constructor keyword, forwarded from the
  adapter's ``algo_kwargs``; registry names of :data:`~research_rl.core.registry.ADVANTAGE_SHAPERS`
  are accepted). Each :class:`AdvantageShaper` maps the generation batch TRL returned to
  :class:`ShapedAdvantages`. Shapers run in order right after TRL's
  ``_generate_and_score_completions`` (train and eval batches), before the signal flags are
  computed and the batch is logged, and ``output["advantages"]`` is replaced by the last result
  (shape, dtype, device and finiteness are checked; see :meth:`RLRolloutMixin._rr_check_advantages`).
  Signal with shapers: the last shaper's ``signal`` (AND completion mask non-empty) if it returns
  one, else "final advantage exactly non-zero" — conservative, a sample with any non-zero
  advantage is never treated as signal-free. A shaper that keeps TRL's zeros (e.g. a positive
  rescaling) can return :attr:`RLRolloutMixin.trl_advantage_signal`, today's reward-based flags.
  TRL's own completions table (``log_completions``, ``_logs["advantages"]``) is filled inside TRL's
  ``_generate_and_score_completions`` and so still records TRL's *unshaped* advantages; the shaped
  values are in :attr:`RLRolloutMixin.trajectory_buffer` and the ``rollout/*`` metrics.
* **Post-hoc modification detection.** A detached copy of the advantages the flags were computed
  from travels with the batch under :data:`ADVANTAGES_STAMP_KEY` (TRL's ``shuffle_sequence_dict``
  and ``split_tensor_dict`` permute and slice every tensor with ``ndim >= 1`` along dim 0, so it
  stays aligned with its samples; TRL's losses only read named keys). If a micro-batch's
  ``advantages`` differ from the stamp in ``_prepare_inputs`` (a subclass overriding
  ``_generate_and_score_completions`` on top of this mixin, a callback editing the buffered
  inputs, ...), the micro-batch's flags become "advantage exactly non-zero" and a
  ``UserWarning`` is emitted once per trainer.
* **Zero-advantage step skipping** (``OptimSpec.skip_zero_advantage_steps``). TRL generates once
  per ``steps_per_generation × num_iterations`` micro-batches, shuffles the generation batch and
  slices it into micro-batches. So the decision cannot be made at generation time. It is made per
  optimizer window instead. The flags travel with the samples through TRL's shuffle and split.
  ``_prepare_inputs`` ORs them over the micro-batches of the current gradient-accumulation window,
  i.e. exactly the samples that reach ``compute_loss``. At ``on_pre_optimizer_step`` (after HF
  clipping, before ``optimizer.step()``), a window with no signal on any process leads to
  ``HookedOptimizer.request_skip()``. The step still counts as nominal, so the LR schedule
  advances and ``global_step`` increments, but parameters and optimizer state stay bit-identical.
  A skip would also drop every gradient that does not come from the advantage, so
  ``skip_zero_advantage_steps=True`` is refused (``ValueError``) when such a term is active
  (checked against TRL 1.18's ``_compute_loss``): at construction for GRPO ``beta != 0`` (the KL
  term is added to the per-token loss), the GRPO entropy bonus (``entropy_coef != 0`` or
  ``use_adaptive_entropy``) and the MoE router aux loss of both trainers (TRL's
  ``aux_loss_enabled``, i.e. the config declares ``output_router_logits`` and the coefficient is
  non-zero, or ``output_router_logits=True`` in the model config); at ``on_train_begin`` (the
  HookedOptimizer exists by then) for any :class:`~research_rl.optim.UpdateRule` with
  ``needs_every_step=True``. RLOO ``beta != 0`` stays allowed: TRL subtracts the KL penalty from
  the *reward* before the leave-one-out baseline, so it is part of the advantage. All other GRPO
  terms (clipping, ``top_entropy_quantile`` mask, off-policy mask, vLLM IS ratios, every
  ``loss_type``) multiply the advantage.
* **Resume alignment.** TRL keeps the current generation batch (``_buffered_inputs``) and its
  micro-batch counter only in memory. After ``train(resume_from_checkpoint=...)`` it generates a
  fresh batch at once. The resumed run therefore matches the uninterrupted one only if the
  checkpoint lies on a generation boundary, i.e.
  ``(global_step × gradient_accumulation_steps) % (steps_per_generation × num_iterations) == 0``
  (:func:`resume_is_generation_aligned`). For example, with ``gradient_accumulation_steps=1``,
  ``save_steps`` must be a multiple of ``steps_per_generation × num_iterations``. Otherwise a
  ``UserWarning`` is emitted at the start of the resumed training. ResearchRL's own state
  (policy version, counters, HookedOptimizer) is restored either way, and so is the state of every
  advantage shaper that implements ``state_dict``/``load_state_dict``
  (:class:`~research_rl.core.checkpoint.StatefulComponent`; registered as research component
  ``f"advantage_shaper/{index}/{name}"``). A shaper without these methods restarts from its
  constructor state on resume.
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

import abc
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from transformers import TrainerCallback

from research_rl.core.checkpoint import StatefulComponent
from research_rl.core.config import ExperimentConfig
from research_rl.core.experiment import ExperimentRecorder
from research_rl.core.registry import ADVANTAGE_SHAPERS
from research_rl.rollout.policy_version import policy_lag, policy_version_of, stamp_policy_version
from research_rl.rollout.trajectory import RolloutBatch, TrajectoryBuffer
from research_rl.trainers.base import ResearchRun, TrainerAdapter

ROLLOUT_COMPONENT = "rl_rollouts"
# research_components key prefix of stateful advantage shapers: f"{prefix}/{index}/{name}".
ADVANTAGE_SHAPER_COMPONENT = "advantage_shaper"
# Per-sample bool (B,) stamped on generation batches: does the sample carry a learning signal?
SIGNAL_KEY = "rr_signal"
# Detached copy of the advantages SIGNAL_KEY was computed from, (B,) or (B, T): detects later edits.
ADVANTAGES_STAMP_KEY = "rr_signal_advantages"
# Per-token fields a GRPO rollout_func may return next to ``completion_ids``.
PER_TOKEN_ROLLOUT_KEYS = ("logprobs", "logprob_token_ids", "env_mask")

RolloutFunc = Callable[[list[Any], Any], dict[str, Any]]


# ---------------------------------------------------------------------------------- advantage shaping
@dataclass(frozen=True)
class ShapedAdvantages:
    """Result of an :class:`AdvantageShaper`.

    * ``advantages``: the new *local* advantages, ``(B,)`` (GRPO and RLOO) or ``(B, T)`` with
      ``T = completion_ids.shape[1]`` (GRPO only: its loss unsqueezes ``(B,)`` and uses ``(B, T)``
      per token; RLOO's loss is per sequence). Same dtype and device as TRL's advantages, finite.
      A tensor that requires grad is detached.
    * ``signal``: optional local ``(B,)`` bool, "does this sample carry a learning signal?". Only
      the last shaper's ``signal`` is used; ``None`` means "advantage exactly non-zero".
    """

    advantages: torch.Tensor
    signal: torch.Tensor | None = None


class AdvantageShaper(abc.ABC):
    """Rewrites the advantages of a generation batch (turn/step credit, sign rebalancing, per-token
    KL rewards, ...). See the module docstring for where it runs and how its ``signal`` is used.

    ``output`` is the local generation batch dict TRL's ``_generate_and_score_completions``
    returned (``prompt_ids``, ``completion_ids``, ``completion_mask``, ``advantages`` (already
    replaced by earlier shapers), ...); treat it as read-only. ``trainer`` is the research trainer:
    ``trainer.model.training`` tells train from eval batches, :attr:`RLRolloutMixin.trl_advantage_signal`
    holds the reward-based flags of TRL's own advantages. Register reusable shapers in
    :data:`~research_rl.core.registry.ADVANTAGE_SHAPERS` to refer to them by name.

    State across a resume: a shaper with ``state_dict()``/``load_state_dict(state)`` (the
    :class:`~research_rl.core.checkpoint.StatefulComponent` protocol) is checkpointed with the
    research components and restored on ``train(resume_from_checkpoint=...)``; the state must be
    loadable with ``torch.load(weights_only=True)`` (tensors, numbers, strings, lists, dicts). Any
    other shaper is not saved: it restarts from its constructor state.
    """

    name: str = "advantage_shaper"

    @abc.abstractmethod
    def __call__(self, output: Mapping[str, Any], trainer: Any) -> ShapedAdvantages:
        """Return the new advantages (and optionally per-sample signal flags)."""


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

    At ``on_train_begin`` it refuses ``needs_every_step`` update rules under step skipping and warns
    when a resumed run starts off a generation boundary.
    """

    def __init__(self, trainer: RLRolloutMixin) -> None:
        self._trainer = trainer

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        check = getattr(self._trainer, "_rr_check_every_step_rules", None)
        if check is not None:
            check()
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
    """Policy-version stamps, trajectory log, advantage shaping and zero-advantage skipping for TRL
    GRPO/RLOO trainers.

    Extra constructor keywords: ``skip_zero_advantage_steps`` (requires the hooked optimizer and
    no advantage-independent loss term, see the module docstring), ``trajectory_capacity`` (max
    generation batches kept in :attr:`trajectory_buffer`) and ``advantage_shapers`` (a sequence of
    :class:`AdvantageShaper` objects or :data:`~research_rl.core.registry.ADVANTAGE_SHAPERS` names,
    applied in order).
    """

    # Accepted ``advantages.ndim`` of the trainer's loss (RLOO: per-sequence only; GRPO adds (B, T)).
    _rr_advantage_ndims: ClassVar[tuple[int, ...]] = (1,)

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
        advantage_shapers: Sequence[AdvantageShaper | str] = (),
        **kwargs: Any,
    ) -> None:
        self.skip_zero_advantage_steps = bool(skip_zero_advantage_steps)
        self.trajectory_buffer = TrajectoryBuffer(trajectory_capacity)
        self.advantage_shapers: list[AdvantageShaper] = self._rr_resolve_shapers(advantage_shapers)
        #: Reward-based signal flags of TRL's *own* advantages for the batch being shaped (local (B,)
        #: bool, set only while the shapers run). A shaper that keeps TRL's zeros may return it.
        self.trl_advantage_signal: torch.Tensor | None = None
        self._rr_window_signal = False
        self._rr_window_microbatches = 0
        self._rr_rewards_per_func: torch.Tensor | None = None
        self._rr_warned_modified = False
        super().__init__(*args, **kwargs)
        if self.skip_zero_advantage_steps and not self._rr_hooked:
            raise ValueError(
                "skip_zero_advantage_steps needs the HookedOptimizer (OptimSpec.hooked=True): only it can "
                "turn an optimizer step into an exact no-op"
            )
        if self.skip_zero_advantage_steps:
            conflicts = self._rr_skip_conflicts()
            if conflicts:
                raise ValueError(
                    "skip_zero_advantage_steps would silently drop gradient that does not come from the "
                    "advantage: " + "; ".join(conflicts) + ". Disable these terms or the step skipping."
                )
        self.research_components.setdefault(ROLLOUT_COMPONENT, self.trajectory_buffer)
        for index, shaper in enumerate(self.advantage_shapers):
            if isinstance(shaper, StatefulComponent):  # its state travels with checkpoints
                key = f"{ADVANTAGE_SHAPER_COMPONENT}/{index}/{self._rr_shaper_name(shaper)}"
                self.research_components.setdefault(key, shaper)
        self.add_callback(_WindowCallback(self))  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ helpers
    @classmethod
    def _rr_resolve_shapers(cls, shapers: Any) -> list[AdvantageShaper]:
        """``advantage_shapers`` as a list of shaper instances (registry names are built).

        A single shaper or a single registry name is accepted as a one-element sequence (a bare
        string must not be iterated character by character).
        """
        if isinstance(shapers, (str, type, AdvantageShaper)):  # a class is rejected with a clear message
            shapers = [shapers]
        elif not isinstance(shapers, Sequence):
            raise TypeError(
                "advantage_shapers must be a sequence of AdvantageShaper instances or ADVANTAGE_SHAPERS "
                f"names, got {shapers!r}"
            )
        return [cls._rr_resolve_shaper(s) for s in shapers]

    @staticmethod
    def _rr_resolve_shaper(shaper: AdvantageShaper | str) -> AdvantageShaper:
        resolved = ADVANTAGE_SHAPERS.build(shaper) if isinstance(shaper, str) else shaper
        if isinstance(resolved, type):
            raise TypeError(
                f"advantage shaper {resolved.__name__} is a class; pass an instance "
                f"({resolved.__name__}(...)) or an ADVANTAGE_SHAPERS registry name"
            )
        if not callable(resolved):
            raise TypeError(f"advantage shaper {shaper!r} is not callable")
        return resolved  # type: ignore[no-any-return]

    @staticmethod
    def _rr_shaper_name(shaper: Any) -> str:
        return str(getattr(shaper, "name", type(shaper).__name__))

    def _rr_skip_conflicts(self) -> list[str]:
        """Active loss terms that do not depend on the advantage (a skipped step would drop them).

        Called after TRL's constructor, so it reads TRL's own flags. Both TRL 1.18 trainers add the
        MoE router aux loss when ``aux_loss_enabled`` (config declares ``output_router_logits`` and
        the coefficient is non-zero); ``output_router_logits=True`` in the model config is refused
        as well. The GRPO trainer adds its KL and entropy terms.
        """
        conflicts = []
        text_config = self.model.config.get_text_config()
        router_logits = getattr(text_config, "output_router_logits", False) is True
        if getattr(self, "aux_loss_enabled", False) or router_logits:
            conflicts.append(
                "the MoE router auxiliary loss is (or may be) active (model config output_router_logits="
                f"{getattr(text_config, 'output_router_logits', None)}, router_aux_loss_coef="
                f"{getattr(self, 'router_aux_loss_coef', None)}); set router_aux_loss_coef=0 in the TRL "
                "config and output_router_logits=False in the model config"
            )
        return conflicts

    def _rr_check_every_step_rules(self) -> None:
        """Raise if step skipping is on and a HookedOptimizer rule must act every step (train begin)."""
        if not self.skip_zero_advantage_steps:
            return
        hooked = self.hooked_optimizer  # type: ignore[attr-defined]
        rules = [] if hooked is None else hooked.rules
        needy = [self._rr_shaper_name(r) for r in rules if getattr(r, "needs_every_step", False)]
        if needy:
            raise ValueError(
                f"update rules {needy} have needs_every_step=True, but skip_zero_advantage_steps=True turns "
                "zero-signal steps into no-ops (only UpdateRule.on_skip runs). Disable the step skipping."
            )

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

    @staticmethod
    def _rr_exact_signal(advantages: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
        """Per sample: some advantage entry is exactly non-zero AND the completion mask is non-empty."""
        n = advantages.shape[0]
        exact_nonzero = (advantages != 0).reshape(n, -1).any(dim=1)
        return exact_nonzero & completion_mask.reshape(n, -1).bool().any(dim=1).to(exact_nonzero.device)

    def _rr_local_signal(
        self, mode: str, advantages: torch.Tensor, completion_mask: torch.Tensor
    ) -> torch.Tensor:
        """Local per-sample learning-signal flags of TRL's own advantages (see the module docstring)."""
        n_local = advantages.shape[0]
        signal = self._rr_exact_signal(advantages, completion_mask)
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

    def _rr_check_advantages(
        self, shaper: Any, value: Any, reference: torch.Tensor, output: Mapping[str, Any]
    ) -> torch.Tensor:
        """Validate a shaper's advantages against what this trainer's loss accepts; returns them detached.

        TRL 1.18: GRPO's ``_compute_loss`` unsqueezes ``(B,)`` advantages to ``(B, 1)`` and otherwise
        multiplies them with the ``(B, T)`` per-token terms, so ``(B,)`` and ``(B, T)`` work. RLOO's
        loss is per sequence (``coef_1`` is ``(B,)``), so only ``(B,)`` is accepted.
        """
        who = f"advantage shaper {self._rr_shaper_name(shaper)!r}"
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{who} returned advantages of type {type(value).__name__}, expected a torch.Tensor"
            )
        n = reference.shape[0]
        allowed: list[tuple[int, ...]] = [(n,)]
        if 2 in self._rr_advantage_ndims:
            allowed.append((n, int(output["completion_ids"].shape[1])))
        if tuple(value.shape) not in allowed:
            raise ValueError(
                f"{who} returned advantages of shape {tuple(value.shape)}; the {type(self).__name__} loss "
                f"accepts {' or '.join(str(s) for s in allowed)} (local batch size, completion length)"
            )
        if value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(
                f"{who} returned advantages with dtype {value.dtype} on {value.device}; expected "
                f"{reference.dtype} on {reference.device} (as TRL's advantages)"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{who} returned non-finite advantages")
        return value.detach()

    def _rr_apply_shapers(self, output: dict[str, Any]) -> torch.Tensor | None:
        """Run the shapers on ``output`` (in place); returns the last shaper's validated signal."""
        signal: torch.Tensor | None = None
        for shaper in self.advantage_shapers:
            who = f"advantage shaper {self._rr_shaper_name(shaper)!r}"
            result = shaper(output, self)
            if not isinstance(result, ShapedAdvantages):
                raise TypeError(f"{who} must return ShapedAdvantages, got {type(result).__name__}")
            reference = output["advantages"]
            output["advantages"] = self._rr_check_advantages(shaper, result.advantages, reference, output)
            signal = result.signal
            if signal is not None:
                n = reference.shape[0]
                if (
                    not isinstance(signal, torch.Tensor)
                    or signal.dtype != torch.bool
                    or tuple(signal.shape) != (n,)
                ):
                    raise ValueError(f"{who} must return signal=None or a bool tensor of shape ({n},)")
                signal = signal.detach().to(reference.device)
        return signal

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
        completion_mask = output["completion_mask"]
        # Always evaluated, also with shapers: it consumes per-generation captures (RLOO log-probs).
        signal = self._rr_local_signal(mode, output["advantages"], completion_mask)
        if self.advantage_shapers:
            self.trl_advantage_signal = signal
            try:
                shaper_signal = self._rr_apply_shapers(output)
            finally:
                self.trl_advantage_signal = None
            if shaper_signal is not None:
                non_empty = completion_mask.reshape(len(shaper_signal), -1).bool().any(dim=1)
                signal = shaper_signal & non_empty.to(shaper_signal.device)
            else:
                signal = self._rr_exact_signal(output["advantages"], completion_mask)
        advantages = output["advantages"]
        batch = RolloutBatch.from_tensors(
            prompt_ids=output["prompt_ids"],
            prompt_mask=output["prompt_mask"],
            completion_ids=output["completion_ids"],
            completion_mask=completion_mask,
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
        output[ADVANTAGES_STAMP_KEY] = advantages.detach().clone()
        stamp_policy_version(output, version)
        return output

    def _rr_microbatch_flags(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        """Signal flags of a training micro-batch; exact non-zero test if its advantages were edited."""
        advantages = inputs["advantages"]
        flags = inputs.get(SIGNAL_KEY)
        if flags is None:  # batch not produced by this mixin: exact test on the advantages
            return advantages != 0  # type: ignore[no-any-return]
        stamp = inputs.get(ADVANTAGES_STAMP_KEY)
        if stamp is None:
            return flags  # type: ignore[no-any-return]
        unchanged = (
            isinstance(advantages, torch.Tensor)
            and advantages.shape == stamp.shape
            and advantages.dtype == stamp.dtype
            and advantages.device == stamp.device
            and torch.equal(advantages, stamp)
        )
        if unchanged:
            return flags  # type: ignore[no-any-return]
        if not self._rr_warned_modified:
            self._rr_warned_modified = True
            warnings.warn(
                "the advantages of a training micro-batch differ from the ones its learning-signal flags were "
                "computed from (modified after RLRolloutMixin._generate_and_score_completions, e.g. by a "
                "subclass or a callback); using 'advantage exactly non-zero' flags for such micro-batches. "
                "Use advantage_shapers=[...] to shape advantages with an explicit signal.",
                UserWarning,
                stacklevel=2,
            )
        return self._rr_exact_signal(advantages, inputs["completion_mask"])

    def _prepare_inputs(self, generation_batch: Any) -> Any:
        inputs = super()._prepare_inputs(generation_batch)  # type: ignore[misc]
        if self.model.training and isinstance(inputs, Mapping) and "advantages" in inputs:
            self._rr_window_microbatches += 1
            flags = self._rr_microbatch_flags(inputs)
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
    (``reward_funcs``, ``rollout_func``, ``peft_config``, ``trajectory_capacity``,
    ``advantage_shapers``, ...) go to the
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
