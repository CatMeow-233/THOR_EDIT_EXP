"""Capability retention: does RL(VR) training degrade other capabilities, and does recovery help?

Research setting: a base policy is trained (e.g. RLVR on math), then optionally "recovered"
(replay, SFT on general data, merging, ...). A :class:`RetentionSuite` holds small probe tasks,
each measuring one capability, and evaluates any model on them with deterministic greedy
decoding. :class:`RetentionReport` turns the scores of ``base -> after_training -> after_recovery``
into drop / retention / recovery numbers, and :func:`logit_kl` measures how far the next-token
distributions on the probe prompts moved (``KL(base || current)``), which detects drift before it
shows up in task accuracy.

What is implemented: greedy generation via ``model.generate`` (bounded ``max_new_tokens``,
truncation at the tokenizer's EOS), per-capability mean scores, report arithmetic, logit-level KL,
and logging into :class:`~research_rl.core.experiment.ExperimentRecorder`. Probe *content* (which
prompts measure which capability) is the researcher's responsibility; nothing here claims that the
tiny test probes measure anything about real LLM capabilities.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from transformers import GenerationConfig

from research_rl.evaluation.metrics import exact_match, normalize_answer

Scorer = Callable[[str, str], float]  # (prediction, reference answer) -> score


def exact_match_scorer(prediction: str, answer: str) -> float:
    """1.0 if the whitespace-normalised prediction equals the answer, else 0.0."""
    return 1.0 if exact_match(prediction, answer) else 0.0


def prefix_match_scorer(prediction: str, answer: str) -> float:
    """1.0 if the whitespace-normalised prediction *starts with* the answer.

    Useful when generation runs to ``max_new_tokens`` without emitting EOS.
    """
    return 1.0 if normalize_answer(prediction).startswith(normalize_answer(answer)) else 0.0


@dataclass(frozen=True)
class ProbeTask:
    """One capability probe: ``(prompt, answer)`` examples scored by ``scorer``."""

    name: str
    examples: tuple[tuple[str, str], ...]
    scorer: Scorer = exact_match_scorer
    max_new_tokens: int | None = None  # None: use the suite default

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ProbeTask needs a non-empty name")
        examples = tuple((str(p), str(a)) for p, a in self.examples)
        if not examples:
            raise ValueError(f"ProbeTask {self.name!r} has no examples")
        if any(not p for p, _ in examples):
            raise ValueError(f"ProbeTask {self.name!r} has an empty prompt")
        object.__setattr__(self, "examples", examples)

    @property
    def prompts(self) -> list[str]:
        return [p for p, _ in self.examples]


@dataclass
class SuiteResult:
    """Per-capability mean score plus the raw predictions they were computed from."""

    scores: dict[str, float]
    predictions: dict[str, list[str]] = field(default_factory=dict)
    per_example_scores: dict[str, list[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class KLResult:
    """Mean per-token ``KL(base || current)`` over probe prompt positions (natural log)."""

    mean_kl: float
    num_tokens: int
    per_prompt_kl: tuple[float, ...]  # summed over the positions of each prompt


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _neutralise_generation_configs(model: nn.Module) -> list[tuple[nn.Module, GenerationConfig]]:
    """Swap every ``GenerationConfig`` stored on ``model``'s modules for a default one.

    Returns ``(module, original)`` pairs so the caller can restore them.
    """
    saved: list[tuple[nn.Module, GenerationConfig]] = []
    for module in model.modules():
        original = module.__dict__.get("generation_config")
        if isinstance(original, GenerationConfig):
            saved.append((module, original))
    for module, _ in saved:
        setattr(module, "generation_config", GenerationConfig())  # noqa: B010
    return saved


@torch.no_grad()
def greedy_generate(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    batch_size: int = 1,
) -> list[str]:
    """Deterministic greedy continuations of ``prompts`` (decoded, EOS and everything after removed).

    This is plain argmax decoding: the only stopping rules are the tokenizer's EOS and
    ``max_new_tokens``. Whatever the model's ``generation_config`` carries (sampling, beams,
    repetition penalties, ``no_repeat_ngram_size``, ``suppress_tokens``, ``bad_words_ids``,
    ``min_length`` / ``min_new_tokens``, ``stop_strings``, forced tokens, ...) is ignored: HF
    ``generate`` fills every unset field of a passed config from ``self.generation_config``, so for
    the duration of the call every ``GenerationConfig`` held by ``model`` or one of its submodules
    (this also reaches the inner model of a PEFT wrapper) is replaced by a default one and restored
    afterwards. Not thread-safe with concurrent use of the same model. With ``batch_size > 1``
    prompts are left-padded (built here; the tokenizer's ``padding_side`` is not touched). The
    model's training mode is restored.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    if pad is None:
        raise ValueError("tokenizer needs a pad_token_id or eos_token_id for generation")
    device = _device(model)
    was_training = model.training
    config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos,
        pad_token_id=int(pad),
    )
    saved = _neutralise_generation_configs(model)
    model.eval()
    outputs: list[str] = []
    try:
        for start in range(0, len(prompts), batch_size):
            encoded = [_encode(tokenizer, p) for p in prompts[start : start + batch_size]]
            width = max(len(ids) for ids in encoded)
            input_ids = torch.full((len(encoded), width), int(pad), dtype=torch.long)
            attention = torch.zeros(len(encoded), width, dtype=torch.long)
            for row, ids in enumerate(encoded):
                if not ids:
                    raise ValueError("cannot generate from an empty prompt")
                input_ids[row, width - len(ids) :] = torch.tensor(ids)
                attention[row, width - len(ids) :] = 1
            generated = model.generate(  # type: ignore[operator]
                input_ids=input_ids.to(device),
                attention_mask=attention.to(device),
                generation_config=config,
            )
            for row in generated[:, width:].tolist():
                if eos is not None and eos in row:
                    row = row[: row.index(eos)]
                outputs.append(tokenizer.decode(row, skip_special_tokens=True))
    finally:
        model.train(was_training)
        for owner, original in saved:
            setattr(owner, "generation_config", original)  # noqa: B010
    return outputs


def _kl_terms(lp: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
    """Elementwise ``p (log p - log q)`` with the limits ``0 log 0 = 0`` and ``p log(p / 0) = +inf``.

    The naive product is NaN where ``log p = -inf`` (masked / constrained vocab heads).
    """
    zero_p = torch.isneginf(lp)
    terms = lp.exp() * (lp - lq)
    terms = torch.where(torch.isneginf(lq) & ~zero_p, torch.full_like(terms, math.inf), terms)
    return torch.where(zero_p, torch.zeros_like(terms), terms)


@torch.no_grad()
def logit_kl(base_model: nn.Module, model: nn.Module, tokenizer: Any, prompts: Iterable[str]) -> KLResult:
    """Token-weighted mean of ``KL(p_base || p_model)`` over every position of every prompt.

    At position ``t`` both models predict the next token from ``prompt[: t + 1]``;
    ``KL = sum_v p_base(v) (log p_base(v) - log p_model(v))`` is computed in float64. Each prompt is
    run unpadded, so the value does not depend on batching. Identical models give exactly 0.
    Vocabulary entries with ``p_base = 0`` (logit ``-inf``) contribute 0; an entry with ``p_base > 0``
    but ``p_model = 0`` makes the KL ``+inf``.
    """
    per_prompt: list[float] = []
    total, count = 0.0, 0
    modes = (base_model.training, model.training)
    base_model.eval()
    model.eval()
    try:
        for prompt in prompts:
            ids = _encode(tokenizer, prompt)
            if not ids:
                raise ValueError("cannot compute KL on an empty prompt")
            x = torch.tensor([ids], dtype=torch.long)
            lp = torch.log_softmax(
                base_model(input_ids=x.to(_device(base_model))).logits[0].double().cpu(), -1
            )
            lq = torch.log_softmax(model(input_ids=x.to(_device(model))).logits[0].double().cpu(), -1)
            kl = float(_kl_terms(lp, lq).sum())
            per_prompt.append(kl)
            total += kl
            count += len(ids)
    finally:
        base_model.train(modes[0])
        model.train(modes[1])
    if count == 0:
        raise ValueError("no prompts given")
    return KLResult(mean_kl=total / count, num_tokens=count, per_prompt_kl=tuple(per_prompt))


ALL_TASKS_KEY = "all"  # key of the token-weighted aggregate in RetentionSuite.logit_kl()


class RetentionSuite:
    """A set of :class:`ProbeTask` evaluated with deterministic greedy decoding."""

    def __init__(self, tasks: Sequence[ProbeTask], *, max_new_tokens: int = 32, batch_size: int = 1) -> None:
        names = [t.name for t in tasks]
        if not names:
            raise ValueError("RetentionSuite needs at least one task")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate probe task names: {names}")
        if ALL_TASKS_KEY in names:
            raise ValueError(f"probe task name {ALL_TASKS_KEY!r} is reserved for the aggregate in logit_kl()")
        self.tasks = list(tasks)
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    def evaluate(self, model: nn.Module, tokenizer: Any) -> SuiteResult:
        result = SuiteResult(scores={})
        for task in self.tasks:
            preds = greedy_generate(
                model,
                tokenizer,
                task.prompts,
                max_new_tokens=task.max_new_tokens or self.max_new_tokens,
                batch_size=self.batch_size,
            )
            scores = [float(task.scorer(p, a)) for p, (_, a) in zip(preds, task.examples)]
            result.predictions[task.name] = preds
            result.per_example_scores[task.name] = scores
            result.scores[task.name] = sum(scores) / len(scores)
        return result

    def logit_kl(self, base_model: nn.Module, model: nn.Module, tokenizer: Any) -> dict[str, float]:
        """Per-task mean per-token KL plus ``"all"`` (token-weighted over every probe prompt)."""
        out: dict[str, float] = {}
        total, count = 0.0, 0
        for task in self.tasks:
            res = logit_kl(base_model, model, tokenizer, task.prompts)
            out[task.name] = res.mean_kl
            total += sum(res.per_prompt_kl)
            count += res.num_tokens
        out[ALL_TASKS_KEY] = total / count
        return out

    def report(
        self,
        base_model: nn.Module,
        trained_model: nn.Module,
        tokenizer: Any,
        recovered_model: nn.Module | None = None,
        *,
        with_kl: bool = True,
        drop_tolerance: float = 0.0,
    ) -> RetentionReport:
        """Evaluate all models and build a :class:`RetentionReport`."""
        base = self.evaluate(base_model, tokenizer).scores
        trained = self.evaluate(trained_model, tokenizer).scores
        recovered = self.evaluate(recovered_model, tokenizer).scores if recovered_model is not None else None
        kl_t = self.logit_kl(base_model, trained_model, tokenizer) if with_kl else None
        kl_r = (
            self.logit_kl(base_model, recovered_model, tokenizer)
            if with_kl and recovered_model is not None
            else None
        )
        return RetentionReport.from_scores(
            base,
            trained,
            recovered,
            drop_tolerance=drop_tolerance,
            kl_after_training=kl_t,
            kl_after_recovery=kl_r,
        )


@dataclass(frozen=True)
class CapabilityRetention:
    """Retention numbers of one capability. ``None`` marks a quantity that is undefined.

    * ``absolute_drop = base - after_training`` (negative = training improved the capability)
    * ``relative_drop = absolute_drop / base``; ``None`` when ``base == 0``
    * ``retention_ratio = after_training / base``; ``None`` when ``base == 0``
    * ``recovered_retention_ratio = after_recovery / base``; ``None`` when ``base == 0`` or no recovery
    * ``recovery_ratio = (after_recovery - after_training) / absolute_drop``: the fraction of the lost
      score won back (1 = fully recovered, >1 overshoot, <0 recovery made it worse). It is ``None``
      when there was no recovery stage or when there was **no degradation**
      (``absolute_drop <= drop_tolerance``): with nothing lost the ratio has no meaning, and
      reporting 1.0 or 0.0 would bias averages.
    """

    name: str
    base: float
    after_training: float
    after_recovery: float | None
    absolute_drop: float
    relative_drop: float | None
    retention_ratio: float | None
    recovered_retention_ratio: float | None
    recovery_ratio: float | None
    degraded: bool

    @classmethod
    def compute(
        cls,
        name: str,
        base: float,
        after_training: float,
        after_recovery: float | None = None,
        *,
        drop_tolerance: float = 0.0,
    ) -> CapabilityRetention:
        for label, v in (
            ("base", base),
            ("after_training", after_training),
            ("after_recovery", after_recovery),
        ):
            if v is not None and not math.isfinite(v):
                raise ValueError(f"{name}: {label} score is not finite ({v})")
        if drop_tolerance < 0:
            raise ValueError("drop_tolerance must be >= 0")
        drop = base - after_training
        degraded = drop > drop_tolerance
        has_base = base != 0
        return cls(
            name=name,
            base=base,
            after_training=after_training,
            after_recovery=after_recovery,
            absolute_drop=drop,
            relative_drop=drop / base if has_base else None,
            retention_ratio=after_training / base if has_base else None,
            recovered_retention_ratio=(
                after_recovery / base if has_base and after_recovery is not None else None
            ),
            recovery_ratio=(
                (after_recovery - after_training) / drop if degraded and after_recovery is not None else None
            ),
            degraded=degraded,
        )


def _mean_defined(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


@dataclass(frozen=True)
class RetentionReport:
    """Comparison ``base -> after_training -> after_recovery`` over all capabilities."""

    capabilities: dict[str, CapabilityRetention]
    drop_tolerance: float = 0.0
    kl_after_training: dict[str, float] | None = None
    kl_after_recovery: dict[str, float] | None = None

    @classmethod
    def from_scores(
        cls,
        base: Mapping[str, float],
        after_training: Mapping[str, float],
        after_recovery: Mapping[str, float] | None = None,
        *,
        drop_tolerance: float = 0.0,
        kl_after_training: Mapping[str, float] | None = None,
        kl_after_recovery: Mapping[str, float] | None = None,
    ) -> RetentionReport:
        keys = set(base)
        stages = [("after_training", after_training)]
        if after_recovery is not None:
            stages.append(("after_recovery", after_recovery))
        for label, scores in stages:
            if set(scores) != keys:
                raise ValueError(f"{label} capabilities {sorted(scores)} != base capabilities {sorted(keys)}")
        caps = {
            name: CapabilityRetention.compute(
                name,
                float(base[name]),
                float(after_training[name]),
                None if after_recovery is None else float(after_recovery[name]),
                drop_tolerance=drop_tolerance,
            )
            for name in base
        }
        return cls(
            capabilities=caps,
            drop_tolerance=drop_tolerance,
            kl_after_training=dict(kl_after_training) if kl_after_training is not None else None,
            kl_after_recovery=dict(kl_after_recovery) if kl_after_recovery is not None else None,
        )

    def summary(self) -> dict[str, Any]:
        """Unweighted means over capabilities; ratio means use only capabilities where defined."""
        caps = list(self.capabilities.values())
        has_recovery = all(c.after_recovery is not None for c in caps)
        return {
            "num_capabilities": len(caps),
            "num_degraded": sum(c.degraded for c in caps),
            "mean_base": _mean_defined(c.base for c in caps),
            "mean_after_training": _mean_defined(c.after_training for c in caps),
            "mean_after_recovery": _mean_defined(c.after_recovery for c in caps) if has_recovery else None,
            "mean_absolute_drop": _mean_defined(c.absolute_drop for c in caps),
            "mean_retention_ratio": _mean_defined(c.retention_ratio for c in caps),
            "mean_recovery_ratio": _mean_defined(c.recovery_ratio for c in caps),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "drop_tolerance": self.drop_tolerance,
            "capabilities": {
                n: {k: v for k, v in vars(c).items() if k != "name"} for n, c in self.capabilities.items()
            },
            "summary": self.summary(),
            "kl_after_training": self.kl_after_training,
            "kl_after_recovery": self.kl_after_recovery,
        }

    def log_to(self, recorder: Any, name: str = "capability_retention", step: int | None = None) -> None:
        """Append this report to an :class:`~research_rl.core.experiment.ExperimentRecorder`."""
        recorder.add_evaluation(name, self.to_dict(), step=step)
