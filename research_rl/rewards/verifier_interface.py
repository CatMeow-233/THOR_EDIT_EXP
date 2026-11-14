"""Verifier contract for verifiable rewards (RLVR) and the adapter to TRL reward functions.

* :class:`VerificationResult` - the outcome of grading one completion against one reference.
* :class:`Verifier` - the protocol every verifier satisfies; :class:`BaseVerifier` is an optional base
  class that maps statuses to rewards through a :class:`RewardScheme` and converts unexpected internal
  exceptions into ``status="error"`` results (except :class:`FatalVerifierError` subclasses, which always
  propagate - e.g. a disabled code sandbox must never be silently scored as 0).
* :func:`as_trl_reward_func` - wraps a verifier into a callable accepted by TRL's ``GRPOTrainer`` /
  ``RLOOTrainer`` ``reward_funcs`` (TRL 1.18 calls ``f(prompts=..., completions=..., completion_ids=...,
  trainer_state=..., log_extra=..., log_metric=..., **dataset_columns)`` and accepts ``None`` entries).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Status = Literal["correct", "incorrect", "format_error", "no_answer", "truncated", "error"]
STATUSES: tuple[str, ...] = ("correct", "incorrect", "format_error", "no_answer", "truncated", "error")


class FatalVerifierError(RuntimeError):
    """An error that must abort reward computation instead of being scored as ``status="error"``."""


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of verifying one completion.

    ``correct`` is ``True`` iff ``status == "correct"``. ``parsed_answer`` is the (normalised) answer
    extracted from the completion, ``reference_answer`` the normalised reference; ``details`` carries
    verifier-specific diagnostics (e.g. ``{"reason": "conflicting_boxes"}``).
    """

    reward: float
    correct: bool
    status: Status
    parsed_answer: str | None = None
    reference_answer: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown verification status {self.status!r}; expected one of {STATUSES}")
        if self.correct != (self.status == "correct"):
            raise ValueError(f"correct={self.correct} is inconsistent with status={self.status!r}")
        if not isinstance(self.reward, (int, float)) or not math.isfinite(self.reward):
            raise ValueError(f"reward must be a finite float, got {self.reward!r}")


@dataclass(frozen=True)
class RewardScheme:
    """Reward assigned to each verification status (defaults: binary 1/0 RLVR reward)."""

    correct: float = 1.0
    incorrect: float = 0.0
    format_error: float = 0.0
    no_answer: float = 0.0
    truncated: float = 0.0
    error: float = 0.0

    def __post_init__(self) -> None:
        for status in STATUSES:
            value = getattr(self, status)
            if not math.isfinite(value):
                raise ValueError(f"reward for {status!r} must be finite, got {value!r}")

    def reward_for(self, status: Status) -> float:
        return float(getattr(self, status))


@runtime_checkable
class Verifier(Protocol):
    """Anything with ``verify(completion, reference, **meta) -> VerificationResult``.

    ``completion`` is plain text (adapters convert chat messages to text first); ``reference`` is the
    dataset's gold answer; ``meta`` carries per-sample flags such as ``truncated=True``.
    """

    def verify(self, completion: str | None, reference: Any, **meta: Any) -> VerificationResult: ...


class BaseVerifier(ABC):
    """Convenience base: subclasses implement :meth:`_verify` returning ``(status, parsed, reference, details)``."""

    name: str = "verifier"

    def __init__(self, rewards: RewardScheme | None = None) -> None:
        self.rewards = rewards if rewards is not None else RewardScheme()

    @abstractmethod
    def _verify(
        self, completion: str | None, reference: Any, **meta: Any
    ) -> tuple[Status, str | None, str | None, dict[str, Any]]:
        """Grade one completion; may raise (non-fatal exceptions become ``status="error"``)."""

    def make_result(
        self,
        status: Status,
        parsed: str | None = None,
        reference: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> VerificationResult:
        return VerificationResult(
            reward=self.rewards.reward_for(status),
            correct=status == "correct",
            status=status,
            parsed_answer=parsed,
            reference_answer=reference,
            details=dict(details or {}),
        )

    def verify(self, completion: str | None, reference: Any, **meta: Any) -> VerificationResult:
        try:
            status, parsed, ref, details = self._verify(completion, reference, **meta)
        except FatalVerifierError:
            raise
        except Exception as exc:  # a grading bug must not crash a long RL run; it is surfaced as status
            return self.make_result("error", details={"exception": f"{type(exc).__name__}: {exc}"})
        return self.make_result(status, parsed, ref, details)


def completion_to_text(completion: Any) -> str | None:
    """Text of a TRL completion.

    * ``str`` -> itself; ``None`` -> ``None``.
    * conversational completion (list of ``{"role", "content"}`` dicts, as TRL passes when prompts are
      conversational) -> content of the **last assistant message** (earlier assistant / tool turns of a
      multi-turn rollout are reasoning, the final answer is expected in the last assistant turn).
      Content given as a list of parts (``[{"type": "text", "text": ...}]``) is joined.
    """
    if completion is None or isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        completion = [completion]
    if isinstance(completion, Sequence):
        for message in reversed(completion):
            if isinstance(message, Mapping) and message.get("role", "assistant") == "assistant":
                return _content_to_text(message.get("content"))
        return None
    raise TypeError(f"unsupported completion type {type(completion).__name__}")


def _content_to_text(content: Any) -> str | None:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping) and part.get("type", "text") == "text":
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    raise TypeError(f"unsupported message content type {type(content).__name__}")


class TRLRewardFunction:
    """Picklable TRL reward callable built by :func:`as_trl_reward_func` (``__name__`` drives TRL logging)."""

    def __init__(
        self,
        verifier: Verifier,
        answer_key: str = "answer",
        name: str | None = None,
        *,
        eos_token_ids: Sequence[int] | None = None,
        truncated_key: str | None = None,
        none_on_error: bool = False,
        log_status_metrics: bool = True,
    ) -> None:
        if not isinstance(verifier, Verifier):
            raise TypeError(f"{verifier!r} does not implement Verifier.verify")
        self.verifier = verifier
        self.answer_key = answer_key
        self.__name__ = name or f"{getattr(verifier, 'name', type(verifier).__name__)}_reward"
        self.eos_token_ids = frozenset(int(i) for i in eos_token_ids) if eos_token_ids is not None else None
        self.truncated_key = truncated_key
        self.none_on_error = none_on_error
        self.log_status_metrics = log_status_metrics
        self.last_results: list[VerificationResult] = []

    def _truncation_flags(self, n: int, completion_ids: Any, kwargs: Mapping[str, Any]) -> list[bool | None]:
        if self.truncated_key is not None:
            if self.truncated_key not in kwargs:
                raise KeyError(
                    f"truncated_key {self.truncated_key!r} not among reward kwargs {sorted(kwargs)}"
                )
            flags = list(kwargs[self.truncated_key])
            _check_len(self.truncated_key, flags, n)
            return [bool(f) for f in flags]
        if self.eos_token_ids is not None and completion_ids is not None:
            ids = list(completion_ids)
            _check_len("completion_ids", ids, n)
            # TRL keeps completion ids up to and including the first EOS: no final EOS == cut by length.
            return [len(seq) == 0 or int(seq[-1]) not in self.eos_token_ids for seq in ids]
        return [None] * n

    def __call__(
        self, prompts: Any = None, completions: Any = None, completion_ids: Any = None, **kwargs: Any
    ) -> list[float | None]:
        if completions is None:
            raise ValueError("reward function called without completions")
        completions = list(completions)
        n = len(completions)
        if self.answer_key not in kwargs:
            raise KeyError(
                f"reference column {self.answer_key!r} missing from reward kwargs "
                f"(dataset columns seen: {sorted(k for k in kwargs if k not in _TRL_RESERVED)})"
            )
        references = list(kwargs[self.answer_key])
        _check_len(self.answer_key, references, n)
        truncated = self._truncation_flags(n, completion_ids, kwargs)
        results = []
        for completion, reference, cut in zip(completions, references, truncated, strict=True):
            meta = {} if cut is None else {"truncated": cut}
            results.append(self.verifier.verify(completion_to_text(completion), reference, **meta))
        self.last_results = results
        log_metric = kwargs.get("log_metric")
        if self.log_status_metrics and callable(log_metric) and n:
            counts: Counter[str] = Counter(r.status for r in results)
            for status in STATUSES:
                log_metric(f"verifier/{self.__name__}/{status}_frac", counts.get(status, 0) / n)
        return [None if (self.none_on_error and r.status == "error") else r.reward for r in results]

    def __repr__(self) -> str:
        return f"TRLRewardFunction({self.__name__!r}, verifier={self.verifier!r}, answer_key={self.answer_key!r})"


_TRL_RESERVED = ("trainer_state", "log_extra", "log_metric", "environments")


def _check_len(key: str, values: Sequence[Any], n: int) -> None:
    if len(values) != n:
        raise ValueError(f"reward kwarg {key!r} has {len(values)} entries for {n} completions")


def as_trl_reward_func(
    verifier: Verifier,
    answer_key: str = "answer",
    name: str | None = None,
    *,
    eos_token_ids: Sequence[int] | None = None,
    truncated_key: str | None = None,
    none_on_error: bool = False,
    log_status_metrics: bool = True,
) -> TRLRewardFunction:
    """Wrap ``verifier`` as a TRL ``reward_funcs`` entry.

    Args:
        verifier: any :class:`Verifier`.
        answer_key: dataset column holding the reference (TRL forwards all extra columns as kwargs).
        name: ``__name__`` used by TRL for ``rewards/<name>/mean`` logging (default ``<verifier.name>_reward``).
        eos_token_ids: if given, a completion whose ``completion_ids`` do not end in one of these ids was cut
            by ``max_completion_length`` and is verified with ``truncated=True``. TRL decodes completions with
            ``skip_special_tokens=True``, so token ids are the only truncation signal in the TRL path.
        truncated_key: alternatively, a boolean dataset/extra-field column with the truncation flag.
        none_on_error: return ``None`` (TRL: "reward not applicable", excluded from the sum) for
            ``status="error"`` instead of the scheme's error reward.
        log_status_metrics: log the fraction of each status via TRL's ``log_metric`` kwarg when available.

    Raises (at call time) ``KeyError`` if the reference column is missing and ``ValueError`` on length
    mismatches - a misconfigured dataset must fail loudly instead of training on zero rewards.
    """
    return TRLRewardFunction(
        verifier,
        answer_key,
        name,
        eos_token_ids=eos_token_ids,
        truncated_key=truncated_key,
        none_on_error=none_on_error,
        log_status_metrics=log_status_metrics,
    )
