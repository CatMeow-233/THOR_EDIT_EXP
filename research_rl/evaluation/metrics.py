"""Evaluation metrics: exact match, unbiased pass@k, bootstrap confidence intervals, token-level NLL.

All functions are deterministic. ``sequence_nll`` runs a causal LM forward pass in ``eval`` mode
under ``torch.no_grad`` and restores the model's previous training mode afterwards.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------------- exact match
def normalize_answer(text: str, *, case_sensitive: bool = True) -> str:
    """Strip surrounding whitespace and collapse internal whitespace runs to one space."""
    out = " ".join(text.split())
    return out if case_sensitive else out.casefold()


def exact_match(
    prediction: str, reference: str, *, normalize: bool = True, case_sensitive: bool = True
) -> bool:
    """String equality, optionally after :func:`normalize_answer`."""
    if normalize:
        return normalize_answer(prediction, case_sensitive=case_sensitive) == normalize_answer(
            reference, case_sensitive=case_sensitive
        )
    return prediction == reference


def exact_match_accuracy(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    normalize: bool = True,
    case_sensitive: bool = True,
) -> float:
    """Fraction of predictions that exactly match their reference."""
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} predictions vs {len(references)} references")
    if not predictions:
        raise ValueError("exact_match_accuracy of an empty set is undefined")
    hits = sum(
        exact_match(p, r, normalize=normalize, case_sensitive=case_sensitive)
        for p, r in zip(predictions, references)
    )
    return hits / len(predictions)


# ---------------------------------------------------------------------------------- pass@k
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021): ``1 - C(n-c, k) / C(n, k)``.

    ``n`` samples were drawn for a problem and ``c`` of them are correct. Computed in the
    numerically stable product form ``1 - prod_{i=n-c+1}^{n} (1 - k / i)``, which never forms
    the (potentially huge) binomial coefficients.
    """
    if n < 1 or k < 1:
        raise ValueError(f"need n >= 1 and k >= 1, got n={n}, k={k}")
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if k > n:
        raise ValueError(f"k={k} exceeds the number of samples n={n}")
    if n - c < k:  # every size-k subset contains at least one correct sample
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def mean_pass_at_k(num_samples: Sequence[int], num_correct: Sequence[int], k: int) -> float:
    """Average of :func:`pass_at_k` over problems (``num_samples[i]``, ``num_correct[i]``)."""
    if len(num_samples) != len(num_correct):
        raise ValueError("num_samples and num_correct must have the same length")
    if not num_samples:
        raise ValueError("mean_pass_at_k of an empty set is undefined")
    return float(sum(pass_at_k(n, c, k) for n, c in zip(num_samples, num_correct)) / len(num_samples))


# ---------------------------------------------------------------------------------- bootstrap
@dataclass(frozen=True)
class MeanCI:
    """Sample mean with a percentile-bootstrap confidence interval."""

    mean: float
    low: float
    high: float
    confidence: float
    n: int
    n_resamples: int
    seed: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "ci_low": self.low,
            "ci_high": self.high,
            "confidence": self.confidence,
            "n": self.n,
            "n_resamples": self.n_resamples,
            "seed": self.seed,
        }


def mean_with_bootstrap_ci(
    values: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> MeanCI:
    """Mean and percentile-bootstrap CI, deterministic given ``seed``.

    Procedure (exactly): ``rng = numpy.random.default_rng(seed)``; draw an index matrix
    ``rng.integers(0, n, size=(n_resamples, n))``; resample means are the row means of
    ``values[idx]``; the interval is ``numpy.quantile(means, [(1-confidence)/2, (1+confidence)/2])``
    with linear interpolation. A single value gives a degenerate interval ``[x, x]``.
    """
    if isinstance(values, torch.Tensor):
        x = values.detach().cpu().double().reshape(-1).numpy()
    else:
        x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("mean of an empty set is undefined")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    means = x[idx].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return MeanCI(
        mean=float(x.mean()),
        low=float(low),
        high=float(high),
        confidence=confidence,
        n=int(x.size),
        n_resamples=n_resamples,
        seed=seed,
    )


# ---------------------------------------------------------------------------------- token NLL
@dataclass
class NLLResult:
    """Token-level negative log-likelihood (natural log) of a causal LM.

    ``mean_nll`` is the *micro* average over all scored tokens; ``perplexity = exp(mean_nll)``.
    ``per_example_nll``/``per_example_tokens`` hold the summed NLL and token count of each pair.
    """

    nll_sum: float
    num_tokens: int
    mean_nll: float
    perplexity: float
    per_example_nll: list[float] = field(default_factory=list)
    per_example_tokens: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nll_sum": self.nll_sum,
            "num_tokens": self.num_tokens,
            "mean_nll": self.mean_nll,
            "perplexity": self.perplexity,
        }


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def sequence_nll(
    model: nn.Module,
    tokenizer: Any,
    pairs: Sequence[tuple[str, str]],
    *,
    completion_only: bool = True,
    batch_size: int = 8,
    add_bos: bool = False,
) -> NLLResult:
    """Summed / mean NLL of ``(prompt, completion)`` pairs under a causal LM.

    Prompt and completion are tokenized *separately* (no special tokens) and concatenated, so the
    prompt/completion boundary is exact even for tokenizers that would merge across it. Token
    ``t`` (``t >= 1``) is scored with the logits at position ``t - 1``; the very first token of a
    sequence therefore has no score. With ``completion_only`` only completion tokens are scored
    (the first completion token is predicted from the last prompt token, or from BOS when the
    prompt is empty and ``add_bos``). Sequences are right-padded within a batch and padded
    positions are excluded through ``attention_mask`` and the label mask.
    """
    if not pairs:
        raise ValueError("no (prompt, completion) pairs given")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    seqs: list[list[int]] = []
    scored: list[list[bool]] = []
    for prompt, completion in pairs:
        p_ids, c_ids = _encode(tokenizer, prompt), _encode(tokenizer, completion)
        if add_bos:
            if tokenizer.bos_token_id is None:
                raise ValueError("add_bos=True but the tokenizer has no bos_token_id")
            p_ids = [int(tokenizer.bos_token_id), *p_ids]
        ids = p_ids + c_ids
        if not ids:
            raise ValueError(f"pair {(prompt, completion)!r} tokenizes to an empty sequence")
        flags = [False] * len(p_ids) + [True] * len(c_ids) if completion_only else [True] * len(ids)
        flags[0] = False  # nothing predicts the first token
        if not any(flags):
            raise ValueError(f"pair {(prompt, completion)!r} has no scorable tokens")
        seqs.append(ids)
        scored.append(flags)

    device = _model_device(model)
    was_training = model.training
    model.eval()
    per_nll: list[float] = []
    per_tok: list[int] = []
    try:
        for start in range(0, len(seqs), batch_size):
            chunk = seqs[start : start + batch_size]
            flags_chunk = scored[start : start + batch_size]
            width = max(len(s) for s in chunk)
            input_ids = torch.zeros(len(chunk), width, dtype=torch.long)
            attention = torch.zeros(len(chunk), width, dtype=torch.long)
            target_mask = torch.zeros(len(chunk), width, dtype=torch.bool)
            for row, (ids, flags) in enumerate(zip(chunk, flags_chunk)):
                input_ids[row, : len(ids)] = torch.tensor(ids)
                attention[row, : len(ids)] = 1
                target_mask[row, : len(ids)] = torch.tensor(flags)
            logits = model(input_ids=input_ids.to(device), attention_mask=attention.to(device)).logits
            logp = torch.log_softmax(logits[:, :-1].double(), dim=-1).cpu()
            targets = input_ids[:, 1:]
            token_nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            mask = target_mask[:, 1:]
            token_nll = torch.where(mask, token_nll, torch.zeros_like(token_nll))
            per_nll.extend(float(v) for v in token_nll.sum(dim=1))
            per_tok.extend(int(v) for v in mask.sum(dim=1))
    finally:
        model.train(was_training)
    total = float(sum(per_nll))
    count = int(sum(per_tok))
    mean = total / count
    return NLLResult(
        nll_sum=total,
        num_tokens=count,
        mean_nll=mean,
        perplexity=math.exp(mean),
        per_example_nll=per_nll,
        per_example_tokens=per_tok,
    )
