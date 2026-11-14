"""Multi-teacher on-policy distillation building blocks.

This is *distillation* (not reinforcement learning) and it is distinct from the P²-MOPD parameter
prior (:mod:`research_rl.optim.parameter_prior`): here several teachers define the target
next-token distribution; there, task vectors constrain the student's weights. The two compose:
use :class:`TeacherEnsemble` / the KL helpers for the loss and ``P2PriorRule`` for the update.

Implemented and numerically tested:

* :func:`aggregate_teacher_logprobs` – ``"mixture"``: log Σ_i w_i p_i; ``"product"``: the
  normalised geometric mixture Σ_i w_i log p_i − logsumexp_v(Σ_i w_i log p_i). Weights may be
  global ``[K]``, per sample ``[K, B]`` (routing, e.g. one-hot per prompt domain) or per token.
* :func:`masked_forward_kl` (KL(p_t‖p_s)) and :func:`masked_reverse_kl` (KL(p_s‖p_t)) with masks.
* :class:`MultiTeacherBatch` – student completion ids/mask + per-teacher log-probs + weights.
* :class:`TeacherEnsemble` – an ``nn.Module`` over several HF causal LMs whose ``forward().logits``
  are the aggregated log-probabilities, and which also exposes what TRL ``DistillationTrainer``
  actually reads from a teacher (TRL 1.18 computes the loss chunk-wise from
  ``teacher.base_model(...).last_hidden_state`` and ``teacher.get_output_embeddings()``; it never
  calls ``teacher(...).logits``):

  - product mode: hidden = concat_i(w_i·s_i·h_i), head = concat_i(W_i) (+ Σ w_i s_i b_i) — exact
    and memory-linear (a copy of the teachers' heads, V × Σ H_i),
  - mixture mode is not linear in any hidden state, so hidden = log-mixture (width V) and the head
    is the V×V identity. Exact, but O(V²) memory: guarded by ``max_identity_vocab``.

Not covered (documented only): TRL's ``trl.experimental.async_distillation.AsyncDistillationTrainer``
needs a running vLLM server for teacher log-probs and was not exercised here; DeepSpeed/FSDP
sharding of the ensemble is NOT_TESTED (CPU only).
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithPast

AGGREGATION_MODES = ("mixture", "product")
# Config attributes TRL reads to post-process teacher logits (see DistillationTrainer._compute_loss).
_LOGIT_SCALE_ATTRS = ("logit_scale", "output_multiplier", "lm_head_multiplier", "logits_scaling")
_SOFTCAP_ATTR = "final_logit_softcapping"


# ===================================================================================== aggregation
def normalize_weights(weights: torch.Tensor | Sequence[float]) -> torch.Tensor:
    """Non-negative weights normalised to sum to 1 along dim 0 (the teacher axis), in float64."""
    w = torch.as_tensor(weights, dtype=torch.float64)
    if (w < 0).any():
        raise ValueError("teacher weights must be non-negative")
    total = w.sum(0, keepdim=True)
    if (total <= 0).any():
        raise ValueError("teacher weights must not all be zero")
    return w / total


def _broadcast_weights(weights: torch.Tensor, logprobs: torch.Tensor) -> torch.Tensor:
    """Reshape ``[K, *prefix]`` weights to broadcast against ``[K, ..., V]`` log-probs."""
    w = weights.to(device=logprobs.device, dtype=logprobs.dtype)
    if w.shape[0] != logprobs.shape[0]:
        raise ValueError(f"{w.shape[0]} weights for {logprobs.shape[0]} teachers")
    if w.dim() > logprobs.dim() - 1 or tuple(w.shape[1:]) != tuple(logprobs.shape[1 : w.dim()]):
        raise ValueError(f"weights of shape {tuple(w.shape)} do not prefix log-probs {tuple(logprobs.shape)}")
    return w.reshape(*w.shape, *([1] * (logprobs.dim() - w.dim())))


def aggregate_teacher_logprobs(
    teacher_logprobs: torch.Tensor,
    weights: torch.Tensor | Sequence[float],
    mode: str = "mixture",
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Combine K teachers' log-probabilities ``[K, ..., V]`` into one log-distribution ``[..., V]``.

    ``mixture``: log Σ_i w_i p_i (zero-weight teachers contribute nothing).
    ``product``: Σ_i w_i log p_i, renormalised over the vocabulary.
    ``weights`` is ``[K]`` or ``[K, *prefix]`` (e.g. ``[K, B]`` per-sample routing).
    """
    if mode not in AGGREGATION_MODES:
        raise ValueError(f"mode must be one of {AGGREGATION_MODES}, got {mode!r}")
    w = normalize_weights(weights) if normalize else torch.as_tensor(weights, dtype=torch.float64)
    w = _broadcast_weights(w, teacher_logprobs)
    if mode == "mixture":
        log_w = torch.where(w > 0, w.clamp_min(torch.finfo(w.dtype).tiny).log(), -torch.inf)
        return torch.logsumexp(log_w + teacher_logprobs, dim=0)
    # product: avoid 0 * (-inf) = nan for zero-weight teachers
    terms = torch.where(w > 0, w * teacher_logprobs, torch.zeros_like(teacher_logprobs))
    s = terms.sum(0)
    return s - torch.logsumexp(s, dim=-1, keepdim=True)


# ===================================================================================== KL helpers
def _reduce(per_token: torch.Tensor, mask: torch.Tensor | None, reduction: str) -> torch.Tensor:
    if mask is not None:
        m = mask.to(dtype=per_token.dtype, device=per_token.device)
        per_token = per_token * m
    if reduction == "none":
        return per_token
    if reduction == "sum":
        return per_token.sum()
    if reduction == "mean":
        n = per_token.numel() if mask is None else mask.sum()
        return per_token.sum() / torch.as_tensor(n, dtype=per_token.dtype).clamp_min(1)
    raise ValueError(f"reduction must be none/sum/mean, got {reduction!r}")


def _kl(p_logprobs: torch.Tensor, q_logprobs: torch.Tensor) -> torch.Tensor:
    """Σ_v p_v (log p_v − log q_v) with 0·log 0 = 0."""
    p = p_logprobs.exp()
    diff = torch.where(p > 0, p_logprobs - q_logprobs, torch.zeros_like(p_logprobs))
    return (p * diff).sum(-1)


def masked_forward_kl(
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """KL(p_teacher ‖ p_student) per token over the last (vocab) dim; ``mean`` averages valid tokens."""
    return _reduce(_kl(teacher_logprobs, student_logprobs), mask, reduction)


def masked_reverse_kl(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """KL(p_student ‖ p_teacher) per token (the on-policy distillation objective)."""
    return _reduce(_kl(student_logprobs, teacher_logprobs), mask, reduction)


# ===================================================================================== batch / specs
@dataclass
class TeacherSpec:
    """A teacher: a model id/path or an instantiated causal LM, plus its aggregation weight."""

    name: str
    model: str | nn.Module
    weight: float = 1.0
    revision: str | None = None

    def load(self, dtype: torch.dtype | str | None = None) -> nn.Module:
        if isinstance(self.model, nn.Module):
            return self.model
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(self.model, revision=self.revision, dtype=dtype)


@dataclass
class MultiTeacherBatch:
    """Student completions with K teachers' next-token log-probs.

    ``teacher_logprobs`` is ``[K, B, T, V]`` aligned with ``completion_ids`` ``[B, T]`` (position t
    holds the distribution that predicts ``completion_ids[:, t]``); ``weights`` is ``[K]``,
    ``[K, B]`` (per-sample routing) or ``[K, B, T]``.
    """

    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    teacher_logprobs: torch.Tensor
    weights: torch.Tensor
    teacher_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        b, t = self.completion_ids.shape
        if tuple(self.completion_mask.shape) != (b, t):
            raise ValueError("completion_mask must match completion_ids")
        if self.teacher_logprobs.dim() != 4 or tuple(self.teacher_logprobs.shape[1:3]) != (b, t):
            raise ValueError(
                f"teacher_logprobs must be [K, {b}, {t}, V], got {tuple(self.teacher_logprobs.shape)}"
            )
        self.weights = torch.as_tensor(self.weights, dtype=torch.float64)
        _broadcast_weights(self.weights, self.teacher_logprobs)
        if self.teacher_names and len(self.teacher_names) != self.num_teachers:
            raise ValueError("one name per teacher")

    @property
    def num_teachers(self) -> int:
        return int(self.teacher_logprobs.shape[0])

    def aggregated(self, mode: str = "mixture") -> torch.Tensor:
        return aggregate_teacher_logprobs(self.teacher_logprobs, self.weights, mode)

    def teacher_token_logprobs(self) -> torch.Tensor:
        """``[K, B, T]`` log-prob each teacher assigns to the sampled completion tokens."""
        idx = self.completion_ids.unsqueeze(0).unsqueeze(-1).expand(self.num_teachers, -1, -1, 1)
        return self.teacher_logprobs.gather(-1, idx).squeeze(-1)

    def reverse_kl(self, student_logprobs: torch.Tensor, mode: str = "mixture") -> torch.Tensor:
        return masked_reverse_kl(student_logprobs, self.aggregated(mode), self.completion_mask)

    def forward_kl(self, student_logprobs: torch.Tensor, mode: str = "mixture") -> torch.Tensor:
        return masked_forward_kl(self.aggregated(mode), student_logprobs, self.completion_mask)


# ===================================================================================== ensemble
def _text_config(model: nn.Module) -> Any:
    cfg = model.config
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


def _trl_logit_scale(cfg: Any) -> float:
    """The logit multiplier TRL's DistillationTrainer applies for this config (1.0 if none)."""
    for attr in ("logit_scale", "output_multiplier", "lm_head_multiplier"):
        v = getattr(cfg, attr, None)
        if v is not None:
            return float(v)
    if getattr(cfg, "logits_scaling", None) is not None:
        s = float(cfg.logits_scaling)
        return s if getattr(cfg, "model_type", None) == "hyperclovax" else 1.0 / s
    return 1.0


class _EnsembleHead(nn.Module):
    """Stand-in ``lm_head``: ``logits = hidden @ weight.T + bias`` (what TRL's chunked loss computes)."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
        super().__init__()
        self.register_buffer("weight", weight, persistent=False)
        self.bias: torch.Tensor | None
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias, persistent=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(hidden, self.weight, self.bias)


class _EnsembleBackbone:
    """Callable returned by ``TeacherEnsemble.base_model`` (TRL calls ``base_model(**inputs)``)."""

    def __init__(self, ensemble: TeacherEnsemble) -> None:
        self._ensemble = ensemble

    def __call__(self, **inputs: Any) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self._ensemble.trl_hidden_states(**inputs))


class TeacherEnsemble(nn.Module):
    """Several frozen HF causal LMs (shared vocabulary) acting as one teacher.

    ``forward(...).logits`` are the aggregated log-probabilities (already normalised), so the
    module can be used anywhere a teacher's logits are consumed. For TRL ``DistillationTrainer``
    (``teacher_model=ensemble``) it also provides ``config``, ``base_model`` and
    ``get_output_embeddings()`` such that ``base_model(...).last_hidden_state @ head.weight.T +
    head.bias`` reproduces the aggregated distribution up to a per-position constant.
    """

    def __init__(
        self,
        teachers: Sequence[nn.Module],
        weights: Sequence[float] | None = None,
        mode: str = "mixture",
        *,
        names: Sequence[str] | None = None,
        max_identity_vocab: int = 8192,
    ) -> None:
        super().__init__()
        if mode not in AGGREGATION_MODES:
            raise ValueError(f"mode must be one of {AGGREGATION_MODES}, got {mode!r}")
        if not teachers:
            raise ValueError("at least one teacher is required")
        vocab = {int(_text_config(t).vocab_size) for t in teachers}
        if len(vocab) != 1:
            raise ValueError(f"teachers must share a vocabulary, got vocab sizes {sorted(vocab)}")
        self.vocab_size = vocab.pop()
        self.mode = mode
        self.max_identity_vocab = max_identity_vocab
        self.teachers = nn.ModuleList(teachers)
        self.teachers.requires_grad_(False)
        self.teachers.eval()
        w = normalize_weights([1.0] * len(teachers) if weights is None else list(weights))
        if w.shape != (len(teachers),):
            raise ValueError("one weight per teacher")
        self.register_buffer("weights", w, persistent=False)
        self.names = list(names) if names is not None else [f"teacher{i}" for i in range(len(teachers))]
        cfg = copy.deepcopy(teachers[0].config)
        text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        for attr in (*_LOGIT_SCALE_ATTRS, _SOFTCAP_ATTR):  # scaling is folded into hidden/head below
            if getattr(text_cfg, attr, None) is not None:
                setattr(text_cfg, attr, None)
        self.config = cfg
        self._head: _EnsembleHead | None = None

    @classmethod
    def from_specs(
        cls, specs: Sequence[TeacherSpec], mode: str = "mixture", dtype: torch.dtype | str | None = None
    ) -> TeacherEnsemble:
        return cls(
            [s.load(dtype) for s in specs], [s.weight for s in specs], mode, names=[s.name for s in specs]
        )

    # ------------------------------------------------------------------ plain forward
    def teacher_logprobs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        """``[K, B, L, V]`` fp32 log-probs of every teacher."""
        kwargs.pop("use_cache", None)
        out = []
        for t in self.teachers:
            logits = t(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **kwargs).logits
            out.append(torch.log_softmax(logits.float(), dim=-1))
        return torch.stack(out)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: Any
    ) -> CausalLMOutputWithPast:
        lp = self.teacher_logprobs(input_ids, attention_mask, **kwargs)
        return CausalLMOutputWithPast(logits=aggregate_teacher_logprobs(lp, self.weights, self.mode))

    # ------------------------------------------------------------------ TRL DistillationTrainer view
    @property
    def base_model(self) -> _EnsembleBackbone:
        return _EnsembleBackbone(self)

    def trl_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        logits_to_keep: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Hidden states that, through :meth:`get_output_embeddings`, give the aggregated logits."""
        kwargs.pop("use_cache", None)
        if self.mode == "mixture":
            self._check_identity_size()
            h = self.forward(input_ids, attention_mask, **kwargs).logits
        else:
            parts = []
            for w, t in zip(self.weights, self.teachers):
                cfg = _text_config(t)
                if getattr(cfg, _SOFTCAP_ATTR, None) is not None:
                    raise ValueError(
                        "product mode cannot fold final_logit_softcapping (nonlinear) into a head"
                    )
                hs = t.base_model(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **kwargs
                ).last_hidden_state
                parts.append(hs * (float(w) * _trl_logit_scale(cfg)))
            h = torch.cat(parts, dim=-1)
        if logits_to_keep:
            h = h[:, -int(logits_to_keep) :, :]
        return h

    def _check_identity_size(self) -> None:
        if self.vocab_size > self.max_identity_vocab:
            raise ValueError(
                f"mixture mode exposes a {self.vocab_size}x{self.vocab_size} identity head to TRL (O(V^2) memory); "
                f"vocab exceeds max_identity_vocab={self.max_identity_vocab}. Use mode='product', raise the limit, "
                "or compute the mixture loss yourself with aggregate_teacher_logprobs + masked_reverse_kl."
            )

    def get_output_embeddings(self) -> _EnsembleHead:
        if self._head is None:
            first = self.teachers[0].get_output_embeddings()
            device = first.weight.device
            if self.mode == "mixture":
                self._check_identity_size()
                self._head = _EnsembleHead(torch.eye(self.vocab_size, device=device), None)
            else:
                weights, bias = [], None
                for w, t in zip(self.weights, self.teachers):
                    head = t.get_output_embeddings()
                    weights.append(head.weight.detach().to(device))
                    if getattr(head, "bias", None) is not None:
                        b = head.bias.detach().to(device) * (float(w) * _trl_logit_scale(_text_config(t)))
                        bias = b if bias is None else bias + b
                self._head = _EnsembleHead(torch.cat(weights, dim=1), bias)
        return self._head

    def get_input_embeddings(self) -> nn.Module:
        return self.teachers[0].get_input_embeddings()
