"""Parameter priors built from task vectors (P²-MOPD-style parameter-direction fusion).

This module is about a **parameter prior**: the student's weights are constrained to / steered
along directions spanned by *task vectors* τ_k = θ_expert_k − θ_base of fine-tuned experts.
It is **not** multi-teacher distillation, and the P²-MOPD distillation loss (per-prompt routed,
support-restricted reverse KL) is **not** implemented in ResearchRL: :mod:`research_rl.trainers.multi_teacher`
only provides building blocks (offline routed aggregation + KL helpers, and a static-weight
teacher ensemble for TRL's full-vocabulary JSD). Every rule here works with *any* loss.

What is implemented (and numerically tested):

* :class:`TaskVector` – named parameter differences with arithmetic, norms and cosine similarity.
* :class:`TaskVectorPrior` – θ(α) = θ_base + Σ_i α_i τ_i; exact ``apply_to(model, α)`` and a
  differentiable ``functional_call`` path (α as tensors with ``requires_grad``).
* :func:`project_onto_span` – least-squares projection of a displacement onto span{τ_1..τ_K}
  (minimum-norm solution via SVD, identical to ``torch.linalg.lstsq`` with ``gelsd``).
* :class:`PriorSubspaceRule` – an :class:`UpdateRule` that projects the candidate update Δ onto
  (``mode="onto"``) or away from (``mode="away"``) the per-parameter span of the task vectors.
* :class:`P2PriorRule` – the two-channel optimiser step of P²-MOPD (Algorithm 1) for
  :class:`~research_rl.optim.optimizer_wrapper.HookedOptimizer`, see its docstring.

What is only a contract: nothing in this file chooses *which* experts/task vectors to use or how to
initialise λ beyond uniform/zero; those are research decisions left to the caller.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
from torch import nn
from transformers import TrainerCallback

from research_rl.core.registry import PARAMETER_PRIORS, UPDATE_RULES
from research_rl.optim.optimizer_wrapper import HookedOptimizer, unwrap_hooked
from research_rl.optim.update_interface import StepContext, UpdateRule

Coefficients: TypeAlias = Mapping[str, float | torch.Tensor] | Sequence[float | torch.Tensor] | torch.Tensor

# Linear projections of a Llama/Qwen-style decoder layer (attention q/k/v/o and SwiGLU gate/up/down).
PROJECTION_PATTERN = re.compile(r"(^|\.)(q|k|v|o|gate|up|down)_proj\.weight$")
_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_CLIP_EPS = 1e-6  # same epsilon as torch.nn.utils.clip_grad_norm_


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """fp32 for half/bf16/fp32 tensors, the tensor's own dtype if wider (float64)."""
    return torch.promote_types(dtype, torch.float32)


def _overrides(rule: UpdateRule, method: str) -> bool:
    return getattr(type(rule), method) is not getattr(UpdateRule, method)


# ====================================================================================== task vectors
@PARAMETER_PRIORS.register("task_vector")
class TaskVector:
    """τ = θ_expert − θ_base over a set of named parameters (stored in ``dtype``, default fp32)."""

    def __init__(self, tensors: Mapping[str, torch.Tensor]) -> None:
        self._t: dict[str, torch.Tensor] = {n: t.detach() for n, t in tensors.items()}

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_state_dicts(
        cls,
        expert: Mapping[str, torch.Tensor],
        base: Mapping[str, torch.Tensor],
        names: Iterable[str] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> TaskVector:
        keys = list(names) if names is not None else [k for k in base if k in expert]
        out = {}
        for k in keys:
            if k not in expert or k not in base:
                raise KeyError(f"parameter {k!r} missing from expert or base state dict")
            if expert[k].shape != base[k].shape:
                raise ValueError(
                    f"shape mismatch for {k!r}: {tuple(expert[k].shape)} vs {tuple(base[k].shape)}"
                )
            out[k] = expert[k].detach().to(dtype) - base[k].detach().to(dtype)
        return cls(out)

    @classmethod
    def from_models(
        cls,
        expert: nn.Module,
        base: nn.Module,
        names: Iterable[str] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> TaskVector:
        return cls.from_state_dicts(
            dict(expert.named_parameters()), dict(base.named_parameters()), names=names, dtype=dtype
        )

    # ------------------------------------------------------------------ container protocol
    def names(self) -> list[str]:
        return list(self._t)

    def items(self) -> Iterable[tuple[str, torch.Tensor]]:
        return self._t.items()

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._t[name]

    def __contains__(self, name: object) -> bool:
        return name in self._t

    def __len__(self) -> int:
        return len(self._t)

    def __repr__(self) -> str:
        return f"TaskVector({len(self)} tensors, norm={self.norm():.4g})"

    # ------------------------------------------------------------------ arithmetic
    def _check_same_keys(self, other: TaskVector) -> None:
        if set(self._t) != set(other._t):
            raise KeyError("task vectors cover different parameter names")

    def __add__(self, other: TaskVector) -> TaskVector:
        self._check_same_keys(other)
        return TaskVector({n: t + other._t[n] for n, t in self._t.items()})

    def __sub__(self, other: TaskVector) -> TaskVector:
        self._check_same_keys(other)
        return TaskVector({n: t - other._t[n] for n, t in self._t.items()})

    def __neg__(self) -> TaskVector:
        return TaskVector({n: -t for n, t in self._t.items()})

    def __mul__(self, scale: float) -> TaskVector:
        if not isinstance(scale, int | float):
            return NotImplemented
        return TaskVector({n: t * scale for n, t in self._t.items()})

    __rmul__ = __mul__

    def to(self, *args: Any, **kwargs: Any) -> TaskVector:
        return TaskVector({n: t.to(*args, **kwargs) for n, t in self._t.items()})

    # ------------------------------------------------------------------ geometry
    def param_norms(self) -> dict[str, float]:
        return {n: float(torch.linalg.vector_norm(t.double())) for n, t in self._t.items()}

    def norm(self) -> float:
        """Global L2 norm over all covered parameters (computed in fp64)."""
        return math.sqrt(sum(float(t.double().pow(2).sum()) for t in self._t.values()))

    def dot(self, other: TaskVector) -> float:
        self._check_same_keys(other)
        return sum(float((t.double() * other._t[n].double()).sum()) for n, t in self._t.items())

    def cosine_similarity(self, other: TaskVector) -> float:
        """Cosine of the angle between the two flattened task vectors (0.0 if either is zero)."""
        denom = self.norm() * other.norm()
        return self.dot(other) / denom if denom > 0 else 0.0

    def param_cosine_similarity(self, other: TaskVector) -> dict[str, float]:
        self._check_same_keys(other)
        out = {}
        for n, t in self._t.items():
            a, b = t.double().flatten(), other._t[n].double().flatten()
            d = float(a.norm() * b.norm())
            out[n] = float(a @ b) / d if d > 0 else 0.0
        return out

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> dict[str, torch.Tensor]:
        return {n: t.cpu() for n, t in self._t.items()}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor]) -> TaskVector:
        return cls(dict(state))


# ====================================================================================== span projection
@dataclass
class SpanProjection:
    """Least-squares decomposition ``d = Σ_k c_k τ_k + residual``."""

    coefficients: torch.Tensor  # [K], minimum-norm least-squares solution
    projection: torch.Tensor  # same shape as the displacement
    residual: torch.Tensor  # displacement − projection
    relative_residual: float  # ||residual|| / ||displacement|| (0.0 for a zero displacement)


def _basis_matrix(basis: Sequence[torch.Tensor] | torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(basis, torch.Tensor):
        return basis.reshape(basis.shape[0], -1).to(dtype).T
    return torch.stack([b.reshape(-1).to(dtype) for b in basis], dim=1)


def project_onto_span(
    displacement: torch.Tensor,
    basis: Sequence[torch.Tensor] | torch.Tensor,
    *,
    rcond: float = 1e-10,
    dtype: torch.dtype = torch.float64,
) -> SpanProjection:
    """Project ``displacement`` onto span{basis_k} by least squares (paper's analysis utility).

    ``basis`` is a list of K tensors shaped like ``displacement`` (or one ``[K, ...]`` tensor).
    The minimum-norm solution is computed with an SVD (singular values below ``rcond·σ_max`` are
    dropped), so rank-deficient bases (e.g. duplicated task vectors) are handled.
    """
    t = _basis_matrix(basis, dtype)  # [N, K]
    d = displacement.reshape(-1).to(dtype)
    if t.shape[0] != d.shape[0]:
        raise ValueError(f"basis vectors have {t.shape[0]} elements, displacement has {d.shape[0]}")
    u, s, vh = torch.linalg.svd(t, full_matrices=False)
    keep = s > rcond * (s.max() if s.numel() else 0.0)
    inv = torch.where(keep, 1.0 / torch.where(keep, s, torch.ones_like(s)), torch.zeros_like(s))
    coeffs = vh.T @ (inv * (u.T @ d))
    proj = t @ coeffs
    res = d - proj
    dn = float(d.norm())
    return SpanProjection(
        coefficients=coeffs,
        projection=proj.reshape(displacement.shape),
        residual=res.reshape(displacement.shape),
        relative_residual=float(res.norm()) / dn if dn > 0 else 0.0,
    )


def orthonormal_span_basis(basis: torch.Tensor, rcond: float = 1e-6) -> torch.Tensor:
    """Orthonormal basis ``U`` [N, r] of the column span of ``basis`` [N, K] (SVD, rank-revealing)."""
    u, s, _ = torch.linalg.svd(basis, full_matrices=False)
    if s.numel() == 0 or float(s.max()) == 0.0:
        return basis.new_zeros(basis.shape[0], 0)
    return u[:, s > rcond * s.max()]


# ====================================================================================== prior
def _coefficient_list(coefficients: Coefficients, names: Sequence[str]) -> list[float | torch.Tensor]:
    if isinstance(coefficients, Mapping):
        unknown = set(coefficients) - set(names)
        if unknown:
            raise KeyError(f"unknown task vectors {sorted(unknown)}; registered: {list(names)}")
        return [coefficients.get(n, 0.0) for n in names]
    if isinstance(coefficients, torch.Tensor):
        if coefficients.dim() != 1:
            raise ValueError("tensor coefficients must be 1-D [K]")
        coeffs: list[float | torch.Tensor] = list(coefficients.unbind(0))
    else:
        coeffs = list(coefficients)
    if len(coeffs) != len(names):
        raise ValueError(f"got {len(coeffs)} coefficients for {len(names)} task vectors")
    return coeffs


@PARAMETER_PRIORS.register("task_vector_prior")
class TaskVectorPrior:
    """θ(α) = θ_base + Σ_i α_i τ_i over named parameters.

    ``base`` is a state dict or a module (its parameters are snapshotted). Task vectors may cover
    a subset of the base names; uncovered parameters get no contribution from that vector.
    """

    def __init__(
        self, base: Mapping[str, torch.Tensor] | nn.Module, names: Iterable[str] | None = None
    ) -> None:
        src = dict(base.named_parameters()) if isinstance(base, nn.Module) else dict(base)
        keys = list(names) if names is not None else list(src)
        self.base: dict[str, torch.Tensor] = {k: src[k].detach().clone() for k in keys}
        self.task_vectors: dict[str, TaskVector] = {}

    def register(self, name: str, task_vector: TaskVector) -> None:
        if name in self.task_vectors:
            raise KeyError(f"task vector {name!r} already registered")
        for k, t in task_vector.items():
            if k not in self.base:
                raise KeyError(f"task vector {name!r} covers {k!r}, which is not in the base")
            if t.shape != self.base[k].shape:
                raise ValueError(f"task vector {name!r}: shape mismatch for {k!r}")
        self.task_vectors[name] = task_vector

    @property
    def task_names(self) -> list[str]:
        return list(self.task_vectors)

    def combine(self, coefficients: Coefficients) -> dict[str, torch.Tensor]:
        """Effective parameters θ_base + Σ_i α_i τ_i, returned in the base dtype.

        Accumulated in ``promote_types(base dtype, float32)``: fp32 for half/bf16/fp32 bases, float64
        for float64 bases (so α = 0 restores any base bit-for-bit). With tensor coefficients that
        require grad the result is differentiable w.r.t. α.
        """
        coeffs = _coefficient_list(coefficients, self.task_names)
        out: dict[str, torch.Tensor] = {}
        for k, b in self.base.items():
            cd = _compute_dtype(b.dtype)
            acc = b.to(cd)
            for a, tv in zip(coeffs, self.task_vectors.values()):
                if k in tv:
                    acc = acc + a * tv[k].to(device=acc.device, dtype=cd)
            out[k] = acc.to(b.dtype)
        return out

    @torch.no_grad()
    def apply_to(self, model: nn.Module, coefficients: Coefficients) -> dict[str, torch.Tensor]:
        """Write ``combine(coefficients)`` into ``model`` bit-for-bit; returns the written tensors."""
        coeffs = [float(a) for a in _coefficient_list(coefficients, self.task_names)]
        values = self.combine(coeffs)
        params = dict(model.named_parameters())
        for k, v in values.items():
            if k not in params:
                raise KeyError(f"model has no parameter {k!r}")
            params[k].copy_(v.to(device=params[k].device, dtype=params[k].dtype))
        return values

    def functional_call(
        self,
        model: nn.Module,
        coefficients: Coefficients,
        args: Any = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Run ``model`` with θ(α) substituted via ``torch.func.functional_call`` (differentiable in α).

        Parameters outside the prior keep the model's own (trainable) tensors.
        """
        values = self.combine(coefficients)
        device = next(model.parameters()).device
        values = {k: v.to(device) for k, v in values.items()}
        return torch.func.functional_call(model, values, args, dict(kwargs or {}))

    def task_matrix(self, name: str) -> torch.Tensor:
        """``[K, *shape]`` stack of the task vectors at parameter ``name`` (zeros where uncovered)."""
        b = self.base[name]
        return torch.stack(
            [
                tv[name].float() if name in tv else torch.zeros(b.shape, dtype=torch.float32)
                for tv in self.task_vectors.values()
            ]
        )

    def project(self, name: str, displacement: torch.Tensor, rcond: float = 1e-10) -> SpanProjection:
        """Least-squares projection of a displacement of parameter ``name`` onto span{τ_k[name]}."""
        return project_onto_span(displacement, self.task_matrix(name), rcond=rcond)

    def state_dict(self) -> dict[str, Any]:
        return {
            "base": {k: v.cpu() for k, v in self.base.items()},
            "task_vectors": {n: tv.state_dict() for n, tv in self.task_vectors.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.base = {k: v.clone() for k, v in state["base"].items()}
        self.task_vectors = {}
        for n, tv in state["task_vectors"].items():
            self.register(n, TaskVector.from_state_dict(tv))


def _stack_task_vectors(
    task_vectors: TaskVectorPrior | Mapping[str, TaskVector] | Sequence[TaskVector], names: Iterable[str]
) -> dict[str, torch.Tensor]:
    if isinstance(task_vectors, TaskVectorPrior):
        tvs = list(task_vectors.task_vectors.values())
    elif isinstance(task_vectors, Mapping):
        tvs = list(task_vectors.values())
    else:
        tvs = list(task_vectors)
    if not tvs:
        raise ValueError("at least one task vector is required")
    out: dict[str, torch.Tensor] = {}
    for n in names:
        shapes = {tuple(tv[n].shape) for tv in tvs if n in tv}
        if not shapes:
            raise KeyError(f"no task vector covers parameter {n!r}")
        if len(shapes) > 1:
            raise ValueError(f"task vectors disagree on the shape of {n!r}: {shapes}")
        shape = shapes.pop()
        out[n] = torch.stack(
            [tv[n].float() if n in tv else torch.zeros(shape, dtype=torch.float32) for tv in tvs]
        )
    return out


def _all_names(task_vectors: TaskVectorPrior | Mapping[str, TaskVector] | Sequence[TaskVector]) -> list[str]:
    if isinstance(task_vectors, TaskVectorPrior):
        tvs = list(task_vectors.task_vectors.values())
    elif isinstance(task_vectors, Mapping):
        tvs = list(task_vectors.values())
    else:
        tvs = list(task_vectors)
    seen: dict[str, None] = {}
    for tv in tvs:
        for n in tv.names():
            seen.setdefault(n, None)
    return list(seen)


# ====================================================================================== subspace rule
@UPDATE_RULES.register("prior_subspace")
class PriorSubspaceRule(UpdateRule):
    """Project the candidate update Δ (after Adam/decay) onto or away from span{τ_k[p]} per parameter.

    ``mode="onto"``: Δ' = U Uᵀ Δ, ``mode="away"``: Δ' = Δ − U Uᵀ Δ, where U is an orthonormal basis of
    the span of the K task vectors restricted to parameter ``p`` (computed once by SVD, fp32).
    Parameters not covered by any task vector are untouched. Acts in ``transform_update`` so the
    optimizer state (moments) is unchanged; only the written displacement is constrained.
    """

    name = "prior_subspace"

    def __init__(
        self,
        task_vectors: TaskVectorPrior | Mapping[str, TaskVector] | Sequence[TaskVector],
        *,
        mode: str = "onto",
        names: Iterable[str] | None = None,
        rcond: float = 1e-6,
    ) -> None:
        if mode not in ("onto", "away"):
            raise ValueError(f"mode must be 'onto' or 'away', got {mode!r}")
        self.mode = mode
        keys = list(names) if names is not None else _all_names(task_vectors)
        stacked = _stack_task_vectors(task_vectors, keys)
        self._basis: dict[str, torch.Tensor] = {
            n: orthonormal_span_basis(t.reshape(t.shape[0], -1).T, rcond) for n, t in stacked.items()
        }
        self._device_cache: dict[tuple[str, torch.device], torch.Tensor] = {}

    def basis(self, name: str, device: torch.device | None = None) -> torch.Tensor:
        u = self._basis[name]
        if device is None:
            return u
        key = (name, device)
        if key not in self._device_cache:
            self._device_cache[key] = u.to(device)
        return self._device_cache[key]

    def tracks(self, name: str, param: torch.nn.Parameter) -> bool:
        return name in self._basis

    def transform_update(
        self, ctx: StepContext, name: str, param: torch.nn.Parameter, old: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor | None:
        u = self.basis(name, delta.device)
        d = delta.reshape(-1)
        proj = u @ (u.T @ d)
        out = proj if self.mode == "onto" else d - proj
        return out.reshape(delta.shape)

    def state_dict(self) -> dict[str, Any]:
        return {"mode": self.mode}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if "mode" in state and state["mode"] != self.mode:
            raise ValueError(f"checkpoint was saved with mode={state['mode']!r}, rule has mode={self.mode!r}")


# ====================================================================================== P²-MOPD rule
def _finite_grad_norm(named_params: Iterable[tuple[str, torch.nn.Parameter]]) -> bool:
    """Whether the global L2 norm of all ``.grad`` (fp32, as ``clip_grad_norm_`` computes it) is finite."""
    norms = [torch.linalg.vector_norm(p.grad.detach().float()) for _, p in named_params if p.grad is not None]
    if not norms:
        return True
    return bool(torch.isfinite(torch.linalg.vector_norm(torch.stack([n.cpu() for n in norms]))))


def default_prior_modules(names: Iterable[str]) -> list[str]:
    """Parameter names of the linear projections q/k/v/o and gate/up/down (``*_proj.weight``)."""
    return [n for n in names if PROJECTION_PATTERN.search(n)]


def _group_key(name: str, granularity: str) -> str:
    if granularity == "module":
        return name
    if granularity == "model":
        return "model"
    m = _LAYER_PATTERN.search(name)
    if m is None:
        raise ValueError(f"granularity='layer' needs a decoder-layer index in {name!r} ('layers.<i>.')")
    return f"layer{m.group(1)}"


@UPDATE_RULES.register("p2_prior")
class P2PriorRule(UpdateRule):
    """Two-channel parameter-prior update of P²-MOPD (Algorithm 1) for :class:`HookedOptimizer`.

    Conceptual model per prior module m (a linear projection weight)::

        W_m = B_m + Σ_k α_{k,m} τ_{k,m}        τ frozen task vectors, B and α trainable.

    The optimizer keeps the *materialised* W (so the forward pass is the unmodified model). One step:

    0. non-finite guard (paper App. B): if the global weight-gradient norm or the coefficient
       gradient is non-finite, the **whole step is skipped** (``skip_nonfinite=True``) via
       ``HookedOptimizer.request_skip()`` from ``post_grad``: the inner optimizer, its state, the
       weights, α, its Adam moments and step count are untouched, and the step is counted as
       skipped (not as an applied update); ``nonfinite_skips`` counts these steps. With
       ``skip_nonfinite=False`` only the coefficient channel is skipped (α is never poisoned).
    1. ``post_grad``: g_c = Σ_{(k,m)→c} <G_m, τ_{k,m}> from the **unclipped** weight gradient G
       (chain rule ∂L/∂α_{k,m} = <∂L/∂W_m, τ_{k,m}>; tied coefficients sum their contributions).
    2. weight channel (inner optimizer): G clipped to global norm c_W, AdamW step on W. This is an
       AdamW step on B because ∂L/∂B = G — valid **only with weight decay 0 on prior modules**
       (decay would shrink B + Σατ instead of B). :meth:`attach` validates this and raises.
    3. coefficient channel: g clipped to norm ``clip_norm`` (c_α; scale ``min(1, c_α/(‖g‖+1e-6))``
       like ``clip_grad_norm_``), then Adam (own β1, β2, eps, no decay) on the free coefficients
       with learning rate η_α(t) = ``lr`` · factor(t), where t is the HookedOptimizer's 1-based
       nominal step (the counter HF's LR scheduler follows; skipped steps advance it, as in App. B
       step 7, but not Adam's bias-correction count). factor = ``lr_lambda(t)`` if given, else
       ``min(1, t / warmup_steps)`` (paper Table 10 reports a warm-up then constant η_α; the linear
       ramp is this implementation's choice, and unlike HF's linear warm-up step 1 is not lr 0), else 1. The callable is configuration and is
       not saved; the position is restored with the optimizer state. If no prior module received a
       gradient the coefficient step is skipped (as torch skips parameters whose grad is None).
    4. ``pre_write``: W_{t+1,m} = W_after_AdamW + Σ_k (α_{t+1} − α_t)_{k,m} τ_{k,m}, computed in
       ``promote_types(param dtype, float32)`` (fp32 for bf16/fp16/fp32 weights) and rounded once.

    Requirements (documented, partly validated):

    * Clipping of W must happen **after** step 1: run HF/TRL with ``max_grad_norm=0`` and let the
      wrapper clip (``clip_in_wrapper=c_W`` → ``HookedOptimizer(clip_norm=c_W)``). The rule refuses
      to run when the trainer mixin reports ``hf_max_grad_norm > 0``.
    * Rules registered *before* this one may edit gradients first; register it first for raw G.
    * No other rule may rewrite a prior module's update (``transform_update`` / ``pre_write``, e.g.
      :class:`~research_rl.optim.update_interface.MaskRule` or :class:`PriorSubspaceRule`): after
      such a rule W ≠ B + Σ α τ (or a frozen element moves). This is checked at attach and at every
      step and raises unless ``allow_foreign_writes=True`` (then the identity is the caller's
      responsibility). Rules that only observe (``post_step``) or touch other parameters are fine.
    * The rule must be attached to its optimizer (:meth:`attach`, or :meth:`callback` for TRL,
      which attaches at ``on_train_begin``); an unattached rule raises at the first step.

    Granularity: ``"module"`` (K coefficients per module), ``"layer"`` (shared by all prior modules
    of a decoder layer), ``"model"`` (K in total). Initialisation: ``init`` = λ (scalar or K values,
    e.g. ``1/K`` for a task-arithmetic start; the model must then hold the TA merge, see
    :meth:`materialize`) or ``None`` for α = 0 (model holds the base).

    Memory: the task vectors are held as ``[K, *shape]`` tensors in ``tau_dtype`` (default fp32) on
    CPU for every prior module, i.e. K × (#prior params) × 4 bytes (Qwen3-4B projections, K=4:
    ≈ 58 GB). With ``cache_tau_on_device=True`` (default) a second copy is cached per device on first
    use and never evicted — on an accelerator that is the same amount of device memory on top of
    the model and its AdamW state. ``cache_tau_on_device=False`` moves each module's τ to the device
    inside the per-module loop and drops it (one module resident at a time, more host→device
    traffic); ``tau_dtype=torch.bfloat16`` halves both copies (τ is rounded once at construction and
    the prior is then *defined* by the rounded τ; all arithmetic is still fp32). Memory-mapped
    streaming from checkpoints (paper App. B) is not implemented. Device behaviour is NOT_TESTED
    (CPU only).

    State (coefficients, Adam moments, step count, trajectory, skip count) is tensors/primitives
    only and lives in the optimizer state file, so HF checkpoint resume restores it. Task vectors
    are configuration and are not saved. This is the parameter-prior optimisation of P²-MOPD only;
    it works with any loss (the MOPD distillation loss is not implemented in ResearchRL, see
    :mod:`research_rl.trainers.multi_teacher` for the available building blocks).
    """

    name = "p2_prior"

    def __init__(
        self,
        task_vectors: TaskVectorPrior | Mapping[str, TaskVector] | Sequence[TaskVector],
        modules: Sequence[str] | None = None,
        *,
        granularity: str = "module",
        init: float | Sequence[float] | None = None,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        clip_norm: float | None = None,
        record_trajectory: bool = True,
        warmup_steps: int = 0,
        lr_lambda: Callable[[int], float] | None = None,
        skip_nonfinite: bool = True,
        allow_foreign_writes: bool = False,
        tau_dtype: torch.dtype = torch.float32,
        cache_tau_on_device: bool = True,
    ) -> None:
        if granularity not in ("module", "layer", "model"):
            raise ValueError(f"granularity must be module/layer/model, got {granularity!r}")
        names = list(modules) if modules is not None else default_prior_modules(_all_names(task_vectors))
        if not names:
            raise ValueError("no prior modules: pass modules=... or task vectors covering *_proj.weight")
        self.modules = names
        self.granularity = granularity
        self.lr = float(lr)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.clip_norm = clip_norm
        self.record_trajectory = record_trajectory
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if warmup_steps and lr_lambda is not None:
            raise ValueError("pass either warmup_steps or lr_lambda, not both")
        self.warmup_steps = int(warmup_steps)
        self.lr_lambda = lr_lambda
        self.skip_nonfinite = skip_nonfinite
        self.allow_foreign_writes = allow_foreign_writes
        self.cache_tau_on_device = cache_tau_on_device
        if not tau_dtype.is_floating_point:
            raise ValueError("tau_dtype must be a floating-point dtype")
        # name -> [K, *shape] in tau_dtype (CPU)
        self._tau = {n: t.to(tau_dtype) for n, t in _stack_task_vectors(task_vectors, names).items()}
        self.num_tasks = int(next(iter(self._tau.values())).shape[0])
        k = self.num_tasks

        self.group_names: list[str] = []
        group_index: dict[str, int] = {}
        self._index: dict[str, torch.Tensor] = {}
        for n in names:
            g = _group_key(n, granularity)
            if g not in group_index:
                group_index[g] = len(self.group_names)
                self.group_names.append(g)
            self._index[n] = torch.arange(k) + group_index[g] * k
        self.num_free = len(self.group_names) * k

        if init is None:
            lam = torch.zeros(k)
        elif isinstance(init, int | float):
            lam = torch.full((k,), float(init))
        else:
            lam = torch.tensor([float(x) for x in init])
            if lam.shape != (k,):
                raise ValueError(f"init needs {k} values (one per task vector), got {lam.numel()}")
        self.coefficients = lam.repeat(len(self.group_names)).float()
        self.exp_avg = torch.zeros(self.num_free)
        self.exp_avg_sq = torch.zeros(self.num_free)
        self.step_count = 0
        self.trajectory: list[torch.Tensor] = [self.coefficients.clone()] if record_trajectory else []
        self.last_grad: torch.Tensor | None = None
        self.last_clipped_grad: torch.Tensor | None = None
        self.last_lr: float | None = None
        self.nonfinite_skips = 0
        self._delta: torch.Tensor | None = None
        self._tau_cache: dict[tuple[str, torch.device], torch.Tensor] = {}
        self._optimizer: HookedOptimizer | None = None
        self._prior_ids: set[int] = set()
        self._prior_params: list[torch.nn.Parameter] = []

    # ------------------------------------------------------------------ setup
    def attach(self, optimizer: Any) -> None:
        """Bind to the optimizer and validate it (fails closed).

        ``optimizer`` may be a :class:`HookedOptimizer` or any wrapper around one (accelerate).
        Raises if this rule is not in its rules, a prior module is not optimised or has
        ``requires_grad=False``, a param group holding a prior module has non-zero
        ``weight_decay``, or another rule rewrites a prior module's update (unless
        ``allow_foreign_writes``). The frozen/decay/foreign-write checks are repeated at every step.
        """
        hooked = unwrap_hooked(optimizer)
        if hooked is None:
            raise TypeError("P2PriorRule needs a HookedOptimizer (or a wrapper around one)")
        if not any(r is self for r in hooked.rules):
            raise ValueError("this P2PriorRule is not registered in the HookedOptimizer's rules")
        named = dict(hooked.named_parameters())
        missing = [n for n in self.modules if n not in named]
        if missing:
            raise ValueError(f"prior modules not optimised by the optimizer: {missing[:5]}")
        for n in self.modules:
            if tuple(named[n].shape) != tuple(self._tau[n].shape[1:]):
                raise ValueError(f"task-vector shape for {n!r} does not match the parameter")
        self._prior_params = [named[n] for n in self.modules]
        self._prior_ids = {id(p) for p in self._prior_params}
        self._optimizer = hooked
        self._validate()

    def _validate(self) -> None:
        assert self._optimizer is not None
        frozen = [n for n, p in zip(self.modules, self._prior_params) if not p.requires_grad]
        if frozen:
            raise ValueError(
                f"prior modules with requires_grad=False: {frozen[:5]}; the two-channel update needs trainable W "
                "(drop them from `modules` instead of freezing them)"
            )
        for i, group in enumerate(self._optimizer.inner.param_groups):
            if any(id(p) in self._prior_ids for p in group["params"]) and float(
                group.get("weight_decay", 0.0)
            ):
                raise ValueError(
                    f"param group {i} holds P2-prior modules with weight_decay={group['weight_decay']}; it must "
                    "be 0 (with decay, optimising the materialised W is no longer equivalent to optimising B)"
                )
        if self.allow_foreign_writes:
            return
        for rule in self._optimizer.rules:
            if rule is self or not (_overrides(rule, "transform_update") or _overrides(rule, "pre_write")):
                continue
            clash = [n for n, p in zip(self.modules, self._prior_params) if rule.tracks(n, p)]
            if clash:
                raise ValueError(
                    f"rule {rule.name!r} rewrites the update of P2-prior modules {clash[:3]} (transform_update/"
                    "pre_write): in either order this breaks W = B + sum(alpha*tau) or the other rule's guarantee "
                    "(e.g. a MaskRule's frozen elements move). Drop those modules from one of the rules, or pass "
                    "allow_foreign_writes=True if you have checked the composition yourself."
                )

    def on_attach(self, optimizer: Any) -> None:
        """HookedOptimizer calls this when the rule is registered: validate immediately (fail closed)."""
        self.attach(optimizer)

    def callback(self) -> TrainerCallback:
        """A ``TrainerCallback`` that calls :meth:`attach` with the trainer's optimizer at train begin."""
        return P2PriorAttachCallback(self)

    @torch.no_grad()
    def materialize(self, model: nn.Module, base: Mapping[str, torch.Tensor]) -> None:
        """Write W_m = B_m + Σ_k α_{k,m} τ_{k,m} (current α) into ``model`` for every prior module."""
        params = dict(model.named_parameters())
        for n in self.modules:
            p = params[n]
            cd = _compute_dtype(p.dtype)
            alpha = self.coefficients[self._index[n]].to(device=p.device, dtype=cd)
            tau = self._tau_on(n, p.device).to(cd)
            w = base[n].to(device=p.device, dtype=cd) + torch.tensordot(alpha, tau, dims=1)
            p.copy_(w.to(p.dtype))

    def _tau_on(self, name: str, device: torch.device) -> torch.Tensor:
        """τ of ``name`` on ``device`` (in ``tau_dtype``); cached only with ``cache_tau_on_device``."""
        if not self.cache_tau_on_device:
            return self._tau[name].to(device)
        key = (name, device)
        if key not in self._tau_cache:
            self._tau_cache[key] = self._tau[name].to(device)
        return self._tau_cache[key]

    # ------------------------------------------------------------------ analysis
    def coefficient_table(self) -> dict[str, torch.Tensor]:
        """Effective α_{·,m} (shape [K]) for each prior module m."""
        return {n: self.coefficients[self._index[n]].clone() for n in self.modules}

    def free_coefficients(self) -> dict[str, torch.Tensor]:
        """Free coefficients per group (module / layer / model), shape [K] each."""
        k = self.num_tasks
        return {g: self.coefficients[i * k : (i + 1) * k].clone() for i, g in enumerate(self.group_names)}

    def trajectory_tensor(self) -> torch.Tensor:
        """``[T+1, F]`` free coefficients after each coefficient update (row 0 = initialisation)."""
        if not self.trajectory:
            return torch.zeros(0, self.num_free)
        return torch.stack(self.trajectory)

    def coefficient_gradient(self, named_params: Iterable[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
        """g [F] = tied Σ <G_m, τ_{k,m}> from the current ``.grad`` (fp32, on CPU)."""
        g = torch.zeros(self.num_free, dtype=torch.float32)
        for n, p in named_params:
            if n in self._index and p.grad is not None:
                tau = self._tau_on(n, p.grad.device).float()
                contrib = tau.reshape(self.num_tasks, -1) @ p.grad.detach().float().reshape(-1)
                g.index_add_(0, self._index[n], contrib.cpu())
        return g

    # ------------------------------------------------------------------ hooks
    def tracks(self, name: str, param: torch.nn.Parameter) -> bool:
        return name in self._index

    def post_grad(self, ctx: StepContext, named_params: list[tuple[str, torch.nn.Parameter]]) -> None:
        if self._optimizer is None:
            raise RuntimeError(
                "P2PriorRule is not attached: call rule.attach(optimizer) (or add rule.callback() to the "
                "trainer callbacks) so weight decay on prior modules can be validated"
            )
        if float(ctx.extras.get("hf_max_grad_norm", 0.0)) > 0:
            raise RuntimeError(
                "P2PriorRule needs the UNCLIPPED weight gradient, but the trainer clips before optimizer.step() "
                "(max_grad_norm > 0). Use OptimSpec.clip_in_wrapper=c_W (HookedOptimizer clip_norm) and "
                "max_grad_norm=0."
            )
        self._validate()
        self._delta = None
        if self.skip_nonfinite and not _finite_grad_norm(named_params):
            self._skip_step(named_params)
            return
        if not any(n in self._index and p.grad is not None for n, p in named_params):
            return
        g = self.coefficient_gradient(named_params)
        self.last_grad = g.clone()
        if not bool(torch.isfinite(g).all()):
            if self.skip_nonfinite:
                self._skip_step(named_params)
            else:
                self.nonfinite_skips += 1  # coefficient channel only; the weight step still runs
            return
        if self.clip_norm is not None and self.clip_norm > 0:
            total = torch.linalg.vector_norm(g)
            g = g * torch.clamp(self.clip_norm / (total + _CLIP_EPS), max=1.0)
        self.last_clipped_grad = g.clone()
        lr = self.lr * self.lr_factor(ctx.nominal_step)
        self.last_lr = lr
        # Adam (no decay), same arithmetic as torch.optim.Adam.
        b1, b2 = self.betas
        self.step_count += 1
        self.exp_avg.mul_(b1).add_(g, alpha=1 - b1)
        self.exp_avg_sq.mul_(b2).addcmul_(g, g, value=1 - b2)
        bc1 = 1 - b1**self.step_count
        bc2 = 1 - b2**self.step_count
        denom = (self.exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(self.eps)
        new = self.coefficients - (lr / bc1) * (self.exp_avg / denom)
        self._delta = new - self.coefficients
        self.coefficients = new
        if self.record_trajectory:
            self.trajectory.append(new.clone())

    def lr_factor(self, step: int) -> float:
        """Multiplier of ``lr`` at 1-based optimizer step ``step`` (``lr_lambda`` / warm-up / 1)."""
        if self.lr_lambda is not None:
            return float(self.lr_lambda(step))
        if self.warmup_steps > 0:
            return min(1.0, step / self.warmup_steps)
        return 1.0

    def _skip_step(self, named_params: list[tuple[str, torch.nn.Parameter]]) -> None:
        """Non-finite gradient: veto the current optimizer step; keep α/moments."""
        assert self._optimizer is not None
        self._optimizer.request_skip()
        self.nonfinite_skips += 1
        self._delta = None

    def pre_write(
        self, ctx: StepContext, name: str, param: torch.nn.Parameter, old: torch.Tensor, new: torch.Tensor
    ) -> torch.Tensor | None:
        if self._delta is None:
            return None
        d = self._delta[self._index[name]]
        if not bool(d.any()):
            return None
        cd = _compute_dtype(new.dtype)
        corr = torch.tensordot(
            d.to(device=new.device, dtype=cd), self._tau_on(name, new.device).to(cd), dims=1
        )
        return (new.to(cd) + corr).to(new.dtype)

    def on_skip(self, ctx: StepContext) -> None:
        self._delta = None

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> dict[str, Any]:
        return {
            "coefficients": self.coefficients.clone(),
            "exp_avg": self.exp_avg.clone(),
            "exp_avg_sq": self.exp_avg_sq.clone(),
            "step": self.step_count,
            "trajectory": self.trajectory_tensor(),
            "granularity": self.granularity,
            "group_names": list(self.group_names),
            "num_tasks": self.num_tasks,
            "nonfinite_skips": self.nonfinite_skips,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("granularity", self.granularity) != self.granularity or list(
            state.get("group_names", self.group_names)
        ) != list(self.group_names):
            raise ValueError("checkpoint coefficient layout (granularity/groups) does not match this rule")
        if int(state.get("num_tasks", self.num_tasks)) != self.num_tasks:
            raise ValueError("checkpoint has a different number of task vectors")
        self.coefficients = state["coefficients"].detach().clone().float().cpu()
        self.exp_avg = state["exp_avg"].detach().clone().float().cpu()
        self.exp_avg_sq = state["exp_avg_sq"].detach().clone().float().cpu()
        self.step_count = int(state["step"])
        traj = state.get("trajectory")
        self.trajectory = list(traj.unbind(0)) if isinstance(traj, torch.Tensor) and traj.numel() else []
        self.nonfinite_skips = int(state.get("nonfinite_skips", 0))
        self._delta = None


class P2PriorAttachCallback(TrainerCallback):
    """Attaches a :class:`P2PriorRule` to the (accelerate-wrapped) trainer optimizer at train begin."""

    def __init__(self, rule: P2PriorRule) -> None:
        self.rule = rule

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            raise RuntimeError("P2PriorAttachCallback: the trainer passed no optimizer to on_train_begin")
        self.rule.attach(optimizer)
