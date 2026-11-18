"""Forward-only evolution-strategies update operator (``CAPABILITY.md`` section 1.7).

:class:`ESOperator` implements :class:`research_rl.capability.types.UpdateOperator` over any
:class:`research_rl.capability.types.ParameterSpace` (e.g. the spaces of
:mod:`research_rl.capability.spaces`). It never calls ``backward``: every parameter change is a
write through the space.

One :meth:`ESOperator.step` (update ``t``) does exactly this:

1. ``space.rebase()`` - the base is the model's weights at the start of the step (see "Base").
2. For every member ``i = 0 .. population-1``: write ``sigma * eps_i`` through the space, call
   ``fitness(model, t, i)`` under ``torch.no_grad()``, then ``space.write(None)`` (base restored
   bit-identically). The restore also runs when ``fitness`` raises, and a non-finite
   ``reward_v1`` / ``reward_v2`` raises :class:`ESError` *after* the restore.
3. Fitness ``f_i = reward_v1`` of member ``i`` (the training reward, V1). Shaping:

   * ``"zscore"``: ``s_i = (f_i - mean(f)) / std(f)`` with the population std (ddof = 0); all
     ``s_i = 0`` when ``std(f) <= 1e-12`` (constant fitness gives a zero step).
   * ``"rank"``: centred ranks ``s_i = r_i / (n - 1) - 0.5`` with ``r_i`` the 0-based ascending
     rank of ``f_i`` (ties get their average rank), so ``s_i`` is in ``[-0.5, 0.5]``.

4. Plain ES direction ``d = sum_i s_i * eps_i / (n * sigma)``.
   With ``consolidate=True`` (agreement re-weighting in the spirit of CoRP-style consolidation;
   this exact formula is this package's definition, not a reproduction of a published
   implementation): with ``c_i = s_i * eps_i`` the member's contribution,
   ``a_i = max(0, cos(c_i, d))`` (0 when ``c_i = 0`` or ``d = 0``), ``w_i = s_i * a_i`` and
   ``d_c = sum_i w_i * eps_i / (n * sigma)`` rescaled to ``||d_c|| = ||d||`` (only the direction
   changes; members whose contribution disagrees with the population direction get weight 0).
   Because ``d . d_c > 0`` whenever ``d != 0``, the consolidated direction is never opposed to
   ``d``.
5. Step ``delta = lr * d`` (or ``lr * d_c``); if ``max_step_norm`` is set and
   ``||delta||_2 > max_step_norm`` it is rescaled to norm ``max_step_norm``.
6. ``space.write(delta)`` then ``space.rebase()``: the accepted direction is now the base. When
   ``delta`` is exactly zero (e.g. constant fitness) both calls are skipped and the model stays
   bit-identical to the base.

Step scale and defaults
-----------------------
Under ``"zscore"`` shaping the step size does not depend on the fitness scale. For antithetic
pairs on a locally linear objective (``s_{2j+1} ~ -s_{2j}``) the RMS per-coordinate step is about
``(lr / sigma) * sqrt(2 / population)``, i.e. about ``(lr / sigma**2) * sqrt(2 / population)``
times the per-coordinate perturbation ``sigma`` (an approximation for zscore + antithetic only;
``"rank"`` steps are roughly 0.3x as large because ``|s_i| <= 0.5``). The defaults
``population=8, sigma=0.02, lr=0.001`` give a ratio of about 1.25: the step is of the order of one
perturbation. Validated only on this module's CPU toy problems (a multiplicative-gain objective
over 16 or 64 :class:`~research_rl.capability.spaces.ChannelGainSpace` channels of a tiny Llama and
a quadratic over a small ``nn.Linear``), where they improve the objective for every tested seed; a
larger ratio (e.g. the former ``lr=0.01``, ratio about 12.5) overshoots and made the gain
objective worse. Tune ``lr`` with ``sigma**2`` for a real task; ``max_step_norm`` (default
``None``) is the hard bound.

Perturbations (common random numbers)
--------------------------------------
``eps`` for noise index ``j`` of update ``t`` is ``torch.randn(dim, dtype=float32)`` drawn on CPU
from a fresh ``torch.Generator`` seeded with the first 8 bytes (little endian, top bit cleared) of
``SHA-256("research_rl.capability.es:{seed}:{t}:{j}")``. With ``antithetic=True`` member ``i``
uses noise index ``j = i // 2`` with sign ``+1`` for even ``i`` and ``-1`` for odd ``i`` (so the
pair shares one draw); otherwise ``j = i`` with sign ``+1``. A perturbation therefore depends only
on ``(seed, t, i)``: it is reproducible, independent of call history and of resume. Every member is
evaluated on the same items because the caller's ``fitness`` must draw its items from
``update_index`` only (never from ``member_index``); this operator cannot check that. Noise is
regenerated when needed instead of being stored, so memory is ``O(dim)``, not
``O(population * dim)`` (a :class:`~research_rl.capability.spaces.FullParameterSpace` over a large
model is still impractical: one CPU float32 vector of ``dim`` elements is drawn per member).

Report and alarm feed
---------------------
The :class:`~research_rl.capability.types.OperatorReport` of an update carries the rollouts of
**every** member, concatenated in member order (``v1_pass``, ``v2_pass``, ``completion_lengths``):
the tier-1 alarm sees the whole population's training rollouts, i.e. the rollouts actually used to
form the update. ``reward_v1`` is the population mean of ``f_i`` and ``reward_v2`` the mean of the
members' ``reward_v2`` (``None`` if any member returned ``None``). ``completion_lengths`` is empty
unless every member returned lengths. ``screened = 0``: ES uses one scalar per member, so
rollout-level V1/V2 screening does not apply. ``forward_passes = sum_i (1 +
extra_i["extra_forward_passes"])``: one per fitness call plus the extra generation calls the
fitness reports in ``FitnessResult.extra["extra_forward_passes"]`` (a non-negative int, default 0;
e.g. a fitness that runs 4 batched ``generate`` calls reports 3). The unperturbed model is not
evaluated by the operator. ``backward_passes = 0`` always. ``extra`` holds JSON-able diagnostics
(fitness, shaped values, step norms, agreements).

Base
----
The step starts with ``space.rebase()`` so it always perturbs around the model's current weights,
even if something wrote the model between steps. :meth:`ESOperator.on_decision` also re-bases from
the current model weights for every decision (the gated loop restores / shrinks / reverts weights
*before* calling it), never by subtracting regenerated noise. :meth:`ESOperator.load_state_dict`
re-bases too (load the model weights first).

Verified on CPU with tiny models only (``tests/research_rl/test_capability_es.py``); nothing here
has run on a GPU (NOT_TESTED).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch
from torch import nn

from research_rl.capability.types import GateDecision, OperatorReport, ParameterSpace

FitnessShaping = Literal["zscore", "rank"]
_ZSCORE_EPS = 1e-12
_STATE_VERSION = 1


class ESError(ValueError):
    """Invalid ES configuration, fitness result or state."""


# ============================================================================== fitness result
def _as_bool_tuple(values: Sequence[bool], what: str) -> tuple[bool, ...]:
    out: list[bool] = []
    for v in values:
        if isinstance(v, bool):
            out.append(v)
        elif isinstance(v, int) and v in (0, 1):
            out.append(bool(v))
        else:
            raise ESError(f"{what} entries must be bools, got {v!r}")
    return tuple(out)


@dataclass(frozen=True)
class FitnessResult:
    """Outcome of evaluating one ES member on the update's training items.

    ``reward_v1`` is the fitness the operator maximises (mean V1 reward over the member's
    rollouts; finite). ``reward_v2`` is the strict-verifier mean (finite) or ``None``.
    ``v1_pass`` / ``v2_pass`` are per-rollout verifier outcomes (same length) and ``lengths`` the
    per-rollout completion lengths in tokens (empty, or the same length). ``extra`` may carry
    ``"extra_forward_passes"`` (non-negative int, default 0): generation calls beyond the one
    the operator counts per fitness call. Sequences are stored as tuples.
    """

    reward_v1: float
    reward_v2: float | None
    v1_pass: tuple[bool, ...]
    v2_pass: tuple[bool, ...]
    lengths: tuple[int, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.reward_v1, bool) or not isinstance(self.reward_v1, (int, float)):
            raise ESError(f"reward_v1 must be a number, got {self.reward_v1!r}")
        if self.reward_v2 is not None and (
            isinstance(self.reward_v2, bool) or not isinstance(self.reward_v2, (int, float))
        ):
            raise ESError(f"reward_v2 must be a number or None, got {self.reward_v2!r}")
        v1 = _as_bool_tuple(self.v1_pass, "v1_pass")
        v2 = _as_bool_tuple(self.v2_pass, "v2_pass")
        if len(v1) != len(v2):
            raise ESError(f"v1_pass ({len(v1)}) and v2_pass ({len(v2)}) differ in length")
        lengths = tuple(self.lengths)
        for n in lengths:
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise ESError(f"lengths must be non-negative ints, got {n!r}")
        if lengths and len(lengths) != len(v1):
            raise ESError(f"lengths ({len(lengths)}) must be empty or match v1_pass ({len(v1)})")
        if not isinstance(self.extra, Mapping):
            raise ESError(f"extra must be a mapping, got {type(self.extra).__name__}")
        extra_fp = self.extra.get("extra_forward_passes", 0)
        if isinstance(extra_fp, bool) or not isinstance(extra_fp, int) or extra_fp < 0:
            raise ESError(f"extra['extra_forward_passes'] must be a non-negative int, got {extra_fp!r}")
        object.__setattr__(self, "reward_v1", float(self.reward_v1))
        if self.reward_v2 is not None:
            object.__setattr__(self, "reward_v2", float(self.reward_v2))
        object.__setattr__(self, "v1_pass", v1)
        object.__setattr__(self, "v2_pass", v2)
        object.__setattr__(self, "lengths", lengths)

    @property
    def forward_passes(self) -> int:
        """Generation calls of this evaluation: ``1 + extra.get("extra_forward_passes", 0)``."""
        return 1 + int(self.extra.get("extra_forward_passes", 0))


FitnessFn = Callable[[nn.Module, int, int], FitnessResult]


# ============================================================================== config
@dataclass
class ESConfig:
    """Hyper-parameters of :class:`ESOperator` (JSON-able via :meth:`to_dict`).

    ``population``: members per update (>= 2; even when ``antithetic``). ``sigma``: perturbation
    scale (> 0) in space coordinates. ``lr``: step size (> 0) applied to the ES direction (see
    "Step scale and defaults" in the module docstring for how it relates to ``sigma``).
    ``fitness_shaping``: ``"zscore"`` or ``"rank"``. ``consolidate``: agreement re-weighting (see
    the module docstring). ``max_step_norm``: ``None`` or > 0, L2 clip of the step. ``seed``:
    root of every perturbation.
    """

    population: int = 8
    sigma: float = 0.02
    lr: float = 0.001
    antithetic: bool = True
    fitness_shaping: FitnessShaping = "zscore"
    consolidate: bool = False
    max_step_norm: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.population, bool) or not isinstance(self.population, int) or self.population < 2:
            raise ESError(f"population must be an int >= 2, got {self.population!r}")
        if self.antithetic and self.population % 2:
            raise ESError(f"population must be even with antithetic=True, got {self.population}")
        for name in ("sigma", "lr"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ESError(f"{name} must be a finite number > 0, got {v!r}")
        if self.fitness_shaping not in ("zscore", "rank"):
            raise ESError(f"fitness_shaping must be 'zscore' or 'rank', got {self.fitness_shaping!r}")
        m = self.max_step_norm
        if m is not None and (
            isinstance(m, bool) or not isinstance(m, (int, float)) or not math.isfinite(m) or m <= 0
        ):
            raise ESError(f"max_step_norm must be None or a finite number > 0, got {m!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ESError(f"seed must be an int, got {self.seed!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ESConfig:
        return cls(**dict(d))


# ============================================================================== shaping
def shape_fitness(fitness: Sequence[float], method: FitnessShaping) -> list[float]:
    """Shaped fitness values (see the module docstring for the exact formulas)."""
    f = [float(x) for x in fitness]
    n = len(f)
    if n == 0:
        return []
    if method == "zscore":
        mean = sum(f) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in f) / n)
        if std <= _ZSCORE_EPS:
            return [0.0] * n
        return [(x - mean) / std for x in f]
    if method == "rank":
        if n == 1:
            return [0.0]
        order = sorted(range(n), key=lambda i: f[i])
        ranks = [0.0] * n
        pos = 0
        while pos < n:
            end = pos
            while end + 1 < n and f[order[end + 1]] == f[order[pos]]:
                end += 1
            avg = (pos + end) / 2.0
            for k in range(pos, end + 1):
                ranks[order[k]] = avg
            pos = end + 1
        return [r / (n - 1) - 0.5 for r in ranks]
    raise ESError(f"unknown fitness_shaping {method!r}")


def _noise_seed(seed: int, update_index: int, noise_index: int) -> int:
    digest = hashlib.sha256(
        f"research_rl.capability.es:{seed}:{update_index}:{noise_index}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


# ============================================================================== operator
class ESOperator:
    """Evolution-strategies :class:`~research_rl.capability.types.UpdateOperator` over a space.

    ``fitness(model, update_index, member_index)`` evaluates the *current* (perturbed) weights and
    returns a :class:`FitnessResult`; it must not modify the model and must draw its items from
    ``update_index`` only. ``model`` passed to :meth:`step` must be the model the space writes to
    (checked when the space exposes a ``model`` attribute). ``optimizer`` is always ``None``.
    Only the ``ParameterSpace`` protocol (``dim``, ``parameter_names``, ``write``, ``rebase``)
    is used. See the module docstring for the exact update, seeding and report semantics.
    """

    optimizer: torch.optim.Optimizer | None = None

    def __init__(
        self, space: ParameterSpace, fitness: FitnessFn, config: ESConfig, *, name: str = "es"
    ) -> None:
        if not isinstance(space, ParameterSpace):
            raise ESError(f"space must implement ParameterSpace, got {type(space).__name__}")
        if not isinstance(config, ESConfig):
            raise ESError(f"config must be an ESConfig, got {type(config).__name__}")
        if not callable(fitness):
            raise ESError("fitness must be callable")
        dim = space.dim
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ESError(f"space.dim must be a positive int, got {dim!r}")
        self.name = name
        self.optimizer = None
        self.space = space
        self.fitness = fitness
        self.config = config
        self.dim = dim
        self.steps_done = 0
        self.last_update_index: int | None = None
        self.forward_passes_total = 0

    # ------------------------------------------------------------------ perturbations
    def noise_index(self, member_index: int) -> tuple[int, float]:
        """``(noise index j, sign)`` of member ``member_index``."""
        if isinstance(member_index, bool) or not isinstance(member_index, int):
            raise ESError(f"member_index must be an int, got {member_index!r}")
        if not 0 <= member_index < self.config.population:
            raise ESError(f"member_index {member_index} outside [0, {self.config.population})")
        if self.config.antithetic:
            return member_index // 2, (1.0 if member_index % 2 == 0 else -1.0)
        return member_index, 1.0

    def noise(self, update_index: int, noise_index: int) -> torch.Tensor:
        """Unsigned standard-normal draw ``eps`` (CPU float32, length ``dim``) for ``(t, j)``."""
        if isinstance(update_index, bool) or not isinstance(update_index, int) or update_index < 0:
            raise ESError(f"update_index must be an int >= 0, got {update_index!r}")
        g = torch.Generator(device="cpu")
        g.manual_seed(_noise_seed(self.config.seed, update_index, noise_index))
        return torch.randn(self.dim, generator=g, dtype=torch.float32)

    def perturbation(self, update_index: int, member_index: int) -> torch.Tensor:
        """Signed unit-scale perturbation of member ``i`` (the space receives ``sigma`` times it)."""
        j, sign = self.noise_index(member_index)
        e = self.noise(update_index, j)
        return e if sign > 0 else -e

    # ------------------------------------------------------------------ UpdateOperator
    def parameter_names(self) -> list[str]:
        return list(self.space.parameter_names())

    def _check_model(self, model: nn.Module) -> None:
        owner = getattr(self.space, "model", None)
        if owner is not None and owner is not model:
            raise ESError("step() got a model that is not the model the space writes to")

    def _evaluate(self, model: nn.Module, update_index: int) -> list[FitnessResult]:
        cfg = self.config
        results: list[FitnessResult] = []
        try:
            for i in range(cfg.population):
                self.space.write(cfg.sigma * self.perturbation(update_index, i))
                try:
                    with torch.no_grad():
                        res = self.fitness(model, update_index, i)
                finally:
                    self.space.write(None)
                if not isinstance(res, FitnessResult):
                    raise ESError(f"fitness must return a FitnessResult, got {type(res).__name__}")
                if not math.isfinite(res.reward_v1) or (
                    res.reward_v2 is not None and not math.isfinite(res.reward_v2)
                ):
                    raise ESError(
                        f"update {update_index} member {i}: non-finite fitness "
                        f"(reward_v1={res.reward_v1}, reward_v2={res.reward_v2})"
                    )
                results.append(res)
        finally:
            self.space.write(None)
        return results

    def _direction(self, update_index: int, shaped: Sequence[float]) -> tuple[torch.Tensor, list[float]]:
        """Plain (or consolidated) ES direction in float64, and the agreements (consolidate only)."""
        cfg = self.config
        n = cfg.population
        scale = 1.0 / (n * cfg.sigma)
        d = torch.zeros(self.dim, dtype=torch.float64)
        for i, s in enumerate(shaped):
            if s != 0.0:
                d.add_(self.perturbation(update_index, i).double(), alpha=s * scale)
        if not cfg.consolidate:
            return d, []
        d_norm = float(torch.linalg.vector_norm(d))
        agreements: list[float] = []
        dc = torch.zeros(self.dim, dtype=torch.float64)
        for i, s in enumerate(shaped):
            if s == 0.0 or d_norm == 0.0:
                agreements.append(0.0)
                continue
            c = self.perturbation(update_index, i).double() * s
            cos = float(torch.dot(c, d)) / (float(torch.linalg.vector_norm(c)) * d_norm)
            a = max(0.0, cos)
            agreements.append(a)
            if a > 0.0:
                dc.add_(c, alpha=a * scale)
        dc_norm = float(torch.linalg.vector_norm(dc))
        if dc_norm > 0.0:
            dc.mul_(d_norm / dc_norm)
        return dc, agreements

    def step(self, model: nn.Module, update_index: int) -> OperatorReport:
        """One ES update in place on ``model`` (see the module docstring)."""
        if isinstance(update_index, bool) or not isinstance(update_index, int) or update_index < 0:
            raise ESError(f"update_index must be an int >= 0, got {update_index!r}")
        self._check_model(model)
        cfg = self.config
        self.space.rebase()
        results = self._evaluate(model, update_index)
        fitness = [r.reward_v1 for r in results]
        shaped = shape_fitness(fitness, cfg.fitness_shaping)
        direction, agreements = self._direction(update_index, shaped)
        step = direction * cfg.lr
        norm_unclipped = float(torch.linalg.vector_norm(step))
        clipped = cfg.max_step_norm is not None and norm_unclipped > cfg.max_step_norm
        if clipped:
            assert cfg.max_step_norm is not None
            step.mul_(cfg.max_step_norm / norm_unclipped)
        norm = float(torch.linalg.vector_norm(step))
        if norm > 0.0:
            self.space.write(step)
            self.space.rebase()

        v1 = tuple(b for r in results for b in r.v1_pass)
        v2 = tuple(b for r in results for b in r.v2_pass)
        lengths: tuple[int, ...] = ()
        if all(len(r.lengths) == len(r.v1_pass) for r in results):
            lengths = tuple(n for r in results for n in r.lengths)
        v2_means = [r.reward_v2 for r in results]
        reward_v2 = (
            None
            if any(x is None for x in v2_means)
            else sum(x for x in v2_means if x is not None) / len(results)
        )
        forward = sum(r.forward_passes for r in results)
        self.steps_done += 1
        self.last_update_index = update_index
        self.forward_passes_total += forward
        mean = sum(fitness) / len(fitness)
        std = math.sqrt(sum((x - mean) ** 2 for x in fitness) / len(fitness))
        extra: dict[str, Any] = {
            "operator": self.name,
            "population": cfg.population,
            "sigma": cfg.sigma,
            "lr": cfg.lr,
            "fitness": fitness,
            "fitness_mean": mean,
            "fitness_std": std,
            "shaped": shaped,
            "step_norm": norm,
            "step_norm_unclipped": norm_unclipped,
            "clipped": bool(clipped),
            "rollouts_per_member": [len(r.v1_pass) for r in results],
        }
        if cfg.consolidate:
            extra["agreement"] = agreements
        return OperatorReport(
            update_index=update_index,
            reward_v1=mean,
            reward_v2=reward_v2,
            v1_pass=v1,
            v2_pass=v2,
            completion_lengths=lengths,
            screened=0,
            forward_passes=forward,
            backward_passes=0,
            extra=extra,
        )

    def on_decision(self, decision: GateDecision) -> None:
        """Re-base the space from the model's current weights (every decision; never by
        subtracting regenerated noise). The caller has already restored / shrunk / reverted."""
        if not isinstance(decision, GateDecision):
            raise ESError(f"decision must be a GateDecision, got {decision!r}")
        self.space.rebase()

    # ------------------------------------------------------------------ state
    def state_dict(self) -> dict[str, Any]:
        """JSON-able counters and config. Weights (and the space's base) are not included: they
        are part of the model / the gate's snapshots. Perturbations depend only on
        ``(seed, update_index, member_index)``, so no RNG state is needed."""
        return {
            "version": _STATE_VERSION,
            "name": self.name,
            "config": self.config.to_dict(),
            "dim": self.dim,
            "steps_done": self.steps_done,
            "last_update_index": self.last_update_index,
            "forward_passes_total": self.forward_passes_total,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Load counters after checking that the config and ``dim`` match exactly (a different
        seed, population, etc. would silently change the perturbation stream), then re-base the
        space from the model's current weights (load the model weights before calling this).

        Atomic: the whole state is validated first and the counters are assigned only after the
        re-base succeeded, so a failure leaves the operator unchanged. Raises :class:`ESError` for
        a non-mapping state, a missing key, a wrong version, ``dim`` or config, or a counter that
        is not a non-negative int (``last_update_index`` may be ``None``); errors of
        ``space.rebase()`` propagate unchanged.
        """
        if not isinstance(state, Mapping):
            raise ESError(f"ES state must be a mapping, got {type(state).__name__}")
        required = ("version", "dim", "config", "steps_done", "last_update_index", "forward_passes_total")
        missing = [k for k in required if k not in state]
        if missing:
            raise ESError(f"ES state is missing keys {missing}")
        if state["version"] != _STATE_VERSION:
            raise ESError(f"unsupported ES state version {state['version']!r}")

        def count(key: str, *, allow_none: bool = False) -> int | None:
            v = state[key]
            if v is None and allow_none:
                return None
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ESError(f"ES state {key!r} must be a non-negative int, got {v!r}")
            return v

        dim = count("dim")
        if dim != self.dim:
            raise ESError(f"state dim {dim} != space dim {self.dim}")
        if not isinstance(state["config"], Mapping):
            raise ESError(f"ES state 'config' must be a mapping, got {type(state['config']).__name__}")
        mine = self.config.to_dict()
        theirs = dict(state["config"])
        diff = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
        if diff:
            raise ESError(f"ES config differs from the checkpoint in {diff}")
        steps_done = count("steps_done")
        last = count("last_update_index", allow_none=True)
        forward_total = count("forward_passes_total")
        assert steps_done is not None and forward_total is not None
        self.space.rebase()
        self.steps_done = steps_done
        self.last_update_index = last
        self.forward_passes_total = forward_total


__all__ = [
    "ESConfig",
    "ESError",
    "ESOperator",
    "FitnessFn",
    "FitnessResult",
    "FitnessShaping",
    "shape_fitness",
]
