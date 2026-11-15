"""``HookedOptimizer``: controls the *actual* parameter displacement of any torch optimizer.

A normal ``TrainerCallback`` (``on_pre_optimizer_step`` / ``on_optimizer_step``) cannot change
what Adam writes into the weights. ``HookedOptimizer`` wraps the optimizer that TRL/Transformers
creates and runs the :class:`~research_rl.optim.update_interface.UpdateRule` pipeline around
the inner ``step()``:

1. ``post_grad`` hooks on the (already accumulated, already HF-clipped) gradients,
2. optional clipping *inside* the wrapper (after the hooks),
3. snapshot of tracked parameters, inner ``step()`` (preconditioning, weight decay, state update),
4. ``transform_update`` on the fp32 candidate Δ, then ``pre_write`` on the final value,
5. ``post_step`` with the actual displacement.

Design guarantees (each covered by a numerical test):

* With no rules the wrapper is bit-identical to the inner optimizer.
* Parameters are only rewritten when some rule changes them; untouched parameters keep the
  inner optimizer's result bit-for-bit (no lossy ``old + (new - old)`` round trip).
* Masks restore frozen elements with ``torch.where`` so momentum and decoupled weight decay can
  never move them.
* Hooks run once per *optimizer step*, not per micro-batch, so gradient accumulation is honoured.
* Rule state and step counters live in ``state_dict()`` (weights-only-loadable), so HF
  checkpoint resume restores them together with the Adam moments.
* ``request_skip()`` turns the next ``step()`` into a no-op (no parameter or optimizer-state
  change) while still counting it as a nominal step.

Precision (bf16/fp16 parameters)
--------------------------------
A parameter *stored* in bf16 loses every update below ≈ ``|w| * 2**-8`` (torch writes the result
with round-to-nearest), see :mod:`research_rl.optim.precision`. The wrapper therefore checks the
dtypes at construction (``precision_check``) and offers ``master_weights=True`` (fp32 master copy
of every bf16/fp16 optimised parameter; the inner optimizer, its state and every rule's
``transform_update`` / ``pre_write`` run on the fp32 master, the served weight is
``round_to(master, dtype, write_rounding)``) and ``write_rounding="stochastic"`` (unbiased
rounding of the served weight). All defaults (``master_weights=False``, ``"nearest"``) keep the
previous behaviour bit-for-bit. Details in :class:`HookedOptimizer`.
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast

import torch
from torch.optim import Optimizer

from research_rl.optim.precision import (
    ROUNDING_MODES,
    is_low_precision,
    low_precision_trainable,
    round_to,
    rounding_generator,
)
from research_rl.optim.update_interface import StepContext, UpdateRule

_STATE_KEY = "research_rl"
_PRECISION_CHECKS = ("warn", "error", "off")


def _overrides(rule: UpdateRule, method: str) -> bool:
    return getattr(type(rule), method) is not getattr(UpdateRule, method)


class HookedOptimizer(Optimizer):
    """Wraps ``inner`` (any ``torch.optim.Optimizer``) and exposes the same interface.

    Keyword-only precision options (the defaults keep the wrapper bit-identical to before):

    ``master_weights`` (default ``False``)
        Every optimised parameter whose dtype is bf16/fp16 gets an fp32 master copy (created
        from the current value, which is exact). In :meth:`step`:

        1. ``post_grad`` rules see the model parameters and their (low-precision) gradients, and
           in-wrapper clipping (``clip_norm``) acts on those gradients — both exactly as without
           masters;
        2. for master parameters ``param.data`` is swapped to the fp32 master and ``param.grad``
           to ``grad.float()``; the snapshot, ``inner.step()`` and the ``transform_update`` /
           ``pre_write`` rules then run in fp32 (rules see fp32 ``param``/``old``/``new``), so the
           inner optimizer's state (Adam moments) is fp32;
        3. the new master is kept, ``param.data`` is restored to the original low-precision tensor
           and overwritten **in place** (aliases/tied weights stay valid) with
           ``round_to(master, dtype, write_rounding)``; the original gradient tensor is restored.
           Parameters without a gradient and not tracked by any rule are not rewritten. For
           tracked parameters, elements whose master did not change keep their served bits, so a
           :class:`~research_rl.optim.update_interface.MaskRule` keeps frozen elements bit-exact in
           both the master and the served weights even when the mask is installed after
           stochastic steps (stochastic rounding of a representable value is the identity anyway).

        ``post_step`` displacements of master parameters are **master (fp32) displacements**
        θ_master,new − θ_master,old, not differences of the served bf16 weights. Gradient
        accumulation still happens in the parameter dtype (bf16 ``.grad``). Memory: 4 bytes per
        master element, plus a transient fp32 copy of the gradients during ``step()``; the inner
        state is fp32 (8 bytes per element for AdamW instead of 4). ``step(closure)`` is refused
        (the closure would run the model on fp32 data).

        External writes: when a master parameter is modified outside the optimizer (detected
        through its storage pointer / version counter, i.e. ``param.copy_`` under ``no_grad``,
        ``load_state_dict``, ``param.data = ...``; writes through ``param.data.<op>_`` are *not*
        detected), the next step re-derives the master element-wise from the served value for
        every element that is no longer one of the two bf16/fp16 neighbours of its master, warns
        and counts it in ``master_resyncs``. Use :meth:`set_master` to write an exact fp32 value or
        :meth:`resync_masters` to reset masters explicitly.

    ``write_rounding`` (``"nearest"`` | ``"stochastic"``, default ``"nearest"``)
        How fp32 values are rounded when the wrapper writes a low-precision parameter.
        With ``master_weights=True`` it applies to every master → served write (step 3 above),
        i.e. to the inner optimizer's update and every rule's result. **Without** masters the
        inner optimizer writes the parameter itself (torch: round-to-nearest) and
        ``write_rounding`` only applies to the one value the wrapper computes and casts itself:
        ``old + Δ'`` after a ``transform_update`` rule changed Δ. ``pre_write`` results are
        written as returned (e.g. :class:`~research_rl.optim.parameter_prior.P2PriorRule` rounds
        its fp32 result to nearest itself). Stochastic rounding is bf16 only; an optimised fp16
        parameter with ``"stochastic"`` raises ``ValueError`` at construction.
    ``rounding_seed`` (default ``0``)
        Stochastic draws use :func:`~research_rl.optim.precision.rounding_generator` seeded from
        ``(rounding_seed, nominal step, flat parameter index in param_groups)`` on the parameter's
        device, so runs and resumes are reproducible without storing RNG state.
    ``precision_check`` (``"warn"`` | ``"error"`` | ``"off"``, default ``"warn"``)
        At construction, if an optimised parameter with ``requires_grad=True`` is bf16/fp16 and
        ``master_weights`` is ``False``, emit one ``UserWarning`` (``"warn"``) or raise
        ``ValueError`` (``"error"``).

    Persistence: ``state_dict()`` stores the options and the masters (plain fp32 tensors keyed by
    the flat parameter index, the same numbering torch uses for optimizer state) under the
    ``"research_rl"`` key, loadable with ``torch.load(weights_only=True)``. ``load_state_dict``
    installs the masters first and loads the inner state with the parameters swapped to their
    masters, because torch casts loaded state to ``param.dtype`` (the moments would otherwise be
    rounded to bf16) — this also keeps Accelerate's ``AcceleratedOptimizer`` (state_dict → move to
    device → load_state_dict at prepare time) and HF Trainer resume exact. A checkpoint without
    masters loaded into a ``master_weights`` optimizer re-initialises them from the current
    parameters (warning); masters in a checkpoint loaded without ``master_weights`` are dropped
    (warning). Load the optimizer state *after* the model weights (as HF does).
    """

    def __init__(
        self,
        inner: Optimizer,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        rules: Sequence[UpdateRule] = (),
        *,
        clip_norm: float | None = None,
        master_weights: bool = False,
        write_rounding: str = "nearest",
        rounding_seed: int = 0,
        precision_check: str = "warn",
    ) -> None:
        # Deliberately *not* calling Optimizer.__init__: param_groups/state are delegated to
        # ``inner`` (see properties below) so LR schedulers and load_state_dict stay consistent.
        self.inner = inner
        self.defaults = inner.defaults
        self.rules: list[UpdateRule] = list(rules)
        self.clip_norm = clip_norm
        self._optimizer_step_pre_hooks = OrderedDict()
        self._optimizer_step_post_hooks = OrderedDict()
        self._optimizer_state_dict_pre_hooks = OrderedDict()
        self._optimizer_state_dict_post_hooks = OrderedDict()
        self._optimizer_load_state_dict_pre_hooks = OrderedDict()
        self._optimizer_load_state_dict_post_hooks = OrderedDict()
        self._zero_grad_profile_name = "Optimizer.zero_grad#HookedOptimizer.zero_grad"

        optimised = {id(p) for g in inner.param_groups for p in g["params"]}
        self._named: list[tuple[str, torch.nn.Parameter]] = [
            (n, p) for n, p in named_parameters if id(p) in optimised
        ]
        known = {id(p) for _, p in self._named}
        for g in inner.param_groups:
            for p in g["params"]:
                if id(p) not in known:
                    self._named.append((f"<unnamed:{id(p)}>", p))
                    known.add(id(p))

        self.nominal_steps = 0
        self.applied_updates = 0
        self.skipped_steps = 0
        self.last_grad_norm: float | None = None
        self._skip_next = False
        # Copied into StepContext.extras at every step (e.g. the trainer mixin records whether HF
        # clips gradients *before* optimizer.step(), which some rules must refuse).
        self.step_extras: dict[str, Any] = {}

        if write_rounding not in ROUNDING_MODES:
            raise ValueError(f"write_rounding must be one of {ROUNDING_MODES}, got {write_rounding!r}")
        if precision_check not in _PRECISION_CHECKS:
            raise ValueError(f"precision_check must be one of {_PRECISION_CHECKS}, got {precision_check!r}")
        self.master_weights = bool(master_weights)
        self.write_rounding = write_rounding
        self.rounding_seed = int(rounding_seed)
        self.precision_check = precision_check
        self.master_resyncs = 0  # elements re-derived from externally written parameters
        self._masters: dict[int, torch.Tensor] = {}  # id(param) -> fp32 master
        self._fingerprints: dict[int, tuple[int, int, torch.dtype, torch.device]] = {}
        self._generators: dict[torch.device, torch.Generator] = {}
        self._check_precision()
        if self.master_weights:
            self._sync_masters(warn=False)
        for rule in self.rules:
            rule.on_attach(self)

    # ------------------------------------------------------------------ delegation
    @property  # type: ignore[override]
    def param_groups(self) -> list[dict[str, Any]]:
        return self.inner.param_groups

    @param_groups.setter
    def param_groups(self, value: list[dict[str, Any]]) -> None:
        self.inner.param_groups = value

    @property  # type: ignore[override]
    def state(self) -> Any:
        return self.inner.state

    @state.setter
    def state(self, value: Any) -> None:
        self.inner.state = value

    def __getattr__(self, item: str) -> Any:
        # Only reached when normal lookup fails (e.g. schedule-free optimizers' .train()/.eval()).
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(item)
        return getattr(inner, item)

    def __repr__(self) -> str:
        rules = ", ".join(r.name for r in self.rules) or "-"
        precision = (
            f", master_weights=True, write_rounding={self.write_rounding!r}"
            if getattr(self, "master_weights", False)
            else ""
        )
        return (
            f"HookedOptimizer(inner={type(self.inner).__name__}, rules=[{rules}], clip_norm={self.clip_norm}"
            f"{precision})"
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        self.inner.add_param_group(param_group)
        if "inner" in self.__dict__:  # called by Optimizer internals only after construction
            known = {id(p) for _, p in self._named}
            params = param_group["params"]
            for p in [params] if isinstance(params, torch.Tensor) else params:
                if id(p) not in known:
                    self._named.append((f"<unnamed:{id(p)}>", cast(torch.nn.Parameter, p)))

    # ------------------------------------------------------------------ rule management
    def add_rule(self, rule: UpdateRule) -> None:
        self.rules.append(rule)
        rule.on_attach(self)

    def request_skip(self) -> None:
        """Make a ``step()`` a no-op (counted in ``skipped_steps``, not in ``applied_updates``).

        Called before ``step()`` it skips that step; called from a rule's ``post_grad`` hook it skips
        the *current* step (before clipping, the inner optimizer and the write hooks).
        """
        self._skip_next = True

    @property
    def policy_version(self) -> int:
        """Number of parameter updates actually applied (skipped steps excluded)."""
        return self.applied_updates

    def named_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        return list(self._named)

    # ------------------------------------------------------------------ precision
    def _flat_params(self) -> list[torch.nn.Parameter]:
        """Optimised parameters in torch's state_dict numbering (param_groups order)."""
        out: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for g in self.inner.param_groups:
            for p in g["params"]:
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
        return out

    def _check_precision(self) -> None:
        names = {id(p): n for n, p in self._named}
        flat = self._flat_params()
        if self.write_rounding == "stochastic":
            fp16 = [names.get(id(p), "?") for p in flat if p.dtype == torch.float16]
            if fp16:
                raise ValueError(
                    f"write_rounding='stochastic' is only implemented for bfloat16, but {len(fp16)} optimised "
                    f"parameters are float16 (e.g. {fp16[0]!r}); use write_rounding='nearest'"
                )
        if self.master_weights or self.precision_check == "off":
            return
        low = low_precision_trainable([(names.get(id(p), "?"), p) for p in flat])
        if not low:
            return
        msg = (
            f"HookedOptimizer: {len(low)} trainable parameters are stored in low precision (e.g. {low[0]!r}) "
            "and master_weights=False. The inner optimizer writes them with round-to-nearest, so every update "
            "smaller than ~|w|*2**-8 (bf16; ~|w|*2**-11 for fp16) is silently lost. Remedies: load the model in "
            "float32 and train with bf16 autocast (TrainingArguments(bf16=True)), or pass master_weights=True "
            "(OptimSpec.master_weights) to keep fp32 master weights"
            + (
                "; write_rounding='stochastic' without master weights only affects values the wrapper casts "
                "itself (transform_update results), not the inner optimizer's write"
                if self.write_rounding == "stochastic"
                else ""
            )
            + ". Set precision_check='off' to silence this check."
        )
        if self.precision_check == "error":
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)

    def has_master(self, param: torch.Tensor) -> bool:
        """True if ``param`` is (or, at the next step, will be) backed by an fp32 master copy."""
        if not self.master_weights:
            return False
        return id(param) in self._masters or is_low_precision(param.dtype)

    def master_of(self, param: torch.Tensor) -> torch.Tensor | None:
        """The fp32 master of ``param`` (the live tensor, do not modify), or ``None``."""
        return self._masters.get(id(param))

    @staticmethod
    def _fingerprint(p: torch.Tensor) -> tuple[int, int, torch.dtype, torch.device]:
        return (p.data_ptr(), int(p._version), p.dtype, p.device)

    def _upcast_inner_state(self, p: torch.nn.Parameter) -> None:
        st = self.inner.state.get(p)
        if not st:
            return
        for k, v in list(st.items()):
            if k != "step" and isinstance(v, torch.Tensor) and is_low_precision(v.dtype):
                st[k] = v.float()

    @staticmethod
    def _neighbours(master: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """The two ``dtype`` values bracketing ``master`` (toward / away from zero)."""
        if dtype == torch.bfloat16:
            bits = master.contiguous().view(torch.int32) & -65536
            lo = bits.view(torch.float32)
            hi = (bits + 65536).view(torch.float32)
            return lo.to(dtype), hi.to(dtype)
        near = master.to(dtype)
        return near, near

    @torch.no_grad()
    def _sync_masters(self, warn: bool = True) -> None:
        """Create missing masters, drop stale ones, reconcile externally written parameters."""
        live: set[int] = set()
        resynced = 0
        for p in self._flat_params():
            if not is_low_precision(p.dtype):
                continue
            key = id(p)
            live.add(key)
            master = self._masters.get(key)
            if master is None or master.shape != p.shape:
                self._masters[key] = p.detach().to(torch.float32, copy=True)
                self._upcast_inner_state(p)
            elif self._fingerprints.get(key) != self._fingerprint(p):
                if master.device != p.device:
                    master = master.to(p.device)
                served = p.detach()
                lo, hi = self._neighbours(master, p.dtype)
                ok = (served == lo) | (served == hi) | (served.float() == master)
                n_bad = int((~ok).sum())
                if n_bad:
                    master = torch.where(ok, master, served.float())
                    resynced += n_bad
                self._masters[key] = master
            self._fingerprints[key] = self._fingerprint(p)
        for key in [k for k in self._masters if k not in live]:
            del self._masters[key]
            self._fingerprints.pop(key, None)
        if resynced:
            self.master_resyncs += resynced
            if warn:
                warnings.warn(
                    f"HookedOptimizer: {resynced} elements of master-weight parameters were modified outside the "
                    "optimizer; their fp32 masters were re-initialised from the (low-precision) parameters. Use "
                    "HookedOptimizer.set_master() to write exact values.",
                    UserWarning,
                    stacklevel=3,
                )

    @torch.no_grad()
    def set_master(self, param: torch.nn.Parameter, value: torch.Tensor) -> None:
        """Set the fp32 master of ``param`` to ``value`` and write ``round_to(value, nearest)`` into it.

        Without master weights (or for a full-precision parameter) this is ``param.copy_(value)``.
        """
        if not self.has_master(param):
            param.copy_(value)
            return
        master = value.detach().to(device=param.device, dtype=torch.float32).clone()
        self._masters[id(param)] = master
        param.data.copy_(round_to(master, param.dtype, "nearest"))
        self._fingerprints[id(param)] = self._fingerprint(param)

    @torch.no_grad()
    def resync_masters(self) -> None:
        """Reset every master to ``param.float()`` (discards the sub-ulp information they hold)."""
        if not self.master_weights:
            return
        self._masters.clear()
        self._fingerprints.clear()
        self._sync_masters(warn=False)

    def _generator(self, device: torch.device, step: int, index: int) -> torch.Generator:
        gen = self._generators.get(device)
        gen = rounding_generator(self.rounding_seed, step, index, device, generator=gen)
        self._generators[device] = gen
        return gen

    def _round(self, x: torch.Tensor, dtype: torch.dtype, step: int, index: int) -> torch.Tensor:
        if self.write_rounding == "stochastic" and dtype == torch.bfloat16 and x.dtype == torch.float32:
            return round_to(x, dtype, "stochastic", self._generator(x.device, step, index))
        return x.to(dtype)

    # ------------------------------------------------------------------ the step
    def step(self, closure: Callable[[], float] | None = None) -> Any:  # type: ignore[override]
        if self.master_weights and closure is not None:
            raise ValueError("HookedOptimizer(master_weights=True) does not support step(closure)")
        self.nominal_steps += 1
        ctx = StepContext(
            nominal_step=self.nominal_steps,
            update_step=self.applied_updates,
            lrs=[float(g["lr"]) for g in self.inner.param_groups],
            extras=dict(self.step_extras),
        )
        if self._skip_next:
            return self._skip(ctx)

        # Every optimised parameter, not only requires_grad=True ones: a parameter frozen via
        # requires_grad=False that still holds a stale .grad would otherwise be moved by the inner
        # optimizer behind the rules' back.
        named = list(self._named)
        for rule in self.rules:
            if _overrides(rule, "post_grad"):
                rule.post_grad(ctx, named)
        if self._skip_next:  # a post_grad hook may veto THIS step (e.g. non-finite gradients)
            return self._skip(ctx)

        if self.clip_norm is not None and self.clip_norm > 0:
            grads = [p for _, p in named if p.grad is not None]
            total = torch.nn.utils.clip_grad_norm_(grads, self.clip_norm) if grads else torch.tensor(0.0)
            self.last_grad_norm = float(total)
            ctx.grad_norm = self.last_grad_norm
            ctx.extras["nonfinite_grad_norm"] = not math.isfinite(self.last_grad_norm)

        # Per-parameter list of rules that track it, in registration order.
        tracked: list[tuple[str, torch.nn.Parameter, list[UpdateRule]]] = []
        for n, p in named:
            rs = [r for r in self.rules if r.tracks(n, p)]
            if rs:
                tracked.append((n, p, rs))

        swapped: list[tuple[int, torch.nn.Parameter, torch.Tensor, torch.Tensor | None]] = []
        candidates: list[tuple[int, torch.nn.Parameter]] = []
        if self.master_weights:
            self._sync_masters()
            candidates = [(i, p) for i, p in enumerate(self._flat_params()) if id(p) in self._masters]
        try:
            # fp32 masters in, fp32 grads in (the original low-precision tensors are restored below).
            for index, p in candidates:
                low, grad = p.data, p.grad
                p.data = self._masters[id(p)]
                swapped.append((index, p, low, grad))
                if grad is not None:
                    p.grad = grad.float()
            old = {n: p.detach().clone() for n, p, _ in tracked}
            index_cache: dict[int, int] = {}

            def flat_index(param: torch.Tensor) -> int:
                if not index_cache:
                    index_cache.update({id(q): i for i, q in enumerate(self._flat_params())})
                return index_cache.get(id(param), -1)

            loss = self.inner.step(closure) if closure is not None else self.inner.step()

            displacements: dict[str, torch.Tensor] = {}
            want_disp = any(_overrides(r, "post_step") for r in self.rules)
            with torch.no_grad():
                for n, p, rs in tracked:
                    o = old[n]
                    changed = False
                    delta: torch.Tensor | None = None
                    for rule in rs:
                        if _overrides(rule, "transform_update"):
                            if delta is None:
                                delta = p.detach().float() - o.float()
                            out = rule.transform_update(ctx, n, p, o, delta)
                            if out is not None:
                                delta = out
                                changed = True
                    if changed:
                        assert delta is not None
                        candidate = self._round(o.float() + delta, p.dtype, ctx.nominal_step, flat_index(p))
                    else:
                        candidate = p.detach()
                    for rule in rs:
                        if _overrides(rule, "pre_write"):
                            out = rule.pre_write(ctx, n, p, o, candidate)
                            if out is not None:
                                candidate = out
                                changed = True
                    if changed:
                        p.copy_(candidate)
                    if want_disp:
                        displacements[n] = p.detach().float() - o.float()
        finally:
            for _, p, low, grad in swapped:
                current = p.data
                if (
                    current.data_ptr() != self._masters[id(p)].data_ptr()
                ):  # the inner optimizer replaced .data
                    self._masters[id(p)] = current
                p.data = low
                p.grad = grad
        if swapped:
            self._write_served(ctx, swapped, {id(p): old[n] for n, p, _ in tracked})
        for rule in self.rules:
            if _overrides(rule, "post_step"):
                rule.post_step(ctx, displacements)
        self.applied_updates += 1
        return loss

    @torch.no_grad()
    def _write_served(
        self,
        ctx: StepContext,
        swapped: list[tuple[int, torch.nn.Parameter, torch.Tensor, torch.Tensor | None]],
        old_masters: dict[int, torch.Tensor],
    ) -> None:
        """served = round_to(master) in place; unchanged master elements of tracked params keep their bits."""
        for index, p, low, grad in swapped:
            before = old_masters.get(id(p))
            if grad is None and before is None:
                continue  # the inner optimizer skipped it and no rule touched it
            master = self._masters[id(p)]
            rounded = self._round(master, low.dtype, ctx.nominal_step, index)
            if before is not None:
                rounded = torch.where(master == before, low, rounded)
            low.copy_(rounded)
            self._fingerprints[id(p)] = self._fingerprint(p)

    def _skip(self, ctx: StepContext) -> None:
        self._skip_next = False
        self.skipped_steps += 1
        for rule in self.rules:
            rule.on_skip(ctx)
        return None

    # ------------------------------------------------------------------ persistence
    def _rule_key(self, i: int, rule: UpdateRule) -> str:
        return f"{i}:{rule.name}"

    def state_dict(self) -> dict[str, Any]:
        if self.master_weights:
            self._sync_masters()
        sd = self.inner.state_dict()
        flat = self._flat_params()
        masters = {i: self._masters[id(p)] for i, p in enumerate(flat) if id(p) in self._masters}
        sd[_STATE_KEY] = {
            "nominal_steps": self.nominal_steps,
            "applied_updates": self.applied_updates,
            "skipped_steps": self.skipped_steps,
            "rules": {self._rule_key(i, r): r.state_dict() for i, r in enumerate(self.rules)},
            "precision": {
                "master_weights": self.master_weights,
                "write_rounding": self.write_rounding,
                "rounding_seed": self.rounding_seed,
                "masters": masters,
            },
        }
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state_dict = dict(state_dict)
        extra = state_dict.pop(_STATE_KEY, None)
        precision = extra.get("precision") if isinstance(extra, dict) else None
        if self.master_weights:
            self._load_masters(precision)
            self._load_inner_on_masters(state_dict)
        else:
            if precision and precision.get("masters"):
                warnings.warn(
                    "HookedOptimizer: the checkpoint holds fp32 master weights but this optimizer has "
                    "master_weights=False; the masters are dropped and the optimizer state is cast to the "
                    "parameter dtype.",
                    UserWarning,
                    stacklevel=2,
                )
            self.inner.load_state_dict(state_dict)
        if extra is None:
            return
        self.nominal_steps = int(extra.get("nominal_steps", 0))
        self.applied_updates = int(extra.get("applied_updates", 0))
        self.skipped_steps = int(extra.get("skipped_steps", 0))
        saved = extra.get("rules", {})
        for i, rule in enumerate(self.rules):
            key = self._rule_key(i, rule)
            if key in saved:
                rule.load_state_dict(saved[key])

    @torch.no_grad()
    def _load_masters(self, precision: Any) -> None:
        # Fallback masters from the current params (the checkpoint's masters replace them below).
        previous = dict(self._masters)
        self._masters.clear()
        self._fingerprints.clear()
        self._sync_masters(warn=False)
        if precision:
            for opt in ("write_rounding", "rounding_seed"):
                if opt in precision and precision[opt] != getattr(self, opt):
                    warnings.warn(
                        f"HookedOptimizer: checkpoint {opt}={precision[opt]!r} differs from this optimizer's "
                        f"{getattr(self, opt)!r}; the current setting is used.",
                        UserWarning,
                        stacklevel=3,
                    )
        saved = (precision or {}).get("masters") or {}
        missing = 0
        for index, p in enumerate(self._flat_params()):
            key = id(p)
            if key not in self._masters:
                continue
            value = saved.get(index)
            if value is None:
                missing += 1
                continue
            if tuple(value.shape) != tuple(p.shape):
                raise ValueError(
                    f"checkpoint master weight {index} has shape {tuple(value.shape)}, parameter has {tuple(p.shape)}"
                )
            if value.device == p.device and value is previous.get(key):
                # Accelerate's prepare round trip hands back the same objects: keep them.
                self._masters[key] = previous[key]
            else:
                self._masters[key] = value.detach().to(device=p.device, dtype=torch.float32, copy=True)
            self._fingerprints[key] = self._fingerprint(p)
        if missing:
            warnings.warn(
                f"HookedOptimizer: the checkpoint has no fp32 master weights for {missing} low-precision "
                "parameters; they were re-initialised from the current parameter values (sub-ulp progress "
                "of the run that wrote the checkpoint, if any, is lost).",
                UserWarning,
                stacklevel=3,
            )

    def _load_inner_on_masters(self, state_dict: dict[str, Any]) -> None:
        """inner.load_state_dict with params swapped to fp32 (torch casts loaded state to param.dtype)."""
        swapped: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        try:
            for p in self._flat_params():
                master = self._masters.get(id(p))
                if master is not None:
                    low = p.data
                    p.data = master
                    swapped.append((p, low))
            self.inner.load_state_dict(state_dict)
        finally:
            for p, low in swapped:
                p.data = low
                self._fingerprints[id(p)] = self._fingerprint(p)


def unwrap_hooked(optimizer: Any) -> HookedOptimizer | None:
    """Find a HookedOptimizer inside accelerate/deepspeed wrappers (or return None)."""
    seen = 0
    while optimizer is not None and seen < 8:
        if isinstance(optimizer, HookedOptimizer):
            return optimizer
        optimizer = getattr(optimizer, "optimizer", None)
        seen += 1
    return None
