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
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast

import torch
from torch.optim import Optimizer

from research_rl.optim.update_interface import StepContext, UpdateRule

_STATE_KEY = "research_rl"


def _overrides(rule: UpdateRule, method: str) -> bool:
    return getattr(type(rule), method) is not getattr(UpdateRule, method)


class HookedOptimizer(Optimizer):
    """Wraps ``inner`` (any ``torch.optim.Optimizer``) and exposes the same interface."""

    def __init__(
        self,
        inner: Optimizer,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        rules: Sequence[UpdateRule] = (),
        *,
        clip_norm: float | None = None,
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
        return (
            f"HookedOptimizer(inner={type(self.inner).__name__}, rules=[{rules}], clip_norm={self.clip_norm})"
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

    # ------------------------------------------------------------------ the step
    def step(self, closure: Callable[[], float] | None = None) -> Any:  # type: ignore[override]
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
        old = {n: p.detach().clone() for n, p, _ in tracked}

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
                    candidate = (o.float() + delta).to(p.dtype)
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
        for rule in self.rules:
            if _overrides(rule, "post_step"):
                rule.post_step(ctx, displacements)
        self.applied_updates += 1
        return loss

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
        sd = self.inner.state_dict()
        sd[_STATE_KEY] = {
            "nominal_steps": self.nominal_steps,
            "applied_updates": self.applied_updates,
            "skipped_steps": self.skipped_steps,
            "rules": {self._rule_key(i, r): r.state_dict() for i, r in enumerate(self.rules)},
        }
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state_dict = dict(state_dict)
        extra = state_dict.pop(_STATE_KEY, None)
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


def unwrap_hooked(optimizer: Any) -> HookedOptimizer | None:
    """Find a HookedOptimizer inside accelerate/deepspeed wrappers (or return None)."""
    seen = 0
    while optimizer is not None and seen < 8:
        if isinstance(optimizer, HookedOptimizer):
            return optimizer
        optimizer = getattr(optimizer, "optimizer", None)
        seen += 1
    return None
