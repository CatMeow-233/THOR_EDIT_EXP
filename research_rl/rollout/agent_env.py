"""Agent–environment contract compatible with TRL ``GRPOTrainer(environment_factory=...)``.

What TRL 1.18 requires (``trl/trainer/grpo_trainer.py``), and how this module meets it:

* ``environment_factory`` is a zero-argument callable returning an instance (pass the class,
  e.g. ``environment_factory=CalculatorEnv``). Instances are pooled and **reused**: TRL calls
  ``reset(**row)`` with the whole dataset row (``prompt`` and any other columns) before every
  rollout, so ``reset`` must clear all episode state. Returning a string appends it to the last
  user message; returning ``None`` leaves the prompt unchanged.
* Every *public function* of the class (found with ``inspect.getmembers(type(env),
  inspect.isfunction)``, inherited ones included) except ``reset``/``get_reward`` becomes a tool;
  TRL calls it as ``tool(**arguments)`` and renders its JSON schema via the chat template
  (``transformers.utils.get_json_schema``), which needs type hints and a Google-style docstring
  with an ``Args:`` section. :class:`ResearchEnvironment` therefore exposes no other public
  functions — its helpers are private or properties.
* An optional ``get_reward()`` makes the environment a reward source (weight 1). The base class
  defines it as the episode's total reward.

:class:`EpisodeLog` records per-turn rewards for long-horizon credit assignment (discounted
returns, per-turn credit, GAE, spreading turn values to tokens). Note that TRL's GRPO itself only
consumes one scalar reward per rollout; the per-turn quantities are provided for research
trainers/losses built on top and are not wired into TRL.

Not covered: TRL's OpenEnv harness (``trl.experimental.async_grpo.openenv_harness``) imports the
separate ``openenv`` package, which is not installed here; it is neither wrapped nor tested.
No environment here executes generated code.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from research_rl.core.registry import ENVIRONMENTS


# ===================================================================================== credit math
def discounted_returns(rewards: Sequence[float] | torch.Tensor, gamma: float) -> torch.Tensor:
    """G_t = r_t + γ G_{t+1} (G_T = 0), float64."""
    r = torch.as_tensor(rewards, dtype=torch.float64)
    out = torch.zeros_like(r)
    running = 0.0
    for t in range(r.shape[0] - 1, -1, -1):
        running = float(r[t]) + gamma * running
        out[t] = running
    return out


def generalized_advantages(
    rewards: Sequence[float] | torch.Tensor,
    values: Sequence[float] | torch.Tensor,
    gamma: float,
    lam: float,
    last_value: float = 0.0,
) -> torch.Tensor:
    """GAE: δ_t = r_t + γ V_{t+1} − V_t (V_T = ``last_value``), A_t = Σ_l (γλ)^l δ_{t+l}."""
    r = torch.as_tensor(rewards, dtype=torch.float64)
    v = torch.as_tensor(values, dtype=torch.float64)
    if v.shape != r.shape:
        raise ValueError("one value estimate per turn is required")
    adv = torch.zeros_like(r)
    running = 0.0
    for t in range(r.shape[0] - 1, -1, -1):
        nxt = float(v[t + 1]) if t + 1 < r.shape[0] else last_value
        delta = float(r[t]) + gamma * nxt - float(v[t])
        running = delta + gamma * lam * running
        adv[t] = running
    return adv


def spread_to_tokens(per_turn: Sequence[float] | torch.Tensor, token_counts: Sequence[int]) -> torch.Tensor:
    """Repeat each turn's value over that turn's tokens (``[Σ token_counts]``)."""
    v = torch.as_tensor(per_turn, dtype=torch.float64)
    counts = torch.as_tensor(list(token_counts), dtype=torch.long)
    if counts.shape != v.shape or (counts < 0).any():
        raise ValueError("one non-negative token count per turn is required")
    return torch.repeat_interleave(v, counts)


@dataclass
class Turn:
    action: str
    observation: str
    reward: float
    info: dict[str, Any] = field(default_factory=dict)


class EpisodeLog:
    """Turns of one episode with per-turn rewards and an optional terminal reward (added to the last turn)."""

    def __init__(self) -> None:
        self.turns: list[Turn] = []
        self.final_reward: float | None = None

    def add(self, action: str, observation: str = "", reward: float = 0.0, **info: Any) -> Turn:
        turn = Turn(action=action, observation=observation, reward=float(reward), info=dict(info))
        self.turns.append(turn)
        return turn

    def set_final_reward(self, reward: float) -> None:
        self.final_reward = float(reward)

    def __len__(self) -> int:
        return len(self.turns)

    def rewards(self) -> torch.Tensor:
        """Per-turn rewards ``[T]`` (terminal reward added to the last turn; a lone terminal reward if T=0)."""
        r = [t.reward for t in self.turns]
        if self.final_reward is not None:
            if r:
                r[-1] += self.final_reward
            else:
                r = [self.final_reward]
        return torch.tensor(r, dtype=torch.float64)

    def total_reward(self) -> float:
        return float(self.rewards().sum())

    def discounted_returns(self, gamma: float = 1.0) -> torch.Tensor:
        return discounted_returns(self.rewards(), gamma)

    def turn_credit(
        self, gamma: float = 1.0, baseline: float | Sequence[float] | torch.Tensor = 0.0
    ) -> torch.Tensor:
        """Per-turn credit G_t − b_t (``baseline`` scalar or per turn)."""
        g = self.discounted_returns(gamma)
        return g - torch.as_tensor(baseline, dtype=torch.float64)

    def advantages(
        self,
        values: Sequence[float] | torch.Tensor,
        gamma: float = 1.0,
        lam: float = 0.95,
        last_value: float = 0.0,
    ) -> torch.Tensor:
        return generalized_advantages(self.rewards(), values, gamma, lam, last_value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": [
                {"action": t.action, "observation": t.observation, "reward": t.reward} for t in self.turns
            ],
            "final_reward": self.final_reward,
        }


# ===================================================================================== environments
class ResearchEnvironment:
    """Base for TRL ``environment_factory`` environments.

    Subclasses add tools as public methods (typed, Google-style docstrings) and override
    :meth:`_on_reset` (called by :meth:`reset` after the episode log is cleared) and, if needed,
    :meth:`_terminal_reward`. Use :meth:`_record` inside tools to log turns. Do not add other
    public methods: TRL would expose them to the model as tools.
    """

    def __init__(self) -> None:
        self._episode = EpisodeLog()
        self._num_resets = 0

    @property
    def episode(self) -> EpisodeLog:
        return self._episode

    @property
    def num_resets(self) -> int:
        return self._num_resets

    def reset(self, **kwargs: Any) -> str | None:
        """Start a new episode. ``kwargs`` is the dataset row TRL passes (``prompt`` and other columns)."""
        self._episode = EpisodeLog()
        self._num_resets += 1
        return self._on_reset(**kwargs)

    def get_reward(self) -> float:
        """Total episode reward (per-turn rewards + terminal reward); called once per rollout by TRL."""
        if self._episode.final_reward is None:
            self._episode.set_final_reward(self._terminal_reward())
        return self._episode.total_reward()

    # ------------------------------------------------------------------ hooks for subclasses (private)
    def _on_reset(self, **kwargs: Any) -> str | None:
        return None

    def _terminal_reward(self) -> float:
        return 0.0

    def _record(self, action: str, observation: str, reward: float = 0.0, **info: Any) -> None:
        self._episode.add(action, observation, reward, **info)


def _number(x: Any) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"non-finite number {x!r}")
    return v


@ENVIRONMENTS.register("calculator")
class CalculatorEnv(ResearchEnvironment):
    """Deterministic toy: reach ``target`` using the ``add`` / ``multiply`` tools.

    ``reset(target=..., **row)`` reads the target from the dataset row. Each tool call costs
    ``step_penalty`` (a per-turn reward); the terminal reward is 1.0 if the last tool result equals
    the target (|diff| ≤ 1e-9), else 0.0. Pure arithmetic on floats — no code is executed.
    """

    def __init__(self, step_penalty: float = 0.01, max_calls: int = 8) -> None:
        super().__init__()
        self._step_penalty = float(step_penalty)
        self._max_calls = int(max_calls)
        self._target: float | None = None
        self._last: float | None = None

    def _on_reset(self, target: float | None = None, **kwargs: Any) -> str | None:
        self._target = None if target is None else _number(target)
        self._last = None
        if self._target is None:
            return None
        return f"\nReach the number {self._target:g} using the add and multiply tools."

    def _apply(self, name: str, a: Any, b: Any, value: float) -> float:
        if len(self._episode) >= self._max_calls:
            raise RuntimeError(f"tool-call budget of {self._max_calls} exhausted")
        self._last = value
        self._record(f"{name}({a}, {b})", f"{value:g}", -self._step_penalty, result=value)
        return value

    def _terminal_reward(self) -> float:
        if self._target is None or self._last is None:
            return 0.0
        return 1.0 if abs(self._last - self._target) <= 1e-9 else 0.0

    def add(self, a: float, b: float) -> float:
        """Add two numbers.

        Args:
            a: The first number.
            b: The second number.

        Returns:
            The sum a + b.
        """
        return self._apply("add", a, b, _number(a) + _number(b))

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers.

        Args:
            a: The first number.
            b: The second number.

        Returns:
            The product a * b.
        """
        return self._apply("multiply", a, b, _number(a) * _number(b))
