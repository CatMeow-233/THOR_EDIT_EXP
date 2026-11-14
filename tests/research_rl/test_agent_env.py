"""Agent environment contract: TRL tool discovery / schemas, CalculatorEnv, credit-assignment math."""

from __future__ import annotations

import inspect

import pytest
import torch
from transformers.utils import get_json_schema

from research_rl.core.registry import ENVIRONMENTS
from research_rl.rollout.agent_env import (
    CalculatorEnv,
    EpisodeLog,
    ResearchEnvironment,
    discounted_returns,
    generalized_advantages,
    spread_to_tokens,
)


def trl_tools(env: object) -> list:
    """Exactly how GRPOTrainer (TRL 1.18) collects an environment's tools."""
    return [
        getattr(env, name)
        for name, _ in inspect.getmembers(type(env), predicate=inspect.isfunction)
        if name not in ("reset", "get_reward") and not name.startswith("_")
    ]


def test_tool_discovery_and_json_schemas():
    assert trl_tools(ResearchEnvironment()) == []  # the base leaks no public functions
    env = CalculatorEnv()
    tools = trl_tools(env)
    assert sorted(t.__name__ for t in tools) == ["add", "multiply"]
    for tool in tools:
        schema = get_json_schema(tool)
        fn = schema["function"]
        assert schema["type"] == "function" and fn["name"] == tool.__name__
        assert fn["description"].startswith(("Add", "Multiply"))
        props = fn["parameters"]["properties"]
        assert set(props) == {"a", "b"} and fn["parameters"]["required"] == ["a", "b"]
        assert all(p["type"] == "number" and p["description"] for p in props.values())
        assert "return" in fn or "returns" in str(schema).lower()
    # TRL's own validation: callable reset that returns None or str
    members = dict(inspect.getmembers(CalculatorEnv, predicate=inspect.isfunction))
    assert "reset" in members and "get_reward" in members


def test_calculator_episode_with_trl_style_calls_and_pooled_reuse():
    env = ENVIRONMENTS.build("calculator", step_penalty=0.1)
    row = {"prompt": [{"role": "user", "content": "Compute (2+3)*4"}], "target": 20}
    obs = env.reset(**row)
    assert isinstance(obs, str) and "20" in obs
    tools = {t.__name__: t for t in trl_tools(env)}
    assert tools["add"](**{"a": 2, "b": 3}) == 5.0
    assert tools["multiply"](**{"a": 5, "b": "4"}) == 20.0  # string args from JSON are coerced
    assert env.get_reward() == pytest.approx(1.0 - 0.2)
    torch.testing.assert_close(env.episode.rewards(), torch.tensor([-0.1, 0.9], dtype=torch.float64))
    # TRL reuses the instance: reset must clear everything
    obs = env.reset(prompt="x", target=7)
    assert len(env.episode) == 0 and env.episode.final_reward is None and env.num_resets == 2
    tools["multiply"](a=2, b=3)
    assert env.get_reward() == pytest.approx(-0.1)  # wrong final answer -> no terminal reward
    assert env.reset(prompt="no target") is None
    assert env.get_reward() == 0.0
    with pytest.raises(ValueError):
        tools["add"](a=float("nan"), b=1)
    budget = CalculatorEnv(max_calls=2)
    budget.reset(target=1)
    budget.add(a=0, b=1)
    budget.add(a=0, b=1)
    with pytest.raises(RuntimeError, match="budget"):
        budget.add(a=0, b=1)


def test_credit_assignment_math_vs_manual():
    r = [0.0, -0.5, 1.0, 0.25, 2.0]
    gamma, lam = 0.9, 0.8
    g = discounted_returns(r, gamma)
    for t in range(len(r)):
        assert float(g[t]) == pytest.approx(sum(gamma ** (j - t) * r[j] for j in range(t, len(r))), rel=1e-12)
    v = [0.3, 0.1, -0.2, 0.5, 1.0]
    adv = generalized_advantages(r, v, gamma, lam, last_value=0.7)
    vv = [*v, 0.7]
    deltas = [r[t] + gamma * vv[t + 1] - vv[t] for t in range(len(r))]
    for t in range(len(r)):
        ref = sum((gamma * lam) ** (j - t) * deltas[j] for j in range(t, len(r)))
        assert float(adv[t]) == pytest.approx(ref, rel=1e-12)
    # λ = 1, V ≡ 0 reduces GAE to discounted returns (plus the bootstrapped tail)
    torch.testing.assert_close(generalized_advantages(r, [0.0] * 5, gamma, 1.0), g)
    tok = spread_to_tokens([1.0, 2.0, 3.0], [2, 0, 3])
    assert tok.tolist() == [1.0, 1.0, 3.0, 3.0, 3.0]

    log = EpisodeLog()
    for x in r[:-1]:
        log.add("act", "obs", x)
    log.add("final", "obs", 0.5)
    log.set_final_reward(1.5)  # terminal reward is added to the last turn: 0.5 + 1.5 = r[-1]
    torch.testing.assert_close(log.discounted_returns(gamma), g)
    torch.testing.assert_close(log.turn_credit(gamma, baseline=g.mean()), g - g.mean())
    torch.testing.assert_close(log.advantages(v, gamma, lam, 0.7), adv)
    assert log.total_reward() == pytest.approx(sum(r))
    lone = EpisodeLog()
    lone.set_final_reward(3.0)
    assert lone.rewards().tolist() == [3.0]
