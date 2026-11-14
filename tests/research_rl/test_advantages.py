"""GRPO / RLOO advantages from TRL against independent reference formulas.

Rewards are controlled: every injected completion is distinct and a lookup table maps it to its
reward. The reference implementations use only Python's ``statistics`` module (and, for the KL
term, a plain per-sample forward pass), never TRL helpers.
"""

from __future__ import annotations

import statistics

import pytest
import torch
from datasets import Dataset
from trl import GRPOConfig, RLOOConfig

from research_rl.models.tiny import build_tiny_llama
from research_rl.trainers.grpo_adapter import ResearchGRPOTrainer
from research_rl.trainers.rl_common import SIGNAL_KEY
from research_rl.trainers.rloo_adapter import ResearchRLOOTrainer

EOS = 2


@pytest.fixture(autouse=True)
def _single_thread():
    """Tiny models are dominated by thread-sync overhead on a loaded CPU: 1 thread is ~20x faster."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


G = 4  # num_generations
PROMPTS = ["1+1=", "2+3=", "4+4="]
# Group 0 and 1 have spread, group 2 has zero variance. Dyadic values keep group means exact.
REWARDS = [1.0, 0.0, 0.5, 0.25, 3.0, -1.0, 2.0, 7.0, 0.5, 0.5, 0.5, 0.5]
COMPLETIONS = [[10 + i, 30 + (i % 3), EOS] for i in range(len(REWARDS))]  # distinct, equal width
TABLE = {tuple(c): r for c, r in zip(COMPLETIONS, REWARDS)}


def table_reward(completion_ids, **kwargs):
    return [TABLE[tuple(c)] for c in completion_ids]


def inputs() -> list[dict[str, str]]:
    return [{"prompt": PROMPTS[i // G]} for i in range(len(REWARDS))]


def grpo_reference(rewards: list[float], scale: str, eps: float = 1e-4) -> list[float]:
    batch_std = statistics.stdev(rewards)  # sample (n-1) std
    out = []
    for g in range(0, len(rewards), G):
        group = rewards[g : g + G]
        mean, std = statistics.fmean(group), statistics.stdev(group)
        for r in group:
            a = r - mean
            if a != 0.0 and scale == "group":  # (a == 0 only in the zero-variance group: 0 / eps = 0)
                a /= std + eps
            elif a != 0.0 and scale == "batch":
                a /= batch_std + eps
            out.append(a)
    return out


def rloo_reference(rewards: list[float], normalize: bool, eps: float = 1e-4) -> list[float]:
    out = []
    for g in range(0, len(rewards), G):
        group = rewards[g : g + G]
        for i, r in enumerate(group):
            others = group[:i] + group[i + 1 :]
            out.append(r - statistics.fmean(others))
    if normalize:
        mean, std = statistics.fmean(out), statistics.stdev(out)
        out = [(a - mean) / (std + eps) for a in out]
    return out


def make_rollout(tokenizer):
    def rollout(prompts, trainer):
        return {
            "prompt_ids": [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts],
            "completion_ids": [list(c) for c in COMPLETIONS],
            "logprobs": [[-1.0] * len(c) for c in COMPLETIONS],
        }

    return rollout


def patch_generate(model):
    def fake_generate(input_ids=None, attention_mask=None, **kwargs):
        completion = torch.tensor(COMPLETIONS, dtype=input_ids.dtype)
        return torch.cat([input_ids, completion], dim=1)

    model.generate = fake_generate


def dataset():
    return Dataset.from_list([{"prompt": p} for p in PROMPTS])


def as_tensor(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


# ---------------------------------------------------------------------------------- GRPO
def test_grpo_defaults_are_what_the_references_assume(tmp_path):
    cfg = GRPOConfig(output_dir=str(tmp_path), use_cpu=True)
    assert cfg.scale_rewards == "group" and cfg.loss_type == "dapo" and cfg.beta == 0.0
    assert cfg.multi_objective_aggregation == "sum_then_normalize"


@pytest.mark.parametrize("scale", ["group", "batch", "none"])
def test_grpo_advantages_match_reference(tokenizer, tmp_path, scale):
    args = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=len(REWARDS),
        num_generations=G,
        max_completion_length=8,
        scale_rewards=scale,
        report_to=[],
        use_cpu=True,
        bf16=False,
    )
    trainer = ResearchGRPOTrainer(
        model=build_tiny_llama(len(tokenizer), seed=0),
        reward_funcs=table_reward,
        args=args,
        train_dataset=dataset(),
        processing_class=tokenizer,
        rollout_func=make_rollout(tokenizer),
    )
    trainer.model.train()
    out = trainer._generate_and_score_completions(inputs())
    adv = out["advantages"].cpu()
    ref = as_tensor(grpo_reference(REWARDS, scale))
    torch.testing.assert_close(adv, ref, rtol=1e-5, atol=1e-6)
    # zero-variance group -> exactly zero, in every scaling mode
    assert torch.equal(adv[2 * G :], torch.zeros(G))
    # sensitivity of the reference: a biased std or a missing +1e-4 would be detected
    if scale != "none":
        assert not torch.allclose(
            adv, as_tensor(grpo_reference(REWARDS, scale, eps=0.0)), rtol=1e-5, atol=1e-6
        )
        biased = [
            a * (statistics.stdev(REWARDS[:G]) / statistics.pstdev(REWARDS[:G])) for a in ref[:G].tolist()
        ]
        assert not torch.allclose(adv[:G], as_tensor(biased), rtol=1e-4, atol=1e-5)
    # the trajectory log saw the same numbers
    logged = trainer.trajectory_buffer.latest()
    assert logged is not None
    torch.testing.assert_close(logged.rewards, as_tensor(REWARDS))
    torch.testing.assert_close(logged.advantages, adv)
    assert logged.frac_zero_advantage == pytest.approx(G / len(REWARDS))
    assert logged.has_nonzero_advantage


def test_grpo_all_equal_rewards_give_all_zero_advantages(tokenizer, tmp_path):
    args = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=len(REWARDS),
        num_generations=G,
        max_completion_length=8,
        report_to=[],
        use_cpu=True,
        bf16=False,
    )
    trainer = ResearchGRPOTrainer(
        model=build_tiny_llama(len(tokenizer), seed=0),
        reward_funcs=lambda completion_ids, **kw: [0.3] * len(completion_ids),
        args=args,
        train_dataset=dataset(),
        processing_class=tokenizer,
        rollout_func=make_rollout(tokenizer),
    )
    trainer.model.train()
    out = trainer._generate_and_score_completions(inputs())
    assert torch.equal(out["advantages"].cpu(), torch.zeros(len(REWARDS)))
    logged = trainer.trajectory_buffer.latest()
    assert logged is not None and not logged.has_nonzero_advantage and logged.frac_zero_advantage == 1.0


# ---------------------------------------------------------------------------------- RLOO
def rloo(tokenizer, tmp_path, model=None, **cfg):
    args = RLOOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=len(REWARDS),
        num_generations=G,
        max_completion_length=8,
        report_to=[],
        use_cpu=True,
        bf16=False,
        **cfg,
    )
    trainer = ResearchRLOOTrainer(
        model=model if model is not None else build_tiny_llama(len(tokenizer), seed=0),
        reward_funcs=table_reward,
        args=args,
        train_dataset=dataset(),
        processing_class=tokenizer,
    )
    trainer.model.train()
    patch_generate(trainer.model)
    return trainer


def test_rloo_defaults_are_what_the_references_assume(tmp_path):
    cfg = RLOOConfig(output_dir=str(tmp_path), use_cpu=True)
    assert cfg.normalize_advantages is False and cfg.beta == 0.05 and cfg.reward_clip_range is None


@pytest.mark.parametrize("normalize", [False, True])
def test_rloo_advantages_match_leave_one_out_reference(tokenizer, tmp_path, normalize):
    trainer = rloo(tokenizer, tmp_path, beta=0.0, normalize_advantages=normalize)
    out = trainer._generate_and_score_completions(inputs())
    adv = out["advantages"].cpu()
    torch.testing.assert_close(adv, as_tensor(rloo_reference(REWARDS, normalize)), rtol=1e-5, atol=1e-6)
    zero_group = adv[2 * G :]
    if normalize:  # group sums of LOO advantages are 0, so the batch mean is ~0 and stays ~0
        assert zero_group.abs().max() < 1e-5
    else:
        assert torch.equal(zero_group, torch.zeros(G))
    # the plain group-mean baseline (GRPO's) is a different estimator: make sure the test can tell
    grpo_like = as_tensor(grpo_reference(REWARDS, "none"))
    assert not torch.allclose(adv, grpo_like, atol=1e-3)


def completion_logp_sum(model, prompt_ids: list[int], completion: list[int]) -> float:
    """Σ_t log p(c_t | prompt, c_<t) from a single unpadded forward pass."""
    ids = torch.tensor([prompt_ids + completion])
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, len(prompt_ids) - 1 : -1].float()
    logp = torch.log_softmax(logits, dim=-1).gather(-1, torch.tensor(completion)[:, None])
    return float(logp.sum())


def test_rloo_default_beta_kl_penalty_matches_reference(tokenizer, tiny_model_dir, tmp_path):
    """Default ``beta=0.05``: reward' = r - β Σ_t (logp_policy - logp_ref), then leave-one-out."""
    trainer = rloo(tokenizer, tmp_path, model=tiny_model_dir)
    assert trainer.beta == 0.05 and trainer.ref_model is not None
    torch.manual_seed(0)
    with torch.no_grad():  # move the policy away from the (disk-loaded) reference
        for p in trainer.model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    patch_generate(trainer.model)
    out = trainer._generate_and_score_completions(inputs())
    shaped = []
    for i, (completion, r) in enumerate(zip(COMPLETIONS, REWARDS)):
        prompt_ids = tokenizer(PROMPTS[i // G], add_special_tokens=False)["input_ids"]
        kl = completion_logp_sum(trainer.model, prompt_ids, completion) - completion_logp_sum(
            trainer.ref_model, prompt_ids, completion
        )
        shaped.append(r - 0.05 * kl)
    ref = as_tensor(rloo_reference(shaped, normalize=False))
    torch.testing.assert_close(out["advantages"].cpu(), ref, rtol=1e-4, atol=1e-5)
    # the KL term is not negligible in this test, so dropping it would be caught
    assert not torch.allclose(out["advantages"].cpu(), as_tensor(rloo_reference(REWARDS, False)), atol=1e-3)


# ---------------------------------------------------------------------------------- learning signal
# TRL computes advantages in float32. A constant group whose value is not exactly representable
# (0.1, 0.9) leaves round-off advantages; the learning-signal flags must still say "no signal".
def const_table_reward(values: list[float | None]):
    table = {tuple(c): v for c, v in zip(COMPLETIONS, values)}

    def reward(completion_ids, **kwargs):
        return [table[tuple(c)] for c in completion_ids]

    return reward


def grpo_with(tokenizer, tmp_path, values, num_generations=G, reward_funcs=None, **cfg):
    args = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=len(REWARDS),
        num_generations=num_generations,
        max_completion_length=8,
        report_to=[],
        use_cpu=True,
        bf16=False,
        **cfg,
    )
    trainer = ResearchGRPOTrainer(
        model=build_tiny_llama(len(tokenizer), seed=0),
        reward_funcs=reward_funcs if reward_funcs is not None else const_table_reward(values),
        args=args,
        train_dataset=dataset(),
        processing_class=tokenizer,
        rollout_func=make_rollout(tokenizer),
    )
    trainer.model.train()
    return trainer


def group_inputs(num_generations: int) -> list[dict[str, str]]:
    return [{"prompt": PROMPTS[(i // num_generations) % len(PROMPTS)]} for i in range(len(REWARDS))]


@pytest.mark.parametrize(("num_generations", "value"), [(12, 0.1), (3, 0.9), (4, 0.1)])
def test_grpo_round_off_constant_groups_have_no_signal(tokenizer, tmp_path, num_generations, value):
    trainer = grpo_with(tokenizer, tmp_path, [value] * len(REWARDS), num_generations=num_generations)
    out = trainer._generate_and_score_completions(group_inputs(num_generations))
    adv = out["advantages"].cpu()
    if num_generations != 4:  # TRL really produces round-off here (so the test is not vacuous)
        assert adv.abs().max() > 1e-6
    assert adv.abs().max() < 1e-2  # ... but only round-off
    assert not out[SIGNAL_KEY].any()
    logged = trainer.trajectory_buffer.latest()
    assert logged is not None and not logged.has_nonzero_advantage and logged.frac_zero_advantage == 1.0
    assert trainer.trajectory_buffer.counters()["zero_signal_batches"] == 1


def test_grpo_signal_flags_one_ulp_apart_rewards(tokenizer, tmp_path):
    """Over-skip guard: rewards one float32 ulp apart are a (tiny) real signal, not round-off."""
    v = 0.9
    w = float(torch.nextafter(torch.tensor(v), torch.tensor(2.0)))
    values = [v] * 4 + [v, w, v, v] + [0.1] * 4
    trainer = grpo_with(tokenizer, tmp_path, values)
    out = trainer._generate_and_score_completions(inputs())
    signal, adv = out[SIGNAL_KEY].cpu(), out["advantages"].cpu()
    # float32 r - mean rounds to exactly 0 for the three v's here (TRL trains on w only); the flag
    # follows TRL wherever the group really varies, and is off for the constant groups.
    assert adv[5] != 0 and signal[5]
    assert torch.equal(signal[4:8], adv[4:8] != 0)
    assert not signal[:4].any() and not signal[8:].any()
    assert trainer.trajectory_buffer.latest().has_nonzero_advantage


def test_grpo_normalize_then_sum_signal(tokenizer, tmp_path):
    """normalize_then_sum: an all-constant batch gives O(round-off / 1e-4) advantages in TRL; no signal."""
    trainer = grpo_with(
        tokenizer, tmp_path, [0.7] * 12, num_generations=12, multi_objective_aggregation="normalize_then_sum"
    )
    out = trainer._generate_and_score_completions(group_inputs(12))
    assert out["advantages"].abs().max() > 1e-7  # TRL's round-off is real here
    assert not out[SIGNAL_KEY].any()
    # mixed: a varied group carries signal, the constant non-dyadic groups do not
    values = [0.1] * 4 + [1.0, 0.0, 0.5, 0.25] + [0.9] * 4
    trainer = grpo_with(tokenizer, tmp_path, values, multi_objective_aggregation="normalize_then_sum")
    out = trainer._generate_and_score_completions(inputs())
    assert out[SIGNAL_KEY].cpu().tolist() == [False] * 4 + [True] * 4 + [False] * 4


def test_grpo_unscorable_rows_are_nan_excluded_and_without_signal(tokenizer, tmp_path):
    values: list[float | None] = [1.0, None, 0.5, 0.25, None, None, None, None, 2.0, 2.0, None, 0.0]
    trainer = grpo_with(tokenizer, tmp_path, values)
    out = trainer._generate_and_score_completions(inputs())
    logged = trainer.trajectory_buffer.latest()
    assert logged is not None
    nan = torch.isnan(logged.rewards)
    assert nan.tolist() == [v is None for v in values]
    scorable = [v for v in values if v is not None]
    assert logged.mean_reward == pytest.approx(sum(scorable) / len(scorable))
    adv = out["advantages"].cpu()
    assert torch.equal(adv[nan], torch.zeros(int(nan.sum())))
    # reference: group mean / unbiased std over scorable members only
    for g in range(3):
        group = [(i, v) for i, v in enumerate(values[g * G : (g + 1) * G], start=g * G) if v is not None]
        rs = [v for _, v in group]
        for i, r in group:
            expected = (r - statistics.fmean(rs)) / (statistics.stdev(rs) + 1e-4) if len(rs) > 1 else 0.0
            assert float(adv[i]) == pytest.approx(expected, rel=1e-5, abs=1e-6)
    expected_signal = [False] * 12
    for i in (0, 2, 3, 8, 9, 11):
        expected_signal[i] = True
    assert out[SIGNAL_KEY].cpu().tolist() == expected_signal


@pytest.mark.parametrize("normalize", [False, True])
def test_rloo_round_off_constant_groups_have_no_signal(tokenizer, tmp_path, normalize):
    trainer = rloo(tokenizer, tmp_path, beta=0.0, normalize_advantages=normalize)
    trainer.reward_funcs = [const_table_reward([0.9] * len(REWARDS))]
    out = trainer._generate_and_score_completions(inputs())
    if not normalize:  # (normalising identical round-off values gives exact zeros)
        assert out["advantages"].abs().max() > 0  # TRL leaves round-off (not vacuous)
    assert not out[SIGNAL_KEY].any()
    # mixed batch: the varied group has signal, the constant 0.9 groups do not
    trainer.reward_funcs = [const_table_reward([0.9] * 4 + [3.0, -1.0, 2.0, 7.0] + [0.9] * 4)]
    out = trainer._generate_and_score_completions(inputs())
    assert out[SIGNAL_KEY].cpu().tolist() == [False] * 4 + [True] * 4 + [False] * 4
    assert (out["advantages"][:4] != 0).all()  # round-off (amplified when normalised), not signal


def test_rloo_signal_uses_clipped_rewards(tokenizer, tmp_path):
    """reward_clip_range can make a varied group constant (here: all clipped to 0.9)."""
    trainer = rloo(tokenizer, tmp_path, beta=0.0, reward_clip_range=(-10.0, 0.9))
    trainer.reward_funcs = [const_table_reward([1.0, 3.0, 2.0, 7.0] * 3)]
    out = trainer._generate_and_score_completions(inputs())
    assert out["advantages"].abs().max() > 0
    assert not out[SIGNAL_KEY].any()


def test_rloo_kl_shaped_signal_when_policy_equals_reference(tokenizer, tiny_model_dir, tmp_path):
    """Default beta: policy == reference -> KL is exactly 0, constant 0.9 rewards -> no signal;
    once the policy moves, the KL differs per completion -> signal everywhere."""
    trainer = rloo(tokenizer, tmp_path, model=tiny_model_dir)
    trainer.reward_funcs = [const_table_reward([0.9] * len(REWARDS))]
    patch_generate(trainer.model)
    out = trainer._generate_and_score_completions(inputs())
    assert out["advantages"].abs().max() > 0  # TRL's round-off
    assert not out[SIGNAL_KEY].any()
    torch.manual_seed(0)
    with torch.no_grad():
        for p in trainer.model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    out = trainer._generate_and_score_completions(inputs())
    assert out[SIGNAL_KEY].all() and (out["advantages"].abs() > 1e-5).all()
