"""GRPO / RLOO through ResearchRL: loss gradients vs. references, versions, skipping, learning, resume.

* Gradient tests (fast): on one generation batch, the gradient that TRL's training steps
  accumulate equals the gradient of an independently written reference loss. The on-policy loss
  *value* is ~0, so the gradient is the real check.
* Integration tests (``@pytest.mark.integration``): full ``build_run`` runs on the tiny model.
  The skip/version scenarios use injected completions and an identity shuffle. Which
  micro-batches fall into which optimizer window is then known, and policy version (applied
  updates) differs from ``global_step``.
"""

from __future__ import annotations

import json

import pytest
import torch
import trl.trainer.grpo_trainer as grpo_module
from datasets import Dataset
from transformers import TrainerCallback
from trl import GRPOConfig, RLOOConfig

from research_rl.core.checkpoint import load_research_state
from research_rl.core.config import DataSpec, ExperimentConfig, ModelSpec, OptimSpec, TrainerSpec
from research_rl.core.experiment import ExperimentRecorder
from research_rl.models.model_interface import snapshot
from research_rl.models.tiny import build_tiny_llama
from research_rl.rollout import POLICY_VERSION_KEY
from research_rl.trainers import build_run, get_adapter
from research_rl.trainers.grpo_adapter import GRPOAdapter, ResearchGRPOTrainer
from research_rl.trainers.rl_common import (
    ROLLOUT_COMPONENT,
    SIGNAL_KEY,
    _WindowCallback,
    resume_is_generation_aligned,
)
from research_rl.trainers.rloo_adapter import ResearchRLOOTrainer, RLOOAdapter

EOS = 2


@pytest.fixture(autouse=True)
def _single_thread():
    """Tiny models are dominated by thread-sync overhead on a loaded CPU: 1 thread is ~20x faster."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


# ================================================================================== references
def logp_tokens(model, prompt_ids: list[int], completion: list[int]) -> torch.Tensor:
    """Per-token log p(c_t | prompt, c_<t) from one unpadded forward pass (with grad)."""
    ids = torch.tensor([prompt_ids + completion])
    logits = model(input_ids=ids).logits[0, len(prompt_ids) - 1 : -1].float()
    return torch.log_softmax(logits, dim=-1).gather(-1, torch.tensor(completion)[:, None])[:, 0]


def grads(model) -> dict[str, torch.Tensor]:
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def assert_grads_close(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert a.keys() == b.keys()
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=1e-4, atol=1e-6, msg=name)


GRAD_PROMPTS = ["1+1=", "22+2="]  # different lengths -> TRL left-pads prompts
GRAD_COMPLETIONS = {  # variable lengths, some without EOS
    "1+1=": [[15, 16, EOS], [17, EOS], [18, 19, 20, 21, 22, EOS], [23, 24]],
    "22+2=": [[25, EOS], [26, 27, 28], [29, 30, 31, EOS], [EOS]],
}


def grad_reward(completion_ids, **kwargs):
    return [0.5 * len(c) + (c[0] % 3) for c in completion_ids]


def grad_rollout(tokenizer, record: dict):
    def rollout(prompts, trainer):
        seen: dict[str, int] = {}
        prompt_ids, completions = [], []
        for p in prompts:
            k = seen.get(p, 0)
            seen[p] = k + 1
            prompt_ids.append(tokenizer(p, add_special_tokens=False)["input_ids"])
            completions.append(list(GRAD_COMPLETIONS[p][k]))
        record["prompt_ids"], record["completions"] = prompt_ids, completions
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completions,
            "logprobs": [[0.0] * len(c) for c in completions],
        }

    return rollout


def run_one_window(trainer, generation_batch) -> dict[str, torch.Tensor]:
    """Accumulate gradients over one optimizer window exactly as the HF loop does."""
    model = trainer.model
    model.train()
    model.zero_grad()
    trainer.current_gradient_accumulation_steps = trainer.args.gradient_accumulation_steps
    for _ in range(trainer.args.gradient_accumulation_steps):
        trainer.training_step(model, generation_batch, None)
    return grads(model)


@pytest.mark.parametrize(("loss_type", "mask_truncated"), [(None, False), (None, True), ("grpo", False)])
def test_grpo_gradient_matches_reference_loss(tokenizer, tmp_path, loss_type, mask_truncated):
    record: dict = {}
    extra = {} if loss_type is None else {"loss_type": loss_type}
    args = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,  # generation batch of 8 split into 2 micro-batches
        num_generations=4,
        max_completion_length=8,
        mask_truncated_completions=mask_truncated,
        report_to=[],
        use_cpu=True,
        bf16=False,
        **extra,
    )
    model = build_tiny_llama(len(tokenizer), seed=0)
    trainer = ResearchGRPOTrainer(
        model=model,
        reward_funcs=grad_reward,
        args=args,
        train_dataset=Dataset.from_list([{"prompt": p} for p in GRAD_PROMPTS]),
        processing_class=tokenizer,
        rollout_func=grad_rollout(tokenizer, record),
    )
    effective = trainer.loss_type
    assert effective == (loss_type or "dapo")
    generation_batch = [{"prompt": p} for p in GRAD_PROMPTS for _ in range(4)]
    g_trl = run_one_window(trainer, generation_batch)

    # ---- independent reference: advantages from the rewards, loss from per-sample forwards
    completions = record["completions"]
    r = torch.tensor(grad_reward(completions))
    groups = r.view(-1, 4)
    adv = ((groups - groups.mean(1, keepdim=True)) / (groups.std(1, keepdim=True) + 1e-4)).view(-1)
    model.zero_grad()
    per_seq_sums, per_seq_lens = [], []
    for i, (p_ids, comp) in enumerate(zip(record["prompt_ids"], completions)):
        if mask_truncated and comp[-1] != EOS:
            per_seq_sums.append(torch.zeros(()))
            per_seq_lens.append(0)
            continue
        logp = logp_tokens(model, p_ids, comp)
        ratio = torch.exp(logp - logp.detach())  # on-policy importance ratio (== 1, gradient ∇logp)
        per_seq_sums.append((-adv[i] * ratio).sum())
        per_seq_lens.append(len(comp))
    total_tokens = sum(per_seq_lens)
    if effective == "dapo":  # token sum over the whole window / loss tokens of the generation batch
        loss = torch.stack(per_seq_sums).sum() / total_tokens
    else:  # "grpo": mean over sequences of the per-sequence token mean
        loss = torch.stack([s / max(n, 1) for s, n in zip(per_seq_sums, per_seq_lens)]).mean()
    loss.backward()
    g_ref = grads(model)
    assert max(float(g.abs().max()) for g in g_ref.values()) > 1e-3  # a non-trivial gradient
    assert_grads_close(g_trl, g_ref)


def test_rloo_gradient_matches_reference_loss(tokenizer, tmp_path):
    completions = [c for p in GRAD_PROMPTS for c in GRAD_COMPLETIONS[p]]
    width = max(len(c) for c in completions)
    # fixed-width rows for the generate path; junk after EOS is cut by TRL
    rows = [c + [40] * (width - len(c)) for c in completions]
    truncated_at_eos = [c[: c.index(EOS) + 1] if EOS in c else r for c, r in zip(completions, rows)]
    args = RLOOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_generations=4,
        max_completion_length=width,
        beta=0.0,
        report_to=[],
        use_cpu=True,
        bf16=False,
    )
    model = build_tiny_llama(len(tokenizer), seed=0)
    trainer = ResearchRLOOTrainer(
        model=model,
        reward_funcs=grad_reward,
        args=args,
        train_dataset=Dataset.from_list([{"prompt": p} for p in GRAD_PROMPTS]),
        processing_class=tokenizer,
    )

    def fake_generate(input_ids=None, attention_mask=None, **kwargs):
        return torch.cat([input_ids, torch.tensor(rows, dtype=input_ids.dtype)], dim=1)

    trainer.model.generate = fake_generate
    generation_batch = [{"prompt": p} for p in GRAD_PROMPTS for _ in range(4)]
    g_trl = run_one_window(trainer, generation_batch)

    r = grad_reward(truncated_at_eos)
    adv = []
    for g in range(0, len(r), 4):
        group = r[g : g + 4]
        adv += [x - (sum(group) - x) / 3 for x in group]
    model.zero_grad()
    terms = []
    for i, comp in enumerate(truncated_at_eos):
        p_ids = tokenizer(GRAD_PROMPTS[i // 4], add_special_tokens=False)["input_ids"]
        seq_logp = logp_tokens(model, p_ids, comp).sum()
        terms.append(-adv[i] * torch.exp(seq_logp - seq_logp.detach()))
    torch.stack(terms).mean().backward()  # per-sequence loss, mean over the window
    g_ref = grads(model)
    assert max(float(g.abs().max()) for g in g_ref.values()) > 1e-3
    assert_grads_close(g_trl, g_ref)


# ================================================================================== adapters
def test_adapters_are_registered_and_labelled_rl():
    assert isinstance(get_adapter("grpo"), GRPOAdapter) and isinstance(get_adapter("rloo"), RLOOAdapter)
    assert GRPOAdapter.is_reinforcement_learning and RLOOAdapter.is_reinforcement_learning
    assert GRPOAdapter().trainer_class() is ResearchGRPOTrainer
    assert RLOOAdapter().trainer_class() is ResearchRLOOTrainer


def make_config(tmp_path, algorithm, args, *, skip=False, run_name=None, hooked=True) -> ExperimentConfig:
    return ExperimentConfig(
        study_id="rlvr_test",
        model=ModelSpec(name_or_path="tiny-llama-in-memory"),
        data=DataSpec(name="synthetic"),
        trainer=TrainerSpec(algorithm=algorithm, args=args),
        seed=0,
        output_dir=str(tmp_path),
        run_name=run_name,
        optim=OptimSpec(hooked=hooked, skip_zero_advantage_steps=skip),
    )


def test_skip_requires_hooked_optimizer(tokenizer, tmp_path):
    exp = make_config(
        tmp_path,
        "grpo",
        {"per_device_train_batch_size": 4, "num_generations": 4, "max_completion_length": 4, "max_steps": 1},
        skip=True,
        hooked=False,
    )
    with pytest.raises(ValueError, match="HookedOptimizer"):
        build_run(
            exp,
            train_dataset=Dataset.from_list([{"prompt": "1+1="}]),
            model=build_tiny_llama(len(tokenizer)),
            processing_class=tokenizer,
            reward_funcs=grad_reward,
            record=False,
        )


# ================================================================================== integration
def digit_fraction(completions, **kwargs):
    """Dense learnable reward: fraction of completion characters that are digits."""
    return [sum(ch.isdigit() for ch in c) / max(len(c), 1) for c in completions]


def constant_reward(completions, **kwargs):
    return [1.0] * len(completions)


def arithmetic_dataset(n: int = 16) -> Dataset:
    return Dataset.from_list(
        [{"prompt": f"{a}+{b}=", "answer": str(a + b)} for a in range(4) for b in range(n // 4)]
    )


def rl_args(algorithm: str, **overrides) -> dict:
    args = {
        "per_device_train_batch_size": 8,
        "num_generations": 4,
        "max_completion_length": 8,
        "max_steps": 3,
        "learning_rate": 1e-3,
        "logging_steps": 1,
    }
    if algorithm == "rloo":
        args["beta"] = 0.0  # the in-memory tiny model cannot be re-loaded as a reference model
    args.update(overrides)
    return args


class ParamTrace(TrainerCallback):
    """Snapshot of all parameters after every optimizer step."""

    def __init__(self):
        self.after_step: list[dict[str, torch.Tensor]] = []
        self.model = None

    def on_step_end(self, args, state, control, model=None, **kwargs):
        self.after_step.append(snapshot(model))


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["grpo", "rloo"])
def test_run_records_metrics_and_stamps_policy_versions(tokenizer, tmp_path, algorithm):
    exp = make_config(tmp_path, algorithm, rl_args(algorithm))
    run = build_run(
        exp,
        train_dataset=arithmetic_dataset(),
        model=build_tiny_llama(len(tokenizer), seed=0),
        processing_class=tokenizer,
        reward_funcs=digit_fraction,
    )
    record = run.train()
    trainer = run.trainer
    opt = run.hooked_optimizer
    assert opt is not None and (opt.nominal_steps, opt.applied_updates, opt.skipped_steps) == (3, 3, 0)
    # one generation per step (grad_accum = steps_per_generation = 1): version k for step k
    assert trainer.trajectory_buffer.policy_versions() == [0, 1, 2]
    assert record["steps"] == {
        "nominal_optimizer_steps": 3,
        "applied_updates": 3,
        "skipped_steps": 0,
        "global_step": 3,
    }
    for key in ("reward", "loss", "grad_norm", "num_tokens"):
        assert key in record["final_metrics"], key
        assert len(run.recorder.history(key)) == 3, key
    assert record["training_tokens"] == run.recorder.history("num_tokens")[-1] > 0
    assert record["rollout"]["total_batches"] == 3 and record["rollout"]["windows"] == 3
    rows = [json.loads(line) for line in (run.recorder.run_dir / "metrics.jsonl").read_text().splitlines()]
    step_rows = [r for r in rows if "rollout/policy_version" in r]
    assert [r["rollout/policy_version"] for r in step_rows] == [0, 1, 2]
    assert [r["rr/applied_updates"] for r in step_rows] == [1, 2, 3]
    # the rewards stored in the trajectory log are the TRL-logged mean rewards
    for batch, logged in zip(trainer.trajectory_buffer, run.recorder.history("reward")):
        assert batch.mean_reward == pytest.approx(logged, abs=1e-6)


# ---- deterministic window scenario -----------------------------------------------------------
CONST_COMPLETION = [11, EOS]
VARIED_COMPLETIONS = [[12, EOS], [13, 14, EOS], [15, EOS], [16, 17, 18, EOS]]


def keyed_reward(completion_ids, **kwargs):
    return [0.0 if list(c) == CONST_COMPLETION else float(c[0] % 4) for c in completion_ids]


def window_rollout(tokenizer):
    def rollout(prompts, trainer):
        seen: dict[str, int] = {}
        completions = []
        for p in prompts:
            k = seen.get(p, 0)
            seen[p] = k + 1
            completions.append(VARIED_COMPLETIONS[k % 4] if p.startswith("v") else CONST_COMPLETION)
        return {
            "prompt_ids": [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts],
            "completion_ids": completions,
            "logprobs": [[0.0] * len(c) for c in completions],
        }

    return rollout


def window_run(
    tokenizer, tmp_path, monkeypatch, prompts, *, ga, spg, max_steps, run_name, skip=True, **extra
):
    """GRPO with injected completions; micro-batch i of a generation batch is prompt i's group."""
    monkeypatch.setattr(grpo_module, "shuffle_sequence_dict", lambda d: d)
    args = {
        "per_device_train_batch_size": 4,
        "num_generations": 4,
        "gradient_accumulation_steps": ga,
        "steps_per_generation": spg,
        "max_completion_length": 8,
        "max_steps": max_steps,
        "learning_rate": 1e-2,
        "weight_decay": 0.1,
        "shuffle_dataset": False,
        **extra,
    }
    exp = make_config(tmp_path, "grpo", args, skip=skip, run_name=run_name)
    trace = ParamTrace()
    run = build_run(
        exp,
        train_dataset=Dataset.from_list([{"prompt": p} for p in prompts]),
        model=build_tiny_llama(len(tokenizer), seed=0),
        processing_class=tokenizer,
        reward_funcs=keyed_reward,
        rollout_func=window_rollout(tokenizer),
        callbacks=[trace],
    )
    return run, trace


def lags(run) -> list[float]:
    return run.recorder.history("rollout/policy_lag")


@pytest.mark.integration
def test_skip_decision_is_per_window_and_version_counts_applied_updates(tokenizer, tmp_path, monkeypatch):
    # generation batch = [varied, const, const]; grad_accum 1 -> windows: apply, skip, skip
    run, trace = window_run(
        tokenizer, tmp_path, monkeypatch, ["v0", "c1", "c2"], ga=1, spg=3, max_steps=6, run_name="a"
    )
    record = run.train()
    opt = run.hooked_optimizer
    assert (opt.nominal_steps, opt.applied_updates, opt.skipped_steps) == (6, 2, 4)
    assert record["steps"]["applied_updates"] == 2 and record["steps"]["skipped_steps"] == 4
    assert run.trainer.state.global_step == 6
    # two generations: before any update (v0) and after one applied update (v1), at global_step 3
    buf = run.trainer.trajectory_buffer
    assert buf.policy_versions() == [0, 1] and [b.global_step for b in buf] == [0, 3]
    assert lags(run) == [0, 1, 1, 0, 1, 1]  # stamp survives TRL's split; lag = applied since generation
    assert buf.counters()["skipped_windows"] == 4 and buf.counters()["zero_signal_windows"] == 4
    # skipped steps leave every parameter bit-identical (despite weight decay and Adam momentum)
    for step in (1, 2, 4, 5):  # 0-based steps 1,2 and 4,5 are skipped
        prev, cur = trace.after_step[step - 1], trace.after_step[step]
        assert all(torch.equal(prev[n], cur[n]) for n in prev), step
    assert not torch.equal(trace.after_step[2]["lm_head.weight"], trace.after_step[3]["lm_head.weight"])


@pytest.mark.integration
@pytest.mark.parametrize("order", [["v0", "c1"], ["c0", "v1"]])
def test_window_with_one_signal_microbatch_is_applied(tokenizer, tmp_path, monkeypatch, order):
    # grad_accum 2: each window holds one const and one varied micro-batch -> never skipped,
    # whichever of them comes last (a last-micro-batch-only check would fail one order).
    run, _ = window_run(tokenizer, tmp_path, monkeypatch, order, ga=2, spg=2, max_steps=3, run_name="b")
    run.train()
    opt = run.hooked_optimizer
    assert (opt.nominal_steps, opt.applied_updates, opt.skipped_steps) == (3, 3, 0)
    assert run.trainer.trajectory_buffer.policy_versions() == [0, 1, 2]


@pytest.mark.integration
def test_alternating_windows_with_grad_accum_one(tokenizer, tmp_path, monkeypatch):
    # grad_accum 1, steps_per_generation 2: [const, varied] -> skip, apply, skip, apply
    # (a per-generation-batch decision would apply every step)
    run, _ = window_run(
        tokenizer, tmp_path, monkeypatch, ["c0", "v1"], ga=1, spg=2, max_steps=4, run_name="c"
    )
    run.train()
    opt = run.hooked_optimizer
    assert (opt.nominal_steps, opt.applied_updates, opt.skipped_steps) == (4, 2, 2)
    assert run.trainer.trajectory_buffer.policy_versions() == [0, 1]
    assert lags(run) == [0, 0, 0, 0]  # the only update of each block happens on its last step


@pytest.mark.integration
def test_num_iterations_reuses_generation_with_growing_lag(tokenizer, tmp_path, monkeypatch):
    # num_iterations 2, steps_per_generation 2, grad_accum 1: one generation feeds 4 steps
    # [varied, const, varied, const] -> apply, skip, apply, skip; lag counts updates since generation.
    run, _ = window_run(
        tokenizer,
        tmp_path,
        monkeypatch,
        ["v0", "c1"],
        ga=1,
        spg=2,
        max_steps=4,
        run_name="d",
        num_iterations=2,
    )
    run.train()
    opt = run.hooked_optimizer
    assert (opt.nominal_steps, opt.applied_updates, opt.skipped_steps) == (4, 2, 2)
    assert run.trainer.trajectory_buffer.policy_versions() == [0]
    assert lags(run) == [0, 1, 1, 2]


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["grpo", "rloo"])
def test_constant_reward_skip_keeps_parameters_bit_identical(tokenizer, tmp_path, algorithm):
    results = {}
    for skip in (True, False):
        exp = make_config(
            tmp_path,
            algorithm,
            rl_args(algorithm, learning_rate=1e-2, weight_decay=0.1),
            skip=skip,
            run_name=f"{algorithm}-skip{skip}",
        )
        model = build_tiny_llama(len(tokenizer), seed=0)
        before = snapshot(model)
        run = build_run(
            exp,
            train_dataset=arithmetic_dataset(),
            model=model,
            processing_class=tokenizer,
            reward_funcs=constant_reward,
        )
        record = run.train()
        results[skip] = (before, snapshot(run.trainer.model), record, run)
    before, after, record, run = results[True]
    assert record["steps"] == {
        "nominal_optimizer_steps": 3,
        "applied_updates": 0,
        "skipped_steps": 3,
        "global_step": 3,
    }
    assert all(torch.equal(before[n], after[n]) for n in before)
    assert run.trainer.trajectory_buffer.policy_versions() == [0, 0, 0]
    assert all(not b.has_nonzero_advantage for b in run.trainer.trajectory_buffer)
    counters = run.trainer.trajectory_buffer.counters()
    assert counters["zero_signal_batches"] == 3 and counters["total_batches"] == 3
    assert counters["zero_signal_windows"] == 3 and counters["skipped_windows"] == 3
    # sensitivity: without the flag the zero-gradient steps still move weights (AdamW decay)
    before, after, record, _ = results[False]
    assert record["steps"]["applied_updates"] == 3
    name = "model.layers.0.mlp.gate_proj.weight"
    # zero grads -> Adam's update is 0; only decoupled decay p *= 1 - lr_k * wd acts (linear LR schedule)
    expected = before[name].clone()
    for lr_factor in (1.0, 2 / 3, 1 / 3):
        expected = expected * (1 - 1e-2 * lr_factor * 0.1)
    assert not torch.equal(before[name], after[name])
    torch.testing.assert_close(after[name], expected, rtol=1e-6, atol=1e-8)


def const_reward_of(value: float):
    def reward(completions, **kwargs):
        return [value] * len(completions)

    return reward


@pytest.mark.integration
@pytest.mark.parametrize(
    ("algorithm", "value", "num_generations", "beta"),
    [("grpo", 0.1, 8, 0.0), ("grpo", 0.9, 3, 0.0), ("rloo", 0.9, 4, 0.0), ("rloo", 0.9, 4, 0.05)],
)
def test_constant_non_dyadic_reward_is_skipped_despite_round_off(
    tokenizer, tiny_model_dir, tmp_path, algorithm, value, num_generations, beta
):
    """TRL's float32 advantages of these constant groups are round-off (~1e-8..1e-3), not zero.

    Adam would turn them into full lr-sized updates; the reward-based signal flags must skip.
    ``beta=0.05`` (RLOO default) loads the policy from disk so a reference model exists.
    """
    args = rl_args(
        algorithm,
        per_device_train_batch_size=2 * num_generations,
        num_generations=num_generations,
        learning_rate=1e-2,
        weight_decay=0.1,
    )
    if algorithm == "rloo":
        args["beta"] = beta
    exp = make_config(tmp_path, algorithm, args, skip=True, run_name=f"{algorithm}-{value}-{beta}")
    model = build_tiny_llama(len(tokenizer), seed=0) if beta == 0.0 else tiny_model_dir
    run = build_run(
        exp,
        train_dataset=arithmetic_dataset(),
        model=model,
        processing_class=tokenizer,
        reward_funcs=const_reward_of(value),
    )
    before = snapshot(run.trainer.model)
    record = run.train()
    buf = run.trainer.trajectory_buffer
    # not vacuous: TRL really handed round-off advantages to the loss
    assert max(float(b.advantages.abs().max()) for b in buf) > 0
    assert record["steps"]["applied_updates"] == 0 and record["steps"]["skipped_steps"] == 3
    after = snapshot(run.trainer.model)
    assert all(torch.equal(before[n], after[n]) for n in before)
    assert buf.counters()["zero_signal_batches"] == 3 and buf.counters()["skipped_windows"] == 3
    assert all(b.frac_zero_advantage == 1.0 for b in buf)


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["grpo", "rloo"])
def test_dense_reward_increases(tokenizer, tmp_path, algorithm):
    # Seeds 0-4 all rise from ~0.1 to >= 0.7 within 16 steps (checked when writing the test).
    exp = make_config(
        tmp_path,
        algorithm,
        rl_args(
            algorithm, per_device_train_batch_size=16, num_generations=8, max_steps=16, learning_rate=3e-2
        ),
    )
    run = build_run(
        exp,
        train_dataset=arithmetic_dataset(),
        model=build_tiny_llama(len(tokenizer), seed=0),
        processing_class=tokenizer,
        reward_funcs=digit_fraction,
        trajectory_capacity=4,
    )
    run.train()
    rewards = run.recorder.history("reward")
    assert len(rewards) == 16
    first, last = sum(rewards[:4]) / 4, sum(rewards[-4:]) / 4
    assert last > first + 0.3, rewards
    # the trajectory log is bounded and keeps the most recent generation batches
    buf = run.trainer.trajectory_buffer
    assert len(buf) == 4 and buf.counters()["total_batches"] == 16
    assert buf.policy_versions() == [12, 13, 14, 15]


@pytest.mark.integration
def test_grpo_resume_restores_policy_version_counters_and_optimizer(
    tokenizer, tmp_path, monkeypatch, recwarn
):
    prompts = ["v0", "c1", "c2"]
    full, _ = window_run(tokenizer, tmp_path, monkeypatch, prompts, ga=1, spg=3, max_steps=6, run_name="full")
    full.train()
    full_params = snapshot(full.trainer.model)

    part, _ = window_run(
        tokenizer,
        tmp_path,
        monkeypatch,
        prompts,
        ga=1,
        spg=3,
        max_steps=6,
        run_name="part",
        save_strategy="steps",
        save_steps=3,
    )
    part.train()
    ckpt = part.config.run_dir / "trainer" / "checkpoint-3"
    state = load_research_state(ckpt)
    # nominal, applied, skipped all differ: the version is applied updates, not global_step
    assert state["global_step"] == 3 and state["policy_version"] == 1
    assert state["steps"] == {"nominal": 3, "applied": 1, "skipped": 2, "global_step": 3}

    resumed, _ = window_run(
        tokenizer,
        tmp_path,
        monkeypatch,
        prompts,
        ga=1,
        spg=3,
        max_steps=6,
        run_name="part",  # same trainer output dir
        save_strategy="steps",
        save_steps=3,
    )
    with torch.no_grad():  # a different start: only a genuine resume can match the full run
        for p in resumed.trainer.model.parameters():
            p.add_(1.0)
    resumed.train(resume_from_checkpoint=str(ckpt))
    assert not [w for w in recwarn if "generation boundary" in str(w.message)]  # step 3 is aligned
    opt = resumed.hooked_optimizer
    assert (opt.nominal_steps, opt.applied_updates, opt.skipped_steps) == (6, 2, 4)
    buf = resumed.trainer.trajectory_buffer
    assert buf.policy_versions() == [1]  # first post-resume rollout carries the restored version
    assert buf.counters()["total_batches"] == 2 and buf.counters()["skipped_windows"] == 4
    assert ROLLOUT_COMPONENT in resumed.trainer.research_components
    resumed_params = snapshot(resumed.trainer.model)
    for name, tensor in full_params.items():
        assert torch.equal(tensor, resumed_params[name]), name
    record = ExperimentRecorder.load(resumed.recorder.run_dir)
    assert record["steps"]["applied_updates"] == 2 and record["steps"]["skipped_steps"] == 4


def test_policy_version_key_is_scalar_and_survives_trl_split(tokenizer, tmp_path):
    """Unit check of the stamp mechanics TRL relies on (shuffle + split keep 0-dim tensors)."""
    from trl.trainer.utils import shuffle_sequence_dict, split_tensor_dict

    from research_rl.rollout import read_policy_version, stamp_policy_version

    batch = {"advantages": torch.arange(6.0), "completion_ids": torch.zeros(6, 3)}
    batch[SIGNAL_KEY] = batch["advantages"] > 2  # per-sample flags travel with their samples
    stamp_policy_version(batch, 7)
    assert batch[POLICY_VERSION_KEY].ndim == 0
    parts = split_tensor_dict(shuffle_sequence_dict(batch), 3)
    assert [read_policy_version(p) for p in parts] == [7, 7, 7]
    assert all(p["advantages"].shape == (2,) for p in parts)
    for p in parts:
        assert p[SIGNAL_KEY].dtype == torch.bool and torch.equal(p[SIGNAL_KEY], p["advantages"] > 2)


@pytest.mark.parametrize(
    ("step", "ga", "spg", "num_iterations", "aligned"),
    [
        (3, 1, 3, 1, True),
        (2, 1, 3, 1, False),
        (1, 2, 4, 1, False),
        (2, 2, 4, 1, True),
        (2, 1, 2, 2, False),
        (4, 1, 2, 2, True),
        (3, 4, 2, 1, True),  # ga > spg: every step ends on a boundary
    ],
)
def test_resume_alignment_rule(step, ga, spg, num_iterations, aligned):
    # ``aligned`` is hand-derived: TRL generates before micro-batch k iff k % (spg * num_iterations) == 0,
    # and step s starts at micro-batch s * ga.
    assert resume_is_generation_aligned(step, ga, spg, num_iterations) is aligned


def test_unaligned_resume_warns(recwarn):
    class Fake:
        num_iterations = 1

    class Args:
        gradient_accumulation_steps = 1
        steps_per_generation = 3

    class State:
        global_step = 2

    callback = _WindowCallback(Fake())  # type: ignore[arg-type]
    with pytest.warns(UserWarning, match="not a generation boundary"):
        callback.on_train_begin(Args(), State(), None)
    State.global_step = 3
    callback.on_train_begin(Args(), State(), None)
    State.global_step = 0
    callback.on_train_begin(Args(), State(), None)
    assert not [w for w in recwarn if "generation boundary" in str(w.message)]
