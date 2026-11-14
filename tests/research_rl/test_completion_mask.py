"""TRL's completion_mask (GRPO and RLOO) against an independent first-EOS reference.

Completions are injected, never sampled, so every EOS position is controlled:

* GRPO ``rollout_func`` path: TRL takes the returned token lists verbatim and right-pads them.
* GRPO and RLOO ``transformers`` generation path: ``model.generate`` is replaced by a function
  that returns crafted fixed-width rows, including junk and a second EOS after the first EOS.
  This exercises TRL's own "mask after the first EOS" code. RLOO has no ``rollout_func`` in
  TRL 1.18, so this is its only injection point.

Reference semantics: the mask is 1 up to and including the first EOS and 0 after it. A row
without EOS is all ones, or all zeros with ``mask_truncated_completions=True``.
"""

from __future__ import annotations

import pytest
import torch
from datasets import Dataset
from trl import GRPOConfig, RLOOConfig

from research_rl.models.tiny import build_tiny_llama
from research_rl.rollout import RolloutBatch, TrajectoryBuffer
from research_rl.trainers.grpo_adapter import ResearchGRPOTrainer
from research_rl.trainers.rl_common import SIGNAL_KEY, eos_cut_length, eos_truncating_rollout
from research_rl.trainers.rloo_adapter import ResearchRLOOTrainer

EOS, PAD = 2, 0


@pytest.fixture(autouse=True)
def _single_thread():
    """Tiny models are dominated by thread-sync overhead on a loaded CPU: 1 thread is ~20x faster."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


WIDTH = 6
PROMPTS = ["1+1=", "12+3=", "7*7="]  # different lengths -> TRL left-pads the prompts

# Fixed-width rows for the generate path (what HF generate would return after the prompt).
GENERATE_ROWS = [
    [EOS, 11, 12, EOS, 13, 14],  # EOS at position 0, junk + second EOS after it
    [11, 12, EOS, 13, EOS, 14],  # EOS in the middle, junk after it
    [11, 12, 13, 14, 15, EOS],  # EOS at the last position
    [11, 12, 13, 14, 15, 16],  # never EOS (truncated at max_completion_length)
    [11, 12, EOS, PAD, PAD, PAD],  # EOS then right padding (what HF generate really emits)
    [EOS, EOS, EOS, EOS, EOS, EOS],  # only EOS tokens
]

# Variable-length lists for the rollout_func path; each stops at its own EOS (or never has one).
ROLLOUT_ROWS = [
    [EOS],  # EOS at position 0
    [11, 12, EOS],  # EOS in the middle of the padded batch
    [11, 12, 13, 14, 15, EOS],  # EOS at the end, full width
    [11, 12, 13, 14, 15, 16],  # never EOS, full width (truncated)
    [11, 12, 13, EOS],  # right-padded by TRL
    [17, 18, 19],  # never EOS, shorter than the batch (truncated early by the rollout engine)
]


def reference_mask(rows: list[list[int]], width: int, mask_truncated: bool) -> torch.Tensor:
    """Independent reference: 1 up to and including the first EOS, else whole row (or none)."""
    out = torch.zeros(len(rows), width, dtype=torch.long)
    for i, row in enumerate(rows):
        if EOS in row:
            out[i, : row.index(EOS) + 1] = 1
        elif not mask_truncated:
            out[i, : len(row)] = 1
    return out


def reference_ids(rows: list[list[int]], width: int) -> torch.Tensor:
    """Tokens kept by TRL: everything up to the first EOS, PAD afterwards."""
    out = torch.full((len(rows), width), PAD, dtype=torch.long)
    for i, row in enumerate(rows):
        keep = row[: row.index(EOS) + 1] if EOS in row else row
        out[i, : len(keep)] = torch.tensor(keep)
    return out


def length_reward(completion_ids, **kwargs):
    return [float(len(c)) for c in completion_ids]


def inputs_for(n: int) -> list[dict[str, str]]:
    return [{"prompt": PROMPTS[(i // 2) % len(PROMPTS)]} for i in range(n)]


def make_rollout(tokenizer, rows: list[list[int]]):
    def rollout(prompts, trainer):
        assert len(prompts) == len(rows)
        prompt_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": [list(r) for r in rows],
            "logprobs": [[-(t + 1.0) for t in range(len(r))] for r in rows],  # position-dependent
        }

    return rollout


def patch_generate(model, rows: list[list[int]]) -> list[torch.Tensor]:
    seen: list[torch.Tensor] = []

    def fake_generate(input_ids=None, attention_mask=None, **kwargs):
        seen.append(input_ids.clone())
        completion = torch.tensor(rows, dtype=input_ids.dtype, device=input_ids.device)
        assert completion.shape[0] == input_ids.shape[0]
        return torch.cat([input_ids, completion], dim=1)

    model.generate = fake_generate
    return seen


def grpo_trainer(tokenizer, tmp_path, rollout_func=None, **cfg):
    model = build_tiny_llama(len(tokenizer), seed=0)
    args = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=6,
        num_generations=2,
        max_completion_length=WIDTH,
        report_to=[],
        use_cpu=True,
        bf16=False,
        **cfg,
    )
    trainer = ResearchGRPOTrainer(
        model=model,
        reward_funcs=length_reward,
        args=args,
        train_dataset=Dataset.from_list([{"prompt": p} for p in PROMPTS]),
        processing_class=tokenizer,
        rollout_func=rollout_func,
    )
    trainer.model.train()
    return trainer


def rloo_trainer(tokenizer, tmp_path, **cfg):
    model = build_tiny_llama(len(tokenizer), seed=0)
    args = RLOOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=6,
        num_generations=2,
        max_completion_length=WIDTH,
        report_to=[],
        use_cpu=True,
        bf16=False,
        beta=0.0,
        **cfg,
    )
    trainer = ResearchRLOOTrainer(
        model=model,
        reward_funcs=length_reward,
        args=args,
        train_dataset=Dataset.from_list([{"prompt": p} for p in PROMPTS]),
        processing_class=tokenizer,
    )
    trainer.model.train()
    return trainer


def check_against_reference(out, rows, mask_truncated):
    ref = reference_mask(rows, WIDTH, mask_truncated)
    assert out["completion_mask"].shape == ref.shape
    assert torch.equal(out["completion_mask"].cpu().long(), ref)
    assert torch.equal(out["completion_ids"].cpu().long(), reference_ids(rows, WIDTH))
    # padding / post-EOS masking must actually be exercised by this fixture
    assert 0 < int(ref.sum()) < ref.numel()


# ---------------------------------------------------------------------------------- GRPO
@pytest.mark.parametrize("mask_truncated", [False, True])
def test_grpo_generate_path_mask_matches_first_eos_reference(tokenizer, tmp_path, mask_truncated):
    trainer = grpo_trainer(tokenizer, tmp_path, mask_truncated_completions=mask_truncated)
    seen = patch_generate(trainer.model, GENERATE_ROWS)
    out = trainer._generate_and_score_completions(inputs_for(len(GENERATE_ROWS)))
    assert len(seen) == 1  # the injected generate really was used
    check_against_reference(out, GENERATE_ROWS, mask_truncated)
    # loss-token count used by the default "dapo" loss normaliser
    assert int(out["num_items_in_batch"]) == int(reference_mask(GENERATE_ROWS, WIDTH, mask_truncated).sum())


@pytest.mark.parametrize("mask_truncated", [False, True])
def test_grpo_rollout_func_mask_matches_reference(tokenizer, tmp_path, mask_truncated):
    trainer = grpo_trainer(
        tokenizer,
        tmp_path,
        rollout_func=make_rollout(tokenizer, ROLLOUT_ROWS),
        mask_truncated_completions=mask_truncated,
    )
    out = trainer._generate_and_score_completions(inputs_for(len(ROLLOUT_ROWS)))
    check_against_reference(out, ROLLOUT_ROWS, mask_truncated)
    assert int(out["num_items_in_batch"]) == int(reference_mask(ROLLOUT_ROWS, WIDTH, mask_truncated).sum())
    # learning-signal flags: a fully masked row (truncated, mask_truncated_completions) has none,
    # even though its advantage is non-zero (row 5: length 3 vs. its sibling's 4)
    adv = out["advantages"].cpu()
    assert adv[5] != 0
    expected = (adv != 0) & reference_mask(ROLLOUT_ROWS, WIDTH, mask_truncated).bool().any(dim=1)
    assert torch.equal(out[SIGNAL_KEY].cpu(), expected)
    assert bool(out[SIGNAL_KEY][5]) is (not mask_truncated)


# Rows that *continue* after EOS or end in padding without EOS: TRL's rollout path differs from
# the first-EOS reference here (it trusts the rollout). These tests pin that behaviour down and
# show that ``eos_truncating_rollout`` restores the reference semantics.
QUIRK_ROWS = [
    [11, EOS, 13, 14],  # tokens after EOS
    [EOS, EOS, 15],  # tokens after EOS at position 0
    [11, 12, PAD, PAD],  # no EOS but ends in padding
    [11, 12, 13, 14, 15, 16],
    [11, EOS, PAD, PAD, PAD, PAD],
    [16, 17, EOS],
]


def test_grpo_rollout_func_trusts_returned_tokens(tokenizer, tmp_path):
    """Documented TRL 1.18 behaviour: mask = 1 on every returned token; last-token pad = 'terminated'."""
    trainer = grpo_trainer(
        tokenizer, tmp_path, rollout_func=make_rollout(tokenizer, QUIRK_ROWS), mask_truncated_completions=True
    )
    out = trainer._generate_and_score_completions(inputs_for(len(QUIRK_ROWS)))
    length_mask = torch.zeros(len(QUIRK_ROWS), WIDTH, dtype=torch.long)
    for i, row in enumerate(QUIRK_ROWS):
        terminated = row[-1] in (EOS, PAD)
        length_mask[i, : len(row)] = int(terminated)
    assert torch.equal(out["completion_mask"].cpu().long(), length_mask)
    assert not torch.equal(length_mask, reference_mask(QUIRK_ROWS, WIDTH, True))  # i.e. a real difference


@pytest.mark.parametrize("mask_truncated", [False, True])
def test_eos_truncating_rollout_restores_reference(tokenizer, tmp_path, mask_truncated):
    rollout = eos_truncating_rollout(make_rollout(tokenizer, QUIRK_ROWS))
    trainer = grpo_trainer(
        tokenizer, tmp_path, rollout_func=rollout, mask_truncated_completions=mask_truncated
    )
    out = trainer._generate_and_score_completions(inputs_for(len(QUIRK_ROWS)))
    # Reference on the rows with trailing padding stripped (padding is not content).
    stripped = []
    for row in QUIRK_ROWS:
        n = len(row)
        while EOS not in row[:n] and n and row[n - 1] == PAD:
            n -= 1
        stripped.append(row[:n])
    ref = reference_mask(stripped, WIDTH, mask_truncated)
    assert torch.equal(out["completion_mask"].cpu().long(), ref)
    assert torch.equal(out["completion_ids"].cpu().long(), reference_ids(stripped, WIDTH))
    # logprobs were cut consistently with the ids: -(t+1) on kept tokens, TRL's 0.0 padding after
    expected = torch.zeros(len(stripped), WIDTH)
    for i, row in enumerate(stripped):
        n = row.index(EOS) + 1 if EOS in row else len(row)  # kept length
        expected[i, :n] = -torch.arange(1.0, n + 1)
    assert torch.equal(out["sampling_per_token_logps"].cpu(), expected)


def test_eos_cut_length_and_empty_completion(tokenizer, tmp_path):
    assert eos_cut_length([5, EOS, 7], [EOS], PAD) == 2
    assert eos_cut_length([5, 6, PAD, PAD], [EOS], PAD) == 2
    assert eos_cut_length([5, 6, 7], [EOS], None) == 3
    rollout = eos_truncating_rollout(make_rollout(tokenizer, [[PAD, PAD]] * 6))
    with pytest.raises(ValueError, match="empty"):
        rollout(["1+1="] * 6, grpo_trainer(tokenizer, tmp_path))


# ---------------------------------------------------------------------------------- RLOO
@pytest.mark.parametrize("mask_truncated", [False, True])
def test_rloo_generate_path_mask_matches_first_eos_reference(tokenizer, tmp_path, mask_truncated):
    trainer = rloo_trainer(tokenizer, tmp_path, mask_truncated_completions=mask_truncated)
    seen = patch_generate(trainer.model, GENERATE_ROWS)
    out = trainer._generate_and_score_completions(inputs_for(len(GENERATE_ROWS)))
    assert len(seen) == 1
    check_against_reference(out, GENERATE_ROWS, mask_truncated)


def test_rloo_has_no_rollout_func_in_this_trl():
    """If TRL adds ``rollout_func`` to RLOO, the rollout-path tests above should be extended to it."""
    import inspect

    from trl import RLOOTrainer

    assert "rollout_func" not in inspect.signature(RLOOTrainer.__init__).parameters


def test_trajectory_log_unbatches_rows(tokenizer, tmp_path):
    """RolloutBatch.trajectories(): prompt left-padding removed, rows/scores aligned with TRL's output."""
    trainer = grpo_trainer(tokenizer, tmp_path, rollout_func=make_rollout(tokenizer, ROLLOUT_ROWS))
    batch_inputs = inputs_for(len(ROLLOUT_ROWS))
    out = trainer._generate_and_score_completions(batch_inputs)
    logged = trainer.trajectory_buffer.latest()
    assert logged is not None and len(trainer.trajectory_buffer) == 1
    trajectories = logged.trajectories()
    assert len(trajectories) == len(ROLLOUT_ROWS)
    for i, traj in enumerate(trajectories):
        assert traj.prompt_ids == tokenizer(batch_inputs[i]["prompt"], add_special_tokens=False)["input_ids"]
        assert traj.completion_ids == out["completion_ids"][i].tolist()
        assert traj.completion_mask == out["completion_mask"][i].tolist()
        assert traj.num_loss_tokens == len(ROLLOUT_ROWS[i])
        assert traj.reward == float(len(ROLLOUT_ROWS[i]))  # length_reward on the unpadded ids
        assert traj.advantage == pytest.approx(float(out["advantages"][i]))
        assert traj.policy_version == 0
    stats = logged.stats()
    assert stats["num_samples"] == 6 and stats["num_loss_tokens"] == sum(len(r) for r in ROLLOUT_ROWS)
    assert stats["mean_reward"] == pytest.approx(sum(len(r) for r in ROLLOUT_ROWS) / 6)
    assert int(out["policy_version"]) == 0


def test_eval_rollouts_do_not_touch_the_training_log_or_window(tokenizer, tmp_path):
    trainer = grpo_trainer(tokenizer, tmp_path, rollout_func=make_rollout(tokenizer, ROLLOUT_ROWS))
    trainer.model.eval()
    out = trainer._prepare_inputs(inputs_for(len(ROLLOUT_ROWS)))
    assert out["completion_ids"].shape[0] == len(ROLLOUT_ROWS)  # the eval rollout really ran
    assert out[SIGNAL_KEY].any()
    assert len(trainer.trajectory_buffer) == 0 and trainer.trajectory_buffer.counters()["total_batches"] == 0
    assert trainer._rr_window_microbatches == 0 and not trainer._rr_window_signal
    assert trainer.trajectory_buffer.counters()["windows"] == 0


def _batch(advantages: torch.Tensor, signal: torch.Tensor | None = None) -> RolloutBatch:
    n = advantages.shape[0]
    return RolloutBatch.from_tensors(
        prompt_ids=torch.ones(n, 2, dtype=torch.long),
        prompt_mask=torch.ones(n, 2, dtype=torch.long),
        completion_ids=torch.ones(n, 3, dtype=torch.long),
        completion_mask=torch.ones(n, 3, dtype=torch.long),
        rewards=torch.tensor([1.0, float("nan"), 3.0])[:n],
        advantages=advantages,
        policy_version=4,
        signal=signal,
    )


def test_rollout_batch_token_level_advantages_and_signal():
    # token-level (B, C) advantages: a sample is zero only if every token is zero
    adv = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
    batch = _batch(adv)
    assert batch.frac_zero_advantage == pytest.approx(2 / 3) and batch.has_nonzero_advantage
    assert batch.mean_reward == pytest.approx(2.0)  # NaN (unscorable) excluded
    assert not _batch(torch.zeros(3, 3)).has_nonzero_advantage
    # an explicit signal flag overrides the exact test (round-off advantages, no signal)
    flagged = _batch(torch.tensor([1e-7, -1e-7, 0.0]), signal=torch.tensor([False, False, False]))
    assert not flagged.has_nonzero_advantage and flagged.frac_zero_advantage == 1.0
    flagged = _batch(torch.tensor([1e-7, 0.5, 0.0]), signal=torch.tensor([False, True, False]))
    assert flagged.has_nonzero_advantage and flagged.frac_zero_advantage == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="signal"):
        _batch(torch.zeros(3), signal=torch.tensor([True]))


def test_trajectory_buffer_counters_round_trip():
    buf = TrajectoryBuffer(capacity=2)
    buf.append(_batch(torch.zeros(3)))
    buf.append(_batch(torch.tensor([0.0, 1.0, 0.0])))
    buf.record_window(has_signal=False, skipped=True)
    buf.record_window(has_signal=True, skipped=False)
    state = buf.state_dict()
    assert state["zero_signal_batches"] == 1 and state["last_policy_version"] == 4
    restored = TrajectoryBuffer(capacity=2)
    restored.load_state_dict(state)
    assert restored.counters() == buf.counters()
    assert restored.last_policy_version == 4 and len(restored) == 0  # tensors are not checkpointed
