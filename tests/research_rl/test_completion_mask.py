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
from research_rl.trainers.grpo_adapter import ResearchGRPOTrainer
from research_rl.trainers.rl_common import eos_cut_length, eos_truncating_rollout
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
            "logprobs": [[-1.0] * len(r) for r in rows],
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
    # logprobs were cut consistently with the ids
    assert out["sampling_per_token_logps"].shape == out["completion_ids"].shape


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
