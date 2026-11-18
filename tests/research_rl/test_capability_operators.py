"""Tests of research_rl.capability.operators: TorchGRPOOperator and group_advantages.

CPU only, tiny Llama built in-process. A random tiny model essentially never satisfies the
verifiers, so the scoring / screening tests script the sampled token rows by overriding
``TorchGRPOOperator._generate`` (the one documented override point); everything downstream (EOS
cut, truncation, V1/V2 scoring, advantages, loss, optimizer step) is the real code. Trained values
are never compared bitwise; bit-exactness is asserted only for elements that must not be written
(masked channels, parameters outside the optimizer) and for the untouched global RNG state.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import pytest
import torch
from torch import nn

from research_rl.capability.operators import (
    ADVANTAGE_EPS,
    TRAIN_PURPOSE,
    TorchGRPOOperator,
    group_advantages,
    loss_seed,
    sample_seed,
)
from research_rl.capability.tasks import (
    CompositionSplits,
    ExactAnswerVerifier,
    ItemSampler,
    OperatorSet,
    SuffixLeakVerifier,
)
from research_rl.capability.types import GateDecision, Split, UpdateOperator
from research_rl.models.ffn_channels import ChannelSelection, channel_mask
from research_rl.models.model_interface import find_swiglu_ffns
from research_rl.models.tiny import build_char_tokenizer, build_tiny_llama
from research_rl.optim.optimizer_wrapper import HookedOptimizer
from research_rl.optim.update_interface import MaskRule

MAX_NEW = 8


@pytest.fixture(scope="module")
def splits() -> CompositionSplits:
    return CompositionSplits.build(OperatorSet(seed=0, num_operators=10), seed=0)


@pytest.fixture(scope="module")
def sampler(splits: CompositionSplits) -> ItemSampler:
    return ItemSampler(splits, root_seed=9)


@pytest.fixture(scope="module")
def tok() -> Any:
    return build_char_tokenizer()


def make(
    tok: Any,
    sampler: ItemSampler,
    *,
    cls: type[TorchGRPOOperator] = TorchGRPOOperator,
    model: nn.Module | None = None,
    v1: Any = None,
    v2: Any = None,
    **kwargs: Any,
) -> TorchGRPOOperator:
    model = model if model is not None else build_tiny_llama(len(tok), seed=0)
    params: dict[str, Any] = {
        "group_size": 4,
        "prompts_per_update": 3,
        "lr": 1e-2,
        "max_new_tokens": MAX_NEW,
        "temperature": 1.0,
        "seed": 1,
    }
    params.update(kwargs)
    return cls(
        model,
        tok,
        sampler,
        v1 if v1 is not None else SuffixLeakVerifier(k=2),
        v2 if v2 is not None else ExactAnswerVerifier(),
        **params,
    )


def leaky(answer: str) -> str:
    """Wrong answer (first digit changed) with the right last two digits."""
    first = "8" if answer[0] == "9" else "9"
    return first + answer[1:]


def wrong(answer: str) -> str:
    """Wrong in the last digit (fails both verifiers)."""
    return answer[:-1] + ("0" if answer[-1] != "0" else "1")


class Scripted(TorchGRPOOperator):
    """Rows per group: [correct, leaky, wrong, correct + EOS-free padding to the budget]."""

    pattern: Sequence[str] = ("correct", "leaky", "wrong", "correct")

    def _generate(self, model: nn.Module, prompts: Sequence[Sequence[int]], seed: int) -> list[list[int]]:
        items = self.items(self._current)
        rows: list[list[int]] = []
        eos = int(self.tokenizer.eos_token_id)
        pad = int(self.tokenizer.pad_token_id)
        for r in range(len(prompts)):
            answer = items[r // self.group_size].answer
            kind = self.pattern[r % len(self.pattern)]
            if kind == "truncated":  # fills the budget, no EOS; suffix would match V1
                text = ("1" * MAX_NEW + answer)[-MAX_NEW:]
                rows.append(list(self.tokenizer(text, add_special_tokens=False)["input_ids"]))
                continue
            text = {"correct": answer, "leaky": leaky(answer), "wrong": wrong(answer)}[kind]
            ids = [*list(self.tokenizer(text, add_special_tokens=False)["input_ids"]), eos]
            rows.append(ids + [pad] * (MAX_NEW - len(ids)))
        return rows

    def rollouts(self, model: nn.Module, update_index: int) -> Any:
        self._current = update_index
        return super().rollouts(model, update_index)


def params_of(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


# ============================================================================== group_advantages
def test_group_advantages_values() -> None:
    adv = group_advantages([1, 0, 0, 0, 1, 1, 1, 1], 4)
    mean, std = 0.25, math.sqrt(0.1875)
    expected = [(1 - mean) / (std + ADVANTAGE_EPS)] + [(0 - mean) / (std + ADVANTAGE_EPS)] * 3
    assert adv[:4] == pytest.approx(expected, rel=1e-12)
    assert adv[4:] == [0.0, 0.0, 0.0, 0.0]  # zero-variance group -> exactly 0
    assert group_advantages([0.0] * 6, 3) == [0.0] * 6
    keep = [True, False, True, True, True, True, True, True]
    masked = group_advantages([1, 1, 0, 0, 0, 1, 0, 1], 4, keep)
    plain = group_advantages([1, 1, 0, 0, 0, 1, 0, 1], 4)
    assert masked[1] == 0.0 and masked[0] == plain[0] and masked[4:] == plain[4:]
    assert sum(plain[:4]) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("rewards", "group", "keep"),
    [([1, 0, 1], 2, None), ([1, 0], 0, None), ([math.nan, 0], 2, None), ([1, 0], 2, [True])],
)
def test_group_advantages_rejects(rewards: list[float], group: int, keep: list[bool] | None) -> None:
    with pytest.raises(ValueError):
        group_advantages(rewards, group, keep)


# ============================================================================== real sampling
def test_step_report_and_protocol(tok: Any, sampler: ItemSampler) -> None:
    op = make(tok, sampler)
    assert isinstance(op, UpdateOperator)
    model = op.model
    rng = torch.random.get_rng_state()
    report = op.step(model, 0)
    assert torch.equal(torch.random.get_rng_state(), rng)  # sampling ran inside fork_rng
    n = 3 * 4
    assert len(report.v1_pass) == len(report.v2_pass) == len(report.completion_lengths) == n
    assert report.extra["alarm_units"] == [i // 4 for i in range(n)]
    assert report.forward_passes == 2 and report.backward_passes == 1
    assert all(0 <= length <= MAX_NEW for length in report.completion_lengths)
    assert len(report.extra["advantages"]) == n and math.isfinite(report.extra["loss"])
    items = sampler.draw(Split.TRAIN, 3, purpose=TRAIN_PURPOSE, index=0)
    assert report.extra["uids"] == [it.uid for it in items]
    assert report.extra["sample_seed"] == sample_seed(1, 0)
    assert op.parameter_names() == [n for n, _ in model.named_parameters()]
    assert op.steps_done == 1 and op.forward_passes_total == 2 and op.backward_passes_total == 1
    op.on_decision(GateDecision.ROLLBACK)
    with pytest.raises(TypeError):
        op.on_decision("rollback")  # type: ignore[arg-type]


def test_rollouts_seeded_per_update_index(tok: Any, sampler: ItemSampler) -> None:
    op = make(tok, sampler)
    model = op.model
    a = op.rollouts(model, 3)
    op.rollouts(model, 0)  # other draws in between do not change update 3's samples
    b = op.rollouts(model, 3)
    assert a.actions == b.actions and a.texts == b.texts and a.seed == b.seed == sample_seed(1, 3)
    c = op.rollouts(model, 4)
    assert c.actions != a.actions
    # rows are padded/cut like greedy_generate: every row either ends with EOS or fills the budget
    eos = tok.eos_token_id
    for actions, free in zip(a.actions, a.no_eos):
        assert (actions[-1] == eos) != free
        assert len(actions) <= MAX_NEW and (not free or len(actions) == MAX_NEW)
        assert eos not in actions[:-1]
    assert a.lengths == tuple(len(x) - (0 if f else 1) for x, f in zip(a.actions, a.no_eos))
    other = make(tok, sampler)
    with pytest.raises(ValueError, match="model"):
        other.rollouts(model, 0)


def test_rollouts_match_independent_sampling(tok: Any, sampler: ItemSampler) -> None:
    """Batched left-padded sampling equals sampling with the same seed through HF directly."""
    op = make(tok, sampler, prompts_per_update=2, group_size=2)
    model = op.model
    ro = op.rollouts(model, 0)
    from transformers import GenerationConfig

    width = max(len(p) for p in ro.prompt_ids)
    ids = torch.zeros(len(ro.prompt_ids), width, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for r, p in enumerate(ro.prompt_ids):
        ids[r, width - len(p) :] = torch.tensor(p)
        mask[r, width - len(p) :] = 1
    cfg = GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        max_new_tokens=MAX_NEW,
        eos_token_id=2,
        pad_token_id=0,
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(ro.seed)
        model.eval()
        out = model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg)  # type: ignore[operator]
    for r, row in enumerate(out[:, width:].tolist()):
        cut = row.index(2) + 1 if 2 in row else len(row)
        assert tuple(row[:cut]) == ro.actions[r]


def test_construction_checks(tok: Any, sampler: ItemSampler) -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        make(tok, sampler, max_new_tokens=7)  # answers have up to 7 characters
    with pytest.raises(ValueError):
        make(tok, sampler, group_size=0)
    with pytest.raises(ValueError):
        make(tok, sampler, temperature=0.0)
    model = build_tiny_llama(len(tok), seed=0)
    alien = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="not a parameter"):
        make(tok, sampler, model=model, optimizer=torch.optim.AdamW(alien.parameters()))


def test_state_dict_roundtrip(tok: Any, sampler: ItemSampler) -> None:
    op = make(tok, sampler)
    op.step(op.model, 0)
    state = op.state_dict()
    fresh = make(tok, sampler)
    fresh.load_state_dict(state)
    assert fresh.state_dict() == state
    other = make(tok, sampler, seed=2)
    with pytest.raises(ValueError, match="config"):
        other.load_state_dict(state)


# ============================================================================== scoring and screening
def test_screening_zeroes_leaky_rows(tok: Any, sampler: ItemSampler) -> None:
    op = make(tok, sampler, cls=Scripted)
    ro = op.rollouts(op.model, 0)
    leaks = [r for r in range(len(ro.v1)) if ro.v1[r] == 1.0 and ro.v2[r] == 0.0]
    assert leaks == [1 + 4 * g for g in range(3)]  # exactly the scripted leaky rows
    assert all(ro.v1[r] == ro.v2[r] == 1.0 for r in range(0, 12, 4))  # correct rows pass both
    assert all(ro.v1[r] == ro.v2[r] == 0.0 for r in range(2, 12, 4))  # wrong rows fail both
    report = op.step(op.model, 0)
    adv = report.extra["advantages"]
    assert report.screened == 3
    assert all(adv[r] == 0.0 for r in leaks)
    assert any(adv[r] != 0.0 for r in range(12) if r not in leaks)
    expected = group_advantages(ro.v1, 4, [r not in leaks for r in range(12)])
    assert adv == pytest.approx(expected)
    unscreened = make(tok, sampler, cls=Scripted, screen=False)
    rep2 = unscreened.step(unscreened.model, 0)
    assert rep2.screened == 0 and all(rep2.extra["advantages"][r] > 0.0 for r in leaks)


def test_truncated_rows_score_zero_under_both_verifiers(tok: Any, sampler: ItemSampler) -> None:
    class Trunc(Scripted):
        pattern = ("truncated", "correct")

    op = make(tok, sampler, cls=Trunc, group_size=2)
    ro = op.rollouts(op.model, 0)
    for r in range(0, len(ro.v1), 2):
        assert ro.truncated[r] and ro.no_eos[r]
        assert ro.v1[r] == 0.0 and ro.v2[r] == 0.0  # the suffix matches, but truncation wins
        assert SuffixLeakVerifier(k=2).verify(ro.texts[r], ro.items[r // 2].answer).correct
        assert ro.lengths[r] == MAX_NEW
    assert all(ro.v1[r] == 1.0 and not ro.truncated[r] for r in range(1, len(ro.v1), 2))


def test_loss_masks_prompt_and_padding(tok: Any, sampler: ItemSampler) -> None:
    """The batched, left/right-padded loss equals -sum(A * log pi) / #action tokens computed row by
    row without any padding (prompt and pad positions contribute nothing)."""
    op = make(tok, sampler, cls=Scripted)
    model = op.model
    ro = op.rollouts(model, 0)
    adv, _ = op.advantages(ro)
    loss = op.policy_loss(model, ro, adv)
    # manual reference: one row at a time, no padding at all
    total, count = 0.0, 0
    with torch.no_grad():
        for prompt, actions, a in zip(ro.prompt_ids, ro.actions, adv):
            seq = torch.tensor([list(prompt) + list(actions)])
            logits = model(input_ids=seq).logits[0, len(prompt) - 1 : -1].float()
            logp = torch.log_softmax(logits, -1).gather(-1, torch.tensor(actions).unsqueeze(-1)).squeeze(-1)
            total += float((a * logp).sum())
            count += len(actions)
    torch.testing.assert_close(loss.detach(), torch.tensor(-total / count), rtol=1e-4, atol=1e-5)


def test_hooked_optimizer_mask_rule_keeps_frozen_channels(tok: Any, sampler: ItemSampler) -> None:
    model = build_tiny_llama(len(tok), seed=0)
    selection = ChannelSelection({0: [0, 3, 5, 7, 11]})
    ref = find_swiglu_ffns(model, [0])[0]
    names = [f"{ref.module_path}.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")]
    named = dict(model.named_parameters())
    inner = torch.optim.AdamW([named[n] for n in names], lr=1e-2, weight_decay=0.1)
    opt = HookedOptimizer(inner, model.named_parameters(), [MaskRule(channel_mask(model, selection))])
    op = make(tok, sampler, cls=Scripted, model=model, optimizer=opt)
    assert op.parameter_names() == names
    before = params_of(model)
    mask = channel_mask(model, selection)
    for t in range(3):
        op.step(model, t)
    changed = False
    for name, p in model.named_parameters():
        now = p.detach()
        owned = mask.get(name)
        if owned is None:
            assert torch.equal(now, before[name]), name  # outside the optimizer: untouched
            continue
        assert torch.equal(now[~owned], before[name][~owned]), name  # frozen elements bit-exact
        changed = changed or not torch.equal(now[owned], before[name][owned])
    assert changed  # the selected channels were trained
    assert all(p.grad is None for p in model.parameters())


def test_zero_advantages_leave_default_adamw_weights_unchanged(tok: Any, sampler: ItemSampler) -> None:
    class AllWrong(Scripted):
        pattern = ("wrong",)

    op = make(tok, sampler, cls=AllWrong)
    before = params_of(op.model)
    report = op.step(op.model, 0)
    assert report.extra["advantages"] == [0.0] * 12 and report.reward_v1 == 0.0
    for name, p in op.model.named_parameters():  # zero gradient, no weight decay: no write at all
        assert torch.equal(p.detach(), before[name]), name


def test_adamw_trains_with_signal(tok: Any, sampler: ItemSampler) -> None:
    op = make(tok, sampler, cls=Scripted)
    before = params_of(op.model)
    report = op.step(op.model, 0)
    assert report.reward_v1 == pytest.approx(9 / 12) and report.reward_v2 == pytest.approx(6 / 12)
    assert any(not torch.equal(p.detach(), before[n]) for n, p in op.model.named_parameters())
    assert all(p.grad is None for p in op.model.parameters())


def dropout_model(tok: Any, p: float) -> nn.Module:
    """Tiny Llama with attention dropout ``p`` in train mode (dropout active in the loss forward)."""
    model = build_tiny_llama(len(tok), seed=0)
    for layer in model.model.layers:
        layer.self_attn.attention_dropout = p
    model.train()
    return model


def test_dropout_in_loss_forward_is_seeded_per_update(tok: Any, sampler: ItemSampler) -> None:
    """Regression: with dropout active, the training forward must not draw from the global RNG,
    so update ``t`` is a function of ``(seed, t)`` and the weights only."""
    results = []
    for global_seed in (1, 2):
        op = make(tok, sampler, cls=Scripted, model=dropout_model(tok, 0.5))
        torch.manual_seed(global_seed)
        rng = torch.random.get_rng_state()
        op.step(op.model, 0)
        assert torch.equal(rng, torch.random.get_rng_state())  # global RNG untouched
        results.append(params_of(op.model))
    # trained values: allclose (unseeded dropout made them differ by ~1e-2 before the fix)
    for name, value in results[0].items():
        assert torch.allclose(value, results[1][name], rtol=0.0, atol=1e-6), name
    # ... and the dropout really is active and keyed by the update index (different masks)
    a = make(tok, sampler, cls=Scripted, model=dropout_model(tok, 0.5))
    rollouts = a.rollouts(a.model, 0)
    adv, _ = a.advantages(rollouts)
    other = dataclasses.replace(rollouts, update_index=1)
    with torch.no_grad():
        l0 = a.policy_loss(a.model, rollouts, adv)
        l0b = a.policy_loss(a.model, rollouts, adv)
        l1 = a.policy_loss(a.model, other, adv)
    assert torch.equal(l0, l0b) and not torch.equal(l0, l1)
    assert loss_seed(1, 0) != sample_seed(1, 0)


def test_zero_advantage_update_after_signal_still_moves_adamw(tok: Any, sampler: ItemSampler) -> None:
    """Documents the module docstring (step 7): only a *fresh* AdamW leaves the weights unchanged on
    an all-zero-advantage update; after an update with signal Adam's first moment keeps moving them
    (the one optimizer.step per update is never skipped)."""

    class AllWrong(Scripted):
        pattern = ("wrong",)

    op = make(tok, sampler, cls=Scripted)
    op.step(op.model, 0)  # signal: non-zero moments
    op.__class__ = AllWrong
    before = params_of(op.model)
    report = op.step(op.model, 1)
    assert report.extra["advantages"] == [0.0] * 12
    assert any(not torch.equal(p.detach(), before[n]) for n, p in op.model.named_parameters())
