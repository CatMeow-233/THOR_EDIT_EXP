"""The capability gate inside TRL GRPO: callback schedule, exact rollback, buffer, resume, screen.

* Unit tests (fast): the screen on hand-built generation batches, argument checks, optimizer
  unwrapping.
* Integration tests (``@pytest.mark.integration``): ``ResearchGRPOTrainer`` through the GRPO adapter
  on the tiny CPU model with a scripted probe (forced ACCEPT / HOLD / REJECT evidence), a real
  ``CapabilityProbe`` for the disabled-gate catch-all, and the screen fed by injected completions.

Nothing here runs on a GPU.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from datasets import Dataset
from transformers import TrainerCallback

from research_rl.capability.alarm import UNITS_KEY, DisagreementAlarm
from research_rl.capability.gate import CapabilityGate, GateConfig
from research_rl.capability.probe import CapabilityProbe
from research_rl.capability.tasks import (
    CompositionSplits,
    ExactAnswerVerifier,
    ItemSampler,
    OperatorSet,
    SuffixLeakVerifier,
    make_score_fn,
)
from research_rl.capability.trl_gate import (
    GATE_COMPONENT,
    CapabilityGateCallback,
    DisagreementScreen,
    base_optimizer,
    generate_every,
)
from research_rl.capability.types import GateDecision, GateEvidence, PairedScores, RollbackUnit, Split
from research_rl.core.config import DataSpec, ExperimentConfig, ModelSpec, OptimSpec, TrainerSpec
from research_rl.models.model_interface import fingerprint, snapshot
from research_rl.models.tiny import build_tiny_llama
from research_rl.optim import HookedOptimizer
from research_rl.trainers import build_run
from research_rl.trainers.rl_common import ADVANTAGE_SHAPER_COMPONENT

EOS = 2


@pytest.fixture(autouse=True)
def _single_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


# ============================================================================== scripted probe
TRANSFER_SLICE = "t_rule/unseen_pairs"
RETENTION_SLICE = "retention/copy"


def scripted_evidence(decision_index: int, attempt: int, outcome: str, alarm: bool) -> GateEvidence:
    """``pass``: transfer gain + retention ok; ``hold``: no gain, retention ok; ``fail``: retention
    drop of 25 points (passable in principle, so the reason is ``retention_failed``)."""
    gain = outcome == "pass"
    ok = outcome in ("pass", "hold")
    key = f"{decision_index}/{attempt}"
    nt, nr = 20, 400
    transfer = PairedScores(
        slice=TRANSFER_SLICE,
        split=Split.T_RULE,
        uids=tuple(f"t/{key}/{i}" for i in range(nt)),
        incumbent=(0.0,) * nt,
        candidate=(1.0 if gain else 0.0,) * nt,
    )
    retention = PairedScores(
        slice=RETENTION_SLICE,
        split=Split.RETENTION,
        uids=tuple(f"r/{key}/{i}" for i in range(nr)),
        incumbent=(1.0,) * nr,
        candidate=tuple(1.0 if (ok or i >= 100) else 0.0 for i in range(nr)),
    )
    return GateEvidence(
        decision_index=decision_index, transfer=(transfer,), retention=(retention,), alarm=alarm
    )


class ScriptedProbe:
    """Evidence follows ``plan[decision_index]`` (last entry repeats); records what it saw."""

    def __init__(self, plan: Sequence[str]) -> None:
        self.plan = list(plan)
        self.calls: list[dict[str, Any]] = []

    def evidence(self, model, incumbent, decision_index, *, attempt=0, alarm=False) -> GateEvidence:
        self.calls.append(
            {
                "decision_index": decision_index,
                "attempt": attempt,
                "alarm": alarm,
                "incumbent": incumbent.fingerprint,
                "candidate": fingerprint(snapshot(model, incumbent.names())),
            }
        )
        outcome = self.plan[min(decision_index, len(self.plan) - 1)]
        return scripted_evidence(decision_index, attempt, outcome, alarm)


def gate_config(**overrides: Any) -> GateConfig:
    kwargs: dict[str, Any] = {
        "alpha": 0.05,
        "alpha_schedule": "per_decision",
        "shrink_lambdas": (),
        "block_updates": 2,
        "items_per_transfer_slice": 20,
        "items_per_retention_slice": 400,
    }
    kwargs.update(overrides)
    return GateConfig(**kwargs)


def make_gate(probe: Any, config: GateConfig, **kwargs: Any) -> CapabilityGateCallback:
    kwargs.setdefault("block_updates", config.block_updates)
    return CapabilityGateCallback(
        probe, CapabilityGate(config), rollback_unit=RollbackUnit.PARAMETERS, **kwargs
    )


# ============================================================================== unit tests
def test_explicit_block_updates_is_authoritative():
    config = gate_config(block_updates=2)
    cb = CapabilityGateCallback(
        ScriptedProbe(["pass"]),
        CapabilityGate(config),
        block_updates=3,
        rollback_unit=RollbackUnit.PARAMETERS,
    )
    assert cb.block_updates == 3 and cb.state_dict()["block_updates"] == 3
    with pytest.raises(ValueError, match="block_updates"):
        make_gate(ScriptedProbe(["pass"]), config, block_updates=0)
    with pytest.raises(TypeError, match="alarm"):
        make_gate(ScriptedProbe(["pass"]), config, alarm=object())


def test_on_train_begin_requires_bind():
    cb = make_gate(ScriptedProbe(["pass"]), gate_config())
    args = SimpleNamespace(gradient_accumulation_steps=1, steps_per_generation=1, world_size=1)
    with pytest.raises(RuntimeError, match="bind"):
        cb.on_train_begin(args, SimpleNamespace(global_step=0), None)


def test_base_optimizer_unwraps_wrappers(tiny_model):
    adam = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)

    class Wrapper(torch.optim.Optimizer):  # accelerate's AcceleratedOptimizer is a subclass too
        def __init__(self, inner):
            self.optimizer = inner

    assert base_optimizer(Wrapper(adam)) is adam
    assert base_optimizer(None) is None
    hooked = HookedOptimizer(
        torch.optim.AdamW(tiny_model.parameters(), lr=1e-3), tiny_model.named_parameters()
    )
    assert base_optimizer(Wrapper(hooked)) is hooked
    args = SimpleNamespace(gradient_accumulation_steps=2, steps_per_generation=None)
    assert generate_every(args, SimpleNamespace(num_iterations=3)) == 6


# ---- the screen on hand-built batches
def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def fake_batch(tokenizer, prompts: Sequence[str], completions: Sequence[list[int]], *, per_token=False):
    """Left-padded prompts, right-padded completions with TRL's mask (1 up to and incl. first EOS)."""
    p_rows = [_ids(tokenizer, p) for p in prompts]
    width = max(len(r) for r in p_rows)
    prompt_ids = torch.zeros(len(p_rows), width, dtype=torch.long)
    prompt_mask = torch.zeros_like(prompt_ids)
    for i, r in enumerate(p_rows):
        prompt_ids[i, width - len(r) :] = torch.tensor(r)
        prompt_mask[i, width - len(r) :] = 1
    cw = max(len(c) for c in completions)
    completion_ids = torch.zeros(len(completions), cw, dtype=torch.long)
    completion_mask = torch.zeros_like(completion_ids)
    for i, c in enumerate(completions):
        completion_ids[i, : len(c)] = torch.tensor(c)
        n = c.index(EOS) + 1 if EOS in c else len(c)
        completion_mask[i, :n] = 1
    n_rows = len(completions)
    adv = torch.arange(1, n_rows + 1, dtype=torch.float32)
    if per_token:
        adv = adv[:, None].repeat(1, cw)
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "advantages": adv,
    }


def fake_trainer(*, training=True, num_generations=4, shapers=(), trl_signal=None):
    return SimpleNamespace(
        model=SimpleNamespace(training=training),
        num_generations=num_generations,
        num_generations_eval=num_generations,
        eos_token_ids=[EOS],
        accelerator=SimpleNamespace(process_index=0),
        state=SimpleNamespace(global_step=7),
        advantage_shapers=list(shapers),
        trl_advantage_signal=trl_signal,
        train_dataset=None,
    )


def leak_alarm() -> DisagreementAlarm:
    return DisagreementAlarm(threshold=0.05, min_rollouts=1, drift=0.0, baseline_gap=0.0)


def screen_case(tokenizer):
    # prompt "12+3=" answer "15": correct "15", leak "95" (suffix k=1 passes V1 only), wrong "33",
    # no EOS "15" + budget filler (truncated -> 0 under both verifiers).
    comps = [
        [*_ids(tokenizer, "15"), EOS],
        [*_ids(tokenizer, "95"), EOS],
        [*_ids(tokenizer, "33"), EOS],
        _ids(tokenizer, "15555"),  # no EOS: truncated
    ]
    return ["12+3="] * 4, comps


@pytest.mark.parametrize("per_token", [False, True])
def test_screen_zeroes_v1_only_rows_and_feeds_alarm(tokenizer, per_token):
    alarm = leak_alarm()
    screen = DisagreementScreen(
        SuffixLeakVerifier(k=1),
        ExactAnswerVerifier(),
        alarm,
        tokenizer,
        max_new_tokens=8,
        answers={"12+3=": "15"},
    )
    prompts, comps = screen_case(tokenizer)
    out = fake_batch(tokenizer, prompts, comps, per_token=per_token)
    trainer = fake_trainer(shapers=[screen], trl_signal=torch.tensor([True, True, False, True]))
    result = screen(out, trainer)
    expected_rows = torch.tensor([1.0, 0.0, 3.0, 4.0])
    if per_token:
        assert torch.equal(result.advantages, expected_rows[:, None].repeat(1, out["advantages"].shape[1]))
    else:
        assert torch.equal(result.advantages, expected_rows)
    assert result.advantages.dtype == out["advantages"].dtype
    assert torch.equal(out["advantages"][:, 0] if per_token else out["advantages"], torch.arange(1.0, 5.0))
    # first shaper: TRL's reward-based flags AND the keep mask
    assert result.signal.tolist() == [True, False, False, True]
    assert screen.last_keep == [True, False, True, True]
    assert (screen.train_batches, screen.rows, screen.screened) == (1, 4, 1)
    state = screen.last_state
    assert state is not None and state.alarm and state.gap == pytest.approx(0.25)
    assert alarm.updates == 1 and alarm.gap_alarms == 1
    # V1 pass rate 2/4 (correct + leak; the truncated row scores 0 under V1 too), V2 1/4
    assert screen.recent(1) == [(0.5, 0.25)]


def test_screen_signal_when_not_first_shaper_and_units(tokenizer, monkeypatch):
    seen: list = []
    alarm = leak_alarm()
    monkeypatch.setattr(alarm, "update", lambda report: seen.append(report) or None)
    screen = DisagreementScreen(
        SuffixLeakVerifier(k=1),
        ExactAnswerVerifier(),
        alarm,
        tokenizer,
        max_new_tokens=8,
        answers=lambda p: "15",
    )
    prompts, comps = screen_case(tokenizer)
    out = fake_batch(tokenizer, prompts * 2, comps * 2)
    out["advantages"][2] = 0.0
    trainer = fake_trainer(
        num_generations=4, shapers=[object(), screen], trl_signal=torch.ones(8, dtype=torch.bool)
    )
    result = screen(out, trainer)
    # not the first shaper: exact non-zero test on the incoming advantages, then the keep mask
    assert result.signal.tolist() == [True, False, False, True, True, False, True, True]
    (report,) = seen
    assert report.extra[UNITS_KEY] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert report.update_index == 7 and report.screened == 2
    assert report.completion_lengths == (2, 2, 2, 5) * 2
    assert report.v1_pass == (True, True, False, False) * 2
    assert report.v2_pass == (True, False, False, False) * 2


def test_screen_eval_mode_and_errors(tokenizer):
    alarm = leak_alarm()
    screen = DisagreementScreen(
        SuffixLeakVerifier(k=1),
        ExactAnswerVerifier(),
        alarm,
        tokenizer,
        max_new_tokens=8,
        answers={"12+3=": "15"},
    )
    prompts, comps = screen_case(tokenizer)
    # eval batches are returned untouched, also for prompts outside the answer table
    result = screen(fake_batch(tokenizer, ["77+7="] * 4, comps), fake_trainer(training=False))
    assert result.advantages.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert result.signal.tolist() == [True] * 4
    assert alarm.updates == 0 and screen.train_batches == 0 and screen.last_keep == []
    with pytest.raises(ValueError, match="different prompts"):
        mixed = DisagreementScreen(
            SuffixLeakVerifier(k=1),
            ExactAnswerVerifier(),
            alarm,
            tokenizer,
            max_new_tokens=8,
            answers=lambda p: "15",
        )
        mixed(fake_batch(tokenizer, ["12+3=", "21+4=", "12+3=", "12+3="], comps), fake_trainer())
    with pytest.raises(KeyError, match="no reference answer"):
        screen(fake_batch(tokenizer, ["99+1="] * 4, comps), fake_trainer())
    no_answers = DisagreementScreen(
        SuffixLeakVerifier(k=1), ExactAnswerVerifier(), None, tokenizer, max_new_tokens=8
    )
    with pytest.raises(ValueError, match="answer"):
        no_answers(fake_batch(tokenizer, prompts, comps), fake_trainer())
    with pytest.raises(ValueError, match="max_new_tokens"):
        DisagreementScreen(SuffixLeakVerifier(), ExactAnswerVerifier(), None, tokenizer, max_new_tokens=0)


def test_screen_truncation_rule_by_budget(tokenizer):
    """A completion with EOS whose text fills the budget is truncated (gate rule) -> 0 under V1/V2."""
    screen = DisagreementScreen(
        SuffixLeakVerifier(k=1),
        ExactAnswerVerifier(),
        None,
        tokenizer,
        max_new_tokens=2,
        answers={"12+3=": "15"},
    )
    comps = [[*_ids(tokenizer, "15"), EOS], [*_ids(tokenizer, "5"), EOS]] * 2
    screen(fake_batch(tokenizer, ["12+3="] * 4, comps), fake_trainer())
    # "15" re-encodes to 2 tokens = the budget -> truncated, 0 under both; "5" passes V1 only
    assert screen.recent(1) == [(0.5, 0.0)]
    assert screen.last_keep == [True, False, True, False]


def test_screen_recovers_masked_truncated_rows(tokenizer):
    """``mask_truncated_completions=True`` zeroes the mask of rows without EOS; the screen recovers
    them from the pad-stripped ids, so its report equals the one for the unmasked batch (regression:
    lengths 0 for budget-filling rows looked like a length collapse)."""

    def make_screen(alarm):
        return DisagreementScreen(
            SuffixLeakVerifier(k=1),
            ExactAnswerVerifier(),
            alarm,
            tokenizer,
            max_new_tokens=6,
            answers={"12+3=": "15"},
        )

    def run(masked: bool):
        alarm = DisagreementAlarm(threshold=10.0, min_rollouts=1, drift=0.0, baseline_updates=1, window=1)
        screen = make_screen(alarm)
        reports: list = []
        original = alarm.update
        monkey = lambda report: (reports.append(report), original(report))[1]  # noqa: E731
        alarm.update = monkey  # type: ignore[method-assign]
        trainer = fake_trainer()
        trainer.mask_truncated_completions = masked
        trainer._tokenizer = tokenizer
        # baseline batch: four "15" + EOS (length 2)
        screen(fake_batch(tokenizer, ["12+3="] * 4, [[*_ids(tokenizer, "15"), EOS]] * 4), trainer)
        # the completions grow: three rows fill the 6-token budget without EOS
        comps = [[*_ids(tokenizer, "15"), EOS]] + [_ids(tokenizer, "155555")] * 3
        out = fake_batch(tokenizer, ["12+3="] * 4, comps)
        if masked:  # what TRL does for rows whose last token is neither EOS nor pad
            out["completion_mask"][1:] = 0
        result = screen(out, trainer)
        return reports[-1], alarm, screen, result

    plain, plain_alarm, _, plain_result = run(False)
    masked, masked_alarm, masked_screen, masked_result = run(True)
    assert plain.completion_lengths == (2, 6, 6, 6)
    assert masked.completion_lengths == plain.completion_lengths
    assert (masked.v1_pass, masked.v2_pass, masked.extra) == (plain.v1_pass, plain.v2_pass, plain.extra)
    assert masked.extra["truncated"] == 3
    assert masked_alarm.length_alarms == plain_alarm.length_alarms == 0
    assert masked_screen.last_state is not None and not masked_screen.last_state.alarm
    assert torch.equal(masked_result.advantages, plain_result.advantages)
    # without the flag an all-zero mask row stays an empty completion
    screen = make_screen(None)
    out = fake_batch(tokenizer, ["12+3="] * 4, [[*_ids(tokenizer, "15"), EOS]] * 4)
    out["completion_mask"][1] = 0
    screen(out, fake_trainer())
    assert screen.recent(1) == [(0.75, 0.75)]
    # recovering a masked row needs a pad id
    no_pad = make_screen(None)
    no_pad.tokenizer = SimpleNamespace(pad_token_id=None)
    trainer = fake_trainer()
    trainer.mask_truncated_completions = True
    trainer._tokenizer = SimpleNamespace(pad_token_id=None)
    with pytest.raises(ValueError, match="pad token id"):
        no_pad(fake_batch(tokenizer, ["12+3="] * 4, [[*_ids(tokenizer, "15"), EOS]] * 4), trainer)


def test_screen_default_answers_from_dataset_and_state(tokenizer):
    alarm = leak_alarm()
    screen = DisagreementScreen(
        SuffixLeakVerifier(k=1), ExactAnswerVerifier(), alarm, tokenizer, max_new_tokens=8
    )
    trainer = fake_trainer()
    trainer.train_dataset = Dataset.from_list(
        [{"prompt": "12+3=", "answer": "15"}, {"prompt": "1+1=", "answer": "2"}]
    )
    prompts, comps = screen_case(tokenizer)
    screen(fake_batch(tokenizer, prompts, comps), trainer)
    assert screen.answers == {"12+3=": "15", "1+1=": "2"}
    state = copy.deepcopy(screen.state_dict())
    other_alarm = leak_alarm()
    other = DisagreementScreen(
        SuffixLeakVerifier(k=1), ExactAnswerVerifier(), other_alarm, tokenizer, max_new_tokens=8
    )
    other.load_state_dict(state)
    assert other_alarm.state_dict() == alarm.state_dict()
    assert (other.train_batches, other.rows, other.screened, other.recent(5)) == (1, 4, 1, [(0.5, 0.25)])
    with pytest.raises(ValueError, match="alarm"):
        DisagreementScreen(
            SuffixLeakVerifier(), ExactAnswerVerifier(), None, tokenizer, max_new_tokens=8
        ).load_state_dict(state)
    bad = Dataset.from_list([{"prompt": "1+1=", "answer": "2"}, {"prompt": "1+1=", "answer": "3"}])
    conflicting = DisagreementScreen(
        SuffixLeakVerifier(), ExactAnswerVerifier(), None, tokenizer, max_new_tokens=8
    )
    with pytest.raises(ValueError, match="two different answers"):
        conflicting(
            fake_batch(tokenizer, ["1+1="] * 4, comps),
            SimpleNamespace(**{**vars(trainer), "train_dataset": bad}),
        )


# ============================================================================== trainer runs
def make_config(
    tmp_path, args: dict[str, Any], *, run_name: str, optim: OptimSpec | None = None
) -> ExperimentConfig:
    return ExperimentConfig(
        study_id="trl_gate_test",
        model=ModelSpec(name_or_path="tiny-llama-in-memory"),
        data=DataSpec(name="synthetic"),
        trainer=TrainerSpec(algorithm="grpo", args=args),
        seed=0,
        output_dir=str(tmp_path),
        run_name=run_name,
        optim=optim or OptimSpec(hooked=True),
    )


def digit_fraction(completions, **kwargs):
    return [sum(ch.isdigit() for ch in c) / max(len(c), 1) for c in completions]


def grpo_args(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "per_device_train_batch_size": 4,
        "num_generations": 4,
        "gradient_accumulation_steps": 1,
        "max_completion_length": 6,
        "max_steps": 4,
        "learning_rate": 1e-2,
        "logging_steps": 1,
    }
    args.update(overrides)
    return args


ARITH = Dataset.from_list([{"prompt": f"{a}+{b}=", "answer": str(a + b)} for a in range(4) for b in range(4)])


class Trace(TrainerCallback):
    """Copies of the trainable parameters, optimizer state and lr at every on_step_end."""

    def __init__(self) -> None:
        self.trainer: Any = None
        self.params: dict[int, dict[str, torch.Tensor]] = {}
        self.optim: dict[int, dict[str, Any]] = {}
        self.masters: dict[int, dict[str, torch.Tensor]] = {}
        self.lr: dict[int, list[float]] = {}

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        self.params[step] = snapshot(model)
        opt = base_optimizer(self.trainer.optimizer)
        inner = opt.inner if isinstance(opt, HookedOptimizer) else opt
        self.optim[step] = copy.deepcopy(inner.state_dict())
        self.lr[step] = [g["lr"] for g in opt.param_groups]
        if isinstance(opt, HookedOptimizer) and opt.master_weights:
            self.masters[step] = {
                n: opt.master_of(p).detach().clone() for n, p in model.named_parameters() if opt.has_master(p)
            }


class PostTrace(Trace):
    """The same trace, registered after the gate callback (sees the weights after the decision)."""


def build(tmp_path, tokenizer, *, run_name, args, probe=None, config=None, gated=True, **kwargs):
    """GRPO run with a pre-gate trace (before the decision) and a post-gate trace (after it)."""
    model_seed = kwargs.pop("model_seed", 0)
    dtype = kwargs.pop("dtype", torch.float32)
    optim = kwargs.pop("optim", None)
    shapers = kwargs.pop("advantage_shapers", ())
    gate_kwargs = kwargs.pop("gate_kwargs", {})
    pre = Trace()
    run = build_run(
        make_config(tmp_path, args, run_name=run_name, optim=optim),
        train_dataset=kwargs.pop("dataset", ARITH),
        model=build_tiny_llama(len(tokenizer), seed=model_seed).to(dtype),
        processing_class=tokenizer,
        reward_funcs=kwargs.pop("reward_funcs", digit_fraction),
        callbacks=[pre],
        advantage_shapers=list(shapers),
        **kwargs,
    )
    pre.trainer = run.trainer
    gens: list[dict[str, torch.Tensor]] = []
    original = run.trainer._generate_and_score_completions

    def recording(inputs):
        gens.append(snapshot(run.trainer.model))
        return original(inputs)

    run.trainer._generate_and_score_completions = recording
    cb = None
    if gated:
        cb = make_gate(probe, config, **gate_kwargs).bind(run.trainer)
    post = PostTrace()
    post.trainer = run.trainer
    run.trainer.add_callback(post)
    return SimpleNamespace(run=run, trainer=run.trainer, gate=cb, pre=pre, post=post, gens=gens)


def same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def optim_state_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    sa, sb = a["state"], b["state"]
    if sa.keys() != sb.keys():
        return False
    for k in sa:
        if sa[k].keys() != sb[k].keys():
            return False
        for name in sa[k]:
            x, y = sa[k][name], sb[k][name]
            if isinstance(x, torch.Tensor):
                if not (isinstance(y, torch.Tensor) and x.dtype == y.dtype and torch.equal(x.cpu(), y.cpu())):
                    return False
            elif x != y:
                return False
    return True


OPTIM_VARIANTS = {
    "hooked": (OptimSpec(hooked=True), torch.float32),
    "plain": (OptimSpec(hooked=False), torch.float32),
    "bf16_masters": (OptimSpec(hooked=True, master_weights=True), torch.bfloat16),
}


@pytest.mark.integration
@pytest.mark.parametrize("variant", sorted(OPTIM_VARIANTS))
def test_forced_reject_restores_params_optimizer_and_next_generation(tokenizer, tmp_path, variant):
    optim, dtype = OPTIM_VARIANTS[variant]
    config = gate_config(block_updates=2)
    probe = ScriptedProbe(["pass", "fail"])  # decision 0 (step 2) ACCEPT, decision 1 (step 4) REJECT
    log = tmp_path / "decisions.jsonl"
    r = build(
        tmp_path,
        tokenizer,
        run_name="reject",
        args=grpo_args(max_steps=5),
        probe=probe,
        config=config,
        gate_kwargs={"log_path": log},
        optim=optim,
        dtype=dtype,
    )
    assert r.trainer.research_components[GATE_COMPONENT] is r.gate
    r.trainer.train()
    gate = r.gate
    assert [rec.decision for rec in gate.records] == [GateDecision.ACCEPT, GateDecision.ROLLBACK]
    assert gate.records[1].reason == "retention_failed"
    assert [rec.update_index for rec in gate.records] == [1, 3]
    # non-vacuous: the step-4 candidate differs from the step-2 incumbent in weights and moments
    assert not same(r.pre.params[4], r.pre.params[2])
    assert not optim_state_equal(r.pre.optim[4], r.pre.optim[2])
    assert r.pre.optim[2]["state"]  # the incumbent's Adam moments are non-empty
    # ROLLBACK: parameters and per-parameter optimizer state bit-identical to the step-2 copies
    assert same(r.post.params[4], r.pre.params[2])
    assert optim_state_equal(r.post.optim[4], r.pre.optim[2])
    if variant == "bf16_masters":  # fp32 masters restored bit-identically too
        assert r.pre.masters[2] and not same(r.pre.masters[4], r.pre.masters[2])
        assert same(r.post.masters[4], r.pre.masters[2])
    # the lr stays the scheduler's (linear decay), not the incumbent's
    assert r.post.lr[4] == r.pre.lr[4]
    assert r.post.lr[4] != r.pre.lr[2]
    # the generation of step 5 (one generation per step) used the restored weights
    assert len(r.gens) == 5 and same(r.gens[4], r.pre.params[2])
    assert gate.buffer_invalidations == 0  # aligned decisions rely on TRL's own regeneration
    # policy versions stay monotonic: counters keep counting rolled-back steps
    opt = r.run.hooked_optimizer
    if variant == "plain":
        assert opt is None
    else:
        assert (opt.nominal_steps, opt.applied_updates, opt.skipped_steps) == (5, 5, 0)
    assert r.trainer.trajectory_buffer.policy_versions() == [0, 1, 2, 3, 4]
    assert min(r.run.recorder.history("rollout/policy_lag")) >= 0
    # the probe evaluated the step-2 incumbent at decision 1
    assert probe.calls[1]["incumbent"] == fingerprint(
        {n: t for n, t in r.pre.params[2].items() if n in gate.incumbent.params}
    )
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(x["decision"], x["global_step"], x["trigger"], x["buffer_invalidated"]) for x in rows] == [
        ("accept", 2, "block", False),
        ("rollback", 4, "block", False),
    ]


@pytest.mark.integration
def test_misaligned_block_is_refused(tokenizer, tmp_path):
    config = gate_config(block_updates=1)
    r = build(
        tmp_path,
        tokenizer,
        run_name="misaligned",
        args=grpo_args(steps_per_generation=2),
        probe=ScriptedProbe(["pass"]),
        config=config,
    )
    with pytest.raises(ValueError, match="multiple of"):
        r.trainer.train()


# ---- screen in the trainer, alarm decisions off the generation boundary
SCREEN_PROMPTS = ["12+3=", "21+4="]
SCREEN_ANSWERS = {"12+3=": "15", "21+4=": "25"}


def leak_rollout(tokenizer):
    """Per prompt group: correct, V1-only leak, wrong, wrong (gap 0.25 per batch)."""

    def texts(prompt: str) -> list[str]:
        ans = SCREEN_ANSWERS[prompt]
        return [ans, "9" + ans[-1], "33", "4"]

    def rollout(prompts, trainer):
        seen: dict[str, int] = {}
        completions = []
        for p in prompts:
            k = seen.get(p, 0)
            seen[p] = k + 1
            completions.append([*_ids(tokenizer, texts(p)[k % 4]), EOS])
        return {
            "prompt_ids": [_ids(tokenizer, p) for p in prompts],
            "completion_ids": completions,
            "logprobs": [[0.0] * len(c) for c in completions],
        }

    return rollout


def length_reward(completion_ids, **kwargs):
    return [float(len(c)) for c in completion_ids]


@pytest.mark.integration
def test_alarm_decision_off_boundary_invalidates_buffer(tokenizer, tmp_path):
    alarm = leak_alarm()
    screen = DisagreementScreen(
        SuffixLeakVerifier(k=1), ExactAnswerVerifier(), alarm, tokenizer, max_new_tokens=6
    )
    config = gate_config(block_updates=2)
    probe = ScriptedProbe(["fail"])  # every decision: retention fails; with the alarm -> confirmed_alarm
    r = build(
        tmp_path,
        tokenizer,
        run_name="alarm",
        args=grpo_args(steps_per_generation=2, max_steps=4, shuffle_dataset=False),
        probe=probe,
        config=config,
        dataset=Dataset.from_list([{"prompt": p, "answer": a} for p, a in SCREEN_ANSWERS.items()]),
        reward_funcs=length_reward,
        rollout_func=leak_rollout(tokenizer),
        advantage_shapers=[screen],
        gate_kwargs={"alarm": alarm},
    )
    key = f"{ADVANTAGE_SHAPER_COMPONENT}/0/{DisagreementScreen.name}"
    assert r.trainer.research_components[key] is screen  # auto-registered stateful shaper
    initial = snapshot(r.trainer.model)
    r.trainer.train()
    gate = r.gate
    # generate_every = 2: TRL alone generates at micro-batches 0 and 2; the alarm decisions at steps 1
    # and 3 (off the boundary) rolled back and invalidated the buffer -> extra generations at 1 and 3
    assert [(rec.decision, rec.reason) for rec in gate.records] == [
        (GateDecision.ROLLBACK, "confirmed_alarm")
    ] * 4
    assert gate.buffer_invalidations == 2
    assert len(r.gens) == 4
    for gen in r.gens:
        assert same(gen, initial)  # every generation saw the restored (initial) weights
    assert not same(r.pre.params[1], initial)  # non-vacuous: step 1 did move the weights
    # the step-2 block decision is aligned (micro-batch 2): TRL's own regeneration served it
    assert not same(r.pre.params[2], initial)
    for step in (1, 2, 3, 4):
        assert same(r.post.params[step], initial)
    assert all(call["alarm"] for call in probe.calls)
    # screen in the real trainer: V1-only rows have advantage 0 and no signal
    batch = r.trainer.trajectory_buffer.latest()
    leak_rows = [i for i in range(8) if i % 4 == 1]
    assert all(float(batch.advantages[i]) == 0.0 for i in leak_rows)
    assert not any(bool(batch.signal[i]) for i in leak_rows)
    assert screen.train_batches == 4 and screen.screened == 8
    opt = r.run.hooked_optimizer
    assert (opt.nominal_steps, opt.applied_updates) == (4, 4)
    assert r.trainer.trajectory_buffer.policy_versions() == [0, 1, 2, 3]
    assert min(r.run.recorder.history("rollout/policy_lag")) >= 0
    # rewards of the block are recorded beside the decision
    assert gate.records[0].reward_v2 == pytest.approx(0.25)
    assert gate.records[0].reward_v1 is not None


class RecordingScreen(DisagreementScreen):
    """Keeps the completion masks it was handed and the alarm reports it produced."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.masks: list[torch.Tensor] = []
        self.reports: list[Any] = []

    def __call__(self, output, trainer):
        self.masks.append(output["completion_mask"].detach().cpu().clone())
        assert self.alarm is not None
        original = self.alarm.update
        self.alarm.update = lambda report: (self.reports.append(report), original(report))[1]  # type: ignore[method-assign]
        try:
            return super().__call__(output, trainer)
        finally:
            del self.alarm.update


@pytest.mark.integration
def test_screen_lengths_with_mask_truncated_completions(tokenizer, tmp_path):
    """The real trainer with ``mask_truncated_completions=True`` hands the screen all-zero mask rows
    for budget-filling completions; the reported lengths equal those of the unmasked run."""
    runs = {}
    for flag in (False, True):
        screen = RecordingScreen(
            SuffixLeakVerifier(k=1),
            ExactAnswerVerifier(),
            DisagreementAlarm(threshold=10.0, min_rollouts=1, drift=0.0),
            tokenizer,
            max_new_tokens=6,
        )
        r = build(
            tmp_path,
            tokenizer,
            run_name=f"mask{flag}",
            args=grpo_args(max_steps=1, mask_truncated_completions=flag),
            gated=False,
            advantage_shapers=[screen],
        )
        r.trainer.train()
        runs[flag] = screen
    plain, masked = runs[False], runs[True]
    (plain_report,), (masked_report,) = plain.reports, masked.reports
    assert masked_report.extra["truncated"] > 0  # non-vacuous: some rows filled the budget
    zero_rows = (masked.masks[0].sum(dim=1) == 0).sum().item()
    assert zero_rows > 0 and (plain.masks[0].sum(dim=1) == 0).sum().item() == 0
    assert max(plain_report.completion_lengths) == 6
    assert masked_report.completion_lengths == plain_report.completion_lengths
    assert (masked_report.v1_pass, masked_report.v2_pass) == (plain_report.v1_pass, plain_report.v2_pass)


# ---- resume
@pytest.mark.integration
def test_resume_restores_gate_state(tokenizer, tmp_path):
    config = gate_config(block_updates=2, hold_limit=1)
    args = grpo_args(max_steps=4, save_strategy="steps", save_steps=2)
    original = snapshot(build_tiny_llama(len(tokenizer), seed=0))

    full = build(
        tmp_path,
        tokenizer,
        run_name="full",
        args=grpo_args(max_steps=4),
        probe=ScriptedProbe(["hold"]),
        config=config,
    )
    full.trainer.train()
    assert [(rec.decision, rec.reason) for rec in full.gate.records] == [
        (GateDecision.HOLD, "no_transfer_gain"),
        (GateDecision.ROLLBACK, "hold_limit"),
    ]

    log = tmp_path / "part.jsonl"
    part = build(
        tmp_path,
        tokenizer,
        run_name="part",
        args=args,
        probe=ScriptedProbe(["hold"]),
        config=config,
        gate_kwargs={"log_path": log},
    )
    part.trainer.train()
    ckpt = part.run.config.run_dir / "trainer" / "checkpoint-2"
    saved = torch.load(ckpt / "research_rl_components.pt", weights_only=True)[GATE_COMPONENT]
    assert saved["decision_index"] == 1 and saved["controller"]["holds"] == 1
    assert saved["controller"]["verdict_gate"]["ledger"]["count"] == 1
    assert len(log.read_text().splitlines()) == 2

    probe = ScriptedProbe(["hold"])
    resumed = build(
        tmp_path,
        tokenizer,
        run_name="part",
        args=args,
        probe=probe,
        config=config,
        model_seed=123,
        gate_kwargs={"log_path": log},
    )
    assert not same(snapshot(resumed.trainer.model), original)
    resumed.trainer.train(resume_from_checkpoint=str(ckpt))
    gate = resumed.gate
    # only decision 1 ran, and the restored hold streak turned its HOLD into a hold_limit ROLLBACK
    assert [(rec.decision_index, rec.decision, rec.reason) for rec in gate.records] == [
        (1, GateDecision.ROLLBACK, "hold_limit")
    ]
    assert gate.gate.ledger.count == 2 and gate.decision_index == 2
    # the incumbent is the original run's start, not the checkpoint's or the resumed init
    assert [c["decision_index"] for c in probe.calls] == [1]
    assert probe.calls[0]["incumbent"] == fingerprint({n: original[n] for n in gate.incumbent.params})
    final = snapshot(resumed.trainer.model)
    assert same(final, original) and same(final, snapshot(full.trainer.model))
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(x["decision_index"], x["decision"]) for x in rows] == [(0, "hold"), (1, "rollback")]


@pytest.mark.integration
def test_second_train_continues_the_gate(tokenizer, tmp_path):
    """A second ``train()`` without a checkpoint continues the same gate (regression: it recaptured
    the held, never-accepted weights as the incumbent and reset the hold streak)."""
    config = gate_config(block_updates=2, hold_limit=1)
    probe = ScriptedProbe(["hold"])
    log = tmp_path / "twice.jsonl"
    r = build(
        tmp_path,
        tokenizer,
        run_name="twice",
        args=grpo_args(max_steps=2),
        probe=probe,
        config=config,
        gate_kwargs={"log_path": log},
    )
    initial = snapshot(r.trainer.model)
    r.trainer.train()
    assert [rec.decision for rec in r.gate.records] == [GateDecision.HOLD]
    held = snapshot(r.trainer.model)
    assert not same(held, initial) and r.gate.controller.holds == 1
    r.trainer.train()
    gate = r.gate
    assert [(rec.decision_index, rec.decision, rec.reason) for rec in gate.records] == [
        (0, GateDecision.HOLD, "no_transfer_gain"),
        (1, GateDecision.ROLLBACK, "hold_limit"),
    ]
    initial_fp = fingerprint({n: initial[n] for n in gate.incumbent.params})
    assert [c["incumbent"] for c in probe.calls] == [initial_fp, initial_fp]
    assert probe.calls[1]["candidate"] != initial_fp
    assert same(snapshot(r.trainer.model), initial)
    assert gate.decision_index == 2 and gate.gate.ledger.count == 2 and gate.controller.holds == 0
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(x["decision_index"], x["decision"], x["global_step"]) for x in rows] == [
        (0, "hold", 2),
        (1, "rollback", 2),  # HF restarts global_step in every train()
    ]
    # a continued gate refuses a model whose trainable parameters no longer match the incumbent
    next(p for p in r.trainer.model.parameters() if p.requires_grad).requires_grad_(False)
    with pytest.raises(ValueError, match="trainable parameters changed"):
        r.trainer.train()


def test_load_state_dict_rejects_mismatch(tokenizer):
    cb = make_gate(ScriptedProbe(["pass"]), gate_config())
    state = cb.state_dict()
    other = make_gate(ScriptedProbe(["pass"]), gate_config(hold_limit=2))
    with pytest.raises(ValueError, match="GateConfig"):
        other.load_state_dict(state)
    with pytest.raises(ValueError, match="version"):
        cb.load_state_dict({**state, "version": 99})
    with pytest.raises(ValueError, match="decision_index"):
        cb.load_state_dict({**state, "decision_index": -1})


# ---- catch-all: a disabled gate with a real probe changes nothing in training
@pytest.fixture(scope="module")
def real_probe_parts(tokenizer):
    splits = CompositionSplits.build(OperatorSet(seed=0, num_operators=10), seed=0)
    sampler = ItemSampler(splits, root_seed=3)
    score = make_score_fn(tokenizer, ExactAnswerVerifier(), max_new_tokens=8, batch_size=16)
    return sampler, score


@pytest.mark.integration
def test_disabled_gate_with_real_probe_leaves_training_bit_identical(tokenizer, tmp_path, real_probe_parts):
    sampler, score = real_probe_parts
    config = gate_config(
        enabled=False, block_updates=2, items_per_transfer_slice=6, items_per_retention_slice=6
    )
    probe = CapabilityProbe(
        sampler,
        score,
        transfer_slices=(TRANSFER_SLICE,),
        retention_slices=(RETENTION_SLICE,),
        items_per_transfer_slice=6,
        items_per_retention_slice=6,
    )
    plain = build(tmp_path, tokenizer, run_name="plain", args=grpo_args(max_steps=4), gated=False)
    plain.trainer.train()
    gated = build(
        tmp_path, tokenizer, run_name="disabled", args=grpo_args(max_steps=4), probe=probe, config=config
    )
    gated.trainer.train()
    assert [(rec.decision, rec.override) for rec in gated.gate.records] == [
        (GateDecision.ACCEPT, "gate_disabled")
    ] * 2
    assert all(rec.reason.startswith("would=") for rec in gated.gate.records)
    assert same(snapshot(gated.trainer.model), snapshot(plain.trainer.model))
    assert optim_state_equal(
        gated.run.hooked_optimizer.inner.state_dict(), plain.run.hooked_optimizer.inner.state_dict()
    )
    assert gated.gate.gate.ledger.count == 0  # the disabled arm spends a private ledger
