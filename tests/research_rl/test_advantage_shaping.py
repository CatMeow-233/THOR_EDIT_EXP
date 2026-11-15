"""Advantage shapers, the learning-signal source of truth, and refusing silent gradient loss.

* Shapers replace TRL's advantages before the signal flags are computed and logged. With shapers
  the flags come from the last shaper's ``signal`` or, if it returns ``None``, from "advantage
  exactly non-zero" (never from the reward rule, which only holds for TRL's own advantages).
* Advantages edited after the mixin computed the flags are detected per micro-batch.
* ``skip_zero_advantage_steps`` is refused when a loss term does not depend on the advantage
  (GRPO ``beta``, GRPO entropy bonus, MoE aux loss) or an update rule must act every step.

Integration tests (``@pytest.mark.integration``) are short ``build_run`` runs on the tiny model.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
from datasets import Dataset
from transformers import TrainerCallback

from research_rl.core.config import DataSpec, ExperimentConfig, ModelSpec, OptimSpec, TrainerSpec
from research_rl.core.registry import ADVANTAGE_SHAPERS
from research_rl.models.model_interface import snapshot
from research_rl.models.tiny import build_tiny_llama
from research_rl.optim import UpdateRule, unwrap_hooked
from research_rl.trainers import build_run
from research_rl.trainers.grpo_adapter import GRPOAdapter, ResearchGRPOTrainer
from research_rl.trainers.rl_common import (
    ADVANTAGES_STAMP_KEY,
    SIGNAL_KEY,
    AdvantageShaper,
    ShapedAdvantages,
)

MODIFIED_WARNING = "differ from the ones its learning-signal flags were computed from"


@pytest.fixture(autouse=True)
def _single_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


# ================================================================================== helpers
def constant_reward(completions, **kwargs):
    return [1.0] * len(completions)


def varied_reward(completion_ids, **kwargs):
    return [float(len(c)) + float(c[0] % 3) for c in completion_ids]


def dataset() -> Dataset:
    return Dataset.from_list([{"prompt": f"{a}+{b}="} for a in range(4) for b in range(4)])


def rl_args(algorithm: str, **overrides) -> dict:
    args = {
        "per_device_train_batch_size": 4,
        "num_generations": 4,
        "max_completion_length": 4,
        "max_steps": 2,
        "learning_rate": 1e-2,
        "weight_decay": 0.1,
        "logging_steps": 1,
    }
    if algorithm == "rloo":
        args["beta"] = 0.0  # the in-memory tiny model cannot be re-loaded as a reference model
    args.update(overrides)
    return args


def make_config(tmp_path, algorithm: str, args: dict, *, skip: bool, run_name: str) -> ExperimentConfig:
    return ExperimentConfig(
        study_id="shaping_test",
        model=ModelSpec(name_or_path="tiny-llama-in-memory"),
        data=DataSpec(name="synthetic"),
        trainer=TrainerSpec(algorithm=algorithm, args=args),
        seed=0,
        output_dir=str(tmp_path),
        run_name=run_name,
        optim=OptimSpec(skip_zero_advantage_steps=skip),
    )


def make_run(
    tokenizer,
    tmp_path,
    *,
    algorithm: str = "grpo",
    reward=constant_reward,
    shapers=(),
    skip: bool = True,
    run_name: str = "run",
    adapter=None,
    model=None,
    callbacks=(),
    update_rules=(),
    eval_dataset=None,
    **overrides,
):
    exp = make_config(tmp_path, algorithm, rl_args(algorithm, **overrides), skip=skip, run_name=run_name)
    kwargs = dict(
        train_dataset=dataset(),
        model=model if model is not None else build_tiny_llama(len(tokenizer), seed=0),
        processing_class=tokenizer,
        reward_funcs=reward,
        callbacks=list(callbacks),
        update_rules=list(update_rules),
    )
    if eval_dataset is not None:
        kwargs["eval_dataset"] = eval_dataset
    if shapers:  # passed as given: a single shaper / a bare name is accepted too
        kwargs["advantage_shapers"] = shapers
    if adapter is not None:
        return adapter.build(exp, **kwargs)
    return build_run(exp, **kwargs)


def adam_state(hooked) -> list[dict]:
    out = []
    for group in hooked.param_groups:
        for p in group["params"]:
            st = hooked.state.get(p, {})
            out.append({k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in st.items()})
    return out


def same_adam_state(a: list[dict], b: list[dict]) -> bool:
    if len(a) != len(b):
        return False
    for sa, sb in zip(a, b):
        if sa.keys() != sb.keys():
            return False
        for k in sa:
            va, vb = sa[k], sb[k]
            if isinstance(va, torch.Tensor) != isinstance(vb, torch.Tensor):
                return False
            if isinstance(va, torch.Tensor) and not torch.equal(va, vb):
                return False
            if not isinstance(va, torch.Tensor) and va != vb:
                return False
    return True


class StepTrace(TrainerCallback):
    """Parameters and inner (Adam) optimizer state after every optimizer step."""

    def __init__(self):
        self.params: list[dict[str, torch.Tensor]] = []
        self.optim: list[list[dict]] = []

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        self.params.append(snapshot(model))
        self.optim.append(adam_state(unwrap_hooked(optimizer)))


def same_params(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[n], b[n]) for n in a)


def steps(run) -> tuple[int, int, int]:
    opt = run.hooked_optimizer
    return opt.nominal_steps, opt.applied_updates, opt.skipped_steps


# ---- shapers ---------------------------------------------------------------------------------
class AddBonus(AdvantageShaper):
    """Adds a constant to every advantage (e.g. a per-sample credit) and declares nothing."""

    name = "add_bonus"

    def __init__(self, bonus: float = 0.5) -> None:
        self.bonus = bonus

    def __call__(self, output, trainer):
        return ShapedAdvantages(output["advantages"] + self.bonus)


class ApplyThenDeclareZero(AdvantageShaper):
    """Non-zero advantages always; ``signal`` all False from the second generation batch on."""

    name = "apply_then_declare_zero"

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, output, trainer):
        self.calls += 1
        adv = output["advantages"] + 0.5
        if self.calls == 1:
            return ShapedAdvantages(adv)
        return ShapedAdvantages(adv, signal=torch.zeros(adv.shape[0], dtype=torch.bool))


class Fixed(AdvantageShaper):
    """Zeros everywhere, optionally ``value`` on sample 0; signal left to the framework."""

    name = "fixed"

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self, output, trainer):
        adv = torch.zeros_like(output["advantages"])
        adv[0] = self.value
        return ShapedAdvantages(adv)


class PerToken(AdvantageShaper):
    """Token-level credit: 0.3 on every completion token, (B, T)."""

    name = "per_token"

    def __call__(self, output, trainer):
        mask = output["completion_mask"]
        return ShapedAdvantages(0.3 * mask.to(output["advantages"].dtype))


class KeepTRLSignal(AdvantageShaper):
    """A positive rescaling keeps TRL's zeros, so it may reuse the reward-based flags."""

    name = "keep_trl_signal"

    def __call__(self, output, trainer):
        assert trainer.trl_advantage_signal is not None
        return ShapedAdvantages(2.0 * output["advantages"], signal=trainer.trl_advantage_signal)


class Returning(AdvantageShaper):
    name = "returning"

    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, output, trainer):
        return self.fn(output)


# ================================================================================== unit tests
@pytest.fixture
def grpo_trainer(tokenizer, tmp_path):
    return make_run(tokenizer, tmp_path, skip=False).trainer


def fake_output(b: int = 4, t: int = 3) -> dict:
    return {
        "prompt_ids": torch.zeros(b, 2, dtype=torch.long),
        "completion_ids": torch.ones(b, t, dtype=torch.long),
        "completion_mask": torch.ones(b, t, dtype=torch.long),
        "advantages": torch.zeros(b),
    }


@pytest.mark.parametrize(
    ("fn", "error", "match"),
    [
        (lambda o: ShapedAdvantages(torch.zeros(3)), ValueError, r"shape \(3,\)"),
        (lambda o: ShapedAdvantages(torch.zeros(4, 5)), ValueError, r"shape \(4, 5\).*\(4,\) or \(4, 3\)"),
        (lambda o: ShapedAdvantages(torch.zeros(4, 3, 1)), ValueError, "shape"),
        (lambda o: ShapedAdvantages(torch.zeros(4, dtype=torch.float64)), ValueError, "dtype torch.float64"),
        (lambda o: ShapedAdvantages(torch.zeros(4, device="meta")), ValueError, "on meta"),
        (lambda o: ShapedAdvantages(torch.tensor([0.0, float("nan"), 0.0, 0.0])), ValueError, "non-finite"),
        (lambda o: ShapedAdvantages([0.0] * 4), TypeError, "torch.Tensor"),  # type: ignore[arg-type]
        (lambda o: o["advantages"], TypeError, "ShapedAdvantages"),
        (lambda o: ShapedAdvantages(torch.zeros(4), signal=torch.zeros(4)), ValueError, "bool tensor"),
        (
            lambda o: ShapedAdvantages(torch.zeros(4), signal=torch.zeros(3, dtype=torch.bool)),
            ValueError,
            "bool",
        ),
    ],
)
def test_shaper_output_is_validated(grpo_trainer, fn, error, match):
    grpo_trainer.advantage_shapers = [Returning(fn)]
    with pytest.raises(error, match=match):
        grpo_trainer._rr_apply_shapers(fake_output())


def test_grpo_accepts_per_token_and_detaches(grpo_trainer):
    w = torch.ones(4, 3, requires_grad=True)
    grpo_trainer.advantage_shapers = [Returning(lambda o: ShapedAdvantages(w * 0.5))]
    out = fake_output()
    assert grpo_trainer._rr_apply_shapers(out) is None
    assert out["advantages"].shape == (4, 3) and not out["advantages"].requires_grad


def test_shapers_run_in_order_and_last_signal_wins(grpo_trainer):
    seen = []

    def first(o):
        seen.append(o["advantages"].clone())
        return ShapedAdvantages(o["advantages"] + 1.0, signal=torch.ones(4, dtype=torch.bool))

    def second(o):
        seen.append(o["advantages"].clone())
        return ShapedAdvantages(o["advantages"] * 3.0)

    grpo_trainer.advantage_shapers = [Returning(first), Returning(second)]
    out = fake_output()
    assert grpo_trainer._rr_apply_shapers(out) is None  # the last shaper declared nothing
    assert torch.equal(seen[1], torch.ones(4)) and torch.equal(out["advantages"], torch.full((4,), 3.0))


def test_rloo_rejects_per_token_advantages(tokenizer, tmp_path):
    trainer = make_run(tokenizer, tmp_path, algorithm="rloo", skip=False).trainer
    trainer.advantage_shapers = [Returning(lambda o: ShapedAdvantages(torch.zeros(4, 3)))]
    with pytest.raises(ValueError, match=r"ResearchRLOOTrainer loss accepts \(4,\)"):
        trainer._rr_apply_shapers(fake_output())


def test_shapers_resolve_from_registry(tokenizer, tmp_path):
    ADVANTAGE_SHAPERS.add("test_add_bonus", AddBonus, overwrite=True)
    trainer = make_run(tokenizer, tmp_path, skip=False, shapers=["test_add_bonus", AddBonus(0.1)]).trainer
    assert [type(s) for s in trainer.advantage_shapers] == [AddBonus, AddBonus]
    assert trainer.advantage_shapers[0].bonus == 0.5 and trainer.advantage_shapers[1].bonus == 0.1


@pytest.mark.parametrize("given", ["test_add_bonus", AddBonus(0.5)])
def test_single_shaper_or_name_is_one_shaper(tokenizer, tmp_path, given):
    """A bare registry name must not be iterated per character; a bare shaper is wrapped."""
    ADVANTAGE_SHAPERS.add("test_add_bonus", AddBonus, overwrite=True)
    trainer = make_run(tokenizer, tmp_path, skip=False, shapers=given).trainer
    assert len(trainer.advantage_shapers) == 1 and isinstance(trainer.advantage_shapers[0], AddBonus)


@pytest.mark.parametrize("given", [[AddBonus], AddBonus, (AddBonus(), AddBonus)])
def test_shaper_class_is_rejected_at_construction(tokenizer, tmp_path, given):
    with pytest.raises(TypeError, match="AddBonus is a class; pass an instance"):
        make_run(tokenizer, tmp_path, skip=False, shapers=given)


def test_non_sequence_shapers_are_rejected(tokenizer, tmp_path):
    with pytest.raises(TypeError, match="sequence of AdvantageShaper"):
        make_run(tokenizer, tmp_path, skip=False, shapers=lambda o, t: None)


def test_advantage_stamp_travels_through_trl_shuffle_and_split():
    from trl.trainer.utils import shuffle_sequence_dict, split_tensor_dict

    for adv in (torch.arange(6.0), torch.arange(18.0).view(6, 3)):
        batch: dict[str, Any] = {"prompt_ids": torch.arange(6)[:, None], "advantages": adv}
        batch[SIGNAL_KEY] = torch.arange(6) % 2 == 0
        batch[ADVANTAGES_STAMP_KEY] = adv.clone()
        parts: list[Any] = split_tensor_dict(shuffle_sequence_dict(batch), 3)  # type: ignore[arg-type]
        for part in parts:
            assert torch.equal(part["advantages"], part[ADVANTAGES_STAMP_KEY])
            assert torch.equal(part[SIGNAL_KEY], part["prompt_ids"][:, 0] % 2 == 0)


def test_microbatch_flags_detect_edits(grpo_trainer):
    adv = torch.tensor([0.0, 0.5, 0.0, 0.0])
    inputs = {
        "advantages": adv.clone(),
        "completion_mask": torch.ones(4, 2),
        SIGNAL_KEY: torch.zeros(4, dtype=torch.bool),
        ADVANTAGES_STAMP_KEY: adv.clone(),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert not grpo_trainer._rr_microbatch_flags(inputs).any()  # unchanged: the stamped flags
        inputs["advantages"] = adv * 2  # edited: exact non-zero test
        assert grpo_trainer._rr_microbatch_flags(inputs).tolist() == [False, True, False, False]
        inputs["advantages"] = adv.double()  # a dtype change is an edit too
        assert grpo_trainer._rr_microbatch_flags(inputs).tolist() == [False, True, False, False]
    assert sum(MODIFIED_WARNING in str(w.message) for w in caught) == 1  # warned once


# ================================================================================== refusals
def test_grpo_beta_with_skip_raises(tokenizer, tmp_path, tiny_model_dir):
    with pytest.raises(ValueError, match=r"GRPO beta=0\.04 adds a KL loss term"):
        make_run(tokenizer, tmp_path, model=tiny_model_dir, beta=0.04)


@pytest.mark.parametrize("entropy", [{"entropy_coef": 0.01}, {"use_adaptive_entropy": True}])
def test_grpo_entropy_bonus_with_skip_raises(tokenizer, tmp_path, entropy):
    with pytest.raises(ValueError, match="entropy bonus"):
        make_run(tokenizer, tmp_path, **entropy)
    make_run(tokenizer, tmp_path, skip=False, **entropy)  # fine without skipping


@pytest.mark.parametrize("algorithm", ["grpo", "rloo"])
@pytest.mark.parametrize(
    "config", [{"output_router_logits": True}, {"output_router_logits": False, "router_aux_loss_coef": 0.01}]
)
def test_moe_aux_loss_with_skip_raises(tokenizer, tmp_path, algorithm, config):
    """Both the task's rule (flag True) and TRL's own gate (flag declared, coefficient non-zero)."""
    model = build_tiny_llama(len(tokenizer), seed=0)
    for key, value in config.items():
        setattr(model.config, key, value)
    with pytest.raises(ValueError, match="router auxiliary loss"):
        make_run(tokenizer, tmp_path, algorithm=algorithm, model=model)


def test_aux_loss_disabled_by_coefficient_is_allowed(tokenizer, tmp_path):
    model = build_tiny_llama(len(tokenizer), seed=0)
    model.config.output_router_logits = False
    model.config.router_aux_loss_coef = 0.01
    trainer = make_run(tokenizer, tmp_path, model=model, router_aux_loss_coef=0.0).trainer
    assert not trainer.aux_loss_enabled


class Pull(UpdateRule):
    name = "drift_pull"
    needs_every_step = True


@pytest.mark.integration
def test_every_step_rule_with_skip_raises_at_train_begin(tokenizer, tmp_path):
    run = make_run(tokenizer, tmp_path, update_rules=[Pull()])  # construction is fine
    with pytest.raises(ValueError, match=r"\['drift_pull'\] have needs_every_step=True"):
        run.trainer.train()
    assert run.trainer.state.global_step == 0
    # without skipping the rule simply runs
    run = make_run(tokenizer, tmp_path, update_rules=[Pull()], skip=False, max_steps=1, run_name="noskip")
    run.train()
    assert steps(run) == (1, 1, 0)


# ================================================================================== integration
@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["grpo", "rloo"])
def test_shaper_gives_signal_to_constant_reward_group(tokenizer, tmp_path, algorithm):
    """Constant reward: TRL's advantages are exactly 0 and the reward rule says "no signal".
    A shaped non-zero advantage must win: the steps are applied and the log holds the shaped values."""
    model = build_tiny_llama(len(tokenizer), seed=0)
    before = snapshot(model)
    run = make_run(tokenizer, tmp_path, algorithm=algorithm, model=model, shapers=[AddBonus(0.5)])
    record = run.train()
    assert steps(run) == (2, 2, 0) and record["steps"]["skipped_steps"] == 0
    assert not same_params(before, snapshot(run.trainer.model))
    buf = run.trainer.trajectory_buffer
    assert len(buf) == 2
    for batch in buf:
        assert torch.equal(batch.advantages, torch.full((4,), 0.5))  # shaped, not TRL's zeros
        assert batch.has_nonzero_advantage and batch.frac_zero_advantage == 0.0
    assert buf.counters()["zero_signal_windows"] == 0


@pytest.mark.integration
def test_declared_zero_signal_skips_bit_identically(tokenizer, tmp_path):
    trace = StepTrace()
    run = make_run(
        tokenizer,
        tmp_path,
        shapers=[ApplyThenDeclareZero()],
        callbacks=[trace],
        reward=varied_reward,
        max_steps=3,
    )
    run.train()
    assert steps(run) == (3, 1, 2)
    # step 1 applied (non-vacuous: Adam state exists), steps 2 and 3 are exact no-ops
    assert trace.optim[0] and all(trace.optim[0])
    for k in (1, 2):
        assert same_params(trace.params[0], trace.params[k]), k
        assert same_adam_state(trace.optim[0], trace.optim[k]), k
    buf = run.trainer.trajectory_buffer
    assert [b.has_nonzero_advantage for b in buf] == [True, False, False]
    assert all(bool((b.advantages != 0).all()) for b in buf)  # skipped despite non-zero advantages


@pytest.mark.integration
@pytest.mark.parametrize(
    ("value", "applied"), [(0.0, 0), (1e-6, 2), (1e-30, 2), (torch.finfo(torch.float32).tiny, 2)]
)
def test_none_signal_means_exact_non_zero(tokenizer, tmp_path, value, applied):
    """TRL's advantages are non-zero here (varied reward); the shaper's are exactly 0 or carry one
    tiny round-off-sized value. ``signal=None`` must skip only the exact zeros."""
    model = build_tiny_llama(len(tokenizer), seed=0)
    before = snapshot(model)
    run = make_run(tokenizer, tmp_path, model=model, reward=varied_reward, shapers=[Fixed(value)])
    run.train()
    assert steps(run) == (2, applied, 2 - applied)
    assert same_params(before, snapshot(run.trainer.model)) is (applied == 0)
    frac = [b.frac_zero_advantage for b in run.trainer.trajectory_buffer]
    assert frac == ([1.0, 1.0] if applied == 0 else [0.75, 0.75])


@pytest.mark.integration
def test_shaper_can_keep_trl_reward_signal(tokenizer, tmp_path):
    run = make_run(tokenizer, tmp_path, shapers=[KeepTRLSignal()])  # constant reward
    run.train()
    assert steps(run) == (2, 0, 2)


@pytest.mark.integration
def test_per_token_advantages_train_and_are_logged(tokenizer, tmp_path):
    """(B, T) advantages through TRL's real shuffle and split (steps_per_generation=2): no false
    'modified' detection, the steps are applied and the log keeps the token-level tensor."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = make_run(
            tokenizer, tmp_path, shapers=[PerToken()], per_device_train_batch_size=4, steps_per_generation=2
        )
        run.train()
    assert not [w for w in caught if MODIFIED_WARNING in str(w.message)]
    assert steps(run) == (2, 2, 0)
    (batch,) = list(run.trainer.trajectory_buffer)  # one generation feeds both steps
    assert batch.advantages.shape == batch.completion_mask.shape == (8, 4)
    torch.testing.assert_close(batch.advantages, 0.3 * batch.completion_mask.float())


@pytest.mark.integration
@pytest.mark.parametrize("shapers", [(), (AddBonus(0.0),)])
def test_no_false_modification_detection_with_real_shuffle(tokenizer, tmp_path, shapers):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = make_run(
            tokenizer, tmp_path, reward=varied_reward, shapers=shapers, steps_per_generation=2, max_steps=2
        )
        run.train()
    assert not [w for w in caught if MODIFIED_WARNING in str(w.message)]
    assert steps(run)[0] == 2


class PostHocTrainer(ResearchGRPOTrainer):
    """A user subclass that edits the advantages *after* the mixin computed the flags."""

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        output["advantages"] = output["advantages"] + 0.5
        return output


class InPlacePostHocTrainer(ResearchGRPOTrainer):
    """The same edit in place: a stamp that aliases the advantages would change with them."""

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        output["advantages"].add_(0.5)
        return output


class PostHocAdapter(GRPOAdapter):
    research_trainer_cls = PostHocTrainer


class InPlacePostHocAdapter(GRPOAdapter):
    research_trainer_cls = InPlacePostHocTrainer


@pytest.mark.integration
@pytest.mark.parametrize("adapter_cls", [PostHocAdapter, InPlacePostHocAdapter])
def test_post_hoc_modification_is_detected(tokenizer, tmp_path, adapter_cls):
    """Constant reward: the stamped flags say "no signal", but the trained advantages are 0.5."""
    model = build_tiny_llama(len(tokenizer), seed=0)
    before = snapshot(model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = make_run(tokenizer, tmp_path, model=model, adapter=adapter_cls(), max_steps=3)
        run.train()
    assert sum(MODIFIED_WARNING in str(w.message) for w in caught) == 1  # once, not per micro-batch
    assert steps(run) == (3, 3, 0)
    assert not same_params(before, snapshot(run.trainer.model))


class EditBufferedInputs(TrainerCallback):
    """Edits TRL's buffered (already split) micro-batches between optimizer steps."""

    def __init__(self) -> None:
        self.trainer: Any = None
        self.edited = 0

    def on_step_begin(self, args, state, control, **kwargs):
        buffered = getattr(self.trainer, "_buffered_inputs", None)
        if buffered is not None:
            buffered[self.trainer._step % args.steps_per_generation]["advantages"].add_(0.5)
            self.edited += 1


@pytest.mark.integration
def test_callback_editing_buffered_inputs_is_detected(tokenizer, tmp_path):
    """Constant reward, one generation per two steps: step 1 has no signal and is skipped; the
    callback edits the second micro-batch before step 2, which must then be applied."""
    edit = EditBufferedInputs()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = make_run(tokenizer, tmp_path, callbacks=[edit], steps_per_generation=2, max_steps=2)
        edit.trainer = run.trainer
        run.train()
    assert edit.edited == 1  # the buffer exists only after the first generation
    assert sum(MODIFIED_WARNING in str(w.message) for w in caught) == 1
    assert steps(run) == (2, 1, 1)


# ---- empty completion masks: no sample has signal, whatever the shaper says --------------------
class DeclareAllSignal(AdvantageShaper):
    name = "declare_all_signal"

    def __call__(self, output, trainer):
        adv = output["advantages"] + 0.5
        return ShapedAdvantages(adv, signal=torch.ones(adv.shape[0], dtype=torch.bool))


@pytest.mark.integration
@pytest.mark.parametrize(
    ("shapers", "adapter_cls"),
    [((DeclareAllSignal(),), None), ((AddBonus(0.5),), None), ((), PostHocAdapter)],
    ids=["declared_signal", "none_signal", "post_hoc_edit"],
)
def test_empty_completion_masks_are_never_signal(tokenizer, tmp_path, shapers, adapter_cls):
    """``mask_truncated_completions`` with ``max_completion_length=2`` masks every completion (the
    tiny model does not emit EOS that early). Non-zero advantages or a declared signal must not
    count: the steps carry no advantage gradient and are skipped, parameters bit-identical."""
    model = build_tiny_llama(len(tokenizer), seed=0)
    before = snapshot(model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = make_run(
            tokenizer,
            tmp_path,
            model=model,
            shapers=shapers,
            adapter=adapter_cls() if adapter_cls is not None else None,
            mask_truncated_completions=True,
            max_completion_length=2,
        )
        run.train()
    buf = run.trainer.trajectory_buffer
    assert len(buf) == 2
    for batch in buf:  # not vacuous: every mask is empty
        assert int(batch.completion_mask.sum()) == 0
        if shapers:
            assert bool((batch.advantages != 0).all())
    if adapter_cls is not None:  # the edit was seen and the fallback flags still honour the mask
        assert sum(MODIFIED_WARNING in str(w.message) for w in caught) == 1
    assert steps(run) == (2, 0, 2)
    assert same_params(before, snapshot(run.trainer.model))


# ---- eval batches and the transient trl_advantage_signal --------------------------------------
class Recording(AdvantageShaper):
    name = "recording"

    def __init__(self) -> None:
        self.calls: list[tuple[bool, bool]] = []  # (model.training, trl_advantage_signal set)

    def __call__(self, output, trainer):
        self.calls.append((bool(trainer.model.training), trainer.trl_advantage_signal is not None))
        return ShapedAdvantages(output["advantages"] + 0.5)


@pytest.mark.integration
def test_shapers_run_on_eval_batches_and_signal_is_transient(tokenizer, tmp_path):
    shaper = Recording()
    run = make_run(
        tokenizer,
        tmp_path,
        shapers=[shaper],
        eval_dataset=dataset().select(range(4)),
        per_device_eval_batch_size=4,
        max_steps=1,
    )
    run.train()
    assert shaper.calls == [(True, True)]
    assert run.trainer.trl_advantage_signal is None  # reset after the shapers ran
    metrics = run.trainer.evaluate()
    assert shaper.calls[1:] and all(c == (False, True) for c in shaper.calls[1:])
    assert run.trainer.trl_advantage_signal is None
    assert metrics["eval_rollout/frac_zero_advantage"] == 0.0  # the shaped (eval) advantages
    assert len(run.trainer.trajectory_buffer) == 1  # eval batches are not logged in the buffer


# ---- shaper state across a checkpoint resume --------------------------------------------------
class StatefulApplyThenDeclareZero(ApplyThenDeclareZero):
    name = "stateful_apply_then_declare_zero"

    def state_dict(self) -> dict:
        return {"calls": self.calls}

    def load_state_dict(self, state) -> None:
        self.calls = int(state["calls"])


@pytest.mark.integration
def test_stateful_shaper_is_restored_on_resume(tokenizer, tmp_path):
    """Uninterrupted: step 1 applied, steps 2-4 declared signal-free. A resume from step 2 must
    continue the shaper's call count (a fresh shaper would apply step 3)."""
    common: dict[str, Any] = dict(
        reward=varied_reward,
        max_steps=4,
        lr_scheduler_type="constant",
        save_strategy="steps",
        save_steps=2,
        run_name="resume",
    )
    first = StatefulApplyThenDeclareZero()
    full = make_run(tokenizer, tmp_path, shapers=[first], **common)
    full.train()
    assert steps(full) == (4, 1, 3) and first.calls == 4
    key = "advantage_shaper/0/stateful_apply_then_declare_zero"
    assert full.trainer.research_components[key] is first
    full_params = snapshot(full.trainer.model)

    second = StatefulApplyThenDeclareZero()
    resumed = make_run(tokenizer, tmp_path, shapers=[second], **common)
    ckpt = resumed.config.run_dir / "trainer" / "checkpoint-2"
    resumed.train(resume_from_checkpoint=str(ckpt))
    assert second.calls == 4  # 2 restored + 2 after the resume
    assert steps(resumed) == (4, 1, 3)
    assert same_params(full_params, snapshot(resumed.trainer.model))


def test_stateless_shaper_is_not_a_component(grpo_trainer):
    assert not [k for k in grpo_trainer.research_components if k.startswith("advantage_shaper")]


@pytest.mark.integration
def test_rloo_beta_with_skip_is_allowed_and_skips_only_signal_free_windows(
    tokenizer, tmp_path, tiny_model_dir
):
    """RLOO folds ``beta * KL`` into the reward, i.e. into the advantage: no separate loss term."""
    run = make_run(
        tokenizer,
        tmp_path,
        algorithm="rloo",
        model=tiny_model_dir,
        reward=varied_reward,
        beta=0.05,
        max_steps=3,
    )
    assert run.trainer.beta == 0.05
    run.train()
    buf = run.trainer.trajectory_buffer
    has_signal = [b.has_nonzero_advantage for b in buf]
    assert len(has_signal) == 3 and any(has_signal)
    nominal, applied, skipped = steps(run)
    assert nominal == 3 and applied == sum(has_signal) and skipped == 3 - applied


@pytest.mark.integration
@pytest.mark.parametrize("perturb", [False, True], ids=["policy_is_ref", "policy_moved"])
def test_rloo_beta_constant_reward_skips_iff_kl_is_zero(tokenizer, tmp_path, tiny_model_dir, perturb):
    """Constant reward, ``beta=0.05``. While the policy equals the reference the KL penalty is 0,
    every advantage is (round-off) zero and every step is skipped bit-identically. A policy moved
    away from the reference makes the KL, and with it the advantage, differ per sample: the
    reward rule (KL rebuilt from TRL's log-probs) sees signal and the steps are applied."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir)  # TRL reloads the ref from this path
    if perturb:
        gen = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.05 * torch.randn(p.shape, generator=gen))
    before = snapshot(model)
    run = make_run(tokenizer, tmp_path, algorithm="rloo", model=model, beta=0.05, max_steps=2)
    assert run.trainer.ref_model is not None
    run.train()
    buf = run.trainer.trajectory_buffer
    if perturb:
        assert all(bool((b.advantages != 0).any()) for b in buf)
        assert [b.has_nonzero_advantage for b in buf] == [True, True]
        assert steps(run) == (2, 2, 0)
        assert not same_params(before, snapshot(run.trainer.model))
    else:
        assert [b.has_nonzero_advantage for b in buf] == [False, False]
        assert steps(run) == (2, 0, 2)
        assert same_params(before, snapshot(run.trainer.model))
