"""Response-level experts in RL rollouts: every forward of a row goes through that row's expert.

The routing is checked from the outside, with forward hooks and no knowledge of the
implementation:

* a pre-hook on the top-level model derives the expected expert of every row from the row's first
  real token (each expert's prompts start with their own letter), and keeps it for the cached
  decode steps of ``generate`` (and for gradient-checkpointing recomputation in backward);
* a hook on each expert's SwiGLU module finds, by exact comparison of hidden states, which rows of
  the enclosing ExpertFFN input that expert processed, and checks they all belong to it, while the
  ExpertFFN hooks check that every row was processed exactly once.

The experts are perturbed copies of the base FFN (``create_expert`` makes exact copies, under which
mis-routing would be numerically invisible).
"""

from __future__ import annotations

import io
from functools import partial
from typing import Any

import pytest
import torch
import trl.trainer.grpo_trainer as grpo_trainer_module
import trl.trainer.rloo_trainer as rloo_trainer_module
from datasets import Dataset
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer, RLOOConfig, RLOOTrainer

from research_rl.core.checkpoint import COMPONENTS_FILE
from research_rl.core.config import DataSpec, ExperimentConfig, ModelSpec, OptimSpec, TrainerSpec
from research_rl.models.expert_manager import ExpertError, ExpertFFN, ExpertManager
from research_rl.models.model_interface import snapshot
from research_rl.models.parameter_mask import ParameterMask
from research_rl.models.tiny import build_tiny_llama
from research_rl.optim.update_interface import MaskRule
from research_rl.rollout.policy_version import read_policy_version
from research_rl.rollout.trajectory import RolloutBatch, TrajectoryBuffer
from research_rl.trainers import build_run
from research_rl.trainers.grpo_adapter import ResearchGRPOTrainer
from research_rl.trainers.rl_common import EXPERT_KEY, ROLLOUT_COMPONENT
from research_rl.trainers.rloo_adapter import ResearchRLOOTrainer

# expert of a prompt, by its first character ("c" rows have no expert -> the active expert "math")
EXPERT_OF_CHAR = {"a": "math", "b": "code", "c": "math"}


@pytest.fixture(autouse=True)
def _single_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


# ================================================================================== helpers
def perturb(manager: ExpertManager, name: str, seed: int, scale: float = 0.05) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in manager.expert_parameters(name):
            p.add_(torch.randn(p.shape, generator=gen) * scale)


def attach_experts(model) -> ExpertManager:
    """Experts "math" (trainable, active) and "code" (frozen), both perturbed copies of the base FFN."""
    mgr = ExpertManager(model)
    mgr.attach()
    for name, seed in (("math", 1), ("code", 2)):
        mgr.create_expert(name)
        perturb(mgr, name, seed)
    mgr.set_trainable("math")
    return mgr


def experts_model(tokenizer):
    model = build_tiny_llama(len(tokenizer), seed=0)
    return model, attach_experts(model)


def mixed_dataset(n: int = 4) -> Dataset:
    rows: list[dict[str, Any]] = []
    for i in range(n):
        rows.append({"prompt": f"a{i}+1=", "expert": "math"})
        rows.append({"prompt": f"b{i}+2=", "expert": "code"})
        rows.append({"prompt": f"c{i}+3=", "expert": None})  # missing -> manager.active
    return Dataset.from_list(rows)


def digit_fraction(completions, **kwargs):
    return [sum(ch.isdigit() for ch in c) / max(len(c), 1) for c in completions]


def rl_config(algorithm: str, tmp_path, **overrides):
    """Generation batch of 8 rows (4 prompts x 2), micro-batches of 2, two of them per optimizer step.

    ``steps_per_generation (4) > gradient_accumulation_steps (2)``: TRL computes the old log-probs
    at generation time (GRPO; RLOO always does), in chunks of ``per_device_train_batch_size`` rows.
    """
    args: dict[str, Any] = {
        "output_dir": str(tmp_path / "out"),
        "per_device_train_batch_size": 2,
        "num_generations": 2,
        "gradient_accumulation_steps": 2,
        "steps_per_generation": 4,
        "max_completion_length": 6,
        "max_steps": 4,
        "learning_rate": 1e-2,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        "use_cpu": True,
        "bf16": False,
    }
    args.update(overrides)
    if algorithm == "grpo":
        return GRPOConfig(**args)
    args.setdefault("beta", 0.0)
    return RLOOConfig(**args)


class _Capture:
    """Records (micro-batch, loss-forward per-token logps) for micro-batches with policy lag 0."""

    lag0: list | None = None
    _cap_inputs: Any = None
    policy_version: int

    def compute_loss(self, model, inputs, *args, **kwargs):
        self._cap_inputs = inputs
        try:
            return super().compute_loss(model, inputs, *args, **kwargs)  # type: ignore[misc]
        finally:
            self._cap_inputs = None

    def _get_per_token_logps_and_entropies(self, model, *args, **kwargs):
        out = super()._get_per_token_logps_and_entropies(model, *args, **kwargs)  # type: ignore[misc]
        inputs = self._cap_inputs
        if (
            self.lag0 is not None
            and inputs is not None
            and torch.is_grad_enabled()
            and read_policy_version(inputs) == self.policy_version
        ):
            self.lag0.append((inputs, out[0].detach().clone()))
        return out


class RecordingGRPO(_Capture, ResearchGRPOTrainer):  # type: ignore[misc]  # TRL attribute refinements
    pass


class RecordingRLOO(_Capture, ResearchRLOOTrainer):  # type: ignore[misc]  # TRL attribute refinements
    pass


TRAINERS = {"grpo": RecordingGRPO, "rloo": RecordingRLOO}


def make_trainer(algorithm, tokenizer, tmp_path, model, mgr, dataset=None, trainer_kwargs=None, **overrides):
    return TRAINERS[algorithm](
        model=model,
        reward_funcs=digit_fraction,
        args=rl_config(algorithm, tmp_path, **overrides),
        train_dataset=mixed_dataset() if dataset is None else dataset,
        processing_class=tokenizer,
        expert_manager=mgr,
        **(trainer_kwargs or {}),
    )


class RoutingChecker:
    """Forward hooks verifying that every row of every forward went through its own expert."""

    def __init__(self, model, mgr: ExpertManager, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.rows: list[str] | None = None  # expected expert per row of the current top-level forward
        self.errors: list[str] = []
        self.events: list[tuple[str, int, bool]] = []  # (kind, rows, grad enabled)
        self.rows_by_kind: dict[tuple[str, str], int] = {}  # (kind, expert) -> rows processed
        self.mixed_kinds: set[str] = set()  # kinds of forwards whose rows belong to several experts
        self.kind = ""
        self._x: dict[int, torch.Tensor] = {}
        self._seen: dict[int, int] = {}
        self.handles = [model.register_forward_pre_hook(self._top, with_kwargs=True)]
        for ffn in mgr.ffn_modules().values():
            self.handles.append(ffn.register_forward_pre_hook(self._ffn_pre))
            self.handles.append(ffn.register_forward_hook(self._ffn_post))
            for name, expert in ffn.experts.items():
                self.handles.append(expert.register_forward_hook(partial(self._expert, ffn, name)))

    def close(self) -> None:
        for h in self.handles:
            h.remove()

    def _expected(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> list[str]:
        out = []
        for b in range(input_ids.shape[0]):
            real = (attention_mask[b] != 0) if attention_mask is not None else (input_ids[b] != 0)
            first = int(input_ids[b][real.nonzero()[0, 0]])
            out.append(EXPERT_OF_CHAR.get(self.tokenizer.convert_ids_to_tokens(first), "?"))
        return out

    def _top(self, module, args, kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        grad = torch.is_grad_enabled()
        if input_ids.shape[1] > 1:  # prefill of generate, or a log-prob / loss forward
            self.rows = self._expected(input_ids, kwargs.get("attention_mask"))
            self.kind = ("loss" if grad else "logp") if kwargs.get("fused_lm_head") else "prefill"
        else:  # cached decode step: same rows, same order
            self.kind = "decode"
            if self.rows is None or len(self.rows) != input_ids.shape[0]:
                self.errors.append(f"decode step of {input_ids.shape[0]} rows without a matching prefill")
        self.events.append((self.kind, int(input_ids.shape[0]), grad))
        if self.rows is not None and len(set(self.rows)) > 1:
            self.mixed_kinds.add(self.kind)

    def _ffn_pre(self, module, args):
        self._x[id(module)] = args[0].detach()
        self._seen[id(module)] = 0

    def _expert(self, ffn, name, module, args, output):
        full = self._x.get(id(ffn))
        if full is None or self.rows is None:
            return  # a direct call (e.g. a reference computation), not part of a model forward
        sub = args[0].detach()
        for j in range(sub.shape[0]):
            matches = [i for i in range(full.shape[0]) if torch.equal(full[i], sub[j])]
            owners = {self.rows[i] for i in matches}
            if owners != {name}:
                self.errors.append(
                    f"{self.kind}: layer {ffn.layer_index} expert {name!r} got rows of {owners}"
                )
        self._seen[id(ffn)] += sub.shape[0]
        key = (self.kind, name)
        self.rows_by_kind[key] = self.rows_by_kind.get(key, 0) + int(sub.shape[0])

    def _ffn_post(self, module, args, output):
        full = self._x.pop(id(module))
        if self.rows is not None and len(self.rows) != full.shape[0]:
            self.errors.append(
                f"{self.kind}: FFN input has {full.shape[0]} rows, the forward {len(self.rows)}"
            )
        if self._seen.pop(id(module)) != full.shape[0]:
            self.errors.append(
                f"{self.kind}: layer {module.layer_index}: not every row processed exactly once"
            )


def first_char(tokenizer, ids: torch.Tensor, mask: torch.Tensor) -> str:
    return tokenizer.convert_ids_to_tokens(int(ids[mask.bool()][0]))


def full_generation_batch(trainer) -> dict:
    parts = trainer._buffered_inputs
    out = {}
    for key, value in parts[0].items():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            out[key] = torch.cat([p[key] for p in parts])
        else:
            out[key] = value
    return out


def trainable_grads(trainer, batch) -> dict[str, torch.Tensor | None]:
    model = trainer.model
    model.zero_grad(set_to_none=True)
    with trainer.expert_routing(batch):  # gradient checkpointing (TRL default) recomputes in backward
        trainer.compute_loss(model, batch).backward()
    return {
        n: (None if p.grad is None else p.grad.clone())
        for n, p in model.named_parameters()
        if p.requires_grad
    }


# ================================================================================== unit level
def test_rows_context_uses_one_expert_or_routes_and_restores(tiny_model):
    mgr = attach_experts(tiny_model)
    with mgr.rows(["code", "code"]):
        assert (mgr.active, mgr.routing) == ("code", None)
    with mgr.rows(["code", "math"]):
        assert mgr.routing == ("code", "math")
    assert (mgr.active, mgr.routing) == ("math", None)
    with pytest.raises(RuntimeError, match="boom"), mgr.rows(["code", "base"]):
        raise RuntimeError("boom")
    assert (mgr.active, mgr.routing) == ("math", None)
    with pytest.raises(ExpertError):
        mgr.rows(["nope"]).__enter__()
    with pytest.raises(ExpertError):
        mgr.rows([]).__enter__()
    assert set(mgr.ffn_modules()) == set(mgr.layers)


def test_rollout_batch_expert_ids_and_buffer_counters_round_trip():
    def batch(ids):
        n = len(ids) if ids is not None else 2
        z = torch.zeros(n, 3, dtype=torch.long)
        return RolloutBatch.from_tensors(
            prompt_ids=z,
            prompt_mask=z + 1,
            completion_ids=z,
            completion_mask=z + 1,
            rewards=torch.zeros(n),
            advantages=torch.ones(n),
            policy_version=0,
            expert_ids=ids,
        )

    with pytest.raises(ValueError, match="expert_ids"):
        RolloutBatch(
            prompt_ids=torch.zeros(2, 1),
            prompt_mask=torch.zeros(2, 1),
            completion_ids=torch.zeros(2, 1),
            completion_mask=torch.zeros(2, 1),
            rewards=torch.zeros(2),
            advantages=torch.zeros(2),
            policy_version=0,
            expert_ids=("a",),
        )
    buf = TrajectoryBuffer()
    buf.append(batch(None))
    assert "expert_samples" not in buf.counters()  # no experts: counters unchanged
    buf.append(batch(["math", "code", "math"]))
    latest = buf.latest()
    assert latest is not None and latest.expert_ids == ("math", "code", "math")
    assert [t.expert for t in latest.trajectories()] == ["math", "code", "math"]
    assert [t.expert for t in batch(None).trajectories()] == [None, None]
    assert buf.counters()["expert_samples"] == {"math": 2, "code": 1}
    raw = io.BytesIO()
    torch.save(buf.state_dict(), raw)
    raw.seek(0)
    restored = TrajectoryBuffer()
    restored.load_state_dict(torch.load(raw, weights_only=True))
    assert restored.counters() == buf.counters()
    old = {k: v for k, v in buf.state_dict().items() if k != "expert_samples"}  # pre-expert checkpoint
    restored.load_state_dict(old)
    assert restored.expert_samples == {} and "expert_samples" not in restored.counters()


def test_checker_detects_misrouting(tokenizer, tiny_model):
    """Negative control: the hook checker fails when rows go through the wrong expert."""
    mgr = attach_experts(tiny_model)
    ids = torch.tensor([tokenizer("a1+1=")["input_ids"], tokenizer("b1+1=")["input_ids"]])
    checker = RoutingChecker(tiny_model, mgr, tokenizer)
    try:
        with torch.no_grad(), mgr.rows(["math", "code"]):
            tiny_model(input_ids=ids)
        assert checker.errors == []
        with torch.no_grad(), mgr.rows(["code", "math"]):
            tiny_model(input_ids=ids)
        assert checker.errors and all("got rows of" in e for e in checker.errors)
        checker.errors.clear()
        with torch.no_grad():  # no routing: both rows through the active expert
            tiny_model(input_ids=ids)
        assert checker.errors
    finally:
        checker.close()


# ================================================================================== rejected paths
@pytest.mark.parametrize("algorithm", ["grpo", "rloo"])
def test_use_vllm_with_experts_raises(tokenizer, tmp_path, algorithm):
    model, mgr = experts_model(tokenizer)
    with pytest.raises(ExpertError, match="use_vllm"):
        make_trainer(algorithm, tokenizer, tmp_path, model, mgr, use_vllm=True)
    assert (mgr.active, mgr.routing) == ("math", None)


def test_use_vllm_with_experts_raises_through_the_adapter(tokenizer, tmp_path):
    model, mgr = experts_model(tokenizer)
    exp = ExperimentConfig(
        study_id="experts",
        model=ModelSpec(name_or_path="tiny"),
        data=DataSpec(name="synthetic"),
        trainer=TrainerSpec(
            algorithm="grpo",
            args={"per_device_train_batch_size": 2, "num_generations": 2, "use_vllm": True},
        ),
        output_dir=str(tmp_path),
    )
    with pytest.raises(ExpertError, match="use_vllm"):
        build_run(
            exp,
            train_dataset=mixed_dataset(),
            model=model,
            processing_class=tokenizer,
            reward_funcs=digit_fraction,
            expert_manager=mgr,
            record=False,
        )


def test_invalid_expert_setups_raise(tokenizer, tmp_path):
    model, mgr = experts_model(tokenizer)
    with pytest.raises(ExpertError, match="sync_ref_model"):
        make_trainer("grpo", tokenizer, tmp_path, model, mgr, beta=0.1, sync_ref_model=True)
    with pytest.raises(ExpertError, match="remove_unused_columns"):
        make_trainer("grpo", tokenizer, tmp_path, model, mgr, remove_unused_columns=True)
    with pytest.raises(ExpertError, match="num_return_sequences"):
        make_trainer("grpo", tokenizer, tmp_path, model, mgr, generation_kwargs={"num_return_sequences": 2})
    with pytest.raises(ExpertError, match="num_beams"):
        make_trainer("rloo", tokenizer, tmp_path, model, mgr, generation_kwargs={"num_beams": 2})
    with pytest.raises(ExpertError, match="different model"):
        make_trainer("grpo", tokenizer, tmp_path, build_tiny_llama(len(tokenizer)), mgr)
    with pytest.raises(ExpertError, match="no experts attached"):
        make_trainer("grpo", tokenizer, tmp_path, model, ExpertManager(build_tiny_llama(len(tokenizer))))
    with pytest.raises(ExpertError, match="model instance"):
        ResearchGRPOTrainer(
            model="some/path",
            reward_funcs=digit_fraction,
            args=rl_config("grpo", tmp_path),
            train_dataset=mixed_dataset(),
            processing_class=tokenizer,
            expert_manager=mgr,
        )
    from peft import LoraConfig

    with pytest.raises(ExpertError, match="PEFT"):
        ResearchGRPOTrainer(
            model=model,
            reward_funcs=digit_fraction,
            args=rl_config("grpo", tmp_path),
            train_dataset=mixed_dataset(),
            processing_class=tokenizer,
            expert_manager=mgr,
            peft_config=LoraConfig(target_modules=["q_proj"]),
        )
    assert (mgr.active, mgr.routing) == ("math", None)


def test_mixed_batch_unsupported_forward_paths_raise(tokenizer, tmp_path):
    model, mgr = experts_model(tokenizer)
    trainer = make_trainer("grpo", tokenizer, tmp_path, model, mgr)
    rows = ("math", "code", "math")
    ids = torch.tensor([tokenizer("a1+1=")["input_ids"]] * 3)
    with trainer._rr_rows_scope(rows):
        with pytest.raises(ExpertError, match="subset"):  # e.g. the tool loop re-generating 2 of 3 rows
            trainer._generate_single_turn([[5, 6], [7, 8]], None, {}, 1)
        with pytest.raises(ExpertError, match="multimodal"):
            trainer._get_per_token_logps_and_entropies(
                model, ids, torch.ones_like(ids), 2, pixel_values=torch.zeros(3, 1)
            )
        with pytest.raises(ExpertError, match="3 rows"):
            trainer._get_per_token_logps_and_entropies(model, ids[:2], torch.ones_like(ids[:2]), 2)
        trainer.use_transformers_continuous_batching = True
        with pytest.raises(ExpertError, match="continuous-batching"):
            trainer._generate_single_turn(ids.tolist(), None, {}, 1)
        trainer.use_transformers_continuous_batching = False
        assert mgr.routing == rows
    assert (mgr.active, mgr.routing) == ("math", None)
    with pytest.raises(ExpertError, match="rr_expert"):
        trainer.compute_loss(model, {"prompt_ids": ids})
    # the reference model must be the vanilla base model
    with pytest.raises(ExpertError, match="different FFN"):
        trainer._rr_check_reference(build_tiny_llama(len(tokenizer), seed=1))
    with pytest.raises(ExpertError, match="shares the policy's"):
        trainer._rr_check_reference(model)
    other = build_tiny_llama(len(tokenizer), seed=0)
    ExpertManager(other).attach()
    with pytest.raises(ExpertError, match="has its own"):
        trainer._rr_check_reference(other)
    trainer._rr_check_reference(build_tiny_llama(len(tokenizer), seed=0))  # vanilla base: accepted


@pytest.mark.integration
def test_errors_inside_routed_regions_restore_the_selection(tokenizer, tmp_path):
    model, mgr = experts_model(tokenizer)
    bad = Dataset.from_list([{"prompt": "a1=", "expert": "math"}, {"prompt": "b1=", "expert": "nope"}] * 2)
    trainer = make_trainer("grpo", tokenizer, tmp_path, model, mgr, dataset=bad)
    with pytest.raises(ExpertError, match="'nope' is not an expert"):
        trainer.train()
    assert (mgr.active, mgr.routing) == ("math", None)

    calls = []

    def failing_reward(completions, **kwargs):
        calls.append(mgr.routing)  # rewards run inside the routed generation region
        raise RuntimeError("reward failed")

    trainer = ResearchGRPOTrainer(
        model=model,
        reward_funcs=failing_reward,
        args=rl_config("grpo", tmp_path),
        train_dataset=mixed_dataset(),
        processing_class=tokenizer,
        expert_manager=mgr,
    )
    with pytest.raises(RuntimeError, match="reward failed"):
        trainer.train()
    assert calls and calls[0] is not None and len(calls[0]) == 8
    assert (mgr.active, mgr.routing) == ("math", None) and trainer._rr_rows is None

    def rollout(prompts, trainer):
        raise AssertionError("not reached")

    trainer = ResearchGRPOTrainer(
        model=model,
        reward_funcs=digit_fraction,
        args=rl_config("grpo", tmp_path),
        train_dataset=mixed_dataset(),
        processing_class=tokenizer,
        expert_manager=mgr,
        rollout_func=rollout,
    )
    with pytest.raises(ExpertError, match="rollout_func"):
        trainer.train()
    assert (mgr.active, mgr.routing) == ("math", None)


# ================================================================================== integration
@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["grpo", "rloo"])
def test_mixed_experts_route_every_forward_record_and_train_only_their_expert(tokenizer, tmp_path, algorithm):
    model, mgr = experts_model(tokenizer)
    before = {n: mgr.expert_fingerprint(n) for n in mgr.expert_names}
    trainer = make_trainer(algorithm, tokenizer, tmp_path, model, mgr)
    trainer.lag0 = []
    checker = RoutingChecker(model, mgr, tokenizer)
    try:
        trainer.train()
    finally:
        checker.close()

    # ---- every row of every forward went through its own expert
    assert checker.errors == []
    kinds = {(kind, n, grad) for kind, n, grad in checker.events}
    assert ("prefill", 8, False) in kinds and ("decode", 8, False) in kinds  # generation of 8 rows
    assert ("logp", 2, False) in kinds  # old log-probs: TRL's chunks of per_device_train_batch_size rows
    assert sum(1 for e in checker.events if e == ("logp", 2, False)) == 2 * 4  # 2 generations x 4 chunks
    assert ("loss", 2, True) in kinds and all(n == 2 for kind, n, _ in checker.events if kind == "loss")
    for kind in ("prefill", "decode", "logp", "loss"):
        for name in ("math", "code"):
            assert checker.rows_by_kind.get((kind, name), 0) > 0, (kind, name)
        assert checker.rows_by_kind.get((kind, "base"), 0) == 0, kind

    # ---- the trajectory records each row's expert
    buf = trainer.trajectory_buffer
    assert len(buf) == 2
    counts: dict[str, int] = {}
    for batch in buf:
        expected = tuple(
            EXPERT_OF_CHAR[first_char(tokenizer, ids, mask)]
            for ids, mask in zip(batch.prompt_ids, batch.prompt_mask)
        )
        assert batch.expert_ids == expected
        for name in expected:
            counts[name] = counts.get(name, 0) + 1
    assert buf.counters()["expert_samples"] == counts and sum(counts.values()) == buf.total_samples
    assert set(counts) == {"math", "code"}
    for micro in trainer._buffered_inputs:  # the index travelled through TRL's shuffle and split
        names = [trainer.expert_vocab[int(i)] for i in micro[EXPERT_KEY]]
        chars = [first_char(tokenizer, i, m) for i, m in zip(micro["prompt_ids"], micro["prompt_mask"])]
        assert names == [EXPERT_OF_CHAR[c] for c in chars]
    logged = [row for row in trainer.state.log_history if "rollout/expert_frac/code" in row]
    assert len(logged) == 2  # one per generation batch (logging_steps=1)
    for row, batch in zip(logged, buf):
        for name in ("base", "math", "code"):
            assert row[f"rollout/expert_frac/{name}"] == pytest.approx(
                batch.expert_ids.count(name) / len(batch)
            )

    # ---- generation-time old log-probs == first loss forward of the same rows (ratio exactly 1)
    assert len(trainer.lag0) == 4  # 2 lag-0 micro-batches per generation
    for inputs, logps in trainer.lag0:
        mask = inputs["completion_mask"].bool()
        # shuffled micro-batches differ in composition from the chunks: allow float32 round-off
        # (bit-exact equality with an identity shuffle: test_log_prob_chunks_and_micro_batches_...)
        if algorithm == "grpo":
            new, old = logps[mask], inputs["old_per_token_logps"][mask]
        else:
            new, old = (logps * inputs["completion_mask"]).sum(1), inputs["old_logps"]
        torch.testing.assert_close(new, old, rtol=0, atol=1e-4)

    # ---- integrity: backbone, base and the frozen expert bit-unchanged; the trainable one trained
    mgr.assert_backbone_unchanged()
    assert mgr.expert_fingerprint("code") == before["code"]
    assert mgr.expert_fingerprint("base") == before["base"]
    assert mgr.expert_fingerprint("math") != before["math"]
    assert (mgr.active, mgr.routing) == ("math", None)
    assert mgr.trainable_parameter_names() == mgr.expert_parameter_names("math")

    # ---- the trainable expert receives gradient only from its own rows
    trainer.lag0 = None
    model.train()
    batch = full_generation_batch(trainer)
    is_math = torch.tensor([trainer.expert_vocab[int(i)] == "math" for i in batch[EXPERT_KEY]])
    assert is_math.any() and (~is_math).any()
    n = is_math.numel()
    batch["advantages"] = torch.linspace(-1.0, 1.0, n) + 0.05  # all non-zero
    g_full = trainable_grads(trainer, batch)
    assert all(g is not None for g in g_full.values())
    assert max(float(g.abs().max()) for g in g_full.values() if g is not None) > 0
    only_math = dict(batch, advantages=torch.where(is_math, batch["advantages"], 0.0))
    for name, g in trainable_grads(trainer, only_math).items():
        torch.testing.assert_close(g, g_full[name], rtol=1e-5, atol=1e-8, msg=name)
    no_math = dict(batch, advantages=torch.where(is_math, 0.0, batch["advantages"]))
    for name, g in trainable_grads(trainer, no_math).items():
        assert g is None or not g.any(), name
    assert (mgr.active, mgr.routing) == ("math", None)


@pytest.mark.integration
def test_mixed_experts_with_gradient_checkpointing_route_the_recomputation(tokenizer, tmp_path):
    finals = []
    for gc in (False, True):
        model, mgr = experts_model(tokenizer)
        trainer = make_trainer("grpo", tokenizer, tmp_path / str(gc), model, mgr, activation_checkpointing=gc)
        checker = RoutingChecker(model, mgr, tokenizer)
        try:
            trainer.train()
        finally:
            checker.close()
        assert checker.errors == []
        finals.append(snapshot(model))
        mgr.assert_backbone_unchanged()
    for name, value in finals[0].items():
        torch.testing.assert_close(finals[1][name], value, rtol=1e-5, atol=1e-7, msg=name)


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["grpo", "rloo"])
def test_mixed_experts_with_num_iterations_and_evaluation(tokenizer, tmp_path, algorithm):
    """Re-used generation batches (num_iterations=2) and the eval loop route every row as well."""
    model, mgr = experts_model(tokenizer)
    trainer = TRAINERS[algorithm](
        model=model,
        reward_funcs=digit_fraction,
        args=rl_config(
            algorithm,
            tmp_path,
            num_iterations=2,
            eval_strategy="steps",
            eval_steps=2,
            per_device_eval_batch_size=4,
        ),
        train_dataset=mixed_dataset(),
        eval_dataset=mixed_dataset(2),
        processing_class=tokenizer,
        expert_manager=mgr,
    )
    checker = RoutingChecker(model, mgr, tokenizer)
    try:
        trainer.train()
        metrics = trainer.evaluate()
    finally:
        checker.close()
    assert checker.errors == []
    assert ("prefill", 4, False) in checker.events and ("logp", 4, False) in checker.events  # eval batches
    assert "eval_rollout/expert_frac/code" in metrics
    assert (mgr.active, mgr.routing) == ("math", None)
    mgr.assert_backbone_unchanged()


@pytest.mark.integration
@pytest.mark.parametrize("checkpointing", [False, True])
def test_rows_of_frozen_experts_only_leave_the_trainable_expert_unchanged(tokenizer, tmp_path, checkpointing):
    """No row reaches a trainable parameter: the loss has no graph, grads stay None, nothing moves."""
    model, mgr = experts_model(tokenizer)
    before = snapshot(model)
    data = Dataset.from_list([{"prompt": f"b{i}+1=", "expert": "code"} for i in range(4)])
    # a HookedOptimizer rule on the trainable expert must cope with its grad=None
    rule = MaskRule(ParameterMask.all_trainable(model, mgr.expert_parameter_names("math")))
    trainer = make_trainer(
        "grpo",
        tokenizer,
        tmp_path,
        model,
        mgr,
        dataset=data,
        trainer_kwargs={"update_rules": [rule]},
        activation_checkpointing=checkpointing,
    )
    trainer.train()
    opt = trainer.hooked_optimizer
    assert opt is not None and opt.nominal_steps == 4
    after = snapshot(model)
    for name, value in before.items():
        assert torch.equal(after[name], value), name
    assert trainer.trajectory_buffer.counters()["expert_samples"] == {"code": 16}


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["grpo", "rloo"])
def test_single_expert_run_is_bit_identical_with_and_without_manager(tokenizer, tmp_path, algorithm):
    """No expert column: every row uses the active expert, exactly as plain ``activate`` training."""
    dataset = Dataset.from_list([{"prompt": f"{a}+{b}="} for a in range(4) for b in range(2)])
    finals, versions = [], []
    for with_manager in (False, True):
        model, mgr = experts_model(tokenizer)
        exp = ExperimentConfig(
            study_id="experts",
            model=ModelSpec(name_or_path="tiny"),
            data=DataSpec(name="synthetic"),
            trainer=TrainerSpec(
                algorithm=algorithm,
                args={
                    "per_device_train_batch_size": 2,
                    "num_generations": 2,
                    "gradient_accumulation_steps": 2,
                    "steps_per_generation": 4,
                    "max_completion_length": 6,
                    "max_steps": 4,
                    "learning_rate": 1e-2,
                    "save_strategy": "no",
                    **({"beta": 0.0} if algorithm == "rloo" else {}),
                },
            ),
            seed=0,
            output_dir=str(tmp_path / str(with_manager)),
            optim=OptimSpec(hooked=True),
        )
        extra = {"expert_manager": mgr} if with_manager else {}
        run = build_run(
            exp,
            train_dataset=dataset,
            model=model,
            processing_class=tokenizer,
            reward_funcs=digit_fraction,
            record=False,
            **extra,
        )
        run.train()
        finals.append(snapshot(model))
        versions.append([b.expert_ids for b in run.trainer.trajectory_buffer])
        assert (mgr.active, mgr.routing) == ("math", None)
    assert finals[0].keys() == finals[1].keys()
    for name in finals[0]:
        assert torch.equal(finals[0][name], finals[1][name]), name
    assert versions[0] == [None, None] and versions[1] == [("math",) * 8] * 2


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["grpo", "rloo"])
def test_kl_reference_is_the_base_model_for_every_row(tokenizer, tiny_model_dir, tmp_path, algorithm):
    """beta > 0: TRL loads a vanilla copy from ``_name_or_path``; its log-probs equal the base expert's."""
    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    mgr = attach_experts(model)
    trainer = make_trainer(algorithm, tokenizer, tmp_path, model, mgr, beta=0.1, max_steps=2)
    assert trainer.ref_model is not None
    assert not any(isinstance(m, ExpertFFN) for m in trainer.ref_model.modules())
    captured = []
    if algorithm == "rloo":  # RLOO keeps only the KL-shaped reward; capture the reference forward
        original = trainer._get_per_token_logps_and_entropies

        def spy(m, *args, **kwargs):
            out = original(m, *args, **kwargs)
            if m is trainer.ref_model:
                captured.append((args[0], args[1], out[0]))
            return out

        trainer._get_per_token_logps_and_entropies = spy
    trainer.train()
    if algorithm == "grpo":
        for micro in trainer._buffered_inputs:
            ids = torch.cat([micro["prompt_ids"], micro["completion_ids"]], 1)
            am = torch.cat([micro["prompt_mask"], micro["completion_mask"]], 1)
            captured.append((ids, am, micro["ref_per_token_logps"]))
    assert captured
    trl_logps = (GRPOTrainer if algorithm == "grpo" else RLOOTrainer)._get_per_token_logps_and_entropies
    for ids, am, ref_logps in captured:
        keep = ref_logps.shape[1]
        with torch.no_grad():
            with mgr.using("base"):
                base, _, _ = trl_logps(trainer, trainer.model, ids, am, keep)
            with mgr.using("code"):
                code, _, _ = trl_logps(trainer, trainer.model, ids, am, keep)
        mask = am[:, -keep:].bool()
        torch.testing.assert_close(ref_logps[mask], base[mask], rtol=0, atol=1e-6)
        assert (ref_logps[mask] - code[mask]).abs().max() > 1e-3
    mgr.assert_backbone_unchanged()


@pytest.mark.integration
@pytest.mark.parametrize("shuffle", ["identity", "random"])
@pytest.mark.parametrize("algorithm", ["grpo", "rloo"])
def test_log_prob_chunks_and_micro_batches_mixing_experts(
    tokenizer, tmp_path, monkeypatch, algorithm, shuffle
):
    """Chunks / micro-batches of 4 rows span two prompts, so single chunks mix experts.

    With TRL's shuffle replaced by the identity, micro-batch ``i`` holds exactly the rows of
    old-log-prob chunk ``i`` (same rows, same order, same routing groups): the two forwards must be
    bit-identical (ratio exactly 1). With the real shuffle, the batch composition differs and the
    expert sub-batches differ in size, which moves float32 results by a few ulp (observed <= 4e-6
    for RLOO's summed log-probs over 20 seeds); a misrouted chunk moves them by ~0.1-1.
    """
    if shuffle == "identity":
        monkeypatch.setattr(grpo_trainer_module, "shuffle_sequence_dict", lambda batch: batch)
        monkeypatch.setattr(rloo_trainer_module, "shuffle_sequence_dict", lambda batch: batch)
    model, mgr = experts_model(tokenizer)
    trainer = make_trainer(
        algorithm,
        tokenizer,
        tmp_path,
        model,
        mgr,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        steps_per_generation=2,  # > gradient_accumulation_steps: GRPO computes old log-probs
        max_steps=6,
    )
    trainer.lag0 = []
    checker = RoutingChecker(model, mgr, tokenizer)
    try:
        trainer.train()
    finally:
        checker.close()
    assert checker.errors == []
    assert {"prefill", "decode", "logp", "loss"} <= checker.mixed_kinds
    assert ("logp", 4, False) in checker.events and ("loss", 4, True) in checker.events
    assert len(trainer.lag0) == 3  # the first micro-batch of each of the 3 generations
    mixed = 0
    for inputs, logps in trainer.lag0:
        mixed += len(set(inputs[EXPERT_KEY].tolist())) > 1
        mask = inputs["completion_mask"].bool()
        if algorithm == "grpo":
            new, old = logps[mask], inputs["old_per_token_logps"][mask]
        else:
            new, old = (logps * inputs["completion_mask"]).sum(1), inputs["old_logps"]
        if shuffle == "identity":
            assert torch.equal(new, old)
        else:
            torch.testing.assert_close(new, old, rtol=0, atol=1e-4)
    assert mixed > 0
    mgr.assert_backbone_unchanged()


@pytest.mark.integration
def test_expert_counters_survive_checkpoint_resume(tokenizer, tmp_path):
    def run(name):
        model, mgr = experts_model(tokenizer)
        exp = ExperimentConfig(
            study_id="experts",
            model=ModelSpec(name_or_path="tiny"),
            data=DataSpec(name="synthetic"),
            trainer=TrainerSpec(
                algorithm="grpo",
                args={
                    "per_device_train_batch_size": 2,
                    "num_generations": 2,
                    "gradient_accumulation_steps": 2,
                    "steps_per_generation": 4,
                    "max_completion_length": 6,
                    "max_steps": 4,
                    "save_strategy": "steps",
                    "save_steps": 2,  # a generation boundary
                },
            ),
            seed=0,
            output_dir=str(tmp_path),
            run_name=name,
            optim=OptimSpec(hooked=True),
        )
        return build_run(
            exp,
            train_dataset=mixed_dataset(),
            model=model,
            processing_class=tokenizer,
            reward_funcs=digit_fraction,
            expert_manager=mgr,
            record=False,
        )

    first = run("a")
    first.train()
    ckpt = first.config.run_dir / "trainer" / "checkpoint-2"
    saved = torch.load(ckpt / COMPONENTS_FILE, weights_only=True)[ROLLOUT_COMPONENT]
    at_ckpt = first.trainer.trajectory_buffer  # first generation batch only at step 2
    assert sum(saved["expert_samples"].values()) == 8 == saved["total_samples"]
    resumed = run("a")
    resumed.train(resume_from_checkpoint=str(ckpt))
    buf = resumed.trainer.trajectory_buffer
    assert buf.total_samples == 16 == at_ckpt.total_samples
    latest = buf.latest()
    assert latest is not None and latest.expert_ids is not None
    expected = dict(saved["expert_samples"])
    for name in latest.expert_ids:
        expected[name] = expected.get(name, 0) + 1
    assert buf.expert_samples == expected
