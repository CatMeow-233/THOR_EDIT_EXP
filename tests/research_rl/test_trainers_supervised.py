"""SFT / DPO / distillation adapters, checked against independent plain-torch reference losses."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from trl import DPOConfig

import research_rl.trainers.distillation_adapter
import research_rl.trainers.dpo_adapter
import research_rl.trainers.sft_adapter  # explicit imports register the adapters
from research_rl.core.checkpoint import checkpoint_inventory, load_research_state
from research_rl.core.config import DataSpec, ExperimentConfig, ModelSpec, TrainerSpec
from research_rl.core.registry import TRAINERS
from research_rl.models.model_interface import count_parameters, fingerprint, snapshot
from research_rl.models.tiny import build_tiny_llama
from research_rl.optim import HookedOptimizer
from research_rl.trainers import build_run
from research_rl.trainers.distillation_adapter import DistillationAdapter
from research_rl.trainers.dpo_adapter import DPOAdapter
from research_rl.trainers.sft_adapter import SFTAdapter

REPO = Path(__file__).resolve().parents[2]

# Prompts/completions of different lengths so that right padding and prompt masking both matter.
SFT_ROWS = [
    {"prompt": "1+1=", "completion": "2"},
    {"prompt": "12+30=", "completion": "42"},
    {"prompt": "7*8=", "completion": "56"},
    {"prompt": "100-1=", "completion": "99"},
]
DPO_ROWS = [
    {"prompt": "2+2=", "chosen": "4", "rejected": "22"},
    {"prompt": "10*10=", "chosen": "100", "rejected": "1"},
    {"prompt": "9-3=", "chosen": "6", "rejected": "12"},
]


def make_exp(
    tmp_path: Path, algorithm: str, run_name: str = "run", seed: int = 0, **args
) -> ExperimentConfig:
    return ExperimentConfig(
        study_id="supervised-tests",
        model=ModelSpec(name_or_path="tiny-llama-offline"),
        data=DataSpec(name="toy-arithmetic"),
        trainer=TrainerSpec(algorithm=algorithm, args=args),
        seed=seed,
        output_dir=str(tmp_path),
        run_name=run_name,
    )


def ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def token_logps(model, seq: list[int]) -> torch.Tensor:
    """log p(seq[t] | seq[:t]) for t = 1..len-1 of one *unpadded* sequence (float64)."""
    logits = model(torch.tensor([seq])).logits[0, :-1].double()
    return torch.log_softmax(logits, -1).gather(-1, torch.tensor(seq[1:])[:, None])[:, 0]


# ---------------------------------------------------------------------------------------------- registry


def test_registry_builds_adapters_and_labels_them_non_rl():
    for name, cls in (("sft", SFTAdapter), ("dpo", DPOAdapter), ("distillation", DistillationAdapter)):
        adapter = TRAINERS.build(name)
        assert type(adapter) is cls and adapter.algorithm == name
        assert adapter.is_reinforcement_learning is False
    assert "offline preference optimisation" in (research_rl.trainers.dpo_adapter.__doc__ or "")
    assert "not reinforcement learning" in (research_rl.trainers.distillation_adapter.__doc__ or "")
    # TRL's default DPO loss in the installed version is what test_dpo_loss_matches_reference assumes.
    assert DPOConfig(output_dir="unused", report_to=[], use_cpu=True).loss_type == ["sigmoid"]


@pytest.mark.parametrize("name", ["sft", "dpo", "distillation"])
def test_precision_is_fp32_unless_bf16_requested(tmp_path, name):
    """TRL's own default would be bf16=True; the adapters require an explicit request."""
    adapter = TRAINERS.build(name)
    assert adapter.build_trl_config(make_exp(tmp_path, name)).bf16 is False
    assert adapter.build_trl_config(make_exp(tmp_path, name, bf16=True)).bf16 is True


# ---------------------------------------------------------------------------------------------- SFT


@pytest.mark.integration
def test_sft_overfits_small_data_and_writes_record(tokenizer, tmp_path):
    steps = 40
    exp = make_exp(
        tmp_path,
        "sft",
        max_steps=steps,
        per_device_train_batch_size=4,
        learning_rate=1e-2,
        lr_scheduler_type="constant",
        warmup_steps=0,
    )
    model = build_tiny_llama(len(tokenizer), seed=0)
    data = Dataset.from_list(SFT_ROWS)
    init_fp = fingerprint(model)
    run = build_run(exp, train_dataset=data, model=model, processing_class=tokenizer)
    record = run.train()

    losses = run.recorder.history("loss")
    assert len(losses) == steps
    assert losses[0] > 2.0  # random init: close to log(vocab)
    assert losses[-1] < 0.2 * losses[0], (losses[0], losses[-1])

    on_disk = json.loads((exp.run_dir / "record.json").read_text())
    assert on_disk == record
    assert record["study_id"] == "supervised-tests" and record["seed"] == 0 and record["algorithm"] == "sft"
    assert record["status"] == "completed"
    assert record["model"]["class"] == "LlamaForCausalLM"
    assert record["model"]["weights_sha256"] == init_fp  # identity of the *initial* weights
    assert record["dataset"]["num_rows"] == 4 and record["dataset"]["name"] == "toy-arithmetic"
    assert record["dataset"]["content_sha256"]
    assert record["trainable_parameters"] == count_parameters(model, trainable_only=True)
    assert record["steps"]["applied_updates"] == steps == record["steps"]["global_step"]
    assert record["steps"]["nominal_optimizer_steps"] == steps
    # Every step sees all 4 rows unpadded-counted: prompt + completion + EOS tokens.
    tokens_per_step = sum(len(ids(tokenizer, r["prompt"] + r["completion"])) + 1 for r in SFT_ROWS)
    assert record["training_tokens"] == steps * tokens_per_step
    assert record["final_metrics"]["loss"] == pytest.approx(losses[-1])
    assert math.isfinite(record["final_metrics"]["grad_norm"])
    cfg = ExperimentConfig.load(exp.run_dir / "config.json")
    assert cfg.to_dict() == exp.to_dict()
    assert isinstance(run.hooked_optimizer, HookedOptimizer)


def _sft_reference(model, tokenizer, rows, completion_only: bool) -> torch.Tensor:
    """Token-mean NLL over (completion + EOS) tokens, or over all tokens; each row unpadded."""
    total, count = torch.zeros((), dtype=torch.float64), 0
    for row in rows:
        p = ids(tokenizer, row["prompt"])
        c = [*ids(tokenizer, row["completion"]), tokenizer.eos_token_id]
        nll = -token_logps(model, p + c)  # nll[t-1] scores target seq[t]
        if completion_only:
            nll = nll[len(p) - 1 :]
        total, count = total + nll.sum(), count + nll.numel()
    return total / count


@pytest.mark.parametrize("completion_only", [None, False])
def test_sft_completion_only_loss_matches_reference(tokenizer, tmp_path, completion_only):
    args = {"max_steps": 1} if completion_only is None else {"max_steps": 1, "completion_only_loss": False}
    exp = make_exp(tmp_path, "sft", **args)
    model = build_tiny_llama(len(tokenizer), seed=3)
    run = build_run(
        exp, train_dataset=Dataset.from_list(SFT_ROWS), model=model, processing_class=tokenizer, record=False
    )
    trainer = run.trainer
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(len(SFT_ROWS))])
    assert (batch["attention_mask"] == 0).any()  # right padding is present in this batch
    with torch.no_grad():
        trl_loss = trainer.compute_loss(trainer.model, batch).double()
        ref_completion = _sft_reference(model, tokenizer, SFT_ROWS, completion_only=True)
        ref_full = _sft_reference(model, tokenizer, SFT_ROWS, completion_only=False)
    assert abs(ref_completion - ref_full) > 1e-2  # including prompt tokens changes the value
    expected = ref_completion if completion_only is None else ref_full
    assert trl_loss.item() == pytest.approx(expected.item(), abs=2e-5)


@pytest.mark.integration
def test_sft_gradient_accumulation_is_token_weighted(tokenizer, tmp_path):
    """One SGD step with GA=2 x bs=2 equals one step with bs=4 (loss normalised by all tokens)."""

    def one_step(run_name: str, bs: int, ga: int) -> dict[str, torch.Tensor]:
        exp = make_exp(
            tmp_path,
            "sft",
            run_name=run_name,
            max_steps=1,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=ga,
            optim="sgd",
            learning_rate=1.0,
            max_grad_norm=0.0,
        )
        model = build_tiny_llama(len(tokenizer), seed=0)
        run = build_run(
            exp, train_dataset=Dataset.from_list(SFT_ROWS), model=model, processing_class=tokenizer
        )
        run.train()
        return snapshot(run.trainer.model)

    init = snapshot(build_tiny_llama(len(tokenizer), seed=0))
    full = one_step("bs4", bs=4, ga=1)
    accum = one_step("bs2ga2", bs=2, ga=2)
    moved = max((full[k] - init[k]).abs().max().item() for k in init)
    assert moved > 1e-3
    for k in init:
        torch.testing.assert_close(accum[k], full[k], rtol=0, atol=1e-6)


@pytest.mark.integration
def test_sft_bf16_autocast_and_fp32_paths_differ_but_agree(tokenizer, tmp_path):
    losses = {}
    for bf16 in (False, True):
        exp = make_exp(
            tmp_path, "sft", run_name=f"bf16-{bf16}", max_steps=2, bf16=bf16, gradient_accumulation_steps=2
        )
        run = build_run(
            exp,
            train_dataset=Dataset.from_list(SFT_ROWS),
            model=build_tiny_llama(len(tokenizer), seed=0),
            processing_class=tokenizer,
        )
        assert run.trainer.accelerator.mixed_precision == ("bf16" if bf16 else "no")
        record = run.train()
        assert record["steps"]["applied_updates"] == 2
        assert all(p.dtype == torch.float32 for p in run.trainer.model.parameters())  # fp32 master weights
        losses[bf16] = run.recorder.history("loss")[0]
    # Autocast really ran: same init and data, different numerics, still close.
    assert losses[True] != losses[False]
    assert losses[True] == pytest.approx(losses[False], rel=5e-2)


@pytest.mark.integration
@pytest.mark.parametrize("bf16", [False, True])
def test_sft_checkpoint_resume_restores_step_and_optimizer_counters(tokenizer, tmp_path, bf16):
    common = {
        "max_steps": 4,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "learning_rate": 5e-3,
        "bf16": bf16,
        "save_only_model": False,
    }
    data = Dataset.from_list(SFT_ROWS)

    def run_for(run_name: str, model_seed: int, **extra):
        exp = make_exp(tmp_path, "sft", run_name=run_name, **common, **extra)
        model = build_tiny_llama(len(tokenizer), seed=model_seed)
        return build_run(exp, train_dataset=data, model=model, processing_class=tokenizer)

    full = run_for("full", 0)
    full_record = full.train()
    part = run_for("part", 0, save_strategy="steps", save_steps=2)
    part.train()
    ckpt = Path(part.trainer.args.output_dir) / "checkpoint-2"
    inv = checkpoint_inventory(ckpt)
    assert inv["optimizer"] and inv["trainer_state"] and inv["model_weights"] and inv["research_state"]
    assert load_research_state(ckpt)["global_step"] == 2

    resumed = run_for("part", 123, save_strategy="steps", save_steps=2)  # different init on purpose
    record = resumed.train(resume_from_checkpoint=str(ckpt))
    opt = resumed.hooked_optimizer
    assert resumed.trainer.state.global_step == 4
    assert opt is not None and (opt.nominal_steps, opt.applied_updates, opt.skipped_steps) == (4, 4, 0)
    assert record["steps"]["applied_updates"] == 4 and record["steps"]["global_step"] == 4
    # TRL's num_tokens counter is not in HF checkpoints; the adapter restores it (else: post-resume only).
    assert load_research_state(ckpt)["algorithm_state"]["trl_total_train_tokens"] > 0
    assert record["training_tokens"] == full_record["training_tokens"] > 0
    # Adam's step counter came back from optimizer.pt (it would be 2 after a cold start).
    adam_steps = {int(s["step"]) for s in opt.inner.state.values()}
    assert adam_steps == {4}
    full_params, resumed_params = snapshot(full.trainer.model), snapshot(resumed.trainer.model)
    for name, tensor in full_params.items():
        assert torch.equal(tensor, resumed_params[name]), name


# ---------------------------------------------------------------------------------------------- DPO


def _dpo_reference(policy, ref, tokenizer, rows, beta: float) -> torch.Tensor:
    losses = []
    for row in rows:
        p = ids(tokenizer, row["prompt"])
        seq_logp = {}
        for key in ("chosen", "rejected"):
            seq = [*p, *ids(tokenizer, row[key]), tokenizer.eos_token_id]
            for name, net in (("pi", policy), ("ref", ref)):
                seq_logp[key, name] = token_logps(net, seq)[len(p) - 1 :].sum()
        margin = (seq_logp["chosen", "pi"] - seq_logp["chosen", "ref"]) - (
            seq_logp["rejected", "pi"] - seq_logp["rejected", "ref"]
        )
        losses.append(-F.logsigmoid(beta * margin))
    return torch.stack(losses).mean()


def test_dpo_loss_matches_reference(tokenizer, tmp_path):
    beta = 0.25
    exp = make_exp(tmp_path, "dpo", max_steps=1, beta=beta, per_device_train_batch_size=3)
    model = build_tiny_llama(len(tokenizer), seed=0)
    run = build_run(
        exp, train_dataset=Dataset.from_list(DPO_ROWS), model=model, processing_class=tokenizer, record=False
    )
    trainer = run.trainer
    assert trainer.loss_types == ["sigmoid"] and trainer.beta == beta
    ref = trainer.ref_model
    assert ref is not model and fingerprint(ref) == fingerprint(model)
    assert not any(p.requires_grad for p in ref.parameters())
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(len(DPO_ROWS))])

    with torch.no_grad():
        loss0 = trainer.compute_loss(trainer.model, dict(batch)).item()
        assert loss0 == pytest.approx(math.log(2), abs=1e-6)
        assert _dpo_reference(model, ref, tokenizer, DPO_ROWS, beta).item() == pytest.approx(math.log(2))

        gen = torch.Generator().manual_seed(0)
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=gen))
        trl_loss = trainer.compute_loss(trainer.model, dict(batch)).item()
        expected = _dpo_reference(model, ref, tokenizer, DPO_ROWS, beta).item()
        wrong_beta = _dpo_reference(model, ref, tokenizer, DPO_ROWS, 0.1).item()
    assert abs(expected - math.log(2)) > 1e-2  # the perturbation is large enough to matter
    assert abs(expected - wrong_beta) > 1e-3
    assert trl_loss == pytest.approx(expected, abs=1e-5)


@pytest.mark.integration
def test_dpo_short_run_and_reference_identity_recorded(tokenizer, tmp_path):
    exp = make_exp(tmp_path, "dpo", max_steps=4, learning_rate=5e-3, per_device_train_batch_size=3, beta=0.5)
    model = build_tiny_llama(len(tokenizer), seed=0)
    init_fp = fingerprint(model)
    run = build_run(exp, train_dataset=Dataset.from_list(DPO_ROWS), model=model, processing_class=tokenizer)
    record = run.train()
    losses = run.recorder.history("loss")
    assert losses[0] == pytest.approx(math.log(2), abs=1e-5)
    assert losses[-1] < losses[0] - 1e-3  # same 3 pairs every step: the margin grows
    assert record["steps"]["applied_updates"] == 4
    assert record["reference_model"]["weights_sha256"] == init_fp
    assert fingerprint(run.trainer.ref_model) == init_fp  # reference stays frozen
    assert fingerprint(run.trainer.model) != init_fp


def test_dpo_reference_model_from_path(tokenizer, tmp_path, tiny_model_dir):
    """``ref_model="<path>"`` is loaded and used: policy (seed 5) != reference (saved seed 0)."""
    exp = make_exp(tmp_path, "dpo", max_steps=1, per_device_train_batch_size=3)
    policy = build_tiny_llama(len(tokenizer), seed=5)
    run = build_run(
        exp,
        train_dataset=Dataset.from_list(DPO_ROWS),
        model=policy,
        ref_model=tiny_model_dir,
        processing_class=tokenizer,
    )
    trainer = run.trainer
    saved_fp = fingerprint(build_tiny_llama(len(tokenizer), seed=0))
    assert fingerprint(trainer.ref_model) == saved_fp != fingerprint(policy)
    assert run.recorder.record["reference_model"]["weights_sha256"] == saved_fp
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(len(DPO_ROWS))])
    with torch.no_grad():
        trl_loss = trainer.compute_loss(trainer.model, dict(batch)).item()
        expected = _dpo_reference(policy, trainer.ref_model, tokenizer, DPO_ROWS, 0.1).item()  # TRL beta
    assert abs(expected - math.log(2)) > 1e-2
    assert trl_loss == pytest.approx(expected, abs=1e-5)


def test_dpo_with_lora_uses_adapter_free_base_as_reference(tokenizer, tmp_path):
    from peft import LoraConfig

    exp = make_exp(tmp_path, "dpo", max_steps=1, per_device_train_batch_size=3, beta=0.5)
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    run = build_run(
        exp,
        train_dataset=Dataset.from_list(DPO_ROWS),
        model=build_tiny_llama(len(tokenizer), seed=0),
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer = run.trainer
    assert trainer.ref_model is None
    assert run.recorder.record["reference_model"] == "policy with PEFT adapters disabled"
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(len(DPO_ROWS))])
    peft_model = trainer.model
    with torch.no_grad():
        assert trainer.compute_loss(peft_model, dict(batch)).item() == pytest.approx(math.log(2), abs=1e-6)
        gen = torch.Generator().manual_seed(0)
        for name, p in peft_model.named_parameters():
            if "lora_B" in name:
                p.add_(0.5 * torch.randn(p.shape, generator=gen))
        trl_loss = trainer.compute_loss(peft_model, dict(batch)).item()

        class AdapterFree(torch.nn.Module):
            def forward(self, input_ids):
                with peft_model.disable_adapter():
                    return peft_model(input_ids=input_ids)

        expected = _dpo_reference(lambda x: peft_model(input_ids=x), AdapterFree(), tokenizer, DPO_ROWS, 0.5)
    assert abs(expected.item() - math.log(2)) > 1e-2
    assert trl_loss == pytest.approx(expected.item(), abs=1e-5)


def test_dpo_with_prewrapped_peft_model_uses_frozen_initial_adapter(tokenizer, tmp_path):
    """A ready-made ``PeftModel`` (non-zero adapter): no frozen policy copy; TRL's "ref" adapter copy is used."""
    import copy

    from peft import LoraConfig, PeftModel, get_peft_model

    beta = 0.5
    exp = make_exp(tmp_path, "dpo", max_steps=1, per_device_train_batch_size=3, beta=beta)
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    policy = get_peft_model(build_tiny_llama(len(tokenizer), seed=0), lora)
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():  # a "pretrained" adapter: base model != base + adapter
        for name, p in policy.named_parameters():
            if "lora_B" in name:
                p.add_(0.5 * torch.randn(p.shape, generator=gen))
    initial = copy.deepcopy(policy).eval()  # independent reference: the policy before training
    run = build_run(exp, train_dataset=Dataset.from_list(DPO_ROWS), model=policy, processing_class=tokenizer)
    trainer = run.trainer
    assert isinstance(trainer.model, PeftModel) and trainer.ref_model is None
    assert "ref" in trainer.model.peft_config
    assert run.recorder.record["reference_model"] == "policy with frozen copy of its initial PEFT adapter ('ref')"
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(len(DPO_ROWS))])
    peft_model = trainer.model
    with torch.no_grad():
        assert trainer.compute_loss(peft_model, dict(batch)).item() == pytest.approx(math.log(2), abs=1e-6)
        for name, p in peft_model.named_parameters():
            if "lora_B.default" in name:
                p.add_(0.5 * torch.randn(p.shape, generator=gen))
        trl_loss = trainer.compute_loss(peft_model, dict(batch)).item()

        def policy_fn(x):
            return peft_model(input_ids=x)

        def initial_fn(x):
            return initial(input_ids=x)

        def base_fn(x):
            with initial.disable_adapter():
                return initial(input_ids=x)

        expected = _dpo_reference(policy_fn, initial_fn, tokenizer, DPO_ROWS, beta).item()
        adapter_free = _dpo_reference(policy_fn, base_fn, tokenizer, DPO_ROWS, beta).item()
    assert abs(expected - math.log(2)) > 1e-2
    assert abs(expected - adapter_free) > 1e-3  # the reference choice matters
    assert trl_loss == pytest.approx(expected, abs=1e-5)


def test_dpo_precompute_ref_log_probs_keeps_no_reference_copy(tokenizer, tmp_path):
    """With ``precompute_ref_log_probs`` TRL uses the untrained policy once in ``__init__``; no copy is kept."""
    beta = 0.5
    exp = make_exp(
        tmp_path, "dpo", max_steps=1, per_device_train_batch_size=3, beta=beta, precompute_ref_log_probs=True
    )
    model = build_tiny_llama(len(tokenizer), seed=0)
    init_fp = fingerprint(model)
    run = build_run(exp, train_dataset=Dataset.from_list(DPO_ROWS), model=model, processing_class=tokenizer)
    trainer = run.trainer
    assert trainer.ref_model is None
    recorded = run.recorder.record["reference_model"]
    assert recorded["weights_sha256"] == init_fp and "precomputed" in recorded["reference"]
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(len(DPO_ROWS))])
    assert "ref_chosen_logps" in batch and "ref_rejected_logps" in batch
    reference = build_tiny_llama(len(tokenizer), seed=0)  # independent copy of the initial policy
    with torch.no_grad():
        assert trainer.compute_loss(trainer.model, dict(batch)).item() == pytest.approx(math.log(2), abs=1e-6)
        gen = torch.Generator().manual_seed(0)
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=gen))
        trl_loss = trainer.compute_loss(trainer.model, dict(batch)).item()
        expected = _dpo_reference(model, reference, tokenizer, DPO_ROWS, beta).item()
    assert abs(expected - math.log(2)) > 1e-2
    assert trl_loss == pytest.approx(expected, abs=1e-5)


@pytest.mark.integration
@pytest.mark.parametrize("logging_steps", [1, 3])
def test_dpo_checkpoint_resume_restores_training_tokens(tokenizer, tmp_path, logging_steps):
    """``logging_steps=3`` with ``save_steps=2``: the checkpoint's log history has no ``num_tokens`` yet."""
    common = {
        "max_steps": 4,
        "per_device_train_batch_size": 2,
        "learning_rate": 5e-3,
        "logging_steps": logging_steps,
        "save_only_model": False,
    }
    data = Dataset.from_list(DPO_ROWS)

    def run_for(run_name: str, **extra):
        exp = make_exp(tmp_path, "dpo", run_name=run_name, **common, **extra)
        # Same seed everywhere: the frozen reference is copied from the model given at build time.
        model = build_tiny_llama(len(tokenizer), seed=0)
        return build_run(exp, train_dataset=data, model=model, processing_class=tokenizer)

    full_record = run_for("full").train()
    part = run_for("part", save_strategy="steps", save_steps=2)
    part.train()
    ckpt = Path(part.trainer.args.output_dir) / "checkpoint-2"
    saved = load_research_state(ckpt)["algorithm_state"]["trl_total_train_tokens"]
    resumed = run_for("part", save_strategy="steps", save_steps=2)
    record = resumed.train(resume_from_checkpoint=str(ckpt))
    assert resumed.trainer.state.global_step == 4
    assert 0 < saved < full_record["training_tokens"]
    assert record["training_tokens"] == full_record["training_tokens"]


# ---------------------------------------------------------------------------------------------- distillation


def _peaky_teacher(tokenizer, seed: int = 1):
    teacher = build_tiny_llama(len(tokenizer), seed=seed)
    with torch.no_grad():
        teacher.lm_head.weight.mul_(25.0)  # far from uniform, so divergences are not ~0
    return teacher


def _gjsd_reference(zs: torch.Tensor, zt: torch.Tensor, beta: float, temperature: float) -> torch.Tensor:
    """Per-token generalized JSD of TRL's DistillationTrainer, written from its definition."""
    ls = torch.log_softmax(zs.double() / temperature, -1)
    lt = torch.log_softmax(zt.double() / temperature, -1)
    ps, pt = ls.exp(), lt.exp()
    if beta == 0.0:
        return (pt * (lt - ls)).sum(-1)  # KL(teacher || student)
    if beta == 1.0:
        return (ps * (ls - lt)).sum(-1)  # KL(student || teacher)
    lm = torch.log(beta * pt + (1 - beta) * ps)
    return beta * (pt * (lt - lm)).sum(-1) + (1 - beta) * (ps * (ls - lm)).sum(-1)


def _distill_batch(tokenizer, prompts: list[str], completions: list[str]) -> dict[str, torch.Tensor]:
    """Left-padded prompts + right-padded completions, exactly the layout TRL builds after generation."""
    p_ids = [ids(tokenizer, p) for p in prompts]
    c_ids = [ids(tokenizer, c) for c in completions]
    lp, lc = max(map(len, p_ids)), max(map(len, c_ids))
    pad = tokenizer.pad_token_id
    return {
        "prompt_ids": torch.tensor([[pad] * (lp - len(x)) + x for x in p_ids]),
        "prompt_mask": torch.tensor([[0] * (lp - len(x)) + [1] * len(x) for x in p_ids]),
        "completion_ids": torch.tensor([x + [pad] * (lc - len(x)) for x in c_ids]),
        "completion_mask": torch.tensor([[1] * len(x) + [0] * (lc - len(x)) for x in c_ids]),
    }


def _distill_reference(student, teacher, tokenizer, prompts, completions, beta, temperature):
    """Sum of per-token divergences over completion tokens, each row run unpadded; and the token count."""
    total, count = torch.zeros((), dtype=torch.float64), 0
    for prompt, completion in zip(prompts, completions):
        p, c = ids(tokenizer, prompt), ids(tokenizer, completion)
        seq = torch.tensor([p + c])
        positions = slice(len(p) - 1, len(p) + len(c) - 1)  # logits that predict the completion tokens
        zs = student(seq).logits[0, positions]
        with torch.no_grad():
            zt = teacher(seq).logits[0, positions]
        total = total + _gjsd_reference(zs, zt, beta, temperature).sum()
        count += len(c)
    return total, count


PROMPTS = ["1+1=", "12+30=", "7*8="]
COMPLETIONS = ["2</s>", "42 is it", "56"]


@pytest.mark.parametrize(("beta", "temperature"), [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (0.3, 2.0)])
def test_distillation_loss_matches_generalized_jsd_reference(tokenizer, tmp_path, beta, temperature):
    exp = make_exp(tmp_path, "distillation", max_steps=1, beta=beta, temperature=temperature)
    student = build_tiny_llama(len(tokenizer), seed=0)
    teacher = _peaky_teacher(tokenizer)
    run = build_run(
        exp,
        train_dataset=Dataset.from_list([{"prompt": p} for p in PROMPTS]),
        model=student,
        processing_class=tokenizer,
        teacher_model=teacher,
        record=False,
    )
    trainer = run.trainer
    assert trainer.beta == beta and trainer.temperature == temperature
    batch = _distill_batch(tokenizer, PROMPTS, COMPLETIONS)
    assert (batch["prompt_mask"] == 0).any() and (batch["completion_mask"] == 0).any()

    trl_loss = trainer.compute_loss(trainer.model, dict(batch))
    ref_sum, n_tokens = _distill_reference(
        student, teacher, tokenizer, PROMPTS, COMPLETIONS, beta, temperature
    )
    ref_loss = ref_sum / n_tokens
    assert ref_loss.item() > 0.05  # not vacuously ~0
    assert trl_loss.item() == pytest.approx(ref_loss.item(), rel=1e-4, abs=1e-6)

    # Sensitivity of the reference itself: the temperature and beta settings change the value.
    other_t, _ = _distill_reference(
        student, teacher, tokenizer, PROMPTS, COMPLETIONS, beta, temperature + 1.0
    )
    other_b, _ = _distill_reference(
        student, teacher, tokenizer, PROMPTS, COMPLETIONS, 1.0 - beta if beta != 0.5 else 0.0, temperature
    )
    assert abs(other_t.item() / n_tokens - ref_loss.item()) > 1e-3
    assert abs(other_b.item() / n_tokens - ref_loss.item()) > 1e-3

    # Gradient-accumulation normalisation: with num_items_in_batch the sum is divided by that count.
    trainer.model.train()
    scaled = trainer.compute_loss(trainer.model, {**batch, "num_items_in_batch": torch.tensor(4 * n_tokens)})
    assert scaled.item() == pytest.approx(ref_sum.item() / (4 * n_tokens), rel=1e-4, abs=1e-6)

    # Student gradient (custom chunked backward in TRL) equals autograd through the reference.
    student.zero_grad()
    trl_loss.backward()
    trl_grads = {n: p.grad.clone() for n, p in student.named_parameters()}
    student.zero_grad()
    ref_sum2, _ = _distill_reference(student, teacher, tokenizer, PROMPTS, COMPLETIONS, beta, temperature)
    (ref_sum2 / n_tokens).backward()
    assert all(p.grad is None for p in teacher.parameters())
    for name, param in student.named_parameters():
        torch.testing.assert_close(trl_grads[name], param.grad.float(), rtol=1e-3, atol=1e-6, msg=name)


@pytest.mark.integration
def test_distillation_short_on_policy_run(tokenizer, tmp_path):
    exp = make_exp(
        tmp_path,
        "distillation",
        max_steps=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_completion_length=12,
        learning_rate=1e-3,
        beta=0.5,
    )
    student = build_tiny_llama(len(tokenizer), seed=0)
    teacher = _peaky_teacher(tokenizer)
    teacher_fp, student_fp = fingerprint(teacher), fingerprint(student)
    data = Dataset.from_list([{"prompt": f"{a}+{b}="} for a in range(4) for b in range(4)])
    run = build_run(exp, train_dataset=data, model=student, processing_class=tokenizer, teacher_model=teacher)
    record = run.train()
    losses = run.recorder.history("loss")
    assert len(losses) == 3 and all(x > 0 and math.isfinite(x) for x in losses)
    # Completions were sampled from the student, and some ended with EOS before the length limit, so the
    # loss saw ragged, right-padded completions (not only max-length ones).
    assert min(run.recorder.history("completions/clipped_ratio")) < 1.0
    mins, maxs = (
        run.recorder.history("completions/min_length"),
        run.recorder.history("completions/max_length"),
    )
    assert any(lo < hi for lo, hi in zip(mins, maxs))
    assert record["steps"]["applied_updates"] == 3
    assert record["teacher_model"]["weights_sha256"] == teacher_fp
    assert fingerprint(teacher) == teacher_fp  # teacher is never updated
    assert fingerprint(run.trainer.model) != student_fp
    assert record["training_tokens"] > 0


def test_distillation_requires_teacher(tokenizer, tmp_path):
    exp = make_exp(tmp_path, "distillation", max_steps=1)
    with pytest.raises(ValueError, match="teacher"):
        build_run(
            exp,
            train_dataset=Dataset.from_list([{"prompt": "1+1="}]),
            model=build_tiny_llama(len(tokenizer), seed=0),
            processing_class=tokenizer,
            record=False,
        )


def test_distillation_rejects_teacher_of_wrong_type(tokenizer, tmp_path):
    exp = make_exp(tmp_path, "distillation", max_steps=1)
    with pytest.raises(TypeError, match="teacher_model must be"):
        build_run(
            exp,
            train_dataset=Dataset.from_list([{"prompt": "1+1="}]),
            model=build_tiny_llama(len(tokenizer), seed=0),
            processing_class=tokenizer,
            teacher_model=123,
            record=False,
        )


@pytest.mark.parametrize("via", ["kwarg", "config"])
def test_distillation_teacher_from_path(tokenizer, tmp_path, tiny_model_dir, via):
    """A teacher path reaches TRL, as ``teacher_model=`` or as ``trainer.args.teacher_model_name_or_path``."""
    args = (
        {"max_steps": 1} if via == "kwarg" else {"max_steps": 1, "teacher_model_name_or_path": tiny_model_dir}
    )
    exp = make_exp(tmp_path, "distillation", **args)
    student = build_tiny_llama(len(tokenizer), seed=7)
    extra = {"teacher_model": tiny_model_dir} if via == "kwarg" else {}
    run = build_run(
        exp,
        train_dataset=Dataset.from_list([{"prompt": p} for p in PROMPTS]),
        model=student,
        processing_class=tokenizer,
        **extra,
    )
    saved = build_tiny_llama(len(tokenizer), seed=0)
    teacher = run.trainer.teacher_model
    assert fingerprint(teacher) == fingerprint(saved) != fingerprint(student)
    assert run.recorder.record["teacher_model"]["weights_sha256"] == fingerprint(saved)
    batch = _distill_batch(tokenizer, PROMPTS, COMPLETIONS)
    with torch.no_grad():
        trl_loss = run.trainer.compute_loss(run.trainer.model, dict(batch)).item()
        ref_sum, n_tokens = _distill_reference(student, saved, tokenizer, PROMPTS, COMPLETIONS, 1.0, 1.0)
    assert trl_loss == pytest.approx(ref_sum.item() / n_tokens, rel=1e-4, abs=1e-6)  # TRL default: reverse KL


def test_distillation_teacher_kwarg_takes_precedence_over_config(tokenizer, tmp_path, tiny_model_dir):
    exp = make_exp(tmp_path, "distillation", max_steps=1, teacher_model_name_or_path=tiny_model_dir)
    teacher = _peaky_teacher(tokenizer, seed=3)
    run = build_run(
        exp,
        train_dataset=Dataset.from_list([{"prompt": p} for p in PROMPTS]),
        model=build_tiny_llama(len(tokenizer), seed=0),
        processing_class=tokenizer,
        teacher_model=teacher,
        record=False,
    )
    assert run.trainer.teacher_model is teacher


# ---------------------------------------------------------------------------------------------- examples


@pytest.mark.parametrize("name", ["sft", "dpo", "distillation"])
def test_example_configs_load(name):
    exp = ExperimentConfig.load(REPO / "examples" / "configs" / f"{name}_tiny.yaml")
    assert exp.trainer.algorithm == name
    assert isinstance(exp.trainer.args["learning_rate"], float)  # YAML "1e-3" would load as a string
    assert exp.trainer.args["bf16"] is False
    TRAINERS.build(name).build_trl_config(exp)  # every key is a valid TRL config argument


def _load_example(name: str):
    path = REPO / "examples" / f"{name}_tiny.py"
    spec = importlib.util.spec_from_file_location(f"example_{name}_tiny", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
@pytest.mark.parametrize("name", ["sft", "dpo", "distillation"])
def test_example_scripts_run_offline(name, tmp_path):
    record = _load_example(name).main(["--output-dir", str(tmp_path), "--max-steps", "2"])
    assert record["status"] == "completed" and record["algorithm"] == name
    assert record["steps"]["applied_updates"] == 2
    run_dirs = list(tmp_path.glob("*/*/record.json"))
    assert len(run_dirs) == 1
