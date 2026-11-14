"""HookedOptimizer inside a real TRL trainer: masks, counters and bit-exact checkpoint resume."""

from __future__ import annotations

import torch
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from research_rl.core.checkpoint import checkpoint_inventory, load_research_state
from research_rl.models.model_interface import snapshot
from research_rl.models.parameter_mask import ParameterMask
from research_rl.models.tiny import build_tiny_llama
from research_rl.optim import DisplacementLedger, HookedOptimizer, MaskRule
from research_rl.trainers.base import ResearchCallback, research_trainer_class

FROZEN_ROWS = slice(16, 64)  # gate_proj rows 16.. are frozen in layer 0


def sft_data() -> Dataset:
    rows = [{"prompt": f"{a}+{b}=", "completion": str(a + b)} for a in range(4) for b in range(4)]
    return Dataset.from_list(rows)


def make_mask(model) -> ParameterMask:
    w = dict(model.named_parameters())["model.layers.0.mlp.gate_proj.weight"]
    m = torch.ones_like(w, dtype=torch.bool)
    m[FROZEN_ROWS] = False
    return ParameterMask({"model.layers.0.mlp.gate_proj.weight": m})


class StepCounter(TrainerCallback):
    def __init__(self):
        self.steps = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.steps += 1


def make_trainer(tokenizer, out_dir, max_steps, save_steps=None, model_seed=0):
    model = build_tiny_llama(len(tokenizer), seed=model_seed)
    args = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        max_steps=max_steps,
        learning_rate=5e-3,
        weight_decay=0.1,
        lr_scheduler_type="linear",
        logging_steps=1,
        report_to=[],
        use_cpu=True,
        seed=7,
        save_strategy="steps" if save_steps else "no",
        save_steps=save_steps or 500,
        save_only_model=False,
    )
    cls = research_trainer_class(SFTTrainer)
    callback = ResearchCallback()
    counter = StepCounter()
    trainer = cls(
        model=model,
        args=args,
        train_dataset=sft_data(),
        processing_class=tokenizer,
        update_rules=[MaskRule(make_mask(model)), DisplacementLedger()],
        callbacks=[callback, counter],
    )
    callback.bind(trainer)
    trainer.step_counter = counter
    return trainer


def test_hooked_optimizer_is_installed_and_mask_holds(tokenizer, tmp_path):
    trainer = make_trainer(tokenizer, tmp_path / "a", max_steps=3)
    before = snapshot(trainer.model)
    trainer.train()
    after = snapshot(trainer.model)
    opt = trainer.hooked_optimizer
    assert isinstance(opt, HookedOptimizer)
    # 3 optimizer steps (gradient accumulation of 2 does not multiply hook calls)
    assert (opt.nominal_steps, opt.applied_updates, trainer.state.global_step) == (3, 3, 3)
    name = "model.layers.0.mlp.gate_proj.weight"
    assert torch.equal(before[name][FROZEN_ROWS], after[name][FROZEN_ROWS])
    assert not torch.equal(before[name][:16], after[name][:16])


def test_bit_exact_resume_through_trl(tokenizer, tmp_path):
    full = make_trainer(tokenizer, tmp_path / "full", max_steps=4)
    full.train()
    full_params = snapshot(full.model)

    part = make_trainer(tokenizer, tmp_path / "part", max_steps=4, save_steps=2)
    part.train()  # writes checkpoint-2 and checkpoint-4
    ckpt = tmp_path / "part" / "checkpoint-2"
    inv = checkpoint_inventory(ckpt)
    assert all(
        inv[k]
        for k in ("optimizer", "scheduler", "rng_state", "trainer_state", "model_weights", "research_state")
    )
    state = load_research_state(ckpt)
    assert state["policy_version"] == 2 and state["global_step"] == 2

    # Different random init: only a genuine resume from checkpoint-2 can reproduce the full run.
    resumed = make_trainer(tokenizer, tmp_path / "part", max_steps=4, save_steps=2, model_seed=123)
    resumed.train(resume_from_checkpoint=str(ckpt))
    assert resumed.step_counter.steps == 2  # only steps 3 and 4 were executed
    opt = resumed.hooked_optimizer
    assert opt is not None and opt.applied_updates == 4 and opt.nominal_steps == 4
    resumed_params = snapshot(resumed.model)
    for name, tensor in full_params.items():
        assert torch.equal(tensor, resumed_params[name]), name
    ledger_full = full.hooked_optimizer.rules[1].total_sq_norm
    ledger_resumed = opt.rules[1].total_sq_norm
    assert ledger_full.keys() == ledger_resumed.keys()
    for k in ledger_full:
        assert abs(ledger_full[k] - ledger_resumed[k]) <= 1e-9 * max(1.0, ledger_full[k])
