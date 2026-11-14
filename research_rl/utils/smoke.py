"""End-to-end smoke check: a real TRL trainer step with ResearchRL update rules installed.

Used by ``scripts/gpu/cuda_smoke_test.py`` on a CUDA host (where Accelerate moves the optimizer
state, including rule state, to the device) and by the CPU test-suite. It exercises
``ResearchTrainerMixin`` -> ``HookedOptimizer`` inside Accelerate's ``AcceleratedOptimizer`` with a
``MaskRule`` and a ``P2PriorRule`` (clip inside the wrapper, TRL ``max_grad_norm=0``), then checks
that the frozen region is bit-identical and that the P²-prior coefficients moved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import Dataset

from research_rl.core.config import DataSpec, ExperimentConfig, ModelSpec, OptimSpec, TrainerSpec
from research_rl.models.model_interface import snapshot
from research_rl.models.parameter_mask import ParameterMask
from research_rl.models.tiny import build_char_tokenizer, build_tiny_llama
from research_rl.optim import MaskRule
from research_rl.optim.parameter_prior import P2PriorRule, TaskVector

PRIOR_MODULES = [f"model.layers.0.mlp.{p}_proj.weight" for p in ("gate", "up", "down")]
MASKED = "model.layers.1.mlp.gate_proj.weight"


def trl_rules_smoke(output_dir: str | Path, steps: int = 2, bf16: bool = False) -> dict[str, Any]:
    """Run ``steps`` SFT steps through TRL with MaskRule + P2PriorRule; return the checks."""
    import research_rl.trainers.sft_adapter  # noqa: F401  (registers "sft")
    from research_rl.trainers import build_run

    tok = build_char_tokenizer()
    model = build_tiny_llama(len(tok), seed=0)
    base = snapshot(model, PRIOR_MODULES)
    g = torch.Generator().manual_seed(1)
    task_vectors = [
        TaskVector({n: 0.01 * torch.randn(t.shape, generator=g) for n, t in base.items()}) for _ in range(2)
    ]
    w = dict(model.named_parameters())[MASKED]
    mask = torch.ones_like(w, dtype=torch.bool)
    mask[w.shape[0] // 2 :] = False
    prior = P2PriorRule(task_vectors, PRIOR_MODULES, lr=1e-2)
    rules = [MaskRule(ParameterMask({MASKED: mask})), prior]

    exp = ExperimentConfig(
        study_id="smoke",
        run_name="trl-rules",
        output_dir=str(output_dir),
        model=ModelSpec(name_or_path="tiny"),
        data=DataSpec(name="toy"),
        trainer=TrainerSpec(
            algorithm="sft",
            args={
                "max_steps": steps,
                "per_device_train_batch_size": 4,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "bf16": bf16,
            },
        ),
        optim=OptimSpec(clip_in_wrapper=1.0),
    )
    data = Dataset.from_list([{"prompt": f"{a}+{a}=", "completion": str(2 * a)} for a in range(8)])
    frozen_before = w.detach()[w.shape[0] // 2 :].clone().cpu()
    alpha_before = prior.coefficients.clone()
    run = build_run(exp, train_dataset=data, model=model, processing_class=tok, update_rules=rules)
    record = run.train()
    w_after = dict(run.trainer.model.named_parameters())[MASKED].detach()
    opt = run.hooked_optimizer
    return {
        "device": str(w_after.device),
        "applied_updates": opt.applied_updates if opt is not None else None,
        "mask_bit_exact": bool(torch.equal(frozen_before, w_after[w_after.shape[0] // 2 :].cpu())),
        "p2_coefficients_moved": bool(not torch.equal(alpha_before, prior.coefficients)),
        "p2_step_count": prior.step_count,
        "final_loss": record.get("final_metrics", {}).get("loss"),
    }
