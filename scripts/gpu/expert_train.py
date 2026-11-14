#!/usr/bin/env python
"""Train one independent FFN expert (TLC-style) on a single GPU (run at home; also runs on CPU).

Pipeline:
  1. load the base causal LM, attach ExpertFFN containers to the chosen decoder layers,
  2. create expert ``--expert`` as an exact copy of the base FFN, freeze the shared backbone,
  3. SFT only that expert through ResearchRL's ``sft`` adapter (TRL SFTTrainer + HookedOptimizer),
  4. verify bit-exactly that the backbone did not change, save the expert (safetensors + metadata),
  5. optionally export a merged vanilla HF checkpoint for inference with any HF/vLLM stack.

Response-level expert selection (TLC) is NOT token-level MoE routing: one expert path is used for
a whole response.

    python scripts/gpu/expert_train.py --model Qwen/Qwen2.5-0.5B-Instruct --layers 12-23 \
        --expert math --dataset <hf prompt/completion dataset> --bf16 --max-steps 200 --output-dir runs/experts

Default (no ``--model``) is the offline tiny model with a toy dataset: a pipeline check only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import torch
from datasets import Dataset, load_dataset

from research_rl.core.config import DataSpec, ExperimentConfig, ModelSpec, TrainerSpec
from research_rl.models.expert_manager import ExpertManager
from research_rl.models.tiny import save_tiny_model
from research_rl.trainers import build_run
from research_rl.trainers.base import load_causal_lm, load_tokenizer


def parse_layers(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="HF id or path (default: offline tiny model)")
    p.add_argument("--layers", default="0-1", help="decoder layers whose FFN becomes an expert, e.g. 12-23")
    p.add_argument("--expert", default="expert0")
    p.add_argument("--dataset", default=None, help="HF dataset with prompt/completion columns")
    p.add_argument("--split", default="train")
    p.add_argument("--max-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--output-dir", default="runs/experts")
    p.add_argument("--export-merged", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    model_path = args.model or str(save_tiny_model(out_dir / "tiny_model", seed=args.seed))
    exp = ExperimentConfig(
        study_id="tlc-expert",
        run_name=f"{args.expert}-seed{args.seed}",
        seed=args.seed,
        output_dir=str(out_dir),
        model=ModelSpec(name_or_path=model_path, dtype="bfloat16" if args.bf16 else "float32"),
        data=DataSpec(name=args.dataset or "toy-arithmetic", split=args.split),
        trainer=TrainerSpec(
            algorithm="sft",
            args={
                "max_steps": args.max_steps,
                "learning_rate": args.lr,
                "per_device_train_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.grad_accum,
                "bf16": args.bf16,
                "weight_decay": 0.0,
                "logging_steps": 1,
            },
        ),
        tags=["tlc", "expert"],
    )
    if args.dataset:
        dataset = load_dataset(args.dataset, split=args.split)
    else:
        dataset = Dataset.from_list(
            [{"prompt": f"{a}*{b}=", "completion": str(a * b)} for a in range(10) for b in range(10)]
        )

    model = load_causal_lm(exp.model)
    tokenizer = load_tokenizer(exp.model)
    mgr = ExpertManager(model)
    mgr.attach(parse_layers(args.layers))
    mgr.create_expert(args.expert)
    mgr.set_trainable(args.expert)
    backbone_ref = mgr.freeze_backbone()

    run = build_run(
        exp, train_dataset=dataset, model=model, processing_class=tokenizer, components={"experts": mgr}
    )
    record = run.train()
    mgr.assert_backbone_unchanged(backbone_ref)  # raises if any shared weight moved
    expert_file = mgr.save_expert(args.expert, exp.run_dir / f"expert_{args.expert}.safetensors")
    result = {
        "run_dir": str(exp.run_dir),
        "expert_file": str(expert_file),
        "backbone_unchanged": True,
        "trainable_parameters": record.get("trainable_parameters"),
        "steps": record.get("steps"),
        "final_metrics": record.get("final_metrics"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if args.export_merged:
        result["merged_checkpoint"] = str(mgr.export_merged(args.expert, exp.run_dir / "merged"))
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
