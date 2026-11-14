#!/usr/bin/env python
"""Math RLVR with GRPO (reinforcement learning from a verifiable reward) through ResearchRL.

By default this writes a random tiny Llama + char tokenizer into the output dir and runs a few CPU
steps on a synthetic arithmetic set (prompt asks for ``\\boxed{answer}``; ``MathVerifier`` grades the
final answer). This is a smoke run: a random tiny model essentially never earns a reward, so with
``optim.skip_zero_advantage_steps: true`` (the example config) most or all steps are *skipped*.
The record states this honestly as ``applied_updates`` vs ``nominal_optimizer_steps``. CPU numbers
say nothing about real model quality.

NOT_TESTED (no GPU in the dev environment): on a CUDA machine run e.g. ``--model
Qwen/Qwen2.5-0.5B-Instruct --bf16 --max-steps 200 --num-generations 8`` (and raise
``max_completion_length`` in a copy of the config) for real training with the same code path. Only
the CPU path above has been run. The experiment record is written to
``<output_dir>/<study_id>/<run_name>/record.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from datasets import Dataset
from transformers import GenerationConfig

from research_rl.core.config import ExperimentConfig
from research_rl.models.tiny import save_tiny_model
from research_rl.trainers import build_run
from research_rl.trainers.base import load_tokenizer

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
TINY = "research_rl:tiny"


def arithmetic_dataset(n: int, seed: int) -> Dataset:
    """Synthetic (prompt, answer) pairs; conversational prompts so chat models see their template."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = rng.randint(0, 20), rng.randint(0, 20)
        op = rng.choice("+-*")
        value = a + b if op == "+" else a - b if op == "-" else a * b
        question = f"What is {a}{op}{b}? Put the final answer in \\boxed{{}}."
        rows.append({"prompt": [{"role": "user", "content": question}], "answer": str(value)})
    return Dataset.from_list(rows)


def eos_token_ids(exp: ExperimentConfig, tokenizer: Any) -> list[int]:
    """EOS ids the way TRL collects them: the tokenizer's plus ``generation_config.eos_token_id``."""
    ids = [tokenizer.eos_token_id]
    try:
        gen = GenerationConfig.from_pretrained(exp.model.name_or_path, revision=exp.model.revision or "main")
    except (OSError, ValueError):  # no generation_config.json: the tokenizer's EOS is all TRL uses as well
        return ids
    extra = gen.eos_token_id
    for i in [extra] if isinstance(extra, int) else list(extra or []):
        if i not in ids:
            ids.append(int(i))
    return ids


def parse_args(argv: list[str] | None, algorithm: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config", default=str(CONFIG_DIR / f"{algorithm}_math_tiny.yaml"), help="ExperimentConfig"
    )
    p.add_argument("--model", default=None, help="HF hub id or local path (default: offline tiny model)")
    p.add_argument("--output-dir", default=None, help="overrides config output_dir")
    p.add_argument("--max-steps", type=int, default=None, help="overrides trainer.args.max_steps")
    p.add_argument("--num-generations", type=int, default=None, help="overrides trainer.args.num_generations")
    p.add_argument("--bf16", action="store_true", help="bf16 mixed precision (autocast; on CPU too)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, algorithm: str = "grpo") -> dict[str, Any]:
    from research_rl.rewards.math_verifier import MathVerifier
    from research_rl.rewards.verifier_interface import as_trl_reward_func

    args = parse_args(argv, algorithm)
    exp = ExperimentConfig.load(args.config)
    if exp.trainer.algorithm != algorithm:
        raise SystemExit(
            f"{args.config} is a {exp.trainer.algorithm!r} config, this script runs {algorithm!r}"
        )
    if args.output_dir is not None:
        exp.output_dir = args.output_dir
    if args.max_steps is not None:
        exp.trainer.args["max_steps"] = args.max_steps
    if args.num_generations is not None:
        exp.trainer.args["num_generations"] = args.num_generations
    if args.bf16:
        exp.trainer.args["bf16"] = True
    if args.model is not None:
        exp.model.name_or_path = args.model
    if exp.model.name_or_path == TINY:
        exp.model.name_or_path = str(save_tiny_model(Path(exp.output_dir) / "tiny_model", seed=exp.seed))

    tokenizer = load_tokenizer(exp.model)
    reward = as_trl_reward_func(
        MathVerifier(), answer_key="answer", eos_token_ids=eos_token_ids(exp, tokenizer)
    )
    dataset = arithmetic_dataset(int(exp.extra.get("num_train_examples", 64)), exp.seed)

    run = build_run(exp, train_dataset=dataset, processing_class=tokenizer, reward_funcs=[reward])
    record = run.train()
    keys = ("study_id", "algorithm", "status", "steps", "training_tokens", "final_metrics", "rollout")
    print(json.dumps({"run_dir": str(exp.run_dir), **{k: record.get(k) for k in keys}}, indent=2))
    return record


if __name__ == "__main__":
    main()
