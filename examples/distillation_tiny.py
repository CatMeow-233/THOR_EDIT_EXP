#!/usr/bin/env python
"""On-policy knowledge distillation (not RL) through ResearchRL's ``distillation`` adapter.

The student samples completions for each prompt and is trained to match the teacher's next-token
distribution on them (generalized JSD; ``beta``/``temperature`` in the config). By default this
writes two random tiny Llamas (student seed = config seed, teacher seed = config seed + 1) into the
output dir and runs a few CPU steps on in-memory arithmetic prompts (a smoke run: the numbers say
nothing about real model quality). On a CUDA machine pass ``--model <student> --teacher <teacher>
--bf16`` (same vocabulary) and optionally ``--dataset`` to run real training with the same code path.
The experiment record is written to ``<output_dir>/<study_id>/<run_name>/record.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from datasets import Dataset, load_dataset

import research_rl.trainers.distillation_adapter  # noqa: F401  (registers "distillation")
from research_rl.core.config import ExperimentConfig
from research_rl.models.tiny import save_tiny_model
from research_rl.trainers import build_run

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "distillation_tiny.yaml"
TINY = "research_rl:tiny"


def toy_dataset() -> Dataset:
    return Dataset.from_list([{"prompt": f"{a}+{b}="} for a in range(10) for b in range(10)])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="ExperimentConfig YAML/JSON")
    p.add_argument(
        "--model", default=None, help="student: HF hub id or local path (default: offline tiny model)"
    )
    p.add_argument(
        "--teacher", default=None, help="teacher: HF hub id or local path (default: from the config)"
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="HF dataset with a prompt column, loaded with datasets.load_dataset (needs network or a local "
        "copy; not exercised in CI). Default: in-memory toy arithmetic prompts.",
    )
    p.add_argument("--output-dir", default=None, help="overrides config output_dir")
    p.add_argument("--max-steps", type=int, default=None, help="overrides trainer.args.max_steps")
    p.add_argument("--bf16", action="store_true", help="bf16 mixed precision (autocast; on CPU too)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    exp = ExperimentConfig.load(args.config)
    if args.output_dir is not None:
        exp.output_dir = args.output_dir
    if args.max_steps is not None:
        exp.trainer.args["max_steps"] = args.max_steps
    if args.bf16:
        exp.trainer.args["bf16"] = True
    if args.model is not None:
        exp.model.name_or_path = args.model
    if exp.model.name_or_path == TINY:
        exp.model.name_or_path = str(save_tiny_model(Path(exp.output_dir) / "tiny_student", seed=exp.seed))
    teacher = args.teacher or exp.trainer.args.get("teacher_model_name_or_path")
    if teacher is None:
        raise SystemExit("no teacher: pass --teacher or set trainer.args.teacher_model_name_or_path")
    if teacher == TINY:
        teacher = str(save_tiny_model(Path(exp.output_dir) / "tiny_teacher", seed=exp.seed + 1))
    # Stored in the config (and so in the experiment record); the adapter passes it to DistillationTrainer.
    exp.trainer.args["teacher_model_name_or_path"] = teacher
    if args.dataset is not None:
        dataset = load_dataset(args.dataset, split=exp.data.split)
        exp.data.name = args.dataset
    else:
        dataset = toy_dataset()

    run = build_run(exp, train_dataset=dataset)
    record = run.train()
    summary = {
        k: record[k] for k in ("study_id", "algorithm", "status", "steps", "training_tokens", "final_metrics")
    }
    print(json.dumps({"run_dir": str(exp.run_dir), **summary}, indent=2))
    return record


if __name__ == "__main__":
    main()
