#!/usr/bin/env python
"""Full training-step throughput + GPU memory profile on ONE GPU (run at home; not in CI).

Measures complete optimizer steps (forward + backward + HookedOptimizer(AdamW) step) of a causal
LM in BF16 on a single CUDA device and writes a JSON report with:

* tokens/s and seconds/step (warm-up steps excluded, CUDA-synchronised timing),
* peak allocated / reserved memory, parameter / gradient / optimizer-state footprint,
* optionally ``torch.cuda.memory_summary()`` and a memory snapshot pickle for
  https://pytorch.org/memory_viz.

Each RTX 5090 (32 GB) is a separate host: run this script on each machine separately; the two
GPUs are never treated as one 64 GB device.

    python scripts/gpu/train_step_benchmark.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --batch-size 4 --seq-len 1024 --steps 20 --out runs/gpu/bench_qwen05b.json

Without CUDA the script writes ``"status": "NOT_TESTED"`` and exits with code 2; it never reports
CPU numbers as GPU results. ``--model research_rl:tiny`` uses the offline tiny model (for a quick
pipeline check only).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, cast

import torch

from research_rl.models.tiny import build_char_tokenizer, build_tiny_llama
from research_rl.optim import HookedOptimizer
from research_rl.utils.profiling import gpu_memory_report, hardware_report, measure_throughput

TINY = "research_rl:tiny"


def _bytes(tensors) -> int:
    return int(sum(t.numel() * t.element_size() for t in tensors))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=TINY)
    p.add_argument("--revision", default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--memory-summary", action="store_true", help="include torch.cuda.memory_summary()")
    p.add_argument("--memory-snapshot", default=None, help="write a CUDA memory snapshot pickle here")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="runs/gpu/train_step_benchmark.json")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"script": "train_step_benchmark", "args": vars(args), "hardware": hardware_report()}
    if not torch.cuda.is_available():
        report.update(status="NOT_TESTED", reason="CUDA not available; no GPU numbers are produced on CPU")
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 2

    dev = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model: Any
    if args.model == TINY:
        tok = build_char_tokenizer()
        model = cast(Any, build_tiny_llama(len(tok))).to(dtype)
        vocab = len(tok)
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=dtype)
        vocab = model.config.vocab_size
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(dev).train()

    if args.memory_snapshot:
        torch.cuda.memory._record_memory_history(max_entries=200_000)
    torch.cuda.reset_peak_memory_stats(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = HookedOptimizer(
        torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, fused=True), model.named_parameters()
    )
    g = torch.Generator(device=dev).manual_seed(0)
    ids = torch.randint(0, vocab, (args.batch_size, args.seq_len), device=dev, generator=g)

    def train_step() -> None:
        opt.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    tokens_per_step = args.batch_size * args.seq_len
    tp = measure_throughput(
        train_step, n_steps=args.steps, warmup=args.warmup, tokens_per_step=tokens_per_step
    )
    report["throughput"] = tp
    report["memory"] = gpu_memory_report(dev.index or 0)
    report["footprint_bytes"] = {
        "parameters": _bytes(model.parameters()),
        "gradients": _bytes(p.grad for p in params if p.grad is not None),
        "optimizer_state": _bytes(
            v for s in opt.state.values() for v in s.values() if isinstance(v, torch.Tensor)
        ),
    }
    report["num_parameters"] = sum(p.numel() for p in model.parameters())
    report["optimizer_steps_applied"] = opt.applied_updates
    if args.memory_summary:
        report["memory_summary"] = torch.cuda.memory_summary(dev)
    if args.memory_snapshot:
        snap = torch.cuda.memory._snapshot()
        Path(args.memory_snapshot).write_bytes(pickle.dumps(snap))
        torch.cuda.memory._record_memory_history(enabled=None)
        report["memory_snapshot"] = args.memory_snapshot
    report["status"] = "MEASURED"
    report["measured_on"] = torch.cuda.get_device_name(dev)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "memory_summary"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
