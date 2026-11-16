"""PyTorch + transformers + PEFT baseline for rrc_bench (rrc/docs/DESIGN.md §15.4).

Runs the same workload as ``rrc_bench`` and writes a JSON document with the same keys:

* base model in bf16 (``--dtype``), ``attn_implementation="sdpa"``, loaded from a local directory
  only (``local_files_only=True``, ``HF_HUB_OFFLINE=1``; never ``trust_remote_code``);
* LoRA with the same r / alpha / target modules, dropout 0, adapters in fp32 (checked, not assumed);
* ``micro_batch`` rows x ``seq_len`` synthetic random token ids per micro-batch, one document per
  row, every position labelled (``labels = input_ids``), ``grad_accum`` micro-batches per step;
* gradient checkpointing on or off (``--gradient-checkpointing``; on corresponds to rrc's
  ``--recompute full``, off to ``--recompute none``), stated in the output;
* ``torch.optim.AdamW`` with the same lr / betas / eps / weight decay (``--adamw-impl``: fused by
  default on CUDA, the single-tensor loop elsewhere), ``clip_grad_norm_(max_norm)``, constant LR.

tokens/s = tokens per step (micro_batch * seq_len * grad_accum) / mean measured step time, where a
step time is the ``torch.cuda.Event`` time between the start of the first forward and the end of
``optimizer.step``. Step times are reported as mean / p50 / p90 (nearest rank) / min / max.

``measured_on`` names the GPU only for a CUDA run. ``--device cpu`` is a smoke test of this script
(argument parsing, model loading, the training step, the JSON shape); its output says so and holds
no timing claims.

    python rrc/bench/torch_baseline.py --model /models/my-8b --seq-len 2048 --micro-batch 1 \\
        --gradient-checkpointing on --out torch_gc_on.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import sys
import time
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PyTorch + PEFT LoRA training-step baseline (same JSON as rrc_bench)")
    p.add_argument("--model", required=True, help="local HF checkpoint directory (config.json + safetensors)")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--dtype", choices=["bf16", "f32"], default="bf16", help="base weights / activations")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--targets", default="all-linear", help="all-linear or a comma list of q_proj,k_proj,...")
    p.add_argument("--gradient-checkpointing", choices=["on", "off"], required=True)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--adamw-impl", choices=["auto", "fused", "foreach", "single"], default="auto",
                   help="auto: fused on CUDA, single-tensor loop elsewhere")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda", help="cuda, cuda:N or cpu (cpu = smoke test only)")
    p.add_argument("--out", default=None, help="write the JSON here (default: stdout)")
    p.add_argument("--label", default=None)
    a = p.parse_args(argv)
    for name in ("seq_len", "micro_batch", "grad_accum", "steps", "lora_r"):
        if getattr(a, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be >= 1")
    if a.seq_len < 2:
        p.error("--seq-len must be >= 2")
    if a.warmup < 0:
        p.error("--warmup must be >= 0")
    if a.lora_alpha <= 0:
        p.error("--lora-alpha must be > 0")
    if a.targets == "all-linear":
        a.target_list = list(ALL_LINEAR)
    else:
        a.target_list = [t for t in a.targets.split(",") if t]
        bad = [t for t in a.target_list if t not in ALL_LINEAR]
        if bad or not a.target_list:
            p.error(f"--targets: unknown module(s) {bad}; allowed: all-linear or {','.join(ALL_LINEAR)}")
    if not (a.device == "cpu" or a.device == "cuda" or a.device.startswith("cuda:")):
        p.error("--device must be cuda, cuda:N or cpu")
    return a


def stats(x: list[float]) -> dict[str, float] | None:
    if not x:
        return None
    s = sorted(x)
    n = len(s)

    def pct(q: float) -> float:  # nearest rank, as rrc_bench
        r = max(1, min(n, math.ceil(q / 100.0 * n)))
        return s[r - 1]

    return {"mean": sum(s) / n, "p50": pct(50), "p90": pct(90), "min": s[0], "max": s[-1]}


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    import torch
    import transformers
    import peft

    dev = torch.device(a.device)
    is_cuda = dev.type == "cuda"
    if is_cuda and not torch.cuda.is_available():
        print("torch_baseline: --device cuda but torch.cuda.is_available() is False", file=sys.stderr)
        return 2
    impl = a.adamw_impl if a.adamw_impl != "auto" else ("fused" if is_cuda else "single")
    out: dict[str, Any] = {
        "schema": "rrc-bench-v1",
        "tool": "torch_baseline",
        "status": "measured" if is_cuda else "smoke (CPU run of the baseline script: not a benchmark)",
        "error": None,
        "measured_on": None,
        "timing_source": "torch.cuda.Event elapsed time" if is_cuda else "host wall clock (not a GPU measurement)",
        "label": a.label,
        "started_utc": started,
        "command_line": [sys.argv[0]] + (argv if argv is not None else sys.argv[1:]),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        },
        "device": None,
    }
    if is_cuda:
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        out["device"] = {
            "name": prop.name,
            "compute_capability": f"{prop.major}.{prop.minor}",
            "sm_count": prop.multi_processor_count,
            "total_mem_bytes": prop.total_memory,
        }
    torch.manual_seed(a.seed)
    base_dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        a.model, dtype=base_dtype, attn_implementation="sdpa", local_files_only=True, trust_remote_code=False
    )
    cfg = model.config
    model.config.use_cache = False
    lcfg = peft.LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, target_modules=a.target_list, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
    )
    model = peft.get_peft_model(model, lcfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:  # PEFT autocasts bf16 adapters to fp32 by default; make sure, do not assume
        if p.dtype != torch.float32:
            p.data = p.data.float()
    assert trainable and all(p.dtype == torch.float32 for p in trainable), "LoRA adapters must be fp32"
    if a.gradient_checkpointing == "on":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(dev)
    model.train()
    kw: dict[str, Any] = {"lr": a.lr, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": a.weight_decay}
    if impl == "fused":
        kw["fused"] = True
    elif impl == "foreach":
        kw["foreach"] = True
    else:
        kw["foreach"] = False
    opt = torch.optim.AdamW(trainable, **kw)
    g = torch.Generator().manual_seed(a.seed)
    vocab = int(cfg.vocab_size)
    batches = [torch.randint(0, vocab, (a.micro_batch, a.seq_len), generator=g).to(dev) for _ in range(a.grad_accum)]
    tokens_per_step = a.micro_batch * a.seq_len * a.grad_accum
    gpu_s: list[float] = []
    wall_s: list[float] = []
    losses: list[float] = []
    gnorms: list[float] = []
    used_max = None
    total = a.warmup + a.steps
    for s in range(total):
        if is_cuda and s == a.warmup:
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
        if is_cuda:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
        w0 = time.perf_counter()
        step_loss = torch.zeros((), device=dev)
        for x in batches:
            loss = model(input_ids=x, labels=x).loss / a.grad_accum  # every micro-batch has seq_len - 1 targets per row
            loss.backward()
            step_loss += loss.detach()
        gn = torch.nn.utils.clip_grad_norm_(trainable, a.max_grad_norm) if a.max_grad_norm > 0 else None
        opt.step()
        opt.zero_grad(set_to_none=True)
        if is_cuda:
            ev1.record()
            torch.cuda.synchronize(dev)
            gpu_s.append(ev0.elapsed_time(ev1) / 1000.0)
            free, tot = torch.cuda.mem_get_info(dev)
            used_max = max(used_max or 0, tot - free)
        wall_s.append(time.perf_counter() - w0)
        losses.append(float(step_loss))
        if gn is None:
            gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in trainable if p.grad is not None))
        gnorms.append(float(gn))
        print(f"torch_baseline: step {s + 1}/{total}{' (warmup)' if s < a.warmup else ''} loss {losses[-1]:.6f}",
              file=sys.stderr)
    meas_gpu = gpu_s[a.warmup:]
    meas_wall = wall_s[a.warmup:]
    timing_valid = is_cuda and len(meas_gpu) == a.steps
    if timing_valid:
        out["measured_on"] = out["device"]["name"]
    mean = sum(meas_gpu) / len(meas_gpu) if timing_valid else None
    out["config"] = {
        "model": a.model,
        "model_type": getattr(cfg, "model_type", None),
        "num_layers": getattr(cfg, "num_hidden_layers", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "vocab_size": vocab,
        "base_dtype": a.dtype,
        "seq_len": a.seq_len,
        "micro_batch": a.micro_batch,
        "grad_accum": a.grad_accum,
        "tokens_per_step": tokens_per_step,
        "warmup_steps": a.warmup,
        "measured_steps": a.steps,
        "data": "synthetic random token ids, one document per row, every position labelled, no padding",
        "lora": {
            "r": a.lora_r,
            "alpha": a.lora_alpha,
            "dropout": 0.0,
            "target_modules": a.target_list,
            "adapter_dtype": "f32",
            "trainable_params": sum(p.numel() for p in trainable),
        },
        "recompute": "full" if a.gradient_checkpointing == "on" else "none",
        "gradient_checkpointing": a.gradient_checkpointing == "on",
        "attention": f"transformers attn_implementation={model.config._attn_implementation}",
        "ce_chunk": None,
        "optimizer": {
            "type": "AdamW",
            "impl": f"torch.optim.AdamW ({impl}), torch.nn.utils.clip_grad_norm_",
            "lr": a.lr,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": a.weight_decay,
            "max_grad_norm": a.max_grad_norm,
            "lr_schedule": "constant",
        },
    }
    out["results"] = {
        "tokens_per_s": tokens_per_step / mean if mean else None,
        "step_time_s": stats(meas_gpu) if timing_valid else None,
        "wall_step_time_s": stats(meas_wall) if timing_valid else None,
        "smoke_wall_step_time_s": None if timing_valid else (sum(meas_wall) / len(meas_wall) if meas_wall else None),
        "losses": losses,
        "grad_norms": gnorms,
        "peak_memory": {
            "torch_max_allocated_bytes": torch.cuda.max_memory_allocated(dev) if is_cuda else None,
            "torch_max_reserved_bytes": torch.cuda.max_memory_reserved(dev) if is_cuda else None,
            "torch_memory_note": "caching allocator peaks over the measured steps (allocated = live tensors, "
                                 "reserved = held by the allocator)",
            "device_used_max_sampled_bytes": used_max,
            "device_used_note": "torch.cuda.mem_get_info total - free, sampled after every step: device-wide, "
                                "not a true peak",
            "model_workspace_bytes": None,
        },
    }
    out["gemm"] = None
    text = json.dumps(out, indent=1)
    if a.out:
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            f.write(text + "\n")
        os.replace(tmp, a.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
