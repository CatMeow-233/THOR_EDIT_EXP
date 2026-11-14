#!/usr/bin/env python
"""CUDA smoke test for one RTX 5090 host (run at home; not executed in CI).

Checks, on a single GPU:
  1. CUDA is visible, device name / capability / memory.
  2. BF16 matmul is numerically sane vs. an FP32 reference.
  3. A tiny Llama forward + backward + HookedOptimizer(AdamW) step in BF16 autocast on GPU,
     with a MaskRule proving frozen elements stay bit-identical on the device.
  4. Peak memory of that step.

Writes a JSON report. Exit code 0 = all checks passed, 2 = CUDA unavailable (report says
NOT_TESTED), 1 = a check failed.

    python scripts/gpu/cuda_smoke_test.py --out runs/gpu/cuda_smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from research_rl.models.parameter_mask import ParameterMask
from research_rl.models.tiny import build_char_tokenizer, build_tiny_llama
from research_rl.optim import HookedOptimizer, MaskRule


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default="runs/gpu/cuda_smoke.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    report: dict = {"script": "cuda_smoke_test", "torch": torch.__version__, "cuda_build": torch.version.cuda}
    if not torch.cuda.is_available():
        report["status"] = "NOT_TESTED"
        report["reason"] = "torch.cuda.is_available() is False"
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 2

    dev = torch.device(args.device)
    props = torch.cuda.get_device_properties(dev)
    report["device"] = {
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "total_memory_gib": round(props.total_memory / 2**30, 2),
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }
    checks: dict = {}

    # 2. BF16 matmul vs FP32 reference
    g = torch.Generator(device=dev).manual_seed(0)
    a = torch.randn(1024, 1024, device=dev, generator=g)
    b = torch.randn(1024, 1024, device=dev, generator=g)
    ref = a @ b
    got = (a.bfloat16() @ b.bfloat16()).float()
    rel = float((got - ref).norm() / ref.norm())
    checks["bf16_matmul_rel_error"] = rel
    checks["bf16_matmul_ok"] = rel < 2e-2

    # 3. tiny model fwd/bwd/step with HookedOptimizer + mask, bf16 autocast
    tok = build_char_tokenizer()
    model = build_tiny_llama(len(tok), seed=0)
    model.to(dev)  # type: ignore[arg-type]
    name = "model.layers.0.mlp.gate_proj.weight"
    w = dict(model.named_parameters())[name]
    m = torch.ones_like(w, dtype=torch.bool)
    m[w.shape[0] // 2 :] = False
    mask = ParameterMask({name: m})
    opt = HookedOptimizer(
        torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1),
        model.named_parameters(),
        [MaskRule(mask)],
    )
    ids = torch.randint(4, len(tok), (8, 64), device=dev, generator=g)
    torch.cuda.reset_peak_memory_stats(dev)
    frozen_before = w.detach()[w.shape[0] // 2 :].clone()
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(3):
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
    torch.cuda.synchronize(dev)
    checks["train_step_seconds_avg"] = (time.perf_counter() - t0) / 3
    checks["loss_finite"] = bool(torch.isfinite(loss).item())
    checks["mask_bit_exact_on_gpu"] = bool(torch.equal(frozen_before, w.detach()[w.shape[0] // 2 :]))
    checks["peak_memory_mib"] = round(torch.cuda.max_memory_allocated(dev) / 2**20, 2)
    # 4. a real TRL step with MaskRule + P2PriorRule: Accelerate moves optimizer/rule state to the GPU
    from research_rl.utils.smoke import trl_rules_smoke

    trl = trl_rules_smoke(out.parent / "cuda_smoke_trl", steps=2, bf16=True)
    checks["trl_rules"] = trl
    report["checks"] = checks
    ok = (
        checks["bf16_matmul_ok"]
        and checks["loss_finite"]
        and checks["mask_bit_exact_on_gpu"]
        and trl["mask_bit_exact"]
        and trl["p2_coefficients_moved"]
        and trl["device"].startswith("cuda")
    )
    report["status"] = "PASSED" if ok else "FAILED"
    report["measured_on"] = "cuda"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
