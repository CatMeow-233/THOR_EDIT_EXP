#!/usr/bin/env python
"""Exploratory positive control of the capability gate on CPU (``CAPABILITY.md`` section 5).

CPU tiny-model exploratory positive control (post hoc). Not evidence about 7-8B models; GPU metrics
NOT_TESTED. A thin CLI over :func:`research_rl.capability.calibration.run_calibration`: from the P0
start checkpoint (rebuilt with the pilot's SFT, or ``--start`` a ``start_model.safetensors``), every
seed builds the candidates null_exact, null_noise, good@{0.25,0.5,1.0}, bad_hack and bad_retention,
gives each one first gate verdict with the start as incumbent and measures it with the oracle; the
section-5 pass / fail table is evaluated automatically. Outputs in ``--output-dir``:
``results.json``, ``summary.md``, ``calibration_config.json`` (and ``start_model.safetensors`` when the
start is rebuilt). Nothing is downloaded.

Examples::

    python examples/capability_calibration_tiny.py --smoke --output-dir runs/calibration_smoke
    python examples/capability_calibration_tiny.py --output-dir runs/calibration --jobs 2
    python examples/capability_calibration_tiny.py --output-dir runs/calibration --start runs/pilot_p0/start_model.safetensors

``--smoke`` (one seed, smoke-size model and item counts) is a plumbing check: every rule is
INCONCLUSIVE and its numbers are not results (about 15 s measured on 2 threads). The registered full
run (5 seeds x 7 candidates at the section-5 gate sizes) took 1211 s with ``--jobs 4`` on the 4-core
container (``docs/research_rl/results/calibration_cpu/``). Before it, measured on the 4-core
development container at 2 threads with ``--start`` on development seed 99 (not a registered seed):
one seed took 76 s with the shipped item plan (the ``good`` ladder stopped at its first rung, 500
steps) and 115 s with the plan before the oracle's SFT filter (the 500-step rung's training-side
retention/copy drop was 0.0215, just over the 0.02 margin, so the 1000-step rung was trained). The
rung choice flips on differences of that size, so expect about 75-115 s per seed, roughly 6-10
minutes for the five seeds sequentially, plus about 2.5 min when the start is rebuilt, and up to
about an hour if every ladder runs to its last rung; ``--jobs 2`` roughly halves the per-seed part.
Estimates for the registered seeds, not measurements.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from research_rl.capability.calibration import (
    CalibrationConfig,
    calibration_smoke_config,
    run_calibration,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--smoke", action="store_true", help="CI smoke run: seed 0, smoke-size model, tiny item counts"
    )
    p.add_argument("--output-dir", required=True, help="directory for results.json, summary.md, ...")
    p.add_argument(
        "--config",
        default=None,
        help="JSON file of CalibrationConfig fields (applied before the flags; with --smoke, its pilot / gate "
        "dicts are merged key by key into the smoke's)",
    )
    p.add_argument("--seeds", default=None, help="comma-separated seeds (section 5: 0,1,2,3,4)")
    p.add_argument(
        "--start",
        default=None,
        help="a P0 start_model.safetensors (safetensors only); default: rebuild it with the pilot's SFT",
    )
    p.add_argument("--jobs", type=int, default=None, help="seeds run in this many spawned processes")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> CalibrationConfig:
    overrides: dict[str, Any] = {}
    if args.config is not None:
        overrides.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    if args.seeds is not None:
        overrides["seeds"] = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    if args.jobs is not None:
        overrides["jobs"] = args.jobs
    if args.smoke:
        return calibration_smoke_config(**overrides)
    base = CalibrationConfig().to_dict()
    base.update(overrides)
    return CalibrationConfig.from_dict(base)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = build_config(args)
    results = run_calibration(config, args.output_dir, start_path=args.start)
    out = Path(args.output_dir)
    missing = [
        n for n in ("results.json", "summary.md", "calibration_config.json") if not (out / n).is_file()
    ]
    if missing:
        raise SystemExit(f"missing calibration outputs: {missing}")
    verdicts = {
        k: v["verdict"] for k, v in results["rules"].items() if isinstance(v, dict) and "verdict" in v
    }
    print(
        json.dumps({"output_dir": args.output_dir, "banner": results["banner"], "rules": verdicts}, indent=2)
    )
    return results


if __name__ == "__main__":
    main()
