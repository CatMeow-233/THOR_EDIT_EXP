#!/usr/bin/env python
"""P0 capability pilot on CPU (thin CLI over :func:`research_rl.capability.pilot.run_pilot`).

CPU tiny-model pilot (P0). Not evidence about 7-8B models; no GPU numbers. A tiny Llama is built
from scratch in-process (nothing is downloaded), trained by SFT to a low start, then the arms
{grpo, es_full, es_informed, es_random} x {gate_off, gate_on} x seeds x {leak, clean} run behind the
capability gate. Outputs: ``results.json``, ``summary.md``, ``decisions/*.jsonl``,
``start_model.safetensors`` and ``pilot_config.json`` in ``--output-dir``.

Examples::

    python examples/capability_pilot_tiny.py --smoke --output-dir runs/pilot_smoke
    python examples/capability_pilot_tiny.py --output-dir runs/pilot_p0 --jobs 4
    python examples/capability_pilot_tiny.py --arms grpo,es_informed+gate_on --variants leak --seeds 0,1,2

The full default configuration (48 arms) took about 11 s per block per arm on the development
container, i.e. roughly 20-40 minutes with ``--jobs 4`` on 4 cores and 1.5-2.5 hours sequentially
(extrapolated from the measured block time; see ``PilotConfig``); every run times one block per
operator first and prints its measured ETA. At the default configuration the measured per-block
T_rule effect is far below what the oracle resolves, so the gate's false-accept / false-reject
rates are expected to be not evaluable; ``results.json`` (``gate_exercise``), the warnings and
``summary.md`` say so whenever that happens.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from research_rl.capability.pilot import (
    GATES,
    OPERATORS,
    PilotConfig,
    expected_outputs,
    run_pilot,
    smoke_config,
)


def _arms(text: str) -> list[str]:
    """``grpo,es_informed+gate_on`` -> every listed ``operator+gate`` (a bare operator means both gates)."""
    out: list[str] = []
    for token in (t.strip() for t in text.split(",")):
        if not token:
            continue
        if "+" in token:
            out.append(token)
        elif token in OPERATORS:
            out.extend(f"{token}+{g}" for g in GATES)
        else:
            raise argparse.ArgumentTypeError(f"unknown arm {token!r}; operators: {OPERATORS}, gates: {GATES}")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="CI smoke run: 1 seed, 2 arms, leak variant, < 60 s")
    p.add_argument("--output-dir", required=True, help="directory for results.json, summary.md, ...")
    p.add_argument(
        "--config", default=None, help="JSON file of PilotConfig fields (applied before the flags)"
    )
    p.add_argument("--seeds", default=None, help="comma-separated seeds, e.g. 0,1,2")
    p.add_argument(
        "--arms",
        default=None,
        help="comma-separated operator or operator+gate, e.g. grpo,es_informed+gate_on",
    )
    p.add_argument("--variants", default=None, help="comma-separated subset of leak,clean")
    p.add_argument(
        "--updates", type=int, default=None, help="operator updates per arm (ES too unless es_updates is set)"
    )
    p.add_argument(
        "--jobs", type=int, default=None, help="arms run in this many spawned processes (deterministic)"
    )
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> PilotConfig:
    overrides: dict[str, Any] = {}
    if args.config is not None:
        overrides.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    if args.seeds is not None:
        overrides["seeds"] = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    if args.arms is not None:
        overrides["arms"] = tuple(_arms(args.arms))
    if args.variants is not None:
        overrides["variants"] = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    if args.updates is not None:
        overrides["updates"] = args.updates
    if args.jobs is not None:
        overrides["jobs"] = args.jobs
    if args.smoke:
        return smoke_config(**overrides)
    base = PilotConfig().to_dict()
    base.update(overrides)
    return PilotConfig.from_dict(base)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = build_config(args)
    results = run_pilot(config, args.output_dir)
    missing = [str(f) for f in expected_outputs(args.output_dir, config) if not f.is_file()]
    if missing:
        raise SystemExit(f"missing pilot outputs: {missing}")
    verdicts = {
        k: v["verdict"] for k, v in results["criteria"].items() if isinstance(v, dict) and "verdict" in v
    }
    print(
        json.dumps(
            {"output_dir": args.output_dir, "banner": results["banner"], "criteria": verdicts}, indent=2
        )
    )
    return results


if __name__ == "__main__":
    main()
