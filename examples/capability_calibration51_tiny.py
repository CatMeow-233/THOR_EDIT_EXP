#!/usr/bin/env python
"""Follow-up 5.1 of the gate positive control on CPU: the tier-1 alarm (``CAPABILITY.md`` section 5.1).

CPU tiny-model exploratory follow-up (post hoc). Not evidence about 7-8B models; GPU metrics
NOT_TESTED. A thin CLI over :func:`research_rl.capability.calibration_alarm.run_calibration_51`: from
the P0 start checkpoint (rebuilt with the pilot's SFT, or ``--start`` a ``start_model.safetensors``),
every seed builds ``hack_replay`` and ``good_hack``, gives each one tier-2 first verdict with the start
as incumbent, feeds a ``DisagreementAlarm()`` 3 incumbent + 16 candidate reports (and the null stream:
the incumbent in all 19), measures the oracle and evaluates the 5.1 pass / fail table. Outputs in
``--output-dir``: ``results.json``, ``summary.md``, ``calibration51_config.json`` (and
``start_model.safetensors`` when the start is rebuilt). Nothing is downloaded.

Examples::

    python examples/capability_calibration51_tiny.py --smoke --output-dir runs/calibration51_smoke
    python examples/capability_calibration51_tiny.py --output-dir runs/calibration51 --jobs 2 \\
        --start runs/pilot_p0/start_model.safetensors

``--smoke`` (one seed, the section-5 smoke start and item counts, the 5.1 stream shape) is a plumbing
check: every rule is INCONCLUSIVE and its numbers are not results (about 11 s measured on 2 threads).
The registered run (seeds 0-4, ``--start`` the P0 start, ``--jobs 4``) has been run once (423 s on the
4-core container); results in ``docs/research_rl/results/calibration51_cpu/`` (see ``CAPABILITY.md`` 4.5).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from research_rl.capability.calibration_alarm import (
    CalibrationAlarmConfig,
    calibration51_smoke_config,
    run_calibration_51,
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
        help="JSON file of CalibrationAlarmConfig fields (applied before the flags; with --smoke, its pilot / "
        "gate dicts are merged key by key into the smoke's)",
    )
    p.add_argument("--seeds", default=None, help="comma-separated seeds (section 5.1: 0,1,2,3,4)")
    p.add_argument(
        "--start",
        default=None,
        help="a P0 start_model.safetensors (safetensors only); default: rebuild it with the pilot's SFT",
    )
    p.add_argument("--jobs", type=int, default=None, help="seeds run in this many spawned processes")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> CalibrationAlarmConfig:
    overrides: dict[str, Any] = {}
    if args.config is not None:
        overrides.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    if args.seeds is not None:
        overrides["seeds"] = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    if args.jobs is not None:
        overrides["jobs"] = args.jobs
    if args.smoke:
        return calibration51_smoke_config(**overrides)
    base = CalibrationAlarmConfig().to_dict()
    base.update(overrides)
    return CalibrationAlarmConfig.from_dict(base)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = build_config(args)
    results = run_calibration_51(config, args.output_dir, start_path=args.start)
    out = Path(args.output_dir)
    missing = [
        n for n in ("results.json", "summary.md", "calibration51_config.json") if not (out / n).is_file()
    ]
    if missing:
        raise SystemExit(f"missing 5.1 outputs: {missing}")
    verdicts = {
        k: v["verdict"] for k, v in results["rules"].items() if isinstance(v, dict) and "verdict" in v
    }
    print(
        json.dumps({"output_dir": args.output_dir, "banner": results["banner"], "rules": verdicts}, indent=2)
    )
    return results


if __name__ == "__main__":
    main()
