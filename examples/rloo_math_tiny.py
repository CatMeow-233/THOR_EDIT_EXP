#!/usr/bin/env python
"""Math RLVR with RLOO (REINFORCE + leave-one-out baseline) through ResearchRL.

Same flags, data and verifier as ``grpo_math_tiny.py`` (see its docstring for the CPU-smoke vs CUDA
usage); only the algorithm and config (``configs/rloo_math_tiny.yaml``) differ. RLOO's default
``beta=0.05`` KL penalty re-loads the reference model from the model path, which works here because
the tiny model is written to disk first. CPU numbers say nothing about real model quality.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grpo_math_tiny import main as _main


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return _main(argv, algorithm="rloo")


if __name__ == "__main__":
    main()
