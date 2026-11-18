#!/usr/bin/env bash
# Runs every example end-to-end on a tiny offline model (CPU, a few steps).
# Used by CI; numbers produced here are smoke-test values, not research results.
set -euo pipefail
OUT="${1:-runs/ci_smoke}"
for ex in sft_tiny dpo_tiny distillation_tiny grpo_math_tiny rloo_math_tiny; do
  echo "::group::examples/${ex}.py"
  python "examples/${ex}.py" --output-dir "${OUT}/${ex}" --max-steps 2
  echo "::endgroup::"
done
python examples/capability_pilot_tiny.py --smoke --output-dir "${OUT}/capability_pilot_tiny"
python examples/capability_calibration_tiny.py --smoke --output-dir "${OUT}/capability_calibration_tiny"
