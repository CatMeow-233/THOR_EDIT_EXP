#!/usr/bin/env bash
# Run the full GPU validation on ONE host with ONE RTX 5090. Run it separately on each machine.
# Usage: bash scripts/gpu/run_home.sh [hf_model_id_or_path] [max_steps]
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
STEPS="${2:-50}"
OUT="runs/gpu/$(hostname)"
mkdir -p "${OUT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRL_EXPERIMENTAL_SILENCE=1

echo "== 1/6 CUDA smoke test"
python scripts/gpu/cuda_smoke_test.py --out "${OUT}/cuda_smoke.json"

echo "== 2/6 BF16 SFT"
python examples/sft_tiny.py --model "${MODEL}" --bf16 --max-steps "${STEPS}" --output-dir "${OUT}/sft"

echo "== 3/6 GRPO math RLVR"
python examples/grpo_math_tiny.py --model "${MODEL}" --bf16 --max-steps "${STEPS}" --num-generations 8 \
  --output-dir "${OUT}/grpo"

echo "== 4/6 RLOO math RLVR"
python examples/rloo_math_tiny.py --model "${MODEL}" --bf16 --max-steps "${STEPS}" --num-generations 8 \
  --output-dir "${OUT}/rloo"

echo "== 5/6 FFN expert training"
python scripts/gpu/expert_train.py --model "${MODEL}" --layers 0-3 --expert smoke --bf16 \
  --max-steps "${STEPS}" --output-dir "${OUT}/experts"

echo "== 6/6 full-step throughput + memory"
python scripts/gpu/train_step_benchmark.py --model "${MODEL}" --batch-size 4 --seq-len 1024 --steps 20 \
  --memory-summary --memory-snapshot "${OUT}/mem_snapshot.pickle" --out "${OUT}/train_step_benchmark.json"

echo "All reports written to ${OUT}"
