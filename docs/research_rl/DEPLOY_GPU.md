# Deploying ResearchRL on the two RTX 5090 hosts

Hardware assumption: **two independent machines, one RTX 5090 (32 GB) each**. They are never
treated as one 64 GB device and nothing here uses cross-machine distributed training. Run each
experiment on one host; to use both GPUs, run *different* seeds / studies on the two hosts.

Every number below is **NOT_TESTED** until you produce it on the hardware with these scripts. The
CPU CI never writes GPU numbers; the scripts refuse to (exit code 2, `"status": "NOT_TESTED"`) when
CUDA is unavailable.

| Measurement | Script | Status |
|---|---|---|
| CUDA smoke test (BF16 matmul, masked train step on GPU, peak memory) | `scripts/gpu/cuda_smoke_test.py` | NOT_TESTED |
| BF16 SFT training | `examples/sft_tiny.py --model … --bf16` | NOT_TESTED |
| Real GRPO / RLOO (math RLVR) | `examples/grpo_math_tiny.py`, `examples/rloo_math_tiny.py` | NOT_TESTED |
| FFN expert training (TLC) | `scripts/gpu/expert_train.py` | NOT_TESTED |
| GPU memory analysis | `scripts/gpu/train_step_benchmark.py --memory-summary --memory-snapshot …` | NOT_TESTED |
| Full training-step throughput | `scripts/gpu/train_step_benchmark.py` | NOT_TESTED |

## 1. Environment (on each host)

The RTX 5090 (Blackwell, sm_120) needs a PyTorch build with CUDA ≥ 12.8. The default PyPI wheel of
torch 2.15.0 is built against CUDA 13, which supports it.

```bash
git clone <this repo> && cd THOR_EDIT_EXP && git checkout claude/laughing-sagan-p6qzj6
```

```bash
python3.13 -m venv .venv && source .venv/bin/activate
```

```bash
pip install "torch==2.15.0" && pip install -e ".[dev]"
```

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Log in to Hugging Face only through the CLI / environment (never commit tokens):

```bash
hf auth login
```

Optional sanity check that the CPU suite passes on the host too:

```bash
pytest -m "not integration" -q
```

## 2. CUDA smoke test

```bash
python scripts/gpu/cuda_smoke_test.py --out runs/gpu/cuda_smoke_$(hostname).json
```

Checks: BF16 matmul error vs FP32, a tiny-Llama forward/backward/`HookedOptimizer` step under BF16
autocast, that a `MaskRule`-frozen region stays **bit-identical on the GPU**, and peak memory.
Exit code 0 = passed.

## 3. BF16 SFT

```bash
python examples/sft_tiny.py --model Qwen/Qwen2.5-0.5B-Instruct --bf16 --max-steps 200 \
  --dataset <hf dataset with prompt/completion columns> --output-dir runs/sft
```

`--bf16` enables BF16 mixed precision (TRL `bf16=True`); loading the weights directly in BF16 is
controlled by `model.dtype` in the YAML config (`examples/configs/sft_tiny.yaml`). The run record is
written to `runs/sft/<study_id>/<run_name>/record.json`.

## 4. Real GRPO / RLOO (math RLVR)

```bash
python examples/grpo_math_tiny.py --model Qwen/Qwen2.5-0.5B-Instruct --bf16 \
  --max-steps 200 --num-generations 8 --output-dir runs/grpo
```

```bash
python examples/rloo_math_tiny.py --model Qwen/Qwen2.5-0.5B-Instruct --bf16 \
  --max-steps 200 --num-generations 8 --output-dir runs/rloo
```

The examples generate a synthetic arithmetic set; for real studies build `train_dataset` from your
data (columns `prompt`, `answer`) and pass
`reward_funcs=[as_trl_reward_func(MathVerifier(), answer_key="answer")]` to `build_run`. Edit
`trainer.args` in `examples/configs/{grpo,rloo}_math_tiny.yaml` for real settings (learning rate,
`max_completion_length`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `beta`,
`use_vllm`, `gradient_checkpointing`). Things to watch in `record.json` / `metrics.jsonl`:
`applied_updates` vs `nominal_optimizer_steps` (skipped zero-advantage windows),
`rollout/frac_zero_advantage`, `rollout/policy_lag`, `rewards/*`, `completions/clipped_ratio`.
RLOO with `beta>0` reloads the reference model from the model path.

Memory guide for a 32 GB card (estimates to verify, not measurements): full fine-tuning with AdamW
in BF16 weights + FP32 Adam states needs roughly 12–16 bytes/parameter before activations, so
0.5B–1.5B models fit comfortably; ~3B needs gradient checkpointing and small micro-batches; for ≥7B
use PEFT/LoRA (`peft_config=` in `build_run`) or train only selected FFN experts/channels.

## 5. FFN expert training (TLC)

```bash
python scripts/gpu/expert_train.py --model Qwen/Qwen2.5-0.5B-Instruct --layers 12-23 \
  --expert math --dataset <hf prompt/completion dataset> --bf16 --max-steps 200 \
  --output-dir runs/experts --export-merged
```

Only the chosen expert's FFN weights are trainable; the script asserts bit-exactly that the shared
backbone did not change, saves `expert_<name>.safetensors` (with compatibility metadata) and, with
`--export-merged`, a vanilla HF checkpoint for inference. Train different experts on the two hosts
in parallel (same base model, different `--expert` / data); expert files load on either host with
`ExpertManager.load_expert`, which validates the backbone fingerprint.

## 6. GPU memory analysis and full-step throughput

```bash
python scripts/gpu/train_step_benchmark.py --model Qwen/Qwen2.5-0.5B-Instruct \
  --batch-size 4 --seq-len 1024 --steps 20 --warmup 3 --memory-summary \
  --memory-snapshot runs/gpu/mem_snapshot.pickle --out runs/gpu/bench_$(hostname).json
```

Reports tokens/s and seconds/step over complete optimizer steps (forward + backward + clip +
`HookedOptimizer` AdamW step, CUDA-synchronised, warm-up excluded), peak allocated/reserved memory and
the parameter / gradient / optimizer-state footprint. Open the snapshot at
<https://pytorch.org/memory_viz>. Add `--gradient-checkpointing` to compare activation memory.

To store measured GPU metrics in an experiment record (only possible on CUDA):

```python
from research_rl.utils.profiling import gpu_memory_report
run.recorder.record_gpu_metrics(gpu_memory_report())
```

## 7. Everything at once

`scripts/gpu/run_home.sh` runs steps 2–6 with a small model and writes all reports to
`runs/gpu/<hostname>/`:

```bash
bash scripts/gpu/run_home.sh Qwen/Qwen2.5-0.5B-Instruct
```

After the run, replace the `NOT_TESTED` entries in your own notes with the measured JSON reports;
keep the host name and `torch` / CUDA versions that each report records.
