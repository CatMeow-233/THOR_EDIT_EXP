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

## 8. Capability pilot P1 on a local 7-8B checkpoint (`scripts/gpu/capability_p1.py`)

**Status: NOT_TESTED.** Nobody has run this script on a GPU or on a 7-8B checkpoint. Only its
`--dry-run` mode is covered by the test suite: a tiny Llama built in-process, bf16, on CPU, with
outputs labelled `DRY_RUN_CPU`. A dry run checks the plumbing and validates nothing about large
models. The design is in `docs/research_rl/CAPABILITY.md` (sections 1.14-1.16 and 3).

P1 runs the same pipeline as the CPU pilot (`research_rl/capability/pilot.py`):

1. SFT to a low start;
2. start map;
3. forward-only channel selection;
4. the arms {`grpo`, `es_full`, `es_informed`, `es_random`} x {`gate_off`, `gate_on`} x {`leak`,
   `clean`} x seeds, each behind the capability gate with the oracle and tracker.

The arms run one after another on one GPU, and the start weights are restored bit-exactly between
arms. What is specific to P1:

- **Local and offline only.** `--model-dir` must be an existing directory containing
  `config.json` and either `model.safetensors` or `model.safetensors.index.json`. These are the
  only files `transformers` opens with `use_safetensors=True`, and the script checks exactly them.
  - A directory that holds only pickle weights (`pytorch_model.bin`, `.pt`, `.pth`, `.pkl`,
    `.ckpt`) is refused.
  - A directory whose only safetensors file has a non-standard name (for example
    `consolidated.safetensors`) is refused. Rename or convert it first.
  - The index must be valid JSON with a non-empty `weight_map`. Every shard it lists must be a
    bare `*.safetensors` file name that exists in the directory. An index that points at `.bin`
    shards, or at paths outside the directory, is refused.
  - A `config.json` that sets `transformers_weights` is refused, because it overrides the weight
    file selection and can name a pickle file. A PEFT adapter directory (`adapter_config.json`)
    is refused, because the load would go to its base model. Merge the adapter first.
  - Other pickle files in the directory are listed in `ignored_pickle_files` and never read.
  - The script sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before `transformers` is
    imported. It loads with `local_files_only=True`, `use_safetensors=True` and
    `trust_remote_code=False`. Hub ids are rejected.
- **Tokenizer check.** Every character of the task alphabet and every sampled prompt and answer
  must survive an encode/decode round trip, or the script refuses to run. `max_new_tokens` is set
  to at least the longest answer in tokens + 1.
- **bf16 weights with fp32 masters.** SFT and the GRPO arm train through
  `HookedOptimizer(AdamW, master_weights=True)`.
  - SFT trains the FFN weights of `--sft-layers`.
  - GRPO trains only the informed channel subset: a `MaskRule` from `setup_channel_training`, with
    every other parameter set to `requires_grad=False`.
  - The ES arms use `ChannelGainSpace`; `es_full` is a gain on every channel. P1 refuses
    `es_space="weight"` and `es_full_space="ffn_weights"`.
- **Memory plan, computed and not measured.** The plan is printed before SFT and again after
  channel selection, and written to `memory_plan.json`. Before SFT it is an upper bound: it
  assumes every channel of `--channel-layers` (all layers if unset). After selection it is exact.
  - It counts parameter bytes and the optimizer bytes for the trained subset.
  - The run is refused when the estimate exceeds `--memory-fraction` (default 0.9) of the
    device's total memory; `--allow-over-budget` overrides this.
  - The plan does not include activations, the KV cache, logits, the CUDA context or allocator
    fragmentation.
- **Labels.**
  - `results.json` has `"measured_on"` set to `torch.cuda.get_device_name()`; a dry run writes
    `"DRY_RUN_CPU"`.
  - Every output carries `"script_status": "NOT_TESTED"` and the label `P1_GPU` or `DRY_RUN_CPU`.
    This covers `results.json`, `summary.md`, `memory_plan.json`, `p1_config.json`, the
    `start_delta.safetensors` metadata and the `decisions/<arm>.<label>.jsonl` file names.
  - The script measures GPU peak memory per arm (`torch.cuda.max_memory_allocated`). Throughput
    is `NOT_TESTED`.
- **No CUDA.** Without CUDA the script exits with code 2 and `"status": "NOT_TESTED"`, unless
  `--dry-run` is given. Refusals of the model directory, the tokenizer or the memory budget exit
  with code 3.

### 8.1 Commands (one RTX 5090)

Run the plumbing check first, on the host. It takes about 15 s on 2 CPU threads:

```bash
python scripts/gpu/capability_p1.py --dry-run --output-dir runs/p1_dry
```

Then run P1 on a local Llama-3.1-8B-class checkpoint (download and convert it yourself beforehand;
the script never downloads). Start with one seed and two gated arms to measure time and memory:

```bash
python scripts/gpu/capability_p1.py --model-dir /models/llama-3.1-8b \
  --output-dir runs/p1_s0 --channel-layers 24-25 --sft-layers 24-25 \
  --arms grpo+gate_on,es_informed+gate_on --variants leak --seeds 0
```

If that run fits and its timing is acceptable, run the full design. The section-3 criteria need
at least three seeds:

```bash
python scripts/gpu/capability_p1.py --model-dir /models/llama-3.1-8b \
  --output-dir runs/p1_full --channel-layers 24-25 --sft-layers 24-25 --seeds 0,1,2
```

To run different seeds on the two hosts, give each host its own seeds and output directory.

The values in `P1_GPU_DEFAULTS` are untested starting points, not tuned values: `sft_lr=1e-4`,
`grpo_lr=1e-5`, `grpo_prompts_per_update=8` and `eval_batch_size=64`. Other `PilotConfig` fields
can be overridden with `--config my_p1.json`, a JSON file of `PilotConfig` fields. Use `--sft-steps
0` only when the checkpoint already solves the atomic skills; otherwise the start-retention
warning fires and gate verdicts are not meaningful.

### 8.2 Expected memory (computed, NOT_TESTED)

The planner's formulas, in bytes per element, for bf16 weights:

- **Weights:** 2 per parameter.
- **SFT:** per trained element, fp32 master 4 + AdamW moments 8 + bf16 grad 2 + transient fp32
  grad 4, i.e. 18.
- **GRPO:** the optimizer holds the **whole** gate/up/down tensors of every layer the channel
  selection touches, because the `MaskRule` does not shrink optimizer state. Per element: the 18
  above + the per-step fp32 clone of tracked parameters 4 + bool masks 2 + the bf16 clone used to
  score the incumbent 2, i.e. 26.
- **ES (`ChannelGainSpace`):** a bf16 copy of each selected `down_proj` column + a bf16 clone of
  the touched `down_proj` tensors + small fp32 vectors.
- **Host RAM:** the start-state snapshot plus up to two gate snapshots of the running arm (the
  incumbent and a SHRINK candidate, or the old and new incumbent while promoting). For GRPO a
  snapshot also holds the fp32 masters and AdamW moments.

Worked example, computed with `memory_plan()` on a meta-device model of Llama-3.1-8B shape. The
shape is an assumption: hidden 4096, FFN 14336, 32 layers, vocab 128256, untied embeddings, 8.03 B
parameters. The example uses `--channel-layers 24-25` and `--sft-layers 24-25`, and assumes the
device reports about 31.8 GiB; the script reads the real value.

| Item | Computed |
|---|---|
| bf16 weights | 14.96 GiB |
| SFT phase (FFN of 2 layers) | 5.91 GiB |
| GRPO phase (FFN of 2 layers) | 8.53 GiB |
| `es_full` phase (gain on all 458752 channels) | 7.02 GiB |
| `es_informed` / `es_random` phase | 0.22 GiB |
| Device peak estimate (weights + largest phase) | 23.49 GiB, budget 28.66 GiB at 0.9 |
| Host RAM estimate: start snapshot 3.94 GiB + 2 x the largest gate snapshot (GRPO 4.59 GiB; the gate can hold incumbent and candidate together) | 13.12 GiB |

The remaining room on the device, about 5 GiB in this example, has to hold activations, logits
and the KV cache of generation. None of that is computed.

The planner refuses two settings in this example, before SFT starts:

- **GRPO on 8 layers** (`--channel-layers 20-27`): its phase alone is 34.12 GiB.
- **The global top-k selection** (no `--channel-layers`): the upper bound is all 32 layers, and
  GRPO alone is then 136.5 GiB.

If generation or the GRPO forward runs out of memory, lower `eval_batch_size` and
`grpo_prompts_per_update` with `--config`.

### 8.3 What is NOT_TESTED

Everything about the GPU run is NOT_TESTED:

- loading a real 7-8B checkpoint;
- whether its tokenizer passes the round-trip check;
- bf16 generation and training on CUDA;
- the peak memory, compared with the computed plan;
- run time;
- whether SFT on 2 FFN layers reaches a usable low start (atomic retention pass@1 >= 0.5);
- every hyper-parameter in `P1_GPU_DEFAULTS`.

Replace each with what you measure, and keep `measured_on`, `torch` / CUDA versions and the run's
`results.json`. Results from the CPU pilot or `--dry-run` are never evidence about this run.

### 8.4 Kill criteria (CAPABILITY.md section 3, fixed before any run)

`summary.md` evaluates these automatically (`evaluate_criteria`); fewer than three seeds gives
`INCONCLUSIVE`.

| Question | Success | Kill |
|---|---|---|
| Gate | On planted-leak runs: lower final hack rate and higher T_rule with the gate than without, paired-over-seeds interval excluding zero. On clean runs: T_rule within one standard error of gate-off. | Clean-run T_rule cost exceeds the planted-run T_rule gain, or the gate rejects more than half of the blocks the oracle marks as improved. |
| Subset | Informed subset beats the density-matched random subset on R or T_rule (paired interval excluding zero) at equal A. | No separation across seeds: drop the search, keep channel groups only as the rollback unit. |
| Operator | ES reaches at least half of GRPO's T_rule gain within ten times its forward-pass-equivalent budget (one backward counted as two forwards). | Below that: keep GRPO behind the gate; do not spend the GPU run on ES. |
| Low start | At least one arm exceeds the start checkpoint's pass@k on unseen pairs. | None does: the task is acquisition-limited; report it instead of weakening the task. |

The oracle for the gate's error rates is the exact verifier on a large fresh AUDIT-free evaluation
(T_rule), measured before and after each block.
