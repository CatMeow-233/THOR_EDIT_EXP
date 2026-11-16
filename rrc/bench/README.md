# First run on the RTX 5090: selftest, benchmark, baseline

Everything in this directory and everything that runs on a GPU in `rrc` is **COMPILED-ONLY**:
none of it has run on a GPU. Nothing in this repository is a measured performance number. The
commands below produce the first measured numbers. Send back the files listed in step 6, unedited.

Design and rationale: [DESIGN.md §15](../docs/DESIGN.md).

## 0. Requirements on the GPU machine

* NVIDIA driver with CUDA 12.8 or newer (sm_120 needs 12.8+) and a CUDA toolkit with `nvcc`,
  `libcudart`, `libcublasLt` (for example `/usr/local/cuda`).
* gcc 13 (C11, OpenMP) and GNU make.
* For the baseline (step 5): python3 with a CUDA build of torch, transformers 5.23 and peft.
* A local HF checkpoint directory of a supported model (`llama`, `qwen2` or `qwen3`
  `model_type`, bf16 safetensors), called `$MODEL` below. Nothing is downloaded by any of these
  commands; keep `HF_HUB_OFFLINE=1` set.

```sh
export MODEL=/path/to/local/checkpoint     # config.json + *.safetensors
export CUDA_HOME=/usr/local/cuda
export HF_HUB_OFFLINE=1
mkdir -p gpu_run && cd gpu_run             # all outputs go here; run the commands from this directory
R=../rrc                                    # path of the rrc/ directory from here
```

## 1. Record the machine and the source revision

```sh
git -C $R log -1 --format='%H %s' > env.txt
nvidia-smi >> env.txt
nvcc --version >> env.txt
gcc --version | head -1 >> env.txt
python3 -c "import torch, transformers, peft; print(torch.__version__, torch.version.cuda, transformers.__version__, peft.__version__)" >> env.txt
```

## 2. Build

```sh
make -C $R clean
make -C $R cpu test 2>&1 | tail -n 30 > build_cpu_test.log              # CPU tests: must end without an error
make -C $R cuda CUDA_HOME=$CUDA_HOME CUDA_ARCH=sm_120 2>&1 | tail -n 20 > build_cuda.log
ls -la $R/build/rrc_selftest $R/build/rrc_bench $R/build/rrc_train_cuda >> build_cuda.log
```

## 3. Selftest with the report (the most important file)

```sh
$R/build/rrc_selftest --report selftest.json 2>&1 | tee selftest.log
```

The last line is `rrc_selftest: PASS` or `rrc_selftest: FAIL`. It runs, in this order: every op
of the CUDA backend against the CPU backend (random inputs, tolerances per op class), the GEMM
path of every dtype combination (the dense aligned cases must reach cuBLASLt), a pass with a tiny
GEMM scratch cap and a pass with a 3-block grid, then the end-to-end checks: tiny llama, qwen2 and
qwen3 checkpoints written by the program itself, f32 and bf16, logits, loss, every LoRA gradient,
a 20-step training run through the trainer, and bit-for-bit determinism. The end-to-end CPU
references run on the CPU of the machine (a few minutes; the duration has not been measured).

`selftest.json` is rewritten after every stage (`"complete": false` until the end), so if the
program crashes or hangs, the file still holds every finished stage; send it anyway.

Only if step 3 printed FAIL or crashed, also run these (each localizes a different class of
problem) and send their outputs:

```sh
$R/build/rrc_selftest --sync-each-op --no-e2e --report selftest_sync.json > selftest_sync.log 2>&1
$R/build/rrc_selftest --only e2e --e2e-keep --e2e-dir . -v --report selftest_e2e.json > selftest_e2e.log 2>&1
compute-sanitizer --tool memcheck $R/build/rrc_selftest --quick --no-e2e --no-grid-pass > sanitizer.log 2>&1   # if installed
```

## 4. rrc benchmark

`rrc_bench` loads `$MODEL` (bf16 base), adds LoRA (r 16, alpha 32, all seven projections, fp32),
and times warmup + measured optimizer steps (forward + backward + clip + AdamW) on synthetic token
ids with cudaEvents. Run the standard configuration first, then the others (each is a separate
process; a configuration that runs out of memory writes `"status": "error"` with the message: send
that file too).

```sh
B=$R/build/rrc_bench
$B --model $MODEL --seq-len 2048 --micro-batch 1 --warmup 3 --steps 10 --recompute full --out rrc_T2048_full.json
$B --model $MODEL --seq-len 4096 --micro-batch 1 --warmup 3 --steps 10 --recompute full --out rrc_T4096_full.json
$B --model $MODEL --seq-len 1024 --micro-batch 1 --warmup 3 --steps 10 --recompute full --out rrc_T1024_full.json
$B --model $MODEL --seq-len 2048 --micro-batch 1 --warmup 3 --steps 10 --recompute none --out rrc_T2048_none.json
```

Per run it reports tokens/s, step time mean / p50 / p90, peak memory (three different measures,
each labelled in the file), per-step losses and gradient norms, and the GEMM census of one step
(every GEMM shape, its cuBLASLt path or kernel fallback, computed FLOPs). `"measured_on"` names the
GPU only when the times come from cudaEvents on a real device.

## 5. PyTorch + PEFT baseline (same configurations)

Same model, same LoRA, same sequence length and micro-batch, bf16 base, `attn_implementation="sdpa"`,
fused AdamW, gradient checkpointing on (the counterpart of `--recompute full`) and off (`none`):

```sh
T=$R/bench/torch_baseline.py
python3 $T --model $MODEL --seq-len 2048 --micro-batch 1 --warmup 3 --steps 10 --gradient-checkpointing on  --out torch_T2048_gc_on.json
python3 $T --model $MODEL --seq-len 4096 --micro-batch 1 --warmup 3 --steps 10 --gradient-checkpointing on  --out torch_T4096_gc_on.json
python3 $T --model $MODEL --seq-len 1024 --micro-batch 1 --warmup 3 --steps 10 --gradient-checkpointing on  --out torch_T1024_gc_on.json
python3 $T --model $MODEL --seq-len 2048 --micro-batch 1 --warmup 3 --steps 10 --gradient-checkpointing off --out torch_T2048_gc_off.json
```

Do not run the two benchmarks at the same time (they share the GPU and its memory).

## 6. What to send back

```sh
tar czf gpu_run.tgz env.txt build_cpu_test.log build_cuda.log selftest.json selftest.log \
    rrc_*.json torch_*.json $(ls selftest_sync.* selftest_e2e.* sanitizer.log 2>/dev/null)
```

Send `gpu_run.tgz` (or the files themselves). If something failed before a JSON file was
written, send the terminal output instead. Nothing in these files is secret (paths, versions,
device name, numbers); the model directory path appears in the benchmark files.

## Reading the results (for whoever analyses them)

* `selftest.json`: `result`, then `op_passes[].ops[]` (worst error / tolerance per op),
  `op_passes[].gemm` (path per dtype combination, kernel-fallback cases), `e2e.items[]`
  (`dev_err` vs `tol`, `ref_err` = the CPU reference's own error used as the anchor), and
  `e2e.model_gemm_tally` / `e2e.model_gemm_census` (which of the model's GEMMs ran on cuBLASLt;
  `kernel_fallback_flop_share` > 0 means a correct but slow path was taken).
* `rrc_*.json` vs `torch_*.json`: same keys. Compare `results.tokens_per_s` and
  `results.step_time_s` for the same `seq_len` / `recompute` = `gradient_checkpointing`;
  `results.losses` are not comparable (different random tokens); `gemm.census` shows where rrc's
  time can go. The peak memory measures differ by tool and are labelled in the files.
* Known differences of the workloads (by design, not measured): rrc synchronizes the host once per
  micro-batch (loss) and once per step (gradient norm for clipping); rrc's attention is chunked
  GEMM + softmax, not a fused flash kernel; torch uses fused AdamW.

## Smoke tests without a GPU (what CI runs)

`make -C rrc cpu` builds `build/rrc_bench_host` (CPU backend, JSON marked `"status": "smoke"`,
no timing claims) and `build/rrc_selftest_host`; `rrc/tests/test_gpu_tools_smoke.py` runs both
and `torch_baseline.py --device cpu` on a tiny local model for one step (a smoke test of the
script, not a benchmark). With `CUDA_HOME` set, `make -C rrc test` also runs the CUDA mains of
`rrc_selftest` and `rrc_bench` on a fake CUDA runtime on the CPU (HOST-ONLY; no GPU, no timing).
