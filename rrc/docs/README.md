# rrc — building and testing

`rrc` is the C/CUDA LoRA training engine described in [DESIGN.md](DESIGN.md). This page covers
what exists after stage A: the build, the frozen backend interface, the CPU reference backend,
the JSON / safetensors / config loaders, and their tests.

Status: everything listed here is **TESTED (CPU)**, except the nvcc header check, which is
**COMPILED-ONLY** (no GPU in the development environment). Nothing here makes a performance claim.

## Requirements

* gcc 13 (C11, OpenMP), GNU make; no other C dependencies (libc, libm, pthreads, OpenMP only).
* For the Python tests: python3 with torch, transformers 5.23, safetensors (no network needed;
  run with `HF_HUB_OFFLINE=1`).
* Optional: a CUDA 12.8+ toolkit for the nvcc compile check (`CUDA_HOME` or `RRC_NVCC`).

## Build

```sh
make -C rrc cpu                 # build/librrc.a and build/rrc_inspect (CPU backend)
make -C rrc test                # build and run the C unit tests (build/rrc_tests)
make -C rrc test CUDA_HOME=/usr/local/cuda        # + nvcc compile check of all headers (sm_120)
make -C rrc test RRC_NVCC=/path/to/bin/nvcc       # same, with an explicit nvcc
make -C rrc cuda CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_120   # compile cuda/*.cu (later stages)
make -C rrc clean
```

Everything is built out of tree in `rrc/build/` (gitignored). Warnings are errors
(`-std=c11 -O2 -Wall -Wextra -Werror -fopenmp`). nvcc builds use `-arch=sm_120` plus
`-gencode arch=compute_120,code=compute_120` (PTX for forward compatibility).

## Tests

```sh
make -C rrc clean cpu test                       # C unit tests
HF_HUB_OFFLINE=1 python -m pytest rrc/tests -q   # from the repository root
```

* `rrc/tests/c/` — one binary, `build/rrc_tests [substring]` runs the matching tests:
  * `common`: status/error reporting; bf16 conversion exhaustive over all 65536 bf16 patterns,
    stratified f32 patterns against an independent double-precision RNE reference, boundaries.
  * `json`: strict parser (every malformed form rejected with line:col), UTF-8 / `\u` surrogate
    pairs, int64 limits, depth limit, mutation fuzzing, writer round trip.
  * `safetensors`: writer/reader round trip, every malformed-header class, sharded index
    checks, exhaustive f16 widening, and interop with the Python `safetensors` package (a file
    written by Python is read by C and vice versa; bf16 rounding is checked bit-for-bit against
    torch). The interop sub-test is skipped with a message if python3/torch/safetensors are absent.
  * `config`: defaults per family, rope spellings and precedence, every refused value, expected
    tensor set and the tensor-set checker.
  * `kernels`: the shared `kernels_common` element functions (online logsumexp merge, edge cases).
  * `ops`: every op on both CPU backends (double and fp32 accumulation) against naive
    double-precision references, all GEMM transpose and dtype combinations, batched/broadcast/
    interleaved GEMM, precondition and aliasing errors, AdamW against `torch.optim.AdamW`
    itself (moments bit-identical; parameters within one ulp where torch's vectorized sqrt
    rounds differently; skipped without python3/torch), and every op bit-identical for 1 and 4
    OpenMP threads at sizes that run every parallel loop.
  * `grad`: every backward op (rmsnorm, rope, attention softmax, swiglu, cross entropy, GEMM
    transposes) against central finite differences of double-precision forward references.
* `rrc/tests/test_config_loading.py` — tiny random checkpoints from `rrc/tools/make_tiny_model.py`
  (llama, llama tied, llama3 rope scaling, qwen2, qwen2 tied, qwen3, qwen3 tied, qwen3 with
  head_dim != hidden/heads and attention biases, a bf16 sharded checkpoint) loaded by
  `build/rrc_inspect` and compared with transformers: derived config, biases and q/k norm
  presence, exact tensor set and shapes, per-tensor sums, RoPE inv_freq and cos/sin tables; a
  table of config.json variants checks precedence/default rules and every refusal.
  `rrc/tests/conftest.py` builds the C code once per session (`make -C rrc cpu`).

## Tools

* `build/rrc_inspect [--config-only] [--allow-unknown-keys] [--positions p0,p1,...] <model_dir>`
  prints the derived config, RoPE frequencies/tables, expected tensors, checkpoint tensors and
  the tensor-set check as JSON.
* `python rrc/tools/make_tiny_model.py --variant qwen3 --out DIR` (or `--family llama --tied
  --llama3-rope`) writes a tiny random fp32 checkpoint with `save_pretrained`.

## Source map (stage A)

| path | contents |
| --- | --- |
| `include/rrc/common.h` | `rrc_status`, thread-local last error with file:line, `RRC_CHECK*`, dtypes, bf16<->f32 (`RRC_HD`) |
| `include/rrc/json.h`, `src/json.c` | strict JSON DOM parser and writer |
| `include/rrc/safetensors.h`, `src/safetensors.c` | mmap reader (single + sharded), writer |
| `include/rrc/config.h`, `src/config.c` | config.json rules, expected tensors, RoPE inv_freq/tables |
| `include/rrc/tensor.h`, `src/tensor.c` | `rrc_buf` device buffers and views |
| `include/rrc/backend.h`, `src/ops.c` | the frozen op table and its host-side checks |
| `kernels_common/rrc_kc*.h` | per-element / per-row math shared by gcc and nvcc |
| `src/cpu_backend.c` | CPU reference backend (OpenMP) |
| `tests/cuda/headers_check.cu` | nvcc compile check of all headers and every `RRC_HD` function |
