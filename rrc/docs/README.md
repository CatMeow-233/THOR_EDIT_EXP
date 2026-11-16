# rrc — building and testing

`rrc` is the C/CUDA LoRA training engine described in [DESIGN.md](DESIGN.md). This page covers
what exists so far: the build, the frozen backend interface, the CPU reference backend, the JSON /
safetensors / config loaders (stage A); the model forward / backward with LoRA and the SFT training
driver `rrc_train` with its data tools (stage B, model line; CPU backend only); and their tests.

Status: everything listed here is **TESTED (CPU)**, except the nvcc header check, which is
**COMPILED-ONLY** (no GPU in the development environment). Nothing here makes a performance claim.

## Requirements

* gcc 13 (C11, OpenMP), GNU make; no other C dependencies (libc, libm, pthreads, OpenMP only).
* For the Python tests: python3 with torch, transformers 5.23, safetensors (no network needed;
  run with `HF_HUB_OFFLINE=1`).
* Optional: a CUDA 12.8+ toolkit for the nvcc compile check (`CUDA_HOME` or `RRC_NVCC`).

## Build

```sh
make -C rrc cpu                 # build/librrc.a, rrc_inspect, rrc_fwdbwd, rrc_train, rrc_data (CPU backend)
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
* `rrc/tests/test_model_vs_peft.py` (stage B) — `build/rrc_fwdbwd` against transformers + PEFT:
  base logits / hidden states / HF's own loss for every tiny variant; logits, loss and every LoRA
  gradient with randomized non-zero A and B, fp32 (tight) and bf16 (tolerance anchored to HF's own
  bf16 error), both CPU backends, packed rows checked against per-document HF runs, small attention
  and LM-head chunks, both loss reductions, empty micro-batches and windows, upper-layer-only
  adapters, init, refusals. The C suite `model` adds finite-difference gradient checks.
* `rrc/tests/test_train_vs_torch.py` (stage B, training driver) — `build/rrc_train` against
  PyTorch + transformers + PEFT: the LR sequence against `transformers.get_scheduler` (exact, every
  schedule, warmup 0 / >= total / ratio, total 1), clipping against `torch.nn.utils.clip_grad_norm_`
  (the coefficient bit for bit, AdamW's first moment against the clipped torch gradient), an 8-step
  trajectory (grad accumulation 2, clipping active on some steps, warmup + cosine, weight decay, two
  epoch boundaries; both CPU backends and both loss reductions) against a `torch.optim.AdamW(foreach=False)` + PEFT loop fed the batches rrc
  trained on (`rrc/tools/torch_reference.py`; fp32 tight, bf16 anchored to HF-bf16's own error),
  bit-identical resume (every checkpoint file, metrics) with the prefetch thread on, loading the
  checkpoint with `peft.PeftModel.from_pretrained` (logits equal rrc's), `prepare_sft.py` with a
  tokenizer trained in the test, every malformed `.rrcbin`, packing determinism and the packing spec,
  config strictness, refusal of an `init_adapter` whose `adapter_config.json` disagrees with `lora`,
  and that every config field is documented below. The C suites `data` and
  `train` cover the reader / loader / optimizer / state file / an end-to-end stop + resume without
  Python.
* `make -C rrc asan` — the C tests under AddressSanitizer + UBSan (build/asan).
* `make -C rrc tsan` — the `data` and `train` C suites (prefetch thread next to the training loop)
  under ThreadSanitizer (build/tsan; `OMP_NUM_THREADS=1` because libgomp is not instrumented).

## Tools

* `build/rrc_inspect [--config-only] [--allow-unknown-keys] [--positions p0,p1,...] <model_dir>`
  prints the derived config, RoPE frequencies/tables, expected tensors, checkpoint tensors and
  the tensor-set check as JSON.
* `build/rrc_fwdbwd --model DIR --batch FILE --out FILE [--lora adapter_model.safetensors] [--dtype f32|bf16]
  [--backend cpu|cpu-f32acc] [--r N --alpha X --targets q_proj,... --layers 0,1] [--attn-chunk N]
  [--ce-chunk N] [--reduction window_tokens|microbatch_mean] [--logits] [--hidden]` runs forward +
  backward of the micro-batches in FILE (`tokens`/`labels`/`position_ids`/`doc_ids`, [M, B, T]) and
  writes logits, the window loss (metadata) and every LoRA gradient with PEFT key names.
* `python rrc/tools/make_tiny_model.py --variant qwen3 --out DIR` (or `--family llama --tied
  --llama3-rope`) writes a tiny random fp32 checkpoint with `save_pretrained`.
* `build/rrc_data check FILE [--vocab V]` validates a `.rrcbin` (against a model vocabulary V) and
  prints `{num_docs, vocab_size, total_tokens, label_tokens, max_len, hash}`; `build/rrc_data pack
  FILE --seq-len T --rows B --count N --out OUT [--seed S] [--no-shuffle] [--truncation drop]
  [--pad-id P] [--no-prefetch] [--hold H] [--start-epoch E --start-index I]` writes the first N
  packed micro-batches exactly as the trainer's loader builds them.
* `python rrc/tools/prepare_sft.py` and `build/rrc_train`: see the next section.

## Training (`rrc_train`)

Status: **TESTED (CPU)** against PyTorch + PEFT on tiny models (above). Nothing has run on a GPU;
there is no CUDA backend in the training path yet, and no performance figure exists (NOT_TESTED).
The `*_time_s` / `*_per_s` metrics are wall-clock measurements of whatever machine runs the job.

```sh
# 1. tokenize (offline, local tokenizer directory only; nothing is downloaded)
python rrc/tools/prepare_sft.py --tokenizer /models/my-model --input train.jsonl --output train.rrcbin
# 2. train
make -C rrc cpu
rrc/build/rrc_train --config train.json
# helpers
rrc/build/rrc_train --config train.json --print-schedule   # LR of every step (no model / data loaded)
rrc/build/rrc_train --print-default-config                 # every field with its default (null = required)
```

`prepare_sft.py` input: one JSON object per line, either `{"messages": [{"role", "content"}, ...]}`
(rendered with the tokenizer's chat template or `--chat-template FILE`; only assistant turns,
including the end-of-turn marker the template puts inside the turn, are targets; the template must
be prefix-stable or the example is refused) or `{"prompt": str, "completion": str}` (prompt with the
tokenizer's special tokens, completion without, then EOS unless `--no-eos`; completion + EOS are the
targets). Other keys are refused. `--vocab-size` overrides the header's `len(tokenizer)`.

`.rrcbin` (little-endian, no padding): bytes 0–7 `RRCBIN01`, 8–15 u64 `num_docs` (≥ 1), 16–23 u64
`vocab_size` (1 … 2³¹−1), 24–31 u64 `total_tokens`, then per document u32 `len` (≥ 1), u32
`tokens[len]`, u8 `label_mask[len]` (0/1). The file size must be exactly 32 + Σ(4 + 5·len). Every field
is checked when the trainer opens the file (truncation, magic, counts, len 0, token ≥ vocab_size or ≥
the model vocabulary, mask values, trailing bytes); see `include/rrc/data.h`.

Packing: per epoch the documents are visited in a seeded order (`seed`, epoch; identity with
`shuffle: false`), epochs are concatenated, and documents are placed greedily into `micro_batch_size`
rows of `seq_len` tokens: a document that does not fit in the rest of a row starts the next row; the
rest of the row is padding (`pad_token_id`, label −100). Positions restart at 0 and doc ids count up
per document within a row (no cross-document attention); documents longer than `seq_len` are cut to
their first `seq_len` tokens (`truncation: "truncate"`) or skipped (`"drop"`). A prefetch thread
builds the next window's micro-batches while the current one trains; the batches do not depend on it.

One optimizer step = `grad_accum_steps` micro-batches: gradients of the window loss (sum of per-token
cross entropy / labelled tokens of the whole window with `window_tokens`), global gradient norm,
`clip_grad_norm_` coefficient fused into AdamW (`torch.optim.AdamW` semantics), LR from the
`transformers.get_scheduler` formula of step − 1 (so with warmup the first step has lr 0, as in HF).

Outputs in `output_dir`:

* `metrics.jsonl`, one line per optimizer step: `step`, `loss` (window loss), `lr` (used for this
  step), `grad_norm` (before clipping), `clip_coef`, `tokens` (labelled tokens of the step),
  `input_tokens` (non-padding tokens), `tokens_seen` (cumulative labelled), `epoch` (data epoch after
  the step), `step_time_s`, `elapsed_s` (since this process started; measured with the monotonic
  clock, never synthesized), `input_tokens_per_s` (`input_tokens / step_time_s`).
* `checkpoint-<step>/` every `save_steps`, at `stop_after_step` and at `max_steps`:
  `adapter_model.safetensors` (fp32, PEFT key names such as
  `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`, metadata `format: pt`) and
  `adapter_config.json` (PEFT `LoraConfig`, `task_type: CAUSAL_LM`), loadable with
  `peft.PeftModel.from_pretrained(base, "checkpoint-<step>")`, plus `rrc_state.bin` (AdamW m / v, step,
  data seed and cursor, tokens seen, hashes of the parameters and the data, the config fingerprint;
  layout in `include/rrc/train.h`). The files are written (as `.tmp` + rename, and fsynced) inside
  `.checkpoint-<step>.tmp/`, which is then renamed into place (DESIGN.md §10 item 23).
* `batches/step-<N>.safetensors` with `dump_batches: true` (tokens / labels / position_ids / doc_ids
  `[grad_accum_steps, micro_batch_size, seq_len]`).

Resume: `"resume_from": "<output_dir>/checkpoint-<K>"` continues bit-identically (CPU, tested) to
the uninterrupted run. The config fields that determine the trajectory must be unchanged (they are
compared with the fingerprint stored in `rrc_state.bin`; output_dir, save_steps, stop_after_step,
prefetch, recompute, dump_batches may change), the data file must be the same (hash), and the
adapter must hold the saved parameters (hash). Lines of `metrics.jsonl` after step K (and a partial
last line) are dropped before appending.

### Config fields

All fields of the JSON object; unknown keys (also inside `lora`) are errors, as are wrong types and
out-of-range values. Required: `model_dir`, `data_path`, `output_dir`, `seq_len`, `max_steps`.

<!-- rrc_train config fields -->
| field | default | meaning |
| --- | --- | --- |
| `model_dir` | required | HF checkpoint directory (config.json + safetensors, single or sharded) |
| `data_path` | required | `.rrcbin` file from `prepare_sft.py` |
| `output_dir` | required | metrics, checkpoints (created if missing) |
| `init_adapter` | null | start from these LoRA weights: an `adapter_model.safetensors` file or a PEFT adapter directory (its `adapter_config.json` must agree with `lora`); null: rrc's seeded init (A uniform, B = 0) |
| `resume_from` | null | a `checkpoint-<K>` directory to continue from (exclusive with `init_adapter`) |
| `overwrite_output_dir` | false | allow a fresh run where `metrics.jsonl` already exists (it is truncated) |
| `dtype` | `"bf16"` | base weights / activations: `"bf16"` or `"f32"` (LoRA, grads, AdamW are always fp32) |
| `backend` | `"cpu"` | `"cpu"` (double accumulation) or `"cpu-f32acc"` (fp32 accumulation) |
| `seq_len` | required | tokens per packed row (≤ the model's positions) |
| `micro_batch_size` | 1 | rows per micro-batch |
| `grad_accum_steps` | 1 | micro-batches per optimizer step (1 … 65536) |
| `max_steps` | required | optimizer steps (also the schedule length) |
| `learning_rate` | 5e-5 | peak LR |
| `lr_scheduler_type` | `"linear"` | `"constant"`, `"constant_with_warmup"`, `"linear"`, `"cosine"` (transformers formulas) |
| `warmup_steps` | 0 | ≥ 1: warmup steps (truncated to an integer); in [0, 1): ratio of `max_steps`, rounded up (transformers 5 semantics) |
| `adam_beta1` | 0.9 | AdamW β1 |
| `adam_beta2` | 0.999 | AdamW β2 |
| `adam_epsilon` | 1e-8 | AdamW ε |
| `weight_decay` | 0.0 | decoupled weight decay (applied to every LoRA parameter) |
| `max_grad_norm` | 1.0 | global L2 clipping threshold; 0 disables |
| `loss_reduction` | `"window_tokens"` | `"window_tokens"` (mean over the labelled tokens of the whole window) or `"microbatch_mean"` (mean of per-micro-batch means) |
| `seed` | 42 | data shuffling seed |
| `shuffle` | true | shuffle documents every epoch |
| `truncation` | `"truncate"` | documents longer than `seq_len`: `"truncate"` (keep the first `seq_len` tokens) or `"drop"` |
| `pad_token_id` | 0 | token id of padding positions (labels there are −100) |
| `lora.r` | 8 | LoRA rank |
| `lora.alpha` | 16 | `lora_alpha` (scaling alpha / r) |
| `lora.use_rslora` | false | scaling alpha / sqrt(r) |
| `lora.dropout` | 0.0 | must be 0 (phase 1) |
| `lora.target_modules` | all seven | list of `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, or `"all-linear"` |
| `lora.layers_to_transform` | null | null = every layer, else a list of layer indices (PEFT semantics) |
| `lora.init_seed` | 0 | seed of rrc's A initialisation (ignored with `init_adapter` / `resume_from`) |
| `attn_chunk` | 256 | query rows per attention chunk (memory bound of the fp32 scores) |
| `ce_chunk` | 1024 | labelled rows per LM-head / cross-entropy chunk |
| `max_positions` | 0 | RoPE table rows; 0 = `max_position_embeddings` |
| `recompute` | `"full"` | `"full"` (save layer inputs, recompute in backward) or `"none"` |
| `prefetch` | true | build the next window's micro-batches on a second thread |
| `save_steps` | 0 | checkpoint every N steps (0: only at the end / `stop_after_step`) |
| `stop_after_step` | 0 | > 0: checkpoint and exit after this step (the schedule still spans `max_steps`) |
| `dump_batches` | false | write the packed batches of every step to `output_dir/batches/` (tests, debugging) |
<!-- end of config fields -->

Example (`train.json`):

```json
{
  "model_dir": "/models/my-8b", "data_path": "train.rrcbin", "output_dir": "runs/sft1",
  "seq_len": 4096, "micro_batch_size": 1, "grad_accum_steps": 16, "max_steps": 1000,
  "learning_rate": 2e-4, "lr_scheduler_type": "cosine", "warmup_steps": 0.03,
  "lora": {"r": 16, "alpha": 32, "target_modules": "all-linear"},
  "save_steps": 100
}
```

(On the CPU backend an 8B model is far too slow to train; the example shows the fields. The CUDA
backend is a later stage.)

## Source map

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
| `include/rrc/model.h`, `src/model.c`, `src/model_exec.c` | model loading; decoder forward / backward with LoRA (stage B, DESIGN.md §10) |
| `include/rrc/lora.h`, `src/lora.c` | LoRA adapter: layout, PEFT names, seeded init, safetensors I/O |
| `src/rrc_fwdbwd.c` | `build/rrc_fwdbwd`: one forward/backward window -> safetensors (used by the pytest comparison) |
| `include/rrc/data.h`, `src/data.c` | `.rrcbin` reader (validation), shuffling, packing, prefetch thread (stage B, step 2) |
| `include/rrc/train.h`, `src/train.c` | LR schedules, clip + AdamW, training config, `rrc_state.bin`, checkpoints, training loop |
| `src/rrc_train.c`, `src/rrc_data.c` | `build/rrc_train`, `build/rrc_data` |
| `tools/prepare_sft.py` | JSONL -> `.rrcbin` with a local tokenizer |
| `tools/torch_reference.py` | PyTorch + PEFT reference training loop and the packing spec in Python (tests) |
