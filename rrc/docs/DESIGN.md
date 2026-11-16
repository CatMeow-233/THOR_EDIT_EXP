# rrc — a C/CUDA training engine for LoRA fine-tuning of Llama/Qwen models

`rrc` ("ResearchRL-C") is a standalone training engine. The host code is C11; GPU kernels are
CUDA C++ (`.cu`, compiled by nvcc) behind a C ABI. It does not import or replace the Python
package `research_rl`. The Python/PyTorch stack (transformers + PEFT) is the correctness
reference: every numerical claim of `rrc` is checked against it on tiny random models.

Status vocabulary used throughout code and docs:

* **TESTED (CPU)** — exercised by CI on the CPU backend and compared with PyTorch.
* **COMPILED-ONLY** — CUDA code that nvcc compiles for `sm_120` in CI but that has never been
  executed (no GPU in the development environment or CI).
* **NOT_TESTED** — any performance figure, any RTX 5090 result, anything about real 7–8B
  checkpoints. These become measured values only after `rrc_selftest` / `rrc_bench` are run on a
  real GPU, and the measured output is what gets recorded.

## 1. Scope

Phase 1 (this document): supervised fine-tuning (SFT) with LoRA on a frozen bf16 base.
Phase 2 (later): GRPO (KV-cache sampling, group advantages, TRL-compatible loss), with rewards
from an external reward process that is **disabled by default** (it executes user code).

Target models (exact HF semantics, derived from `transformers` 5.23 source, not from memory):

| family | `model_type` | differences handled |
| --- | --- | --- |
| Llama 3 / 3.1 | `llama` | `rope_scaling.rope_type == "llama3"` (or none); `attention_bias`, `mlp_bias` must be false |
| Qwen2 / Qwen2.5 | `qwen2` | q/k/v bias = true, o bias = false; `use_sliding_window` must be false |
| Qwen3 | `qwen3` | per-head RMSNorm `q_norm`/`k_norm` on `head_dim` before RoPE; `attention_bias` honoured |

Common: RMSNorm (`x * rsqrt(mean(x²) + eps) * w`, computed in fp32), rotary embedding with
HF's `rotate_half` convention, GQA (`num_key_value_heads`), SwiGLU MLP
(`down(silu(gate(x)) * up(x))`), optional `tie_word_embeddings`, `head_dim` from config or
`hidden_size / num_attention_heads`. The loader **fails** on any config field value it does not
implement (e.g. other rope types, sliding window, mlp bias, partial rotary) and on any tensor in
the checkpoint it does not consume or any tensor it expects but cannot find. Both `rope_scaling`
and `rope_parameters` spellings are accepted (checkpoint `config.json` files use either).

Target hardware: one RTX 5090 (32 GB, `sm_120`, CUDA ≥ 12.8). Memory plan for an 8B model:
frozen bf16 base ≈ 16 GB; LoRA r=16 on all seven projections ≈ 42 M params × 16 B (fp32 weight,
grad, Adam m, v) ≈ 0.7 GB; per-layer activation recompute keeps only each layer's input
(`T × hidden` bf16) plus one layer's working set; the LM head + cross-entropy run in token chunks
so `T × vocab` logits never exist at once. These are estimates — NOT_TESTED.

Where the time goes at 8B: ≈ 90 % of FLOPs are the linear layers (cuBLASLt GEMMs). Gains come
from fusion, packing (no padding), avoiding the frozen-weight gradient GEMMs, and keeping the
GPU busy without a Python interpreter — not from the language itself.

## 2. Repository layout

```
rrc/
  Makefile                 make cpu | make cuda CUDA_HOME=... CUDA_ARCH=sm_120 | make test
  include/rrc/             public C headers (C11, usable from C++)
    rrc.h                  umbrella header
    common.h               status codes, logging, RRC_CHECK, dtype enum, bf16<->f32 helpers
    json.h                 minimal JSON parser (DOM, UTF-8, strict)
    safetensors.h          mmap reader (single file + sharded index), writer
    config.h               model config (parsed from HF config.json) + validation
    backend.h              THE op table (frozen interface, see §4)
    tensor.h               device buffers: rrc_buf {backend, dtype, numel, ptr}
    model.h                model weights, LoRA params, forward/backward
    train.h                training config, data loader, optimizer, checkpoint
  src/                     host C code (backend-independent) + CPU backend
    cpu_backend.c          reference ops (fp32 math, OpenMP)
    ...
  cuda/                    CUDA backend: *.cu kernels + cuda_backend.cu (op table)
  kernels_common/          per-element math as static inline functions in headers compiled
                           by both gcc (CPU backend uses them) and nvcc (__host__ __device__)
  tools/                   Python helpers (not on the training hot path)
    prepare_sft.py         tokenize + chat template -> .rrcbin
    make_tiny_model.py     tiny random Llama/Qwen2/Qwen3 checkpoints for tests
    peft_export.py         (optional) checks / converts rrc LoRA output for PEFT
    torch_reference.py     PyTorch+PEFT reference runs used by the tests
  tests/                   C unit tests (tests/c/*.c) + pytest comparisons (tests/test_*.py)
  bench/                   rrc_bench driver + PyTorch baseline script (results NOT_TESTED)
  docs/DESIGN.md, docs/README.md
```

Binaries: `rrc_train` (training CLI), `rrc_selftest` (each backend op vs CPU on random data, then
an end-to-end tiny-model check; the user's first command on the 5090), `rrc_bench`.

## 3. Numerical conventions (each one must match PyTorch+PEFT and has a test)

* **Weights**: base weights stored in the checkpoint dtype (bf16 for real models; fp32 allowed
  so tests can compare at 1e-5). Base weights are frozen: no weight gradients are ever computed
  for them; backward only propagates input gradients through them.
* **LoRA** (PEFT semantics): on each targeted `nn.Linear` W [out, in]:
  `y = x Wᵀ (+ b) + s · (dropout(x) Aᵀ) Bᵀ`, A [r, in], B [out, r], `s = alpha / r`
  (`use_rslora`: `alpha / sqrt(r)`). Init: A ~ kaiming_uniform(a = √5) i.e.
  U(−1/√in, 1/√in)·… exactly as PEFT's `reset_lora_parameters("default")`; B = 0.
  Phase 1 supports `lora_dropout = 0` only (refused otherwise). Target modules from the set
  {q,k,v,o,gate,up,down}_proj. LoRA params, their grads and AdamW state are **fp32** always;
  the CUDA backend may keep a bf16 copy of A/B for the GEMM, refreshed after every step.
* **Loss**: causal LM, labels shifted inside the engine exactly like HF
  (`logits[t]` predicts `labels[t+1]`), `ignore_index = -100`, mean over non-ignored tokens.
  With gradient accumulation the mean is over **all non-ignored tokens of the whole optimizer
  window** (sum of per-token losses / total count across micro-batches), which equals TRL SFT
  with `average_tokens_across_devices`-style token averaging; this choice is configurable
  (`loss_reduction = "window_tokens" | "microbatch_mean"`) and both are tested.
* **Packing**: sequences are packed into rows of length T with per-token `position_ids` that
  restart at 0 at every document and a `doc_id` used to block cross-document attention
  (equivalent to HF/TRL `padding_free` packing with `position_ids`). A document longer than T is
  truncated (policy configurable: `truncate` | `drop`). The label of a document's first token is
  never predicted from the previous document's last token.
* **Attention**: `softmax(q kᵀ · head_dim^-0.5 + mask) v`, causal within each document, softmax
  in fp32. Phase 1 implementation (both backends): per KV-head group, scores for a chunk of query
  rows via (batched) GEMM, fused mask+softmax kernel, then P·V GEMM; backward via the standard
  `dS = P ∘ (dP − rowsum(dP ∘ P))`. Chunk size bounds the fp32 score memory. A fused
  flash-attention kernel is a later optimisation behind the same op.
* **RMSNorm / RoPE / SwiGLU / residual**: computed in fp32 from bf16 (or fp32) inputs, outputs
  stored in the activation dtype. RoPE cos/sin computed in fp32 (double for inv_freq setup) from
  `rope_theta` (+ llama3 scaling) and `position_ids`. Concretely (stage A): `rrc_rope_tables`
  builds f32 tables `[n_pos, head_dim/2]` once per model — `inv` is the double inv_freq rounded
  to f32, the angle is the f32 product `(float)p * inv` (as HF's `position_ids.float() *
  inv_freq`), cos/sin of that f32 angle are evaluated in double and rounded to f32 — and
  `rope_fwd/bwd` index the tables by `position_ids`. Both backends use the same tables, so they do
  not differ by their `cosf/sinf`. With bf16 activations HF rounds some intermediates to bf16
  (e.g. RMSNorm rounds `x·rstd` before multiplying by the weight; RoPE rounds cos/sin and each
  product); rrc rounds once at the output, so bf16 runs are compared with tolerances.
* **AdamW** (PyTorch `torch.optim.AdamW` semantics, `foreach=False`): decoupled weight decay
  `p ← p − lr·wd·p` applied before the Adam update, bias correction with step count starting at 1,
  `eps` added to `sqrt(v̂)`. Defaults: betas (0.9, 0.999), eps 1e-8, wd 0.
  Gradient clipping: global L2 norm over all LoRA grads, `clip_coef = max_norm / (norm + 1e-6)`
  applied when `< 1` (torch `clip_grad_norm_`).
* **LR schedule**: `constant`, `linear` and `cosine` with linear warmup, matching
  `transformers.get_scheduler` formulas step for step.
* **Determinism**: given the same seed, data and backend, two runs are bit-identical on CPU;
  resume from a checkpoint continues bit-identically (CPU, tested).

## 4. Backend interface (`include/rrc/backend.h`) — frozen before parallel work

A backend is a `const rrc_backend *` with a name, device memory functions and op function
pointers. All model/trainer code calls ops only through this table, so the CPU and CUDA backends
share one training loop. Memory is opaque device memory (`void *`); dtypes are
`RRC_F32 | RRC_BF16 | RRC_I32`. Shapes are passed explicitly as `int64_t`. Every op returns
`rrc_status`; shape/dtype preconditions are checked on the host side in the op wrappers (one
place, `src/ops.c`), not inside each backend.

Ops (row-major everywhere; HF `nn.Linear` weight W is [out, in] row-major):

* memory: `alloc`, `free`, `memcpy_h2d`, `memcpy_d2h`, `memcpy_d2d`, `memset0`, `sync`.
* `gemm`: `C[m,n] = alpha · op(A) · op(B) + beta · C` with `op ∈ {N, T}`, A/B in
  {bf16, f32}, C in {bf16, f32}, fp32 accumulation, optional strided batch. This one op covers
  linear forward (`x Wᵀ`), input-gradient (`dy W`), LoRA weight gradients, and attention.
  The CUDA backend implements it with cuBLASLt in a single wrapper that maps row-major to
  column-major and asserts shapes; bf16×bf16→fp32-compute is the only tensor-core path.
* `add_bias`, `bias_grad` (bias is frozen in phase 1 → only needed if a future option trains it;
  forward only now), `embedding_fwd`, `rmsnorm_fwd` / `rmsnorm_bwd` (input grad only; weight is
  frozen; also used for Qwen3 per-head q/k norm), `rope_fwd` / `rope_bwd`,
  `attn_softmax_fwd` / `attn_softmax_bwd` (mask by causal + doc_id, scale, fp32),
  `swiglu_fwd` / `swiglu_bwd`, `add` / `axpy` (residuals, LoRA scale-accumulate),
  `cross_entropy_fwd_bwd` (chunk of logits → per-token loss, dlogits in place, ignore_index),
  `convert` (f32↔bf16), `sum_squares` (grad norm), `scale`, `adamw_step` (fused, fp32).
* All reductions accumulate in fp32 (CPU may use double where noted for reference ops).

Frozen signatures (stage A; full contracts, aliasing rules and preconditions are documented per op
in `include/rrc/backend.h`). Model code calls the `rrc_<op>` wrappers; the table entries take the
same arguments with `const rrc_backend *be` first. Buffers are `rrc_buf {be, dtype, numel, ptr,
owned}` (views via `rrc_view`):

```c
rrc_status rrc_gemm(be, const rrc_gemm_args *g);  /* {trans_a, trans_b, m, n, k, alpha, beta,
                         a, lda, stride_a, b, ldb, stride_b, c, ldc, stride_c, batch}; rrc_gemm2d = batch 1 */
rrc_status rrc_embedding_fwd(be, n_tokens, dim, vocab, ids, table, out);
rrc_status rrc_add_bias(be, rows, cols, x, ldx, bias);
rrc_status rrc_bias_grad(be, rows, cols, dy, ldy, dbias, accumulate);
rrc_status rrc_rmsnorm_fwd(be, rows, dim, ld, x, w, eps, y);
rrc_status rrc_rmsnorm_bwd(be, rows, dim, ld, x, w, eps, dy, dx, accumulate);
rrc_status rrc_rope_fwd(be, n_tokens, n_heads, head_dim, ld, x, pos, cos_tab, sin_tab, n_pos, y);
rrc_status rrc_rope_bwd(be, n_tokens, n_heads, head_dim, ld, dy, pos, cos_tab, sin_tab, n_pos, dx);
rrc_status rrc_attn_softmax_fwd(be, batch, rows, cols, ld, batch_stride, s, scale, row_offset, causal,
                                doc_ids, p);
rrc_status rrc_attn_softmax_bwd(be, batch, rows, cols, ld, batch_stride, p, dp, scale, ds);
rrc_status rrc_swiglu_fwd(be, rows, cols, gate, up, ld_in, h, ld_out);
rrc_status rrc_swiglu_bwd(be, rows, cols, gate, up, ld_in, dh, ld_dh, dgate, dup, ld_out);
rrc_status rrc_add(be, n, a, b, out);
rrc_status rrc_axpy(be, n, alpha, x, y);
rrc_status rrc_scale(be, n, alpha, x);
rrc_status rrc_convert(be, n, src, dst);
rrc_status rrc_sum_squares(be, n, x, out, accumulate);
rrc_status rrc_cross_entropy_fwd_bwd(be, rows, vocab, logits, ld, labels, loss, grad_scale, row_scale,
                                     write_grad);
rrc_status rrc_adamw_step(be, n, param, grad, m, v, const rrc_adamw_hparams *h);
          /* {lr, beta1, beta2, eps, weight_decay, step, grad_scale} */
```

Memory entries of the table: `alloc(be, bytes, &ptr)`, `free(be, ptr)`, `memcpy_h2d/d2h/d2d(be,
dst, src, bytes)`, `memset0(be, ptr, bytes)`, `sync(be)`; there are no stream or handle
arguments — each backend owns one in-order queue, `memcpy_d2h` and `sync` wait for it.

Per-element math lives in `kernels_common/*.h` as `RRC_HD static inline` functions
(`RRC_HD` = `__host__ __device__` under nvcc, empty under gcc). The CPU backend calls them in
loops; the CUDA kernels call them per thread. This way the math is executed and tested on CPU;
only the CUDA launch/indexing code is COMPILED-ONLY.

## 5. Model execution

Forward per layer (activations bf16 on GPU, f32 on CPU unless the test sets otherwise):
`h = x + attn(rmsnorm(x))`, `out = h + mlp(rmsnorm(h))`. Each projection = base GEMM (+bias)
+ optional LoRA (`t = x Aᵀ` [T, r] then `y += s · t Bᵀ`, the second GEMM accumulates into y with
`beta = 1`). Saved for backward per layer: only the layer input `x` (recompute policy `full`,
default) — or everything (`none`, for small models / speed tests). Backward recomputes the
layer forward from the saved input, then runs the layer backward:
for each projection `dx += dy Wᵀ`-style input gradient through the frozen W, plus LoRA:
`dB += s · dyᵀ t`, `dt = s · dy B`, `dA += dtᵀ x`, `dx += dt A`.
LM head: hidden states of all tokens with a label → chunks of `ce_chunk` tokens →
`logits = h W_lmᵀ` (bf16/f32) → fused CE fwd+bwd (dlogits in place, already scaled by
1/total_tokens) → `dh = dlogits W_lm`. lm_head/embeddings are frozen (no LoRA on them in phase 1).

## 6. Training driver

`rrc_train --config train.json` (all fields documented in `docs/README.md`; unknown fields are
errors). Data: `.rrcbin` (little-endian; magic `RRCBIN01`, header {num_docs, vocab_size,
total_tokens}, then per doc {length u32, tokens u32[length], label_mask u8[length]}) produced by
`tools/prepare_sft.py` from JSONL (`messages` with the tokenizer's chat template, or
`prompt`/`completion`), labels = -100 on prompt tokens. The loader packs documents into
`micro_batch × T` rows, shuffles documents per epoch with a seeded PRNG (xoshiro/splitmix,
implemented in C, deterministic), and runs a prefetch thread that builds the next micro-batch
on the host while the GPU computes (CUDA backend: pinned buffers + async copy on a stream).
Metrics: one JSON line per optimizer step to `metrics.jsonl` (step, loss, lr, grad_norm, tokens,
tokens_per_s, elapsed, max memory). Checkpoints (`save_steps`): `adapter_model.safetensors` with
PEFT key names (`base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight`, …) and
`adapter_config.json`, so `peft.PeftModel.from_pretrained` loads them (tested); plus
`rrc_state.bin` (optimizer m/v, step, PRNG + loader position) for exact resume.

## 7. Testing strategy

1. C unit tests (`make test`): JSON, safetensors round trip, config validation (every refused
   field), each CPU op against a naive double-precision reference and finite differences.
2. pytest (`rrc/tests`, run from the repo root): tiny random Llama, Qwen2 and Qwen3 checkpoints
   (fp32, saved with `save_pretrained`, a few hundred KB) → compare with transformers+PEFT:
   logits (1e-5), loss, every LoRA gradient after one micro-batch, parameters after N AdamW
   steps with grad accumulation, clipping and a schedule; packing vs per-document runs; resume
   bit-identity; PEFT loading of the saved adapter; bf16 base weights with looser tolerances.
3. CUDA: `make cuda` with nvcc 12.9 `-arch=sm_120` (plus `compute_120` PTX) in CI —
   COMPILED-ONLY. `rrc_selftest` (user runs on GPU): every op vs the CPU backend on random
   inputs with stated tolerances, then the tiny-model end-to-end comparison, then 20 training
   steps on a tiny model checking the loss decreases.
4. `rrc_bench` + `bench/torch_baseline.py` (same model, LoRA config, sequence length, packing,
   gradient checkpointing, PyTorch's best settings: bf16, SDPA/flash, fused AdamW,
   `torch.compile` optional). Output is a JSON record; nothing is reported as measured until the
   user runs it on the 5090.

## 8. Explicit non-goals of phase 1

Multi-GPU, full-parameter training, QLoRA/4-bit base, LoRA dropout > 0, LoRA on embeddings or
lm_head, sliding-window attention, MoE models, a hand-written tensor-core flash-attention kernel,
tokenization in C (tokenization is done once, offline, in Python).

## 9. Stage A decisions and deviations (recorded with the code that implements them)

Building and testing: see [README.md](README.md). Each item below is a decision this document
left open or a place where the implementation differs from the text above, and why.

1. **`rrc_buf` has an `owned` field** (in addition to {backend, dtype, numel, ptr}) so that
   `rrc_buf_free` can refuse views; an out-of-range `rrc_view` yields `numel = -1`, which every op
   rejects ("invalid view"), so view arithmetic can be written inline.
2. **RoPE uses precomputed f32 cos/sin tables** (`rrc_rope_tables`, see §3) passed to
   `rope_fwd/bwd` with `n_pos`, instead of evaluating cos/sin inside the op: the CPU and CUDA
   backends then agree exactly on the rotation inputs. Positions outside the table are
   `RRC_ERR_RANGE`.
3. **CPU reductions**: `rrc_backend_cpu()` accumulates in double; a second static backend
   `rrc_backend_cpu_f32acc()` runs the same code with fp32 accumulation so the arithmetic a device
   backend uses is also executed and tested. Both use a fixed reduction order (bit-identical for
   any thread count, tested).
4. **GEMM dtypes**: A, B and C are each f32 or bf16 independently (all 8 combinations, as §4
   says). Assumption (not verified here, no GPU): cuBLASLt does not support every mixed
   combination, so the CUDA backend may have to convert operands to a supported combination
   internally. `stride_a/stride_b = 0` (broadcast over the batch) and C slices interleaved within
   rows are part of the contract; overlapping C slices are refused.
5. **Op extensions beyond the §4 list** (all optional to use): `cross_entropy_fwd_bwd` takes an
   optional per-row f32 `row_scale` and a `write_grad` flag (loss-only evaluation, phase 2 token
   weights); `adamw_step` takes `grad_scale` so clipping is fused into the update; `bias_grad` is
   implemented now so the table does not change when biases become trainable.
6. **Cross entropy** is evaluated in the log-softmax form `(max - z[y]) + log(sum exp(z - max))`
   (the gradient as `exp((z - max) - log s)`), which avoids cancellation for large logits.
7. **AdamW** follows the operation order of `torch/optim/adam.py::_single_tensor_adam` (Python
   scalars rounded to f32, `bias_correction2 ** 0.5`, `addcdiv` with `-step_size`) with the
   element rounding of ATen's vectorized CPU kernels, determined empirically against torch 2.15:
   `lerp_` and `addcmul_` are fused multiply-adds (`fmaf`). Tested against
   `torch.optim.AdamW(foreach=False)` itself: `exp_avg` and `exp_avg_sq` are bit-identical; the
   parameters differ by one ulp in a small fraction of elements because torch's vectorized CPU
   `sqrt` is not correctly rounded (`sqrtf` is), so parameters are compared with a tolerance.
8. **bf16 NaN**: f32 → bf16 keeps sign and upper payload bits and sets the quiet bit
   (`(bits >> 16) | 0x40`); PyTorch's CPU path returns the canonical `0x7FC0`. Every non-NaN
   value matches torch bit-for-bit (tested against values converted by torch).
9. **config.json strictness** (stricter than §1, which only requires refusing unimplemented
   values): unknown top-level keys are refused unless they are on a documented ignore list
   (metadata, token ids, generic PreTrainedConfig fields, generation parameters), because an
   unknown key may carry semantics; `RRC_CONFIG_ALLOW_UNKNOWN_KEYS` / `rrc_inspect
   --allow-unknown-keys` turns this into a warning. A config with both a non-empty
   `rope_scaling` and a non-empty `rope_parameters` is refused as ambiguous (transformers silently
   uses `rope_scaling`); keys inside the rope dict that the rope type does not use are refused
   (transformers only warns). A top-level `original_max_position_embeddings` overrides the rope
   dict's value, exactly as transformers does at rope init.
10. **llama3 rope scaling is accepted for every family** (the formula does not depend on the
    family; §1 lists it under Llama because that is where it occurs). Qwen2's
    `attention_bias`/`mlp_bias` and Qwen3's `mlp_bias` keys are ignored because the transformers
    models do not read them (the tensor-set check still catches any mismatch).
11. **Safetensors strictness**: like the reference `safetensors` package, the data block must be
    tiled exactly (overlaps, holes and uncovered bytes are errors); in addition every tensor entry
    must have exactly the keys dtype/shape/data_offsets. F16 checkpoints are read and converted
    exactly to f32 by `rrc_st_to_f32` (or RNE to bf16), but F16 is not an engine dtype
    (`rrc_st_dtype_to_rrc` refuses it); I64 converts to I32 with a range check. The tensor-set
    checker accepts only F32/BF16 model weights.
12. **Extra files not in §2**: `src/rrc_inspect.c` (`build/rrc_inspect`, prints the loader's view
    of a checkpoint as JSON for the pytest comparison), `tests/cuda/headers_check.cu` (the nvcc
    compile check run by `make test` when `CUDA_HOME`/`RRC_NVCC` is set), `tests/c/rt.h` (test
    runner). `make cuda` currently runs the header check and compiles `cuda/*.cu` when present.
13. **No pinned / asynchronous host-to-device entry in the table** (§6 mentions pinned buffers
    and async copies): `memcpy_h2d` is synchronous with respect to the host buffer, so a CUDA
    backend that wants overlap must stage through its own pinned buffer inside `memcpy_h2d`.
    The per-step uploads (token ids, labels, positions, doc ids) are small.
14. **RoPE precision vs transformers grows with position**: transformers evaluates `inv_freq` in
    float32, rrc in double (rounded to f32 for the tables), so the angle `p * inv_freq` differs by
    up to about `p` times one ulp of `inv_freq`. The pytest comparison covers positions up to 255
    (cos/sin within 3e-5); long-sequence comparisons in later stages must budget for this.
