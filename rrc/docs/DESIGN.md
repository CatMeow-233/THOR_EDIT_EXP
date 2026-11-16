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
  GEMM call pattern per packed row and query chunk (Hq query heads, Hkv KV heads, G = Hq/Hkv,
  query head `h` uses KV head `h / G` as in HF `repeat_kv`): scores, P·V, dP and dQ are one
  batched call per KV group (batch G over the group's query heads, K/V broadcast with stride 0,
  dQ written as interleaved C slices `stride_c = head_dim`, `ldc = Hq·head_dim`). dK and dV sum
  over the G query heads of a group, and a batched call cannot reduce over its batch
  (`stride_c = 0` is refused: overlapping C slices). They are instead G calls per chunk, one per
  in-group head index `u`, each batched over the Hkv groups and accumulating with `beta = 1`:
  `A = dS` slices of heads `g·G+u` (`stride_a = G·bs`, bs = the per-head score-slice stride),
  `B = Q` (or `dO`) heads
  `g·G+u` (`stride_b = G·head_dim`), `C = dK` (or `dV`) heads `g` (`stride_c = head_dim`,
  `ldc = Hkv·head_dim`). Per layer backward that is `2·G·B·ceil(T/Tc)` GEMMs with `k = Tc` for
  dK/dV (tested: `contracts.gqa_dk_batched_over_kv_groups`). Stacking the heads along `k` instead
  would need uniformly strided Q rows, which the `[N, Hq·head_dim]` layout has only for Hkv = 1.
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
* Contract details settled before the freeze (§9, items 15–18): `gemm` requires m, n, k,
  batch ≤ INT32_MAX and resolves k = 0 itself (C = beta·C, never dispatched); matrix operands of
  `gemm` and `swiglu_fwd/bwd` that are disjoint column blocks of one buffer (same row pitch) are not
  an alias; operand pointers are only element-aligned; ops may use internal scratch and return
  `RRC_ERR_OOM`.

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
The op table has no dedicated gather/scatter; both are `embedding_fwd` calls (tested:
`contracts.lm_head_row_gather_scatter`): the gather is `embedding_fwd(n_sel, D, vocab = N, ids =
sel, table = h_final [N, D], out = h_sel)`; the scatter of `dh_sel` back to `[N, D]` is
`embedding_fwd(N, D, vocab = n_sel + 1, ids = inv, table = dh_ext, out = dh)` where `dh_ext` is
`[n_sel + 1, D]` with a zero last row and `inv[t]` is the row of token `t` in `sel`, or `n_sel` for
an unlabelled token (both index arrays are built on the host with the micro-batch). Per-token
losses of all chunks go into one f32 buffer and are summed either on the host (one `memcpy_d2h`
per micro-batch) or on the device by `gemm(m = 1, n = 1, k = rows, A = ones, B = loss, beta = 1)`.
Computing the head over all N rows instead (CE zeroes the ignored rows) would cost
`(N − n_sel)·V·D` multiply-adds in each of the two head GEMMs.

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
    models do not read them (the tensor-set check still catches any mismatch). Qwen2's `use_mrope`
    (carried by some qwen2-architecture checkpoints; the transformers qwen2 model does not read it)
    is accepted when false or null and refused when true.
    Upper bounds (config.json is untrusted input; review finding: a `num_hidden_layers` of
    11⁻¹ mod 2⁶⁴ wrapped the size_t tensor-spec count and the layer loop wrote past the array):
    `vocab_size`, `hidden_size`, `intermediate_size`, `num_attention_heads`,
    `num_key_value_heads` and `head_dim` ≤ 2³¹ − 1, `num_hidden_layers` ≤ 1024 (the LoRA layer
    table), and the base model's weight element count, every product and sum checked, ≤ 2⁴⁰
    (`include/rrc/config.h`); `rrc_config_expected_tensors` re-checks the count with checked
    arithmetic for configs built elsewhere. `max_position_embeddings` bounds no tensor and stays
    unbounded (the RoPE table is refused above 2³¹ − 1 rows unless `max_positions` bounds it).
11. **Safetensors strictness**: like the reference `safetensors` package, the data block must be
    tiled exactly (overlaps, holes and uncovered bytes are errors); in addition every tensor entry
    must have exactly the keys dtype/shape/data_offsets. F16 checkpoints are read and converted
    exactly to f32 by `rrc_st_to_f32` (or RNE to bf16), but F16 is not an engine dtype
    (`rrc_st_dtype_to_rrc` refuses it); I64 converts to I32 with a range check. The tensor-set
    checker accepts only F32/BF16 model weights. Untrusted-input hardening (review findings):
    duplicate JSON keys are detected by a linear scan in objects of up to 16 keys and by sorting
    (key, offset) in larger ones (a safetensors header or an index weight_map), so a 200k-entry
    header parses in 0.4 s (measured here; the review measured 158 s before); the shard list of an index is
    deduplicated by sorting too. Every input file (config.json, safetensors, index, `.rrcbin`,
    `rrc_state.bin`, training config) is opened with `O_NONBLOCK` and refused unless it is a regular
    file (a FIFO no longer blocks the program); temporary files of atomic writes are created with
    `O_CREAT | O_EXCL | O_NOFOLLOW` after unlinking a stale entry (`include/rrc/fileio.h`), so a
    planted symlink is never written through.
12. **Extra files not in §2**: `src/rrc_inspect.c` (`build/rrc_inspect`, prints the loader's view
    of a checkpoint as JSON for the pytest comparison), `tests/cuda/headers_check.cu` (the nvcc
    compile check run by `make test` when `CUDA_HOME`/`RRC_NVCC` is set), `tests/c/rt.h` (test
    runner), `tests/c/test_contracts.c` (interface-contract tests: column-block aliasing, gemm
    k = 0 and int32 limits through a recording backend, the GQA and LM-head call patterns, bounded
    CPU gemm scratch; the test binary is linked with `-Wl,--wrap=malloc` for the last one). `make cuda` currently runs the header check and compiles `cuda/*.cu` when present.
13. **No pinned / asynchronous host-to-device entry in the table** (§6 mentions pinned buffers
    and async copies): `memcpy_h2d` is synchronous with respect to the host buffer, so a CUDA
    backend that wants overlap must stage through its own pinned buffer inside `memcpy_h2d`.
    The per-step uploads (token ids, labels, positions, doc ids) are small.
14. **RoPE precision vs transformers grows with position**: transformers evaluates `inv_freq` in
    float32 (f32 `pow`, and for llama3 the smoothing arithmetic in f32 too), rrc in double, rounded
    once to f32 for the tables. The f32 `inv_freq` values differ by up to 1 ulp for default rope
    (24 of 64 values for theta 1e6, head_dim 128) and up to 3 ulp for llama3 (21 of 64 for theta
    5e5, factor 8, original 8192; the 3 ulp are in the medium band), so the angle `p * inv_freq`
    differs by up to about `3p` ulp of `inv_freq`. Measured max |cos/sin difference| against
    transformers (one-off probe over the full head): llama3 (8B-like) 1.4e-5 at position 255,
    4.0e-4 at 8191, 1.9e-3 at 32767, 3.9e-3 at 65535, 5.7e-3 at 131071; default rope theta 1e6
    4.8e-4 at 8191, 1.95e-3 at 32767, 3.0e-3 at 65535. `test_rope_long_positions_bound` checks the
    ulp bounds and cos/sin within 3e-5 (255), 1e-3 (8191), 4e-3 (32767), 1e-2 (131071);
    long-sequence comparisons must use tolerances of that size. rrc keeps the correctly rounded
    values (reproducing transformers' f32 llama3 arithmetic was considered and not done: the
    default-rope `pow` ulp would remain, and the tables would no longer be reproducible from the
    formula alone).
15. **Aliasing is per element for column blocks (gemm, swiglu_fwd/bwd)**: two matrix operands with
    the same row pitch in bytes whose column ranges are disjoint do not overlap, even though their
    address ranges interleave. This makes the fused `[gate | up]` layout usable end to end: the
    swiglu backward may write `[dgate | dup]` into one fused buffer (the A operand of one backward
    GEMM with `k = 2·I` against the stacked `[W_gate; W_up]`) or in place into `[gate | up]`, and a
    gemm may write one column block of a buffer while reading another. Batched operands use the
    columns covered by all their slices when the slices are interleaved within rows. Other ops keep
    the address-range test (their in-place uses are exact aliases).
16. **gemm with k = 0** (m, n > 0) is `C = beta·C` (zero when beta = 0, NaN not propagated), computed
    in `ops.c` with `memset0` / `scale` per contiguous run of C; the backend's gemm is never called
    with k = 0, and A, B may be `rrc_buf_none()`. It occurs e.g. for the LoRA `dA += dtᵀx` GEMM of
    a micro-batch with no tokens.
17. **gemm limits and alignment**: m, n, k and batch must each be ≤ INT32_MAX (`RRC_ERR_SHAPE`),
    because cuBLASLt's `BATCH_COUNT` is int32 and ROWS/COLS are int32 in practice; leading
    dimensions and strides stay int64 (they are int64 in cuBLASLt). Operand pointers are only
    element-aligned (views at per-head offsets), so the CUDA gemm must set
    `CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_*_BYTES` from the actual pointers on every call.
18. **Internal scratch**: ops may allocate internal scratch and fail with `RRC_ERR_OOM`. The CPU gemm
    packs one 32-row block of op(A) and one 32-column block of op(B) per thread (2·32·k floats)
    instead of whole matrices, with the same dot-product order as before; this bound is tested
    by counting allocation sizes (the test binary wraps `malloc`).
19. **Rejected additions to the op table** (recorded so the frozen table is a decision, not an
    omission): a row-scatter op (composable from `embedding_fwd` with an inverse index, §5, one
    call), a plain sum reduction (per-token losses: host sum or a ones-vector gemm, §5), a gemm
    that reduces over its batch for GQA dK/dV (G calls batched over the KV groups instead, §3).
    A fused attention op remains the later optimisation named in §3, behind the same layer code.
20. **Cross entropy with -inf logits**: the online log-sum-exp ignores -inf values
    (`rrc_kc_lse_push`, and the double-accumulation row loop), so a -inf logit anywhere in a row,
    including index 0, contributes 0 and its gradient is 0, as in torch. Before this rule a leading
    -inf produced `exp(-inf - (-inf)) = NaN` for the whole row. A row of only -inf logits gives
    NaN (torch does the same).

## 10. Stage B model line: decisions and deviations

Step 1 of the model line: model loading, LoRA, and the decoder forward / backward on the CPU
backend, verified against transformers + PEFT. Code: `include/rrc/{model,lora}.h`,
`src/{model,model_exec,lora}.c`, `src/model_internal.h`, `src/rrc_fwdbwd.c` (test CLI),
`tests/c/test_model.c`, `tests/test_model_vs_peft.py`. Status: **TESTED (CPU)** on tiny random
models for both CPU backends; nothing here has run on a GPU and no performance figure exists
(NOT_TESTED). No change to the op table or its wrappers was needed.

1. **Separate q/k/v GEMMs** (three projections into three contiguous buffers `[N, Q]`, `[N, KV]`,
   `[N, KV]`) instead of one fused `[Q + 2KV, H]` GEMM. With contiguous Q and K, Qwen3's per-head
   `q_norm`/`k_norm` is one `rmsnorm_fwd/bwd` call on the `[N·heads, head_dim]` view and RoPE /
   attention see uniform strides; a fused output would need one rmsnorm call per head. Cost: three
   GEMM launches instead of one per layer (forward and backward). **gate/up are fused**: the
   checkpoint's `gate_proj` and `up_proj` are stored adjacently in the weight allocation, so
   `[W_gate; W_up]` is one `[2I, H]` matrix: one forward GEMM into a `[N, 2I]` buffer, `swiglu` on
   its column blocks, and one backward GEMM with `k = 2I` (§9 item 15).
2. **Weights**: all base tensors (norm weights and biases included) live in one backend allocation
   in the model dtype (bf16 default, f32 for tests), converted on the host by `rrc_st_to_f32/bf16`
   (RNE for an fp32 checkpoint loaded as bf16, the same rounding as torch's `.to(bfloat16)`), in the
   order of `rrc_config_expected_tensors`, after the exact tensor-set check of stage A. Tied models
   use the embedding as the LM head.
3. **Precision of intermediates**: activations, the residual stream and its gradient are in the
   model dtype (as in HF). fp32 regardless of the model dtype: attention scores, probabilities and
   dP/dS (HF rounds P to bf16; rrc does not), the LoRA `t = x Aᵀ` and `dt`, the per-token losses,
   and three gradient accumulators: dK and dV (accumulated with `beta = 1` over `2·G` calls per query
   chunk, so a bf16 accumulator would round at every call) and `dxn`, the input gradient of a norm
   (sum of the three q/k/v — or the two gate/up — input gradients plus their LoRA terms). dQ is
   written once per chunk and stays in the model dtype. Measured effect of the three fp32
   accumulators (bf16 model, the window of `test_lora_bf16_matches_peft`, T = 23, attention chunk 5;
   mean over all LoRA tensors of the relative L2 gradient error against fp32 on the same rounded
   weights; one-off A/B, not a test): llama3_rope 0.00794 → 0.00750, qwen2_tied 0.00877 → 0.00820,
   qwen3_hd32_bias 0.01008 → 0.00976 (HF-bf16 itself: 0.00952, 0.00999, 0.01088). A small,
   consistent gain at this size; at long sequences the number of dK/dV accumulations grows with
   T / chunk, which is the reason to keep them (not measured at that size). The input-gradient GEMMs
   through the frozen k_proj / v_proj weights read a copy of dK / dV rounded once to bf16, so they
   run bf16 x bf16 -> f32 like every other base-weight GEMM (§13 item 3; before that they promoted
   the weight to f32 on the device). The LoRA branch follows PEFT: the bf16
   input is widened exactly to fp32 by the GEMM (`t` is fp32), and `y += s · t Bᵀ` is one GEMM with
   `alpha = s`, `beta = 1` into the base output, rounded once (PEFT: fp32 sum of the bf16 base result
   and the fp32 LoRA term, then one cast). Biases are added after the base GEMM and before LoRA.
4. **GEMM dtype combinations issued** (A, B, C), recorded by `model.model_gemm_dtype_combinations`:
   f32 model: only (f32, f32, f32); bf16 model: (bf16, bf16, bf16) base linears, LM head and its
   input gradient, dact/dob; (bf16, bf16, f32) scores, dP, the base-weight input gradients into
   `dxn` (q, gate/up, and k/v from the bf16 copy of dK/dV, §13); (bf16, f32, f32) LoRA `t`, `dB`,
   `dt` from a bf16 dy; (f32, bf16, bf16) P·V and dQ; (f32, bf16, f32) dK, dV, `dA`; (f32, f32,
   bf16) LoRA `y += s t Bᵀ`, LoRA `dx` into a bf16 gradient; (f32, f32, f32) the rest of the LoRA
   backward (including dB / dt of k_proj / v_proj from the f32 dK / dV). (bf16, f32, bf16) is never
   used. Which of these the CUDA GEMM plan runs natively is counted by `rrc_census` (§13).
5. **Attention key range**: a query chunk `[q0, q0 + R)` of a packed row reads keys
   `[k0, q0 + R)` where `k0` is the start of the document of query `q0` (keys before it are masked
   for every query of the chunk), via `row_offset = q0 − k0` and a doc-id view starting at `k0`.
   This is exact only for contiguous documents, so a row in which a document id reappears after a
   different one is refused (`RRC_ERR_INVALID_ARG`). Packed rows then cost about the attention FLOPs
   of their documents rather than of the whole row (the saving is not measured: NOT_TESTED).
6. **Recompute / activation cache**: a layer's forward writes into an `rrc_layer_cache` (xn1, q_raw
   / k_raw for q/k norm, post-rope q, k, v, attention output, h, xn2, [gate | up], act, the LoRA `t`
   of every adapted module). `RRC_RECOMPUTE_FULL` (default) has one scratch cache: the forward saves
   only each layer's input (`xs[l]`), the backward recomputes the layer into the scratch cache
   (skipping the down projection and residual, whose outputs it does not need; the top layer's
   cache is still valid from the forward and is not recomputed). `RRC_RECOMPUTE_NONE` has one
   persistent cache per layer and no recompute; this is the switch for a later "store activations
   when they fit" policy. Both give bit-identical results (tested).
7. **Backward stops at the lowest adapted layer**: no layer below it is recomputed or
   back-propagated, and that layer skips its final `rmsnorm_bwd` (the input gradient). No finer
   pruning inside a layer yet (e.g. the attention backward of a lowest layer whose adapters are all
   in the MLP).
8. **Workspace**: one backend allocation per model, laid out by a single function run twice
   (first only summing offsets — no pointer is formed — then binding 256-byte-aligned typed views
   of the allocation). It is sized from `N = B·T` (the labelled-row buffers are bounded by `N`) and
   reallocated only when a step needs more (one allocation for repeated steps: tested). It is not
   zeroed between steps: the dK/dV accumulators and the zero row of the LM-head scatter table are
   zeroed explicitly every step. Per-step host allocations are the O(N) int32 batch-preparation
   arrays and the per-token loss download; there are no per-op allocations besides the backend's
   internal GEMM scratch.
9. **Batch semantics** (`rrc_batch`): HF-style unshifted labels; the engine's target for token t is
   `labels[t + 1]` if t + 1 is in the same document (same row and doc id), else ignored, so the
   first label of a document is never a target and nothing leaks across documents or rows.
   `doc_ids` and `position_ids` may each be omitted (positions restart at every document; documents
   start where `position_ids == 0`; one document per row if both are absent). Padding is a document
   whose labels are all -100. Tokens, labels and positions are range-checked on the host.
10. **Loss reductions and zero labelled tokens**: `window_tokens` scales by `1 / window_tokens`
    (the caller sums `rrc_batch_count_labels` over the window first; a micro-batch with more
    labelled tokens than `window_tokens` is an error); `microbatch_mean` by `1 / (n_mb · M)`. The
    scale is folded into the cross entropy (`grad_scale`), so dlogits are already scaled. A
    micro-batch with zero labelled tokens contributes loss 0 and gradient 0 under both reductions
    (it still counts in M for `microbatch_mean`; HF's per-micro-batch mean would be NaN there), and
    a window with `window_tokens = 0` gives loss 0 and zero gradients. Per-token losses (f32) are
    summed in double on the host, one `memcpy_d2h` per micro-batch.
11. **LoRA**: `rrc_lora` is an object separate from the model (an adapter for a model shape on a
    backend; the forward accepts `NULL` for the base model, e.g. a phase-2 reference policy). A, B,
    dA, dB are fp32 in one contiguous parameter and one gradient buffer (layer-major, modules in the
    order q, k, v, o, gate, up, down, A before B), so AdamW and the gradient norm are single op
    calls. Names are PEFT's saved-adapter keys (`base_model.model.model.layers.{l}.{self_attn|mlp}.
    {m}.lora_{A|B}.weight`, taken from a file PEFT wrote). `layers_to_transform` is supported.
    Initialisation: A uniform in ±1/√in from the splitmix64 scheme documented in `lora.h` (not
    torch's RNG), B = 0. `lora_dropout != 0`, unknown targets and adapter files whose tensor set or
    shapes differ are refused. `rrc_fwdbwd` takes the LoRA hyper-parameters as flags; reading
    PEFT's `adapter_config.json` is left to the trainer step.
12. **HF hidden-state convention** (`rrc_model_forward_logits(..., hidden_out)`): entries 0..L−1
    are the embedding output and the outputs of layers 0..L−2, the last entry is the final-norm
    output, exactly as transformers' `output_hidden_states` (verified by the test).
13. **Tests** (`tests/test_model_vs_peft.py`): LoRA A and B are randomized and non-zero (A within
    the kaiming bound, B with std 0.05 so that `s·BA` is comparable to the base weights; with a
    LoRA delta about ten times the base weights the tiny models' attention saturates and bf16
    rounding of the base weights alone moved gradients by over 10 %, which would make any bf16
    comparison meaningless). fp32 base: logits within 2e-5 (cpu) / 1e-4 (cpu-f32acc), loss within
    the same relative tolerance, every LoRA gradient within a per-tensor relative L2 error of 5e-5
    / 5e-4; gradient name sets must equal PEFT's and every reference gradient must be non-zero.
    bf16 base: rrc's error against an fp32 run on the same bf16-rounded weights must be at most
    twice HF-bf16's own error (+0.005), per logit block and per gradient tensor. The packing
    reference runs every document separately through HF/PEFT. Required traps covered: attention
    chunk 5 < T = 23 (not a divisor), LM-head chunk 7 < labelled rows, ≥ 2 documents per row with a
    document crossing chunk boundaries, unequal labelled counts per micro-batch (the two reductions
    are asserted to differ), an empty micro-batch inside a window and a fully empty window, GQA with
    G = 2 and Hkv = 2, MHA (G = 1) and MQA (Hkv = 1) with attention chunks 1, 6 and T (item 30),
    runs of length-1 documents on chunk boundaries and rows of T = 1, Qwen3 with head_dim ≠ hidden/heads, llama3 rope, tied heads, Qwen2 biases,
    adapters on the layer sets [1, 2], [2], [1], [0], [0, 2] and all layers of a 3-layer model (the
    backward stopping below an unadapted layer, the input gradient crossing an unadapted gap), both CPU backends, a bf16
    checkpoint saved by HF in several shards (bit-identical to the single-file path). HF's own
    `model(..., labels=...).loss` pins the label shift. `tests/c/test_model.c` adds directional
    finite differences of the loss (fp32 model, double-accumulation backend), chunking / recompute /
    backend invariance, batch validation, and runs under `make asan`.
14. **Sanitizers**: `make asan` builds the library, binaries and C tests into `build/asan/` with
    the normal flags (`-std=c11 -Wall -Wextra -Werror -fopenmp`) plus `RRC_SAN_FLAGS =
    -fsanitize=address,undefined -fno-sanitize-recover=all` and runs the C tests with leak detection
    and two OpenMP threads (the `--wrap=malloc` test link works with ASan). `build/asan/rrc_fwdbwd`
    was also run by hand under ASan on two tiny models (f32 and bf16, both recompute policies, an
    error path); that run is not part of `make asan`.
    (After the merge with the CUDA line, `make asan` runs the whole `make test` with these flags,
    including `rrc_selftest_host` and, with `CUDA_HOME`, `cuda-host-test`; §12.)

### Step 2: training driver (data, optimizer, trainer)

Code: `include/rrc/{data,train}.h`, `src/{data,train}.c`, `src/rrc_train.c`, `src/rrc_data.c`,
`tools/prepare_sft.py`, `tools/torch_reference.py`, `tests/c/test_{data,train}.c`,
`tests/test_train_vs_torch.py`. Status: **TESTED (CPU)** against torch + transformers + PEFT on tiny
models; nothing has run on a GPU; the trainer had no CUDA path in this step (added at the
integration, §12, COMPILED-ONLY); no performance figure exists
(NOT_TESTED). No change to the op table or its wrappers was needed (the optimizer uses
`sum_squares` and `adamw_step` with `grad_scale` as frozen).

15. **`.rrcbin` header**: §6 names the fields; their widths are fixed here as three u64 after the
    8-byte magic (32-byte header), documents u32 len / u32 tokens / u8 mask, no padding. Validation
    covers every field (data.h); in addition a header `vocab_size` above the model's vocabulary is
    refused even if every token would fit (it means another tokenizer). Bounds are checked in u64
    before any allocation (`num_docs ≤ (size − 32) / 9`), tokens are checked `< vocab` (≤ 2³¹ − 1)
    before the int32 cast. The file is mmapped and checked in one pass (FNV-1a 64 of the bytes
    identifies it in `rrc_state.bin`).
16. **Packing is greedy and never splits a document** across rows: a document that does not fit in
    the rest of a row closes the row (padding: one extra document with label −100 and positions
    restarting at 0). Epochs are concatenated into one document stream, so a row or window may hold
    the end of one epoch and the start of the next; the cursor (epoch, index into that epoch's order)
    at a micro-batch boundary is the loader's whole state. This is not TRL's packing strategy
    (`bfd` / wrapped): which documents share a row differs, so per-step losses are not comparable to
    a TRL run step by step, only the per-document semantics are (padding-free packing with
    position ids). The torch comparison feeds the reference the batches rrc trained on
    (`dump_batches`), and an independent Python implementation of the spec
    (`torch_reference.pack_reference`) reproduces those batches exactly.
17. **Shuffling** is rrc's own generator (splitmix64 Fisher–Yates with modulo reduction, seeded by
    (seed, epoch); data.h), not torch's `RandomSampler`: the document order differs from an HF
    Trainer run with the same seed. It is the only random generator after initialisation, so
    (seed, epoch, index) is the full RNG + data state stored for resume.
18. **Prefetch** is a pool of `2 × grad_accum_steps` host micro-batches filled by one pthread (slots
    are not used round-robin: item 31): the
    `window_tokens` reduction needs the label count of the whole window before the first
    forward/backward, so the trainer takes all micro-batches of a window at once and the thread fills
    the next window meanwhile. The thread touches host memory only (backend calls stay on one thread,
    backend.h), copies its error message into the slot (the error state is thread-local) and the
    consumer re-raises it. Batches are a pure function of the cursor (tested: prefetch on / off,
    hold counts, repeated runs, start from a saved cursor; the data / train suites run under
    `make tsan`). There is no pinned / asynchronous upload (§9 item 13). Whether the thread helps or
    competes with the OpenMP threads of the CPU backend is not measured.
19. **LR schedule** (deviation from §3): `constant_with_warmup` is added; transformers' `constant`
    ignores warmup, so rrc's does too. transformers 5.23 has no `warmup_ratio`: `warmup_steps` is one
    float (≥ 1: `int()` steps, < 1: `ceil(ratio · max_steps)`), and rrc's config follows that.
    Optimizer step s uses `lr · λ(s − 1)` (LambdaLR evaluates λ(last_epoch) before each step), so with
    warmup the first step has lr 0. Tested exact (`==`) against `get_scheduler` for 15 cases incl.
    warmup 0, warmup ≥ total, total 1, a ratio and a fractional step count (C `cos` equals
    `math.cos`).
20. **Clipping**: total norm = `sqrtf(sum_squares(all LoRA grads))`, one reduction over the
    contiguous gradient buffer; torch takes the norm of per-tensor norms, so the two differ in the last
    bits (measured relative difference ≤ 3.3e-7 over the trajectory). The coefficient is torch's
    exactly: `max_norm / (total_norm + 1e-6)` on an f32 tensor is `reciprocal(total_norm + 1e-6f) ·
    (float)max_norm` (Python's `__rtruediv__`; plain f32 division differs in 17 % of random cases, a
    20000-sample probe), clamped at 1 and always applied (multiplying by 1 is exact); it is fused
    into AdamW via `grad_scale`. Tested bit-exact against torch's formula on rrc's norm, and m against
    the clipped torch gradient. `max_grad_norm = 0` disables clipping (HF: `None` or 0). A non-finite
    norm stops training with `RRC_ERR_RANGE` before any update (HF would continue with NaN).
21. **A window without labelled tokens** gives loss 0 and zero gradients (§10 item 10); rrc still
    performs the AdamW step with g = 0 (the moments decay and the parameters move by the remaining
    momentum) and logs a warning. torch would skip parameters whose `.grad` is `None`, and HF Trainer
    would compute 0/0. Tested against torch AdamW fed explicit zero gradients
    (`torch_reference.RefConfig.zero_grad_if_empty`, both reductions, item 32).
22. **Metrics** (deviation from §6): `tokens_per_s` is `input_tokens_per_s` (non-padding tokens of the
    step / measured step time), `elapsed_s` counts from the start of the process (a resumed run starts
    again at 0), and there is no "max memory" field (the op table has no allocator statistics). Wall
    times are measured with `CLOCK_MONOTONIC`, never synthesized.
23. **Checkpoints**: `adapter_config.json` holds a subset of the fields PEFT 0.21 writes; every one
    of them equals what PEFT's own `save_pretrained` writes for the same `LoraConfig` (tested), except
    `base_model_name_or_path`, which is `model_dir` as given. `adapter_model.safetensors` carries
    PEFT's metadata `{"format": "pt"}`. All files are written into `.checkpoint-<step>.tmp/`:
    `adapter_config.json` and `rrc_state.bin` as `.tmp`, fsynced, renamed; `adapter_model.safetensors`
    by the stage A writer (`.tmp` + rename, no fsync of its own) and fsynced after its rename. The
    directory is then fsynced and renamed to `checkpoint-<step>` (an existing one is first renamed to
    `.old` and removed afterwards; if the final rename fails it is renamed back, and a
    `.checkpoint-<step>.old` without a `checkpoint-<step>` is restored, not deleted, by the next save
    of that step: item 33), and `output_dir` is fsynced. Nothing time- or path-dependent (except the model
    path above) enters a checkpoint, so checkpoints of equal runs are byte-identical.
24. **`rrc_state.bin`** (layout in train.h) stores m and v, not the parameters: those are read back
    from the checkpoint's `adapter_model.safetensors` (fp32, exact) and checked against the stored
    FNV-1a hash. Resume also refuses another data file (hash) and a different config fingerprint:
    every field that changes the trajectory (dtype, backend, shapes, optimizer and schedule values,
    resolved warmup, reduction, data seed / shuffle / truncation / pad id, LoRA r / alpha / rslora /
    targets / layers, attention / CE chunk sizes, max_positions). Not part of it: paths,
    `save_steps`, `stop_after_step`, `prefetch`, `recompute` (bit-identical by §10 item 6),
    `dump_batches`, the LoRA init seed. The model itself is not hashed (16 GB); a different
    checkpoint with the same shapes would be accepted. A trailing FNV-1a 64 over the file detects
    corruption (tested at every 5th byte); it is not a MAC, so the counters are also range-checked
    (item 34). On resume into the checkpoint's own `output_dir`, `metrics.jsonl` lines after the
    checkpoint step and a partial last line are dropped; resuming into another directory is item 35.
25. **Measured agreement** (tiny llama, 8 steps, grad accumulation 2, seq_len 32, 2 rows, clipping
    on 4 of 8 steps, warmup 2 + cosine, wd 0.01, two epoch boundaries; CPU double-accumulation
    backend): fp32 max relative error of the loss 5.0e-8, grad norm 3.3e-7, final LoRA parameters
    2.8e-7, Adam m 8.3e-7, v 7.8e-7 (parameters moved by ≥ 26 % relative; clipping active on steps 2,
    5, 6, 7). The fp32-accumulation backend: loss 5.9e-8, grad norm 1.2e-6, parameters 2.8e-7, m / v
    1.1e-6; `microbatch_mean`: loss 5.0e-8, grad norm 3.1e-7, parameters 2.6e-7, m / v 9.7e-7. The test
    tolerances are about 10x these. bf16 base: rrc's max loss
    error against the fp32 truth on the same rounded weights 2.2e-4 vs HF-bf16's 2.5e-4; final
    parameters at most 0.32 of the allowed `2·e_HF + 0.005`. The PEFT-loaded checkpoint reproduces
    rrc's logits within 1.8e-7 (adapter effect on the logits 0.69). Mutation checks (by hand, not in
    CI): using λ(s) instead of λ(s − 1) fails the trajectory test; dropping the clip epsilon, and
    (run separately) using plain f32 division for the coefficient, each passed every loss / gradient-norm
    / parameter / moment tolerance (Adam is nearly invariant to a 1e-6 relative gradient scale); both
    are caught by the exact coefficient check, which was added for that reason.
26. **`prepare_sft.py` assistant spans** are located on the rendered text (prefix renderings with
    `add_generation_prompt`), and a token is a target iff its first character lies in a span; a
    template that is not prefix-stable is refused rather than guessed. Tested with a WordLevel
    tokenizer trained in the test (`tokenizers`), a two-turn conversation with a system message, and a
    prompt/completion pair.
27. **Not done in step 2**: an epoch-based run length (only `max_steps`), evaluation / validation
    loss, several data files or weighting, the CUDA training path (the loop calls only the op table
    and is backend-agnostic, but `rrc_train` only offers the CPU backends), pinned host buffers,
    a memory metric, and LR schedules beyond the four above.

### Step 3: review repairs (coverage, loader, robustness)

28. **Fail-first evidence**. Each defect below got a regression test that failed on the code before
    the fix (recorded in the commit message): the loader test failed without prefetch
    (`RRC_ERR_INTERNAL: loader slot not free` on the second batch) and hung with prefetch (killed
    by `timeout`); the state, cursor, checkpoint, resume and adapter-config tests returned the wrong
    status; the RoPE test crashed (SIGSEGV after a wrapped allocation size) and, under `make asan`,
    reported the signed overflow at the old `model.c:165`; the permutation call reported the signed
    overflow at the old `data.c:200` under UBSan; the CLI tests got exit 0 (`rrc_fwdbwd`, an empty
    batch, an output file written) and exit 0 / SIGFPE (`rrc_data`). The coverage additions
    (items 30, 32) found no defect: they pass on the unchanged library.
29. **Bounds instead of wider types**: counters accepted from outside (a start cursor, a
    checkpoint) are at most `RRC_DATA_COUNTER_MAX = 2^62` (data.h), far beyond any run and far enough
    below `INT64_MAX` that the loader's `epoch++` / `micro_batches++` and the trainer's
    `tokens_seen += window` cannot overflow. The permutation seed uses `(uint64_t)epoch + 1`, the
    same value for every valid epoch (the Python reference is unchanged).
30. **Coverage added (no defect found)**: tiny `llama_mha` (4 / 4 heads) and `llama_mqa` (4 / 1)
    variants in `make_tiny_model.VARIANTS`, in `LORA_CASES` (both backends) and in a test over
    attention chunks 1, 6 and T; a layout with eight length-1 documents (at 0..5, 10 and 15, i.e. on
    the attention chunk boundaries 5, 10 and 15 of chunk 5; 20 is not covered) and a length-2
    document starting on 10, over attention
    chunks 1, 2, 5, T and CE chunks 1, 7; rows with T = 1 (counts 0, loss 0, gradients exactly 0,
    logits against HF); the layer sets of item 13.
31. **Loader slots are claimed by stream position**, not `index % ring size`: each slot carries the
    stream position of its batch; the producer fills any free slot with the next position (under
    the mutex), the consumer takes the ready slot at its position. data.h allows release in any
    order, and a batch held while more than a full turn of the ring went through the other hold used
    to deadlock (prefetch) or fail with `RRC_ERR_INTERNAL` (no prefetch). With prefetch the consumer
    holds at most `max_held − 1` of `2 · max_held` slots when it asks for a batch, so a free slot
    always exists for the producer and the wait terminates. Tested by holding `max_held − 1` batches
    for `8 · max_held` further batches, max_held 2 and 3, both modes, against the reference stream;
    the data / train suites also run under `make tsan`. The slot search is linear in `2 · max_held`
    (≤ 2^17), negligible next to a micro-batch.
32. **Trainer coverage added (no defect found)**: two more fp32 trajectories against torch AdamW +
    PEFT, on data with documents of length 1, seq_len, seq_len + 1 and longer than seq_len:
    (a) truncate, grad_accum 3, micro_batch_size 1, betas (0.8, 0.95), eps 1e-6, weight decay 0.3,
    linear schedule with warmup ratio 0.3, clipping on some steps — measured max relative error loss
    2.9e-8, grad norm 2.0e-7, parameters 2.4e-7, m 1.1e-6, v 8.7e-7; (b) drop, grad_accum 3,
    micro_batch_size 2, `microbatch_mean`, constant, no clipping, beta1 0.85, eps 1e-4 — loss 3.0e-8,
    grad norm 7.9e-7, parameters 8.9e-7, m 3.5e-6, v 4.2e-6. The all-ignored window (item 21):
    `window_tokens` at eps 1e-8 parameters 5.8e-7, m 1.2e-6; `microbatch_mean` at eps 1e-4
    parameters 5.1e-7, m 1.9e-6. At eps 1e-8 `microbatch_mean` measured parameters 1.5e-5 (m 6.1e-6),
    which falls to 7.7e-6 at eps 1e-6 and 5.1e-7 at eps 1e-4. That points to elements with |g| near
    eps, where Adam's `m / (√v + eps)` is ill-conditioned (inferred from this eps sweep and the
    agreement of loss / grad norm / m / v; the per-element analysis was not repeated here), so the
    test uses eps 1e-4 there rather than a looser tolerance. All use the existing tolerances (1e-6, 3e-6, 5e-6, 2e-5).
33. **Checkpoint replacement**: see item 23. Tested with `-Wl,--wrap=rename` on the C test binary
    (the `.tmp → checkpoint-<step>` rename of a replacing save fails; then the state a crash between
    the two renames leaves): the previous checkpoint stays byte-identical and resumable. If the
    rename back fails as well, the error says the previous checkpoint is left as `.old`, and the
    next save of that step restores it. A resume whose `checkpoint-<step>` is missing but whose
    `.old` exists is not repaired automatically (the error is `RRC_ERR_NOT_FOUND`; renaming it back
    by hand works).
34. **`rrc_state.bin` counters** (step, epoch, index, micro_batches, tokens_seen) above 2^62 are
    `RRC_ERR_PARSE`; the loader refuses a start cursor above 2^62 (`RRC_ERR_INVALID_ARG`), which also
    covers `rrc_data --start-epoch`.
35. **Resume into another output directory**: allowed when `output_dir` is the directory of
    `resume_from` (compared by device / inode), when it holds no `metrics.jsonl` and no
    `checkpoint-*` entry, or with `overwrite_output_dir`; otherwise refused like a fresh run (a
    resume used to append to and replace another run's metrics and checkpoints). Outside the
    checkpoint's own directory the metrics file starts afresh with the resumed steps.
    `overwrite_output_dir` does not delete checkpoints that the new run does not write.
36. **RoPE table size**: `max_position_embeddings` from config.json bounds no tensor; with
    `max_positions = 0` (the trainer's default) the table had `n_pos · head_dim` floats without an
    overflow check. Now `n_pos > 2^31 − 1` or an overflowing byte count is `RRC_ERR_UNSUPPORTED`
    before any allocation (the message names `max_positions`, which bounds the table); the
    multiplication by `head_dim` is overflow-checked too, since config.json bounds `head_dim` only
    to be positive and even (tested with `head_dim` 2^60). The stage A
    config parser is unchanged. The table is still built for all `n_pos` positions even when
    `seq_len` is smaller (a few MB at 8B shapes; not changed).
37. **Debug CLIs**: `rrc_fwdbwd` refuses a batch with an empty dimension (tested, including
    `[0, 2^32, 2^32]`; an empty batch used to write an output file with exit 0) or `M · B · T ≥ 2^31`
    (not tested: such a batch file is 8 GB); `rrc_data` checks `T ≤ 2^31 − 1` and
    `B ≤ (2^31 − 1) / T` before forming `T · B` (tested). Both were also run by hand under ASan /
    UBSan with the reviewer's inputs: no report.
38. **`adapter_config.json` checks** (init_adapter / resume): a root that is not an object and a
    `target_modules` that is not a list (PEFT writes a string for a regex target) are refused,
    because the alpha / rslora / dropout / DoRA / layer checks could not run on them. The training
    config's own `"target_modules": "all-linear"` is unaffected.
39. **Memory**: there is no pre-flight memory check (NOT DONE in this step; added in §13 item 5). The trainer logs, after the first
    step, the per-step workspace bytes and the LoRA parameter + gradient + Adam state bytes (base
    weights not included). A reviewer computed from the measuring pass of the workspace layout (not
    run on a GPU, NOT_TESTED): an 8B-class bf16 model with r = 16 on all linears would need about
    20 GB at B = 1, T = 4096 with `recompute = full`, 27 GB at B = 4, 36 GB at B = 8, and 39 GB
    with `recompute = none` at B = 1, excluding GEMM scratch, the CUDA context and cuBLAS workspace.
    `recompute = none` is meant for small models only; an infeasible configuration is reported only
    when the backend allocation fails.

## 11. Stage B CUDA line

Status: everything in this section that runs on a GPU is **COMPILED-ONLY** (nvcc 12.9,
`sm_120` + `compute_120` PTX, `-Werror`; never executed: there is no GPU in the development
environment). What is **TESTED (CPU)** is the pure-C part (GEMM plan, ladder driver, the kernel
bodies, the self-test harness) through a host emulation of the backend. Nothing here is a
performance claim; every performance property is NOT_TESTED. The supported-type table of §11.2 is
an expectation, not a measurement, until `rrc_selftest` has run on the device.

### 11.1 Files and build

| path | contents | status |
| --- | --- | --- |
| `kernels_common/rrc_kc_items.h` | `RRC_HD` per-item bodies of every kernel (parameters structs, lane partials, fixed trees, finalizers, element items, range checks) | TESTED (CPU, via kc-emu) + compiled for sm_120 |
| `cuda/gemm_plan.{h,c}` | pure host row-major → column-major translation, promotion/chunking, ladder driver `rrc_gemm_execute` | TESTED (CPU) |
| `cuda/kc_emu.{h,c}` | `kc-emu`: an `rrc_backend` on host memory running the item bodies in shuffled order and the GEMM ladder with a simulated library | TESTED (CPU) |
| `cuda/selftest_ops.{h,c}` | backend-agnostic op self-test (the body of `rrc_selftest`) | executed on CPU against kc-emu and the CPU backend |
| `cuda/selftest.c` | `rrc_selftest` (CUDA) and `rrc_selftest_host` (`-DRRC_SELFTEST_HOST`) | host build executed in `make test`; CUDA build linked only |
| `cuda/rrc_cuda.h` | public API: `rrc_cuda_backend_create[_ex](device, [opts,] &be)`, `rrc_cuda_backend_destroy`, options, device info, last GEMM path | — |
| `cuda/cuda_kernels.cu`, `cuda/cuda_internal.h` | the `__global__` shells and their launchers | COMPILED-ONLY (+ HOST-ONLY execution, §11.4) |
| `cuda/cuda_backend.cu` | stream, cuBLASLt handle/workspace, descriptor cache, error flag, op table | COMPILED-ONLY (+ HOST-ONLY execution, below) |
| `tests/cuda_host/{shim.h,fake_cuda.{h,cpp},test_cuda_host.c}` | HOST-ONLY test build: the real `cuda_kernels.cu` / `cuda_backend.cu` compiled by g++ against a fake CUDA runtime and a cuBLASLt stand-in, kernels on host threads (§11.4) | executed by `make test` / `make asan` when `CUDA_HOME` is set (headers only) |
| `tools/fetch_cuda_toolkit.sh` | reproduces the compile-only toolkit from pinned PyPI wheels + conda packages (sha256-checked) | run once by hand from cached package files (`--from`); never run by tests |

```sh
make -C rrc clean cpu test                       # includes the new CPU tests and rrc_selftest_host
make -C rrc asan                                 # C tests + rrc_selftest_host with -fsanitize=address,undefined
make -C rrc cuda-objs CUDA_HOME=/path/to/cuda    # compile every cuda/*.cu (CI; no GPU, no link)
make -C rrc cuda CUDA_HOME=/path/to/cuda [ARCH=sm_120]   # build/librrc_cuda.a + build/rrc_selftest
make -C rrc cuda-host-test CUDA_HOME=/path/to/cuda       # HOST-ONLY: real cuda/*.cu on a fake runtime (also in test/asan)
build/rrc_selftest [--device N] [--tf32] [--sync-each-op] [--gemm-level 0|1|2] [--only OP] [--quick]
                   [--scratch-cap N] [--max-grid N] [--no-capped-pass] [--no-grid-pass] [--allow-fallback] [-v]
```

`make -C rrc clean cpu test` without `CUDA_HOME` skips both the nvcc header check and
`cuda-host-test` (it says so); with `CUDA_HOME` both run (`make asan` likewise, under ASan/UBSan).

Link line for users of the backend: `librrc_cuda.a librrc.a -lcublasLt -lcudart -lstdc++ -ldl -lrt`
(plus `-fopenmp -lm -lpthread` for librrc). `rrc_cuda.h` is in `rrc/cuda/` (add `-Irrc/cuda`); the
frozen `backend.h` is unchanged. nvcc flags are explicit: `-fmad=false -prec-div=true
-prec-sqrt=true -ftz=false` (§11.3). `ARCH` is accepted as an alias of `CUDA_ARCH` only when it
looks like `sm_*` (so an exported `ARCH=x86_64` is ignored; `ARCH=sm_100` was compiled once). The CUDA
objects are ordinary prerequisites of `cuda-objs` and `librrc_cuda.a`, so `make -j cuda` builds each
once. The CPU build never needs CUDA.

### 11.2 GEMM: plan, dtype paths, ladder

`rrc_gemm_plan_make` (pure host) turns validated row-major args into column-major cuBLASLt calls:
row-major X (r × c, ld) is column-major Xᵀ, so `C = op(A) op(B)` is computed as
`Cᵀ = op(B)ᵀ op(A)ᵀ`: cuBLASLt's A is rrc's B with `transa = trans_b`, cuBLASLt's B is rrc's A with
`transb = trans_a`, `M = n`, `N = m`, `K = k`, D = C in place with `ld = ldc`, batch strides passed
through (stride 0 = broadcast, interleaved C slices as given). Alignment for
`CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_{A,B,C,D}_BYTES` is the largest power of two ≤ 256 dividing the
byte address, `ld · elsize` and (batched) `stride · elsize` of the actual operand, on every call.

cuBLASLt (CUBLAS_COMPUTE_32F, scale type f32) is expected to support A/B/C = f32/f32/f32,
bf16/bf16/f32 and bf16/bf16/bf16, and no mixed A/B pair or f32×f32→bf16 (no such row in the
cuBLASLt type table as recalled; the header has no table; unverified). Paths at the first ladder
level:

| A | B | C | level 0 (NATIVE) path | cuBLASLt types |
| --- | --- | --- | --- | --- |
| f32 | f32 | f32 | native | f32 f32 f32 |
| f32 | f32 | bf16 | C via f32 scratch, RNE to bf16 once | f32 f32 f32 |
| f32 | bf16 | f32 | B promoted bf16 → f32 | f32 f32 f32 |
| f32 | bf16 | bf16 | B promoted, C via f32 scratch | f32 f32 f32 |
| bf16 | f32 | f32 | A promoted | f32 f32 f32 |
| bf16 | f32 | bf16 | A promoted, C via f32 scratch | f32 f32 f32 |
| bf16 | bf16 | f32 | native (tensor cores) | bf16 bf16 f32 |
| bf16 | bf16 | bf16 | native (tensor cores) | bf16 bf16 bf16 |

Promotion is exact widening; an f32 operand is never narrowed; a bf16 C is rounded once from the
f32 result, as on the CPU. The ladder (`rrc_gemm_execute`, shared by the CUDA backend and kc-emu):
level 0 above → level 1 PROMOTE_ALL (every bf16 operand promoted and bf16 C via scratch, i.e. an
f32×f32→f32 call; skipped when identical to level 0) → level 2 KERNEL (`rrc_kc_item_lt_matmul`, a
SIMT kernel that executes the unpromoted column-major call; always available). A level is used
when cuBLASLt's heuristic returns an algorithm for every chunk; `NOT_SUPPORTED`, zero results or a
failed result state move to the next level (not an error). A scratch allocation failure also moves
on (logged at warn level). `rrc_cuda_last_gemm` reports the path; `rrc_selftest` prints it per
dtype combination. Options: `gemm_min_level` (2 = always the kernel), `allow_tf32` (default off;
env `RRC_CUDA_ALLOW_TF32=1`) selects `CUBLAS_COMPUTE_32F_FAST_TF32` for f32×f32 calls only.

Bounded scratch: promoted operands and the f32 C go through one backend-owned device scratch of at
most `gemm_scratch_cap_elems` floats (default 16 Mi = 64 MiB). The problem is split into chunks of
(batch range, row range of C, column range of C): the batch extent is the largest whose full-size
slices fit (binary search on the actual padded layout), then the row or column extent is halved,
the one that shrinks the scratch most, until it fits; a halving that does not shrink a padded
region is still taken when a smaller extent of that dimension would. Broadcast operands are
converted once per chunk. Bound (TESTED, `gemm_plan.plan_scratch_bound`, 40 000 random plans):
`scratch_elems <= cap` whenever the minimum chunk (one batch slice, one row and one column of C)
fits; otherwise the plan needs exactly what the minimum chunk needs (`scratch_elems` reports it). Example (tested): `dh = dlogits[4096, 152064] (f32) × W_lm[152064, 4096] (bf16)`
promotes W in column chunks within 64 MiB instead of 2.3 GiB. Scratch leading dimensions ≥ 64 are
padded to multiples of 8 elements (aligned columns).

Semantics-relevant cuBLASLt settings: `CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK =
COMPUTE_TYPE` (split-K partials only in fp32: the `INPLACE` / `OUTPUT_TYPE` schemes would add
partial sums in the output type, i.e. round to bf16 several times). cublasLt.h does not say
whether a mask without a bit for `NONE` (= 0) still admits the non-split-K algorithms, so when
that query returns nothing a second query with the default mask takes the first of up to 8
results whose own configuration is non-split (`CUBLASLT_ALGO_CONFIG_SPLITK_NUM <= 1`) or uses
`REDUCTION_SCHEME_COMPUTE_TYPE` (read with `cublasLtMatmulAlgoConfigGetAttribute`). The selection
logic is HOST-ONLY tested against a stand-in whose masked query finds nothing and whose unmasked
query lists a split-K `INPLACE` algorithm first; the real library's mask semantics are NOT_TESTED; `beta == 0` on a scratch path
zero-fills the scratch instead of loading C; on native paths C is the caller's buffer and the
selftest runs NaN-prefilled C with `beta = 0` on every path. Descriptors, layouts and the heuristic
result are cached per call geometry (64 entries, keyed on shapes, strides, types, alignment, TF32),
so repeated calls of one shape use one algorithm.

Determinism of GEMM: the same call geometry uses the same cached algorithm; cuBLAS documents
bit-wise reproducibility for a given toolkit, architecture and SM count. Calls of equal shape but
different pointer alignment may pick different algorithms and differ in the last bits. Checked by
the selftest's run-twice comparison (NOT_TESTED until run).

### 11.3 Kernels and determinism

Every `__global__` kernel is a thin shell (grid-stride index loop + bounds check, or the row
skeleton below) over a body in `rrc_kc_items.h` built on the existing `rrc_kc_*` math. Element
items (embedding, bias, rope pair, swiglu (dgate, dup) pair, add/axpy/scale/convert, adamw, GEMM
conversion and fallback GEMM element) read all inputs before writing and never read another item's
output, so every exact alias allowed by backend.h is safe in any execution order. `bias_grad` is
one item per column summing rows sequentially (the CPU f32 order).

Row reductions (rmsnorm fwd/bwd sums, softmax max/sum and p·dp, cross-entropy log-sum-exp) use one
block of `RRC_KC_LANES = 256` threads per row: lane l accumulates elements l, l+256, … in fp32, then
a fixed pairwise tree (`part[l] = part[l] ⊕ part[l+s]`, s = 128 … 1, one `__syncthreads` per level),
lane 0 finalizes the row statistics (rstd, coefficient, max, 1/Z, loss — the loss reads `z[label]`
before any gradient write), then the element items run. `sum_squares` reduces fixed chunks of 4096
elements the same way into a partial buffer and reduces the partials in one more block. The order
depends only on the shape — not on gridDim, blockDim choice or scheduling — so results are
bit-identical run to run (TESTED on CPU by running every case twice with different item orders;
on the device checked by `rrc_selftest`). The order differs from the CPU backend's sequential sums,
so reductions agree with `rrc_backend_cpu_f32acc` within a reassociation tolerance, not bitwise.

Device math: `-fmad=false` (no contraction of `a*b+c`; the bodies call `fmaf` exactly where the
CPU semantics have one, e.g. AdamW), IEEE `/` and `sqrtf`, denormals kept, `RNE` stores through
`rrc_f32_to_bf16`. With these, ops without transcendental functions (embedding, bias, rope, add,
axpy, scale, convert, adamw, the kernel GEMM) are expected bit-identical to the CPU backend.
CUDA's `expf`/`logf` (≤ 2 / 1 ulp) are not glibc's, so swiglu, softmax and cross entropy differ in
the last bits.

### 11.4 Execution coverage without a GPU

`kc-emu` (`cuda/kc_emu.c`) is an `rrc_backend` whose ops run the same bodies with the same phases
as the kernels, each phase over a fresh pseudo-random permutation of its items (the stand-in for GPU
scheduling), and whose gemm runs `rrc_gemm_execute` with a simulated library (configurable
refusals by type triple and minimum alignment). Tests (`make test`):

* `gemm_plan.plan_reference_matches_cpu_gemm`: a column-major reference (`rrc_kc_item_lt_matmul`,
  the device fallback body, in reverse item order) consuming the plan's chunks, conversions and
  calls reproduces `rrc_gemm` on `rrc_backend_cpu_f32acc` **bit for bit** for 8 dtype combinations
  × 4 transposes × {plain, batched, broadcast A, broadcast B, interleaved C} × 3 levels × {default
  cap, 40-element cap (multi-chunk)} × beta {0 with NaN-filled C, 0.75}; scratch is allocated with
  exactly `scratch_elems` (overruns are caught by the ASan build). Plus the promotion table, the
  operand swap, bounded scratch for the LM-head case and 2-D chunking, alignment, TF32 propagation,
  and the ladder with simulated refusals (refused bf16 types → level 1; misaligned bf16 view →
  promoted into aligned scratch; misaligned f32 → kernel).
* `gemm_plan.plan_padded_ld_and_batch_chunks`: padded scratch leading dimensions (≥ 64, not a
  multiple of 8) and batch chunks with 1 < nb < batch, bit-identical to the CPU backend.
* `kc_items.*`: the selftest harness with kc-emu as the device against `rrc_backend_cpu_f32acc`,
  in strict mode (GEMM on every path, swiglu and bias_grad compared bitwise, since the emulator
  uses the CPU's arithmetic; the other classes as in §11.6) (all ops, aliases, range errors with immediate and deferred reporting, forced kernel and
  promote-all GEMM paths with a tiny scratch cap); the same harness with the double-accumulation
  CPU backend as the device (its tolerances accept a correct backend with different arithmetic);
  bad token ids neither read out of bounds nor write.
* `rrc_selftest_host --backend emu --deferred --quick` runs the program's `main` path; `make test`
  also checks that `--reject-mask 0xFF` (no library) and `--only nonexistent_op` exit non-zero.

HOST-ONLY execution of the real CUDA sources (`tests/cuda_host`, run by `make test` / `make asan`
when `CUDA_HOME` is set; needs the toolkit headers only). `cuda_kernels.cu` and `cuda_backend.cu`
are compiled by g++ as C++ with `shim.h` force-included; `fake_cuda.cpp` implements the runtime
entry points on host memory (live-pointer tracking, injectable `cudaMalloc` failure) and a
cuBLASLt stand-in (documented column-major semantics, the assumed type table, checks of the
promised alignment and of descriptor consistency). Kernel launches go through `RRC_CU_LAUNCH`
(nvcc: exactly `<<<grid, block, 0, s>>>`): row kernels run one host thread per CUDA thread with
a barrier for `__syncthreads` and function-local statics for `__shared__` (blocks in order);
element kernels (`RRC_CU_LAUNCH_NOSYNC`) run their threads one after the other. This executes the
kernel shells (grid/row-stride loops, shared-memory trees, lane-0 finalizers, the error flag),
the launch geometry and the whole backend (descriptor cache, ladder, chunking, scratch, error
paths) — not device code generation, the real cuBLASLt, concurrency between blocks, or timing.
`test_cuda_host`: sum_squares partial-buffer growth failure (no dangling pointer, no double
free), the error flag naming one op with that op's index, the reduction-scheme query fallback,
and the whole selftest harness through the backend (default grid; a 3-block grid with a
1024-float scratch cap, where every grid-stride and row-stride loop wraps and GEMMs chunk; the
stand-in refusing everything, which must fail). `rrc_selftest_shim` is `selftest.c`'s CUDA `main`
on the fake runtime (pass logic, options, failing exit statuses).
* Mutation checks done while writing this (not automated): swapping trans flags, a wrong chunk
  offset, a wrong broadcast stride, a too-small C scratch stride, a dropped post-conversion, and
  four item-body mutations were each caught.

### 11.5 Errors, memory and stream semantics

One in-order non-blocking stream per backend. Range errors: embedding ids, rope positions and CE
labels are checked in the item before the dependent read; a failing item writes nothing and sets a
device flag (first op id via `atomicCAS`, then the smallest offending row/token of that op via
`atomicMin`, taken only by items whose CAS returned 0 or the same op: an item of a later failing op
never lowers the index reported under the first op's name). `sync` and `memcpy_d2h`
(both wait for the stream) read and clear the flag and return `RRC_ERR_RANGE` naming the op;
`sync_each_op` (env `RRC_CUDA_SYNC_EACH_OP=1`) makes every op synchronize, so the op itself
reports. Unlike the CPU backend, which validates before writing, the outputs of the ops enqueued
since the previous sync are unspecified after a device range error. Launch failures and CUDA
errors are `RRC_ERR_BACKEND` (allocation failures `RRC_ERR_OOM`).

`memcpy_h2d` goes through a backend-owned pinned staging ring and returns without a stream
synchronize (the contract — the host source may be reused on return — is kept; §15.3; before
§15 it was `cudaMemcpyAsync` + stream synchronize, which remains the path when the ring is
disabled). `memcpy_d2d`/`memset0` are asynchronous on the stream; `free` synchronizes the stream, then
calls `cudaFree`. The GEMM scratch and the sum_squares partial buffer grow on
demand (stream synchronize before freeing the old buffer); `grow()` always updates the owning
backend members, so after a failed growth they are (NULL, 0), never a freed pointer. All calls on one backend must come from
one host thread (backend.h); the backend calls `cudaSetDevice` at creation only.

### 11.6 rrc_selftest

The first program to run on the 5090. It prints device name, compute capability, SM count, total
and free memory, driver / runtime / cuBLASLt versions and the options; then for every op runs random
inputs (odd sizes, padded leading dimensions, batched/broadcast/interleaved/misaligned/fused-block
GEMMs, a dense aligned 256×384×512 GEMM per dtype combination, and the in-place / alias uses: rmsnorm y = x,
dx = dy, dx = x; rope y = x; softmax p = s, ds = dp, ds = p; swiglu h = gate block, in-place and fused
[dgate | dup] backward; add out = a; axpy y = x; gemm C as a column block beside A; cross entropy
in place) on the CUDA
backend and on `rrc_backend_cpu_f32acc`, prints per-op case counts, the worst error/tolerance ratio
and PASS/FAIL (every case with `-v`, failing cases always, with max abs/rel error and the worst
element), prints the GEMM path per dtype combination and a tally of ladder levels per GEMM layout (plain,
batched, broadcast A/B, interleaved C, misaligned, fused column blocks, aligned), naming every case
that fell back to the kernel GEMM (a correct but slow path that would otherwise be invisible
behind a PASS), checks range-error reporting (op or next
sync, then a clean sync), and runs every case twice requiring bit-identical results. Exit status 0
only if everything passes, and never when no check ran (e.g. `--only` matching no op).

GEMM path requirement (`rrc_st_opts.require_lt`, on unless `--gemm-level` > 0 or
`--allow-fallback`): the dense aligned case of every dtype combination must not run on the kernel
fallback, that of f32/f32/f32, bf16/bf16/f32 and bf16/bf16/bf16 must run natively (level 0), and
at least one GEMM case must reach cuBLASLt; a violation is a FAIL. This turns the note of a
correct-but-naive fallback into a failure for exactly the cases cuBLASLt is expected to serve, so a
broken integration (wrong preference attribute, alignment, an unexpected type table) cannot pass
on the first GPU run; misaligned, interleaved and other layouts may still fall back with a note.
Extra passes after the main one: a *capped pass* (every GEMM case again with a 1024-float scratch
cap, which must chunk at least one case: per-chunk offsets, alignment and descriptor keys, scratch
reuse on one stream, the bf16 post-conversion; skipped with `--scratch-cap N` or
`--no-capped-pass`) and, in the CUDA build, a *grid pass* (every op with at most 3 blocks per
launch via `rrc_cuda_options.max_grid_blocks`, so every grid-stride and row-stride loop wraps at the
selftest's sizes; `--max-grid N` / `--no-grid-pass`). Softmax cases include -inf scores (about
10%, a row of all -inf, -inf on the causal diagonal so row 0 of a causal case has only -inf
allowed) for f32 and bf16 s, and sum_squares includes n = 2²⁰ + 4097 (257 partials: the final
block's lane loop wraps). Tolerances (applied per element, `scale` = max |ref| of the output):

| class | ops | tolerance |
| --- | --- | --- |
| EXACT | memory, embedding, add_bias, rope fwd/bwd, add, axpy, scale, convert, adamw | bitwise |
| ULP | swiglu fwd/bwd (`expf`) | 2e-6 · (\|ref\| + scale) |
| REDUCE | rmsnorm fwd/bwd, softmax fwd/bwd, cross entropy, sum_squares, bias_grad | (1e-5 + n·2⁻²⁴) · (\|ref\| + scale), n = terms per reduction (the CPU reference's own sequential error bound) |
| GEMM | gemm (any path) | 2·k·2⁻²⁴ · (\|α\|Σ\|a\|\|b\| + \|β\|\|c\|) (+ 2⁻⁹ of that with `--tf32`) |

plus one bf16 ulp (2⁻⁷ · max(\|ref\|, \|got\|)) for bf16 outputs of non-EXACT ops. It measures and
prints no timing. The tiny-model end-to-end comparison, the 20-step training check of §7 and the
`--report` JSON were added with §15 (§15.1, §15.2).

### 11.7 Not done / open

* Nothing has run on a GPU: kernel launch code, cuBLASLt calls, the supported-type table, the
  alignment behaviour of cuBLASLt with 2-byte-aligned views, and determinism on the device are
  unverified until `rrc_selftest` runs.
* No performance work: the fallback GEMM kernel is a naive one-thread-per-element kernel, the
  row kernels use one block per row (inefficient for very short rows such as per-head q/k norm
  with head_dim 128), no fused attention, no CUDA graphs (pinned upload staging: §15.3).
* Single device per process thread (`cudaSetDevice` only at creation).
* Not executed anywhere: element kernels with more than 2²⁸ items or row kernels with more than
  2²⁰ rows at the default grid cap (the wrap paths themselves run through the grid pass and the
  HOST-ONLY build with a 3-block cap; the int64 index arithmetic beyond 2³¹ only by reading).
* The HOST-ONLY build's cuBLASLt stand-in encodes the same type table and alignment rules the plan
  assumes, so its passes are no evidence for them; the reduction-scheme mask semantics of the real
  library are NOT_TESTED.

### 11.8 Review fixes of the CUDA line (decisions)

* sum_squares partials: `grow()` is called on the backend members themselves (`ss_part`,
  `ss_part_bytes`), as for the GEMM scratch. Before, a failed growth left the freed pointer and its
  old capacity in the backend (use-after-free by the next call's kernel, second `cudaFree` at
  destroy). Regression: `cuda_host.sum_squares_grow_failure`. With only the fix reverted (in the
  tree that has the host build), it fails without ASan (`cudaFree` of a dead pointer, then
  `RRC_ERR_BACKEND`) and, in the ASan build, ASan reports a heap-use-after-free WRITE in
  `k_ss_chunks` of memory freed by the failed growth.
* Error flag: the index is lowered only by items of the recorded op
  (`cuda_host.error_flag_op_and_index`; with only the fix reverted it reports
  `embedding_fwd` without its offending token 5).
* NaN logits in cross entropy: `rrc_kc_lse_push` poisons the state with a NaN input and
  `rrc_kc_lse_merge` returns a NaN state before its -inf shortcuts, so the device tree keeps a NaN
  that the CPU's serial reduction (and torch) propagate; non-NaN inputs take the same branches as
  before (CPU results unchanged). Regression: `kc_items.emu_xent_nan_propagates`.
* GEMM plan bound: see §11.2 (`gemm_plan.plan_scratch_bound`; the old linear fit and early stop
  gave up to 21x the cap).
* Selftest: path requirement, capped and grid passes, -inf softmax cases, empty-run failure (§11.6).
  This replaces the earlier decision that fallback notes are warnings only.
* `rrc_cuda_options.max_grid_blocks` (default 0 = 2²⁰ blocks) and the launch macros are additions
  to the CUDA line's own API; `backend.h` and the op wrappers are unchanged.

## 12. Integration of the stage B lines

The model line (§10) and the CUDA line (§11) were merged with merge commits; neither changed
`backend.h` or the op wrappers, and the merged tree builds and tests both. Decisions:

1. **Build**: one Makefile with both lines' targets. The sanitizer variable is `RRC_SAN_FLAGS`
   (the CUDA line's `SAN` is kept as an alias). `make asan` runs the whole `make test` (C tests,
   `rrc_selftest_host` and its negative checks, with `CUDA_HOME` the HOST-ONLY `cuda-host-test`)
   with the model line's flags (`-fno-sanitize-recover=all`, leak detection, `OMP_NUM_THREADS=2`);
   before, each line's `asan` covered only its own part. The C test binary links both lines'
   objects with `--wrap=malloc` and `--wrap=rename`.
2. **`"backend": "cuda"` in the trainer**: the config value is parsed everywhere (so
   `--print-schedule` works with it) and is part of the resume fingerprint. `librrc.a` stays free of
   CUDA symbols: `rrc_train_run_on(config, device_backend)` takes a caller-supplied backend.
   `src/rrc_train.c` is compiled twice: `build/rrc_train` (CPU only; `rrc_train_run` refuses
   `"cuda"` with RRC_ERR_UNSUPPORTED before creating the output directory or loading anything) and
   `build/rrc_train_cuda` (`-DRRC_WITH_CUDA`, its own object file, linked with `librrc_cuda.a`; built
   by `make cuda`). It creates the CUDA backend on device 0 and destroys it after
   `rrc_train_run_on` has freed every buffer of the run. No new config key (device selection is
   `CUDA_VISIBLE_DEVICES`). COMPILED-ONLY: never run on a GPU.
3. **Host access audit**: the model, LoRA and trainer code reach backend memory only through the op
   wrappers and `rrc_buf_upload` / `rrc_buf_download` (the only pointer use is address arithmetic
   for the gate/up view in `model.c`). This was first checked by reading the code (the fake runtime
   of `tests/cuda_host` maps device memory to host memory and would not catch a direct access); it
   is now checked by running on a backend whose device pointers fault on host access (§14).
4. **HOST-ONLY trainer run**: `make cuda-host-train CUDA_HOME=...` links the `rrc_train_cuda` main
   with the fake runtime (`build/rrc_train_shim`), so the model and trainer drive the real
   `cuda_backend.cu` / `cuda_kernels.cu` on host threads. `rrc/tests/test_e2e_cpu.py` runs it on the
   tiny qwen3 model when `CUDA_HOME` and `RRC_TEST_HOST_SHIM_TRAIN=1` are set (opt-in: the fake
   runtime runs one host thread per CUDA thread, about four minutes per 4-micro-batch step) and
   compares losses, gradient norms and the saved adapter with the CPU run. On one run (f32, 3
   steps) the losses agreed to about 1e-8 relative. This is evidence about the host-side code and the
   kernel bodies only: no device, no real cuBLASLt, no inter-block races.
5. **End-to-end CPU smoke**: `rrc/tests/test_e2e_cpu.py` (tiny qwen3 + tokenizer trained in the test
   -> `prepare_sft.py` -> `rrc_train` cpu -> `metrics.jsonl`, checkpoints -> `PeftModel`). It is a
   plumbing check, not a numerical one (`test_train_vs_torch.py` is) and not a GPU validation.

## 13. Stage C census: GEMM paths and memory at 8B shapes (computed, not measured)

Status: the dry-run backend, the shape-only model, `rrc_census` and the trainer's memory pre-flight
are **TESTED (CPU)**: they are pure host C and are checked against real CPU runs of tiny models
(item 6). Every number they print for an 8B model is **computed** from the op-call sequence of one
training step; nothing ran on a GPU and nothing in this section is a measurement. "Native" means the
level-0 path of the CUDA GEMM plan (§11.2) under its assumed cuBLASLt type table, which is
NOT_TESTED on a device. The 8B shapes are **assumed**: built-in presets typed in from the public
model cards of Qwen3-8B, Qwen2.5-7B and Llama-3.1-8B, NOT_TESTED against the real `config.json`
files (huggingface.co is not reachable here); `rrc_census --config` takes the real file.

1. **Dry-run backend** (`include/rrc/dryrun.h`, `src/dryrun_backend.c`): an op table that computes
   nothing (`host_accessible = 0`). `alloc` returns fake handles in the upper (kernel) half of the
   address space (`0xffff800000000000 + offset`, 256-byte aligned, never reused): a user-mode
   dereference faults on x86-64 and AArch64, distinct allocations never overlap, so `ops.c`'s alias
   checks and the plan's alignment computation see a consistent address space. It tracks live and
   peak bytes; `memcpy_h2d/d2d`, `memset0`, `sync` only count; `memcpy_d2h` zero-fills the host side
   (the loss download and the gradient norm read 0, deterministically). Every op records its shapes,
   dtypes and computed bytes moved (one read of each input, one write of each output); `gemm` also
   records m, n, k, batch and the A/B/C dtypes and runs `rrc_gemm_plan_make` at level NATIVE plus
   every chunk (`rrc_gemm_plan_chunk` with a fake scratch base): promotions, C scratch, chunks,
   scratch bytes, converted elements and bytes, the smallest operand alignment. The plan
   (`cuda/gemm_plan.c`, pure host C) moved from `librrc_cuda.a` into `librrc.a` for this;
   `librrc_cuda.a` resolves it from there (link order `librrc_cuda.a librrc.a`, unchanged).
2. **Shape-only model** (`rrc_model_create_shape_only(be, cfg, opts, &m)`): the loader's weight
   layout (one allocation in expected-tensor order, the fused `[gate; up]` view), views and RoPE
   allocation built from a config alone; nothing is read or computed, the zero-filled allocations
   stay zero. `rrc_model_load` now calls the same builder; its behaviour (error order, statuses,
   `*out = NULL` on failure) is unchanged (the existing model tests pass unmodified).
3. **Finding and fix.** The census found exactly one class of base-weight GEMM off the native bf16
   path: the input gradients through the frozen k_proj / v_proj weights, `dxn += dK Wk` and
   `dxn += dV Wv`, whose A operand is the f32 dK / dV accumulator (§10 item 3), so the plan promotes
   the bf16 weight (f32 × bf16 → f32). That is 2 GEMMs per layer: 72 / 56 / 64 calls per step and
   2.47 / 0.84 / 2.20 TFLOP (computed) for the three presets at T = 4096. Every other base-weight
   GEMM (q/o/gate-up/down forward and input gradients, the LM head both ways) was already
   bf16 × bf16. Fix (model code): with a bf16 model the layer backward converts dK, then dV, once to
   bf16 into `w->dob` (free after `attn_bwd`; Q ≥ KV) and runs those GEMMs bf16 × bf16 → f32; the
   accumulators stay f32 and the LoRA k/v backward keeps reading them (fp32, like PEFT), so only the
   base-weight input gradient sees the rounded copy — HF in bf16 keeps dK itself in bf16. Numerics
   (one-off A/B, not a test; the window of `test_lora_bf16_matches_peft`, mean relative L2 error of
   the LoRA gradients against fp32 on the same bf16-rounded weights, `cpu` backend; `cpu-f32acc`
   alike): llama3_rope 0.00750 → 0.00754, qwen2_tied 0.00820 → 0.00821, qwen3_hd32_bias 0.00976 →
   0.00977, HF-bf16 itself 0.00952 / 0.00999 / 0.01088; the worst per-tensor ratio to the test's
   tolerance `2·e_HF + 0.005` is unchanged (0.430 / 0.431 / 0.583); logits and loss are
   bit-identical (forward untouched). The "before" values reproduce §10 item 3's exactly. The A/B
   covers tiny models at T = 23 only; the effect at 8B shapes and T = 4096 is NOT_TESTED. HOST-ONLY
   (by hand): one bf16 step of a tiny qwen3 model through `build/rrc_train_shim` (the real
   `cuda_backend.cu` on the fake runtime, so the new convert and the bf16 k/v GEMMs went through the
   CUDA backend's code) gave the same loss as `cpu-f32acc` bit for bit and a gradient norm within
   1.8e-7 relative. Tests:
   `census.census_8b_presets_base_weight_gemms_native` (exact call counts per role and every
   base-weight call on a bf16 × bf16 combination, for all three presets; it failed on the code before
   the fix with "base_dx: 72 of 216 ...") and the same assertion in `rrc/tests/test_census.py`.
4. **`rrc_census`** (`include/rrc/census.h`, `src/census.c`, `build/rrc_census`): builds the
   shape-only model, the LoRA adapter (`rrc_lora_create`) and the AdamW state
   (`rrc_optimizer_create`) on a dry-run backend, then runs two optimizer steps exactly as the
   trainer does (zero grads; `grad_accum` micro-batches of forward + backward; clip + AdamW; sync) on
   rows of one full-length document, every token labelled (the worst case for attention and the LM
   head), and reports the second step (steady state: the first also allocates the workspace and
   grows the scratch). GEMM roles are decided by operand addresses (B inside the LM-head weight →
   `lm_head_*`, inside the other base weights → `base_fwd` / `base_dx` by `trans_b`, any operand in the
   LoRA parameters / gradients → `lora`, the rest → `attention`). JSON: FLOPs by (A, B, C dtypes, plan
   path) and by role, native and native-bf16 fractions, conversion bytes, syncs and transfers, per-op
   calls and bytes, memory by category; every field labelled computed.

   Tables (computed; presets at seq_len 4096, micro-batch 1, LoRA r 16 / alpha 32 on all seven
   modules of every layer, recompute full, attn_chunk 256, ce_chunk 1024, bf16, CUDA target; after
   the fix of item 3, "before" in parentheses):

   | preset | base params | LoRA params | GEMM calls / step | GEMM TFLOP / step | native share | native bf16 share |
   | --- | --- | --- | --- | --- | --- | --- |
   | Qwen3-8B | 8.191 B | 43.65 M | 39190 | 189.79 | 0.9244 | 0.9238 (0.9107) |
   | Qwen2.5-7B | 7.616 B | 40.37 M | 20694 | 169.64 | 0.9404 | 0.9400 (0.9350) |
   | Llama-3.1-8B | 8.030 B | 41.94 M | 34806 | 185.66 | 0.9308 | 0.9302 (0.9184) |

   By role (TFLOP per step and share of all GEMM FLOPs; A B C dtypes: calls):

   | role | Qwen3-8B | Qwen2.5-7B | Llama-3.1-8B | combinations (plan path) |
   | --- | --- | --- | --- | --- |
   | base_fwd | 97.79 (0.515) | 89.98 (0.531) | 97.65 (0.526) | bf16 bf16 bf16 (native) |
   | base_dx | 56.90 (0.300) | 53.46 (0.315) | 57.17 (0.308) | bf16 bf16 f32 (native: q, k, v, gate-up into the f32 `dxn`), bf16 bf16 bf16 (native: o, down) |
   | lm_head_fwd | 5.10 (0.027) | 4.46 (0.026) | 4.30 (0.023) | bf16 bf16 bf16 (native) |
   | lm_head_dx | 5.10 (0.027) | 4.46 (0.026) | 4.30 (0.023) | bf16 bf16 bf16 (native) |
   | attention | 23.51 (0.124) | 15.97 (0.094) | 20.88 (0.113) | bf16 bf16 f32 scores / dP (native); f32 bf16 bf16 P·V / dQ (promote B, C scratch); f32 bf16 f32 dV / dK (promote B) |
   | lora | 1.40 (0.007) | 1.30 (0.008) | 1.35 (0.007) | f32 f32 f32 (native); f32 f32 bf16 `y += s t Bᵀ` (C scratch); bf16 f32 f32 `t`, `dB`, `dt` (promote A); f32 bf16 f32 `dA` (promote B) |

   The base-weight GEMM call counts are exact formulas of the code (forward 6L, recompute 5(L−1)
   without down_proj, input gradients 6L, LM head ⌈4095 / 1024⌉ = 4 each way), and every role's
   FLOPs equal closed-form expressions of the shapes written independently in
   `rrc/tests/test_census.py`. Native-bf16 share of the attention FLOPs: 0.444 for all three (scores
   in forward, recompute and backward, and dP: 4 of the 9 equal-sized GEMM sets per recomputed
   layer); the rest runs promoted because P and dS are f32 (§10 item 3). No GEMM of these steps has
   an operand aligned to less than 16 bytes (`align_lt16_calls` = 0).

   Conversions, transfers, synchronisation (per step, computed):

   | preset | plan conversions (bytes read + written; before fix) | max GEMM scratch | sync | memcpy_h2d | memcpy_d2h | memset0 | CUDA host waits |
   | --- | --- | --- | --- | --- | --- | --- | --- |
   | Qwen3-8B | 368.0 GB (369.8 GB) | 64 MiB | 2 | 6 (98 296 B) | 2 (16 384 B) | 74 | 10 |
   | Qwen2.5-7B | 335.9 GB (336.6 GB) | 56 MiB | 2 | 6 (98 296 B) | 2 (16 384 B) | 58 | 10 |
   | Llama-3.1-8B | 353.9 GB (355.5 GB) | 64 MiB | 2 | 6 (98 296 B) | 2 (16 384 B) | 66 | 10 |

   The LoRA branch causes 87 % / 92 % / 88 % of the conversion bytes (Qwen3-8B: 319.9 of 368.0 GB,
   of which 157.7 GB is the f32-scratch round trip of the bf16 C of `y += s t Bᵀ` and 129.6 GB the
   widening of bf16 activations / output gradients for `t = x Aᵀ`, `dB`, `dt`); the rest is attention
   (P·V, dQ, dV, dK: 48.1 / 26.0 / 42.7 GB). "CUDA host waits" = sync + memcpy_h2d +
   memcpy_d2h + free, the calls on which the CUDA backend waits for its stream (§11.5). Since the
   integration with §15 the pinned h2d staging ring is on by default: a staged `memcpy_h2d` waits at
   most for the event of an earlier copy out of the same slot, not for the whole stream, so the
   column is an upper bound (exact with `RRC_CUDA_H2D_STAGING_BYTES=0`; counted from the code, the
   full-stream waits of a steady-state step are then sync + d2h + free = 4 for each preset).

   Memory (GiB, computed; op-table allocations plus the CUDA backend's internal buffers):

   | preset | weights | RoPE tables | LoRA params + grads | AdamW m, v | workspace | GEMM scratch | cuBLASLt ws | peak total |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | Qwen3-8B | 15.256 | 0.020 | 0.325 | 0.325 | 2.729 | 0.062 | 0.031 | 18.750 |
   | Qwen2.5-7B | 14.185 | 0.062 | 0.301 | 0.301 | 2.529 | 0.055 | 0.031 | 17.465 |
   | Llama-3.1-8B | 14.958 | 0.062 | 0.312 | 0.313 | 2.614 | 0.062 | 0.031 | 18.353 |

   (sum_squares partials ≈ 40 KiB each; not included: the CUDA context, allocator rounding and
   fragmentation, host memory.) This agrees with the reviewer's estimate of §10 item 39 (about 20 GB
   at B = 1, T = 4096): 18.75 GiB = 20.1 GB for the Qwen3-8B preset. The RoPE tables cover
   `max_position_embeddings` (40960 / 131072 / 131072 rows).
5. **Memory pre-flight** (`rrc_train_memory_preflight`, run by `trainer_setup` before the model is
   loaded, anything is allocated on the backend or `output_dir` is created): the census dry run of
   one step of the config's shapes (`grad_accum` 1; memory does not depend on it; CPU target for the
   CPU backends, CUDA target on `"cuda"`), then the peak against `memory_budget_bytes`: null
   (default) = automatic — device total memory minus `RRC_TRAIN_CUDA_MEMORY_MARGIN` (1 GiB, an
   assumption for the CUDA context and fragmentation, NOT_TESTED) on `"cuda"` (`build/rrc_train_cuda`
   passes `cudaMemGetInfo`'s total via `rrc_train_run_on_device`; unknown total → a warning and no
   check, as on the HOST-ONLY fake runtime, which reports 0), unlimited on the CPU backends; 0 = no
   check; N > 0 = N bytes. Over budget: `RRC_ERR_OOM` with the breakdown and the remedies. The
   seq_len / pad_token_id checks now also run on `config.json` before the pre-flight (same messages
   and statuses, so a too-long `seq_len` is not reported as a pre-flight failure). Not part of the
   resume fingerprint. Tested on CPU with small fake budgets (`train.train_memory_preflight`: the
   boundary peak / peak − 1, the CUDA automatic budget with a fake device total, the refusal leaving
   no `output_dir`; `test_census.py::test_rrc_train_memory_preflight_refuses_over_budget` through
   `build/rrc_train`). The CUDA path is COMPILED-ONLY; by hand (not part of the tests),
   `build/rrc_train_shim` (HOST-ONLY, the fake runtime reports a total of 0) trained one step of a
   tiny qwen3 model with the "total memory unknown" warning and refused a 1000-byte
   `memory_budget_bytes` before creating `output_dir`.
6. **Census checked against reality** (TESTED, CPU): `census.census_dryrun_matches_cpu_run` runs one
   training step of tiny qwen3 (bf16 and f32, both recompute policies) and llama (MQA) models on a
   counting CPU backend and the same step on the dry-run backend with a shape-only model: identical
   GEMM sequences (every field of every call), equal FLOPs, equal op-table peak and workspace bytes,
   the shape-only layout equal to the loaded one view by view, and `rrc_census_run`'s totals and
   memory categories equal to the CPU run's. `test_census.py` checks the shape-only parameter count
   against transformers' `model.parameters()` for four tiny variants, the FLOPs of every role against
   closed-form formulas at the preset shapes, and the report's fields. The suite runs under
   `make asan`.
7. **Not done / open** (NOT_TESTED unless stated):
   * Whether cuBLASLt accepts each "native" call on the 5090 (type table, alignment, heuristics) —
     `rrc_selftest` on the device; the census reports what the plan would ask for. Of the Qwen3-8B
     preset's native-bf16 share of 0.924, 0.869 is the plain 2-D base-weight GEMMs and 0.055 the
     batched attention scores / dP GEMMs with a broadcast K or V operand (stride 0) and per-head
     views, where library support is least certain.
   * `align_lt16_calls = 0` holds for the dry run's 256-byte-aligned fake allocations and the
     workspace's 256-byte view offsets; the alignment `cudaMalloc` returns is NOT_TESTED here.
   * Attention: half of its FLOPs run promoted (f32 P / dS against bf16 Q, K, V; §10 item 3 keeps them
     f32). A bf16 P as in HF, or the fused attention kernel of §3, would move them to the native path;
     a numerics change that needs its own A/B against the PEFT tolerances.
   * LoRA: its conversions are most of the converted bytes (item 4: ~0.32 TB per step at these
     shapes, computed); the bf16 copy of A / B allowed by
     §3 or a fused LoRA epilogue would remove them; not done (fp32 LoRA GEMMs like PEFT).
   * The presets are unverified transcriptions; real `config.json` files may carry keys outside rrc's
     ignore list, which `rrc_census` / `rrc_train` refuse unless `--allow-unknown-keys` /
     `RRC_CONFIG_ALLOW_UNKNOWN_KEYS` (`rrc_train` has no such option; qwen2's `use_mrope: false`,
     seen on some qwen2-architecture checkpoints, is accepted since item 10 of §9).
   * No time estimate is derived from these counts.

## 14. Device-pointer audit

Question: does any host code outside the backends read or write backend memory directly (a
`rrc_buf.ptr` dereference, `memcpy` / `memset` on it, a pointer handed to a host function),
instead of `memcpy_h2d` / `memcpy_d2h` (`rrc_buf_upload` / `rrc_buf_download`)? On the CPU backend
this works by accident; on the CUDA backend it would crash or read garbage. §12 item 3 answered it
by reading the code. This section makes it a test.

1. **The off-device test backend** (`include/rrc/offdev.h`, `src/offdev_backend.c`; TEST ONLY, not
   in `rrc/rrc.h`). `rrc_offdev_backend_create(inner)` wraps `rrc_backend_cpu()` or
   `rrc_backend_cpu_f32acc()` ("cpu-offdev", "cpu-f32acc-offdev", `host_accessible = 0`). It
   reserves one PROT_NONE, MAP_NORESERVE address range (64 GiB of address space, falling back to 16 /
   4 / 1 GiB; no fallback to host pointers) and hands out device addresses from it: page aligned,
   bump allocated, an unmapped guard page after every allocation, never reused. The bytes are a
   separate allocation of the inner backend. Every memory function and every op translates each
   pointer through a table sorted by device address (base + offset, so views at any element work;
   memcpy / memset also check the extent) and calls the inner backend's function directly with the
   real pointers. Offsets keep their alignment modulo 64 (inner allocations are 64-byte aligned,
   device bases page aligned) and distinct allocations are disjoint in both spaces, so every alias
   / overlap decision of `src/ops.c` (made on device addresses) is the inner backend's. A pointer in
   no live allocation is refused with RRC_ERR_INVALID_ARG; frees of non-base pointers are counted;
   `rrc_offdev_backend_destroy` reports live allocations, bad frees and refused translations as
   RRC_ERR_INTERNAL. Any host access to a device pointer is a SIGSEGV.
2. **Selection** (all TEST ONLY, nothing in the config file, the fingerprint or the outputs changes):
   * C tests: `rrc_tests --offdev [filters]`; `rt_cpu()` / `rt_cpu_f32acc()` (tests/c/ops_util.h)
     return the wrappers, the test trainer helper calls `rrc_train_run_test_offdev`.
   * `rrc_train --test-offdev` (hidden; `rrc_train_run_test_offdev` in train.h wraps the CPU
     backend the config names, so `rrc_train_run_on`'s refusal of a device backend for a CPU config
     stays as documented; "cuda" is refused). The wrapper's counters are printed on stderr.
   * `rrc_fwdbwd --backend cpu-offdev|cpu-f32acc-offdev` (hidden). Counters on stderr.
3. **What runs on it** (TESTED, CPU):
   * `make test` (and `make asan`): `rrc_tests --offdev model. train. offdev.`: every test of the
     model and train suites (LoRA layout / init / safetensors round trip, batch validation and its
     data-dependent errors, finite-difference gradients for qwen3 / llama / qwen2, recompute /
     chunking / bf16 invariances, workspace reuse, zero-label windows, load failures, the GEMM dtype
     recorder, the fused clip + AdamW step, state files, end-to-end stop + resume, failure-safe
     checkpoint replacement, output-dir guards, init_adapter checks) with every buffer off-device.
   * The whole C suite also passes with `rrc_tests --offdev` and no filter (every test that uses
     `rt_backend()` / `rt_cpu()`, e.g. the `grad` and `contracts` suites, then runs off-device);
     `make test` runs only the three suites above.
   * Bit-for-bit comparisons (normal `rrc_tests` run, so also under `make asan`; `make tsan` runs
     only the `data` and `train` suites, so it covers `train_offdev_bit_identical` with the
     prefetch thread, not the model comparison):
     `model.model_offdev_bit_identical`: qwen3 / llama (upper layers only) / qwen2, f32 and bf16,
     recompute full and none, cpu and (qwen3) cpu-f32acc: LoRA load from safetensors, three
     micro-batches with a larger third one (workspace re-plan), loss-only, no-LoRA, microbatch_mean,
     logits and every hidden state, an out-of-range token (RRC_ERR_RANGE on both), workspace size
     and allocation count, and the LoRA params + grads written to safetensors: all equal byte for
     byte (16 combinations). `train.train_offdev_bit_identical`: two settings (f32 /
     window_tokens / recompute full / prefetch off; bf16 / microbatch_mean / recompute none /
     cpu-f32acc / prefetch on) with an init adapter, dump_batches and checkpoints every 2 steps:
     checkpoints (adapter, adapter_config.json, rrc_state.bin), batch dumps and metrics (minus
     wall-clock fields) equal; resuming on offdev from the CPU checkpoint and on the CPU from the
     offdev checkpoint reach the same final bytes.
   * pytest (`-k offdev`, 29 tests): `rrc_fwdbwd` output files identical for every
     tiny variant of the PEFT comparison (llama, MHA, MQA, llama3 RoPE, tied, qwen2, qwen2 tied,
     qwen3, qwen3 head_dim 32 + biases) in f32 and bf16 with non-zero PEFT adapters, `--hidden` and
     `--logits`, plus f32acc, recompute none, microbatch_mean, upper layers only, no LoRA, loss
     only, seeded init and an out-of-range token (same exit status and message); `rrc_train` on the
     8-step PEFT-init trajectory (two settings) with checkpoints, batch dumps and metrics identical,
     and cross-backend resumes byte-identical; `--test-offdev` with a "cuda" config is refused.
   * The death tests of `offdev.offdev_host_access_faults` show the detector fires: forked children
     reading / writing a device pointer (base, a view, the last element, a freed range) fault, the
     same child code on a host pointer completes (under ASan the sanitizer's SEGV report and exit
     status count as the fault).
4. **Result: 0 violations found** in the model, LoRA, trainer, optimizer and checkpoint code; no
   library code was changed for the audit (the trainer only gained the test entry point, a
   refactor of `rrc_train_run_on` into a static `train_run`). `src/data.c`, `safetensors.c`,
   `config.c` and `json.c` hold no `rrc_buf` / backend reference at all, so they cannot touch
   backend memory: that part is shown by inspection, not by the run. What the run covers is what
   the listed tests execute; code paths no test reaches are not covered (e.g. I/O error paths
   inside checkpoint writing are covered only where the existing failure-injection tests reach
   them).
5. **The check was seen to fail** (manual, once, not committed): a `memset(l->grads.ptr, ...)` in
   `rrc_lora_zero_grads` made `rrc_tests --offdev model.` and the normal `rrc_tests model.` (via
   `model_offdev_bit_identical`) die with SIGSEGV; `ss = *(const float *)o->norm.ptr` in
   `rrc_optimizer_step` made `rrc_tests --offdev train_end_to_end` die with SIGSEGV while the
   plain CPU `train.` suite still passed.
6. **Limits**: this proves the absence of host accesses on the paths run, with the CPU kernels as
   the device. It says nothing about the CUDA backend's own code (COMPILED-ONLY / HOST-ONLY, §11),
   stream ordering, or device-side bugs. Cost (computed from the code, not measured as a
   performance number): one binary search over the live allocations per pointer per op.

## 15. GPU first-run plan

Goal: the user's single first run on the RTX 5090 must answer as much as possible, and its output
must come back as files. Status of everything in this section: what runs on the GPU is
**COMPILED-ONLY**; the same code paths were executed **TESTED (CPU)** through host backends (the
emulation backend kc-emu, the CPU backends, two test devices) and **HOST-ONLY** through the real
`cuda/*.cu` sources on the fake CUDA runtime of `tests/cuda_host`. No performance figure exists;
every one is NOT_TESTED until the commands of `bench/README.md` have been run. Files:
`cuda/selftest_e2e.{h,c}`, `cuda/selftest.c`, `cuda/cuda_backend.cu`, `cuda/rrc_cuda.h`,
`bench/rrc_bench.c`, `bench/torch_baseline.py`, `bench/README.md` (the exact commands),
`tests/cuda_host/*`, `tests/test_gpu_tools_smoke.py`.

### 15.1 rrc_selftest end-to-end checks (§7 item 3)

After the op checks (§11.6), `rrc_selftest` runs `rrc_e2e_run(dev)`: for each family (llama with
llama3 rope scaling, untied; qwen2 with q/k/v biases, untied; qwen3 with q/k norm, attention
biases, tied embeddings and head_dim ≠ hidden / heads; all with GQA) and dtype (f32, bf16), a tiny
checkpoint is written **in C** (`config.json` + `model.safetensors`, random weights from a fixed
seed, bf16 tensors stored as BF16) into a fresh work directory and read back by the normal loader;
no Python is needed on the GPU machine. Sizes: `full` (default on the GPU: hidden 96–128, two
layers, vocab 256–384, multiples of 16 so the bf16 GEMMs are tensor-core aligned; 2 rows × 64
tokens, attention chunks of 24 and CE chunks of 40 so every chunk loop runs ragged), `quick` (host
runs in `make test`) and `tiny` (the HOST-ONLY build). Checks, per family × dtype:

| check | what is compared (dev vs `rrc_backend_cpu_f32acc`) |
| --- | --- |
| `logits`, `loss` | `rrc_model_forward_logits` and the loss sum of `rrc_model_forward_backward` on one packed micro-batch (several documents per row, prompt tokens unlabelled) with a randomized non-zero adapter (A and B) |
| `grad`, `grad_recompute_none` | every LoRA gradient (seven projections, every layer), per tensor; the second with recompute `none` on dev |
| `train_loss`, `train_grad_norm`, `train_params` | 20 optimizer steps through `rrc_train_run_on` (the path of `rrc_train_cuda`: data loader + prefetch thread, grad accumulation 2, clipping at 0.7 × the initial gradient norm so early steps clip, warmup 3 + cosine, weight decay 0.01, checkpoint): loss and grad norm per step from `metrics.jsonl`, final parameters from `checkpoint-20/adapter_model.safetensors` |
| `train_loss_decreases` | mean of the last three losses < first loss (the small data set is memorized; from 10 steps) |
| `det_forward`, `det_grad`, `det_train` | dev twice: logits, loss, gradients bit for bit; the two training runs' metrics bit for bit and byte-identical adapter and `rrc_state.bin` (AdamW m, v) |

Tolerance: with `err` the normalized maximum difference (logits, grads per tensor, per-step series:
max |d| / max |ref|; loss: relative; final parameters: max |d| / (lr · steps), i.e. in units of the
largest displacement Adam can produce), a check passes when
`err(dev, cpu-f32acc) ≤ max(K · err(cpu-f32acc, cpu), floor)` with K = 10: the anchor is the fp32
reference's own error against the double-accumulation backend, so a device whose fp32 arithmetic
is as accurate as the CPU's lands within about twice the anchor. The floors cover an anchor that is
accidentally tiny. In bf16 the anchored term of the logits and grad checks is capped:
`tol = max(min(K · err(cpu-f32acc, cpu), cap), floor)` (the report's `e2e.tolerances`):

| quantity | floor f32 | floor bf16 | cap bf16 |
| --- | --- | --- | --- |
| logits | 6e-5 | 3e-2 | 4.5e-2 |
| loss | 6e-6 | 1e-2 | none |
| grad (per tensor) | 3e-4 | 6e-2 | 8e-2 |
| train loss (per step) | 2e-5 | 2e-2 | none |
| train grad norm (per step) | 1e-4 | 6e-2 | none |
| train params (lr · steps units) | 8e-4 | 1e-1 | none |

Why the cap (review finding): at the quick and tiny sizes the bf16 anchor is small and the floor
decides (K · anchor / tol ≤ 0.1 for bf16 logits, loss and grad), but at the full GPU size the bf16
anchor is about 0.02 for the grads (llama, qwen2) and 0.04 for the final parameters, so the
uncapped rule gave effective bf16 tolerances of 0.2 (grad) and 0.41–0.45 (params): a 10% error in
the attention softmax backward passed every llama and qwen2 bf16 check (grad errors 0.147 /
0.145). Effective bf16 tolerances now, at the full size: grad 0.08 (llama, qwen2) and 0.06 (qwen3,
floor); logits 0.03–0.045; params unchanged, 0.19–0.45. The training quantities stay uncapped:
their anchors grow legitimately with few steps (params anchor 0.16–0.52 at `--e2e-steps 2`, where
the jitter device reaches 0.26–0.75), and the grad check is the one a backward fault shows in.
Calibration of the caps (host test devices on the CPU, no GPU; full size; the fault device's errors
were the same at 1, 2 and 4 OpenMP threads, the jitter device was measured at 4 threads, and at 2
threads with `--e2e-steps 2` in `make asan`): the jitter device's largest bf16 errors are 0.026
(grad) and 0.0093 (logits), so the caps are about 3× and 4.8× those and about 4× the anchors; the
fault device with `--fault-amp 0.1` (10%) fails the bf16 grad check of every family (0.147 / 0.145
/ 0.126) and `make test` requires exactly that (a control with `--e2e-size full --e2e-filter bf16
--e2e-steps 2`, next to a jitter run of the same set that must pass). A 5% fault is not reliably
caught in bf16 (qwen2 0.074 passes, llama 0.081 fails by 1%, qwen3 0.065 fails on the floor);
smaller bf16-only errors are left to the f32 end-to-end checks and the per-op bf16 checks. The
GPU's own bf16 error against `cpu-f32acc` is NOT_TESTED: a cap too tight for it shows as a bf16
logits / grad FAIL with `dev_err` just above the cap while everything else passes.

Calibration (TESTED on CPU, `make test`): two host test devices over `cpu-f32acc`. `jitter`
perturbs every GEMM output element by a deterministic pseudo-random factor up to ±k·2⁻²⁴ (the
fp32 error bound of a different summation order; deterministic so the det checks hold) and must
PASS; at the full GPU sizes its largest f32 errors were 1.9e-5 (logits), 1.1e-6 (loss), 7.7e-5
(grad), 1.6e-4 (params), each at most 1/3 of the floor. `fault` adds a relative error of 1e-3 to
`attn_softmax_bwd` (a subtly wrong backward) and must FAIL (`expect_fail` in the Makefile): its f32
grad errors are about 1.4e-3, ≈ 5× the floor (10 failing checks at the quick size, 12 at the full
size: grads 1.2e-3–1.4e-3, grad norms 2.7e-4, params 8.3e-4–1.1e-3 against 8e-4); forward checks
pass (the fault is backward-only), and in bf16 the fault is below the bf16 noise, so the f32 checks
are the ones that catch it (the bf16 checks catch a 10% fault of the same op, above). On kc-emu (CPU arithmetic, tree reductions, shuffled item order) the
whole set passes with dev_err ≈ ref_err. HOST-ONLY: the tiny qwen3 f32 + bf16 set passes through
the real backend on the fake runtime (`make test` with `CUDA_HOME`). TF32 (`--tf32`)
replaces the f32 floors by the bf16 ones / 8. The work directory (`--e2e-dir`, default `$TMPDIR`)
is removed unless `--e2e-keep`.

Run control: `--only e2e` runs only these checks; any other `--only` filter runs only matching op
checks (so the existing `--only convert` / `--only gemm` uses are unchanged); `--no-e2e`;
`--e2e-size`, `--e2e-steps`, `--e2e-filter family/dtype`. Every check is recorded; a failing stage
(an error from a run) becomes a failing item carrying the error message, and the other families
still run.

### 15.2 The report (`--report FILE`)

One JSON document (`schema: rrc-selftest-report-v1`): `result` (PASS / FAIL / INCOMPLETE),
`complete`, `fatal_error`, `started_utc`, `command_line`, `build` (git revision via
`git describe --dirty` at compile time, host compiler, nvcc version, CUDA runtime and cuBLAS
header versions, `__CUDA_ARCH_LIST__`, host-shim flag), `device` (name, compute capability, SMs,
memory, driver / runtime / cuBLASLt versions, and the binary and PTX versions of the kernel image
the runtime actually loaded, from `cudaFuncGetAttributes`, which shows whether sm_120 SASS or
JIT-compiled PTX runs), `options`, `op_passes[]` (main, capped, grid: summary counts, per-op cases /
max abs / max rel / worst error over tolerance, GEMM path per dtype combination, cases per layout
and ladder level, kernel-fallback examples, failing cases), and `e2e` (every item with `dev_err`,
`ref_err`, `tol`, `pass`, `detail`; in the CUDA build also `model_gemm_tally`: the device's GEMM
calls and computed FLOPs per dtype combination and ladder level during the end-to-end runs, with
the kernel-fallback FLOP share printed loudly, the h2d and stream-synchronize counts, the staging
ring that ran (`staging_bytes`, `staging_slots`, `h2d_direct`), and `model_gemm_census`: every
distinct GEMM geometry of the model with its path), and in the CUDA build `h2d_staging` (the
requested ring size and the one the backend actually set up, `rrc_cuda_stats.staging_bytes`: 0 if
disabled or if the pinned allocation failed; the `options:` line prints the same) and
`h2d_ring_pass` (the small-ring pass: a 4 KiB ring in 4 slots, odd-sized uploads and one of about
1 MiB in 1025 chunks from one host array reused at once, downloaded and compared bit for bit,
with the chunk / slot-wait counters checked; the multi-chunk path of `memcpy_h2d` that a real
weight load takes and that the default 32 MiB ring never splits at the selftest's sizes;
`--only h2d_ring` runs it alone; HOST-ONLY on the fake runtime in `make test`, NOT_TESTED on a
GPU). The file is
rewritten atomically after every stage and after every end-to-end family × dtype case (with
`complete: false`, and `e2e.result` INCOMPLETE while that stage runs), and on the fatal path (e.g.
the backend cannot be created), so a crash or hang leaves the finished stages on disk; stdout is
line-buffered, so a log piped through `tee` keeps every finished line too. The HOST-ONLY run
checks the CUDA-only sections (build, device, tally, census; `tests/cuda_host/check_reports.py`). The last stdout
line stays exactly `rrc_selftest: PASS|FAIL` (the Makefile's checks depend on it).

### 15.3 Pinned staging for memcpy_h2d (no signature change)

`memcpy_h2d` was `cudaMemcpyAsync` from pageable memory + a full stream synchronize per call,
i.e. one host-device round trip per upload (every micro-batch uploads its token, label, position
and index arrays). Now the CUDA backend owns a pinned ring (`rrc_cuda_options.h2d_staging_bytes`,
default 32 MiB in 4 slots, env `RRC_CUDA_H2D_STAGING_BYTES`; 0 disables). Per slot-sized chunk:
if the slot is busy, `cudaEventSynchronize(slot.event)` (the copy that last read it has completed);
`memcpy` of the host source into the slot; `cudaMemcpyAsync(slot → device)` on the stream;
`cudaEventRecord(slot.event)`. The call returns without a stream synchronize: the host source has
already been copied (contract kept: it may be reused on return), and the device copy is ordered
with every other op by the stream. Large copies (model weights) stream through the ring in chunks,
the host memcpy of one slot overlapping the DMA of the previous ones. With `sync_each_op` the call
also synchronizes. A failed pinned allocation at creation disables the ring with a warning;
`destroy` frees the ring and events after its stream synchronize. Counted on the fake runtime
(code-derived counts, not timings) for one configuration, `rrc_bench --synthetic-model qwen3`
with 1 row × 8 tokens and grad accumulation 2: one optimizer step issues 12 uploads and 6 full
stream synchronizes; before the ring each upload added one synchronize. The ring is on by default
for every user of the backend, `rrc_train_cuda` included.

Tests (HOST-ONLY, `cuda_host.h2d_staging_ring`): the fake runtime's host → device copies are lazy
(queued; the source is read only when the queue is flushed by a later stream operation, a stream
or event synchronize, or a free), so a slot overwritten before its copy completed, or a copy still
reading the caller's buffer, delivers wrong bytes. The test uploads ten buffers (1 … 1000 floats)
through a 4 × 256-byte ring, overwriting the single host source right after each call, and checks:
the data, that copies were still pending (not vacuous), zero stream synchronizes during the
uploads, slot event waits = chunks − slots, zero async copies from pageable memory, stream order
with ops in between (y = 2 x₁ + 2 x₂ re-uploading x), the disabled ring, a pinned allocation
failure, `sync_each_op`, and no pending copy or pinned block after destroy. Mutations checked by
hand: removing the event wait (1812 wrong elements, plus failures of the ordering and
sync-each-op cases) and copying from the caller's buffer instead of the slot (ordering case fails)
are both caught. The whole existing HOST-ONLY suite (harness, e2e) also runs on the lazy copies.

`backend.h` gained one comment-only clarification (no signature change): on device backends, the
outputs of the ops enqueued after an op that hit a data-dependent range error are unspecified
until the next sync reports the error.

Other additions to the CUDA line's own API (`rrc_cuda.h`): `rrc_cuda_stats` (live / peak bytes of
`be->alloc` allocations, backend-internal bytes, h2d / staging / stream-sync counters, GEMM calls
and computed FLOPs per dtype combination × ladder level), the GEMM census, a cudaEvent timer,
`rrc_cuda_build_info_get` / `rrc_cuda_build_is_host_shim`, `rrc_cuda_kernel_versions`.

### 15.4 rrc_bench and the PyTorch baseline

`rrc_bench --model DIR` (`bench/rrc_bench.c`; built by `make cuda`) loads the model (bf16 base),
creates LoRA (default r 16, alpha 32, all seven projections, fp32, PEFT init) and M synthetic
micro-batches of B × T random token ids (one document per row, every position labelled), then runs
W warmup + N measured optimizer steps (M × `rrc_model_forward_backward` + clip + AdamW, constant
LR: the trainer's step without the loader). Step time = cudaEvent time between marks recorded on
the backend stream before and after the step; tokens/s = B·T·M / mean step time; mean / p50 / p90
(nearest rank) / min / max; host wall time per step separately. Memory, each labelled in the
file: `allocator_peak_bytes` (true peak of `be->alloc` bytes over the measured steps),
`backend_total_peak_bytes` (plus workspace / scratch / partials), `device_used_max_sampled_bytes`
(`cudaMemGetInfo` total − free after each step: device-wide, not a peak). The GEMM tally and census
of the first measured step show every GEMM shape, its path and computed FLOPs. `"measured_on"`
names the device only when the build is the real CUDA build (`rrc_cuda_build_is_host_shim() == 0`)
and the device was queried; `rrc_bench_host` (CPU backend, `make cpu`) and `rrc_bench_shim`
(HOST-ONLY) emit the same keys with `"status": "smoke"`, null timings and `"measured_on": null`.
`--synthetic-model FAMILY` replaces `--model` with a tiny model written in C (smoke runs only).

`bench/torch_baseline.py`: same keys, same workload — bf16 base, `attn_implementation="sdpa"`,
PEFT LoRA with the same r / alpha / targets, adapters checked to be fp32, same B, T, M, synthetic
tokens with `labels = input_ids`, gradient checkpointing `on|off` (required argument; mapped to
`recompute` full / none in the output), `torch.optim.AdamW` with the same hyper-parameters (fused on
CUDA by default, stated in `optimizer.impl`), `clip_grad_norm_`, constant LR, `torch.cuda.Event`
timing of the same span, `max_memory_allocated` / `max_memory_reserved` and the sampled device
use. `local_files_only=True`, `HF_HUB_OFFLINE=1`, never `trust_remote_code`. Smoke tests
(`tests/test_gpu_tools_smoke.py`, CPU): argument errors of both tools, the JSON shape and the
"no timing claim" fields, a non-zero LoRA gradient norm with checkpointing on and off, identical key
sets of the two tools, and the same tokens per step and trainable parameter count for the same
configuration (the LoRA layouts agree). Known workload differences (by design): rrc synchronizes
once per micro-batch (loss) and once per step (gradient norm for clipping), its attention is
chunked GEMM + softmax (no fused flash kernel), and its AdamW is the `foreach=False` arithmetic;
rrc computes the LM head and cross-entropy in `ce_chunk`-row chunks, while transformers
materialises the full [B, T, V] logits (bf16, plus an f32 upcast in its loss), so the peak-memory
figures are not like-for-like; rrc uploads each micro-batch's token ids / labels / positions /
document ids inside the timed step, while the baseline's batches are on the device before timing.

### 15.5 The run, and what each outcome tells

The exact commands are in `bench/README.md` (environment record, CPU build + tests, `make cuda`,
`rrc_selftest --report selftest.json`, four `rrc_bench` and four baseline configurations, one
tarball back). Reading the outcome:

1. Backend creation fails (`fatal_error` in the report): driver / runtime / device problem, no rrc
   op ran.
2. Op checks fail: the kernel or the GEMM integration on real hardware (`op_passes[].ops`,
   `failing_cases`; a FAIL PATH means cuBLASLt refused a case the plan expects it to serve: the
   type table, alignment or preference attributes). `--sync-each-op` localizes asynchronous
   errors to the op.
3. Op checks pass, end-to-end fails: the model / trainer composition on the device (e.g. an
   aliasing or ordering assumption the CPU never exercised, or the staging ring); the failing item
   names family, dtype and quantity, `detail` the worst tensor or step.
4. Determinism fails: an algorithm choice or a reduction that is not run-to-run stable on the device.
5. Everything passes: `e2e.model_gemm_tally.kernel_fallback_flop_share` and the census say whether
   the model's GEMMs reach cuBLASLt; then the benchmark files give the first measured tokens/s,
   step times and memory for rrc and the baseline under the same configuration.

### 15.6 Status

| item | status |
| --- | --- |
| e2e checks, report JSON, jitter / fault controls (`rrc_selftest_host`) | TESTED (CPU) |
| e2e checks + report through the real `cuda/*.cu` (`rrc_selftest_shim`, tiny qwen3) | HOST-ONLY |
| staging ring, counters, census, timer API (`test_cuda_host`) | HOST-ONLY |
| `rrc_bench` CUDA main (`rrc_bench_shim`), `rrc_bench_host`, baseline `--device cpu` | HOST-ONLY / TESTED (CPU) smoke |
| `rrc_selftest`, `rrc_bench` on a GPU; cuBLASLt behaviour; the ring on a real DMA engine | COMPILED-ONLY |
| any tokens/s, step time, memory figure; any comparison with PyTorch | NOT_TESTED |

### 15.7 Not done

* Nothing ran on a GPU. Whether the tolerances fit real cuBLASLt / tensor-core arithmetic is
  argued from the jitter calibration, not observed; a FAIL comes with every number in the report.
* `rrc_train_cuda` is exercised by the e2e training runs, not separately on a real model; the
  e2e models are tiny (no 7–8B checkpoint was ever loaded).
* No overlap of uploads with compute beyond what one stream gives (the ring removes the
  synchronize, the copies stay on the compute stream); no CUDA graphs, no fused attention.
* The baseline's `torch.compile` option (§7 item 4) is not implemented.
