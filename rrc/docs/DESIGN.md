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
  `rope_theta` (+ llama3 scaling) and `position_ids`.
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
