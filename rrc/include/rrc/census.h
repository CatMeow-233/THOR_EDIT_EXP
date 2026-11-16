/* rrc/census.h — GEMM census and memory pre-flight of one training step, computed by a dry run
 * (rrc/dryrun.h) of the real model / LoRA / optimizer code on a shape-only model
 * (rrc/docs/DESIGN.md §13).
 *
 * rrc_census_run builds the shape-only model, the LoRA adapter and the AdamW state on a dry-run
 * backend with the real constructors, then runs two optimizer steps exactly as the trainer does
 * (zero grads, grad_accum micro-batches of forward + backward, clip + AdamW, sync) on packed rows
 * of one full-length document each, every token labelled (the worst case for attention and the LM
 * head). The counters of the SECOND step are reported (steady state: the first step additionally
 * allocates the workspace and grows the GEMM scratch). Every value is COMPUTED, never measured.
 *
 * GEMM roles are decided by operand addresses, not by dtypes: B inside the LM-head weight ->
 * lm_head_fwd (B transposed) / lm_head_dx; B inside the other base weights -> base_fwd / base_dx;
 * any operand inside the LoRA parameters or gradients -> lora; everything else -> attention (the
 * only remaining GEMMs of src/model_exec.c are the attention ones).
 *
 * Memory (bytes, computed): op-table allocations (weights, RoPE tables, LoRA params + grads, AdamW
 * m / v, the per-step workspace; their peak is what the dry-run backend tracked) plus, for the CUDA
 * target, the backend-internal buffers the op table does not see: the GEMM promotion scratch (max
 * over the step's calls of the plan's scratch), the sum_squares partial buffer and the cuBLASLt
 * workspace. Not included: the CUDA context, allocator rounding / fragmentation, host memory.
 */
#ifndef RRC_CENSUS_H
#define RRC_CENSUS_H

#include <stdint.h>

#include "rrc/common.h"
#include "rrc/config.h"
#include "rrc/dryrun.h"
#include "rrc/lora.h"
#include "rrc/model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum rrc_census_target {
    RRC_CENSUS_TARGET_CUDA = 0, /* add the CUDA backend's internal buffers (rrc/cuda/rrc_cuda.h) */
    RRC_CENSUS_TARGET_CPU = 1   /* op-table allocations only */
} rrc_census_target;

typedef enum rrc_census_role {
    RRC_CR_BASE_FWD = 0,   /* x W^T against a frozen q/k/v/o/gate_up/down weight (forward and recompute) */
    RRC_CR_BASE_DX = 1,    /* dy W: the input gradient through a frozen weight */
    RRC_CR_LM_HEAD_FWD = 2,
    RRC_CR_LM_HEAD_DX = 3,
    RRC_CR_LORA = 4,       /* every GEMM touching the LoRA parameters or gradients (fp32, PEFT) */
    RRC_CR_ATTENTION = 5,  /* scores, P V, dP, dV, dQ, dK */
    RRC_CR_N = 6
} rrc_census_role;

/* "base_fwd", "base_dx", "lm_head_fwd", "lm_head_dx", "lora", "attention" (NULL if invalid). */
const char *rrc_census_role_name(int role);
/* A frozen base weight is operand B (base_fwd, base_dx, lm_head_fwd, lm_head_dx). */
int rrc_census_role_is_base(int role);

/* dtype combination index: 4 * a + 2 * b + c with f32 = 0, bf16 = 1. */
#define RRC_CENSUS_N_COMBOS 8
int rrc_census_combo(rrc_dtype a, rrc_dtype b, rrc_dtype c);
rrc_dtype rrc_census_combo_dtype(int combo, int operand /* 0 a, 1 b, 2 c */);
/* The level-0 plan path of a combination: "native" or e.g. "promote B (bf16->f32)", "C via f32 scratch". */
const char *rrc_census_combo_path(int combo);
int rrc_census_combo_native(int combo);
int rrc_census_combo_native_bf16(int combo);

typedef struct rrc_census_opts {
    rrc_model_opts model;     /* dtype, attn_chunk, ce_chunk, recompute, max_positions */
    rrc_lora_config lora;
    int64_t seq_len;          /* tokens per packed row (required, <= the model's positions) */
    int64_t micro_batch;      /* rows per micro-batch (default 1) */
    int64_t grad_accum;       /* micro-batches per optimizer step (default 1) */
    rrc_census_target target; /* default CUDA */
    int64_t gemm_scratch_cap_elems;    /* <= 0: RRC_GEMM_SCRATCH_CAP_DEFAULT (64 MiB) */
    int allow_tf32;
    uint64_t cublaslt_workspace_bytes; /* CUDA target: default 32 MiB (rrc_cuda_options default) */
} rrc_census_opts;

/* model opts default (bf16, attn_chunk 256, ce_chunk 1024, recompute full), LoRA default (r 8,
 * alpha 16, all seven modules), micro_batch 1, grad_accum 1, target CUDA, seq_len 0 (must be set). */
void rrc_census_opts_default(rrc_census_opts *o);

typedef struct rrc_census_cell {
    int64_t calls;
    double flops;
    uint64_t convert_elems, convert_bytes;
    int64_t align_lt16_calls; /* calls with an operand (any chunk) aligned to < 16 bytes */
} rrc_census_cell;

typedef struct rrc_census_result {
    rrc_model_config cfg;
    int64_t base_params;   /* elements of the base weight allocation (norms and biases included) */
    int64_t lora_params;
    int64_t tokens_per_step, labelled_per_step;
    rrc_census_cell cell[RRC_CR_N][RRC_CENSUS_N_COMBOS]; /* the reported (second) step */
    rrc_dryrun_stats step;                               /* dry-run counters of that step */
    /* memory (bytes) */
    uint64_t weights_bytes, rope_bytes, lora_bytes, adam_bytes, workspace_bytes;
    uint64_t op_peak_bytes;                     /* peak of the op-table allocations (dry-run backend) */
    uint64_t gemm_scratch_bytes, sum_squares_partials_bytes, cublaslt_workspace_bytes; /* CUDA target */
    uint64_t peak_total_bytes;                  /* op_peak + the CUDA-internal buffers */
} rrc_census_result;

rrc_status rrc_census_run(const rrc_model_config *cfg, const rrc_census_opts *o, rrc_census_result *r);

/* JSON report. `source` describes where the config came from (a path or "preset:<name>"),
 * `preset_note` (may be NULL) is printed with it. malloc'd. */
rrc_status rrc_census_to_json(const rrc_census_result *r, const rrc_census_opts *o, const char *source,
                              const char *preset_note, int pretty, char **out);

/* ---- built-in shape presets ----------------------------------------------------------------------
 * config.json text with the values of the public model cards of three 7-8B models, typed in from
 * the published files as recalled: ASSUMED shapes, NOT_TESTED against the real files (huggingface.co
 * is not reachable from the development environment). They go through the same strict parser as a
 * real config.json. Use a real config.json (rrc_census --config) for anything that matters. */
typedef struct rrc_census_preset {
    const char *name;        /* "qwen3-8b", "qwen2.5-7b", "llama-3.1-8b" */
    const char *label;       /* human-readable model name */
    const char *config_json; /* config.json text */
} rrc_census_preset;

size_t rrc_census_num_presets(void);
const rrc_census_preset *rrc_census_preset_at(size_t i);
const rrc_census_preset *rrc_census_preset_find(const char *name); /* NULL if unknown */
/* The note attached to every preset in the JSON output. */
const char *rrc_census_preset_note(void);
/* Parse a preset's config.json text with rrc_config_from_json (flags 0). */
rrc_status rrc_census_preset_config(const rrc_census_preset *p, rrc_model_config *out);

#ifdef __cplusplus
}
#endif

#endif /* RRC_CENSUS_H */
