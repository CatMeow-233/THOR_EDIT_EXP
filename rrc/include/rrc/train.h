/* rrc/train.h — SFT training driver: LR schedules, gradient clipping + AdamW, training config,
 * checkpoints and exact resume (rrc/docs/DESIGN.md §3, §6, §10; config fields: docs/README.md).
 *
 * Optimizer step (one window of grad_accum_steps micro-batches):
 *   grads = 0; for each micro-batch: rrc_model_forward_backward (accumulates, scaled so that the
 *   gradient is that of the window loss);
 *   total_norm = sqrtf(sum_squares(all LoRA grads))                 (f32, one op call)
 *   clip_coef  = min(1, (1 / (total_norm + 1e-6f)) * (float)max_norm)   (torch clip_grad_norm_:
 *                `max_norm / (total_norm + 1e-6)` on an f32 tensor is reciprocal() * max_norm;
 *                max_norm <= 0 disables clipping, clip_coef = 1)
 *   rrc_adamw_step(params, grads, m, v, {lr, betas, eps, wd, step, grad_scale = clip_coef})
 * lr of optimizer step s (1-based) = learning_rate * lambda(s - 1), lambda from
 * transformers.get_scheduler (LambdaLR evaluates lambda(last_epoch) before each step):
 *   warmup (k < W):          k / max(1, W)
 *   constant:                1                       (no warmup at all)
 *   constant_with_warmup:    1
 *   linear:                  max(0, (N - k) / max(1, N - W))
 *   cosine:                  max(0, 0.5 * (1 + cos(pi * 0.5 * 2 * (k - W) / max(1, N - W))))
 * with N = max_steps and W = warmup_steps > 0 ? warmup_steps : ceil(N * warmup_ratio)
 * (TrainingArguments.get_warmup_steps). All in double, as Python does.
 */
#ifndef RRC_TRAIN_H
#define RRC_TRAIN_H

#include <stdint.h>

#include "rrc/backend.h"
#include "rrc/common.h"
#include "rrc/data.h"
#include "rrc/lora.h"
#include "rrc/model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- LR schedule ----------------------------------------------------------------------------- */

typedef enum rrc_lr_schedule {
    RRC_LR_CONSTANT = 0,
    RRC_LR_CONSTANT_WITH_WARMUP = 1,
    RRC_LR_LINEAR = 2,
    RRC_LR_COSINE = 3
} rrc_lr_schedule;

const char *rrc_lr_schedule_name(rrc_lr_schedule s); /* "constant", ... (NULL if invalid) */
rrc_status rrc_lr_schedule_parse(const char *name, rrc_lr_schedule *out);
/* TrainingArguments.get_warmup_steps: warmup_steps if > 0, else ceil(total * warmup_ratio). */
int64_t rrc_warmup_steps(int64_t warmup_steps, double warmup_ratio, int64_t total);
/* The multiplier lambda(k) for scheduler step k = 0, 1, ... (see the header comment). */
double rrc_lr_lambda(rrc_lr_schedule s, int64_t k, int64_t warmup, int64_t total);

/* torch.nn.utils.clip_grad_norm_ coefficient in f32 (see the header comment). */
float rrc_clip_coef(float total_norm, double max_norm);

/* ---- optimizer -------------------------------------------------------------------------------- */

typedef struct rrc_optim_hparams {
    double beta1, beta2, eps, weight_decay;
    double max_grad_norm; /* <= 0: no clipping */
} rrc_optim_hparams;

typedef struct rrc_optim_stats {
    float grad_norm; /* total L2 norm of the gradients before clipping */
    float clip_coef; /* the factor applied (1 when not clipped) */
} rrc_optim_stats;

typedef struct rrc_optimizer rrc_optimizer;

/* fp32 AdamW state (m, v zero) for n parameters on `be`. */
rrc_status rrc_optimizer_create(const rrc_backend *be, int64_t n, rrc_optimizer **out);
void rrc_optimizer_free(rrc_optimizer *o);
int64_t rrc_optimizer_step_count(const rrc_optimizer *o);
rrc_buf rrc_optimizer_m(const rrc_optimizer *o);
rrc_buf rrc_optimizer_v(const rrc_optimizer *o);
/* Restore m, v (host arrays of n floats) and the step count. */
rrc_status rrc_optimizer_restore(rrc_optimizer *o, const float *m, const float *v, int64_t step);
/* Clip (fused) + AdamW update of params from grads (grads are not modified). Non-finite
 * grad_norm is an error (RRC_ERR_RANGE) and nothing is updated. */
rrc_status rrc_optimizer_step(rrc_optimizer *o, rrc_buf params, rrc_buf grads, double lr, const rrc_optim_hparams *h,
                              rrc_optim_stats *stats);

/* ---- training config (JSON; unknown keys are errors; docs/README.md documents each field) ---- */

#define RRC_PATH_MAX 4096

typedef struct rrc_train_config {
    char model_dir[RRC_PATH_MAX];
    char data_path[RRC_PATH_MAX];
    char output_dir[RRC_PATH_MAX];
    char init_adapter[RRC_PATH_MAX]; /* "" = rrc's seeded init */
    char resume_from[RRC_PATH_MAX];  /* "" = fresh run */
    int overwrite_output_dir;
    rrc_dtype dtype;
    int backend_f32acc; /* 0: "cpu", 1: "cpu-f32acc" */
    int64_t seq_len, micro_batch_size, grad_accum_steps, max_steps;
    double learning_rate;
    rrc_lr_schedule lr_scheduler_type;
    int64_t warmup_steps;
    double warmup_ratio;
    rrc_optim_hparams optim;
    rrc_loss_reduction loss_reduction;
    uint64_t seed;
    int shuffle;
    rrc_truncation truncation;
    int32_t pad_token_id;
    rrc_lora_config lora;
    int64_t attn_chunk, ce_chunk, max_positions;
    rrc_recompute recompute;
    int prefetch;
    int64_t save_steps;      /* 0: only the final checkpoint */
    int64_t stop_after_step; /* 0: run to max_steps; else checkpoint and stop after this step */
    int dump_batches;        /* write output_dir/batches/step-N.safetensors (tests, debugging) */
} rrc_train_config;

void rrc_train_config_default(rrc_train_config *c);
/* Parse a JSON object onto the defaults. Unknown keys, wrong types, out-of-range values and
 * missing required keys (model_dir, data_path, output_dir, seq_len, max_steps) are errors. */
rrc_status rrc_train_config_parse(const char *json_text, size_t len, rrc_train_config *out);
rrc_status rrc_train_config_load(const char *path, rrc_train_config *out);
/* The resolved number of warmup steps of this config. */
int64_t rrc_train_config_warmup(const rrc_train_config *c);
/* The config as JSON (every field; malloc'd string). */
rrc_status rrc_train_config_to_json(const rrc_train_config *c, char **out);
/* Canonical "key=value\n" text of the fields that determine the training trajectory (stored in
 * rrc_state.bin; resume refuses a config whose text differs). malloc'd. */
rrc_status rrc_train_config_fingerprint(const rrc_train_config *c, char **out);

/* ---- rrc_state.bin (exact resume) -------------------------------------------------------------
 *
 * Little-endian, written atomically (tmp + rename):
 *   0    8  magic "RRCSTAT1"
 *   8    8  u64 version (1)
 *   16   8  u64 step               optimizer steps completed (= AdamW step count)
 *   24   8  u64 n_params           LoRA parameter count
 *   32   8  u64 data_seed          shuffle seed (the only random generator of the run: epoch e's order is
 *                                  rrc_data_permutation(data_seed, e), so (seed, epoch) is its full state)
 *   40   8  i64 data_epoch         data cursor: epoch ...
 *   48   8  i64 data_index         ... and index of the next document in that epoch's order
 *   56   8  i64 micro_batches      micro-batches consumed so far
 *   64   8  i64 tokens_seen        labelled tokens trained on so far
 *   72   8  u64 params_hash        FNV-1a 64 of the fp32 adapter parameters (bytes, rrc layout)
 *   80   8  u64 data_hash          FNV-1a 64 of the .rrcbin file
 *   88   8  u64 cfg_len            length of the config fingerprint text
 *   96   cfg_len bytes            rrc_train_config_fingerprint text
 *   ...  4 * n_params             AdamW m (f32)
 *   ...  4 * n_params             AdamW v (f32)
 *   ...  8  u64 FNV-1a 64 of all preceding bytes
 */
typedef struct rrc_train_state {
    int64_t step, n_params;
    uint64_t data_seed;
    rrc_data_cursor cursor;
    int64_t tokens_seen;
    uint64_t params_hash, data_hash;
    char *config_text; /* NUL-terminated, malloc'd */
    float *m, *v;      /* n_params each, malloc'd */
} rrc_train_state;

rrc_status rrc_train_state_write(const char *path, const rrc_train_state *s);
rrc_status rrc_train_state_read(const char *path, rrc_train_state *out);
void rrc_train_state_free(rrc_train_state *s);

/* FNV-1a 64 */
uint64_t rrc_fnv1a64(const void *data, size_t n);

/* ---- adapter_config.json ------------------------------------------------------------------------ */

/* PEFT LoraConfig JSON for `lc` (peft_type LORA, task_type CAUSAL_LM, r, lora_alpha, lora_dropout,
 * target_modules, use_rslora, layers_to_transform, bias "none", base_model_name_or_path, ...).
 * malloc'd. */
rrc_status rrc_peft_adapter_config_json(const rrc_lora_config *lc, const char *base_model, char **out);

/* ---- trainer ---------------------------------------------------------------------------------------
 *
 * Runs the config: loads model + data, creates / loads the adapter, trains to max_steps (or
 * stop_after_step), appends one line per optimizer step to output_dir/metrics.jsonl and writes
 * checkpoints output_dir/checkpoint-<step>/{adapter_model.safetensors, adapter_config.json,
 * rrc_state.bin} (every save_steps, at stop_after_step and at max_steps). */
rrc_status rrc_train_run(const rrc_train_config *c);

#ifdef __cplusplus
}
#endif

#endif /* RRC_TRAIN_H */
