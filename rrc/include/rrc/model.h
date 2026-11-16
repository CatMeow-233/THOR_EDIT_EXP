/* rrc/model.h — frozen decoder weights (llama / qwen2 / qwen3) and the LoRA forward / backward
 * (rrc/docs/DESIGN.md §3, §5, §10).
 *
 * All computation goes through the op table (rrc/backend.h). Base weights are frozen: no weight
 * gradient is ever computed for them; the only gradients are the fp32 LoRA gradients, which
 * rrc_model_forward_backward ACCUMULATES (+=) into rrc_lora_grads().
 *
 * Micro-batch (packed rows): rows x seq_len tokens. Several documents may share a row; each
 * document restarts position_ids at 0 and attends only within itself (causal + doc_id mask).
 * Labels are HF-style (unshifted, RRC_IGNORE_INDEX = -100 for "no loss"): the engine shifts them,
 * logits[t] predicts labels[t + 1], and never across a document or row boundary — the target of
 * the last token of a document is ignored. Padding is simply a document (any doc id) whose labels
 * are all -100.
 */
#ifndef RRC_MODEL_H
#define RRC_MODEL_H

#include <stdint.h>

#include "rrc/backend.h"
#include "rrc/common.h"
#include "rrc/config.h"
#include "rrc/lora.h"
#include "rrc/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum rrc_recompute {
    RRC_RECOMPUTE_FULL = 0, /* save only each layer's input; recompute the layer in backward */
    RRC_RECOMPUTE_NONE = 1  /* keep every layer's activations (more memory, no recompute) */
} rrc_recompute;

typedef struct rrc_model_opts {
    rrc_dtype dtype;          /* base weights and activations: RRC_BF16 (default) or RRC_F32 */
    int64_t max_positions;    /* RoPE table rows; 0 -> config max_position_embeddings */
    int64_t attn_chunk;       /* query rows per attention chunk (default 256) */
    int64_t ce_chunk;         /* labelled rows per LM-head / cross-entropy chunk (default 1024) */
    rrc_recompute recompute;  /* default RRC_RECOMPUTE_FULL */
    unsigned config_flags;    /* rrc_config_load flags (RRC_CONFIG_ALLOW_UNKNOWN_KEYS) */
} rrc_model_opts;

void rrc_model_opts_default(rrc_model_opts *o);

typedef struct rrc_model rrc_model;

/* Load config.json + safetensors (single file or sharded index) from `dir`. The checkpoint's
 * tensor set must match the config exactly (rrc_config_check_tensors); F32/BF16 tensors are
 * converted on the host to opts->dtype (f32 -> bf16 with RNE) and uploaded to `be`. */
rrc_status rrc_model_load(const rrc_backend *be, const char *dir, const rrc_model_opts *opts, rrc_model **out);
void rrc_model_free(rrc_model *m);
/* Shape-only model for dry runs and memory planning (DESIGN.md §13): the same weight layout,
 * views and RoPE table allocation as rrc_model_load for `cfg` (a config from rrc_config_load /
 * rrc_config_from_json), allocated on `be` but never filled: no checkpoint is read, the weights and
 * the RoPE tables keep the zeros of rrc_buf_alloc. Intended for the dry-run backend (rrc/dryrun.h);
 * on a real backend it gives a model of zero weights. */
rrc_status rrc_model_create_shape_only(const rrc_backend *be, const rrc_model_config *cfg, const rrc_model_opts *opts,
                                       rrc_model **out);

const rrc_model_config *rrc_model_get_config(const rrc_model *m);
const rrc_backend *rrc_model_backend(const rrc_model *m);
const rrc_model_opts *rrc_model_get_opts(const rrc_model *m);
/* Change the chunk sizes / recompute policy of a loaded model (values as in rrc_model_opts; the
 * workspace is re-planned on the next call). */
rrc_status rrc_model_set_exec(rrc_model *m, int64_t attn_chunk, int64_t ce_chunk, rrc_recompute recompute);
/* Bytes of the current per-step workspace (0 before the first step). */
size_t rrc_model_workspace_bytes(const rrc_model *m);
/* Number of workspace (re)allocations so far (the workspace grows only when a step needs more). */
int64_t rrc_model_workspace_allocs(const rrc_model *m);

/* One packed micro-batch (host arrays, rows * seq_len elements each, row-major).
 *   tokens:       required, each in [0, vocab).
 *   labels:       HF-style unshifted labels, each in [0, vocab) or -100; NULL = all -100.
 *   doc_ids:      document id per token; NULL = derived from position_ids (a document starts at
 *                 t = 0 and wherever position_ids[t] == 0), or one document per row if both NULL.
 *   position_ids: NULL = derived from doc_ids (0, 1, ... within each document); each in
 *                 [0, max_positions). */
typedef struct rrc_batch {
    int64_t rows, seq_len;
    const int32_t *tokens;
    const int32_t *labels;
    const int32_t *position_ids;
    const int32_t *doc_ids;
} rrc_batch;

/* Number of labelled positions after the shift (the cross-entropy rows of this micro-batch). */
rrc_status rrc_batch_count_labels(const rrc_batch *b, int64_t *n_out);

typedef enum rrc_loss_reduction {
    /* loss = sum of per-token CE over all micro-batches of the window / window_tokens (the total
     * labelled count of the window, rrc_batch_count_labels summed by the caller). */
    RRC_LOSS_WINDOW_TOKENS = 0,
    /* loss = (1 / num_micro_batches) * sum over micro-batches of (that micro-batch's mean CE). */
    RRC_LOSS_MICROBATCH_MEAN = 1
} rrc_loss_reduction;

typedef struct rrc_fb_opts {
    rrc_loss_reduction reduction;
    int64_t window_tokens;     /* RRC_LOSS_WINDOW_TOKENS: labelled tokens in the whole window */
    int64_t num_micro_batches; /* RRC_LOSS_MICROBATCH_MEAN: micro-batches in the window (>= 1) */
    int compute_grad;          /* 0: loss only (no backward, grads untouched) */
} rrc_fb_opts;

/* Zero labelled tokens (DESIGN.md §10): a micro-batch with no labelled position contributes
 * loss 0 and gradient 0 under both reductions (no division by zero, no NaN); under
 * RRC_LOSS_MICROBATCH_MEAN it still counts in num_micro_batches. A window with window_tokens == 0
 * likewise gives loss 0 and zero gradients. */
typedef struct rrc_fb_result {
    double loss_sum;   /* sum of per-token cross entropy over this micro-batch's labelled tokens */
    int64_t n_tokens;  /* labelled tokens in this micro-batch */
    double loss;       /* this micro-batch's contribution to the window loss (loss_sum * scale) */
    float grad_scale;  /* the factor applied to d(loss_sum) (1/window_tokens, or 1/(n*M), or 0) */
} rrc_fb_result;

/* Forward + backward of one micro-batch with LoRA adapter `lora` (NULL = base model, loss only).
 * Gradients are accumulated into rrc_lora_grads(lora). */
rrc_status rrc_model_forward_backward(rrc_model *m, rrc_lora *lora, const rrc_batch *b, const rrc_fb_opts *o,
                                      rrc_fb_result *res);

/* Forward only: logits of every token as f32 into host memory [rows * seq_len, vocab] (chunked
 * LM head). lora may be NULL. If hidden_out is non-NULL it receives the hidden states as f32,
 * [(num_layers + 1), rows * seq_len, hidden]: the embedding output, each layer's output, with the
 * last entry replaced by the final-norm output (the convention of HF output_hidden_states). */
rrc_status rrc_model_forward_logits(rrc_model *m, const rrc_lora *lora, const rrc_batch *b, float *logits_out,
                                    float *hidden_out);

#ifdef __cplusplus
}
#endif

#endif /* RRC_MODEL_H */
