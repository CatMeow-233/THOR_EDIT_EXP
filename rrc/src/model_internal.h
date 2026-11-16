/* rrc/src/model_internal.h — private structures shared by model.c, lora.c and model_exec.c. */
#ifndef RRC_MODEL_INTERNAL_H
#define RRC_MODEL_INTERNAL_H

#include "rrc/model.h"

typedef struct rrc_layer_weights {
    rrc_buf ln1, ln2;          /* [H] */
    rrc_buf wq, wk, wv, wo;    /* [Q, H], [KV, H], [KV, H], [H, Q] */
    rrc_buf bq, bk, bv, bo;    /* optional biases (rrc_buf_none() when absent) */
    rrc_buf qn, kn;            /* qwen3 q_norm / k_norm [head_dim] (none when absent) */
    rrc_buf wgu;               /* [2I, H]: gate_proj rows then up_proj rows (one GEMM) */
    rrc_buf wd;                /* [H, I] */
} rrc_layer_weights;

/* Activations of one layer that its backward reads. With RRC_RECOMPUTE_FULL one scratch instance
 * is shared by all layers (the forward is recomputed into it); with RRC_RECOMPUTE_NONE there is
 * one persistent instance per layer. */
typedef struct rrc_layer_cache {
    rrc_buf xn1;            /* rmsnorm(x) [N, H] */
    rrc_buf q_raw, k_raw;   /* projection outputs before q/k norm (qk_norm only) [N, Q], [N, KV] */
    rrc_buf q, k, v;        /* attention inputs after norm + rope [N, Q], [N, KV], [N, KV] */
    rrc_buf o;              /* attention output (o_proj input) [N, Q] */
    rrc_buf h;              /* x + attn [N, H] */
    rrc_buf xn2;            /* rmsnorm(h) [N, H] */
    rrc_buf gu;             /* [gate | up] pre-activation [N, 2I] */
    rrc_buf act;            /* silu(gate) * up [N, I] */
    rrc_buf t[RRC_LORA_N_MODULES]; /* LoRA x A^T [N, r] f32 (none when not targeted) */
    int valid_layer;        /* layer whose activations are in this cache (-1: none) */
} rrc_layer_cache;

/* Per-step workspace: one backend allocation, carved into typed views. */
typedef struct rrc_workspace {
    void *base;
    size_t bytes;
    int64_t allocs;
    /* plan the views were laid out for */
    int64_t N, n_sel, n_cache, attn_rows, ce_rows, T;
    int with_grad;
    /* views */
    rrc_buf tokens, pos, doc, sel, inv, lab_sel; /* i32 */
    rrc_buf xs;            /* (L + 1) x [N, H]: layer inputs and the last layer's output */
    rrc_buf hf;            /* final norm output [N, H] */
    rrc_layer_cache *cache; /* n_cache entries (host array, views into the arena) */
    rrc_buf dres, dxn, dq, dk, dv, dob, dgu, dact; /* backward scratch */
    rrc_buf dt;            /* [N, r] f32 */
    rrc_buf sc, dp;        /* attention scores / probabilities and dP: [Hq, attn_rows, T] f32 */
    rrc_buf hsel, logits;  /* [ce_rows, H], [ce_rows, V] */
    rrc_buf loss;          /* [n_sel] f32 */
    rrc_buf dh_ext;        /* [n_sel + 1, H] */
} rrc_workspace;

struct rrc_model {
    const rrc_backend *be;
    rrc_model_config cfg;
    rrc_model_opts opts;
    int64_t n_pos;
    rrc_buf embed;     /* [V, H] */
    rrc_buf norm;      /* [H] */
    rrc_buf lm_head;   /* [V, H]; a view of embed when tied */
    rrc_layer_weights *layers;
    rrc_buf cos_tab, sin_tab; /* f32 [n_pos, head_dim / 2] */
    rrc_buf storage;   /* one allocation holding all weights (dtype of the views varies) */
    rrc_buf rope_storage;
    rrc_workspace ws;
    rrc_layer_cache *cache_store; /* host array owned by the model */
    int64_t cache_store_n;
};

/* LoRA internals used by model_exec.c */
typedef struct rrc_lora_slot {
    int present;
    int64_t in, out;
    int64_t off_a, off_b; /* element offsets in params / grads */
} rrc_lora_slot;

struct rrc_lora {
    const rrc_backend *be;
    rrc_lora_config cfg;
    int64_t num_layers;
    int64_t hidden, inter, q_dim, kv_dim;
    float scaling;
    int64_t n_params;
    rrc_buf params, grads;
    rrc_lora_slot *slots; /* [num_layers * RRC_LORA_N_MODULES] */
    rrc_lora_tensor_info *tensors;
    size_t n_tensors;
    int64_t lowest_layer;
};

static inline const rrc_lora_slot *rrc_lora_slot_at(const rrc_lora *l, int64_t layer, int module) {
    return &l->slots[layer * RRC_LORA_N_MODULES + module];
}

/* Typed view of `numel` elements at byte offset `off` of a backend allocation (not owned). */
static inline rrc_buf rrc_raw_view(const rrc_backend *be, void *base, size_t off, rrc_dtype dt, int64_t numel) {
    rrc_buf b;
    b.be = be;
    b.dtype = dt;
    b.numel = numel;
    b.ptr = numel > 0 ? (void *)((char *)base + off) : NULL;
    b.owned = 0;
    return b;
}

#endif
