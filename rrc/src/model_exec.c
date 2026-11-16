/* rrc/src/model_exec.c — decoder forward / backward with LoRA through the op table
 * (rrc/docs/DESIGN.md §3, §5, §10).
 *
 * Token layout: a micro-batch of B packed rows of T tokens is one [N = B*T, *] activation matrix
 * per tensor (row-major, token-major). Attention runs per packed row and per chunk of query rows;
 * everything else runs on all N tokens at once.
 *
 * Per layer (activation dtype = model dtype; LoRA branch fp32):
 *   xn1 = rmsnorm(x); q/k/v = xn1 W^T (+b) + s (xn1 A^T) B^T  (three GEMMs, separate buffers)
 *   [qwen3] q, k = rmsnorm_per_head(q_raw), rmsnorm_per_head(k_raw); rope(q), rope(k)
 *   o = attention(q, k, v); h = x + o Wo^T (+b) + LoRA_o
 *   xn2 = rmsnorm(h); [gate | up] = xn2 [Wg; Wu]^T + LoRA (one GEMM into a fused [N, 2I] buffer)
 *   act = silu(gate) * up; out = h + act Wd^T + LoRA_d
 * Backward recomputes the layer from its saved input x (RRC_RECOMPUTE_FULL) into a scratch
 * rrc_layer_cache, then propagates input gradients through the frozen weights and accumulates
 * the LoRA gradients: dB += s dy^T t, dt = s dy B, dA += dt^T x, dx += dt A.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model_internal.h"

/* ------------------------------------------------------------------------------------------ */
/* Host-side batch preparation                                                                 */
/* ------------------------------------------------------------------------------------------ */

typedef struct prep {
    int64_t B, T, N, n_sel;
    int32_t *pos, *doc, *dstart, *eff, *sel, *inv, *lab_sel;
} prep;

static void prep_free(prep *p) {
    free(p->pos);
    free(p->doc);
    free(p->dstart);
    free(p->eff);
    free(p->sel);
    free(p->inv);
    free(p->lab_sel);
    memset(p, 0, sizeof *p);
}

/* Validate the batch, derive doc ids / positions, shift the labels within documents, and build the
 * selected-row (labelled) index and its inverse. vocab / n_pos <= 0 skip those range checks. */
static rrc_status prep_batch(const rrc_batch *b, int64_t vocab, int64_t n_pos, int all_rows, prep *p) {
    memset(p, 0, sizeof *p);
    RRC_REQUIRE(b, RRC_ERR_INVALID_ARG, "batch is NULL");
    RRC_REQUIRE(b->rows > 0 && b->seq_len > 0, RRC_ERR_SHAPE, "batch: rows = %lld and seq_len = %lld must be > 0",
                (long long)b->rows, (long long)b->seq_len);
    RRC_REQUIRE(b->seq_len <= INT32_MAX && b->rows <= INT32_MAX / b->seq_len, RRC_ERR_SHAPE,
                "batch: rows * seq_len exceeds int32");
    RRC_REQUIRE(b->tokens || vocab <= 0, RRC_ERR_INVALID_ARG, "batch: tokens is NULL");
    const int64_t B = b->rows, T = b->seq_len, N = B * T;
    p->B = B;
    p->T = T;
    p->N = N;
    const size_t sz = (size_t)N * sizeof(int32_t);
    p->pos = (int32_t *)malloc(sz);
    p->doc = (int32_t *)malloc(sz);
    p->dstart = (int32_t *)malloc(sz);
    p->eff = (int32_t *)malloc(sz);
    p->sel = (int32_t *)malloc(sz);
    p->inv = (int32_t *)malloc(sz);
    p->lab_sel = (int32_t *)malloc(sz);
    if (!p->pos || !p->doc || !p->dstart || !p->eff || !p->sel || !p->inv || !p->lab_sel) {
        prep_free(p);
        return RRC_ERROR(RRC_ERR_OOM, "batch preparation (%lld tokens)", (long long)N);
    }
    rrc_status st = RRC_OK;
    for (int64_t r = 0; r < B && st == RRC_OK; r++) {
        int32_t cur_doc = 0, cur_pos = 0;
        int64_t start = 0;
        for (int64_t t = 0; t < T; t++) {
            const int64_t i = r * T + t;
            int new_doc;
            if (b->doc_ids) {
                new_doc = t == 0 || b->doc_ids[i] != b->doc_ids[i - 1];
                cur_doc = b->doc_ids[i];
            } else if (b->position_ids) {
                new_doc = t == 0 || b->position_ids[i] == 0;
                if (new_doc && t > 0) cur_doc++;
            } else {
                new_doc = t == 0;
            }
            if (new_doc) {
                /* documents must be contiguous within a row (the attention key range relies on it) */
                if (b->doc_ids && t > 0)
                    for (int64_t s = r * T; s < i; s++)
                        if (p->doc[s] == cur_doc) {
                            st = RRC_ERROR(RRC_ERR_INVALID_ARG,
                                           "batch: row %lld: document id %d appears again at position %lld after a "
                                           "different document (documents must be contiguous)",
                                           (long long)r, cur_doc, (long long)t);
                            break;
                        }
                start = t;
                cur_pos = 0;
            }
            p->doc[i] = cur_doc;
            p->dstart[i] = (int32_t)start;
            p->pos[i] = b->position_ids ? b->position_ids[i] : cur_pos;
            cur_pos++;
            if (st != RRC_OK) break;
        }
    }
    for (int64_t i = 0; i < N && st == RRC_OK; i++) {
        if (vocab > 0 && (b->tokens[i] < 0 || b->tokens[i] >= vocab))
            st = RRC_ERROR(RRC_ERR_RANGE, "batch: token %d at index %lld outside [0, %lld)", b->tokens[i], (long long)i,
                           (long long)vocab);
        else if (n_pos > 0 && (p->pos[i] < 0 || p->pos[i] >= n_pos))
            st = RRC_ERROR(RRC_ERR_RANGE, "batch: position id %d at index %lld outside [0, %lld)", p->pos[i],
                           (long long)i, (long long)n_pos);
        else if (b->labels && b->labels[i] != RRC_IGNORE_INDEX && (b->labels[i] < 0 || (vocab > 0 && b->labels[i] >= vocab)))
            st = RRC_ERROR(RRC_ERR_RANGE, "batch: label %d at index %lld outside [0, %lld) and not %d", b->labels[i],
                           (long long)i, (long long)vocab, RRC_IGNORE_INDEX);
    }
    if (st != RRC_OK) {
        prep_free(p);
        return st;
    }
    /* HF shift inside documents: logits[t] predicts labels[t + 1] of the same document */
    int64_t ns = 0;
    for (int64_t r = 0; r < B; r++)
        for (int64_t t = 0; t < T; t++) {
            const int64_t i = r * T + t;
            int32_t y = RRC_IGNORE_INDEX;
            if (b->labels && t + 1 < T && p->doc[i + 1] == p->doc[i] && p->dstart[i + 1] == p->dstart[i])
                y = b->labels[i + 1];
            p->eff[i] = y;
            if (all_rows || y != RRC_IGNORE_INDEX) {
                p->inv[i] = (int32_t)ns;
                p->sel[ns] = (int32_t)i;
                p->lab_sel[ns] = y;
                ns++;
            } else {
                p->inv[i] = -1;
            }
        }
    for (int64_t i = 0; i < N; i++)
        if (p->inv[i] < 0) p->inv[i] = (int32_t)ns; /* the zero row of the scatter table */
    p->n_sel = ns;
    return RRC_OK;
}

rrc_status rrc_batch_count_labels(const rrc_batch *b, int64_t *n_out) {
    RRC_REQUIRE(n_out, RRC_ERR_INVALID_ARG, "null argument");
    *n_out = 0;
    prep p;
    RRC_CHECK(prep_batch(b, 0, 0, 0, &p));
    *n_out = p.n_sel;
    prep_free(&p);
    return RRC_OK;
}

/* ------------------------------------------------------------------------------------------ */
/* Workspace                                                                                   */
/* ------------------------------------------------------------------------------------------ */

typedef struct arena {
    const rrc_backend *be;
    char *base; /* NULL: measuring only (no pointer is formed) */
    size_t off;
    int overflow;
} arena;

static rrc_buf take(arena *a, rrc_dtype dt, int64_t n) {
    const size_t align = 256;
    size_t off = (a->off + align - 1) & ~(align - 1), bytes;
    if (n < 0 || __builtin_mul_overflow((size_t)n, rrc_dtype_size(dt), &bytes) ||
        __builtin_add_overflow(off, bytes, &a->off))
        a->overflow = 1;
    if (!a->base || a->overflow) return rrc_buf_none();
    return rrc_raw_view(a->be, a->base, off, dt, n);
}

typedef struct exec {
    rrc_model *m;
    const rrc_lora *lora;
    const rrc_backend *be;
    rrc_workspace *ws;
    int64_t B, T, N;
    int64_t H, Q, KV, I, hd, Hq, Hkv, G, V, L, r;
    float eps, scale;
    rrc_dtype adt;
    const int32_t *dstart; /* host doc start per token */
} exec;

static int64_t mul3(int64_t a, int64_t b, int64_t c) {
    int64_t x;
    if (__builtin_mul_overflow(a, b, &x) || __builtin_mul_overflow(x, c, &x)) return -1;
    return x;
}

/* Lay out every view of the workspace for a plan; with a->base == NULL only the size is computed. */
static void ws_layout(exec *e, arena *a, int with_grad) {
    rrc_workspace *w = e->ws;
    const int64_t N = e->N, H = e->H, L = e->L;
    const rrc_dtype adt = e->adt;
    w->tokens = take(a, RRC_I32, N);
    w->pos = take(a, RRC_I32, N);
    w->doc = take(a, RRC_I32, N);
    w->sel = take(a, RRC_I32, N);
    w->inv = take(a, RRC_I32, N);
    w->lab_sel = take(a, RRC_I32, N);
    w->xs = take(a, adt, mul3(L + 1, N, H));
    w->hf = take(a, adt, N * H);
    for (int64_t c = 0; c < w->n_cache; c++) {
        rrc_layer_cache *lc = &w->cache[c];
        lc->xn1 = take(a, adt, N * H);
        lc->q_raw = e->m->cfg.qk_norm ? take(a, adt, N * e->Q) : rrc_buf_none();
        lc->k_raw = e->m->cfg.qk_norm ? take(a, adt, N * e->KV) : rrc_buf_none();
        lc->q = take(a, adt, N * e->Q);
        lc->k = take(a, adt, N * e->KV);
        lc->v = take(a, adt, N * e->KV);
        lc->o = take(a, adt, N * e->Q);
        lc->h = take(a, adt, N * H);
        lc->xn2 = take(a, adt, N * H);
        lc->gu = take(a, adt, mul3(N, 2, e->I));
        lc->act = take(a, adt, N * e->I);
        for (int mod = 0; mod < RRC_LORA_N_MODULES; mod++) {
            int any = 0;
            for (int64_t l = 0; e->lora && l < L; l++) any |= rrc_lora_targets(e->lora, l, mod);
            lc->t[mod] = any ? take(a, RRC_F32, N * e->r) : rrc_buf_none();
        }
        lc->valid_layer = -1;
    }
    const int64_t arows = e->m->opts.attn_chunk < e->T ? e->m->opts.attn_chunk : e->T;
    w->attn_rows = arows;
    w->sc = take(a, RRC_F32, mul3(e->Hq, arows, e->T));
    const int64_t crows = e->m->opts.ce_chunk < N ? e->m->opts.ce_chunk : N;
    w->ce_rows = crows;
    w->hsel = take(a, adt, crows * H);
    w->logits = take(a, adt, crows * e->V);
    w->loss = take(a, RRC_F32, N);
    if (with_grad) {
        w->dp = take(a, RRC_F32, mul3(e->Hq, arows, e->T));
        w->dh_ext = take(a, adt, (N + 1) * H);
        w->dres = take(a, adt, N * H);
        w->dxn = take(a, RRC_F32, N * H);
        w->dq = take(a, adt, N * e->Q);
        w->dk = take(a, RRC_F32, N * e->KV);
        w->dv = take(a, RRC_F32, N * e->KV);
        w->dob = take(a, adt, N * e->Q);
        w->dgu = take(a, adt, mul3(N, 2, e->I));
        w->dact = take(a, adt, N * e->I);
        w->dt = take(a, RRC_F32, N * (e->r > 0 ? e->r : 1));
    } else {
        w->dp = w->dh_ext = w->dres = w->dxn = w->dq = w->dk = w->dv = w->dob = w->dgu = w->dact = w->dt =
            rrc_buf_none();
    }
}

static rrc_status ws_prepare(exec *e, int with_grad) {
    rrc_model *m = e->m;
    rrc_workspace *w = &m->ws;
    e->ws = w;
    const int64_t n_cache = m->opts.recompute == RRC_RECOMPUTE_NONE ? e->L : 1;
    if (m->cache_store_n < n_cache) {
        rrc_layer_cache *cs = (rrc_layer_cache *)realloc(m->cache_store, (size_t)n_cache * sizeof(rrc_layer_cache));
        if (!cs) return RRC_ERROR(RRC_ERR_OOM, "layer cache table");
        m->cache_store = cs;
        m->cache_store_n = n_cache;
    }
    memset(m->cache_store, 0, (size_t)n_cache * sizeof(rrc_layer_cache));
    w->cache = m->cache_store;
    w->n_cache = n_cache;
    arena a = {e->be, NULL, 0, 0};
    ws_layout(e, &a, with_grad);
    RRC_REQUIRE(!a.overflow, RRC_ERR_SHAPE, "workspace size overflows for %lld tokens", (long long)e->N);
    if (a.off > w->bytes) {
        if (w->base) m->be->free(m->be, w->base);
        w->base = NULL;
        w->bytes = 0;
        void *p = NULL;
        RRC_CHECK_CTX(m->be->alloc(m->be, a.off, &p), "workspace of %zu bytes", a.off);
        w->base = p;
        w->bytes = a.off;
        w->allocs++;
    }
    arena b = {e->be, (char *)w->base, 0, 0};
    ws_layout(e, &b, with_grad);
    if (b.overflow || b.off != a.off) return RRC_ERROR(RRC_ERR_INTERNAL, "workspace layout is not reproducible");
    w->N = e->N;
    w->T = e->T;
    w->with_grad = with_grad;
    return RRC_OK;
}

static void exec_init(exec *e, rrc_model *m, const rrc_lora *lora, const prep *p) {
    memset(e, 0, sizeof *e);
    const rrc_model_config *c = &m->cfg;
    e->m = m;
    e->lora = lora;
    e->be = m->be;
    e->B = p->B;
    e->T = p->T;
    e->N = p->N;
    e->H = c->hidden_size;
    e->hd = c->head_dim;
    e->Hq = c->num_heads;
    e->Hkv = c->num_kv_heads;
    e->G = e->Hq / e->Hkv;
    e->Q = e->Hq * e->hd;
    e->KV = e->Hkv * e->hd;
    e->I = c->intermediate_size;
    e->V = c->vocab_size;
    e->L = c->num_layers;
    e->r = lora ? lora->cfg.r : 0;
    e->eps = (float)c->rms_norm_eps;
    e->scale = (float)(1.0 / sqrt((double)e->hd));
    e->adt = m->opts.dtype;
    e->dstart = p->dstart;
}

static rrc_status upload_batch(exec *e, const rrc_batch *b, const prep *p) {
    rrc_workspace *w = e->ws;
    RRC_CHECK(rrc_buf_upload(w->tokens, b->tokens, e->N));
    RRC_CHECK(rrc_buf_upload(w->pos, p->pos, e->N));
    RRC_CHECK(rrc_buf_upload(w->doc, p->doc, e->N));
    RRC_CHECK(rrc_buf_upload(w->sel, p->sel, p->n_sel));
    RRC_CHECK(rrc_buf_upload(w->inv, p->inv, e->N));
    RRC_CHECK(rrc_buf_upload(w->lab_sel, p->lab_sel, p->n_sel));
    return RRC_OK;
}

/* ------------------------------------------------------------------------------------------ */
/* Linear layers with LoRA                                                                     */
/* ------------------------------------------------------------------------------------------ */

/* y (ld ldy) = x (ld ldx) W^T, W [out, in] */
static rrc_status linear_fwd(exec *e, rrc_buf x, int64_t ldx, int64_t in, rrc_buf W, rrc_buf y, int64_t ldy,
                             int64_t out) {
    return rrc_gemm2d(e->be, RRC_OP_N, RRC_OP_T, e->N, out, in, 1.0f, x, ldx, W, in, 0.0f, y, ldy);
}

/* dx (ld ldx) (beta) = dy (ld ldy) W */
static rrc_status linear_bwd(exec *e, rrc_buf dy, int64_t ldy, int64_t out, rrc_buf W, rrc_buf dx, int64_t ldx,
                             int64_t in, float beta) {
    return rrc_gemm2d(e->be, RRC_OP_N, RRC_OP_N, e->N, in, out, 1.0f, dy, ldy, W, in, beta, dx, ldx);
}

/* t = x A^T (fp32) and, if `apply`, y += s * t B^T. No-op if the module is not targeted. */
static rrc_status lora_fwd(exec *e, int64_t layer, int mod, rrc_buf x, int64_t ldx, rrc_buf t, rrc_buf y, int64_t ldy,
                           int apply) {
    if (!e->lora || !rrc_lora_targets(e->lora, layer, mod)) return RRC_OK;
    const rrc_lora_slot *s = rrc_lora_slot_at(e->lora, layer, mod);
    rrc_buf A, Bm;
    RRC_CHECK(rrc_lora_module_bufs(e->lora, layer, mod, &A, &Bm, NULL, NULL));
    RRC_CHECK(rrc_gemm2d(e->be, RRC_OP_N, RRC_OP_T, e->N, e->r, s->in, 1.0f, x, ldx, A, s->in, 0.0f, t, e->r));
    if (!apply) return RRC_OK;
    return rrc_gemm2d(e->be, RRC_OP_N, RRC_OP_T, e->N, s->out, e->r, e->lora->scaling, t, e->r, Bm, e->r, 1.0f, y, ldy);
}

/* dy: gradient of the module output (ld ldy); x: module input (ld ldx); t: saved x A^T.
 * dB += s dy^T t; dt = s dy B; dA += dt^T x; dx += dt A (dx may be none). */
static rrc_status lora_bwd(exec *e, int64_t layer, int mod, rrc_buf dy, int64_t ldy, rrc_buf x, int64_t ldx, rrc_buf t,
                           rrc_buf dx, int64_t ldxo) {
    if (!e->lora || !rrc_lora_targets(e->lora, layer, mod)) return RRC_OK;
    const rrc_lora_slot *s = rrc_lora_slot_at(e->lora, layer, mod);
    const float sc = e->lora->scaling;
    const int64_t r = e->r, N = e->N;
    rrc_buf A, Bm, gA, gB;
    RRC_CHECK(rrc_lora_module_bufs(e->lora, layer, mod, &A, &Bm, &gA, &gB));
    rrc_buf dt = e->ws->dt;
    RRC_CHECK(rrc_gemm2d(e->be, RRC_OP_T, RRC_OP_N, s->out, r, N, sc, dy, ldy, t, r, 1.0f, gB, r));
    RRC_CHECK(rrc_gemm2d(e->be, RRC_OP_N, RRC_OP_N, N, r, s->out, sc, dy, ldy, Bm, r, 0.0f, dt, r));
    RRC_CHECK(rrc_gemm2d(e->be, RRC_OP_T, RRC_OP_N, r, s->in, N, 1.0f, dt, r, x, ldx, 1.0f, gA, s->in));
    if (rrc_buf_is_none(dx)) return RRC_OK;
    return rrc_gemm2d(e->be, RRC_OP_N, RRC_OP_N, N, s->in, r, 1.0f, dt, r, A, s->in, 1.0f, dx, ldxo);
}

/* ------------------------------------------------------------------------------------------ */
/* Attention (per packed row, per chunk of query rows; keys from the chunk's first document)    */
/* ------------------------------------------------------------------------------------------ */

static rrc_gemm_args gargs(rrc_trans ta, rrc_trans tb, int64_t m, int64_t n, int64_t k, float beta, rrc_buf a,
                           int64_t lda, int64_t sa, rrc_buf b, int64_t ldb, int64_t sb, rrc_buf c, int64_t ldc,
                           int64_t sc, int64_t batch) {
    rrc_gemm_args g;
    g.trans_a = ta;
    g.trans_b = tb;
    g.m = m;
    g.n = n;
    g.k = k;
    g.alpha = 1.0f;
    g.beta = beta;
    g.a = a;
    g.lda = lda;
    g.stride_a = sa;
    g.b = b;
    g.ldb = ldb;
    g.stride_b = sb;
    g.c = c;
    g.ldc = ldc;
    g.stride_c = sc;
    g.batch = batch;
    return g;
}

typedef struct chunk {
    int64_t b, q0, R, k0, Kc, bs; /* row, first query, rows, first key, keys, per-head slice */
} chunk;

/* S_h = Q_h K_g^T for the chunk, then P = masked softmax (in place in w->sc). */
static rrc_status attn_scores(exec *e, const rrc_layer_cache *c, const chunk *ch) {
    rrc_workspace *w = e->ws;
    const int64_t G = e->G, hd = e->hd, Q = e->Q, KV = e->KV, T = e->T;
    for (int64_t g = 0; g < e->Hkv; g++) {
        rrc_gemm_args a = gargs(RRC_OP_N, RRC_OP_T, ch->R, ch->Kc, hd, 0.0f,
                                rrc_view_from(c->q, (ch->b * T + ch->q0) * Q + g * G * hd), Q, hd,
                                rrc_view_from(c->k, (ch->b * T + ch->k0) * KV + g * hd), KV, 0,
                                rrc_view_from(w->sc, g * G * ch->bs), ch->Kc, ch->bs, G);
        RRC_CHECK(rrc_gemm(e->be, &a));
    }
    return rrc_attn_softmax_fwd(e->be, e->Hq, ch->R, ch->Kc, ch->Kc, ch->bs, w->sc, e->scale, ch->q0 - ch->k0, 1,
                                rrc_view(w->doc, ch->b * T + ch->k0, T - ch->k0), w->sc);
}

static void chunk_at(const exec *e, int64_t b, int64_t q0, chunk *ch) {
    const int64_t Tc = e->ws->attn_rows;
    ch->b = b;
    ch->q0 = q0;
    ch->R = Tc < e->T - q0 ? Tc : e->T - q0;
    ch->k0 = e->dstart[b * e->T + q0]; /* keys before the first query's document are masked for all rows */
    ch->Kc = q0 + ch->R - ch->k0;
    ch->bs = ch->R * ch->Kc;
}

static rrc_status attn_fwd(exec *e, rrc_layer_cache *c) {
    rrc_workspace *w = e->ws;
    const int64_t G = e->G, hd = e->hd, Q = e->Q, KV = e->KV, T = e->T;
    for (int64_t b = 0; b < e->B; b++)
        for (int64_t q0 = 0; q0 < T; q0 += w->attn_rows) {
            chunk ch;
            chunk_at(e, b, q0, &ch);
            RRC_CHECK(attn_scores(e, c, &ch));
            for (int64_t g = 0; g < e->Hkv; g++) {
                rrc_gemm_args a = gargs(RRC_OP_N, RRC_OP_N, ch.R, hd, ch.Kc, 0.0f, rrc_view_from(w->sc, g * G * ch.bs),
                                        ch.Kc, ch.bs, rrc_view_from(c->v, (b * T + ch.k0) * KV + g * hd), KV, 0,
                                        rrc_view_from(c->o, (b * T + q0) * Q + g * G * hd), Q, hd, G);
                RRC_CHECK(rrc_gemm(e->be, &a));
            }
        }
    return RRC_OK;
}

/* dO (w->dob) -> dQ (w->dq, written), dK, dV (w->dk, w->dv, accumulated: zeroed by the caller). */
static rrc_status attn_bwd(exec *e, const rrc_layer_cache *c) {
    rrc_workspace *w = e->ws;
    const int64_t G = e->G, hd = e->hd, Q = e->Q, KV = e->KV, T = e->T, Hkv = e->Hkv;
    for (int64_t b = 0; b < e->B; b++)
        for (int64_t q0 = 0; q0 < T; q0 += w->attn_rows) {
            chunk ch;
            chunk_at(e, b, q0, &ch);
            RRC_CHECK(attn_scores(e, c, &ch)); /* P recomputed into w->sc */
            const int64_t qrow = (b * T + q0), krow = (b * T + ch.k0);
            /* dP_h = dO_h V_g^T */
            for (int64_t g = 0; g < Hkv; g++) {
                rrc_gemm_args a = gargs(RRC_OP_N, RRC_OP_T, ch.R, ch.Kc, hd, 0.0f,
                                        rrc_view_from(w->dob, qrow * Q + g * G * hd), Q, hd,
                                        rrc_view_from(c->v, krow * KV + g * hd), KV, 0,
                                        rrc_view_from(w->dp, g * G * ch.bs), ch.Kc, ch.bs, G);
                RRC_CHECK(rrc_gemm(e->be, &a));
            }
            /* dV_g += sum_u P_(gG+u)^T dO_(gG+u): one call per in-group head u, batched over g */
            for (int64_t u = 0; u < G; u++) {
                rrc_gemm_args a = gargs(RRC_OP_T, RRC_OP_N, ch.Kc, hd, ch.R, 1.0f, rrc_view_from(w->sc, u * ch.bs),
                                        ch.Kc, G * ch.bs, rrc_view_from(w->dob, qrow * Q + u * hd), Q, G * hd,
                                        rrc_view_from(w->dv, krow * KV), KV, hd, Hkv);
                RRC_CHECK(rrc_gemm(e->be, &a));
            }
            /* dS (in place over dP) */
            RRC_CHECK(rrc_attn_softmax_bwd(e->be, e->Hq, ch.R, ch.Kc, ch.Kc, ch.bs, w->sc, w->dp, e->scale, w->dp));
            /* dQ_h = dS_h K_g */
            for (int64_t g = 0; g < Hkv; g++) {
                rrc_gemm_args a = gargs(RRC_OP_N, RRC_OP_N, ch.R, hd, ch.Kc, 0.0f, rrc_view_from(w->dp, g * G * ch.bs),
                                        ch.Kc, ch.bs, rrc_view_from(c->k, krow * KV + g * hd), KV, 0,
                                        rrc_view_from(w->dq, qrow * Q + g * G * hd), Q, hd, G);
                RRC_CHECK(rrc_gemm(e->be, &a));
            }
            /* dK_g += sum_u dS_(gG+u)^T Q_(gG+u) */
            for (int64_t u = 0; u < G; u++) {
                rrc_gemm_args a = gargs(RRC_OP_T, RRC_OP_N, ch.Kc, hd, ch.R, 1.0f, rrc_view_from(w->dp, u * ch.bs),
                                        ch.Kc, G * ch.bs, rrc_view_from(c->q, qrow * Q + u * hd), Q, G * hd,
                                        rrc_view_from(w->dk, krow * KV), KV, hd, Hkv);
                RRC_CHECK(rrc_gemm(e->be, &a));
            }
        }
    return RRC_OK;
}

/* ------------------------------------------------------------------------------------------ */
/* Decoder layer                                                                               */
/* ------------------------------------------------------------------------------------------ */

/* Forward of layer l from input x into cache c. out == none (recompute): the down projection and
 * the residual are skipped (only the LoRA t of down_proj is produced). */
static rrc_status layer_fwd(exec *e, int64_t l, rrc_buf x, rrc_layer_cache *c, rrc_buf out) {
    const rrc_model *m = e->m;
    const rrc_layer_weights *wt = &m->layers[l];
    const rrc_workspace *w = e->ws;
    const rrc_backend *be = e->be;
    const int64_t N = e->N, H = e->H, Q = e->Q, KV = e->KV, I = e->I, hd = e->hd;
    const int qkn = m->cfg.qk_norm;
    c->valid_layer = -1;
    RRC_CHECK(rrc_rmsnorm_fwd(be, N, H, H, x, wt->ln1, e->eps, c->xn1));
    rrc_buf qd = qkn ? c->q_raw : c->q, kd = qkn ? c->k_raw : c->k;
    RRC_CHECK(linear_fwd(e, c->xn1, H, H, wt->wq, qd, Q, Q));
    if (!rrc_buf_is_none(wt->bq)) RRC_CHECK(rrc_add_bias(be, N, Q, qd, Q, wt->bq));
    RRC_CHECK(lora_fwd(e, l, RRC_LORA_Q, c->xn1, H, c->t[RRC_LORA_Q], qd, Q, 1));
    RRC_CHECK(linear_fwd(e, c->xn1, H, H, wt->wk, kd, KV, KV));
    if (!rrc_buf_is_none(wt->bk)) RRC_CHECK(rrc_add_bias(be, N, KV, kd, KV, wt->bk));
    RRC_CHECK(lora_fwd(e, l, RRC_LORA_K, c->xn1, H, c->t[RRC_LORA_K], kd, KV, 1));
    RRC_CHECK(linear_fwd(e, c->xn1, H, H, wt->wv, c->v, KV, KV));
    if (!rrc_buf_is_none(wt->bv)) RRC_CHECK(rrc_add_bias(be, N, KV, c->v, KV, wt->bv));
    RRC_CHECK(lora_fwd(e, l, RRC_LORA_V, c->xn1, H, c->t[RRC_LORA_V], c->v, KV, 1));
    if (qkn) {
        RRC_CHECK(rrc_rmsnorm_fwd(be, N * e->Hq, hd, hd, c->q_raw, wt->qn, e->eps, c->q));
        RRC_CHECK(rrc_rmsnorm_fwd(be, N * e->Hkv, hd, hd, c->k_raw, wt->kn, e->eps, c->k));
    }
    RRC_CHECK(rrc_rope_fwd(be, N, e->Hq, hd, Q, c->q, w->pos, m->cos_tab, m->sin_tab, m->n_pos, c->q));
    RRC_CHECK(rrc_rope_fwd(be, N, e->Hkv, hd, KV, c->k, w->pos, m->cos_tab, m->sin_tab, m->n_pos, c->k));
    RRC_CHECK(attn_fwd(e, c));
    RRC_CHECK(linear_fwd(e, c->o, Q, Q, wt->wo, c->h, H, H));
    if (!rrc_buf_is_none(wt->bo)) RRC_CHECK(rrc_add_bias(be, N, H, c->h, H, wt->bo));
    RRC_CHECK(lora_fwd(e, l, RRC_LORA_O, c->o, Q, c->t[RRC_LORA_O], c->h, H, 1));
    RRC_CHECK(rrc_add(be, N * H, c->h, x, c->h));
    RRC_CHECK(rrc_rmsnorm_fwd(be, N, H, H, c->h, wt->ln2, e->eps, c->xn2));
    RRC_CHECK(linear_fwd(e, c->xn2, H, H, wt->wgu, c->gu, 2 * I, 2 * I));
    RRC_CHECK(lora_fwd(e, l, RRC_LORA_GATE, c->xn2, H, c->t[RRC_LORA_GATE], c->gu, 2 * I, 1));
    RRC_CHECK(lora_fwd(e, l, RRC_LORA_UP, c->xn2, H, c->t[RRC_LORA_UP], rrc_view_from(c->gu, I), 2 * I, 1));
    RRC_CHECK(rrc_swiglu_fwd(be, N, I, c->gu, rrc_view_from(c->gu, I), 2 * I, c->act, I));
    if (rrc_buf_is_none(out)) {
        RRC_CHECK(lora_fwd(e, l, RRC_LORA_DOWN, c->act, I, c->t[RRC_LORA_DOWN], rrc_buf_none(), H, 0));
    } else {
        RRC_CHECK(linear_fwd(e, c->act, I, I, wt->wd, out, H, H));
        RRC_CHECK(lora_fwd(e, l, RRC_LORA_DOWN, c->act, I, c->t[RRC_LORA_DOWN], out, H, 1));
        RRC_CHECK(rrc_add(be, N * H, out, c->h, out));
    }
    c->valid_layer = (int)l;
    return RRC_OK;
}

/* Backward of layer l: w->dres holds d(out) on entry and d(x) on exit (if need_dx). */
static rrc_status layer_bwd(exec *e, int64_t l, rrc_buf x, const rrc_layer_cache *c, int need_dx) {
    const rrc_model *m = e->m;
    const rrc_layer_weights *wt = &m->layers[l];
    const rrc_workspace *w = e->ws;
    const rrc_backend *be = e->be;
    const int64_t N = e->N, H = e->H, Q = e->Q, KV = e->KV, I = e->I, hd = e->hd;
    /* ---- MLP ---- */
    RRC_CHECK(linear_bwd(e, w->dres, H, H, wt->wd, w->dact, I, I, 0.0f));
    RRC_CHECK(lora_bwd(e, l, RRC_LORA_DOWN, w->dres, H, c->act, I, c->t[RRC_LORA_DOWN], w->dact, I));
    RRC_CHECK(rrc_swiglu_bwd(be, N, I, c->gu, rrc_view_from(c->gu, I), 2 * I, w->dact, I, w->dgu,
                             rrc_view_from(w->dgu, I), 2 * I));
    RRC_CHECK(linear_bwd(e, w->dgu, 2 * I, 2 * I, wt->wgu, w->dxn, H, H, 0.0f));
    RRC_CHECK(lora_bwd(e, l, RRC_LORA_GATE, w->dgu, 2 * I, c->xn2, H, c->t[RRC_LORA_GATE], w->dxn, H));
    RRC_CHECK(lora_bwd(e, l, RRC_LORA_UP, rrc_view_from(w->dgu, I), 2 * I, c->xn2, H, c->t[RRC_LORA_UP], w->dxn, H));
    RRC_CHECK(rrc_rmsnorm_bwd(be, N, H, H, c->h, wt->ln2, e->eps, w->dxn, w->dres, 1));
    /* ---- attention ---- */
    RRC_CHECK(linear_bwd(e, w->dres, H, H, wt->wo, w->dob, Q, Q, 0.0f));
    RRC_CHECK(lora_bwd(e, l, RRC_LORA_O, w->dres, H, c->o, Q, c->t[RRC_LORA_O], w->dob, Q));
    RRC_CHECK(rrc_buf_zero(w->dk, N * KV));
    RRC_CHECK(rrc_buf_zero(w->dv, N * KV));
    RRC_CHECK(attn_bwd(e, c));
    RRC_CHECK(rrc_rope_bwd(be, N, e->Hq, hd, Q, w->dq, w->pos, m->cos_tab, m->sin_tab, m->n_pos, w->dq));
    RRC_CHECK(rrc_rope_bwd(be, N, e->Hkv, hd, KV, w->dk, w->pos, m->cos_tab, m->sin_tab, m->n_pos, w->dk));
    if (m->cfg.qk_norm) {
        RRC_CHECK(rrc_rmsnorm_bwd(be, N * e->Hq, hd, hd, c->q_raw, wt->qn, e->eps, w->dq, w->dq, 0));
        RRC_CHECK(rrc_rmsnorm_bwd(be, N * e->Hkv, hd, hd, c->k_raw, wt->kn, e->eps, w->dk, w->dk, 0));
    }
    RRC_CHECK(linear_bwd(e, w->dq, Q, Q, wt->wq, w->dxn, H, H, 0.0f));
    RRC_CHECK(linear_bwd(e, w->dk, KV, KV, wt->wk, w->dxn, H, H, 1.0f));
    RRC_CHECK(linear_bwd(e, w->dv, KV, KV, wt->wv, w->dxn, H, H, 1.0f));
    RRC_CHECK(lora_bwd(e, l, RRC_LORA_Q, w->dq, Q, c->xn1, H, c->t[RRC_LORA_Q], w->dxn, H));
    RRC_CHECK(lora_bwd(e, l, RRC_LORA_K, w->dk, KV, c->xn1, H, c->t[RRC_LORA_K], w->dxn, H));
    RRC_CHECK(lora_bwd(e, l, RRC_LORA_V, w->dv, KV, c->xn1, H, c->t[RRC_LORA_V], w->dxn, H));
    if (need_dx) RRC_CHECK(rrc_rmsnorm_bwd(be, N, H, H, x, wt->ln1, e->eps, w->dxn, w->dres, 1));
    return RRC_OK;
}

/* ------------------------------------------------------------------------------------------ */
/* Model forward / backward                                                                    */
/* ------------------------------------------------------------------------------------------ */

static rrc_buf xs_at(const exec *e, int64_t l) { return rrc_view(e->ws->xs, l * e->N * e->H, e->N * e->H); }
static rrc_layer_cache *cache_for(const exec *e, int64_t l) { return &e->ws->cache[e->ws->n_cache == 1 ? 0 : l]; }

static rrc_status forward_layers(exec *e) {
    const rrc_model *m = e->m;
    const rrc_workspace *w = e->ws;
    RRC_CHECK(rrc_embedding_fwd(e->be, e->N, e->H, e->V, w->tokens, m->embed, xs_at(e, 0)));
    for (int64_t l = 0; l < e->L; l++)
        RRC_CHECK_CTX(layer_fwd(e, l, xs_at(e, l), cache_for(e, l), xs_at(e, l + 1)), "layer %lld forward",
                      (long long)l);
    return rrc_rmsnorm_fwd(e->be, e->N, e->H, e->H, xs_at(e, e->L), m->norm, e->eps, w->hf);
}

static rrc_status check_lora(const rrc_model *m, const rrc_lora *lora) {
    if (!lora) return RRC_OK;
    const rrc_model_config *c = &m->cfg;
    RRC_REQUIRE(lora->be == m->be, RRC_ERR_INVALID_ARG, "LoRA adapter and model are on different backends");
    RRC_REQUIRE(lora->num_layers == c->num_layers && lora->hidden == c->hidden_size &&
                    lora->inter == c->intermediate_size && lora->q_dim == c->num_heads * c->head_dim &&
                    lora->kv_dim == c->num_kv_heads * c->head_dim,
                RRC_ERR_SHAPE, "LoRA adapter was created for a different model shape");
    return RRC_OK;
}

rrc_status rrc_model_forward_backward(rrc_model *m, rrc_lora *lora, const rrc_batch *b, const rrc_fb_opts *o,
                                      rrc_fb_result *res) {
    RRC_REQUIRE(m && b && o && res, RRC_ERR_INVALID_ARG, "null argument");
    memset(res, 0, sizeof *res);
    RRC_CHECK(check_lora(m, lora));
    RRC_REQUIRE(o->reduction == RRC_LOSS_WINDOW_TOKENS || o->reduction == RRC_LOSS_MICROBATCH_MEAN,
                RRC_ERR_INVALID_ARG, "invalid loss reduction %d", (int)o->reduction);
    prep p;
    RRC_CHECK(prep_batch(b, m->cfg.vocab_size, m->n_pos, 0, &p));
    rrc_status st = RRC_OK;
    double scale = 0.0;
    if (o->reduction == RRC_LOSS_WINDOW_TOKENS) {
        if (o->window_tokens < p.n_sel) {
            st = RRC_ERROR(RRC_ERR_INVALID_ARG, "window_tokens = %lld is smaller than this micro-batch's %lld labelled "
                                                "tokens",
                           (long long)o->window_tokens, (long long)p.n_sel);
            goto done;
        }
        scale = o->window_tokens > 0 ? 1.0 / (double)o->window_tokens : 0.0;
    } else {
        if (o->num_micro_batches < 1) {
            st = RRC_ERROR(RRC_ERR_INVALID_ARG, "num_micro_batches = %lld must be >= 1", (long long)o->num_micro_batches);
            goto done;
        }
        scale = p.n_sel > 0 ? 1.0 / ((double)p.n_sel * (double)o->num_micro_batches) : 0.0;
    }
    res->n_tokens = p.n_sel;
    res->grad_scale = (float)scale;
    if (p.n_sel == 0) goto done; /* zero labelled tokens: loss 0, gradient 0 (DESIGN.md §10) */
    {
        exec e;
        exec_init(&e, m, lora, &p);
        const int with_grad = o->compute_grad && lora && lora->lowest_layer < e.L;
        RRC_CHECK_GOTO(st, ws_prepare(&e, with_grad), done);
        rrc_workspace *w = e.ws;
        RRC_CHECK_GOTO(st, upload_batch(&e, b, &p), done);
        RRC_CHECK_GOTO(st, forward_layers(&e), done);
        /* LM head + fused cross entropy over the labelled rows, ce_rows at a time */
        const int64_t H = e.H, V = e.V;
        for (int64_t c0 = 0; c0 < p.n_sel; c0 += w->ce_rows) {
            const int64_t nc = w->ce_rows < p.n_sel - c0 ? w->ce_rows : p.n_sel - c0;
            RRC_CHECK_GOTO(st, rrc_embedding_fwd(e.be, nc, H, e.N, rrc_view(w->sel, c0, nc), w->hf, w->hsel), done);
            RRC_CHECK_GOTO(st,
                           rrc_gemm2d(e.be, RRC_OP_N, RRC_OP_T, nc, V, H, 1.0f, w->hsel, H, m->lm_head, H, 0.0f,
                                      w->logits, V),
                           done);
            RRC_CHECK_GOTO(st,
                           rrc_cross_entropy_fwd_bwd(e.be, nc, V, w->logits, V, rrc_view(w->lab_sel, c0, nc),
                                                     rrc_view(w->loss, c0, nc), (float)scale, rrc_buf_none(), with_grad),
                           done);
            if (with_grad)
                RRC_CHECK_GOTO(st,
                               rrc_gemm2d(e.be, RRC_OP_N, RRC_OP_N, nc, H, V, 1.0f, w->logits, V, m->lm_head, H, 0.0f,
                                          rrc_view_from(w->dh_ext, c0 * H), H),
                               done);
        }
        float *lh = (float *)malloc((size_t)p.n_sel * sizeof(float));
        if (!lh) {
            st = RRC_ERROR(RRC_ERR_OOM, "loss download");
            goto done;
        }
        st = rrc_buf_download(w->loss, lh, p.n_sel);
        double sum = 0.0;
        for (int64_t i = 0; i < p.n_sel; i++) sum += (double)lh[i];
        free(lh);
        if (st != RRC_OK) goto done;
        res->loss_sum = sum;
        res->loss = sum * scale;
        if (!with_grad || scale == 0.0) goto done;
        /* scatter d(final hidden) back to all N rows (unlabelled rows get the zero row n_sel) */
        RRC_CHECK_GOTO(st, rrc_buf_zero(rrc_view(w->dh_ext, p.n_sel * H, H), H), done);
        RRC_CHECK_GOTO(st, rrc_embedding_fwd(e.be, e.N, H, p.n_sel + 1, w->inv, w->dh_ext, w->dres), done);
        RRC_CHECK_GOTO(st, rrc_rmsnorm_bwd(e.be, e.N, H, H, xs_at(&e, e.L), m->norm, e.eps, w->dres, w->dres, 0), done);
        const int64_t lo = lora->lowest_layer;
        for (int64_t l = e.L - 1; l >= lo; l--) {
            rrc_layer_cache *c = cache_for(&e, l);
            if (c->valid_layer != (int)l) {
                st = layer_fwd(&e, l, xs_at(&e, l), c, rrc_buf_none());
                if (st != RRC_OK) {
                    rrc_error_context(st, __FILE__, __LINE__, "layer %lld recompute", (long long)l);
                    goto done;
                }
            }
            st = layer_bwd(&e, l, xs_at(&e, l), c, l > lo);
            if (st != RRC_OK) {
                rrc_error_context(st, __FILE__, __LINE__, "layer %lld backward", (long long)l);
                goto done;
            }
        }
        st = rrc_sync(e.be);
    }
done:
    prep_free(&p);
    return st;
}

static rrc_status download_f32(rrc_buf src, int64_t n, float *dst) {
    if (src.dtype == RRC_F32) return rrc_buf_download(src, dst, n);
    rrc_bf16 *tmp = (rrc_bf16 *)malloc((size_t)(n > 0 ? n : 1) * sizeof(rrc_bf16));
    if (!tmp) return RRC_ERROR(RRC_ERR_OOM, "download staging");
    rrc_status st = rrc_buf_download(src, tmp, n);
    for (int64_t i = 0; st == RRC_OK && i < n; i++) dst[i] = rrc_bf16_to_f32(tmp[i]);
    free(tmp);
    return st;
}

rrc_status rrc_model_forward_logits(rrc_model *m, const rrc_lora *lora, const rrc_batch *b, float *logits_out,
                                    float *hidden_out) {
    RRC_REQUIRE(m && b && logits_out, RRC_ERR_INVALID_ARG, "null argument");
    RRC_CHECK(check_lora(m, lora));
    prep p;
    RRC_CHECK(prep_batch(b, m->cfg.vocab_size, m->n_pos, 1, &p));
    rrc_status st = RRC_OK;
    exec e;
    exec_init(&e, m, lora, &p);
    RRC_CHECK_GOTO(st, ws_prepare(&e, 0), done);
    rrc_workspace *w = e.ws;
    RRC_CHECK_GOTO(st, upload_batch(&e, b, &p), done);
    RRC_CHECK_GOTO(st, forward_layers(&e), done);
    const int64_t H = e.H, V = e.V;
    for (int64_t c0 = 0; c0 < p.n_sel; c0 += w->ce_rows) {
        const int64_t nc = w->ce_rows < p.n_sel - c0 ? w->ce_rows : p.n_sel - c0;
        RRC_CHECK_GOTO(st, rrc_embedding_fwd(e.be, nc, H, e.N, rrc_view(w->sel, c0, nc), w->hf, w->hsel), done);
        RRC_CHECK_GOTO(
            st, rrc_gemm2d(e.be, RRC_OP_N, RRC_OP_T, nc, V, H, 1.0f, w->hsel, H, m->lm_head, H, 0.0f, w->logits, V),
            done);
        RRC_CHECK_GOTO(st, download_f32(w->logits, nc * V, logits_out + c0 * V), done);
    }
    if (hidden_out) {
        for (int64_t l = 0; l < e.L; l++)
            RRC_CHECK_GOTO(st, download_f32(xs_at(&e, l), e.N * H, hidden_out + l * e.N * H), done);
        RRC_CHECK_GOTO(st, download_f32(w->hf, e.N * H, hidden_out + e.L * e.N * H), done);
    }
done:
    prep_free(&p);
    return st;
}
