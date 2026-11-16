/* rrc/cuda/kc_emu.c — host emulation of the CUDA backend, see kc_emu.h.
 *
 * Every op mirrors the structure of its CUDA kernel(s) in cuda_kernels.cu: the same item bodies,
 * the same phases, the same fixed tree. Within a phase the items run in a pseudo-random order
 * (a fresh permutation per phase), which stands in for the arbitrary scheduling of GPU threads;
 * a phase completes before the next starts (the __syncthreads / kernel boundary on the device).
 * Range errors behave like the device: the offending items write nothing, the rest of the op runs,
 * and the error is reported by the op (default) or by the next sync / memcpy_d2h
 * (deferred_errors). */
#include "kc_emu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rrc_kc_items.h"

_Static_assert(RRC_KC_IGNORE_INDEX == RRC_IGNORE_INDEX, "kernels_common ignore index must match backend.h");

#define LANES RRC_KC_LANES

typedef struct emu {
    rrc_backend table; /* first member: the backend handed out */
    rrc_emu_options o;
    uint64_t rng;
    void *scratch;
    size_t scratch_bytes;
    int pending;
    char pending_msg[256];
    int have_rep;
    rrc_gemm_exec_report rep;
} emu;

static emu *E(const rrc_backend *be) { return (emu *)be->impl; }

static uint64_t rng_next(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* A shuffled permutation of [0, n). */
static rrc_status perm(emu *e, int64_t n, int64_t **out) {
    int64_t *p = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    if (!p) return RRC_ERROR(RRC_ERR_OOM, "kc-emu: permutation of %lld items", (long long)n);
    for (int64_t i = 0; i < n; i++) p[i] = i;
    for (int64_t i = n - 1; i > 0; i--) {
        int64_t j = (int64_t)(rng_next(&e->rng) % (uint64_t)(i + 1));
        int64_t t = p[i];
        p[i] = p[j];
        p[j] = t;
    }
    *out = p;
    return RRC_OK;
}

static rrc_status range_error(emu *e, const char *op, int64_t index) {
    if (e->o.deferred_errors) {
        if (!e->pending) {
            e->pending = 1;
            snprintf(e->pending_msg, sizeof e->pending_msg, "%s: data-dependent index out of range (row/token %lld)", op,
                     (long long)index);
        }
        return RRC_OK;
    }
    return RRC_ERROR(RRC_ERR_RANGE, "%s: data-dependent index out of range (row/token %lld)", op, (long long)index);
}

static rrc_status check_pending(emu *e) {
    if (!e->pending) return RRC_OK;
    e->pending = 0;
    return RRC_ERROR(RRC_ERR_RANGE, "kc-emu (reported at sync): %s", e->pending_msg);
}

/* Run item(ctx, idx) for idx in [0, n) in shuffled order. item returns non-zero for a range error
 * and stores the offending row/token in *bad. */
typedef int (*item_fn)(const void *ctx, int64_t idx, int64_t *bad);
static rrc_status run_items(emu *e, int64_t n, item_fn f, const void *ctx, const char *op) {
    int64_t *p;
    RRC_CHECK(perm(e, n, &p));
    int64_t worst = -1;
    for (int64_t i = 0; i < n; i++) {
        int64_t bad = -1;
        if (f(ctx, p[i], &bad) && (worst < 0 || bad < worst)) worst = bad;
    }
    free(p);
    return worst >= 0 ? range_error(e, op, worst) : RRC_OK;
}

/* Phase 1 + 2 of a row reduction: lane partials for all (row, lane) items in shuffled order, then
 * the fixed tree per row. kind 0 = sum, 1 = max. */
typedef float (*lane_fn)(const void *ctx, int64_t row, int lane, const float *aux);
static rrc_status rows_reduce(emu *e, int64_t rows, lane_fn f, const void *ctx, const float *aux, int kind,
                              float *out) {
    float *part = (float *)malloc((size_t)rows * LANES * sizeof(float));
    int64_t *p = NULL;
    if (!part) return RRC_ERROR(RRC_ERR_OOM, "kc-emu: lane partials");
    rrc_status s = perm(e, rows * LANES, &p);
    if (s != RRC_OK) {
        free(part);
        return s;
    }
    for (int64_t i = 0; i < rows * LANES; i++) {
        int64_t row = p[i] / LANES;
        int lane = (int)(p[i] % LANES);
        part[p[i]] = f(ctx, row, lane, aux);
    }
    for (int64_t r = 0; r < rows; r++)
        out[r] = kind == 0 ? rrc_kc_tree_sum(part + r * LANES) : rrc_kc_tree_max(part + r * LANES);
    free(p);
    free(part);
    return RRC_OK;
}

static float *falloc(int64_t n) { return (float *)malloc((size_t)(n > 0 ? n : 1) * sizeof(float)); }

/* ---- memory ------------------------------------------------------------------------------ */

static rrc_status emu_alloc(const rrc_backend *be, size_t bytes, void **out) {
    (void)be;
    *out = NULL;
    if (bytes == 0) return RRC_OK;
    void *p = NULL;
    if (posix_memalign(&p, 256, (bytes + 255u) & ~(size_t)255u) != 0 || !p)
        return RRC_ERROR(RRC_ERR_OOM, "kc-emu alloc of %zu bytes", bytes);
    *out = p;
    return RRC_OK;
}
static void emu_free(const rrc_backend *be, void *ptr) {
    (void)be;
    free(ptr);
}
static rrc_status emu_memcpy(const rrc_backend *be, void *dst, const void *src, size_t bytes) {
    (void)be;
    if (bytes) memcpy(dst, src, bytes);
    return RRC_OK;
}
static rrc_status emu_memcpy_d2h(const rrc_backend *be, void *dst, const void *src, size_t bytes) {
    if (bytes) memcpy(dst, src, bytes);
    return check_pending(E(be));
}
static rrc_status emu_memset0(const rrc_backend *be, void *ptr, size_t bytes) {
    (void)be;
    if (bytes) memset(ptr, 0, bytes);
    return RRC_OK;
}
static rrc_status emu_sync(const rrc_backend *be) { return check_pending(E(be)); }

/* ---- gemm: the plan ladder with a simulated library matmul ------------------------------------ */

static int it_lt(const void *ctx, int64_t idx, int64_t *bad) {
    (void)bad;
    rrc_kc_item_lt_matmul((const rrc_kc_lt_p *)ctx, idx);
    return 0;
}
static int it_conv(const void *ctx, int64_t idx, int64_t *bad) {
    (void)bad;
    rrc_kc_item_conv2d((const rrc_kc_conv2d_p *)ctx, idx);
    return 0;
}
static rrc_status ex_scratch(void *ctx, size_t bytes, void **ptr) {
    emu *e = (emu *)ctx;
    if (bytes > e->scratch_bytes) {
        void *p = NULL;
        if (posix_memalign(&p, 256, bytes) != 0 || !p) return RRC_ERROR(RRC_ERR_OOM, "kc-emu gemm scratch %zu", bytes);
        free(e->scratch);
        e->scratch = p;
        e->scratch_bytes = bytes;
    }
    *ptr = e->scratch;
    return RRC_OK;
}
static int ex_supported(void *ctx, const rrc_lt_call *c) {
    emu *e = (emu *)ctx;
    unsigned bit = 4u * (c->p.adt == RRC_BF16) + 2u * (c->p.bdt == RRC_BF16) + (c->p.cdt == RRC_BF16);
    if (e->o.lt_reject_mask & (1u << bit)) return 0;
    if (c->align_a < e->o.lt_min_align || c->align_b < e->o.lt_min_align || c->align_c < e->o.lt_min_align) return 0;
    return 1;
}
static rrc_status ex_lt_run(void *ctx, const rrc_lt_call *c) {
    return run_items((emu *)ctx, c->p.batch * c->p.m * c->p.n, it_lt, &c->p, "gemm");
}
static rrc_status ex_kernel_run(void *ctx, const rrc_kc_lt_p *p) {
    return run_items((emu *)ctx, p->batch * p->m * p->n, it_lt, p, "gemm");
}
static rrc_status ex_convert(void *ctx, const rrc_kc_conv2d_p *p) {
    return run_items((emu *)ctx, p->batch * p->rows * p->cols, it_conv, p, "gemm convert");
}
static const rrc_gemm_exec_ops k_ex_ops = {ex_scratch, ex_supported, ex_lt_run, ex_kernel_run, ex_convert};

static rrc_status emu_gemm(const rrc_backend *be, const rrc_gemm_args *g) {
    emu *e = E(be);
    rrc_gemm_plan_opts po = {e->o.scratch_cap_elems, 0, 0};
    e->have_rep = 0;
    RRC_CHECK(rrc_gemm_execute(g, &po, e->o.gemm_min_level, &k_ex_ops, e, &e->rep));
    e->have_rep = 1;
    return RRC_OK;
}

/* ---- embedding / bias ------------------------------------------------------------------------- */

static int it_embed(const void *ctx, int64_t idx, int64_t *bad) {
    return rrc_kc_item_embedding((const rrc_kc_embed_p *)ctx, idx, bad);
}
static rrc_status emu_embedding_fwd(const rrc_backend *be, int64_t n_tokens, int64_t dim, int64_t vocab, rrc_buf ids,
                                    rrc_buf table, rrc_buf out) {
    rrc_kc_embed_p p = {n_tokens, dim, vocab, (const int32_t *)ids.ptr, table.ptr, table.dtype, out.ptr, out.dtype};
    return run_items(E(be), n_tokens * dim, it_embed, &p, "embedding_fwd");
}

static int it_add_bias(const void *ctx, int64_t idx, int64_t *bad) {
    (void)bad;
    rrc_kc_item_add_bias((const rrc_kc_bias_p *)ctx, idx);
    return 0;
}
static int it_bias_grad(const void *ctx, int64_t idx, int64_t *bad) {
    (void)bad;
    rrc_kc_item_bias_grad((const rrc_kc_bias_p *)ctx, idx);
    return 0;
}
static rrc_status emu_add_bias(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf x, int64_t ldx, rrc_buf bias) {
    rrc_kc_bias_p p = {rows, cols, ldx, x.ptr, x.dtype, bias.ptr, bias.dtype, 0};
    return run_items(E(be), rows * cols, it_add_bias, &p, "add_bias");
}
static rrc_status emu_bias_grad(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf dy, int64_t ldy,
                                rrc_buf dbias, int accumulate) {
    rrc_kc_bias_p p = {rows, cols, ldy, dy.ptr, dy.dtype, dbias.ptr, dbias.dtype, accumulate};
    return run_items(E(be), cols, it_bias_grad, &p, "bias_grad");
}

/* ---- rmsnorm ----------------------------------------------------------------------------------- */

typedef struct {
    const rrc_kc_rmsnorm_p *p;
    const float *rstd, *k;
} rms_ctx;

static float ln_sumsq(const void *ctx, int64_t row, int lane, const float *aux) {
    (void)aux;
    return rrc_kc_lane_rmsnorm_sumsq((const rrc_kc_rmsnorm_p *)ctx, row, lane);
}
static float ln_rdot(const void *ctx, int64_t row, int lane, const float *aux) {
    (void)aux;
    return rrc_kc_lane_rmsnorm_dot((const rrc_kc_rmsnorm_p *)ctx, row, lane);
}
static int it_rms_fwd(const void *ctx, int64_t idx, int64_t *bad) {
    (void)bad;
    const rms_ctx *c = (const rms_ctx *)ctx;
    int64_t row = idx / c->p->dim;
    rrc_kc_item_rmsnorm_fwd(c->p, row, idx % c->p->dim, c->rstd[row]);
    return 0;
}
static int it_rms_bwd(const void *ctx, int64_t idx, int64_t *bad) {
    (void)bad;
    const rms_ctx *c = (const rms_ctx *)ctx;
    int64_t row = idx / c->p->dim;
    rrc_kc_item_rmsnorm_bwd(c->p, row, idx % c->p->dim, c->rstd[row], c->k[row]);
    return 0;
}

static rrc_status emu_rmsnorm_fwd(const rrc_backend *be, int64_t rows, int64_t dim, int64_t ld, rrc_buf x, rrc_buf w,
                                  float eps, rrc_buf y) {
    rrc_kc_rmsnorm_p p = {rows, dim, ld, x.ptr, x.dtype, w.ptr, w.dtype, NULL, RRC_F32, y.ptr, y.dtype, eps, 0};
    float *rstd = falloc(rows);
    if (!rstd) return RRC_ERROR(RRC_ERR_OOM, "kc-emu rmsnorm");
    rrc_status s = rows_reduce(E(be), rows, ln_sumsq, &p, NULL, 0, rstd);
    if (s == RRC_OK) {
        for (int64_t r = 0; r < rows; r++) rstd[r] = rrc_kc_rmsnorm_finalize_rstd(&p, rstd[r]);
        rms_ctx c = {&p, rstd, NULL};
        s = run_items(E(be), rows * dim, it_rms_fwd, &c, "rmsnorm_fwd");
    }
    free(rstd);
    return s;
}

static rrc_status emu_rmsnorm_bwd(const rrc_backend *be, int64_t rows, int64_t dim, int64_t ld, rrc_buf x, rrc_buf w,
                                  float eps, rrc_buf dy, rrc_buf dx, int accumulate) {
    rrc_kc_rmsnorm_p p = {rows, dim, ld, x.ptr, x.dtype, w.ptr, w.dtype, dy.ptr, dy.dtype, dx.ptr, dx.dtype, eps,
                          accumulate};
    float *rstd = falloc(rows), *k = falloc(rows);
    rrc_status s = RRC_OK;
    if (!rstd || !k) s = RRC_ERROR(RRC_ERR_OOM, "kc-emu rmsnorm_bwd");
    if (s == RRC_OK) s = rows_reduce(E(be), rows, ln_sumsq, &p, NULL, 0, rstd);
    if (s == RRC_OK) s = rows_reduce(E(be), rows, ln_rdot, &p, NULL, 0, k);
    if (s == RRC_OK) {
        for (int64_t r = 0; r < rows; r++) {
            rstd[r] = rrc_kc_rmsnorm_finalize_rstd(&p, rstd[r]);
            k[r] = rrc_kc_rmsnorm_finalize_coef(&p, k[r], rstd[r]);
        }
        rms_ctx c = {&p, rstd, k};
        s = run_items(E(be), rows * dim, it_rms_bwd, &c, "rmsnorm_bwd");
    }
    free(rstd);
    free(k);
    return s;
}

/* ---- rope -------------------------------------------------------------------------------------- */

static int it_rope(const void *ctx, int64_t idx, int64_t *bad) { return rrc_kc_item_rope((const rrc_kc_rope_p *)ctx, idx, bad); }
static rrc_status emu_rope(const rrc_backend *be, int64_t n_tokens, int64_t n_heads, int64_t head_dim, int64_t ld,
                           rrc_buf x, rrc_buf pos, rrc_buf cos_tab, rrc_buf sin_tab, int64_t n_pos, rrc_buf y,
                           int backward) {
    rrc_kc_rope_p p = {n_tokens, n_heads, head_dim, ld, n_pos, x.ptr, x.dtype, y.ptr, y.dtype,
                       (const int32_t *)pos.ptr, (const float *)cos_tab.ptr, (const float *)sin_tab.ptr, backward};
    return run_items(E(be), n_tokens * n_heads * (head_dim / 2), it_rope, &p, backward ? "rope_bwd" : "rope_fwd");
}
static rrc_status emu_rope_fwd(const rrc_backend *be, int64_t n_tokens, int64_t n_heads, int64_t head_dim, int64_t ld,
                               rrc_buf x, rrc_buf pos, rrc_buf cos_tab, rrc_buf sin_tab, int64_t n_pos, rrc_buf y) {
    return emu_rope(be, n_tokens, n_heads, head_dim, ld, x, pos, cos_tab, sin_tab, n_pos, y, 0);
}
static rrc_status emu_rope_bwd(const rrc_backend *be, int64_t n_tokens, int64_t n_heads, int64_t head_dim, int64_t ld,
                               rrc_buf dy, rrc_buf pos, rrc_buf cos_tab, rrc_buf sin_tab, int64_t n_pos, rrc_buf dx) {
    return emu_rope(be, n_tokens, n_heads, head_dim, ld, dy, pos, cos_tab, sin_tab, n_pos, dx, 1);
}

/* ---- attention softmax ------------------------------------------------------------------------- */

typedef struct {
    const rrc_kc_softmax_p *p;
    const float *a, *b; /* fwd: m, inv_z; bwd: dot */
} sm_ctx;

static float ln_smax(const void *ctx, int64_t row, int lane, const float *aux) {
    (void)aux;
    return rrc_kc_lane_softmax_max((const rrc_kc_softmax_p *)ctx, row, lane);
}
static float ln_ssum(const void *ctx, int64_t row, int lane, const float *aux) {
    return rrc_kc_lane_softmax_sum((const rrc_kc_softmax_p *)ctx, row, lane, aux[row]);
}
static float ln_sdot(const void *ctx, int64_t row, int lane, const float *aux) {
    (void)aux;
    return rrc_kc_lane_softmax_dot((const rrc_kc_softmax_p *)ctx, row, lane);
}
static int it_sm_fwd(const void *ctx, int64_t idx, int64_t *bad) {
    (void)bad;
    const sm_ctx *c = (const sm_ctx *)ctx;
    int64_t row = idx / c->p->cols;
    rrc_kc_item_softmax_fwd(c->p, row, idx % c->p->cols, c->a[row], c->b[row]);
    return 0;
}
static int it_sm_bwd(const void *ctx, int64_t idx, int64_t *bad) {
    (void)bad;
    const sm_ctx *c = (const sm_ctx *)ctx;
    int64_t row = idx / c->p->cols;
    rrc_kc_item_softmax_bwd(c->p, row, idx % c->p->cols, c->a[row]);
    return 0;
}

static rrc_status emu_attn_softmax_fwd(const rrc_backend *be, int64_t batch, int64_t rows, int64_t cols, int64_t ld,
                                       int64_t batch_stride, rrc_buf s, float scale, int64_t row_offset, int causal,
                                       rrc_buf doc_ids, rrc_buf p) {
    rrc_kc_softmax_p q = {batch, rows, cols, ld, batch_stride, row_offset, s.ptr, s.dtype, NULL, RRC_F32, p.ptr,
                          p.dtype, scale, causal, rrc_buf_is_none(doc_ids) ? NULL : (const int32_t *)doc_ids.ptr};
    const int64_t nr = batch * rows;
    float *m = falloc(nr), *iz = falloc(nr);
    rrc_status st = RRC_OK;
    if (!m || !iz) st = RRC_ERROR(RRC_ERR_OOM, "kc-emu softmax");
    if (st == RRC_OK) st = rows_reduce(E(be), nr, ln_smax, &q, NULL, 1, m);
    if (st == RRC_OK) st = rows_reduce(E(be), nr, ln_ssum, &q, m, 0, iz);
    if (st == RRC_OK) {
        for (int64_t r = 0; r < nr; r++) iz[r] = m[r] == -INFINITY ? 0.0f : 1.0f / iz[r];
        sm_ctx c = {&q, m, iz};
        st = run_items(E(be), nr * cols, it_sm_fwd, &c, "attn_softmax_fwd");
    }
    free(m);
    free(iz);
    return st;
}

static rrc_status emu_attn_softmax_bwd(const rrc_backend *be, int64_t batch, int64_t rows, int64_t cols, int64_t ld,
                                       int64_t batch_stride, rrc_buf p, rrc_buf dp, float scale, rrc_buf ds) {
    rrc_kc_softmax_p q = {batch, rows, cols, ld, batch_stride, 0, p.ptr, p.dtype, dp.ptr, dp.dtype, ds.ptr, ds.dtype,
                          scale, 0, NULL};
    const int64_t nr = batch * rows;
    float *dot = falloc(nr);
    if (!dot) return RRC_ERROR(RRC_ERR_OOM, "kc-emu softmax_bwd");
    rrc_status st = rows_reduce(E(be), nr, ln_sdot, &q, NULL, 0, dot);
    if (st == RRC_OK) {
        sm_ctx c = {&q, dot, NULL};
        st = run_items(E(be), nr * cols, it_sm_bwd, &c, "attn_softmax_bwd");
    }
    free(dot);
    return st;
}

/* ---- swiglu / elementwise ---------------------------------------------------------------------- */

static int it_swf(const void *ctx, int64_t idx, int64_t *bad) {
    (void)bad;
    rrc_kc_item_swiglu_fwd((const rrc_kc_swiglu_p *)ctx, idx);
    return 0;
}
static int it_swb(const void *ctx, int64_t idx, int64_t *bad) {
    (void)bad;
    rrc_kc_item_swiglu_bwd((const rrc_kc_swiglu_p *)ctx, idx);
    return 0;
}
static rrc_status emu_swiglu_fwd(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf gate, rrc_buf up,
                                 int64_t ld_in, rrc_buf h, int64_t ld_out) {
    rrc_kc_swiglu_p p = {rows, cols, ld_in, 0, ld_out, gate.ptr, gate.dtype, up.ptr, up.dtype, NULL, RRC_F32,
                         h.ptr, h.dtype, NULL, RRC_F32};
    return run_items(E(be), rows * cols, it_swf, &p, "swiglu_fwd");
}
static rrc_status emu_swiglu_bwd(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf gate, rrc_buf up,
                                 int64_t ld_in, rrc_buf dh, int64_t ld_dh, rrc_buf dgate, rrc_buf dup, int64_t ld_out) {
    rrc_kc_swiglu_p p = {rows, cols, ld_in, ld_dh, ld_out, gate.ptr, gate.dtype, up.ptr, up.dtype, dh.ptr, dh.dtype,
                         dgate.ptr, dgate.dtype, dup.ptr, dup.dtype};
    return run_items(E(be), rows * cols, it_swb, &p, "swiglu_bwd");
}

static int it_add(const void *ctx, int64_t i, int64_t *bad) {
    (void)bad;
    rrc_kc_item_add((const rrc_kc_ew_p *)ctx, i);
    return 0;
}
static int it_axpy(const void *ctx, int64_t i, int64_t *bad) {
    (void)bad;
    rrc_kc_item_axpy((const rrc_kc_ew_p *)ctx, i);
    return 0;
}
static int it_scale(const void *ctx, int64_t i, int64_t *bad) {
    (void)bad;
    rrc_kc_item_scale((const rrc_kc_ew_p *)ctx, i);
    return 0;
}
static int it_convert(const void *ctx, int64_t i, int64_t *bad) {
    (void)bad;
    rrc_kc_item_convert((const rrc_kc_ew_p *)ctx, i);
    return 0;
}
static rrc_status emu_add(const rrc_backend *be, int64_t n, rrc_buf a, rrc_buf b, rrc_buf out) {
    rrc_kc_ew_p p = {n, 0.0f, a.ptr, a.dtype, b.ptr, b.dtype, out.ptr, out.dtype};
    return run_items(E(be), n, it_add, &p, "add");
}
static rrc_status emu_axpy(const rrc_backend *be, int64_t n, float alpha, rrc_buf x, rrc_buf y) {
    rrc_kc_ew_p p = {n, alpha, x.ptr, x.dtype, NULL, RRC_F32, y.ptr, y.dtype};
    return run_items(E(be), n, it_axpy, &p, "axpy");
}
static rrc_status emu_scale(const rrc_backend *be, int64_t n, float alpha, rrc_buf x) {
    rrc_kc_ew_p p = {n, alpha, NULL, RRC_F32, NULL, RRC_F32, x.ptr, x.dtype};
    return run_items(E(be), n, it_scale, &p, "scale");
}
static rrc_status emu_convert(const rrc_backend *be, int64_t n, rrc_buf src, rrc_buf dst) {
    if (src.dtype == dst.dtype) { /* the device uses cudaMemcpyAsync (also for I32) */
        if (src.ptr != dst.ptr) memcpy(dst.ptr, src.ptr, (size_t)n * rrc_dtype_size(src.dtype));
        return RRC_OK;
    }
    rrc_kc_ew_p p = {n, 0.0f, src.ptr, src.dtype, NULL, RRC_F32, dst.ptr, dst.dtype};
    return run_items(E(be), n, it_convert, &p, "convert");
}

/* ---- sum_squares -------------------------------------------------------------------------------- */

static float ln_ss(const void *ctx, int64_t row, int lane, const float *aux) {
    (void)aux;
    return rrc_kc_lane_ss_chunk((const rrc_kc_ss_p *)ctx, row, lane);
}
static float ln_ss2(const void *ctx, int64_t row, int lane, const float *aux) {
    (void)aux;
    (void)row;
    return rrc_kc_lane_ss_parts((const rrc_kc_ss_p *)ctx, lane);
}
static rrc_status emu_sum_squares(const rrc_backend *be, int64_t n, rrc_buf x, rrc_buf out, int accumulate) {
    const int64_t nc = rrc_kc_ss_chunks(n);
    float *part = falloc(nc);
    if (!part) return RRC_ERROR(RRC_ERR_OOM, "kc-emu sum_squares");
    rrc_kc_ss_p p = {n, x.ptr, x.dtype, part, (float *)out.ptr, accumulate};
    rrc_status s = rows_reduce(E(be), nc, ln_ss, &p, NULL, 0, part);
    float total = 0.0f;
    if (s == RRC_OK) s = rows_reduce(E(be), 1, ln_ss2, &p, NULL, 0, &total);
    if (s == RRC_OK) rrc_kc_ss_finalize(&p, total);
    free(part);
    return s;
}

/* ---- cross entropy -------------------------------------------------------------------------------- */

typedef struct {
    const rrc_kc_xent_p *p;
    const rrc_kc_xent_stats *st;
} xe_ctx;
static int it_xent(const void *ctx, int64_t idx, int64_t *bad) {
    (void)bad;
    const xe_ctx *c = (const xe_ctx *)ctx;
    int64_t row = idx / c->p->vocab;
    rrc_kc_item_xent_grad(c->p, row, idx % c->p->vocab, &c->st[row]);
    return 0;
}

static rrc_status emu_cross_entropy(const rrc_backend *be, int64_t rows, int64_t vocab, rrc_buf logits, int64_t ld,
                                    rrc_buf labels, rrc_buf loss, float grad_scale, rrc_buf row_scale, int write_grad) {
    emu *e = E(be);
    rrc_kc_xent_p p = {rows, vocab, ld, logits.ptr, logits.dtype, (const int32_t *)labels.ptr, (float *)loss.ptr,
                       rrc_buf_is_none(row_scale) ? NULL : (const float *)row_scale.ptr, grad_scale, write_grad};
    rrc_kc_lse *part = (rrc_kc_lse *)malloc((size_t)rows * LANES * sizeof(rrc_kc_lse));
    rrc_kc_xent_stats *st = (rrc_kc_xent_stats *)malloc((size_t)rows * sizeof(rrc_kc_xent_stats));
    int64_t *pm = NULL;
    rrc_status s = RRC_OK;
    if (!part || !st) s = RRC_ERROR(RRC_ERR_OOM, "kc-emu cross entropy");
    if (s == RRC_OK) s = perm(e, rows * LANES, &pm);
    int64_t worst = -1;
    if (s == RRC_OK) {
        for (int64_t i = 0; i < rows * LANES; i++)
            part[pm[i]] = rrc_kc_lane_xent(&p, pm[i] / LANES, (int)(pm[i] % LANES));
        free(pm);
        pm = NULL;
        s = perm(e, rows, &pm);
    }
    if (s == RRC_OK) {
        for (int64_t i = 0; i < rows; i++) {
            int64_t r = pm[i];
            rrc_kc_lse a = rrc_kc_tree_lse(part + r * LANES);
            if (rrc_kc_xent_finalize(&p, r, a, &st[r]) && (worst < 0 || r < worst)) worst = r;
        }
        if (write_grad) {
            xe_ctx c = {&p, st};
            s = run_items(e, rows * vocab, it_xent, &c, "cross_entropy_fwd_bwd");
        }
    }
    free(pm);
    free(part);
    free(st);
    if (s == RRC_OK && worst >= 0) s = range_error(e, "cross_entropy_fwd_bwd", worst);
    return s;
}

/* ---- adamw ------------------------------------------------------------------------------------------ */

static int it_adamw(const void *ctx, int64_t i, int64_t *bad) {
    (void)bad;
    rrc_kc_item_adamw((const rrc_kc_adamw_p *)ctx, i);
    return 0;
}
static rrc_status emu_adamw_step(const rrc_backend *be, int64_t n, rrc_buf param, rrc_buf grad, rrc_buf m, rrc_buf v,
                                 const rrc_adamw_hparams *h) {
    rrc_kc_adamw_p p = {n, (float *)param.ptr, (const float *)grad.ptr, (float *)m.ptr, (float *)v.ptr,
                        rrc_kc_adamw_prepare(h->lr, h->beta1, h->beta2, h->eps, h->weight_decay, h->step, h->grad_scale)};
    return run_items(E(be), n, it_adamw, &p, "adamw_step");
}

/* ---- construction ------------------------------------------------------------------------------------- */

void rrc_emu_options_default(rrc_emu_options *o) {
    memset(o, 0, sizeof *o);
    o->seed = 0x5EEDull;
    o->gemm_min_level = RRC_GEMM_LEVEL_NATIVE;
}

static const char k_emu_name[] = "kc-emu";

rrc_status rrc_emu_backend_create(const rrc_emu_options *o, const rrc_backend **out) {
    RRC_REQUIRE(out, RRC_ERR_INVALID_ARG, "kc-emu: out is NULL");
    *out = NULL;
    emu *e = (emu *)calloc(1, sizeof *e);
    if (!e) return RRC_ERROR(RRC_ERR_OOM, "kc-emu backend");
    if (o) e->o = *o;
    else rrc_emu_options_default(&e->o);
    e->rng = e->o.seed;
    rrc_backend *t = &e->table;
    t->name = k_emu_name;
    t->host_accessible = 1;
    t->impl = e;
    t->alloc = emu_alloc;
    t->free = emu_free;
    t->memcpy_h2d = emu_memcpy;
    t->memcpy_d2h = emu_memcpy_d2h;
    t->memcpy_d2d = emu_memcpy;
    t->memset0 = emu_memset0;
    t->sync = emu_sync;
    t->gemm = emu_gemm;
    t->embedding_fwd = emu_embedding_fwd;
    t->add_bias = emu_add_bias;
    t->bias_grad = emu_bias_grad;
    t->rmsnorm_fwd = emu_rmsnorm_fwd;
    t->rmsnorm_bwd = emu_rmsnorm_bwd;
    t->rope_fwd = emu_rope_fwd;
    t->rope_bwd = emu_rope_bwd;
    t->attn_softmax_fwd = emu_attn_softmax_fwd;
    t->attn_softmax_bwd = emu_attn_softmax_bwd;
    t->swiglu_fwd = emu_swiglu_fwd;
    t->swiglu_bwd = emu_swiglu_bwd;
    t->add = emu_add;
    t->axpy = emu_axpy;
    t->scale = emu_scale;
    t->convert = emu_convert;
    t->sum_squares = emu_sum_squares;
    t->cross_entropy_fwd_bwd = emu_cross_entropy;
    t->adamw_step = emu_adamw_step;
    *out = t;
    return RRC_OK;
}

int rrc_backend_is_emu(const rrc_backend *be) { return be && be->name == k_emu_name; }

void rrc_emu_backend_destroy(const rrc_backend *be) {
    if (!rrc_backend_is_emu(be)) return;
    emu *e = E(be);
    free(e->scratch);
    free(e);
}

int rrc_emu_last_gemm(const rrc_backend *be, rrc_gemm_exec_report *rep) {
    if (!rrc_backend_is_emu(be) || !E(be)->have_rep) return 0;
    if (rep) *rep = E(be)->rep;
    return 1;
}
