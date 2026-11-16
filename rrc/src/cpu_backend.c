/* rrc/src/cpu_backend.c — CPU reference backend.
 *
 * fp32 math through the shared kernels_common functions; reductions accumulate in double
 * (rrc_backend_cpu) or float (rrc_backend_cpu_f32acc). Every output element / row is computed by
 * exactly one thread with a fixed reduction order, so results are bit-identical for any
 * OMP_NUM_THREADS. Preconditions (shapes, dtypes, aliasing) were checked by src/ops.c; this file
 * only checks data-dependent ranges (token ids, labels, position ids). */
#include <stdlib.h>
#include <string.h>

#include "rrc/backend.h"
#include "rrc_kc.h"

typedef struct cpu_impl {
    int acc64;
} cpu_impl;

static int acc64_of(const rrc_backend *be) { return ((const cpu_impl *)be->impl)->acc64; }

/* Only parallelise loops with enough work to amortise the fork/join. */
#define PAR_MIN 16384

/* ---- memory ------------------------------------------------------------------------------ */

static rrc_status cpu_alloc(const rrc_backend *be, size_t bytes, void **out) {
    (void)be;
    *out = NULL;
    if (bytes == 0) return RRC_OK;
    size_t rounded = (bytes + 63u) & ~(size_t)63u;
    void *p = NULL;
    if (posix_memalign(&p, 64, rounded) != 0 || !p) return RRC_ERROR(RRC_ERR_OOM, "cpu alloc of %zu bytes", bytes);
    *out = p;
    return RRC_OK;
}
static void cpu_free(const rrc_backend *be, void *ptr) {
    (void)be;
    free(ptr);
}
static rrc_status cpu_memcpy(const rrc_backend *be, void *dst, const void *src, size_t bytes) {
    (void)be;
    if (bytes) memcpy(dst, src, bytes);
    return RRC_OK;
}
static rrc_status cpu_memset0(const rrc_backend *be, void *ptr, size_t bytes) {
    (void)be;
    if (bytes) memset(ptr, 0, bytes);
    return RRC_OK;
}
static rrc_status cpu_sync(const rrc_backend *be) {
    (void)be;
    return RRC_OK;
}

/* ---- gemm ---------------------------------------------------------------------------------- */

/* Pack rows [r0, r1) of op(X) (rows x cols in the op's orientation) of a stored matrix into a dense
 * float array with row stride `cols`. trans == T means the stored matrix is cols x rows. */
static void pack_rows(const void *base, rrc_dtype dt, int64_t ld, rrc_trans trans, int64_t r0, int64_t r1, int64_t cols,
                      float *dst) {
    for (int64_t r = r0; r < r1; r++)
        for (int64_t c = 0; c < cols; c++)
            dst[(r - r0) * cols + c] = trans == RRC_OP_N ? rrc_kc_ld(base, dt, r * ld + c) : rrc_kc_ld(base, dt, c * ld + r);
}

/* Output tiles of TB x TB; each thread packs the TB rows of op(A) and the TB columns of op(B) of its
 * current tile into its own scratch (2 * TB * k floats per thread), so the scratch does not grow with
 * m or n. Every dot product runs over p = 0..k-1 in the same order whatever the tiling or thread
 * count. ops.c never dispatches k == 0. */
#define GEMM_TB 32

static rrc_status cpu_gemm(const rrc_backend *be, const rrc_gemm_args *g) {
    const int acc64 = acc64_of(be);
    const int64_t m = g->m, n = g->n, k = g->k;
    const size_t ea = rrc_dtype_size(g->a.dtype), eb = rrc_dtype_size(g->b.dtype), ec = rrc_dtype_size(g->c.dtype);
    const int64_t TB = GEMM_TB, ti = (m + TB - 1) / TB, tj = (n + TB - 1) / TB;
    const float alpha = g->alpha, beta = g->beta;
    const rrc_dtype cdt = g->c.dtype;
    const int64_t ldc = g->ldc;
    /* op(B) is k x n; we pack its transpose (n x k) so that each dot product is contiguous:
     * op(B)^T = B^T when trans_b == N (B stored k x n), = B when trans_b == T (B stored n x k). */
    const rrc_trans bt_trans = g->trans_b == RRC_OP_N ? RRC_OP_T : RRC_OP_N;
    int fail = 0;
#pragma omp parallel if (m * n * (k + 1) > PAR_MIN)
    {
        float *ap = NULL, *bp = NULL;
        if (k > 0) {
            ap = (float *)malloc((size_t)(TB * k) * sizeof(float));
            bp = (float *)malloc((size_t)(TB * k) * sizeof(float));
            if (!ap || !bp) {
#pragma omp atomic write
                fail = 1;
            }
        }
#pragma omp barrier
        int failed;
#pragma omp atomic read
        failed = fail;
        if (!failed)
            for (int64_t b = 0; b < g->batch; b++) {
                const char *abase = (const char *)g->a.ptr + (size_t)(b * g->stride_a) * ea;
                const char *bbase = (const char *)g->b.ptr + (size_t)(b * g->stride_b) * eb;
                char *cbase = (char *)g->c.ptr + (size_t)(b * g->stride_c) * ec;
                int64_t packed_i0 = -1;
#pragma omp for schedule(static)
                for (int64_t t = 0; t < ti * tj; t++) {
                    const int64_t i0 = (t / tj) * TB, j0 = (t % tj) * TB;
                    const int64_t i1 = i0 + TB < m ? i0 + TB : m, j1 = j0 + TB < n ? j0 + TB : n;
                    if (k > 0) {
                        if (i0 != packed_i0) pack_rows(abase, g->a.dtype, g->lda, g->trans_a, i0, i1, k, ap);
                        packed_i0 = i0;
                        pack_rows(bbase, g->b.dtype, g->ldb, bt_trans, j0, j1, k, bp);
                    }
                    for (int64_t i = i0; i < i1; i++) {
                        const float *arow = ap ? ap + (i - i0) * k : NULL;
                        for (int64_t j = j0; j < j1; j++) {
                            float dot = 0.0f;
                            if (k > 0) {
                                const float *brow = bp + (j - j0) * k;
                                if (acc64) {
                                    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                                    int64_t p = 0;
                                    for (; p + 4 <= k; p += 4) {
                                        s0 += (double)arow[p] * brow[p];
                                        s1 += (double)arow[p + 1] * brow[p + 1];
                                        s2 += (double)arow[p + 2] * brow[p + 2];
                                        s3 += (double)arow[p + 3] * brow[p + 3];
                                    }
                                    for (; p < k; p++) s0 += (double)arow[p] * brow[p];
                                    dot = (float)((s0 + s1) + (s2 + s3));
                                } else {
                                    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                                    int64_t p = 0;
                                    for (; p + 4 <= k; p += 4) {
                                        s0 += arow[p] * brow[p];
                                        s1 += arow[p + 1] * brow[p + 1];
                                        s2 += arow[p + 2] * brow[p + 2];
                                        s3 += arow[p + 3] * brow[p + 3];
                                    }
                                    for (; p < k; p++) s0 += arow[p] * brow[p];
                                    dot = (s0 + s1) + (s2 + s3);
                                }
                            }
                            float r = alpha * dot;
                            if (beta != 0.0f) r += beta * rrc_kc_ld(cbase, cdt, i * ldc + j);
                            rrc_kc_st(cbase, cdt, i * ldc + j, r);
                        }
                    }
                }
            }
        free(ap);
        free(bp);
    }
    if (fail) return RRC_ERROR(RRC_ERR_OOM, "cpu gemm: packing scratch (2 x %lld floats per thread)", (long long)(TB * k));
    return RRC_OK;
}

/* ---- embedding / bias ------------------------------------------------------------------------ */

static rrc_status cpu_embedding_fwd(const rrc_backend *be, int64_t n_tokens, int64_t dim, int64_t vocab, rrc_buf ids,
                                    rrc_buf table, rrc_buf out) {
    (void)be;
    const int32_t *id = (const int32_t *)ids.ptr;
    for (int64_t t = 0; t < n_tokens; t++)
        if (id[t] < 0 || id[t] >= vocab)
            return RRC_ERROR(RRC_ERR_RANGE, "embedding_fwd: ids[%lld] = %d outside [0, %lld)", (long long)t, id[t],
                             (long long)vocab);
#pragma omp parallel for schedule(static) if (n_tokens * dim > PAR_MIN)
    for (int64_t t = 0; t < n_tokens; t++)
        for (int64_t c = 0; c < dim; c++)
            rrc_kc_st(out.ptr, out.dtype, t * dim + c, rrc_kc_ld(table.ptr, table.dtype, (int64_t)id[t] * dim + c));
    return RRC_OK;
}

static rrc_status cpu_add_bias(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf x, int64_t ldx, rrc_buf bias) {
    (void)be;
#pragma omp parallel for schedule(static) if (rows * cols > PAR_MIN)
    for (int64_t r = 0; r < rows; r++)
        for (int64_t c = 0; c < cols; c++)
            rrc_kc_st(x.ptr, x.dtype, r * ldx + c,
                      rrc_kc_ld(x.ptr, x.dtype, r * ldx + c) + rrc_kc_ld(bias.ptr, bias.dtype, c));
    return RRC_OK;
}

static rrc_status cpu_bias_grad(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf dy, int64_t ldy,
                                rrc_buf dbias, int accumulate) {
    const int acc64 = acc64_of(be);
    float *db = (float *)dbias.ptr;
#pragma omp parallel for schedule(static) if (rows * cols > PAR_MIN)
    for (int64_t c = 0; c < cols; c++) {
        float s;
        if (acc64) {
            double a = 0.0;
            for (int64_t r = 0; r < rows; r++) a += (double)rrc_kc_ld(dy.ptr, dy.dtype, r * ldy + c);
            s = (float)a;
        } else {
            float a = 0.0f;
            for (int64_t r = 0; r < rows; r++) a += rrc_kc_ld(dy.ptr, dy.dtype, r * ldy + c);
            s = a;
        }
        db[c] = accumulate ? db[c] + s : s;
    }
    return RRC_OK;
}

/* ---- rmsnorm --------------------------------------------------------------------------------- */

static const void *row_c(rrc_buf b, int64_t off) { return (const char *)b.ptr + (size_t)off * rrc_dtype_size(b.dtype); }
static void *row_m(rrc_buf b, int64_t off) { return (char *)b.ptr + (size_t)off * rrc_dtype_size(b.dtype); }

static rrc_status cpu_rmsnorm_fwd(const rrc_backend *be, int64_t rows, int64_t dim, int64_t ld, rrc_buf x, rrc_buf w,
                                  float eps, rrc_buf y) {
    const int acc64 = acc64_of(be);
#pragma omp parallel for schedule(static) if (rows * dim > PAR_MIN)
    for (int64_t r = 0; r < rows; r++)
        rrc_kc_rmsnorm_row_fwd(row_c(x, r * ld), x.dtype, w.ptr, w.dtype, row_m(y, r * ld), y.dtype, dim, eps, acc64);
    return RRC_OK;
}

static rrc_status cpu_rmsnorm_bwd(const rrc_backend *be, int64_t rows, int64_t dim, int64_t ld, rrc_buf x, rrc_buf w,
                                  float eps, rrc_buf dy, rrc_buf dx, int accumulate) {
    const int acc64 = acc64_of(be);
#pragma omp parallel for schedule(static) if (rows * dim > PAR_MIN)
    for (int64_t r = 0; r < rows; r++)
        rrc_kc_rmsnorm_row_bwd(row_c(x, r * ld), x.dtype, w.ptr, w.dtype, row_c(dy, r * ld), dy.dtype,
                               row_m(dx, r * ld), dx.dtype, dim, eps, accumulate, acc64);
    return RRC_OK;
}

/* ---- rope ------------------------------------------------------------------------------------- */

static rrc_status cpu_rope(int64_t n_tokens, int64_t n_heads, int64_t head_dim, int64_t ld, rrc_buf x, rrc_buf pos,
                           rrc_buf cos_tab, rrc_buf sin_tab, int64_t n_pos, rrc_buf y, int backward) {
    const int32_t *p = (const int32_t *)pos.ptr;
    for (int64_t t = 0; t < n_tokens; t++)
        if (p[t] < 0 || p[t] >= n_pos)
            return RRC_ERROR(RRC_ERR_RANGE, "%s: pos[%lld] = %d outside the table [0, %lld)",
                             backward ? "rope_bwd" : "rope_fwd", (long long)t, p[t], (long long)n_pos);
    const int64_t half = head_dim / 2;
    const float *ct = (const float *)cos_tab.ptr, *st = (const float *)sin_tab.ptr;
#pragma omp parallel for schedule(static) if (n_tokens * n_heads * head_dim > PAR_MIN)
    for (int64_t t = 0; t < n_tokens; t++)
        for (int64_t h = 0; h < n_heads; h++) {
            int64_t off = t * ld + h * head_dim;
            rrc_kc_rope_head(row_c(x, off), x.dtype, row_m(y, off), y.dtype, head_dim, ct + (int64_t)p[t] * half,
                             st + (int64_t)p[t] * half, backward);
        }
    return RRC_OK;
}

static rrc_status cpu_rope_fwd(const rrc_backend *be, int64_t n_tokens, int64_t n_heads, int64_t head_dim, int64_t ld,
                               rrc_buf x, rrc_buf pos, rrc_buf cos_tab, rrc_buf sin_tab, int64_t n_pos, rrc_buf y) {
    (void)be;
    return cpu_rope(n_tokens, n_heads, head_dim, ld, x, pos, cos_tab, sin_tab, n_pos, y, 0);
}
static rrc_status cpu_rope_bwd(const rrc_backend *be, int64_t n_tokens, int64_t n_heads, int64_t head_dim, int64_t ld,
                               rrc_buf dy, rrc_buf pos, rrc_buf cos_tab, rrc_buf sin_tab, int64_t n_pos, rrc_buf dx) {
    (void)be;
    return cpu_rope(n_tokens, n_heads, head_dim, ld, dy, pos, cos_tab, sin_tab, n_pos, dx, 1);
}

/* ---- attention softmax ------------------------------------------------------------------------- */

static rrc_status cpu_attn_softmax_fwd(const rrc_backend *be, int64_t batch, int64_t rows, int64_t cols, int64_t ld,
                                       int64_t batch_stride, rrc_buf s, float scale, int64_t row_offset, int causal,
                                       rrc_buf doc_ids, rrc_buf p) {
    const int acc64 = acc64_of(be);
    const int32_t *doc = rrc_buf_is_none(doc_ids) ? NULL : (const int32_t *)doc_ids.ptr;
#pragma omp parallel for schedule(static) if (batch * rows * cols > PAR_MIN)
    for (int64_t br = 0; br < batch * rows; br++) {
        int64_t b = br / rows, r = br % rows;
        int64_t off = b * batch_stride + r * ld;
        rrc_kc_attn_softmax_row_fwd(row_c(s, off), s.dtype, row_m(p, off), p.dtype, cols, scale, row_offset + r, causal,
                                    doc, acc64);
    }
    return RRC_OK;
}

static rrc_status cpu_attn_softmax_bwd(const rrc_backend *be, int64_t batch, int64_t rows, int64_t cols, int64_t ld,
                                       int64_t batch_stride, rrc_buf p, rrc_buf dp, float scale, rrc_buf ds) {
    const int acc64 = acc64_of(be);
#pragma omp parallel for schedule(static) if (batch * rows * cols > PAR_MIN)
    for (int64_t br = 0; br < batch * rows; br++) {
        int64_t b = br / rows, r = br % rows;
        int64_t off = b * batch_stride + r * ld;
        rrc_kc_attn_softmax_row_bwd(row_c(p, off), p.dtype, row_c(dp, off), dp.dtype, row_m(ds, off), ds.dtype, cols,
                                    scale, acc64);
    }
    return RRC_OK;
}

/* ---- swiglu ------------------------------------------------------------------------------------ */

static rrc_status cpu_swiglu_fwd(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf gate, rrc_buf up,
                                 int64_t ld_in, rrc_buf h, int64_t ld_out) {
    (void)be;
#pragma omp parallel for schedule(static) if (rows * cols > PAR_MIN)
    for (int64_t r = 0; r < rows; r++)
        for (int64_t c = 0; c < cols; c++) {
            float g = rrc_kc_ld(gate.ptr, gate.dtype, r * ld_in + c), u = rrc_kc_ld(up.ptr, up.dtype, r * ld_in + c);
            rrc_kc_st(h.ptr, h.dtype, r * ld_out + c, rrc_kc_swiglu_fwd(g, u));
        }
    return RRC_OK;
}

static rrc_status cpu_swiglu_bwd(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf gate, rrc_buf up,
                                 int64_t ld_in, rrc_buf dh, int64_t ld_dh, rrc_buf dgate, rrc_buf dup, int64_t ld_out) {
    (void)be;
#pragma omp parallel for schedule(static) if (rows * cols > PAR_MIN)
    for (int64_t r = 0; r < rows; r++)
        for (int64_t c = 0; c < cols; c++) {
            float g = rrc_kc_ld(gate.ptr, gate.dtype, r * ld_in + c), u = rrc_kc_ld(up.ptr, up.dtype, r * ld_in + c);
            float d = rrc_kc_ld(dh.ptr, dh.dtype, r * ld_dh + c);
            float dg, du;
            rrc_kc_swiglu_bwd(g, u, d, &dg, &du);
            rrc_kc_st(dgate.ptr, dgate.dtype, r * ld_out + c, dg);
            rrc_kc_st(dup.ptr, dup.dtype, r * ld_out + c, du);
        }
    return RRC_OK;
}

/* ---- elementwise -------------------------------------------------------------------------------- */

static rrc_status cpu_add(const rrc_backend *be, int64_t n, rrc_buf a, rrc_buf b, rrc_buf out) {
    (void)be;
#pragma omp parallel for schedule(static) if (n > PAR_MIN)
    for (int64_t i = 0; i < n; i++)
        rrc_kc_st(out.ptr, out.dtype, i, rrc_kc_ld(a.ptr, a.dtype, i) + rrc_kc_ld(b.ptr, b.dtype, i));
    return RRC_OK;
}

static rrc_status cpu_axpy(const rrc_backend *be, int64_t n, float alpha, rrc_buf x, rrc_buf y) {
    (void)be;
#pragma omp parallel for schedule(static) if (n > PAR_MIN)
    for (int64_t i = 0; i < n; i++)
        rrc_kc_st(y.ptr, y.dtype, i, rrc_kc_ld(y.ptr, y.dtype, i) + alpha * rrc_kc_ld(x.ptr, x.dtype, i));
    return RRC_OK;
}

static rrc_status cpu_scale(const rrc_backend *be, int64_t n, float alpha, rrc_buf x) {
    (void)be;
#pragma omp parallel for schedule(static) if (n > PAR_MIN)
    for (int64_t i = 0; i < n; i++) rrc_kc_st(x.ptr, x.dtype, i, alpha * rrc_kc_ld(x.ptr, x.dtype, i));
    return RRC_OK;
}

static rrc_status cpu_convert(const rrc_backend *be, int64_t n, rrc_buf src, rrc_buf dst) {
    (void)be;
    if (src.dtype == dst.dtype) {
        memcpy(dst.ptr, src.ptr, (size_t)n * rrc_dtype_size(src.dtype));
        return RRC_OK;
    }
#pragma omp parallel for schedule(static) if (n > PAR_MIN)
    for (int64_t i = 0; i < n; i++) rrc_kc_st(dst.ptr, dst.dtype, i, rrc_kc_ld(src.ptr, src.dtype, i));
    return RRC_OK;
}

#define SS_BLOCK 4096
static rrc_status cpu_sum_squares(const rrc_backend *be, int64_t n, rrc_buf x, rrc_buf out, int accumulate) {
    const int acc64 = acc64_of(be);
    const int64_t nb = (n + SS_BLOCK - 1) / SS_BLOCK;
    double *part = (double *)malloc((size_t)(nb > 0 ? nb : 1) * sizeof(double));
    if (!part) return RRC_ERROR(RRC_ERR_OOM, "sum_squares partials");
#pragma omp parallel for schedule(static) if (n > PAR_MIN)
    for (int64_t b = 0; b < nb; b++) {
        int64_t i0 = b * SS_BLOCK, i1 = i0 + SS_BLOCK < n ? i0 + SS_BLOCK : n;
        if (acc64) {
            double s = 0.0;
            for (int64_t i = i0; i < i1; i++) {
                double v = rrc_kc_ld(x.ptr, x.dtype, i);
                s += v * v;
            }
            part[b] = s;
        } else {
            float s = 0.0f;
            for (int64_t i = i0; i < i1; i++) {
                float v = rrc_kc_ld(x.ptr, x.dtype, i);
                s += v * v;
            }
            part[b] = s;
        }
    }
    float total;
    if (acc64) {
        double s = 0.0;
        for (int64_t b = 0; b < nb; b++) s += part[b];
        total = (float)s;
    } else {
        float s = 0.0f;
        for (int64_t b = 0; b < nb; b++) s += (float)part[b];
        total = s;
    }
    free(part);
    float *o = (float *)out.ptr;
    o[0] = accumulate ? o[0] + total : total;
    return RRC_OK;
}

/* ---- cross entropy ------------------------------------------------------------------------------- */

static rrc_status cpu_cross_entropy(const rrc_backend *be, int64_t rows, int64_t vocab, rrc_buf logits, int64_t ld,
                                    rrc_buf labels, rrc_buf loss, float grad_scale, rrc_buf row_scale, int write_grad) {
    const int acc64 = acc64_of(be);
    const int32_t *lab = (const int32_t *)labels.ptr;
    for (int64_t r = 0; r < rows; r++)
        if (lab[r] != RRC_IGNORE_INDEX && (lab[r] < 0 || lab[r] >= vocab))
            return RRC_ERROR(RRC_ERR_RANGE, "cross_entropy_fwd_bwd: labels[%lld] = %d outside [0, %lld) and not %d",
                             (long long)r, lab[r], (long long)vocab, RRC_IGNORE_INDEX);
    const float *rs = rrc_buf_is_none(row_scale) ? NULL : (const float *)row_scale.ptr;
    float *lo = (float *)loss.ptr;
#pragma omp parallel for schedule(static) if (rows * vocab > PAR_MIN)
    for (int64_t r = 0; r < rows; r++) {
        int32_t y = lab[r] == RRC_IGNORE_INDEX ? -1 : lab[r];
        float sc = grad_scale * (rs ? rs[r] : 1.0f);
        lo[r] = rrc_kc_xent_row(row_m(logits, r * ld), logits.dtype, vocab, y, sc, write_grad, acc64);
    }
    return RRC_OK;
}

/* ---- adamw ----------------------------------------------------------------------------------------- */

static rrc_status cpu_adamw_step(const rrc_backend *be, int64_t n, rrc_buf param, rrc_buf grad, rrc_buf m, rrc_buf v,
                                 const rrc_adamw_hparams *h) {
    (void)be;
    const rrc_kc_adamw_scalars s =
        rrc_kc_adamw_prepare(h->lr, h->beta1, h->beta2, h->eps, h->weight_decay, h->step, h->grad_scale);
    float *p = (float *)param.ptr, *mm = (float *)m.ptr, *vv = (float *)v.ptr;
    const float *g = (const float *)grad.ptr;
#pragma omp parallel for schedule(static) if (n > PAR_MIN)
    for (int64_t i = 0; i < n; i++) rrc_kc_adamw_elem(&p[i], g[i], &mm[i], &vv[i], &s);
    return RRC_OK;
}

/* ---- tables ------------------------------------------------------------------------------------------ */

#define CPU_TABLE(NAME, IMPL)                                                                                  \
    {                                                                                                          \
        .name = NAME, .host_accessible = 1, .impl = (void *)&IMPL, .alloc = cpu_alloc, .free = cpu_free,       \
        .memcpy_h2d = cpu_memcpy, .memcpy_d2h = cpu_memcpy, .memcpy_d2d = cpu_memcpy, .memset0 = cpu_memset0,  \
        .sync = cpu_sync, .gemm = cpu_gemm, .embedding_fwd = cpu_embedding_fwd, .add_bias = cpu_add_bias,      \
        .bias_grad = cpu_bias_grad, .rmsnorm_fwd = cpu_rmsnorm_fwd, .rmsnorm_bwd = cpu_rmsnorm_bwd,            \
        .rope_fwd = cpu_rope_fwd, .rope_bwd = cpu_rope_bwd, .attn_softmax_fwd = cpu_attn_softmax_fwd,          \
        .attn_softmax_bwd = cpu_attn_softmax_bwd, .swiglu_fwd = cpu_swiglu_fwd, .swiglu_bwd = cpu_swiglu_bwd,  \
        .add = cpu_add, .axpy = cpu_axpy, .scale = cpu_scale, .convert = cpu_convert,                          \
        .sum_squares = cpu_sum_squares, .cross_entropy_fwd_bwd = cpu_cross_entropy, .adamw_step = cpu_adamw_step \
    }

static cpu_impl g_impl64 = {1};
static cpu_impl g_impl32 = {0};
static const rrc_backend g_cpu64 = CPU_TABLE("cpu", g_impl64);
static const rrc_backend g_cpu32 = CPU_TABLE("cpu-f32acc", g_impl32);

const rrc_backend *rrc_backend_cpu(void) { return &g_cpu64; }
const rrc_backend *rrc_backend_cpu_f32acc(void) { return &g_cpu32; }
