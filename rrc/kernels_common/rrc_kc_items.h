/* rrc/kernels_common/rrc_kc_items.h — per-item bodies of every CUDA kernel (DESIGN.md §11).
 *
 * Each __global__ kernel in rrc/cuda/cuda_kernels.cu is a thin shell: it computes an item index,
 * checks bounds and calls one of the RRC_HD functions below. The same functions are executed on
 * the CPU by the host emulation backend (rrc/cuda/kc_emu.c) in shuffled item order, which is how
 * the kernel bodies get execution coverage without a GPU (tests/c/test_kc_items.c).
 *
 * Item kinds:
 *   - element items  rrc_kc_item_<op>(p, idx): one output element (or one rope pair, one swiglu
 *     (dgate, dup) pair). Each item reads all of its inputs before it writes, and never reads an
 *     element another item writes, so any execution order gives the same bits, and the exact
 *     aliases allowed by backend.h are safe.
 *   - row reductions in three phases with a fixed order that depends only on the shape:
 *       1. lane partials  rrc_kc_lane_<op>(p, row, lane), lane in [0, RRC_KC_LANES): lane l
 *          accumulates elements l, l + LANES, l + 2*LANES, ... sequentially in fp32;
 *       2. a fixed pairwise tree over the LANES partials (rrc_kc_tree_*_step for s = LANES/2,
 *          LANES/4, ..., 1: part[l] = part[l] (+) part[l + s] for l < s);
 *       3. finalize (row statistics) and element items that read the statistics.
 *     The CUDA kernels run one block of RRC_KC_LANES threads per row with __syncthreads between
 *     the phases (and between tree levels), so the summation order never depends on gridDim or
 *     on scheduling: results are bit-identical run to run. All reads of phase 1 and 3 happen
 *     before any write of phase 3 within a row (in place / alias safe).
 *   - sum_squares: fixed chunks of RRC_KC_SS_CHUNK elements, each reduced like a row; then one
 *     row over the chunk partials.
 * Data-dependent range checks (token ids, position ids, labels) are done before the dependent
 * read; an item that fails returns 1 and writes nothing (the caller records a range error).
 */
#ifndef RRC_KC_ITEMS_H
#define RRC_KC_ITEMS_H

#include "rrc_kc.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RRC_KC_LANES 256      /* lanes per row reduction = CUDA block size of the row kernels */
#define RRC_KC_SS_CHUNK 4096  /* elements per sum_squares chunk */
#define RRC_KC_IGNORE_INDEX (-100) /* == RRC_IGNORE_INDEX of backend.h (checked in kc_emu.c) */

/* ------------------------------------------------------------------------------------------ */
/* Fixed-order tree reductions                                                                 */
/* ------------------------------------------------------------------------------------------ */

RRC_HD static inline void rrc_kc_tree_sum_step(float *part, int lane, int s) {
    if (lane < s) part[lane] = part[lane] + part[lane + s];
}
RRC_HD static inline void rrc_kc_tree_max_step(float *part, int lane, int s) {
    if (lane < s && part[lane + s] > part[lane]) part[lane] = part[lane + s];
}
RRC_HD static inline void rrc_kc_tree_lse_step(rrc_kc_lse *part, int lane, int s) {
    if (lane < s) part[lane] = rrc_kc_lse_merge(part[lane], part[lane + s]);
}
/* Serial execution of the whole tree (same result as the parallel levels: at level s, lanes < s
 * read part[l] and part[l + s], and part[l + s] is not written at that level). */
RRC_HD static inline float rrc_kc_tree_sum(float *part) {
    for (int s = RRC_KC_LANES / 2; s > 0; s >>= 1)
        for (int l = 0; l < s; l++) rrc_kc_tree_sum_step(part, l, s);
    return part[0];
}
RRC_HD static inline float rrc_kc_tree_max(float *part) {
    for (int s = RRC_KC_LANES / 2; s > 0; s >>= 1)
        for (int l = 0; l < s; l++) rrc_kc_tree_max_step(part, l, s);
    return part[0];
}
RRC_HD static inline rrc_kc_lse rrc_kc_tree_lse(rrc_kc_lse *part) {
    for (int s = RRC_KC_LANES / 2; s > 0; s >>= 1)
        for (int l = 0; l < s; l++) rrc_kc_tree_lse_step(part, l, s);
    return part[0];
}

/* ------------------------------------------------------------------------------------------ */
/* GEMM: column-major matmul (cuBLASLt semantics) and the operand conversion of the GEMM plan   */
/* ------------------------------------------------------------------------------------------ */

/* D = alpha * op(A) * op(B) + beta * C in column-major storage, D == C (cuBLASLt's in-place form).
 * op(A) is m x k: transa == 0 -> A(i, q) = a[i + q*lda]; 1 -> a[q + i*lda]. op(B) is k x n:
 * transb == 0 -> B(q, j) = b[q + j*ldb]; 1 -> b[j + q*ldb]. C(i, j) = c[i + j*ldc]. Batch slice
 * t adds t*sa / t*sb / t*sc elements. Any dtype mix (f32 | bf16 each). */
typedef struct rrc_kc_lt_p {
    int transa, transb;
    int64_t m, n, k, batch;
    float alpha, beta;
    const void *a;
    rrc_dtype adt;
    int64_t lda, sa;
    const void *b;
    rrc_dtype bdt;
    int64_t ldb, sb;
    void *c;
    rrc_dtype cdt;
    int64_t ldc, sc;
} rrc_kc_lt_p;

/* One element of D, idx in [0, batch*m*n) (i fastest, so consecutive items write consecutive
 * column-major addresses). The dot product runs over q = 0..k-1 in four interleaved fp32 partial
 * sums combined as (s0 + s1) + (s2 + s3): the order of rrc_backend_cpu_f32acc's gemm, so the
 * result is bit-identical to it (the products are commutative and the CPU computes the same
 * op(A)(i_r, q) * op(B)(q, j_r) pairs). This is the device fallback GEMM kernel and the
 * reference that consumes GEMM plans in the tests. */
RRC_HD static inline void rrc_kc_item_lt_matmul(const rrc_kc_lt_p *p, int64_t idx) {
    const int64_t mn = p->m * p->n;
    const int64_t t = idx / mn, r = idx % mn;
    const int64_t i = r % p->m, j = r / p->m;
    const int64_t ab = t * p->sa, bb = t * p->sb;
    const int64_t ai = p->transa ? i * p->lda : i, aq = p->transa ? 1 : p->lda;
    const int64_t bj = p->transb ? j : j * p->ldb, bq = p->transb ? p->ldb : 1;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int64_t q = 0;
    for (; q + 4 <= p->k; q += 4) {
        s0 += rrc_kc_ld(p->a, p->adt, ab + ai + q * aq) * rrc_kc_ld(p->b, p->bdt, bb + bj + q * bq);
        s1 += rrc_kc_ld(p->a, p->adt, ab + ai + (q + 1) * aq) * rrc_kc_ld(p->b, p->bdt, bb + bj + (q + 1) * bq);
        s2 += rrc_kc_ld(p->a, p->adt, ab + ai + (q + 2) * aq) * rrc_kc_ld(p->b, p->bdt, bb + bj + (q + 2) * bq);
        s3 += rrc_kc_ld(p->a, p->adt, ab + ai + (q + 3) * aq) * rrc_kc_ld(p->b, p->bdt, bb + bj + (q + 3) * bq);
    }
    for (; q < p->k; q++) s0 += rrc_kc_ld(p->a, p->adt, ab + ai + q * aq) * rrc_kc_ld(p->b, p->bdt, bb + bj + q * bq);
    const float dot = (s0 + s1) + (s2 + s3);
    const int64_t ci = t * p->sc + i + j * p->ldc;
    float v = p->alpha * dot;
    if (p->beta != 0.0f) v += p->beta * rrc_kc_ld(p->c, p->cdt, ci);
    rrc_kc_st(p->c, p->cdt, ci, v);
}

/* Batched 2-D copy with dtype conversion: dst[t*dstride + r*dld + c] = src[t*sstride + r*sld + c]
 * for t < batch, r < rows, c < cols (exact widening bf16 -> f32, RNE narrowing f32 -> bf16).
 * src == NULL writes zeros. */
typedef struct rrc_kc_conv2d_p {
    const void *src;
    rrc_dtype sdt;
    int64_t sld, sstride;
    void *dst;
    rrc_dtype ddt;
    int64_t dld, dstride;
    int64_t rows, cols, batch;
} rrc_kc_conv2d_p;

RRC_HD static inline void rrc_kc_item_conv2d(const rrc_kc_conv2d_p *p, int64_t idx) {
    const int64_t rc = p->rows * p->cols;
    const int64_t t = idx / rc, e = idx % rc, r = e / p->cols, c = e % p->cols;
    const float v = p->src ? rrc_kc_ld(p->src, p->sdt, t * p->sstride + r * p->sld + c) : 0.0f;
    rrc_kc_st(p->dst, p->ddt, t * p->dstride + r * p->dld + c, v);
}

/* ------------------------------------------------------------------------------------------ */
/* Embedding / bias                                                                            */
/* ------------------------------------------------------------------------------------------ */

typedef struct rrc_kc_embed_p {
    int64_t n_tokens, dim, vocab;
    const int32_t *ids;
    const void *table;
    rrc_dtype tdt;
    void *out;
    rrc_dtype odt;
} rrc_kc_embed_p;

/* idx in [0, n_tokens*dim). Returns 1 (nothing read from the table, nothing written) if the
 * token's id is outside [0, vocab); *bad receives the token index. */
RRC_HD static inline int rrc_kc_item_embedding(const rrc_kc_embed_p *p, int64_t idx, int64_t *bad) {
    const int64_t t = idx / p->dim, c = idx % p->dim;
    const int32_t id = p->ids[t];
    if (id < 0 || (int64_t)id >= p->vocab) {
        *bad = t;
        return 1;
    }
    rrc_kc_st(p->out, p->odt, idx, rrc_kc_ld(p->table, p->tdt, (int64_t)id * p->dim + c));
    return 0;
}

typedef struct rrc_kc_bias_p {
    int64_t rows, cols, ld;
    void *x; /* add_bias: x (in place); bias_grad: dy (read only) */
    rrc_dtype xdt;
    void *bias; /* add_bias: bias (read only); bias_grad: dbias f32 (written) */
    rrc_dtype bdt;
    int accumulate;
} rrc_kc_bias_p;

/* add_bias: idx in [0, rows*cols) */
RRC_HD static inline void rrc_kc_item_add_bias(const rrc_kc_bias_p *p, int64_t idx) {
    const int64_t r = idx / p->cols, c = idx % p->cols, o = r * p->ld + c;
    rrc_kc_st(p->x, p->xdt, o, rrc_kc_ld(p->x, p->xdt, o) + rrc_kc_ld(p->bias, p->bdt, c));
}
/* bias_grad: one item per column c in [0, cols), rows summed sequentially in fp32 (the order of
 * rrc_backend_cpu_f32acc: bit-identical). */
RRC_HD static inline void rrc_kc_item_bias_grad(const rrc_kc_bias_p *p, int64_t c) {
    float s = 0.0f;
    for (int64_t r = 0; r < p->rows; r++) s += rrc_kc_ld(p->x, p->xdt, r * p->ld + c);
    float *db = (float *)p->bias;
    db[c] = p->accumulate ? db[c] + s : s;
}

/* ------------------------------------------------------------------------------------------ */
/* RMSNorm                                                                                     */
/* ------------------------------------------------------------------------------------------ */

typedef struct rrc_kc_rmsnorm_p {
    int64_t rows, dim, ld;
    const void *x;
    rrc_dtype xdt;
    const void *w;
    rrc_dtype wdt;
    const void *dy; /* backward only */
    rrc_dtype dydt;
    void *y; /* forward: y; backward: dx */
    rrc_dtype ydt;
    float eps;
    int accumulate; /* backward only */
} rrc_kc_rmsnorm_p;

/* lane partial of sum x^2 (forward and backward) */
RRC_HD static inline float rrc_kc_lane_rmsnorm_sumsq(const rrc_kc_rmsnorm_p *p, int64_t row, int lane) {
    const int64_t o = row * p->ld;
    float s = 0.0f;
    for (int64_t i = lane; i < p->dim; i += RRC_KC_LANES) {
        float v = rrc_kc_ld(p->x, p->xdt, o + i);
        s += v * v;
    }
    return s;
}
/* lane partial of sum dy*w*x (backward) */
RRC_HD static inline float rrc_kc_lane_rmsnorm_dot(const rrc_kc_rmsnorm_p *p, int64_t row, int lane) {
    const int64_t o = row * p->ld;
    float s = 0.0f;
    for (int64_t i = lane; i < p->dim; i += RRC_KC_LANES)
        s += rrc_kc_ld(p->dy, p->dydt, o + i) * rrc_kc_ld(p->w, p->wdt, i) * rrc_kc_ld(p->x, p->xdt, o + i);
    return s;
}
RRC_HD static inline float rrc_kc_rmsnorm_finalize_rstd(const rrc_kc_rmsnorm_p *p, float sumsq) {
    return rrc_kc_rmsnorm_rstd_f(sumsq, p->dim, p->eps);
}
RRC_HD static inline float rrc_kc_rmsnorm_finalize_coef(const rrc_kc_rmsnorm_p *p, float dot, float rstd) {
    return rrc_kc_rmsnorm_bwd_coef_f(dot, p->dim, rstd);
}
/* element i of row `row` */
RRC_HD static inline void rrc_kc_item_rmsnorm_fwd(const rrc_kc_rmsnorm_p *p, int64_t row, int64_t i, float rstd) {
    const int64_t o = row * p->ld + i;
    rrc_kc_st(p->y, p->ydt, o, rrc_kc_rmsnorm_elem_fwd(rrc_kc_ld(p->x, p->xdt, o), rstd, rrc_kc_ld(p->w, p->wdt, i)));
}
RRC_HD static inline void rrc_kc_item_rmsnorm_bwd(const rrc_kc_rmsnorm_p *p, int64_t row, int64_t i, float rstd,
                                                  float k) {
    const int64_t o = row * p->ld + i;
    float g = rrc_kc_rmsnorm_elem_bwd(rrc_kc_ld(p->x, p->xdt, o), rrc_kc_ld(p->w, p->wdt, i),
                                      rrc_kc_ld(p->dy, p->dydt, o), rstd, k);
    if (p->accumulate) g += rrc_kc_ld(p->y, p->ydt, o);
    rrc_kc_st(p->y, p->ydt, o, g);
}

/* ------------------------------------------------------------------------------------------ */
/* RoPE                                                                                        */
/* ------------------------------------------------------------------------------------------ */

typedef struct rrc_kc_rope_p {
    int64_t n_tokens, n_heads, head_dim, ld, n_pos;
    const void *x;
    rrc_dtype xdt;
    void *y;
    rrc_dtype ydt;
    const int32_t *pos;
    const float *cos_tab, *sin_tab;
    int backward;
} rrc_kc_rope_p;

/* One rotation pair (i, i + head_dim/2): idx in [0, n_tokens*n_heads*(head_dim/2)). Reads both
 * elements before writing both (y may alias x). Returns 1 (nothing read from the tables, nothing
 * written) if the token's position is outside [0, n_pos); *bad receives the token index. */
RRC_HD static inline int rrc_kc_item_rope(const rrc_kc_rope_p *p, int64_t idx, int64_t *bad) {
    const int64_t half = p->head_dim / 2, per_tok = p->n_heads * half;
    const int64_t t = idx / per_tok, e = idx % per_tok, h = e / half, i = e % half;
    const int32_t ps = p->pos[t];
    if (ps < 0 || (int64_t)ps >= p->n_pos) {
        *bad = t;
        return 1;
    }
    const int64_t o = t * p->ld + h * p->head_dim + i;
    const float c = p->cos_tab[(int64_t)ps * half + i], s = p->sin_tab[(int64_t)ps * half + i];
    const float a = rrc_kc_ld(p->x, p->xdt, o), b = rrc_kc_ld(p->x, p->xdt, o + half);
    float o1, o2;
    if (p->backward) rrc_kc_rope_pair_bwd(a, b, c, s, &o1, &o2);
    else rrc_kc_rope_pair_fwd(a, b, c, s, &o1, &o2);
    rrc_kc_st(p->y, p->ydt, o, o1);
    rrc_kc_st(p->y, p->ydt, o + half, o2);
    return 0;
}

/* ------------------------------------------------------------------------------------------ */
/* Attention softmax                                                                           */
/* ------------------------------------------------------------------------------------------ */

typedef struct rrc_kc_softmax_p {
    int64_t batch, rows, cols, ld, bs, row_offset;
    const void *s; /* forward: scores; backward: p */
    rrc_dtype sdt;
    const void *dp; /* backward only */
    rrc_dtype dpdt;
    void *out; /* forward: p; backward: ds */
    rrc_dtype odt;
    float scale;
    int causal;
    const int32_t *doc; /* NULL: no document mask */
} rrc_kc_softmax_p;

/* row br in [0, batch*rows): element offset of its first column and its query position */
RRC_HD static inline int64_t rrc_kc_softmax_row_off(const rrc_kc_softmax_p *p, int64_t br) {
    return (br / p->rows) * p->bs + (br % p->rows) * p->ld;
}
RRC_HD static inline int64_t rrc_kc_softmax_row_q(const rrc_kc_softmax_p *p, int64_t br) {
    return p->row_offset + br % p->rows;
}
/* forward phase 1: lane max of scale*s over allowed keys (-inf if none) */
RRC_HD static inline float rrc_kc_lane_softmax_max(const rrc_kc_softmax_p *p, int64_t br, int lane) {
    const int64_t o = rrc_kc_softmax_row_off(p, br), q = rrc_kc_softmax_row_q(p, br);
    float m = -INFINITY;
    for (int64_t j = lane; j < p->cols; j += RRC_KC_LANES)
        if (rrc_kc_attn_allowed(j, q, p->causal, p->doc)) {
            float x = p->scale * rrc_kc_ld(p->s, p->sdt, o + j);
            if (x > m) m = x;
        }
    return m;
}
/* forward phase 2: lane sum of exp(scale*s - m) over allowed keys */
RRC_HD static inline float rrc_kc_lane_softmax_sum(const rrc_kc_softmax_p *p, int64_t br, int lane, float m) {
    const int64_t o = rrc_kc_softmax_row_off(p, br), q = rrc_kc_softmax_row_q(p, br);
    float z = 0.0f;
    if (m == -INFINITY) return z; /* no allowed key: zero row, nothing to sum */
    for (int64_t j = lane; j < p->cols; j += RRC_KC_LANES)
        if (rrc_kc_attn_allowed(j, q, p->causal, p->doc)) z += expf(p->scale * rrc_kc_ld(p->s, p->sdt, o + j) - m);
    return z;
}
/* forward element j; m == -inf (no allowed key) writes a zero row */
RRC_HD static inline void rrc_kc_item_softmax_fwd(const rrc_kc_softmax_p *p, int64_t br, int64_t j, float m,
                                                  float inv_z) {
    const int64_t o = rrc_kc_softmax_row_off(p, br), q = rrc_kc_softmax_row_q(p, br);
    float v = 0.0f;
    if (m != -INFINITY && rrc_kc_attn_allowed(j, q, p->causal, p->doc))
        v = rrc_kc_softmax_elem(p->scale * rrc_kc_ld(p->s, p->sdt, o + j), m, inv_z);
    rrc_kc_st(p->out, p->odt, o + j, v);
}
/* backward phase 1: lane partial of sum p*dp */
RRC_HD static inline float rrc_kc_lane_softmax_dot(const rrc_kc_softmax_p *p, int64_t br, int lane) {
    const int64_t o = rrc_kc_softmax_row_off(p, br);
    float d = 0.0f;
    for (int64_t j = lane; j < p->cols; j += RRC_KC_LANES)
        d += rrc_kc_ld(p->s, p->sdt, o + j) * rrc_kc_ld(p->dp, p->dpdt, o + j);
    return d;
}
RRC_HD static inline void rrc_kc_item_softmax_bwd(const rrc_kc_softmax_p *p, int64_t br, int64_t j, float dot) {
    const int64_t o = rrc_kc_softmax_row_off(p, br) + j;
    rrc_kc_st(p->out, p->odt, o,
              rrc_kc_softmax_bwd_elem(rrc_kc_ld(p->s, p->sdt, o), rrc_kc_ld(p->dp, p->dpdt, o), dot, p->scale));
}

/* ------------------------------------------------------------------------------------------ */
/* SwiGLU and elementwise ops                                                                  */
/* ------------------------------------------------------------------------------------------ */

typedef struct rrc_kc_swiglu_p {
    int64_t rows, cols, ld_in, ld_dh, ld_out;
    const void *gate;
    rrc_dtype gdt;
    const void *up;
    rrc_dtype udt;
    const void *dh; /* backward only */
    rrc_dtype dhdt;
    void *out0; /* forward: h; backward: dgate */
    rrc_dtype o0dt;
    void *out1; /* backward: dup */
    rrc_dtype o1dt;
} rrc_kc_swiglu_p;

RRC_HD static inline void rrc_kc_item_swiglu_fwd(const rrc_kc_swiglu_p *p, int64_t idx) {
    const int64_t r = idx / p->cols, c = idx % p->cols;
    const float g = rrc_kc_ld(p->gate, p->gdt, r * p->ld_in + c), u = rrc_kc_ld(p->up, p->udt, r * p->ld_in + c);
    rrc_kc_st(p->out0, p->o0dt, r * p->ld_out + c, rrc_kc_swiglu_fwd(g, u));
}
/* one item writes both dgate and dup of (r, c), after reading gate, up and dh */
RRC_HD static inline void rrc_kc_item_swiglu_bwd(const rrc_kc_swiglu_p *p, int64_t idx) {
    const int64_t r = idx / p->cols, c = idx % p->cols;
    const float g = rrc_kc_ld(p->gate, p->gdt, r * p->ld_in + c), u = rrc_kc_ld(p->up, p->udt, r * p->ld_in + c);
    const float d = rrc_kc_ld(p->dh, p->dhdt, r * p->ld_dh + c);
    float dg, du;
    rrc_kc_swiglu_bwd(g, u, d, &dg, &du);
    rrc_kc_st(p->out0, p->o0dt, r * p->ld_out + c, dg);
    rrc_kc_st(p->out1, p->o1dt, r * p->ld_out + c, du);
}

typedef struct rrc_kc_ew_p {
    int64_t n;
    float alpha;
    const void *a; /* add: a; axpy: x; convert: src */
    rrc_dtype adt;
    const void *b; /* add: b */
    rrc_dtype bdt;
    void *out; /* add: out; axpy: y; scale: x; convert: dst */
    rrc_dtype odt;
} rrc_kc_ew_p;

RRC_HD static inline void rrc_kc_item_add(const rrc_kc_ew_p *p, int64_t i) {
    rrc_kc_st(p->out, p->odt, i, rrc_kc_ld(p->a, p->adt, i) + rrc_kc_ld(p->b, p->bdt, i));
}
RRC_HD static inline void rrc_kc_item_axpy(const rrc_kc_ew_p *p, int64_t i) {
    rrc_kc_st(p->out, p->odt, i, rrc_kc_ld(p->out, p->odt, i) + p->alpha * rrc_kc_ld(p->a, p->adt, i));
}
RRC_HD static inline void rrc_kc_item_scale(const rrc_kc_ew_p *p, int64_t i) {
    rrc_kc_st(p->out, p->odt, i, p->alpha * rrc_kc_ld(p->out, p->odt, i));
}
RRC_HD static inline void rrc_kc_item_convert(const rrc_kc_ew_p *p, int64_t i) {
    rrc_kc_st(p->out, p->odt, i, rrc_kc_ld(p->a, p->adt, i));
}

/* ------------------------------------------------------------------------------------------ */
/* sum_squares                                                                                 */
/* ------------------------------------------------------------------------------------------ */

typedef struct rrc_kc_ss_p {
    int64_t n;
    const void *x;
    rrc_dtype xdt;
    float *part; /* [n_chunks] chunk sums */
    float *out;
    int accumulate;
} rrc_kc_ss_p;

RRC_HD static inline int64_t rrc_kc_ss_chunks(int64_t n) { return (n + RRC_KC_SS_CHUNK - 1) / RRC_KC_SS_CHUNK; }
/* stage 1: lane partial of chunk `chunk` */
RRC_HD static inline float rrc_kc_lane_ss_chunk(const rrc_kc_ss_p *p, int64_t chunk, int lane) {
    const int64_t i0 = chunk * RRC_KC_SS_CHUNK;
    const int64_t i1 = i0 + RRC_KC_SS_CHUNK < p->n ? i0 + RRC_KC_SS_CHUNK : p->n;
    float s = 0.0f;
    for (int64_t i = i0 + lane; i < i1; i += RRC_KC_LANES) {
        float v = rrc_kc_ld(p->x, p->xdt, i);
        s += v * v;
    }
    return s;
}
/* stage 2: lane partial over the chunk sums */
RRC_HD static inline float rrc_kc_lane_ss_parts(const rrc_kc_ss_p *p, int lane) {
    const int64_t nc = rrc_kc_ss_chunks(p->n);
    float s = 0.0f;
    for (int64_t c = lane; c < nc; c += RRC_KC_LANES) s += p->part[c];
    return s;
}
RRC_HD static inline void rrc_kc_ss_finalize(const rrc_kc_ss_p *p, float total) {
    p->out[0] = p->accumulate ? p->out[0] + total : total;
}

/* ------------------------------------------------------------------------------------------ */
/* Cross entropy                                                                               */
/* ------------------------------------------------------------------------------------------ */

typedef struct rrc_kc_xent_p {
    int64_t rows, vocab, ld;
    void *z;
    rrc_dtype zdt;
    const int32_t *labels;
    float *loss;
    const float *row_scale; /* NULL: 1 */
    float grad_scale;
    int write_grad;
} rrc_kc_xent_p;

typedef struct rrc_kc_xent_stats {
    float m, log_s, scale;
    int32_t label; /* -1: ignored row (zero gradient) */
    int valid;     /* 0: invalid label, row is not written */
} rrc_kc_xent_stats;

/* phase 1: online (max, sum exp) of the lane's logits (empty for an ignored row) */
RRC_HD static inline rrc_kc_lse rrc_kc_lane_xent(const rrc_kc_xent_p *p, int64_t row, int lane) {
    rrc_kc_lse a = rrc_kc_lse_init();
    if (p->labels[row] == RRC_KC_IGNORE_INDEX) return a;
    const int64_t o = row * p->ld;
    for (int64_t j = lane; j < p->vocab; j += RRC_KC_LANES) a = rrc_kc_lse_push(a, rrc_kc_ld(p->z, p->zdt, o + j));
    return a;
}
/* phase 2 (after the tree): checks the label (returns 1 and writes nothing if it is neither the
 * ignore index nor in [0, vocab)), writes loss[row] (reads z[label] before any gradient write) */
RRC_HD static inline int rrc_kc_xent_finalize(const rrc_kc_xent_p *p, int64_t row, rrc_kc_lse a,
                                              rrc_kc_xent_stats *st) {
    const int32_t y = p->labels[row];
    st->m = 0.0f;
    st->log_s = 0.0f;
    st->scale = 0.0f;
    st->label = -1;
    st->valid = 1;
    if (y == RRC_KC_IGNORE_INDEX) {
        p->loss[row] = 0.0f;
        return 0;
    }
    if (y < 0 || (int64_t)y >= p->vocab) {
        st->valid = 0;
        return 1;
    }
    st->m = a.m;
    st->log_s = logf(a.s);
    st->label = y;
    st->scale = p->grad_scale * (p->row_scale ? p->row_scale[row] : 1.0f);
    p->loss[row] = rrc_kc_xent_loss(st->m, st->log_s, rrc_kc_ld(p->z, p->zdt, row * p->ld + y));
    return 0;
}
/* phase 3: gradient element j (only when write_grad) */
RRC_HD static inline void rrc_kc_item_xent_grad(const rrc_kc_xent_p *p, int64_t row, int64_t j,
                                                const rrc_kc_xent_stats *st) {
    if (!st->valid) return;
    const int64_t o = row * p->ld + j;
    if (st->label < 0) {
        rrc_kc_st(p->z, p->zdt, o, 0.0f);
        return;
    }
    rrc_kc_st(p->z, p->zdt, o, rrc_kc_xent_grad_elem(rrc_kc_ld(p->z, p->zdt, o), st->m, st->log_s, j == st->label, st->scale));
}

/* ------------------------------------------------------------------------------------------ */
/* AdamW                                                                                       */
/* ------------------------------------------------------------------------------------------ */

typedef struct rrc_kc_adamw_p {
    int64_t n;
    float *param;
    const float *grad;
    float *m, *v;
    rrc_kc_adamw_scalars s;
} rrc_kc_adamw_p;

RRC_HD static inline void rrc_kc_item_adamw(const rrc_kc_adamw_p *p, int64_t i) {
    rrc_kc_adamw_elem(&p->param[i], p->grad[i], &p->m[i], &p->v[i], &p->s);
}

#ifdef __cplusplus
}
#endif

#endif /* RRC_KC_ITEMS_H */
