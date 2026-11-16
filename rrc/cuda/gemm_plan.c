/* rrc/cuda/gemm_plan.c — see gemm_plan.h. Pure host C11 (no CUDA). */
#include "gemm_plan.h"

#include <stdio.h>
#include <string.h>

static int64_t sat_mul(int64_t a, int64_t b) {
    int64_t r;
    if (a < 0 || b < 0 || __builtin_mul_overflow(a, b, &r)) return INT64_MAX;
    return r;
}
static int64_t sat_add(int64_t a, int64_t b) {
    int64_t r;
    if (__builtin_add_overflow(a, b, &r)) return INT64_MAX;
    return r;
}
static int64_t round_up(int64_t v, int64_t q) {
    if (v > INT64_MAX - q) return INT64_MAX;
    return (v + q - 1) / q * q;
}
static int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }
/* Leading dimension of a dense scratch block: rounded up to RRC_GEMM_SCRATCH_LD_ALIGN elements
 * (aligned columns) unless the block is narrow, where the padding would dominate its size. */
static int64_t scratch_ld(int64_t cols) { return cols >= 64 ? round_up(cols, RRC_GEMM_SCRATCH_LD_ALIGN) : cols; }

unsigned rrc_gemm_align_bytes(const void *ptr, size_t es, int64_t ld, int64_t stride, int64_t batch, unsigned cap) {
    uint64_t v = (uint64_t)(uintptr_t)ptr | ((uint64_t)ld * (uint64_t)es);
    if (batch > 1 && stride != 0) v |= (uint64_t)stride * (uint64_t)es;
    unsigned a = 1;
    while (a < cap && (v & a) == 0) a <<= 1;
    return a;
}

/* Stored shape of the promoted block of A for a chunk of lm rows of C (rows x cols, row-major). */
static void a_block(const rrc_gemm_args *g, int64_t lm, int64_t *rows, int64_t *cols) {
    if (g->trans_a == RRC_OP_N) {
        *rows = lm;
        *cols = g->k;
    } else {
        *rows = g->k;
        *cols = lm;
    }
}
/* Stored shape of the promoted block of B for a chunk of ln columns of C. */
static void b_block(const rrc_gemm_args *g, int64_t ln, int64_t *rows, int64_t *cols) {
    if (g->trans_b == RRC_OP_N) {
        *rows = g->k;
        *cols = ln;
    } else {
        *rows = ln;
        *cols = g->k;
    }
}

/* Scratch layout for chunk extents (lm, ln, nb); returns the total in f32 elements. */
static int64_t layout(const rrc_gemm_plan *p, int64_t lm, int64_t ln, int64_t nb, int64_t *off_a, int64_t *off_b,
                      int64_t *off_c) {
    const int64_t pad = RRC_GEMM_MAX_ALIGN / 4; /* regions start 256-byte aligned */
    int64_t r, c, size_a = 0, size_b = 0, size_c = 0;
    if (p->promote_a) {
        a_block(&p->g, lm, &r, &c);
        size_a = sat_mul(sat_mul(r, scratch_ld(c)), p->a_bcast ? 1 : nb);
    }
    if (p->promote_b) {
        b_block(&p->g, ln, &r, &c);
        size_b = sat_mul(sat_mul(r, scratch_ld(c)), p->b_bcast ? 1 : nb);
    }
    if (p->c_scratch) size_c = sat_mul(sat_mul(lm, scratch_ld(ln)), nb);
    int64_t o = 0;
    if (off_a) *off_a = o;
    o = round_up(sat_add(o, size_a), pad);
    if (off_b) *off_b = o;
    o = round_up(sat_add(o, size_b), pad);
    if (off_c) *off_c = o;
    o = sat_add(o, size_c);
    return (size_a | size_b | size_c) == 0 ? 0 : o;
}

rrc_status rrc_gemm_plan_make(const rrc_gemm_args *g, const rrc_gemm_plan_opts *opts, rrc_gemm_level level,
                              rrc_gemm_plan *p) {
    RRC_REQUIRE(g && p, RRC_ERR_INVALID_ARG, "gemm plan: null argument");
    RRC_REQUIRE(level >= RRC_GEMM_LEVEL_NATIVE && level <= RRC_GEMM_LEVEL_KERNEL, RRC_ERR_INVALID_ARG,
                "gemm plan: invalid level %d", (int)level);
    RRC_REQUIRE(g->m > 0 && g->n > 0 && g->k > 0 && g->batch >= 1, RRC_ERR_SHAPE,
                "gemm plan: m, n, k must be > 0 and batch >= 1 (k == 0 is resolved in ops.c)");
    memset(p, 0, sizeof *p);
    p->g = *g;
    p->level = level;
    p->max_align = opts && opts->max_align ? opts->max_align : RRC_GEMM_MAX_ALIGN;
    const int64_t cap = opts && opts->scratch_cap_elems > 0 ? opts->scratch_cap_elems : RRC_GEMM_SCRATCH_CAP_DEFAULT;
    const int a16 = g->a.dtype == RRC_BF16, b16 = g->b.dtype == RRC_BF16, c16 = g->c.dtype == RRC_BF16;
    if (level == RRC_GEMM_LEVEL_NATIVE) {
        p->promote_a = a16 && !b16;
        p->promote_b = b16 && !a16;
        p->c_scratch = c16 && !(a16 && b16);
    } else if (level == RRC_GEMM_LEVEL_PROMOTE_ALL) {
        p->promote_a = a16;
        p->promote_b = b16;
        p->c_scratch = c16;
    }
    p->a_bcast = g->stride_a == 0;
    p->b_bcast = g->stride_b == 0;
    const int lt_f32 = level != RRC_GEMM_LEVEL_KERNEL && (g->a.dtype == RRC_F32 || p->promote_a) &&
                       (g->b.dtype == RRC_F32 || p->promote_b);
    p->tf32 = opts && opts->allow_tf32 && lt_f32;

    /* Chunk search. layout() is non-decreasing in each of lm, ln, nb (regions are padded, so not
     * strictly). Batch first: the largest nb whose full-size slices fit (binary search on the
     * actual layout, not a linear fit, which the 64-element region padding defeats). */
    int64_t lm = g->m, ln = g->n, nb = g->batch;
    if (layout(p, lm, ln, nb, NULL, NULL, NULL) > cap) {
        int64_t lo = 1, hi = g->batch; /* largest nb in [1, batch] with layout <= cap, else 1 */
        while (lo < hi) {
            const int64_t mid = lo + (hi - lo + 1) / 2;
            if (layout(p, lm, ln, mid, NULL, NULL, NULL) <= cap) lo = mid;
            else hi = mid - 1;
        }
        nb = lo;
    }
    /* Then rows/columns of C: halve the dimension that shrinks the scratch most. A halving that
     * does not shrink it (e.g. 3 -> 2 rows inside one padded region) is still taken when a smaller
     * extent of that dimension would; the search stops only when no smaller extent of either
     * dimension helps. Then the scratch equals the minimum chunk's (lm = ln = nb = 1): with a C
     * scratch region the size is strictly increasing in lm and ln, and without one it is
     * pad(A(lm)) + pad(B(ln)), so equality at both one-row and one-column extents implies it. */
    for (;;) {
        const int64_t e = layout(p, lm, ln, nb, NULL, NULL, NULL);
        if (e <= cap) break;
        const int64_t em_min = lm > 1 ? layout(p, 1, ln, nb, NULL, NULL, NULL) : INT64_MAX;
        const int64_t en_min = ln > 1 ? layout(p, lm, 1, nb, NULL, NULL, NULL) : INT64_MAX;
        if (em_min >= e && en_min >= e) break; /* minimum chunk: one row/column of k elements exceeds the cap */
        const int64_t em = em_min < e ? layout(p, (lm + 1) / 2, ln, nb, NULL, NULL, NULL) : INT64_MAX;
        const int64_t en = en_min < e ? layout(p, lm, (ln + 1) / 2, nb, NULL, NULL, NULL) : INT64_MAX;
        if (em < e || en < e) {
            if (em <= en) lm = (lm + 1) / 2;
            else ln = (ln + 1) / 2;
        } else if (em_min <= en_min) { /* no single halving helps yet; a smaller extent does */
            lm = (lm + 1) / 2;
        } else {
            ln = (ln + 1) / 2;
        }
    }
    p->lm = lm;
    p->ln = ln;
    p->nb = nb;
    p->cm = (g->m + lm - 1) / lm;
    p->cn = (g->n + ln - 1) / ln;
    p->cb = (g->batch + nb - 1) / nb;
    p->n_chunks = p->cm * p->cn * p->cb;
    p->scratch_elems = layout(p, lm, ln, nb, &p->off_a, &p->off_b, &p->off_c);
    RRC_REQUIRE(p->scratch_elems < INT64_MAX / 8, RRC_ERR_SHAPE, "gemm plan: scratch size overflow");
    return RRC_OK;
}

static const char *off_ptr(const void *base, rrc_dtype dt, int64_t off) {
    return (const char *)base + (size_t)off * rrc_dtype_size(dt);
}

void rrc_gemm_plan_chunk(const rrc_gemm_plan *p, void *scratch, int64_t idx, rrc_gemm_chunk *c) {
    const rrc_gemm_args *g = &p->g;
    memset(c, 0, sizeof *c);
    const int64_t bi = idx / (p->cm * p->cn), rem = idx % (p->cm * p->cn), mi = rem / p->cn, ni = rem % p->cn;
    const int64_t b0 = bi * p->nb, bn = min64(p->nb, g->batch - b0);
    const int64_t i0 = mi * p->lm, lm = min64(p->lm, g->m - i0);
    const int64_t j0 = ni * p->ln, ln = min64(p->ln, g->n - j0);
    float *sc = (float *)scratch;

    /* row-major sub-problem operands (pointer, dtype, ld, batch stride) */
    const void *pa = off_ptr(g->a.ptr, g->a.dtype, b0 * g->stride_a + (g->trans_a == RRC_OP_N ? i0 * g->lda : i0));
    rrc_dtype adt = g->a.dtype;
    int64_t lda = g->lda, sa = g->stride_a;
    const void *pb = off_ptr(g->b.ptr, g->b.dtype, b0 * g->stride_b + (g->trans_b == RRC_OP_N ? j0 : j0 * g->ldb));
    rrc_dtype bdt = g->b.dtype;
    int64_t ldb = g->ldb, sb = g->stride_b;
    void *pc = (void *)off_ptr(g->c.ptr, g->c.dtype, b0 * g->stride_c + i0 * g->ldc + j0);
    rrc_dtype cdt = g->c.dtype;
    int64_t ldc = g->ldc, scs = g->stride_c;

    if (p->promote_a) {
        int64_t r, cc;
        a_block(g, lm, &r, &cc);
        rrc_kc_conv2d_p *v = &c->pre[c->n_pre++];
        const int64_t dld = scratch_ld(cc);
        *v = (rrc_kc_conv2d_p){pa, adt, lda, sa, sc + p->off_a, RRC_F32, dld, p->a_bcast ? 0 : r * dld, r, cc,
                               p->a_bcast ? 1 : bn};
        pa = sc + p->off_a;
        adt = RRC_F32;
        lda = dld;
        sa = v->dstride;
    }
    if (p->promote_b) {
        int64_t r, cc;
        b_block(g, ln, &r, &cc);
        rrc_kc_conv2d_p *v = &c->pre[c->n_pre++];
        const int64_t dld = scratch_ld(cc);
        *v = (rrc_kc_conv2d_p){pb, bdt, ldb, sb, sc + p->off_b, RRC_F32, dld, p->b_bcast ? 0 : r * dld, r, cc,
                               p->b_bcast ? 1 : bn};
        pb = sc + p->off_b;
        bdt = RRC_F32;
        ldb = dld;
        sb = v->dstride;
    }
    if (p->c_scratch) {
        const int64_t dld = scratch_ld(ln), ds = lm * dld;
        float *dst = sc + p->off_c;
        /* beta == 0: C is not read; the scratch is zero-filled so the call never sees stale data */
        rrc_kc_conv2d_p *v = &c->pre[c->n_pre++];
        *v = (rrc_kc_conv2d_p){g->beta != 0.0f ? pc : NULL, cdt, ldc, scs, dst, RRC_F32, dld, ds, lm, ln, bn};
        c->has_post = 1;
        c->post = (rrc_kc_conv2d_p){dst, RRC_F32, dld, ds, pc, cdt, ldc, scs, lm, ln, bn};
        pc = dst;
        cdt = RRC_F32;
        ldc = dld;
        scs = ds;
    }

    /* column-major call: C^T (ln x lm) = op(B)^T (ln x k) * op(A)^T (k x lm) */
    rrc_lt_call *L = &c->call;
    L->p.transa = g->trans_b == RRC_OP_T;
    L->p.transb = g->trans_a == RRC_OP_T;
    L->p.m = ln;
    L->p.n = lm;
    L->p.k = g->k;
    L->p.batch = bn;
    L->p.alpha = g->alpha;
    L->p.beta = g->beta;
    L->p.a = pb;
    L->p.adt = bdt;
    L->p.lda = ldb;
    L->p.sa = sb;
    L->p.b = pa;
    L->p.bdt = adt;
    L->p.ldb = lda;
    L->p.sb = sa;
    L->p.c = pc;
    L->p.cdt = cdt;
    L->p.ldc = ldc;
    L->p.sc = scs;
    L->align_a = rrc_gemm_align_bytes(pb, rrc_dtype_size(bdt), ldb, sb, bn, p->max_align);
    L->align_b = rrc_gemm_align_bytes(pa, rrc_dtype_size(adt), lda, sa, bn, p->max_align);
    L->align_c = rrc_gemm_align_bytes(pc, rrc_dtype_size(cdt), ldc, scs, bn, p->max_align);
    L->tf32 = p->tf32;
}

void rrc_gemm_plan_describe(const rrc_gemm_plan *p, char *buf, size_t n) {
    if (!buf || n == 0) return;
    if (p->level == RRC_GEMM_LEVEL_KERNEL) {
        snprintf(buf, n, "kernel (rrc_kc_item_lt_matmul, no library call)");
        return;
    }
    char extra[96] = "";
    if (p->n_chunks > 1) snprintf(extra, sizeof extra, ", %lld chunks", (long long)p->n_chunks);
    if (!p->promote_a && !p->promote_b && !p->c_scratch) {
        snprintf(buf, n, "%s cuBLASLt%s%s", p->level == RRC_GEMM_LEVEL_NATIVE ? "native" : "promote-all",
                 p->tf32 ? " (tf32)" : "", extra);
        return;
    }
    snprintf(buf, n, "%s cuBLASLt:%s%s%s%s%s", p->level == RRC_GEMM_LEVEL_NATIVE ? "promoted" : "promote-all",
             p->promote_a ? " A(bf16->f32)" : "", p->promote_b ? " B(bf16->f32)" : "",
             p->c_scratch ? " C(f32 scratch->bf16)" : "", p->tf32 ? " (tf32)" : "", extra);
}

static void fill_report(const rrc_gemm_plan *p, rrc_gemm_exec_report *rep) {
    if (!rep) return;
    rep->level = p->level;
    rep->promote_a = p->promote_a;
    rep->promote_b = p->promote_b;
    rep->c_scratch = p->c_scratch;
    rep->n_chunks = p->n_chunks;
    rrc_gemm_plan_describe(p, rep->desc, sizeof rep->desc);
}

rrc_status rrc_gemm_execute(const rrc_gemm_args *g, const rrc_gemm_plan_opts *opts, rrc_gemm_level min_level,
                            const rrc_gemm_exec_ops *ops, void *ctx, rrc_gemm_exec_report *rep) {
    RRC_REQUIRE(ops && ops->scratch && ops->lt_supported && ops->lt_run && ops->kernel_run && ops->convert,
                RRC_ERR_INVALID_ARG, "gemm execute: incomplete ops");
    rrc_gemm_plan prev;
    int have_prev = 0;
    rrc_gemm_chunk ch;
    for (int lv = (int)min_level; lv <= (int)RRC_GEMM_LEVEL_KERNEL; lv++) {
        rrc_gemm_plan p;
        RRC_CHECK(rrc_gemm_plan_make(g, opts, (rrc_gemm_level)lv, &p));
        if (lv == RRC_GEMM_LEVEL_KERNEL) {
            rrc_gemm_plan_chunk(&p, NULL, 0, &ch);
            RRC_CHECK(ops->kernel_run(ctx, &ch.call.p));
            fill_report(&p, rep);
            return RRC_OK;
        }
        /* PROMOTE_ALL identical to the NATIVE plan just refused: nothing new to try */
        if (have_prev && prev.promote_a == p.promote_a && prev.promote_b == p.promote_b &&
            prev.c_scratch == p.c_scratch)
            continue;
        prev = p;
        have_prev = 1;
        void *scratch = NULL;
        if (p.scratch_elems > 0) {
            rrc_status s = ops->scratch(ctx, (size_t)p.scratch_elems * sizeof(float), &scratch);
            if (s != RRC_OK) {
                rrc_log(RRC_LOG_WARN, "gemm: %lld bytes of conversion scratch unavailable (%s); trying the next path",
                        (long long)p.scratch_elems * 4, rrc_last_error());
                rrc_clear_error();
                continue;
            }
        }
        int ok = 1;
        for (int64_t i = 0; i < p.n_chunks && ok; i++) {
            rrc_gemm_plan_chunk(&p, scratch, i, &ch);
            ok = ops->lt_supported(ctx, &ch.call);
        }
        if (!ok) continue;
        for (int64_t i = 0; i < p.n_chunks; i++) {
            rrc_gemm_plan_chunk(&p, scratch, i, &ch);
            for (int j = 0; j < ch.n_pre; j++) RRC_CHECK(ops->convert(ctx, &ch.pre[j]));
            RRC_CHECK(ops->lt_run(ctx, &ch.call));
            if (ch.has_post) RRC_CHECK(ops->convert(ctx, &ch.post));
        }
        fill_report(&p, rep);
        return RRC_OK;
    }
    return RRC_ERROR(RRC_ERR_INTERNAL, "gemm execute: no level ran");
}
