/* rrc/src/ops.c — host-side precondition checks for every op, then dispatch to the backend.
 * The rules implemented here are documented per op in include/rrc/backend.h. */
#include <stdint.h>

#include "rrc/backend.h"

#define M_F32 (1u << RRC_F32)
#define M_BF16 (1u << RRC_BF16)
#define M_I32 (1u << RRC_I32)
#define M_FLT (M_F32 | M_BF16)

static rrc_status chk_be(const rrc_backend *be, const char *op) {
    RRC_REQUIRE(be != NULL, RRC_ERR_INVALID_ARG, "%s: backend is NULL", op);
    return RRC_OK;
}

static rrc_status chk_nonneg(const char *op, const char *what, int64_t v) {
    RRC_REQUIRE(v >= 0, RRC_ERR_SHAPE, "%s: %s = %lld must be >= 0", op, what, (long long)v);
    return RRC_OK;
}

static rrc_status mul_ck(const char *op, int64_t a, int64_t b, int64_t *out) {
    if (__builtin_mul_overflow(a, b, out)) return RRC_ERROR(RRC_ERR_SHAPE, "%s: size overflow", op);
    return RRC_OK;
}
static rrc_status add_ck(const char *op, int64_t a, int64_t b, int64_t *out) {
    if (__builtin_add_overflow(a, b, out)) return RRC_ERROR(RRC_ERR_SHAPE, "%s: size overflow", op);
    return RRC_OK;
}

/* Elements spanned by `batch` slices of a rows x cols matrix with leading dimension ld. */
static rrc_status extent(const char *op, int64_t batch, int64_t stride, int64_t rows, int64_t cols, int64_t ld,
                         int64_t *out) {
    if (batch == 0 || rows == 0 || cols == 0) {
        *out = 0;
        return RRC_OK;
    }
    int64_t e, t;
    RRC_CHECK(mul_ck(op, rows - 1, ld, &e));
    RRC_CHECK(add_ck(op, e, cols, &e));
    RRC_CHECK(mul_ck(op, batch - 1, stride, &t));
    RRC_CHECK(add_ck(op, e, t, &e));
    *out = e;
    return RRC_OK;
}

static const char *mask_str(unsigned m) {
    switch (m) {
    case M_F32: return "f32";
    case M_BF16: return "bf16";
    case M_I32: return "i32";
    case M_FLT: return "f32|bf16";
    default: return "?";
    }
}

/* Check one operand: valid view, owned by `be`, dtype in `mask`, at least `need` elements.
 * need == 0 accepts any (even empty) buffer of an allowed dtype. */
static rrc_status chk_buf(const rrc_backend *be, const char *op, const char *arg, rrc_buf b, unsigned mask,
                          int64_t need) {
    RRC_REQUIRE(b.numel >= 0, RRC_ERR_SHAPE, "%s: %s is an invalid (out-of-range) view", op, arg);
    if (need > 0) {
        RRC_REQUIRE(b.ptr != NULL, RRC_ERR_INVALID_ARG, "%s: %s has a NULL pointer", op, arg);
        RRC_REQUIRE(b.be == be, RRC_ERR_INVALID_ARG, "%s: %s belongs to backend '%s', op runs on '%s'", op, arg,
                    b.be ? b.be->name : "(none)", be->name);
    }
    RRC_REQUIRE(rrc_dtype_valid((int)b.dtype) && ((1u << b.dtype) & mask), RRC_ERR_DTYPE, "%s: %s has dtype %s, expected %s",
                op, arg, rrc_dtype_name(b.dtype), mask_str(mask));
    RRC_REQUIRE(b.numel >= need, RRC_ERR_SHAPE, "%s: %s needs %lld elements, view holds %lld", op, arg, (long long)need,
                (long long)b.numel);
    return RRC_OK;
}

/* Byte ranges [ptr, ptr + elems*size) intersect? */
static int overlaps(rrc_buf a, int64_t ea, rrc_buf b, int64_t eb) {
    if (ea <= 0 || eb <= 0 || !a.ptr || !b.ptr) return 0;
    uintptr_t a0 = (uintptr_t)a.ptr, a1 = a0 + (uintptr_t)ea * rrc_dtype_size(a.dtype);
    uintptr_t b0 = (uintptr_t)b.ptr, b1 = b0 + (uintptr_t)eb * rrc_dtype_size(b.dtype);
    return a0 < b1 && b0 < a1;
}

/* Output `o` vs operand `i`: overlap allowed only as an exact alias when `allow_exact`
 * (same ptr, same dtype; the caller guarantees equal leading dimensions via `same_ld`). */
static rrc_status chk_alias(const char *op, const char *oname, rrc_buf o, int64_t eo, const char *iname, rrc_buf i,
                            int64_t ei, int allow_exact, int same_ld) {
    if (!overlaps(o, eo, i, ei)) return RRC_OK;
    if (allow_exact && o.ptr == i.ptr && o.dtype == i.dtype && same_ld && eo == ei) return RRC_OK;
    if (allow_exact)
        return RRC_ERROR(RRC_ERR_ALIAS, "%s: %s overlaps %s (only an exact alias: same pointer, dtype and layout is "
                                        "allowed)",
                         op, oname, iname);
    return RRC_ERROR(RRC_ERR_ALIAS, "%s: %s must not overlap %s", op, oname, iname);
}

/* ------------------------------------------------------------------------------------------ */

rrc_status rrc_sync(const rrc_backend *be) {
    RRC_CHECK(chk_be(be, "sync"));
    return be->sync(be);
}

rrc_status rrc_gemm(const rrc_backend *be, const rrc_gemm_args *g) {
    const char *op = "gemm";
    RRC_CHECK(chk_be(be, op));
    RRC_REQUIRE(g, RRC_ERR_INVALID_ARG, "gemm: args is NULL");
    RRC_REQUIRE((g->trans_a == RRC_OP_N || g->trans_a == RRC_OP_T) && (g->trans_b == RRC_OP_N || g->trans_b == RRC_OP_T),
                RRC_ERR_INVALID_ARG, "gemm: invalid transpose flag");
    RRC_CHECK(chk_nonneg(op, "m", g->m));
    RRC_CHECK(chk_nonneg(op, "n", g->n));
    RRC_CHECK(chk_nonneg(op, "k", g->k));
    RRC_REQUIRE(g->batch >= 1, RRC_ERR_SHAPE, "gemm: batch = %lld must be >= 1", (long long)g->batch);
    RRC_REQUIRE(g->stride_a >= 0 && g->stride_b >= 0 && g->stride_c >= 0, RRC_ERR_SHAPE, "gemm: negative batch stride");
    /* stored shapes */
    int64_t ar = g->trans_a == RRC_OP_N ? g->m : g->k, ac = g->trans_a == RRC_OP_N ? g->k : g->m;
    int64_t br = g->trans_b == RRC_OP_N ? g->k : g->n, bc = g->trans_b == RRC_OP_N ? g->n : g->k;
    RRC_REQUIRE(g->lda >= (ac > 1 ? ac : 1), RRC_ERR_SHAPE, "gemm: lda = %lld < %lld (A stored %lld x %lld, trans_a=%d)",
                (long long)g->lda, (long long)ac, (long long)ar, (long long)ac, (int)g->trans_a);
    RRC_REQUIRE(g->ldb >= (bc > 1 ? bc : 1), RRC_ERR_SHAPE, "gemm: ldb = %lld < %lld (B stored %lld x %lld, trans_b=%d)",
                (long long)g->ldb, (long long)bc, (long long)br, (long long)bc, (int)g->trans_b);
    RRC_REQUIRE(g->ldc >= (g->n > 1 ? g->n : 1), RRC_ERR_SHAPE, "gemm: ldc = %lld < n = %lld", (long long)g->ldc,
                (long long)g->n);
    int64_t ea = 0, eb = 0, ec = 0;
    int need_ab = g->m > 0 && g->n > 0 && g->k > 0;
    if (need_ab) {
        RRC_CHECK(extent(op, g->batch, g->stride_a, ar, ac, g->lda, &ea));
        RRC_CHECK(extent(op, g->batch, g->stride_b, br, bc, g->ldb, &eb));
    }
    RRC_CHECK(extent(op, g->batch, g->stride_c, g->m, g->n, g->ldc, &ec));
    RRC_CHECK(chk_buf(be, op, "A", g->a, M_FLT, ea));
    RRC_CHECK(chk_buf(be, op, "B", g->b, M_FLT, eb));
    RRC_CHECK(chk_buf(be, op, "C", g->c, M_FLT, ec));
    if (g->batch > 1 && g->m > 0 && g->n > 0) {
        int64_t slice;
        RRC_CHECK(extent(op, 1, 0, g->m, g->n, g->ldc, &slice));
        int disjoint = g->stride_c >= slice;
        int interleaved = g->n <= g->stride_c && (g->batch - 1) * g->stride_c + g->n <= g->ldc;
        RRC_REQUIRE(disjoint || interleaved, RRC_ERR_ALIAS,
                    "gemm: C batch slices overlap (stride_c = %lld, ldc = %lld, m = %lld, n = %lld, batch = %lld)",
                    (long long)g->stride_c, (long long)g->ldc, (long long)g->m, (long long)g->n, (long long)g->batch);
    }
    RRC_CHECK(chk_alias(op, "C", g->c, ec, "A", g->a, ea, 0, 0));
    RRC_CHECK(chk_alias(op, "C", g->c, ec, "B", g->b, eb, 0, 0));
    if (ec == 0) return RRC_OK;
    return be->gemm(be, g);
}

rrc_status rrc_gemm2d(const rrc_backend *be, rrc_trans ta, rrc_trans tb, int64_t m, int64_t n, int64_t k, float alpha,
                      rrc_buf a, int64_t lda, rrc_buf b, int64_t ldb, float beta, rrc_buf c, int64_t ldc) {
    rrc_gemm_args g;
    g.trans_a = ta;
    g.trans_b = tb;
    g.m = m;
    g.n = n;
    g.k = k;
    g.alpha = alpha;
    g.beta = beta;
    g.a = a;
    g.lda = lda;
    g.stride_a = 0;
    g.b = b;
    g.ldb = ldb;
    g.stride_b = 0;
    g.c = c;
    g.ldc = ldc;
    g.stride_c = 0;
    g.batch = 1;
    return rrc_gemm(be, &g);
}

rrc_status rrc_embedding_fwd(const rrc_backend *be, int64_t n_tokens, int64_t dim, int64_t vocab, rrc_buf ids,
                             rrc_buf table, rrc_buf out) {
    const char *op = "embedding_fwd";
    RRC_CHECK(chk_be(be, op));
    RRC_CHECK(chk_nonneg(op, "n_tokens", n_tokens));
    RRC_CHECK(chk_nonneg(op, "dim", dim));
    RRC_REQUIRE(vocab > 0, RRC_ERR_SHAPE, "embedding_fwd: vocab must be > 0");
    int64_t et, eo;
    RRC_CHECK(mul_ck(op, vocab, dim, &et));
    RRC_CHECK(mul_ck(op, n_tokens, dim, &eo));
    if (eo == 0) return RRC_OK;
    RRC_CHECK(chk_buf(be, op, "ids", ids, M_I32, n_tokens));
    RRC_CHECK(chk_buf(be, op, "table", table, M_FLT, et));
    RRC_CHECK(chk_buf(be, op, "out", out, M_FLT, eo));
    RRC_CHECK(chk_alias(op, "out", out, eo, "table", table, et, 0, 0));
    RRC_CHECK(chk_alias(op, "out", out, eo, "ids", ids, n_tokens, 0, 0));
    return be->embedding_fwd(be, n_tokens, dim, vocab, ids, table, out);
}

rrc_status rrc_add_bias(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf x, int64_t ldx, rrc_buf bias) {
    const char *op = "add_bias";
    RRC_CHECK(chk_be(be, op));
    RRC_CHECK(chk_nonneg(op, "rows", rows));
    RRC_CHECK(chk_nonneg(op, "cols", cols));
    RRC_REQUIRE(ldx >= cols, RRC_ERR_SHAPE, "add_bias: ldx < cols");
    int64_t ex;
    RRC_CHECK(extent(op, 1, 0, rows, cols, ldx, &ex));
    if (ex == 0) return RRC_OK;
    RRC_CHECK(chk_buf(be, op, "x", x, M_FLT, ex));
    RRC_CHECK(chk_buf(be, op, "bias", bias, M_FLT, cols));
    RRC_CHECK(chk_alias(op, "x", x, ex, "bias", bias, cols, 0, 0));
    return be->add_bias(be, rows, cols, x, ldx, bias);
}

rrc_status rrc_bias_grad(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf dy, int64_t ldy, rrc_buf dbias,
                         int accumulate) {
    const char *op = "bias_grad";
    RRC_CHECK(chk_be(be, op));
    RRC_CHECK(chk_nonneg(op, "rows", rows));
    RRC_CHECK(chk_nonneg(op, "cols", cols));
    RRC_REQUIRE(ldy >= cols, RRC_ERR_SHAPE, "bias_grad: ldy < cols");
    if (cols == 0) return RRC_OK;
    int64_t ey;
    RRC_CHECK(extent(op, 1, 0, rows, cols, ldy, &ey));
    RRC_CHECK(chk_buf(be, op, "dy", dy, M_FLT, ey));
    RRC_CHECK(chk_buf(be, op, "dbias", dbias, M_F32, cols));
    RRC_CHECK(chk_alias(op, "dbias", dbias, cols, "dy", dy, ey, 0, 0));
    return be->bias_grad(be, rows, cols, dy, ldy, dbias, accumulate ? 1 : 0);
}

rrc_status rrc_rmsnorm_fwd(const rrc_backend *be, int64_t rows, int64_t dim, int64_t ld, rrc_buf x, rrc_buf w,
                           float eps, rrc_buf y) {
    const char *op = "rmsnorm_fwd";
    RRC_CHECK(chk_be(be, op));
    RRC_CHECK(chk_nonneg(op, "rows", rows));
    RRC_REQUIRE(dim > 0, RRC_ERR_SHAPE, "rmsnorm_fwd: dim must be > 0");
    RRC_REQUIRE(ld >= dim, RRC_ERR_SHAPE, "rmsnorm_fwd: ld < dim");
    RRC_REQUIRE(eps > 0.0f, RRC_ERR_INVALID_ARG, "rmsnorm_fwd: eps must be > 0");
    int64_t ex;
    RRC_CHECK(extent(op, 1, 0, rows, dim, ld, &ex));
    if (ex == 0) return RRC_OK;
    RRC_CHECK(chk_buf(be, op, "x", x, M_FLT, ex));
    RRC_CHECK(chk_buf(be, op, "w", w, M_FLT, dim));
    RRC_CHECK(chk_buf(be, op, "y", y, M_FLT, ex));
    RRC_CHECK(chk_alias(op, "y", y, ex, "x", x, ex, 1, 1));
    RRC_CHECK(chk_alias(op, "y", y, ex, "w", w, dim, 0, 0));
    return be->rmsnorm_fwd(be, rows, dim, ld, x, w, eps, y);
}

rrc_status rrc_rmsnorm_bwd(const rrc_backend *be, int64_t rows, int64_t dim, int64_t ld, rrc_buf x, rrc_buf w,
                           float eps, rrc_buf dy, rrc_buf dx, int accumulate) {
    const char *op = "rmsnorm_bwd";
    RRC_CHECK(chk_be(be, op));
    RRC_CHECK(chk_nonneg(op, "rows", rows));
    RRC_REQUIRE(dim > 0, RRC_ERR_SHAPE, "rmsnorm_bwd: dim must be > 0");
    RRC_REQUIRE(ld >= dim, RRC_ERR_SHAPE, "rmsnorm_bwd: ld < dim");
    RRC_REQUIRE(eps > 0.0f, RRC_ERR_INVALID_ARG, "rmsnorm_bwd: eps must be > 0");
    int64_t ex;
    RRC_CHECK(extent(op, 1, 0, rows, dim, ld, &ex));
    if (ex == 0) return RRC_OK;
    RRC_CHECK(chk_buf(be, op, "x", x, M_FLT, ex));
    RRC_CHECK(chk_buf(be, op, "w", w, M_FLT, dim));
    RRC_CHECK(chk_buf(be, op, "dy", dy, M_FLT, ex));
    RRC_CHECK(chk_buf(be, op, "dx", dx, M_FLT, ex));
    int exact_ok = accumulate ? 0 : 1;
    RRC_CHECK(chk_alias(op, "dx", dx, ex, "x", x, ex, exact_ok, 1));
    RRC_CHECK(chk_alias(op, "dx", dx, ex, "dy", dy, ex, exact_ok, 1));
    RRC_CHECK(chk_alias(op, "dx", dx, ex, "w", w, dim, 0, 0));
    return be->rmsnorm_bwd(be, rows, dim, ld, x, w, eps, dy, dx, accumulate ? 1 : 0);
}

static rrc_status rope_check(const rrc_backend *be, const char *op, int64_t n_tokens, int64_t n_heads,
                             int64_t head_dim, int64_t ld, rrc_buf x, rrc_buf pos, rrc_buf cos_tab, rrc_buf sin_tab,
                             int64_t n_pos, rrc_buf y, int64_t *ex_out) {
    RRC_CHECK(chk_be(be, op));
    RRC_CHECK(chk_nonneg(op, "n_tokens", n_tokens));
    RRC_CHECK(chk_nonneg(op, "n_heads", n_heads));
    RRC_REQUIRE(head_dim > 0 && head_dim % 2 == 0, RRC_ERR_SHAPE, "%s: head_dim = %lld must be even and > 0", op,
                (long long)head_dim);
    int64_t row;
    RRC_CHECK(mul_ck(op, n_heads, head_dim, &row));
    RRC_REQUIRE(ld >= row, RRC_ERR_SHAPE, "%s: ld = %lld < n_heads*head_dim = %lld", op, (long long)ld, (long long)row);
    RRC_REQUIRE(n_pos > 0, RRC_ERR_SHAPE, "%s: n_pos must be > 0", op);
    int64_t ex, et;
    RRC_CHECK(extent(op, 1, 0, n_tokens, row, ld, &ex));
    *ex_out = ex;
    if (ex == 0) return RRC_OK;
    RRC_CHECK(mul_ck(op, n_pos, head_dim / 2, &et));
    RRC_CHECK(chk_buf(be, op, "x", x, M_FLT, ex));
    RRC_CHECK(chk_buf(be, op, "pos", pos, M_I32, n_tokens));
    RRC_CHECK(chk_buf(be, op, "cos_tab", cos_tab, M_F32, et));
    RRC_CHECK(chk_buf(be, op, "sin_tab", sin_tab, M_F32, et));
    RRC_CHECK(chk_buf(be, op, "out", y, M_FLT, ex));
    RRC_CHECK(chk_alias(op, "out", y, ex, "in", x, ex, 1, 1));
    RRC_CHECK(chk_alias(op, "out", y, ex, "pos", pos, n_tokens, 0, 0));
    RRC_CHECK(chk_alias(op, "out", y, ex, "cos_tab", cos_tab, et, 0, 0));
    RRC_CHECK(chk_alias(op, "out", y, ex, "sin_tab", sin_tab, et, 0, 0));
    return RRC_OK;
}

rrc_status rrc_rope_fwd(const rrc_backend *be, int64_t n_tokens, int64_t n_heads, int64_t head_dim, int64_t ld,
                        rrc_buf x, rrc_buf pos, rrc_buf cos_tab, rrc_buf sin_tab, int64_t n_pos, rrc_buf y) {
    int64_t ex;
    RRC_CHECK(rope_check(be, "rope_fwd", n_tokens, n_heads, head_dim, ld, x, pos, cos_tab, sin_tab, n_pos, y, &ex));
    if (ex == 0) return RRC_OK;
    return be->rope_fwd(be, n_tokens, n_heads, head_dim, ld, x, pos, cos_tab, sin_tab, n_pos, y);
}

rrc_status rrc_rope_bwd(const rrc_backend *be, int64_t n_tokens, int64_t n_heads, int64_t head_dim, int64_t ld,
                        rrc_buf dy, rrc_buf pos, rrc_buf cos_tab, rrc_buf sin_tab, int64_t n_pos, rrc_buf dx) {
    int64_t ex;
    RRC_CHECK(rope_check(be, "rope_bwd", n_tokens, n_heads, head_dim, ld, dy, pos, cos_tab, sin_tab, n_pos, dx, &ex));
    if (ex == 0) return RRC_OK;
    return be->rope_bwd(be, n_tokens, n_heads, head_dim, ld, dy, pos, cos_tab, sin_tab, n_pos, dx);
}

static rrc_status attn_layout(const char *op, int64_t batch, int64_t rows, int64_t cols, int64_t ld,
                              int64_t batch_stride, int64_t *ex) {
    RRC_CHECK(chk_nonneg(op, "batch", batch));
    RRC_CHECK(chk_nonneg(op, "rows", rows));
    RRC_CHECK(chk_nonneg(op, "cols", cols));
    RRC_REQUIRE(ld >= cols, RRC_ERR_SHAPE, "%s: ld = %lld < cols = %lld", op, (long long)ld, (long long)cols);
    if (batch > 1 && rows > 0 && cols > 0) {
        int64_t slice;
        RRC_CHECK(extent(op, 1, 0, rows, cols, ld, &slice));
        RRC_REQUIRE(batch_stride >= slice, RRC_ERR_SHAPE, "%s: batch_stride = %lld < rows*ld extent %lld", op,
                    (long long)batch_stride, (long long)slice);
    }
    return extent(op, batch, batch_stride, rows, cols, ld, ex);
}

rrc_status rrc_attn_softmax_fwd(const rrc_backend *be, int64_t batch, int64_t rows, int64_t cols, int64_t ld,
                                int64_t batch_stride, rrc_buf s, float scale, int64_t row_offset, int causal,
                                rrc_buf doc_ids, rrc_buf p) {
    const char *op = "attn_softmax_fwd";
    RRC_CHECK(chk_be(be, op));
    int64_t ex;
    RRC_CHECK(attn_layout(op, batch, rows, cols, ld, batch_stride, &ex));
    RRC_CHECK(chk_nonneg(op, "row_offset", row_offset));
    if (ex == 0) return RRC_OK;
    RRC_CHECK(chk_buf(be, op, "s", s, M_FLT, ex));
    RRC_CHECK(chk_buf(be, op, "p", p, M_FLT, ex));
    int64_t nd = 0;
    if (!rrc_buf_is_none(doc_ids)) {
        RRC_CHECK(add_ck(op, row_offset, rows, &nd));
        if (cols > nd) nd = cols;
        RRC_CHECK(chk_buf(be, op, "doc_ids", doc_ids, M_I32, nd));
        RRC_CHECK(chk_alias(op, "p", p, ex, "doc_ids", doc_ids, nd, 0, 0));
    }
    RRC_CHECK(chk_alias(op, "p", p, ex, "s", s, ex, 1, 1));
    return be->attn_softmax_fwd(be, batch, rows, cols, ld, batch_stride, s, scale, row_offset, causal ? 1 : 0, doc_ids,
                                p);
}

rrc_status rrc_attn_softmax_bwd(const rrc_backend *be, int64_t batch, int64_t rows, int64_t cols, int64_t ld,
                                int64_t batch_stride, rrc_buf p, rrc_buf dp, float scale, rrc_buf ds) {
    const char *op = "attn_softmax_bwd";
    RRC_CHECK(chk_be(be, op));
    int64_t ex;
    RRC_CHECK(attn_layout(op, batch, rows, cols, ld, batch_stride, &ex));
    if (ex == 0) return RRC_OK;
    RRC_CHECK(chk_buf(be, op, "p", p, M_FLT, ex));
    RRC_CHECK(chk_buf(be, op, "dp", dp, M_FLT, ex));
    RRC_CHECK(chk_buf(be, op, "ds", ds, M_FLT, ex));
    RRC_CHECK(chk_alias(op, "ds", ds, ex, "p", p, ex, 1, 1));
    RRC_CHECK(chk_alias(op, "ds", ds, ex, "dp", dp, ex, 1, 1));
    return be->attn_softmax_bwd(be, batch, rows, cols, ld, batch_stride, p, dp, scale, ds);
}

rrc_status rrc_swiglu_fwd(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf gate, rrc_buf up, int64_t ld_in,
                          rrc_buf h, int64_t ld_out) {
    const char *op = "swiglu_fwd";
    RRC_CHECK(chk_be(be, op));
    RRC_CHECK(chk_nonneg(op, "rows", rows));
    RRC_CHECK(chk_nonneg(op, "cols", cols));
    RRC_REQUIRE(ld_in >= cols && ld_out >= cols, RRC_ERR_SHAPE, "swiglu_fwd: leading dimension < cols");
    int64_t ei, eo;
    RRC_CHECK(extent(op, 1, 0, rows, cols, ld_in, &ei));
    RRC_CHECK(extent(op, 1, 0, rows, cols, ld_out, &eo));
    if (eo == 0) return RRC_OK;
    RRC_CHECK(chk_buf(be, op, "gate", gate, M_FLT, ei));
    RRC_CHECK(chk_buf(be, op, "up", up, M_FLT, ei));
    RRC_CHECK(chk_buf(be, op, "h", h, M_FLT, eo));
    RRC_CHECK(chk_alias(op, "h", h, eo, "gate", gate, ei, 1, ld_in == ld_out));
    RRC_CHECK(chk_alias(op, "h", h, eo, "up", up, ei, 1, ld_in == ld_out));
    return be->swiglu_fwd(be, rows, cols, gate, up, ld_in, h, ld_out);
}

rrc_status rrc_swiglu_bwd(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf gate, rrc_buf up, int64_t ld_in,
                          rrc_buf dh, int64_t ld_dh, rrc_buf dgate, rrc_buf dup, int64_t ld_out) {
    const char *op = "swiglu_bwd";
    RRC_CHECK(chk_be(be, op));
    RRC_CHECK(chk_nonneg(op, "rows", rows));
    RRC_CHECK(chk_nonneg(op, "cols", cols));
    RRC_REQUIRE(ld_in >= cols && ld_dh >= cols && ld_out >= cols, RRC_ERR_SHAPE, "swiglu_bwd: leading dimension < cols");
    int64_t ei, ed, eo;
    RRC_CHECK(extent(op, 1, 0, rows, cols, ld_in, &ei));
    RRC_CHECK(extent(op, 1, 0, rows, cols, ld_dh, &ed));
    RRC_CHECK(extent(op, 1, 0, rows, cols, ld_out, &eo));
    if (eo == 0) return RRC_OK;
    RRC_CHECK(chk_buf(be, op, "gate", gate, M_FLT, ei));
    RRC_CHECK(chk_buf(be, op, "up", up, M_FLT, ei));
    RRC_CHECK(chk_buf(be, op, "dh", dh, M_FLT, ed));
    RRC_CHECK(chk_buf(be, op, "dgate", dgate, M_FLT, eo));
    RRC_CHECK(chk_buf(be, op, "dup", dup, M_FLT, eo));
    RRC_CHECK(chk_alias(op, "dgate", dgate, eo, "dup", dup, eo, 0, 0));
    RRC_CHECK(chk_alias(op, "dgate", dgate, eo, "gate", gate, ei, 1, ld_in == ld_out));
    RRC_CHECK(chk_alias(op, "dgate", dgate, eo, "up", up, ei, 0, 0));
    RRC_CHECK(chk_alias(op, "dgate", dgate, eo, "dh", dh, ed, 1, ld_dh == ld_out));
    RRC_CHECK(chk_alias(op, "dup", dup, eo, "up", up, ei, 1, ld_in == ld_out));
    RRC_CHECK(chk_alias(op, "dup", dup, eo, "gate", gate, ei, 0, 0));
    RRC_CHECK(chk_alias(op, "dup", dup, eo, "dh", dh, ed, 1, ld_dh == ld_out));
    return be->swiglu_bwd(be, rows, cols, gate, up, ld_in, dh, ld_dh, dgate, dup, ld_out);
}

rrc_status rrc_add(const rrc_backend *be, int64_t n, rrc_buf a, rrc_buf b, rrc_buf out) {
    const char *op = "add";
    RRC_CHECK(chk_be(be, op));
    RRC_CHECK(chk_nonneg(op, "n", n));
    if (n == 0) return RRC_OK;
    RRC_CHECK(chk_buf(be, op, "a", a, M_FLT, n));
    RRC_CHECK(chk_buf(be, op, "b", b, M_FLT, n));
    RRC_CHECK(chk_buf(be, op, "out", out, M_FLT, n));
    RRC_CHECK(chk_alias(op, "out", out, n, "a", a, n, 1, 1));
    RRC_CHECK(chk_alias(op, "out", out, n, "b", b, n, 1, 1));
    return be->add(be, n, a, b, out);
}

rrc_status rrc_axpy(const rrc_backend *be, int64_t n, float alpha, rrc_buf x, rrc_buf y) {
    const char *op = "axpy";
    RRC_CHECK(chk_be(be, op));
    RRC_CHECK(chk_nonneg(op, "n", n));
    if (n == 0) return RRC_OK;
    RRC_CHECK(chk_buf(be, op, "x", x, M_FLT, n));
    RRC_CHECK(chk_buf(be, op, "y", y, M_FLT, n));
    RRC_CHECK(chk_alias(op, "y", y, n, "x", x, n, 1, 1));
    return be->axpy(be, n, alpha, x, y);
}

rrc_status rrc_scale(const rrc_backend *be, int64_t n, float alpha, rrc_buf x) {
    const char *op = "scale";
    RRC_CHECK(chk_be(be, op));
    RRC_CHECK(chk_nonneg(op, "n", n));
    if (n == 0) return RRC_OK;
    RRC_CHECK(chk_buf(be, op, "x", x, M_FLT, n));
    return be->scale(be, n, alpha, x);
}

rrc_status rrc_convert(const rrc_backend *be, int64_t n, rrc_buf src, rrc_buf dst) {
    const char *op = "convert";
    RRC_CHECK(chk_be(be, op));
    RRC_CHECK(chk_nonneg(op, "n", n));
    if (n == 0) return RRC_OK;
    unsigned mask = (src.dtype == RRC_I32) ? M_I32 : M_FLT;
    RRC_CHECK(chk_buf(be, op, "src", src, M_FLT | M_I32, n));
    RRC_CHECK(chk_buf(be, op, "dst", dst, mask, n));
    if (src.ptr == dst.ptr && src.dtype == dst.dtype) return RRC_OK; /* exact alias: no-op */
    RRC_CHECK(chk_alias(op, "dst", dst, n, "src", src, n, 0, 0));
    return be->convert(be, n, src, dst);
}

rrc_status rrc_sum_squares(const rrc_backend *be, int64_t n, rrc_buf x, rrc_buf out, int accumulate) {
    const char *op = "sum_squares";
    RRC_CHECK(chk_be(be, op));
    RRC_CHECK(chk_nonneg(op, "n", n));
    RRC_CHECK(chk_buf(be, op, "x", x, M_FLT, n));
    RRC_CHECK(chk_buf(be, op, "out", out, M_F32, 1));
    RRC_CHECK(chk_alias(op, "out", out, 1, "x", x, n, 0, 0));
    return be->sum_squares(be, n, x, out, accumulate ? 1 : 0);
}

rrc_status rrc_cross_entropy_fwd_bwd(const rrc_backend *be, int64_t rows, int64_t vocab, rrc_buf logits, int64_t ld,
                                     rrc_buf labels, rrc_buf loss, float grad_scale, rrc_buf row_scale,
                                     int write_grad) {
    const char *op = "cross_entropy_fwd_bwd";
    RRC_CHECK(chk_be(be, op));
    RRC_CHECK(chk_nonneg(op, "rows", rows));
    RRC_REQUIRE(vocab > 0, RRC_ERR_SHAPE, "cross_entropy_fwd_bwd: vocab must be > 0");
    RRC_REQUIRE(vocab <= INT32_MAX, RRC_ERR_SHAPE, "cross_entropy_fwd_bwd: vocab exceeds int32 labels");
    RRC_REQUIRE(ld >= vocab, RRC_ERR_SHAPE, "cross_entropy_fwd_bwd: ld < vocab");
    if (rows == 0) return RRC_OK;
    int64_t el;
    RRC_CHECK(extent(op, 1, 0, rows, vocab, ld, &el));
    RRC_CHECK(chk_buf(be, op, "logits", logits, M_FLT, el));
    RRC_CHECK(chk_buf(be, op, "labels", labels, M_I32, rows));
    RRC_CHECK(chk_buf(be, op, "loss", loss, M_F32, rows));
    RRC_CHECK(chk_alias(op, "loss", loss, rows, "logits", logits, el, 0, 0));
    RRC_CHECK(chk_alias(op, "loss", loss, rows, "labels", labels, rows, 0, 0));
    if (!rrc_buf_is_none(row_scale)) {
        RRC_CHECK(chk_buf(be, op, "row_scale", row_scale, M_F32, rows));
        RRC_CHECK(chk_alias(op, "loss", loss, rows, "row_scale", row_scale, rows, 0, 0));
        if (write_grad) RRC_CHECK(chk_alias(op, "logits", logits, el, "row_scale", row_scale, rows, 0, 0));
    }
    if (write_grad) RRC_CHECK(chk_alias(op, "logits", logits, el, "labels", labels, rows, 0, 0));
    return be->cross_entropy_fwd_bwd(be, rows, vocab, logits, ld, labels, loss, grad_scale, row_scale,
                                     write_grad ? 1 : 0);
}

rrc_status rrc_adamw_step(const rrc_backend *be, int64_t n, rrc_buf param, rrc_buf grad, rrc_buf m, rrc_buf v,
                          const rrc_adamw_hparams *h) {
    const char *op = "adamw_step";
    RRC_CHECK(chk_be(be, op));
    RRC_REQUIRE(h, RRC_ERR_INVALID_ARG, "adamw_step: hparams is NULL");
    RRC_CHECK(chk_nonneg(op, "n", n));
    RRC_REQUIRE(h->step >= 1, RRC_ERR_INVALID_ARG, "adamw_step: step = %lld must be >= 1", (long long)h->step);
    RRC_REQUIRE(h->lr >= 0.0 && h->eps >= 0.0 && h->weight_decay >= 0.0, RRC_ERR_INVALID_ARG,
                "adamw_step: lr, eps and weight_decay must be >= 0");
    RRC_REQUIRE(h->beta1 >= 0.0 && h->beta1 < 1.0 && h->beta2 >= 0.0 && h->beta2 < 1.0, RRC_ERR_INVALID_ARG,
                "adamw_step: betas must be in [0, 1)");
    if (n == 0) return RRC_OK;
    RRC_CHECK(chk_buf(be, op, "param", param, M_F32, n));
    RRC_CHECK(chk_buf(be, op, "grad", grad, M_F32, n));
    RRC_CHECK(chk_buf(be, op, "m", m, M_F32, n));
    RRC_CHECK(chk_buf(be, op, "v", v, M_F32, n));
    const char *names[4] = {"param", "grad", "m", "v"};
    rrc_buf bufs[4] = {param, grad, m, v};
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 4; j++) RRC_CHECK(chk_alias(op, names[i], bufs[i], n, names[j], bufs[j], n, 0, 0));
    return be->adamw_step(be, n, param, grad, m, v, h);
}
