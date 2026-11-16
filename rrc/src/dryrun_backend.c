/* rrc/src/dryrun_backend.c — the recording dry-run backend (include/rrc/dryrun.h). */
#include "rrc/dryrun.h"

#include <stdlib.h>
#include <string.h>

#include "gemm_plan.h"

/* Fake device addresses: the upper (kernel) half of the address space, never dereferenced. The
 * first RRC_DR_SCRATCH_SPAN bytes stand for the CUDA backend's GEMM scratch (only used to form the
 * plan's chunk pointers for the alignment report); allocations start after it. */
#define RRC_DR_BASE 0xffff800000000000ull
#define RRC_DR_SCRATCH_SPAN (1ull << 36)
#define RRC_DR_ALIGN 256ull
#define RRC_DR_DEFAULT_CAPACITY (1ull << 44)

typedef struct dr_alloc {
    uint64_t addr, size;
} dr_alloc;

typedef struct dr_impl {
    rrc_dryrun_opts o;
    uint64_t next;          /* next free fake address (offset from RRC_DR_BASE) */
    dr_alloc *live;
    size_t n_live, cap_live;
    rrc_dryrun_stats st;
    rrc_dryrun_hook hook;
    void *hook_ctx;
} dr_impl;

static const char *const k_op_names[RRC_DR_N_OPS] = {
    "alloc",       "free",       "memcpy_h2d",       "memcpy_d2h",       "memcpy_d2d", "memset0",
    "sync",        "gemm",       "embedding_fwd",    "add_bias",         "bias_grad",  "rmsnorm_fwd",
    "rmsnorm_bwd", "rope_fwd",   "rope_bwd",         "attn_softmax_fwd", "attn_softmax_bwd",
    "swiglu_fwd",  "swiglu_bwd", "add",              "axpy",             "scale",      "convert",
    "sum_squares", "cross_entropy_fwd_bwd",          "adamw_step"};

const char *rrc_dryrun_op_name(int op) { return op >= 0 && op < RRC_DR_N_OPS ? k_op_names[op] : NULL; }

void rrc_dryrun_opts_default(rrc_dryrun_opts *o) {
    memset(o, 0, sizeof *o);
    o->gemm_scratch_cap_elems = 0;
    o->allow_tf32 = 0;
    o->capacity_bytes = 0;
    o->limit_bytes = 0;
}

static dr_impl *IMPL(const rrc_backend *be) { return (dr_impl *)be->impl; }

static uint64_t es(rrc_dtype d) { return (uint64_t)rrc_dtype_size(d); }
static uint64_t u(int64_t v) { return v > 0 ? (uint64_t)v : 0; }

/* Record one call: counters, then the hook. */
static void record(const rrc_backend *be, rrc_dryrun_call *c) {
    dr_impl *d = IMPL(be);
    d->st.calls[c->op]++;
    d->st.bytes[c->op] += c->bytes;
    if (d->hook) d->hook(d->hook_ctx, c);
}

static void call_init(rrc_dryrun_call *c, rrc_dryrun_op op) {
    memset(c, 0, sizeof *c);
    c->op = op;
}
static void add_shape(rrc_dryrun_call *c, int64_t v) {
    if (c->n_shape < 4) c->shape[c->n_shape++] = v;
}
static void add_dtype(rrc_dryrun_call *c, rrc_buf b) {
    if (c->n_dtype < 6 && !rrc_buf_is_none(b)) c->dtype[c->n_dtype++] = b.dtype;
}

/* ---- memory ----------------------------------------------------------------------------------- */

static rrc_status dr_alloc_fn(const rrc_backend *be, size_t bytes, void **out) {
    dr_impl *d = IMPL(be);
    *out = NULL;
    if (bytes == 0) return RRC_OK;
    const uint64_t cap = d->o.capacity_bytes ? d->o.capacity_bytes : RRC_DR_DEFAULT_CAPACITY;
    const uint64_t size = ((uint64_t)bytes + RRC_DR_ALIGN - 1) & ~(RRC_DR_ALIGN - 1);
    if ((uint64_t)bytes > cap || size > cap - d->next)
        return RRC_ERROR(RRC_ERR_OOM, "dry-run backend: fake address space of %llu bytes exhausted (%zu requested)",
                         (unsigned long long)cap, bytes);
    /* bytes <= cap <= 2^44 and live <= cap: the sum cannot overflow */
    if (d->o.limit_bytes && d->st.live_bytes + (uint64_t)bytes > d->o.limit_bytes)
        return RRC_ERROR(RRC_ERR_OOM, "dry-run backend: %zu bytes requested, %llu live, limit %llu", bytes,
                         (unsigned long long)d->st.live_bytes, (unsigned long long)d->o.limit_bytes);
    if (d->n_live == d->cap_live) {
        const size_t nc = d->cap_live ? 2 * d->cap_live : 64;
        dr_alloc *p = (dr_alloc *)realloc(d->live, nc * sizeof *p);
        if (!p) return RRC_ERROR(RRC_ERR_OOM, "dry-run backend: allocation table");
        d->live = p;
        d->cap_live = nc;
    }
    const uint64_t addr = RRC_DR_BASE + RRC_DR_SCRATCH_SPAN + d->next;
    d->live[d->n_live].addr = addr;
    d->live[d->n_live].size = (uint64_t)bytes;
    d->n_live++;
    d->next += size;
    d->st.live_bytes += (uint64_t)bytes;
    if (d->st.live_bytes > d->st.peak_bytes) d->st.peak_bytes = d->st.live_bytes;
    d->st.live_allocs++;
    d->st.n_alloc++;
    *out = (void *)(uintptr_t)addr;
    rrc_dryrun_call c;
    call_init(&c, RRC_DR_ALLOC);
    add_shape(&c, (int64_t)bytes);
    record(be, &c);
    return RRC_OK;
}

static void dr_free_fn(const rrc_backend *be, void *ptr) {
    dr_impl *d = IMPL(be);
    if (!ptr) return;
    const uint64_t a = (uint64_t)(uintptr_t)ptr;
    for (size_t i = d->n_live; i-- > 0;)
        if (d->live[i].addr == a) {
            d->st.live_bytes -= d->live[i].size;
            d->st.live_allocs--;
            d->st.n_free++;
            rrc_dryrun_call c;
            call_init(&c, RRC_DR_FREE);
            add_shape(&c, (int64_t)d->live[i].size);
            d->live[i] = d->live[--d->n_live];
            record(be, &c);
            return;
        }
    rrc_log(RRC_LOG_WARN, "dry-run backend: free of an unknown handle %p ignored", ptr);
}

static rrc_status dr_xfer(const rrc_backend *be, rrc_dryrun_op op, size_t bytes) {
    dr_impl *d = IMPL(be);
    if (op == RRC_DR_H2D) d->st.h2d_bytes += bytes;
    else if (op == RRC_DR_D2H) d->st.d2h_bytes += bytes;
    else if (op == RRC_DR_D2D) d->st.d2d_bytes += bytes;
    else d->st.memset_bytes += bytes;
    rrc_dryrun_call c;
    call_init(&c, op);
    add_shape(&c, (int64_t)bytes);
    c.bytes = (uint64_t)bytes;
    record(be, &c);
    return RRC_OK;
}

static rrc_status dr_h2d(const rrc_backend *be, void *dst, const void *src, size_t bytes) {
    (void)dst;
    (void)src;
    return dr_xfer(be, RRC_DR_H2D, bytes);
}
static rrc_status dr_d2h(const rrc_backend *be, void *dst, const void *src, size_t bytes) {
    (void)src;
    if (dst && bytes) memset(dst, 0, bytes); /* the host side is real memory: deterministic zeros */
    return dr_xfer(be, RRC_DR_D2H, bytes);
}
static rrc_status dr_d2d(const rrc_backend *be, void *dst, const void *src, size_t bytes) {
    (void)dst;
    (void)src;
    return dr_xfer(be, RRC_DR_D2D, bytes);
}
static rrc_status dr_memset0(const rrc_backend *be, void *ptr, size_t bytes) {
    (void)ptr;
    return dr_xfer(be, RRC_DR_MEMSET0, bytes);
}
static rrc_status dr_sync(const rrc_backend *be) {
    rrc_dryrun_call c;
    call_init(&c, RRC_DR_SYNC);
    record(be, &c);
    return RRC_OK;
}

/* ---- gemm --------------------------------------------------------------------------------------- */

static uint64_t slices(int64_t stride, int64_t batch) { return stride == 0 ? 1 : u(batch); }

static rrc_status dr_gemm(const rrc_backend *be, const rrc_gemm_args *g) {
    dr_impl *d = IMPL(be);
    rrc_gemm_plan_opts po;
    memset(&po, 0, sizeof po);
    po.scratch_cap_elems = d->o.gemm_scratch_cap_elems;
    po.allow_tf32 = d->o.allow_tf32;
    rrc_gemm_plan p;
    RRC_CHECK_CTX(rrc_gemm_plan_make(g, &po, RRC_GEMM_LEVEL_NATIVE, &p), "dry-run gemm plan");
    rrc_dryrun_gemm gi;
    memset(&gi, 0, sizeof gi);
    gi.args = *g;
    gi.promote_a = p.promote_a;
    gi.promote_b = p.promote_b;
    gi.c_scratch = p.c_scratch;
    gi.native = !p.promote_a && !p.promote_b && !p.c_scratch;
    gi.native_bf16 = gi.native && g->a.dtype == RRC_BF16 && g->b.dtype == RRC_BF16;
    gi.tf32 = p.tf32;
    gi.n_chunks = p.n_chunks;
    gi.scratch_bytes = u(p.scratch_elems) * 4u;
    gi.min_align = RRC_GEMM_MAX_ALIGN;
    gi.flops = 2.0 * (double)g->m * (double)g->n * (double)g->k * (double)g->batch;
    rrc_gemm_plan_describe(&p, gi.desc, sizeof gi.desc);
    /* walk the chunks with a fake scratch base: conversions and the alignment of every call */
    void *scratch = (void *)(uintptr_t)RRC_DR_BASE;
    for (int64_t i = 0; i < p.n_chunks; i++) {
        rrc_gemm_chunk ch;
        rrc_gemm_plan_chunk(&p, scratch, i, &ch);
        for (int j = 0; j < ch.n_pre + ch.has_post; j++) {
            const rrc_kc_conv2d_p *v = j < ch.n_pre ? &ch.pre[j] : &ch.post;
            const uint64_t n = u(v->rows) * u(v->cols) * u(v->batch);
            gi.convert_elems += n;
            gi.convert_bytes += n * ((v->src ? es(v->sdt) : 0) + es(v->ddt));
        }
        const unsigned a = ch.call.align_a < ch.call.align_b ? ch.call.align_a : ch.call.align_b;
        const unsigned m = a < ch.call.align_c ? a : ch.call.align_c;
        if (m < gi.min_align) gi.min_align = m;
    }
    rrc_dryrun_stats *s = &d->st;
    s->gemm_flops += gi.flops;
    if (gi.native) s->gemm_flops_native += gi.flops;
    if (gi.native_bf16) s->gemm_flops_native_bf16 += gi.flops;
    if (gi.scratch_bytes > s->gemm_scratch_max_bytes) s->gemm_scratch_max_bytes = gi.scratch_bytes;
    s->gemm_convert_elems += gi.convert_elems;
    s->gemm_convert_bytes += gi.convert_bytes;

    rrc_dryrun_call c;
    call_init(&c, RRC_DR_GEMM);
    add_shape(&c, g->m);
    add_shape(&c, g->n);
    add_shape(&c, g->k);
    add_shape(&c, g->batch);
    add_dtype(&c, g->a);
    add_dtype(&c, g->b);
    add_dtype(&c, g->c);
    c.bytes = u(g->m) * u(g->k) * slices(g->stride_a, g->batch) * es(g->a.dtype) +
              u(g->k) * u(g->n) * slices(g->stride_b, g->batch) * es(g->b.dtype) +
              u(g->m) * u(g->n) * u(g->batch) * es(g->c.dtype) * (g->beta != 0.0f ? 2u : 1u);
    c.gemm = &gi;
    c.ptr[0] = g->a.ptr;
    c.ptr[1] = g->b.ptr;
    c.ptr[2] = g->c.ptr;
    record(be, &c);
    return RRC_OK;
}

/* ---- the other ops: record shapes, dtypes and the computed bytes moved -------------------------- */

static rrc_status dr_embedding_fwd(const rrc_backend *be, int64_t n_tokens, int64_t dim, int64_t vocab, rrc_buf ids,
                                   rrc_buf table, rrc_buf out) {
    rrc_dryrun_call c;
    call_init(&c, RRC_DR_EMBEDDING_FWD);
    add_shape(&c, n_tokens);
    add_shape(&c, dim);
    add_shape(&c, vocab);
    add_dtype(&c, ids);
    add_dtype(&c, table);
    add_dtype(&c, out);
    c.bytes = u(n_tokens) * 4u + u(n_tokens) * u(dim) * (es(table.dtype) + es(out.dtype));
    record(be, &c);
    return RRC_OK;
}

static rrc_status dr_add_bias(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf x, int64_t ldx,
                              rrc_buf bias) {
    (void)ldx;
    rrc_dryrun_call c;
    call_init(&c, RRC_DR_ADD_BIAS);
    add_shape(&c, rows);
    add_shape(&c, cols);
    add_dtype(&c, x);
    add_dtype(&c, bias);
    c.bytes = 2u * u(rows) * u(cols) * es(x.dtype) + u(cols) * es(bias.dtype);
    record(be, &c);
    return RRC_OK;
}

static rrc_status dr_bias_grad(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf dy, int64_t ldy,
                               rrc_buf dbias, int accumulate) {
    (void)ldy;
    rrc_dryrun_call c;
    call_init(&c, RRC_DR_BIAS_GRAD);
    add_shape(&c, rows);
    add_shape(&c, cols);
    add_dtype(&c, dy);
    add_dtype(&c, dbias);
    c.bytes = u(rows) * u(cols) * es(dy.dtype) + u(cols) * 4u * (accumulate ? 2u : 1u);
    record(be, &c);
    return RRC_OK;
}

static rrc_status dr_rmsnorm_fwd(const rrc_backend *be, int64_t rows, int64_t dim, int64_t ld, rrc_buf x, rrc_buf w,
                                 float eps, rrc_buf y) {
    (void)ld;
    (void)eps;
    rrc_dryrun_call c;
    call_init(&c, RRC_DR_RMSNORM_FWD);
    add_shape(&c, rows);
    add_shape(&c, dim);
    add_dtype(&c, x);
    add_dtype(&c, w);
    add_dtype(&c, y);
    c.bytes = u(rows) * u(dim) * (es(x.dtype) + es(y.dtype)) + u(dim) * es(w.dtype);
    record(be, &c);
    return RRC_OK;
}

static rrc_status dr_rmsnorm_bwd(const rrc_backend *be, int64_t rows, int64_t dim, int64_t ld, rrc_buf x, rrc_buf w,
                                 float eps, rrc_buf dy, rrc_buf dx, int accumulate) {
    (void)ld;
    (void)eps;
    rrc_dryrun_call c;
    call_init(&c, RRC_DR_RMSNORM_BWD);
    add_shape(&c, rows);
    add_shape(&c, dim);
    add_dtype(&c, x);
    add_dtype(&c, w);
    add_dtype(&c, dy);
    add_dtype(&c, dx);
    c.bytes = u(rows) * u(dim) * (es(x.dtype) + es(dy.dtype) + es(dx.dtype) * (accumulate ? 2u : 1u)) +
              u(dim) * es(w.dtype);
    record(be, &c);
    return RRC_OK;
}

static rrc_status dr_rope(const rrc_backend *be, rrc_dryrun_op op, int64_t n_tokens, int64_t n_heads,
                          int64_t head_dim, rrc_buf x, rrc_buf y) {
    rrc_dryrun_call c;
    call_init(&c, op);
    add_shape(&c, n_tokens);
    add_shape(&c, n_heads);
    add_shape(&c, head_dim);
    add_dtype(&c, x);
    add_dtype(&c, y);
    /* x read, y written, positions, one cos/sin table row pair per token */
    c.bytes = u(n_tokens) * u(n_heads) * u(head_dim) * (es(x.dtype) + es(y.dtype)) + u(n_tokens) * 4u +
              u(n_tokens) * u(head_dim) * 4u;
    record(be, &c);
    return RRC_OK;
}
static rrc_status dr_rope_fwd(const rrc_backend *be, int64_t n_tokens, int64_t n_heads, int64_t head_dim, int64_t ld,
                              rrc_buf x, rrc_buf pos, rrc_buf cos_tab, rrc_buf sin_tab, int64_t n_pos, rrc_buf y) {
    (void)ld;
    (void)pos;
    (void)cos_tab;
    (void)sin_tab;
    (void)n_pos;
    return dr_rope(be, RRC_DR_ROPE_FWD, n_tokens, n_heads, head_dim, x, y);
}
static rrc_status dr_rope_bwd(const rrc_backend *be, int64_t n_tokens, int64_t n_heads, int64_t head_dim, int64_t ld,
                              rrc_buf dy, rrc_buf pos, rrc_buf cos_tab, rrc_buf sin_tab, int64_t n_pos, rrc_buf dx) {
    (void)ld;
    (void)pos;
    (void)cos_tab;
    (void)sin_tab;
    (void)n_pos;
    return dr_rope(be, RRC_DR_ROPE_BWD, n_tokens, n_heads, head_dim, dy, dx);
}

static rrc_status dr_softmax_fwd(const rrc_backend *be, int64_t batch, int64_t rows, int64_t cols, int64_t ld,
                                 int64_t batch_stride, rrc_buf s, float scale, int64_t row_offset, int causal,
                                 rrc_buf doc_ids, rrc_buf p) {
    (void)ld;
    (void)batch_stride;
    (void)scale;
    (void)row_offset;
    (void)causal;
    rrc_dryrun_call c;
    call_init(&c, RRC_DR_ATTN_SOFTMAX_FWD);
    add_shape(&c, batch);
    add_shape(&c, rows);
    add_shape(&c, cols);
    add_dtype(&c, s);
    add_dtype(&c, p);
    c.bytes = u(batch) * u(rows) * u(cols) * (es(s.dtype) + es(p.dtype)) + (rrc_buf_is_none(doc_ids) ? 0 : u(cols) * 4u);
    record(be, &c);
    return RRC_OK;
}

static rrc_status dr_softmax_bwd(const rrc_backend *be, int64_t batch, int64_t rows, int64_t cols, int64_t ld,
                                 int64_t batch_stride, rrc_buf p, rrc_buf dp, float scale, rrc_buf ds) {
    (void)ld;
    (void)batch_stride;
    (void)scale;
    rrc_dryrun_call c;
    call_init(&c, RRC_DR_ATTN_SOFTMAX_BWD);
    add_shape(&c, batch);
    add_shape(&c, rows);
    add_shape(&c, cols);
    add_dtype(&c, p);
    add_dtype(&c, dp);
    add_dtype(&c, ds);
    c.bytes = u(batch) * u(rows) * u(cols) * (es(p.dtype) + es(dp.dtype) + es(ds.dtype));
    record(be, &c);
    return RRC_OK;
}

static rrc_status dr_swiglu_fwd(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf gate, rrc_buf up,
                                int64_t ld_in, rrc_buf h, int64_t ld_out) {
    (void)ld_in;
    (void)ld_out;
    rrc_dryrun_call c;
    call_init(&c, RRC_DR_SWIGLU_FWD);
    add_shape(&c, rows);
    add_shape(&c, cols);
    add_dtype(&c, gate);
    add_dtype(&c, up);
    add_dtype(&c, h);
    c.bytes = u(rows) * u(cols) * (es(gate.dtype) + es(up.dtype) + es(h.dtype));
    record(be, &c);
    return RRC_OK;
}

static rrc_status dr_swiglu_bwd(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf gate, rrc_buf up,
                                int64_t ld_in, rrc_buf dh, int64_t ld_dh, rrc_buf dgate, rrc_buf dup, int64_t ld_out) {
    (void)ld_in;
    (void)ld_dh;
    (void)ld_out;
    rrc_dryrun_call c;
    call_init(&c, RRC_DR_SWIGLU_BWD);
    add_shape(&c, rows);
    add_shape(&c, cols);
    add_dtype(&c, gate);
    add_dtype(&c, up);
    add_dtype(&c, dh);
    add_dtype(&c, dgate);
    add_dtype(&c, dup);
    c.bytes = u(rows) * u(cols) *
              (es(gate.dtype) + es(up.dtype) + es(dh.dtype) + es(dgate.dtype) + es(dup.dtype));
    record(be, &c);
    return RRC_OK;
}

static rrc_status dr_elem(const rrc_backend *be, rrc_dryrun_op op, int64_t n, rrc_buf a, rrc_buf b, rrc_buf out,
                          uint64_t bytes) {
    rrc_dryrun_call c;
    call_init(&c, op);
    add_shape(&c, n);
    add_dtype(&c, a);
    add_dtype(&c, b);
    add_dtype(&c, out);
    c.bytes = bytes;
    record(be, &c);
    return RRC_OK;
}
static rrc_status dr_add(const rrc_backend *be, int64_t n, rrc_buf a, rrc_buf b, rrc_buf out) {
    return dr_elem(be, RRC_DR_ADD, n, a, b, out, u(n) * (es(a.dtype) + es(b.dtype) + es(out.dtype)));
}
static rrc_status dr_axpy(const rrc_backend *be, int64_t n, float alpha, rrc_buf x, rrc_buf y) {
    (void)alpha;
    return dr_elem(be, RRC_DR_AXPY, n, x, y, rrc_buf_none(), u(n) * (es(x.dtype) + 2u * es(y.dtype)));
}
static rrc_status dr_scale(const rrc_backend *be, int64_t n, float alpha, rrc_buf x) {
    (void)alpha;
    return dr_elem(be, RRC_DR_SCALE, n, x, rrc_buf_none(), rrc_buf_none(), 2u * u(n) * es(x.dtype));
}
static rrc_status dr_convert(const rrc_backend *be, int64_t n, rrc_buf src, rrc_buf dst) {
    return dr_elem(be, RRC_DR_CONVERT, n, src, dst, rrc_buf_none(), u(n) * (es(src.dtype) + es(dst.dtype)));
}
static rrc_status dr_sum_squares(const rrc_backend *be, int64_t n, rrc_buf x, rrc_buf out, int accumulate) {
    dr_impl *d = IMPL(be);
    /* the CUDA backend reduces fixed chunks of 4096 elements into a partial buffer (DESIGN.md §11.3) */
    const uint64_t part = (u(n) + 4095u) / 4096u * 4u;
    if (part > d->st.sum_squares_partials_max_bytes) d->st.sum_squares_partials_max_bytes = part;
    return dr_elem(be, RRC_DR_SUM_SQUARES, n, x, out, rrc_buf_none(), u(n) * es(x.dtype) + (accumulate ? 8u : 4u));
}

static rrc_status dr_cross_entropy(const rrc_backend *be, int64_t rows, int64_t vocab, rrc_buf logits, int64_t ld,
                                   rrc_buf labels, rrc_buf loss, float grad_scale, rrc_buf row_scale,
                                   int write_grad) {
    (void)ld;
    (void)grad_scale;
    rrc_dryrun_call c;
    call_init(&c, RRC_DR_CROSS_ENTROPY);
    add_shape(&c, rows);
    add_shape(&c, vocab);
    add_dtype(&c, logits);
    add_dtype(&c, labels);
    add_dtype(&c, loss);
    c.bytes = u(rows) * u(vocab) * es(logits.dtype) * (write_grad ? 2u : 1u) + u(rows) * 8u +
              (rrc_buf_is_none(row_scale) ? 0 : u(rows) * 4u);
    record(be, &c);
    return RRC_OK;
}

static rrc_status dr_adamw(const rrc_backend *be, int64_t n, rrc_buf param, rrc_buf grad, rrc_buf m, rrc_buf v,
                           const rrc_adamw_hparams *h) {
    (void)grad;
    (void)m;
    (void)v;
    (void)h;
    rrc_dryrun_call c;
    call_init(&c, RRC_DR_ADAMW);
    add_shape(&c, n);
    add_dtype(&c, param);
    c.bytes = u(n) * 4u * 7u; /* param, grad, m, v read; param, m, v written */
    record(be, &c);
    return RRC_OK;
}

/* ---- construction ------------------------------------------------------------------------------- */

rrc_status rrc_dryrun_backend_create(const rrc_dryrun_opts *o, const rrc_backend **out) {
    RRC_REQUIRE(out, RRC_ERR_INVALID_ARG, "null argument");
    *out = NULL;
    rrc_backend *t = (rrc_backend *)calloc(1, sizeof *t);
    dr_impl *d = (dr_impl *)calloc(1, sizeof *d);
    if (!t || !d) {
        free(t);
        free(d);
        return RRC_ERROR(RRC_ERR_OOM, "dry-run backend");
    }
    if (o)
        d->o = *o;
    else
        rrc_dryrun_opts_default(&d->o);
    if (d->o.capacity_bytes > RRC_DR_DEFAULT_CAPACITY) {
        free(t);
        free(d);
        return RRC_ERROR(RRC_ERR_INVALID_ARG, "dry-run backend: capacity_bytes must be <= 2^44");
    }
    t->name = "dryrun";
    t->host_accessible = 0;
    t->impl = d;
    t->alloc = dr_alloc_fn;
    t->free = dr_free_fn;
    t->memcpy_h2d = dr_h2d;
    t->memcpy_d2h = dr_d2h;
    t->memcpy_d2d = dr_d2d;
    t->memset0 = dr_memset0;
    t->sync = dr_sync;
    t->gemm = dr_gemm;
    t->embedding_fwd = dr_embedding_fwd;
    t->add_bias = dr_add_bias;
    t->bias_grad = dr_bias_grad;
    t->rmsnorm_fwd = dr_rmsnorm_fwd;
    t->rmsnorm_bwd = dr_rmsnorm_bwd;
    t->rope_fwd = dr_rope_fwd;
    t->rope_bwd = dr_rope_bwd;
    t->attn_softmax_fwd = dr_softmax_fwd;
    t->attn_softmax_bwd = dr_softmax_bwd;
    t->swiglu_fwd = dr_swiglu_fwd;
    t->swiglu_bwd = dr_swiglu_bwd;
    t->add = dr_add;
    t->axpy = dr_axpy;
    t->scale = dr_scale;
    t->convert = dr_convert;
    t->sum_squares = dr_sum_squares;
    t->cross_entropy_fwd_bwd = dr_cross_entropy;
    t->adamw_step = dr_adamw;
    *out = t;
    return RRC_OK;
}

int rrc_backend_is_dryrun(const rrc_backend *be) { return be && be->alloc == dr_alloc_fn; }

void rrc_dryrun_backend_destroy(const rrc_backend *be) {
    if (!rrc_backend_is_dryrun(be)) return;
    dr_impl *d = IMPL(be);
    free(d->live);
    free(d);
    free((void *)(uintptr_t)be);
}

void rrc_dryrun_set_hook(const rrc_backend *be, rrc_dryrun_hook fn, void *ctx) {
    if (!rrc_backend_is_dryrun(be)) return;
    IMPL(be)->hook = fn;
    IMPL(be)->hook_ctx = ctx;
}

void rrc_dryrun_get_stats(const rrc_backend *be, rrc_dryrun_stats *out) {
    if (!out) return;
    if (!rrc_backend_is_dryrun(be)) {
        memset(out, 0, sizeof *out);
        return;
    }
    *out = IMPL(be)->st;
}

void rrc_dryrun_reset_counters(const rrc_backend *be) {
    if (!rrc_backend_is_dryrun(be)) return;
    rrc_dryrun_stats *s = &IMPL(be)->st;
    const uint64_t live = s->live_bytes;
    const int64_t la = s->live_allocs;
    memset(s, 0, sizeof *s);
    s->live_bytes = live;
    s->peak_bytes = live;
    s->live_allocs = la;
}
