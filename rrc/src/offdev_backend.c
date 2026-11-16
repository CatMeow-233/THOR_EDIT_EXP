/* rrc/src/offdev_backend.c — TEST ONLY off-device wrapper of a CPU backend (rrc/offdev.h,
 * rrc/docs/DESIGN.md §14).
 *
 * Device addresses are handed out from one PROT_NONE reservation (bump allocation, page aligned,
 * an unmapped guard page after every allocation, never reused); the bytes are an allocation of the
 * inner backend. A table sorted by device address maps [dev, dev + bytes] to the real storage, so
 * a view (base + offset) translates to real + offset. Offsets keep their alignment modulo 64 (the
 * inner CPU allocation is 64-byte aligned, every device base is page aligned), and distinct
 * allocations stay disjoint in both spaces, so every aliasing / overlap decision of src/ops.c is
 * the same as on the inner backend. */
#define _DEFAULT_SOURCE /* MAP_ANONYMOUS, MAP_NORESERVE */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "rrc/offdev.h"

typedef struct od_entry {
    uintptr_t dev;
    size_t bytes;
    void *real;
} od_entry;

typedef struct od_state {
    uint32_t magic;
    const rrc_backend *inner;
    rrc_backend table;
    char name[64];
    uintptr_t base;    /* reservation */
    size_t size;       /* reservation bytes */
    size_t next;       /* bump offset into the reservation */
    size_t page;
    od_entry *e;       /* live allocations, sorted by dev (bump order) */
    size_t n, cap;
    rrc_offdev_stats st;
} od_state;

#define OD_MAGIC 0x0FFDE71Cu

static od_state *st_of(const rrc_backend *be) { return (od_state *)be->impl; }

/* Index of the live allocation containing [p, p + need] (p == dev + bytes allowed for need == 0),
 * or -1. */
static ptrdiff_t od_find(const od_state *s, uintptr_t p, size_t need) {
    size_t lo = 0, hi = s->n; /* first entry with dev > p */
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (s->e[mid].dev <= p) lo = mid + 1;
        else hi = mid;
    }
    if (lo == 0) return -1;
    const od_entry *e = &s->e[lo - 1];
    const size_t off = (size_t)(p - e->dev);
    if (off > e->bytes || need > e->bytes - off) return -1;
    return (ptrdiff_t)(lo - 1);
}

/* Device pointer -> real storage; NULL stays NULL. `need` bytes from ptr must lie inside the
 * allocation. */
static rrc_status od_xlate(od_state *s, const void *ptr, size_t need, void **real, const char *what) {
    *real = NULL;
    if (!ptr) return RRC_OK;
    const uintptr_t p = (uintptr_t)ptr;
    const ptrdiff_t i = od_find(s, p, need);
    if (i < 0) {
        s->st.bad_ptrs++;
        return RRC_ERROR(RRC_ERR_INVALID_ARG, "%s: %s: pointer %p (+%zu bytes) is in no live allocation", s->name,
                         what, ptr, need);
    }
    *real = (void *)((uintptr_t)s->e[i].real + (p - s->e[i].dev));
    return RRC_OK;
}

static rrc_status od_buf(od_state *s, rrc_buf *b, const char *what) {
    if (rrc_buf_is_none(*b)) return RRC_OK;
    void *r = NULL;
    RRC_CHECK(od_xlate(s, b->ptr, 0, &r, what));
    b->ptr = r;
    b->be = s->inner;
    return RRC_OK;
}

/* ---- memory ------------------------------------------------------------------------------- */

static rrc_status od_alloc(const rrc_backend *be, size_t bytes, void **out) {
    od_state *s = st_of(be);
    *out = NULL;
    if (bytes == 0) return RRC_OK;
    const size_t span = ((bytes + s->page - 1) / s->page) * s->page;
    if (span < bytes || span > s->size - s->next || s->page > s->size - s->next - span)
        return RRC_ERROR(RRC_ERR_OOM, "%s: address reservation exhausted (%zu bytes requested, %zu of %zu used)",
                         s->name, bytes, s->next, s->size);
    if (s->n == s->cap) {
        const size_t cap = s->cap ? 2 * s->cap : 64;
        od_entry *e = (od_entry *)realloc(s->e, cap * sizeof *e);
        if (!e) return RRC_ERROR(RRC_ERR_OOM, "%s: allocation table", s->name);
        s->e = e;
        s->cap = cap;
    }
    void *real = NULL;
    RRC_CHECK(s->inner->alloc(s->inner, bytes, &real));
    const uintptr_t dev = s->base + s->next;
    s->next += span + s->page; /* guard page */
    s->e[s->n++] = (od_entry){dev, bytes, real};
    s->st.allocs++;
    s->st.live++;
    s->st.live_bytes += (int64_t)bytes;
    if (s->st.live_bytes > s->st.peak_bytes) s->st.peak_bytes = s->st.live_bytes;
    *out = (void *)dev;
    return RRC_OK;
}

static void od_free(const rrc_backend *be, void *ptr) {
    od_state *s = st_of(be);
    if (!ptr) return;
    const ptrdiff_t i = od_find(s, (uintptr_t)ptr, 0);
    if (i < 0 || s->e[i].dev != (uintptr_t)ptr) {
        s->st.bad_frees++;
        return;
    }
    s->inner->free(s->inner, s->e[i].real);
    s->st.frees++;
    s->st.live--;
    s->st.live_bytes -= (int64_t)s->e[i].bytes;
    memmove(&s->e[i], &s->e[i + 1], (s->n - (size_t)i - 1) * sizeof *s->e);
    s->n--;
}

static rrc_status od_h2d(const rrc_backend *be, void *dst, const void *src, size_t bytes) {
    od_state *s = st_of(be);
    void *d;
    RRC_CHECK(od_xlate(s, dst, bytes, &d, "memcpy_h2d"));
    return s->inner->memcpy_h2d(s->inner, d, src, bytes);
}
static rrc_status od_d2h(const rrc_backend *be, void *dst, const void *src, size_t bytes) {
    od_state *s = st_of(be);
    void *r;
    RRC_CHECK(od_xlate(s, src, bytes, &r, "memcpy_d2h"));
    return s->inner->memcpy_d2h(s->inner, dst, r, bytes);
}
static rrc_status od_d2d(const rrc_backend *be, void *dst, const void *src, size_t bytes) {
    od_state *s = st_of(be);
    void *d, *r;
    RRC_CHECK(od_xlate(s, dst, bytes, &d, "memcpy_d2d dst"));
    RRC_CHECK(od_xlate(s, src, bytes, &r, "memcpy_d2d src"));
    return s->inner->memcpy_d2d(s->inner, d, r, bytes);
}
static rrc_status od_memset0(const rrc_backend *be, void *ptr, size_t bytes) {
    od_state *s = st_of(be);
    void *d;
    RRC_CHECK(od_xlate(s, ptr, bytes, &d, "memset0"));
    return s->inner->memset0(s->inner, d, bytes);
}
static rrc_status od_sync(const rrc_backend *be) {
    od_state *s = st_of(be);
    return s->inner->sync(s->inner);
}

/* ---- ops: translate every buffer, call the inner backend's op directly ----------------------- */

#define OD_S od_state *s = st_of(be)
#define TR(b) RRC_CHECK(od_buf(s, &(b), #b))

static rrc_status od_gemm(const rrc_backend *be, const rrc_gemm_args *g) {
    OD_S;
    rrc_gemm_args x = *g;
    TR(x.a);
    TR(x.b);
    TR(x.c);
    return s->inner->gemm(s->inner, &x);
}
static rrc_status od_embedding_fwd(const rrc_backend *be, int64_t n_tokens, int64_t dim, int64_t vocab, rrc_buf ids,
                                   rrc_buf table, rrc_buf out) {
    OD_S;
    TR(ids);
    TR(table);
    TR(out);
    return s->inner->embedding_fwd(s->inner, n_tokens, dim, vocab, ids, table, out);
}
static rrc_status od_add_bias(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf x, int64_t ldx, rrc_buf bias) {
    OD_S;
    TR(x);
    TR(bias);
    return s->inner->add_bias(s->inner, rows, cols, x, ldx, bias);
}
static rrc_status od_bias_grad(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf dy, int64_t ldy,
                               rrc_buf dbias, int accumulate) {
    OD_S;
    TR(dy);
    TR(dbias);
    return s->inner->bias_grad(s->inner, rows, cols, dy, ldy, dbias, accumulate);
}
static rrc_status od_rmsnorm_fwd(const rrc_backend *be, int64_t rows, int64_t dim, int64_t ld, rrc_buf x, rrc_buf w,
                                 float eps, rrc_buf y) {
    OD_S;
    TR(x);
    TR(w);
    TR(y);
    return s->inner->rmsnorm_fwd(s->inner, rows, dim, ld, x, w, eps, y);
}
static rrc_status od_rmsnorm_bwd(const rrc_backend *be, int64_t rows, int64_t dim, int64_t ld, rrc_buf x, rrc_buf w,
                                 float eps, rrc_buf dy, rrc_buf dx, int accumulate) {
    OD_S;
    TR(x);
    TR(w);
    TR(dy);
    TR(dx);
    return s->inner->rmsnorm_bwd(s->inner, rows, dim, ld, x, w, eps, dy, dx, accumulate);
}
static rrc_status od_rope_fwd(const rrc_backend *be, int64_t n_tokens, int64_t n_heads, int64_t head_dim, int64_t ld,
                              rrc_buf x, rrc_buf pos, rrc_buf cos_tab, rrc_buf sin_tab, int64_t n_pos, rrc_buf y) {
    OD_S;
    TR(x);
    TR(pos);
    TR(cos_tab);
    TR(sin_tab);
    TR(y);
    return s->inner->rope_fwd(s->inner, n_tokens, n_heads, head_dim, ld, x, pos, cos_tab, sin_tab, n_pos, y);
}
static rrc_status od_rope_bwd(const rrc_backend *be, int64_t n_tokens, int64_t n_heads, int64_t head_dim, int64_t ld,
                              rrc_buf dy, rrc_buf pos, rrc_buf cos_tab, rrc_buf sin_tab, int64_t n_pos, rrc_buf dx) {
    OD_S;
    TR(dy);
    TR(pos);
    TR(cos_tab);
    TR(sin_tab);
    TR(dx);
    return s->inner->rope_bwd(s->inner, n_tokens, n_heads, head_dim, ld, dy, pos, cos_tab, sin_tab, n_pos, dx);
}
static rrc_status od_attn_softmax_fwd(const rrc_backend *be, int64_t batch, int64_t rows, int64_t cols, int64_t ld,
                                      int64_t batch_stride, rrc_buf sc, float scale, int64_t row_offset, int causal,
                                      rrc_buf doc_ids, rrc_buf p) {
    OD_S;
    TR(sc);
    TR(doc_ids);
    TR(p);
    return s->inner->attn_softmax_fwd(s->inner, batch, rows, cols, ld, batch_stride, sc, scale, row_offset, causal,
                                      doc_ids, p);
}
static rrc_status od_attn_softmax_bwd(const rrc_backend *be, int64_t batch, int64_t rows, int64_t cols, int64_t ld,
                                      int64_t batch_stride, rrc_buf p, rrc_buf dp, float scale, rrc_buf ds) {
    OD_S;
    TR(p);
    TR(dp);
    TR(ds);
    return s->inner->attn_softmax_bwd(s->inner, batch, rows, cols, ld, batch_stride, p, dp, scale, ds);
}
static rrc_status od_swiglu_fwd(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf gate, rrc_buf up,
                                int64_t ld_in, rrc_buf h, int64_t ld_out) {
    OD_S;
    TR(gate);
    TR(up);
    TR(h);
    return s->inner->swiglu_fwd(s->inner, rows, cols, gate, up, ld_in, h, ld_out);
}
static rrc_status od_swiglu_bwd(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf gate, rrc_buf up,
                                int64_t ld_in, rrc_buf dh, int64_t ld_dh, rrc_buf dgate, rrc_buf dup, int64_t ld_out) {
    OD_S;
    TR(gate);
    TR(up);
    TR(dh);
    TR(dgate);
    TR(dup);
    return s->inner->swiglu_bwd(s->inner, rows, cols, gate, up, ld_in, dh, ld_dh, dgate, dup, ld_out);
}
static rrc_status od_add(const rrc_backend *be, int64_t n, rrc_buf a, rrc_buf b, rrc_buf out) {
    OD_S;
    TR(a);
    TR(b);
    TR(out);
    return s->inner->add(s->inner, n, a, b, out);
}
static rrc_status od_axpy(const rrc_backend *be, int64_t n, float alpha, rrc_buf x, rrc_buf y) {
    OD_S;
    TR(x);
    TR(y);
    return s->inner->axpy(s->inner, n, alpha, x, y);
}
static rrc_status od_scale(const rrc_backend *be, int64_t n, float alpha, rrc_buf x) {
    OD_S;
    TR(x);
    return s->inner->scale(s->inner, n, alpha, x);
}
static rrc_status od_convert(const rrc_backend *be, int64_t n, rrc_buf src, rrc_buf dst) {
    OD_S;
    TR(src);
    TR(dst);
    return s->inner->convert(s->inner, n, src, dst);
}
static rrc_status od_sum_squares(const rrc_backend *be, int64_t n, rrc_buf x, rrc_buf out, int accumulate) {
    OD_S;
    TR(x);
    TR(out);
    return s->inner->sum_squares(s->inner, n, x, out, accumulate);
}
static rrc_status od_cross_entropy(const rrc_backend *be, int64_t rows, int64_t vocab, rrc_buf logits, int64_t ld,
                                   rrc_buf labels, rrc_buf loss, float grad_scale, rrc_buf row_scale, int write_grad) {
    OD_S;
    TR(logits);
    TR(labels);
    TR(loss);
    TR(row_scale);
    return s->inner->cross_entropy_fwd_bwd(s->inner, rows, vocab, logits, ld, labels, loss, grad_scale, row_scale,
                                           write_grad);
}
static rrc_status od_adamw_step(const rrc_backend *be, int64_t n, rrc_buf param, rrc_buf grad, rrc_buf m, rrc_buf v,
                                const rrc_adamw_hparams *h) {
    OD_S;
    TR(param);
    TR(grad);
    TR(m);
    TR(v);
    return s->inner->adamw_step(s->inner, n, param, grad, m, v, h);
}

#undef TR
#undef OD_S

/* ---- create / destroy --------------------------------------------------------------------- */

rrc_status rrc_offdev_backend_create(const rrc_backend *inner, const rrc_backend **out) {
    RRC_REQUIRE(inner && out, RRC_ERR_INVALID_ARG, "null argument");
    *out = NULL;
    RRC_REQUIRE(inner->host_accessible, RRC_ERR_INVALID_ARG,
                "offdev: the inner backend '%s' must be host accessible (a CPU backend)", inner->name);
    const long pg = sysconf(_SC_PAGESIZE);
    RRC_REQUIRE(pg > 0, RRC_ERR_BACKEND, "offdev: page size unknown");
    od_state *s = (od_state *)calloc(1, sizeof *s);
    if (!s) return RRC_ERROR(RRC_ERR_OOM, "offdev state");
    s->magic = OD_MAGIC;
    s->inner = inner;
    s->page = (size_t)pg;
    snprintf(s->name, sizeof s->name, "%s-offdev", inner->name);
    /* PROT_NONE + MAP_NORESERVE costs address space only; the largest size that maps wins (no
     * fallback to host pointers if none does) */
    static const size_t sizes[] = {(size_t)64 << 30, (size_t)16 << 30, (size_t)4 << 30, (size_t)1 << 30};
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0] && !s->base; i++) {
        void *p = mmap(NULL, sizes[i], PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p != MAP_FAILED) {
            s->base = (uintptr_t)p;
            s->size = sizes[i];
        }
    }
    if (!s->base) {
        free(s);
        return RRC_ERROR(RRC_ERR_OOM, "offdev: cannot reserve a PROT_NONE address range");
    }
    s->table = (rrc_backend){
        .name = s->name, .host_accessible = 0, .impl = s, .alloc = od_alloc, .free = od_free,
        .memcpy_h2d = od_h2d, .memcpy_d2h = od_d2h, .memcpy_d2d = od_d2d, .memset0 = od_memset0, .sync = od_sync,
        .gemm = od_gemm, .embedding_fwd = od_embedding_fwd, .add_bias = od_add_bias, .bias_grad = od_bias_grad,
        .rmsnorm_fwd = od_rmsnorm_fwd, .rmsnorm_bwd = od_rmsnorm_bwd, .rope_fwd = od_rope_fwd,
        .rope_bwd = od_rope_bwd, .attn_softmax_fwd = od_attn_softmax_fwd, .attn_softmax_bwd = od_attn_softmax_bwd,
        .swiglu_fwd = od_swiglu_fwd, .swiglu_bwd = od_swiglu_bwd, .add = od_add, .axpy = od_axpy,
        .scale = od_scale, .convert = od_convert, .sum_squares = od_sum_squares,
        .cross_entropy_fwd_bwd = od_cross_entropy, .adamw_step = od_adamw_step,
    };
    *out = &s->table;
    return RRC_OK;
}

int rrc_backend_is_offdev(const rrc_backend *be) {
    return be && be->alloc == od_alloc && be->impl && ((const od_state *)be->impl)->magic == OD_MAGIC;
}

rrc_status rrc_offdev_get_stats(const rrc_backend *be, rrc_offdev_stats *out) {
    RRC_REQUIRE(out && rrc_backend_is_offdev(be), RRC_ERR_INVALID_ARG, "not an offdev backend");
    *out = st_of(be)->st;
    return RRC_OK;
}

rrc_status rrc_offdev_backend_destroy(const rrc_backend *be, rrc_offdev_stats *final) {
    if (!be) return RRC_OK;
    RRC_REQUIRE(rrc_backend_is_offdev(be), RRC_ERR_INVALID_ARG, "not an offdev backend");
    od_state *s = st_of(be);
    const rrc_offdev_stats st = s->st;
    if (final) *final = st;
    for (size_t i = 0; i < s->n; i++) s->inner->free(s->inner, s->e[i].real);
    free(s->e);
    const int unmapped = munmap((void *)s->base, s->size) == 0;
    char name[64];
    snprintf(name, sizeof name, "%s", s->name);
    s->magic = 0;
    free(s);
    if (!unmapped) return RRC_ERROR(RRC_ERR_BACKEND, "%s: munmap of the address reservation failed", name);
    if (st.live || st.bad_frees || st.bad_ptrs)
        return RRC_ERROR(RRC_ERR_INTERNAL, "%s: %lld allocations (%lld bytes) still live, %lld bad frees, %lld "
                         "refused translations", name, (long long)st.live, (long long)st.live_bytes,
                         (long long)st.bad_frees, (long long)st.bad_ptrs);
    return RRC_OK;
}
