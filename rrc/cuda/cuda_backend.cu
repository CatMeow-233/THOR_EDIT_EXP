// rrc/cuda/cuda_backend.cu — the CUDA backend: memory, stream, device error flag, cuBLASLt GEMM
// through the plan ladder of gemm_plan.c, and the op table (DESIGN.md §11). COMPILED-ONLY until
// rrc_selftest has been run on a GPU.
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_internal.h"
#include "rrc_cuda.h"

static_assert(RRC_KC_IGNORE_INDEX == RRC_IGNORE_INDEX, "kernels_common ignore index must match backend.h");

#define LT_CACHE 64

namespace {

struct lt_key {
    int transa, transb, tf32;
    int adt, bdt, cdt;
    int64_t m, n, k, batch, lda, ldb, ldc, sa, sb, sc;
    unsigned align_a, align_b, align_c;
};

struct lt_entry {
    int used, ok;
    lt_key key;
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulAlgo_t algo;
};

struct cuda_be {
    rrc_backend table; // first member: the backend handed out
    int device;
    rrc_cuda_options o;
    cudaStream_t stream;
    cublasLtHandle_t lt;
    void *workspace;
    size_t ws_bytes;
    rrc_cuda_err *err_dev;
    rrc_cuda_err *err_host; // pinned
    void *scratch;
    size_t scratch_bytes;
    void *ss_part; // sum_squares partials (f32)
    size_t ss_part_bytes;
    lt_entry cache[LT_CACHE];
    int cache_next;
    int have_rep;
    rrc_gemm_exec_report rep;
};

const char k_cuda_name[] = "cuda";

cuda_be *CB(const rrc_backend *be) { return (cuda_be *)be->impl; }

rrc_status cu_fail(cudaError_t e, const char *what) {
    return RRC_ERROR(e == cudaErrorMemoryAllocation ? RRC_ERR_OOM : RRC_ERR_BACKEND, "cuda: %s: %s (%s)", what,
                     cudaGetErrorName(e), cudaGetErrorString(e));
}
#define CU(expr, what)                                                                                  \
    do {                                                                                                \
        cudaError_t cu_e_ = (expr);                                                                     \
        if (cu_e_ != cudaSuccess) return cu_fail(cu_e_, what);                                          \
    } while (0)
#define LT(expr, what)                                                                                  \
    do {                                                                                                \
        cublasStatus_t lt_s_ = (expr);                                                                  \
        if (lt_s_ != CUBLAS_STATUS_SUCCESS)                                                             \
            return RRC_ERROR(RRC_ERR_BACKEND, "cublasLt: %s: status %d", what, (int)lt_s_);            \
    } while (0)

const char *op_name(unsigned op) {
    switch (op) {
    case RRC_CU_OP_EMBEDDING: return "embedding_fwd (token id outside [0, vocab))";
    case RRC_CU_OP_ROPE_FWD: return "rope_fwd (position id outside [0, n_pos))";
    case RRC_CU_OP_ROPE_BWD: return "rope_bwd (position id outside [0, n_pos))";
    case RRC_CU_OP_XENT: return "cross_entropy_fwd_bwd (label outside [0, vocab) and not the ignore index)";
    default: return "?";
    }
}

rrc_status err_reset(cuda_be *c) {
    CU(cudaMemsetAsync(&c->err_dev->op, 0, sizeof(unsigned int), c->stream), "reset error flag");
    CU(cudaMemsetAsync(&c->err_dev->index, 0xFF, sizeof(unsigned long long), c->stream), "reset error flag");
    return RRC_OK;
}

// Wait for the stream, then report (and clear) a device range error.
rrc_status sync_and_check(cuda_be *c) {
    CU(cudaMemcpyAsync(c->err_host, c->err_dev, sizeof(rrc_cuda_err), cudaMemcpyDeviceToHost, c->stream),
       "read error flag");
    CU(cudaStreamSynchronize(c->stream), "stream synchronize");
    if (c->err_host->op != RRC_CU_OP_NONE) {
        unsigned op = c->err_host->op;
        unsigned long long idx = c->err_host->index;
        RRC_CHECK(err_reset(c));
        CU(cudaStreamSynchronize(c->stream), "stream synchronize");
        return RRC_ERROR(RRC_ERR_RANGE, "cuda: device range error in %s, smallest offending row/token %llu (reported at "
                                        "sync; outputs of the ops since the previous sync are unspecified)",
                         op_name(op), idx);
    }
    return RRC_OK;
}

// After every enqueue: launch errors, and with sync_each_op the op's own completion and range check.
rrc_status after(cuda_be *c, cudaError_t launch, const char *what) {
    if (launch != cudaSuccess) return cu_fail(launch, what);
    if (c->o.sync_each_op) return sync_and_check(c);
    return RRC_OK;
}

// ---- memory ---------------------------------------------------------------------------------------

rrc_status cb_alloc(const rrc_backend *be, size_t bytes, void **out) {
    (void)be;
    *out = NULL;
    if (bytes == 0) return RRC_OK;
    cudaError_t e = cudaMalloc(out, bytes);
    if (e != cudaSuccess) {
        *out = NULL;
        (void)cudaGetLastError();
        return e == cudaErrorMemoryAllocation ? RRC_ERROR(RRC_ERR_OOM, "cuda alloc of %zu bytes", bytes)
                                              : cu_fail(e, "cudaMalloc");
    }
    return RRC_OK;
}
void cb_free(const rrc_backend *be, void *ptr) {
    if (!ptr) return;
    // work enqueued on the stream may still use ptr: wait for it before freeing (errors of that
    // work, if any, resurface at the next sync)
    (void)cudaStreamSynchronize(CB(be)->stream);
    (void)cudaFree(ptr);
}
rrc_status cb_h2d(const rrc_backend *be, void *dst, const void *src, size_t bytes) {
    cuda_be *c = CB(be);
    if (!bytes) return RRC_OK;
    CU(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, c->stream), "memcpy_h2d");
    CU(cudaStreamSynchronize(c->stream), "memcpy_h2d synchronize"); // the host source may be reused on return
    return RRC_OK;
}
rrc_status cb_d2h(const rrc_backend *be, void *dst, const void *src, size_t bytes) {
    cuda_be *c = CB(be);
    if (bytes) CU(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, c->stream), "memcpy_d2h");
    return sync_and_check(c);
}
rrc_status cb_d2d(const rrc_backend *be, void *dst, const void *src, size_t bytes) {
    cuda_be *c = CB(be);
    if (!bytes) return RRC_OK;
    return after(c, cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, c->stream), "memcpy_d2d");
}
rrc_status cb_memset0(const rrc_backend *be, void *ptr, size_t bytes) {
    cuda_be *c = CB(be);
    if (!bytes) return RRC_OK;
    return after(c, cudaMemsetAsync(ptr, 0, bytes, c->stream), "memset0");
}
rrc_status cb_sync(const rrc_backend *be) { return sync_and_check(CB(be)); }

// Grow a backend-owned device buffer (waits for the stream before freeing the old one). `buf` and
// `have` must be the owning members: on any failure after the old buffer is freed they are left
// as (NULL, 0), so no dangling pointer survives.
rrc_status grow(cuda_be *c, void **buf, size_t *have, size_t need, const char *what) {
    if (need <= *have) return RRC_OK;
    CU(cudaStreamSynchronize(c->stream), what);
    if (*buf) CU(cudaFree(*buf), what);
    *buf = NULL;
    *have = 0;
    cudaError_t e = cudaMalloc(buf, need);
    if (e != cudaSuccess) {
        *buf = NULL;
        (void)cudaGetLastError();
        return cu_fail(e, what);
    }
    *have = need;
    return RRC_OK;
}

// ---- gemm: plan ladder with cuBLASLt ----------------------------------------------------------------

cudaDataType_t lt_type(rrc_dtype d) { return d == RRC_BF16 ? CUDA_R_16BF : CUDA_R_32F; }

lt_key make_key(const rrc_lt_call *call) {
    lt_key k;
    memset(&k, 0, sizeof k); // padding bytes take part in memcmp
    const rrc_kc_lt_p *p = &call->p;
    k.transa = p->transa;
    k.transb = p->transb;
    k.tf32 = call->tf32;
    k.adt = p->adt;
    k.bdt = p->bdt;
    k.cdt = p->cdt;
    k.m = p->m;
    k.n = p->n;
    k.k = p->k;
    k.batch = p->batch;
    k.lda = p->lda;
    k.ldb = p->ldb;
    k.ldc = p->ldc;
    k.sa = p->sa;
    k.sb = p->sb;
    k.sc = p->sc;
    k.align_a = call->align_a;
    k.align_b = call->align_b;
    k.align_c = call->align_c;
    return k;
}

void entry_release(lt_entry *e) {
    if (!e->used) return;
    if (e->la) cublasLtMatrixLayoutDestroy(e->la);
    if (e->lb) cublasLtMatrixLayoutDestroy(e->lb);
    if (e->lc) cublasLtMatrixLayoutDestroy(e->lc);
    if (e->desc) cublasLtMatmulDescDestroy(e->desc);
    memset(e, 0, sizeof *e);
}

rrc_status make_layout(cublasLtMatrixLayout_t *l, rrc_dtype dt, int64_t rows, int64_t cols, int64_t ld, int64_t batch,
                       int64_t stride) {
    LT(cublasLtMatrixLayoutCreate(l, lt_type(dt), (uint64_t)rows, (uint64_t)cols, ld), "layout create");
    if (batch > 1) {
        int32_t b = (int32_t)batch; // <= INT32_MAX (checked in ops.c)
        LT(cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &b, sizeof b), "batch count");
        LT(cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof stride),
           "batch stride");
    }
    return RRC_OK;
}

// An algorithm whose partial sums (if any) are reduced in the compute type (fp32).
bool algo_reduces_fp32(const cublasLtMatmulAlgo_t *algo) {
    int32_t splitk = 1;
    uint32_t scheme = CUBLASLT_REDUCTION_SCHEME_NONE;
    size_t w = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof splitk, &w) !=
        CUBLAS_STATUS_SUCCESS)
        return false;
    if (splitk <= 1) return true;
    if (cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &scheme, sizeof scheme, &w) !=
        CUBLAS_STATUS_SUCCESS)
        return false;
    return scheme == CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
}

// One heuristic query for e's descriptors; *found = an acceptable algorithm was stored in e->algo.
// NOT_SUPPORTED, no result or failed result states are not errors (the ladder moves on).
rrc_status lt_query(cuda_be *c, const rrc_lt_call *call, lt_entry *e, bool fp32_mask, bool *found) {
    *found = false;
    cublasLtMatmulPreference_t pref;
    LT(cublasLtMatmulPreferenceCreate(&pref), "preference");
    uint64_t ws = c->ws_bytes;
    uint32_t red = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    uint32_t aa = call->align_a, ab = call->align_b, ac = call->align_c;
    cublasStatus_t st = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws);
    if (st == CUBLAS_STATUS_SUCCESS && fp32_mask)
        st = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &red, sizeof red);
    if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &aa, sizeof aa);
    if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &ab, sizeof ab);
    if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &ac, sizeof ac);
    if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &ac, sizeof ac);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulPreferenceDestroy(pref);
        return RRC_ERROR(RRC_ERR_BACKEND, "cublasLt: preference attribute: status %d", (int)st);
    }
    enum { NREQ = 8 };
    cublasLtMatmulHeuristicResult_t res[NREQ];
    memset(res, 0, sizeof res);
    int nres = 0;
    st = cublasLtMatmulAlgoGetHeuristic(c->lt, e->desc, e->la, e->lb, e->lc, e->lc, pref, fp32_mask ? 1 : NREQ, res, &nres);
    cublasLtMatmulPreferenceDestroy(pref);
    if (st != CUBLAS_STATUS_SUCCESS) return RRC_OK;
    for (int i = 0; i < nres && i < NREQ; i++)
        if (res[i].state == CUBLAS_STATUS_SUCCESS && (fp32_mask || algo_reduces_fp32(&res[i].algo))) {
            e->algo = res[i].algo;
            *found = true;
            break;
        }
    return RRC_OK;
}

// Build descriptors and query the heuristic for one column-major call; e->ok = an algorithm exists.
rrc_status entry_build(cuda_be *c, const rrc_lt_call *call, lt_entry *e) {
    const rrc_kc_lt_p *p = &call->p;
    e->used = 1;
    e->key = make_key(call);
    const cublasComputeType_t ct = call->tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    LT(cublasLtMatmulDescCreate(&e->desc, ct, CUDA_R_32F), "matmul desc");
    cublasOperation_t ta = p->transa ? CUBLAS_OP_T : CUBLAS_OP_N, tb = p->transb ? CUBLAS_OP_T : CUBLAS_OP_N;
    LT(cublasLtMatmulDescSetAttribute(e->desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta), "transa");
    LT(cublasLtMatmulDescSetAttribute(e->desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb), "transb");
    RRC_CHECK(make_layout(&e->la, p->adt, p->transa ? p->k : p->m, p->transa ? p->m : p->k, p->lda, p->batch, p->sa));
    RRC_CHECK(make_layout(&e->lb, p->bdt, p->transb ? p->n : p->k, p->transb ? p->k : p->n, p->ldb, p->batch, p->sb));
    RRC_CHECK(make_layout(&e->lc, p->cdt, p->m, p->n, p->ldc, p->batch, p->sc));
    // INPLACE / OUTPUT_TYPE split-K reductions accumulate in the output type (bf16 for a bf16 C):
    // only reductions in the compute type (fp32) are allowed, so the result is rounded once.
    // First query: REDUCTION_SCHEME_MASK = COMPUTE_TYPE. cublasLt.h does not say whether such a
    // mask (no bit for NONE = 0) still admits the non-split-K algorithms; if that query finds
    // nothing, a second one with the default mask takes the first of up to 8 results whose own
    // configuration is non-split (SPLITK_NUM <= 1) or reduces in the compute type.
    bool found = false;
    RRC_CHECK(lt_query(c, call, e, true, &found));
    if (!found) RRC_CHECK(lt_query(c, call, e, false, &found));
    e->ok = found ? 1 : 0;
    return RRC_OK;
}

// Cached entry for a call (built on a miss); NULL with an error set on failure.
lt_entry *entry_get(cuda_be *c, const rrc_lt_call *call) {
    lt_key k = make_key(call);
    for (int i = 0; i < LT_CACHE; i++)
        if (c->cache[i].used && memcmp(&c->cache[i].key, &k, sizeof k) == 0) return &c->cache[i];
    lt_entry *e = &c->cache[c->cache_next];
    c->cache_next = (c->cache_next + 1) % LT_CACHE;
    entry_release(e);
    if (entry_build(c, call, e) != RRC_OK) {
        entry_release(e);
        return NULL;
    }
    return e;
}

rrc_status ex_scratch(void *ctx, size_t bytes, void **ptr) {
    cuda_be *c = (cuda_be *)ctx;
    RRC_CHECK(grow(c, &c->scratch, &c->scratch_bytes, bytes, "gemm scratch"));
    *ptr = c->scratch;
    return RRC_OK;
}
int ex_supported(void *ctx, const rrc_lt_call *call) {
    cuda_be *c = (cuda_be *)ctx;
    lt_entry *e = entry_get(c, call);
    if (!e) {
        rrc_log(RRC_LOG_WARN, "gemm: cuBLASLt descriptor setup failed (%s); using the next path", rrc_last_error());
        rrc_clear_error();
        return 0;
    }
    return e->ok;
}
rrc_status ex_lt_run(void *ctx, const rrc_lt_call *call) {
    cuda_be *c = (cuda_be *)ctx;
    lt_entry *e = entry_get(c, call);
    if (!e) return rrc_last_status() != RRC_OK ? rrc_last_status() : RRC_ERROR(RRC_ERR_BACKEND, "cublasLt setup");
    RRC_REQUIRE(e->ok, RRC_ERR_INTERNAL, "gemm: lt_run on a call without an algorithm");
    const float alpha = call->p.alpha, beta = call->p.beta;
    LT(cublasLtMatmul(c->lt, e->desc, &alpha, call->p.a, e->la, call->p.b, e->lb, &beta, call->p.c, e->lc, call->p.c,
                      e->lc, &e->algo, c->workspace, c->ws_bytes, c->stream),
       "matmul");
    return after(c, cudaGetLastError(), "cublasLtMatmul");
}
rrc_status ex_kernel_run(void *ctx, const rrc_kc_lt_p *p) {
    cuda_be *c = (cuda_be *)ctx;
    return after(c, rrc_cu_lt_matmul(c->stream, c->o.max_grid_blocks, p), "gemm kernel");
}
rrc_status ex_convert(void *ctx, const rrc_kc_conv2d_p *p) {
    cuda_be *c = (cuda_be *)ctx;
    return after(c, rrc_cu_conv2d(c->stream, c->o.max_grid_blocks, p), "gemm convert");
}
const rrc_gemm_exec_ops k_ops = {ex_scratch, ex_supported, ex_lt_run, ex_kernel_run, ex_convert};

rrc_status cb_gemm(const rrc_backend *be, const rrc_gemm_args *g) {
    cuda_be *c = CB(be);
    rrc_gemm_plan_opts po;
    po.scratch_cap_elems = c->o.gemm_scratch_cap_elems;
    po.allow_tf32 = c->o.allow_tf32;
    po.max_align = 0;
    c->have_rep = 0;
    RRC_CHECK(rrc_gemm_execute(g, &po, (rrc_gemm_level)c->o.gemm_min_level, &k_ops, c, &c->rep));
    c->have_rep = 1;
    return RRC_OK;
}

// ---- ops --------------------------------------------------------------------------------------------

rrc_status cb_embedding_fwd(const rrc_backend *be, int64_t n_tokens, int64_t dim, int64_t vocab, rrc_buf ids,
                            rrc_buf table, rrc_buf out) {
    cuda_be *c = CB(be);
    rrc_kc_embed_p p = {n_tokens, dim, vocab, (const int32_t *)ids.ptr, table.ptr, table.dtype, out.ptr, out.dtype};
    return after(c, rrc_cu_embedding(c->stream, c->o.max_grid_blocks, &p, c->err_dev), "embedding_fwd");
}
rrc_status cb_add_bias(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf x, int64_t ldx, rrc_buf bias) {
    cuda_be *c = CB(be);
    rrc_kc_bias_p p = {rows, cols, ldx, x.ptr, x.dtype, bias.ptr, bias.dtype, 0};
    return after(c, rrc_cu_add_bias(c->stream, c->o.max_grid_blocks, &p), "add_bias");
}
rrc_status cb_bias_grad(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf dy, int64_t ldy, rrc_buf dbias,
                        int accumulate) {
    cuda_be *c = CB(be);
    rrc_kc_bias_p p = {rows, cols, ldy, dy.ptr, dy.dtype, dbias.ptr, dbias.dtype, accumulate};
    return after(c, rrc_cu_bias_grad(c->stream, c->o.max_grid_blocks, &p), "bias_grad");
}
rrc_status cb_rmsnorm_fwd(const rrc_backend *be, int64_t rows, int64_t dim, int64_t ld, rrc_buf x, rrc_buf w, float eps,
                          rrc_buf y) {
    cuda_be *c = CB(be);
    rrc_kc_rmsnorm_p p = {rows, dim, ld, x.ptr, x.dtype, w.ptr, w.dtype, NULL, RRC_F32, y.ptr, y.dtype, eps, 0};
    return after(c, rrc_cu_rmsnorm_fwd(c->stream, c->o.max_grid_blocks, &p), "rmsnorm_fwd");
}
rrc_status cb_rmsnorm_bwd(const rrc_backend *be, int64_t rows, int64_t dim, int64_t ld, rrc_buf x, rrc_buf w, float eps,
                          rrc_buf dy, rrc_buf dx, int accumulate) {
    cuda_be *c = CB(be);
    rrc_kc_rmsnorm_p p = {rows, dim, ld, x.ptr, x.dtype, w.ptr, w.dtype, dy.ptr, dy.dtype, dx.ptr, dx.dtype, eps,
                          accumulate};
    return after(c, rrc_cu_rmsnorm_bwd(c->stream, c->o.max_grid_blocks, &p), "rmsnorm_bwd");
}
rrc_status cb_rope(const rrc_backend *be, int64_t n_tokens, int64_t n_heads, int64_t head_dim, int64_t ld, rrc_buf x,
                   rrc_buf pos, rrc_buf cos_tab, rrc_buf sin_tab, int64_t n_pos, rrc_buf y, int backward) {
    cuda_be *c = CB(be);
    rrc_kc_rope_p p = {n_tokens, n_heads, head_dim, ld, n_pos, x.ptr, x.dtype, y.ptr, y.dtype, (const int32_t *)pos.ptr,
                       (const float *)cos_tab.ptr, (const float *)sin_tab.ptr, backward};
    return after(c, rrc_cu_rope(c->stream, c->o.max_grid_blocks, &p, c->err_dev), backward ? "rope_bwd" : "rope_fwd");
}
rrc_status cb_rope_fwd(const rrc_backend *be, int64_t n_tokens, int64_t n_heads, int64_t head_dim, int64_t ld, rrc_buf x,
                       rrc_buf pos, rrc_buf cos_tab, rrc_buf sin_tab, int64_t n_pos, rrc_buf y) {
    return cb_rope(be, n_tokens, n_heads, head_dim, ld, x, pos, cos_tab, sin_tab, n_pos, y, 0);
}
rrc_status cb_rope_bwd(const rrc_backend *be, int64_t n_tokens, int64_t n_heads, int64_t head_dim, int64_t ld, rrc_buf dy,
                       rrc_buf pos, rrc_buf cos_tab, rrc_buf sin_tab, int64_t n_pos, rrc_buf dx) {
    return cb_rope(be, n_tokens, n_heads, head_dim, ld, dy, pos, cos_tab, sin_tab, n_pos, dx, 1);
}
rrc_status cb_softmax_fwd(const rrc_backend *be, int64_t batch, int64_t rows, int64_t cols, int64_t ld,
                          int64_t batch_stride, rrc_buf s, float scale, int64_t row_offset, int causal, rrc_buf doc_ids,
                          rrc_buf p) {
    cuda_be *c = CB(be);
    rrc_kc_softmax_p q = {batch, rows, cols, ld, batch_stride, row_offset, s.ptr, s.dtype, NULL, RRC_F32, p.ptr,
                          p.dtype, scale, causal, rrc_buf_is_none(doc_ids) ? NULL : (const int32_t *)doc_ids.ptr};
    return after(c, rrc_cu_softmax_fwd(c->stream, c->o.max_grid_blocks, &q), "attn_softmax_fwd");
}
rrc_status cb_softmax_bwd(const rrc_backend *be, int64_t batch, int64_t rows, int64_t cols, int64_t ld,
                          int64_t batch_stride, rrc_buf p, rrc_buf dp, float scale, rrc_buf ds) {
    cuda_be *c = CB(be);
    rrc_kc_softmax_p q = {batch, rows, cols, ld, batch_stride, 0, p.ptr, p.dtype, dp.ptr, dp.dtype, ds.ptr, ds.dtype,
                          scale, 0, NULL};
    return after(c, rrc_cu_softmax_bwd(c->stream, c->o.max_grid_blocks, &q), "attn_softmax_bwd");
}
rrc_status cb_swiglu_fwd(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf gate, rrc_buf up, int64_t ld_in,
                         rrc_buf h, int64_t ld_out) {
    cuda_be *c = CB(be);
    rrc_kc_swiglu_p p = {rows, cols, ld_in, 0, ld_out, gate.ptr, gate.dtype, up.ptr, up.dtype, NULL, RRC_F32,
                         h.ptr, h.dtype, NULL, RRC_F32};
    return after(c, rrc_cu_swiglu_fwd(c->stream, c->o.max_grid_blocks, &p), "swiglu_fwd");
}
rrc_status cb_swiglu_bwd(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf gate, rrc_buf up, int64_t ld_in,
                         rrc_buf dh, int64_t ld_dh, rrc_buf dgate, rrc_buf dup, int64_t ld_out) {
    cuda_be *c = CB(be);
    rrc_kc_swiglu_p p = {rows, cols, ld_in, ld_dh, ld_out, gate.ptr, gate.dtype, up.ptr, up.dtype, dh.ptr, dh.dtype,
                         dgate.ptr, dgate.dtype, dup.ptr, dup.dtype};
    return after(c, rrc_cu_swiglu_bwd(c->stream, c->o.max_grid_blocks, &p), "swiglu_bwd");
}
rrc_status cb_add(const rrc_backend *be, int64_t n, rrc_buf a, rrc_buf b, rrc_buf out) {
    cuda_be *c = CB(be);
    rrc_kc_ew_p p = {n, 0.0f, a.ptr, a.dtype, b.ptr, b.dtype, out.ptr, out.dtype};
    return after(c, rrc_cu_add(c->stream, c->o.max_grid_blocks, &p), "add");
}
rrc_status cb_axpy(const rrc_backend *be, int64_t n, float alpha, rrc_buf x, rrc_buf y) {
    cuda_be *c = CB(be);
    rrc_kc_ew_p p = {n, alpha, x.ptr, x.dtype, NULL, RRC_F32, y.ptr, y.dtype};
    return after(c, rrc_cu_axpy(c->stream, c->o.max_grid_blocks, &p), "axpy");
}
rrc_status cb_scale(const rrc_backend *be, int64_t n, float alpha, rrc_buf x) {
    cuda_be *c = CB(be);
    rrc_kc_ew_p p = {n, alpha, NULL, RRC_F32, NULL, RRC_F32, x.ptr, x.dtype};
    return after(c, rrc_cu_scale(c->stream, c->o.max_grid_blocks, &p), "scale");
}
rrc_status cb_convert(const rrc_backend *be, int64_t n, rrc_buf src, rrc_buf dst) {
    cuda_be *c = CB(be);
    if (src.dtype == dst.dtype) {
        if (src.ptr == dst.ptr) return RRC_OK;
        return after(c,
                     cudaMemcpyAsync(dst.ptr, src.ptr, (size_t)n * rrc_dtype_size(src.dtype), cudaMemcpyDeviceToDevice,
                                     c->stream),
                     "convert (copy)");
    }
    rrc_kc_ew_p p = {n, 0.0f, src.ptr, src.dtype, NULL, RRC_F32, dst.ptr, dst.dtype};
    return after(c, rrc_cu_convert(c->stream, c->o.max_grid_blocks, &p), "convert");
}
rrc_status cb_sum_squares(const rrc_backend *be, int64_t n, rrc_buf x, rrc_buf out, int accumulate) {
    cuda_be *c = CB(be);
    const int64_t nc = rrc_kc_ss_chunks(n);
    // grow() updates the members themselves: after a failed growth they are (NULL, 0), never a
    // freed pointer with its old capacity
    RRC_CHECK(grow(c, &c->ss_part, &c->ss_part_bytes, (size_t)(nc > 0 ? nc : 1) * sizeof(float), "sum_squares partials"));
    rrc_kc_ss_p p = {n, x.ptr, x.dtype, (float *)c->ss_part, (float *)out.ptr, accumulate};
    return after(c, rrc_cu_sum_squares(c->stream, c->o.max_grid_blocks, &p), "sum_squares");
}
rrc_status cb_xent(const rrc_backend *be, int64_t rows, int64_t vocab, rrc_buf logits, int64_t ld, rrc_buf labels,
                   rrc_buf loss, float grad_scale, rrc_buf row_scale, int write_grad) {
    cuda_be *c = CB(be);
    rrc_kc_xent_p p = {rows, vocab, ld, logits.ptr, logits.dtype, (const int32_t *)labels.ptr, (float *)loss.ptr,
                       rrc_buf_is_none(row_scale) ? NULL : (const float *)row_scale.ptr, grad_scale, write_grad};
    return after(c, rrc_cu_xent(c->stream, c->o.max_grid_blocks, &p, c->err_dev), "cross_entropy_fwd_bwd");
}
rrc_status cb_adamw(const rrc_backend *be, int64_t n, rrc_buf param, rrc_buf grad, rrc_buf m, rrc_buf v,
                    const rrc_adamw_hparams *h) {
    cuda_be *c = CB(be);
    rrc_kc_adamw_p p = {n, (float *)param.ptr, (const float *)grad.ptr, (float *)m.ptr, (float *)v.ptr,
                        rrc_kc_adamw_prepare(h->lr, h->beta1, h->beta2, h->eps, h->weight_decay, h->step, h->grad_scale)};
    return after(c, rrc_cu_adamw(c->stream, c->o.max_grid_blocks, &p), "adamw_step");
}

void release(cuda_be *c) {
    if (c->stream) (void)cudaStreamSynchronize(c->stream);
    for (int i = 0; i < LT_CACHE; i++) entry_release(&c->cache[i]);
    if (c->lt) cublasLtDestroy(c->lt);
    if (c->workspace) (void)cudaFree(c->workspace);
    if (c->scratch) (void)cudaFree(c->scratch);
    if (c->ss_part) (void)cudaFree(c->ss_part);
    if (c->err_dev) (void)cudaFree(c->err_dev);
    if (c->err_host) (void)cudaFreeHost(c->err_host);
    if (c->stream) (void)cudaStreamDestroy(c->stream);
    free(c);
}

rrc_status init(cuda_be *c) {
    CU(cudaSetDevice(c->device), "cudaSetDevice");
    CU(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking), "stream create");
    LT(cublasLtCreate(&c->lt), "create handle");
    c->ws_bytes = c->o.workspace_bytes;
    if (c->ws_bytes) CU(cudaMalloc(&c->workspace, c->ws_bytes), "cuBLASLt workspace");
    CU(cudaMalloc((void **)&c->err_dev, sizeof(rrc_cuda_err)), "error flag");
    CU(cudaHostAlloc((void **)&c->err_host, sizeof(rrc_cuda_err), cudaHostAllocDefault), "error flag (host)");
    RRC_CHECK(err_reset(c));
    CU(cudaStreamSynchronize(c->stream), "init synchronize");
    return RRC_OK;
}

} // namespace

extern "C" {

void rrc_cuda_options_default(rrc_cuda_options *o) {
    memset(o, 0, sizeof *o);
    o->gemm_min_level = RRC_GEMM_LEVEL_NATIVE;
    o->gemm_scratch_cap_elems = RRC_GEMM_SCRATCH_CAP_DEFAULT;
    o->workspace_bytes = (size_t)32 << 20;
    const char *e = getenv("RRC_CUDA_ALLOW_TF32");
    if (e && e[0] == '1') o->allow_tf32 = 1;
    e = getenv("RRC_CUDA_SYNC_EACH_OP");
    if (e && e[0] == '1') o->sync_each_op = 1;
}

rrc_status rrc_cuda_backend_create_ex(int device, const rrc_cuda_options *o, const rrc_backend **out) {
    RRC_REQUIRE(out, RRC_ERR_INVALID_ARG, "cuda: out is NULL");
    *out = NULL;
    cuda_be *c = (cuda_be *)calloc(1, sizeof(cuda_be));
    if (!c) return RRC_ERROR(RRC_ERR_OOM, "cuda backend");
    c->device = device;
    if (o) c->o = *o;
    else rrc_cuda_options_default(&c->o);
    const int lvl = c->o.gemm_min_level;
    if (lvl < RRC_GEMM_LEVEL_NATIVE || lvl > RRC_GEMM_LEVEL_KERNEL) {
        free(c);
        return RRC_ERROR(RRC_ERR_INVALID_ARG, "cuda: gemm_min_level %d not in [0, 2]", lvl);
    }
    rrc_status s = init(c);
    if (s != RRC_OK) {
        release(c);
        return s;
    }
    rrc_backend *t = &c->table;
    t->name = k_cuda_name;
    t->host_accessible = 0;
    t->impl = c;
    t->alloc = cb_alloc;
    t->free = cb_free;
    t->memcpy_h2d = cb_h2d;
    t->memcpy_d2h = cb_d2h;
    t->memcpy_d2d = cb_d2d;
    t->memset0 = cb_memset0;
    t->sync = cb_sync;
    t->gemm = cb_gemm;
    t->embedding_fwd = cb_embedding_fwd;
    t->add_bias = cb_add_bias;
    t->bias_grad = cb_bias_grad;
    t->rmsnorm_fwd = cb_rmsnorm_fwd;
    t->rmsnorm_bwd = cb_rmsnorm_bwd;
    t->rope_fwd = cb_rope_fwd;
    t->rope_bwd = cb_rope_bwd;
    t->attn_softmax_fwd = cb_softmax_fwd;
    t->attn_softmax_bwd = cb_softmax_bwd;
    t->swiglu_fwd = cb_swiglu_fwd;
    t->swiglu_bwd = cb_swiglu_bwd;
    t->add = cb_add;
    t->axpy = cb_axpy;
    t->scale = cb_scale;
    t->convert = cb_convert;
    t->sum_squares = cb_sum_squares;
    t->cross_entropy_fwd_bwd = cb_xent;
    t->adamw_step = cb_adamw;
    *out = t;
    return RRC_OK;
}

rrc_status rrc_cuda_backend_create(int device, const rrc_backend **out) {
    return rrc_cuda_backend_create_ex(device, NULL, out);
}

int rrc_backend_is_cuda(const rrc_backend *be) { return be && be->name == k_cuda_name; }

void rrc_cuda_backend_destroy(const rrc_backend *be) {
    if (!rrc_backend_is_cuda(be)) return;
    release(CB(be));
}

rrc_status rrc_cuda_device_info_get(const rrc_backend *be, rrc_cuda_device_info *info) {
    RRC_REQUIRE(rrc_backend_is_cuda(be) && info, RRC_ERR_INVALID_ARG, "cuda: not a CUDA backend");
    cuda_be *c = CB(be);
    memset(info, 0, sizeof *info);
    cudaDeviceProp prop;
    CU(cudaGetDeviceProperties(&prop, c->device), "cudaGetDeviceProperties");
    snprintf(info->name, sizeof info->name, "%s", prop.name);
    info->device = c->device;
    info->cc_major = prop.major;
    info->cc_minor = prop.minor;
    info->sm_count = prop.multiProcessorCount;
    info->total_mem = prop.totalGlobalMem;
    CU(cudaMemGetInfo(&info->free_mem, &info->total_mem), "cudaMemGetInfo");
    CU(cudaDriverGetVersion(&info->driver_version), "cudaDriverGetVersion");
    CU(cudaRuntimeGetVersion(&info->runtime_version), "cudaRuntimeGetVersion");
    info->cublaslt_version = cublasLtGetVersion();
    return RRC_OK;
}

int rrc_cuda_last_gemm(const rrc_backend *be, rrc_gemm_exec_report *rep) {
    if (!rrc_backend_is_cuda(be) || !CB(be)->have_rep) return 0;
    if (rep) *rep = CB(be)->rep;
    return 1;
}

} // extern "C"
