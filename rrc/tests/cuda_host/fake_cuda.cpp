// rrc/tests/cuda_host/fake_cuda.cpp — fake CUDA runtime + cuBLASLt + kernel-launch thread pool for
// the HOST-ONLY test build of rrc/cuda/*.cu (see shim.h). "Device" memory is host memory (so the
// ASan build sees device-side use-after-free and overruns); streams are synchronous. The cuBLASLt
// stand-in implements the documented column-major semantics (D = alpha op(A) op(B) + beta C,
// layouts rows x cols with ld, strided batch) and checks descriptor consistency, the type table the
// backend assumes, and the alignment the backend promised in its preference. It is evidence about
// rrc's own code paths only, never about the real library.
#include "shim.h"

#include <cublasLt.h>

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <new>
#include <set>
#include <vector>

#include "fake_cuda.h"
#include "rrc/common.h"

// ---- lazy host -> device copies (fake_cuda.h) ----------------------------------------------------

namespace {
struct pending_h2d {
    void *dst;
    const void *src;
    size_t n;
};
std::vector<pending_h2d> g_pending; // in stream order
unsigned long long g_h2d_enq = 0, g_h2d_done = 0; // copies enqueued / executed so far

// Execute the queued copies up to (excluding) sequence number `upto` (all: ~0ull).
void flush_h2d(unsigned long long upto) {
    size_t i = 0;
    while (i < g_pending.size() && g_h2d_done < upto) {
        memmove(g_pending[i].dst, g_pending[i].src, g_pending[i].n);
        g_h2d_done++;
        i++;
    }
    g_pending.erase(g_pending.begin(), g_pending.begin() + (long)i);
}
void flush_all() { flush_h2d(~0ull); }
} // namespace

// ---- kernel launches ---------------------------------------------------------------------------

thread_local rrc_shim_dim3 rrc_shim_tid, rrc_shim_bid;
rrc_shim_dim3 rrc_shim_gdim, rrc_shim_bdim;

namespace {

const unsigned k_pool = 256; // every rrc kernel uses 256 threads per block
std::mutex g_mu;
std::condition_variable g_cv_start, g_cv_done;
const std::function<void()> *g_job = nullptr;
unsigned g_job_grid = 0, g_job_block = 0, g_running = 0;
unsigned long g_gen = 0;
bool g_stop = false, g_started = false;
pthread_barrier_t g_bar;
pthread_t g_threads[k_pool];

void *worker(void *arg) {
    const unsigned t = (unsigned)(uintptr_t)arg;
    unsigned long seen = 0;
    for (;;) {
        std::unique_lock<std::mutex> lk(g_mu);
        g_cv_start.wait(lk, [&] { return g_stop || g_gen != seen; });
        if (g_stop) return nullptr;
        seen = g_gen;
        const std::function<void()> *fn = g_job;
        const unsigned grid = g_job_grid, block = g_job_block;
        lk.unlock();
        if (t < block) {
            rrc_shim_tid = {t, 0, 0};
            for (unsigned b = 0; b < grid; b++) {
                rrc_shim_bid = {b, 0, 0};
                (*fn)();
                pthread_barrier_wait(&g_bar); // the block is complete before the next one starts
            }
        }
        lk.lock();
        if (--g_running == 0) g_cv_done.notify_all();
    }
}

void start_pool() {
    for (unsigned t = 0; t < k_pool; t++)
        if (pthread_create(&g_threads[t], nullptr, worker, (void *)(uintptr_t)t) != 0) {
            fprintf(stderr, "fake_cuda: pthread_create failed\n");
            abort();
        }
    g_started = true;
    atexit(rrc_fake_shutdown);
}

long g_launches = 0;

} // namespace

namespace {
bool g_sequential = false;
}

void rrc_shim_syncthreads(void) {
    if (g_sequential) {
        fprintf(stderr, "fake_cuda: __syncthreads in a kernel launched with RRC_CU_LAUNCH_NOSYNC\n");
        abort();
    }
    pthread_barrier_wait(&g_bar);
}

void rrc_shim_launch(unsigned grid, unsigned block, int syncs, const std::function<void()> &fn) {
    if (block == 0 || block > k_pool || grid == 0) {
        fprintf(stderr, "fake_cuda: unsupported launch <<<%u, %u>>>\n", grid, block);
        abort();
    }
    g_launches++;
    flush_all(); // the kernel runs after every copy enqueued before it
    rrc_shim_gdim = {grid, 1, 1};
    rrc_shim_bdim = {block, 1, 1};
    if (!syncs) { // no barrier: the block's threads one after the other, on this thread
        g_sequential = true;
        const rrc_shim_dim3 tid0 = rrc_shim_tid, bid0 = rrc_shim_bid;
        for (unsigned b = 0; b < grid; b++)
            for (unsigned t = 0; t < block; t++) {
                rrc_shim_bid = {b, 0, 0};
                rrc_shim_tid = {t, 0, 0};
                fn();
            }
        rrc_shim_tid = tid0;
        rrc_shim_bid = bid0;
        g_sequential = false;
        return;
    }
    if (!g_started) start_pool();
    pthread_barrier_init(&g_bar, nullptr, block);
    {
        std::unique_lock<std::mutex> lk(g_mu);
        g_job = &fn;
        g_job_grid = grid;
        g_job_block = block;
        g_running = k_pool;
        g_gen++;
        g_cv_start.notify_all();
        g_cv_done.wait(lk, [] { return g_running == 0; });
    }
    pthread_barrier_destroy(&g_bar);
}

extern "C" void rrc_fake_shutdown(void) {
    if (!g_started) return;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_stop = true;
        g_cv_start.notify_all();
    }
    for (unsigned t = 0; t < k_pool; t++) pthread_join(g_threads[t], nullptr);
    g_started = false;
    g_stop = false;
}

// ---- runtime -------------------------------------------------------------------------------------

namespace {
std::set<void *> g_live;
std::map<uintptr_t, size_t> g_pinned; // live cudaHostAlloc blocks
long g_mallocs = 0, g_frees = 0, g_bad_frees = 0;
long g_h2d_async = 0, g_h2d_pageable = 0, g_stream_syncs = 0, g_event_syncs = 0;
int g_fail_after = -1, g_fail_host_after = -1;
bool is_pinned(const void *p, size_t n) {
    const uintptr_t a = (uintptr_t)p;
    auto it = g_pinned.upper_bound(a);
    if (it == g_pinned.begin()) return false;
    --it;
    return a >= it->first && a + n <= it->first + it->second;
}
struct fake_event {
    unsigned long long seq; // copies enqueued before the record
    int recorded;
    double t_ms;            // host wall clock at the record (timing events only; never GPU time)
};
double host_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
int g_lt_mode = -1; // -1: from the environment on first use
long g_lt_heur = 0, g_lt_runs = 0, g_lt_viol = 0;
int dummy_handle;

int lt_mode() {
    if (g_lt_mode < 0) {
        const char *e = getenv("RRC_FAKE_LT");
        g_lt_mode = RRC_FAKE_LT_ACCEPT;
        if (e && !strcmp(e, "reject")) g_lt_mode = RRC_FAKE_LT_REJECT;
        if (e && !strcmp(e, "mask-drops-nonsplit")) g_lt_mode = RRC_FAKE_LT_MASK_DROPS_NONSPLIT;
    }
    return g_lt_mode;
}
} // namespace

#define VIOL(...)                                                                                       \
    do {                                                                                                \
        g_lt_viol++;                                                                                    \
        fprintf(stderr, "fake cuBLASLt: " __VA_ARGS__);                                                 \
        fprintf(stderr, "\n");                                                                          \
    } while (0)

extern "C" {

void rrc_fake_set_lt_mode(int mode) { g_lt_mode = mode; }
void rrc_fake_fail_malloc_after(int n) { g_fail_after = n; }
void rrc_fake_fail_host_alloc_after(int n) { g_fail_host_after = n; }
void rrc_fake_get_stats(rrc_fake_stats *s) {
    s->mallocs = g_mallocs;
    s->frees = g_frees;
    s->bad_frees = g_bad_frees;
    s->live = (long)g_live.size();
    s->launches = g_launches;
    s->lt_heuristics = g_lt_heur;
    s->lt_matmuls = g_lt_runs;
    s->lt_violations = g_lt_viol;
    s->h2d_async = g_h2d_async;
    s->h2d_pageable = g_h2d_pageable;
    s->h2d_pending = (long)g_pending.size();
    s->stream_syncs = g_stream_syncs;
    s->event_syncs = g_event_syncs;
    s->host_allocs_live = (long)g_pinned.size();
}

const char *cudaGetErrorName(cudaError_t e) {
    return e == cudaSuccess ? "cudaSuccess" : (e == cudaErrorMemoryAllocation ? "cudaErrorMemoryAllocation" : "cudaErrorFake");
}
const char *cudaGetErrorString(cudaError_t e) { return e == cudaSuccess ? "no error" : "fake CUDA error"; }
cudaError_t cudaGetLastError(void) { return cudaSuccess; }
cudaError_t cudaMalloc(void **p, size_t n) {
    if (g_fail_after == 0) {
        g_fail_after = -1;
        *p = nullptr;
        return cudaErrorMemoryAllocation;
    }
    if (g_fail_after > 0) g_fail_after--;
    void *q = nullptr;
    if (posix_memalign(&q, 256, (n + 255) & ~(size_t)255) != 0 || !q) {
        *p = nullptr;
        return cudaErrorMemoryAllocation;
    }
    g_mallocs++;
    g_live.insert(q);
    *p = q;
    return cudaSuccess;
}
cudaError_t cudaFree(void *p) {
    flush_all(); // cudaFree synchronizes the device
    if (!p) return cudaSuccess;
    if (!g_live.count(p)) {
        g_bad_frees++;
        fprintf(stderr, "fake CUDA: cudaFree of a pointer that is not live: %p\n", p);
        return cudaErrorInvalidValue;
    }
    g_live.erase(p);
    g_frees++;
    free(p);
    return cudaSuccess;
}
cudaError_t cudaHostAlloc(void **p, size_t n, unsigned int) {
    if (g_fail_host_after == 0) {
        g_fail_host_after = -1;
        *p = nullptr;
        return cudaErrorMemoryAllocation;
    }
    if (g_fail_host_after > 0) g_fail_host_after--;
    *p = malloc(n ? n : 1);
    if (!*p) return cudaErrorMemoryAllocation;
    g_pinned[(uintptr_t)*p] = n ? n : 1;
    return cudaSuccess;
}
cudaError_t cudaFreeHost(void *p) {
    flush_all(); // synchronizes the device
    if (p) g_pinned.erase((uintptr_t)p);
    free(p);
    return cudaSuccess;
}
cudaError_t cudaMemsetAsync(void *p, int v, size_t n, cudaStream_t) {
    flush_all();
    memset(p, v, n);
    return cudaSuccess;
}
cudaError_t cudaMemcpyAsync(void *d, const void *s, size_t n, enum cudaMemcpyKind k, cudaStream_t) {
    if (k == cudaMemcpyHostToDevice) { // lazy: the source is read at the next flush
        g_h2d_async++;
        if (!is_pinned(s, n)) g_h2d_pageable++;
        g_pending.push_back(pending_h2d{d, s, n});
        g_h2d_enq++;
        return cudaSuccess;
    }
    flush_all();
    memmove(d, s, n);
    return cudaSuccess;
}
cudaError_t cudaStreamSynchronize(cudaStream_t) {
    g_stream_syncs++;
    flush_all();
    return cudaSuccess;
}
cudaError_t cudaDeviceSynchronize(void) {
    flush_all();
    return cudaSuccess;
}
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *e, unsigned int) {
    fake_event *f = new (std::nothrow) fake_event{0, 0, 0.0};
    *e = (cudaEvent_t)f;
    return f ? cudaSuccess : cudaErrorMemoryAllocation;
}
cudaError_t cudaEventCreate(cudaEvent_t *e) { return cudaEventCreateWithFlags(e, 0); }
cudaError_t cudaEventDestroy(cudaEvent_t e) {
    delete (fake_event *)e;
    return cudaSuccess;
}
cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t) {
    fake_event *f = (fake_event *)e;
    f->seq = g_h2d_enq;
    f->recorded = 1;
    f->t_ms = host_ms();
    return cudaSuccess;
}
cudaError_t cudaEventSynchronize(cudaEvent_t e) {
    g_event_syncs++;
    fake_event *f = (fake_event *)e;
    if (f->recorded) flush_h2d(f->seq); // only the copies enqueued before the record
    return cudaSuccess;
}
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t a, cudaEvent_t b) {
    const fake_event *x = (const fake_event *)a, *y = (const fake_event *)b;
    if (!x->recorded || !y->recorded) return cudaErrorInvalidResourceHandle;
    *ms = (float)(y->t_ms - x->t_ms); // host wall clock: the HOST-ONLY build has no GPU time
    return cudaSuccess;
}
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *a, const void *) {
    memset(a, 0, sizeof *a); // no device image: versions 0
    return cudaSuccess;
}
cudaError_t cudaSetDevice(int) { return cudaSuccess; }
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *s, unsigned int) {
    *s = (cudaStream_t)&dummy_handle;
    return cudaSuccess;
}
cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *p, int) {
    memset(p, 0, sizeof *p);
    snprintf(p->name, sizeof p->name, "fake host device (HOST-ONLY test build, no GPU)");
    return cudaSuccess;
}
cudaError_t cudaMemGetInfo(size_t *f, size_t *t) {
    *f = *t = 0;
    return cudaSuccess;
}
cudaError_t cudaDriverGetVersion(int *v) {
    *v = 0;
    return cudaSuccess;
}
cudaError_t cudaRuntimeGetVersion(int *v) {
    *v = 0;
    return cudaSuccess;
}

} // extern "C"

// ---- cuBLASLt -------------------------------------------------------------------------------------

namespace {
struct Lay {
    cudaDataType_t t;
    uint64_t rows, cols;
    int64_t ld;
    int32_t batch;
    int64_t stride;
};
struct Desc {
    cublasComputeType_t ct;
    cudaDataType_t st;
    cublasOperation_t ta, tb;
};
struct Pref {
    uint64_t ws;
    uint32_t red;
    int red_set;
    uint32_t align[4];
};
// algo.data: [0..3] promised alignment A/B/C/D, [4] workspace, [5] split-K, [6] reduction scheme
size_t esz(cudaDataType_t t) { return t == CUDA_R_16BF ? 2 : 4; }
float ldv(const void *p, cudaDataType_t t, int64_t i) {
    return t == CUDA_R_16BF ? rrc_bf16_to_f32(((const rrc_bf16 *)p)[i]) : ((const float *)p)[i];
}
void stv(void *p, cudaDataType_t t, int64_t i, float v) {
    if (t == CUDA_R_16BF) ((rrc_bf16 *)p)[i] = rrc_f32_to_bf16(v);
    else ((float *)p)[i] = v;
}
bool type_ok(const Desc *d, const Lay *a, const Lay *b, const Lay *c) {
    if (d->st != CUDA_R_32F) return false;
    const bool fff = a->t == CUDA_R_32F && b->t == CUDA_R_32F && c->t == CUDA_R_32F;
    if (d->ct == CUBLAS_COMPUTE_32F_FAST_TF32) return fff;
    if (d->ct != CUBLAS_COMPUTE_32F) return false;
    return fff || (a->t == CUDA_R_16BF && b->t == CUDA_R_16BF && (c->t == CUDA_R_16BF || c->t == CUDA_R_32F));
}
bool dims(const Desc *d, const Lay *a, const Lay *b, const Lay *c, uint64_t *M, uint64_t *N, uint64_t *K) {
    *M = c->rows;
    *N = c->cols;
    const uint64_t am = d->ta == CUBLAS_OP_N ? a->rows : a->cols, ak = d->ta == CUBLAS_OP_N ? a->cols : a->rows;
    const uint64_t bk = d->tb == CUBLAS_OP_N ? b->rows : b->cols, bn = d->tb == CUBLAS_OP_N ? b->cols : b->rows;
    *K = ak;
    return am == *M && bn == *N && bk == ak && a->ld >= (int64_t)a->rows && b->ld >= (int64_t)b->rows &&
           c->ld >= (int64_t)c->rows && a->batch == c->batch && b->batch == c->batch;
}
void chk_align(const char *nm, const void *ptr, const Lay *l, uint64_t need) {
    for (int32_t t = 0; t < l->batch; t++) {
        const uintptr_t a = (uintptr_t)ptr + (uintptr_t)((int64_t)t * l->stride) * esz(l->t);
        if (need && a % need) {
            VIOL("matmul: %s batch %d at %p not aligned to the promised %llu bytes", nm, t, (void *)a,
                 (unsigned long long)need);
            return;
        }
    }
}
} // namespace

extern "C" {

size_t cublasLtGetVersion(void) { return 0; }
cublasStatus_t cublasLtCreate(cublasLtHandle_t *h) {
    *h = (cublasLtHandle_t)&dummy_handle;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasLtDestroy(cublasLtHandle_t) { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *l, cudaDataType t, uint64_t rows, uint64_t cols,
                                          int64_t ld) {
    *l = (cublasLtMatrixLayout_t) new Lay{t, rows, cols, ld, 1, 0};
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t l) {
    delete (Lay *)l;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t l, cublasLtMatrixLayoutAttribute_t a,
                                                const void *b, size_t n) {
    Lay *x = (Lay *)l;
    if (a == CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT && n == 4) memcpy(&x->batch, b, 4);
    else if (a == CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET && n == 8) memcpy(&x->stride, b, 8);
    else return CUBLAS_STATUS_INVALID_VALUE;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *d, cublasComputeType_t ct, cudaDataType_t st) {
    *d = (cublasLtMatmulDesc_t) new Desc{ct, st, CUBLAS_OP_N, CUBLAS_OP_N};
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t d) {
    delete (Desc *)d;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t d, cublasLtMatmulDescAttributes_t a, const void *b,
                                              size_t n) {
    Desc *x = (Desc *)d;
    if (n != sizeof(cublasOperation_t)) return CUBLAS_STATUS_INVALID_VALUE;
    if (a == CUBLASLT_MATMUL_DESC_TRANSA) memcpy(&x->ta, b, n);
    else if (a == CUBLASLT_MATMUL_DESC_TRANSB) memcpy(&x->tb, b, n);
    else return CUBLAS_STATUS_INVALID_VALUE;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *p) {
    *p = (cublasLtMatmulPreference_t) new Pref{0, CUBLASLT_REDUCTION_SCHEME_MASK, 0, {256, 256, 256, 256}};
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t p) {
    delete (Pref *)p;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t p, cublasLtMatmulPreferenceAttributes_t a,
                                                    const void *b, size_t n) {
    Pref *x = (Pref *)p;
    switch (a) {
    case CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES:
        if (n != 8) return CUBLAS_STATUS_INVALID_VALUE;
        memcpy(&x->ws, b, 8);
        break;
    case CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK:
        if (n != 4) return CUBLAS_STATUS_INVALID_VALUE;
        memcpy(&x->red, b, 4);
        x->red_set = 1;
        break;
    case CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES:
    case CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES:
    case CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES:
    case CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES: {
        if (n != 4) return CUBLAS_STATUS_INVALID_VALUE;
        const int i = a == CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES   ? 0
                      : a == CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES ? 1
                      : a == CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES ? 2
                                                                         : 3;
        memcpy(&x->align[i], b, 4);
    } break;
    default: return CUBLAS_STATUS_INVALID_VALUE;
    }
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t, cublasLtMatmulDesc_t od, cublasLtMatrixLayout_t oa,
                                              cublasLtMatrixLayout_t ob, cublasLtMatrixLayout_t oc,
                                              cublasLtMatrixLayout_t odd, cublasLtMatmulPreference_t op, int req,
                                              cublasLtMatmulHeuristicResult_t *res, int *nres) {
    g_lt_heur++;
    const Desc *d = (const Desc *)od;
    const Lay *a = (const Lay *)oa, *b = (const Lay *)ob, *c = (const Lay *)oc, *dd = (const Lay *)odd;
    const Pref *p = (const Pref *)op;
    *nres = 0;
    if (req < 1) return CUBLAS_STATUS_INVALID_VALUE;
    uint64_t M, N, K;
    if (!dims(d, a, b, c, &M, &N, &K)) {
        VIOL("heuristic: inconsistent layouts");
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (memcmp(c, dd, sizeof *c) != 0) return CUBLAS_STATUS_NOT_SUPPORTED;
    const int mode = lt_mode();
    if (mode == RRC_FAKE_LT_REJECT || !type_ok(d, a, b, c)) return CUBLAS_STATUS_NOT_SUPPORTED;
    if (mode == RRC_FAKE_LT_MASK_DROPS_NONSPLIT && p->red_set && p->red != CUBLASLT_REDUCTION_SCHEME_MASK)
        return CUBLAS_STATUS_NOT_SUPPORTED;
    struct cfg {
        int32_t splitk;
        uint32_t scheme;
    };
    const cfg accept[1] = {{1, CUBLASLT_REDUCTION_SCHEME_NONE}};
    const cfg nomask[2] = {{4, CUBLASLT_REDUCTION_SCHEME_INPLACE}, {1, CUBLASLT_REDUCTION_SCHEME_NONE}};
    const cfg *list = mode == RRC_FAKE_LT_MASK_DROPS_NONSPLIT ? nomask : accept;
    const int n_list = mode == RRC_FAKE_LT_MASK_DROPS_NONSPLIT ? 2 : 1;
    for (int i = 0; i < n_list && i < req; i++) {
        memset(&res[i], 0, sizeof res[i]);
        for (int j = 0; j < 4; j++) res[i].algo.data[j] = p->align[j];
        res[i].algo.data[4] = p->ws;
        res[i].algo.data[5] = (uint64_t)list[i].splitk;
        res[i].algo.data[6] = list[i].scheme;
        res[i].state = CUBLAS_STATUS_SUCCESS;
        res[i].workspaceSize = 0;
        (*nres)++;
    }
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t *algo,
                                                    cublasLtMatmulAlgoConfigAttributes_t attr, void *buf,
                                                    size_t size, size_t *written) {
    if (attr == CUBLASLT_ALGO_CONFIG_SPLITK_NUM && size == 4) {
        int32_t v = (int32_t)algo->data[5];
        memcpy(buf, &v, 4);
    } else if (attr == CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME && size == 4) {
        uint32_t v = (uint32_t)algo->data[6];
        memcpy(buf, &v, 4);
    } else {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (written) *written = 4;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasLtMatmul(cublasLtHandle_t, cublasLtMatmulDesc_t od, const void *alpha, const void *A,
                              cublasLtMatrixLayout_t oa, const void *B, cublasLtMatrixLayout_t ob, const void *beta,
                              const void *C, cublasLtMatrixLayout_t oc, void *Dp, cublasLtMatrixLayout_t odd,
                              const cublasLtMatmulAlgo_t *algo, void *, size_t, cudaStream_t) {
    g_lt_runs++;
    flush_all(); // stream order: after every copy enqueued before it
    const Desc *d = (const Desc *)od;
    const Lay *a = (const Lay *)oa, *b = (const Lay *)ob, *c = (const Lay *)oc, *dd = (const Lay *)odd;
    uint64_t M, N, K;
    if (!dims(d, a, b, c, &M, &N, &K) || !type_ok(d, a, b, c)) {
        VIOL("matmul: unsupported or inconsistent call");
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!algo) return CUBLAS_STATUS_INVALID_VALUE;
    if (algo->data[5] > 1 && algo->data[6] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)
        VIOL("matmul: split-K algorithm with reduction scheme %llu (partial sums not in fp32)",
             (unsigned long long)algo->data[6]);
    chk_align("A", A, a, algo->data[0]);
    chk_align("B", B, b, algo->data[1]);
    chk_align("C", C, c, algo->data[2]);
    chk_align("D", Dp, dd, algo->data[3]);
    const float al = *(const float *)alpha, be = *(const float *)beta;
    std::vector<float> out((size_t)(M * N));
    for (int32_t t = 0; t < c->batch; t++) {
        const char *pa = (const char *)A + (size_t)((int64_t)t * a->stride) * esz(a->t);
        const char *pb = (const char *)B + (size_t)((int64_t)t * b->stride) * esz(b->t);
        const char *pc = (const char *)C + (size_t)((int64_t)t * c->stride) * esz(c->t);
        char *pd = (char *)Dp + (size_t)((int64_t)t * dd->stride) * esz(dd->t);
        for (uint64_t j = 0; j < N; j++)
            for (uint64_t i = 0; i < M; i++) {
                double s = 0.0;
                for (uint64_t q = 0; q < K; q++) {
                    const float x = d->ta == CUBLAS_OP_N ? ldv(pa, a->t, (int64_t)(i + q * a->ld))
                                                         : ldv(pa, a->t, (int64_t)(q + i * a->ld));
                    const float y = d->tb == CUBLAS_OP_N ? ldv(pb, b->t, (int64_t)(q + j * b->ld))
                                                         : ldv(pb, b->t, (int64_t)(j + q * b->ld));
                    s += (double)x * (double)y;
                }
                float v = al * (float)s;
                if (be != 0.0f) v += be * ldv(pc, c->t, (int64_t)(i + j * c->ld)); // beta == 0: C not read
                out[i + j * M] = v;
            }
        // all of a slice is computed before any of it is stored (D == C in place)
        for (uint64_t j = 0; j < N; j++)
            for (uint64_t i = 0; i < M; i++) stv(pd, dd->t, (int64_t)(i + j * dd->ld), out[i + j * M]);
    }
    return CUBLAS_STATUS_SUCCESS;
}

} // extern "C"
