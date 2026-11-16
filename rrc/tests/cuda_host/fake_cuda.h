/* rrc/tests/cuda_host/fake_cuda.h — controls and counters of the fake CUDA runtime / cuBLASLt of
 * the HOST-ONLY test build (fake_cuda.cpp). C and C++. */
#ifndef RRC_TESTS_FAKE_CUDA_H
#define RRC_TESTS_FAKE_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

/* Simulated cuBLASLt heuristic (env RRC_FAKE_LT = accept | reject | mask-drops-nonsplit):
 *   ACCEPT   an algorithm for every call whose types are in the expected table (f32 f32 f32;
 *            bf16 bf16 f32; bf16 bf16 bf16; COMPUTE_32F[_FAST_TF32 for f32], scale f32);
 *   REJECT   CUBLAS_STATUS_NOT_SUPPORTED for every call (a misconfigured integration);
 *   MASK_DROPS_NONSPLIT  a preference with a reduction-scheme mask other than the default gets
 *            no algorithm (as if a mask without a NONE bit excluded the non-split-K algorithms);
 *            without a mask, the first result is a split-K algorithm with the INPLACE scheme
 *            (bf16 partial sums: must not be chosen) and the second a non-split one. */
enum { RRC_FAKE_LT_ACCEPT = 0, RRC_FAKE_LT_REJECT = 1, RRC_FAKE_LT_MASK_DROPS_NONSPLIT = 2 };
void rrc_fake_set_lt_mode(int mode);

/* The n-th cudaMalloc from now (0 = the next one) fails with cudaErrorMemoryAllocation; -1: none. */
void rrc_fake_fail_malloc_after(int n);
/* The same for cudaHostAlloc (pinned host memory). */
void rrc_fake_fail_host_alloc_after(int n);

/* Host -> device copies are LAZY: cudaMemcpyAsync(HostToDevice) only queues (dst, src, n) and
 * reads the source when the queue is flushed, as a DMA engine reads it some time after the call.
 * The queue is flushed, in order, before any other stream operation (kernel launch, cuBLASLt
 * matmul, device/host-bound copy, memset), by cudaStreamSynchronize / cudaDeviceSynchronize, by
 * cudaEventSynchronize up to the copies enqueued before the event's record, and by cudaFree /
 * cudaFreeHost. So code that overwrites a host source (e.g. a staging slot) before the copy that
 * reads it is known complete delivers the overwritten bytes, and a test sees it. */

typedef struct rrc_fake_stats {
    long mallocs, frees;
    long bad_frees;      /* cudaFree of a pointer that is not live (double free, foreign pointer) */
    long live;           /* device allocations currently live */
    long launches;       /* kernel launches */
    long lt_heuristics;  /* cublasLtMatmulAlgoGetHeuristic calls */
    long lt_matmuls;     /* cublasLtMatmul calls */
    long lt_violations;  /* inconsistent descriptors, unsupported types, broken alignment promises,
                          * a split-K algorithm with a non-fp32 reduction passed to cublasLtMatmul */
    long h2d_async;      /* cudaMemcpyAsync host -> device calls */
    long h2d_pageable;   /* ... of them with a source outside every live cudaHostAlloc block */
    long h2d_pending;    /* host -> device copies queued and not yet executed */
    long stream_syncs;   /* cudaStreamSynchronize calls */
    long event_syncs;    /* cudaEventSynchronize calls */
    long host_allocs_live; /* cudaHostAlloc blocks currently live */
} rrc_fake_stats;
void rrc_fake_get_stats(rrc_fake_stats *s);

/* Stops the launch thread pool (also registered with atexit). */
void rrc_fake_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
