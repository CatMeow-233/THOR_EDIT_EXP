/* rrc/tests/cuda_host/shim.h — HOST-ONLY test build of the real rrc/cuda .cu sources (DESIGN.md
 * §11.4). Force-included (g++ -include) before cuda_kernels.cu and cuda_backend.cu, which are
 * compiled as plain C++ (no __CUDACC__) together with -DRRC_CUDA_HOST_SHIM. The toolkit headers
 * supply the types only; fake_cuda.cpp implements the runtime and cuBLASLt entry points used by the
 * backend on host memory, and runs every kernel launch on a pool of host threads:
 *   - one host thread per CUDA thread of a block (blockDim.x threads), blocks run one after the
 *     other (so a function-local static stands in for __shared__ memory); kernels launched with
 *     RRC_CU_LAUNCH_NOSYNC (no __syncthreads) run their threads one after the other instead;
 *   - __syncthreads is a barrier of the block's threads;
 *   - atomicCAS / atomicMin are host atomics.
 * This executes the kernel shells (grid-stride and row-stride loops, the shared-memory trees,
 * lane 0's finalizers, the error flag) and the whole backend on a CPU. It says nothing about
 * device code generation, the real cuBLASLt, or timing. Never part of librrc_cuda. */
#ifndef RRC_TESTS_CUDA_HOST_SHIM_H
#define RRC_TESTS_CUDA_HOST_SHIM_H

#ifdef __cplusplus
#include <cuda_runtime.h>

#include <functional>

#undef __host__
#undef __device__
#undef __global__
#undef __shared__
#undef __launch_bounds__
#define __host__
#define __device__
#define __global__
#define __shared__ static
#define __launch_bounds__(...)

struct rrc_shim_dim3 {
    unsigned x, y, z;
};
extern thread_local rrc_shim_dim3 rrc_shim_tid, rrc_shim_bid;
extern rrc_shim_dim3 rrc_shim_gdim, rrc_shim_bdim;

#undef threadIdx
#undef blockIdx
#undef gridDim
#undef blockDim
#define threadIdx rrc_shim_tid
#define blockIdx rrc_shim_bid
#define gridDim rrc_shim_gdim
#define blockDim rrc_shim_bdim

void rrc_shim_syncthreads(void);
#define __syncthreads() rrc_shim_syncthreads()

static inline unsigned atomicCAS(unsigned *p, unsigned cmp, unsigned val) {
    unsigned c = cmp;
    __atomic_compare_exchange_n(p, &c, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return c; /* the old value, as on the device */
}
static inline unsigned long long atomicMin(unsigned long long *p, unsigned long long v) {
    unsigned long long o = __atomic_load_n(p, __ATOMIC_SEQ_CST);
    while (v < o && !__atomic_compare_exchange_n(p, &o, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
    }
    return o;
}

/* Runs fn for grid x block CUDA threads (blocks in order); returns when all are done. With
 * `syncs`, a block's threads run concurrently on a pool of host threads (__syncthreads is a
 * barrier); without, they run one after the other on the calling thread and __syncthreads aborts. */
void rrc_shim_launch(unsigned grid, unsigned block, int syncs, const std::function<void()> &fn);
#endif /* __cplusplus */

#endif
