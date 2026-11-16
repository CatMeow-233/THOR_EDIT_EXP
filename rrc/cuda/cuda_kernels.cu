// rrc/cuda/cuda_kernels.cu — CUDA kernels of the backend (DESIGN.md §11.3). COMPILED-ONLY.
//
// Every kernel is a thin shell over an RRC_HD body of kernels_common/rrc_kc_items.h:
//   - element kernels: a grid-stride loop over item indices calling one body per item;
//   - row kernels: one block of RRC_KC_LANES threads per row (grid-stride over rows). Thread `lane`
//     computes the lane partial, the block runs the fixed tree (one __syncthreads per level), lane 0
//     finalizes the row statistics into shared memory, then the threads run the element items of
//     the row. The order of every reduction depends only on the shape, never on gridDim or on
//     scheduling, so results are bit-identical run to run.
// The same bodies run on the CPU, in shuffled order, in rrc/cuda/kc_emu.c (tests/c/test_kc_items.c).
// Compiled with -fmad=false -prec-div=true -prec-sqrt=true -ftz=false (Makefile): no contraction of
// a*b+c into fma (the bodies call fmaf where the CPU semantics have one), IEEE division and sqrt.
#include <string.h>

#include "cuda_internal.h"

#define LANES RRC_KC_LANES

static_assert((LANES & (LANES - 1)) == 0, "RRC_KC_LANES must be a power of two");

static unsigned grid_cap(unsigned max_blocks) {
    return max_blocks == 0 || max_blocks > RRC_CU_MAX_BLOCKS ? RRC_CU_MAX_BLOCKS : max_blocks;
}
static unsigned grid_elems(int64_t n, unsigned max_blocks) {
    const int64_t b = (n + 255) / 256, cap = grid_cap(max_blocks);
    return (unsigned)(b < 1 ? 1 : (b > cap ? cap : b));
}
static unsigned grid_rows(int64_t rows, unsigned max_blocks) {
    const int64_t cap = grid_cap(max_blocks);
    return (unsigned)(rows < 1 ? 1 : (rows > cap ? cap : rows));
}

// Kernel launch. The host test build (rrc/tests/cuda_host, -DRRC_CUDA_HOST_SHIM, HOST-ONLY) runs
// the same kernels on host threads instead; nvcc always takes the <<<>>> form for both macros.
// RRC_CU_LAUNCH_NOSYNC marks kernels without __syncthreads (the element kernels), which the host
// build may run one CUDA thread after the other (it aborts if such a kernel synchronizes).
#if defined(RRC_CUDA_HOST_SHIM) && !defined(__CUDACC__)
#define RRC_CU_LAUNCH(kernel, grid, block, s, ...) ((void)(s), rrc_shim_launch((grid), (block), 1, [=] { kernel(__VA_ARGS__); }))
#define RRC_CU_LAUNCH_NOSYNC(kernel, grid, block, s, ...) ((void)(s), rrc_shim_launch((grid), (block), 0, [=] { kernel(__VA_ARGS__); }))
#else
#define RRC_CU_LAUNCH(kernel, grid, block, s, ...) kernel<<<(grid), (block), 0, (s)>>>(__VA_ARGS__)
#define RRC_CU_LAUNCH_NOSYNC(kernel, grid, block, s, ...) kernel<<<(grid), (block), 0, (s)>>>(__VA_ARGS__)
#endif

#define GRID_STRIDE(i, n) for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (int64_t)gridDim.x * blockDim.x)
#define ROW_STRIDE(row, rows) for (int64_t row = blockIdx.x; row < (rows); row += gridDim.x)

// Records a range error: the first failing op wins the CAS, and only items of that op lower the
// index (an item of a later failing op must not put its index under the first op's name). Ops on
// one stream run in order, so every item of the recorded op has done its CAS before a later op's.
__device__ static void dev_range(rrc_cuda_err *err, unsigned op, int64_t index) {
    const unsigned prev = atomicCAS(&err->op, 0u, op);
    if (prev == 0u || prev == op) atomicMin(&err->index, (unsigned long long)index);
}

// Fixed trees over shared memory (all LANES threads call them; they end synchronized).
__device__ static void tree_sum(float *sh, int lane) {
    __syncthreads();
    for (int s = LANES / 2; s > 0; s >>= 1) {
        rrc_kc_tree_sum_step(sh, lane, s);
        __syncthreads();
    }
}
__device__ static void tree_sum2(float *a, float *b, int lane) {
    __syncthreads();
    for (int s = LANES / 2; s > 0; s >>= 1) {
        rrc_kc_tree_sum_step(a, lane, s);
        rrc_kc_tree_sum_step(b, lane, s);
        __syncthreads();
    }
}
__device__ static void tree_max(float *sh, int lane) {
    __syncthreads();
    for (int s = LANES / 2; s > 0; s >>= 1) {
        rrc_kc_tree_max_step(sh, lane, s);
        __syncthreads();
    }
}
__device__ static void tree_lse(rrc_kc_lse *sh, int lane) {
    __syncthreads();
    for (int s = LANES / 2; s > 0; s >>= 1) {
        rrc_kc_tree_lse_step(sh, lane, s);
        __syncthreads();
    }
}

// ---- element kernels -------------------------------------------------------------------------

__global__ void k_lt_matmul(rrc_kc_lt_p p, int64_t n) { GRID_STRIDE(i, n) rrc_kc_item_lt_matmul(&p, i); }
__global__ void k_conv2d(rrc_kc_conv2d_p p, int64_t n) { GRID_STRIDE(i, n) rrc_kc_item_conv2d(&p, i); }
__global__ void k_embedding(rrc_kc_embed_p p, int64_t n, rrc_cuda_err *err) {
    GRID_STRIDE(i, n) {
        int64_t bad;
        if (rrc_kc_item_embedding(&p, i, &bad)) dev_range(err, RRC_CU_OP_EMBEDDING, bad);
    }
}
__global__ void k_add_bias(rrc_kc_bias_p p, int64_t n) { GRID_STRIDE(i, n) rrc_kc_item_add_bias(&p, i); }
__global__ void k_bias_grad(rrc_kc_bias_p p, int64_t n) { GRID_STRIDE(i, n) rrc_kc_item_bias_grad(&p, i); }
__global__ void k_rope(rrc_kc_rope_p p, int64_t n, rrc_cuda_err *err) {
    GRID_STRIDE(i, n) {
        int64_t bad;
        if (rrc_kc_item_rope(&p, i, &bad)) dev_range(err, p.backward ? RRC_CU_OP_ROPE_BWD : RRC_CU_OP_ROPE_FWD, bad);
    }
}
__global__ void k_swiglu_fwd(rrc_kc_swiglu_p p, int64_t n) { GRID_STRIDE(i, n) rrc_kc_item_swiglu_fwd(&p, i); }
__global__ void k_swiglu_bwd(rrc_kc_swiglu_p p, int64_t n) { GRID_STRIDE(i, n) rrc_kc_item_swiglu_bwd(&p, i); }
__global__ void k_add(rrc_kc_ew_p p, int64_t n) { GRID_STRIDE(i, n) rrc_kc_item_add(&p, i); }
__global__ void k_axpy(rrc_kc_ew_p p, int64_t n) { GRID_STRIDE(i, n) rrc_kc_item_axpy(&p, i); }
__global__ void k_scale(rrc_kc_ew_p p, int64_t n) { GRID_STRIDE(i, n) rrc_kc_item_scale(&p, i); }
__global__ void k_convert(rrc_kc_ew_p p, int64_t n) { GRID_STRIDE(i, n) rrc_kc_item_convert(&p, i); }
__global__ void k_adamw(rrc_kc_adamw_p p, int64_t n) { GRID_STRIDE(i, n) rrc_kc_item_adamw(&p, i); }

// ---- row kernels (blockDim.x == LANES) --------------------------------------------------------------

__global__ void __launch_bounds__(LANES) k_rmsnorm_fwd(rrc_kc_rmsnorm_p p) {
    __shared__ float sh[LANES];
    __shared__ float rstd;
    const int lane = threadIdx.x;
    ROW_STRIDE(row, p.rows) {
        sh[lane] = rrc_kc_lane_rmsnorm_sumsq(&p, row, lane);
        tree_sum(sh, lane);
        if (lane == 0) rstd = rrc_kc_rmsnorm_finalize_rstd(&p, sh[0]);
        __syncthreads();
        for (int64_t i = lane; i < p.dim; i += LANES) rrc_kc_item_rmsnorm_fwd(&p, row, i, rstd);
        __syncthreads();
    }
}

__global__ void __launch_bounds__(LANES) k_rmsnorm_bwd(rrc_kc_rmsnorm_p p) {
    __shared__ float sq[LANES], dt[LANES];
    __shared__ float rstd, coef;
    const int lane = threadIdx.x;
    ROW_STRIDE(row, p.rows) {
        sq[lane] = rrc_kc_lane_rmsnorm_sumsq(&p, row, lane);
        dt[lane] = rrc_kc_lane_rmsnorm_dot(&p, row, lane);
        tree_sum2(sq, dt, lane);
        if (lane == 0) {
            rstd = rrc_kc_rmsnorm_finalize_rstd(&p, sq[0]);
            coef = rrc_kc_rmsnorm_finalize_coef(&p, dt[0], rstd);
        }
        __syncthreads();
        for (int64_t i = lane; i < p.dim; i += LANES) rrc_kc_item_rmsnorm_bwd(&p, row, i, rstd, coef);
        __syncthreads();
    }
}

__global__ void __launch_bounds__(LANES) k_softmax_fwd(rrc_kc_softmax_p p) {
    __shared__ float sh[LANES];
    __shared__ float m, inv_z;
    const int lane = threadIdx.x;
    ROW_STRIDE(row, p.batch * p.rows) {
        sh[lane] = rrc_kc_lane_softmax_max(&p, row, lane);
        tree_max(sh, lane);
        if (lane == 0) m = sh[0];
        __syncthreads();
        sh[lane] = rrc_kc_lane_softmax_sum(&p, row, lane, m);
        tree_sum(sh, lane);
        if (lane == 0) inv_z = m == -INFINITY ? 0.0f : 1.0f / sh[0];
        __syncthreads();
        for (int64_t j = lane; j < p.cols; j += LANES) rrc_kc_item_softmax_fwd(&p, row, j, m, inv_z);
        __syncthreads();
    }
}

__global__ void __launch_bounds__(LANES) k_softmax_bwd(rrc_kc_softmax_p p) {
    __shared__ float sh[LANES];
    __shared__ float dot;
    const int lane = threadIdx.x;
    ROW_STRIDE(row, p.batch * p.rows) {
        sh[lane] = rrc_kc_lane_softmax_dot(&p, row, lane);
        tree_sum(sh, lane);
        if (lane == 0) dot = sh[0];
        __syncthreads();
        for (int64_t j = lane; j < p.cols; j += LANES) rrc_kc_item_softmax_bwd(&p, row, j, dot);
        __syncthreads();
    }
}

__global__ void __launch_bounds__(LANES) k_xent(rrc_kc_xent_p p, rrc_cuda_err *err) {
    __shared__ rrc_kc_lse sh[LANES];
    __shared__ rrc_kc_xent_stats st;
    const int lane = threadIdx.x;
    ROW_STRIDE(row, p.rows) {
        sh[lane] = rrc_kc_lane_xent(&p, row, lane);
        tree_lse(sh, lane);
        if (lane == 0 && rrc_kc_xent_finalize(&p, row, sh[0], &st)) dev_range(err, RRC_CU_OP_XENT, row);
        __syncthreads();
        if (p.write_grad)
            for (int64_t j = lane; j < p.vocab; j += LANES) rrc_kc_item_xent_grad(&p, row, j, &st);
        __syncthreads();
    }
}

__global__ void __launch_bounds__(LANES) k_ss_chunks(rrc_kc_ss_p p, int64_t n_chunks) {
    __shared__ float sh[LANES];
    const int lane = threadIdx.x;
    ROW_STRIDE(c, n_chunks) {
        sh[lane] = rrc_kc_lane_ss_chunk(&p, c, lane);
        tree_sum(sh, lane);
        if (lane == 0) p.part[c] = sh[0];
        __syncthreads();
    }
}

__global__ void __launch_bounds__(LANES) k_ss_final(rrc_kc_ss_p p) {
    __shared__ float sh[LANES];
    const int lane = threadIdx.x;
    sh[lane] = rrc_kc_lane_ss_parts(&p, lane);
    tree_sum(sh, lane);
    if (lane == 0) rrc_kc_ss_finalize(&p, sh[0]);
}

// ---- launchers ---------------------------------------------------------------------------------------

#define LAUNCH_ELEMS(kernel, n, ...)                                                                    \
    do {                                                                                                \
        if ((n) <= 0) return cudaSuccess;                                                               \
        RRC_CU_LAUNCH_NOSYNC(kernel, grid_elems(n, max_blocks), 256, s, __VA_ARGS__);                   \
        return cudaGetLastError();                                                                      \
    } while (0)
#define LAUNCH_ROWS(kernel, rows, ...)                                                                  \
    do {                                                                                                \
        if ((rows) <= 0) return cudaSuccess;                                                            \
        RRC_CU_LAUNCH(kernel, grid_rows(rows, max_blocks), LANES, s, __VA_ARGS__);                      \
        return cudaGetLastError();                                                                      \
    } while (0)

cudaError_t rrc_cu_lt_matmul(cudaStream_t s, unsigned max_blocks, const rrc_kc_lt_p *p) {
    const int64_t n = p->batch * p->m * p->n;
    LAUNCH_ELEMS(k_lt_matmul, n, *p, n);
}
cudaError_t rrc_cu_conv2d(cudaStream_t s, unsigned max_blocks, const rrc_kc_conv2d_p *p) {
    const int64_t n = p->batch * p->rows * p->cols;
    LAUNCH_ELEMS(k_conv2d, n, *p, n);
}
cudaError_t rrc_cu_embedding(cudaStream_t s, unsigned max_blocks, const rrc_kc_embed_p *p, rrc_cuda_err *err) {
    const int64_t n = p->n_tokens * p->dim;
    LAUNCH_ELEMS(k_embedding, n, *p, n, err);
}
cudaError_t rrc_cu_add_bias(cudaStream_t s, unsigned max_blocks, const rrc_kc_bias_p *p) {
    const int64_t n = p->rows * p->cols;
    LAUNCH_ELEMS(k_add_bias, n, *p, n);
}
cudaError_t rrc_cu_bias_grad(cudaStream_t s, unsigned max_blocks, const rrc_kc_bias_p *p) { LAUNCH_ELEMS(k_bias_grad, p->cols, *p, p->cols); }
cudaError_t rrc_cu_rmsnorm_fwd(cudaStream_t s, unsigned max_blocks, const rrc_kc_rmsnorm_p *p) { LAUNCH_ROWS(k_rmsnorm_fwd, p->rows, *p); }
cudaError_t rrc_cu_rmsnorm_bwd(cudaStream_t s, unsigned max_blocks, const rrc_kc_rmsnorm_p *p) { LAUNCH_ROWS(k_rmsnorm_bwd, p->rows, *p); }
cudaError_t rrc_cu_rope(cudaStream_t s, unsigned max_blocks, const rrc_kc_rope_p *p, rrc_cuda_err *err) {
    const int64_t n = p->n_tokens * p->n_heads * (p->head_dim / 2);
    LAUNCH_ELEMS(k_rope, n, *p, n, err);
}
cudaError_t rrc_cu_softmax_fwd(cudaStream_t s, unsigned max_blocks, const rrc_kc_softmax_p *p) {
    LAUNCH_ROWS(k_softmax_fwd, p->batch * p->rows, *p);
}
cudaError_t rrc_cu_softmax_bwd(cudaStream_t s, unsigned max_blocks, const rrc_kc_softmax_p *p) {
    LAUNCH_ROWS(k_softmax_bwd, p->batch * p->rows, *p);
}
cudaError_t rrc_cu_swiglu_fwd(cudaStream_t s, unsigned max_blocks, const rrc_kc_swiglu_p *p) {
    const int64_t n = p->rows * p->cols;
    LAUNCH_ELEMS(k_swiglu_fwd, n, *p, n);
}
cudaError_t rrc_cu_swiglu_bwd(cudaStream_t s, unsigned max_blocks, const rrc_kc_swiglu_p *p) {
    const int64_t n = p->rows * p->cols;
    LAUNCH_ELEMS(k_swiglu_bwd, n, *p, n);
}
cudaError_t rrc_cu_add(cudaStream_t s, unsigned max_blocks, const rrc_kc_ew_p *p) { LAUNCH_ELEMS(k_add, p->n, *p, p->n); }
cudaError_t rrc_cu_axpy(cudaStream_t s, unsigned max_blocks, const rrc_kc_ew_p *p) { LAUNCH_ELEMS(k_axpy, p->n, *p, p->n); }
cudaError_t rrc_cu_scale(cudaStream_t s, unsigned max_blocks, const rrc_kc_ew_p *p) { LAUNCH_ELEMS(k_scale, p->n, *p, p->n); }
cudaError_t rrc_cu_convert(cudaStream_t s, unsigned max_blocks, const rrc_kc_ew_p *p) { LAUNCH_ELEMS(k_convert, p->n, *p, p->n); }
cudaError_t rrc_cu_sum_squares(cudaStream_t s, unsigned max_blocks, const rrc_kc_ss_p *p) {
    const int64_t nc = rrc_kc_ss_chunks(p->n);
    if (nc > 0) {
        RRC_CU_LAUNCH(k_ss_chunks, grid_rows(nc, max_blocks), LANES, s, *p, nc);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }
    RRC_CU_LAUNCH(k_ss_final, 1, LANES, s, *p);
    return cudaGetLastError();
}
cudaError_t rrc_cu_xent(cudaStream_t s, unsigned max_blocks, const rrc_kc_xent_p *p, rrc_cuda_err *err) { LAUNCH_ROWS(k_xent, p->rows, *p, err); }
cudaError_t rrc_cu_adamw(cudaStream_t s, unsigned max_blocks, const rrc_kc_adamw_p *p) { LAUNCH_ELEMS(k_adamw, p->n, *p, p->n); }

// The kernel image the runtime loaded for this device (SASS vs. JIT-compiled PTX), for rrc_selftest's
// report. Any rrc kernel would do; all are in this translation unit.
cudaError_t rrc_cu_kernel_versions(int *binary_version, int *ptx_version) {
    cudaFuncAttributes a;
    memset(&a, 0, sizeof a);
    cudaError_t e = cudaFuncGetAttributes(&a, reinterpret_cast<const void *>(&k_add));
    *binary_version = a.binaryVersion;
    *ptx_version = a.ptxVersion;
    return e;
}
