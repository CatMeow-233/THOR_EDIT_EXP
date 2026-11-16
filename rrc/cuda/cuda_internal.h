/* rrc/cuda/cuda_internal.h — launchers of the CUDA kernels (cuda_kernels.cu), used by the backend
 * (cuda_backend.cu). CUDA C++ only. Every launcher enqueues on `s` and returns
 * cudaGetLastError() of the launch. COMPILED-ONLY. */
#ifndef RRC_CUDA_INTERNAL_H
#define RRC_CUDA_INTERNAL_H

#include <cuda_runtime.h>

#include "rrc_kc_items.h"

/* Device error flag: the first op that saw a range error and the smallest offending row/token
 * index seen for it. Reset: op = 0, index = ~0ull. */
struct rrc_cuda_err {
    unsigned int op;
    unsigned int pad;
    unsigned long long index;
};

enum {
    RRC_CU_OP_NONE = 0,
    RRC_CU_OP_EMBEDDING = 1,
    RRC_CU_OP_ROPE_FWD = 2,
    RRC_CU_OP_ROPE_BWD = 3,
    RRC_CU_OP_XENT = 4,
};

cudaError_t rrc_cu_lt_matmul(cudaStream_t s, const rrc_kc_lt_p *p);
cudaError_t rrc_cu_conv2d(cudaStream_t s, const rrc_kc_conv2d_p *p);
cudaError_t rrc_cu_embedding(cudaStream_t s, const rrc_kc_embed_p *p, rrc_cuda_err *err);
cudaError_t rrc_cu_add_bias(cudaStream_t s, const rrc_kc_bias_p *p);
cudaError_t rrc_cu_bias_grad(cudaStream_t s, const rrc_kc_bias_p *p);
cudaError_t rrc_cu_rmsnorm_fwd(cudaStream_t s, const rrc_kc_rmsnorm_p *p);
cudaError_t rrc_cu_rmsnorm_bwd(cudaStream_t s, const rrc_kc_rmsnorm_p *p);
cudaError_t rrc_cu_rope(cudaStream_t s, const rrc_kc_rope_p *p, rrc_cuda_err *err);
cudaError_t rrc_cu_softmax_fwd(cudaStream_t s, const rrc_kc_softmax_p *p);
cudaError_t rrc_cu_softmax_bwd(cudaStream_t s, const rrc_kc_softmax_p *p);
cudaError_t rrc_cu_swiglu_fwd(cudaStream_t s, const rrc_kc_swiglu_p *p);
cudaError_t rrc_cu_swiglu_bwd(cudaStream_t s, const rrc_kc_swiglu_p *p);
cudaError_t rrc_cu_add(cudaStream_t s, const rrc_kc_ew_p *p);
cudaError_t rrc_cu_axpy(cudaStream_t s, const rrc_kc_ew_p *p);
cudaError_t rrc_cu_scale(cudaStream_t s, const rrc_kc_ew_p *p);
cudaError_t rrc_cu_convert(cudaStream_t s, const rrc_kc_ew_p *p);
/* p->part must hold rrc_kc_ss_chunks(p->n) floats */
cudaError_t rrc_cu_sum_squares(cudaStream_t s, const rrc_kc_ss_p *p);
cudaError_t rrc_cu_xent(cudaStream_t s, const rrc_kc_xent_p *p, rrc_cuda_err *err);
cudaError_t rrc_cu_adamw(cudaStream_t s, const rrc_kc_adamw_p *p);

#endif /* RRC_CUDA_INTERNAL_H */
