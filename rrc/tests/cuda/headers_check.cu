// rrc/tests/cuda/headers_check.cu — COMPILED-ONLY check (never executed): every public header
// compiles as CUDA C++, and every RRC_HD function of common.h and kernels_common/ is instantiated
// in device code (unused static inline functions would not be code-generated otherwise).
#include "rrc/rrc.h"
#include "rrc_kc.h"

__global__ void rrc_check_common(const rrc_bf16 *in, float *out, rrc_bf16 *o16, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float f = rrc_bf16_to_f32(in[i]);
    o16[i] = rrc_f32_to_bf16(f * 3.0f);
    out[i] = rrc_round_bf16(f) + rrc_bits_f32(rrc_f32_bits(f)) + (float)rrc_dtype_size(RRC_BF16) +
             (float)rrc_dtype_valid(i) + (float)rrc_dtype_is_float(RRC_F32);
}

__global__ void rrc_check_io(const void *x, void *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = rrc_kc_ld(x, RRC_BF16, i) + rrc_kc_ld(x, RRC_F32, i);
    rrc_kc_st(y, RRC_BF16, i, rrc_kc_round_to(RRC_BF16, v));
    rrc_kc_st(y, RRC_F32, i, v);
}

__global__ void rrc_check_rmsnorm(const float *x, const float *w, float *y, int rows, int dim, int acc64) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    rrc_kc_rmsnorm_row_fwd(x + r * dim, RRC_F32, w, RRC_F32, y + r * dim, RRC_F32, dim, 1e-6f, acc64);
    rrc_kc_rmsnorm_row_bwd(x + r * dim, RRC_F32, w, RRC_F32, x + r * dim, RRC_F32, y + r * dim, RRC_F32, dim, 1e-6f, 1,
                           acc64);
    float rs = rrc_kc_rmsnorm_rstd(1.0, dim, 1e-6f) + rrc_kc_rmsnorm_rstd_f(1.0f, dim, 1e-6f) +
               rrc_kc_rmsnorm_row_rstd(x, RRC_F32, dim, 1e-6f, acc64);
    float k = rrc_kc_rmsnorm_bwd_coef(1.0, dim, rs) + rrc_kc_rmsnorm_bwd_coef_f(1.0f, dim, rs);
    y[r] += rrc_kc_rmsnorm_elem_fwd(x[r], rs, w[0]) + rrc_kc_rmsnorm_elem_bwd(x[r], w[0], y[r], rs, k);
}

__global__ void rrc_check_rope(float *x, const float *cs, const float *sn, int heads, int hd) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= heads) return;
    rrc_kc_rope_head(x + h * hd, RRC_F32, x + h * hd, RRC_F32, hd, cs, sn, 0);
    rrc_kc_rope_head(x + h * hd, RRC_F32, x + h * hd, RRC_F32, hd, cs, sn, 1);
    float a, b;
    rrc_kc_rope_pair_fwd(x[0], x[1], cs[0], sn[0], &a, &b);
    rrc_kc_rope_pair_bwd(a, b, cs[0], sn[0], &a, &b);
    x[h] = a + b;
}

__global__ void rrc_check_swiglu(const float *g, const float *u, float *o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float dg, du;
    rrc_kc_swiglu_bwd(g[i], u[i], o[i], &dg, &du);
    o[i] = rrc_kc_swiglu_fwd(g[i], u[i]) + dg + du + rrc_kc_sigmoid(g[i]) + rrc_kc_silu(g[i]) + rrc_kc_silu_grad(g[i]);
}

__global__ void rrc_check_softmax(float *s, const int32_t *doc, int rows, int cols, int acc64) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    rrc_kc_attn_softmax_row_fwd(s + r * cols, RRC_F32, s + r * cols, RRC_BF16, cols, 0.125f, r, 1, doc, acc64);
    rrc_kc_attn_softmax_row_bwd(s + r * cols, RRC_F32, s + r * cols, RRC_F32, s + r * cols, RRC_F32, cols, 0.125f,
                                acc64);
    rrc_kc_lse a = rrc_kc_lse_push(rrc_kc_lse_init(), s[r]);
    a = rrc_kc_lse_merge(a, rrc_kc_lse_push(rrc_kc_lse_init(), s[0]));
    s[r] = a.m + a.s + (float)rrc_kc_attn_allowed(0, r, 1, doc) + rrc_kc_softmax_elem(s[r], a.m, 1.0f / a.s) +
           rrc_kc_softmax_bwd_elem(s[r], s[0], 1.0f, 0.5f);
}

__global__ void rrc_check_xent(float *z, const int32_t *labels, float *loss, int rows, int vocab, int acc64) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    loss[r] = rrc_kc_xent_row(z + r * vocab, RRC_F32, vocab, labels[r], 0.5f, 1, acc64) +
              rrc_kc_xent_grad_elem(z[0], 1.0f, 0.5f, 1, 1.0f) +
              rrc_kc_xent_loss(z[0], 0.5f, z[1]);
}

__global__ void rrc_check_adamw(float *p, const float *g, float *m, float *v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    rrc_kc_adamw_scalars s = rrc_kc_adamw_prepare(1e-3, 0.9, 0.999, 1e-8, 0.01, 3, 1.0f);
    rrc_kc_adamw_elem(&p[i], g[i], &m[i], &v[i], &s);
    p[i] += rrc_kc_lerp(m[i], v[i], 0.1f);
}
