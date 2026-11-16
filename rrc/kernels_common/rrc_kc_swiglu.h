/* SiLU and the SwiGLU gate: h = silu(g) * u with silu(g) = g / (1 + exp(-g)) (the formula of
 * torch's silu kernels). Backward with s = sigmoid(g):
 *   du = dh * silu(g);  dg = dh * u * s * (1 + g * (1 - s)) */
#ifndef RRC_KC_SWIGLU_H
#define RRC_KC_SWIGLU_H

#include "rrc_kc.h"

#ifdef __cplusplus
extern "C" {
#endif

RRC_HD static inline float rrc_kc_sigmoid(float g) { return 1.0f / (1.0f + expf(-g)); }
RRC_HD static inline float rrc_kc_silu(float g) { return g / (1.0f + expf(-g)); }
RRC_HD static inline float rrc_kc_silu_grad(float g) {
    float s = rrc_kc_sigmoid(g);
    return s * (1.0f + g * (1.0f - s));
}

RRC_HD static inline float rrc_kc_swiglu_fwd(float g, float u) { return rrc_kc_silu(g) * u; }

RRC_HD static inline void rrc_kc_swiglu_bwd(float g, float u, float dh, float *dg, float *du) {
    *du = dh * rrc_kc_silu(g);
    *dg = dh * u * rrc_kc_silu_grad(g);
}

#ifdef __cplusplus
}
#endif

#endif
