/* AdamW element update with torch.optim.AdamW semantics (torch/optim/adam.py,
 * _single_tensor_adam with decoupled_weight_decay=True, non-capturable path):
 *   step_t += 1
 *   param.mul_(1 - lr * weight_decay)                 (only if weight_decay != 0)
 *   exp_avg.lerp_(grad, 1 - beta1)
 *   exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
 *   bias_correction1 = 1 - beta1 ** step; bias_correction2 = 1 - beta2 ** step   (python floats)
 *   step_size = lr / bias_correction1; bias_correction2_sqrt = bias_correction2 ** 0.5
 *   denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
 *   param.addcdiv_(exp_avg, denom, value=-step_size)
 * Python scalars enter the fp32 tensor ops rounded to float. The element-level rounding follows
 * ATen's vectorized CPU kernels, which were checked empirically against torch 2.15 on CPU:
 *   lerp    = fma(w, end - self, self) if |w| < 0.5 else fma(w - 1, end - self, end)
 *   addcmul = fma(value * t1, t2, self)
 *   addcdiv = self + (value * t1) / t2          (no fusion)
 * With these, the update is bit-identical to torch.optim.AdamW(foreach=False) on CPU except where
 * torch's vectorized sqrt is not correctly rounded (sqrtf here is): a 1-ulp difference in a small
 * fraction of elements (rrc/tests/c/test_ops.c: adamw_vs_torch). */
#ifndef RRC_KC_ADAMW_H
#define RRC_KC_ADAMW_H

#include "rrc_kc.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rrc_kc_adamw_scalars {
    float wd_factor;     /* (float)(1 - lr * wd) */
    int apply_wd;        /* wd != 0 */
    float lerp_w;        /* (float)(1 - beta1) */
    float beta2;         /* (float)beta2 */
    float one_m_beta2;   /* (float)(1 - beta2) */
    float bc2_sqrt;      /* (float)((1 - beta2^step) ** 0.5) */
    float eps;           /* (float)eps */
    float neg_step_size; /* (float)(-(lr / (1 - beta1^step))) */
    float grad_scale;
} rrc_kc_adamw_scalars;

/* Host-side: derive the f32 scalars (double math, like the Python side of torch). */
RRC_HD static inline rrc_kc_adamw_scalars rrc_kc_adamw_prepare(double lr, double beta1, double beta2, double eps,
                                                               double wd, int64_t step, float grad_scale) {
    rrc_kc_adamw_scalars s;
    s.apply_wd = wd != 0.0;
    s.wd_factor = (float)(1.0 - lr * wd);
    s.lerp_w = (float)(1.0 - beta1);
    s.beta2 = (float)beta2;
    s.one_m_beta2 = (float)(1.0 - beta2);
    double bc1 = 1.0 - pow(beta1, (double)step);
    double bc2 = 1.0 - pow(beta2, (double)step);
    s.bc2_sqrt = (float)pow(bc2, 0.5); /* python: bias_correction2 ** 0.5 */
    s.eps = (float)eps;
    s.neg_step_size = (float)(-(lr / bc1));
    s.grad_scale = grad_scale;
    return s;
}

/* torch lerp (ATen vectorized CPU kernel, fused multiply-add) */
RRC_HD static inline float rrc_kc_lerp(float self, float end, float w) {
    float d = end - self;
    return (fabsf(w) < 0.5f) ? fmaf(w, d, self) : fmaf(w - 1.0f, d, end);
}

RRC_HD static inline void rrc_kc_adamw_elem(float *p, float grad, float *m, float *v, const rrc_kc_adamw_scalars *s) {
    float g = grad * s->grad_scale;
    float pv = *p;
    if (s->apply_wd) pv = pv * s->wd_factor;
    float mv = rrc_kc_lerp(*m, g, s->lerp_w);
    float vv = *v * s->beta2;
    vv = fmaf(s->one_m_beta2 * g, g, vv); /* addcmul_(grad, grad, value=1 - beta2) */
    float denom = sqrtf(vv) / s->bc2_sqrt + s->eps;
    pv = pv + (s->neg_step_size * mv) / denom;
    *p = pv;
    *m = mv;
    *v = vv;
}

#ifdef __cplusplus
}
#endif

#endif
