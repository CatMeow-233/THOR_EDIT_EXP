/* RMSNorm (HF LlamaRMSNorm / Qwen2RMSNorm / Qwen3RMSNorm), fp32 math:
 *   var = mean(x^2); rstd = 1 / sqrt(var + eps); y = w * (x * rstd)
 * Backward (input gradient, w frozen), with D = dim:
 *   dx_i = rstd * dy_i * w_i - x_i * rstd^3 * (sum_j dy_j * w_j * x_j) / D
 */
#ifndef RRC_KC_RMSNORM_H
#define RRC_KC_RMSNORM_H

#include "rrc_kc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* rstd from the sum of squares of a row. */
RRC_HD static inline float rrc_kc_rmsnorm_rstd(double sumsq, int64_t dim, float eps) {
    float var = (float)(sumsq / (double)dim);
    return 1.0f / sqrtf(var + eps);
}
RRC_HD static inline float rrc_kc_rmsnorm_rstd_f(float sumsq, int64_t dim, float eps) {
    float var = sumsq / (float)dim;
    return 1.0f / sqrtf(var + eps);
}

RRC_HD static inline float rrc_kc_rmsnorm_elem_fwd(float x, float rstd, float w) { return w * (x * rstd); }

/* k = rstd^3 * dot / dim where dot = sum_j dy_j * w_j * x_j */
RRC_HD static inline float rrc_kc_rmsnorm_bwd_coef(double dot, int64_t dim, float rstd) {
    return rstd * rstd * rstd * (float)(dot / (double)dim);
}
RRC_HD static inline float rrc_kc_rmsnorm_bwd_coef_f(float dot, int64_t dim, float rstd) {
    return rstd * rstd * rstd * (dot / (float)dim);
}

RRC_HD static inline float rrc_kc_rmsnorm_elem_bwd(float x, float w, float dy, float rstd, float k) {
    return rstd * (dy * w) - k * x;
}

/* Serial row helpers. Row pointers already include the row offset. */
RRC_HD static inline float rrc_kc_rmsnorm_row_rstd(const void *x, rrc_dtype xdt, int64_t dim, float eps, int acc64) {
    if (acc64) {
        double s = 0.0;
        for (int64_t i = 0; i < dim; i++) {
            double v = (double)rrc_kc_ld(x, xdt, i);
            s += v * v;
        }
        return rrc_kc_rmsnorm_rstd(s, dim, eps);
    }
    float s = 0.0f;
    for (int64_t i = 0; i < dim; i++) {
        float v = rrc_kc_ld(x, xdt, i);
        s += v * v;
    }
    return rrc_kc_rmsnorm_rstd_f(s, dim, eps);
}

RRC_HD static inline void rrc_kc_rmsnorm_row_fwd(const void *x, rrc_dtype xdt, const void *w, rrc_dtype wdt, void *y,
                                                 rrc_dtype ydt, int64_t dim, float eps, int acc64) {
    float rstd = rrc_kc_rmsnorm_row_rstd(x, xdt, dim, eps, acc64);
    for (int64_t i = 0; i < dim; i++)
        rrc_kc_st(y, ydt, i, rrc_kc_rmsnorm_elem_fwd(rrc_kc_ld(x, xdt, i), rstd, rrc_kc_ld(w, wdt, i)));
}

/* dx (accumulate ? += : =) rmsnorm input gradient. All reads of the row happen before the first
 * write, so dx may exactly alias x or dy when accumulate == 0. */
RRC_HD static inline void rrc_kc_rmsnorm_row_bwd(const void *x, rrc_dtype xdt, const void *w, rrc_dtype wdt,
                                                 const void *dy, rrc_dtype dydt, void *dx, rrc_dtype dxdt, int64_t dim,
                                                 float eps, int accumulate, int acc64) {
    float rstd = rrc_kc_rmsnorm_row_rstd(x, xdt, dim, eps, acc64);
    float k;
    if (acc64) {
        double dot = 0.0;
        for (int64_t i = 0; i < dim; i++)
            dot += (double)rrc_kc_ld(dy, dydt, i) * (double)rrc_kc_ld(w, wdt, i) * (double)rrc_kc_ld(x, xdt, i);
        k = rrc_kc_rmsnorm_bwd_coef(dot, dim, rstd);
    } else {
        float dot = 0.0f;
        for (int64_t i = 0; i < dim; i++) dot += rrc_kc_ld(dy, dydt, i) * rrc_kc_ld(w, wdt, i) * rrc_kc_ld(x, xdt, i);
        k = rrc_kc_rmsnorm_bwd_coef_f(dot, dim, rstd);
    }
    for (int64_t i = 0; i < dim; i++) {
        float g = rrc_kc_rmsnorm_elem_bwd(rrc_kc_ld(x, xdt, i), rrc_kc_ld(w, wdt, i), rrc_kc_ld(dy, dydt, i), rstd, k);
        if (accumulate) g += rrc_kc_ld(dx, dxdt, i);
        rrc_kc_st(dx, dxdt, i, g);
    }
}

#ifdef __cplusplus
}
#endif

#endif
