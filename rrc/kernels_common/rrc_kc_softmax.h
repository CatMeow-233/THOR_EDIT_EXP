/* Masked attention softmax rows (fp32).
 * Forward for query position q over keys j < cols:
 *   allowed(j) = (!causal || j <= q) && (no doc ids || doc[j] == doc[q])
 *   m = max_{allowed} scale*s_j; Z = sum_{allowed} exp(scale*s_j - m); p_j = exp(scale*s_j - m) / Z
 *   p_j = 0 if not allowed; all-zero row if nothing is allowed.
 * Backward: ds_j = scale * p_j * (dp_j - sum_k p_k dp_k). */
#ifndef RRC_KC_SOFTMAX_H
#define RRC_KC_SOFTMAX_H

#include "rrc_kc.h"

#ifdef __cplusplus
extern "C" {
#endif

RRC_HD static inline int rrc_kc_attn_allowed(int64_t j, int64_t q, int causal, const int32_t *doc) {
    if (causal && j > q) return 0;
    if (doc && doc[j] != doc[q]) return 0;
    return 1;
}

/* Online (max, sum-of-exp) state, mergeable in any tree order by a device reduction. */
typedef struct rrc_kc_lse {
    float m; /* running max (-INFINITY when empty) */
    float s; /* sum of exp(x - m) */
} rrc_kc_lse;

RRC_HD static inline rrc_kc_lse rrc_kc_lse_init(void) {
    rrc_kc_lse a;
    a.m = -INFINITY;
    a.s = 0.0f;
    return a;
}
RRC_HD static inline rrc_kc_lse rrc_kc_lse_push(rrc_kc_lse a, float x) {
    if (x > a.m) {
        a.s = a.s * expf(a.m - x) + 1.0f; /* exp(-inf) = 0 for the first element */
        a.m = x;
    } else {
        a.s += expf(x - a.m);
    }
    return a;
}
RRC_HD static inline rrc_kc_lse rrc_kc_lse_merge(rrc_kc_lse a, rrc_kc_lse b) {
    if (b.m == -INFINITY) return a;
    if (a.m == -INFINITY) return b;
    rrc_kc_lse r;
    if (a.m >= b.m) {
        r.m = a.m;
        r.s = a.s + b.s * expf(b.m - a.m);
    } else {
        r.m = b.m;
        r.s = b.s + a.s * expf(a.m - b.m);
    }
    return r;
}

RRC_HD static inline float rrc_kc_softmax_elem(float x_scaled, float m, float inv_z) { return expf(x_scaled - m) * inv_z; }

RRC_HD static inline float rrc_kc_softmax_bwd_elem(float p, float dp, float dot, float scale) {
    return scale * p * (dp - dot);
}

/* Serial row forward (two passes: max, then sum; then write). s and p may alias. */
RRC_HD static inline void rrc_kc_attn_softmax_row_fwd(const void *s, rrc_dtype sdt, void *p, rrc_dtype pdt,
                                                      int64_t cols, float scale, int64_t q, int causal,
                                                      const int32_t *doc, int acc64) {
    float m = -INFINITY;
    for (int64_t j = 0; j < cols; j++)
        if (rrc_kc_attn_allowed(j, q, causal, doc)) {
            float x = scale * rrc_kc_ld(s, sdt, j);
            if (x > m) m = x;
        }
    if (m == -INFINITY) { /* no allowed key (or all scores -inf): zero row */
        for (int64_t j = 0; j < cols; j++) rrc_kc_st(p, pdt, j, 0.0f);
        return;
    }
    float inv_z;
    if (acc64) {
        double z = 0.0;
        for (int64_t j = 0; j < cols; j++)
            if (rrc_kc_attn_allowed(j, q, causal, doc)) z += (double)expf(scale * rrc_kc_ld(s, sdt, j) - m);
        inv_z = (float)(1.0 / z);
    } else {
        float z = 0.0f;
        for (int64_t j = 0; j < cols; j++)
            if (rrc_kc_attn_allowed(j, q, causal, doc)) z += expf(scale * rrc_kc_ld(s, sdt, j) - m);
        inv_z = 1.0f / z;
    }
    for (int64_t j = 0; j < cols; j++) {
        float v = rrc_kc_attn_allowed(j, q, causal, doc) ? rrc_kc_softmax_elem(scale * rrc_kc_ld(s, sdt, j), m, inv_z)
                                                         : 0.0f;
        rrc_kc_st(p, pdt, j, v);
    }
}

/* Serial row backward. ds may alias p or dp (the dot product is complete before any write). */
RRC_HD static inline void rrc_kc_attn_softmax_row_bwd(const void *p, rrc_dtype pdt, const void *dp, rrc_dtype dpdt,
                                                      void *ds, rrc_dtype dsdt, int64_t cols, float scale, int acc64) {
    float dot;
    if (acc64) {
        double d = 0.0;
        for (int64_t j = 0; j < cols; j++) d += (double)rrc_kc_ld(p, pdt, j) * (double)rrc_kc_ld(dp, dpdt, j);
        dot = (float)d;
    } else {
        float d = 0.0f;
        for (int64_t j = 0; j < cols; j++) d += rrc_kc_ld(p, pdt, j) * rrc_kc_ld(dp, dpdt, j);
        dot = d;
    }
    for (int64_t j = 0; j < cols; j++)
        rrc_kc_st(ds, dsdt, j, rrc_kc_softmax_bwd_elem(rrc_kc_ld(p, pdt, j), rrc_kc_ld(dp, dpdt, j), dot, scale));
}

#ifdef __cplusplus
}
#endif

#endif
