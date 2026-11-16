/* Cross entropy for one row of logits z (vocab V) and a label y (or RRC ignore index), in the
 * log_softmax form (no cancellation for large logits):
 *   m = max(z), s = sum_j exp(z_j - m) (online, rrc_kc_lse), ls = log(s)
 *   loss = (m - z[y]) + ls                                  (= logsumexp(z) - z[y])
 *   dz_j = (exp((z_j - m) - ls) - [j == y]) * scale         (scale = grad_scale * row_scale)
 * Ignored rows: loss 0 and dz = 0. */
#ifndef RRC_KC_XENT_H
#define RRC_KC_XENT_H

#include "rrc_kc.h"
#include "rrc_kc_softmax.h"

#ifdef __cplusplus
extern "C" {
#endif

RRC_HD static inline float rrc_kc_xent_grad_elem(float z, float m, float log_s, int is_label, float scale) {
    return (expf((z - m) - log_s) - (is_label ? 1.0f : 0.0f)) * scale;
}

RRC_HD static inline float rrc_kc_xent_loss(float m, float log_s, float z_label) { return (m - z_label) + log_s; }

/* Serial row: returns the loss; writes the gradient in place when write_grad. In acc64 mode the
 * sum of exponentials is accumulated in double (online rescaling as well) and the loss is
 * computed in double; the gradient exponent uses (float)m and (float)log(s).
 * `label` must be in [0, V), or negative for an ignored row. */
RRC_HD static inline float rrc_kc_xent_row(void *z, rrc_dtype zdt, int64_t vocab, int32_t label, float scale,
                                           int write_grad, int acc64) {
    if (label < 0) {
        if (write_grad)
            for (int64_t j = 0; j < vocab; j++) rrc_kc_st(z, zdt, j, 0.0f);
        return 0.0f;
    }
    float m_f, log_s;
    float loss;
    if (acc64) {
        double m = -INFINITY, s = 0.0;
        for (int64_t j = 0; j < vocab; j++) {
            double x = (double)rrc_kc_ld(z, zdt, j);
            if (x > m) {
                s = s * exp(m - x) + 1.0;
                m = x;
            } else {
                s += exp(x - m);
            }
        }
        double ls = log(s);
        m_f = (float)m; /* m is one of the (float) logits: exact */
        log_s = (float)ls;
        loss = (float)((m - (double)rrc_kc_ld(z, zdt, label)) + ls);
    } else {
        rrc_kc_lse a = rrc_kc_lse_init();
        for (int64_t j = 0; j < vocab; j++) a = rrc_kc_lse_push(a, rrc_kc_ld(z, zdt, j));
        m_f = a.m;
        log_s = logf(a.s);
        loss = rrc_kc_xent_loss(m_f, log_s, rrc_kc_ld(z, zdt, label));
    }
    if (write_grad)
        for (int64_t j = 0; j < vocab; j++)
            rrc_kc_st(z, zdt, j, rrc_kc_xent_grad_elem(rrc_kc_ld(z, zdt, j), m_f, log_s, j == label, scale));
    return loss;
}

#ifdef __cplusplus
}
#endif

#endif
