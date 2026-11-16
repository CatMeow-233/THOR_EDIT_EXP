/* Rotary position embedding, HF rotate_half convention: for a head vector x of size hd, h = hd/2,
 *   y[i]     = x[i] * c[i] - x[i+h] * s[i]
 *   y[i+h]   = x[i+h] * c[i] + x[i] * s[i]          (i < h)
 * which is `x * cos + rotate_half(x) * sin` with cos/sin = cat(freqs, freqs). The backward is the
 * transposed rotation (rotation by -theta). */
#ifndef RRC_KC_ROPE_H
#define RRC_KC_ROPE_H

#include "rrc_kc.h"

#ifdef __cplusplus
extern "C" {
#endif

RRC_HD static inline void rrc_kc_rope_pair_fwd(float x1, float x2, float c, float s, float *y1, float *y2) {
    *y1 = x1 * c - x2 * s;
    *y2 = x2 * c + x1 * s;
}

RRC_HD static inline void rrc_kc_rope_pair_bwd(float dy1, float dy2, float c, float s, float *dx1, float *dx2) {
    *dx1 = dy1 * c + dy2 * s;
    *dx2 = dy2 * c - dy1 * s;
}

/* One head: x/y point at the head's first element; cos_row/sin_row at the position's table row.
 * y may alias x (each pair is read before it is written). */
RRC_HD static inline void rrc_kc_rope_head(const void *x, rrc_dtype xdt, void *y, rrc_dtype ydt, int64_t head_dim,
                                           const float *cos_row, const float *sin_row, int backward) {
    const int64_t h = head_dim / 2;
    for (int64_t i = 0; i < h; i++) {
        float a = rrc_kc_ld(x, xdt, i), b = rrc_kc_ld(x, xdt, i + h);
        float o1, o2;
        if (backward) rrc_kc_rope_pair_bwd(a, b, cos_row[i], sin_row[i], &o1, &o2);
        else rrc_kc_rope_pair_fwd(a, b, cos_row[i], sin_row[i], &o1, &o2);
        rrc_kc_st(y, ydt, i, o1);
        rrc_kc_st(y, ydt, i + h, o2);
    }
}

#ifdef __cplusplus
}
#endif

#endif
