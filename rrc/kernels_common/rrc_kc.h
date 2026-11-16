/* rrc/kernels_common/rrc_kc.h — per-element and per-row math shared by the CPU backend (gcc) and
 * the CUDA backend (nvcc, __host__ __device__). Everything is an RRC_HD static inline function.
 *
 * Two levels are provided for each op:
 *   - element / finalize functions (rrc_kc_*_elem*, *_coef, pair rotations, lse merge, ...) that a
 *     CUDA kernel calls per thread after its own (block) reductions;
 *   - serial row functions (rrc_kc_*_row*) built only from those element functions, used by the
 *     CPU backend per row. Their reductions accumulate in double when `acc64` is non-zero and in
 *     float otherwise, always in index order.
 * Loads widen bf16 exactly; stores round to the destination dtype once (RNE).
 */
#ifndef RRC_KC_H
#define RRC_KC_H

#include <math.h>
#include <stdint.h>

#include "rrc/common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------------------------ */
/* Typed load / store                                                                          */
/* ------------------------------------------------------------------------------------------ */

RRC_HD static inline float rrc_kc_ld(const void *p, rrc_dtype dt, int64_t i) {
    return dt == RRC_BF16 ? rrc_bf16_to_f32(((const rrc_bf16 *)p)[i]) : ((const float *)p)[i];
}

RRC_HD static inline void rrc_kc_st(void *p, rrc_dtype dt, int64_t i, float v) {
    if (dt == RRC_BF16) ((rrc_bf16 *)p)[i] = rrc_f32_to_bf16(v);
    else ((float *)p)[i] = v;
}

/* Value after a store/load round trip in dtype `dt` (identity for f32). */
RRC_HD static inline float rrc_kc_round_to(rrc_dtype dt, float v) { return dt == RRC_BF16 ? rrc_round_bf16(v) : v; }

#ifdef __cplusplus
}
#endif

#include "rrc_kc_adamw.h"
#include "rrc_kc_rmsnorm.h"
#include "rrc_kc_rope.h"
#include "rrc_kc_softmax.h"
#include "rrc_kc_swiglu.h"
#include "rrc_kc_xent.h"

#endif /* RRC_KC_H */
