/* rrc/cuda/kc_emu.h — host emulation of the CUDA backend (DESIGN.md §11.4).
 *
 * An rrc_backend ("kc-emu") on host memory whose ops execute exactly the per-item bodies of the
 * CUDA kernels (kernels_common/rrc_kc_items.h) with the same phase structure (lane partials, fixed
 * tree, finalize, elements), each phase over its items in a pseudo-random order (seeded), and
 * whose gemm runs the GEMM plan ladder of gemm_plan.c with a simulated library matmul
 * (rrc_kc_item_lt_matmul, which is also the device fallback kernel). It gives the kernel bodies,
 * the plan and the ladder execution coverage on CPU; the CUDA launch code itself stays
 * COMPILED-ONLY. Pure C11, no CUDA dependency.
 */
#ifndef RRC_KC_EMU_H
#define RRC_KC_EMU_H

#include <stdint.h>

#include "gemm_plan.h"
#include "rrc/backend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rrc_emu_options {
    uint64_t seed;                 /* item order seed */
    rrc_gemm_level gemm_min_level; /* first ladder level tried */
    int64_t scratch_cap_elems;     /* gemm scratch cap (f32 elements), <= 0: default */
    /* simulated library support: bit (4*adt + 2*bdt + cdt) with f32 = 0, bf16 = 1 set means "no
     * algorithm" for that column-major (A, B, C) type triple */
    unsigned lt_reject_mask;
    unsigned lt_min_align;         /* calls with an operand alignment below this are refused */
    int deferred_errors;           /* range errors reported by the next sync / memcpy_d2h */
} rrc_emu_options;

void rrc_emu_options_default(rrc_emu_options *o);
rrc_status rrc_emu_backend_create(const rrc_emu_options *o, const rrc_backend **out);
void rrc_emu_backend_destroy(const rrc_backend *be);
/* Path of the most recent gemm; returns 0 if `be` is not an emulation backend or ran no gemm. */
int rrc_emu_last_gemm(const rrc_backend *be, rrc_gemm_exec_report *rep);
int rrc_backend_is_emu(const rrc_backend *be);

#ifdef __cplusplus
}
#endif

#endif /* RRC_KC_EMU_H */
