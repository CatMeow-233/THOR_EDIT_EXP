/* rrc/cuda/selftest_ops.h — backend-agnostic op self-test (DESIGN.md §11.5).
 *
 * Runs every op of the frozen table on random inputs (several shapes, odd sizes, padded leading
 * dimensions, the exact aliases backend.h allows) on a device backend `dev` and on a reference
 * backend `ref` (rrc_backend_cpu_f32acc), and compares with the documented tolerance of each op:
 *   EXACT   bit-identical (same arithmetic: IEEE fp32 add/mul/div/sqrt, explicit fma, RNE stores)
 *   ULP     ops using expf (swiglu): f32 outputs within 2e-6 relative to the output scale
 *   REDUCE  fp32 reductions in a different association (rmsnorm, softmax, CE, sum_squares):
 *           (1e-5 + n_red * 2^-24) relative to the output scale, n_red = terms per reduction
 *   GEMM    2 * k * 2^-24 * (|alpha| sum |a||b| + |beta||c|) per element (+ 2^-9 of that with TF32)
 * plus one bf16 ulp (2^-7 relative) for bf16 outputs of non-EXACT ops. Every case is run twice
 * on `dev` and must be bit-identical (determinism). Range errors (token ids, positions, labels)
 * must be reported as RRC_ERR_RANGE by the op or by the next sync, after which the backend must
 * work normally. Pure C11; prints no timing of any kind.
 */
#ifndef RRC_SELFTEST_OPS_H
#define RRC_SELFTEST_OPS_H

#include <stdint.h>
#include <stdio.h>

#include "rrc/backend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rrc_st_opts {
    uint64_t seed;
    int verbose;     /* print every case, not only failures and per-op summaries */
    int tf32;        /* dev GEMMs may use TF32 (widens the GEMM tolerance) */
    int quick;       /* smaller large-GEMM case (host runs) */
    FILE *out;       /* NULL: stdout */
    const char *only; /* run only ops whose name contains this substring (NULL: all) */
    /* describes the path of dev's most recent gemm into buf; returns 0 if unknown */
    int (*gemm_path)(const rrc_backend *dev, char *buf, size_t n);
} rrc_st_opts;

typedef struct rrc_st_result {
    int checks, failures;
    int det_checks, det_failures;
    int range_checks, range_failures;
} rrc_st_result;

/* Returns the number of failed checks (0 = all PASS). */
int rrc_st_run(const rrc_backend *dev, const rrc_backend *ref, const rrc_st_opts *o, rrc_st_result *res);

#ifdef __cplusplus
}
#endif

#endif /* RRC_SELFTEST_OPS_H */
