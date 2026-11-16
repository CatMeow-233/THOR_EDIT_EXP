/* rrc/cuda/gemm_plan.h — pure host translation of a row-major rrc_gemm into column-major
 * cuBLASLt calls (DESIGN.md §11.2). Compiled by gcc (CPU tests, host emulation backend) and
 * linked into librrc_cuda; no CUDA dependency.
 *
 * Row-major X (r x c, ld) is column-major X^T (c x r, ld), so C = op(A) op(B) is computed as the
 * column-major C^T = op(B)^T op(A)^T: cuBLASLt's A is rrc's B (transa = trans_b), cuBLASLt's B is
 * rrc's A (transb = trans_a), M = n, N = m, K = k, D == C with ld = ldc, batch strides unchanged.
 *
 * Levels (a ladder tried in order by rrc_gemm_execute):
 *   NATIVE       A/B/C types passed as they are when the triple is one cuBLASLt is expected to
 *                support with CUBLAS_COMPUTE_32F (f32 f32 f32, bf16 bf16 f32, bf16 bf16 bf16);
 *                otherwise the bf16 operand of a mixed pair is promoted to f32 and a bf16 C next to
 *                f32 operands goes through an f32 scratch (computed in f32, rounded RNE once).
 *   PROMOTE_ALL  every bf16 operand promoted, bf16 C through f32 scratch: an f32 x f32 -> f32 call.
 *   KERNEL       the rrc_kc_item_lt_matmul kernel executes the (unpromoted) column-major call.
 * Promotion is exact (bf16 -> f32 widening). f32 is never demoted. Promoted operands and C go
 * through a scratch bounded by opts.scratch_cap_elems: the problem is split into chunks of
 * (batch range, row range of C, column range of C); a chunk never needs more than the cap unless
 * a single row/column of an operand (k elements) already exceeds it (then the minimum chunk is
 * used and scratch_elems says how much that needs).
 */
#ifndef RRC_GEMM_PLAN_H
#define RRC_GEMM_PLAN_H

#include <stddef.h>
#include <stdint.h>

#include "rrc/backend.h"
#include "rrc_kc_items.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RRC_GEMM_SCRATCH_CAP_DEFAULT ((int64_t)16 * 1024 * 1024) /* f32 elements = 64 MiB */
#define RRC_GEMM_MAX_ALIGN 256u
#define RRC_GEMM_SCRATCH_LD_ALIGN 8 /* scratch leading dimensions >= 64 are multiples of 8 elements */

typedef enum rrc_gemm_level {
    RRC_GEMM_LEVEL_NATIVE = 0,
    RRC_GEMM_LEVEL_PROMOTE_ALL = 1,
    RRC_GEMM_LEVEL_KERNEL = 2
} rrc_gemm_level;

typedef struct rrc_gemm_plan_opts {
    int64_t scratch_cap_elems; /* <= 0: RRC_GEMM_SCRATCH_CAP_DEFAULT */
    int allow_tf32;            /* f32 x f32 calls use CUBLAS_COMPUTE_32F_FAST_TF32 (default 0) */
    unsigned max_align;        /* 0: RRC_GEMM_MAX_ALIGN */
} rrc_gemm_plan_opts;

/* One column-major call (cuBLASLt matmul or the fallback kernel). */
typedef struct rrc_lt_call {
    rrc_kc_lt_p p;
    unsigned align_a, align_b, align_c; /* bytes guaranteed for every batch slice (power of 2) */
    int tf32;                           /* compute type CUBLAS_COMPUTE_32F_FAST_TF32 */
} rrc_lt_call;

typedef struct rrc_gemm_chunk {
    int n_pre;
    rrc_kc_conv2d_p pre[3]; /* executed in order before the call */
    rrc_lt_call call;
    int has_post;
    rrc_kc_conv2d_p post; /* f32 scratch -> bf16 C */
} rrc_gemm_chunk;

typedef struct rrc_gemm_plan {
    rrc_gemm_args g;
    rrc_gemm_level level;
    int promote_a, promote_b, c_scratch;
    int a_bcast, b_bcast;     /* stride 0 operands: promoted once per chunk */
    int64_t lm, ln, nb;       /* chunk extents: rows of C, columns of C, batch slices */
    int64_t cm, cn, cb;       /* number of chunks along each */
    int64_t n_chunks;         /* cm * cn * cb */
    int64_t off_a, off_b, off_c; /* element offsets of the scratch regions (f32) */
    int64_t scratch_elems;    /* f32 elements of scratch needed (0: none) */
    int tf32;
    unsigned max_align;
} rrc_gemm_plan;

/* Largest power of two <= cap dividing the byte address of ptr, ld*es, and stride*es when
 * batch > 1 and stride != 0 (the alignment of every column and every batch slice). */
unsigned rrc_gemm_align_bytes(const void *ptr, size_t es, int64_t ld, int64_t stride, int64_t batch, unsigned cap);

/* Build the plan of `level` for validated args (as rrc_gemm passes them to a backend: k > 0). */
rrc_status rrc_gemm_plan_make(const rrc_gemm_args *g, const rrc_gemm_plan_opts *opts, rrc_gemm_level level,
                              rrc_gemm_plan *p);
/* Chunk `idx` in [0, p->n_chunks) with scratch regions at `scratch` (>= scratch_elems floats;
 * may be NULL when scratch_elems == 0). */
void rrc_gemm_plan_chunk(const rrc_gemm_plan *p, void *scratch, int64_t idx, rrc_gemm_chunk *c);
/* Human-readable path, e.g. "native", "promoted A(bf16->f32) C(f32 scratch)", "kernel". */
void rrc_gemm_plan_describe(const rrc_gemm_plan *p, char *buf, size_t n);

/* ---- ladder driver (shared by the CUDA backend and the host emulation backend) ---------------- */

typedef struct rrc_gemm_exec_ops {
    /* device scratch of at least `bytes` bytes, kept by the backend (grown on demand) */
    rrc_status (*scratch)(void *ctx, size_t bytes, void **ptr);
    /* 1 if the column-major call can run as a library matmul (cuBLASLt heuristic found an
     * algorithm), 0 otherwise (the ladder moves to the next level) */
    int (*lt_supported)(void *ctx, const rrc_lt_call *c);
    rrc_status (*lt_run)(void *ctx, const rrc_lt_call *c);
    rrc_status (*kernel_run)(void *ctx, const rrc_kc_lt_p *p);
    rrc_status (*convert)(void *ctx, const rrc_kc_conv2d_p *p);
} rrc_gemm_exec_ops;

typedef struct rrc_gemm_exec_report {
    rrc_gemm_level level;
    int promote_a, promote_b, c_scratch;
    int64_t n_chunks;
    char desc[128];
} rrc_gemm_exec_report;

/* Run g through the ladder starting at min_level: the first level whose every chunk is
 * lt_supported runs; KERNEL always runs. `rep` (may be NULL) receives the path taken. */
rrc_status rrc_gemm_execute(const rrc_gemm_args *g, const rrc_gemm_plan_opts *opts, rrc_gemm_level min_level,
                            const rrc_gemm_exec_ops *ops, void *ctx, rrc_gemm_exec_report *rep);

#ifdef __cplusplus
}
#endif

#endif /* RRC_GEMM_PLAN_H */
