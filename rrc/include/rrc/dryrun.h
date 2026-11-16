/* rrc/dryrun.h — a recording "dry-run" backend: every op of the frozen table (backend.h) is
 * accepted and recorded, nothing is computed (rrc/docs/DESIGN.md §13).
 *
 * Memory: alloc returns fake, non-dereferenceable device handles (host_accessible = 0) and tracks
 * live and peak bytes. Handles lie in the upper (kernel) half of the 64-bit address space
 * (0xffff800000000000 + offset, 256-byte aligned, never reused, never wrapping), where a user-mode
 * load or store faults on x86-64 and AArch64, so an accidental dereference crashes instead of
 * reading garbage. Distinct allocations never overlap, so the alias checks of src/ops.c and the
 * alignment computation of the GEMM plan see a consistent address space.
 *
 * Transfers: memcpy_h2d / memcpy_d2d / memset0 / sync only count; memcpy_d2h zero-fills the host
 * destination (so code that reads a downloaded value sees 0, deterministically).
 *
 * GEMM: each call records m, n, k, batch and the A/B/C dtypes, and asks the CUDA line's pure host
 * plan (cuda/gemm_plan.c, level NATIVE = the first rung of the CUDA backend's ladder) which path it
 * would take: native cuBLASLt types, or promoted operands / an f32 C scratch, how many chunks, how
 * much scratch, and how many elements the plan converts (bf16 -> f32 before the call, f32 -> bf16
 * after it). "Native" is what the plan decides from the dtypes under the assumed cuBLASLt type table
 * of DESIGN.md §11.2; whether the real library accepts a call is NOT_TESTED (no GPU).
 *
 * Everything this backend reports is COMPUTED from the call sequence, never measured.
 * One backend per thread (backend.h); each instance has its own counters.
 */
#ifndef RRC_DRYRUN_H
#define RRC_DRYRUN_H

#include <stddef.h>
#include <stdint.h>

#include "rrc/backend.h"
#include "rrc/common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum rrc_dryrun_op {
    RRC_DR_ALLOC = 0,
    RRC_DR_FREE,
    RRC_DR_H2D,
    RRC_DR_D2H,
    RRC_DR_D2D,
    RRC_DR_MEMSET0,
    RRC_DR_SYNC,
    RRC_DR_GEMM,
    RRC_DR_EMBEDDING_FWD,
    RRC_DR_ADD_BIAS,
    RRC_DR_BIAS_GRAD,
    RRC_DR_RMSNORM_FWD,
    RRC_DR_RMSNORM_BWD,
    RRC_DR_ROPE_FWD,
    RRC_DR_ROPE_BWD,
    RRC_DR_ATTN_SOFTMAX_FWD,
    RRC_DR_ATTN_SOFTMAX_BWD,
    RRC_DR_SWIGLU_FWD,
    RRC_DR_SWIGLU_BWD,
    RRC_DR_ADD,
    RRC_DR_AXPY,
    RRC_DR_SCALE,
    RRC_DR_CONVERT,
    RRC_DR_SUM_SQUARES,
    RRC_DR_CROSS_ENTROPY,
    RRC_DR_ADAMW,
    RRC_DR_N_OPS
} rrc_dryrun_op;

/* "alloc", "gemm", "rmsnorm_fwd", ... (the op-table field names); NULL for an invalid value. */
const char *rrc_dryrun_op_name(int op);

/* What the GEMM plan says about one gemm call (level NATIVE of cuda/gemm_plan.h). */
typedef struct rrc_dryrun_gemm {
    rrc_gemm_args args;           /* the call as the backend received it (validated, k > 0) */
    int promote_a, promote_b;     /* the bf16 operand of a mixed pair is widened to f32 in scratch */
    int c_scratch;                /* a bf16 C next to f32 operands goes through an f32 scratch */
    int native;                   /* !promote_a && !promote_b && !c_scratch */
    int native_bf16;              /* native with bf16 A and B (the bf16 x bf16 -> f32-compute path) */
    int tf32;                     /* the plan selected TF32 (only with allow_tf32) */
    int64_t n_chunks;             /* chunks of the plan (scratch bounded by the cap) */
    uint64_t scratch_bytes;       /* scratch the plan needs (f32 elements x 4) */
    uint64_t convert_elems;       /* elements written by the plan's conversions, all chunks */
    uint64_t convert_bytes;       /* bytes read + written by those conversions */
    unsigned min_align;           /* smallest operand alignment (bytes) over all chunks' calls */
    double flops;                 /* 2 m n k batch */
    char desc[128];               /* rrc_gemm_plan_describe */
} rrc_dryrun_gemm;

/* One recorded call. shape[] is op-specific (gemm: m, n, k, batch; row ops: rows, cols / dim;
 * element ops: n; memory ops: bytes); dtype[] lists the float/int operands in argument order. */
typedef struct rrc_dryrun_call {
    rrc_dryrun_op op;
    int n_shape, n_dtype;
    int64_t shape[4];
    rrc_dtype dtype[6];
    uint64_t bytes;               /* computed: one read of every input + one write of every output */
    const rrc_dryrun_gemm *gemm;  /* gemm calls only, else NULL */
    const void *ptr[3];           /* gemm: a, b, c pointers (fake handles); else NULL */
} rrc_dryrun_call;

typedef void (*rrc_dryrun_hook)(void *ctx, const rrc_dryrun_call *call);

typedef struct rrc_dryrun_opts {
    int64_t gemm_scratch_cap_elems; /* <= 0: RRC_GEMM_SCRATCH_CAP_DEFAULT (the CUDA backend's default) */
    int allow_tf32;                 /* as rrc_cuda_options.allow_tf32 (default 0) */
    uint64_t capacity_bytes;        /* fake address space; 0: 2^44 bytes (16 TiB). Exhausted: RRC_ERR_OOM */
    uint64_t limit_bytes;           /* > 0: alloc fails with RRC_ERR_OOM when live bytes would exceed it */
} rrc_dryrun_opts;

void rrc_dryrun_opts_default(rrc_dryrun_opts *o);

typedef struct rrc_dryrun_stats {
    uint64_t live_bytes, peak_bytes; /* op-table allocations (alloc - free) */
    int64_t live_allocs, n_alloc, n_free;
    int64_t calls[RRC_DR_N_OPS];     /* per op (alloc / free / transfers included) */
    uint64_t bytes[RRC_DR_N_OPS];    /* computed bytes moved per op */
    uint64_t h2d_bytes, d2h_bytes, d2d_bytes, memset_bytes;
    double gemm_flops, gemm_flops_native, gemm_flops_native_bf16;
    uint64_t gemm_scratch_max_bytes; /* max scratch_bytes over the gemm calls (the CUDA backend grows to it) */
    uint64_t gemm_convert_elems, gemm_convert_bytes;
    uint64_t sum_squares_partials_max_bytes; /* the CUDA backend's partial buffer: ceil(n / 4096) floats */
} rrc_dryrun_stats;

rrc_status rrc_dryrun_backend_create(const rrc_dryrun_opts *o, const rrc_backend **out);
/* Frees the table and its bookkeeping (live fake allocations are simply forgotten). */
void rrc_dryrun_backend_destroy(const rrc_backend *be);
int rrc_backend_is_dryrun(const rrc_backend *be);

/* Called once per recorded op (after the counters are updated); NULL removes the hook. */
void rrc_dryrun_set_hook(const rrc_backend *be, rrc_dryrun_hook fn, void *ctx);
void rrc_dryrun_get_stats(const rrc_backend *be, rrc_dryrun_stats *out);
/* Zero the per-op counters and the gemm / transfer totals; live bytes and live allocations are
 * kept, the peak is reset to the current live bytes. */
void rrc_dryrun_reset_counters(const rrc_backend *be);

#ifdef __cplusplus
}
#endif

#endif /* RRC_DRYRUN_H */
