/* rrc/cuda/rrc_cuda.h — the CUDA backend (DESIGN.md §11). COMPILED-ONLY until rrc_selftest has
 * been run on a GPU.
 *
 * rrc_cuda_backend_create returns a `const rrc_backend *` implementing every op of
 * include/rrc/backend.h with the CPU backend's semantics: fp32 math, fp32 reductions in a fixed
 * order (bit-identical run to run), one RNE rounding at the output, the aliasing backend.h allows.
 * One backend owns one CUDA stream, one cuBLASLt handle and workspace, a device error flag and
 * a GEMM conversion scratch; all ops are enqueued on that stream.
 *
 * Range errors (token ids, position ids, labels outside their range) are detected on the device:
 * the offending items write nothing, the error flag is set, and the error is returned as
 * RRC_ERR_RANGE by the next rrc_sync / memcpy_d2h (or by the op itself with sync_each_op). After
 * a range error the outputs of the ops enqueued since the previous sync are unspecified.
 * Link with: librrc_cuda.a librrc.a -lcublasLt -lcudart -lstdc++ (see the Makefile).
 */
#ifndef RRC_CUDA_H
#define RRC_CUDA_H

#include <stddef.h>
#include <stdint.h>

#include "gemm_plan.h"
#include "rrc/backend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rrc_cuda_options {
    int allow_tf32;                 /* f32 x f32 GEMMs may use TF32 tensor cores (default 0; env RRC_CUDA_ALLOW_TF32=1) */
    int sync_each_op;               /* synchronize + check the error flag after every op (default 0; env RRC_CUDA_SYNC_EACH_OP=1) */
    int gemm_min_level;             /* rrc_gemm_level to start the ladder at (default 0 = native; 2 = always the kernel) */
    int64_t gemm_scratch_cap_elems; /* promotion scratch cap in f32 elements (default RRC_GEMM_SCRATCH_CAP_DEFAULT) */
    size_t workspace_bytes;         /* cuBLASLt workspace (default 32 MiB) */
    unsigned max_grid_blocks;       /* cap on the kernels' grid size (0 = default 2^20 blocks); a small value makes every
                                     * grid-stride / row-stride loop wrap at small sizes (rrc_selftest's grid pass) */
    /* Pinned staging ring of memcpy_h2d (DESIGN.md §15.3): h2d_staging_slots slots of
     * h2d_staging_bytes / h2d_staging_slots bytes each. memcpy_h2d copies the host source into the
     * next slot, enqueues the slot -> device copy and records the slot's event, and returns without
     * a stream synchronize; a slot is reused only after its event has completed; larger copies go
     * through the ring in slot-sized chunks. 0 bytes (env RRC_CUDA_H2D_STAGING_BYTES=0) disables
     * the ring: then every memcpy_h2d is cudaMemcpyAsync + stream synchronize, as before. If the
     * pinned allocation fails at creation the ring is disabled with a warning. */
    size_t h2d_staging_bytes;       /* default 32 MiB (env RRC_CUDA_H2D_STAGING_BYTES overrides) */
    int h2d_staging_slots;          /* 1 .. RRC_CUDA_MAX_STAGING_SLOTS, default 4 */
} rrc_cuda_options;

#define RRC_CUDA_MAX_STAGING_SLOTS 16

/* Defaults, then the environment overrides named above. */
void rrc_cuda_options_default(rrc_cuda_options *o);

rrc_status rrc_cuda_backend_create(int device, const rrc_backend **out);
rrc_status rrc_cuda_backend_create_ex(int device, const rrc_cuda_options *o, const rrc_backend **out);
/* Synchronizes, then releases the stream, handle, workspace, scratch and the table itself. */
void rrc_cuda_backend_destroy(const rrc_backend *be);
int rrc_backend_is_cuda(const rrc_backend *be);

typedef struct rrc_cuda_device_info {
    char name[256];
    int device, cc_major, cc_minor, sm_count;
    size_t total_mem, free_mem;
    int driver_version, runtime_version; /* e.g. 12090 */
    size_t cublaslt_version;
} rrc_cuda_device_info;

rrc_status rrc_cuda_device_info_get(const rrc_backend *be, rrc_cuda_device_info *info);

/* Path of the most recent gemm (level, promotions, chunks); returns 0 if none or not CUDA. */
int rrc_cuda_last_gemm(const rrc_backend *be, rrc_gemm_exec_report *rep);

/* ---- build, counters, census, timing (rrc_selftest --report, rrc_bench) ---------------------------- */

/* What this library was built with (compile-time values; 0 / "" where unknown). */
typedef struct rrc_cuda_build_info {
    int host_shim;            /* 1: the HOST-ONLY test build on the fake runtime (tests/cuda_host): no GPU, no timing */
    int nvcc_major, nvcc_minor, nvcc_build; /* __CUDACC_VER_*__ of the kernels' compiler */
    int cudart_header;        /* CUDART_VERSION of the headers compiled against */
    int cublas_header;        /* CUBLAS_VERSION of the headers (major * 10000 + minor * 100 + patch) */
    char arch_list[64];       /* __CUDA_ARCH_LIST__ of the kernels (e.g. "1200") */
} rrc_cuda_build_info;
void rrc_cuda_build_info_get(rrc_cuda_build_info *b);
/* 1 in the HOST-ONLY build on the fake runtime (event timings there are host wall clock, never GPU
 * time), 0 in the real CUDA build. */
int rrc_cuda_build_is_host_shim(void);

/* Kernel image actually loaded on the device (cudaFuncGetAttributes of one rrc kernel): binary
 * (SASS) and PTX versions, e.g. 120 / 120. Needs a backend (device selected). */
rrc_status rrc_cuda_kernel_versions(const rrc_backend *be, int *binary_version, int *ptx_version);

/* Counters of one backend since creation (or the last reset). Byte counts of device memory; FLOP
 * counts are computed from GEMM shapes (2 m n k batch), never measured. */
typedef struct rrc_cuda_stats {
    size_t alloc_bytes, alloc_peak_bytes; /* live / peak bytes of be->alloc allocations (rrc_buf memory) */
    size_t internal_bytes;                /* cuBLASLt workspace + GEMM scratch + sum_squares partials + error flag */
    size_t total_peak_bytes;              /* peak of alloc_bytes + internal_bytes (sampled at every change) */
    int64_t n_allocs, n_frees;
    size_t staging_bytes;                 /* pinned staging ring (host memory), 0 if disabled */
    int staging_slots;
    int64_t h2d_calls, h2d_bytes;
    int64_t h2d_staged_chunks;            /* slot-sized pieces that went through the ring */
    int64_t h2d_slot_waits;               /* chunks that had to wait for the slot's previous copy (event sync) */
    int64_t h2d_direct;                   /* h2d calls without the ring (copy + stream synchronize) */
    int64_t stream_syncs;                 /* full stream synchronizes issued by the backend */
    int64_t gemm_calls[8][3];             /* [4 (A bf16) + 2 (B bf16) + (C bf16)][ladder level 0 / 1 / 2] */
    double gemm_flops[8][3];              /* computed 2 m n k batch of those calls */
    int64_t gemm_chunked;                 /* calls that ran in more than one scratch chunk */
} rrc_cuda_stats;
rrc_status rrc_cuda_stats_get(const rrc_backend *be, rrc_cuda_stats *s);
/* Zero the h2d / sync / gemm counters and the census, set the peaks to the current values. */
rrc_status rrc_cuda_stats_reset(const rrc_backend *be);

/* GEMM census: one entry per distinct call geometry (row-major args) since enabling it. */
typedef struct rrc_cuda_gemm_shape {
    int trans_a, trans_b;
    int dt_a, dt_b, dt_c; /* rrc_dtype */
    int64_t m, n, k, batch, lda, ldb, ldc;
    int64_t calls;
    int level;            /* ladder level of the last call (0 native/promoted, 1 promote-all, 2 kernel) */
    int64_t chunks;       /* scratch chunks of the last call */
    char path[128];       /* rrc_gemm_exec_report.desc of the last call */
} rrc_cuda_gemm_shape;
#define RRC_CUDA_CENSUS_MAX 512
/* Enable (clears it) or disable recording. */
rrc_status rrc_cuda_gemm_census_enable(const rrc_backend *be, int on);
/* Copies up to `cap` entries; returns the number of entries; *dropped = calls of geometries that
 * did not fit in RRC_CUDA_CENSUS_MAX entries. */
size_t rrc_cuda_gemm_census_get(const rrc_backend *be, rrc_cuda_gemm_shape *out, size_t cap, int64_t *dropped);

/* cudaEvent timing on the backend's stream: marks 0 .. n_marks-1, elapsed_ms(i, j) waits for mark j
 * and returns the GPU time between the two marks. In the HOST-ONLY build the value is host wall
 * clock (rrc_cuda_build_is_host_shim). */
typedef struct rrc_cuda_timer rrc_cuda_timer;
rrc_status rrc_cuda_timer_create(const rrc_backend *be, int n_marks, rrc_cuda_timer **out);
rrc_status rrc_cuda_timer_mark(rrc_cuda_timer *t, int i);
rrc_status rrc_cuda_timer_elapsed_ms(rrc_cuda_timer *t, int i, int j, float *ms);
void rrc_cuda_timer_destroy(rrc_cuda_timer *t);

#ifdef __cplusplus
}
#endif

#endif /* RRC_CUDA_H */
