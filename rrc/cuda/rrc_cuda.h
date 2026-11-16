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
} rrc_cuda_options;

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

#ifdef __cplusplus
}
#endif

#endif /* RRC_CUDA_H */
