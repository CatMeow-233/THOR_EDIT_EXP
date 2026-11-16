/* rrc/tensor.h — device buffers.
 *
 * An rrc_buf is a typed view of device memory owned by one backend: {backend, dtype, numel, ptr}.
 * `numel` is the number of elements addressable from `ptr` (the op wrappers check every access
 * against it). Views (rrc_view) share the storage of their parent and must not be freed; only a
 * buffer returned by rrc_buf_alloc is `owned` and may be passed to rrc_buf_free.
 */
#ifndef RRC_TENSOR_H
#define RRC_TENSOR_H

#include <stdint.h>

#include "rrc/common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rrc_backend rrc_backend;

typedef struct rrc_buf {
    const rrc_backend *be; /* backend that owns the memory (NULL for the empty buffer) */
    rrc_dtype dtype;
    int64_t numel; /* elements addressable from ptr; -1 marks an invalid view */
    void *ptr;     /* device pointer (a host pointer for the CPU backend) */
    int owned;     /* 1 if returned by rrc_buf_alloc (freeable), 0 for views */
} rrc_buf;

/* The empty buffer (used for optional operands, e.g. "no doc_ids"). */
static inline rrc_buf rrc_buf_none(void) {
    rrc_buf b;
    b.be = 0;
    b.dtype = RRC_F32;
    b.numel = 0;
    b.ptr = 0;
    b.owned = 0;
    return b;
}
static inline int rrc_buf_is_none(rrc_buf b) { return b.ptr == 0 && b.numel == 0 && b.be == 0; }

/* Allocate `numel` elements of `dtype` on `be`, zero-initialised. */
rrc_status rrc_buf_alloc(const rrc_backend *be, rrc_dtype dtype, int64_t numel, rrc_buf *out);
/* Free an owned buffer and reset it to rrc_buf_none(); views are rejected (RRC_ERR_INVALID_ARG). */
rrc_status rrc_buf_free(rrc_buf *b);

/* View of elements [offset, offset + numel) of `b`. An out-of-range request returns a view with
 * numel = -1 and ptr = NULL, which every op rejects with RRC_ERR_SHAPE ("invalid view"), so view
 * arithmetic can be written inline and checked at the next op. */
rrc_buf rrc_view(rrc_buf b, int64_t offset, int64_t numel);
/* View from `offset` to the end of `b`. */
rrc_buf rrc_view_from(rrc_buf b, int64_t offset);

/* Host <-> device transfers of `numel` elements of b.dtype (synchronous, see backend.h). */
rrc_status rrc_buf_upload(rrc_buf dst, const void *host, int64_t numel);
rrc_status rrc_buf_download(rrc_buf src, void *host, int64_t numel);
/* Device copy of `numel` elements; same dtype, non-overlapping. */
rrc_status rrc_buf_copy(rrc_buf dst, rrc_buf src, int64_t numel);
/* Zero `numel` elements. */
rrc_status rrc_buf_zero(rrc_buf b, int64_t numel);

#ifdef __cplusplus
}
#endif

#endif /* RRC_TENSOR_H */
