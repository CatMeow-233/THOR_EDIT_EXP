/* rrc/offdev.h — TEST ONLY: the "off-device" CPU backend (rrc/docs/DESIGN.md §14).
 *
 * rrc_offdev_backend_create(inner) wraps a host-accessible backend (rrc_backend_cpu() or
 * rrc_backend_cpu_f32acc()) so that its "device pointers" are NOT host pointers: every allocation
 * gets an address range inside one PROT_NONE reservation (mapped neither readable nor writable),
 * and the bytes live in a separate allocation of the inner backend. Every memory function and op
 * translates the handles it receives (base + offset, so views work) to the real storage and calls
 * the inner backend's function, so results are the inner backend's bit for bit. Any direct host
 * read or write of a device pointer (instead of memcpy_d2h / memcpy_h2d) faults with SIGSEGV.
 *
 * host_accessible is 0. The table name is the inner name + "-offdev" ("cpu-offdev",
 * "cpu-f32acc-offdev"). Device address ranges are never reused (bump allocation with an unmapped
 * guard page after every allocation), so a pointer kept after its free does not translate.
 * A pointer that lies in no live allocation is refused with RRC_ERR_INVALID_ARG; a free of such a
 * pointer is counted in bad_frees. Same threading rule as every backend: one host thread at a time.
 *
 * Not part of rrc/rrc.h: used by the C tests (rrc_tests --offdev), by rrc_fwdbwd --backend
 * cpu-offdev|cpu-f32acc-offdev and by rrc_train --test-offdev (both hidden, test-only options).
 */
#ifndef RRC_OFFDEV_H
#define RRC_OFFDEV_H

#include <stdint.h>

#include "rrc/backend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rrc_offdev_stats {
    int64_t allocs;     /* successful non-empty allocations */
    int64_t frees;      /* frees of live allocations */
    int64_t live;       /* allocations not freed yet */
    int64_t live_bytes; /* their bytes */
    int64_t peak_bytes; /* maximum of live_bytes */
    int64_t bad_frees;  /* frees of a pointer that is not the base of a live allocation */
    int64_t bad_ptrs;   /* translations refused (pointer or extent outside every live allocation) */
} rrc_offdev_stats;

/* Create an off-device wrapper of `inner` (must be host_accessible). Fails with RRC_ERR_OOM if the
 * address-space reservation cannot be made; there is no fallback to host pointers. */
rrc_status rrc_offdev_backend_create(const rrc_backend *inner, const rrc_backend **out);
/* 1 if `be` was created by rrc_offdev_backend_create (and not destroyed yet). */
int rrc_backend_is_offdev(const rrc_backend *be);
/* Counters so far (RRC_ERR_INVALID_ARG if `be` is not an offdev backend). */
rrc_status rrc_offdev_get_stats(const rrc_backend *be, rrc_offdev_stats *out);
/* Destroy the wrapper: frees what is still allocated, releases the reservation and, if `final` is
 * non-NULL, stores the final counters. Returns RRC_ERR_INTERNAL (with a message) if allocations
 * were still live or a bad free / refused translation happened, RRC_ERR_BACKEND if munmap failed,
 * RRC_OK otherwise. NULL is a no-op. */
rrc_status rrc_offdev_backend_destroy(const rrc_backend *be, rrc_offdev_stats *final);

#ifdef __cplusplus
}
#endif

#endif /* RRC_OFFDEV_H */
