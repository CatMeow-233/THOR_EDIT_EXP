/* rrc/tests/cuda_host/test_cuda_host.c — HOST-ONLY tests of the real CUDA backend and kernel
 * sources (rrc/cuda/cuda_backend.cu, cuda_kernels.cu) built with g++ against the fake runtime of
 * fake_cuda.cpp (see shim.h). Nothing here runs on a GPU; the cuBLASLt calls go to a host
 * stand-in. What this checks is rrc's own host and kernel-shell logic: buffer ownership, the
 * device error flag, the GEMM ladder's use of the heuristic, the launch geometry, and the whole
 * selftest harness through the backend's op table.
 *
 *   test_cuda_host [filter]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fake_cuda.h"
#include "rrc/rrc.h"
#include "rrc_cuda.h"
#include "selftest_ops.h"

static int g_fail = 0, g_checks = 0;
#define EXPECT(cond, ...)                                                                               \
    do {                                                                                                \
        g_checks++;                                                                                     \
        if (!(cond)) {                                                                                  \
            g_fail++;                                                                                   \
            fprintf(stderr, "    FAILED %s:%d: (%s) ", __FILE__, __LINE__, #cond);                      \
            fprintf(stderr, __VA_ARGS__);                                                               \
            fprintf(stderr, "\n");                                                                      \
        }                                                                                               \
    } while (0)
#define OK(expr)                                                                                        \
    do {                                                                                                \
        rrc_status ok_s_ = (expr);                                                                      \
        g_checks++;                                                                                     \
        if (ok_s_ != RRC_OK) {                                                                          \
            g_fail++;                                                                                   \
            fprintf(stderr, "    FAILED %s:%d: %s returned %s: %s\n", __FILE__, __LINE__, #expr,        \
                    rrc_status_str(ok_s_), rrc_last_error());                                           \
            return;                                                                                     \
        }                                                                                               \
    } while (0)

/* rrc_cuda_options_default reads RRC_CUDA_SYNC_EACH_OP, RRC_CUDA_ALLOW_TF32 and
 * RRC_CUDA_H2D_STAGING_BYTES: the tests pin every such option, so the caller's environment cannot
 * change what they measure (review finding: RRC_CUDA_SYNC_EACH_OP=1 failed the deferred-error test). */
static void pin_env_options(rrc_cuda_options *o) {
    o->sync_each_op = 0;
    o->allow_tf32 = 0;
    o->h2d_staging_bytes = (size_t)32 << 20;
    o->h2d_staging_slots = 4;
}

static const rrc_backend *make_be(unsigned max_grid, int64_t scratch_cap) {
    rrc_cuda_options o;
    rrc_cuda_options_default(&o);
    pin_env_options(&o);
    o.max_grid_blocks = max_grid;
    if (scratch_cap > 0) o.gemm_scratch_cap_elems = scratch_cap;
    const rrc_backend *be = NULL;
    if (rrc_cuda_backend_create_ex(0, &o, &be) != RRC_OK) {
        fprintf(stderr, "cannot create the CUDA backend on the fake runtime: %s\n", rrc_last_error());
        exit(2);
    }
    return be;
}

static rrc_buf up_f32(const rrc_backend *be, const float *h, int64_t n) {
    rrc_buf b;
    if (rrc_buf_alloc(be, RRC_F32, n, &b) != RRC_OK || rrc_buf_upload(b, h, n) != RRC_OK) {
        fprintf(stderr, "upload failed: %s\n", rrc_last_error());
        exit(2);
    }
    return b;
}
static rrc_buf up_i32(const rrc_backend *be, const int32_t *h, int64_t n) {
    rrc_buf b;
    if (rrc_buf_alloc(be, RRC_I32, n, &b) != RRC_OK || rrc_buf_upload(b, h, n) != RRC_OK) {
        fprintf(stderr, "upload failed: %s\n", rrc_last_error());
        exit(2);
    }
    return b;
}

/* A failed growth of the sum_squares partial buffer must leave the backend without a dangling
 * pointer. Regression: grow() freed the old buffer and failed, but c->ss_part kept the freed
 * pointer (use-after-free by the next call's kernel, second cudaFree at destroy). */
static void t_sum_squares_grow_failure(void) {
    rrc_fake_stats s0, s1;
    rrc_fake_get_stats(&s0);
    const rrc_backend *be = make_be(0, 0);
    const int64_t n = 3 * 4096;
    float *h = (float *)malloc((size_t)n * sizeof(float));
    for (int64_t i = 0; i < n; i++) h[i] = 1.0f;
    rrc_buf x = up_f32(be, h, n), out;
    OK(rrc_buf_alloc(be, RRC_F32, 1, &out));
    float got = 0.0f;
    OK(rrc_sum_squares(be, 10, x, out, 0)); /* partials: 1 chunk */
    OK(rrc_buf_download(out, &got, 1));
    EXPECT(got == 10.0f, "first call: %g", (double)got);
    rrc_fake_fail_malloc_after(0); /* the growth to 3 chunks fails */
    rrc_clear_error();
    rrc_status st = rrc_sum_squares(be, n, x, out, 0);
    EXPECT(st == RRC_ERR_OOM, "growth failure reported as OOM: %s", rrc_status_str(st));
    rrc_clear_error();
    OK(rrc_sum_squares(be, 10, x, out, 0)); /* must get a live buffer again */
    OK(rrc_buf_download(out, &got, 1));
    EXPECT(got == 10.0f, "after the failure: %g", (double)got);
    OK(rrc_sum_squares(be, n, x, out, 0)); /* and the growth works once memory is there */
    OK(rrc_buf_download(out, &got, 1));
    EXPECT(got == (float)n, "after the failure, n=%lld: %g", (long long)n, (double)got);
    rrc_buf_free(&x);
    rrc_buf_free(&out);
    rrc_cuda_backend_destroy(be);
    free(h);
    rrc_fake_get_stats(&s1);
    EXPECT(s1.bad_frees == s0.bad_frees, "%ld cudaFree calls of pointers that are not live (double free)",
           s1.bad_frees - s0.bad_frees);
    EXPECT(s1.live == s0.live, "%ld device allocations leaked", s1.live - s0.live);
}

/* The error flag names one op and that op's smallest offending index. Regression: the op of the
 * first failure was kept but atomicMin ran for every failing op, so a later op's smaller index
 * was reported under the first op's name. */
static void t_error_flag_op_and_index(void) {
    for (int order = 0; order < 2; order++) {
        const rrc_backend *be = make_be(0, 0);
        float table[8] = {0}, x[8] = {1, 2, 3, 4, 5, 6, 7, 8}, ct[2] = {1, 1}, stab[2] = {0, 0};
        int32_t ids[8] = {0, 1, 2, 3, 0, 99, 1, 2}; /* token 5 bad (vocab 4) */
        int32_t pos[4] = {0, 7, 0, 0};              /* token 1 bad (n_pos 2) */
        rrc_buf tb = up_f32(be, table, 8), ib = up_i32(be, ids, 8), ob, xb = up_f32(be, x, 8), pb = up_i32(be, pos, 4),
                cb = up_f32(be, ct, 2), sb = up_f32(be, stab, 2), yb;
        OK(rrc_buf_alloc(be, RRC_F32, 16, &ob));
        OK(rrc_buf_alloc(be, RRC_F32, 8, &yb));
        rrc_status s1, s2;
        if (order == 0) {
            s1 = rrc_embedding_fwd(be, 8, 2, 4, ib, tb, ob);
            s2 = rrc_rope_fwd(be, 4, 1, 2, 2, xb, pb, cb, sb, 2, yb);
        } else {
            s1 = rrc_rope_fwd(be, 4, 1, 2, 2, xb, pb, cb, sb, 2, yb);
            s2 = rrc_embedding_fwd(be, 8, 2, 4, ib, tb, ob);
        }
        EXPECT(s1 == RRC_OK && s2 == RRC_OK, "deferred: the ops enqueue (%s, %s)", rrc_status_str(s1), rrc_status_str(s2));
        rrc_clear_error();
        rrc_status ss = rrc_sync(be);
        char msg[RRC_ERROR_MSG_MAX];
        snprintf(msg, sizeof msg, "%s", rrc_last_error());
        EXPECT(ss == RRC_ERR_RANGE, "sync reports the range error: %s", rrc_status_str(ss));
        const char *want_op = order == 0 ? "embedding_fwd" : "rope_fwd";
        const char *want_idx = order == 0 ? "row/token 5 " : "row/token 1 ";
        EXPECT(strstr(msg, want_op) && strstr(msg, want_idx), "order %d: expected %s with %s; got: %s", order, want_op,
               want_idx, msg);
        rrc_clear_error();
        OK(rrc_sync(be)); /* flag cleared */
        rrc_buf_free(&tb);
        rrc_buf_free(&ib);
        rrc_buf_free(&ob);
        rrc_buf_free(&xb);
        rrc_buf_free(&pb);
        rrc_buf_free(&cb);
        rrc_buf_free(&sb);
        rrc_buf_free(&yb);
        rrc_cuda_backend_destroy(be);
    }
}

static int cuda_path(const rrc_backend *be, char *buf, size_t n) {
    rrc_gemm_exec_report rep;
    if (!rrc_cuda_last_gemm(be, &rep)) return 0;
    snprintf(buf, n, "%s", rep.desc);
    return 1;
}

/* the selftest harness through the real backend + kernel shells, quietly into a log file */
static int run_harness(const rrc_backend *be, const char *only, int require_lt, int require_chunked, const char *log,
                       rrc_st_result *r) {
    char path[512];
    const char *dir = getenv("RRC_TEST_TMPDIR");
    snprintf(path, sizeof path, "%s/%s", dir ? dir : ".", log);
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    rrc_st_opts o;
    memset(&o, 0, sizeof o);
    o.seed = 4321;
    o.quick = 1;
    o.out = f;
    o.only = only;
    o.gemm_path = cuda_path;
    o.require_lt = require_lt;
    o.require_chunked = require_chunked;
    int fails = rrc_st_run(be, rrc_backend_cpu_f32acc(), &o, r);
    fclose(f);
    if (fails) fprintf(stderr, "    harness failures: see %s\n", path);
    return fails;
}

/* The heuristic query must not depend on a reduction-scheme mask admitting non-split-K
 * algorithms (cublasLt.h does not say whether a mask without a NONE bit does): with a stand-in
 * whose masked query returns nothing and whose unmasked query lists a split-K INPLACE algorithm
 * first, the backend must still reach cuBLASLt and must never run the INPLACE algorithm.
 * Evidence about rrc's selection logic only; the real library's mask semantics stay NOT_TESTED. */
static void t_reduction_scheme_fallback(void) {
    rrc_fake_set_lt_mode(RRC_FAKE_LT_MASK_DROPS_NONSPLIT);
    rrc_fake_stats s0, s1;
    rrc_fake_get_stats(&s0);
    const rrc_backend *be = make_be(0, 0);
    rrc_st_result r;
    int fails = run_harness(be, "gemm", 1, 0, "cuda_host_mask.log", &r);
    rrc_fake_get_stats(&s1);
    EXPECT(fails == 0, "%d failures (path failures %d)", fails, r.path_failures);
    EXPECT(s1.lt_matmuls > s0.lt_matmuls, "cuBLASLt was used (%ld matmuls)", s1.lt_matmuls - s0.lt_matmuls);
    EXPECT(s1.lt_violations == s0.lt_violations, "%ld stand-in violations (a split-K INPLACE algorithm was run?)",
           s1.lt_violations - s0.lt_violations);
    rrc_cuda_backend_destroy(be);
    rrc_fake_set_lt_mode(RRC_FAKE_LT_ACCEPT);
}

/* Every op of the selftest through the real backend and kernel shells, default grid; then with a
 * 3-block grid (every grid-stride / row-stride loop wraps) and a 1024-float GEMM scratch cap
 * (multi-chunk GEMMs); then with the library refusing everything, which must fail the
 * path requirement. */
static void t_harness_through_backend(void) {
    rrc_fake_stats s0, s1;
    rrc_fake_get_stats(&s0);
    rrc_st_result r;
    const rrc_backend *be = make_be(0, 0);
    int fails = run_harness(be, NULL, 1, 0, "cuda_host_default.log", &r);
    EXPECT(fails == 0 && r.checks > 400 && r.range_checks == 8, "default grid: %d failures of %d checks", fails, r.checks);
    rrc_cuda_backend_destroy(be);
    be = make_be(3, 1024);
    fails = run_harness(be, NULL, 0, 1, "cuda_host_grid3_cap1024.log", &r);
    EXPECT(fails == 0 && r.checks > 400, "3-block grid, 1024-float scratch: %d failures of %d checks", fails, r.checks);
    rrc_cuda_backend_destroy(be);
    rrc_fake_set_lt_mode(RRC_FAKE_LT_REJECT);
    be = make_be(0, 0);
    fails = run_harness(be, "gemm", 1, 0, "cuda_host_reject.log", &r);
    EXPECT(fails > 0 && r.failures == 0 && r.path_failures >= 9, "library refuses all: %d fails, %d path failures", fails,
           r.path_failures);
    rrc_cuda_backend_destroy(be);
    rrc_fake_set_lt_mode(RRC_FAKE_LT_ACCEPT);
    rrc_fake_get_stats(&s1);
    EXPECT(s1.lt_violations == s0.lt_violations, "%ld cuBLASLt stand-in violations (descriptors, types, alignment)",
           s1.lt_violations - s0.lt_violations);
    EXPECT(s1.bad_frees == s0.bad_frees && s1.live == s0.live, "device memory: %ld bad frees, %ld leaked",
           s1.bad_frees - s0.bad_frees, s1.live - s0.live);
    printf("    (%ld kernel launches on host threads, %ld cuBLASLt stand-in matmuls)\n", s1.launches - s0.launches,
           s1.lt_matmuls - s0.lt_matmuls);
}

static const rrc_backend *make_be_staging(size_t bytes, int slots, int sync_each) {
    rrc_cuda_options o;
    rrc_cuda_options_default(&o);
    pin_env_options(&o);
    o.h2d_staging_bytes = bytes;
    o.h2d_staging_slots = slots;
    o.sync_each_op = sync_each;
    const rrc_backend *be = NULL;
    if (rrc_cuda_backend_create_ex(0, &o, &be) != RRC_OK) {
        fprintf(stderr, "cannot create the CUDA backend on the fake runtime: %s\n", rrc_last_error());
        exit(2);
    }
    return be;
}

/* Uploads `nb` buffers of the given sizes from ONE host array that is overwritten right after each
 * memcpy_h2d returns (the contract: the source may be reused on return), without any sync in
 * between, then downloads and compares. With the fake's lazy host->device copies (fake_cuda.h), a
 * copy that still read the caller's source, or a staging slot overwritten before the copy out of
 * it completed, delivers wrong bytes. Returns the number of wrong elements. */
static long upload_burst(const rrc_backend *be, const int64_t *sizes, int nb, rrc_fake_stats *mid) {
    rrc_buf bufs[16];
    int64_t maxn = 0;
    for (int i = 0; i < nb; i++) maxn = sizes[i] > maxn ? sizes[i] : maxn;
    float *src = (float *)malloc((size_t)maxn * sizeof(float)), *got = (float *)malloc((size_t)maxn * sizeof(float));
    long bad = 0;
    if (!src || !got || nb > 16) {
        free(src);
        free(got);
        return -1;
    }
    for (int i = 0; i < nb; i++) {
        if (rrc_buf_alloc(be, RRC_F32, sizes[i], &bufs[i]) != RRC_OK) {
            bad = -1;
            nb = i;
            break;
        }
    }
    for (int i = 0; bad == 0 && i < nb; i++) {
        for (int64_t j = 0; j < sizes[i]; j++) src[j] = (float)(1000 * i + j);
        if (rrc_buf_upload(bufs[i], src, sizes[i]) != RRC_OK) bad = -1;
        for (int64_t j = 0; j < maxn; j++) src[j] = -7.0f; /* reuse the source at once */
    }
    if (mid) rrc_fake_get_stats(mid);
    for (int i = 0; bad >= 0 && i < nb; i++) {
        if (rrc_buf_download(bufs[i], got, sizes[i]) != RRC_OK) {
            bad = -1;
            break;
        }
        for (int64_t j = 0; j < sizes[i]; j++) bad += got[j] != (float)(1000 * i + j);
    }
    for (int i = 0; i < nb; i++) rrc_buf_free(&bufs[i]);
    free(src);
    free(got);
    return bad;
}

/* memcpy_h2d through the pinned staging ring (DESIGN.md §15.3): correct data with the source
 * reused at once, chunking of large copies through a small ring that wraps several times, no full
 * stream synchronize, no pageable async copy, stream order with ops in between, and the fallbacks
 * (ring disabled, pinned allocation failure, sync_each_op). */
static void t_h2d_staging_ring(void) {
    static const int64_t sizes[] = {37, 1000, 5, 64, 64, 65, 3, 700, 1, 129}; /* floats; slot = 64 floats */
    const int nb = (int)(sizeof sizes / sizeof sizes[0]);
    int64_t chunks = 0;
    for (int i = 0; i < nb; i++) chunks += (sizes[i] * 4 + 255) / 256;
    rrc_fake_stats f0, f1, f2;
    rrc_cuda_stats cs;
    /* 4 slots x 256 bytes */
    const rrc_backend *be = make_be_staging(1024, 4, 0);
    OK(rrc_cuda_stats_get(be, &cs));
    EXPECT(cs.staging_bytes == 1024 && cs.staging_slots == 4, "ring %zu bytes, %d slots", cs.staging_bytes, cs.staging_slots);
    OK(rrc_cuda_stats_reset(be));
    rrc_fake_get_stats(&f0);
    long bad = upload_burst(be, sizes, nb, &f1);
    rrc_fake_get_stats(&f2);
    OK(rrc_cuda_stats_get(be, &cs));
    EXPECT(bad == 0, "%ld wrong elements after staged uploads", bad);
    EXPECT(f1.h2d_pending > 0, "the fake deferred the last copies (%ld pending): the check is not vacuous", f1.h2d_pending);
    EXPECT(f1.stream_syncs == f0.stream_syncs, "%ld stream synchronizes during the uploads (want 0)",
           f1.stream_syncs - f0.stream_syncs);
    EXPECT(f1.event_syncs > f0.event_syncs, "the ring wrapped and waited on slot events (%ld)", f1.event_syncs - f0.event_syncs);
    EXPECT(f2.h2d_pageable == f0.h2d_pageable, "%ld async copies from pageable memory", f2.h2d_pageable - f0.h2d_pageable);
    EXPECT(cs.h2d_calls == nb && cs.h2d_staged_chunks == chunks && cs.h2d_direct == 0,
           "calls %lld (want %d), chunks %lld (want %lld), direct %lld", (long long)cs.h2d_calls, nb,
           (long long)cs.h2d_staged_chunks, (long long)chunks, (long long)cs.h2d_direct);
    EXPECT(cs.h2d_slot_waits == chunks - 4, "slot waits %lld (want %lld)", (long long)cs.h2d_slot_waits,
           (long long)(chunks - 4));

    /* stream order with ops in between: y = 2 x1 + 2 x2, x re-uploaded into the same buffer */
    {
        const int64_t n = 300;
        float *h = (float *)malloc((size_t)n * sizeof(float)), *y = (float *)malloc((size_t)n * sizeof(float));
        rrc_buf xb, yb;
        OK(rrc_buf_alloc(be, RRC_F32, n, &xb));
        OK(rrc_buf_alloc(be, RRC_F32, n, &yb));
        for (int64_t j = 0; j < n; j++) h[j] = (float)j;
        OK(rrc_buf_upload(xb, h, n));
        OK(rrc_axpy(be, n, 2.0f, xb, yb));
        for (int64_t j = 0; j < n; j++) h[j] = (float)(3 * j);
        OK(rrc_buf_upload(xb, h, n));
        for (int64_t j = 0; j < n; j++) h[j] = -1.0f;
        OK(rrc_axpy(be, n, 2.0f, xb, yb));
        OK(rrc_buf_download(yb, y, n));
        long wrong = 0;
        for (int64_t j = 0; j < n; j++) wrong += y[j] != (float)(8 * j);
        EXPECT(wrong == 0, "%ld wrong elements of y = 2 x1 + 2 x2", wrong);
        rrc_buf_free(&xb);
        rrc_buf_free(&yb);
        free(h);
        free(y);
    }
    rrc_cuda_backend_destroy(be);

    /* ring disabled: every upload is copy + stream synchronize */
    be = make_be_staging(0, 4, 0);
    rrc_fake_get_stats(&f0);
    bad = upload_burst(be, sizes, nb, &f1);
    OK(rrc_cuda_stats_get(be, &cs));
    EXPECT(bad == 0 && cs.staging_bytes == 0 && cs.h2d_direct == nb && cs.h2d_staged_chunks == 0,
           "disabled ring: %ld wrong, direct %lld", bad, (long long)cs.h2d_direct);
    EXPECT(f1.stream_syncs - f0.stream_syncs == nb, "disabled ring: %ld syncs (want %d)", f1.stream_syncs - f0.stream_syncs, nb);
    rrc_cuda_backend_destroy(be);

    /* pinned allocation failure at creation: the backend works without the ring */
    rrc_fake_fail_host_alloc_after(1); /* the error flag's pinned word first, then the ring */
    be = make_be_staging(1024, 4, 0);
    bad = upload_burst(be, sizes, nb, NULL);
    OK(rrc_cuda_stats_get(be, &cs));
    EXPECT(bad == 0 && cs.staging_bytes == 0 && cs.h2d_direct == nb, "pinned failure: %ld wrong, ring %zu bytes", bad,
           cs.staging_bytes);
    rrc_cuda_backend_destroy(be);
    rrc_fake_fail_host_alloc_after(-1);

    /* sync_each_op: the ring is used and every upload also synchronizes */
    be = make_be_staging(1024, 4, 1);
    rrc_fake_get_stats(&f0);
    bad = upload_burst(be, sizes, nb, &f1);
    EXPECT(bad == 0 && f1.stream_syncs - f0.stream_syncs >= nb, "sync_each_op: %ld wrong, %ld syncs", bad,
           f1.stream_syncs - f0.stream_syncs);
    rrc_cuda_backend_destroy(be);
    rrc_fake_get_stats(&f2);
    EXPECT(f2.h2d_pending == 0 && f2.host_allocs_live == 0, "after destroy: %ld copies pending, %ld pinned blocks live",
           f2.h2d_pending, f2.host_allocs_live);
}

/* Memory counters, GEMM tally and census of the backend. */
static void t_stats_and_census(void) {
    const rrc_backend *be = make_be(0, 0);
    rrc_cuda_stats s;
    OK(rrc_cuda_stats_get(be, &s));
    const size_t internal0 = s.internal_bytes;
    EXPECT(s.alloc_bytes == 0 && internal0 >= ((size_t)32 << 20), "fresh backend: alloc %zu, internal %zu", s.alloc_bytes,
           internal0);
    rrc_buf a, b, c;
    OK(rrc_buf_alloc(be, RRC_F32, 1000, &a));
    OK(rrc_buf_alloc(be, RRC_F32, 24 * 16, &b));
    OK(rrc_buf_alloc(be, RRC_F32, 1000, &c));
    OK(rrc_cuda_stats_get(be, &s));
    EXPECT(s.alloc_bytes == 4 * (2000 + 384) && s.alloc_peak_bytes == s.alloc_bytes && s.n_allocs == 3,
           "alloc %zu peak %zu n %lld", s.alloc_bytes, s.alloc_peak_bytes, (long long)s.n_allocs);
    OK(rrc_cuda_gemm_census_enable(be, 1));
    OK(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, 10, 16, 24, 1.0f, a, 24, b, 16, 0.0f, c, 16));
    OK(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, 10, 16, 24, 1.0f, a, 24, b, 16, 1.0f, c, 16));
    OK(rrc_gemm2d(be, RRC_OP_N, RRC_OP_T, 10, 10, 24, 1.0f, a, 24, a, 24, 0.0f, c, 10));
    OK(rrc_sync(be));
    OK(rrc_cuda_stats_get(be, &s));
    EXPECT(s.gemm_calls[0][0] == 3 && s.gemm_flops[0][0] == 2.0 * (2 * 10 * 16 * 24 + 10 * 10 * 24),
           "tally f32 level 0: %lld calls, %g flops", (long long)s.gemm_calls[0][0], s.gemm_flops[0][0]);
    rrc_cuda_gemm_shape sh[4];
    int64_t dropped = -1;
    size_t ne = rrc_cuda_gemm_census_get(be, sh, 4, &dropped);
    EXPECT(ne == 2 && dropped == 0 && sh[0].calls == 2 && sh[0].m == 10 && sh[0].n == 16 && sh[0].k == 24 &&
               sh[1].calls == 1 && sh[1].trans_b == 1 && sh[0].level == 0 && sh[0].path[0],
           "census: %zu entries, first %lld calls", ne, ne ? (long long)sh[0].calls : -1LL);
    rrc_buf_free(&a);
    OK(rrc_cuda_stats_get(be, &s));
    EXPECT(s.alloc_bytes == 4 * (1000 + 384) && s.alloc_peak_bytes == 4 * (2000 + 384) && s.n_frees == 1,
           "after a free: alloc %zu peak %zu", s.alloc_bytes, s.alloc_peak_bytes);
    OK(rrc_cuda_stats_reset(be));
    OK(rrc_cuda_stats_get(be, &s));
    EXPECT(s.alloc_peak_bytes == s.alloc_bytes && s.gemm_calls[0][0] == 0 && rrc_cuda_gemm_census_get(be, NULL, 0, NULL) == 0,
           "reset: peak %zu, calls %lld", s.alloc_peak_bytes, (long long)s.gemm_calls[0][0]);
    rrc_cuda_timer *tm = NULL;
    OK(rrc_cuda_timer_create(be, 2, &tm));
    float ms = -1.0f;
    OK(rrc_cuda_timer_mark(tm, 0));
    OK(rrc_cuda_timer_mark(tm, 1));
    OK(rrc_cuda_timer_elapsed_ms(tm, 0, 1, &ms));
    EXPECT(ms >= 0.0f, "timer %g ms", (double)ms);
    EXPECT(rrc_cuda_build_is_host_shim() == 1, "the HOST-ONLY build says so");
    rrc_cuda_timer_destroy(tm);
    rrc_buf_free(&b);
    rrc_buf_free(&c);
    rrc_cuda_backend_destroy(be);
}

typedef struct {
    const char *name;
    void (*fn)(void);
} tcase;

int main(int argc, char **argv) {
    const char *filter = argc > 1 ? argv[1] : NULL;
    static const tcase cases[] = {
        {"sum_squares_grow_failure", t_sum_squares_grow_failure},
        {"error_flag_op_and_index", t_error_flag_op_and_index},
        {"reduction_scheme_fallback", t_reduction_scheme_fallback},
        {"h2d_staging_ring", t_h2d_staging_ring},
        {"stats_and_census", t_stats_and_census},
        {"harness_through_backend", t_harness_through_backend},
    };
    int n_run = 0, n_failed = 0;
    printf("cuda_host tests: the real rrc/cuda/*.cu sources on a fake runtime (HOST-ONLY, no GPU)\n");
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        if (filter && !strstr(cases[i].name, filter)) continue;
        const int before = g_fail;
        rrc_clear_error();
        cases[i].fn();
        n_run++;
        if (g_fail != before) n_failed++;
        printf("[%s] cuda_host.%s\n", g_fail != before ? "FAIL" : " OK ", cases[i].name);
    }
    rrc_fake_shutdown();
    printf("\n%d tests, %d failed, %d checks\n", n_run, n_failed, g_checks);
    return n_failed ? 1 : 0;
}
