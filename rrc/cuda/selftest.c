/* rrc/cuda/selftest.c — rrc_selftest: the first program to run on the GPU (DESIGN.md §11.5).
 *
 *   rrc_selftest [--device N] [--seed S] [--tf32] [--sync-each-op] [--gemm-level L] [--only OP] [--quick]
 *                [--scratch-cap N] [--no-capped-pass] [--allow-fallback] [-v]
 *
 * Prints the device, driver and runtime, then checks every op of the CUDA backend against
 * rrc_backend_cpu_f32acc on random inputs (selftest_ops.c), the GEMM path taken per dtype
 * combination, range-error reporting and run-to-run determinism. Exit status 0 only if every
 * check passes. It measures and prints no timing or throughput.
 *
 * Built with -DRRC_SELFTEST_HOST it becomes rrc_selftest_host, which runs the same checks with
 * the host emulation backend (kc_emu.c, `--backend emu`, default) or the CPU double-accumulation
 * backend (`--backend cpu`) as the device: this is how `make test` executes the harness itself
 * without a GPU. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rrc/rrc.h"
#include "selftest_ops.h"

#ifdef RRC_SELFTEST_HOST
#include "kc_emu.h"
#else
#include "rrc_cuda.h"
#endif

static int path_of(const rrc_backend *be, char *buf, size_t n) {
    rrc_gemm_exec_report rep;
#ifdef RRC_SELFTEST_HOST
    if (!rrc_emu_last_gemm(be, &rep)) return 0;
#else
    if (!rrc_cuda_last_gemm(be, &rep)) return 0;
#endif
    snprintf(buf, n, "%s", rep.desc);
    return 1;
}

static void usage(void) {
    fprintf(stderr,
            "usage: rrc_selftest [--device N] [--seed S] [--tf32] [--sync-each-op] [--gemm-level 0|1|2] [--only OP] [--quick]\n"
            "                    [--scratch-cap N] [--no-capped-pass] [--allow-fallback] [-v]\n"
#ifndef RRC_SELFTEST_HOST
            "                    [--max-grid N] [--no-grid-pass]\n"
#endif
#ifdef RRC_SELFTEST_HOST
            "       (host build) [--backend emu|cpu] [--deferred] [--reject-mask M] [--min-align A]\n"
#endif
    );
}

typedef struct cfg {
    int device, tf32, sync_each, level;
    int64_t scratch_cap; /* 0: the backend default */
    unsigned max_grid;   /* CUDA: rrc_cuda_options.max_grid_blocks of the first pass (0: default) */
#ifdef RRC_SELFTEST_HOST
    const char *backend;
    rrc_emu_options eo;
#endif
} cfg;

/* Creates the device backend for one pass; prints its description once (`describe`). */
static rrc_status make_dev(const cfg *c, int64_t scratch_cap, unsigned max_grid, int describe, const rrc_backend **dev,
                           int *tf32_out) {
#ifdef RRC_SELFTEST_HOST
    (void)max_grid;
    *tf32_out = 0;
    if (!strcmp(c->backend, "cpu")) {
        *dev = rrc_backend_cpu();
        if (describe) printf("rrc_selftest (host build): no GPU is used; device backend = '%s'\n", (*dev)->name);
        return RRC_OK;
    }
    rrc_emu_options eo = c->eo;
    eo.gemm_min_level = (rrc_gemm_level)c->level;
    if (scratch_cap > 0) eo.scratch_cap_elems = scratch_cap;
    RRC_CHECK(rrc_emu_backend_create(&eo, dev));
    if (describe) printf("rrc_selftest (host build): no GPU is used; device backend = '%s'\n", (*dev)->name);
    return RRC_OK;
#else
    rrc_cuda_options co;
    rrc_cuda_options_default(&co);
    if (c->tf32) co.allow_tf32 = 1;
    if (c->sync_each) co.sync_each_op = 1;
    co.gemm_min_level = c->level;
    if (scratch_cap > 0) co.gemm_scratch_cap_elems = scratch_cap;
    co.max_grid_blocks = max_grid;
    *tf32_out = co.allow_tf32;
    rrc_status s = rrc_cuda_backend_create_ex(c->device, &co, dev);
    if (s != RRC_OK) return s;
    if (!describe) return RRC_OK;
    rrc_cuda_device_info info;
    if (rrc_cuda_device_info_get(*dev, &info) == RRC_OK) {
        printf("device %d: %s, compute capability %d.%d (sm_%d%d), %d SMs\n", info.device, info.name, info.cc_major,
               info.cc_minor, info.cc_major, info.cc_minor, info.sm_count);
        printf("memory: %.2f GiB total, %.2f GiB free\n", (double)info.total_mem / 1073741824.0,
               (double)info.free_mem / 1073741824.0);
        printf("driver API version %d, runtime version %d, cuBLASLt version %zu\n", info.driver_version,
               info.runtime_version, info.cublaslt_version);
    } else {
        printf("device info unavailable: %s\n", rrc_last_error());
        rrc_clear_error();
    }
    printf("options: tf32 %s, sync_each_op %s, gemm ladder starts at level %d, gemm scratch cap %lld floats, max grid %u "
           "blocks%s\n",
           co.allow_tf32 ? "on" : "off", co.sync_each_op ? "on" : "off", co.gemm_min_level,
           (long long)co.gemm_scratch_cap_elems, co.max_grid_blocks, co.max_grid_blocks ? "" : " (default)");
    return RRC_OK;
#endif
}

static void destroy_dev(const rrc_backend *dev) {
#ifdef RRC_SELFTEST_HOST
    rrc_emu_backend_destroy(dev); /* no-op for the CPU backend */
#else
    rrc_cuda_backend_destroy(dev);
#endif
}

/* Scratch cap of the extra GEMM pass: small enough that most promoted cases run in several
 * chunks (per-chunk pointer offsets, alignment and descriptor-cache keys, scratch reuse across
 * chunks on one stream, the post-conversion into a bf16 C). */
#define CAPPED_PASS_SCRATCH 1024
/* Grid cap of the extra CUDA pass: every element kernel's grid-stride loop and every row kernel's
 * row-stride loop runs several iterations per block even at the selftest's small sizes (with the
 * default cap of 2^20 blocks that needs more than 2^28 elements or 2^20 rows). */
#define GRID_PASS_BLOCKS 3u

int main(int argc, char **argv) {
    cfg c;
    memset(&c, 0, sizeof c);
    int verbose = 0, quick = 0, allow_fallback = 0, capped_pass = 1, grid_pass = 1;
    uint64_t seed = 0x5E1F7E57ull;
    const char *only = NULL;
#ifdef RRC_SELFTEST_HOST
    c.backend = "emu";
    rrc_emu_options_default(&c.eo);
#endif
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--device") && v) c.device = atoi(argv[++i]);
        else if (!strcmp(a, "--seed") && v) seed = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(a, "--tf32")) c.tf32 = 1;
        else if (!strcmp(a, "--sync-each-op")) c.sync_each = 1;
        else if (!strcmp(a, "--gemm-level") && v) c.level = atoi(argv[++i]);
        else if (!strcmp(a, "--only") && v) only = argv[++i];
        else if (!strcmp(a, "-v") || !strcmp(a, "--verbose")) verbose = 1;
        else if (!strcmp(a, "--quick")) quick = 1;
        else if (!strcmp(a, "--scratch-cap") && v) c.scratch_cap = strtoll(argv[++i], NULL, 0);
        else if (!strcmp(a, "--no-capped-pass")) capped_pass = 0;
        else if (!strcmp(a, "--allow-fallback")) allow_fallback = 1;
#ifndef RRC_SELFTEST_HOST
        else if (!strcmp(a, "--max-grid") && v) c.max_grid = (unsigned)strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--no-grid-pass")) grid_pass = 0;
#endif
#ifdef RRC_SELFTEST_HOST
        else if (!strcmp(a, "--backend") && v) c.backend = argv[++i];
        else if (!strcmp(a, "--deferred")) c.eo.deferred_errors = 1;
        else if (!strcmp(a, "--reject-mask") && v) c.eo.lt_reject_mask = (unsigned)strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--min-align") && v) c.eo.lt_min_align = (unsigned)strtoul(argv[++i], NULL, 0);
#endif
        else {
            usage();
            return 2;
        }
    }
    if (c.level < 0 || c.level > 2 || c.scratch_cap < 0) {
        usage();
        return 2;
    }
#ifdef RRC_SELFTEST_HOST
    if (strcmp(c.backend, "cpu") && strcmp(c.backend, "emu")) {
        usage();
        return 2;
    }
    if (c.tf32) fprintf(stderr, "note: --tf32 has no effect on host backends\n");
    c.tf32 = 0;
    c.eo.seed = seed ^ 0xA5A5u;
    const int strict = !strcmp(c.backend, "emu"); /* the emulator uses the CPU's arithmetic */
    const int has_paths = !strcmp(c.backend, "emu");
    grid_pass = 0; /* host backends have no grid */
#else
    const int strict = 0, has_paths = 1;
    if (c.max_grid) grid_pass = 0; /* the first pass already runs with a grid cap */
#endif
    const rrc_backend *dev = NULL;
    int tf32 = 0;
    rrc_status s = make_dev(&c, c.scratch_cap, c.max_grid, 1, &dev, &tf32);
    if (s != RRC_OK) {
        fprintf(stderr, "rrc_selftest: cannot create the device backend (device %d): %s\n", c.device, rrc_last_error());
        return 1;
    }
    rrc_st_opts o;
    memset(&o, 0, sizeof o);
    o.seed = seed;
    o.verbose = verbose;
    o.quick = quick;
    o.tf32 = tf32;
    o.only = only;
    o.gemm_path = path_of;
    o.strict = strict;
    /* the ladder must reach cuBLASLt unless it is forced past level 0 */
    o.require_lt = has_paths && c.level == 0 && !allow_fallback;
    rrc_st_result r;
    int fails = rrc_st_run(dev, rrc_backend_cpu_f32acc(), &o, &r);
    destroy_dev(dev);
    /* second pass: every GEMM case again with a small scratch cap (multi-chunk paths), unless the
     * cap was given explicitly, gemm is filtered out, or the backend has no scratch */
    const int want_gemm = !only || strstr("gemm", only) != NULL; /* the harness filter: op name contains `only` */
    if (capped_pass && c.scratch_cap == 0 && want_gemm && c.level < 2 && has_paths) {
        printf("capped pass: gemm with a %d-float scratch cap (multi-chunk paths)\n", CAPPED_PASS_SCRATCH);
        s = make_dev(&c, CAPPED_PASS_SCRATCH, c.max_grid, 0, &dev, &tf32);
        if (s != RRC_OK) {
            fprintf(stderr, "rrc_selftest: cannot create the capped-pass backend: %s\n", rrc_last_error());
            return 1;
        }
        o.only = "gemm";
        /* tiny chunks may legitimately lose the operand alignment cuBLASLt needs: the path
         * requirement is checked by the first pass; this one must chunk */
        o.require_lt = 0;
        o.require_chunked = !allow_fallback; /* everything on the kernel level never chunks */
        fails += rrc_st_run(dev, rrc_backend_cpu_f32acc(), &o, &r);
        destroy_dev(dev);
    }
    /* third pass (CUDA): every op again with a grid of at most GRID_PASS_BLOCKS blocks */
    if (grid_pass) {
        printf("grid pass: every op with at most %u blocks per launch (grid-stride / row-stride wrap)\n", GRID_PASS_BLOCKS);
        s = make_dev(&c, c.scratch_cap, GRID_PASS_BLOCKS, 0, &dev, &tf32);
        if (s != RRC_OK) {
            fprintf(stderr, "rrc_selftest: cannot create the grid-pass backend: %s\n", rrc_last_error());
            return 1;
        }
        o.only = only;
        o.require_lt = has_paths && c.level == 0 && !allow_fallback;
        o.require_chunked = 0;
        fails += rrc_st_run(dev, rrc_backend_cpu_f32acc(), &o, &r);
        destroy_dev(dev);
    }
    printf("rrc_selftest: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
