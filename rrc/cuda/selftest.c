/* rrc/cuda/selftest.c — rrc_selftest: the first program to run on the GPU (DESIGN.md §11.5).
 *
 *   rrc_selftest [--device N] [--seed S] [--tf32] [--sync-each-op] [--gemm-level L] [--only OP] [--quick] [-v]
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
            "usage: rrc_selftest [--device N] [--seed S] [--tf32] [--sync-each-op] [--gemm-level 0|1|2] [--only OP] [--quick] [-v]\n"
#ifdef RRC_SELFTEST_HOST
            "       (host build) [--backend emu|cpu] [--deferred] [--reject-mask M] [--min-align A] [--scratch-cap N]\n"
#endif
    );
}

int main(int argc, char **argv) {
    int device = 0, tf32 = 0, sync_each = 0, verbose = 0, level = 0, quick = 0;
    uint64_t seed = 0x5E1F7E57ull;
    const char *only = NULL;
#ifdef RRC_SELFTEST_HOST
    const char *backend = "emu";
    rrc_emu_options eo;
    rrc_emu_options_default(&eo);
#endif
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--device") && v) device = atoi(argv[++i]);
        else if (!strcmp(a, "--seed") && v) seed = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(a, "--tf32")) tf32 = 1;
        else if (!strcmp(a, "--sync-each-op")) sync_each = 1;
        else if (!strcmp(a, "--gemm-level") && v) level = atoi(argv[++i]);
        else if (!strcmp(a, "--only") && v) only = argv[++i];
        else if (!strcmp(a, "-v") || !strcmp(a, "--verbose")) verbose = 1;
        else if (!strcmp(a, "--quick")) quick = 1;
#ifdef RRC_SELFTEST_HOST
        else if (!strcmp(a, "--backend") && v) backend = argv[++i];
        else if (!strcmp(a, "--deferred")) eo.deferred_errors = 1;
        else if (!strcmp(a, "--reject-mask") && v) eo.lt_reject_mask = (unsigned)strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--min-align") && v) eo.lt_min_align = (unsigned)strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--scratch-cap") && v) eo.scratch_cap_elems = strtoll(argv[++i], NULL, 0);
#endif
        else {
            usage();
            return 2;
        }
    }
    if (level < 0 || level > 2) {
        usage();
        return 2;
    }
    const rrc_backend *dev = NULL;
    rrc_status s;
#ifdef RRC_SELFTEST_HOST
    (void)device;
    (void)sync_each;
    if (tf32) fprintf(stderr, "note: --tf32 has no effect on host backends\n");
    tf32 = 0;
    eo.gemm_min_level = (rrc_gemm_level)level;
    eo.seed = seed ^ 0xA5A5u;
    if (!strcmp(backend, "cpu")) {
        dev = rrc_backend_cpu();
        s = RRC_OK;
    } else if (!strcmp(backend, "emu")) {
        s = rrc_emu_backend_create(&eo, &dev);
    } else {
        usage();
        return 2;
    }
    if (s != RRC_OK) {
        fprintf(stderr, "rrc_selftest: %s\n", rrc_last_error());
        return 1;
    }
    printf("rrc_selftest (host build): no GPU is used; device backend = '%s'\n", dev->name);
#else
    rrc_cuda_options co;
    rrc_cuda_options_default(&co);
    if (tf32) co.allow_tf32 = 1;
    if (sync_each) co.sync_each_op = 1;
    co.gemm_min_level = level;
    tf32 = co.allow_tf32;
    s = rrc_cuda_backend_create_ex(device, &co, &dev);
    if (s != RRC_OK) {
        fprintf(stderr, "rrc_selftest: cannot create the CUDA backend on device %d: %s\n", device, rrc_last_error());
        return 1;
    }
    rrc_cuda_device_info info;
    if (rrc_cuda_device_info_get(dev, &info) == RRC_OK) {
        printf("device %d: %s, compute capability %d.%d (sm_%d%d), %d SMs\n", info.device, info.name, info.cc_major,
               info.cc_minor, info.cc_major, info.cc_minor, info.sm_count);
        printf("memory: %.2f GiB total, %.2f GiB free\n", (double)info.total_mem / 1073741824.0,
               (double)info.free_mem / 1073741824.0);
        printf("driver API version %d, runtime version %d, cuBLASLt version %zu\n", info.driver_version,
               info.runtime_version, info.cublaslt_version);
    } else {
        printf("device info unavailable: %s\n", rrc_last_error());
    }
    printf("options: tf32 %s, sync_each_op %s, gemm ladder starts at level %d\n", co.allow_tf32 ? "on" : "off",
           co.sync_each_op ? "on" : "off", co.gemm_min_level);
#endif
    rrc_st_opts o;
    memset(&o, 0, sizeof o);
    o.seed = seed;
    o.verbose = verbose;
    o.quick = quick;
    o.tf32 = tf32;
    o.only = only;
    o.gemm_path = path_of;
    rrc_st_result r;
    int fails = rrc_st_run(dev, rrc_backend_cpu_f32acc(), &o, &r);
#ifdef RRC_SELFTEST_HOST
    rrc_emu_backend_destroy(dev);
#else
    rrc_cuda_backend_destroy(dev);
#endif
    printf("rrc_selftest: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
