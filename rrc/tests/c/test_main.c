/* rrc/tests/c/test_main.c — runs every suite.
 *
 *   rrc_tests [--offdev] [substring ...]
 *
 * With substrings, only the tests whose "suite.name" contains one of them run. --offdev (TEST
 * ONLY, DESIGN.md §14) makes rt_cpu() / rt_cpu_f32acc() (ops_util.h) return off-device wrappers of
 * the CPU backends (rrc/offdev.h): device pointers are not host-mapped, so any host access to
 * backend memory in the library or the tests faults. The wrappers are destroyed at exit; a live
 * allocation, a bad free or a refused translation is a failure, and so is a run that allocated
 * nothing through them (the mode would then have proved nothing). */
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ops_util.h"
#include "rrc/offdev.h"
#include "rt.h"

#define RT_SUITE(s)                                                                             \
    extern const rt_case rt_suite_##s[];                                                        \
    extern const size_t rt_suite_##s##_n;
RT_SUITE(common)
RT_SUITE(json)
RT_SUITE(safetensors)
RT_SUITE(config)
RT_SUITE(kernels)
RT_SUITE(ops)
RT_SUITE(grad)
RT_SUITE(contracts)
RT_SUITE(model)
RT_SUITE(data)
RT_SUITE(train)
RT_SUITE(gemm_plan)
RT_SUITE(kc_items)
RT_SUITE(offdev)

typedef struct {
    const char *name;
    const rt_case *cases;
    const size_t *n;
} rt_suite;

static const rt_suite k_suites[] = {
    {"common", rt_suite_common, &rt_suite_common_n},
    {"json", rt_suite_json, &rt_suite_json_n},
    {"safetensors", rt_suite_safetensors, &rt_suite_safetensors_n},
    {"config", rt_suite_config, &rt_suite_config_n},
    {"kernels", rt_suite_kernels, &rt_suite_kernels_n},
    {"ops", rt_suite_ops, &rt_suite_ops_n},
    {"grad", rt_suite_grad, &rt_suite_grad_n},
    {"contracts", rt_suite_contracts, &rt_suite_contracts_n},
    {"model", rt_suite_model, &rt_suite_model_n},
    {"data", rt_suite_data, &rt_suite_data_n},
    {"train", rt_suite_train, &rt_suite_train_n},
    {"gemm_plan", rt_suite_gemm_plan, &rt_suite_gemm_plan_n},
    {"kc_items", rt_suite_kc_items, &rt_suite_kc_items_n},
    {"offdev", rt_suite_offdev, &rt_suite_offdev_n},
};

static int g_case_failures = 0;
static long g_checks = 0;
static int g_skipped = 0;
static char g_tmpdir[1024];

void rt_note_check(void) { g_checks++; }

void rt_fail(const char *file, int line, const char *expr, const char *fmt, ...) {
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    if (g_case_failures < 20)
        fprintf(stderr, "    FAILED %s:%d: (%s) %s\n", file, line, expr, msg);
    else if (g_case_failures == 20)
        fprintf(stderr, "    ... further failures in this test suppressed\n");
    g_case_failures++;
}

void rt_skip(const char *why) {
    fprintf(stderr, "    SKIP: %s\n", why);
    g_skipped++;
}

const char *rt_tmpdir(void) { return g_tmpdir; }

const char *rt_path(const char *name) {
    static char slots[8][1200];
    static int next = 0;
    char *p = slots[next];
    next = (next + 1) % 8;
    snprintf(p, sizeof slots[0], "%s/%s", g_tmpdir, name);
    return p;
}

static int g_offdev = 0;
static const rrc_backend *g_od[2];

int rt_offdev_mode(void) { return g_offdev; }

static const rrc_backend *od_get(int i) {
    if (!g_od[i] && rrc_offdev_backend_create(i ? rrc_backend_cpu_f32acc() : rrc_backend_cpu(), &g_od[i]) != RRC_OK) {
        fprintf(stderr, "rrc_tests --offdev: %s\n", rrc_last_error());
        exit(2); /* never fall back to host pointers */
    }
    return g_od[i];
}
const rrc_backend *rt_cpu(void) { return g_offdev ? od_get(0) : rrc_backend_cpu(); }
const rrc_backend *rt_cpu_f32acc(void) { return g_offdev ? od_get(1) : rrc_backend_cpu_f32acc(); }

/* Destroys the wrappers; returns the number of problems found. */
static int offdev_finish(void) {
    int bad = 0;
    int64_t allocs = 0;
    for (int i = 0; i < 2; i++) {
        if (!g_od[i]) continue;
        const char *name = i ? "cpu-f32acc-offdev" : "cpu-offdev";
        rrc_offdev_stats st;
        memset(&st, 0, sizeof st);
        if (rrc_offdev_backend_destroy(g_od[i], &st) != RRC_OK) {
            fprintf(stderr, "offdev: %s\n", rrc_last_error());
            bad++;
        }
        g_od[i] = NULL;
        allocs += st.allocs;
        fprintf(stderr, "offdev: %s: %lld allocations, %lld frees, peak %lld bytes, %lld bad frees, %lld refused "
                        "translations\n", name, (long long)st.allocs, (long long)st.frees, (long long)st.peak_bytes,
                (long long)st.bad_frees, (long long)st.bad_ptrs);
    }
    if (allocs == 0) {
        fprintf(stderr, "offdev: no allocation went through the off-device backends: nothing was checked\n");
        bad++;
    }
    return bad;
}

static void setup_tmpdir(void) {
    const char *env = getenv("RRC_TEST_TMPDIR");
    if (env && env[0]) {
        snprintf(g_tmpdir, sizeof g_tmpdir, "%s", env);
        mkdir(g_tmpdir, 0755);
        return;
    }
    snprintf(g_tmpdir, sizeof g_tmpdir, "/tmp/rrc_tests_XXXXXX");
    if (!mkdtemp(g_tmpdir)) {
        perror("mkdtemp");
        exit(2);
    }
}

static int matches(const char *full, char **filters, int n_filters) {
    if (n_filters == 0) return 1;
    for (int i = 0; i < n_filters; i++)
        if (strstr(full, filters[i])) return 1;
    return 0;
}

int main(int argc, char **argv) {
    int first = 1;
    if (argc > 1 && !strcmp(argv[1], "--offdev")) {
        g_offdev = 1;
        first = 2;
    }
    char **filters = argv + first;
    const int n_filters = argc - first;
    setup_tmpdir();
    int n_run = 0, n_failed = 0;
    for (size_t s = 0; s < sizeof k_suites / sizeof k_suites[0]; s++) {
        const rt_suite *suite = &k_suites[s];
        for (size_t i = 0; i < *suite->n; i++) {
            const rt_case *c = &suite->cases[i];
            char full[256];
            snprintf(full, sizeof full, "%s.%s", suite->name, c->name);
            if (!matches(full, filters, n_filters)) continue;
            g_case_failures = 0;
            int skipped_before = g_skipped;
            rrc_clear_error();
            c->fn();
            n_run++;
            if (g_case_failures) {
                n_failed++;
                fprintf(stderr, "[FAIL] %s (%d failed checks)\n", full, g_case_failures);
            } else {
                fprintf(stderr, "[ OK ] %s%s\n", full, g_skipped > skipped_before ? " (partly skipped)" : "");
            }
        }
    }
    fprintf(stderr, "\n%d tests, %d failed, %ld checks, %d skipped sub-tests\n", n_run, n_failed, g_checks,
            g_skipped);
    if (n_run == 0) {
        fprintf(stderr, "no test matched the filter\n");
        return 1;
    }
    if (g_offdev) {
        const int bad = offdev_finish();
        fprintf(stderr, "offdev mode (device pointers not host-mapped): %s\n", bad ? "FAILED" : "OK");
        if (bad) return 1;
    }
    return n_failed ? 1 : 0;
}
