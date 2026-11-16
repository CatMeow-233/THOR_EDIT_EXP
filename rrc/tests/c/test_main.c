/* rrc/tests/c/test_main.c — runs every suite; `rrc_tests [substring]` runs matching tests only. */
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

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

int main(int argc, char **argv) {
    const char *filter = argc > 1 ? argv[1] : NULL;
    setup_tmpdir();
    int n_run = 0, n_failed = 0;
    for (size_t s = 0; s < sizeof k_suites / sizeof k_suites[0]; s++) {
        const rt_suite *suite = &k_suites[s];
        for (size_t i = 0; i < *suite->n; i++) {
            const rt_case *c = &suite->cases[i];
            char full[256];
            snprintf(full, sizeof full, "%s.%s", suite->name, c->name);
            if (filter && !strstr(full, filter)) continue;
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
        fprintf(stderr, "no test matched filter '%s'\n", filter ? filter : "");
        return 1;
    }
    return n_failed ? 1 : 0;
}
