/* The off-device test backend itself (rrc/offdev.h, DESIGN.md §14): device pointers really fault
 * when the host touches them (forked children), views translate base + offset, memory functions
 * and ops give the inner backend's results, and bad pointers / frees / leaks are reported. The
 * model and trainer runs on it are in test_model.c / test_train.c (and `rrc_tests --offdev`). */
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>

#include "ops_util.h"
#include "rrc/offdev.h"

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define RT_SANITIZED 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define RT_SANITIZED 1
#endif
#endif
#ifndef RT_SANITIZED
#define RT_SANITIZED 0
#endif

static void child_segv(int sig) {
    (void)sig;
    _exit(77);
}

typedef enum { ACC_READ, ACC_WRITE } acc_kind;

/* Runs `*p` (read or write) in a forked child. Returns 1 if the child faulted: killed by SIGSEGV /
 * SIGBUS, or our SIGSEGV handler's exit 77, or (sanitizer builds, whose SEGV handler reports and
 * exits) any non-zero exit. 0 if the access completed (exit 0), -1 if fork failed. */
static int faults(volatile float *p, acc_kind k) {
    fflush(NULL);
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        int nul = open("/dev/null", O_WRONLY);
        if (nul >= 0) dup2(nul, 2);
        struct sigaction sa;
        memset(&sa, 0, sizeof sa);
        sa.sa_handler = child_segv;
        sigaction(SIGSEGV, &sa, NULL);
        sigaction(SIGBUS, &sa, NULL);
        if (k == ACC_READ) {
            float v = *p;
            _exit(v == 12345.0f ? 3 : 0);
        }
        *p = 1.0f;
        _exit(0);
    }
    int ws = 0;
    if (waitpid(pid, &ws, 0) != pid) return -1;
    if (WIFSIGNALED(ws)) return WTERMSIG(ws) == SIGSEGV || WTERMSIG(ws) == SIGBUS;
    if (!WIFEXITED(ws)) return -1;
    if (WEXITSTATUS(ws) == 77) return 1;
    if (WEXITSTATUS(ws) == 0) return 0;
    return RT_SANITIZED ? 1 : -1;
}

RT_TEST(offdev_host_access_faults) {
    const rrc_backend *od = NULL;
    RT_OK(rrc_offdev_backend_create(rrc_backend_cpu(), &od));
    RT_EXPECT(od->host_accessible == 0 && rrc_backend_is_offdev(od) && !rrc_backend_is_offdev(rrc_backend_cpu()),
              "flags");
    RT_EXPECT(!strcmp(od->name, "cpu-offdev"), "%s", od->name);
    rrc_buf b;
    RT_OK(rrc_buf_alloc(od, RRC_F32, 1000, &b));
    float h[1000];
    for (int i = 0; i < 1000; i++) h[i] = (float)i;
    RT_OK(rrc_buf_upload(b, h, 1000));
    /* the control: the same child code on a host pointer completes */
    RT_EXPECT(faults(&h[5], ACC_READ) == 0, "host read in the child did not complete");
    RT_EXPECT(faults(&h[5], ACC_WRITE) == 0, "host write in the child did not complete");
    /* device pointers: base, a view (base + offset), the last element, read and write */
    rrc_buf v = rrc_view(b, 517, 10);
    volatile float *ptrs[3] = {(volatile float *)b.ptr, (volatile float *)v.ptr, (volatile float *)b.ptr + 999};
    for (int i = 0; i < 3; i++) {
        RT_EXPECT(faults(ptrs[i], ACC_READ) == 1, "host read of device pointer %d did not fault", i);
        RT_EXPECT(faults(ptrs[i], ACC_WRITE) == 1, "host write of device pointer %d did not fault", i);
    }
    /* the data is intact and reachable through the backend */
    float g[10];
    RT_OK(rrc_buf_download(v, g, 10));
    RT_EXPECT(g[0] == 517.0f && g[9] == 526.0f, "view download %g %g", (double)g[0], (double)g[9]);
    /* a freed range is never handed out again and still faults */
    volatile float *old = (volatile float *)b.ptr;
    RT_OK(rrc_buf_free(&b));
    RT_OK(rrc_buf_alloc(od, RRC_F32, 1000, &b));
    RT_EXPECT((volatile float *)b.ptr != old, "address reused");
    RT_EXPECT(faults(old, ACC_READ) == 1, "freed device pointer did not fault");
    RT_OK(rrc_buf_free(&b));
    rrc_offdev_stats st;
    RT_OK(rrc_offdev_backend_destroy(od, &st));
    RT_EXPECT(st.allocs == 2 && st.frees == 2 && st.live == 0 && st.peak_bytes == 4000, "stats %lld %lld %lld %lld",
              (long long)st.allocs, (long long)st.frees, (long long)st.live, (long long)st.peak_bytes);
}

/* Memory functions and an op on views at odd offsets give the inner backend's bytes. */
RT_TEST(offdev_views_and_ops_match_inner) {
    const rrc_backend *od = NULL, *cpu = rrc_backend_cpu();
    RT_OK(rrc_offdev_backend_create(cpu, &od));
    float x[301], y[301];
    rt_rng r = {9};
    rt_fill(&r, x, 301, 1.0f);
    rt_fill(&r, y, 301, 1.0f);
    float out[2][301];
    for (int k = 0; k < 2; k++) {
        const rrc_backend *be = k ? od : cpu;
        rrc_buf a = rt_upload(be, RRC_F32, x, 301), b = rt_upload(be, RRC_BF16, y, 301), c;
        RT_OK(rrc_buf_alloc(be, RRC_F32, 301, &c));
        /* views starting at odd elements; an exact in-place alias; d2d copy; zero */
        RT_OK(rrc_add(be, 100, rrc_view(a, 3, 100), rrc_view(b, 7, 100), rrc_view(c, 1, 100)));
        RT_OK(rrc_axpy(be, 50, 0.5f, rrc_view(a, 200, 50), rrc_view(a, 200, 50)));
        RT_OK(rrc_buf_copy(rrc_view(c, 150, 60), rrc_view(a, 190, 60), 60));
        RT_OK(rrc_buf_zero(rrc_view(c, 280, 5), 5));
        rrc_gemm_args g;
        memset(&g, 0, sizeof g);
        g.trans_a = RRC_OP_N;
        g.trans_b = RRC_OP_T;
        g.m = 3;
        g.n = 4;
        g.k = 5;
        g.alpha = 1.0f;
        g.beta = 0.0f;
        g.a = rrc_view(a, 11, 15);
        g.lda = 5;
        g.b = rrc_view(b, 33, 20);
        g.ldb = 5;
        g.c = rrc_view(c, 230, 12);
        g.ldc = 4;
        g.batch = 1;
        RT_OK(rrc_gemm(be, &g));
        RT_OK(rrc_sum_squares(be, 301, a, rrc_view(c, 300, 1), 0));
        RT_OK(rrc_sync(be));
        RT_OK(rrc_buf_download(c, out[k], 301));
        RT_OK(rrc_buf_free(&a));
        RT_OK(rrc_buf_free(&b));
        RT_OK(rrc_buf_free(&c));
    }
    RT_EXPECT(!memcmp(out[0], out[1], sizeof out[0]), "offdev results differ from the inner backend");
    RT_OK(rrc_offdev_backend_destroy(od, NULL));
}

/* Pointers outside live allocations are refused, bad frees and leaks are reported at destroy, an
 * offdev backend cannot wrap another one, zero-byte allocations are NULL. */
RT_TEST(offdev_refusals_and_accounting) {
    const rrc_backend *od = NULL, *od2 = NULL;
    RT_OK(rrc_offdev_backend_create(rrc_backend_cpu_f32acc(), &od));
    RT_EXPECT(!strcmp(od->name, "cpu-f32acc-offdev"), "%s", od->name);
    RT_STATUS(rrc_offdev_backend_create(od, &od2), RRC_ERR_INVALID_ARG);
    void *p = (void *)1;
    RT_OK(od->alloc(od, 0, &p));
    RT_EXPECT(p == NULL, "zero-byte allocation");
    rrc_buf a, b;
    RT_OK(rrc_buf_alloc(od, RRC_F32, 16, &a));
    RT_OK(rrc_buf_alloc(od, RRC_F32, 16, &b));
    /* a hand-made buffer past the end of `a` (the op wrappers only check numel) */
    rrc_buf bad = a;
    bad.owned = 0;
    bad.ptr = (char *)a.ptr + 128; /* inside the page span of a, past its 64 bytes */
    RT_STATUS(rrc_scale(od, 16, 2.0f, bad), RRC_ERR_INVALID_ARG);
    float h[16] = {0};
    /* an extent running past the allocation: the memcpy translation checks base + bytes */
    RT_STATUS(od->memcpy_d2h(od, h, (char *)a.ptr + 8, 64), RRC_ERR_INVALID_ARG);
    RT_STATUS(od->memcpy_h2d(od, (char *)b.ptr + 60, h, 8), RRC_ERR_INVALID_ARG);
    od->free(od, (char *)a.ptr + 4); /* not a base: counted, nothing freed */
    rrc_offdev_stats st;
    RT_OK(rrc_offdev_get_stats(od, &st));
    RT_EXPECT(st.bad_ptrs == 3 && st.bad_frees == 1 && st.live == 2, "stats %lld %lld %lld", (long long)st.bad_ptrs,
              (long long)st.bad_frees, (long long)st.live);
    RT_OK(rrc_buf_free(&a));
    /* b stays live: destroy frees it and reports the leak + the bad pointers */
    RT_STATUS(rrc_offdev_backend_destroy(od, &st), RRC_ERR_INTERNAL);
    RT_EXPECT(strstr(rrc_last_error(), "1 allocations (64 bytes) still live") != NULL, "%s", rrc_last_error());
    RT_EXPECT(st.live == 1 && st.frees == 1, "final stats");
    RT_STATUS(rrc_offdev_get_stats(rrc_backend_cpu(), &st), RRC_ERR_INVALID_ARG);
}

const rt_case rt_suite_offdev[] = {
    RT_CASE(offdev_host_access_faults),
    RT_CASE(offdev_views_and_ops_match_inner),
    RT_CASE(offdev_refusals_and_accounting),
};
const size_t rt_suite_offdev_n = sizeof rt_suite_offdev / sizeof rt_suite_offdev[0];
