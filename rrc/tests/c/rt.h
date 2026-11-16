/* rrc/tests/c/rt.h — a tiny test runner.
 *
 *   RT_TEST(name) { RT_EXPECT(cond, "fmt", ...); RT_OK(rrc_call(...)); }
 *
 * A test function is registered in its file's suite table (see test_main.c). RT_EXPECT records a
 * failure and continues; RT_ASSERT records a failure and returns from the test function. Every
 * failure prints file:line, the failed expression and a formatted explanation.
 */
#ifndef RRC_TESTS_RT_H
#define RRC_TESTS_RT_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "rrc/common.h"

typedef void (*rt_fn)(void);
typedef struct rt_case {
    const char *name;
    rt_fn fn;
} rt_case;

void rt_fail(const char *file, int line, const char *expr, const char *fmt, ...) RRC_PRINTF(4, 5);
void rt_note_check(void);
void rt_skip(const char *why);
/* Scratch directory for files written by tests (created by the runner). */
const char *rt_tmpdir(void);
/* Path inside rt_tmpdir(); returns a static buffer rotated among 8 slots. */
const char *rt_path(const char *name);

#define RT_TEST(name) static void name(void)
#define RT_CASE(name) {#name, name}

#define RT_EXPECT(cond, ...)                                                                    \
    do {                                                                                        \
        rt_note_check();                                                                        \
        if (!(cond)) rt_fail(__FILE__, __LINE__, #cond, __VA_ARGS__);                           \
    } while (0)

#define RT_ASSERT(cond, ...)                                                                    \
    do {                                                                                        \
        rt_note_check();                                                                        \
        if (!(cond)) {                                                                          \
            rt_fail(__FILE__, __LINE__, #cond, __VA_ARGS__);                                    \
            return;                                                                             \
        }                                                                                       \
    } while (0)

/* Expect an rrc call to succeed; on failure print the thread's last error and return. */
#define RT_OK(expr)                                                                             \
    do {                                                                                        \
        rt_note_check();                                                                        \
        rrc_status rt_s_ = (expr);                                                              \
        if (rt_s_ != RRC_OK) {                                                                  \
            rt_fail(__FILE__, __LINE__, #expr, "returned %s: %s", rrc_status_str(rt_s_),       \
                    rrc_last_error());                                                          \
            return;                                                                             \
        }                                                                                       \
    } while (0)

/* Expect an rrc call to fail with exactly `want`; the error message must be non-empty. */
#define RT_STATUS(expr, want)                                                                   \
    do {                                                                                        \
        rt_note_check();                                                                        \
        rrc_clear_error();                                                                      \
        rrc_status rt_s_ = (expr);                                                              \
        if (rt_s_ != (want))                                                                    \
            rt_fail(__FILE__, __LINE__, #expr, "returned %s, expected %s (msg: %s)",            \
                    rrc_status_str(rt_s_), rrc_status_str(want), rrc_last_error());             \
        else if ((want) != RRC_OK && rrc_last_error()[0] == '\0')                               \
            rt_fail(__FILE__, __LINE__, #expr, "failed with %s but set no error message",       \
                    rrc_status_str(rt_s_));                                                     \
    } while (0)

/* |got - want| <= atol + rtol * |want| (NaN never passes unless both NaN). */
static inline int rt_close(double got, double want, double atol, double rtol) {
    if (isnan(got) || isnan(want)) return isnan(got) && isnan(want);
    if (isinf(got) || isinf(want)) return got == want;
    return fabs(got - want) <= atol + rtol * fabs(want);
}

#define RT_CLOSE(got, want, atol, rtol)                                                         \
    do {                                                                                        \
        double rt_g_ = (double)(got), rt_w_ = (double)(want);                                   \
        RT_EXPECT(rt_close(rt_g_, rt_w_, (atol), (rtol)), "got %.9g want %.9g (diff %.3g)",    \
                  rt_g_, rt_w_, rt_g_ - rt_w_);                                                 \
    } while (0)

#endif
