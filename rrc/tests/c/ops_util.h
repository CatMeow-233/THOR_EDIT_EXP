/* Helpers shared by the op tests: deterministic RNG, host<->backend buffers, comparisons. */
#ifndef RRC_TESTS_OPS_UTIL_H
#define RRC_TESTS_OPS_UTIL_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "rrc/backend.h"
#include "rt.h"

typedef struct {
    uint64_t s;
} rt_rng;

static inline uint64_t rt_next(rt_rng *r) {
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
/* uniform in [-1, 1) */
static inline float rt_uniform(rt_rng *r) { return (float)((double)(rt_next(r) >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0); }
static inline void rt_fill(rt_rng *r, float *x, int64_t n, float scale) {
    for (int64_t i = 0; i < n; i++) x[i] = rt_uniform(r) * scale;
}

/* Upload a host f32 array into a new backend buffer of dtype dt (rounded to bf16 if needed).
 * If `rounded` is non-NULL it receives the values actually stored (as doubles). */
static inline rrc_buf rt_upload(const rrc_backend *be, rrc_dtype dt, const float *x, int64_t n) {
    rrc_buf b;
    if (rrc_buf_alloc(be, dt, n, &b) != RRC_OK) return rrc_buf_none();
    if (n == 0) return b;
    if (dt == RRC_F32) {
        rrc_buf_upload(b, x, n);
    } else if (dt == RRC_BF16) {
        rrc_bf16 *t = (rrc_bf16 *)malloc((size_t)n * 2);
        for (int64_t i = 0; i < n; i++) t[i] = rrc_f32_to_bf16(x[i]);
        rrc_buf_upload(b, t, n);
        free(t);
    }
    return b;
}
static inline rrc_buf rt_upload_i32(const rrc_backend *be, const int32_t *x, int64_t n) {
    rrc_buf b;
    if (rrc_buf_alloc(be, RRC_I32, n, &b) != RRC_OK) return rrc_buf_none();
    if (n) rrc_buf_upload(b, x, n);
    return b;
}
/* Download any float buffer as f32. Returns malloc'd array. */
static inline float *rt_download(rrc_buf b, int64_t n) {
    float *out = (float *)malloc((size_t)(n > 0 ? n : 1) * sizeof(float));
    if (b.dtype == RRC_F32) {
        rrc_buf_download(b, out, n);
    } else {
        rrc_bf16 *t = (rrc_bf16 *)malloc((size_t)(n > 0 ? n : 1) * 2);
        rrc_buf_download(b, t, n);
        for (int64_t i = 0; i < n; i++) out[i] = rrc_bf16_to_f32(t[i]);
        free(t);
    }
    return out;
}
/* Values as stored in dtype dt. */
static inline void rt_round_inplace(float *x, int64_t n, rrc_dtype dt) {
    if (dt == RRC_BF16)
        for (int64_t i = 0; i < n; i++) x[i] = rrc_round_bf16(x[i]);
}

/* Max of |got - want| / (atol + rtol*|want|); <= 1 passes. Reports the first offending index. */
static inline double rt_max_err(const float *got, const double *want, int64_t n, double atol, double rtol,
                                int64_t *worst) {
    double m = 0;
    *worst = -1;
    for (int64_t i = 0; i < n; i++) {
        double d = fabs((double)got[i] - want[i]);
        double e = d == 0.0 ? 0.0 : d / (atol + rtol * fabs(want[i]));
        if (!(e <= m)) { /* also catches NaN */
            m = isnan(e) ? INFINITY : e;
            *worst = i;
        }
    }
    return m;
}

#define RT_ALLCLOSE(got, want, n, atol, rtol, label)                                                          \
    do {                                                                                                    \
        int64_t rt_w_;                                                                                      \
        double rt_e_ = rt_max_err((got), (want), (n), (atol), (rtol), &rt_w_);                              \
        RT_EXPECT(rt_e_ <= 1.0, "%s: max scaled error %.3g at %lld (got %.9g want %.9g)", (const char *)(label), \
                  rt_e_, (long long)rt_w_, rt_w_ >= 0 ? (double)(got)[rt_w_] : 0.0,                         \
                  rt_w_ >= 0 ? (want)[rt_w_] : 0.0);                                                        \
    } while (0)

static inline double *rt_dalloc(int64_t n) { return (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double)); }

/* Both CPU backends (double and float accumulation). */
/* The CPU backends the tests run on: rrc_backend_cpu() / rrc_backend_cpu_f32acc(), or with
 * `rrc_tests --offdev` their off-device wrappers (rrc/offdev.h, DESIGN.md §14), created once by the
 * runner and checked for leaks at exit (test_main.c). */
int rt_offdev_mode(void);
const rrc_backend *rt_cpu(void);
const rrc_backend *rt_cpu_f32acc(void);
static inline const rrc_backend *rt_backend(int i) { return i == 0 ? rt_cpu() : rt_cpu_f32acc(); }

#endif
