/* Interface-contract tests of the op table (include/rrc/backend.h): aliasing of column blocks of
 * one fused buffer, gemm with k == 0 and the int32 limits (checked with a recording backend that
 * wraps the CPU backend), the multi-call patterns the model code relies on (GQA dK/dV batched
 * over KV groups, LM-head row gather/scatter), the CPU gemm's bounded packing scratch, and cross
 * entropy with -inf logits. */
#include <stdatomic.h>

#include "ops_util.h"
#include "rrc_kc.h"

static const rrc_dtype k_fdt[2] = {RRC_F32, RRC_BF16};
static double rtol_of(rrc_dtype dt) { return dt == RRC_BF16 ? 1.0 / 128.0 : 2e-5; }

/* ---- malloc accounting (the test binary is linked with -Wl,--wrap=malloc) ------------------- */

void *__real_malloc(size_t size);
void *__wrap_malloc(size_t size);
static atomic_int g_track_malloc;
static atomic_size_t g_max_malloc;

void *__wrap_malloc(size_t size) {
    if (atomic_load(&g_track_malloc)) {
        size_t cur = atomic_load(&g_max_malloc);
        while (size > cur && !atomic_compare_exchange_weak(&g_max_malloc, &cur, size)) {
        }
    }
    return __real_malloc(size);
}

/* ---- recording backend: the CPU table with gemm replaced ------------------------------------ */

static rrc_backend g_rec;
static int g_rec_forward; /* call the CPU gemm (real buffers) or only record (fabricated ones) */
static int64_t g_rec_calls, g_rec_k0_calls, g_rec_last_batch, g_rec_last_m, g_rec_last_n, g_rec_last_k;

static rrc_status rec_gemm(const rrc_backend *be, const rrc_gemm_args *g) {
    (void)be;
    g_rec_calls++;
    if (g->k == 0) g_rec_k0_calls++;
    g_rec_last_batch = g->batch;
    g_rec_last_m = g->m;
    g_rec_last_n = g->n;
    g_rec_last_k = g->k;
    if (!g_rec_forward) return RRC_OK;
    /* run the CPU reference on the same memory (its buffers carry the CPU backend) */
    rrc_gemm_args c = *g;
    c.a.be = c.b.be = c.c.be = rrc_backend_cpu();
    return rrc_backend_cpu()->gemm(rrc_backend_cpu(), &c);
}

static const rrc_backend *rec_backend(int forward) {
    g_rec = *rrc_backend_cpu();
    g_rec.name = "recording";
    g_rec.gemm = rec_gemm;
    g_rec_forward = forward;
    g_rec_calls = g_rec_k0_calls = 0;
    g_rec_last_batch = g_rec_last_m = g_rec_last_n = g_rec_last_k = -1;
    return &g_rec;
}

/* A buffer descriptor that is never dereferenced (recording backend without forwarding). */
static rrc_buf fake_buf(const rrc_backend *be, uint64_t addr, int64_t numel) {
    rrc_buf b = rrc_buf_none();
    b.be = be;
    b.dtype = RRC_F32;
    b.numel = numel;
    b.ptr = (void *)(uintptr_t)addr;
    return b;
}

/* ---- column blocks of one fused buffer -------------------------------------------------------- */

static void ref_swiglu_bwd(double g, double u, double dh, double *dg, double *du) {
    double s = 1.0 / (1.0 + exp(-g));
    *du = dh * g * s;
    *dg = dh * u * s * (1.0 + g * (1.0 - s));
}

RT_TEST(swiglu_fused_column_blocks) {
    rt_rng rng = {101};
    const int64_t N = 5, I = 7, L = 2 * I;
    for (int bi = 0; bi < 2; bi++)
        for (int dt = 0; dt < 2; dt++) {
            const rrc_backend *be = rt_backend(bi);
            const rrc_dtype d = k_fdt[dt];
            float gu[5 * 14], dh[5 * 7];
            rt_fill(&rng, gu, N * L, 4.0f);
            rt_fill(&rng, dh, N * I, 2.0f);
            rt_round_inplace(gu, N * L, d);
            rt_round_inplace(dh, N * I, d);
            double wg[35], wu[35], wh[35];
            for (int64_t r = 0; r < N; r++)
                for (int64_t c = 0; c < I; c++) {
                    double g = gu[r * L + c], u = gu[r * L + I + c];
                    ref_swiglu_bwd(g, u, dh[r * I + c], &wg[r * I + c], &wu[r * I + c]);
                    wh[r * I + c] = g / (1.0 + exp(-g)) * u;
                }
            rrc_buf dhb = rt_upload(be, d, dh, N * I);

            /* (1) fused inputs, outputs as the two column blocks of one fused dgu [N, 2I] */
            rrc_buf in = rt_upload(be, d, gu, N * L), dgu;
            RT_OK(rrc_buf_alloc(be, d, N * L, &dgu));
            RT_OK(rrc_swiglu_bwd(be, N, I, in, rrc_view_from(in, I), L, dhb, I, dgu, rrc_view_from(dgu, I), L));
            float *got = rt_download(dgu, N * L);
            double *want = rt_dalloc(N * L);
            for (int64_t r = 0; r < N; r++)
                for (int64_t c = 0; c < I; c++) {
                    want[r * L + c] = wg[r * I + c];
                    want[r * L + I + c] = wu[r * I + c];
                }
            RT_ALLCLOSE(got, want, N * L, 1e-6, rtol_of(d), "swiglu_bwd fused out");
            free(got);

            /* (2) in place on the fused buffer: dgate = gate, dup = up */
            RT_OK(rrc_swiglu_bwd(be, N, I, in, rrc_view_from(in, I), L, dhb, I, in, rrc_view_from(in, I), L));
            got = rt_download(in, N * L);
            RT_ALLCLOSE(got, want, N * L, 1e-6, rtol_of(d), "swiglu_bwd in place on fused buffer");
            free(got);

            /* (3) dgate = gate (exact alias), dup in a separate [N, 2I] buffer with the same ld */
            rrc_buf_free(&in);
            in = rt_upload(be, d, gu, N * L);
            rrc_buf sep;
            RT_OK(rrc_buf_alloc(be, d, N * L, &sep));
            RT_OK(rrc_swiglu_bwd(be, N, I, in, rrc_view_from(in, I), L, dhb, I, in, sep, L));
            float *g1 = rt_download(in, N * L), *g2 = rt_download(sep, N * L);
            double *wg1 = rt_dalloc(N * I), *wu1 = rt_dalloc(N * I);
            float *gg = (float *)malloc(sizeof(float) * N * I), *uu = (float *)malloc(sizeof(float) * N * I);
            for (int64_t r = 0; r < N; r++)
                for (int64_t c = 0; c < I; c++) {
                    gg[r * I + c] = g1[r * L + c];
                    uu[r * I + c] = g2[r * L + c];
                    wg1[r * I + c] = wg[r * I + c];
                    wu1[r * I + c] = wu[r * I + c];
                    RT_EXPECT(g1[r * L + I + c] == gu[r * L + I + c], "up block must be untouched");
                }
            RT_ALLCLOSE(gg, wg1, N * I, 1e-6, rtol_of(d), "dgate in place");
            RT_ALLCLOSE(uu, wu1, N * I, 1e-6, rtol_of(d), "dup separate");
            free(g1);
            free(g2);
            free(gg);
            free(uu);
            free(wg1);
            free(wu1);
            rrc_buf_free(&sep);

            /* (4) forward in place: h = gate on the fused buffer */
            rrc_buf_free(&in);
            in = rt_upload(be, d, gu, N * L);
            RT_OK(rrc_swiglu_fwd(be, N, I, in, rrc_view_from(in, I), L, in, L));
            got = rt_download(in, N * L);
            for (int64_t r = 0; r < N; r++)
                for (int64_t c = 0; c < I; c++) {
                    want[r * L + c] = wh[r * I + c];
                    want[r * L + I + c] = gu[r * L + I + c];
                }
            RT_ALLCLOSE(got, want, N * L, 1e-6, rtol_of(d), "swiglu_fwd h = gate on fused buffer");
            free(got);

            /* still refused: blocks overlapping by one column, and a block crossing the row pitch */
            rrc_buf big;
            RT_OK(rrc_buf_alloc(be, d, (N + 1) * L, &big));
            RT_STATUS(rrc_swiglu_bwd(be, N, I, in, rrc_view_from(in, I), L, dhb, I, big, rrc_view_from(big, I - 1), L),
                      RRC_ERR_ALIAS);
            RT_STATUS(rrc_swiglu_bwd(be, N, I, in, rrc_view_from(in, I), L, dhb, I, big, rrc_view_from(big, L - 2), L),
                      RRC_ERR_ALIAS);
            RT_STATUS(rrc_swiglu_bwd(be, N, I, in, rrc_view_from(in, I), L, dhb, I, rrc_view_from(big, L - 2), big, L),
                      RRC_ERR_ALIAS);
            /* h in the up block but with a different leading dimension: not an exact alias */
            RT_STATUS(rrc_swiglu_fwd(be, N - 1, I, in, rrc_view_from(in, I), L, rrc_view_from(in, I), L + 1),
                      RRC_ERR_ALIAS);
            /* h = up block: an exact alias of up, column-disjoint from gate */
            RT_OK(rrc_swiglu_fwd(be, N, I, in, rrc_view_from(in, I), L, rrc_view_from(in, I), L));
            rrc_buf_free(&big);
            rrc_buf_free(&in);
            rrc_buf_free(&dgu);
            rrc_buf_free(&dhb);
            free(want);
        }
}

/* gemm: C in column block 0 of a buffer while B (or A) is read from column block 1. */
RT_TEST(gemm_fused_column_blocks) {
    rt_rng rng = {102};
    const int64_t K = 6, n = 5, m = 4, L = 2 * n; /* X: [K, 2n]; B = X[:, n:2n] (k x n), C = X[0:m, 0:n] */
    for (int bi = 0; bi < 2; bi++) {
        const rrc_backend *be = rt_backend(bi);
        float x[6 * 10], a[4 * 6];
        rt_fill(&rng, x, K * L, 1.0f);
        rt_fill(&rng, a, m * K, 1.0f);
        rrc_buf xb = rt_upload(be, RRC_F32, x, K * L), ab = rt_upload(be, RRC_F32, a, m * K);
        RT_OK(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, m, n, K, 1.0f, ab, K, rrc_view_from(xb, n), L, 0.0f, xb, L));
        float *got = rt_download(xb, K * L);
        double want[60];
        for (int64_t i = 0; i < K * L; i++) want[i] = x[i];
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < n; j++) {
                double s = 0;
                for (int64_t p = 0; p < K; p++) s += (double)a[i * K + p] * x[p * L + n + j];
                want[i * L + j] = s;
            }
        RT_ALLCLOSE(got, want, K * L, 1e-6, 2e-5, "gemm C/B column blocks");
        free(got);
        /* A from column block 1 as well (trans_a = T: A stored k x m = X[:, n:n+m]) */
        RT_OK(rrc_gemm2d(be, RRC_OP_T, RRC_OP_N, m, n - 1, K, 1.0f, rrc_view_from(xb, n), L,
                         rrc_view_from(xb, n + 1), L, 0.0f, xb, L));
        /* refused: overlapping columns, different pitch, C slices of a batch crossing into B */
        RT_STATUS(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, m, n, K, 1.0f, ab, K, rrc_view_from(xb, n - 1), L, 0.0f, xb, L),
                  RRC_ERR_ALIAS);
        RT_STATUS(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, m, n, K - 1, 1.0f, ab, K, rrc_view_from(xb, n), L + 1, 0.0f, xb,
                             L),
                  RRC_ERR_ALIAS);
        rrc_gemm_args g = {RRC_OP_N, RRC_OP_N, m, 2, K, 1, 0, ab, K, 0, rrc_view_from(xb, n), L, 0, xb, L, 2, 3};
        RT_STATUS(rrc_gemm(be, &g), RRC_ERR_ALIAS); /* C slices at columns [0,2), [2,4), [4,6) reach B */
        g.batch = 2;                                /* [0,2), [2,4): disjoint from B's [5,7) */
        g.n = 2;
        g.b = rrc_view_from(xb, n);
        RT_OK(rrc_gemm(be, &g));
        rrc_buf_free(&xb);
        rrc_buf_free(&ab);
    }
}

/* ---- gemm k == 0 and int32 limits -------------------------------------------------------------- */

RT_TEST(gemm_k0_never_reaches_backend) {
    const rrc_backend *be = rec_backend(1);
    for (int dt = 0; dt < 2; dt++) {
        const int64_t m = 3, n = 4, ldc = 6, B = 2, sc = m * ldc + 1, numel = (B - 1) * sc + (m - 1) * ldc + n + 2;
        float c0[40];
        rt_rng rng = {103};
        rt_fill(&rng, c0, numel, 4.0f);
        rt_round_inplace(c0, numel, k_fdt[dt]);
        const float betas[3] = {0.5f, 0.0f, 1.0f};
        for (int bk = 0; bk < 3; bk++) {
            float init[40];
            memcpy(init, c0, sizeof init);
            init[1] = NAN; /* in a C slice: overwritten when beta == 0 */
            rrc_buf cb = rt_upload(be, k_fdt[dt], init, numel);
            rrc_gemm_args g = {RRC_OP_N, RRC_OP_T, m, n, 0, 2.0f, betas[bk], rrc_buf_none(), 1, 0, rrc_buf_none(), 1,
                               0, cb, ldc, sc, B};
            RT_OK(rrc_gemm(be, &g));
            float *got = rt_download(cb, numel);
            for (int64_t i = 0; i < numel; i++) {
                int in_c = 0;
                for (int64_t b = 0; b < B; b++)
                    for (int64_t r = 0; r < m; r++) {
                        int64_t o = b * sc + r * ldc;
                        if (i >= o && i < o + n) in_c = 1;
                    }
                double w = in_c ? (betas[bk] == 0.0f ? 0.0 : (double)rrc_round_bf16(betas[bk] * init[i])) : init[i];
                if (k_fdt[dt] == RRC_F32 && in_c && betas[bk] != 0.0f) w = (double)(betas[bk] * init[i]);
                if (isnan(w))
                    RT_EXPECT(isnan(got[i]), "k=0 beta=%g: element %lld should stay NaN", (double)betas[bk],
                              (long long)i);
                else
                    RT_EXPECT(got[i] == w, "k=0 beta=%g dtype %s: element %lld got %.9g want %.9g", (double)betas[bk],
                              rrc_dtype_name(k_fdt[dt]), (long long)i, (double)got[i], w);
            }
            free(got);
            rrc_buf_free(&cb);
        }
    }
    RT_EXPECT(g_rec_k0_calls == 0, "backend gemm was called %lld times with k == 0", (long long)g_rec_k0_calls);
    /* a normal call still reaches the backend */
    rrc_buf a, b, c;
    RT_OK(rrc_buf_alloc(be, RRC_F32, 4, &a));
    RT_OK(rrc_buf_alloc(be, RRC_F32, 4, &b));
    RT_OK(rrc_buf_alloc(be, RRC_F32, 4, &c));
    RT_OK(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, 2, 2, 2, 1.0f, a, 2, b, 2, 0.0f, c, 2));
    RT_EXPECT(g_rec_calls == 1 && g_rec_last_k == 2, "calls %lld last k %lld", (long long)g_rec_calls,
              (long long)g_rec_last_k);
    rrc_buf_free(&a);
    rrc_buf_free(&b);
    rrc_buf_free(&c);
}

RT_TEST(gemm_int32_limits) {
    const rrc_backend *be = rec_backend(0);
    const int64_t big = (int64_t)INT32_MAX + 5;
    const uint64_t A0 = 1ull << 40, B0 = 2ull << 40, C0 = 3ull << 40;
    rrc_buf a = fake_buf(be, A0, (int64_t)1 << 34), b = fake_buf(be, B0, (int64_t)1 << 34),
            c = fake_buf(be, C0, (int64_t)1 << 34);
    /* batch beyond int32 (A and B broadcast, C slices of one element) */
    rrc_gemm_args g = {RRC_OP_N, RRC_OP_N, 1, 1, 1, 1, 0, a, 1, 0, b, 1, 0, c, 1, 1, big};
    RT_STATUS(rrc_gemm(be, &g), RRC_ERR_SHAPE);
    RT_EXPECT(strstr(rrc_last_error(), "INT32_MAX") != NULL, "%s", rrc_last_error());
    g.batch = INT32_MAX; /* the limit itself is accepted */
    RT_OK(rrc_gemm(be, &g));
    RT_EXPECT(g_rec_calls == 1 && g_rec_last_batch == INT32_MAX, "calls %lld batch %lld", (long long)g_rec_calls,
              (long long)g_rec_last_batch);
    /* m, n, k beyond int32 */
    rrc_gemm_args h = {RRC_OP_N, RRC_OP_N, 1, big, 1, 1, 0, a, 1, 0, b, big, 0, c, big, 0, 1};
    RT_STATUS(rrc_gemm(be, &h), RRC_ERR_SHAPE);
    rrc_gemm_args hm = {RRC_OP_N, RRC_OP_N, big, 1, 1, 1, 0, a, 1, 0, b, 1, 0, c, 1, 0, 1};
    RT_STATUS(rrc_gemm(be, &hm), RRC_ERR_SHAPE);
    rrc_gemm_args hk = {RRC_OP_N, RRC_OP_N, 1, 1, big, 1, 0, a, big, 0, b, 1, 0, c, 1, 0, 1};
    RT_STATUS(rrc_gemm(be, &hk), RRC_ERR_SHAPE);
    RT_EXPECT(g_rec_calls == 1, "a call beyond the int32 limits reached the backend (%lld calls, n = %lld)",
              (long long)g_rec_calls, (long long)g_rec_last_n);
}

/* ---- GQA: dK_g = sum_u dS_(gG+u)^T Q_(gG+u), one gemm per in-group head u batched over g ------- */

RT_TEST(gqa_dk_batched_over_kv_groups) {
    rt_rng rng = {104};
    const int64_t Hq = 6, Hkv = 2, G = Hq / Hkv, hd = 4, T = 7, Tc = 3, q0 = 4, Tk = q0 + Tc;
    const int64_t ldq = Hq * hd, ldk = Hkv * hd, bs = Tc * Tk;
    for (int bi = 0; bi < 2; bi++) {
        const rrc_backend *be = rt_backend(bi);
        float q[7 * 24], ds[6 * 3 * 7], dk[7 * 8];
        rt_fill(&rng, q, T * ldq, 1.0f);
        rt_fill(&rng, ds, Hq * bs, 1.0f);
        rt_fill(&rng, dk, T * ldk, 1.0f);
        rrc_buf qb = rt_upload(be, RRC_F32, q, T * ldq), dsb = rt_upload(be, RRC_F32, ds, Hq * bs),
                dkb = rt_upload(be, RRC_F32, dk, T * ldk);
        for (int64_t u = 0; u < G; u++) {
            rrc_gemm_args g = {RRC_OP_T,
                               RRC_OP_N,
                               Tk,
                               hd,
                               Tc,
                               1.0f,
                               1.0f,
                               rrc_view_from(dsb, u * bs),
                               Tk,
                               G * bs,
                               rrc_view_from(qb, q0 * ldq + u * hd),
                               ldq,
                               G * hd,
                               dkb,
                               ldk,
                               hd,
                               Hkv};
            RT_OK(rrc_gemm(be, &g));
        }
        double want[56];
        for (int64_t j = 0; j < T; j++)
            for (int64_t gk = 0; gk < Hkv; gk++)
                for (int64_t e = 0; e < hd; e++) {
                    double s = dk[j * ldk + gk * hd + e];
                    if (j < Tk)
                        for (int64_t u = 0; u < G; u++) {
                            const int64_t h = gk * G + u; /* repeat_kv: query head h uses kv head h / G */
                            for (int64_t i = 0; i < Tc; i++)
                                s += (double)ds[h * bs + i * Tk + j] * q[(q0 + i) * ldq + h * hd + e];
                        }
                    want[j * ldk + gk * hd + e] = s;
                }
        float *got = rt_download(dkb, T * ldk);
        RT_ALLCLOSE(got, want, T * ldk, 1e-6, 2e-5, "GQA dK batched over KV groups");
        free(got);
        /* the sum over the in-group heads in one call (stride_c = 0) stays refused */
        rrc_gemm_args s0 = {RRC_OP_T, RRC_OP_N, Tk, hd, Tc, 1.0f, 1.0f, dsb, Tk, bs, rrc_view_from(qb, q0 * ldq), ldq,
                            hd, dkb, ldk, 0, G};
        RT_STATUS(rrc_gemm(be, &s0), RRC_ERR_ALIAS);
        rrc_buf_free(&qb);
        rrc_buf_free(&dsb);
        rrc_buf_free(&dkb);
    }
}

/* ---- LM head on labelled rows: gather and scatter with embedding_fwd, loss sum with a gemm ---- */

RT_TEST(lm_head_row_gather_scatter) {
    rt_rng rng = {105};
    const int64_t N = 10, D = 6;
    const int lab[10] = {0, 1, 1, 0, 0, 1, 1, 1, 0, 1}; /* three labelled runs */
    int32_t sel[10], inv[10];
    int64_t ns = 0;
    for (int64_t t = 0; t < N; t++) inv[t] = -1;
    for (int64_t t = 0; t < N; t++)
        if (lab[t]) {
            inv[t] = (int32_t)ns;
            sel[ns++] = (int32_t)t;
        }
    for (int64_t t = 0; t < N; t++)
        if (inv[t] < 0) inv[t] = (int32_t)ns; /* the extra zero row */
    for (int bi = 0; bi < 2; bi++)
        for (int dt = 0; dt < 2; dt++) {
            const rrc_backend *be = rt_backend(bi);
            const rrc_dtype d = k_fdt[dt];
            float h[60], dsel[60];
            rt_fill(&rng, h, N * D, 2.0f);
            rt_fill(&rng, dsel, ns * D, 2.0f);
            rt_round_inplace(h, N * D, d);
            rt_round_inplace(dsel, ns * D, d);
            rrc_buf hb = rt_upload(be, d, h, N * D), selb = rt_upload_i32(be, sel, ns), invb = rt_upload_i32(be, inv, N);
            rrc_buf hs, dext, dh;
            RT_OK(rrc_buf_alloc(be, d, ns * D, &hs));
            RT_OK(rrc_embedding_fwd(be, ns, D, N, selb, hb, hs)); /* gather */
            float *got = rt_download(hs, ns * D);
            double want[60];
            for (int64_t i = 0; i < ns; i++)
                for (int64_t c = 0; c < D; c++) want[i * D + c] = h[sel[i] * D + c];
            RT_ALLCLOSE(got, want, ns * D, 0, 0, "row gather");
            free(got);
            RT_OK(rrc_buf_alloc(be, d, (ns + 1) * D, &dext)); /* zero-initialised: row ns stays 0 */
            rrc_buf up = rt_upload(be, d, dsel, ns * D);
            RT_OK(rrc_buf_copy(dext, up, ns * D));
            RT_OK(rrc_buf_alloc(be, d, N * D, &dh));
            RT_OK(rrc_embedding_fwd(be, N, D, ns + 1, invb, dext, dh)); /* scatter into [N, D], zeros elsewhere */
            got = rt_download(dh, N * D);
            for (int64_t t = 0; t < N; t++)
                for (int64_t c = 0; c < D; c++) want[t * D + c] = lab[t] ? dsel[inv[t] * D + c] : 0.0;
            RT_ALLCLOSE(got, want, N * D, 0, 0, "row scatter via inverse index");
            free(got);
            /* per-row losses of two CE chunks summed on the device by gemm with a ones row */
            float loss[7], ones[7];
            rt_fill(&rng, loss, ns, 3.0f);
            for (int64_t i = 0; i < ns; i++) ones[i] = 1.0f;
            rrc_buf lb = rt_upload(be, RRC_F32, loss, ns), ob = rt_upload(be, RRC_F32, ones, ns), acc;
            RT_OK(rrc_buf_alloc(be, RRC_F32, 1, &acc));
            const int64_t c1 = 4;
            RT_OK(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, 1, 1, c1, 1.0f, ob, c1, lb, 1, 1.0f, acc, 1));
            RT_OK(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, 1, 1, ns - c1, 1.0f, ob, ns - c1, rrc_view_from(lb, c1), 1, 1.0f,
                             acc, 1));
            float s;
            RT_OK(rrc_buf_download(acc, &s, 1));
            double ws = 0;
            for (int64_t i = 0; i < ns; i++) ws += loss[i];
            RT_CLOSE(s, ws, 1e-6, 1e-6);
            rrc_buf bufs[] = {hb, selb, invb, hs, dext, dh, up, lb, ob, acc};
            for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) rrc_buf_free(&bufs[i]);
        }
}

/* ---- CPU gemm packing scratch is bounded by the output tile, not by n*k ------------------------- */

RT_TEST(gemm_cpu_scratch_bounded) {
    rt_rng rng = {106};
    const int64_t m = 3, n = 1024, k = 96;
    float *a = (float *)malloc(sizeof(float) * m * k), *b = (float *)malloc(sizeof(float) * n * k);
    rt_fill(&rng, a, m * k, 1.0f);
    rt_fill(&rng, b, n * k, 1.0f);
    for (int bi = 0; bi < 2; bi++) {
        const rrc_backend *be = rt_backend(bi);
        rrc_buf ab = rt_upload(be, RRC_F32, a, m * k), bb = rt_upload(be, RRC_BF16, b, n * k), cb;
        RT_OK(rrc_buf_alloc(be, RRC_F32, m * n, &cb));
        atomic_store(&g_max_malloc, 0);
        atomic_store(&g_track_malloc, 1);
        rrc_status st = rrc_gemm2d(be, RRC_OP_N, RRC_OP_T, m, n, k, 1.0f, ab, k, bb, k, 0.0f, cb, n);
        atomic_store(&g_track_malloc, 0);
        RT_OK(st);
        size_t mx = atomic_load(&g_max_malloc), bound = 2 * 32 * (size_t)k * sizeof(float);
        RT_EXPECT(mx <= bound, "largest malloc during gemm: %zu bytes (op(B) is %zu bytes as f32, bound %zu)", mx,
                  (size_t)(n * k) * sizeof(float), bound);
        /* values (the dot-product order is unchanged, so double accumulation is near exact) */
        float *got = rt_download(cb, m * n);
        double *want = rt_dalloc(m * n);
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < n; j++) {
                double s = 0;
                for (int64_t p = 0; p < k; p++) s += (double)a[i * k + p] * rrc_round_bf16(b[j * k + p]);
                want[i * n + j] = s;
            }
        RT_ALLCLOSE(got, want, m * n, 1e-5, 1e-5, "gemm tiled packing");
        free(got);
        free(want);
        rrc_buf_free(&ab);
        rrc_buf_free(&bb);
        rrc_buf_free(&cb);
    }
    free(a);
    free(b);
}

/* ---- cross entropy with -inf logits (leading, and elsewhere) ------------------------------------ */

RT_TEST(cross_entropy_neg_inf_logits) {
    const int64_t R = 3, V = 4;
    const float z[12] = {-INFINITY, 1.0f, 2.0f, 0.5f, -INFINITY, -INFINITY, 0.25f, 3.0f, 1.5f, -INFINITY, -2.0f, 0.0f};
    const int32_t lab[3] = {2, 3, 0};
    double wl[3], wg[12];
    for (int64_t r = 0; r < R; r++) {
        double m = -INFINITY, s = 0;
        for (int64_t j = 0; j < V; j++) m = z[r * V + j] > m ? z[r * V + j] : m;
        for (int64_t j = 0; j < V; j++) s += exp((double)z[r * V + j] - m);
        wl[r] = m + log(s) - z[r * V + lab[r]];
        for (int64_t j = 0; j < V; j++) wg[r * V + j] = (exp((double)z[r * V + j] - m) / s - (j == lab[r])) * 0.5;
    }
    for (int bi = 0; bi < 2; bi++)
        for (int dt = 0; dt < 2; dt++) {
            const rrc_backend *be = rt_backend(bi);
            rrc_buf zb = rt_upload(be, k_fdt[dt], z, R * V), lb = rt_upload_i32(be, lab, R), loss;
            RT_OK(rrc_buf_alloc(be, RRC_F32, R, &loss));
            RT_OK(rrc_cross_entropy_fwd_bwd(be, R, V, zb, V, lb, loss, 0.5f, rrc_buf_none(), 1));
            float *gl = rt_download(loss, R), *gg = rt_download(zb, R * V);
            RT_ALLCLOSE(gl, wl, R, 1e-6, 1e-6, "CE loss with -inf logits");
            RT_ALLCLOSE(gg, wg, R * V, 1e-7, rtol_of(k_fdt[dt]), "CE grad with -inf logits");
            free(gl);
            free(gg);
            rrc_buf_free(&zb);
            rrc_buf_free(&lb);
            rrc_buf_free(&loss);
        }
}

/* Kernel level: an empty or -inf-only state must stay (m = -inf, s = 0) under push and merge. */
RT_TEST(kc_lse_neg_inf) {
    rrc_kc_lse a = rrc_kc_lse_push(rrc_kc_lse_init(), -INFINITY);
    RT_EXPECT(a.m == -INFINITY && a.s == 0.0f, "push(-inf): m %g s %g", (double)a.m, (double)a.s);
    a = rrc_kc_lse_push(a, 1.0f);
    RT_EXPECT(a.m == 1.0f && a.s == 1.0f, "push(-inf), push(1): m %g s %g", (double)a.m, (double)a.s);
    a = rrc_kc_lse_push(a, -INFINITY);
    RT_EXPECT(a.m == 1.0f && a.s == 1.0f, "push(-inf) after a finite max: m %g s %g", (double)a.m, (double)a.s);
    rrc_kc_lse b = rrc_kc_lse_push(rrc_kc_lse_init(), 0.0f);
    rrc_kc_lse r1 = rrc_kc_lse_merge(a, b), r2 = rrc_kc_lse_merge(b, a);
    const double want = 1.0 + exp(-1.0);
    RT_CLOSE(r1.s, want, 0, 1e-6);
    RT_CLOSE(r2.s, want, 0, 1e-6);
    RT_EXPECT(r1.m == 1.0f && r2.m == 1.0f, "merge max");
    rrc_kc_lse e = rrc_kc_lse_push(rrc_kc_lse_push(rrc_kc_lse_init(), -INFINITY), -INFINITY);
    rrc_kc_lse r3 = rrc_kc_lse_merge(e, b);
    RT_EXPECT(r3.m == 0.0f && r3.s == 1.0f, "merge(-inf-only, {0}) : m %g s %g", (double)r3.m, (double)r3.s);
}

const rt_case rt_suite_contracts[] = {
    RT_CASE(kc_lse_neg_inf),
    RT_CASE(swiglu_fused_column_blocks),   RT_CASE(gemm_fused_column_blocks), RT_CASE(gemm_k0_never_reaches_backend),
    RT_CASE(gemm_int32_limits),            RT_CASE(gqa_dk_batched_over_kv_groups),
    RT_CASE(lm_head_row_gather_scatter),   RT_CASE(gemm_cpu_scratch_bounded),
    RT_CASE(cross_entropy_neg_inf_logits),
};
const size_t rt_suite_contracts_n = sizeof rt_suite_contracts / sizeof rt_suite_contracts[0];
