/* Forward tests of every op on both CPU backends against naive double-precision references,
 * precondition / aliasing checks of src/ops.c, and thread-count determinism. */
#include <omp.h>

#include "ops_util.h"
#include "rrc/config.h"
#include "rrc/safetensors.h"
#include "rrc_kc.h"

static const rrc_dtype k_fdt[2] = {RRC_F32, RRC_BF16};

/* tolerance for a value stored in dtype dt computed from exact (double) math */
static double rtol_of(rrc_dtype dt) { return dt == RRC_BF16 ? 1.0 / 128.0 : 2e-5; }

/* ---- gemm --------------------------------------------------------------------------------- */

typedef struct {
    rrc_trans ta, tb;
    int64_t m, n, k, batch;
    rrc_dtype da, db, dc;
    float alpha, beta;
    int broadcast_b; /* stride_b = 0 */
    int interleave_c; /* C slices interleaved in rows: stride_c = n, ldc = batch*n + 1 */
} gemm_case;

static void gemm_case_run(const rrc_backend *be, const gemm_case *gc, rt_rng *rng) {
    const int64_t m = gc->m, n = gc->n, k = gc->k, B = gc->batch;
    const int64_t ar = gc->ta == RRC_OP_N ? m : k, ac = gc->ta == RRC_OP_N ? k : m;
    const int64_t br = gc->tb == RRC_OP_N ? k : n, bc = gc->tb == RRC_OP_N ? n : k;
    const int64_t lda = ac + 3, ldb = bc + 2;
    const int64_t sa = ar * lda + 1, sb = gc->broadcast_b ? 0 : br * ldb + 5;
    int64_t ldc, sc;
    if (gc->interleave_c) {
        ldc = B * n + 1;
        sc = n;
    } else {
        ldc = n + 1;
        sc = m * ldc + 2;
    }
    const int64_t na = (B - 1) * sa + ar * lda, nb = (B - 1) * sb + br * ldb, nc = (B - 1) * sc + m * ldc;
    float *A = (float *)malloc((size_t)na * 4), *Bm = (float *)malloc((size_t)nb * 4), *C = (float *)malloc((size_t)nc * 4);
    rt_fill(rng, A, na, 1.0f);
    rt_fill(rng, Bm, nb, 1.0f);
    rt_fill(rng, C, nc, 1.0f);
    if (gc->beta == 0.0f)
        for (int64_t i = 0; i < nc; i++) C[i] = (i % 3 == 0) ? NAN : C[i]; /* must not be read */
    rt_round_inplace(A, na, gc->da);
    rt_round_inplace(Bm, nb, gc->db);
    rt_round_inplace(C, nc, gc->dc);
    double *want = rt_dalloc(nc);
    for (int64_t i = 0; i < nc; i++) want[i] = C[i];
    for (int64_t b = 0; b < B; b++)
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < n; j++) {
                double s = 0;
                for (int64_t p = 0; p < k; p++) {
                    double av = gc->ta == RRC_OP_N ? A[b * sa + i * lda + p] : A[b * sa + p * lda + i];
                    double bv = gc->tb == RRC_OP_N ? Bm[b * sb + p * ldb + j] : Bm[b * sb + j * ldb + p];
                    s += av * bv;
                }
                int64_t ci = b * sc + i * ldc + j;
                want[ci] = gc->alpha * s + (gc->beta != 0.0f ? gc->beta * (double)C[ci] : 0.0);
            }
    rrc_buf a = rt_upload(be, gc->da, A, na), bb = rt_upload(be, gc->db, Bm, nb), c = rt_upload(be, gc->dc, C, nc);
    rrc_gemm_args g = {gc->ta, gc->tb, m, n, k, gc->alpha, gc->beta, a, lda, sa, bb, ldb, sb, c, ldc, sc, B};
    RT_OK(rrc_gemm(be, &g));
    float *got = rt_download(c, nc);
    /* the result is rounded once to C's dtype; accumulation error ~ k * eps_f32 */
    double rtol = gc->dc == RRC_BF16 ? 1.0 / 128.0 : 1e-5;
    char label[128];
    snprintf(label, sizeof label, "gemm %s ta=%d tb=%d %s*%s->%s m=%lld n=%lld k=%lld batch=%lld", be->name, gc->ta,
             gc->tb, rrc_dtype_name(gc->da), rrc_dtype_name(gc->db), rrc_dtype_name(gc->dc), (long long)m,
             (long long)n, (long long)k, (long long)B);
    /* positions outside the C slices keep their initial value (NaN in some beta == 0 cases) */
    int bad = 0;
    for (int64_t i = 0; i < nc; i++) {
        if (isnan(want[i]) && isnan(got[i])) continue;
        if (!(fabs(got[i] - want[i]) <= 1e-5 + rtol * fabs(want[i]))) {
            if (bad == 0)
                RT_EXPECT(0, "%s: element %lld got %.9g want %.9g", label, (long long)i, (double)got[i], want[i]);
            bad++;
        }
    }
    RT_EXPECT(bad == 0, "%s: %d bad elements", label, bad);
    free(got);
    free(A);
    free(Bm);
    free(C);
    free(want);
    rrc_buf_free(&a);
    rrc_buf_free(&bb);
    rrc_buf_free(&c);
}

RT_TEST(gemm_all_transposes_and_dtypes) {
    rt_rng rng = {11};
    for (int bi = 0; bi < 2; bi++) {
        const rrc_backend *be = rt_backend(bi);
        for (int ta = 0; ta < 2; ta++)
            for (int tb = 0; tb < 2; tb++)
                for (int combo = 0; combo < 8; combo++)
                    for (int bz = 0; bz < 2; bz++) {
                        gemm_case gc = {(rrc_trans)ta, (rrc_trans)tb, 7, 5, 9, 1, k_fdt[combo & 1],
                                        k_fdt[(combo >> 1) & 1], k_fdt[(combo >> 2) & 1], 0.75f, bz ? 0.5f : 0.0f, 0, 0};
                        gemm_case_run(be, &gc, &rng);
                    }
    }
}

RT_TEST(gemm_batched_broadcast_interleaved) {
    rt_rng rng = {12};
    for (int bi = 0; bi < 2; bi++) {
        const rrc_backend *be = rt_backend(bi);
        gemm_case g1 = {RRC_OP_N, RRC_OP_T, 6, 8, 4, 3, RRC_F32, RRC_F32, RRC_F32, 1.0f, 0.0f, 1, 0};
        gemm_case_run(be, &g1, &rng);
        gemm_case g2 = {RRC_OP_N, RRC_OP_N, 5, 4, 6, 4, RRC_BF16, RRC_BF16, RRC_F32, -0.5f, 1.0f, 0, 1};
        gemm_case_run(be, &g2, &rng);
        gemm_case g3 = {RRC_OP_T, RRC_OP_N, 4, 6, 5, 2, RRC_F32, RRC_BF16, RRC_BF16, 2.0f, 0.25f, 1, 1};
        gemm_case_run(be, &g3, &rng);
        /* larger: several output tiles, parallel path */
        gemm_case g4 = {RRC_OP_N, RRC_OP_T, 70, 65, 130, 1, RRC_F32, RRC_F32, RRC_F32, 1.0f, 0.0f, 0, 0};
        gemm_case_run(be, &g4, &rng);
        gemm_case g5 = {RRC_OP_T, RRC_OP_T, 33, 40, 67, 2, RRC_BF16, RRC_F32, RRC_F32, 1.0f, 1.0f, 0, 0};
        gemm_case_run(be, &g5, &rng);
        /* k = 0: C = beta * C */
        gemm_case g6 = {RRC_OP_N, RRC_OP_N, 3, 4, 0, 1, RRC_F32, RRC_F32, RRC_F32, 1.0f, 0.5f, 0, 0};
        gemm_case_run(be, &g6, &rng);
    }
}

RT_TEST(gemm_preconditions) {
    const rrc_backend *be = rrc_backend_cpu();
    rrc_buf a, b, c, i32, other;
    RT_OK(rrc_buf_alloc(be, RRC_F32, 64, &a));
    RT_OK(rrc_buf_alloc(be, RRC_F32, 64, &b));
    RT_OK(rrc_buf_alloc(be, RRC_F32, 64, &c));
    RT_OK(rrc_buf_alloc(be, RRC_I32, 64, &i32));
    RT_OK(rrc_buf_alloc(rrc_backend_cpu_f32acc(), RRC_F32, 64, &other));
    RT_OK(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, 4, 4, 4, 1, a, 4, b, 4, 0, c, 4));
    RT_STATUS(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, 4, 4, 4, 1, a, 3, b, 4, 0, c, 4), RRC_ERR_SHAPE);  /* lda < k */
    RT_STATUS(rrc_gemm2d(be, RRC_OP_T, RRC_OP_N, 4, 4, 17, 1, a, 4, b, 4, 0, c, 4), RRC_ERR_SHAPE); /* A, B too small */
    RT_STATUS(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, 4, 4, 4, 1, a, 4, b, 4, 0, c, 3), RRC_ERR_SHAPE);  /* ldc < n */
    RT_STATUS(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, 9, 8, 1, 1, a, 1, b, 8, 0, c, 8), RRC_ERR_SHAPE);  /* C too small */
    RT_STATUS(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, 4, 4, 4, 1, i32, 4, b, 4, 0, c, 4), RRC_ERR_DTYPE);
    RT_STATUS(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, 4, 4, 4, 1, a, 4, b, 4, 0, a, 4), RRC_ERR_ALIAS);  /* C == A */
    RT_STATUS(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, 2, 2, 2, 1, a, 2, b, 2, 0, rrc_view(a, 3, 8), 2), RRC_ERR_ALIAS);
    RT_STATUS(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, 4, 4, 4, 1, other, 4, b, 4, 0, c, 4), RRC_ERR_INVALID_ARG);
    RT_STATUS(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, 4, 4, 4, 1, rrc_view(a, 60, 8), 4, b, 4, 0, c, 4), RRC_ERR_SHAPE);
    RT_STATUS(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, -1, 4, 4, 1, a, 4, b, 4, 0, c, 4), RRC_ERR_SHAPE);
    RT_STATUS(rrc_gemm2d(be, (rrc_trans)5, RRC_OP_N, 4, 4, 4, 1, a, 4, b, 4, 0, c, 4), RRC_ERR_INVALID_ARG);
    RT_STATUS(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, 4, 4, 4, 1, rrc_buf_none(), 4, b, 4, 0, c, 4), RRC_ERR_INVALID_ARG);
    RT_OK(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, 0, 4, 4, 1, rrc_buf_none(), 4, rrc_buf_none(), 4, 0, rrc_buf_none(), 4));
    rrc_gemm_args g = {RRC_OP_N, RRC_OP_N, 2, 2, 2, 1, 0, a, 2, 4, b, 2, 4, c, 2, 0, 2};
    RT_STATUS(rrc_gemm(be, &g), RRC_ERR_ALIAS); /* stride_c = 0 with batch 2: slices overlap */
    g.stride_c = 3;                            /* overlapping by one row */
    RT_STATUS(rrc_gemm(be, &g), RRC_ERR_ALIAS);
    g.stride_c = 4;
    RT_OK(rrc_gemm(be, &g));
    g.batch = 0;
    RT_STATUS(rrc_gemm(be, &g), RRC_ERR_SHAPE);
    RT_STATUS(rrc_gemm(be, NULL), RRC_ERR_INVALID_ARG);
    RT_STATUS(rrc_gemm(NULL, &g), RRC_ERR_INVALID_ARG);
    /* invalid view from out-of-range view arithmetic */
    rrc_buf bad = rrc_view(a, 10, 100);
    RT_EXPECT(bad.numel == -1 && bad.ptr == NULL, "invalid view");
    RT_STATUS(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, 1, 1, 1, 1, bad, 1, b, 1, 0, c, 1), RRC_ERR_SHAPE);
    RT_EXPECT(strstr(rrc_last_error(), "invalid") != NULL, "%s", rrc_last_error());
    RT_STATUS(rrc_buf_free(&bad), RRC_ERR_INVALID_ARG);
    rrc_buf_free(&a);
    rrc_buf_free(&b);
    rrc_buf_free(&c);
    rrc_buf_free(&i32);
    rrc_buf_free(&other);
}

/* ---- embedding / bias ----------------------------------------------------------------------- */

RT_TEST(embedding_and_bias) {
    rt_rng rng = {21};
    for (int bi = 0; bi < 2; bi++)
        for (int td = 0; td < 2; td++)
            for (int od = 0; od < 2; od++) {
                const rrc_backend *be = rt_backend(bi);
                const int64_t V = 11, D = 6, T = 9;
                float tab[66];
                rt_fill(&rng, tab, V * D, 2.0f);
                rt_round_inplace(tab, V * D, k_fdt[td]);
                int32_t ids[9] = {0, 10, 3, 3, 7, 1, 0, 9, 5};
                rrc_buf t = rt_upload(be, k_fdt[td], tab, V * D), id = rt_upload_i32(be, ids, T), out;
                RT_OK(rrc_buf_alloc(be, k_fdt[od], T * D, &out));
                RT_OK(rrc_embedding_fwd(be, T, D, V, id, t, out));
                float *got = rt_download(out, T * D);
                double want[54];
                for (int64_t i = 0; i < T; i++)
                    for (int64_t c = 0; c < D; c++) want[i * D + c] = tab[ids[i] * D + c];
                RT_ALLCLOSE(got, want, T * D, 0, k_fdt[od] == RRC_BF16 ? 1.0 / 256 : 0, "embedding");
                free(got);
                int32_t bad_ids[2] = {0, 11};
                rrc_buf bid = rt_upload_i32(be, bad_ids, 2);
                RT_STATUS(rrc_embedding_fwd(be, 2, D, V, bid, t, out), RRC_ERR_RANGE);
                RT_STATUS(rrc_embedding_fwd(be, T, D, V, id, t, rrc_view(out, 1, T * D - 1)), RRC_ERR_SHAPE);
                RT_STATUS(rrc_embedding_fwd(be, T, D, V, t, t, out), RRC_ERR_DTYPE);
                /* add_bias / bias_grad on a padded matrix */
                const int64_t R = 5, Cc = 7, ld = 9;
                float x[45], bias[7];
                rt_fill(&rng, x, R * ld, 1.0f);
                rt_fill(&rng, bias, Cc, 1.0f);
                rt_round_inplace(x, R * ld, k_fdt[od]);
                rt_round_inplace(bias, Cc, k_fdt[td]);
                rrc_buf xb = rt_upload(be, k_fdt[od], x, R * ld), bb = rt_upload(be, k_fdt[td], bias, Cc), db;
                RT_OK(rrc_add_bias(be, R, Cc, xb, ld, bb));
                float *gx = rt_download(xb, R * ld);
                double wx[45];
                for (int64_t r = 0; r < R; r++)
                    for (int64_t c = 0; c < ld; c++) wx[r * ld + c] = x[r * ld + c] + (c < Cc ? bias[c] : 0.0);
                RT_ALLCLOSE(gx, wx, R * ld, 1e-7, rtol_of(k_fdt[od]), "add_bias");
                RT_OK(rrc_buf_alloc(be, RRC_F32, Cc, &db));
                RT_OK(rrc_bias_grad(be, R, Cc, xb, ld, db, 0));
                RT_OK(rrc_bias_grad(be, R, Cc, xb, ld, db, 1)); /* accumulate: 2x */
                float *gdb = rt_download(db, Cc);
                double wdb[7] = {0};
                for (int64_t c = 0; c < Cc; c++)
                    for (int64_t r = 0; r < R; r++) wdb[c] += 2.0 * gx[r * ld + c];
                RT_ALLCLOSE(gdb, wdb, Cc, 1e-6, 1e-5, "bias_grad");
                RT_STATUS(rrc_add_bias(be, R, Cc, xb, ld, rrc_view(xb, 0, Cc)), RRC_ERR_ALIAS);
                RT_STATUS(rrc_add_bias(be, R, Cc, xb, Cc - 1, bb), RRC_ERR_SHAPE);
                free(gx);
                free(gdb);
                rrc_buf_free(&t);
                rrc_buf_free(&id);
                rrc_buf_free(&out);
                rrc_buf_free(&bid);
                rrc_buf_free(&xb);
                rrc_buf_free(&bb);
                rrc_buf_free(&db);
            }
}

/* ---- rmsnorm -------------------------------------------------------------------------------- */

static void ref_rmsnorm(const float *x, const float *w, int64_t rows, int64_t dim, int64_t ld, double eps, double *y) {
    for (int64_t r = 0; r < rows; r++) {
        double s = 0;
        for (int64_t i = 0; i < dim; i++) s += (double)x[r * ld + i] * x[r * ld + i];
        double rstd = 1.0 / sqrt(s / dim + eps);
        for (int64_t i = 0; i < dim; i++) y[r * ld + i] = (double)w[i] * x[r * ld + i] * rstd;
    }
}

RT_TEST(rmsnorm_fwd_ref) {
    rt_rng rng = {31};
    for (int bi = 0; bi < 2; bi++)
        for (int xd = 0; xd < 2; xd++)
            for (int wd = 0; wd < 2; wd++)
                for (int yd = 0; yd < 2; yd++) {
                    const rrc_backend *be = rt_backend(bi);
                    const int64_t R = 6, D = 37, ld = 40;
                    float x[240], w[37];
                    rt_fill(&rng, x, R * ld, 3.0f);
                    rt_fill(&rng, w, D, 1.5f);
                    rt_round_inplace(x, R * ld, k_fdt[xd]);
                    rt_round_inplace(w, D, k_fdt[wd]);
                    double want[240] = {0};
                    ref_rmsnorm(x, w, R, D, ld, 1e-6, want);
                    rrc_buf xb = rt_upload(be, k_fdt[xd], x, R * ld), wb = rt_upload(be, k_fdt[wd], w, D), yb;
                    RT_OK(rrc_buf_alloc(be, k_fdt[yd], R * ld, &yb));
                    RT_OK(rrc_rmsnorm_fwd(be, R, D, ld, xb, wb, 1e-6f, yb));
                    float *got = rt_download(yb, R * ld);
                    for (int64_t r = 0; r < R; r++)
                        for (int64_t i = D; i < ld; i++) want[r * ld + i] = 0.0; /* padding untouched (zero) */
                    RT_ALLCLOSE(got, want, R * ld, 1e-6, rtol_of(k_fdt[yd]), "rmsnorm_fwd");
                    free(got);
                    if (xd == yd) { /* in place */
                        RT_OK(rrc_rmsnorm_fwd(be, R, D, ld, xb, wb, 1e-6f, xb));
                        got = rt_download(xb, R * ld);
                        for (int64_t r = 0; r < R; r++)
                            for (int64_t i = D; i < ld; i++) want[r * ld + i] = x[r * ld + i];
                        RT_ALLCLOSE(got, want, R * ld, 1e-6, rtol_of(k_fdt[yd]), "rmsnorm_fwd in place");
                        free(got);
                        RT_STATUS(rrc_rmsnorm_fwd(be, R - 1, D, ld, xb, wb, 1e-6f, rrc_view(xb, 1, R * ld - 1)),
                                  RRC_ERR_ALIAS);
                    }
                    rrc_buf_free(&xb);
                    rrc_buf_free(&wb);
                    rrc_buf_free(&yb);
                }
    /* q/k-norm usage: [tokens*heads, head_dim] view of a [tokens, heads*head_dim] buffer */
    const rrc_backend *be = rrc_backend_cpu();
    float q[3 * 4 * 8], w[8];
    rt_fill(&rng, q, 96, 1.0f);
    rt_fill(&rng, w, 8, 1.0f);
    double want[96];
    ref_rmsnorm(q, w, 12, 8, 8, 1e-6, want);
    rrc_buf qb = rt_upload(be, RRC_F32, q, 96), wb = rt_upload(be, RRC_F32, w, 8);
    RT_OK(rrc_rmsnorm_fwd(be, 12, 8, 8, qb, wb, 1e-6f, qb));
    float *got = rt_download(qb, 96);
    RT_ALLCLOSE(got, want, 96, 1e-7, 2e-6, "per-head norm");
    free(got);
    RT_STATUS(rrc_rmsnorm_fwd(be, 12, 8, 8, qb, wb, 0.0f, qb), RRC_ERR_INVALID_ARG);
    RT_STATUS(rrc_rmsnorm_fwd(be, 12, 8, 7, qb, wb, 1e-6f, qb), RRC_ERR_SHAPE);
    RT_STATUS(rrc_rmsnorm_bwd(be, 12, 8, 8, qb, wb, 1e-6f, qb, qb, 1), RRC_ERR_ALIAS); /* accumulate + alias */
    rrc_buf_free(&qb);
    rrc_buf_free(&wb);
}

/* ---- rope ----------------------------------------------------------------------------------- */

RT_TEST(rope_fwd_ref) {
    rt_rng rng = {41};
    rrc_model_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.head_dim = 8;
    cfg.rope_theta = 10000.0;
    cfg.rope_type = RRC_ROPE_DEFAULT;
    cfg.rope_attention_scaling = 1.0;
    const int64_t NP = 16, half = 4, T = 5, H = 3, hd = 8, ld = 3 * 8 + 2;
    float cs[64], sn[64];
    RT_OK(rrc_rope_tables(&cfg, NP, cs, sn));
    int32_t pos[5] = {0, 1, 15, 7, 3};
    for (int bi = 0; bi < 2; bi++)
        for (int dt = 0; dt < 2; dt++) {
            const rrc_backend *be = rt_backend(bi);
            float x[5 * 26];
            rt_fill(&rng, x, T * ld, 2.0f);
            rt_round_inplace(x, T * ld, k_fdt[dt]);
            double want[130];
            for (int64_t i = 0; i < T * ld; i++) want[i] = x[i];
            for (int64_t t = 0; t < T; t++)
                for (int64_t h = 0; h < H; h++)
                    for (int64_t i = 0; i < half; i++) {
                        double c = cs[pos[t] * half + i], s = sn[pos[t] * half + i];
                        double a = x[t * ld + h * hd + i], b = x[t * ld + h * hd + i + half];
                        want[t * ld + h * hd + i] = a * c - b * s;
                        want[t * ld + h * hd + i + half] = b * c + a * s;
                    }
            rrc_buf xb = rt_upload(be, k_fdt[dt], x, T * ld), pb = rt_upload_i32(be, pos, T);
            rrc_buf cb = rt_upload(be, RRC_F32, cs, NP * half), sb = rt_upload(be, RRC_F32, sn, NP * half);
            RT_OK(rrc_rope_fwd(be, T, H, hd, ld, xb, pb, cb, sb, NP, xb)); /* in place */
            float *got = rt_download(xb, T * ld);
            RT_ALLCLOSE(got, want, T * ld, 1e-6, rtol_of(k_fdt[dt]), "rope_fwd");
            /* bwd(fwd(x)) == x for f32 (rotation is orthogonal) */
            RT_OK(rrc_rope_bwd(be, T, H, hd, ld, xb, pb, cb, sb, NP, xb));
            free(got);
            got = rt_download(xb, T * ld);
            double orig[130];
            for (int64_t i = 0; i < T * ld; i++) orig[i] = x[i];
            RT_ALLCLOSE(got, orig, T * ld, dt ? 0.03 : 1e-6, dt ? 1.0 / 128 : 1e-6, "rope_bwd(rope_fwd(x))");
            free(got);
            int32_t badpos[1] = {16};
            rrc_buf bp = rt_upload_i32(be, badpos, 1);
            RT_STATUS(rrc_rope_fwd(be, 1, H, hd, ld, xb, bp, cb, sb, NP, xb), RRC_ERR_RANGE);
            RT_STATUS(rrc_rope_fwd(be, T, H, 7, ld, xb, pb, cb, sb, NP, xb), RRC_ERR_SHAPE); /* odd head_dim */
            RT_STATUS(rrc_rope_fwd(be, T, H, hd, 23, xb, pb, cb, sb, NP, xb), RRC_ERR_SHAPE); /* ld < H*hd */
            RT_STATUS(rrc_rope_fwd(be, T, H, hd, ld, xb, pb, cb, sb, NP + 1, xb), RRC_ERR_SHAPE); /* table small */
            RT_STATUS(rrc_rope_fwd(be, T - 1, H, hd, ld, xb, pb, cb, sb, NP, rrc_view(xb, 2, (T - 1) * ld)),
                      RRC_ERR_ALIAS);
            RT_STATUS(rrc_rope_fwd(be, T, H, hd, ld, xb, cb, cb, sb, NP, xb), RRC_ERR_DTYPE); /* pos not I32 */
            rrc_buf_free(&xb);
            rrc_buf_free(&pb);
            rrc_buf_free(&cb);
            rrc_buf_free(&sb);
            rrc_buf_free(&bp);
        }
}

/* ---- attention softmax ---------------------------------------------------------------------- */

static void ref_softmax(const float *s, int64_t batch, int64_t rows, int64_t cols, int64_t ld, int64_t bs,
                        double scale, int64_t off, int causal, const int32_t *doc, double *p) {
    for (int64_t b = 0; b < batch; b++)
        for (int64_t r = 0; r < rows; r++) {
            int64_t q = off + r;
            const float *row = s + b * bs + r * ld;
            double *pr = p + b * bs + r * ld;
            double m = -INFINITY;
            for (int64_t j = 0; j < cols; j++)
                if (rrc_kc_attn_allowed(j, q, causal, doc) && scale * row[j] > m) m = scale * row[j];
            double z = 0;
            for (int64_t j = 0; j < cols; j++)
                if (rrc_kc_attn_allowed(j, q, causal, doc)) z += exp(scale * row[j] - m);
            for (int64_t j = 0; j < cols; j++)
                pr[j] = (m == -INFINITY || !rrc_kc_attn_allowed(j, q, causal, doc)) ? 0.0 : exp(scale * row[j] - m) / z;
        }
}

RT_TEST(attn_softmax_fwd_ref) {
    rt_rng rng = {51};
    /* doc ids: three packed documents in a row of 12 tokens */
    int32_t doc[12] = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2};
    for (int bi = 0; bi < 2; bi++)
        for (int sd = 0; sd < 2; sd++)
            for (int pd = 0; pd < 2; pd++)
                for (int mode = 0; mode < 3; mode++) {
                    const rrc_backend *be = rt_backend(bi);
                    const int64_t B = 3, rows = 5, cols = 12, ld = 13, bs = 5 * 13 + 1, off = 6;
                    int causal = mode != 2;
                    const int32_t *dp = mode >= 1 ? doc : NULL;
                    const int64_t n = (B - 1) * bs + rows * ld;
                    float *s = (float *)malloc((size_t)n * 4);
                    rt_fill(&rng, s, n, 4.0f);
                    rt_round_inplace(s, n, k_fdt[sd]);
                    double *want = rt_dalloc(n);
                    ref_softmax(s, B, rows, cols, ld, bs, 0.35, off, causal, dp, want);
                    rrc_buf sb = rt_upload(be, k_fdt[sd], s, n), pb, db = dp ? rt_upload_i32(be, doc, 12) : rrc_buf_none();
                    RT_OK(rrc_buf_alloc(be, k_fdt[pd], n, &pb));
                    RT_OK(rrc_attn_softmax_fwd(be, B, rows, cols, ld, bs, sb, 0.35f, off, causal, db, pb));
                    float *got = rt_download(pb, n);
                    /* padding columns / gaps are untouched (zero in the fresh buffer) */
                    for (int64_t b = 0; b < B; b++)
                        for (int64_t r = 0; r < rows; r++) want[b * bs + r * ld + 12] = 0.0;
                    for (int64_t i = rows * ld; i < bs && B > 1; i++) want[i] = 0.0;
                    for (int64_t i = bs + rows * ld; i < 2 * bs; i++) want[i] = 0.0;
                    RT_ALLCLOSE(got, want, n, 1e-7, rtol_of(k_fdt[pd]), "attn_softmax_fwd");
                    /* rows sum to 1 */
                    for (int64_t b = 0; b < B; b++)
                        for (int64_t r = 0; r < rows; r++) {
                            double z = 0;
                            for (int64_t j = 0; j < cols; j++) z += got[b * bs + r * ld + j];
                            RT_CLOSE(z, 1.0, pd ? 0.02 : 1e-5, 0);
                        }
                    if (sd == pd) { /* in place */
                        RT_OK(rrc_attn_softmax_fwd(be, B, rows, cols, ld, bs, sb, 0.35f, off, causal, db, sb));
                        float *g2 = rt_download(sb, n);
                        int same = 1;
                        for (int64_t b = 0; b < B; b++)
                            for (int64_t r = 0; r < rows; r++)
                                for (int64_t j = 0; j < cols; j++)
                                    same &= g2[b * bs + r * ld + j] == got[b * bs + r * ld + j];
                        RT_EXPECT(same, "in-place softmax differs");
                        free(g2);
                    }
                    if (dp) {
                        RT_STATUS(rrc_attn_softmax_fwd(be, B, rows, cols, ld, bs, sb, 0.35f, off + 2, causal,
                                                       rrc_view(db, 0, 12), pb),
                                  RRC_ERR_SHAPE); /* doc_ids shorter than row_offset + rows */
                    }
                    RT_STATUS(rrc_attn_softmax_fwd(be, B, rows, cols, ld, (rows - 1) * ld + cols - 1, sb, 0.35f, off, causal, db, pb),
                              RRC_ERR_SHAPE);
                    RT_STATUS(rrc_attn_softmax_fwd(be, B, rows, cols, 11, bs, sb, 0.35f, off, causal, db, pb),
                              RRC_ERR_SHAPE);
                    free(got);
                    free(s);
                    free(want);
                    rrc_buf_free(&sb);
                    rrc_buf_free(&pb);
                    rrc_buf_free(&db);
                }
    /* a query with no allowed key (causal, cols <= q is impossible; use doc mismatch on all keys
     * by giving the query row a doc id no key in [0, cols) has) -> zero row */
    const rrc_backend *be = rrc_backend_cpu();
    int32_t d2[4] = {0, 0, 0, 1};
    float s[3] = {1, 2, 3};
    rrc_buf sb = rt_upload(be, RRC_F32, s, 3), db = rt_upload_i32(be, d2, 4), pb;
    RT_OK(rrc_buf_alloc(be, RRC_F32, 3, &pb));
    RT_OK(rrc_attn_softmax_fwd(be, 1, 1, 3, 3, 0, sb, 1.0f, 3, 1, db, pb));
    float *g = rt_download(pb, 3);
    RT_EXPECT(g[0] == 0 && g[1] == 0 && g[2] == 0, "no allowed key -> zeros");
    free(g);
    rrc_buf_free(&sb);
    rrc_buf_free(&db);
    rrc_buf_free(&pb);
}

/* ---- swiglu / elementwise ------------------------------------------------------------------- */

RT_TEST(swiglu_and_elementwise) {
    rt_rng rng = {61};
    for (int bi = 0; bi < 2; bi++)
        for (int dt = 0; dt < 2; dt++)
            for (int ot = 0; ot < 2; ot++) {
                const rrc_backend *be = rt_backend(bi);
                const int64_t R = 4, Cc = 9, ldi = 2 * 9; /* fused [gate | up] layout */
                float gu[72];
                rt_fill(&rng, gu, R * ldi, 6.0f);
                gu[0] = -100.0f; /* silu saturation */
                gu[1] = 100.0f;
                rt_round_inplace(gu, R * ldi, k_fdt[dt]);
                rrc_buf in = rt_upload(be, k_fdt[dt], gu, R * ldi), h;
                RT_OK(rrc_buf_alloc(be, k_fdt[ot], R * Cc, &h));
                RT_OK(rrc_swiglu_fwd(be, R, Cc, in, rrc_view_from(in, Cc), ldi, h, Cc));
                float *got = rt_download(h, R * Cc);
                double want[36];
                for (int64_t r = 0; r < R; r++)
                    for (int64_t c = 0; c < Cc; c++) {
                        double g = gu[r * ldi + c], u = gu[r * ldi + Cc + c];
                        want[r * Cc + c] = g / (1.0 + exp(-g)) * u;
                    }
                RT_ALLCLOSE(got, want, R * Cc, 1e-6, rtol_of(k_fdt[ot]), "swiglu_fwd");
                free(got);
                RT_STATUS(rrc_swiglu_fwd(be, R, Cc, in, rrc_view_from(in, Cc), ldi, rrc_view(in, 1, R * ldi - 1), ldi),
                          RRC_ERR_ALIAS);
                rrc_buf_free(&in);
                rrc_buf_free(&h);

                /* add / axpy / scale / convert */
                const int64_t n = 37;
                float a[37], b[37];
                rt_fill(&rng, a, n, 3.0f);
                rt_fill(&rng, b, n, 3.0f);
                rt_round_inplace(a, n, k_fdt[dt]);
                rt_round_inplace(b, n, k_fdt[ot]);
                rrc_buf ab = rt_upload(be, k_fdt[dt], a, n), bb = rt_upload(be, k_fdt[ot], b, n), ob;
                RT_OK(rrc_buf_alloc(be, k_fdt[ot], n, &ob));
                RT_OK(rrc_add(be, n, ab, bb, ob));
                double w[37];
                for (int64_t i = 0; i < n; i++) w[i] = (double)a[i] + b[i];
                got = rt_download(ob, n);
                RT_ALLCLOSE(got, w, n, 1e-7, rtol_of(k_fdt[ot]), "add");
                free(got);
                RT_OK(rrc_add(be, n, ab, bb, bb)); /* out aliases b */
                got = rt_download(bb, n);
                RT_ALLCLOSE(got, w, n, 1e-7, rtol_of(k_fdt[ot]), "add in place");
                free(got);
                RT_OK(rrc_axpy(be, n, -0.5f, ab, ob));
                for (int64_t i = 0; i < n; i++) w[i] = rrc_kc_round_to(k_fdt[ot], (float)((double)a[i] + b[i])) - 0.5 * a[i];
                got = rt_download(ob, n);
                RT_ALLCLOSE(got, w, n, 1e-6, rtol_of(k_fdt[ot]), "axpy");
                free(got);
                RT_OK(rrc_scale(be, n, 3.0f, ob));
                for (int64_t i = 0; i < n; i++) w[i] *= 3.0;
                got = rt_download(ob, n);
                RT_ALLCLOSE(got, w, n, 1e-5, 2 * rtol_of(k_fdt[ot]), "scale");
                free(got);
                RT_OK(rrc_convert(be, n, ab, ob));
                for (int64_t i = 0; i < n; i++) w[i] = rrc_kc_round_to(k_fdt[ot], a[i]);
                got = rt_download(ob, n);
                RT_ALLCLOSE(got, w, n, 0, 0, "convert");
                free(got);
                RT_STATUS(rrc_add(be, n - 1, ab, bb, rrc_view(ab, 1, n - 1)), RRC_ERR_ALIAS);
                RT_STATUS(rrc_axpy(be, n - 1, 1.0f, ab, rrc_view(ab, 1, n - 1)), RRC_ERR_ALIAS);
                RT_STATUS(rrc_convert(be, n - 1, ab, rrc_view(ab, 1, n - 1)), RRC_ERR_ALIAS);
                RT_OK(rrc_convert(be, n, ab, ab)); /* exact alias, same dtype: no-op */
                rrc_buf ib;
                RT_OK(rrc_buf_alloc(be, RRC_I32, n, &ib));
                RT_STATUS(rrc_convert(be, n, ab, ib), RRC_ERR_DTYPE);
                RT_STATUS(rrc_add(be, n, ab, ib, ob), RRC_ERR_DTYPE);
                RT_STATUS(rrc_add(be, n + 1, ab, bb, ob), RRC_ERR_SHAPE);
                rrc_buf_free(&ab);
                rrc_buf_free(&bb);
                rrc_buf_free(&ob);
                rrc_buf_free(&ib);
            }
}

RT_TEST(sum_squares_ref) {
    rt_rng rng = {71};
    for (int bi = 0; bi < 2; bi++)
        for (int dt = 0; dt < 2; dt++) {
            const rrc_backend *be = rt_backend(bi);
            const int64_t n = 50000;
            float *x = (float *)malloc((size_t)n * 4);
            rt_fill(&rng, x, n, 1.0f);
            rt_round_inplace(x, n, k_fdt[dt]);
            double want = 0;
            for (int64_t i = 0; i < n; i++) want += (double)x[i] * x[i];
            rrc_buf xb = rt_upload(be, k_fdt[dt], x, n), ob;
            RT_OK(rrc_buf_alloc(be, RRC_F32, 2, &ob));
            RT_OK(rrc_sum_squares(be, n, xb, rrc_view(ob, 1, 1), 0));
            RT_OK(rrc_sum_squares(be, n, xb, rrc_view(ob, 1, 1), 1));
            float got[2];
            RT_OK(rrc_buf_download(ob, got, 2));
            RT_CLOSE(got[1], 2 * want, 0, bi == 0 ? 1e-6 : 1e-4);
            RT_EXPECT(got[0] == 0.0f, "other slot untouched");
            RT_OK(rrc_sum_squares(be, 0, xb, ob, 0));
            RT_OK(rrc_buf_download(ob, got, 1));
            RT_EXPECT(got[0] == 0.0f, "empty sum");
            RT_STATUS(rrc_sum_squares(be, n, xb, xb, 0), dt ? RRC_ERR_DTYPE : RRC_ERR_ALIAS);
            free(x);
            rrc_buf_free(&xb);
            rrc_buf_free(&ob);
        }
}

/* ---- cross entropy -------------------------------------------------------------------------- */

RT_TEST(cross_entropy_ref) {
    rt_rng rng = {81};
    for (int bi = 0; bi < 2; bi++)
        for (int dt = 0; dt < 2; dt++)
            for (int with_rs = 0; with_rs < 2; with_rs++) {
                const rrc_backend *be = rt_backend(bi);
                const int64_t R = 6, V = 50, ld = 53;
                float z[6 * 53], rs[6];
                rt_fill(&rng, z, R * ld, 8.0f);
                rt_fill(&rng, rs, R, 1.0f);
                z[2 * ld + 7] = 60.0f; /* large logit (stability) */
                rt_round_inplace(z, R * ld, k_fdt[dt]);
                int32_t lab[6] = {3, RRC_IGNORE_INDEX, 7, 49, 0, 12};
                double wl[6], wg[6 * 53];
                const double gs = 0.125;
                for (int64_t r = 0; r < R; r++) {
                    double m = -INFINITY;
                    for (int64_t j = 0; j < V; j++) m = fmax(m, z[r * ld + j]);
                    double s = 0;
                    for (int64_t j = 0; j < V; j++) s += exp(z[r * ld + j] - m);
                    double lse = m + log(s);
                    wl[r] = lab[r] < 0 ? 0.0 : lse - z[r * ld + lab[r]];
                    double sc = gs * (with_rs ? rs[r] : 1.0);
                    for (int64_t j = 0; j < ld; j++)
                        wg[r * ld + j] = j >= V ? z[r * ld + j]
                                         : lab[r] < 0 ? 0.0
                                                      : (exp(z[r * ld + j] - lse) - (j == lab[r])) * sc;
                }
                rrc_buf zb = rt_upload(be, k_fdt[dt], z, R * ld), lb = rt_upload_i32(be, lab, R), lossb,
                        rsb = with_rs ? rt_upload(be, RRC_F32, rs, R) : rrc_buf_none();
                RT_OK(rrc_buf_alloc(be, RRC_F32, R, &lossb));
                RT_OK(rrc_cross_entropy_fwd_bwd(be, R, V, zb, ld, lb, lossb, (float)gs, rsb, 0));
                float *gl = rt_download(lossb, R);
                RT_ALLCLOSE(gl, wl, R, 1e-5, 1e-5, "xent loss (no grad)");
                float *gz = rt_download(zb, R * ld);
                double orig[6 * 53];
                for (int64_t i = 0; i < R * ld; i++) orig[i] = z[i];
                RT_ALLCLOSE(gz, orig, R * ld, 0, 0, "logits untouched without write_grad");
                free(gz);
                free(gl);
                RT_OK(rrc_cross_entropy_fwd_bwd(be, R, V, zb, ld, lb, lossb, (float)gs, rsb, 1));
                gl = rt_download(lossb, R);
                gz = rt_download(zb, R * ld);
                RT_ALLCLOSE(gl, wl, R, 1e-5, 1e-5, "xent loss");
                RT_ALLCLOSE(gz, wg, R * ld, 1e-7, dt ? 1.0 / 128 : 1e-5, "xent dlogits");
                /* each non-ignored gradient row sums to ~0 */
                for (int64_t r = 0; r < R; r++) {
                    double s = 0;
                    for (int64_t j = 0; j < V; j++) s += gz[r * ld + j];
                    RT_CLOSE(s, 0.0, dt ? 2e-3 : 1e-6, 0);
                }
                free(gl);
                free(gz);
                int32_t bad[1] = {50};
                rrc_buf bl = rt_upload_i32(be, bad, 1);
                RT_STATUS(rrc_cross_entropy_fwd_bwd(be, 1, V, zb, ld, bl, lossb, 1.0f, rrc_buf_none(), 1), RRC_ERR_RANGE);
                bad[0] = -5;
                RT_OK(rrc_buf_upload(bl, bad, 1));
                RT_STATUS(rrc_cross_entropy_fwd_bwd(be, 1, V, zb, ld, bl, lossb, 1.0f, rrc_buf_none(), 1), RRC_ERR_RANGE);
                RT_STATUS(rrc_cross_entropy_fwd_bwd(be, R, V, zb, ld, lb, rrc_view(zb, 0, R), 1.0f, rrc_buf_none(), 1),
                          dt ? RRC_ERR_DTYPE : RRC_ERR_ALIAS);
                RT_STATUS(rrc_cross_entropy_fwd_bwd(be, R, V, zb, V - 1, lb, lossb, 1.0f, rrc_buf_none(), 1),
                          RRC_ERR_SHAPE);
                rrc_buf_free(&zb);
                rrc_buf_free(&lb);
                rrc_buf_free(&lossb);
                rrc_buf_free(&rsb);
                rrc_buf_free(&bl);
            }
}

/* ---- adamw ---------------------------------------------------------------------------------- */

/* Literal transcription of torch/optim/adam.py::_single_tensor_adam (decoupled weight decay,
 * not capturable, not amsgrad, not maximize) for one fp32 element. Each Python-scalar operand is
 * rounded to float where it meets the fp32 tensor; lerp_ and addcmul_ are fused multiply-adds as
 * in ATen's vectorized CPU kernels (checked against torch itself in adamw_vs_torch below). */
static void torch_adamw_ref(float *param, float grad, float *exp_avg, float *exp_avg_sq, int64_t step, double lr,
                            double beta1, double beta2, double eps, double weight_decay) {
    if (weight_decay != 0) *param = *param * (float)(1 - lr * weight_decay); /* param.mul_(1 - lr * wd) */
    {                                                                         /* exp_avg.lerp_(grad, 1 - beta1) */
        float w = (float)(1 - beta1);
        float self = *exp_avg;
        *exp_avg = fabsf(w) < 0.5f ? fmaf(w, grad - self, self) : fmaf(w - 1.0f, grad - self, grad);
    }
    *exp_avg_sq = *exp_avg_sq * (float)beta2;                             /* exp_avg_sq.mul_(beta2) */
    *exp_avg_sq = fmaf((float)(1 - beta2) * grad, grad, *exp_avg_sq);    /* .addcmul_(grad, grad, 1 - beta2) */
    double bias_correction1 = 1 - pow(beta1, (double)step);
    double bias_correction2 = 1 - pow(beta2, (double)step);
    double step_size = lr / bias_correction1;
    double bias_correction2_sqrt = pow(bias_correction2, 0.5);
    float denom = sqrtf(*exp_avg_sq) / (float)bias_correction2_sqrt + (float)eps; /* (sqrt / bc2_sqrt).add_(eps) */
    *param = *param + ((float)(-step_size) * *exp_avg) / denom; /* param.addcdiv_(exp_avg, denom, value=-step_size) */
}

RT_TEST(adamw_torch_semantics) {
    rt_rng rng = {91};
    const int64_t n = 257;
    const double cfgs[4][5] = {
        {1e-3, 0.9, 0.999, 1e-8, 0.0}, {5e-2, 0.9, 0.95, 1e-6, 0.1}, {1e-2, 0.3, 0.99, 1e-8, 0.01}, {0.0, 0.9, 0.999, 1e-8, 0.5}};
    for (int bi = 0; bi < 2; bi++)
        for (int ci = 0; ci < 4; ci++) {
            const rrc_backend *be = rt_backend(bi);
            const double lr = cfgs[ci][0], b1 = cfgs[ci][1], b2 = cfgs[ci][2], eps = cfgs[ci][3], wd = cfgs[ci][4];
            float p[257], m[257], v[257], g[257];
            rt_fill(&rng, p, n, 1.0f);
            memset(m, 0, sizeof m);
            memset(v, 0, sizeof v);
            float rp[257], rm[257], rv[257];
            memcpy(rp, p, sizeof p);
            memcpy(rm, m, sizeof m);
            memcpy(rv, v, sizeof v);
            rrc_buf pb = rt_upload(be, RRC_F32, p, n), gb, mb = rt_upload(be, RRC_F32, m, n),
                    vb = rt_upload(be, RRC_F32, v, n);
            RT_OK(rrc_buf_alloc(be, RRC_F32, n, &gb));
            /* double-precision textbook AdamW for a tolerance check */
            double dp[257], dm[257] = {0}, dv[257] = {0};
            for (int64_t i = 0; i < n; i++) dp[i] = p[i];
            for (int64_t step = 1; step <= 5; step++) {
                rt_fill(&rng, g, n, step == 3 ? 100.0f : 0.5f);
                g[0] = 0.0f;
                float clip = step == 4 ? 0.25f : 1.0f;
                RT_OK(rrc_buf_upload(gb, g, n));
                rrc_adamw_hparams h = {lr, b1, b2, eps, wd, step, clip};
                RT_OK(rrc_adamw_step(be, n, pb, gb, mb, vb, &h));
                for (int64_t i = 0; i < n; i++) {
                    float gc = g[i] * clip;
                    torch_adamw_ref(&rp[i], gc, &rm[i], &rv[i], step, lr, b1, b2, eps, wd);
                    double gd = (double)gc;
                    dp[i] *= 1 - lr * wd;
                    dm[i] = b1 * dm[i] + (1 - b1) * gd;
                    dv[i] = b2 * dv[i] + (1 - b2) * gd * gd;
                    double mh = dm[i] / (1 - pow(b1, step)), vh = dv[i] / (1 - pow(b2, step));
                    dp[i] -= lr * mh / (sqrt(vh) + eps);
                }
            }
            float *gp = rt_download(pb, n), *gm = rt_download(mb, n), *gv = rt_download(vb, n);
            int exact = 1;
            for (int64_t i = 0; i < n; i++) exact &= gp[i] == rp[i] && gm[i] == rm[i] && gv[i] == rv[i];
            RT_EXPECT(exact, "adamw (%s, cfg %d) is not bit-identical to the torch-order reference", be->name, ci);
            RT_ALLCLOSE(gp, dp, n, 1e-6, 1e-5, "adamw vs double textbook formula");
            float *gg = rt_download(gb, n);
            RT_EXPECT(!memcmp(gg, g, sizeof g), "grad must be read-only");
            free(gg);
            free(gp);
            free(gm);
            free(gv);
            rrc_adamw_hparams h0 = {lr, b1, b2, eps, wd, 0, 1.0f};
            RT_STATUS(rrc_adamw_step(be, n, pb, gb, mb, vb, &h0), RRC_ERR_INVALID_ARG);
            rrc_adamw_hparams h1 = {lr, 1.0, b2, eps, wd, 1, 1.0f};
            RT_STATUS(rrc_adamw_step(be, n, pb, gb, mb, vb, &h1), RRC_ERR_INVALID_ARG);
            rrc_adamw_hparams h2 = {lr, b1, b2, eps, wd, 1, 1.0f};
            RT_STATUS(rrc_adamw_step(be, n, pb, pb, mb, vb, &h2), RRC_ERR_ALIAS);
            RT_STATUS(rrc_adamw_step(be, n, pb, gb, mb, rrc_view(vb, 1, n - 1), &h2), RRC_ERR_SHAPE);
            rrc_buf_free(&pb);
            rrc_buf_free(&gb);
            rrc_buf_free(&mb);
            rrc_buf_free(&vb);
        }
}

/* AdamW against torch.optim.AdamW(foreach=False) itself (python3 + torch + safetensors required;
 * skipped with a message otherwise): m and v must be bit-identical; p may differ where torch's
 * CPU kernels round the update term differently per build / SIMD path (bounded by 8 ulp of the
 * largest step change per step, plus 2 ulp of the value). */
static const char *k_py_adamw =
    "import sys, torch\n"
    "from safetensors.torch import save_file\n"
    "cfgs = [(1e-3, .9, .999, 1e-8, 0.0), (5e-2, .9, .95, 1e-6, .1), (1e-2, .3, .99, 1e-8, .01)]\n"
    "n, steps, out = 3000, 6, {}\n"
    "for ci, (lr, b1, b2, eps, wd) in enumerate(cfgs):\n"
    "    g0 = torch.Generator().manual_seed(ci)\n"
    "    p0 = torch.randn(n, generator=g0)\n"
    "    grads = torch.randn(steps, n, generator=g0) * 0.5\n"
    "    grads[2] *= 200\n"
    "    param = torch.nn.Parameter(p0.clone())\n"
    "    opt = torch.optim.AdamW([param], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd, foreach=False, fused=False)\n"
    "    for s in range(steps):\n"
    "        param.grad = grads[s].clone()\n"
    "        opt.step()\n"
    "    st = opt.state[param]\n"
    "    out[f'{ci}.p0'] = p0; out[f'{ci}.grads'] = grads; out[f'{ci}.p'] = param.detach().clone()\n"
    "    out[f'{ci}.m'] = st['exp_avg']; out[f'{ci}.v'] = st['exp_avg_sq']\n"
    "save_file(out, sys.argv[1])\n";

RT_TEST(adamw_vs_torch) {
    if (system("python3 -c 'import torch, safetensors.torch' >/dev/null 2>&1") != 0) {
        rt_skip("python3 with torch + safetensors not available: AdamW-vs-torch sub-test skipped");
        return;
    }
    const char *script = rt_path("adamw_ref.py"), *file = rt_path("adamw_ref.safetensors");
    FILE *f = fopen(script, "wb");
    RT_ASSERT(f, "cannot write %s", script);
    fputs(k_py_adamw, f);
    fclose(f);
    char cmd[4096];
    snprintf(cmd, sizeof cmd, "python3 -I '%s' '%s'", script, file);
    RT_ASSERT(system(cmd) == 0, "python AdamW reference failed: %s", cmd);
    rrc_st_set *set = NULL;
    RT_OK(rrc_st_open_file(file, &set));
    const double cfgs[3][5] = {{1e-3, .9, .999, 1e-8, 0.0}, {5e-2, .9, .95, 1e-6, .1}, {1e-2, .3, .99, 1e-8, .01}};
    const int64_t n = 3000, steps = 6;
    float *buf = (float *)malloc((size_t)(steps * n) * 4), *want[3];
    for (int bi = 0; bi < 2; bi++)
        for (int ci = 0; ci < 3; ci++) {
            const rrc_backend *be = rt_backend(bi);
            char key[32];
            snprintf(key, sizeof key, "%d.p0", ci);
            RT_OK(rrc_st_to_f32(rrc_st_find(set, key), buf, n));
            rrc_buf p = rt_upload(be, RRC_F32, buf, n), g, m, v;
            RT_OK(rrc_buf_alloc(be, RRC_F32, n, &g));
            RT_OK(rrc_buf_alloc(be, RRC_F32, n, &m));
            RT_OK(rrc_buf_alloc(be, RRC_F32, n, &v));
            snprintf(key, sizeof key, "%d.grads", ci);
            RT_OK(rrc_st_to_f32(rrc_st_find(set, key), buf, steps * n));
            /* largest single-step parameter change of this run: the scale of torch's rounding
             * differences in the update term (its vectorized sqrt / division differ by CPU) */
            double max_upd = 0;
            float *prev = rt_download(p, n);
            for (int64_t s = 0; s < steps; s++) {
                RT_OK(rrc_buf_upload(g, buf + s * n, n));
                rrc_adamw_hparams h = {cfgs[ci][0], cfgs[ci][1], cfgs[ci][2], cfgs[ci][3], cfgs[ci][4], s + 1, 1.0f};
                RT_OK(rrc_adamw_step(be, n, p, g, m, v, &h));
                float *cur = rt_download(p, n);
                for (int64_t i = 0; i < n; i++) max_upd = fmax(max_upd, fabs((double)cur[i] - prev[i]));
                free(prev);
                prev = cur;
            }
            free(prev);
            const char *names[3] = {"p", "m", "v"};
            rrc_buf bufs[3] = {p, m, v};
            for (int k = 0; k < 3; k++) {
                want[k] = (float *)malloc((size_t)n * 4);
                snprintf(key, sizeof key, "%d.%s", ci, names[k]);
                RT_OK(rrc_st_to_f32(rrc_st_find(set, key), want[k], n));
                float *got = rt_download(bufs[k], n);
                int64_t mism = 0, over = 0;
                double maxrel = 0, worst = 0;
                for (int64_t i = 0; i < n; i++) {
                    if (got[i] != want[k][i]) mism++;
                    double d = fabs((double)got[i] - want[k][i]);
                    maxrel = fmax(maxrel, d / (fabs(want[k][i]) + 1e-30));
                    /* p: 2 ulp of the value plus 8 ulp of the largest step change per step. torch's
                     * CPU kernels round sqrt / division differently per SIMD path (the PyPI CUDA
                     * wheel and the +cpu wheel on a GitHub runner differ), so p is not bit-exact;
                     * m and v below must be. */
                    double tol = 2.0 * ldexp(fabs((double)want[k][i]), -23) + (double)steps * 8.0 * ldexp(max_upd, -24);
                    if (d > tol) over++;
                    worst = fmax(worst, d / tol);
                }
                if (k == 0)
                    RT_EXPECT(over == 0,
                              "%s cfg %d: param differs from torch beyond the rounding bound in %lld of %lld elements "
                              "(%lld differ at all, max rel %.3g, worst/tol %.3g)",
                              be->name, ci, (long long)over, (long long)n, (long long)mism, maxrel, worst);
                else
                    RT_EXPECT(mism == 0, "%s cfg %d: %s differs from torch in %lld elements (max rel %.3g)", be->name,
                              ci, names[k], (long long)mism, maxrel);
                free(got);
                free(want[k]);
            }
            rrc_buf_free(&p);
            rrc_buf_free(&g);
            rrc_buf_free(&m);
            rrc_buf_free(&v);
        }
    free(buf);
    rrc_st_close(set);
}

/* ---- determinism ---------------------------------------------------------------------------- */

/* Every op with sizes above the CPU backend's parallel threshold (PAR_MIN = 16384 elements), so
 * each OpenMP loop really runs multi-threaded; outputs must be bit-identical for 1 and 4 threads. */
#define DET_MAX_OUT 24
typedef struct {
    float *v[DET_MAX_OUT];
    int64_t n[DET_MAX_OUT];
    int count;
} det_outs;

static void det_keep(det_outs *o, rrc_buf b, int64_t n) {
    o->v[o->count] = rt_download(b, n);
    o->n[o->count] = n;
    o->count++;
}

static void run_det_ops(const rrc_backend *be, det_outs *o) {
    rt_rng rng = {101};
    const int64_t M = 160, N = 128, K = 200, H = 8, HD = 16, V = 300, NP = 200;
    o->count = 0;
    float *A = (float *)malloc((size_t)(M * K) * 4), *B = (float *)malloc((size_t)(N * K) * 4);
    float *T = (float *)malloc((size_t)(V * N) * 4), *X = (float *)malloc((size_t)(M * N) * 4);
    rt_fill(&rng, A, M * K, 1.0f);
    rt_fill(&rng, B, N * K, 1.0f);
    rt_fill(&rng, T, V * N, 1.0f);
    rt_fill(&rng, X, M * N, 2.0f);
    rrc_buf a = rt_upload(be, RRC_F32, A, M * K), b = rt_upload(be, RRC_BF16, B, N * K), tab = rt_upload(be, RRC_F32, T, V * N),
            x = rt_upload(be, RRC_F32, X, M * N), x2 = rt_upload(be, RRC_BF16, X, M * N);
    rrc_buf c, ss, p, ds, w, y, dxn, lab, loss, emb, ids, db, pos, cs, sn, rq, h, dg, du, sum, cv, pm, gm, mm, vm;
    /* gemm (packing + tiles) */
    rrc_buf_alloc(be, RRC_F32, M * N, &c);
    rrc_gemm2d(be, RRC_OP_N, RRC_OP_T, M, N, K, 1.0f, a, K, b, K, 0.0f, c, N);
    det_keep(o, c, M * N);
    /* sum of squares */
    rrc_buf_alloc(be, RRC_F32, 1, &ss);
    rrc_sum_squares(be, M * K, a, ss, 0);
    det_keep(o, ss, 1);
    /* attention softmax fwd / bwd */
    rrc_buf_alloc(be, RRC_F32, M * N, &p);
    rrc_attn_softmax_fwd(be, 1, M, N, N, 0, c, 0.1f, 0, 1, rrc_buf_none(), p);
    det_keep(o, p, M * N);
    rrc_buf_alloc(be, RRC_F32, M * N, &ds);
    rrc_attn_softmax_bwd(be, 1, M, N, N, 0, p, x, 0.1f, ds);
    det_keep(o, ds, M * N);
    /* rmsnorm fwd / bwd */
    float wv[200];
    rt_fill(&rng, wv, K, 1.0f);
    w = rt_upload(be, RRC_F32, wv, K);
    rrc_buf_alloc(be, RRC_F32, M * K, &y);
    rrc_rmsnorm_fwd(be, M, K, K, a, w, 1e-5f, y);
    det_keep(o, y, M * K);
    rrc_buf_alloc(be, RRC_F32, M * K, &dxn);
    rrc_rmsnorm_bwd(be, M, K, K, a, w, 1e-5f, y, dxn, 0);
    det_keep(o, dxn, M * K);
    /* embedding, bias */
    int32_t idv[160];
    for (int i = 0; i < 160; i++) idv[i] = (i * 37) % (int)V;
    ids = rt_upload_i32(be, idv, M);
    rrc_buf_alloc(be, RRC_BF16, M * N, &emb);
    rrc_embedding_fwd(be, M, N, V, ids, tab, emb);
    det_keep(o, emb, M * N);
    rrc_add_bias(be, M, N, emb, N, rrc_view(tab, 0, N));
    det_keep(o, emb, M * N);
    rrc_buf_alloc(be, RRC_F32, N, &db);
    rrc_bias_grad(be, M, N, x, N, db, 0);
    det_keep(o, db, N);
    /* rope fwd / bwd: M tokens x 8 heads x 16 */
    rrc_model_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.head_dim = HD;
    cfg.rope_theta = 10000.0;
    cfg.rope_attention_scaling = 1.0;
    float *ct = (float *)malloc((size_t)(NP * HD / 2) * 4), *st = (float *)malloc((size_t)(NP * HD / 2) * 4);
    rrc_rope_tables(&cfg, NP, ct, st);
    cs = rt_upload(be, RRC_F32, ct, NP * HD / 2);
    sn = rt_upload(be, RRC_F32, st, NP * HD / 2);
    int32_t pv[160];
    for (int i = 0; i < 160; i++) pv[i] = (i * 13) % (int)NP;
    pos = rt_upload_i32(be, pv, M);
    rrc_buf_alloc(be, RRC_F32, M * H * HD, &rq);
    rrc_rope_fwd(be, M, H, HD, H * HD, x, pos, cs, sn, NP, rq);
    det_keep(o, rq, M * H * HD);
    rrc_rope_bwd(be, M, H, HD, H * HD, rq, pos, cs, sn, NP, rq);
    det_keep(o, rq, M * H * HD);
    /* swiglu fwd / bwd */
    rrc_buf_alloc(be, RRC_F32, M * N, &h);
    rrc_swiglu_fwd(be, M, N, x, x2, N, h, N);
    det_keep(o, h, M * N);
    rrc_buf_alloc(be, RRC_F32, M * N, &dg);
    rrc_buf_alloc(be, RRC_BF16, M * N, &du);
    rrc_swiglu_bwd(be, M, N, x, x2, N, h, N, dg, du, N);
    det_keep(o, dg, M * N);
    det_keep(o, du, M * N);
    /* elementwise */
    rrc_buf_alloc(be, RRC_F32, M * N, &sum);
    rrc_add(be, M * N, x, x2, sum);
    rrc_axpy(be, M * N, 0.3f, x2, sum);
    rrc_scale(be, M * N, -1.5f, sum);
    det_keep(o, sum, M * N);
    rrc_buf_alloc(be, RRC_BF16, M * N, &cv);
    rrc_convert(be, M * N, sum, cv);
    det_keep(o, cv, M * N);
    /* cross entropy */
    int32_t labels[160];
    for (int i = 0; i < 160; i++) labels[i] = i % 11 == 0 ? RRC_IGNORE_INDEX : (i * 7) % (int)N;
    lab = rt_upload_i32(be, labels, M);
    rrc_buf_alloc(be, RRC_F32, M, &loss);
    rrc_cross_entropy_fwd_bwd(be, M, N, c, N, lab, loss, 0.5f, rrc_buf_none(), 1);
    det_keep(o, loss, M);
    det_keep(o, c, M * N);
    /* adamw */
    pm = rt_upload(be, RRC_F32, A, M * K);
    gm = rt_upload(be, RRC_F32, A + 1, M * K - 1);
    rrc_buf_alloc(be, RRC_F32, M * K, &mm);
    rrc_buf_alloc(be, RRC_F32, M * K, &vm);
    rrc_adamw_hparams hp = {1e-3, 0.9, 0.999, 1e-8, 0.01, 1, 1.0f};
    rrc_adamw_step(be, M * K - 1, pm, gm, mm, vm, &hp);
    det_keep(o, pm, M * K);
    det_keep(o, vm, M * K);
    rrc_buf *all[] = {&a, &b, &tab, &x, &x2, &c, &ss, &p, &ds, &w, &y, &dxn, &lab, &loss, &emb, &ids, &db, &pos,
                      &cs, &sn, &rq, &h, &dg, &du, &sum, &cv, &pm, &gm, &mm, &vm};
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) rrc_buf_free(all[i]);
    free(A);
    free(B);
    free(T);
    free(X);
    free(ct);
    free(st);
}

RT_TEST(determinism_across_thread_counts) {
    int saved = omp_get_max_threads();
    for (int bi = 0; bi < 2; bi++) {
        det_outs r1, r4;
        omp_set_num_threads(1);
        run_det_ops(rt_backend(bi), &r1);
        omp_set_num_threads(4);
        run_det_ops(rt_backend(bi), &r4);
        RT_EXPECT(r1.count == r4.count && r1.count == 20, "output count %d", r1.count);
        for (int k = 0; k < r1.count; k++) {
            int nonzero = 0;
            for (int64_t i = 0; i < r1.n[k]; i++) nonzero |= r1.v[k][i] != 0.0f;
            RT_EXPECT(nonzero, "%s: output %d is all zeros (op did not run?)", rt_backend(bi)->name, k);
            RT_EXPECT(!memcmp(r1.v[k], r4.v[k], (size_t)r1.n[k] * 4), "%s: output %d differs between 1 and 4 threads",
                      rt_backend(bi)->name, k);
            free(r1.v[k]);
            free(r4.v[k]);
        }
    }
    omp_set_num_threads(saved);
}

RT_TEST(buffers_and_views) {
    const rrc_backend *be = rrc_backend_cpu();
    rrc_buf b;
    RT_OK(rrc_buf_alloc(be, RRC_BF16, 10, &b));
    rrc_bf16 z[10];
    RT_OK(rrc_buf_download(b, z, 10));
    int zero = 1;
    for (int i = 0; i < 10; i++) zero &= z[i] == 0;
    RT_EXPECT(zero, "alloc zero-initialises");
    rrc_buf v = rrc_view(b, 4, 6);
    RT_EXPECT(v.ptr == (char *)b.ptr + 8 && v.numel == 6 && !v.owned, "view");
    RT_EXPECT(rrc_view(b, 4, 7).numel == -1 && rrc_view(b, -1, 1).numel == -1 && rrc_view(b, 11, 0).numel == -1,
              "out-of-range views are invalid");
    RT_EXPECT(rrc_view(b, 10, 0).numel == 0, "empty view at end");
    RT_EXPECT(rrc_view_from(b, 3).numel == 7, "view_from");
    RT_STATUS(rrc_buf_free(&v), RRC_ERR_INVALID_ARG);
    RT_STATUS(rrc_buf_upload(b, z, 11), RRC_ERR_SHAPE);
    RT_STATUS(rrc_buf_copy(rrc_view(b, 1, 5), rrc_view(b, 0, 5), 5), RRC_ERR_ALIAS);
    RT_OK(rrc_buf_copy(rrc_view(b, 5, 5), rrc_view(b, 0, 5), 5));
    rrc_buf f;
    RT_OK(rrc_buf_alloc(be, RRC_F32, 5, &f));
    RT_STATUS(rrc_buf_copy(f, rrc_view(b, 0, 5), 5), RRC_ERR_DTYPE);
    RT_OK(rrc_buf_zero(rrc_view(b, 2, 3), 3));
    RT_OK(rrc_buf_free(&f));
    RT_STATUS(rrc_buf_alloc(be, (rrc_dtype)9, 5, &f), RRC_ERR_DTYPE);
    RT_OK(rrc_buf_free(&b));
    RT_EXPECT(rrc_buf_is_none(b), "freed buffer reset");
    RT_OK(rrc_buf_free(&b)); /* freeing none is fine */
    RT_OK(rrc_sync(be));
    RT_EXPECT(!strcmp(rrc_backend_cpu()->name, "cpu") && !strcmp(rrc_backend_cpu_f32acc()->name, "cpu-f32acc"),
              "names");
}

const rt_case rt_suite_ops[] = {
    RT_CASE(gemm_all_transposes_and_dtypes),
    RT_CASE(gemm_batched_broadcast_interleaved),
    RT_CASE(gemm_preconditions),
    RT_CASE(embedding_and_bias),
    RT_CASE(rmsnorm_fwd_ref),
    RT_CASE(rope_fwd_ref),
    RT_CASE(attn_softmax_fwd_ref),
    RT_CASE(swiglu_and_elementwise),
    RT_CASE(sum_squares_ref),
    RT_CASE(cross_entropy_ref),
    RT_CASE(adamw_torch_semantics),
    RT_CASE(adamw_vs_torch),
    RT_CASE(determinism_across_thread_counts),
    RT_CASE(buffers_and_views),
};
const size_t rt_suite_ops_n = sizeof rt_suite_ops / sizeof rt_suite_ops[0];
