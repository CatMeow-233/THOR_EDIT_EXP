/* Backward ops against central finite differences of double-precision forward references:
 * for an objective f(x) = <fwd_ref(x), dy> computed in double, every element of the op's
 * gradient (computed in f32 by the backend) is compared with (f(x + h e_i) - f(x - h e_i)) / 2h.
 * Inputs may be bf16 (the differences are taken at the bf16-rounded values). */
#include "ops_util.h"
#include "rrc/config.h"
#include "rrc_kc.h"

typedef double (*objective)(const double *x, void *ctx);

/* Elementwise central differences of f at x (n values) into g. */
static void fd_grad(objective f, void *ctx, double *x, int64_t n, double h, double *g) {
    for (int64_t i = 0; i < n; i++) {
        double x0 = x[i];
        x[i] = x0 + h;
        double fp = f(x, ctx);
        x[i] = x0 - h;
        double fm = f(x, ctx);
        x[i] = x0;
        g[i] = (fp - fm) / (2 * h);
    }
}

static const rrc_dtype k_fdt[2] = {RRC_F32, RRC_BF16};

/* ---- rmsnorm ---------------------------------------------------------------------------------- */

typedef struct {
    int64_t rows, dim;
    const double *w, *dy;
    double eps;
} rms_ctx;

static double rms_obj(const double *x, void *vc) {
    const rms_ctx *c = (const rms_ctx *)vc;
    double f = 0;
    for (int64_t r = 0; r < c->rows; r++) {
        double s = 0;
        for (int64_t i = 0; i < c->dim; i++) s += x[r * c->dim + i] * x[r * c->dim + i];
        double rstd = 1.0 / sqrt(s / c->dim + c->eps);
        for (int64_t i = 0; i < c->dim; i++) f += c->w[i] * x[r * c->dim + i] * rstd * c->dy[r * c->dim + i];
    }
    return f;
}

RT_TEST(grad_rmsnorm) {
    rt_rng rng = {201};
    for (int bi = 0; bi < 2; bi++)
        for (int xd = 0; xd < 2; xd++)
            for (int acc = 0; acc < 2; acc++) {
                const rrc_backend *be = rt_backend(bi);
                const int64_t R = 3, D = 10, n = R * D;
                float x[30], w[10], dy[30], dx0[30];
                rt_fill(&rng, x, n, 2.0f);
                rt_fill(&rng, w, D, 1.5f);
                rt_fill(&rng, dy, n, 1.0f);
                rt_fill(&rng, dx0, n, 1.0f);
                rt_round_inplace(x, n, k_fdt[xd]);
                rt_round_inplace(w, D, k_fdt[xd]);
                rt_round_inplace(dy, n, k_fdt[xd]);
                double xd_[30], wd[10], dyd[30], g[30];
                for (int64_t i = 0; i < n; i++) xd_[i] = x[i], dyd[i] = dy[i];
                for (int64_t i = 0; i < D; i++) wd[i] = w[i];
                rms_ctx c = {R, D, wd, dyd, 1e-6};
                fd_grad(rms_obj, &c, xd_, n, 1e-5, g);
                if (acc)
                    for (int64_t i = 0; i < n; i++) g[i] += dx0[i];
                rrc_buf xb = rt_upload(be, k_fdt[xd], x, n), wb = rt_upload(be, k_fdt[xd], w, D),
                        dyb = rt_upload(be, k_fdt[xd], dy, n), dxb = rt_upload(be, RRC_F32, dx0, n);
                RT_OK(rrc_rmsnorm_bwd(be, R, D, D, xb, wb, 1e-6f, dyb, dxb, acc));
                float *got = rt_download(dxb, n);
                RT_ALLCLOSE(got, g, n, 1e-5, 1e-4, "rmsnorm_bwd vs finite differences");
                free(got);
                if (!acc && xd == 0) { /* in place: dx aliases dy */
                    RT_OK(rrc_rmsnorm_bwd(be, R, D, D, xb, wb, 1e-6f, dyb, dyb, 0));
                    got = rt_download(dyb, n);
                    RT_ALLCLOSE(got, g, n, 1e-5, 1e-4, "rmsnorm_bwd in place");
                    free(got);
                }
                rrc_buf_free(&xb);
                rrc_buf_free(&wb);
                rrc_buf_free(&dyb);
                rrc_buf_free(&dxb);
            }
}

/* ---- rope ------------------------------------------------------------------------------------- */

typedef struct {
    int64_t T, H, hd;
    const int32_t *pos;
    const float *cs, *sn;
    const double *dy;
} rope_ctx;

static double rope_obj(const double *x, void *vc) {
    const rope_ctx *c = (const rope_ctx *)vc;
    const int64_t h2 = c->hd / 2;
    double f = 0;
    for (int64_t t = 0; t < c->T; t++)
        for (int64_t h = 0; h < c->H; h++)
            for (int64_t i = 0; i < h2; i++) {
                int64_t o = t * c->H * c->hd + h * c->hd;
                double cc = c->cs[c->pos[t] * h2 + i], ss = c->sn[c->pos[t] * h2 + i];
                double a = x[o + i], b = x[o + i + h2];
                f += (a * cc - b * ss) * c->dy[o + i] + (b * cc + a * ss) * c->dy[o + i + h2];
            }
    return f;
}

RT_TEST(grad_rope) {
    rt_rng rng = {211};
    rrc_model_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.head_dim = 6;
    cfg.rope_theta = 500.0;
    cfg.rope_attention_scaling = 1.0;
    const int64_t NP = 9, T = 4, H = 2, hd = 6, n = T * H * hd;
    float cs[27], sn[27];
    RT_OK(rrc_rope_tables(&cfg, NP, cs, sn));
    int32_t pos[4] = {8, 0, 3, 3};
    for (int bi = 0; bi < 2; bi++)
        for (int dt = 0; dt < 2; dt++) {
            const rrc_backend *be = rt_backend(bi);
            float x[48], dy[48];
            rt_fill(&rng, x, n, 1.0f);
            rt_fill(&rng, dy, n, 1.0f);
            rt_round_inplace(dy, n, k_fdt[dt]);
            double xd[48], dyd[48], g[48];
            for (int64_t i = 0; i < n; i++) xd[i] = x[i], dyd[i] = dy[i];
            rope_ctx c = {T, H, hd, pos, cs, sn, dyd};
            fd_grad(rope_obj, &c, xd, n, 1e-5, g);
            rrc_buf dyb = rt_upload(be, k_fdt[dt], dy, n), pb = rt_upload_i32(be, pos, T),
                    cb = rt_upload(be, RRC_F32, cs, NP * 3), sb = rt_upload(be, RRC_F32, sn, NP * 3), dxb;
            RT_OK(rrc_buf_alloc(be, RRC_F32, n, &dxb));
            RT_OK(rrc_rope_bwd(be, T, H, hd, H * hd, dyb, pb, cb, sb, NP, dxb));
            float *got = rt_download(dxb, n);
            RT_ALLCLOSE(got, g, n, 1e-6, 1e-5, "rope_bwd vs finite differences");
            free(got);
            rrc_buf_free(&dyb);
            rrc_buf_free(&pb);
            rrc_buf_free(&cb);
            rrc_buf_free(&sb);
            rrc_buf_free(&dxb);
        }
}

/* ---- attention softmax ------------------------------------------------------------------------- */

typedef struct {
    int64_t B, rows, cols, off;
    double scale;
    int causal;
    const int32_t *doc;
    const double *dp;
} sm_ctx;

static double sm_obj(const double *s, void *vc) {
    const sm_ctx *c = (const sm_ctx *)vc;
    double f = 0;
    for (int64_t b = 0; b < c->B; b++)
        for (int64_t r = 0; r < c->rows; r++) {
            const double *row = s + (b * c->rows + r) * c->cols;
            int64_t q = c->off + r;
            double m = -INFINITY, z = 0;
            for (int64_t j = 0; j < c->cols; j++)
                if (rrc_kc_attn_allowed(j, q, c->causal, c->doc)) m = fmax(m, c->scale * row[j]);
            for (int64_t j = 0; j < c->cols; j++)
                if (rrc_kc_attn_allowed(j, q, c->causal, c->doc)) z += exp(c->scale * row[j] - m);
            for (int64_t j = 0; j < c->cols; j++)
                if (rrc_kc_attn_allowed(j, q, c->causal, c->doc))
                    f += exp(c->scale * row[j] - m) / z * c->dp[(b * c->rows + r) * c->cols + j];
        }
    return f;
}

RT_TEST(grad_attn_softmax) {
    rt_rng rng = {221};
    int32_t doc[9] = {0, 0, 0, 1, 1, 1, 1, 2, 2};
    for (int bi = 0; bi < 2; bi++)
        for (int dt = 0; dt < 2; dt++)
            for (int mode = 0; mode < 2; mode++) {
                const rrc_backend *be = rt_backend(bi);
                const int64_t B = 2, rows = 4, cols = 9, off = 3, n = B * rows * cols;
                float s[72], dp[72];
                rt_fill(&rng, s, n, 3.0f);
                rt_fill(&rng, dp, n, 1.0f);
                rt_round_inplace(s, n, k_fdt[dt]);
                rt_round_inplace(dp, n, k_fdt[dt]);
                double sd[72], dpd[72], g[72];
                for (int64_t i = 0; i < n; i++) sd[i] = s[i], dpd[i] = dp[i];
                sm_ctx c = {B, rows, cols, off, 0.4, 1, mode ? doc : NULL, dpd};
                fd_grad(sm_obj, &c, sd, n, 1e-5, g);
                rrc_buf sb = rt_upload(be, k_fdt[dt], s, n), dpb = rt_upload(be, k_fdt[dt], dp, n), pb, dsb,
                        db = mode ? rt_upload_i32(be, doc, 9) : rrc_buf_none();
                RT_OK(rrc_buf_alloc(be, RRC_F32, n, &pb));
                RT_OK(rrc_buf_alloc(be, RRC_F32, n, &dsb));
                RT_OK(rrc_attn_softmax_fwd(be, B, rows, cols, cols, rows * cols, sb, 0.4f, off, 1, db, pb));
                RT_OK(rrc_attn_softmax_bwd(be, B, rows, cols, cols, rows * cols, pb, dpb, 0.4f, dsb));
                float *got = rt_download(dsb, n);
                RT_ALLCLOSE(got, g, n, 1e-6, 1e-4, "attn_softmax_bwd vs finite differences");
                free(got);
                if (dt == 0) { /* in place: ds aliases dp */
                    RT_OK(rrc_attn_softmax_bwd(be, B, rows, cols, cols, rows * cols, pb, dpb, 0.4f, dpb));
                    got = rt_download(dpb, n);
                    RT_ALLCLOSE(got, g, n, 1e-6, 1e-4, "attn_softmax_bwd in place");
                    free(got);
                }
                rrc_buf_free(&sb);
                rrc_buf_free(&dpb);
                rrc_buf_free(&pb);
                rrc_buf_free(&dsb);
                rrc_buf_free(&db);
            }
}

/* ---- swiglu -------------------------------------------------------------------------------------- */

typedef struct {
    int64_t n;
    const double *dh;
} sw_ctx;

/* x = [gate (n) | up (n)] */
static double sw_obj(const double *x, void *vc) {
    const sw_ctx *c = (const sw_ctx *)vc;
    double f = 0;
    for (int64_t i = 0; i < c->n; i++) {
        double g = x[i], u = x[c->n + i];
        f += g / (1 + exp(-g)) * u * c->dh[i];
    }
    return f;
}

RT_TEST(grad_swiglu) {
    rt_rng rng = {231};
    for (int bi = 0; bi < 2; bi++)
        for (int dt = 0; dt < 2; dt++) {
            const rrc_backend *be = rt_backend(bi);
            const int64_t R = 3, Cc = 7, n = R * Cc;
            float g[21], u[21], dh[21];
            rt_fill(&rng, g, n, 6.0f);
            rt_fill(&rng, u, n, 2.0f);
            rt_fill(&rng, dh, n, 1.0f);
            g[0] = -30.0f;
            g[1] = 25.0f;
            rt_round_inplace(g, n, k_fdt[dt]);
            rt_round_inplace(u, n, k_fdt[dt]);
            rt_round_inplace(dh, n, k_fdt[dt]);
            double x[42], dhd[21], gr[42];
            for (int64_t i = 0; i < n; i++) x[i] = g[i], x[n + i] = u[i], dhd[i] = dh[i];
            sw_ctx c = {n, dhd};
            fd_grad(sw_obj, &c, x, 2 * n, 1e-5, gr);
            rrc_buf gb = rt_upload(be, k_fdt[dt], g, n), ub = rt_upload(be, k_fdt[dt], u, n),
                    dhb = rt_upload(be, k_fdt[dt], dh, n), dgb, dub;
            RT_OK(rrc_buf_alloc(be, RRC_F32, n, &dgb));
            RT_OK(rrc_buf_alloc(be, RRC_F32, n, &dub));
            RT_OK(rrc_swiglu_bwd(be, R, Cc, gb, ub, Cc, dhb, Cc, dgb, dub, Cc));
            float *gg = rt_download(dgb, n), *gu = rt_download(dub, n);
            RT_ALLCLOSE(gg, gr, n, 1e-6, 1e-4, "swiglu_bwd dgate vs finite differences");
            RT_ALLCLOSE(gu, gr + n, n, 1e-6, 1e-4, "swiglu_bwd dup vs finite differences");
            free(gg);
            free(gu);
            if (dt == 0) { /* in place: dgate -> gate, dup -> up */
                RT_OK(rrc_swiglu_bwd(be, R, Cc, gb, ub, Cc, dhb, Cc, gb, ub, Cc));
                gg = rt_download(gb, n);
                gu = rt_download(ub, n);
                RT_ALLCLOSE(gg, gr, n, 1e-6, 1e-4, "swiglu_bwd in place dgate");
                RT_ALLCLOSE(gu, gr + n, n, 1e-6, 1e-4, "swiglu_bwd in place dup");
                free(gg);
                free(gu);
                RT_STATUS(rrc_swiglu_bwd(be, R, Cc, gb, ub, Cc, dhb, Cc, dgb, rrc_view(dgb, 1, n - 1), Cc - 1),
                          RRC_ERR_SHAPE);
                RT_STATUS(rrc_swiglu_bwd(be, R, Cc, gb, ub, Cc, dhb, Cc, dgb, dgb, Cc), RRC_ERR_ALIAS);
                RT_STATUS(rrc_swiglu_bwd(be, R, Cc, gb, ub, Cc, dhb, Cc, ub, gb, Cc), RRC_ERR_ALIAS);
            }
            rrc_buf_free(&gb);
            rrc_buf_free(&ub);
            rrc_buf_free(&dhb);
            rrc_buf_free(&dgb);
            rrc_buf_free(&dub);
        }
}

/* ---- cross entropy ------------------------------------------------------------------------------- */

typedef struct {
    int64_t R, V;
    const int32_t *lab;
    const double *scale; /* per row: grad_scale * row_scale */
} ce_ctx;

static double ce_obj(const double *z, void *vc) {
    const ce_ctx *c = (const ce_ctx *)vc;
    double f = 0;
    for (int64_t r = 0; r < c->R; r++) {
        if (c->lab[r] < 0) continue;
        const double *row = z + r * c->V;
        double m = -INFINITY, s = 0;
        for (int64_t j = 0; j < c->V; j++) m = fmax(m, row[j]);
        for (int64_t j = 0; j < c->V; j++) s += exp(row[j] - m);
        f += c->scale[r] * (m + log(s) - row[c->lab[r]]);
    }
    return f;
}

RT_TEST(grad_cross_entropy) {
    rt_rng rng = {241};
    for (int bi = 0; bi < 2; bi++)
        for (int dt = 0; dt < 2; dt++) {
            const rrc_backend *be = rt_backend(bi);
            const int64_t R = 4, V = 13, n = R * V;
            float z[52], rs[4] = {1.0f, 0.5f, 2.0f, 1.25f};
            rt_fill(&rng, z, n, 5.0f);
            rt_round_inplace(z, n, k_fdt[dt]);
            int32_t lab[4] = {2, RRC_IGNORE_INDEX, 12, 0};
            const float gs = 1.0f / 3.0f;
            double zd[52], g[52], sc[4];
            for (int64_t i = 0; i < n; i++) zd[i] = z[i];
            for (int64_t r = 0; r < R; r++) sc[r] = (double)(gs * rs[r]);
            ce_ctx c = {R, V, lab, sc};
            fd_grad(ce_obj, &c, zd, n, 1e-5, g);
            rrc_buf zb = rt_upload(be, RRC_F32, z, n), lb = rt_upload_i32(be, lab, R), rsb = rt_upload(be, RRC_F32, rs, R),
                    lossb;
            RT_OK(rrc_buf_alloc(be, RRC_F32, R, &lossb));
            RT_OK(rrc_cross_entropy_fwd_bwd(be, R, V, zb, V, lb, lossb, gs, rsb, 1));
            float *got = rt_download(zb, n);
            RT_ALLCLOSE(got, g, n, 1e-6, 1e-4, "cross_entropy dlogits vs finite differences");
            free(got);
            rrc_buf_free(&zb);
            rrc_buf_free(&lb);
            rrc_buf_free(&rsb);
            rrc_buf_free(&lossb);
        }
}

/* ---- gemm transposes ------------------------------------------------------------------------------ */

typedef struct {
    rrc_trans ta, tb;
    int64_t m, n, k;
    double alpha;
    const double *other; /* the matrix that is held fixed */
    const double *dc;
    int wrt_b; /* differentiate with respect to B (else A) */
} mm_ctx;

static double mm_obj(const double *x, void *vc) {
    const mm_ctx *c = (const mm_ctx *)vc;
    const double *A = c->wrt_b ? c->other : x, *B = c->wrt_b ? x : c->other;
    const int64_t lda = c->ta == RRC_OP_N ? c->k : c->m, ldb = c->tb == RRC_OP_N ? c->n : c->k;
    double f = 0;
    for (int64_t i = 0; i < c->m; i++)
        for (int64_t j = 0; j < c->n; j++) {
            double s = 0;
            for (int64_t p = 0; p < c->k; p++)
                s += (c->ta == RRC_OP_N ? A[i * lda + p] : A[p * lda + i]) *
                     (c->tb == RRC_OP_N ? B[p * ldb + j] : B[j * ldb + p]);
            f += c->alpha * s * c->dc[i * c->n + j];
        }
    return f;
}

/* The backward GEMMs of C = alpha op(A) op(B), written with the transpose flags the model uses:
 *   ta == N: dA = alpha dC op(B)^T          ta == T: dA = alpha op(B) dC^T   (stored layout of A)
 *   tb == N: dB = alpha op(A)^T dC          tb == T: dB = alpha dC^T op(A)   (stored layout of B) */
RT_TEST(grad_gemm_transposes) {
    rt_rng rng = {251};
    for (int bi = 0; bi < 2; bi++)
        for (int ta = 0; ta < 2; ta++)
            for (int tb = 0; tb < 2; tb++)
                for (int dt = 0; dt < 2; dt++) {
                    const rrc_backend *be = rt_backend(bi);
                    const rrc_trans TA = (rrc_trans)ta, TB = (rrc_trans)tb;
                    const int64_t m = 4, n = 5, k = 3;
                    const int64_t lda = TA == RRC_OP_N ? k : m, ldb = TB == RRC_OP_N ? n : k;
                    const int64_t na = m * k, nb = k * n;
                    float A[12], B[15], dC[20];
                    rt_fill(&rng, A, na, 1.0f);
                    rt_fill(&rng, B, nb, 1.0f);
                    rt_fill(&rng, dC, m * n, 1.0f);
                    rt_round_inplace(A, na, k_fdt[dt]);
                    rt_round_inplace(B, nb, k_fdt[dt]);
                    rt_round_inplace(dC, m * n, k_fdt[dt]);
                    double Ad[12], Bd[15], dCd[20], gA[12], gB[15];
                    for (int64_t i = 0; i < na; i++) Ad[i] = A[i];
                    for (int64_t i = 0; i < nb; i++) Bd[i] = B[i];
                    for (int64_t i = 0; i < m * n; i++) dCd[i] = dC[i];
                    const double alpha = 0.7;
                    mm_ctx ca = {TA, TB, m, n, k, alpha, Bd, dCd, 0};
                    fd_grad(mm_obj, &ca, Ad, na, 1e-5, gA);
                    mm_ctx cb = {TA, TB, m, n, k, alpha, Ad, dCd, 1};
                    fd_grad(mm_obj, &cb, Bd, nb, 1e-5, gB);
                    rrc_buf a = rt_upload(be, k_fdt[dt], A, na), b = rt_upload(be, k_fdt[dt], B, nb),
                            dc = rt_upload(be, k_fdt[dt], dC, m * n), da, db;
                    RT_OK(rrc_buf_alloc(be, RRC_F32, na, &da));
                    RT_OK(rrc_buf_alloc(be, RRC_F32, nb, &db));
                    const rrc_trans flipB = TB == RRC_OP_N ? RRC_OP_T : RRC_OP_N;
                    const rrc_trans flipA = TA == RRC_OP_N ? RRC_OP_T : RRC_OP_N;
                    if (TA == RRC_OP_N)
                        RT_OK(rrc_gemm2d(be, RRC_OP_N, flipB, m, k, n, (float)alpha, dc, n, b, ldb, 0.0f, da, lda));
                    else
                        RT_OK(rrc_gemm2d(be, TB, RRC_OP_T, k, m, n, (float)alpha, b, ldb, dc, n, 0.0f, da, lda));
                    if (TB == RRC_OP_N)
                        RT_OK(rrc_gemm2d(be, flipA, RRC_OP_N, k, n, m, (float)alpha, a, lda, dc, n, 0.0f, db, ldb));
                    else
                        RT_OK(rrc_gemm2d(be, RRC_OP_T, TA, n, k, m, (float)alpha, dc, n, a, lda, 0.0f, db, ldb));
                    float *ga = rt_download(da, na), *gb = rt_download(db, nb);
                    char label[96];
                    snprintf(label, sizeof label, "gemm dA ta=%d tb=%d %s", ta, tb, rrc_dtype_name(k_fdt[dt]));
                    RT_ALLCLOSE(ga, gA, na, 1e-6, 1e-5, label);
                    snprintf(label, sizeof label, "gemm dB ta=%d tb=%d %s", ta, tb, rrc_dtype_name(k_fdt[dt]));
                    RT_ALLCLOSE(gb, gB, nb, 1e-6, 1e-5, label);
                    free(ga);
                    free(gb);
                    rrc_buf_free(&a);
                    rrc_buf_free(&b);
                    rrc_buf_free(&dc);
                    rrc_buf_free(&da);
                    rrc_buf_free(&db);
                }
}

const rt_case rt_suite_grad[] = {
    RT_CASE(grad_rmsnorm), RT_CASE(grad_rope),          RT_CASE(grad_attn_softmax),
    RT_CASE(grad_swiglu),  RT_CASE(grad_cross_entropy), RT_CASE(grad_gemm_transposes),
};
const size_t rt_suite_grad_n = sizeof rt_suite_grad / sizeof rt_suite_grad[0];
