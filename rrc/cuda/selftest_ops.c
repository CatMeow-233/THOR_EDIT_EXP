/* rrc/cuda/selftest_ops.c — see selftest_ops.h. */
#include "selftest_ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "rrc/rrc.h"

#define U24 (1.0 / 16777216.0) /* 2^-24: unit roundoff of fp32 */
#define BF16_ULP (1.0 / 128.0) /* one bf16 ulp relative to the value (bottom of a binade) */

enum { CLS_EXACT = 0, CLS_ULP = 1, CLS_REDUCE = 2, CLS_GEMM = 3 };
static const char *k_cls_name[] = {"EXACT", "ULP", "REDUCE", "GEMM"};

typedef struct st {
    const rrc_backend *dev, *ref;
    const rrc_st_opts *o;
    FILE *out;
    rrc_st_result *res;
    uint64_t rng;
    const char *op;
    int op_checks, op_fail;
    double op_worst, op_abs, op_rel;
    /* gemm path tally per layout (level 0 / 1 / 2) and the cases that ran at the kernel level */
    int tally[8][3];
    int n_kernel_cases;
    int n_gemm_cases, n_chunked_cases;
    char paths[8][160]; /* gemm paths seen per (a, b, c) dtype triple */
} st;

/* ---- small utilities ------------------------------------------------------------------------- */

static uint64_t rnext(st *t) {
    uint64_t z = (t->rng += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float runif(st *t) { return (float)((double)(rnext(t) >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0); }

/* n random values in [-scale, scale), rounded to dt (so both backends see identical inputs) */
static float *hrand(st *t, int64_t n, float scale, rrc_dtype dt) {
    float *h = (float *)malloc((size_t)(n > 0 ? n : 1) * sizeof(float));
    if (!h) return NULL;
    for (int64_t i = 0; i < n; i++) h[i] = runif(t) * scale;
    if (dt == RRC_BF16)
        for (int64_t i = 0; i < n; i++) h[i] = rrc_round_bf16(h[i]);
    return h;
}

typedef struct bufset {
    rrc_buf b[16];
    int n;
} bufset;

static void bs_free(bufset *s) {
    for (int i = 0; i < s->n; i++) rrc_buf_free(&s->b[i]);
    s->n = 0;
}

/* upload n host floats as dtype dt into a new buffer of `cap` (>= n) elements */
static rrc_status upf(bufset *s, const rrc_backend *be, rrc_dtype dt, const float *h, int64_t n, rrc_buf *out) {
    RRC_REQUIRE(s->n < 16, RRC_ERR_INTERNAL, "selftest: too many buffers");
    rrc_buf b;
    RRC_CHECK(rrc_buf_alloc(be, dt, n, &b));
    s->b[s->n++] = b;
    *out = b;
    if (n == 0) return RRC_OK;
    if (dt == RRC_F32) return rrc_buf_upload(b, h, n);
    rrc_bf16 *tmp = (rrc_bf16 *)malloc((size_t)n * sizeof(rrc_bf16));
    if (!tmp) return RRC_ERROR(RRC_ERR_OOM, "selftest upload");
    for (int64_t i = 0; i < n; i++) tmp[i] = rrc_f32_to_bf16(h[i]);
    rrc_status st_ = rrc_buf_upload(b, tmp, n);
    free(tmp);
    return st_;
}
static rrc_status upi(bufset *s, const rrc_backend *be, const int32_t *h, int64_t n, rrc_buf *out) {
    RRC_REQUIRE(s->n < 16, RRC_ERR_INTERNAL, "selftest: too many buffers");
    rrc_buf b;
    RRC_CHECK(rrc_buf_alloc(be, RRC_I32, n, &b));
    s->b[s->n++] = b;
    *out = b;
    return n ? rrc_buf_upload(b, h, n) : RRC_OK;
}
static rrc_status dnf(rrc_buf b, int64_t n, float *out) {
    if (n == 0) return RRC_OK;
    if (b.dtype == RRC_F32) return rrc_buf_download(b, out, n);
    rrc_bf16 *tmp = (rrc_bf16 *)malloc((size_t)n * sizeof(rrc_bf16));
    if (!tmp) return RRC_ERROR(RRC_ERR_OOM, "selftest download");
    rrc_status s = rrc_buf_download(b, tmp, n);
    for (int64_t i = 0; s == RRC_OK && i < n; i++) out[i] = rrc_bf16_to_f32(tmp[i]);
    free(tmp);
    return s;
}

static const char *dn(rrc_dtype d) { return rrc_dtype_name(d); }

/* ---- comparison -------------------------------------------------------------------------------- */

typedef struct seg {
    int64_t n;
    rrc_dtype dt;
} seg;
typedef struct outspec {
    int nseg;
    seg s[3];
} outspec;
static int64_t os_total(const outspec *os) {
    int64_t t = 0;
    for (int i = 0; i < os->nseg; i++) t += os->s[i].n;
    return t;
}

typedef rrc_status (*runfn)(const rrc_backend *be, const void *ctx, float *out);

static void op_begin(st *t, const char *op) {
    t->op = op;
    t->op_checks = t->op_fail = 0;
    t->op_worst = t->op_abs = t->op_rel = 0.0;
}
static void note(char (*notes)[200], int *n, const char *what) {
    if (*n < RRC_ST_MAX_NOTES) snprintf(notes[*n], 200, "%s", what);
    (*n)++;
}
static void op_end(st *t) {
    if (t->op_checks == 0) return;
    if (t->res->n_ops < RRC_ST_MAX_OPS) {
        rrc_st_op_summary *os = &t->res->ops[t->res->n_ops++];
        snprintf(os->op, sizeof os->op, "%s", t->op);
        os->cases = t->op_checks;
        os->failures = t->op_fail;
        os->max_abs = t->op_abs;
        os->max_rel = t->op_rel;
        os->worst_ratio = t->op_worst;
    }
    fprintf(t->out, "  %-22s %4d cases  max_abs %-9.3g max_rel %-9.3g worst err/tol %-9.3g %s\n", t->op, t->op_checks,
            t->op_abs, t->op_rel, t->op_worst, t->op_fail ? "FAIL" : "PASS");
}
static int want(st *t, const char *op) { return !t->o->only || strstr(op, t->o->only) != NULL; }

static int same_bits(float a, float b) { return rrc_f32_bits(a) == rrc_f32_bits(b); }

/* Runs f on ref once and on dev twice, compares. n_red: terms per reduction (REDUCE); mag:
 * per-element magnitude (GEMM, may be NULL otherwise); k: GEMM inner size. */
static void check(st *t, const char *name, int cls, double n_red, const double *mag, int64_t k, runfn f,
                  const void *ctx, const outspec *os) {
    const int64_t n = os_total(os);
    float *r = (float *)malloc((size_t)(n > 0 ? n : 1) * sizeof(float));
    float *d1 = (float *)malloc((size_t)(n > 0 ? n : 1) * sizeof(float));
    float *d2 = (float *)malloc((size_t)(n > 0 ? n : 1) * sizeof(float));
    char path[160] = "";
    int fail = 0;
    double worst = 0.0, max_abs = 0.0, max_rel = 0.0;
    int64_t worst_i = -1;
    rrc_status s1 = RRC_ERR_OOM, s2 = RRC_ERR_OOM, s3 = RRC_ERR_OOM;
    char err[RRC_ERROR_MSG_MAX] = "";
    if (r && d1 && d2) {
        rrc_clear_error();
        s1 = f(t->ref, ctx, r);
        if (s1 != RRC_OK) snprintf(err, sizeof err, "ref: %s", rrc_last_error());
        if (s1 == RRC_OK) {
            s2 = f(t->dev, ctx, d1);
            if (s2 != RRC_OK) snprintf(err, sizeof err, "dev: %s", rrc_last_error());
            if (s2 == RRC_OK && t->o->gemm_path) t->o->gemm_path(t->dev, path, sizeof path);
        }
        if (s2 == RRC_OK) {
            s3 = f(t->dev, ctx, d2);
            if (s3 != RRC_OK) snprintf(err, sizeof err, "dev (2nd run): %s", rrc_last_error());
        }
    } else {
        snprintf(err, sizeof err, "host allocation failed");
    }
    int det_fail = 0;
    if (s1 == RRC_OK && s2 == RRC_OK && s3 == RRC_OK) {
        for (int64_t i = 0; i < n; i++)
            if (!same_bits(d1[i], d2[i])) {
                det_fail = 1;
                break;
            }
        int64_t base = 0;
        for (int sg = 0; sg < os->nseg; sg++) {
            const int64_t sn = os->s[sg].n;
            const int bf = os->s[sg].dt == RRC_BF16;
            double scale = 0.0;
            for (int64_t i = 0; i < sn; i++)
                if (isfinite(r[base + i]) && fabs(r[base + i]) > scale) scale = fabs(r[base + i]);
            for (int64_t i = base; i < base + sn; i++) {
                const double g = d1[i], w = r[i];
                if (isnan(w) || isnan(g) || isinf(w) || isinf(g)) {
                    if (!(isnan(w) && isnan(g)) && g != w) {
                        fail = 1;
                        worst = INFINITY;
                        worst_i = i;
                    }
                    continue;
                }
                const double diff = fabs(g - w);
                double tol;
                switch (cls) {
                case CLS_EXACT: tol = 0.0; break;
                case CLS_ULP: tol = 2e-6 * (fabs(w) + scale) + (bf ? BF16_ULP * fmax(fabs(w), fabs(g)) : 0.0); break;
                case CLS_REDUCE: tol = (1e-5 + n_red * U24) * (fabs(w) + scale) + (bf ? BF16_ULP * fmax(fabs(w), fabs(g)) : 0.0); break;
                default: {
                    double m = mag ? mag[i] : 0.0;
                    tol = 2.0 * (double)k * U24 * m + (t->o->tf32 ? m / 512.0 : 0.0) + (bf ? BF16_ULP * fmax(fabs(w), fabs(g)) : 0.0);
                } break;
                }
                if (diff > max_abs) max_abs = diff;
                if (w != 0.0 && diff / fabs(w) > max_rel) max_rel = diff / fabs(w);
                double ratio = diff == 0.0 ? 0.0 : (tol > 0.0 ? diff / tol : INFINITY);
                if (ratio > worst) {
                    worst = ratio;
                    worst_i = i;
                }
                if (diff > tol) fail = 1;
            }
            base += sn;
        }
    } else {
        fail = 1;
    }
    t->res->checks++;
    t->res->det_checks++;
    t->op_checks++;
    if (fail) {
        t->res->failures++;
        t->op_fail++;
    }
    if (det_fail) {
        t->res->det_failures++;
        t->op_fail++;
    }
    if (worst > t->op_worst) t->op_worst = worst;
    if (max_abs > t->op_abs) t->op_abs = max_abs;
    if (max_rel > t->op_rel) t->op_rel = max_rel;
    if (fail || det_fail) {
        char nb[200];
        snprintf(nb, sizeof nb, "%.150s%s: err/tol %.3g", name, det_fail ? " (nondeterministic)" : "", worst);
        note(t->res->fail_notes, &t->res->n_fail_notes, nb);
    }
    if (fail || det_fail || t->o->verbose) {
        fprintf(t->out, "    %s %-6s %s  max_abs %.3g  max_rel %.3g  err/tol %.3g%s%s%s\n",
                fail ? "FAIL" : "PASS", k_cls_name[cls], name, max_abs, max_rel, worst,
                det_fail ? "  NONDETERMINISTIC (two runs differ)" : "  det ok", path[0] ? "  path: " : "", path);
        if (err[0]) fprintf(t->out, "      error: %s\n", err);
        if (fail && worst_i >= 0 && s1 == RRC_OK && s2 == RRC_OK)
            fprintf(t->out, "      worst element %lld: dev %.9g ref %.9g\n", (long long)worst_i, (double)d1[worst_i],
                    (double)r[worst_i]);
    }
    free(r);
    free(d1);
    free(d2);
}

/* Range error: bad() must fail with RRC_ERR_RANGE on ref; on dev the op or the next sync must
 * report it; afterwards sync must succeed. */
typedef rrc_status (*badfn)(const rrc_backend *be, const void *ctx, bufset *bs);
static void range_check(st *t, const char *name, badfn f, const void *ctx) {
    bufset bs = {{{0}}, 0};
    rrc_clear_error();
    rrc_status sr = f(t->ref, ctx, &bs);
    bs_free(&bs);
    rrc_clear_error();
    rrc_status sd = f(t->dev, ctx, &bs);
    const char *where = "op";
    if (sd == RRC_OK) {
        sd = rrc_sync(t->dev);
        where = "next sync";
    }
    char msg[RRC_ERROR_MSG_MAX];
    snprintf(msg, sizeof msg, "%s", rrc_last_error());
    rrc_status after = rrc_sync(t->dev);
    bs_free(&bs);
    int ok = sr == RRC_ERR_RANGE && sd == RRC_ERR_RANGE && after == RRC_OK;
    t->res->range_checks++;
    t->op_checks++;
    if (!ok) {
        t->res->range_failures++;
        t->op_fail++;
        char nb[200];
        snprintf(nb, sizeof nb, "range error %.100s: ref %s, dev %s, sync afterwards %s", name, rrc_status_str(sr),
                 rrc_status_str(sd), rrc_status_str(after));
        note(t->res->fail_notes, &t->res->n_fail_notes, nb);
    }
    if (!ok || t->o->verbose)
        fprintf(t->out, "    %s RANGE  %s: ref %s, dev %s (reported by the %s), sync afterwards %s\n      dev message: %s\n",
                ok ? "PASS" : "FAIL", name, rrc_status_str(sr), rrc_status_str(sd), where, rrc_status_str(after), msg);
}

/* ---- gemm ---------------------------------------------------------------------------------------- */

typedef struct gcase {
    rrc_trans ta, tb;
    int64_t m, n, k, batch;
    rrc_dtype da, db, dc;
    float alpha, beta;
    int64_t lda, ldb, ldc, sa, sb, sc, na, nb, nc;
    int64_t a_off; /* A starts a_off elements into its buffer (misaligned views) */
    int fused;     /* A and C are column blocks of one buffer: A cols [0, k), C cols [k, k + n) */
    float *A, *B, *C;
} gcase;

static rrc_status run_gemm(const rrc_backend *be, const void *ctx, float *out) {
    const gcase *g = (const gcase *)ctx;
    bufset bs = {{{0}}, 0};
    rrc_buf a, b, c, cbase;
    rrc_status s = RRC_OK;
    if (g->fused) {
        RRC_CHECK_GOTO(s, upf(&bs, be, g->dc, g->C, g->nc, &cbase), done);
        a = rrc_view(cbase, 0, g->nc);
        c = rrc_view_from(cbase, g->k);
    } else {
        RRC_CHECK_GOTO(s, upf(&bs, be, g->da, g->A, g->na + g->a_off, &a), done);
        a = rrc_view_from(a, g->a_off);
        RRC_CHECK_GOTO(s, upf(&bs, be, g->dc, g->C, g->nc, &cbase), done);
        c = cbase;
    }
    RRC_CHECK_GOTO(s, upf(&bs, be, g->db, g->B, g->nb, &b), done);
    rrc_gemm_args ga = {g->ta, g->tb, g->m, g->n, g->k, g->alpha, g->beta, a, g->lda, g->sa,
                        b, g->ldb, g->sb, c, g->ldc, g->sc, g->batch};
    RRC_CHECK_GOTO(s, rrc_gemm(be, &ga), done);
    RRC_CHECK_GOTO(s, dnf(cbase, g->nc, out), done);
done:
    bs_free(&bs);
    return s;
}

/* |alpha| sum |a||b| + |beta||c| per element of the C buffer (0 outside the C slices) */
static double *gemm_mag(const gcase *g) {
    double *mag = (double *)calloc((size_t)g->nc, sizeof(double));
    if (!mag) return NULL;
    const float *A = g->fused ? g->C : g->A + g->a_off;
    const int64_t c0 = g->fused ? g->k : 0;
    for (int64_t t = 0; t < g->batch; t++)
        for (int64_t i = 0; i < g->m; i++)
            for (int64_t j = 0; j < g->n; j++) {
                double s = 0.0;
                for (int64_t q = 0; q < g->k; q++) {
                    double av = g->ta == RRC_OP_N ? A[t * g->sa + i * g->lda + q] : A[t * g->sa + q * g->lda + i];
                    double bv = g->tb == RRC_OP_N ? g->B[t * g->sb + q * g->ldb + j] : g->B[t * g->sb + j * g->ldb + q];
                    s += fabs(av * bv);
                }
                int64_t ci = c0 + t * g->sc + i * g->ldc + j;
                mag[ci] = fabs(g->alpha) * s + (g->beta != 0.0f ? fabs(g->beta) * fabs(g->C[ci]) : 0.0);
            }
    return mag;
}

enum { L_PLAIN = 0, L_BATCH, L_BCAST_A, L_BCAST_B, L_INTERLEAVE_C, L_MISALIGNED, L_FUSED, L_ALIGNED, L_N };
static const char *k_layout[] = {"plain", "batched", "bcastA", "bcastB", "interleavedC", "misaligned", "fusedAC", "aligned"};
const char *rrc_st_layout_name(int l) { return l >= 0 && l < L_N ? k_layout[l] : NULL; }

static void gemm_case(st *t, rrc_trans ta, rrc_trans tb, int64_t m, int64_t n, int64_t k, int layout, rrc_dtype da,
                      rrc_dtype db, rrc_dtype dc, float beta) {
    gcase g;
    memset(&g, 0, sizeof g);
    g.ta = ta;
    g.tb = tb;
    g.m = m;
    g.n = n;
    g.k = k;
    g.da = da;
    g.db = db;
    g.dc = dc;
    g.alpha = 1.25f;
    g.beta = beta;
    g.batch = (layout == L_PLAIN || layout == L_MISALIGNED || layout == L_FUSED || layout == L_ALIGNED) ? 1 : 3;
    const int64_t ar = ta == RRC_OP_N ? m : k, ac = ta == RRC_OP_N ? k : m;
    const int64_t br = tb == RRC_OP_N ? k : n, bc = tb == RRC_OP_N ? n : k;
    g.lda = ac + (layout == L_MISALIGNED ? 1 : 3);
    g.ldb = bc + 2;
    if (layout == L_ALIGNED) { /* dense: 256-byte aligned buffers, ld = cols (tensor-core friendly sizes) */
        g.lda = ac;
        g.ldb = bc;
    }
    g.sa = layout == L_BCAST_A ? 0 : ar * g.lda + 1;
    g.sb = layout == L_BCAST_B ? 0 : br * g.ldb + 5;
    if (layout == L_INTERLEAVE_C) {
        g.ldc = g.batch * n + 1;
        g.sc = n;
    } else {
        g.ldc = layout == L_ALIGNED ? n : n + 1;
        g.sc = m * g.ldc + 2;
    }
    g.a_off = layout == L_MISALIGNED ? 1 : 0;
    if (layout == L_FUSED) { /* A = cols [0, k) and C = cols [k, k + n) of one m x (k + n + 1) buffer */
        g.ta = RRC_OP_N;
        g.da = dc;
        g.lda = g.ldc = k + n + 1;
        g.sa = g.sc = 0;
    }
    g.na = (g.batch - 1) * g.sa + ar * g.lda;
    g.nb = (g.batch - 1) * g.sb + br * g.ldb;
    g.nc = layout == L_FUSED ? m * g.ldc : (g.batch - 1) * g.sc + m * g.ldc;
    g.A = hrand(t, g.na + g.a_off, 1.0f, g.da);
    g.B = hrand(t, g.nb, 1.0f, g.db);
    g.C = hrand(t, g.nc, 1.0f, g.dc);
    if (g.A && g.B && g.C && beta == 0.0f && layout != L_FUSED)
        for (int64_t i = 0; i < g.nc; i += 3) g.C[i] = NAN; /* beta == 0: C must not be read */
    double *mag = (g.A && g.B && g.C) ? gemm_mag(&g) : NULL;
    char name[200];
    snprintf(name, sizeof name, "gemm %s x %s -> %s %c%c m=%lld n=%lld k=%lld batch=%lld %s beta=%g", dn(g.da), dn(db),
             dn(dc), g.ta == RRC_OP_N ? 'N' : 'T', tb == RRC_OP_N ? 'N' : 'T', (long long)m, (long long)n,
             (long long)k, (long long)g.batch, k_layout[layout], (double)beta);
    outspec os = {1, {{g.nc, dc}}};
    if (mag) {
        check(t, name, t->o->strict ? CLS_EXACT : CLS_GEMM, 0.0, mag, k, run_gemm, &g, &os);
        char path[160];
        if (t->o->gemm_path && t->o->gemm_path(t->dev, path, sizeof path)) {
            int bit = 4 * (g.da == RRC_BF16) + 2 * (db == RRC_BF16) + (dc == RRC_BF16);
            if (layout == L_PLAIN && !t->paths[bit][0]) snprintf(t->paths[bit], sizeof t->paths[bit], "%s", path);
            int level = !strncmp(path, "kernel", 6) ? 2 : (!strncmp(path, "promote-all", 11) ? 1 : 0);
            t->tally[layout][level]++;
            t->n_gemm_cases++;
            if (strstr(path, "chunks")) t->n_chunked_cases++;
            if (level == 2) {
                if (t->n_kernel_cases < 40) fprintf(t->out, "    note: kernel GEMM (no cuBLASLt algorithm): %s\n", name);
                t->n_kernel_cases++;
                note(t->res->kernel_notes, &t->res->n_kernel_notes, name);
            }
            if (t->o->require_lt && layout == L_ALIGNED) {
                /* natively supported triples: f32 f32 f32 (bit 0), bf16 bf16 f32 (6), bf16 bf16 bf16 (7) */
                const int native = bit == 0 || bit == 6 || bit == 7;
                const int ok = native ? level == 0 : level < 2;
                t->res->path_checks++;
                if (!ok) {
                    t->res->path_failures++;
                    t->op_fail++;
                    fprintf(t->out, "    FAIL PATH   %s: path '%s' (level %d), required %s\n", name, path, level,
                            native ? "level 0 (native cuBLASLt)" : "a cuBLASLt level (0 or 1), not the kernel fallback");
                    char nb[200];
                    snprintf(nb, sizeof nb, "gemm path: %.150s ran at level %d", name, level);
                    note(t->res->fail_notes, &t->res->n_fail_notes, nb);
                }
            }
        }
    } else {
        fprintf(t->out, "    FAIL %s: host allocation failed\n", name);
        t->res->failures++;
        t->op_fail++;
    }
    free(mag);
    free(g.A);
    free(g.B);
    free(g.C);
}

static void test_gemm(st *t) {
    if (!want(t, "gemm")) return;
    op_begin(t, "gemm");
    static const rrc_dtype F[2] = {RRC_F32, RRC_BF16};
    for (int combo = 0; combo < 8; combo++) {
        rrc_dtype da = F[(combo >> 2) & 1], db = F[(combo >> 1) & 1], dc = F[combo & 1];
        for (int tr = 0; tr < 4; tr++) {
            rrc_trans ta = (tr & 2) ? RRC_OP_T : RRC_OP_N, tb = (tr & 1) ? RRC_OP_T : RRC_OP_N;
            for (int layout = 0; layout < L_ALIGNED; layout++) {
                if (layout == L_FUSED && ta == RRC_OP_T) continue; /* the fused case is defined for ta = N */
                gemm_case(t, ta, tb, 37, 29, 53, layout, da, db, dc, 0.75f);
                if (layout <= L_BATCH) gemm_case(t, ta, tb, 37, 29, 53, layout, da, db, dc, 0.0f);
            }
        }
        /* a larger aligned problem (tensor-core paths on the device) and degenerate sizes */
        if (t->o->quick) gemm_case(t, RRC_OP_N, RRC_OP_T, 64, 96, 128, L_ALIGNED, da, db, dc, 0.0f);
        else gemm_case(t, RRC_OP_N, RRC_OP_T, 256, 384, 512, L_ALIGNED, da, db, dc, 0.0f);
        gemm_case(t, RRC_OP_N, RRC_OP_N, 1, 1, 1, L_PLAIN, da, db, dc, 0.5f);
        gemm_case(t, RRC_OP_T, RRC_OP_N, 64, 8, 1, L_BATCH, da, db, dc, 1.0f);
    }
    op_end(t);
    if (t->o->gemm_path) {
        fprintf(t->out, "  gemm path per dtype combination (A x B -> C, first plain case):\n");
        for (int combo = 0; combo < 8; combo++)
            fprintf(t->out, "    %-4s x %-4s -> %-4s : %s\n", dn((combo & 4) ? RRC_BF16 : RRC_F32),
                    dn((combo & 2) ? RRC_BF16 : RRC_F32), dn((combo & 1) ? RRC_BF16 : RRC_F32),
                    t->paths[combo][0] ? t->paths[combo] : "(unknown)");
        fprintf(t->out, "  gemm cases per layout and ladder level (0 native/promoted, 1 promote-all, 2 kernel):\n");
        for (int l = 0; l < L_N; l++)
            fprintf(t->out, "    %-13s level0 %3d  level1 %3d  level2 %3d\n", k_layout[l], t->tally[l][0], t->tally[l][1],
                    t->tally[l][2]);
        if (t->n_kernel_cases)
            fprintf(t->out, "  %d gemm cases ran on the kernel fallback (correct, but not cuBLASLt)\n", t->n_kernel_cases);
        fprintf(t->out, "  %d of %d gemm cases ran in more than one scratch chunk\n", t->n_chunked_cases, t->n_gemm_cases);
        if (t->o->require_lt && t->n_gemm_cases > 0) {
            int lt_cases = 0;
            for (int l = 0; l < L_N; l++) lt_cases += t->tally[l][0] + t->tally[l][1];
            t->res->path_checks++;
            if (lt_cases == 0) {
                t->res->path_failures++;
                fprintf(t->out, "  FAIL PATH   no gemm case reached cuBLASLt (every case ran on the kernel fallback)\n");
            }
        }
        if (t->o->require_chunked) {
            t->res->path_checks++;
            if (t->n_chunked_cases == 0) {
                t->res->path_failures++;
                fprintf(t->out, "  FAIL PATH   no gemm case ran in more than one scratch chunk (the capped pass tested nothing)\n");
            }
        }
    }
}

/* ---- embedding / bias ------------------------------------------------------------------------------ */

typedef struct ecase {
    int64_t n_tokens, dim, vocab;
    rrc_dtype tdt, odt;
    int32_t *ids;
    float *table;
} ecase;

static rrc_status run_embed(const rrc_backend *be, const void *ctx, float *out) {
    const ecase *e = (const ecase *)ctx;
    bufset bs = {{{0}}, 0};
    rrc_buf ids, tab, o;
    rrc_status s;
    RRC_CHECK_GOTO(s, upi(&bs, be, e->ids, e->n_tokens, &ids), done);
    RRC_CHECK_GOTO(s, upf(&bs, be, e->tdt, e->table, e->vocab * e->dim, &tab), done);
    RRC_CHECK_GOTO(s, rrc_buf_alloc(be, e->odt, e->n_tokens * e->dim, &o), done);
    bs.b[bs.n++] = o;
    RRC_CHECK_GOTO(s, rrc_embedding_fwd(be, e->n_tokens, e->dim, e->vocab, ids, tab, o), done);
    RRC_CHECK_GOTO(s, dnf(o, e->n_tokens * e->dim, out), done);
done:
    bs_free(&bs);
    return s;
}
static rrc_status bad_embed(const rrc_backend *be, const void *ctx, bufset *bs) {
    const ecase *e = (const ecase *)ctx;
    rrc_buf ids, tab, o;
    RRC_CHECK(upi(bs, be, e->ids, e->n_tokens, &ids));
    RRC_CHECK(upf(bs, be, e->tdt, e->table, e->vocab * e->dim, &tab));
    RRC_CHECK(rrc_buf_alloc(be, e->odt, e->n_tokens * e->dim, &o));
    bs->b[bs->n++] = o;
    return rrc_embedding_fwd(be, e->n_tokens, e->dim, e->vocab, ids, tab, o);
}

static void test_embedding(st *t) {
    if (!want(t, "embedding_fwd")) return;
    op_begin(t, "embedding_fwd");
    static const rrc_dtype F[2] = {RRC_F32, RRC_BF16};
    for (int c = 0; c < 4; c++) {
        ecase e = {37, c < 2 ? 29 : 64, 50, F[c & 1], F[(c >> 1) & 1], NULL, NULL};
        e.ids = (int32_t *)malloc((size_t)e.n_tokens * sizeof(int32_t));
        e.table = hrand(t, e.vocab * e.dim, 2.0f, e.tdt);
        if (!e.ids || !e.table) {
            free(e.ids);
            free(e.table);
            continue;
        }
        for (int64_t i = 0; i < e.n_tokens; i++) e.ids[i] = (int32_t)(rnext(t) % (uint64_t)e.vocab);
        char name[160];
        snprintf(name, sizeof name, "embedding_fwd %s -> %s n_tokens=%lld dim=%lld vocab=%lld", dn(e.tdt), dn(e.odt),
                 (long long)e.n_tokens, (long long)e.dim, (long long)e.vocab);
        outspec os = {1, {{e.n_tokens * e.dim, e.odt}}};
        check(t, name, CLS_EXACT, 0, NULL, 0, run_embed, &e, &os);
        if (c == 0) {
            e.ids[5] = (int32_t)e.vocab;
            range_check(t, "embedding_fwd ids[5] = vocab", bad_embed, &e);
            e.ids[5] = -1;
            range_check(t, "embedding_fwd ids[5] = -1", bad_embed, &e);
        }
        free(e.ids);
        free(e.table);
    }
    op_end(t);
}

typedef struct bcase {
    int64_t rows, cols, ld;
    rrc_dtype xdt, bdt;
    int accumulate;
    float *x, *bias;
} bcase;

static rrc_status run_add_bias(const rrc_backend *be, const void *ctx, float *out) {
    const bcase *k = (const bcase *)ctx;
    bufset bs = {{{0}}, 0};
    rrc_buf x, b;
    rrc_status s;
    const int64_t nx = (k->rows - 1) * k->ld + k->cols;
    RRC_CHECK_GOTO(s, upf(&bs, be, k->xdt, k->x, nx, &x), done);
    RRC_CHECK_GOTO(s, upf(&bs, be, k->bdt, k->bias, k->cols, &b), done);
    RRC_CHECK_GOTO(s, rrc_add_bias(be, k->rows, k->cols, x, k->ld, b), done);
    RRC_CHECK_GOTO(s, dnf(x, nx, out), done);
done:
    bs_free(&bs);
    return s;
}
static rrc_status run_bias_grad(const rrc_backend *be, const void *ctx, float *out) {
    const bcase *k = (const bcase *)ctx;
    bufset bs = {{{0}}, 0};
    rrc_buf x, b;
    rrc_status s;
    const int64_t nx = (k->rows - 1) * k->ld + k->cols;
    RRC_CHECK_GOTO(s, upf(&bs, be, k->xdt, k->x, nx, &x), done);
    RRC_CHECK_GOTO(s, upf(&bs, be, RRC_F32, k->bias, k->cols, &b), done);
    RRC_CHECK_GOTO(s, rrc_bias_grad(be, k->rows, k->cols, x, k->ld, b, k->accumulate), done);
    RRC_CHECK_GOTO(s, dnf(b, k->cols, out), done);
done:
    bs_free(&bs);
    return s;
}

static void test_bias(st *t) {
    static const rrc_dtype F[2] = {RRC_F32, RRC_BF16};
    if (want(t, "add_bias")) {
        op_begin(t, "add_bias");
        for (int c = 0; c < 4; c++) {
            bcase k = {33, 47, 50, F[c & 1], F[(c >> 1) & 1], 0, NULL, NULL};
            k.x = hrand(t, (k.rows - 1) * k.ld + k.cols, 1.0f, k.xdt);
            k.bias = hrand(t, k.cols, 1.0f, k.bdt);
            char name[128];
            snprintf(name, sizeof name, "add_bias x %s bias %s rows=%lld cols=%lld ld=%lld", dn(k.xdt), dn(k.bdt),
                     (long long)k.rows, (long long)k.cols, (long long)k.ld);
            outspec os = {1, {{(k.rows - 1) * k.ld + k.cols, k.xdt}}};
            if (k.x && k.bias) check(t, name, CLS_EXACT, 0, NULL, 0, run_add_bias, &k, &os);
            free(k.x);
            free(k.bias);
        }
        op_end(t);
    }
    if (want(t, "bias_grad")) {
        op_begin(t, "bias_grad");
        for (int c = 0; c < 4; c++) {
            bcase k = {65, 47, 49, F[c & 1], RRC_F32, c >> 1, NULL, NULL};
            k.x = hrand(t, (k.rows - 1) * k.ld + k.cols, 1.0f, k.xdt);
            k.bias = hrand(t, k.cols, 1.0f, RRC_F32);
            char name[128];
            snprintf(name, sizeof name, "bias_grad dy %s rows=%lld cols=%lld ld=%lld accumulate=%d", dn(k.xdt),
                     (long long)k.rows, (long long)k.cols, (long long)k.ld, k.accumulate);
            outspec os = {1, {{k.cols, RRC_F32}}};
            /* a reduction over rows: the kernel sums sequentially like cpu_f32acc (expected bit-identical),
             * checked with the REDUCE tolerance so that other correct summation orders pass */
            if (k.x && k.bias)
                check(t, name, t->o->strict ? CLS_EXACT : CLS_REDUCE, (double)k.rows, NULL, 0, run_bias_grad, &k, &os);
            free(k.x);
            free(k.bias);
        }
        op_end(t);
    }
}

/* ---- rmsnorm ---------------------------------------------------------------------------------------- */

typedef struct rcase {
    int64_t rows, dim, ld;
    rrc_dtype xdt, wdt, ydt;
    int bwd, accumulate;
    int alias; /* fwd: 1 = y is x; bwd: 1 = dx is dy, 2 = dx is x */
    float *x, *w, *dy, *dx;
} rcase;

static rrc_status run_rms(const rrc_backend *be, const void *ctx, float *out) {
    const rcase *k = (const rcase *)ctx;
    bufset bs = {{{0}}, 0};
    rrc_buf x, w, dy, y;
    rrc_status s;
    const int64_t n = (k->rows - 1) * k->ld + k->dim;
    RRC_CHECK_GOTO(s, upf(&bs, be, k->xdt, k->x, n, &x), done);
    RRC_CHECK_GOTO(s, upf(&bs, be, k->wdt, k->w, k->dim, &w), done);
    if (!k->bwd) {
        if (k->alias) y = x;
        else RRC_CHECK_GOTO(s, upf(&bs, be, k->ydt, k->dx, n, &y), done);
        RRC_CHECK_GOTO(s, rrc_rmsnorm_fwd(be, k->rows, k->dim, k->ld, x, w, 1e-5f, y), done);
    } else {
        RRC_CHECK_GOTO(s, upf(&bs, be, k->xdt, k->dy, n, &dy), done);
        if (k->alias == 1) y = dy;
        else if (k->alias == 2) y = x;
        else RRC_CHECK_GOTO(s, upf(&bs, be, k->ydt, k->dx, n, &y), done);
        RRC_CHECK_GOTO(s, rrc_rmsnorm_bwd(be, k->rows, k->dim, k->ld, x, w, 1e-5f, dy, y, k->accumulate), done);
    }
    RRC_CHECK_GOTO(s, dnf(y, n, out), done);
done:
    bs_free(&bs);
    return s;
}

static void test_rmsnorm(st *t) {
    static const int64_t dims[] = {1, 63, 300, 4096};
    for (int bwd = 0; bwd < 2; bwd++) {
        const char *op = bwd ? "rmsnorm_bwd" : "rmsnorm_fwd";
        if (!want(t, op)) continue;
        op_begin(t, op);
        for (int di = 0; di < 4; di++)
            for (int c = 0; c < 6; c++) {
                rcase k;
                memset(&k, 0, sizeof k);
                k.rows = dims[di] == 4096 ? 4 : 7;
                k.dim = dims[di];
                k.ld = k.dim + 3;
                k.bwd = bwd;
                k.xdt = (c & 1) ? RRC_BF16 : RRC_F32;
                k.wdt = (c & 2) ? RRC_BF16 : RRC_F32;
                k.ydt = k.xdt;
                if (c >= 4) { /* aliases (same dtype, same ld) */
                    k.alias = bwd ? (c == 4 ? 1 : 2) : 1;
                    k.wdt = RRC_F32;
                } else if (bwd) {
                    k.accumulate = c >> 1;
                    k.wdt = RRC_F32;
                    k.ydt = (c & 2) ? RRC_F32 : k.xdt;
                }
                const int64_t n = (k.rows - 1) * k.ld + k.dim;
                k.x = hrand(t, n, 2.0f, k.xdt);
                k.w = hrand(t, k.dim, 1.5f, k.wdt);
                k.dy = hrand(t, n, 1.0f, k.xdt);
                k.dx = hrand(t, n, 1.0f, k.ydt);
                char name[160];
                snprintf(name, sizeof name, "%s x %s w %s out %s rows=%lld dim=%lld ld=%lld%s%s", op, dn(k.xdt),
                         dn(k.wdt), dn(k.ydt), (long long)k.rows, (long long)k.dim, (long long)k.ld,
                         k.accumulate ? " accumulate" : "",
                         k.alias == 1 ? (bwd ? " dx=dy" : " in-place") : (k.alias == 2 ? " dx=x" : ""));
                outspec os = {1, {{n, k.ydt}}};
                if (k.x && k.w && k.dy && k.dx) check(t, name, CLS_REDUCE, (double)k.dim, NULL, 0, run_rms, &k, &os);
                free(k.x);
                free(k.w);
                free(k.dy);
                free(k.dx);
            }
        op_end(t);
    }
}

/* ---- rope -------------------------------------------------------------------------------------------- */

typedef struct pcase {
    int64_t n_tokens, n_heads, head_dim, ld, n_pos;
    rrc_dtype xdt;
    int bwd, inplace;
    float *x, *y, *cs, *sn;
    int32_t *pos;
} pcase;

static rrc_status rope_call(const rrc_backend *be, const pcase *k, bufset *bs, rrc_buf *yout) {
    rrc_buf x, y, pos, cs, sn;
    const int64_t n = (k->n_tokens - 1) * k->ld + k->n_heads * k->head_dim;
    const int64_t nt = k->n_pos * (k->head_dim / 2);
    RRC_CHECK(upf(bs, be, k->xdt, k->x, n, &x));
    if (k->inplace) y = x;
    else RRC_CHECK(upf(bs, be, k->xdt, k->y, n, &y));
    RRC_CHECK(upi(bs, be, k->pos, k->n_tokens, &pos));
    RRC_CHECK(upf(bs, be, RRC_F32, k->cs, nt, &cs));
    RRC_CHECK(upf(bs, be, RRC_F32, k->sn, nt, &sn));
    *yout = y;
    if (k->bwd) return rrc_rope_bwd(be, k->n_tokens, k->n_heads, k->head_dim, k->ld, x, pos, cs, sn, k->n_pos, y);
    return rrc_rope_fwd(be, k->n_tokens, k->n_heads, k->head_dim, k->ld, x, pos, cs, sn, k->n_pos, y);
}
static rrc_status run_rope(const rrc_backend *be, const void *ctx, float *out) {
    const pcase *k = (const pcase *)ctx;
    bufset bs = {{{0}}, 0};
    rrc_buf y;
    rrc_status s;
    RRC_CHECK_GOTO(s, rope_call(be, k, &bs, &y), done);
    RRC_CHECK_GOTO(s, dnf(y, (k->n_tokens - 1) * k->ld + k->n_heads * k->head_dim, out), done);
done:
    bs_free(&bs);
    return s;
}
static rrc_status bad_rope(const rrc_backend *be, const void *ctx, bufset *bs) {
    rrc_buf y;
    return rope_call(be, (const pcase *)ctx, bs, &y);
}

static void test_rope(st *t) {
    for (int bwd = 0; bwd < 2; bwd++) {
        const char *op = bwd ? "rope_bwd" : "rope_fwd";
        if (!want(t, op)) continue;
        op_begin(t, op);
        for (int c = 0; c < 6; c++) {
            pcase k;
            memset(&k, 0, sizeof k);
            k.n_tokens = 11;
            k.n_heads = 3;
            k.head_dim = (c & 2) ? 64 : 16;
            k.ld = k.n_heads * k.head_dim + 5;
            k.n_pos = 40;
            k.xdt = (c & 1) ? RRC_BF16 : RRC_F32;
            k.bwd = bwd;
            k.inplace = c >= 4;
            const int64_t n = (k.n_tokens - 1) * k.ld + k.n_heads * k.head_dim, nt = k.n_pos * (k.head_dim / 2);
            k.x = hrand(t, n, 2.0f, k.xdt);
            k.y = hrand(t, n, 1.0f, k.xdt);
            k.cs = (float *)malloc((size_t)nt * sizeof(float));
            k.sn = (float *)malloc((size_t)nt * sizeof(float));
            k.pos = (int32_t *)malloc((size_t)k.n_tokens * sizeof(int32_t));
            if (k.x && k.y && k.cs && k.sn && k.pos) {
                for (int64_t i = 0; i < nt; i++) {
                    double a = (double)runif(t) * 3.14159;
                    k.cs[i] = (float)cos(a);
                    k.sn[i] = (float)sin(a);
                }
                for (int64_t i = 0; i < k.n_tokens; i++) k.pos[i] = (int32_t)(rnext(t) % (uint64_t)k.n_pos);
                char name[160];
                snprintf(name, sizeof name, "%s %s n_tokens=%lld heads=%lld head_dim=%lld ld=%lld%s", op, dn(k.xdt),
                         (long long)k.n_tokens, (long long)k.n_heads, (long long)k.head_dim, (long long)k.ld,
                         k.inplace ? " in-place" : "");
                outspec os = {1, {{n, k.xdt}}};
                check(t, name, CLS_EXACT, 0, NULL, 0, run_rope, &k, &os);
                if (c == 0) {
                    k.pos[3] = (int32_t)k.n_pos;
                    range_check(t, bwd ? "rope_bwd pos[3] = n_pos" : "rope_fwd pos[3] = n_pos", bad_rope, &k);
                    k.pos[3] = -2;
                    range_check(t, bwd ? "rope_bwd pos[3] = -2" : "rope_fwd pos[3] = -2", bad_rope, &k);
                }
            }
            free(k.x);
            free(k.y);
            free(k.cs);
            free(k.sn);
            free(k.pos);
        }
        op_end(t);
    }
}

/* ---- attention softmax -------------------------------------------------------------------------------- */

typedef struct scase {
    int64_t batch, rows, cols, ld, bs, row_offset;
    rrc_dtype sdt, pdt, dpdt;
    int causal, docs, bwd;
    int alias; /* fwd: p = s; bwd: 1 = ds is dp, 2 = ds is p */
    float scale;
    float *s, *p, *dp, *ds;
    int32_t *doc;
    int64_t ndoc;
} scase;

static int64_t sc_n(const scase *k) { return (k->batch - 1) * k->bs + (k->rows - 1) * k->ld + k->cols; }

static rrc_status run_softmax(const rrc_backend *be, const void *ctx, float *out) {
    const scase *k = (const scase *)ctx;
    bufset bs = {{{0}}, 0};
    rrc_buf s0, p, dp, ds, doc = rrc_buf_none();
    rrc_status st_;
    const int64_t n = sc_n(k);
    if (!k->bwd) {
        RRC_CHECK_GOTO(st_, upf(&bs, be, k->sdt, k->s, n, &s0), done);
        if (k->alias) p = s0;
        else RRC_CHECK_GOTO(st_, upf(&bs, be, k->pdt, k->p, n, &p), done);
        if (k->docs) RRC_CHECK_GOTO(st_, upi(&bs, be, k->doc, k->ndoc, &doc), done);
        RRC_CHECK_GOTO(st_, rrc_attn_softmax_fwd(be, k->batch, k->rows, k->cols, k->ld, k->bs, s0, k->scale,
                                                 k->row_offset, k->causal, doc, p),
                       done);
        RRC_CHECK_GOTO(st_, dnf(p, n, out), done);
    } else {
        RRC_CHECK_GOTO(st_, upf(&bs, be, k->pdt, k->p, n, &p), done);
        RRC_CHECK_GOTO(st_, upf(&bs, be, k->dpdt, k->dp, n, &dp), done);
        if (k->alias == 1) ds = dp;
        else if (k->alias == 2) ds = p;
        else RRC_CHECK_GOTO(st_, upf(&bs, be, k->sdt, k->ds, n, &ds), done);
        RRC_CHECK_GOTO(st_, rrc_attn_softmax_bwd(be, k->batch, k->rows, k->cols, k->ld, k->bs, p, dp, k->scale, ds),
                       done);
        RRC_CHECK_GOTO(st_, dnf(ds, n, out), done);
    }
done:
    bs_free(&bs);
    return st_;
}

static void test_softmax(st *t) {
    for (int bwd = 0; bwd < 2; bwd++) {
        const char *op = bwd ? "attn_softmax_bwd" : "attn_softmax_fwd";
        if (!want(t, op)) continue;
        op_begin(t, op);
        for (int ci = 0; ci < 2; ci++)
            for (int c = 0; c < 11; c++) {
                /* c 8..10: -inf scores (about 10%, one row of all -inf, -inf on the causal diagonal of
                 * every third row, so row 0 of a causal case has only -inf allowed): f32 causal, bf16
                 * causal, f32 non-causal */
                const int ninf = c >= 8;
                scase k;
                memset(&k, 0, sizeof k);
                k.batch = 2;
                k.rows = 13;
                k.cols = ci ? 300 : 37;
                k.ld = k.cols + 3;
                k.bs = k.rows * k.ld + 7;
                k.bwd = bwd;
                k.scale = 0.125f;
                k.causal = ninf ? c != 10 : c & 1;
                k.docs = ninf ? 0 : (c >> 1) & 1;
                k.row_offset = !ninf && (c & 4) ? k.cols - 2 : 0; /* rows with q >= cols */
                k.sdt = k.pdt = k.dpdt = RRC_F32;
                if (c == 2 || c == 3 || c == 9) k.sdt = k.pdt = k.dpdt = RRC_BF16;
                if (c == 5) k.pdt = RRC_BF16; /* f32 in, bf16 out (fwd) / bf16 p (bwd) */
                if (c >= 6 && c < 8) k.alias = bwd ? (c == 6 ? 1 : 2) : 1;
                if (bwd && k.alias == 2) k.dpdt = k.pdt;
                if (bwd && k.alias == 1) k.sdt = k.dpdt; /* ds has dp's dtype */
                if (bwd && k.alias == 2) k.sdt = k.pdt;
                if (!bwd && k.alias) k.pdt = k.sdt;
                const int64_t n = sc_n(&k);
                k.s = hrand(t, n, 8.0f, k.sdt);
                k.dp = hrand(t, n, 1.0f, k.dpdt);
                k.ds = hrand(t, n, 1.0f, k.sdt);
                k.p = (float *)malloc((size_t)n * sizeof(float));
                k.ndoc = k.row_offset + k.rows > k.cols ? k.row_offset + k.rows : k.cols;
                k.doc = (int32_t *)malloc((size_t)k.ndoc * sizeof(int32_t));
                if (k.s && k.dp && k.ds && k.p && k.doc) {
                    if (ninf) {
                        for (int64_t i = 0; i < n; i++)
                            if (rnext(t) % 10 == 0) k.s[i] = -INFINITY;
                        for (int64_t j = 0; j < k.cols; j++) k.s[k.ld + j] = -INFINITY; /* batch 0, row 1: all -inf */
                        for (int64_t b = 0; b < k.batch; b++)
                            for (int64_t r = 0; r < k.rows && r < k.cols; r += 3) k.s[b * k.bs + r * k.ld + r] = -INFINITY;
                    }
                    for (int64_t i = 0; i < k.ndoc; i++) k.doc[i] = (int32_t)(i / 7);
                    if (k.docs && k.row_offset) k.doc[k.row_offset + 4] = 1000; /* q >= cols, unique doc: no allowed key */
                    if (bwd) { /* a valid probability layout: p = softmax of random scores */
                        for (int64_t i = 0; i < n; i++) k.p[i] = 0.0f;
                        for (int64_t b = 0; b < k.batch; b++)
                            for (int64_t r = 0; r < k.rows; r++) {
                                double z = 0.0;
                                float *row = k.p + b * k.bs + r * k.ld;
                                for (int64_t j = 0; j < k.cols; j++) z += row[j] = expf(k.s[b * k.bs + r * k.ld + j]);
                                for (int64_t j = 0; j < k.cols; j++) row[j] = z > 0.0 ? (float)(row[j] / z) : 0.0f;
                            }
                        if (k.pdt == RRC_BF16)
                            for (int64_t i = 0; i < n; i++) k.p[i] = rrc_round_bf16(k.p[i]);
                    } else {
                        for (int64_t i = 0; i < n; i++) k.p[i] = runif(t);
                        if (k.pdt == RRC_BF16)
                            for (int64_t i = 0; i < n; i++) k.p[i] = rrc_round_bf16(k.p[i]);
                    }
                    char name[200];
                    snprintf(name, sizeof name, "%s s %s p %s dp %s batch=%lld rows=%lld cols=%lld ld=%lld%s%s%s%s%s", op,
                             dn(k.sdt), dn(k.pdt), dn(k.dpdt), (long long)k.batch, (long long)k.rows,
                             (long long)k.cols, (long long)k.ld, k.causal ? " causal" : "", k.docs ? " docs" : "",
                             k.row_offset ? " row_offset" : "",
                             k.alias ? (bwd ? (k.alias == 1 ? " ds=dp" : " ds=p") : " in-place") : "",
                             ninf ? (bwd ? " p from -inf scores" : " -inf scores") : "");
                    outspec os = {1, {{n, bwd ? k.sdt : k.pdt}}};
                    if (!bwd && k.alias) os.s[0].dt = k.sdt;
                    check(t, name, CLS_REDUCE, (double)k.cols, NULL, 0, run_softmax, &k, &os);
                }
                free(k.s);
                free(k.p);
                free(k.dp);
                free(k.ds);
                free(k.doc);
            }
        op_end(t);
    }
}

/* ---- swiglu -------------------------------------------------------------------------------------------- */

typedef struct wcase {
    int64_t rows, cols;
    rrc_dtype dt;
    int bwd, mode; /* fwd: 0 separate h, 1 h = gate block; bwd: 0 separate, 1 fused [dgate|dup], 2 in place */
    float *gu, *dh, *o;
} wcase;

static rrc_status run_swiglu(const rrc_backend *be, const void *ctx, float *out) {
    const wcase *k = (const wcase *)ctx;
    bufset bs = {{{0}}, 0};
    rrc_buf gu, dh, o, o2;
    rrc_status s;
    const int64_t ld = 2 * k->cols, n = k->rows * ld;
    RRC_CHECK_GOTO(s, upf(&bs, be, k->dt, k->gu, n, &gu), done);
    rrc_buf gate = rrc_view(gu, 0, n), up = rrc_view_from(gu, k->cols);
    if (!k->bwd) {
        if (k->mode == 1) {
            RRC_CHECK_GOTO(s, rrc_swiglu_fwd(be, k->rows, k->cols, gate, up, ld, gate, ld), done);
            RRC_CHECK_GOTO(s, dnf(gu, n, out), done);
        } else {
            RRC_CHECK_GOTO(s, upf(&bs, be, k->dt, k->o, k->rows * (k->cols + 1), &o), done);
            RRC_CHECK_GOTO(s, rrc_swiglu_fwd(be, k->rows, k->cols, gate, up, ld, o, k->cols + 1), done);
            RRC_CHECK_GOTO(s, dnf(o, k->rows * (k->cols + 1), out), done);
        }
    } else {
        RRC_CHECK_GOTO(s, upf(&bs, be, k->dt, k->dh, k->rows * (k->cols + 3), &dh), done);
        if (k->mode == 2) {
            RRC_CHECK_GOTO(s, rrc_swiglu_bwd(be, k->rows, k->cols, gate, up, ld, dh, k->cols + 3, gate, up, ld), done);
            RRC_CHECK_GOTO(s, dnf(gu, n, out), done);
        } else if (k->mode == 1) {
            RRC_CHECK_GOTO(s, upf(&bs, be, k->dt, k->o, n, &o), done);
            o2 = rrc_view_from(o, k->cols);
            RRC_CHECK_GOTO(s, rrc_swiglu_bwd(be, k->rows, k->cols, gate, up, ld, dh, k->cols + 3, o, o2, ld), done);
            RRC_CHECK_GOTO(s, dnf(o, n, out), done);
        } else {
            RRC_CHECK_GOTO(s, upf(&bs, be, k->dt, k->o, n, &o), done);
            RRC_CHECK_GOTO(s, upf(&bs, be, k->dt, k->o, n, &o2), done);
            RRC_CHECK_GOTO(s, rrc_swiglu_bwd(be, k->rows, k->cols, gate, up, ld, dh, k->cols + 3, o, o2, k->cols), done);
            RRC_CHECK_GOTO(s, dnf(o, k->rows * k->cols, out), done);
            RRC_CHECK_GOTO(s, dnf(o2, k->rows * k->cols, out + k->rows * k->cols), done);
        }
    }
done:
    bs_free(&bs);
    return s;
}

static void test_swiglu(st *t) {
    for (int bwd = 0; bwd < 2; bwd++) {
        const char *op = bwd ? "swiglu_bwd" : "swiglu_fwd";
        if (!want(t, op)) continue;
        op_begin(t, op);
        for (int c = 0; c < 6; c++) {
            wcase k = {9, 37, (c & 1) ? RRC_BF16 : RRC_F32, bwd, c >> 1, NULL, NULL, NULL};
            if (!bwd && k.mode > 1) continue;
            const int64_t n = k.rows * 2 * k.cols;
            k.gu = hrand(t, n, 4.0f, k.dt);
            k.dh = hrand(t, k.rows * (k.cols + 3), 1.0f, k.dt);
            k.o = hrand(t, n + k.rows, 1.0f, k.dt);
            outspec os = {1, {{0, k.dt}}};
            if (!bwd) os.s[0].n = k.mode == 1 ? n : k.rows * (k.cols + 1);
            else os.s[0].n = k.mode == 0 ? 2 * k.rows * k.cols : n;
            char name[160];
            static const char *mf[] = {"separate h", "h = gate block (in place)"};
            static const char *mb[] = {"separate dgate/dup", "fused [dgate|dup]", "in place dgate=gate dup=up"};
            snprintf(name, sizeof name, "%s %s rows=%lld cols=%lld fused [gate|up], %s", op, dn(k.dt),
                     (long long)k.rows, (long long)k.cols, bwd ? mb[k.mode] : mf[k.mode]);
            if (k.gu && k.dh && k.o) check(t, name, t->o->strict ? CLS_EXACT : CLS_ULP, 0, NULL, 0, run_swiglu, &k, &os);
            free(k.gu);
            free(k.dh);
            free(k.o);
        }
        op_end(t);
    }
}

/* ---- elementwise ------------------------------------------------------------------------------------------ */

typedef struct xcase {
    int op; /* 0 add, 1 axpy, 2 scale, 3 convert */
    int64_t n;
    rrc_dtype adt, bdt, odt;
    int alias; /* add: out = a; axpy: y = x */
    float alpha;
    float *a, *b, *o;
} xcase;

static rrc_status run_ew(const rrc_backend *be, const void *ctx, float *out) {
    const xcase *k = (const xcase *)ctx;
    bufset bs = {{{0}}, 0};
    rrc_buf a, b, o;
    rrc_status s;
    RRC_CHECK_GOTO(s, upf(&bs, be, k->adt, k->a, k->n, &a), done);
    switch (k->op) {
    case 0:
        RRC_CHECK_GOTO(s, upf(&bs, be, k->bdt, k->b, k->n, &b), done);
        if (k->alias) o = a;
        else RRC_CHECK_GOTO(s, upf(&bs, be, k->odt, k->o, k->n, &o), done);
        RRC_CHECK_GOTO(s, rrc_add(be, k->n, a, b, o), done);
        break;
    case 1:
        if (k->alias) o = a; /* y += alpha * y */
        else RRC_CHECK_GOTO(s, upf(&bs, be, k->odt, k->o, k->n, &o), done);
        RRC_CHECK_GOTO(s, rrc_axpy(be, k->n, k->alpha, a, o), done);
        break;
    case 2:
        o = a;
        RRC_CHECK_GOTO(s, rrc_scale(be, k->n, k->alpha, a), done);
        break;
    default:
        RRC_CHECK_GOTO(s, upf(&bs, be, k->odt, k->o, k->n, &o), done);
        RRC_CHECK_GOTO(s, rrc_convert(be, k->n, a, o), done);
        break;
    }
    RRC_CHECK_GOTO(s, dnf(o, k->n, out), done);
done:
    bs_free(&bs);
    return s;
}

static void test_elementwise(st *t) {
    static const char *ops[] = {"add", "axpy", "scale", "convert"};
    static const rrc_dtype F[2] = {RRC_F32, RRC_BF16};
    for (int op = 0; op < 4; op++) {
        if (!want(t, ops[op])) continue;
        op_begin(t, ops[op]);
        for (int c = 0; c < 9; c++) {
            xcase k = {op, c == 8 ? 1 : 1007, F[c & 1], F[(c >> 1) & 1], F[(c >> 2) & 1], 0, -0.75f, NULL, NULL, NULL};
            if (op <= 1 && c == 8) {
                k.n = 1007;
                k.alias = 1;
                k.odt = k.adt;
            }
            if (op == 2) k.odt = k.adt;
            if (op == 3) k.odt = (c & 1) ? RRC_F32 : RRC_BF16;
            k.a = hrand(t, k.n, 3.0f, k.adt);
            k.b = hrand(t, k.n, 3.0f, k.bdt);
            k.o = hrand(t, k.n, 3.0f, k.odt);
            char name[128];
            snprintf(name, sizeof name, "%s a %s b %s out %s n=%lld%s", ops[op], dn(k.adt), dn(k.bdt), dn(k.odt),
                     (long long)k.n, k.alias ? (op == 0 ? " out=a" : " y=x") : "");
            outspec os = {1, {{k.n, k.odt}}};
            if (k.a && k.b && k.o) check(t, name, CLS_EXACT, 0, NULL, 0, run_ew, &k, &os);
            free(k.a);
            free(k.b);
            free(k.o);
        }
        op_end(t);
    }
}

/* ---- sum_squares ------------------------------------------------------------------------------------------- */

typedef struct qcase {
    int64_t n;
    rrc_dtype dt;
    int accumulate;
    float *x;
    float init;
} qcase;

static rrc_status run_ss(const rrc_backend *be, const void *ctx, float *out) {
    const qcase *k = (const qcase *)ctx;
    bufset bs = {{{0}}, 0};
    rrc_buf x, o;
    rrc_status s;
    RRC_CHECK_GOTO(s, upf(&bs, be, k->dt, k->x, k->n, &x), done);
    RRC_CHECK_GOTO(s, upf(&bs, be, RRC_F32, &k->init, 1, &o), done);
    RRC_CHECK_GOTO(s, rrc_sum_squares(be, k->n, x, o, k->accumulate), done);
    RRC_CHECK_GOTO(s, dnf(o, 1, out), done);
done:
    bs_free(&bs);
    return s;
}

static void test_sum_squares(st *t) {
    if (!want(t, "sum_squares")) return;
    op_begin(t, "sum_squares");
    /* (1 << 20) + 4097: 257 partials, more than the 256 lanes of the final block (its lane loop wraps) */
    static const int64_t ns[] = {1, 255, 4095, 4097, 100003, ((int64_t)1 << 20) + 4097};
    for (int i = 0; i < 6; i++)
        for (int c = 0; c < 3; c++) {
            qcase k = {ns[i], (c & 1) ? RRC_BF16 : RRC_F32, c == 2, NULL, 1.5f};
            k.x = hrand(t, k.n, 1.0f, k.dt);
            char name[128];
            snprintf(name, sizeof name, "sum_squares %s n=%lld%s", dn(k.dt), (long long)k.n,
                     k.accumulate ? " accumulate" : "");
            outspec os = {1, {{1, RRC_F32}}};
            /* the reference sums RRC_KC_SS_CHUNK-element blocks sequentially */
            double nred = k.n < 4096 ? (double)k.n : 4096.0 + (double)k.n / 4096.0;
            if (k.x) check(t, name, CLS_REDUCE, nred, NULL, 0, run_ss, &k, &os);
            free(k.x);
        }
    op_end(t);
}

/* ---- cross entropy ------------------------------------------------------------------------------------------ */

typedef struct ccase {
    int64_t rows, vocab, ld;
    rrc_dtype zdt;
    int write_grad, row_scale;
    float grad_scale;
    float *z, *rs;
    int32_t *labels;
} ccase;

static rrc_status ce_call(const rrc_backend *be, const ccase *k, bufset *bs, rrc_buf *z, rrc_buf *loss) {
    rrc_buf lab, rs = rrc_buf_none();
    const int64_t nz = (k->rows - 1) * k->ld + k->vocab;
    RRC_CHECK(upf(bs, be, k->zdt, k->z, nz, z));
    RRC_CHECK(upi(bs, be, k->labels, k->rows, &lab));
    RRC_CHECK(rrc_buf_alloc(be, RRC_F32, k->rows, loss));
    bs->b[bs->n++] = *loss;
    if (k->row_scale) RRC_CHECK(upf(bs, be, RRC_F32, k->rs, k->rows, &rs));
    return rrc_cross_entropy_fwd_bwd(be, k->rows, k->vocab, *z, k->ld, lab, *loss, k->grad_scale, rs, k->write_grad);
}
static rrc_status run_ce(const rrc_backend *be, const void *ctx, float *out) {
    const ccase *k = (const ccase *)ctx;
    bufset bs = {{{0}}, 0};
    rrc_buf z, loss;
    rrc_status s;
    RRC_CHECK_GOTO(s, ce_call(be, k, &bs, &z, &loss), done);
    RRC_CHECK_GOTO(s, dnf(loss, k->rows, out), done);
    RRC_CHECK_GOTO(s, dnf(z, (k->rows - 1) * k->ld + k->vocab, out + k->rows), done);
done:
    bs_free(&bs);
    return s;
}
static rrc_status bad_ce(const rrc_backend *be, const void *ctx, bufset *bs) {
    rrc_buf z, loss;
    return ce_call(be, (const ccase *)ctx, bs, &z, &loss);
}

static void test_cross_entropy(st *t) {
    if (!want(t, "cross_entropy")) return;
    op_begin(t, "cross_entropy_fwd_bwd");
    for (int vi = 0; vi < 2; vi++)
        for (int c = 0; c < 6; c++) {
            ccase k;
            memset(&k, 0, sizeof k);
            k.rows = 9;
            k.vocab = vi ? 1000 : 50;
            k.ld = k.vocab + 3;
            k.zdt = (c & 1) ? RRC_BF16 : RRC_F32;
            k.write_grad = c < 4;
            k.row_scale = (c >> 1) & 1;
            k.grad_scale = 1.0f / 7.0f;
            const int64_t nz = (k.rows - 1) * k.ld + k.vocab;
            k.z = hrand(t, nz, 6.0f, k.zdt);
            k.rs = hrand(t, k.rows, 1.0f, RRC_F32);
            k.labels = (int32_t *)malloc((size_t)k.rows * sizeof(int32_t));
            if (k.z && k.rs && k.labels) {
                for (int64_t r = 0; r < k.rows; r++) k.labels[r] = (int32_t)(rnext(t) % (uint64_t)k.vocab);
                k.labels[2] = RRC_IGNORE_INDEX;
                k.z[4 * k.ld + 0] = -INFINITY; /* -inf logits contribute 0 */
                k.z[4 * k.ld + 7] = -INFINITY;
                char name[160];
                snprintf(name, sizeof name, "cross_entropy %s rows=%lld vocab=%lld ld=%lld%s%s", dn(k.zdt),
                         (long long)k.rows, (long long)k.vocab, (long long)k.ld, k.write_grad ? " grad" : " loss-only",
                         k.row_scale ? " row_scale" : "");
                outspec os = {2, {{k.rows, RRC_F32}, {nz, k.zdt}}};
                check(t, name, CLS_REDUCE, (double)k.vocab, NULL, 0, run_ce, &k, &os);
                if (c == 0 && vi == 0) {
                    int32_t keep = k.labels[6];
                    k.labels[6] = (int32_t)k.vocab;
                    range_check(t, "cross_entropy labels[6] = vocab", bad_ce, &k);
                    k.labels[6] = -5;
                    range_check(t, "cross_entropy labels[6] = -5", bad_ce, &k);
                    k.labels[6] = keep;
                }
            }
            free(k.z);
            free(k.rs);
            free(k.labels);
        }
    op_end(t);
}

/* ---- adamw ------------------------------------------------------------------------------------------------- */

typedef struct acase {
    int64_t n;
    rrc_adamw_hparams h;
    float *p, *g, *m, *v;
} acase;

static rrc_status run_adamw(const rrc_backend *be, const void *ctx, float *out) {
    const acase *k = (const acase *)ctx;
    bufset bs = {{{0}}, 0};
    rrc_buf p, g, m, v;
    rrc_status s;
    RRC_CHECK_GOTO(s, upf(&bs, be, RRC_F32, k->p, k->n, &p), done);
    RRC_CHECK_GOTO(s, upf(&bs, be, RRC_F32, k->g, k->n, &g), done);
    RRC_CHECK_GOTO(s, upf(&bs, be, RRC_F32, k->m, k->n, &m), done);
    RRC_CHECK_GOTO(s, upf(&bs, be, RRC_F32, k->v, k->n, &v), done);
    RRC_CHECK_GOTO(s, rrc_adamw_step(be, k->n, p, g, m, v, &k->h), done);
    RRC_CHECK_GOTO(s, dnf(p, k->n, out), done);
    RRC_CHECK_GOTO(s, dnf(m, k->n, out + k->n), done);
    RRC_CHECK_GOTO(s, dnf(v, k->n, out + 2 * k->n), done);
done:
    bs_free(&bs);
    return s;
}

static void test_adamw(st *t) {
    if (!want(t, "adamw")) return;
    op_begin(t, "adamw_step");
    for (int c = 0; c < 4; c++) {
        acase k;
        memset(&k, 0, sizeof k);
        k.n = 1003;
        k.h.lr = 1e-3;
        k.h.beta1 = 0.9;
        k.h.beta2 = 0.999;
        k.h.eps = 1e-8;
        k.h.weight_decay = (c & 1) ? 0.01 : 0.0;
        k.h.step = (c & 2) ? 7 : 1;
        k.h.grad_scale = (c & 2) ? 0.5f : 1.0f;
        k.p = hrand(t, k.n, 1.0f, RRC_F32);
        k.g = hrand(t, k.n, 1.0f, RRC_F32);
        k.m = hrand(t, k.n, 0.1f, RRC_F32);
        k.v = hrand(t, k.n, 0.01f, RRC_F32);
        if (k.p && k.g && k.m && k.v) {
            for (int64_t i = 0; i < k.n; i++) k.v[i] = fabsf(k.v[i]);
            char name[128];
            snprintf(name, sizeof name, "adamw_step n=%lld step=%lld wd=%g grad_scale=%g", (long long)k.n,
                     (long long)k.h.step, k.h.weight_decay, (double)k.h.grad_scale);
            outspec os = {3, {{k.n, RRC_F32}, {k.n, RRC_F32}, {k.n, RRC_F32}}};
            check(t, name, CLS_EXACT, 0, NULL, 0, run_adamw, &k, &os);
        }
        free(k.p);
        free(k.g);
        free(k.m);
        free(k.v);
    }
    op_end(t);
}

/* ---- memory ------------------------------------------------------------------------------------------------- */

typedef struct mcase {
    int64_t n; /* bytes */
    unsigned char *h;
} mcase;

static rrc_status run_mem(const rrc_backend *be, const void *ctx, float *out) {
    const mcase *k = (const mcase *)ctx;
    void *a = NULL, *b = NULL;
    rrc_status s;
    unsigned char *back = (unsigned char *)malloc((size_t)k->n);
    if (!back) return RRC_ERROR(RRC_ERR_OOM, "selftest memory");
    RRC_CHECK_GOTO(s, be->alloc(be, (size_t)k->n, &a), done);
    RRC_CHECK_GOTO(s, be->alloc(be, (size_t)k->n, &b), done);
    RRC_CHECK_GOTO(s, be->memcpy_h2d(be, a, k->h, (size_t)k->n), done);
    RRC_CHECK_GOTO(s, be->memcpy_d2d(be, b, a, (size_t)k->n), done);
    RRC_CHECK_GOTO(s, be->memset0(be, (char *)b + 3, (size_t)(k->n / 3)), done);
    RRC_CHECK_GOTO(s, be->memcpy_d2h(be, back, b, (size_t)k->n), done);
    RRC_CHECK_GOTO(s, rrc_sync(be), done);
    /* report the bytes as floats (exact compare) */
    for (int64_t i = 0; i < k->n; i++) out[i] = (float)back[i];
done:
    if (a) be->free(be, a);
    if (b) be->free(be, b);
    free(back);
    return s;
}

static void test_memory(st *t) {
    if (!want(t, "memory")) return;
    op_begin(t, "memory");
    static const int64_t ns[] = {1, 4099, 9 * 1024 * 1024 + 13};
    for (int i = 0; i < 3; i++) {
        mcase k = {ns[i], (unsigned char *)malloc((size_t)ns[i])};
        if (!k.h) continue;
        for (int64_t j = 0; j < k.n; j++) k.h[j] = (unsigned char)(rnext(t) & 0xFF);
        char name[96];
        snprintf(name, sizeof name, "h2d, d2d, memset0 (partial), d2h of %lld bytes", (long long)k.n);
        outspec os = {1, {{k.n, RRC_F32}}};
        check(t, name, CLS_EXACT, 0, NULL, 0, run_mem, &k, &os);
        free(k.h);
    }
    op_end(t);
}

/* ------------------------------------------------------------------------------------------------------------ */

int rrc_st_run(const rrc_backend *dev, const rrc_backend *ref, const rrc_st_opts *o, rrc_st_result *res) {
    st t;
    memset(&t, 0, sizeof t);
    rrc_st_opts dflt;
    memset(&dflt, 0, sizeof dflt);
    t.dev = dev;
    t.ref = ref;
    t.o = o ? o : &dflt;
    t.out = t.o->out ? t.o->out : stdout;
    rrc_st_result local;
    memset(&local, 0, sizeof local);
    t.res = res ? res : &local;
    memset(t.res, 0, sizeof *t.res);
    t.rng = t.o->seed ? t.o->seed : 0x5E1F7E57ull;
    fprintf(t.out, "op checks: dev '%s' vs ref '%s' (classes: EXACT bitwise, ULP expf, REDUCE fp32 reassociation, "
                   "GEMM 2k*2^-24*sum|a||b|; +1 bf16 ulp for bf16 outputs of non-EXACT ops)\n",
            dev->name, ref->name);
    test_memory(&t);
    test_gemm(&t);
    test_embedding(&t);
    test_bias(&t);
    test_rmsnorm(&t);
    test_rope(&t);
    test_softmax(&t);
    test_swiglu(&t);
    test_elementwise(&t);
    test_sum_squares(&t);
    test_cross_entropy(&t);
    test_adamw(&t);
    memcpy(t.res->gemm_paths, t.paths, sizeof t.paths);
    memcpy(t.res->gemm_tally, t.tally, sizeof t.tally);
    t.res->gemm_cases = t.n_gemm_cases;
    t.res->gemm_chunked_cases = t.n_chunked_cases;
    t.res->gemm_kernel_cases = t.n_kernel_cases;
    fprintf(t.out, "summary: %d value checks (%d failed), %d determinism checks (%d failed), %d range-error checks (%d "
                   "failed), %d gemm path checks (%d failed)\n",
            t.res->checks, t.res->failures, t.res->det_checks, t.res->det_failures, t.res->range_checks,
            t.res->range_failures, t.res->path_checks, t.res->path_failures);
    int fails = t.res->failures + t.res->det_failures + t.res->range_failures + t.res->path_failures;
    if (t.res->checks + t.res->range_checks == 0) {
        /* e.g. --only with a typo: nothing was tested, which must not read as a pass */
        fprintf(t.out, "FAIL: no check ran%s%s%s\n", t.o->only ? " (--only '" : "", t.o->only ? t.o->only : "",
                t.o->only ? "' matches no op)" : "");
        fails++;
    }
    return fails;
}
