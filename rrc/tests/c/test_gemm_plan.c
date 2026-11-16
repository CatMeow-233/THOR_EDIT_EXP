/* Tests of the row-major -> column-major GEMM plan (rrc/cuda/gemm_plan.c): a column-major
 * reference (rrc_kc_item_lt_matmul, the device fallback kernel body) consuming the plan's chunks,
 * conversions and calls must reproduce rrc_backend_cpu_f32acc's rrc_gemm bit for bit, for all 8
 * dtype combinations x 4 transposes x {plain, batched, broadcast A, broadcast B, interleaved C} x
 * every ladder level x {default scratch cap, a tiny cap that forces chunking} x beta {0, 0.75};
 * plus the promotion table, bounded scratch for an LM-head-sized bf16 weight, pointer-derived
 * alignment, TF32 flag propagation, and the ladder driver with simulated library refusals. */
#include "gemm_plan.h"
#include "kc_emu.h"
#include "ops_util.h"

static const rrc_dtype k_f[2] = {RRC_F32, RRC_BF16};

enum { LP_PLAIN = 0, LP_BATCH, LP_BCAST_A, LP_BCAST_B, LP_INTERLEAVE_C, LP_N };

typedef struct {
    rrc_gemm_args g;
    int64_t na, nb, nc;
    float *A, *B, *C;
} pcase;

static void pcase_make(pcase *p, rt_rng *rng, rrc_trans ta, rrc_trans tb, int64_t m, int64_t n, int64_t k, int layout,
                       rrc_dtype da, rrc_dtype db, rrc_dtype dc, float beta) {
    memset(p, 0, sizeof *p);
    const int64_t B = layout == LP_PLAIN ? 1 : 3;
    const int64_t ar = ta == RRC_OP_N ? m : k, ac = ta == RRC_OP_N ? k : m;
    const int64_t br = tb == RRC_OP_N ? k : n, bc = tb == RRC_OP_N ? n : k;
    rrc_gemm_args *g = &p->g;
    g->trans_a = ta;
    g->trans_b = tb;
    g->m = m;
    g->n = n;
    g->k = k;
    g->alpha = 1.25f;
    g->beta = beta;
    g->batch = B;
    g->lda = ac + 3;
    g->ldb = bc + 1;
    g->stride_a = layout == LP_BCAST_A ? 0 : ar * g->lda + 1;
    g->stride_b = layout == LP_BCAST_B ? 0 : br * g->ldb + 5;
    if (layout == LP_INTERLEAVE_C) {
        g->ldc = B * n + 1;
        g->stride_c = n;
    } else {
        g->ldc = n + 2;
        g->stride_c = m * g->ldc + 3;
    }
    p->na = (B - 1) * g->stride_a + ar * g->lda;
    p->nb = (B - 1) * g->stride_b + br * g->ldb;
    p->nc = (B - 1) * g->stride_c + m * g->ldc;
    p->A = (float *)malloc((size_t)p->na * 4);
    p->B = (float *)malloc((size_t)p->nb * 4);
    p->C = (float *)malloc((size_t)p->nc * 4);
    rt_fill(rng, p->A, p->na, 1.0f);
    rt_fill(rng, p->B, p->nb, 1.0f);
    rt_fill(rng, p->C, p->nc, 1.0f);
    if (beta == 0.0f)
        for (int64_t i = 0; i < p->nc; i += 3) p->C[i] = NAN;
    rt_round_inplace(p->A, p->na, da);
    rt_round_inplace(p->B, p->nb, db);
    rt_round_inplace(p->C, p->nc, dc);
    g->a.dtype = da;
    g->b.dtype = db;
    g->c.dtype = dc;
}
static void pcase_free(pcase *p) {
    free(p->A);
    free(p->B);
    free(p->C);
}

/* Execute a plan on host buffers in index order (conversions, then the column-major call). */
static int exec_plan(const rrc_gemm_plan *pl) {
    float *scratch = NULL;
    if (pl->scratch_elems > 0) {
        scratch = (float *)malloc((size_t)pl->scratch_elems * sizeof(float)); /* exactly the size the plan asks */
        if (!scratch) return 0;
    }
    rrc_gemm_chunk ch;
    for (int64_t i = 0; i < pl->n_chunks; i++) {
        rrc_gemm_plan_chunk(pl, scratch, i, &ch);
        for (int j = 0; j < ch.n_pre; j++) {
            const rrc_kc_conv2d_p *v = &ch.pre[j];
            for (int64_t e = 0; e < v->batch * v->rows * v->cols; e++) rrc_kc_item_conv2d(v, e);
        }
        /* reverse order: any item order must give the same bits */
        for (int64_t e = ch.call.p.batch * ch.call.p.m * ch.call.p.n - 1; e >= 0; e--) rrc_kc_item_lt_matmul(&ch.call.p, e);
        if (ch.has_post)
            for (int64_t e = 0; e < ch.post.batch * ch.post.rows * ch.post.cols; e++) rrc_kc_item_conv2d(&ch.post, e);
    }
    free(scratch);
    return 1;
}

static int same_bits(float a, float b) { return rrc_f32_bits(a) == rrc_f32_bits(b); }

RT_TEST(plan_reference_matches_cpu_gemm) {
    rt_rng rng = {4242};
    const rrc_backend *cpu = rrc_backend_cpu_f32acc();
    int n_cases = 0, n_chunked = 0;
    for (int combo = 0; combo < 8; combo++)
        for (int tr = 0; tr < 4; tr++)
            for (int layout = 0; layout < LP_N; layout++)
                for (int bz = 0; bz < 2; bz++) {
                    const rrc_dtype da = k_f[(combo >> 2) & 1], db = k_f[(combo >> 1) & 1], dc = k_f[combo & 1];
                    const rrc_trans ta = (tr & 2) ? RRC_OP_T : RRC_OP_N, tb = (tr & 1) ? RRC_OP_T : RRC_OP_N;
                    pcase pc;
                    pcase_make(&pc, &rng, ta, tb, 13, 11, 9, layout, da, db, dc, bz ? 0.75f : 0.0f);
                    /* reference: the CPU backend through the op wrapper */
                    rrc_buf a = rt_upload(cpu, da, pc.A, pc.na), b = rt_upload(cpu, db, pc.B, pc.nb),
                            c = rt_upload(cpu, dc, pc.C, pc.nc);
                    rrc_gemm_args gr = pc.g;
                    gr.a = a;
                    gr.b = b;
                    gr.c = c;
                    RT_OK(rrc_gemm(cpu, &gr));
                    float *want = rt_download(c, pc.nc);
                    for (int level = 0; level <= RRC_GEMM_LEVEL_KERNEL; level++)
                        for (int cap = 0; cap < 2; cap++) {
                            rrc_buf c2 = rt_upload(cpu, dc, pc.C, pc.nc);
                            rrc_gemm_args g = pc.g;
                            g.a = a;
                            g.b = b;
                            g.c = c2;
                            rrc_gemm_plan_opts po = {cap ? 40 : 0, 0, 0};
                            rrc_gemm_plan pl;
                            RT_OK(rrc_gemm_plan_make(&g, &po, (rrc_gemm_level)level, &pl));
                            if (pl.n_chunks > 1) n_chunked++;
                            RT_ASSERT(exec_plan(&pl), "scratch allocation");
                            float *got = rt_download(c2, pc.nc);
                            int64_t bad = -1;
                            for (int64_t i = 0; i < pc.nc && bad < 0; i++)
                                if (!same_bits(got[i], want[i]) && !(isnan(got[i]) && isnan(want[i]))) bad = i;
                            char desc[128];
                            rrc_gemm_plan_describe(&pl, desc, sizeof desc);
                            RT_EXPECT(bad < 0,
                                      "%s x %s -> %s ta=%d tb=%d layout=%d beta=%g level=%d cap=%d (%s): element %lld "
                                      "got %.9g want %.9g",
                                      rrc_dtype_name(da), rrc_dtype_name(db), rrc_dtype_name(dc), ta, tb, layout,
                                      (double)pc.g.beta, level, cap, desc, (long long)bad,
                                      bad >= 0 ? (double)got[bad] : 0.0, bad >= 0 ? (double)want[bad] : 0.0);
                            free(got);
                            rrc_buf_free(&c2);
                            n_cases++;
                        }
                    free(want);
                    rrc_buf_free(&a);
                    rrc_buf_free(&b);
                    rrc_buf_free(&c);
                    pcase_free(&pc);
                }
    RT_EXPECT(n_cases == 8 * 4 * LP_N * 2 * 3 * 2, "case count %d", n_cases);
    RT_EXPECT(n_chunked > 100, "the tiny cap must force multi-chunk plans (%d chunked)", n_chunked);
}

/* Scratch leading dimensions >= 64 that are padded (cols not a multiple of 8) and batch chunks with
 * 1 < nb < batch, bit-identical to the CPU backend. */
RT_TEST(plan_padded_ld_and_batch_chunks) {
    rt_rng rng = {991};
    const rrc_backend *cpu = rrc_backend_cpu_f32acc();
    static const int combos[3][4] = {/* da db dc level */ {1, 0, 1, 0}, {0, 1, 1, 0}, {1, 1, 1, 1}};
    int n_mid = 0;
    for (int ci = 0; ci < 3; ci++)
        for (int layout = LP_BATCH; layout <= LP_BCAST_B; layout++)
            for (int tr = 0; tr < 4; tr++) {
                const rrc_dtype da = k_f[combos[ci][0]], db = k_f[combos[ci][1]], dc = k_f[combos[ci][2]];
                const rrc_gemm_level level = (rrc_gemm_level)combos[ci][3];
                const rrc_trans ta = (tr & 2) ? RRC_OP_T : RRC_OP_N, tb = (tr & 1) ? RRC_OP_T : RRC_OP_N;
                pcase pc;
                pcase_make(&pc, &rng, ta, tb, 70, 67, 69, layout, da, db, dc, 0.5f);
                pc.g.batch = 3;
                rrc_buf a = rt_upload(cpu, da, pc.A, pc.na), b = rt_upload(cpu, db, pc.B, pc.nb),
                        c = rt_upload(cpu, dc, pc.C, pc.nc), c2 = rt_upload(cpu, dc, pc.C, pc.nc);
                rrc_gemm_args g = pc.g;
                g.a = a;
                g.b = b;
                g.c = c;
                RT_OK(rrc_gemm(cpu, &g));
                g.c = c2;
                rrc_gemm_plan pl;
                RT_OK(rrc_gemm_plan_make(&g, NULL, level, &pl));
                RT_ASSERT(pl.nb == 3 && pl.n_chunks == 1, "default cap: one chunk");
                rrc_gemm_plan_opts po = {pl.scratch_elems * 2 / 3 + 1, 0, 0}; /* room for about 2 of 3 slices */
                RT_OK(rrc_gemm_plan_make(&g, &po, level, &pl));
                if (pl.nb > 1 && pl.nb < 3) n_mid++;
                rrc_gemm_chunk ch;
                rrc_gemm_plan_chunk(&pl, (void *)&rng, 0, &ch);
                int padded = 0;
                for (int j = 0; j < ch.n_pre; j++) padded |= ch.pre[j].dld > ch.pre[j].cols;
                RT_EXPECT(padded, "a padded scratch ld (cols >= 64, not a multiple of 8) is exercised");
                RT_ASSERT(exec_plan(&pl), "scratch allocation");
                float *want = rt_download(c, pc.nc), *got = rt_download(c2, pc.nc);
                int same = 1;
                for (int64_t i = 0; i < pc.nc; i++) same &= same_bits(got[i], want[i]);
                RT_EXPECT(same, "combo %d layout %d tr %d: nb %lld lm %lld ln %lld chunks %lld differs from the CPU", ci,
                          layout, tr, (long long)pl.nb, (long long)pl.lm, (long long)pl.ln, (long long)pl.n_chunks);
                free(want);
                free(got);
                rrc_buf_free(&a);
                rrc_buf_free(&b);
                rrc_buf_free(&c);
                rrc_buf_free(&c2);
                pcase_free(&pc);
            }
    RT_EXPECT(n_mid >= 6, "batch chunks with 1 < nb < batch exercised in %d cases", n_mid);
}

RT_TEST(plan_promotion_table) {
    /* level 0: native for f32 f32 f32 / bf16 bf16 f32 / bf16 bf16 bf16; promote the bf16 operand of
     * a mixed pair; bf16 C next to f32 operands via f32 scratch. level 1: everything f32. */
    static const int want0[8][3] = {
        {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {0, 0, 0}, {0, 0, 0},
    };
    char fake[64];
    for (int combo = 0; combo < 8; combo++) {
        rrc_gemm_args g;
        memset(&g, 0, sizeof g);
        g.trans_a = RRC_OP_N;
        g.trans_b = RRC_OP_T;
        g.m = 4;
        g.n = 5;
        g.k = 6;
        g.alpha = 1.0f;
        g.batch = 1;
        g.lda = 6;
        g.ldb = 6;
        g.ldc = 5;
        g.a.ptr = g.b.ptr = g.c.ptr = fake;
        g.a.dtype = k_f[(combo >> 2) & 1];
        g.b.dtype = k_f[(combo >> 1) & 1];
        g.c.dtype = k_f[combo & 1];
        rrc_gemm_plan p0, p1, p2;
        RT_OK(rrc_gemm_plan_make(&g, NULL, RRC_GEMM_LEVEL_NATIVE, &p0));
        RT_OK(rrc_gemm_plan_make(&g, NULL, RRC_GEMM_LEVEL_PROMOTE_ALL, &p1));
        RT_OK(rrc_gemm_plan_make(&g, NULL, RRC_GEMM_LEVEL_KERNEL, &p2));
        RT_EXPECT(p0.promote_a == want0[combo][0] && p0.promote_b == want0[combo][1] && p0.c_scratch == want0[combo][2],
                  "combo %d level 0 flags %d %d %d", combo, p0.promote_a, p0.promote_b, p0.c_scratch);
        RT_EXPECT(p1.promote_a == (g.a.dtype == RRC_BF16) && p1.promote_b == (g.b.dtype == RRC_BF16) &&
                      p1.c_scratch == (g.c.dtype == RRC_BF16),
                  "combo %d level 1 flags", combo);
        RT_EXPECT(!p2.promote_a && !p2.promote_b && !p2.c_scratch && p2.n_chunks == 1 && p2.scratch_elems == 0,
                  "kernel level converts nothing");
        rrc_gemm_chunk ch;
        rrc_gemm_plan_chunk(&p0, NULL, 0, &ch);
        /* operand swap of the row-major -> column-major translation */
        RT_EXPECT(ch.call.p.transa == 1 && ch.call.p.transb == 0 && ch.call.p.m == 5 && ch.call.p.n == 4 &&
                      ch.call.p.k == 6,
                  "swap: transa=%d transb=%d M=%lld N=%lld", ch.call.p.transa, ch.call.p.transb, (long long)ch.call.p.m,
                  (long long)ch.call.p.n);
        /* no plan ever narrows an f32 operand */
        RT_EXPECT(!(g.a.dtype == RRC_F32 && ch.call.p.bdt == RRC_BF16) && !(g.b.dtype == RRC_F32 && ch.call.p.adt == RRC_BF16),
                  "f32 operand demoted");
    }
}

RT_TEST(plan_bounded_scratch_lm_head) {
    /* dh = dlogits [T, V] (f32) x W_lm [V, D] (bf16), T = 4096, V = 152064, D = 4096: B is promoted
     * in column chunks, never whole (V * D * 4 bytes = 2.3 GiB) */
    char fake[64];
    rrc_gemm_args g;
    memset(&g, 0, sizeof g);
    g.trans_a = RRC_OP_N;
    g.trans_b = RRC_OP_N;
    g.m = 4096;
    g.n = 4096;
    g.k = 152064;
    g.alpha = 1.0f;
    g.batch = 1;
    g.lda = g.k;
    g.ldb = g.n;
    g.ldc = g.n;
    g.a.ptr = g.b.ptr = g.c.ptr = fake;
    g.a.dtype = RRC_F32;
    g.b.dtype = RRC_BF16;
    g.c.dtype = RRC_BF16;
    rrc_gemm_plan p;
    RT_OK(rrc_gemm_plan_make(&g, NULL, RRC_GEMM_LEVEL_NATIVE, &p));
    RT_EXPECT(p.promote_b && p.c_scratch, "B promoted, C via scratch");
    RT_EXPECT(p.scratch_elems <= RRC_GEMM_SCRATCH_CAP_DEFAULT, "scratch %lld > cap", (long long)p.scratch_elems);
    RT_EXPECT(p.n_chunks > 30 && p.lm == g.m, "column chunks: %lld chunks, lm %lld, ln %lld", (long long)p.n_chunks,
              (long long)p.lm, (long long)p.ln);
    /* both operands promoted (level 1, A bf16 too): 2-D chunking stays within the cap */
    g.a.dtype = RRC_BF16;
    RT_OK(rrc_gemm_plan_make(&g, NULL, RRC_GEMM_LEVEL_PROMOTE_ALL, &p));
    RT_EXPECT(p.promote_a && p.promote_b && p.scratch_elems <= RRC_GEMM_SCRATCH_CAP_DEFAULT,
              "2-D chunking: scratch %lld", (long long)p.scratch_elems);
    /* every chunk's regions lie inside the scratch, chunks tile C exactly once */
    int64_t covered = 0;
    rrc_gemm_chunk ch;
    for (int64_t i = 0; i < p.n_chunks; i++) {
        rrc_gemm_plan_chunk(&p, (void *)fake, i, &ch);
        covered += ch.call.p.m * ch.call.p.n * ch.call.p.batch;
        for (int j = 0; j < ch.n_pre; j++) {
            const rrc_kc_conv2d_p *v = &ch.pre[j];
            int64_t last = (v->batch - 1) * v->dstride + (v->rows - 1) * v->dld + v->cols;
            int64_t off = (int64_t)(((const char *)v->dst - fake) / 4);
            RT_ASSERT(off >= 0 && off + last <= p.scratch_elems, "chunk %lld region outside scratch", (long long)i);
        }
    }
    RT_EXPECT(covered == g.m * g.n, "chunks cover C: %lld of %lld", (long long)covered, (long long)(g.m * g.n));
    /* one column of k elements above the cap: the minimum chunk is used, scratch_elems reports it */
    rrc_gemm_plan_opts small = {1000, 0, 0};
    g.m = 3;
    g.n = 2;
    g.k = 5000;
    g.lda = g.k;
    g.ldb = g.n;
    g.ldc = g.n;
    RT_OK(rrc_gemm_plan_make(&g, &small, RRC_GEMM_LEVEL_PROMOTE_ALL, &p));
    RT_EXPECT(p.lm == 1 && p.ln == 1 && p.scratch_elems > 1000 && p.scratch_elems < 3 * 5000 + 1000,
              "minimum chunk: lm %lld ln %lld scratch %lld", (long long)p.lm, (long long)p.ln, (long long)p.scratch_elems);
}

RT_TEST(plan_alignment_and_tf32) {
    char *base = (char *)(uintptr_t)0x10000; /* never dereferenced */
    RT_EXPECT(rrc_gemm_align_bytes(base, 4, 64, 0, 1, 256) == 256, "aligned");
    RT_EXPECT(rrc_gemm_align_bytes(base + 2, 2, 64, 0, 1, 256) == 2, "bf16 view at element 1");
    RT_EXPECT(rrc_gemm_align_bytes(base, 4, 3, 0, 1, 256) == 4, "odd ld");
    RT_EXPECT(rrc_gemm_align_bytes(base, 2, 64, 1, 2, 256) == 2, "odd batch stride");
    RT_EXPECT(rrc_gemm_align_bytes(base, 2, 64, 1, 1, 256) == 128, "stride ignored for batch 1");
    RT_EXPECT(rrc_gemm_align_bytes(base, 4, 64, 0, 4, 16) == 16, "cap");
    rrc_gemm_args g;
    memset(&g, 0, sizeof g);
    g.trans_a = g.trans_b = RRC_OP_N;
    g.m = g.n = g.k = 8;
    g.alpha = 1.0f;
    g.batch = 1;
    g.lda = g.ldb = g.ldc = 8;
    g.a.ptr = base + 4;
    g.b.ptr = base;
    g.c.ptr = base + 32;
    g.a.dtype = g.b.dtype = g.c.dtype = RRC_F32;
    rrc_gemm_plan_opts t = {0, 1, 0};
    rrc_gemm_plan p;
    rrc_gemm_chunk ch;
    RT_OK(rrc_gemm_plan_make(&g, &t, RRC_GEMM_LEVEL_NATIVE, &p));
    rrc_gemm_plan_chunk(&p, NULL, 0, &ch);
    RT_EXPECT(ch.call.tf32 == 1, "tf32 on for f32 x f32 when allowed");
    RT_EXPECT(ch.call.align_b == 4 && ch.call.align_a == 32 && ch.call.align_c == 32, "alignment of swapped operands: %u %u %u",
              ch.call.align_a, ch.call.align_b, ch.call.align_c);
    RT_OK(rrc_gemm_plan_make(&g, NULL, RRC_GEMM_LEVEL_NATIVE, &p));
    rrc_gemm_plan_chunk(&p, NULL, 0, &ch);
    RT_EXPECT(ch.call.tf32 == 0, "tf32 off by default");
    g.a.dtype = g.b.dtype = RRC_BF16;
    RT_OK(rrc_gemm_plan_make(&g, &t, RRC_GEMM_LEVEL_NATIVE, &p));
    rrc_gemm_plan_chunk(&p, NULL, 0, &ch);
    RT_EXPECT(ch.call.tf32 == 0, "tf32 never for bf16 operands");
}

/* The ladder driver with a simulated library: refusals move to the next level; every path gives
 * the CPU backend's bits. */
static void ladder_case(rt_rng *rng, unsigned reject, unsigned min_align, int64_t a_off, rrc_dtype da, rrc_dtype db,
                        rrc_dtype dc, rrc_gemm_level want_level, const char *what) {
    rrc_emu_options eo;
    rrc_emu_options_default(&eo);
    eo.lt_reject_mask = reject;
    eo.lt_min_align = min_align;
    const rrc_backend *emu = NULL;
    RT_OK(rrc_emu_backend_create(&eo, &emu));
    const rrc_backend *cpu = rrc_backend_cpu_f32acc();
    const int64_t m = 16, n = 24, k = 32, lda = 40, ldb = 24, ldc = 32;
    float *A = (float *)malloc((size_t)(m * lda + a_off) * 4), *B = (float *)malloc((size_t)(k * ldb) * 4),
          *C = (float *)malloc((size_t)(m * ldc) * 4);
    rt_fill(rng, A, m * lda + a_off, 1.0f);
    rt_fill(rng, B, k * ldb, 1.0f);
    rt_fill(rng, C, m * ldc, 1.0f);
    float *res[2];
    const rrc_backend *bes[2] = {cpu, emu};
    for (int i = 0; i < 2; i++) {
        rrc_buf a = rt_upload(bes[i], da, A, m * lda + a_off), b = rt_upload(bes[i], db, B, k * ldb),
                c = rt_upload(bes[i], dc, C, m * ldc);
        rrc_gemm_args g = {RRC_OP_N, RRC_OP_N, m, n, k, 0.5f, 1.0f, rrc_view_from(a, a_off), lda, 0, b, ldb, 0, c, ldc, 0, 1};
        RT_OK(rrc_gemm(bes[i], &g));
        res[i] = rt_download(c, m * ldc);
        rrc_buf_free(&a);
        rrc_buf_free(&b);
        rrc_buf_free(&c);
    }
    rrc_gemm_exec_report rep;
    RT_EXPECT(rrc_emu_last_gemm(emu, &rep), "report");
    RT_EXPECT(rep.level == want_level, "%s: level %d want %d (%s)", what, (int)rep.level, (int)want_level, rep.desc);
    int same = 1;
    for (int64_t i = 0; i < m * ldc; i++) same &= same_bits(res[0][i], res[1][i]);
    RT_EXPECT(same, "%s: result differs from the CPU backend (%s)", what, rep.desc);
    free(res[0]);
    free(res[1]);
    free(A);
    free(B);
    free(C);
    rrc_emu_backend_destroy(emu);
}

RT_TEST(ladder_levels) {
    rt_rng rng = {77};
    /* bit = 4*adt + 2*bdt + cdt of the column-major call (A = rrc's B) */
    ladder_case(&rng, 0, 0, 0, RRC_BF16, RRC_BF16, RRC_F32, RRC_GEMM_LEVEL_NATIVE, "bf16 bf16 f32 native");
    ladder_case(&rng, 1u << 6, 0, 0, RRC_BF16, RRC_BF16, RRC_F32, RRC_GEMM_LEVEL_PROMOTE_ALL,
                "bf16 bf16 f32 refused -> promote all");
    ladder_case(&rng, 1u << 7 | 1u << 0, 0, 0, RRC_BF16, RRC_BF16, RRC_BF16, RRC_GEMM_LEVEL_KERNEL,
                "bbb and fff refused -> kernel");
    ladder_case(&rng, 0xFFu, 0, 0, RRC_F32, RRC_BF16, RRC_F32, RRC_GEMM_LEVEL_KERNEL, "everything refused -> kernel");
    ladder_case(&rng, 0, 16, 1, RRC_BF16, RRC_BF16, RRC_F32, RRC_GEMM_LEVEL_PROMOTE_ALL,
                "misaligned bf16 A -> promoted into aligned scratch");
    ladder_case(&rng, 0, 16, 1, RRC_F32, RRC_F32, RRC_F32, RRC_GEMM_LEVEL_KERNEL, "misaligned f32 A -> kernel");
    ladder_case(&rng, 0, 0, 0, RRC_F32, RRC_BF16, RRC_BF16, RRC_GEMM_LEVEL_NATIVE, "mixed: promoted at level 0");
}

const rt_case rt_suite_gemm_plan[] = {
    RT_CASE(plan_reference_matches_cpu_gemm), RT_CASE(plan_padded_ld_and_batch_chunks), RT_CASE(plan_promotion_table), RT_CASE(plan_bounded_scratch_lm_head),
    RT_CASE(plan_alignment_and_tf32),         RT_CASE(ladder_levels),
};
const size_t rt_suite_gemm_plan_n = sizeof rt_suite_gemm_plan / sizeof rt_suite_gemm_plan[0];
