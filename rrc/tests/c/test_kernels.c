/* Direct tests of the kernels_common element/row functions (the code the CUDA kernels will call
 * per thread): edge cases, merge properties of the online logsumexp, fp32 vs double accumulation
 * agreement, typed load/store rounding. */
#include "ops_util.h"
#include "rrc_kc.h"

RT_TEST(kc_load_store) {
    rrc_bf16 hb[2];
    float f[2];
    rrc_kc_st(hb, RRC_BF16, 1, 1.00390625f); /* tie -> even */
    rrc_kc_st(f, RRC_F32, 0, 1.00390625f);
    RT_EXPECT(hb[1] == 0x3F80, "bf16 store rounds RNE: 0x%04x", hb[1]);
    RT_EXPECT(f[0] == 1.00390625f, "f32 store exact");
    RT_EXPECT(rrc_kc_ld(hb, RRC_BF16, 1) == 1.0f && rrc_kc_ld(f, RRC_F32, 0) == 1.00390625f, "loads");
    RT_EXPECT(rrc_kc_round_to(RRC_BF16, 1.01171875f) == rrc_bf16_to_f32(0x3F82), "round_to");
    RT_EXPECT(rrc_kc_round_to(RRC_F32, 1.01171875f) == 1.01171875f, "round_to f32 identity");
}

RT_TEST(kc_lse_merge) {
    rt_rng rng = {301};
    float x[1000];
    rt_fill(&rng, x, 1000, 30.0f);
    x[500] = 85.0f; /* exp overflow without the shift */
    rrc_kc_lse seq = rrc_kc_lse_init();
    for (int i = 0; i < 1000; i++) seq = rrc_kc_lse_push(seq, x[i]);
    /* tree merge in blocks of 37 (any order must give the same value up to rounding) */
    rrc_kc_lse parts[28];
    int np = 0;
    for (int b = 0; b < 1000; b += 37) {
        rrc_kc_lse a = rrc_kc_lse_init();
        for (int i = b; i < b + 37 && i < 1000; i++) a = rrc_kc_lse_push(a, x[i]);
        parts[np++] = a;
    }
    while (np > 1) {
        int k = 0;
        for (int i = 0; i < np; i += 2) parts[k++] = i + 1 < np ? rrc_kc_lse_merge(parts[i + 1], parts[i]) : parts[i];
        np = k;
    }
    double ref_m = -INFINITY, ref_s = 0;
    for (int i = 0; i < 1000; i++) ref_m = fmax(ref_m, x[i]);
    for (int i = 0; i < 1000; i++) ref_s += exp((double)x[i] - ref_m);
    double ref = ref_m + log(ref_s);
    RT_CLOSE(seq.m + logf(seq.s), ref, 1e-4, 1e-6);
    RT_CLOSE(parts[0].m + logf(parts[0].s), ref, 1e-4, 1e-6);
    RT_EXPECT(seq.m == 85.0f && parts[0].m == 85.0f, "max tracked");
    rrc_kc_lse e = rrc_kc_lse_init();
    rrc_kc_lse m1 = rrc_kc_lse_merge(e, seq), m2 = rrc_kc_lse_merge(seq, e);
    RT_EXPECT(m1.m == seq.m && m1.s == seq.s && m2.m == seq.m && m2.s == seq.s, "empty is the identity");
    rrc_kc_lse ee = rrc_kc_lse_merge(e, e);
    RT_EXPECT(ee.m == -INFINITY && ee.s == 0.0f, "empty + empty");
}

RT_TEST(kc_silu_edges) {
    RT_EXPECT(rrc_kc_silu(0.0f) == 0.0f && rrc_kc_sigmoid(0.0f) == 0.5f && rrc_kc_silu_grad(0.0f) == 0.5f, "zero");
    RT_EXPECT(rrc_kc_silu(100.0f) == 100.0f && rrc_kc_silu_grad(100.0f) == 1.0f, "large positive");
    float neg = rrc_kc_silu(-100.0f);
    RT_EXPECT(neg <= 0.0f && neg > -1e-30f, "large negative silu %g", (double)neg);
    RT_EXPECT(rrc_kc_silu(-1000.0f) == 0.0f && !isnan(rrc_kc_silu_grad(-1000.0f)), "exp overflow handled");
    float dg, du;
    rrc_kc_swiglu_bwd(2.0f, 3.0f, 0.5f, &dg, &du);
    double s = 1.0 / (1.0 + exp(-2.0));
    RT_CLOSE(du, 0.5 * 2.0 * s, 0, 1e-6);
    RT_CLOSE(dg, 0.5 * 3.0 * s * (1 + 2.0 * (1 - s)), 0, 1e-6);
    RT_CLOSE(rrc_kc_swiglu_fwd(2.0f, 3.0f), 2.0 * s * 3.0, 0, 1e-6);
}

RT_TEST(kc_lerp_branches) {
    /* torch lerp: both branches agree with the exact value up to rounding */
    float ws[] = {0.0f, 0.1f, 0.4999f, 0.5f, 0.7f, 1.0f};
    for (size_t i = 0; i < sizeof ws / sizeof ws[0]; i++) {
        float r = rrc_kc_lerp(2.0f, -6.0f, ws[i]);
        RT_CLOSE(r, 2.0 + ws[i] * (-8.0), 1e-6, 1e-6);
    }
    RT_EXPECT(rrc_kc_lerp(2.0f, -6.0f, 1.0f) == -6.0f, "w = 1 hits end exactly");
    RT_EXPECT(rrc_kc_lerp(2.0f, -6.0f, 0.0f) == 2.0f, "w = 0 keeps self exactly");
    rrc_kc_adamw_scalars s = rrc_kc_adamw_prepare(1e-3, 0.9, 0.999, 1e-8, 0.0, 1, 1.0f);
    RT_EXPECT(!s.apply_wd && s.lerp_w == (float)(1.0 - 0.9) && s.bc2_sqrt == (float)pow(1 - 0.999, 0.5) &&
                  s.neg_step_size == (float)(-(1e-3 / (1 - 0.9))),
              "adamw scalars");
    /* first step from zero state: update magnitude ~ lr (Adam property) */
    float p = 1.0f, m = 0.0f, v = 0.0f;
    rrc_kc_adamw_elem(&p, 0.3f, &m, &v, &s);
    RT_CLOSE(p, 1.0 - 1e-3, 1e-7, 0);
}

RT_TEST(kc_softmax_rows) {
    float s[6] = {1.0f, 2.0f, -INFINITY, 4.0f, 0.5f, 3.0f};
    float p[6];
    /* non-causal, no docs, a -inf score: weight 0 */
    rrc_kc_attn_softmax_row_fwd(s, RRC_F32, p, RRC_F32, 6, 1.0f, 0, 0, NULL, 1);
    RT_EXPECT(p[2] == 0.0f, "-inf -> 0");
    double z = 0;
    for (int j = 0; j < 6; j++) z += p[j];
    RT_CLOSE(z, 1.0, 1e-6, 0);
    /* causal at q = 1: only keys 0, 1 */
    rrc_kc_attn_softmax_row_fwd(s, RRC_F32, p, RRC_F32, 6, 1.0f, 1, 1, NULL, 0);
    RT_CLOSE(p[1], exp(2.0) / (exp(1.0) + exp(2.0)), 0, 1e-6);
    RT_EXPECT(p[2] == 0 && p[3] == 0 && p[4] == 0 && p[5] == 0, "causal zeros");
    /* doc mask */
    int32_t doc[6] = {0, 1, 1, 0, 1, 1};
    rrc_kc_attn_softmax_row_fwd(s, RRC_F32, p, RRC_F32, 6, 0.5f, 4, 1, doc, 1);
    double zz = exp(0.5 * 2.0) + exp(0.5 * 0.5); /* keys 1 (-inf key 2 is excluded by its score) and 4 */
    RT_CLOSE(p[1], exp(0.5 * 2.0) / zz, 0, 1e-6);
    RT_CLOSE(p[4], exp(0.25) / zz, 0, 1e-6);
    RT_EXPECT(p[0] == 0 && p[3] == 0 && p[5] == 0 && p[2] == 0, "doc zeros");
    /* all masked */
    float all_inf[3] = {-INFINITY, -INFINITY, -INFINITY};
    rrc_kc_attn_softmax_row_fwd(all_inf, RRC_F32, p, RRC_F32, 3, 1.0f, 2, 0, NULL, 0);
    RT_EXPECT(p[0] == 0 && p[1] == 0 && p[2] == 0, "all -inf -> zeros (no NaN)");
    /* backward of a uniform dp is zero (softmax rows sum to 1) */
    float pr[4] = {0.1f, 0.2f, 0.3f, 0.4f}, dp[4] = {2, 2, 2, 2}, ds[4];
    rrc_kc_attn_softmax_row_bwd(pr, RRC_F32, dp, RRC_F32, ds, RRC_F32, 4, 0.5f, 1);
    for (int j = 0; j < 4; j++) RT_CLOSE(ds[j], 0.0, 1e-7, 0);
}

RT_TEST(kc_acc32_vs_acc64) {
    rt_rng rng = {311};
    float x[4096], w[4096], z1[4096], z2[4096], y1[4096], y2[4096];
    rt_fill(&rng, x, 4096, 3.0f);
    rt_fill(&rng, w, 4096, 1.0f);
    rrc_kc_rmsnorm_row_fwd(x, RRC_F32, w, RRC_F32, y1, RRC_F32, 4096, 1e-6f, 1);
    rrc_kc_rmsnorm_row_fwd(x, RRC_F32, w, RRC_F32, y2, RRC_F32, 4096, 1e-6f, 0);
    double md = 0;
    for (int i = 0; i < 4096; i++) md = fmax(md, fabs((double)y1[i] - y2[i]) / (1e-6 + fabs(y1[i])));
    RT_EXPECT(md < 1e-5, "rmsnorm acc32 vs acc64 rel diff %g", md);
    memcpy(z1, x, sizeof x);
    memcpy(z2, x, sizeof x);
    float l1 = rrc_kc_xent_row(z1, RRC_F32, 4096, 17, 1.0f, 1, 1);
    float l2 = rrc_kc_xent_row(z2, RRC_F32, 4096, 17, 1.0f, 1, 0);
    RT_CLOSE(l2, l1, 1e-5, 1e-5);
    md = 0;
    for (int i = 0; i < 4096; i++) md = fmax(md, fabs((double)z1[i] - z2[i]));
    RT_EXPECT(md < 1e-6, "xent grad acc32 vs acc64 abs diff %g", md);
    /* ignored label */
    float l3 = rrc_kc_xent_row(z1, RRC_F32, 4096, -1, 1.0f, 1, 1);
    int zero = 1;
    for (int i = 0; i < 4096; i++) zero &= z1[i] == 0.0f;
    RT_EXPECT(l3 == 0.0f && zero, "ignored row");
    /* xent with huge logits does not overflow */
    float big[3] = {1000.0f, 999.0f, -1000.0f};
    float lb = rrc_kc_xent_row(big, RRC_F32, 3, 1, 1.0f, 0, 0);
    RT_CLOSE(lb, 1.0 + log(1.0 + exp(-1.0)), 1e-5, 1e-6);
}

RT_TEST(kc_rope_pairs) {
    /* rotation by theta then by -theta is the identity; |y| = |x| */
    float c = cosf(0.7f), s = sinf(0.7f), y1, y2, x1, x2;
    rrc_kc_rope_pair_fwd(0.3f, -1.2f, c, s, &y1, &y2);
    RT_CLOSE(sqrt((double)y1 * y1 + (double)y2 * y2), sqrt(0.09 + 1.44), 0, 1e-6);
    rrc_kc_rope_pair_bwd(y1, y2, c, s, &x1, &x2);
    RT_CLOSE(x1, 0.3, 1e-6, 0);
    RT_CLOSE(x2, -1.2, 1e-6, 0);
    /* HF rotate_half form: q*cos + rotate_half(q)*sin with rotate_half(q) = cat(-q2, q1) */
    RT_CLOSE(y1, 0.3 * c - (-1.2) * s, 1e-7, 0);
    RT_CLOSE(y2, -1.2 * c + 0.3 * s, 1e-7, 0);
}

const rt_case rt_suite_kernels[] = {
    RT_CASE(kc_load_store), RT_CASE(kc_lse_merge), RT_CASE(kc_silu_edges),  RT_CASE(kc_lerp_branches),
    RT_CASE(kc_softmax_rows), RT_CASE(kc_acc32_vs_acc64), RT_CASE(kc_rope_pairs),
};
const size_t rt_suite_kernels_n = sizeof rt_suite_kernels / sizeof rt_suite_kernels[0];
