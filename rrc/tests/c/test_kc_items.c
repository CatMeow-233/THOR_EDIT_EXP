/* Execution coverage of the CUDA kernel bodies on CPU (DESIGN.md §11.4): the host emulation
 * backend (rrc/cuda/kc_emu.c) runs every op through the per-item bodies of
 * kernels_common/rrc_kc_items.h with the kernels' phase structure, each phase over its items in
 * shuffled order, and the backend-agnostic selftest harness (rrc/cuda/selftest_ops.c, the code
 * of rrc_selftest) compares it with rrc_backend_cpu_f32acc:
 *   - bit-identical for every op whose arithmetic is the same (embedding, bias, rope, swiglu,
 *     add/axpy/scale/convert, adamw, every GEMM path including promoted and kernel ones);
 *   - REDUCE tolerance for the reductions (rmsnorm, softmax, cross entropy, sum_squares), whose
 *     fixed lane/tree order differs from the CPU's sequential order: fp32 reassociation error is
 *     bounded by n * 2^-24 relative to the sum of magnitudes (n = terms), the harness allows
 *     (1e-5 + n * 2^-24) of the output scale;
 *   - each case runs twice with different item orders and must be bit-identical (the reduction
 *     order depends only on the shape);
 *   - range errors are reported (by the op, or by the next sync in deferred mode) and the
 *     offending items write nothing and read nothing out of bounds.
 * The harness is also run with the double-accumulation CPU backend as the device, which checks
 * that its tolerances accept a correct backend with different (more exact) arithmetic. */
#include "kc_emu.h"
#include "ops_util.h"
#include "rrc_kc_items.h"
#include "selftest_ops.h"

static int emu_path(const rrc_backend *be, char *buf, size_t n) {
    rrc_gemm_exec_report rep;
    if (!rrc_emu_last_gemm(be, &rep)) return 0;
    snprintf(buf, n, "%s", rep.desc);
    return 1;
}

/* Run the harness, its log goes to a file that is printed on failure. */
static void harness(const rrc_backend *dev, const char *only, const char *log) {
    FILE *f = fopen(rt_path(log), "w+");
    RT_ASSERT(f, "cannot open %s", rt_path(log));
    rrc_st_opts o;
    memset(&o, 0, sizeof o);
    o.seed = 99;
    o.quick = 1;
    o.out = f;
    o.only = only;
    o.gemm_path = rrc_backend_is_emu(dev) ? emu_path : NULL;
    rrc_st_result r;
    int fails = rrc_st_run(dev, rrc_backend_cpu_f32acc(), &o, &r);
    RT_EXPECT(fails == 0, "%s: %d failed checks (log: %s)", dev->name, fails, rt_path(log));
    RT_EXPECT(r.checks > 0 && r.det_checks == r.checks, "harness ran (%d checks)", r.checks);
    if (!only) RT_EXPECT(r.checks > 400 && r.range_checks == 8, "full run: %d checks, %d range checks", r.checks, r.range_checks);
    if (fails) {
        rewind(f);
        char line[512];
        int shown = 0;
        while (fgets(line, sizeof line, f) && shown < 60)
            if (strstr(line, "FAIL") || strstr(line, "error") || strstr(line, "worst element")) {
                fputs(line, stderr);
                shown++;
            }
    }
    fclose(f);
}

RT_TEST(emu_vs_cpu_all_ops) {
    rrc_emu_options eo;
    rrc_emu_options_default(&eo);
    eo.seed = 1234;
    const rrc_backend *emu = NULL;
    RT_OK(rrc_emu_backend_create(&eo, &emu));
    harness(emu, NULL, "selftest_emu.log");
    rrc_emu_backend_destroy(emu);
}

RT_TEST(emu_deferred_errors_and_forced_paths) {
    rrc_emu_options eo;
    rrc_emu_options_default(&eo);
    eo.deferred_errors = 1; /* the CUDA backend's default: reported by the next sync */
    const rrc_backend *emu = NULL;
    RT_OK(rrc_emu_backend_create(&eo, &emu));
    harness(emu, "embedding", "selftest_emu_deferred_embed.log");
    harness(emu, "rope", "selftest_emu_deferred_rope.log");
    harness(emu, "cross_entropy", "selftest_emu_deferred_ce.log");
    rrc_emu_backend_destroy(emu);
    /* every GEMM through the kernel level, and through promote-all with a tiny scratch cap */
    rrc_emu_options_default(&eo);
    eo.gemm_min_level = RRC_GEMM_LEVEL_KERNEL;
    RT_OK(rrc_emu_backend_create(&eo, &emu));
    harness(emu, "gemm", "selftest_emu_kernel_gemm.log");
    rrc_emu_backend_destroy(emu);
    rrc_emu_options_default(&eo);
    eo.gemm_min_level = RRC_GEMM_LEVEL_PROMOTE_ALL;
    eo.scratch_cap_elems = 300;
    RT_OK(rrc_emu_backend_create(&eo, &emu));
    harness(emu, "gemm", "selftest_emu_promote_gemm.log");
    rrc_emu_backend_destroy(emu);
}

RT_TEST(cpu_double_passes_harness_tolerances) { harness(rrc_backend_cpu(), NULL, "selftest_cpu64.log"); }

RT_TEST(emu_range_items_write_nothing) {
    /* a bad token id must neither read the table out of bounds (ASan build) nor write its row */
    const rrc_backend *emu = NULL;
    RT_OK(rrc_emu_backend_create(NULL, &emu));
    const int64_t T = 6, D = 5, V = 4;
    float table[20], out_init[30];
    for (int i = 0; i < 20; i++) table[i] = (float)i;
    for (int i = 0; i < 30; i++) out_init[i] = -7.0f;
    int32_t ids[6] = {0, 3, 1000000, 2, -50, 1};
    rrc_buf tb = rt_upload(emu, RRC_F32, table, V * D), ob = rt_upload(emu, RRC_F32, out_init, T * D),
            ib = rt_upload_i32(emu, ids, T);
    RT_STATUS(rrc_embedding_fwd(emu, T, D, V, ib, tb, ob), RRC_ERR_RANGE);
    float *got = rt_download(ob, T * D);
    for (int64_t t = 0; t < T; t++)
        for (int64_t c = 0; c < D; c++) {
            float want = (ids[t] >= 0 && ids[t] < V) ? table[ids[t] * D + c] : -7.0f;
            RT_EXPECT(got[t * D + c] == want, "token %lld col %lld: %g want %g", (long long)t, (long long)c,
                      (double)got[t * D + c], (double)want);
        }
    free(got);
    rrc_buf_free(&tb);
    rrc_buf_free(&ob);
    rrc_buf_free(&ib);
    rrc_emu_backend_destroy(emu);
}

RT_TEST(tree_order_fixed) {
    /* the tree's result depends only on the partials, and differs (in general) from a sequential
     * sum: the device order is not the CPU order, which is why reductions have a tolerance */
    float part[RRC_KC_LANES], copy[RRC_KC_LANES];
    rt_rng rng = {5};
    for (int i = 0; i < RRC_KC_LANES; i++) part[i] = rt_uniform(&rng) * (float)(1 + i % 7) * 1e3f;
    memcpy(copy, part, sizeof part);
    double ref = 0.0;
    for (int i = 0; i < RRC_KC_LANES; i++) ref += part[i];
    float a = rrc_kc_tree_sum(part);
    /* same levels executed lane-descending within each level */
    for (int s = RRC_KC_LANES / 2; s > 0; s >>= 1)
        for (int l = s - 1; l >= 0; l--) rrc_kc_tree_sum_step(copy, l, s);
    RT_EXPECT(rrc_f32_bits(a) == rrc_f32_bits(copy[0]), "tree level order irrelevant");
    RT_CLOSE(a, ref, 1e-2, 1e-5);
    rrc_kc_lse l[RRC_KC_LANES];
    for (int i = 0; i < RRC_KC_LANES; i++) l[i] = rrc_kc_lse_push(rrc_kc_lse_init(), i % 5 == 0 ? -INFINITY : rt_uniform(&rng) * 30);
    rrc_kc_lse r = rrc_kc_tree_lse(l);
    RT_EXPECT(isfinite(r.m) && isfinite(r.s) && r.s >= 1.0f, "lse tree with -inf lanes: m %g s %g", (double)r.m,
              (double)r.s);
}

const rt_case rt_suite_kc_items[] = {
    RT_CASE(emu_vs_cpu_all_ops),
    RT_CASE(emu_deferred_errors_and_forced_paths),
    RT_CASE(cpu_double_passes_harness_tolerances),
    RT_CASE(emu_range_items_write_nothing),
    RT_CASE(tree_order_fixed),
};
const size_t rt_suite_kc_items_n = sizeof rt_suite_kc_items / sizeof rt_suite_kc_items[0];
