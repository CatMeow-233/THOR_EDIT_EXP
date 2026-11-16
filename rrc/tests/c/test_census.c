/* Dry-run backend, shape-only model and GEMM census (DESIGN.md §13).
 *
 * The census is checked against reality, not only against itself: on tiny models the dry run of a
 * training step issues exactly the GEMM sequence of a real CPU run (shapes, dtypes, transposes,
 * strides, scalars), its op-table peak equals the bytes the CPU backend really had allocated, and the
 * shape-only model has the loaded model's layout. At the 8B preset shapes every GEMM against a
 * frozen base weight must be on a native bf16 x bf16 plan path, with exact call counts. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "../../src/model_internal.h"
#include "ops_util.h"
#include "rrc/rrc.h"
#include "rrc/census.h"
#include "rrc/dryrun.h"
#include "rrc/json.h"
#include "rrc/train.h"

static const char *k_cfg_qwen3 =
    "{\"model_type\": \"qwen3\", \"vocab_size\": 32, \"hidden_size\": 16, \"intermediate_size\": 24, "
    "\"num_hidden_layers\": 3, \"num_attention_heads\": 4, \"num_key_value_heads\": 2, \"head_dim\": 8, "
    "\"max_position_embeddings\": 64, \"rms_norm_eps\": 1e-5, \"attention_bias\": true, "
    "\"tie_word_embeddings\": true, \"rope_theta\": 10000.0}";
static const char *k_cfg_llama =
    "{\"model_type\": \"llama\", \"vocab_size\": 40, \"hidden_size\": 16, \"intermediate_size\": 20, "
    "\"num_hidden_layers\": 3, \"num_attention_heads\": 4, \"num_key_value_heads\": 1, "
    "\"max_position_embeddings\": 64, \"rms_norm_eps\": 1e-5, \"tie_word_embeddings\": false, "
    "\"rope_theta\": 10000.0}";

static rrc_status write_ckpt(const char *name, const char *cfg_json, uint64_t seed, const char **dir_out) {
    const char *dir = rt_path(name);
    *dir_out = dir;
    mkdir(dir, 0755);
    char file[1400];
    snprintf(file, sizeof file, "%s/config.json", dir);
    RRC_CHECK(rrc_write_file_atomic(file, cfg_json, strlen(cfg_json)));
    rrc_model_config c;
    RRC_CHECK(rrc_config_load(dir, 0, &c));
    rrc_tensor_spec *specs = NULL;
    size_t n = 0;
    RRC_CHECK(rrc_config_expected_tensors(&c, &specs, &n));
    rrc_st_writer *w = NULL;
    rrc_status st = rrc_st_writer_create(&w);
    rt_rng rng = {seed};
    for (size_t i = 0; st == RRC_OK && i < n; i++) {
        int64_t numel = specs[i].shape[0] * (specs[i].ndim == 2 ? specs[i].shape[1] : 1);
        float *x = (float *)malloc((size_t)numel * sizeof(float));
        if (!x) {
            st = RRC_ERR_OOM;
            break;
        }
        for (int64_t j = 0; j < numel; j++) x[j] = 0.3f * rt_uniform(&rng);
        st = rrc_st_writer_add(w, specs[i].name, RRC_F32, specs[i].ndim, specs[i].shape, x);
        free(x);
    }
    snprintf(file, sizeof file, "%s/model.safetensors", dir);
    if (st == RRC_OK) st = rrc_st_writer_write(w, file);
    rrc_st_writer_free(w);
    free(specs);
    return st;
}

/* ---- dry-run backend basics ------------------------------------------------------------------- */

RT_TEST(census_dryrun_memory_and_transfers) {
    rrc_dryrun_opts o;
    rrc_dryrun_opts_default(&o);
    o.limit_bytes = 10000;
    const rrc_backend *be = NULL;
    RT_OK(rrc_dryrun_backend_create(&o, &be));
    RT_EXPECT(rrc_backend_is_dryrun(be) && !be->host_accessible, "dry-run identity");
    void *p = NULL, *q = NULL, *z = (void *)1;
    RT_OK(be->alloc(be, 1000, &p));
    RT_OK(be->alloc(be, 3000, &q));
    RT_OK(be->alloc(be, 0, &z));
    RT_EXPECT(z == NULL, "alloc of 0 bytes returns NULL");
    const uintptr_t a = (uintptr_t)p, b = (uintptr_t)q;
    RT_EXPECT(a >= 0xffff800000000000ull && b >= 0xffff800000000000ull, "handles in the upper half: %p %p", p, q);
    RT_EXPECT(a % 256 == 0 && b % 256 == 0 && b >= a + 1000, "aligned, disjoint: %p %p", p, q);
    rrc_dryrun_stats s;
    rrc_dryrun_get_stats(be, &s);
    RT_EXPECT(s.live_bytes == 4000 && s.peak_bytes == 4000 && s.live_allocs == 2, "live %llu peak %llu",
              (unsigned long long)s.live_bytes, (unsigned long long)s.peak_bytes);
    be->free(be, p);
    rrc_dryrun_get_stats(be, &s);
    RT_EXPECT(s.live_bytes == 3000 && s.peak_bytes == 4000 && s.n_free == 1, "after free: live %llu peak %llu",
              (unsigned long long)s.live_bytes, (unsigned long long)s.peak_bytes);
    void *big = NULL;
    RT_STATUS(be->alloc(be, 8000, &big), RRC_ERR_OOM); /* 3000 + 8000 > limit 10000 */
    /* transfers: d2h zero-fills the host side, the others only count */
    unsigned char host[64];
    memset(host, 0xAB, sizeof host);
    RT_OK(be->memcpy_d2h(be, host, q, sizeof host));
    int zero = 1;
    for (size_t i = 0; i < sizeof host; i++) zero &= host[i] == 0;
    RT_EXPECT(zero, "memcpy_d2h zero-fills the host destination");
    RT_OK(be->memcpy_h2d(be, q, host, 32));
    RT_OK(be->sync(be));
    rrc_dryrun_get_stats(be, &s);
    RT_EXPECT(s.calls[RRC_DR_D2H] == 1 && s.d2h_bytes == 64 && s.calls[RRC_DR_H2D] == 1 && s.h2d_bytes == 32 &&
                  s.calls[RRC_DR_SYNC] == 1,
              "transfer counters");
    rrc_dryrun_reset_counters(be);
    rrc_dryrun_get_stats(be, &s);
    RT_EXPECT(s.calls[RRC_DR_SYNC] == 0 && s.live_bytes == 3000 && s.peak_bytes == 3000 && s.live_allocs == 1,
              "reset keeps live bytes");
    rrc_dryrun_backend_destroy(be);
}

RT_TEST(census_dryrun_gemm_plan_paths) {
    const rrc_backend *be = NULL;
    RT_OK(rrc_dryrun_backend_create(NULL, &be));
    rrc_buf a16, b16, a32, b32, c32, c16;
    const int64_t m = 64, n = 48, k = 80;
    RT_OK(rrc_buf_alloc(be, RRC_BF16, m * k, &a16));
    RT_OK(rrc_buf_alloc(be, RRC_BF16, k * n, &b16));
    RT_OK(rrc_buf_alloc(be, RRC_F32, m * k, &a32));
    RT_OK(rrc_buf_alloc(be, RRC_F32, k * n, &b32));
    RT_OK(rrc_buf_alloc(be, RRC_F32, m * n, &c32));
    RT_OK(rrc_buf_alloc(be, RRC_BF16, m * n, &c16));
    rrc_dryrun_reset_counters(be);
    /* bf16 x bf16 -> f32: native, nothing converted */
    RT_OK(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, m, n, k, 1.0f, a16, k, b16, n, 0.0f, c32, n));
    rrc_dryrun_stats s;
    rrc_dryrun_get_stats(be, &s);
    RT_EXPECT(s.gemm_flops == 2.0 * m * n * k && s.gemm_flops_native_bf16 == s.gemm_flops && s.gemm_convert_elems == 0,
              "native bf16: flops %.0f native_bf16 %.0f convert %llu", s.gemm_flops, s.gemm_flops_native_bf16,
              (unsigned long long)s.gemm_convert_elems);
    /* f32 x bf16 -> f32: B promoted once (k x n elements; one chunk at the default cap) */
    RT_OK(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, m, n, k, 1.0f, a32, k, b16, n, 0.0f, c32, n));
    rrc_dryrun_get_stats(be, &s);
    RT_EXPECT(s.gemm_convert_elems == (uint64_t)(k * n) && s.gemm_convert_bytes == (uint64_t)(k * n) * 6u &&
                  s.gemm_flops_native == 2.0 * m * n * k,
              "promote B: convert %llu bytes %llu", (unsigned long long)s.gemm_convert_elems,
              (unsigned long long)s.gemm_convert_bytes);
    RT_EXPECT(s.gemm_scratch_max_bytes >= (uint64_t)(k * n) * 4u, "scratch %llu", (unsigned long long)s.gemm_scratch_max_bytes);
    /* f32 x f32 -> bf16 with beta = 1: C read into the f32 scratch and written back (2 m n elements) */
    rrc_dryrun_reset_counters(be);
    RT_OK(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, m, n, k, 1.0f, a32, k, b32, n, 1.0f, c16, n));
    rrc_dryrun_get_stats(be, &s);
    RT_EXPECT(s.gemm_convert_elems == (uint64_t)(2 * m * n) && s.gemm_flops_native == 0,
              "C scratch: convert %llu", (unsigned long long)s.gemm_convert_elems);
    /* a tiny scratch cap chunks the promotion: more elements are converted (B once per row chunk) */
    rrc_dryrun_backend_destroy(be);
    rrc_dryrun_opts o;
    rrc_dryrun_opts_default(&o);
    o.gemm_scratch_cap_elems = 1000;
    RT_OK(rrc_dryrun_backend_create(&o, &be));
    RT_OK(rrc_buf_alloc(be, RRC_BF16, m * k, &a16));
    RT_OK(rrc_buf_alloc(be, RRC_F32, k * n, &b32));
    RT_OK(rrc_buf_alloc(be, RRC_F32, m * n, &c32));
    rrc_dryrun_reset_counters(be);
    RT_OK(rrc_gemm2d(be, RRC_OP_N, RRC_OP_N, m, n, k, 1.0f, a16, k, b32, n, 0.0f, c32, n));
    rrc_dryrun_get_stats(be, &s);
    RT_EXPECT(s.gemm_scratch_max_bytes <= 4000 && s.gemm_convert_elems >= (uint64_t)(m * k),
              "capped: scratch %llu convert %llu", (unsigned long long)s.gemm_scratch_max_bytes,
              (unsigned long long)s.gemm_convert_elems);
    rrc_dryrun_backend_destroy(be);
}

/* ---- a counting CPU backend: every GEMM recorded, allocation sizes tracked ------------------- */

typedef struct gemm_rec {
    rrc_trans ta, tb;
    int64_t m, n, k, lda, ldb, ldc, sa, sb, sc, batch;
    float alpha, beta;
    rrc_dtype a, b, c;
} gemm_rec;

typedef struct rec_log {
    gemm_rec *v;
    size_t n, cap;
    int overflow;
} rec_log;

static void log_push(rec_log *l, const rrc_gemm_args *g) {
    if (l->n == l->cap) {
        size_t nc = l->cap ? 2 * l->cap : 256;
        gemm_rec *p = (gemm_rec *)realloc(l->v, nc * sizeof *p);
        if (!p) {
            l->overflow = 1;
            return;
        }
        l->v = p;
        l->cap = nc;
    }
    gemm_rec *r = &l->v[l->n++];
    memset(r, 0, sizeof *r); /* padding too: records are compared with memcmp */
    r->ta = g->trans_a;
    r->tb = g->trans_b;
    r->m = g->m;
    r->n = g->n;
    r->k = g->k;
    r->lda = g->lda;
    r->ldb = g->ldb;
    r->ldc = g->ldc;
    r->sa = g->stride_a;
    r->sb = g->stride_b;
    r->sc = g->stride_c;
    r->batch = g->batch;
    r->alpha = g->alpha;
    r->beta = g->beta;
    r->a = g->a.dtype;
    r->b = g->b.dtype;
    r->c = g->c.dtype;
}

static rrc_backend g_cnt_be;
static rec_log g_cpu_log;
static double g_cpu_flops;
static struct {
    void *p;
    size_t n;
} g_live[256];
static size_t g_n_live;
static uint64_t g_cpu_live, g_cpu_peak;

static rrc_status cnt_alloc(const rrc_backend *be, size_t bytes, void **out) {
    (void)be;
    RRC_CHECK(rrc_backend_cpu_f32acc()->alloc(rrc_backend_cpu_f32acc(), bytes, out));
    if (*out) {
        RRC_REQUIRE(g_n_live < 256, RRC_ERR_INTERNAL, "test allocation table full");
        g_live[g_n_live].p = *out;
        g_live[g_n_live].n = bytes;
        g_n_live++;
        g_cpu_live += bytes;
        if (g_cpu_live > g_cpu_peak) g_cpu_peak = g_cpu_live;
    }
    return RRC_OK;
}
static void cnt_free(const rrc_backend *be, void *p) {
    (void)be;
    for (size_t i = 0; p && i < g_n_live; i++)
        if (g_live[i].p == p) {
            g_cpu_live -= g_live[i].n;
            g_live[i] = g_live[--g_n_live];
            break;
        }
    rrc_backend_cpu_f32acc()->free(rrc_backend_cpu_f32acc(), p);
}
static rrc_status cnt_gemm(const rrc_backend *be, const rrc_gemm_args *g) {
    (void)be;
    log_push(&g_cpu_log, g);
    g_cpu_flops += 2.0 * (double)g->m * (double)g->n * (double)g->k * (double)g->batch;
    return rrc_backend_cpu_f32acc()->gemm(rrc_backend_cpu_f32acc(), g);
}
static const rrc_backend *counting_cpu(void) {
    g_cnt_be = *rrc_backend_cpu_f32acc();
    g_cnt_be.name = "cpu-counting";
    g_cnt_be.alloc = cnt_alloc;
    g_cnt_be.free = cnt_free;
    g_cnt_be.gemm = cnt_gemm;
    g_cpu_log.n = 0;
    g_cpu_flops = 0;
    g_n_live = 0;
    g_cpu_live = g_cpu_peak = 0;
    return &g_cnt_be;
}

static rec_log g_dry_log;
static void dry_hook(void *ctx, const rrc_dryrun_call *c) {
    (void)ctx;
    if (c->op == RRC_DR_GEMM) log_push(&g_dry_log, &c->gemm->args);
}

static int64_t view_off(const rrc_model *m, rrc_buf v) {
    if (rrc_buf_is_none(v)) return -1;
    return (int64_t)(((const char *)v.ptr - (const char *)m->storage.ptr) / (ptrdiff_t)rrc_dtype_size(v.dtype));
}

/* One training step (as rrc_train_run_on): zero grads, forward + backward, clip + AdamW, sync. */
static rrc_status one_step(rrc_model *m, rrc_lora *l, rrc_optimizer *o, const rrc_batch *b) {
    int64_t nl = 0;
    RRC_CHECK(rrc_batch_count_labels(b, &nl));
    RRC_CHECK(rrc_lora_zero_grads(l));
    rrc_fb_opts fo = {RRC_LOSS_WINDOW_TOKENS, nl, 1, 1};
    rrc_fb_result res;
    RRC_CHECK(rrc_model_forward_backward(m, l, b, &fo, &res));
    rrc_train_config d;
    rrc_train_config_default(&d);
    rrc_optim_stats os;
    RRC_CHECK(rrc_optimizer_step(o, rrc_lora_params(l), rrc_lora_grads(l), 1e-3, &d.optim, &os));
    return rrc_sync(rrc_model_backend(m));
}

static void compare_with_cpu(const char *name, const char *cfg_json, rrc_dtype dt, rrc_recompute rc, int64_t B,
                             int64_t T, int64_t attn_chunk, int64_t ce_chunk) {
    const char *dir;
    RT_OK(write_ckpt(name, cfg_json, 77, &dir));
    rrc_model_opts mo;
    rrc_model_opts_default(&mo);
    mo.dtype = dt;
    mo.recompute = rc;
    mo.attn_chunk = attn_chunk;
    mo.ce_chunk = ce_chunk;
    rrc_lora_config lc;
    rrc_lora_config_default(&lc);
    lc.r = 4;
    int32_t tok[256];
    for (int64_t i = 0; i < B * T; i++) tok[i] = (int32_t)(i % 7);
    rrc_batch b = {B, T, tok, tok, NULL, NULL};

    /* real CPU run */
    const rrc_backend *cpu = counting_cpu();
    rrc_model *m = NULL;
    RT_OK(rrc_model_load(cpu, dir, &mo, &m));
    rrc_lora *l = NULL;
    RT_OK(rrc_lora_create(cpu, rrc_model_get_config(m), &lc, &l));
    rrc_optimizer *o = NULL;
    RT_OK(rrc_optimizer_create(cpu, rrc_lora_num_params(l), &o));
    RT_OK(one_step(m, l, o, &b));
    const uint64_t cpu_peak = g_cpu_peak;
    const double cpu_flops = g_cpu_flops;
    const size_t cpu_ws = rrc_model_workspace_bytes(m);

    /* dry run of the same step on a shape-only model of the same config */
    const rrc_backend *dry = NULL;
    RT_OK(rrc_dryrun_backend_create(NULL, &dry));
    rrc_model *dm = NULL;
    RT_OK(rrc_model_create_shape_only(dry, rrc_model_get_config(m), &mo, &dm));
    /* the shape-only model has the loaded model's layout */
    RT_EXPECT(dm->storage.numel == m->storage.numel && dm->storage.dtype == m->storage.dtype &&
                  dm->rope_storage.numel == m->rope_storage.numel && dm->n_pos == m->n_pos,
              "%s: storage %lld vs %lld, rope %lld vs %lld", name, (long long)dm->storage.numel,
              (long long)m->storage.numel, (long long)dm->rope_storage.numel, (long long)m->rope_storage.numel);
    RT_EXPECT(view_off(dm, dm->embed) == view_off(m, m->embed) && view_off(dm, dm->lm_head) == view_off(m, m->lm_head) &&
                  view_off(dm, dm->norm) == view_off(m, m->norm),
              "%s: embed / lm_head / norm offsets", name);
    for (int64_t li = 0; li < m->cfg.num_layers; li++) {
        const rrc_layer_weights *x = &m->layers[li], *y = &dm->layers[li];
        const rrc_buf xs[] = {x->ln1, x->ln2, x->wq, x->wk, x->wv, x->wo, x->bq, x->bk, x->bv, x->bo, x->qn, x->kn, x->wgu, x->wd};
        const rrc_buf ys[] = {y->ln1, y->ln2, y->wq, y->wk, y->wv, y->wo, y->bq, y->bk, y->bv, y->bo, y->qn, y->kn, y->wgu, y->wd};
        for (size_t j = 0; j < sizeof xs / sizeof xs[0]; j++)
            RT_EXPECT(view_off(m, xs[j]) == view_off(dm, ys[j]) && xs[j].numel == ys[j].numel,
                      "%s: layer %lld view %zu differs", name, (long long)li, j);
    }
    rrc_lora *dl = NULL;
    RT_OK(rrc_lora_create(dry, rrc_model_get_config(dm), &lc, &dl));
    rrc_optimizer *dopt = NULL;
    RT_OK(rrc_optimizer_create(dry, rrc_lora_num_params(dl), &dopt));
    g_dry_log.n = 0;
    rrc_dryrun_set_hook(dry, dry_hook, NULL);
    RT_OK(one_step(dm, dl, dopt, &b));
    rrc_dryrun_set_hook(dry, NULL, NULL);
    rrc_dryrun_stats s;
    rrc_dryrun_get_stats(dry, &s);
    RT_EXPECT(!g_cpu_log.overflow && !g_dry_log.overflow, "log allocation");
    RT_EXPECT(g_dry_log.n == g_cpu_log.n && g_dry_log.n > 0, "%s: %zu dry-run GEMMs vs %zu on the CPU", name,
              g_dry_log.n, g_cpu_log.n);
    size_t first_diff = g_dry_log.n;
    for (size_t i = 0; i < g_dry_log.n && i < g_cpu_log.n; i++)
        if (memcmp(&g_dry_log.v[i], &g_cpu_log.v[i], sizeof(gemm_rec)) != 0) {
            first_diff = i;
            break;
        }
    RT_EXPECT(first_diff == g_dry_log.n, "%s: GEMM %zu differs between the dry run and the CPU run", name, first_diff);
    RT_EXPECT(s.gemm_flops == cpu_flops, "%s: flops %.0f vs %.0f", name, s.gemm_flops, cpu_flops);
    RT_EXPECT(s.peak_bytes == cpu_peak, "%s: dry-run peak %llu, CPU backend peak %llu", name,
              (unsigned long long)s.peak_bytes, (unsigned long long)cpu_peak);
    RT_EXPECT(rrc_model_workspace_bytes(dm) == cpu_ws, "%s: workspace %zu vs %zu", name, rrc_model_workspace_bytes(dm),
              cpu_ws);

    /* rrc_census_run (CPU target) reports the same step totals and memory */
    rrc_census_opts co;
    rrc_census_opts_default(&co);
    co.model = mo;
    co.lora = lc;
    co.seq_len = T;
    co.micro_batch = B;
    co.target = RRC_CENSUS_TARGET_CPU;
    rrc_census_result *r = (rrc_census_result *)calloc(1, sizeof *r);
    RT_ASSERT(r, "alloc");
    rrc_status st = rrc_census_run(rrc_model_get_config(m), &co, r);
    RT_EXPECT(st == RRC_OK, "census: %s", rrc_last_error());
    if (st == RRC_OK) {
        double fl = 0;
        int64_t calls = 0;
        for (int ro = 0; ro < RRC_CR_N; ro++)
            for (int k = 0; k < RRC_CENSUS_N_COMBOS; k++) {
                fl += r->cell[ro][k].flops;
                calls += r->cell[ro][k].calls;
            }
        RT_EXPECT(fl == cpu_flops && calls == (int64_t)g_cpu_log.n, "%s: census flops %.0f calls %lld vs CPU %.0f %zu",
                  name, fl, (long long)calls, cpu_flops, g_cpu_log.n);
        RT_EXPECT(r->op_peak_bytes == cpu_peak && r->peak_total_bytes == cpu_peak, "%s: census peak %llu vs %llu", name,
                  (unsigned long long)r->op_peak_bytes, (unsigned long long)cpu_peak);
        RT_EXPECT(r->weights_bytes + r->rope_bytes + r->lora_bytes + r->adam_bytes + r->workspace_bytes == cpu_peak,
                  "%s: memory categories add up to the CPU peak", name);
    }
    free(r);
    rrc_optimizer_free(dopt);
    rrc_lora_free(dl);
    rrc_model_free(dm);
    rrc_dryrun_backend_destroy(dry);
    rrc_optimizer_free(o);
    rrc_lora_free(l);
    rrc_model_free(m);
}

RT_TEST(census_dryrun_matches_cpu_run) {
    compare_with_cpu("dry_qwen3_bf16", k_cfg_qwen3, RRC_BF16, RRC_RECOMPUTE_FULL, 2, 9, 4, 5);
    compare_with_cpu("dry_qwen3_f32_none", k_cfg_qwen3, RRC_F32, RRC_RECOMPUTE_NONE, 1, 13, 5, 64);
    compare_with_cpu("dry_llama_bf16", k_cfg_llama, RRC_BF16, RRC_RECOMPUTE_FULL, 3, 7, 7, 3);
    free(g_cpu_log.v);
    free(g_dry_log.v);
    memset(&g_cpu_log, 0, sizeof g_cpu_log);
    memset(&g_dry_log, 0, sizeof g_dry_log);
}

/* ---- 8B shapes: every base-weight GEMM on a native bf16 path ----------------------------------- */

RT_TEST(census_8b_presets_base_weight_gemms) {
    RT_ASSERT(rrc_census_num_presets() == 3, "three presets");
    static const double lo[] = {8.0e9, 7.4e9, 7.9e9}, hi[] = {8.4e9, 7.8e9, 8.2e9};
    for (size_t i = 0; i < rrc_census_num_presets(); i++) {
        const rrc_census_preset *p = rrc_census_preset_at(i);
        rrc_model_config cfg;
        RT_OK(rrc_census_preset_config(p, &cfg));
        rrc_census_opts o;
        rrc_census_opts_default(&o);
        o.seq_len = 4096;
        o.lora.r = 16;
        o.lora.alpha = 32;
        rrc_census_result *r = (rrc_census_result *)calloc(1, sizeof *r);
        RT_ASSERT(r, "alloc");
        rrc_status st = rrc_census_run(&cfg, &o, r);
        RT_EXPECT(st == RRC_OK, "%s: %s", p->name, rrc_last_error());
        if (st != RRC_OK) {
            free(r);
            continue;
        }
        /* nominal sizes catch a mistyped preset value */
        RT_EXPECT((double)r->base_params > lo[i] && (double)r->base_params < hi[i], "%s: %lld base parameters", p->name,
                  (long long)r->base_params);
        const int64_t L = cfg.num_layers, n_ce = (4095 + 1023) / 1024;
        /* forward: q, k, v, o, gate_up, down per layer; recompute (full): every layer but the top one again
         * without down; backward: the six input gradients per layer; LM head: one GEMM per CE chunk each way */
        const int64_t want[RRC_CR_N] = {6 * L + 5 * (L - 1), 6 * L, n_ce, n_ce, -1, -1};
        for (int ro = 0; ro < RRC_CR_N; ro++) {
            int64_t calls = 0, non_native = 0, non_bf16 = 0;
            for (int k = 0; k < RRC_CENSUS_N_COMBOS; k++) {
                calls += r->cell[ro][k].calls;
                if (!rrc_census_combo_native(k)) non_native += r->cell[ro][k].calls;
                if (!rrc_census_combo_native_bf16(k)) non_bf16 += r->cell[ro][k].calls;
            }
            if (rrc_census_role_is_base(ro)) {
                RT_EXPECT(calls == want[ro], "%s %s: %lld GEMMs, expected %lld", p->name, rrc_census_role_name(ro),
                          (long long)calls, (long long)want[ro]);
                (void)non_native;
                (void)non_bf16;
            } else if (ro == RRC_CR_LORA) {
                /* LoRA stays fp32 like PEFT: never a bf16 x bf16 GEMM */
                RT_EXPECT(calls > 0 && r->cell[ro][6].calls == 0 && r->cell[ro][7].calls == 0,
                          "%s: LoRA GEMMs with bf16 A and B", p->name);
            }
        }
        RT_EXPECT(r->peak_total_bytes > r->weights_bytes && r->workspace_bytes > 0 && r->gemm_scratch_bytes > 0 &&
                      r->cublaslt_workspace_bytes == (32u << 20),
                  "%s: memory categories", p->name);
        free(r);
    }
}

/* ---- JSON report fields ------------------------------------------------------------------------ */

static double jnum(const rrc_json *o, const char *k) {
    const rrc_json *v = o ? rrc_json_get(o, k) : NULL;
    if (!v) return -1e300;
    return v->type == RRC_JSON_INT ? (double)v->u.i : v->type == RRC_JSON_DOUBLE ? v->u.d : -1e300;
}

RT_TEST(census_json_fields) {
    rrc_model_config cfg;
    const rrc_census_preset *p = rrc_census_preset_find("qwen2.5-7b");
    RT_ASSERT(p && !rrc_census_preset_find("nonexistent"), "preset lookup");
    RT_OK(rrc_census_preset_config(p, &cfg));
    rrc_census_opts o;
    rrc_census_opts_default(&o);
    o.seq_len = 512;
    o.micro_batch = 2;
    o.grad_accum = 2;
    o.model.attn_chunk = 128;
    o.model.ce_chunk = 300;
    rrc_census_result *r = (rrc_census_result *)calloc(1, sizeof *r);
    RT_ASSERT(r, "alloc");
    RT_OK(rrc_census_run(&cfg, &o, r));
    char *js = NULL;
    rrc_status st = rrc_census_to_json(r, &o, "preset:qwen2.5-7b", rrc_census_preset_note(), 0, &js);
    free(r);
    RT_ASSERT(st == RRC_OK, "to_json: %s", rrc_last_error());
    rrc_json_doc *doc = NULL;
    st = rrc_json_parse(js, strlen(js), &doc);
    free(js);
    RT_ASSERT(st == RRC_OK, "census JSON does not parse: %s", rrc_last_error());
    const rrc_json *root = rrc_json_root(doc);
    const rrc_json *status = rrc_json_get(root, "status"), *model = rrc_json_get(root, "model"),
                   *gemm = rrc_json_get(root, "gemm"), *mem = rrc_json_get(root, "memory"),
                   *tr = rrc_json_get(root, "transfers"), *run = rrc_json_get(root, "run");
    RT_EXPECT(status && status->type == RRC_JSON_STRING && !strcmp(status->u.s.ptr, "computed"), "status computed");
    const rrc_json *note = model ? rrc_json_get(model, "preset_note") : NULL;
    RT_EXPECT(note && note->type == RRC_JSON_STRING && strstr(note->u.s.ptr, "NOT_TESTED"), "preset note");
    RT_EXPECT(jnum(run, "tokens_per_step") == 2 * 2 * 512 && jnum(run, "labelled_tokens_per_step") == 2 * 2 * 511,
              "tokens per step");
    const double tot = jnum(gemm, "flops_total"), f16 = jnum(gemm, "native_bf16_fraction");
    RT_EXPECT(tot > 0 && f16 > 0 && f16 <= 1 && jnum(gemm, "native_fraction") >= f16, "fractions");
    const rrc_json *bc = gemm ? rrc_json_get(gemm, "by_combo") : NULL, *br = gemm ? rrc_json_get(gemm, "by_role") : NULL;
    RT_ASSERT(bc && bc->type == RRC_JSON_ARRAY && br && br->type == RRC_JSON_ARRAY, "by_combo / by_role arrays");
    double sum_c = 0, sum_r = 0, calls_r = 0;
    for (size_t i = 0; i < rrc_json_arr_count(bc); i++) sum_c += jnum(rrc_json_arr_get(bc, i), "flops");
    for (size_t i = 0; i < rrc_json_arr_count(br); i++) {
        sum_r += jnum(rrc_json_arr_get(br, i), "flops");
        calls_r += jnum(rrc_json_arr_get(br, i), "calls");
    }
    RT_EXPECT(sum_c == tot && sum_r == tot && calls_r == jnum(gemm, "calls"), "flop tables add up (%.0f %.0f %.0f)",
              sum_c, sum_r, tot);
    RT_EXPECT(rrc_json_arr_count(br) == RRC_CR_N, "every role listed");
    const rrc_json *bw = rrc_json_get(gemm, "base_weight");
    RT_EXPECT(jnum(bw, "calls") > 0 && jnum(bw, "non_native_calls") >= 0, "base_weight block");
    /* one step with grad_accum 2: the trainer's syncs (optimizer d2h + loop sync) and loss downloads */
    RT_EXPECT(jnum(tr, "sync") == 2 * 1 + 1 && jnum(tr, "memcpy_d2h") == 2 + 1 && jnum(tr, "memcpy_h2d") == 2 * 6 &&
                  jnum(tr, "alloc") == 0 && jnum(tr, "free") == 0,
              "transfers: sync %.0f d2h %.0f h2d %.0f", jnum(tr, "sync"), jnum(tr, "memcpy_d2h"), jnum(tr, "memcpy_h2d"));
    const double parts = jnum(mem, "weights_bytes") + jnum(mem, "rope_tables_bytes") +
                         jnum(mem, "lora_params_grads_bytes") + jnum(mem, "adam_bytes") + jnum(mem, "workspace_bytes");
    RT_EXPECT(parts == jnum(mem, "op_table_peak_bytes"), "op-table peak = sum of categories");
    RT_EXPECT(jnum(mem, "peak_total_bytes") == jnum(mem, "op_table_peak_bytes") + jnum(mem, "gemm_scratch_bytes") +
                                                   jnum(mem, "sum_squares_partials_bytes") +
                                                   jnum(mem, "cublaslt_workspace_bytes"),
              "peak total");
    rrc_json_doc_free(doc);
}

RT_TEST(census_refusals) {
    rrc_model_config cfg;
    RT_OK(rrc_census_preset_config(rrc_census_preset_find("qwen3-8b"), &cfg));
    rrc_census_opts o;
    rrc_census_opts_default(&o);
    rrc_census_result *r = (rrc_census_result *)calloc(1, sizeof *r);
    RT_ASSERT(r, "alloc");
    RT_STATUS(rrc_census_run(&cfg, &o, r), RRC_ERR_INVALID_ARG); /* seq_len 0 */
    o.seq_len = 40961;                                             /* > max_position_embeddings */
    RT_STATUS(rrc_census_run(&cfg, &o, r), RRC_ERR_INVALID_ARG);
    o.seq_len = 128;
    o.lora.layers[0] = 99;
    o.lora.n_layers = 1;
    RT_STATUS(rrc_census_run(&cfg, &o, r), RRC_ERR_INVALID_ARG); /* LoRA layer out of range (rrc_lora_create) */
    free(r);
    rrc_model_config bad = cfg;
    bad.num_kv_heads = 5;
    rrc_model *m = (rrc_model *)(void *)&bad; /* must be reset on failure */
    RT_STATUS(rrc_model_create_shape_only(rrc_backend_cpu(), &bad, NULL, &m), RRC_ERR_INVALID_ARG);
    RT_EXPECT(m == NULL, "model pointer set on failure");
}

const rt_case rt_suite_census[] = {
    RT_CASE(census_dryrun_memory_and_transfers), RT_CASE(census_dryrun_gemm_plan_paths),
    RT_CASE(census_dryrun_matches_cpu_run),      RT_CASE(census_8b_presets_base_weight_gemms),
    RT_CASE(census_json_fields),                 RT_CASE(census_refusals),
};
const size_t rt_suite_census_n = sizeof rt_suite_census / sizeof rt_suite_census[0];
