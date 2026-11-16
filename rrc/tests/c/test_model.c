/* Model line tests (no PyTorch needed): tiny checkpoints written with the C safetensors writer,
 * LoRA layout / init / I/O, batch validation, the LoRA gradient against central finite differences
 * of the loss, recompute and chunking invariance, workspace reuse, zero-labelled micro-batches.
 * The comparison with transformers + PEFT is rrc/tests/test_model_vs_peft.py. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "ops_util.h"
#include "rrc/rrc.h"

/* qwen3 with q/k norm, attention biases, tied embeddings, GQA (4 query heads, 2 kv heads) and
 * head_dim (8) != hidden / heads (4). */
static const char *k_cfg_qwen3 =
    "{\"model_type\": \"qwen3\", \"vocab_size\": 32, \"hidden_size\": 16, \"intermediate_size\": 24, "
    "\"num_hidden_layers\": 2, \"num_attention_heads\": 4, \"num_key_value_heads\": 2, \"head_dim\": 8, "
    "\"max_position_embeddings\": 64, \"rms_norm_eps\": 1e-5, \"attention_bias\": true, "
    "\"tie_word_embeddings\": true, \"rope_theta\": 10000.0}";
/* llama, untied, llama3 rope scaling, 3 layers */
static const char *k_cfg_llama =
    "{\"model_type\": \"llama\", \"vocab_size\": 40, \"hidden_size\": 16, \"intermediate_size\": 20, "
    "\"num_hidden_layers\": 3, \"num_attention_heads\": 4, \"num_key_value_heads\": 2, "
    "\"max_position_embeddings\": 64, \"rms_norm_eps\": 1e-5, \"tie_word_embeddings\": false, "
    "\"rope_scaling\": {\"rope_type\": \"llama3\", \"rope_theta\": 500000.0, \"factor\": 8.0, "
    "\"low_freq_factor\": 1.0, \"high_freq_factor\": 4.0, \"original_max_position_embeddings\": 16}}";
/* qwen2: q/k/v biases, untied */
static const char *k_cfg_qwen2 =
    "{\"model_type\": \"qwen2\", \"vocab_size\": 24, \"hidden_size\": 16, \"intermediate_size\": 16, "
    "\"num_hidden_layers\": 2, \"num_attention_heads\": 4, \"num_key_value_heads\": 2, "
    "\"max_position_embeddings\": 64, \"rms_norm_eps\": 1e-6, \"tie_word_embeddings\": false, "
    "\"rope_theta\": 1000000.0}";

/* Write config.json + model.safetensors with random weights into rt_path(name). */
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
        const char *nm = specs[i].name;
        const int is_norm = strstr(nm, "norm.weight") != NULL, is_bias = strstr(nm, ".bias") != NULL;
        for (int64_t j = 0; j < numel; j++) {
            float u = rt_uniform(&rng);
            x[j] = is_norm ? 1.0f + 0.2f * u : is_bias ? 0.1f * u : 0.35f * u;
        }
        st = rrc_st_writer_add(w, nm, RRC_F32, specs[i].ndim, specs[i].shape, x);
        free(x);
    }
    snprintf(file, sizeof file, "%s/model.safetensors", dir);
    if (st == RRC_OK) st = rrc_st_writer_write(w, file);
    rrc_st_writer_free(w);
    free(specs);
    return st;
}

/* Packed micro-batch: rows x T with documents of the given lengths per row (0-terminated),
 * random tokens, labels = tokens except the first `prompt` tokens after the first one. */
typedef struct tb {
    int64_t B, T;
    int32_t tok[256], lab[256], pos[256], doc[256];
} tb;

static void make_batch(tb *b, int64_t B, int64_t T, const int *lens, int64_t vocab, uint64_t seed, int prompt) {
    rt_rng rng = {seed};
    b->B = B;
    b->T = T;
    int64_t i = 0;
    for (int64_t r = 0; r < B; r++) {
        int64_t t = 0;
        int k = 0;
        while (t < T) {
            int len = *lens ? *lens++ : (int)(T - t);
            for (int j = 0; j < len && t < T; j++, t++, i++) {
                b->tok[i] = (int32_t)(rt_next(&rng) % (uint64_t)vocab);
                b->lab[i] = (j >= 1 && j <= prompt) ? RRC_IGNORE_INDEX : b->tok[i];
                b->pos[i] = j;
                b->doc[i] = 10 * k + (int)r;
            }
            k++;
        }
        lens++; /* skip the row terminator */
    }
}

static rrc_batch as_batch(const tb *b) {
    rrc_batch x = {b->B, b->T, b->tok, b->lab, b->pos, b->doc};
    return x;
}

static void randomize_lora(rrc_lora *l, uint64_t seed, float a_scale, float b_scale) {
    const int64_t n = rrc_lora_num_params(l);
    float *h = (float *)malloc((size_t)n * sizeof(float));
    rt_rng rng = {seed};
    for (size_t i = 0; i < rrc_lora_num_tensors(l); i++) {
        rrc_lora_tensor_info ti;
        if (rrc_lora_tensor(l, i, &ti) != RRC_OK) break;
        for (int64_t j = 0; j < ti.numel; j++) h[ti.offset + j] = rt_uniform(&rng) * (ti.is_b ? b_scale : a_scale);
    }
    rrc_buf_upload(rrc_lora_params(l), h, n);
    free(h);
}

static float *dl_all(rrc_buf b, int64_t n) { return rt_download(b, n); }

static rrc_fb_opts window_opts(int64_t window_tokens) {
    rrc_fb_opts o = {RRC_LOSS_WINDOW_TOKENS, window_tokens, 1, 1};
    return o;
}

/* ---- LoRA layout, names, init, refusals ----------------------------------------------------- */

RT_TEST(model_lora_layout_and_init) {
    const char *dir;
    RT_OK(write_ckpt("m_qwen3", k_cfg_qwen3, 1, &dir));
    rrc_model_config c;
    RT_OK(rrc_config_load(dir, 0, &c));
    const rrc_backend *be = rrc_backend_cpu();
    rrc_lora_config lc;
    rrc_lora_config_default(&lc);
    lc.r = 3;
    lc.alpha = 6;
    lc.n_layers = 1;
    lc.layers[0] = 1;
    lc.seed = 42;
    RT_OK(rrc_lora_parse_targets("q_proj,down_proj", &lc.target_mask));
    rrc_lora *l = NULL;
    RT_OK(rrc_lora_create(be, &c, &lc, &l));
    RT_EXPECT(rrc_lora_num_tensors(l) == 4, "tensors %zu", rrc_lora_num_tensors(l));
    RT_EXPECT(rrc_lora_lowest_layer(l) == 1, "lowest layer");
    RT_EXPECT(fabsf(rrc_lora_scaling(l) - 2.0f) < 1e-7f, "scaling");
    rrc_lora_tensor_info ti;
    RT_OK(rrc_lora_tensor(l, 0, &ti));
    RT_EXPECT(!strcmp(ti.name, "base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight"), "%s", ti.name);
    RT_EXPECT(ti.shape[0] == 3 && ti.shape[1] == 16, "q A shape");
    RT_OK(rrc_lora_tensor(l, 3, &ti));
    RT_EXPECT(!strcmp(ti.name, "base_model.model.model.layers.1.mlp.down_proj.lora_B.weight"), "%s", ti.name);
    RT_EXPECT(ti.shape[0] == 16 && ti.shape[1] == 3, "down B shape");
    RT_EXPECT(rrc_lora_num_params(l) == 3 * 16 + 32 * 3 + 3 * 24 + 16 * 3, "params %lld",
              (long long)rrc_lora_num_params(l));
    /* init: A in (-1/sqrt(in), 1/sqrt(in)), B = 0, grads 0; same seed -> same values */
    float *p = dl_all(rrc_lora_params(l), rrc_lora_num_params(l));
    rrc_lora *l2 = NULL;
    RT_OK(rrc_lora_create(be, &c, &lc, &l2));
    float *p2 = dl_all(rrc_lora_params(l2), rrc_lora_num_params(l2));
    RT_EXPECT(!memcmp(p, p2, sizeof(float) * (size_t)rrc_lora_num_params(l)), "init not deterministic");
    for (size_t i = 0; i < rrc_lora_num_tensors(l); i++) {
        RT_OK(rrc_lora_tensor(l, i, &ti));
        double bound = 1.0 / sqrt((double)ti.shape[1]), mx = 0;
        for (int64_t j = 0; j < ti.numel; j++) {
            float v = p[ti.offset + j];
            if (ti.is_b) RT_EXPECT(v == 0.0f, "B not zero");
            mx = fabs(v) > mx ? fabs(v) : mx;
        }
        if (!ti.is_b) RT_EXPECT(mx <= bound && mx > 0.5 * bound, "A bound %g max %g", bound, mx);
    }
    free(p);
    free(p2);
    rrc_lora_free(l2);
    /* save + load round trip with PEFT names */
    rrc_st_writer *w = NULL;
    RT_OK(rrc_st_writer_create(&w));
    randomize_lora(l, 9, 0.3f, 0.3f);
    RT_OK(rrc_lora_add_to_writer(l, 0, NULL, w));
    RT_OK(rrc_st_writer_write(w, rt_path("lora.safetensors")));
    rrc_st_writer_free(w);
    RT_OK(rrc_lora_create(be, &c, &lc, &l2));
    RT_OK(rrc_lora_load_safetensors(l2, rt_path("lora.safetensors")));
    p = dl_all(rrc_lora_params(l), rrc_lora_num_params(l));
    p2 = dl_all(rrc_lora_params(l2), rrc_lora_num_params(l2));
    RT_EXPECT(!memcmp(p, p2, sizeof(float) * (size_t)rrc_lora_num_params(l)), "load round trip");
    free(p);
    free(p2);
    rrc_lora_free(l2);
    /* a different adapter shape refuses the file */
    lc.r = 2;
    RT_OK(rrc_lora_create(be, &c, &lc, &l2));
    RT_STATUS(rrc_lora_load_safetensors(l2, rt_path("lora.safetensors")), RRC_ERR_SHAPE);
    rrc_lora_free(l2);
    lc.r = 3;
    lc.target_mask = 1u << RRC_LORA_Q;
    RT_OK(rrc_lora_create(be, &c, &lc, &l2));
    RT_STATUS(rrc_lora_load_safetensors(l2, rt_path("lora.safetensors")), RRC_ERR_UNSUPPORTED);
    rrc_lora_free(l2);
    lc.target_mask = RRC_LORA_ALL_MODULES;
    RT_OK(rrc_lora_create(be, &c, &lc, &l2));
    RT_STATUS(rrc_lora_load_safetensors(l2, rt_path("lora.safetensors")), RRC_ERR_NOT_FOUND);
    rrc_lora_free(l2);
    /* refusals */
    rrc_lora_config bad = lc;
    bad.dropout = 0.05;
    RT_STATUS(rrc_lora_create(be, &c, &bad, &l2), RRC_ERR_UNSUPPORTED);
    RT_EXPECT(strstr(rrc_last_error(), "lora_dropout") != NULL, "%s", rrc_last_error());
    bad = lc;
    bad.r = 0;
    RT_STATUS(rrc_lora_create(be, &c, &bad, &l2), RRC_ERR_INVALID_ARG);
    bad = lc;
    bad.layers[0] = 2;
    RT_STATUS(rrc_lora_create(be, &c, &bad, &l2), RRC_ERR_INVALID_ARG);
    bad = lc;
    bad.n_layers = 2;
    bad.layers[1] = 1;
    RT_STATUS(rrc_lora_create(be, &c, &bad, &l2), RRC_ERR_INVALID_ARG);
    unsigned mask;
    RT_STATUS(rrc_lora_parse_targets("q_proj,lm_head", &mask), RRC_ERR_UNSUPPORTED);
    RT_STATUS(rrc_lora_parse_targets("q_proj,", &mask), RRC_ERR_INVALID_ARG);
    RT_OK(rrc_lora_parse_targets("all-linear", &mask));
    RT_EXPECT(mask == RRC_LORA_ALL_MODULES, "all-linear");
    rrc_lora_free(l);
}

/* ---- batch preparation ---------------------------------------------------------------------- */

RT_TEST(model_batch_validation) {
    const char *dir;
    RT_OK(write_ckpt("m_qwen2", k_cfg_qwen2, 2, &dir));
    rrc_model_opts mo;
    rrc_model_opts_default(&mo);
    mo.dtype = RRC_F32;
    rrc_model *m = NULL;
    RT_OK(rrc_model_load(rrc_backend_cpu(), dir, &mo, &m));
    tb b;
    const int lens[] = {4, 5, 0, 0};
    make_batch(&b, 2, 9, lens, 24, 3, 1);
    rrc_batch x = as_batch(&b);
    int64_t n;
    RT_OK(rrc_batch_count_labels(&x, &n));
    /* row 0: docs of 4 (targets t=1..3, label 1 masked -> 2) and 5 (4 - 1 = 3); row 1: one doc of 9 (8 - 1) */
    RT_EXPECT(n == 2 + 3 + 7, "labelled %lld", (long long)n);
    rrc_fb_opts fo = window_opts(n);
    rrc_fb_result res;
    tb bad = b;
    bad.tok[3] = 24;
    x = as_batch(&bad);
    RT_STATUS(rrc_model_forward_backward(m, NULL, &x, &fo, &res), RRC_ERR_RANGE);
    bad = b;
    bad.pos[5] = 64;
    x = as_batch(&bad);
    RT_STATUS(rrc_model_forward_backward(m, NULL, &x, &fo, &res), RRC_ERR_RANGE);
    bad = b;
    bad.lab[5] = -3;
    x = as_batch(&bad);
    RT_STATUS(rrc_model_forward_backward(m, NULL, &x, &fo, &res), RRC_ERR_RANGE);
    bad = b;
    bad.doc[8] = bad.doc[0]; /* doc 0 reappears after doc 10 */
    x = as_batch(&bad);
    RT_STATUS(rrc_model_forward_backward(m, NULL, &x, &fo, &res), RRC_ERR_INVALID_ARG);
    RT_EXPECT(strstr(rrc_last_error(), "contiguous") != NULL, "%s", rrc_last_error());
    x = as_batch(&b);
    fo.window_tokens = n - 1;
    RT_STATUS(rrc_model_forward_backward(m, NULL, &x, &fo, &res), RRC_ERR_INVALID_ARG);
    fo.window_tokens = n;
    fo.reduction = RRC_LOSS_MICROBATCH_MEAN;
    fo.num_micro_batches = 0;
    RT_STATUS(rrc_model_forward_backward(m, NULL, &x, &fo, &res), RRC_ERR_INVALID_ARG);
    /* derived ids: only doc_ids, only position_ids, neither (one document per row) */
    x = as_batch(&b);
    x.position_ids = NULL;
    int64_t n2;
    RT_OK(rrc_batch_count_labels(&x, &n2));
    RT_EXPECT(n2 == n, "positions derived from doc ids");
    x = as_batch(&b);
    x.doc_ids = NULL;
    RT_OK(rrc_batch_count_labels(&x, &n2));
    RT_EXPECT(n2 == n, "docs derived from positions");
    x.position_ids = NULL;
    RT_OK(rrc_batch_count_labels(&x, &n2));
    /* one document per row: 8 targets each; row 0 masks positions 1 and 5, row 1 position 1 */
    RT_EXPECT(n2 == 6 + 7, "one document per row: %lld", (long long)n2);
    x.labels = NULL;
    RT_OK(rrc_batch_count_labels(&x, &n2));
    RT_EXPECT(n2 == 0, "no labels");
    x.rows = 0;
    RT_STATUS(rrc_batch_count_labels(&x, &n2), RRC_ERR_SHAPE);
    rrc_model_free(m);
}

/* ---- gradient vs central finite differences of the loss --------------------------------------- */

static rrc_status loss_at(rrc_model *m, rrc_lora *l, const rrc_batch *x, int64_t nt, double *loss) {
    rrc_fb_opts fo = window_opts(nt);
    fo.compute_grad = 0;
    rrc_fb_result res;
    RRC_CHECK(rrc_model_forward_backward(m, l, x, &fo, &res));
    *loss = res.loss;
    return RRC_OK;
}

static void fd_check(const char *cfg, const char *name, uint64_t seed, unsigned targets, int64_t n_layers,
                     const int64_t *layers, int use_rslora) {
    const char *dir;
    RT_OK(write_ckpt(name, cfg, seed, &dir));
    rrc_model_opts mo;
    rrc_model_opts_default(&mo);
    mo.dtype = RRC_F32;
    mo.attn_chunk = 3;
    mo.ce_chunk = 4;
    rrc_model *m = NULL;
    const rrc_backend *be = rrc_backend_cpu();
    RT_OK(rrc_model_load(be, dir, &mo, &m));
    const rrc_model_config *c = rrc_model_get_config(m);
    rrc_lora_config lc;
    rrc_lora_config_default(&lc);
    lc.r = 2;
    lc.alpha = 3;
    lc.use_rslora = use_rslora;
    lc.target_mask = targets;
    lc.n_layers = n_layers;
    for (int64_t i = 0; i < n_layers; i++) lc.layers[i] = layers[i];
    rrc_lora *l = NULL;
    RT_OK(rrc_lora_create(be, c, &lc, &l));
    randomize_lora(l, seed + 7, 0.4f, 0.4f);
    tb b;
    const int lens[] = {3, 5, 0, 2, 0}; /* row 0: 3 + 5 + 3; row 1: 2 + 9 */
    make_batch(&b, 2, 11, lens, c->vocab_size, seed + 3, 1);
    rrc_batch x = as_batch(&b);
    int64_t nt;
    RT_OK(rrc_batch_count_labels(&x, &nt));
    RT_ASSERT(nt > 4, "labelled %lld", (long long)nt);
    rrc_fb_opts fo = window_opts(nt);
    rrc_fb_result res;
    RT_OK(rrc_lora_zero_grads(l));
    RT_OK(rrc_model_forward_backward(m, l, &x, &fo, &res));
    RT_EXPECT(res.n_tokens == nt && isfinite(res.loss) && res.loss > 0, "loss %g", res.loss);
    const int64_t np = rrc_lora_num_params(l);
    float *g = dl_all(rrc_lora_grads(l), np), *p0 = dl_all(rrc_lora_params(l), np);
    float *pp = (float *)malloc((size_t)np * sizeof(float));
    double gnorm = 0;
    for (int64_t i = 0; i < np; i++) gnorm += (double)g[i] * g[i];
    RT_EXPECT(gnorm > 0, "zero gradient");
    /* two random directions over all parameters, then one per tensor (catches a wrong tensor) */
    rt_rng rng = {seed + 11};
    const size_t nt_ = rrc_lora_num_tensors(l);
    for (size_t dirn = 0; dirn < 2 + nt_; dirn++) {
        double dd = 0;
        float *v = (float *)calloc((size_t)np, sizeof(float));
        rrc_lora_tensor_info ti;
        int64_t lo = 0, hi = np;
        if (dirn >= 2) {
            RT_OK(rrc_lora_tensor(l, dirn - 2, &ti));
            lo = ti.offset;
            hi = ti.offset + ti.numel;
        }
        for (int64_t i = lo; i < hi; i++) {
            v[i] = rt_uniform(&rng);
            dd += (double)v[i] * g[i];
        }
        const double eps = 1e-3;
        double lp, lm;
        for (int64_t i = 0; i < np; i++) pp[i] = (float)(p0[i] + eps * v[i]);
        RT_OK(rrc_buf_upload(rrc_lora_params(l), pp, np));
        RT_OK(loss_at(m, l, &x, nt, &lp));
        for (int64_t i = 0; i < np; i++) pp[i] = (float)(p0[i] - eps * v[i]);
        RT_OK(rrc_buf_upload(rrc_lora_params(l), pp, np));
        RT_OK(loss_at(m, l, &x, nt, &lm));
        RT_OK(rrc_buf_upload(rrc_lora_params(l), p0, np));
        const double fd = (lp - lm) / (2 * eps);
        RT_EXPECT(fabs(fd - dd) <= 2e-3 * fabs(dd) + 2e-4, "%s direction %zu: finite difference %.8g, gradient %.8g",
                  name, dirn, fd, dd);
        RT_EXPECT(fabs(dd) > 1e-3, "%s direction %zu: directional derivative %.3g too small to test", name, dirn, dd);
        free(v);
    }
    free(g);
    free(p0);
    free(pp);
    rrc_lora_free(l);
    rrc_model_free(m);
}

RT_TEST(model_grad_finite_differences) {
    /* qwen3: q/k norm, biases, tied head, GQA, head_dim != hidden/heads; all modules, all layers */
    fd_check(k_cfg_qwen3, "fd_qwen3", 21, RRC_LORA_ALL_MODULES, 0, NULL, 0);
    /* llama3 rope, untied; upper layers only (backward stops at layer 1) */
    const int64_t up[2] = {1, 2};
    fd_check(k_cfg_llama, "fd_llama", 22, (1u << RRC_LORA_K) | (1u << RRC_LORA_O) | (1u << RRC_LORA_GATE), 2, up, 1);
    /* qwen2 biases, all modules */
    fd_check(k_cfg_qwen2, "fd_qwen2", 23, RRC_LORA_ALL_MODULES, 0, NULL, 0);
}

/* ---- invariances: recompute policy, chunk sizes, backends, workspace reuse, bf16 --------------- */

typedef struct run_out {
    double loss;
    float *grads, *logits;
    int64_t np, nl;
} run_out;

static rrc_status run_once(const char *dir, const rrc_backend *be, rrc_dtype dt, int64_t ac, int64_t cc,
                           rrc_recompute rc, const rrc_batch *x, int64_t mb, run_out *o, int64_t *allocs) {
    rrc_model_opts mo;
    rrc_model_opts_default(&mo);
    mo.dtype = dt;
    mo.attn_chunk = ac;
    mo.ce_chunk = cc;
    mo.recompute = rc;
    rrc_model *m = NULL;
    RRC_CHECK(rrc_model_load(be, dir, &mo, &m));
    const rrc_model_config *c = rrc_model_get_config(m);
    rrc_lora_config lc;
    rrc_lora_config_default(&lc);
    lc.r = 3;
    rrc_lora *l = NULL;
    rrc_status st = rrc_lora_create(be, c, &lc, &l);
    if (st == RRC_OK) randomize_lora(l, 77, 0.3f, 0.3f);
    int64_t total = 0;
    for (int64_t i = 0; st == RRC_OK && i < mb; i++) {
        int64_t n;
        st = rrc_batch_count_labels(&x[i], &n);
        total += n;
    }
    o->loss = 0;
    for (int64_t i = 0; st == RRC_OK && i < mb; i++) {
        rrc_fb_opts fo = window_opts(total);
        rrc_fb_result res;
        st = rrc_model_forward_backward(m, l, &x[i], &fo, &res);
        o->loss += res.loss;
    }
    if (st == RRC_OK) {
        o->np = rrc_lora_num_params(l);
        o->grads = dl_all(rrc_lora_grads(l), o->np);
        o->nl = x[0].rows * x[0].seq_len * c->vocab_size;
        o->logits = (float *)malloc((size_t)o->nl * sizeof(float));
        st = o->logits ? rrc_model_forward_logits(m, l, &x[0], o->logits, NULL) : RRC_ERR_OOM;
    }
    if (allocs) *allocs = rrc_model_workspace_allocs(m);
    rrc_lora_free(l);
    rrc_model_free(m);
    return st;
}

static void run_free(run_out *o) {
    free(o->grads);
    free(o->logits);
}

static double max_rel(const float *a, const float *b, int64_t n) {
    double num = 0, den = 0;
    for (int64_t i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        num += d * d;
        den += (double)b[i] * b[i];
    }
    return den > 0 ? sqrt(num / den) : sqrt(num);
}

RT_TEST(model_invariances) {
    const char *dir;
    RT_OK(write_ckpt("inv_qwen3", k_cfg_qwen3, 31, &dir));
    tb b0, b1;
    const int l0[] = {4, 6, 0, 0}, l1[] = {2, 2, 2, 0, 7, 0};
    make_batch(&b0, 2, 13, l0, 32, 5, 2);
    make_batch(&b1, 2, 13, l1, 32, 6, 1);
    rrc_batch x[2] = {as_batch(&b0), as_batch(&b1)};
    const rrc_backend *be = rrc_backend_cpu();
    run_out ref, o;
    int64_t allocs = 0;
    RT_OK(run_once(dir, be, RRC_F32, 64, 64, RRC_RECOMPUTE_FULL, x, 2, &ref, &allocs));
    /* the workspace is planned on the first step and reused (logits pass included) */
    RT_EXPECT(allocs == 1, "workspace allocations: %lld", (long long)allocs);
    RT_OK(run_once(dir, be, RRC_F32, 64, 64, RRC_RECOMPUTE_NONE, x, 2, &o, NULL));
    RT_EXPECT(o.loss == ref.loss && !memcmp(o.grads, ref.grads, sizeof(float) * (size_t)ref.np) &&
                  !memcmp(o.logits, ref.logits, sizeof(float) * (size_t)ref.nl),
              "recompute none is not bit-identical to full");
    run_free(&o);
    const int64_t chunks[3][2] = {{1, 1}, {4, 3}, {5, 1000}};
    for (int k = 0; k < 3; k++) {
        RT_OK(run_once(dir, be, RRC_F32, chunks[k][0], chunks[k][1], RRC_RECOMPUTE_FULL, x, 2, &o, NULL));
        RT_EXPECT(fabs(o.loss - ref.loss) <= 1e-6 * ref.loss, "chunks %lld/%lld loss", (long long)chunks[k][0],
                  (long long)chunks[k][1]);
        RT_EXPECT(max_rel(o.grads, ref.grads, ref.np) < 1e-5, "chunks %lld/%lld grads %g", (long long)chunks[k][0],
                  (long long)chunks[k][1], max_rel(o.grads, ref.grads, ref.np));
        RT_EXPECT(max_rel(o.logits, ref.logits, ref.nl) < 1e-6, "chunks logits");
        run_free(&o);
    }
    RT_OK(run_once(dir, rrc_backend_cpu_f32acc(), RRC_F32, 3, 2, RRC_RECOMPUTE_FULL, x, 2, &o, NULL));
    RT_EXPECT(max_rel(o.grads, ref.grads, ref.np) < 1e-4, "f32acc grads %g", max_rel(o.grads, ref.grads, ref.np));
    run_free(&o);
    for (int bi = 0; bi < 2; bi++) {
        RT_OK(run_once(dir, rt_backend(bi), RRC_BF16, 3, 2, RRC_RECOMPUTE_FULL, x, 2, &o, NULL));
        int finite = 1;
        for (int64_t i = 0; i < o.np; i++) finite &= isfinite(o.grads[i]) != 0;
        RT_EXPECT(finite, "bf16 grads finite");
        RT_EXPECT(fabs(o.loss - ref.loss) < 0.02 * ref.loss, "bf16 loss %g vs %g", o.loss, ref.loss);
        RT_EXPECT(max_rel(o.grads, ref.grads, ref.np) < 0.1, "bf16 grads rel %g", max_rel(o.grads, ref.grads, ref.np));
        run_free(&o);
    }
    run_free(&ref);
}

/* ---- zero labelled tokens, no LoRA, loss-only ------------------------------------------------ */

RT_TEST(model_zero_labels_and_loss_only) {
    const char *dir;
    RT_OK(write_ckpt("zl_llama", k_cfg_llama, 41, &dir));
    rrc_model_opts mo;
    rrc_model_opts_default(&mo);
    mo.dtype = RRC_F32;
    mo.attn_chunk = 2;
    rrc_model *m = NULL;
    const rrc_backend *be = rrc_backend_cpu();
    RT_OK(rrc_model_load(be, dir, &mo, &m));
    rrc_lora_config lc;
    rrc_lora_config_default(&lc);
    rrc_lora *l = NULL;
    RT_OK(rrc_lora_create(be, rrc_model_get_config(m), &lc, &l));
    randomize_lora(l, 3, 0.3f, 0.3f);
    tb b;
    const int lens[] = {5, 0, 0};
    make_batch(&b, 2, 5, lens, 40, 8, 10); /* every target masked */
    rrc_batch x = as_batch(&b);
    for (int red = 0; red < 2; red++)
        for (int64_t wt = 0; wt <= 10; wt += 10) {
            rrc_fb_opts fo = {(rrc_loss_reduction)red, wt, 3, 1};
            rrc_fb_result res;
            RT_OK(rrc_model_forward_backward(m, l, &x, &fo, &res));
            RT_EXPECT(res.n_tokens == 0 && res.loss == 0.0 && res.loss_sum == 0.0, "zero-label micro-batch: loss %g",
                      res.loss);
            /* window_tokens: the window's factor is still reported; microbatch_mean / empty window: 0 */
            RT_EXPECT(res.grad_scale == ((red == 0 && wt > 0) ? 0.1f : 0.0f), "grad_scale %g", (double)res.grad_scale);
        }
    float *g = dl_all(rrc_lora_grads(l), rrc_lora_num_params(l));
    int zero = 1;
    for (int64_t i = 0; i < rrc_lora_num_params(l); i++) zero &= g[i] == 0.0f;
    RT_EXPECT(zero, "gradients must stay zero");
    free(g);
    /* loss-only and no-LoRA runs leave the gradients untouched and agree on the loss with B = 0 */
    make_batch(&b, 2, 5, lens, 40, 8, 0);
    x = as_batch(&b);
    int64_t nt;
    RT_OK(rrc_batch_count_labels(&x, &nt));
    rrc_fb_opts fo = window_opts(nt);
    fo.compute_grad = 0;
    rrc_fb_result r1, r2;
    RT_OK(rrc_model_forward_backward(m, l, &x, &fo, &r1));
    g = dl_all(rrc_lora_grads(l), rrc_lora_num_params(l));
    zero = 1;
    for (int64_t i = 0; i < rrc_lora_num_params(l); i++) zero &= g[i] == 0.0f;
    RT_EXPECT(zero, "compute_grad = 0 must not touch the gradients");
    free(g);
    rrc_lora_free(l);
    RT_OK(rrc_lora_create(be, rrc_model_get_config(m), &lc, &l)); /* B = 0 */
    RT_OK(rrc_model_forward_backward(m, l, &x, &fo, &r1));
    RT_OK(rrc_model_forward_backward(m, NULL, &x, &fo, &r2));
    RT_EXPECT(r1.loss == r2.loss && r1.n_tokens == nt, "B = 0 adapter vs no adapter: %.9g vs %.9g", r1.loss, r2.loss);
    /* mismatched adapter */
    rrc_model *m2 = NULL;
    const char *dir2;
    RT_OK(write_ckpt("zl_qwen2", k_cfg_qwen2, 42, &dir2));
    rrc_model_opts_default(&mo);
    RT_OK(rrc_model_load(be, dir2, &mo, &m2));
    RT_STATUS(rrc_model_forward_backward(m2, l, &x, &fo, &r1), RRC_ERR_SHAPE);
    rrc_model_free(m2);
    rrc_lora_free(l);
    rrc_model_free(m);
    /* load failures release everything (run under make asan): missing directory, a config whose
     * tensor set does not match the checkpoint, a missing LoRA file */
    RT_STATUS(rrc_model_load(be, rt_path("no_such_model_dir"), NULL, &m), RRC_ERR_NOT_FOUND);
    RT_EXPECT(m == NULL, "model pointer set on failure");
    {
        char file[1400];
        snprintf(file, sizeof file, "%s/config.json", dir2);
        const char *more_layers = "{\"model_type\": \"qwen2\", \"vocab_size\": 24, \"hidden_size\": 16, "
                                  "\"intermediate_size\": 16, \"num_hidden_layers\": 3, \"num_attention_heads\": 4, "
                                  "\"num_key_value_heads\": 2, \"max_position_embeddings\": 64}";
        RT_OK(rrc_write_file_atomic(file, more_layers, strlen(more_layers)));
        RT_STATUS(rrc_model_load(be, dir2, NULL, &m), RRC_ERR_NOT_FOUND);
        RT_EXPECT(strstr(rrc_last_error(), "model.layers.2.") != NULL, "%s", rrc_last_error());
        RT_OK(rrc_write_file_atomic(file, k_cfg_qwen2, strlen(k_cfg_qwen2)));
        rrc_model_opts_default(&mo);
        RT_OK(rrc_model_load(be, dir2, &mo, &m2));
        RT_OK(rrc_lora_create(be, rrc_model_get_config(m2), &lc, &l));
        RT_STATUS(rrc_lora_load_safetensors(l, rt_path("no_such_lora.safetensors")), RRC_ERR_NOT_FOUND);
        rrc_lora_free(l);
        rrc_model_free(m2);
    }
    /* the model loader refuses an f16 model dtype and a bad chunk size */
    rrc_model_opts_default(&mo);
    mo.dtype = RRC_I32;
    RT_STATUS(rrc_model_load(be, dir, &mo, &m), RRC_ERR_DTYPE);
    rrc_model_opts_default(&mo);
    mo.attn_chunk = 0;
    RT_STATUS(rrc_model_load(be, dir, &mo, &m), RRC_ERR_INVALID_ARG);
}

/* config.json's max_position_embeddings is only checked > 0 by the config parser and bounds no
 * tensor: the RoPE table size n_pos * head_dim must not overflow (it is refused before any
 * allocation, so this also runs under ASan's default allocator limits). */
RT_TEST(model_load_refuses_huge_rope_table) {
    const rrc_backend *be = rrc_backend_cpu();
    const char *dir;
    RT_OK(write_ckpt("huge_pos", k_cfg_qwen2, 3, &dir));
    char file[1400];
    snprintf(file, sizeof file, "%s/config.json", dir);
    const char *values[] = {"4611686018427387904", "2147483648", "9223372036854775807"};
    for (int i = 0; i < 3; i++) {
        char cfg[1024];
        snprintf(cfg, sizeof cfg,
                 "{\"model_type\": \"qwen2\", \"vocab_size\": 24, \"hidden_size\": 16, \"intermediate_size\": 16, "
                 "\"num_hidden_layers\": 2, \"num_attention_heads\": 4, \"num_key_value_heads\": 2, "
                 "\"max_position_embeddings\": %s, \"rms_norm_eps\": 1e-6, \"tie_word_embeddings\": false}",
                 values[i]);
        RT_OK(rrc_write_file_atomic(file, cfg, strlen(cfg)));
        rrc_model *m = NULL;
        rrc_model_opts mo;
        rrc_model_opts_default(&mo); /* max_positions 0: the table size comes from config.json */
        RT_STATUS(rrc_model_load(be, dir, &mo, &m), RRC_ERR_UNSUPPORTED);
        RT_EXPECT(m == NULL && strstr(rrc_last_error(), "max_position") != NULL, "%s", rrc_last_error());
        /* an explicit max_positions bounds the table and the model loads */
        mo.max_positions = 32;
        RT_OK(rrc_model_load(be, dir, &mo, &m));
        rrc_model_free(m);
    }
}

/* ---- the GEMM dtype combinations the model issues (input for device backends) ---------------- */

static rrc_backend g_rec_be;
static unsigned g_combo_seen; /* bit (a * 4 + b * 2 + c) for dtypes f32 = 0, bf16 = 1 */
static rrc_status rec_gemm(const rrc_backend *be, const rrc_gemm_args *g) {
    (void)be;
    g_combo_seen |= 1u << ((g->a.dtype == RRC_BF16) * 4 + (g->b.dtype == RRC_BF16) * 2 + (g->c.dtype == RRC_BF16));
    return rrc_backend_cpu()->gemm(rrc_backend_cpu(), g);
}

RT_TEST(model_gemm_dtype_combinations) {
    const char *dir;
    RT_OK(write_ckpt("combo_qwen3", k_cfg_qwen3, 51, &dir));
    g_rec_be = *rrc_backend_cpu();
    g_rec_be.name = "cpu-gemm-recorder";
    g_rec_be.gemm = rec_gemm;
    const rrc_backend *be = &g_rec_be;
    tb b;
    const int lens[] = {4, 0, 0};
    make_batch(&b, 2, 7, lens, 32, 9, 1);
    rrc_batch x = as_batch(&b);
    int64_t nt;
    RT_OK(rrc_batch_count_labels(&x, &nt));
    /* documented in DESIGN.md §10 (A, B, C): with bf16 activations */
    const unsigned want_bf16 = (1u << 7) /* bf16 bf16 bf16 */ | (1u << 6) /* bf16 bf16 f32 */ |
                               (1u << 4) /* bf16 f32 f32 */ | (1u << 3) /* f32 bf16 bf16 */ |
                               (1u << 2) /* f32 bf16 f32 */ | (1u << 1) /* f32 f32 bf16 */ | (1u << 0) /* f32 x3 */;
    for (int dt = 0; dt < 2; dt++) {
        rrc_model_opts mo;
        rrc_model_opts_default(&mo);
        mo.dtype = dt ? RRC_BF16 : RRC_F32;
        mo.attn_chunk = 3;
        rrc_model *m = NULL;
        RT_OK(rrc_model_load(be, dir, &mo, &m));
        rrc_lora_config lc;
        rrc_lora_config_default(&lc);
        rrc_lora *l = NULL;
        RT_OK(rrc_lora_create(be, rrc_model_get_config(m), &lc, &l));
        randomize_lora(l, 5, 0.3f, 0.3f);
        g_combo_seen = 0;
        rrc_fb_opts fo = window_opts(nt);
        rrc_fb_result res;
        RT_OK(rrc_model_forward_backward(m, l, &x, &fo, &res));
        float lg[2 * 7 * 32];
        RT_OK(rrc_model_forward_logits(m, l, &x, lg, NULL));
        RT_EXPECT(g_combo_seen == (dt ? want_bf16 : 1u), "%s model: gemm dtype combinations 0x%02x, documented 0x%02x",
                  dt ? "bf16" : "f32", g_combo_seen, dt ? want_bf16 : 1u);
        rrc_lora_free(l);
        rrc_model_free(m);
    }
}

const rt_case rt_suite_model[] = {
    RT_CASE(model_lora_layout_and_init), RT_CASE(model_batch_validation),      RT_CASE(model_grad_finite_differences),
    RT_CASE(model_invariances),          RT_CASE(model_zero_labels_and_loss_only), RT_CASE(model_gemm_dtype_combinations),
    RT_CASE(model_load_refuses_huge_rope_table),
};
const size_t rt_suite_model_n = sizeof rt_suite_model / sizeof rt_suite_model[0];
