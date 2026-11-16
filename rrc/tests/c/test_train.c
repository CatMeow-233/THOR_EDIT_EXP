/* Training-driver tests (no PyTorch needed): LR schedule values, the clip coefficient, the fused
 * clip + AdamW step against the element kernel, config parsing / round trip / refusals,
 * rrc_state.bin round trip and corruption, adapter_config.json, and a full rrc_train_run on a tiny
 * checkpoint with stop + resume compared byte for byte (runs under `make asan`). The comparison with
 * transformers / torch / PEFT is rrc/tests/test_train_vs_torch.py. */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "ops_util.h"
#include "rrc/rrc.h"
#include "rrc_kc_adamw.h"

RT_TEST(train_lr_schedule_values) {
    /* hand-evaluated transformers formulas */
    RT_EXPECT(rrc_lr_lambda(RRC_LR_CONSTANT, 0, 5, 10) == 1.0, "constant ignores warmup");
    RT_EXPECT(rrc_lr_lambda(RRC_LR_LINEAR, 0, 4, 10) == 0.0, "warmup starts at 0");
    RT_EXPECT(rrc_lr_lambda(RRC_LR_LINEAR, 2, 4, 10) == 0.5, "warmup 2/4");
    RT_EXPECT(rrc_lr_lambda(RRC_LR_LINEAR, 4, 4, 10) == 1.0, "linear peak");
    RT_EXPECT(rrc_lr_lambda(RRC_LR_LINEAR, 7, 4, 10) == 0.5, "linear (10 - 7) / 6");
    RT_EXPECT(rrc_lr_lambda(RRC_LR_LINEAR, 12, 4, 10) == 0.0, "linear clamps at 0");
    RT_EXPECT(rrc_lr_lambda(RRC_LR_COSINE, 7, 4, 10) == 0.5 * (1.0 + cos(3.141592653589793 * 0.5)), "cosine mid");
    RT_EXPECT(rrc_lr_lambda(RRC_LR_COSINE, 10, 4, 10) == 0.0, "cosine end");
    RT_EXPECT(rrc_lr_lambda(RRC_LR_CONSTANT_WITH_WARMUP, 1, 3, 9) == 1.0 / 3.0, "cww warmup");
    RT_EXPECT(rrc_lr_lambda(RRC_LR_CONSTANT_WITH_WARMUP, 5, 3, 9) == 1.0, "cww after");
    RT_EXPECT(rrc_lr_lambda(RRC_LR_LINEAR, 0, 0, 1) == 1.0, "total 1");
    RT_EXPECT(rrc_lr_lambda(RRC_LR_COSINE, 0, 1, 1) == 0.0, "warmup == total");
    RT_EXPECT(rrc_warmup_steps(3.0, 100) == 3 && rrc_warmup_steps(2.7, 100) == 2 && rrc_warmup_steps(0.25, 9) == 3 &&
                  rrc_warmup_steps(0.0, 9) == 0,
              "warmup resolution");
    rrc_lr_schedule s;
    RT_OK(rrc_lr_schedule_parse("cosine", &s));
    RT_EXPECT(s == RRC_LR_COSINE && !strcmp(rrc_lr_schedule_name(s), "cosine"), "parse");
    RT_STATUS(rrc_lr_schedule_parse("cosine_with_restarts", &s), RRC_ERR_UNSUPPORTED);
}

RT_TEST(train_clip_coef) {
    const float n = 2.5f;
    const float want = (1.0f / (n + 1e-6f)) * 0.75f;
    RT_EXPECT(rrc_clip_coef(n, 0.75) == want, "reciprocal * max_norm in f32");
    RT_EXPECT(rrc_clip_coef(0.5f, 1.0) == 1.0f, "no clipping below max_norm");
    RT_EXPECT(rrc_clip_coef(0.0f, 1.0) == 1.0f, "zero norm");
    RT_EXPECT(rrc_clip_coef(100.0f, 0.0) == 1.0f, "max_norm 0 disables");
}

RT_TEST(train_optimizer_step_matches_kernel) {
    const rrc_backend *be = rrc_backend_cpu();
    const int64_t n = 1000;
    float *p = (float *)malloc(n * sizeof(float)), *g = (float *)malloc(n * sizeof(float));
    float *pr = (float *)malloc(n * sizeof(float)), *mr = (float *)calloc((size_t)n, sizeof(float)),
          *vr = (float *)calloc((size_t)n, sizeof(float));
    RT_ASSERT(p && g && pr && mr && vr, "oom");
    rt_rng r = {4};
    rt_fill(&r, p, n, 1.0f);
    memcpy(pr, p, n * sizeof(float));
    rrc_buf bp = rt_upload(be, RRC_F32, p, n), bg = rrc_buf_none();
    rrc_optimizer *o = NULL;
    RT_OK(rrc_optimizer_create(be, n, &o));
    rrc_optim_hparams h = {0.9, 0.999, 1e-8, 0.01, 0.3};
    for (int step = 1; step <= 3; step++) {
        rt_fill(&r, g, n, step == 2 ? 0.001f : 0.05f); /* step 2 is not clipped */
        rrc_buf_free(&bg);
        bg = rt_upload(be, RRC_F32, g, n);
        rrc_optim_stats os;
        RT_OK(rrc_optimizer_step(o, bp, bg, 1e-3 * step, &h, &os));
        double ss = 0;
        for (int64_t i = 0; i < n; i++) ss += (double)g[i] * g[i];
        const float norm = sqrtf((float)ss), coef = rrc_clip_coef(norm, h.max_grad_norm);
        RT_EXPECT(os.grad_norm == norm && os.clip_coef == coef, "step %d norm %g/%g coef %g/%g", step,
                  (double)os.grad_norm, (double)norm, (double)os.clip_coef, (double)coef);
        RT_EXPECT((step == 2) == (coef == 1.0f), "clipping on steps 1 and 3 only");
        rrc_kc_adamw_scalars s = rrc_kc_adamw_prepare(1e-3 * step, 0.9, 0.999, 1e-8, 0.01, step, coef);
        for (int64_t i = 0; i < n; i++) rrc_kc_adamw_elem(&pr[i], g[i], &mr[i], &vr[i], &s);
    }
    float *got = rt_download(bp, n);
    RT_EXPECT(!memcmp(got, pr, n * sizeof(float)), "params bit-identical to the element kernel");
    RT_EXPECT(rrc_optimizer_step_count(o) == 3, "step count");
    free(got);
    /* a non-finite gradient is an error and updates nothing */
    g[17] = NAN;
    rrc_buf_free(&bg);
    bg = rt_upload(be, RRC_F32, g, n);
    rrc_optim_stats os;
    RT_STATUS(rrc_optimizer_step(o, bp, bg, 1e-3, &h, &os), RRC_ERR_RANGE);
    got = rt_download(bp, n);
    RT_EXPECT(!memcmp(got, pr, n * sizeof(float)) && rrc_optimizer_step_count(o) == 3, "unchanged after error");
    free(got);
    RT_STATUS(rrc_optimizer_step(o, rrc_view(bp, 0, 10), bg, 1e-3, &h, &os), RRC_ERR_SHAPE);
    rrc_optimizer_free(o);
    rrc_buf_free(&bp);
    rrc_buf_free(&bg);
    free(p);
    free(g);
    free(pr);
    free(mr);
    free(vr);
}

static const char *k_min_cfg = "{\"model_dir\": \"m\", \"data_path\": \"d\", \"output_dir\": \"o\", \"seq_len\": 8, "
                               "\"max_steps\": 3";

static rrc_status parse_with(const char *extra, rrc_train_config *c) {
    char buf[2048];
    snprintf(buf, sizeof buf, "%s%s}", k_min_cfg, extra);
    return rrc_train_config_parse(buf, strlen(buf), c);
}

RT_TEST(train_config_parse_and_round_trip) {
    static rrc_train_config c, d;
    RT_OK(parse_with("", &c));
    RT_EXPECT(c.dtype == RRC_BF16 && c.grad_accum_steps == 1 && c.learning_rate == 5e-5 &&
                  c.lr_scheduler_type == RRC_LR_LINEAR && c.optim.max_grad_norm == 1.0 && c.seed == 42 && c.prefetch &&
                  c.lora.r == 8 && c.lora.target_mask == RRC_LORA_ALL_MODULES,
              "defaults");
    RT_OK(parse_with(", \"dtype\": \"f32\", \"lora\": {\"r\": 6, \"alpha\": 12, \"target_modules\": [\"v_proj\", "
                     "\"q_proj\"], \"layers_to_transform\": [1, 0], \"use_rslora\": true, \"init_seed\": 9}, "
                     "\"warmup_steps\": 0.25, \"lr_scheduler_type\": \"cosine\", \"resume_from\": null",
                     &c));
    RT_EXPECT(c.lora.r == 6 && c.lora.alpha == 12.0 && c.lora.target_mask == 5u && c.lora.n_layers == 2 &&
                  c.lora.layers[0] == 1 && c.lora.use_rslora && c.lora.seed == 9 && rrc_train_config_warmup(&c) == 1,
              "values");
    /* to_json -> parse gives the same config (fingerprints equal) */
    char *js = NULL, *f1 = NULL, *f2 = NULL;
    RT_OK(rrc_train_config_to_json(&c, &js));
    rrc_status st = rrc_train_config_parse(js, strlen(js), &d);
    free(js);
    RT_OK(st);
    RT_OK(rrc_train_config_fingerprint(&c, &f1));
    RT_OK(rrc_train_config_fingerprint(&d, &f2));
    RT_EXPECT(!strcmp(f1, f2), "round trip:\n%s\nvs\n%s", f1, f2);
    free(f1);
    free(f2);
    struct {
        const char *extra, *msg;
    } bad[] = {
        {", \"bogus\": 1", "unknown key \"bogus\""},
        {", \"lora\": {\"rank\": 4}", "unknown key \"lora.rank\""},
        {", \"lora\": {\"dropout\": 0.05}", "lora.dropout"},
        {", \"lora\": {\"target_modules\": []}", "non-empty"},
        {", \"lora\": {\"target_modules\": [\"q_proj\", \"q_proj\"]}", "twice"},
        {", \"lora\": {\"target_modules\": [\"embed_tokens\"]}", "embed_tokens"},
        {", \"micro_batch_size\": 0", "micro_batch_size"},
        {", \"seq_len\": 8", "duplicate"},
        {", \"learning_rate\": -1", "learning_rate"},
        {", \"adam_beta2\": 1.0", "adam_beta1 / adam_beta2"},
        {", \"truncation\": \"wrap\"", "\"truncation\" = \"wrap\""},
        {", \"stop_after_step\": 4", "stop_after_step"},
        {", \"init_adapter\": \"a\", \"resume_from\": \"b\"", "exclusive"},
        {", \"shuffle\": 1", "shuffle"},
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        rrc_clear_error();
        st = parse_with(bad[i].extra, &c);
        RT_EXPECT(st != RRC_OK && strstr(rrc_last_error(), bad[i].msg) != NULL, "%s -> %s", bad[i].extra,
                  rrc_last_error());
    }
    const char *empty = "{\"model_dir\": \"\", \"data_path\": \"d\", \"output_dir\": \"o\", \"seq_len\": 8, "
                        "\"max_steps\": 3}";
    RT_STATUS(rrc_train_config_parse(empty, strlen(empty), &c), RRC_ERR_INVALID_ARG);
    RT_EXPECT(strstr(rrc_last_error(), "non-empty") != NULL, "%s", rrc_last_error());
    const char *missing = "{\"model_dir\": \"m\", \"data_path\": \"d\", \"output_dir\": \"o\", \"seq_len\": 8}";
    RT_STATUS(rrc_train_config_parse(missing, strlen(missing), &c), RRC_ERR_PARSE);
    RT_EXPECT(strstr(rrc_last_error(), "max_steps") != NULL, "%s", rrc_last_error());
}

RT_TEST(train_state_round_trip_and_corruption) {
    rrc_train_state s, r;
    memset(&s, 0, sizeof s);
    float m[5] = {1, 2, 3, 4, 5}, v[5] = {0.5f, 0, -0.0f, 1e-30f, 7};
    char cfg[] = "a=1\nb=2\n";
    s.step = 7;
    s.n_params = 5;
    s.data_seed = 0xFFFFFFFFFFFFFFFFull;
    s.cursor.epoch = 3;
    s.cursor.index = 11;
    s.cursor.micro_batches = 14;
    s.tokens_seen = 1234;
    s.params_hash = 99;
    s.data_hash = 0x8000000000000000ull;
    s.config_text = cfg;
    s.m = m;
    s.v = v;
    const char *p = rt_path("state.bin");
    RT_OK(rrc_train_state_write(p, &s));
    RT_OK(rrc_train_state_read(p, &r));
    RT_EXPECT(r.step == 7 && r.n_params == 5 && r.data_seed == s.data_seed && r.cursor.epoch == 3 &&
                  r.cursor.index == 11 && r.cursor.micro_batches == 14 && r.tokens_seen == 1234 && r.params_hash == 99 &&
                  r.data_hash == s.data_hash && !strcmp(r.config_text, cfg) && !memcmp(r.m, m, sizeof m) &&
                  !memcmp(r.v, v, sizeof v),
              "round trip");
    rrc_train_state_free(&r);
    FILE *f = fopen(p, "rb");
    RT_ASSERT(f, "open");
    unsigned char buf[512];
    size_t n = fread(buf, 1, sizeof buf, f);
    fclose(f);
    for (size_t cut = 0; cut < n; cut += 7) {
        f = fopen(p, "wb");
        RT_ASSERT(f && fwrite(buf, 1, cut, f) == cut, "write");
        fclose(f);
        RT_EXPECT(rrc_train_state_read(p, &r) != RRC_OK, "truncated at %zu accepted", cut);
    }
    for (size_t at = 0; at < n; at += 5) {
        buf[at] ^= 0x10;
        f = fopen(p, "wb");
        RT_ASSERT(f && fwrite(buf, 1, n, f) == n, "write");
        fclose(f);
        RT_EXPECT(rrc_train_state_read(p, &r) != RRC_OK, "corruption at byte %zu accepted", at);
        buf[at] ^= 0x10;
    }
}

RT_TEST(train_adapter_config_json) {
    rrc_lora_config lc;
    rrc_lora_config_default(&lc);
    lc.r = 6;
    lc.alpha = 12.5;
    lc.target_mask = (1u << RRC_LORA_Q) | (1u << RRC_LORA_DOWN);
    lc.n_layers = 1;
    lc.layers[0] = 3;
    char *js = NULL;
    RT_OK(rrc_peft_adapter_config_json(&lc, "/models/x", &js));
    rrc_json_doc *doc = NULL;
    rrc_status st = rrc_json_parse(js, strlen(js), &doc);
    free(js);
    RT_OK(st);
    const rrc_json *root = rrc_json_root(doc);
    const rrc_json *tm = rrc_json_get(root, "target_modules"), *lt = rrc_json_get(root, "layers_to_transform");
    RT_EXPECT(!strcmp(rrc_json_str(rrc_json_get(root, "peft_type")), "LORA") &&
                  !strcmp(rrc_json_str(rrc_json_get(root, "task_type")), "CAUSAL_LM") &&
                  rrc_json_get(root, "r")->u.i == 6 && rrc_json_get(root, "lora_alpha")->u.d == 12.5 &&
                  rrc_json_arr_count(tm) == 2 && !strcmp(rrc_json_str(rrc_json_arr_get(tm, 1)), "down_proj") &&
                  rrc_json_arr_count(lt) == 1 && rrc_json_arr_get(lt, 0)->u.i == 3,
              "fields");
    rrc_json_doc_free(doc);
}

/* ---- end to end ------------------------------------------------------------------------------- */

static const char *k_tiny_cfg =
    "{\"model_type\": \"qwen2\", \"vocab_size\": 24, \"hidden_size\": 16, \"intermediate_size\": 16, "
    "\"num_hidden_layers\": 2, \"num_attention_heads\": 4, \"num_key_value_heads\": 2, "
    "\"max_position_embeddings\": 64, \"rms_norm_eps\": 1e-6, \"tie_word_embeddings\": false, "
    "\"rope_theta\": 10000.0}";

static rrc_status write_tiny_ckpt(const char *dir) {
    mkdir(dir, 0755);
    char file[1400];
    snprintf(file, sizeof file, "%s/config.json", dir);
    RRC_CHECK(rrc_write_file_atomic(file, k_tiny_cfg, strlen(k_tiny_cfg)));
    rrc_model_config c;
    RRC_CHECK(rrc_config_load(dir, 0, &c));
    rrc_tensor_spec *specs = NULL;
    size_t n = 0;
    RRC_CHECK(rrc_config_expected_tensors(&c, &specs, &n));
    rrc_st_writer *w = NULL;
    rrc_status st = rrc_st_writer_create(&w);
    rt_rng rng = {21};
    for (size_t i = 0; st == RRC_OK && i < n; i++) {
        int64_t numel = specs[i].shape[0] * (specs[i].ndim == 2 ? specs[i].shape[1] : 1);
        float *x = (float *)malloc((size_t)numel * sizeof(float));
        if (!x) {
            st = RRC_ERR_OOM;
            break;
        }
        const int is_norm = strstr(specs[i].name, "norm.weight") != NULL;
        for (int64_t j = 0; j < numel; j++) x[j] = is_norm ? 1.0f + 0.1f * rt_uniform(&rng) : 0.3f * rt_uniform(&rng);
        st = rrc_st_writer_add(w, specs[i].name, RRC_F32, specs[i].ndim, specs[i].shape, x);
        free(x);
    }
    snprintf(file, sizeof file, "%s/model.safetensors", dir);
    if (st == RRC_OK) st = rrc_st_writer_write(w, file);
    rrc_st_writer_free(w);
    free(specs);
    return st;
}

static rrc_status write_tiny_data(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return RRC_ERROR(RRC_ERR_IO, "open %s", path);
    rt_rng r = {5};
    const int nd = 9;
    uint32_t lens[9];
    uint64_t total = 0;
    for (int d = 0; d < nd; d++) total += lens[d] = 2 + (uint32_t)(rt_next(&r) % 9);
    unsigned char h[32];
    memcpy(h, "RRCBIN01", 8);
    const uint64_t hv[3] = {(uint64_t)nd, 24, total};
    for (int k = 0; k < 3; k++)
        for (int i = 0; i < 8; i++) h[8 + 8 * k + i] = (unsigned char)(hv[k] >> (8 * i));
    int ok = fwrite(h, 1, 32, f) == 32;
    for (int d = 0; d < nd && ok; d++) {
        unsigned char b[4];
        for (int i = 0; i < 4; i++) b[i] = (unsigned char)(lens[d] >> (8 * i));
        ok = fwrite(b, 1, 4, f) == 4;
        for (uint32_t t = 0; t < lens[d] && ok; t++) {
            const uint32_t tok = (uint32_t)(rt_next(&r) % 24);
            for (int i = 0; i < 4; i++) b[i] = (unsigned char)(tok >> (8 * i));
            ok = fwrite(b, 1, 4, f) == 4;
        }
        for (uint32_t t = 0; t < lens[d] && ok; t++) ok = fputc(t >= 1, f) != EOF;
    }
    ok = (fclose(f) == 0) && ok;
    return ok ? RRC_OK : RRC_ERROR(RRC_ERR_IO, "write %s", path);
}

static int files_equal(const char *a, const char *b) {
    FILE *fa = fopen(a, "rb"), *fb = fopen(b, "rb");
    int eq = fa && fb;
    while (eq) {
        int ca = fgetc(fa), cb = fgetc(fb);
        if (ca != cb) eq = 0;
        if (ca == EOF || cb == EOF) break;
    }
    if (fa) fclose(fa);
    if (fb) fclose(fb);
    return eq;
}

static rrc_status run_cfg(const char *json) { /* parse + run */
    static rrc_train_config c;
    RRC_CHECK(rrc_train_config_parse(json, strlen(json), &c));
    return rrc_train_run(&c);
}

RT_TEST(train_end_to_end_stop_and_resume_bit_identical) {
    char model[1200], data[1200], full[1200], part[1200], json[8192];
    snprintf(model, sizeof model, "%s", rt_path("e2e_model"));
    snprintf(data, sizeof data, "%s", rt_path("e2e.rrcbin"));
    snprintf(full, sizeof full, "%s", rt_path("e2e_full"));
    snprintf(part, sizeof part, "%s", rt_path("e2e_part"));
    RT_OK(write_tiny_ckpt(model));
    RT_OK(write_tiny_data(data));
    const char *common = "\"dtype\": \"f32\", \"seq_len\": 12, \"micro_batch_size\": 2, \"grad_accum_steps\": 2, "
                         "\"max_steps\": 5, \"learning_rate\": 0.01, \"lr_scheduler_type\": \"linear\", "
                         "\"warmup_steps\": 1, \"max_grad_norm\": 0.05, \"seed\": 7, \"overwrite_output_dir\": true, "
                         "\"lora\": {\"r\": 2, \"alpha\": 4, \"init_seed\": 3}, \"attn_chunk\": 5, \"ce_chunk\": 3";
    snprintf(json, sizeof json,
             "{\"model_dir\": \"%s\", \"data_path\": \"%s\", \"output_dir\": \"%s\", \"prefetch\": false, %s}", model,
             data, full, common);
    RT_OK(run_cfg(json));
    snprintf(json, sizeof json,
             "{\"model_dir\": \"%s\", \"data_path\": \"%s\", \"output_dir\": \"%s\", \"stop_after_step\": 2, %s}",
             model, data, part, common);
    RT_OK(run_cfg(json));
    snprintf(json, sizeof json,
             "{\"model_dir\": \"%s\", \"data_path\": \"%s\", \"output_dir\": \"%s\", \"resume_from\": "
             "\"%s/checkpoint-2\", %s}",
             model, data, part, part, common);
    RT_OK(run_cfg(json));
    static const char *const files[] = {"adapter_model.safetensors", "adapter_config.json", "rrc_state.bin"};
    char a[1400], b[1400];
    for (int i = 0; i < 3; i++) {
        snprintf(a, sizeof a, "%s/checkpoint-5/%s", full, files[i]);
        snprintf(b, sizeof b, "%s/checkpoint-5/%s", part, files[i]);
        RT_EXPECT(files_equal(a, b), "%s differs after resume", files[i]);
    }
    /* a resume with a changed trajectory setting is refused */
    char changed[2048];
    snprintf(changed, sizeof changed, "%s", common);
    char *lr = strstr(changed, "0.01");
    RT_ASSERT(lr, "lr");
    lr[3] = '2';
    snprintf(json, sizeof json,
             "{\"model_dir\": \"%s\", \"data_path\": \"%s\", \"output_dir\": \"%s\", \"resume_from\": "
             "\"%s/checkpoint-2\", %s}",
             model, data, part, part, changed);
    RT_STATUS(run_cfg(json), RRC_ERR_INVALID_ARG);
    RT_EXPECT(strstr(rrc_last_error(), "learning_rate") != NULL, "%s", rrc_last_error());
    /* the metrics of the resumed run are the uninterrupted run's (minus the wall-clock fields) */
    snprintf(a, sizeof a, "%s/metrics.jsonl", full);
    snprintf(b, sizeof b, "%s/metrics.jsonl", part);
    FILE *fa = fopen(a, "r"), *fb = fopen(b, "r");
    RT_ASSERT(fa && fb, "metrics files");
    char la[1024], lb[1024];
    int lines = 0;
    while (fgets(la, sizeof la, fa)) {
        RT_ASSERT(fgets(lb, sizeof lb, fb), "resumed metrics shorter");
        char *ta = strstr(la, "\"step_time_s\""), *tb = strstr(lb, "\"step_time_s\"");
        RT_ASSERT(ta && tb, "fields");
        *ta = *tb = '\0';
        RT_EXPECT(!strcmp(la, lb), "metrics line %d: %s vs %s", lines + 1, la, lb);
        lines++;
    }
    RT_EXPECT(lines == 5 && !fgets(lb, sizeof lb, fb), "5 lines each");
    fclose(fa);
    fclose(fb);
}

/* ---- counters, checkpoint replacement, output_dir guard, adapter_config roots ------------------ */

/* rename() fault injection (the test binary is linked with -Wl,--wrap=rename): the rename
 * g_fail_rename_src -> g_fail_rename_dst fails with EIO. */
int __real_rename(const char *from, const char *to);
int __wrap_rename(const char *from, const char *to);
static char g_fail_rename_src[1400], g_fail_rename_dst[1400];
int __wrap_rename(const char *from, const char *to) {
    if (g_fail_rename_dst[0] && !strcmp(to, g_fail_rename_dst) && !strcmp(from, g_fail_rename_src)) {
        errno = EIO;
        return -1;
    }
    return __real_rename(from, to);
}

/* A crafted rrc_state.bin (checksum recomputed by the writer) with a counter near INT64_MAX is
 * refused at read time; resuming from it would overflow the loader / tokens_seen counters. */
RT_TEST(train_state_counter_bounds) {
    float m[1] = {0}, v[1] = {0};
    char cfg[] = "x";
    const char *p = rt_path("state_bounds.bin");
    const int64_t bad_values[] = {INT64_MAX, INT64_MAX - 8, RRC_DATA_COUNTER_MAX + 1};
    for (int field = 0; field < 5; field++)
        for (size_t k = 0; k < sizeof bad_values / sizeof bad_values[0] + 1; k++) {
            rrc_train_state s, r;
            memset(&s, 0, sizeof s);
            s.step = 1;
            s.n_params = 1;
            s.config_text = cfg;
            s.m = m;
            s.v = v;
            const int ok = k == sizeof bad_values / sizeof bad_values[0];
            const int64_t x = ok ? RRC_DATA_COUNTER_MAX : bad_values[k];
            int64_t *dst[5] = {&s.step, &s.cursor.epoch, &s.cursor.index, &s.cursor.micro_batches, &s.tokens_seen};
            *dst[field] = x;
            RT_OK(rrc_train_state_write(p, &s));
            if (ok) {
                RT_OK(rrc_train_state_read(p, &r));
                rrc_train_state_free(&r);
            } else {
                RT_STATUS(rrc_train_state_read(p, &r), RRC_ERR_PARSE);
                RT_EXPECT(strstr(rrc_last_error(), "out of range") != NULL, "field %d: %s", field, rrc_last_error());
            }
        }
}

static rrc_status copy_file(const char *from, const char *to) {
    FILE *a = fopen(from, "rb"), *b = a ? fopen(to, "wb") : NULL;
    int ok = a && b;
    char buf[4096];
    size_t n;
    while (ok && (n = fread(buf, 1, sizeof buf, a)) > 0) ok = fwrite(buf, 1, n, b) == n;
    if (a) fclose(a);
    if (b) ok = (fclose(b) == 0) && ok;
    return ok ? RRC_OK : RRC_ERROR(RRC_ERR_IO, "copy %s -> %s", from, to);
}

static int ckpt_equal(const char *a, const char *b) {
    static const char *const files[] = {"adapter_model.safetensors", "adapter_config.json", "rrc_state.bin"};
    for (int i = 0; i < 3; i++) {
        char pa[1400], pb[1400];
        snprintf(pa, sizeof pa, "%s/%s", a, files[i]);
        snprintf(pb, sizeof pb, "%s/%s", b, files[i]);
        if (!files_equal(pa, pb)) return 0;
    }
    return 1;
}

static int exists(const char *p) {
    struct stat sb;
    return stat(p, &sb) == 0;
}

/* Replacing an existing checkpoint-N: if the final rename fails, the previous checkpoint-N is put
 * back (still byte-identical and resumable); a checkpoint left as .checkpoint-N.old by a crash
 * between the two renames is restored rather than deleted by the next save of step N. */
RT_TEST(train_checkpoint_replace_is_failure_safe) {
    char model[1200], data[1200], ref[1200], out[1200], json[8192], ck[1400], old[1400], ref_ck[1400];
    snprintf(model, sizeof model, "%s", rt_path("rp_model"));
    snprintf(data, sizeof data, "%s", rt_path("rp.rrcbin"));
    snprintf(ref, sizeof ref, "%s", rt_path("rp_ref"));
    snprintf(out, sizeof out, "%s", rt_path("rp_out"));
    RT_OK(write_tiny_ckpt(model));
    RT_OK(write_tiny_data(data));
    const char *common = "\"dtype\": \"f32\", \"seq_len\": 12, \"micro_batch_size\": 2, \"max_steps\": 3, "
                         "\"seed\": 1, \"prefetch\": false, \"overwrite_output_dir\": true, \"lora\": {\"r\": 2}";
#define RUN_TO(dir, extra)                                                                                             \
    (snprintf(json, sizeof json, "{\"model_dir\": \"%s\", \"data_path\": \"%s\", \"output_dir\": \"%s\", %s%s}",     \
              model, data, dir, common, extra),                                                                       \
     run_cfg(json))
    RT_OK(RUN_TO(ref, ", \"stop_after_step\": 1"));
    RT_OK(RUN_TO(out, ", \"stop_after_step\": 1"));
    snprintf(ref_ck, sizeof ref_ck, "%s/checkpoint-1", ref);
    snprintf(ck, sizeof ck, "%s/checkpoint-1", out);
    snprintf(old, sizeof old, "%s/.checkpoint-1.old", out);
    RT_ASSERT(ckpt_equal(ck, ref_ck), "equal runs give byte-identical checkpoints");
    /* 1. the .tmp -> checkpoint-1 rename of the replacing save fails (after checkpoint-1 -> .old) */
    snprintf(g_fail_rename_dst, sizeof g_fail_rename_dst, "%s", ck);
    snprintf(g_fail_rename_src, sizeof g_fail_rename_src, "%s/.checkpoint-1.tmp", out);
    rrc_status st = RUN_TO(out, ", \"stop_after_step\": 1");
    g_fail_rename_dst[0] = '\0';
    RT_EXPECT(st == RRC_ERR_IO, "failed publish returned %s", rrc_status_str(st));
    RT_EXPECT(ckpt_equal(ck, ref_ck), "checkpoint-1 is not the previous one after a failed replace");
    RT_EXPECT(!exists(old), ".checkpoint-1.old left behind");
    /* 2. a crash between the two renames: checkpoint-1 missing, .checkpoint-1.old present */
    RT_ASSERT(__real_rename(ck, old) == 0, "simulate the crash");
    snprintf(g_fail_rename_dst, sizeof g_fail_rename_dst, "%s", ck);
    snprintf(g_fail_rename_src, sizeof g_fail_rename_src, "%s/.checkpoint-1.tmp", out);
    st = RUN_TO(out, ", \"stop_after_step\": 1");
    g_fail_rename_dst[0] = '\0';
    RT_EXPECT(st == RRC_ERR_IO, "failed publish returned %s", rrc_status_str(st));
    RT_EXPECT(ckpt_equal(ck, ref_ck), "the .old copy was not restored");
    /* 3. the restored checkpoint resumes */
    char res[1200];
    snprintf(res, sizeof res, "%s", rt_path("rp_res"));
    char extra[2000];
    snprintf(extra, sizeof extra, ", \"resume_from\": \"%s\"", ck);
    RT_OK(RUN_TO(res, extra));
    /* 4. a successful replace leaves no .old / .tmp */
    RT_OK(RUN_TO(out, ", \"stop_after_step\": 1"));
    RT_EXPECT(ckpt_equal(ck, ref_ck) && !exists(old), "replace");
    snprintf(old, sizeof old, "%s/.checkpoint-1.tmp", out);
    RT_EXPECT(!exists(old), ".tmp left behind");
#undef RUN_TO
}

/* Resuming into an output_dir that holds another run (metrics.jsonl or checkpoints) is refused like
 * a fresh run, unless overwrite_output_dir is set or output_dir is the checkpoint's parent. */
RT_TEST(train_resume_output_dir_guard) {
    char model[1200], data[1200], a[1200], b[1200], c[1200], json[8192], ck[1400], pa[1400], pb[1400];
    snprintf(model, sizeof model, "%s", rt_path("og_model"));
    snprintf(data, sizeof data, "%s", rt_path("og.rrcbin"));
    snprintf(a, sizeof a, "%s", rt_path("og_A"));
    snprintf(b, sizeof b, "%s", rt_path("og_B"));
    snprintf(c, sizeof c, "%s", rt_path("og_C"));
    RT_OK(write_tiny_ckpt(model));
    RT_OK(write_tiny_data(data));
    const char *common = "\"dtype\": \"f32\", \"seq_len\": 12, \"max_steps\": 3, \"save_steps\": 1, "
                         "\"prefetch\": false, \"lora\": {\"r\": 2}";
#define RUN_TO(dir, extra)                                                                                             \
    (snprintf(json, sizeof json, "{\"model_dir\": \"%s\", \"data_path\": \"%s\", \"output_dir\": \"%s\", %s%s}",     \
              model, data, dir, common, extra),                                                                       \
     run_cfg(json))
    RT_OK(RUN_TO(a, ", \"learning_rate\": 0.5, \"seed\": 11, \"overwrite_output_dir\": true"));
    RT_OK(RUN_TO(b, ", \"stop_after_step\": 1, \"overwrite_output_dir\": true"));
    snprintf(ck, sizeof ck, "%s/checkpoint-1", b);
    char extra[2000];
    snprintf(extra, sizeof extra, ", \"resume_from\": \"%s/\"", ck); /* trailing slash: same parent */
    /* refused: A's metrics.jsonl and checkpoints stay untouched */
    snprintf(pa, sizeof pa, "%s/checkpoint-3/adapter_model.safetensors", a);
    snprintf(pb, sizeof pb, "%s/a3.safetensors", c);
    mkdir(c, 0755);
    RT_OK(copy_file(pa, pb));
    RT_STATUS(RUN_TO(a, extra), RRC_ERR_INVALID_ARG);
    RT_EXPECT(strstr(rrc_last_error(), "overwrite_output_dir") != NULL, "%s", rrc_last_error());
    RT_EXPECT(files_equal(pa, pb), "A's checkpoint-3 was rewritten");
    /* allowed: the checkpoint's own parent, a fresh directory, and A with overwrite_output_dir */
    RT_OK(RUN_TO(b, extra));
    snprintf(pa, sizeof pa, "%s/fresh", c);
    RT_OK(RUN_TO(pa, extra));
    snprintf(pa, sizeof pa, "%s/checkpoint-3", b);
    snprintf(pb, sizeof pb, "%s/fresh/checkpoint-3", c);
    RT_EXPECT(ckpt_equal(pa, pb), "resume into a fresh directory");
    char extra2[2100];
    snprintf(extra2, sizeof extra2, "%s, \"overwrite_output_dir\": true", extra);
    RT_OK(RUN_TO(a, extra2));
    snprintf(pa, sizeof pa, "%s/checkpoint-3", a);
    RT_EXPECT(ckpt_equal(pa, pb), "resume into A with overwrite_output_dir");
#undef RUN_TO
}

/* An adapter_config.json whose root is not an object, or whose target_modules is not a list, cannot
 * be verified against the lora settings and is refused. */
RT_TEST(train_init_adapter_config_roots) {
    char model[1200], data[1200], out[1200], ad[1200], json[8192], p[1400], q[1400];
    snprintf(model, sizeof model, "%s", rt_path("ar_model"));
    snprintf(data, sizeof data, "%s", rt_path("ar.rrcbin"));
    snprintf(out, sizeof out, "%s", rt_path("ar_out"));
    snprintf(ad, sizeof ad, "%s", rt_path("ar_adapter"));
    RT_OK(write_tiny_ckpt(model));
    RT_OK(write_tiny_data(data));
    const char *common = "\"dtype\": \"f32\", \"seq_len\": 12, \"max_steps\": 1, \"prefetch\": false, "
                         "\"overwrite_output_dir\": true, \"lora\": {\"r\": 2}";
    snprintf(json, sizeof json, "{\"model_dir\": \"%s\", \"data_path\": \"%s\", \"output_dir\": \"%s\", %s}", model,
             data, out, common);
    RT_OK(run_cfg(json));
    mkdir(ad, 0755);
    snprintf(p, sizeof p, "%s/checkpoint-1/adapter_model.safetensors", out);
    snprintf(q, sizeof q, "%s/adapter_model.safetensors", ad);
    RT_OK(copy_file(p, q));
    snprintf(p, sizeof p, "%s/checkpoint-1/adapter_config.json", out);
    snprintf(q, sizeof q, "%s/adapter_config.json", ad);
    RT_OK(copy_file(p, q));
    snprintf(json, sizeof json,
             "{\"model_dir\": \"%s\", \"data_path\": \"%s\", \"output_dir\": \"%s\", \"init_adapter\": \"%s\", %s}",
             model, data, out, ad, common);
    RT_OK(run_cfg(json)); /* rrc's own adapter_config.json is accepted */
    const char *bad[] = {"[]", "\"x\"", "7", "null", "{\"target_modules\": \"all-linear\"}",
                         "{\"target_modules\": \".*q_proj\"}", "{\"target_modules\": null}"};
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        RT_OK(rrc_write_file_atomic(q, bad[i], strlen(bad[i])));
        RT_STATUS(run_cfg(json), RRC_ERR_INVALID_ARG);
        RT_EXPECT(strstr(rrc_last_error(), "adapter_config.json") != NULL, "%s: %s", bad[i], rrc_last_error());
    }
    const char *good = "{\"r\": 2, \"target_modules\": [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", "
                       "\"gate_proj\", \"up_proj\", \"down_proj\"]}";
    RT_OK(rrc_write_file_atomic(q, good, strlen(good)));
    RT_OK(run_cfg(json));
}

/* ---- memory pre-flight (DESIGN.md §13) --------------------------------------------------------- */

RT_TEST(train_memory_preflight) {
    char model[1200], data[1200], out[1200], json[8192];
    snprintf(model, sizeof model, "%s", rt_path("pf_model"));
    snprintf(data, sizeof data, "%s", rt_path("pf.rrcbin"));
    snprintf(out, sizeof out, "%s", rt_path("pf_out"));
    RT_OK(write_tiny_ckpt(model));
    RT_OK(write_tiny_data(data));
    const char *common = "\"dtype\": \"bf16\", \"seq_len\": 12, \"micro_batch_size\": 2, \"max_steps\": 1, "
                         "\"lora\": {\"r\": 2, \"alpha\": 4}, \"attn_chunk\": 5, \"ce_chunk\": 3";
    /* config field: null = automatic (-1), 0 = no check, > 0 bytes; negative refused; not in the fingerprint */
    rrc_train_config c, c2;
    RT_OK(parse_with("", &c));
    RT_EXPECT(c.memory_budget_bytes == -1, "default %lld", (long long)c.memory_budget_bytes);
    RT_OK(parse_with(", \"memory_budget_bytes\": null", &c));
    RT_EXPECT(c.memory_budget_bytes == -1, "null");
    RT_OK(parse_with(", \"memory_budget_bytes\": 0", &c));
    RT_EXPECT(c.memory_budget_bytes == 0, "0");
    RT_OK(parse_with(", \"memory_budget_bytes\": 123456789", &c2));
    RT_EXPECT(c2.memory_budget_bytes == 123456789, "bytes");
    RT_STATUS(parse_with(", \"memory_budget_bytes\": -5", &c), RRC_ERR_INVALID_ARG);
    RT_STATUS(parse_with(", \"memory_budget_bytes\": 1.5", &c), RRC_ERR_PARSE);
    RT_OK(parse_with("", &c));
    char *f1 = NULL, *f2 = NULL;
    RT_OK(rrc_train_config_fingerprint(&c, &f1));
    RT_OK(rrc_train_config_fingerprint(&c2, &f2));
    RT_EXPECT(!strcmp(f1, f2), "memory_budget_bytes must not be in the resume fingerprint");
    free(f1);
    free(f2);
    char *js = NULL;
    RT_OK(rrc_train_config_to_json(&c2, &js));
    RT_EXPECT(strstr(js, "\"memory_budget_bytes\": 123456789") != NULL, "to_json: %s", js);
    free(js);

    /* the computed peak of this config, and the budget rules */
    snprintf(json, sizeof json, "{\"model_dir\": \"%s\", \"data_path\": \"%s\", \"output_dir\": \"%s\", %s}", model,
             data, out, common);
    RT_OK(rrc_train_config_parse(json, strlen(json), &c));
    rrc_model_config mc;
    RT_OK(rrc_config_load(model, 0, &mc));
    rrc_train_preflight pf;
    RT_OK(rrc_train_memory_preflight(&c, &mc, 0, 0, &pf)); /* CPU, automatic: unlimited */
    RT_EXPECT(pf.peak_bytes > 0 && pf.budget_bytes == 0 && strstr(pf.budget_source, "CPU"), "cpu auto: %s",
              pf.budget_source);
    RT_EXPECT(pf.peak_bytes == pf.weights_bytes + pf.lora_bytes + pf.adam_bytes + pf.workspace_bytes &&
                  pf.internal_bytes == 0,
              "cpu breakdown");
    const uint64_t peak = pf.peak_bytes;
    c.memory_budget_bytes = (int64_t)peak; /* exactly enough */
    RT_OK(rrc_train_memory_preflight(&c, &mc, 0, 0, &pf));
    c.memory_budget_bytes = (int64_t)peak - 1;
    RT_STATUS(rrc_train_memory_preflight(&c, &mc, 0, 0, &pf), RRC_ERR_OOM);
    RT_EXPECT(strstr(rrc_last_error(), "memory pre-flight") && strstr(rrc_last_error(), "memory_budget_bytes"), "%s",
              rrc_last_error());
    /* CUDA target: + GEMM scratch, cuBLASLt workspace, partials; automatic budget = total - margin */
    c.memory_budget_bytes = -1;
    RT_OK(rrc_train_memory_preflight(&c, &mc, 1, 0, &pf)); /* total unknown: no budget */
    RT_EXPECT(pf.budget_bytes == 0 && strstr(pf.budget_source, "unknown") && pf.internal_bytes >= (32u << 20) &&
                  pf.peak_bytes == peak + pf.internal_bytes,
              "cuda unknown total: %s, internal %llu", pf.budget_source, (unsigned long long)pf.internal_bytes);
    const uint64_t margin = (uint64_t)RRC_TRAIN_CUDA_MEMORY_MARGIN;
    RT_OK(rrc_train_memory_preflight(&c, &mc, 1, margin + pf.peak_bytes, &pf));
    RT_EXPECT(pf.budget_bytes == pf.peak_bytes && strstr(pf.budget_source, "device total"), "cuda budget %s",
              pf.budget_source);
    RT_STATUS(rrc_train_memory_preflight(&c, &mc, 1, margin + pf.peak_bytes - 1, &pf), RRC_ERR_OOM);
    c.memory_budget_bytes = 0; /* disabled */
    RT_OK(rrc_train_memory_preflight(&c, &mc, 1, margin + 1, &pf));

    /* the trainer refuses over budget before it creates output_dir or loads anything */
    snprintf(json, sizeof json,
             "{\"model_dir\": \"%s\", \"data_path\": \"%s\", \"output_dir\": \"%s\", \"memory_budget_bytes\": "
             "%llu, %s}",
             model, data, out, (unsigned long long)(peak - 1), common);
    RT_STATUS(run_cfg(json), RRC_ERR_OOM);
    struct stat sb;
    RT_EXPECT(stat(out, &sb) != 0, "output_dir was created by a refused run");
    snprintf(json, sizeof json,
             "{\"model_dir\": \"%s\", \"data_path\": \"%s\", \"output_dir\": \"%s\", \"memory_budget_bytes\": "
             "%llu, %s}",
             model, data, out, (unsigned long long)peak, common);
    RT_OK(run_cfg(json));
    /* a seq_len beyond the positions keeps its own error (not a pre-flight one) */
    snprintf(json, sizeof json,
             "{\"model_dir\": \"%s\", \"data_path\": \"%s\", \"output_dir\": \"%s\", \"seq_len\": 65, "
             "\"max_steps\": 1, \"overwrite_output_dir\": true}",
             model, data, out);
    RT_STATUS(run_cfg(json), RRC_ERR_INVALID_ARG);
    RT_EXPECT(strstr(rrc_last_error(), "exceeds the model's positions") != NULL, "%s", rrc_last_error());
}

const rt_case rt_suite_train[] = {
    RT_CASE(train_lr_schedule_values),         RT_CASE(train_clip_coef),
    RT_CASE(train_optimizer_step_matches_kernel), RT_CASE(train_config_parse_and_round_trip),
    RT_CASE(train_state_round_trip_and_corruption), RT_CASE(train_adapter_config_json),
    RT_CASE(train_end_to_end_stop_and_resume_bit_identical), RT_CASE(train_state_counter_bounds),
    RT_CASE(train_checkpoint_replace_is_failure_safe), RT_CASE(train_resume_output_dir_guard),
    RT_CASE(train_init_adapter_config_roots),         RT_CASE(train_memory_preflight),
};
const size_t rt_suite_train_n = sizeof rt_suite_train / sizeof rt_suite_train[0];
