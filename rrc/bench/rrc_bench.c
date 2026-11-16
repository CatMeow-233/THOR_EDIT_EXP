/* rrc/bench/rrc_bench.c — training-step benchmark of the rrc engine (DESIGN.md §15.4).
 *
 *   rrc_bench (--model DIR | --synthetic-model llama|qwen2|qwen3) [--seq-len T] [--micro-batch B]
 *             [--grad-accum M] [--warmup W] [--steps N] [--dtype bf16|f32] [--lora-r R] [--lora-alpha A]
 *             [--targets all-linear|q_proj,...] [--recompute full|none] [--attn-chunk C] [--ce-chunk C]
 *             [--lr X] [--weight-decay X] [--max-grad-norm X] [--seed S] [--device D] [--out FILE] [--label TEXT]
 *
 * Loads the model (bf16 base by default) and a LoRA adapter (PEFT init: A uniform, B = 0), builds M
 * synthetic micro-batches of B rows x T random token ids (one document per row, every position
 * labelled: B * T * M input tokens per optimizer step, no padding), then runs W warmup and N
 * measured optimizer steps, each = M x rrc_model_forward_backward + clip + AdamW (constant LR),
 * exactly the trainer's step without the data loader. Output: one JSON document (stdout, or --out)
 * with the same keys as bench/torch_baseline.py: tokens/s (= tokens per step / mean step time),
 * step time mean / p50 / p90 (nearest rank), peak memory, per-step losses, and the GEMM tally and
 * census of one measured step (calls per dtype combination and ladder level, FLOPs computed from
 * the shapes, not measured).
 *
 * Timing: the CUDA build records a cudaEvent on the backend's stream before and after every step
 * and reports the GPU time between them; "measured_on" names the device only then (a CUDA build that
 * is not the HOST-ONLY fake-runtime build, rrc_cuda_build_is_host_shim() == 0). The host build
 * (-DRRC_BENCH_HOST, CPU backend) and the HOST-ONLY build report host wall clock with "status":
 * "smoke" and "measured_on": null: they test the program, they are not benchmarks. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "rrc/rrc.h"
#include "selftest_e2e.h"

#ifndef RRC_BENCH_HOST
#include "rrc_cuda.h"
#endif

#ifndef RRC_GIT_REV
#define RRC_GIT_REV "unknown"
#endif

#define MAX_STEPS 10000

typedef struct args {
    const char *model, *synthetic, *out, *label, *targets;
    int64_t seq_len, micro_batch, grad_accum, warmup, steps, r, attn_chunk, ce_chunk;
    double alpha, lr, weight_decay, max_grad_norm;
    rrc_dtype dtype;
    rrc_recompute recompute;
    uint64_t seed;
    int device;
} args;

static void usage(FILE *f) {
    fprintf(f, "usage: rrc_bench (--model DIR | --synthetic-model llama|qwen2|qwen3) [--seq-len T] [--micro-batch B]\n"
               "                 [--grad-accum M] [--warmup W] [--steps N] [--dtype bf16|f32] [--lora-r R] [--lora-alpha A]\n"
               "                 [--targets all-linear|q_proj,...] [--recompute full|none] [--attn-chunk C] [--ce-chunk C]\n"
               "                 [--lr X] [--weight-decay X] [--max-grad-norm X] [--seed S] [--device D] [--out FILE]\n"
               "                 [--label TEXT]\n");
}

static int parse_i64(const char *s, int64_t lo, int64_t hi, int64_t *out) {
    char *e = NULL;
    long long v = strtoll(s, &e, 10);
    if (!e || *e || v < lo || v > hi) return 0;
    *out = v;
    return 1;
}
static int parse_f64(const char *s, double *out) {
    char *e = NULL;
    double v = strtod(s, &e);
    if (!e || *e || !isfinite(v)) return 0;
    *out = v;
    return 1;
}

/* 0 ok, 1 --help, 2 error (message printed) */
static int parse_args(int argc, char **argv, args *a) {
    memset(a, 0, sizeof *a);
    a->seq_len = 2048;
    a->micro_batch = 1;
    a->grad_accum = 1;
    a->warmup = 3;
    a->steps = 10;
    a->r = 16;
    a->alpha = 32.0;
    a->targets = "all-linear";
    a->attn_chunk = 256;
    a->ce_chunk = 1024;
    a->lr = 1e-4;
    a->weight_decay = 0.0;
    a->max_grad_norm = 1.0;
    a->dtype = RRC_BF16;
    a->recompute = RRC_RECOMPUTE_FULL;
    for (int i = 1; i < argc; i++) {
        const char *k = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        int ok = 1;
        if (!strcmp(k, "--help") || !strcmp(k, "-h")) return 1;
        if (!v) {
            fprintf(stderr, "rrc_bench: %s needs a value (or is unknown)\n", k);
            return 2;
        }
        i++;
        if (!strcmp(k, "--model")) a->model = v;
        else if (!strcmp(k, "--synthetic-model")) a->synthetic = v;
        else if (!strcmp(k, "--out")) a->out = v;
        else if (!strcmp(k, "--label")) a->label = v;
        else if (!strcmp(k, "--targets")) a->targets = v;
        else if (!strcmp(k, "--seq-len")) ok = parse_i64(v, 2, 1 << 20, &a->seq_len);
        else if (!strcmp(k, "--micro-batch")) ok = parse_i64(v, 1, 4096, &a->micro_batch);
        else if (!strcmp(k, "--grad-accum")) ok = parse_i64(v, 1, 65536, &a->grad_accum);
        else if (!strcmp(k, "--warmup")) ok = parse_i64(v, 0, MAX_STEPS, &a->warmup);
        else if (!strcmp(k, "--steps")) ok = parse_i64(v, 1, MAX_STEPS, &a->steps);
        else if (!strcmp(k, "--lora-r")) ok = parse_i64(v, 1, 65536, &a->r);
        else if (!strcmp(k, "--attn-chunk")) ok = parse_i64(v, 1, 1 << 20, &a->attn_chunk);
        else if (!strcmp(k, "--ce-chunk")) ok = parse_i64(v, 1, 1 << 24, &a->ce_chunk);
        else if (!strcmp(k, "--lora-alpha")) ok = parse_f64(v, &a->alpha) && a->alpha > 0;
        else if (!strcmp(k, "--lr")) ok = parse_f64(v, &a->lr) && a->lr >= 0;
        else if (!strcmp(k, "--weight-decay")) ok = parse_f64(v, &a->weight_decay) && a->weight_decay >= 0;
        else if (!strcmp(k, "--max-grad-norm")) ok = parse_f64(v, &a->max_grad_norm);
        else if (!strcmp(k, "--seed")) {
            int64_t s = 0;
            ok = parse_i64(v, 0, INT64_MAX, &s);
            a->seed = (uint64_t)s;
        } else if (!strcmp(k, "--device")) {
            int64_t d = 0;
            ok = parse_i64(v, 0, 1024, &d);
            a->device = (int)d;
        } else if (!strcmp(k, "--dtype")) {
            if (!strcmp(v, "bf16")) a->dtype = RRC_BF16;
            else if (!strcmp(v, "f32")) a->dtype = RRC_F32;
            else ok = 0;
        } else if (!strcmp(k, "--recompute")) {
            if (!strcmp(v, "full")) a->recompute = RRC_RECOMPUTE_FULL;
            else if (!strcmp(v, "none")) a->recompute = RRC_RECOMPUTE_NONE;
            else ok = 0;
        } else {
            fprintf(stderr, "rrc_bench: unknown option %s\n", k);
            return 2;
        }
        if (!ok) {
            fprintf(stderr, "rrc_bench: bad value for %s: '%s'\n", k, v);
            return 2;
        }
    }
    if (!a->model == !a->synthetic) {
        fprintf(stderr, "rrc_bench: exactly one of --model DIR and --synthetic-model FAMILY is required\n");
        return 2;
    }
    if (a->synthetic && strcmp(a->synthetic, "llama") && strcmp(a->synthetic, "qwen2") && strcmp(a->synthetic, "qwen3")) {
        fprintf(stderr, "rrc_bench: --synthetic-model must be llama, qwen2 or qwen3\n");
        return 2;
    }
    if (a->steps + a->warmup > MAX_STEPS) {
        fprintf(stderr, "rrc_bench: warmup + steps must be <= %d\n", MAX_STEPS);
        return 2;
    }
    return 0;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_d(const void *x, const void *y) {
    const double a = *(const double *)x, b = *(const double *)y;
    return a < b ? -1 : a > b;
}
/* nearest-rank percentile of sorted v[n] */
static double pct(const double *v, int64_t n, double p) {
    int64_t r = (int64_t)ceil(p / 100.0 * (double)n);
    if (r < 1) r = 1;
    if (r > n) r = n;
    return v[r - 1];
}
static void jw_stats(rrc_jw *w, const char *key, const double *x, int64_t n, int valid) {
    rrc_jw_key(w, key);
    if (!valid || n <= 0) {
        rrc_jw_null(w);
        return;
    }
    double *s = (double *)malloc((size_t)n * sizeof(double));
    if (!s) {
        rrc_jw_null(w);
        return;
    }
    memcpy(s, x, (size_t)n * sizeof(double));
    qsort(s, (size_t)n, sizeof(double), cmp_d);
    double sum = 0;
    for (int64_t i = 0; i < n; i++) sum += s[i];
    rrc_jw_obj_begin(w);
    rrc_jw_key(w, "mean");
    rrc_jw_double(w, sum / (double)n);
    rrc_jw_key(w, "p50");
    rrc_jw_double(w, pct(s, n, 50));
    rrc_jw_key(w, "p90");
    rrc_jw_double(w, pct(s, n, 90));
    rrc_jw_key(w, "min");
    rrc_jw_double(w, s[0]);
    rrc_jw_key(w, "max");
    rrc_jw_double(w, s[n - 1]);
    rrc_jw_obj_end(w);
    free(s);
}

static void targets_json(rrc_jw *w, unsigned mask) {
    rrc_jw_arr_begin(w);
    for (int i = 0; i < RRC_LORA_N_MODULES; i++)
        if (mask & (1u << i)) rrc_jw_str(w, rrc_lora_module_name(i));
    rrc_jw_arr_end(w);
}

typedef struct run {
    const rrc_backend *be;
    rrc_model *model;
    rrc_lora *lora;
    rrc_optimizer *opt;
    int32_t *tok, *lab, *pos, *doc; /* M micro-batches */
    double *gpu_s, *wall_s, *loss, *gnorm;
    int64_t n_rec;
    int timing_valid;
    char measured_on[300];
    char status[200];
    char timing_source[120];
    size_t dev_used_max, dev_total;
    int have_dev_mem;
    char tmpdir[512];
#ifndef RRC_BENCH_HOST
    rrc_cuda_timer *timer;
    rrc_cuda_device_info info;
    int have_info;
    rrc_cuda_stats stats_step, stats_all;
    rrc_cuda_gemm_shape *census;
    size_t n_census;
    int64_t census_dropped;
#endif
} run;

#ifndef RRC_BENCH_HOST
static void sample_mem(run *R) {
    rrc_cuda_device_info in;
    if (rrc_cuda_device_info_get(R->be, &in) == RRC_OK) {
        const size_t used = in.total_mem - in.free_mem;
        if (used > R->dev_used_max) R->dev_used_max = used;
        R->dev_total = in.total_mem;
        R->have_dev_mem = 1;
    } else {
        rrc_clear_error();
    }
}
#endif

static rrc_status one_step(run *R, const args *a, int64_t window_tokens, double *loss, double *gnorm) {
    const int64_t BT = a->micro_batch * a->seq_len;
    RRC_CHECK(rrc_lora_zero_grads(R->lora));
    *loss = 0.0;
    for (int64_t i = 0; i < a->grad_accum; i++) {
        rrc_batch b = {a->micro_batch, a->seq_len, R->tok + i * BT, R->lab + i * BT, R->pos + i * BT, R->doc + i * BT};
        rrc_fb_opts fo = {RRC_LOSS_WINDOW_TOKENS, window_tokens, a->grad_accum, 1};
        rrc_fb_result fr;
        RRC_CHECK(rrc_model_forward_backward(R->model, R->lora, &b, &fo, &fr));
        *loss += fr.loss;
    }
    rrc_optim_hparams h = {0.9, 0.999, 1e-8, a->weight_decay, a->max_grad_norm};
    rrc_optim_stats os;
    RRC_CHECK(rrc_optimizer_step(R->opt, rrc_lora_params(R->lora), rrc_lora_grads(R->lora), a->lr, &h, &os));
    *gnorm = (double)os.grad_norm;
    return RRC_OK;
}

static rrc_status bench(run *R, const args *a, const char *model_dir) {
    rrc_model_opts mo;
    rrc_model_opts_default(&mo);
    mo.dtype = a->dtype;
    mo.attn_chunk = a->attn_chunk;
    mo.ce_chunk = a->ce_chunk;
    mo.recompute = a->recompute;
    mo.max_positions = a->seq_len; /* RoPE table rows: only the positions used */
    RRC_CHECK_CTX(rrc_model_load(R->be, model_dir, &mo, &R->model), "loading %s", model_dir);
    const rrc_model_config *mc = rrc_model_get_config(R->model);
    rrc_lora_config lc;
    rrc_lora_config_default(&lc);
    lc.r = a->r;
    lc.alpha = a->alpha;
    RRC_CHECK(rrc_lora_parse_targets(a->targets, &lc.target_mask));
    lc.seed = a->seed;
    RRC_CHECK(rrc_lora_create(R->be, mc, &lc, &R->lora));
    RRC_CHECK(rrc_optimizer_create(R->be, rrc_lora_num_params(R->lora), &R->opt));
    const int64_t BT = a->micro_batch * a->seq_len, n = BT * a->grad_accum;
    R->tok = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    R->lab = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    R->pos = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    R->doc = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    const int64_t total = a->warmup + a->steps;
    R->gpu_s = (double *)calloc((size_t)total, sizeof(double));
    R->wall_s = (double *)calloc((size_t)total, sizeof(double));
    R->loss = (double *)calloc((size_t)total, sizeof(double));
    R->gnorm = (double *)calloc((size_t)total, sizeof(double));
    if (!R->tok || !R->lab || !R->pos || !R->doc || !R->gpu_s || !R->wall_s || !R->loss || !R->gnorm)
        return RRC_ERROR(RRC_ERR_OOM, "bench host buffers");
    uint64_t z = a->seed * 0x9E3779B97F4A7C15ull + 1;
    for (int64_t i = 0; i < n; i++) {
        z ^= z << 13;
        z ^= z >> 7;
        z ^= z << 17;
        R->tok[i] = (int32_t)(z % (uint64_t)mc->vocab_size);
        R->lab[i] = R->tok[i];
        R->pos[i] = (int32_t)(i % a->seq_len);
        R->doc[i] = (int32_t)(i / a->seq_len); /* one document per row */
    }
    int64_t window_tokens = 0;
    for (int64_t i = 0; i < a->grad_accum; i++) {
        rrc_batch b = {a->micro_batch, a->seq_len, R->tok + i * BT, R->lab + i * BT, R->pos + i * BT, R->doc + i * BT};
        int64_t nl = 0;
        RRC_CHECK(rrc_batch_count_labels(&b, &nl));
        window_tokens += nl;
    }
#ifndef RRC_BENCH_HOST
    RRC_CHECK(rrc_cuda_timer_create(R->be, (int)(2 * total), &R->timer));
#endif
    for (int64_t s = 0; s < total; s++) {
        const int measured = s >= a->warmup;
#ifndef RRC_BENCH_HOST
        if (s == a->warmup) { /* counters of the measured steps only; census of the first measured step */
            RRC_CHECK(rrc_sync(R->be));
            RRC_CHECK(rrc_cuda_stats_reset(R->be));
            RRC_CHECK(rrc_cuda_gemm_census_enable(R->be, 1));
        }
        RRC_CHECK(rrc_cuda_timer_mark(R->timer, (int)(2 * s)));
#endif
        const double w0 = now_s();
        RRC_CHECK_CTX(one_step(R, a, window_tokens, &R->loss[s], &R->gnorm[s]), "step %lld", (long long)(s + 1));
#ifndef RRC_BENCH_HOST
        RRC_CHECK(rrc_cuda_timer_mark(R->timer, (int)(2 * s + 1)));
#endif
        RRC_CHECK(rrc_sync(R->be));
        R->wall_s[s] = now_s() - w0;
#ifndef RRC_BENCH_HOST
        float ms = 0.0f;
        RRC_CHECK(rrc_cuda_timer_elapsed_ms(R->timer, (int)(2 * s), (int)(2 * s + 1), &ms));
        R->gpu_s[s] = (double)ms / 1000.0;
        if (measured && s == a->warmup) {
            RRC_CHECK(rrc_cuda_stats_get(R->be, &R->stats_step));
            const size_t nc = rrc_cuda_gemm_census_get(R->be, NULL, 0, NULL);
            R->census = (rrc_cuda_gemm_shape *)calloc(nc ? nc : 1, sizeof *R->census);
            if (!R->census) return RRC_ERROR(RRC_ERR_OOM, "census copy");
            R->n_census = rrc_cuda_gemm_census_get(R->be, R->census, nc, &R->census_dropped);
            RRC_CHECK(rrc_cuda_gemm_census_enable(R->be, 0));
        }
        sample_mem(R);
#else
        R->gpu_s[s] = R->wall_s[s];
#endif
        fprintf(stderr, "rrc_bench: step %lld/%lld%s loss %.6f  %.4f s\n", (long long)(s + 1), (long long)total,
                measured ? "" : " (warmup)", R->loss[s], R->gpu_s[s]);
    }
#ifndef RRC_BENCH_HOST
    RRC_CHECK(rrc_cuda_stats_get(R->be, &R->stats_all));
#endif
    R->n_rec = total;
    return RRC_OK;
}

#ifndef RRC_BENCH_HOST
static const char *combo_name(int combo, char *buf, size_t n) {
    snprintf(buf, n, "%s x %s -> %s", (combo & 4) ? "bf16" : "f32", (combo & 2) ? "bf16" : "f32", (combo & 1) ? "bf16" : "f32");
    return buf;
}
#endif

static rrc_status write_json(const run *R, const args *a, const char *model_dir, int argc, char **argv,
                             const char *started, rrc_status run_st, const char *err) {
    rrc_jw w;
    rrc_jw_init(&w, 1);
    rrc_jw_obj_begin(&w);
    rrc_jw_key(&w, "schema");
    rrc_jw_str(&w, "rrc-bench-v1");
    rrc_jw_key(&w, "tool");
    rrc_jw_str(&w, "rrc_bench");
    rrc_jw_key(&w, "status");
    rrc_jw_str(&w, run_st != RRC_OK ? "error" : R->status);
    rrc_jw_key(&w, "error");
    if (run_st != RRC_OK) rrc_jw_str(&w, err);
    else rrc_jw_null(&w);
    rrc_jw_key(&w, "measured_on");
    if (R->timing_valid && run_st == RRC_OK) rrc_jw_str(&w, R->measured_on);
    else rrc_jw_null(&w);
    rrc_jw_key(&w, "timing_source");
    rrc_jw_str(&w, R->timing_source);
    rrc_jw_key(&w, "label");
    if (a->label) rrc_jw_str(&w, a->label);
    else rrc_jw_null(&w);
    rrc_jw_key(&w, "started_utc");
    rrc_jw_str(&w, started);
    rrc_jw_key(&w, "command_line");
    rrc_jw_arr_begin(&w);
    for (int i = 0; i < argc; i++) rrc_jw_str(&w, argv[i]);
    rrc_jw_arr_end(&w);
    rrc_jw_key(&w, "versions");
    rrc_jw_obj_begin(&w);
    rrc_jw_key(&w, "git_rev");
    rrc_jw_str(&w, RRC_GIT_REV);
    rrc_jw_key(&w, "host_compiler");
    rrc_jw_str(&w, __VERSION__);
#ifndef RRC_BENCH_HOST
    rrc_cuda_build_info bi;
    rrc_cuda_build_info_get(&bi);
    char vb[64];
    snprintf(vb, sizeof vb, "%d.%d.%d", bi.nvcc_major, bi.nvcc_minor, bi.nvcc_build);
    rrc_jw_key(&w, "nvcc");
    rrc_jw_str(&w, vb);
    rrc_jw_key(&w, "cuda_runtime");
    if (R->have_info) rrc_jw_int(&w, R->info.runtime_version);
    else rrc_jw_null(&w);
    rrc_jw_key(&w, "cuda_driver");
    if (R->have_info) rrc_jw_int(&w, R->info.driver_version);
    else rrc_jw_null(&w);
    rrc_jw_key(&w, "cublaslt");
    if (R->have_info) rrc_jw_int(&w, (int64_t)R->info.cublaslt_version);
    else rrc_jw_null(&w);
    rrc_jw_key(&w, "host_shim");
    rrc_jw_bool(&w, bi.host_shim);
#endif
    rrc_jw_obj_end(&w);
    rrc_jw_key(&w, "device");
#ifndef RRC_BENCH_HOST
    if (R->have_info) {
        rrc_jw_obj_begin(&w);
        rrc_jw_key(&w, "name");
        rrc_jw_str(&w, R->info.name);
        rrc_jw_key(&w, "compute_capability");
        char cc[16];
        snprintf(cc, sizeof cc, "%d.%d", R->info.cc_major, R->info.cc_minor);
        rrc_jw_str(&w, cc);
        rrc_jw_key(&w, "sm_count");
        rrc_jw_int(&w, R->info.sm_count);
        rrc_jw_key(&w, "total_mem_bytes");
        rrc_jw_int(&w, (int64_t)R->info.total_mem);
        rrc_jw_obj_end(&w);
    } else {
        rrc_jw_null(&w);
    }
#else
    rrc_jw_null(&w);
#endif
    const rrc_model_config *mc = R->model ? rrc_model_get_config(R->model) : NULL;
    const int64_t tokens_per_step = a->micro_batch * a->seq_len * a->grad_accum;
    rrc_jw_key(&w, "config");
    rrc_jw_obj_begin(&w);
    rrc_jw_key(&w, "model");
    rrc_jw_str(&w, a->synthetic ? "synthetic tiny model (smoke test only)" : model_dir);
    rrc_jw_key(&w, "model_type");
    if (mc) rrc_jw_str(&w, rrc_family_name(mc->family));
    else rrc_jw_null(&w);
    rrc_jw_key(&w, "num_layers");
    if (mc) rrc_jw_int(&w, mc->num_layers);
    else rrc_jw_null(&w);
    rrc_jw_key(&w, "hidden_size");
    if (mc) rrc_jw_int(&w, mc->hidden_size);
    else rrc_jw_null(&w);
    rrc_jw_key(&w, "vocab_size");
    if (mc) rrc_jw_int(&w, mc->vocab_size);
    else rrc_jw_null(&w);
    rrc_jw_key(&w, "base_dtype");
    rrc_jw_str(&w, a->dtype == RRC_BF16 ? "bf16" : "f32");
    rrc_jw_key(&w, "seq_len");
    rrc_jw_int(&w, a->seq_len);
    rrc_jw_key(&w, "micro_batch");
    rrc_jw_int(&w, a->micro_batch);
    rrc_jw_key(&w, "grad_accum");
    rrc_jw_int(&w, a->grad_accum);
    rrc_jw_key(&w, "tokens_per_step");
    rrc_jw_int(&w, tokens_per_step);
    rrc_jw_key(&w, "warmup_steps");
    rrc_jw_int(&w, a->warmup);
    rrc_jw_key(&w, "measured_steps");
    rrc_jw_int(&w, a->steps);
    rrc_jw_key(&w, "data");
    rrc_jw_str(&w, "synthetic random token ids, one document per row, every position labelled, no padding");
    rrc_jw_key(&w, "lora");
    rrc_jw_obj_begin(&w);
    rrc_jw_key(&w, "r");
    rrc_jw_int(&w, a->r);
    rrc_jw_key(&w, "alpha");
    rrc_jw_double(&w, a->alpha);
    rrc_jw_key(&w, "dropout");
    rrc_jw_double(&w, 0.0);
    rrc_jw_key(&w, "target_modules");
    unsigned mask = 0;
    if (rrc_lora_parse_targets(a->targets, &mask) == RRC_OK) targets_json(&w, mask);
    else {
        rrc_clear_error();
        rrc_jw_null(&w);
    }
    rrc_jw_key(&w, "adapter_dtype");
    rrc_jw_str(&w, "f32");
    rrc_jw_key(&w, "trainable_params");
    if (R->lora) rrc_jw_int(&w, rrc_lora_num_params(R->lora));
    else rrc_jw_null(&w);
    rrc_jw_obj_end(&w);
    rrc_jw_key(&w, "recompute");
    rrc_jw_str(&w, a->recompute == RRC_RECOMPUTE_FULL ? "full" : "none");
    rrc_jw_key(&w, "gradient_checkpointing");
    rrc_jw_bool(&w, a->recompute == RRC_RECOMPUTE_FULL); /* full = save each layer's input, recompute in backward */
    rrc_jw_key(&w, "attention");
    char at[160];
    snprintf(at, sizeof at, "rrc chunked attention (%lld query rows per chunk): scores GEMM + masked softmax kernel + PV GEMM",
             (long long)a->attn_chunk);
    rrc_jw_str(&w, at);
    rrc_jw_key(&w, "ce_chunk");
    rrc_jw_int(&w, a->ce_chunk);
    rrc_jw_key(&w, "optimizer");
    rrc_jw_obj_begin(&w);
    rrc_jw_key(&w, "type");
    rrc_jw_str(&w, "AdamW");
    rrc_jw_key(&w, "impl");
    rrc_jw_str(&w, "rrc fused adamw_step kernel (torch.optim.AdamW foreach=False semantics), clip_grad_norm fused");
    rrc_jw_key(&w, "lr");
    rrc_jw_double(&w, a->lr);
    rrc_jw_key(&w, "betas");
    rrc_jw_arr_begin(&w);
    rrc_jw_double(&w, 0.9);
    rrc_jw_double(&w, 0.999);
    rrc_jw_arr_end(&w);
    rrc_jw_key(&w, "eps");
    rrc_jw_double(&w, 1e-8);
    rrc_jw_key(&w, "weight_decay");
    rrc_jw_double(&w, a->weight_decay);
    rrc_jw_key(&w, "max_grad_norm");
    rrc_jw_double(&w, a->max_grad_norm);
    rrc_jw_key(&w, "lr_schedule");
    rrc_jw_str(&w, "constant");
    rrc_jw_obj_end(&w);
    rrc_jw_obj_end(&w); /* config */
    const int ok = run_st == RRC_OK && R->n_rec == a->warmup + a->steps;
    const double *gm = R->gpu_s ? R->gpu_s + a->warmup : NULL, *wm = R->wall_s ? R->wall_s + a->warmup : NULL;
    rrc_jw_key(&w, "results");
    rrc_jw_obj_begin(&w);
    double mean = 0.0;
    if (ok)
        for (int64_t i = 0; i < a->steps; i++) mean += gm[i] / (double)a->steps;
    rrc_jw_key(&w, "tokens_per_s");
    if (ok && R->timing_valid && mean > 0) rrc_jw_double(&w, (double)tokens_per_step / mean);
    else rrc_jw_null(&w);
    jw_stats(&w, "step_time_s", gm, a->steps, ok && R->timing_valid);
    jw_stats(&w, "wall_step_time_s", wm, a->steps, ok && R->timing_valid);
    rrc_jw_key(&w, "smoke_wall_step_time_s");
    if (ok && !R->timing_valid) {
        double s = 0;
        for (int64_t i = 0; i < a->steps; i++) s += wm[i] / (double)a->steps;
        rrc_jw_double(&w, s); /* host smoke runs: the program's own wall clock, not a benchmark */
    } else {
        rrc_jw_null(&w);
    }
    rrc_jw_key(&w, "losses");
    rrc_jw_arr_begin(&w);
    for (int64_t i = 0; R->loss && i < R->n_rec; i++) rrc_jw_double(&w, R->loss[i]);
    rrc_jw_arr_end(&w);
    rrc_jw_key(&w, "grad_norms");
    rrc_jw_arr_begin(&w);
    for (int64_t i = 0; R->gnorm && i < R->n_rec; i++) rrc_jw_double(&w, R->gnorm[i]);
    rrc_jw_arr_end(&w);
    rrc_jw_key(&w, "peak_memory");
    rrc_jw_obj_begin(&w);
#ifndef RRC_BENCH_HOST
    rrc_jw_key(&w, "allocator_peak_bytes");
    rrc_jw_int(&w, (int64_t)R->stats_all.alloc_peak_bytes);
    rrc_jw_key(&w, "allocator_peak_note");
    rrc_jw_str(&w, "true peak of the bytes allocated through the rrc backend (weights, LoRA, optimizer, workspace) during "
                   "the measured steps");
    rrc_jw_key(&w, "backend_total_peak_bytes");
    rrc_jw_int(&w, (int64_t)R->stats_all.total_peak_bytes);
    rrc_jw_key(&w, "backend_total_peak_note");
    rrc_jw_str(&w, "allocator peak plus the backend's own buffers (cuBLASLt workspace, GEMM scratch, partials)");
    rrc_jw_key(&w, "device_used_max_sampled_bytes");
    if (R->have_dev_mem) rrc_jw_int(&w, (int64_t)R->dev_used_max);
    else rrc_jw_null(&w);
    rrc_jw_key(&w, "device_used_note");
    rrc_jw_str(&w, "cudaMemGetInfo total - free, sampled after every step: device-wide (CUDA context, other processes), "
                   "not a true peak");
#else
    rrc_jw_key(&w, "allocator_peak_bytes");
    rrc_jw_null(&w);
    rrc_jw_key(&w, "device_used_max_sampled_bytes");
    rrc_jw_null(&w);
#endif
    rrc_jw_key(&w, "model_workspace_bytes");
    if (R->model) rrc_jw_int(&w, (int64_t)rrc_model_workspace_bytes(R->model));
    else rrc_jw_null(&w);
    rrc_jw_obj_end(&w);
    rrc_jw_obj_end(&w); /* results */
    rrc_jw_key(&w, "gemm");
#ifndef RRC_BENCH_HOST
    if (ok) {
        const rrc_cuda_stats *s = &R->stats_step;
        rrc_jw_obj_begin(&w);
        rrc_jw_key(&w, "note");
        rrc_jw_str(&w, "one measured optimizer step: GEMM calls per dtype combination and ladder level (0 native/promoted "
                       "cuBLASLt, 1 promote-all cuBLASLt, 2 kernel fallback); flops computed from shapes (2*m*n*k*batch), "
                       "not measured");
        char nb[64];
        int64_t tot = 0, l2 = 0;
        double ft = 0, f2 = 0;
        rrc_jw_key(&w, "per_dtype_combination");
        rrc_jw_obj_begin(&w);
        for (int c = 0; c < 8; c++) {
            if (!(s->gemm_calls[c][0] + s->gemm_calls[c][1] + s->gemm_calls[c][2])) continue;
            rrc_jw_key(&w, combo_name(c, nb, sizeof nb));
            rrc_jw_obj_begin(&w);
            rrc_jw_key(&w, "calls_per_level");
            rrc_jw_arr_begin(&w);
            for (int v = 0; v < 3; v++) rrc_jw_int(&w, s->gemm_calls[c][v]);
            rrc_jw_arr_end(&w);
            rrc_jw_key(&w, "flops_per_level");
            rrc_jw_arr_begin(&w);
            for (int v = 0; v < 3; v++) rrc_jw_double(&w, s->gemm_flops[c][v]);
            rrc_jw_arr_end(&w);
            rrc_jw_obj_end(&w);
            for (int v = 0; v < 3; v++) {
                tot += s->gemm_calls[c][v];
                ft += s->gemm_flops[c][v];
            }
            l2 += s->gemm_calls[c][2];
            f2 += s->gemm_flops[c][2];
        }
        rrc_jw_obj_end(&w);
        rrc_jw_key(&w, "calls_per_step");
        rrc_jw_int(&w, tot);
        rrc_jw_key(&w, "flops_per_step_computed");
        rrc_jw_double(&w, ft);
        rrc_jw_key(&w, "kernel_fallback_calls_per_step");
        rrc_jw_int(&w, l2);
        rrc_jw_key(&w, "kernel_fallback_flop_share");
        rrc_jw_double(&w, ft > 0 ? f2 / ft : 0.0);
        rrc_jw_key(&w, "h2d_calls_per_step");
        rrc_jw_int(&w, s->h2d_calls);
        rrc_jw_key(&w, "stream_syncs_per_step");
        rrc_jw_int(&w, s->stream_syncs);
        rrc_jw_key(&w, "census");
        rrc_jw_arr_begin(&w);
        for (size_t i = 0; R->census && i < R->n_census; i++) {
            const rrc_cuda_gemm_shape *g = &R->census[i];
            rrc_jw_obj_begin(&w);
            char ob[160];
            snprintf(ob, sizeof ob, "%c%c %s x %s -> %s", g->trans_a ? 'T' : 'N', g->trans_b ? 'T' : 'N',
                     rrc_dtype_name((rrc_dtype)g->dt_a), rrc_dtype_name((rrc_dtype)g->dt_b),
                     rrc_dtype_name((rrc_dtype)g->dt_c));
            rrc_jw_key(&w, "op");
            rrc_jw_str(&w, ob);
            rrc_jw_key(&w, "m");
            rrc_jw_int(&w, g->m);
            rrc_jw_key(&w, "n");
            rrc_jw_int(&w, g->n);
            rrc_jw_key(&w, "k");
            rrc_jw_int(&w, g->k);
            rrc_jw_key(&w, "batch");
            rrc_jw_int(&w, g->batch);
            rrc_jw_key(&w, "calls");
            rrc_jw_int(&w, g->calls);
            rrc_jw_key(&w, "level");
            rrc_jw_int(&w, g->level);
            rrc_jw_key(&w, "path");
            rrc_jw_str(&w, g->path);
            rrc_jw_obj_end(&w);
        }
        rrc_jw_arr_end(&w);
        rrc_jw_key(&w, "census_dropped_calls");
        rrc_jw_int(&w, R->census_dropped);
        rrc_jw_obj_end(&w);
    } else {
        rrc_jw_null(&w);
    }
#else
    rrc_jw_null(&w);
#endif
    rrc_jw_obj_end(&w);
    char *txt = NULL;
    size_t len = 0;
    RRC_CHECK(rrc_jw_finish(&w, &txt, &len));
    rrc_status st = RRC_OK;
    if (a->out) st = rrc_write_file_atomic(a->out, txt, len);
    else if (fwrite(txt, 1, len, stdout) != len || fputc('\n', stdout) == EOF) st = RRC_ERROR(RRC_ERR_IO, "stdout");
    free(txt);
    return st;
}

int main(int argc, char **argv) {
    args a;
    const int pr = parse_args(argc, argv, &a);
    if (pr == 1) {
        usage(stdout);
        return 0;
    }
    if (pr) {
        usage(stderr);
        return 2;
    }
    char started[40] = "";
    {
        time_t t = time(NULL);
        struct tm tmv;
        if (gmtime_r(&t, &tmv)) strftime(started, sizeof started, "%Y-%m-%dT%H:%M:%SZ", &tmv);
    }
    run R;
    memset(&R, 0, sizeof R);
    rrc_status st = RRC_OK;
    char model_dir[600];
    snprintf(model_dir, sizeof model_dir, "%s", a.model ? a.model : "");
    if (a.synthetic) {
        const char *base = getenv("TMPDIR");
        snprintf(R.tmpdir, sizeof R.tmpdir, "%s/rrc_bench_XXXXXX", base && base[0] ? base : "/tmp");
        if (!mkdtemp(R.tmpdir)) {
            fprintf(stderr, "rrc_bench: cannot create a temporary directory\n");
            return 1;
        }
        snprintf(model_dir, sizeof model_dir, "%s/model", R.tmpdir);
        const int fam = !strcmp(a.synthetic, "llama") ? 0 : !strcmp(a.synthetic, "qwen2") ? 1 : 2;
        st = rrc_e2e_write_model(model_dir, fam, RRC_E2E_SIZE_QUICK, a.dtype, 99);
    }
#ifdef RRC_BENCH_HOST
    R.be = rrc_backend_cpu_f32acc();
    snprintf(R.status, sizeof R.status, "smoke (host build on the CPU backend: not a GPU benchmark)");
    snprintf(R.timing_source, sizeof R.timing_source, "host wall clock (not a GPU measurement)");
    R.timing_valid = 0;
#else
    if (st == RRC_OK) {
        rrc_cuda_options co;
        rrc_cuda_options_default(&co);
        st = rrc_cuda_backend_create_ex(a.device, &co, &R.be);
    }
    if (st == RRC_OK && rrc_cuda_device_info_get(R.be, &R.info) == RRC_OK) R.have_info = 1;
    rrc_clear_error();
    if (rrc_cuda_build_is_host_shim()) {
        snprintf(R.status, sizeof R.status, "smoke (HOST-ONLY fake CUDA runtime: no GPU, timings are host wall clock)");
        snprintf(R.timing_source, sizeof R.timing_source, "host wall clock via the fake runtime's events (not GPU time)");
        R.timing_valid = 0;
    } else {
        snprintf(R.status, sizeof R.status, a.synthetic ? "measured (synthetic tiny model: not a model benchmark)" : "measured");
        snprintf(R.timing_source, sizeof R.timing_source, "cudaEvent elapsed time on the backend stream");
        R.timing_valid = R.have_info;
        snprintf(R.measured_on, sizeof R.measured_on, "%s", R.have_info ? R.info.name : "");
    }
#endif
    if (st == RRC_OK) st = bench(&R, &a, model_dir);
    char err[RRC_ERROR_MSG_MAX] = "";
    if (st != RRC_OK) {
        snprintf(err, sizeof err, "%s", rrc_last_error());
        fprintf(stderr, "rrc_bench: %s\n", err);
    }
    rrc_status ws = write_json(&R, &a, model_dir, argc, argv, started, st, err);
    if (ws != RRC_OK) fprintf(stderr, "rrc_bench: writing the result failed: %s\n", rrc_last_error());
#ifndef RRC_BENCH_HOST
    rrc_cuda_timer_destroy(R.timer);
    free(R.census);
#endif
    rrc_optimizer_free(R.opt);
    rrc_lora_free(R.lora);
    rrc_model_free(R.model);
#ifndef RRC_BENCH_HOST
    if (R.be) rrc_cuda_backend_destroy(R.be);
#endif
    free(R.tok);
    free(R.lab);
    free(R.pos);
    free(R.doc);
    free(R.gpu_s);
    free(R.wall_s);
    free(R.loss);
    free(R.gnorm);
    if (R.tmpdir[0]) { /* the synthetic model: two files and two directories, nothing else */
        char p[700];
        snprintf(p, sizeof p, "%s/model/config.json", R.tmpdir);
        (void)unlink(p);
        snprintf(p, sizeof p, "%s/model/model.safetensors", R.tmpdir);
        (void)unlink(p);
        snprintf(p, sizeof p, "%s/model", R.tmpdir);
        (void)rmdir(p);
        (void)rmdir(R.tmpdir);
    }
    return st == RRC_OK && ws == RRC_OK ? 0 : 1;
}
