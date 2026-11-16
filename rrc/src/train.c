/* rrc/src/train.c — LR schedules, clip + AdamW, training config, rrc_state.bin, checkpoints and the
 * SFT training loop (rrc/include/rrc/train.h, rrc/docs/DESIGN.md §6 / §10). */
#include "rrc/train.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "rrc/census.h"
#include "rrc/json.h"
#include "rrc/safetensors.h"

/* ---- LR schedule ------------------------------------------------------------------------------ */

/* math.pi (the double nearest pi; M_PI is not ISO C) */
static const double k_pi = 3.141592653589793;

static const char *const k_sched_names[] = {"constant", "constant_with_warmup", "linear", "cosine"};

const char *rrc_lr_schedule_name(rrc_lr_schedule s) {
    if ((int)s < 0 || (int)s > (int)RRC_LR_COSINE) return NULL;
    return k_sched_names[s];
}

rrc_status rrc_lr_schedule_parse(const char *name, rrc_lr_schedule *out) {
    RRC_REQUIRE(name && out, RRC_ERR_INVALID_ARG, "null argument");
    for (int i = 0; i <= (int)RRC_LR_COSINE; i++)
        if (!strcmp(name, k_sched_names[i])) {
            *out = (rrc_lr_schedule)i;
            return RRC_OK;
        }
    return RRC_ERROR(RRC_ERR_UNSUPPORTED,
                     "lr_scheduler_type \"%s\" is not supported (constant, constant_with_warmup, linear, cosine)", name);
}

int64_t rrc_warmup_steps(double warmup, int64_t total) {
    if (warmup >= 1.0) return (int64_t)warmup; /* python int(): truncation */
    return (int64_t)ceil((double)total * warmup);
}

double rrc_lr_lambda(rrc_lr_schedule s, int64_t k, int64_t warmup, int64_t total) {
    if (s == RRC_LR_CONSTANT) return 1.0;
    if (k < warmup) {
        /* constant_with_warmup divides by max(1.0, W), the others by max(1, W): equal values */
        return (double)k / (double)(warmup > 1 ? warmup : 1);
    }
    const double rest = (double)(total - warmup > 1 ? total - warmup : 1);
    switch (s) {
    case RRC_LR_CONSTANT_WITH_WARMUP: return 1.0;
    case RRC_LR_LINEAR: {
        const double f = (double)(total - k) / rest;
        return f > 0.0 ? f : 0.0;
    }
    case RRC_LR_COSINE: {
        const double progress = (double)(k - warmup) / rest;
        const double num_cycles = 0.5;
        double factor = 0.5 * (1.0 + cos(k_pi * num_cycles * 2.0 * progress));
        factor = factor * (1.0 - 0.0) + 0.0; /* min_lr_rate = 0 (transformers' formula, exact) */
        return factor > 0.0 ? factor : 0.0;
    }
    default: return 1.0;
    }
}

float rrc_clip_coef(float total_norm, double max_norm) {
    if (!(max_norm > 0.0)) return 1.0f;
    const float denom = total_norm + 1e-6f; /* f32, as torch's tensor + python scalar */
    const float coef = (1.0f / denom) * (float)max_norm;
    return coef < 1.0f ? coef : 1.0f; /* torch.clamp(max=1.0) */
}

/* ---- optimizer -------------------------------------------------------------------------------- */

struct rrc_optimizer {
    const rrc_backend *be;
    int64_t n, step;
    rrc_buf m, v, norm;
};

rrc_status rrc_optimizer_create(const rrc_backend *be, int64_t n, rrc_optimizer **out) {
    RRC_REQUIRE(be && out && n >= 0, RRC_ERR_INVALID_ARG, "bad argument");
    *out = NULL;
    rrc_optimizer *o = (rrc_optimizer *)calloc(1, sizeof *o);
    if (!o) return RRC_ERROR(RRC_ERR_OOM, "rrc_optimizer");
    o->be = be;
    o->n = n;
    o->m = o->v = o->norm = rrc_buf_none();
    rrc_status st = RRC_OK;
    RRC_CHECK_GOTO(st, rrc_buf_alloc(be, RRC_F32, n, &o->m), fail);
    RRC_CHECK_GOTO(st, rrc_buf_alloc(be, RRC_F32, n, &o->v), fail);
    RRC_CHECK_GOTO(st, rrc_buf_alloc(be, RRC_F32, 1, &o->norm), fail);
    *out = o;
    return RRC_OK;
fail:
    rrc_optimizer_free(o);
    return st;
}

void rrc_optimizer_free(rrc_optimizer *o) {
    if (!o) return;
    rrc_buf_free(&o->m);
    rrc_buf_free(&o->v);
    rrc_buf_free(&o->norm);
    free(o);
}

int64_t rrc_optimizer_step_count(const rrc_optimizer *o) { return o ? o->step : 0; }
rrc_buf rrc_optimizer_m(const rrc_optimizer *o) { return o ? rrc_view(o->m, 0, o->n) : rrc_buf_none(); }
rrc_buf rrc_optimizer_v(const rrc_optimizer *o) { return o ? rrc_view(o->v, 0, o->n) : rrc_buf_none(); }

rrc_status rrc_optimizer_restore(rrc_optimizer *o, const float *m, const float *v, int64_t step) {
    RRC_REQUIRE(o && (m || o->n == 0) && (v || o->n == 0) && step >= 0, RRC_ERR_INVALID_ARG, "bad argument");
    if (o->n) {
        RRC_CHECK(rrc_buf_upload(o->m, m, o->n));
        RRC_CHECK(rrc_buf_upload(o->v, v, o->n));
    }
    o->step = step;
    return RRC_OK;
}

rrc_status rrc_optimizer_step(rrc_optimizer *o, rrc_buf params, rrc_buf grads, double lr, const rrc_optim_hparams *h,
                              rrc_optim_stats *stats) {
    RRC_REQUIRE(o && h && stats, RRC_ERR_INVALID_ARG, "null argument");
    RRC_REQUIRE(params.numel == o->n && grads.numel == o->n, RRC_ERR_SHAPE,
                "optimizer has %lld parameters, got params %lld / grads %lld", (long long)o->n, (long long)params.numel,
                (long long)grads.numel);
    float ss = 0.0f;
    RRC_CHECK(rrc_sum_squares(o->be, o->n, grads, o->norm, 0));
    RRC_CHECK(rrc_buf_download(o->norm, &ss, 1));
    const float norm = sqrtf(ss);
    RRC_REQUIRE(isfinite(norm), RRC_ERR_RANGE, "non-finite gradient norm (%g) at optimizer step %lld", (double)norm,
                (long long)(o->step + 1));
    stats->grad_norm = norm;
    stats->clip_coef = rrc_clip_coef(norm, h->max_grad_norm);
    rrc_adamw_hparams ah;
    ah.lr = lr;
    ah.beta1 = h->beta1;
    ah.beta2 = h->beta2;
    ah.eps = h->eps;
    ah.weight_decay = h->weight_decay;
    ah.step = o->step + 1;
    ah.grad_scale = stats->clip_coef;
    if (o->n) RRC_CHECK(rrc_adamw_step(o->be, o->n, params, grads, o->m, o->v, &ah));
    o->step++;
    return RRC_OK;
}

/* ---- FNV ------------------------------------------------------------------------------------------ */

#define FNV_INIT 0xcbf29ce484222325ull
static uint64_t fnv_update(uint64_t h, const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}
uint64_t rrc_fnv1a64(const void *data, size_t n) { return fnv_update(FNV_INIT, data, n); }

/* ---- training config ------------------------------------------------------------------------------ */

void rrc_train_config_default(rrc_train_config *c) {
    memset(c, 0, sizeof *c);
    c->dtype = RRC_BF16;
    c->micro_batch_size = 1;
    c->grad_accum_steps = 1;
    c->learning_rate = 5e-5;
    c->lr_scheduler_type = RRC_LR_LINEAR;
    c->optim.beta1 = 0.9;
    c->optim.beta2 = 0.999;
    c->optim.eps = 1e-8;
    c->optim.weight_decay = 0.0;
    c->optim.max_grad_norm = 1.0;
    c->loss_reduction = RRC_LOSS_WINDOW_TOKENS;
    c->seed = 42;
    c->shuffle = 1;
    c->truncation = RRC_TRUNC_TRUNCATE;
    c->pad_token_id = 0;
    rrc_lora_config_default(&c->lora);
    rrc_model_opts mo;
    rrc_model_opts_default(&mo);
    c->attn_chunk = mo.attn_chunk;
    c->ce_chunk = mo.ce_chunk;
    c->max_positions = 0;
    c->recompute = RRC_RECOMPUTE_FULL;
    c->prefetch = 1;
    c->memory_budget_bytes = -1;
}

static rrc_status get_str(const rrc_json *v, const char *key, int nullable, char *dst, size_t cap) {
    if (nullable && v->type == RRC_JSON_NULL) {
        dst[0] = '\0';
        return RRC_OK;
    }
    RRC_REQUIRE(v->type == RRC_JSON_STRING, RRC_ERR_PARSE, "config: \"%s\" must be a string%s (got %s)", key,
                nullable ? " or null" : "", rrc_json_type_name(v->type));
    RRC_REQUIRE(v->u.s.len < cap && strlen(v->u.s.ptr) == v->u.s.len, RRC_ERR_PARSE,
                "config: \"%s\" is too long or contains a NUL character", key);
    memcpy(dst, v->u.s.ptr, v->u.s.len + 1);
    return RRC_OK;
}

static rrc_status get_int(const rrc_json *v, const char *key, int64_t lo, int64_t hi, int64_t *out) {
    RRC_REQUIRE(v->type == RRC_JSON_INT, RRC_ERR_PARSE, "config: \"%s\" must be an integer (got %s)", key,
                rrc_json_type_name(v->type));
    RRC_REQUIRE(v->u.i >= lo && v->u.i <= hi, RRC_ERR_INVALID_ARG, "config: \"%s\" = %lld outside [%lld, %lld]", key,
                (long long)v->u.i, (long long)lo, (long long)hi);
    *out = v->u.i;
    return RRC_OK;
}

static rrc_status get_num(const rrc_json *v, const char *key, double *out) {
    double d;
    RRC_CHECK(rrc_json_as_number(v, key, &d));
    RRC_REQUIRE(isfinite(d), RRC_ERR_INVALID_ARG, "config: \"%s\" must be finite", key);
    *out = d;
    return RRC_OK;
}

static rrc_status get_bool(const rrc_json *v, const char *key, int *out) { return rrc_json_as_bool(v, key, out); }

static rrc_status get_enum(const rrc_json *v, const char *key, const char *const *names, int n, int *out) {
    const char *s;
    RRC_CHECK(rrc_json_as_string(v, key, &s));
    for (int i = 0; i < n; i++)
        if (!strcmp(s, names[i])) {
            *out = i;
            return RRC_OK;
        }
    char allowed[256] = "";
    size_t len = 0;
    for (int i = 0; i < n && len < sizeof allowed; i++)
        len += (size_t)snprintf(allowed + len, sizeof allowed - len, "%s%s", i ? ", " : "", names[i]);
    return RRC_ERROR(RRC_ERR_UNSUPPORTED, "config: \"%s\" = \"%s\" is not one of: %s", key, s, allowed);
}

static const char *const k_dtype_names[] = {"f32", "bf16"};
static const char *const k_backend_names[] = {"cpu", "cpu-f32acc", "cuda"};
static const char *const k_reduction_names[] = {"window_tokens", "microbatch_mean"};
static const char *const k_trunc_names[] = {"truncate", "drop"};
static const char *const k_recompute_names[] = {"full", "none"};

static rrc_status parse_lora(const rrc_json *obj, rrc_lora_config *lc) {
    RRC_REQUIRE(obj->type == RRC_JSON_OBJECT, RRC_ERR_PARSE, "config: \"lora\" must be an object");
    for (size_t i = 0; i < rrc_json_obj_count(obj); i++) {
        const char *k = rrc_json_obj_key(obj, i);
        const rrc_json *v = rrc_json_obj_val(obj, i);
        int64_t iv;
        int b;
        if (!strcmp(k, "r")) {
            RRC_CHECK(get_int(v, "lora.r", 1, 65536, &iv));
            lc->r = iv;
        } else if (!strcmp(k, "alpha")) {
            RRC_CHECK(get_num(v, "lora.alpha", &lc->alpha));
            RRC_REQUIRE(lc->alpha > 0.0, RRC_ERR_INVALID_ARG, "config: \"lora.alpha\" must be > 0");
        } else if (!strcmp(k, "use_rslora")) {
            RRC_CHECK(get_bool(v, "lora.use_rslora", &b));
            lc->use_rslora = b;
        } else if (!strcmp(k, "dropout")) {
            RRC_CHECK(get_num(v, "lora.dropout", &lc->dropout));
            RRC_REQUIRE(lc->dropout == 0.0, RRC_ERR_UNSUPPORTED,
                        "config: lora.dropout = %g is not supported (phase 1 implements lora_dropout = 0 only)",
                        lc->dropout);
        } else if (!strcmp(k, "init_seed")) {
            RRC_CHECK(get_int(v, "lora.init_seed", 0, INT64_MAX, &iv));
            lc->seed = (uint64_t)iv;
        } else if (!strcmp(k, "target_modules")) {
            if (v->type == RRC_JSON_STRING) {
                RRC_CHECK(rrc_lora_parse_targets(v->u.s.ptr, &lc->target_mask));
            } else {
                RRC_REQUIRE(v->type == RRC_JSON_ARRAY && rrc_json_arr_count(v) > 0, RRC_ERR_PARSE,
                            "config: \"lora.target_modules\" must be a non-empty list of module names or "
                            "\"all-linear\"");
                unsigned mask = 0;
                for (size_t j = 0; j < rrc_json_arr_count(v); j++) {
                    const char *s = rrc_json_str(rrc_json_arr_get(v, j));
                    RRC_REQUIRE(s && !strchr(s, ','), RRC_ERR_PARSE,
                                "config: \"lora.target_modules\" entries must be module names");
                    unsigned one;
                    RRC_CHECK(rrc_lora_parse_targets(s, &one));
                    RRC_REQUIRE(!(mask & one), RRC_ERR_INVALID_ARG, "config: lora.target_modules lists \"%s\" twice",
                                s);
                    mask |= one;
                }
                lc->target_mask = mask;
            }
        } else if (!strcmp(k, "layers_to_transform")) {
            lc->n_layers = 0;
            if (v->type == RRC_JSON_NULL) continue;
            RRC_REQUIRE(v->type == RRC_JSON_ARRAY && rrc_json_arr_count(v) > 0 &&
                            rrc_json_arr_count(v) <= RRC_LORA_MAX_LAYERS,
                        RRC_ERR_PARSE, "config: \"lora.layers_to_transform\" must be null or a non-empty list of layer "
                                       "indices");
            for (size_t j = 0; j < rrc_json_arr_count(v); j++) {
                RRC_CHECK(get_int(rrc_json_arr_get(v, j), "lora.layers_to_transform[]", 0, RRC_LORA_MAX_LAYERS - 1,
                                  &iv));
                lc->layers[lc->n_layers++] = iv;
            }
        } else {
            return RRC_ERROR(RRC_ERR_PARSE, "config: unknown key \"lora.%s\" (see rrc/docs/README.md)", k);
        }
    }
    return RRC_OK;
}

rrc_status rrc_train_config_parse(const char *text, size_t len, rrc_train_config *c) {
    RRC_REQUIRE(text && c, RRC_ERR_INVALID_ARG, "null argument");
    rrc_train_config_default(c);
    rrc_json_doc *doc = NULL;
    RRC_CHECK_CTX(rrc_json_parse(text, len, &doc), "training config");
    const rrc_json *root = rrc_json_root(doc);
    rrc_status st = RRC_OK;
    int have_model = 0, have_data = 0, have_out = 0, have_seq = 0, have_steps = 0;
    if (root->type != RRC_JSON_OBJECT) {
        st = RRC_ERROR(RRC_ERR_PARSE, "training config must be a JSON object");
        goto done;
    }
    for (size_t i = 0; i < rrc_json_obj_count(root); i++) {
        const char *k = rrc_json_obj_key(root, i);
        const rrc_json *v = rrc_json_obj_val(root, i);
        int64_t iv;
        int e;
#define S(x) RRC_CHECK_GOTO(st, (x), done)
        if (!strcmp(k, "model_dir")) {
            S(get_str(v, k, 0, c->model_dir, sizeof c->model_dir));
            have_model = 1;
        } else if (!strcmp(k, "data_path")) {
            S(get_str(v, k, 0, c->data_path, sizeof c->data_path));
            have_data = 1;
        } else if (!strcmp(k, "output_dir")) {
            S(get_str(v, k, 0, c->output_dir, sizeof c->output_dir));
            have_out = 1;
        } else if (!strcmp(k, "init_adapter")) {
            S(get_str(v, k, 1, c->init_adapter, sizeof c->init_adapter));
        } else if (!strcmp(k, "resume_from")) {
            S(get_str(v, k, 1, c->resume_from, sizeof c->resume_from));
        } else if (!strcmp(k, "overwrite_output_dir")) {
            S(get_bool(v, k, &c->overwrite_output_dir));
        } else if (!strcmp(k, "dtype")) {
            S(get_enum(v, k, k_dtype_names, 2, &e));
            c->dtype = e ? RRC_BF16 : RRC_F32;
        } else if (!strcmp(k, "backend")) {
            S(get_enum(v, k, k_backend_names, 3, &e));
            c->backend = (rrc_train_backend)e;
        } else if (!strcmp(k, "seq_len")) {
            S(get_int(v, k, 1, INT32_MAX, &c->seq_len));
            have_seq = 1;
        } else if (!strcmp(k, "micro_batch_size")) {
            S(get_int(v, k, 1, INT32_MAX, &c->micro_batch_size));
        } else if (!strcmp(k, "grad_accum_steps")) {
            S(get_int(v, k, 1, 65536, &c->grad_accum_steps));
        } else if (!strcmp(k, "max_steps")) {
            S(get_int(v, k, 1, INT32_MAX, &c->max_steps));
            have_steps = 1;
        } else if (!strcmp(k, "learning_rate")) {
            S(get_num(v, k, &c->learning_rate));
        } else if (!strcmp(k, "lr_scheduler_type")) {
            S(get_enum(v, k, k_sched_names, 4, &e));
            c->lr_scheduler_type = (rrc_lr_schedule)e;
        } else if (!strcmp(k, "warmup_steps")) {
            S(get_num(v, k, &c->warmup_steps));
        } else if (!strcmp(k, "adam_beta1")) {
            S(get_num(v, k, &c->optim.beta1));
        } else if (!strcmp(k, "adam_beta2")) {
            S(get_num(v, k, &c->optim.beta2));
        } else if (!strcmp(k, "adam_epsilon")) {
            S(get_num(v, k, &c->optim.eps));
        } else if (!strcmp(k, "weight_decay")) {
            S(get_num(v, k, &c->optim.weight_decay));
        } else if (!strcmp(k, "max_grad_norm")) {
            S(get_num(v, k, &c->optim.max_grad_norm));
        } else if (!strcmp(k, "loss_reduction")) {
            S(get_enum(v, k, k_reduction_names, 2, &e));
            c->loss_reduction = (rrc_loss_reduction)e;
        } else if (!strcmp(k, "seed")) {
            S(get_int(v, k, 0, INT64_MAX, &iv));
            c->seed = (uint64_t)iv;
        } else if (!strcmp(k, "shuffle")) {
            S(get_bool(v, k, &c->shuffle));
        } else if (!strcmp(k, "truncation")) {
            S(get_enum(v, k, k_trunc_names, 2, &e));
            c->truncation = (rrc_truncation)e;
        } else if (!strcmp(k, "pad_token_id")) {
            S(get_int(v, k, 0, INT32_MAX, &iv));
            c->pad_token_id = (int32_t)iv;
        } else if (!strcmp(k, "lora")) {
            S(parse_lora(v, &c->lora));
        } else if (!strcmp(k, "attn_chunk")) {
            S(get_int(v, k, 1, INT32_MAX, &c->attn_chunk));
        } else if (!strcmp(k, "ce_chunk")) {
            S(get_int(v, k, 1, INT32_MAX, &c->ce_chunk));
        } else if (!strcmp(k, "max_positions")) {
            S(get_int(v, k, 0, INT32_MAX, &c->max_positions));
        } else if (!strcmp(k, "recompute")) {
            S(get_enum(v, k, k_recompute_names, 2, &e));
            c->recompute = (rrc_recompute)e;
        } else if (!strcmp(k, "prefetch")) {
            S(get_bool(v, k, &c->prefetch));
        } else if (!strcmp(k, "save_steps")) {
            S(get_int(v, k, 0, INT32_MAX, &c->save_steps));
        } else if (!strcmp(k, "stop_after_step")) {
            S(get_int(v, k, 0, INT32_MAX, &c->stop_after_step));
        } else if (!strcmp(k, "dump_batches")) {
            S(get_bool(v, k, &c->dump_batches));
        } else if (!strcmp(k, "memory_budget_bytes")) {
            if (v->type == RRC_JSON_NULL) c->memory_budget_bytes = -1;
            else S(get_int(v, k, 0, INT64_MAX, &c->memory_budget_bytes));
        } else {
            st = RRC_ERROR(RRC_ERR_PARSE, "config: unknown key \"%s\" (every field is listed in rrc/docs/README.md)", k);
            goto done;
        }
#undef S
    }
    if (!have_model || !have_data || !have_out || !have_seq || !have_steps) {
        st = RRC_ERROR(RRC_ERR_PARSE, "config: missing required key \"%s\"",
                       !have_model ? "model_dir"
                       : !have_data ? "data_path"
                       : !have_out  ? "output_dir"
                       : !have_seq  ? "seq_len"
                                    : "max_steps");
        goto done;
    }
#define RANGE(cond, ...)                                                                                               \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            st = RRC_ERROR(RRC_ERR_INVALID_ARG, __VA_ARGS__);                                                          \
            goto done;                                                                                                 \
        }                                                                                                              \
    } while (0)
    RANGE(c->model_dir[0] && c->data_path[0] && c->output_dir[0], "config: model_dir, data_path, output_dir must be "
                                                                  "non-empty");
    RANGE(c->micro_batch_size <= INT32_MAX / c->seq_len, "config: micro_batch_size * seq_len must be < 2^31");
    RANGE(c->learning_rate >= 0.0, "config: learning_rate must be >= 0");
    RANGE(c->warmup_steps >= 0.0 && c->warmup_steps <= (double)INT32_MAX,
          "config: warmup_steps must be in [0, 2^31) (>= 1: steps, < 1: ratio of max_steps)");
    RANGE(c->optim.beta1 >= 0.0 && c->optim.beta1 < 1.0 && c->optim.beta2 >= 0.0 && c->optim.beta2 < 1.0,
          "config: adam_beta1 / adam_beta2 must be in [0, 1)");
    RANGE(c->optim.eps >= 0.0 && c->optim.weight_decay >= 0.0 && c->optim.max_grad_norm >= 0.0,
          "config: adam_epsilon, weight_decay and max_grad_norm must be >= 0");
    RANGE(c->stop_after_step <= c->max_steps, "config: stop_after_step = %lld > max_steps = %lld",
          (long long)c->stop_after_step, (long long)c->max_steps);
    RANGE(!(c->resume_from[0] && c->init_adapter[0]),
          "config: init_adapter and resume_from are exclusive (a checkpoint carries its adapter)");
#undef RANGE
done:
    rrc_json_doc_free(doc);
    return st;
}

rrc_status rrc_train_config_load(const char *path, rrc_train_config *out) {
    RRC_REQUIRE(path && out, RRC_ERR_INVALID_ARG, "null argument");
    FILE *f = fopen(path, "rb");
    if (!f) return RRC_ERROR(errno == ENOENT ? RRC_ERR_NOT_FOUND : RRC_ERR_IO, "cannot open %s: %s", path,
                             strerror(errno));
    char *buf = NULL;
    size_t len = 0, cap = 0;
    rrc_status st = RRC_OK;
    for (;;) {
        if (len + 4096 > cap) {
            if (cap > (16u << 20)) {
                st = RRC_ERROR(RRC_ERR_PARSE, "%s: larger than 16 MiB", path);
                break;
            }
            size_t nc = cap ? 2 * cap : 8192;
            char *nb = (char *)realloc(buf, nc);
            if (!nb) {
                st = RRC_ERROR(RRC_ERR_OOM, "reading %s", path);
                break;
            }
            buf = nb;
            cap = nc;
        }
        size_t r = fread(buf + len, 1, cap - len, f);
        len += r;
        if (r == 0) {
            if (ferror(f)) st = RRC_ERROR(RRC_ERR_IO, "read %s failed", path);
            break;
        }
    }
    fclose(f);
    if (st == RRC_OK) st = rrc_train_config_parse(buf ? buf : "", len, out);
    free(buf);
    if (st != RRC_OK) return rrc_error_context(st, __FILE__, __LINE__, "%s", path);
    return RRC_OK;
}

int64_t rrc_train_config_warmup(const rrc_train_config *c) {
    return rrc_warmup_steps(c->warmup_steps, c->max_steps);
}

static void jw_path(rrc_jw *w, const char *key, const char *s) {
    rrc_jw_key(w, key);
    if (s[0]) rrc_jw_str(w, s);
    else rrc_jw_null(w);
}

static void jw_targets(rrc_jw *w, unsigned mask) {
    rrc_jw_arr_begin(w);
    for (int m = 0; m < RRC_LORA_N_MODULES; m++)
        if (mask & (1u << m)) rrc_jw_str(w, rrc_lora_module_name(m));
    rrc_jw_arr_end(w);
}

static void jw_layers(rrc_jw *w, const rrc_lora_config *lc) {
    if (lc->n_layers == 0) {
        rrc_jw_null(w);
        return;
    }
    rrc_jw_arr_begin(w);
    for (int64_t i = 0; i < lc->n_layers; i++) rrc_jw_int(w, lc->layers[i]);
    rrc_jw_arr_end(w);
}

rrc_status rrc_train_config_to_json(const rrc_train_config *c, char **out) {
    RRC_REQUIRE(c && out, RRC_ERR_INVALID_ARG, "null argument");
    rrc_jw w;
    rrc_jw_init(&w, 1);
    rrc_jw_obj_begin(&w);
    jw_path(&w, "model_dir", c->model_dir);
    jw_path(&w, "data_path", c->data_path);
    jw_path(&w, "output_dir", c->output_dir);
    jw_path(&w, "init_adapter", c->init_adapter);
    jw_path(&w, "resume_from", c->resume_from);
    rrc_jw_key(&w, "overwrite_output_dir");
    rrc_jw_bool(&w, c->overwrite_output_dir);
    rrc_jw_key(&w, "dtype");
    rrc_jw_str(&w, c->dtype == RRC_F32 ? "f32" : "bf16");
    rrc_jw_key(&w, "backend");
    rrc_jw_str(&w, k_backend_names[c->backend]);
    rrc_jw_key(&w, "seq_len"); /* required: null until set */
    if (c->seq_len) rrc_jw_int(&w, c->seq_len);
    else rrc_jw_null(&w);
    rrc_jw_key(&w, "micro_batch_size");
    rrc_jw_int(&w, c->micro_batch_size);
    rrc_jw_key(&w, "grad_accum_steps");
    rrc_jw_int(&w, c->grad_accum_steps);
    rrc_jw_key(&w, "max_steps");
    if (c->max_steps) rrc_jw_int(&w, c->max_steps);
    else rrc_jw_null(&w);
    rrc_jw_key(&w, "learning_rate");
    rrc_jw_double(&w, c->learning_rate);
    rrc_jw_key(&w, "lr_scheduler_type");
    rrc_jw_str(&w, rrc_lr_schedule_name(c->lr_scheduler_type));
    rrc_jw_key(&w, "warmup_steps");
    if (c->warmup_steps == floor(c->warmup_steps)) rrc_jw_int(&w, (int64_t)c->warmup_steps);
    else rrc_jw_double(&w, c->warmup_steps);
    rrc_jw_key(&w, "adam_beta1");
    rrc_jw_double(&w, c->optim.beta1);
    rrc_jw_key(&w, "adam_beta2");
    rrc_jw_double(&w, c->optim.beta2);
    rrc_jw_key(&w, "adam_epsilon");
    rrc_jw_double(&w, c->optim.eps);
    rrc_jw_key(&w, "weight_decay");
    rrc_jw_double(&w, c->optim.weight_decay);
    rrc_jw_key(&w, "max_grad_norm");
    rrc_jw_double(&w, c->optim.max_grad_norm);
    rrc_jw_key(&w, "loss_reduction");
    rrc_jw_str(&w, k_reduction_names[c->loss_reduction == RRC_LOSS_MICROBATCH_MEAN]);
    rrc_jw_key(&w, "seed");
    rrc_jw_int(&w, (int64_t)c->seed);
    rrc_jw_key(&w, "shuffle");
    rrc_jw_bool(&w, c->shuffle);
    rrc_jw_key(&w, "truncation");
    rrc_jw_str(&w, k_trunc_names[c->truncation == RRC_TRUNC_DROP]);
    rrc_jw_key(&w, "pad_token_id");
    rrc_jw_int(&w, c->pad_token_id);
    rrc_jw_key(&w, "lora");
    rrc_jw_obj_begin(&w);
    rrc_jw_key(&w, "r");
    rrc_jw_int(&w, c->lora.r);
    rrc_jw_key(&w, "alpha");
    rrc_jw_double(&w, c->lora.alpha);
    rrc_jw_key(&w, "use_rslora");
    rrc_jw_bool(&w, c->lora.use_rslora);
    rrc_jw_key(&w, "dropout");
    rrc_jw_double(&w, c->lora.dropout);
    rrc_jw_key(&w, "target_modules");
    jw_targets(&w, c->lora.target_mask);
    rrc_jw_key(&w, "layers_to_transform");
    jw_layers(&w, &c->lora);
    rrc_jw_key(&w, "init_seed");
    rrc_jw_int(&w, (int64_t)c->lora.seed);
    rrc_jw_obj_end(&w);
    rrc_jw_key(&w, "attn_chunk");
    rrc_jw_int(&w, c->attn_chunk);
    rrc_jw_key(&w, "ce_chunk");
    rrc_jw_int(&w, c->ce_chunk);
    rrc_jw_key(&w, "max_positions");
    rrc_jw_int(&w, c->max_positions);
    rrc_jw_key(&w, "recompute");
    rrc_jw_str(&w, k_recompute_names[c->recompute == RRC_RECOMPUTE_NONE]);
    rrc_jw_key(&w, "prefetch");
    rrc_jw_bool(&w, c->prefetch);
    rrc_jw_key(&w, "save_steps");
    rrc_jw_int(&w, c->save_steps);
    rrc_jw_key(&w, "stop_after_step");
    rrc_jw_int(&w, c->stop_after_step);
    rrc_jw_key(&w, "dump_batches");
    rrc_jw_bool(&w, c->dump_batches);
    rrc_jw_key(&w, "memory_budget_bytes");
    if (c->memory_budget_bytes >= 0) rrc_jw_int(&w, c->memory_budget_bytes);
    else rrc_jw_null(&w);
    rrc_jw_obj_end(&w);
    size_t n;
    return rrc_jw_finish(&w, out, &n);
}

/* growable text buffer */
typedef struct sbuf {
    char *p;
    size_t len, cap;
    int oom;
} sbuf;

static void sb_printf(sbuf *b, const char *fmt, ...) RRC_PRINTF(2, 3);
static void sb_printf(sbuf *b, const char *fmt, ...) {
    if (b->oom) return;
    for (;;) {
        va_list ap;
        va_start(ap, fmt);
        int n = vsnprintf(b->p ? b->p + b->len : NULL, b->p ? b->cap - b->len : 0, fmt, ap);
        va_end(ap);
        if (n < 0) {
            b->oom = 1;
            return;
        }
        if (b->p && (size_t)n < b->cap - b->len) {
            b->len += (size_t)n;
            return;
        }
        size_t nc = (b->cap ? b->cap * 2 : 1024) + (size_t)n;
        char *np = (char *)realloc(b->p, nc);
        if (!np) {
            b->oom = 1;
            return;
        }
        b->p = np;
        b->cap = nc;
    }
}

rrc_status rrc_train_config_fingerprint(const rrc_train_config *c, char **out) {
    RRC_REQUIRE(c && out, RRC_ERR_INVALID_ARG, "null argument");
    sbuf b = {NULL, 0, 0, 0};
    sb_printf(&b, "dtype=%s\n", c->dtype == RRC_F32 ? "f32" : "bf16");
    sb_printf(&b, "backend=%s\n", k_backend_names[c->backend]);
    sb_printf(&b, "seq_len=%lld\nmicro_batch_size=%lld\ngrad_accum_steps=%lld\nmax_steps=%lld\n",
              (long long)c->seq_len, (long long)c->micro_batch_size, (long long)c->grad_accum_steps,
              (long long)c->max_steps);
    sb_printf(&b, "learning_rate=%.17g\nlr_scheduler_type=%s\nwarmup=%lld\n", c->learning_rate,
              rrc_lr_schedule_name(c->lr_scheduler_type), (long long)rrc_train_config_warmup(c));
    sb_printf(&b, "adam_beta1=%.17g\nadam_beta2=%.17g\nadam_epsilon=%.17g\nweight_decay=%.17g\nmax_grad_norm=%.17g\n",
              c->optim.beta1, c->optim.beta2, c->optim.eps, c->optim.weight_decay, c->optim.max_grad_norm);
    sb_printf(&b, "loss_reduction=%s\nseed=%llu\nshuffle=%d\ntruncation=%s\npad_token_id=%d\n",
              k_reduction_names[c->loss_reduction == RRC_LOSS_MICROBATCH_MEAN], (unsigned long long)c->seed,
              c->shuffle, k_trunc_names[c->truncation == RRC_TRUNC_DROP], (int)c->pad_token_id);
    sb_printf(&b, "lora.r=%lld\nlora.alpha=%.17g\nlora.use_rslora=%d\nlora.target_mask=%u\nlora.layers=",
              (long long)c->lora.r, c->lora.alpha, c->lora.use_rslora, c->lora.target_mask);
    for (int64_t i = 0; i < c->lora.n_layers; i++) sb_printf(&b, "%s%lld", i ? "," : "", (long long)c->lora.layers[i]);
    sb_printf(&b, "\nattn_chunk=%lld\nce_chunk=%lld\nmax_positions=%lld\n", (long long)c->attn_chunk,
              (long long)c->ce_chunk, (long long)c->max_positions);
    if (b.oom) {
        free(b.p);
        return RRC_ERROR(RRC_ERR_OOM, "config fingerprint");
    }
    *out = b.p;
    return RRC_OK;
}

/* ---- file helpers ------------------------------------------------------------------------------- */

static rrc_status fsync_path(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return RRC_ERROR(RRC_ERR_IO, "open %s for fsync: %s", path, strerror(errno));
    int r = fsync(fd);
    int e = errno;
    close(fd);
    if (r != 0 && e != EINVAL) return RRC_ERROR(RRC_ERR_IO, "fsync %s: %s", path, strerror(e));
    return RRC_OK;
}

static int path_exists(const char *p) {
    struct stat sb;
    return stat(p, &sb) == 0;
}
static int is_dir(const char *p) {
    struct stat sb;
    return stat(p, &sb) == 0 && S_ISDIR(sb.st_mode);
}

static rrc_status mkdir_p(const char *path) {
    char tmp[RRC_PATH_MAX];
    RRC_REQUIRE(strlen(path) < sizeof tmp, RRC_ERR_INVALID_ARG, "path too long: %s", path);
    snprintf(tmp, sizeof tmp, "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0755) != 0 && errno != EEXIST)
            return RRC_ERROR(RRC_ERR_IO, "mkdir %s: %s", tmp, strerror(errno));
        *p = '/';
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return RRC_ERROR(RRC_ERR_IO, "mkdir %s: %s", tmp, strerror(errno));
    RRC_REQUIRE(is_dir(tmp), RRC_ERR_IO, "%s exists and is not a directory", tmp);
    return RRC_OK;
}

static rrc_status join(char *dst, size_t cap, const char *a, const char *b) {
    int n = snprintf(dst, cap, "%s/%s", a, b);
    RRC_REQUIRE(n > 0 && (size_t)n < cap, RRC_ERR_INVALID_ARG, "path too long: %s/%s", a, b);
    return RRC_OK;
}

/* Write a file from pieces atomically: path.tmp, fsync, rename. */
static rrc_status write_atomic_fsync(const char *path, const void *data, size_t len) {
    char tmp[RRC_PATH_MAX + 8];
    RRC_REQUIRE(strlen(path) < RRC_PATH_MAX, RRC_ERR_INVALID_ARG, "path too long");
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return RRC_ERROR(RRC_ERR_IO, "cannot create %s: %s", tmp, strerror(errno));
    int ok = (len == 0 || fwrite(data, 1, len, f) == len) && fflush(f) == 0 && fsync(fileno(f)) == 0;
    ok = (fclose(f) == 0) && ok;
    if (!ok) {
        remove(tmp);
        return RRC_ERROR(RRC_ERR_IO, "write to %s failed", tmp);
    }
    if (rename(tmp, path) != 0) {
        rrc_status st = RRC_ERROR(RRC_ERR_IO, "rename %s -> %s: %s", tmp, path, strerror(errno));
        remove(tmp);
        return st;
    }
    return RRC_OK;
}

/* ---- rrc_state.bin -------------------------------------------------------------------------------- */

#define STATE_MAGIC "RRCSTAT1"
#define STATE_FIXED 96

static void put_u64(unsigned char *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (unsigned char)(v >> (8 * i));
}
static uint64_t get_u64(const unsigned char *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

rrc_status rrc_train_state_write(const char *path, const rrc_train_state *s) {
    RRC_REQUIRE(path && s && s->config_text && s->n_params >= 0 && (s->n_params == 0 || (s->m && s->v)),
                RRC_ERR_INVALID_ARG, "bad argument");
    const uint16_t one = 1;
    unsigned char le;
    memcpy(&le, &one, 1);
    RRC_REQUIRE(le == 1, RRC_ERR_UNSUPPORTED, "rrc_state.bin needs a little-endian host");
    const size_t cfg_len = strlen(s->config_text);
    const size_t fbytes = (size_t)s->n_params * sizeof(float);
    RRC_REQUIRE(s->n_params <= (int64_t)(SIZE_MAX / 16), RRC_ERR_INVALID_ARG, "n_params too large");
    const size_t total = STATE_FIXED + cfg_len + 2 * fbytes + 8;
    unsigned char *buf = (unsigned char *)malloc(total);
    if (!buf) return RRC_ERROR(RRC_ERR_OOM, "rrc_state.bin buffer (%zu bytes)", total);
    memcpy(buf, STATE_MAGIC, 8);
    put_u64(buf + 8, 1);
    put_u64(buf + 16, (uint64_t)s->step);
    put_u64(buf + 24, (uint64_t)s->n_params);
    put_u64(buf + 32, s->data_seed);
    put_u64(buf + 40, (uint64_t)s->cursor.epoch);
    put_u64(buf + 48, (uint64_t)s->cursor.index);
    put_u64(buf + 56, (uint64_t)s->cursor.micro_batches);
    put_u64(buf + 64, (uint64_t)s->tokens_seen);
    put_u64(buf + 72, s->params_hash);
    put_u64(buf + 80, s->data_hash);
    put_u64(buf + 88, (uint64_t)cfg_len);
    memcpy(buf + STATE_FIXED, s->config_text, cfg_len);
    if (fbytes) {
        memcpy(buf + STATE_FIXED + cfg_len, s->m, fbytes);
        memcpy(buf + STATE_FIXED + cfg_len + fbytes, s->v, fbytes);
    }
    put_u64(buf + total - 8, rrc_fnv1a64(buf, total - 8));
    rrc_status st = write_atomic_fsync(path, buf, total);
    free(buf);
    return st;
}

void rrc_train_state_free(rrc_train_state *s) {
    if (!s) return;
    free(s->config_text);
    free(s->m);
    free(s->v);
    memset(s, 0, sizeof *s);
}

rrc_status rrc_train_state_read(const char *path, rrc_train_state *out) {
    RRC_REQUIRE(path && out, RRC_ERR_INVALID_ARG, "null argument");
    memset(out, 0, sizeof *out);
    FILE *f = fopen(path, "rb");
    if (!f) return RRC_ERROR(errno == ENOENT ? RRC_ERR_NOT_FOUND : RRC_ERR_IO, "cannot open %s: %s", path,
                             strerror(errno));
    struct stat sb;
    if (fstat(fileno(f), &sb) != 0 || !S_ISREG(sb.st_mode)) {
        fclose(f);
        return RRC_ERROR(RRC_ERR_IO, "%s is not a regular file", path);
    }
    const size_t size = (size_t)sb.st_size;
    if (size < STATE_FIXED + 8) {
        fclose(f);
        return RRC_ERROR(RRC_ERR_PARSE, "%s: truncated (%zu bytes)", path, size);
    }
    unsigned char *buf = (unsigned char *)malloc(size);
    if (!buf) {
        fclose(f);
        return RRC_ERROR(RRC_ERR_OOM, "reading %s", path);
    }
    const size_t got = fread(buf, 1, size, f);
    fclose(f);
    rrc_status st = RRC_OK;
#define BAD(...)                                                                                                       \
    do {                                                                                                               \
        st = RRC_ERROR(RRC_ERR_PARSE, __VA_ARGS__);                                                                    \
        goto done;                                                                                                     \
    } while (0)
    if (got != size) BAD("%s: short read", path);
    if (memcmp(buf, STATE_MAGIC, 8) != 0) BAD("%s: bad magic (not an rrc_state.bin)", path);
    if (get_u64(buf + 8) != 1) BAD("%s: unsupported version %llu", path, (unsigned long long)get_u64(buf + 8));
    if (get_u64(buf + size - 8) != rrc_fnv1a64(buf, size - 8)) BAD("%s: checksum mismatch (corrupt file)", path);
    {
        const uint64_t n = get_u64(buf + 24), cfg_len = get_u64(buf + 88);
        if (n > (uint64_t)INT64_MAX / 8 || cfg_len > size || STATE_FIXED + cfg_len + 8 * n + 8 != size)
            BAD("%s: sizes inconsistent (n_params %llu, config %llu bytes, file %zu bytes)", path,
                (unsigned long long)n, (unsigned long long)cfg_len, size);
        out->step = (int64_t)get_u64(buf + 16);
        out->n_params = (int64_t)n;
        out->data_seed = get_u64(buf + 32);
        out->cursor.epoch = (int64_t)get_u64(buf + 40);
        out->cursor.index = (int64_t)get_u64(buf + 48);
        out->cursor.micro_batches = (int64_t)get_u64(buf + 56);
        out->tokens_seen = (int64_t)get_u64(buf + 64);
        out->params_hash = get_u64(buf + 72);
        out->data_hash = get_u64(buf + 80);
        if (out->step < 0 || out->cursor.epoch < 0 || out->cursor.index < 0 || out->cursor.micro_batches < 0 ||
            out->tokens_seen < 0)
            BAD("%s: negative counter", path);
        /* the checksum is not a MAC: a crafted file passes it, and counters near INT64_MAX would
         * overflow on resume (loader epoch / micro-batch increments, tokens_seen += window) */
        if (out->step > RRC_DATA_COUNTER_MAX || out->cursor.epoch > RRC_DATA_COUNTER_MAX ||
            out->cursor.index > RRC_DATA_COUNTER_MAX || out->cursor.micro_batches > RRC_DATA_COUNTER_MAX ||
            out->tokens_seen > RRC_DATA_COUNTER_MAX)
            BAD("%s: counter out of range (step %lld, epoch %lld, index %lld, micro_batches %lld, tokens_seen %lld; "
                "at most 2^62)",
                path, (long long)out->step, (long long)out->cursor.epoch, (long long)out->cursor.index,
                (long long)out->cursor.micro_batches, (long long)out->tokens_seen);
        out->config_text = (char *)malloc((size_t)cfg_len + 1);
        out->m = (float *)malloc((size_t)(n ? n : 1) * sizeof(float));
        out->v = (float *)malloc((size_t)(n ? n : 1) * sizeof(float));
        if (!out->config_text || !out->m || !out->v) {
            st = RRC_ERROR(RRC_ERR_OOM, "reading %s", path);
            goto done;
        }
        memcpy(out->config_text, buf + STATE_FIXED, (size_t)cfg_len);
        out->config_text[cfg_len] = '\0';
        if (strlen(out->config_text) != cfg_len) BAD("%s: config text contains NUL", path);
        memcpy(out->m, buf + STATE_FIXED + cfg_len, (size_t)n * sizeof(float));
        memcpy(out->v, buf + STATE_FIXED + cfg_len + (size_t)n * sizeof(float), (size_t)n * sizeof(float));
    }
#undef BAD
done:
    free(buf);
    if (st != RRC_OK) rrc_train_state_free(out);
    return st;
}

/* ---- adapter_config.json ---------------------------------------------------------------------------- */

rrc_status rrc_peft_adapter_config_json(const rrc_lora_config *lc, const char *base_model, char **out) {
    RRC_REQUIRE(lc && base_model && out, RRC_ERR_INVALID_ARG, "null argument");
    rrc_jw w;
    rrc_jw_init(&w, 1);
    rrc_jw_obj_begin(&w);
    rrc_jw_key(&w, "alpha_pattern");
    rrc_jw_obj_begin(&w);
    rrc_jw_obj_end(&w);
    rrc_jw_key(&w, "base_model_name_or_path");
    rrc_jw_str(&w, base_model);
    rrc_jw_key(&w, "bias");
    rrc_jw_str(&w, "none");
    rrc_jw_key(&w, "fan_in_fan_out");
    rrc_jw_bool(&w, 0);
    rrc_jw_key(&w, "inference_mode");
    rrc_jw_bool(&w, 1);
    rrc_jw_key(&w, "init_lora_weights");
    rrc_jw_bool(&w, 1);
    rrc_jw_key(&w, "layers_pattern");
    rrc_jw_null(&w);
    rrc_jw_key(&w, "layers_to_transform");
    jw_layers(&w, lc);
    rrc_jw_key(&w, "lora_alpha");
    if (lc->alpha == floor(lc->alpha) && fabs(lc->alpha) < 9007199254740992.0) rrc_jw_int(&w, (int64_t)lc->alpha);
    else rrc_jw_double(&w, lc->alpha);
    rrc_jw_key(&w, "lora_bias");
    rrc_jw_bool(&w, 0);
    rrc_jw_key(&w, "lora_dropout");
    rrc_jw_double(&w, lc->dropout);
    rrc_jw_key(&w, "modules_to_save");
    rrc_jw_null(&w);
    rrc_jw_key(&w, "peft_type");
    rrc_jw_str(&w, "LORA");
    rrc_jw_key(&w, "r");
    rrc_jw_int(&w, lc->r);
    rrc_jw_key(&w, "rank_pattern");
    rrc_jw_obj_begin(&w);
    rrc_jw_obj_end(&w);
    rrc_jw_key(&w, "revision");
    rrc_jw_null(&w);
    rrc_jw_key(&w, "target_modules");
    jw_targets(&w, lc->target_mask);
    rrc_jw_key(&w, "task_type");
    rrc_jw_str(&w, "CAUSAL_LM");
    rrc_jw_key(&w, "use_dora");
    rrc_jw_bool(&w, 0);
    rrc_jw_key(&w, "use_rslora");
    rrc_jw_bool(&w, lc->use_rslora);
    rrc_jw_obj_end(&w);
    size_t n;
    return rrc_jw_finish(&w, out, &n);
}

/* If dir/adapter_config.json exists, refuse values that would silently change the adapter's
 * function (alpha, rslora, r, dropout, bias, target set, layers). */
static rrc_status check_adapter_config(const char *dir, const rrc_lora_config *lc) {
    char p[RRC_PATH_MAX + 32];
    snprintf(p, sizeof p, "%s/adapter_config.json", dir);
    if (!path_exists(p)) return RRC_OK;
    rrc_json_doc *doc = NULL;
    RRC_CHECK(rrc_json_parse_file(p, 0, &doc));
    const rrc_json *r = rrc_json_root(doc);
    rrc_status st = RRC_OK;
#define MISMATCH(what)                                                                                                 \
    do {                                                                                                               \
        st = RRC_ERROR(RRC_ERR_INVALID_ARG, "%s: %s differs from the training config's lora settings", p, what);       \
        goto done;                                                                                                     \
    } while (0)
    if (!r || r->type != RRC_JSON_OBJECT) {
        st = RRC_ERROR(RRC_ERR_INVALID_ARG, "%s: the root is not a JSON object (not a PEFT adapter config)", p);
        goto done;
    }
    {
        const rrc_json *v;
        double d;
        int b;
        if ((v = rrc_json_get(r, "r")) && (v->type != RRC_JSON_INT || v->u.i != lc->r)) MISMATCH("r");
        if ((v = rrc_json_get(r, "lora_alpha")) && (rrc_json_as_number(v, "lora_alpha", &d) != RRC_OK || d != lc->alpha))
            MISMATCH("lora_alpha");
        if ((v = rrc_json_get(r, "use_rslora")) && (rrc_json_as_bool(v, "use_rslora", &b) != RRC_OK || b != lc->use_rslora))
            MISMATCH("use_rslora");
        if ((v = rrc_json_get(r, "lora_dropout")) && (rrc_json_as_number(v, "lora_dropout", &d) != RRC_OK || d != 0.0))
            MISMATCH("lora_dropout (only 0 is supported)");
        if ((v = rrc_json_get(r, "bias")) && (!rrc_json_str(v) || strcmp(rrc_json_str(v), "none")))
            MISMATCH("bias (only \"none\" is supported)");
        if ((v = rrc_json_get(r, "use_dora")) && (rrc_json_as_bool(v, "use_dora", &b) != RRC_OK || b))
            MISMATCH("use_dora (not supported)");
        /* PEFT writes a list of module names, or a string (a regex matched against module paths)
         * that this check cannot evaluate: only a list is accepted */
        if ((v = rrc_json_get(r, "target_modules")) && v->type != RRC_JSON_ARRAY)
            MISMATCH("target_modules (only a list of module names can be checked; a string is a PEFT regex)");
        if (v) {
            unsigned mask = 0;
            for (size_t i = 0; i < rrc_json_arr_count(v); i++) {
                unsigned one;
                const char *s = rrc_json_str(rrc_json_arr_get(v, i));
                if (!s || rrc_lora_parse_targets(s, &one) != RRC_OK) MISMATCH("target_modules");
                mask |= one;
            }
            if (mask != lc->target_mask) MISMATCH("target_modules");
        }
        if ((v = rrc_json_get(r, "layers_to_transform"))) {
            if (v->type == RRC_JSON_NULL) {
                if (lc->n_layers) MISMATCH("layers_to_transform");
            } else if (v->type == RRC_JSON_ARRAY) {
                if ((int64_t)rrc_json_arr_count(v) != lc->n_layers) MISMATCH("layers_to_transform");
                for (size_t i = 0; i < rrc_json_arr_count(v); i++) {
                    const rrc_json *x = rrc_json_arr_get(v, i);
                    if (x->type != RRC_JSON_INT || x->u.i != lc->layers[i]) MISMATCH("layers_to_transform");
                }
            } else {
                MISMATCH("layers_to_transform");
            }
        }
    }
#undef MISMATCH
done:
    if (st == RRC_OK) rrc_clear_error(); /* drop errors recorded by the optional-field probes */
    rrc_json_doc_free(doc);
    return st;
}

/* ---- trainer --------------------------------------------------------------------------------------- */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

typedef struct trainer {
    const rrc_train_config *c;
    const rrc_backend *be;
    rrc_model *model;
    rrc_lora *lora;
    rrc_optimizer *opt;
    rrc_dataset *data;
    rrc_loader *loader;
    char *fingerprint;
    char *adapter_cfg;
    float *host; /* n_params staging */
    FILE *metrics;
    rrc_data_cursor cursor;
    int64_t tokens_seen;
    int32_t *dump; /* 4 * M * B * T when dump_batches */
} trainer;

static void trainer_free(trainer *t) {
    if (t->metrics) fclose(t->metrics);
    rrc_loader_free(t->loader);
    rrc_dataset_free(t->data);
    rrc_optimizer_free(t->opt);
    rrc_lora_free(t->lora);
    rrc_model_free(t->model);
    free(t->fingerprint);
    free(t->adapter_cfg);
    free(t->host);
    free(t->dump);
}

static rrc_status adapter_file_of(const char *path, char *dst, size_t cap) {
    if (is_dir(path)) return join(dst, cap, path, "adapter_model.safetensors");
    RRC_REQUIRE(strlen(path) < cap, RRC_ERR_INVALID_ARG, "path too long");
    snprintf(dst, cap, "%s", path);
    return RRC_OK;
}

/* Remove a checkpoint directory written by this trainer (its known files, then the directory). */
static rrc_status remove_ckpt_dir(const char *dir) {
    static const char *const files[] = {"adapter_model.safetensors", "adapter_config.json", "rrc_state.bin",
                                        "adapter_model.safetensors.tmp", "adapter_config.json.tmp",
                                        "rrc_state.bin.tmp"};
    char p[RRC_PATH_MAX + 64];
    for (size_t i = 0; i < sizeof files / sizeof files[0]; i++) {
        snprintf(p, sizeof p, "%s/%s", dir, files[i]);
        if (remove(p) != 0 && errno != ENOENT) return RRC_ERROR(RRC_ERR_IO, "remove %s: %s", p, strerror(errno));
    }
    if (rmdir(dir) != 0 && errno != ENOENT)
        return RRC_ERROR(RRC_ERR_IO, "rmdir %s: %s (unexpected files in a checkpoint directory?)", dir, strerror(errno));
    return RRC_OK;
}

static rrc_status save_checkpoint(trainer *t, int64_t step) {
    const rrc_train_config *c = t->c;
    char name[64], dir[RRC_PATH_MAX], tmp[RRC_PATH_MAX], old[RRC_PATH_MAX], p[RRC_PATH_MAX];
    snprintf(name, sizeof name, "checkpoint-%lld", (long long)step);
    RRC_CHECK(join(dir, sizeof dir, c->output_dir, name));
    snprintf(name, sizeof name, ".checkpoint-%lld.tmp", (long long)step);
    RRC_CHECK(join(tmp, sizeof tmp, c->output_dir, name));
    snprintf(name, sizeof name, ".checkpoint-%lld.old", (long long)step);
    RRC_CHECK(join(old, sizeof old, c->output_dir, name));
    RRC_CHECK(remove_ckpt_dir(tmp));
    /* checkpoint-<step> missing but .checkpoint-<step>.old present: an earlier replace of this step
     * stopped between its two renames. Put the old copy back instead of deleting the only one. */
    if (!path_exists(dir) && is_dir(old)) {
        if (rename(old, dir) != 0)
            return RRC_ERROR(RRC_ERR_IO, "restore %s -> %s: %s", old, dir, strerror(errno));
        rrc_log(RRC_LOG_WARN, "restored %s from %s (an earlier save of step %lld was interrupted)", dir, old,
                (long long)step);
    }
    RRC_CHECK(remove_ckpt_dir(old));
    if (mkdir(tmp, 0755) != 0) return RRC_ERROR(RRC_ERR_IO, "mkdir %s: %s", tmp, strerror(errno));
    /* adapter weights (PEFT key names, fp32, metadata format=pt as PEFT writes) */
    rrc_st_writer *w = NULL;
    RRC_CHECK(rrc_st_writer_create(&w));
    rrc_status st = rrc_lora_add_to_writer(t->lora, 0, "", w);
    if (st == RRC_OK) st = rrc_st_writer_set_metadata(w, "format", "pt");
    if (st == RRC_OK) st = join(p, sizeof p, tmp, "adapter_model.safetensors");
    if (st == RRC_OK) st = rrc_st_writer_write(w, p);
    rrc_st_writer_free(w);
    RRC_CHECK(st);
    RRC_CHECK(fsync_path(p));
    RRC_CHECK(join(p, sizeof p, tmp, "adapter_config.json"));
    RRC_CHECK(write_atomic_fsync(p, t->adapter_cfg, strlen(t->adapter_cfg)));
    /* optimizer + data state */
    const int64_t n = rrc_lora_num_params(t->lora);
    rrc_train_state s;
    memset(&s, 0, sizeof s);
    s.step = step;
    s.n_params = n;
    s.data_seed = c->seed;
    s.cursor = t->cursor;
    s.tokens_seen = t->tokens_seen;
    s.data_hash = rrc_dataset_hash(t->data);
    s.config_text = t->fingerprint;
    RRC_CHECK(rrc_buf_download(rrc_lora_params(t->lora), t->host, n));
    s.params_hash = rrc_fnv1a64(t->host, (size_t)n * sizeof(float));
    s.m = (float *)malloc((size_t)(n ? n : 1) * sizeof(float));
    s.v = (float *)malloc((size_t)(n ? n : 1) * sizeof(float));
    if (!s.m || !s.v) {
        free(s.m);
        free(s.v);
        return RRC_ERROR(RRC_ERR_OOM, "optimizer state download");
    }
    st = rrc_buf_download(rrc_optimizer_m(t->opt), s.m, n);
    if (st == RRC_OK) st = rrc_buf_download(rrc_optimizer_v(t->opt), s.v, n);
    if (st == RRC_OK) st = join(p, sizeof p, tmp, "rrc_state.bin");
    if (st == RRC_OK) st = rrc_train_state_write(p, &s);
    free(s.m);
    free(s.v);
    RRC_CHECK(st);
    RRC_CHECK(fsync_path(tmp));
    /* publish: an existing checkpoint-<step> is moved aside first, then removed */
    const int replacing = path_exists(dir);
    if (replacing && rename(dir, old) != 0)
        return RRC_ERROR(RRC_ERR_IO, "rename %s -> %s: %s", dir, old, strerror(errno));
    if (rename(tmp, dir) != 0) {
        const int e = errno;
        /* keep a resumable checkpoint-<step>: move the previous one back (if that fails too, the
         * next save of this step restores it from .old, see above) */
        const int restored = replacing && rename(old, dir) == 0;
        return RRC_ERROR(RRC_ERR_IO, "rename %s -> %s: %s%s", tmp, dir, strerror(e),
                         !replacing ? "" : restored ? " (the previous checkpoint was kept)"
                                                    : " (the previous checkpoint is left as .old)");
    }
    RRC_CHECK(fsync_path(c->output_dir));
    RRC_CHECK(remove_ckpt_dir(old));
    rrc_log(RRC_LOG_INFO, "saved %s", dir);
    return RRC_OK;
}

/* On resume: keep the metrics lines of steps <= `step` (a crash may have left later lines or a
 * partial last line), rewrite the file atomically. */
static rrc_status truncate_metrics(const char *path, int64_t step) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        if (errno == ENOENT) return RRC_OK;
        return RRC_ERROR(RRC_ERR_IO, "open %s: %s", path, strerror(errno));
    }
    sbuf keep = {NULL, 0, 0, 0};
    char *line = NULL;
    size_t cap = 0;
    ssize_t n;
    int64_t lineno = 0;
    rrc_status st = RRC_OK;
    while ((n = getline(&line, &cap, f)) >= 0) {
        lineno++;
        const int complete = n > 0 && line[n - 1] == '\n';
        rrc_json_doc *doc = NULL;
        int64_t s = -1;
        if (rrc_json_parse(line, (size_t)n, &doc) == RRC_OK) {
            const rrc_json *v = rrc_json_get(rrc_json_root(doc), "step");
            if (v && v->type == RRC_JSON_INT) s = v->u.i;
        }
        rrc_json_doc_free(doc);
        if (s < 0) {
            if (!complete) break; /* a partial last line from an interrupted write */
            st = RRC_ERROR(RRC_ERR_PARSE, "%s line %lld is not a metrics record", path, (long long)lineno);
            break;
        }
        if (s <= step) sb_printf(&keep, "%.*s%s", (int)n, line, complete ? "" : "\n");
    }
    free(line);
    fclose(f);
    if (st == RRC_OK) rrc_clear_error(); /* parse probes of skipped lines */
    if (st == RRC_OK && keep.oom) st = RRC_ERROR(RRC_ERR_OOM, "metrics rewrite");
    if (st == RRC_OK) st = write_atomic_fsync(path, keep.p ? keep.p : "", keep.len);
    free(keep.p);
    return st;
}

static rrc_status write_metrics(trainer *t, int64_t step, double loss, double lr, const rrc_optim_stats *os,
                                int64_t tokens, int64_t input_tokens, double step_time, double elapsed) {
    rrc_jw w;
    rrc_jw_init(&w, 0);
    rrc_jw_obj_begin(&w);
    rrc_jw_key(&w, "step");
    rrc_jw_int(&w, step);
    rrc_jw_key(&w, "loss");
    rrc_jw_double(&w, loss);
    rrc_jw_key(&w, "lr");
    rrc_jw_double(&w, lr);
    rrc_jw_key(&w, "grad_norm");
    rrc_jw_double(&w, (double)os->grad_norm);
    rrc_jw_key(&w, "clip_coef");
    rrc_jw_double(&w, (double)os->clip_coef);
    rrc_jw_key(&w, "tokens");
    rrc_jw_int(&w, tokens);
    rrc_jw_key(&w, "input_tokens");
    rrc_jw_int(&w, input_tokens);
    rrc_jw_key(&w, "tokens_seen");
    rrc_jw_int(&w, t->tokens_seen);
    rrc_jw_key(&w, "epoch");
    rrc_jw_int(&w, t->cursor.epoch);
    rrc_jw_key(&w, "step_time_s");
    rrc_jw_double(&w, step_time);
    rrc_jw_key(&w, "elapsed_s");
    rrc_jw_double(&w, elapsed);
    rrc_jw_key(&w, "input_tokens_per_s");
    rrc_jw_double(&w, step_time > 0 ? (double)input_tokens / step_time : 0.0);
    rrc_jw_obj_end(&w);
    char *s = NULL;
    size_t len = 0;
    RRC_CHECK(rrc_jw_finish(&w, &s, &len));
    const int ok = fwrite(s, 1, len, t->metrics) == len && fputc('\n', t->metrics) != EOF && fflush(t->metrics) == 0;
    free(s);
    RRC_REQUIRE(ok, RRC_ERR_IO, "writing metrics.jsonl failed: %s", strerror(errno));
    return RRC_OK;
}

static rrc_status dump_window(trainer *t, int64_t step, const rrc_microbatch *const *mbs, int64_t M) {
    const int64_t B = t->c->micro_batch_size, T = t->c->seq_len, n = B * T;
    char dir[RRC_PATH_MAX], p[RRC_PATH_MAX], name[64];
    RRC_CHECK(join(dir, sizeof dir, t->c->output_dir, "batches"));
    RRC_CHECK(mkdir_p(dir));
    snprintf(name, sizeof name, "step-%06lld.safetensors", (long long)step);
    RRC_CHECK(join(p, sizeof p, dir, name));
    for (int64_t mb = 0; mb < M; mb++) {
        memcpy(t->dump + (0 * M + mb) * n, mbs[mb]->tokens, (size_t)n * sizeof(int32_t));
        memcpy(t->dump + (1 * M + mb) * n, mbs[mb]->labels, (size_t)n * sizeof(int32_t));
        memcpy(t->dump + (2 * M + mb) * n, mbs[mb]->position_ids, (size_t)n * sizeof(int32_t));
        memcpy(t->dump + (3 * M + mb) * n, mbs[mb]->doc_ids, (size_t)n * sizeof(int32_t));
    }
    static const char *const keys[4] = {"tokens", "labels", "position_ids", "doc_ids"};
    const int64_t shape[3] = {M, B, T};
    rrc_st_writer *w = NULL;
    RRC_CHECK(rrc_st_writer_create(&w));
    rrc_status st = RRC_OK;
    for (int k = 0; k < 4 && st == RRC_OK; k++) st = rrc_st_writer_add(w, keys[k], RRC_I32, 3, shape, t->dump + k * M * n);
    if (st == RRC_OK) st = rrc_st_writer_write(w, p);
    rrc_st_writer_free(w);
    return st;
}

/* 1 if `dir` is the directory that contains `path` (compared by device / inode, not by spelling). */
static int is_parent_of(const char *dir, const char *path) {
    char p[RRC_PATH_MAX];
    if (strlen(path) >= sizeof p) return 0;
    snprintf(p, sizeof p, "%s", path);
    size_t n = strlen(p);
    while (n > 1 && p[n - 1] == '/') p[--n] = '\0';
    char *slash = strrchr(p, '/');
    if (!slash) snprintf(p, sizeof p, ".");
    else if (slash == p) p[1] = '\0';
    else *slash = '\0';
    struct stat a, b;
    return stat(dir, &a) == 0 && stat(p, &b) == 0 && a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}

/* 1 if output_dir holds a metrics.jsonl or a checkpoint-* entry (a run's outputs). */
static int holds_run(const char *dir, const char *mpath) {
    if (path_exists(mpath)) return 1;
    DIR *d = opendir(dir);
    if (!d) return 0;
    int found = 0;
    for (struct dirent *e; !found && (e = readdir(d));) found = !strncmp(e->d_name, "checkpoint-", 11);
    closedir(d);
    return found;
}

static double gib(uint64_t b) { return (double)b / (double)(1ull << 30); }

rrc_status rrc_train_memory_preflight(const rrc_train_config *c, const rrc_model_config *mc, int cuda,
                                      uint64_t device_total_bytes, rrc_train_preflight *out) {
    RRC_REQUIRE(c && mc, RRC_ERR_INVALID_ARG, "null argument");
    rrc_train_preflight pf;
    memset(&pf, 0, sizeof pf);
    if (c->memory_budget_bytes > 0) {
        pf.budget_bytes = (uint64_t)c->memory_budget_bytes;
        snprintf(pf.budget_source, sizeof pf.budget_source, "memory_budget_bytes");
    } else if (c->memory_budget_bytes == 0) {
        snprintf(pf.budget_source, sizeof pf.budget_source, "none (memory_budget_bytes = 0)");
    } else if (!cuda) {
        snprintf(pf.budget_source, sizeof pf.budget_source, "none (CPU backend: unlimited by default)");
    } else if (device_total_bytes > (uint64_t)RRC_TRAIN_CUDA_MEMORY_MARGIN) {
        pf.budget_bytes = device_total_bytes - (uint64_t)RRC_TRAIN_CUDA_MEMORY_MARGIN;
        snprintf(pf.budget_source, sizeof pf.budget_source, "device total %.2f GiB - margin %.2f GiB",
                 gib(device_total_bytes), gib((uint64_t)RRC_TRAIN_CUDA_MEMORY_MARGIN));
    } else {
        snprintf(pf.budget_source, sizeof pf.budget_source, "none (device total memory unknown)");
    }
    rrc_census_opts o;
    rrc_census_opts_default(&o);
    o.model.dtype = c->dtype;
    o.model.max_positions = c->max_positions;
    o.model.attn_chunk = c->attn_chunk;
    o.model.ce_chunk = c->ce_chunk;
    o.model.recompute = c->recompute;
    o.lora = c->lora;
    o.seq_len = c->seq_len;
    o.micro_batch = c->micro_batch_size;
    o.grad_accum = 1; /* memory does not depend on the window length */
    o.target = cuda ? RRC_CENSUS_TARGET_CUDA : RRC_CENSUS_TARGET_CPU;
    rrc_census_result *r = (rrc_census_result *)malloc(sizeof *r);
    if (!r) return RRC_ERROR(RRC_ERR_OOM, "memory pre-flight result");
    rrc_status st = rrc_census_run(mc, &o, r);
    if (st == RRC_OK) {
        pf.peak_bytes = r->peak_total_bytes;
        pf.weights_bytes = r->weights_bytes + r->rope_bytes;
        pf.lora_bytes = r->lora_bytes;
        pf.adam_bytes = r->adam_bytes;
        pf.workspace_bytes = r->workspace_bytes;
        pf.internal_bytes = r->gemm_scratch_bytes + r->cublaslt_workspace_bytes + r->sum_squares_partials_bytes;
    }
    free(r);
    if (out) *out = pf;
    if (st != RRC_OK) return rrc_error_context(st, __FILE__, __LINE__, "memory pre-flight");
    if (pf.budget_bytes && pf.peak_bytes > pf.budget_bytes)
        return RRC_ERROR(RRC_ERR_OOM,
                         "memory pre-flight: this run needs about %.2f GiB (%llu bytes) of %s memory, computed by a "
                         "dry run of one step (not measured): weights + RoPE tables %.2f GiB, LoRA params + grads "
                         "%.2f GiB, AdamW m / v %.2f GiB, workspace %.2f GiB, backend-internal buffers %.2f GiB; the "
                         "budget is %.2f GiB (%llu bytes, %s). Lower seq_len or micro_batch_size, use \"recompute\": "
                         "\"full\", smaller attn_chunk / ce_chunk, or set \"memory_budget_bytes\" (0 disables the "
                         "check)",
                         gib(pf.peak_bytes), (unsigned long long)pf.peak_bytes, cuda ? "device" : "backend",
                         gib(pf.weights_bytes), gib(pf.lora_bytes), gib(pf.adam_bytes), gib(pf.workspace_bytes),
                         gib(pf.internal_bytes), gib(pf.budget_bytes), (unsigned long long)pf.budget_bytes,
                         pf.budget_source);
    return RRC_OK;
}

static rrc_status trainer_setup(trainer *t, const rrc_backend *device_be, uint64_t device_total, int64_t *start_step) {
    const rrc_train_config *c = t->c;
    *start_step = 0;
    t->be = device_be ? device_be : c->backend == RRC_TRAIN_BACKEND_CPU_F32ACC ? rrc_backend_cpu_f32acc() : rrc_backend_cpu();
    {
        /* The shape checks (the same as after the model load below, same messages and statuses) and
         * the memory pre-flight run on config.json alone, before anything is allocated on the
         * backend or output_dir is created. */
        rrc_model_config mc0;
        RRC_CHECK_CTX(rrc_config_load(c->model_dir, 0, &mc0), "loading the model %s", c->model_dir);
        const int64_t npos0 = c->max_positions > 0 ? c->max_positions : mc0.max_position_embeddings;
        RRC_REQUIRE(c->seq_len <= npos0, RRC_ERR_INVALID_ARG,
                    "seq_len = %lld exceeds the model's positions (%lld; max_positions / max_position_embeddings)",
                    (long long)c->seq_len, (long long)npos0);
        RRC_REQUIRE(c->pad_token_id < mc0.vocab_size, RRC_ERR_RANGE, "pad_token_id = %d >= vocab_size %lld",
                    (int)c->pad_token_id, (long long)mc0.vocab_size);
        rrc_train_preflight pf;
        RRC_CHECK(rrc_train_memory_preflight(c, &mc0, device_be != NULL, device_total, &pf));
        rrc_log(RRC_LOG_INFO, "memory pre-flight (computed by a dry run, not measured): peak %.3f GiB, budget %s%.3f GiB",
                gib(pf.peak_bytes), pf.budget_bytes ? "" : "none: ", pf.budget_bytes ? gib(pf.budget_bytes) : 0.0);
        if (!pf.budget_bytes) rrc_log(RRC_LOG_INFO, "memory pre-flight budget: %s", pf.budget_source);
        if (device_be && c->memory_budget_bytes < 0 && !device_total)
            rrc_log(RRC_LOG_WARN, "memory pre-flight: the device's total memory is unknown, no budget applied");
    }
    rrc_model_opts mo;
    rrc_model_opts_default(&mo);
    mo.dtype = c->dtype;
    mo.max_positions = c->max_positions;
    mo.attn_chunk = c->attn_chunk;
    mo.ce_chunk = c->ce_chunk;
    mo.recompute = c->recompute;
    RRC_CHECK_CTX(rrc_model_load(t->be, c->model_dir, &mo, &t->model), "loading the model %s", c->model_dir);
    const rrc_model_config *mc = rrc_model_get_config(t->model);
    const int64_t npos = c->max_positions > 0 ? c->max_positions : mc->max_position_embeddings;
    RRC_REQUIRE(c->seq_len <= npos, RRC_ERR_INVALID_ARG,
                "seq_len = %lld exceeds the model's positions (%lld; max_positions / max_position_embeddings)",
                (long long)c->seq_len, (long long)npos);
    RRC_REQUIRE(c->pad_token_id < mc->vocab_size, RRC_ERR_RANGE, "pad_token_id = %d >= vocab_size %lld",
                (int)c->pad_token_id, (long long)mc->vocab_size);
    RRC_CHECK_CTX(rrc_dataset_open(c->data_path, mc->vocab_size, &t->data), "training data");
    RRC_CHECK_CTX(rrc_lora_create(t->be, mc, &c->lora, &t->lora), "LoRA adapter");
    const int64_t n = rrc_lora_num_params(t->lora);
    RRC_CHECK(rrc_optimizer_create(t->be, n, &t->opt));
    t->host = (float *)malloc((size_t)(n ? n : 1) * sizeof(float));
    if (!t->host) return RRC_ERROR(RRC_ERR_OOM, "host staging");
    RRC_CHECK(rrc_train_config_fingerprint(c, &t->fingerprint));
    RRC_CHECK(rrc_peft_adapter_config_json(&c->lora, c->model_dir, &t->adapter_cfg));
    if (c->dump_batches) {
        const int64_t total = 4 * c->grad_accum_steps * c->micro_batch_size * c->seq_len;
        t->dump = (int32_t *)malloc((size_t)total * sizeof(int32_t));
        if (!t->dump) return RRC_ERROR(RRC_ERR_OOM, "batch dump buffer");
    }
    RRC_CHECK(mkdir_p(c->output_dir));
    char mpath[RRC_PATH_MAX];
    RRC_CHECK(join(mpath, sizeof mpath, c->output_dir, "metrics.jsonl"));
    if (c->resume_from[0]) {
        char p[RRC_PATH_MAX];
        rrc_train_state s;
        RRC_CHECK(join(p, sizeof p, c->resume_from, "rrc_state.bin"));
        RRC_CHECK_CTX(rrc_train_state_read(p, &s), "resume");
        rrc_status st = RRC_OK;
        if (strcmp(s.config_text, t->fingerprint) != 0) {
            const char *a = s.config_text, *b = t->fingerprint;
            while (*a && *a == *b) a++, b++;
            while (a > s.config_text && a[-1] != '\n') a--;
            const char *ea = strchr(a, '\n');
            st = RRC_ERROR(RRC_ERR_INVALID_ARG,
                           "resume: the training config differs from the checkpoint's at \"%.*s\" (the checkpoint "
                           "was written with a different trajectory-relevant setting)",
                           ea ? (int)(ea - a) : (int)strlen(a), a);
        } else if (s.n_params != n) {
            st = RRC_ERROR(RRC_ERR_SHAPE, "resume: checkpoint has %lld parameters, the adapter %lld",
                           (long long)s.n_params, (long long)n);
        } else if (s.data_hash != rrc_dataset_hash(t->data)) {
            st = RRC_ERROR(RRC_ERR_INVALID_ARG, "resume: %s is not the data file the checkpoint was trained on",
                           c->data_path);
        } else if (s.step > c->max_steps) {
            st = RRC_ERROR(RRC_ERR_INVALID_ARG, "resume: checkpoint step %lld > max_steps", (long long)s.step);
        }
        if (st == RRC_OK) st = check_adapter_config(c->resume_from, &c->lora);
        if (st == RRC_OK) st = join(p, sizeof p, c->resume_from, "adapter_model.safetensors");
        if (st == RRC_OK) st = rrc_lora_load_safetensors(t->lora, p);
        if (st == RRC_OK) st = rrc_buf_download(rrc_lora_params(t->lora), t->host, n);
        if (st == RRC_OK && rrc_fnv1a64(t->host, (size_t)n * sizeof(float)) != s.params_hash)
            st = RRC_ERROR(RRC_ERR_INVALID_ARG, "resume: %s does not hold the parameters rrc_state.bin was saved "
                                                "with",
                           p);
        if (st == RRC_OK) st = rrc_optimizer_restore(t->opt, s.m, s.v, s.step);
        if (st == RRC_OK) {
            t->cursor = s.cursor;
            t->tokens_seen = s.tokens_seen;
            *start_step = s.step;
        }
        rrc_train_state_free(&s);
        RRC_CHECK(st);
        /* resume into the checkpoint's own output_dir: keep its metrics up to the step. Into another
         * directory: only an empty one (or overwrite_output_dir), whose metrics start afresh. */
        if (is_parent_of(c->output_dir, c->resume_from)) {
            RRC_CHECK(truncate_metrics(mpath, *start_step));
            t->metrics = fopen(mpath, "ab");
        } else {
            RRC_REQUIRE(c->overwrite_output_dir || !holds_run(c->output_dir, mpath), RRC_ERR_INVALID_ARG,
                        "%s already holds another run (metrics.jsonl or checkpoint-*) and is not the directory of "
                        "resume_from %s: resume into that directory or an empty one, or set "
                        "\"overwrite_output_dir\": true",
                        c->output_dir, c->resume_from);
            t->metrics = fopen(mpath, "wb");
        }
    } else {
        if (c->init_adapter[0]) {
            char p[RRC_PATH_MAX];
            RRC_CHECK(adapter_file_of(c->init_adapter, p, sizeof p));
            if (is_dir(c->init_adapter)) RRC_CHECK(check_adapter_config(c->init_adapter, &c->lora));
            RRC_CHECK_CTX(rrc_lora_load_safetensors(t->lora, p), "init_adapter");
        }
        RRC_REQUIRE(c->overwrite_output_dir || !path_exists(mpath), RRC_ERR_INVALID_ARG,
                    "%s already exists: set \"overwrite_output_dir\": true, or \"resume_from\" a checkpoint", mpath);
        t->metrics = fopen(mpath, "wb");
    }
    if (!t->metrics) return RRC_ERROR(RRC_ERR_IO, "cannot open %s: %s", mpath, strerror(errno));
    rrc_loader_opts lo;
    memset(&lo, 0, sizeof lo);
    lo.seq_len = c->seq_len;
    lo.rows = c->micro_batch_size;
    lo.seed = c->seed;
    lo.shuffle = c->shuffle;
    lo.truncation = c->truncation;
    lo.pad_token_id = c->pad_token_id;
    lo.max_held = c->grad_accum_steps;
    lo.prefetch = c->prefetch;
    RRC_CHECK_CTX(rrc_loader_create(t->data, &lo, &t->cursor, &t->loader), "data loader");
    return RRC_OK;
}

rrc_status rrc_train_run(const rrc_train_config *c) { return rrc_train_run_on_device(c, NULL, 0); }

rrc_status rrc_train_run_on(const rrc_train_config *c, const rrc_backend *device_be) {
    return rrc_train_run_on_device(c, device_be, 0);
}

rrc_status rrc_train_run_on_device(const rrc_train_config *c, const rrc_backend *device_be, uint64_t device_total) {
    RRC_REQUIRE(c, RRC_ERR_INVALID_ARG, "null argument");
    RRC_REQUIRE(c->backend >= RRC_TRAIN_BACKEND_CPU && c->backend <= RRC_TRAIN_BACKEND_CUDA, RRC_ERR_INVALID_ARG,
                "config: invalid backend %d", (int)c->backend);
    RRC_REQUIRE(c->backend == RRC_TRAIN_BACKEND_CUDA || !device_be, RRC_ERR_INVALID_ARG,
                "a device backend was supplied but the config's backend is \"%s\"", k_backend_names[c->backend]);
    RRC_REQUIRE(c->backend != RRC_TRAIN_BACKEND_CUDA || device_be, RRC_ERR_UNSUPPORTED,
                "config: \"backend\": \"cuda\" needs the CUDA build of rrc_train: this program has only the CPU "
                "backends (\"cpu\", \"cpu-f32acc\"); build it with `make -C rrc cuda CUDA_HOME=...` and run "
                "build/rrc_train_cuda");
    const double t_start = now_s();
    trainer t;
    memset(&t, 0, sizeof t);
    t.c = c;
    int64_t step0 = 0;
    rrc_status st = trainer_setup(&t, device_be, device_total, &step0);
    const int64_t M = c->grad_accum_steps;
    const int64_t end = c->stop_after_step > 0 ? c->stop_after_step : c->max_steps;
    const int64_t warmup = rrc_train_config_warmup(c);
    const rrc_microbatch **mbs = (const rrc_microbatch **)calloc((size_t)M, sizeof *mbs);
    if (st == RRC_OK && !mbs) st = RRC_ERROR(RRC_ERR_OOM, "micro-batch table");
    if (st == RRC_OK && step0 >= end) rrc_log(RRC_LOG_WARN, "nothing to do: checkpoint step %lld >= %lld",
                                              (long long)step0, (long long)end);
    for (int64_t step = step0 + 1; st == RRC_OK && step <= end; step++) {
        const double t0 = now_s();
        int64_t window_tokens = 0, input_tokens = 0, held = 0;
        for (int64_t i = 0; i < M && st == RRC_OK; i++) {
            st = rrc_loader_next(t.loader, &mbs[i]);
            if (st != RRC_OK) break;
            held++;
            rrc_batch b = {mbs[i]->rows, mbs[i]->seq_len, mbs[i]->tokens, mbs[i]->labels, mbs[i]->position_ids,
                           mbs[i]->doc_ids};
            int64_t nl = 0;
            st = rrc_batch_count_labels(&b, &nl);
            window_tokens += nl;
            input_tokens += mbs[i]->n_input_tokens;
        }
        double loss = 0.0;
        if (st == RRC_OK) st = rrc_lora_zero_grads(t.lora);
        for (int64_t i = 0; i < M && st == RRC_OK; i++) {
            rrc_batch b = {mbs[i]->rows, mbs[i]->seq_len, mbs[i]->tokens, mbs[i]->labels, mbs[i]->position_ids,
                           mbs[i]->doc_ids};
            rrc_fb_opts fo = {c->loss_reduction, window_tokens, M, 1};
            rrc_fb_result res;
            memset(&res, 0, sizeof res);
            st = rrc_model_forward_backward(t.model, t.lora, &b, &fo, &res);
            if (st != RRC_OK) st = rrc_error_context(st, __FILE__, __LINE__, "step %lld micro-batch %lld",
                                                     (long long)step, (long long)i);
            loss += res.loss;
        }
        if (st == RRC_OK && c->dump_batches) st = dump_window(&t, step, mbs, M);
        if (st == RRC_OK) t.cursor = mbs[M - 1]->end;
        for (int64_t i = 0; i < held; i++) {
            rrc_status rs = rrc_loader_release(t.loader, mbs[i]);
            if (st == RRC_OK) st = rs;
        }
        if (st != RRC_OK) break;
        if (window_tokens == 0)
            rrc_log(RRC_LOG_WARN, "step %lld: no labelled token in the window; the update uses a zero gradient",
                    (long long)step);
        const double lr = c->learning_rate * rrc_lr_lambda(c->lr_scheduler_type, step - 1, warmup, c->max_steps);
        rrc_optim_stats os;
        st = rrc_optimizer_step(t.opt, rrc_lora_params(t.lora), rrc_lora_grads(t.lora), lr, &c->optim, &os);
        if (st != RRC_OK) break;
        st = rrc_sync(t.be);
        if (st != RRC_OK) break;
        if (step == step0 + 1) /* the workspace is planned by the first forward / backward */
            rrc_log(RRC_LOG_INFO, "per-step workspace %.3f GB (recompute %s), LoRA parameters + grads + Adam m / v "
                    "%.3f GB (base weights not included)",
                    (double)rrc_model_workspace_bytes(t.model) / 1e9, c->recompute == RRC_RECOMPUTE_NONE ? "none" : "full",
                    (double)rrc_lora_num_params(t.lora) * 16.0 / 1e9);
        t.tokens_seen += window_tokens;
        const double t1 = now_s();
        st = write_metrics(&t, step, loss, lr, &os, window_tokens, input_tokens, t1 - t0, t1 - t_start);
        if (st == RRC_OK && ((c->save_steps > 0 && step % c->save_steps == 0) || step == end))
            st = save_checkpoint(&t, step);
    }
    free(mbs);
    trainer_free(&t);
    return st;
}
