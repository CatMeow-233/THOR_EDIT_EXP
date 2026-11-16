/* rrc/src/census.c — GEMM census and memory pre-flight by a dry run (include/rrc/census.h). */
#include "rrc/census.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gemm_plan.h"
#include "model_internal.h"
#include "rrc/json.h"
#include "rrc/train.h"

/* ---- roles and dtype combinations ---------------------------------------------------------------- */

static const char *const k_roles[RRC_CR_N] = {"base_fwd", "base_dx", "lm_head_fwd", "lm_head_dx", "lora", "attention"};

const char *rrc_census_role_name(int role) { return role >= 0 && role < RRC_CR_N ? k_roles[role] : NULL; }
int rrc_census_role_is_base(int role) { return role >= RRC_CR_BASE_FWD && role <= RRC_CR_LM_HEAD_DX; }

int rrc_census_combo(rrc_dtype a, rrc_dtype b, rrc_dtype c) {
    return 4 * (a == RRC_BF16) + 2 * (b == RRC_BF16) + (c == RRC_BF16);
}
rrc_dtype rrc_census_combo_dtype(int combo, int operand) {
    return (combo >> (2 - operand)) & 1 ? RRC_BF16 : RRC_F32;
}
/* The level-0 flags of cuda/gemm_plan.c rrc_gemm_plan_make, as a function of the dtypes. */
static void combo_flags(int combo, int *pa, int *pb, int *cs) {
    const int a16 = (combo >> 2) & 1, b16 = (combo >> 1) & 1, c16 = combo & 1;
    *pa = a16 && !b16;
    *pb = b16 && !a16;
    *cs = c16 && !(a16 && b16);
}
int rrc_census_combo_native(int combo) {
    int pa, pb, cs;
    combo_flags(combo, &pa, &pb, &cs);
    return !pa && !pb && !cs;
}
int rrc_census_combo_native_bf16(int combo) { return rrc_census_combo_native(combo) && ((combo >> 1) & 3) == 3; }
const char *rrc_census_combo_path(int combo) {
    static const char *const k[RRC_CENSUS_N_COMBOS] = {
        "native",                                 /* f32 f32 f32 */
        "C via f32 scratch",                      /* f32 f32 bf16 */
        "promote B (bf16->f32)",                  /* f32 bf16 f32 */
        "promote B (bf16->f32), C via f32 scratch", /* f32 bf16 bf16 */
        "promote A (bf16->f32)",                  /* bf16 f32 f32 */
        "promote A (bf16->f32), C via f32 scratch", /* bf16 f32 bf16 */
        "native",                                 /* bf16 bf16 f32 */
        "native",                                 /* bf16 bf16 bf16 */
    };
    return combo >= 0 && combo < RRC_CENSUS_N_COMBOS ? k[combo] : NULL;
}

void rrc_census_opts_default(rrc_census_opts *o) {
    memset(o, 0, sizeof *o);
    rrc_model_opts_default(&o->model);
    rrc_lora_config_default(&o->lora);
    o->seq_len = 0;
    o->micro_batch = 1;
    o->grad_accum = 1;
    o->target = RRC_CENSUS_TARGET_CUDA;
    o->gemm_scratch_cap_elems = 0;
    o->allow_tf32 = 0;
    o->cublaslt_workspace_bytes = (uint64_t)32 << 20;
}

/* ---- the dry run ------------------------------------------------------------------------------------ */

typedef struct range {
    uintptr_t lo, hi; /* [lo, hi) */
} range;

static range buf_range(rrc_buf b) {
    range r = {0, 0};
    if (b.ptr && b.numel > 0) {
        r.lo = (uintptr_t)b.ptr;
        r.hi = r.lo + (uintptr_t)b.numel * (uintptr_t)rrc_dtype_size(b.dtype);
    }
    return r;
}
static int in_range(range r, const void *p) { return p && (uintptr_t)p >= r.lo && (uintptr_t)p < r.hi; }

typedef struct census_ctx {
    range base, lm_head, lora_p, lora_g;
    rrc_census_result *r;
} census_ctx;

static void census_hook(void *vctx, const rrc_dryrun_call *c) {
    census_ctx *x = (census_ctx *)vctx;
    if (c->op != RRC_DR_GEMM || !c->gemm) return;
    const rrc_dryrun_gemm *g = c->gemm;
    int role;
    if (in_range(x->lm_head, c->ptr[1]))
        role = g->args.trans_b == RRC_OP_T ? RRC_CR_LM_HEAD_FWD : RRC_CR_LM_HEAD_DX;
    else if (in_range(x->base, c->ptr[1]))
        role = g->args.trans_b == RRC_OP_T ? RRC_CR_BASE_FWD : RRC_CR_BASE_DX;
    else if (in_range(x->lora_p, c->ptr[0]) || in_range(x->lora_p, c->ptr[1]) || in_range(x->lora_g, c->ptr[2]))
        role = RRC_CR_LORA;
    else
        role = RRC_CR_ATTENTION;
    const int combo = rrc_census_combo(g->args.a.dtype, g->args.b.dtype, g->args.c.dtype);
    rrc_census_cell *cell = &x->r->cell[role][combo];
    cell->calls++;
    cell->flops += g->flops;
    cell->convert_elems += g->convert_elems;
    cell->convert_bytes += g->convert_bytes;
    if (g->min_align < 16) cell->align_lt16_calls++;
}

/* One optimizer step exactly as rrc_train_run_on does it. */
static rrc_status train_step(rrc_model *m, rrc_lora *lora, rrc_optimizer *opt, const rrc_batch *b, int64_t M,
                             const rrc_train_config *defaults) {
    int64_t window = 0;
    for (int64_t i = 0; i < M; i++) {
        int64_t nl = 0;
        RRC_CHECK(rrc_batch_count_labels(b, &nl));
        window += nl;
    }
    RRC_CHECK(rrc_lora_zero_grads(lora));
    for (int64_t i = 0; i < M; i++) {
        rrc_fb_opts fo = {defaults->loss_reduction, window, M, 1};
        rrc_fb_result res;
        RRC_CHECK_CTX(rrc_model_forward_backward(m, lora, b, &fo, &res), "dry-run micro-batch %lld", (long long)i);
    }
    rrc_optim_stats os;
    RRC_CHECK(rrc_optimizer_step(opt, rrc_lora_params(lora), rrc_lora_grads(lora), defaults->learning_rate,
                                 &defaults->optim, &os));
    return rrc_sync(rrc_model_backend(m));
}

rrc_status rrc_census_run(const rrc_model_config *cfg, const rrc_census_opts *o, rrc_census_result *r) {
    RRC_REQUIRE(cfg && o && r, RRC_ERR_INVALID_ARG, "null argument");
    memset(r, 0, sizeof *r);
    r->cfg = *cfg;
    RRC_REQUIRE(o->seq_len > 0 && o->micro_batch > 0 && o->grad_accum > 0, RRC_ERR_INVALID_ARG,
                "census: seq_len = %lld, micro_batch = %lld and grad_accum = %lld must be > 0", (long long)o->seq_len,
                (long long)o->micro_batch, (long long)o->grad_accum);
    RRC_REQUIRE(o->seq_len <= INT32_MAX && o->micro_batch <= INT32_MAX / o->seq_len && o->grad_accum <= 65536,
                RRC_ERR_INVALID_ARG, "census: micro_batch * seq_len must be < 2^31 and grad_accum <= 65536");
    RRC_REQUIRE(o->target == RRC_CENSUS_TARGET_CUDA || o->target == RRC_CENSUS_TARGET_CPU, RRC_ERR_INVALID_ARG,
                "census: invalid target %d", (int)o->target);
    const int64_t npos = o->model.max_positions > 0 ? o->model.max_positions : cfg->max_position_embeddings;
    RRC_REQUIRE(o->seq_len <= npos, RRC_ERR_INVALID_ARG,
                "census: seq_len = %lld exceeds the model's positions (%lld; max_positions / max_position_embeddings)",
                (long long)o->seq_len, (long long)npos);

    rrc_dryrun_opts dro;
    rrc_dryrun_opts_default(&dro);
    dro.gemm_scratch_cap_elems = o->gemm_scratch_cap_elems;
    dro.allow_tf32 = o->allow_tf32;
    const rrc_backend *be = NULL;
    RRC_CHECK(rrc_dryrun_backend_create(&dro, &be));
    rrc_model *m = NULL;
    rrc_lora *lora = NULL;
    rrc_optimizer *opt = NULL;
    int32_t *tokens = NULL;
    rrc_status st = RRC_OK;
    rrc_dryrun_stats s;

    RRC_CHECK_GOTO(st, rrc_model_create_shape_only(be, cfg, &o->model, &m), done);
    r->base_params = m->storage.numel;
    r->weights_bytes = (uint64_t)m->storage.numel * rrc_dtype_size(m->storage.dtype);
    r->rope_bytes = (uint64_t)m->rope_storage.numel * sizeof(float);
    rrc_dryrun_get_stats(be, &s);
    const uint64_t live_model = s.live_bytes;
    RRC_CHECK_GOTO(st, rrc_lora_create(be, cfg, &o->lora, &lora), done);
    r->lora_params = rrc_lora_num_params(lora);
    rrc_dryrun_get_stats(be, &s);
    r->lora_bytes = s.live_bytes - live_model;
    const uint64_t live_lora = s.live_bytes;
    RRC_CHECK_GOTO(st, rrc_optimizer_create(be, r->lora_params, &opt), done);
    rrc_dryrun_get_stats(be, &s);
    r->adam_bytes = s.live_bytes - live_lora;

    /* packed rows of one document each, every token labelled */
    const int64_t N = o->micro_batch * o->seq_len;
    tokens = (int32_t *)calloc((size_t)N, sizeof(int32_t));
    if (!tokens) {
        st = RRC_ERROR(RRC_ERR_OOM, "census batch (%lld tokens)", (long long)N);
        goto done;
    }
    rrc_batch b = {o->micro_batch, o->seq_len, tokens, tokens, NULL, NULL};
    rrc_train_config defaults;
    rrc_train_config_default(&defaults);
    census_ctx cx;
    cx.base = buf_range(m->storage);
    cx.lm_head = buf_range(m->lm_head);
    cx.lora_p = buf_range(rrc_lora_params(lora));
    cx.lora_g = buf_range(rrc_lora_grads(lora));
    cx.r = r;
    /* step 1 (warm-up: workspace allocation, scratch growth), then the reported step 2 */
    RRC_CHECK_GOTO(st, train_step(m, lora, opt, &b, o->grad_accum, &defaults), done);
    rrc_dryrun_get_stats(be, &s);
    const uint64_t peak_first = s.peak_bytes, scratch_first = s.gemm_scratch_max_bytes,
                   ss_first = s.sum_squares_partials_max_bytes;
    rrc_dryrun_reset_counters(be);
    rrc_dryrun_set_hook(be, census_hook, &cx);
    RRC_CHECK_GOTO(st, train_step(m, lora, opt, &b, o->grad_accum, &defaults), done);
    rrc_dryrun_set_hook(be, NULL, NULL);
    rrc_dryrun_get_stats(be, &r->step);
    {
        int64_t nl = 0;
        RRC_CHECK_GOTO(st, rrc_batch_count_labels(&b, &nl), done);
        r->tokens_per_step = N * o->grad_accum;
        r->labelled_per_step = nl * o->grad_accum;
    }
    r->workspace_bytes = rrc_model_workspace_bytes(m);
    r->op_peak_bytes = peak_first > r->step.peak_bytes ? peak_first : r->step.peak_bytes;
    if (o->target == RRC_CENSUS_TARGET_CUDA) {
        r->gemm_scratch_bytes = scratch_first > r->step.gemm_scratch_max_bytes ? scratch_first
                                                                               : r->step.gemm_scratch_max_bytes;
        r->sum_squares_partials_bytes = ss_first > r->step.sum_squares_partials_max_bytes
                                            ? ss_first
                                            : r->step.sum_squares_partials_max_bytes;
        r->cublaslt_workspace_bytes = o->cublaslt_workspace_bytes;
    }
    r->peak_total_bytes = r->op_peak_bytes + r->gemm_scratch_bytes + r->sum_squares_partials_bytes +
                          r->cublaslt_workspace_bytes;
done:
    if (st != RRC_OK) st = rrc_error_context(st, __FILE__, __LINE__, "census dry run");
    free(tokens);
    rrc_optimizer_free(opt);
    rrc_lora_free(lora);
    rrc_model_free(m);
    rrc_dryrun_backend_destroy(be);
    return st;
}

/* ---- JSON ------------------------------------------------------------------------------------------- */

static void kv_int(rrc_jw *w, const char *k, int64_t v) {
    rrc_jw_key(w, k);
    rrc_jw_int(w, v);
}
static void kv_u64(rrc_jw *w, const char *k, uint64_t v) {
    rrc_jw_key(w, k);
    rrc_jw_int(w, v > (uint64_t)INT64_MAX ? INT64_MAX : (int64_t)v);
}
static void kv_dbl(rrc_jw *w, const char *k, double v) {
    rrc_jw_key(w, k);
    rrc_jw_double(w, v);
}
static void kv_str(rrc_jw *w, const char *k, const char *v) {
    rrc_jw_key(w, k);
    if (v)
        rrc_jw_str(w, v);
    else
        rrc_jw_null(w);
}
static void kv_bool(rrc_jw *w, const char *k, int v) {
    rrc_jw_key(w, k);
    rrc_jw_bool(w, v);
}
/* FLOP counts are exact integers well below 2^53 for any realistic step */
static void kv_flops(rrc_jw *w, const char *k, double v) {
    rrc_jw_key(w, k);
    if (v < 9.0e15)
        rrc_jw_int(w, (int64_t)v);
    else
        rrc_jw_double(w, v);
}
static double frac(double a, double b) { return b > 0 ? a / b : 0.0; }

static void write_targets(rrc_jw *w, unsigned mask) {
    rrc_jw_arr_begin(w);
    for (int i = 0; i < RRC_LORA_N_MODULES; i++)
        if (mask & (1u << i)) rrc_jw_str(w, rrc_lora_module_name(i));
    rrc_jw_arr_end(w);
}

rrc_status rrc_census_to_json(const rrc_census_result *r, const rrc_census_opts *o, const char *source,
                              const char *preset_note, int pretty, char **out) {
    RRC_REQUIRE(r && o && out, RRC_ERR_INVALID_ARG, "null argument");
    *out = NULL;
    const rrc_model_config *c = &r->cfg;
    rrc_jw jw, *w = &jw;
    rrc_jw_init(w, pretty);
    rrc_jw_obj_begin(w);
    kv_int(w, "rrc_census_version", 1);
    kv_str(w, "status", "computed");
    kv_str(w, "status_note",
           "Every number is COMPUTED from the op-call sequence of one optimizer step on the dry-run backend "
           "(rrc/dryrun.h): nothing ran on a GPU and nothing was measured. 'native' is the level-0 path of the CUDA "
           "GEMM plan (rrc/cuda/gemm_plan.c) under the assumed cuBLASLt type table of DESIGN.md 11.2; whether "
           "cuBLASLt accepts each call on the device is NOT_TESTED.");
    /* model */
    rrc_jw_key(w, "model");
    rrc_jw_obj_begin(w);
    kv_str(w, "source", source);
    kv_str(w, "preset_note", preset_note);
    kv_str(w, "model_type", rrc_family_name(c->family));
    kv_int(w, "vocab_size", c->vocab_size);
    kv_int(w, "hidden_size", c->hidden_size);
    kv_int(w, "intermediate_size", c->intermediate_size);
    kv_int(w, "num_hidden_layers", c->num_layers);
    kv_int(w, "num_attention_heads", c->num_heads);
    kv_int(w, "num_key_value_heads", c->num_kv_heads);
    kv_int(w, "head_dim", c->head_dim);
    kv_int(w, "max_position_embeddings", c->max_position_embeddings);
    kv_bool(w, "tie_word_embeddings", c->tie_word_embeddings);
    kv_bool(w, "qkv_bias", c->qkv_bias);
    kv_bool(w, "qk_norm", c->qk_norm);
    kv_int(w, "base_params", r->base_params);
    rrc_jw_obj_end(w);
    /* run */
    rrc_jw_key(w, "run");
    rrc_jw_obj_begin(w);
    kv_str(w, "dtype", rrc_dtype_name(o->model.dtype));
    kv_int(w, "seq_len", o->seq_len);
    kv_int(w, "micro_batch", o->micro_batch);
    kv_int(w, "grad_accum", o->grad_accum);
    kv_int(w, "tokens_per_step", r->tokens_per_step);
    kv_int(w, "labelled_tokens_per_step", r->labelled_per_step);
    kv_str(w, "batch", "one full-length document per packed row, every token labelled (worst case)");
    kv_str(w, "recompute", o->model.recompute == RRC_RECOMPUTE_NONE ? "none" : "full");
    kv_int(w, "attn_chunk", o->model.attn_chunk);
    kv_int(w, "ce_chunk", o->model.ce_chunk);
    rrc_jw_key(w, "lora");
    rrc_jw_obj_begin(w);
    kv_int(w, "r", o->lora.r);
    kv_dbl(w, "alpha", o->lora.alpha);
    kv_bool(w, "use_rslora", o->lora.use_rslora);
    rrc_jw_key(w, "target_modules");
    write_targets(w, o->lora.target_mask);
    kv_int(w, "num_layers_adapted", o->lora.n_layers ? o->lora.n_layers : c->num_layers);
    kv_int(w, "params", r->lora_params);
    rrc_jw_obj_end(w);
    kv_str(w, "target", o->target == RRC_CENSUS_TARGET_CUDA ? "cuda" : "cpu");
    kv_int(w, "gemm_scratch_cap_elems",
           o->gemm_scratch_cap_elems > 0 ? o->gemm_scratch_cap_elems : RRC_GEMM_SCRATCH_CAP_DEFAULT);
    kv_bool(w, "allow_tf32", o->allow_tf32);
    kv_str(w, "step", "the second optimizer step (steady state; the first also allocates the workspace and grows "
                      "the GEMM scratch)");
    rrc_jw_obj_end(w);

    /* gemm */
    double tot = 0, nat = 0, nat16 = 0, conv_b = 0;
    double base = 0, base_nat16 = 0;
    int64_t base_calls = 0, base_nonnat = 0;
    for (int ro = 0; ro < RRC_CR_N; ro++)
        for (int k = 0; k < RRC_CENSUS_N_COMBOS; k++) {
            const rrc_census_cell *e = &r->cell[ro][k];
            tot += e->flops;
            conv_b += (double)e->convert_bytes;
            if (rrc_census_combo_native(k)) nat += e->flops;
            if (rrc_census_combo_native_bf16(k)) nat16 += e->flops;
            if (rrc_census_role_is_base(ro)) {
                base += e->flops;
                base_calls += e->calls;
                if (!rrc_census_combo_native(k)) base_nonnat += e->calls;
                if (rrc_census_combo_native_bf16(k)) base_nat16 += e->flops;
            }
        }
    rrc_jw_key(w, "gemm");
    rrc_jw_obj_begin(w);
    kv_int(w, "calls", r->step.calls[RRC_DR_GEMM]);
    kv_flops(w, "flops_total", tot);
    kv_flops(w, "flops_native", nat);
    kv_flops(w, "flops_native_bf16", nat16);
    kv_dbl(w, "native_fraction", frac(nat, tot));
    kv_dbl(w, "native_bf16_fraction", frac(nat16, tot));
    rrc_jw_key(w, "by_combo");
    rrc_jw_arr_begin(w);
    for (int k = 0; k < RRC_CENSUS_N_COMBOS; k++) {
        rrc_census_cell sum;
        memset(&sum, 0, sizeof sum);
        for (int ro = 0; ro < RRC_CR_N; ro++) {
            sum.calls += r->cell[ro][k].calls;
            sum.flops += r->cell[ro][k].flops;
            sum.convert_bytes += r->cell[ro][k].convert_bytes;
            sum.convert_elems += r->cell[ro][k].convert_elems;
            sum.align_lt16_calls += r->cell[ro][k].align_lt16_calls;
        }
        if (!sum.calls) continue;
        rrc_jw_obj_begin(w);
        kv_str(w, "a", rrc_dtype_name(rrc_census_combo_dtype(k, 0)));
        kv_str(w, "b", rrc_dtype_name(rrc_census_combo_dtype(k, 1)));
        kv_str(w, "c", rrc_dtype_name(rrc_census_combo_dtype(k, 2)));
        kv_str(w, "path", rrc_census_combo_path(k));
        kv_bool(w, "native", rrc_census_combo_native(k));
        kv_int(w, "calls", sum.calls);
        kv_flops(w, "flops", sum.flops);
        kv_dbl(w, "flop_fraction", frac(sum.flops, tot));
        kv_u64(w, "convert_elems", sum.convert_elems);
        kv_u64(w, "convert_bytes", sum.convert_bytes);
        kv_int(w, "align_lt16_calls", sum.align_lt16_calls);
        rrc_jw_obj_end(w);
    }
    rrc_jw_arr_end(w);
    rrc_jw_key(w, "by_role");
    rrc_jw_arr_begin(w);
    for (int ro = 0; ro < RRC_CR_N; ro++) {
        rrc_jw_obj_begin(w);
        kv_str(w, "role", rrc_census_role_name(ro));
        int64_t calls = 0, nonnat = 0, a16 = 0;
        double fl = 0, fl16 = 0;
        uint64_t cb = 0;
        for (int k = 0; k < RRC_CENSUS_N_COMBOS; k++) {
            const rrc_census_cell *e = &r->cell[ro][k];
            calls += e->calls;
            fl += e->flops;
            cb += e->convert_bytes;
            a16 += e->align_lt16_calls;
            if (!rrc_census_combo_native(k)) nonnat += e->calls;
            if (rrc_census_combo_native_bf16(k)) fl16 += e->flops;
        }
        kv_int(w, "calls", calls);
        kv_flops(w, "flops", fl);
        kv_dbl(w, "flop_fraction", frac(fl, tot));
        kv_flops(w, "flops_native_bf16", fl16);
        kv_int(w, "non_native_calls", nonnat);
        kv_u64(w, "convert_bytes", cb);
        kv_int(w, "align_lt16_calls", a16);
        rrc_jw_key(w, "combos");
        rrc_jw_arr_begin(w);
        for (int k = 0; k < RRC_CENSUS_N_COMBOS; k++) {
            const rrc_census_cell *e = &r->cell[ro][k];
            if (!e->calls) continue;
            rrc_jw_obj_begin(w);
            char nm[32];
            snprintf(nm, sizeof nm, "%s %s %s", rrc_dtype_name(rrc_census_combo_dtype(k, 0)),
                     rrc_dtype_name(rrc_census_combo_dtype(k, 1)), rrc_dtype_name(rrc_census_combo_dtype(k, 2)));
            kv_str(w, "abc", nm);
            kv_str(w, "path", rrc_census_combo_path(k));
            kv_int(w, "calls", e->calls);
            kv_flops(w, "flops", e->flops);
            rrc_jw_obj_end(w);
        }
        rrc_jw_arr_end(w);
        rrc_jw_obj_end(w);
    }
    rrc_jw_arr_end(w);
    rrc_jw_key(w, "base_weight");
    rrc_jw_obj_begin(w);
    kv_int(w, "calls", base_calls);
    kv_int(w, "non_native_calls", base_nonnat);
    kv_flops(w, "flops", base);
    kv_dbl(w, "flop_fraction_of_total", frac(base, tot));
    kv_dbl(w, "native_bf16_fraction", frac(base_nat16, base));
    rrc_jw_obj_end(w);
    rrc_jw_key(w, "scratch");
    rrc_jw_obj_begin(w);
    kv_u64(w, "max_bytes", r->step.gemm_scratch_max_bytes);
    kv_u64(w, "convert_elems", r->step.gemm_convert_elems);
    kv_u64(w, "convert_bytes", r->step.gemm_convert_bytes);
    rrc_jw_obj_end(w);
    rrc_jw_obj_end(w);

    /* transfers and synchronisation */
    const rrc_dryrun_stats *s = &r->step;
    rrc_jw_key(w, "transfers");
    rrc_jw_obj_begin(w);
    kv_int(w, "sync", s->calls[RRC_DR_SYNC]);
    kv_int(w, "memcpy_h2d", s->calls[RRC_DR_H2D]);
    kv_u64(w, "h2d_bytes", s->h2d_bytes);
    kv_int(w, "memcpy_d2h", s->calls[RRC_DR_D2H]);
    kv_u64(w, "d2h_bytes", s->d2h_bytes);
    kv_int(w, "memcpy_d2d", s->calls[RRC_DR_D2D]);
    kv_int(w, "memset0", s->calls[RRC_DR_MEMSET0]);
    kv_u64(w, "memset_bytes", s->memset_bytes);
    kv_int(w, "alloc", s->calls[RRC_DR_ALLOC]);
    kv_int(w, "free", s->calls[RRC_DR_FREE]);
    /* rrc/cuda: memcpy_h2d and memcpy_d2h wait for the stream, sync does, free synchronizes first */
    kv_int(w, "cuda_host_waits",
           s->calls[RRC_DR_SYNC] + s->calls[RRC_DR_H2D] + s->calls[RRC_DR_D2H] + s->calls[RRC_DR_FREE]);
    kv_str(w, "cuda_host_waits_note", "sync + memcpy_h2d + memcpy_d2h + free: the calls on which the CUDA backend "
                                      "waits for its stream (DESIGN.md 11.5); scratch growth adds more, but not in a "
                                      "steady-state step");
    rrc_jw_obj_end(w);
    /* ops */
    rrc_jw_key(w, "ops");
    rrc_jw_arr_begin(w);
    for (int op = 0; op < RRC_DR_N_OPS; op++) {
        if (!s->calls[op]) continue;
        rrc_jw_obj_begin(w);
        kv_str(w, "op", rrc_dryrun_op_name(op));
        kv_int(w, "calls", s->calls[op]);
        kv_u64(w, "bytes", s->bytes[op]);
        rrc_jw_obj_end(w);
    }
    rrc_jw_arr_end(w);
    kv_str(w, "ops_bytes_note", "computed: one read of every input and one write of every output per call (caches, "
                                "re-reads and kernel-internal traffic not modelled)");
    /* memory */
    rrc_jw_key(w, "memory");
    rrc_jw_obj_begin(w);
    kv_u64(w, "weights_bytes", r->weights_bytes);
    kv_u64(w, "rope_tables_bytes", r->rope_bytes);
    kv_u64(w, "lora_params_grads_bytes", r->lora_bytes);
    kv_u64(w, "adam_bytes", r->adam_bytes);
    kv_u64(w, "workspace_bytes", r->workspace_bytes);
    kv_u64(w, "op_table_peak_bytes", r->op_peak_bytes);
    kv_u64(w, "gemm_scratch_bytes", r->gemm_scratch_bytes);
    kv_u64(w, "sum_squares_partials_bytes", r->sum_squares_partials_bytes);
    kv_u64(w, "cublaslt_workspace_bytes", r->cublaslt_workspace_bytes);
    kv_u64(w, "peak_total_bytes", r->peak_total_bytes);
    kv_dbl(w, "peak_total_gib", (double)r->peak_total_bytes / (double)(1ull << 30));
    kv_str(w, "not_included", "CUDA context, allocator rounding and fragmentation, host memory");
    rrc_jw_obj_end(w);
    rrc_jw_obj_end(w);
    size_t n;
    return rrc_jw_finish(w, out, &n);
}

/* ---- presets ---------------------------------------------------------------------------------------- */

static const rrc_census_preset k_presets[] = {
    {"qwen3-8b", "Qwen3-8B",
     "{\"architectures\": [\"Qwen3ForCausalLM\"], \"attention_bias\": false, \"attention_dropout\": 0.0, "
     "\"bos_token_id\": 151643, \"eos_token_id\": 151645, \"head_dim\": 128, \"hidden_act\": \"silu\", "
     "\"hidden_size\": 4096, \"initializer_range\": 0.02, \"intermediate_size\": 12288, "
     "\"max_position_embeddings\": 40960, \"max_window_layers\": 36, \"model_type\": \"qwen3\", "
     "\"num_attention_heads\": 32, \"num_hidden_layers\": 36, \"num_key_value_heads\": 8, \"rms_norm_eps\": 1e-06, "
     "\"rope_scaling\": null, \"rope_theta\": 1000000, \"sliding_window\": null, \"tie_word_embeddings\": false, "
     "\"torch_dtype\": \"bfloat16\", \"use_cache\": true, \"use_sliding_window\": false, \"vocab_size\": 151936}"},
    {"qwen2.5-7b", "Qwen2.5-7B",
     "{\"architectures\": [\"Qwen2ForCausalLM\"], \"attention_dropout\": 0.0, \"bos_token_id\": 151643, "
     "\"eos_token_id\": 151643, \"hidden_act\": \"silu\", \"hidden_size\": 3584, \"initializer_range\": 0.02, "
     "\"intermediate_size\": 18944, \"max_position_embeddings\": 131072, \"max_window_layers\": 28, "
     "\"model_type\": \"qwen2\", \"num_attention_heads\": 28, \"num_hidden_layers\": 28, \"num_key_value_heads\": 4, "
     "\"rms_norm_eps\": 1e-06, \"rope_theta\": 1000000.0, \"sliding_window\": 131072, \"tie_word_embeddings\": false, "
     "\"torch_dtype\": \"bfloat16\", \"use_cache\": true, \"use_sliding_window\": false, \"vocab_size\": 152064}"},
    {"llama-3.1-8b", "Llama-3.1-8B",
     "{\"architectures\": [\"LlamaForCausalLM\"], \"attention_bias\": false, \"attention_dropout\": 0.0, "
     "\"bos_token_id\": 128000, \"eos_token_id\": 128001, \"hidden_act\": \"silu\", \"hidden_size\": 4096, "
     "\"initializer_range\": 0.02, \"intermediate_size\": 14336, \"max_position_embeddings\": 131072, "
     "\"mlp_bias\": false, \"model_type\": \"llama\", \"num_attention_heads\": 32, \"num_hidden_layers\": 32, "
     "\"num_key_value_heads\": 8, \"pretraining_tp\": 1, \"rms_norm_eps\": 1e-05, \"rope_scaling\": {\"factor\": "
     "8.0, \"low_freq_factor\": 1.0, \"high_freq_factor\": 4.0, \"original_max_position_embeddings\": 8192, "
     "\"rope_type\": \"llama3\"}, \"rope_theta\": 500000.0, \"tie_word_embeddings\": false, \"torch_dtype\": "
     "\"bfloat16\", \"use_cache\": true, \"vocab_size\": 128256}"},
};

size_t rrc_census_num_presets(void) { return sizeof k_presets / sizeof k_presets[0]; }
const rrc_census_preset *rrc_census_preset_at(size_t i) { return i < rrc_census_num_presets() ? &k_presets[i] : NULL; }
const rrc_census_preset *rrc_census_preset_find(const char *name) {
    for (size_t i = 0; name && i < rrc_census_num_presets(); i++)
        if (!strcmp(name, k_presets[i].name)) return &k_presets[i];
    return NULL;
}
const char *rrc_census_preset_note(void) {
    return "ASSUMED shapes: config.json values of the public model card, typed in as recalled; NOT_TESTED against "
           "the real file (huggingface.co is blocked in the development environment). Pass the real config.json "
           "with --config for anything that matters.";
}

rrc_status rrc_census_preset_config(const rrc_census_preset *p, rrc_model_config *out) {
    RRC_REQUIRE(p && out, RRC_ERR_INVALID_ARG, "null argument");
    rrc_json_doc *doc = NULL;
    RRC_CHECK_CTX(rrc_json_parse(p->config_json, strlen(p->config_json), &doc), "preset %s", p->name);
    rrc_status st = rrc_config_from_json(rrc_json_root(doc), 0, out);
    rrc_json_doc_free(doc);
    if (st != RRC_OK) st = rrc_error_context(st, __FILE__, __LINE__, "preset %s", p->name);
    return st;
}
