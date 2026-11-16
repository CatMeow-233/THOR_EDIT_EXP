/* rrc/src/config.c — HF config.json parsing / validation, expected tensors, RoPE frequencies.
 * The rules are documented in include/rrc/config.h. */
#include "rrc/config.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

const char *rrc_family_name(rrc_family f) {
    switch (f) {
    case RRC_FAMILY_LLAMA: return "llama";
    case RRC_FAMILY_QWEN2: return "qwen2";
    case RRC_FAMILY_QWEN3: return "qwen3";
    }
    return "?";
}

/* Keys every family may carry that do not change the decoder-only forward/backward computation:
 * file metadata, token ids, generic PreTrainedConfig fields, and generation parameters (popped by
 * PreTrainedConfig.__init__ via GenerationConfig._get_default_generation_params). */
static const char *const k_ignored_keys[] = {
    /* metadata */
    "architectures", "transformers_version", "torch_dtype", "dtype", "_name_or_path", "name_or_path",
    "_commit_hash",
    /* token ids (pad_token_id only sets nn.Embedding.padding_idx, which affects the frozen
     * embedding's gradient only) */
    "bos_token_id", "eos_token_id", "pad_token_id", "sep_token_id", "decoder_start_token_id",
    /* runtime / initialisation flags */
    "use_cache", "initializer_range", "_attn_implementation", "attn_implementation", "_attn_implementation_autoset",
    "output_attentions", "output_hidden_states", "return_dict", "torchscript", "use_bfloat16",
    /* generic PreTrainedConfig fields irrelevant for a decoder-only causal LM */
    "chunk_size_feed_forward", "is_encoder_decoder", "is_decoder", "add_cross_attention",
    "cross_attention_hidden_size", "tie_encoder_decoder", "id2label", "label2id", "num_labels", "problem_type",
    "finetuning_task", "prefix", "task_specific_params", "tokenizer_class", "tf_legacy_loss",
    /* generation parameters */
    "assistant_confidence_threshold", "assistant_lookbehind", "bad_words_ids", "begin_suppress_tokens",
    "diversity_penalty", "do_sample", "early_stopping", "encoder_no_repeat_ngram_size", "encoder_repetition_penalty",
    "epsilon_cutoff", "eta_cutoff", "exponential_decay_length_penalty", "forced_bos_token_id", "forced_eos_token_id",
    "length_penalty", "max_length", "min_length", "no_repeat_ngram_size", "num_assistant_tokens",
    "num_assistant_tokens_schedule", "num_beam_groups", "num_beams", "num_return_sequences", "output_scores",
    "remove_invalid_values", "repetition_penalty", "return_dict_in_generate", "suppress_tokens", "target_lookbehind",
    "temperature", "top_k", "top_p", "typical_p",
};

/* Keys the parser reads (common to all families). */
static const char *const k_common_keys[] = {
    "model_type", "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "hidden_act", "max_position_embeddings", "rms_norm_eps",
    "tie_word_embeddings", "attention_dropout", "rope_scaling", "rope_parameters", "rope_theta",
    "partial_rotary_factor", "original_max_position_embeddings", "quantization_config", "pruned_heads",
};
static const char *const k_llama_keys[] = {"attention_bias", "mlp_bias", "pretraining_tp"};
/* qwen2: attention_bias / mlp_bias are not read by the qwen2 model (biases are fixed). */
/* qwen2: use_mrope (carried by some qwen2-architecture checkpoints derived from multimodal ones) is
 * not read by transformers' qwen2 model; false / null is accepted, true is refused. */
static const char *const k_qwen2_keys[] = {"use_sliding_window", "sliding_window", "max_window_layers", "layer_types",
                                           "attention_bias", "mlp_bias", "use_mrope"};
static const char *const k_qwen3_keys[] = {"use_sliding_window", "sliding_window", "max_window_layers", "layer_types",
                                           "attention_bias", "mlp_bias"};

static int in_list(const char *k, const char *const *list, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (!strcmp(k, list[i])) return 1;
    return 0;
}
#define IN_LIST(k, arr) in_list((k), (arr), sizeof(arr) / sizeof((arr)[0]))

#define UNSUP(...) RRC_ERROR(RRC_ERR_UNSUPPORTED, __VA_ARGS__)
#define BAD(...) RRC_ERROR(RRC_ERR_PARSE, __VA_ARGS__)

static int is_null_or_absent(const rrc_json *v) { return !v || v->type == RRC_JSON_NULL; }

/* Integer field: absent -> default; null or another type -> RRC_ERR_PARSE. */
static rrc_status get_int_field(const rrc_json *root, const char *key, int64_t def, int64_t *out) {
    const rrc_json *v = rrc_json_get(root, key);
    if (!v) {
        *out = def;
        return RRC_OK;
    }
    RRC_REQUIRE(v->type == RRC_JSON_INT, RRC_ERR_PARSE, "config.json: \"%s\" must be an integer, got %s", key,
                rrc_json_type_name(v->type));
    *out = v->u.i;
    return RRC_OK;
}

static rrc_status get_num_field(const rrc_json *root, const char *key, double def, double *out) {
    const rrc_json *v = rrc_json_get(root, key);
    if (!v) {
        *out = def;
        return RRC_OK;
    }
    RRC_REQUIRE(rrc_json_is_number(v), RRC_ERR_PARSE, "config.json: \"%s\" must be a number, got %s", key,
                rrc_json_type_name(v->type));
    *out = v->type == RRC_JSON_INT ? (double)v->u.i : v->u.d;
    return RRC_OK;
}

static rrc_status get_bool_field(const rrc_json *root, const char *key, int def, int *out) {
    const rrc_json *v = rrc_json_get(root, key);
    if (!v) {
        *out = def;
        return RRC_OK;
    }
    RRC_REQUIRE(v->type == RRC_JSON_BOOL, RRC_ERR_PARSE, "config.json: \"%s\" must be true/false, got %s", key,
                rrc_json_type_name(v->type));
    *out = v->u.boolean;
    return RRC_OK;
}

static rrc_status require_positive(const char *key, int64_t v) {
    RRC_REQUIRE(v > 0, RRC_ERR_UNSUPPORTED, "config.json: \"%s\" = %lld must be > 0", key, (long long)v);
    return RRC_OK;
}

/* upper bounds: config.json is untrusted input, and its sizes feed size_t / int64 products */
static rrc_status require_at_most(const char *key, int64_t v, int64_t hi) {
    RRC_REQUIRE(v <= hi, RRC_ERR_UNSUPPORTED, "config.json: \"%s\" = %lld is larger than the supported maximum %lld",
                key, (long long)v, (long long)hi);
    return RRC_OK;
}

static rrc_status parse_rope(const rrc_json *root, rrc_model_config *c) {
    const rrc_json *rs = rrc_json_get(root, "rope_scaling");
    const rrc_json *rp = rrc_json_get(root, "rope_parameters");
    if (rs && rs->type != RRC_JSON_NULL && rs->type != RRC_JSON_OBJECT)
        return BAD("config.json: \"rope_scaling\" must be an object or null");
    if (rp && rp->type != RRC_JSON_NULL && rp->type != RRC_JSON_OBJECT)
        return BAD("config.json: \"rope_parameters\" must be an object or null");
    int rs_set = rs && rs->type == RRC_JSON_OBJECT && rrc_json_obj_count(rs) > 0;
    int rp_set = rp && rp->type == RRC_JSON_OBJECT && rrc_json_obj_count(rp) > 0;
    if (rs_set && rp_set)
        return UNSUP("config.json: both \"rope_scaling\" and \"rope_parameters\" are set; refusing the ambiguous "
                     "config (transformers would use rope_scaling)");
    const rrc_json *d = rs_set ? rs : (rp_set ? rp : NULL);
    const char *dname = rs_set ? "rope_scaling" : "rope_parameters";

    /* rope_theta: dict value wins over the top-level key; default 10000 */
    double top_theta;
    RRC_CHECK(get_num_field(root, "rope_theta", 10000.0, &top_theta));
    c->rope_theta = top_theta;
    const char *rtype = "default";
    if (d) {
        const rrc_json *t = rrc_json_get(d, "rope_theta");
        if (t) RRC_CHECK(rrc_json_as_number(t, "rope_theta", &c->rope_theta));
        const rrc_json *rt = rrc_json_get(d, "rope_type");
        if (!rt) rt = rrc_json_get(d, "type");
        if (rt) RRC_CHECK(rrc_json_as_string(rt, "rope_type", &rtype));
    }
    RRC_REQUIRE(isfinite(c->rope_theta) && c->rope_theta > 0, RRC_ERR_UNSUPPORTED,
                "config.json: rope_theta = %g must be > 0", c->rope_theta);

    /* partial_rotary_factor (top level or in the dict) must be 1 */
    const rrc_json *prf_top = rrc_json_get(root, "partial_rotary_factor");
    const rrc_json *prf_d = d ? rrc_json_get(d, "partial_rotary_factor") : NULL;
    const rrc_json *prfs[2] = {prf_top, prf_d};
    for (int i = 0; i < 2; i++) {
        if (is_null_or_absent(prfs[i])) continue;
        double v;
        RRC_CHECK(rrc_json_as_number(prfs[i], "partial_rotary_factor", &v));
        if (v != 1.0) return UNSUP("config.json: partial_rotary_factor = %g is not supported (only 1.0)", v);
    }

    c->rope_attention_scaling = 1.0;
    c->rope_factor = 1.0;
    c->rope_low_freq_factor = 0.0;
    c->rope_high_freq_factor = 0.0;
    c->rope_original_max_position_embeddings = 0;
    if (!strcmp(rtype, "default")) {
        c->rope_type = RRC_ROPE_DEFAULT;
        for (size_t i = 0; d && i < rrc_json_obj_count(d); i++) {
            const char *k = rrc_json_obj_key(d, i);
            if (strcmp(k, "rope_theta") && strcmp(k, "rope_type") && strcmp(k, "type") &&
                strcmp(k, "partial_rotary_factor"))
                return UNSUP("config.json: %s key \"%s\" is not supported for rope_type \"default\"", dname, k);
        }
        return RRC_OK;
    }
    if (!strcmp(rtype, "llama3")) {
        c->rope_type = RRC_ROPE_LLAMA3;
        for (size_t i = 0; i < rrc_json_obj_count(d); i++) {
            const char *k = rrc_json_obj_key(d, i);
            if (strcmp(k, "rope_theta") && strcmp(k, "rope_type") && strcmp(k, "type") &&
                strcmp(k, "partial_rotary_factor") && strcmp(k, "factor") && strcmp(k, "low_freq_factor") &&
                strcmp(k, "high_freq_factor") && strcmp(k, "original_max_position_embeddings"))
                return UNSUP("config.json: %s key \"%s\" is not supported for rope_type \"llama3\"", dname, k);
        }
        const char *req[3] = {"factor", "low_freq_factor", "high_freq_factor"};
        double *dst[3] = {&c->rope_factor, &c->rope_low_freq_factor, &c->rope_high_freq_factor};
        for (int i = 0; i < 3; i++) {
            const rrc_json *v = rrc_json_get(d, req[i]);
            RRC_REQUIRE(v, RRC_ERR_PARSE, "config.json: %s (llama3) is missing \"%s\"", dname, req[i]);
            RRC_CHECK(rrc_json_as_number(v, req[i], dst[i]));
        }
        RRC_REQUIRE(c->rope_factor >= 1.0, RRC_ERR_UNSUPPORTED, "config.json: llama3 factor = %g must be >= 1",
                    c->rope_factor);
        RRC_REQUIRE(c->rope_high_freq_factor > c->rope_low_freq_factor && c->rope_low_freq_factor > 0,
                    RRC_ERR_UNSUPPORTED,
                    "config.json: llama3 needs high_freq_factor > low_freq_factor > 0 (got %g, %g)",
                    c->rope_high_freq_factor, c->rope_low_freq_factor);
        /* transformers: a top-level original_max_position_embeddings overrides the dict's value
         * (RotaryEmbeddingConfigMixin.standardize_rope_params, re-run at rope init); otherwise the
         * dict value; otherwise max_position_embeddings. */
        const rrc_json *top = rrc_json_get(root, "original_max_position_embeddings");
        const rrc_json *inner = rrc_json_get(d, "original_max_position_embeddings");
        int64_t omp = c->max_position_embeddings;
        if (!is_null_or_absent(top)) RRC_CHECK(rrc_json_as_int(top, "original_max_position_embeddings", &omp));
        else if (!is_null_or_absent(inner)) RRC_CHECK(rrc_json_as_int(inner, "original_max_position_embeddings", &omp));
        RRC_CHECK(require_positive("original_max_position_embeddings", omp));
        c->rope_original_max_position_embeddings = omp;
        return RRC_OK;
    }
    return UNSUP("config.json: rope_type \"%s\" is not supported (only \"default\" and \"llama3\")", rtype);
}

/* Number of base-model weight elements of `c` (the expected tensors of rrc_config_expected_tensors),
 * with every product and sum checked: RRC_ERR_SHAPE when it does not fit in int64. */
static rrc_status config_weight_elems(const rrc_model_config *c, int64_t *out) {
    const int64_t H = c->hidden_size, I = c->intermediate_size, V = c->vocab_size, D = c->head_dim;
    int64_t Q = 0, KV = 0, emb = 0, qo = 0, kv = 0, mlp = 0, per = 0, t = 0;
    int ov = 0;
#define MUL(a, b, r) (ov |= __builtin_mul_overflow((int64_t)(a), (int64_t)(b), (r)))
#define ADD(a, r) (ov |= __builtin_add_overflow(*(r), (int64_t)(a), (r)))
    MUL(c->num_heads, D, &Q);
    MUL(c->num_kv_heads, D, &KV);
    MUL(V, H, &emb);
    MUL(Q, H, &qo);
    MUL(KV, H, &kv);
    MUL(I, H, &mlp);
    /* per layer: q, o (Q x H), k, v (KV x H), gate, up, down (I x H), two norms (H), biases, q/k norms (D) */
    MUL(qo, 2, &per);
    MUL(kv, 2, &t);
    ADD(t, &per);
    MUL(mlp, 3, &t);
    ADD(t, &per);
    MUL(H, 2, &t);
    ADD(t, &per);
    if (c->qkv_bias) {
        ADD(Q, &per);
        ADD(KV, &per);
        ADD(KV, &per);
    }
    if (c->o_bias) ADD(H, &per);
    if (c->qk_norm) {
        ADD(D, &per);
        ADD(D, &per);
    }
    MUL(per, c->num_layers, &t);
    ADD(emb, &t);
    ADD(H, &t);
    if (!c->tie_word_embeddings) ADD(emb, &t);
#undef MUL
#undef ADD
    if (ov || Q <= 0 || KV <= 0)
        return RRC_ERROR(RRC_ERR_SHAPE,
                         "config: the base-model weight element count overflows (hidden %lld, intermediate %lld, "
                         "vocab %lld, layers %lld, heads %lld, kv heads %lld, head_dim %lld)",
                         (long long)H, (long long)I, (long long)V, (long long)c->num_layers, (long long)c->num_heads,
                         (long long)c->num_kv_heads, (long long)D);
    *out = t;
    return RRC_OK;
}

rrc_status rrc_config_from_json(const rrc_json *root, unsigned flags, rrc_model_config *out) {
    RRC_REQUIRE(root && out, RRC_ERR_INVALID_ARG, "null argument");
    RRC_REQUIRE(root->type == RRC_JSON_OBJECT, RRC_ERR_PARSE, "config.json: top level must be an object");
    memset(out, 0, sizeof *out);
    rrc_model_config *c = out;
    const char *mt = rrc_json_str(rrc_json_get(root, "model_type"));
    RRC_REQUIRE(mt, RRC_ERR_PARSE, "config.json: \"model_type\" (string) is required");
    if (!strcmp(mt, "llama")) c->family = RRC_FAMILY_LLAMA;
    else if (!strcmp(mt, "qwen2")) c->family = RRC_FAMILY_QWEN2;
    else if (!strcmp(mt, "qwen3")) c->family = RRC_FAMILY_QWEN3;
    else return UNSUP("config.json: model_type \"%s\" is not supported (llama, qwen2, qwen3)", mt);
    const int llama = c->family == RRC_FAMILY_LLAMA;
    const int qwen3 = c->family == RRC_FAMILY_QWEN3;

    /* unknown keys */
    for (size_t i = 0; i < rrc_json_obj_count(root); i++) {
        const char *k = rrc_json_obj_key(root, i);
        int known = IN_LIST(k, k_common_keys) || IN_LIST(k, k_ignored_keys) ||
                    (llama && IN_LIST(k, k_llama_keys)) ||
                    (c->family == RRC_FAMILY_QWEN2 && IN_LIST(k, k_qwen2_keys)) || (qwen3 && IN_LIST(k, k_qwen3_keys));
        if (!known) {
            if (flags & RRC_CONFIG_ALLOW_UNKNOWN_KEYS)
                rrc_log(RRC_LOG_WARN, "config.json: ignoring unknown key \"%s\" (%s)", k, mt);
            else
                return UNSUP("config.json: unknown key \"%s\" for model_type %s (refused: it may carry semantics the "
                             "engine does not implement; RRC_CONFIG_ALLOW_UNKNOWN_KEYS ignores it)",
                             k, mt);
        }
    }
    if (!is_null_or_absent(rrc_json_get(root, "quantization_config")))
        return UNSUP("config.json: quantized checkpoints (quantization_config) are not supported");
    const rrc_json *pruned = rrc_json_get(root, "pruned_heads");
    if (pruned && !(pruned->type == RRC_JSON_OBJECT && rrc_json_obj_count(pruned) == 0) &&
        pruned->type != RRC_JSON_NULL)
        return UNSUP("config.json: pruned_heads is not supported");

    RRC_CHECK(get_int_field(root, "vocab_size", llama ? 32000 : 151936, &c->vocab_size));
    RRC_CHECK(get_int_field(root, "hidden_size", 4096, &c->hidden_size));
    RRC_CHECK(get_int_field(root, "intermediate_size", llama ? 11008 : 22016, &c->intermediate_size));
    RRC_CHECK(get_int_field(root, "num_hidden_layers", 32, &c->num_layers));
    RRC_CHECK(get_int_field(root, "num_attention_heads", 32, &c->num_heads));
    RRC_CHECK(get_int_field(root, "max_position_embeddings", llama ? 2048 : 32768, &c->max_position_embeddings));
    RRC_CHECK(require_positive("vocab_size", c->vocab_size));
    RRC_CHECK(require_positive("hidden_size", c->hidden_size));
    RRC_CHECK(require_positive("intermediate_size", c->intermediate_size));
    RRC_CHECK(require_positive("num_hidden_layers", c->num_layers));
    RRC_CHECK(require_positive("num_attention_heads", c->num_heads));
    RRC_CHECK(require_positive("max_position_embeddings", c->max_position_embeddings));
    RRC_CHECK(require_at_most("vocab_size", c->vocab_size, RRC_CONFIG_MAX_DIM));
    RRC_CHECK(require_at_most("hidden_size", c->hidden_size, RRC_CONFIG_MAX_DIM));
    RRC_CHECK(require_at_most("intermediate_size", c->intermediate_size, RRC_CONFIG_MAX_DIM));
    RRC_CHECK(require_at_most("num_hidden_layers", c->num_layers, RRC_CONFIG_MAX_LAYERS));
    RRC_CHECK(require_at_most("num_attention_heads", c->num_heads, RRC_CONFIG_MAX_DIM));

    /* num_key_value_heads: absent -> heads (llama) / 32 (qwen); null -> heads */
    const rrc_json *kv = rrc_json_get(root, "num_key_value_heads");
    if (!kv) c->num_kv_heads = llama ? c->num_heads : 32;
    else if (kv->type == RRC_JSON_NULL) c->num_kv_heads = c->num_heads;
    else RRC_CHECK(rrc_json_as_int(kv, "num_key_value_heads", &c->num_kv_heads));
    RRC_CHECK(require_positive("num_key_value_heads", c->num_kv_heads));
    RRC_CHECK(require_at_most("num_key_value_heads", c->num_kv_heads, RRC_CONFIG_MAX_DIM));
    RRC_REQUIRE(c->num_heads % c->num_kv_heads == 0, RRC_ERR_UNSUPPORTED,
                "config.json: num_attention_heads (%lld) is not a multiple of num_key_value_heads (%lld)",
                (long long)c->num_heads, (long long)c->num_kv_heads);

    if (llama && c->hidden_size % c->num_heads != 0)
        return UNSUP("config.json: hidden_size (%lld) is not a multiple of num_attention_heads (%lld)",
                     (long long)c->hidden_size, (long long)c->num_heads);
    /* head_dim */
    const rrc_json *hd = rrc_json_get(root, "head_dim");
    if (!hd) {
        c->head_dim = qwen3 ? 128 : c->hidden_size / c->num_heads;
    } else if (hd->type == RRC_JSON_NULL) {
        if (!llama) return UNSUP("config.json: head_dim = null is not supported for %s", mt);
        c->head_dim = c->hidden_size / c->num_heads;
    } else {
        RRC_CHECK(rrc_json_as_int(hd, "head_dim", &c->head_dim));
    }
    RRC_CHECK(require_positive("head_dim", c->head_dim));
    RRC_CHECK(require_at_most("head_dim", c->head_dim, RRC_CONFIG_MAX_DIM));
    RRC_REQUIRE(c->head_dim % 2 == 0, RRC_ERR_UNSUPPORTED, "config.json: head_dim = %lld must be even for RoPE",
                (long long)c->head_dim);

    const rrc_json *act = rrc_json_get(root, "hidden_act");
    if (act) {
        const char *a;
        RRC_CHECK(rrc_json_as_string(act, "hidden_act", &a));
        if (strcmp(a, "silu")) return UNSUP("config.json: hidden_act \"%s\" is not supported (only silu)", a);
    }
    RRC_CHECK(get_num_field(root, "rms_norm_eps", 1e-6, &c->rms_norm_eps));
    RRC_REQUIRE(isfinite(c->rms_norm_eps) && c->rms_norm_eps > 0, RRC_ERR_UNSUPPORTED,
                "config.json: rms_norm_eps = %g must be > 0", c->rms_norm_eps);
    RRC_CHECK(get_bool_field(root, "tie_word_embeddings", 0, &c->tie_word_embeddings));
    double dropout;
    RRC_CHECK(get_num_field(root, "attention_dropout", 0.0, &dropout));
    if (dropout != 0.0) return UNSUP("config.json: attention_dropout = %g is not supported (only 0)", dropout);

    if (llama) {
        int ab, mb;
        RRC_CHECK(get_bool_field(root, "attention_bias", 0, &ab));
        RRC_CHECK(get_bool_field(root, "mlp_bias", 0, &mb));
        if (ab) return UNSUP("config.json: attention_bias = true is not supported for llama");
        if (mb) return UNSUP("config.json: mlp_bias = true is not supported for llama");
        const rrc_json *tp = rrc_json_get(root, "pretraining_tp");
        if (tp && tp->type != RRC_JSON_NULL && tp->type != RRC_JSON_INT)
            return BAD("config.json: pretraining_tp must be an integer");
        c->qkv_bias = c->o_bias = 0;
        c->qk_norm = 0;
    } else {
        if (qwen3) {
            int ab;
            RRC_CHECK(get_bool_field(root, "attention_bias", 0, &ab));
            c->qkv_bias = c->o_bias = ab;
            c->qk_norm = 1;
        } else {
            c->qkv_bias = 1;
            c->o_bias = 0;
            c->qk_norm = 0;
            const rrc_json *mr = rrc_json_get(root, "use_mrope");
            if (!is_null_or_absent(mr)) {
                int mrope;
                RRC_CHECK(get_bool_field(root, "use_mrope", 0, &mrope));
                if (mrope)
                    return UNSUP("config.json: use_mrope = true (multimodal RoPE) is not supported for qwen2 (only "
                                 "false; transformers' qwen2 model does not read the key)");
            }
        }
        /* sliding window: Qwen2Config/Qwen3Config.__post_init__ */
        int use_sw;
        RRC_CHECK(get_bool_field(root, "use_sliding_window", 0, &use_sw));
        const rrc_json *sw = rrc_json_get(root, "sliding_window");
        int64_t sliding = 4096;
        int sliding_null = 0;
        if (sw && sw->type == RRC_JSON_NULL) sliding_null = 1;
        else if (sw) RRC_CHECK(rrc_json_as_int(sw, "sliding_window", &sliding));
        int64_t mwl;
        const rrc_json *mw = rrc_json_get(root, "max_window_layers");
        if (mw && mw->type == RRC_JSON_NULL) return BAD("config.json: max_window_layers must be an integer");
        RRC_CHECK(get_int_field(root, "max_window_layers", 28, &mwl));
        int effective_sw = use_sw && !sliding_null;
        const rrc_json *lt = rrc_json_get(root, "layer_types");
        if (!is_null_or_absent(lt)) {
            if (lt->type != RRC_JSON_ARRAY) return BAD("config.json: layer_types must be a list");
            if ((int64_t)rrc_json_arr_count(lt) != c->num_layers)
                return BAD("config.json: layer_types has %zu entries, num_hidden_layers = %lld", rrc_json_arr_count(lt),
                           (long long)c->num_layers);
            for (size_t i = 0; i < rrc_json_arr_count(lt); i++) {
                const char *s = rrc_json_str(rrc_json_arr_get(lt, i));
                if (!s) return BAD("config.json: layer_types[%zu] is not a string", i);
                if (strcmp(s, "full_attention"))
                    return UNSUP("config.json: layer_types[%zu] = \"%s\" is not supported (sliding-window attention)", i,
                                 s);
            }
        } else if (effective_sw && mwl < c->num_layers) {
            return UNSUP("config.json: use_sliding_window with sliding_window = %lld makes layers >= %lld "
                         "sliding-window layers, which is not supported",
                         (long long)sliding, (long long)mwl);
        }
    }

    /* the whole base model (products and sums checked): every later size computation (weight
     * layout, byte counts, LoRA shapes) then stays far inside int64 */
    int64_t welems = 0;
    RRC_CHECK(config_weight_elems(c, &welems));
    RRC_REQUIRE(welems <= RRC_CONFIG_MAX_WEIGHT_ELEMS, RRC_ERR_UNSUPPORTED,
                "config.json: the base model has %lld weight elements, more than the supported %lld (2^40)",
                (long long)welems, (long long)RRC_CONFIG_MAX_WEIGHT_ELEMS);

    RRC_CHECK(parse_rope(root, c));

    const char *dt = rrc_json_str(rrc_json_get(root, "torch_dtype"));
    if (!dt) dt = rrc_json_str(rrc_json_get(root, "dtype"));
    if (dt) snprintf(c->torch_dtype, sizeof c->torch_dtype, "%s", dt);
    return RRC_OK;
}

rrc_status rrc_config_load(const char *path, unsigned flags, rrc_model_config *out) {
    RRC_REQUIRE(path && out, RRC_ERR_INVALID_ARG, "null argument");
    struct stat st;
    char buf[4096];
    const char *file = path;
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
        int n = snprintf(buf, sizeof buf, "%s/config.json", path);
        RRC_REQUIRE(n > 0 && (size_t)n < sizeof buf, RRC_ERR_INVALID_ARG, "path too long");
        file = buf;
    }
    rrc_json_doc *doc = NULL;
    RRC_CHECK(rrc_json_parse_file(file, 16u << 20, &doc));
    rrc_status s = rrc_config_from_json(rrc_json_root(doc), flags, out);
    rrc_json_doc_free(doc);
    if (s != RRC_OK) return rrc_error_context(s, __FILE__, __LINE__, "%s", file);
    return RRC_OK;
}

/* ------------------------------------------------------------------------------------------ */
/* RoPE                                                                                         */
/* ------------------------------------------------------------------------------------------ */

rrc_status rrc_rope_inv_freq(const rrc_model_config *c, double *out, int64_t n) {
    RRC_REQUIRE(c && out, RRC_ERR_INVALID_ARG, "null argument");
    RRC_REQUIRE(n == c->head_dim / 2, RRC_ERR_SHAPE, "inv_freq needs head_dim/2 = %lld values, got %lld",
                (long long)(c->head_dim / 2), (long long)n);
    const double dim = (double)c->head_dim;
    for (int64_t i = 0; i < n; i++) out[i] = 1.0 / pow(c->rope_theta, (double)(2 * i) / dim);
    if (c->rope_type == RRC_ROPE_LLAMA3) {
        const double factor = c->rope_factor, lo = c->rope_low_freq_factor, hi = c->rope_high_freq_factor;
        const double omp = (double)c->rope_original_max_position_embeddings;
        const double low_wl = omp / lo, high_wl = omp / hi;
        const double pi = 3.14159265358979323846;
        for (int64_t i = 0; i < n; i++) {
            double inv = out[i];
            double wl = 2.0 * pi / inv;
            double r;
            if (wl > low_wl) {
                r = inv / factor;
            } else if (wl < high_wl) {
                r = inv;
            } else {
                double smooth = (omp / wl - lo) / (hi - lo);
                r = (1.0 - smooth) * inv / factor + smooth * inv;
            }
            out[i] = r;
        }
    }
    return RRC_OK;
}

rrc_status rrc_rope_tables(const rrc_model_config *c, int64_t n_pos, float *cos_out, float *sin_out) {
    RRC_REQUIRE(c && cos_out && sin_out, RRC_ERR_INVALID_ARG, "null argument");
    RRC_REQUIRE(n_pos > 0, RRC_ERR_SHAPE, "n_pos must be > 0");
    const int64_t half = c->head_dim / 2;
    double *inv = (double *)malloc((size_t)half * sizeof(double));
    if (!inv) return RRC_ERROR(RRC_ERR_OOM, "inv_freq");
    rrc_status s = rrc_rope_inv_freq(c, inv, half);
    if (s != RRC_OK) {
        free(inv);
        return s;
    }
    const double scale = c->rope_attention_scaling;
#pragma omp parallel for schedule(static)
    for (int64_t p = 0; p < n_pos; p++) {
        for (int64_t i = 0; i < half; i++) {
            float invf = (float)inv[i];
            float angle = (float)p * invf;
            cos_out[p * half + i] = (float)(cos((double)angle) * scale);
            sin_out[p * half + i] = (float)(sin((double)angle) * scale);
        }
    }
    free(inv);
    return RRC_OK;
}

/* ------------------------------------------------------------------------------------------ */
/* Tensor names                                                                                 */
/* ------------------------------------------------------------------------------------------ */

static void spec(rrc_tensor_spec *s, size_t *k, const char *name, int ndim, int64_t d0, int64_t d1) {
    rrc_tensor_spec *t = &s[(*k)++];
    snprintf(t->name, sizeof t->name, "%s", name);
    t->ndim = ndim;
    t->shape[0] = d0;
    t->shape[1] = d1;
}

rrc_status rrc_config_expected_tensors(const rrc_model_config *c, rrc_tensor_spec **out, size_t *n) {
    RRC_REQUIRE(c && out && n, RRC_ERR_INVALID_ARG, "null argument");
    /* the config may not come from rrc_config_from_json: bound it here as well (the spec count and
     * every shape product below must not wrap) */
    RRC_REQUIRE(c->num_layers > 0 && c->num_layers <= RRC_CONFIG_MAX_LAYERS, RRC_ERR_SHAPE,
                "num_layers = %lld out of range [1, %d]", (long long)c->num_layers, RRC_CONFIG_MAX_LAYERS);
    int64_t welems = 0;
    RRC_CHECK(config_weight_elems(c, &welems));
    size_t per_layer = 2 + 4 + 3 + (c->qkv_bias ? 3 : 0) + (c->o_bias ? 1 : 0) + (c->qk_norm ? 2 : 0);
    size_t total = 0;
    if (__builtin_mul_overflow((size_t)c->num_layers, per_layer, &total) ||
        __builtin_add_overflow(total, (size_t)(2 + (c->tie_word_embeddings ? 0 : 1)), &total))
        return RRC_ERROR(RRC_ERR_SHAPE, "tensor spec count overflows");
    rrc_tensor_spec *s = (rrc_tensor_spec *)calloc(total, sizeof(rrc_tensor_spec));
    if (!s) return RRC_ERROR(RRC_ERR_OOM, "tensor specs");
    size_t k = 0;
    const int64_t H = c->hidden_size, I = c->intermediate_size, V = c->vocab_size;
    const int64_t Q = c->num_heads * c->head_dim, KV = c->num_kv_heads * c->head_dim;
    char nm[96];
    spec(s, &k, "model.embed_tokens.weight", 2, V, H);
    for (int64_t l = 0; l < c->num_layers; l++) {
#define LNAME(suffix) (snprintf(nm, sizeof nm, "model.layers.%lld.%s", (long long)l, suffix), nm)
        spec(s, &k, LNAME("input_layernorm.weight"), 1, H, 0);
        spec(s, &k, LNAME("self_attn.q_proj.weight"), 2, Q, H);
        if (c->qkv_bias) spec(s, &k, LNAME("self_attn.q_proj.bias"), 1, Q, 0);
        spec(s, &k, LNAME("self_attn.k_proj.weight"), 2, KV, H);
        if (c->qkv_bias) spec(s, &k, LNAME("self_attn.k_proj.bias"), 1, KV, 0);
        spec(s, &k, LNAME("self_attn.v_proj.weight"), 2, KV, H);
        if (c->qkv_bias) spec(s, &k, LNAME("self_attn.v_proj.bias"), 1, KV, 0);
        spec(s, &k, LNAME("self_attn.o_proj.weight"), 2, H, Q);
        if (c->o_bias) spec(s, &k, LNAME("self_attn.o_proj.bias"), 1, H, 0);
        if (c->qk_norm) {
            spec(s, &k, LNAME("self_attn.q_norm.weight"), 1, c->head_dim, 0);
            spec(s, &k, LNAME("self_attn.k_norm.weight"), 1, c->head_dim, 0);
        }
        spec(s, &k, LNAME("post_attention_layernorm.weight"), 1, H, 0);
        spec(s, &k, LNAME("mlp.gate_proj.weight"), 2, I, H);
        spec(s, &k, LNAME("mlp.up_proj.weight"), 2, I, H);
        spec(s, &k, LNAME("mlp.down_proj.weight"), 2, H, I);
#undef LNAME
    }
    spec(s, &k, "model.norm.weight", 1, H, 0);
    if (!c->tie_word_embeddings) spec(s, &k, "lm_head.weight", 2, V, H);
    if (k != total) {
        free(s);
        return RRC_ERROR(RRC_ERR_INTERNAL, "tensor spec count %zu != %zu", k, total);
    }
    *out = s;
    *n = total;
    return RRC_OK;
}

typedef struct {
    char buf[1024];
    size_t len;
    int count;
} name_list;

static void nl_add(name_list *l, const char *name) {
    l->count++;
    if (l->count > 8) return;
    int w = snprintf(l->buf + l->len, sizeof l->buf - l->len, "%s%s", l->len ? ", " : "", name);
    if (w > 0) l->len += (size_t)w < sizeof l->buf - l->len ? (size_t)w : sizeof l->buf - l->len - 1;
}

rrc_status rrc_config_check_tensors(const rrc_model_config *c, const rrc_st_set *set) {
    RRC_REQUIRE(c && set, RRC_ERR_INVALID_ARG, "null argument");
    rrc_tensor_spec *specs = NULL;
    size_t n = 0;
    RRC_CHECK(rrc_config_expected_tensors(c, &specs, &n));
    name_list missing = {{0}, 0, 0}, shape = {{0}, 0, 0}, dtype = {{0}, 0, 0}, extra = {{0}, 0, 0};
    for (size_t i = 0; i < n; i++) {
        const rrc_st_tensor *t = rrc_st_find(set, specs[i].name);
        if (!t) {
            nl_add(&missing, specs[i].name);
            continue;
        }
        int ok = t->ndim == specs[i].ndim;
        for (int d = 0; ok && d < t->ndim; d++) ok = t->shape[d] == specs[i].shape[d];
        if (!ok) {
            char msg[200];
            snprintf(msg, sizeof msg, "%s (got %dd [%lld,%lld], want [%lld%s%lld])", specs[i].name, t->ndim,
                     (long long)(t->ndim > 0 ? t->shape[0] : -1), (long long)(t->ndim > 1 ? t->shape[1] : -1),
                     (long long)specs[i].shape[0], specs[i].ndim > 1 ? "," : "",
                     specs[i].ndim > 1 ? (long long)specs[i].shape[1] : 0LL);
            nl_add(&shape, msg);
        }
        if (t->dtype != RRC_ST_F32 && t->dtype != RRC_ST_BF16) nl_add(&dtype, specs[i].name);
    }
    /* unexpected: set entries not in specs (both sorted differently; specs is small enough for a
     * linear check per tensor against a sorted copy of names) */
    size_t m = rrc_st_count(set);
    for (size_t j = 0; j < m; j++) {
        const char *name = rrc_st_at(set, j)->name;
        int found = 0;
        for (size_t i = 0; i < n && !found; i++) found = !strcmp(specs[i].name, name);
        if (!found) nl_add(&extra, name);
    }
    free(specs);
    if (missing.count)
        return RRC_ERROR(RRC_ERR_NOT_FOUND, "checkpoint is missing %d expected tensor(s): %s%s", missing.count,
                         missing.buf, missing.count > 8 ? ", ..." : "");
    if (extra.count)
        return RRC_ERROR(RRC_ERR_UNSUPPORTED, "checkpoint has %d unexpected tensor(s)%s: %s%s", extra.count,
                         c->tie_word_embeddings ? " (tie_word_embeddings = true: lm_head.weight must be absent)" : "",
                         extra.buf, extra.count > 8 ? ", ..." : "");
    if (shape.count)
        return RRC_ERROR(RRC_ERR_SHAPE, "%d tensor(s) with unexpected shape: %s%s", shape.count, shape.buf,
                         shape.count > 8 ? ", ..." : "");
    if (dtype.count)
        return RRC_ERROR(RRC_ERR_DTYPE, "%d tensor(s) not F32/BF16: %s%s", dtype.count, dtype.buf,
                         dtype.count > 8 ? ", ..." : "");
    return RRC_OK;
}
