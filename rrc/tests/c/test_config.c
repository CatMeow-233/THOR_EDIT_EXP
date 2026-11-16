/* Tests for config.json parsing, refusal of unimplemented values, RoPE frequencies and the
 * expected-tensor checker. (Differential tests against transformers live in
 * rrc/tests/test_config_loading.py.) */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rrc/config.h"
#include "rt.h"

#define TINY "\"hidden_size\": 64, \"intermediate_size\": 128, \"num_hidden_layers\": 2, " \
             "\"num_attention_heads\": 4, \"num_key_value_heads\": 2, \"vocab_size\": 128"

static rrc_status parse_cfg(const char *json, unsigned flags, rrc_model_config *c) {
    rrc_json_doc *doc = NULL;
    rrc_status s = rrc_json_parse(json, strlen(json), &doc);
    if (s != RRC_OK) return s;
    s = rrc_config_from_json(rrc_json_root(doc), flags, c);
    rrc_json_doc_free(doc);
    return s;
}

RT_TEST(config_llama_defaults) {
    rrc_model_config c;
    RT_OK(parse_cfg("{\"model_type\": \"llama\"}", 0, &c));
    RT_EXPECT(c.family == RRC_FAMILY_LLAMA, "family");
    RT_EXPECT(c.vocab_size == 32000 && c.hidden_size == 4096 && c.intermediate_size == 11008, "sizes");
    RT_EXPECT(c.num_layers == 32 && c.num_heads == 32 && c.num_kv_heads == 32, "kv heads default = heads");
    RT_EXPECT(c.head_dim == 128 && c.max_position_embeddings == 2048, "head_dim %lld", (long long)c.head_dim);
    RT_EXPECT(c.rms_norm_eps == 1e-6 && !c.tie_word_embeddings, "eps/tie");
    RT_EXPECT(!c.qkv_bias && !c.o_bias && !c.qk_norm, "no bias / no qk norm");
    RT_EXPECT(c.rope_type == RRC_ROPE_DEFAULT && c.rope_theta == 10000.0 && c.rope_attention_scaling == 1.0, "rope");
    /* tiny file as written by save_pretrained in transformers 5.23 */
    RT_OK(parse_cfg("{\"attention_bias\": false, \"attention_dropout\": 0.0, \"bos_token_id\": 1, \"eos_token_id\": 2,"
                    " \"head_dim\": 16, \"hidden_act\": \"silu\", \"initializer_range\": 0.02, \"max_position_embeddings\":"
                    " 2048, \"mlp_bias\": false, \"model_type\": \"llama\", \"pad_token_id\": null, \"pretraining_tp\": 1,"
                    " \"rms_norm_eps\": 1e-05, \"rope_parameters\": {\"rope_theta\": 500000.0, \"rope_type\": \"default\"},"
                    " \"tie_word_embeddings\": true, \"transformers_version\": \"5.23.0\", \"use_cache\": true, "
                    "\"architectures\": [\"LlamaForCausalLM\"], \"dtype\": \"float32\", " TINY "}",
                    0, &c));
    RT_EXPECT(c.head_dim == 16 && c.num_kv_heads == 2 && c.rms_norm_eps == 1e-5 && c.tie_word_embeddings, "tiny");
    RT_EXPECT(c.rope_theta == 500000.0 && !strcmp(c.torch_dtype, "float32"), "theta / dtype");
    /* head_dim null -> hidden / heads; kv heads null -> heads */
    RT_OK(parse_cfg("{\"model_type\": \"llama\", \"head_dim\": null, \"hidden_size\": 96, \"num_attention_heads\": 6, "
                    "\"num_key_value_heads\": null}",
                    0, &c));
    RT_EXPECT(c.head_dim == 16 && c.num_kv_heads == 6, "nulls");
}

RT_TEST(config_qwen_defaults) {
    rrc_model_config c;
    RT_OK(parse_cfg("{\"model_type\": \"qwen2\", \"hidden_size\": 256, \"num_attention_heads\": 32}", 0, &c));
    RT_EXPECT(c.family == RRC_FAMILY_QWEN2 && c.vocab_size == 151936 && c.intermediate_size == 22016, "sizes");
    RT_EXPECT(c.num_kv_heads == 32, "qwen2 kv heads default is 32, got %lld", (long long)c.num_kv_heads);
    RT_STATUS(parse_cfg("{\"model_type\": \"qwen2\", \"hidden_size\": 64, \"num_attention_heads\": 8}", 0, &c),
              RRC_ERR_UNSUPPORTED); /* 8 % 32 != 0 */
    RT_OK(parse_cfg("{\"model_type\": \"qwen2\", " TINY "}", 0, &c));
    RT_EXPECT(c.head_dim == 16 && c.qkv_bias && !c.o_bias && !c.qk_norm, "qwen2 biases");
    RT_EXPECT(c.max_position_embeddings == 32768, "qwen2 max pos");
    /* attention_bias / mlp_bias keys are ignored for qwen2 (not read by the model) */
    RT_OK(parse_cfg("{\"model_type\": \"qwen2\", \"attention_bias\": false, \"mlp_bias\": true, " TINY "}", 0, &c));
    RT_EXPECT(c.qkv_bias && !c.o_bias, "qwen2 ignores attention_bias");
    /* qwen2 head_dim present in config.json is honoured */
    RT_OK(parse_cfg("{\"model_type\": \"qwen2\", \"head_dim\": 32, " TINY "}", 0, &c));
    RT_EXPECT(c.head_dim == 32, "qwen2 explicit head_dim");
    /* qwen3: head_dim defaults to 128 independent of hidden size; q/k norm; attention_bias honoured */
    RT_OK(parse_cfg("{\"model_type\": \"qwen3\", " TINY "}", 0, &c));
    RT_EXPECT(c.head_dim == 128 && c.qk_norm && !c.qkv_bias && !c.o_bias, "qwen3 defaults");
    RT_OK(parse_cfg("{\"model_type\": \"qwen3\", \"attention_bias\": true, \"head_dim\": 8, " TINY "}", 0, &c));
    RT_EXPECT(c.head_dim == 8 && c.qkv_bias && c.o_bias, "qwen3 attention_bias");
    /* sliding window configs that never produce sliding layers are accepted */
    RT_OK(parse_cfg("{\"model_type\": \"qwen2\", \"use_sliding_window\": false, \"sliding_window\": 4096, "
                    "\"max_window_layers\": 28, " TINY "}",
                    0, &c));
    RT_OK(parse_cfg("{\"model_type\": \"qwen2\", \"use_sliding_window\": true, \"sliding_window\": 4096, "
                    "\"max_window_layers\": 2, " TINY "}",
                    0, &c));
    RT_OK(parse_cfg("{\"model_type\": \"qwen2\", \"use_sliding_window\": true, \"sliding_window\": null, "
                    "\"max_window_layers\": 0, " TINY "}",
                    0, &c));
    RT_OK(parse_cfg("{\"model_type\": \"qwen3\", \"layer_types\": [\"full_attention\", \"full_attention\"], " TINY "}",
                    0, &c));
    RT_OK(parse_cfg("{\"model_type\": \"qwen2\", \"rope_scaling\": null, \"rope_theta\": 1000000.0, " TINY "}", 0, &c));
    RT_EXPECT(c.rope_theta == 1e6 && c.rope_type == RRC_ROPE_DEFAULT, "qwen legacy theta");
}

RT_TEST(config_rope_spellings) {
    rrc_model_config c;
    const char *l3 = "\"factor\": 8.0, \"low_freq_factor\": 1.0, \"high_freq_factor\": 4.0";
    char buf[1024];
    /* legacy rope_scaling with "type" and a top-level rope_theta */
    snprintf(buf, sizeof buf,
             "{\"model_type\": \"llama\", \"rope_theta\": 500000.0, \"rope_scaling\": {\"type\": \"llama3\", %s, "
             "\"original_max_position_embeddings\": 8192}, " TINY "}",
             l3);
    RT_OK(parse_cfg(buf, 0, &c));
    RT_EXPECT(c.rope_type == RRC_ROPE_LLAMA3 && c.rope_theta == 500000.0 && c.rope_factor == 8.0 &&
                  c.rope_low_freq_factor == 1.0 && c.rope_high_freq_factor == 4.0 &&
                  c.rope_original_max_position_embeddings == 8192,
              "legacy llama3");
    /* rope_parameters spelling; dict theta wins over top-level */
    snprintf(buf, sizeof buf,
             "{\"model_type\": \"llama\", \"rope_theta\": 1.0, \"rope_parameters\": {\"rope_type\": \"llama3\", "
             "\"rope_theta\": 500000.0, %s}, \"max_position_embeddings\": 4096, " TINY "}",
             l3);
    RT_OK(parse_cfg(buf, 0, &c));
    RT_EXPECT(c.rope_theta == 500000.0, "dict theta wins");
    RT_EXPECT(c.rope_original_max_position_embeddings == 4096, "omp defaults to max_position_embeddings");
    /* top-level original_max_position_embeddings has priority over the dict (transformers) */
    snprintf(buf, sizeof buf,
             "{\"model_type\": \"llama\", \"original_max_position_embeddings\": 2048, \"rope_scaling\": {\"rope_type\":"
             " \"llama3\", %s, \"original_max_position_embeddings\": 1024}, " TINY "}",
             l3);
    RT_OK(parse_cfg(buf, 0, &c));
    RT_EXPECT(c.rope_original_max_position_embeddings == 2048, "top-level omp priority");
    /* rope_type wins over type */
    RT_OK(parse_cfg("{\"model_type\": \"llama\", \"rope_scaling\": {\"type\": \"linear\", \"rope_type\": \"default\"}, "
                    TINY "}",
                    0, &c));
    RT_EXPECT(c.rope_type == RRC_ROPE_DEFAULT, "rope_type wins");
    /* empty rope_scaling falls back to rope_parameters / top-level theta */
    RT_OK(parse_cfg("{\"model_type\": \"llama\", \"rope_scaling\": {}, \"rope_theta\": 123.0, " TINY "}", 0, &c));
    RT_EXPECT(c.rope_theta == 123.0, "empty rope_scaling");
    RT_OK(parse_cfg("{\"model_type\": \"llama\", \"rope_parameters\": {\"rope_type\": \"default\"}, \"rope_theta\": "
                    "3000.0, " TINY "}",
                    0, &c));
    RT_EXPECT(c.rope_theta == 3000.0, "top-level theta fills the dict");
    RT_OK(parse_cfg("{\"model_type\": \"llama\", \"partial_rotary_factor\": 1.0, " TINY "}", 0, &c));
}

typedef struct {
    const char *json;
    rrc_status want;
    const char *needle; /* substring expected in the error message */
} refuse_case;

RT_TEST(config_refusals) {
    static const refuse_case cases[] = {
        {"{\"model_type\": \"mistral\"}", RRC_ERR_UNSUPPORTED, "model_type"},
        {"{\"hidden_size\": 64}", RRC_ERR_PARSE, "model_type"},
        {"[1]", RRC_ERR_PARSE, "object"},
        {"{\"model_type\": \"llama\", \"hidden_act\": \"gelu\"}", RRC_ERR_UNSUPPORTED, "hidden_act"},
        {"{\"model_type\": \"llama\", \"attention_bias\": true}", RRC_ERR_UNSUPPORTED, "attention_bias"},
        {"{\"model_type\": \"llama\", \"mlp_bias\": true}", RRC_ERR_UNSUPPORTED, "mlp_bias"},
        {"{\"model_type\": \"llama\", \"attention_dropout\": 0.1}", RRC_ERR_UNSUPPORTED, "attention_dropout"},
        {"{\"model_type\": \"qwen3\", \"attention_dropout\": 0.1}", RRC_ERR_UNSUPPORTED, "attention_dropout"},
        {"{\"model_type\": \"llama\", \"rope_scaling\": {\"rope_type\": \"linear\", \"factor\": 2.0}}",
         RRC_ERR_UNSUPPORTED, "linear"},
        {"{\"model_type\": \"llama\", \"rope_scaling\": {\"rope_type\": \"dynamic\", \"factor\": 2.0}}",
         RRC_ERR_UNSUPPORTED, "dynamic"},
        {"{\"model_type\": \"llama\", \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2.0, "
         "\"original_max_position_embeddings\": 1024}}",
         RRC_ERR_UNSUPPORTED, "yarn"},
        {"{\"model_type\": \"llama\", \"rope_scaling\": {\"type\": \"longrope\"}}", RRC_ERR_UNSUPPORTED, "longrope"},
        {"{\"model_type\": \"llama\", \"rope_scaling\": {\"rope_type\": \"llama3\", \"low_freq_factor\": 1.0, "
         "\"high_freq_factor\": 4.0}}",
         RRC_ERR_PARSE, "factor"},
        {"{\"model_type\": \"llama\", \"rope_scaling\": {\"rope_type\": \"llama3\", \"factor\": 8.0, "
         "\"low_freq_factor\": 4.0, \"high_freq_factor\": 1.0}}",
         RRC_ERR_UNSUPPORTED, "high_freq_factor"},
        {"{\"model_type\": \"llama\", \"rope_scaling\": {\"rope_type\": \"llama3\", \"factor\": 0.5, "
         "\"low_freq_factor\": 1.0, \"high_freq_factor\": 4.0}}",
         RRC_ERR_UNSUPPORTED, "factor"},
        {"{\"model_type\": \"llama\", \"rope_scaling\": {\"rope_type\": \"llama3\", \"factor\": 8.0, "
         "\"low_freq_factor\": 1.0, \"high_freq_factor\": 4.0, \"attention_factor\": 1.0}}",
         RRC_ERR_UNSUPPORTED, "attention_factor"},
        {"{\"model_type\": \"llama\", \"rope_scaling\": {\"rope_type\": \"default\", \"factor\": 2.0}}",
         RRC_ERR_UNSUPPORTED, "factor"},
        {"{\"model_type\": \"llama\", \"rope_scaling\": {\"rope_type\": \"default\"}, \"rope_parameters\": "
         "{\"rope_type\": \"default\"}}",
         RRC_ERR_UNSUPPORTED, "both"},
        {"{\"model_type\": \"llama\", \"rope_scaling\": 3}", RRC_ERR_PARSE, "rope_scaling"},
        {"{\"model_type\": \"llama\", \"rope_theta\": -1.0}", RRC_ERR_UNSUPPORTED, "rope_theta"},
        {"{\"model_type\": \"llama\", \"rope_theta\": \"10000\"}", RRC_ERR_PARSE, "rope_theta"},
        {"{\"model_type\": \"llama\", \"partial_rotary_factor\": 0.5}", RRC_ERR_UNSUPPORTED, "partial_rotary_factor"},
        {"{\"model_type\": \"llama\", \"rope_parameters\": {\"rope_type\": \"default\", \"partial_rotary_factor\": "
         "0.25}}",
         RRC_ERR_UNSUPPORTED, "partial_rotary_factor"},
        {"{\"model_type\": \"qwen2\", \"use_sliding_window\": true, \"sliding_window\": 4096, \"max_window_layers\": 1, "
         TINY "}",
         RRC_ERR_UNSUPPORTED, "sliding"},
        {"{\"model_type\": \"qwen2\", \"use_sliding_window\": true, " TINY ", \"max_window_layers\": 0}",
         RRC_ERR_UNSUPPORTED, "sliding"},
        {"{\"model_type\": \"qwen3\", \"layer_types\": [\"full_attention\", \"sliding_attention\"], " TINY "}",
         RRC_ERR_UNSUPPORTED, "layer_types"},
        {"{\"model_type\": \"qwen3\", \"layer_types\": [\"full_attention\"], " TINY "}", RRC_ERR_PARSE, "layer_types"},
        {"{\"model_type\": \"llama\", \"use_sliding_window\": false}", RRC_ERR_UNSUPPORTED, "use_sliding_window"},
        {"{\"model_type\": \"llama\", \"quantization_config\": {\"bits\": 4}}", RRC_ERR_UNSUPPORTED,
         "quantization_config"},
        {"{\"model_type\": \"llama\", \"pruned_heads\": {\"0\": [1]}}", RRC_ERR_UNSUPPORTED, "pruned_heads"},
        {"{\"model_type\": \"llama\", \"some_new_feature\": 1}", RRC_ERR_UNSUPPORTED, "some_new_feature"},
        {"{\"model_type\": \"llama\", \"head_dim\": 15, \"hidden_size\": 60, \"num_attention_heads\": 4}",
         RRC_ERR_UNSUPPORTED, "even"},
        {"{\"model_type\": \"llama\", \"hidden_size\": 65, \"num_attention_heads\": 4, \"head_dim\": 16}",
         RRC_ERR_UNSUPPORTED, "multiple"},
        {"{\"model_type\": \"llama\", \"num_attention_heads\": 6, \"num_key_value_heads\": 4, \"hidden_size\": 96}",
         RRC_ERR_UNSUPPORTED, "multiple"},
        {"{\"model_type\": \"qwen2\", \"head_dim\": null, " TINY "}", RRC_ERR_UNSUPPORTED, "head_dim"},
        {"{\"model_type\": \"llama\", \"hidden_size\": 0}", RRC_ERR_UNSUPPORTED, "hidden_size"},
        {"{\"model_type\": \"llama\", \"vocab_size\": -5}", RRC_ERR_UNSUPPORTED, "vocab_size"},
        {"{\"model_type\": \"llama\", \"num_hidden_layers\": 2.5}", RRC_ERR_PARSE, "num_hidden_layers"},
        {"{\"model_type\": \"llama\", \"rms_norm_eps\": 0}", RRC_ERR_UNSUPPORTED, "rms_norm_eps"},
        {"{\"model_type\": \"llama\", \"rms_norm_eps\": null}", RRC_ERR_PARSE, "rms_norm_eps"},
        {"{\"model_type\": \"llama\", \"tie_word_embeddings\": 1}", RRC_ERR_PARSE, "tie_word_embeddings"},
        {"{\"model_type\": \"llama\", \"num_key_value_heads\": 0}", RRC_ERR_UNSUPPORTED, "num_key_value_heads"},
        {"{\"model_type\": \"qwen3\", \"attention_bias\": \"yes\", " TINY "}", RRC_ERR_PARSE, "attention_bias"},
        {"{\"model_type\": \"qwen2\", \"max_window_layers\": null, " TINY "}", RRC_ERR_PARSE, "max_window_layers"},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        rrc_model_config c;
        rrc_clear_error();
        rrc_status s = parse_cfg(cases[i].json, 0, &c);
        RT_EXPECT(s == cases[i].want, "case %zu %s: got %s want %s (%s)", i, cases[i].json, rrc_status_str(s),
                  rrc_status_str(cases[i].want), rrc_last_error());
        RT_EXPECT(s == RRC_OK || strstr(rrc_last_error(), cases[i].needle), "case %zu: message lacks '%s': %s", i,
                  cases[i].needle, rrc_last_error());
    }
    /* unknown keys can be downgraded to warnings */
    rrc_model_config c;
    setenv("RRC_LOG_LEVEL", "error", 1);
    RT_OK(parse_cfg("{\"model_type\": \"llama\", \"some_new_feature\": 1}", RRC_CONFIG_ALLOW_UNKNOWN_KEYS, &c));
}

RT_TEST(config_inv_freq) {
    rrc_model_config c;
    RT_OK(parse_cfg("{\"model_type\": \"llama\", " TINY ", \"rope_theta\": 10000.0}", 0, &c));
    double inv[8];
    RT_OK(rrc_rope_inv_freq(&c, inv, 8));
    for (int i = 0; i < 8; i++) RT_CLOSE(inv[i], pow(10000.0, -(double)(2 * i) / 16.0), 0, 1e-15);
    RT_STATUS(rrc_rope_inv_freq(&c, inv, 7), RRC_ERR_SHAPE);
    /* Llama 3.1 rope: values from transformers' _compute_llama3_parameters (float32) */
    RT_OK(parse_cfg("{\"model_type\": \"llama\", \"hidden_size\": 4096, \"num_attention_heads\": 32, "
                    "\"max_position_embeddings\": 131072, \"rope_scaling\": {\"rope_type\": \"llama3\", \"factor\": 8.0,"
                    " \"low_freq_factor\": 1.0, \"high_freq_factor\": 4.0, \"original_max_position_embeddings\": 8192},"
                    " \"rope_theta\": 500000.0}",
                    0, &c));
    double v[64];
    RT_OK(rrc_rope_inv_freq(&c, v, 64));
    static const struct {
        int i;
        double hf;
    } ref[] = {{0, 1.0},
               {20, 0.016560440883040428},
               {30, 0.0013718936825171113},
               {40, 3.428102354519069e-05},
               {45, 1.2297638932068367e-05},
               {50, 4.411534519022098e-06},
               {63, 3.068925877869333e-07}};
    for (size_t k = 0; k < sizeof ref / sizeof ref[0]; k++)
        RT_EXPECT(rt_close(v[ref[k].i], ref[k].hf, 0, 2e-6), "llama3 inv_freq[%d] = %.17g, transformers %.17g",
                  ref[k].i, v[ref[k].i], ref[k].hf);
    /* tables: angle is an f32 product, cos/sin from that angle */
    float cs[3 * 64], sn[3 * 64];
    RT_OK(rrc_rope_tables(&c, 3, cs, sn));
    for (int p = 0; p < 3; p++)
        for (int i = 0; i < 64; i++) {
            float ang = (float)p * (float)v[i];
            RT_EXPECT(cs[p * 64 + i] == (float)cos((double)ang) && sn[p * 64 + i] == (float)sin((double)ang),
                      "table p=%d i=%d", p, i);
        }
    RT_EXPECT(cs[0] == 1.0f && sn[0] == 0.0f, "position 0");
}

/* Build a fake checkpoint for cfg with every expected tensor (zeros), then perturb it. */
static rrc_status write_ckpt(const rrc_model_config *c, const char *path, const char *drop, const char *add,
                             const char *reshape, int bf16) {
    rrc_tensor_spec *specs = NULL;
    size_t n = 0;
    RRC_CHECK(rrc_config_expected_tensors(c, &specs, &n));
    rrc_st_writer *w = NULL;
    rrc_status s = rrc_st_writer_create(&w);
    static float zeros[1 << 16];
    for (size_t i = 0; s == RRC_OK && i < n; i++) {
        if (drop && !strcmp(specs[i].name, drop)) continue;
        int64_t shape[2] = {specs[i].shape[0], specs[i].shape[1]};
        if (reshape && !strcmp(specs[i].name, reshape)) shape[0] += 1;
        s = rrc_st_writer_add(w, specs[i].name, bf16 ? RRC_BF16 : RRC_F32, specs[i].ndim, shape, zeros);
    }
    if (s == RRC_OK && add) {
        int64_t shape[1] = {4};
        s = rrc_st_writer_add(w, add, RRC_F32, 1, shape, zeros);
    }
    if (s == RRC_OK) s = rrc_st_writer_write(w, path);
    rrc_st_writer_free(w);
    free(specs);
    return s;
}

static rrc_status check_ckpt(const rrc_model_config *c, const char *path) {
    rrc_st_set *set = NULL;
    RRC_CHECK(rrc_st_open_file(path, &set));
    rrc_status s = rrc_config_check_tensors(c, set);
    rrc_st_close(set);
    return s;
}

RT_TEST(config_expected_tensors) {
    rrc_model_config c;
    RT_OK(parse_cfg("{\"model_type\": \"qwen3\", \"head_dim\": 8, \"attention_bias\": true, " TINY "}", 0, &c));
    rrc_tensor_spec *specs = NULL;
    size_t n = 0;
    RT_OK(rrc_config_expected_tensors(&c, &specs, &n));
    RT_EXPECT(n == 1 + 2 * 15 + 1 + 1, "qwen3 count %zu", n);
    RT_EXPECT(!strcmp(specs[0].name, "model.embed_tokens.weight") && specs[0].shape[0] == 128 && specs[0].shape[1] == 64,
              "embed");
    int found_qn = 0, found_ob = 0;
    for (size_t i = 0; i < n; i++) {
        if (!strcmp(specs[i].name, "model.layers.1.self_attn.q_norm.weight")) found_qn = specs[i].shape[0] == 8;
        if (!strcmp(specs[i].name, "model.layers.0.self_attn.o_proj.bias")) found_ob = specs[i].shape[0] == 64;
        if (!strcmp(specs[i].name, "model.layers.0.self_attn.q_proj.weight"))
            RT_EXPECT(specs[i].shape[0] == 32 && specs[i].shape[1] == 64, "q_proj shape uses head_dim 8");
        if (!strcmp(specs[i].name, "model.layers.0.self_attn.k_proj.weight"))
            RT_EXPECT(specs[i].shape[0] == 16 && specs[i].shape[1] == 64, "k_proj shape");
        if (!strcmp(specs[i].name, "model.layers.0.self_attn.o_proj.weight"))
            RT_EXPECT(specs[i].shape[0] == 64 && specs[i].shape[1] == 32, "o_proj shape");
    }
    RT_EXPECT(found_qn && found_ob, "q_norm / o_proj.bias");
    RT_EXPECT(!strcmp(specs[n - 1].name, "lm_head.weight"), "lm_head last when untied");
    free(specs);

    const char *p = rt_path("ckpt.safetensors");
    RT_OK(write_ckpt(&c, p, NULL, NULL, NULL, 0));
    RT_OK(check_ckpt(&c, p));
    RT_OK(write_ckpt(&c, p, NULL, NULL, NULL, 1));
    RT_OK(check_ckpt(&c, p)); /* bf16 checkpoint */
    RT_OK(write_ckpt(&c, p, "model.layers.1.mlp.up_proj.weight", NULL, NULL, 0));
    RT_STATUS(check_ckpt(&c, p), RRC_ERR_NOT_FOUND);
    RT_EXPECT(strstr(rrc_last_error(), "model.layers.1.mlp.up_proj.weight") != NULL, "%s", rrc_last_error());
    RT_OK(write_ckpt(&c, p, NULL, "model.layers.0.self_attn.rotary_emb.inv_freq", NULL, 0));
    RT_STATUS(check_ckpt(&c, p), RRC_ERR_UNSUPPORTED);
    RT_EXPECT(strstr(rrc_last_error(), "rotary_emb.inv_freq") != NULL, "%s", rrc_last_error());
    RT_OK(write_ckpt(&c, p, NULL, NULL, "model.layers.0.self_attn.k_proj.weight", 0));
    RT_STATUS(check_ckpt(&c, p), RRC_ERR_SHAPE);

    /* tied: lm_head.weight must be absent */
    RT_OK(parse_cfg("{\"model_type\": \"qwen2\", \"tie_word_embeddings\": true, " TINY "}", 0, &c));
    RT_OK(rrc_config_expected_tensors(&c, &specs, &n));
    RT_EXPECT(n == 1 + 2 * 12 + 1, "qwen2 tied count %zu", n);
    RT_EXPECT(strcmp(specs[n - 1].name, "lm_head.weight") != 0, "no lm_head when tied");
    free(specs);
    RT_OK(write_ckpt(&c, p, NULL, NULL, NULL, 0));
    RT_OK(check_ckpt(&c, p));
    RT_OK(write_ckpt(&c, p, NULL, "lm_head.weight", NULL, 0));
    RT_STATUS(check_ckpt(&c, p), RRC_ERR_UNSUPPORTED);
    RT_EXPECT(strstr(rrc_last_error(), "tie_word_embeddings") != NULL, "%s", rrc_last_error());
    /* untied llama without lm_head is missing it */
    RT_OK(parse_cfg("{\"model_type\": \"llama\", " TINY "}", 0, &c));
    RT_OK(write_ckpt(&c, p, "lm_head.weight", NULL, NULL, 0));
    RT_STATUS(check_ckpt(&c, p), RRC_ERR_NOT_FOUND);
    RT_EXPECT(strstr(rrc_last_error(), "lm_head.weight") != NULL, "%s", rrc_last_error());
}

RT_TEST(config_load_file) {
    const char *dir = rt_path("cfgdir");
    char cmd[2048];
    snprintf(cmd, sizeof cmd, "mkdir -p '%s'", dir);
    RT_ASSERT(system(cmd) == 0, "mkdir");
    char file[1400];
    snprintf(file, sizeof file, "%s/config.json", dir);
    const char *txt = "{\"model_type\": \"qwen3\", \"head_dim\": 16, " TINY "}";
    RT_OK(rrc_write_file_atomic(file, txt, strlen(txt)));
    rrc_model_config c;
    RT_OK(rrc_config_load(dir, 0, &c));
    RT_EXPECT(c.family == RRC_FAMILY_QWEN3 && c.head_dim == 16, "dir load");
    RT_OK(rrc_config_load(file, 0, &c));
    const char *bad = "{\"model_type\": \"qwen3\", \"hidden_act\": \"relu\"}";
    RT_OK(rrc_write_file_atomic(file, bad, strlen(bad)));
    RT_STATUS(rrc_config_load(dir, 0, &c), RRC_ERR_UNSUPPORTED);
    RT_EXPECT(strstr(rrc_last_error(), "config.json") != NULL, "path in message: %s", rrc_last_error());
}

const rt_case rt_suite_config[] = {
    RT_CASE(config_llama_defaults), RT_CASE(config_qwen_defaults),    RT_CASE(config_rope_spellings),
    RT_CASE(config_refusals),       RT_CASE(config_inv_freq),         RT_CASE(config_expected_tensors),
    RT_CASE(config_load_file),
};
const size_t rt_suite_config_n = sizeof rt_suite_config / sizeof rt_suite_config[0];
