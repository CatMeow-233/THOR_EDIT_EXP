/* rrc/config.h — Hugging Face config.json for llama / qwen2 / qwen3, validated against what the
 * engine implements, plus the expected checkpoint tensor set and RoPE inverse frequencies.
 *
 * Every default and rule below was derived from the transformers 5.23 sources
 * (models/{llama,qwen2,qwen3}/configuration_*.py + modeling_*.py, modeling_rope_utils.py,
 * configuration_utils.py) and is checked against transformers by rrc/tests/test_config_loading.py.
 *
 * Defaults when a key is absent (per family: llama | qwen2 | qwen3):
 *   vocab_size 32000 | 151936 | 151936; hidden_size 4096; intermediate_size 11008 | 22016 | 22016;
 *   num_hidden_layers 32; num_attention_heads 32; num_key_value_heads = num_attention_heads | 32 |
 *   32 (an explicit null means num_attention_heads in all families); head_dim = hidden_size /
 *   num_attention_heads | same | 128 (llama: null also means hidden/heads; qwen2/qwen3: null is
 *   refused); max_position_embeddings 2048 | 32768 | 32768; rms_norm_eps 1e-6; hidden_act "silu";
 *   tie_word_embeddings false; attention_dropout 0.
 * Biases: llama q/k/v/o = attention_bias (must be false), mlp_bias must be false; qwen2 q/k/v
 *   bias always, o never (attention_bias / mlp_bias keys are not read by the qwen2 model and are
 *   ignored); qwen3 q/k/v/o = attention_bias (default false, honoured), mlp_bias ignored.
 * Qwen3 has per-head RMSNorm q_norm / k_norm over head_dim (eps = rms_norm_eps) before RoPE.
 * llama additionally requires hidden_size % num_attention_heads == 0 (LlamaConfig validation).
 *
 * RoPE: the parameter dict is `rope_scaling` if it is a non-empty object, else `rope_parameters`
 *   (both present and non-empty is refused as ambiguous; HF would silently use rope_scaling).
 *   rope_theta = dict["rope_theta"] if present, else top-level "rope_theta", else 10000.
 *   rope_type = dict["rope_type"], else dict["type"] (legacy), else "default".
 *   Supported: "default" and "llama3" (factor, low_freq_factor, high_freq_factor required;
 *   original_max_position_embeddings = top-level key if present (HF gives it priority), else the
 *   dict's, else max_position_embeddings). partial_rotary_factor must be absent or 1.0. Any other
 *   key in the dict is refused. The attention scaling factor is 1.0 for both types.
 *
 * Refused values (RRC_ERR_UNSUPPORTED, message names the key): other model_type, hidden_act !=
 *   silu, llama attention_bias / mlp_bias true, attention_dropout != 0, any sliding-window layer
 *   (qwen2/qwen3: use_sliding_window with a non-null sliding_window and max_window_layers <
 *   num_hidden_layers, or any layer_types entry != "full_attention"), other rope types,
 *   partial_rotary_factor != 1, quantization_config, odd head_dim, num_attention_heads not a
 *   multiple of num_key_value_heads, non-positive sizes or eps, sizes above the upper bounds below.
 * Unknown top-level keys are refused (RRC_ERR_UNSUPPORTED) unless they are on the documented
 *   ignore list (metadata, token ids, generation parameters, ...), because an unknown key may
 *   carry semantics the engine does not implement; RRC_CONFIG_ALLOW_UNKNOWN_KEYS turns this into a
 *   logged warning.
 */
#ifndef RRC_CONFIG_H
#define RRC_CONFIG_H

#include <stdint.h>

#include "rrc/common.h"
#include "rrc/json.h"
#include "rrc/safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum rrc_family { RRC_FAMILY_LLAMA = 0, RRC_FAMILY_QWEN2 = 1, RRC_FAMILY_QWEN3 = 2 } rrc_family;
typedef enum rrc_rope_type { RRC_ROPE_DEFAULT = 0, RRC_ROPE_LLAMA3 = 1 } rrc_rope_type;

#define RRC_CONFIG_ALLOW_UNKNOWN_KEYS 1u

/* Upper bounds rrc_config_from_json enforces (RRC_ERR_UNSUPPORTED; config.json is untrusted input):
 * vocab_size, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads and head_dim
 * at most RRC_CONFIG_MAX_DIM; num_hidden_layers at most RRC_CONFIG_MAX_LAYERS (the LoRA layer
 * table, RRC_LORA_MAX_LAYERS); the base model's weight element count (every product checked) at
 * most RRC_CONFIG_MAX_WEIGHT_ELEMS. max_position_embeddings bounds no tensor and is not limited
 * here (the model refuses a RoPE table above 2^31 - 1 rows unless max_positions bounds it). */
#define RRC_CONFIG_MAX_DIM ((int64_t)2147483647)
#define RRC_CONFIG_MAX_LAYERS 1024
#define RRC_CONFIG_MAX_WEIGHT_ELEMS ((int64_t)1 << 40)

typedef struct rrc_model_config {
    rrc_family family;
    int64_t vocab_size;
    int64_t hidden_size;
    int64_t intermediate_size;
    int64_t num_layers;
    int64_t num_heads;
    int64_t num_kv_heads;
    int64_t head_dim;
    int64_t max_position_embeddings;
    double rms_norm_eps;
    int tie_word_embeddings;
    int qkv_bias; /* q/k/v projections have a bias */
    int o_bias;   /* o_proj has a bias */
    int qk_norm;  /* qwen3 per-head q_norm / k_norm */
    /* RoPE */
    rrc_rope_type rope_type;
    double rope_theta;
    double rope_factor;           /* llama3 only */
    double rope_low_freq_factor;  /* llama3 only */
    double rope_high_freq_factor; /* llama3 only */
    int64_t rope_original_max_position_embeddings; /* llama3 only */
    double rope_attention_scaling; /* multiplies cos/sin; 1.0 for default and llama3 */
    char torch_dtype[16];          /* informational ("bfloat16", "float32", "" if absent) */
} rrc_model_config;

const char *rrc_family_name(rrc_family f); /* "llama", "qwen2", "qwen3" */

rrc_status rrc_config_from_json(const rrc_json *root, unsigned flags, rrc_model_config *out);
/* `path` is a config.json file or a directory containing one. */
rrc_status rrc_config_load(const char *path, unsigned flags, rrc_model_config *out);

/* RoPE inverse frequencies in double precision, n = head_dim / 2 values:
 *   default: inv_freq[i] = 1 / theta^(2i / head_dim)
 *   llama3:  wavelen = 2*pi / inv_freq; low_wl = omp / low_freq_factor; high_wl = omp / high_freq_factor;
 *            wavelen > low_wl -> inv_freq / factor; wavelen < high_wl -> inv_freq; otherwise
 *            smooth = (omp / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor),
 *            (1 - smooth) * inv_freq / factor + smooth * inv_freq.
 * transformers evaluates the same formulas in float32, so its values differ by rounding only. */
rrc_status rrc_rope_inv_freq(const rrc_model_config *cfg, double *out, int64_t n);

/* cos/sin tables [n_pos, head_dim/2] (row p = position p) in f32, matching HF's evaluation order:
 * inv = (float)inv_freq_double; angle = (float)p * inv (an f32 product, like
 * `position_ids.float() * inv_freq`); cos/sin evaluated in double from that f32 angle and rounded
 * to f32, times rope_attention_scaling. These tables are the precomputed input of rope_fwd/bwd. */
rrc_status rrc_rope_tables(const rrc_model_config *cfg, int64_t n_pos, float *cos_out, float *sin_out);

/* Expected checkpoint tensors (HF names) with shapes, in a fixed order: embed_tokens, per layer
 * (input_layernorm, self_attn.{q,k,v,o}_proj.{weight,bias?}, self_attn.{q,k}_norm (qwen3),
 * post_attention_layernorm, mlp.{gate,up,down}_proj.weight), model.norm, lm_head (iff untied). */
typedef struct rrc_tensor_spec {
    char name[96];
    int ndim;
    int64_t shape[2];
} rrc_tensor_spec;

/* *out is malloc'd (caller frees). */
rrc_status rrc_config_expected_tensors(const rrc_model_config *cfg, rrc_tensor_spec **out, size_t *n);

/* The checkpoint's tensor set must equal the expected set exactly: missing tensors
 * (RRC_ERR_NOT_FOUND), unexpected tensors (RRC_ERR_UNSUPPORTED, e.g. lm_head.weight in a tied
 * model, rotary inv_freq buffers, extra heads) and shape mismatches (RRC_ERR_SHAPE) are errors;
 * float dtypes other than F32/BF16 (and any integer dtype) are RRC_ERR_DTYPE. The message lists
 * up to 8 offending names. */
rrc_status rrc_config_check_tensors(const rrc_model_config *cfg, const rrc_st_set *set);

#ifdef __cplusplus
}
#endif

#endif /* RRC_CONFIG_H */
