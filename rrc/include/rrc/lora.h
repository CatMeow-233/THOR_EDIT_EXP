/* rrc/lora.h — LoRA adapters with PEFT semantics (rrc/docs/DESIGN.md §3, §10).
 *
 * On each targeted projection W [out, in] of a decoder layer:
 *   y = x W^T (+ b) + s * (x A^T) B^T,  A [r, in], B [out, r],  s = alpha / r  (rslora: alpha / sqrt(r))
 * The LoRA branch is computed in fp32 (PEFT casts the input to the fp32 adapter dtype). A, B and
 * their gradients are fp32 on the backend, in ONE contiguous parameter buffer and ONE contiguous
 * gradient buffer of the same layout (so the optimizer and the gradient norm are single op calls).
 * Layout: layers in increasing order, within a layer the modules in rrc_lora_module order, per
 * module A (r * in floats) then B (out * r floats); untargeted (layer, module) pairs take no space.
 *
 * Initialisation (PEFT init_lora_weights=True, "default"): A ~ U(-1/sqrt(in), 1/sqrt(in)), which
 * is torch's kaiming_uniform_(a = sqrt(5)) on [r, in] (gain sqrt(2 / (1 + 5)), bound
 * gain * sqrt(3 / in) = 1 / sqrt(in)); B = 0. The random numbers come from a documented
 * deterministic generator, NOT torch's: tensor number i (in the layout order above, counting A
 * tensors only) uses the splitmix64 sequence started at state
 *   s = splitmix64_mix(seed) ^ (0x9E3779B97F4A7C15 * (i + 1)),
 * each element takes the next 64-bit output x, u = (x >> 40) * 2^-24 in [0, 1), and
 * a = (float)(bound * (2u - 1)) (double arithmetic, rounded once). Tests load identical A/B into
 * rrc and PEFT instead of relying on equal initialisation.
 *
 * Tensor names are PEFT's saved-adapter keys (adapter_model.safetensors):
 *   base_model.model.model.layers.{l}.self_attn.{q,k,v,o}_proj.lora_{A,B}.weight
 *   base_model.model.model.layers.{l}.mlp.{gate,up,down}_proj.lora_{A,B}.weight
 */
#ifndef RRC_LORA_H
#define RRC_LORA_H

#include <stdint.h>

#include "rrc/common.h"
#include "rrc/config.h"
#include "rrc/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum rrc_lora_module {
    RRC_LORA_Q = 0,
    RRC_LORA_K = 1,
    RRC_LORA_V = 2,
    RRC_LORA_O = 3,
    RRC_LORA_GATE = 4,
    RRC_LORA_UP = 5,
    RRC_LORA_DOWN = 6,
    RRC_LORA_N_MODULES = 7
} rrc_lora_module;

/* "q_proj", "k_proj", ..., "down_proj" (NULL for an invalid value). */
const char *rrc_lora_module_name(int module);
/* Parse a comma-separated list of module names ("q_proj,v_proj", or "all-linear" for all seven)
 * into a bit mask (bit i = module i). Unknown names: RRC_ERR_UNSUPPORTED. */
rrc_status rrc_lora_parse_targets(const char *list, unsigned *mask_out);

#define RRC_LORA_ALL_MODULES ((1u << RRC_LORA_N_MODULES) - 1u)
#define RRC_LORA_MAX_LAYERS 1024

typedef struct rrc_lora_config {
    int64_t r;              /* rank, > 0 */
    double alpha;           /* lora_alpha, > 0 */
    int use_rslora;         /* s = alpha / sqrt(r) instead of alpha / r */
    double dropout;         /* must be 0 (phase 1): anything else is RRC_ERR_UNSUPPORTED */
    unsigned target_mask;   /* bit i = rrc_lora_module i; must be non-zero */
    /* PEFT layers_to_transform: n_layers == 0 means every layer; otherwise the listed layer
     * indices (each in [0, num_layers), no duplicates). */
    int64_t n_layers;
    int64_t layers[RRC_LORA_MAX_LAYERS];
    uint64_t seed; /* initialisation seed (see header comment) */
} rrc_lora_config;

/* r = 8, alpha = 16, no rslora, dropout 0, all seven modules, all layers, seed 0. */
void rrc_lora_config_default(rrc_lora_config *c);

typedef struct rrc_lora rrc_lora;

/* One LoRA tensor (A or B of one targeted module of one layer). */
typedef struct rrc_lora_tensor_info {
    char name[128];   /* PEFT saved-adapter key */
    int64_t layer;
    int module;       /* rrc_lora_module */
    int is_b;         /* 0: lora_A [r, in], 1: lora_B [out, r] */
    int64_t shape[2];
    int64_t offset;   /* element offset in the parameter / gradient buffers */
    int64_t numel;
} rrc_lora_tensor_info;

/* Create the adapter for a model config on backend `be`: validates `lc` against `cfg`, allocates
 * the fp32 parameter and gradient buffers, initialises A (seeded uniform) and B = 0, zero grads. */
rrc_status rrc_lora_create(const rrc_backend *be, const rrc_model_config *cfg, const rrc_lora_config *lc,
                           rrc_lora **out);
void rrc_lora_free(rrc_lora *l);

const rrc_lora_config *rrc_lora_get_config(const rrc_lora *l);
float rrc_lora_scaling(const rrc_lora *l);
/* Lowest layer index with at least one targeted module (num_layers if none). */
int64_t rrc_lora_lowest_layer(const rrc_lora *l);
int rrc_lora_targets(const rrc_lora *l, int64_t layer, int module);

size_t rrc_lora_num_tensors(const rrc_lora *l);
rrc_status rrc_lora_tensor(const rrc_lora *l, size_t i, rrc_lora_tensor_info *out);
/* Index of the tensor with PEFT name `name`, or RRC_ERR_NOT_FOUND. */
rrc_status rrc_lora_find(const rrc_lora *l, const char *name, size_t *index);

/* All parameters / all gradients (fp32, rrc_lora_num_params elements, layout above). Views. */
int64_t rrc_lora_num_params(const rrc_lora *l);
rrc_buf rrc_lora_params(const rrc_lora *l);
rrc_buf rrc_lora_grads(const rrc_lora *l);
/* A, B, dA, dB views of one (layer, module); RRC_ERR_NOT_FOUND if that module is not targeted. */
rrc_status rrc_lora_module_bufs(const rrc_lora *l, int64_t layer, int module, rrc_buf *a, rrc_buf *b, rrc_buf *ga,
                                rrc_buf *gb);

rrc_status rrc_lora_zero_grads(rrc_lora *l);

/* Set the parameters from a safetensors file with PEFT key names (e.g. PEFT's
 * adapter_model.safetensors): the file's tensor set must equal the adapter's exactly (missing:
 * RRC_ERR_NOT_FOUND, extra: RRC_ERR_UNSUPPORTED, shape: RRC_ERR_SHAPE); F32/BF16/F16 are
 * converted to fp32 exactly. */
rrc_status rrc_lora_load_safetensors(rrc_lora *l, const char *path);
/* Write the parameters (grads == 0) or the gradients (grads == 1) with PEFT key names, prefixed
 * with `prefix` (may be NULL or ""), into the writer (fp32). */
rrc_status rrc_lora_add_to_writer(const rrc_lora *l, int grads, const char *prefix, rrc_st_writer *w);

#ifdef __cplusplus
}
#endif

#endif /* RRC_LORA_H */
