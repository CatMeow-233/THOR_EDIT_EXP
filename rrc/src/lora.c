/* rrc/src/lora.c — LoRA adapter parameters: layout, PEFT names, deterministic init, I/O. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model_internal.h"
#include "rrc/backend.h"
#include "rrc/safetensors.h"

static const char *const k_module_names[RRC_LORA_N_MODULES] = {"q_proj",    "k_proj",  "v_proj",   "o_proj",
                                                               "gate_proj", "up_proj", "down_proj"};

const char *rrc_lora_module_name(int module) {
    if (module < 0 || module >= RRC_LORA_N_MODULES) return NULL;
    return k_module_names[module];
}

rrc_status rrc_lora_parse_targets(const char *list, unsigned *mask_out) {
    RRC_REQUIRE(list && mask_out, RRC_ERR_INVALID_ARG, "null argument");
    *mask_out = 0;
    if (!strcmp(list, "all-linear")) {
        *mask_out = RRC_LORA_ALL_MODULES;
        return RRC_OK;
    }
    unsigned mask = 0;
    const char *s = list;
    while (*s) {
        const char *e = strchr(s, ',');
        size_t len = e ? (size_t)(e - s) : strlen(s);
        int found = -1;
        for (int i = 0; i < RRC_LORA_N_MODULES; i++)
            if (strlen(k_module_names[i]) == len && !strncmp(s, k_module_names[i], len)) found = i;
        RRC_REQUIRE(found >= 0, RRC_ERR_UNSUPPORTED,
                    "LoRA target module '%.*s' is not supported (supported: q_proj, k_proj, v_proj, o_proj, gate_proj, "
                    "up_proj, down_proj, all-linear)",
                    (int)len, s);
        mask |= 1u << found;
        if (!e) break;
        s = e + 1;
        RRC_REQUIRE(*s, RRC_ERR_INVALID_ARG, "LoRA target list ends with ','");
    }
    RRC_REQUIRE(mask != 0, RRC_ERR_INVALID_ARG, "empty LoRA target list");
    *mask_out = mask;
    return RRC_OK;
}

void rrc_lora_config_default(rrc_lora_config *c) {
    memset(c, 0, sizeof *c);
    c->r = 8;
    c->alpha = 16.0;
    c->use_rslora = 0;
    c->dropout = 0.0;
    c->target_mask = RRC_LORA_ALL_MODULES;
    c->n_layers = 0;
    c->seed = 0;
}

/* in / out features of module `m` */
static void module_dims(const rrc_lora *l, int m, int64_t *in, int64_t *out) {
    switch (m) {
    case RRC_LORA_Q: *in = l->hidden, *out = l->q_dim; break;
    case RRC_LORA_K:
    case RRC_LORA_V: *in = l->hidden, *out = l->kv_dim; break;
    case RRC_LORA_O: *in = l->q_dim, *out = l->hidden; break;
    case RRC_LORA_GATE:
    case RRC_LORA_UP: *in = l->hidden, *out = l->inter; break;
    default: *in = l->inter, *out = l->hidden; break;
    }
}

static uint64_t splitmix64_next(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static uint64_t splitmix64_mix(uint64_t x) {
    uint64_t s = x;
    return splitmix64_next(&s);
}

static rrc_status lora_init_values(rrc_lora *l) {
    float *host = (float *)calloc((size_t)(l->n_params > 0 ? l->n_params : 1), sizeof(float));
    if (!host) return RRC_ERROR(RRC_ERR_OOM, "LoRA init host buffer (%lld floats)", (long long)l->n_params);
    uint64_t a_index = 0;
    const uint64_t base = splitmix64_mix(l->cfg.seed);
    for (int64_t layer = 0; layer < l->num_layers; layer++)
        for (int m = 0; m < RRC_LORA_N_MODULES; m++) {
            const rrc_lora_slot *s = rrc_lora_slot_at(l, layer, m);
            if (!s->present) continue;
            uint64_t st = base ^ (0x9E3779B97F4A7C15ull * (a_index + 1));
            a_index++;
            const double bound = 1.0 / sqrt((double)s->in);
            const int64_t na = l->cfg.r * s->in;
            for (int64_t i = 0; i < na; i++) {
                uint64_t x = splitmix64_next(&st);
                double u = (double)(x >> 40) * (1.0 / 16777216.0);
                host[s->off_a + i] = (float)(bound * (2.0 * u - 1.0));
            }
            /* B stays 0 (calloc) */
        }
    rrc_status st = rrc_buf_upload(l->params, host, l->n_params);
    free(host);
    return st;
}

static int64_t layer_targeted(const rrc_lora_config *c, int64_t layer) {
    if (c->n_layers == 0) return 1;
    for (int64_t i = 0; i < c->n_layers; i++)
        if (c->layers[i] == layer) return 1;
    return 0;
}

rrc_status rrc_lora_create(const rrc_backend *be, const rrc_model_config *cfg, const rrc_lora_config *lc,
                           rrc_lora **out) {
    RRC_REQUIRE(be && cfg && lc && out, RRC_ERR_INVALID_ARG, "null argument");
    *out = NULL;
    RRC_REQUIRE(lc->r > 0 && lc->r <= 65536, RRC_ERR_INVALID_ARG, "LoRA r = %lld must be in [1, 65536]",
                (long long)lc->r);
    RRC_REQUIRE(lc->alpha > 0.0 && isfinite(lc->alpha), RRC_ERR_INVALID_ARG, "LoRA alpha = %g must be > 0",
                lc->alpha);
    RRC_REQUIRE(lc->dropout == 0.0, RRC_ERR_UNSUPPORTED,
                "lora_dropout = %g is not supported (phase 1 implements lora_dropout = 0 only)", lc->dropout);
    RRC_REQUIRE(lc->target_mask != 0 && (lc->target_mask & ~RRC_LORA_ALL_MODULES) == 0, RRC_ERR_INVALID_ARG,
                "LoRA target mask 0x%x is empty or has unknown bits", lc->target_mask);
    RRC_REQUIRE(lc->n_layers >= 0 && lc->n_layers <= RRC_LORA_MAX_LAYERS, RRC_ERR_INVALID_ARG,
                "LoRA n_layers = %lld out of range", (long long)lc->n_layers);
    RRC_REQUIRE(cfg->num_layers > 0 && cfg->num_layers <= RRC_LORA_MAX_LAYERS, RRC_ERR_UNSUPPORTED,
                "num_layers = %lld exceeds the LoRA layer limit %d", (long long)cfg->num_layers, RRC_LORA_MAX_LAYERS);
    for (int64_t i = 0; i < lc->n_layers; i++) {
        RRC_REQUIRE(lc->layers[i] >= 0 && lc->layers[i] < cfg->num_layers, RRC_ERR_INVALID_ARG,
                    "LoRA layer %lld outside [0, %lld)", (long long)lc->layers[i], (long long)cfg->num_layers);
        for (int64_t j = 0; j < i; j++)
            RRC_REQUIRE(lc->layers[j] != lc->layers[i], RRC_ERR_INVALID_ARG, "LoRA layer %lld listed twice",
                        (long long)lc->layers[i]);
    }
    rrc_lora *l = (rrc_lora *)calloc(1, sizeof *l);
    if (!l) return RRC_ERROR(RRC_ERR_OOM, "rrc_lora");
    l->be = be;
    l->cfg = *lc;
    l->num_layers = cfg->num_layers;
    l->hidden = cfg->hidden_size;
    l->inter = cfg->intermediate_size;
    l->q_dim = cfg->num_heads * cfg->head_dim;
    l->kv_dim = cfg->num_kv_heads * cfg->head_dim;
    l->scaling = (float)(lc->use_rslora ? lc->alpha / sqrt((double)lc->r) : lc->alpha / (double)lc->r);
    l->params = rrc_buf_none();
    l->grads = rrc_buf_none();
    l->slots = (rrc_lora_slot *)calloc((size_t)(l->num_layers * RRC_LORA_N_MODULES), sizeof(rrc_lora_slot));
    l->tensors =
        (rrc_lora_tensor_info *)calloc((size_t)(l->num_layers * RRC_LORA_N_MODULES * 2), sizeof(rrc_lora_tensor_info));
    rrc_status st = RRC_OK;
    if (!l->slots || !l->tensors) {
        st = RRC_ERROR(RRC_ERR_OOM, "LoRA tables");
        goto fail;
    }
    int64_t off = 0;
    l->lowest_layer = l->num_layers;
    for (int64_t layer = 0; layer < l->num_layers; layer++) {
        if (!layer_targeted(lc, layer)) continue;
        for (int m = 0; m < RRC_LORA_N_MODULES; m++) {
            if (!(lc->target_mask & (1u << m))) continue;
            rrc_lora_slot *s = &l->slots[layer * RRC_LORA_N_MODULES + m];
            s->present = 1;
            module_dims(l, m, &s->in, &s->out);
            s->off_a = off;
            off += lc->r * s->in;
            s->off_b = off;
            off += s->out * lc->r;
            if (layer < l->lowest_layer) l->lowest_layer = layer;
            const char *grp = m <= RRC_LORA_O ? "self_attn" : "mlp";
            for (int ab = 0; ab < 2; ab++) {
                rrc_lora_tensor_info *t = &l->tensors[l->n_tensors++];
                snprintf(t->name, sizeof t->name, "base_model.model.model.layers.%lld.%s.%s.lora_%c.weight",
                         (long long)layer, grp, k_module_names[m], ab ? 'B' : 'A');
                t->layer = layer;
                t->module = m;
                t->is_b = ab;
                t->shape[0] = ab ? s->out : lc->r;
                t->shape[1] = ab ? lc->r : s->in;
                t->offset = ab ? s->off_b : s->off_a;
                t->numel = t->shape[0] * t->shape[1];
            }
        }
    }
    l->n_params = off;
    RRC_CHECK_GOTO(st, rrc_buf_alloc(be, RRC_F32, off, &l->params), fail);
    RRC_CHECK_GOTO(st, rrc_buf_alloc(be, RRC_F32, off, &l->grads), fail);
    RRC_CHECK_GOTO(st, lora_init_values(l), fail);
    *out = l;
    return RRC_OK;
fail:
    rrc_lora_free(l);
    return st;
}

void rrc_lora_free(rrc_lora *l) {
    if (!l) return;
    rrc_buf_free(&l->params);
    rrc_buf_free(&l->grads);
    free(l->slots);
    free(l->tensors);
    free(l);
}

const rrc_lora_config *rrc_lora_get_config(const rrc_lora *l) { return l ? &l->cfg : NULL; }
float rrc_lora_scaling(const rrc_lora *l) { return l ? l->scaling : 0.0f; }
int64_t rrc_lora_lowest_layer(const rrc_lora *l) { return l ? l->lowest_layer : 0; }
int rrc_lora_targets(const rrc_lora *l, int64_t layer, int module) {
    if (!l || layer < 0 || layer >= l->num_layers || module < 0 || module >= RRC_LORA_N_MODULES) return 0;
    return rrc_lora_slot_at(l, layer, module)->present;
}
size_t rrc_lora_num_tensors(const rrc_lora *l) { return l ? l->n_tensors : 0; }

rrc_status rrc_lora_tensor(const rrc_lora *l, size_t i, rrc_lora_tensor_info *out) {
    RRC_REQUIRE(l && out, RRC_ERR_INVALID_ARG, "null argument");
    RRC_REQUIRE(i < l->n_tensors, RRC_ERR_RANGE, "LoRA tensor index %zu >= %zu", i, l->n_tensors);
    *out = l->tensors[i];
    return RRC_OK;
}

rrc_status rrc_lora_find(const rrc_lora *l, const char *name, size_t *index) {
    RRC_REQUIRE(l && name && index, RRC_ERR_INVALID_ARG, "null argument");
    for (size_t i = 0; i < l->n_tensors; i++)
        if (!strcmp(l->tensors[i].name, name)) {
            *index = i;
            return RRC_OK;
        }
    return RRC_ERROR(RRC_ERR_NOT_FOUND, "no LoRA tensor named '%s'", name);
}

int64_t rrc_lora_num_params(const rrc_lora *l) { return l ? l->n_params : 0; }
rrc_buf rrc_lora_params(const rrc_lora *l) { return l ? rrc_view(l->params, 0, l->n_params) : rrc_buf_none(); }
rrc_buf rrc_lora_grads(const rrc_lora *l) { return l ? rrc_view(l->grads, 0, l->n_params) : rrc_buf_none(); }

rrc_status rrc_lora_module_bufs(const rrc_lora *l, int64_t layer, int module, rrc_buf *a, rrc_buf *b, rrc_buf *ga,
                                rrc_buf *gb) {
    RRC_REQUIRE(l, RRC_ERR_INVALID_ARG, "null argument");
    RRC_REQUIRE(rrc_lora_targets(l, layer, module), RRC_ERR_NOT_FOUND, "layer %lld module %d has no LoRA",
                (long long)layer, module);
    const rrc_lora_slot *s = rrc_lora_slot_at(l, layer, module);
    const int64_t na = l->cfg.r * s->in, nb = s->out * l->cfg.r;
    if (a) *a = rrc_view(l->params, s->off_a, na);
    if (b) *b = rrc_view(l->params, s->off_b, nb);
    if (ga) *ga = rrc_view(l->grads, s->off_a, na);
    if (gb) *gb = rrc_view(l->grads, s->off_b, nb);
    return RRC_OK;
}

rrc_status rrc_lora_zero_grads(rrc_lora *l) {
    RRC_REQUIRE(l, RRC_ERR_INVALID_ARG, "null argument");
    return rrc_buf_zero(l->grads, l->n_params);
}

rrc_status rrc_lora_load_safetensors(rrc_lora *l, const char *path) {
    RRC_REQUIRE(l && path, RRC_ERR_INVALID_ARG, "null argument");
    rrc_st_set *set = NULL;
    RRC_CHECK_CTX(rrc_st_open_checkpoint(path, &set), "LoRA weights '%s'", path);
    rrc_status st = RRC_OK;
    float *host = NULL;
    /* exact tensor set: every file tensor must be an adapter tensor, every adapter tensor present */
    for (size_t i = 0; i < rrc_st_count(set); i++) {
        const rrc_st_tensor *t = rrc_st_at(set, i);
        size_t idx;
        if (rrc_lora_find(l, t->name, &idx) != RRC_OK) {
            st = RRC_ERROR(RRC_ERR_UNSUPPORTED, "LoRA weights '%s': tensor '%s' is not part of this adapter (targets / "
                                                "layers / names differ)",
                           path, t->name);
            goto done;
        }
    }
    host = (float *)malloc((size_t)(l->n_params > 0 ? l->n_params : 1) * sizeof(float));
    if (!host) {
        st = RRC_ERROR(RRC_ERR_OOM, "LoRA host staging");
        goto done;
    }
    for (size_t i = 0; i < l->n_tensors; i++) {
        const rrc_lora_tensor_info *ti = &l->tensors[i];
        const rrc_st_tensor *t = rrc_st_find(set, ti->name);
        if (!t) {
            st = RRC_ERROR(RRC_ERR_NOT_FOUND, "LoRA weights '%s': tensor '%s' missing", path, ti->name);
            goto done;
        }
        if (t->ndim != 2 || t->shape[0] != ti->shape[0] || t->shape[1] != ti->shape[1]) {
            st = RRC_ERROR(RRC_ERR_SHAPE, "LoRA weights '%s': tensor '%s' has shape [%lld, %lld] (ndim %d), expected "
                                          "[%lld, %lld]",
                           path, ti->name, t->ndim > 0 ? (long long)t->shape[0] : -1LL,
                           t->ndim > 1 ? (long long)t->shape[1] : -1LL, t->ndim, (long long)ti->shape[0],
                           (long long)ti->shape[1]);
            goto done;
        }
        RRC_CHECK_GOTO(st, rrc_st_to_f32(t, host + ti->offset, ti->numel), done);
    }
    RRC_CHECK_GOTO(st, rrc_buf_upload(l->params, host, l->n_params), done);
done:
    free(host);
    rrc_st_close(set);
    return st;
}

rrc_status rrc_lora_add_to_writer(const rrc_lora *l, int grads, const char *prefix, rrc_st_writer *w) {
    RRC_REQUIRE(l && w, RRC_ERR_INVALID_ARG, "null argument");
    float *host = (float *)malloc((size_t)(l->n_params > 0 ? l->n_params : 1) * sizeof(float));
    if (!host) return RRC_ERROR(RRC_ERR_OOM, "LoRA host staging");
    rrc_status st = rrc_buf_download(grads ? l->grads : l->params, host, l->n_params);
    for (size_t i = 0; st == RRC_OK && i < l->n_tensors; i++) {
        const rrc_lora_tensor_info *ti = &l->tensors[i];
        char name[256];
        snprintf(name, sizeof name, "%s%s", prefix ? prefix : "", ti->name);
        st = rrc_st_writer_add(w, name, RRC_F32, 2, ti->shape, host + ti->offset);
    }
    free(host);
    return st;
}
