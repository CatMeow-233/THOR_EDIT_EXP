/* rrc/src/model.c — load a HF checkpoint (llama / qwen2 / qwen3) into frozen base weights. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model_internal.h"
#include "rrc/safetensors.h"

void rrc_model_opts_default(rrc_model_opts *o) {
    memset(o, 0, sizeof *o);
    o->dtype = RRC_BF16;
    o->max_positions = 0;
    o->attn_chunk = 256;
    o->ce_chunk = 1024;
    o->recompute = RRC_RECOMPUTE_FULL;
    o->config_flags = 0;
}

static rrc_status check_exec(int64_t attn_chunk, int64_t ce_chunk, rrc_recompute rc) {
    RRC_REQUIRE(attn_chunk > 0 && attn_chunk <= INT32_MAX, RRC_ERR_INVALID_ARG, "attn_chunk = %lld must be in [1, 2^31)",
                (long long)attn_chunk);
    RRC_REQUIRE(ce_chunk > 0 && ce_chunk <= INT32_MAX, RRC_ERR_INVALID_ARG, "ce_chunk = %lld must be in [1, 2^31)",
                (long long)ce_chunk);
    RRC_REQUIRE(rc == RRC_RECOMPUTE_FULL || rc == RRC_RECOMPUTE_NONE, RRC_ERR_INVALID_ARG, "invalid recompute policy %d",
                (int)rc);
    return RRC_OK;
}

typedef struct load_ctx {
    rrc_model *m;
    const rrc_tensor_spec *specs;
    size_t n_specs;
    int64_t *offsets;
} load_ctx;

static rrc_buf weight_view(const load_ctx *c, const char *name) {
    for (size_t i = 0; i < c->n_specs; i++)
        if (!strcmp(c->specs[i].name, name)) {
            int64_t n = c->specs[i].shape[0] * (c->specs[i].ndim == 2 ? c->specs[i].shape[1] : 1);
            return rrc_view(c->m->storage, c->offsets[i], n);
        }
    return rrc_buf_none();
}

static rrc_buf layer_view(const load_ctx *c, int64_t l, const char *suffix) {
    char nm[128];
    snprintf(nm, sizeof nm, "model.layers.%lld.%s", (long long)l, suffix);
    return weight_view(c, nm);
}

static rrc_status upload_weights(load_ctx *c, const rrc_st_set *set) {
    rrc_model *m = c->m;
    int64_t max_n = 1;
    for (size_t i = 0; i < c->n_specs; i++) {
        int64_t n = c->specs[i].shape[0] * (c->specs[i].ndim == 2 ? c->specs[i].shape[1] : 1);
        if (n > max_n) max_n = n;
    }
    void *host = malloc((size_t)max_n * rrc_dtype_size(m->opts.dtype));
    if (!host) return RRC_ERROR(RRC_ERR_OOM, "weight staging buffer (%lld elements)", (long long)max_n);
    rrc_status st = RRC_OK;
    for (size_t i = 0; i < c->n_specs && st == RRC_OK; i++) {
        const rrc_st_tensor *t = rrc_st_find(set, c->specs[i].name);
        if (!t) {
            st = RRC_ERROR(RRC_ERR_INTERNAL, "tensor %s vanished after the tensor-set check", c->specs[i].name);
            break;
        }
        int64_t n = t->numel;
        if (m->opts.dtype == RRC_F32)
            st = rrc_st_to_f32(t, (float *)host, n);
        else
            st = rrc_st_to_bf16(t, (rrc_bf16 *)host, n);
        if (st == RRC_OK) st = rrc_buf_upload(rrc_view(m->storage, c->offsets[i], n), host, n);
        if (st != RRC_OK) st = rrc_error_context(st, __FILE__, __LINE__, "loading %s", c->specs[i].name);
    }
    free(host);
    return st;
}

rrc_status rrc_model_load(const rrc_backend *be, const char *dir, const rrc_model_opts *opts, rrc_model **out) {
    RRC_REQUIRE(be && dir && out, RRC_ERR_INVALID_ARG, "null argument");
    *out = NULL;
    rrc_model_opts o;
    if (opts)
        o = *opts;
    else
        rrc_model_opts_default(&o);
    RRC_REQUIRE(o.dtype == RRC_F32 || o.dtype == RRC_BF16, RRC_ERR_DTYPE, "model dtype must be f32 or bf16, got %s",
                rrc_dtype_name(o.dtype));
    RRC_CHECK(check_exec(o.attn_chunk, o.ce_chunk, o.recompute));
    RRC_REQUIRE(o.max_positions >= 0 && o.max_positions <= INT32_MAX, RRC_ERR_INVALID_ARG,
                "max_positions = %lld out of range", (long long)o.max_positions);

    rrc_model *m = (rrc_model *)calloc(1, sizeof *m);
    if (!m) return RRC_ERROR(RRC_ERR_OOM, "rrc_model");
    m->be = be;
    m->opts = o;
    m->storage = rrc_buf_none();
    m->rope_storage = rrc_buf_none();
    rrc_st_set *set = NULL;
    rrc_tensor_spec *specs = NULL;
    size_t n_specs = 0;
    int64_t *offsets = NULL;
    float *cs = NULL, *sn = NULL;
    rrc_status st;
    RRC_CHECK_GOTO(st, rrc_config_load(dir, o.config_flags, &m->cfg), fail);
    const rrc_model_config *cfg = &m->cfg;
    m->n_pos = o.max_positions > 0 ? o.max_positions : cfg->max_position_embeddings;
    RRC_CHECK_GOTO(st, rrc_st_open_checkpoint(dir, &set), fail);
    RRC_CHECK_GOTO(st, rrc_config_check_tensors(cfg, set), fail);
    RRC_CHECK_GOTO(st, rrc_config_expected_tensors(cfg, &specs, &n_specs), fail);
    offsets = (int64_t *)malloc(n_specs * sizeof(int64_t));
    if (!offsets) {
        st = RRC_ERROR(RRC_ERR_OOM, "weight offsets");
        goto fail;
    }
    /* All weights in one allocation, in the expected-tensor order, so gate_proj and up_proj of a
     * layer are adjacent and form the fused [2I, H] matrix. */
    int64_t total = 0;
    for (size_t i = 0; i < n_specs; i++) {
        int64_t n = specs[i].shape[0] * (specs[i].ndim == 2 ? specs[i].shape[1] : 1);
        offsets[i] = total;
        if (__builtin_add_overflow(total, n, &total)) {
            st = RRC_ERROR(RRC_ERR_SHAPE, "weight element count overflows");
            goto fail;
        }
    }
    RRC_CHECK_GOTO(st, rrc_buf_alloc(be, o.dtype, total, &m->storage), fail);
    load_ctx c = {m, specs, n_specs, offsets};
    RRC_CHECK_GOTO(st, upload_weights(&c, set), fail);

    m->layers = (rrc_layer_weights *)calloc((size_t)cfg->num_layers, sizeof(rrc_layer_weights));
    if (!m->layers) {
        st = RRC_ERROR(RRC_ERR_OOM, "layer table");
        goto fail;
    }
    m->embed = weight_view(&c, "model.embed_tokens.weight");
    m->norm = weight_view(&c, "model.norm.weight");
    m->lm_head = cfg->tie_word_embeddings ? m->embed : weight_view(&c, "lm_head.weight");
    for (int64_t l = 0; l < cfg->num_layers; l++) {
        rrc_layer_weights *w = &m->layers[l];
        w->ln1 = layer_view(&c, l, "input_layernorm.weight");
        w->ln2 = layer_view(&c, l, "post_attention_layernorm.weight");
        w->wq = layer_view(&c, l, "self_attn.q_proj.weight");
        w->wk = layer_view(&c, l, "self_attn.k_proj.weight");
        w->wv = layer_view(&c, l, "self_attn.v_proj.weight");
        w->wo = layer_view(&c, l, "self_attn.o_proj.weight");
        w->bq = layer_view(&c, l, "self_attn.q_proj.bias");
        w->bk = layer_view(&c, l, "self_attn.k_proj.bias");
        w->bv = layer_view(&c, l, "self_attn.v_proj.bias");
        w->bo = layer_view(&c, l, "self_attn.o_proj.bias");
        w->qn = layer_view(&c, l, "self_attn.q_norm.weight");
        w->kn = layer_view(&c, l, "self_attn.k_norm.weight");
        rrc_buf g = layer_view(&c, l, "mlp.gate_proj.weight"), u = layer_view(&c, l, "mlp.up_proj.weight");
        const int64_t gi = cfg->intermediate_size * cfg->hidden_size;
        if (g.numel != gi || u.numel != gi || (char *)u.ptr != (char *)g.ptr + (size_t)gi * rrc_dtype_size(o.dtype)) {
            st = RRC_ERROR(RRC_ERR_INTERNAL, "gate_proj / up_proj of layer %lld are not adjacent", (long long)l);
            goto fail;
        }
        w->wgu = rrc_view(m->storage, (int64_t)(((char *)g.ptr - (char *)m->storage.ptr) / rrc_dtype_size(o.dtype)),
                          2 * gi);
        w->wd = layer_view(&c, l, "mlp.down_proj.weight");
    }
    /* RoPE tables */
    const int64_t half = cfg->head_dim / 2;
    cs = (float *)malloc((size_t)(m->n_pos * half) * sizeof(float));
    sn = (float *)malloc((size_t)(m->n_pos * half) * sizeof(float));
    if (!cs || !sn) {
        st = RRC_ERROR(RRC_ERR_OOM, "rope tables");
        goto fail;
    }
    RRC_CHECK_GOTO(st, rrc_rope_tables(cfg, m->n_pos, cs, sn), fail);
    RRC_CHECK_GOTO(st, rrc_buf_alloc(be, RRC_F32, 2 * m->n_pos * half, &m->rope_storage), fail);
    m->cos_tab = rrc_view(m->rope_storage, 0, m->n_pos * half);
    m->sin_tab = rrc_view(m->rope_storage, m->n_pos * half, m->n_pos * half);
    RRC_CHECK_GOTO(st, rrc_buf_upload(m->cos_tab, cs, m->n_pos * half), fail);
    RRC_CHECK_GOTO(st, rrc_buf_upload(m->sin_tab, sn, m->n_pos * half), fail);
    free(cs);
    free(sn);
    free(offsets);
    free(specs);
    rrc_st_close(set);
    *out = m;
    return RRC_OK;
fail:
    free(cs);
    free(sn);
    free(offsets);
    free(specs);
    rrc_st_close(set);
    rrc_model_free(m);
    return st;
}

void rrc_model_free(rrc_model *m) {
    if (!m) return;
    if (m->ws.base) m->be->free(m->be, m->ws.base);
    free(m->cache_store);
    rrc_buf_free(&m->storage);
    rrc_buf_free(&m->rope_storage);
    free(m->layers);
    free(m);
}

const rrc_model_config *rrc_model_get_config(const rrc_model *m) { return m ? &m->cfg : NULL; }
const rrc_backend *rrc_model_backend(const rrc_model *m) { return m ? m->be : NULL; }
const rrc_model_opts *rrc_model_get_opts(const rrc_model *m) { return m ? &m->opts : NULL; }
size_t rrc_model_workspace_bytes(const rrc_model *m) { return m ? m->ws.bytes : 0; }
int64_t rrc_model_workspace_allocs(const rrc_model *m) { return m ? m->ws.allocs : 0; }

rrc_status rrc_model_set_exec(rrc_model *m, int64_t attn_chunk, int64_t ce_chunk, rrc_recompute recompute) {
    RRC_REQUIRE(m, RRC_ERR_INVALID_ARG, "null argument");
    RRC_CHECK(check_exec(attn_chunk, ce_chunk, recompute));
    m->opts.attn_chunk = attn_chunk;
    m->opts.ce_chunk = ce_chunk;
    m->opts.recompute = recompute;
    m->ws.N = -1; /* force a re-plan */
    return RRC_OK;
}
