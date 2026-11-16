/* rrc_inspect — load a Hugging Face checkpoint directory (config.json + safetensors) the way the
 * engine does and print what was derived as JSON on stdout. Used by rrc/tests/test_config_loading.py
 * to compare the C loader with transformers.
 *
 *   rrc_inspect [--config-only] [--allow-unknown-keys] [--positions p0,p1,...] <model_dir>
 *
 * Output (success, exit 0):
 *   {"ok": true, "config": {...derived fields...}, "inv_freq": [...], "inv_freq_sum": x,
 *    "rope": {"positions": [...], "cos": [[...]], "sin": [[...]]},
 *    "expected_tensors": [{"name", "shape"}],
 *    "tensors": [{"name", "dtype", "shape", "sum"}],          (unless --config-only)
 *    "check": "ok" | "<error message>", "check_status": "RRC_OK" | "RRC_ERR_..."}
 * Failure to load the config (exit 2): {"ok": false, "status": "RRC_ERR_...", "error": "..."}
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rrc/rrc.h"

static int fail(rrc_status st) {
    rrc_jw w;
    rrc_jw_init(&w, 0);
    rrc_jw_obj_begin(&w);
    rrc_jw_key(&w, "ok");
    rrc_jw_bool(&w, 0);
    rrc_jw_key(&w, "status");
    rrc_jw_str(&w, rrc_status_str(st));
    rrc_jw_key(&w, "error");
    rrc_jw_str(&w, rrc_last_error());
    rrc_jw_obj_end(&w);
    char *out = NULL;
    if (rrc_jw_finish(&w, &out, NULL) == RRC_OK) {
        puts(out);
        free(out);
    }
    return 2;
}

static void usage(void) {
    fprintf(stderr, "usage: rrc_inspect [--config-only] [--allow-unknown-keys] [--positions p0,p1,...] <model_dir>\n");
}

int main(int argc, char **argv) {
    int config_only = 0;
    unsigned flags = 0;
    const char *dir = NULL;
    int64_t positions[64];
    int n_pos = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--config-only")) {
            config_only = 1;
        } else if (!strcmp(argv[i], "--allow-unknown-keys")) {
            flags |= RRC_CONFIG_ALLOW_UNKNOWN_KEYS;
        } else if (!strcmp(argv[i], "--positions") && i + 1 < argc) {
            char *s = argv[++i];
            while (*s && n_pos < 64) {
                char *end;
                long long v = strtoll(s, &end, 10);
                if (end == s || v < 0) {
                    usage();
                    return 1;
                }
                positions[n_pos++] = v;
                s = *end == ',' ? end + 1 : end;
                if (*end && *end != ',') {
                    usage();
                    return 1;
                }
            }
        } else if (argv[i][0] == '-') {
            usage();
            return 1;
        } else {
            dir = argv[i];
        }
    }
    if (!dir) {
        usage();
        return 1;
    }
    rrc_model_config c;
    rrc_status st = rrc_config_load(dir, flags, &c);
    if (st != RRC_OK) return fail(st);

    rrc_jw w;
    rrc_jw_init(&w, 1);
    rrc_jw_obj_begin(&w);
    rrc_jw_key(&w, "ok");
    rrc_jw_bool(&w, 1);
    rrc_jw_key(&w, "config");
    rrc_jw_obj_begin(&w);
    rrc_jw_key(&w, "model_type");
    rrc_jw_str(&w, rrc_family_name(c.family));
    rrc_jw_key(&w, "vocab_size");
    rrc_jw_int(&w, c.vocab_size);
    rrc_jw_key(&w, "hidden_size");
    rrc_jw_int(&w, c.hidden_size);
    rrc_jw_key(&w, "intermediate_size");
    rrc_jw_int(&w, c.intermediate_size);
    rrc_jw_key(&w, "num_hidden_layers");
    rrc_jw_int(&w, c.num_layers);
    rrc_jw_key(&w, "num_attention_heads");
    rrc_jw_int(&w, c.num_heads);
    rrc_jw_key(&w, "num_key_value_heads");
    rrc_jw_int(&w, c.num_kv_heads);
    rrc_jw_key(&w, "head_dim");
    rrc_jw_int(&w, c.head_dim);
    rrc_jw_key(&w, "max_position_embeddings");
    rrc_jw_int(&w, c.max_position_embeddings);
    rrc_jw_key(&w, "rms_norm_eps");
    rrc_jw_double(&w, c.rms_norm_eps);
    rrc_jw_key(&w, "tie_word_embeddings");
    rrc_jw_bool(&w, c.tie_word_embeddings);
    rrc_jw_key(&w, "qkv_bias");
    rrc_jw_bool(&w, c.qkv_bias);
    rrc_jw_key(&w, "o_bias");
    rrc_jw_bool(&w, c.o_bias);
    rrc_jw_key(&w, "qk_norm");
    rrc_jw_bool(&w, c.qk_norm);
    rrc_jw_key(&w, "rope_type");
    rrc_jw_str(&w, c.rope_type == RRC_ROPE_LLAMA3 ? "llama3" : "default");
    rrc_jw_key(&w, "rope_theta");
    rrc_jw_double(&w, c.rope_theta);
    if (c.rope_type == RRC_ROPE_LLAMA3) {
        rrc_jw_key(&w, "rope_factor");
        rrc_jw_double(&w, c.rope_factor);
        rrc_jw_key(&w, "rope_low_freq_factor");
        rrc_jw_double(&w, c.rope_low_freq_factor);
        rrc_jw_key(&w, "rope_high_freq_factor");
        rrc_jw_double(&w, c.rope_high_freq_factor);
        rrc_jw_key(&w, "rope_original_max_position_embeddings");
        rrc_jw_int(&w, c.rope_original_max_position_embeddings);
    }
    rrc_jw_key(&w, "rope_attention_scaling");
    rrc_jw_double(&w, c.rope_attention_scaling);
    rrc_jw_key(&w, "torch_dtype");
    rrc_jw_str(&w, c.torch_dtype);
    rrc_jw_obj_end(&w);

    const int64_t half = c.head_dim / 2;
    double *inv = (double *)malloc((size_t)half * sizeof(double));
    if (!inv) return fail(RRC_ERROR(RRC_ERR_OOM, "inv_freq"));
    st = rrc_rope_inv_freq(&c, inv, half);
    if (st != RRC_OK) return fail(st);
    double sum = 0;
    rrc_jw_key(&w, "inv_freq");
    rrc_jw_arr_begin(&w);
    for (int64_t i = 0; i < half; i++) {
        rrc_jw_double(&w, inv[i]);
        sum += inv[i];
    }
    rrc_jw_arr_end(&w);
    rrc_jw_key(&w, "inv_freq_sum");
    rrc_jw_double(&w, sum);
    free(inv);

    if (n_pos > 0) {
        int64_t maxp = 0;
        for (int i = 0; i < n_pos; i++) maxp = positions[i] > maxp ? positions[i] : maxp;
        float *cs = (float *)malloc((size_t)((maxp + 1) * half) * sizeof(float));
        float *sn = (float *)malloc((size_t)((maxp + 1) * half) * sizeof(float));
        if (!cs || !sn) return fail(RRC_ERROR(RRC_ERR_OOM, "rope tables"));
        st = rrc_rope_tables(&c, maxp + 1, cs, sn);
        if (st != RRC_OK) return fail(st);
        rrc_jw_key(&w, "rope");
        rrc_jw_obj_begin(&w);
        rrc_jw_key(&w, "positions");
        rrc_jw_arr_begin(&w);
        for (int i = 0; i < n_pos; i++) rrc_jw_int(&w, positions[i]);
        rrc_jw_arr_end(&w);
        const char *names[2] = {"cos", "sin"};
        float *tabs[2] = {cs, sn};
        for (int t = 0; t < 2; t++) {
            rrc_jw_key(&w, names[t]);
            rrc_jw_arr_begin(&w);
            for (int i = 0; i < n_pos; i++) {
                rrc_jw_arr_begin(&w);
                for (int64_t k = 0; k < half; k++) rrc_jw_double(&w, (double)tabs[t][positions[i] * half + k]);
                rrc_jw_arr_end(&w);
            }
            rrc_jw_arr_end(&w);
        }
        rrc_jw_obj_end(&w);
        free(cs);
        free(sn);
    }

    rrc_tensor_spec *specs = NULL;
    size_t n_specs = 0;
    st = rrc_config_expected_tensors(&c, &specs, &n_specs);
    if (st != RRC_OK) return fail(st);
    rrc_jw_key(&w, "expected_tensors");
    rrc_jw_arr_begin(&w);
    for (size_t i = 0; i < n_specs; i++) {
        rrc_jw_obj_begin(&w);
        rrc_jw_key(&w, "name");
        rrc_jw_str(&w, specs[i].name);
        rrc_jw_key(&w, "shape");
        rrc_jw_arr_begin(&w);
        for (int d = 0; d < specs[i].ndim; d++) rrc_jw_int(&w, specs[i].shape[d]);
        rrc_jw_arr_end(&w);
        rrc_jw_obj_end(&w);
    }
    rrc_jw_arr_end(&w);
    free(specs);

    if (!config_only) {
        rrc_st_set *set = NULL;
        st = rrc_st_open_checkpoint(dir, &set);
        if (st != RRC_OK) {
            rrc_jw_free(&w);
            return fail(st);
        }
        rrc_jw_key(&w, "tensors");
        rrc_jw_arr_begin(&w);
        for (size_t i = 0; i < rrc_st_count(set); i++) {
            const rrc_st_tensor *t = rrc_st_at(set, i);
            rrc_jw_obj_begin(&w);
            rrc_jw_key(&w, "name");
            rrc_jw_str(&w, t->name);
            rrc_jw_key(&w, "dtype");
            rrc_jw_str(&w, rrc_st_dtype_name(t->dtype));
            rrc_jw_key(&w, "shape");
            rrc_jw_arr_begin(&w);
            for (int d = 0; d < t->ndim; d++) rrc_jw_int(&w, t->shape[d]);
            rrc_jw_arr_end(&w);
            float *buf = (float *)malloc((size_t)(t->numel > 0 ? t->numel : 1) * sizeof(float));
            if (buf && rrc_st_to_f32(t, buf, t->numel) == RRC_OK) {
                double s = 0;
                for (int64_t k = 0; k < t->numel; k++) s += buf[k];
                rrc_jw_key(&w, "sum");
                rrc_jw_double(&w, s);
            }
            free(buf);
            rrc_jw_obj_end(&w);
        }
        rrc_jw_arr_end(&w);
        rrc_clear_error();
        st = rrc_config_check_tensors(&c, set);
        rrc_jw_key(&w, "check");
        rrc_jw_str(&w, st == RRC_OK ? "ok" : rrc_last_error());
        rrc_jw_key(&w, "check_status");
        rrc_jw_str(&w, rrc_status_str(st));
        rrc_st_close(set);
    }
    rrc_jw_obj_end(&w);
    char *out = NULL;
    st = rrc_jw_finish(&w, &out, NULL);
    if (st != RRC_OK) return fail(st);
    fputs(out, stdout);
    free(out);
    return 0;
}
