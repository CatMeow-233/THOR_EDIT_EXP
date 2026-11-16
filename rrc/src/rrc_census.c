/* rrc_census — GEMM census and memory estimate of one LoRA training step, COMPUTED by a dry run of
 * the real model / LoRA / optimizer code on a shape-only model (rrc/include/rrc/census.h,
 * rrc/docs/DESIGN.md §13). Nothing is executed on a device and nothing is measured.
 *
 *   rrc_census --config PATH | --preset NAME  --seq-len T [options]
 *   rrc_census --list-presets
 *
 * PATH is a Hugging Face config.json (or a directory holding one), e.g. from the GPU machine.
 * Presets (qwen3-8b, qwen2.5-7b, llama-3.1-8b) are ASSUMED shapes typed in from the public model
 * cards, NOT_TESTED against the real files (the output says so).
 *
 * Options (defaults are rrc_train's):
 *   --seq-len T            tokens per packed row (required)
 *   --micro-batch B        rows per micro-batch (1)
 *   --grad-accum M         micro-batches per optimizer step (1)
 *   --r N --alpha X --rslora --targets q_proj,...|all-linear --layers 0,1,...   LoRA (8, 16, all seven, all)
 *   --recompute full|none  (full)   --attn-chunk N (256)   --ce-chunk N (1024)
 *   --dtype bf16|f32       (bf16)   --max-positions N (0 = max_position_embeddings)
 *   --target cuda|cpu      add the CUDA backend's internal buffers to the memory total (cuda)
 *   --scratch-cap N        GEMM promotion scratch cap in f32 elements (16 Mi)
 *   --tf32                 plan f32 x f32 GEMMs with TF32 (as RRC_CUDA_ALLOW_TF32)
 *   --allow-unknown-keys   unknown config.json keys are warnings (rrc_inspect's flag)
 *   --pretty               indented JSON
 *   --out FILE             write the JSON there instead of stdout
 *
 * Exit status: 0 success, 1 usage error, 2 engine / config error (message on stderr).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rrc/census.h"
#include "rrc/json.h"

static void usage(void) {
    fprintf(stderr,
            "usage: rrc_census (--config PATH | --preset NAME) --seq-len T [--micro-batch B] [--grad-accum M]\n"
            "       [--r N] [--alpha X] [--rslora] [--targets LIST] [--layers L0,L1,...] [--recompute full|none]\n"
            "       [--attn-chunk N] [--ce-chunk N] [--dtype bf16|f32] [--max-positions N] [--target cuda|cpu]\n"
            "       [--scratch-cap N] [--tf32] [--allow-unknown-keys] [--pretty] [--out FILE]\n"
            "       rrc_census --list-presets\n");
}

static int fail(const char *what) {
    fprintf(stderr, "rrc_census: %s: %s\n", what, rrc_last_error());
    return 2;
}

static int parse_i64(const char *s, int64_t *out) {
    char *end;
    long long v = strtoll(s, &end, 10);
    if (end == s || *end) return 0;
    *out = v;
    return 1;
}
static int parse_f64(const char *s, double *out) {
    char *end;
    double v = strtod(s, &end);
    if (end == s || *end) return 0;
    *out = v;
    return 1;
}

static int parse_layers(const char *s, rrc_lora_config *lc) {
    lc->n_layers = 0;
    const char *p = s;
    while (*p) {
        char *end;
        long long v = strtoll(p, &end, 10);
        if (end == p || v < 0 || lc->n_layers >= RRC_LORA_MAX_LAYERS) return 0;
        lc->layers[lc->n_layers++] = v;
        if (*end == ',') end++;
        else if (*end) return 0;
        p = end;
    }
    return lc->n_layers > 0;
}

int main(int argc, char **argv) {
    const char *config = NULL, *preset = NULL, *out_path = NULL;
    int pretty = 0, list = 0;
    unsigned cfg_flags = 0;
    rrc_census_opts o;
    rrc_census_opts_default(&o);
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        int64_t iv;
        double dv;
#define NEXT()                                                                                                         \
    do {                                                                                                               \
        if (!v) {                                                                                                      \
            usage();                                                                                                   \
            return 1;                                                                                                  \
        }                                                                                                              \
        i++;                                                                                                           \
    } while (0)
#define INT_ARG(dst, lo)                                                                                               \
    do {                                                                                                               \
        NEXT();                                                                                                        \
        if (!parse_i64(v, &iv) || iv < (lo)) {                                                                         \
            fprintf(stderr, "rrc_census: %s needs an integer >= %lld, got \"%s\"\n", a, (long long)(lo), v);           \
            return 1;                                                                                                  \
        }                                                                                                              \
        (dst) = iv;                                                                                                    \
    } while (0)
        if (!strcmp(a, "--config")) {
            NEXT();
            config = v;
        } else if (!strcmp(a, "--preset")) {
            NEXT();
            preset = v;
        } else if (!strcmp(a, "--list-presets")) {
            list = 1;
        } else if (!strcmp(a, "--seq-len")) {
            INT_ARG(o.seq_len, 1);
        } else if (!strcmp(a, "--micro-batch")) {
            INT_ARG(o.micro_batch, 1);
        } else if (!strcmp(a, "--grad-accum")) {
            INT_ARG(o.grad_accum, 1);
        } else if (!strcmp(a, "--r")) {
            INT_ARG(o.lora.r, 1);
        } else if (!strcmp(a, "--alpha")) {
            NEXT();
            if (!parse_f64(v, &dv)) {
                usage();
                return 1;
            }
            o.lora.alpha = dv;
        } else if (!strcmp(a, "--rslora")) {
            o.lora.use_rslora = 1;
        } else if (!strcmp(a, "--targets")) {
            NEXT();
            if (rrc_lora_parse_targets(v, &o.lora.target_mask) != RRC_OK) return fail("--targets");
        } else if (!strcmp(a, "--layers")) {
            NEXT();
            if (!parse_layers(v, &o.lora)) {
                fprintf(stderr, "rrc_census: --layers needs a comma-separated list of layer indices\n");
                return 1;
            }
        } else if (!strcmp(a, "--recompute")) {
            NEXT();
            if (!strcmp(v, "full")) o.model.recompute = RRC_RECOMPUTE_FULL;
            else if (!strcmp(v, "none")) o.model.recompute = RRC_RECOMPUTE_NONE;
            else {
                usage();
                return 1;
            }
        } else if (!strcmp(a, "--attn-chunk")) {
            INT_ARG(o.model.attn_chunk, 1);
        } else if (!strcmp(a, "--ce-chunk")) {
            INT_ARG(o.model.ce_chunk, 1);
        } else if (!strcmp(a, "--dtype")) {
            NEXT();
            if (!strcmp(v, "bf16")) o.model.dtype = RRC_BF16;
            else if (!strcmp(v, "f32")) o.model.dtype = RRC_F32;
            else {
                usage();
                return 1;
            }
        } else if (!strcmp(a, "--max-positions")) {
            INT_ARG(o.model.max_positions, 0);
        } else if (!strcmp(a, "--target")) {
            NEXT();
            if (!strcmp(v, "cuda")) o.target = RRC_CENSUS_TARGET_CUDA;
            else if (!strcmp(v, "cpu")) o.target = RRC_CENSUS_TARGET_CPU;
            else {
                usage();
                return 1;
            }
        } else if (!strcmp(a, "--scratch-cap")) {
            INT_ARG(o.gemm_scratch_cap_elems, 1);
        } else if (!strcmp(a, "--tf32")) {
            o.allow_tf32 = 1;
        } else if (!strcmp(a, "--allow-unknown-keys")) {
            cfg_flags |= RRC_CONFIG_ALLOW_UNKNOWN_KEYS;
        } else if (!strcmp(a, "--pretty")) {
            pretty = 1;
        } else if (!strcmp(a, "--out")) {
            NEXT();
            out_path = v;
        } else {
            usage();
            return 1;
        }
#undef INT_ARG
#undef NEXT
    }
    if (list) {
        if (config || preset) {
            usage();
            return 1;
        }
        for (size_t i = 0; i < rrc_census_num_presets(); i++) {
            const rrc_census_preset *p = rrc_census_preset_at(i);
            printf("%s\t%s\n", p->name, p->label);
        }
        printf("(%s)\n", rrc_census_preset_note());
        return 0;
    }
    if (!config == !preset || o.seq_len <= 0) {
        usage();
        return 1;
    }
    o.model.config_flags = cfg_flags;
    rrc_model_config cfg;
    char source[4200];
    const char *note = NULL;
    if (preset) {
        const rrc_census_preset *p = rrc_census_preset_find(preset);
        if (!p) {
            fprintf(stderr, "rrc_census: unknown preset \"%s\" (see --list-presets)\n", preset);
            return 1;
        }
        if (rrc_census_preset_config(p, &cfg) != RRC_OK) return fail("preset");
        snprintf(source, sizeof source, "preset:%s (%s)", p->name, p->label);
        note = rrc_census_preset_note();
    } else {
        if (rrc_config_load(config, cfg_flags, &cfg) != RRC_OK) return fail("config");
        snprintf(source, sizeof source, "config:%s", config);
    }
    rrc_census_result *r = (rrc_census_result *)malloc(sizeof *r);
    if (!r) {
        fprintf(stderr, "rrc_census: out of memory\n");
        return 2;
    }
    if (rrc_census_run(&cfg, &o, r) != RRC_OK) {
        free(r);
        return fail("census");
    }
    char *json = NULL;
    rrc_status st = rrc_census_to_json(r, &o, source, note, pretty, &json);
    free(r);
    if (st != RRC_OK) return fail("json");
    int rc = 0;
    if (out_path) {
        if (rrc_write_file_atomic(out_path, json, strlen(json)) != RRC_OK) rc = fail("output");
    } else {
        printf("%s\n", json);
    }
    free(json);
    return rc;
}
