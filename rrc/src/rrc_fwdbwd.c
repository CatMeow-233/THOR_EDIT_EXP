/* rrc_fwdbwd — run one optimizer window (forward + backward of M micro-batches, no update) of a
 * model with a LoRA adapter and write the results as safetensors. Used by
 * rrc/tests/test_model_vs_peft.py to compare with transformers + PEFT.
 *
 *   rrc_fwdbwd --model DIR --batch FILE --out FILE [options]
 *
 * Batch file (safetensors, I32 or I64): "tokens" [M, B, T] (or [B, T] for M = 1), optional
 * "labels", "position_ids", "doc_ids" of the same shape (see rrc_batch in rrc/model.h).
 *
 * Options:
 *   --dtype f32|bf16            base weight / activation dtype (default bf16)
 *   --backend cpu|cpu-f32acc    CPU backend with double / fp32 accumulation (default cpu)
 *   --lora FILE                 LoRA weights with PEFT key names (else: rrc's seeded init)
 *   --no-lora                   base model only (loss and logits; no gradients)
 *   --r N --alpha X --rslora --targets q_proj,v_proj,... --layers 0,2 --seed S --dropout X
 *   --attn-chunk N --ce-chunk N --recompute full|none --max-positions N
 *   --reduction window_tokens|microbatch_mean   (default window_tokens)
 *   --no-grad                   loss only
 *   --logits                    also write "logits" [M, B*T, V] (forward only, f32)
 *   --hidden                    also write "hidden" [M, L + 1, B*T, H] (HF output_hidden_states order)
 *
 * Output tensors: "param.<peft name>" and "grad.<peft name>" (fp32) for every LoRA tensor,
 * "n_tokens" I32 [M], "loss_mb" F32 [M] (per micro-batch contribution), optional "logits" /
 * "hidden". Metadata (exact decimal strings): loss (window loss), loss_sum, n_tokens, window_tokens,
 * loss_mb (comma-separated, %.17g), workspace_bytes, workspace_allocs.
 * Exit status: 0 on success, 1 on usage errors, 2 on engine errors (message on stderr).
 *
 * TEST ONLY, not in the usage text: --backend cpu-offdev|cpu-f32acc-offdev runs the same CPU backend
 * behind the off-device wrapper (rrc/offdev.h, DESIGN.md §14: device pointers are not host-mapped,
 * so any host access to backend memory faults); the output file is the plain backend's bit for
 * bit. The wrapper's allocation counters are printed on stderr.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rrc/offdev.h"
#include "rrc/rrc.h"

static void usage(void) {
    fprintf(stderr,
            "usage: rrc_fwdbwd --model DIR --batch FILE --out FILE [--dtype f32|bf16] [--backend cpu|cpu-f32acc]\n"
            "       [--lora FILE | --no-lora] [--r N] [--alpha X] [--rslora] [--targets LIST] [--layers L0,L1,...]\n"
            "       [--seed S] [--dropout X] [--attn-chunk N] [--ce-chunk N] [--recompute full|none]\n"
            "       [--max-positions N] [--reduction window_tokens|microbatch_mean] [--no-grad] [--logits] "
            "[--hidden]\n");
}

static int fail(const char *what) {
    fprintf(stderr, "rrc_fwdbwd: %s: %s\n", what, rrc_last_error());
    return 2;
}

static int parse_i64(const char *s, int64_t *out) {
    char *end;
    long long v = strtoll(s, &end, 10);
    if (end == s || *end) return 0;
    *out = v;
    return 1;
}

typedef struct batch_file {
    int64_t M, B, T;
    int32_t *arr[4]; /* tokens, labels, position_ids, doc_ids (NULL if absent) */
} batch_file;

static const char *const k_batch_keys[4] = {"tokens", "labels", "position_ids", "doc_ids"};

static rrc_status read_batch(const char *path, batch_file *bf) {
    memset(bf, 0, sizeof *bf);
    rrc_st_set *set = NULL;
    RRC_CHECK_CTX(rrc_st_open_file(path, &set), "batch file %s", path);
    rrc_status st = RRC_OK;
    const rrc_st_tensor *tok = rrc_st_find(set, "tokens");
    if (!tok) {
        st = RRC_ERROR(RRC_ERR_NOT_FOUND, "batch file %s has no 'tokens' tensor", path);
        goto done;
    }
    if (tok->ndim == 2) {
        bf->M = 1;
        bf->B = tok->shape[0];
        bf->T = tok->shape[1];
    } else if (tok->ndim == 3) {
        bf->M = tok->shape[0];
        bf->B = tok->shape[1];
        bf->T = tok->shape[2];
    } else {
        st = RRC_ERROR(RRC_ERR_SHAPE, "batch 'tokens' must be [B, T] or [M, B, T]");
        goto done;
    }
    {
        int64_t n = 0;
        if (bf->M < 1 || bf->B < 1 || bf->T < 1 || __builtin_mul_overflow(bf->B, bf->T, &n) ||
            __builtin_mul_overflow(n, bf->M, &n) || n > INT32_MAX) {
            st = RRC_ERROR(RRC_ERR_SHAPE, "batch 'tokens' of shape [%lld, %lld, %lld]: every dimension must be >= 1 "
                           "and M * B * T < 2^31", (long long)bf->M, (long long)bf->B, (long long)bf->T);
            goto done;
        }
    }
    for (size_t i = 0; i < rrc_st_count(set); i++) {
        const char *nm = rrc_st_at(set, i)->name;
        int known = 0;
        for (int k = 0; k < 4; k++) known |= !strcmp(nm, k_batch_keys[k]);
        if (!known) {
            st = RRC_ERROR(RRC_ERR_UNSUPPORTED, "batch file %s: unknown tensor '%s'", path, nm);
            goto done;
        }
    }
    for (int k = 0; k < 4; k++) {
        const rrc_st_tensor *t = rrc_st_find(set, k_batch_keys[k]);
        if (!t) continue;
        if (t->ndim != tok->ndim || memcmp(t->shape, tok->shape, sizeof(int64_t) * (size_t)tok->ndim) != 0) {
            st = RRC_ERROR(RRC_ERR_SHAPE, "batch '%s' must have the shape of 'tokens'", k_batch_keys[k]);
            goto done;
        }
        bf->arr[k] = (int32_t *)malloc((size_t)(t->numel > 0 ? t->numel : 1) * sizeof(int32_t));
        if (!bf->arr[k]) {
            st = RRC_ERROR(RRC_ERR_OOM, "batch");
            goto done;
        }
        RRC_CHECK_GOTO(st, rrc_st_to_i32(t, bf->arr[k], t->numel), done);
    }
done:
    rrc_st_close(set);
    return st;
}

static void batch_at(const batch_file *bf, int64_t mb, rrc_batch *b) {
    const int64_t n = bf->B * bf->T;
    b->rows = bf->B;
    b->seq_len = bf->T;
    b->tokens = bf->arr[0] ? bf->arr[0] + mb * n : NULL;
    b->labels = bf->arr[1] ? bf->arr[1] + mb * n : NULL;
    b->position_ids = bf->arr[2] ? bf->arr[2] + mb * n : NULL;
    b->doc_ids = bf->arr[3] ? bf->arr[3] + mb * n : NULL;
}

int main(int argc, char **argv) {
    const char *model_dir = NULL, *batch_path = NULL, *out_path = NULL, *lora_path = NULL;
    rrc_model_opts mo;
    rrc_model_opts_default(&mo);
    rrc_lora_config lc;
    rrc_lora_config_default(&lc);
    const rrc_backend *be = rrc_backend_cpu();
    int offdev = 0;
    int no_lora = 0, no_grad = 0, want_logits = 0, want_hidden = 0;
    rrc_loss_reduction red = RRC_LOSS_WINDOW_TOKENS;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
#define NEED_VAL                                                                                                       \
    do {                                                                                                               \
        if (!v) {                                                                                                      \
            usage();                                                                                                   \
            return 1;                                                                                                  \
        }                                                                                                              \
        i++;                                                                                                           \
    } while (0)
        int64_t iv;
        if (!strcmp(a, "--model")) {
            NEED_VAL;
            model_dir = v;
        } else if (!strcmp(a, "--batch")) {
            NEED_VAL;
            batch_path = v;
        } else if (!strcmp(a, "--out")) {
            NEED_VAL;
            out_path = v;
        } else if (!strcmp(a, "--lora")) {
            NEED_VAL;
            lora_path = v;
        } else if (!strcmp(a, "--no-lora")) {
            no_lora = 1;
        } else if (!strcmp(a, "--no-grad")) {
            no_grad = 1;
        } else if (!strcmp(a, "--logits")) {
            want_logits = 1;
        } else if (!strcmp(a, "--hidden")) {
            want_hidden = 1;
        } else if (!strcmp(a, "--rslora")) {
            lc.use_rslora = 1;
        } else if (!strcmp(a, "--dtype")) {
            NEED_VAL;
            if (!strcmp(v, "f32")) mo.dtype = RRC_F32;
            else if (!strcmp(v, "bf16")) mo.dtype = RRC_BF16;
            else {
                usage();
                return 1;
            }
        } else if (!strcmp(a, "--backend")) {
            NEED_VAL;
            offdev = !strcmp(v, "cpu-offdev") || !strcmp(v, "cpu-f32acc-offdev");
            if (!strcmp(v, "cpu") || !strcmp(v, "cpu-offdev")) be = rrc_backend_cpu();
            else if (!strcmp(v, "cpu-f32acc") || !strcmp(v, "cpu-f32acc-offdev")) be = rrc_backend_cpu_f32acc();
            else {
                usage();
                return 1;
            }
        } else if (!strcmp(a, "--recompute")) {
            NEED_VAL;
            if (!strcmp(v, "full")) mo.recompute = RRC_RECOMPUTE_FULL;
            else if (!strcmp(v, "none")) mo.recompute = RRC_RECOMPUTE_NONE;
            else {
                usage();
                return 1;
            }
        } else if (!strcmp(a, "--reduction")) {
            NEED_VAL;
            if (!strcmp(v, "window_tokens")) red = RRC_LOSS_WINDOW_TOKENS;
            else if (!strcmp(v, "microbatch_mean")) red = RRC_LOSS_MICROBATCH_MEAN;
            else {
                usage();
                return 1;
            }
        } else if (!strcmp(a, "--targets")) {
            NEED_VAL;
            if (rrc_lora_parse_targets(v, &lc.target_mask) != RRC_OK) return fail("--targets");
        } else if (!strcmp(a, "--layers")) {
            NEED_VAL;
            const char *s = v;
            lc.n_layers = 0;
            while (*s) {
                char *end;
                long long x = strtoll(s, &end, 10);
                if (end == s || lc.n_layers >= RRC_LORA_MAX_LAYERS || (*end && *end != ',')) {
                    usage();
                    return 1;
                }
                lc.layers[lc.n_layers++] = x;
                s = *end ? end + 1 : end;
            }
        } else if (!strcmp(a, "--alpha") || !strcmp(a, "--dropout")) {
            NEED_VAL;
            char *end;
            double d = strtod(v, &end);
            if (end == v || *end) {
                usage();
                return 1;
            }
            if (!strcmp(a, "--alpha")) lc.alpha = d;
            else lc.dropout = d;
        } else if (!strcmp(a, "--r") || !strcmp(a, "--seed") || !strcmp(a, "--attn-chunk") ||
                   !strcmp(a, "--ce-chunk") || !strcmp(a, "--max-positions")) {
            NEED_VAL;
            if (!parse_i64(v, &iv)) {
                usage();
                return 1;
            }
            if (!strcmp(a, "--r")) lc.r = iv;
            else if (!strcmp(a, "--seed")) lc.seed = (uint64_t)iv;
            else if (!strcmp(a, "--attn-chunk")) mo.attn_chunk = iv;
            else if (!strcmp(a, "--ce-chunk")) mo.ce_chunk = iv;
            else mo.max_positions = iv;
        } else {
            usage();
            return 1;
        }
#undef NEED_VAL
    }
    if (!model_dir || !batch_path || !out_path || (no_lora && lora_path)) {
        usage();
        return 1;
    }

    int rc = 2;
    rrc_model *m = NULL;
    rrc_lora *lora = NULL;
    rrc_st_writer *w = NULL;
    batch_file bf;
    memset(&bf, 0, sizeof bf);
    float *logits = NULL, *hidden = NULL, *loss_mb = NULL;
    int32_t *ntok = NULL;
    const rrc_backend *od = NULL;
    if (offdev) {
        if (rrc_offdev_backend_create(be, &od) != RRC_OK) {
            rc = fail("creating the off-device test backend");
            goto out;
        }
        be = od;
    }
    if (rrc_model_load(be, model_dir, &mo, &m) != RRC_OK) {
        rc = fail("loading the model");
        goto out;
    }
    const rrc_model_config *cfg = rrc_model_get_config(m);
    if (!no_lora) {
        if (rrc_lora_create(be, cfg, &lc, &lora) != RRC_OK) {
            rc = fail("creating the LoRA adapter");
            goto out;
        }
        if (lora_path && rrc_lora_load_safetensors(lora, lora_path) != RRC_OK) {
            rc = fail("loading the LoRA weights");
            goto out;
        }
    }
    if (read_batch(batch_path, &bf) != RRC_OK) {
        rc = fail("reading the batch");
        goto out;
    }
    const int64_t M = bf.M, N = bf.B * bf.T;
    int64_t window_tokens = 0;
    for (int64_t mb = 0; mb < M; mb++) {
        rrc_batch b;
        batch_at(&bf, mb, &b);
        int64_t n;
        if (rrc_batch_count_labels(&b, &n) != RRC_OK) {
            rc = fail("counting labels");
            goto out;
        }
        window_tokens += n;
    }
    loss_mb = (float *)calloc((size_t)(M > 0 ? M : 1), sizeof(float));
    ntok = (int32_t *)calloc((size_t)(M > 0 ? M : 1), sizeof(int32_t));
    if (!loss_mb || !ntok) {
        fprintf(stderr, "rrc_fwdbwd: out of memory\n");
        goto out;
    }
    if (rrc_st_writer_create(&w) != RRC_OK) {
        rc = fail("writer");
        goto out;
    }
    char *mb_str = (char *)calloc((size_t)(M > 0 ? M : 1), 32);
    if (!mb_str) {
        fprintf(stderr, "rrc_fwdbwd: out of memory\n");
        goto out;
    }
    double loss = 0.0, loss_sum = 0.0;
    size_t mb_len = 0;
    for (int64_t mb = 0; mb < M; mb++) {
        rrc_batch b;
        batch_at(&bf, mb, &b);
        rrc_fb_opts fo = {red, window_tokens, M, !no_grad};
        rrc_fb_result res;
        if (rrc_model_forward_backward(m, lora, &b, &fo, &res) != RRC_OK) {
            free(mb_str);
            rc = fail("forward_backward");
            goto out;
        }
        loss += res.loss;
        loss_sum += res.loss_sum;
        loss_mb[mb] = (float)res.loss;
        ntok[mb] = (int32_t)res.n_tokens;
        mb_len += (size_t)snprintf(mb_str + mb_len, 32, "%s%.17g", mb ? "," : "", res.loss);
    }
    if (want_logits || want_hidden) {
        const int64_t V = cfg->vocab_size, H = cfg->hidden_size, L1 = cfg->num_layers + 1;
        if (want_logits) logits = (float *)malloc((size_t)(M * N * V) * sizeof(float));
        if (want_hidden) hidden = (float *)malloc((size_t)(M * L1 * N * H) * sizeof(float));
        if ((want_logits && !logits) || (want_hidden && !hidden)) {
            free(mb_str);
            fprintf(stderr, "rrc_fwdbwd: out of memory for logits / hidden states\n");
            goto out;
        }
        float *lg_tmp = want_logits ? NULL : (float *)malloc((size_t)(N * V) * sizeof(float));
        if (!want_logits && !lg_tmp) {
            free(mb_str);
            fprintf(stderr, "rrc_fwdbwd: out of memory\n");
            goto out;
        }
        for (int64_t mb = 0; mb < M; mb++) {
            rrc_batch b;
            batch_at(&bf, mb, &b);
            if (rrc_model_forward_logits(m, lora, &b, want_logits ? logits + mb * N * V : lg_tmp,
                                         want_hidden ? hidden + mb * L1 * N * H : NULL) != RRC_OK) {
                free(lg_tmp);
                free(mb_str);
                rc = fail("forward_logits");
                goto out;
            }
        }
        free(lg_tmp);
        int64_t shl[3] = {M, N, V}, shh[4] = {M, L1, N, H};
        if ((want_logits && rrc_st_writer_add(w, "logits", RRC_F32, 3, shl, logits) != RRC_OK) ||
            (want_hidden && rrc_st_writer_add(w, "hidden", RRC_F32, 4, shh, hidden) != RRC_OK)) {
            free(mb_str);
            rc = fail("writing logits");
            goto out;
        }
    }
    char num[64];
    int64_t shm[1] = {M};
    rrc_status st = rrc_st_writer_add(w, "n_tokens", RRC_I32, 1, shm, ntok);
    if (st == RRC_OK) st = rrc_st_writer_add(w, "loss_mb", RRC_F32, 1, shm, loss_mb);
    if (st == RRC_OK) st = rrc_st_writer_set_metadata(w, "loss_mb", mb_str);
    free(mb_str);
    snprintf(num, sizeof num, "%.17g", loss);
    if (st == RRC_OK) st = rrc_st_writer_set_metadata(w, "loss", num);
    snprintf(num, sizeof num, "%.17g", loss_sum);
    if (st == RRC_OK) st = rrc_st_writer_set_metadata(w, "loss_sum", num);
    snprintf(num, sizeof num, "%lld", (long long)window_tokens);
    if (st == RRC_OK) st = rrc_st_writer_set_metadata(w, "window_tokens", num);
    snprintf(num, sizeof num, "%zu", rrc_model_workspace_bytes(m));
    if (st == RRC_OK) st = rrc_st_writer_set_metadata(w, "workspace_bytes", num);
    snprintf(num, sizeof num, "%lld", (long long)rrc_model_workspace_allocs(m));
    if (st == RRC_OK) st = rrc_st_writer_set_metadata(w, "workspace_allocs", num);
    if (st == RRC_OK && lora) st = rrc_lora_add_to_writer(lora, 0, "param.", w);
    if (st == RRC_OK && lora && !no_grad) st = rrc_lora_add_to_writer(lora, 1, "grad.", w);
    if (st == RRC_OK) st = rrc_st_writer_write(w, out_path);
    if (st != RRC_OK) {
        rc = fail("writing the output");
        goto out;
    }
    rc = 0;
out:
    rrc_st_writer_free(w);
    for (int k = 0; k < 4; k++) free(bf.arr[k]);
    free(logits);
    free(hidden);
    free(loss_mb);
    free(ntok);
    rrc_lora_free(lora);
    rrc_model_free(m);
    if (od) {
        rrc_offdev_stats os;
        memset(&os, 0, sizeof os);
        const rrc_status ds = rrc_offdev_backend_destroy(od, &os);
        fprintf(stderr, "rrc_fwdbwd: test-offdev backend: %lld allocations, %lld frees, %lld live, peak %lld bytes, %lld bad "
                        "frees, %lld refused translations\n", (long long)os.allocs,
                (long long)os.frees, (long long)os.live, (long long)os.peak_bytes, (long long)os.bad_frees,
                (long long)os.bad_ptrs);
        if (ds != RRC_OK && rc == 0) rc = fail("off-device test backend");
    }
    return rc;
}
