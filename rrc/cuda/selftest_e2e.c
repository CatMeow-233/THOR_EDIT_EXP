/* rrc/cuda/selftest_e2e.c — see selftest_e2e.h. */
#include "selftest_e2e.h"

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "rrc/rrc.h"

/* ---- tolerance floors (DESIGN.md §15.1) ------------------------------------------------------------
 * Used when K * anchor is smaller (the anchor err(cpu-f32acc, cpu) can be accidentally tiny).
 * Calibrated with the two host control devices of rrc_selftest_host (make test): the f32 floors are
 * at least 3x the largest error of the `jitter` device (every GEMM output perturbed by up to
 * k * 2^-24 relative, the fp32 bound of a different summation order) at the full GPU sizes, and the
 * `fault` device (attn_softmax_bwd off by 1e-3 relative) exceeds the grad floor about 5x. bf16: a few
 * bf16 ulps (2^-8 = 3.9e-3), since one flipped bf16 rounding of an activation already moves the
 * outputs by about that much. At the quick and tiny sizes the bf16 anchor is small and the floor
 * decides (K * anchor / tol <= 0.1 for bf16 logits, loss and grad); at the full GPU size the bf16
 * anchor is about 0.02 for the grads and 0.04 for the final parameters, so K * anchor alone would
 * allow 0.2 / 0.45: a bf16-only fault of 10% in the attention backward passed for two families.
 * The bf16 logits and grad tolerances are therefore capped (k_cap_bf16, below the uncapped values):
 *       tol = max(floor, min(K * anchor, cap))   (bf16; f32 and TF32 are uncapped)
 * Calibrated HOST-ONLY at the full size (rrc_selftest_host, DESIGN.md §15.1): each cap is about 3x
 * the jitter device's largest bf16 error there (grad 0.026, logits 0.0093: 4.8x) and about 4x the
 * anchor (grad 0.020, logits 0.0093); the `fault` device with --fault-amp 0.1 (a 10% softmax-backward error) fails
 * the bf16 grad check of every family (grad errors 0.126-0.147 against tolerances 0.06-0.08; `make
 * test` runs this control). A 5% fault is not reliably caught in bf16 (qwen2 0.074 passes, llama
 * 0.081 fails by 1%, qwen3 0.065 fails on the floor): bf16 errors that small are left to the f32
 * end-to-end checks and the per-op bf16 checks. The GPU's own bf16 error is NOT_TESTED: a cap too
 * tight for it would show as a bf16 FAIL near the cap on the first GPU run. */
enum { Q_LOGITS = 0, Q_LOSS, Q_GRAD, Q_TLOSS, Q_TGNORM, Q_TPARAM, Q_N };
static const double k_floor[Q_N][2] = {
    /* f32     bf16 */
    {6e-5, 3e-2}, /* logits: max |d| / max |ref| */
    {6e-6, 1e-2}, /* loss: relative */
    {3e-4, 6e-2}, /* LoRA grads: per tensor max |d| / max |ref| */
    {2e-5, 2e-2}, /* training loss per step: max over steps |d| / max |ref| */
    {1e-4, 6e-2}, /* grad norm per step */
    {8e-4, 1e-1}, /* final parameters: max |d| / (lr * steps) */
};
/* INFINITY: uncapped. The training quantities stay uncapped: their anchors grow legitimately when
 * --e2e-steps is small (the final parameters in units of lr * steps: anchor 0.16-0.52 at 2 steps,
 * the jitter device 0.26-0.75), and the grad check is the one that sees a backward fault. */
static const double k_cap_bf16[Q_N] = {
    4.5e-2,   /* logits */
    INFINITY, /* loss (K * anchor stays below the floor at every size) */
    8e-2,     /* LoRA grads */
    INFINITY, /* training loss */
    INFINITY, /* grad norm */
    INFINITY, /* final parameters */
};
static const char *k_qname[Q_N] = {"logits", "loss", "grad", "train_loss", "train_grad_norm", "train_params"};

int rrc_e2e_num_quantities(void) { return Q_N; }
const char *rrc_e2e_quantity_name(int q) { return q >= 0 && q < Q_N ? k_qname[q] : NULL; }
double rrc_e2e_floor(int q, int bf16) { return q >= 0 && q < Q_N ? k_floor[q][bf16 ? 1 : 0] : NAN; }
double rrc_e2e_bf16_cap(int q) { return q >= 0 && q < Q_N ? k_cap_bf16[q] : NAN; }

/* the tolerance of quantity q (the rule above); tf32: dev f32 GEMMs may use TF32 */
static double e2e_tol(int q, int bf, int tf32, double ref_err) {
    double fl = k_floor[q][bf];
    if (!bf && tf32) fl = k_floor[q][1] / 8.0; /* TF32: 10-bit mantissa products */
    double anchored = RRC_E2E_K * ref_err;
    if (bf && anchored > k_cap_bf16[q]) anchored = k_cap_bf16[q];
    return fmax(anchored, fl);
}

/* ---- model shapes ------------------------------------------------------------------------------------ */

typedef struct shape {
    int64_t vocab, hidden, inter, layers, heads, kv, head_dim; /* head_dim 0: hidden / heads */
} shape;

static const char *k_fam[3] = {"llama", "qwen2", "qwen3"};
const char *rrc_e2e_family_name(int i) { return i >= 0 && i < 3 ? k_fam[i] : NULL; }

static shape fam_shape(int fam, rrc_e2e_size sz) {
    /* full: multiples of 16 (bf16 tensor-core alignment), GQA everywhere, qwen3 head_dim != hidden / heads */
    static const shape full[3] = {{384, 128, 352, 2, 4, 2, 0}, {320, 128, 256, 2, 8, 2, 0}, {256, 96, 224, 2, 4, 2, 32}};
    static const shape quick[3] = {{96, 32, 64, 2, 4, 2, 0}, {80, 32, 48, 2, 4, 2, 0}, {64, 48, 80, 2, 4, 2, 16}};
    static const shape tiny[3] = {{32, 16, 16, 1, 2, 1, 0}, {32, 16, 16, 1, 2, 1, 0}, {32, 16, 16, 1, 2, 1, 16}};
    return sz == RRC_E2E_SIZE_FULL ? full[fam] : sz == RRC_E2E_SIZE_QUICK ? quick[fam] : tiny[fam];
}

static void fam_config(int fam, const shape *s, char *buf, size_t cap) {
    char hd[64] = "";
    if (s->head_dim) snprintf(hd, sizeof hd, "\"head_dim\": %lld, ", (long long)s->head_dim);
    const char *extra = fam == 0 ? "\"tie_word_embeddings\": false, \"rope_scaling\": {\"rope_type\": \"llama3\", "
                                   "\"rope_theta\": 500000.0, \"factor\": 8.0, \"low_freq_factor\": 1.0, "
                                   "\"high_freq_factor\": 4.0, \"original_max_position_embeddings\": 32}"
                       : fam == 1 ? "\"tie_word_embeddings\": false, \"rope_theta\": 1000000.0"
                                  : "\"attention_bias\": true, \"tie_word_embeddings\": true, \"rope_theta\": 1000000.0";
    snprintf(buf, cap,
             "{\"model_type\": \"%s\", \"vocab_size\": %lld, \"hidden_size\": %lld, \"intermediate_size\": %lld, "
             "\"num_hidden_layers\": %lld, \"num_attention_heads\": %lld, \"num_key_value_heads\": %lld, %s"
             "\"max_position_embeddings\": 256, \"rms_norm_eps\": 1e-6, %s}",
             k_fam[fam], (long long)s->vocab, (long long)s->hidden, (long long)s->inter, (long long)s->layers,
             (long long)s->heads, (long long)s->kv, hd, extra);
}

/* ---- deterministic random numbers (splitmix64) --------------------------------------------------------- */

typedef struct rng {
    uint64_t s;
} rng;
static uint64_t rnext(rng *r) {
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float runi(rng *r) { return (float)((double)(rnext(r) >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0); }

/* ---- files ------------------------------------------------------------------------------------------------ */

static rrc_status join2(char *dst, size_t cap, const char *a, const char *b) {
    int n = snprintf(dst, cap, "%s/%s", a, b);
    RRC_REQUIRE(n > 0 && (size_t)n < cap, RRC_ERR_INVALID_ARG, "e2e: path too long: %s/%s", a, b);
    return RRC_OK;
}

static rrc_status mkdir1(const char *p) {
    if (mkdir(p, 0755) != 0 && errno != EEXIST) return RRC_ERROR(RRC_ERR_IO, "e2e: mkdir %s: %s", p, strerror(errno));
    return RRC_OK;
}

/* rm -r of a directory this module created (regular files and directories only). */
static void rm_tree(const char *path) {
    struct stat stt;
    if (lstat(path, &stt) != 0) return;
    if (S_ISDIR(stt.st_mode)) {
        DIR *d = opendir(path);
        if (d) {
            struct dirent *e;
            while ((e = readdir(d)) != NULL) {
                if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
                char sub[1024];
                if (snprintf(sub, sizeof sub, "%s/%s", path, e->d_name) < (int)sizeof sub) rm_tree(sub);
            }
            closedir(d);
        }
        (void)rmdir(path);
    } else {
        (void)unlink(path);
    }
}

static rrc_status read_all(const char *path, unsigned char **out, size_t *n) {
    *out = NULL;
    *n = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return RRC_ERROR(RRC_ERR_IO, "e2e: open %s: %s", path, strerror(errno));
    size_t cap = 1 << 16, len = 0;
    unsigned char *b = (unsigned char *)malloc(cap);
    rrc_status st = b ? RRC_OK : RRC_ERROR(RRC_ERR_OOM, "e2e: read %s", path);
    while (st == RRC_OK) {
        if (len == cap) {
            unsigned char *nb = (unsigned char *)realloc(b, cap * 2);
            if (!nb) {
                st = RRC_ERROR(RRC_ERR_OOM, "e2e: read %s", path);
                break;
            }
            b = nb;
            cap *= 2;
        }
        size_t got = fread(b + len, 1, cap - len, f);
        len += got;
        if (got == 0) {
            if (ferror(f)) st = RRC_ERROR(RRC_ERR_IO, "e2e: read %s", path);
            break;
        }
    }
    fclose(f);
    if (st != RRC_OK) {
        free(b);
        return st;
    }
    *out = b;
    *n = len;
    return RRC_OK;
}

/* config.json + model.safetensors with random weights. bf16: the tensors are stored as BF16 (the
 * dtype of real checkpoints; both backends then load identical values). */
static rrc_status write_checkpoint(const char *dir, const char *cfg_json, rrc_dtype dt, uint64_t seed) {
    char file[1024];
    RRC_CHECK(mkdir1(dir));
    RRC_CHECK(join2(file, sizeof file, dir, "config.json"));
    RRC_CHECK(rrc_write_file_atomic(file, cfg_json, strlen(cfg_json)));
    rrc_model_config c;
    RRC_CHECK(rrc_config_load(dir, 0, &c));
    rrc_tensor_spec *specs = NULL;
    size_t n = 0;
    RRC_CHECK(rrc_config_expected_tensors(&c, &specs, &n));
    rrc_st_writer *w = NULL;
    rrc_status st = rrc_st_writer_create(&w);
    rng r = {seed};
    for (size_t i = 0; st == RRC_OK && i < n; i++) {
        const int64_t numel = specs[i].shape[0] * (specs[i].ndim == 2 ? specs[i].shape[1] : 1);
        float *x = (float *)malloc((size_t)numel * sizeof(float));
        rrc_bf16 *xb = (rrc_bf16 *)malloc((size_t)numel * sizeof(rrc_bf16));
        if (!x || !xb) {
            free(x);
            free(xb);
            st = RRC_ERROR(RRC_ERR_OOM, "e2e: checkpoint tensor");
            break;
        }
        const char *nm = specs[i].name;
        const int is_norm = strstr(nm, "norm.weight") != NULL, is_bias = strstr(nm, ".bias") != NULL;
        const int is_embed = strstr(nm, "embed_tokens") != NULL || strstr(nm, "lm_head") != NULL;
        /* unit-variance-preserving linear layers: U(-1, 1) * sqrt(3 / in) */
        const double in = specs[i].ndim == 2 ? (double)specs[i].shape[1] : 1.0;
        const float lin = (float)sqrt(3.0 / in);
        for (int64_t j = 0; j < numel; j++) {
            const float u = runi(&r);
            x[j] = is_norm ? 1.0f + 0.2f * u : is_bias ? 0.1f * u : is_embed ? u : lin * u;
        }
        if (dt == RRC_BF16) {
            for (int64_t j = 0; j < numel; j++) xb[j] = rrc_f32_to_bf16(x[j]);
            st = rrc_st_writer_add(w, nm, RRC_BF16, specs[i].ndim, specs[i].shape, xb);
        } else {
            st = rrc_st_writer_add(w, nm, RRC_F32, specs[i].ndim, specs[i].shape, x);
        }
        free(x);
        free(xb);
    }
    if (st == RRC_OK) st = join2(file, sizeof file, dir, "model.safetensors");
    if (st == RRC_OK) st = rrc_st_writer_write(w, file);
    rrc_st_writer_free(w);
    free(specs);
    return st;
}

rrc_status rrc_e2e_write_model(const char *dir, int fam, rrc_e2e_size size, rrc_dtype dt, uint64_t seed) {
    RRC_REQUIRE(dir && fam >= 0 && fam < 3 && size >= RRC_E2E_SIZE_FULL && size <= RRC_E2E_SIZE_TINY, RRC_ERR_INVALID_ARG,
                "e2e: bad model request");
    const shape sh = fam_shape(fam, size);
    char cfg[1024];
    fam_config(fam, &sh, cfg, sizeof cfg);
    return write_checkpoint(dir, cfg, dt, seed);
}

/* .rrcbin with n_docs documents: lengths in [3, max_len], the first one or two tokens of each a
 * prompt (label mask 0). Little-endian (rrc/data.h). */
static rrc_status write_rrcbin(const char *path, int64_t vocab, int n_docs, int max_len, uint64_t seed) {
    rng r = {seed};
    uint32_t lens[64];
    uint64_t total = 0;
    if (n_docs > 64) n_docs = 64;
    for (int d = 0; d < n_docs; d++) total += lens[d] = 3 + (uint32_t)(rnext(&r) % (uint64_t)(max_len - 2));
    FILE *f = fopen(path, "wb");
    if (!f) return RRC_ERROR(RRC_ERR_IO, "e2e: open %s: %s", path, strerror(errno));
    unsigned char h[32];
    memcpy(h, "RRCBIN01", 8);
    const uint64_t hv[3] = {(uint64_t)n_docs, (uint64_t)vocab, total};
    for (int k = 0; k < 3; k++)
        for (int i = 0; i < 8; i++) h[8 + 8 * k + i] = (unsigned char)(hv[k] >> (8 * i));
    int ok = fwrite(h, 1, 32, f) == 32;
    for (int d = 0; d < n_docs && ok; d++) {
        unsigned char b[4];
        for (int i = 0; i < 4; i++) b[i] = (unsigned char)(lens[d] >> (8 * i));
        ok = fwrite(b, 1, 4, f) == 4;
        for (uint32_t t = 0; t < lens[d] && ok; t++) {
            const uint32_t tok = (uint32_t)(rnext(&r) % (uint64_t)vocab);
            for (int i = 0; i < 4; i++) b[i] = (unsigned char)(tok >> (8 * i));
            ok = fwrite(b, 1, 4, f) == 4;
        }
        const uint32_t prompt = 1 + (uint32_t)(d % 2);
        for (uint32_t t = 0; t < lens[d] && ok; t++) ok = fputc(t >= prompt, f) != EOF;
    }
    ok = (fclose(f) == 0) && ok;
    return ok ? RRC_OK : RRC_ERROR(RRC_ERR_IO, "e2e: write %s", path);
}

static void lora_cfg(rrc_lora_config *lc) {
    rrc_lora_config_default(lc); /* r 8, alpha 16, all seven modules, all layers */
}

/* Randomized non-zero A and B (B of the order of A, so the LoRA branch matters) as a PEFT-named
 * adapter_model.safetensors. */
static rrc_status write_adapter(const char *model_dir, const char *path, uint64_t seed) {
    rrc_model_config mc;
    RRC_CHECK(rrc_config_load(model_dir, 0, &mc));
    rrc_lora_config lc;
    lora_cfg(&lc);
    rrc_lora *l = NULL;
    RRC_CHECK(rrc_lora_create(rrc_backend_cpu(), &mc, &lc, &l));
    const int64_t n = rrc_lora_num_params(l);
    float *h = (float *)malloc((size_t)(n ? n : 1) * sizeof(float));
    rrc_status st = h ? RRC_OK : RRC_ERROR(RRC_ERR_OOM, "e2e: adapter");
    rng r = {seed};
    for (size_t i = 0; st == RRC_OK && i < rrc_lora_num_tensors(l); i++) {
        rrc_lora_tensor_info ti;
        st = rrc_lora_tensor(l, i, &ti);
        const float sc = ti.is_b ? 0.15f : (float)(1.0 / sqrt((double)ti.shape[1]));
        for (int64_t j = 0; st == RRC_OK && j < ti.numel; j++) h[ti.offset + j] = sc * runi(&r);
    }
    if (st == RRC_OK) st = rrc_buf_upload(rrc_lora_params(l), h, n);
    rrc_st_writer *w = NULL;
    if (st == RRC_OK) st = rrc_st_writer_create(&w);
    if (st == RRC_OK) st = rrc_lora_add_to_writer(l, 0, NULL, w);
    if (st == RRC_OK) st = rrc_st_writer_write(w, path);
    rrc_st_writer_free(w);
    free(h);
    rrc_lora_free(l);
    return st;
}

/* ---- the forward / backward micro-batch --------------------------------------------------------------------- */

typedef struct mb {
    int64_t B, T;
    int32_t *tok, *lab, *pos, *doc;
} mb;

static void mb_free(mb *b) {
    free(b->tok);
    free(b->lab);
    free(b->pos);
    free(b->doc);
    memset(b, 0, sizeof *b);
}

/* rows of several documents (positions restart, doc ids per document, the first token(s) of each
 * a prompt), the last document of a row cut at T */
static rrc_status mb_make(mb *b, int64_t B, int64_t T, int64_t vocab, uint64_t seed) {
    memset(b, 0, sizeof *b);
    const size_t n = (size_t)(B * T);
    b->B = B;
    b->T = T;
    b->tok = (int32_t *)malloc(n * sizeof(int32_t));
    b->lab = (int32_t *)malloc(n * sizeof(int32_t));
    b->pos = (int32_t *)malloc(n * sizeof(int32_t));
    b->doc = (int32_t *)malloc(n * sizeof(int32_t));
    if (!b->tok || !b->lab || !b->pos || !b->doc) {
        mb_free(b);
        return RRC_ERROR(RRC_ERR_OOM, "e2e: micro-batch");
    }
    rng r = {seed};
    int32_t doc = 0;
    for (int64_t row = 0; row < B; row++) {
        int64_t t = 0;
        while (t < T) {
            const int64_t len = 2 + (int64_t)(rnext(&r) % (uint64_t)(T / 2 > 2 ? T / 2 : 2));
            const int64_t prompt = 1 + (doc % 3);
            for (int64_t j = 0; j < len && t < T; j++, t++) {
                const int64_t i = row * T + t;
                b->tok[i] = (int32_t)(rnext(&r) % (uint64_t)vocab);
                b->lab[i] = j < prompt ? RRC_IGNORE_INDEX : b->tok[i];
                b->pos[i] = (int32_t)j;
                b->doc[i] = doc;
            }
            doc++;
        }
    }
    return RRC_OK;
}

/* ---- comparisons ------------------------------------------------------------------------------------------- */

/* max |x - ref| / max |ref| (NaN / Inf where ref is finite: infinite error) */
static double err_max(const float *x, const float *ref, int64_t n) {
    double scale = 0.0, d = 0.0;
    for (int64_t i = 0; i < n; i++)
        if (isfinite(ref[i]) && fabs(ref[i]) > scale) scale = fabs(ref[i]);
    for (int64_t i = 0; i < n; i++) {
        if (!isfinite(x[i]) || !isfinite(ref[i])) {
            if (!(isnan(x[i]) && isnan(ref[i])) && x[i] != ref[i]) return INFINITY;
            continue;
        }
        const double e = fabs((double)x[i] - (double)ref[i]);
        if (e > d) d = e;
    }
    return scale > 0.0 ? d / scale : d;
}
static double err_maxd(const double *x, const double *ref, int64_t n) {
    double scale = 0.0, d = 0.0;
    for (int64_t i = 0; i < n; i++)
        if (isfinite(ref[i]) && fabs(ref[i]) > scale) scale = fabs(ref[i]);
    for (int64_t i = 0; i < n; i++) {
        if (!isfinite(x[i]) || !isfinite(ref[i])) {
            if (x[i] != ref[i]) return INFINITY;
            continue;
        }
        const double e = fabs(x[i] - ref[i]);
        if (e > d) d = e;
    }
    return scale > 0.0 ? d / scale : d;
}
static double err_abs_over(const float *x, const float *ref, int64_t n, double unit) {
    double d = 0.0;
    for (int64_t i = 0; i < n; i++) {
        if (!isfinite(x[i]) || !isfinite(ref[i])) return INFINITY;
        const double e = fabs((double)x[i] - (double)ref[i]);
        if (e > d) d = e;
    }
    return d / unit;
}
static int64_t count_diff_bits(const void *a, const void *b, int64_t n_words32) {
    const uint32_t *x = (const uint32_t *)a, *y = (const uint32_t *)b;
    int64_t c = 0;
    for (int64_t i = 0; i < n_words32; i++) c += x[i] != y[i];
    return c;
}

typedef struct ctx {
    const rrc_e2e_opts *o;
    rrc_e2e_result *res;
    FILE *out;
    const rrc_backend *dev;
} ctx;

static rrc_e2e_item *item_new(ctx *c, const char *fam_dt, const char *what) {
    rrc_e2e_result *r = c->res;
    if (r->n_items >= RRC_E2E_MAX_ITEMS) return NULL;
    rrc_e2e_item *it = &r->items[r->n_items++];
    memset(it, 0, sizeof *it);
    snprintf(it->name, sizeof it->name, "%s/%s", fam_dt, what);
    it->ref_err = -1.0;
    return it;
}

static void item_done(ctx *c, rrc_e2e_item *it) {
    if (!it) return;
    c->res->checks++;
    if (!it->pass) c->res->failures++;
    fprintf(c->out, "  %-4s %-34s dev_err %-10.3g ref_err %-10.3g tol %-10.3g%s%s\n", it->pass ? "PASS" : "FAIL", it->name,
            it->dev_err, it->ref_err, it->tol, it->detail[0] ? "  " : "", it->detail);
}

/* an anchored comparison item */
static void cmp_item(ctx *c, const char *fd, int q, int bf, double dev_err, double ref_err, const char *detail) {
    rrc_e2e_item *it = item_new(c, fd, k_qname[q]);
    if (!it) {
        c->res->checks++;
        c->res->failures++; /* never drop a check silently */
        return;
    }
    it->dev_err = dev_err;
    it->ref_err = ref_err;
    it->tol = e2e_tol(q, bf, c->o->tf32, ref_err);
    it->pass = isfinite(dev_err) && dev_err <= it->tol;
    if (detail) snprintf(it->detail, sizeof it->detail, "%s", detail);
    item_done(c, it);
}

/* a failed stage: one failing item carrying the error */
static void fail_item(ctx *c, const char *fd, const char *what, const char *msg) {
    rrc_e2e_item *it = item_new(c, fd, what);
    if (!it) {
        c->res->checks++;
        c->res->failures++;
        return;
    }
    it->dev_err = INFINITY;
    it->pass = 0;
    snprintf(it->detail, sizeof it->detail, "error: %s", msg);
    item_done(c, it);
}

static void det_item(ctx *c, const char *fd, const char *what, int64_t ndiff, const char *detail) {
    rrc_e2e_item *it = item_new(c, fd, what);
    if (!it) {
        c->res->checks++;
        c->res->failures++;
        return;
    }
    it->dev_err = (double)ndiff;
    it->tol = 0.0;
    it->pass = ndiff == 0;
    snprintf(it->detail, sizeof it->detail, "%s", detail);
    item_done(c, it);
}

/* ---- forward / backward on one backend ----------------------------------------------------------------------- */

typedef struct fb_out {
    float *logits; /* [N, V] */
    double loss;   /* loss_sum of the micro-batch */
    float *grads;  /* LoRA gradients, rrc layout */
    float *grads_nr; /* dev only: recompute "none" */
    float grad_norm;
} fb_out;

static void fb_free(fb_out *f) {
    free(f->logits);
    free(f->grads);
    free(f->grads_nr);
    memset(f, 0, sizeof *f);
}

typedef struct run_shape {
    int64_t attn_chunk, ce_chunk;
} run_shape;

/* load (model, adapter) on be; logits; forward + backward. `with_none`: also the recompute "none" grads. */
static rrc_status fb_run(const rrc_backend *be, const char *model_dir, const char *adapter, rrc_dtype dt,
                         const run_shape *rs, const mb *b, int with_none, fb_out *out) {
    memset(out, 0, sizeof *out);
    rrc_model_opts mo;
    rrc_model_opts_default(&mo);
    mo.dtype = dt;
    mo.attn_chunk = rs->attn_chunk;
    mo.ce_chunk = rs->ce_chunk;
    mo.recompute = RRC_RECOMPUTE_FULL;
    rrc_model *m = NULL;
    rrc_lora *l = NULL;
    rrc_status st = rrc_model_load(be, model_dir, &mo, &m);
    const rrc_model_config *mc = st == RRC_OK ? rrc_model_get_config(m) : NULL;
    rrc_lora_config lc;
    lora_cfg(&lc);
    if (st == RRC_OK) st = rrc_lora_create(be, mc, &lc, &l);
    if (st == RRC_OK) st = rrc_lora_load_safetensors(l, adapter);
    const int64_t N = b->B * b->T, np = l ? rrc_lora_num_params(l) : 0;
    if (st == RRC_OK) {
        out->logits = (float *)malloc((size_t)(N * mc->vocab_size) * sizeof(float));
        out->grads = (float *)malloc((size_t)(np ? np : 1) * sizeof(float));
        if (with_none) out->grads_nr = (float *)malloc((size_t)(np ? np : 1) * sizeof(float));
        if (!out->logits || !out->grads || (with_none && !out->grads_nr)) st = RRC_ERROR(RRC_ERR_OOM, "e2e: outputs");
    }
    rrc_batch bt = {b->B, b->T, b->tok, b->lab, b->pos, b->doc};
    int64_t nl = 0;
    if (st == RRC_OK) st = rrc_batch_count_labels(&bt, &nl);
    if (st == RRC_OK) st = rrc_model_forward_logits(m, l, &bt, out->logits, NULL);
    rrc_fb_opts fo = {RRC_LOSS_WINDOW_TOKENS, nl, 1, 1};
    rrc_fb_result fr;
    memset(&fr, 0, sizeof fr);
    if (st == RRC_OK) st = rrc_lora_zero_grads(l);
    if (st == RRC_OK) st = rrc_model_forward_backward(m, l, &bt, &fo, &fr);
    if (st == RRC_OK) st = rrc_buf_download(rrc_lora_grads(l), out->grads, np);
    out->loss = fr.loss_sum;
    if (st == RRC_OK && with_none) {
        st = rrc_model_set_exec(m, rs->attn_chunk, rs->ce_chunk, RRC_RECOMPUTE_NONE);
        if (st == RRC_OK) st = rrc_lora_zero_grads(l);
        if (st == RRC_OK) st = rrc_model_forward_backward(m, l, &bt, &fo, &fr);
        if (st == RRC_OK) st = rrc_buf_download(rrc_lora_grads(l), out->grads_nr, np);
    }
    if (st == RRC_OK) {
        double ss = 0.0;
        for (int64_t i = 0; i < np; i++) ss += (double)out->grads[i] * out->grads[i];
        out->grad_norm = (float)sqrt(ss);
    }
    rrc_lora_free(l);
    rrc_model_free(m);
    if (st != RRC_OK) fb_free(out);
    return st;
}

/* per-tensor grad comparison: max over tensors of err(x, ref); worst tensor name into `worst` */
static double grad_err(const rrc_lora *layout, const float *x, const float *ref, char *worst, size_t cap) {
    double e = 0.0;
    if (worst && cap) worst[0] = '\0';
    for (size_t i = 0; i < rrc_lora_num_tensors(layout); i++) {
        rrc_lora_tensor_info ti;
        if (rrc_lora_tensor(layout, i, &ti) != RRC_OK) return INFINITY;
        const double ei = err_max(x + ti.offset, ref + ti.offset, ti.numel);
        if (ei > e || (isinf(ei) && !isinf(e))) {
            e = ei;
            if (worst) snprintf(worst, cap, "worst tensor %s", ti.name);
        }
    }
    return e;
}

/* ---- training ---------------------------------------------------------------------------------------------- */

typedef struct train_out {
    int n;
    double loss[256], gnorm[256], clip[256], lr[256];
    float *params;
    int64_t n_params;
    unsigned char *adapter_bytes, *state_bytes;
    size_t adapter_len, state_len;
} train_out;

static void train_free(train_out *t) {
    free(t->params);
    free(t->adapter_bytes);
    free(t->state_bytes);
    memset(t, 0, sizeof *t);
}

typedef struct train_spec {
    const char *model_dir, *data, *adapter;
    rrc_dtype dt;
    int64_t seq_len, rows, steps, attn_chunk, ce_chunk;
    double lr, max_grad_norm;
} train_spec;

static rrc_status json_num(const rrc_json *o, const char *k, double *v) {
    return rrc_json_as_number(rrc_json_get(o, k), k, v);
}

static rrc_status train_run(const rrc_backend *dev, rrc_train_backend kind, const train_spec *ts, const char *out_dir,
                            train_out *out) {
    memset(out, 0, sizeof *out);
    static rrc_train_config c; /* large (paths); one run at a time */
    rrc_train_config_default(&c);
    snprintf(c.model_dir, sizeof c.model_dir, "%s", ts->model_dir);
    snprintf(c.data_path, sizeof c.data_path, "%s", ts->data);
    snprintf(c.output_dir, sizeof c.output_dir, "%s", out_dir);
    snprintf(c.init_adapter, sizeof c.init_adapter, "%s", ts->adapter);
    c.overwrite_output_dir = 1;
    c.dtype = ts->dt;
    c.backend = kind;
    c.seq_len = ts->seq_len;
    c.micro_batch_size = ts->rows;
    c.grad_accum_steps = 2;
    c.max_steps = ts->steps;
    c.learning_rate = ts->lr;
    c.lr_scheduler_type = RRC_LR_COSINE;
    c.warmup_steps = ts->steps >= 10 ? 3 : 1;
    c.optim.weight_decay = 0.01;
    c.optim.max_grad_norm = ts->max_grad_norm;
    c.seed = 7;
    c.shuffle = 1;
    lora_cfg(&c.lora);
    c.attn_chunk = ts->attn_chunk;
    c.ce_chunk = ts->ce_chunk;
    c.recompute = RRC_RECOMPUTE_FULL;
    c.prefetch = 1;
    c.save_steps = 0;
    RRC_CHECK(rrc_train_run_on(&c, kind == RRC_TRAIN_BACKEND_CUDA ? dev : NULL));
    /* metrics.jsonl */
    char p[1200];
    RRC_CHECK(join2(p, sizeof p, out_dir, "metrics.jsonl"));
    unsigned char *txt = NULL;
    size_t len = 0;
    RRC_CHECK(read_all(p, &txt, &len));
    rrc_status st = RRC_OK;
    size_t a = 0;
    while (st == RRC_OK && a < len) {
        size_t e = a;
        while (e < len && txt[e] != '\n') e++;
        if (e > a) {
            rrc_json_doc *doc = NULL;
            st = rrc_json_parse((const char *)txt + a, e - a, &doc);
            if (st == RRC_OK && out->n >= 256) st = RRC_ERROR(RRC_ERR_INTERNAL, "e2e: too many metrics lines");
            if (st == RRC_OK) {
                const rrc_json *r = rrc_json_root(doc);
                const int i = out->n;
                st = json_num(r, "loss", &out->loss[i]);
                if (st == RRC_OK) st = json_num(r, "grad_norm", &out->gnorm[i]);
                if (st == RRC_OK) st = json_num(r, "clip_coef", &out->clip[i]);
                if (st == RRC_OK) st = json_num(r, "lr", &out->lr[i]);
                out->n++;
            }
            rrc_json_doc_free(doc);
        }
        a = e + 1;
    }
    free(txt);
    RRC_CHECK(st);
    RRC_REQUIRE(out->n == ts->steps, RRC_ERR_INTERNAL, "e2e: metrics.jsonl has %d lines, expected %lld", out->n,
                (long long)ts->steps);
    /* final checkpoint: adapter (params) and optimizer state (bytes) */
    char ck[1200], name[64];
    snprintf(name, sizeof name, "checkpoint-%lld", (long long)ts->steps);
    RRC_CHECK(join2(ck, sizeof ck, out_dir, name));
    RRC_CHECK(join2(p, sizeof p, ck, "adapter_model.safetensors"));
    RRC_CHECK(read_all(p, &out->adapter_bytes, &out->adapter_len));
    rrc_st_set *set = NULL;
    RRC_CHECK(rrc_st_open_file(p, &set));
    int64_t total = 0;
    for (size_t i = 0; i < rrc_st_count(set); i++) total += rrc_st_at(set, i)->numel;
    out->params = (float *)malloc((size_t)(total ? total : 1) * sizeof(float));
    st = out->params ? RRC_OK : RRC_ERROR(RRC_ERR_OOM, "e2e: params");
    int64_t off = 0;
    for (size_t i = 0; st == RRC_OK && i < rrc_st_count(set); i++) { /* name order: the same in every file */
        const rrc_st_tensor *t = rrc_st_at(set, i);
        st = rrc_st_to_f32(t, out->params + off, t->numel);
        off += t->numel;
    }
    out->n_params = total;
    rrc_st_close(set);
    RRC_CHECK(st);
    RRC_CHECK(join2(p, sizeof p, ck, "rrc_state.bin"));
    RRC_CHECK(read_all(p, &out->state_bytes, &out->state_len));
    return RRC_OK;
}

/* ---- one family x dtype ---------------------------------------------------------------------------------------- */

static void run_case(ctx *c, const char *work, int fam, rrc_dtype dt) {
    const rrc_e2e_opts *o = c->o;
    const int bf = dt == RRC_BF16;
    char fd[32], dir[700], model[800], adapter[800], data[800], msg[RRC_ERROR_MSG_MAX + 64];
    snprintf(fd, sizeof fd, "%s/%s", k_fam[fam], bf ? "bf16" : "f32");
    snprintf(dir, sizeof dir, "%s/%s-%s", work, k_fam[fam], bf ? "bf16" : "f32");
    const shape sh = fam_shape(fam, o->size);
    const int64_t T = o->size == RRC_E2E_SIZE_FULL ? 64 : o->size == RRC_E2E_SIZE_QUICK ? 32 : 8;
    const int64_t B = o->size == RRC_E2E_SIZE_TINY ? 1 : 2;
    const run_shape rs = {o->size == RRC_E2E_SIZE_FULL ? 24 : o->size == RRC_E2E_SIZE_QUICK ? 12 : 4,
                          o->size == RRC_E2E_SIZE_FULL ? 40 : o->size == RRC_E2E_SIZE_QUICK ? 20 : 4};
    const uint64_t seed = 1000 + 17 * (uint64_t)fam + (uint64_t)bf;
    char cfg[1024];
    fam_config(fam, &sh, cfg, sizeof cfg);
    rrc_status st = mkdir1(dir);
    if (st == RRC_OK) st = join2(model, sizeof model, dir, "model");
    if (st == RRC_OK) st = join2(adapter, sizeof adapter, dir, "init_adapter.safetensors");
    if (st == RRC_OK) st = join2(data, sizeof data, dir, "train.rrcbin");
    if (st == RRC_OK) st = write_checkpoint(model, cfg, dt, seed);
    if (st == RRC_OK) st = write_adapter(model, adapter, seed + 5);
    if (st == RRC_OK) st = write_rrcbin(data, sh.vocab, 16, (int)T, seed + 9);
    if (st != RRC_OK) {
        snprintf(msg, sizeof msg, "writing the tiny checkpoint / data: %s", rrc_last_error());
        fail_item(c, fd, "setup", msg);
        rrc_clear_error();
        return;
    }
    if (o->verbose) fprintf(c->out, "  %s: %s\n", fd, cfg);

    /* forward / backward: cpu_f32acc (reference), cpu (anchor), dev twice (+ recompute none) */
    mb b;
    double max_grad_norm = 1.0;
    st = mb_make(&b, B, T, sh.vocab, seed + 3);
    fb_out fr, fc, fd1, fd2;
    memset(&fr, 0, sizeof fr);
    memset(&fc, 0, sizeof fc);
    memset(&fd1, 0, sizeof fd1);
    memset(&fd2, 0, sizeof fd2);
    if (st == RRC_OK) st = fb_run(rrc_backend_cpu_f32acc(), model, adapter, dt, &rs, &b, 0, &fr);
    if (st == RRC_OK) st = fb_run(rrc_backend_cpu(), model, adapter, dt, &rs, &b, 0, &fc);
    if (st != RRC_OK) {
        snprintf(msg, sizeof msg, "CPU reference forward/backward: %s", rrc_last_error());
        fail_item(c, fd, "reference", msg);
        rrc_clear_error();
    } else {
        max_grad_norm = 0.7 * (double)fr.grad_norm; /* clipping active on the early steps */
        rrc_status s1 = fb_run(c->dev, model, adapter, dt, &rs, &b, 0, &fd1);
        rrc_status s2 = s1 == RRC_OK ? fb_run(c->dev, model, adapter, dt, &rs, &b, 1, &fd2) : s1;
        if (s1 != RRC_OK || s2 != RRC_OK) {
            snprintf(msg, sizeof msg, "device forward/backward: %s", rrc_last_error());
            fail_item(c, fd, "forward_backward", msg);
            rrc_clear_error();
        } else {
            rrc_model_config mc;
            rrc_lora_config lc;
            rrc_lora *layout = NULL;
            lora_cfg(&lc);
            const int64_t N = B * T;
            if (rrc_config_load(model, 0, &mc) == RRC_OK && rrc_lora_create(rrc_backend_cpu(), &mc, &lc, &layout) == RRC_OK) {
                const int64_t nv = N * mc.vocab_size, np = rrc_lora_num_params(layout);
                cmp_item(c, fd, Q_LOGITS, bf, err_max(fd1.logits, fr.logits, nv), err_max(fr.logits, fc.logits, nv), NULL);
                const double le = fabs(fd1.loss - fr.loss) / fmax(fabs(fr.loss), 1e-30);
                const double la = fabs(fr.loss - fc.loss) / fmax(fabs(fr.loss), 1e-30);
                char det[160];
                snprintf(det, sizeof det, "loss_sum dev %.9g ref %.9g", fd1.loss, fr.loss);
                cmp_item(c, fd, Q_LOSS, bf, le, la, det);
                char w1[160], w2[160];
                const double ga = grad_err(layout, fr.grads, fc.grads, NULL, 0);
                cmp_item(c, fd, Q_GRAD, bf, grad_err(layout, fd1.grads, fr.grads, w1, sizeof w1), ga, w1);
                {
                    /* recompute none on dev against the same reference */
                    const double e = grad_err(layout, fd2.grads_nr, fr.grads, w2, sizeof w2);
                    rrc_e2e_item *ri = item_new(c, fd, "grad_recompute_none");
                    if (ri) {
                        ri->dev_err = e;
                        ri->ref_err = ga;
                        ri->tol = e2e_tol(Q_GRAD, bf, o->tf32, ga);
                        ri->pass = isfinite(e) && e <= ri->tol;
                        snprintf(ri->detail, sizeof ri->detail, "%s", w2);
                        item_done(c, ri);
                    } else {
                        c->res->checks++;
                        c->res->failures++;
                    }
                }
                int64_t nd = count_diff_bits(fd1.logits, fd2.logits, nv);
                nd += memcmp(&fd1.loss, &fd2.loss, sizeof fd1.loss) != 0;
                det_item(c, fd, "det_forward", nd, "logits + loss of two dev runs, bit for bit");
                det_item(c, fd, "det_grad", count_diff_bits(fd1.grads, fd2.grads, np),
                         "LoRA gradients of two dev runs, bit for bit");
                rrc_lora_free(layout);
            } else {
                snprintf(msg, sizeof msg, "LoRA layout: %s", rrc_last_error());
                fail_item(c, fd, "layout", msg);
                rrc_clear_error();
            }
        }
    }
    fb_free(&fr);
    fb_free(&fc);
    fb_free(&fd1);
    fb_free(&fd2);
    mb_free(&b);
    if (o->no_train) return;

    /* training: cpu_f32acc, cpu, dev twice */
    const int64_t steps = o->steps > 0 ? o->steps : (o->size == RRC_E2E_SIZE_TINY ? 2 : 20);
    train_spec ts = {model, data, adapter, dt, T, B, steps, rs.attn_chunk, rs.ce_chunk, 2e-3, max_grad_norm};
    train_out tr, tc, t1, t2;
    memset(&tr, 0, sizeof tr);
    memset(&tc, 0, sizeof tc);
    memset(&t1, 0, sizeof t1);
    memset(&t2, 0, sizeof t2);
    char od[800];
    rrc_status sr = join2(od, sizeof od, dir, "run-cpu-f32acc");
    if (sr == RRC_OK) sr = train_run(NULL, RRC_TRAIN_BACKEND_CPU_F32ACC, &ts, od, &tr);
    if (sr == RRC_OK) sr = join2(od, sizeof od, dir, "run-cpu");
    if (sr == RRC_OK) sr = train_run(NULL, RRC_TRAIN_BACKEND_CPU, &ts, od, &tc);
    if (sr != RRC_OK) {
        snprintf(msg, sizeof msg, "CPU reference training: %s", rrc_last_error());
        fail_item(c, fd, "train_reference", msg);
        rrc_clear_error();
    } else {
        rrc_status s1 = join2(od, sizeof od, dir, "run-dev-1");
        if (s1 == RRC_OK) s1 = train_run(c->dev, RRC_TRAIN_BACKEND_CUDA, &ts, od, &t1);
        rrc_status s2 = s1;
        if (s1 == RRC_OK) s2 = join2(od, sizeof od, dir, "run-dev-2");
        if (s1 == RRC_OK && s2 == RRC_OK) s2 = train_run(c->dev, RRC_TRAIN_BACKEND_CUDA, &ts, od, &t2);
        if (s1 != RRC_OK || s2 != RRC_OK) {
            snprintf(msg, sizeof msg, "device training run: %s", rrc_last_error());
            fail_item(c, fd, "train", msg);
            rrc_clear_error();
        } else if (t1.n_params != tr.n_params || tc.n_params != tr.n_params) {
            fail_item(c, fd, "train", "adapter sizes differ between the runs");
        } else {
            int clipped = 0, worst_step = 0;
            double wd = -1.0;
            for (int i = 0; i < tr.n; i++) {
                clipped += tr.clip[i] < 1.0;
                const double d = fabs(t1.loss[i] - tr.loss[i]);
                if (d > wd) {
                    wd = d;
                    worst_step = i + 1;
                }
            }
            char det[192];
            snprintf(det, sizeof det, "worst step %d: dev %.9g ref %.9g; %d of %d steps clipped", worst_step,
                     t1.loss[worst_step - 1], tr.loss[worst_step - 1], clipped, tr.n);
            cmp_item(c, fd, Q_TLOSS, bf, err_maxd(t1.loss, tr.loss, tr.n), err_maxd(tr.loss, tc.loss, tr.n), det);
            cmp_item(c, fd, Q_TGNORM, bf, err_maxd(t1.gnorm, tr.gnorm, tr.n), err_maxd(tr.gnorm, tc.gnorm, tr.n), NULL);
            const double unit = ts.lr * (double)steps;
            snprintf(det, sizeof det, "max |dp| in units of lr * steps = %.3g", unit);
            cmp_item(c, fd, Q_TPARAM, bf, err_abs_over(t1.params, tr.params, tr.n_params, unit),
                     err_abs_over(tr.params, tc.params, tr.n_params, unit), det);
            /* the loss must go down (memorizing the small data set); not meaningful for a few steps
             * that start with warmup (lr 0 at step 1) */
            rrc_e2e_item *it = steps >= 10 ? item_new(c, fd, "train_loss_decreases") : NULL;
            if (steps < 10) {
            } else if (it) {
                const int k = t1.n >= 3 ? 3 : t1.n;
                double last = 0.0;
                for (int i = t1.n - k; i < t1.n; i++) last += t1.loss[i] / k;
                it->dev_err = last / t1.loss[0];
                it->tol = 1.0;
                it->pass = isfinite(last) && last < t1.loss[0];
                snprintf(it->detail, sizeof it->detail, "first %.6g, mean of the last %d %.6g (ratio must be < 1)",
                         t1.loss[0], k, last);
                item_done(c, it);
            } else {
                c->res->checks++;
                c->res->failures++;
            }
            int64_t nd = 0;
            for (int i = 0; i < t1.n; i++)
                nd += memcmp(&t1.loss[i], &t2.loss[i], sizeof(double)) != 0 ||
                      memcmp(&t1.gnorm[i], &t2.gnorm[i], sizeof(double)) != 0 ||
                      memcmp(&t1.clip[i], &t2.clip[i], sizeof(double)) != 0;
            const int64_t nm = nd;
            int64_t nb = 0;
            if (t1.adapter_len != t2.adapter_len) nb += 1 + (int64_t)(t1.adapter_len > t2.adapter_len ? t1.adapter_len : t2.adapter_len);
            else
                for (size_t i = 0; i < t1.adapter_len; i++) nb += t1.adapter_bytes[i] != t2.adapter_bytes[i];
            int64_t ns = 0;
            if (t1.state_len != t2.state_len) ns += 1 + (int64_t)(t1.state_len > t2.state_len ? t1.state_len : t2.state_len);
            else
                for (size_t i = 0; i < t1.state_len; i++) ns += t1.state_bytes[i] != t2.state_bytes[i];
            snprintf(det, sizeof det, "two dev runs: %lld metric steps, %lld adapter bytes, %lld rrc_state.bin bytes differ",
                     (long long)nm, (long long)nb, (long long)ns);
            det_item(c, fd, "det_train", nm + nb + ns, det);
        }
    }
    train_free(&tr);
    train_free(&tc);
    train_free(&t1);
    train_free(&t2);
}

int rrc_e2e_run(const rrc_backend *dev, const rrc_e2e_opts *o, rrc_e2e_result *res) {
    rrc_e2e_opts dflt;
    memset(&dflt, 0, sizeof dflt);
    if (!o) o = &dflt;
    rrc_e2e_result local;
    if (!res) res = &local;
    memset(res, 0, sizeof *res);
    ctx c = {o, res, o->out ? o->out : stdout, dev};
    const char *base = o->workdir ? o->workdir : getenv("TMPDIR");
    if (!base || !base[0]) base = "/tmp";
    snprintf(res->workdir, sizeof res->workdir, "%s/rrc_e2e_XXXXXX", base);
    fprintf(c.out, "end-to-end checks: dev '%s' vs ref 'cpu-f32acc' (tiny models written in C; tolerance = max(%g x "
                   "err(cpu-f32acc, cpu), floor), the anchored part capped for bf16)\n",
            dev->name, RRC_E2E_K);
    if (!mkdtemp(res->workdir)) {
        fprintf(c.out, "FAIL: cannot create a work directory under %s: %s\n", base, strerror(errno));
        res->checks = res->failures = 1;
        return 1;
    }
    fprintf(c.out, "  work directory %s\n", res->workdir);
    const int fm = o->families_mask ? o->families_mask : 7, dm = o->dtypes_mask ? o->dtypes_mask : 3;
    for (int fam = 0; fam < 3; fam++) {
        if (!(fm & (1 << fam))) continue;
        for (int d = 0; d < 2; d++) {
            if (!(dm & (1 << d))) continue;
            char fd[32];
            snprintf(fd, sizeof fd, "%s/%s", k_fam[fam], d ? "bf16" : "f32");
            if (o->only && !strstr(fd, o->only)) continue;
            run_case(&c, res->workdir, fam, d ? RRC_BF16 : RRC_F32);
            if (o->on_case_done) o->on_case_done(o->user);
        }
    }
    if (!o->keep) rm_tree(res->workdir);
    fprintf(c.out, "end-to-end summary: %d checks, %d failed\n", res->checks, res->failures);
    if (res->checks == 0) {
        fprintf(c.out, "FAIL: no end-to-end check ran\n");
        return 1;
    }
    return res->failures;
}
