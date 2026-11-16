/* rrc/cuda/selftest_e2e.h — end-to-end checks of a device backend (DESIGN.md §7 item 3, §15.1).
 *
 * Builds tiny llama-, qwen2- and qwen3-shaped checkpoints IN C (random weights from a fixed seed,
 * written as config.json + model.safetensors into a work directory, so the normal loader reads
 * them; no Python), then compares the device backend `dev` with rrc_backend_cpu_f32acc for, per
 * family and per dtype (f32, bf16):
 *   logits, loss       rrc_model_forward_logits / the loss of rrc_model_forward_backward on one
 *                      packed micro-batch (several documents per row, prompt tokens unlabelled)
 *   grad               every LoRA gradient (all seven projections, every layer) with randomized
 *                      non-zero A and B; also with recompute "none" on dev
 *   train_*            a training run of `steps` optimizer steps through rrc_train_run_on (the
 *                      trainer rrc_train_cuda runs: loader, prefetch, AdamW, clipping, warmup +
 *                      cosine, weight decay, grad accumulation 2, checkpoint): loss and grad norm
 *                      per step, final parameters; the loss must decrease
 *   det_*              dev twice: logits, loss, gradients bit for bit; two training runs give the
 *                      same metrics bit for bit and byte-identical adapter and optimizer state
 * Tolerances are anchored to the reference's own arithmetic error: with err(x, y) a normalized
 * maximum difference (below), a check passes when
 *       err(dev, cpu_f32acc) <= max(K * err(cpu_f32acc, cpu), floor[quantity][dtype])
 * where cpu is the double-accumulation backend, K = RRC_E2E_K: a device whose fp32 arithmetic is as
 * accurate as the CPU's fp32 reference lands within about 2x that anchor. Normalizations:
 * logits / grads / grad norms / losses per step: max |d| / max |ref| over the quantity (grads
 * per tensor, the worst tensor reported); loss: relative; final parameters: max |d| / (lr * steps),
 * i.e. in units of the largest displacement Adam can produce. The floors are in selftest_e2e.c
 * (and DESIGN.md §15.1); with tf32 the f32 floors are raised to bf16-like values.
 * Pure C11 (no CUDA): the same code runs with the host emulation backend in `make test`. Prints no
 * timing of any kind. */
#ifndef RRC_SELFTEST_E2E_H
#define RRC_SELFTEST_E2E_H

#include <stdint.h>
#include <stdio.h>

#include "rrc/backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RRC_E2E_K 10.0

typedef enum rrc_e2e_size {
    RRC_E2E_SIZE_FULL = 0,  /* the GPU run: hidden ~128, seq 64, 2 rows */
    RRC_E2E_SIZE_QUICK = 1, /* host runs (make test): hidden ~32, seq 32 */
    RRC_E2E_SIZE_TINY = 2   /* the HOST-ONLY fake-runtime build: one layer, seq 8, one row */
} rrc_e2e_size;

typedef struct rrc_e2e_opts {
    const char *workdir;  /* parent of the work directory (NULL: $TMPDIR or /tmp); a fresh rrc_e2e_XXXXXX inside */
    rrc_e2e_size size;
    int steps;            /* training steps (0: 20) */
    int tf32;             /* dev f32 GEMMs may use TF32 */
    const char *only;     /* run only families / dtypes whose "family/dtype" contains this (NULL: all) */
    int families_mask;    /* bit 0 llama, 1 qwen2, 2 qwen3 (0: all) */
    int dtypes_mask;      /* bit 0 f32, 1 bf16 (0: both) */
    int no_train;         /* skip the training runs */
    int keep;             /* keep the work directory */
    int verbose;
    FILE *out;            /* NULL: stdout */
    /* called after every family x dtype case with the partial result (e.g. to rewrite a report) */
    void (*on_case_done)(void *user);
    void *user;
} rrc_e2e_opts;

typedef struct rrc_e2e_item {
    char name[64];     /* "<family>/<dtype>/<check>" */
    double dev_err;    /* err(dev, cpu_f32acc); for det_* the number of differing values / bytes */
    double ref_err;    /* err(cpu_f32acc, cpu) (the anchor); -1 where not applicable */
    double tol;        /* the bound dev_err must not exceed */
    int pass;
    char detail[192];  /* worst tensor / step, or the error message of a failed run */
} rrc_e2e_item;

#define RRC_E2E_MAX_ITEMS 160

typedef struct rrc_e2e_result {
    int checks, failures;
    int n_items;
    rrc_e2e_item items[RRC_E2E_MAX_ITEMS];
    char workdir[512];
} rrc_e2e_result;

/* Returns the number of failed checks; 0 checks run counts as one failure. */
int rrc_e2e_run(const rrc_backend *dev, const rrc_e2e_opts *o, rrc_e2e_result *res);

/* "llama", "qwen2", "qwen3" for bit 0, 1, 2. */
const char *rrc_e2e_family_name(int i);

/* Write the tiny random checkpoint of family `fam` (0 llama, 1 qwen2, 2 qwen3) at `size` the
 * end-to-end checks use (config.json + model.safetensors, tensors stored as `dt`) into `dir`
 * (created if missing). Used by rrc_bench --synthetic-model (smoke runs only). */
rrc_status rrc_e2e_write_model(const char *dir, int fam, rrc_e2e_size size, rrc_dtype dt, uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif /* RRC_SELFTEST_E2E_H */
