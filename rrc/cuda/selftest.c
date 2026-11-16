/* rrc/cuda/selftest.c — rrc_selftest: the first program to run on the GPU (DESIGN.md §11.6, §15).
 *
 *   rrc_selftest [--device N] [--seed S] [--tf32] [--sync-each-op] [--gemm-level L] [--only OP|e2e] [--quick]
 *                [--scratch-cap N] [--no-capped-pass] [--allow-fallback] [-v]
 *                [--report FILE] [--no-e2e] [--e2e-size full|quick|tiny] [--e2e-steps N] [--e2e-dir DIR] [--e2e-keep]
 *                [--e2e-filter FAMILY/DTYPE]
 *
 * Prints the device, driver and runtime, then checks every op of the CUDA backend against
 * rrc_backend_cpu_f32acc on random inputs (selftest_ops.c), the GEMM path taken per dtype
 * combination, range-error reporting and run-to-run determinism; then the end-to-end checks of
 * selftest_e2e.c (tiny llama / qwen2 / qwen3 models written in C: logits, loss, every LoRA
 * gradient, a 20-step training run through the trainer, determinism). `--only e2e` runs only the
 * end-to-end checks; any other --only filter runs only matching op checks (no end-to-end). Exit
 * status 0 only if every check passes. The last line of stdout is exactly "rrc_selftest: PASS" or
 * "rrc_selftest: FAIL". It measures and prints no timing or throughput.
 *
 * --report FILE writes everything as one JSON document (device, versions, options, per-op errors,
 * GEMM paths and tallies, end-to-end results, the model GEMM census, PASS / FAIL), rewritten after
 * every stage with "complete": false until the end, so a crash leaves the finished stages on disk.
 *
 * Built with -DRRC_SELFTEST_HOST it becomes rrc_selftest_host, which runs the same checks with a
 * host backend as the device: the host emulation backend (kc_emu.c, `--backend emu`, default), the
 * CPU double-accumulation backend (`--backend cpu`), or two test devices over cpu-f32acc:
 * `jitter` (every GEMM output perturbed by up to k * 2^-24 relative: a correct device with
 * different rounding, which must PASS) and `fault` (jitter plus attn_softmax_bwd scaled by
 * 1 + 1e-3: a subtly wrong backward, which the end-to-end checks must FAIL). No GPU is used. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rrc/rrc.h"
#include "selftest_e2e.h"
#include "selftest_ops.h"

#ifdef RRC_SELFTEST_HOST
#include "kc_emu.h"
#else
#include "rrc_cuda.h"
#endif

#ifndef RRC_GIT_REV
#define RRC_GIT_REV "unknown"
#endif

static int path_of(const rrc_backend *be, char *buf, size_t n) {
    rrc_gemm_exec_report rep;
#ifdef RRC_SELFTEST_HOST
    if (!rrc_emu_last_gemm(be, &rep)) return 0;
#else
    if (!rrc_cuda_last_gemm(be, &rep)) return 0;
#endif
    snprintf(buf, n, "%s", rep.desc);
    return 1;
}

static void usage(void) {
    fprintf(stderr,
            "usage: rrc_selftest [--device N] [--seed S] [--tf32] [--sync-each-op] [--gemm-level 0|1|2] [--only OP|e2e] [--quick]\n"
            "                    [--scratch-cap N] [--no-capped-pass] [--allow-fallback] [-v]\n"
            "                    [--report FILE] [--no-e2e] [--e2e-size full|quick|tiny] [--e2e-steps N] [--e2e-dir DIR] [--e2e-keep]\n"
            "                    [--e2e-filter FAMILY/DTYPE]\n"
#ifndef RRC_SELFTEST_HOST
            "                    [--max-grid N] [--no-grid-pass]\n"
#endif
#ifdef RRC_SELFTEST_HOST
            "       (host build) [--backend emu|cpu|jitter|fault] [--deferred] [--reject-mask M] [--min-align A]\n"
#endif
    );
}

typedef struct cfg {
    int device, tf32, sync_each, level;
    int64_t scratch_cap; /* 0: the backend default */
    unsigned max_grid;   /* CUDA: rrc_cuda_options.max_grid_blocks of the first pass (0: default) */
#ifdef RRC_SELFTEST_HOST
    const char *backend;
    rrc_emu_options eo;
#endif
} cfg;

/* ---- host test devices (jitter / fault) over cpu-f32acc -------------------------------------------------- */
#ifdef RRC_SELFTEST_HOST
static rrc_backend g_wrap;

/* A deterministic pseudo-random u in [-1, 1) from (value bits, index, k): the same inputs give the
 * same perturbation, so the jitter device is deterministic (as a GPU must be) but rounds
 * differently from the reference. */
static double jit_u(uint64_t a, uint64_t b) {
    uint64_t z = a * 0x9E3779B97F4A7C15ull + b + 0x7A11E5ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (double)(z >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
}
/* x[i] *= 1 + amp * u (u from the element's bits when jitter, else u = 1) */
static void perturb(void *p, rrc_dtype dt, int64_t i, double amp, int jitter, uint64_t salt) {
    if (dt == RRC_BF16) {
        rrc_bf16 *b = (rrc_bf16 *)p;
        const double u = jitter ? jit_u((uint64_t)b[i] | (salt << 16), (uint64_t)i) : 1.0;
        b[i] = rrc_f32_to_bf16((float)(rrc_bf16_to_f32(b[i]) * (1.0 + amp * u)));
    } else {
        float *f = (float *)p;
        const double u = jitter ? jit_u((uint64_t)rrc_f32_bits(f[i]) | (salt << 32), (uint64_t)i) : 1.0;
        f[i] = (float)(f[i] * (1.0 + amp * u));
    }
}
static rrc_status w_gemm(const rrc_backend *be, const rrc_gemm_args *g) {
    (void)be;
    const rrc_backend *base = rrc_backend_cpu_f32acc();
    RRC_CHECK(base->gemm(base, g));
    const double amp = (double)g->k * (1.0 / 16777216.0); /* k * 2^-24: the fp32 bound of a different order */
    for (int64_t b = 0; b < g->batch; b++)
        for (int64_t i = 0; i < g->m; i++)
            for (int64_t j = 0; j < g->n; j++) perturb(g->c.ptr, g->c.dtype, b * g->stride_c + i * g->ldc + j, amp, 1, (uint64_t)g->k);
    return RRC_OK;
}
static rrc_status w_softmax_bwd(const rrc_backend *be, int64_t batch, int64_t rows, int64_t cols, int64_t ld,
                                int64_t batch_stride, rrc_buf p, rrc_buf dp, float scale, rrc_buf ds) {
    (void)be;
    const rrc_backend *base = rrc_backend_cpu_f32acc();
    RRC_CHECK(base->attn_softmax_bwd(base, batch, rows, cols, ld, batch_stride, p, dp, scale, ds));
    for (int64_t b = 0; b < batch; b++)
        for (int64_t r = 0; r < rows; r++)
            for (int64_t j = 0; j < cols; j++) perturb(ds.ptr, ds.dtype, b * batch_stride + r * ld + j, 1e-3, 0, 0);
    return RRC_OK;
}
static const rrc_backend *make_wrap(int fault) {
    g_wrap = *rrc_backend_cpu_f32acc(); /* same impl: the CPU ops read be->impl */
    g_wrap.name = fault ? "cpu-f32acc+fault" : "cpu-f32acc+jitter";
    g_wrap.gemm = w_gemm;
    if (fault) g_wrap.attn_softmax_bwd = w_softmax_bwd;
    return &g_wrap;
}
#endif

/* ---- report ----------------------------------------------------------------------------------------------------- */

typedef struct pass_rec {
    const char *name;
    int ran, fails;
    char opts[200];
    rrc_st_result r;
} pass_rec;

typedef struct report {
    const char *path;
    int argc;
    char **argv;
    char started[40];
    char dev_name[256];
    int have_info;
#ifndef RRC_SELFTEST_HOST
    rrc_cuda_device_info info;
    int kbin, kptx, have_kver;
    rrc_cuda_stats e2e_stats;
    int have_e2e_stats;
    rrc_cuda_gemm_shape *census;
    size_t n_census;
    int64_t census_dropped;
#endif
    char options[512];
    pass_rec passes[3];
    int e2e_ran, e2e_fails, e2e_running;
    rrc_e2e_result e2e;
    int complete, fails;
    char fatal[RRC_ERROR_MSG_MAX + 64];
} report;

static report g_rep;

static const char *combo_name(int combo, char *buf, size_t n) {
    snprintf(buf, n, "%s x %s -> %s", (combo & 4) ? "bf16" : "f32", (combo & 2) ? "bf16" : "f32", (combo & 1) ? "bf16" : "f32");
    return buf;
}

static void jw_num_or_null(rrc_jw *w, double v) {
    if (isfinite(v)) rrc_jw_double(w, v);
    else rrc_jw_null(w);
}

static void report_pass(rrc_jw *w, const pass_rec *p) {
    rrc_jw_obj_begin(w);
    rrc_jw_key(w, "name");
    rrc_jw_str(w, p->name);
    rrc_jw_key(w, "options");
    rrc_jw_str(w, p->opts);
    rrc_jw_key(w, "result");
    rrc_jw_str(w, p->fails ? "FAIL" : "PASS");
    const rrc_st_result *r = &p->r;
    rrc_jw_key(w, "summary");
    rrc_jw_obj_begin(w);
    rrc_jw_key(w, "value_checks");
    rrc_jw_int(w, r->checks);
    rrc_jw_key(w, "value_failures");
    rrc_jw_int(w, r->failures);
    rrc_jw_key(w, "determinism_checks");
    rrc_jw_int(w, r->det_checks);
    rrc_jw_key(w, "determinism_failures");
    rrc_jw_int(w, r->det_failures);
    rrc_jw_key(w, "range_checks");
    rrc_jw_int(w, r->range_checks);
    rrc_jw_key(w, "range_failures");
    rrc_jw_int(w, r->range_failures);
    rrc_jw_key(w, "gemm_path_checks");
    rrc_jw_int(w, r->path_checks);
    rrc_jw_key(w, "gemm_path_failures");
    rrc_jw_int(w, r->path_failures);
    rrc_jw_obj_end(w);
    rrc_jw_key(w, "ops");
    rrc_jw_arr_begin(w);
    for (int i = 0; i < r->n_ops; i++) {
        const rrc_st_op_summary *o = &r->ops[i];
        rrc_jw_obj_begin(w);
        rrc_jw_key(w, "op");
        rrc_jw_str(w, o->op);
        rrc_jw_key(w, "cases");
        rrc_jw_int(w, o->cases);
        rrc_jw_key(w, "failures");
        rrc_jw_int(w, o->failures);
        rrc_jw_key(w, "max_abs_err");
        jw_num_or_null(w, o->max_abs);
        rrc_jw_key(w, "max_rel_err");
        jw_num_or_null(w, o->max_rel);
        rrc_jw_key(w, "worst_err_over_tol");
        jw_num_or_null(w, o->worst_ratio);
        rrc_jw_obj_end(w);
    }
    rrc_jw_arr_end(w);
    rrc_jw_key(w, "gemm");
    rrc_jw_obj_begin(w);
    rrc_jw_key(w, "path_per_dtype_combination");
    rrc_jw_obj_begin(w);
    char nb[64];
    for (int c = 0; c < 8; c++) {
        rrc_jw_key(w, combo_name(c, nb, sizeof nb));
        if (r->gemm_paths[c][0]) rrc_jw_str(w, r->gemm_paths[c]);
        else rrc_jw_null(w);
    }
    rrc_jw_obj_end(w);
    rrc_jw_key(w, "cases_per_layout_and_level");
    rrc_jw_obj_begin(w);
    for (int l = 0; l < 8; l++) {
        rrc_jw_key(w, rrc_st_layout_name(l));
        rrc_jw_arr_begin(w);
        for (int v = 0; v < 3; v++) rrc_jw_int(w, r->gemm_tally[l][v]);
        rrc_jw_arr_end(w);
    }
    rrc_jw_obj_end(w);
    rrc_jw_key(w, "cases");
    rrc_jw_int(w, r->gemm_cases);
    rrc_jw_key(w, "chunked_cases");
    rrc_jw_int(w, r->gemm_chunked_cases);
    rrc_jw_key(w, "kernel_fallback_cases");
    rrc_jw_int(w, r->gemm_kernel_cases);
    rrc_jw_key(w, "kernel_fallback_examples");
    rrc_jw_arr_begin(w);
    for (int i = 0; i < r->n_kernel_notes && i < RRC_ST_MAX_NOTES; i++) rrc_jw_str(w, r->kernel_notes[i]);
    rrc_jw_arr_end(w);
    rrc_jw_obj_end(w);
    rrc_jw_key(w, "failing_cases");
    rrc_jw_arr_begin(w);
    for (int i = 0; i < r->n_fail_notes && i < RRC_ST_MAX_NOTES; i++) rrc_jw_str(w, r->fail_notes[i]);
    rrc_jw_arr_end(w);
    rrc_jw_key(w, "failing_cases_total");
    rrc_jw_int(w, r->n_fail_notes);
    rrc_jw_obj_end(w);
}

#ifndef RRC_SELFTEST_HOST
static void report_e2e_gemm(rrc_jw *w, const report *R) {
    const rrc_cuda_stats *s = &R->e2e_stats;
    rrc_jw_key(w, "model_gemm_tally");
    rrc_jw_obj_begin(w);
    rrc_jw_key(w, "note");
    rrc_jw_str(w, "GEMM calls of the device's end-to-end runs per dtype combination and ladder level (0 native/promoted "
                  "cuBLASLt, 1 promote-all cuBLASLt, 2 kernel fallback); flops computed as 2*m*n*k*batch, not measured");
    char nb[64];
    int64_t tot = 0, l2 = 0;
    double ftot = 0, fl2 = 0;
    rrc_jw_key(w, "per_dtype_combination");
    rrc_jw_obj_begin(w);
    for (int c = 0; c < 8; c++) {
        const int64_t any = s->gemm_calls[c][0] + s->gemm_calls[c][1] + s->gemm_calls[c][2];
        if (!any) continue;
        rrc_jw_key(w, combo_name(c, nb, sizeof nb));
        rrc_jw_obj_begin(w);
        rrc_jw_key(w, "calls_per_level");
        rrc_jw_arr_begin(w);
        for (int v = 0; v < 3; v++) rrc_jw_int(w, s->gemm_calls[c][v]);
        rrc_jw_arr_end(w);
        rrc_jw_key(w, "flops_per_level");
        rrc_jw_arr_begin(w);
        for (int v = 0; v < 3; v++) rrc_jw_double(w, s->gemm_flops[c][v]);
        rrc_jw_arr_end(w);
        rrc_jw_obj_end(w);
        for (int v = 0; v < 3; v++) {
            tot += s->gemm_calls[c][v];
            ftot += s->gemm_flops[c][v];
        }
        l2 += s->gemm_calls[c][2];
        fl2 += s->gemm_flops[c][2];
    }
    rrc_jw_obj_end(w);
    rrc_jw_key(w, "calls_total");
    rrc_jw_int(w, tot);
    rrc_jw_key(w, "kernel_fallback_calls");
    rrc_jw_int(w, l2);
    rrc_jw_key(w, "kernel_fallback_flop_share");
    rrc_jw_double(w, ftot > 0 ? fl2 / ftot : 0.0);
    rrc_jw_key(w, "chunked_calls");
    rrc_jw_int(w, s->gemm_chunked);
    rrc_jw_key(w, "h2d_calls");
    rrc_jw_int(w, s->h2d_calls);
    rrc_jw_key(w, "h2d_staged_chunks");
    rrc_jw_int(w, s->h2d_staged_chunks);
    rrc_jw_key(w, "h2d_slot_waits");
    rrc_jw_int(w, s->h2d_slot_waits);
    rrc_jw_key(w, "stream_syncs");
    rrc_jw_int(w, s->stream_syncs);
    rrc_jw_obj_end(w);
    rrc_jw_key(w, "model_gemm_census");
    rrc_jw_arr_begin(w);
    for (size_t i = 0; R->census && i < R->n_census && i < RRC_CUDA_CENSUS_MAX; i++) {
        const rrc_cuda_gemm_shape *g = &R->census[i];
        rrc_jw_obj_begin(w);
        rrc_jw_key(w, "op");
        char ob[160];
        snprintf(ob, sizeof ob, "%c%c %s x %s -> %s", g->trans_a ? 'T' : 'N', g->trans_b ? 'T' : 'N',
                 rrc_dtype_name((rrc_dtype)g->dt_a), rrc_dtype_name((rrc_dtype)g->dt_b), rrc_dtype_name((rrc_dtype)g->dt_c));
        rrc_jw_str(w, ob);
        rrc_jw_key(w, "m");
        rrc_jw_int(w, g->m);
        rrc_jw_key(w, "n");
        rrc_jw_int(w, g->n);
        rrc_jw_key(w, "k");
        rrc_jw_int(w, g->k);
        rrc_jw_key(w, "batch");
        rrc_jw_int(w, g->batch);
        rrc_jw_key(w, "ld");
        rrc_jw_arr_begin(w);
        rrc_jw_int(w, g->lda);
        rrc_jw_int(w, g->ldb);
        rrc_jw_int(w, g->ldc);
        rrc_jw_arr_end(w);
        rrc_jw_key(w, "calls");
        rrc_jw_int(w, g->calls);
        rrc_jw_key(w, "level");
        rrc_jw_int(w, g->level);
        rrc_jw_key(w, "path");
        rrc_jw_str(w, g->path);
        rrc_jw_obj_end(w);
    }
    rrc_jw_arr_end(w);
    rrc_jw_key(w, "model_gemm_census_dropped_calls");
    rrc_jw_int(w, R->census_dropped);
}
#endif

/* Writes the whole report (atomically). Called after every stage; errors are printed, not fatal. */
static void report_write(void) {
    report *R = &g_rep;
    if (!R->path) return;
    rrc_jw w;
    rrc_jw_init(&w, 1);
    rrc_jw_obj_begin(&w);
    rrc_jw_key(&w, "schema");
    rrc_jw_str(&w, "rrc-selftest-report-v1");
    rrc_jw_key(&w, "tool");
#ifdef RRC_SELFTEST_HOST
    rrc_jw_str(&w, "rrc_selftest_host");
    rrc_jw_key(&w, "status_note");
    rrc_jw_str(&w, "host build: the device is a host backend, no GPU was used");
#else
    rrc_jw_str(&w, "rrc_selftest");
    rrc_jw_key(&w, "status_note");
    rrc_jw_str(&w, rrc_cuda_build_is_host_shim() ? "HOST-ONLY build on the fake CUDA runtime: no GPU was used"
                                                 : "CUDA build: the results below were computed on the device named in 'device'");
#endif
    rrc_jw_key(&w, "complete");
    rrc_jw_bool(&w, R->complete);
    rrc_jw_key(&w, "result");
    rrc_jw_str(&w, !R->complete ? "INCOMPLETE" : R->fails ? "FAIL" : "PASS");
    rrc_jw_key(&w, "fatal_error");
    if (R->fatal[0]) rrc_jw_str(&w, R->fatal);
    else rrc_jw_null(&w);
    rrc_jw_key(&w, "started_utc");
    rrc_jw_str(&w, R->started);
    rrc_jw_key(&w, "command_line");
    rrc_jw_arr_begin(&w);
    for (int i = 0; i < R->argc; i++) rrc_jw_str(&w, R->argv[i]);
    rrc_jw_arr_end(&w);
    rrc_jw_key(&w, "build");
    rrc_jw_obj_begin(&w);
    rrc_jw_key(&w, "git_rev");
    rrc_jw_str(&w, RRC_GIT_REV);
    rrc_jw_key(&w, "host_compiler");
    rrc_jw_str(&w, __VERSION__);
#ifndef RRC_SELFTEST_HOST
    rrc_cuda_build_info bi;
    rrc_cuda_build_info_get(&bi);
    rrc_jw_key(&w, "host_shim");
    rrc_jw_bool(&w, bi.host_shim);
    rrc_jw_key(&w, "nvcc_version");
    char vb[64];
    snprintf(vb, sizeof vb, "%d.%d.%d", bi.nvcc_major, bi.nvcc_minor, bi.nvcc_build);
    rrc_jw_str(&w, vb);
    rrc_jw_key(&w, "cudart_header_version");
    rrc_jw_int(&w, bi.cudart_header);
    rrc_jw_key(&w, "cublas_header_version");
    rrc_jw_int(&w, bi.cublas_header);
    rrc_jw_key(&w, "cuda_arch_list");
    rrc_jw_str(&w, bi.arch_list);
#endif
    rrc_jw_obj_end(&w);
    rrc_jw_key(&w, "device");
    if (R->have_info) {
        rrc_jw_obj_begin(&w);
#ifndef RRC_SELFTEST_HOST
        const rrc_cuda_device_info *in = &R->info;
        rrc_jw_key(&w, "name");
        rrc_jw_str(&w, in->name);
        rrc_jw_key(&w, "index");
        rrc_jw_int(&w, in->device);
        rrc_jw_key(&w, "compute_capability");
        char cc[16];
        snprintf(cc, sizeof cc, "%d.%d", in->cc_major, in->cc_minor);
        rrc_jw_str(&w, cc);
        rrc_jw_key(&w, "sm_count");
        rrc_jw_int(&w, in->sm_count);
        rrc_jw_key(&w, "total_mem_bytes");
        rrc_jw_int(&w, (int64_t)in->total_mem);
        rrc_jw_key(&w, "free_mem_bytes_at_start");
        rrc_jw_int(&w, (int64_t)in->free_mem);
        rrc_jw_key(&w, "driver_api_version");
        rrc_jw_int(&w, in->driver_version);
        rrc_jw_key(&w, "runtime_version");
        rrc_jw_int(&w, in->runtime_version);
        rrc_jw_key(&w, "cublaslt_version");
        rrc_jw_int(&w, (int64_t)in->cublaslt_version);
        rrc_jw_key(&w, "kernel_binary_version");
        if (R->have_kver) rrc_jw_int(&w, R->kbin);
        else rrc_jw_null(&w);
        rrc_jw_key(&w, "kernel_ptx_version");
        if (R->have_kver) rrc_jw_int(&w, R->kptx);
        else rrc_jw_null(&w);
#else
        rrc_jw_key(&w, "name");
        rrc_jw_str(&w, R->dev_name);
#endif
        rrc_jw_obj_end(&w);
    } else {
        rrc_jw_null(&w);
    }
    rrc_jw_key(&w, "options");
    rrc_jw_str(&w, R->options);
    rrc_jw_key(&w, "op_passes");
    rrc_jw_arr_begin(&w);
    for (int i = 0; i < 3; i++)
        if (R->passes[i].ran) report_pass(&w, &R->passes[i]);
    rrc_jw_arr_end(&w);
    rrc_jw_key(&w, "e2e");
    if (R->e2e_ran) {
        const rrc_e2e_result *e = &R->e2e;
        rrc_jw_obj_begin(&w);
        rrc_jw_key(&w, "result");
        rrc_jw_str(&w, R->e2e_running ? "INCOMPLETE" : R->e2e_fails ? "FAIL" : "PASS");
        rrc_jw_key(&w, "checks");
        rrc_jw_int(&w, e->checks);
        rrc_jw_key(&w, "failures");
        rrc_jw_int(&w, e->failures);
        rrc_jw_key(&w, "tolerance_rule");
        rrc_jw_str(&w, "dev_err <= max(K * ref_err, floor); dev_err = err(dev, cpu-f32acc), ref_err = err(cpu-f32acc, cpu "
                       "double accumulation); det_*: number of differing values/bytes, must be 0");
        rrc_jw_key(&w, "K");
        rrc_jw_double(&w, RRC_E2E_K);
        rrc_jw_key(&w, "items");
        rrc_jw_arr_begin(&w);
        for (int i = 0; i < e->n_items; i++) {
            const rrc_e2e_item *it = &e->items[i];
            rrc_jw_obj_begin(&w);
            rrc_jw_key(&w, "name");
            rrc_jw_str(&w, it->name);
            rrc_jw_key(&w, "pass");
            rrc_jw_bool(&w, it->pass);
            rrc_jw_key(&w, "dev_err");
            jw_num_or_null(&w, it->dev_err);
            rrc_jw_key(&w, "ref_err");
            if (it->ref_err >= 0) jw_num_or_null(&w, it->ref_err);
            else rrc_jw_null(&w);
            rrc_jw_key(&w, "tol");
            jw_num_or_null(&w, it->tol);
            rrc_jw_key(&w, "detail");
            rrc_jw_str(&w, it->detail);
            rrc_jw_obj_end(&w);
        }
        rrc_jw_arr_end(&w);
#ifndef RRC_SELFTEST_HOST
        if (R->have_e2e_stats) report_e2e_gemm(&w, R);
#endif
        rrc_jw_obj_end(&w);
    } else {
        rrc_jw_null(&w);
    }
    rrc_jw_obj_end(&w);
    char *txt = NULL;
    size_t len = 0;
    rrc_status s = rrc_jw_finish(&w, &txt, &len);
    if (s == RRC_OK) s = rrc_write_file_atomic(R->path, txt, len);
    free(txt);
    if (s != RRC_OK) {
        fprintf(stderr, "rrc_selftest: cannot write the report %s: %s\n", R->path, rrc_last_error());
        rrc_clear_error();
    }
}

/* ---- device backends -------------------------------------------------------------------------------------------- */

/* Creates the device backend for one pass; prints its description once (`describe`). */
static rrc_status make_dev(const cfg *c, int64_t scratch_cap, unsigned max_grid, int describe, const rrc_backend **dev,
                           int *tf32_out) {
#ifdef RRC_SELFTEST_HOST
    (void)max_grid;
    *tf32_out = 0;
    if (!strcmp(c->backend, "cpu") || !strcmp(c->backend, "jitter") || !strcmp(c->backend, "fault")) {
        *dev = !strcmp(c->backend, "cpu") ? rrc_backend_cpu() : make_wrap(!strcmp(c->backend, "fault"));
    } else {
        rrc_emu_options eo = c->eo;
        eo.gemm_min_level = (rrc_gemm_level)c->level;
        if (scratch_cap > 0) eo.scratch_cap_elems = scratch_cap;
        RRC_CHECK(rrc_emu_backend_create(&eo, dev));
    }
    if (describe) {
        printf("rrc_selftest (host build): no GPU is used; device backend = '%s'\n", (*dev)->name);
        snprintf(g_rep.dev_name, sizeof g_rep.dev_name, "host backend '%s' (no GPU)", (*dev)->name);
        g_rep.have_info = 1;
    }
    return RRC_OK;
#else
    rrc_cuda_options co;
    rrc_cuda_options_default(&co);
    if (c->tf32) co.allow_tf32 = 1;
    if (c->sync_each) co.sync_each_op = 1;
    co.gemm_min_level = c->level;
    if (scratch_cap > 0) co.gemm_scratch_cap_elems = scratch_cap;
    co.max_grid_blocks = max_grid;
    *tf32_out = co.allow_tf32;
    rrc_status s = rrc_cuda_backend_create_ex(c->device, &co, dev);
    if (s != RRC_OK) return s;
    if (!describe) return RRC_OK;
    rrc_cuda_device_info info;
    if (rrc_cuda_device_info_get(*dev, &info) == RRC_OK) {
        printf("device %d: %s, compute capability %d.%d (sm_%d%d), %d SMs\n", info.device, info.name, info.cc_major,
               info.cc_minor, info.cc_major, info.cc_minor, info.sm_count);
        printf("memory: %.2f GiB total, %.2f GiB free\n", (double)info.total_mem / 1073741824.0,
               (double)info.free_mem / 1073741824.0);
        printf("driver API version %d, runtime version %d, cuBLASLt version %zu\n", info.driver_version,
               info.runtime_version, info.cublaslt_version);
        g_rep.info = info;
        g_rep.have_info = 1;
    } else {
        printf("device info unavailable: %s\n", rrc_last_error());
        rrc_clear_error();
    }
    if (rrc_cuda_kernel_versions(*dev, &g_rep.kbin, &g_rep.kptx) == RRC_OK) {
        g_rep.have_kver = 1;
        printf("kernel image: binary (SASS) version %d, PTX version %d\n", g_rep.kbin, g_rep.kptx);
    } else {
        printf("kernel image versions unavailable: %s\n", rrc_last_error());
        rrc_clear_error();
    }
    printf("options: tf32 %s, sync_each_op %s, gemm ladder starts at level %d, gemm scratch cap %lld floats, max grid %u "
           "blocks%s, h2d staging %zu bytes in %d slots\n",
           co.allow_tf32 ? "on" : "off", co.sync_each_op ? "on" : "off", co.gemm_min_level,
           (long long)co.gemm_scratch_cap_elems, co.max_grid_blocks, co.max_grid_blocks ? "" : " (default)",
           co.h2d_staging_bytes, co.h2d_staging_slots);
    return RRC_OK;
#endif
}

static void destroy_dev(const rrc_backend *dev) {
#ifdef RRC_SELFTEST_HOST
    rrc_emu_backend_destroy(dev); /* no-op for the other host devices */
#else
    rrc_cuda_backend_destroy(dev);
#endif
}

/* Scratch cap of the extra GEMM pass: small enough that most promoted cases run in several
 * chunks (per-chunk pointer offsets, alignment and descriptor-cache keys, scratch reuse across
 * chunks on one stream, the post-conversion into a bf16 C). */
#define CAPPED_PASS_SCRATCH 1024
/* Grid cap of the extra CUDA pass: every element kernel's grid-stride loop and every row kernel's
 * row-stride loop runs several iterations per block even at the selftest's small sizes (with the
 * default cap of 2^20 blocks that needs more than 2^28 elements or 2^20 rows). */
#define GRID_PASS_BLOCKS 3u

/* rewrite the report after every end-to-end case (the longest stage) */
static void e2e_case_done(void *user) {
    (void)user;
    g_rep.e2e_ran = 1;
    report_write();
}

/* The last line of stdout is always "rrc_selftest: PASS|FAIL" (the Makefile's checks rely on it). */
static int finish(int fails) {
    g_rep.fails = fails;
    g_rep.complete = 1;
    report_write();
    if (g_rep.path) printf("report written to %s\n", g_rep.path);
#ifndef RRC_SELFTEST_HOST
    free(g_rep.census);
    g_rep.census = NULL;
#endif
    printf("rrc_selftest: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

static int fatal(const char *what) {
    fprintf(stderr, "rrc_selftest: %s: %s\n", what, rrc_last_error());
    snprintf(g_rep.fatal, sizeof g_rep.fatal, "%s: %s", what, rrc_last_error());
    return finish(1);
}

int main(int argc, char **argv) {
    /* line-buffered even into a pipe (| tee): a crash must not lose the lines of finished checks */
    setvbuf(stdout, NULL, _IOLBF, 0);
    cfg c;
    memset(&c, 0, sizeof c);
    memset(&g_rep, 0, sizeof g_rep);
    g_rep.argc = argc;
    g_rep.argv = argv;
    {
        time_t now = time(NULL);
        struct tm tmv;
        if (gmtime_r(&now, &tmv)) strftime(g_rep.started, sizeof g_rep.started, "%Y-%m-%dT%H:%M:%SZ", &tmv);
    }
    int verbose = 0, quick = 0, allow_fallback = 0, capped_pass = 1, grid_pass = 1, e2e = 1, e2e_keep = 0, e2e_steps = 0;
    rrc_e2e_size e2e_size = RRC_E2E_SIZE_FULL;
    const char *e2e_dir = NULL, *e2e_filter = NULL;
    int e2e_size_set = 0;
    uint64_t seed = 0x5E1F7E57ull;
    const char *only = NULL;
#ifdef RRC_SELFTEST_HOST
    c.backend = "emu";
    rrc_emu_options_default(&c.eo);
#endif
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--device") && v) c.device = atoi(argv[++i]);
        else if (!strcmp(a, "--seed") && v) seed = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(a, "--tf32")) c.tf32 = 1;
        else if (!strcmp(a, "--sync-each-op")) c.sync_each = 1;
        else if (!strcmp(a, "--gemm-level") && v) c.level = atoi(argv[++i]);
        else if (!strcmp(a, "--only") && v) only = argv[++i];
        else if (!strcmp(a, "-v") || !strcmp(a, "--verbose")) verbose = 1;
        else if (!strcmp(a, "--quick")) quick = 1;
        else if (!strcmp(a, "--scratch-cap") && v) c.scratch_cap = strtoll(argv[++i], NULL, 0);
        else if (!strcmp(a, "--no-capped-pass")) capped_pass = 0;
        else if (!strcmp(a, "--allow-fallback")) allow_fallback = 1;
        else if (!strcmp(a, "--report") && v) g_rep.path = argv[++i];
        else if (!strcmp(a, "--no-e2e")) e2e = 0;
        else if (!strcmp(a, "--e2e-keep")) e2e_keep = 1;
        else if (!strcmp(a, "--e2e-dir") && v) e2e_dir = argv[++i];
        else if (!strcmp(a, "--e2e-filter") && v) e2e_filter = argv[++i];
        else if (!strcmp(a, "--e2e-steps") && v) e2e_steps = atoi(argv[++i]);
        else if (!strcmp(a, "--e2e-size") && v) {
            const char *z = argv[++i];
            e2e_size_set = 1;
            if (!strcmp(z, "full")) e2e_size = RRC_E2E_SIZE_FULL;
            else if (!strcmp(z, "quick")) e2e_size = RRC_E2E_SIZE_QUICK;
            else if (!strcmp(z, "tiny")) e2e_size = RRC_E2E_SIZE_TINY;
            else {
                usage();
                return 2;
            }
        }
#ifndef RRC_SELFTEST_HOST
        else if (!strcmp(a, "--max-grid") && v) c.max_grid = (unsigned)strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--no-grid-pass")) grid_pass = 0;
#endif
#ifdef RRC_SELFTEST_HOST
        else if (!strcmp(a, "--backend") && v) c.backend = argv[++i];
        else if (!strcmp(a, "--deferred")) c.eo.deferred_errors = 1;
        else if (!strcmp(a, "--reject-mask") && v) c.eo.lt_reject_mask = (unsigned)strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--min-align") && v) c.eo.lt_min_align = (unsigned)strtoul(argv[++i], NULL, 0);
#endif
        else {
            usage();
            return 2;
        }
    }
    if (c.level < 0 || c.level > 2 || c.scratch_cap < 0 || e2e_steps < 0 || e2e_steps > 200) {
        usage();
        return 2;
    }
    /* --only e2e: only the end-to-end checks; any other filter: only matching op checks */
    const int only_e2e = only && !strcmp(only, "e2e");
    const int run_ops = !only_e2e;
    if (only && !only_e2e) e2e = 0;
#ifdef RRC_SELFTEST_HOST
    if (strcmp(c.backend, "cpu") && strcmp(c.backend, "emu") && strcmp(c.backend, "jitter") && strcmp(c.backend, "fault")) {
        usage();
        return 2;
    }
    if (c.tf32) fprintf(stderr, "note: --tf32 has no effect on host backends\n");
    c.tf32 = 0;
    c.eo.seed = seed ^ 0xA5A5u;
    const int strict = !strcmp(c.backend, "emu"); /* the emulator uses the CPU's arithmetic */
    const int has_paths = !strcmp(c.backend, "emu");
    grid_pass = 0; /* host backends have no grid */
    if (!e2e_size_set) e2e_size = RRC_E2E_SIZE_QUICK;
#else
    const int strict = 0, has_paths = 1;
    if (c.max_grid) grid_pass = 0; /* the first pass already runs with a grid cap */
    if (!e2e_size_set && rrc_cuda_build_is_host_shim()) e2e_size = RRC_E2E_SIZE_TINY;
#endif
    snprintf(g_rep.options, sizeof g_rep.options,
             "seed 0x%llx, tf32 %d, sync_each_op %d, gemm_level %d, scratch_cap %lld, quick %d, only %s, allow_fallback %d, "
             "e2e %d (size %s, steps %d, filter %s)",
             (unsigned long long)seed, c.tf32, c.sync_each, c.level, (long long)c.scratch_cap, quick, only ? only : "-",
             allow_fallback, e2e, e2e_size == RRC_E2E_SIZE_FULL ? "full" : e2e_size == RRC_E2E_SIZE_QUICK ? "quick" : "tiny",
             e2e_steps ? e2e_steps : (e2e_size == RRC_E2E_SIZE_TINY ? 2 : 20), e2e_filter ? e2e_filter : "-");
    const rrc_backend *dev = NULL;
    int tf32 = 0;
    rrc_status s = make_dev(&c, c.scratch_cap, c.max_grid, 1, &dev, &tf32);
    if (s != RRC_OK) {
        char w[96];
        snprintf(w, sizeof w, "cannot create the device backend (device %d)", c.device);
        return fatal(w);
    }
    report_write();
    rrc_st_opts o;
    memset(&o, 0, sizeof o);
    o.seed = seed;
    o.verbose = verbose;
    o.quick = quick;
    o.tf32 = tf32;
    o.only = only;
    o.gemm_path = path_of;
    o.strict = strict;
    /* the ladder must reach cuBLASLt unless it is forced past level 0 */
    o.require_lt = has_paths && c.level == 0 && !allow_fallback;
    int fails = 0;
    if (run_ops) {
        pass_rec *p = &g_rep.passes[0];
        p->name = "main";
        snprintf(p->opts, sizeof p->opts, "every op; require_lt %d", o.require_lt);
        p->fails = rrc_st_run(dev, rrc_backend_cpu_f32acc(), &o, &p->r);
        p->ran = 1;
        fails += p->fails;
    }
    destroy_dev(dev);
    report_write();
    /* second pass: every GEMM case again with a small scratch cap (multi-chunk paths), unless the
     * cap was given explicitly, gemm is filtered out, or the backend has no scratch */
    const int want_gemm = run_ops && (!only || strstr("gemm", only) != NULL); /* the harness filter: op name contains `only` */
    if (capped_pass && c.scratch_cap == 0 && want_gemm && c.level < 2 && has_paths) {
        printf("capped pass: gemm with a %d-float scratch cap (multi-chunk paths)\n", CAPPED_PASS_SCRATCH);
        s = make_dev(&c, CAPPED_PASS_SCRATCH, c.max_grid, 0, &dev, &tf32);
        if (s != RRC_OK) return fatal("cannot create the capped-pass backend");
        o.only = "gemm";
        /* tiny chunks may legitimately lose the operand alignment cuBLASLt needs: the path
         * requirement is checked by the first pass; this one must chunk */
        o.require_lt = 0;
        o.require_chunked = !allow_fallback; /* everything on the kernel level never chunks */
        pass_rec *p = &g_rep.passes[1];
        p->name = "capped";
        snprintf(p->opts, sizeof p->opts, "gemm only, %d-float scratch cap, require_chunked %d", CAPPED_PASS_SCRATCH,
                 o.require_chunked);
        p->fails = rrc_st_run(dev, rrc_backend_cpu_f32acc(), &o, &p->r);
        p->ran = 1;
        fails += p->fails;
        destroy_dev(dev);
        report_write();
    }
    /* third pass (CUDA): every op again with a grid of at most GRID_PASS_BLOCKS blocks */
    if (grid_pass && run_ops) {
        printf("grid pass: every op with at most %u blocks per launch (grid-stride / row-stride wrap)\n", GRID_PASS_BLOCKS);
        s = make_dev(&c, c.scratch_cap, GRID_PASS_BLOCKS, 0, &dev, &tf32);
        if (s != RRC_OK) return fatal("cannot create the grid-pass backend");
        o.only = only;
        o.require_lt = has_paths && c.level == 0 && !allow_fallback;
        o.require_chunked = 0;
        pass_rec *p = &g_rep.passes[2];
        p->name = "grid";
        snprintf(p->opts, sizeof p->opts, "every op, at most %u blocks per launch", GRID_PASS_BLOCKS);
        p->fails = rrc_st_run(dev, rrc_backend_cpu_f32acc(), &o, &p->r);
        p->ran = 1;
        fails += p->fails;
        destroy_dev(dev);
        report_write();
    }
    /* end-to-end: tiny models, trainer, determinism (default device options) */
    if (e2e) {
        s = make_dev(&c, c.scratch_cap, c.max_grid, 0, &dev, &tf32);
        if (s != RRC_OK) return fatal("cannot create the end-to-end backend");
#ifndef RRC_SELFTEST_HOST
        if (rrc_cuda_stats_reset(dev) != RRC_OK || rrc_cuda_gemm_census_enable(dev, 1) != RRC_OK) rrc_clear_error();
#endif
        rrc_e2e_opts eo;
        memset(&eo, 0, sizeof eo);
        eo.workdir = e2e_dir;
        eo.only = e2e_filter;
        eo.size = e2e_size;
        eo.steps = e2e_steps;
        eo.tf32 = tf32;
        eo.keep = e2e_keep;
        eo.verbose = verbose;
        eo.on_case_done = e2e_case_done;
        g_rep.e2e_running = 1;
        g_rep.e2e_fails = rrc_e2e_run(dev, &eo, &g_rep.e2e);
        g_rep.e2e_running = 0;
        g_rep.e2e_ran = 1;
        fails += g_rep.e2e_fails;
#ifndef RRC_SELFTEST_HOST
        if (rrc_cuda_stats_get(dev, &g_rep.e2e_stats) == RRC_OK) {
            g_rep.have_e2e_stats = 1;
            const rrc_cuda_stats *st = &g_rep.e2e_stats;
            int64_t tot = 0, l2 = 0;
            double ft = 0, f2 = 0;
            for (int k = 0; k < 8; k++)
                for (int v = 0; v < 3; v++) {
                    tot += st->gemm_calls[k][v];
                    ft += st->gemm_flops[k][v];
                    if (v == 2) {
                        l2 += st->gemm_calls[k][v];
                        f2 += st->gemm_flops[k][v];
                    }
                }
            printf("end-to-end model GEMMs on the device: %lld calls, %lld on the kernel fallback (%.1f%% of the computed "
                   "FLOPs)%s\n",
                   (long long)tot, (long long)l2, ft > 0 ? 100.0 * f2 / ft : 0.0,
                   l2 ? "  <-- WARNING: correct but slow path; see e2e.model_gemm_census in the report" : "");
        } else {
            rrc_clear_error();
        }
        const size_t nc = rrc_cuda_gemm_census_get(dev, NULL, 0, NULL);
        if (nc) {
            g_rep.census = (rrc_cuda_gemm_shape *)calloc(nc, sizeof *g_rep.census);
            if (g_rep.census) g_rep.n_census = rrc_cuda_gemm_census_get(dev, g_rep.census, nc, &g_rep.census_dropped);
            else fprintf(stderr, "rrc_selftest: no memory for the GEMM census copy (report without it)\n");
        }
#endif
        destroy_dev(dev);
        report_write();
    }
    if (!run_ops && !e2e) {
        printf("FAIL: no check ran\n");
        fails++;
    }
    return finish(fails);
}
