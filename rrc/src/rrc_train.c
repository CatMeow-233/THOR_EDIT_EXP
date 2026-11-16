/* rrc_train — LoRA SFT training driver (rrc/include/rrc/train.h; config fields: rrc/docs/README.md).
 *
 *   rrc_train --config train.json                   train (or resume, per the config)
 *   rrc_train --config train.json --print-schedule  print {"warmup_steps": W, "lr": [lr of step 1..max_steps]}
 *                                                   without loading the model or the data
 *   rrc_train --print-default-config                every config field with its default (null = required)
 *
 * Exit status: 0 success, 1 usage error, 2 engine / config error (message on stderr).
 *
 * TEST ONLY, not in the usage text: --test-offdev runs the config's CPU backend behind the
 * off-device wrapper (rrc/offdev.h, DESIGN.md §14: device pointers are not host-mapped, so any host
 * access to backend memory faults) and prints its allocation counters on stderr. Outputs are the
 * normal run's bit for bit.
 *
 * Built twice: build/rrc_train (librrc.a only: "backend" "cpu" / "cpu-f32acc"; "cuda" is refused with
 * a message) and, by `make cuda`, build/rrc_train_cuda (-DRRC_WITH_CUDA, linked with librrc_cuda.a):
 * it also runs "backend": "cuda" on device 0 (CUDA_VISIBLE_DEVICES selects the GPU). The CUDA
 * build is COMPILED-ONLY until it has been run on a GPU.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rrc/json.h"
#include "rrc/offdev.h"
#include "rrc/train.h"
#ifdef RRC_WITH_CUDA
#include "rrc_cuda.h"
#endif

static void usage(void) {
    fprintf(stderr, "usage: rrc_train --config FILE [--print-schedule]\n"
                    "       rrc_train --print-default-config\n");
}

static int fail(const char *what) {
    fprintf(stderr, "rrc_train: %s: %s\n", what, rrc_last_error());
    return 2;
}

int main(int argc, char **argv) {
    const char *config = NULL;
    int print_schedule = 0, print_default = 0, test_offdev = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--config") && i + 1 < argc) config = argv[++i];
        else if (!strcmp(argv[i], "--print-schedule")) print_schedule = 1;
        else if (!strcmp(argv[i], "--print-default-config")) print_default = 1;
        else if (!strcmp(argv[i], "--test-offdev")) test_offdev = 1;
        else {
            usage();
            return 1;
        }
    }
    if (print_default) {
        if (config || print_schedule || test_offdev) {
            usage();
            return 1;
        }
        rrc_train_config c;
        rrc_train_config_default(&c);
        char *s = NULL;
        if (rrc_train_config_to_json(&c, &s) != RRC_OK) return fail("config");
        printf("%s\n", s);
        free(s);
        return 0;
    }
    if (!config) {
        usage();
        return 1;
    }
    rrc_train_config c;
    if (rrc_train_config_load(config, &c) != RRC_OK) return fail("config");
    if (print_schedule) {
        const int64_t w = rrc_train_config_warmup(&c);
        rrc_jw jw;
        rrc_jw_init(&jw, 0);
        rrc_jw_obj_begin(&jw);
        rrc_jw_key(&jw, "warmup_steps");
        rrc_jw_int(&jw, w);
        rrc_jw_key(&jw, "lr");
        rrc_jw_arr_begin(&jw);
        for (int64_t s = 1; s <= c.max_steps; s++)
            rrc_jw_double(&jw, c.learning_rate * rrc_lr_lambda(c.lr_scheduler_type, s - 1, w, c.max_steps));
        rrc_jw_arr_end(&jw);
        rrc_jw_obj_end(&jw);
        char *s = NULL;
        size_t n;
        if (rrc_jw_finish(&jw, &s, &n) != RRC_OK) return fail("schedule");
        printf("%s\n", s);
        free(s);
        return 0;
    }
    if (test_offdev && !print_schedule) {
        rrc_offdev_stats os;
        memset(&os, 0, sizeof os);
        const rrc_status st = rrc_train_run_test_offdev(&c, &os);
        fprintf(stderr, "rrc_train: test-offdev backend: %lld allocations, %lld frees, %lld live, peak %lld bytes, "
                        "%lld bad frees, %lld refused translations\n",
                (long long)os.allocs, (long long)os.frees, (long long)os.live, (long long)os.peak_bytes,
                (long long)os.bad_frees, (long long)os.bad_ptrs);
        return st != RRC_OK ? fail("training (test-offdev)") : 0;
    }
#ifdef RRC_WITH_CUDA
    if (c.backend == RRC_TRAIN_BACKEND_CUDA) {
        const rrc_backend *be = NULL;
        if (rrc_cuda_backend_create(0, &be) != RRC_OK) return fail("CUDA backend");
        /* rrc_train_run_on frees every buffer of the run (model, adapter, optimizer) before it
         * returns, on error paths too, so the backend is destroyed last. */
        /* the device's total memory sets the automatic memory pre-flight budget (DESIGN.md §13) */
        rrc_cuda_device_info info;
        uint64_t total = 0;
        if (rrc_cuda_device_info_get(be, &info) == RRC_OK) total = (uint64_t)info.total_mem;
        else rrc_clear_error();
        const rrc_status st = rrc_train_run_on_device(&c, be, total);
        int rc = 0;
        if (st != RRC_OK) rc = fail("training");
        rrc_cuda_backend_destroy(be);
        return rc;
    }
#endif
    if (rrc_train_run(&c) != RRC_OK) return fail("training");
    return 0;
}
