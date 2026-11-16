/* rrc_data — inspect / pack .rrcbin files with the trainer's loader (rrc/include/rrc/data.h).
 *
 *   rrc_data check FILE [--vocab V]
 *       validate FILE (against a model vocabulary V if given) and print a JSON summary
 *   rrc_data pack FILE --seq-len T --rows B --count N --out OUT.safetensors [--seed S] [--no-shuffle]
 *       [--truncation truncate|drop] [--pad-id P] [--no-prefetch] [--hold H] [--start-epoch E]
 *       [--start-index I] [--vocab V]
 *       write the first N micro-batches from the start cursor (E, I) as "tokens", "labels",
 *       "position_ids", "doc_ids" [N, B, T] (I32); metadata "cursors" = "epoch:index" after each
 *       micro-batch, comma-separated. The consumer takes H micro-batches at a time (default 1), then
 *       releases them, like the trainer with grad_accum_steps = H.
 *
 * Exit status: 0 success, 1 usage error, 2 data error (message on stderr).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rrc/data.h"
#include "rrc/json.h"
#include "rrc/safetensors.h"

static void usage(void) {
    fprintf(stderr, "usage: rrc_data check FILE [--vocab V]\n"
                    "       rrc_data pack FILE --seq-len T --rows B --count N --out OUT [--seed S] [--no-shuffle]\n"
                    "                [--truncation truncate|drop] [--pad-id P] [--no-prefetch] [--hold H]\n"
                    "                [--start-epoch E] [--start-index I] [--vocab V]\n");
}

static int fail(const char *what) {
    fprintf(stderr, "rrc_data: %s: %s\n", what, rrc_last_error());
    return 2;
}

static int parse_i64(const char *s, int64_t *out) {
    char *end;
    long long v = strtoll(s, &end, 10);
    if (end == s || *end) return 0;
    *out = v;
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        usage();
        return 1;
    }
    const char *mode = argv[1], *file = argv[2], *out = NULL;
    int64_t vocab = 0, T = 0, B = 0, N = 0, seed = 0, pad = 0, hold = 1, e0 = 0, i0 = 0;
    int shuffle = 1, prefetch = 1;
    rrc_truncation trunc = RRC_TRUNC_TRUNCATE;
    for (int i = 3; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        int64_t *dst = NULL;
        if (!strcmp(a, "--no-shuffle")) shuffle = 0;
        else if (!strcmp(a, "--no-prefetch")) prefetch = 0;
        else if (!strcmp(a, "--out") && v) out = argv[++i];
        else if (!strcmp(a, "--truncation") && v) {
            i++;
            if (!strcmp(v, "truncate")) trunc = RRC_TRUNC_TRUNCATE;
            else if (!strcmp(v, "drop")) trunc = RRC_TRUNC_DROP;
            else {
                usage();
                return 1;
            }
        } else {
            if (!strcmp(a, "--vocab")) dst = &vocab;
            else if (!strcmp(a, "--seq-len")) dst = &T;
            else if (!strcmp(a, "--rows")) dst = &B;
            else if (!strcmp(a, "--count")) dst = &N;
            else if (!strcmp(a, "--seed")) dst = &seed;
            else if (!strcmp(a, "--pad-id")) dst = &pad;
            else if (!strcmp(a, "--hold")) dst = &hold;
            else if (!strcmp(a, "--start-epoch")) dst = &e0;
            else if (!strcmp(a, "--start-index")) dst = &i0;
            if (!dst || !v || !parse_i64(v, dst)) {
                usage();
                return 1;
            }
            i++;
        }
    }
    rrc_dataset *d = NULL;
    if (rrc_dataset_open(file, vocab, &d) != RRC_OK) return fail("invalid data file");
    if (!strcmp(mode, "check")) {
        printf("{\"num_docs\": %lld, \"vocab_size\": %lld, \"total_tokens\": %lld, \"label_tokens\": %lld, "
               "\"max_len\": %lld, \"hash\": \"%016llx\"}\n",
               (long long)rrc_dataset_num_docs(d), (long long)rrc_dataset_vocab_size(d),
               (long long)rrc_dataset_total_tokens(d), (long long)rrc_dataset_label_tokens(d),
               (long long)rrc_dataset_max_len(d), (unsigned long long)rrc_dataset_hash(d));
        rrc_dataset_free(d);
        return 0;
    }
    if (strcmp(mode, "pack") || !out || T < 1 || B < 1 || N < 1 || hold < 1 || pad < 0 || pad > INT32_MAX ||
        N > INT32_MAX / (T * B)) {
        rrc_dataset_free(d);
        usage();
        return 1;
    }
    int rc = 2;
    rrc_loader *l = NULL;
    rrc_st_writer *w = NULL;
    const int64_t n = B * T;
    int32_t *arr = (int32_t *)malloc((size_t)(4 * N * n) * sizeof(int32_t));
    char *cursors = (char *)malloc((size_t)N * 48 + 1);
    const rrc_microbatch **held = (const rrc_microbatch **)calloc((size_t)hold, sizeof *held);
    if (!arr || !cursors || !held) {
        fprintf(stderr, "rrc_data: out of memory\n");
        goto out;
    }
    cursors[0] = '\0';
    rrc_loader_opts lo;
    memset(&lo, 0, sizeof lo);
    lo.seq_len = T;
    lo.rows = B;
    lo.seed = (uint64_t)seed;
    lo.shuffle = shuffle;
    lo.truncation = trunc;
    lo.pad_token_id = (int32_t)pad;
    lo.max_held = hold;
    lo.prefetch = prefetch;
    rrc_data_cursor start = {e0, i0, 0};
    if (rrc_loader_create(d, &lo, &start, &l) != RRC_OK) {
        rc = fail("loader");
        goto out;
    }
    size_t clen = 0;
    for (int64_t k = 0; k < N;) {
        int64_t h = 0;
        for (; h < hold && k + h < N; h++) {
            if (rrc_loader_next(l, &held[h]) != RRC_OK) {
                rc = fail("loader");
                goto out;
            }
        }
        for (int64_t j = 0; j < h; j++, k++) {
            const rrc_microbatch *mb = held[j];
            memcpy(arr + (0 * N + k) * n, mb->tokens, (size_t)n * sizeof(int32_t));
            memcpy(arr + (1 * N + k) * n, mb->labels, (size_t)n * sizeof(int32_t));
            memcpy(arr + (2 * N + k) * n, mb->position_ids, (size_t)n * sizeof(int32_t));
            memcpy(arr + (3 * N + k) * n, mb->doc_ids, (size_t)n * sizeof(int32_t));
            clen += (size_t)snprintf(cursors + clen, 48, "%s%lld:%lld", k ? "," : "", (long long)mb->end.epoch,
                                     (long long)mb->end.index);
            if (rrc_loader_release(l, mb) != RRC_OK) {
                rc = fail("loader");
                goto out;
            }
        }
    }
    static const char *const keys[4] = {"tokens", "labels", "position_ids", "doc_ids"};
    const int64_t shape[3] = {N, B, T};
    rrc_status st = rrc_st_writer_create(&w);
    for (int k = 0; k < 4 && st == RRC_OK; k++) st = rrc_st_writer_add(w, keys[k], RRC_I32, 3, shape, arr + k * N * n);
    if (st == RRC_OK) st = rrc_st_writer_set_metadata(w, "cursors", cursors);
    if (st == RRC_OK) st = rrc_st_writer_write(w, out);
    if (st != RRC_OK) {
        rc = fail("writing the output");
        goto out;
    }
    rc = 0;
out:
    rrc_st_writer_free(w);
    rrc_loader_free(l);
    rrc_dataset_free(d);
    free(arr);
    free(cursors);
    free(held);
    return rc;
}
