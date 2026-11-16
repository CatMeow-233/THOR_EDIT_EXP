/* Data line tests: .rrcbin validation (every malformed form, every truncation length, random byte
 * flips), the shuffle permutation, packing invariants, determinism of the prefetch thread, cursor
 * resume, loader misuse. The comparison with the Python packing spec / prepare_sft.py is in
 * rrc/tests/test_train_vs_torch.py. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ops_util.h"
#include "rrc/data.h"

typedef struct bbuf {
    unsigned char b[1 << 16];
    size_t n;
} bbuf;

static void put32(bbuf *b, uint32_t v) {
    for (int i = 0; i < 4; i++) b->b[b->n++] = (unsigned char)(v >> (8 * i));
}
static void put64(bbuf *b, uint64_t v) {
    for (int i = 0; i < 8; i++) b->b[b->n++] = (unsigned char)(v >> (8 * i));
}
static void header(bbuf *b, uint64_t nd, uint64_t vocab, uint64_t total) {
    b->n = 0;
    memcpy(b->b, "RRCBIN01", 8);
    b->n = 8;
    put64(b, nd);
    put64(b, vocab);
    put64(b, total);
}
static void doc(bbuf *b, int n, const uint32_t *t, const uint8_t *m) {
    put32(b, (uint32_t)n);
    for (int i = 0; i < n; i++) put32(b, t[i]);
    for (int i = 0; i < n; i++) b->b[b->n++] = m[i];
}

static int write_file(const char *path, const void *p, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    int ok = fwrite(p, 1, n, f) == n;
    return (fclose(f) == 0) && ok;
}

/* Random documents: lengths in [1, maxlen], tokens < vocab, prompt prefix mask 0. */
static size_t make_docs(bbuf *b, int nd, int maxlen, uint64_t seed, int vocab) {
    rt_rng r = {seed};
    uint32_t t[256];
    uint8_t m[256];
    bbuf body;
    body.n = 0;
    uint64_t total = 0;
    for (int d = 0; d < nd; d++) {
        int n = 1 + (int)(rt_next(&r) % (uint64_t)maxlen);
        int p = (int)(rt_next(&r) % (uint64_t)n);
        for (int i = 0; i < n; i++) {
            t[i] = (uint32_t)(rt_next(&r) % (uint64_t)vocab);
            m[i] = i >= p;
        }
        doc(&body, n, t, m);
        total += (uint64_t)n;
    }
    header(b, (uint64_t)nd, (uint64_t)vocab, total);
    memcpy(b->b + b->n, body.b, body.n);
    b->n += body.n;
    return b->n;
}

RT_TEST(rrcbin_valid_and_doc_access) {
    static bbuf b;
    const uint32_t t0[3] = {1, 2, 3}, t1[2] = {9, 0};
    const uint8_t m0[3] = {0, 1, 1}, m1[2] = {1, 0};
    header(&b, 2, 10, 5);
    doc(&b, 3, t0, m0);
    doc(&b, 2, t1, m1);
    const char *p = rt_path("ok.rrcbin");
    RT_ASSERT(write_file(p, b.b, b.n), "write");
    rrc_dataset *d = NULL;
    RT_OK(rrc_dataset_open(p, 10, &d));
    RT_EXPECT(rrc_dataset_num_docs(d) == 2 && rrc_dataset_vocab_size(d) == 10 && rrc_dataset_total_tokens(d) == 5,
              "header fields");
    RT_EXPECT(rrc_dataset_label_tokens(d) == 3 && rrc_dataset_max_len(d) == 3, "label count / max len");
    RT_EXPECT(rrc_dataset_doc_len(d, 1) == 2 && rrc_dataset_doc_len(d, 2) == 0, "doc_len");
    int32_t tok[4] = {-1, -1, -1, -1};
    uint8_t mask[4] = {7, 7, 7, 7};
    RT_OK(rrc_dataset_doc_copy(d, 1, 4, tok, mask));
    RT_EXPECT(tok[0] == 9 && tok[1] == 0 && tok[2] == -1 && mask[0] == 1 && mask[1] == 0 && mask[2] == 7,
              "doc copy");
    RT_STATUS(rrc_dataset_doc_copy(d, 2, 4, tok, mask), RRC_ERR_RANGE);
    rrc_dataset_free(d);
    /* model vocabulary smaller than the header's */
    RT_STATUS(rrc_dataset_open(p, 9, &d), RRC_ERR_PARSE);
    RT_EXPECT(strstr(rrc_last_error(), "larger than the model's vocabulary") != NULL, "%s", rrc_last_error());
    RT_STATUS(rrc_dataset_open(rt_path("does_not_exist.rrcbin"), 0, &d), RRC_ERR_NOT_FOUND);
}

RT_TEST(rrcbin_malformed_cases) {
    static bbuf b;
    const uint32_t t[3] = {1, 2, 3}, big[3] = {1, 10, 3}, huge[1] = {0x80000000u};
    const uint8_t m[3] = {0, 1, 1}, m2[3] = {0, 2, 1};
    struct {
        const char *name, *msg;
        rrc_status st;
    } cases[] = {
        {"bad_magic", "bad magic", RRC_ERR_PARSE},       {"num_docs_0", "num_docs is 0", RRC_ERR_PARSE},
        {"too_many", "can hold at most", RRC_ERR_PARSE}, {"vocab_0", "vocab_size = 0", RRC_ERR_PARSE},
        {"vocab_huge", "out of range [1", RRC_ERR_PARSE}, {"len_0", "len 0", RRC_ERR_PARSE},
        {"token_ge_vocab", "out of range", RRC_ERR_RANGE}, {"token_2^31", "out of range", RRC_ERR_RANGE},
        {"mask_2", "must be 0 or 1", RRC_ERR_PARSE},     {"trailing", "trailing bytes", RRC_ERR_PARSE},
        {"total", "total_tokens", RRC_ERR_PARSE},        {"len_past_end", "truncated", RRC_ERR_PARSE},
        {"short", "header alone", RRC_ERR_PARSE},
    };
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        const char *nm = cases[c].name;
        header(&b, 1, 10, 3);
        doc(&b, 3, t, m);
        if (!strcmp(nm, "bad_magic")) b.b[7] = '2';
        else if (!strcmp(nm, "num_docs_0")) {
            header(&b, 0, 10, 0);
            memset(b.b + b.n, 0, 16);
            b.n += 16;
        } else if (!strcmp(nm, "too_many")) {
            header(&b, 3, 10, 3);
            doc(&b, 3, t, m);
        } else if (!strcmp(nm, "vocab_0")) {
            header(&b, 1, 0, 3);
            doc(&b, 3, t, m);
        } else if (!strcmp(nm, "vocab_huge")) {
            header(&b, 1, 1ull << 31, 3);
            doc(&b, 3, t, m);
        } else if (!strcmp(nm, "len_0")) {
            header(&b, 2, 10, 3);
            doc(&b, 3, t, m);
            put32(&b, 0);
            memset(b.b + b.n, 0, 8);
            b.n += 8;
        } else if (!strcmp(nm, "token_ge_vocab")) {
            header(&b, 1, 10, 3);
            doc(&b, 3, big, m);
        } else if (!strcmp(nm, "token_2^31")) {
            header(&b, 1, 10, 1);
            doc(&b, 1, huge, m);
        } else if (!strcmp(nm, "mask_2")) {
            header(&b, 1, 10, 3);
            doc(&b, 3, t, m2);
        } else if (!strcmp(nm, "trailing")) {
            b.b[b.n++] = 0;
        } else if (!strcmp(nm, "total")) {
            header(&b, 1, 10, 4);
            doc(&b, 3, t, m);
        } else if (!strcmp(nm, "len_past_end")) {
            header(&b, 1, 10, 3);
            put32(&b, 1000000);
            memset(b.b + b.n, 0, 15);
            b.n += 15;
        } else if (!strcmp(nm, "short")) {
            b.n = 31;
        }
        const char *p = rt_path("bad.rrcbin");
        RT_ASSERT(write_file(p, b.b, b.n), "write");
        rrc_dataset *d = NULL;
        RT_STATUS(rrc_dataset_open(p, 0, &d), cases[c].st);
        RT_EXPECT(d == NULL, "%s: no dataset on failure", nm);
        RT_EXPECT(strstr(rrc_last_error(), cases[c].msg) != NULL, "%s: message '%s'", nm, rrc_last_error());
    }
}

/* Every proper prefix of a valid file is refused; random single-byte corruptions never crash
 * (they are either refused or, e.g. a token changed to another valid id, accepted). Run under
 * ASan by `make asan`. */
RT_TEST(rrcbin_truncation_and_corruption_fuzz) {
    static bbuf b;
    make_docs(&b, 12, 9, 3, 50);
    const char *p = rt_path("fuzz.rrcbin");
    rrc_dataset *d = NULL;
    for (size_t n = 0; n < b.n; n++) {
        RT_ASSERT(write_file(p, b.b, n), "write");
        rrc_status st = rrc_dataset_open(p, 0, &d);
        RT_EXPECT(st != RRC_OK && d == NULL, "prefix of %zu / %zu bytes accepted", n, b.n);
        if (d) rrc_dataset_free(d), d = NULL;
    }
    RT_ASSERT(write_file(p, b.b, b.n), "write");
    RT_OK(rrc_dataset_open(p, 0, &d));
    rrc_dataset_free(d);
    d = NULL;
    rt_rng r = {77};
    int refused = 0;
    for (int it = 0; it < 400; it++) {
        static bbuf c;
        memcpy(c.b, b.b, b.n);
        c.n = b.n;
        const size_t at = (size_t)(rt_next(&r) % b.n);
        c.b[at] ^= (unsigned char)(1u << (rt_next(&r) % 8));
        RT_ASSERT(write_file(p, c.b, c.n), "write");
        if (rrc_dataset_open(p, 0, &d) != RRC_OK) refused++;
        else {
            /* an accepted file is internally consistent */
            int64_t sum = 0;
            for (int64_t i = 0; i < rrc_dataset_num_docs(d); i++) sum += rrc_dataset_doc_len(d, i);
            RT_EXPECT(sum == rrc_dataset_total_tokens(d), "accepted corrupt file is inconsistent");
            rrc_dataset_free(d);
        }
        d = NULL;
    }
    RT_EXPECT(refused > 200, "only %d of 400 corruptions refused", refused);
}

RT_TEST(data_permutation) {
    int64_t a[50], b2[50], c[50];
    rrc_data_permutation(5, 0, 50, a);
    rrc_data_permutation(5, 0, 50, b2);
    rrc_data_permutation(5, 1, 50, c);
    int seen[50] = {0}, same = 1, ident = 1, diff_epoch = 0;
    for (int i = 0; i < 50; i++) {
        RT_ASSERT(a[i] >= 0 && a[i] < 50, "range");
        seen[a[i]]++;
        same &= a[i] == b2[i];
        ident &= a[i] == i;
        diff_epoch += a[i] != c[i];
    }
    for (int i = 0; i < 50; i++) RT_EXPECT(seen[i] == 1, "not a permutation");
    RT_EXPECT(same && !ident && diff_epoch > 10, "deterministic, shuffled, epoch-dependent");
    /* pinned values of the documented generator (also implemented in tools/torch_reference.py) */
    int64_t p[5];
    rrc_data_permutation(0, 0, 5, p);
    uint64_t s0 = 0;
    uint64_t st = rrc_splitmix64_next(&s0) ^ 0xD1B54A32D192ED03ull;
    int64_t q[5] = {0, 1, 2, 3, 4};
    for (int i = 4; i >= 1; i--) {
        int64_t j = (int64_t)(rrc_splitmix64_next(&st) % (uint64_t)(i + 1)), t = q[i];
        q[i] = q[j];
        q[j] = t;
    }
    for (int i = 0; i < 5; i++) RT_EXPECT(p[i] == q[i], "permutation formula");
    int64_t one = -1;
    rrc_data_permutation(1, 0, 1, &one);
    RT_EXPECT(one == 0, "n = 1");
}

typedef struct packed {
    int32_t tok[64 * 4 * 32], lab[64 * 4 * 32], pos[64 * 4 * 32], did[64 * 4 * 32];
    rrc_data_cursor end[64];
} packed;

static rrc_status pack_n(const rrc_dataset *d, rrc_loader_opts o, rrc_data_cursor start, int n, int hold,
                         packed *out) {
    o.max_held = hold;
    rrc_loader *l = NULL;
    RRC_CHECK(rrc_loader_create(d, &o, &start, &l));
    const int64_t sz = o.rows * o.seq_len;
    const rrc_microbatch *held[8];
    rrc_status st = RRC_OK;
    for (int k = 0; k < n && st == RRC_OK;) {
        int h = 0;
        for (; h < hold && k + h < n && st == RRC_OK; h++) st = rrc_loader_next(l, &held[h]);
        if (st != RRC_OK) break;
        for (int j = 0; j < h; j++, k++) {
            memcpy(out->tok + k * sz, held[j]->tokens, (size_t)sz * 4);
            memcpy(out->lab + k * sz, held[j]->labels, (size_t)sz * 4);
            memcpy(out->pos + k * sz, held[j]->position_ids, (size_t)sz * 4);
            memcpy(out->did + k * sz, held[j]->doc_ids, (size_t)sz * 4);
            out->end[k] = held[j]->end;
            if (st == RRC_OK) st = rrc_loader_release(l, held[j]);
        }
    }
    rrc_loader_free(l);
    return st;
}

RT_TEST(data_packing_invariants_and_prefetch_determinism) {
    static bbuf b;
    make_docs(&b, 23, 40, 9, 60); /* some documents longer than seq_len = 32 */
    const char *p = rt_path("pack.rrcbin");
    RT_ASSERT(write_file(p, b.b, b.n), "write");
    rrc_dataset *d = NULL;
    RT_OK(rrc_dataset_open(p, 60, &d));
    static packed ref, other;
    for (int trunc = 0; trunc < 2; trunc++) {
        rrc_loader_opts o;
        memset(&o, 0, sizeof o);
        o.seq_len = 32;
        o.rows = 4;
        o.seed = 123;
        o.shuffle = 1;
        o.truncation = trunc ? RRC_TRUNC_DROP : RRC_TRUNC_TRUNCATE;
        o.pad_token_id = 7;
        rrc_data_cursor c0 = {0, 0, 0};
        const int N = 40;
        o.prefetch = 0;
        RT_OK(pack_n(d, o, c0, N, 1, &ref));
        /* prefetch on / off and different hold counts: identical */
        for (int variant = 0; variant < 4; variant++) {
            o.prefetch = variant & 1;
            memset(&other, 0, sizeof other);
            RT_OK(pack_n(d, o, c0, N, variant & 2 ? 3 : 1, &other));
            RT_EXPECT(!memcmp(ref.tok, other.tok, sizeof ref.tok) && !memcmp(ref.lab, other.lab, sizeof ref.lab) &&
                          !memcmp(ref.pos, other.pos, sizeof ref.pos) && !memcmp(ref.did, other.did, sizeof ref.did),
                      "trunc %d variant %d differs", trunc, variant);
            for (int k = 0; k < N; k++)
                RT_EXPECT(ref.end[k].epoch == other.end[k].epoch && ref.end[k].index == other.end[k].index,
                          "cursor %d", k);
        }
        /* resume from the cursor after micro-batch 9 reproduces micro-batches 10.. */
        memset(&other, 0, sizeof other);
        o.prefetch = 1;
        RT_OK(pack_n(d, o, ref.end[9], N - 10, 2, &other));
        const int64_t sz = 4 * 32;
        RT_EXPECT(!memcmp(ref.tok + 10 * sz, other.tok, (size_t)((N - 10) * sz) * 4) &&
                      !memcmp(ref.lab + 10 * sz, other.lab, (size_t)((N - 10) * sz) * 4),
                  "resume from cursor");
        RT_EXPECT(ref.end[N - 1].epoch >= 2, "the run should cross epochs (epoch %lld)",
                  (long long)ref.end[N - 1].epoch);
        /* row invariants: positions restart per document, doc ids increase by one, labels are -100
         * where tokens are padding, every document is contiguous and fits */
        for (int64_t r = 0; r < (int64_t)N * 4; r++) {
            const int32_t *pos = ref.pos + r * 32, *did = ref.did + r * 32, *tok = ref.tok + r * 32,
                          *lab = ref.lab + r * 32;
            RT_ASSERT(did[0] == 0 && pos[0] == 0, "row %lld start", (long long)r);
            for (int t = 1; t < 32; t++) {
                if (did[t] == did[t - 1]) RT_ASSERT(pos[t] == pos[t - 1] + 1, "row %lld pos %d", (long long)r, t);
                else RT_ASSERT(did[t] == did[t - 1] + 1 && pos[t] == 0, "row %lld doc boundary %d", (long long)r, t);
                RT_ASSERT(lab[t] == RRC_IGNORE_INDEX || lab[t] == tok[t], "label is the token or -100");
            }
        }
    }
    rrc_dataset_free(d);
}

RT_TEST(data_loader_misuse_and_errors) {
    static bbuf b;
    make_docs(&b, 5, 10, 1, 20);
    const char *p = rt_path("misuse.rrcbin");
    RT_ASSERT(write_file(p, b.b, b.n), "write");
    rrc_dataset *d = NULL;
    RT_OK(rrc_dataset_open(p, 0, &d));
    for (int pf = 0; pf < 2; pf++) {
        rrc_loader_opts o;
        memset(&o, 0, sizeof o);
        o.seq_len = 16;
        o.rows = 2;
        o.max_held = 2;
        o.prefetch = pf;
        rrc_loader *l = NULL;
        RT_OK(rrc_loader_create(d, &o, NULL, &l));
        const rrc_microbatch *a = NULL, *c = NULL, *e = NULL;
        RT_OK(rrc_loader_next(l, &a));
        RT_OK(rrc_loader_next(l, &c));
        RT_STATUS(rrc_loader_next(l, &e), RRC_ERR_INVALID_ARG);
        RT_OK(rrc_loader_release(l, c)); /* out of order is fine */
        RT_STATUS(rrc_loader_release(l, c), RRC_ERR_INVALID_ARG);
        rrc_microbatch fake;
        RT_STATUS(rrc_loader_release(l, &fake), RRC_ERR_INVALID_ARG);
        RT_OK(rrc_loader_release(l, a));
        RT_OK(rrc_loader_next(l, &a));
        rrc_loader_free(l); /* with a held batch and the producer possibly waiting */
    }
    rrc_loader_opts o;
    memset(&o, 0, sizeof o);
    o.seq_len = 0;
    o.rows = 1;
    o.max_held = 1;
    rrc_loader *l = NULL;
    RT_STATUS(rrc_loader_create(d, &o, NULL, &l), RRC_ERR_INVALID_ARG);
    o.seq_len = 16;
    o.pad_token_id = 20;
    RT_STATUS(rrc_loader_create(d, &o, NULL, &l), RRC_ERR_RANGE);
    o.pad_token_id = 0;
    rrc_data_cursor bad = {0, 99, 0};
    RT_STATUS(rrc_loader_create(d, &o, &bad, &l), RRC_ERR_INVALID_ARG);
    o.seq_len = 1;
    o.truncation = RRC_TRUNC_DROP;
    /* documents of length 1 exist only by chance; with seq_len 1 and drop, either some remain or the
     * loader reports that none is usable */
    rrc_status st = rrc_loader_create(d, &o, NULL, &l);
    if (st == RRC_OK) rrc_loader_free(l);
    else RT_EXPECT(strstr(rrc_last_error(), "no usable documents") != NULL, "%s", rrc_last_error());
    /* many short-lived prefetching loaders (thread start / join while the ring is being filled) */
    o.seq_len = 8;
    o.truncation = RRC_TRUNC_TRUNCATE;
    o.prefetch = 1;
    o.max_held = 3;
    for (int i = 0; i < 50; i++) {
        RT_OK(rrc_loader_create(d, &o, NULL, &l));
        const rrc_microbatch *mb = NULL;
        if (i % 2) RT_OK(rrc_loader_next(l, &mb));
        rrc_loader_free(l);
    }
    rrc_dataset_free(d);
}

/* data.h: release may happen in any order, and at most max_held batches are held at once. Holding
 * one batch while more than a full turn of the ring passes through the second hold must neither
 * fail (no prefetch) nor deadlock (prefetch), and the stream must stay the reference stream. */
RT_TEST(data_loader_hold_across_ring_wrap) {
    static bbuf b;
    make_docs(&b, 13, 9, 4, 30);
    const char *p = rt_path("wrap.rrcbin");
    RT_ASSERT(write_file(p, b.b, b.n), "write");
    rrc_dataset *d = NULL;
    RT_OK(rrc_dataset_open(p, 30, &d));
    rrc_loader_opts o;
    memset(&o, 0, sizeof o);
    o.seq_len = 8;
    o.rows = 2;
    o.seed = 5;
    o.shuffle = 1;
    o.pad_token_id = 1;
    static packed ref;
    const int N = 40;
    rrc_data_cursor c0 = {0, 0, 0};
    RT_OK(pack_n(d, o, c0, N, 1, &ref));
    const int64_t sz = o.rows * o.seq_len;
    for (int variant = 0; variant < 4; variant++) {
        o.prefetch = variant & 1;
        o.max_held = variant & 2 ? 3 : 2;
        const int keep = (int)o.max_held - 1; /* batches held for the whole run */
        rrc_loader *l = NULL;
        RT_OK(rrc_loader_create(d, &o, NULL, &l));
        const rrc_microbatch *held[4] = {NULL, NULL, NULL, NULL}, *mb = NULL;
        int k = 0, ok = 1;
        for (; k < keep && ok; k++) {
            ok = rrc_loader_next(l, &held[k]) == RRC_OK;
            ok = ok && !memcmp(held[k]->tokens, ref.tok + k * sz, (size_t)sz * 4);
        }
        RT_EXPECT(ok, "variant %d: first %d batches (%s)", variant, keep, rrc_last_error());
        /* 8 * max_held further batches: four turns of the prefetch ring (2 * max_held slots) */
        for (; ok && k < keep + 8 * (int)o.max_held && k < N; k++) {
            rrc_status st = rrc_loader_next(l, &mb);
            RT_EXPECT(st == RRC_OK, "variant %d: next() #%d with %d of %lld held failed: %s", variant, k, keep,
                      (long long)o.max_held, rrc_last_error());
            if (st != RRC_OK) {
                ok = 0;
                break;
            }
            RT_EXPECT(!memcmp(mb->tokens, ref.tok + k * sz, (size_t)sz * 4) &&
                          !memcmp(mb->labels, ref.lab + k * sz, (size_t)sz * 4) &&
                          !memcmp(mb->doc_ids, ref.did + k * sz, (size_t)sz * 4) && mb->end.index == ref.end[k].index &&
                          mb->end.epoch == ref.end[k].epoch,
                      "variant %d: batch %d is not the reference stream", variant, k);
            RT_OK(rrc_loader_release(l, mb));
        }
        /* the long-held batches are still intact */
        for (int j = 0; j < keep; j++)
            if (held[j]) {
                RT_EXPECT(!memcmp(held[j]->tokens, ref.tok + j * sz, (size_t)sz * 4), "variant %d: held batch %d was "
                          "overwritten", variant, j);
                RT_OK(rrc_loader_release(l, held[j]));
            }
        rrc_loader_free(l);
    }
    rrc_dataset_free(d);
}

/* Counters near INT64_MAX: the permutation of any epoch is defined (unsigned arithmetic), and the
 * loader refuses a start cursor beyond RRC_DATA_COUNTER_MAX instead of overflowing. */
RT_TEST(data_cursor_counter_bounds) {
    int64_t perm[7];
    rrc_data_permutation(9, INT64_MAX, 7, perm);
    int seen = 0;
    for (int i = 0; i < 7; i++) seen |= 1 << perm[i];
    RT_EXPECT(seen == 0x7f, "permutation of epoch INT64_MAX");
    static bbuf b;
    make_docs(&b, 5, 6, 2, 20);
    const char *p = rt_path("bounds.rrcbin");
    RT_ASSERT(write_file(p, b.b, b.n), "write");
    rrc_dataset *d = NULL;
    RT_OK(rrc_dataset_open(p, 0, &d));
    for (int pf = 0; pf < 2; pf++) {
        rrc_loader_opts o;
        memset(&o, 0, sizeof o);
        o.seq_len = 8;
        o.rows = 1;
        o.max_held = 1;
        o.shuffle = 1;
        o.prefetch = pf;
        rrc_loader *l = NULL;
        const rrc_data_cursor bad[] = {{INT64_MAX, 5, 0}, {INT64_MAX, 0, 0}, {INT64_MAX - 1, 0, 0},
                                       {0, 0, INT64_MAX}, {RRC_DATA_COUNTER_MAX + 1, 0, 0}};
        for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
            RT_STATUS(rrc_loader_create(d, &o, &bad[i], &l), RRC_ERR_INVALID_ARG);
            RT_EXPECT(l == NULL, "loader set on failure");
        }
        const rrc_data_cursor edge = {RRC_DATA_COUNTER_MAX, 5, RRC_DATA_COUNTER_MAX};
        RT_OK(rrc_loader_create(d, &o, &edge, &l));
        const rrc_microbatch *mb = NULL;
        RT_OK(rrc_loader_next(l, &mb));
        RT_EXPECT(mb->start.epoch == RRC_DATA_COUNTER_MAX + 1 && mb->start.index == 0, "edge cursor normalized");
        RT_OK(rrc_loader_release(l, mb));
        rrc_loader_free(l);
    }
    rrc_dataset_free(d);
}

const rt_case rt_suite_data[] = {
    RT_CASE(rrcbin_valid_and_doc_access), RT_CASE(rrcbin_malformed_cases), RT_CASE(rrcbin_truncation_and_corruption_fuzz),
    RT_CASE(data_permutation), RT_CASE(data_packing_invariants_and_prefetch_determinism),
    RT_CASE(data_loader_misuse_and_errors), RT_CASE(data_loader_hold_across_ring_wrap),
    RT_CASE(data_cursor_counter_bounds),
};
const size_t rt_suite_data_n = sizeof rt_suite_data / sizeof rt_suite_data[0];
