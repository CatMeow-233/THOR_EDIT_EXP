/* rrc/src/data.c — .rrcbin reader, deterministic shuffling / packing, prefetch thread
 * (format and semantics: rrc/include/rrc/data.h). */
#include "rrc/data.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "rrc/backend.h" /* RRC_IGNORE_INDEX */

/* ---- dataset ---------------------------------------------------------------------------------- */

struct rrc_dataset {
    const unsigned char *map;
    size_t size;
    int64_t num_docs, vocab, total, label_tokens, max_len;
    uint64_t hash;
    int64_t *off; /* byte offset of each document's len field */
};

static uint64_t rd_u64(const unsigned char *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}
static uint32_t rd_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t fnv1a64(const unsigned char *p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

void rrc_dataset_free(rrc_dataset *d) {
    if (!d) return;
    if (d->map) munmap((void *)d->map, d->size);
    free(d->off);
    free(d);
}

static rrc_status validate(rrc_dataset *d, const char *path, int64_t model_vocab) {
    const unsigned char *m = d->map;
    const uint64_t size = (uint64_t)d->size;
    RRC_REQUIRE(memcmp(m, RRC_RRCBIN_MAGIC, 8) == 0, RRC_ERR_PARSE,
                "%s: bad magic (expected \"RRCBIN01\"): not an .rrcbin file or an unsupported version", path);
    const uint64_t nd = rd_u64(m + 8), vocab = rd_u64(m + 16), total = rd_u64(m + 24);
    RRC_REQUIRE(nd >= 1, RRC_ERR_PARSE, "%s: num_docs is 0 (no documents)", path);
    /* each document takes at least 4 + 5 bytes: bound num_docs before allocating anything */
    RRC_REQUIRE(nd <= (size - RRC_RRCBIN_HEADER_BYTES) / 9, RRC_ERR_PARSE,
                "%s: header says num_docs = %llu but the file (%llu bytes) can hold at most %llu documents "
                "(truncated file?)",
                path, (unsigned long long)nd, (unsigned long long)size,
                (unsigned long long)((size - RRC_RRCBIN_HEADER_BYTES) / 9));
    RRC_REQUIRE(vocab >= 1 && vocab <= (uint64_t)INT32_MAX, RRC_ERR_PARSE,
                "%s: vocab_size = %llu out of range [1, 2^31 - 1]", path, (unsigned long long)vocab);
    RRC_REQUIRE(model_vocab <= 0 || vocab <= (uint64_t)model_vocab, RRC_ERR_PARSE,
                "%s: vocab_size = %llu is larger than the model's vocabulary (%lld): tokenized with a different "
                "tokenizer?",
                path, (unsigned long long)vocab, (long long)model_vocab);
    const uint64_t lim = model_vocab > 0 && (uint64_t)model_vocab < vocab ? (uint64_t)model_vocab : vocab;
    d->off = (int64_t *)malloc((size_t)nd * sizeof(int64_t));
    if (!d->off) return RRC_ERROR(RRC_ERR_OOM, "%s: document index (%llu documents)", path, (unsigned long long)nd);
    uint64_t pos = RRC_RRCBIN_HEADER_BYTES, sum = 0, labels = 0, max_len = 0;
    for (uint64_t i = 0; i < nd; i++) {
        RRC_REQUIRE(size - pos >= 4, RRC_ERR_PARSE,
                    "%s: truncated file: document %llu's length field at byte %llu runs past the end (%llu bytes)",
                    path, (unsigned long long)i, (unsigned long long)pos, (unsigned long long)size);
        const uint64_t len = rd_u32(m + pos);
        RRC_REQUIRE(len >= 1, RRC_ERR_PARSE, "%s: document %llu (byte %llu) has len 0", path, (unsigned long long)i,
                    (unsigned long long)pos);
        RRC_REQUIRE((size - pos - 4) / 5 >= len, RRC_ERR_PARSE,
                    "%s: truncated file: document %llu (byte %llu, len %llu) needs %llu bytes, only %llu remain", path,
                    (unsigned long long)i, (unsigned long long)pos, (unsigned long long)len,
                    (unsigned long long)(4 + 5 * len), (unsigned long long)(size - pos));
        d->off[i] = (int64_t)pos;
        const unsigned char *tok = m + pos + 4, *mask = tok + 4 * len;
        for (uint64_t t = 0; t < len; t++) {
            const uint32_t v = rd_u32(tok + 4 * t);
            if ((uint64_t)v >= lim)
                return RRC_ERROR(RRC_ERR_RANGE,
                                 "%s: document %llu token %llu = %u is out of range (must be < %llu: header vocab_size "
                                 "%llu, model vocabulary %lld)",
                                 path, (unsigned long long)i, (unsigned long long)t, v, (unsigned long long)lim,
                                 (unsigned long long)vocab, (long long)model_vocab);
            if (mask[t] > 1)
                return RRC_ERROR(RRC_ERR_PARSE, "%s: document %llu label_mask[%llu] = %u (must be 0 or 1)", path,
                                 (unsigned long long)i, (unsigned long long)t, (unsigned)mask[t]);
            labels += mask[t];
        }
        sum += len;
        if (len > max_len) max_len = len;
        pos += 4 + 5 * len;
    }
    RRC_REQUIRE(pos == size, RRC_ERR_PARSE, "%s: %llu trailing bytes after the last of %llu documents", path,
                (unsigned long long)(size - pos), (unsigned long long)nd);
    RRC_REQUIRE(sum == total, RRC_ERR_PARSE, "%s: header total_tokens = %llu but the documents hold %llu tokens", path,
                (unsigned long long)total, (unsigned long long)sum);
    d->num_docs = (int64_t)nd;
    d->vocab = (int64_t)vocab;
    d->total = (int64_t)total;
    d->label_tokens = (int64_t)labels;
    d->max_len = (int64_t)max_len;
    d->hash = fnv1a64(m, d->size);
    return RRC_OK;
}

rrc_status rrc_dataset_open(const char *path, int64_t model_vocab, rrc_dataset **out) {
    RRC_REQUIRE(path && out, RRC_ERR_INVALID_ARG, "null argument");
    *out = NULL;
    const uint16_t one = 1;
    unsigned char first;
    memcpy(&first, &one, 1);
    RRC_REQUIRE(first == 1, RRC_ERR_UNSUPPORTED, ".rrcbin reading needs a little-endian host");
    int fd = open(path, O_RDONLY);
    if (fd < 0) return RRC_ERROR(errno == ENOENT ? RRC_ERR_NOT_FOUND : RRC_ERR_IO, "cannot open %s: %s", path,
                                 strerror(errno));
    struct stat sb;
    if (fstat(fd, &sb) != 0) {
        rrc_status st = RRC_ERROR(RRC_ERR_IO, "fstat %s: %s", path, strerror(errno));
        close(fd);
        return st;
    }
    if (!S_ISREG(sb.st_mode)) {
        close(fd);
        return RRC_ERROR(RRC_ERR_IO, "%s is not a regular file", path);
    }
    if ((uint64_t)sb.st_size < RRC_RRCBIN_HEADER_BYTES) {
        close(fd);
        return RRC_ERROR(RRC_ERR_PARSE, "%s: truncated file: %lld bytes, the header alone is %d bytes", path,
                         (long long)sb.st_size, RRC_RRCBIN_HEADER_BYTES);
    }
    rrc_dataset *d = (rrc_dataset *)calloc(1, sizeof *d);
    if (!d) {
        close(fd);
        return RRC_ERROR(RRC_ERR_OOM, "rrc_dataset");
    }
    d->size = (size_t)sb.st_size;
    void *map = mmap(NULL, d->size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        free(d);
        return RRC_ERROR(RRC_ERR_IO, "mmap %s: %s", path, strerror(errno));
    }
    d->map = (const unsigned char *)map;
    rrc_status st = validate(d, path, model_vocab);
    if (st != RRC_OK) {
        rrc_dataset_free(d);
        return st;
    }
    *out = d;
    return RRC_OK;
}

int64_t rrc_dataset_num_docs(const rrc_dataset *d) { return d ? d->num_docs : 0; }
int64_t rrc_dataset_vocab_size(const rrc_dataset *d) { return d ? d->vocab : 0; }
int64_t rrc_dataset_total_tokens(const rrc_dataset *d) { return d ? d->total : 0; }
int64_t rrc_dataset_label_tokens(const rrc_dataset *d) { return d ? d->label_tokens : 0; }
int64_t rrc_dataset_max_len(const rrc_dataset *d) { return d ? d->max_len : 0; }
uint64_t rrc_dataset_hash(const rrc_dataset *d) { return d ? d->hash : 0; }

int64_t rrc_dataset_doc_len(const rrc_dataset *d, int64_t i) {
    if (!d || i < 0 || i >= d->num_docs) return 0;
    return (int64_t)rd_u32(d->map + d->off[i]);
}

rrc_status rrc_dataset_doc_copy(const rrc_dataset *d, int64_t i, int64_t n, int32_t *tokens, uint8_t *mask) {
    RRC_REQUIRE(d && (tokens || n == 0) && (mask || n == 0) && n >= 0, RRC_ERR_INVALID_ARG, "null argument");
    RRC_REQUIRE(i >= 0 && i < d->num_docs, RRC_ERR_RANGE, "document %lld outside [0, %lld)", (long long)i,
                (long long)d->num_docs);
    const unsigned char *p = d->map + d->off[i];
    const int64_t len = (int64_t)rd_u32(p);
    const int64_t c = n < len ? n : len;
    for (int64_t t = 0; t < c; t++) tokens[t] = (int32_t)rd_u32(p + 4 + 4 * t); /* validated < 2^31 */
    if (c) memcpy(mask, p + 4 + 4 * len, (size_t)c);
    return RRC_OK;
}

/* ---- shuffling -------------------------------------------------------------------------------- */

uint64_t rrc_splitmix64_next(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void rrc_data_permutation(uint64_t seed, int64_t epoch, int64_t n, int64_t *perm) {
    uint64_t s0 = seed;
    uint64_t state = rrc_splitmix64_next(&s0) ^ (0xD1B54A32D192ED03ull * (uint64_t)(epoch + 1));
    for (int64_t i = 0; i < n; i++) perm[i] = i;
    for (int64_t i = n - 1; i >= 1; i--) {
        const int64_t j = (int64_t)(rrc_splitmix64_next(&state) % (uint64_t)(i + 1));
        const int64_t t = perm[i];
        perm[i] = perm[j];
        perm[j] = t;
    }
}

/* ---- loader ----------------------------------------------------------------------------------- */

enum { SLOT_FREE = 0, SLOT_BUSY = 1, SLOT_READY = 2, SLOT_HELD = 3 };

typedef struct slot {
    rrc_microbatch mb; /* first member: a held rrc_microbatch pointer maps back to its slot */
    int state;
    rrc_status st;
    char err[RRC_ERROR_MSG_MAX];
} slot;

struct rrc_loader {
    const rrc_dataset *d;
    rrc_loader_opts o;
    int64_t *usable;
    int64_t n_usable;
    /* producer-owned state */
    int64_t *order;
    int64_t order_epoch; /* epoch `order` was built for (-1: none) */
    rrc_data_cursor cur;
    uint8_t *mask_tmp;
    /* ring */
    slot *slots;
    int64_t cap;
    int64_t next_produce, next_consume, held;
    int64_t *int_store;
    /* thread */
    int threaded;
    pthread_t th;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int sync_init;
    int stop;
};

static void normalize_cursor(rrc_loader *l, rrc_data_cursor *c) {
    while (c->index >= l->n_usable) {
        c->index -= l->n_usable;
        c->epoch++;
    }
}

static rrc_status ensure_order(rrc_loader *l, int64_t epoch) {
    if (l->order_epoch == epoch) return RRC_OK;
    if (l->o.shuffle) rrc_data_permutation(l->o.seed, epoch, l->n_usable, l->order);
    else
        for (int64_t i = 0; i < l->n_usable; i++) l->order[i] = i;
    l->order_epoch = epoch;
    return RRC_OK;
}

/* Build the next micro-batch at the producer cursor (host memory only). */
static rrc_status produce(rrc_loader *l, rrc_microbatch *mb) {
    const int64_t T = l->o.seq_len;
    mb->start = l->cur;
    mb->n_docs = mb->n_input_tokens = mb->n_truncated = 0;
    for (int64_t r = 0; r < l->o.rows; r++) {
        int32_t *tok = mb->tokens + r * T, *lab = mb->labels + r * T, *pos = mb->position_ids + r * T,
                *doc = mb->doc_ids + r * T;
        int64_t t = 0, k = 0;
        while (t < T) {
            RRC_CHECK(ensure_order(l, l->cur.epoch));
            const int64_t di = l->usable[l->order[l->cur.index]];
            const int64_t len = rrc_dataset_doc_len(l->d, di);
            const int64_t n = len < T ? len : T;
            if (n > T - t) break; /* does not fit: close the row (t > 0 here because n <= T) */
            RRC_CHECK(rrc_dataset_doc_copy(l->d, di, n, tok + t, l->mask_tmp));
            for (int64_t i = 0; i < n; i++) {
                lab[t + i] = l->mask_tmp[i] ? tok[t + i] : RRC_IGNORE_INDEX;
                pos[t + i] = (int32_t)i;
                doc[t + i] = (int32_t)k;
            }
            t += n;
            k++;
            mb->n_docs++;
            mb->n_input_tokens += n;
            mb->n_truncated += len > n;
            l->cur.index++;
            normalize_cursor(l, &l->cur);
        }
        for (int64_t i = t; i < T; i++) {
            tok[i] = l->o.pad_token_id;
            lab[i] = RRC_IGNORE_INDEX;
            pos[i] = (int32_t)(i - t);
            doc[i] = (int32_t)k;
        }
    }
    l->cur.micro_batches++;
    mb->end = l->cur;
    return RRC_OK;
}

static void slot_store_error(slot *s, rrc_status st) {
    s->st = st;
    snprintf(s->err, sizeof s->err, "%s", st == RRC_OK ? "" : rrc_last_error());
}

static void *producer_main(void *arg) {
    rrc_loader *l = (rrc_loader *)arg;
    for (;;) {
        slot *s = &l->slots[l->next_produce % l->cap];
        pthread_mutex_lock(&l->mu);
        while (s->state != SLOT_FREE && !l->stop) pthread_cond_wait(&l->cv, &l->mu);
        if (l->stop) {
            pthread_mutex_unlock(&l->mu);
            break;
        }
        s->state = SLOT_BUSY;
        pthread_mutex_unlock(&l->mu);
        rrc_status st = produce(l, &s->mb);
        pthread_mutex_lock(&l->mu);
        slot_store_error(s, st);
        s->state = SLOT_READY;
        pthread_cond_broadcast(&l->cv);
        pthread_mutex_unlock(&l->mu);
        if (st != RRC_OK) break; /* the consumer re-raises it when it reaches this slot */
        l->next_produce++;
    }
    return NULL;
}

void rrc_loader_free(rrc_loader *l) {
    if (!l) return;
    if (l->threaded) {
        pthread_mutex_lock(&l->mu);
        l->stop = 1;
        pthread_cond_broadcast(&l->cv);
        pthread_mutex_unlock(&l->mu);
        pthread_join(l->th, NULL);
    }
    if (l->sync_init) {
        pthread_mutex_destroy(&l->mu);
        pthread_cond_destroy(&l->cv);
    }
    free(l->usable);
    free(l->order);
    free(l->mask_tmp);
    free(l->int_store);
    free(l->slots);
    free(l);
}

rrc_status rrc_loader_create(const rrc_dataset *d, const rrc_loader_opts *o, const rrc_data_cursor *start,
                             rrc_loader **out) {
    RRC_REQUIRE(d && o && out, RRC_ERR_INVALID_ARG, "null argument");
    *out = NULL;
    RRC_REQUIRE(o->seq_len >= 1 && o->seq_len <= INT32_MAX, RRC_ERR_INVALID_ARG, "seq_len = %lld out of range",
                (long long)o->seq_len);
    RRC_REQUIRE(o->rows >= 1 && o->rows <= INT32_MAX / o->seq_len, RRC_ERR_INVALID_ARG,
                "rows = %lld out of range (rows * seq_len must be < 2^31)", (long long)o->rows);
    RRC_REQUIRE(o->max_held >= 1 && o->max_held <= 65536, RRC_ERR_INVALID_ARG, "max_held = %lld out of range",
                (long long)o->max_held);
    RRC_REQUIRE(o->truncation == RRC_TRUNC_TRUNCATE || o->truncation == RRC_TRUNC_DROP, RRC_ERR_INVALID_ARG,
                "invalid truncation policy %d", (int)o->truncation);
    RRC_REQUIRE(o->pad_token_id >= 0 && o->pad_token_id < d->vocab, RRC_ERR_RANGE,
                "pad_token_id = %d outside the vocabulary [0, %lld)", (int)o->pad_token_id, (long long)d->vocab);
    RRC_REQUIRE(!start || (start->epoch >= 0 && start->index >= 0 && start->micro_batches >= 0), RRC_ERR_INVALID_ARG,
                "negative data cursor");
    rrc_loader *l = (rrc_loader *)calloc(1, sizeof *l);
    if (!l) return RRC_ERROR(RRC_ERR_OOM, "rrc_loader");
    l->d = d;
    l->o = *o;
    l->order_epoch = -1;
    rrc_status st = RRC_OK;
    l->usable = (int64_t *)malloc((size_t)d->num_docs * sizeof(int64_t));
    l->order = (int64_t *)malloc((size_t)d->num_docs * sizeof(int64_t));
    l->mask_tmp = (uint8_t *)malloc((size_t)o->seq_len);
    if (!l->usable || !l->order || !l->mask_tmp) {
        st = RRC_ERROR(RRC_ERR_OOM, "loader tables");
        goto fail;
    }
    for (int64_t i = 0; i < d->num_docs; i++)
        if (o->truncation == RRC_TRUNC_TRUNCATE || rrc_dataset_doc_len(d, i) <= o->seq_len) l->usable[l->n_usable++] = i;
    if (l->n_usable == 0) {
        st = RRC_ERROR(RRC_ERR_INVALID_ARG,
                       "no usable documents: all %lld documents are longer than seq_len = %lld and truncation is "
                       "\"drop\"",
                       (long long)d->num_docs, (long long)o->seq_len);
        goto fail;
    }
    if (start) {
        if (start->index > l->n_usable) {
            st = RRC_ERROR(RRC_ERR_INVALID_ARG,
                           "data cursor index %lld exceeds the %lld usable documents (different data or truncation "
                           "policy?)",
                           (long long)start->index, (long long)l->n_usable);
            goto fail;
        }
        l->cur = *start;
    }
    normalize_cursor(l, &l->cur);
    l->cap = o->prefetch ? 2 * o->max_held : o->max_held;
    const int64_t n = o->rows * o->seq_len;
    l->slots = (slot *)calloc((size_t)l->cap, sizeof(slot));
    l->int_store = (int64_t *)malloc((size_t)(l->cap * 2 * n) * sizeof(int64_t)); /* 4 int32 arrays per slot */
    if (!l->slots || !l->int_store) {
        st = RRC_ERROR(RRC_ERR_OOM, "loader ring (%lld micro-batches of %lld tokens)", (long long)l->cap, (long long)n);
        goto fail;
    }
    for (int64_t s = 0; s < l->cap; s++) {
        int32_t *base = (int32_t *)(l->int_store + s * 2 * n);
        rrc_microbatch *mb = &l->slots[s].mb;
        mb->rows = o->rows;
        mb->seq_len = o->seq_len;
        mb->tokens = base;
        mb->labels = base + n;
        mb->position_ids = base + 2 * n;
        mb->doc_ids = base + 3 * n;
    }
    if (pthread_mutex_init(&l->mu, NULL) != 0) {
        st = RRC_ERROR(RRC_ERR_INTERNAL, "pthread_mutex_init failed");
        goto fail;
    }
    if (pthread_cond_init(&l->cv, NULL) != 0) {
        pthread_mutex_destroy(&l->mu);
        st = RRC_ERROR(RRC_ERR_INTERNAL, "pthread_cond_init failed");
        goto fail;
    }
    l->sync_init = 1;
    if (o->prefetch) {
        int e = pthread_create(&l->th, NULL, producer_main, l);
        if (e != 0) {
            st = RRC_ERROR(RRC_ERR_INTERNAL, "cannot start the prefetch thread: %s", strerror(e));
            goto fail;
        }
        l->threaded = 1;
    }
    *out = l;
    return RRC_OK;
fail:
    rrc_loader_free(l);
    return st;
}

int64_t rrc_loader_usable_docs(const rrc_loader *l) { return l ? l->n_usable : 0; }

rrc_status rrc_loader_next(rrc_loader *l, const rrc_microbatch **out) {
    RRC_REQUIRE(l && out, RRC_ERR_INVALID_ARG, "null argument");
    *out = NULL;
    RRC_REQUIRE(l->held < l->o.max_held, RRC_ERR_INVALID_ARG,
                "the consumer already holds %lld micro-batches (max_held): release one first", (long long)l->held);
    slot *s = &l->slots[l->next_consume % l->cap];
    if (!l->threaded) {
        RRC_REQUIRE(s->state == SLOT_FREE, RRC_ERR_INTERNAL, "loader slot not free");
        RRC_CHECK_CTX(produce(l, &s->mb), "data loader");
        s->state = SLOT_HELD;
    } else {
        pthread_mutex_lock(&l->mu);
        while (s->state != SLOT_READY) pthread_cond_wait(&l->cv, &l->mu);
        const rrc_status st = s->st;
        if (st == RRC_OK) s->state = SLOT_HELD;
        pthread_mutex_unlock(&l->mu);
        if (st != RRC_OK) return RRC_ERROR(st, "data loader (prefetch thread): %s", s->err);
    }
    l->next_consume++;
    l->held++;
    *out = &s->mb;
    return RRC_OK;
}

rrc_status rrc_loader_release(rrc_loader *l, const rrc_microbatch *mb) {
    RRC_REQUIRE(l && mb, RRC_ERR_INVALID_ARG, "null argument");
    const char *p = (const char *)mb, *b = (const char *)l->slots;
    RRC_REQUIRE(p >= b && p < b + (size_t)l->cap * sizeof(slot) && (size_t)(p - b) % sizeof(slot) == 0,
                RRC_ERR_INVALID_ARG, "micro-batch does not belong to this loader");
    slot *s = &l->slots[(size_t)(p - b) / sizeof(slot)];
    if (l->threaded) pthread_mutex_lock(&l->mu);
    const int ok = s->state == SLOT_HELD;
    if (ok) {
        s->state = SLOT_FREE;
        l->held--;
        if (l->threaded) pthread_cond_broadcast(&l->cv);
    }
    if (l->threaded) pthread_mutex_unlock(&l->mu);
    RRC_REQUIRE(ok, RRC_ERR_INVALID_ARG, "micro-batch released twice");
    return RRC_OK;
}
