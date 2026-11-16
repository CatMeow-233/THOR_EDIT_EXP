/* rrc/src/safetensors.c — mmap reader (single file + sharded index) and writer. */
#include "rrc/safetensors.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "rrc/fileio.h"
#include "rrc/json.h"

static const char *const k_st_names[RRC_ST_DTYPE_COUNT_] = {"F32", "BF16", "F16", "I32", "I64"};
static const size_t k_st_sizes[RRC_ST_DTYPE_COUNT_] = {4, 2, 2, 4, 8};
/* Dtypes of the safetensors format that rrc recognises but does not support. */
static const char *const k_st_unsupported[] = {"F64", "I8", "U8", "I16", "U16", "U32", "U64", "BOOL",
                                               "F8_E4M3", "F8_E5M2", "F8_E8M0", "F4", "F6_E2M3",
                                               "F6_E3M2", "C64", "F8_E4M3FNUZ", "F8_E5M2FNUZ"};

const char *rrc_st_dtype_name(rrc_st_dtype dt) {
    return ((int)dt >= 0 && dt < RRC_ST_DTYPE_COUNT_) ? k_st_names[dt] : "?";
}
size_t rrc_st_dtype_size(rrc_st_dtype dt) { return ((int)dt >= 0 && dt < RRC_ST_DTYPE_COUNT_) ? k_st_sizes[dt] : 0; }

rrc_status rrc_st_dtype_to_rrc(rrc_st_dtype dt, rrc_dtype *out) {
    switch (dt) {
    case RRC_ST_F32: *out = RRC_F32; return RRC_OK;
    case RRC_ST_BF16: *out = RRC_BF16; return RRC_OK;
    case RRC_ST_I32: *out = RRC_I32; return RRC_OK;
    default:
        return RRC_ERROR(RRC_ERR_UNSUPPORTED, "safetensors dtype %s has no engine dtype (convert explicitly)",
                         rrc_st_dtype_name(dt));
    }
}

/* ------------------------------------------------------------------------------------------ */

typedef struct st_file {
    char *path;
    void *map;
    size_t map_len;
    rrc_json_doc *header;
    const rrc_json *metadata; /* may be NULL */
} st_file;

struct rrc_st_set {
    st_file *files;
    size_t n_files;
    rrc_st_tensor *tensors;
    size_t n_tensors;
};

void rrc_st_close(rrc_st_set *set) {
    if (!set) return;
    for (size_t i = 0; i < set->n_files; i++) {
        st_file *f = &set->files[i];
        if (f->map && f->map != MAP_FAILED) munmap(f->map, f->map_len);
        rrc_json_doc_free(f->header);
        free(f->path);
    }
    free(set->files);
    free(set->tensors);
    free(set);
}

static uint64_t rd_u64le(const unsigned char *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

static int cmp_tensor_name(const void *a, const void *b) {
    return strcmp(((const rrc_st_tensor *)a)->name, ((const rrc_st_tensor *)b)->name);
}

typedef struct {
    uint64_t begin, end;
    const char *name;
} span;

static int cmp_span(const void *a, const void *b) {
    const span *x = (const span *)a, *y = (const span *)b;
    if (x->begin != y->begin) return x->begin < y->begin ? -1 : 1;
    if (x->end != y->end) return x->end < y->end ? -1 : 1;
    return 0;
}

static rrc_status parse_dtype(const char *s, const char *tname, rrc_st_dtype *out) {
    for (int i = 0; i < RRC_ST_DTYPE_COUNT_; i++)
        if (!strcmp(s, k_st_names[i])) {
            *out = (rrc_st_dtype)i;
            return RRC_OK;
        }
    for (size_t i = 0; i < sizeof k_st_unsupported / sizeof k_st_unsupported[0]; i++)
        if (!strcmp(s, k_st_unsupported[i]))
            return RRC_ERROR(RRC_ERR_UNSUPPORTED, "tensor '%s': dtype %s is not supported (F32/BF16/F16/I32/I64)",
                             tname, s);
    return RRC_ERROR(RRC_ERR_PARSE, "tensor '%s': unknown dtype '%s'", tname, s);
}

/* Map one file, parse and validate its header, and append its tensors to `set`. */
static rrc_status load_file(rrc_st_set *set, const char *path) {
    st_file *nf = (st_file *)realloc(set->files, (set->n_files + 1) * sizeof(st_file));
    if (!nf) return RRC_ERROR(RRC_ERR_OOM, "safetensors file table");
    set->files = nf;
    st_file *f = &set->files[set->n_files];
    memset(f, 0, sizeof *f);
    set->n_files++;
    int shard = (int)(set->n_files - 1);
    f->path = strdup(path);
    if (!f->path) return RRC_ERROR(RRC_ERR_OOM, "path");

    int fd = -1;
    RRC_CHECK(rrc_open_regular(path, &fd));
    struct stat stt;
    if (fstat(fd, &stt) != 0) {
        rrc_status s = RRC_ERROR(RRC_ERR_IO, "fstat %s: %s", path, strerror(errno));
        close(fd);
        return s;
    }
    uint64_t fsize = (uint64_t)stt.st_size;
    if (fsize < 8) {
        close(fd);
        return RRC_ERROR(RRC_ERR_PARSE, "%s: file too small for a safetensors header (%llu bytes)", path,
                         (unsigned long long)fsize);
    }
    f->map_len = (size_t)fsize;
    f->map = mmap(NULL, f->map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (f->map == MAP_FAILED) {
        f->map = NULL;
        return RRC_ERROR(RRC_ERR_IO, "mmap %s: %s", path, strerror(errno));
    }
    const unsigned char *base = (const unsigned char *)f->map;
    uint64_t hlen = rd_u64le(base);
    if (hlen == 0 || hlen > RRC_ST_MAX_HEADER)
        return RRC_ERROR(RRC_ERR_PARSE, "%s: invalid header length %llu", path, (unsigned long long)hlen);
    if (hlen > fsize - 8)
        return RRC_ERROR(RRC_ERR_PARSE, "%s: header length %llu exceeds file size %llu", path,
                         (unsigned long long)hlen, (unsigned long long)fsize);
    if (base[8] != '{') return RRC_ERROR(RRC_ERR_PARSE, "%s: header does not start with '{'", path);
    RRC_CHECK_CTX(rrc_json_parse((const char *)base + 8, (size_t)hlen, &f->header), "%s: header", path);
    const rrc_json *root = rrc_json_root(f->header);
    if (root->type != RRC_JSON_OBJECT) return RRC_ERROR(RRC_ERR_PARSE, "%s: header is not an object", path);
    const unsigned char *data = base + 8 + hlen;
    uint64_t data_len = fsize - 8 - hlen;

    size_t n_entries = rrc_json_obj_count(root);
    rrc_st_tensor *nt = (rrc_st_tensor *)realloc(set->tensors, (set->n_tensors + n_entries + 1) * sizeof(rrc_st_tensor));
    if (!nt) return RRC_ERROR(RRC_ERR_OOM, "tensor table");
    set->tensors = nt;
    span *spans = (span *)malloc((n_entries + 1) * sizeof(span));
    if (!spans) return RRC_ERROR(RRC_ERR_OOM, "span table");
    size_t n_spans = 0;
    rrc_status st = RRC_OK;

    for (size_t i = 0; i < n_entries; i++) {
        const char *name = rrc_json_obj_key(root, i);
        const rrc_json *e = rrc_json_obj_val(root, i);
        if (!strcmp(name, "__metadata__")) {
            if (e->type != RRC_JSON_OBJECT) {
                st = RRC_ERROR(RRC_ERR_PARSE, "%s: __metadata__ is not an object", path);
                goto done;
            }
            for (size_t k = 0; k < rrc_json_obj_count(e); k++)
                if (rrc_json_obj_val(e, k)->type != RRC_JSON_STRING) {
                    st = RRC_ERROR(RRC_ERR_PARSE, "%s: __metadata__ value of '%s' is not a string", path,
                                   rrc_json_obj_key(e, k));
                    goto done;
                }
            f->metadata = e;
            continue;
        }
        if (name[0] == '\0' || strlen(name) != root->u.o.key_lens[i]) {
            st = RRC_ERROR(RRC_ERR_PARSE, "%s: empty tensor name or name with NUL byte", path);
            goto done;
        }
        if (e->type != RRC_JSON_OBJECT || rrc_json_obj_count(e) != 3) {
            st = RRC_ERROR(RRC_ERR_PARSE, "%s: tensor '%s': entry must be an object with dtype, shape, data_offsets",
                           path, name);
            goto done;
        }
        const char *dts = rrc_json_str(rrc_json_get(e, "dtype"));
        const rrc_json *shape = rrc_json_get(e, "shape");
        const rrc_json *offs = rrc_json_get(e, "data_offsets");
        if (!dts || !shape || shape->type != RRC_JSON_ARRAY || !offs || offs->type != RRC_JSON_ARRAY ||
            rrc_json_arr_count(offs) != 2) {
            st = RRC_ERROR(RRC_ERR_PARSE, "%s: tensor '%s': malformed dtype/shape/data_offsets", path, name);
            goto done;
        }
        rrc_st_tensor *t = &set->tensors[set->n_tensors];
        memset(t, 0, sizeof *t);
        t->name = name;
        t->shard = shard;
        if ((st = parse_dtype(dts, name, &t->dtype)) != RRC_OK) goto done;
        size_t nd = rrc_json_arr_count(shape);
        if (nd > RRC_ST_MAX_DIMS) {
            st = RRC_ERROR(RRC_ERR_PARSE, "%s: tensor '%s': %zu dims > %d", path, name, nd, RRC_ST_MAX_DIMS);
            goto done;
        }
        t->ndim = (int)nd;
        int64_t numel = 1;
        uint64_t esz = rrc_st_dtype_size(t->dtype);
        for (size_t d = 0; d < nd; d++) {
            const rrc_json *dim = rrc_json_arr_get(shape, d);
            if (dim->type != RRC_JSON_INT || dim->u.i < 0) {
                st = RRC_ERROR(RRC_ERR_PARSE, "%s: tensor '%s': shape[%zu] is not a non-negative integer", path, name, d);
                goto done;
            }
            t->shape[d] = dim->u.i;
            if (dim->u.i != 0 && numel > (int64_t)(INT64_MAX / (int64_t)esz) / dim->u.i) {
                st = RRC_ERROR(RRC_ERR_PARSE, "%s: tensor '%s': element count overflows", path, name);
                goto done;
            }
            numel *= dim->u.i;
        }
        t->numel = numel;
        const rrc_json *ob = rrc_json_arr_get(offs, 0), *oe = rrc_json_arr_get(offs, 1);
        if (ob->type != RRC_JSON_INT || oe->type != RRC_JSON_INT || ob->u.i < 0 || oe->u.i < 0) {
            st = RRC_ERROR(RRC_ERR_PARSE, "%s: tensor '%s': data_offsets must be non-negative integers", path, name);
            goto done;
        }
        uint64_t b = (uint64_t)ob->u.i, en = (uint64_t)oe->u.i;
        if (en < b) {
            st = RRC_ERROR(RRC_ERR_PARSE, "%s: tensor '%s': data_offsets end < begin", path, name);
            goto done;
        }
        if (en - b != (uint64_t)numel * esz) {
            st = RRC_ERROR(RRC_ERR_PARSE, "%s: tensor '%s': %llu bytes in data_offsets but shape needs %llu", path,
                           name, (unsigned long long)(en - b), (unsigned long long)((uint64_t)numel * esz));
            goto done;
        }
        if (en > data_len) {
            st = RRC_ERROR(RRC_ERR_PARSE, "%s: tensor '%s': data_offsets end %llu beyond data block (%llu bytes)",
                           path, name, (unsigned long long)en, (unsigned long long)data_len);
            goto done;
        }
        t->data = data + b;
        t->nbytes = en - b;
        spans[n_spans].begin = b;
        spans[n_spans].end = en;
        spans[n_spans].name = name;
        n_spans++;
        set->n_tensors++;
    }
    /* Overlap / hole / coverage check (the reference implementation requires exact tiling). */
    qsort(spans, n_spans, sizeof(span), cmp_span);
    uint64_t cursor = 0;
    for (size_t i = 0; i < n_spans; i++) {
        if (spans[i].begin < cursor) {
            st = RRC_ERROR(RRC_ERR_PARSE, "%s: tensor '%s' overlaps the previous tensor (offset %llu < %llu)", path,
                           spans[i].name, (unsigned long long)spans[i].begin, (unsigned long long)cursor);
            goto done;
        }
        if (spans[i].begin > cursor) {
            st = RRC_ERROR(RRC_ERR_PARSE, "%s: hole in data block before tensor '%s' (offset %llu, expected %llu)",
                           path, spans[i].name, (unsigned long long)spans[i].begin, (unsigned long long)cursor);
            goto done;
        }
        cursor = spans[i].end;
    }
    if (cursor != data_len) {
        st = RRC_ERROR(RRC_ERR_PARSE, "%s: data block not fully covered by tensors (%llu of %llu bytes)", path,
                       (unsigned long long)cursor, (unsigned long long)data_len);
        goto done;
    }
done:
    free(spans);
    return st;
}

static rrc_status finalize(rrc_st_set *set) {
    qsort(set->tensors, set->n_tensors, sizeof(rrc_st_tensor), cmp_tensor_name);
    for (size_t i = 1; i < set->n_tensors; i++)
        if (!strcmp(set->tensors[i - 1].name, set->tensors[i].name))
            return RRC_ERROR(RRC_ERR_PARSE, "tensor '%s' appears in more than one shard", set->tensors[i].name);
    return RRC_OK;
}

rrc_status rrc_st_open_file(const char *path, rrc_st_set **out) {
    RRC_REQUIRE(path && out, RRC_ERR_INVALID_ARG, "null argument");
    *out = NULL;
    rrc_st_set *set = (rrc_st_set *)calloc(1, sizeof *set);
    if (!set) return RRC_ERROR(RRC_ERR_OOM, "safetensors set");
    rrc_status st = load_file(set, path);
    if (st == RRC_OK) st = finalize(set);
    if (st != RRC_OK) {
        rrc_st_close(set);
        return st;
    }
    *out = set;
    return RRC_OK;
}

static int ends_with(const char *s, const char *suf) {
    size_t a = strlen(s), b = strlen(suf);
    return a >= b && !strcmp(s + a - b, suf);
}

static int valid_shard_name(const char *s) {
    if (!s[0] || !strcmp(s, ".") || !strcmp(s, "..")) return 0;
    if (strchr(s, '/') || strchr(s, '\\')) return 0;
    return 1;
}

static char *join_path(const char *dir, const char *name) {
    size_t a = strlen(dir), b = strlen(name);
    char *p = (char *)malloc(a + b + 2);
    if (!p) return NULL;
    memcpy(p, dir, a);
    p[a] = '/';
    memcpy(p + a + 1, name, b + 1);
    return p;
}

typedef struct shard_ref {
    const char *name;
    size_t index;
} shard_ref;
static int shard_ref_cmp(const void *pa, const void *pb) {
    const shard_ref *a = (const shard_ref *)pa, *b = (const shard_ref *)pb;
    const int c = strcmp(a->name, b->name);
    return c ? c : (a->index < b->index ? -1 : a->index > b->index);
}

static rrc_status open_index(const char *index_path, const char *dir, rrc_st_set **out) {
    rrc_json_doc *doc = NULL;
    RRC_CHECK(rrc_json_parse_file(index_path, 0, &doc));
    rrc_st_set *set = NULL;
    const char **shards = NULL;
    size_t n_shards = 0;
    rrc_status st = RRC_OK;
    const rrc_json *root = rrc_json_root(doc);
    const rrc_json *wm = rrc_json_get(root, "weight_map");
    if (root->type != RRC_JSON_OBJECT || !wm || wm->type != RRC_JSON_OBJECT) {
        st = RRC_ERROR(RRC_ERR_PARSE, "%s: missing \"weight_map\" object", index_path);
        goto done;
    }
    size_t n = rrc_json_obj_count(wm);
    shards = (const char **)calloc(n + 1, sizeof(char *));
    if (!shards) {
        st = RRC_ERROR(RRC_ERR_OOM, "shard list");
        goto done;
    }
    for (size_t i = 0; i < n; i++) {
        const char *fn = rrc_json_str(rrc_json_obj_val(wm, i));
        if (!fn || !valid_shard_name(fn)) {
            st = RRC_ERROR(RRC_ERR_PARSE, "%s: weight_map['%s'] is not a plain shard file name", index_path,
                           rrc_json_obj_key(wm, i));
            goto done;
        }
    }
    /* distinct shard names in order of first appearance: sort (name, index), O(n log n) */
    {
        shard_ref *r = (shard_ref *)malloc((n ? n : 1) * sizeof(shard_ref));
        unsigned char *first = (unsigned char *)calloc(n ? n : 1, 1);
        if (!r || !first) {
            free(r);
            free(first);
            st = RRC_ERROR(RRC_ERR_OOM, "shard list");
            goto done;
        }
        for (size_t i = 0; i < n; i++) {
            r[i].name = rrc_json_str(rrc_json_obj_val(wm, i));
            r[i].index = i;
        }
        qsort(r, n, sizeof(shard_ref), shard_ref_cmp);
        for (size_t i = 0; i < n; i++)
            if (i == 0 || strcmp(r[i].name, r[i - 1].name)) first[r[i].index] = 1;
        for (size_t i = 0; i < n; i++)
            if (first[i]) shards[n_shards++] = rrc_json_str(rrc_json_obj_val(wm, i));
        free(r);
        free(first);
    }
    set = (rrc_st_set *)calloc(1, sizeof *set);
    if (!set) {
        st = RRC_ERROR(RRC_ERR_OOM, "safetensors set");
        goto done;
    }
    for (size_t k = 0; k < n_shards; k++) {
        char *p = join_path(dir, shards[k]);
        if (!p) {
            st = RRC_ERROR(RRC_ERR_OOM, "path");
            goto done;
        }
        st = load_file(set, p);
        free(p);
        if (st != RRC_OK) goto done;
    }
    if ((st = finalize(set)) != RRC_OK) goto done;
    /* every weight_map entry is in its shard; every tensor is listed for its shard */
    for (size_t i = 0; i < n; i++) {
        const char *tn = rrc_json_obj_key(wm, i);
        const char *fn = rrc_json_str(rrc_json_obj_val(wm, i));
        const rrc_st_tensor *t = rrc_st_find(set, tn);
        if (!t) {
            st = RRC_ERROR(RRC_ERR_NOT_FOUND, "%s: tensor '%s' listed for %s is missing", index_path, tn, fn);
            goto done;
        }
        if (strcmp(shards[t->shard], fn)) {
            st = RRC_ERROR(RRC_ERR_PARSE, "%s: tensor '%s' is listed for %s but stored in %s", index_path, tn, fn,
                           shards[t->shard]);
            goto done;
        }
    }
    if (set->n_tensors != n) {
        for (size_t i = 0; i < set->n_tensors; i++)
            if (!rrc_json_get(wm, set->tensors[i].name)) {
                st = RRC_ERROR(RRC_ERR_PARSE, "%s: tensor '%s' in %s is not listed in weight_map", index_path,
                               set->tensors[i].name, shards[set->tensors[i].shard]);
                goto done;
            }
        st = RRC_ERROR(RRC_ERR_INTERNAL, "%s: tensor count mismatch", index_path);
        goto done;
    }
done:
    free(shards);
    rrc_json_doc_free(doc); /* tensor names point into shard headers, not into the index */
    if (st != RRC_OK) {
        rrc_st_close(set);
        return st;
    }
    *out = set;
    return RRC_OK;
}

rrc_status rrc_st_open_checkpoint(const char *path, rrc_st_set **out) {
    RRC_REQUIRE(path && out, RRC_ERR_INVALID_ARG, "null argument");
    *out = NULL;
    struct stat stt;
    if (stat(path, &stt) != 0) return RRC_ERROR(RRC_ERR_NOT_FOUND, "%s: %s", path, strerror(errno));
    if (S_ISDIR(stt.st_mode)) {
        char *idx = join_path(path, "model.safetensors.index.json");
        char *single = join_path(path, "model.safetensors");
        rrc_status st;
        if (!idx || !single) st = RRC_ERROR(RRC_ERR_OOM, "path");
        else if (stat(idx, &stt) == 0) st = open_index(idx, path, out);
        else if (stat(single, &stt) == 0) st = rrc_st_open_file(single, out);
        else st = RRC_ERROR(RRC_ERR_NOT_FOUND, "%s: neither model.safetensors.index.json nor model.safetensors", path);
        free(idx);
        free(single);
        return st;
    }
    if (ends_with(path, ".index.json")) {
        char *dir = strdup(path);
        if (!dir) return RRC_ERROR(RRC_ERR_OOM, "path");
        char *slash = strrchr(dir, '/');
        if (slash) *slash = '\0';
        else strcpy(dir, ".");
        rrc_status st = open_index(path, dir, out);
        free(dir);
        return st;
    }
    return rrc_st_open_file(path, out);
}

size_t rrc_st_count(const rrc_st_set *set) { return set ? set->n_tensors : 0; }
const rrc_st_tensor *rrc_st_at(const rrc_st_set *set, size_t i) {
    return (set && i < set->n_tensors) ? &set->tensors[i] : NULL;
}
const rrc_st_tensor *rrc_st_find(const rrc_st_set *set, const char *name) {
    if (!set || !name) return NULL;
    size_t lo = 0, hi = set->n_tensors;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        int c = strcmp(set->tensors[mid].name, name);
        if (c == 0) return &set->tensors[mid];
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    return NULL;
}
size_t rrc_st_num_shards(const rrc_st_set *set) { return set ? set->n_files : 0; }
const char *rrc_st_metadata(const rrc_st_set *set, size_t shard, const char *key) {
    if (!set || shard >= set->n_files) return NULL;
    return rrc_json_str(rrc_json_get(set->files[shard].metadata, key));
}

/* ------------------------------------------------------------------------------------------ */
/* Conversion                                                                                   */
/* ------------------------------------------------------------------------------------------ */

float rrc_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else { /* subnormal: normalise */
            int e = -1;
            do {
                e++;
                man <<= 1;
            } while (!(man & 0x400u));
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((man & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    return rrc_bits_f32(bits);
}

static rrc_status check_n(const rrc_st_tensor *t, int64_t n) {
    RRC_REQUIRE(t, RRC_ERR_INVALID_ARG, "tensor is NULL");
    RRC_REQUIRE(n == t->numel, RRC_ERR_SHAPE, "tensor '%s' has %lld elements, buffer holds %lld", t->name,
                (long long)t->numel, (long long)n);
    return RRC_OK;
}

rrc_status rrc_st_to_f32(const rrc_st_tensor *t, float *dst, int64_t n) {
    RRC_CHECK(check_n(t, n));
    const unsigned char *s = (const unsigned char *)t->data;
    switch (t->dtype) {
    case RRC_ST_F32: memcpy(dst, s, (size_t)n * 4); return RRC_OK;
    case RRC_ST_BF16:
        for (int64_t i = 0; i < n; i++) {
            uint16_t h;
            memcpy(&h, s + 2 * i, 2);
            dst[i] = rrc_bf16_to_f32(h);
        }
        return RRC_OK;
    case RRC_ST_F16:
        for (int64_t i = 0; i < n; i++) {
            uint16_t h;
            memcpy(&h, s + 2 * i, 2);
            dst[i] = rrc_f16_to_f32(h);
        }
        return RRC_OK;
    default: return RRC_ERROR(RRC_ERR_DTYPE, "tensor '%s': %s is not a float dtype", t->name, rrc_st_dtype_name(t->dtype));
    }
}

rrc_status rrc_st_to_bf16(const rrc_st_tensor *t, rrc_bf16 *dst, int64_t n) {
    RRC_CHECK(check_n(t, n));
    const unsigned char *s = (const unsigned char *)t->data;
    switch (t->dtype) {
    case RRC_ST_BF16: memcpy(dst, s, (size_t)n * 2); return RRC_OK;
    case RRC_ST_F32:
        for (int64_t i = 0; i < n; i++) {
            float f;
            memcpy(&f, s + 4 * i, 4);
            dst[i] = rrc_f32_to_bf16(f);
        }
        return RRC_OK;
    case RRC_ST_F16:
        for (int64_t i = 0; i < n; i++) {
            uint16_t h;
            memcpy(&h, s + 2 * i, 2);
            dst[i] = rrc_f32_to_bf16(rrc_f16_to_f32(h));
        }
        return RRC_OK;
    default: return RRC_ERROR(RRC_ERR_DTYPE, "tensor '%s': %s is not a float dtype", t->name, rrc_st_dtype_name(t->dtype));
    }
}

rrc_status rrc_st_to_i32(const rrc_st_tensor *t, int32_t *dst, int64_t n) {
    RRC_CHECK(check_n(t, n));
    const unsigned char *s = (const unsigned char *)t->data;
    switch (t->dtype) {
    case RRC_ST_I32: memcpy(dst, s, (size_t)n * 4); return RRC_OK;
    case RRC_ST_I64:
        for (int64_t i = 0; i < n; i++) {
            int64_t v;
            memcpy(&v, s + 8 * i, 8);
            if (v < INT32_MIN || v > INT32_MAX)
                return RRC_ERROR(RRC_ERR_RANGE, "tensor '%s'[%lld] = %lld does not fit int32", t->name, (long long)i,
                                 (long long)v);
            dst[i] = (int32_t)v;
        }
        return RRC_OK;
    default: return RRC_ERROR(RRC_ERR_DTYPE, "tensor '%s': %s is not an integer dtype", t->name, rrc_st_dtype_name(t->dtype));
    }
}

/* ------------------------------------------------------------------------------------------ */
/* Writer                                                                                       */
/* ------------------------------------------------------------------------------------------ */

typedef struct w_tensor {
    char *name;
    rrc_dtype dtype;
    int ndim;
    int64_t shape[RRC_ST_MAX_DIMS];
    void *data;
    size_t nbytes;
} w_tensor;

struct rrc_st_writer {
    w_tensor *t;
    size_t n, cap;
    char **mkeys, **mvals;
    size_t nm;
};

rrc_status rrc_st_writer_create(rrc_st_writer **out) {
    RRC_REQUIRE(out, RRC_ERR_INVALID_ARG, "out is NULL");
    *out = (rrc_st_writer *)calloc(1, sizeof(rrc_st_writer));
    return *out ? RRC_OK : RRC_ERROR(RRC_ERR_OOM, "writer");
}

void rrc_st_writer_free(rrc_st_writer *w) {
    if (!w) return;
    for (size_t i = 0; i < w->n; i++) {
        free(w->t[i].name);
        free(w->t[i].data);
    }
    for (size_t i = 0; i < w->nm; i++) {
        free(w->mkeys[i]);
        free(w->mvals[i]);
    }
    free(w->t);
    free(w->mkeys);
    free(w->mvals);
    free(w);
}

rrc_status rrc_st_writer_add(rrc_st_writer *w, const char *name, rrc_dtype dtype, int ndim, const int64_t *shape,
                             const void *data) {
    RRC_REQUIRE(w && name, RRC_ERR_INVALID_ARG, "null argument");
    RRC_REQUIRE(name[0] && strcmp(name, "__metadata__"), RRC_ERR_INVALID_ARG, "invalid tensor name '%s'", name);
    RRC_REQUIRE(dtype == RRC_F32 || dtype == RRC_BF16 || dtype == RRC_I32, RRC_ERR_DTYPE,
                "tensor '%s': writer supports f32, bf16, i32", name);
    RRC_REQUIRE(ndim >= 0 && ndim <= RRC_ST_MAX_DIMS && (ndim == 0 || shape), RRC_ERR_SHAPE,
                "tensor '%s': bad ndim %d", name, ndim);
    for (size_t i = 0; i < w->n; i++)
        RRC_REQUIRE(strcmp(w->t[i].name, name), RRC_ERR_INVALID_ARG, "duplicate tensor name '%s'", name);
    int64_t numel = 1;
    for (int d = 0; d < ndim; d++) {
        RRC_REQUIRE(shape[d] >= 0, RRC_ERR_SHAPE, "tensor '%s': negative dim", name);
        RRC_REQUIRE(shape[d] == 0 || numel <= (INT64_MAX / 8) / shape[d], RRC_ERR_SHAPE, "tensor '%s': too large", name);
        numel *= shape[d];
    }
    size_t nbytes = (size_t)numel * rrc_dtype_size(dtype);
    RRC_REQUIRE(data || nbytes == 0, RRC_ERR_INVALID_ARG, "tensor '%s': data is NULL", name);
    if (w->n == w->cap) {
        size_t cap = w->cap ? 2 * w->cap : 16;
        w_tensor *nt = (w_tensor *)realloc(w->t, cap * sizeof(w_tensor));
        if (!nt) return RRC_ERROR(RRC_ERR_OOM, "writer");
        w->t = nt;
        w->cap = cap;
    }
    w_tensor *t = &w->t[w->n];
    memset(t, 0, sizeof *t);
    t->name = strdup(name);
    t->data = malloc(nbytes ? nbytes : 1);
    if (!t->name || !t->data) {
        free(t->name);
        free(t->data);
        return RRC_ERROR(RRC_ERR_OOM, "writer tensor '%s'", name);
    }
    if (nbytes) memcpy(t->data, data, nbytes);
    t->dtype = dtype;
    t->ndim = ndim;
    for (int d = 0; d < ndim; d++) t->shape[d] = shape[d];
    t->nbytes = nbytes;
    w->n++;
    return RRC_OK;
}

rrc_status rrc_st_writer_set_metadata(rrc_st_writer *w, const char *key, const char *value) {
    RRC_REQUIRE(w && key && value, RRC_ERR_INVALID_ARG, "null argument");
    for (size_t i = 0; i < w->nm; i++)
        if (!strcmp(w->mkeys[i], key)) {
            char *v = strdup(value);
            if (!v) return RRC_ERROR(RRC_ERR_OOM, "metadata");
            free(w->mvals[i]);
            w->mvals[i] = v;
            return RRC_OK;
        }
    char **nk = (char **)realloc(w->mkeys, (w->nm + 1) * sizeof(char *));
    if (!nk) return RRC_ERROR(RRC_ERR_OOM, "metadata");
    w->mkeys = nk;
    char **nv = (char **)realloc(w->mvals, (w->nm + 1) * sizeof(char *));
    if (!nv) return RRC_ERROR(RRC_ERR_OOM, "metadata");
    w->mvals = nv;
    w->mkeys[w->nm] = strdup(key);
    w->mvals[w->nm] = strdup(value);
    if (!w->mkeys[w->nm] || !w->mvals[w->nm]) {
        free(w->mkeys[w->nm]);
        free(w->mvals[w->nm]);
        return RRC_ERROR(RRC_ERR_OOM, "metadata");
    }
    w->nm++;
    return RRC_OK;
}

static int cmp_wt(const w_tensor *x, const w_tensor *y) {
    size_t sx = rrc_dtype_size(x->dtype), sy = rrc_dtype_size(y->dtype);
    if (sx != sy) return sx > sy ? -1 : 1;
    return strcmp(x->name, y->name);
}

rrc_status rrc_st_writer_write(rrc_st_writer *w, const char *path) {
    RRC_REQUIRE(w && path, RRC_ERR_INVALID_ARG, "null argument");
    size_t *order = (size_t *)malloc((w->n + 1) * sizeof(size_t));
    if (!order) return RRC_ERROR(RRC_ERR_OOM, "writer");
    for (size_t i = 0; i < w->n; i++) order[i] = i;
    /* insertion sort: element size descending, then name (no global state, thread-safe) */
    for (size_t i = 1; i < w->n; i++) {
        size_t v = order[i], j = i;
        while (j > 0 && cmp_wt(&w->t[order[j - 1]], &w->t[v]) > 0) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = v;
    }

    rrc_jw jw;
    rrc_jw_init(&jw, 0);
    rrc_jw_obj_begin(&jw);
    if (w->nm) {
        rrc_jw_key(&jw, "__metadata__");
        rrc_jw_obj_begin(&jw);
        for (size_t i = 0; i < w->nm; i++) {
            rrc_jw_key(&jw, w->mkeys[i]);
            rrc_jw_str(&jw, w->mvals[i]);
        }
        rrc_jw_obj_end(&jw);
    }
    uint64_t off = 0;
    for (size_t k = 0; k < w->n; k++) {
        const w_tensor *t = &w->t[order[k]];
        rrc_jw_key(&jw, t->name);
        rrc_jw_obj_begin(&jw);
        rrc_jw_key(&jw, "dtype");
        rrc_jw_str(&jw, t->dtype == RRC_F32 ? "F32" : t->dtype == RRC_BF16 ? "BF16" : "I32");
        rrc_jw_key(&jw, "shape");
        rrc_jw_arr_begin(&jw);
        for (int d = 0; d < t->ndim; d++) rrc_jw_int(&jw, t->shape[d]);
        rrc_jw_arr_end(&jw);
        rrc_jw_key(&jw, "data_offsets");
        rrc_jw_arr_begin(&jw);
        rrc_jw_int(&jw, (int64_t)off);
        rrc_jw_int(&jw, (int64_t)(off + t->nbytes));
        rrc_jw_arr_end(&jw);
        rrc_jw_obj_end(&jw);
        off += t->nbytes;
    }
    rrc_jw_obj_end(&jw);
    char *hdr = NULL;
    size_t hlen = 0;
    rrc_status st = rrc_jw_finish(&jw, &hdr, &hlen);
    if (st != RRC_OK) {
        free(order);
        return st;
    }
    size_t padded = (hlen + 7u) & ~(size_t)7u;
    size_t total = 8 + padded + (size_t)off;
    unsigned char *buf = (unsigned char *)malloc(total);
    if (!buf) {
        free(hdr);
        free(order);
        return RRC_ERROR(RRC_ERR_OOM, "writer buffer (%zu bytes)", total);
    }
    uint64_t hl = padded;
    for (int i = 0; i < 8; i++) buf[i] = (unsigned char)(hl >> (8 * i));
    memcpy(buf + 8, hdr, hlen);
    memset(buf + 8 + hlen, ' ', padded - hlen);
    size_t pos = 8 + padded;
    for (size_t k = 0; k < w->n; k++) {
        const w_tensor *t = &w->t[order[k]];
        if (t->nbytes) memcpy(buf + pos, t->data, t->nbytes);
        pos += t->nbytes;
    }
    st = rrc_write_file_atomic(path, buf, total);
    free(buf);
    free(hdr);
    free(order);
    return st;
}
