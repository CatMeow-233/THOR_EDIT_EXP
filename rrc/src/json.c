/* rrc/src/json.c — strict JSON parser (DOM in an arena) and writer. See json.h for the rules. */
#include "rrc/json.h"
#include "rrc/fileio.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* ------------------------------------------------------------------------------------------ */
/* Arena                                                                                        */
/* ------------------------------------------------------------------------------------------ */

typedef struct arena_block {
    struct arena_block *next;
    size_t used, cap;
    max_align_t data[]; /* flexible, aligned */
} arena_block;

struct rrc_json_doc {
    arena_block *blocks;
    rrc_json *root;
};

static void *arena_alloc(rrc_json_doc *doc, size_t n) {
    n = (n + 15u) & ~(size_t)15u;
    arena_block *b = doc->blocks;
    if (!b || b->cap - b->used < n) {
        size_t cap = n > 65536 ? n : 65536;
        arena_block *nb = (arena_block *)malloc(sizeof(arena_block) + cap);
        if (!nb) return NULL;
        nb->next = doc->blocks;
        nb->used = 0;
        nb->cap = cap;
        doc->blocks = nb;
        b = nb;
    }
    void *p = (char *)b->data + b->used;
    b->used += n;
    return p;
}

void rrc_json_doc_free(rrc_json_doc *doc) {
    if (!doc) return;
    arena_block *b = doc->blocks;
    while (b) {
        arena_block *n = b->next;
        free(b);
        b = n;
    }
    free(doc);
}

const rrc_json *rrc_json_root(const rrc_json_doc *doc) { return doc ? doc->root : NULL; }

/* ------------------------------------------------------------------------------------------ */
/* Parser                                                                                       */
/* ------------------------------------------------------------------------------------------ */

typedef struct {
    const unsigned char *p, *end, *begin;
    rrc_json_doc *doc;
    int depth;
    /* scratch buffer for string decoding */
    char *sbuf;
    size_t scap;
} parser;

static void pos_of(const parser *ps, const unsigned char *at, int *line, int *col) {
    int l = 1, c = 1;
    for (const unsigned char *q = ps->begin; q < at && q < ps->end; q++) {
        if (*q == '\n') {
            l++;
            c = 1;
        } else {
            c++;
        }
    }
    *line = l;
    *col = c;
}

#define PERR(ps, at, ...) parse_error((ps), (at), __LINE__, __VA_ARGS__)
static rrc_status parse_error(const parser *ps, const unsigned char *at, int srcline, const char *fmt, ...)
    RRC_PRINTF(4, 5);
static rrc_status parse_error(const parser *ps, const unsigned char *at, int srcline, const char *fmt, ...) {
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    int line, col;
    pos_of(ps, at, &line, &col);
    return rrc_set_error(RRC_ERR_PARSE, __FILE__, srcline, "JSON %d:%d: %s", line, col, msg);
}

static void skip_ws(parser *ps) {
    while (ps->p < ps->end && (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r')) ps->p++;
}

static rrc_json *new_node(parser *ps, rrc_json_type t) {
    rrc_json *n = (rrc_json *)arena_alloc(ps->doc, sizeof(rrc_json));
    if (n) {
        memset(n, 0, sizeof *n);
        n->type = t;
    }
    return n;
}

static int sbuf_push(parser *ps, size_t *len, const char *bytes, size_t n) {
    if (*len + n + 1 > ps->scap) {
        size_t cap = ps->scap ? ps->scap * 2 : 256;
        while (cap < *len + n + 1) cap *= 2;
        char *nb = (char *)realloc(ps->sbuf, cap);
        if (!nb) return 0;
        ps->sbuf = nb;
        ps->scap = cap;
    }
    memcpy(ps->sbuf + *len, bytes, n);
    *len += n;
    return 1;
}

/* Length of a valid UTF-8 sequence starting at p (1..4), or 0 if invalid. */
static int utf8_seq_len(const unsigned char *p, const unsigned char *end) {
    unsigned c = p[0];
    if (c < 0x80) return 1;
    size_t avail = (size_t)(end - p);
    if (c >= 0xC2 && c <= 0xDF) {
        if (avail < 2 || (p[1] & 0xC0) != 0x80) return 0;
        return 2;
    }
    if (c >= 0xE0 && c <= 0xEF) {
        if (avail < 3 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80) return 0;
        if (c == 0xE0 && p[1] < 0xA0) return 0; /* overlong */
        if (c == 0xED && p[1] > 0x9F) return 0; /* UTF-16 surrogate */
        return 3;
    }
    if (c >= 0xF0 && c <= 0xF4) {
        if (avail < 4 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80 || (p[3] & 0xC0) != 0x80) return 0;
        if (c == 0xF0 && p[1] < 0x90) return 0; /* overlong */
        if (c == 0xF4 && p[1] > 0x8F) return 0; /* > U+10FFFF */
        return 4;
    }
    return 0;
}

static int hex4(const unsigned char *p, unsigned *out) {
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        unsigned c = p[i], d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else return 0;
        v = v * 16 + d;
    }
    *out = v;
    return 1;
}

static size_t utf8_encode(unsigned cp, char out[4]) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* Parses a string starting at the opening quote; returns arena copy. */
static rrc_status parse_string(parser *ps, const char **out, size_t *out_len) {
    const unsigned char *start = ps->p;
    ps->p++; /* opening quote */
    size_t len = 0;
    for (;;) {
        if (ps->p >= ps->end) return PERR(ps, start, "unterminated string");
        unsigned c = *ps->p;
        if (c == '"') {
            ps->p++;
            break;
        }
        if (c < 0x20) return PERR(ps, ps->p, "raw control character 0x%02x in string", c);
        if (c == '\\') {
            const unsigned char *esc = ps->p;
            if (ps->p + 1 >= ps->end) return PERR(ps, esc, "unterminated escape");
            unsigned e = ps->p[1];
            char ch;
            switch (e) {
            case '"': ch = '"'; break;
            case '\\': ch = '\\'; break;
            case '/': ch = '/'; break;
            case 'b': ch = '\b'; break;
            case 'f': ch = '\f'; break;
            case 'n': ch = '\n'; break;
            case 'r': ch = '\r'; break;
            case 't': ch = '\t'; break;
            case 'u': {
                unsigned cp;
                if (ps->end - ps->p < 6 || !hex4(ps->p + 2, &cp)) return PERR(ps, esc, "bad \\u escape");
                ps->p += 6;
                if (cp >= 0xDC00 && cp <= 0xDFFF) return PERR(ps, esc, "lone low surrogate \\u%04X", cp);
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    unsigned lo;
                    if (ps->end - ps->p < 6 || ps->p[0] != '\\' || ps->p[1] != 'u' || !hex4(ps->p + 2, &lo) ||
                        lo < 0xDC00 || lo > 0xDFFF)
                        return PERR(ps, esc, "high surrogate \\u%04X not followed by a low surrogate", cp);
                    ps->p += 6;
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                }
                char enc[4];
                size_t n = utf8_encode(cp, enc);
                if (!sbuf_push(ps, &len, enc, n)) return RRC_ERROR(RRC_ERR_OOM, "JSON string buffer");
                continue;
            }
            default: return PERR(ps, esc, "invalid escape \\%c", (e >= 0x20 && e < 0x7F) ? (char)e : '?');
            }
            if (!sbuf_push(ps, &len, &ch, 1)) return RRC_ERROR(RRC_ERR_OOM, "JSON string buffer");
            ps->p += 2;
            continue;
        }
        int n = utf8_seq_len(ps->p, ps->end);
        if (n == 0) return PERR(ps, ps->p, "invalid UTF-8 byte 0x%02x in string", c);
        if (!sbuf_push(ps, &len, (const char *)ps->p, (size_t)n)) return RRC_ERROR(RRC_ERR_OOM, "JSON string");
        ps->p += n;
    }
    char *copy = (char *)arena_alloc(ps->doc, len + 1);
    if (!copy) return RRC_ERROR(RRC_ERR_OOM, "JSON arena");
    if (len) memcpy(copy, ps->sbuf, len);
    copy[len] = '\0';
    *out = copy;
    *out_len = len;
    return RRC_OK;
}

static int is_digit(unsigned c) { return c >= '0' && c <= '9'; }

static rrc_status parse_number(parser *ps, rrc_json **out) {
    const unsigned char *s = ps->p;
    const unsigned char *q = s;
    int neg = 0, is_int = 1;
    if (q < ps->end && *q == '-') {
        neg = 1;
        q++;
    }
    if (q >= ps->end || !is_digit(*q)) return PERR(ps, s, "invalid number");
    if (*q == '0') {
        q++;
        if (q < ps->end && is_digit(*q)) return PERR(ps, s, "leading zero in number");
    } else {
        while (q < ps->end && is_digit(*q)) q++;
    }
    if (q < ps->end && *q == '.') {
        is_int = 0;
        q++;
        if (q >= ps->end || !is_digit(*q)) return PERR(ps, s, "digit expected after '.'");
        while (q < ps->end && is_digit(*q)) q++;
    }
    if (q < ps->end && (*q == 'e' || *q == 'E')) {
        is_int = 0;
        q++;
        if (q < ps->end && (*q == '+' || *q == '-')) q++;
        if (q >= ps->end || !is_digit(*q)) return PERR(ps, s, "digit expected in exponent");
        while (q < ps->end && is_digit(*q)) q++;
    }
    size_t n = (size_t)(q - s);
    rrc_json *node = NULL;
    if (is_int) {
        /* accumulate as negative to cover INT64_MIN */
        int64_t acc = 0;
        int overflow = 0;
        for (const unsigned char *d = s + neg; d < q; d++) {
            int digit = *d - '0';
            if (acc < (INT64_MIN + digit) / 10) {
                overflow = 1;
                break;
            }
            acc = acc * 10 - digit;
        }
        if (!overflow && !neg && acc == INT64_MIN) overflow = 1;
        if (!overflow) {
            node = new_node(ps, RRC_JSON_INT);
            if (!node) return RRC_ERROR(RRC_ERR_OOM, "JSON arena");
            node->u.i = neg ? acc : -acc;
            ps->p = q;
            *out = node;
            return RRC_OK;
        }
        /* too large for int64: fall through to double */
    }
    char local[64];
    char *tmp = local;
    if (n + 1 > sizeof local) {
        tmp = (char *)malloc(n + 1);
        if (!tmp) return RRC_ERROR(RRC_ERR_OOM, "JSON number");
    }
    memcpy(tmp, s, n);
    tmp[n] = '\0';
    errno = 0;
    char *endp = NULL;
    double d = strtod(tmp, &endp);
    int consumed = (endp == tmp + n);
    if (tmp != local) free(tmp);
    if (!consumed) return PERR(ps, s, "invalid number");
    if (!isfinite(d)) return PERR(ps, s, "number out of double range");
    node = new_node(ps, RRC_JSON_DOUBLE);
    if (!node) return RRC_ERROR(RRC_ERR_OOM, "JSON arena");
    node->u.d = d;
    ps->p = q;
    *out = node;
    return RRC_OK;
}

static rrc_status parse_value(parser *ps, rrc_json **out);

/* Dynamic vector used while an array/object is open; copied into the arena when closed. */
typedef struct {
    void **items;
    size_t n, cap;
} pvec;

static int pvec_push(pvec *v, void *x) {
    if (v->n == v->cap) {
        size_t cap = v->cap ? v->cap * 2 : 8;
        void **ni = (void **)realloc(v->items, cap * sizeof(void *));
        if (!ni) return 0;
        v->items = ni;
        v->cap = cap;
    }
    v->items[v->n++] = x;
    return 1;
}

static rrc_status parse_array(parser *ps, rrc_json **out) {
    const unsigned char *start = ps->p;
    ps->p++;
    pvec v = {0};
    rrc_status st = RRC_OK;
    skip_ws(ps);
    if (ps->p < ps->end && *ps->p == ']') {
        ps->p++;
    } else {
        for (;;) {
            rrc_json *item = NULL;
            skip_ws(ps);
            if ((st = parse_value(ps, &item)) != RRC_OK) goto done;
            if (!pvec_push(&v, item)) {
                st = RRC_ERROR(RRC_ERR_OOM, "JSON array");
                goto done;
            }
            skip_ws(ps);
            if (ps->p >= ps->end) {
                st = PERR(ps, start, "unterminated array");
                goto done;
            }
            if (*ps->p == ',') {
                ps->p++;
                skip_ws(ps);
                if (ps->p < ps->end && *ps->p == ']') {
                    st = PERR(ps, ps->p, "trailing comma in array");
                    goto done;
                }
                continue;
            }
            if (*ps->p == ']') {
                ps->p++;
                break;
            }
            st = PERR(ps, ps->p, "expected ',' or ']' in array");
            goto done;
        }
    }
    {
        rrc_json *node = new_node(ps, RRC_JSON_ARRAY);
        if (!node) {
            st = RRC_ERROR(RRC_ERR_OOM, "JSON arena");
            goto done;
        }
        if (v.n) {
            node->u.a.items = (rrc_json **)arena_alloc(ps->doc, v.n * sizeof(rrc_json *));
            if (!node->u.a.items) {
                st = RRC_ERROR(RRC_ERR_OOM, "JSON arena");
                goto done;
            }
            memcpy(node->u.a.items, v.items, v.n * sizeof(rrc_json *));
        }
        node->u.a.count = v.n;
        *out = node;
    }
done:
    free(v.items);
    return st;
}

/* Duplicate keys: objects with up to DUP_LINEAR_MAX keys are checked as each key is read (linear
 * scan); a larger object (a safetensors header holds one key per tensor) is checked once at its end
 * by sorting (key, position), O(n log n) instead of O(n^2). Both report the earliest repeated
 * occurrence, at its offset. */
#define DUP_LINEAR_MAX 16
typedef struct key_ref {
    const char *key;
    size_t len;
    const unsigned char *at; /* position of the key in the input */
} key_ref;
static int key_ref_cmp(const void *pa, const void *pb) {
    const key_ref *a = (const key_ref *)pa, *b = (const key_ref *)pb;
    const size_t n = a->len < b->len ? a->len : b->len;
    const int c = memcmp(a->key, b->key, n);
    if (c) return c;
    if (a->len != b->len) return a->len < b->len ? -1 : 1;
    return a->at < b->at ? -1 : a->at > b->at;
}
static rrc_status check_dup_sorted(parser *ps, const pvec *keys, const pvec *lens, const pvec *ats) {
    key_ref *r = (key_ref *)malloc(keys->n * sizeof(key_ref));
    if (!r) return RRC_ERROR(RRC_ERR_OOM, "JSON object");
    for (size_t i = 0; i < keys->n; i++) {
        r[i].key = (const char *)keys->items[i];
        r[i].len = (size_t)(uintptr_t)lens->items[i];
        r[i].at = (const unsigned char *)ats->items[i];
    }
    qsort(r, keys->n, sizeof(key_ref), key_ref_cmp);
    const key_ref *first = NULL; /* the earliest second occurrence of any key */
    for (size_t i = 1; i < keys->n; i++)
        if (r[i].len == r[i - 1].len && !memcmp(r[i].key, r[i - 1].key, r[i].len) &&
            (i < 2 || r[i - 1].len != r[i - 2].len || memcmp(r[i - 1].key, r[i - 2].key, r[i].len)) &&
            (!first || r[i].at < first->at))
            first = &r[i];
    rrc_status st = RRC_OK;
    if (first) st = PERR(ps, first->at, "duplicate key \"%s\"", first->key);
    free(r);
    return st;
}

static rrc_status parse_object(parser *ps, rrc_json **out) {
    const unsigned char *start = ps->p;
    ps->p++;
    pvec keys = {0}, lens = {0}, vals = {0}, ats = {0};
    rrc_status st = RRC_OK;
    skip_ws(ps);
    if (ps->p < ps->end && *ps->p == '}') {
        ps->p++;
    } else {
        for (;;) {
            skip_ws(ps);
            if (ps->p >= ps->end || *ps->p != '"') {
                st = PERR(ps, ps->p, ps->p < ps->end && *ps->p == '}' ? "trailing comma in object"
                                                                       : "expected string key in object");
                goto done;
            }
            const unsigned char *key_at = ps->p;
            const char *key;
            size_t klen;
            if ((st = parse_string(ps, &key, &klen)) != RRC_OK) goto done;
            for (size_t i = 0; keys.n < DUP_LINEAR_MAX && i < keys.n; i++) {
                if ((size_t)(uintptr_t)lens.items[i] == klen && memcmp(keys.items[i], key, klen) == 0) {
                    st = PERR(ps, key_at, "duplicate key \"%s\"", key);
                    goto done;
                }
            }
            skip_ws(ps);
            if (ps->p >= ps->end || *ps->p != ':') {
                st = PERR(ps, ps->p, "expected ':' after object key");
                goto done;
            }
            ps->p++;
            skip_ws(ps);
            rrc_json *val = NULL;
            if ((st = parse_value(ps, &val)) != RRC_OK) goto done;
            if (!pvec_push(&keys, (void *)key) || !pvec_push(&lens, (void *)(uintptr_t)klen) ||
                !pvec_push(&vals, val) || !pvec_push(&ats, (void *)(uintptr_t)key_at)) {
                st = RRC_ERROR(RRC_ERR_OOM, "JSON object");
                goto done;
            }
            skip_ws(ps);
            if (ps->p >= ps->end) {
                st = PERR(ps, start, "unterminated object");
                goto done;
            }
            if (*ps->p == ',') {
                ps->p++;
                continue;
            }
            if (*ps->p == '}') {
                ps->p++;
                break;
            }
            st = PERR(ps, ps->p, "expected ',' or '}' in object");
            goto done;
        }
    }
    if (keys.n > DUP_LINEAR_MAX && (st = check_dup_sorted(ps, &keys, &lens, &ats)) != RRC_OK) goto done;
    {
        rrc_json *node = new_node(ps, RRC_JSON_OBJECT);
        if (!node) {
            st = RRC_ERROR(RRC_ERR_OOM, "JSON arena");
            goto done;
        }
        size_t n = keys.n;
        if (n) {
            node->u.o.keys = (const char **)arena_alloc(ps->doc, n * sizeof(char *));
            node->u.o.key_lens = (size_t *)arena_alloc(ps->doc, n * sizeof(size_t));
            node->u.o.vals = (rrc_json **)arena_alloc(ps->doc, n * sizeof(rrc_json *));
            if (!node->u.o.keys || !node->u.o.key_lens || !node->u.o.vals) {
                st = RRC_ERROR(RRC_ERR_OOM, "JSON arena");
                goto done;
            }
            for (size_t i = 0; i < n; i++) {
                node->u.o.keys[i] = (const char *)keys.items[i];
                node->u.o.key_lens[i] = (size_t)(uintptr_t)lens.items[i];
                node->u.o.vals[i] = (rrc_json *)vals.items[i];
            }
        }
        node->u.o.count = n;
        *out = node;
    }
done:
    free(keys.items);
    free(lens.items);
    free(vals.items);
    free(ats.items);
    return st;
}

static int match_lit(parser *ps, const char *lit) {
    size_t n = strlen(lit);
    if ((size_t)(ps->end - ps->p) < n || memcmp(ps->p, lit, n) != 0) return 0;
    ps->p += n;
    return 1;
}

static rrc_status parse_value(parser *ps, rrc_json **out) {
    if (ps->p >= ps->end) return PERR(ps, ps->p, "unexpected end of input, value expected");
    unsigned c = *ps->p;
    rrc_status st;
    switch (c) {
    case '{':
    case '[':
        if (ps->depth >= RRC_JSON_MAX_DEPTH) return PERR(ps, ps->p, "nesting deeper than %d", RRC_JSON_MAX_DEPTH);
        ps->depth++;
        st = c == '{' ? parse_object(ps, out) : parse_array(ps, out);
        ps->depth--;
        return st;
    case '"': {
        const char *s;
        size_t len;
        RRC_CHECK(parse_string(ps, &s, &len));
        rrc_json *node = new_node(ps, RRC_JSON_STRING);
        if (!node) return RRC_ERROR(RRC_ERR_OOM, "JSON arena");
        node->u.s.ptr = s;
        node->u.s.len = len;
        *out = node;
        return RRC_OK;
    }
    case 't':
    case 'f':
    case 'n': {
        const unsigned char *at = ps->p;
        rrc_json *node = NULL;
        if (match_lit(ps, "true")) {
            node = new_node(ps, RRC_JSON_BOOL);
            if (node) node->u.boolean = 1;
        } else if (match_lit(ps, "false")) {
            node = new_node(ps, RRC_JSON_BOOL);
        } else if (match_lit(ps, "null")) {
            node = new_node(ps, RRC_JSON_NULL);
        } else {
            return PERR(ps, at, "invalid literal");
        }
        if (!node) return RRC_ERROR(RRC_ERR_OOM, "JSON arena");
        /* a literal must not run into other identifier characters ("truex") */
        if (ps->p < ps->end && ((*ps->p >= 'a' && *ps->p <= 'z') || (*ps->p >= 'A' && *ps->p <= 'Z') ||
                                is_digit(*ps->p) || *ps->p == '_'))
            return PERR(ps, at, "invalid literal");
        *out = node;
        return RRC_OK;
    }
    default:
        if (c == '-' || is_digit(c)) return parse_number(ps, out);
        if (c >= 0x20 && c < 0x7F) return PERR(ps, ps->p, "unexpected character '%c'", (char)c);
        return PERR(ps, ps->p, "unexpected byte 0x%02x", c);
    }
}

rrc_status rrc_json_parse(const char *text, size_t len, rrc_json_doc **out) {
    RRC_REQUIRE(out != NULL, RRC_ERR_INVALID_ARG, "out is NULL");
    *out = NULL;
    RRC_REQUIRE(text != NULL || len == 0, RRC_ERR_INVALID_ARG, "text is NULL");
    rrc_json_doc *doc = (rrc_json_doc *)calloc(1, sizeof *doc);
    if (!doc) return RRC_ERROR(RRC_ERR_OOM, "JSON doc");
    parser ps;
    memset(&ps, 0, sizeof ps);
    ps.begin = ps.p = (const unsigned char *)(text ? text : "");
    ps.end = ps.p + len;
    ps.doc = doc;
    rrc_status st;
    if (len >= 3 && ps.p[0] == 0xEF && ps.p[1] == 0xBB && ps.p[2] == 0xBF) {
        st = PERR(&ps, ps.p, "byte order mark is not allowed");
        goto fail;
    }
    skip_ws(&ps);
    rrc_json *root = NULL;
    if ((st = parse_value(&ps, &root)) != RRC_OK) goto fail;
    skip_ws(&ps);
    if (ps.p != ps.end) {
        st = PERR(&ps, ps.p, "trailing characters after JSON value");
        goto fail;
    }
    free(ps.sbuf);
    doc->root = root;
    *out = doc;
    return RRC_OK;
fail:
    free(ps.sbuf);
    rrc_json_doc_free(doc);
    return st;
}

rrc_status rrc_json_parse_file(const char *path, size_t max_bytes, rrc_json_doc **out) {
    RRC_REQUIRE(path && out, RRC_ERR_INVALID_ARG, "null argument");
    *out = NULL;
    if (max_bytes == 0) max_bytes = (size_t)256 << 20;
    FILE *f = NULL;
    RRC_CHECK(rrc_fopen_regular(path, &f));
    struct stat stt;
    if (fstat(fileno(f), &stt) != 0) {
        rrc_status s = RRC_ERROR(RRC_ERR_IO, "fstat %s: %s", path, strerror(errno));
        fclose(f);
        return s;
    }
    if ((uint64_t)stt.st_size > (uint64_t)max_bytes) {
        fclose(f);
        return RRC_ERROR(RRC_ERR_PARSE, "%s is larger than %zu bytes", path, max_bytes);
    }
    size_t n = (size_t)stt.st_size;
    char *buf = (char *)malloc(n ? n : 1);
    if (!buf) {
        fclose(f);
        return RRC_ERROR(RRC_ERR_OOM, "reading %s", path);
    }
    size_t got = n ? fread(buf, 1, n, f) : 0;
    fclose(f);
    if (got != n) {
        free(buf);
        return RRC_ERROR(RRC_ERR_IO, "short read on %s", path);
    }
    rrc_status st = rrc_json_parse(buf, n, out);
    free(buf);
    if (st != RRC_OK) return rrc_error_context(st, __FILE__, __LINE__, "%s", path);
    return RRC_OK;
}

/* ------------------------------------------------------------------------------------------ */
/* Lookup                                                                                       */
/* ------------------------------------------------------------------------------------------ */

const char *rrc_json_type_name(rrc_json_type t) {
    switch (t) {
    case RRC_JSON_NULL: return "null";
    case RRC_JSON_BOOL: return "bool";
    case RRC_JSON_INT: return "int";
    case RRC_JSON_DOUBLE: return "double";
    case RRC_JSON_STRING: return "string";
    case RRC_JSON_ARRAY: return "array";
    case RRC_JSON_OBJECT: return "object";
    }
    return "?";
}

const rrc_json *rrc_json_get(const rrc_json *obj, const char *key) {
    if (!obj || obj->type != RRC_JSON_OBJECT || !key) return NULL;
    size_t klen = strlen(key);
    for (size_t i = 0; i < obj->u.o.count; i++)
        if (obj->u.o.key_lens[i] == klen && memcmp(obj->u.o.keys[i], key, klen) == 0) return obj->u.o.vals[i];
    return NULL;
}
size_t rrc_json_obj_count(const rrc_json *obj) {
    return (obj && obj->type == RRC_JSON_OBJECT) ? obj->u.o.count : 0;
}
const char *rrc_json_obj_key(const rrc_json *obj, size_t i) {
    return (obj && obj->type == RRC_JSON_OBJECT && i < obj->u.o.count) ? obj->u.o.keys[i] : NULL;
}
const rrc_json *rrc_json_obj_val(const rrc_json *obj, size_t i) {
    return (obj && obj->type == RRC_JSON_OBJECT && i < obj->u.o.count) ? obj->u.o.vals[i] : NULL;
}
size_t rrc_json_arr_count(const rrc_json *arr) { return (arr && arr->type == RRC_JSON_ARRAY) ? arr->u.a.count : 0; }
const rrc_json *rrc_json_arr_get(const rrc_json *arr, size_t i) {
    return (arr && arr->type == RRC_JSON_ARRAY && i < arr->u.a.count) ? arr->u.a.items[i] : NULL;
}
int rrc_json_is_number(const rrc_json *v) { return v && (v->type == RRC_JSON_INT || v->type == RRC_JSON_DOUBLE); }
const char *rrc_json_str(const rrc_json *v) { return (v && v->type == RRC_JSON_STRING) ? v->u.s.ptr : NULL; }

static const char *tname(const rrc_json *v) { return v ? rrc_json_type_name(v->type) : "missing"; }

rrc_status rrc_json_as_int(const rrc_json *v, const char *what, int64_t *out) {
    RRC_REQUIRE(v && v->type == RRC_JSON_INT, RRC_ERR_PARSE, "%s: expected an integer, got %s", what, tname(v));
    *out = v->u.i;
    return RRC_OK;
}
rrc_status rrc_json_as_number(const rrc_json *v, const char *what, double *out) {
    RRC_REQUIRE(rrc_json_is_number(v), RRC_ERR_PARSE, "%s: expected a number, got %s", what, tname(v));
    *out = v->type == RRC_JSON_INT ? (double)v->u.i : v->u.d;
    return RRC_OK;
}
rrc_status rrc_json_as_bool(const rrc_json *v, const char *what, int *out) {
    RRC_REQUIRE(v && v->type == RRC_JSON_BOOL, RRC_ERR_PARSE, "%s: expected true/false, got %s", what, tname(v));
    *out = v->u.boolean;
    return RRC_OK;
}
rrc_status rrc_json_as_string(const rrc_json *v, const char *what, const char **out) {
    RRC_REQUIRE(v && v->type == RRC_JSON_STRING, RRC_ERR_PARSE, "%s: expected a string, got %s", what, tname(v));
    *out = v->u.s.ptr;
    return RRC_OK;
}

/* ------------------------------------------------------------------------------------------ */
/* Writer                                                                                       */
/* ------------------------------------------------------------------------------------------ */

void rrc_jw_init(rrc_jw *w, int pretty) {
    memset(w, 0, sizeof *w);
    w->pretty = pretty;
}

void rrc_jw_free(rrc_jw *w) {
    free(w->buf);
    int pretty = w->pretty;
    memset(w, 0, sizeof *w);
    w->pretty = pretty;
}

static void jw_raw(rrc_jw *w, const char *s, size_t n) {
    if (w->err) return;
    if (w->len + n + 1 > w->cap) {
        size_t cap = w->cap ? w->cap * 2 : 256;
        while (cap < w->len + n + 1) cap *= 2;
        char *nb = (char *)realloc(w->buf, cap);
        if (!nb) {
            w->err = RRC_ERR_OOM;
            return;
        }
        w->buf = nb;
        w->cap = cap;
    }
    memcpy(w->buf + w->len, s, n);
    w->len += n;
    w->buf[w->len] = '\0';
}

static void jw_newline(rrc_jw *w) {
    if (!w->pretty) return;
    jw_raw(w, "\n", 1);
    for (int i = 0; i < w->depth; i++) jw_raw(w, "  ", 2);
}

/* Called before every value (scalar or container start). */
static void jw_before_value(rrc_jw *w) {
    if (w->err) return;
    if (w->depth == 0) {
        if (w->len != 0) w->err = RRC_ERR_INVALID_ARG; /* second top-level value */
        return;
    }
    if (w->kind[w->depth - 1] == 'o') {
        if (!w->expect_value) {
            w->err = RRC_ERR_INVALID_ARG; /* value without key */
            return;
        }
        w->expect_value = 0;
        return;
    }
    if (w->count[w->depth - 1]) jw_raw(w, ",", 1);
    w->count[w->depth - 1] = 1;
    jw_newline(w);
}

static void jw_open(rrc_jw *w, char kind) {
    jw_before_value(w);
    if (w->err) return;
    if (w->depth >= RRC_JSON_MAX_DEPTH) {
        w->err = RRC_ERR_INVALID_ARG;
        return;
    }
    jw_raw(w, kind == 'o' ? "{" : "[", 1);
    w->kind[w->depth] = (unsigned char)kind;
    w->count[w->depth] = 0;
    w->depth++;
}

static void jw_close(rrc_jw *w, char kind) {
    if (w->err) return;
    if (w->depth == 0 || w->kind[w->depth - 1] != (unsigned char)kind || w->expect_value) {
        w->err = RRC_ERR_INVALID_ARG;
        return;
    }
    int had = w->count[w->depth - 1];
    w->depth--;
    if (had) jw_newline(w);
    jw_raw(w, kind == 'o' ? "}" : "]", 1);
}

void rrc_jw_obj_begin(rrc_jw *w) { jw_open(w, 'o'); }
void rrc_jw_obj_end(rrc_jw *w) { jw_close(w, 'o'); }
void rrc_jw_arr_begin(rrc_jw *w) { jw_open(w, 'a'); }
void rrc_jw_arr_end(rrc_jw *w) { jw_close(w, 'a'); }

static void jw_string_body(rrc_jw *w, const char *s, size_t n) {
    jw_raw(w, "\"", 1);
    size_t run = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        const char *esc = NULL;
        char ubuf[8];
        switch (c) {
        case '"': esc = "\\\""; break;
        case '\\': esc = "\\\\"; break;
        case '\b': esc = "\\b"; break;
        case '\f': esc = "\\f"; break;
        case '\n': esc = "\\n"; break;
        case '\r': esc = "\\r"; break;
        case '\t': esc = "\\t"; break;
        default:
            if (c < 0x20) {
                snprintf(ubuf, sizeof ubuf, "\\u%04x", c);
                esc = ubuf;
            }
        }
        if (esc) {
            jw_raw(w, s + i - run, run);
            run = 0;
            jw_raw(w, esc, strlen(esc));
        } else {
            run++;
        }
    }
    jw_raw(w, s + n - run, run);
    jw_raw(w, "\"", 1);
}

void rrc_jw_key(rrc_jw *w, const char *key) {
    if (w->err) return;
    if (w->depth == 0 || w->kind[w->depth - 1] != 'o' || w->expect_value || !key) {
        w->err = RRC_ERR_INVALID_ARG;
        return;
    }
    if (w->count[w->depth - 1]) jw_raw(w, ",", 1);
    w->count[w->depth - 1] = 1;
    jw_newline(w);
    jw_string_body(w, key, strlen(key));
    jw_raw(w, w->pretty ? ": " : ":", w->pretty ? 2 : 1);
    w->expect_value = 1;
}

void rrc_jw_strn(rrc_jw *w, const char *s, size_t n) {
    jw_before_value(w);
    if (!s) {
        w->err = RRC_ERR_INVALID_ARG;
        return;
    }
    jw_string_body(w, s, n);
}
void rrc_jw_str(rrc_jw *w, const char *s) { rrc_jw_strn(w, s, s ? strlen(s) : 0); }

void rrc_jw_int(rrc_jw *w, int64_t v) {
    jw_before_value(w);
    char b[32];
    int n = snprintf(b, sizeof b, "%lld", (long long)v);
    jw_raw(w, b, (size_t)n);
}

void rrc_jw_double(rrc_jw *w, double v) {
    jw_before_value(w);
    if (!isfinite(v)) {
        jw_raw(w, "null", 4);
        return;
    }
    char b[40];
    int n = 0;
    for (int prec = 15; prec <= 17; prec++) {
        n = snprintf(b, sizeof b, "%.*g", prec, v);
        if (strtod(b, NULL) == v) break;
    }
    if (!strpbrk(b, ".eEn")) {
        b[n++] = '.';
        b[n++] = '0';
        b[n] = '\0';
    }
    jw_raw(w, b, (size_t)n);
}

void rrc_jw_bool(rrc_jw *w, int v) {
    jw_before_value(w);
    jw_raw(w, v ? "true" : "false", v ? 4 : 5);
}
void rrc_jw_null(rrc_jw *w) {
    jw_before_value(w);
    jw_raw(w, "null", 4);
}

rrc_status rrc_jw_finish(rrc_jw *w, char **out, size_t *out_len) {
    rrc_status st = RRC_OK;
    if (w->err == RRC_ERR_OOM) st = RRC_ERROR(RRC_ERR_OOM, "JSON writer buffer");
    else if (w->err) st = RRC_ERROR(RRC_ERR_INVALID_ARG, "JSON writer misuse (value without key, unbalanced, ...)");
    else if (w->depth != 0 || w->len == 0) st = RRC_ERROR(RRC_ERR_INVALID_ARG, "JSON writer: incomplete document");
    if (st == RRC_OK && w->pretty) jw_raw(w, "\n", 1);
    if (st == RRC_OK && w->err) st = RRC_ERROR(RRC_ERR_OOM, "JSON writer buffer");
    if (st != RRC_OK) {
        rrc_jw_free(w);
        if (out) *out = NULL;
        return st;
    }
    if (out) *out = w->buf;
    else free(w->buf);
    if (out_len) *out_len = w->len;
    w->buf = NULL;
    rrc_jw_free(w);
    return RRC_OK;
}

rrc_status rrc_write_file_atomic(const char *path, const void *data, size_t len) {
    RRC_REQUIRE(path && (data || len == 0), RRC_ERR_INVALID_ARG, "null argument");
    size_t pl = strlen(path);
    char *tmp = (char *)malloc(pl + 8);
    if (!tmp) return RRC_ERROR(RRC_ERR_OOM, "path");
    memcpy(tmp, path, pl);
    memcpy(tmp + pl, ".tmp", 5);
    FILE *f = NULL;
    rrc_status cs = rrc_create_new_file(tmp, &f);
    if (cs != RRC_OK) {
        free(tmp);
        return cs;
    }
    size_t w = len ? fwrite(data, 1, len, f) : 0;
    int ok = (w == len) && fflush(f) == 0;
    ok = (fclose(f) == 0) && ok;
    if (!ok) {
        remove(tmp);
        free(tmp);
        return RRC_ERROR(RRC_ERR_IO, "write to %s failed", path);
    }
    if (rename(tmp, path) != 0) {
        rrc_status st = RRC_ERROR(RRC_ERR_IO, "rename %s -> %s: %s", tmp, path, strerror(errno));
        remove(tmp);
        free(tmp);
        return st;
    }
    free(tmp);
    return RRC_OK;
}
