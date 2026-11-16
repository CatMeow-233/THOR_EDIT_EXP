/* rrc/json.h — strict JSON (RFC 8259) parser producing a DOM, and a small writer.
 *
 * Parser rules (anything else is RRC_ERR_PARSE with "line:col" in the message):
 *   - input must be valid UTF-8 (no overlong forms, no encoded surrogates, <= U+10FFFF), no BOM;
 *   - exactly one value, surrounded only by JSON whitespace (space, \t, \n, \r);
 *   - no comments, no trailing commas, no NaN/Infinity, no leading zeros, no leading '+';
 *   - strings: no raw control characters (< 0x20); escapes \" \\ \/ \b \f \n \r \t \uXXXX;
 *     \u surrogate pairs are combined, lone surrogates are rejected; \u0000 is allowed (strings
 *     carry an explicit length and are also NUL-terminated);
 *   - numbers without fraction/exponent that fit int64 are RRC_JSON_INT, every other number is
 *     RRC_JSON_DOUBLE (non-finite results such as 1e400 are rejected);
 *   - objects keep insertion order; duplicate keys are rejected;
 *   - nesting depth is limited to RRC_JSON_MAX_DEPTH.
 * All nodes live in an arena owned by the rrc_json_doc.
 */
#ifndef RRC_JSON_H
#define RRC_JSON_H

#include <stdint.h>

#include "rrc/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RRC_JSON_MAX_DEPTH 256

typedef enum rrc_json_type {
    RRC_JSON_NULL = 0,
    RRC_JSON_BOOL,
    RRC_JSON_INT,
    RRC_JSON_DOUBLE,
    RRC_JSON_STRING,
    RRC_JSON_ARRAY,
    RRC_JSON_OBJECT
} rrc_json_type;

typedef struct rrc_json rrc_json;
typedef struct rrc_json_doc rrc_json_doc;

struct rrc_json {
    rrc_json_type type;
    union {
        int boolean;
        int64_t i;
        double d;
        struct {
            const char *ptr; /* NUL-terminated, may contain embedded NULs */
            size_t len;
        } s;
        struct {
            rrc_json **items;
            size_t count;
        } a;
        struct {
            const char **keys;
            size_t *key_lens;
            rrc_json **vals;
            size_t count;
        } o;
    } u;
};

/* Parse `len` bytes. On success *out owns all nodes (free with rrc_json_doc_free). */
rrc_status rrc_json_parse(const char *text, size_t len, rrc_json_doc **out);
/* Read a whole file (at most `max_bytes`, 0 = 256 MiB) and parse it. */
rrc_status rrc_json_parse_file(const char *path, size_t max_bytes, rrc_json_doc **out);
const rrc_json *rrc_json_root(const rrc_json_doc *doc);
void rrc_json_doc_free(rrc_json_doc *doc);

const char *rrc_json_type_name(rrc_json_type t);

/* Lookup helpers. All return NULL / 0 when `v` is NULL or has the wrong type. */
const rrc_json *rrc_json_get(const rrc_json *obj, const char *key); /* NULL if absent */
size_t rrc_json_obj_count(const rrc_json *obj);
const char *rrc_json_obj_key(const rrc_json *obj, size_t i);
const rrc_json *rrc_json_obj_val(const rrc_json *obj, size_t i);
size_t rrc_json_arr_count(const rrc_json *arr);
const rrc_json *rrc_json_arr_get(const rrc_json *arr, size_t i);
int rrc_json_is_number(const rrc_json *v); /* INT or DOUBLE */
const char *rrc_json_str(const rrc_json *v); /* NULL unless STRING */

/* Typed getters: RRC_ERR_PARSE (with the key in the message) if the type is wrong.
 * get_int accepts only RRC_JSON_INT. get_number accepts INT or DOUBLE. */
rrc_status rrc_json_as_int(const rrc_json *v, const char *what, int64_t *out);
rrc_status rrc_json_as_number(const rrc_json *v, const char *what, double *out);
rrc_status rrc_json_as_bool(const rrc_json *v, const char *what, int *out);
rrc_status rrc_json_as_string(const rrc_json *v, const char *what, const char **out);

/* ------------------------------------------------------------------------------------------ */
/* Writer: builds a JSON text in a growable buffer. Misuse (value without key inside an object,
 * unbalanced end, ...) or OOM is recorded and reported by rrc_jw_finish.                       */
/* ------------------------------------------------------------------------------------------ */

typedef struct rrc_jw {
    char *buf;
    size_t len, cap;
    int depth;
    int pretty;
    rrc_status err;
    unsigned char kind[RRC_JSON_MAX_DEPTH];  /* 'o' object, 'a' array */
    unsigned char count[RRC_JSON_MAX_DEPTH]; /* 0: no element yet, 1: has elements */
    int expect_value;                        /* a key was written, a value must follow */
} rrc_jw;

void rrc_jw_init(rrc_jw *w, int pretty);
void rrc_jw_obj_begin(rrc_jw *w);
void rrc_jw_obj_end(rrc_jw *w);
void rrc_jw_arr_begin(rrc_jw *w);
void rrc_jw_arr_end(rrc_jw *w);
void rrc_jw_key(rrc_jw *w, const char *key);
void rrc_jw_str(rrc_jw *w, const char *s);
void rrc_jw_strn(rrc_jw *w, const char *s, size_t n);
void rrc_jw_int(rrc_jw *w, int64_t v);
/* Shortest representation that round-trips; integral values get ".0"; NaN/Inf are written as
 * null (JSON has no non-finite numbers). */
void rrc_jw_double(rrc_jw *w, double v);
void rrc_jw_bool(rrc_jw *w, int v);
void rrc_jw_null(rrc_jw *w);
/* Check balance and errors; on success *out is a malloc'd NUL-terminated string that the caller
 * frees (ownership moves to the caller). The writer is reset in all cases. */
rrc_status rrc_jw_finish(rrc_jw *w, char **out, size_t *out_len);
void rrc_jw_free(rrc_jw *w);

/* Write `text` to `path` atomically (path.tmp + rename). */
rrc_status rrc_write_file_atomic(const char *path, const void *data, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* RRC_JSON_H */
