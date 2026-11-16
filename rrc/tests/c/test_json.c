/* Tests for the strict JSON parser and the writer. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rrc/json.h"
#include "rt.h"

static rrc_status parse_cstr(const char *s, rrc_json_doc **doc) { return rrc_json_parse(s, strlen(s), doc); }

RT_TEST(json_basic_types) {
    rrc_json_doc *doc = NULL;
    RT_OK(parse_cstr(" {\"a\": 1, \"b\": -2.5e3, \"c\": \"x\\u00e9\\ud83d\\ude00\", \"d\": [true, false, null],"
                     " \"e\": {}, \"f\": [], \"big\": 9223372036854775807, \"neg\": -9223372036854775808,"
                     " \"huge\": 9223372036854775808, \"z\": 0, \"mz\": -0, \"fr\": 0.5} ",
                     &doc));
    const rrc_json *r = rrc_json_root(doc);
    RT_ASSERT(r && r->type == RRC_JSON_OBJECT, "root type");
    RT_EXPECT(rrc_json_obj_count(r) == 12, "count %zu", rrc_json_obj_count(r));
    /* insertion order is kept */
    const char *order[] = {"a", "b", "c", "d", "e", "f", "big", "neg", "huge", "z", "mz", "fr"};
    for (size_t i = 0; i < 12; i++)
        RT_EXPECT(!strcmp(rrc_json_obj_key(r, i), order[i]), "key %zu = %s", i, rrc_json_obj_key(r, i));
    const rrc_json *a = rrc_json_get(r, "a");
    RT_EXPECT(a && a->type == RRC_JSON_INT && a->u.i == 1, "a");
    const rrc_json *b = rrc_json_get(r, "b");
    RT_EXPECT(b && b->type == RRC_JSON_DOUBLE && b->u.d == -2500.0, "b");
    const rrc_json *c = rrc_json_get(r, "c");
    RT_EXPECT(c && c->type == RRC_JSON_STRING && c->u.s.len == 7 &&
                  !memcmp(c->u.s.ptr, "x\xc3\xa9\xf0\x9f\x98\x80", 7),
              "utf-8 decoding of escapes incl. surrogate pair");
    const rrc_json *d = rrc_json_get(r, "d");
    RT_EXPECT(rrc_json_arr_count(d) == 3 && rrc_json_arr_get(d, 0)->u.boolean == 1 &&
                  rrc_json_arr_get(d, 1)->u.boolean == 0 && rrc_json_arr_get(d, 2)->type == RRC_JSON_NULL,
              "array");
    RT_EXPECT(rrc_json_get(r, "e")->type == RRC_JSON_OBJECT && rrc_json_obj_count(rrc_json_get(r, "e")) == 0, "e");
    RT_EXPECT(rrc_json_get(r, "f")->type == RRC_JSON_ARRAY && rrc_json_arr_count(rrc_json_get(r, "f")) == 0, "f");
    RT_EXPECT(rrc_json_get(r, "big")->type == RRC_JSON_INT && rrc_json_get(r, "big")->u.i == INT64_MAX, "max");
    RT_EXPECT(rrc_json_get(r, "neg")->type == RRC_JSON_INT && rrc_json_get(r, "neg")->u.i == INT64_MIN, "min");
    RT_EXPECT(rrc_json_get(r, "huge")->type == RRC_JSON_DOUBLE && rrc_json_get(r, "huge")->u.d == 9223372036854775808.0,
              "int64 overflow becomes double");
    RT_EXPECT(rrc_json_get(r, "mz")->type == RRC_JSON_INT && rrc_json_get(r, "mz")->u.i == 0, "-0");
    RT_EXPECT(rrc_json_get(r, "missing") == NULL, "missing key");
    int64_t iv;
    double dv;
    RT_OK(rrc_json_as_int(a, "a", &iv));
    RT_STATUS(rrc_json_as_int(b, "b", &iv), RRC_ERR_PARSE);
    RT_OK(rrc_json_as_number(a, "a", &dv));
    RT_EXPECT(dv == 1.0, "as_number int");
    RT_STATUS(rrc_json_as_number(c, "c", &dv), RRC_ERR_PARSE);
    RT_STATUS(rrc_json_as_int(NULL, "absent", &iv), RRC_ERR_PARSE);
    RT_EXPECT(strstr(rrc_last_error(), "absent") != NULL, "message names the key: %s", rrc_last_error());
    rrc_json_doc_free(doc);
}

RT_TEST(json_embedded_nul_and_escapes) {
    rrc_json_doc *doc = NULL;
    RT_OK(parse_cstr("\"a\\u0000b\\\"\\\\\\/\\b\\f\\n\\r\\t\"", &doc));
    const rrc_json *r = rrc_json_root(doc);
    RT_EXPECT(r->type == RRC_JSON_STRING && r->u.s.len == 11, "len %zu", r->u.s.len);
    RT_EXPECT(!memcmp(r->u.s.ptr, "a\0b\"\\/\b\f\n\r\t", 11), "content");
    rrc_json_doc_free(doc);
    /* raw multi-byte UTF-8 is kept as-is */
    RT_OK(parse_cstr("[\"\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80\xf4\x8f\xbf\xbf\"]", &doc));
    RT_EXPECT(rrc_json_arr_get(rrc_json_root(doc), 0)->u.s.len == 13, "utf8 len");
    rrc_json_doc_free(doc);
    /* top-level scalars */
    RT_OK(parse_cstr(" 12 ", &doc));
    RT_EXPECT(rrc_json_root(doc)->type == RRC_JSON_INT, "scalar root");
    rrc_json_doc_free(doc);
    RT_OK(parse_cstr("1E+2", &doc));
    RT_EXPECT(rrc_json_root(doc)->type == RRC_JSON_DOUBLE && rrc_json_root(doc)->u.d == 100.0, "1E+2");
    rrc_json_doc_free(doc);
}

/* Every one of these must be rejected with RRC_ERR_PARSE and a message. */
static const char *const k_bad[] = {
    "",
    " ",
    "{",
    "}",
    "[",
    "[1,]",
    "[,1]",
    "{\"a\":1,}",
    "{\"a\" 1}",
    "{a:1}",
    "{'a':1}",
    "{\"a\":1 \"b\":2}",
    "[1 2]",
    "01",
    "-01",
    "+1",
    "1.",
    ".5",
    "1e",
    "1e+",
    "-",
    "NaN",
    "Infinity",
    "-Infinity",
    "nul",
    "truex",
    "tru",
    "nulll",
    "1e400",
    "-1e400",
    "\"abc",
    "\"\\x\"",
    "\"\\u12\"",
    "\"\\u12G4\"",
    "\"\\ud800\"",
    "\"\\ud800x\"",
    "\"\\ud800\\u0041\"",
    "\"\\udc00\"",
    "\"tab\there\"",
    "\"nl\nhere\"",
    "\"\xff\"",
    "\"\xc0\xaf\"",         /* overlong '/' */
    "\"\xe0\x80\xaf\"",     /* overlong */
    "\"\xed\xa0\x80\"",     /* encoded surrogate */
    "\"\xf4\x90\x80\x80\"", /* > U+10FFFF */
    "\"\xc3\"",             /* truncated sequence */
    "\"\x80\"",             /* lone continuation byte */
    "\xef\xbb\xbf{}",       /* BOM */
    "{} {}",
    "[1] x",
    "{\"a\":1,\"a\":2}",
    "/* c */ {}",
    "{} // c",
    "[1,2,3,]",
    "{\"a\":[}",
    "[\"a\":1]",
    "\x01",
    "{\"a\":1}\x00",
};

RT_TEST(json_rejects_malformed) {
    for (size_t i = 0; i < sizeof k_bad / sizeof k_bad[0]; i++) {
        rrc_json_doc *doc = (rrc_json_doc *)0x1;
        size_t len = strlen(k_bad[i]);
        if (i == sizeof k_bad / sizeof k_bad[0] - 1) len += 1; /* include the embedded NUL */
        rrc_clear_error();
        rrc_status st = rrc_json_parse(k_bad[i], len, &doc);
        RT_EXPECT(st == RRC_ERR_PARSE, "input #%zu (%s) gave %s", i, k_bad[i], rrc_status_str(st));
        RT_EXPECT(doc == NULL, "doc must be NULL on failure (#%zu)", i);
        RT_EXPECT(strstr(rrc_last_error(), "JSON ") != NULL, "message for #%zu: %s", i, rrc_last_error());
        if (st == RRC_OK) rrc_json_doc_free(doc);
    }
}

RT_TEST(json_error_position) {
    rrc_json_doc *doc = NULL;
    RT_STATUS(parse_cstr("{\n  \"a\": 1,\n  \"b\": tru\n}", &doc), RRC_ERR_PARSE);
    RT_EXPECT(strstr(rrc_last_error(), "JSON 3:8:") != NULL, "position: %s", rrc_last_error());
}

RT_TEST(json_depth_limit) {
    char buf[2 * (RRC_JSON_MAX_DEPTH + 2) + 1];
    rrc_json_doc *doc = NULL;
    for (int depth = RRC_JSON_MAX_DEPTH; depth <= RRC_JSON_MAX_DEPTH + 1; depth++) {
        int n = 0;
        for (int i = 0; i < depth; i++) buf[n++] = '[';
        for (int i = 0; i < depth; i++) buf[n++] = ']';
        if (depth == RRC_JSON_MAX_DEPTH) {
            RT_OK(rrc_json_parse(buf, (size_t)n, &doc));
            rrc_json_doc_free(doc);
        } else {
            RT_STATUS(rrc_json_parse(buf, (size_t)n, &doc), RRC_ERR_PARSE);
        }
    }
}

/* Deterministic pseudo-random mutation fuzzing: truncations, byte flips and insertions of a valid
 * document must either parse or fail cleanly with RRC_ERR_PARSE (never crash, never another
 * status). Run under the test binary; ASan/valgrind can be added by the user. */
RT_TEST(json_mutation_fuzz) {
    const char *seed =
        "{\"model_type\": \"llama\", \"rope_scaling\": {\"factor\": 8.0, \"rope_type\": \"llama3\"},"
        " \"list\": [1, -2, 3.5e-3, true, null, \"s\\u00e9\\ud83d\\ude00\"], \"n\": {\"x\": []}}";
    size_t n = strlen(seed);
    char *buf = (char *)malloc(n + 8);
    RT_ASSERT(buf, "oom");
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    int bad = 0, ok = 0, rejected = 0;
    for (int iter = 0; iter < 20000; iter++) {
        memcpy(buf, seed, n);
        size_t len = n;
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        int mode = (int)(rng % 4);
        size_t pos = (size_t)((rng >> 8) % n);
        unsigned char byte = (unsigned char)(rng >> 40);
        if (mode == 0) len = pos;                        /* truncate */
        else if (mode == 1) buf[pos] = (char)byte;       /* replace */
        else if (mode == 2) buf[pos] ^= (char)(1u << (byte % 8)); /* bit flip */
        else {                                           /* insert */
            memmove(buf + pos + 1, buf + pos, n - pos);
            buf[pos] = (char)byte;
            len = n + 1;
        }
        rrc_json_doc *doc = NULL;
        rrc_status st = rrc_json_parse(buf, len, &doc);
        if (st == RRC_OK) {
            ok++;
            RT_EXPECT(rrc_json_root(doc) != NULL, "root");
            rrc_json_doc_free(doc);
        } else if (st == RRC_ERR_PARSE) {
            rejected++;
        } else if (bad++ < 3) {
            RT_EXPECT(0, "mutation %d gave %s", iter, rrc_status_str(st));
        }
    }
    free(buf);
    RT_EXPECT(bad == 0, "%d unexpected statuses", bad);
    RT_EXPECT(ok > 0 && rejected > 0, "fuzz coverage ok=%d rejected=%d", ok, rejected);
}

RT_TEST(json_writer_roundtrip) {
    rrc_jw w;
    rrc_jw_init(&w, 0);
    rrc_jw_obj_begin(&w);
    rrc_jw_key(&w, "step");
    rrc_jw_int(&w, 12);
    rrc_jw_key(&w, "loss");
    rrc_jw_double(&w, 0.1);
    rrc_jw_key(&w, "two");
    rrc_jw_double(&w, 2.0);
    rrc_jw_key(&w, "nan");
    rrc_jw_double(&w, NAN);
    rrc_jw_key(&w, "s\"q");
    rrc_jw_str(&w, "a\nb\x01\xc3\xa9");
    rrc_jw_key(&w, "arr");
    rrc_jw_arr_begin(&w);
    rrc_jw_bool(&w, 1);
    rrc_jw_null(&w);
    rrc_jw_double(&w, 1e-6);
    rrc_jw_arr_begin(&w);
    rrc_jw_arr_end(&w);
    rrc_jw_arr_end(&w);
    rrc_jw_key(&w, "empty");
    rrc_jw_obj_begin(&w);
    rrc_jw_obj_end(&w);
    rrc_jw_obj_end(&w);
    char *out = NULL;
    size_t len = 0;
    RT_OK(rrc_jw_finish(&w, &out, &len));
    const char *want = "{\"step\":12,\"loss\":0.1,\"two\":2.0,\"nan\":null,\"s\\\"q\":\"a\\nb\\u0001\xc3\xa9\","
                       "\"arr\":[true,null,1e-06,[]],\"empty\":{}}";
    RT_EXPECT(!strcmp(out, want), "got %s", out);
    RT_EXPECT(len == strlen(out), "len");
    rrc_json_doc *doc = NULL;
    RT_OK(rrc_json_parse(out, len, &doc));
    double d;
    RT_OK(rrc_json_as_number(rrc_json_get(rrc_json_root(doc), "loss"), "loss", &d));
    RT_EXPECT(d == 0.1, "loss roundtrip");
    RT_EXPECT(rrc_json_get(rrc_json_root(doc), "two")->type == RRC_JSON_DOUBLE, "2.0 stays a double");
    rrc_json_doc_free(doc);
    free(out);

    /* pretty output parses too */
    rrc_jw_init(&w, 1);
    rrc_jw_obj_begin(&w);
    rrc_jw_key(&w, "r");
    rrc_jw_int(&w, 16);
    rrc_jw_key(&w, "targets");
    rrc_jw_arr_begin(&w);
    rrc_jw_str(&w, "q_proj");
    rrc_jw_str(&w, "v_proj");
    rrc_jw_arr_end(&w);
    rrc_jw_obj_end(&w);
    RT_OK(rrc_jw_finish(&w, &out, &len));
    RT_EXPECT(!strcmp(out, "{\n  \"r\": 16,\n  \"targets\": [\n    \"q_proj\",\n    \"v_proj\"\n  ]\n}\n"), "pretty: %s",
              out);
    RT_OK(rrc_json_parse(out, len, &doc));
    rrc_json_doc_free(doc);
    free(out);

    /* round-trip of doubles that need 17 digits */
    double vals[] = {0.1 + 0.2, 1.0 / 3.0, 5e-324, 1.7976931348623157e308, -123456.789};
    for (size_t i = 0; i < sizeof vals / sizeof vals[0]; i++) {
        rrc_jw_init(&w, 0);
        rrc_jw_double(&w, vals[i]);
        RT_OK(rrc_jw_finish(&w, &out, &len));
        RT_OK(rrc_json_parse(out, len, &doc));
        RT_EXPECT(rrc_json_root(doc)->u.d == vals[i], "double %zu: %s", i, out);
        rrc_json_doc_free(doc);
        free(out);
    }
}

RT_TEST(json_writer_misuse) {
    rrc_jw w;
    char *out = NULL;
    rrc_jw_init(&w, 0);
    rrc_jw_obj_begin(&w);
    rrc_jw_int(&w, 1); /* value without key */
    rrc_jw_obj_end(&w);
    RT_STATUS(rrc_jw_finish(&w, &out, NULL), RRC_ERR_INVALID_ARG);
    RT_EXPECT(out == NULL, "out");
    rrc_jw_init(&w, 0);
    rrc_jw_arr_begin(&w);
    RT_STATUS(rrc_jw_finish(&w, &out, NULL), RRC_ERR_INVALID_ARG); /* unbalanced */
    rrc_jw_init(&w, 0);
    rrc_jw_arr_begin(&w);
    rrc_jw_obj_end(&w); /* wrong close */
    RT_STATUS(rrc_jw_finish(&w, &out, NULL), RRC_ERR_INVALID_ARG);
    rrc_jw_init(&w, 0);
    rrc_jw_int(&w, 1);
    rrc_jw_int(&w, 2); /* two top-level values */
    RT_STATUS(rrc_jw_finish(&w, &out, NULL), RRC_ERR_INVALID_ARG);
    rrc_jw_init(&w, 0);
    rrc_jw_obj_begin(&w);
    rrc_jw_key(&w, "a");
    rrc_jw_obj_end(&w); /* key without value */
    RT_STATUS(rrc_jw_finish(&w, &out, NULL), RRC_ERR_INVALID_ARG);
}

RT_TEST(json_file_io) {
    const char *p = rt_path("j.json");
    const char *text = "{\"k\": [1, 2]}";
    RT_OK(rrc_write_file_atomic(p, text, strlen(text)));
    rrc_json_doc *doc = NULL;
    RT_OK(rrc_json_parse_file(p, 0, &doc));
    RT_EXPECT(rrc_json_arr_count(rrc_json_get(rrc_json_root(doc), "k")) == 2, "content");
    rrc_json_doc_free(doc);
    RT_STATUS(rrc_json_parse_file(rt_path("does_not_exist.json"), 0, &doc), RRC_ERR_NOT_FOUND);
    RT_STATUS(rrc_json_parse_file(p, 4, &doc), RRC_ERR_PARSE); /* larger than max_bytes */
    const char *bad = "{\"k\": [1, 2,]}";
    RT_OK(rrc_write_file_atomic(p, bad, strlen(bad)));
    RT_STATUS(rrc_json_parse_file(p, 0, &doc), RRC_ERR_PARSE);
    RT_EXPECT(strstr(rrc_last_error(), "j.json") != NULL, "file name in message: %s", rrc_last_error());
}

static double mono_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Builds {"k0000000": i, ...} with n keys; key `dup_of` (>= 0) is repeated as the key at `dup_at`. */
static char *big_object(size_t n, long dup_at, long dup_of, size_t *len) {
    char *b = (char *)malloc(n * 16 + 16);
    if (!b) return NULL;
    size_t k = 0;
    b[k++] = '{';
    for (size_t i = 0; i < n; i++) {
        const size_t key = (long)i == dup_at ? (size_t)dup_of : i;
        k += (size_t)sprintf(b + k, "%s\"k%07zu\":%zu", i ? "," : "", key, i % 10);
    }
    b[k++] = '}';
    b[k] = '\0';
    *len = k;
    return b;
}

/* Duplicate keys in large objects (review finding: the per-key linear scan made a 200k-entry
 * safetensors header take minutes): detected at the same key and offset as before, and the parse is
 * near-linear. */
RT_TEST(json_duplicate_keys_large_objects) {
    rrc_json_doc *doc = NULL;
    /* small objects: the first repeated occurrence, at its offset */
    RT_STATUS(parse_cstr("{\"a\": 1, \"b\": 2, \"a\": 3}", &doc), RRC_ERR_PARSE);
    RT_EXPECT(strstr(rrc_last_error(), "duplicate key \"a\"") && strstr(rrc_last_error(), "1:18:"), "%s",
              rrc_last_error());
    /* sizes around the linear / sorted switch, duplicate early, late, and several (earliest reported) */
    const size_t sizes[] = {15, 16, 17, 18, 100, 5000};
    for (size_t si = 0; si < sizeof sizes / sizeof sizes[0]; si++) {
        const size_t n = sizes[si];
        size_t len = 0;
        char *b = big_object(n, -1, -1, &len);
        RT_ASSERT(b, "oom");
        RT_OK(rrc_json_parse(b, len, &doc));
        RT_EXPECT(rrc_json_obj_count(rrc_json_root(doc)) == n, "count");
        rrc_json_doc_free(doc);
        free(b);
        const long at[][2] = {{1, 0}, {(long)n - 1, 0}, {(long)n - 1, (long)n - 2}, {(long)n / 2, 1}};
        for (int c = 0; c < 4; c++) {
            b = big_object(n, at[c][0], at[c][1], &len);
            RT_ASSERT(b, "oom");
            char want[64];
            snprintf(want, sizeof want, "duplicate key \"k%07ld\"", at[c][1]);
            /* key i starts at byte 1 + 13 i (strlen("\"k0000000\":d,") = 13): line 1, column 2 + 13 i */
            char off[64];
            snprintf(off, sizeof off, "1:%ld:", 2 + 13 * at[c][0]);
            RT_STATUS(rrc_json_parse(b, len, &doc), RRC_ERR_PARSE);
            RT_EXPECT(strstr(rrc_last_error(), want) && strstr(rrc_last_error(), off), "n %zu case %d: want %s at %s: %s",
                      n, c, want, off, rrc_last_error());
            free(b);
        }
    }
    {
        /* two different repeated keys: the one repeated first in the input is reported */
        size_t len = 0;
        char *b = big_object(40, 30, 3, &len);
        RT_ASSERT(b, "oom");
        memcpy(b + 1 + 13 * 20 + 2, "0000007", 7); /* key 20 -> k0000007 (repeats earlier than key 30) */
        RT_STATUS(rrc_json_parse(b, len, &doc), RRC_ERR_PARSE);
        RT_EXPECT(strstr(rrc_last_error(), "duplicate key \"k0000007\"") != NULL, "%s", rrc_last_error());
        free(b);
    }
    /* 300k keys (about 3.6 MB, a safetensors header is allowed up to 100 MB): with the quadratic scan
     * this took minutes; the bound is generous for sanitizer builds */
    size_t len = 0;
    char *b = big_object(300000, -1, -1, &len);
    RT_ASSERT(b, "oom");
    const double t0 = mono_s();
    RT_OK(rrc_json_parse(b, len, &doc));
    const double dt = mono_s() - t0;
    rrc_json_doc_free(doc);
    free(b);
    RT_EXPECT(dt < 20.0, "300k-key object took %.1f s to parse", dt);
}

const rt_case rt_suite_json[] = {
    RT_CASE(json_basic_types),   RT_CASE(json_embedded_nul_and_escapes), RT_CASE(json_rejects_malformed),
    RT_CASE(json_error_position), RT_CASE(json_depth_limit),             RT_CASE(json_mutation_fuzz),
    RT_CASE(json_writer_roundtrip), RT_CASE(json_writer_misuse),         RT_CASE(json_file_io),
    RT_CASE(json_duplicate_keys_large_objects),
};
const size_t rt_suite_json_n = sizeof rt_suite_json / sizeof rt_suite_json[0];
