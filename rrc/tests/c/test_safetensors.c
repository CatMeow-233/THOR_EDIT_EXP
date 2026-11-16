/* Tests for the safetensors reader/writer, sharded checkpoints and interop with the Python
 * `safetensors` package (skipped with a message when python3 + torch + safetensors are missing). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "rrc/config.h"
#include "rrc/data.h"
#include "rrc/fileio.h"
#include "rrc/json.h"
#include "rrc/safetensors.h"
#include "rrc/train.h"
#include "rt.h"

/* Write a raw safetensors file: 8-byte length (or `hlen_override` if >= 0), header, data. */
static void write_raw(const char *path, const char *header, const void *data, size_t data_len, long long hlen_override) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    uint64_t hl = hlen_override >= 0 ? (uint64_t)hlen_override : (uint64_t)strlen(header);
    unsigned char b[8];
    for (int i = 0; i < 8; i++) b[i] = (unsigned char)(hl >> (8 * i));
    fwrite(b, 1, 8, f);
    fwrite(header, 1, strlen(header), f);
    if (data_len) fwrite(data, 1, data_len, f);
    fclose(f);
}

RT_TEST(st_roundtrip) {
    rrc_st_writer *w = NULL;
    RT_OK(rrc_st_writer_create(&w));
    float a[6] = {1, 2, 3, 4, 5, -6.5f};
    rrc_bf16 b[3] = {rrc_f32_to_bf16(1.5f), rrc_f32_to_bf16(-2.0f), rrc_f32_to_bf16(1e-3f)};
    int32_t c[2] = {7, -8};
    float s = 42.0f;
    int64_t sa[2] = {2, 3}, sb[1] = {3}, sc[2] = {1, 2}, se[2] = {0, 5};
    RT_OK(rrc_st_writer_add(w, "z.bf16", RRC_BF16, 1, sb, b));
    RT_OK(rrc_st_writer_add(w, "a.f32", RRC_F32, 2, sa, a));
    RT_OK(rrc_st_writer_add(w, "m.i32", RRC_I32, 2, sc, c));
    RT_OK(rrc_st_writer_add(w, "scalar", RRC_F32, 0, NULL, &s));
    RT_OK(rrc_st_writer_add(w, "empty", RRC_F32, 2, se, NULL));
    RT_STATUS(rrc_st_writer_add(w, "a.f32", RRC_F32, 2, sa, a), RRC_ERR_INVALID_ARG); /* duplicate */
    RT_STATUS(rrc_st_writer_add(w, "__metadata__", RRC_F32, 2, sa, a), RRC_ERR_INVALID_ARG);
    RT_OK(rrc_st_writer_set_metadata(w, "format", "pt"));
    RT_OK(rrc_st_writer_set_metadata(w, "note", "x"));
    RT_OK(rrc_st_writer_set_metadata(w, "note", "rrc \"quoted\""));
    const char *p = rt_path("rt.safetensors");
    RT_OK(rrc_st_writer_write(w, p));
    rrc_st_writer_free(w);

    rrc_st_set *set = NULL;
    RT_OK(rrc_st_open_file(p, &set));
    RT_EXPECT(rrc_st_count(set) == 5, "count %zu", rrc_st_count(set));
    RT_EXPECT(!strcmp(rrc_st_at(set, 0)->name, "a.f32") && !strcmp(rrc_st_at(set, 4)->name, "z.bf16"), "sorted");
    const rrc_st_tensor *t = rrc_st_find(set, "a.f32");
    RT_ASSERT(t, "a.f32 missing");
    RT_EXPECT(t->dtype == RRC_ST_F32 && t->ndim == 2 && t->shape[0] == 2 && t->shape[1] == 3 && t->numel == 6, "meta");
    RT_EXPECT(((uintptr_t)t->data % 4) == 0, "f32 data aligned (writer orders by element size)");
    float got[6];
    RT_OK(rrc_st_to_f32(t, got, 6));
    RT_EXPECT(!memcmp(got, a, sizeof a), "f32 data");
    RT_STATUS(rrc_st_to_f32(t, got, 5), RRC_ERR_SHAPE);
    t = rrc_st_find(set, "z.bf16");
    RT_ASSERT(t && t->dtype == RRC_ST_BF16, "bf16");
    rrc_bf16 gb[3];
    RT_OK(rrc_st_to_bf16(t, gb, 3));
    RT_EXPECT(!memcmp(gb, b, sizeof b), "bf16 data");
    RT_OK(rrc_st_to_f32(t, got, 3));
    RT_EXPECT(got[0] == 1.5f && got[1] == -2.0f, "bf16 widen");
    t = rrc_st_find(set, "m.i32");
    int32_t gi[2];
    RT_OK(rrc_st_to_i32(t, gi, 2));
    RT_EXPECT(gi[0] == 7 && gi[1] == -8, "i32");
    RT_STATUS(rrc_st_to_f32(t, got, 2), RRC_ERR_DTYPE);
    t = rrc_st_find(set, "scalar");
    RT_EXPECT(t && t->ndim == 0 && t->numel == 1, "scalar");
    t = rrc_st_find(set, "empty");
    RT_EXPECT(t && t->numel == 0 && t->nbytes == 0, "empty");
    RT_EXPECT(rrc_st_find(set, "nope") == NULL, "find missing");
    RT_EXPECT(!strcmp(rrc_st_metadata(set, 0, "format"), "pt"), "metadata");
    RT_EXPECT(!strcmp(rrc_st_metadata(set, 0, "note"), "rrc \"quoted\""), "metadata overwrite + escaping");
    RT_EXPECT(rrc_st_metadata(set, 0, "absent") == NULL, "absent metadata");
    rrc_dtype dt;
    RT_OK(rrc_st_dtype_to_rrc(RRC_ST_BF16, &dt));
    RT_EXPECT(dt == RRC_BF16, "dtype map");
    RT_STATUS(rrc_st_dtype_to_rrc(RRC_ST_F16, &dt), RRC_ERR_UNSUPPORTED);
    RT_STATUS(rrc_st_dtype_to_rrc(RRC_ST_I64, &dt), RRC_ERR_UNSUPPORTED);
    rrc_st_close(set);

    /* header length is padded to a multiple of 8 */
    FILE *f = fopen(p, "rb");
    unsigned char hb[8];
    RT_ASSERT(f && fread(hb, 1, 8, f) == 8, "read header len");
    fclose(f);
    RT_EXPECT(hb[0] % 8 == 0, "header length %u not 8-aligned", hb[0]);
}

typedef struct {
    const char *what;
    const char *header;
    size_t data_len;
    long long hlen_override;
    rrc_status want;
} bad_case;

RT_TEST(st_rejects_malformed) {
    static const bad_case cases[] = {
        {"hlen 0", "", 0, 0, RRC_ERR_PARSE},
        {"hlen beyond file", "{}", 0, 1000, RRC_ERR_PARSE},
        {"hlen huge", "{}", 0, 200000000LL, RRC_ERR_PARSE},
        {"not json", "{abc}", 0, -1, RRC_ERR_PARSE},
        {"not object", "[1]", 0, -1, RRC_ERR_PARSE},
        {"trailing comma", "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},}", 4, -1, RRC_ERR_PARSE},
        {"missing offsets", "{\"a\":{\"dtype\":\"F32\",\"shape\":[1]}}", 4, -1, RRC_ERR_PARSE},
        {"extra key", "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4],\"x\":1}}", 4, -1, RRC_ERR_PARSE},
        {"entry not object", "{\"a\":3}", 0, -1, RRC_ERR_PARSE},
        {"unknown dtype", "{\"a\":{\"dtype\":\"Q7\",\"shape\":[1],\"data_offsets\":[0,4]}}", 4, -1, RRC_ERR_PARSE},
        {"f64 unsupported", "{\"a\":{\"dtype\":\"F64\",\"shape\":[1],\"data_offsets\":[0,8]}}", 8, -1, RRC_ERR_UNSUPPORTED},
        {"bool unsupported", "{\"a\":{\"dtype\":\"BOOL\",\"shape\":[1],\"data_offsets\":[0,1]}}", 1, -1, RRC_ERR_UNSUPPORTED},
        {"negative dim", "{\"a\":{\"dtype\":\"F32\",\"shape\":[-1],\"data_offsets\":[0,4]}}", 4, -1, RRC_ERR_PARSE},
        {"float dim", "{\"a\":{\"dtype\":\"F32\",\"shape\":[1.0],\"data_offsets\":[0,4]}}", 4, -1, RRC_ERR_PARSE},
        {"too many dims", "{\"a\":{\"dtype\":\"F32\",\"shape\":[1,1,1,1,1,1,1,1,1],\"data_offsets\":[0,4]}}", 4, -1,
         RRC_ERR_PARSE},
        {"size mismatch", "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}", 4, -1, RRC_ERR_PARSE},
        {"end < begin", "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,0]}}", 4, -1, RRC_ERR_PARSE},
        {"end beyond data", "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", 4, -1, RRC_ERR_PARSE},
        {"overlap",
         "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},\"b\":{\"dtype\":\"F32\",\"shape\":[1],"
         "\"data_offsets\":[4,8]}}",
         8, -1, RRC_ERR_PARSE},
        {"hole", "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}", 8, -1, RRC_ERR_PARSE},
        {"not covered", "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", 8, -1, RRC_ERR_PARSE},
        {"metadata not object", "{\"__metadata__\":[],\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", 4,
         -1, RRC_ERR_PARSE},
        {"metadata value not string",
         "{\"__metadata__\":{\"k\":1},\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", 4, -1,
         RRC_ERR_PARSE},
        {"negative offset", "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[-4,0]}}", 4, -1, RRC_ERR_PARSE},
        {"empty name", "{\"\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", 4, -1, RRC_ERR_PARSE},
        {"dup key", "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"a\":{\"dtype\":\"F32\",\"shape\":[1],"
                    "\"data_offsets\":[0,4]}}",
         4, -1, RRC_ERR_PARSE},
        {"numel overflow", "{\"a\":{\"dtype\":\"F32\",\"shape\":[4294967296,4294967296],\"data_offsets\":[0,4]}}", 4, -1,
         RRC_ERR_PARSE},
    };
    unsigned char data[16] = {0};
    const char *p = rt_path("bad.safetensors");
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        write_raw(p, cases[i].header, data, cases[i].data_len, cases[i].hlen_override);
        rrc_st_set *set = NULL;
        rrc_clear_error();
        rrc_status st = rrc_st_open_file(p, &set);
        RT_EXPECT(st == cases[i].want, "%s: got %s want %s (%s)", cases[i].what, rrc_status_str(st),
                  rrc_status_str(cases[i].want), rrc_last_error());
        RT_EXPECT(set == NULL, "%s: set must be NULL", cases[i].what);
        RT_EXPECT(st == RRC_OK || rrc_last_error()[0], "%s: no message", cases[i].what);
        if (set) rrc_st_close(set);
    }
    /* tiny files */
    FILE *f = fopen(p, "wb");
    fwrite("abc", 1, 3, f);
    fclose(f);
    rrc_st_set *set = NULL;
    RT_STATUS(rrc_st_open_file(p, &set), RRC_ERR_PARSE);
    RT_STATUS(rrc_st_open_file(rt_path("no_such.safetensors"), &set), RRC_ERR_NOT_FOUND);
    /* a valid hand-made file with header padding spaces is accepted */
    write_raw(p, "{\"a\":{\"dtype\":\"I32\",\"shape\":[1],\"data_offsets\":[0,4]}}   ", data, 4, -1);
    RT_OK(rrc_st_open_file(p, &set));
    rrc_st_close(set);
}

static void write_shard(const char *dir, const char *name, const char **tnames, int n, float base) {
    rrc_st_writer *w = NULL;
    if (rrc_st_writer_create(&w) != RRC_OK) return;
    for (int i = 0; i < n; i++) {
        float v[2] = {base + (float)i, base - (float)i};
        int64_t shape[1] = {2};
        rrc_st_writer_add(w, tnames[i], RRC_F32, 1, shape, v);
    }
    char path[1400];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    rrc_st_writer_write(w, path);
    rrc_st_writer_free(w);
}

static void write_text(const char *path, const char *text) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fputs(text, f);
    fclose(f);
}

RT_TEST(st_sharded) {
    char dir[1200];
    snprintf(dir, sizeof dir, "%s", rt_path("sharded"));
    mkdir(dir, 0755);
    const char *s1[] = {"model.embed_tokens.weight", "model.layers.0.w"};
    const char *s2[] = {"lm_head.weight"};
    write_shard(dir, "model-00001-of-00002.safetensors", s1, 2, 10.0f);
    write_shard(dir, "model-00002-of-00002.safetensors", s2, 1, 20.0f);
    char idx[1400];
    snprintf(idx, sizeof idx, "%s/model.safetensors.index.json", dir);
    write_text(idx, "{\"metadata\": {\"total_size\": 24}, \"weight_map\": {"
                    "\"model.embed_tokens.weight\": \"model-00001-of-00002.safetensors\","
                    "\"model.layers.0.w\": \"model-00001-of-00002.safetensors\","
                    "\"lm_head.weight\": \"model-00002-of-00002.safetensors\"}}");
    rrc_st_set *set = NULL;
    RT_OK(rrc_st_open_checkpoint(dir, &set));
    RT_EXPECT(rrc_st_count(set) == 3 && rrc_st_num_shards(set) == 2, "count");
    const rrc_st_tensor *t = rrc_st_find(set, "lm_head.weight");
    RT_ASSERT(t, "lm_head");
    float v[2];
    RT_OK(rrc_st_to_f32(t, v, 2));
    RT_EXPECT(v[0] == 20.0f, "shard 2 data");
    rrc_st_close(set);
    RT_OK(rrc_st_open_checkpoint(idx, &set)); /* the index file path works too */
    rrc_st_close(set);

    /* missing tensor in shard */
    write_text(idx, "{\"weight_map\": {\"model.embed_tokens.weight\": \"model-00001-of-00002.safetensors\","
                    "\"model.layers.0.w\": \"model-00001-of-00002.safetensors\","
                    "\"lm_head.weight\": \"model-00002-of-00002.safetensors\","
                    "\"ghost\": \"model-00002-of-00002.safetensors\"}}");
    RT_STATUS(rrc_st_open_checkpoint(dir, &set), RRC_ERR_NOT_FOUND);
    /* tensor present in a shard but not listed */
    write_text(idx, "{\"weight_map\": {\"model.embed_tokens.weight\": \"model-00001-of-00002.safetensors\","
                    "\"lm_head.weight\": \"model-00002-of-00002.safetensors\"}}");
    RT_STATUS(rrc_st_open_checkpoint(dir, &set), RRC_ERR_PARSE);
    /* listed for the wrong shard */
    write_text(idx, "{\"weight_map\": {\"model.embed_tokens.weight\": \"model-00002-of-00002.safetensors\","
                    "\"model.layers.0.w\": \"model-00001-of-00002.safetensors\","
                    "\"lm_head.weight\": \"model-00002-of-00002.safetensors\"}}");
    RT_STATUS(rrc_st_open_checkpoint(dir, &set), RRC_ERR_PARSE);
    /* path traversal in shard names */
    write_text(idx, "{\"weight_map\": {\"lm_head.weight\": \"../model-00002-of-00002.safetensors\"}}");
    RT_STATUS(rrc_st_open_checkpoint(dir, &set), RRC_ERR_PARSE);
    /* missing shard file */
    write_text(idx, "{\"weight_map\": {\"lm_head.weight\": \"model-00009-of-00009.safetensors\"}}");
    RT_STATUS(rrc_st_open_checkpoint(dir, &set), RRC_ERR_NOT_FOUND);
    /* no weight_map */
    write_text(idx, "{\"metadata\": {}}");
    RT_STATUS(rrc_st_open_checkpoint(dir, &set), RRC_ERR_PARSE);
    /* duplicate tensor across shards */
    const char *dup[] = {"lm_head.weight"};
    write_shard(dir, "model-00001-of-00002.safetensors", dup, 1, 1.0f);
    write_text(idx, "{\"weight_map\": {\"lm_head.weight\": \"model-00001-of-00002.safetensors\","
                    "\"x\": \"model-00002-of-00002.safetensors\"}}");
    RT_STATUS(rrc_st_open_checkpoint(dir, &set), RRC_ERR_PARSE);
    /* empty directory */
    char empty[1300];
    snprintf(empty, sizeof empty, "%s/empty", dir);
    mkdir(empty, 0755);
    RT_STATUS(rrc_st_open_checkpoint(empty, &set), RRC_ERR_NOT_FOUND);
}

/* Reference half -> float via ldexp (independent of the bit-manipulation implementation). */
static double ref_f16(uint16_t h) {
    int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    double v;
    if (e == 0) v = ldexp((double)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = ldexp(1024.0 + m, e - 25);
    return s ? -v : v;
}

RT_TEST(st_f16_exhaustive) {
    long bad = 0;
    for (uint32_t h = 0; h < 65536u; h++) {
        float f = rrc_f16_to_f32((uint16_t)h);
        double r = ref_f16((uint16_t)h);
        if (isnan(r)) {
            if (!isnan(f) && bad++ < 5) RT_EXPECT(0, "f16 0x%04x not NaN", h);
            /* payload is kept: upper mantissa bits match */
            if (isnan(f) && ((rrc_f32_bits(f) >> 13) & 0x3FF) != (h & 0x3FF) && bad++ < 5)
                RT_EXPECT(0, "payload 0x%04x", h);
        } else if ((double)f != r || !signbit(f) != !signbit(r)) {
            if (bad++ < 5) RT_EXPECT(0, "f16 0x%04x -> %a want %a", h, (double)f, r);
        }
    }
    RT_EXPECT(bad == 0, "%ld bad", bad);
}

RT_TEST(st_i64_and_f16_conversion) {
    const char *p = rt_path("conv.safetensors");
    int64_t iv[3] = {1, -2, (int64_t)1 << 40};
    uint16_t hv[2] = {0x3C00, 0xC000}; /* 1.0, -2.0 */
    unsigned char data[28];
    memcpy(data, iv, 24);
    memcpy(data + 24, hv, 4);
    write_raw(p,
              "{\"i\":{\"dtype\":\"I64\",\"shape\":[3],\"data_offsets\":[0,24]},"
              "\"h\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[24,28]}}",
              data, 28, -1);
    rrc_st_set *set = NULL;
    RT_OK(rrc_st_open_file(p, &set));
    int32_t out[3];
    RT_STATUS(rrc_st_to_i32(rrc_st_find(set, "i"), out, 3), RRC_ERR_RANGE);
    float f[2];
    RT_OK(rrc_st_to_f32(rrc_st_find(set, "h"), f, 2));
    RT_EXPECT(f[0] == 1.0f && f[1] == -2.0f, "f16 widen");
    rrc_bf16 b[2];
    RT_OK(rrc_st_to_bf16(rrc_st_find(set, "h"), b, 2));
    RT_EXPECT(b[0] == 0x3F80 && b[1] == 0xC000, "f16 -> bf16");
    RT_STATUS(rrc_st_to_i32(rrc_st_find(set, "h"), out, 2), RRC_ERR_DTYPE);
    rrc_st_close(set);
    iv[2] = -3;
    memcpy(data, iv, 24);
    write_raw(p,
              "{\"i\":{\"dtype\":\"I64\",\"shape\":[3],\"data_offsets\":[0,24]},"
              "\"h\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[24,28]}}",
              data, 28, -1);
    RT_OK(rrc_st_open_file(p, &set));
    RT_OK(rrc_st_to_i32(rrc_st_find(set, "i"), out, 3));
    RT_EXPECT(out[0] == 1 && out[1] == -2 && out[2] == -3, "i64 -> i32");
    rrc_st_close(set);
}

static int python_available(void) {
    return system("python3 -c 'import torch, safetensors.torch' >/dev/null 2>&1") == 0;
}

static const char *k_py_writer =
    "import sys, torch\n"
    "from safetensors.torch import save_file\n"
    "g = torch.Generator().manual_seed(0)\n"
    "x = torch.randn(4000, generator=g) * torch.logspace(-30, 30, 4000)\n"
    "sp = torch.tensor([0.0, -0.0, 1.0, 1.00390625, 1.01171875, float('inf'), float('-inf'),\n"
    "                   3.4028234663852886e38, 3.3895313892515355e38, 1e-40, -1e-45, 65504.0, 65520.0,\n"
    "                   1e-8, 6e-8, float('nan')])\n"
    "f = torch.cat([x, sp])\n"
    "save_file({'f32': f, 'bf16': f.to(torch.bfloat16), 'f16': f.to(torch.float16),\n"
    "           'f16_as_f32': f.to(torch.float16).float(),\n"
    "           'i64': torch.arange(-5, 5, dtype=torch.int64).reshape(2, 5),\n"
    "           'i32': torch.tensor([[1, -2], [3, 2147483647]], dtype=torch.int32),\n"
    "           'scalar': torch.tensor(2.5), 'empty': torch.zeros(0, 3),\n"
    "           'x3d': torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)},\n"
    "          sys.argv[1], metadata={'format': 'pt', 'who': 'python'})\n";

static const char *k_py_reader =
    "import sys, torch\n"
    "from safetensors import safe_open\n"
    "from safetensors.torch import load_file\n"
    "t = load_file(sys.argv[1])\n"
    "assert sorted(t) == ['a', 'b', 'i'], sorted(t)\n"
    "assert t['a'].dtype == torch.float32 and t['a'].shape == (2, 3)\n"
    "assert torch.equal(t['a'], torch.arange(6, dtype=torch.float32).reshape(2, 3) - 2.5)\n"
    "ref = torch.tensor([0.1, -3.0, 1e30, 7.0]).to(torch.bfloat16)\n"
    "assert t['b'].dtype == torch.bfloat16 and torch.equal(t['b'], ref), (t['b'], ref)\n"
    "assert t['i'].dtype == torch.int32 and t['i'].tolist() == [[1, 2], [3, -4]]\n"
    "with safe_open(sys.argv[1], 'pt') as f:\n"
    "    assert f.metadata() == {'format': 'pt', 'lib': 'rrc'}, f.metadata()\n"
    "print('python reader OK')\n";

RT_TEST(st_python_interop) {
    if (!python_available()) {
        rt_skip("python3 with torch + safetensors not available: interop sub-test skipped");
        return;
    }
    const char *script = rt_path("st_writer.py");
    const char *file = rt_path("py.safetensors");
    write_text(script, k_py_writer);
    char cmd[4096];
    snprintf(cmd, sizeof cmd, "python3 -I '%s' '%s'", script, file);
    RT_ASSERT(system(cmd) == 0, "python writer failed: %s", cmd);
    rrc_st_set *set = NULL;
    RT_OK(rrc_st_open_file(file, &set));
    RT_EXPECT(rrc_st_count(set) == 9, "count %zu", rrc_st_count(set));
    RT_EXPECT(!strcmp(rrc_st_metadata(set, 0, "who"), "python"), "metadata");
    const rrc_st_tensor *tf = rrc_st_find(set, "f32"), *tb = rrc_st_find(set, "bf16"), *th = rrc_st_find(set, "f16"),
                        *thf = rrc_st_find(set, "f16_as_f32");
    RT_ASSERT(tf && tb && th && thf, "tensors missing");
    int64_t n = tf->numel;
    RT_ASSERT(n == 4016 && tb->numel == n && th->numel == n && thf->numel == n, "numel");
    float *f = (float *)malloc((size_t)n * 4), *hf = (float *)malloc((size_t)n * 4), *ours = (float *)malloc((size_t)n * 4);
    rrc_bf16 *b = (rrc_bf16 *)malloc((size_t)n * 2), *bours = (rrc_bf16 *)malloc((size_t)n * 2);
    RT_ASSERT(f && hf && ours && b && bours, "oom");
    RT_OK(rrc_st_to_f32(tf, f, n));
    RT_OK(rrc_st_to_bf16(tb, b, n));
    RT_OK(rrc_st_to_f32(thf, hf, n));
    RT_OK(rrc_st_to_f32(th, ours, n));
    RT_OK(rrc_st_to_bf16(tf, bours, n));
    long bad_bf = 0, bad_h = 0;
    for (int64_t i = 0; i < n; i++) {
        /* bf16 RNE must match torch bit-for-bit (NaN: both NaN; torch canonicalises the payload) */
        if (isnan(f[i])) {
            if (!isnan(rrc_bf16_to_f32(b[i])) || !isnan(rrc_bf16_to_f32(bours[i]))) bad_bf++;
        } else if (b[i] != bours[i]) {
            if (bad_bf++ < 3) RT_EXPECT(0, "bf16 mismatch at %lld: f=%a torch=0x%04x rrc=0x%04x", (long long)i,
                                        (double)f[i], b[i], bours[i]);
        }
        /* f16 widening must match torch exactly */
        if (isnan(hf[i]) ? !isnan(ours[i]) : rrc_f32_bits(hf[i]) != rrc_f32_bits(ours[i])) bad_h++;
    }
    RT_EXPECT(bad_bf == 0, "%ld bf16 mismatches vs torch", bad_bf);
    RT_EXPECT(bad_h == 0, "%ld f16 widening mismatches vs torch", bad_h);
    free(f);
    free(hf);
    free(ours);
    free(b);
    free(bours);
    const rrc_st_tensor *t = rrc_st_find(set, "i64");
    int32_t iv[10];
    RT_OK(rrc_st_to_i32(t, iv, 10));
    RT_EXPECT(t->shape[0] == 2 && t->shape[1] == 5 && iv[0] == -5 && iv[9] == 4, "i64");
    t = rrc_st_find(set, "x3d");
    RT_EXPECT(t && t->ndim == 3 && t->shape[2] == 4, "3d");
    t = rrc_st_find(set, "scalar");
    RT_EXPECT(t && t->ndim == 0 && t->numel == 1, "scalar");
    t = rrc_st_find(set, "empty");
    RT_EXPECT(t && t->numel == 0 && t->shape[0] == 0 && t->shape[1] == 3, "empty");
    rrc_st_close(set);

    /* rrc writes, the Python package reads */
    rrc_st_writer *w = NULL;
    RT_OK(rrc_st_writer_create(&w));
    float a[6];
    for (int i = 0; i < 6; i++) a[i] = (float)i - 2.5f;
    float braw[4] = {0.1f, -3.0f, 1e30f, 7.0f};
    rrc_bf16 bb[4];
    for (int i = 0; i < 4; i++) bb[i] = rrc_f32_to_bf16(braw[i]);
    int32_t ii[4] = {1, 2, 3, -4};
    int64_t sa[2] = {2, 3}, sb[1] = {4}, si[2] = {2, 2};
    RT_OK(rrc_st_writer_add(w, "b", RRC_BF16, 1, sb, bb));
    RT_OK(rrc_st_writer_add(w, "a", RRC_F32, 2, sa, a));
    RT_OK(rrc_st_writer_add(w, "i", RRC_I32, 2, si, ii));
    RT_OK(rrc_st_writer_set_metadata(w, "format", "pt"));
    RT_OK(rrc_st_writer_set_metadata(w, "lib", "rrc"));
    const char *out = rt_path("rrc_written.safetensors");
    RT_OK(rrc_st_writer_write(w, out));
    rrc_st_writer_free(w);
    const char *rscript = rt_path("st_reader.py");
    write_text(rscript, k_py_reader);
    snprintf(cmd, sizeof cmd, "python3 -I '%s' '%s' >/dev/null", rscript, out);
    RT_EXPECT(system(cmd) == 0, "python could not read the rrc-written file (%s)", cmd);
}

/* Input files that are FIFOs (review finding: open() ran before the regular-file check, so a FIFO
 * named model.safetensors or config.json blocked the program forever). alarm(): a regression fails
 * the run instead of hanging it. */
RT_TEST(inputs_refuse_fifo_without_blocking) {
    const char *dir = rt_path("fifo_dir");
    mkdir(dir, 0755);
    char fifo[1400], cfgp[1400];
    snprintf(fifo, sizeof fifo, "%s/model.safetensors", dir);
    snprintf(cfgp, sizeof cfgp, "%s/config.json", dir);
    unlink(fifo);
    unlink(cfgp);
    RT_ASSERT(mkfifo(fifo, 0644) == 0 && mkfifo(cfgp, 0644) == 0, "mkfifo");
    alarm(60);
    rrc_st_set *set = NULL;
    RT_STATUS(rrc_st_open_file(fifo, &set), RRC_ERR_IO);
    RT_EXPECT(strstr(rrc_last_error(), "not a regular file") != NULL, "%s", rrc_last_error());
    RT_STATUS(rrc_st_open_checkpoint(dir, &set), RRC_ERR_IO);
    rrc_json_doc *doc = NULL;
    RT_STATUS(rrc_json_parse_file(cfgp, 0, &doc), RRC_ERR_IO);
    rrc_model_config c;
    RT_STATUS(rrc_config_load(dir, 0, &c), RRC_ERR_IO);
    rrc_dataset *d = NULL;
    RT_STATUS(rrc_dataset_open(fifo, 24, &d), RRC_ERR_IO);
    rrc_train_state ts;
    RT_STATUS(rrc_train_state_read(fifo, &ts), RRC_ERR_IO);
    static rrc_train_config tc;
    RT_STATUS(rrc_train_config_load(cfgp, &tc), RRC_ERR_IO);
    RT_EXPECT(strstr(rrc_last_error(), "not a regular file") != NULL, "%s", rrc_last_error());
    alarm(0);
    /* a missing file is still NOT_FOUND */
    RT_STATUS(rrc_json_parse_file(rt_path("fifo_dir_missing.json"), 0, &doc), RRC_ERR_NOT_FOUND);
    unlink(fifo);
    unlink(cfgp);
}

/* Temporary files of atomic writes are created fresh (review finding: fopen(path.tmp, "wb") followed
 * a planted symlink and overwrote its target). */
RT_TEST(atomic_write_does_not_follow_tmp_symlink) {
    const char *victim = rt_path("symlink_victim.txt");
    const char *target = rt_path("symlink_target.json");
    char tmp[1400];
    snprintf(tmp, sizeof tmp, "%s.tmp", target);
    RT_OK(rrc_write_file_atomic(victim, "precious", 8));
    unlink(tmp);
    RT_ASSERT(symlink(victim, tmp) == 0, "symlink");
    RT_OK(rrc_write_file_atomic(target, "{}", 2));
    char buf[32] = {0};
    FILE *f = fopen(victim, "rb");
    RT_ASSERT(f, "victim");
    const size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    RT_EXPECT(n == 8 && !memcmp(buf, "precious", 8), "the symlink target was overwritten: '%s'", buf);
    struct stat sb;
    RT_EXPECT(lstat(target, &sb) == 0 && S_ISREG(sb.st_mode), "target is a regular file");
    RT_EXPECT(lstat(tmp, &sb) != 0, "no .tmp left");
    /* rrc_fopen_output_nofollow refuses a symlink, rrc_create_new_file replaces it */
    RT_ASSERT(symlink(victim, tmp) == 0, "symlink");
    FILE *o = NULL;
    RT_STATUS(rrc_fopen_output_nofollow(tmp, 0, &o), RRC_ERR_IO);
    RT_STATUS(rrc_fopen_output_nofollow(tmp, 1, &o), RRC_ERR_IO);
    RT_OK(rrc_create_new_file(tmp, &o));
    fclose(o);
    RT_EXPECT(lstat(tmp, &sb) == 0 && S_ISREG(sb.st_mode), "created as a regular file");
    unlink(tmp);
}

const rt_case rt_suite_safetensors[] = {
    RT_CASE(st_roundtrip),     RT_CASE(st_rejects_malformed),     RT_CASE(st_sharded),
    RT_CASE(st_f16_exhaustive), RT_CASE(st_i64_and_f16_conversion), RT_CASE(st_python_interop),
    RT_CASE(inputs_refuse_fifo_without_blocking), RT_CASE(atomic_write_does_not_follow_tmp_symlink),
};
const size_t rt_suite_safetensors_n = sizeof rt_suite_safetensors / sizeof rt_suite_safetensors[0];
