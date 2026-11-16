/* rrc/safetensors.h — read (mmap) and write safetensors files.
 *
 * Format: 8-byte little-endian header length N, N bytes of JSON header, then the data block.
 * The header maps tensor names to {"dtype", "shape", "data_offsets": [begin, end]} (offsets relative
 * to the data block) plus an optional "__metadata__" object of string -> string.
 *
 * Validation (all RRC_ERR_PARSE unless noted; mirrors the reference `safetensors` package, which
 * rejects the same files):
 *   - file shorter than 8 bytes, N == 0, N > 100 MB, or 8 + N > file size;
 *   - header that is not a strict-JSON object, an entry that is not an object with exactly the
 *     keys dtype/shape/data_offsets, non-string metadata values;
 *   - unknown dtype string (RRC_ERR_UNSUPPORTED for known-but-unsupported safetensors dtypes such
 *     as F64, U8, BOOL, F8_*), more than RRC_ST_MAX_DIMS dims, negative dims, numel overflow;
 *   - end < begin, end - begin != numel * element size, end beyond the data block;
 *   - overlapping tensors ("overlap"), gaps between tensors or a data block that is not exactly
 *     covered ("hole" / "not fully covered").
 *
 * Supported dtypes: F32, BF16, F16, I32, I64. F16 is read and converted exactly to f32
 * (rrc_st_to_f32); the engine itself has no f16 dtype, so rrc_st_dtype_to_rrc() refuses F16 and
 * I64 (RRC_ERR_UNSUPPORTED) and callers must convert explicitly (F16 -> f32 exact; f32 -> bf16 RNE).
 *
 * Sharded checkpoints: a directory containing model.safetensors.index.json ({"weight_map":
 * {tensor: shard file}}) is opened as one set; every weight_map entry must exist in its shard,
 * every tensor of every shard must be listed for that shard, shard names must be plain file names
 * (no '/', no "..") and tensor names must be unique across shards.
 */
#ifndef RRC_SAFETENSORS_H
#define RRC_SAFETENSORS_H

#include <stdint.h>

#include "rrc/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RRC_ST_MAX_DIMS 8
#define RRC_ST_MAX_HEADER (100u * 1000u * 1000u)

typedef enum rrc_st_dtype {
    RRC_ST_F32 = 0,
    RRC_ST_BF16,
    RRC_ST_F16,
    RRC_ST_I32,
    RRC_ST_I64,
    RRC_ST_DTYPE_COUNT_
} rrc_st_dtype;

const char *rrc_st_dtype_name(rrc_st_dtype dt); /* "F32", "BF16", ... */
size_t rrc_st_dtype_size(rrc_st_dtype dt);
/* F32/BF16/I32 map to the engine dtypes; F16 and I64 -> RRC_ERR_UNSUPPORTED. */
rrc_status rrc_st_dtype_to_rrc(rrc_st_dtype dt, rrc_dtype *out);

typedef struct rrc_st_tensor {
    const char *name;
    rrc_st_dtype dtype;
    int ndim;
    int64_t shape[RRC_ST_MAX_DIMS];
    int64_t numel;     /* product of shape (1 for a 0-d tensor) */
    const void *data;  /* points into the read-only mapping; may be unaligned */
    uint64_t nbytes;
    int shard;         /* index of the file the tensor lives in */
} rrc_st_tensor;

typedef struct rrc_st_set rrc_st_set;

/* Open one .safetensors file. */
rrc_status rrc_st_open_file(const char *path, rrc_st_set **out);
/* Open a checkpoint: a directory (model.safetensors.index.json if present, else
 * model.safetensors), an *.index.json file, or a single .safetensors file. */
rrc_status rrc_st_open_checkpoint(const char *path, rrc_st_set **out);
void rrc_st_close(rrc_st_set *set);

size_t rrc_st_count(const rrc_st_set *set);
/* Tensors sorted by name (strcmp order). */
const rrc_st_tensor *rrc_st_at(const rrc_st_set *set, size_t i);
const rrc_st_tensor *rrc_st_find(const rrc_st_set *set, const char *name);
size_t rrc_st_num_shards(const rrc_st_set *set);
/* "__metadata__" value of the given shard (NULL if absent). */
const char *rrc_st_metadata(const rrc_st_set *set, size_t shard, const char *key);

/* Copy/convert a tensor's data into a host buffer of exactly t->numel elements.
 *   to_f32:  F32 copy, BF16 / F16 exact widening; integer dtypes -> RRC_ERR_DTYPE.
 *   to_bf16: BF16 copy, F32 RNE rounding, F16 -> f32 -> bf16 (RNE); integers -> RRC_ERR_DTYPE.
 *   to_i32:  I32 copy, I64 with range check (RRC_ERR_RANGE); floats -> RRC_ERR_DTYPE. */
rrc_status rrc_st_to_f32(const rrc_st_tensor *t, float *dst, int64_t n);
rrc_status rrc_st_to_bf16(const rrc_st_tensor *t, rrc_bf16 *dst, int64_t n);
rrc_status rrc_st_to_i32(const rrc_st_tensor *t, int32_t *dst, int64_t n);

/* Exact IEEE half -> float (denormals, inf, NaN payload kept). */
float rrc_f16_to_f32(uint16_t h);

/* ------------------------------------------------------------------------------------------ */
/* Writer. Tensors are copied at add time. The file is written as path.tmp then renamed. Data is
 * laid out contiguously (no holes), ordered by element size (descending) then name, and the
 * header is padded with spaces so the data block starts at an 8-byte aligned offset (like the
 * reference implementation). */
/* ------------------------------------------------------------------------------------------ */

typedef struct rrc_st_writer rrc_st_writer;

rrc_status rrc_st_writer_create(rrc_st_writer **out);
/* dtype: RRC_F32, RRC_BF16 or RRC_I32. `data` holds prod(shape) elements of that dtype. */
rrc_status rrc_st_writer_add(rrc_st_writer *w, const char *name, rrc_dtype dtype, int ndim, const int64_t *shape,
                             const void *data);
rrc_status rrc_st_writer_set_metadata(rrc_st_writer *w, const char *key, const char *value);
rrc_status rrc_st_writer_write(rrc_st_writer *w, const char *path);
void rrc_st_writer_free(rrc_st_writer *w);

#ifdef __cplusplus
}
#endif

#endif /* RRC_SAFETENSORS_H */
