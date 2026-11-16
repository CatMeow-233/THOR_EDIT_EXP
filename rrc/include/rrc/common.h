/* rrc/common.h — status codes, error reporting, dtypes, bf16 <-> f32.
 *
 * This header is C11 and also compiles as C++ under nvcc: everything used from device code is a
 * `RRC_HD static inline` function (RRC_HD = __host__ __device__ under nvcc, empty otherwise).
 */
#ifndef RRC_COMMON_H
#define RRC_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__CUDACC__)
#define RRC_HD __host__ __device__
#else
#define RRC_HD
#endif

#if defined(__GNUC__)
#define RRC_PRINTF(fmt_idx, arg_idx) __attribute__((format(printf, fmt_idx, arg_idx)))
#else
#define RRC_PRINTF(fmt_idx, arg_idx)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------------------------ */
/* Status codes                                                                                */
/* ------------------------------------------------------------------------------------------ */

typedef enum rrc_status {
    RRC_OK = 0,
    RRC_ERR_INVALID_ARG = 1, /* null pointer, negative size, bad enum value, bad option      */
    RRC_ERR_SHAPE = 2,       /* shape / leading dimension / buffer extent mismatch          */
    RRC_ERR_DTYPE = 3,       /* dtype (combination) not allowed for this op                 */
    RRC_ERR_ALIAS = 4,       /* buffers overlap in a way the op does not allow              */
    RRC_ERR_RANGE = 5,       /* index out of range (token id, label, position id)           */
    RRC_ERR_OOM = 6,         /* host or device allocation failed                            */
    RRC_ERR_IO = 7,          /* open / read / write / mmap failed                           */
    RRC_ERR_PARSE = 8,       /* malformed JSON / safetensors / binary input                 */
    RRC_ERR_UNSUPPORTED = 9, /* well-formed input the engine does not implement (refused)   */
    RRC_ERR_NOT_FOUND = 10,  /* missing file, key or tensor                                 */
    RRC_ERR_BACKEND = 11,    /* backend / device failure                                    */
    RRC_ERR_INTERNAL = 12,   /* bug: invariant violated                                     */
    RRC_STATUS_COUNT_ = 13
} rrc_status;

/* Stable upper-case name of a status ("RRC_OK", "RRC_ERR_SHAPE", ...). Unknown -> "RRC_ERR_?". */
const char *rrc_status_str(rrc_status s);

/* ------------------------------------------------------------------------------------------ */
/* Error reporting: one thread-local "last error" per thread.                                  */
/* ------------------------------------------------------------------------------------------ */

#define RRC_ERROR_MSG_MAX 1024

/* Record an error for the calling thread and return `s` unchanged. The stored message is
 * "<file>:<line>: <STATUS>: <formatted message>". Passing RRC_OK clears the error. */
rrc_status rrc_set_error(rrc_status s, const char *file, int line, const char *fmt, ...)
    RRC_PRINTF(4, 5);

/* Prefix the current thread's message with context ("<file>:<line>: <ctx>: <old message>") and
 * return `s`. Used when propagating an error through several layers. */
rrc_status rrc_error_context(rrc_status s, const char *file, int line, const char *fmt, ...)
    RRC_PRINTF(4, 5);

/* Message of the most recent error on this thread ("" if none). Valid until the next error. */
const char *rrc_last_error(void);
/* Status of the most recent recorded error on this thread (RRC_OK if none / cleared). */
rrc_status rrc_last_status(void);
void rrc_clear_error(void);

#define RRC_ERROR(status, ...) rrc_set_error((status), __FILE__, __LINE__, __VA_ARGS__)

/* Return the status of `expr` from the enclosing function if it is not RRC_OK. */
#define RRC_CHECK(expr)                                                                          \
    do {                                                                                         \
        rrc_status rrc_check_s_ = (expr);                                                        \
        if (rrc_check_s_ != RRC_OK) return rrc_check_s_;                                         \
    } while (0)

/* Like RRC_CHECK but prefixes the error message with a context string. */
#define RRC_CHECK_CTX(expr, ...)                                                                 \
    do {                                                                                         \
        rrc_status rrc_check_s_ = (expr);                                                        \
        if (rrc_check_s_ != RRC_OK)                                                              \
            return rrc_error_context(rrc_check_s_, __FILE__, __LINE__, __VA_ARGS__);             \
    } while (0)

/* Store the status of `expr` in `st` and jump to `label` if it is not RRC_OK. */
#define RRC_CHECK_GOTO(st, expr, label)                                                          \
    do {                                                                                         \
        (st) = (expr);                                                                           \
        if ((st) != RRC_OK) goto label;                                                          \
    } while (0)

/* If `cond` is false: record an error with `status` and the printf-style message, and return. */
#define RRC_REQUIRE(cond, status, ...)                                                           \
    do {                                                                                         \
        if (!(cond)) return RRC_ERROR((status), __VA_ARGS__);                                    \
    } while (0)

/* ------------------------------------------------------------------------------------------ */
/* Logging (stderr). Level from env RRC_LOG_LEVEL = error|warn|info|debug (default warn).      */
/* ------------------------------------------------------------------------------------------ */

typedef enum rrc_log_level { RRC_LOG_ERROR = 0, RRC_LOG_WARN = 1, RRC_LOG_INFO = 2, RRC_LOG_DEBUG = 3 } rrc_log_level;
void rrc_log(rrc_log_level level, const char *fmt, ...) RRC_PRINTF(2, 3);

/* ------------------------------------------------------------------------------------------ */
/* Dtypes                                                                                      */
/* ------------------------------------------------------------------------------------------ */

typedef enum rrc_dtype { RRC_F32 = 0, RRC_BF16 = 1, RRC_I32 = 2, RRC_DTYPE_COUNT_ = 3 } rrc_dtype;

/* Element size in bytes; 0 for an invalid dtype. */
RRC_HD static inline size_t rrc_dtype_size(rrc_dtype dt) {
    switch (dt) {
    case RRC_F32: return 4;
    case RRC_BF16: return 2;
    case RRC_I32: return 4;
    default: return 0;
    }
}
RRC_HD static inline int rrc_dtype_valid(int dt) { return dt >= 0 && dt < (int)RRC_DTYPE_COUNT_; }
RRC_HD static inline int rrc_dtype_is_float(rrc_dtype dt) { return dt == RRC_F32 || dt == RRC_BF16; }
/* "f32", "bf16", "i32" or "?" */
const char *rrc_dtype_name(rrc_dtype dt);

/* ------------------------------------------------------------------------------------------ */
/* bf16 <-> f32 (exact semantics; host and device)                                             */
/* ------------------------------------------------------------------------------------------ */

typedef uint16_t rrc_bf16; /* raw bfloat16 bits */

RRC_HD static inline uint32_t rrc_f32_bits(float f) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(f);
#else
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
#endif
}

RRC_HD static inline float rrc_bits_f32(uint32_t u) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(u);
#else
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
#endif
}

/* bf16 -> f32 is exact: the bf16 bits are the upper 16 bits of the f32. */
RRC_HD static inline float rrc_bf16_to_f32(rrc_bf16 h) { return rrc_bits_f32((uint32_t)h << 16); }

/* f32 -> bf16 with round-to-nearest-even. Overflow rounds to +-Inf exactly like IEEE RNE (values
 * >= 0x1.FFp127 + half an ulp become Inf). NaN stays NaN: the result keeps the sign and the upper
 * 7 payload bits and has the quiet bit set, i.e. (bits >> 16) | 0x0040. (PyTorch's CPU path returns
 * the canonical 0x7FC0 for every NaN; both are NaN, only the payload differs.) Denormals are kept
 * (no flush to zero). */
RRC_HD static inline rrc_bf16 rrc_f32_to_bf16(float f) {
    uint32_t u = rrc_f32_bits(f);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (rrc_bf16)((u >> 16) | 0x0040u);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (rrc_bf16)(u >> 16);
}

/* Round an f32 value to the nearest bf16-representable value (result as f32). */
RRC_HD static inline float rrc_round_bf16(float f) { return rrc_bf16_to_f32(rrc_f32_to_bf16(f)); }

#ifdef __cplusplus
}
#endif

#endif /* RRC_COMMON_H */
