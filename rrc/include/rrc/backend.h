/* rrc/backend.h — THE op table (frozen interface, rrc/docs/DESIGN.md §4).
 *
 * A backend is a `const rrc_backend *`: a name, device-memory functions and one function pointer
 * per op. Model and trainer code call the `rrc_<op>()` wrappers declared below (src/ops.c), which
 * check every precondition on the host (null pointers, backend ownership, dtypes, shapes, extents
 * against rrc_buf.numel, aliasing, the int32 limits of gemm) and then dispatch to `be-><op>`.
 * Backends may assume that all of these preconditions hold; data-dependent conditions (token ids, labels and position ids in
 * range) are checked by the CPU backend and are documented preconditions for device backends
 * (which report RRC_ERR_RANGE from the op or from the next sync). On device backends, the outputs
 * of the ops enqueued after an op that hit such a data-dependent range error are unspecified until
 * the next sync reports the error.
 *
 * Execution model: there are no stream or handle arguments. Each backend owns one in-order queue
 * (the CUDA backend: one stream, one cuBLASLt handle and workspace). Ops and memcpy_d2d/memset0
 * are enqueued in call order; memcpy_d2h and sync wait for all previously enqueued work;
 * memcpy_h2d returns once the host source may be reused. All calls on one backend must come from
 * one host thread at a time.
 *
 * Conventions:
 *   - Row-major everywhere. A "rows x cols matrix with leading dimension ld" is the element
 *     (r, c) at offset r*ld + c (ld >= cols). HF nn.Linear weights W are [out, in] row-major.
 *   - Shapes and leading dimensions are int64 element counts. Zero-sized ops (no output element)
 *     are valid no-ops and are not dispatched. gemm with k == 0 but m, n > 0 is not zero-sized: it
 *     computes C = beta * C and is resolved in ops.c (memset0 / scale), so a backend's gemm never
 *     sees k == 0.
 *   - Operand pointers carry no alignment guarantee beyond the element size: a view may start at
 *     any element (e.g. per-head offsets h * head_dim). A device backend that needs alignment
 *     (cuBLASLt's PREF_MIN_ALIGNMENT_*_BYTES) must derive it from the actual pointers and leading
 *     dimensions on every call.
 *   - Ops may use internal scratch memory (CPU gemm: 2 * 32 * k floats per thread for packing;
 *     the CUDA backend: its cuBLASLt workspace) and can then fail with RRC_ERR_OOM.
 *   - Float math is fp32; inputs are widened from bf16 exactly; results are rounded to the output
 *     dtype once (round-to-nearest-even). Reductions accumulate in fp32 on device backends; the
 *     CPU reference backend accumulates in double (rrc_backend_cpu) or in fp32
 *     (rrc_backend_cpu_f32acc), always in a fixed order, so results do not depend on the number
 *     of threads.
 *   - Aliasing: "exact alias" means same ptr, same dtype and same leading dimension. Where an op
 *     allows an output to exactly alias an input it is stated below; any other overlap between an
 *     output and any operand, or between two outputs, is RRC_ERR_ALIAS. For the matrix operands of
 *     gemm and swiglu_fwd/bwd, overlap is decided per element, not by address range: two operands
 *     with the same row pitch in bytes (ld * element size) whose column ranges are disjoint (e.g.
 *     the [gate | up] column blocks of one fused [rows, 2*cols] buffer) do not overlap. A block
 *     whose columns run past the end of the pitch, or operands with different pitches, fall back
 *     to the address-range test.
 */
#ifndef RRC_BACKEND_H
#define RRC_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#include "rrc/common.h"
#include "rrc/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RRC_IGNORE_INDEX (-100)

typedef enum rrc_trans { RRC_OP_N = 0, RRC_OP_T = 1 } rrc_trans;

/* C[b] = alpha * op(A[b]) * op(B[b]) + beta * C[b], for b in [0, batch).
 *   op(A) is m x k: trans_a == N -> A stored m x k (lda >= k); T -> A stored k x m (lda >= m).
 *   op(B) is k x n: trans_b == N -> B stored k x n (ldb >= n); T -> B stored n x k (ldb >= k).
 *   C is m x n (ldc >= n). A[b] = a.ptr + b*stride_a (elements), likewise B and C.
 *   Limits: m, n, k and batch must each be <= INT32_MAX (cuBLASLt's BATCH_COUNT is int32 and its
 *   ROWS/COLS are int32 in practice); leading dimensions and strides are int64.
 *   k == 0: C = beta * C (zeroed when beta == 0); A and B are not accessed and may be
 *   rrc_buf_none(); resolved in ops.c, never dispatched to the backend.
 *   stride_a / stride_b may be 0 (broadcast one matrix over the batch); C slices must not overlap
 *   each other (stride_c >= extent of one C slice, or C slices interleaved within rows with
 *   n <= stride_c and (batch-1)*stride_c + n <= ldc).
 *   dtypes: a, b, c each RRC_F32 or RRC_BF16 independently; fp32 accumulation. beta == 0 means C
 *   is not read (NaN/Inf in C are overwritten).
 *   Aliasing: C must not overlap A or B (per element for column blocks with the same pitch, see
 *   the conventions above), so C may be one column block of a buffer whose other block is A or B.
 *   Patterns (DESIGN.md §3): GQA dK_g = sum_u dS_(g*G+u)^T Q_(g*G+u) is one call per in-group head
 *   u, batched over the KV groups g with beta = 1 (C interleaved: stride_c = head_dim); a single
 *   call summing over u (stride_c = 0) is refused because its C slices overlap. */
typedef struct rrc_gemm_args {
    rrc_trans trans_a, trans_b;
    int64_t m, n, k;
    float alpha, beta;
    rrc_buf a;
    int64_t lda, stride_a;
    rrc_buf b;
    int64_t ldb, stride_b;
    rrc_buf c;
    int64_t ldc, stride_c;
    int64_t batch; /* >= 1 */
} rrc_gemm_args;

/* torch.optim.AdamW hyper-parameters for one step (foreach=False semantics, see adamw_step). */
typedef struct rrc_adamw_hparams {
    double lr, beta1, beta2, eps, weight_decay;
    int64_t step;      /* step count after increment: 1 for the first update */
    float grad_scale;  /* g = grad * grad_scale before the update (1, or the clip coefficient) */
} rrc_adamw_hparams;

struct rrc_backend {
    const char *name;
    int host_accessible; /* device pointers are host pointers (CPU backend) */
    void *impl;          /* backend-private state */

    /* ---- memory (bytes) ------------------------------------------------------------------ */
    rrc_status (*alloc)(const rrc_backend *be, size_t bytes, void **out); /* bytes == 0 -> NULL */
    void (*free)(const rrc_backend *be, void *ptr);
    rrc_status (*memcpy_h2d)(const rrc_backend *be, void *dst, const void *src, size_t bytes);
    rrc_status (*memcpy_d2h)(const rrc_backend *be, void *dst, const void *src, size_t bytes);
    rrc_status (*memcpy_d2d)(const rrc_backend *be, void *dst, const void *src, size_t bytes);
    rrc_status (*memset0)(const rrc_backend *be, void *ptr, size_t bytes);
    rrc_status (*sync)(const rrc_backend *be);

    /* ---- ops: same arguments as the rrc_<op> wrappers below, already validated by them ------ */
    rrc_status (*gemm)(const rrc_backend *be, const rrc_gemm_args *g);
    rrc_status (*embedding_fwd)(const rrc_backend *be, int64_t n_tokens, int64_t dim, int64_t vocab, rrc_buf ids,
                                rrc_buf table, rrc_buf out);
    rrc_status (*add_bias)(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf x, int64_t ldx, rrc_buf bias);
    rrc_status (*bias_grad)(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf dy, int64_t ldy, rrc_buf dbias,
                            int accumulate);
    rrc_status (*rmsnorm_fwd)(const rrc_backend *be, int64_t rows, int64_t dim, int64_t ld, rrc_buf x, rrc_buf w,
                              float eps, rrc_buf y);
    rrc_status (*rmsnorm_bwd)(const rrc_backend *be, int64_t rows, int64_t dim, int64_t ld, rrc_buf x, rrc_buf w,
                              float eps, rrc_buf dy, rrc_buf dx, int accumulate);
    rrc_status (*rope_fwd)(const rrc_backend *be, int64_t n_tokens, int64_t n_heads, int64_t head_dim, int64_t ld,
                           rrc_buf x, rrc_buf pos, rrc_buf cos_tab, rrc_buf sin_tab, int64_t n_pos, rrc_buf y);
    rrc_status (*rope_bwd)(const rrc_backend *be, int64_t n_tokens, int64_t n_heads, int64_t head_dim, int64_t ld,
                           rrc_buf dy, rrc_buf pos, rrc_buf cos_tab, rrc_buf sin_tab, int64_t n_pos, rrc_buf dx);
    rrc_status (*attn_softmax_fwd)(const rrc_backend *be, int64_t batch, int64_t rows, int64_t cols, int64_t ld,
                                   int64_t batch_stride, rrc_buf s, float scale, int64_t row_offset, int causal,
                                   rrc_buf doc_ids, rrc_buf p);
    rrc_status (*attn_softmax_bwd)(const rrc_backend *be, int64_t batch, int64_t rows, int64_t cols, int64_t ld,
                                   int64_t batch_stride, rrc_buf p, rrc_buf dp, float scale, rrc_buf ds);
    rrc_status (*swiglu_fwd)(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf gate, rrc_buf up,
                             int64_t ld_in, rrc_buf h, int64_t ld_out);
    rrc_status (*swiglu_bwd)(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf gate, rrc_buf up,
                             int64_t ld_in, rrc_buf dh, int64_t ld_dh, rrc_buf dgate, rrc_buf dup, int64_t ld_out);
    rrc_status (*add)(const rrc_backend *be, int64_t n, rrc_buf a, rrc_buf b, rrc_buf out);
    rrc_status (*axpy)(const rrc_backend *be, int64_t n, float alpha, rrc_buf x, rrc_buf y);
    rrc_status (*scale)(const rrc_backend *be, int64_t n, float alpha, rrc_buf x);
    rrc_status (*convert)(const rrc_backend *be, int64_t n, rrc_buf src, rrc_buf dst);
    rrc_status (*sum_squares)(const rrc_backend *be, int64_t n, rrc_buf x, rrc_buf out, int accumulate);
    rrc_status (*cross_entropy_fwd_bwd)(const rrc_backend *be, int64_t rows, int64_t vocab, rrc_buf logits,
                                        int64_t ld, rrc_buf labels, rrc_buf loss, float grad_scale,
                                        rrc_buf row_scale, int write_grad);
    rrc_status (*adamw_step)(const rrc_backend *be, int64_t n, rrc_buf param, rrc_buf grad, rrc_buf m, rrc_buf v,
                             const rrc_adamw_hparams *h);
};

/* ---- backends --------------------------------------------------------------------------------- */

/* CPU reference backend (OpenMP; double accumulation for reductions). Static, never freed. */
const rrc_backend *rrc_backend_cpu(void);
/* Same backend with fp32 accumulation (the arithmetic a device backend uses), for tests. */
const rrc_backend *rrc_backend_cpu_f32acc(void);

/* ---- memory wrappers ------------------------------------------------------------------------- */
rrc_status rrc_sync(const rrc_backend *be);

/* ---- op wrappers (host-side checks + dispatch) ----------------------------------------------- */

/* See rrc_gemm_args. */
rrc_status rrc_gemm(const rrc_backend *be, const rrc_gemm_args *g);
/* Convenience: batch 1. */
rrc_status rrc_gemm2d(const rrc_backend *be, rrc_trans ta, rrc_trans tb, int64_t m, int64_t n, int64_t k, float alpha,
                      rrc_buf a, int64_t lda, rrc_buf b, int64_t ldb, float beta, rrc_buf c, int64_t ldc);

/* out[t, :] = table[ids[t], :] for t < n_tokens. ids: I32 [n_tokens], each in [0, vocab)
 * (RRC_ERR_RANGE otherwise). table: [vocab, dim] f32|bf16; out: [n_tokens, dim] f32|bf16
 * (converted). out must not overlap table or ids. Also the dense row gather / scatter of the LM
 * head (DESIGN.md §5): table = hidden states [N, dim] with ids = selected rows, and back with
 * table = [n_sel + 1, dim] (last row zero) and an inverse index. */
rrc_status rrc_embedding_fwd(const rrc_backend *be, int64_t n_tokens, int64_t dim, int64_t vocab, rrc_buf ids,
                             rrc_buf table, rrc_buf out);

/* x[r, c] += bias[c] in place. x: rows x cols (ldx) f32|bf16; bias: [cols] f32|bf16 (not
 * overlapping x). */
rrc_status rrc_add_bias(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf x, int64_t ldx, rrc_buf bias);

/* dbias[c] (accumulate ? += : =) sum_r dy[r, c]. dy: rows x cols (ldy) f32|bf16; dbias: f32 [cols]
 * (not overlapping dy). Phase 1 never trains biases; provided so the table stays frozen. */
rrc_status rrc_bias_grad(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf dy, int64_t ldy, rrc_buf dbias,
                         int accumulate);

/* RMSNorm over the last dim: y[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * w, fp32 math.
 * x, y: rows x dim (ld) f32|bf16; w: [dim] f32|bf16. Per-head q/k norm (Qwen3) is the same op on
 * the [n_tokens * n_heads, head_dim] view. y may exactly alias x (in place). */
rrc_status rrc_rmsnorm_fwd(const rrc_backend *be, int64_t rows, int64_t dim, int64_t ld, rrc_buf x, rrc_buf w,
                           float eps, rrc_buf y);
/* Input gradient only (w is frozen): g = d rmsnorm(x) / dx applied to dy;
 * dx (accumulate ? += : =) g. x, dy, dx: rows x dim (ld). With accumulate == 0, dx may exactly
 * alias dy or x; with accumulate == 1, dx must not overlap x or dy. */
rrc_status rrc_rmsnorm_bwd(const rrc_backend *be, int64_t rows, int64_t dim, int64_t ld, rrc_buf x, rrc_buf w,
                           float eps, rrc_buf dy, rrc_buf dx, int accumulate);

/* Rotary embedding (HF rotate_half convention) on x viewed as [n_tokens, n_heads, head_dim] where
 * token t starts at t*ld (ld >= n_heads*head_dim) and heads are contiguous within a token:
 *   for i < head_dim/2, with c = cos_tab[pos[t], i], s = sin_tab[pos[t], i]:
 *     y[i]      = x[i] * c - x[i + h/2] * s
 *     y[i + h/2] = x[i + h/2] * c + x[i] * s
 * pos: I32 [n_tokens], each in [0, n_pos) (RRC_ERR_RANGE otherwise); cos_tab/sin_tab: f32
 * [n_pos, head_dim/2] (rrc_rope_tables). x, y f32|bf16 with the same ld; y may exactly alias x.
 * rope_bwd applies the transposed rotation (dx = R^T dy) with the same argument rules. */
rrc_status rrc_rope_fwd(const rrc_backend *be, int64_t n_tokens, int64_t n_heads, int64_t head_dim, int64_t ld,
                        rrc_buf x, rrc_buf pos, rrc_buf cos_tab, rrc_buf sin_tab, int64_t n_pos, rrc_buf y);
rrc_status rrc_rope_bwd(const rrc_backend *be, int64_t n_tokens, int64_t n_heads, int64_t head_dim, int64_t ld,
                        rrc_buf dy, rrc_buf pos, rrc_buf cos_tab, rrc_buf sin_tab, int64_t n_pos, rrc_buf dx);

/* Masked softmax over attention scores for a chunk of query rows. s and p are batch x rows x cols
 * (row stride ld >= cols, batch stride batch_stride >= ld*rows unless batch == 1). Local row r
 * is query position q = row_offset + r; key j in [0, cols) is allowed iff
 * (!causal || j <= q) && (doc_ids is none || doc_ids[j] == doc_ids[q]).
 *   p[b, r, j] = exp(scale*s[b,r,j] - max) / sum over allowed j, and 0 for disallowed keys;
 *   a row with no allowed key is all zeros.
 * s: f32|bf16; p: f32|bf16; p may exactly alias s. doc_ids: I32 with at least
 * max(row_offset + rows, cols) elements, or rrc_buf_none(). */
rrc_status rrc_attn_softmax_fwd(const rrc_backend *be, int64_t batch, int64_t rows, int64_t cols, int64_t ld,
                                int64_t batch_stride, rrc_buf s, float scale, int64_t row_offset, int causal,
                                rrc_buf doc_ids, rrc_buf p);
/* ds = scale * p * (dp - sum_j p*dp) (gradient with respect to the unscaled scores s). p, dp, ds
 * have the layout of attn_softmax_fwd; f32|bf16 each; ds may exactly alias dp or p. Masked
 * entries have p = 0 and therefore ds = 0. */
rrc_status rrc_attn_softmax_bwd(const rrc_backend *be, int64_t batch, int64_t rows, int64_t cols, int64_t ld,
                                int64_t batch_stride, rrc_buf p, rrc_buf dp, float scale, rrc_buf ds);

/* h = silu(gate) * up, elementwise over rows x cols. gate and up share ld_in (they may be two
 * column blocks of one fused buffer); h has ld_out. h may exactly alias gate or up (with a fused
 * [gate | up] buffer, h = gate writes the gate block in place and leaves the up block intact). */
rrc_status rrc_swiglu_fwd(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf gate, rrc_buf up, int64_t ld_in,
                          rrc_buf h, int64_t ld_out);
/* dgate = dh * up * silu'(gate), dup = dh * silu(gate). gate/up: ld_in; dh: ld_dh; dgate/dup:
 * ld_out. dgate may exactly alias gate or dh; dup may exactly alias up or dh; dgate and dup must
 * not overlap each other. Column blocks of one buffer count as disjoint (conventions above), so
 * with a fused [gate | up] input the outputs may be the two blocks of a fused [dgate | dup] buffer
 * (ld_out = 2*cols, the A operand of one backward GEMM with k = 2*cols) or the input blocks
 * themselves (dgate = gate, dup = up: in place). */
rrc_status rrc_swiglu_bwd(const rrc_backend *be, int64_t rows, int64_t cols, rrc_buf gate, rrc_buf up, int64_t ld_in,
                          rrc_buf dh, int64_t ld_dh, rrc_buf dgate, rrc_buf dup, int64_t ld_out);

/* out = a + b (n elements, f32|bf16 each); out may exactly alias a or b. */
rrc_status rrc_add(const rrc_backend *be, int64_t n, rrc_buf a, rrc_buf b, rrc_buf out);
/* y += alpha * x (f32|bf16 each); y may exactly alias x. */
rrc_status rrc_axpy(const rrc_backend *be, int64_t n, float alpha, rrc_buf x, rrc_buf y);
/* x *= alpha in place (f32|bf16). */
rrc_status rrc_scale(const rrc_backend *be, int64_t n, float alpha, rrc_buf x);
/* dst = src converted (f32 <-> bf16 with RNE, or a copy for equal float dtypes; I32 -> I32 copy).
 * src and dst must not overlap (an exact alias with equal dtype is a no-op). */
rrc_status rrc_convert(const rrc_backend *be, int64_t n, rrc_buf src, rrc_buf dst);
/* out[0] (accumulate ? += : =) sum_i x[i]^2. x: f32|bf16; out: f32 with >= 1 element, not
 * overlapping x. Deterministic. */
rrc_status rrc_sum_squares(const rrc_backend *be, int64_t n, rrc_buf x, rrc_buf out, int accumulate);

/* Fused causal-LM cross entropy over a chunk of rows. logits: rows x vocab (ld) f32|bf16;
 * -inf logits are allowed (they contribute exp(-inf) = 0; a row whose logits are all -inf yields
 * NaN, as in torch);
 * labels: I32 [rows], each in [0, vocab) or RRC_IGNORE_INDEX (RRC_ERR_RANGE otherwise);
 * loss: f32 [rows] receives (max(z) - z[label]) + log(sum exp(z - max)) = logsumexp(z) - z[label]
 * (0 for ignored rows). If write_grad, logits
 * are overwritten in place with (softmax(z) - onehot(label)) * grad_scale * row_scale[r]
 * (row_scale: f32 [rows] or rrc_buf_none() for 1); ignored rows become 0. loss must not overlap
 * logits/labels/row_scale. */
rrc_status rrc_cross_entropy_fwd_bwd(const rrc_backend *be, int64_t rows, int64_t vocab, rrc_buf logits, int64_t ld,
                                     rrc_buf labels, rrc_buf loss, float grad_scale, rrc_buf row_scale,
                                     int write_grad);

/* Fused AdamW update of n fp32 parameters (param, m, v updated in place; grad read-only), in the
 * operation order of torch.optim.adamw (_single_tensor_adam, decoupled weight decay) with the
 * element rounding of ATen's CPU kernels (fma = fused multiply-add, one rounding):
 *   g = grad * grad_scale
 *   if weight_decay != 0: p = p * (float)(1 - lr*weight_decay)
 *   m = lerp(m, g, w = (float)(1 - beta1))  = |w| < 0.5 ? fma(w, g - m, m) : fma(w - 1, g - m, g)
 *   v = fma((float)(1 - beta2) * g, g, v * (float)beta2)
 *   denom = sqrt(v) / (float)((1 - beta2^step) ** 0.5) + (float)eps
 *   p = p + ((float)(-lr / (1 - beta1^step)) * m) / denom
 * with the scalars computed in double and rounded to f32 as shown. All four buffers f32 and
 * mutually non-overlapping; step >= 1. */
rrc_status rrc_adamw_step(const rrc_backend *be, int64_t n, rrc_buf param, rrc_buf grad, rrc_buf m, rrc_buf v,
                          const rrc_adamw_hparams *h);

#ifdef __cplusplus
}
#endif

#endif /* RRC_BACKEND_H */
