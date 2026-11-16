/* rrc/data.h — tokenized SFT data (.rrcbin), deterministic shuffling + packing, prefetch thread
 * (rrc/docs/DESIGN.md §6, §10).
 *
 * .rrcbin format (little-endian, no padding, no alignment), written by rrc/tools/prepare_sft.py:
 *
 *   offset  size  field
 *   0       8     magic, the 8 ASCII bytes "RRCBIN01"
 *   8       8     u64 num_docs      (>= 1)
 *   16      8     u64 vocab_size    (tokenizer vocabulary, 1 .. 2^31 - 1; every token is < it)
 *   24      8     u64 total_tokens  (sum of all document lengths)
 *   32      ...   num_docs documents, back to back, each:
 *                   u32 len                  (>= 1)
 *                   u32 tokens[len]
 *                   u8  label_mask[len]      (0 or 1; 1 = this token is a training target)
 *
 * The file size must be exactly 32 + sum over documents of (4 + 5 * len). label_mask marks the
 * tokens whose (unshifted, HF-style) label is the token itself; every other label is -100. The
 * engine shifts labels (DESIGN.md §10 item 9): token t is predicted from position t - 1 of the same
 * document, so the first token of a document is never a target whatever its mask.
 *
 * Every field is validated when the file is opened (rrc_dataset_open): short / truncated file,
 * bad magic, num_docs == 0 or inconsistent with the file size, vocab_size out of range or above
 * the model's vocabulary, a document with len == 0 or running past the end of the file, a token
 * >= vocab_size or >= the model vocabulary, a mask byte other than 0/1, total_tokens != the sum of
 * lengths, trailing bytes. All are RRC_ERR_PARSE (RRC_ERR_RANGE for token ids, RRC_ERR_IO for
 * open/read failures) with the document index and byte offset in the message.
 *
 * Packing (rrc_loader): an epoch visits the usable documents (all documents; with
 * RRC_TRUNC_DROP the documents longer than seq_len are excluded) in an order that is the identity
 * (shuffle == 0) or the seeded permutation of rrc_data_permutation(seed, epoch). Epochs are
 * concatenated into one document stream. A micro-batch is `rows` rows of `seq_len` tokens filled
 * greedily in stream order: the next document (truncated to its first seq_len tokens with
 * RRC_TRUNC_TRUNCATE) is appended to the current row if it fits in the remaining space; otherwise
 * the row is closed and the document starts the next row. A closed row's remaining positions are
 * padding: token pad_token_id, label -100, one extra document id. Within a row, document k has
 * doc id k and position ids 0, 1, ...; the padding document has the next id and positions 0, 1,
 * .... Rows never share a document, so the packing of a micro-batch depends only on the cursor
 * (epoch, index of the next document in that epoch's order) at its start.
 *
 * Determinism: the micro-batch sequence is a pure function of (dataset, options, start cursor);
 * the prefetch thread only changes when a micro-batch is built, not what it contains (tested:
 * prefetch on / off and repeated runs give identical batches).
 */
#ifndef RRC_DATA_H
#define RRC_DATA_H

#include <stdint.h>

#include "rrc/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RRC_RRCBIN_MAGIC "RRCBIN01"
#define RRC_RRCBIN_HEADER_BYTES 32

/* ---- dataset ------------------------------------------------------------------------------ */

typedef struct rrc_dataset rrc_dataset;

/* Open and fully validate a .rrcbin file (memory-mapped, read-only). model_vocab > 0 additionally
 * requires vocab_size <= model_vocab (and therefore every token < model_vocab); model_vocab <= 0
 * checks against the header's vocab_size only. */
rrc_status rrc_dataset_open(const char *path, int64_t model_vocab, rrc_dataset **out);
void rrc_dataset_free(rrc_dataset *d);

int64_t rrc_dataset_num_docs(const rrc_dataset *d);
int64_t rrc_dataset_vocab_size(const rrc_dataset *d);
int64_t rrc_dataset_total_tokens(const rrc_dataset *d);
int64_t rrc_dataset_label_tokens(const rrc_dataset *d); /* number of mask bytes == 1 */
int64_t rrc_dataset_max_len(const rrc_dataset *d);
/* FNV-1a 64 of the whole file (identifies the data in rrc_state.bin). */
uint64_t rrc_dataset_hash(const rrc_dataset *d);
/* Length of document i (0 <= i < num_docs). */
int64_t rrc_dataset_doc_len(const rrc_dataset *d, int64_t i);
/* Copy document i: tokens (as int32) and mask into caller arrays of at least n elements; copies
 * min(n, len) elements. */
rrc_status rrc_dataset_doc_copy(const rrc_dataset *d, int64_t i, int64_t n, int32_t *tokens, uint8_t *mask);

/* ---- shuffling ------------------------------------------------------------------------------ */

/* splitmix64 step (state += golden; mix), the generator used for shuffling. */
uint64_t rrc_splitmix64_next(uint64_t *state);
/* The permutation of [0, n) for (seed, epoch): Fisher-Yates from the back,
 *   state = splitmix64_next(&s0) with s0 = seed, then state ^= 0xD1B54A32D192ED03 * (epoch + 1);
 *   perm = identity; for i = n-1 down to 1: j = splitmix64_next(&state) % (i + 1); swap(perm[i], perm[j]).
 * (modulo bias <= n / 2^64). perm must hold n elements. */
void rrc_data_permutation(uint64_t seed, int64_t epoch, int64_t n, int64_t *perm);

/* ---- loader --------------------------------------------------------------------------------- */

typedef enum rrc_truncation {
    RRC_TRUNC_TRUNCATE = 0, /* keep the first seq_len tokens of a longer document */
    RRC_TRUNC_DROP = 1      /* skip documents longer than seq_len */
} rrc_truncation;

typedef struct rrc_loader_opts {
    int64_t seq_len;        /* tokens per row (>= 1) */
    int64_t rows;           /* rows per micro-batch (>= 1) */
    uint64_t seed;          /* shuffle seed */
    int shuffle;            /* 0: identity order every epoch */
    rrc_truncation truncation;
    int32_t pad_token_id;   /* token id of padding positions (must be < vocab_size) */
    int64_t max_held;       /* micro-batches the consumer holds at once (the trainer: grad_accum) */
    int prefetch;           /* 1: a producer thread builds up to max_held micro-batches ahead */
} rrc_loader_opts;

/* Position in the document stream: the next document to place is entry `index` of epoch
 * `epoch`'s order (0 <= index <= usable docs; index == usable docs rolls over to the next epoch). */
typedef struct rrc_data_cursor {
    int64_t epoch;
    int64_t index;
    int64_t micro_batches; /* micro-batches produced before this point (informational) */
} rrc_data_cursor;

typedef struct rrc_microbatch {
    int64_t rows, seq_len;
    int32_t *tokens, *labels, *position_ids, *doc_ids; /* rows * seq_len each */
    int64_t n_docs;          /* documents placed (padding not counted) */
    int64_t n_input_tokens;  /* non-padding positions */
    int64_t n_truncated;     /* documents that were cut to seq_len */
    rrc_data_cursor start;   /* cursor before this micro-batch */
    rrc_data_cursor end;     /* cursor after it (what a checkpoint taken after it stores) */
} rrc_microbatch;

typedef struct rrc_loader rrc_loader;

/* opts->max_held >= 1. The loader keeps a pointer to `d` (which must outlive it). With prefetch a
 * producer thread is started; it reads only host memory. */
rrc_status rrc_loader_create(const rrc_dataset *d, const rrc_loader_opts *opts, const rrc_data_cursor *start,
                             rrc_loader **out);
/* Stops and joins the producer thread. */
void rrc_loader_free(rrc_loader *l);
/* Number of usable documents per epoch. */
int64_t rrc_loader_usable_docs(const rrc_loader *l);
/* Next micro-batch in stream order. The returned batch stays valid until rrc_loader_release;
 * at most max_held may be held at once (one more is RRC_ERR_INVALID_ARG). A producer-thread error
 * is re-raised here with its message. */
rrc_status rrc_loader_next(rrc_loader *l, const rrc_microbatch **out);
/* Give a held micro-batch back (any order). */
rrc_status rrc_loader_release(rrc_loader *l, const rrc_microbatch *mb);

#ifdef __cplusplus
}
#endif

#endif /* RRC_DATA_H */
