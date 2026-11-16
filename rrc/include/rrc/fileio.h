/* rrc/fileio.h — file opening helpers for untrusted paths.
 *
 * Inputs (config.json, safetensors, .rrcbin, rrc_state.bin, training configs) are opened with
 * rrc_open_regular / rrc_fopen_regular: O_NONBLOCK during the open, so a FIFO or a device named like
 * an input cannot block the program, and anything but a regular file is refused (RRC_ERR_IO,
 * "<path> is not a regular file"; a missing file is RRC_ERR_NOT_FOUND).
 *
 * Outputs inside a run's directory are created with rrc_create_new_file (temporary files: a stale
 * entry, including a planted symlink, is unlinked and the file is created with O_EXCL | O_NOFOLLOW,
 * so the write never goes through a symlink) and rrc_fopen_output_nofollow (metrics.jsonl: refuses a
 * symlink as the last path component and anything but a regular file).
 */
#ifndef RRC_FILEIO_H
#define RRC_FILEIO_H

#include <stdio.h>

#include "rrc/common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Read-only file descriptor of the regular file `path` (O_CLOEXEC, blocking mode restored). */
rrc_status rrc_open_regular(const char *path, int *fd_out);
/* The same as a FILE * opened "rb". */
rrc_status rrc_fopen_regular(const char *path, FILE **out);
/* A new file at `path` for writing ("wb"): an existing non-directory entry at `path` is removed first
 * (unlink does not follow a symlink), then created with O_CREAT | O_EXCL | O_NOFOLLOW, mode 0644. */
rrc_status rrc_create_new_file(const char *path, FILE **out);
/* `path` opened for writing, truncated (append = 0) or appended to (append = 1), created if absent;
 * refused when its last component is a symlink or it is not a regular file. */
rrc_status rrc_fopen_output_nofollow(const char *path, int append, FILE **out);

#ifdef __cplusplus
}
#endif

#endif
