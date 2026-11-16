/* rrc/src/fileio.c — see include/rrc/fileio.h. */
#include "rrc/fileio.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

rrc_status rrc_open_regular(const char *path, int *fd_out) {
    RRC_REQUIRE(path && fd_out, RRC_ERR_INVALID_ARG, "null argument");
    *fd_out = -1;
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0)
        return RRC_ERROR(errno == ENOENT ? RRC_ERR_NOT_FOUND : RRC_ERR_IO, "cannot open %s: %s", path, strerror(errno));
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        close(fd);
        return RRC_ERROR(RRC_ERR_IO, "%s is not a regular file", path);
    }
    const int fl = fcntl(fd, F_GETFL);
    if (fl < 0 || fcntl(fd, F_SETFL, fl & ~O_NONBLOCK) != 0) {
        rrc_status s = RRC_ERROR(RRC_ERR_IO, "fcntl %s: %s", path, strerror(errno));
        close(fd);
        return s;
    }
    *fd_out = fd;
    return RRC_OK;
}

rrc_status rrc_fopen_regular(const char *path, FILE **out) {
    RRC_REQUIRE(out, RRC_ERR_INVALID_ARG, "null argument");
    *out = NULL;
    int fd = -1;
    RRC_CHECK(rrc_open_regular(path, &fd));
    FILE *f = fdopen(fd, "rb");
    if (!f) {
        rrc_status s = RRC_ERROR(RRC_ERR_IO, "fdopen %s: %s", path, strerror(errno));
        close(fd);
        return s;
    }
    *out = f;
    return RRC_OK;
}

rrc_status rrc_create_new_file(const char *path, FILE **out) {
    RRC_REQUIRE(path && out, RRC_ERR_INVALID_ARG, "null argument");
    *out = NULL;
    if (unlink(path) != 0 && errno != ENOENT)
        return RRC_ERROR(RRC_ERR_IO, "cannot remove the stale %s: %s", path, strerror(errno));
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0644);
    if (fd < 0) return RRC_ERROR(RRC_ERR_IO, "cannot create %s: %s", path, strerror(errno));
    FILE *f = fdopen(fd, "wb");
    if (!f) {
        rrc_status s = RRC_ERROR(RRC_ERR_IO, "fdopen %s: %s", path, strerror(errno));
        close(fd);
        unlink(path);
        return s;
    }
    *out = f;
    return RRC_OK;
}

rrc_status rrc_fopen_output_nofollow(const char *path, int append, FILE **out) {
    RRC_REQUIRE(path && out, RRC_ERR_INVALID_ARG, "null argument");
    *out = NULL;
    int fd = open(path, O_WRONLY | O_CREAT | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK | (append ? O_APPEND : 0), 0644);
    if (fd < 0)
        return RRC_ERROR(RRC_ERR_IO, "cannot open %s: %s%s", path, strerror(errno),
                         errno == ELOOP ? " (a symlink: refused)" : "");
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        close(fd);
        return RRC_ERROR(RRC_ERR_IO, "%s is not a regular file", path);
    }
    const int fl = fcntl(fd, F_GETFL);
    if (fl < 0 || fcntl(fd, F_SETFL, fl & ~O_NONBLOCK) != 0 || (!append && ftruncate(fd, 0) != 0)) {
        rrc_status s = RRC_ERROR(RRC_ERR_IO, "%s: %s", path, strerror(errno));
        close(fd);
        return s;
    }
    FILE *f = fdopen(fd, append ? "ab" : "wb");
    if (!f) {
        rrc_status s = RRC_ERROR(RRC_ERR_IO, "fdopen %s: %s", path, strerror(errno));
        close(fd);
        return s;
    }
    *out = f;
    return RRC_OK;
}
