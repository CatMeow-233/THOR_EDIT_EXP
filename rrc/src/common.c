/* rrc/src/common.c — status strings, thread-local error buffer, logging. */
#include "rrc/common.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

static const char *const k_status_names[RRC_STATUS_COUNT_] = {
    "RRC_OK",           "RRC_ERR_INVALID_ARG", "RRC_ERR_SHAPE",       "RRC_ERR_DTYPE",
    "RRC_ERR_ALIAS",    "RRC_ERR_RANGE",       "RRC_ERR_OOM",         "RRC_ERR_IO",
    "RRC_ERR_PARSE",    "RRC_ERR_UNSUPPORTED", "RRC_ERR_NOT_FOUND",   "RRC_ERR_BACKEND",
    "RRC_ERR_INTERNAL",
};

const char *rrc_status_str(rrc_status s) {
    if ((int)s < 0 || (int)s >= (int)RRC_STATUS_COUNT_) return "RRC_ERR_?";
    return k_status_names[s];
}

const char *rrc_dtype_name(rrc_dtype dt) {
    switch (dt) {
    case RRC_F32: return "f32";
    case RRC_BF16: return "bf16";
    case RRC_I32: return "i32";
    default: return "?";
    }
}

static _Thread_local char t_err_msg[RRC_ERROR_MSG_MAX];
static _Thread_local rrc_status t_err_status = RRC_OK;

static const char *short_file(const char *file) {
    /* Keep paths short and stable: strip everything before "rrc/". */
    const char *p = strstr(file, "rrc/");
    const char *last = NULL;
    while (p) {
        last = p;
        p = strstr(p + 1, "rrc/");
    }
    return last ? last : file;
}

rrc_status rrc_set_error(rrc_status s, const char *file, int line, const char *fmt, ...) {
    if (s == RRC_OK) {
        rrc_clear_error();
        return s;
    }
    char msg[RRC_ERROR_MSG_MAX];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    int n = snprintf(t_err_msg, sizeof t_err_msg, "%s:%d: %s: %s", short_file(file), line, rrc_status_str(s), msg);
    if (n < 0) t_err_msg[0] = '\0'; /* output longer than the buffer is truncated (intended) */
    t_err_status = s;
    return s;
}

rrc_status rrc_error_context(rrc_status s, const char *file, int line, const char *fmt, ...) {
    char ctx[RRC_ERROR_MSG_MAX];
    char old[RRC_ERROR_MSG_MAX];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(ctx, sizeof ctx, fmt, ap);
    va_end(ap);
    memcpy(old, t_err_msg, sizeof old);
    if (old[0] == '\0') snprintf(old, sizeof old, "%s", rrc_status_str(s));
    int n = snprintf(t_err_msg, sizeof t_err_msg, "%s:%d: %s: %s", short_file(file), line, ctx, old);
    if (n < 0) t_err_msg[0] = '\0';
    t_err_status = s;
    return s;
}

const char *rrc_last_error(void) { return t_err_msg; }
rrc_status rrc_last_status(void) { return t_err_status; }
void rrc_clear_error(void) {
    t_err_msg[0] = '\0';
    t_err_status = RRC_OK;
}

static int log_threshold(void) {
    int lvl = RRC_LOG_WARN;
    const char *e = getenv("RRC_LOG_LEVEL");
    if (e) {
        if (!strcmp(e, "error")) lvl = RRC_LOG_ERROR;
        else if (!strcmp(e, "warn")) lvl = RRC_LOG_WARN;
        else if (!strcmp(e, "info")) lvl = RRC_LOG_INFO;
        else if (!strcmp(e, "debug")) lvl = RRC_LOG_DEBUG;
    }
    return lvl;
}

void rrc_log(rrc_log_level level, const char *fmt, ...) {
    if ((int)level > log_threshold()) return;
    static const char *const names[] = {"error", "warn", "info", "debug"};
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    fprintf(stderr, "[rrc %s] %s\n", names[level], msg);
}
