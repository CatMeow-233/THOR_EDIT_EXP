/* rrc/src/tensor.c — rrc_buf allocation, views and transfers (through the backend table). */
#include "rrc/tensor.h"

#include "rrc/backend.h"

rrc_status rrc_buf_alloc(const rrc_backend *be, rrc_dtype dtype, int64_t numel, rrc_buf *out) {
    RRC_REQUIRE(be && out, RRC_ERR_INVALID_ARG, "null argument");
    *out = rrc_buf_none();
    RRC_REQUIRE(rrc_dtype_valid((int)dtype), RRC_ERR_DTYPE, "invalid dtype %d", (int)dtype);
    RRC_REQUIRE(numel >= 0 && (uint64_t)numel <= (uint64_t)INT64_MAX / 8, RRC_ERR_INVALID_ARG,
                "invalid element count %lld", (long long)numel);
    size_t bytes = (size_t)numel * rrc_dtype_size(dtype);
    void *p = NULL;
    if (bytes) {
        RRC_CHECK(be->alloc(be, bytes, &p));
        rrc_status s = be->memset0(be, p, bytes);
        if (s != RRC_OK) {
            be->free(be, p);
            return s;
        }
    }
    out->be = be;
    out->dtype = dtype;
    out->numel = numel;
    out->ptr = p;
    out->owned = 1;
    return RRC_OK;
}

rrc_status rrc_buf_free(rrc_buf *b) {
    RRC_REQUIRE(b, RRC_ERR_INVALID_ARG, "null argument");
    if (rrc_buf_is_none(*b)) return RRC_OK;
    RRC_REQUIRE(b->owned, RRC_ERR_INVALID_ARG, "rrc_buf_free on a view");
    if (b->ptr) b->be->free(b->be, b->ptr);
    *b = rrc_buf_none();
    return RRC_OK;
}

rrc_buf rrc_view(rrc_buf b, int64_t offset, int64_t numel) {
    rrc_buf v = b;
    v.owned = 0;
    if (b.numel < 0 || offset < 0 || numel < 0 || offset > b.numel || numel > b.numel - offset) {
        v.ptr = NULL;
        v.numel = -1;
        return v;
    }
    v.ptr = b.ptr ? (void *)((char *)b.ptr + (size_t)offset * rrc_dtype_size(b.dtype)) : NULL;
    v.numel = numel;
    return v;
}

rrc_buf rrc_view_from(rrc_buf b, int64_t offset) { return rrc_view(b, offset, b.numel - offset); }

static rrc_status check_xfer(rrc_buf b, int64_t numel, const char *what) {
    RRC_REQUIRE(b.be != NULL, RRC_ERR_INVALID_ARG, "%s: buffer has no backend", what);
    RRC_REQUIRE(b.numel >= 0, RRC_ERR_SHAPE, "%s: invalid view", what);
    RRC_REQUIRE(numel >= 0 && numel <= b.numel, RRC_ERR_SHAPE, "%s: %lld elements requested, buffer holds %lld", what,
                (long long)numel, (long long)b.numel);
    RRC_REQUIRE(numel == 0 || b.ptr, RRC_ERR_INVALID_ARG, "%s: null device pointer", what);
    return RRC_OK;
}

rrc_status rrc_buf_upload(rrc_buf dst, const void *host, int64_t numel) {
    RRC_CHECK(check_xfer(dst, numel, "upload"));
    if (numel == 0) return RRC_OK;
    RRC_REQUIRE(host, RRC_ERR_INVALID_ARG, "upload: host pointer is NULL");
    return dst.be->memcpy_h2d(dst.be, dst.ptr, host, (size_t)numel * rrc_dtype_size(dst.dtype));
}

rrc_status rrc_buf_download(rrc_buf src, void *host, int64_t numel) {
    RRC_CHECK(check_xfer(src, numel, "download"));
    if (numel == 0) return RRC_OK;
    RRC_REQUIRE(host, RRC_ERR_INVALID_ARG, "download: host pointer is NULL");
    return src.be->memcpy_d2h(src.be, host, src.ptr, (size_t)numel * rrc_dtype_size(src.dtype));
}

rrc_status rrc_buf_copy(rrc_buf dst, rrc_buf src, int64_t numel) {
    RRC_CHECK(check_xfer(dst, numel, "copy dst"));
    RRC_CHECK(check_xfer(src, numel, "copy src"));
    RRC_REQUIRE(dst.be == src.be, RRC_ERR_INVALID_ARG, "copy: buffers belong to different backends");
    RRC_REQUIRE(dst.dtype == src.dtype, RRC_ERR_DTYPE, "copy: dtype %s -> %s (use rrc_convert)",
                rrc_dtype_name(src.dtype), rrc_dtype_name(dst.dtype));
    if (numel == 0) return RRC_OK;
    size_t bytes = (size_t)numel * rrc_dtype_size(dst.dtype);
    uintptr_t a = (uintptr_t)dst.ptr, b = (uintptr_t)src.ptr;
    RRC_REQUIRE(a + bytes <= b || b + bytes <= a, RRC_ERR_ALIAS, "copy: source and destination overlap");
    return dst.be->memcpy_d2d(dst.be, dst.ptr, src.ptr, bytes);
}

rrc_status rrc_buf_zero(rrc_buf b, int64_t numel) {
    RRC_CHECK(check_xfer(b, numel, "zero"));
    if (numel == 0) return RRC_OK;
    return b.be->memset0(b.be, b.ptr, (size_t)numel * rrc_dtype_size(b.dtype));
}
