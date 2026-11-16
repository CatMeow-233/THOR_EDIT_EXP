/* Tests for common.h: status strings, error reporting, dtypes, bf16 conversion. */
#include <float.h>
#include <math.h>
#include <string.h>

#include "rt.h"

RT_TEST(status_strings) {
    RT_EXPECT(!strcmp(rrc_status_str(RRC_OK), "RRC_OK"), "%s", rrc_status_str(RRC_OK));
    RT_EXPECT(!strcmp(rrc_status_str(RRC_ERR_UNSUPPORTED), "RRC_ERR_UNSUPPORTED"), "bad name");
    RT_EXPECT(!strcmp(rrc_status_str((rrc_status)999), "RRC_ERR_?"), "out of range");
    for (int s = 0; s < RRC_STATUS_COUNT_; s++)
        RT_EXPECT(strncmp(rrc_status_str((rrc_status)s), "RRC_", 4) == 0, "status %d", s);
}

static rrc_status fails_deep(void) { return RRC_ERROR(RRC_ERR_SHAPE, "inner %d", 42); }
static rrc_status fails_outer(void) {
    RRC_CHECK_CTX(fails_deep(), "while doing %s", "outer");
    return RRC_OK;
}
static rrc_status require_positive(int x) {
    RRC_REQUIRE(x > 0, RRC_ERR_INVALID_ARG, "x must be > 0, got %d", x);
    return RRC_OK;
}

RT_TEST(error_reporting) {
    rrc_clear_error();
    RT_EXPECT(rrc_last_error()[0] == '\0', "not cleared");
    RT_EXPECT(rrc_last_status() == RRC_OK, "status not cleared");
    rrc_status s = fails_deep();
    RT_EXPECT(s == RRC_ERR_SHAPE, "status");
    RT_EXPECT(strstr(rrc_last_error(), "test_common.c:") != NULL, "file missing: %s", rrc_last_error());
    RT_EXPECT(strstr(rrc_last_error(), "RRC_ERR_SHAPE: inner 42") != NULL, "msg: %s", rrc_last_error());
    s = fails_outer();
    RT_EXPECT(s == RRC_ERR_SHAPE, "propagated status");
    RT_EXPECT(strstr(rrc_last_error(), "while doing outer: ") != NULL, "ctx: %s", rrc_last_error());
    RT_EXPECT(strstr(rrc_last_error(), "inner 42") != NULL, "ctx keeps inner: %s", rrc_last_error());
    RT_STATUS(require_positive(0), RRC_ERR_INVALID_ARG);
    RT_STATUS(require_positive(1), RRC_OK);
}

RT_TEST(dtypes) {
    RT_EXPECT(rrc_dtype_size(RRC_F32) == 4, "f32");
    RT_EXPECT(rrc_dtype_size(RRC_BF16) == 2, "bf16");
    RT_EXPECT(rrc_dtype_size(RRC_I32) == 4, "i32");
    RT_EXPECT(rrc_dtype_size((rrc_dtype)7) == 0, "invalid");
    RT_EXPECT(rrc_dtype_valid(0) && rrc_dtype_valid(2) && !rrc_dtype_valid(3) && !rrc_dtype_valid(-1), "valid");
    RT_EXPECT(!strcmp(rrc_dtype_name(RRC_BF16), "bf16"), "name");
}

static int is_nan_bits32(uint32_t u) { return (u & 0x7FFFFFFFu) > 0x7F800000u; }

/* Exhaustive over all 65536 bf16 bit patterns: widening is exact and narrowing round-trips. */
RT_TEST(bf16_exhaustive_roundtrip) {
    long bad = 0;
    for (uint32_t h = 0; h < 65536u; h++) {
        float f = rrc_bf16_to_f32((rrc_bf16)h);
        uint32_t u = rrc_f32_bits(f);
        if (u != (h << 16)) {
            if (bad++ < 5) RT_EXPECT(0, "widen 0x%04x gave 0x%08x", h, u);
            continue;
        }
        rrc_bf16 back = rrc_f32_to_bf16(f);
        if (is_nan_bits32(u)) {
            /* NaN stays NaN with sign/payload kept and the quiet bit set. */
            if (back != (rrc_bf16)(h | 0x0040u) || !isnan(rrc_bf16_to_f32(back)))
                if (bad++ < 5) RT_EXPECT(0, "nan 0x%04x -> 0x%04x", h, back);
        } else if (back != h) {
            if (bad++ < 5) RT_EXPECT(0, "roundtrip 0x%04x -> 0x%04x", h, back);
        }
    }
    RT_EXPECT(bad == 0, "%ld bad patterns", bad);
}

/* Independent reference for f32 -> bf16 RNE using double arithmetic: choose the nearer of the two
 * neighbouring bf16 values (treating the value past the largest finite bf16 as 2^128, as IEEE
 * overflow rounding does), ties to the even mantissa. */
static uint16_t ref_f32_to_bf16(float f) {
    uint32_t u = rrc_f32_bits(f);
    if (is_nan_bits32(u)) return 0xFFFF; /* caller handles NaN */
    uint16_t h0 = (uint16_t)(u >> 16);   /* truncation toward zero */
    if ((u & 0xFFFFu) == 0) return h0;
    if ((h0 & 0x7FFF) == 0x7F80) return h0; /* already +-inf: impossible with nonzero low bits */
    uint16_t h1 = (uint16_t)(h0 + 1);       /* next magnitude up, same sign */
    double x = fabs((double)f);
    double lo = fabs((double)rrc_bf16_to_f32(h0));
    double hi = ((h1 & 0x7FFF) == 0x7F80) ? ldexp(1.0, 128) : fabs((double)rrc_bf16_to_f32(h1));
    double dlo = x - lo, dhi = hi - x;
    if (dlo < dhi) return h0;
    if (dhi < dlo) return h1;
    return (h0 & 1) ? h1 : h0;
}

RT_TEST(bf16_rne_stratified) {
    static const uint32_t lows[] = {0x0000, 0x0001, 0x3FFF, 0x7FFF, 0x8000, 0x8001, 0xBFFF, 0xFFFF, 0x1234, 0xC001};
    long bad = 0, n = 0;
    for (uint32_t hi = 0; hi < 65536u; hi++) {
        for (size_t k = 0; k < sizeof lows / sizeof lows[0]; k++) {
            uint32_t u = (hi << 16) | lows[k];
            float f = rrc_bits_f32(u);
            uint16_t got = rrc_f32_to_bf16(f);
            n++;
            if (is_nan_bits32(u)) {
                if (!isnan(rrc_bf16_to_f32(got)) || got != (uint16_t)((u >> 16) | 0x40))
                    if (bad++ < 5) RT_EXPECT(0, "nan 0x%08x -> 0x%04x", u, got);
                continue;
            }
            uint16_t want = ref_f32_to_bf16(f);
            if (got != want)
                if (bad++ < 5) RT_EXPECT(0, "0x%08x -> 0x%04x want 0x%04x", u, got, want);
        }
    }
    RT_EXPECT(bad == 0, "%ld of %ld patterns wrong", bad, n);
}

RT_TEST(bf16_boundaries) {
    RT_EXPECT(rrc_f32_to_bf16(0.0f) == 0x0000, "+0");
    RT_EXPECT(rrc_f32_to_bf16(-0.0f) == 0x8000, "-0");
    RT_EXPECT(rrc_f32_to_bf16(1.0f) == 0x3F80, "1");
    RT_EXPECT(rrc_f32_to_bf16(1.00390625f) == 0x3F80, "tie 1+2^-8 -> even 1.0");
    RT_EXPECT(rrc_f32_to_bf16(1.01171875f) == 0x3F82, "tie 1+3*2^-8 -> even 0x3F82");
    RT_EXPECT(rrc_f32_to_bf16(nextafterf(1.00390625f, 2.0f)) == 0x3F81, "just above tie");
    RT_EXPECT(rrc_f32_to_bf16(INFINITY) == 0x7F80, "+inf");
    RT_EXPECT(rrc_f32_to_bf16(-INFINITY) == 0xFF80, "-inf");
    RT_EXPECT(rrc_f32_to_bf16(FLT_MAX) == 0x7F80, "FLT_MAX rounds to inf");
    RT_EXPECT(rrc_f32_to_bf16(-FLT_MAX) == 0xFF80, "-FLT_MAX rounds to -inf");
    /* largest finite bf16 = 0x7F7F; values below max + half ulp stay finite */
    float maxbf = rrc_bf16_to_f32(0x7F7F);
    RT_EXPECT(rrc_f32_to_bf16(maxbf) == 0x7F7F, "max bf16");
    RT_EXPECT(rrc_f32_to_bf16(rrc_bits_f32(0x7F7F7FFFu)) == 0x7F7F, "below half-ulp stays finite");
    RT_EXPECT(rrc_f32_to_bf16(rrc_bits_f32(0x7F7F8000u)) == 0x7F80, "tie at max -> even = inf");
    RT_EXPECT(rrc_f32_to_bf16(FLT_MIN) == 0x0080, "FLT_MIN");
    RT_EXPECT(rrc_f32_to_bf16(rrc_bits_f32(0x00000001u)) == 0x0000, "min denormal -> 0");
    RT_EXPECT(rrc_f32_to_bf16(rrc_bits_f32(0x00008000u)) == 0x0000, "denormal tie -> even 0");
    RT_EXPECT(rrc_f32_to_bf16(rrc_bits_f32(0x00018000u)) == 0x0002, "denormal tie -> even 2");
    RT_EXPECT(rrc_f32_to_bf16(rrc_bits_f32(0x007FFFFFu)) == 0x0080, "max denormal rounds up to FLT_MIN");
    RT_EXPECT(rrc_f32_to_bf16(NAN) != 0x7F80 && isnan(rrc_bf16_to_f32(rrc_f32_to_bf16(NAN))), "nan");
    RT_EXPECT(rrc_f32_to_bf16(rrc_bits_f32(0x7F800001u)) == 0x7FC0, "sNaN low payload -> quiet NaN");
    RT_EXPECT(rrc_f32_to_bf16(rrc_bits_f32(0xFFC00000u)) == 0xFFC0, "negative qNaN keeps sign");
    RT_EXPECT(rrc_round_bf16(3.14159265f) == rrc_bf16_to_f32(0x4049), "pi -> 0x4049 (%a)",
              (double)rrc_round_bf16(3.14159265f));
}

const rt_case rt_suite_common[] = {
    RT_CASE(status_strings),         RT_CASE(error_reporting), RT_CASE(dtypes),
    RT_CASE(bf16_exhaustive_roundtrip), RT_CASE(bf16_rne_stratified), RT_CASE(bf16_boundaries),
};
const size_t rt_suite_common_n = sizeof rt_suite_common / sizeof rt_suite_common[0];
