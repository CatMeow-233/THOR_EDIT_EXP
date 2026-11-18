"""Tests of research_rl.capability.stats: exactness, edge cases and Monte-Carlo calibration.

All simulations draw from seeded ``random.Random`` streams directly at the level of discordant
pairs (no models), so the file runs in seconds on CPU.
"""

from __future__ import annotations

import itertools
import math
import random
import time
from collections.abc import Callable
from fractions import Fraction

import pytest

from research_rl.capability.stats import (
    AlphaLedger,
    binomial_upper_tail,
    clopper_pearson_lower,
    discordant_mass,
    fractional_sign_test_p_value,
    noninferiority_upper_bound,
    normal_cdf,
    normal_quantile,
    paired_gain_lower_bound,
    pool_paired,
    regularized_incomplete_beta,
    required_paired_n,
    retention_power_floor_n,
    sign_test_p_value,
    student_t_quantile,
    student_t_sf,
    tango_restricted_mle,
    tango_z,
)
from research_rl.capability.types import PairedScores, Split

# Monte-Carlo settings fixed up front (not tuned): R replications, tolerance = 3 standard errors.
REPS = 4000
ALPHA = 0.05
MC_TOL = 3.0 * math.sqrt(ALPHA * (1.0 - ALPHA) / REPS)  # ~0.0103


# ============================================================================== helpers
def make_binary(
    n01: int,
    n10: int,
    n11: int,
    n00: int,
    *,
    name: str = "s",
    split: Split = Split.RETENTION,
    prefix: str = "",
) -> PairedScores:
    """Binary paired scores with the given cell counts (incumbent, candidate)."""
    inc: list[float] = [0.0] * n01 + [1.0] * n10 + [1.0] * n11 + [0.0] * n00
    cand: list[float] = [1.0] * n01 + [0.0] * n10 + [1.0] * n11 + [0.0] * n00
    uids = tuple(f"{prefix}{name}-{i}" for i in range(len(inc)))
    return PairedScores(name, split, uids, tuple(inc), tuple(cand))


def make_scores(inc: list[float], cand: list[float], name: str = "s") -> PairedScores:
    return PairedScores(
        name, Split.T_RULE, tuple(f"{name}-{i}" for i in range(len(inc))), tuple(inc), tuple(cand)
    )


def exact_binomial_tail(x: int, m: int, p: Fraction) -> Fraction:
    return sum((Fraction(math.comb(m, k)) * p**k * (1 - p) ** (m - k) for k in range(x, m + 1)), Fraction(0))


def draw_counts(rng: random.Random, n: int, p01: float, p10: float) -> tuple[int, int]:
    """One trinomial draw of (n01, n10) from n pairs."""
    n01 = n10 = 0
    q = p01 + p10
    for _ in range(n):
        u = rng.random()
        if u < p01:
            n01 += 1
        elif u < q:
            n10 += 1
    return n01, n10


def trinomial_cells(n: int, p01: float, p10: float, sd_span: float = 12.0):
    """Cells (n01, n10, probability) of the trinomial, restricted to +- sd_span sd per margin."""
    lf = math.lgamma
    pc = 1.0 - p01 - p10

    def window(p: float) -> range:
        mu, sd = n * p, math.sqrt(n * p * (1.0 - p))
        return range(max(0, int(mu - sd_span * sd - 2)), min(n, int(mu + sd_span * sd + 2)) + 1)

    for a in window(p01):
        for b in window(p10):
            if a + b > n:
                continue
            lp = lf(n + 1) - lf(a + 1) - lf(b + 1) - lf(n - a - b + 1)
            lp += a * math.log(p01) + b * math.log(p10) + (n - a - b) * math.log(pc)
            yield a, b, math.exp(lp)


# ============================================================================== numerical blocks
def test_normal_quantile_known_values_and_round_trip() -> None:
    assert normal_quantile(0.5) == pytest.approx(0.0, abs=1e-15)
    assert normal_quantile(0.95) == pytest.approx(1.6448536269514722, abs=1e-12)
    assert normal_quantile(0.975) == pytest.approx(1.959963984540054, abs=1e-12)
    assert normal_quantile(0.8) == pytest.approx(0.8416212335729143, abs=1e-12)
    assert normal_quantile(0.05) == pytest.approx(-1.6448536269514722, abs=1e-12)
    for p in (1e-12, 1e-8, 0.001, 0.02, 0.0243, 0.3, 0.7, 0.9757, 0.999, 1 - 1e-10):
        assert normal_cdf(normal_quantile(p)) == pytest.approx(p, rel=1e-10, abs=1e-15)
    for p in (1 - 1e-10, 1 - 1e-6, 0.999, 0.98, 0.9, 0.6):
        assert normal_quantile(p) == -normal_quantile(1.0 - p)
    assert normal_quantile(1e-10) == pytest.approx(-6.361340902404056, abs=1e-12)
    for bad in (0.0, 1.0, -0.1, math.nan):
        with pytest.raises(ValueError):
            normal_quantile(bad)


def test_incomplete_beta_matches_exact_binomial_sums() -> None:
    for m in (1, 5, 12, 30):
        for x in range(0, m + 2):
            for p in (Fraction(1, 7), Fraction(1, 2), Fraction(9, 10)):
                exact = float(exact_binomial_tail(x, m, p)) if x <= m else 0.0
                got = binomial_upper_tail(x, m, float(p))
                assert got == pytest.approx(exact, rel=1e-11, abs=1e-15)
    assert regularized_incomplete_beta(2.0, 3.0, 0.0) == 0.0
    assert regularized_incomplete_beta(2.0, 3.0, 1.0) == 1.0
    # I_x(1, 1) = x
    assert regularized_incomplete_beta(1.0, 1.0, 0.37) == pytest.approx(0.37, rel=1e-13)


def test_student_t_against_closed_forms() -> None:
    for p in (0.6, 0.9, 0.95, 0.975, 0.999):
        # df = 1 (Cauchy) and df = 2 have closed-form quantiles.
        assert student_t_quantile(p, 1.0) == pytest.approx(math.tan(math.pi * (p - 0.5)), rel=1e-12)
        assert student_t_quantile(p, 2.0) == pytest.approx(
            (2 * p - 1) / math.sqrt(2 * p * (1 - p)), rel=1e-12
        )
        assert student_t_quantile(1.0 - p, 2.0) == -student_t_quantile(p, 2.0)
    # Table value t_{0.975, 5} = 2.5706 (4 decimals).
    assert student_t_quantile(0.975, 5.0) == pytest.approx(2.5706, abs=5e-5)
    # Large df approaches the normal quantile.
    assert student_t_quantile(0.95, 1e6) == pytest.approx(normal_quantile(0.95), abs=1e-5)
    assert student_t_sf(0.0, 7.0) == pytest.approx(0.5, abs=1e-15)
    assert student_t_sf(1.5, 7.0) + student_t_sf(-1.5, 7.0) == pytest.approx(1.0, abs=1e-14)


# ============================================================================== sign test
def test_sign_test_known_values_and_no_discordance() -> None:
    assert sign_test_p_value(0, 0) == 1.0
    assert sign_test_p_value(8, 2) == pytest.approx(56 / 1024, abs=0.0)
    assert sign_test_p_value(0, 5) == 1.0
    assert sign_test_p_value(5, 0) == 1 / 32
    with pytest.raises(ValueError):
        sign_test_p_value(-1, 2)


def test_sign_test_matches_brute_force_exact() -> None:
    for m in range(0, 25):
        for n01 in range(0, m + 1):
            exact = exact_binomial_tail(n01, m, Fraction(1, 2)) if m else Fraction(1)
            assert sign_test_p_value(n01, m - n01) == float(exact)


def test_sign_test_monotone_and_symmetric() -> None:
    for m in (1, 7, 40, 120):
        ps = [sign_test_p_value(k, m - k) for k in range(m + 1)]
        if m <= 50:  # beyond that 1 - 2^-m rounds to 1.0 in double precision
            assert all(a > b for a, b in itertools.pairwise(ps)), "more candidate wins must lower p"
        else:
            assert all(a >= b for a, b in itertools.pairwise(ps))
            assert ps[-1] < ps[m // 2] < ps[1]
        for k in range(m + 1):
            # P[X >= k] + P[X >= m - k] = 1 + P[X = k] for X ~ Bin(m, 1/2)
            point = math.comb(m, k) / 2**m
            assert sign_test_p_value(k, m - k) + sign_test_p_value(m - k, k) == pytest.approx(
                1 + point, abs=1e-12
            )
    # More incumbent wins at fixed candidate wins also raises p.
    assert sign_test_p_value(10, 2) < sign_test_p_value(10, 5) < sign_test_p_value(10, 9)


def test_sign_test_large_m_fast_and_accurate() -> None:
    """Regression: the per-k math.comb sum took ~27 s at m = 20000 (plus ~26 s inside CP)."""
    m = 20000
    x = m // 2 + 283
    start = time.perf_counter()
    p = sign_test_p_value(x, m - x)
    low = clopper_pearson_lower(x, m, 0.05)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"sign test + Clopper-Pearson took {elapsed:.2f} s"
    assert p == pytest.approx(binomial_upper_tail(x, m, 0.5), rel=1e-9)
    assert p < 0.05 and 0.5 < low < x / m
    # Both tails of the shorter-tail summation agree with the complement identity.
    for k in (0, 1, 9999, 10000, 10001, 19999, 20000):
        both = sign_test_p_value(k, m - k) + sign_test_p_value(m - k, k)
        assert both == pytest.approx(1.0 + math.comb(m, k) / 2**m, rel=1e-12)


def test_clopper_pearson_matches_brute_force_and_sign_test() -> None:
    def brute_lower(x: int, m: int, alpha: float) -> float:
        lo, hi = Fraction(0), Fraction(1)
        for _ in range(60):
            mid = (lo + hi) / 2
            if exact_binomial_tail(x, m, mid) > Fraction(alpha):
                hi = mid
            else:
                lo = mid
        return float(lo)

    for m in (1, 4, 9, 15):
        for x in range(0, m + 1):
            for alpha in (0.05, 0.01):
                got = clopper_pearson_lower(x, m, alpha)
                expected = 0.0 if x == 0 else brute_lower(x, m, alpha)
                assert got == pytest.approx(expected, abs=1e-9)
                assert (got >= 0.5) == (sign_test_p_value(x, m - x) <= alpha)
    # Textbook value: 8/10 one-sided 95% lower limit = 0.4931.
    assert clopper_pearson_lower(8, 10, 0.05) == pytest.approx(0.49310, abs=1e-5)


# ============================================================================== gain test
def test_gain_binary_decision_bound_and_point() -> None:
    p = make_binary(20, 2, 30, 48, split=Split.T_RULE)
    t = paired_gain_lower_bound(p, 0.05)
    assert t.method == "sign_test_exact"
    assert t.n == 100
    assert t.point == pytest.approx(0.18)
    assert t.p_value == sign_test_p_value(20, 2)
    assert t.passed
    p_low = clopper_pearson_lower(20, 22, 0.05)
    assert t.bound == pytest.approx((2 * p_low - 1) * 22 / 100)
    assert 0.0 < t.bound < t.point
    assert t.alpha == 0.05 and t.margin is None


def test_gain_edge_cases() -> None:
    empty = PairedScores("e", Split.T_RULE, (), (), ())
    t = paired_gain_lower_bound(empty, 0.05)
    assert (t.n, t.passed, t.method, t.bound) == (0, False, "empty", -1.0)
    no_disc = make_binary(0, 0, 10, 10)
    t = paired_gain_lower_bound(no_disc, 0.05)
    assert t.passed is False and t.p_value == 1.0 and t.bound == 0.0 and t.point == 0.0
    all_disc = make_binary(10, 0, 0, 0)
    t = paired_gain_lower_bound(all_disc, 0.05)
    assert t.passed and t.p_value == 2**-10 and t.bound > 0.0
    with pytest.raises(ValueError):
        paired_gain_lower_bound(no_disc, 0.0)


def test_gain_passed_iff_bound_positive_grid() -> None:
    for n in (10, 60):
        for n01 in range(0, n + 1, 3):
            for n10 in range(0, n - n01 + 1, 3):
                t = paired_gain_lower_bound(make_binary(n01, n10, 0, n - n01 - n10), 0.05)
                if t.passed:
                    assert t.bound >= 0.0
                    if t.p_value is not None and t.p_value < 0.05 * (1 - 1e-9):
                        assert t.bound > 0.0
                else:
                    assert t.bound <= 0.0


def test_gain_monotone_more_wins_lower_p_higher_bound() -> None:
    # Turning an incumbent win into a candidate win (m fixed): p falls, bound rises strictly.
    prev_p, prev_b = 2.0, -2.0
    for n01 in range(0, 31):
        t = paired_gain_lower_bound(make_binary(n01, 30 - n01, 0, 170), 0.05)
        assert t.p_value is not None
        assert t.p_value < prev_p
        assert t.bound > prev_b
        prev_p, prev_b = t.p_value, t.bound
    # Turning a concordant pair into a candidate win: p always falls; the conditional bound rises
    # once the test passes. (It is not monotone in the non-passing region because it scales with
    # the number of discordant pairs m: e.g. n01=0 -> -0.025, n01=1 -> -0.0295 at n10=5, n=200.)
    prev_p, prev_b = 2.0, -2.0
    for n01 in range(0, 40):
        t = paired_gain_lower_bound(make_binary(n01, 5, 0, 200 - n01 - 5), 0.05)
        assert t.p_value is not None
        assert t.p_value < prev_p
        if t.passed:
            assert t.bound > prev_b
        prev_p, prev_b = t.p_value, t.bound


def test_gain_pooling_concatenates_and_rejects_duplicate_uids() -> None:
    a = make_binary(6, 1, 3, 10, name="t_rule/unseen_pairs", split=Split.T_RULE)
    b = make_binary(5, 0, 2, 13, name="t_task/argmax", split=Split.T_TASK)
    pooled = pool_paired([a, b])
    assert pooled.n == a.n + b.n
    assert pooled.slice == "t_rule/unseen_pairs+t_task/argmax"
    assert pooled.discordant() == (11, 1)
    t = paired_gain_lower_bound([a, b], 0.05)
    assert t.n == 40 and t.p_value == sign_test_p_value(11, 1)
    clash = make_binary(1, 0, 0, 0, name="t_rule/unseen_pairs", prefix="")  # uid "t_rule/unseen_pairs-0"
    clash = PairedScores("other", Split.T_TASK, clash.uids, clash.incumbent, clash.candidate)
    with pytest.raises(ValueError, match="appears in slices"):
        paired_gain_lower_bound([a, clash], 0.05)
    with pytest.raises(ValueError):
        pool_paired([])
    assert pool_paired([a]) is a


def test_gain_swapped_roles_complementary_p() -> None:
    p = make_binary(9, 4, 5, 2)
    swapped = PairedScores("s", p.split, p.uids, p.candidate, p.incumbent)
    pg = paired_gain_lower_bound(p, 0.05).p_value
    ps = paired_gain_lower_bound(swapped, 0.05).p_value
    assert pg is not None and ps is not None
    assert pg + ps == pytest.approx(1.0 + math.comb(13, 9) / 2**13, abs=1e-12)


def test_gain_non_binary_fractional_by_hand() -> None:
    inc = [0.2, 0.5, 0.1, 0.4, 0.3, 0.9]
    cand = [0.5, 0.6, 0.3, 0.4, 0.7, 0.8]
    ps = make_scores(inc, cand)
    gain, loss = discordant_mass(ps)
    assert gain == pytest.approx(0.3 + 0.1 + 0.2 + 0.4) and loss == pytest.approx(0.1)
    t = paired_gain_lower_bound(ps, 0.05)
    assert t.method == "sign_test_fractional"
    assert t.point == pytest.approx((gain - loss) / 6)
    assert t.p_value == pytest.approx(regularized_incomplete_beta(gain, loss + 1.0, 0.5), rel=1e-12)
    assert t.p_value == fractional_sign_test_p_value(gain, loss)
    assert not t.passed and t.bound <= 0.0  # one pair's worth of gain mass cannot be significant
    # Ten items each improved by +0.75: 7.5 pairs of gain mass, p = 2^-7.5 < 0.05.
    strong = paired_gain_lower_bound(make_scores([0.25] * 10, [1.0] * 10), 0.05)
    assert strong.passed and strong.p_value == pytest.approx(2**-7.5, rel=1e-12)
    assert 0.0 < strong.bound < strong.point
    assert strong.bound == pytest.approx((2 * _fractional_p_low(7.5, 0.0, 0.05) - 1) * 7.5 / 10, rel=1e-9)
    # A single item is handled like the binary case (no separate "insufficient" path).
    one = paired_gain_lower_bound(make_scores([0.2], [0.7]), 0.05)
    assert not one.passed and one.method == "sign_test_fractional"
    zero = paired_gain_lower_bound(make_scores([0.2, 0.3], [0.2, 0.3]), 0.05)
    assert not zero.passed and zero.bound == 0.0 and zero.p_value == 1.0


def _fractional_p_low(gain: float, loss: float, alpha: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if regularized_incomplete_beta(gain, loss + 1.0, mid) > alpha:
            hi = mid
        else:
            lo = mid
    return lo


def test_fractional_reduces_to_binary_on_integer_mass() -> None:
    for n01, n10 in ((0, 0), (5, 0), (8, 2), (20, 2), (3, 9), (60, 41)):
        p = make_binary(n01, n10, 7, 3)
        assert discordant_mass(p) == (float(n01), float(n10))
        assert fractional_sign_test_p_value(n01, n10) == pytest.approx(
            sign_test_p_value(n01, n10), rel=1e-11, abs=1e-15
        )
        assert tango_z(float(n10), float(n01), p.n, 0.02) == tango_z(n10, n01, p.n, 0.02)
    with pytest.raises(ValueError):
        fractional_sign_test_p_value(-0.5, 1.0)
    with pytest.raises(ValueError):
        tango_restricted_mle(math.nan, 0.0, 10, 0.02)


def test_non_binary_zero_variance_has_no_shortcut() -> None:
    """Regression: all-tied or constant-difference non-binary slices used to pass with p = 0."""
    tied = make_scores([0.5, 0.5], [0.5, 0.5])
    t = noninferiority_upper_bound(tied, 0.02, 0.05)
    assert not t.passed and t.p_value is not None and t.p_value > 0.05
    # Ten tied items, one of them partial credit: same decision and p-value as ten binary ties.
    t10 = noninferiority_upper_bound(make_scores([1.0] * 9 + [0.5], [1.0] * 9 + [0.5]), 0.02, 0.05)
    b10 = noninferiority_upper_bound(make_binary(0, 0, 10, 0), 0.02, 0.05)
    assert not t10.passed and t10.p_value == b10.p_value and t10.bound == b10.bound
    const_gain = paired_gain_lower_bound(make_scores([0.0, 0.25], [0.25, 0.5]), 0.05)
    assert not const_gain.passed and const_gain.p_value == pytest.approx(0.5**0.5, rel=1e-12)
    # Large all-tied partial-credit slices pass exactly where binary ones do (the power floor).
    floor = retention_power_floor_n(0.02, 0.05)
    for n, expect in ((floor, True), (floor - 1, False)):
        assert noninferiority_upper_bound(make_scores([0.5] * n, [0.5] * n), 0.02, 0.05).passed is expect


def test_mixed_binary_and_partial_tie_matches_binary() -> None:
    """Regression: one partial-credit tie used to move the whole slice to an anti-conservative t test."""
    base = make_binary(3, 8, 400, 89)
    with_half = make_scores([*base.incumbent, 0.5], [*base.candidate, 0.5])
    with_tie = make_scores([*base.incumbent, 1.0], [*base.candidate, 1.0])
    a = noninferiority_upper_bound(with_half, 0.02, 0.05)
    b = noninferiority_upper_bound(with_tie, 0.02, 0.05)
    assert a.method == "tango_score_fractional" and b.method == "tango_score"
    assert (a.p_value, a.bound, a.passed, a.point) == (b.p_value, b.bound, b.passed, b.point)
    ga = paired_gain_lower_bound(with_half, 0.05)
    gb = paired_gain_lower_bound(with_tie, 0.05)
    assert ga.passed == gb.passed and ga.p_value == pytest.approx(gb.p_value, rel=1e-11)


def test_tango_z_monotone_in_delta_for_fractional_mass() -> None:
    """The bound bisection needs Z(d) non-increasing in d, also for fractional counts."""
    for n in (3, 17, 80):
        for b in (0.0, 0.25, 1.5, n / 3):
            for c in (0.0, 0.75, 2.25, n / 4):
                if b + c > n:
                    continue
                prev = math.inf
                for i in range(-200, 201):
                    z = tango_z(b, c, n, i / 200)
                    assert z <= prev + 1e-12, (n, b, c, i)
                    prev = z


# ============================================================================== Tango
def _constrained_loglik(b: int, c: int, n: int, delta: float, q: float) -> float:
    d = n - b - c
    total = 0.0
    for count, prob in ((b, q + delta), (c, q), (d, 1 - 2 * q - delta)):
        if count:
            if prob <= 0:
                return -math.inf
            total += count * math.log(prob)
    return total


def test_tango_restricted_mle_matches_numerical_maximisation() -> None:
    for n in (5, 12, 30):
        for b in range(0, n + 1, 2):
            for c in range(0, n - b + 1, 3):
                for delta in (0.0, 0.02, 0.1, 0.3, 0.7):
                    q_closed = tango_restricted_mle(b, c, n, delta)
                    hi = (1 - delta) / 2
                    assert 0.0 <= q_closed <= hi + 1e-15
                    # Golden-section search on the (concave) constrained log-likelihood.
                    lo_, hi_ = 0.0, hi
                    g = (math.sqrt(5) - 1) / 2
                    for _ in range(200):
                        x1, x2 = hi_ - g * (hi_ - lo_), lo_ + g * (hi_ - lo_)
                        if _constrained_loglik(b, c, n, delta, x1) < _constrained_loglik(b, c, n, delta, x2):
                            lo_ = x1
                        else:
                            hi_ = x2
                    q_num = 0.5 * (lo_ + hi_)
                    ll_closed = _constrained_loglik(b, c, n, delta, q_closed)
                    ll_num = _constrained_loglik(b, c, n, delta, q_num)
                    assert ll_closed >= ll_num - 1e-9, (b, c, n, delta, q_closed, q_num)


def test_tango_symmetry_and_mcnemar_special_case() -> None:
    for b, c, n in ((3, 7, 40), (0, 5, 20), (9, 0, 9), (4, 4, 100)):
        for delta in (0.05, 0.2, 0.5):
            assert tango_z(b, c, n, -delta) == pytest.approx(-tango_z(c, b, n, delta), abs=1e-12)
    # margin 0: Tango reduces to McNemar's statistic (b - c) / sqrt(b + c).
    assert tango_z(12, 5, 80, 0.0) == pytest.approx(7 / math.sqrt(17), abs=1e-12)
    # No discordant pairs: Z = -sqrt(n margin / (1 - margin)).
    assert tango_z(0, 0, 300, 0.02) == pytest.approx(-math.sqrt(300 * 0.02 / 0.98), abs=1e-12)
    assert tango_z(0, 0, 10, 0.0) == 0.0
    # All pairs discordant: MLE sits on the boundary q = (1 - delta) / 2.
    assert tango_restricted_mle(6, 4, 10, 0.1) == pytest.approx(0.45, abs=1e-12)


def test_noninferiority_decision_and_edges() -> None:
    clean = make_binary(0, 0, 250, 50)
    t = noninferiority_upper_bound(clean, 0.02, 0.05)
    assert t.method == "tango_score" and t.margin == 0.02
    assert t.passed and t.point == 0.0 and t.bound < 0.02
    assert t.p_value == pytest.approx(normal_cdf(-math.sqrt(300 * 0.02 / 0.98)))
    big_drop = make_binary(0, 30, 220, 50)
    t = noninferiority_upper_bound(big_drop, 0.02, 0.05)
    assert not t.passed and t.point == pytest.approx(0.1) and t.bound > 0.1
    # Bound is where the score statistic crosses -z.
    assert tango_z(30, 0, 300, t.bound) == pytest.approx(-normal_quantile(0.95), abs=1e-6)
    empty = noninferiority_upper_bound(PairedScores("e", Split.RETENTION, (), (), ()), 0.02, 0.05)
    assert (empty.passed, empty.bound, empty.method) == (False, 1.0, "empty")
    all_inc = make_binary(0, 40, 0, 0)  # every pair an incumbent win: drop = 1
    t = noninferiority_upper_bound(all_inc, 0.02, 0.05)
    assert not t.passed and t.bound == 1.0
    all_cand = make_binary(40, 0, 0, 0)
    t = noninferiority_upper_bound(all_cand, 0.02, 0.05)
    assert t.passed and t.point == -1.0 and t.bound < 0.0
    small = make_binary(0, 0, 50, 50)  # below the power floor: cannot pass at margin 0.02
    assert not noninferiority_upper_bound(small, 0.02, 0.05).passed
    for bad in (-0.1, 1.0, math.nan):
        with pytest.raises(ValueError):
            noninferiority_upper_bound(clean, bad, 0.05)


def test_noninferiority_passed_iff_bound_below_margin_grid() -> None:
    for n in (20, 120):
        for b in range(0, n + 1, 4):
            for c in range(0, n - b + 1, 4):
                for margin in (0.0, 0.02, 0.1):
                    t = noninferiority_upper_bound(make_binary(c, b, 0, n - b - c), margin, 0.05)
                    if t.passed:
                        assert t.bound <= margin
                    else:
                        assert t.bound >= margin


def test_noninferiority_monotone_and_symmetric() -> None:
    prev_p, prev_b = -1.0, -2.0
    for n10 in range(0, 30):
        t = noninferiority_upper_bound(make_binary(3, n10, 200, 297 - n10), 0.02, 0.05)
        assert t.p_value is not None
        assert t.p_value > prev_p, "more incumbent wins must raise the p-value"
        assert t.bound >= prev_b
        prev_p, prev_b = t.p_value, t.bound
    # Swapping roles negates the drop: point flips sign.
    p = make_binary(7, 2, 50, 41)
    swapped = PairedScores("s", p.split, p.uids, p.candidate, p.incumbent)
    assert (
        noninferiority_upper_bound(p, 0.02, 0.05).point
        == -noninferiority_upper_bound(swapped, 0.02, 0.05).point
    )


def test_noninferiority_non_binary_fractional() -> None:
    inc = [0.5, 0.6, 0.7, 0.4, 0.5, 0.6] * 50
    cand = [0.5, 0.55, 0.7, 0.45, 0.5, 0.6] * 50
    ps = make_scores(inc, cand)
    t = noninferiority_upper_bound(ps, 0.05, 0.05)
    gain, loss = discordant_mass(ps)
    assert gain == pytest.approx(2.5) and loss == pytest.approx(2.5)
    assert t.method == "tango_score_fractional"
    assert t.point == pytest.approx(0.0, abs=1e-15)
    assert t.p_value == pytest.approx(normal_cdf(tango_z(loss, gain, 300, 0.05)), rel=1e-12)
    assert t.passed and t.bound < 0.05
    assert tango_z(loss, gain, 300, t.bound) == pytest.approx(-normal_quantile(0.95), abs=1e-6)
    one = noninferiority_upper_bound(make_scores([0.5], [0.4]), 0.05, 0.05)
    assert not one.passed and one.method == "tango_score_fractional" and one.bound >= 0.05


def test_retention_power_floor_n_is_tight() -> None:
    for margin, alpha in ((0.02, 0.05), (0.02, 0.05 * 6 / math.pi**2), (0.05, 0.01)):
        n = retention_power_floor_n(margin, alpha)
        assert noninferiority_upper_bound(make_binary(0, 0, n, 0), margin, alpha).passed
        assert not noninferiority_upper_bound(make_binary(0, 0, n - 1, 0), margin, alpha).passed
    ledger = AlphaLedger(0.05, "spending")
    floors = [retention_power_floor_n(0.02, ledger.alpha_at(k)) for k in (1, 7, 10)]
    assert floors == [173, 512, 576]  # the numbers quoted in GateConfig's docstring


# ============================================================================== calibration
@pytest.mark.parametrize("n", [50, 200])
@pytest.mark.parametrize("q", [0.1, 0.3])
def test_calibration_gain_false_accept_at_zero_gain(n: int, q: float) -> None:
    rng = random.Random(1000 + n + int(q * 100))
    cache: dict[tuple[int, int], bool] = {}
    accepts = 0
    for _ in range(REPS):
        key = draw_counts(rng, n, q / 2, q / 2)
        if key not in cache:
            cache[key] = paired_gain_lower_bound(make_binary(key[0], key[1], 0, n - sum(key)), ALPHA).passed
        accepts += cache[key]
    rate = accepts / REPS
    assert rate <= ALPHA + MC_TOL, f"false-accept rate {rate:.4f} at n={n}, q={q}"


@pytest.mark.parametrize("n", [50, 200])
@pytest.mark.parametrize("q", [0.1, 0.3])
@pytest.mark.parametrize("margin", [0.02, 0.05])
def test_calibration_noninferiority_false_pass_at_margin(n: int, q: float, margin: float) -> None:
    p10, p01 = (q + margin) / 2, (q - margin) / 2  # true drop = margin, discordance = q
    rng = random.Random(2000 + n + int(q * 100) + int(margin * 1000))
    cache: dict[tuple[int, int], bool] = {}
    passes = 0
    for _ in range(REPS):
        key = draw_counts(rng, n, p01, p10)
        if key not in cache:
            ps = make_binary(key[0], key[1], 0, n - sum(key))
            cache[key] = noninferiority_upper_bound(ps, margin, ALPHA).passed
        passes += cache[key]
    rate = passes / REPS
    assert rate <= ALPHA + MC_TOL, f"false-pass rate {rate:.4f} at n={n}, q={q}, margin={margin}"


def test_exact_size_by_enumeration() -> None:
    """Exact (enumerated) rejection probabilities at the calibration settings.

    The sign test is exact, so its size never exceeds alpha. Tango's score test is asymptotic:
    its measured exact size is at most 0.0549 at these settings (n=50, q=0.1, margin=0.02); the
    asserted bound alpha + 0.006 documents that excess.
    """
    worst = 0.0
    for n in (50, 200):
        for q in (0.1, 0.3):
            size = sum(
                pr for a, b, pr in trinomial_cells(n, q / 2, q / 2) if sign_test_p_value(a, b) <= ALPHA
            )
            assert size <= ALPHA
            for margin in (0.02, 0.05):
                p10, p01 = (q + margin) / 2, (q - margin) / 2
                size = sum(
                    pr
                    for a, b, pr in trinomial_cells(n, p01, p10)
                    if normal_cdf(tango_z(b, a, n, margin)) <= ALPHA
                )
                worst = max(worst, size)
                assert size <= ALPHA + 0.006, (n, q, margin, size)
    assert worst > ALPHA  # the documented small-sample excess is real, not an artefact


PARTIAL = (0.0, 0.25, 0.5, 0.75, 1.0)
Pair = tuple[float, float]


# Gain scenarios: (incumbent, candidate) generators with true mean gain exactly 0.
def _gain_partial_symmetric(rng: random.Random) -> Pair:
    x = rng.choice(PARTIAL)
    return (x, x if rng.random() < 0.6 else rng.choice(PARTIAL))


def _gain_skewed(rng: random.Random) -> Pair:  # +0.25 w.p. 0.16, -0.5 w.p. 0.08 (reviewer case)
    u = rng.random()
    return (0.5, 0.75) if u < 0.16 else ((0.75, 0.25) if u < 0.24 else (0.5, 0.5))


def _gain_small_wins_rare_full_losses(rng: random.Random) -> Pair:  # +0.1 w.p. 0.9, -1 w.p. 0.09
    u = rng.random()
    return (0.5, 0.6) if u < 0.9 else ((1.0, 0.0) if u < 0.99 else (0.5, 0.5))


# Retention scenarios with true mean drop exactly 0.02 = margin.
def _ret_rare_half_drops(rng: random.Random) -> Pair:  # 0.5 drop w.p. 0.04 (reviewer case)
    x = rng.choice((0.5, 1.0))
    return (x, x - 0.5 if (x == 1.0 and rng.random() < 0.08) else x)


def _ret_binary_q03_with_partial_ties(rng: random.Random) -> Pair:  # binary q=0.03 + 1% 0.5 ties
    u = rng.random()
    if u < 0.005:
        return (0.0, 1.0)
    if u < 0.03:
        return (1.0, 0.0)
    return (0.5, 0.5) if u < 0.04 else (1.0, 1.0)


def _ret_small_drops_rare_gains(rng: random.Random) -> Pair:  # -0.1 w.p. 0.4, +1 w.p. 0.02
    u = rng.random()
    return (0.8, 0.7) if u < 0.4 else ((0.0, 1.0) if u < 0.42 else (0.5, 0.5))


Gen = Callable[[random.Random], Pair]


def _simulate_pass_rate(gen: Gen, n: int, seed: int, test: Callable[[PairedScores], bool]) -> float:
    rng = random.Random(seed)
    hits = 0
    for _ in range(REPS):
        pairs = [gen(rng) for _ in range(n)]
        ps = make_scores([a for a, _ in pairs], [b for _, b in pairs])
        hits += test(ps)
    return hits / REPS


@pytest.mark.parametrize("n", [20, 50, 200])
@pytest.mark.parametrize(
    "gen",
    [_gain_partial_symmetric, _gain_skewed, _gain_small_wins_rare_full_losses],
    ids=lambda g: g.__name__,
)
def test_calibration_non_binary_gain_fractional(n: int, gen: Gen) -> None:
    """False-accept of the fractional sign test at true gain 0 (no loosened tolerance).

    Regression: the former paired-t fallback accepted 0.079 of the skewed scenario at n=50.
    """
    rate = _simulate_pass_rate(gen, n, 3000 + n, lambda ps: paired_gain_lower_bound(ps, ALPHA).passed)
    assert rate <= ALPHA + MC_TOL, f"false-accept {rate:.4f}"


@pytest.mark.parametrize("n", [50, 200, 512])
@pytest.mark.parametrize(
    "gen",
    [_ret_rare_half_drops, _ret_binary_q03_with_partial_ties, _ret_small_drops_rare_gains],
    ids=lambda g: g.__name__,
)
def test_calibration_non_binary_retention_fractional(n: int, gen: Gen) -> None:
    """False-pass of the fractional Tango test at true drop = margin (no loosened tolerance).

    Regression: the former paired-t fallback passed 0.1328 / 0.0993 / 0.0810 of the rare-drop
    scenario at n = 50 / 200 / 512, and 0.0935 of binary q=0.03 data plus one 0.5 tie at n=200.
    """
    rate = _simulate_pass_rate(
        gen, n, 4000 + n, lambda ps: noninferiority_upper_bound(ps, 0.02, ALPHA).passed
    )
    assert rate <= ALPHA + MC_TOL, f"false-pass {rate:.4f}"


def test_power_sanity_clear_gain_detected() -> None:
    rng = random.Random(7)
    hits = 0
    for _ in range(200):
        n01, n10 = draw_counts(rng, 200, 0.15, 0.03)
        hits += paired_gain_lower_bound(make_binary(n01, n10, 0, 200 - n01 - n10), ALPHA).passed
    assert hits / 200 > 0.9


# ============================================================================== sample size
def test_required_paired_n_hand_computed() -> None:
    assert required_paired_n(0.05, 0.2) == 493  # (1.6449*0.4472 + 0.8416*0.4444)^2 / 0.0025 = 492.5
    assert abs(required_paired_n(0.05, 0.2) - 490) <= 10
    assert required_paired_n(0.1, 0.3) == 184  # 183.37
    assert required_paired_n(0.05, 0.2, alpha=0.025, power=0.9) == 837  # 836.43
    assert required_paired_n(0.02, 0.1) == 1544  # 1543.54
    assert required_paired_n(0.1, 0.2) < required_paired_n(0.05, 0.2) < required_paired_n(0.03, 0.2)
    for kwargs in (
        {"delta": 0.0, "discordant_rate": 0.2},
        {"delta": 0.3, "discordant_rate": 0.2},
        {"delta": 0.05, "discordant_rate": 1.2},
        {"delta": 0.05, "discordant_rate": 0.2, "power": 1.0},
        {"delta": 0.05, "discordant_rate": 0.2, "alpha": 0.0},
    ):
        with pytest.raises(ValueError):
            required_paired_n(**kwargs)


# ============================================================================== alpha ledger
def test_alpha_ledger_per_decision() -> None:
    ledger = AlphaLedger(0.05, "per_decision")
    assert [ledger.next_alpha() for _ in range(5)] == [0.05] * 5
    assert ledger.count == 5 and ledger.spent == pytest.approx(0.25)


def test_alpha_ledger_spending_schedule_and_total() -> None:
    ledger = AlphaLedger(0.05, "spending")
    first = [ledger.next_alpha() for _ in range(3)]
    c = 0.05 * 6 / math.pi**2
    assert first == pytest.approx([c, c / 4, c / 9], rel=1e-15)
    for _ in range(200_000):
        ledger.next_alpha()
        assert ledger.spent <= ledger.total_alpha
    assert ledger.spent == pytest.approx(0.05, abs=1e-6)
    assert ledger.alpha_at(1) == pytest.approx(c)
    with pytest.raises(ValueError):
        ledger.alpha_at(0)


def test_alpha_ledger_state_round_trip_and_validation() -> None:
    ledger = AlphaLedger(0.1, "spending")
    for _ in range(7):
        ledger.next_alpha()
    state = ledger.state_dict()
    clone = AlphaLedger(0.1, "spending")
    clone.load_state_dict(state)
    assert clone.count == 7 and clone.spent == ledger.spent
    assert [clone.next_alpha() for _ in range(3)] == [ledger.next_alpha() for _ in range(3)]
    with pytest.raises(ValueError):
        AlphaLedger(0.05, "spending").load_state_dict(state)
    with pytest.raises(ValueError):
        AlphaLedger(0.1, "per_decision").load_state_dict(state)
    with pytest.raises(ValueError):
        AlphaLedger(0.05, "bonferroni")
    with pytest.raises(ValueError):
        AlphaLedger(1.5, "spending")


def test_alpha_ledger_load_is_atomic_and_consistent() -> None:
    """Regression: a failed load used to leave count half-restored; inconsistent spent was accepted."""
    base = {"total_alpha": 0.05, "schedule": "spending"}
    for schedule in ("spending", "per_decision"):
        ledger = AlphaLedger(0.05, schedule)
        for _ in range(3):
            ledger.next_alpha()
        before = ledger.state_dict()
        bad_states: list[dict[str, object]] = [
            {**base, "schedule": schedule, "count": 9, "spent": "oops"},
            {**base, "schedule": schedule, "count": 0, "spent": 3.0},  # spent > total_alpha
            {**base, "schedule": schedule, "count": 9, "spent": ledger.spent},  # count/spent disagree
            {**base, "schedule": schedule, "count": 2, "spent": math.inf},
            {**base, "schedule": schedule, "count": True, "spent": 0.0},
            {**base, "schedule": schedule, "count": 4},  # missing key
            {**base, "schedule": schedule, "count": 4, "spent": True},
        ]
        for bad in bad_states:
            with pytest.raises(ValueError):
                ledger.load_state_dict(bad)
            assert ledger.state_dict() == before
        assert ledger.expected_spent(3) == ledger.spent
    big = AlphaLedger(0.05, "spending")
    for _ in range(5000):
        big.next_alpha()
    clone = AlphaLedger(0.05, "spending")
    clone.load_state_dict(big.state_dict())
    assert clone.spent == big.spent and clone.count == 5000
    with pytest.raises(ValueError):
        clone.expected_spent(-1)
