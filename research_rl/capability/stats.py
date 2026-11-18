"""Paired statistics for the capability gate (pure functions; standard library only).

Every test in this module compares an *incumbent* and a *candidate* scored on the **same** items
(:class:`~research_rl.capability.types.PairedScores`). Two questions are asked:

* transfer (:func:`paired_gain_lower_bound`): is the candidate better? H0 "candidate not better".
* retention (:func:`noninferiority_upper_bound`): did the candidate drop by less than a margin?
  H0 "drop >= margin".

Binary scores (0/1 under the strict verifier) are the main case and use exact (sign test) or
score-test (Tango 1998) procedures on the discordant pairs. Non-binary scores fall back to a
one-sided paired Student-t bound, which is an approximation (see the docstrings for limits).

Numerical building blocks are implemented here because scipy is not a dependency:

* :func:`normal_cdf` (via :func:`math.erfc`) and :func:`normal_quantile` (Acklam's rational
  approximation followed by one Halley refinement step; absolute error < 1e-12 on
  ``[1e-12, 1 - 1e-12]``).
* :func:`regularized_incomplete_beta` (modified Lentz continued fraction, Numerical Recipes
  ``betacf``) used for the Clopper-Pearson bound and the Student-t distribution.

Agreement between ``passed`` and ``bound`` is guaranteed by construction: every bound is found by
a bisection bracketed on the side the exact decision dictates, so for the transfer test
``passed`` implies ``bound >= 0`` (equality only when the p-value equals alpha to floating
precision) and ``not passed`` implies ``bound <= 0``; for a retention test ``passed`` implies
``bound <= margin`` and ``not passed`` implies ``bound >= margin``. Callers should branch on
``SliceTest.passed``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import Any

from research_rl.capability.types import PairedScores, SliceTest

__all__ = [
    "AlphaLedger",
    "binomial_upper_tail",
    "clopper_pearson_lower",
    "noninferiority_upper_bound",
    "normal_cdf",
    "normal_quantile",
    "paired_gain_lower_bound",
    "pool_paired",
    "regularized_incomplete_beta",
    "required_paired_n",
    "retention_power_floor_n",
    "sign_test_p_value",
    "student_t_quantile",
    "student_t_sf",
    "tango_restricted_mle",
    "tango_z",
]

_BISECTION_STEPS = 200


# ============================================================================== validation helpers
def _check_alpha(alpha: float) -> None:
    if not (isinstance(alpha, (int, float)) and math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")


def _check_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")


# ============================================================================== normal distribution
def normal_cdf(x: float) -> float:
    """Standard normal CDF ``Phi(x) = erfc(-x / sqrt(2)) / 2`` (accurate in both tails)."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# Acklam's coefficients (https://web.archive.org/web/2015/http://home.online.no/~pjacklam/notes/invnorm/).
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02)
_A += (-3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01)
_B += (-1.328068155288572e01,)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00)
_C += (4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
_P_LOW = 0.02425


def normal_quantile(p: float) -> float:
    """Inverse standard normal CDF ``Phi^{-1}(p)`` for ``p`` in ``(0, 1)``.

    Acklam's rational approximation (relative error 1.15e-9) refined by one Halley step on
    :func:`normal_cdf`, which brings the error to the level of double precision for
    ``p`` in ``[1e-12, 1 - 1e-12]``. Raises ``ValueError`` outside ``(0, 1)``.
    """
    if not (math.isfinite(p) and 0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p!r}")
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
            ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        )
    else:
        q = math.sqrt(-2.0 * math.log1p(-p))
        x = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    # One Halley step: e = Phi(x) - p, u = e * sqrt(2 pi) * exp(x^2 / 2).
    e = normal_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


# ============================================================================== incomplete beta
def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz, NR ``betacf``)."""
    tiny = 1e-300
    eps = 1e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    max_iter = 100_000
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise ArithmeticError(f"incomplete beta continued fraction did not converge (a={a}, b={b}, x={x})")


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """``I_x(a, b)`` for ``a, b > 0`` and ``x`` in ``[0, 1]`` (relative accuracy ~1e-14).

    Uses the continued fraction directly for ``x < (a + 1) / (a + b + 2)`` and the symmetry
    ``I_x(a, b) = 1 - I_{1-x}(b, a)`` otherwise, so the fraction converges in ``O(sqrt(max(a, b)))``
    iterations.
    """
    if not (a > 0.0 and b > 0.0):
        raise ValueError(f"a and b must be > 0, got a={a!r}, b={b!r}")
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"x must be in [0, 1], got {x!r}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _beta_continued_fraction(a, b, x) / a
    return 1.0 - math.exp(log_front) * _beta_continued_fraction(b, a, 1.0 - x) / b


# ============================================================================== binomial
def binomial_upper_tail(x: int, m: int, p: float) -> float:
    """``P[X >= x]`` for ``X ~ Bin(m, p)``.

    Exact boundary cases: 1.0 for ``x <= 0``, 0.0 for ``x > m``, and the degenerate ``p`` in
    ``{0, 1}``. Otherwise ``I_p(x, m - x + 1)`` (:func:`regularized_incomplete_beta`).
    """
    _check_count("m", m)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    if x <= 0:
        return 1.0
    if x > m:
        return 0.0
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    return regularized_incomplete_beta(float(x), float(m - x + 1), p)


def sign_test_p_value(n01: int, n10: int) -> float:
    """Exact one-sided sign-test p-value of H0 "candidate not better".

    ``n01`` = pairs the candidate solved and the incumbent did not, ``n10`` = the reverse.
    Returns ``P[X >= n01]`` for ``X ~ Bin(n01 + n10, 1/2)``, computed exactly with integer
    binomial coefficients and one correctly rounded division; 1.0 when there are no discordant
    pairs.
    """
    _check_count("n01", n01)
    _check_count("n10", n10)
    m = n01 + n10
    if m == 0:
        return 1.0
    numerator = sum(math.comb(m, k) for k in range(n01, m + 1))
    return min(1.0, numerator / (1 << m))


def clopper_pearson_lower(x: int, m: int, alpha: float) -> float:
    """One-sided ``(1 - alpha)`` Clopper-Pearson lower confidence limit of a binomial proportion.

    The limit ``p_L`` solves ``P_{p_L}[X >= x] = alpha`` for ``X ~ Bin(m, p_L)`` (0.0 when
    ``x == 0``), found by bisection on :func:`binomial_upper_tail` (200 halvings). The bracket
    is ``[1/2, 1]`` when ``P_{1/2}[X >= x] <= alpha`` (computed exactly by
    :func:`sign_test_p_value`) and ``[0, 1/2]`` otherwise, so ``p_L >= 1/2`` holds exactly iff
    the exact sign test rejects at ``alpha``. The lower end of the final bracket is returned
    (conservative). ``m == 0`` raises ``ValueError``.
    """
    _check_count("x", x)
    _check_count("m", m)
    _check_alpha(alpha)
    if m == 0:
        raise ValueError("the Clopper-Pearson bound needs m >= 1 trials")
    if x > m:
        raise ValueError(f"x={x} exceeds m={m}")
    if x == 0:
        return 0.0
    if sign_test_p_value(x, m - x) <= alpha:
        lo, hi = 0.5, 1.0
    else:
        lo, hi = 0.0, 0.5
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        if binomial_upper_tail(x, m, mid) > alpha:
            hi = mid
        else:
            lo = mid
    return lo


# ============================================================================== Student t
def student_t_sf(t: float, df: float) -> float:
    """Upper tail ``P[T >= t]`` of Student's t with ``df > 0`` degrees of freedom."""
    if not df > 0.0:
        raise ValueError(f"df must be > 0, got {df!r}")
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * regularized_incomplete_beta(0.5 * df, 0.5, df / (df + t * t))
    return tail if t >= 0.0 else 1.0 - tail


@lru_cache(maxsize=1024)
def student_t_quantile(p: float, df: float) -> float:
    """``t`` with ``P[T <= t] = p`` for Student's t (bisection on :func:`student_t_sf`)."""
    if not (math.isfinite(p) and 0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p!r}")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -student_t_quantile(1.0 - p, df)
    target = 1.0 - p  # upper tail mass
    lo, hi = 0.0, 1.0
    while student_t_sf(hi, df) > target:
        hi *= 2.0
        if hi > 1e300:
            raise ArithmeticError("student_t_quantile: no finite upper bracket")
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        if student_t_sf(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return hi


# ============================================================================== pooling
def pool_paired(paired: PairedScores | Sequence[PairedScores]) -> PairedScores:
    """Concatenate the items of several slices into one :class:`PairedScores`.

    A single :class:`PairedScores` is returned unchanged. For a sequence, items are concatenated
    in order; the pooled slice name is the slice names joined by ``"+"`` and the pooled ``split``
    is the first slice's split. Uids must be unique across slices (the pooled ``PairedScores``
    raises ``ValueError`` on duplicates, so no item is counted twice). An empty sequence raises
    ``ValueError`` (there is nothing to name); use ``n == 0`` evidence to express "no items".
    """
    if isinstance(paired, PairedScores):
        return paired
    parts = list(paired)
    if not parts:
        raise ValueError("cannot pool an empty sequence of PairedScores")
    if len(parts) == 1:
        return parts[0]
    uids: list[str] = []
    seen: dict[str, str] = {}
    for p in parts:
        for uid in p.uids:
            if uid in seen:
                raise ValueError(f"uid {uid!r} appears in slices {seen[uid]!r} and {p.slice!r}")
            seen[uid] = p.slice
        uids.extend(p.uids)
    return PairedScores(
        slice="+".join(p.slice for p in parts),
        split=parts[0].split,
        uids=tuple(uids),
        incumbent=tuple(v for p in parts for v in p.incumbent),
        candidate=tuple(v for p in parts for v in p.candidate),
    )


def _diff_stats(differences: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (n - 1 denominator) via :func:`math.fsum`."""
    n = len(differences)
    mean = math.fsum(differences) / n
    var = math.fsum((d - mean) ** 2 for d in differences) / (n - 1)
    return mean, math.sqrt(max(var, 0.0))


# ============================================================================== transfer: gain
def paired_gain_lower_bound(paired: PairedScores | Sequence[PairedScores], alpha: float) -> SliceTest:
    """One-sided test of H0 "candidate not better" with a lower confidence bound on the gain.

    A sequence of slices is pooled first (:func:`pool_paired`). ``point`` is the mean gain
    ``mean(candidate - incumbent)``.

    *Binary scores* (``method="sign_test_exact"``): with ``n01`` candidate-only wins, ``n10``
    incumbent-only wins and ``m = n01 + n10`` discordant pairs, ``p_value`` is the exact
    :func:`sign_test_p_value` and ``passed = p_value <= alpha`` (exact level-alpha test). The
    bound is ``(2 p_L - 1) * m / n`` with ``p_L`` the one-sided Clopper-Pearson lower limit of
    ``n01`` successes in ``m`` trials (:func:`clopper_pearson_lower`). The bound is
    **conditional on the observed number of discordant pairs** ``m`` (it treats ``m / n`` as
    known); only the decision is exact. When ``m == 0`` the bound is 0.0 and ``passed`` is False.

    *Non-binary scores* (``method="paired_t"``): one-sided Student-t lower bound
    ``mean - t_{1-alpha, n-1} * sd / sqrt(n)`` of the per-item differences; ``passed`` iff the
    bound is > 0 and ``p_value = P[T_{n-1} >= mean / (sd / sqrt(n))]``. This is approximate (it
    assumes the mean difference is close to normal); its false-accept rate is checked by
    simulation in the tests for n in {50, 200}. With zero variance the bound is the mean itself
    (``passed`` iff the common difference is > 0). ``n < 2`` gives ``bound = -1`` (the worst
    possible gain), ``passed=False``, ``method="insufficient"``.

    ``n == 0`` gives ``point=0, bound=-1, passed=False, method="empty"``. Consistency:
    ``passed`` implies ``bound >= 0`` (``> 0`` except when the p-value equals alpha to floating
    precision) and ``not passed`` implies ``bound <= 0``.
    """
    _check_alpha(alpha)
    pooled = pool_paired(paired)
    n = pooled.n
    if n == 0:
        return SliceTest(pooled.slice, 0, 0.0, -1.0, False, "empty", alpha, p_value=1.0)
    point = pooled.mean_difference()
    if pooled.is_binary:
        n01, n10 = pooled.discordant()
        m = n01 + n10
        p_value = sign_test_p_value(n01, n10)
        passed = p_value <= alpha
        if m == 0:
            bound = 0.0
        else:
            p_low = clopper_pearson_lower(n01, m, alpha)
            bound = (2.0 * p_low - 1.0) * m / n
            # Clamp floating residue onto the side the exact decision dictates.
            bound = max(bound, 0.0) if passed else min(bound, 0.0)
        return SliceTest(pooled.slice, n, point, bound, passed, "sign_test_exact", alpha, p_value=p_value)
    if n < 2:
        return SliceTest(pooled.slice, n, point, -1.0, False, "insufficient", alpha, p_value=1.0)
    diffs = [b - a for a, b in zip(pooled.incumbent, pooled.candidate)]
    mean, sd = _diff_stats(diffs)
    if sd == 0.0:
        passed = mean > 0.0
        p_value = 0.0 if passed else 1.0
        return SliceTest(pooled.slice, n, point, mean, passed, "paired_t", alpha, p_value=p_value)
    se = sd / math.sqrt(n)
    df = float(n - 1)
    bound = mean - student_t_quantile(1.0 - alpha, df) * se
    p_value = student_t_sf(mean / se, df)
    return SliceTest(pooled.slice, n, point, bound, bound > 0.0, "paired_t", alpha, p_value=p_value)


# ============================================================================== retention: Tango
def tango_restricted_mle(n10: int, n01: int, n: int, delta: float) -> float:
    """Restricted MLE of the discordant probability ``q = P(incumbent=0, candidate=1)`` under
    the constraint ``drop = P(inc=1, cand=0) - P(inc=0, cand=1) = delta`` (``delta`` in [0, 1]).

    Tango (1998, Stat. Med. 17:891-908; Nam 1997): with ``b = n10``, ``c = n01``, the constrained
    log-likelihood ``b log(q + delta) + c log q + (n - b - c) log(1 - 2q - delta)`` is maximised
    at the non-negative root of ``A q^2 + B q + C = 0`` with ``A = 2n``,
    ``B = -b - c + (2n - b + c) delta`` and ``C = -c delta (1 - delta)``:
    ``q = (sqrt(B^2 - 4AC) - B) / (2A)``. The root lies in ``[0, (1 - delta) / 2]`` (also when
    every pair is discordant). For negative ``delta`` use the symmetry in :func:`tango_z`.
    """
    _check_count("n10", n10)
    _check_count("n01", n01)
    _check_count("n", n)
    if n == 0 or n10 + n01 > n:
        raise ValueError(f"need n >= 1 and n10 + n01 <= n (got n10={n10}, n01={n01}, n={n})")
    if not (0.0 <= delta <= 1.0):
        raise ValueError(f"delta must be in [0, 1] here, got {delta!r}")
    b, c = float(n10), float(n01)
    a_coef = 2.0 * n
    b_coef = -b - c + (2.0 * n - b + c) * delta
    c_coef = -c * delta * (1.0 - delta)
    disc = b_coef * b_coef - 4.0 * a_coef * c_coef
    root = math.sqrt(max(disc, 0.0))
    # Numerically stable form of the non-negative root.
    if b_coef <= 0.0:
        q = (root - b_coef) / (2.0 * a_coef)
    else:
        q = (-2.0 * c_coef) / (root + b_coef) if root + b_coef > 0.0 else 0.0
    return min(max(q, 0.0), max(0.0, (1.0 - delta) / 2.0))


def tango_z(n10: int, n01: int, n: int, delta: float) -> float:
    """Tango's score statistic for H0 ``drop = delta`` (``drop = p10 - p01``, ``delta`` in [-1, 1]).

    ``Z = (n10 - n01 - n delta) / sqrt(n (2 q~ + delta (1 - delta)))`` with ``q~`` from
    :func:`tango_restricted_mle`. For ``delta < 0`` the roles of the discordant cells are swapped:
    ``Z(n10, n01, delta) = -Z(n01, n10, -delta)``. A zero variance (only possible at
    ``|delta| = 1`` or ``delta = 0`` without discordant pairs) yields ``+inf`` / ``-inf`` by the
    sign of the numerator, or 0.0 when the numerator is zero too. Small Z = evidence for a
    smaller drop.
    """
    if not (-1.0 <= delta <= 1.0):
        raise ValueError(f"delta must be in [-1, 1], got {delta!r}")
    if delta < 0.0:
        return -tango_z(n01, n10, n, -delta)
    q = tango_restricted_mle(n10, n01, n, delta)
    numerator = n10 - n01 - n * delta
    variance = n * (2.0 * q + delta * (1.0 - delta))
    if variance <= 0.0:
        if numerator > 0.0:
            return math.inf
        if numerator < 0.0:
            return -math.inf
        return 0.0
    return numerator / math.sqrt(variance)


def noninferiority_upper_bound(
    paired: PairedScores | Sequence[PairedScores], margin: float, alpha: float
) -> SliceTest:
    """Non-inferiority test of H0 "drop >= margin" with an upper confidence bound on the drop.

    ``drop = mean(incumbent - candidate)`` is ``point``. A sequence of slices is pooled first.

    *Binary scores* (``method="tango_score"``): Tango's score test for the paired difference of
    proportions. ``p_value = Phi(Z)`` with ``Z = tango_z(n10, n01, n, margin)``; ``passed`` iff
    ``p_value <= alpha`` (H0 rejected). ``bound`` is the upper limit of the one-sided
    ``(1 - alpha)`` score interval: the drop ``d`` where ``Z(d) = -z_{1-alpha}``, found by
    bisection bracketed on ``[-1, margin]`` when passed and ``[margin, 1]`` otherwise (1.0 when
    no ``d <= 1`` is rejected). The test is asymptotic; its exact size at the calibration settings
    is checked in the tests (it can slightly exceed alpha at small n and low discordance).
    With no discordant pairs a slice passes iff ``n * margin / (1 - margin) >= z_{1-alpha}^2``,
    i.e. small slices can never pass (a power floor, not a bug).

    *Non-binary scores* (``method="paired_t"``): one-sided Student-t upper bound
    ``mean_drop + t_{1-alpha, n-1} * sd / sqrt(n)``; ``passed`` iff ``bound < margin``;
    ``p_value = P[T_{n-1} <= (mean_drop - margin) / (sd / sqrt(n))]``. Approximate (see
    :func:`paired_gain_lower_bound`). Zero variance: ``bound = mean_drop``, passed iff it is
    < margin. ``n < 2`` gives ``bound = 1``, ``passed=False``, ``method="insufficient"``.

    ``margin`` must be in ``[0, 1)``. ``n == 0`` gives ``point=0, bound=1, passed=False,
    method="empty"``.
    """
    _check_alpha(alpha)
    if not (isinstance(margin, (int, float)) and math.isfinite(margin) and 0.0 <= margin < 1.0):
        raise ValueError(f"margin must be in [0, 1), got {margin!r}")
    margin = float(margin)
    pooled = pool_paired(paired)
    n = pooled.n
    if n == 0:
        return SliceTest(pooled.slice, 0, 0.0, 1.0, False, "empty", alpha, p_value=1.0, margin=margin)
    point = -pooled.mean_difference()
    if pooled.is_binary:
        n01, n10 = pooled.discordant()
        z = tango_z(n10, n01, n, margin)
        p_value = normal_cdf(z)
        passed = p_value <= alpha

        def rejected(d: float) -> bool:
            return normal_cdf(tango_z(n10, n01, n, d)) <= alpha

        if passed:
            bound = _bisect_last_accepted(rejected, -1.0, margin, lo_default=-1.0)
        elif not rejected(1.0):
            bound = 1.0
        else:
            bound = _bisect_last_accepted(rejected, margin, 1.0, lo_default=margin)
        return SliceTest(
            pooled.slice, n, point, bound, passed, "tango_score", alpha, p_value=p_value, margin=margin
        )
    if n < 2:
        return SliceTest(pooled.slice, n, point, 1.0, False, "insufficient", alpha, p_value=1.0, margin=margin)
    drops = [a - b for a, b in zip(pooled.incumbent, pooled.candidate)]
    mean, sd = _diff_stats(drops)
    if sd == 0.0:
        passed = mean < margin
        return SliceTest(
            pooled.slice,
            n,
            point,
            mean,
            passed,
            "paired_t",
            alpha,
            p_value=0.0 if passed else 1.0,
            margin=margin,
        )
    se = sd / math.sqrt(n)
    df = float(n - 1)
    bound = mean + student_t_quantile(1.0 - alpha, df) * se
    p_value = 1.0 - student_t_sf((mean - margin) / se, df)
    return SliceTest(
        pooled.slice, n, point, bound, bound < margin, "paired_t", alpha, p_value=p_value, margin=margin
    )


def _bisect_last_accepted(rejected: Callable[[float], bool], lo: float, hi: float, *, lo_default: float) -> float:
    """Boundary between not-rejected values (``<= boundary``) and rejected values in ``[lo, hi]``.

    Precondition: ``rejected(hi)`` holds. If ``rejected(lo)`` also holds, ``lo_default`` is
    returned. Returns the upper end of the final bracket (conservative for an upper bound).
    """
    if rejected(lo):
        return lo_default
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        if rejected(mid):
            hi = mid
        else:
            lo = mid
    return hi


# ============================================================================== sample size
def required_paired_n(delta: float, discordant_rate: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Pairs needed to detect a paired gain ``delta`` (Connor 1987, one-sided).

    ``n = ceil((z_a sqrt(q) + z_b sqrt(q - delta^2))^2 / delta^2)`` with ``q`` the discordance
    rate ``P(inc != cand)``, ``z_a = Phi^{-1}(1 - alpha)`` (**one-sided**, matching the gate's
    tests; a two-sided design would use ``alpha / 2``) and ``z_b = Phi^{-1}(power)``. Requires
    ``0 < delta <= q <= 1`` (a gain cannot exceed the discordance rate), ``alpha`` in (0, 1) and
    ``power`` in (0, 1). Example: ``delta=0.05, q=0.2`` gives 493.
    """
    _check_alpha(alpha)
    if not (math.isfinite(power) and 0.0 < power < 1.0):
        raise ValueError(f"power must be in (0, 1), got {power!r}")
    if not (math.isfinite(delta) and delta > 0.0):
        raise ValueError(f"delta must be > 0, got {delta!r}")
    if not (math.isfinite(discordant_rate) and delta <= discordant_rate <= 1.0):
        raise ValueError(f"discordant_rate must be in [delta, 1], got {discordant_rate!r}")
    q = discordant_rate
    z_a = normal_quantile(1.0 - alpha)
    z_b = normal_quantile(power)
    n = (z_a * math.sqrt(q) + z_b * math.sqrt(max(q - delta * delta, 0.0))) ** 2 / (delta * delta)
    # Guard against values like 492.0000000001 produced by rounding of an exact integer.
    return max(1, math.ceil(n - 1e-9))


def retention_power_floor_n(margin: float, alpha: float) -> int:
    """Smallest slice size at which a binary retention slice can pass at all.

    With zero discordant pairs (the best possible evidence) Tango's statistic at ``margin`` is
    ``Z = -sqrt(n margin / (1 - margin))``, so :func:`noninferiority_upper_bound` can pass only if
    ``n >= z_{1-alpha}^2 (1 - margin) / margin``. Returns that ``n`` (ceil). Example: margin 0.02
    at the first ``"spending"`` alpha ``0.05 * 6 / pi^2`` needs 173 items; at the 10th decision
    576. Requires ``margin`` in ``(0, 1)``.
    """
    _check_alpha(alpha)
    if not (math.isfinite(margin) and 0.0 < margin < 1.0):
        raise ValueError(f"margin must be in (0, 1), got {margin!r}")
    z = normal_quantile(1.0 - alpha)
    n = math.ceil(z * z * (1.0 - margin) / margin - 1e-9)
    # Guard the rounding at the boundary with the statistic itself.
    while n > 1 and normal_cdf(-math.sqrt((n - 1) * margin / (1.0 - margin))) <= alpha:
        n -= 1
    while normal_cdf(-math.sqrt(n * margin / (1.0 - margin))) > alpha:
        n += 1
    return max(1, n)


# ============================================================================== alpha ledger
class AlphaLedger:
    """Hands out the significance level of each gate decision.

    Schedules:

    * ``"per_decision"``: every call of :meth:`next_alpha` returns ``total_alpha``. The type-I
      guarantee holds **per decision only**; over K decisions up to ``K * total_alpha`` false
      accepts are expected under H0. ``spent`` then grows without bound (bookkeeping only).
    * ``"spending"``: the k-th call (k = 1, 2, ...) returns ``total_alpha * 6 / (pi^2 k^2)``.
      Because ``sum_k 6 / (pi^2 k^2) = 1``, ``spent`` stays below ``total_alpha`` for any number of
      decisions and the Bonferroni argument gives a run-level family-wise guarantee for accepts:
      P[any false accept] <= total_alpha.

    Every test of one decision (the pooled transfer test and each retention slice) uses the same
    alpha: the accept rule is an intersection of rejections (intersection-union), which needs no
    split. Separate re-tests inside one gate procedure (shrink / group re-tests) each draw their
    own alpha.
    """

    SCHEDULES = ("per_decision", "spending")

    def __init__(self, total_alpha: float = 0.05, schedule: str = "spending") -> None:
        _check_alpha(total_alpha)
        if schedule not in self.SCHEDULES:
            raise ValueError(f"schedule must be one of {self.SCHEDULES}, got {schedule!r}")
        self.total_alpha = float(total_alpha)
        self.schedule = schedule
        self._count = 0
        self._spent = 0.0

    @property
    def count(self) -> int:
        """Number of alphas handed out so far."""
        return self._count

    @property
    def spent(self) -> float:
        """Sum of all alphas handed out so far."""
        return self._spent

    def alpha_at(self, k: int) -> float:
        """Alpha of the k-th decision (1-based) without consuming it."""
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be an int >= 1, got {k!r}")
        if self.schedule == "per_decision":
            return self.total_alpha
        return self.total_alpha * 6.0 / (math.pi**2 * k * k)

    def next_alpha(self) -> float:
        """Consume and return the next decision's alpha."""
        alpha = self.alpha_at(self._count + 1)
        self._count += 1
        self._spent += alpha
        return alpha

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_alpha": self.total_alpha,
            "schedule": self.schedule,
            "count": self._count,
            "spent": self._spent,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the position. ``total_alpha`` and ``schedule`` must match this ledger's."""
        if float(state["total_alpha"]) != self.total_alpha or state["schedule"] != self.schedule:
            raise ValueError(
                f"ledger mismatch: state has total_alpha={state['total_alpha']!r}, "
                f"schedule={state['schedule']!r}; this ledger has {self.total_alpha!r}, {self.schedule!r}"
            )
        count = state["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"invalid count {count!r}")
        self._count = count
        self._spent = float(state["spent"])

    def __repr__(self) -> str:
        return (
            f"AlphaLedger(total_alpha={self.total_alpha}, schedule={self.schedule!r}, "
            f"count={self._count}, spent={self._spent:.6g})"
        )
