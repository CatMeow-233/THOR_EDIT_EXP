"""Paired statistics for the capability gate (pure functions; standard library only).

Every test in this module compares an *incumbent* and a *candidate* scored on the **same** items
(:class:`~research_rl.capability.types.PairedScores`). Two questions are asked:

* transfer (:func:`paired_gain_lower_bound`): is the candidate better? H0 "candidate not better".
* retention (:func:`noninferiority_upper_bound`): did the candidate drop by less than a margin?
  H0 "drop >= margin".

Binary scores (0/1 under the strict verifier) are the main case and use exact (sign test) or
score-test (Tango 1998) procedures on the discordant pairs. Non-binary scores in ``[0, 1]`` use
the **same** procedures on *fractional discordant mass*: an item whose candidate score exceeds the
incumbent's by ``d > 0`` contributes ``d`` to the candidate-win count, one with ``d < 0``
contributes ``|d|`` to the incumbent-win count, ties contribute nothing. On binary data this is
exactly the binary procedure; a partial-credit tie (e.g. 0.5 vs 0.5) changes nothing but ``n``.
Because ``|d| <= 1`` implies ``Var(d) <= E|d| - E[d]^2``, the score variance computed from the
fractional mass is never smaller than the true variance of ``d`` (asymptotically conservative);
the price is power: a difference of 0.01 counts as 1% of a discordant pair (see the docstrings).
No paired-t / zero-variance shortcut exists, so all-tied partial-credit slices face the same
power floor as all-tied binary slices.

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
    "discordant_mass",
    "fractional_sign_test_p_value",
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


# Acklam's coefficients for the rational approximation of the inverse normal CDF.
_A: tuple[float, ...] = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B: tuple[float, ...] = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C: tuple[float, ...] = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D: tuple[float, ...] = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425


def normal_quantile(p: float) -> float:
    """Inverse standard normal CDF ``Phi^{-1}(p)`` for ``p`` in ``(0, 1)``.

    Acklam's rational approximation (relative error 1.15e-9) refined by one Halley step on
    :func:`normal_cdf`; ``p > 1/2`` is computed as ``-Phi^{-1}(1 - p)`` (exact reflection), so
    the result is antisymmetric and close to double precision (tested to 1e-12 absolute against
    reference values and the reflection). Raises ``ValueError`` outside ``(0, 1)``.
    """
    if not (math.isfinite(p) and 0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p!r}")
    if p > 0.5:
        # Reflect: 1 - p is exact for p >= 1/2, and the lower-tail refinement keeps relative
        # accuracy where a direct upper-tail Halley step would lose it (Phi(x) - p cancels).
        return -normal_quantile(1.0 - p)
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = (
            (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5])
            * q
            / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
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
    pairs. The shorter tail is summed with the exact recurrence
    ``C(m, k + 1) = C(m, k) (m - k) / (k + 1)`` (the complement is taken in integers,
    ``2^m - lower``), so the cost is ``O(m)`` big-integer steps (about 0.1 s at ``m = 20000``).
    """
    _check_count("n01", n01)
    _check_count("n10", n10)
    m = n01 + n10
    if m == 0:
        return 1.0
    if 2 * n01 >= m:
        # Upper tail sum_{k = n01}^{m} C(m, k) = sum_{j = 0}^{n10} C(m, j) by symmetry.
        numerator = _binomial_prefix_sum(m, n10)
    else:
        numerator = (1 << m) - _binomial_prefix_sum(m, n01 - 1)
    return min(1.0, numerator / (1 << m))


def _binomial_prefix_sum(m: int, last: int) -> int:
    """``sum_{j = 0}^{last} C(m, j)`` exactly (0 for ``last < 0``)."""
    total = 0
    coef = 1
    for j in range(last + 1):
        total += coef
        coef = coef * (m - j) // (j + 1)
    return total


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


# ============================================================================== fractional mass
def _check_mass(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return float(value)


def discordant_mass(paired: PairedScores) -> tuple[float, float]:
    """``(gain_mass, loss_mass)`` = ``(sum max(cand - inc, 0), sum max(inc - cand, 0))``.

    The fractional generalisation of :meth:`PairedScores.discordant` (``(n01, n10)``): on binary
    scores it returns exactly ``(float(n01), float(n10))``. Sums use :func:`math.fsum`.
    ``gain_mass + loss_mass = sum |cand - inc| <= n``.
    """
    gain = math.fsum(max(b - a, 0.0) for a, b in zip(paired.incumbent, paired.candidate))
    loss = math.fsum(max(a - b, 0.0) for a, b in zip(paired.incumbent, paired.candidate))
    return gain, loss


def fractional_sign_test_p_value(gain_mass: float, loss_mass: float) -> float:
    """Sign-test p-value extended to fractional discordant mass.

    ``I_{1/2}(gain_mass, loss_mass + 1)`` (:func:`regularized_incomplete_beta`), the continuous
    extension of ``P[X >= n01]``, ``X ~ Bin(n01 + n10, 1/2)``: for integer masses it equals
    :func:`sign_test_p_value` to ~1e-14. 1.0 when ``gain_mass == 0``. It is **not** an exact test
    for fractional mass; its validity on bounded non-binary differences rests on the variance
    argument in the module docstring and is checked by Monte-Carlo in the tests (it was at or
    below alpha in every scenario tried, typically far below: conservative).
    """
    gain = _check_mass("gain_mass", gain_mass)
    loss = _check_mass("loss_mass", loss_mass)
    if gain == 0.0:
        return 1.0
    return regularized_incomplete_beta(gain, loss + 1.0, 0.5)


def _fractional_lower_limit(gain: float, loss: float, alpha: float, passed: bool) -> float:
    """``p_L`` with ``I_{p_L}(gain, loss + 1) = alpha`` (0.0 for ``gain == 0``).

    Bracketed on ``[1/2, 1]`` when ``passed`` (the p-value at 1/2 is <= alpha) and on
    ``[0, 1/2]`` otherwise, exactly like :func:`clopper_pearson_lower`; the lower end of the
    final bracket is returned (conservative).
    """
    if gain == 0.0:
        return 0.0
    lo, hi = (0.5, 1.0) if passed else (0.0, 0.5)
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        if regularized_incomplete_beta(gain, loss + 1.0, mid) > alpha:
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
    known); only the decision is exact. Because it scales with ``m``, the bound is not monotone
    in ``n01`` at fixed ``n10`` while the test does not pass (one extra candidate win can lower a
    negative bound); it is monotone when an incumbent win turns into a candidate win, and in
    ``n01`` once the test passes. When ``m == 0`` the bound is 0.0 and ``passed`` is False.

    *Non-binary scores* (any score outside ``{0, 1}``; ``method="sign_test_fractional"``): the
    same test on fractional discordant mass (:func:`discordant_mass`): ``gain_mass`` replaces
    ``n01``, ``loss_mass`` replaces ``n10``, ``p_value`` is
    :func:`fractional_sign_test_p_value` and the bound is ``(2 p_L - 1) * m / n`` with
    ``m = gain_mass + loss_mass`` and ``p_L`` solving ``I_{p_L}(gain_mass, loss_mass + 1) = alpha``.
    Ties (also partial-credit ties) contribute nothing, so binary data plus tied partial-credit
    items gives the binary p-value. This is approximate, not exact; it is conservative in every
    Monte-Carlo scenario in the tests (true gain 0, n in {20, 50, 200}: symmetric partial credit,
    skewed +0.25 / -0.5 differences, frequent small gains with rare full losses), all within
    ``alpha + 3 MC standard errors`` without loosening. Power cost: a difference of ``d`` counts as
    ``|d|`` of a discordant pair, so many small consistent gains (all ``+0.01``) carry the
    evidence of ``0.01 n`` candidate wins; fine-grained scores need many more items than a t
    test would. There is no zero-variance shortcut (two items improved by +0.25 do not pass).

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
        bound = 0.0 if m == 0 else (2.0 * clopper_pearson_lower(n01, m, alpha) - 1.0) * m / n
        method = "sign_test_exact"
    else:
        gain, loss = discordant_mass(pooled)
        mass = gain + loss
        p_value = fractional_sign_test_p_value(gain, loss)
        passed = p_value <= alpha
        bound = (
            0.0
            if mass == 0.0
            else (2.0 * _fractional_lower_limit(gain, loss, alpha, passed) - 1.0) * mass / n
        )
        method = "sign_test_fractional"
    # Clamp floating residue onto the side the decision dictates.
    bound = max(bound, 0.0) if passed else min(bound, 0.0)
    return SliceTest(pooled.slice, n, point, bound, passed, method, alpha, p_value=p_value)


# ============================================================================== retention: Tango
def tango_restricted_mle(n10: float, n01: float, n: int, delta: float) -> float:
    """Restricted MLE of the discordant probability ``q = P(incumbent=0, candidate=1)`` under
    the constraint ``drop = P(inc=1, cand=0) - P(inc=0, cand=1) = delta`` (``delta`` in [0, 1]).

    Tango (1998, Stat. Med. 17:891-908; Nam 1997): with ``b = n10``, ``c = n01``, the constrained
    log-likelihood ``b log(q + delta) + c log q + (n - b - c) log(1 - 2q - delta)`` is maximised
    at the non-negative root of ``A q^2 + B q + C = 0`` with ``A = 2n``,
    ``B = -b - c + (2n - b + c) delta`` and ``C = -c delta (1 - delta)``:
    ``q = (sqrt(B^2 - 4AC) - B) / (2A)``. The root lies in ``[0, (1 - delta) / 2]`` (also when
    every pair is discordant). For negative ``delta`` use the symmetry in :func:`tango_z`.
    ``n10`` / ``n01`` may be fractional discordant mass (:func:`discordant_mass`); the same
    maximisation applies to the pseudo-likelihood (``n10 + n01 <= n`` up to 1e-9 relative).
    """
    b = _check_mass("n10", n10)
    c = _check_mass("n01", n01)
    _check_count("n", n)
    if n == 0 or b + c > n * (1.0 + 1e-9):
        raise ValueError(f"need n >= 1 and n10 + n01 <= n (got n10={n10}, n01={n01}, n={n})")
    if not (0.0 <= delta <= 1.0):
        raise ValueError(f"delta must be in [0, 1] here, got {delta!r}")
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


def tango_z(n10: float, n01: float, n: int, delta: float) -> float:
    """Tango's score statistic for H0 ``drop = delta`` (``drop = p10 - p01``, ``delta`` in [-1, 1]).

    ``n10`` / ``n01`` are discordant counts or fractional discordant mass (see
    :func:`tango_restricted_mle`).

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

    *Non-binary scores* (any score outside ``{0, 1}``; ``method="tango_score_fractional"``): the
    same score test with the fractional discordant mass of :func:`discordant_mass`
    (``n10 = loss_mass``, ``n01 = gain_mass``). On binary data this is Tango's statistic; a
    partial-credit tie only adds 1 to ``n``, so a binary slice with one tied 0.5 item gets the
    same p-value as the binary slice with one extra concordant pair. Since ``|d| <= 1`` gives
    ``Var(d) <= E|d| - margin^2`` under H0, the restricted variance estimate is asymptotically
    conservative. Measured by Monte-Carlo at true drop = margin = 0.02 (tests), n in
    {50, 200, 512}: rare 0.5 drops (probability 0.04); 0.1 drops with probability 0.4 plus full
    candidate-only wins with probability 0.02; binary discordance q = 0.03 with 1% tied 0.5
    items. All stay within ``alpha + 3 MC standard errors`` without loosening. There is no
    zero-variance shortcut. Power cost: a drop of ``d`` counts as ``|d|`` of a discordant pair.

    Power floor (binary and fractional alike): with no discordant mass a slice passes iff
    ``n * margin / (1 - margin) >= z_{1-alpha}^2``, so small slices can never pass (see
    :func:`retention_power_floor_n`).

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
    n10: float
    n01: float
    if pooled.is_binary:
        n01_int, n10_int = pooled.discordant()
        n01, n10 = float(n01_int), float(n10_int)
        method = "tango_score"
    else:
        n01, n10 = discordant_mass(pooled)
        method = "tango_score_fractional"
    p_value = normal_cdf(tango_z(n10, n01, n, margin))
    passed = p_value <= alpha

    def rejected(d: float) -> bool:
        return normal_cdf(tango_z(n10, n01, n, d)) <= alpha

    if passed:
        bound = _bisect_last_accepted(rejected, -1.0, margin, lo_default=-1.0)
    elif not rejected(1.0):
        bound = 1.0
    else:
        bound = _bisect_last_accepted(rejected, margin, 1.0, lo_default=margin)
    return SliceTest(pooled.slice, n, point, bound, passed, method, alpha, p_value=p_value, margin=margin)


def _bisect_last_accepted(
    rejected: Callable[[float], bool], lo: float, hi: float, *, lo_default: float
) -> float:
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

    def expected_spent(self, count: int) -> float:
        """``sum_{k=1}^{count} alpha_at(k)``, accumulated in the same order as :meth:`next_alpha`
        (so it reproduces ``spent`` of a ledger that handed out ``count`` alphas bit for bit)."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"count must be an int >= 0, got {count!r}")
        total = 0.0
        for k in range(1, count + 1):
            total += self.alpha_at(k)
        return total

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the position; atomic (nothing changes unless every check passes).

        ``total_alpha`` and ``schedule`` must match this ledger's, ``count`` must be an int >= 0
        and ``spent`` a finite number equal to :meth:`expected_spent` of ``count`` within
        ``1e-12 + 1e-9 * expected`` (so a state can never claim ``spent > total_alpha`` under
        ``"spending"`` or disagree with its own count). The stored ``spent`` is kept as is.
        Missing keys or malformed values raise ``ValueError``. Cost ``O(count)``.
        """
        try:
            total_alpha = float(state["total_alpha"])
            schedule = state["schedule"]
            count = state["count"]
            spent_raw = state["spent"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed AlphaLedger state: {exc!r}") from exc
        if total_alpha != self.total_alpha or schedule != self.schedule:
            raise ValueError(
                f"ledger mismatch: state has total_alpha={total_alpha!r}, "
                f"schedule={schedule!r}; this ledger has {self.total_alpha!r}, {self.schedule!r}"
            )
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"invalid count {count!r}")
        if (
            isinstance(spent_raw, bool)
            or not isinstance(spent_raw, (int, float))
            or not math.isfinite(spent_raw)
        ):
            raise ValueError(f"invalid spent {spent_raw!r}")
        spent = float(spent_raw)
        expected = self.expected_spent(count)
        if abs(spent - expected) > 1e-12 + 1e-9 * expected:
            raise ValueError(
                f"inconsistent ledger state: spent={spent!r} but count={count} implies {expected!r}"
            )
        self._count = count
        self._spent = spent

    def __repr__(self) -> str:
        return (
            f"AlphaLedger(total_alpha={self.total_alpha}, schedule={self.schedule!r}, "
            f"count={self._count}, spent={self._spent:.6g})"
        )
