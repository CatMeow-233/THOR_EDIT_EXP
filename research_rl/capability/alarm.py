"""Tier-1 verifier-disagreement alarm, rollout screening and the static leak audit.

The training reward comes from a lenient verifier V1; capability is measured with a strict
verifier V2. A reward hack that exploits V1 shows up as a *growing* share of rollouts that pass V1
but fail V2. This module watches that gap on the training rollouts themselves (no extra
generation), so it reacts within a few updates, before the next gate decision.

:class:`DisagreementAlarm` (one call of :meth:`~DisagreementAlarm.update` per operator update)
---------------------------------------------------------------------------------------------

* **Statistic.** For update ``t`` with ``n_t`` rollouts, ``g(t) = pass_V1(t) - pass_V2(t)`` (the
  share of rollouts V1 passes minus the share V2 passes, i.e. the V1-only share minus the
  V2-only share, pooled over all rollouts of the report).
* **Independent units and eligibility.** The noise of ``g(t)`` depends on how many *independent*
  outcomes the report holds, not on how many rollouts. Rollouts that share a prompt/item (GRPO
  groups; ES members scored on the same items with common random numbers, near-copies under
  greedy decoding) are correlated: with groups of ``G`` rollouts and intra-group correlation
  ``rho`` the variance of ``g`` is inflated by the design effect ``1 + (G - 1) * rho`` (up to
  ``G`` for exact replicates). The alarm therefore counts the Kish effective size
  ``n_eff = n_t**2 / sum_j s_j**2`` where ``s_j`` is the number of rollouts of unit ``j``:
  ``n_eff = n_t`` for independent rollouts, the number of units for equal-size units. It is exact
  for exact replicates and conservative (never larger than the true effective size) for
  independent units with ``0 <= rho <= 1`` inside each unit. Units come from, in this order:
  ``report.extra[UNITS_KEY]`` (``"alarm_units"``; one int or str id per rollout, same length as
  ``v1_pass``; rollouts with equal ids form one unit), else the constructor's
  ``rollouts_per_unit`` (every unit has exactly that many rollouts, so ``n_eff = n_t /
  rollouts_per_unit``; ``n_t`` must be a multiple of it; e.g. the ES population size when every
  member scores each item once, or the GRPO group size), else every rollout is its own unit.
  **Without unit information a clustered feed violates the independence assumption and the
  documented false-alarm level does not hold** (simulation: an ES-like feed of 8 members x 4
  shared items alarms on 40-94 % of clean 200-update streams, the more the members replicate
  each other's outcomes, e.g. under greedy decoding).
  Reports with ``n_eff < min_rollouts`` are *skipped*: they change no statistic (CUSUM,
  baselines, length window, counters other than ``updates``, ``skipped_updates`` and, in latched
  mode, ``gap_alarms``). Consequence: with unit information an ES feed of the default
  population 8 scored on 4 items per update (``n_eff = 4``) is always skipped, i.e. both the
  gap alarm and the length-collapse flag are silent (check ``skipped_updates``);
  it needs ``min_rollouts`` independent items per update.
  :class:`~research_rl.capability.es.ESOperator` reports carry no unit ids, so for an ES feed
  the caller must construct the alarm with ``rollouts_per_unit=population`` (one rollout per
  item per member) or add the ids itself.
* **Baseline.** ``b`` = the pooled gap of the first ``baseline_updates`` eligible reports
  (``sum(V1 passes - V2 passes) / sum(rollouts)``), or the explicit ``baseline_gap`` (e.g.
  :attr:`LeakAudit.gap` of a :func:`static_leak_audit` of the start checkpoint; note that this is
  ``v1_pass_rate - v2_pass_rate``, the same quantity as ``g(t)``, not ``leak_rate``). While the
  baseline is being estimated the CUSUM does not run and the alarm cannot fire on the gap.
* **CUSUM.** One-sided upper CUSUM on the per-update gap, in absolute pass-rate units:
  ``S(t) = max(0, S(t-1) + g(t) - b - drift)``; it fires when ``S(t) > threshold``. ``drift`` is
  the allowance (an excess gap below ``drift`` never accumulates). With ``reset_on_alarm=True``
  (default) ``S`` is set to 0 right after it fires, so the alarm is a one-update pulse: a
  persistent leak fires again after it has re-accumulated ``threshold``; once the stream is back
  at its baseline (e.g. after a rollback) the alarm stays off. With ``reset_on_alarm=False`` the
  alarm stays on for as long as ``S > threshold`` (it then decays by about ``b + drift - g`` per
  update once the gap is back at baseline).
* **Length collapse.** The mean completion length of the last ``window`` eligible reports that
  carry ``completion_lengths`` (pooled: total tokens / total completions) is divided by the
  baseline length (pooled mean of the first ``baseline_updates`` eligible reports that carry
  lengths, or the explicit ``baseline_length``). ``length_ratio < length_fraction`` puts
  ``"length_collapse"`` in ``reasons`` for every update the condition holds (level). It sets
  ``alarm`` only on the *rising edge* (the first update of a collapse episode) and only if
  ``length_collapse_alarm=True`` (default): a model that legitimately learns shorter answers must
  not veto every later ACCEPT. Reports without lengths leave the length statistics unchanged.

Default parameters and their simulated behaviour
------------------------------------------------

Defaults: ``drift=0.1``, ``threshold=0.3``, ``min_rollouts=32``, ``baseline_updates=3``,
``window=5``, ``length_fraction=0.5``, ``rollouts_per_unit=1``. They were chosen on one set of simulation seeds and are
verified on other seeds in ``tests/research_rl/test_capability_alarm.py`` (1000 streams per
setting; each rollout: V2 passes with probability ``p = 0.3``; a V2-failing rollout passes V1
with the probability that makes the expected V1-only share equal the stream's gap; the
baseline is estimated from the first 3 updates of the stream itself):

* **Clean streams** (constant gap ``c`` in {0, 0.05, 0.1}, 32 or 64 rollouts per update, 200
  updates): the share of streams with *any* gap alarm is documented to be at most **2 %** (the
  test allows three Monte-Carlo standard errors above it; observed 0-3 of 1000 per setting, the
  highest at ``c = 0.1`` with 32 rollouts).
* **Leaking streams** (gap rising linearly from 0 to 0.5 over 100 updates, 32 or 64 rollouts):
  the first alarm fires at update index **<= 40** (gap about 0.2) in at least **99 %** of streams
  (observed median 30-31, i.e. gap ~0.15; never during the baseline updates). The noise-free
  stream (200 rollouts, exact counts) fires at update 32 (gap 0.16).
* **Two-sided discordance** (V1-only and V2-only passes, multinomial draws, inside the scope
  below: 0.05 / 0.05 at 32 rollouts, 0.1 / 0.1 and 0.15 / 0.05 at 64 rollouts): clean streams
  stay within the same **2 %** level (500 streams per setting; observed 2-5 of 500).
* **Clustered streams with unit information** (ES-like: members replicate the outcome of shared
  items exactly, or re-draw it with probability 0.5): with 32 shared items per update the level and
  the delay bound above hold as for 32 independent rollouts (500 streams per setting; observed
  0-2 of 500 clean streams alarm, every leaking stream fires by update 40, median 30-31); with 4
  shared items the report has ``n_eff = 4`` and every update is skipped (the same feed without
  unit information alarms on 94 of 100 clean streams).

Scope of the false-alarm level. Per unit the gap has variance
``sigma**2 = p10 + p01 - (p10 - p01)**2`` (``p10`` = V1-only share, ``p01`` = V2-only share), so
``Var g(t) <= sigma**2 / n_eff`` for independent units. The level is stated only for
``sigma**2 / n_eff <= 0.1 / 32`` (about 0.0031) with ``n_eff >= 32`` (the default
``min_rollouts``): e.g. a V1-only share up to 0.1 without V2-only passes at ``n_eff >= 32``, or
V1-only and V2-only shares of 0.05 each at ``n_eff = 32`` (0.1 each needs ``n_eff >= 64``). It
is the *discordant* share that matters, not the net baseline gap: V1-only 0.1 plus V2-only 0.1
(net gap 0) at ``n_eff = 32`` alarms on 10-12 % of clean streams. Extra simulations on fresh
seeds (1000 streams each, two-sided multinomial draws) gave 0.1-1.2 % inside this scope. Outside
it the level does not hold; lowering ``min_rollouts`` or feeding noisier gaps requires raising
``threshold`` or ``drift`` (re-run the simulation). A larger ``drift`` lowers the false alarm
rate and lengthens the delay; the alarm reacts to an excess gap of roughly ``drift`` and more,
not to smaller sustained increases.

Limits: the alarm is a heuristic early-warning signal, not a calibrated test, and it never
decides anything on its own; the gate (:mod:`research_rl.capability.gate`) does. It only sees a
leak that V1 and V2 disagree on; a hack that fools both verifiers is invisible to it.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from research_rl.capability.types import OperatorReport
from research_rl.rewards.math_verifier import MathVerifier
from research_rl.rewards.verifier_interface import Verifier

__all__ = [
    "UNITS_KEY",
    "AlarmState",
    "DisagreementAlarm",
    "LeakAudit",
    "screen_mask",
    "static_leak_audit",
    "strict_math_verifier",
]

REASON_GAP = "gap_cusum"
REASON_LENGTH = "length_collapse"
UNITS_KEY = "alarm_units"
"""``OperatorReport.extra`` key of the optional per-rollout independent-unit ids (int or str)."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _as_bool(value: Any, what: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1) and not isinstance(value, str):
        return bool(value)
    raise TypeError(f"{what} must be a bool (or 0/1), got {value!r}")


# ============================================================================== alarm state
@dataclass(frozen=True)
class AlarmState:
    """The alarm's output for one update. All fields are finite floats.

    * ``alarm``: the gap CUSUM fired at this update (with ``reset_on_alarm=False``: ``S`` is above
      the threshold, also on skipped reports), or a length collapse started at this update (and
      ``length_collapse_alarm`` is on).
    * ``gap``: ``g(t)`` of this report (0.0 if it has no rollouts; reported also for skipped
      reports, which do not enter any statistic).
    * ``baseline_gap``: the fixed baseline ``b``; while it is still being estimated, the pooled
      gap of the eligible reports seen so far (0.0 before the first one).
    * ``cusum``: ``S(t)`` *before* a reset-on-alarm (so a firing state shows ``cusum > threshold``).
    * ``length_ratio``: windowed mean length / baseline length; 1.0 while either is unavailable.
    * ``reasons``: the conditions present at this update, in the order ``"gap_cusum"``,
      ``"length_collapse"`` (the latter for every update the collapse lasts, even when it does
      not set ``alarm``).
    """

    alarm: bool
    gap: float
    baseline_gap: float
    cusum: float
    length_ratio: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "alarm": self.alarm,
            "gap": self.gap,
            "baseline_gap": self.baseline_gap,
            "cusum": self.cusum,
            "length_ratio": self.length_ratio,
            "reasons": list(self.reasons),
        }


# ============================================================================== the alarm
class DisagreementAlarm:
    """One-sided CUSUM on the V1-V2 pass-rate gap plus a completion-length collapse flag.

    See the module docstring for the exact statistics, the defaults and their simulated
    false-alarm rate and detection delay.

    Args:
        window: number of most recent eligible reports *with lengths* pooled for the mean
            completion length compared to the baseline length (>= 1).
        threshold: CUSUM decision threshold ``h`` in pass-rate units (> 0).
        min_rollouts: reports whose effective number of independent units ``n_eff`` is below
            this are skipped (>= 1).
        drift: CUSUM allowance ``k`` in pass-rate units (>= 0).
        baseline_updates: eligible reports used to estimate the gap baseline (when
            ``baseline_gap`` is None) and the length baseline (when ``baseline_length`` is None).
        baseline_gap: explicit static baseline for ``g`` in [-1, 1]; the CUSUM runs from the
            first eligible report.
        baseline_length: explicit baseline mean completion length (> 0).
        length_fraction: collapse when ``length_ratio < length_fraction`` (in (0, 1)).
        length_collapse_alarm: whether the start of a collapse sets ``alarm``.
        reset_on_alarm: reset ``S`` to 0 after it fires (one-update pulses).
        rollouts_per_unit: number of correlated rollouts per independent unit for reports that
            carry no ``extra[UNITS_KEY]`` ids (>= 1; 1 = independent rollouts). A report whose
            rollout count is not a multiple of it raises ValueError in :meth:`update`.

    ``min_rollouts`` is compared with the effective number of independent units ``n_eff``
    (:meth:`effective_size`), which equals the rollout count only for independent rollouts.
    """

    def __init__(
        self,
        window: int = 5,
        threshold: float = 0.3,
        min_rollouts: int = 32,
        *,
        drift: float = 0.1,
        baseline_updates: int = 3,
        baseline_gap: float | None = None,
        baseline_length: float | None = None,
        length_fraction: float = 0.5,
        length_collapse_alarm: bool = True,
        reset_on_alarm: bool = True,
        rollouts_per_unit: int = 1,
    ) -> None:
        if not _is_int(window) or window < 1:
            raise ValueError(f"window must be an int >= 1, got {window!r}")
        if not _is_number(threshold) or threshold <= 0:
            raise ValueError(f"threshold must be a finite number > 0, got {threshold!r}")
        if not _is_int(min_rollouts) or min_rollouts < 1:
            raise ValueError(f"min_rollouts must be an int >= 1, got {min_rollouts!r}")
        if not _is_number(drift) or drift < 0:
            raise ValueError(f"drift must be a finite number >= 0, got {drift!r}")
        if not _is_int(rollouts_per_unit) or rollouts_per_unit < 1:
            raise ValueError(f"rollouts_per_unit must be an int >= 1, got {rollouts_per_unit!r}")
        if not _is_int(baseline_updates) or baseline_updates < 1:
            raise ValueError(f"baseline_updates must be an int >= 1, got {baseline_updates!r}")
        if baseline_gap is not None and (not _is_number(baseline_gap) or not -1.0 <= baseline_gap <= 1.0):
            raise ValueError(f"baseline_gap must be None or in [-1, 1], got {baseline_gap!r}")
        if baseline_length is not None and (not _is_number(baseline_length) or baseline_length <= 0):
            raise ValueError(f"baseline_length must be None or a finite number > 0, got {baseline_length!r}")
        if not _is_number(length_fraction) or not 0.0 < length_fraction < 1.0:
            raise ValueError(f"length_fraction must be in (0, 1), got {length_fraction!r}")
        for name, flag in (
            ("length_collapse_alarm", length_collapse_alarm),
            ("reset_on_alarm", reset_on_alarm),
        ):
            if not isinstance(flag, bool):
                raise ValueError(f"{name} must be a bool, got {flag!r}")
        self.window = window
        self.threshold = float(threshold)
        self.min_rollouts = min_rollouts
        self.drift = float(drift)
        self.baseline_updates = baseline_updates
        self.static_baseline_gap = None if baseline_gap is None else float(baseline_gap)
        self.static_baseline_length = None if baseline_length is None else float(baseline_length)
        self.length_fraction = float(length_fraction)
        self.length_collapse_alarm = length_collapse_alarm
        self.reset_on_alarm = reset_on_alarm
        self.rollouts_per_unit = rollouts_per_unit
        self._reset_state()

    # ------------------------------------------------------------------ state
    def _reset_state(self) -> None:
        # gap baseline as integer counts: sum(V1 passes - V2 passes), sum(rollouts), reports used
        self._gap_diff = 0
        self._gap_rollouts = 0
        self._gap_reports = 0
        # length baseline: sum(lengths), number of completions, reports used
        self._len_sum = 0
        self._len_count = 0
        self._len_reports = 0
        self._window: deque[tuple[int, int]] = deque(maxlen=self.window)  # (sum lengths, count)
        self._cusum = 0.0
        self._collapsed = False
        self._updates = 0
        self._eligible = 0
        self._skipped = 0
        self._gap_alarms = 0
        self._length_alarms = 0

    def config(self) -> dict[str, Any]:
        """The constructor arguments as a JSON-able dict (``DisagreementAlarm(**config())``)."""
        return {
            "window": self.window,
            "threshold": self.threshold,
            "min_rollouts": self.min_rollouts,
            "drift": self.drift,
            "baseline_updates": self.baseline_updates,
            "baseline_gap": self.static_baseline_gap,
            "baseline_length": self.static_baseline_length,
            "length_fraction": self.length_fraction,
            "length_collapse_alarm": self.length_collapse_alarm,
            "reset_on_alarm": self.reset_on_alarm,
            "rollouts_per_unit": self.rollouts_per_unit,
        }

    @property
    def baseline_ready(self) -> bool:
        """True once the gap baseline is fixed (static, or ``baseline_updates`` reports seen)."""
        return self.static_baseline_gap is not None or self._gap_reports >= self.baseline_updates

    @property
    def length_baseline_ready(self) -> bool:
        return self.static_baseline_length is not None or self._len_reports >= self.baseline_updates

    @property
    def baseline_gap(self) -> float:
        """Current gap baseline (partial pooled estimate while not ready, 0.0 before any data)."""
        if self.static_baseline_gap is not None:
            return self.static_baseline_gap
        return self._gap_diff / self._gap_rollouts if self._gap_rollouts else 0.0

    @property
    def baseline_length(self) -> float | None:
        """Baseline mean completion length, or None while not fixed (or zero: undefined ratio)."""
        if self.static_baseline_length is not None:
            return self.static_baseline_length
        if not self.length_baseline_ready or self._len_sum <= 0:
            return None
        return self._len_sum / self._len_count

    @property
    def cusum(self) -> float:
        return self._cusum

    @property
    def updates(self) -> int:
        """Reports seen (eligible + skipped)."""
        return self._updates

    @property
    def skipped_updates(self) -> int:
        return self._skipped

    @property
    def gap_alarms(self) -> int:
        """Number of updates whose state carries ``"gap_cusum"`` (in latched mode this includes
        skipped reports returned while ``S > threshold``)."""
        return self._gap_alarms

    @property
    def length_alarms(self) -> int:
        """Number of collapse episodes that set ``alarm``."""
        return self._length_alarms

    def _length_ratio(self) -> float | None:
        base = self.baseline_length
        count = sum(c for _, c in self._window)
        if base is None or not self.length_baseline_ready or count == 0:
            return None
        return (sum(s for s, _ in self._window) / count) / base

    # ------------------------------------------------------------------ units
    def _unit_sizes_sq(self, report: OperatorReport) -> int:
        """``sum_j s_j**2`` over the report's independent units (see the module docstring)."""
        n = len(report.v1_pass)
        units = report.extra.get(UNITS_KEY) if report.extra else None
        if units is not None:
            if isinstance(units, (str, bytes)) or not isinstance(units, Sequence):
                raise ValueError(
                    f"extra[{UNITS_KEY!r}] must be a sequence of ids, got {type(units).__name__}"
                )
            if len(units) != n:
                raise ValueError(f"extra[{UNITS_KEY!r}] has {len(units)} ids but the report has {n} rollouts")
            sizes: dict[int | str, int] = {}
            for unit in units:
                if not (_is_int(unit) or isinstance(unit, str)):
                    raise ValueError(f"extra[{UNITS_KEY!r}] ids must be ints or strs, got {unit!r}")
                sizes[unit] = sizes.get(unit, 0) + 1
            return sum(c * c for c in sizes.values())
        if n % self.rollouts_per_unit:
            raise ValueError(
                f"report has {n} rollouts, not a multiple of rollouts_per_unit={self.rollouts_per_unit}"
            )
        return n * self.rollouts_per_unit

    def effective_size(self, report: OperatorReport) -> float:
        """Kish effective number of independent units ``n**2 / sum_j s_j**2`` (0.0 if empty).

        Raises ValueError for malformed ``extra[UNITS_KEY]`` ids or a rollout count that is not a
        multiple of ``rollouts_per_unit``.
        """
        n = len(report.v1_pass)
        return n * n / self._unit_sizes_sq(report) if n else 0.0

    # ------------------------------------------------------------------ update
    def update(self, report: OperatorReport) -> AlarmState:
        """Feed one operator update; returns the alarm state after it (see :class:`AlarmState`)."""
        n = len(report.v1_pass)
        v1 = sum(_as_bool(v, "v1_pass entries") for v in report.v1_pass)
        v2 = sum(_as_bool(v, "v2_pass entries") for v in report.v2_pass)
        lengths = report.completion_lengths
        for length in lengths:
            if not _is_int(length) or length < 0:
                raise ValueError(f"completion_lengths must be non-negative ints, got {length!r}")
        sum_sq = self._unit_sizes_sq(report)
        gap = (v1 - v2) / n if n else 0.0
        self._updates += 1
        fired = False
        cusum_shown = self._cusum
        if n == 0 or n * n < self.min_rollouts * sum_sq:  # n_eff < min_rollouts, in exact ints
            self._skipped += 1
        else:
            self._eligible += 1
            if self.baseline_ready:
                self._cusum = max(0.0, self._cusum + gap - self.baseline_gap - self.drift)
                cusum_shown = self._cusum
                if self._cusum > self.threshold:
                    fired = True
                    if self.reset_on_alarm:
                        self._cusum = 0.0
            else:
                self._gap_diff += v1 - v2
                self._gap_rollouts += n
                self._gap_reports += 1
            if lengths:
                if self.length_baseline_ready:
                    self._window.append((sum(lengths), len(lengths)))
                else:
                    self._len_sum += sum(lengths)
                    self._len_count += len(lengths)
                    self._len_reports += 1
        if not self.reset_on_alarm and self._cusum > self.threshold:
            fired = True  # latched mode: on while S > h (also on skipped reports)
        if fired:
            self._gap_alarms += 1
        ratio = self._length_ratio()
        collapsed = ratio is not None and ratio < self.length_fraction
        length_alarm = collapsed and not self._collapsed and self.length_collapse_alarm
        if length_alarm:
            self._length_alarms += 1
        self._collapsed = collapsed
        reasons = tuple(r for r, on in ((REASON_GAP, fired), (REASON_LENGTH, collapsed)) if on)
        return AlarmState(
            alarm=fired or length_alarm,
            gap=gap,
            baseline_gap=self.baseline_gap,
            cusum=cusum_shown,
            length_ratio=1.0 if ratio is None else ratio,
            reasons=reasons,
        )

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> dict[str, Any]:
        """JSON-able state (integers and finite floats only); includes the config for checking."""
        return {
            "config": self.config(),
            "gap_diff": self._gap_diff,
            "gap_rollouts": self._gap_rollouts,
            "gap_reports": self._gap_reports,
            "len_sum": self._len_sum,
            "len_count": self._len_count,
            "len_reports": self._len_reports,
            "window": [[s, c] for s, c in self._window],
            "cusum": self._cusum,
            "collapsed": self._collapsed,
            "updates": self._updates,
            "eligible": self._eligible,
            "skipped": self._skipped,
            "gap_alarms": self._gap_alarms,
            "length_alarms": self._length_alarms,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a :meth:`state_dict`; the stored config must equal this alarm's config.

        After loading, the alarm returns exactly the same :class:`AlarmState` sequence on the same
        subsequent reports as the alarm that produced the state.
        """
        stored = dict(state["config"])
        if stored != self.config():
            raise ValueError(f"alarm config mismatch: state has {stored}, this alarm has {self.config()}")
        ints = (
            "gap_diff",
            "gap_rollouts",
            "gap_reports",
            "len_sum",
            "len_count",
            "len_reports",
            "updates",
            "eligible",
            "skipped",
            "gap_alarms",
            "length_alarms",
        )
        for key in ints:
            value = state[key]
            if not _is_int(value) or (key != "gap_diff" and value < 0):
                raise ValueError(f"invalid {key} {value!r}")
        cusum = state["cusum"]
        if not _is_number(cusum) or cusum < 0:
            raise ValueError(f"invalid cusum {cusum!r}")
        if not isinstance(state["collapsed"], bool):
            raise ValueError(f"invalid collapsed {state['collapsed']!r}")
        entries = [tuple(e) for e in state["window"]]
        if len(entries) > self.window or any(
            len(e) != 2 or not all(_is_int(x) and x >= 0 for x in e) for e in entries
        ):
            raise ValueError(f"invalid length window {state['window']!r}")
        self._reset_state()
        for key in ints:
            setattr(self, f"_{key}", state[key])
        self._cusum = float(cusum)
        self._collapsed = state["collapsed"]
        self._window.extend((int(s), int(c)) for s, c in entries)


# ============================================================================== screening
def screen_mask(v1_pass: Sequence[Any], v2_pass: Sequence[Any]) -> list[bool]:
    """Per-rollout keep mask: ``False`` exactly where V1 passes and V2 fails, ``True`` otherwise.

    Screened rollouts are meant to get advantage 0 (excluded from the update), not a penalty.
    V1-fail / V2-pass rollouts are kept. Entries must be bools (or 0/1); the two sequences must
    have the same length (ValueError otherwise).
    """
    if len(v1_pass) != len(v2_pass):
        raise ValueError(f"v1_pass has {len(v1_pass)} entries but v2_pass has {len(v2_pass)}")
    return [
        not (_as_bool(a, "v1_pass entries") and not _as_bool(b, "v2_pass entries"))
        for a, b in zip(v1_pass, v2_pass)
    ]


# ============================================================================== static audit
@dataclass(frozen=True)
class LeakAudit:
    """Static comparison of V1 and V2 on a fixed set of completions.

    ``leak_rate`` is the share of completions that pass V1 and fail V2. ``gap`` (derived) is
    ``v1_pass_rate - v2_pass_rate``, the quantity :class:`DisagreementAlarm` monitors; use it
    (not ``leak_rate``) as an explicit ``baseline_gap``.
    """

    n: int
    v1_pass_rate: float
    v2_pass_rate: float
    leak_rate: float

    @property
    def gap(self) -> float:
        return self.v1_pass_rate - self.v2_pass_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "v1_pass_rate": self.v1_pass_rate,
            "v2_pass_rate": self.v2_pass_rate,
            "leak_rate": self.leak_rate,
            "gap": self.gap,
        }


def static_leak_audit(
    completions: Sequence[str | None],
    references: Sequence[Any],
    v1: Verifier,
    v2: Verifier,
) -> LeakAudit:
    """Grade every completion with both verifiers (``verify(completion, reference).correct``).

    Works with any :class:`~research_rl.rewards.verifier_interface.Verifier` (e.g. any
    ``BaseVerifier``). Raises ValueError for mismatched lengths or an empty set.
    """
    if len(completions) != len(references):
        raise ValueError(f"{len(completions)} completions but {len(references)} references")
    n = len(completions)
    if n == 0:
        raise ValueError("static_leak_audit needs at least one completion")
    p1 = p2 = leak = 0
    for completion, reference in zip(completions, references):
        a = v1.verify(completion, reference).correct
        b = v2.verify(completion, reference).correct
        p1 += a
        p2 += b
        leak += a and not b
    return LeakAudit(n=n, v1_pass_rate=p1 / n, v2_pass_rate=p2 / n, leak_rate=leak / n)


def strict_math_verifier(**overrides: Any) -> MathVerifier:
    """The strict V2 configuration of :class:`MathVerifier` for math.

    Defaults ``extractors=("boxed",)``, ``strict_boxed=True`` (exactly one distinct boxed answer;
    no box is a format error), ``strict_trailing=True`` (no text after the answer except closing
    math delimiters, ``*`` and a period) and ``strip_percent=False``; ``overrides`` replace any of
    these or set other constructor arguments.
    """
    kwargs: dict[str, Any] = {
        "extractors": ("boxed",),
        "strict_boxed": True,
        "strict_trailing": True,
        "strip_percent": False,
    }
    kwargs.update(overrides)
    return MathVerifier(**kwargs)
