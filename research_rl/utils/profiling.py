"""Timing, throughput and hardware/memory reports.

Honesty rule: GPU numbers are only ever returned when they were measured through ``torch.cuda`` in
this process. Without CUDA every GPU field is reported as ``"NOT_TESTED"``; nothing is estimated.
The CUDA branch of :func:`gpu_memory_report` is NOT_TESTED in CI (no GPU there); its key mapping
is unit-tested against a mocked ``torch.cuda`` only.
"""

from __future__ import annotations

import os
import platform
import statistics
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Literal

import torch

from research_rl.core.experiment import NOT_TESTED

Clock = Callable[[], float]
Sync = Callable[[], None] | None | Literal["auto"]


def _resolve_sync(synchronize: Sync) -> Callable[[], None] | None:
    """``"auto"``: ``torch.cuda.synchronize`` if CUDA is available (checked now), else no sync.

    ``None`` disables synchronisation; a callable is used as given.
    """
    if synchronize == "auto":
        return torch.cuda.synchronize if torch.cuda.is_available() else None
    return synchronize


class StepTimer:
    """Records wall-clock durations of steps (and optionally tokens processed per step).

    Use ``with timer.step(tokens=n): ...`` or explicit ``start()`` / ``stop(tokens=n)``. The clock is
    injectable so arithmetic can be tested with a fake clock.

    ``synchronize`` runs right before reading the clock at start and stop. The default ``"auto"``
    (resolved when the timer is constructed) is ``torch.cuda.synchronize`` when CUDA is available,
    so asynchronous kernel launches are not mistaken for step time; pass ``None`` to disable or any
    callable (e.g. a per-device sync). The CUDA path is NOT_TESTED in CI (no GPU; mocked only).
    """

    def __init__(self, clock: Clock = time.perf_counter, synchronize: Sync = "auto") -> None:
        self._clock = clock
        self._sync = _resolve_sync(synchronize)
        self._t0: float | None = None
        self.durations: list[float] = []
        self.tokens: list[int | None] = []

    def start(self) -> None:
        if self._t0 is not None:
            raise RuntimeError("StepTimer.start() called twice without stop()")
        if self._sync is not None:
            self._sync()
        self._t0 = self._clock()

    def stop(self, tokens: int | None = None) -> float:
        if self._t0 is None:
            raise RuntimeError("StepTimer.stop() called before start()")
        if self._sync is not None:
            self._sync()
        dt = self._clock() - self._t0
        self._t0 = None
        self.durations.append(dt)
        self.tokens.append(tokens)
        return dt

    @contextmanager
    def step(self, tokens: int | None = None) -> Iterator[StepTimer]:
        self.start()
        try:
            yield self
        finally:
            self.stop(tokens)

    @property
    def count(self) -> int:
        return len(self.durations)

    @property
    def total(self) -> float:
        return float(sum(self.durations))

    @property
    def last(self) -> float | None:
        return self.durations[-1] if self.durations else None

    def summary(self) -> dict[str, float | int | None]:
        """Aggregate statistics; ``tokens_per_second`` only when every step reported tokens."""
        if not self.durations:
            return {"steps": 0}
        out: dict[str, float | int | None] = {
            "steps": self.count,
            "total_seconds": self.total,
            "mean_seconds": self.total / self.count,
            "median_seconds": float(statistics.median(self.durations)),
            "min_seconds": float(min(self.durations)),
            "max_seconds": float(max(self.durations)),
        }
        if all(t is not None for t in self.tokens):
            n_tokens = sum(t for t in self.tokens if t is not None)
            out["total_tokens"] = n_tokens
            out["tokens_per_second"] = n_tokens / self.total if self.total > 0 else None
        return out


def measure_throughput(
    fn: Callable[[], Any],
    n_steps: int,
    warmup: int = 1,
    tokens_per_step: int | None = None,
    *,
    clock: Clock = time.perf_counter,
    synchronize: Sync = "auto",
) -> dict[str, Any]:
    """Call ``fn`` ``warmup`` times untimed, then ``n_steps`` times timed as one block.

    Returns seconds per step, steps per second and (if ``tokens_per_step``) tokens per second.
    ``synchronize`` runs before and after the timed block; as in :class:`StepTimer` the default
    ``"auto"`` means ``torch.cuda.synchronize`` when CUDA is available (``None`` disables it). The
    report says whether CUDA was available and whether a sync ran; it does not claim on which
    device ``fn`` ran (this function cannot know that).
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    sync = _resolve_sync(synchronize)
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()
    t0 = clock()
    for _ in range(n_steps):
        fn()
    if sync is not None:
        sync()
    elapsed = clock() - t0
    out: dict[str, Any] = {
        "steps": n_steps,
        "warmup_steps": warmup,
        "total_seconds": elapsed,
        "seconds_per_step": elapsed / n_steps,
        "steps_per_second": n_steps / elapsed if elapsed > 0 else None,
        "cuda_available": torch.cuda.is_available(),
        "synchronized": sync is not None,
    }
    if tokens_per_step is not None:
        out["tokens_per_step"] = tokens_per_step
        out["tokens_per_second"] = tokens_per_step * n_steps / elapsed if elapsed > 0 else None
    return out


def gpu_memory_report(device: int | None = None) -> dict[str, Any]:
    """Measured CUDA memory statistics, or ``{"status": "NOT_TESTED", ...}`` without CUDA.

    Values come straight from ``torch.cuda.max_memory_allocated`` / ``max_memory_reserved`` /
    ``memory_allocated`` / ``memory_reserved`` (bytes, peak since process start or the last
    ``torch.cuda.reset_peak_memory_stats``) and ``get_device_properties().total_memory``.
    """
    if not torch.cuda.is_available():
        return {
            "status": NOT_TESTED,
            "reason": "CUDA is not available in this process; no GPU memory was measured",
            "device": "cpu",
        }
    dev = torch.cuda.current_device() if device is None else device
    return {
        "status": "measured",
        "device_index": int(dev),
        "device_name": torch.cuda.get_device_name(dev),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(dev)),
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(dev)),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(dev)),
        "total_memory_bytes": int(torch.cuda.get_device_properties(dev).total_memory),
    }


def hardware_report() -> dict[str, Any]:
    """What this process runs on. GPU entries are NOT_TESTED unless CUDA is available."""
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if cuda else 0,
        "cuda_devices": (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else NOT_TESTED
        ),
        "gpu_memory": gpu_memory_report(),
    }
