"""Profiling/hardware reports (CPU paths; GPU numbers must never be fabricated) and logging."""

from __future__ import annotations

import io
import json
import re

import pytest
import torch

from research_rl.core.experiment import NOT_TESTED
from research_rl.utils import (
    StepTimer,
    get_logger,
    gpu_memory_report,
    hardware_report,
    measure_throughput,
    profiling,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


# ---------------------------------------------------------------------------------- StepTimer
def test_step_timer_arithmetic_with_fake_clock():
    clock = FakeClock()
    timer = StepTimer(clock=clock)
    # asymmetric durations: total (4.0) != steps (3) and median (1.0) != mean (4/3)
    for dt, tok in [(0.5, 100), (2.5, 300), (1.0, 200)]:
        with timer.step(tokens=tok):
            clock.advance(dt)
        clock.advance(10.0)  # time between steps is not counted
    assert timer.durations == [0.5, 2.5, 1.0]
    assert timer.count == 3 and timer.total == 4.0 and timer.last == 1.0
    s = timer.summary()
    assert s == {
        "steps": 3,
        "total_seconds": 4.0,
        "mean_seconds": pytest.approx(4.0 / 3),
        "median_seconds": 1.0,
        "min_seconds": 0.5,
        "max_seconds": 2.5,
        "total_tokens": 600,
        "tokens_per_second": 150.0,
    }


def test_step_timer_start_stop_and_errors():
    clock = FakeClock()
    calls: list[str] = []
    timer = StepTimer(clock=clock, synchronize=lambda: calls.append("sync"))
    timer.start()
    clock.advance(2.0)
    assert timer.stop() == 2.0
    assert calls == ["sync", "sync"]
    assert "tokens_per_second" not in timer.summary()  # a step without tokens -> no token rate
    with pytest.raises(RuntimeError):
        timer.stop()
    timer.start()
    with pytest.raises(RuntimeError):
        timer.start()
    assert StepTimer(clock=clock).summary() == {"steps": 0}


def test_step_timer_records_step_even_if_body_raises():
    clock = FakeClock()
    timer = StepTimer(clock=clock)
    with pytest.raises(ValueError), timer.step():
        clock.advance(0.25)
        raise ValueError("boom")
    assert timer.durations == [0.25]


# ---------------------------------------------------------------------------------- throughput
def test_measure_throughput_excludes_warmup_with_fake_clock():
    clock = FakeClock()
    calls = {"n": 0}
    syncs: list[int] = []

    def fn() -> None:
        calls["n"] += 1
        clock.advance(7.0 if calls["n"] <= 2 else 0.25)  # first two (warmup) calls are slow

    out = measure_throughput(
        fn, n_steps=8, warmup=2, tokens_per_step=64, clock=clock, synchronize=lambda: syncs.append(calls["n"])
    )
    assert calls["n"] == 10
    assert syncs == [2, 10]  # synchronised right before and right after the timed block
    assert out["total_seconds"] == 2.0
    assert out["seconds_per_step"] == 0.25
    assert out["steps_per_second"] == 4.0
    assert out["tokens_per_second"] == 64 * 8 / 2.0
    assert out["steps"] == 8 and out["warmup_steps"] == 2
    assert out["cuda_available"] is torch.cuda.is_available() and out["synchronized"] is True
    assert "device" not in out  # fn's device is unknown here; not guessed
    assert "tokens_per_second" not in measure_throughput(fn, 1, warmup=0, clock=clock)
    with pytest.raises(ValueError):
        measure_throughput(fn, 0)
    with pytest.raises(ValueError):
        measure_throughput(fn, 1, warmup=-1)


def test_auto_sync_plumbing_with_mocked_cuda(monkeypatch):
    """Plumbing only (torch.cuda mocked; the real CUDA path is NOT_TESTED in CPU CI): StepTimer and
    measure_throughput share the same default, which syncs whenever CUDA is available."""
    calls: list[str] = []
    monkeypatch.setattr(profiling.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(profiling.torch.cuda, "synchronize", lambda: calls.append("sync"))
    clock = FakeClock()
    timer = StepTimer(clock=clock)
    with timer.step():
        clock.advance(1.0)
    assert calls == ["sync", "sync"]
    calls.clear()
    out = measure_throughput(lambda: clock.advance(0.5), n_steps=2, warmup=0, clock=clock)
    assert calls == ["sync", "sync"] and out["synchronized"] is True and out["cuda_available"] is True
    calls.clear()
    with StepTimer(clock=clock, synchronize=None).step():
        clock.advance(1.0)
    out = measure_throughput(lambda: None, n_steps=1, warmup=0, clock=clock, synchronize=None)
    assert calls == [] and out["synchronized"] is False


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the CPU (no CUDA) path")
def test_auto_sync_is_off_without_cuda():
    clock = FakeClock()
    out = measure_throughput(lambda: clock.advance(1.0), n_steps=1, warmup=0, clock=clock)
    assert out["synchronized"] is False and out["cuda_available"] is False


def test_measure_throughput_real_clock_runs_cpu_workload():
    x = torch.randn(64, 64)
    out = measure_throughput(lambda: x @ x, n_steps=5, warmup=1)
    assert out["total_seconds"] > 0 and out["seconds_per_step"] == pytest.approx(out["total_seconds"] / 5)


# ---------------------------------------------------------------------------------- GPU honesty
@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the CPU (no CUDA) path")
def test_gpu_memory_report_is_not_tested_on_cpu():
    rep = gpu_memory_report()
    assert rep["status"] == NOT_TESTED == "NOT_TESTED"
    assert not any(k.endswith("_bytes") for k in rep)  # no numbers fabricated
    hw = hardware_report()
    assert hw["cuda_available"] is False and hw["cuda_device_count"] == 0
    assert hw["cuda_devices"] == NOT_TESTED and hw["gpu_memory"]["status"] == NOT_TESTED
    assert hw["torch"] == torch.__version__ and hw["cpu_count"] >= 1
    json.dumps(hw)


def test_gpu_memory_report_key_mapping_with_mocked_cuda(monkeypatch):
    """Plumbing only: torch.cuda is mocked, so this says nothing about real GPU measurements
    (the real CUDA branch is NOT_TESTED in CPU CI)."""

    class Props:
        total_memory = 80

    fake = {
        "is_available": lambda: True,
        "current_device": lambda: 1,
        "get_device_name": lambda d: f"FakeGPU{d}",
        "max_memory_allocated": lambda d: 11 * (d + 1),
        "max_memory_reserved": lambda d: 13 * (d + 1),
        "memory_allocated": lambda d: 5,
        "memory_reserved": lambda d: 7,
        "get_device_properties": lambda d: Props(),
    }
    for name, fn in fake.items():
        monkeypatch.setattr(profiling.torch.cuda, name, fn)
    rep = gpu_memory_report()
    assert rep == {
        "status": "measured",
        "device_index": 1,
        "device_name": "FakeGPU1",
        "max_memory_allocated_bytes": 22,
        "max_memory_reserved_bytes": 26,
        "memory_allocated_bytes": 5,
        "memory_reserved_bytes": 7,
        "total_memory_bytes": 80,
    }
    assert gpu_memory_report(device=0)["max_memory_allocated_bytes"] == 11


# ---------------------------------------------------------------------------------- logging
LINE = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \| (\w+)\s* \| rank (\d+) \| ([\w.]+) \| (.*)$")


def test_get_logger_format_and_idempotence(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    buf = io.StringIO()
    log = get_logger("research_rl.test.fmt", stream=buf)
    log = get_logger("research_rl.test.fmt", stream=buf)  # no duplicate handler
    log.info("hello %d", 3)
    log.debug("hidden")
    lines = buf.getvalue().splitlines()
    assert len(lines) == 1
    m = LINE.match(lines[0])
    assert m is not None and m.groups() == ("INFO", "0", "research_rl.test.fmt", "hello 3")
    assert len(log.handlers) == 1 and log.propagate is False


@pytest.mark.parametrize(("rank", "expected"), [("0", ["info", "warn", "err"]), ("3", ["err"])])
def test_get_logger_rank_zero_only(monkeypatch, rank, expected):
    monkeypatch.setenv("RANK", rank)
    buf = io.StringIO()
    log = get_logger(f"research_rl.test.rank{rank}", stream=buf, rank_zero_only=True)
    log.info("info")
    log.warning("warn")
    log.error("err")
    got = [LINE.match(line).group(4) for line in buf.getvalue().splitlines()]  # type: ignore[union-attr]
    assert got == expected
    assert all(f"rank {rank} |" in line for line in buf.getvalue().splitlines())


def test_get_logger_all_ranks_by_default(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    buf = io.StringIO()
    log = get_logger("research_rl.test.allranks", level="DEBUG", stream=buf)
    log.debug("d")
    assert "rank 2 | research_rl.test.allranks | d" in buf.getvalue()
