"""The single model worker and its job queue.

Every operation that reads or writes model weights, generates text, or touches the
:class:`~research_rl.capability.edits.VersionStore` runs as a :class:`Job` on **one** worker
thread (:class:`ModelWorker`), so the HTTP threads of the server never touch weights and two
model operations never overlap. A job has an id, a kind, a status (``queued`` -> ``running`` ->
``done`` / ``failed``), a progress fraction with a message, and a JSON-able result or an error.

The queue is bounded (:attr:`ModelWorker.max_pending`); a full queue raises :class:`QueueFullError`
instead of growing without limit. Finished jobs are kept for :attr:`ModelWorker.keep_finished`
jobs (oldest dropped first). An exception inside a job marks that job ``failed`` and never stops
the worker thread.
"""

from __future__ import annotations

import queue
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Job",
    "JobContext",
    "ModelWorker",
    "QueueFullError",
    "WorkerStoppedError",
]

JOB_STATUSES = ("queued", "running", "done", "failed")


class QueueFullError(RuntimeError):
    """The worker already has :attr:`ModelWorker.max_pending` unfinished jobs."""


class WorkerStoppedError(RuntimeError):
    """The worker was stopped; no job can be submitted."""


@dataclass
class Job:
    """One unit of model work. Mutable fields are written only by the worker under ``lock``."""

    id: str
    kind: str
    params: dict[str, Any]
    fn: Callable[[JobContext], Any]
    status: str = "queued"
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: dict[str, Any] | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    exception: BaseException | None = None
    done_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def to_dict(self, *, include_result: bool = True) -> dict[str, Any]:
        """JSON-able view (the callable and the raw exception are never exposed)."""
        with self.lock:
            out: dict[str, Any] = {
                "id": self.id,
                "kind": self.kind,
                "params": self.params,
                "status": self.status,
                "progress": self.progress,
                "message": self.message,
                "error": self.error,
                "created": self.created,
                "started": self.started,
                "finished": self.finished,
            }
            if include_result:
                out["result"] = self.result
            return out

    @property
    def finished_ok(self) -> bool:
        """True iff the job finished without an exception."""
        return self.status == "done"


class JobContext:
    """Handed to a job function: report progress, and check that the code runs on the worker."""

    def __init__(self, job: Job) -> None:
        self._job = job

    @property
    def job_id(self) -> str:
        """Id of the running job."""
        return self._job.id

    def progress(self, fraction: float, message: str = "") -> None:
        """Set the job's progress (clamped to [0, 1]) and an optional short message."""
        value = min(1.0, max(0.0, float(fraction)))
        with self._job.lock:
            self._job.progress = value
            if message:
                self._job.message = message[:500]


ErrorMapper = Callable[[BaseException], dict[str, Any]]


def _default_error(exc: BaseException) -> dict[str, Any]:
    return {"type": type(exc).__name__, "message": str(exc)[:2000]}


class ModelWorker:
    """One daemon thread executing queued jobs in submission order.

    Args:
        max_pending: unfinished (queued or running) jobs allowed at once.
        keep_finished: finished jobs kept for :meth:`get` / :meth:`list`.
        error_mapper: turns a job's exception into the JSON ``error`` dict (type and message;
            never a traceback).
    """

    def __init__(
        self, *, max_pending: int = 32, keep_finished: int = 200, error_mapper: ErrorMapper | None = None
    ) -> None:
        if max_pending < 1 or keep_finished < 1:
            raise ValueError("max_pending and keep_finished must be >= 1")
        self.max_pending = int(max_pending)
        self.keep_finished = int(keep_finished)
        self._error_mapper = error_mapper or _default_error
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._pending = 0
        self._stopped = False
        self._current: str | None = None
        self._thread = threading.Thread(target=self._run, name="research-rl-ui-model-worker", daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- identity
    @property
    def thread(self) -> threading.Thread:
        """The worker thread."""
        return self._thread

    def on_worker(self) -> bool:
        """True iff the calling code runs on the worker thread."""
        return threading.current_thread() is self._thread

    @property
    def current_job(self) -> str | None:
        """Id of the running job, if any."""
        with self._lock:
            return self._current

    @property
    def pending(self) -> int:
        """Unfinished (queued or running) jobs."""
        with self._lock:
            return self._pending

    # ---------------------------------------------------------------- submit / query
    def submit(self, kind: str, fn: Callable[[JobContext], Any], params: dict[str, Any] | None = None) -> Job:
        """Queue ``fn(ctx)`` as a new job and return it (``QueueFullError`` when the queue is full)."""
        job = Job(id=secrets.token_hex(8), kind=str(kind), params=dict(params or {}), fn=fn)
        with self._lock:
            if self._stopped:
                raise WorkerStoppedError("the model worker is stopped")
            if self._pending >= self.max_pending:
                raise QueueFullError(f"{self._pending} jobs are already waiting; try again later")
            self._pending += 1
            self._jobs[job.id] = job
            self._trim()
        self._queue.put(job)
        return job

    def call(
        self, kind: str, fn: Callable[[JobContext], Any], params: dict[str, Any] | None = None, *, wait: float
    ) -> Job:
        """:meth:`submit`, then wait up to ``wait`` seconds for the job to finish; returns the job
        (finished or not)."""
        job = self.submit(kind, fn, params)
        if wait > 0:
            job.done_event.wait(wait)
        return job

    def get(self, job_id: str) -> Job | None:
        """The job with ``job_id`` (``None`` if unknown or already dropped)."""
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        """Every known job, newest first."""
        with self._lock:
            return list(reversed(self._jobs.values()))

    def stop(self, timeout: float = 10.0) -> None:
        """Stop accepting jobs, let the queued ones finish, and join the thread."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
        self._queue.put(None)
        if not self.on_worker():
            self._thread.join(timeout)

    # ---------------------------------------------------------------- internals
    def _trim(self) -> None:
        finished = [j for j in self._jobs.values() if j.status in ("done", "failed")]
        extra = len(finished) - self.keep_finished
        for job in finished[: max(0, extra)]:
            self._jobs.pop(job.id, None)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            with self._lock:
                self._current = job.id
            with job.lock:
                job.status = "running"
                job.started = time.time()
            try:
                result = job.fn(JobContext(job))
            except BaseException as exc:  # a job failure must never stop the worker
                error = self._error_mapper(exc)
                with job.lock:
                    job.status = "failed"
                    job.error = error
                    job.exception = exc
                    job.finished = time.time()
            else:
                with job.lock:
                    job.status = "done"
                    job.result = result
                    job.progress = 1.0
                    job.finished = time.time()
            finally:
                with self._lock:
                    self._current = None
                    self._pending -= 1
                    self._trim()
                job.done_event.set()
