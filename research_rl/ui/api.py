"""The JSON API of the UI: routing, parameter validation (with bounds) and error mapping.

:meth:`Api.handle` receives an already authenticated request (the server checks Host, Origin,
the session token and the body size before calling it) as ``(method, path segments after
"api", query, body)`` and returns ``(HTTP status, JSON-able payload)``.

Weight access. Endpoints that read or change weights, generate, or use the version store submit
a job to the :class:`~research_rl.ui.jobs.ModelWorker`:

* long jobs (``POST /api/stats``, ``POST /api/search``, ``POST /api/evaluate``) return ``202`` with
  the job at once; poll ``GET /api/jobs/<id>``;
* short jobs (``GET /api/channel/...``, ``POST /api/stage|commit|discard|restore``) wait up to
  ``sync_wait`` seconds and return ``200`` with ``{"result": ...}`` (or the mapped error status);
  when the worker is busy longer they return ``202`` with the job.

Everything else (status, model description, stored-score ranking, history, jobs, run logs) reads
the workspace's published view or files and never touches weights.

AUDIT. A ``split`` of ``audit`` (any case) or a ``slice`` in ``audit/...`` is refused with ``403``
before anything is queued; the workspace and the probe refuse AUDIT again underneath.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from research_rl.capability.edits import CommitRefusedError, EditError, StoreError
from research_rl.capability.search import CONTROL_KINDS
from research_rl.capability.tasks import AuditAccessError
from research_rl.capability.types import Split
from research_rl.ui.jobs import Job, JobContext, ModelWorker, QueueFullError, WorkerStoppedError
from research_rl.ui.runs import RunAccessError, list_runs, read_decisions
from research_rl.ui.workspace import SCORE_KINDS, STATS_SPLITS, SearchParams, Workspace

__all__ = ["LIMITS", "Api", "ApiError", "error_payload", "map_exception"]

LIMITS: dict[str, tuple[float, float]] = {
    "stats_n": (1, 1024),
    "threshold": (0.0, 100.0),
    "top": (1, 2000),
    "bands": (1, 16),
    "k": (1, 100),
    "rounds": (1, 4),
    "burst_updates": (1, 8),
    "repeats_final": (1, 4),
    "search_items_transfer": (1, 512),
    "search_items_retention": (1, 2048),
    "population": (2, 16),
    "sigma": (1e-6, 1.0),
    "lr": (1e-6, 1.0),
    "max_step_norm": (1e-6, 100.0),
    "fitness_items": (1, 128),
    "seed": (0, 2**31 - 1),
    "layer": (0, 100_000),
    "fraction": (1e-6, 1.0),
    "max_candidates": (1, 32),
    "edits": (1, 16),
    "decisions_limit": (1, 100_000),
}
"""Inclusive bounds of every numeric API parameter."""

MAX_SEARCH_CANDIDATES = 16


class ApiError(Exception):
    """An error with an HTTP status, returned as ``{"error": {...}}``."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = int(status)
        self.message = message


def map_exception(exc: BaseException) -> tuple[int, str]:
    """HTTP status and error type for an exception (specific classes first; no tracebacks)."""
    if isinstance(exc, ApiError):
        return exc.status, "ApiError"
    if isinstance(exc, AuditAccessError):
        return 403, "AuditAccessError"
    if isinstance(exc, RunAccessError):
        return 403, "RunAccessError"
    if isinstance(exc, CommitRefusedError):
        return 409, "CommitRefusedError"
    if isinstance(exc, StoreError):
        return 409, "StoreError"
    if isinstance(exc, EditError):
        return 400, "EditError"
    if isinstance(exc, QueueFullError):
        return 429, "QueueFullError"
    if isinstance(exc, WorkerStoppedError):
        return 503, "WorkerStoppedError"
    if isinstance(exc, (KeyError, FileNotFoundError)):
        return 404, type(exc).__name__
    if isinstance(exc, LookupError):
        return 409, type(exc).__name__
    if isinstance(exc, (ValueError, TypeError)):
        return 400, type(exc).__name__
    return 500, type(exc).__name__


def _message(exc: BaseException) -> str:
    if isinstance(exc, ApiError):
        return exc.message
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])[:2000]
    status, _ = map_exception(exc)
    if status == 500:
        return "internal error (see the server log)"
    if isinstance(exc, FileNotFoundError):
        return "not found"
    return str(exc)[:2000]


def error_payload(exc: BaseException) -> dict[str, Any]:
    """``{"status", "type", "message"}`` of an exception (used for jobs and responses)."""
    status, kind = map_exception(exc)
    return {"status": status, "type": kind, "message": _message(exc)}


# ============================================================================== parameter parsing
def _get(source: Mapping[str, Any], key: str, default: Any) -> Any:
    value = source.get(key, default)
    if isinstance(value, list):  # query strings: last value wins
        value = value[-1] if value else default
    return value


def _as_int(source: Mapping[str, Any], key: str, default: int | None, limit: str) -> int | None:
    value = _get(source, key, default)
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ApiError(400, f"{key} must be an integer")
    if isinstance(value, str):
        try:
            value = int(value, 10)
        except ValueError:
            raise ApiError(400, f"{key} must be an integer") from None
    if not isinstance(value, int):
        raise ApiError(400, f"{key} must be an integer")
    lo, hi = LIMITS[limit]
    if not lo <= value <= hi:
        raise ApiError(400, f"{key} must be in [{int(lo)}, {int(hi)}], got {value}")
    return value


def _require_int(source: Mapping[str, Any], key: str, default: int, limit: str) -> int:
    value = _as_int(source, key, default, limit)
    assert value is not None
    return value


def _as_float(source: Mapping[str, Any], key: str, default: float | None, limit: str) -> float | None:
    value = _get(source, key, default)
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ApiError(400, f"{key} must be a number")
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise ApiError(400, f"{key} must be a number") from None
    if not isinstance(value, (int, float)):
        raise ApiError(400, f"{key} must be a number")
    # an int is range-checked exactly (math.isfinite / float() overflow on ints over ~309 digits)
    if isinstance(value, float) and not math.isfinite(value):
        raise ApiError(400, f"{key} must be a finite number")
    lo, hi = LIMITS[limit]
    if not lo <= value <= hi:
        raise ApiError(400, f"{key} must be in [{lo}, {hi}], got {value}")
    return float(value)


def _require_float(source: Mapping[str, Any], key: str, default: float, limit: str) -> float:
    value = _as_float(source, key, default, limit)
    assert value is not None
    return value


def _as_bool(source: Mapping[str, Any], key: str, default: bool) -> bool:
    value = _get(source, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "1", "false", "0"):
        return value.lower() in ("true", "1")
    raise ApiError(400, f"{key} must be a boolean")


def _as_str(source: Mapping[str, Any], key: str, default: str | None, *, limit: int = 200) -> str | None:
    value = _get(source, key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ApiError(400, f"{key} must be a string")
    if len(value) > limit:
        raise ApiError(400, f"{key} is longer than {limit} characters")
    return value


def _reject_audit_text(value: str, what: str) -> None:
    low = value.strip().lower()
    if low == Split.AUDIT.value or low.startswith(f"{Split.AUDIT.value}/") or low.startswith("audit"):
        raise ApiError(403, f"the AUDIT split is not reachable through the API ({what}={value!r})")


def parse_split(value: Any, *, allowed: tuple[Split, ...] = STATS_SPLITS) -> Split:
    """A split name (case-insensitive) among ``allowed``; AUDIT -> 403, anything else unknown -> 400."""
    if not isinstance(value, str) or not value.strip():
        raise ApiError(400, "split must be a non-empty string")
    _reject_audit_text(value, "split")
    low = value.strip().lower()
    for split in allowed:
        if split.value == low:
            return split
    raise ApiError(400, f"unknown split {value!r}; allowed: {[s.value for s in allowed]}")


def _check_body(body: Any, allowed: set[str]) -> dict[str, Any]:
    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise ApiError(400, "the request body must be a JSON object")
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise ApiError(400, f"unknown fields {unknown}; allowed: {sorted(allowed)}")
    return body


# ============================================================================== the API
class Api:
    """Routes authenticated JSON requests (see the module docstring)."""

    def __init__(
        self,
        workspace: Workspace,
        worker: ModelWorker,
        *,
        runs_dir: Path | None = None,
        sync_wait: float = 60.0,
    ) -> None:
        self.workspace = workspace
        self.worker = worker
        self.runs_dir = None if runs_dir is None else Path(runs_dir).resolve(strict=True)
        self.sync_wait = float(sync_wait)

    # ---------------------------------------------------------------- dispatch
    def handle(
        self, method: str, segments: list[str], query: Mapping[str, list[str]], body: Any
    ) -> tuple[int, Any]:
        """``(status, payload)`` for one request; never raises for request errors."""
        try:
            return self._route(method, segments, query, body)
        except Exception as exc:  # mapped to a JSON error; internal errors carry no details
            status, kind = map_exception(exc)
            return status, {"error": {"status": status, "type": kind, "message": _message(exc)}}

    def _route(
        self, method: str, segments: list[str], query: Mapping[str, list[str]], body: Any
    ) -> tuple[int, Any]:
        if not segments:
            raise ApiError(404, "unknown endpoint")
        head, rest = segments[0], segments[1:]
        routes_get: dict[str, Callable[[list[str], Mapping[str, list[str]]], tuple[int, Any]]] = {
            "status": self._status,
            "model": self._model,
            "channels": self._channels,
            "channel": self._channel,
            "history": self._history,
            "jobs": self._jobs,
            "runs": self._runs,
            "scores": self._scores,
            "search": self._last_search,
        }
        routes_post: dict[str, Callable[[list[str], Any], tuple[int, Any]]] = {
            "stats": self._stats,
            "search": self._search,
            "stage": self._stage,
            "evaluate": self._evaluate,
            "commit": self._commit,
            "discard": self._discard,
            "restore": self._restore,
        }
        if method == "GET":
            if head in routes_get:
                return routes_get[head](rest, query)
            if head in routes_post:
                raise ApiError(405, f"/api/{head} needs POST")
        elif method == "POST":
            if head in routes_post:
                if rest:
                    raise ApiError(404, "unknown endpoint")
                return routes_post[head](rest, body)
            if head in routes_get:
                raise ApiError(405, f"/api/{head} needs GET")
        else:
            raise ApiError(405, f"method {method} is not allowed")
        raise ApiError(404, "unknown endpoint")

    # ---------------------------------------------------------------- job helpers
    def _async(self, kind: str, fn: Callable[[JobContext], Any], params: dict[str, Any]) -> tuple[int, Any]:
        job = self.worker.submit(kind, fn, params)
        return 202, {"job": job.to_dict(include_result=False)}

    def _sync(self, kind: str, fn: Callable[[JobContext], Any], params: dict[str, Any]) -> tuple[int, Any]:
        job = self.worker.call(kind, fn, params, wait=self.sync_wait)
        return self._job_response(job)

    @staticmethod
    def _job_response(job: Job) -> tuple[int, Any]:
        if not job.done_event.is_set():
            return 202, {"job": job.to_dict(include_result=False)}
        if job.status == "failed":
            error = job.error or {"status": 500, "type": "Error", "message": "job failed"}
            return int(error.get("status", 500)), {"error": error, "job": job.id}
        return 200, {"result": job.result, "job": job.id}

    # ---------------------------------------------------------------- GET endpoints
    def _no_rest(self, rest: list[str]) -> None:
        if rest:
            raise ApiError(404, "unknown endpoint")

    def _status(self, rest: list[str], query: Mapping[str, list[str]]) -> tuple[int, Any]:
        self._no_rest(rest)
        status = self.workspace.public_status()
        status["worker"] = {"current_job": self.worker.current_job, "pending": self.worker.pending}
        status["runs_dir"] = self.runs_dir is not None
        return 200, status

    def _model(self, rest: list[str], query: Mapping[str, list[str]]) -> tuple[int, Any]:
        self._no_rest(rest)
        return 200, self.workspace.public_model()

    def _scores(self, rest: list[str], query: Mapping[str, list[str]]) -> tuple[int, Any]:
        self._no_rest(rest)
        return 200, {"scores": self.workspace.public_scores()}

    def _history(self, rest: list[str], query: Mapping[str, list[str]]) -> tuple[int, Any]:
        self._no_rest(rest)
        return 200, self.workspace.public_history()

    def _last_search(self, rest: list[str], query: Mapping[str, list[str]]) -> tuple[int, Any]:
        self._no_rest(rest)
        last = self.workspace.public_last_search()
        if last is None:
            raise ApiError(404, "no search has run in this session")
        return 200, last

    def _channels(self, rest: list[str], query: Mapping[str, list[str]]) -> tuple[int, Any]:
        self._no_rest(rest)
        kind = _as_str(query, "score", "wanda") or "wanda"
        if kind not in SCORE_KINDS:
            raise ApiError(400, f"score must be one of {list(SCORE_KINDS)}")
        split: str | None = None
        if kind in ("wanda", "contrast"):
            split = parse_split(_as_str(query, "split", Split.TRAIN.value)).value
        else:
            raw = _as_str(query, "split", None)
            if raw:
                _reject_audit_text(raw, "split")
        layer = _as_int(query, "layer", None, "layer")
        group = _as_str(query, "group", None, limit=64)
        group = group or None
        bands = _require_int(query, "bands", 4, "bands")
        top = _require_int(query, "top", 50, "top")
        return 200, self.workspace.rank_channels(
            kind=kind, split=split, layer=layer, group=group, bands=bands, top=top
        )

    def _channel(self, rest: list[str], query: Mapping[str, list[str]]) -> tuple[int, Any]:
        if len(rest) != 2:
            raise ApiError(404, "use /api/channel/<layer>/<index>")
        try:
            layer, index = int(rest[0], 10), int(rest[1], 10)
        except ValueError:
            raise ApiError(400, "layer and index must be integers") from None
        if layer < 0 or index < 0:
            raise ApiError(400, "layer and index must be >= 0")
        k = _require_int(query, "k", 10, "k")
        ws = self.workspace
        return self._sync(
            "channel",
            lambda _ctx: ws.channel_detail(layer, index, k=k),
            {"layer": layer, "index": index, "k": k},
        )

    def _jobs(self, rest: list[str], query: Mapping[str, list[str]]) -> tuple[int, Any]:
        if not rest:
            return 200, {"jobs": [j.to_dict(include_result=False) for j in self.worker.list()]}
        if len(rest) != 1:
            raise ApiError(404, "use /api/jobs/<id>")
        job = self.worker.get(rest[0])
        if job is None:
            raise ApiError(404, f"no job {rest[0][:40]!r}")
        return 200, {"job": job.to_dict()}

    def _runs(self, rest: list[str], query: Mapping[str, list[str]]) -> tuple[int, Any]:
        if self.runs_dir is None:
            if not rest:
                return 200, {"runs": [], "note": "start the UI with --runs-dir to read decision logs"}
            raise ApiError(404, "no --runs-dir was given")
        if not rest:
            return 200, {"runs": list_runs(self.runs_dir)}
        if len(rest) != 2 or rest[1] != "decisions":
            raise ApiError(404, "use /api/runs/<name>/decisions")
        limit = _require_int(query, "limit", 1000, "decisions_limit")
        return 200, read_decisions(self.runs_dir, rest[0], limit=limit)

    # ---------------------------------------------------------------- POST endpoints
    def _stats(self, rest: list[str], body: Any) -> tuple[int, Any]:
        b = _check_body(body, {"split", "n", "slice", "contrast", "threshold"})
        split = parse_split(b.get("split", Split.TRAIN.value))
        slice_name = _as_str(b, "slice", None)
        if slice_name:
            _reject_audit_text(slice_name, "slice")
        else:
            slice_name = None
        n = _require_int(b, "n", 64, "stats_n")
        contrast = _as_bool(b, "contrast", False)
        threshold = _require_float(b, "threshold", 0.0, "threshold")
        ws = self.workspace
        params = {
            "split": split.value,
            "n": n,
            "slice": slice_name,
            "contrast": contrast,
            "threshold": threshold,
        }
        return self._async(
            "stats",
            lambda ctx: ws.collect_stats(
                split, n, slice_name=slice_name, contrast=contrast, threshold=threshold, ctx=ctx
            ),
            params,
        )

    def _search(self, rest: list[str], body: Any) -> tuple[int, Any]:
        fields = {
            "score",
            "split",
            "bands",
            "layers",
            "groups",
            "rounds",
            "burst_updates",
            "survivors_fraction",
            "repeats_final",
            "controls",
            "items_per_transfer_slice",
            "items_per_retention_slice",
            "population",
            "sigma",
            "lr",
            "max_step_norm",
            "fitness_items",
            "seed",
        }
        b = _check_body(body, fields)
        kind = _as_str(b, "score", "wanda") or "wanda"
        if kind not in SCORE_KINDS:
            raise ApiError(400, f"score must be one of {list(SCORE_KINDS)}")
        split = (
            parse_split(b.get("split", Split.TRAIN.value)).value if kind in ("wanda", "contrast") else None
        )
        layers_raw = b.get("layers")
        layers: tuple[int, ...] | None = None
        if layers_raw is not None:
            if not isinstance(layers_raw, list) or not layers_raw or len(layers_raw) > 256:
                raise ApiError(400, "layers must be a non-empty list of layer indices")
            if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in layers_raw):
                raise ApiError(400, "layers must be non-negative integers")
            layers = tuple(sorted(set(layers_raw)))
        groups_raw = b.get("groups")
        groups: tuple[str, ...] | None = None
        if groups_raw is not None:
            if not isinstance(groups_raw, list) or not groups_raw or len(groups_raw) > MAX_SEARCH_CANDIDATES:
                raise ApiError(400, f"groups must be a list of 1..{MAX_SEARCH_CANDIDATES} group names")
            if not all(isinstance(g, str) and 0 < len(g) <= 64 for g in groups_raw):
                raise ApiError(400, "group names must be short strings")
            if len(set(groups_raw)) != len(groups_raw):
                raise ApiError(400, "duplicate group names")
            groups = tuple(groups_raw)
        controls_raw = b.get("controls", list(CONTROL_KINDS))
        if not isinstance(controls_raw, list) or not all(isinstance(c, str) for c in controls_raw):
            raise ApiError(400, f"controls must be a list drawn from {list(CONTROL_KINDS)}")
        if any(c not in CONTROL_KINDS for c in controls_raw) or len(set(controls_raw)) != len(controls_raw):
            raise ApiError(400, f"controls must be distinct values from {list(CONTROL_KINDS)}")
        frac = _require_float(b, "survivors_fraction", 0.5, "fraction")
        population = _require_int(b, "population", 4, "population")
        if population % 2:
            raise ApiError(400, "population must be even (antithetic pairs)")
        params = SearchParams(
            score=kind,
            split=split,
            bands=_require_int(b, "bands", 4, "bands"),
            layers=layers,
            groups=groups,
            rounds=_require_int(b, "rounds", 1, "rounds"),
            burst_updates=_require_int(b, "burst_updates", 1, "burst_updates"),
            survivors_fraction=frac,
            repeats_final=_require_int(b, "repeats_final", 1, "repeats_final"),
            controls=tuple(controls_raw),
            items_per_transfer_slice=_require_int(b, "items_per_transfer_slice", 32, "search_items_transfer"),
            items_per_retention_slice=_require_int(
                b, "items_per_retention_slice", 64, "search_items_retention"
            ),
            population=population,
            sigma=_require_float(b, "sigma", 0.05, "sigma"),
            lr=_require_float(b, "lr", 0.05, "lr"),
            max_step_norm=_as_float(b, "max_step_norm", 0.5, "max_step_norm"),
            fitness_items=_require_int(b, "fitness_items", 16, "fitness_items"),
            seed=_require_int(b, "seed", 0, "seed"),
        )
        ws = self.workspace
        return self._async(
            "search",
            lambda ctx: ws.run_search(params, max_candidates=MAX_SEARCH_CANDIDATES, ctx=ctx),
            params.to_dict(),
        )

    def _stage(self, rest: list[str], body: Any) -> tuple[int, Any]:
        if not isinstance(body, dict):
            raise ApiError(400, "the request body must be a JSON object")
        if "edits" in body:
            b = _check_body(body, {"edits"})
            edits = b["edits"]
            lo, hi = LIMITS["edits"]
            if not isinstance(edits, list) or not lo <= len(edits) <= hi:
                raise ApiError(400, f"edits must be a list of {int(lo)}..{int(hi)} edit records")
        else:
            edits = [body]
        if not all(isinstance(e, dict) for e in edits):
            raise ApiError(400, "every edit must be a JSON object (a declarative EditRecord)")
        ws = self.workspace
        return self._sync("stage", lambda _ctx: ws.stage(edits), {"edits": len(edits)})

    def _evaluate(self, rest: list[str], body: Any) -> tuple[int, Any]:
        _check_body(body, set())
        ws = self.workspace
        return self._async("evaluate", lambda ctx: ws.evaluate(ctx), {})

    def _commit(self, rest: list[str], body: Any) -> tuple[int, Any]:
        b = _check_body(body, {"override", "note"})
        override = _as_str(b, "override", None, limit=10_000)
        if override is not None and not override.strip():
            override = None
        note = _as_str(b, "note", "", limit=10_000) or ""
        ws = self.workspace
        return self._sync(
            "commit", lambda _ctx: ws.commit(override, note), {"override": override is not None}
        )

    def _discard(self, rest: list[str], body: Any) -> tuple[int, Any]:
        _check_body(body, set())
        ws = self.workspace
        return self._sync("discard", lambda _ctx: ws.discard(), {})

    def _restore(self, rest: list[str], body: Any) -> tuple[int, Any]:
        b = _check_body(body, {"version"})
        version = _as_str(b, "version", None, limit=64)
        if not version:
            raise ApiError(400, "version is required")
        ws = self.workspace
        return self._sync("restore", lambda _ctx: ws.restore(version), {"version": version})
