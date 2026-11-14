"""Code-execution rewards - an interface that is **disabled by default**.

Running model-generated code is dangerous, so this module ships no backend that executes code in the
local process or in a local subprocess. What exists:

* :class:`SandboxPolicy` - what the operator allows (``enabled=False`` and ``network_access=False`` by
  default, time / memory / size limits).
* :class:`SandboxBackend` - the protocol for an *isolated* executor; it must declare
  :class:`IsolationGuarantees`.
* :class:`CodeExecutionVerifier` - extracts the last fenced code block from a completion, runs it on the
  reference test cases through the backend and compares stdout. It raises :class:`SandboxDisabledError`
  (at construction and again before every verification) unless ``policy.enabled`` is true, a backend is
  supplied, and the backend declares guarantees covering the policy: isolated from the trainer host, no
  network, enforced time and memory limits. ``network_access=True`` is always refused.
* :class:`SandboxFusionBackend` - an HTTP client for an external SandboxFusion server
  (``SandboxFusion/run_sandbox.sh`` starts one; this repo's scripts POST to ``http://<host>:8080/run_code/``).
  It requires an explicit endpoint URL **and** an explicit operator declaration of the server's isolation
  guarantees - the client cannot verify them. Status: tested only against a fake transport; it has
  **NOT_TESTED** against a live SandboxFusion server. The request fields ``code``, ``language``,
  ``run_timeout``, ``compile_timeout`` and the response field ``run_result.{return_code,stdout,stderr}`` are
  used by this repository's own scripts; ``stdin``, ``memory_limit_MB``, ``status`` /
  ``run_result.status`` / ``compile_result`` follow the upstream SandboxFusion API and are unverified here,
  so unknown response shapes map to ``status="sandbox_error"``.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from research_rl.core.registry import VERIFIERS
from research_rl.rewards.verifier_interface import (
    BaseVerifier,
    FatalVerifierError,
    RewardScheme,
    Status,
    VerificationResult,
)

ExecStatus = Literal["ok", "runtime_error", "compile_error", "timeout", "sandbox_error"]


class SandboxDisabledError(FatalVerifierError):
    """Code execution was requested but the sandbox is disabled or not safely configured."""


@dataclass(frozen=True)
class SandboxPolicy:
    """Operator policy for executing generated code. Everything is off / tight by default."""

    enabled: bool = False
    network_access: bool = False
    time_limit_s: float = 10.0
    memory_limit_mb: int = 1024
    max_code_chars: int = 65536
    max_output_chars: int = 65536
    languages: tuple[str, ...] = ("python",)

    def __post_init__(self) -> None:
        if not 0 < self.time_limit_s <= 600:
            raise ValueError(f"time_limit_s must be in (0, 600], got {self.time_limit_s}")
        if not 0 < self.memory_limit_mb <= 65536:
            raise ValueError(f"memory_limit_mb must be in (0, 65536], got {self.memory_limit_mb}")
        if self.max_code_chars <= 0 or self.max_output_chars <= 0:
            raise ValueError("max_code_chars and max_output_chars must be positive")
        if not self.languages:
            raise ValueError("languages must not be empty")


@dataclass(frozen=True)
class IsolationGuarantees:
    """What a backend promises about where and how code runs. Defaults promise nothing."""

    isolated_from_host: bool = False  # code never runs in the trainer's process, host user or filesystem
    network_disabled: bool = False
    enforces_time_limit: bool = False
    enforces_memory_limit: bool = False


@dataclass(frozen=True)
class ExecutionRequest:
    code: str
    language: str
    stdin: str
    time_limit_s: float
    memory_limit_mb: int


@dataclass(frozen=True)
class ExecutionResult:
    status: ExecStatus
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SandboxBackend(Protocol):
    """An isolated code executor. Implementations must never run code in the caller's process."""

    @property
    def guarantees(self) -> IsolationGuarantees: ...

    def run(self, request: ExecutionRequest) -> ExecutionResult: ...


def check_sandbox(policy: SandboxPolicy, backend: SandboxBackend | None) -> None:
    """Raise :class:`SandboxDisabledError` unless ``policy`` + ``backend`` form a safe configuration."""
    if not policy.enabled:
        raise SandboxDisabledError("code execution is disabled (SandboxPolicy.enabled=False)")
    if policy.network_access:
        raise SandboxDisabledError("network access for generated code is not supported")
    if backend is None:
        raise SandboxDisabledError("no sandbox backend supplied; ResearchRL ships no local executor")
    if not isinstance(backend, SandboxBackend):
        raise SandboxDisabledError(f"{type(backend).__name__} does not implement SandboxBackend")
    g = backend.guarantees
    if not isinstance(g, IsolationGuarantees):
        raise SandboxDisabledError("backend.guarantees must be an IsolationGuarantees instance")
    missing = [
        name
        for name, ok in (
            ("isolated_from_host", g.isolated_from_host),
            ("network_disabled", g.network_disabled),
            ("enforces_time_limit", g.enforces_time_limit),
            ("enforces_memory_limit", g.enforces_memory_limit),
        )
        if ok is not True
    ]
    if missing:
        raise SandboxDisabledError(f"backend does not guarantee: {', '.join(missing)}")


def extract_code(text: str, language: str = "python") -> str | None:
    """Body of the last fenced code block (```` ```python ```` / ```` ```py ```` / bare ```` ``` ````).

    Blocks tagged with another language are skipped. Linear scan, no regex.
    """
    aliases = {language.lower(), *(("py", "python3") if language.lower() == "python" else ())}
    found: str | None = None
    pos = 0
    while True:
        start = text.find("```", pos)
        if start < 0:
            return found
        header_end = text.find("\n", start + 3)
        if header_end < 0:
            return found
        close = text.find("```", header_end + 1)
        if close < 0:
            return found
        tag = text[start + 3 : header_end].strip().lower()
        if tag == "" or tag in aliases:
            found = text[header_end + 1 : close]
        pos = close + 3


def _normalize_output(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip("\n").splitlines()).rstrip()


def _test_cases(reference: Any) -> list[tuple[str, str]]:
    """``"out"`` | ``{"input", "output"}`` | ``{"tests": [...]}`` | ``[{"input", "output"}, ...]``."""
    if isinstance(reference, str):
        return [("", reference)]
    if isinstance(reference, Mapping):
        if "tests" in reference:
            return _test_cases(list(reference["tests"]))
        return [(str(reference.get("input", "")), str(reference["output"]))]
    if isinstance(reference, Sequence) and reference:
        cases: list[tuple[str, str]] = []
        for case in reference:
            if not isinstance(case, Mapping):
                raise ValueError("each test case must be a mapping with 'input' and 'output'")
            cases.extend(_test_cases(case))
        return cases
    raise ValueError(f"unsupported code reference {type(reference).__name__}")


class CodeExecutionVerifier(BaseVerifier):
    """stdout-matching code verifier on top of an isolated :class:`SandboxBackend` (disabled by default).

    All test cases must pass for ``status="correct"``; a backend failure gives ``status="error"`` and a
    compile / runtime error, timeout or wrong output ``status="incorrect"``.
    """

    name = "code_execution"

    def __init__(
        self,
        policy: SandboxPolicy | None = None,
        backend: SandboxBackend | None = None,
        *,
        language: str = "python",
        rewards: RewardScheme | None = None,
    ) -> None:
        super().__init__(rewards)
        self.policy = policy if policy is not None else SandboxPolicy()
        self.backend = backend
        self.language = language
        check_sandbox(self.policy, backend)
        if language not in self.policy.languages:
            raise SandboxDisabledError(f"language {language!r} not allowed by policy {self.policy.languages}")

    def verify(self, completion: str | None, reference: Any, **meta: Any) -> VerificationResult:
        check_sandbox(self.policy, self.backend)  # re-checked: guarantees may be dynamic properties
        return super().verify(completion, reference, **meta)

    def _verify(
        self, completion: str | None, reference: Any, *, truncated: bool = False, **meta: Any
    ) -> tuple[Status, str | None, str | None, dict[str, Any]]:
        if truncated:
            return "truncated", None, None, {"reason": "cut_by_length"}
        try:
            cases = _test_cases(reference)
        except (KeyError, ValueError) as exc:
            return "error", None, None, {"reason": "bad_reference", "message": str(exc)}
        code = extract_code(completion or "", self.language)
        if code is None:
            return "no_answer", None, None, {"reason": "no_code_block"}
        if not code.strip():
            return "format_error", None, None, {"reason": "empty_code"}
        if len(code) > self.policy.max_code_chars:
            return "format_error", None, None, {"reason": "code_too_long"}
        assert self.backend is not None  # guaranteed by check_sandbox
        outcomes: list[str] = []
        for stdin, expected in cases:
            result = self.backend.run(
                ExecutionRequest(
                    code=code,
                    language=self.language,
                    stdin=stdin,
                    time_limit_s=self.policy.time_limit_s,
                    memory_limit_mb=self.policy.memory_limit_mb,
                )
            )
            if result.status == "sandbox_error":
                return "error", code, None, {"reason": "sandbox_error", "backend": result.details}
            stdout = result.stdout[: self.policy.max_output_chars]
            ok = result.status == "ok" and _normalize_output(stdout) == _normalize_output(expected)
            outcomes.append("pass" if ok else result.status if result.status != "ok" else "wrong_output")
        passed = outcomes.count("pass")
        details = {"passed": passed, "total": len(cases), "outcomes": outcomes}
        return ("correct" if passed == len(cases) else "incorrect"), code, None, details


Transport = Callable[[str, bytes, Mapping[str, str], float], tuple[int, bytes]]


def urllib_transport(url: str, body: bytes, headers: Mapping[str, str], timeout: float) -> tuple[int, bytes]:
    """Default HTTP POST transport (stdlib only)."""
    req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # scheme validated in __init__
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()


class SandboxFusionBackend:
    """HTTP client for an external SandboxFusion ``run_code`` endpoint (see module docstring caveats).

    Args:
        url: full endpoint URL, e.g. ``"http://127.0.0.1:8080/run_code"``. Required; there is no default
            and no environment-variable fallback.
        guarantees: the operator's declaration of what *their* SandboxFusion deployment enforces (e.g.
            container without network, cgroup memory limit). Required; nothing is assumed.
        transport: ``(url, body, headers, timeout) -> (http_status, body)``; defaults to
            :func:`urllib_transport`. Tests inject a fake transport.
    """

    def __init__(
        self,
        url: str,
        guarantees: IsolationGuarantees,
        *,
        transport: Transport | None = None,
        request_timeout_s: float = 60.0,
    ) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"SandboxFusion url must be an explicit http(s) URL, got {url!r}")
        if not isinstance(guarantees, IsolationGuarantees):
            raise TypeError("guarantees must be an IsolationGuarantees declaration")
        self.url = url
        self._guarantees = guarantees
        self.transport: Transport = transport if transport is not None else urllib_transport
        self.request_timeout_s = request_timeout_s

    @property
    def guarantees(self) -> IsolationGuarantees:
        return self._guarantees

    def payload(self, request: ExecutionRequest) -> dict[str, Any]:
        return {
            "code": request.code,
            "language": request.language,
            "stdin": request.stdin,
            "run_timeout": request.time_limit_s,
            "compile_timeout": request.time_limit_s,
            "memory_limit_MB": request.memory_limit_mb,
        }

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        body = json.dumps(self.payload(request)).encode()
        headers = {"Content-Type": "application/json"}
        try:
            http_status, raw = self.transport(self.url, body, headers, self.request_timeout_s)
        except OSError as exc:
            return ExecutionResult("sandbox_error", details={"error": f"{type(exc).__name__}: {exc}"})
        if http_status != 200:
            return ExecutionResult("sandbox_error", details={"http_status": http_status})
        try:
            resp = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return ExecutionResult("sandbox_error", details={"error": "invalid JSON response"})
        return parse_sandbox_fusion_response(resp)


def parse_sandbox_fusion_response(resp: Any) -> ExecutionResult:
    """Map a SandboxFusion ``run_code`` JSON response to an :class:`ExecutionResult` (defensively)."""
    if not isinstance(resp, Mapping):
        return ExecutionResult("sandbox_error", details={"error": "response is not an object"})
    if resp.get("status") == "SandboxError":
        return ExecutionResult("sandbox_error", details={"message": str(resp.get("message", ""))})
    run = resp.get("run_result")
    compile_result = resp.get("compile_result")
    if not isinstance(run, Mapping):
        if isinstance(compile_result, Mapping) and compile_result.get("return_code") not in (0, None):
            return ExecutionResult(
                "compile_error",
                stderr=str(compile_result.get("stderr") or ""),
                return_code=_as_int(compile_result.get("return_code")),
            )
        return ExecutionResult("sandbox_error", details={"error": "missing run_result"})
    stdout, stderr = str(run.get("stdout") or ""), str(run.get("stderr") or "")
    return_code = _as_int(run.get("return_code"))
    if run.get("status") == "TimeLimitExceeded":
        return ExecutionResult("timeout", stdout, stderr, return_code)
    if return_code is None:
        return ExecutionResult("sandbox_error", stdout, stderr, details={"error": "missing return_code"})
    return ExecutionResult("ok" if return_code == 0 else "runtime_error", stdout, stderr, return_code)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"-?[0-9]{1,18}", value.strip()):
        return int(value.strip())
    return None


VERIFIERS.register("code_execution")(CodeExecutionVerifier)
