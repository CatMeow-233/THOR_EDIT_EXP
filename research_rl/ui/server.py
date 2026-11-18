"""Local HTTP server of the UI (Python stdlib ``http.server`` only).

Security model (``CAPABILITY.md`` 1.13):

* **Bind address.** ``127.0.0.1`` by default. A non-loopback address is refused unless
  ``--allow-remote`` is given, and then a warning is printed. The host must be an IP literal or
  ``localhost``.
* **Session token.** Generated with :func:`secrets.token_urlsafe` at start (or taken from the
  environment variable ``RESEARCH_RL_UI_TOKEN``), printed **once** to stdout inside the URL
  fragment (``#token=...``, which browsers never send to the server) and never written to disk.
  Every ``/api`` request must carry it in ``X-Session-Token``; it is compared with
  :func:`hmac.compare_digest` on bytes. Missing or wrong -> ``401``.
* **Host / Origin.** The ``Host`` header must be exactly the bound ``host:port`` (for a loopback
  bind also ``localhost:port``), on every request including static files (DNS-rebinding
  protection); an ``Origin`` header, if present, must be ``http://`` + an allowed host; an API
  request with ``Sec-Fetch-Site: cross-site`` is refused (the static page may be opened from a
  link elsewhere). Violations -> ``403``. Order of checks: Host, Origin, (cross-site,) token.
  Wildcard binds (``0.0.0.0``, ``::``) are refused: bind the address clients will use.
* **Bodies.** ``POST`` bodies must declare ``Content-Length`` (chunked bodies are refused, 411),
  be at most ``max_body`` bytes (default 1 MiB; larger -> ``413`` without reading the body, and
  the connection is closed), have ``Content-Type: application/json`` (415) and parse as a JSON
  object (400; ``NaN`` / ``Infinity`` constants are rejected).
* **Static files** are served only from the package's ``static`` directory: the path is split
  into segments, each percent-decoded once; ``..``, ``.``, empty segments, backslashes, NUL and
  encoded slashes are refused (403); the resolved path must lie inside the resolved static
  directory (403) and be a regular file with an allow-listed suffix (else 404). No directory
  listing.
* **Headers.** Every response (errors included) is sent with
  ``Content-Security-Policy: default-src 'self'; ... frame-ancestors 'none'``,
  ``X-Content-Type-Options: nosniff``, ``X-Frame-Options: DENY``, ``Referrer-Policy: no-referrer``
  and ``Cache-Control: no-store``. Errors are JSON.
* **Connections.** At most ``max_connections`` (default 64) requests are handled at once; a
  connection over the cap is closed on accept, before a thread is started or anything is read. The
  request line and headers must arrive within ``header_timeout`` seconds in total (default 30;
  every socket operation also has a 30 s timeout): a client that trickles header bytes is
  disconnected and its partial request is never dispatched. This bounds the threads and the time
  a client without the token can hold; a client that keeps reconnecting can still occupy the slots
  (the server is meant for one local user, not for hostile networks).

Model access goes through :class:`~research_rl.ui.jobs.ModelWorker` (see :mod:`research_rl.ui.api`).
"""

from __future__ import annotations

import argparse
import contextlib
import hmac
import ipaddress
import json
import logging
import math
import os
import re
import secrets
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote

from research_rl.ui.api import Api, error_payload
from research_rl.ui.jobs import ModelWorker
from research_rl.ui.loading import LoadedModel, ModelLoadError, build_tiny_demo, load_model_dir
from research_rl.ui.workspace import Workspace, WorkspaceConfig, call_on_worker

__all__ = [
    "DEFAULT_HEADER_TIMEOUT",
    "DEFAULT_MAX_BODY",
    "DEFAULT_MAX_CONNECTIONS",
    "MAX_JSON_DEPTH",
    "STATIC_DIR",
    "TOKEN_ENV",
    "BindError",
    "ServerOptions",
    "UIApp",
    "UIServer",
    "build_parser",
    "check_bind_host",
    "create_app",
    "json_depth",
    "jsonable",
    "main",
    "resolve_token",
]

logger = logging.getLogger("research_rl.ui")

TOKEN_ENV = "RESEARCH_RL_UI_TOKEN"
DEFAULT_MAX_BODY = 1024 * 1024
DEFAULT_MAX_CONNECTIONS = 64
"""Requests handled at once; further connections are closed on accept."""
DEFAULT_HEADER_TIMEOUT = 30.0
"""Seconds a client has, in total, to send the request line and headers."""
MAX_JSON_DEPTH = 32
"""Deepest nesting of arrays / objects accepted in a request body (edit records need about 4)."""
_MAX_LENGTH_DIGITS = 15
STATIC_DIR = Path(__file__).resolve().parent / "static"
_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{16,256}")
_STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)
SECURITY_HEADERS = (
    ("Content-Security-Policy", CSP),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "no-referrer"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Cache-Control", "no-store"),
)


class BindError(ValueError):
    """The bind address is not allowed (non-loopback without ``allow_remote``, or not an IP)."""


def json_depth(obj: Any) -> int:
    """Nesting depth of a parsed JSON value (a scalar is 0, ``[]`` / ``{}`` 1), computed iteratively."""
    deepest = 0
    stack: list[tuple[Any, int]] = [(obj, 0)]
    while stack:
        value, depth = stack.pop()
        if isinstance(value, dict):
            children: Any = value.values()
        elif isinstance(value, list):
            children = value
        else:
            deepest = max(deepest, depth)
            continue
        deepest = max(deepest, depth + 1)
        stack.extend((child, depth + 1) for child in children)
    return deepest


def jsonable(obj: Any) -> Any:
    """Copy of ``obj`` that ``json.dumps(..., allow_nan=False)`` accepts: non-finite floats become
    ``None``, tuples / sets lists, enums their value, other objects their ``str``."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [jsonable(v) for v in sorted(obj, key=str)]
    value = getattr(obj, "value", None)
    if isinstance(value, (str, int)):
        return value
    return str(obj)


def check_bind_host(host: str, allow_remote: bool) -> tuple[str, bool]:
    """Normalise the bind host; returns ``(ip, is_loopback)``.

    ``localhost`` becomes ``127.0.0.1``. Anything that is not an IP literal is refused, so is a
    wildcard address (``0.0.0.0`` / ``::``: the Host allowlist must name the address clients
    connect to), and a non-loopback address is refused unless ``allow_remote`` (:class:`BindError`)."""
    name = host.strip()
    if name.lower() == "localhost":
        name = "127.0.0.1"
    try:
        ip = ipaddress.ip_address(name.strip("[]"))
    except ValueError:
        raise BindError(f"--host must be an IP address or 'localhost', got {host!r}") from None
    loopback = ip.is_loopback
    if ip.is_unspecified:
        raise BindError(
            f"refusing the wildcard address {ip}: the Host check needs the address clients use; "
            "bind 127.0.0.1, or (with --allow-remote) the IP of one interface"
        )
    if not loopback and not allow_remote:
        raise BindError(
            f"refusing to bind to the non-loopback address {ip}: the UI can change model weights. "
            "Pass --allow-remote to do it anyway (anyone who can reach the port and has the token gets access)."
        )
    return str(ip), loopback


def resolve_token(env: dict[str, str] | None = None) -> str:
    """The session token: ``RESEARCH_RL_UI_TOKEN`` if set (16-256 URL-safe characters, else
    ``ValueError``), otherwise :func:`secrets.token_urlsafe` (32 bytes)."""
    source = os.environ if env is None else env
    given = source.get(TOKEN_ENV)
    if given is not None and given != "":
        if not _TOKEN_RE.fullmatch(given):
            raise ValueError(f"{TOKEN_ENV} must be 16-256 characters from [A-Za-z0-9_-]")
        return given
    return secrets.token_urlsafe(32)


@dataclass
class ServerOptions:
    """Network and security settings of :class:`UIServer`."""

    host: str = "127.0.0.1"
    port: int = 8765
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32), repr=False)
    allow_remote: bool = False
    max_body: int = DEFAULT_MAX_BODY
    static_dir: Path = STATIC_DIR
    max_connections: int = DEFAULT_MAX_CONNECTIONS
    header_timeout: float = DEFAULT_HEADER_TIMEOUT


class UIServer(ThreadingHTTPServer):
    """Threading HTTP server holding the API, the token and the allowed Host / Origin values.

    Concurrency is capped at ``options.max_connections`` (a connection over the cap is closed on
    accept) and the request line + headers of each connection must arrive within
    ``options.header_timeout`` seconds in total; one watchdog thread shuts down connections past
    that deadline (a per-read socket timeout alone does not stop a client that trickles bytes)."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, options: ServerOptions, api: Api) -> None:
        ip, loopback = check_bind_host(options.host, options.allow_remote)
        if not _TOKEN_RE.fullmatch(options.token):
            raise ValueError("the session token must be 16-256 characters from [A-Za-z0-9_-]")
        if options.max_body < 1:
            raise ValueError("max_body must be >= 1")
        if options.max_connections < 1:
            raise ValueError("max_connections must be >= 1")
        if not (math.isfinite(options.header_timeout) and options.header_timeout > 0):
            raise ValueError("header_timeout must be a finite number of seconds > 0")
        self.options = options
        self.api = api
        self.loopback = loopback
        self.address_family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        self._token = options.token.encode("ascii")
        self.static_dir = Path(options.static_dir).resolve(strict=True)
        super().__init__((ip, int(options.port)), UIRequestHandler)
        host, port = self.server_address[0], self.server_address[1]
        self.bound_host = str(host)
        self.bound_port = int(port)
        shown = f"[{self.bound_host}]" if ":" in self.bound_host else self.bound_host
        self.authority = f"{shown}:{self.bound_port}"
        allowed = {self.authority}
        if loopback:
            allowed.add(f"localhost:{self.bound_port}")
        self.allowed_hosts = frozenset(allowed)
        self.allowed_origins = frozenset(f"http://{h}" for h in allowed)
        self.connections_refused = 0
        """Connections closed on accept because ``max_connections`` requests were in progress."""
        self.header_timeouts = 0
        """Connections dropped because their request line + headers missed ``header_timeout``."""
        self._slots = threading.BoundedSemaphore(options.max_connections)
        self._header_deadlines: dict[socket.socket, float] = {}
        self._deadline_lock = threading.Lock()
        self._watchdog_stop = threading.Event()
        self._watchdog = threading.Thread(
            target=self._watch_headers, name="research-rl-ui-watchdog", daemon=True
        )
        self._watchdog.start()

    # ---------------------------------------------------------------- connection cap
    def process_request(self, request: Any, client_address: Any) -> None:
        """Start a handler thread, or close the connection when ``max_connections`` are in progress
        (runs on the accepting thread: nothing is read or written here)."""
        if not self._slots.acquire(blocking=False):
            self.connections_refused += 1
            log = logger.warning if self.connections_refused == 1 else logger.debug  # no log flood
            log("connection limit (%d) reached: closing new connections", self.options.max_connections)
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        """Handle one connection on its thread and free its slot afterwards."""
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    # ---------------------------------------------------------------- header deadline
    def begin_header_phase(self, conn: socket.socket) -> None:
        """Start the ``header_timeout`` clock of a new connection."""
        with self._deadline_lock:
            self._header_deadlines[conn] = time.monotonic() + self.options.header_timeout

    def end_header_phase(self, conn: socket.socket) -> bool:
        """Stop the clock; ``False`` if the deadline had already fired (the connection is dead)."""
        with self._deadline_lock:
            return self._header_deadlines.pop(conn, None) is not None

    def _watch_headers(self) -> None:
        interval = min(1.0, self.options.header_timeout / 4)
        while not self._watchdog_stop.wait(interval):
            now = time.monotonic()
            with self._deadline_lock:
                expired = [c for c, deadline in self._header_deadlines.items() if deadline <= now]
                for conn in expired:
                    del self._header_deadlines[conn]
            for conn in expired:
                self.header_timeouts += 1
                logger.info(
                    "dropping a connection that did not send its headers within %ss",
                    self.options.header_timeout,
                )
                with contextlib.suppress(OSError):
                    conn.shutdown(socket.SHUT_RDWR)  # the blocked read in the handler thread returns EOF

    def server_close(self) -> None:
        """Close the listening socket and stop the header watchdog."""
        self._watchdog_stop.set()
        super().server_close()

    @property
    def url(self) -> str:
        """Base URL of the UI (without the token)."""
        return f"http://{self.authority}/"

    def token_url(self) -> str:
        """The URL with the token in its fragment (printed once at start; never logged)."""
        return f"{self.url}#token={self.options.token}"

    def token_ok(self, presented: str | None) -> bool:
        """Constant-time comparison of a presented token (bytes; non-latin-1 values never match)."""
        if presented is None:
            return False
        try:
            raw = presented.encode("latin-1")
        except UnicodeEncodeError:
            return False
        return hmac.compare_digest(raw, self._token)


def _segments(raw_path: str) -> list[str] | None:
    """Path segments of ``raw_path`` (each percent-decoded once), or ``None`` if any segment is
    unsafe (``..``, ``.``, empty inside the path, NUL, backslash, an encoded ``/``)."""
    if not raw_path.startswith("/"):
        return None
    parts = raw_path[1:].split("/")
    if parts and parts[-1] == "":
        parts = parts[:-1]
    out: list[str] = []
    for part in parts:
        if part == "":
            return None
        decoded = unquote(part, errors="strict") if "%" in part else part
        if decoded in (".", "..") or any(ch in decoded for ch in ("/", "\\", "\x00")):
            return None
        out.append(decoded)
    return out


class UIRequestHandler(BaseHTTPRequestHandler):
    """Request handler: security checks, static files and the JSON API."""

    server: UIServer
    server_version = "research-rl-ui"
    sys_version = ""
    protocol_version = "HTTP/1.0"
    timeout = 30.0  # seconds per socket operation; the total header deadline is UIServer's watchdog

    # ---------------------------------------------------------------- header deadline
    def setup(self) -> None:
        """Start the connection's header deadline (see :class:`UIServer`)."""
        super().setup()
        self.server.begin_header_phase(self.connection)

    def parse_request(self) -> bool:
        """Parse the request; refuse it if the header deadline fired while it was being read (the
        watchdog shut the socket down, so the headers read so far may be truncated)."""
        try:
            ok = super().parse_request()
        finally:
            in_time = self.server.end_header_phase(self.connection)
        if not in_time:
            self.close_connection = True
            return False
        return ok

    def finish(self) -> None:
        """Flush and close; always clears the header deadline."""
        try:
            super().finish()
        finally:
            self.server.end_header_phase(self.connection)

    # ---------------------------------------------------------------- logging (never the token)
    def log_message(self, format: str, *args: Any) -> None:
        """Request log through :mod:`logging` (the token is in a header, never in the line)."""
        logger.info("%s %s", self.address_string(), format % args)

    # ---------------------------------------------------------------- responses
    def _headers(self, status: int, content_type: str, length: int, *, close: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        for name, value in SECURITY_HEADERS:
            self.send_header(name, value)
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()

    def _send_json(self, status: int, payload: Any, *, close: bool = False) -> None:
        body = json.dumps(jsonable(payload), allow_nan=False, ensure_ascii=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(body), close=close)
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, status: int, message: str, *, close: bool = False) -> None:
        self._send_json(status, {"error": {"status": status, "message": message}}, close=close)

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        """JSON errors with the security headers (also for errors raised by the base class)."""
        try:
            phrase = HTTPStatus(code).phrase
        except ValueError:
            phrase = "error"
        self.close_connection = True
        with contextlib.suppress(OSError):  # the client (or the header watchdog) already closed the socket
            self._error(code, message or phrase, close=True)

    # ---------------------------------------------------------------- methods
    def do_GET(self) -> None:
        """Static files and GET API endpoints."""
        self._dispatch("GET")

    def do_POST(self) -> None:
        """POST API endpoints (JSON bodies)."""
        self._dispatch("POST")

    def _method_not_allowed(self) -> None:
        self._error(405, f"method {self.command} is not allowed")

    do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = do_TRACE = do_CONNECT = _method_not_allowed

    # ---------------------------------------------------------------- checks
    def _check_host_origin(self) -> str | None:
        hosts = self.headers.get_all("Host") or []
        if len(hosts) != 1 or hosts[0].strip().lower() not in self.server.allowed_hosts:
            return "bad Host header"
        origins = self.headers.get_all("Origin") or []
        if len(origins) > 1 or (origins and origins[0].strip().lower() not in self.server.allowed_origins):
            return "foreign Origin"
        return None

    def _cross_site(self) -> bool:
        return (self.headers.get("Sec-Fetch-Site") or "").strip().lower() == "cross-site"

    def _read_json_body(self) -> tuple[bool, Any]:
        if self.headers.get("Transfer-Encoding"):
            self._error(411, "chunked request bodies are not accepted; send Content-Length", close=True)
            return False, None
        lengths = self.headers.get_all("Content-Length") or []
        if not lengths:
            self._error(411, "Content-Length is required", close=True)
            return False, None
        text = lengths[0].strip() if len(lengths) == 1 else ""
        if not (text.isascii() and text.isdigit()):  # ASCII only: int() rejects e.g. superscripts
            self._error(400, "invalid Content-Length", close=True)
            return False, None
        digits = text.lstrip("0") or "0"
        # bounded before int(): a value with thousands of digits is over any cap (and over int()'s limit)
        if len(digits) > _MAX_LENGTH_DIGITS or int(digits) > self.server.options.max_body:
            self._error(413, f"request body exceeds {self.server.options.max_body} bytes", close=True)
            return False, None
        length = int(digits)
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            self._error(415, "Content-Type must be application/json", close=True)
            return False, None
        raw = self.rfile.read(length) if length else b""
        if len(raw) != length:
            self._error(400, "incomplete request body", close=True)
            return False, None
        if not raw.strip():
            return True, {}

        def no_constants(name: str) -> Any:
            raise ValueError(f"non-standard JSON constant {name}")

        try:
            body = json.loads(raw.decode("utf-8"), parse_constant=no_constants)
        except (UnicodeDecodeError, ValueError, RecursionError):
            # ValueError covers JSONDecodeError and integers over Python's int-string limit
            self._error(400, "the request body is not valid JSON")
            return False, None
        if not isinstance(body, dict):
            self._error(400, "the request body must be a JSON object")
            return False, None
        if json_depth(body) > MAX_JSON_DEPTH:
            self._error(400, f"the request body is nested deeper than {MAX_JSON_DEPTH} levels")
            return False, None
        return True, body

    # ---------------------------------------------------------------- dispatch
    def _dispatch(self, method: str) -> None:
        try:
            self._dispatch_checked(method)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            logger.exception("unhandled error in request handler")
            try:
                self._error(500, "internal error")
            except OSError:
                self.close_connection = True

    def _dispatch_checked(self, method: str) -> None:
        target = self.path
        if not target.startswith("/"):
            self._error(400, "request target must be an absolute path")
            return
        raw_path, _, raw_query = target.partition("?")
        raw_path = raw_path.partition("#")[0]
        problem = self._check_host_origin()
        if problem is not None:
            self._error(403, problem)
            return
        try:
            segments = _segments(raw_path)
        except UnicodeDecodeError:
            segments = None
        is_api = raw_path == "/api" or raw_path.startswith("/api/")
        if is_api:
            if self._cross_site():  # the page itself may be opened from another site; the API not
                self._error(403, "cross-site request")
                return
            if not self.server.token_ok(self.headers.get("X-Session-Token")):
                self._error(401, "missing or wrong X-Session-Token")
                return
            if segments is None:
                self._error(404, "unknown endpoint")
                return
            body: Any = None
            if method == "POST":
                ok, body = self._read_json_body()
                if not ok:
                    return
            try:
                query = parse_qs(raw_query, keep_blank_values=True, max_num_fields=50, errors="strict")
            except (ValueError, UnicodeDecodeError):
                self._error(400, "invalid query string")
                return
            status, payload = self.server.api.handle(method, segments[1:], query, body)
            self._send_json(status, payload)
            return
        if method != "GET":
            self._error(405, "static files are GET only")
            return
        if segments is None:
            self._error(403, "path not allowed")
            return
        self._static(segments)

    def _static(self, segments: list[str]) -> None:
        rel = segments or ["index.html"]
        root = self.server.static_dir
        candidate = root.joinpath(*rel)
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            self._error(404, "not found")
            return
        if not resolved.is_relative_to(root):
            self._error(403, "path not allowed")
            return
        ctype = _STATIC_TYPES.get(resolved.suffix.lower())
        if not resolved.is_file() or ctype is None:
            self._error(404, "not found")
            return
        data = resolved.read_bytes()
        self._headers(200, ctype, len(data))
        self.wfile.write(data)


# ============================================================================== application
@dataclass
class UIApp:
    """A running UI: worker, workspace, API and (not yet serving) HTTP server."""

    worker: ModelWorker
    workspace: Workspace
    api: Api
    server: UIServer
    _thread: threading.Thread | None = None

    def serve_in_thread(self) -> threading.Thread:
        """Start ``serve_forever`` on a daemon thread (tests, notebooks)."""
        thread = threading.Thread(target=self.server.serve_forever, name="research-rl-ui-http", daemon=True)
        thread.start()
        self._thread = thread
        return thread

    def close(self) -> None:
        """Stop serving, close the socket and stop the worker."""
        if self._thread is not None:
            self.server.shutdown()
            self._thread.join(10)
        self.server.server_close()
        self.worker.stop()


def create_app(
    loaded: LoadedModel,
    *,
    store: str | Path,
    options: ServerOptions,
    config: WorkspaceConfig | None = None,
    runs_dir: str | Path | None = None,
    score_fn: Any = None,
    sync_wait: float = 60.0,
) -> UIApp:
    """Build worker, workspace (on the worker), API and server. Nothing is served yet."""
    check_bind_host(options.host, options.allow_remote)  # before any work
    worker = ModelWorker(error_mapper=error_payload)
    try:
        workspace = call_on_worker(
            worker, lambda: Workspace(loaded, store, worker, config, score_fn=score_fn)
        )
        api = Api(
            workspace, worker, runs_dir=None if runs_dir is None else Path(runs_dir), sync_wait=sync_wait
        )
        server = UIServer(options, api)
    except BaseException:
        worker.stop()
        raise
    return UIApp(worker=worker, workspace=workspace, api=api, server=server)


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface of ``python -m research_rl.ui``."""
    parser = argparse.ArgumentParser(
        prog="python -m research_rl.ui",
        description=(
            "Local channel search + edit tool (research_rl.capability). Binds 127.0.0.1; every API "
            "request needs the session token printed at start. Numbers come from the model loaded on "
            "this machine."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model-dir", type=Path, help="local HF model dir (config.json + *.safetensors + tokenizer)"
    )
    source.add_argument(
        "--tiny-demo", action="store_true", help="build a random tiny Llama in-process (demo/tests)"
    )
    parser.add_argument(
        "--store", type=Path, help="VersionStore directory (default with --tiny-demo: a new temp dir)"
    )
    parser.add_argument("--runs-dir", type=Path, help="directory with GatedLoop decision logs (<name>.jsonl)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allow-remote", action="store_true", help="allow a non-loopback bind (prints a warning)"
    )
    parser.add_argument("--max-body", type=int, default=DEFAULT_MAX_BODY)
    parser.add_argument("--seed", type=int, default=0, help="operator names / splits seed")
    parser.add_argument("--operators", type=int, default=10)
    parser.add_argument("--sampler-seed", type=int, default=1)
    parser.add_argument("--model-seed", type=int, default=0, help="tiny-demo model seed")
    parser.add_argument("--items-transfer", type=int, default=128, help="gate items per transfer slice")
    parser.add_argument("--items-retention", type=int, default=512, help="gate items per retention slice")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--alpha-schedule", choices=("spending", "per_decision"), default="spending")
    parser.add_argument("--retention-margin", type=float, default=0.02)
    parser.add_argument("--reward-verifier", choices=("exact", "suffix_leak"), default="exact")
    parser.add_argument("--reward-items", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--verbose", action="store_true", help="log requests (never the token)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``python -m research_rl.ui``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    try:
        _, loopback = check_bind_host(args.host, args.allow_remote)
        token = resolve_token()
    except (BindError, ValueError) as exc:
        parser.error(str(exc))
    if not loopback:
        print(
            f"WARNING: binding to the non-loopback address {args.host} (--allow-remote). Anyone who can "
            "reach this port and knows the token can read and change the model's weights. Traffic is "
            "plain HTTP (not encrypted).",
            file=sys.stderr,
            flush=True,
        )
    if args.model_dir is not None and args.store is None:
        parser.error("--model-dir needs --store DIR")
    store = args.store or Path(tempfile.mkdtemp(prefix="research_rl_ui_store_"))
    try:
        loaded = build_tiny_demo(args.model_seed) if args.tiny_demo else load_model_dir(args.model_dir)
    except ModelLoadError as exc:
        parser.error(str(exc))
    config = WorkspaceConfig(
        seed=args.seed,
        num_operators=args.operators,
        sampler_seed=args.sampler_seed,
        batch_size=args.batch_size,
        items_per_transfer_slice=args.items_transfer,
        items_per_retention_slice=args.items_retention,
        alpha=args.alpha,
        alpha_schedule=args.alpha_schedule,
        retention_margin=args.retention_margin,
        reward_verifier=args.reward_verifier,
        reward_items=args.reward_items,
    )
    options = ServerOptions(
        host=args.host, port=args.port, token=token, allow_remote=args.allow_remote, max_body=args.max_body
    )
    try:
        app = create_app(loaded, store=store, options=options, config=config, runs_dir=args.runs_dir)
    except (ValueError, OSError) as exc:
        # StoreError (a ValueError): the store holds another model's weights or is damaged; ValueError:
        # its alpha ledger is pinned to other settings, or a bad option; OSError: e.g. the port is in use
        logger.debug("startup refused", exc_info=True)
        parser.error(f"cannot start: {type(exc).__name__}: {exc}")
    print(f"research_rl.ui: model = {loaded.source}; store = {store}", flush=True)
    if loaded.tiny_demo:
        print("research_rl.ui: tiny demo model - numbers say nothing about real large models", flush=True)
    print(f"research_rl.ui: open {app.server.token_url()}", flush=True)
    try:
        app.server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.server.server_close()
        app.worker.stop()
    return 0
