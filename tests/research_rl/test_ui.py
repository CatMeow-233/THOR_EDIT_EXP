"""Tests for research_rl.ui: the local channel search + edit server.

Every test starts a real server on 127.0.0.1, port 0, in a thread, with the in-process tiny demo
model (``research_rl.models.tiny``; nothing is downloaded) and talks to it with ``http.client``.
The random tiny model solves nothing, so its gate verdicts do not pass; the passing path uses a
cheap weight-sensitive stub score function (as in ``test_capability_edits.py``). These are CPU
tests of the tool's plumbing and security checks, not model-quality measurements.
"""

from __future__ import annotations

import http.client
import json
import re
import socket
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from research_rl.capability.tasks import AuditAccessError
from research_rl.capability.types import (
    DecisionRecord,
    GateDecision,
    GateVerdict,
    Item,
    SliceTest,
    Split,
)
from research_rl.models.model_interface import snapshot
from research_rl.models.tiny import save_tiny_model
from research_rl.ui import server as server_mod
from research_rl.ui.jobs import ModelWorker
from research_rl.ui.loading import ModelLoadError, build_tiny_demo, check_model_dir, load_model_dir
from research_rl.ui.server import (
    TOKEN_ENV,
    BindError,
    ServerOptions,
    UIApp,
    check_bind_host,
    create_app,
    resolve_token,
)
from research_rl.ui.workspace import WorkspaceConfig, call_on_worker

TOKEN = "test-token-" + "x" * 24
SMALL = WorkspaceConfig(
    items_per_transfer_slice=4,
    items_per_retention_slice=16,
    reward_items=8,
    alpha_schedule="per_decision",
    batch_size=64,
)


# ============================================================================== helpers
class Client:
    """Minimal HTTP client that controls every header (Host included)."""

    def __init__(self, app: UIApp) -> None:
        self.app = app
        self.host = app.server.bound_host
        self.port = app.server.bound_port
        self.authority = app.server.authority

    def raw(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        token: str | None = TOKEN,
        host: str | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=120)
        try:
            conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            if host != "":
                conn.putheader("Host", host or self.authority)
            if token is not None:
                conn.putheader("X-Session-Token", token)
            for k, v in (headers or {}).items():
                conn.putheader(k, v)
            if body is not None:
                conn.putheader("Content-Length", str(len(body)))
            conn.endheaders(body)
            res = conn.getresponse()
            data = res.read()
            return res.status, {k.lower(): v for k, v in res.getheaders()}, data
        finally:
            conn.close()

    def json(self, method: str, path: str, body: Any = None, **kw: Any) -> tuple[int, Any]:
        headers = dict(kw.pop("headers", {}) or {})
        raw = None
        if body is not None:
            raw = json.dumps(body).encode()
            headers.setdefault("Content-Type", "application/json")
        status, _, data = self.raw(method, path, body=raw, headers=headers, **kw)
        return status, json.loads(data) if data else None

    def wait(self, job: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            status, data = self.json("GET", f"/api/jobs/{job['id']}")
            assert status == 200, data
            if data["job"]["status"] in ("done", "failed"):
                return data["job"]
            time.sleep(0.02)
        raise AssertionError(f"job {job['id']} did not finish")

    def result(self, method: str, path: str, body: Any = None) -> Any:
        """Result of an endpoint that answers 200 {result} or 202 {job} (polled)."""
        status, data = self.json(method, path, body)
        if status == 202:
            job = self.wait(data["job"])
            assert job["status"] == "done", job
            return job["result"]
        assert status == 200, data
        return data.get("result", data)


def _make_app(tmp_path: Path, *, config: WorkspaceConfig = SMALL, **kw: Any) -> UIApp:
    options = kw.pop("options", None) or ServerOptions(
        port=0, token=TOKEN, max_body=kw.pop("max_body", 1024 * 1024)
    )
    store = kw.pop("store_dir", None) or tmp_path / "store"
    app = create_app(build_tiny_demo(0), store=store, options=options, config=config, **kw)
    app.serve_in_thread()
    return app


@pytest.fixture
def app(tmp_path: Path) -> Iterator[UIApp]:
    runs = tmp_path / "runs"
    runs.mkdir()
    a = _make_app(tmp_path, runs_dir=runs)
    try:
        yield a
    finally:
        a.close()


@pytest.fixture
def client(app: UIApp) -> Client:
    return Client(app)


def _params(app: UIApp) -> dict[str, torch.Tensor]:
    return call_on_worker(app.worker, lambda: snapshot(app.workspace.model))


def _assert_bitwise(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert set(a) == set(b)
    for name in a:
        assert a[name].dtype == b[name].dtype
        assert torch.equal(a[name], b[name]), name


# ============================================================================== authentication / Host / Origin
def test_token_missing_or_wrong_is_401(client: Client) -> None:
    assert client.json("GET", "/api/status", token=None)[0] == 401
    assert client.json("GET", "/api/status", token="wrong-token-" + "y" * 20)[0] == 401
    assert client.json("GET", "/api/status", token=TOKEN + "z")[0] == 401
    assert client.json("GET", "/api/status", token="")[0] == 401
    # a non-ASCII header value must not crash the constant-time compare
    status, _, _ = client.raw("GET", "/api/status", token=None, headers={"X-Session-Token": "tést"})
    assert status == 401
    assert client.json("POST", "/api/stats", {"split": "train"}, token=None)[0] == 401
    status, data = client.json("GET", "/api/status")
    assert status == 200
    assert data["tiny_demo"] is True and "this machine" in data["honesty"]


def test_bad_host_is_403(client: Client) -> None:
    port = client.port
    for host in (f"evil.example:{port}", "127.0.0.1:1", "127.0.0.1", f"127.0.0.1.evil:{port}"):
        assert client.raw("GET", "/api/status", host=host)[0] == 403, host
        assert client.raw("GET", "/", host=host)[0] == 403, host  # static files too
    assert client.raw("GET", "/api/status", host="")[0] == 403  # no Host header at all
    assert client.raw("GET", "/api/status", host=f"localhost:{port}")[0] == 200
    assert client.raw("GET", "/api/status")[0] == 200


def test_foreign_origin_is_403(client: Client) -> None:
    for origin in (
        "http://evil.example",
        "null",
        f"https://{client.authority}",
        f"http://127.0.0.1:{client.port + 1}",
    ):
        status, _, _ = client.raw("GET", "/api/status", headers={"Origin": origin})
        assert status == 403, origin
    assert client.raw("GET", "/api/status", headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
    # the page itself may be opened from a link on another site (it holds no secret)
    assert client.raw("GET", "/", token=None, headers={"Sec-Fetch-Site": "cross-site"})[0] == 200
    assert client.raw("GET", "/api/status", headers={"Origin": f"http://{client.authority}"})[0] == 200
    # Host is checked before the token: a bad Host with a missing token is 403, not 401
    assert client.raw("GET", "/api/status", token=None, host="evil:1")[0] == 403


def test_security_headers_on_every_response(client: Client) -> None:
    for path, token in (
        ("/api/status", TOKEN),
        ("/", None),
        ("/app.js", None),
        ("/api/status", None),
        ("/nope.js", None),
    ):
        _, headers, _ = client.raw("GET", path, token=token)
        csp = headers["content-security-policy"]
        assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["x-frame-options"] == "DENY"
        assert headers["cache-control"] == "no-store"
    status, headers, body = client.raw("GET", "/")
    assert status == 200 and headers["content-type"].startswith("text/html")
    assert b"app.js" in body and b"<script>" not in body  # no inline script (CSP)
    status, headers, _ = client.raw("GET", "/app.js")
    assert status == 200 and headers["content-type"].startswith("text/javascript")
    # an error from the base class is JSON with the same headers
    status, headers, body = client.raw("PUT", "/api/status")
    assert status == 405 and "frame-ancestors 'none'" in headers["content-security-policy"]
    assert json.loads(body)["error"]["status"] == 405


# ============================================================================== bodies
def test_oversize_body_is_413(tmp_path: Path) -> None:
    app = _make_app(tmp_path, max_body=1024)
    try:
        c = Client(app)
        conn = http.client.HTTPConnection(c.host, c.port, timeout=30)
        conn.putrequest("POST", "/api/stage", skip_host=True)
        conn.putheader("Host", c.authority)
        conn.putheader("X-Session-Token", TOKEN)
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(10 * 1024 * 1024))
        conn.endheaders()  # the body is never sent: the server decides from Content-Length
        res = conn.getresponse()
        assert res.status == 413
        assert json.loads(res.read())["error"]["status"] == 413
        conn.close()
        # just below the limit is read (and rejected for its content, not its size)
        small = json.dumps(
            {"op": "ablate", "selection": {"0": [0]}, "params": {}, "note": "x" * 900}
        ).encode()
        assert len(small) <= 1024
        status, _, _ = c.raw("POST", "/api/stage", body=small, headers={"Content-Type": "application/json"})
        assert status == 200
        assert c.json("POST", "/api/discard", {})[0] == 200
        assert (
            c.raw("POST", "/api/stage", body=b"x" * 1025, headers={"Content-Type": "application/json"})[0]
            == 413
        )
    finally:
        app.close()


# ============================================================================== connections
def _closed_by_server(sock: socket.socket, timeout: float) -> bool:
    """``True`` once the server closes ``sock`` (EOF or reset) within ``timeout`` seconds."""
    sock.settimeout(timeout)
    try:
        return sock.recv(1024) == b""
    except (ConnectionResetError, BrokenPipeError):
        return True
    except TimeoutError:
        return False


def test_connection_cap_closes_extra_connections(tmp_path: Path) -> None:
    app = _make_app(tmp_path, options=ServerOptions(port=0, token=TOKEN, max_connections=3))
    held: list[socket.socket] = []
    try:
        c = Client(app)
        for _ in range(10):  # more sequential requests than the cap: every slot is given back
            assert c.raw("GET", "/", token=None)[0] == 200
        for _ in range(3):
            s = socket.create_connection((c.host, c.port), timeout=10)
            s.sendall(b"GET / HTTP/1.1\r\n")  # an unfinished request holds its slot
            held.append(s)
        deadline = time.time() + 10
        while app.server._slots._value != 0 and time.time() < deadline:  # type: ignore[attr-defined]
            time.sleep(0.01)
        extra = socket.create_connection((c.host, c.port), timeout=10)
        try:
            assert _closed_by_server(extra, 10.0)  # closed on accept, nothing read or answered
        finally:
            extra.close()
        assert app.server.connections_refused == 1
        for s in held:
            s.close()
        held.clear()
        deadline = time.time() + 10
        while time.time() < deadline:  # the held slots come back once their handlers see EOF
            if app.server._slots._value == 3:  # type: ignore[attr-defined]
                break
            time.sleep(0.01)
        assert c.raw("GET", "/", token=None)[0] == 200
    finally:
        for s in held:
            s.close()
        app.close()


def test_header_trickle_is_dropped_and_never_dispatched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatched: list[str] = []
    real = server_mod.UIRequestHandler._dispatch_checked

    def record(self: server_mod.UIRequestHandler, method: str) -> None:
        dispatched.append(self.path)
        real(self, method)

    monkeypatch.setattr(server_mod.UIRequestHandler, "_dispatch_checked", record)
    app = _make_app(
        tmp_path, options=ServerOptions(port=0, token=TOKEN, max_connections=2, header_timeout=1.0)
    )
    try:
        c = Client(app)
        s = socket.create_connection((c.host, c.port), timeout=10)
        try:
            head = f"POST /api/discard HTTP/1.1\r\nHost: {c.authority}\r\nX-Session-Token: {TOKEN}\r\n"
            s.sendall(head.encode())
            start = time.monotonic()
            closed = False
            while (
                time.monotonic() - start < 8.0
            ):  # one byte every 0.2 s, far inside the 30 s per-read timeout
                try:
                    s.sendall(b"X")
                except (BrokenPipeError, ConnectionResetError):
                    closed = True
                    break
                if _closed_by_server(s, 0.2):
                    closed = True
                    break
            elapsed = time.monotonic() - start
        finally:
            s.close()
        assert closed and elapsed < 4.0, elapsed  # dropped near header_timeout, not after 30 s per read
        assert app.server.header_timeouts == 1
        assert dispatched == []  # the truncated request was never handled
        assert c.raw("GET", "/", token=None)[0] == 200  # the slot was given back
        assert dispatched == ["/"]
    finally:
        app.close()


def test_server_rejects_bad_connection_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_connections"):
        _make_app(tmp_path, options=ServerOptions(port=0, token=TOKEN, max_connections=0))
    with pytest.raises(ValueError, match="header_timeout"):
        _make_app(tmp_path, options=ServerOptions(port=0, token=TOKEN, header_timeout=float("nan")))


def test_body_validation(client: Client) -> None:
    ct = {"Content-Type": "application/json"}
    # no Content-Length
    assert client.raw("POST", "/api/discard", headers=ct)[0] == 411
    assert client.raw("POST", "/api/discard", body=b"{}", headers={"Content-Type": "text/plain"})[0] == 415
    assert client.raw("POST", "/api/discard", body=b"{not json", headers=ct)[0] == 400
    assert client.raw("POST", "/api/discard", body=b"[1, 2]", headers=ct)[0] == 400
    assert client.raw("POST", "/api/stats", body=b'{"threshold": NaN}', headers=ct)[0] == 400
    assert client.raw("POST", "/api/stats", body=b"\xff\xfe", headers=ct)[0] == 400
    status, data = client.json("POST", "/api/stats", {"split": "train", "bogus": 1})
    assert status == 400 and "unknown fields" in data["error"]["message"]
    status, data = client.json("POST", "/api/stats", {"split": "train", "n": 10**9})
    assert status == 400


def test_malformed_lengths_nesting_and_huge_numbers_are_4xx(client: Client) -> None:
    ct = {"Content-Type": "application/json"}
    # a Content-Length over int()'s 4300-digit limit is over the cap: 413 before int() runs
    assert client.raw("POST", "/api/discard", headers={**ct, "Content-Length": "9" * 5000})[0] == 413
    # leading zeros do not count as size (the 2-byte body is read: nothing staged -> 409)
    conn = http.client.HTTPConnection(client.host, client.port, timeout=30)
    try:
        conn.putrequest("POST", "/api/discard", skip_host=True, skip_accept_encoding=True)
        for k, v in {**ct, "Host": client.authority, "X-Session-Token": TOKEN}.items():
            conn.putheader(k, v)
        conn.putheader("Content-Length", "0" * 5000 + "2")
        conn.endheaders(b"{}")
        res = conn.getresponse()
        assert res.status == 409, res.read()
    finally:
        conn.close()
    # a non-ASCII digit (str.isdigit accepts it, int() does not) is malformed
    assert client.raw("POST", "/api/discard", headers={**ct, "Content-Length": "\u00b2"})[0] == 400
    # nesting too deep for the JSON parser (RecursionError) and nesting deeper than the cap
    deep = b'{"a": ' + b"[" * 100_000 + b"]" * 100_000 + b"}"
    assert client.raw("POST", "/api/discard", body=deep, headers=ct)[0] == 400
    nested = b'{"a": ' + b"[" * 40 + b"]" * 40 + b"}"
    status, _, data = client.raw("POST", "/api/discard", body=nested, headers=ct)
    assert status == 400 and b"nested deeper" in data
    # an integer too large for a float, in float parameters (body and query)
    big = "9" * 400
    for body in (
        '{"split": "train", "threshold": ' + big + "}",
        '{"split": "train", "threshold": -' + big + "}",
    ):
        status, _, data = client.raw("POST", "/api/stats", body=body.encode(), headers=ct)
        assert status == 400 and b"threshold must be in" in data, data
    status, data = client.json("POST", "/api/search", {"sigma": int(big)})
    assert status == 400 and "sigma must be in" in data["error"]["message"]
    status, _, data = client.raw(
        "POST",
        "/api/stage",
        body=('{"op": "scale", "selection": {"0": [0]}, "params": {"factor": ' + big + "}}").encode(),
        headers=ct,
    )
    assert status == 400, data
    # every case above is a 4xx, so the server still answers normally
    assert client.json("GET", "/api/status")[0] == 200


# ============================================================================== AUDIT
def test_audit_is_403_and_never_queued(client: Client, app: UIApp) -> None:
    before = len(app.worker.list())
    for body in (
        {"split": "audit"},
        {"split": "AUDIT"},
        {"split": " Audit "},
        {"split": "train", "slice": "audit/pairs"},
        {"split": "t_rule", "slice": "AUDIT/depth3"},
    ):
        status, data = client.json("POST", "/api/stats", body)
        assert status == 403, body
        assert "AUDIT" in data["error"]["message"]
    for q in ("score=wanda&split=audit", "score=contrast&split=AUDIT", "score=usage_ratio&split=audit"):
        assert client.json("GET", f"/api/channels?{q}")[0] == 403, q
    assert client.json("POST", "/api/search", {"score": "wanda", "split": "audit"})[0] == 403
    assert len(app.worker.list()) == before  # nothing reached the model worker
    ws = app.workspace
    with pytest.raises(AuditAccessError):
        call_on_worker(app.worker, lambda: ws.collect_stats(Split.AUDIT, 8))
    assert ws.sampler.allow_audit is False
    assert all(not s.startswith("audit/") for s in ws.public_model()["slices"])


# ============================================================================== static files
def test_static_path_traversal(client: Client, tmp_path: Path) -> None:
    for path in (
        "/../server.py",
        "/%2e%2e/server.py",
        "/%2E%2E/%2e%2e/etc/passwd",
        "/..%2fserver.py",
        "/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "//etc/passwd",
        "/etc/passwd",
        "/./index.html",
        "/..%5cserver.py",
        "/index.html%00.js",
        "/static/../../server.py",
        "/%ff",
        "/server.py",
    ):
        status, _, body = client.raw("GET", path, token=None)
        assert status in (403, 404), (path, status)
        assert b"ThreadingHTTPServer" not in body and b"root:" not in body
    assert client.raw("GET", "/", token=None)[0] == 200
    assert client.raw("GET", "/style.css", token=None)[0] == 200
    assert (
        client.raw("POST", "/index.html", body=b"{}", headers={"Content-Type": "application/json"})[0] == 405
    )


def test_static_symlink_escape_and_no_listing(tmp_path: Path) -> None:
    static = tmp_path / "static"
    (static / "sub").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>t</title>")
    secret = tmp_path / "secret.js"
    secret.write_text("SECRET")
    (static / "link.js").symlink_to(secret)
    options = ServerOptions(port=0, token=TOKEN, static_dir=static)
    app = _make_app(tmp_path / "a", options=options)
    try:
        c = Client(app)
        status, _, body = c.raw("GET", "/link.js", token=None)
        assert status == 403 and b"SECRET" not in body
        assert c.raw("GET", "/sub", token=None)[0] == 404  # a directory is never listed
        assert c.raw("GET", "/sub/", token=None)[0] == 404
        assert c.raw("GET", "/", token=None)[0] == 200
    finally:
        app.close()


# ============================================================================== endpoints
def test_status_model_and_routing(client: Client) -> None:
    status, model = client.json("GET", "/api/model")
    assert status == 200
    assert [lay["intermediate_size"] for lay in model["layers"]] == [64, 64]
    assert model["num_layers"] == 2 and model["tiny_demo"] is True
    assert model["tiny_note"] and "say nothing" in model["tiny_note"]
    assert client.json("GET", "/api/nope")[0] == 404
    assert client.json("GET", "/api/stage")[0] == 405
    assert client.json("POST", "/api/status", {})[0] == 405
    assert client.json("GET", "/api/jobs/doesnotexist")[0] == 404
    assert client.json("GET", "/api/search")[0] == 404  # no search yet


def test_stats_channels_and_inspect(client: Client) -> None:
    status, data = client.json(
        "POST", "/api/stats", {"split": "train", "n": 24, "contrast": True, "threshold": 0.01}
    )
    assert status == 202
    job = client.wait(data["job"])
    assert job["status"] == "done", job
    assert job["result"]["stats"]["split"] == "train"
    client.result("POST", "/api/stats", {"split": "retention", "n": 24, "threshold": 0.01})
    scores = client.json("GET", "/api/scores")[1]["scores"]
    assert scores["wanda:train"]["available"] and scores["usage_ratio"]["available"]
    assert scores["random"]["available"]
    # the random tiny model solves nothing: the contrast is unavailable, never silently zeros
    assert scores["contrast:train"]["available"] is False
    assert "correct" in scores["contrast:train"]["reason"]
    assert client.json("GET", "/api/channels?score=contrast&split=train")[0] == 409

    status, ranked = client.json("GET", "/api/channels?score=wanda&split=train&top=5")
    assert status == 200 and len(ranked["channels"]) == 5 and ranked["total"] == 128
    vals = [c["score"] for c in ranked["channels"]]
    assert vals == sorted(vals, reverse=True)
    status, by_layer = client.json("GET", "/api/channels?score=usage_ratio&layer=1&top=2000")
    assert status == 200 and {c["layer"] for c in by_layer["channels"]} == {1} and by_layer["total"] == 64
    status, by_group = client.json("GET", "/api/channels?score=random&group=L0/b0&bands=4&top=2000")
    assert (
        status == 200 and by_group["total"] == 16 and {c["group"] for c in by_group["channels"]} == {"L0/b0"}
    )
    assert client.json("GET", "/api/channels?score=random&group=L9/b0")[0] == 404
    assert client.json("GET", "/api/channels?score=fisher")[0] == 400
    assert client.json("GET", "/api/channels?score=random&top=0")[0] == 400

    detail = client.result("GET", "/api/channel/1/5?k=3")
    assert detail["layer"] == 1 and detail["channel"] == 5
    assert len(detail["projection"]["top"]) == 3 and len(detail["projection"]["bottom"]) == 3
    tops = [t["logit"] for t in detail["projection"]["top"]]
    assert tops == sorted(tops, reverse=True)
    assert {s["split"] for s in detail["activation"]["splits"]} == {"train", "retention"}
    hist = detail["activation"]["splits"][0]["layer_mean_abs_histogram"]
    assert sum(hist["counts"]) == 64 and 0 <= hist["channel_bin"] < 20
    assert detail["versions"][-1]["version"] == "live" and len(detail["versions"]) == 2
    assert "wanda:train" in detail["scores"]
    assert client.json("GET", "/api/channel/1/64")[0] == 404
    assert client.json("GET", "/api/channel/7/0")[0] == 404
    assert client.json("GET", "/api/channel/x/0")[0] == 400


def test_search_job(client: Client, app: UIApp) -> None:
    client.result("POST", "/api/stats", {"split": "train", "n": 16})
    before = _params(app)
    body = {
        "score": "wanda",
        "split": "train",
        "bands": 2,
        "layers": [0],
        "items_per_transfer_slice": 2,
        "items_per_retention_slice": 4,
        "fitness_items": 2,
        "population": 2,
        "controls": ["random"],
    }
    res = client.result("POST", "/api/search", body)
    report = res["report"]
    assert report["winner"] in ("L0/b0", "L0/b1")
    assert set(report["controls"]) == {"random"}
    _assert_bitwise(before, _params(app))  # the search restores the start weights
    assert client.json("GET", "/api/search")[0] == 200
    # bounds are enforced before anything runs
    for bad in (
        {"rounds": 99},
        {"population": 3},
        {"controls": ["bogus"]},
        {"groups": ["x"] * 40},
        {"sigma": 0},
    ):
        assert client.json("POST", "/api/search", {**body, **bad})[0] == 400, bad
    _, data = client.json("POST", "/api/search", {**body, "groups": ["L0/b7"]})
    job = client.wait(data["job"])
    assert job["status"] == "failed" and job["error"]["status"] == 404


def test_searches_draw_fresh_items(client: Client, app: UIApp) -> None:
    # Regression: every UI search used SubsetSearch's default index=0, so all searches of one
    # process ranked candidates on the same round / final item sets (only the ES fitness stream
    # was keyed per search). Search k now uses index=k: same purposes, different items.
    client.result("POST", "/api/stats", {"split": "train", "n": 16})
    body = {
        "score": "wanda",
        "split": "train",
        "bands": 2,
        "layers": [0],
        "items_per_transfer_slice": 2,
        "items_per_retention_slice": 4,
        "fitness_items": 2,
        "population": 2,
        "controls": ["random"],
    }
    first = client.result("POST", "/api/search", body)
    second = client.result("POST", "/api/search", {**body, "seed": 7})
    assert second["search_index"] == first["search_index"] + 1
    for res in (first, second):
        assert res["report"]["config"]["index"] == res["search_index"]
        assert {e["index"] for e in res["report"]["evaluations"]} == {res["search_index"]}

    def digests(res: dict[str, object]) -> dict[str, str]:
        report = res["report"]
        assert isinstance(report, dict)
        return {e["purpose"]: e["uid_digest"] for e in report["evaluations"]}

    d1, d2 = digests(first), digests(second)
    common = set(d1) & set(d2)
    assert common  # the same round / final purposes ...
    assert all(d1[p] != d2[p] for p in common)  # ... on different items


def test_stage_evaluate_commit_requires_pass_or_override(client: Client, app: UIApp) -> None:
    root = client.json("GET", "/api/history")[1]["head"]
    staged = client.result("POST", "/api/stage", {"op": "ablate", "selection": {"0": [3, 4]}, "params": {}})
    assert staged["parent"] == root
    assert (
        client.json("POST", "/api/stage", {"op": "ablate", "selection": {"0": [5]}, "params": {}})[0] == 409
    )
    # never evaluated -> refused
    status, data = client.json("POST", "/api/commit", {})
    assert status == 409 and data["error"]["type"] == "CommitRefusedError"
    res = client.result("POST", "/api/evaluate", {})
    verdict = res["verdict"]
    assert res["passed"] is False and not (verdict["transfer_ok"] and verdict["retention_ok"])
    assert "transfer" in verdict and verdict["retention"] and all("bound" in t for t in verdict["retention"])
    assert res["reward"]["verifier_v1"] == "exact" and "change_v1" in res["reward"]
    assert "d" in res["divergence"] and "this machine" in res["honesty"]
    # not passed, no override -> refused; a blank override counts as none
    for body in ({}, {"override": ""}, {"override": "   "}):
        status, data = client.json("POST", "/api/commit", body)
        assert status == 409, body
    assert client.json("GET", "/api/status")[1]["staged"] is True
    status, data = client.json("POST", "/api/commit", {"override": "demo: keep the ablation", "note": "n1"})
    assert status == 200, data
    version = data["result"]["version"]
    assert version["override"] == "demo: keep the ablation" and version["parent"] == root
    assert version["passed"] is False and version["verdict"] is not None
    history = client.json("GET", "/api/history")[1]
    assert history["head"] == version["id"] and history["lineage"] == [root, version["id"]]
    assert history["staged"] is None
    # declarative only: code-like or unknown fields are rejected before any write
    assert client.json("POST", "/api/stage", {"op": "exec", "selection": {"0": [1]}, "params": {}})[0] == 400
    assert (
        client.json(
            "POST",
            "/api/stage",
            {"op": "scale", "selection": {"0": [1]}, "params": {"factor": 2, "code": "x"}},
        )[0]
        == 400
    )
    assert (
        client.json("POST", "/api/stage", {"op": "ablate", "selection": {"0": [99]}, "params": {}})[0] == 400
    )
    assert client.json("POST", "/api/evaluate", {})[0] == 202  # job fails: nothing staged
    status, data = client.json("POST", "/api/discard", {})
    assert status == 409


def test_evaluate_reward_failure_still_publishes_the_recorded_verdict(client: Client, app: UIApp) -> None:
    client.result("POST", "/api/stage", {"op": "ablate", "selection": {"0": [2]}, "params": {}})
    ws = app.workspace

    def oom(*_a: Any, **_k: Any) -> Any:
        raise MemoryError("simulated out-of-memory while generating reward completions")

    before = _params(app)
    original = ws.v1_fn.generate
    ws.v1_fn.generate = oom  # type: ignore[method-assign]
    try:
        res = client.result("POST", "/api/evaluate", {})
    finally:
        ws.v1_fn.generate = original  # type: ignore[method-assign]
    # the verdict the store recorded (alpha spent) is returned and published, reward marked missing
    assert res["verdict"] is not None and res["reward"] is None and res["divergence"] is None
    assert "MemoryError" in res["reward_error"] and "verdict" in res["reward_error"]
    store_count, store_passed = call_on_worker(
        app.worker,
        lambda: (ws.store.evaluation_count, ws.store.staged.passed),  # type: ignore[union-attr]
    )
    assert store_count == 1
    status = client.json("GET", "/api/status")[1]
    assert status["evaluation_count"] == store_count
    assert status["last_evaluation"]["verdict"] == res["verdict"]
    staged = client.json("GET", "/api/history")[1]["staged"]
    assert staged["evaluations"] == 1 and staged["passed"] is store_passed
    assert staged["last_verdict"] == res["verdict"]
    _assert_bitwise(before, _params(app))  # the failed reward step left the weights as they were
    # the commit rule follows the verdict that is shown
    assert client.json("POST", "/api/commit", {})[0] == (200 if store_passed else 409)


def test_restore_is_bitwise(client: Client, app: UIApp) -> None:
    root_params = _params(app)
    root = client.json("GET", "/api/history")[1]["head"]
    client.result(
        "POST",
        "/api/stage",
        {"op": "scale", "selection": {"1": [0, 7, 9]}, "params": {"factor": 0.5, "up": True}},
    )
    staged_params = _params(app)
    assert not torch.equal(
        staged_params["model.layers.1.mlp.down_proj.weight"],
        root_params["model.layers.1.mlp.down_proj.weight"],
    )
    # discard puts the parent back bitwise
    client.result("POST", "/api/discard", {})
    _assert_bitwise(root_params, _params(app))
    client.result(
        "POST",
        "/api/stage",
        {"op": "scale", "selection": {"1": [0, 7, 9]}, "params": {"factor": 0.5, "up": True}},
    )
    child = client.result("POST", "/api/commit", {"override": "test restore"})["version"]["id"]
    child_params = _params(app)
    _assert_bitwise(staged_params, child_params)
    client.result("POST", "/api/restore", {"version": root})
    _assert_bitwise(root_params, _params(app))
    assert client.json("GET", "/api/status")[1]["head"] == root
    client.result("POST", "/api/restore", {"version": child})
    _assert_bitwise(child_params, _params(app))
    assert client.json("POST", "/api/restore", {"version": "nothex"})[0] == 400
    assert client.json("POST", "/api/restore", {"version": "a" * 64})[0] == 404
    # an edit referencing a version (revert) is data, too
    client.result(
        "POST", "/api/stage", {"op": "revert", "selection": {"1": [0, 7, 9]}, "params": {"version": root}}
    )
    _assert_bitwise(root_params, _params(app))
    client.result("POST", "/api/discard", {})
    # the channel's change across versions is visible in Inspect
    rows = client.result("GET", "/api/channel/1/7")["versions"]
    assert [r["version"] for r in rows] == [root, child, "live"]
    assert rows[1]["change_from_parent"] > 0 and rows[1]["distance_to_live"] == 0.0


class SwitchScore:
    """Transfer items score 1 iff layer-0 channel-0's down column is zero; retention always 1."""

    def __call__(self, m: nn.Module, items: Sequence[Item]) -> list[float]:
        ablated = bool((m.model.layers[0].mlp.down_proj.weight[:, 0] == 0).all())  # type: ignore[union-attr, index]
        return [1.0 if it.split is Split.RETENTION or ablated else 0.0 for it in items]


def test_passing_gate_commits_without_override(tmp_path: Path) -> None:
    cfg = WorkspaceConfig(
        items_per_transfer_slice=20,
        items_per_retention_slice=150,
        retention_margin=0.1,
        alpha_schedule="per_decision",
        reward_items=4,
    )
    app = _make_app(tmp_path, config=cfg, score_fn=SwitchScore())
    try:
        c = Client(app)
        c.result("POST", "/api/stage", {"op": "ablate", "selection": {"0": [0]}, "params": {}})
        res = c.result("POST", "/api/evaluate", {})
        assert res["passed"] is True and res["verdict"]["transfer_ok"] and res["verdict"]["retention_ok"]
        assert res["verdict"]["transfer"]["bound"] > 0
        version = c.result("POST", "/api/commit", {})["version"]
        assert version["override"] is None and version["passed"] is True
    finally:
        app.close()


# ============================================================================== the worker
def test_worker_serialises_jobs_and_survives_failures() -> None:
    worker = ModelWorker()
    active = [0]
    peak = [0]
    lock = threading.Lock()
    order: list[int] = []
    threads: set[str] = set()

    def make(i: int) -> Any:
        def fn(ctx: Any) -> int:
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            threads.add(threading.current_thread().name)
            time.sleep(0.01)
            order.append(i)
            with lock:
                active[0] -= 1
            if i == 3:
                raise ValueError("boom")
            return i

        return fn

    try:
        jobs = [worker.submit("t", make(i)) for i in range(8)]
        for j in jobs:
            assert j.done_event.wait(10)
        assert peak[0] == 1 and order == list(range(8))
        assert threads == {worker.thread.name}
        assert jobs[3].status == "failed" and jobs[3].error is not None and "boom" in jobs[3].error["message"]
        assert [j.result for j in jobs if j.status == "done"] == [0, 1, 2, 4, 5, 6, 7]
    finally:
        worker.stop()


def test_http_requests_never_touch_weights_concurrently(client: Client, app: UIApp) -> None:
    ws = app.workspace
    model = ws.model
    active = [0]
    peak = [0]
    seen: set[str] = set()
    lock = threading.Lock()

    def pre(_module: nn.Module, _args: Any) -> None:
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            seen.add(threading.current_thread().name)

    def post(_module: nn.Module, _args: Any, _out: Any) -> None:
        with lock:
            active[0] -= 1

    original = ws.channel_detail

    def detail(*args: Any, **kwargs: Any) -> Any:
        pre(model, ())
        try:
            time.sleep(0.01)
            return original(*args, **kwargs)
        finally:
            post(model, (), None)

    h1 = model.register_forward_pre_hook(pre)
    h2 = model.register_forward_hook(post)
    ws.channel_detail = detail  # type: ignore[method-assign]
    try:
        errors: list[BaseException] = []

        def go(i: int) -> None:
            try:
                if i % 3 == 0:
                    client.result(
                        "POST",
                        "/api/stats",
                        {"split": "train" if i % 2 else "retention", "n": 12, "contrast": True},
                    )
                else:
                    client.result("GET", f"/api/channel/{i % 2}/{i}")
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=go, args=(i,)) for i in range(9)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
        assert not errors, errors
        assert peak[0] == 1
        assert seen == {app.worker.thread.name}
    finally:
        h1.remove()
        h2.remove()
        ws.channel_detail = original  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="worker"):
        ws.channel_detail(0, 0)  # from the test thread: refused


# ============================================================================== model loading
def test_non_safetensors_model_dir_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_pickle(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("torch.load must never be called")

    monkeypatch.setattr(torch, "load", no_pickle)
    bin_only = tmp_path / "bin_only"
    bin_only.mkdir()
    (bin_only / "config.json").write_text("{}")
    (bin_only / "pytorch_model.bin").write_bytes(b"\x80\x04not really a pickle")
    with pytest.raises(ModelLoadError, match="non-safetensors"):
        load_model_dir(bin_only)
    mixed = save_tiny_model(tmp_path / "mixed")
    (mixed / "extra.pt").write_bytes(b"\x80\x04")
    with pytest.raises(ModelLoadError, match="non-safetensors"):
        load_model_dir(mixed)
    nested = save_tiny_model(tmp_path / "nested")
    (nested / "sub").mkdir()
    (nested / "sub" / "optimizer.pkl").write_bytes(b"\x80\x04")
    with pytest.raises(ModelLoadError):
        check_model_dir(nested)
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "config.json").write_text("{}")
    with pytest.raises(ModelLoadError, match=r"no \*\.safetensors"):
        load_model_dir(empty)
    with pytest.raises(ModelLoadError, match="not a directory"):
        load_model_dir(tmp_path / "missing")
    no_config = tmp_path / "no_config"
    no_config.mkdir()
    with pytest.raises(ModelLoadError, match=r"config\.json"):
        load_model_dir(no_config)
    good = save_tiny_model(tmp_path / "good")
    loaded = load_model_dir(good)
    assert loaded.tiny_demo is False and sum(p.numel() for p in loaded.model.parameters()) > 0
    # the CLI refuses a non-safetensors dir before serving anything
    with pytest.raises(SystemExit):
        server_mod.main(["--model-dir", str(bin_only), "--store", str(tmp_path / "s"), "--port", "0"])


def _index_checkpoint(root: Path, shard_name: str) -> Path:
    """Tiny checkpoint whose ``model.safetensors.index.json`` maps every key to ``shard_name``, a
    ``torch.save`` pickle (written at ``root / 'm' / shard_name``, which may lie outside the dir),
    with a decoy ``decoy.safetensors`` next to it; ``model.safetensors`` is removed."""
    from safetensors.torch import load_file, save_file

    model_dir = save_tiny_model(root / "m")
    state = load_file(str(model_dir / "model.safetensors"))
    (model_dir / "model.safetensors").unlink()
    target = model_dir / shard_name
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, target)
    save_file({"dummy": torch.zeros(1)}, str(model_dir / "decoy.safetensors"))
    index = {"metadata": {}, "weight_map": {k: shard_name for k in state}}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return model_dir


@pytest.mark.parametrize(
    ("shard_name", "message"),
    [
        ("shard-00001", "non-safetensors shards"),  # no suffix: the suffix blocklist cannot see it
        ("../outside/weights", "not bare file names"),  # outside the directory: rglob never sees it
    ],
)
def test_index_naming_a_pickle_shard_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shard_name: str, message: str
) -> None:
    from transformers import AutoModelForCausalLM

    model_dir = _index_checkpoint(tmp_path, shard_name)
    calls: list[str] = []
    real_load = torch.load
    real_pretrained = AutoModelForCausalLM.from_pretrained

    def spy_load(*args: Any, **kwargs: Any) -> Any:
        calls.append(f"torch.load {args[0] if args else kwargs.get('f')}")
        return real_load(*args, **kwargs)

    def spy_pretrained(*args: Any, **kwargs: Any) -> Any:
        calls.append("from_pretrained")
        return real_pretrained(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy_load)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", spy_pretrained)
    with pytest.raises(ModelLoadError, match=message):
        check_model_dir(model_dir)
    with pytest.raises(ModelLoadError, match=message):
        load_model_dir(model_dir)
    assert calls == []  # refused before transformers is called: no pickle is ever read


def test_weight_override_and_adapter_dirs_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch, "load", lambda *a, **k: calls.append("torch.load"))
    model_dir = save_tiny_model(tmp_path / "m")
    config_text = (model_dir / "config.json").read_text(encoding="utf-8")
    config = json.loads(config_text)
    config["transformers_weights"] = "weights-file"
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="transformers_weights"):
        load_model_dir(model_dir)
    (model_dir / "config.json").write_text(config_text, encoding="utf-8")
    (model_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "x"}), encoding="utf-8"
    )
    with pytest.raises(ModelLoadError, match="PEFT adapter"):
        load_model_dir(model_dir)
    (model_dir / "adapter_config.json").unlink()
    (model_dir / "config.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ModelLoadError, match="not a JSON object"):
        check_model_dir(model_dir)
    (model_dir / "config.json").write_text(config_text, encoding="utf-8")
    (model_dir / "model.safetensors").rename(model_dir / "consolidated.safetensors")
    with pytest.raises(ModelLoadError, match="non-standard safetensors"):
        check_model_dir(model_dir)
    assert calls == []


def test_safetensors_index_checkpoint_loads(tmp_path: Path) -> None:
    from safetensors.torch import load_file, save_file

    model_dir = save_tiny_model(tmp_path / "m")
    state = load_file(str(model_dir / "model.safetensors"))
    (model_dir / "model.safetensors").unlink()
    save_file(state, str(model_dir / "model-00001-of-00001.safetensors"))
    index = {"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in state}}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    assert check_model_dir(model_dir) == [model_dir / "model-00001-of-00001.safetensors"]
    assert load_model_dir(model_dir).tiny_demo is False


def test_corrupt_weights_and_mismatched_store_are_clean_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import pickle

    def no_pickle(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("pickle / torch.load must never be called")

    monkeypatch.setattr(torch, "load", no_pickle)
    monkeypatch.setattr(pickle, "load", no_pickle)
    monkeypatch.setattr(pickle, "loads", no_pickle)
    corrupt = save_tiny_model(tmp_path / "corrupt")
    weights = sorted(corrupt.glob("*.safetensors"))
    assert weights
    weights[0].write_bytes(pickle.dumps({"not": "safetensors"}))  # a pickle under a .safetensors name
    with pytest.raises(ModelLoadError, match="cannot load model dir"):
        load_model_dir(corrupt)
    with pytest.raises(SystemExit) as exc_info:
        server_mod.main(["--model-dir", str(corrupt), "--store", str(tmp_path / "s0"), "--port", "0"])
    assert exc_info.value.code == 2
    assert "cannot load model dir" in capsys.readouterr().err

    # a store whose alpha ledger is pinned (one evaluation) or that holds another model's weights
    store = tmp_path / "store"
    first = _make_app(tmp_path, store_dir=store)
    try:
        c = Client(first)
        c.result("POST", "/api/stage", {"op": "ablate", "selection": {"0": [1]}, "params": {}})
        c.result("POST", "/api/evaluate", {})
        assert c.json("POST", "/api/discard", {})[0] == 200
    finally:
        first.close()

    def stop(self: Any, *a: Any, **k: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(server_mod.UIServer, "serve_forever", stop)
    base = ["--tiny-demo", "--port", "0", "--store", str(store), "--items-transfer", "4"]
    base += ["--items-retention", "16", "--alpha-schedule", "per_decision", "--reward-items", "8"]
    assert server_mod.main(base) == 0  # the same settings reopen the store
    capsys.readouterr()
    for extra, expected in ((["--alpha", "0.1"], "ValueError"), (["--model-seed", "7"], "StoreError")):
        with pytest.raises(SystemExit) as exc_info:
            server_mod.main([*base, *extra])
        assert exc_info.value.code == 2, extra
        err = capsys.readouterr().err
        assert f"cannot start: {expected}" in err and "Traceback" not in err, err


def test_model_dir_app_serves(tmp_path: Path) -> None:
    good = save_tiny_model(tmp_path / "good")
    loaded = load_model_dir(good)
    options = ServerOptions(port=0, token=TOKEN)
    app = create_app(loaded, store=tmp_path / "store", options=options, config=SMALL)
    app.serve_in_thread()
    try:
        c = Client(app)
        status = c.json("GET", "/api/status")[1]
        assert status["tiny_demo"] is False and status["tiny_note"] is None
    finally:
        app.close()
    # reopening the same store restores its head and keeps the versions
    app2 = create_app(
        load_model_dir(good),
        store=tmp_path / "store",
        options=ServerOptions(port=0, token=TOKEN),
        config=SMALL,
    )
    try:
        assert len(app2.workspace.public_history()["versions"]) == 1
    finally:
        app2.close()


# ============================================================================== bind / token / CLI
def test_bind_host_rules() -> None:
    assert check_bind_host("127.0.0.1", False) == ("127.0.0.1", True)
    assert check_bind_host("localhost", False) == ("127.0.0.1", True)
    assert check_bind_host("::1", False) == ("::1", True)
    with pytest.raises(BindError, match="allow-remote"):
        check_bind_host("192.168.1.10", False)
    assert check_bind_host("10.1.2.3", True) == ("10.1.2.3", False)
    for wildcard in ("0.0.0.0", "::"):  # the Host allowlist needs the address clients use
        with pytest.raises(BindError, match="wildcard"):
            check_bind_host(wildcard, True)
    with pytest.raises(BindError, match="IP address"):
        check_bind_host("example.com", True)
    with pytest.raises(BindError):  # refused before the model or the store is touched
        create_app(
            build_tiny_demo(0),
            store="/nonexistent",
            options=ServerOptions(host="10.1.2.3", port=0, token=TOKEN),
        )
    with pytest.raises(SystemExit):
        server_mod.main(["--tiny-demo", "--host", "10.1.2.3", "--port", "0"])


def test_token_source() -> None:
    a, b = resolve_token({}), resolve_token({})
    assert a != b and len(a) >= 32
    assert resolve_token({TOKEN_ENV: TOKEN}) == TOKEN
    with pytest.raises(ValueError):
        resolve_token({TOKEN_ENV: "short"})
    with pytest.raises(ValueError):
        resolve_token({TOKEN_ENV: "x" * 20 + "/../"})


def test_main_prints_token_url_once_and_warns_for_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def stop(self: Any, *a: Any, **k: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(server_mod.UIServer, "serve_forever", stop)
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    store = tmp_path / "store"
    assert (
        server_mod.main(["--tiny-demo", "--port", "0", "--store", str(store), "--items-retention", "16"]) == 0
    )
    out, err = capsys.readouterr()
    assert out.count(f"#token={TOKEN}") == 1 and "WARNING" not in err
    assert "say nothing about real large models" in out
    # the token is never written to disk
    for path in store.rglob("*"):
        if path.is_file():
            assert TOKEN.encode() not in path.read_bytes(), path
    ip = _local_ip()
    if ip is None:
        pytest.skip("no non-loopback IPv4 interface to test --allow-remote on")
    argv = ["--tiny-demo", "--host", ip, "--allow-remote", "--port", "0", "--store", str(tmp_path / "s2")]
    assert server_mod.main([*argv, "--items-retention", "16"]) == 0
    out, err = capsys.readouterr()
    assert "WARNING" in err and "non-loopback" in err and out.count("#token=") == 1


def _local_ip() -> str | None:
    """A non-loopback IPv4 address of this machine (no packet is sent), or ``None``."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 9))
        ip = str(sock.getsockname()[0])
    except OSError:
        return None
    finally:
        sock.close()
    return None if ip.startswith("127.") or ip == "0.0.0.0" else ip


def test_remote_bind_checks_host(tmp_path: Path) -> None:
    ip = _local_ip()
    if ip is None:
        pytest.skip("no non-loopback IPv4 interface")
    options = ServerOptions(host=ip, port=0, token=TOKEN, allow_remote=True)
    app = _make_app(tmp_path, options=options)
    try:
        c = Client(app)
        assert c.authority == f"{ip}:{c.port}"
        assert c.raw("GET", "/api/status")[0] == 200
        assert c.raw("GET", "/api/status", host=f"localhost:{c.port}")[0] == 403  # not a loopback bind
        assert c.raw("GET", "/api/status", host=f"10.0.0.5:{c.port}")[0] == 403
        assert c.raw("GET", "/api/status", token=None)[0] == 401
    finally:
        app.close()


# ============================================================================== decision logs
def _decision(i: int, decision: GateDecision, reward: float, point: float) -> dict[str, Any]:
    verdict = GateVerdict(
        decision_index=i,
        transfer=SliceTest(
            "t_rule/unseen_pairs", 64, point, point - 0.05, point > 0.05, "sign_test_exact", 0.05, 0.01
        ),
        retention=(SliceTest("retention/atomic", 512, 0.0, 0.01, True, "tango_score", 0.05, 0.001, 0.02),),
        alarm=False,
    )
    rec = DecisionRecord(
        decision_index=i,
        update_index=8 * (i + 1),
        decision=decision,
        verdicts=[verdict],
        reason=decision.value,
        reward_v1=reward,
        reward_v2=reward / 2,
    )
    return rec.to_dict()


def test_runs_decision_log(client: Client, app: UIApp, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    lines = [
        json.dumps(_decision(0, GateDecision.ACCEPT, 0.1, 0.2)),
        json.dumps(_decision(1, GateDecision.HOLD, 0.4, 0.0)),
        json.dumps(_decision(2, GateDecision.ACCEPT, 0.5, 0.1)),
        '{"decision_index": 3, "trunc',  # torn last line
    ]
    (runs / "arm_a.jsonl").write_text("\n".join(lines))
    (runs / "arm_b").mkdir()
    (runs / "arm_b" / "decisions.jsonl").write_text(lines[0] + "\n")
    outside = tmp_path / "outside.jsonl"
    outside.write_text(lines[0] + "\n")
    (runs / "escape.jsonl").symlink_to(outside)
    status, data = client.json("GET", "/api/runs")
    assert status == 200 and [r["name"] for r in data["runs"]] == ["arm_a", "arm_b"]
    status, data = client.json("GET", "/api/runs/arm_a/decisions")
    assert status == 200 and data["total_records"] == 3 and data["skipped_lines"] == 0
    recs = data["records"]
    assert [r["decision"] for r in recs] == ["accept", "hold", "accept"]
    assert recs[1]["reward_change_v1"] == pytest.approx(0.3)
    assert recs[2]["accepted_transfer_gain"] == pytest.approx(0.3)
    assert recs[0]["first_verdict"]["retention"][0]["margin"] == 0.02
    assert client.json("GET", "/api/runs/arm_b/decisions")[1]["total_records"] == 1
    assert client.json("GET", "/api/runs/escape/decisions")[0] == 403
    assert client.json("GET", "/api/runs/.hidden/decisions")[0] == 403
    assert client.json("GET", "/api/runs/..%2f..%2fetc/decisions")[0] == 404
    assert client.json("GET", "/api/runs/%2e%2e/decisions")[0] == 404
    assert client.json("GET", "/api/runs/missing/decisions")[0] == 404
    assert client.json("GET", "/api/runs/arm_a/other")[0] == 404


def test_runs_log_skips_lines_json_cannot_hold(client: Client, tmp_path: Path) -> None:
    good = json.dumps(_decision(0, GateDecision.ACCEPT, 0.1, 0.2))
    huge_reward = json.loads(good)
    huge_reward["reward_v1"] = 10**400  # parses, but is too large for a float
    lines = [
        good,
        '{"decision_index": ' + "9" * 5000 + "}",  # over Python's int-string limit (ValueError)
        "[" * 100_000 + "]" * 100_000,  # too deep to parse (RecursionError)
        json.dumps(huge_reward),
        good,
    ]
    (tmp_path / "runs" / "odd.jsonl").write_text("\n".join(lines) + "\n")
    status, data = client.json("GET", "/api/runs/odd/decisions")
    assert status == 200, data
    assert data["total_records"] == 3 and data["skipped_lines"] == 2
    assert [r["reward_v1"] for r in data["records"]] == [0.1, None, 0.1]


def test_tokens_and_store_files_stay_local(client: Client, app: UIApp) -> None:
    client.result(
        "POST",
        "/api/stage",
        {"op": "ablate", "selection": {"1": [2]}, "params": {}, "note": "<script>x</script>"},
    )
    client.result("POST", "/api/commit", {"override": "<img src=x onerror=alert(1)>"})
    root = Path(app.workspace.store.root)
    for path in root.rglob("*"):
        if path.is_file():
            assert TOKEN.encode() not in path.read_bytes()
            assert path.suffix in (".json", ".safetensors"), path
    history = client.json("GET", "/api/history")[1]
    assert any(v["override"] == "<img src=x onerror=alert(1)>" for v in history["versions"])  # stored as data


def test_scores_marked_stale_after_weights_change(client: Client) -> None:
    client.result("POST", "/api/stats", {"split": "train", "n": 12})
    scores = client.json("GET", "/api/scores")[1]["scores"]
    assert scores["wanda:train"]["stale"] is False and scores["random"]["stale"] is False
    client.result(
        "POST", "/api/stage", {"op": "scale", "selection": {"0": [1, 2]}, "params": {"factor": 3.0}}
    )
    client.result("POST", "/api/commit", {"override": "stale test"})
    client.result("POST", "/api/stats", {"split": "retention", "n": 12})
    scores = client.json("GET", "/api/scores")[1]["scores"]
    assert scores["wanda:train"]["stale"] is True  # train activations are from the old weights
    assert scores["wanda:retention"]["stale"] is False
    assert scores["random"]["stale"] is False  # does not depend on the weights
    ranked = client.json("GET", "/api/channels?score=wanda&split=train&top=1")[1]
    assert ranked["score"]["stale"] is True
    status = client.json("GET", "/api/status")[1]
    assert status["stats"]["train"]["stale"] is True and status["stats"]["retention"]["stale"] is False


def test_static_front_end_is_self_contained() -> None:
    static = server_mod.STATIC_DIR
    assert {p.name for p in static.iterdir()} == {"index.html", "app.js", "style.css"}
    html = (static / "index.html").read_text()
    js = (static / "app.js").read_text()
    css = (static / "style.css").read_text()
    for tab in ("search", "inspect", "edit", "versions", "training"):
        assert f'data-tab="{tab}"' in html and f'id="tab-{tab}"' in html
    assert re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html) is None  # no inline script
    assert re.search(r"\son[a-z]+\s*=", html) is None  # no inline event handlers
    assert "style=" not in html and "<style" not in html
    for text in (
        html,
        js.replace('"http://www.w3.org/2000/svg"', "SVG_NS"),
        css,
    ):  # a namespace name, not a fetch
        assert (
            re.search(r"(?:https?:)?//[a-z0-9.-]+\.[a-z]{2,}/", text, re.IGNORECASE) is None
        )  # no external assets
    for forbidden in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
    ):
        assert forbidden not in js, forbidden
    assert "localStorage" not in js and "sessionStorage" not in js and "document.cookie" not in js
