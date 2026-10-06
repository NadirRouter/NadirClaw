"""Tokenless mode serves localhost only; `serve` binds loopback by default."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from starlette.requests import Request

from nadirclaw import auth as auth_mod
from nadirclaw.auth import is_loopback
from nadirclaw.cli import main

LOOPBACK = ("127.0.0.1", 50000)
LAN = ("192.168.1.20", 50000)


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("NADIRCLAW_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("NADIRCLAW_AUTH_TOKEN", raising=False)
    from nadirclaw.server import app
    return app


def test_is_loopback():
    for host in ("localhost", "127.0.0.1", "127.8.0.1", "::1"):
        assert is_loopback(host), host
    for host in ("0.0.0.0", "::", "192.168.1.20", "evil.example", "", None):
        assert not is_loopback(host), host


def test_ipv6_host_header_parses_as_loopback():
    # TestClient cannot take an IPv6 base_url, so check the parse directly.
    req = Request({"type": "http", "scheme": "http", "path": "/", "query_string": b"",
                   "headers": [(b"host", b"[::1]:8856")]})
    assert is_loopback(req.url.hostname)


def test_tokenless_allows_localhost(app):
    for base in ("http://localhost:8856", "http://127.0.0.1:8856"):
        c = TestClient(app, base_url=base, client=LOOPBACK)
        assert c.get("/v1/logs").status_code == 200, base


def test_tokenless_rejects_remote_peer(app):
    c = TestClient(app, base_url="http://localhost:8856", client=LAN)
    assert c.get("/v1/logs").status_code == 401
    assert c.post("/v1/chat/completions", json={"messages": []}).status_code == 401
    assert c.get("/dashboard/api/stats").status_code == 401


def test_tokenless_rejects_rebound_host(app):
    # DNS rebinding: a loopback peer, but the browser sends the attacker's hostname.
    c = TestClient(app, base_url="http://evil.example:8856", client=LOOPBACK)
    assert c.get("/v1/logs").status_code == 401


def test_token_mode_accepts_remote_peer_with_token(app, monkeypatch):
    monkeypatch.setenv("NADIRCLAW_AUTH_TOKEN", "s3cret")
    monkeypatch.setattr(auth_mod, "_LOCAL_USERS", {"s3cret": auth_mod._default_user()})
    c = TestClient(app, base_url="http://10.0.0.5:8856", client=LAN)
    assert c.get("/v1/logs").status_code == 401
    assert c.get("/v1/logs", headers={"Authorization": "Bearer s3cret"}).status_code == 200


@pytest.fixture
def run_serve(monkeypatch):
    # setenv (not delenv) so teardown also undoes the os.environ write `--token` makes.
    monkeypatch.setenv("NADIRCLAW_AUTH_TOKEN", "")

    def run(*args):
        with patch("nadirclaw.setup.is_first_run", return_value=False), \
             patch("uvicorn.run") as uv:
            result = CliRunner().invoke(main, ["serve", *args])
        return result, uv

    return run


def test_serve_binds_loopback_by_default(run_serve):
    result, uv = run_serve()
    assert result.exit_code == 0, result.output
    assert uv.call_args.kwargs["host"] == "127.0.0.1"


def test_serve_refuses_public_bind_without_token(run_serve):
    result, uv = run_serve("--host", "0.0.0.0")
    assert result.exit_code != 0
    assert "NADIRCLAW_AUTH_TOKEN" in result.output
    uv.assert_not_called()


def test_serve_public_bind_with_token(run_serve):
    result, uv = run_serve("--host", "0.0.0.0", "--token", "s3cret")
    assert result.exit_code == 0, result.output
    assert uv.call_args.kwargs["host"] == "0.0.0.0"
