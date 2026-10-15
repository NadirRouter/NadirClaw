"""Usage telemetry: what gets sent, and that opting out sends nothing."""

import json

import pytest

from nadirclaw import analytics, metrics
from nadirclaw.cli import main


@pytest.fixture
def sent(monkeypatch, tmp_path):
    monkeypatch.delenv("NADIRCLAW_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.setattr(analytics, "_STATE", tmp_path / "telemetry.json")
    bodies = []
    monkeypatch.setattr(analytics, "_post", lambda body: bodies.append(json.loads(body)))
    return bodies


def _commands(bodies):
    return [b["properties"]["command"] for b in bodies]


def test_cli_sends_command_names_never_arguments(sent):
    analytics.cli_command(main, ["classify", "my secret prompt"])
    analytics.cli_command(main, ["serve", "--port", "9000", "--verbose"])
    analytics.cli_command(main, ["claude", "onboard", "--api-key", "sk-xyz"])
    analytics.cli_command(main, ["auth", "openai", "login"])
    analytics.cli_command(main, ["claude", "read-hook"])  # runs on every Claude Code Read
    analytics.cli_command(main, ["telemetry", "off"])
    analytics.cli_command(main, ["not-a-command", "x"])
    analytics.cli_command(main, [])
    for t in analytics._pending:
        t.join()

    assert sorted(_commands(sent)) == ["auth openai login", "classify", "claude onboard", "serve"]
    raw = json.dumps(sent)
    assert "secret" not in raw and "sk-xyz" not in raw and "9000" not in raw
    # One stable anonymous id across invocations.
    assert len({b["distinct_id"] for b in sent}) == 1


@pytest.mark.parametrize("var,value", [
    ("NADIRCLAW_TELEMETRY", "0"),
    ("NADIRCLAW_TELEMETRY", "off"),
    ("DO_NOT_TRACK", "1"),
])
def test_env_opt_out_sends_nothing(sent, monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    analytics.capture("nadirclaw_cli", {"command": "status"}, wait=True)
    assert sent == []
    assert not analytics._STATE.exists()  # no id minted either


def test_telemetry_off_persists(sent):
    analytics.set_enabled(False)
    analytics.capture("nadirclaw_cli", wait=True)
    assert sent == []
    analytics.set_enabled(True)
    analytics.capture("nadirclaw_cli", wait=True)
    assert len(sent) == 1


def test_usage_heartbeat_sends_deltas_only(sent, monkeypatch):
    monkeypatch.setattr(analytics, "_last_usage", {})
    for name in ("requests_total", "tokens_prompt_total", "cost_total"):
        monkeypatch.setattr(metrics, name, metrics._Counter())
    metrics.requests_total.inc(("claude-haiku-4-5", "simple", "ok"), 3)
    metrics.requests_total.inc(("claude-opus-5", "complex", "error"), 1)
    metrics.tokens_prompt_total.inc(("claude-haiku-4-5",), 120)

    analytics._send_usage(wait=True)
    analytics._send_usage(wait=True)  # nothing new: no event
    metrics.requests_total.inc(("claude-haiku-4-5", "simple", "ok"), 2)
    analytics._send_usage(wait=True)

    first, second = (b["properties"] for b in sent)
    assert (first["requests"], first["requests_simple"], first["requests_complex"]) == (4, 3, 1)
    assert (first["errors"], first["prompt_tokens"]) == (1, 120)
    assert (second["requests"], second["requests_simple"], second["prompt_tokens"]) == (2, 2, 0)


def test_serve_event_hides_private_model_names(sent, monkeypatch):
    class S:
        tier_models = ["claude-opus-5", "ollama/acme-internal", "openai/ft:gpt-4o:acme:x", "gpt-5"]
        COMPLEXITY_ANALYZER_TYPE = "binary"
        OPTIMIZE = "off"
        AUTH_TOKEN = ""
        PII_REDACTION_MODE = "none"
        PROMPT_GUARD_ACTION = "log"

    monkeypatch.setattr(analytics, "_stop", __import__("threading").Event())
    analytics.server_started(S(), {"anthropic": "oauth"})
    analytics._stop.set()
    for t in analytics._pending:
        t.join()

    props = sent[-1]["properties"]
    assert props["models"] == ["claude-opus-5", "gpt-5"]
    assert props["providers"] == ["anthropic", "ollama", "openai"]
    assert "acme" not in json.dumps(sent)


def test_telemetry_never_raises(sent, monkeypatch):
    monkeypatch.setattr(analytics, "_install_id", lambda: 1 / 0)
    analytics.cli_command(main, ["status"])  # must not raise
