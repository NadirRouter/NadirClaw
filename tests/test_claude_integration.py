"""Tests for nadirclaw.claude_integration — seamless Claude Code onboarding."""

import json
import platform
import stat
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _redirect_paths(tmp_path, monkeypatch):
    """Point all on-disk targets at a temp directory so tests don't touch $HOME."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    fake_config = fake_home / ".nadirclaw"
    fake_config.mkdir()
    fake_env = fake_config / ".env"
    fake_logs = fake_config / "logs"
    fake_logs.mkdir()
    fake_shim_dir = fake_config / "bin"
    fake_shim_path = fake_shim_dir / "claude"

    fake_claude_dir = fake_home / ".claude"
    fake_settings = fake_claude_dir / "settings.json"
    fake_claude_json = fake_home / ".claude.json"

    fake_launchd = fake_home / "Library" / "LaunchAgents" / "com.nadirclaw.daemon.plist"
    fake_systemd = fake_home / ".config" / "systemd" / "user" / "nadirclaw.service"

    monkeypatch.setattr("nadirclaw.setup.CONFIG_DIR", fake_config)
    monkeypatch.setattr("nadirclaw.setup.ENV_FILE", fake_env)
    monkeypatch.setattr("nadirclaw.claude_integration.CONFIG_DIR", fake_config)
    monkeypatch.setattr("nadirclaw.claude_integration.ENV_FILE", fake_env)
    monkeypatch.setattr("nadirclaw.claude_integration.CLAUDE_DIR", fake_claude_dir)
    monkeypatch.setattr("nadirclaw.claude_integration.CLAUDE_SETTINGS_FILE", fake_settings)
    monkeypatch.setattr("nadirclaw.claude_integration.CLAUDE_JSON_FILE", fake_claude_json)
    monkeypatch.setattr("nadirclaw.claude_integration.SHIM_DIR", fake_shim_dir)
    monkeypatch.setattr("nadirclaw.claude_integration.SHIM_PATH", fake_shim_path)
    monkeypatch.setattr("nadirclaw.claude_integration.LAUNCHD_PATH", fake_launchd)
    monkeypatch.setattr("nadirclaw.claude_integration.SYSTEMD_UNIT_PATH", fake_systemd)

    return {
        "home": fake_home,
        "env": fake_env,
        "settings": fake_settings,
        "claude_json": fake_claude_json,
        "shim": fake_shim_path,
        "launchd": fake_launchd,
        "systemd": fake_systemd,
        "logs": fake_logs,
    }


# ---------------------------------------------------------------------------
# detect_models
# ---------------------------------------------------------------------------

def test_detect_models_buckets_by_tier():
    from nadirclaw.claude_integration import detect_models

    detected = detect_models(
        claude_settings={"model": "claude-sonnet-4-5-20250929"},
        claude_json={
            "env": {
                "ANTHROPIC_MODEL": "claude-opus-4-1-20250805",
                "ANTHROPIC_SMALL_FAST_MODEL": "claude-haiku-4-5-20251001",
            },
        },
    )
    assert detected.simple == "claude-haiku-4-5-20251001"
    assert detected.complex == "claude-sonnet-4-5-20250929"
    # Opus has no special "reasoning" marker in classify_model_tier; the
    # complex slot wins, but reasoning falls back to whichever complex won.
    assert detected.reasoning == "claude-sonnet-4-5-20250929"


def test_detect_models_falls_back_to_anthropic_defaults_when_no_config():
    from nadirclaw.claude_integration import detect_models

    detected = detect_models(claude_settings={}, claude_json={})
    # Fallback uses the Anthropic family so onboarding still works without
    # Claude Code installed.
    assert detected.simple and "haiku" in detected.simple
    assert detected.complex and ("sonnet" in detected.complex or "opus" in detected.complex)
    assert "defaults" in detected.sources


def test_detect_models_single_model_fills_both_slots():
    from nadirclaw.claude_integration import detect_models

    detected = detect_models(
        claude_settings={"model": "claude-sonnet-4-5-20250929"},
        claude_json={},
    )
    assert detected.complex == "claude-sonnet-4-5-20250929"
    # Single complex model gets mirrored to simple so the proxy still boots.
    assert detected.simple == "claude-sonnet-4-5-20250929"


# ---------------------------------------------------------------------------
# update_env_file
# ---------------------------------------------------------------------------

def test_update_env_file_writes_tier_keys(_redirect_paths):
    from nadirclaw.claude_integration import DetectedModels, update_env_file

    models = DetectedModels(
        simple="claude-haiku-4-5-20251001",
        complex="claude-sonnet-4-5-20250929",
        reasoning="claude-opus-4-1-20250805",
    )
    update_env_file(models)

    body = _redirect_paths["env"].read_text()
    assert "NADIRCLAW_SIMPLE_MODEL=claude-haiku-4-5-20251001" in body
    assert "NADIRCLAW_COMPLEX_MODEL=claude-sonnet-4-5-20250929" in body
    assert "NADIRCLAW_REASONING_MODEL=claude-opus-4-1-20250805" in body


def test_update_env_file_preserves_other_keys_and_backs_up(_redirect_paths):
    from nadirclaw.claude_integration import DetectedModels, update_env_file

    env = _redirect_paths["env"]
    env.write_text(
        "OPENAI_API_KEY=sk-original\n"
        "NADIRCLAW_PORT=8856\n"
        "NADIRCLAW_SIMPLE_MODEL=old-simple\n"
    )

    update_env_file(DetectedModels(simple="haiku-new", complex="sonnet-new"))

    body = env.read_text()
    assert "OPENAI_API_KEY=sk-original" in body
    assert "NADIRCLAW_PORT=8856" in body
    assert "NADIRCLAW_SIMPLE_MODEL=haiku-new" in body
    assert "old-simple" not in body
    assert "NADIRCLAW_COMPLEX_MODEL=sonnet-new" in body

    backups = list(env.parent.glob(".env.backup-*"))
    assert backups, "expected a backup of the previous .env"


# ---------------------------------------------------------------------------
# patch_claude_settings
# ---------------------------------------------------------------------------

def test_patch_claude_settings_creates_file_with_env_block(_redirect_paths):
    from nadirclaw.claude_integration import patch_claude_settings

    patch_claude_settings("http://localhost:8856/v1", api_key="local")

    cfg = json.loads(_redirect_paths["settings"].read_text())
    assert cfg["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:8856/v1"
    assert cfg["env"]["ANTHROPIC_API_KEY"] == "local"


def test_patch_claude_settings_preserves_existing_keys(_redirect_paths):
    from nadirclaw.claude_integration import patch_claude_settings

    settings_path = _redirect_paths["settings"]
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps({
        "model": "claude-sonnet-4-5-20250929",
        "env": {"CUSTOM": "keep-me"},
        "theme": "dark",
    }))

    patch_claude_settings("http://localhost:8856/v1")

    cfg = json.loads(settings_path.read_text())
    assert cfg["model"] == "claude-sonnet-4-5-20250929"
    assert cfg["theme"] == "dark"
    assert cfg["env"]["CUSTOM"] == "keep-me"
    assert cfg["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:8856/v1"

    backups = list(settings_path.parent.glob("settings.backup-*.json"))
    assert backups, "expected a backup"


def test_unpatch_claude_settings_removes_only_nadirclaw_keys(_redirect_paths):
    from nadirclaw.claude_integration import unpatch_claude_settings

    settings_path = _redirect_paths["settings"]
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps({
        "env": {
            "ANTHROPIC_BASE_URL": "http://localhost:8856/v1",
            "ANTHROPIC_API_KEY": "local",
            "CUSTOM": "keep-me",
        },
    }))

    assert unpatch_claude_settings() is True
    cfg = json.loads(settings_path.read_text())
    assert "ANTHROPIC_BASE_URL" not in cfg["env"]
    assert "ANTHROPIC_API_KEY" not in cfg["env"]
    assert cfg["env"]["CUSTOM"] == "keep-me"


# ---------------------------------------------------------------------------
# Shim
# ---------------------------------------------------------------------------

def test_install_shim_writes_executable(_redirect_paths):
    from nadirclaw.claude_integration import install_shim

    path = install_shim(port=8856)

    assert path.exists()
    assert path.stat().st_mode & stat.S_IXUSR
    content = path.read_text()
    assert "NADIRCLAW_PORT=" in content
    assert "ANTHROPIC_BASE_URL=" in content
    assert "exec \"$REAL_CLAUDE\"" in content


def test_uninstall_shim_idempotent(_redirect_paths):
    from nadirclaw.claude_integration import install_shim, uninstall_shim

    install_shim(port=8856)
    assert uninstall_shim() is True
    assert uninstall_shim() is False


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

def test_install_daemon_writes_launchd_on_darwin(_redirect_paths, monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    # Stub launchctl so we don't actually load the unit during tests.
    monkeypatch.setattr(
        "nadirclaw.claude_integration.subprocess.run",
        lambda *a, **k: None,
    )
    from nadirclaw.claude_integration import install_daemon

    path = install_daemon(port=8856, log_dir=_redirect_paths["logs"])
    assert path == _redirect_paths["launchd"]
    body = path.read_text()
    assert "<key>Label</key>" in body
    assert "com.nadirclaw.daemon" in body
    assert "<string>serve</string>" in body
    assert "<string>8856</string>" in body


def test_install_daemon_writes_systemd_on_linux(_redirect_paths, monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        "nadirclaw.claude_integration.subprocess.run",
        lambda *a, **k: None,
    )
    from nadirclaw.claude_integration import install_daemon

    path = install_daemon(port=8856, log_dir=_redirect_paths["logs"])
    assert path == _redirect_paths["systemd"]
    body = path.read_text()
    assert "[Service]" in body
    assert "serve --port 8856" in body


def test_install_daemon_returns_none_on_unsupported(_redirect_paths, monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    from nadirclaw.claude_integration import install_daemon

    assert install_daemon(port=8856, log_dir=_redirect_paths["logs"]) is None


def test_uninstall_daemon_removes_units(_redirect_paths, monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        "nadirclaw.claude_integration.subprocess.run",
        lambda *a, **k: None,
    )
    from nadirclaw.claude_integration import install_daemon, uninstall_daemon

    install_daemon(port=8856, log_dir=_redirect_paths["logs"])
    removed = uninstall_daemon()
    assert _redirect_paths["launchd"] in removed
    assert not _redirect_paths["launchd"].exists()


# ---------------------------------------------------------------------------
# resolve_profile — nadir-* aliases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,expected", [
    ("nadir-auto", "auto"),
    ("nadir-eco", "eco"),
    ("nadir-premium", "premium"),
    ("nadir-reasoning", "reasoning"),
    ("nadir-free", "free"),
    ("NADIR-ECO", "eco"),
    ("nadirclaw/premium", "premium"),
    ("auto", "auto"),
    ("claude-sonnet-4-5-20250929", None),
    ("", None),
    (None, None),
])
def test_resolve_profile_accepts_nadir_prefix(model, expected):
    from nadirclaw.routing import resolve_profile

    assert resolve_profile(model) == expected
