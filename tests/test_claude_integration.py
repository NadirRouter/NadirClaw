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

    detected = detect_models(
        claude_settings={}, claude_json={}, fetch_live_models=lambda: []
    )
    # Hardcoded fallback fires only when both local config and the live
    # Anthropic API turn up nothing.
    assert detected.simple and "haiku" in detected.simple
    assert detected.complex and ("sonnet" in detected.complex or "opus" in detected.complex)
    assert "defaults" in detected.sources


def test_detect_models_uses_live_anthropic_api_when_config_empty():
    from nadirclaw.claude_integration import detect_models

    detected = detect_models(
        claude_settings={},
        claude_json={},
        fetch_live_models=lambda: [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
        ],
    )
    assert detected.simple == "claude-haiku-4-5"
    assert detected.complex in {"claude-opus-4-6", "claude-sonnet-4-6"}
    assert "anthropic api" in detected.sources
    assert "defaults" not in detected.sources


def test_detect_models_skips_live_fetch_when_config_has_models():
    from nadirclaw.claude_integration import detect_models

    sentinel = {"called": False}

    def fetcher():
        sentinel["called"] = True
        return ["claude-opus-4-6"]

    detected = detect_models(
        claude_settings={"model": "claude-sonnet-4-6"},
        claude_json={},
        fetch_live_models=fetcher,
    )
    assert detected.complex == "claude-sonnet-4-6"
    assert sentinel["called"] is False, "live fetch must not run when local config supplies models"


def test_detect_models_live_fetch_failure_falls_through_to_defaults():
    from nadirclaw.claude_integration import detect_models

    def fetcher():
        raise RuntimeError("network down")

    detected = detect_models(
        claude_settings={}, claude_json={}, fetch_live_models=fetcher
    )
    assert "defaults" in detected.sources
    assert "anthropic api" not in detected.sources


def test_gather_candidate_models_dedupes_and_attributes_sources():
    from nadirclaw.claude_integration import gather_candidate_models

    pool = gather_candidate_models(
        claude_settings={"model": "claude-sonnet-4-6"},
        claude_json={"env": {"ANTHROPIC_SMALL_FAST_MODEL": "claude-haiku-4-5"}},
        fetch_live_models=lambda: ["claude-opus-4-6", "claude-sonnet-4-6"],
    )
    # Config models come first, then live API, deduped.
    assert pool.models[0] == "claude-sonnet-4-6"
    assert "claude-haiku-4-5" in pool.models
    assert "claude-opus-4-6" in pool.models
    assert pool.models.count("claude-sonnet-4-6") == 1
    assert pool.source_of["claude-opus-4-6"] == "anthropic api"
    assert "anthropic api" in pool.sources
    assert "defaults" not in pool.sources


def test_gather_candidate_models_uses_defaults_when_everything_else_empty():
    from nadirclaw.claude_integration import gather_candidate_models

    pool = gather_candidate_models(
        claude_settings={}, claude_json={}, fetch_live_models=lambda: []
    )
    assert pool.sources == ["defaults"]
    assert "claude-haiku-4-5" in pool.models
    assert "claude-sonnet-4-6" in pool.models


def test_interactive_pick_models_number_selection():
    from nadirclaw.claude_integration import (
        CandidatePool,
        DetectedModels,
        interactive_pick_models,
    )

    pool = CandidatePool(
        models=["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-6"],
        sources=["anthropic api"],
        source_of={
            "claude-haiku-4-5": "anthropic api",
            "claude-sonnet-4-6": "anthropic api",
            "claude-opus-4-6": "anthropic api",
        },
    )
    answers = iter(["1", "skip", "3", "3"])  # simple, mid, complex, reasoning

    def prompt(_msg, default=None, show_default=False):
        return next(answers)

    result = interactive_pick_models(pool, defaults=DetectedModels(), prompt_fn=prompt, echo_fn=lambda _msg: None)
    assert result.simple == "claude-haiku-4-5"
    assert result.mid is None
    assert result.complex == "claude-opus-4-6"
    assert result.reasoning == "claude-opus-4-6"


def test_interactive_pick_models_accepts_freeform_id_and_default():
    from nadirclaw.claude_integration import (
        CandidatePool,
        DetectedModels,
        interactive_pick_models,
    )

    pool = CandidatePool(models=["claude-haiku-4-5"], sources=["defaults"])
    defaults = DetectedModels(
        simple="claude-haiku-4-5",
        complex="claude-sonnet-4-6",
        reasoning="claude-sonnet-4-6",
    )

    # User accepts default for simple, types a custom id for complex, skips mid + reasoning.
    answers = iter(["claude-haiku-4-5", "skip", "claude-opus-4-6", "skip"])

    def prompt(_msg, default=None, show_default=False):
        return next(answers)

    result = interactive_pick_models(pool, defaults=defaults, prompt_fn=prompt, echo_fn=lambda _msg: None)
    assert result.simple == "claude-haiku-4-5"
    assert result.complex == "claude-opus-4-6"
    # reasoning was 'skip' but the fill-in rule promotes complex → reasoning
    assert result.reasoning == "claude-opus-4-6"


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


def test_patch_claude_settings_writes_default_profile(_redirect_paths):
    from nadirclaw.claude_integration import patch_claude_settings

    patch_claude_settings(
        "http://localhost:8856/v1", api_key="local", default_profile="nadir-auto"
    )
    cfg = json.loads(_redirect_paths["settings"].read_text())
    assert cfg["env"]["ANTHROPIC_MODEL"] == "nadir-auto"


def test_patch_claude_settings_omits_default_profile_when_none(_redirect_paths):
    from nadirclaw.claude_integration import patch_claude_settings

    settings_path = _redirect_paths["settings"]
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps({"env": {"ANTHROPIC_MODEL": "claude-opus-4-7"}}))

    patch_claude_settings("http://localhost:8856/v1", default_profile=None)
    cfg = json.loads(settings_path.read_text())
    # Existing user-set model is preserved when caller doesn't pass a profile.
    assert cfg["env"]["ANTHROPIC_MODEL"] == "claude-opus-4-7"


def test_unpatch_claude_settings_preserves_user_set_model(_redirect_paths):
    from nadirclaw.claude_integration import unpatch_claude_settings

    settings_path = _redirect_paths["settings"]
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps({
        "env": {
            "ANTHROPIC_BASE_URL": "http://localhost:8856/v1",
            "ANTHROPIC_API_KEY": "local",
            "ANTHROPIC_MODEL": "claude-opus-4-7",  # user's choice, not ours
        },
    }))

    unpatch_claude_settings()
    cfg = json.loads(settings_path.read_text())
    # Our two keys are gone, but the user's hand-set model survived.
    assert "ANTHROPIC_BASE_URL" not in cfg["env"]
    assert cfg["env"]["ANTHROPIC_MODEL"] == "claude-opus-4-7"


def test_unpatch_claude_settings_removes_nadir_profile_model(_redirect_paths):
    from nadirclaw.claude_integration import unpatch_claude_settings

    settings_path = _redirect_paths["settings"]
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps({
        "env": {
            "ANTHROPIC_BASE_URL": "http://localhost:8856/v1",
            "ANTHROPIC_MODEL": "nadir-auto",  # we set this, we own it
        },
    }))

    unpatch_claude_settings()
    cfg = json.loads(settings_path.read_text())
    assert "ANTHROPIC_MODEL" not in cfg.get("env", {})


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
