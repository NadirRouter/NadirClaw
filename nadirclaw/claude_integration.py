"""Seamless Claude Code integration.

Two modes:

1. **Full onboard** (`nadirclaw claude onboard`)
   - Detects models declared in `~/.claude/settings.json` (and project overrides).
   - Maps them into NadirClaw tier env vars (simple / mid / complex / reasoning).
   - Writes `ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY` into Claude Code's
     `env` block so future `claude` invocations talk to NadirClaw automatically.
   - Installs a user-scope launchd plist (macOS) or systemd unit (Linux) so the
     proxy starts on login.

2. **Lightweight shim** (`nadirclaw claude shim`)
   - Drops a `claude` wrapper into `~/.nadirclaw/bin` that lazy-starts the
     proxy on first call, then execs the real Claude binary with the env set.
   - Use when you don't want a background daemon or settings.json edits.

Detection helpers are kept pure so they can be unit-tested without touching
the real filesystem.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from nadirclaw.setup import CONFIG_DIR, ENV_FILE, classify_model_tier


CLAUDE_DIR = Path.home() / ".claude"
CLAUDE_SETTINGS_FILE = CLAUDE_DIR / "settings.json"
CLAUDE_JSON_FILE = Path.home() / ".claude.json"

SHIM_DIR = CONFIG_DIR / "bin"
SHIM_PATH = SHIM_DIR / "claude"

LAUNCHD_LABEL = "com.nadirclaw.daemon"
LAUNCHD_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
SYSTEMD_UNIT_PATH = Path.home() / ".config" / "systemd" / "user" / "nadirclaw.service"


@dataclass
class DetectedModels:
    """Models pulled out of Claude Code config, bucketed by tier."""

    simple: Optional[str] = None
    mid: Optional[str] = None
    complex: Optional[str] = None
    reasoning: Optional[str] = None
    sources: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _candidate_models(claude_settings: Dict, claude_json: Dict) -> List[str]:
    """Pull every model id we can find from Claude Code config files."""
    candidates: List[str] = []

    def push(value):
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    for cfg in (claude_settings, claude_json):
        push(cfg.get("model"))
        env = cfg.get("env") or {}
        if isinstance(env, dict):
            push(env.get("ANTHROPIC_MODEL"))
            push(env.get("ANTHROPIC_SMALL_FAST_MODEL"))
        # ~/.claude.json keeps the most recently selected model
        push(cfg.get("lastSelectedModel"))
        push(cfg.get("defaultModel"))

    seen = set()
    deduped: List[str] = []
    for m in candidates:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped


_CLAUDE_FAMILY_FALLBACK = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-1-20250805",
]


def detect_models(
    claude_settings: Optional[Dict] = None,
    claude_json: Optional[Dict] = None,
) -> DetectedModels:
    """Bucket detected Claude Code models into NadirClaw tiers.

    If no config exists, fall back to the Anthropic family defaults so
    that NadirClaw still has reasonable tier targets for Claude Code.
    """
    if claude_settings is None:
        claude_settings = _read_json(CLAUDE_SETTINGS_FILE)
    if claude_json is None:
        claude_json = _read_json(CLAUDE_JSON_FILE)

    found = _candidate_models(claude_settings, claude_json)
    sources: List[str] = []
    if CLAUDE_SETTINGS_FILE.exists() and claude_settings:
        sources.append(str(CLAUDE_SETTINGS_FILE))
    if CLAUDE_JSON_FILE.exists() and claude_json:
        sources.append(str(CLAUDE_JSON_FILE))

    if not found:
        found = list(_CLAUDE_FAMILY_FALLBACK)
        sources.append("defaults")

    buckets = DetectedModels(sources=sources)
    for m in found:
        tier = classify_model_tier(m)
        if tier == "simple" and not buckets.simple:
            buckets.simple = m
        elif tier == "reasoning" and not buckets.reasoning:
            buckets.reasoning = m
        elif tier == "complex" and not buckets.complex:
            buckets.complex = m
        # "mid" doesn't come out of classify_model_tier today, but we keep
        # the slot so user-supplied mids survive.

    # Promote complex → reasoning fallback if reasoning is empty.
    if not buckets.reasoning and buckets.complex:
        buckets.reasoning = buckets.complex
    # If we somehow only saw one tier, keep the strong model on the complex
    # side and reuse it for simple so the proxy still boots.
    if not buckets.complex and buckets.simple:
        buckets.complex = buckets.simple
    if not buckets.simple and buckets.complex:
        buckets.simple = buckets.complex

    return buckets


# ---------------------------------------------------------------------------
# Env file writing
# ---------------------------------------------------------------------------

_TIER_ENV_KEYS = {
    "simple": "NADIRCLAW_SIMPLE_MODEL",
    "mid": "NADIRCLAW_MID_MODEL",
    "complex": "NADIRCLAW_COMPLEX_MODEL",
    "reasoning": "NADIRCLAW_REASONING_MODEL",
}


def _parse_env(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for line in text.splitlines():
        out.append(("", line))
    return out


def update_env_file(models: DetectedModels, env_path: Optional[Path] = None) -> Path:
    """Merge detected tiers into ~/.nadirclaw/.env without clobbering other keys."""
    if env_path is None:
        env_path = ENV_FILE
    env_path.parent.mkdir(parents=True, exist_ok=True)

    existing_lines: List[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text().splitlines()
        backup = env_path.with_name(
            f".env.backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        shutil.copy2(env_path, backup)

    overrides: Dict[str, str] = {}
    if models.simple:
        overrides[_TIER_ENV_KEYS["simple"]] = models.simple
    if models.mid:
        overrides[_TIER_ENV_KEYS["mid"]] = models.mid
    if models.complex:
        overrides[_TIER_ENV_KEYS["complex"]] = models.complex
    if models.reasoning:
        overrides[_TIER_ENV_KEYS["reasoning"]] = models.reasoning

    seen = set()
    new_lines: List[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in overrides:
            new_lines.append(f"{key}={overrides[key]}")
            seen.add(key)
        else:
            new_lines.append(line)

    # Append untouched keys at the bottom under a Claude Code header.
    missing = [k for k in overrides if k not in seen]
    if missing:
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.append("# Claude Code tier mapping (nadirclaw claude onboard)")
        for k in missing:
            new_lines.append(f"{k}={overrides[k]}")

    env_path.write_text("\n".join(new_lines).rstrip() + "\n")
    if platform.system() != "Windows":
        env_path.chmod(0o600)
    return env_path


# ---------------------------------------------------------------------------
# Claude Code settings.json
# ---------------------------------------------------------------------------

def patch_claude_settings(
    base_url: str,
    api_key: str = "local",
    settings_path: Optional[Path] = None,
) -> Path:
    """Persist ANTHROPIC_BASE_URL + ANTHROPIC_API_KEY into Claude Code settings."""
    if settings_path is None:
        settings_path = CLAUDE_SETTINGS_FILE
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    config: Dict = {}
    if settings_path.exists():
        try:
            config = json.loads(settings_path.read_text())
        except (OSError, json.JSONDecodeError):
            config = {}
        backup = settings_path.with_name(
            f"settings.backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        )
        shutil.copy2(settings_path, backup)

    env = config.get("env")
    if not isinstance(env, dict):
        env = {}
    env["ANTHROPIC_BASE_URL"] = base_url
    env["ANTHROPIC_API_KEY"] = api_key
    config["env"] = env

    settings_path.write_text(json.dumps(config, indent=2) + "\n")
    return settings_path


def unpatch_claude_settings(settings_path: Optional[Path] = None) -> bool:
    """Remove NadirClaw env entries from Claude Code settings. Returns True if changed."""
    if settings_path is None:
        settings_path = CLAUDE_SETTINGS_FILE
    if not settings_path.exists():
        return False
    try:
        config = json.loads(settings_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    env = config.get("env") or {}
    if not isinstance(env, dict):
        return False
    changed = False
    for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"):
        if key in env:
            env.pop(key)
            changed = True
    if changed:
        if env:
            config["env"] = env
        else:
            config.pop("env", None)
        settings_path.write_text(json.dumps(config, indent=2) + "\n")
    return changed


# ---------------------------------------------------------------------------
# Daemon (launchd / systemd)
# ---------------------------------------------------------------------------

def _nadirclaw_binary() -> str:
    """Best-effort path to the installed nadirclaw entry point."""
    found = shutil.which("nadirclaw")
    if found:
        return found
    # Fall back to whichever python is running this module.
    return f"{sys.executable} -m nadirclaw.cli"


def _launchd_plist(port: int, log_dir: Path) -> str:
    program = _nadirclaw_binary()
    # Split so launchd's ProgramArguments has a clean argv.
    parts = program.split()
    args_xml = "\n        ".join(f"<string>{p}</string>" for p in parts)
    args_xml += "\n        <string>serve</string>"
    args_xml += f"\n        <string>--port</string>\n        <string>{port}</string>"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        {args_xml}
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir}/daemon.out.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/daemon.err.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}</string>
    </dict>
</dict>
</plist>
"""


def _systemd_unit(port: int, log_dir: Path) -> str:
    program = _nadirclaw_binary()
    return f"""[Unit]
Description=NadirClaw LLM router
After=network-online.target

[Service]
Type=simple
ExecStart={program} serve --port {port}
Restart=on-failure
RestartSec=5
StandardOutput=append:{log_dir}/daemon.out.log
StandardError=append:{log_dir}/daemon.err.log

[Install]
WantedBy=default.target
"""


def install_daemon(port: int, log_dir: Optional[Path] = None) -> Optional[Path]:
    """Install a user-scope auto-start unit. Returns the file path, or None on unsupported OS."""
    log_dir = log_dir or (CONFIG_DIR / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    system = platform.system()
    if system == "Darwin":
        LAUNCHD_PATH.parent.mkdir(parents=True, exist_ok=True)
        LAUNCHD_PATH.write_text(_launchd_plist(port, log_dir))
        # Best-effort load; ignore failures so the install still succeeds.
        try:
            subprocess.run(
                ["launchctl", "unload", str(LAUNCHD_PATH)],
                check=False, capture_output=True,
            )
            subprocess.run(
                ["launchctl", "load", str(LAUNCHD_PATH)],
                check=False, capture_output=True,
            )
        except FileNotFoundError:
            pass
        return LAUNCHD_PATH

    if system == "Linux":
        SYSTEMD_UNIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SYSTEMD_UNIT_PATH.write_text(_systemd_unit(port, log_dir))
        try:
            subprocess.run(
                ["systemctl", "--user", "daemon-reload"],
                check=False, capture_output=True,
            )
            subprocess.run(
                ["systemctl", "--user", "enable", "--now", "nadirclaw.service"],
                check=False, capture_output=True,
            )
        except FileNotFoundError:
            pass
        return SYSTEMD_UNIT_PATH

    return None


def uninstall_daemon() -> List[Path]:
    """Remove installed auto-start units. Returns the paths that were removed."""
    removed: List[Path] = []
    system = platform.system()
    if system == "Darwin" and LAUNCHD_PATH.exists():
        try:
            subprocess.run(
                ["launchctl", "unload", str(LAUNCHD_PATH)],
                check=False, capture_output=True,
            )
        except FileNotFoundError:
            pass
        LAUNCHD_PATH.unlink()
        removed.append(LAUNCHD_PATH)
    if system == "Linux" and SYSTEMD_UNIT_PATH.exists():
        try:
            subprocess.run(
                ["systemctl", "--user", "disable", "--now", "nadirclaw.service"],
                check=False, capture_output=True,
            )
        except FileNotFoundError:
            pass
        SYSTEMD_UNIT_PATH.unlink()
        removed.append(SYSTEMD_UNIT_PATH)
    return removed


# ---------------------------------------------------------------------------
# Lightweight `claude` shim
# ---------------------------------------------------------------------------

SHIM_TEMPLATE = """#!/usr/bin/env bash
# NadirClaw `claude` shim — lazy-starts the proxy then execs the real binary.
# Managed by `nadirclaw claude shim`; safe to delete.

set -euo pipefail

NADIRCLAW_PORT="${{NADIRCLAW_PORT:-{port}}}"
NADIRCLAW_BIN="{nadirclaw_bin}"
REAL_CLAUDE_HINT="{real_claude}"

# Locate the real `claude` binary by skipping this shim on PATH.
SHIM_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REAL_CLAUDE=""
IFS=':' read -ra PARTS <<< "$PATH"
for dir in "${{PARTS[@]}}"; do
    if [ "$dir" = "$SHIM_DIR" ]; then
        continue
    fi
    candidate="$dir/claude"
    if [ -x "$candidate" ] && [ "$candidate" != "${{BASH_SOURCE[0]}}" ]; then
        REAL_CLAUDE="$candidate"
        break
    fi
done

if [ -z "$REAL_CLAUDE" ] && [ -x "$REAL_CLAUDE_HINT" ]; then
    REAL_CLAUDE="$REAL_CLAUDE_HINT"
fi

if [ -z "$REAL_CLAUDE" ]; then
    echo "nadirclaw shim: could not find the real \\`claude\\` on PATH (and the" >&2
    echo "captured fallback $REAL_CLAUDE_HINT is missing). Install Claude Code first." >&2
    exit 127
fi

# Probe the proxy; start it in the background if it isn't responding.
probe() {{
    command -v curl >/dev/null 2>&1 && \\
        curl -fsS "http://localhost:${{NADIRCLAW_PORT}}/health" >/dev/null 2>&1
}}

if ! probe; then
    mkdir -p "$HOME/.nadirclaw/logs"
    nohup "$NADIRCLAW_BIN" serve --port "$NADIRCLAW_PORT" \\
        >>"$HOME/.nadirclaw/logs/shim.out.log" 2>>"$HOME/.nadirclaw/logs/shim.err.log" &
    # Give it a moment to bind.
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        sleep 0.5
        if probe; then break; fi
    done
fi

export ANTHROPIC_BASE_URL="http://localhost:${{NADIRCLAW_PORT}}/v1"
export ANTHROPIC_API_KEY="${{ANTHROPIC_API_KEY:-local}}"
exec "$REAL_CLAUDE" "$@"
"""


def _resolve_real_claude() -> Optional[str]:
    """Find an existing `claude` on PATH that is not our own shim."""
    found = shutil.which("claude")
    if not found:
        return None
    try:
        if Path(found).resolve() == SHIM_PATH.resolve():
            return None
    except OSError:
        pass
    return found


def install_shim(port: int, shim_path: Optional[Path] = None) -> Path:
    """Install the lazy-start `claude` wrapper at `shim_path`."""
    if shim_path is None:
        shim_path = SHIM_PATH
    shim_path.parent.mkdir(parents=True, exist_ok=True)
    real = _resolve_real_claude() or ""
    content = SHIM_TEMPLATE.format(
        port=port,
        nadirclaw_bin=_nadirclaw_binary().split()[0],
        real_claude=real,
    )
    shim_path.write_text(content)
    mode = shim_path.stat().st_mode
    shim_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return shim_path


def uninstall_shim(shim_path: Optional[Path] = None) -> bool:
    if shim_path is None:
        shim_path = SHIM_PATH
    if shim_path.exists():
        shim_path.unlink()
        return True
    return False
