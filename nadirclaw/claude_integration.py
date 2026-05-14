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
from typing import Callable, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

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
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
]


_ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
_LIVE_FETCH_TIMEOUT = 5.0


def _auth_headers_for_token(token: str) -> List[Dict[str, str]]:
    """Return header sets to try for /v1/models, in order.

    Anthropic accepts two auth styles:
      - `x-api-key: <key>` for regular API keys (`sk-ant-api*`)
      - `Authorization: Bearer <token>` for subscription OAuth tokens
        (`sk-ant-oat*`, minted by `claude setup-token`)
    """
    common = {"anthropic-version": "2023-06-01"}
    bearer = {"Authorization": f"Bearer {token}", **common}
    api_key = {"x-api-key": token, **common}
    if token.startswith("sk-ant-oat"):
        return [bearer, api_key]
    if token.startswith("sk-ant-api"):
        return [api_key, bearer]
    return [bearer, api_key]


def _fetch_anthropic_models(token: Optional[str] = None) -> List[str]:
    """Query Anthropic's /v1/models with the stored credential.

    Returns a list of model IDs, or an empty list on any failure
    (no creds, network error, non-200, bad JSON). Handles both regular
    API keys and subscription OAuth tokens.
    """
    if not token:
        try:
            from nadirclaw.credentials import get_credential

            token = get_credential("anthropic")
        except Exception:
            token = None
    if not token:
        return []

    payload = None
    for headers in _auth_headers_for_token(token):
        req = urllib_request.Request(_ANTHROPIC_MODELS_URL, headers=headers)
        try:
            with urllib_request.urlopen(req, timeout=_LIVE_FETCH_TIMEOUT) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                break
        except urllib_error.HTTPError as e:
            if e.code in (401, 403):
                continue  # try the next auth style
            return []
        except (urllib_error.URLError, TimeoutError, ValueError, OSError):
            return []

    if payload is None:
        return []

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    ids: List[str] = []
    for entry in data:
        if isinstance(entry, dict):
            mid = entry.get("id")
            if isinstance(mid, str) and mid.strip():
                ids.append(mid.strip())
    return ids


@dataclass
class CandidatePool:
    """Pool of model IDs available for tier selection, with source attribution."""

    models: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    source_of: Dict[str, str] = field(default_factory=dict)


def gather_candidate_models(
    claude_settings: Optional[Dict] = None,
    claude_json: Optional[Dict] = None,
    fetch_live_models: Optional[Callable[[], List[str]]] = None,
) -> CandidatePool:
    """Collect every model ID we can offer the user, with where it came from.

    Pool order (deduped, preserves order of first appearance):
      1. Models named in `~/.claude/settings.json` / `~/.claude.json`.
      2. Live `/v1/models` lookup against Anthropic.
      3. Hardcoded Claude 4.6 family defaults.
    """
    if claude_settings is None:
        claude_settings = _read_json(CLAUDE_SETTINGS_FILE)
    if claude_json is None:
        claude_json = _read_json(CLAUDE_JSON_FILE)

    pool = CandidatePool()
    seen: set = set()

    def add(model: str, source: str) -> None:
        if not model or model in seen:
            return
        seen.add(model)
        pool.models.append(model)
        pool.source_of[model] = source

    settings_models = _candidate_models(claude_settings, {})
    if settings_models:
        pool.sources.append(str(CLAUDE_SETTINGS_FILE))
        for m in settings_models:
            add(m, str(CLAUDE_SETTINGS_FILE))

    claude_json_models = _candidate_models({}, claude_json)
    if claude_json_models:
        pool.sources.append(str(CLAUDE_JSON_FILE))
        for m in claude_json_models:
            add(m, str(CLAUDE_JSON_FILE))

    fetcher = fetch_live_models or _fetch_anthropic_models
    try:
        live = fetcher() or []
    except Exception:
        live = []
    if live:
        pool.sources.append("anthropic api")
        for m in live:
            add(m, "anthropic api")

    if not pool.models:
        pool.sources.append("defaults")
        for m in _CLAUDE_FAMILY_FALLBACK:
            add(m, "defaults")

    return pool


def detect_models(
    claude_settings: Optional[Dict] = None,
    claude_json: Optional[Dict] = None,
    fetch_live_models: Optional[Callable[[], List[str]]] = None,
) -> DetectedModels:
    """Bucket detected Claude Code models into NadirClaw tiers.

    Resolution order:
      1. Models named in `~/.claude/settings.json` / `~/.claude.json`.
      2. Live `/v1/models` lookup against Anthropic using the stored
         subscription token or API key (subscription users have no
         model IDs in local files, so this is the common path).
      3. Hardcoded Claude 4.6 family defaults.
    """
    if claude_settings is None:
        claude_settings = _read_json(CLAUDE_SETTINGS_FILE)
    if claude_json is None:
        claude_json = _read_json(CLAUDE_JSON_FILE)

    found = _candidate_models(claude_settings, claude_json)
    sources: List[str] = []
    if found:
        if CLAUDE_SETTINGS_FILE.exists() and _candidate_models(claude_settings, {}):
            sources.append(str(CLAUDE_SETTINGS_FILE))
        if CLAUDE_JSON_FILE.exists() and _candidate_models({}, claude_json):
            sources.append(str(CLAUDE_JSON_FILE))

    if not found:
        fetcher = fetch_live_models or _fetch_anthropic_models
        try:
            live = fetcher() or []
        except Exception:
            live = []
        if live:
            found = live
            sources.append("anthropic api")

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


_TIER_HINTS = {
    "simple": "short questions, formatting, quick reads → cheapest haiku/flash tier",
    "mid": "focused edits, single-function debugging → optional mid tier",
    "complex": "architecture, multi-file refactors, agentic loops → sonnet/opus tier",
    "reasoning": "heavy thinking, long-horizon planning → strongest available model",
}

_TIER_ORDER = ("simple", "mid", "complex", "reasoning")


def interactive_pick_models(
    pool: CandidatePool,
    defaults: Optional[DetectedModels] = None,
    prompt_fn: Optional[Callable[..., str]] = None,
    echo_fn: Optional[Callable[[str], None]] = None,
) -> DetectedModels:
    """Walk the user through picking a model for each routing tier.

    Each tier gets a numbered menu of `pool.models`. The user can:
      - press Enter to accept the suggested default
      - type a number from the list
      - type a model ID not in the list (free-form)
      - type 'skip' / '-' / '' (when no default) to leave the tier empty

    `prompt_fn` and `echo_fn` are injectable so this can be unit-tested
    without spawning a real terminal.
    """
    import click  # local import to keep module importable in headless contexts

    if prompt_fn is None:
        prompt_fn = click.prompt
    if echo_fn is None:
        echo_fn = click.echo

    defaults = defaults or DetectedModels()
    selected = DetectedModels(sources=list(pool.sources))

    echo_fn("\nAvailable models:")
    for idx, model in enumerate(pool.models, 1):
        src = pool.source_of.get(model, "")
        echo_fn(f"  [{idx}] {model}{f'  ({src})' if src else ''}")
    echo_fn("Press Enter to accept the default, type a number, paste a model ID, or type 'skip'.\n")

    for tier in _TIER_ORDER:
        suggestion = getattr(defaults, tier)
        suggestion_label = suggestion or "skip"
        hint = _TIER_HINTS.get(tier, "")
        echo_fn(f"  {tier} — {hint}")
        raw = prompt_fn(
            f"  pick model for '{tier}'",
            default=suggestion_label,
            show_default=True,
        )
        raw = (raw or "").strip()
        chosen: Optional[str] = None
        if raw.lower() in ("skip", "-", "none"):
            chosen = None
        elif raw.isdigit():
            i = int(raw)
            if 1 <= i <= len(pool.models):
                chosen = pool.models[i - 1]
            else:
                echo_fn(f"  ! out of range (1..{len(pool.models)}), leaving '{tier}' unset")
                chosen = None
        elif raw:
            chosen = raw
        setattr(selected, tier, chosen)
        echo_fn(f"    → {tier}: {chosen or '—'}\n")

    if not selected.reasoning and selected.complex:
        selected.reasoning = selected.complex
    if not selected.complex and selected.simple:
        selected.complex = selected.simple
    if not selected.simple and selected.complex:
        selected.simple = selected.complex

    return selected


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

DEFAULT_PROFILES = ("nadir-auto", "nadir-eco", "nadir-premium", "nadir-reasoning", "nadir-free")


def patch_claude_settings(
    base_url: str,
    api_key: str = "local",
    default_profile: Optional[str] = None,
    settings_path: Optional[Path] = None,
) -> Path:
    """Persist ANTHROPIC_BASE_URL + ANTHROPIC_API_KEY (and optionally
    ANTHROPIC_MODEL) into Claude Code settings.

    `default_profile` controls what model Claude Code sends on every
    request — pick `nadir-auto` for smart routing, `nadir-eco`/`-premium`
    to pin a tier, or pass None to leave the existing value alone.
    """
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
    if default_profile:
        env["ANTHROPIC_MODEL"] = default_profile
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
    for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"):
        if key in env:
            # Only remove ANTHROPIC_MODEL if it points at one of our profiles.
            if key == "ANTHROPIC_MODEL" and env[key] not in DEFAULT_PROFILES:
                continue
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

export ANTHROPIC_BASE_URL="http://localhost:${{NADIRCLAW_PORT}}"
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
