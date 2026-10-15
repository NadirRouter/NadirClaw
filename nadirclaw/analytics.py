"""Anonymous usage telemetry for NadirClaw.

Sends a few coarse events to the maintainers' PostHog project so we can see how
many installs are active, which versions and platforms they run, and which
features get used. A random install id (``~/.nadirclaw/telemetry.json``) is the
only identifier.

Never sent: prompts, responses, API keys or tokens, file paths, hostnames,
usernames, command arguments, or custom/self-hosted model names.

Off with any of: ``NADIRCLAW_TELEMETRY=0``, ``DO_NOT_TRACK=1``, or
``nadirclaw telemetry off``. Unrelated to ``nadirclaw.telemetry``, which is
OpenTelemetry tracing to the user's own collector.
"""

import atexit
import functools
import json
import os
import platform
import sys
import threading
import urllib.request
import uuid
from pathlib import Path

from nadirclaw import __version__

# PostHog project keys are write-only and meant to be embedded in clients.
_KEY = "phc_OK7tEfer6lcaiok8bcsssUGcqi4Sapsx57wOl6VsI4x"
_URL = "https://us.i.posthog.com/i/v0/e/"
_STATE = Path.home() / ".nadirclaw" / "telemetry.json"
_HEARTBEAT_S = 6 * 3600
# Model ids from these providers are public catalog names, safe to report.
# Everything else (ollama, together_ai, replicate, ...) can be a private name.
_PUBLIC_PROVIDERS = {"anthropic", "openai", "openai-codex", "google"}

NOTICE = (
    "NadirClaw collects anonymous usage stats (command names, version, OS, "
    "request counts; never prompts, responses or keys).\n"
    "Opt out: `nadirclaw telemetry off` or NADIRCLAW_TELEMETRY=0. "
    "Details: https://github.com/NadirRouter/NadirClaw#telemetry"
)

_pending: list[threading.Thread] = []
_stop = threading.Event()
_last_usage: dict = {}


def _safe(fn):
    """Telemetry must never break or slow the tool it rides on."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None
    return wrapper


def _load() -> dict:
    try:
        return json.loads(_STATE.read_text())
    except (OSError, ValueError):
        return {}


def _save(state: dict) -> None:
    try:
        _STATE.parent.mkdir(parents=True, exist_ok=True)
        _STATE.write_text(json.dumps(state))
    except OSError:
        pass


def enabled() -> bool:
    if os.getenv("NADIRCLAW_TELEMETRY", "").strip().lower() in ("0", "false", "off", "no"):
        return False
    if os.getenv("DO_NOT_TRACK", "").strip().lower() not in ("", "0", "false"):
        return False
    return _load().get("enabled", True)


def set_enabled(on: bool) -> None:
    state = _load()
    state["enabled"] = on
    _save(state)


def _install_id() -> str:
    state = _load()
    if not state.get("id"):
        # ponytail: an unwritable HOME (some containers) mints a new id per run
        # and inflates user counts; hash /etc/machine-id if that shows up in data.
        state["id"] = str(uuid.uuid4())
        _save(state)
        print(NOTICE, file=sys.stderr)
    return state["id"]


def _post(body: bytes) -> None:
    req = urllib.request.Request(_URL, body, {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=3).close()
    except Exception:
        pass


@_safe
def capture(event: str, props: dict | None = None, wait: bool = False) -> None:
    if not enabled():
        return
    env = {
        "version": __version__,
        "python": platform.python_version(),
        "os": platform.system(),
        "arch": platform.machine(),
        "ci": bool(os.getenv("CI")),
        "docker": Path("/.dockerenv").exists(),
    }
    body = json.dumps({
        "api_key": _KEY,
        "event": event,
        "distinct_id": _install_id(),
        "properties": {**env, **(props or {}), "$set": env},
    }).encode()
    t = threading.Thread(target=_post, args=(body,), daemon=True)
    t.start()
    if wait:
        t.join(3)
    else:
        _pending.append(t)


@atexit.register
def _flush() -> None:
    # Short CLI commands exit before the POST finishes; give it a moment.
    for t in _pending:
        t.join(1)


@_safe
def cli_command(group, argv: list[str]) -> None:
    """Record which subcommand ran. Only registered command names are sent."""
    names = []
    cmd = group
    for arg in argv:
        sub = getattr(cmd, "commands", {}).get(arg)
        if sub is None:
            if arg.startswith("-"):
                continue
            break
        names.append(arg)
        cmd = sub
    name = " ".join(names)
    # read-hook runs on every Claude Code file read; telemetry is the opt-out itself.
    if not name or name == "claude read-hook" or names[0] == "telemetry":
        return
    capture("nadirclaw_cli", {"command": name})


def _usage() -> dict:
    from nadirclaw import metrics

    out: dict = {}

    def add(key, n):
        out[key] = out.get(key, 0) + n

    for (_model, tier, status), n in metrics.requests_total.items():
        add("requests", n)
        add(f"requests_{tier}", n)
        if status != "ok":
            add("errors", n)
    for name, counter in (
        ("prompt_tokens", metrics.tokens_prompt_total),
        ("completion_tokens", metrics.tokens_completion_total),
        ("cost_usd", metrics.cost_total),
        ("cache_hits", metrics.cache_hits_total),
        ("fallbacks", metrics.fallbacks_total),
        ("tokens_saved", metrics.tokens_saved_total),
    ):
        for _labels, n in counter.items():
            add(name, n)
    return out


def _send_usage(wait: bool = False) -> None:
    global _last_usage
    now = _usage()
    delta = {k: round(v - _last_usage.get(k, 0), 6) for k, v in now.items()}
    _last_usage = now
    if delta.get("requests"):
        capture("nadirclaw_usage", delta, wait=wait)


@_safe
def server_started(settings, credential_sources: dict) -> None:
    if not enabled():
        return
    from nadirclaw.credentials import detect_provider

    models = settings.tier_models
    providers = {m: detect_provider(m) or "other" for m in models}
    capture("nadirclaw_serve", {
        "tiers": len(models),
        "providers": sorted(set(providers.values())),
        "models": [m for m, p in providers.items() if p in _PUBLIC_PROVIDERS and "ft:" not in m],
        "credential_sources": credential_sources,
        "classifier": settings.COMPLEXITY_ANALYZER_TYPE,
        "optimize": settings.OPTIMIZE,
        "auth": bool(settings.AUTH_TOKEN),
        "pii_redaction": settings.PII_REDACTION_MODE,
        "prompt_guard": settings.PROMPT_GUARD_ACTION,
    })

    def beat():
        while not _stop.wait(_HEARTBEAT_S):
            _send_usage()

    threading.Thread(target=beat, daemon=True, name="nadirclaw-telemetry").start()


@_safe
def server_stopped() -> None:
    _stop.set()
    if enabled():
        _send_usage(wait=True)
