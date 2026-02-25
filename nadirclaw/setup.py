"""Interactive setup wizard for NadirClaw.

Guides users through provider selection, credential entry, and model
configuration on first run or via `nadirclaw setup`.
"""

import json
import os
import re
import shutil
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import click

from nadirclaw.routing import MODEL_REGISTRY

# ---------------------------------------------------------------------------
# Provider metadata
# ---------------------------------------------------------------------------

PROVIDER_INFO: Dict[str, Dict] = {
    "openai": {
        "display": "OpenAI",
        "description": "GPT-4.1, GPT-5, o3, o4-mini",
        "env_var": "OPENAI_API_KEY",
        "key_prefix": "sk-",
        "oauth": True,
        "credential_key": "openai",
    },
    "anthropic": {
        "display": "Anthropic",
        "description": "Claude Opus 4.6, Sonnet 4.5, Haiku 4.5",
        "env_var": "ANTHROPIC_API_KEY",
        "key_prefix": "sk-ant-",
        "oauth": True,
        "credential_key": "anthropic",
    },
    "google": {
        "display": "Google / Gemini",
        "description": "Gemini 2.5 Flash, Pro, 3 Flash",
        "env_var": "GEMINI_API_KEY",
        "key_prefix": "AIza",
        "oauth": True,
        "credential_key": "google",
    },
    "deepseek": {
        "display": "DeepSeek",
        "description": "DeepSeek V3, Reasoner",
        "env_var": "DEEPSEEK_API_KEY",
        "key_prefix": "sk-",
        "oauth": False,
        "credential_key": "deepseek",
    },
    "ollama": {
        "display": "Ollama (local)",
        "description": "Llama, Qwen — no API key",
        "env_var": None,
        "key_prefix": None,
        "oauth": False,
        "credential_key": "ollama",
    },
}

PROVIDER_ORDER = ["openai", "anthropic", "google", "deepseek", "ollama"]

OLLAMA_DEFAULT_API_BASE = "http://localhost:11434"

# Tier defaults — ordered preference per provider
_TIER_DEFAULTS = {
    "simple": {
        "google": "gemini-2.5-flash",
        "openai": "gpt-4.1-mini",
        "deepseek": "deepseek/deepseek-chat",
        "ollama": "ollama/llama3.1:8b",
        "anthropic": "claude-haiku-4-5-20251001",
    },
    "complex": {
        "anthropic": "claude-sonnet-4-5-20250929",
        "openai": "gpt-4.1",
        "google": "gemini-2.5-pro",
        "deepseek": "deepseek/deepseek-reasoner",
        "ollama": "ollama/qwen3:32b",
    },
    "reasoning": {
        "openai": "o3",
        "deepseek": "deepseek/deepseek-reasoner",
        "anthropic": "claude-sonnet-4-5-20250929",
        "google": "gemini-2.5-pro",
    },
    "free": {
        "ollama": "ollama/llama3.1:8b",
    },
}

# Config directory
CONFIG_DIR = Path.home() / ".nadirclaw"
ENV_FILE = CONFIG_DIR / ".env"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_ollama_api_base(raw: str) -> str:
    """Normalize an Ollama API base URL.

    Strips whitespace, defaults to localhost:11434, prepends http:// if no
    scheme is present, and strips any trailing slash.
    """
    raw = raw.strip()
    if not raw:
        return OLLAMA_DEFAULT_API_BASE
    if not raw.startswith("http://") and not raw.startswith("https://"):
        raw = "http://" + raw
    return raw.rstrip("/")


def _check_ollama_connectivity_with_base(api_base: str) -> bool:
    """Check if Ollama is reachable at the given base URL."""
    api_base = _normalize_ollama_api_base(api_base)
    try:
        req = urllib.request.Request(f"{api_base}/api/tags")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def is_first_run() -> bool:
    """Check if NadirClaw has been configured (i.e. .env exists)."""
    return not ENV_FILE.exists()


def detect_existing_config() -> Dict[str, str]:
    """Read existing .env file and return key-value pairs."""
    config: Dict[str, str] = {}
    if not ENV_FILE.exists():
        return config
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip()
    return config


def detect_existing_credentials() -> List[str]:
    """Return list of providers that already have credentials configured."""
    from nadirclaw.credentials import get_credential

    found = []
    for provider_key, info in PROVIDER_INFO.items():
        if provider_key == "ollama":
            continue
        cred_key = info["credential_key"]
        if get_credential(cred_key):
            found.append(provider_key)
        elif info["env_var"] and os.getenv(info["env_var"], ""):
            found.append(provider_key)
    return found


# ---------------------------------------------------------------------------
# API model fetching
# ---------------------------------------------------------------------------

def _fetch_openai_models(credential: str) -> List[str]:
    """Fetch available chat models from the OpenAI API."""
    req = urllib.request.Request(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {credential}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception:
        return []

    models = []
    for m in data.get("data", []):
        mid = m.get("id", "")
        # Only chat/completion models
        if not any(mid.startswith(p) for p in ("gpt-", "o1", "o3", "o4", "chatgpt-")):
            continue
        # Exclude non-chat variants
        if any(x in mid for x in ("-audio", "-realtime", "-search", "dall-e", "whisper", "tts", "embedding")):
            continue
        models.append(mid)
    return sorted(models)


def _fetch_anthropic_models(credential: str) -> List[str]:
    """Fetch all available models from the Anthropic API (handles pagination)."""
    models = []
    base_url = "https://api.anthropic.com/v1/models"
    headers = {
        "x-api-key": credential,
        "anthropic-version": "2023-06-01",
    }
    url = f"{base_url}?limit=1000"

    while url:
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
        except Exception:
            break

        for m in data.get("data", []):
            mid = m.get("id", "")
            if mid.startswith("claude-"):
                models.append(mid)

        # Follow pagination if there are more results
        if data.get("has_more") and data.get("last_id"):
            url = f"{base_url}?limit=1000&after_id={data['last_id']}"
        else:
            url = None

    return sorted(models)


def _fetch_google_models(credential: str) -> List[str]:
    """Fetch available Gemini models from the Google GenAI API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={credential}&pageSize=1000"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception:
        return []

    models = []
    for m in data.get("models", []):
        name = m.get("name", "")  # e.g. "models/gemini-2.5-flash"
        # Strip "models/" prefix
        if name.startswith("models/"):
            name = name[len("models/"):]
        # Only gemini models that support generateContent
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" in methods and "gemini" in name:
            models.append(name)
    return sorted(models)


def _fetch_deepseek_models(credential: str) -> List[str]:
    """Fetch available models from the DeepSeek API."""
    req = urllib.request.Request(
        "https://api.deepseek.com/models",
        headers={"Authorization": f"Bearer {credential}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception:
        return []

    models = []
    for m in data.get("data", []):
        mid = m.get("id", "")
        if mid:
            models.append(f"deepseek/{mid}")
    return sorted(models)


def _fetch_ollama_models(api_base: Optional[str] = None) -> List[str]:
    """Fetch locally installed models from Ollama."""
    base = _normalize_ollama_api_base(api_base or "")
    req = urllib.request.Request(f"{base}/api/tags")
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
    except Exception:
        return []

    models = []
    for m in data.get("models", []):
        name = m.get("name", "")
        if name:
            models.append(f"ollama/{name}")
    return sorted(models)


_DATE_SUFFIX_RE = re.compile(r"-\d{4}-?\d{2}-?\d{2}$")


def _filter_top_models(provider: str, models: List[str]) -> List[str]:
    """Keep only current-generation top models per provider."""
    if provider == "anthropic":
        return _filter_anthropic_top(models)
    if provider == "openai":
        return _filter_openai_top(models)
    if provider == "google":
        return _filter_google_top(models)
    return models  # deepseek, ollama: show all


def _filter_anthropic_top(models: List[str]) -> List[str]:
    """Keep only the latest version of each Claude family (opus/sonnet/haiku)."""
    families: Dict[str, List[tuple]] = {}  # family -> [(model_id, date)]
    for m in models:
        family = None
        for name in ("opus", "sonnet", "haiku"):
            if name in m:
                family = name
                break
        if not family:
            continue
        # Extract date suffix (YYYYMMDD)
        parts = m.split("-")
        date = parts[-1] if parts[-1].isdigit() and len(parts[-1]) == 8 else "0"
        families.setdefault(family, []).append((m, date))

    top = []
    for variants in families.values():
        variants.sort(key=lambda x: x[1], reverse=True)
        top.append(variants[0][0])  # latest version
    return sorted(top)


def _filter_openai_top(models: List[str]) -> List[str]:
    """Remove dated variants and old-generation OpenAI models."""
    old_gen = ("gpt-3.5", "gpt-4-", "gpt-4o", "chatgpt-4o", "ft:")
    top = []
    for m in models:
        if _DATE_SUFFIX_RE.search(m):
            continue
        if any(m.startswith(p) for p in old_gen):
            continue
        top.append(m)
    return sorted(top)


def _filter_google_top(models: List[str]) -> List[str]:
    """Keep only current-generation Gemini models (2.5+)."""
    current_gen = ("gemini-2.5-", "gemini-3-")
    top = []
    for m in models:
        if any(m.startswith(p) for p in current_gen):
            top.append(m)
    return sorted(top)


def fetch_provider_models(
    provider: str, credential: str, ollama_api_base: Optional[str] = None,
) -> List[str]:
    """Fetch available model IDs from a provider's API.

    Returns only top current-generation models, or empty list on failure.
    """
    fetchers = {
        "openai": _fetch_openai_models,
        "anthropic": _fetch_anthropic_models,
        "google": _fetch_google_models,
        "deepseek": _fetch_deepseek_models,
    }
    if provider == "ollama":
        return _fetch_ollama_models(api_base=ollama_api_base)
    fetcher = fetchers.get(provider)
    if fetcher and credential:
        raw = fetcher(credential)
        return _filter_top_models(provider, raw)
    return []


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

def classify_model_tier(model_id: str) -> str:
    """Classify a model into a routing tier based on its name.

    Returns one of: 'simple', 'complex', 'reasoning', 'free'.
    """
    lower = model_id.lower()

    # Free — ollama / local models
    if lower.startswith("ollama/") or lower.startswith("local/"):
        return "free"

    # Reasoning — o-series, reasoner
    if any(lower.startswith(p) for p in ("o1-", "o1", "o3-", "o3", "o4-", "o4")):
        return "reasoning"
    if "reasoner" in lower:
        return "reasoning"

    # Simple — mini (but not gemini), nano, flash, haiku, lite, small
    if any(x in lower for x in ("-mini", "nano", "flash", "haiku", "lite", "small")):
        return "simple"

    # Complex — everything else (pro, opus, sonnet, gpt-4.1, gpt-5, etc.)
    return "complex"


# ---------------------------------------------------------------------------
# Step 1: Welcome
# ---------------------------------------------------------------------------

def print_welcome():
    """Print welcome banner."""
    click.echo()
    click.echo("=" * 56)
    click.echo("  NadirClaw Setup Wizard")
    click.echo("=" * 56)
    click.echo()
    click.echo("This wizard will help you configure NadirClaw by:")
    click.echo("  1. Selecting your LLM providers")
    click.echo("  2. Entering API keys or logging in via OAuth")
    click.echo("  3. Choosing models for each routing tier")
    click.echo()
    click.echo("Your configuration will be saved to ~/.nadirclaw/.env")
    click.echo()


# ---------------------------------------------------------------------------
# Step 2: Provider selection
# ---------------------------------------------------------------------------

def prompt_provider_selection(existing: Optional[List[str]] = None) -> List[str]:
    """Multi-select providers via numbered menu."""
    click.echo("Which LLM providers do you want to use?")
    click.echo()

    for i, key in enumerate(PROVIDER_ORDER, 1):
        info = PROVIDER_INFO[key]
        marker = " *" if existing and key in existing else ""
        click.echo(f"  {i}. {info['display']:20s} ({info['description']}){marker}")

    if existing:
        click.echo()
        click.echo("  * = credential already configured")

    click.echo()
    raw = click.prompt(
        "Select [comma-separated, e.g. 1,3,5]",
        default="1,3" if not existing else None,
    )

    selected = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(PROVIDER_ORDER):
                selected.append(PROVIDER_ORDER[idx])
    if not selected:
        click.echo("No valid selections. Defaulting to Google/Gemini.")
        selected = ["google"]

    names = ", ".join(PROVIDER_INFO[p]["display"] for p in selected)
    click.echo(f"\nSelected: {names}\n")
    return selected


# ---------------------------------------------------------------------------
# Step 3: Credential collection
# ---------------------------------------------------------------------------

def _check_ollama_connectivity() -> bool:
    """Check if Ollama is running at localhost:11434."""
    return _check_ollama_connectivity_with_base(OLLAMA_DEFAULT_API_BASE)


def prompt_credential_for_provider(
    provider: str, reconfigure: bool = False, ollama_api_base: Optional[str] = None,
) -> Optional[str]:
    """Prompt user for credentials for a single provider.

    Returns the credential string, or None if skipped.
    """
    from nadirclaw.credentials import get_credential, save_credential

    info = PROVIDER_INFO[provider]

    # Ollama needs no key
    if provider == "ollama":
        base = _normalize_ollama_api_base(ollama_api_base or "")
        click.echo(f"  {info['display']}: Checking connectivity...")
        if _check_ollama_connectivity_with_base(base):
            click.echo(f"    Ollama is running at {base}")
        else:
            click.echo(f"    Ollama not detected at {base}")
            click.echo("    Make sure Ollama is running before using local models.")
        click.echo()
        return "local"

    # Check existing credential
    cred_key = info["credential_key"]
    existing = get_credential(cred_key)
    if existing and not reconfigure:
        masked = existing[:8] + "..." + existing[-4:] if len(existing) > 12 else existing[:4] + "***"
        click.echo(f"  {info['display']}: credential already configured ({masked})")
        if not click.confirm("    Update this credential?", default=False):
            click.echo()
            return existing

    click.echo(f"\n  {info['display']}:")

    if info["oauth"]:
        click.echo("    1. Enter API key")
        click.echo("    2. Login via OAuth (opens browser)")
        choice = click.prompt("    Choose", type=click.Choice(["1", "2"]), default="1")
    else:
        choice = "1"

    if choice == "1":
        key = click.prompt(f"    {info['display']} API key", hide_input=True)
        key = key.strip()
        if not key:
            click.echo("    Skipped (empty key).")
            click.echo()
            return None
        save_credential(cred_key, key, source="manual")
        click.echo(f"    Saved!")
        click.echo()
        return key
    else:
        # OAuth flow
        return _run_oauth_for_provider(provider)


def _run_oauth_for_provider(provider: str) -> Optional[str]:
    """Run the OAuth flow for a provider. Returns access token or None."""
    from nadirclaw.credentials import save_oauth_credential

    try:
        if provider == "openai":
            from nadirclaw.oauth import login_openai
            token_data = login_openai(timeout=300)
            if token_data and token_data.get("access_token"):
                import time
                expires_in = max(int(token_data.get("expires_at", 0) - time.time()), 3600)
                save_oauth_credential(
                    "openai-codex",
                    token_data["access_token"],
                    token_data.get("refresh_token", ""),
                    expires_in,
                )
                click.echo("    OpenAI OAuth login successful!")
                click.echo()
                return token_data["access_token"]
        elif provider == "anthropic":
            from nadirclaw.oauth import validate_anthropic_setup_token
            click.echo("    Paste your Anthropic setup token (from `claude setup-token`):")
            token = click.prompt("    Token", hide_input=True).strip()
            error = validate_anthropic_setup_token(token)
            if error:
                click.echo(f"    Invalid token: {error}")
                click.echo()
                return None
            from nadirclaw.credentials import save_credential
            save_credential("anthropic", token, source="setup-token")
            click.echo("    Anthropic token saved!")
            click.echo()
            return token
        elif provider == "google":
            from nadirclaw.oauth import login_gemini
            token_data = login_gemini(timeout=300)
            if token_data and token_data.get("access_token"):
                import time
                expires_in = max(int(token_data.get("expires_at", 0) - time.time()), 3600)
                save_oauth_credential(
                    "gemini",
                    token_data["access_token"],
                    token_data.get("refresh_token", ""),
                    expires_in,
                    metadata={
                        "project_id": token_data.get("project_id", ""),
                        "email": token_data.get("email", ""),
                    },
                )
                click.echo("    Gemini OAuth login successful!")
                click.echo()
                return token_data["access_token"]
    except Exception as e:
        click.echo(f"    OAuth failed: {e}")
        click.echo("    You can try again later with `nadirclaw auth <provider> login`.")
        click.echo()
        return None

    click.echo()
    return None


# ---------------------------------------------------------------------------
# Step 4: Model selection
# ---------------------------------------------------------------------------

def get_available_models_for_providers(
    providers: List[str],
    fetched_models: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[dict]]:
    """Build tier-grouped model lists from API-fetched models (with static fallback).

    Args:
        providers: List of provider keys the user selected.
        fetched_models: Optional dict of {provider: [model_ids]} from API calls.
            When provided, these are used as the primary source.
            Falls back to MODEL_REGISTRY for providers with no fetched models.

    Returns dict with keys: simple, complex, reasoning, free.
    Each value is a list of dicts: {model, provider}.
    """
    all_models: List[dict] = []
    providers_covered = set()

    # Use API-fetched models when available
    if fetched_models:
        for provider, model_ids in fetched_models.items():
            if model_ids:
                providers_covered.add(provider)
                for mid in model_ids:
                    all_models.append({"model": mid, "provider": provider})

    # Fall back to MODEL_REGISTRY for providers without fetched models
    skip_prefixed = {m for m in MODEL_REGISTRY if m.startswith("gemini/")}
    for model, info in MODEL_REGISTRY.items():
        if model in skip_prefixed:
            continue
        # Detect provider from model name
        model_provider = _detect_model_provider(model)
        if model_provider and model_provider in providers and model_provider not in providers_covered:
            all_models.append({"model": model, "provider": model_provider})

    # Deduplicate by model name
    seen = set()
    unique = []
    for m in all_models:
        if m["model"] not in seen:
            seen.add(m["model"])
            unique.append(m)
    all_models = unique

    # Classify into tiers
    tiers: Dict[str, List[dict]] = {
        "simple": [],
        "complex": [],
        "reasoning": [],
        "free": [],
    }

    for m in all_models:
        tier = classify_model_tier(m["model"])
        tiers[tier].append(m)

    # Sort each tier alphabetically
    for tier in tiers.values():
        tier.sort(key=lambda x: x["model"])

    return tiers


def _detect_model_provider(model: str) -> Optional[str]:
    """Detect provider key from a model name (for static registry fallback)."""
    lower = model.lower()
    if lower.startswith("gemini") or lower.startswith("google/"):
        return "google"
    if lower.startswith("gpt-") or lower.startswith("o3") or lower.startswith("o4") or lower.startswith("o1") or lower.startswith("openai"):
        return "openai"
    if lower.startswith("claude-") or lower.startswith("anthropic/"):
        return "anthropic"
    if lower.startswith("deepseek/"):
        return "deepseek"
    if lower.startswith("ollama/"):
        return "ollama"
    return None


def format_model_table(models: List[dict], tier: str) -> str:
    """Format a model selection table for display."""
    tier_labels = {
        "simple": "Simple model (cheap/fast)",
        "complex": "Complex model (premium)",
        "reasoning": "Reasoning model (chain-of-thought)",
        "free": "Free model (zero cost)",
    }
    lines = [f"\n{tier_labels.get(tier, tier)}:"]

    for i, m in enumerate(models, 1):
        lines.append(f"  {i}. {m['model']}")

    return "\n".join(lines)


def select_default_model(tier: str, providers: List[str], available: Optional[List[dict]] = None) -> Optional[str]:
    """Pick the best default model for a tier based on configured providers.

    If `available` is provided, only returns a default that appears in the list.
    """
    tier_prefs = _TIER_DEFAULTS.get(tier, {})
    available_names = {m["model"] for m in available} if available else None

    for provider in tier_prefs:
        if provider in providers:
            model = tier_prefs[provider]
            if available_names is not None and model not in available_names:
                continue
            return model
    return None


def prompt_model_selection(tier: str, models: List[dict], providers: List[str]) -> Optional[str]:
    """Show model table and prompt for selection. Returns model name or None."""
    if not models:
        return None

    table = format_model_table(models, tier)
    click.echo(table)

    default_model = select_default_model(tier, providers, available=models)
    default_idx = "1"
    for i, m in enumerate(models, 1):
        if m["model"] == default_model:
            default_idx = str(i)
            break

    is_optional = tier in ("reasoning", "free")
    prompt_text = f"Select [1-{len(models)}]"
    if is_optional:
        prompt_text += " or 's' to skip"

    raw = click.prompt(prompt_text, default=default_idx)
    raw = raw.strip().lower()

    if is_optional and raw in ("s", "skip", ""):
        click.echo(f"  Skipped {tier} tier.\n")
        return None

    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(models):
            chosen = models[idx]["model"]
            click.echo(f"  Selected: {chosen}\n")
            return chosen

    # Fallback to first
    chosen = models[0]["model"]
    click.echo(f"  Selected: {chosen}\n")
    return chosen


# ---------------------------------------------------------------------------
# Step 5: Write config + summary
# ---------------------------------------------------------------------------

def write_env_file(
    simple: str,
    complex_model: str,
    reasoning: Optional[str] = None,
    free: Optional[str] = None,
    api_keys: Optional[Dict[str, str]] = None,
    ollama_api_base: Optional[str] = None,
) -> Path:
    """Write ~/.nadirclaw/.env with model configuration.

    Creates backup of existing .env if present. Sets 0o600 permissions.
    Returns path to written file.
    """
    import platform

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Backup existing .env
    if ENV_FILE.exists():
        backup_name = f".env.backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        backup_path = CONFIG_DIR / backup_name
        shutil.copy2(ENV_FILE, backup_path)
        click.echo(f"  Backed up existing config to {backup_path}")

    lines = [
        "# NadirClaw configuration",
        f"# Generated by `nadirclaw setup` on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # API keys
    if api_keys:
        lines.append("# API Keys")
        for env_var, value in sorted(api_keys.items()):
            lines.append(f"{env_var}={value}")
        lines.append("")

    # Model routing
    lines.append("# Model Routing")
    lines.append(f"NADIRCLAW_SIMPLE_MODEL={simple}")
    lines.append(f"NADIRCLAW_COMPLEX_MODEL={complex_model}")
    if reasoning:
        lines.append(f"NADIRCLAW_REASONING_MODEL={reasoning}")
    if free:
        lines.append(f"NADIRCLAW_FREE_MODEL={free}")
    lines.append("")

    # Ollama
    if ollama_api_base:
        lines.append("# Ollama")
        lines.append(f"OLLAMA_API_BASE={ollama_api_base}")
        lines.append("")

    # Server defaults
    lines.append("# Server")
    lines.append("NADIRCLAW_PORT=8856")
    lines.append("")

    ENV_FILE.write_text("\n".join(lines) + "\n")

    # Restrict permissions
    if platform.system() != "Windows":
        ENV_FILE.chmod(0o600)

    return ENV_FILE


def print_summary(
    providers: List[str],
    simple: str,
    complex_model: str,
    reasoning: Optional[str],
    free: Optional[str],
):
    """Print configuration summary and next steps."""
    click.echo()
    click.echo("=" * 56)
    click.echo("  Setup Complete!")
    click.echo("=" * 56)
    click.echo()
    click.echo("  Configuration:")
    click.echo(f"    Providers:     {', '.join(PROVIDER_INFO[p]['display'] for p in providers)}")
    click.echo(f"    Simple model:  {simple}")
    click.echo(f"    Complex model: {complex_model}")
    if reasoning:
        click.echo(f"    Reasoning:     {reasoning}")
    if free:
        click.echo(f"    Free model:    {free}")
    click.echo(f"    Config file:   {ENV_FILE}")
    click.echo()
    click.echo("  Next steps:")
    click.echo("    nadirclaw serve           # Start the router")
    click.echo("    nadirclaw status          # Check configuration")
    click.echo("    nadirclaw codex onboard   # Configure Codex integration")
    click.echo("    nadirclaw setup --reconfigure  # Re-run this wizard")
    click.echo()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_setup_wizard(reconfigure: bool = False):
    """Run the full interactive setup wizard."""
    # Step 1: Welcome
    print_welcome()

    # Detect existing state
    existing_creds = detect_existing_credentials() if reconfigure else []

    # Step 2: Provider selection
    providers = prompt_provider_selection(existing=existing_creds or None)

    # Step 2.5: Ollama API base (if Ollama selected)
    ollama_api_base: Optional[str] = None
    if "ollama" in providers:
        click.echo("-" * 56)
        click.echo("  Ollama Configuration")
        click.echo("-" * 56)
        click.echo()
        raw_base = click.prompt(
            "  Ollama API base URL",
            default=OLLAMA_DEFAULT_API_BASE,
        )
        ollama_api_base = _normalize_ollama_api_base(raw_base)
        click.echo(f"  Using: {ollama_api_base}\n")

    # Step 3: Credential collection
    click.echo("-" * 56)
    click.echo("  Credentials")
    click.echo("-" * 56)
    click.echo()

    api_keys: Dict[str, str] = {}
    collected_credentials: Dict[str, str] = {}
    for provider in providers:
        cred = prompt_credential_for_provider(
            provider, reconfigure=reconfigure, ollama_api_base=ollama_api_base,
        )
        if cred:
            collected_credentials[provider] = cred
            # Collect API keys for .env (only plain keys, not OAuth tokens)
            if provider != "ollama":
                info = PROVIDER_INFO[provider]
                if info["env_var"]:
                    # Only write to .env if it looks like an API key (not an OAuth token)
                    if not cred.startswith("eyJ"):  # JWT tokens start with eyJ
                        api_keys[info["env_var"]] = cred

    # Step 3.5: Fetch available models from provider APIs
    click.echo("-" * 56)
    click.echo("  Fetching Available Models")
    click.echo("-" * 56)
    click.echo()

    fetched_models: Dict[str, List[str]] = {}
    for provider in providers:
        cred = collected_credentials.get(provider)
        display = PROVIDER_INFO[provider]["display"]
        click.echo(f"  {display}...", nl=False)
        models = fetch_provider_models(provider, cred or "", ollama_api_base=ollama_api_base)
        if models:
            fetched_models[provider] = models
            click.echo(f" {len(models)} models found")
        else:
            click.echo(" using defaults")

    click.echo()

    # Step 4: Model selection
    click.echo("-" * 56)
    click.echo("  Model Selection")
    click.echo("-" * 56)

    tiers = get_available_models_for_providers(providers, fetched_models=fetched_models or None)

    # Simple (required)
    simple_model = prompt_model_selection("simple", tiers["simple"], providers) if tiers["simple"] else None
    if not simple_model:
        simple_model = select_default_model("simple", providers) or "gemini-2.5-flash"
        click.echo(f"  Using default simple model: {simple_model}\n")

    # Complex (required)
    complex_model = prompt_model_selection("complex", tiers["complex"], providers) if tiers["complex"] else None
    if not complex_model:
        complex_model = select_default_model("complex", providers) or "gpt-4.1"
        click.echo(f"  Using default complex model: {complex_model}\n")

    # Reasoning (optional)
    reasoning_model = None
    if tiers["reasoning"]:
        reasoning_model = prompt_model_selection("reasoning", tiers["reasoning"], providers)

    # Free (optional)
    free_model = None
    if tiers["free"]:
        free_model = prompt_model_selection("free", tiers["free"], providers)

    # Step 5: Write config + summary
    click.echo("-" * 56)
    click.echo("  Writing Configuration")
    click.echo("-" * 56)
    click.echo()

    env_path = write_env_file(
        simple=simple_model,
        complex_model=complex_model,
        reasoning=reasoning_model,
        free=free_model,
        api_keys=api_keys,
        ollama_api_base=ollama_api_base,
    )
    click.echo(f"  Wrote {env_path}")

    print_summary(providers, simple_model, complex_model, reasoning_model, free_model)
