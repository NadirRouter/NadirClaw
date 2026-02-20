"""Credential storage and resolution for NadirClaw.

Stores provider API keys/tokens in ~/.nadirclaw/credentials.json.
Resolution chain: OpenClaw stored token (optional) → NadirClaw stored token → env var.
Supports OAuth tokens with automatic refresh for all providers.
OpenClaw integration is optional — NadirClaw works standalone.
"""

import json
import logging
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("nadirclaw")

# Provider name → env var mapping
_ENV_VAR_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-codex": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

# Alternative env vars checked as fallback (order matters)
_ENV_VAR_FALLBACKS = {
    "google": ["GEMINI_API_KEY"],
}

# Model prefix/pattern → provider mapping
# NOTE: order matters — more specific prefixes must come before shorter ones
_MODEL_PROVIDER_PATTERNS = {
    "anthropic/": "anthropic",
    "claude-": "anthropic",
    "openai-codex/": "openai-codex",
    "openai/": "openai",
    "gpt-": "openai",
    "o1-": "openai",
    "o3-": "openai",
    "o4-": "openai",
    "gemini/": "google",
    "gemini-": "google",
    "antigravity/": "antigravity",
    "ollama/": "ollama",
    "cohere/": "cohere",
    "mistral/": "mistral",
    "together_ai/": "together_ai",
    "replicate/": "replicate",
}


def _credentials_path() -> Path:
    return Path.home() / ".nadirclaw" / "credentials.json"


def _read_credentials() -> dict:
    path = _credentials_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read credentials file: %s", e)
        return {}


def _write_credentials(data: dict) -> None:
    path = _credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Advisory file lock prevents concurrent `nadirclaw auth` commands from
    # clobbering each other's writes.
    lock_path = path.parent / ".credentials.lock"
    lock_fd = None
    try:
        if platform.system() != "Windows":
            import fcntl
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

        # Atomic write: write to temp file then rename to prevent partial writes.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".creds-"
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(data, indent=2) + "\n")
            os.replace(tmp_path, str(path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        # Restrict permissions to owner only (Unix)
        if platform.system() != "Windows":
            path.chmod(0o600)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)


def save_credential(provider: str, token: str, source: str = "manual") -> None:
    """Save a credential for a provider.

    Args:
        provider: Provider name (e.g. "anthropic", "openai").
        token: The API key or token.
        source: How it was added ("setup-token", "manual", etc.).
    """
    creds = _read_credentials()
    creds[provider] = {"token": token, "source": source}
    _write_credentials(creds)


def save_oauth_credential(
    provider: str,
    access_token: str,
    refresh_token: str,
    expires_in: int,
    metadata: Optional[dict] = None,
) -> None:
    """Save an OAuth credential with refresh token and expiry.

    Args:
        provider: Provider name (e.g. "openai-codex").
        access_token: The OAuth access token.
        refresh_token: The OAuth refresh token for renewal.
        expires_in: Seconds until the access token expires.
    """
    creds = _read_credentials()
    creds[provider] = {
        "token": access_token,
        "refresh_token": refresh_token,
        "expires_at": int(time.time()) + expires_in,
        "source": "oauth",
    }
    # Add metadata (e.g., project_id, tier, email for Antigravity)
    if metadata:
        creds[provider].update(metadata)
    _write_credentials(creds)


def remove_credential(provider: str) -> bool:
    """Remove a stored credential. Returns True if it existed."""
    creds = _read_credentials()
    if provider in creds:
        del creds[provider]
        _write_credentials(creds)
        return True
    return False





def _get_refresh_func(provider: str):
    """Return the appropriate token refresh function for a provider."""
    from nadirclaw.oauth import (
        refresh_openai_token,
        refresh_anthropic_token,
        refresh_gemini_token,
        refresh_antigravity_token,
    )

    _REFRESH_MAP = {
        "openai": refresh_openai_token,
        "openai-codex": refresh_openai_token,
        "anthropic": refresh_anthropic_token,
        "gemini": refresh_gemini_token,
        "google": refresh_gemini_token,
        "antigravity": refresh_antigravity_token,
    }
    return _REFRESH_MAP.get(provider)


def _maybe_refresh_oauth(provider: str, entry: dict) -> Optional[str]:
    """If the stored credential is an OAuth token that's expired, refresh it.

    Returns the (possibly refreshed) access token, or None on failure.
    """
    if entry.get("source") != "oauth":
        return entry.get("token")

    expires_at = entry.get("expires_at", 0)
    refresh_token = entry.get("refresh_token")

    # Refresh if within 60 seconds of expiry
    if time.time() < (expires_at - 60):
        return entry.get("token")

    if not refresh_token:
        logger.warning("OAuth token expired for %s but no refresh token available", provider)
        return entry.get("token")  # return stale token; the API will reject it

    refresh_func = _get_refresh_func(provider)
    if not refresh_func:
        logger.warning("No refresh function for provider %s", provider)
        return entry.get("token")

    logger.info("Refreshing expired OAuth token for %s...", provider)
    try:
        token_data = refresh_func(refresh_token)
        new_access = token_data["access_token"]
        new_refresh = token_data.get("refresh_token", refresh_token)
        new_expires = token_data.get("expires_in", 3600)

        # Preserve metadata (project_id, email, etc.)
        metadata = {}
        for key in ("project_id", "email", "tier"):
            if key in entry:
                metadata[key] = entry[key]

        save_oauth_credential(provider, new_access, new_refresh, new_expires, metadata=metadata or None)
        logger.info("OAuth token refreshed for %s (expires in %ds)", provider, new_expires)
        return new_access
    except Exception as e:
        logger.error("Failed to refresh OAuth token for %s: %s", provider, e)
        logger.warning(
            "Re-authenticate with: nadirclaw auth %s login", provider
        )
        return None





def get_credential(provider: str) -> Optional[str]:
    """Resolve a credential for a provider.

    Resolution order:
      1. OpenClaw stored token (~/.openclaw/openclaw.json)
      2. NadirClaw stored token (~/.nadirclaw/credentials.json)
         — with automatic OAuth refresh if expired
      3. Environment variable
      4. None

    Args:
        provider: Provider name (e.g. "anthropic", "openai").

    Returns:
        The token string, or None if no credential found.
    """




    # 2. NadirClaw stored credentials (with OAuth auto-refresh)
    creds = _read_credentials()
    entry = creds.get(provider)
    if entry and entry.get("token"):
        return _maybe_refresh_oauth(provider, entry)

    # 3. Environment variable (primary)
    env_var = _ENV_VAR_MAP.get(provider)
    if env_var:
        val = os.getenv(env_var, "")
        if val:
            return val

    # 4. Fallback env vars (e.g. GEMINI_API_KEY for google)
    for fallback_var in _ENV_VAR_FALLBACKS.get(provider, []):
        val = os.getenv(fallback_var, "")
        if val:
            return val

    return None


def get_credential_entry(provider: str) -> Optional[dict]:
    """Return the full credential entry dict for a provider, or None.

    Unlike get_credential() which returns only the token string, this
    returns the full dict including source, refresh_token, expires_at, etc.
    Useful for distinguishing OAuth credentials from API keys.
    """
    creds = _read_credentials()
    entry = creds.get(provider)
    if entry and entry.get("token"):
        # Ensure token is refreshed if needed
        _maybe_refresh_oauth(provider, entry)
        # Re-read in case refresh updated it
        creds = _read_credentials()
        return creds.get(provider)
    return None


def get_credential_source(provider: str) -> Optional[str]:
    """Return the source label for how a credential was resolved.

    Returns one of: "openclaw", "oauth", "setup-token", "manual", "env", or None.
    """


    # 2. NadirClaw stored
    creds = _read_credentials()
    entry = creds.get(provider)
    if entry and entry.get("token"):
        return entry.get("source", "stored")

    # 3. Env var (primary)
    env_var = _ENV_VAR_MAP.get(provider)
    if env_var and os.getenv(env_var, ""):
        return "env"

    # 4. Fallback env vars
    for fallback_var in _ENV_VAR_FALLBACKS.get(provider, []):
        if os.getenv(fallback_var, ""):
            return "env"

    return None


def detect_provider(model: str) -> Optional[str]:
    """Detect provider from a model name.

    Args:
        model: Model name like "claude-sonnet-4-20250514" or "openai/gpt-4o".

    Returns:
        Provider name (e.g. "anthropic") or None if unknown.
    """
    for pattern, provider in _MODEL_PROVIDER_PATTERNS.items():
        if model.startswith(pattern):
            return provider
    return None


def list_credentials() -> list[dict]:
    """List all configured providers with masked tokens and sources.

    Checks all resolution sources for known providers.

    Returns:
        List of dicts with provider, source, and masked_token keys.
    """
    results = []
    # Check all known providers
    providers = set(_ENV_VAR_MAP.keys())
    # Also include any providers in the credentials file
    creds = _read_credentials()
    providers.update(creds.keys())

    for provider in sorted(providers):
        source = get_credential_source(provider)
        if source:
            token = get_credential(provider)
            masked = _mask_token(token) if token else "???"
            results.append({
                "provider": provider,
                "source": source,
                "masked_token": masked,
            })

    return results


def _mask_token(token: str) -> str:
    """Mask a token for display, showing first 8 and last 4 chars."""
    if len(token) <= 12:
        return token[:4] + "***"
    return token[:8] + "..." + token[-4:]
