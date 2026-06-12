"""Standalone OAuth helpers for NadirClaw (OpenAI, Anthropic, Google/Gemini).

Implements native OAuth PKCE flows without requiring external CLIs.
Also supports reading credentials from OpenClaw (optional fallback).
"""

import base64
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Optional
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

logger = logging.getLogger("nadirclaw")

# ---------------------------------------------------------------------------
# OAuth Configuration
# ---------------------------------------------------------------------------

# Local callback server (defined first, used by other constants)
_CALLBACK_PORT = 1455
_CALLBACK_PATH = "/auth/callback"

# OpenAI OAuth (PKCE)
_OPENAI_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_OPENAI_AUTH_BASE = "https://auth.openai.com"
_OPENAI_AUTHORIZE_URL = f"{_OPENAI_AUTH_BASE}/oauth/authorize"
_OPENAI_TOKEN_URL = f"{_OPENAI_AUTH_BASE}/oauth/token"
_OPENAI_AUDIENCE = "https://api.openai.com/v1"
_OPENAI_SCOPES = "openid profile email offline_access"

# Anthropic OAuth (PKCE) - using public client
_ANTHROPIC_CLIENT_ID = "claude-cli"  # Public client ID
_ANTHROPIC_AUTH_BASE = "https://auth.anthropic.com"
_ANTHROPIC_AUTHORIZE_URL = f"{_ANTHROPIC_AUTH_BASE}/authorize"
_ANTHROPIC_TOKEN_URL = f"{_ANTHROPIC_AUTH_BASE}/oauth/token"
_ANTHROPIC_SCOPES = "openid profile email offline_access"

# Google OAuth endpoints (shared by Gemini CLI and Antigravity)
_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo?alt=json"

# Google Antigravity OAuth — requires env vars for client credentials.
# Set NADIRCLAW_ANTIGRAVITY_CLIENT_ID and NADIRCLAW_ANTIGRAVITY_CLIENT_SECRET
# in your environment. These are Google "installed application" OAuth credentials
# (same pattern as gcloud CLI, Gemini CLI, and other Google desktop tools).
_ANTIGRAVITY_CLIENT_ID = os.getenv("NADIRCLAW_ANTIGRAVITY_CLIENT_ID", "")
_ANTIGRAVITY_CLIENT_SECRET = os.getenv("NADIRCLAW_ANTIGRAVITY_CLIENT_SECRET", "")
_ANTIGRAVITY_CALLBACK_PORT = 51121
_ANTIGRAVITY_CALLBACK_PATH = "/oauth-callback"
_ANTIGRAVITY_REDIRECT_URI = f"http://localhost:{_ANTIGRAVITY_CALLBACK_PORT}{_ANTIGRAVITY_CALLBACK_PATH}"
_ANTIGRAVITY_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]
_ANTIGRAVITY_DEFAULT_PROJECT_ID = "rising-fact-p41fc"

# Google Gemini CLI OAuth — credentials extracted from Gemini CLI or env vars
_GEMINI_CALLBACK_PORT = 8085
_GEMINI_CALLBACK_PATH = "/oauth2callback"
_GEMINI_REDIRECT_URI = f"http://localhost:{_GEMINI_CALLBACK_PORT}{_GEMINI_CALLBACK_PATH}"
_GEMINI_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]
_GEMINI_CLIENT_ID_ENV_KEYS = [
    "NADIRCLAW_GEMINI_OAUTH_CLIENT_ID",
    "OPENCLAW_GEMINI_OAUTH_CLIENT_ID",
    "GEMINI_CLI_OAUTH_CLIENT_ID",
]
_GEMINI_CLIENT_SECRET_ENV_KEYS = [
    "NADIRCLAW_GEMINI_OAUTH_CLIENT_SECRET",
    "OPENCLAW_GEMINI_OAUTH_CLIENT_SECRET",
    "GEMINI_CLI_OAUTH_CLIENT_SECRET",
]

# Code Assist endpoints (for project discovery — shared by Gemini CLI and Antigravity)
_CODE_ASSIST_ENDPOINTS = [
    "https://cloudcode-pa.googleapis.com",
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
]


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------

def _generate_code_verifier() -> str:
    """Generate a cryptographically random code verifier (43-128 chars)."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8").rstrip("=")


def _generate_code_challenge(verifier: str) -> str:
    """Generate code challenge from verifier (SHA256 hash, base64url)."""
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def _encode_state_base64url(payload: dict) -> str:
    """Encode state as base64url (Antigravity-style)."""
    json_str = json.dumps(payload)
    # Use base64url encoding (no padding, - instead of +, _ instead of /)
    encoded = base64.urlsafe_b64encode(json_str.encode("utf-8")).decode("utf-8").rstrip("=")
    return encoded


def _decode_state_base64url(state: str) -> dict:
    """Decode base64url state (Antigravity-style)."""
    # Handle both base64url and base64 formats
    normalized = state.replace("-", "+").replace("_", "/")
    # Add padding if needed
    padding = (4 - len(normalized) % 4) % 4
    padded = normalized + ("=" * padding)
    json_str = base64.b64decode(padded).decode("utf-8")
    return json.loads(json_str)


# ---------------------------------------------------------------------------
# Local callback server
# ---------------------------------------------------------------------------

class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP server to receive OAuth callback."""

    def __init__(self, callback_queue, callback_path, *args, **kwargs):
        self.callback_queue = callback_queue
        self.callback_path = callback_path
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_GET(self):
        """Handle OAuth callback."""
        if self.path.startswith(self.callback_path):
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            code = params.get("code", [None])[0]
            error = params.get("error", [None])[0]
            state = params.get("state", [None])[0]

            if error:
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(
                    f"<html><body><h1>Authorization failed</h1><p>{error}</p></body></html>".encode()
                )
                self.callback_queue.put({"error": error})
            elif code:
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h1>Authorization successful!</h1><p>You can close this window.</p></body></html>"
                )
                self.callback_queue.put({"code": code, "state": state})
            else:
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Missing authorization code</h1></body></html>")
                self.callback_queue.put({"error": "missing_code"})
        else:
            self.send_response(404)
            self.end_headers()


def _start_callback_server(
    timeout: int = 300,
    port: int = _CALLBACK_PORT,
    callback_path: str = _CALLBACK_PATH,
):
    """Start local HTTP server to receive OAuth callback.

    Returns (server, queue) where queue receives {"code": "...", "state": "..."} or {"error": "..."}.
    """
    import queue

    callback_queue = queue.Queue()
    redirect_uri = f"http://localhost:{port}{callback_path}"

    def handler_factory(*args, **kwargs):
        return OAuthCallbackHandler(callback_queue, callback_path, *args, **kwargs)

    try:
        server = HTTPServer(("localhost", port), handler_factory)
    except OSError as e:
        if e.errno in (48, 98):  # EADDRINUSE on macOS / Linux
            raise RuntimeError(
                f"Port {port} is already in use. "
                "Make sure no other OAuth login is running, then try again."
            ) from e
        raise

    def serve():
        print(f"Waiting for OAuth callback on {redirect_uri}...")
        server.serve_forever()

    thread = Thread(target=serve, daemon=True)
    thread.start()

    return server, callback_queue


# ---------------------------------------------------------------------------
# OpenAI OAuth
# ---------------------------------------------------------------------------

def login_openai(timeout: int = 300) -> Optional[dict]:
    """Run standalone OpenAI OAuth PKCE flow.

    Returns dict with: access_token, refresh_token, expires_at — or None.
    """
    # Generate PKCE parameters
    code_verifier = _generate_code_verifier()
    code_challenge = _generate_code_challenge(code_verifier)
    state = secrets.token_urlsafe(32)

    # Must exactly match the redirect_uri registered for the OpenAI OAuth client
    # AND the callback server below (which binds/prints localhost). Hydra exact-
    # matches this value against the client allow-list and rejects the authorize
    # request with `authorize_hydra_invalid_request` on any mismatch — using
    # 127.0.0.1 here instead of localhost triggers exactly that error (issue #67).
    redirect_uri = f"http://localhost:{_CALLBACK_PORT}{_CALLBACK_PATH}"

    # Build authorization URL
    auth_params = {
        "response_type": "code",
        "client_id": _OPENAI_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": _OPENAI_SCOPES,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "audience": _OPENAI_AUDIENCE,
    }
    auth_url = f"{_OPENAI_AUTHORIZE_URL}?{urllib.parse.urlencode(auth_params)}"

    logger.info("Starting OpenAI OAuth flow...")
    logger.debug("Authorization URL: %s", auth_url)

    # Start callback server
    server, callback_queue = _start_callback_server(timeout)

    try:
        # Open browser
        print(f"\nOpening browser for OpenAI authorization...")
        print(f"If the browser doesn't open, visit:\n  {auth_url}\n")
        webbrowser.open(auth_url)

        # Wait for callback
        print("Waiting for authorization...")
        try:
            result = callback_queue.get(timeout=timeout)
        except Exception:
            raise RuntimeError(f"Authorization timed out after {timeout}s")

        if "error" in result:
            raise RuntimeError(f"Authorization failed: {result['error']}")

        auth_code = result.get("code")
        if not auth_code:
            raise RuntimeError("No authorization code received")

        # Verify state
        if result.get("state") != state:
            raise RuntimeError("State mismatch — possible CSRF attack")

        # Exchange code for tokens
        token_data = {
            "grant_type": "authorization_code",
            "client_id": _OPENAI_CLIENT_ID,
            "code": auth_code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        }

        req = urllib.request.Request(
            _OPENAI_TOKEN_URL,
            data=urllib.parse.urlencode(token_data).encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                token_response = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Token exchange failed ({e.code}): {body}") from e

        access_token = token_response.get("access_token")
        refresh_token = token_response.get("refresh_token")
        expires_in = token_response.get("expires_in", 3600)

        if not access_token:
            raise RuntimeError("No access token in response")

        return {
            "access_token": access_token,
            "refresh_token": refresh_token or "",
            "expires_at": int(time.time()) + expires_in,
        }

    finally:
        server.shutdown()


def refresh_openai_token(refresh_token: str, *, client_id: str = "") -> dict:
    """Refresh an OpenAI access token using a refresh token.

    Args:
        refresh_token: The OAuth refresh token.
        client_id: Optional override. When refreshing tokens issued by another
            OAuth client (e.g. OpenClaw/pi-ai), the original client_id must be
            used or the refresh will fail with 401.
    """
    data = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "client_id": client_id or _OPENAI_CLIENT_ID,
        "refresh_token": refresh_token,
    }).encode("utf-8")

    req = urllib.request.Request(
        _OPENAI_TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Token refresh failed ({e.code}): {body}") from e


# Keep backward compat alias
refresh_access_token = refresh_openai_token


def refresh_anthropic_token(refresh_token: str, *, client_id: str = "") -> dict:
    """Refresh an Anthropic access token using a refresh token."""
    data = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "client_id": client_id or _ANTHROPIC_CLIENT_ID,
        "refresh_token": refresh_token,
    }).encode("utf-8")

    req = urllib.request.Request(
        _ANTHROPIC_TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic token refresh failed ({e.code}): {body}") from e


def _refresh_google_token(refresh_token: str, client_id: str, client_secret: str = "") -> dict:
    """Refresh a Google OAuth access token using a refresh token."""
    params = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "refresh_token": refresh_token,
    }
    if client_secret:
        params["client_secret"] = client_secret

    data = urllib.parse.urlencode(params).encode("utf-8")
    req = urllib.request.Request(
        _GOOGLE_TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Google token refresh failed ({e.code}): {body}") from e


def refresh_gemini_token(refresh_token: str, *, client_id: str = "") -> dict:
    """Refresh a Gemini CLI OAuth access token.

    Args:
        refresh_token: The OAuth refresh token.
        client_id: Optional override for the OAuth client_id. When refreshing
            tokens issued by OpenClaw, use the client_id from OpenClaw's
            auth-profiles to avoid 401 errors.
    """
    if client_id:
        # Use the provided client_id (e.g. from OpenClaw's auth-profiles).
        # Try to find a matching client_secret from env.
        client_secret = ""
        for skey in _GEMINI_CLIENT_SECRET_ENV_KEYS:
            sval = os.getenv(skey, "").strip()
            if sval:
                client_secret = sval
                break
        return _refresh_google_token(refresh_token, client_id=client_id, client_secret=client_secret)

    client_config = _resolve_gemini_client_config()
    if not client_config.get("client_id"):
        raise RuntimeError("Cannot refresh Gemini token: Gemini CLI not installed")
    return _refresh_google_token(
        refresh_token,
        client_id=client_config["client_id"],
        client_secret=client_config.get("client_secret", ""),
    )


def refresh_antigravity_token(refresh_token: str, *, client_id: str = "") -> dict:
    """Refresh an Antigravity OAuth access token."""
    if not _ANTIGRAVITY_CLIENT_ID or not _ANTIGRAVITY_CLIENT_SECRET:
        raise RuntimeError(
            "Antigravity OAuth credentials not configured. "
            "Set NADIRCLAW_ANTIGRAVITY_CLIENT_ID and NADIRCLAW_ANTIGRAVITY_CLIENT_SECRET env vars."
        )
    return _refresh_google_token(
        refresh_token,
        client_id=_ANTIGRAVITY_CLIENT_ID,
        client_secret=_ANTIGRAVITY_CLIENT_SECRET,
    )


# ---------------------------------------------------------------------------
# Anthropic setup token (like OpenClaw — not full OAuth)
# ---------------------------------------------------------------------------

ANTHROPIC_SETUP_TOKEN_PREFIX = "sk-ant-oat01-"
ANTHROPIC_SETUP_TOKEN_MIN_LENGTH = 80


def validate_anthropic_setup_token(token: str) -> Optional[str]:
    """Validate an Anthropic setup token.

    Returns error message string if invalid, or None if valid.
    """
    trimmed = token.strip()
    if not trimmed:
        return "Token is empty"
    if not trimmed.startswith(ANTHROPIC_SETUP_TOKEN_PREFIX):
        return f"Expected token starting with {ANTHROPIC_SETUP_TOKEN_PREFIX}"
    if len(trimmed) < ANTHROPIC_SETUP_TOKEN_MIN_LENGTH:
        return "Token looks too short — paste the full setup-token"
    return None


def login_anthropic() -> Optional[dict]:
    """Authenticate with Anthropic using a setup token from `claude setup-token`.

    Prompts the user to run `claude setup-token` in another terminal,
    then waits for them to paste the generated token.

    Returns dict with: token — or None.
    """
    print("\n--- Anthropic Setup Token ---")
    print("1. Open another terminal and run:  claude setup-token")
    print("2. Copy the generated token (starts with sk-ant-oat01-...)")
    print("3. Paste it below\n")

    token = input("Paste Anthropic setup-token: ").strip()

    error = validate_anthropic_setup_token(token)
    if error:
        raise RuntimeError(f"Invalid setup token: {error}")

    return {"token": token}


# ---------------------------------------------------------------------------
# Shared Google helpers (used by both Gemini CLI and Antigravity)
# ---------------------------------------------------------------------------

def _fetch_google_user_email(access_token: str) -> Optional[str]:
    """Fetch user email from Google userinfo endpoint."""
    try:
        req = urllib.request.Request(
            _GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("email")
    except Exception:
        return None


def _fetch_project_id(access_token: str) -> str:
    """Discover Google Cloud project ID from Code Assist API.

    Tries multiple endpoints. Returns project ID or empty string.
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "google-api-nodejs-client/9.15.1",
        "X-Goog-Api-Client": "gl-node/nadirclaw",
    }

    load_body = json.dumps({
        "metadata": {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        },
    }).encode("utf-8")

    for endpoint in _CODE_ASSIST_ENDPOINTS:
        try:
            url = f"{endpoint}/v1internal:loadCodeAssist"
            req = urllib.request.Request(
                url, data=load_body, headers=headers, method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

            project = data.get("cloudaicompanionProject")
            if isinstance(project, str) and project:
                return project
            if isinstance(project, dict) and project.get("id"):
                return project["id"]
        except Exception as e:
            logger.debug("Code Assist discovery at %s failed: %s", endpoint, e)

    return ""


def _fetch_project_id_with_onboard(access_token: str) -> str:
    """Discover or provision Google Cloud project via Code Assist API.

    Like _fetch_project_id but also tries onboarding if no project exists.
    Falls back to a default project ID for Antigravity.
    """
    env_project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "google-api-nodejs-client/9.15.1",
        "X-Goog-Api-Client": "gl-node/nadirclaw",
    }

    load_body = json.dumps({
        "cloudaicompanionProject": env_project or "",
        "metadata": {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
            "duetProject": env_project or "",
        },
    }).encode("utf-8")

    endpoint = _CODE_ASSIST_ENDPOINTS[0]
    try:
        url = f"{endpoint}/v1internal:loadCodeAssist"
        req = urllib.request.Request(
            url, data=load_body, headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        # Check for existing project
        project = data.get("cloudaicompanionProject")
        if isinstance(project, str) and project:
            return project
        if isinstance(project, dict) and project.get("id"):
            return project["id"]

        # Try onboarding
        tier_id = "free-tier"
        allowed_tiers = data.get("allowedTiers", [])
        if isinstance(allowed_tiers, list):
            for t in allowed_tiers:
                if isinstance(t, dict) and t.get("isDefault"):
                    tier_id = t.get("id", "free-tier")
                    break

        onboard_body = json.dumps({
            "tierId": tier_id,
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            },
        }).encode("utf-8")

        onboard_req = urllib.request.Request(
            f"{endpoint}/v1internal:onboardUser",
            data=onboard_body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(onboard_req, timeout=30) as resp:
            lro = json.loads(resp.read())

        # Poll long-running operation
        if not lro.get("done") and lro.get("name"):
            op_name = lro["name"]
            for _ in range(24):
                time.sleep(5)
                poll_req = urllib.request.Request(
                    f"{endpoint}/v1internal/{op_name}",
                    headers=headers,
                )
                with urllib.request.urlopen(poll_req, timeout=30) as resp:
                    lro = json.loads(resp.read())
                if lro.get("done"):
                    break

        project_id = (lro.get("response", {}) or {}).get("cloudaicompanionProject", {})
        if isinstance(project_id, dict):
            project_id = project_id.get("id", "")
        if project_id:
            return project_id
    except Exception as e:
        logger.debug("Project discovery/onboard failed: %s", e)

    if env_project:
        return env_project
    return ""


# ---------------------------------------------------------------------------
# Google Antigravity OAuth
# ---------------------------------------------------------------------------

def login_antigravity(timeout: int = 300) -> Optional[dict]:
    """Run standalone Google Antigravity OAuth flow using account-based auth.

    Requires NADIRCLAW_ANTIGRAVITY_CLIENT_ID and NADIRCLAW_ANTIGRAVITY_CLIENT_SECRET env vars.

    Returns dict with: access_token, refresh_token, expires_at, project_id, email — or None.
    """
    if not _ANTIGRAVITY_CLIENT_ID or not _ANTIGRAVITY_CLIENT_SECRET:
        raise RuntimeError(
            "Antigravity OAuth credentials not configured. "
            "Set NADIRCLAW_ANTIGRAVITY_CLIENT_ID and NADIRCLAW_ANTIGRAVITY_CLIENT_SECRET env vars."
        )
    # Generate PKCE parameters
    code_verifier = _generate_code_verifier()
    code_challenge = _generate_code_challenge(code_verifier)
    state = secrets.token_urlsafe(32)

    # Build authorization URL
    auth_params = {
        "client_id": _ANTIGRAVITY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": _ANTIGRAVITY_REDIRECT_URI,
        "scope": " ".join(_ANTIGRAVITY_SCOPES),
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_url = f"{_GOOGLE_AUTH_URL}?{urllib.parse.urlencode(auth_params)}"

    logger.info("Starting Antigravity OAuth flow...")
    logger.debug("Authorization URL: %s", auth_url)

    # Start callback server on Antigravity port
    server, callback_queue = _start_callback_server(
        timeout,
        port=_ANTIGRAVITY_CALLBACK_PORT,
        callback_path=_ANTIGRAVITY_CALLBACK_PATH,
    )

    try:
        print("\nOpening browser for Antigravity authorization...")
        print(f"If the browser doesn't open, visit:\n  {auth_url}\n")
        webbrowser.open(auth_url)

        print("Waiting for authorization...")
        try:
            result = callback_queue.get(timeout=timeout)
        except Exception:
            raise RuntimeError(f"Authorization timed out after {timeout}s")

        if "error" in result:
            raise RuntimeError(f"Authorization failed: {result['error']}")

        auth_code = result.get("code")
        if not auth_code:
            raise RuntimeError("No authorization code received")

        # Verify state
        if result.get("state") != state:
            raise RuntimeError("State mismatch — possible CSRF attack")

        # Exchange code for tokens (with client_secret)
        token_data = {
            "client_id": _ANTIGRAVITY_CLIENT_ID,
            "client_secret": _ANTIGRAVITY_CLIENT_SECRET,
            "code": auth_code,
            "grant_type": "authorization_code",
            "redirect_uri": _ANTIGRAVITY_REDIRECT_URI,
            "code_verifier": code_verifier,
        }

        req = urllib.request.Request(
            _GOOGLE_TOKEN_URL,
            data=urllib.parse.urlencode(token_data).encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                token_response = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Token exchange failed ({e.code}): {body}") from e

        access_token = token_response.get("access_token")
        refresh_token = token_response.get("refresh_token")
        expires_in = token_response.get("expires_in", 3600)

        if not access_token:
            raise RuntimeError("No access token in response")
        if not refresh_token:
            raise RuntimeError("Missing refresh token in response")

        # Fetch user info and project ID
        email = _fetch_google_user_email(access_token)
        project_id = _fetch_project_id(access_token) or _ANTIGRAVITY_DEFAULT_PROJECT_ID

        # Apply 5-minute safety buffer (like OpenClaw)
        expires_at = int(time.time()) + expires_in - 300

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
            "project_id": project_id,
            "email": email,
        }

    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Gemini CLI — delegate to `gemini auth login` and read stored credentials
# ---------------------------------------------------------------------------

_GEMINI_OAUTH_CREDS_PATH = Path.home() / ".gemini" / "oauth_creds.json"
_GEMINI_ACCOUNTS_PATH = Path.home() / ".gemini" / "google_accounts.json"


def _read_gemini_cli_credentials() -> Optional[dict]:
    """Read credentials stored by the Gemini CLI at ~/.gemini/oauth_creds.json.

    Returns dict with: access_token, refresh_token, expires_at, email — or None.
    """
    if not _GEMINI_OAUTH_CREDS_PATH.exists():
        return None

    try:
        data = json.loads(_GEMINI_OAUTH_CREDS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    access_token = data.get("access_token", "")
    refresh_token = data.get("refresh_token", "")
    expiry_date = data.get("expiry_date", 0)  # Gemini CLI uses ms

    if not access_token or not refresh_token:
        return None

    # Convert ms → seconds
    expires_at = int(expiry_date) // 1000 if expiry_date else 0

    # Read email from google_accounts.json
    email = None
    try:
        if _GEMINI_ACCOUNTS_PATH.exists():
            accounts = json.loads(_GEMINI_ACCOUNTS_PATH.read_text())
            email = accounts.get("active")
    except (json.JSONDecodeError, OSError):
        pass

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at,
        "email": email,
    }


def _read_gemini_credentials() -> Optional[dict]:
    """Read Gemini credentials from any available source.

    Checks:
      1. Gemini CLI (~/.gemini/oauth_creds.json)
      2. OpenClaw auth-profiles

    Returns dict with: access_token, refresh_token, expires_at, email, project_id — or None.
    """
    # 1. Try Gemini CLI's own storage (most direct)
    creds = _read_gemini_cli_credentials()
    if creds:
        return creds

    # 2. Try OpenClaw auth-profiles
    for profile_path in [
        Path.home() / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json",
        Path.home() / ".openclaw" / "auth-profiles.json",
    ]:
        if profile_path.exists():
            try:
                data = json.loads(profile_path.read_text())
                profiles = data.get("profiles", {})
                for key, profile in profiles.items():
                    if profile.get("provider") == "google-gemini-cli" and profile.get("access"):
                        return {
                            "access_token": profile["access"],
                            "refresh_token": profile.get("refresh", ""),
                            "expires_at": int(profile.get("expires", 0)) // 1000,
                            "project_id": profile.get("projectId", ""),
                            "email": profile.get("email"),
                        }
            except (json.JSONDecodeError, OSError, KeyError):
                pass

    return None


def _resolve_gemini_client_config() -> dict:
    """Resolve Gemini CLI OAuth client config for token refresh.

    Extracts client_id/secret from the installed Gemini CLI binary by parsing
    its bundled oauth2.js file. This is inherently fragile — if the Gemini CLI
    changes its file structure, minifies differently, or uses a bundler, the
    regex extraction may break. If this happens, set env vars instead:
      NADIRCLAW_GEMINI_OAUTH_CLIENT_ID
      NADIRCLAW_GEMINI_OAUTH_CLIENT_SECRET

    Returns dict with: client_id, client_secret (optional).
    """
    # Check env vars first
    for key in _GEMINI_CLIENT_ID_ENV_KEYS:
        val = os.getenv(key, "").strip()
        if val:
            result = {"client_id": val}
            for skey in _GEMINI_CLIENT_SECRET_ENV_KEYS:
                sval = os.getenv(skey, "").strip()
                if sval:
                    result["client_secret"] = sval
                    break
            return result

    # Extract from Gemini CLI binary
    gemini_path = shutil.which("gemini")
    if not gemini_path:
        return {}

    try:
        resolved = os.path.realpath(gemini_path)
        gemini_cli_dir = os.path.dirname(os.path.dirname(resolved))

        search_paths = [
            os.path.join(
                gemini_cli_dir, "node_modules", "@google", "gemini-cli-core",
                "dist", "src", "code_assist", "oauth2.js",
            ),
            os.path.join(
                gemini_cli_dir, "node_modules", "@google", "gemini-cli-core",
                "dist", "code_assist", "oauth2.js",
            ),
        ]

        for p in search_paths:
            if os.path.isfile(p):
                with open(p) as f:
                    content = f.read()
                id_match = re.search(r"(\d+-[a-z0-9]+\.apps\.googleusercontent\.com)", content)
                secret_match = re.search(r"(GOCSPX-[A-Za-z0-9_-]+)", content)
                if id_match and secret_match:
                    return {
                        "client_id": id_match.group(1),
                        "client_secret": secret_match.group(1),
                    }
    except Exception:
        pass

    return {}


def login_gemini(timeout: int = 300) -> Optional[dict]:
    """Run standalone Gemini OAuth PKCE flow using account-based auth.

    Extracts OAuth client credentials from the installed Gemini CLI,
    opens a browser for authorization, and catches the callback.

    Returns dict with: access_token, refresh_token, expires_at, project_id, email — or None.
    """
    # Resolve client credentials from Gemini CLI or env vars
    client_config = _resolve_gemini_client_config()
    if not client_config.get("client_id"):
        raise RuntimeError(
            "Could not find Gemini CLI OAuth credentials.\n\n"
            "Install the Gemini CLI first:\n"
            "  brew install gemini-cli  (macOS)\n"
            "  npm install -g @google/gemini-cli  (npm)\n\n"
            "Or set env vars:\n"
            "  NADIRCLAW_GEMINI_OAUTH_CLIENT_ID\n"
            "  NADIRCLAW_GEMINI_OAUTH_CLIENT_SECRET"
        )

    client_id = client_config["client_id"]
    client_secret = client_config.get("client_secret", "")

    # Generate PKCE parameters
    code_verifier = _generate_code_verifier()
    code_challenge = _generate_code_challenge(code_verifier)
    state = secrets.token_urlsafe(32)

    # Build authorization URL
    auth_params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": _GEMINI_REDIRECT_URI,
        "scope": " ".join(_GEMINI_SCOPES),
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_url = f"{_GOOGLE_AUTH_URL}?{urllib.parse.urlencode(auth_params)}"

    logger.info("Starting Gemini OAuth flow...")
    logger.debug("Authorization URL: %s", auth_url)

    # Start callback server on Gemini port
    server, callback_queue = _start_callback_server(
        timeout,
        port=_GEMINI_CALLBACK_PORT,
        callback_path=_GEMINI_CALLBACK_PATH,
    )

    try:
        print("\nOpening browser for Gemini authorization...")
        print(f"If the browser doesn't open, visit:\n  {auth_url}\n")
        webbrowser.open(auth_url)

        try:
            result = callback_queue.get(timeout=timeout)
        except Exception:
            raise RuntimeError(f"Authorization timed out after {timeout}s")

        if "error" in result:
            raise RuntimeError(f"Authorization failed: {result['error']}")

        auth_code = result.get("code")
        if not auth_code:
            raise RuntimeError("No authorization code received")

        # Verify state
        if result.get("state") != state:
            raise RuntimeError("State mismatch — possible CSRF attack")

        # Exchange code for tokens
        token_params = {
            "client_id": client_id,
            "code": auth_code,
            "grant_type": "authorization_code",
            "redirect_uri": _GEMINI_REDIRECT_URI,
            "code_verifier": code_verifier,
        }
        if client_secret:
            token_params["client_secret"] = client_secret

        req = urllib.request.Request(
            _GOOGLE_TOKEN_URL,
            data=urllib.parse.urlencode(token_params).encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                token_response = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Token exchange failed ({e.code}): {body}") from e

        access_token = token_response.get("access_token")
        refresh_token = token_response.get("refresh_token")
        expires_in = token_response.get("expires_in", 3600)

        if not access_token:
            raise RuntimeError("No access token in response")
        if not refresh_token:
            raise RuntimeError("Missing refresh token in response")

        # Fetch user info and project ID
        email = _fetch_google_user_email(access_token)
        project_id = _fetch_project_id(access_token)

        # Apply 5-minute safety buffer (like OpenClaw)
        expires_at = int(time.time()) + expires_in - 300

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
            "project_id": project_id,
            "email": email,
        }

    finally:
        server.shutdown()

