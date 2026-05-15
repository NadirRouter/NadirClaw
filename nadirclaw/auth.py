"""
Local bearer token authentication for NadirClaw.

Supports both Authorization: Bearer <token> and X-API-Key: <token>
so any OpenAI-compatible client works out of the box.
"""

import hmac
import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import Header, HTTPException, status

from nadirclaw.settings import settings

logger = logging.getLogger(__name__)


class UserSession:
    """User session for local auth."""

    def __init__(self, user_data: Dict[str, Any]):
        self.id = user_data.get("id", "local-user")
        self.email = user_data.get("email", "local@localhost")
        self.name = user_data.get("name", "Local User")
        self.allowed_providers = user_data.get("allowed_providers", [])
        self.allowed_models = user_data.get("allowed_models", [])
        self.api_key_config = user_data.get("api_key_config", {})
        self.raw_data = user_data


def _load_local_users() -> Dict[str, Dict[str, Any]]:
    """Load user configs from NADIRCLAW_USERS_FILE or env defaults."""
    users_file = os.getenv("NADIRCLAW_USERS_FILE", "")
    if users_file and os.path.exists(users_file):
        try:
            with open(users_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load users file %s: %s", users_file, e)

    default_models = settings.tier_models
    token = settings.AUTH_TOKEN

    return {
        token: {
            "id": "local-user",
            "name": "Local User",
            "email": "local@localhost",
            "allowed_providers": [],
            "allowed_models": default_models,
            "api_key_config": {
                "selected_models": default_models,
                "sort_strategy": "smart-routing",
                "use_fallback": True,
                "name": "local",
                "slug": "local",
            },
        }
    }


_LOCAL_USERS: Dict[str, Dict[str, Any]] = _load_local_users()


async def validate_local_auth(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> UserSession:
    """
    Validate a local bearer token or API key.

    Accepts either:
      - Authorization: Bearer <token>
      - X-API-Key: <token>
    """
    _MAX_TOKEN_LENGTH = 1000

    token: Optional[str] = None

    if authorization:
        token = authorization.removeprefix("Bearer ").strip()
    elif x_api_key:
        token = x_api_key.strip()

    # Reject tokens that are unreasonably long (prevent memory abuse)
    if token and len(token) > _MAX_TOKEN_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token format",
        )

    # If no auth token is configured, allow all requests (local-only mode)
    configured_token = settings.AUTH_TOKEN
    if not configured_token:
        return UserSession(_LOCAL_USERS.get("", _default_user()))

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing auth token. Send Authorization: Bearer <token> or X-API-Key: <token>",
        )

    # Constant-time token comparison to prevent timing side-channel attacks.
    # Iterate all configured tokens so timing does not reveal which (or whether)
    # a token exists in the dict.
    matched_user_data = None
    for candidate_token, candidate_data in _LOCAL_USERS.items():
        if hmac.compare_digest(token.encode("utf-8"), candidate_token.encode("utf-8")):
            matched_user_data = candidate_data
            # Do NOT break — iterate all keys for constant-time behavior

    if matched_user_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid auth token",
        )

    return UserSession(matched_user_data)


def _default_user() -> Dict[str, Any]:
    """Default user when auth is disabled."""
    return {
        "id": "local-user",
        "name": "Local User",
        "allowed_models": settings.tier_models,
    }
