"""OpenAI Codex OAuth runtime + model discovery utilities.

This module isolates endpoint details so server dispatch logic stays stable even if
OpenAI/Codex endpoint details change later.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx

logger = logging.getLogger("nadirclaw")


class OpenAICodexRuntime:
    """Encapsulates endpoint URLs, model mapping, and dynamic discovery cache."""

    def __init__(self) -> None:
        # Codex OAuth (ChatGPT sign-in) uses a dedicated backend endpoint.
        # Keep these configurable because endpoint behavior may evolve.
        self.responses_url = os.getenv(
            "NADIRCLAW_OPENAI_CODEX_RESPONSES_URL",
            "https://chatgpt.com/backend-api/codex/responses",
        ).rstrip("/")
        self.chat_completions_url = os.getenv(
            "NADIRCLAW_OPENAI_CODEX_CHAT_COMPLETIONS_URL",
            "https://api.openai.com/v1/chat/completions",
        ).rstrip("/")

        # Discovery endpoints in priority order.
        models_env = os.getenv("NADIRCLAW_OPENAI_CODEX_MODELS_URLS", "").strip()
        if models_env:
            self.model_urls = [u.strip().rstrip("/") for u in models_env.split(",") if u.strip()]
        else:
            self.model_urls = [
                "https://chatgpt.com/backend-api/codex/models",
                "https://api.openai.com/v1/models",
            ]

        self._cache_path = Path(
            os.getenv(
                "NADIRCLAW_OPENAI_CODEX_MODEL_CACHE",
                str(Path.home() / ".nadirclaw" / "openai_codex_models.json"),
            )
        ).expanduser()
        self._ttl_seconds = int(os.getenv("NADIRCLAW_OPENAI_CODEX_MODEL_TTL", "1800"))
        prefixes_env = os.getenv("NADIRCLAW_OPENAI_CODEX_MODEL_PREFIXES", "").strip()
        if prefixes_env:
            self._allow_prefixes = tuple(p.strip() for p in prefixes_env.split(",") if p.strip())
        else:
            # Pragmatic default: allow GPT-5 family IDs (commonly codex-usable),
            # plus explicit codex markers below.
            self._allow_prefixes = ("gpt-5",)
        contains_env = os.getenv("NADIRCLAW_OPENAI_CODEX_MODEL_CONTAINS", "").strip()
        if contains_env:
            self._allow_contains = tuple(p.strip() for p in contains_env.split(",") if p.strip())
        else:
            self._allow_contains = ("codex",)
        self._cache_include_raw = os.getenv("NADIRCLAW_OPENAI_CODEX_CACHE_INCLUDE_RAW", "").lower() in {
            "1",
            "true",
            "yes",
        }

        self._models: List[str] = []
        self._fetched_at: int = 0
        self._loaded_cache = False
        self._refresh_lock = asyncio.Lock()

    def _is_relevant_model(self, model_id: str) -> bool:
        """Pragmatic allow-list filter for codex-relevant runtime models."""
        lowered = model_id.lower().strip()
        if not lowered:
            return False
        if lowered.startswith("openai-codex/"):
            lowered = lowered[len("openai-codex/") :]
        if any(token in lowered for token in self._allow_contains):
            return True
        if any(lowered.startswith(prefix) for prefix in self._allow_prefixes):
            return True
        return False

    def filter_discovered_models(self, entries: List[Any]) -> List[str]:
        """Filter + normalize discovered model IDs to codex-relevant subset."""
        raw_ids: List[str] = []
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                raw_ids.append(entry["id"].strip())
            elif isinstance(entry, str):
                raw_ids.append(entry.strip())

        filtered = sorted({mid for mid in raw_ids if self._is_relevant_model(mid)})
        dropped = len([mid for mid in raw_ids if mid and mid not in filtered])
        if dropped:
            logger.info(
                "OpenAI Codex discovery filtered %d/%d non-codex models",
                dropped,
                len(raw_ids),
            )
        return filtered

    def _load_cache(self) -> None:
        if self._loaded_cache:
            return
        self._loaded_cache = True
        if not self._cache_path.exists():
            return
        try:
            payload = json.loads(self._cache_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read OpenAI Codex model cache: %s", exc)
            return

        models = payload.get("models") or []
        if isinstance(models, list):
            self._models = [m for m in models if isinstance(m, str) and m.strip()]
        self._fetched_at = int(payload.get("fetched_at", 0) or 0)

    def _write_cache(self, models: List[str], raw_response: Dict[str, Any]) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": int(time.time()),
            "models": models,
        }
        if self._cache_include_raw:
            payload["raw"] = raw_response
        tmp = self._cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(self._cache_path)

    def _is_fresh(self) -> bool:
        return bool(self._fetched_at and (time.time() - self._fetched_at) < self._ttl_seconds)

    def resolve_runtime_model(self, requested_model: str) -> Tuple[str, str]:
        """Map API-visible model name to runtime model id.

        Returns tuple(runtime_model, source).
        """
        self._load_cache()
        model = requested_model.strip()

        discovered_display = set(self.get_discovered_display_models())
        if model in discovered_display:
            return model.removeprefix("openai-codex/"), "discovery"

        if model.startswith("openai-codex/"):
            return model[len("openai-codex/") :], "configured"

        return model, "configured"

    def get_discovered_display_models(self) -> List[str]:
        self._load_cache()
        return [f"openai-codex/{m}" for m in self._models]

    async def refresh_if_stale(self, oauth_token: str | None, force: bool = False) -> List[str]:
        """Refresh discovered model list if cache is stale.

        On failure, returns the last valid in-memory/cache list.
        """
        self._load_cache()
        if not oauth_token:
            return self._models
        if not force and self._is_fresh():
            return self._models

        async with self._refresh_lock:
            if not force and self._is_fresh():
                return self._models
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    data: Dict[str, Any] = {}
                    selected_url = ""
                    for url in self.model_urls:
                        resp = await client.get(
                            url,
                            headers={"Authorization": f"Bearer {oauth_token}"},
                        )
                        if resp.status_code != 200:
                            logger.debug(
                                "OpenAI Codex discovery endpoint failed (%s): %s (%s)",
                                resp.status_code,
                                url,
                                resp.text[:300],
                            )
                            continue
                        selected_url = url
                        data = resp.json()
                        break

                if not data:
                    logger.warning("OpenAI Codex model discovery failed on all endpoints")
                    return self._models

                entries = data.get("data") or data.get("models") or []
                if isinstance(entries, dict):
                    entries = entries.get("data") or []
                models = self.filter_discovered_models(entries)
                if models:
                    self._models = models
                    self._fetched_at = int(time.time())
                    self._write_cache(models, data)
                    logger.info(
                        "OpenAI Codex discovery refreshed: %d models from %s",
                        len(models),
                        selected_url,
                    )
                return self._models
            except Exception as exc:
                logger.warning("OpenAI Codex discovery refresh failed: %s", exc)
                return self._models


_openai_codex_runtime = OpenAICodexRuntime()


def get_openai_codex_runtime() -> OpenAICodexRuntime:
    return _openai_codex_runtime
