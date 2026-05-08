"""In-memory provider health tracking for fallback routing."""

from __future__ import annotations

import collections
import time
from typing import Any


HEALTH_FAILURE_TYPES = {
    "APIConnectionError",
    "BadGatewayError",
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "Timeout",
}


class ProviderHealthTracker:
    """Rolling in-process health tracker keyed by model name."""

    def __init__(
        self,
        *,
        failure_threshold: int = 2,
        cooldown_seconds: int = 60,
        max_models: int = 128,
        now_func=time.time,
    ):
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_seconds = max(1, cooldown_seconds)
        self.max_models = max(1, max_models)
        self._now = now_func
        self._models: collections.OrderedDict[str, dict[str, Any]] = collections.OrderedDict()

    def record_success(self, model: str) -> None:
        state = self._state_for(model)
        state["recent_successes"] += 1
        state["consecutive_failures"] = 0
        state["cooldown_until"] = 0.0
        state["status"] = "healthy"

    def record_failure(self, model: str, error_type: str, message: str = "") -> None:
        state = self._state_for(model)
        state["last_error"] = {"error_type": error_type, "message": message}
        if not self._counts_as_health_failure(error_type):
            return

        state["recent_failures"] += 1
        state["consecutive_failures"] += 1
        if state["consecutive_failures"] >= self.failure_threshold:
            state["cooldown_until"] = self._now() + self.cooldown_seconds
            state["status"] = "cooling_down"

    def ordered_candidates(self, models: list[str]) -> list[str]:
        healthy: list[str] = []
        unhealthy: list[str] = []
        for model in models:
            if self.is_available(model):
                healthy.append(model)
            else:
                unhealthy.append(model)
        return healthy + unhealthy

    def is_available(self, model: str) -> bool:
        state = self._models.get(model)
        if not state:
            return True
        cooldown_until = state.get("cooldown_until", 0.0)
        return not cooldown_until or self._now() >= cooldown_until

    def snapshot(self) -> dict[str, Any]:
        models: dict[str, Any] = {}
        now = self._now()
        for model, state in self._models.items():
            cooldown_until = state.get("cooldown_until", 0.0)
            if cooldown_until and now < cooldown_until:
                status = "cooling_down"
            elif state.get("consecutive_failures", 0) >= self.failure_threshold:
                status = "unhealthy"
            else:
                status = "healthy"
            models[model] = {
                "recent_failures": state["recent_failures"],
                "recent_successes": state["recent_successes"],
                "status": status,
                "last_error": state.get("last_error"),
            }
            if cooldown_until and now < cooldown_until:
                models[model]["cooldown_seconds_remaining"] = int(cooldown_until - now) + 1
        return {"models": models}

    def reset(self) -> None:
        self._models.clear()

    def _state_for(self, model: str) -> dict[str, Any]:
        state = self._models.get(model)
        if state is None:
            state = {
                "recent_failures": 0,
                "recent_successes": 0,
                "consecutive_failures": 0,
                "cooldown_until": 0.0,
                "status": "healthy",
                "last_error": None,
            }
            self._models[model] = state
        else:
            self._models.move_to_end(model)

        while len(self._models) > self.max_models:
            self._models.popitem(last=False)
        return state

    @staticmethod
    def _counts_as_health_failure(error_type: str) -> bool:
        if error_type in HEALTH_FAILURE_TYPES:
            return True
        if error_type.endswith("ServerError"):
            return True
        if error_type in {"InternalServerError", "ServiceUnavailableError"}:
            return True
        return False


provider_health_tracker = ProviderHealthTracker()
