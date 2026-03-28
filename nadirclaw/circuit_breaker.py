"""Circuit breaker for LLM provider resilience.

Tracks failure counts per provider in a rolling time window. When a provider
accumulates too many consecutive failures, the circuit "opens" and requests
skip that provider entirely — avoiding the full timeout penalty on every
request when a provider has systemic issues.

States:
    CLOSED   — Normal operation. Failures are counted.
    OPEN     — Provider is down. Requests are rejected immediately.
    HALF_OPEN — Cooldown expired. One probe request is allowed through.

Configurable via environment variables:
    NADIRCLAW_CB_FAILURE_THRESHOLD  — consecutive failures to trip (default 3)
    NADIRCLAW_CB_WINDOW_SECONDS     — rolling window for failures (default 60)
    NADIRCLAW_CB_COOLDOWN_SECONDS   — seconds before OPEN → HALF_OPEN (default 30)
"""

import logging
import os
import time
from enum import Enum
from threading import Lock
from typing import Any, Dict

logger = logging.getLogger("nadirclaw")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class _ProviderState:
    """Mutable state for a single provider's circuit breaker."""

    __slots__ = ("state", "failure_timestamps", "consecutive_failures",
                 "opened_at", "last_failure_at", "success_count", "failure_count")

    def __init__(self):
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_timestamps: list[float] = []
        self.consecutive_failures: int = 0
        self.opened_at: float = 0.0
        self.last_failure_at: float = 0.0
        self.success_count: int = 0
        self.failure_count: int = 0


class CircuitBreaker:
    """Thread-safe circuit breaker for LLM providers.

    Usage::

        cb = CircuitBreaker()

        if not cb.is_available("google"):
            # skip provider, go to fallback
            ...

        try:
            result = await call_provider(...)
            cb.record_success("google")
        except Exception:
            cb.record_failure("google")
            raise
    """

    def __init__(
        self,
        failure_threshold: int | None = None,
        window_seconds: float | None = None,
        cooldown_seconds: float | None = None,
    ):
        self._failure_threshold = failure_threshold or int(
            os.getenv("NADIRCLAW_CB_FAILURE_THRESHOLD", "3")
        )
        self._window_seconds = window_seconds or float(
            os.getenv("NADIRCLAW_CB_WINDOW_SECONDS", "60")
        )
        self._cooldown_seconds = cooldown_seconds or float(
            os.getenv("NADIRCLAW_CB_COOLDOWN_SECONDS", "30")
        )
        self._providers: Dict[str, _ProviderState] = {}
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Internal helpers (must be called with lock held)
    # ------------------------------------------------------------------

    def _get_state(self, provider: str) -> _ProviderState:
        if provider not in self._providers:
            self._providers[provider] = _ProviderState()
        return self._providers[provider]

    def _prune_old_failures(self, ps: _ProviderState, now: float) -> None:
        """Remove failure timestamps outside the rolling window."""
        cutoff = now - self._window_seconds
        ps.failure_timestamps = [t for t in ps.failure_timestamps if t > cutoff]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self, provider: str) -> bool:
        """Check whether a provider should receive requests.

        Returns True if the circuit is CLOSED or transitions to HALF_OPEN
        (allowing one probe request). Returns False if OPEN and cooldown
        has not yet expired.
        """
        if not provider:
            return True

        now = time.time()
        with self._lock:
            ps = self._get_state(provider)

            if ps.state == CircuitState.CLOSED:
                return True

            if ps.state == CircuitState.OPEN:
                elapsed = now - ps.opened_at
                if elapsed >= self._cooldown_seconds:
                    # Cooldown expired — transition to HALF_OPEN
                    ps.state = CircuitState.HALF_OPEN
                    logger.info(
                        "Circuit breaker HALF_OPEN for provider=%s "
                        "(cooldown %.0fs expired, allowing probe request)",
                        provider, elapsed,
                    )
                    return True
                # Still in cooldown
                return False

            if ps.state == CircuitState.HALF_OPEN:
                # Already allowed one probe — block until result comes in.
                # Actually, we allow the probe through; subsequent calls
                # while the probe is in-flight are also allowed (simpler
                # and avoids needing an "in-flight" sub-state).
                return True

        return True  # pragma: no cover

    def record_success(self, provider: str) -> None:
        """Record a successful call. Resets the circuit to CLOSED."""
        if not provider:
            return

        with self._lock:
            ps = self._get_state(provider)
            prev_state = ps.state

            ps.consecutive_failures = 0
            ps.failure_timestamps.clear()
            ps.success_count += 1
            ps.state = CircuitState.CLOSED

            if prev_state != CircuitState.CLOSED:
                logger.info(
                    "Circuit breaker CLOSED for provider=%s (success after %s)",
                    provider, prev_state.value,
                )

    def record_failure(self, provider: str) -> None:
        """Record a failed call. May trip the circuit to OPEN."""
        if not provider:
            return

        now = time.time()
        with self._lock:
            ps = self._get_state(provider)
            ps.failure_count += 1
            ps.last_failure_at = now
            ps.failure_timestamps.append(now)
            ps.consecutive_failures += 1

            # Prune old failures outside the window
            self._prune_old_failures(ps, now)

            if ps.state == CircuitState.HALF_OPEN:
                # Probe failed — re-open immediately
                ps.state = CircuitState.OPEN
                ps.opened_at = now
                logger.warning(
                    "Circuit breaker re-OPENED for provider=%s "
                    "(probe request failed, cooldown=%.0fs)",
                    provider, self._cooldown_seconds,
                )
                return

            # Check if we should trip the breaker
            if (
                ps.state == CircuitState.CLOSED
                and ps.consecutive_failures >= self._failure_threshold
                and len(ps.failure_timestamps) >= self._failure_threshold
            ):
                ps.state = CircuitState.OPEN
                ps.opened_at = now
                logger.warning(
                    "Circuit breaker OPENED for provider=%s "
                    "(%d consecutive failures in %.0fs window, cooldown=%.0fs)",
                    provider,
                    ps.consecutive_failures,
                    self._window_seconds,
                    self._cooldown_seconds,
                )

    def get_status(self) -> Dict[str, Any]:
        """Return the current state of all tracked providers."""
        now = time.time()
        with self._lock:
            result: Dict[str, Any] = {
                "config": {
                    "failure_threshold": self._failure_threshold,
                    "window_seconds": self._window_seconds,
                    "cooldown_seconds": self._cooldown_seconds,
                },
                "providers": {},
            }
            for provider, ps in self._providers.items():
                self._prune_old_failures(ps, now)
                info: Dict[str, Any] = {
                    "state": ps.state.value,
                    "consecutive_failures": ps.consecutive_failures,
                    "failures_in_window": len(ps.failure_timestamps),
                    "total_successes": ps.success_count,
                    "total_failures": ps.failure_count,
                }
                if ps.state == CircuitState.OPEN:
                    remaining = max(
                        0, self._cooldown_seconds - (now - ps.opened_at)
                    )
                    info["cooldown_remaining_seconds"] = round(remaining, 1)
                if ps.last_failure_at:
                    info["last_failure_seconds_ago"] = round(
                        now - ps.last_failure_at, 1
                    )
                result["providers"][provider] = info
            return result


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_circuit_breaker: CircuitBreaker | None = None
_cb_lock = Lock()


def get_circuit_breaker() -> CircuitBreaker:
    """Get or create the global CircuitBreaker singleton."""
    global _circuit_breaker
    if _circuit_breaker is None:
        with _cb_lock:
            if _circuit_breaker is None:
                _circuit_breaker = CircuitBreaker()
    return _circuit_breaker
