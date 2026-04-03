"""Budget tracking and alerts for NadirClaw.

Tracks cumulative spend against configurable daily/monthly budgets.
When a budget threshold is approached or exceeded, logs warnings.
"""

import json
import logging
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional

from nadirclaw.routing import estimate_cost
from nadirclaw.settings import settings

logger = logging.getLogger("nadirclaw.budget")


def _send_webhook(url: str, payload: Dict[str, Any], timeout: int = 10) -> None:
    """POST a JSON payload to a webhook URL (fire-and-forget in a thread)."""
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            logger.debug("Webhook delivered to %s — HTTP %s", url, resp.status)
    except Exception as e:
        logger.warning("Webhook delivery failed (%s): %s", url, e)


class BudgetTracker:
    """Track spend in real-time with configurable budget limits.

    Spend data is kept in memory and periodically flushed to disk.
    On startup, loads the current day/month totals from the state file.
    """

    def __init__(
        self,
        daily_budget: Optional[float] = None,
        monthly_budget: Optional[float] = None,
        warn_threshold: float = 0.8,
        state_file: Optional[Path] = None,
        webhook_url: Optional[str] = None,
        stdout_alerts: bool = False,
    ):
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self.warn_threshold = warn_threshold
        self.webhook_url = webhook_url
        self.stdout_alerts = stdout_alerts
        self._state_file = state_file or (settings.LOG_DIR / "budget_state.json")
        self._lock = Lock()

        # Spend accumulators
        self._daily_spend: float = 0.0
        self._monthly_spend: float = 0.0
        self._daily_requests: int = 0
        self._monthly_requests: int = 0
        self._current_day: str = ""
        self._current_month: str = ""

        # Per-model spend tracking
        self._model_spend: Dict[str, float] = {}
        self._model_requests: Dict[str, int] = {}

        # Alert state (avoid spamming)
        self._daily_warn_sent = False
        self._daily_limit_sent = False
        self._monthly_warn_sent = False
        self._monthly_limit_sent = False

        self._load_state()

    def _load_state(self) -> None:
        """Load persisted budget state from disk."""
        if not self._state_file.exists():
            self._reset_day()
            self._reset_month()
            return

        try:
            data = json.loads(self._state_file.read_text())
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            month = datetime.now(timezone.utc).strftime("%Y-%m")

            if data.get("day") == today:
                self._daily_spend = data.get("daily_spend", 0.0)
                self._daily_requests = data.get("daily_requests", 0)
                self._current_day = today
            else:
                self._reset_day()

            if data.get("month") == month:
                self._monthly_spend = data.get("monthly_spend", 0.0)
                self._monthly_requests = data.get("monthly_requests", 0)
                self._current_month = month
            else:
                self._reset_month()

            self._model_spend = data.get("model_spend", {})
            self._model_requests = data.get("model_requests", {})

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Budget state file corrupt, resetting: %s", e)
            self._reset_day()
            self._reset_month()

    def _reset_day(self) -> None:
        self._daily_spend = 0.0
        self._daily_requests = 0
        self._current_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._daily_warn_sent = False
        self._daily_limit_sent = False

    def _reset_month(self) -> None:
        self._monthly_spend = 0.0
        self._monthly_requests = 0
        self._current_month = datetime.now(timezone.utc).strftime("%Y-%m")
        self._model_spend = {}
        self._model_requests = {}
        self._monthly_warn_sent = False
        self._monthly_limit_sent = False

    def _save_state(self) -> None:
        """Persist current budget state to disk (atomic write via rename)."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "day": self._current_day,
            "month": self._current_month,
            "daily_spend": round(self._daily_spend, 6),
            "daily_requests": self._daily_requests,
            "monthly_spend": round(self._monthly_spend, 6),
            "monthly_requests": self._monthly_requests,
            "model_spend": {k: round(v, 6) for k, v in self._model_spend.items()},
            "model_requests": dict(self._model_requests),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        # Atomic write: write to temp file, then rename to prevent corruption
        import os
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self._state_file.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, str(self._state_file))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> Dict[str, Any]:
        """Record a completed request's cost. Returns budget status.

        Returns dict with keys: cost, daily_spend, monthly_spend, alerts.
        """
        cost = estimate_cost(model, prompt_tokens, completion_tokens) or 0.0

        with self._lock:
            # Check for day/month rollover
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            month = datetime.now(timezone.utc).strftime("%Y-%m")
            if today != self._current_day:
                self._reset_day()
            if month != self._current_month:
                self._reset_month()

            self._daily_spend += cost
            self._monthly_spend += cost
            self._daily_requests += 1
            self._monthly_requests += 1

            self._model_spend[model] = self._model_spend.get(model, 0.0) + cost
            self._model_requests[model] = self._model_requests.get(model, 0) + 1

            alerts = self._check_alerts()

            # Save every 10 requests to avoid excessive IO,
            # OR immediately on alerts to prevent data loss
            if alerts or self._daily_requests % 10 == 0:
                self._save_state()

            return {
                "cost": round(cost, 6),
                "daily_spend": round(self._daily_spend, 4),
                "monthly_spend": round(self._monthly_spend, 4),
                "daily_requests": self._daily_requests,
                "monthly_requests": self._monthly_requests,
                "alerts": alerts,
            }

    def _check_alerts(self) -> list[str]:
        """Check budgets and return any new alerts."""
        alerts = []

        if self.daily_budget:
            ratio = self._daily_spend / self.daily_budget
            if ratio >= 1.0 and not self._daily_limit_sent:
                self._daily_limit_sent = True
                msg = f"Daily budget exceeded: ${self._daily_spend:.4f} / ${self.daily_budget:.2f}"
                alerts.append(msg)
                logger.warning("🚨 %s", msg)
            elif ratio >= self.warn_threshold and not self._daily_warn_sent:
                self._daily_warn_sent = True
                msg = f"Daily budget warning: ${self._daily_spend:.4f} / ${self.daily_budget:.2f} ({ratio:.0%})"
                alerts.append(msg)
                logger.warning("⚠️ %s", msg)

        if self.monthly_budget:
            ratio = self._monthly_spend / self.monthly_budget
            if ratio >= 1.0 and not self._monthly_limit_sent:
                self._monthly_limit_sent = True
                msg = f"Monthly budget exceeded: ${self._monthly_spend:.4f} / ${self.monthly_budget:.2f}"
                alerts.append(msg)
                logger.warning("🚨 %s", msg)
            elif ratio >= self.warn_threshold and not self._monthly_warn_sent:
                self._monthly_warn_sent = True
                msg = f"Monthly budget warning: ${self._monthly_spend:.4f} / ${self.monthly_budget:.2f} ({ratio:.0%})"
                alerts.append(msg)
                logger.warning("⚠️ %s", msg)

        # Deliver alerts via configured channels
        for msg in alerts:
            self._deliver_alert(msg)

        return alerts

    def _deliver_alert(self, message: str) -> None:
        """Send an alert via stdout and/or webhook."""
        if self.stdout_alerts:
            print(f"[NadirClaw ALERT] {message}", flush=True)

        if self.webhook_url:
            payload = {
                "source": "nadirclaw",
                "type": "budget_alert",
                "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "daily_spend": round(self._daily_spend, 4),
                "daily_budget": self.daily_budget,
                "monthly_spend": round(self._monthly_spend, 4),
                "monthly_budget": self.monthly_budget,
            }
            # Fire-and-forget in background thread to avoid blocking requests
            Thread(
                target=_send_webhook,
                args=(self.webhook_url, payload),
                daemon=True,
            ).start()

    def get_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        with self._lock:
            return {
                "daily_spend": round(self._daily_spend, 4),
                "daily_budget": self.daily_budget,
                "daily_remaining": round(self.daily_budget - self._daily_spend, 4) if self.daily_budget else None,
                "daily_requests": self._daily_requests,
                "monthly_spend": round(self._monthly_spend, 4),
                "monthly_budget": self.monthly_budget,
                "monthly_remaining": round(self.monthly_budget - self._monthly_spend, 4) if self.monthly_budget else None,
                "monthly_requests": self._monthly_requests,
                "top_models": sorted(
                    [
                        {"model": m, "spend": round(s, 4), "requests": self._model_requests.get(m, 0)}
                        for m, s in self._model_spend.items()
                    ],
                    key=lambda x: x["spend"],
                    reverse=True,
                )[:10],
            }

    def flush(self) -> None:
        """Force-save state to disk."""
        with self._lock:
            self._save_state()


# ---------------------------------------------------------------------------
# Global budget tracker (lazy init from env vars)
# ---------------------------------------------------------------------------

_budget_tracker: Optional[BudgetTracker] = None
_budget_init_lock = Lock()


def get_budget_tracker() -> BudgetTracker:
    """Get the global budget tracker, initializing from env vars if needed."""
    global _budget_tracker
    if _budget_tracker is None:
        with _budget_init_lock:
            if _budget_tracker is None:
                import os
                daily = os.getenv("NADIRCLAW_DAILY_BUDGET")
                monthly = os.getenv("NADIRCLAW_MONTHLY_BUDGET")
                warn = float(os.getenv("NADIRCLAW_BUDGET_WARN_THRESHOLD", "0.8"))
                webhook = os.getenv("NADIRCLAW_BUDGET_WEBHOOK_URL")
                stdout = os.getenv("NADIRCLAW_BUDGET_STDOUT_ALERTS", "").lower() in ("1", "true", "yes")
                _budget_tracker = BudgetTracker(
                    daily_budget=float(daily) if daily else None,
                    monthly_budget=float(monthly) if monthly else None,
                    warn_threshold=warn,
                    webhook_url=webhook,
                    stdout_alerts=stdout,
                )
    return _budget_tracker
