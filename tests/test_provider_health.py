"""Tests for provider health tracking."""

from nadirclaw.provider_health import ProviderHealthTracker


def test_health_failure_enters_cooldown_and_reorders_candidates():
    now = [1000.0]
    tracker = ProviderHealthTracker(
        failure_threshold=2,
        cooldown_seconds=30,
        now_func=lambda: now[0],
    )

    tracker.record_failure("model-a", "ReadTimeout", "timed out")
    assert tracker.ordered_candidates(["model-a", "model-b"]) == ["model-a", "model-b"]

    tracker.record_failure("model-a", "ConnectTimeout", "connect timed out")
    assert tracker.ordered_candidates(["model-a", "model-b"]) == ["model-b", "model-a"]
    assert tracker.snapshot()["models"]["model-a"]["status"] == "cooling_down"

    now[0] = 1031.0
    assert tracker.ordered_candidates(["model-a", "model-b"]) == ["model-a", "model-b"]


def test_rate_limit_does_not_trip_health_bit():
    tracker = ProviderHealthTracker(failure_threshold=1, cooldown_seconds=30)

    tracker.record_failure("model-a", "RateLimitExhausted", "rate limited")

    assert tracker.ordered_candidates(["model-a", "model-b"]) == ["model-a", "model-b"]
    snapshot = tracker.snapshot()["models"]["model-a"]
    assert snapshot["recent_failures"] == 0
    assert snapshot["status"] == "healthy"
    assert snapshot["last_error"]["error_type"] == "RateLimitExhausted"


def test_success_resets_cooldown():
    tracker = ProviderHealthTracker(failure_threshold=1, cooldown_seconds=30)

    tracker.record_failure("model-a", "ReadTimeout", "timed out")
    assert tracker.ordered_candidates(["model-a", "model-b"]) == ["model-b", "model-a"]

    tracker.record_success("model-a")

    assert tracker.ordered_candidates(["model-a", "model-b"]) == ["model-a", "model-b"]
    snapshot = tracker.snapshot()["models"]["model-a"]
    assert snapshot["recent_successes"] == 1
    assert snapshot["status"] == "healthy"
