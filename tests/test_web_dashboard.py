"""Tests for web_dashboard model_stats aggregation."""

from nadirclaw.web_dashboard import compute_model_stats, _classify_error_type


def _completion(selected_model: str, fallback_reasons=None, **extra):
    """Build a synthetic completion log entry."""
    entry = {
        "type": "completion",
        "status": "ok",
        "selected_model": selected_model,
        "tier": "complex",
        "fallback_reasons": fallback_reasons or [],
    }
    entry.update(extra)
    return entry


class TestClassifyErrorType:
    def test_rate_limit_class_names(self):
        assert _classify_error_type("RateLimitExhausted") == "rate_limit"
        assert _classify_error_type("RateLimitError") == "rate_limit"

    def test_timeout_class_names(self):
        assert _classify_error_type("ReadTimeout") == "timeout"
        assert _classify_error_type("ConnectTimeout") == "timeout"
        assert _classify_error_type("Timeout") == "timeout"
        assert _classify_error_type("TimeoutError") == "timeout"

    def test_connection_class_names(self):
        assert _classify_error_type("ConnectError") == "connection_error"
        assert _classify_error_type("APIConnectionError") == "connection_error"

    def test_disconnected_class_names(self):
        assert _classify_error_type("RemoteProtocolError") == "disconnected"
        assert _classify_error_type("EndOfStream") == "disconnected"

    def test_server_error_suffix(self):
        assert _classify_error_type("InternalServerError") == "server_error"
        assert _classify_error_type("ServiceUnavailableServerError") == "server_error"

    def test_unknown_falls_through(self):
        assert _classify_error_type("") == "other"
        assert _classify_error_type("SomeRandomThing") == "other"

    def test_no_substring_false_positives(self):
        # Previous substring-match logic classified anything containing
        # "connection" as connection_error — including words like
        # "ConnectionPoolTimeoutError" which is really a timeout. We only
        # match on exact class names now, so unknown-but-connection-shaped
        # names fall through to "other" rather than being miscategorised.
        assert _classify_error_type("ConnectionPoolTimeoutError") == "other"


class TestComputeModelStats:
    def test_single_success(self):
        stats = compute_model_stats([_completion("m1")])
        assert stats == {"m1": {"success": 1, "fail": 0, "fail_reasons": {}}}

    def test_single_fallback_reason(self):
        entry = _completion(
            "m2",
            fallback_reasons=[{"model": "m1", "error_type": "RateLimitExhausted", "message": "429"}],
        )
        stats = compute_model_stats([entry])
        assert stats["m1"]["fail"] == 1
        assert stats["m1"]["success"] == 0
        assert stats["m1"]["fail_reasons"] == {"rate_limit": 1}
        # m2 succeeded after fallback
        assert stats["m2"]["success"] == 1
        assert stats["m2"]["fail"] == 0

    def test_multiple_reasons_bucketed(self):
        entries = [
            _completion("m3", fallback_reasons=[
                {"model": "m1", "error_type": "RateLimitExhausted", "message": "x"},
                {"model": "m2", "error_type": "ReadTimeout", "message": "y"},
            ]),
            _completion("m3", fallback_reasons=[
                {"model": "m1", "error_type": "ConnectError", "message": "z"},
            ]),
        ]
        stats = compute_model_stats(entries)
        assert stats["m1"]["fail"] == 2
        assert stats["m1"]["fail_reasons"] == {"rate_limit": 1, "connection_error": 1}
        assert stats["m2"]["fail"] == 1
        assert stats["m2"]["fail_reasons"] == {"timeout": 1}
        assert stats["m3"]["success"] == 2

    def test_unknown_error_type_lands_in_other(self):
        entry = _completion(
            "m2",
            fallback_reasons=[{"model": "m1", "error_type": "MysteryError", "message": "?"}],
        )
        stats = compute_model_stats([entry])
        assert stats["m1"]["fail_reasons"] == {"other": 1}

    def test_missing_error_type_is_other(self):
        # Backward compat: older logs may omit error_type entirely.
        entry = _completion(
            "m2",
            fallback_reasons=[{"model": "m1"}],
        )
        stats = compute_model_stats([entry])
        assert stats["m1"]["fail_reasons"] == {"other": 1}

    def test_no_fallback_reasons_key(self):
        # Ensure None/missing fallback_reasons doesn't explode.
        entry = {"type": "completion", "status": "ok", "selected_model": "m1"}
        stats = compute_model_stats([entry])
        assert stats == {"m1": {"success": 1, "fail": 0, "fail_reasons": {}}}

    def test_success_rate_scenario_from_review(self):
        """Reviewer's scenario: 50 requests, 2 failed on primary m1 then
        succeeded on m2. Expect m1: 48 success, 2 fail; m2: 2 success, 0 fail."""
        entries = []
        for _ in range(48):
            entries.append(_completion("m1"))
        for _ in range(2):
            entries.append(_completion(
                "m2",
                fallback_reasons=[{"model": "m1", "error_type": "RateLimitExhausted", "message": "429"}],
            ))
        stats = compute_model_stats(entries)
        assert stats["m1"] == {"success": 48, "fail": 2, "fail_reasons": {"rate_limit": 2}}
        assert stats["m2"] == {"success": 2, "fail": 0, "fail_reasons": {}}
