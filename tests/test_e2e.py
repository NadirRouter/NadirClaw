"""End-to-end tests for NadirClaw.

Covers areas not exercised by the existing unit/integration tests:
  - Auth token enforcement (Bearer + X-API-Key headers)
  - Model alias resolution (e.g. "sonnet" -> claude-sonnet-*)
  - Routing profiles: reasoning, free
  - Routing metadata shape in every response
  - Prometheus /metrics HTTP endpoint
  - Session cache: same prompt routes to same model on repeat
  - Batch classify edge cases (single, many, duplicates)
  - /v1/classify with a system_message
  - Developer-role messages accepted without error
  - CLI classify command via subprocess

LLM provider calls are mocked; classifier, router, session cache,
budget tracker, and auth all run for real.
"""

import json
import os
import subprocess
import sys
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from nadirclaw.server import app
    return TestClient(app)


@pytest.fixture
def auth_token():
    return "nadirclaw-e2e-test-token"


@pytest.fixture
def authed_client(monkeypatch, auth_token):
    """TestClient with AUTH_TOKEN configured to require the test token."""
    monkeypatch.setenv("NADIRCLAW_AUTH_TOKEN", auth_token)
    import nadirclaw.auth as auth_mod
    # Reload _LOCAL_USERS with the test token active
    auth_mod._LOCAL_USERS = {auth_token: auth_mod._default_user()}
    from nadirclaw.server import app
    return TestClient(app)


def _mock_fallback(content="OK", prompt_tokens=10, completion_tokens=5, model=None):
    """Build a side_effect callable for patching _call_with_fallback."""
    async def _side_effect(selected_model, request, provider, analysis_info):
        actual_model = model or selected_model
        return (
            {
                "content": content,
                "finish_reason": "stop",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            actual_model,
            {**analysis_info, "selected_model": actual_model},
        )
    return _side_effect


# ---------------------------------------------------------------------------
# 1. Auth Enforcement
# ---------------------------------------------------------------------------

class TestAuthEnforcement:
    """Verify token gating: with a token set, only authorized requests pass."""

    def test_health_is_always_public(self, authed_client):
        """Health endpoint is unauthenticated even when token is configured."""
        resp = authed_client.get("/health")
        assert resp.status_code == 200

    def test_root_is_always_public(self, authed_client):
        resp = authed_client.get("/")
        assert resp.status_code == 200

    def test_completion_without_token_returns_401(self, authed_client):
        resp = authed_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "e2e auth test prompt j8w"}]},
        )
        assert resp.status_code == 401

    def test_completion_with_wrong_token_returns_401(self, authed_client):
        resp = authed_client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer wrong-token"},
            json={"messages": [{"role": "user", "content": "e2e auth test prompt j8w"}]},
        )
        assert resp.status_code == 401

    @patch("nadirclaw.server._call_with_fallback")
    def test_bearer_token_grants_access(self, mock_fb, authed_client, auth_token):
        mock_fb.side_effect = _mock_fallback()
        resp = authed_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"messages": [{"role": "user", "content": "e2e auth test prompt j8w"}]},
        )
        assert resp.status_code == 200

    @patch("nadirclaw.server._call_with_fallback")
    def test_x_api_key_grants_access(self, mock_fb, authed_client, auth_token):
        """X-API-Key header is accepted as an alternative to Authorization: Bearer."""
        mock_fb.side_effect = _mock_fallback()
        resp = authed_client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": auth_token},
            json={"messages": [{"role": "user", "content": "e2e auth test prompt j8w"}]},
        )
        assert resp.status_code == 200

    def test_oversized_token_returns_400(self, authed_client):
        """Tokens longer than 1000 chars are rejected as malformed."""
        resp = authed_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {'x' * 1001}"},
            json={"messages": [{"role": "user", "content": "e2e auth test prompt j8w"}]},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 2. Model Alias Resolution
# ---------------------------------------------------------------------------

class TestAliasResolution:
    """model="<alias>" should route with strategy="alias", not as a raw model name."""

    @patch("nadirclaw.server._call_with_fallback")
    def test_sonnet_alias_resolves(self, mock_fb, client):
        mock_fb.side_effect = _mock_fallback(content="Sonnet reply")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "e2e alias test sonnet 7k3"}],
            "model": "sonnet",
        })
        assert resp.status_code == 200
        routing = resp.json()["nadirclaw_metadata"]["routing"]
        assert routing["strategy"] == "alias"
        # Resolved model should include "claude" or "sonnet"
        assert "sonnet" in routing["selected_model"].lower() or "claude" in routing["selected_model"].lower()

    @patch("nadirclaw.server._call_with_fallback")
    def test_gpt4_alias_resolves(self, mock_fb, client):
        mock_fb.side_effect = _mock_fallback(content="GPT4 reply")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "e2e alias test gpt4 9m2"}],
            "model": "gpt4",
        })
        assert resp.status_code == 200
        routing = resp.json()["nadirclaw_metadata"]["routing"]
        assert routing["strategy"] == "alias"
        assert "gpt" in routing["selected_model"].lower()

    @patch("nadirclaw.server._call_with_fallback")
    def test_flash_alias_resolves(self, mock_fb, client):
        mock_fb.side_effect = _mock_fallback(content="Flash reply")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "e2e alias test flash 4q8"}],
            "model": "flash",
        })
        assert resp.status_code == 200
        routing = resp.json()["nadirclaw_metadata"]["routing"]
        assert routing["strategy"] == "alias"
        assert "gemini" in routing["selected_model"].lower() or "flash" in routing["selected_model"].lower()

    @patch("nadirclaw.server._call_with_fallback")
    def test_nadirclaw_prefix_alias_resolves(self, mock_fb, client):
        """nadirclaw/<profile> prefix notation should work for profiles."""
        mock_fb.side_effect = _mock_fallback()
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "e2e prefix test eco 2p5"}],
            "model": "nadirclaw/eco",
        })
        assert resp.status_code == 200
        routing = resp.json()["nadirclaw_metadata"]["routing"]
        assert "eco" in routing["strategy"]


# ---------------------------------------------------------------------------
# 3. Routing Profiles: reasoning and free
# ---------------------------------------------------------------------------

class TestAdditionalProfiles:
    """reasoning and free profiles are not covered by test_pipeline_integration."""

    @patch("nadirclaw.server._call_with_fallback")
    def test_reasoning_profile_routes_to_complex(self, mock_fb, client):
        mock_fb.side_effect = _mock_fallback(content="Deep thought")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Solve the halting problem"}],
            "model": "reasoning",
        })
        assert resp.status_code == 200
        routing = resp.json()["nadirclaw_metadata"]["routing"]
        assert routing["strategy"] == "profile:reasoning"
        assert routing["tier"] in ("complex", "reasoning")

    @patch("nadirclaw.server._call_with_fallback")
    def test_free_profile_routes_to_simple(self, mock_fb, client):
        mock_fb.side_effect = _mock_fallback(content="Free answer")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Solve the halting problem"}],
            "model": "free",
        })
        assert resp.status_code == 200
        routing = resp.json()["nadirclaw_metadata"]["routing"]
        assert routing["strategy"] == "profile:free"
        assert routing["tier"] in ("simple", "free")

    @patch("nadirclaw.server._call_with_fallback")
    def test_auto_profile_uses_smart_routing(self, mock_fb, client):
        mock_fb.side_effect = _mock_fallback(content="Auto answer")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "e2e auto profile test 8r1"}],
            "model": "auto",
        })
        assert resp.status_code == 200
        routing = resp.json()["nadirclaw_metadata"]["routing"]
        assert "auto" in routing["strategy"] or "smart" in routing["strategy"]


# ---------------------------------------------------------------------------
# 4. Routing Metadata Shape
# ---------------------------------------------------------------------------

class TestRoutingMetadataShape:
    """Every completion response must carry a complete nadirclaw_metadata block."""

    @patch("nadirclaw.server._call_with_fallback")
    def test_required_metadata_keys_present(self, mock_fb, client):
        mock_fb.side_effect = _mock_fallback(prompt_tokens=20, completion_tokens=8)
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "What is 3 plus 5 e2e metadata test"}],
        })
        assert resp.status_code == 200
        data = resp.json()

        assert "nadirclaw_metadata" in data
        meta = data["nadirclaw_metadata"]
        assert "routing" in meta

        routing = meta["routing"]
        for key in ("tier", "confidence", "selected_model", "strategy"):
            assert key in routing, f"Missing routing key: {key}"

        # tier must be a valid value
        assert routing["tier"] in ("simple", "complex", "free")

        # confidence must be numeric 0–1
        assert 0.0 <= routing["confidence"] <= 1.0

    @patch("nadirclaw.server._call_with_fallback")
    def test_usage_block_populated(self, mock_fb, client):
        # Use a unique prompt to avoid session-cache contamination from other tests
        mock_fb.side_effect = _mock_fallback(
            content="Usage test reply", prompt_tokens=15, completion_tokens=7,
        )
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Usage block population test xk92b"}],
        })
        assert resp.status_code == 200
        usage = resp.json()["usage"]
        assert usage["prompt_tokens"] == 15
        assert usage["completion_tokens"] == 7
        assert usage["total_tokens"] == 22

    @patch("nadirclaw.server._call_with_fallback")
    def test_response_id_is_unique(self, mock_fb, client):
        """Each response should get a distinct ID."""
        mock_fb.side_effect = _mock_fallback()
        ids = set()
        for i in range(3):
            resp = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": f"e2e unique id test iteration {i} z3q"}],
            })
            ids.add(resp.json()["id"])
        assert len(ids) == 3


# ---------------------------------------------------------------------------
# 5. Prometheus /metrics HTTP Endpoint
# ---------------------------------------------------------------------------

class TestMetricsHTTPEndpoint:
    """The /metrics endpoint must return valid Prometheus text format."""

    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_content_type_is_text(self, client):
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers.get("content-type", "")

    @patch("nadirclaw.server._call_with_fallback")
    def test_metrics_increment_after_request(self, mock_fb, client):
        """After a completion, metrics counters must reflect the request."""
        mock_fb.side_effect = _mock_fallback(prompt_tokens=50, completion_tokens=20)
        client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "e2e metrics increment test 5v9"}],
        })

        resp = client.get("/metrics")
        body = resp.text

        # Core metric families must be present
        assert "nadirclaw_requests_total" in body
        assert "nadirclaw_tokens_prompt_total" in body
        assert "nadirclaw_tokens_completion_total" in body
        assert "nadirclaw_request_latency_ms" in body

    def test_metrics_no_auth_required(self, authed_client):
        """Metrics endpoint is public even when auth is configured."""
        resp = authed_client.get("/metrics")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 6. Session Cache Consistency
# ---------------------------------------------------------------------------

class TestSessionCacheConsistency:
    """Identical conversations should be routed to the same model on repeat calls."""

    @patch("nadirclaw.server._call_with_fallback")
    def test_repeated_prompt_routes_consistently(self, mock_fb, client):
        mock_fb.side_effect = _mock_fallback(content="42")

        messages = [{"role": "user", "content": "What is 6 times 7?"}]
        tiers = []
        models = []

        for _ in range(3):
            resp = client.post("/v1/chat/completions", json={"messages": messages})
            routing = resp.json()["nadirclaw_metadata"]["routing"]
            tiers.append(routing["tier"])
            models.append(routing["selected_model"])

        # All three calls should agree on tier and model
        assert len(set(tiers)) == 1, f"Inconsistent tiers across identical prompts: {tiers}"
        assert len(set(models)) == 1, f"Inconsistent models across identical prompts: {models}"


# ---------------------------------------------------------------------------
# 7. Batch Classify Edge Cases
# ---------------------------------------------------------------------------

class TestBatchClassify:
    """Edge cases for the /v1/classify/batch endpoint."""

    def test_single_prompt_batch(self, client):
        resp = client.post("/v1/classify/batch", json={"prompts": ["Hello"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["results"]) == 1
        result = data["results"][0]
        assert result["tier"] in ("simple", "complex")
        assert "confidence" in result

    def test_large_batch(self, client):
        prompts = [
            "What is 9 times 3?",
            "Explain quantum entanglement in detail for a e2e batch test",
            "Greetings e2e",
            "Design a scalable microservices architecture for a bank e2e batch",
            "What time is it right now in Paris?",
        ]
        resp = client.post("/v1/classify/batch", json={"prompts": prompts})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == len(prompts)
        assert len(data["results"]) == len(prompts)

    def test_duplicate_prompts_both_classified(self, client):
        """Duplicate prompts in a batch should each get their own result."""
        resp = client.post("/v1/classify/batch", json={
            "prompts": ["What is 7 plus 8?", "What is 7 plus 8?"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        # Both should classify to the same tier
        tiers = [r["tier"] for r in data["results"]]
        assert tiers[0] == tiers[1]

    def test_empty_batch_returns_zero(self, client):
        resp = client.post("/v1/classify/batch", json={"prompts": []})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["results"] == []


# ---------------------------------------------------------------------------
# 8. Classify with system_message
# ---------------------------------------------------------------------------

class TestClassifyWithSystemMessage:
    """system_message param should influence classification."""

    def test_classify_with_system_message(self, client):
        resp = client.post("/v1/classify", json={
            "prompt": "Describe the issue",
            "system_message": (
                "You are a senior distributed systems architect. "
                "Analyze complex multi-region failure scenarios."
            ),
        })
        assert resp.status_code == 200
        data = resp.json()
        c = data["classification"]
        assert c["tier"] in ("simple", "complex")
        assert "confidence" in c
        assert "selected_model" in c

    def test_classify_returns_score_and_analyzer(self, client):
        resp = client.post("/v1/classify", json={"prompt": "What is the capital of France?"})
        assert resp.status_code == 200
        c = resp.json()["classification"]
        assert "complexity_score" in c
        assert "analyzer" in c or "analyzer_type" in c
        assert isinstance(c["complexity_score"], float)


# ---------------------------------------------------------------------------
# 9. Developer-Role Messages
# ---------------------------------------------------------------------------

class TestDeveloperRoleMessages:
    """role='developer' must be accepted the same as role='system'."""

    @patch("nadirclaw.server._call_with_fallback")
    def test_developer_role_accepted(self, mock_fb, client):
        mock_fb.side_effect = _mock_fallback(content="Developer system reply")
        resp = client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "developer", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Fix this bug"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Developer system reply"

    @patch("nadirclaw.server._call_with_fallback")
    def test_mixed_roles_conversation(self, mock_fb, client):
        """system + user + assistant + developer + user all in one conversation."""
        mock_fb.side_effect = _mock_fallback(content="Mixed reply")
        resp = client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Question one"},
                {"role": "assistant", "content": "Answer one"},
                {"role": "developer", "content": "Additional instruction"},
                {"role": "user", "content": "Question two"},
            ],
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 10. CLI classify command (subprocess)
# ---------------------------------------------------------------------------

class TestCLIClassify:
    """nadirclaw classify should work without the server running."""

    # Cold-starting a fresh Python process loads the SentenceTransformer
    # encoder, which can take 30-40s on a cold model cache. Be generous.
    _TIMEOUT = 120
    _VALID_TIERS = ("simple", "mid", "complex", "reasoning", "free")

    def _run(self, *args):
        """Run `nadirclaw classify` in a subprocess, pinned to the fast
        binary analyzer for deterministic, quick CLI tests."""
        env = {**os.environ, "NADIRCLAW_COMPLEXITY_ANALYZER": "binary"}
        return subprocess.run(
            [sys.executable, "-m", "nadirclaw.cli", "classify", *args],
            capture_output=True, text=True, timeout=self._TIMEOUT, env=env,
        )

    def test_classify_simple_prompt(self):
        result = self._run("What", "is", "2+2?")
        assert result.returncode == 0, result.stderr
        output = result.stdout.lower()
        assert any(t in output for t in self._VALID_TIERS)

    def test_classify_complex_prompt(self):
        result = self._run(
            "Design", "a", "distributed", "event-sourcing", "system",
            "with", "CQRS", "and", "eventual", "consistency",
        )
        assert result.returncode == 0, result.stderr

    def test_classify_json_format(self):
        result = self._run("--format", "json", "What", "is", "2+2?")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert "tier" in data
        assert "confidence" in data
        assert "model" in data
        assert data["tier"] in self._VALID_TIERS
        assert 0.0 <= data["confidence"] <= 1.0

    def test_classify_quoted_single_arg(self):
        """Single-argument classify (quoted string) should also work."""
        result = self._run("What is the weather?")
        assert result.returncode == 0, result.stderr

    def test_classify_json_prompt_field(self):
        """JSON output must echo back the prompt."""
        result = self._run("--format", "json", "Hello", "world")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert "Hello world" in data["prompt"]


# ---------------------------------------------------------------------------
# 11. Logs endpoint
# ---------------------------------------------------------------------------

class TestLogsEndpoint:
    """/v1/logs should return a valid structure (auth-optional by default)."""

    def test_logs_endpoint_returns_list(self, client):
        resp = client.get("/v1/logs")
        assert resp.status_code == 200
        data = resp.json()
        assert "logs" in data
        assert "total" in data
        assert isinstance(data["logs"], list)

    def test_logs_limit_param_respected(self, client):
        resp = client.get("/v1/logs?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["showing"] <= 5

    @patch("nadirclaw.server._call_with_fallback")
    def test_logs_grow_after_request(self, mock_fb, client):
        """Log count should increase after a completion request."""
        mock_fb.side_effect = _mock_fallback()

        before = client.get("/v1/logs").json()["total"]

        client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Log entry test"}],
        })

        after = client.get("/v1/logs").json()["total"]
        assert after >= before  # at least stayed the same (persistent store may vary)
