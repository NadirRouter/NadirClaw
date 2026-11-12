"""Tests for the `nadirclaw test` command's auth path (#74).

Subscription/OAuth tokens (sk-ant-oat*) must be probed with Authorization:
Bearer + the oauth beta header — the same path the server uses — rather than
LiteLLM's x-api-key path, which Anthropic restricts for subscription tokens.
"""

from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from nadirclaw.cli import main


class _FakeHTTPResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"content": [{"type": "text", "text": "ok"}]}


def test_oauth_token_uses_bearer_path_not_litellm():
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json
        return _FakeHTTPResponse()

    with patch("nadirclaw.credentials.get_credential", return_value="sk-ant-oat01-subscription"), \
         patch("nadirclaw.settings.settings") as mock_settings, \
         patch("httpx.post", side_effect=_fake_post), \
         patch("litellm.completion") as mock_litellm:
        mock_settings.SIMPLE_MODEL = "claude-haiku-4-5"
        mock_settings.COMPLEX_MODEL = "claude-opus-4-6"
        result = CliRunner().invoke(main, ["test"])

    assert result.exit_code == 0, result.output
    # LiteLLM's x-api-key path must NOT be used for an sk-ant-oat token.
    mock_litellm.assert_not_called()
    # Correct OAuth headers were sent.
    assert captured["headers"]["Authorization"] == "Bearer sk-ant-oat01-subscription"
    assert "oauth-2025-04-20" in captured["headers"]["anthropic-beta"]
    assert captured["url"].endswith("/v1/messages")
    # Both tiers were probed (simple + complex).
    assert "claude-haiku-4-5" in result.output
    assert "claude-opus-4-6" in result.output


def test_api_key_token_uses_litellm_path():
    """A normal API key (sk-ant-api*) keeps using LiteLLM, not the OAuth shim."""
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = "ok"

    with patch("nadirclaw.credentials.get_credential", return_value="sk-ant-api03-regularkey"), \
         patch("nadirclaw.settings.settings") as mock_settings, \
         patch("httpx.post") as mock_post, \
         patch("litellm.completion", return_value=fake_resp) as mock_litellm:
        mock_settings.SIMPLE_MODEL = "claude-haiku-4-5"
        mock_settings.COMPLEX_MODEL = "claude-haiku-4-5"  # same → one probe
        result = CliRunner().invoke(main, ["test"])

    assert result.exit_code == 0, result.output
    mock_post.assert_not_called()
    mock_litellm.assert_called_once()


def _ok_resp():
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = "ok"
    return fake_resp


def test_custom_api_base_is_passed_to_probe():
    """NADIRCLAW_API_BASE must reach the probe, like the server's calls (#116)."""
    with patch("nadirclaw.credentials.get_credential", return_value=None), \
         patch("nadirclaw.settings.settings") as mock_settings, \
         patch("litellm.completion", return_value=_ok_resp()) as mock_litellm:
        mock_settings.SIMPLE_MODEL = "openai/qwen/qwen3.7-flash"
        mock_settings.COMPLEX_MODEL = "gemini/gemini-2.5-pro"
        mock_settings.API_BASE = "https://gateway.example/v1"
        result = CliRunner().invoke(main, ["test"])

    assert result.exit_code == 0, result.output
    calls = {c.kwargs["model"]: c.kwargs for c in mock_litellm.call_args_list}
    assert calls["openai/qwen/qwen3.7-flash"]["api_base"] == "https://gateway.example/v1"
    # Gemini is called natively by the server, so it must not get the gateway base.
    assert "api_base" not in calls["gemini/gemini-2.5-pro"]
