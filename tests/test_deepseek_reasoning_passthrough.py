"""DeepSeek reasoning_content pass-back on tool-call turns.

DeepSeek's thinking-mode API rejects the follow-up round of a tool
conversation when the previous assistant turn's ``reasoning_content`` is not
sent back. The LiteLLM request builders used to preserve the field only for
MiniMax, so a DeepSeek tool conversation routed through NadirClaw failed on
its second round even though the client had stored and resent the value
(NadirClaw's own response side extracts it — see
tests/test_thinking_passthrough.py). These tests pin the pass-back for
DeepSeek models in both the streaming and non-streaming builders, and pin
that other providers still never see the field.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from nadirclaw.server import ChatCompletionRequest, _call_litellm, _stream_litellm

TOOL = {
    "type": "function",
    "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}},
}

MESSAGES = [
    {"role": "user", "content": "Look something up, thinking first."},
    {
        "role": "assistant",
        "content": None,
        "reasoning_content": "Keep this thought",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "Found it"},
]


def _request():
    return ChatCompletionRequest(model="auto", messages=MESSAGES, tools=[TOOL])


def _mock_response():
    msg = SimpleNamespace(content="Done", tool_calls=None, reasoning_content="Considered")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,provider",
    [
        ("deepseek/deepseek-reasoner", None),  # detect_provider has no deepseek pattern
        ("deepseek/deepseek-reasoner", "deepseek"),  # tier routing supplies the provider
        ("deepseek/deepseek-v4-pro", "deepseek"),
        ("deepseek-reasoner", None),  # bare model name
    ],
)
async def test_deepseek_reasoning_content_passed_back(model, provider):
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
        mock_comp.return_value = _mock_response()
        await _call_litellm(model, _request(), provider)

    sent = mock_comp.call_args[1]["messages"]
    assert sent[1]["reasoning_content"] == "Keep this thought"
    assert sent[1]["tool_calls"] == MESSAGES[1]["tool_calls"]
    assert sent[2]["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_minimax_reasoning_content_still_passed_back():
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
        mock_comp.return_value = _mock_response()
        await _call_litellm("minimax/MiniMax-M3", _request(), "minimax")

    sent = mock_comp.call_args[1]["messages"]
    assert sent[1]["reasoning_content"] == "Keep this thought"


@pytest.mark.asyncio
async def test_other_providers_do_not_receive_reasoning_content():
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
        mock_comp.return_value = _mock_response()
        await _call_litellm("gpt-5.2", _request(), "openai")

    sent = mock_comp.call_args[1]["messages"]
    assert "reasoning_content" not in sent[1]


@pytest.mark.asyncio
async def test_deepseek_reasoning_content_passed_back_streaming():
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)

        async def gen():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="Done", tool_calls=None, reasoning_content=None
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        return gen()

    with patch("litellm.acompletion", side_effect=fake_acompletion):
        async for _ in _stream_litellm("deepseek/deepseek-reasoner", _request(), "deepseek"):
            pass

    sent = captured["messages"]
    assert sent[1]["reasoning_content"] == "Keep this thought"
