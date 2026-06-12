"""Tests for the native CCR (Compress-Cache-Retrieve) fetch-back loop.

Offload moves oversized non-user content out of the prompt behind a retrieve
handle, keeping the exact original in a map. The loop resolves the model's
retrieve calls so nothing is ever lost. All deterministic, no Headroom needed.
"""
import json

import pytest

import nadirclaw.ccr as ccr
import nadirclaw.optimize as o


def _big_tool_msgs():
    rows = [{"id": 1000 + i, "user": f"user{i}", "status": "active" if i % 4 else "suspended",
             "plan": "pro" if i % 5 == 0 else "free"} for i in range(60)]
    return [
        {"role": "system", "content": "You are a support assistant."},
        {"role": "user", "content": "How many users are suspended?"},
        {"role": "tool", "content": "get_users() ->\n" + json.dumps(rows, indent=2)},
    ]


# ---------------------------------------------------------------------------
# offload + resolve
# ---------------------------------------------------------------------------

def test_offload_shrinks_and_captures():
    msgs = _big_tool_msgs()
    before = o._estimate_tokens_messages(msgs)
    out, captured, hashes = ccr.offload_messages(msgs)
    after = o._estimate_tokens_messages(out)
    assert after < before * 0.5            # big reduction
    assert len(hashes) == 1                # the one big tool message
    assert hashes[0] in captured


def test_offload_is_byte_exact_recoverable():
    msgs = _big_tool_msgs()
    original = msgs[2]["content"]
    out, captured, hashes = ccr.offload_messages(msgs)
    assert ccr.resolve(captured, hashes[0]) == original   # exact bytes back


def test_user_message_never_offloaded():
    msgs = [{"role": "user", "content": "x" * 5000}]   # huge, but it's the user's turn
    out, captured, hashes = ccr.offload_messages(msgs)
    assert hashes == [] and out == msgs


def test_small_messages_not_offloaded():
    msgs = [{"role": "tool", "content": "short result"}]
    out, captured, hashes = ccr.offload_messages(msgs, min_tokens=400)
    assert hashes == []


def test_marker_carries_the_hash():
    out, captured, hashes = ccr.offload_messages(_big_tool_msgs())
    marker = out[2]["content"]
    assert f'hash="{hashes[0]}"' in marker
    assert ccr.RETRIEVE_TOOL_NAME in marker


# ---------------------------------------------------------------------------
# retrieve tool + response parsing
# ---------------------------------------------------------------------------

def test_retrieve_tool_def_shape():
    tool = ccr.retrieve_tool_def()
    assert tool["function"]["name"] == ccr.RETRIEVE_TOOL_NAME
    assert "hash" in tool["function"]["parameters"]["properties"]


def test_extract_retrieve_calls():
    resp = {"choices": [{"message": {"tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "nadir_retrieve", "arguments": json.dumps({"hash": "deadbeef"})}},
        {"id": "c2", "type": "function",
         "function": {"name": "other_tool", "arguments": "{}"}},
    ]}}]}
    assert ccr.extract_retrieve_calls(resp) == [("c1", "deadbeef")]


# ---------------------------------------------------------------------------
# full fetch-back loop (mock LLM, no provider)
# ---------------------------------------------------------------------------

def test_resolve_loop_recovers_data_and_answers():
    msgs = _big_tool_msgs()
    rows_suspended = sum(1 for i in range(60) if not (i % 4))  # status logic in _big_tool_msgs
    out, captured, hashes = ccr.offload_messages(msgs)

    def mock_llm(convo):
        tool_msgs = [m for m in convo if m.get("role") == "tool" and m.get("name") == "nadir_retrieve"]
        if not tool_msgs:  # round 1: ask to retrieve
            return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "nadir_retrieve", "arguments": json.dumps({"hash": hashes[0]})}}]}}]}
        data = tool_msgs[-1]["content"]      # round 2: answer from the REAL data
        return {"choices": [{"message": {"role": "assistant",
                                         "content": f'{data.count(chr(34) + "suspended" + chr(34))} suspended'}}]}

    final, convo = ccr.resolve_loop(out, mock_llm(out), captured, mock_llm)
    answer = final["choices"][0]["message"]["content"]
    assert str(rows_suspended) in answer            # model answered correctly from recovered data
    # the resolved tool message in the conversation is the exact original
    assert any(m.get("role") == "tool" and m.get("content") == msgs[2]["content"] for m in convo)


def test_resolve_loop_handles_unknown_hash():
    resp = {"choices": [{"message": {"tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "nadir_retrieve", "arguments": json.dumps({"hash": "nope"})}}]}}]}
    final, convo = ccr.resolve_loop([], resp, {}, lambda convo: {"choices": [{"message": {"content": "done"}}]})
    assert any("retrieve failed" in (m.get("content") or "") for m in convo)


# ---------------------------------------------------------------------------
# progressive offload stage
# ---------------------------------------------------------------------------

def test_progressive_offload_gated_off_by_default():
    r = o.compress_progressive(_big_tool_msgs(), target_tokens=200, max_stage="offload")
    assert r.offload_captured == {}
    assert "stage:offload" not in r.optimizations_applied


def test_progressive_offload_engages_and_is_recoverable():
    import re
    msgs = _big_tool_msgs()
    r = o.compress_progressive(msgs, target_tokens=200, max_stage="offload", allow_offload=True)
    assert "stage:offload" in r.optimizations_applied
    assert r.offload_captured
    # Offload captures the (losslessly) compressed content — not byte-identical to
    # the pretty original, but still complete: every user must be recoverable.
    recovered = "".join(r.offload_captured.values())
    assert len(set(re.findall(r"user\d+", recovered))) == 60
