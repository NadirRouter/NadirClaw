"""CCR — Compress-Cache-Retrieve fetch-back loop (native, deterministic).

The biggest prompt-token win is *offloading* large message content out of the
prompt and letting the model pull it back on demand. Headroom does this, but its
store is driven by a ContextVar written from a worker thread with a
non-deterministic key — reliable only inside their own proxy. So NadirClaw owns
the loop natively:

  1. ``offload_messages`` moves oversized content into a ``{hash: original}`` map
     we control, leaving a short preview + a ``nadir_retrieve(hash=...)`` marker.
  2. ``retrieve_tool_def`` is injected so the model knows it can fetch the rest.
  3. ``resolve_loop`` intercepts the model's retrieve calls, serves the exact
     original from the map, and continues — so nothing is ever lost.

This composes with the optimizer's lossless transforms (the inline content still
gets compressed); offload is the last, most aggressive tier and is fully
reversible because we keep every original byte.
"""

from __future__ import annotations

import hashlib
import json
import re

from nadirclaw.optimize import _estimate_tokens_str

RETRIEVE_TOOL_NAME = "nadir_retrieve"

# Default: offload non-user messages whose content exceeds this many tokens.
DEFAULT_MIN_OFFLOAD_TOKENS = 400
_OFFLOAD_ROLES = ("tool", "system", "assistant", "function")
_MARKER_RE = re.compile(r'hash=[\'"]?([a-f0-9]{8,})')


def retrieve_tool_def() -> dict:
    """OpenAI-format function tool the model calls to fetch offloaded content."""
    return {
        "type": "function",
        "function": {
            "name": RETRIEVE_TOOL_NAME,
            "description": (
                "Retrieve the full original content that was offloaded from the "
                "prompt to save tokens. Pass the hash from an offload marker like "
                "'[... offloaded ... retrieve full content with nadir_retrieve(hash=\"abc123\")]'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hash": {"type": "string", "description": "Hash from the offload marker."}
                },
                "required": ["hash"],
            },
        },
    }


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def offload_messages(
    messages: list[dict],
    *,
    min_tokens: int = DEFAULT_MIN_OFFLOAD_TOKENS,
    roles: tuple = _OFFLOAD_ROLES,
):
    """Move oversized message content into a captured map, leaving a retrieve marker.

    The user's turns are never offloaded. Returns ``(messages, captured, hashes)``
    where ``captured[hash]`` is the exact original content.
    """
    captured: dict[str, str] = {}
    hashes: list[str] = []
    out: list[dict] = []
    for m in messages:
        content = m.get("content")
        if (
            isinstance(content, str)
            and m.get("role") in roles
            and _estimate_tokens_str(content) >= min_tokens
        ):
            h = _hash(content)
            captured[h] = content
            preview = re.sub(r"\s+", " ", content[:140]).strip()
            tokens = _estimate_tokens_str(content)
            out.append({
                **m,
                "content": (
                    f"[{tokens} tokens offloaded to save context — preview: {preview}… — "
                    f'retrieve the full content with {RETRIEVE_TOOL_NAME}(hash="{h}")]'
                ),
            })
            if h not in hashes:
                hashes.append(h)
        else:
            out.append(m)
    return out, captured, hashes


def resolve(captured: dict, hash_key: str) -> "str | None":
    """Resolve an offload hash to its exact original content."""
    if not captured or not hash_key:
        return None
    return captured.get(hash_key)


def marker_hashes(messages: list[dict]) -> list[str]:
    """Return offload hashes referenced by markers in *messages* (order-preserving)."""
    seen: list[str] = []
    for h in _MARKER_RE.findall(json.dumps(messages)):
        if h not in seen:
            seen.append(h)
    return seen


def extract_retrieve_calls(response: dict) -> list[tuple[str, str]]:
    """Parse an OpenAI-format response for ``nadir_retrieve`` tool calls -> [(id, hash)]."""
    calls: list[tuple[str, str]] = []
    for choice in (response or {}).get("choices", []):
        message = choice.get("message", {}) or {}
        for tc in message.get("tool_calls", []) or []:
            fn = tc.get("function", {}) or {}
            if fn.get("name") != RETRIEVE_TOOL_NAME:
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            calls.append((tc.get("id", ""), (args or {}).get("hash", "")))
    return calls


def resolve_loop(messages, first_response, captured, call_llm, *, max_rounds: int = 3):
    """Server-side fetch-back loop.

    While the model asks for offloaded content, resolve each hash from *captured*,
    append the originals as tool messages, and re-invoke ``call_llm(messages)``
    until the model returns a final (non-retrieval) answer or *max_rounds* is hit.
    ``call_llm`` is injected so this is testable without a live provider.
    Returns ``(final_response, full_conversation)``.
    """
    convo = list(messages)
    response = first_response
    for _ in range(max_rounds):
        calls = extract_retrieve_calls(response)
        if not calls:
            return response, convo
        convo = convo + [response["choices"][0]["message"]]
        for tool_call_id, h in calls:
            original = resolve(captured, h)
            convo.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": RETRIEVE_TOOL_NAME,
                "content": original if original is not None else f"[retrieve failed: unknown hash {h}]",
            })
        response = call_llm(convo)
    return response, convo
