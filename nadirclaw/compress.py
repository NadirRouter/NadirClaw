"""Selective context compression for NadirClaw.

Compresses conversation history by truncating old tool output and deduplicating
consecutive identical responses. Recent messages are preserved intact to avoid
losing active context.

Designed to reduce token usage for long agentic sessions (e.g., Claude Code)
where tool output can accumulate to hundreds of thousands of tokens.

Configuration is read via Settings properties (not module-level env reads)
so CLI ``serve --set`` overrides work correctly.
"""

import hashlib
import logging
from threading import Lock
from typing import Any, Dict, List, Tuple

from nadirclaw.settings import settings

logger = logging.getLogger("nadirclaw.compress")

# Thread-safe cumulative statistics
_stats_lock = Lock()
_compression_stats: Dict[str, int] = {
    "total_requests_compressed": 0,
    "total_chars_before": 0,
    "total_chars_after": 0,
    "total_truncated": 0,
    "total_deduped": 0,
}


def is_compression_enabled() -> bool:
    return settings.CONTEXT_COMPRESSION


def get_compression_stats() -> Dict[str, int]:
    with _stats_lock:
        return dict(_compression_stats)


def get_compression_config() -> Dict[str, Any]:
    return {
        "enabled": settings.CONTEXT_COMPRESSION,
        "min_messages": settings.COMPRESS_MIN_MESSAGES,
        "recent_window": settings.COMPRESS_RECENT_WINDOW,
        "tool_output_max": settings.COMPRESS_TOOL_OUTPUT_MAX,
    }


def _stable_hash(text: str) -> str:
    """Deterministic hash for deduplication (stable across restarts)."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def _is_tool_result_content(content: Any) -> bool:
    """Check if content contains tool_result blocks."""
    if isinstance(content, list):
        return any(
            isinstance(c, dict) and c.get("type") == "tool_result"
            for c in content
        )
    return False


def _truncate_tool_result(content: Any, max_len: int) -> Tuple[Any, bool]:
    """Truncate tool_result content blocks. Returns (content, was_truncated)."""
    if not isinstance(content, list):
        return content, False

    new_blocks = []
    truncated = False
    for block in content:
        if not isinstance(block, dict):
            new_blocks.append(block)
            continue
        if block.get("type") != "tool_result":
            new_blocks.append(block)
            continue

        result_content = block.get("content", "")
        if isinstance(result_content, str) and len(result_content) > max_len:
            new_block = {
                **block,
                "content": f"{result_content[:max_len]}\n... [truncated: {len(result_content)} chars]",
            }
            new_blocks.append(new_block)
            truncated = True
        elif isinstance(result_content, list):
            text_parts = []
            for rc in result_content:
                if isinstance(rc, dict) and rc.get("type") == "text":
                    text_parts.append(rc.get("text", ""))
            full_text = "\n".join(text_parts)
            if len(full_text) > max_len:
                new_block = {
                    **block,
                    "content": f"{full_text[:max_len]}\n... [truncated: {len(full_text)} chars]",
                }
                new_blocks.append(new_block)
                truncated = True
            else:
                new_blocks.append(block)
        else:
            new_blocks.append(block)

    return new_blocks, truncated


def compress_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compress conversation messages by truncating old tool output.

    Preserves:
    - All system/developer messages
    - All messages with tool_calls (needed for conversation flow)
    - Recent messages (last N turns)

    Compresses:
    - Old tool_result content (truncated to max chars)
    - Consecutive duplicate tool outputs (deduplicated)

    Note: Consecutive dedup means duplicates separated by a kept message
    (e.g. a user turn between two identical tool outputs) will NOT be deduped.
    This is intentional — the intermediate message may change interpretation.

    Args:
        messages: List of message dicts with role/content fields.

    Returns:
        (compressed_messages, stats_dict) where stats always contains
        the full set of keys (compressed=False when below threshold).
    """
    min_messages = settings.COMPRESS_MIN_MESSAGES
    recent_window = settings.COMPRESS_RECENT_WINDOW
    tool_output_max = settings.COMPRESS_TOOL_OUTPUT_MAX

    if len(messages) <= min_messages:
        return messages, {
            "compressed": False,
            "messages_before": len(messages),
            "messages_after": len(messages),
            "truncated": 0,
            "deduped": 0,
            "chars_before": 0,
            "chars_after": 0,
            "compression_ratio": 1.0,
        }

    compressed: List[Dict[str, Any]] = []
    total_before = 0
    total_after = 0
    truncated_count = 0
    deduped_count = 0
    last_kept_hash: str = ""

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        is_recent = i >= len(messages) - recent_window

        # Check for tool_calls in content
        has_tool_calls = False
        if isinstance(content, list):
            has_tool_calls = any(
                isinstance(c, dict) and c.get("type") == "tool_use"
                for c in content
            )

        # Always keep: recent, system/developer/user, messages with tool_calls
        if is_recent or role in ("system", "developer", "user") or has_tool_calls:
            compressed.append(msg)
            content_str = str(content)
            total_before += len(content_str)
            total_after += len(content_str)
            last_kept_hash = ""
            continue

        content_str = str(content)
        total_before += len(content_str)

        # Dedup: skip consecutive identical old content
        content_hash = _stable_hash(content_str[:200])
        if last_kept_hash and content_hash == last_kept_hash and len(content_str) > 100:
            deduped_count += 1
            total_after += 0
            continue

        # Truncate old tool_result content
        if _is_tool_result_content(content):
            new_content, was_truncated = _truncate_tool_result(
                content, tool_output_max
            )
            if was_truncated:
                truncated_count += 1
                new_msg = {**msg, "content": new_content}
                compressed.append(new_msg)
                total_after += len(str(new_content))
            else:
                compressed.append(msg)
                total_after += len(content_str)
            last_kept_hash = content_hash
            continue

        # Old assistant messages with no tool calls — truncate if very long
        if role == "assistant" and len(content_str) > 1000:
            truncated_count += 1
            summary = content_str[:500]
            new_msg = {**msg, "content": f"{summary}\n... [truncated: {len(content_str)} chars]"}
            compressed.append(new_msg)
            total_after += len(new_msg["content"])
            last_kept_hash = content_hash
            continue

        compressed.append(msg)
        total_after += len(content_str)
        last_kept_hash = content_hash

    stats = {
        "compressed": True,
        "messages_before": len(messages),
        "messages_after": len(compressed),
        "truncated": truncated_count,
        "deduped": deduped_count,
        "chars_before": total_before,
        "chars_after": total_after,
        "compression_ratio": round(total_after / total_before, 2) if total_before > 0 else 1.0,
    }

    with _stats_lock:
        _compression_stats["total_requests_compressed"] += 1
        _compression_stats["total_chars_before"] += total_before
        _compression_stats["total_chars_after"] += total_after
        _compression_stats["total_truncated"] += truncated_count
        _compression_stats["total_deduped"] += deduped_count

    return compressed, stats
