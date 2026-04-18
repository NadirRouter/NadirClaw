"""Selective context compression for NadirClaw.

Compresses conversation history by truncating old tool output and deduplicating
consecutive identical responses. Recent messages are preserved intact to avoid
losing active context.

Designed to reduce token usage for long agentic sessions (e.g., Claude Code)
where tool output can accumulate to hundreds of thousands of tokens.
"""

import json
import logging
import os
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("nadirclaw.compress")

# Configuration via environment
_COMPRESS_ENABLED = os.getenv("NADIRCLAW_CONTEXT_COMPRESSION", "false").lower() in ("true", "1", "yes")
_COMPRESS_MIN_MESSAGES = int(os.getenv("NADIRCLAW_COMPRESS_MIN_MESSAGES", "30"))
_COMPRESS_RECENT_WINDOW = int(os.getenv("NADIRCLAW_COMPRESS_RECENT_WINDOW", "20"))
_COMPRESS_TOOL_OUTPUT_MAX = int(os.getenv("NADIRCLAW_COMPRESS_TOOL_MAX", "500"))

# Cumulative statistics
_compression_stats: Dict[str, int] = {
    "total_requests_compressed": 0,
    "total_tokens_before": 0,
    "total_tokens_after": 0,
    "total_truncated": 0,
    "total_deduped": 0,
}


def is_compression_enabled() -> bool:
    """Check if context compression is currently enabled."""
    return _COMPRESS_ENABLED


def get_compression_stats() -> Dict[str, int]:
    """Return cumulative compression statistics."""
    return dict(_compression_stats)


def get_compression_config() -> Dict[str, Any]:
    """Return current compression configuration."""
    return {
        "enabled": _COMPRESS_ENABLED,
        "min_messages": _COMPRESS_MIN_MESSAGES,
        "recent_window": _COMPRESS_RECENT_WINDOW,
        "tool_output_max": _COMPRESS_TOOL_OUTPUT_MAX,
    }


def _is_tool_result_content(content: Any) -> bool:
    """Check if content is a tool_result block (OpenAI or Claude Code format)."""
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


def _content_hash(content: Any) -> int:
    """Generate a hash for deduplication of content."""
    s = str(content)[:200]
    return hash(s)


def compress_messages(messages: List[Any]) -> Tuple[List[Any], Dict[str, Any]]:
    """Compress conversation messages by truncating old tool output.

    Preserves:
    - All system/developer messages
    - All messages with tool_calls (needed for conversation flow)
    - Recent messages (last N turns)

    Compresses:
    - Old tool_result content (truncated to max chars)
    - Consecutive duplicate tool outputs (deduplicated)

    Args:
        messages: List of message dicts with role/content fields.

    Returns:
        (compressed_messages, stats_dict)
    """
    if len(messages) <= _COMPRESS_MIN_MESSAGES:
        return messages, {"skipped": True}

    compressed = []
    total_before = 0
    total_after = 0
    truncated_count = 0
    deduped_count = 0
    prev_hash = None

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        is_recent = i >= len(messages) - _COMPRESS_RECENT_WINDOW

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
            total_before += len(str(content))
            total_after += len(str(content))
            continue

        content_str = str(content)
        total_before += len(content_str)

        # Dedup consecutive identical tool outputs
        content_hash = _content_hash(content)
        if content_hash == prev_hash and len(content_str) > 100:
            deduped_count += 1
            prev_hash = content_hash
            total_after += 0  # skipped
            continue
        prev_hash = content_hash

        # Truncate old tool_result content
        if _is_tool_result_content(content):
            new_content, was_truncated = _truncate_tool_result(
                content, _COMPRESS_TOOL_OUTPUT_MAX
            )
            if was_truncated:
                truncated_count += 1
                new_msg = {**msg, "content": new_content}
                compressed.append(new_msg)
                total_after += len(str(new_content))
            else:
                compressed.append(msg)
                total_after += len(content_str)
            continue

        # Old assistant messages with no tool calls — truncate if very long
        if role == "assistant" and len(content_str) > 1000:
            truncated_count += 1
            summary = content_str[:500]
            new_msg = {**msg, "content": f"{summary}\n... [truncated: {len(content_str)} chars]"}
            compressed.append(new_msg)
            total_after += len(new_msg["content"])
            continue

        compressed.append(msg)
        total_after += len(content_str)

    stats = {
        "messages_before": len(messages),
        "messages_after": len(compressed),
        "truncated": truncated_count,
        "deduped": deduped_count,
        "chars_before": total_before,
        "chars_after": total_after,
        "compression_ratio": round(total_after / total_before, 2) if total_before > 0 else 1.0,
    }

    # Update cumulative stats
    _compression_stats["total_requests_compressed"] += 1
    _compression_stats["total_tokens_before"] += total_before // 4
    _compression_stats["total_tokens_after"] += total_after // 4
    _compression_stats["total_truncated"] += truncated_count
    _compression_stats["total_deduped"] += deduped_count

    return compressed, stats
