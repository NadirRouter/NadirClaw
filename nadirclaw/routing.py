"""Routing intelligence for NadirClaw.

Handles agentic task detection, reasoning detection, routing profiles,
model aliases, context-window filtering, and session persistence.
"""

import hashlib
import logging
import os
import random
import re
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("nadirclaw.routing")

# ---------------------------------------------------------------------------
# Model Pool — weighted load balancing across multiple models
# ---------------------------------------------------------------------------

# Lazy-initialized: pools are built on first access, not at import time,
# so CLI `serve --set NADIRCLAW_MODEL_POOLS=...` works correctly.
_MODEL_POOLS_CACHE: Optional[Dict[str, List[Tuple[str, int]]]] = None
_MODEL_TO_POOL_CACHE: Optional[Dict[str, str]] = None
_POOL_LOCK = Lock()


def _parse_model_pools() -> Tuple[Dict[str, List[Tuple[str, int]]], Dict[str, str]]:
    """Parse NADIRCLAW_MODEL_POOLS env var into pool + reverse-map.

    Format: "pool_name=model1,weight1+model2,weight2;pool_name2=..."
    Example: "turbo=glm-5-turbo,10+kimi-K2.6-code-preview,9+minimax-MiniMax-M2.7,3"
    """
    raw = os.getenv("NADIRCLAW_MODEL_POOLS", "")
    if not raw:
        return {}, {}
    pools: Dict[str, List[Tuple[str, int]]] = {}
    reverse: Dict[str, str] = {}
    for pool_def in raw.split(";"):
        pool_def = pool_def.strip()
        if not pool_def or "=" not in pool_def:
            continue
        pool_name, _, models_str = pool_def.partition("=")
        pool_name = pool_name.strip()
        if not pool_name or not models_str:
            continue
        entries: List[Tuple[str, int]] = []
        for entry in models_str.split("+"):
            entry = entry.strip()
            if not entry:
                continue
            segs = entry.rsplit(",", 1)
            if len(segs) == 2:
                model_name = segs[0].strip()
                try:
                    weight = max(1, int(segs[1].strip()))
                except ValueError:
                    weight = 1
            else:
                model_name = segs[0].strip()
                weight = 1
            if model_name:
                entries.append((model_name, weight))
                reverse[model_name] = pool_name
        if entries:
            pools[pool_name] = entries
    return pools, reverse


def _ensure_pools_loaded() -> Tuple[Dict[str, List[Tuple[str, int]]], Dict[str, str]]:
    """Lazily build and cache model pools on first routing call."""
    global _MODEL_POOLS_CACHE, _MODEL_TO_POOL_CACHE
    if _MODEL_POOLS_CACHE is None:
        with _POOL_LOCK:
            if _MODEL_POOLS_CACHE is None:
                _MODEL_POOLS_CACHE, _MODEL_TO_POOL_CACHE = _parse_model_pools()
    return _MODEL_POOLS_CACHE, _MODEL_TO_POOL_CACHE


def reload_pools() -> None:
    """Force re-read of model pools from env (useful after serve --set)."""
    global _MODEL_POOLS_CACHE, _MODEL_TO_POOL_CACHE
    with _POOL_LOCK:
        _MODEL_POOLS_CACHE, _MODEL_TO_POOL_CACHE = _parse_model_pools()


def select_from_pool(pool_name: str) -> str:
    """Select a model from the pool using weighted random selection.

    Args:
        pool_name: Name of the pool (e.g., "turbo", "reasoning").

    Returns:
        Selected model name.

    Raises:
        KeyError: If pool_name is not a configured pool.
    """
    pools, _ = _ensure_pools_loaded()
    pool = pools.get(pool_name)
    if not pool:
        raise KeyError(f"Unknown model pool: {pool_name!r}. Available: {list(pools.keys())}")
    total_weight = sum(w for _, w in pool)
    r = random.randint(1, total_weight)
    cumulative = 0
    for model, weight in pool:
        cumulative += weight
        if r <= cumulative:
            logger.debug(
                "Pool %s selected: %s (weight=%d, rand=%d/%d)",
                pool_name, model, weight, r, total_weight,
            )
            return model
    return pool[0][0]


def get_pool_for_model(model: str) -> Optional[str]:
    """Return the pool name for a given model, or None if not in any pool."""
    _, reverse = _ensure_pools_loaded()
    return reverse.get(model)

# ---------------------------------------------------------------------------
# Model registry — context windows and capabilities
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Gemini
    "gemini-3-flash-preview": {"context_window": 1_000_000, "cost_per_m_input": 0.50, "cost_per_m_output": 3.00, "has_vision": True},
    "gemini-2.5-pro": {"context_window": 1_000_000, "cost_per_m_input": 1.25, "cost_per_m_output": 10.00, "has_vision": True},
    "gemini-2.5-flash": {"context_window": 1_000_000, "cost_per_m_input": 0.15, "cost_per_m_output": 0.60, "has_vision": True},
    "gemini/gemini-3-flash-preview": {"context_window": 1_000_000, "cost_per_m_input": 0.50, "cost_per_m_output": 3.00, "has_vision": True},
    "gemini/gemini-2.5-pro": {"context_window": 1_000_000, "cost_per_m_input": 1.25, "cost_per_m_output": 10.00, "has_vision": True},
    # OpenAI
    "gpt-5.4": {"context_window": 1_047_576, "cost_per_m_input": 2.00, "cost_per_m_output": 8.00, "has_vision": True},
    "gpt-5.4": {"context_window": 1_047_576, "cost_per_m_input": 0.40, "cost_per_m_output": 1.60, "has_vision": True},
    "gpt-5.4": {"context_window": 1_047_576, "cost_per_m_input": 0.10, "cost_per_m_output": 0.40, "has_vision": True},
    "gpt-5": {"context_window": 400_000, "cost_per_m_input": 1.25, "cost_per_m_output": 10.00, "has_vision": True},
    "gpt-5-mini": {"context_window": 400_000, "cost_per_m_input": 0.25, "cost_per_m_output": 2.00, "has_vision": True},
    "gpt-5.1": {"context_window": 400_000, "cost_per_m_input": 1.25, "cost_per_m_output": 10.00, "has_vision": True},
    "gpt-5.2": {"context_window": 400_000, "cost_per_m_input": 1.75, "cost_per_m_output": 14.00, "has_vision": True},
    "gpt-5.4": {"context_window": 128_000, "cost_per_m_input": 2.50, "cost_per_m_output": 10.00, "has_vision": True},
    "gpt-5.4": {"context_window": 128_000, "cost_per_m_input": 0.15, "cost_per_m_output": 0.60, "has_vision": True},
    "o3": {"context_window": 200_000, "cost_per_m_input": 2.00, "cost_per_m_output": 8.00, "has_vision": True},
    "o3-mini": {"context_window": 200_000, "cost_per_m_input": 1.10, "cost_per_m_output": 4.40, "has_vision": True},
    "o4-mini": {"context_window": 200_000, "cost_per_m_input": 1.10, "cost_per_m_output": 4.40, "has_vision": True},
    "openai-codex/gpt-5.3-codex": {"context_window": 400_000, "cost_per_m_input": 1.75, "cost_per_m_output": 14.00, "has_vision": False},
    # Anthropic
    "claude-opus-4-6-20250918": {"context_window": 200_000, "cost_per_m_input": 5.00, "cost_per_m_output": 25.00, "has_vision": True},
    "claude-sonnet-4-6": {"context_window": 200_000, "cost_per_m_input": 3.00, "cost_per_m_output": 15.00, "has_vision": True},
    "claude-haiku-4-5-20251001": {"context_window": 200_000, "cost_per_m_input": 1.00, "cost_per_m_output": 5.00, "has_vision": True},
    "claude-opus-4-20250514": {"context_window": 200_000, "cost_per_m_input": 5.00, "cost_per_m_output": 25.00, "has_vision": True},
    "claude-sonnet-4-20250514": {"context_window": 200_000, "cost_per_m_input": 3.00, "cost_per_m_output": 15.00, "has_vision": True},
    "claude-haiku-4-20250514": {"context_window": 200_000, "cost_per_m_input": 1.00, "cost_per_m_output": 5.00, "has_vision": True},
    # DeepSeek
    "deepseek/deepseek-chat": {"context_window": 128_000, "cost_per_m_input": 0.28, "cost_per_m_output": 0.42, "has_vision": False},
    "deepseek/deepseek-reasoner": {"context_window": 128_000, "cost_per_m_input": 0.28, "cost_per_m_output": 0.42, "has_vision": False},
    # Ollama (local, no cost, context varies by model)
    "ollama/llama3.1:8b": {"context_window": 128_000, "cost_per_m_input": 0, "cost_per_m_output": 0, "has_vision": False},
    "ollama/qwen3:32b": {"context_window": 128_000, "cost_per_m_input": 0, "cost_per_m_output": 0, "has_vision": False},
    # GLM (Zhipu AI)
    "glm-5": {"context_window": 200_000, "cost_per_m_input": 0.05, "cost_per_m_output": 0.05, "has_vision": False},
    "glm-5.1": {"context_window": 200_000, "cost_per_m_input": 0.05, "cost_per_m_output": 0.05, "has_vision": False},
    "glm-5-turbo": {"context_window": 200_000, "cost_per_m_input": 0.05, "cost_per_m_output": 0.05, "has_vision": False},
    "glm-4.7": {"context_window": 200_000, "cost_per_m_input": 0.05, "cost_per_m_output": 0.05, "has_vision": False},
    "glm-4.7-flash": {"context_window": 200_000, "cost_per_m_input": 0.01, "cost_per_m_output": 0.01, "has_vision": False},
    "zai/glm-5": {"context_window": 200_000, "cost_per_m_input": 0.05, "cost_per_m_output": 0.05, "has_vision": False},
    # MiniMax
    "minimax-MiniMax-M2.7": {"context_window": 200_000, "cost_per_m_input": 0.10, "cost_per_m_output": 0.10, "has_vision": False},
    "minimax-MiniMax-M2.5": {"context_window": 200_000, "cost_per_m_input": 0.10, "cost_per_m_output": 0.10, "has_vision": False},
    # Kimi
    "kimi-K2.6-code-preview": {"context_window": 200_000, "cost_per_m_input": 0.10, "cost_per_m_output": 0.10, "has_vision": False},
    # Gemini 3.1 Pro (long context)
    "gemini-3.1-pro": {"context_window": 2_000_000, "cost_per_m_input": 1.25, "cost_per_m_output": 10.00, "has_vision": True},
}

# ---------------------------------------------------------------------------
# Model aliases — short names to full model IDs
# ---------------------------------------------------------------------------

MODEL_ALIASES: Dict[str, str] = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
    "haiku": "claude-haiku-4-5-20251001",
    "claude": "claude-sonnet-4-6",
    "claude-opus-4-6": "claude-opus-4-6-20250918",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "gpt-5.4": "gpt-5.4",
    "gpt5.4": "gpt-5.4",
    "gpt-5.4": "gpt-5.4",
    "gpt5": "gpt-5.2",
    "gpt5-mini": "gpt-5-mini",
    "o3": "o3",
    "o3-mini": "o3-mini",
    "o4-mini": "o4-mini",
    "flash": "gemini-2.5-flash",
    "gemini-flash": "gemini-2.5-flash",
    "gemini-pro": "gemini-2.5-pro",
    "deepseek": "deepseek/deepseek-chat",
    "deepseek-r1": "deepseek/deepseek-reasoner",
    "llama": "ollama/llama3.1:8b",
    "glm": "glm-5",
    "glm5": "glm-5",
    "minimax": "minimax-MiniMax-M2.7",
    "kimi": "kimi-K2.6-code-preview",
}

# ---------------------------------------------------------------------------
# Routing profiles
# ---------------------------------------------------------------------------

ROUTING_PROFILES = {"auto", "eco", "premium", "free", "reasoning"}


def resolve_profile(model_field: Optional[str]) -> Optional[str]:
    """Check if the model field is a routing profile name.

    Returns the profile name if matched, None otherwise.
    """
    if not model_field:
        return None
    cleaned = model_field.strip().lower()
    # Support "nadirclaw/eco" prefix style
    if cleaned.startswith("nadirclaw/"):
        cleaned = cleaned[len("nadirclaw/"):]
    if cleaned in ROUTING_PROFILES:
        return cleaned
    return None


def resolve_alias(model_field: str) -> Optional[str]:
    """Resolve a model alias to a full model ID.

    Returns the resolved model name, or None if not an alias.
    """
    return MODEL_ALIASES.get(model_field.strip().lower())


# ---------------------------------------------------------------------------
# Agentic task detection
# ---------------------------------------------------------------------------

_AGENTIC_SYSTEM_KEYWORDS = re.compile(
    r"\b("
    r"you are an? (?:ai |coding |software )?agent"
    r"|execute (?:commands?|tools?|code|tasks?)"
    r"|you (?:can|have access to|may) (?:use |call |run |execute )?(?:tools?|functions?|commands?)"
    r"|tool[ _]?(?:use|call|execution)"
    r"|multi[- ]?step"
    r"|(?:read|write|edit|create|delete) files?"
    r"|run (?:commands?|shell|bash|terminal)"
    r"|code execution"
    r"|file (?:system|access)"
    r"|web ?search"
    r"|browser"
    r"|autonomous"
    r")\b",
    re.IGNORECASE,
)


def detect_agentic(
    messages: List[Any],
    has_tools: bool = False,
    tool_count: int = 0,
    system_prompt: str = "",
    system_prompt_length: int = 0,
    message_count: int = 0,
) -> Dict[str, Any]:
    """Score agentic signals in a request.

    Returns {"is_agentic": bool, "confidence": float, "signals": list[str]}.

    NOTE: Threshold raised to 0.80 to avoid false positives from Claude Code's
    default tool-rich environment (200+ tools, 15KB+ system prompt).
    """
    score = 0.0
    signals: List[str] = []

    # Tool definitions present - DISABLED for Claude Code (always 200+ tools)
    # if has_tools and tool_count >= 1:
    #     score += 0.35
    #     signals.append(f"tools_defined({tool_count})")
    # if tool_count >= 4:
    #     score += 0.15
    #     signals.append("many_tools")

    # Tool-role messages in conversation (active agentic loop)
    # Higher threshold for Claude Code: need 3+ tool messages for strong signal
    tool_msgs = sum(1 for m in messages if getattr(m, "role", None) == "tool")
    if tool_msgs >= 3:
        score += 0.60
        signals.append(f"tool_messages({tool_msgs})")
    elif tool_msgs >= 1:
        score += 0.30
        signals.append(f"tool_messages({tool_msgs})")

    # Assistant→tool cycles (multi-step execution)
    # Higher threshold: need 3+ cycles for strong signal
    cycles = _count_agentic_cycles(messages)
    if cycles >= 3:
        score += 0.50
        signals.append(f"agentic_cycles({cycles})")
    elif cycles >= 2:
        score += 0.30
        signals.append(f"agentic_cycles({cycles})")

    # Long system prompt - DISABLED for Claude Code (always 15KB+)
    # if system_prompt_length > 500:
    #     score += 0.10
    #     signals.append("long_system_prompt")

    # System prompt keywords - DISABLED for Claude Code (always present)
    # if system_prompt and _AGENTIC_SYSTEM_KEYWORDS.search(system_prompt):
    #     score += 0.20
    #     signals.append("agentic_keywords")

    # Many messages (deep conversation / multi-turn loop)
    # Higher thresholds for Claude Code's longer sessions
    if message_count > 50:
        score += 0.20
        signals.append("deep_conversation(50+)")
    elif message_count > 20:
        score += 0.10
        signals.append("deep_conversation(20+)")

    # Cap at 1.0
    confidence = min(score, 1.0)
    # Raised threshold from 0.35 to 0.80 for Claude Code compatibility
    is_agentic = confidence >= 0.80

    return {"is_agentic": is_agentic, "confidence": confidence, "signals": signals}


def _count_agentic_cycles(messages: List[Any]) -> int:
    """Count assistant→tool→assistant cycles in the message list."""
    cycles = 0
    roles = [getattr(m, "role", "") for m in messages]
    i = 0
    while i < len(roles) - 2:
        if roles[i] == "assistant" and roles[i + 1] == "tool":
            cycles += 1
            i += 2
        else:
            i += 1
    return cycles


# ---------------------------------------------------------------------------
# Reasoning detection
# ---------------------------------------------------------------------------

_REASONING_MARKERS_EN = re.compile(
    r"\b("
    r"step[- ]by[- ]step"
    r"|think (?:through|carefully|deeply|about)"
    r"|chain[- ]of[- ]thought"
    r"|let'?s? reason"
    r"|reason(?:ing)? (?:about|through)"
    r"|prove (?:that|this|the)"
    r"|formal (?:proof|verification)"
    r"|mathematical(?:ly)? (?:prove|show|derive)"
    r"|derive (?:the|a|an)"
    r"|analyze the (?:tradeoffs?|trade-offs?|implications?|consequences?)"
    r"|compare and contrast"
    r"|what are the (?:pros? and cons?|advantages? and disadvantages?)"
    r"|evaluate (?:the|whether|if)"
    r"|critically (?:analyze|assess|examine)"
    r"|explain (?:why|how|the reasoning)"
    r"|work through"
    r"|break (?:this|it) down"
    r"|logical(?:ly)? (?:deduce|infer|conclude)"
    r"|analyze why (?:this|the|it)"
    r"|diagnose the (?:root )?cause"
    r"|weigh (?:the )?(?:pros|cons|options|alternatives)"
    r"|architectural (?:decision|choice)"
    r"|design (?:a )?(?:system|architecture)"
    r")\b",
    re.IGNORECASE,
)

_REASONING_MARKERS_ZH = re.compile(
    r"("
    r"一步步"
    r"|逐步分析"
    r"|深入思考"
    r"|深入分析"
    r"|推理分析"
    r"|逻辑推理"
    r"|证明\s+(?:以下|这个)"
    r"|推导\s+(?:公式|结论)"
    r"|分析.*利弊"
    r"|权衡.*优劣"
    r"|权衡.*利弊"
    r"|对比分析"
    r"|比较.*差异"
    r"|优缺点"
    r"|批判性分析"
    r"|证明以下"
    r"|证明这个"
    r"|推导公式"
    r"|推导结论"
    r"|详细解释.*原因"
    r"|论证以下"
    r"|论证这个"
    r"|演绎推理"
    r"|归纳推理"
    r"|设计.*系统"
    r"|设计.*方案"
    r")",
)


# Patterns that indicate auto-injected context (not real user requests)
_CONTEXT_INJECTION_PATTERNS = re.compile(
    r"(?i)("
    # CLAUDE.md / config injection by Claude Code
    r"the following (is|are) the user'?s?\s*(claude\.md|claudemd|gemini\.md|agents\.md)"
    r"|contents of .*(claude\.md|settings\.json)"
    r"|project instructions.*checked into"
    r"|user'?s?\s*private global instructions"
    r"|codebase and user instructions"
    r")"
)


def detect_reasoning(prompt: str, system_message: str = "") -> Dict[str, Any]:
    """Detect if a prompt requires reasoning capabilities.

    Uses separate regexes for English (with \\b word boundaries) and Chinese
    (without \\b, since CJK characters have no word boundaries).

    Returns {"is_reasoning": bool, "marker_count": int, "markers": list[str]}.

    NOTE: Only checks the user prompt, NOT the system message.
    System messages in Claude Code contain many reasoning-related instructions
    that would cause false positives for every request.
    """
    en_matches = _REASONING_MARKERS_EN.findall(prompt)
    zh_matches = _REASONING_MARKERS_ZH.findall(prompt)
    matches = list(set(en_matches + zh_matches))
    marker_count = len(matches)

    # 1+ markers = reasoning task
    is_reasoning = marker_count >= 1

    return {
        "is_reasoning": is_reasoning,
        "marker_count": marker_count,
        "markers": matches,
    }


# ---------------------------------------------------------------------------
# Execution task detection
# ---------------------------------------------------------------------------

_EXECUTION_MARKERS = re.compile(
    r"\b("
    r"run\s+(tests?|the|this|build|lint|check|script)"
    r"|execute\s+(this|the|command|script)"
    r"|bash\s+"
    r"|shell\s+"
    r"|git\s+(commit|push|pull|add|status|checkout|merge|rebase|branch)"
    r"|npm\s+(install|run|test|build|start)"
    r"|pip\s+install"
    r"|docker\s+(build|run|push|pull|exec)"
    r"|kubectl\s+"
    r"|compile\s+(this|the|code)"
    r"|lint\s+(this|the|code|check)"
    r"|build\s+(this|the|project)"
    r"|start\s+(the\s+)?(server|service|app)"
    r"|stop\s+(the\s+)?(server|service|app)"
    r"|restart\s+(the\s+)?(server|service|app)"
    r"|deploy\s+(this|the|to)"
    r"|cd\s+\S"
    r"|mkdir\s+"
    r"|rm\s+"
    r"|cp\s+"
    r"|mv\s+"
    r"|ls\s*"
    r"|cat\s+"
    r"|grep\s+"
    r"|find\s+"
    r"|chmod\s+"
    r"|chown\s+"
    r"|apt\s+"
    r"|yum\s+"
    r"|brew\s+install"
    r"|make\s+"
    r"|cargo\s+"
    r"|go\s+(run|build|test|mod)"
    r"|python\s+"
    r"|node\s+"
    r"|pytest\s+"
    r"|jest\s+"
    r"|cargo\s+test"
    r"|go\s+test"
    r"|mvn\s+"
    r"|gradle\s+"
    r")\b",
    re.IGNORECASE,
)

_CONTINUATION_MARKERS = re.compile(
    r"^("
    r"继续$"
    r"|continue$"
    r"|go\s*ahead$"
    r"|执行$"
    r"|proceed$"
    r"|keep\s+going$"
    r"|carry\s+on$"
    r"|next$"
    r"|下一步$"
    r"|then\?$"
    r"|and\s+then$"
    r"|之后呢$"
    r"|接着$"
    r"|继续吧$"
    r"|go\s+on$"
    r")$",
    re.IGNORECASE,
)

_EXECUTION_TOOLS = {
    "Bash", "bash", "shell", "execute", "Execute", "exec", "Exec",
    "Write", "write", "Edit", "edit", "FileEdit", "file_edit",
    "Task", "task", "Run", "run", "Command", "command",
    "NotebookEdit", "notebook_edit",
}


def detect_execution(
    prompt: str,
    tool_names: Optional[List[str]] = None,
    last_tool_call: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect if a prompt is an execution/continuation task.

    Returns {"is_execution": bool, "confidence": float, "signals": list[str]}.
    """
    score = 0.0
    signals: List[str] = []

    # Continuation prompts (very high confidence)
    prompt_stripped = prompt.strip()
    if _CONTINUATION_MARKERS.match(prompt_stripped):
        score += 0.70
        signals.append("continuation_prompt")

    # Execution keywords in prompt
    if _EXECUTION_MARKERS.search(prompt):
        score += 0.40
        signals.append("execution_keywords")

    # Last tool call was an execution tool
    if last_tool_call and last_tool_call in _EXECUTION_TOOLS:
        score += 0.50
        signals.append(f"last_tool={last_tool_call}")

    # Request defines execution tools
    if tool_names:
        exec_tools = [t for t in tool_names if t in _EXECUTION_TOOLS]
        if exec_tools:
            score += 0.30
            signals.append(f"tools={exec_tools[:3]}")

    confidence = min(score, 1.0)
    is_execution = confidence >= 0.40

    return {"is_execution": is_execution, "confidence": confidence, "signals": signals}


# ---------------------------------------------------------------------------
# Complex coding detection
# ---------------------------------------------------------------------------

def detect_complex_coding(
    messages: List[Any],
    tool_names: List[str],
    last_tool_call: Optional[str],
    message_count: int,
    system_prompt_length: int = 0,
) -> Dict[str, Any]:
    """Detect complex coding tasks that should route to Sonnet.

    Complex coding tasks are characterized by:
    - Multiple file edits (3+ Edit/Write calls)
    - Tool combination patterns (Read + Edit + Bash)
    - Deep conversations (10+ messages)
    - Coding task keywords (implement, refactor, fix bug, etc.)

    Returns {"is_complex": bool, "confidence": float, "signals": list}.
    """
    from nadirclaw.settings import settings

    confidence = 0.0
    signals: List[str] = []

    # Count ACTUAL tool calls from RECENT messages only (last 6 assistant msgs)
    # Long conversations accumulate many tool calls — only recent ones matter
    actual_tool_calls: Dict[str, int] = {}
    assistant_count = 0
    for m in reversed(messages):
        if getattr(m, "role", "") == "assistant":
            assistant_count += 1
            if assistant_count > 6:
                break
            content = getattr(m, "content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        name = block.get("name", "")
                        actual_tool_calls[name] = actual_tool_calls.get(name, 0) + 1
            # Also check model_extra for tool_calls (OpenAI format)
            m_extra = getattr(m, "model_extra", None) or {}
            for tc in m_extra.get("tool_calls", []):
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    name = func.get("name", "") if isinstance(func, dict) else ""
                    if name:
                        actual_tool_calls[name] = actual_tool_calls.get(name, 0) + 1

    # Signal 1: Heavy editing (multiple actual Edit/Write calls)
    edit_write_count = sum(
        actual_tool_calls.get(t, 0) for t in ["Edit", "Write", "NotebookEdit"]
    )
    if edit_write_count >= 5:
        confidence += settings.COMPLEX_WEIGHT_EDITING
        signals.append(f"heavy_editing({edit_write_count})")
    elif edit_write_count >= 3:
        confidence += settings.COMPLEX_WEIGHT_EDITING * 0.6  # 0.30 by default
        signals.append(f"moderate_editing({edit_write_count})")

    # Signal 2: Tool combination pattern (actual Read + Edit + Bash calls)
    has_read = actual_tool_calls.get("Read", 0) > 0
    has_edit = any(actual_tool_calls.get(t, 0) > 0 for t in ["Edit", "Write", "NotebookEdit"])
    has_bash = actual_tool_calls.get("Bash", 0) > 0
    if has_read and has_edit and has_bash:
        confidence += settings.COMPLEX_WEIGHT_COMBO
        signals.append("read_edit_bash_combo")
    elif has_read and has_edit:
        confidence += settings.COMPLEX_WEIGHT_COMBO * 0.5  # 0.15 by default
        signals.append("read_edit_combo")

    # Signal 3: Recent tool call density (last 2 assistant turns)
    # High density = actively working on a complex task
    recent_tool_count = sum(actual_tool_calls.values())
    if recent_tool_count >= 8:
        confidence += settings.COMPLEX_WEIGHT_CONVERSATION
        signals.append(f"high_tool_density({recent_tool_count})")
    elif recent_tool_count >= 4:
        confidence += settings.COMPLEX_WEIGHT_CONVERSATION * 0.5
        signals.append(f"moderate_tool_density({recent_tool_count})")

    # Signal 4: Skip large system prompt signal — always true for Claude Code
    # (was: system_prompt_length > 15000 → +0.10, but meaningless for main session)

    # Signal 5: Coding task keywords in last user message
    last_user_text = ""
    for m in reversed(messages):
        if getattr(m, "role", "") == "user":
            last_user_text = getattr(m, "text_content", lambda: "")()
            break

    # Coding keywords (Chinese + English)
    coding_keywords = [
        # Implementation
        r"实现", r"添加.*功能", r"implement", r"add.*feature",
        # Modification
        r"修改", r"更新", r"modify", r"update", r"change",
        # Refactoring
        r"重构", r"优化", r"refactor", r"optimize", r"improve",
        # Debugging
        r"修复.*bug", r"调试", r"排查", r"fix.*bug", r"debug", r"troubleshoot",
        # Creation
        r"创建.*功能", r"生成.*代码", r"构建", r"create.*feature", r"generate.*code", r"build",
        # Multi-file operations
        r"多个文件", r"批量", r"multiple.*files", r"batch",
    ]

    keyword_matches = 0
    matched_keywords = []
    for pattern in coding_keywords:
        if re.search(pattern, last_user_text, re.IGNORECASE):
            keyword_matches += 1
            matched_keywords.append(pattern[:20])  # Store first 20 chars

    if keyword_matches >= 3:
        confidence += settings.COMPLEX_WEIGHT_KEYWORDS * 1.33  # 0.40 by default
        signals.append(f"coding_keywords({keyword_matches})")
    elif keyword_matches >= 2:
        confidence += settings.COMPLEX_WEIGHT_KEYWORDS * 0.83  # 0.25 by default
        signals.append(f"coding_keywords({keyword_matches})")
    elif keyword_matches >= 1:
        confidence += settings.COMPLEX_WEIGHT_KEYWORDS * 0.33  # 0.10 by default
        signals.append(f"coding_keyword({keyword_matches})")

    # Threshold: configurable via NADIRCLAW_COMPLEX_THRESHOLD
    is_complex = confidence >= settings.COMPLEX_THRESHOLD

    return {
        "is_complex": is_complex,
        "confidence": confidence,
        "signals": signals,
    }


# ---------------------------------------------------------------------------
# Code review detection
# ---------------------------------------------------------------------------

# Code review keywords (trigger code verification/review tasks)
# Only Chinese keywords to avoid false positives on common English words
_REVIEW_MARKERS = re.compile(
    r"(审查|核查|检查代码|代码审查|代码质量|静态分析"
    r"|安全检查|漏洞扫描|代码规范|代码风格|验证代码|审核代码)",
    re.IGNORECASE,
)


def detect_code_review(prompt: str, system_message: str = "") -> Dict[str, Any]:
    """Detect code review/verification tasks.

    Returns {"is_review": bool, "confidence": float, "signals": list}.
    """
    confidence = 0.0
    signals: List[str] = []

    text = f"{system_message}\n{prompt}" if system_message else prompt
    if _REVIEW_MARKERS.search(text):
        confidence = 0.90
        signals.append("review_keywords")

    # Additional signals for code review context
    review_context_signals = [
        r"pull\s*request", r"pr\s*review", r"merge\s*request",
        r"commit.*review", r"change.*review",
        r"diff.*check", r"patch.*review",
        r"代码变更", r"变更审查",
    ]

    for pattern in review_context_signals:
        if re.search(pattern, text, re.IGNORECASE):
            confidence = max(confidence, 0.85)
            signals.append("review_context")
            break

    is_review = confidence >= 0.80

    return {
        "is_review": is_review,
        "confidence": confidence,
        "signals": signals,
    }


# ---------------------------------------------------------------------------
# Agent role detection — identify AI coding agent session types
#
# This feature is opt-in via NADIRCLAW_AGENT_ROLE_DETECTION=true.
# It detects coding agent session types (planning, explore, subagent)
# from system prompt markers. Currently tuned for Claude Code;
# additional agent support welcome via PR.
#
# Markers are intentionally matched against system prompts only,
# not user messages, to avoid false positives from career questions
# or general discussion about software architecture.
# ---------------------------------------------------------------------------

# Named constants for session classification thresholds.
MAIN_SESSION_MIN_CHARS = 15000  # chars — main session has long system prompt
SHORT_SESSION_MAX_CHARS = 5000  # chars — likely a subagent/background task

_CLAUDE_CODE_PLANNING_MARKERS = re.compile(
    r"(plan\s*mode\s*is\s*active"
    r"|software\s+architect"
    r"|planning\s+specialist"
    r"|READ-ONLY.*planning"
    r"|architect\s+agent"
    r"|design.*implementation\s+plan)",
    re.IGNORECASE,
)

# User-initiated plan command detection (in user message, not system prompt)
_PLAN_COMMAND_MARKERS = re.compile(
    r"(^|\s)/plan\b"
    r"|实施以下计划"
    r"|implement\s+the\s+following\s+plan"
    r"|帮我.*规划"
    r"|设计.*实现方案"
    r"|制定.*计划",
    re.IGNORECASE,
)

# Explore agent markers (should route to reasoning model like planning)
_CLAUDE_CODE_EXPLORE_MARKERS = re.compile(
    r"(explore\s+agent"
    r"|explore\s+codebase"
    r"|fast\s+agent\s+specialized\s+for\s+exploring)",
    re.IGNORECASE,
)

# Other subagent markers (route to simple model)
_CLAUDE_CODE_SUBAGENT_MARKERS = re.compile(
    r"(haiku\s*4\.?5"
    r"|sonnet\s*4\.?5"
    r"|specialized\s+agent"
    r"|subagent"
    r"|background\s+agent"
    r"|search\s+agent)",
    re.IGNORECASE,
)


def detect_claude_code_role(
    system_prompt: str,
    message_count: int = 0,
    tool_names: Optional[List[str]] = None,
    last_user_message: str = "",
) -> Dict[str, Any]:
    """Detect Claude Code agent role from system prompt signals.

    Currently tuned for Claude Code. Opt-in via NADIRCLAW_AGENT_ROLE_DETECTION=true.

    Returns {"role": str, "confidence": float, "signals": list[str]}.
    role can be: "planning", "explore", "subagent", "execution", or "unknown".
    """
    role = "unknown"
    confidence = 0.0
    signals: List[str] = []
    tool_names = tool_names or []

    # Check for user-initiated /plan command in last user message
    # This handles the case where user sends /plan but system prompt hasn't been updated yet
    has_plan_command = bool(_PLAN_COMMAND_MARKERS.search(last_user_message)) if last_user_message else False

    # Planning mode detection — check for explicit plan mode indicators in system prompt
    # NOTE: ExitPlanMode tool is always present and means "can exit plan mode",
    # NOT "currently in plan mode". So we should NOT use it to detect planning.
    # Only system prompt markers indicate actual planning mode.
    has_plan_markers = bool(_CLAUDE_CODE_PLANNING_MARKERS.search(system_prompt))

    # Planning detected if either:
    # 1. System prompt has plan mode markers (Plan mode already active)
    # 2. User sent /plan command (first request to enter plan mode)
    if has_plan_markers or has_plan_command:
        role = "planning"
        confidence = 0.95
        if has_plan_markers:
            signals.append("planning_markers")
        if has_plan_command:
            signals.append("plan_command")
        return {"role": role, "confidence": confidence, "signals": signals}

    # Distinguish subagents from main sessions.
    # Main sessions have long system prompts with extensive instructions.
    is_main_session = len(system_prompt) > MAIN_SESSION_MIN_CHARS

    # Explore agent detection (route to explore model)
    if _CLAUDE_CODE_EXPLORE_MARKERS.search(system_prompt):
        role = "explore"
        confidence = 0.95
        signals.append("explore_markers")
        return {"role": role, "confidence": confidence, "signals": signals}

    # Subagent detection (model identity in system prompt)
    # ONLY apply if NOT main session
    if not is_main_session and _CLAUDE_CODE_SUBAGENT_MARKERS.search(system_prompt):
        role = "subagent"
        confidence = 0.90
        signals.append("subagent_markers")
        return {"role": role, "confidence": confidence, "signals": signals}

    # Short system prompt = likely subagent (but lower confidence, don't override execution)
    # Only use this as a tiebreaker when no other detection applies
    if not is_main_session and len(system_prompt) < SHORT_SESSION_MAX_CHARS:
        role = "subagent"
        confidence = 0.60  # Matches the routing threshold for subagent tier
        signals.append("short_system_prompt")
        # Don't return immediately - let other detection take precedence

    # Long conversation + execution tools = execution mode
    if message_count > 50 and tool_names:
        exec_tools = [t for t in tool_names if t in _EXECUTION_TOOLS]
        if len(exec_tools) >= 3:
            role = "execution"
            confidence = 0.70
            signals.append(f"long_conversation_exec_tools({len(exec_tools)})")
            return {"role": role, "confidence": confidence, "signals": signals}

    return {"role": role, "confidence": confidence, "signals": signals}


# ---------------------------------------------------------------------------
# Context window check
# ---------------------------------------------------------------------------

def estimate_token_count(messages: List[Any]) -> int:
    """Rough token estimate: ~4 chars per token."""
    total_chars = 0
    for m in messages:
        content = getattr(m, "text_content", lambda: "")()
        if not content:
            content = getattr(m, "content", "") or ""
            if not isinstance(content, str):
                content = str(content)
        total_chars += len(content)
    return total_chars // 4


def check_context_window(model: str, messages: List[Any]) -> bool:
    """Return True if the model can handle the estimated token count.

    Returns True (allow) if the model is not in the registry (assume it fits).
    """
    info = MODEL_REGISTRY.get(model)
    if not info:
        return True
    estimated = estimate_token_count(messages)
    return estimated < info["context_window"]


def get_context_window(model: str) -> Optional[int]:
    """Return context window for a model, or None if unknown."""
    info = MODEL_REGISTRY.get(model)
    return info["context_window"] if info else None


def has_vision(model: str) -> bool:
    """Return True if the model supports vision/image inputs."""
    info = MODEL_REGISTRY.get(model)
    if info is None:
        return False
    return info.get("has_vision", False)


# ---------------------------------------------------------------------------
# Vision / image detection
# ---------------------------------------------------------------------------

def detect_images(messages: List[Any]) -> Dict[str, Any]:
    """Detect if any messages contain image content (image_url or image parts).

    Returns {"has_images": bool, "image_count": int}.
    """
    image_count = 0
    for m in messages:
        content = getattr(m, "content", None)
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in ("image_url", "image"):
                image_count += 1
    return {"has_images": image_count > 0, "image_count": image_count}


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

class SessionCache:
    """Cache routing decisions for multi-turn conversations.

    Keyed by a hash of the system prompt + first user message.
    TTL-based expiry with LRU eviction to cap memory usage.
    """

    def __init__(self, ttl_seconds: int = 1800, max_size: int = 10_000):
        # OrderedDict gives O(1) move-to-end (move_to_end) and O(1) popitem(last=False)
        # for LRU eviction — replaces the old List-based access_order which was O(n).
        self._cache: OrderedDict[str, Tuple[str, str, float]] = OrderedDict()  # key → (model, tier, timestamp)
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._cleanup_counter = 0
        self._cleanup_interval = 100  # run cleanup every N puts
        self._lock = Lock()

    def _make_key(self, messages: List[Any]) -> str:
        """Generate a session key from conversation shape.

        Uses system prompt + last user message so that each new user input
        gets a fresh routing decision instead of being locked to the first one.
        """
        parts: List[str] = []
        for m in messages:
            role = getattr(m, "role", "")
            if role in ("system", "developer"):
                content = getattr(m, "text_content", lambda: "")()
                parts.append(f"sys:{content[:200]}")
                break

        # Last user message — allows re-routing when the user changes topic
        for m in reversed(messages):
            role = getattr(m, "role", "")
            if role == "user":
                content = getattr(m, "text_content", lambda: "")()
                # Strip system-reminder for cache key consistency
                content = re.sub(r'<system-reminder>.*?</system-reminder>', '', content, flags=re.DOTALL).strip()
                parts.append(f"usr:{content[:200]}")
                break

        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _touch(self, key: str) -> None:
        """Move key to most-recently-used position — O(1) with OrderedDict."""
        self._cache.move_to_end(key)

    def _evict_lru(self) -> None:
        """Evict least-recently-used entries until under max size — O(1) per eviction."""
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def get(self, messages: List[Any]) -> Optional[Tuple[str, str]]:
        """Return (model, tier) if a session exists and isn't expired."""
        key = self._make_key(messages)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            model, tier, ts = entry
            if time.time() - ts > self._ttl:
                del self._cache[key]
                return None
            self._touch(key)
            return model, tier

    def put(self, messages: List[Any], model: str, tier: str) -> None:
        """Store a routing decision for this session."""
        key = self._make_key(messages)
        with self._lock:
            # Periodic cleanup of expired entries
            self._cleanup_counter += 1
            if self._cleanup_counter >= self._cleanup_interval:
                self._cleanup_counter = 0
                self.clear_expired()

            self._cache[key] = (model, tier, time.time())
            self._touch(key)

            # Evict if over capacity
            if len(self._cache) > self._max_size:
                self._evict_lru()

    def clear_expired(self) -> int:
        """Remove expired entries. Returns number removed.

        Caller must hold self._lock.
        """
        now = time.time()
        expired = [k for k, (_, _, ts) in self._cache.items() if now - ts > self._ttl]
        for k in expired:
            del self._cache[k]
        return len(expired)


# Global session cache
_session_cache = SessionCache(ttl_seconds=1800)


def get_session_cache() -> SessionCache:
    return _session_cache


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """Estimate cost in USD for a request. Returns None if model not in registry."""
    info = MODEL_REGISTRY.get(model)
    if not info:
        return None
    input_cost = (prompt_tokens / 1_000_000) * info["cost_per_m_input"]
    output_cost = (completion_tokens / 1_000_000) * info["cost_per_m_output"]
    return input_cost + output_cost


# ---------------------------------------------------------------------------
# Main routing modifier — applies all intelligence
# ---------------------------------------------------------------------------

def apply_routing_modifiers(
    base_model: str,
    base_tier: str,
    request_meta: Dict[str, Any],
    messages: List[Any],
    simple_model: str,
    complex_model: str,
    reasoning_model: Optional[str] = None,
    free_model: Optional[str] = None,
    sonnet_model: Optional[str] = None,
    explore_model: Optional[str] = None,
    subagent_model: Optional[str] = None,
    review_model: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Apply all routing modifiers on top of the classifier's base decision.

    Returns (final_model, final_tier, routing_info).
    """
    routing_info: Dict[str, Any] = {
        "base_tier": base_tier,
        "base_model": base_model,
        "modifiers_applied": [],
    }

    final_model = base_model
    final_tier = base_tier

    # --- Claude Code role detection (early check) ---
    system_text = request_meta.get("system_prompt_text", "")
    message_count = request_meta.get("message_count", 0)
    tool_names = request_meta.get("tool_names", [])

    # Extract ALL user messages to check for Plan mode markers in system-reminder
    # Plan mode markers appear in the FIRST user message's <system-reminder> tags
    all_user_texts_for_role = []
    for m in messages:
        if getattr(m, "role", "") == "user":
            user_text = getattr(m, "text_content", lambda: "")() or ""
            all_user_texts_for_role.append(user_text)

    # Combine system prompt + ALL user messages for role detection
    # This ensures we detect Plan mode markers anywhere in the conversation
    combined_text_for_role = system_text + "\n" + "\n".join(all_user_texts_for_role)

    # Get last user message for /plan command detection
    last_user_text_for_role = all_user_texts_for_role[-1] if all_user_texts_for_role else ""

    cc_role = detect_claude_code_role(
        system_prompt=combined_text_for_role,
        message_count=message_count,
        tool_names=tool_names,
        last_user_message=last_user_text_for_role,
    )
    routing_info["claude_code_role"] = cc_role

    # --- Agent role detection (opt-in) ---
    # Detects coding agent session types (planning, explore, subagent).
    # Disabled by default — enable with NADIRCLAW_AGENT_ROLE_DETECTION=true.
    from nadirclaw.settings import settings as _settings
    if _settings.AGENT_ROLE_DETECTION:
        agent_role = detect_agent_role(
            system_prompt=system_text,
            message_count=message_count,
            tool_names=tool_names,
        )
        routing_info["agent_role"] = agent_role
    else:
        routing_info["agent_role"] = {"role": "unknown", "confidence": 0.0, "signals": []}

    # --- Background security monitor detection (early return) ---
    # These are PreToolUse hook requests (action-guard) that check if agent
    # actions are safe. They are binary classification (allow/block) and
    # should use cheap models, not sonnet/opus.
    if "You are a security monitor for autonomous AI coding agents" in system_text:
        target = simple_model or free_model or base_model
        routing_info["modifiers_applied"].append("security_monitor_downgrade")
        logger.debug("Security monitor detected → %s", target)
        return target, "simple", routing_info

    # --- CLAUDE.md / config injection detection (early return) ---
    # These are background context injections, not user requests.
    # Route to execution tier (cheap model) instead of complex/reasoning.
    last_user_for_injection = all_user_texts_for_role[-1] if all_user_texts_for_role else ""
    if _CONTEXT_INJECTION_PATTERNS.search(last_user_for_injection[:500]):
        target = request_meta.get("execution_model") or simple_model or free_model or base_model
        routing_info["modifiers_applied"].append("context_injection_downgrade")
        logger.debug("CLAUDE.md injection detected → %s (execution)", target)
        return target, "execution", routing_info

    # --- Agentic detection ---
    agentic = detect_agentic(
        messages=messages,
        has_tools=request_meta.get("has_tools", False),
        tool_count=request_meta.get("tool_count", 0),
        system_prompt=system_text,
        system_prompt_length=request_meta.get("system_prompt_length", 0),
        message_count=message_count,
    )
    routing_info["agentic"] = agentic

    # Skip agentic override for Claude Code main session — it always has
    # 200+ tools and long conversations, so agentic detection is meaningless.
    # Let complex_coding / code_review / reasoning handle upgrades instead.
    is_main_session = (
        "You are Claude Code, Anthropic's official CLI" in system_text
        or request_meta.get("system_prompt_length", 0) > 15000
    )

    if agentic["is_agentic"] and final_tier == "simple" and not is_main_session:
        final_model = complex_model
        final_tier = "complex"
        routing_info["modifiers_applied"].append("agentic_override")
        logger.info(
            "Agentic override: simple → complex (confidence=%.2f, signals=%s)",
            agentic["confidence"], agentic["signals"],
        )
    elif agentic["is_agentic"] and is_main_session:
        routing_info["modifiers_applied"].append("agentic_skipped(main_session)")
        logger.debug(
            "Agentic skipped for main session (confidence=%.2f, signals=%s)",
            agentic["confidence"], agentic["signals"],
        )

    # --- Reasoning detection (ONLY check last user message, not conversation history) ---
    all_user_texts = []
    for m in messages:
        role = getattr(m, "role", "")
        text = getattr(m, "text_content", lambda: "")()
        if role == "user":
            all_user_texts.append(text)

    # Only use the LAST user message for reasoning detection
    # Strip <system-reminder> tags to avoid false positives from Claude Code internals
    last_user_text = all_user_texts[-1] if all_user_texts else ""
    last_user_text_clean = re.sub(
        r'<system-reminder>.*?</system-reminder>', '', last_user_text, flags=re.DOTALL
    ).strip()

    # Check if the LAST message in the conversation is a tool result
    # We only skip reasoning detection if we're processing tool results
    # (i.e., last message is tool output), NOT if tool results exist anywhere in history
    last_message_is_tool = False
    if messages:
        last_role = getattr(messages[-1], "role", "")
        last_message_is_tool = (last_role == "tool")

    # DEBUG: Log what we found
    logger.debug(
        "Reasoning detection: last_message_is_tool=%s, last_user_text=%.100s...",
        last_message_is_tool, last_user_text[:100] if last_user_text else "",
    )

    # IMPORTANT: Skip reasoning detection only if the last message is a tool result
    # When the last message is a tool result, we're processing tool output, not new user input.
    # But if the user sends a NEW prompt (even with tool results in history),
    # we should still detect reasoning in that new prompt.
    if last_message_is_tool:
        reasoning = {"is_reasoning": False, "marker_count": 0, "markers": [], "skipped": "last_message_is_tool_result"}
        logger.debug("Reasoning skipped: last message is tool result")
    else:
        reasoning = detect_reasoning(last_user_text_clean, system_text)
        logger.debug(
            "Reasoning detection result: is_reasoning=%s, markers=%s",
            reasoning["is_reasoning"], reasoning["markers"],
        )
    routing_info["reasoning"] = reasoning

    # Token-based Sonnet downgrade: >96k tokens → use Sonnet instead of Opus
    HIGH_TOKEN_THRESHOLD = 96_000
    estimated_tokens = estimate_token_count(messages)
    use_sonnet_for_reasoning = False

    if estimated_tokens > HIGH_TOKEN_THRESHOLD and sonnet_model:
        use_sonnet_for_reasoning = True
        routing_info["high_token_downgrade"] = {
            "estimated_tokens": estimated_tokens,
            "threshold": HIGH_TOKEN_THRESHOLD,
            "target": sonnet_model,
        }

    if reasoning["is_reasoning"]:
        # Use Sonnet if high token count, otherwise use reasoning model
        if use_sonnet_for_reasoning:
            target = sonnet_model or reasoning_model or complex_model
            tier_name = "reasoning_sonnet"
        else:
            target = reasoning_model or complex_model
            tier_name = "reasoning"

        if final_model != target:
            final_model = target
            final_tier = tier_name
            routing_info["modifiers_applied"].append("reasoning_override")
            logger.info(
                "Reasoning override: → %s (markers=%d: %s, tokens=%d)",
                target, reasoning["marker_count"], reasoning["markers"], estimated_tokens,
            )

    # --- Complex coding detection ---
    # Detect complex coding tasks that should route to Sonnet
    # Priority: reasoning > complex > execution > subagent
    complex_coding = detect_complex_coding(
        messages=messages,
        tool_names=tool_names,
        last_tool_call=request_meta.get("last_tool_call"),
        message_count=message_count,
        system_prompt_length=request_meta.get("system_prompt_length", 0),
    )
    routing_info["complex_coding"] = complex_coding

    # Only upgrade to complex if not already reasoning/planning
    if complex_coding["is_complex"] and final_tier not in ("reasoning", "reasoning_sonnet"):
        target = complex_model  # claude-sonnet-4-6
        if final_model != target:
            routing_info["modifiers_applied"].append("complex_coding_override")
            logger.info(
                "Complex coding override: → %s (confidence=%.2f, signals=%s)",
                target, complex_coding["confidence"], complex_coding["signals"],
            )
            final_model = target
            final_tier = "complex"

    # --- Code review detection ---
    # Detect code review/verification tasks that should route to review model (Sonnet)
    # Priority: reasoning > review > complex > execution > subagent
    code_review = detect_code_review(
        prompt=last_user_text_clean,
        system_message=system_text,
    )
    routing_info["code_review"] = code_review

    # Only upgrade to review if not already reasoning
    if code_review["is_review"] and final_tier not in ("reasoning", "reasoning_sonnet"):
        target = review_model or complex_model  # claude-sonnet-4-6
        if final_model != target:
            routing_info["modifiers_applied"].append("code_review_override")
            logger.info(
                "Code review override: → %s (confidence=%.2f, signals=%s)",
                target, code_review["confidence"], code_review["signals"],
            )
            final_model = target
            final_tier = "review"

    # --- Execution detection (route to simple model for execution tasks) ---
    # --- Execution detection (route to simple model for execution tasks) ---
    last_user_text = all_user_texts[-1] if all_user_texts else ""
    last_user_text_clean = re.sub(
        r'<system-reminder>.*?</system-reminder>', '', last_user_text, flags=re.DOTALL
    ).strip()
    execution = detect_execution(
        prompt=last_user_text_clean,
        tool_names=tool_names,
        last_tool_call=request_meta.get("last_tool_call"),
    )
    routing_info["execution"] = execution

    # If execution detected AND not already routed to reasoning/complex/review → use simple model
    # Complex coding tasks and code review may include execution commands, so don't downgrade them
    if execution["is_execution"] and final_tier not in ("reasoning", "reasoning_sonnet", "complex", "review"):
        if final_model != simple_model:
            routing_info["modifiers_applied"].append(
                f"execution_override({final_tier}→simple)"
            )
            logger.info(
                "Execution override: → %s (confidence=%.2f, signals=%s)",
                simple_model, execution["confidence"], execution["signals"],
            )
            final_model = simple_model
            final_tier = "execution"
        elif final_tier == "simple":
            # Already at simple model, but record the execution detection and update tier name
            routing_info["modifiers_applied"].append("execution_detected")
            logger.info(
                "Execution detected (already simple): confidence=%.2f, signals=%s",
                execution["confidence"], execution["signals"],
            )
            final_tier = "execution"

    # ============================================================
    # Claude Code Role Override - Plan Mode Routing Decision
    # ============================================================
    #
    # Plan模式下的请求分类（按驱动类型）:
    #
    # ┌─────────────────────────────────────────────────────────────┐
    # │ 驱动类型          │ 触发条件                 │ 路由目标     │
    # ├─────────────────────────────────────────────────────────────┤
    # │ [USER] 用户启动    │ 新请求（无tool result）  │ Opus        │
    # │                    │ 如 /plan 或拒绝后重生成   │             │
    # ├─────────────────────────────────────────────────────────────┤
    # │ [EXPLORATION]      │ 上轮调用探索工具         │ GLM-5       │
    # │ 探索过程中         │ Read/Bash/Glob返回结果   │             │
    # ├─────────────────────────────────────────────────────────────┤
    # │ [PLAN_GENERATION]  │ 上轮调用 Write/Edit/     │ Opus        │
    # │ 生成/更新Plan      │ ExitPlanMode             │             │
    # ├─────────────────────────────────────────────────────────────┤
    # │ [CONTEXT] 系统上下文│ tool result但无法判断    │ GLM-5       │
    # │ (默认fallback)     │ 上轮工具类型             │             │
    # └─────────────────────────────────────────────────────────────┘
    #
    # 完整流程示例:
    # 用户: /plan create deployment
    #   → [USER] Opus（决策：需要探索）
    #   → Opus调用 Read, Glob
    #   → [EXPLORATION] GLM-5（快速处理结果，继续探索）
    #   → GLM-5调用 Bash, Grep
    #   → [EXPLORATION] GLM-5（继续处理）
    #   → GLM-5判断信息足够，调用 Write
    #   → [PLAN_GENERATION] Opus（生成高质量plan）
    #
    cc_role_type = cc_role.get("role", "unknown")
    if cc_role_type == "planning" and cc_role["confidence"] >= 0.90:

        # --- Step 1: 判断当前请求类型 ---
        last_message_is_tool = False
        if messages:
            last_role = getattr(messages[-1], "role", "")
            last_message_is_tool = (last_role == "tool")

        # --- Step 2: 提取上一轮assistant的工具调用 ---
        last_assistant_tool_calls = []
        for msg in reversed(messages):
            if getattr(msg, "role", "") == "assistant":
                content = getattr(msg, "content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_name = block.get("name", "")
                            if tool_name:
                                last_assistant_tool_calls.append(tool_name)
                break

        # --- Step 3: 工具分类 ---
        exploration_tools = {"Read", "Bash", "Glob", "Grep", "WebFetch", "WebSearch"}
        plan_tools = {"Write", "Edit", "ExitPlanMode", "AskUserQuestion"}

        last_called_exploration = bool(set(last_assistant_tool_calls) & exploration_tools)
        last_called_plan = bool(set(last_assistant_tool_calls) & plan_tools)

        # --- Step 4: 路由决策 ---
        # 分类: [USER] [EXPLORATION] [PLAN_GENERATION] [CONTEXT]
        use_reasoning_model = False
        driver_type = "CONTEXT"  # 默认：系统上下文驱动 → GLM-5
        reason = "context_driven"  # 默认原因

        if not last_message_is_tool:
            # [USER] 用户启动驱动 - 用户发送新请求（如/plan）
            # 首次请求或拒绝后重新生成 → 需要Opus的决策能力
            use_reasoning_model = True
            driver_type = "USER"
            reason = "user_initiated"
        elif last_called_plan:
            # [PLAN_GENERATION] Plan生成/更新驱动
            # 上轮已调用Write/Edit → 正在写plan → 用Opus保证质量
            use_reasoning_model = True
            driver_type = "PLAN_GENERATION"
            reason = f"writing_plan({','.join(last_assistant_tool_calls[:3])})"
        elif last_called_exploration:
            # [EXPLORATION] 探索过程中
            # 上轮调用探索工具 → 快速处理结果，继续探索 → GLM-5
            use_reasoning_model = False
            driver_type = "EXPLORATION"
            reason = f"exploring({','.join(last_assistant_tool_calls[:3])})"
        # else: [CONTEXT] 无法判断上轮工具 → 默认GLM-5

        # --- Step 5: 应用路由 ---
        if use_reasoning_model:
            target = reasoning_model or complex_model
            if final_model != target:
                routing_info["modifiers_applied"].append(f"cc_planning[{driver_type}]")
                logger.info(
                    "Plan routing [%s]: → %s (%s)",
                    driver_type, target, reason,
                )
                final_model = target
                final_tier = "reasoning"
        else:
            # [EXPLORATION] 或 [CONTEXT] → 用GLM-5快速探索
            target = subagent_model or simple_model
            if final_model != target:
                routing_info["modifiers_applied"].append(f"planning[{driver_type}]")
                logger.info(
                    "Plan routing [%s]: → %s (%s)",
                    driver_type, target, reason,
                )
                final_model = target
                final_tier = "subagent"
    elif cc_role_type == "explore" and cc_role["confidence"] >= 0.90:
        # Explore agent: use explore_model for codebase search/exploration
        # 驱动类型: [EXPLORE_AGENT] Claude Code Explore子代理
        target = explore_model or complex_model
        if final_model != target:
            routing_info["modifiers_applied"].append("cc_role[EXPLORE]")
            logger.info(
                "Role routing [EXPLORE]: → %s",
                target,
            )
            final_model = target
            final_tier = "explore"
    elif cc_role_type == "subagent" and cc_role["confidence"] >= 0.60:
        # Subagent/Execution tasks → SUBAGENT model (glm-5, coding plan model)
        # 驱动类型: [SUBAGENT] Claude Code子代理后台任务
        # Priority: reasoning > explore > subagent
        target_subagent_model = subagent_model or free_model or simple_model
        if final_tier not in ("reasoning", "reasoning_sonnet", "explore"):
            # High confidence subagent (explicit markers) can override execution
            # Low confidence subagent (short system prompt) only applies if not already execution
            can_override = cc_role["confidence"] >= 0.85 or final_tier not in ("execution",)

            if final_model != target_subagent_model:
                routing_info["modifiers_applied"].append("cc_role[SUBAGENT]")
                logger.info(
                    "Role routing [SUBAGENT]: → %s (conf=%.2f)",
                    target_subagent_model, cc_role["confidence"],
                )
                final_model = target_subagent_model
                final_tier = "subagent"
            elif final_tier in ("simple", "complex") or (can_override and final_tier == "execution"):
                routing_info["modifiers_applied"].append("cc_role[SUBAGENT]_detected")
                logger.info(
                    "Role routing [SUBAGENT]: already at %s (conf=%.2f)",
                    target_subagent_model, cc_role["confidence"],
                )
                final_tier = "subagent"

    # --- Vision detection ---
    if request_meta.get("has_images", False) and not has_vision(final_model):
        for candidate in [complex_model, simple_model]:
            if has_vision(candidate):
                routing_info["modifiers_applied"].append(
                    f"vision_swap({final_model}\u2192{candidate})"
                )
                logger.info(
                    "Vision swap: %s (no vision) \u2192 %s (vision-capable)",
                    final_model, candidate,
                )
                final_model = candidate
                break
        else:
            logger.warning(
                "Vision request but no vision-capable model in tiers. "
                "Proceeding with %s.", final_model,
            )
    if request_meta.get("has_images", False):
        routing_info["has_images"] = True

    # --- Long context auto-routing to Gemini ---
    # When token count exceeds 200K, route to Gemini 3.1 Pro (2M context window)
    LONG_CONTEXT_THRESHOLD = 200_000
    GEMINI_LONG_CONTEXT_MODEL = "gemini-3.1-pro"

    # Reuse estimated_tokens from earlier calculation
    if estimated_tokens > LONG_CONTEXT_THRESHOLD:
        # Check if Gemini is available and has larger context window
        gemini_info = MODEL_REGISTRY.get(GEMINI_LONG_CONTEXT_MODEL)
        if gemini_info and gemini_info.get("context_window", 0) > estimated_tokens:
            routing_info["modifiers_applied"].append(
                f"long_context_routing({final_model}→{GEMINI_LONG_CONTEXT_MODEL}, tokens={estimated_tokens})"
            )
            logger.info(
                "Long context auto-routing: %s → %s (est=%d tokens > %d threshold)",
                final_model, GEMINI_LONG_CONTEXT_MODEL, estimated_tokens, LONG_CONTEXT_THRESHOLD,
            )
            final_model = GEMINI_LONG_CONTEXT_MODEL
            final_tier = "long_context"

    # --- Context window check ---
    if not check_context_window(final_model, messages):
        window = get_context_window(final_model)
        # Try the other model
        alt_model = complex_model if final_model == simple_model else simple_model
        if check_context_window(alt_model, messages):
            routing_info["modifiers_applied"].append(
                f"context_window_swap({final_model}→{alt_model}, est={estimated_tokens}, limit={window})"
            )
            logger.warning(
                "Context window exceeded for %s (est=%d, limit=%s) → swapping to %s",
                final_model, estimated_tokens, window, alt_model,
            )
            final_model = alt_model
        else:
            logger.warning(
                "Context window exceeded for all models (est=%d tokens). Proceeding with %s.",
                estimated_tokens, final_model,
            )

    routing_info["final_model"] = final_model
    routing_info["final_tier"] = final_tier
    return final_model, final_tier, routing_info
