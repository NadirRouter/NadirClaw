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
    Example: "turbo=gemini-2.5-flash,10+gpt-4.1-nano,5;reasoning=gpt-5.2,8+claude-opus-4-6-20250918,4"
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
    "gpt-4.1": {"context_window": 1_047_576, "cost_per_m_input": 2.00, "cost_per_m_output": 8.00, "has_vision": True},
    "gpt-4.1-mini": {"context_window": 1_047_576, "cost_per_m_input": 0.40, "cost_per_m_output": 1.60, "has_vision": True},
    "gpt-4.1-nano": {"context_window": 1_047_576, "cost_per_m_input": 0.10, "cost_per_m_output": 0.40, "has_vision": True},
    "gpt-5": {"context_window": 400_000, "cost_per_m_input": 1.25, "cost_per_m_output": 10.00, "has_vision": True},
    "gpt-5-mini": {"context_window": 400_000, "cost_per_m_input": 0.25, "cost_per_m_output": 2.00, "has_vision": True},
    "gpt-5.1": {"context_window": 400_000, "cost_per_m_input": 1.25, "cost_per_m_output": 10.00, "has_vision": True},
    "gpt-5.2": {"context_window": 400_000, "cost_per_m_input": 1.75, "cost_per_m_output": 14.00, "has_vision": True},
    "gpt-4o": {"context_window": 128_000, "cost_per_m_input": 2.50, "cost_per_m_output": 10.00, "has_vision": True},
    "gpt-4o-mini": {"context_window": 128_000, "cost_per_m_input": 0.15, "cost_per_m_output": 0.60, "has_vision": True},
    "o3": {"context_window": 200_000, "cost_per_m_input": 2.00, "cost_per_m_output": 8.00, "has_vision": True},
    "o3-mini": {"context_window": 200_000, "cost_per_m_input": 1.10, "cost_per_m_output": 4.40, "has_vision": True},
    "o4-mini": {"context_window": 200_000, "cost_per_m_input": 1.10, "cost_per_m_output": 4.40, "has_vision": True},
    "openai-codex/gpt-5.3-codex": {"context_window": 400_000, "cost_per_m_input": 1.75, "cost_per_m_output": 14.00, "has_vision": False},
    # Anthropic
    "claude-opus-4-6-20250918": {"context_window": 200_000, "cost_per_m_input": 5.00, "cost_per_m_output": 25.00, "has_vision": True},
    "claude-sonnet-4-5-20250929": {"context_window": 200_000, "cost_per_m_input": 3.00, "cost_per_m_output": 15.00, "has_vision": True},
    "claude-haiku-4-5-20251001": {"context_window": 200_000, "cost_per_m_input": 1.00, "cost_per_m_output": 5.00, "has_vision": True},
    "claude-opus-4-20250514": {"context_window": 200_000, "cost_per_m_input": 5.00, "cost_per_m_output": 25.00, "has_vision": True},
    "claude-sonnet-4-20250514": {"context_window": 200_000, "cost_per_m_input": 3.00, "cost_per_m_output": 15.00, "has_vision": True},
    "claude-haiku-4-20250514": {"context_window": 200_000, "cost_per_m_input": 1.00, "cost_per_m_output": 5.00, "has_vision": True},
    # DeepSeek
    "deepseek/deepseek-v4-flash": {"context_window": 1_000_000, "cost_per_m_input": 0.14, "cost_per_m_output": 0.28, "has_vision": False},
    "deepseek/deepseek-v4-pro": {"context_window": 1_000_000, "cost_per_m_input": 1.74, "cost_per_m_output": 3.48, "has_vision": False},
    "deepseek/deepseek-chat": {"context_window": 128_000, "cost_per_m_input": 0.28, "cost_per_m_output": 0.42, "has_vision": False},
    "deepseek/deepseek-reasoner": {"context_window": 128_000, "cost_per_m_input": 0.28, "cost_per_m_output": 0.42, "has_vision": False},
    # Ollama (local, no cost, context varies by model)
    "ollama/llama3.1:8b": {"context_window": 128_000, "cost_per_m_input": 0, "cost_per_m_output": 0, "has_vision": False},
    "ollama/qwen3:32b": {"context_window": 128_000, "cost_per_m_input": 0, "cost_per_m_output": 0, "has_vision": False},
}

BUILTIN_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    model: dict(info) for model, info in MODEL_REGISTRY.items()
}


def _merge_external_model_metadata() -> None:
    """Merge generated and user-local model metadata into MODEL_REGISTRY."""
    from nadirclaw.model_metadata import load_model_metadata, metadata_paths

    for path in metadata_paths():
        if not path.exists():
            continue
        try:
            models = load_model_metadata(path)
        except (OSError, ValueError) as e:
            logger.warning("Skipping invalid model metadata file %s: %s", path, e)
            continue
        for model_id, info in models.items():
            current = MODEL_REGISTRY.get(model_id, {})
            MODEL_REGISTRY[model_id] = {**current, **info}


_merge_external_model_metadata()

# ---------------------------------------------------------------------------
# Model aliases — short names to full model IDs
# ---------------------------------------------------------------------------

MODEL_ALIASES: Dict[str, str] = {
    "sonnet": "claude-sonnet-4-5-20250929",
    "opus": "claude-opus-4-6-20250918",
    "haiku": "claude-haiku-4-5-20251001",
    "claude": "claude-sonnet-4-5-20250929",
    "gpt4": "gpt-4.1",
    "gpt4o": "gpt-4o",
    "gpt4-mini": "gpt-4.1-mini",
    "gpt5": "gpt-5.2",
    "gpt5-mini": "gpt-5-mini",
    "o3": "o3",
    "o3-mini": "o3-mini",
    "o4-mini": "o4-mini",
    "flash": "gemini-2.5-flash",
    "gemini-flash": "gemini-2.5-flash",
    "gemini-pro": "gemini-2.5-pro",
    "deepseek": "deepseek/deepseek-chat",
    "deepseek-v4": "deepseek/deepseek-v4-flash",
    "deepseek-v4-flash": "deepseek/deepseek-v4-flash",
    "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
    "deepseek-r1": "deepseek/deepseek-reasoner",
    "llama": "ollama/llama3.1:8b",
}

# ---------------------------------------------------------------------------
# Routing profiles
# ---------------------------------------------------------------------------

ROUTING_PROFILES = {"auto", "eco", "premium", "free", "reasoning"}


def resolve_profile(model_field: Optional[str]) -> Optional[str]:
    """Check if the model field is a routing profile name.

    Accepts: bare names (`auto`, `eco`, `premium`, `free`, `reasoning`),
    the `nadirclaw/` prefix, and the `nadir-` prefix (e.g. `nadir-eco`)
    which is how the profiles surface in Claude Code's `/model` picker.
    """
    if not model_field:
        return None
    cleaned = model_field.strip().lower()
    if cleaned.startswith("nadirclaw/"):
        cleaned = cleaned[len("nadirclaw/"):]
    elif cleaned.startswith("nadir-"):
        cleaned = cleaned[len("nadir-"):]
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
    """
    score = 0.0
    signals: List[str] = []

    # Tool definitions present
    if has_tools and tool_count >= 1:
        score += 0.35
        signals.append(f"tools_defined({tool_count})")
    if tool_count >= 4:
        score += 0.15
        signals.append("many_tools")

    # Tool-role messages in conversation (active agentic loop)
    tool_msgs = sum(1 for m in messages if getattr(m, "role", None) == "tool")
    if tool_msgs >= 1:
        score += 0.30
        signals.append(f"tool_messages({tool_msgs})")

    # Assistant→tool cycles (multi-step execution)
    cycles = _count_agentic_cycles(messages)
    if cycles >= 2:
        score += 0.20
        signals.append(f"agentic_cycles({cycles})")
    elif cycles == 1:
        score += 0.10
        signals.append("single_cycle")

    # Long system prompt (agents have verbose instructions)
    if system_prompt_length > 500:
        score += 0.10
        signals.append("long_system_prompt")

    # System prompt keywords
    if system_prompt and _AGENTIC_SYSTEM_KEYWORDS.search(system_prompt):
        score += 0.20
        signals.append("agentic_keywords")

    # Many messages (deep conversation / multi-turn loop)
    if message_count > 10:
        score += 0.10
        signals.append("deep_conversation")

    # Cap at 1.0
    confidence = min(score, 1.0)
    is_agentic = confidence >= 0.35

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
    r"|优缺点"
    r"|对比分析"
    r"|权衡.*优劣"
    r"|分析.*利弊"
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


def detect_reasoning(prompt: str, system_message: str = "") -> Dict[str, Any]:
    """Detect if a prompt requires reasoning capabilities.

    Uses separate regexes for English (with \\b word boundaries) and Chinese
    (without \\b, since CJK characters have no word boundaries).

    Returns {"is_reasoning": bool, "marker_count": int, "markers": list[str]}.
    """
    combined = f"{system_message} {prompt}"
    en_matches = _REASONING_MARKERS_EN.findall(combined)
    zh_matches = _REASONING_MARKERS_ZH.findall(combined)
    matches = list(set(en_matches + zh_matches))
    marker_count = len(matches)

    # 2+ markers = high confidence reasoning (like ClawRouter)
    is_reasoning = marker_count >= 2

    return {
        "is_reasoning": is_reasoning,
        "marker_count": marker_count,
        "markers": matches,
    }


# ---------------------------------------------------------------------------
# Complex coding detection
# ---------------------------------------------------------------------------

_CODING_KEYWORDS = [
    r"implement", r"add.*feature", r"refactor", r"optimize", r"improve",
    r"fix.*bug", r"debug", r"troubleshoot", r"create.*feature",
    r"generate.*code", r"build", r"multiple.*files", r"batch",
]


def detect_complex_coding(
    messages: List[Any],
    message_count: int = 0,
) -> Dict[str, Any]:
    """Detect complex coding tasks from recent tool usage patterns.

    Complex coding is signaled by:
    - Heavy editing (3+ Edit/Write calls in recent messages)
    - Tool combination patterns (Read + Edit + Bash)
    - Deep conversations (10+ messages)
    - Coding task keywords in last user message

    Returns {"is_complex": bool, "confidence": float, "signals": list}.
    """
    confidence = 0.0
    signals: List[str] = []

    # Count actual tool calls from last 6 assistant messages
    tool_counts: Dict[str, int] = {}
    assistant_seen = 0
    for m in reversed(messages):
        if getattr(m, "role", "") != "assistant":
            continue
        assistant_seen += 1
        if assistant_seen > 6:
            break
        content = getattr(m, "content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name", "")
                    tool_counts[name] = tool_counts.get(name, 0) + 1

    # Signal 1: Heavy editing
    edit_count = sum(tool_counts.get(t, 0) for t in ("Edit", "Write", "NotebookEdit"))
    if edit_count >= 5:
        confidence += 0.50
        signals.append(f"heavy_editing({edit_count})")
    elif edit_count >= 3:
        confidence += 0.30
        signals.append(f"moderate_editing({edit_count})")

    # Signal 2: Tool combination (Read + Edit + Bash)
    has_read = tool_counts.get("Read", 0) > 0
    has_edit = any(tool_counts.get(t, 0) > 0 for t in ("Edit", "Write"))
    has_bash = tool_counts.get("Bash", 0) > 0
    if has_read and has_edit and has_bash:
        confidence += 0.30
        signals.append("read_edit_bash_combo")
    elif has_read and has_edit:
        confidence += 0.15
        signals.append("read_edit_combo")

    # Signal 3: Deep conversation
    if message_count >= 20:
        confidence += 0.20
        signals.append(f"deep_conversation({message_count})")
    elif message_count >= 10:
        confidence += 0.10
        signals.append(f"moderate_conversation({message_count})")

    # Signal 4: Coding keywords in last user message
    last_user_text = ""
    for m in reversed(messages):
        if getattr(m, "role", "") == "user":
            last_user_text = getattr(m, "text_content", lambda: "")()
            break

    keyword_hits = sum(
        1 for p in _CODING_KEYWORDS
        if re.search(p, last_user_text, re.IGNORECASE)
    )
    if keyword_hits >= 3:
        confidence += 0.40
        signals.append(f"coding_keywords({keyword_hits})")
    elif keyword_hits >= 2:
        confidence += 0.25
        signals.append(f"coding_keywords({keyword_hits})")
    elif keyword_hits >= 1:
        confidence += 0.10
        signals.append(f"coding_keyword({keyword_hits})")

    is_complex = confidence >= 0.50
    return {"is_complex": is_complex, "confidence": min(confidence, 1.0), "signals": signals}


# ---------------------------------------------------------------------------
# Code review detection
# ---------------------------------------------------------------------------

_REVIEW_MARKERS = re.compile(
    r"(code\s*review|review\s*(?:the\s+)?(?:code|changes|pr|diff)"
    r"|pull\s*request\s*review|security\s*(?:audit|review)"
    r"|static\s*analysis|lint\s*check)",
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

    is_review = confidence >= 0.80
    return {"is_review": is_review, "confidence": confidence, "signals": signals}


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
# Claude Code's system prompt is ~35KB; Cursor varies.
# Models with < MAIN_SESSION_MIN_CHARS are classified as subagents.
MAIN_SESSION_MIN_CHARS = 15000  # chars — main session has long system prompt
SHORT_SESSION_MAX_CHARS = 5000  # chars — likely a subagent/background task

_PLANNING_MARKERS = re.compile(
    r"(plan\s*mode\s*is\s*active"
    r"|software\s+architect"
    r"|planning\s+specialist"
    r"|READ-ONLY.*planning"
    r"|architect\s+agent"
    r"|design.*implementation\s+plan)",
    re.IGNORECASE,
)

_EXPLORE_MARKERS = re.compile(
    r"(explore\s+agent"
    r"|explore\s+codebase"
    r"|fast\s+agent\s+specialized\s+for\s+exploring)",
    re.IGNORECASE,
)

_SUBAGENT_MARKERS = re.compile(
    r"(specialized\s+agent"
    r"|subagent"
    r"|background\s+agent"
    r"|search\s+agent)",
    re.IGNORECASE,
)

_EXECUTION_TOOLS = {
    "Bash", "bash", "shell", "execute", "Write", "Edit",
    "Task", "Run", "NotebookEdit",
}


def detect_agent_role(
    system_prompt: str,
    message_count: int = 0,
    tool_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Detect the role/type of an AI coding agent session.

    Examines the system prompt for markers that indicate whether this is a
    planning session, an explore agent, a subagent, or a main execution session.

    Currently tuned for Claude Code. Opt-in via NADIRCLAW_AGENT_ROLE_DETECTION=true.

    Returns {"role": str, "confidence": float, "signals": list[str]}.
    Role can be: "planning", "explore", "subagent", or "unknown".
    """
    role = "unknown"
    confidence = 0.0
    signals: List[str] = []
    tool_names = tool_names or []

    if _PLANNING_MARKERS.search(system_prompt):
        return {"role": "planning", "confidence": 0.95, "signals": ["planning_markers"]}

    if _EXPLORE_MARKERS.search(system_prompt):
        return {"role": "explore", "confidence": 0.95, "signals": ["explore_markers"]}

    # Distinguish subagents from main sessions.
    # Main sessions have long system prompts with extensive instructions.
    is_main_session = len(system_prompt) > MAIN_SESSION_MIN_CHARS

    if not is_main_session and _SUBAGENT_MARKERS.search(system_prompt):
        return {"role": "subagent", "confidence": 0.90, "signals": ["subagent_markers"]}

    if not is_main_session and len(system_prompt) < SHORT_SESSION_MAX_CHARS:
        role = "subagent"
        confidence = 0.60  # Matches the routing threshold for subagent tier
        signals.append("short_system_prompt")

    return {"role": role, "confidence": confidence, "signals": signals}


def _get_last_assistant_tool_calls(messages: List[Any]) -> List[str]:
    """Extract tool names from the last assistant message with tool_use blocks."""
    for msg in reversed(messages):
        if getattr(msg, "role", "") != "assistant":
            continue
        content = getattr(msg, "content", [])
        if not isinstance(content, list):
            continue
        calls = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = block.get("name", "")
                if name:
                    calls.append(name)
        return calls
    return []


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
    window = info.get("context_window")
    if not window:
        return True
    estimated = estimate_token_count(messages)
    return estimated < window


def get_context_window(model: str) -> Optional[int]:
    """Return context window for a model, or None if unknown."""
    info = MODEL_REGISTRY.get(model)
    if not info:
        return None
    window = info.get("context_window")
    return int(window) if window else None


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

    Upgrade-only policy: cached tier can only escalate (simple→mid→complex→
    reasoning), never downgrade.  This prevents a complex session from being
    pinned to "simple" while still avoiding jarring model switches downward.
    """

    # Tier ordering — higher index = more capable model.
    TIER_ORDER = {"simple": 0, "mid": 1, "complex": 2, "reasoning": 3}

    def __init__(self, ttl_seconds: int = 300, max_size: int = 10_000):
        # OrderedDict gives O(1) move-to-end (move_to_end) and O(1) popitem(last=False)
        # for LRU eviction — replaces the old List-based access_order which was O(n).
        self._cache: OrderedDict[str, Tuple[str, str, float]] = OrderedDict()  # key → (model, tier, timestamp)
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._cleanup_counter = 0
        self._cleanup_interval = 100  # run cleanup every N puts
        self._lock = Lock()

    def _make_key(self, messages: List[Any]) -> str:
        """Generate a session key from conversation shape."""
        parts: List[str] = []
        for m in messages:
            role = getattr(m, "role", "")
            if role in ("system", "developer"):
                content = getattr(m, "text_content", lambda: "")()
                parts.append(f"sys:{content[:200]}")
                break

        # First user message
        for m in messages:
            role = getattr(m, "role", "")
            if role == "user":
                content = getattr(m, "text_content", lambda: "")()
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
        """Return (model, tier) if a session exists and isn't expired.

        The caller is expected to *always* run the classifier after this.
        If the new classification yields a higher tier, call
        ``upgrade_if_higher`` to atomically escalate the cached entry.
        """
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

    def upgrade_if_higher(
        self, messages: List[Any], new_model: str, new_tier: str
    ) -> Tuple[str, str, str]:
        """Upgrade the cached tier if *new_tier* outranks the stored one.

        Returns ``(model, tier, status)`` where status is one of:

        - ``"new"``      — no entry existed (or was expired); fresh values stored
        - ``"upgraded"`` — cached tier was lower; entry replaced with higher tier
        - ``"kept"``     — cached tier was equal or higher; cached values returned

        Expired entries are treated as missing so a stale high-tier entry
        cannot block a fresh classification.
        """
        key = self._make_key(messages)
        new_rank = self.TIER_ORDER.get(new_tier, 0)
        now = time.time()
        with self._lock:
            entry = self._cache.get(key)
            # Treat expired entries as missing — fresh classification wins.
            if entry is not None and now - entry[2] > self._ttl:
                del self._cache[key]
                entry = None
            if entry is None:
                self._cache[key] = (new_model, new_tier, now)
                self._evict_lru()
                return new_model, new_tier, "new"
            cached_model, cached_tier, _ts = entry
            cached_rank = self.TIER_ORDER.get(cached_tier, 0)
            if new_rank > cached_rank:
                # Escalate — upgrade the cache entry.
                self._cache[key] = (new_model, new_tier, now)
                self._touch(key)
                return new_model, new_tier, "upgraded"
            # Keep the existing (equal or higher) tier.
            self._touch(key)
            return cached_model, cached_tier, "kept"

    def put(self, messages: List[Any], model: str, tier: str) -> None:
        """Store a routing decision for this session (upgrade-only).

        If an entry already exists with a higher tier, this is a no-op.
        """
        key = self._make_key(messages)
        new_rank = self.TIER_ORDER.get(tier, 0)
        with self._lock:
            # Periodic cleanup of expired entries
            self._cleanup_counter += 1
            if self._cleanup_counter >= self._cleanup_interval:
                self._cleanup_counter = 0
                self.clear_expired()

            # Upgrade-only: don't downgrade an existing entry.
            existing = self._cache.get(key)
            if existing is not None:
                _, cached_tier, _ = existing
                if self.TIER_ORDER.get(cached_tier, 0) >= new_rank:
                    return  # existing tier is equal or higher — skip

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
_session_cache = SessionCache(ttl_seconds=300)


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
    input_rate = info.get("cost_per_m_input")
    output_rate = info.get("cost_per_m_output")
    if input_rate is None or output_rate is None:
        return None
    input_cost = (prompt_tokens / 1_000_000) * input_rate
    output_cost = (completion_tokens / 1_000_000) * output_rate
    return input_cost + output_cost


# ---------------------------------------------------------------------------
# Main routing modifier — applies all intelligence
# ---------------------------------------------------------------------------

def _apply_agent_role_routing(
    agent_role: Dict[str, Any],
    messages: List[Any],
    final_model: str,
    final_tier: str,
    simple_model: str,
    complex_model: str,
    reasoning_model: Optional[str],
    explore_model: Optional[str],
    subagent_model: Optional[str],
    free_model: Optional[str],
    routing_info: Dict[str, Any],
) -> None:
    """Apply agent role-based routing decisions.

    Mutates routing_info by setting final_model/final_tier and appending
    modifiers. The caller reads these back and removes the temp keys.
    """
    role_type = agent_role.get("role", "unknown")
    confidence = agent_role.get("confidence", 0.0)

    if role_type == "planning" and confidence >= 0.90:
        _route_planning_session(
            messages, final_model, final_tier,
            simple_model, complex_model, reasoning_model,
            subagent_model, free_model, routing_info,
        )
    elif role_type == "explore" and confidence >= 0.90:
        target = explore_model or complex_model
        routing_info["modifiers_applied"].append("agent_role[EXPLORE]")
        logger.info("Role routing [EXPLORE]: → %s", target)
        routing_info["final_model"] = target
        routing_info["final_tier"] = "explore"
        return

    elif role_type == "subagent" and confidence >= 0.60:
        target = subagent_model or free_model or simple_model
        if final_tier not in ("reasoning", "explore"):
            routing_info["modifiers_applied"].append("agent_role[SUBAGENT]")
            logger.info("Role routing [SUBAGENT]: → %s (conf=%.2f)", target, confidence)
            routing_info["final_model"] = target
            routing_info["final_tier"] = "subagent"
            return

    # No role override — pass through current values
    routing_info["final_model"] = final_model
    routing_info["final_tier"] = final_tier


def _route_planning_session(
    messages: List[Any],
    final_model: str,
    final_tier: str,
    simple_model: str,
    complex_model: str,
    reasoning_model: Optional[str],
    subagent_model: Optional[str],
    free_model: Optional[str],
    routing_info: Dict[str, Any],
) -> None:
    """Route planning sessions based on the driving phase.

    Planning phases:
    - USER: new user request (no tool result) → reasoning model for decision-making
    - EXPLORATION: last tool call was exploration (Read, Glob, etc.) → fast model
    - PLAN_GENERATION: last tool call was write/edit → reasoning model for quality
    - CONTEXT: indeterminate → fast model (default)
    """
    last_message_is_tool = False
    if messages:
        last_message_is_tool = getattr(messages[-1], "role", "") == "tool"

    last_tool_calls = _get_last_assistant_tool_calls(messages)
    exploration_tools = {"Read", "Bash", "Glob", "Grep", "WebFetch", "WebSearch"}
    plan_tools = {"Write", "Edit", "ExitPlanMode", "AskUserQuestion"}

    called_exploration = bool(set(last_tool_calls) & exploration_tools)
    called_plan = bool(set(last_tool_calls) & plan_tools)

    use_reasoning = False
    driver = "CONTEXT"

    if not last_message_is_tool:
        use_reasoning = True
        driver = "USER"
    elif called_plan:
        use_reasoning = True
        driver = "PLAN_GENERATION"
    elif called_exploration:
        use_reasoning = False
        driver = "EXPLORATION"

    if use_reasoning:
        target = reasoning_model or complex_model
        routing_info["modifiers_applied"].append(f"planning[{driver}]")
        logger.info("Plan routing [%s]: → %s", driver, target)
        routing_info["final_model"] = target
        routing_info["final_tier"] = "reasoning"
    else:
        target = subagent_model or free_model or simple_model
        routing_info["modifiers_applied"].append(f"planning[{driver}]")
        logger.info("Plan routing [%s]: → %s", driver, target)
        routing_info["final_model"] = target
        routing_info["final_tier"] = "subagent"


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
    explore_model: Optional[str] = None,
    subagent_model: Optional[str] = None,
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

    # --- Agent role detection ---
    system_text = request_meta.get("system_prompt_text", "")
    tool_names = request_meta.get("tool_names", [])
    message_count = request_meta.get("message_count", 0)

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
    else:
        agent_role = {"role": "unknown", "confidence": 0.0, "signals": []}
    routing_info["agent_role"] = agent_role

    # --- Agentic detection ---
    agentic = detect_agentic(
        messages=messages,
        has_tools=request_meta.get("has_tools", False),
        tool_count=request_meta.get("tool_count", 0),
        system_prompt=request_meta.get("system_prompt_text", ""),
        system_prompt_length=request_meta.get("system_prompt_length", 0),
        message_count=request_meta.get("message_count", 0),
    )
    routing_info["agentic"] = agentic

    if agentic["is_agentic"] and final_tier == "simple":
        final_model = complex_model
        final_tier = "complex"
        routing_info["modifiers_applied"].append("agentic_override")
        logger.info(
            "Agentic override: simple → complex (confidence=%.2f, signals=%s)",
            agentic["confidence"], agentic["signals"],
        )

    # --- Reasoning detection ---
    prompt_text = ""
    system_text = ""
    for m in messages:
        role = getattr(m, "role", "")
        text = getattr(m, "text_content", lambda: "")()
        if role == "user":
            prompt_text = text
        elif role in ("system", "developer"):
            system_text = text

    reasoning = detect_reasoning(prompt_text, system_text)
    routing_info["reasoning"] = reasoning

    if reasoning["is_reasoning"]:
        target = reasoning_model or complex_model
        if final_model != target:
            final_model = target
            final_tier = "reasoning"
            routing_info["modifiers_applied"].append("reasoning_override")
            logger.info(
                "Reasoning override: → %s (markers=%d: %s)",
                target, reasoning["marker_count"], reasoning["markers"],
            )

    # --- Agent role-based routing ---
    _apply_agent_role_routing(
        agent_role, messages, final_model, final_tier,
        simple_model, complex_model, reasoning_model,
        explore_model, subagent_model, free_model,
        routing_info,
    )
    final_model = routing_info["final_model"]
    final_tier = routing_info["final_tier"]
    # Clean up temp keys set by _apply_agent_role_routing
    routing_info.pop("final_model", None)
    routing_info.pop("final_tier", None)

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

    # --- Context window check ---
    if not check_context_window(final_model, messages):
        estimated = estimate_token_count(messages)
        window = get_context_window(final_model)
        # Try the other model
        alt_model = complex_model if final_model == simple_model else simple_model
        if check_context_window(alt_model, messages):
            routing_info["modifiers_applied"].append(
                f"context_window_swap({final_model}→{alt_model}, est={estimated}, limit={window})"
            )
            logger.warning(
                "Context window exceeded for %s (est=%d, limit=%s) → swapping to %s",
                final_model, estimated, window, alt_model,
            )
            final_model = alt_model
        else:
            logger.warning(
                "Context window exceeded for all models (est=%d tokens). Proceeding with %s.",
                estimated, final_model,
            )

    # --- Model Pool Selection ---
    # If the final model belongs to a pool, select from the pool based on weights.
    # Skip pool override for tiers where the model was explicitly chosen by reasoning
    # or agentic detection — pool selection is for load-balancing equivalent models.
    pool_name = get_pool_for_model(final_model)
    if pool_name and final_tier not in ("complex", "reasoning"):
        original_model = final_model
        final_model = select_from_pool(pool_name)
        if final_model != original_model:
            routing_info["modifiers_applied"].append(
                f"pool_selection({pool_name}: {original_model}→{final_model})"
            )
            logger.info(
                "Model pool %s: %s → %s", pool_name, original_model, final_model,
            )
        routing_info["pool_name"] = pool_name

    routing_info["final_model"] = final_model
    routing_info["final_tier"] = final_tier
    return final_model, final_tier, routing_info
