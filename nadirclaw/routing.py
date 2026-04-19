"""Routing intelligence for NadirClaw.

Handles agentic task detection, reasoning detection, routing profiles,
model aliases, context-window filtering, and session persistence.
"""

import hashlib
import logging
import os
import re
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("nadirclaw.routing")

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
    "deepseek/deepseek-chat": {"context_window": 128_000, "cost_per_m_input": 0.28, "cost_per_m_output": 0.42, "has_vision": False},
    "deepseek/deepseek-reasoner": {"context_window": 128_000, "cost_per_m_input": 0.28, "cost_per_m_output": 0.42, "has_vision": False},
    # Ollama (local, no cost, context varies by model)
    "ollama/llama3.1:8b": {"context_window": 128_000, "cost_per_m_input": 0, "cost_per_m_output": 0, "has_vision": False},
    "ollama/qwen3:32b": {"context_window": 128_000, "cost_per_m_input": 0, "cost_per_m_output": 0, "has_vision": False},
}

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
    "deepseek-r1": "deepseek/deepseek-reasoner",
    "llama": "ollama/llama3.1:8b",
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

_REASONING_MARKERS = re.compile(
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
    r")\b",
    re.IGNORECASE,
)


def detect_reasoning(prompt: str, system_message: str = "") -> Dict[str, Any]:
    """Detect if a prompt requires reasoning capabilities.

    Returns {"is_reasoning": bool, "marker_count": int, "markers": list[str]}.
    """
    combined = f"{system_message} {prompt}"
    matches = _REASONING_MARKERS.findall(combined)
    marker_count = len(matches)

    # 2+ markers = high confidence reasoning (like ClawRouter)
    is_reasoning = marker_count >= 2

    return {
        "is_reasoning": is_reasoning,
        "marker_count": marker_count,
        "markers": list(set(matches)),
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
        routing_info["agent_role"] = agent_role
    else:
        routing_info["agent_role"] = {"role": "unknown", "confidence": 0.0, "signals": []}

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

    routing_info["final_model"] = final_model
    routing_info["final_tier"] = final_tier
    return final_model, final_tier, routing_info
