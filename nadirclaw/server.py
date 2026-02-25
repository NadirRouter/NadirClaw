"""
NadirClaw — Lightweight LLM router server.

Routes simple prompts to cheap/local models and complex prompts to premium models.
OpenAI-compatible API at /v1/chat/completions.
"""

import asyncio
import collections
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from nadirclaw import __version__
from nadirclaw.auth import UserSession, validate_local_auth
from nadirclaw.settings import settings

logger = logging.getLogger("nadirclaw")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RateLimitExhausted(Exception):
    """Raised when a model's rate limit is exhausted after retries."""

    def __init__(self, model: str, retry_after: int = 60):
        self.model = model
        self.retry_after = retry_after
        super().__init__(f"Rate limit exhausted for {model} (retry in {retry_after}s)")


# ---------------------------------------------------------------------------
# Request rate limiter (in-memory, per user)
# ---------------------------------------------------------------------------

_MAX_CONTENT_LENGTH = 1_000_000  # 1 MB total across all messages


class _RateLimiter:
    """Sliding-window rate limiter keyed by user ID."""

    def __init__(self, max_requests: int = 120, window_seconds: int = 60):
        self._max = max_requests
        self._window = window_seconds
        self._hits: Dict[str, collections.deque] = {}

    def check(self, key: str) -> Optional[int]:
        """Return seconds until retry if rate-limited, else None."""
        now = time.time()
        q = self._hits.setdefault(key, collections.deque())

        # Evict timestamps outside the window
        while q and q[0] <= now - self._window:
            q.popleft()

        if len(q) >= self._max:
            retry_after = int(q[0] + self._window - now) + 1
            return retry_after

        q.append(now)
        return None


_rate_limiter = _RateLimiter()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NadirClaw",
    version=__version__,
    description="Open-source LLM router — simple prompts to free models, complex to premium",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Validation error handler — log request body for debugging
# ---------------------------------------------------------------------------

from fastapi.exceptions import RequestValidationError


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.error(
        "Validation error on %s %s: %s\nBody: %s",
        request.method,
        request.url.path,
        exc.errors(),
        body[:2000].decode("utf-8", errors="replace"),
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    model_config = {"extra": "allow"}
    role: str
    content: Optional[Union[str, List[Any]]] = None

    def text_content(self) -> str:
        """Extract plain text from content (handles both str and multi-modal array)."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        # Multi-modal: [{"type": "text", "text": "..."}, ...]
        parts = []
        for item in self.content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "allow"}
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False


class ClassifyRequest(BaseModel):
    prompt: str
    system_message: Optional[str] = ""


class ClassifyBatchRequest(BaseModel):
    prompts: List[str]


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

_log_lock = Lock()


def _log_request(entry: Dict[str, Any]) -> None:
    """Append a JSON line to the request log and print to console."""
    log_dir = settings.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    request_log = log_dir / "requests.jsonl"

    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(entry, default=str) + "\n"
    with _log_lock:
        with open(request_log, "a") as f:
            f.write(line)

    tier = entry.get("tier", "?")
    model = entry.get("selected_model", "?")
    conf = entry.get("confidence", 0)
    score = entry.get("complexity_score", 0)
    prompt_preview = entry.get("prompt", "")[:80]
    latency = entry.get("classifier_latency_ms", "?")
    total = entry.get("total_latency_ms", "?")
    logger.info(
        "%-8s model=%-35s conf=%.3f score=%.2f lat=%sms total=%sms  \"%s\"",
        tier, model, conf, score, latency, total, prompt_preview,
    )


def _extract_request_metadata(request: ChatCompletionRequest) -> Dict[str, Any]:
    """Extract structured metadata from a ChatCompletionRequest for logging."""
    messages = request.messages
    system_msgs = [m for m in messages if m.role in ("system", "developer")]
    has_system = bool(system_msgs)
    system_len = sum(len(m.text_content()) for m in system_msgs) if has_system else 0

    # Tool definitions from model_extra (OpenAI-style "tools" field)
    extra = request.model_extra or {}
    tool_defs = extra.get("tools") or []
    # Tool-role messages (tool results in conversation)
    tool_msgs = [m for m in messages if m.role == "tool"]
    tool_count = len(tool_defs) + len(tool_msgs)

    system_text = " ".join(m.text_content() for m in system_msgs) if has_system else ""

    return {
        "stream": bool(request.stream),
        "message_count": len(messages),
        "has_system_prompt": has_system,
        "system_prompt_length": system_len,
        "system_prompt_text": system_text,
        "has_tools": tool_count > 0,
        "tool_count": tool_count,
        "requested_model": request.model,
    }


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    log_dir = settings.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    request_log = log_dir / "requests.jsonl"

    logger.info("=" * 60)
    logger.info("NadirClaw starting...")
    logger.info("Log file: %s", request_log.resolve())
    logger.info("=" * 60)

    # Optional OpenTelemetry
    from nadirclaw.telemetry import instrument_fastapi, setup_telemetry

    if setup_telemetry("nadirclaw"):
        instrument_fastapi(app)

    # Warm up the binary classifier
    try:
        from nadirclaw.classifier import warmup
        logger.info("Warming up binary classifier...")
        warmup()
        logger.info("Binary classifier ready")
    except Exception as e:
        logger.error("Failed to warm binary classifier: %s", e)
        raise

    # Show config
    try:
        import litellm
        litellm.set_verbose = False
        logger.info("Simple model:  %s", settings.SIMPLE_MODEL)
        logger.info("Complex model: %s", settings.COMPLEX_MODEL)
        if settings.has_explicit_tiers:
            logger.info("Tier config:   explicit (env vars)")
        else:
            logger.info("Tier config:   derived from NADIRCLAW_MODELS")
        logger.info("Ollama base:   %s", settings.OLLAMA_API_BASE)
        token = settings.AUTH_TOKEN
        if token:
            logger.info("Auth:          %s***", token[:6] if len(token) >= 6 else token)
        else:
            logger.info("Auth:          disabled (local-only)")
        # Log credential status
        from nadirclaw.credentials import detect_provider, get_credential_source

        for model in settings.tier_models:
            provider = detect_provider(model)
            if provider and provider != "ollama":
                source = get_credential_source(provider)
                if source:
                    logger.info("Credential:    %s → %s", provider, source)
                else:
                    logger.warning("Credential:    %s → NOT CONFIGURED", provider)

    except Exception as e:
        logger.warning("LiteLLM setup issue: %s", e)

    logger.info("Ready! Listening for requests...")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Smart routing internals
# ---------------------------------------------------------------------------

async def _smart_route_analysis(
    prompt: str, system_message: str, user: UserSession
) -> tuple:
    """Run classifier, return (selected_model, analysis_dict). No LLM call."""
    from nadirclaw.classifier import get_binary_classifier
    from nadirclaw.telemetry import trace_span

    with trace_span("smart_route_analysis") as span:
        analyzer = get_binary_classifier()
        result = await analyzer.analyze(text=prompt, system_message=system_message)

        is_complex = result.get("tier_name") == "complex"
        selected = settings.COMPLEX_MODEL if is_complex else settings.SIMPLE_MODEL

        analysis = {
            "strategy": "smart-routing",
            "analyzer": result.get("analyzer_type", "binary"),
            "selected_model": selected,
            "complexity_score": result.get("complexity_score"),
            "tier": result.get("tier_name"),
            "confidence": result.get("confidence"),
            "reasoning": result.get("reasoning"),
            "classifier_latency_ms": result.get("analyzer_latency_ms"),
            "simple_model": settings.SIMPLE_MODEL,
            "complex_model": settings.COMPLEX_MODEL,
            "ranked_models": [
                {"model": m.get("model_name"), "score": m.get("suitability_score")}
                for m in result.get("ranked_models", [])[:5]
            ],
        }

        if span:
            span.set_attribute("nadirclaw.tier", analysis["tier"] or "")
            span.set_attribute("nadirclaw.selected_model", selected)

    return selected, analysis


async def _smart_route_full(
    messages: List[ChatMessage], user: UserSession
) -> tuple:
    """Smart route for full completions."""
    user_msgs = [m.text_content() for m in messages if m.role == "user"]
    prompt = user_msgs[-1] if user_msgs else ""
    system_msg = next((m.text_content() for m in messages if m.role in ("system", "developer")), "")
    return await _smart_route_analysis(prompt, system_msg, user)


# ---------------------------------------------------------------------------
# /v1/classify — dry-run classification (no LLM call)
# ---------------------------------------------------------------------------

@app.post("/v1/classify")
async def classify_prompt(
    request: ClassifyRequest,
    current_user: UserSession = Depends(validate_local_auth),
) -> Dict[str, Any]:
    """Classify a prompt without calling any LLM."""
    _, analysis = await _smart_route_analysis(
        request.prompt, request.system_message or "", current_user
    )

    _log_request({
        "type": "classify",
        "prompt": request.prompt,
        **analysis,
    })

    return {
        "prompt": request.prompt,
        "classification": analysis,
    }


@app.post("/v1/classify/batch")
async def classify_batch(
    request: ClassifyBatchRequest,
    current_user: UserSession = Depends(validate_local_auth),
) -> Dict[str, Any]:
    """Classify multiple prompts at once."""
    results = []
    for prompt in request.prompts:
        _, analysis = await _smart_route_analysis(prompt, "", current_user)
        results.append({
            "prompt": prompt,
            "selected_model": analysis.get("selected_model"),
            "tier": analysis.get("tier"),
            "confidence": analysis.get("confidence"),
            "complexity_score": analysis.get("complexity_score"),
        })
        _log_request({"type": "classify_batch", "prompt": prompt, **analysis})

    simple_count = sum(1 for r in results if r["tier"] == "simple")
    complex_count = sum(1 for r in results if r["tier"] == "complex")

    return {
        "total": len(results),
        "simple": simple_count,
        "complex": complex_count,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Model call helpers
# ---------------------------------------------------------------------------

def _strip_gemini_prefix(model: str) -> str:
    """Remove 'gemini/' prefix if present (LiteLLM style → native name)."""
    return model.removeprefix("gemini/")


# Shared Gemini clients — reused across requests, keyed by API key.
# A lock ensures concurrent requests with different keys don't race.
_gemini_clients: Dict[str, Any] = {}
_gemini_client_lock = Lock()

# Bounded thread pool for Gemini calls. Caps the number of concurrent
# (and leaked-on-timeout) threads so they can't grow unbounded.
_gemini_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="gemini")


def _get_gemini_client(api_key: str):
    """Get or create a thread-safe, per-key google-genai Client."""
    with _gemini_client_lock:
        if api_key not in _gemini_clients:
            from google import genai
            _gemini_clients[api_key] = genai.Client(api_key=api_key)
        return _gemini_clients[api_key]


async def _call_gemini(
    model: str,
    request: "ChatCompletionRequest",
    provider: str,
    _retry_count: int = 0,
) -> Dict[str, Any]:
    """Call a Gemini model using the native Google GenAI SDK.

    Handles 429 rate-limit errors with automatic retry (up to 3 attempts).
    """
    import asyncio
    import re

    from google.genai import types
    from google.genai.errors import ClientError

    from nadirclaw.credentials import get_credential

    MAX_RETRIES = 1  # Keep low — fallback handles the rest

    api_key = get_credential(provider)
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="No Google/Gemini API key configured. "
                   "Set GEMINI_API_KEY or GOOGLE_API_KEY, or run: nadirclaw auth add -p google",
        )

    client = _get_gemini_client(api_key)
    native_model = _strip_gemini_prefix(model)

    # Build contents: separate system instruction from conversation messages
    system_parts = []
    contents = []
    for m in request.messages:
        if m.role in ("system", "developer"):
            system_parts.append(m.text_content())
        else:
            contents.append(
                types.Content(
                    role="user" if m.role == "user" else "model",
                    parts=[types.Part.from_text(text=m.text_content())],
                )
            )

    # Build generation config
    gen_config_kwargs: Dict[str, Any] = {}
    if request.temperature is not None:
        gen_config_kwargs["temperature"] = request.temperature
    if request.max_tokens is not None:
        gen_config_kwargs["max_output_tokens"] = request.max_tokens
    if request.top_p is not None:
        gen_config_kwargs["top_p"] = request.top_p

    # NOTE: Function call parts are filtered out programmatically when
    # extracting the response (see "handle function_call parts" below),
    # so no prompt-level instruction is needed here.

    generate_kwargs: Dict[str, Any] = {
        "model": native_model,
        "contents": contents,
    }
    if gen_config_kwargs:
        generate_kwargs["config"] = types.GenerateContentConfig(
            **gen_config_kwargs,
            system_instruction="\n".join(system_parts) if system_parts else None,
        )
    elif system_parts:
        generate_kwargs["config"] = types.GenerateContentConfig(
            system_instruction="\n".join(system_parts),
        )

    logger.debug("Calling Gemini: model=%s (attempt %d/%d)", native_model, _retry_count + 1, MAX_RETRIES + 1)

    # The google-genai SDK is synchronous; run in a bounded thread pool
    # so timed-out threads don't accumulate unboundedly.
    loop = asyncio.get_running_loop()
    try:
        response = await asyncio.wait_for(
            loop.run_in_executor(
                _gemini_executor,
                lambda: client.models.generate_content(**generate_kwargs),
            ),
            timeout=120,  # 2 minute hard timeout
        )
    except asyncio.TimeoutError:
        logger.error("Gemini API call timed out after 120s for model=%s", native_model)
        return {
            "content": "The model took too long to respond. Please try again.",
            "finish_reason": "stop",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
    except ClientError as e:
        # Handle 429 rate-limit / quota errors with retry
        if e.code == 429 or "RESOURCE_EXHAUSTED" in str(e):
            # Try to extract retry delay from error message
            retry_delay = 60  # default
            err_str = str(e)
            delay_match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
            if delay_match:
                retry_delay = min(int(float(delay_match.group(1))) + 2, 120)

            if _retry_count < MAX_RETRIES:
                logger.warning(
                    "Gemini 429 rate limit for model=%s — retrying in %ds (attempt %d/%d)",
                    native_model, retry_delay, _retry_count + 1, MAX_RETRIES,
                )
                await asyncio.sleep(retry_delay)
                return await _call_gemini(model, request, provider, _retry_count + 1)
            else:
                # Exhausted retries — raise so the caller can try a fallback model
                logger.error(
                    "Gemini 429 rate limit persists after %d retries for model=%s. "
                    "Free tier limit reached. Raising RateLimitExhausted for fallback.",
                    MAX_RETRIES, native_model,
                )
                raise RateLimitExhausted(model=model, retry_after=retry_delay)
        # Non-429 client errors — re-raise
        raise

    # Extract usage metadata
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    completion_tokens = getattr(usage, "candidates_token_count", 0) or 0

    # Extract finish reason and content
    finish_reason = "stop"
    content = ""

    if response.candidates:
        candidate = response.candidates[0]
        raw_reason = getattr(candidate, "finish_reason", None)
        if raw_reason:
            reason_str = str(raw_reason).lower()
            if "safety" in reason_str:
                finish_reason = "content_filter"
            elif "length" in reason_str or "max_tokens" in reason_str:
                finish_reason = "length"
            logger.debug("Gemini finish_reason: %s", reason_str)

        # Extract text from parts (handle function_call parts gracefully)
        if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
            text_parts = []
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
                elif hasattr(part, "function_call") and part.function_call:
                    logger.info("Gemini returned function_call: %s (ignoring — NadirClaw doesn't execute tools)", part.function_call.name)
            content = "".join(text_parts)
    else:
        # No candidates — check for prompt feedback (safety block)
        feedback = getattr(response, "prompt_feedback", None)
        if feedback:
            logger.warning("Gemini blocked request: %s", feedback)

    if not content:
        # Try response.text as a fallback
        try:
            content = response.text or ""
        except (ValueError, AttributeError):
            content = ""
        if not content:
            logger.warning(
                "Gemini returned empty content for model=%s (finish_reason=%s, candidates=%d)",
                native_model, finish_reason, len(response.candidates) if response.candidates else 0,
            )

    return {
        "content": content,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


async def _call_litellm(
    model: str,
    request: "ChatCompletionRequest",
    provider: str | None,
) -> Dict[str, Any]:
    """Call a model via LiteLLM (Anthropic, OpenAI, Ollama, etc.)."""
    import litellm

    from nadirclaw.credentials import get_credential

    # For openai-codex provider, strip the prefix and route as OpenAI model
    if provider == "openai-codex":
        litellm_model = model.removeprefix("openai-codex/")
        cred_provider = "openai-codex"
    else:
        litellm_model = model
        cred_provider = provider

    # LiteLLM's "ollama/" provider uses /api/generate which doesn't support
    # tool calling. Automatically upgrade to "ollama_chat/" (which uses
    # /api/chat) when the request includes tool definitions.
    req_extra = request.model_extra or {}
    if litellm_model.startswith("ollama/") and req_extra.get("tools"):
        litellm_model = "ollama_chat/" + litellm_model.removeprefix("ollama/")
        logger.debug("Upgraded ollama → ollama_chat for tool support: %s", litellm_model)

    # Preserve full message structure (tool_calls, tool_call_id, name, etc.)
    messages = []
    for message in request.messages:
        # Use text_content() for readable text, but preserve None for
        # tool-calling assistant messages (OpenAI/Anthropic require null).
        text = message.text_content()
        msg: dict[str, Any] = {"role": message.role, "content": text if text else message.content}
        extra_fields = message.model_extra or {}
        if "tool_calls" in extra_fields:
            msg["tool_calls"] = extra_fields["tool_calls"]
        if "tool_call_id" in extra_fields:
            msg["tool_call_id"] = extra_fields["tool_call_id"]
        if "name" in extra_fields:
            msg["name"] = extra_fields["name"]
        messages.append(msg)

    call_kwargs: Dict[str, Any] = {"model": litellm_model, "messages": messages}
    if request.temperature is not None:
        call_kwargs["temperature"] = request.temperature
    if request.max_tokens is not None:
        call_kwargs["max_tokens"] = request.max_tokens
    if request.top_p is not None:
        call_kwargs["top_p"] = request.top_p

    # Pass through tool definitions and tool_choice
    extra = request.model_extra or {}
    if extra.get("tools"):
        call_kwargs["tools"] = extra["tools"]
    if extra.get("tool_choice"):
        call_kwargs["tool_choice"] = extra["tool_choice"]

    if cred_provider and cred_provider != "ollama":
        api_key = get_credential(cred_provider)
        if api_key:
            call_kwargs["api_key"] = api_key

    # Pass api_base for Ollama models so LiteLLM reaches the right host
    if litellm_model.startswith("ollama/") or litellm_model.startswith("ollama_chat/"):
        call_kwargs["api_base"] = settings.OLLAMA_API_BASE

    logger.debug("Calling LiteLLM: model=%s (provider=%s)", litellm_model, provider)
    try:
        response = await litellm.acompletion(**call_kwargs)
    except Exception as e:
        # Catch rate limit errors from any provider through LiteLLM
        err_str = str(e).lower()
        if "429" in err_str or "rate" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
            logger.warning("LiteLLM 429 rate limit for model=%s: %s", litellm_model, e)
            raise RateLimitExhausted(model=model, retry_after=60)
        raise

    msg = response.choices[0].message
    result: dict[str, Any] = {
        "content": msg.content,
        "finish_reason": response.choices[0].finish_reason or "stop",
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
    }

    # Preserve tool_calls from LLM response
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = [
            tc.model_dump() if hasattr(tc, "model_dump") else tc
            for tc in tool_calls
        ]

    return result


# ---------------------------------------------------------------------------
# Model dispatch + fallback on rate limit
# ---------------------------------------------------------------------------

async def _dispatch_model(
    model: str,
    request: "ChatCompletionRequest",
    provider: str | None,
) -> Dict[str, Any]:
    """Call the right backend (Gemini native or LiteLLM) for a model.

    Raises RateLimitExhausted if the model is rate-limited after retries.
    """
    from nadirclaw.telemetry import trace_span

    with trace_span("dispatch_model", {"gen_ai.request.model": model, "gen_ai.system": provider or ""}):
        if provider == "google":
            return await _call_gemini(model, request, provider)
        return await _call_litellm(model, request, provider)


async def _call_with_fallback(
    selected_model: str,
    request: "ChatCompletionRequest",
    provider: str | None,
    analysis_info: Dict[str, Any],
) -> tuple:
    """Try the selected model; on rate limit, fall back to the other tier.

    The primary model retries up to MAX_RETRIES times.
    The fallback model is tried once (no retries) to avoid long waits.

    Returns (response_data, actual_model_used, updated_analysis_info).
    """
    from nadirclaw.credentials import detect_provider

    try:
        response_data = await _dispatch_model(selected_model, request, provider)
        return response_data, selected_model, analysis_info
    except RateLimitExhausted as e:
        # Determine fallback model (swap tiers)
        if selected_model == settings.SIMPLE_MODEL:
            fallback_model = settings.COMPLEX_MODEL
        elif selected_model == settings.COMPLEX_MODEL:
            fallback_model = settings.SIMPLE_MODEL
        else:
            # Direct model request — try simple first, then complex
            fallback_model = settings.SIMPLE_MODEL

        # Don't fall back to the same model
        if fallback_model == selected_model:
            logger.error(
                "Rate limit on %s but fallback is the same model. Returning error.",
                selected_model,
            )
            return _rate_limit_error_response(selected_model), selected_model, analysis_info

        logger.warning(
            "⚡ Rate limit on %s — falling back to %s",
            selected_model, fallback_model,
        )
        fallback_provider = detect_provider(fallback_model)

        try:
            # Call fallback without retries — one shot only.
            # Pass _retry_count >= MAX_RETRIES so it raises immediately on 429.
            if fallback_provider == "google":
                response_data = await _call_gemini(
                    fallback_model, request, fallback_provider,
                    _retry_count=99,  # Skip retries — one shot only
                )
            else:
                response_data = await _call_litellm(
                    fallback_model, request, fallback_provider,
                )
            analysis_info = {
                **analysis_info,
                "fallback_from": selected_model,
                "selected_model": fallback_model,
                "strategy": analysis_info.get("strategy", "smart-routing") + "+fallback",
            }
            return response_data, fallback_model, analysis_info
        except RateLimitExhausted:
            # Both models are rate-limited
            logger.error(
                "Both %s and %s are rate-limited. Returning error.",
                selected_model, fallback_model,
            )
            return _rate_limit_error_response(selected_model), selected_model, analysis_info


def _rate_limit_error_response(model: str) -> Dict[str, Any]:
    """Build a graceful response when all models are rate-limited."""
    return {
        "content": (
            "⚠️ All configured models are currently rate-limited. "
            "Please wait a minute and try again, or consider upgrading your API plan. "
            "Check limits at https://ai.google.dev/gemini-api/docs/rate-limits"
        ),
        "finish_reason": "stop",
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


# ---------------------------------------------------------------------------
# /v1/chat/completions — full completion with routing
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    current_user: UserSession = Depends(validate_local_auth),
):
    # --- Rate limiting (per user) ---
    retry_after = _rate_limiter.check(current_user.id)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )

    # --- Input size validation ---
    total_content_len = sum(len(m.text_content()) for m in request.messages)
    if total_content_len > _MAX_CONTENT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Request content too large ({total_content_len:,} chars). "
                   f"Maximum is {_MAX_CONTENT_LENGTH:,} chars.",
        )

    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        # Extract prompt for logging
        user_msgs = [m.text_content() for m in request.messages if m.role == "user"]
        prompt_text = user_msgs[-1] if user_msgs else ""

        # Extract request metadata for enhanced logging
        req_meta = _extract_request_metadata(request)

        from nadirclaw.routing import (
            apply_routing_modifiers,
            get_session_cache,
            resolve_alias,
            resolve_profile,
        )

        # --- Check routing profiles (auto/eco/premium/free/reasoning) ---
        profile = resolve_profile(request.model)

        if profile == "eco":
            selected_model = settings.SIMPLE_MODEL
            analysis_info = {
                "strategy": "profile:eco",
                "selected_model": selected_model,
                "tier": "simple",
                "confidence": 1.0,
                "complexity_score": 0,
            }
        elif profile == "premium":
            selected_model = settings.COMPLEX_MODEL
            analysis_info = {
                "strategy": "profile:premium",
                "selected_model": selected_model,
                "tier": "complex",
                "confidence": 1.0,
                "complexity_score": 0,
            }
        elif profile == "free":
            selected_model = settings.FREE_MODEL
            analysis_info = {
                "strategy": "profile:free",
                "selected_model": selected_model,
                "tier": "free",
                "confidence": 1.0,
                "complexity_score": 0,
            }
        elif profile == "reasoning":
            selected_model = settings.REASONING_MODEL
            analysis_info = {
                "strategy": "profile:reasoning",
                "selected_model": selected_model,
                "tier": "reasoning",
                "confidence": 1.0,
                "complexity_score": 0,
            }
        elif request.model and request.model != "auto" and profile is None:
            # --- Check model aliases ---
            resolved = resolve_alias(request.model)
            if resolved:
                selected_model = resolved
                analysis_info = {
                    "strategy": "alias",
                    "selected_model": selected_model,
                    "alias_from": request.model,
                    "tier": "direct",
                    "confidence": 1.0,
                    "complexity_score": 0,
                }
            else:
                selected_model = request.model
                analysis_info = {
                    "strategy": "direct",
                    "selected_model": selected_model,
                    "tier": "direct",
                    "confidence": 1.0,
                    "complexity_score": 0,
                }
        else:
            # --- Smart routing (auto or no model specified) ---
            # Check session cache first
            session_cache = get_session_cache()
            cached = session_cache.get(request.messages)
            if cached:
                cached_model, cached_tier = cached
                selected_model = cached_model
                analysis_info = {
                    "strategy": "session-cache",
                    "selected_model": selected_model,
                    "tier": cached_tier,
                    "confidence": 1.0,
                    "complexity_score": 0,
                }
                logger.debug("Session cache hit: model=%s tier=%s", cached_model, cached_tier)
            else:
                selected_model, analysis_info = await _smart_route_full(
                    request.messages, current_user
                )

                # Apply routing modifiers (agentic, reasoning, context window)
                selected_model, final_tier, routing_info = apply_routing_modifiers(
                    base_model=selected_model,
                    base_tier=analysis_info.get("tier", "simple"),
                    request_meta=req_meta,
                    messages=request.messages,
                    simple_model=settings.SIMPLE_MODEL,
                    complex_model=settings.COMPLEX_MODEL,
                    reasoning_model=settings.REASONING_MODEL,
                    free_model=settings.FREE_MODEL,
                )
                analysis_info["tier"] = final_tier
                analysis_info["selected_model"] = selected_model
                analysis_info["routing_modifiers"] = routing_info

                # Cache this decision for session persistence
                session_cache.put(request.messages, selected_model, final_tier)

        # Resolve provider credential
        from nadirclaw.credentials import detect_provider, get_credential

        provider = detect_provider(selected_model)

        # ------------------------------------------------------------------
        # Call model — with automatic fallback on rate limit
        # ------------------------------------------------------------------
        from nadirclaw.telemetry import record_llm_call, trace_span

        with trace_span("chat_completion", {"nadirclaw.tier": analysis_info.get("tier")}) as span:
            response_data, selected_model, analysis_info = await _call_with_fallback(
                selected_model, request, provider, analysis_info,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)
            total_tokens = response_data["prompt_tokens"] + response_data["completion_tokens"]

            record_llm_call(
                span,
                model=selected_model,
                provider=provider,
                prompt_tokens=response_data["prompt_tokens"],
                completion_tokens=response_data["completion_tokens"],
                tier=analysis_info.get("tier"),
                latency_ms=elapsed_ms,
            )

        log_entry = {
            "type": "completion",
            "request_id": request_id,
            "prompt": prompt_text,
            "selected_model": selected_model,
            "provider": provider,
            "tier": analysis_info.get("tier"),
            "confidence": analysis_info.get("confidence"),
            "complexity_score": analysis_info.get("complexity_score"),
            "classifier_latency_ms": analysis_info.get("classifier_latency_ms"),
            "total_latency_ms": elapsed_ms,
            "prompt_tokens": response_data["prompt_tokens"],
            "completion_tokens": response_data["completion_tokens"],
            "total_tokens": total_tokens,
            "response_preview": (response_data["content"] or "")[:100],
            "fallback_used": analysis_info.get("fallback_from"),
            "status": "ok",
            **req_meta,
        }

        if settings.LOG_RAW:
            log_entry["raw_messages"] = [
                {"role": m.role, "content": m.text_content()} for m in request.messages
            ]
            log_entry["raw_response"] = response_data.get("content", "")

        _log_request(log_entry)

        # ------------------------------------------------------------------
        # Streaming response (SSE) — for OpenClaw / streaming clients
        # ------------------------------------------------------------------
        if request.stream:
            return _build_streaming_response(
                request_id, selected_model, response_data, analysis_info, elapsed_ms,
            )

        # ------------------------------------------------------------------
        # Non-streaming response (regular JSON)
        # ------------------------------------------------------------------
        message: dict[str, Any] = {
            "role": "assistant",
            "content": response_data["content"],
        }
        if "tool_calls" in response_data:
            message["tool_calls"] = response_data["tool_calls"]

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": selected_model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": response_data["finish_reason"],
                }
            ],
            "usage": {
                "prompt_tokens": response_data["prompt_tokens"],
                "completion_tokens": response_data["completion_tokens"],
                "total_tokens": response_data["prompt_tokens"] + response_data["completion_tokens"],
            },
            "nadirclaw_metadata": {
                "request_id": request_id,
                "response_time_ms": elapsed_ms,
                "routing": analysis_info,
            },
        }

    except HTTPException:
        raise  # Re-raise FastAPI HTTP exceptions as-is
    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error("Completion error: %s", e, exc_info=True)
        _log_request({
            "type": "completion",
            "request_id": request_id,
            "status": "error",
            "error": str(e),
            "total_latency_ms": elapsed_ms,
        })
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred. Request ID: {request_id}",
        )


def _build_streaming_response(
    request_id: str,
    model: str,
    response_data: Dict[str, Any],
    analysis_info: Dict[str, Any],
    elapsed_ms: int,
) -> EventSourceResponse:
    """Wrap a completed response as an OpenAI-compatible SSE stream.

    Sends the full content as a single chunk, then a finish chunk, then [DONE].
    This is a "fake" stream that converts a batch response into SSE format
    so streaming-only clients (like OpenClaw) can consume it.
    """

    async def event_generator():
        created = int(time.time())
        content = response_data.get("content", "") or ""
        tool_calls = response_data.get("tool_calls")

        # Chunk 1: the content (and tool_calls if present)
        # When tool_calls are present, content must be null per OpenAI protocol.
        delta: dict[str, Any] = {"role": "assistant"}
        if tool_calls:
            delta["tool_calls"] = tool_calls
            delta["content"] = None
        else:
            delta["content"] = content
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": None,
                }
            ],
        }
        yield {"data": json.dumps(chunk)}

        # Chunk 2: finish reason + usage
        finish_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": response_data.get("finish_reason", "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": response_data.get("prompt_tokens", 0),
                "completion_tokens": response_data.get("completion_tokens", 0),
                "total_tokens": response_data.get("prompt_tokens", 0) + response_data.get("completion_tokens", 0),
            },
        }
        yield {"data": json.dumps(finish_chunk)}

        # Final: [DONE] sentinel
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# /v1/logs — view request logs
# ---------------------------------------------------------------------------

@app.get("/v1/logs")
async def view_logs(
    limit: int = 20,
    current_user: UserSession = Depends(validate_local_auth),
) -> Dict[str, Any]:
    """View recent request logs."""
    request_log = settings.LOG_DIR / "requests.jsonl"
    if not request_log.exists():
        return {"logs": [], "total": 0}

    lines = request_log.read_text().strip().split("\n")
    recent = lines[-limit:] if len(lines) > limit else lines
    logs = []
    for line in reversed(recent):
        try:
            logs.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    return {"logs": logs, "total": len(lines), "showing": len(logs)}


# ---------------------------------------------------------------------------
# /v1/models & /health
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models(
    current_user: UserSession = Depends(validate_local_auth),
) -> Dict[str, Any]:
    models = settings.tier_models
    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "created": int(time.time()),
                "owned_by": m.split("/")[0] if "/" in m else "api",
            }
            for m in models
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": __version__,
        "simple_model": settings.SIMPLE_MODEL,
        "complex_model": settings.COMPLEX_MODEL,
    }


@app.get("/")
async def root():
    return {
        "name": "NadirClaw",
        "version": __version__,
        "description": "Open-source LLM router",
        "status": "ok",
    }
