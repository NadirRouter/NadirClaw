"""
NadirClaw — Lightweight LLM router server.

Routes simple prompts to cheap/local models and complex prompts to premium models.
OpenAI-compatible API at /v1/chat/completions.
"""

import asyncio
import collections
import json
import logging
import re
import os
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator
from sse_starlette.sse import EventSourceResponse

import os

from nadirclaw import __version__
from nadirclaw.auth import UserSession, validate_local_auth
from nadirclaw.settings import settings

logger = logging.getLogger("nadirclaw")


def _fallback_reason(model: str, error: Exception) -> Dict[str, str]:
    """Build a compact, log-safe fallback failure reason."""
    return {
        "model": model,
        "error_type": type(error).__name__,
        "message": str(error)[:200].replace("\n", " "),
    }


def _record_provider_success(model: str) -> None:
    if settings.PROVIDER_HEALTH is not True:
        return
    provider_health_tracker = _provider_health_tracker()
    provider_health_tracker.record_success(model)


def _record_provider_failure(model: str, error: Exception) -> None:
    if settings.PROVIDER_HEALTH is not True:
        return
    provider_health_tracker = _provider_health_tracker()
    reason = _fallback_reason(model, error)
    provider_health_tracker.record_failure(model, reason["error_type"], reason["message"])


def _order_fallback_candidates(chain: list[str]) -> list[str]:
    if settings.PROVIDER_HEALTH is not True:
        return chain
    provider_health_tracker = _provider_health_tracker()
    return provider_health_tracker.ordered_candidates(chain)


def _provider_health_tracker():
    from nadirclaw.provider_health import provider_health_tracker
    failure_threshold = settings.PROVIDER_HEALTH_FAILURE_THRESHOLD
    cooldown_seconds = settings.PROVIDER_HEALTH_COOLDOWN_SECONDS
    provider_health_tracker.failure_threshold = failure_threshold if isinstance(failure_threshold, int) else 2
    provider_health_tracker.cooldown_seconds = cooldown_seconds if isinstance(cooldown_seconds, int) else 60
    return provider_health_tracker


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

# Register web dashboard routes
from nadirclaw.web_dashboard import router as dashboard_router
app.include_router(dashboard_router)

_ROUTING_HEADERS = ("X-Routed-Model", "X-Routed-Tier", "X-Complexity-Score")

# ---------------------------------------------------------------------------
# CORS — restrict to explicit origins (never wildcard + credentials)
# Reads via `settings.CORS_ORIGINS` so CLI / env overrides applied after this
# module is imported still take effect on subsequent reads (matches the
# Settings pattern used elsewhere in the codebase).
# ---------------------------------------------------------------------------
_cors_origins = settings.CORS_ORIGINS
if _cors_origins:
    _cors_origin_regex = None
    _cors_credentials = True
else:
    # Local-only default: match any port on localhost/127.0.0.1
    # Starlette does exact string matching on allow_origins, so we use
    # allow_origin_regex to support arbitrary ports during development.
    _cors_origin_regex = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
    _cors_credentials = False  # No credentials unless origins are explicitly configured

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=_cors_origin_regex,
    allow_credentials=_cors_credentials,
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    expose_headers=list(_ROUTING_HEADERS),
)


# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject security headers on every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Request-ID"] = str(uuid.uuid4())
        # Prevent caching of API responses
        if request.url.path.startswith("/v1/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        if settings.HSTS:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        return response


app.add_middleware(SecurityHeadersMiddleware)


# ---------------------------------------------------------------------------
# Validation error handler — sanitized responses (no Pydantic internals)
# ---------------------------------------------------------------------------

from fastapi.exceptions import RequestValidationError


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    # Log full details server-side only
    logger.error(
        "Validation error on %s %s: %s\nBody (truncated): %s",
        request.method,
        request.url.path,
        exc.errors(),
        body[:500].decode("utf-8", errors="replace"),
    )
    # Return sanitized error to client — no Pydantic internals
    safe_errors = []
    for err in exc.errors():
        loc = err.get("loc", [])
        field = ".".join(str(part) for part in loc if part != "body")
        safe_errors.append({
            "field": field or "request",
            "message": f"Invalid value for '{field}'" if field else "Invalid request body",
        })
    return JSONResponse(status_code=422, content={"detail": safe_errors})


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
    n: Optional[int] = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ChatCompletionRequest":
        """Enforce safe bounds on all numeric parameters."""
        if len(self.messages) > 500:
            raise ValueError("messages: max 500 messages allowed")
        if self.temperature is not None and not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature: must be between 0.0 and 2.0")
        if self.max_tokens is not None and (self.max_tokens < 1 or self.max_tokens > 100_000):
            raise ValueError("max_tokens: must be between 1 and 100,000")
        if self.top_p is not None and not (0.0 <= self.top_p <= 1.0):
            raise ValueError("top_p: must be between 0.0 and 1.0")
        if self.n is not None and (self.n < 1 or self.n > 8):
            raise ValueError("n: must be between 1 and 8")
        return self


class ClassifyRequest(BaseModel):
    prompt: str
    system_message: Optional[str] = ""


class ClassifyBatchRequest(BaseModel):
    prompts: List[str]


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

# Compiled once: redacts common API-key shapes from logged system prompts.
_API_KEY_PATTERN = re.compile(
    r"(sk-[a-zA-Z0-9_-]{10,}|AIza[a-zA-Z0-9_-]{20,}|ghp_[a-zA-Z0-9]{20,}"
    r"|gho_[a-zA-Z0-9]{20,}|xox[bpars]-[a-zA-Z0-9-]{10,})"
)

_log_lock = Lock()
_log_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nadirclaw-log")


def _log_request_sync(entry: Dict[str, Any]) -> None:
    """Synchronous log writer — runs in thread pool to avoid blocking the event loop."""
    log_dir = settings.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    request_log = log_dir / "requests.jsonl"

    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(entry, default=str) + "\n"
    with _log_lock:
        with open(request_log, "a") as f:
            f.write(line)


def _on_log_done(fut: Future) -> None:
    """Surface exceptions from the async log thread — otherwise they're swallowed."""
    exc = fut.exception()
    if exc is not None:
        logger.error("async request log write failed", exc_info=exc)


def _log_request(entry: Dict[str, Any]) -> None:
    """Non-blocking log request — submits all blocking I/O to thread pool."""
    def _do_log():
        _log_request_sync(entry)
        # SQLite logging (also blocking I/O with threading.Lock)
        from nadirclaw.request_logger import log_request as sqlite_log
        sqlite_log(entry)
        # Prometheus metrics (CPU-only, fast)
        from nadirclaw.metrics import record_request
        record_request(entry)

    fut = _log_executor.submit(_do_log)
    fut.add_done_callback(_on_log_done)

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

    if settings.LOG_SYSTEM_PROMPTS and system_text:
        sanitized_system_text = _API_KEY_PATTERN.sub("[REDACTED_KEY]", system_text)[:500]
    else:
        sanitized_system_text = ""

    from nadirclaw.routing import detect_images
    image_info = detect_images(messages)

    return {
        "stream": bool(request.stream),
        "message_count": len(messages),
        "has_system_prompt": has_system,
        "system_prompt_length": system_len,
        "system_prompt_text": sanitized_system_text,
        "has_tools": tool_count > 0,
        "tool_count": tool_count,
        "requested_model": request.model,
        "has_images": image_info["has_images"],
        "image_count": image_info["image_count"],
    }


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    log_dir = settings.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    request_log = log_dir / "requests.jsonl"

    # Log maintenance (rotation + pruning) — fast no-op if nothing to do
    from nadirclaw.log_maintenance import run_maintenance

    run_maintenance(
        log_dir,
        max_size_mb=settings.LOG_MAX_SIZE_MB,
        retention_days=settings.LOG_RETENTION_DAYS,
        compress=settings.LOG_COMPRESS,
    )

    logger.info("=" * 60)
    logger.info("NadirClaw starting...")
    logger.info("Log file: %s", request_log.resolve())
    logger.info("=" * 60)

    # Optional OpenTelemetry
    from nadirclaw.telemetry import instrument_fastapi, setup_telemetry

    if setup_telemetry("nadirclaw"):
        instrument_fastapi(app)

    # Classifier is lazy-loaded on first request (cuts cold-start time).
    # Pre-warm in background thread so first request is fast.
    import threading

    def _background_warmup():
        try:
            from nadirclaw.classifier import warmup
            warmup()
            logger.info("Binary classifier warmed up (background)")
        except Exception as e:
            logger.warning("Background warmup failed (will retry on first request): %s", e)

    threading.Thread(target=_background_warmup, daemon=True, name="classifier-warmup").start()

    # Show config
    try:
        import litellm
        litellm.set_verbose = False
        logger.info("Simple model:  %s", settings.SIMPLE_MODEL)
        if settings.has_mid_tier:
            logger.info("Mid model:     %s", settings.MID_MODEL)
        logger.info("Complex model: %s", settings.COMPLEX_MODEL)
        if settings.has_explicit_tiers:
            logger.info("Tier config:   explicit (env vars)")
        elif settings.has_mid_tier:
            thresholds = settings.TIER_THRESHOLDS
            logger.info("Tier config:   3-tier (thresholds: %.2f / %.2f)", thresholds[0], thresholds[1])
        else:
            logger.info("Tier config:   derived from NADIRCLAW_MODELS")
        if settings.OPTIMIZE != "off":
            logger.info("Optimize:      %s", settings.OPTIMIZE)
        logger.info("Ollama base:   %s", settings.OLLAMA_API_BASE)
        if settings.API_BASE:
            logger.info("API base:      %s", settings.API_BASE)
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
    logger.info("")
    logger.info("Want a hosted dashboard, trained classifier, and team billing?")
    logger.info("Try Nadir Pro free: https://getnadir.com?ref=cli-serve")
    logger.info("=" * 60)


@app.on_event("shutdown")
def _shutdown_log_executor():
    """Drain pending log writes on SIGTERM / uvicorn shutdown."""
    _log_executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Smart routing internals
# ---------------------------------------------------------------------------

async def _smart_route_analysis(
    prompt: str, system_message: str, user: UserSession
) -> tuple:
    """Run classifier, return (selected_model, analysis_dict). No LLM call."""
    from nadirclaw.classifier import get_classifier
    from nadirclaw.telemetry import trace_span

    with trace_span("smart_route_analysis") as span:
        analyzer = get_classifier()
        result = await analyzer.analyze(text=prompt, system_message=system_message)

        tier_name = result.get("tier_name", "simple")
        if tier_name == "complex":
            selected = settings.COMPLEX_MODEL
        elif tier_name == "mid":
            selected = settings.MID_MODEL
        else:
            selected = settings.SIMPLE_MODEL

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


def _is_oauth_token(token: str) -> bool:
    """Detect if a credential is an OAuth access token vs an API key.

    Google API keys start with 'AIza'. OAuth access tokens typically start
    with 'ya29.' or are JWTs. OpenClaw OAuth tokens may vary but are never
    in AIza format.
    """
    if token.startswith("AIza"):
        return False
    # OAuth access tokens from Google (ya29.*) or other JWT-like tokens
    if token.startswith("ya29.") or token.startswith("eyJ"):
        return True
    # If it's from OpenClaw's auth-profiles, it's OAuth — check via credential source
    from nadirclaw.credentials import get_credential_source
    source = get_credential_source("google")
    return source in ("openclaw", "oauth")


# Default GCP location for Vertex AI when using OAuth tokens.
_VERTEX_DEFAULT_LOCATION = "us-central1"


def _get_gemini_client(api_key: str):
    """Get or create a thread-safe, per-key google-genai Client.

    Handles both API keys (AIza...) and OAuth access tokens (ya29...).
    The google-genai SDK requires either:
      - api_key for the Google AI API, or
      - vertexai=True + credentials + project + location for Vertex AI API.
    OAuth tokens (from OpenClaw/Gemini CLI) must use the Vertex AI path.
    """
    with _gemini_client_lock:
        if api_key not in _gemini_clients:
            from google import genai

            if _is_oauth_token(api_key):
                from google.oauth2.credentials import Credentials
                from nadirclaw.credentials import get_gemini_oauth_config

                oauth_config = get_gemini_oauth_config()
                project_id = (oauth_config or {}).get("project_id") or os.environ.get(
                    "GOOGLE_CLOUD_PROJECT", ""
                )
                if not project_id:
                    logger.warning(
                        "Gemini OAuth token detected but no project_id found. "
                        "Set GOOGLE_CLOUD_PROJECT env var or ensure your "
                        "credentials include a project_id."
                    )
                creds = Credentials(token=api_key)
                _gemini_clients[api_key] = genai.Client(
                    vertexai=True,
                    credentials=creds,
                    project=project_id,
                    location=os.environ.get("GOOGLE_CLOUD_LOCATION", _VERTEX_DEFAULT_LOCATION),
                )
                logger.debug(
                    "Created Gemini client with OAuth credentials (Vertex AI, project=%s)",
                    project_id,
                )
            else:
                _gemini_clients[api_key] = genai.Client(api_key=api_key)
                logger.debug("Created Gemini client with API key")
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

    # Forward thinking config for Gemini thinking models
    req_extra = request.model_extra or {}
    thinking_param = req_extra.get("thinking")
    if thinking_param and isinstance(thinking_param, dict):
        budget = thinking_param.get("budget_tokens")
        if budget:
            gen_config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=budget,
            )

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
        # 400/401/403 — likely auth issue. Surface credential source for debugging.
        if e.code in (400, 401, 403):
            from nadirclaw.credentials import get_credential_source
            cred_source = get_credential_source(provider or "google") or "unknown"
            is_oauth = _is_oauth_token(api_key)
            logger.error(
                "Gemini auth error (%s) for model=%s: %s "
                "[credential_source=%s, is_oauth=%s, token_prefix=%s]",
                e.code, native_model, e,
                cred_source, is_oauth, api_key[:8] + "...",
            )
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

        # Extract text from parts (handle function_call and thought parts)
        thinking_parts = []
        if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
            text_parts = []
            for part in candidate.content.parts:
                if hasattr(part, "thought") and part.thought:
                    # Gemini thinking model thought parts
                    if hasattr(part, "text") and part.text:
                        thinking_parts.append(part.text)
                elif hasattr(part, "text") and part.text:
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

    result = {
        "content": content,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if thinking_parts:
        result["thinking"] = "".join(thinking_parts)
    # Capture thinking token count from Gemini usage metadata
    if usage:
        thoughts_tok = getattr(usage, "thoughts_token_count", None)
        if thoughts_tok:
            result["reasoning_tokens"] = thoughts_tok
    return result


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
        # Preserve multimodal content arrays (image_url parts) as-is.
        if isinstance(message.content, list):
            content = message.content
        else:
            text = message.text_content()
            content = text if text else message.content
        msg: dict[str, Any] = {"role": message.role, "content": content}
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

    # Pass through tool definitions, tool_choice, and thinking/reasoning params
    extra = request.model_extra or {}
    if extra.get("tools"):
        call_kwargs["tools"] = extra["tools"]
    if extra.get("tool_choice"):
        call_kwargs["tool_choice"] = extra["tool_choice"]
    if extra.get("reasoning_effort"):
        call_kwargs["reasoning_effort"] = extra["reasoning_effort"]
    if extra.get("thinking"):
        call_kwargs["thinking"] = extra["thinking"]
    if extra.get("response_format"):
        call_kwargs["response_format"] = extra["response_format"]

    if cred_provider and cred_provider != "ollama":
        api_key = get_credential(cred_provider)
        if api_key:
            # Anthropic OAuth/setup-tokens (sk-ant-oat*) require Bearer auth
            # and the oauth-2025-04-20 beta header. Bypass LiteLLM and call
            # the Anthropic API directly since LiteLLM uses x-api-key.
            if cred_provider == "anthropic" and "sk-ant-oat" in api_key:
                import httpx
                model_id = litellm_model.removeprefix("anthropic/")
                anthropic_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in call_kwargs.get("messages", [])
                    if m.get("content") is not None
                ]
                anthropic_body = {
                    "model": model_id,
                    "messages": anthropic_messages,
                    "max_tokens": call_kwargs.get("max_tokens", 1024),
                }
                if call_kwargs.get("temperature") is not None:
                    anthropic_body["temperature"] = call_kwargs["temperature"]
                req_extra = request.model_extra or {}
                if req_extra.get("tools"):
                    anthropic_body["tools"] = req_extra["tools"]
                if req_extra.get("tool_choice"):
                    anthropic_body["tool_choice"] = req_extra["tool_choice"]
                if req_extra.get("thinking"):
                    anthropic_body["thinking"] = req_extra["thinking"]
                async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "anthropic-version": "2023-06-01",
                            "anthropic-beta": "oauth-2025-04-20,claude-code-20250219",
                            "content-type": "application/json",
                        },
                        json=anthropic_body,
                    )
                if resp.status_code != 200:
                    error_detail = resp.text
                    logger.error("Anthropic OAuth call failed (%s): %s", resp.status_code, error_detail)
                    from litellm.exceptions import AuthenticationError as LiteLLMAuthError
                    raise LiteLLMAuthError(
                        message=f"Anthropic OAuth error: {error_detail}",
                        model=model,
                        llm_provider="anthropic",
                    )
                data = resp.json()
                content_text = ""
                thinking_content = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        content_text += block["text"]
                    elif block.get("type") == "thinking":
                        thinking_content += block.get("thinking", "")
                prompt_tok = data.get("usage", {}).get("input_tokens", 0)
                compl_tok = data.get("usage", {}).get("output_tokens", 0)
                result = {
                    "id": data.get("id", ""),
                    "object": "chat.completion",
                    "created": 0,
                    "model": data.get("model", model_id),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content_text},
                        "finish_reason": data.get("stop_reason", "stop"),
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tok,
                        "completion_tokens": compl_tok,
                        "total_tokens": prompt_tok + compl_tok,
                    },
                    "prompt_tokens": prompt_tok,
                    "completion_tokens": compl_tok,
                    "content": content_text,
                    "finish_reason": data.get("stop_reason", "stop"),
                }
                if thinking_content:
                    result["thinking"] = thinking_content
                return result
            else:
                call_kwargs["api_key"] = api_key

    # Pass api_base for Ollama or custom OpenAI-compatible endpoints
    if litellm_model.startswith("ollama/") or litellm_model.startswith("ollama_chat/"):
        call_kwargs["api_base"] = settings.OLLAMA_API_BASE
    elif settings.API_BASE and "api_base" not in call_kwargs:
        call_kwargs["api_base"] = settings.API_BASE

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

    # Preserve thinking/reasoning content from LLM response
    # DeepSeek and some providers use reasoning_content
    reasoning_content = getattr(msg, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content:
        result["reasoning_content"] = reasoning_content
    # Anthropic extended thinking (via LiteLLM)
    thinking = getattr(msg, "thinking", None)
    if isinstance(thinking, str) and thinking:
        result["thinking"] = thinking

    # Capture reasoning token counts from usage details
    if response.usage:
        ctd = getattr(response.usage, "completion_tokens_details", None)
        if ctd and not callable(ctd):
            reasoning_tokens = getattr(ctd, "reasoning_tokens", None)
            if isinstance(reasoning_tokens, int) and reasoning_tokens:
                result["reasoning_tokens"] = reasoning_tokens

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
    from nadirclaw.rate_limit import get_model_rate_limiter
    from nadirclaw.telemetry import trace_span

    # Check per-model rate limit before making the call
    limiter = get_model_rate_limiter()
    retry_after = limiter.check(model)
    if retry_after is not None:
        logger.warning(
            "Per-model rate limit hit for %s (retry in %ds)", model, retry_after,
        )
        raise RateLimitExhausted(model=model, retry_after=retry_after)

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
    """Try the selected model; on failure, cascade through the fallback chain.

    The fallback chain is configured via NADIRCLAW_FALLBACK_CHAIN env var.
    Each model in the chain is tried once (no retries) after the primary fails.
    Handles 429 rate limits, 5xx errors, and timeouts.

    Returns (response_data, actual_model_used, updated_analysis_info).
    """
    from nadirclaw.credentials import detect_provider

    try:
        response_data = await _dispatch_model(selected_model, request, provider)
        _record_provider_success(selected_model)
        return response_data, selected_model, analysis_info
    except (RateLimitExhausted, Exception) as primary_error:
        if isinstance(primary_error, HTTPException):
            raise  # Don't fallback on validation/auth errors
        _record_provider_failure(selected_model, primary_error)

        # Build fallback chain: use per-tier chain if configured, else global
        tier = analysis_info.get("tier", "")
        full_chain = settings.get_tier_fallback_chain(tier) if tier else settings.FALLBACK_CHAIN
        chain = _order_fallback_candidates([m for m in full_chain if m != selected_model])

        if not chain:
            if isinstance(primary_error, RateLimitExhausted):
                return _rate_limit_error_response(selected_model), selected_model, analysis_info
            raise primary_error

        failed_models = [selected_model]
        analysis_info.setdefault("fallback_reasons", []).append(
            _fallback_reason(selected_model, primary_error)
        )
        last_error = primary_error

        for fallback_model in chain:
            logger.warning(
                "⚡ %s failed (%s) — trying fallback %s (%d/%d in chain)",
                selected_model if len(failed_models) == 1 else failed_models[-1],
                type(last_error).__name__,
                fallback_model,
                len(failed_models),
                len(chain),
            )
            fallback_provider = detect_provider(fallback_model)

            try:
                response_data = await _dispatch_model(
                    fallback_model, request, fallback_provider,
                )
                _record_provider_success(fallback_model)
                analysis_info = {
                    **analysis_info,
                    "fallback_from": selected_model,
                    "fallback_chain_tried": failed_models,
                    "selected_model": fallback_model,
                    "strategy": analysis_info.get("strategy", "smart-routing") + "+fallback",
                }
                return response_data, fallback_model, analysis_info
            except (RateLimitExhausted, Exception) as chain_error:
                if isinstance(chain_error, HTTPException):
                    raise
                _record_provider_failure(fallback_model, chain_error)
                failed_models.append(fallback_model)
                analysis_info.setdefault("fallback_reasons", []).append(
                    _fallback_reason(fallback_model, chain_error)
                )
                last_error = chain_error
                continue

        # All models in chain exhausted
        logger.error(
            "All models in fallback chain exhausted: %s",
            failed_models,
        )
        if isinstance(last_error, RateLimitExhausted):
            return _rate_limit_error_response(selected_model), selected_model, analysis_info
        raise last_error


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

def _routing_headers(model: str, analysis_info: Dict[str, Any]) -> Dict[str, str]:
    """Build X-Routed-* headers from routing analysis."""
    return {
        "X-Routed-Model": model,
        "X-Routed-Tier": str(analysis_info.get("tier", "")),
        "X-Complexity-Score": str(analysis_info.get("complexity_score", "")),
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    raw_request: Request,
    request: ChatCompletionRequest,
    response: Response,
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
            # Always classify the current message, then apply
            # upgrade-only session caching (never downgrade mid-session).
            session_cache = get_session_cache()

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

            # Upgrade-only cache: escalate if new tier is higher,
            # keep cached tier if it's already equal or above.
            selected_model, final_tier, cache_status = session_cache.upgrade_if_higher(
                request.messages, selected_model, final_tier
            )

            analysis_info["tier"] = final_tier
            analysis_info["selected_model"] = selected_model
            analysis_info["routing_modifiers"] = routing_info
            analysis_info["cache_status"] = cache_status
            if cache_status == "kept":
                analysis_info["strategy"] = (
                    analysis_info.get("strategy", "smart-routing") + "+session-cache"
                )

        # ------------------------------------------------------------------
        # Context optimization — compact messages before dispatch
        # ------------------------------------------------------------------
        optimize_mode = (request.model_extra or {}).get("optimize") or settings.OPTIMIZE
        optimization_info = None
        if optimize_mode != "off":
            from nadirclaw.optimize import optimize_messages

            raw_msgs = [
                {"role": m.role, "content": m.text_content()}
                for m in request.messages
            ]
            opt_result = optimize_messages(
                raw_msgs,
                mode=optimize_mode,
                max_turns=settings.OPTIMIZE_MAX_TURNS,
            )
            if opt_result.tokens_saved > 0:
                optimized_msgs = [
                    ChatMessage(role=m["role"], content=m["content"])
                    for m in opt_result.messages
                ]
                request = request.model_copy(update={"messages": optimized_msgs})
            optimization_info = {
                "optimization_mode": opt_result.mode,
                "original_tokens": opt_result.original_tokens,
                "optimized_tokens": opt_result.optimized_tokens,
                "tokens_saved": opt_result.tokens_saved,
                "optimizations_applied": opt_result.optimizations_applied,
            }

        # ------------------------------------------------------------------
        # Context compression — dedup + truncate old turns
        # Runs AFTER optimization, BEFORE dispatch
        # ------------------------------------------------------------------
        compression_info = None
        if settings.CONTEXT_COMPRESSION and len(request.messages) > settings.COMPRESS_MIN_MESSAGES:
            from nadirclaw.compress import compress_messages

            msg_dicts = []
            for m in request.messages:
                d: Dict[str, Any] = {"role": m.role, "content": m.content}
                extra = m.model_extra or {}
                if "tool_calls" in extra:
                    d["tool_calls"] = extra["tool_calls"]
                if "tool_call_id" in extra:
                    d["tool_call_id"] = extra["tool_call_id"]
                if "name" in extra:
                    d["name"] = extra["name"]
                msg_dicts.append(d)
            compressed_msgs, comp_stats = compress_messages(msg_dicts)
            if comp_stats.get("compressed"):
                rebuilt_msgs = []
                for d in compressed_msgs:
                    extras: Dict[str, Any] = {}
                    if "tool_calls" in d:
                        extras["tool_calls"] = d["tool_calls"]
                    if "tool_call_id" in d:
                        extras["tool_call_id"] = d["tool_call_id"]
                    if "name" in d:
                        extras["name"] = d["name"]
                    rebuilt_msgs.append(
                        ChatMessage(role=d["role"], content=d.get("content"), **extras)
                    )
                request = request.model_copy(update={"messages": rebuilt_msgs})
                compression_info = comp_stats
                logger.info(
                    "Context compressed: %d → %d messages (deduped=%d, truncated=%d, ratio=%.2f)",
                    comp_stats["messages_before"], comp_stats["messages_after"],
                    comp_stats["deduped"], comp_stats["truncated"],
                    comp_stats["compression_ratio"],
                )

        # Resolve provider credential
        from nadirclaw.credentials import detect_provider, get_credential

        provider = detect_provider(selected_model)

        # ------------------------------------------------------------------
        # Prompt cache — check before calling the model
        # ------------------------------------------------------------------
        from nadirclaw.cache import _cache_enabled, get_prompt_cache

        prompt_cache = get_prompt_cache()
        cache_hit = False
        if _cache_enabled() and not request.stream:
            cached_response = prompt_cache.get(selected_model, request.messages)
            if cached_response is not None:
                response_data = cached_response
                cache_hit = True

        # ------------------------------------------------------------------
        # TRUE STREAMING — bypass batch call, stream directly from provider
        # ------------------------------------------------------------------
        if request.stream and not cache_hit:
            from nadirclaw.budget import get_budget_tracker
            from nadirclaw.telemetry import trace_span

            _stream_analysis = dict(analysis_info)  # mutable copy for stream callbacks
            _stream_start = start_time
            _stream_req_meta = req_meta
            _stream_prompt = prompt_text

            async def _true_stream_wrapper():
                async for sse_event in _stream_with_fallback(
                    selected_model, request, provider, _stream_analysis, request_id,
                ):
                    yield sse_event

                # After stream completes, log the request
                stream_elapsed = int((time.time() - _stream_start) * 1000)
                stream_model = _stream_analysis.get("_stream_model", selected_model)
                stream_usage = _stream_analysis.get("_stream_usage", {"prompt_tokens": 0, "completion_tokens": 0})

                budget_status = get_budget_tracker().record(
                    stream_model,
                    stream_usage["prompt_tokens"],
                    stream_usage["completion_tokens"],
                )

                _log_request({
                    "type": "completion",
                    "request_id": request_id,
                    "prompt": _stream_prompt,
                    "selected_model": stream_model,
                    "provider": provider,  # approximate; fallback may change provider
                    "tier": _stream_analysis.get("tier"),
                    "confidence": _stream_analysis.get("confidence"),
                    "complexity_score": _stream_analysis.get("complexity_score"),
                    "classifier_latency_ms": _stream_analysis.get("classifier_latency_ms"),
                    "total_latency_ms": stream_elapsed,
                    "prompt_tokens": stream_usage["prompt_tokens"],
                    "completion_tokens": stream_usage["completion_tokens"],
                    "total_tokens": stream_usage["prompt_tokens"] + stream_usage["completion_tokens"],
                    "cost": budget_status["cost"],
                    "daily_spend": budget_status["daily_spend"],
                    "response_preview": "[streamed]",
                    "fallback_used": _stream_analysis.get("fallback_from"),
                    "fallback_reasons": _stream_analysis.get("fallback_reasons", []),
                    "streaming": True,
                    "status": "error" if _stream_analysis.get("_stream_error") else "ok",
                    **_stream_req_meta,
                    **(optimization_info or {}),
                })

            return EventSourceResponse(
                _true_stream_wrapper(),
                media_type="text/event-stream",
                headers=_routing_headers(selected_model, analysis_info),
            )

        # ------------------------------------------------------------------
        # Call model — with automatic fallback on rate limit
        # ------------------------------------------------------------------
        from nadirclaw.telemetry import record_llm_call, trace_span

        if not cache_hit:
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

            # Store in prompt cache
            if _cache_enabled():
                prompt_cache.put(selected_model, request.messages, response_data)
        else:
            elapsed_ms = int((time.time() - start_time) * 1000)
            total_tokens = response_data["prompt_tokens"] + response_data["completion_tokens"]
            analysis_info["strategy"] = analysis_info.get("strategy", "") + "+cache-hit"
            logger.info("Cache HIT — skipped LLM call (elapsed=%dms)", elapsed_ms)

        # --- Budget tracking ---
        from nadirclaw.budget import get_budget_tracker
        budget_status = get_budget_tracker().record(
            selected_model,
            response_data["prompt_tokens"],
            response_data["completion_tokens"],
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
            "cost": budget_status["cost"],
            "daily_spend": budget_status["daily_spend"],
            "response_preview": (response_data["content"] or "")[:100],
            "fallback_used": analysis_info.get("fallback_from"),
            "fallback_reasons": analysis_info.get("fallback_reasons", []),
            "status": "ok",
            **req_meta,
            **(optimization_info or {}),
        }

        if settings.LOG_RAW:
            log_entry["raw_messages"] = [
                {"role": m.role, "content": m.text_content()} for m in request.messages
            ]
            log_entry["raw_response"] = response_data.get("content", "")

        _log_request(log_entry)

        # ------------------------------------------------------------------
        # Streaming response (SSE) — cached stream uses fake wrapper
        # ------------------------------------------------------------------
        if request.stream:
            return _build_streaming_response(
                request_id, selected_model, response_data, analysis_info, elapsed_ms,
            )

        # ------------------------------------------------------------------
        # Non-streaming response (regular JSON)
        # ------------------------------------------------------------------
        for hdr_name, hdr_val in _routing_headers(selected_model, analysis_info).items():
            response.headers[hdr_name] = hdr_val

        message: dict[str, Any] = {
            "role": "assistant",
            "content": response_data["content"],
        }
        if "tool_calls" in response_data:
            message["tool_calls"] = response_data["tool_calls"]
        if "reasoning_content" in response_data:
            message["reasoning_content"] = response_data["reasoning_content"]
        if "thinking" in response_data:
            message["thinking"] = response_data["thinking"]

        usage: dict[str, Any] = {
            "prompt_tokens": response_data["prompt_tokens"],
            "completion_tokens": response_data["completion_tokens"],
            "total_tokens": response_data["prompt_tokens"] + response_data["completion_tokens"],
        }
        if response_data.get("reasoning_tokens"):
            usage["completion_tokens_details"] = {
                "reasoning_tokens": response_data["reasoning_tokens"],
            }

        response_body = {
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
            "usage": usage,
            "nadirclaw_metadata": {
                "request_id": request_id,
                "response_time_ms": elapsed_ms,
                "routing": analysis_info,
                **({"optimization": optimization_info} if optimization_info else {}),
            },
        }

        return JSONResponse(content=response_body)

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
        if response_data.get("reasoning_content"):
            delta["reasoning_content"] = response_data["reasoning_content"]
        if response_data.get("thinking"):
            delta["thinking"] = response_data["thinking"]
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

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=_routing_headers(model, analysis_info),
    )


# ---------------------------------------------------------------------------
# True streaming — real SSE from providers with mid-stream fallback
# ---------------------------------------------------------------------------

async def _stream_litellm(
    model: str,
    request: "ChatCompletionRequest",
    provider: str | None,
):
    """True streaming via LiteLLM. Yields (delta_dict, usage_dict|None, finish_reason|None) tuples.

    Raises on connection/rate-limit errors (before or during streaming).
    """
    import litellm

    from nadirclaw.credentials import get_credential

    if provider == "openai-codex":
        litellm_model = model.removeprefix("openai-codex/")
        cred_provider = "openai-codex"
    else:
        litellm_model = model
        cred_provider = provider

    req_extra = request.model_extra or {}
    if litellm_model.startswith("ollama/") and req_extra.get("tools"):
        litellm_model = "ollama_chat/" + litellm_model.removeprefix("ollama/")

    messages = []
    for message in request.messages:
        if isinstance(message.content, list):
            content = message.content
        else:
            text = message.text_content()
            content = text if text else message.content
        msg: dict[str, Any] = {"role": message.role, "content": content}
        extra_fields = message.model_extra or {}
        if "tool_calls" in extra_fields:
            msg["tool_calls"] = extra_fields["tool_calls"]
        if "tool_call_id" in extra_fields:
            msg["tool_call_id"] = extra_fields["tool_call_id"]
        if "name" in extra_fields:
            msg["name"] = extra_fields["name"]
        messages.append(msg)

    call_kwargs: Dict[str, Any] = {
        "model": litellm_model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if request.temperature is not None:
        call_kwargs["temperature"] = request.temperature
    if request.max_tokens is not None:
        call_kwargs["max_tokens"] = request.max_tokens
    if request.top_p is not None:
        call_kwargs["top_p"] = request.top_p

    extra = request.model_extra or {}
    if extra.get("tools"):
        call_kwargs["tools"] = extra["tools"]
    if extra.get("tool_choice"):
        call_kwargs["tool_choice"] = extra["tool_choice"]
    if extra.get("reasoning_effort"):
        call_kwargs["reasoning_effort"] = extra["reasoning_effort"]
    if extra.get("thinking"):
        call_kwargs["thinking"] = extra["thinking"]
    if extra.get("response_format"):
        call_kwargs["response_format"] = extra["response_format"]

    if cred_provider and cred_provider != "ollama":
        api_key = get_credential(cred_provider)
        if api_key:
            call_kwargs["api_key"] = api_key

    if litellm_model.startswith("ollama/") or litellm_model.startswith("ollama_chat/"):
        call_kwargs["api_base"] = settings.OLLAMA_API_BASE
    elif settings.API_BASE and "api_base" not in call_kwargs:
        call_kwargs["api_base"] = settings.API_BASE

    try:
        response = await litellm.acompletion(**call_kwargs)
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "rate" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
            raise RateLimitExhausted(model=model, retry_after=60)
        raise

    async for chunk in response:
        usage = None
        if hasattr(chunk, "usage") and chunk.usage:
            usage = {
                "prompt_tokens": chunk.usage.prompt_tokens or 0,
                "completion_tokens": chunk.usage.completion_tokens or 0,
            }

        choice = chunk.choices[0] if chunk.choices else None
        if choice is None:
            # Usage-only final chunk (no choices) -- yield usage without content
            if usage:
                yield {}, usage, None
            continue

        delta = choice.delta
        delta_dict: dict[str, Any] = {}
        if hasattr(delta, "role") and delta.role:
            delta_dict["role"] = delta.role
        if hasattr(delta, "content") and delta.content is not None:
            delta_dict["content"] = delta.content
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            delta_dict["tool_calls"] = [
                tc.model_dump() if hasattr(tc, "model_dump") else tc
                for tc in delta.tool_calls
            ]
        # Preserve reasoning/thinking content in streaming deltas
        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
            delta_dict["reasoning_content"] = delta.reasoning_content
        if hasattr(delta, "thinking") and delta.thinking is not None:
            delta_dict["thinking"] = delta.thinking

        yield delta_dict, usage, choice.finish_reason


async def _stream_gemini(
    model: str,
    request: "ChatCompletionRequest",
    provider: str | None,
):
    """True streaming via Gemini. Yields (delta_dict, usage_dict|None, finish_reason|None) tuples."""
    import re

    from google.genai import types
    from google.genai.errors import ClientError

    from nadirclaw.credentials import get_credential

    api_key = get_credential(provider)
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="No Google/Gemini API key configured.",
        )

    client = _get_gemini_client(api_key)
    native_model = _strip_gemini_prefix(model)

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

    gen_config_kwargs: Dict[str, Any] = {}
    if request.temperature is not None:
        gen_config_kwargs["temperature"] = request.temperature
    if request.max_tokens is not None:
        gen_config_kwargs["max_output_tokens"] = request.max_tokens
    if request.top_p is not None:
        gen_config_kwargs["top_p"] = request.top_p

    generate_kwargs: Dict[str, Any] = {"model": native_model, "contents": contents}
    if gen_config_kwargs:
        generate_kwargs["config"] = types.GenerateContentConfig(
            **gen_config_kwargs,
            system_instruction="\n".join(system_parts) if system_parts else None,
        )
    elif system_parts:
        generate_kwargs["config"] = types.GenerateContentConfig(
            system_instruction="\n".join(system_parts),
        )

    loop = asyncio.get_running_loop()

    try:
        # Gemini SDK generate_content_stream is synchronous; wrap in executor
        stream = await asyncio.wait_for(
            loop.run_in_executor(
                _gemini_executor,
                lambda: client.models.generate_content_stream(**generate_kwargs),
            ),
            timeout=120,
        )
    except asyncio.TimeoutError:
        raise Exception(f"Gemini streaming timed out for model={native_model}")
    except ClientError as e:
        if e.code == 429 or "RESOURCE_EXHAUSTED" in str(e):
            raise RateLimitExhausted(model=model, retry_after=60)
        raise

    # Iterate the synchronous stream in executor
    def _iter_stream():
        chunks = []
        for chunk in stream:
            chunks.append(chunk)
        return chunks

    try:
        all_chunks = await asyncio.wait_for(
            loop.run_in_executor(_gemini_executor, _iter_stream),
            timeout=180,
        )
    except asyncio.TimeoutError:
        raise Exception(f"Gemini streaming iteration timed out for model={native_model}")

    for chunk in all_chunks:
        delta_dict: dict[str, Any] = {}
        text = ""
        if hasattr(chunk, "text") and chunk.text:
            text = chunk.text
        elif chunk.candidates:
            candidate = chunk.candidates[0]
            if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
                text_parts = [p.text for p in candidate.content.parts if hasattr(p, "text") and p.text]
                text = "".join(text_parts)

        if text:
            delta_dict["content"] = text

        usage = None
        um = getattr(chunk, "usage_metadata", None)
        if um:
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
            }

        finish_reason = None
        if chunk.candidates:
            raw_reason = getattr(chunk.candidates[0], "finish_reason", None)
            if raw_reason:
                reason_str = str(raw_reason).lower()
                if "safety" in reason_str:
                    finish_reason = "content_filter"
                elif "length" in reason_str or "max_tokens" in reason_str:
                    finish_reason = "length"
                elif "stop" in reason_str:
                    finish_reason = "stop"

        if delta_dict or finish_reason:
            yield delta_dict, usage, finish_reason


async def _dispatch_model_stream(
    model: str,
    request: "ChatCompletionRequest",
    provider: str | None,
):
    """Route to the correct streaming backend. Yields (delta, usage, finish_reason) tuples."""
    from nadirclaw.rate_limit import get_model_rate_limiter

    # Check per-model rate limit before streaming
    limiter = get_model_rate_limiter()
    retry_after = limiter.check(model)
    if retry_after is not None:
        logger.warning(
            "Per-model rate limit hit for %s (streaming, retry in %ds)", model, retry_after,
        )
        raise RateLimitExhausted(model=model, retry_after=retry_after)

    if provider == "google":
        async_gen = None
        # _stream_gemini is a sync generator; wrap it
        for item in _stream_gemini(model, request, provider):
            yield item
    else:
        async for item in _stream_litellm(model, request, provider):
            yield item


async def _stream_with_fallback(
    selected_model: str,
    request: "ChatCompletionRequest",
    provider: str | None,
    analysis_info: Dict[str, Any],
    request_id: str,
):
    """True streaming with automatic fallback on pre-content errors.

    Yields OpenAI-compatible SSE data strings. If the primary model fails
    before yielding any content, transparently switches to fallback models.
    If it fails mid-stream, yields an error notice and stops.
    """
    from nadirclaw.credentials import detect_provider

    tier = analysis_info.get("tier", "")
    full_chain = settings.get_tier_fallback_chain(tier) if tier else settings.FALLBACK_CHAIN
    fallback_chain = _order_fallback_candidates([m for m in full_chain if m != selected_model])
    models_to_try = [selected_model] + fallback_chain
    created = int(time.time())
    failed_models: list[str] = []
    last_error: Exception | None = None

    for i, model in enumerate(models_to_try):
        if i > 0:
            logger.warning(
                "⚡ %s failed (%s) — trying streaming fallback %s (%d/%d)",
                failed_models[-1], type(last_error).__name__, model, i, len(models_to_try) - 1,
            )
            provider = detect_provider(model)

        content_started = False
        accumulated_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        last_finish = None

        try:
            first_chunk = True
            async for delta_dict, usage, finish_reason in _dispatch_model_stream(model, request, provider):
                if usage:
                    accumulated_usage = usage
                if finish_reason:
                    last_finish = finish_reason

                if not delta_dict:
                    continue

                # Add role on first content chunk
                if first_chunk and "role" not in delta_dict:
                    delta_dict["role"] = "assistant"
                first_chunk = False
                content_started = True

                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta_dict, "finish_reason": None}],
                }
                yield {"data": json.dumps(chunk)}

            # Stream completed — send finish chunk with usage
            finish_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": last_finish or "stop"}],
                "usage": {
                    "prompt_tokens": accumulated_usage["prompt_tokens"],
                    "completion_tokens": accumulated_usage["completion_tokens"],
                    "total_tokens": accumulated_usage["prompt_tokens"] + accumulated_usage["completion_tokens"],
                },
            }
            yield {"data": json.dumps(finish_chunk)}
            yield {"data": "[DONE]"}

            _record_provider_success(model)

            # Update analysis_info in-place for logging
            if failed_models:
                analysis_info["fallback_from"] = selected_model
                analysis_info["fallback_chain_tried"] = failed_models
                analysis_info["selected_model"] = model
                analysis_info["strategy"] = analysis_info.get("strategy", "smart-routing") + "+fallback"
            analysis_info["_stream_model"] = model
            analysis_info["_stream_usage"] = accumulated_usage
            return  # Success

        except (RateLimitExhausted, Exception) as e:
            if isinstance(e, HTTPException):
                raise  # Don't fallback on auth/validation errors

            if content_started:
                # Mid-stream failure — can't restart, notify client
                logger.error("Mid-stream failure on %s: %s", model, e)
                error_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": "\n\n[⚠️ Stream interrupted — model error mid-response]"},
                        "finish_reason": None,
                    }],
                }
                yield {"data": json.dumps(error_chunk)}
                finish_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield {"data": json.dumps(finish_chunk)}
                yield {"data": "[DONE]"}
                analysis_info["_stream_model"] = model
                analysis_info["_stream_usage"] = accumulated_usage
                analysis_info["_stream_error"] = str(e)
                _record_provider_failure(model, e)
                return

            # Pre-content failure — can try fallback
            analysis_info.setdefault("fallback_reasons", []).append(
                _fallback_reason(model, e)
            )
            _record_provider_failure(model, e)
            failed_models.append(model)
            last_error = e
            continue

    # All models exhausted
    logger.error("All streaming models exhausted: %s", failed_models)
    error_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": selected_model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": "⚠️ All configured models are currently unavailable. Please try again shortly."},
            "finish_reason": None,
        }],
    }
    yield {"data": json.dumps(error_chunk)}
    finish_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": selected_model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield {"data": json.dumps(finish_chunk)}
    yield {"data": "[DONE]"}
    analysis_info["_stream_model"] = selected_model
    analysis_info["_stream_usage"] = {"prompt_tokens": 0, "completion_tokens": 0}
    analysis_info["_stream_error"] = "all_models_exhausted"


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
# /v1/messages — Anthropic-compatible endpoint (Claude Code talks here)
# ---------------------------------------------------------------------------

_ANTHROPIC_UPSTREAM = "https://api.anthropic.com/v1/messages"
_CLAUDE_OAUTH_BETA = "oauth-2025-04-20,claude-code-20250219"


def _anthropic_messages_to_chat(messages: List[Dict[str, Any]]) -> List[ChatMessage]:
    """Convert Anthropic message blocks to our internal ChatMessage shape.

    Used only for routing classification — the actual upstream call sends
    the original Anthropic body untouched.
    """
    out: List[ChatMessage] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content")
        # Normalize: string stays string; list of blocks → list of dicts the
        # ChatMessage.text_content() can read ({"type":"text","text":...}).
        if isinstance(content, list):
            normalized: List[Any] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    normalized.append({"type": "text", "text": block.get("text", "")})
                elif isinstance(block, dict) and block.get("type") == "tool_result":
                    inner = block.get("content")
                    if isinstance(inner, str):
                        normalized.append({"type": "text", "text": inner})
                    elif isinstance(inner, list):
                        for b in inner:
                            if isinstance(b, dict) and b.get("type") == "text":
                                normalized.append({"type": "text", "text": b.get("text", "")})
            out.append(ChatMessage(role=role, content=normalized))
        else:
            out.append(ChatMessage(role=role, content=content))
    return out


async def _resolve_messages_model(
    requested_model: str,
    chat_messages: List[ChatMessage],
    current_user: UserSession,
) -> Tuple[str, Dict[str, Any]]:
    """Resolve the requested model into an upstream Anthropic model id.

    Mirrors the routing logic in /v1/chat/completions but limited to the
    Anthropic-compatible surface (no LiteLLM aliases — Claude Code is
    talking to us as if we were api.anthropic.com).
    """
    from nadirclaw.routing import resolve_profile

    profile = resolve_profile(requested_model)
    if profile == "eco":
        return settings.SIMPLE_MODEL, {"strategy": "profile:eco", "tier": "simple"}
    if profile == "premium":
        return settings.COMPLEX_MODEL, {"strategy": "profile:premium", "tier": "complex"}
    if profile == "reasoning":
        return settings.REASONING_MODEL, {"strategy": "profile:reasoning", "tier": "reasoning"}
    if profile == "free":
        return settings.FREE_MODEL, {"strategy": "profile:free", "tier": "free"}
    if profile == "auto" or not requested_model:
        selected, analysis = await _smart_route_full(chat_messages, current_user)
        return selected, {"strategy": "smart_route", **(analysis or {})}
    return requested_model, {"strategy": "direct", "tier": "direct"}


def _strip_provider_prefix(model_id: str) -> str:
    """LiteLLM-style prefixes (anthropic/, claude/) → bare model id for Anthropic."""
    if not model_id:
        return model_id
    for prefix in ("anthropic/", "claude/"):
        if model_id.startswith(prefix):
            return model_id[len(prefix):]
    return model_id


def _extract_text_from_anthropic_response(payload: Dict[str, Any]) -> str:
    text = ""
    for block in payload.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            text += block.get("text", "")
    return text


@app.post("/v1/messages")
async def anthropic_messages(
    raw: Request,
    current_user: UserSession = Depends(validate_local_auth),
):
    """Anthropic /v1/messages-compatible endpoint with NadirClaw routing.

    The proxy treats the incoming body as opaque Anthropic JSON and only
    rewrites the `model` field before forwarding upstream. Streaming
    responses are piped through SSE-byte-for-SSE-byte.
    """
    from nadirclaw.credentials import get_credential

    try:
        body = await raw.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {e}")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    requested_model = body.get("model") or ""
    stream = bool(body.get("stream"))
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be an array")

    chat_messages = _anthropic_messages_to_chat(messages)
    selected_model, analysis_info = await _resolve_messages_model(
        requested_model, chat_messages, current_user
    )
    upstream_model = _strip_provider_prefix(selected_model)

    body["model"] = upstream_model

    token = get_credential("anthropic")
    if not token:
        raise HTTPException(
            status_code=401,
            detail="No anthropic credential. Run `nadirclaw auth setup-token`.",
        )
    headers = {
        "Authorization": f"Bearer {token}",
        "anthropic-version": body.pop("anthropic_version", None) or raw.headers.get("anthropic-version") or "2023-06-01",
        "anthropic-beta": raw.headers.get("anthropic-beta") or _CLAUDE_OAUTH_BETA,
        "content-type": "application/json",
    }
    # If the upstream token is an API key (not OAuth), switch to x-api-key.
    if token.startswith("sk-ant-api"):
        headers.pop("Authorization", None)
        headers["x-api-key"] = token

    import httpx
    from fastapi.responses import StreamingResponse

    log_entry: Dict[str, Any] = {
        "type": "messages",
        "requested_model": requested_model,
        "selected_model": upstream_model,
        "streaming": stream,
        **analysis_info,
    }

    if not stream:
        async with httpx.AsyncClient(timeout=300) as client:
            try:
                upstream = await client.post(_ANTHROPIC_UPSTREAM, headers=headers, json=body)
            except httpx.HTTPError as e:
                _log_request({**log_entry, "error": f"upstream_error: {e}"})
                raise HTTPException(status_code=502, detail=f"upstream error: {e}")

        if upstream.status_code != 200:
            _log_request({**log_entry, "error": f"upstream_status_{upstream.status_code}", "body": upstream.text[:500]})
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
            )

        payload = upstream.json()
        usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
        _log_request({
            **log_entry,
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "response_preview": _extract_text_from_anthropic_response(payload)[:200],
        })
        return JSONResponse(content=payload, status_code=200)

    # Streaming: pipe SSE bytes straight through.
    async def proxy_stream():
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST", _ANTHROPIC_UPSTREAM, headers=headers, json=body
            ) as upstream:
                if upstream.status_code != 200:
                    err_body = await upstream.aread()
                    _log_request({
                        **log_entry,
                        "error": f"upstream_stream_status_{upstream.status_code}",
                        "body": err_body[:500].decode("utf-8", errors="replace"),
                    })
                    yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'upstream_error', 'message': err_body.decode('utf-8', errors='replace')[:500]}})}\n\n".encode()
                    return
                async for chunk in upstream.aiter_bytes():
                    if chunk:
                        yield chunk
        _log_request(log_entry)

    return StreamingResponse(proxy_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# /v1/models & /health
# ---------------------------------------------------------------------------

@app.get("/v1/cache")
async def get_cache_stats(
    current_user: UserSession = Depends(validate_local_auth),
) -> Dict[str, Any]:
    """Get prompt cache statistics."""
    from nadirclaw.cache import get_prompt_cache
    return get_prompt_cache().get_stats()


@app.get("/v1/budget")
async def get_budget(
    current_user: UserSession = Depends(validate_local_auth),
) -> Dict[str, Any]:
    """Get current spend and budget status."""
    from nadirclaw.budget import get_budget_tracker
    return get_budget_tracker().get_status()


@app.get("/v1/rate-limits")
async def get_rate_limits(
    current_user: UserSession = Depends(validate_local_auth),
) -> Dict[str, Any]:
    """Get current per-model rate limit status."""
    from nadirclaw.rate_limit import get_model_rate_limiter
    return get_model_rate_limiter().get_status()


@app.get("/v1/models")
async def list_models(
    current_user: UserSession = Depends(validate_local_auth),
) -> Dict[str, Any]:
    now = int(time.time())
    created_at_iso = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Routing profiles first, then tier models. We emit BOTH Anthropic-style
    # fields (`type`, `display_name`, `created_at`) and OpenAI-style fields
    # (`object`, `created`, `owned_by`) so every client (Claude Code, Open
    # WebUI, Cursor, OpenClaw) sees something it understands.
    profile_meta = [
        ("nadir-auto",      "Nadir — Auto",      "Smart routing per prompt (recommended)"),
        ("nadir-eco",       "Nadir — Eco",       "Always cheapest tier (Haiku-class)"),
        ("nadir-premium",   "Nadir — Premium",   "Always strongest tier (Opus-class)"),
        ("nadir-reasoning", "Nadir — Reasoning", "Always reasoning model"),
        ("nadir-free",      "Nadir — Free",      "Always local/free tier"),
        # Legacy short names — kept so existing OpenAI-compatible clients
        # (Open WebUI, Cursor, OpenClaw) don't break.
        ("auto",    "Auto",    "Smart routing per prompt"),
        ("eco",     "Eco",     "Cheapest tier"),
        ("premium", "Premium", "Strongest tier"),
    ]
    profiles = [
        {
            "id": pid,
            "type": "model",
            "display_name": display,
            "description": desc,
            "created_at": created_at_iso,
            # OpenAI-compatible legacy fields
            "object": "model",
            "created": now,
            "owned_by": "nadirclaw",
        }
        for pid, display, desc in profile_meta
    ]
    tier_data = [
        {
            "id": m,
            "type": "model",
            "display_name": m,
            "created_at": created_at_iso,
            "object": "model",
            "created": now,
            "owned_by": m.split("/")[0] if "/" in m else "api",
        }
        for m in settings.tier_models
    ]
    return {"object": "list", "data": profiles + tier_data}


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint — scrape with /metrics."""
    from nadirclaw.metrics import render_metrics
    from fastapi.responses import Response
    return Response(
        content=render_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": __version__,
        "simple_model": settings.SIMPLE_MODEL,
        "complex_model": settings.COMPLEX_MODEL,
    }


@app.get("/internal/provider_health")
async def provider_health():
    if settings.PROVIDER_HEALTH is not True:
        raise HTTPException(status_code=404, detail="Not found")
    return _provider_health_tracker().snapshot()


@app.get("/")
async def root():
    return {
        "name": "NadirClaw",
        "version": __version__,
        "description": "Open-source LLM router",
        "status": "ok",
    }
