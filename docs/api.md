# API Reference

NadirClaw exposes an OpenAI-compatible REST API. The server runs on port 8856 by default.

All authenticated endpoints accept an optional `Authorization: Bearer <token>` header when `NADIRCLAW_AUTH_TOKEN` is set. When auth is disabled (default), no token is needed.

## POST /v1/chat/completions

The main endpoint. Accepts OpenAI-compatible chat completion requests, classifies the prompt, routes to the best model, and returns the response.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `messages` | array | Yes | Array of message objects with `role` and `content` |
| `model` | string | No | Model or routing profile. Use `auto` (default), `eco`, `premium`, `free`, `reasoning`, or a specific model name |
| `temperature` | float | No | Sampling temperature |
| `max_tokens` | int | No | Maximum tokens to generate |
| `top_p` | float | No | Nucleus sampling parameter |
| `stream` | bool | No | Enable SSE streaming (default: `false`) |
| `tools` | array | No | Tool/function definitions (passed through to the model) |
| `tool_choice` | string/object | No | Tool choice preference (passed through) |
| `optimize` | string | No | Per-request optimization override: `off`, `safe`, or `aggressive` |

### Request Headers

| Header | Description |
|--------|-------------|
| `Authorization` | `Bearer <token>` (when auth is enabled) |
| `X-Routing-Priority` | Pareto optimizer weights, e.g. `quality=0.7,cost=0.2,latency=0.1` |

### Response

Standard OpenAI chat completion format with additional `nadirclaw_metadata`:

```json
{
  "id": "request-uuid",
  "object": "chat.completion",
  "created": 1711234567,
  "model": "gemini-3-flash-preview",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The answer is 4."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 5,
    "total_tokens": 17
  },
  "nadirclaw_metadata": {
    "request_id": "abc-123-def",
    "response_time_ms": 450,
    "routing": {
      "strategy": "smart-routing",
      "analyzer": "trained",
      "selected_model": "gemini-3-flash-preview",
      "tier": "simple",
      "confidence": 0.92,
      "complexity_score": 0.15,
      "simple_model": "gemini-3-flash-preview",
      "complex_model": "openai-codex/gpt-5.3-codex"
    }
  }
}
```

### Response Headers

| Header | Description |
|--------|-------------|
| `X-Routed-Model` | The model that handled the request |
| `X-Routed-Tier` | The complexity tier assigned (`simple`, `mid`, `complex`) |
| `X-Complexity-Score` | Numeric complexity score (0.0 to 1.0) |
| `X-Routed-Rule` | Name of the routing rule that matched (if any) |
| `X-Pareto-Score` | Pareto optimization score (if optimizer was used) |

### Streaming

Set `"stream": true` to receive Server-Sent Events (SSE). NadirClaw supports true streaming pass-through for all providers:

```bash
curl http://localhost:8856/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a poem"}],
    "stream": true
  }'
```

---

## POST /v1/classify

Classify a prompt without calling any LLM. Useful for testing the classifier or building custom routing logic.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | Yes | The prompt text to classify |
| `system_message` | string | No | Optional system message for context |

### Response

```json
{
  "prompt": "What is 2+2?",
  "classification": {
    "strategy": "smart-routing",
    "analyzer": "trained",
    "selected_model": "gemini-3-flash-preview",
    "tier": "simple",
    "confidence": 0.95,
    "complexity_score": 0.08,
    "reasoning": "Short factual question"
  }
}
```

---

## POST /v1/classify/batch

Classify multiple prompts in a single request.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompts` | array[string] | Yes | List of prompt texts to classify |

### Response

```json
{
  "total": 3,
  "simple": 2,
  "complex": 1,
  "results": [
    {
      "prompt": "What is 2+2?",
      "selected_model": "gemini-3-flash-preview",
      "tier": "simple",
      "confidence": 0.95,
      "complexity_score": 0.08
    }
  ]
}
```

---

## POST /v1/feedback

Submit feedback on a routing decision. Used to improve classifier accuracy over time.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `request_id` | string | Yes | The request ID from the completion response |
| `rating` | int | No | Quality rating from 1 to 5 |
| `reason` | string | No | Free-text reason for feedback |
| `correct_tier` | string | No | What tier the request should have been (`simple`, `mid`, `complex`) |
| `correct_model` | string | No | What model should have been used |

### Response

```json
{
  "status": "ok",
  "feedback": {
    "request_id": "abc-123",
    "rating": 2,
    "correct_tier": "complex"
  }
}
```

---

## GET /v1/models

List available models and routing profiles.

### Response

```json
{
  "object": "list",
  "data": [
    {"id": "auto", "object": "model", "owned_by": "nadirclaw"},
    {"id": "eco", "object": "model", "owned_by": "nadirclaw"},
    {"id": "premium", "object": "model", "owned_by": "nadirclaw"},
    {"id": "openai-codex/gpt-5.3-codex", "object": "model", "owned_by": "openai-codex"},
    {"id": "gemini-3-flash-preview", "object": "model", "owned_by": "api"}
  ]
}
```

---

## GET /v1/cache

Get prompt cache statistics.

### Response

```json
{
  "enabled": true,
  "entries": 42,
  "max_size": 1000,
  "ttl": 3600,
  "hits": 128,
  "misses": 350,
  "hit_rate": 0.268,
  "total_lookups": 478
}
```

---

## GET /v1/budget

Get current spend and budget status.

### Response

```json
{
  "daily_spend": 1.23,
  "daily_budget": 5.00,
  "daily_requests": 47,
  "monthly_spend": 12.50,
  "monthly_budget": 50.00,
  "monthly_requests": 312,
  "top_models": [
    {"model": "gpt-4.1", "spend": 5.20, "requests": 89}
  ]
}
```

---

## GET /v1/rate-limits

Get current per-model rate limit status.

### Response

Returns the current state of the per-model rate limiter, including configured limits and current usage windows.

---

## GET /v1/logs

View recent request logs.

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 20 | Number of recent log entries to return |

### Response

```json
{
  "logs": [
    {
      "type": "completion",
      "request_id": "abc-123",
      "selected_model": "gemini-3-flash-preview",
      "tier": "simple",
      "total_latency_ms": 450,
      "timestamp": "2026-03-27T10:30:00Z"
    }
  ],
  "total": 1500,
  "showing": 20
}
```

---

## GET /metrics

Prometheus-compatible metrics endpoint. Returns metrics in the Prometheus text exposition format.

```bash
curl http://localhost:8856/metrics
```

---

## GET /health

Health check endpoint.

### Response

```json
{
  "status": "ok",
  "version": "0.13.0",
  "simple_model": "gemini-3-flash-preview",
  "complex_model": "openai-codex/gpt-5.3-codex"
}
```

---

## GET /

Root endpoint. Returns basic server info.

### Response

```json
{
  "name": "NadirClaw",
  "version": "0.13.0",
  "description": "Open-source LLM router",
  "status": "ok"
}
```

---

## Error Responses

All errors follow a consistent format:

| Status | Description |
|--------|-------------|
| 413 | Request content too large (> 1MB) |
| 422 | Validation error (malformed request body) |
| 429 | Rate limit exceeded. `Retry-After` header indicates wait time |
| 500 | Internal server error. Response includes the `request_id` for debugging |
