# Configuration Reference

NadirClaw reads configuration from environment variables. These can be set in `~/.nadirclaw/.env` (loaded automatically), your shell environment, or passed as CLI flags to `nadirclaw serve`.

## General

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NADIRCLAW_PORT` | int | `8856` | Port the server listens on |
| `NADIRCLAW_AUTH_TOKEN` | str | `""` (disabled) | Bearer token for API authentication. When empty, no auth is required (local-only mode) |
| `NADIRCLAW_LOG_DIR` | path | `~/.nadirclaw/logs` | Directory for request logs (JSONL and SQLite) |
| `NADIRCLAW_LOG_RAW` | bool | `false` | When `true`, log full raw request messages and response content to JSONL |
| `NADIRCLAW_API_BASE` | str | `""` | Custom base URL for OpenAI-compatible endpoints (vLLM, LocalAI, etc.). Passed as `api_base` to all non-Ollama, non-Gemini LiteLLM calls |

## Models

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NADIRCLAW_MODELS` | str | `"openai-codex/gpt-5.3-codex,gemini-3-flash-preview"` | Comma-separated list of models. First is used as complex, last as simple |
| `NADIRCLAW_SIMPLE_MODEL` | str | Last model in `NADIRCLAW_MODELS` | Model for simple/easy prompts |
| `NADIRCLAW_COMPLEX_MODEL` | str | First model in `NADIRCLAW_MODELS` | Model for complex/hard prompts |
| `NADIRCLAW_MID_MODEL` | str | Falls back to `SIMPLE_MODEL` | Model for medium-complexity prompts. Setting this enables 3-tier routing |
| `NADIRCLAW_REASONING_MODEL` | str | Falls back to `COMPLEX_MODEL` | Model for reasoning-heavy tasks |
| `NADIRCLAW_FREE_MODEL` | str | Falls back to `SIMPLE_MODEL` | Free/cheapest fallback model |

## Provider API Keys

| Variable | Type | Description |
|----------|------|-------------|
| `ANTHROPIC_API_KEY` | str | Anthropic API key for Claude models |
| `OPENAI_API_KEY` | str | OpenAI API key |
| `GEMINI_API_KEY` | str | Google Gemini API key (also accepts `GOOGLE_API_KEY`) |
| `OLLAMA_API_BASE` | str | Ollama server URL (default: `http://localhost:11434`) |

!!! tip
    You can also manage credentials interactively with `nadirclaw auth add` or use OAuth login with `nadirclaw auth openai login`, `nadirclaw auth anthropic login`, etc.

## Classifier

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NADIRCLAW_CLASSIFIER` | str | `"trained"` | Classifier type: `trained` (sklearn, 96%+ accuracy), `cascade` (ternary with escalation), or `binary` (fast centroid) |
| `NADIRCLAW_CONFIDENCE_THRESHOLD` | float | `0.06` | Confidence threshold for binary classifier routing decisions |

### Cascade Classifier Settings

These only apply when `NADIRCLAW_CLASSIFIER=cascade`:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NADIRCLAW_CASCADE_THRESHOLD` | float | `0.75` | Confidence threshold for cascade escalation. When the fast centroid classifier's confidence is below this, it escalates to structural feature analysis |
| `NADIRCLAW_CASCADE_TEMPERATURE` | float | `0.30` | Softmax temperature for centroid similarity to probability conversion. Lower values produce sharper decisions |
| `NADIRCLAW_CASCADE_SUB_CLUSTERS` | int | `5` | Number of k-means sub-clusters for the complex tier centroid |

## Tier Routing

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NADIRCLAW_TIER_THRESHOLDS` | str | `"0.35,0.65"` | Comma-separated pair of floats defining 3-tier boundaries. Scores <= first value are simple, >= second are complex, between is mid |

## Fallback Chains

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NADIRCLAW_FALLBACK_CHAIN` | str | All tier models (deduplicated) | Global fallback chain. Comma-separated list of models to try when the primary fails. Example: `gpt-4.1,claude-sonnet-4-5-20250929,gemini-2.5-flash` |
| `NADIRCLAW_SIMPLE_FALLBACK` | str | Falls back to global chain | Per-tier fallback chain for simple prompts |
| `NADIRCLAW_MID_FALLBACK` | str | Falls back to global chain | Per-tier fallback chain for mid-complexity prompts |
| `NADIRCLAW_COMPLEX_FALLBACK` | str | Falls back to global chain | Per-tier fallback chain for complex prompts |

## Rate Limiting

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NADIRCLAW_MODEL_RATE_LIMITS` | str | `""` | Per-model rate limits. Format: `model=rpm,model2=rpm2` |
| `NADIRCLAW_DEFAULT_MODEL_RPM` | int | `0` (unlimited) | Default max requests per minute per model |

## Context Optimization

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NADIRCLAW_OPTIMIZE` | str | `"off"` | Context optimization mode: `off`, `safe`, or `aggressive` |
| `NADIRCLAW_OPTIMIZE_MAX_TURNS` | int | `40` | Maximum conversation turns to keep when trimming (minimum: 4) |

See [Context Optimization](context-optimize.md) for details on each mode.

## Budget

Budget variables are checked by the budget tracker at runtime. Set them to receive warnings when approaching limits.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NADIRCLAW_DAILY_BUDGET` | float | None | Daily spend limit in USD |
| `NADIRCLAW_MONTHLY_BUDGET` | float | None | Monthly spend limit in USD |

## Example `.env` File

```bash
# ~/.nadirclaw/.env

# Models
NADIRCLAW_SIMPLE_MODEL=gemini-3-flash-preview
NADIRCLAW_COMPLEX_MODEL=openai-codex/gpt-5.3-codex
NADIRCLAW_MID_MODEL=gpt-4.1-mini

# API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...

# Classifier
NADIRCLAW_CLASSIFIER=trained

# Optimization
NADIRCLAW_OPTIMIZE=safe

# Budget
NADIRCLAW_DAILY_BUDGET=5.00
NADIRCLAW_MONTHLY_BUDGET=50.00

# Fallback
NADIRCLAW_FALLBACK_CHAIN=openai-codex/gpt-5.3-codex,gemini-2.5-flash,gpt-4.1-mini
```
