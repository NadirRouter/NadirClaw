# Getting Started

## Installation

### pip (recommended)

```bash
pip install nadirclaw
```

### From source

```bash
curl -fsSL https://raw.githubusercontent.com/NadirRouter/NadirClaw/main/install.sh | sh
```

### Optional extras

```bash
# Rich terminal dashboard
pip install nadirclaw[dashboard]

# OpenTelemetry tracing
pip install nadirclaw[telemetry]

# Development / testing
pip install nadirclaw[dev]

# Documentation site
pip install nadirclaw[docs]
```

## Setup Wizard

Run the interactive setup wizard to configure your API keys and models:

```bash
nadirclaw setup
```

The wizard will:

1. Ask which providers you want to use (OpenAI, Anthropic, Google, Ollama)
2. Store API keys securely in `~/.nadirclaw/credentials.json`
3. Configure your simple and complex models
4. Write settings to `~/.nadirclaw/.env`

To re-run setup later:

```bash
nadirclaw setup --reconfigure
```

## Starting the Router

```bash
nadirclaw serve
```

This starts a FastAPI server on `http://localhost:8856` with:

- `/v1/chat/completions` -- OpenAI-compatible completions endpoint
- `/dashboard` -- Web-based monitoring dashboard
- `/health` -- Health check endpoint

### Common serve options

```bash
# Custom port
nadirclaw serve --port 9000

# Override models
nadirclaw serve --simple-model gemini-2.5-flash --complex-model gpt-4.1

# Enable verbose logging
nadirclaw serve --verbose

# Enable context optimization
nadirclaw serve --optimize safe
```

## Your First Request

Once the server is running, test it with curl:

```bash
curl http://localhost:8856/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

NadirClaw will classify this as a simple prompt and route it to your cheap model. The response includes routing metadata:

```json
{
  "id": "abc-123",
  "model": "gemini-3-flash-preview",
  "choices": [{"message": {"role": "assistant", "content": "4"}}],
  "nadirclaw_metadata": {
    "routing": {
      "strategy": "smart-routing",
      "tier": "simple",
      "confidence": 0.92,
      "complexity_score": 0.15
    }
  }
}
```

## Testing Your Configuration

Before starting the server, verify your API keys work:

```bash
nadirclaw test
```

This sends a probe request to each configured model and reports latency and status.

## Integrations

NadirClaw works with any tool that supports OpenAI-compatible APIs. Point the tool's base URL to `http://localhost:8856/v1`.

### Cursor

In Cursor settings, add a custom model:

- **Base URL**: `http://localhost:8856/v1`
- **API Key**: `local` (or your configured auth token)
- **Model**: `auto` (for smart routing), `eco` (always cheap), or `premium` (always premium)

### Claude Code

Set the environment variable before running Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8856/v1
```

Or use the setup token flow:

```bash
nadirclaw auth setup-token
```

### Aider

```bash
aider --openai-api-base http://localhost:8856/v1 --openai-api-key local
```

### Continue (VS Code)

In your Continue configuration (`~/.continue/config.json`):

```json
{
  "models": [{
    "title": "NadirClaw Auto",
    "provider": "openai",
    "model": "auto",
    "apiBase": "http://localhost:8856/v1",
    "apiKey": "local"
  }]
}
```

### OpenClaw

Auto-configure OpenClaw to use NadirClaw:

```bash
nadirclaw openclaw onboard
```

### OpenAI Codex CLI

Auto-configure the Codex CLI:

```bash
nadirclaw codex onboard
```

## Routing Profiles

When specifying a model name, you can use built-in routing profiles:

| Profile | Description |
|---------|-------------|
| `auto` | Smart routing based on prompt complexity (default) |
| `eco` | Always use the cheapest model |
| `premium` | Always use the most capable model |
| `free` | Use the free-tier model |
| `reasoning` | Use the reasoning-optimized model |

## Monitoring

### Terminal dashboard

```bash
nadirclaw dashboard
```

### Web dashboard

Visit `http://localhost:8856/dashboard` while the server is running.

### Usage reports

```bash
# Summary report
nadirclaw report

# Cost breakdown by model
nadirclaw report --by-model

# See how much you've saved
nadirclaw savings
```

## Running the Demo

To see NadirClaw in action without any API keys:

```bash
nadirclaw demo
```

This classifies sample prompts and shows projected cost savings.
