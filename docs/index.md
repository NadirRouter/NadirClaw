# NadirClaw

**Open-source LLM router** -- routes simple prompts to cheap/free models and complex prompts to premium models, saving you money without sacrificing quality.

NadirClaw sits between your AI-powered tools (Cursor, Claude Code, Aider, Continue, etc.) and your LLM providers. It classifies each prompt's complexity in ~10ms and routes it to the most cost-effective model.

## Key Features

- **Smart routing** -- Classifies prompts as simple, medium, or complex and picks the right model tier automatically.
- **OpenAI-compatible API** -- Drop-in replacement at `/v1/chat/completions`. Works with any tool that speaks the OpenAI protocol.
- **Multi-provider support** -- OpenAI, Anthropic, Google Gemini, Ollama, DeepSeek, and any OpenAI-compatible endpoint via LiteLLM.
- **Multiple classifiers** -- Binary (fast centroid), cascade (ternary with escalation), and trained (sklearn, 96%+ accuracy).
- **Context optimization** -- Compacts bloated conversation context before dispatch, saving tokens and money.
- **Rules engine** -- YAML-based rules to pin models, force tiers, or cap costs based on prompt content, headers, or time of day.
- **Fallback chains** -- Automatic failover when a model is rate-limited or down.
- **Budget tracking** -- Daily and monthly spend limits with per-model cost tracking.
- **Streaming support** -- Full SSE streaming pass-through, compatible with all major clients.
- **OAuth login** -- Use your existing OpenAI, Anthropic, or Google subscriptions without API keys.

## Quick Install

```bash
pip install nadirclaw
```

Or install from source:

```bash
curl -fsSL https://raw.githubusercontent.com/NadirRouter/NadirClaw/main/install.sh | sh
```

## Quick Start

```bash
# 1. Run the setup wizard
nadirclaw setup

# 2. Start the router
nadirclaw serve

# 3. Point your tools to http://localhost:8856/v1
```

See the [Getting Started](quickstart.md) guide for detailed instructions.

## How It Works

```
Your Tool (Cursor, Claude Code, etc.)
        |
        v
   NadirClaw (localhost:8856)
        |
        +-- Classify prompt (~10ms)
        |
        +-- Simple?  --> Gemini Flash / GPT-4.1-nano / Ollama
        +-- Medium?  --> GPT-4.1-mini / Claude Haiku
        +-- Complex? --> GPT-5 / Claude Opus / Gemini Pro
```

Most coding prompts (formatting, simple questions, boilerplate) are simple and can be handled by cheap or free models. NadirClaw detects this automatically and routes accordingly, typically saving 50-80% on API costs.
