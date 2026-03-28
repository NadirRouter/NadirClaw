# CLI Reference

NadirClaw provides a comprehensive CLI for managing the router, classifying prompts, monitoring usage, and configuring integrations.

```bash
nadirclaw [COMMAND] [OPTIONS]
```

Running `nadirclaw` with no command shows the quick-start help.

---

## Core Commands

### `nadirclaw serve`

Start the NadirClaw router server.

```bash
nadirclaw serve [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--port` | int | 8856 | Port to listen on |
| `--simple-model` | str | From config | Model for simple prompts |
| `--complex-model` | str | From config | Model for complex prompts |
| `--models` | str | From config | Comma-separated model list (legacy) |
| `--token` | str | From config | Auth token |
| `--verbose` | flag | | Enable debug-level logging |
| `--log-raw` | flag | | Log full raw requests and responses |
| `--optimize` | choice | off | Context optimization mode: `off`, `safe`, or `aggressive` |

If no configuration exists, the wizard runs automatically on first start.

### `nadirclaw setup`

Interactive setup wizard for configuring providers and models.

```bash
nadirclaw setup [--reconfigure]
```

| Option | Type | Description |
|--------|------|-------------|
| `--reconfigure` | flag | Re-run setup even if already configured |

### `nadirclaw demo`

Classify sample prompts and show projected savings. No API keys required.

```bash
nadirclaw demo [--count N]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--count` | int | 10 | Number of sample prompts to classify |

### `nadirclaw test`

Send a probe request to each configured model tier to verify API keys and connectivity.

```bash
nadirclaw test [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--simple-model` | str | From config | Override simple model for this test |
| `--complex-model` | str | From config | Override complex model for this test |
| `--timeout` | int | 30 | Request timeout in seconds |

Exits with code 1 if any model fails.

---

## Classification

### `nadirclaw classify`

Classify a prompt as simple, medium, or complex without starting the server.

```bash
nadirclaw classify [OPTIONS] PROMPT...
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | text | Output format: `text` or `json` |
| `--cascade` | flag | | Use cascade classifier (ternary) |

Multi-word prompts do not need quotes:

```bash
nadirclaw classify What is the meaning of life

nadirclaw classify --format json Refactor this auth module
```

JSON output:

```json
{
  "tier": "simple",
  "is_complex": false,
  "confidence": 0.9234,
  "score": 0.1500,
  "model": "gemini-3-flash-preview",
  "prompt": "What is the meaning of life",
  "classifier": "binary"
}
```

### `nadirclaw optimize`

Test context optimization on a file or stdin (dry-run).

```bash
nadirclaw optimize [FILE] [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mode` | choice | safe | Optimization mode: `safe` or `aggressive` |
| `--format` | choice | text | Output format: `text` or `json` |

The input can be a JSON messages array, a JSON object with a `messages` key, or plain text (wrapped as a single user message).

```bash
# From file
nadirclaw optimize conversation.json --mode aggressive

# From stdin
cat messages.json | nadirclaw optimize
```

---

## Monitoring

### `nadirclaw status`

Check if the server is running and show current configuration.

```bash
nadirclaw status
```

Shows: model configuration, port, auth status, credential status, and server reachability.

### `nadirclaw report`

Show a summary report of request logs.

```bash
nadirclaw report [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--since` | str | All time | Time filter: `24h`, `7d`, or a date like `2025-02-01` |
| `--model` | str | All | Filter by model name (substring match) |
| `--format` | choice | text | Output format: `text` or `json` |
| `--export` | path | stdout | Export report to a file |
| `--by-model` | flag | | Show per-model cost breakdown |
| `--by-day` | flag | | Show per-day cost breakdown |

### `nadirclaw dashboard`

Live terminal dashboard showing real-time routing stats.

```bash
nadirclaw dashboard [--refresh 2.0]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--refresh` | float | 2.0 | Refresh interval in seconds |

!!! tip
    A web-based dashboard is also available at `http://localhost:8856/dashboard` while the server is running.

### `nadirclaw savings`

Show how much money NadirClaw saved you.

```bash
nadirclaw savings [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--since` | str | All time | Time filter |
| `--baseline` | str | Auto-detected | Model to compare against (default: most expensive in logs) |
| `--format` | choice | text | Output format: `text` or `json` |

### `nadirclaw budget`

Show current spend and budget status.

```bash
nadirclaw budget [--format text|json]
```

### `nadirclaw cache`

Show prompt cache statistics (queries the running server).

```bash
nadirclaw cache [--format text|json]
```

---

## Data Export

### `nadirclaw export`

Export request logs for offline analysis.

```bash
nadirclaw export [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | csv | Export format: `csv` or `jsonl` |
| `--since` | str | All time | Time filter |
| `--model` | str | All | Filter by model name |
| `-o`, `--output` | path | stdout | Output file |

```bash
nadirclaw export --format csv --since 7d -o last_week.csv
```

### `nadirclaw flag`

Flag a request as misrouted or provide quality feedback.

```bash
nadirclaw flag REQUEST_ID [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--reason` | choice | misrouted | Reason: `misrouted`, `slow`, `bad_quality`, `good`, `other` |
| `--rating` | int (1-5) | None | Quality rating |
| `--tier` | choice | None | Correct tier: `simple`, `mid`, `complex` |
| `--model` | str | None | Correct model |

---

## Authentication

### `nadirclaw auth status`

Show configured credentials (tokens are masked).

### `nadirclaw auth add`

Add an API key for a provider interactively.

```bash
nadirclaw auth add [--provider PROVIDER] [--key KEY]
```

### `nadirclaw auth remove PROVIDER`

Remove a stored credential for a provider.

### `nadirclaw auth setup-token`

Store a Claude subscription token (from `claude setup-token`).

### OAuth Login Commands

```bash
# OpenAI subscription (ChatGPT account)
nadirclaw auth openai login [--timeout 300]
nadirclaw auth openai logout

# Anthropic (setup token or API key)
nadirclaw auth anthropic login
nadirclaw auth anthropic logout

# Google Antigravity
nadirclaw auth antigravity login [--timeout 300]
nadirclaw auth antigravity logout

# Google Gemini CLI
nadirclaw auth gemini login [--timeout 300]
nadirclaw auth gemini logout
```

---

## Integrations

### `nadirclaw openclaw onboard`

Auto-configure OpenClaw to use NadirClaw as a provider. Writes to `~/.openclaw/openclaw.json` and registers `nadirclaw/auto` as a model.

### `nadirclaw codex onboard`

Auto-configure the OpenAI Codex CLI to use NadirClaw. Writes to `~/.codex/config.toml`.

### `nadirclaw openwebui onboard`

Show setup instructions for Open WebUI integration.

---

## Training (Advanced)

### `nadirclaw train`

Retrain routing centroids from production data and user feedback.

```bash
nadirclaw train [OPTIONS]
```

| Option | Type | Description |
|--------|------|-------------|
| `--data` | path | JSONL file with labeled prompts (`{prompt, tier}`) |
| `--validate-only` | flag | Dry-run: show accuracy changes without applying |
| `--rollback` | flag | Revert to previous centroid version |
| `--format` | choice | Output format: `text` or `json` |

Training requires at least 10 samples. New centroids are deployed only if they pass validation gates (>80% accuracy, <20% tier shift).

### `nadirclaw build-centroids` (hidden)

Regenerate centroid `.npy` files from prototype prompts. Used during development.

```bash
nadirclaw build-centroids [--ternary]
```
