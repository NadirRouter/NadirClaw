# Context Optimization

Context optimization compacts bloated conversation context before dispatching to the LLM. This reduces token usage and cost, especially in long multi-turn conversations where earlier context accumulates.

## Enabling Context Optimization

### Via environment variable

```bash
NADIRCLAW_OPTIMIZE=safe   # or: aggressive
```

### Via CLI flag

```bash
nadirclaw serve --optimize safe
```

### Per-request

Include `"optimize": "safe"` or `"optimize": "aggressive"` in the request body:

```json
{
  "messages": [...],
  "optimize": "aggressive"
}
```

Per-request overrides take precedence over the global setting.

## Modes

### `off` (default)

No processing. Zero overhead. All messages are passed through as-is.

### `safe`

Deterministic, lossless transforms only. These never remove semantically meaningful content:

- **System prompt deduplication** -- Removes system prompt text that is duplicated verbatim in later messages. Some tools prepend the system prompt to every turn, leading to massive duplication.
- **Whitespace normalization** -- Collapses excessive whitespace and blank lines without changing meaning.
- **Turn trimming** -- When the conversation exceeds `NADIRCLAW_OPTIMIZE_MAX_TURNS` (default: 40), older turns are removed while preserving the system prompt and most recent context.

Safe mode is recommended for production use. It provides meaningful savings (often 10-30%) with zero risk of losing important context.

### `aggressive`

Includes all safe transforms, plus:

- **Semantic deduplication** -- Uses embeddings to detect and remove near-duplicate messages in the conversation. Useful when a tool resends similar context across turns.
- **More aggressive turn trimming** -- Keeps fewer historical turns.

Aggressive mode can provide 30-60% token savings on long conversations, but may occasionally remove context that is subtly different but appears similar. Use it when cost savings outweigh the small risk of context loss.

## Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NADIRCLAW_OPTIMIZE` | str | `off` | Global optimization mode |
| `NADIRCLAW_OPTIMIZE_MAX_TURNS` | int | `40` | Maximum conversation turns to keep (minimum: 4) |

## Testing Optimization

Use the CLI to preview what optimization does to a conversation without sending it to an LLM:

```bash
# From a file
nadirclaw optimize conversation.json --mode safe

# Aggressive mode with JSON output
nadirclaw optimize conversation.json --mode aggressive --format json

# From stdin
echo '{"messages": [{"role": "user", "content": "hello"}]}' | nadirclaw optimize
```

Output shows:

- Original token count
- Optimized token count
- Tokens saved (and percentage)
- Which transforms were applied

### Example

```
Mode:          safe
Original:      ~12,450 tokens
Optimized:     ~8,230 tokens
Saved:         ~4,220 tokens (33.9%)
Transforms:    system_dedup, whitespace_normalize, turn_trim
```

## How It Fits in the Pipeline

Context optimization runs **after** routing decisions are made and **before** the LLM call:

```
Classify prompt --> Select model --> Optimize context --> Call LLM
```

This means the classifier sees the full context for accurate routing, but the LLM receives a compacted version for cost savings.

When optimization is applied, the response metadata includes optimization details:

```json
{
  "nadirclaw_metadata": {
    "optimization": {
      "optimization_mode": "safe",
      "original_tokens": 12450,
      "optimized_tokens": 8230,
      "tokens_saved": 4220,
      "optimizations_applied": ["system_dedup", "whitespace_normalize", "turn_trim"]
    }
  }
}
```
