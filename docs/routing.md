# How Routing Works

NadirClaw uses a multi-stage routing pipeline to decide which model handles each request. The goal is to route simple prompts to cheap models and complex prompts to capable ones, maximizing quality while minimizing cost.

## The Routing Pipeline

Each request passes through these stages in order:

```
Request arrives
     |
     v
1. Rules Engine (YAML rules, first-match wins)
     |  --> If a rule matches with force_model, skip classification
     v
2. Routing Profiles (auto/eco/premium/free/reasoning)
     |  --> If model=eco or model=premium, skip classification
     v
3. Session Cache (have we seen this conversation before?)
     |  --> If cache hit, reuse the previous routing decision
     v
4. Classifier (binary, cascade, or trained)
     |  --> Produces: tier, confidence, complexity_score
     v
5. Routing Modifiers (agentic detection, reasoning detection, context window)
     |  --> May upgrade tier based on request characteristics
     v
6. Pareto Optimizer (select best model within tier)
     |  --> Balances quality, cost, and latency
     v
7. Post-classification Rules (tier-match rules evaluated after classification)
     |
     v
8. Context Optimization (compact messages if enabled)
     |
     v
9. Prompt Cache (have we answered this exact prompt before?)
     |
     v
10. Model Dispatch + Fallback Chain
```

## Classifier Types

NadirClaw ships with three classifier implementations. Set the type via `NADIRCLAW_CLASSIFIER`.

### Binary (`binary`)

The original, fastest classifier. Uses pre-computed centroid vectors from prototype prompts.

- Embeds the prompt using sentence-transformers (~10ms on warm encoder)
- Computes cosine similarity to simple and complex centroids
- Returns `simple` or `complex` with a confidence score
- Good for 2-tier setups (cheap model + premium model)

### Cascade (`cascade`)

A confidence-aware ternary classifier with automatic escalation:

1. **Fast centroid phase** -- Runs a 3-centroid classifier (simple/medium/complex) with k-means sub-clustering for the complex tier (~10ms)
2. **Escalation** -- If confidence is below `NADIRCLAW_CASCADE_THRESHOLD` (default: 0.75), escalates to a structural feature analyzer that extracts 20+ textual signals
3. **Returns** a calibrated confidence and ternary tier (simple/medium/complex)

Best for 3-tier setups with a mid-tier model.

### Trained (`trained`) -- Default

An sklearn-based classifier trained on labeled data, achieving 96%+ accuracy:

- Uses a pre-trained model shipped with the package (`trained_model.pkl`)
- Extracts features from the prompt text
- Returns tier, confidence, and complexity score
- Can be retrained on your own data with `nadirclaw train`

## Routing Profiles

When a client specifies a model name, NadirClaw checks for routing profiles before classification:

| Profile | Behavior |
|---------|----------|
| `auto` (or omitted) | Run the full classification pipeline |
| `eco` | Always route to `SIMPLE_MODEL` |
| `premium` | Always route to `COMPLEX_MODEL` |
| `free` | Always route to `FREE_MODEL` |
| `reasoning` | Always route to `REASONING_MODEL` |

If the model name is not a profile and not `auto`, NadirClaw checks for model aliases. If no alias matches, the request is sent directly to the specified model (passthrough mode).

## Routing Modifiers

After classification, NadirClaw applies modifiers that can upgrade the routing decision based on request characteristics:

- **Agentic detection** -- Requests with tool definitions or tool-role messages may be upgraded to a more capable model
- **Reasoning detection** -- Requests that appear to require step-by-step reasoning may be routed to the reasoning model
- **Context window filtering** -- If the conversation exceeds a model's context window, NadirClaw automatically selects a model with a larger window
- **Image detection** -- Requests with images are routed to vision-capable models

## Fallback Chains

When a model fails (rate limit, timeout, 5xx error), NadirClaw cascades through a fallback chain:

1. Try the selected model
2. On failure, try each model in the fallback chain in order
3. Skip the model that just failed
4. If all models fail, return a graceful error response

### Configuring Fallbacks

**Global chain** (applies to all tiers):

```bash
NADIRCLAW_FALLBACK_CHAIN=gpt-4.1,claude-sonnet-4-5-20250929,gemini-2.5-flash
```

**Per-tier chains** (override the global chain for specific tiers):

```bash
NADIRCLAW_SIMPLE_FALLBACK=gemini-2.5-flash,gpt-4.1-nano
NADIRCLAW_MID_FALLBACK=gpt-4.1-mini,gemini-2.5-flash
NADIRCLAW_COMPLEX_FALLBACK=claude-sonnet-4-5-20250929,gpt-4.1
```

When no explicit chain is configured, NadirClaw builds a default chain from all configured tier models (complex, mid, simple, reasoning, free), deduplicated.

## Rules Engine

The rules engine allows declarative routing overrides via a YAML file at `~/.nadirclaw/rules.yaml`. Rules are evaluated top-to-bottom; the first match wins.

### Rule Structure

```yaml
rules:
  - name: "coding-agent"
    match:
      system_prompt_contains: "You are a coding assistant"
    action:
      force_model: "openai-codex/gpt-5.3-codex"

  - name: "night-mode-cheap"
    match:
      time_range: "22:00-06:00"
    action:
      force_tier: "simple"

  - name: "complex-cap"
    match:
      tier: "complex"
    action:
      max_cost_per_request: 0.10
```

### Match Conditions

All conditions in a rule must be true for the rule to fire (AND logic).

| Condition | Description |
|-----------|-------------|
| `system_prompt_contains` | Case-insensitive substring match on the system prompt |
| `system_prompt_regex` | Regex match against the system prompt |
| `prompt_contains` | Case-insensitive substring match on the last user message |
| `prompt_regex` | Regex match against the last user message |
| `time_range` | Time-of-day range in `HH:MM-HH:MM` format (local time, wraps past midnight) |
| `header` | HTTP header equality check in `Header-Name: value` format |
| `tier` | Match the classifier-assigned tier (evaluated in post-classification pass) |

### Actions

| Action | Description |
|--------|-------------|
| `force_model` | Bypass classification and use this specific model |
| `force_tier` | Override the tier (`simple`, `mid`, `complex`, `reasoning`) |
| `max_cost_per_request` | Advisory cost cap (stored in metadata) |

!!! note
    Rules with `tier` in their match conditions are evaluated **after** classification (post-classification pass). All other rules are evaluated **before** classification and can bypass it entirely.

## Session Cache

NadirClaw maintains an in-memory LRU cache of routing decisions keyed by conversation hash. When a follow-up message arrives in the same conversation, the cached routing decision is reused instead of re-running the classifier.

This provides:

- Consistent model assignment within a conversation
- Faster response times for follow-up messages
- Zero overhead for the classifier on cache hits

## Pareto Optimizer

When multiple models are available for a tier, the Pareto optimizer selects the best one by balancing:

- **Quality** -- model capability/benchmark scores
- **Cost** -- per-token pricing
- **Latency** -- typical response time

Weights can be customized per-request via the `X-Routing-Priority` header:

```
X-Routing-Priority: quality=0.7,cost=0.2,latency=0.1
```

The optimizer also filters models by required capabilities (e.g., tools, vision) to ensure the selected model can handle the request.
