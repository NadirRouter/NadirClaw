# Routing NadirClaw at a custom OpenAI-compatible gateway (Starveri example)

This page is a worked example of pointing NadirClaw at a third-party
OpenAI-compatible gateway, using [Starveri](https://api.starveri.net) as the
concrete case. It is **not an endorsement**. Starveri is operated independently
from OpenAI; the same recipe applies to any OpenAI-compatible endpoint
(vLLM, LocalAI, LM Studio, OpenRouter-style gateways, etc.).

No NadirClaw code change is needed for this — everything below is configuration
you already control.

## Transport: environment variables

NadirClaw routes any `openai/<model>` ID through `NADIRCLAW_API_BASE` via
LiteLLM. Set the base URL and the per-tier model IDs:

```bash
NADIRCLAW_API_BASE=https://api.starveri.net/v1 \
NADIRCLAW_SIMPLE_MODEL=openai/gpt-5.3-codex-spark \
NADIRCLAW_MID_MODEL=openai/gpt-5.4-mini \
NADIRCLAW_COMPLEX_MODEL=openai/gpt-5.5 \
nadirclaw serve --verbose
```

Use the `openai/` prefix on every model name so LiteLLM treats them as
OpenAI-compatible. `NADIRCLAW_API_BASE` is passed to all non-Ollama,
non-Gemini LiteLLM calls. The `NADIRCLAW_MID_MODEL` tier is optional; if you
omit it, NadirClaw falls back to the simple model for the mid tier.

## Pricing & capabilities: `models.local.json`

So NadirClaw's cost tracker and savings report reflect the gateway's real
prices, add the models to your user-managed override file at
`~/.nadirclaw/models.local.json`. This file merges on top of the generated
`models.json` and is never overwritten by `nadirclaw update-models`, so your
overrides stay durable across registry refreshes.

The snippet below uses the USD-per-1M-token prices published on Starveri's
public `/models` endpoint at the time of writing. **Verify against
<https://api.starveri.net/models> before relying on these — third-party gateway
prices drift faster than this repo's release cadence.**

```json
{
  "models": {
    "openai/gpt-5.1-codex": {
      "cost_per_m_input": 0.4166667,
      "cost_per_m_output": 3.3333333,
      "has_vision": false
    },
    "openai/gpt-5.3-codex-spark": {
      "cost_per_m_input": 0.3333333,
      "cost_per_m_output": 0.6666667,
      "has_vision": false
    },
    "openai/gpt-5.3-codex": {
      "cost_per_m_input": 0.1666667,
      "cost_per_m_output": 0.3333333,
      "has_vision": false
    },
    "openai/gpt-5.4": {
      "cost_per_m_input": 0.3333333,
      "cost_per_m_output": 1.6666667,
      "has_vision": false
    },
    "openai/gpt-5.4-mini": {
      "cost_per_m_input": 0.25,
      "cost_per_m_output": 1.1666667,
      "has_vision": false
    },
    "openai/gpt-5.5": {
      "cost_per_m_input": 0.8333333,
      "cost_per_m_output": 2.5,
      "has_vision": false
    }
  }
}
```

Add a `context_window` field to any entry if you want the router to enforce a
specific input budget for that model.

## Verifying the setup

1. **Smoke-test each tier.** Run a request through NadirClaw with the env vars
   set and confirm all three tiers respond. If a tier misbehaves (refusals,
   truncation, format drift), note it — cheap tiers on third-party gateways
   vary in quality.
2. **Check the cost arithmetic.** Make one routed request and confirm the
   savings report picks up the local metadata (routed cost should reflect the
   prices above, not a fallback of $0).

## Caveats

- **Independent gateway.** Starveri is not OpenAI. Model IDs like `gpt-5.5`
  are the gateway's labels for whatever it serves behind them.
- **Prices drift; `/models` is the source of truth.** The numbers above are a
  point-in-time snapshot. Re-pull <https://api.starveri.net/models> periodically
  and update your `models.local.json`.
- **Cached-input and per-tool fees are not modeled.** NadirClaw's metadata
  schema currently has only `cost_per_m_input` and `cost_per_m_output`.
  Gateways that also bill cached-input tokens or per-tool-call surcharges
  (Starveri publishes both) will cost slightly more than NadirClaw's tracker
  reports. Treat the savings figure as a close estimate, not an invoice.
- **Not a permanent free provider.** Starveri offers a limited text-only demo,
  but paid usage is prepaid credits. Don't configure it expecting free routing.

## See also

- [Usage with Custom OpenAI-Compatible Endpoints](../../README.md#usage-with-custom-openai-compatible-endpoints)
  — the general recipe this page specializes.
- [Multi-provider routing](../multi-provider-routing.md) — routing across more
  than one vendor and the failure modes that introduces.
