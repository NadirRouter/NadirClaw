# Routing all three NadirClaw tiers through one gateway key (UberLLM example)

This page is a worked example of pointing every NadirClaw tier at a single
third-party OpenAI-compatible gateway, using [UberLLM](https://uberllm.dev) as
the concrete case. It is **not an endorsement**. UberLLM is operated
independently from the model labs it serves. The general recipe is the same as
in the [Starveri example](starveri.md); what this page adds is a tier split
where the simple, mid and complex models come from different labs (Qwen,
DeepSeek, Anthropic) but share one base URL and one key.

No NadirClaw code change is needed for this — everything below is configuration
you already control.

## Transport: environment variables

```bash
OPENAI_API_KEY=<your UberLLM key> \
NADIRCLAW_API_BASE=https://api.uberllm.dev/v1 \
NADIRCLAW_SIMPLE_MODEL=openai/qwen/qwen3.7-flash \
NADIRCLAW_MID_MODEL=openai/deepseek/deepseek-v4.1-flash \
NADIRCLAW_COMPLEX_MODEL=openai/anthropic/claude-sonnet-4.6 \
nadirclaw serve --verbose
```

UberLLM model IDs already contain a vendor segment (`deepseek/deepseek-v4.1-flash`),
so keep NadirClaw's `openai/` prefix in front of it: LiteLLM strips the first
`openai/` and sends `deepseek/deepseek-v4.1-flash` to the gateway. The key goes
in `OPENAI_API_KEY` (or `nadirclaw auth add --provider openai`) because these are `openai/` models.
The public model list (no key needed) is at
<https://api.uberllm.dev/v1/models>; swap any tier for another ID from it.

## Pricing & capabilities: `models.local.json`

So the cost tracker and savings report use the gateway's prices, add the three
models to `~/.nadirclaw/models.local.json`. The numbers below are the USD per
1M token prices that `https://api.uberllm.dev/v1/models` published on
2026-11-12. **They are a snapshot: gateway prices drift faster than this
repo's release cadence, so re-check <https://uberllm.dev/models> before relying
on them.**

```json
{
  "models": {
    "openai/qwen/qwen3.7-flash": {
      "cost_per_m_input": 0.0108,
      "cost_per_m_output": 0.0468,
      "has_vision": true
    },
    "openai/deepseek/deepseek-v4.1-flash": {
      "cost_per_m_input": 0.00594,
      "cost_per_m_output": 0.108,
      "has_vision": true
    },
    "openai/anthropic/claude-sonnet-4.6": {
      "cost_per_m_input": 1.26,
      "cost_per_m_output": 6.3,
      "has_vision": true
    }
  }
}
```

## Verifying the setup

1. **`nadirclaw test` per tier.** `nadirclaw test` calls LiteLLM directly and
   does not read `NADIRCLAW_API_BASE`, so give it the base URL through LiteLLM's
   own variable for that run. Without it the probe goes to `api.openai.com`
   with your gateway key.

   ```bash
   export OPENAI_API_BASE=https://api.uberllm.dev/v1
   nadirclaw test                       # simple + complex tiers
   nadirclaw test --simple-model openai/deepseek/deepseek-v4.1-flash \
                  --complex-model openai/deepseek/deepseek-v4.1-flash   # mid tier
   ```

   On 2026-11-12 all three tiers answered `ok`. The simple tier
   (`qwen/qwen3.7-flash`) is a reasoning model: a "2+2" prompt used 171
   reasoning tokens out of 175 completion tokens, so it spends more completion
   tokens than `max_tokens` or the visible answer suggests.
2. **One routed request with a non-zero cost.** Start `nadirclaw serve` with
   the env vars above and the `models.local.json` snippet in place, send one
   request with `"model": "auto"`, then run `nadirclaw savings --since 24h`.
   Each routed model should show a cost from the prices above, not a fallback
   of $0. (In a test run a simple and a complex prompt were routed to the simple
   and complex tiers and both showed up with their own cost.)
3. **Check the balance.** `GET https://api.uberllm.dev/v1/balance` with the same
   key returns `balance_usd`, `held_usd` and `available_usd`, which you can
   compare against NadirClaw's own cost report.

## Caveats

- **Independent gateway.** UberLLM is not Qwen, DeepSeek or Anthropic; the IDs
  are the gateway's labels for the models it serves.
- **Prices drift; `/v1/models` is the source of truth.** Re-pull it and update
  `models.local.json` from time to time.
- **Cached-input fees are not modeled.** The gateway also publishes cached-input
  prices; NadirClaw's metadata schema only has `cost_per_m_input` and
  `cost_per_m_output`, so treat the savings figure as a close estimate, not an
  invoice.
- **Prepaid, not free.** Usage is billed from prepaid credit; failed requests
  are not billed.

## See also

- [Usage with Custom OpenAI-Compatible Endpoints](../../README.md#usage-with-custom-openai-compatible-endpoints)
  — the general recipe this page specializes.
- [Starveri example](starveri.md) — the same recipe with a single-lab tier split.
- [Multi-provider routing](../multi-provider-routing.md) — routing across more
  than one vendor and the failure modes that introduces.
