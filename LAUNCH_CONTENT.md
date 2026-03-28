# Nadir Launch Content — Pro Waitlist

> Internal document. All content ready for publishing. Replace [LINK] placeholders before posting.

---

## 1. LinkedIn Launch Post (Founder Voice)

**We analyzed 50,000 LLM API calls across coding sessions. The result was uncomfortable.**

60-70% of prompts sent to models like Claude Sonnet or GPT-4 were simple tasks — reading files, formatting JSON, writing docstrings, asking short questions. Tasks that a model costing 10-20x less handles just as well.

That means most teams are paying premium prices for work that doesn't require premium intelligence.

We built Nadir to fix this.

Nadir is an open-source LLM router that sits between your AI tools and your providers. It classifies every prompt using ML-based complexity analysis (~10ms overhead) and routes simple requests to cheaper models automatically. Complex tasks still go to the best models available.

The results from our testing:

- Up to 40% cost reduction on LLM API spend
- 96% routing accuracy (validated against human labels)
- Zero perceived latency — classification happens in ~10ms
- Fully local — your API keys never leave your machine

It works as a drop-in proxy with Claude Code, Cursor, Codex, Aider, Windsurf, Continue, and anything that speaks the OpenAI API.

The core router is free and open-source (MIT). Today we're opening the waitlist for Nadir Pro — which adds hosted analytics, team dashboards, and advanced routing profiles. Pricing is savings-based: $9/mo base + a share of the savings we generate. You only pay more when we save you more.

If your team spends $5K+/mo on LLM APIs, this is worth 2 minutes of your time.

Join the Pro waitlist: https://getnadir.dev

GitHub: https://github.com/NadirRouter/NadirClaw

#LLM #AIEngineering #DeveloperTools #OpenSource #CostOptimization #AICoding #GenAI #DevTools

---

## 2. LinkedIn Short Post (Teaser / Day-Before)

**Tomorrow we're launching something we've been working on for months.**

Here's what we know after analyzing thousands of LLM API calls: most of them don't need an expensive model. The "format this JSON" request that costs $0.09 on a premium model? It costs $0.0004 on a lighter one. Same output.

Tomorrow we're opening the waitlist for Nadir Pro — an intelligent LLM router that catches this waste automatically.

Open-source core. Savings-based pricing. Works with the tools you already use.

More details tomorrow. If you want early access, drop a comment or DM me.

#AIEngineering #LLM #DevTools #OpenSource

---

## 3. X (Twitter) Launch Thread

**Tweet 1 (Hook):**
Your "format this JSON" prompt just cost $0.09 on Claude Sonnet.

The same request on Gemini Flash: $0.0004.

Same output. 225x price difference.

This is happening thousands of times a day in your AI workflow.

**Tweet 2 (What Nadir does):**
We built Nadir — an open-source LLM router that fixes this.

It sits between your AI tools and your providers. Every prompt gets classified by ML in ~10ms. Simple tasks route to cheaper models. Complex tasks still go to premium.

Your tools don't notice. Your wallet does.

**Tweet 3 (How it works):**
How it works:

1. pip install nadirclaw
2. nadirclaw setup
3. nadirclaw serve

It runs locally as an OpenAI-compatible proxy. Sentence embeddings classify prompt complexity. Simple/mid/complex tiers route to the right model.

Your API keys never leave your machine.

**Tweet 4 (The numbers):**
The numbers from real usage:

- 96% routing accuracy
- Up to 40% cost savings
- ~10ms classification overhead
- 60-70% of coding prompts are "simple"

That's $4,000 saved on every $10K of LLM spend. Every month.

**Tweet 5 (Pricing):**
Nadir pricing:

- Self-hosted (OSS): Free forever
- Pro ($9/mo + savings share): You only pay more when we save you more
- Enterprise: Custom

The open-source core does the routing. Pro adds analytics, team features, and managed infrastructure.

**Tweet 6 (Integrations):**
Works with:
- Claude Code
- Cursor
- Codex
- Aider
- Windsurf
- Continue
- Open WebUI
- Any OpenAI-compatible client

Two commands to set up. Drop-in proxy. No code changes needed.

**Tweet 7 (CTA):**
We're opening the Nadir Pro waitlist today.

GitHub (free, MIT): https://github.com/NadirRouter/NadirClaw

Pro waitlist: https://getnadir.dev

Star the repo if this is useful. We're building in the open.

---

## 4. X (Twitter) Single Launch Tweet

Most LLM calls don't need a premium model. Nadir routes them to cheaper ones automatically.

Open-source. 96% accuracy. Up to 40% savings. Works with Cursor, Claude Code, Codex, Aider.

Pro waitlist is live: https://getnadir.dev

---

## 5. Product Hunt Teaser Copy

**Tagline (under 60 chars):**
Stop overpaying for simple LLM requests

**Description (under 260 chars):**
Nadir is an open-source LLM router that classifies prompt complexity with ML and routes simple requests to cheaper models automatically. Works with Cursor, Claude Code, Codex, and any OpenAI-compatible tool. Up to 40% cost savings, ~10ms overhead.

**Topics/Tags:**
- Artificial Intelligence
- Developer Tools
- Open Source
- Cost Optimization
- Productivity

---

## 6. Hacker News "Show HN" Post

**Title:**
Show HN: Nadir -- open-source LLM router that cuts API costs 30-40% via ML classification

**Body:**

Hey HN,

I built Nadir because I noticed something while tracking my own LLM API usage: 60-70% of the prompts I was sending to Claude Sonnet and GPT-4 during coding sessions were trivially simple. Reading files, writing docstrings, formatting output, short factual questions. Tasks that models costing 10-20x less handle identically.

Nadir is a local proxy that sits between your AI tools (Cursor, Claude Code, Aider, etc.) and your LLM providers. It uses sentence embeddings to classify prompt complexity in ~10ms and routes simple requests to cheaper models. Complex requests still go to premium models. Your tools don't know the difference — Nadir speaks the OpenAI chat completions API.

Technical details:
- Classification uses sentence-transformers for embedding-based complexity scoring
- Three routing tiers (simple/mid/complex) with configurable thresholds
- Agentic task detection (tool use, multi-step loops) auto-routes to complex tier
- Fallback chains: if a model 429s or times out, cascades to alternatives
- Session pinning keeps multi-turn conversations on the same model
- Context-window filtering auto-swaps models when conversations get long
- In-memory LRU cache for identical completions
- Everything runs locally — your keys never leave your machine

From testing across real coding workflows, we see 96% routing accuracy against human labels and 30-40% cost savings (varies by usage pattern). Classification overhead is ~10ms per request.

Setup is `pip install nadirclaw && nadirclaw setup && nadirclaw serve`.

MIT licensed. The core is free. We're also launching a Pro tier ($9/mo + savings share) for teams that want hosted analytics and dashboards.

GitHub: https://github.com/NadirRouter/NadirClaw
Website: https://getnadir.dev

Happy to answer questions about the classification approach, routing logic, or anything else.

---

## 7. Email to Waitlist (Day of Launch)

### Subject Line Options

1. You're in: Nadir Pro early access is live
2. Your LLM costs are about to drop — Nadir Pro waitlist confirmed
3. Nadir Pro: stop paying premium prices for simple prompts

### Body

**Subject: You're in: Nadir Pro early access is live**

Hey [FIRST_NAME],

Thanks for signing up for the Nadir Pro waitlist. Here's what you need to know.

**What Nadir does**

Nadir is an LLM router. It sits between your AI coding tools (Cursor, Claude Code, Codex, Aider, etc.) and your LLM providers. It classifies every prompt using ML-based complexity analysis and routes simple requests to cheaper models automatically.

60-70% of prompts in typical coding sessions are simple — file reads, formatting, short questions. These don't need a $0.09 premium model call. Nadir catches them and routes them to models that cost $0.0002-$0.001.

Result: up to 40% savings on your LLM API spend with 96% routing accuracy and ~10ms overhead.

**How pricing works**

The open-source core is free forever. Self-host it, run it locally, modify it — it's MIT licensed.

Nadir Pro adds team analytics, hosted dashboards, advanced routing profiles, and priority support. It costs $9/month base plus a share of the savings we generate for you.

That means: if Nadir doesn't save you money, you don't pay beyond the base. Your interests and ours are aligned.

**What happens next**

1. We're onboarding waitlist members in batches over the coming weeks.
2. You'll get an email when your slot opens with setup instructions.
3. Setup takes about 2 minutes: install, configure your providers, start the proxy.

In the meantime, you can start using the open-source version right now:

```
pip install nadirclaw
nadirclaw setup
nadirclaw serve
```

GitHub: https://github.com/NadirRouter/NadirClaw
Docs: https://getnadir.dev/docs

**Have questions?**

Reply to this email. I read every one.

Best,
[FOUNDER_NAME]
Nadir

---

*All content prepared for launch. Review, adjust tone as needed, and replace placeholder links before publishing.*
