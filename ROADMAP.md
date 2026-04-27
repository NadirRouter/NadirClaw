# NadirClaw Roadmap

> **Current version:** v0.10.0 (March 2026) · **Window:** March – June 2026

This is a near-term, concrete roadmap — not a vision doc. Items are grounded in real gaps in the
codebase today. Dates are targets, not guarantees. Check the [CHANGELOG](CHANGELOG.md) for what
has already shipped.

---

## v0.8.0 — Routing & Resilience _(~2–3 weeks)_

- [x] **Multi-tier routing** — added a `mid` tier between `simple` and `complex`; configurable
      score thresholds via `NADIRCLAW_TIER_THRESHOLDS` so users can tune buckets without code changes
- [ ] **Provider health-aware routing** — track rolling error rates per provider (429 / 5xx /
      timeout) and downgrade to the next healthy option automatically; expose health scores in
      `nadirclaw status`
- [x] **`nadirclaw update-models` command** — writes local model metadata to
      `~/.nadirclaw/models.json`, with `models.local.json` support for user overrides

---

## v0.8.1 — Caching & Performance _(~2 weeks)_

- [ ] **Persistent cache** — opt-in SQLite-backed prompt cache that survives restarts
      (proposed: `NADIRCLAW_CACHE_BACKEND=sqlite`); existing in-memory LRU remains the default
- [ ] **Embedding deduplication** — skip recomputing sentence embeddings for prompts seen in the
      last N minutes (configurable); reduces classifier latency on repeated queries
- [x] **Lazy-load sentence transformer** — deferred model load until the first classify call; cuts
      cold-start time for users who run `nadirclaw serve` and immediately send a request

---

## v0.9.0 — Analytics & Insights _(~4 weeks)_

- [x] **Per-model cost breakdown** — `nadirclaw report --by-model --by-day` with anomaly
      flagging when a model's spend spikes more than 2× its 7-day average
- [x] **Log export** — `nadirclaw export --format csv|jsonl --since 7d` for offline analysis
- [ ] **Routing feedback loop** — `nadirclaw flag <request-id> --reason misrouted` writes a
      correction record that future centroid training can consume
- [ ] **Grafana dashboard JSON** — pre-built dashboard definition for the existing Prometheus
      `/metrics` endpoint; documented setup in `docs/grafana.md`

---

## v0.9.1 — Ecosystem Expansion _(~3 weeks)_

- [x] **Open WebUI integration** — `nadirclaw openwebui onboard` with setup instructions;
      `/v1/models` now returns routing profiles (`auto`, `eco`, `premium`) for auto-discovery
- [x] **Editor onboard commands** — `nadirclaw continue onboard` and `nadirclaw cursor onboard`
      for [Continue](https://continue.dev) and [Cursor](https://cursor.sh); mirrors the existing
      `openclaw` and `codex` onboard pattern
- [ ] **OpenRouter-compatible passthrough mode** — accept OpenRouter-format requests
      (`openrouter/` model prefixes) and forward through NadirClaw's routing layer
- [ ] **GitHub Action improvements** — add caching for repeated classifier calls, step-summary
      output, and PR annotation support for cost / routing results

---

## v1.0.0 — Stability & GA _(end of 3-month window)_

- [ ] **Stable API contract** — document and freeze `/v1/*` endpoint shapes; no breaking changes
      after 1.0 without a major version bump
- [ ] **Custom classifier training** — `nadirclaw train --data prompts.jsonl` rebuilds centroids
      from your own labelled data; makes the classifier adapt to domain-specific prompt patterns
- [ ] **Distributed rate limiting** — optional Redis backend
      (proposed: `NADIRCLAW_RATE_LIMIT_BACKEND=redis`) for multi-instance deployments sharing a single
      rate-limit state
- [ ] **Documentation site** — MkDocs (or similar) generated from `docs/`; published via GitHub
      Pages; covers installation, configuration, integrations, and the HTTP API
- [ ] **End-to-end integration test suite** — covers the full request path: classify → route →
      provider call → log; runnable in CI without real API keys via recorded fixtures

---

## Always-on

These happen continuously and are not tied to a milestone:

- **Weekly patch releases** — bug fixes, dependency updates, security patches
- **Provider & pricing updates** — new models, revised token costs, updated context windows

---

## How to Contribute

We welcome PRs for any item above. Before starting on a larger feature, open a GitHub Issue to
discuss the approach — it saves time for everyone.

- See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, testing, and code-style guidelines
- Use [GitHub Discussions] for questions and feature requests
- Use [GitHub Issues] for bugs and tracked work items

If you pick up a roadmap item, comment on the relevant issue so others know it is in progress.
To propose a new integration or feature, open a [GitHub Discussion] first.

[GitHub Discussions]: https://github.com/doramirdor/NadirClaw/discussions
[GitHub Issues]: https://github.com/doramirdor/NadirClaw/issues

---

_Licensed under the [MIT License](LICENSE)._
