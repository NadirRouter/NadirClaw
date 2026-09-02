# Changelog

All notable changes to NadirClaw will be documented in this file.

## [Unreleased]

### Fixed
- **The prompt cache ignored every request parameter except the model and message text, so it could answer a request with a response that does not satisfy it** (#90). The key was `sha256(model + [role, content])`, while the request surface forwarded upstream also carries `tools`, `tool_choice`, `response_format`, `reasoning_effort`, `thinking`, `max_tokens`, `temperature`, `top_p`, `n`, and arbitrary provider extras. Two requests with the same messages therefore shared a cache entry: a plain chat answer could be served to a later request asking for `response_format: {"type": "json_schema"}` or for `tool_calls`, and a response generated under a high `max_tokens` could be served to a request that asked for a low one — silently overriding the client's contract, since the cache sits in front of the provider call. The key now includes every response-shaping field (`request_cache_params()` collects the declared ones plus everything in `model_extra` except `stream`), and message normalization keeps `tool_call_id` / `name` / `tool_calls` so tool-result turns with identical text no longer collide. Non-serializable extras fall back to `repr` rather than raising.

## [0.23.0] - 2026-08-31

### Added
- **`NADIRCLAW_PREFER_ENV_KEYS` — put provider env vars ahead of stored credentials.** The resolution chain runs OpenClaw stored token → NadirClaw stored token → env var, which is right on a laptop and backwards on a server: a stored personal OAuth or subscription token carries *that person's* rate limit, so every request the router proxies is billed and throttled against one human. On a machine that also runs Claude Code there was no way to override it short of deleting a credentials file the box still needs. Set the flag and the environment moves to the front, falling back to the stored credential when the relevant env var is unset. Default off, so existing installs are unchanged. `get_credential()` and `get_credential_source()` now share one `_env_credential()` implementation, so `nadirclaw status` cannot report `oauth` while the server is actually sending the env key.

### Fixed
- **`/v1/messages` forwarded parameters the routed model cannot accept, so cheap tiers 400 on every request** — clients pick request parameters from the model id they can see, which behind the router is an alias (`nadir-auto`), so they assume the newest capabilities. When routing then selected an older, cheaper model, the request carried parameters that model rejects and every request failed (#83). The passthrough now reconciles the request against the 400 it gets back and retries within the same request, bounded to 4 attempts: `thinking: {"type":"adaptive"}` is rewritten to `{"type":"enabled","budget_tokens":N}`, an unsupported `effort` hint is dropped, and `system`/`developer` turns are folded into the top-level `system` field (the same fix already applied to the OAuth completion path in 0.21.1). Each fix is remembered per model, so the discovery costs one round trip per model per process instead of one per request. Applied fixes are recorded in `requests.jsonl` as `modifiers_applied` entries and signalled by an `X-NadirClaw-Params-Reconciled` response header. Verified against a live Anthropic subscription: `claude-haiku-4-5` went from 3 failed upstream calls per Claude Code request to a single successful one.
- **`/v1/chat/completions` had the same 400 on its Anthropic OAuth path** — the reconcile above only covered `/v1/messages`, so OpenClaw and Codex users routed to an older model still hit `adaptive thinking is not supported on this model` (#83). The direct Anthropic call behind the OpenAI-compatible endpoint now runs the same bounded reconcile loop and shares the per-model cache, and the fixes it applies are recorded as `modifiers_applied` on the request log. Two smaller corrections to the same mechanism: a reconciler that matched a 400 but declined to rewrite anything is no longer cached against the model, and the final attempt of the loop no longer applies a fix it has no round trip left to send.
- **Clamping `max_tokens` could strand `thinking.budget_tokens` above it, turning one recoverable 400 into an unrecoverable one** — Anthropic requires `1024 <= budget_tokens < max_tokens`, but the reconcile loop lowered `max_tokens` to the routed model's ceiling (#73) without revisiting a thinking budget the client had sized for the ceiling it thought it had. A request with `max_tokens: 100000` and `budget_tokens: 60000` routed to a model capped lower was clamped into a body that 400s on `budget_tokens`, an error no reconciler matches, so the loop gave up and surfaced the failure. The clamp now lowers the budget to fit under the new ceiling, or drops `thinking` outright when no value clears Anthropic's 1024 floor. Reachable both from a client-supplied budget and from the adaptive downgrade above, whose budget is sized before a later attempt clamps `max_tokens`.
- **`serve` could not start without a tty.** On first run it asked "No configuration found. Run setup wizard?" unconditionally. With no tty `click.confirm` raises `Abort` and the process exits 1, so the shipped `Dockerfile` (`CMD ["nadirclaw", "serve", "--host", "0.0.0.0"]`) could never boot a fresh container, and the same failure hit systemd units and CI. The prompt is now gated on `sys.stdin.isatty()`; headless starts fall through to the existing "Starting with defaults" path.
- **Python 3.10 installs were broken by litellm 1.98.0.** That release imports `NotRequired` from `typing` in `llms/anthropic/experimental_pass_through/context_management/editors/compact.py`, but `typing.NotRequired` only exists on 3.11+, so `import nadirclaw` raised `ImportError` on every fresh 3.10 install — while litellm still declares `Requires-Python: >=3.10`. The dependency is now capped below 1.98.0 for Python < 3.11 only; 3.11 and 3.12 continue to track the latest litellm. The cap comes off once upstream fixes the import.

### Internal
- **CI ran no server tests.** `.github/workflows/ci.yml` had excluded `tests/test_server.py` since the workflow was first added, so every test covering `/v1/messages`, `/v1/chat/completions`, routing headers, request logging, and the parameter reconcile above was skipped on every push and pull request. The exclusion is removed; the file passes on 3.10–3.12.

## [0.22.0] - 2026-07-20

### Changed
- **License changed from MIT to the PolyForm Noncommercial License 1.0.0.** NadirClaw is now free for any noncommercial use (personal, research, education, evaluation, and noncommercial organizations). Commercial use requires a separate commercial license, available via [getnadir.com](https://getnadir.com). This applies to new versions; releases previously published under MIT remain available under MIT. The bundled `wide_deep_asym_v3` classifier weights and the `cascade-verifier-v1` snapshot are now released under the same noncommercial terms.

### Added
- **`v3`+complex-gate is the default wide-deep routing.** The bundled wide-deep classifier now defaults to the `v3` checkpoint under a Neyman-Pearson complex gate (`complex_gate_v1.pt`, τ=0.12) with a head medium/simple split, matching the routing that runs in Nadir Pro. Unlike `asym`, `v3` keeps a real `P(simple)`, and the head split under-routes less than the legacy companion LR. The gate is on by default for `v3` and off for `asym`/`symmetric`, so legacy behaviour is intact. Tunable via `NADIR_COMPLEX_GATE=0`, `NADIR_GATE_THRESHOLD`, and `NADIR_MS_SPLIT=companion`. Measured on RouterArena (n=2,479): 8.2% miss-complex, ~41% cost reduction vs always-premium. NadirClaw's global default classifier stays `binary`; see MODEL_CARD.md.

## [0.21.1] - 2026-06-25

### Added
- **Opt-in Claude Code identity injection for OAuth tokens** (`NADIRCLAW_CLAUDE_CODE_IDENTITY=1`) — Anthropic gates premium models (Sonnet/Opus) behind subscription/OAuth tokens (`sk-ant-oat*`) unless the request leads with the official Claude Code identity system block. The real Claude Code client always sends it; raw API/SDK callers omit it and get a bare `rate_limit_error` on those models while Haiku works (#74). When enabled, `/v1/messages` and the OAuth completion path prepend `"You are Claude Code, Anthropic's official CLI for Claude."` as the first `system` block — only for Bearer/OAuth tokens (no effect on `sk-ant-api*` keys), only when not already present, preserving any caller-supplied system prompt after it. The decision is recorded as `claude_code_identity` on the request log. Default off, since it changes the system prompt the model sees.

### Fixed
- **OAuth completion path sent `system` turns as chat messages** — the direct Anthropic OAuth call in `/v1/chat/completions` forwarded `role: "system"` messages inside the `messages` array, which Anthropic's `/v1/messages` API rejects (system must be a top-level field). System/developer turns are now collected into the top-level `system` field before forwarding (#74).

## [0.21.0] - 2026-06-24

### Added
- **`/v1/messages/count_tokens` endpoint** — Anthropic-native clients (Claude Code, the official `anthropic` SDK) call `count_tokens` to size requests before sending; this previously 404'd, so clients silently fell back to approximate local token estimation. The new route resolves the model through the same router as `/v1/messages` and forwards to Anthropic's real `count_tokens`, returning `{"input_tokens": N}` verbatim. Non-billable, so it is excluded from cost/budget recording (#72).

### Fixed
- **`/v1/messages` traffic was invisible to metrics and budget** — `record_request` short-circuited on `type != "completion"`, dropping every `/v1/messages` log entry (`type="messages"`). Since Claude Code and the Anthropic SDKs talk to `/v1/messages`, all of that traffic was missing from Prometheus counters and the budget tracker. The recorder now accepts `messages` entries, and the `/v1/messages` handler computes cost via the budget tracker and stamps status/latency/token counts on both the streaming (usage recovered from the SSE `message_start`/`message_delta` events) and non-streaming paths (#71).
- **`max_tokens` was not reconciled against the routed model's output ceiling** — when the router rewrote `model` to a tier whose max-output is lower than the client-supplied `max_tokens`, Anthropic returned an intermittent 400 that was proxied straight back. `/v1/messages` now detects a `max_tokens: N > M` 400, clamps `max_tokens` to the reported ceiling `M`, and retries once (both streaming and non-streaming), tagging the non-streaming response with `X-NadirClaw-MaxTokens-Clamped: true` (#73).
- **`nadirclaw test` failed for Claude subscription tokens** — the command called `litellm.completion()` directly, which sends `sk-ant-oat*` subscription/OAuth tokens as `x-api-key` instead of `Authorization: Bearer` + the `oauth-2025-04-20` beta header the server uses. It now probes Anthropic models through the same OAuth path as the running server, so `nadirclaw test` reflects real server behavior (#74).

## [0.20.0] - 2026-06-12

### Added
- **Context-optimizer compression upgrades** (#65):
  - **Pluggable backend** — `NADIRCLAW_OPTIMIZE_BACKEND` selects `native` (default, built-in stdlib pipeline) or `headroom` (opt-in, delegates to the Apache-2.0 [`headroom-ai`](https://pypi.org/project/headroom-ai/) package via `pip install nadirclaw[headroom]`). Headroom is lazy and **fail-open**: if it is not installed or raises, the optimizer transparently falls back to `native` and the request never fails. Per-request override via `optimize_backend` in the body.
  - **Progressive (staged) compression** — `--optimize progressive` / `NADIRCLAW_OPTIMIZE=progressive` runs an escalation ladder (`native_safe → native_aggressive → headroom_structural → headroom_ml`) that **stops as soon as `NADIRCLAW_OPTIMIZE_TARGET_TOKENS` is met**. With no budget set it stops after `native_aggressive` (dependency-free, lossless); Headroom stages are skipped silently when `headroom-ai` is absent; the lossy ML stage runs only when `NADIRCLAW_OPTIMIZE_ALLOW_LOSSY` is on. Tunable via `NADIRCLAW_OPTIMIZE_MAX_STAGE`. New library entrypoint `nadirclaw.optimize.compress_progressive()`.
  - **Columnar JSON-array packing** (`json_array_pack`, aggressive mode) — rewrites homogeneous arrays of same-keyed objects (DB results, API list responses, large tool outputs) into a header plus one value-array per row, emitting each key once. Information-lossless and deterministically reversible; ~68% vs pretty-printed JSON. Never runs in `safe` mode.
  - **Native CCR** (`nadirclaw/ccr.py`) — deterministic offload + `nadir_retrieve` fetch-back loop that moves oversized content out of the prompt behind a retrieve handle, fully reversible because the originals are kept server-side. Library-only for now (not yet wired into `nadirclaw serve`).
  - Apache-2.0 attribution for `headroom-ai` in `THIRD_PARTY_NOTICES.md`; docs, benchmarks, and tests (ccr, progressive, json_array_pack, backends, code-safety).

### Fixed
- **Whitespace normalization corrupted unfenced source code** — the `whitespace_normalize` transform collapsed the *leading indentation* of raw (unfenced) code arriving as file-read tool outputs, flattening nested Python/YAML/diffs into invalid syntax while reporting it as "savings". It now preserves leading indentation and only collapses interior multi-spaces, in both `safe` and `aggressive` modes (#65).

## [0.19.3] - 2026-06-12

### Fixed
- **OpenAI OAuth login failed with `authorize_hydra_invalid_request`** — `nadirclaw auth openai login` sent `redirect_uri=http://127.0.0.1:1455/auth/callback` in the authorize request while the callback server bound and printed `localhost`. OpenAI's Hydra authorization server exact-matches `redirect_uri` against the client allow-list and rejected the `127.0.0.1` variant before the login screen. Now uses `localhost` consistently, matching the callback server and the Antigravity/Gemini flows (#67, #69).

## [0.18.0] - 2026-05-25

### Added
- **Application Default Credentials (ADC) for Gemini** — when no `GOOGLE_API_KEY` is set, the Gemini path now falls back to `google.auth.default()` so users can authenticate via `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` (Vertex AI / gcloud-managed creds) instead of pasting a key. Original work by @froody (#57).
- **`nadirclaw status` displays the mid-tier model** when one is configured, alongside simple/complex (#57).

### Fixed
- **Gemini streaming was broken** — `_dispatch_model_stream` consumed `_stream_gemini` (an async generator) with a plain `for` loop, which would raise `TypeError: 'async_generator' object is not iterable` on any actual streaming Gemini call. Now uses `async for`, and chunk / `finish_reason` parsing is robust to the google-genai SDK returning enum-like objects (#57).
- **`savings` no longer crashes on `None` values** in the request log for `selected_model` and `tier` — these show up for failed / aborted requests and previously broke the report (#57).

## [0.17.0] - 2026-05-15

### Added
- **Configurable embedding backends for the centroid classifier** — `NADIRCLAW_EMBEDDING_BACKEND` (default `sentence-transformers`; also `ollama` via `/api/embed`), `NADIRCLAW_EMBEDDING_MODEL`, `NADIRCLAW_EMBEDDING_API_BASE`, and `NADIRCLAW_CENTROID_DIR`. Custom centroid directories require a `centroid_metadata.json` (schema-versioned, with `prototypes_hash` for traceability) so users never silently mismatch a self-built centroid against a different encoder. `nadirclaw build-centroids` gains `--backend`, `--model`, `--api-base`, `--output-dir` flags. Original work by @clawSean (#50).
- **Optional prompt-injection guard** — `nadirclaw/prompt_guard.py`. Heuristic detection of 7 patterns (instruction override, role reassignment, prompt extraction, JSON role confusion, delimiter injection, encoded payloads, DAN/jailbreak). `NADIRCLAW_PROMPT_GUARD`: `log` (default) / `warn` / `block`. Scans only user/tool messages — system/assistant treated as trusted. Original work by @pradumna-gautam (#55, supersedes #31).
- **Optional PII redactor** — `nadirclaw/pii_redactor.py`. Detects email, US phone, SSN, and Luhn-validated credit-card numbers. `NADIRCLAW_PII_REDACTION`: `none` (default) / `log_only` / `redact`. Non-streaming responses only. Original work by @pradumna-gautam (#55).

### Security
- **Production hardening baseline** — recommended for anyone exposing `nadirclaw serve` beyond localhost. Original work by @pradumna-gautam (#30).
  - **CORS**: explicit allowlist via `NADIRCLAW_CORS_ORIGINS`; localhost regex default; never wildcard + credentials.
  - **Auth**: constant-time token comparison via `hmac.compare_digest` to defeat timing-side-channel guessing.
  - **Security headers** on every response: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, `Cache-Control: no-store` on `/v1/*`, opt-in HSTS via `NADIRCLAW_HSTS=true`.
  - **Bounds validation** on `ChatCompletionRequest`: caps on messages (500), `max_tokens` (100K), `temperature` (0–2), `top_p` (0–1), `n` (1–8) — closes a cost-amplification surface.
  - **Sanitized validation errors** — Pydantic internals no longer leak to clients; full details still server-side logged.
  - **Async logging** — SQLite writes moved off the event loop into a `ThreadPoolExecutor`, with `done_callback` exception logging and `shutdown(wait=True)` on SIGTERM so queued entries drain instead of dropping.
  - **Prompt truncation** — 500-char default in SQLite request logs (configurable via `NADIRCLAW_LOG_PROMPT_TRUNCATE`); API-key shaped tokens (`sk-…`, `AIza…`, `ghp_…`, `gho_…`, `xox[bpars]-…`) redacted from logged system prompts.

## [0.16.0] - 2026-05-14

### Added
- **Anthropic-compatible `/v1/messages` endpoint** — Anthropic-native clients (Claude Code) now route through NadirClaw. The proxy classifies, rewrites the `model` field, forwards to `api.anthropic.com`, and pipes SSE streaming through byte-for-byte (#51).
- **Seamless Claude Code integration** — `nadirclaw claude onboard` / `shim` / `uninstall`. Onboarding detects models, maps them into tiers, persists `ANTHROPIC_BASE_URL` + `ANTHROPIC_MODEL` into `~/.claude/settings.json`, and installs a launchd / systemd auto-start unit (#51).
- **Live model detection** — onboarding queries Anthropic's `/v1/models` using the stored token (Bearer for subscription tokens, `x-api-key` for API keys) instead of a hardcoded list; `--interactive` lets you pick a model per tier (#51).
- **Pluggable complexity classifier** — `NADIRCLAW_COMPLEXITY_ANALYZER=binary` (default, ~10ms centroid) or `distilbert` (3-class fine-tuned DistilBERT predicting simple/mid/complex natively). The DistilBERT artifact downloads from the Hugging Face Hub on first use with a graceful fallback to binary (#51, #52).
- **Pro upsell surfaces** — `nadirclaw savings` / `serve` / `report` and the README now surface Nadir Pro at high-intent moments with attribution-tagged URLs; new `demo/cost_vs_opus.py` zero-API-key demo (#53).
- **Enriched `/v1/models`** — responses now include Anthropic-style `type` / `display_name` / `description` / `created_at` alongside the OpenAI-style fields.

### Fixed
- `ANTHROPIC_BASE_URL` is written as the bare host (Claude Code appends `/v1/messages` itself; a `/v1` suffix produced a broken `/v1/v1/messages` path) (#51).
- Updated the stale Claude model fallback list from the 4.5/4.1 generation to the 4.6 family (#51).

## [0.15.0] - 2026-05-09

### Added
- **`nadirclaw update-models` command** — writes refreshable model metadata to `~/.nadirclaw/models.json`, optionally merging a published registry JSON via `--source-url` or `NADIRCLAW_MODEL_REGISTRY_URL`.
- **Local model metadata overrides** — the router now merges `~/.nadirclaw/models.json` and user-managed `~/.nadirclaw/models.local.json` into the runtime model registry.
- **DeepSeek V4 explicit aliases** — added `deepseek-v4`, `deepseek-v4-flash`, and `deepseek-v4-pro` while preserving the existing `deepseek` alias for `deepseek/deepseek-chat`.
- **Model pool weighted load balancing** — pool tier configuration with weighted round-robin across multiple models in the same tier (#36).
- **Selective context compression module** — opt-in compression for tool-heavy contexts (#40).
- **Complex coding detection and enhanced reasoning markers** — improved tier classification for coding-heavy prompts and Chinese reasoning markers (#38).
- **Upgrade-only session cache for agent frameworks** — caches routing decisions per session to avoid repeated downgrades on multi-turn agent flows (#27).
- **Agent role detection for AI coding assistants** — recognizes Claude Code / Cursor-style system prompts and routes accordingly (#37/#45).
- **Fallback reasons logging** — failed fallback attempts now record ordered per-model `fallback_reasons` with compact error types and sanitized messages (#47).
- **Provider health-aware fallback routing** — optional `NADIRCLAW_PROVIDER_HEALTH=true` mode tracks in-process model health and tries healthy fallback candidates before cooling-down ones; debug snapshot via `/internal/provider_health` (#48).

## [0.14.0] - 2026-04-03

### Added
- **Thinking/reasoning token passthrough** — transparently forwards thinking parameters and extracts reasoning content from all provider paths:
  - **Request forwarding**: `reasoning_effort` (OpenAI o-series), `thinking` (Anthropic extended thinking), `thinking_config` (Gemini), and `response_format` are now passed through to LiteLLM, Anthropic OAuth, and Gemini native paths.
  - **Response extraction**: `reasoning_content` (DeepSeek), `thinking` blocks (Anthropic), and `thought` parts (Gemini) are captured from LLM responses and included in `choices[].message`.
  - **Usage reporting**: `completion_tokens_details.reasoning_tokens` surfaced when providers report thinking token counts.
  - Works in both streaming (real SSE and fake/cached SSE) and non-streaming response formats.
- 15 new tests covering thinking parameter forwarding, response extraction, JSON serialization safety, and streaming passthrough.

## [0.13.0] - 2026-03-20

### Added
- **Context Optimize** — new preprocessing stage that compacts bloated context before LLM dispatch, reducing input token cost by 30-70%. Two modes:
  - **`safe`** — five deterministic, lossless transforms: JSON minification, whitespace normalization, system prompt dedup, tool schema dedup, chat history trimming.
  - **`aggressive`** — all safe transforms + diff-preserving semantic deduplication. Uses sentence embeddings (`all-MiniLM-L6-v2`) to detect near-duplicate messages (cosine similarity >= 0.85), then extracts only the unique diff phrases using `difflib.SequenceMatcher`. Refinements survive dedup — "return values, not indices" is preserved even when 90% similar to an earlier message.
- **Accurate token counting with tiktoken** — uses `cl100k_base` BPE tokenizer instead of `len//4` heuristic. Falls back gracefully if tiktoken is not installed.
- **Shared sentence encoder** — lazy-loaded `SentenceTransformer` singleton in `nadirclaw/encoder.py` for aggressive mode. No import cost when using safe mode or off.
- **`nadirclaw optimize` command** — dry-run CLI tool to test context compaction on files or stdin. Supports `--mode safe|aggressive` and `--format text|json`.
- **`--optimize` flag on `nadirclaw serve`** — set optimization mode at startup (`off`, `safe`, `aggressive`).
- **Per-request `optimize` override** — pass `"optimize": "safe"` in the request body to override the server default for individual requests.
- **Optimization metrics** — `tokens_saved`, `original_tokens`, `optimized_tokens`, and `optimizations_applied` logged per request in JSONL, SQLite, and Prometheus. Web dashboard shows aggregate savings.
- New env vars: `NADIRCLAW_OPTIMIZE` (default: `off`), `NADIRCLAW_OPTIMIZE_MAX_TURNS` (default: `40`).
- 60 automated tests covering safe transforms, aggressive semantic dedup, accuracy preservation, edge cases, and roundtrip integrity.

### Changed
- SQLite schema: added columns `optimization_mode`, `original_tokens`, `optimized_tokens`, `tokens_saved`, `optimizations_applied` (auto-migrated on startup).

## [0.7.0] - 2026-03-02

### Added
- **`nadirclaw test` command** — probes each configured model tier with a short live request and reports latency, response, and pass/fail. Exits with code 1 on failure so it works in CI. Supports `--simple-model`, `--complex-model`, and `--timeout` overrides.
- **`classify --format json`** — new `--format text|json` flag on `nadirclaw classify`. JSON output includes `tier`, `is_complex`, `confidence`, `score`, `model`, and `prompt`. Composable with `jq`.
- **Multi-word prompt support for `classify`** — `nadirclaw classify What is 2+2?` now works without quoting. Previously only the first word was captured.

### Changed
- **`nadirclaw savings` now prefers SQLite** — mirrors `nadirclaw report`: reads from `requests.db` when available, falls back to `requests.jsonl`. Previously only JSONL was read, giving empty or stale results for users without a JSONL file.
- **`nadirclaw dashboard` now prefers SQLite** — same fix as savings; dashboard no longer shows empty data when only `requests.db` exists.
- **`SessionCache` LRU eviction is now O(1)** — replaced `List[str]` + `list.remove()` (O(n) per cache hit) with `collections.OrderedDict` + `move_to_end()` / `popitem(last=False)`, both O(1). Affects `routing.py`.
- **`ModelRateLimiter.get_status` is now thread-safe** — all reads of `_limits`, `_hits`, and `_default_rpm` are now taken inside the lock, eliminating a potential data race under concurrent requests.

### Fixed
- **`auth status` indentation** — the "no credentials" help block was over-indented (12 spaces) and the provider hint strings were misaligned. Fixed to consistent 4-space indentation.
- **Removed redundant `load_dotenv()` in `serve`** — `settings.py` already loads `~/.nadirclaw/.env` at import time; the extra bare `load_dotenv()` call in the `serve` command was a no-op that could cause confusion when debugging env resolution.

## [0.6.1] - 2026-02-28

### Fixed
- OpenClaw onboard: register nadirclaw provider without overriding the agent's primary model

## [0.6.0] - 2026-02-26

### Added
- **Configurable fallback chains** — when a model fails (429, 5xx, timeout), cascade through a configurable list of fallback models. Set `NADIRCLAW_FALLBACK_CHAIN` to customize the order.
- **Real-time spend tracking and budget alerts** — every request's cost is tracked by model, daily, and monthly. Set `NADIRCLAW_DAILY_BUDGET` and `NADIRCLAW_MONTHLY_BUDGET` for alerts at configurable thresholds. New `nadirclaw budget` CLI command and `/v1/budget` API endpoint.
- **Prompt caching** — LRU cache for identical prompts. Configurable TTL (`NADIRCLAW_CACHE_TTL`, default 5min) and max size (`NADIRCLAW_CACHE_MAX_SIZE`, default 1000). New `nadirclaw cache` CLI command and `/v1/cache` API endpoint. Toggle with `NADIRCLAW_CACHE_ENABLED`.
- **Web dashboard** — browser-based dashboard at `/dashboard` with auto-refresh. Shows routing distribution, per-model stats, cost tracking, budget status, and recent requests. Dark theme, zero dependencies.
- **Docker support** — official Dockerfile and docker-compose.yml. `docker compose up` gives you NadirClaw + Ollama for a fully local zero-cost setup.

### Changed
- Fallback logic upgraded from simple tier-swap to full chain cascade
- Request logs now include per-request cost and daily spend
- Budget state persists across restarts via `budget_state.json`

## [0.3.0] - 2025-02-14

### Added
- OAuth login for all major providers: OpenAI, Anthropic, Google Gemini, Google Antigravity
- Interactive Anthropic login — choose between setup token or API key
- Gemini OAuth PKCE flow with browser-based authorization
- Antigravity OAuth with hardcoded public client credentials (matching OpenClaw)
- Provider-specific token refresh (OpenAI, Anthropic, Gemini, Antigravity)
- Atomic credential file writes to prevent corruption
- Port-in-use error handling for OAuth callback server
- Test suite with pytest (credentials, OAuth, classifier, server)
- CONTRIBUTING.md and CHANGELOG.md

### Changed
- Version is now single source of truth in `nadirclaw/__init__.py`
- Credential file writes use atomic temp-file-and-rename pattern
- Token refresh failures return `None` instead of silently returning stale tokens
- OAuth callback server binds to `localhost` (was `127.0.0.1`)

### Fixed
- Version mismatch between `__init__.py`, `cli.py`, `server.py`, and `pyproject.toml`
- README references to `nadirclaw auth gemini-cli` (now `nadirclaw auth gemini`)
- OAuth callback server getting stuck (now uses `serve_forever()`)

## [0.2.0] - 2025-01-20

### Added
- OpenAI OAuth login via Codex CLI
- Credential storage in `~/.nadirclaw/credentials.json`
- Environment variable fallback for API keys
- `nadirclaw auth` command group

## [0.1.0] - 2025-01-10

### Added
- Initial release
- Binary complexity classifier with sentence embeddings
- Smart routing between simple and complex models
- OpenAI-compatible API (`/v1/chat/completions`)
- SSE streaming support
- Rate limit fallback between tiers
- Gemini native SDK integration
- LiteLLM support for 100+ providers
- CLI: `serve`, `classify`, `status`, `build-centroids`
- OpenClaw and Codex onboarding commands
