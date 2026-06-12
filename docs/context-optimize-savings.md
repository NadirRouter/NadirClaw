# Context Optimize — Savings Analysis

## Summary

NadirClaw's Context Optimize compacts bloated context (JSON, tool schemas, chat history, whitespace) before sending to the LLM provider. All transforms are **lossless** — zero semantic degradation.

Combined with smart routing, NadirClaw now saves in two ways:
1. **Route** simpler work to cheaper models
2. **Compact** bloated context before it hits your bill

## Benchmark: Claude Opus 4.6

**Pricing:** $15/1M input tokens, $75/1M output tokens

| Scenario | Before | After | Saved | % | Saved / 1K req |
|---|---:|---:|---:|---:|---:|
| Agentic coding assistant (8 turns, 5 tools repeated) | 3,657 | 1,573 | 2,084 | **57.0%** | $31.26 |
| RAG pipeline (6 chunks, pretty-printed) | 544 | 386 | 158 | **29.0%** | $2.37 |
| API response analysis (nested JSON, 5 orders) | 1,634 | 616 | 1,018 | **62.3%** | $15.27 |
| Long debug session (50 turns, JSON logs) | 3,856 | 1,414 | 2,442 | **63.3%** | $36.63 |
| OpenAPI spec context (5 endpoints) | 2,649 | 762 | 1,887 | **71.2%** | $28.30 |
| **Total** | **12,340** | **4,751** | **7,589** | **61.5%** | **$113.84** |

### Transforms Applied

| Scenario | Transforms |
|---|---|
| Agentic coding assistant | tool_schema_dedup, json_minify, whitespace_normalize |
| RAG pipeline | json_minify |
| API response analysis | json_minify |
| Long debug session | json_minify, chat_history_trim |
| OpenAPI spec context | json_minify |

### Where the Savings Come From

- **JSON minification** — Pretty-printed JSON (indent=2 or indent=4) is common in agent tool outputs, RAG chunks, and API responses. Compact re-serialization removes all formatting whitespace while preserving every value.
- **Tool schema deduplication** — Agent frameworks often re-send the full tool schema with every turn. NadirClaw keeps the first occurrence and replaces repeats with a short reference.
- **Chat history trimming** — Long conversations accumulate tokens that are far from the current task. Trimming to recent turns (default: 40) keeps context relevant and cheap.
- **Whitespace normalization** — Log dumps, stack traces, and verbose output contain runs of blank lines and spaces that carry no semantic value.
- **Columnar JSON-array packing** (`json_array_pack`, aggressive mode) — Large arrays of same-keyed objects (DB query results, API list responses, large tool outputs) repeat every key on every row. Packing them into a header (`⟦cols=[...]⟧`) plus one value-array per row emits each key once. Information-lossless and deterministically reversible, but not byte-identical JSON, so it runs in **aggressive** mode only. On a 100-row homogeneous array this reaches ~68% vs pretty-printed JSON (vs ~45% for `json_minify` alone).

## Projected Monthly Savings (Opus 4.6)

| Daily Requests | Monthly Requests | Tokens Saved | Monthly Savings |
|---:|---:|---:|---:|
| 100 | 3,000 | ~4.5M | **$68** |
| 500 | 15,000 | ~22.8M | **$342** |
| 1,000 | 30,000 | ~45.5M | **$683** |
| 5,000 | 150,000 | ~227.7M | **$3,415** |
| 10,000 | 300,000 | ~455.3M | **$6,830** |

*Average savings per request: ~1,517 tokens (61.5%)*

## Safety Guarantees

All safe-mode transforms are deterministic and lossless:

- JSON values roundtrip exactly (parse + compact re-serialize)
- Code blocks inside fences (```) are never modified
- **Leading indentation is preserved**, so raw (unfenced) source code — e.g. file-read
  tool outputs — stays syntactically valid. Whitespace normalization only collapses
  *interior* multi-spaces and excess blank lines, never indentation.
- URLs are preserved character-for-character
- Unicode and emoji roundtrip correctly
- Deeply nested structures are handled without data loss
- `off` mode has zero overhead — no message copying, no processing

## How to Enable

```bash
# Server-wide
nadirclaw serve --optimize safe

# Or via environment variable
NADIRCLAW_OPTIMIZE=safe nadirclaw serve

# Per-request override (in the request body)
{"model": "auto", "optimize": "safe", "messages": [...]}

# Dry-run on a file
nadirclaw optimize payload.json --mode safe --format json
```

## Backends: native (default) vs headroom

The optimizer has a pluggable backend, selected independently of the `off|safe|aggressive`
mode. The mode still decides *how hard* to compress; the backend decides *who* runs it.

| Backend | Default | Engine | Extra capabilities |
|---|---|---|---|
| `native` | ✅ | Built-in stdlib pipeline (this document) | None — pure Python, no extra deps |
| `headroom` | opt-in | [Headroom](https://github.com/chopratejas/headroom) (Apache-2.0) | Statistical JSON-array crushing (SmartCrusher), AST-aware code compression, content-type routing |

`headroom` delegates to the optional [`headroom-ai`](https://pypi.org/project/headroom-ai/)
package. It ships **installed by default with Nadir Pro** but stays **inactive** until you
select it. In open-source NadirClaw it is an opt-in extra:

```bash
pip install "nadirclaw[headroom]"
```

Activate it:

```bash
# Server-wide
NADIRCLAW_OPTIMIZE=safe NADIRCLAW_OPTIMIZE_BACKEND=headroom nadirclaw serve

# Per-request override (in the request body)
{"model": "auto", "optimize": "safe", "optimize_backend": "headroom", "messages": [...]}
```

Safety and fallback:

- If `headroom-ai` is not installed (or raises), the optimizer **transparently falls back
  to `native`** and logs a one-time warning. Requests never fail because of the backend.
- Token-savings metrics are always recomputed with NadirClaw's own estimator, so reported
  numbers stay consistent across backends (Savings/Billing math is unaffected).
- Headroom's ML text compressor (Kompress) downloads a HuggingFace model on first use, so
  it is kept **disabled** by default. Opt in with `NADIRCLAW_HEADROOM_KOMPRESS=on`.
- The fastest Headroom compressors (SmartCrusher etc.) are a compiled Rust extension bundled
  in the prebuilt wheels. On source installs without the wheel they simply don't run, and
  Headroom fails open — output is still correct, just less compressed.

Attribution for the Apache-2.0 dependency lives in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Progressive (staged) compression

`compress_progressive()` escalates through compression stages and **stops as soon as a
token budget is met** — so you only pay the cost (and fidelity risk) of heavier compression
when lighter stages aren't enough. Headroom is wired in as the middle/late tiers.

The ladder, cheapest/safest first:

| Stage | What runs | Loss | Needs |
|---|---|---|---|
| 1. `native_safe` | system/tool dedup, json minify, whitespace | lossless | — |
| 2. `native_aggressive` | + columnar packing, semantic dedup, Pro transforms | lossless-to-semantic | — |
| 3. `headroom_structural` | Headroom content compressors (SmartCrusher, LogCompressor, …) | high-fidelity | `headroom-ai` |
| 4. `headroom_ml` | Headroom Kompress (ML token-dropping on prose) | lossy | `headroom-ai` + `allow_lossy` |

Rules:

- With **no `target_tokens`**, the ladder stops after `native_aggressive` — Headroom and the
  lossy ML stage are never reached. Default behaviour stays dependency-free and lossless.
- The Headroom stages are **skipped silently** when `headroom-ai` is not installed.
- `headroom_ml` (lossy) only runs when `allow_lossy=True`.
- Chat-history trimming always runs last as a final backstop.

```python
from nadirclaw.optimize import compress_progressive   # or nadir.optimize for Pro

result = compress_progressive(
    messages,
    target_tokens=180_000,     # e.g. the model's context window
    allow_lossy=False,         # set True to permit the lossy ML stage
    max_stage="headroom_structural",
)
# result.optimizations_applied is prefixed with stage:<name> markers that ran
```

Enable it on the server — `progressive` is just a value of the single `optimize`
control, alongside `off` / `safe` / `aggressive`:

```bash
# off | safe | aggressive | progressive  (off = compression disabled)
NADIRCLAW_OPTIMIZE=progressive \
NADIRCLAW_OPTIMIZE_TARGET_TOKENS=180000 \
NADIRCLAW_OPTIMIZE_MAX_STAGE=headroom_structural \
nadirclaw serve

# equivalently: nadirclaw serve --optimize progressive
# per-request:  {"optimize": "progressive", "messages": [...]}
# turn compression off:  {"optimize": "off", ...}
```

On a logs+prose payload where native compression yields ~0%, escalating to
`headroom_structural` reached ~90% — the escalation only spends the Headroom budget when
native genuinely can't deliver.
