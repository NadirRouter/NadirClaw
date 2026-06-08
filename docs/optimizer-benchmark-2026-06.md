# Context Optimizer Benchmark — native vs Headroom, synthetic vs real data

**Date:** 2026-06-06
**Scope:** Evaluate the context optimizer (`nadirclaw.optimize` / `nadir.optimize`) across
backends (`native`, `headroom`) and modes (`safe`, `aggressive`) on both synthetic
payloads and real public coding + chat datasets. Establish whether Headroom improves
on the native pipeline, and where the realistic savings ceiling is.

## TL;DR

- **On real conversational traffic, lossless savings are 1–10%, not the 30–60% synthetic
  payloads suggest.** Real traffic is prose-dominated; structural optimizers (JSON minify,
  schema dedup, whitespace) have little to grab.
- **The `headroom` *library `compress()` wrapper* underdelivers** (it routes conservatively and
  protects messages). But **Headroom's transforms, called directly "as is", do reproduce their
  published numbers** — LogCompressor ~80% on repetitive logs, SmartCrusher ~73% on homogeneous
  JSON arrays. The earlier "Headroom underperforms" verdict was about the wrapper, not the engine.
- **One real capability gap was found and closed natively:** SmartCrusher packs homogeneous JSON
  arrays into a columnar table (~50% beyond our `json_minify`). We now do this losslessly in
  `aggressive` mode via the new `json_array_pack` transform (68% vs pretty JSON, vs Headroom's 73%),
  with no dependency and no CCR machinery.
- **Pro-aggressive is the best performer** everywhere — all native, no new dependencies.
- **Decision:** keep `native` the default, `headroom` a safe opt-in (already shipped this way).
  Remaining gains require prose compression (lossy ML + a CCR recovery path that is not built).

## Method

- **Backends compared:** `native-safe` (lossless, ships today), `pro-aggressive` (native
  ceiling: + secret-mask, tool-schema compaction, log/stack compression, semantic dedup),
  `headroom` (optional `headroom-ai` backend, Kompress disabled unless noted).
- **Token metric:** the optimizer's own tiktoken `cl100k_base` estimator, applied identically
  to every backend's output, so comparisons are fair regardless of each engine's internal count.
- **Datasets (public, no PII):**
  - Chat: `allenai/WildChat-1M` — real multi-turn user↔assistant conversations.
  - Coding/tools: `glaiveai/glaive-function-calling-v2` — tool schemas + function calls + JSON.
  - 200 conversations each, fetched via the HF datasets-server `/rows` API (no full download).
- **Environment note:** `headroom-ai` (Rust/PyO3 ≤ 3.13) **cannot build on Python 3.14**.
  Benchmarks ran on a Python 3.13 venv with the prebuilt wheel. getnadir/Nadir target 3.12, so
  this is fine in production, but 3.13 is the current ceiling for the Headroom dependency.

## Results — synthetic payloads

Hand-built "bloated" payloads (repeated tool schemas, pretty-printed JSON arrays, log dumps):

| Backend | total reduction | notes |
|---|--:|---|
| native-safe | **30.4%** | lossless; strong on repeated tool schemas |
| headroom | 25.2% | worse — no cross-message schema dedup; lossy crush never fired |
| pro-aggressive | **60.3%** | `pattern_compression` took 200 log lines 0% → 87% |

## Results — real data (the important part)

| | raw tokens | native-safe | pro-aggressive | headroom |
|---|--:|--:|--:|--:|
| **Chat** (WildChat-1M, 195 convs) | 191,101 | **1.0%** | **4.7%** | 0.1% |
| **Coding/tools** (glaive, 200 convs) | 111,697 | **8.2%** | **9.6%** | 2.6% |

Transform frequency (real data):
- Chat: `whitespace_normalize` 37×, `semantic_dedup` 20× (the only real lift), `json_minify` 7×.
- Coding: `json_minify` 135× (the workhorse), `tool_schema_compact` 53×, `tool_schema_dedup` **0×**
  (real requests carry each schema once, not repeated across turns).

## Diagnosis — why real savings are low

Token mass is **prose-dominated**, and structural optimizers cannot compress prose:

- **Chat is ~100% natural language.** Native has almost nothing to grab (1%). `semantic_dedup`
  is the only lever that moved it (→ 4.7%).
- **Coding token distribution (glaive, by role):** assistant **67.7k (60%)**, system 22.3k (20%,
  the tool schemas), user 17.3k, tool 4.8k. The compressible part is the 20% of JSON schema;
  the 60% assistant prose is untouchable by structural methods.
- **Lossless levers are exhausted:**
  - Fenced JSON (the minifier skips code fences): **0** minifiable blocks in chat, **1** in glaive.
    The 66 chat code-fences are *code*, not JSON.
  - Verbatim block repetition across turns (≥80-char lines repeated in an earlier message):
    **2.5%** chat / **0.5%** coding — the largest remaining lossless lever, and still small.

## Headroom findings (tested two ways)

**Via the library `compress()` wrapper (what our backend integrates):** underdelivers.
SmartCrusher and CodeCompressor never engaged at any `target_ratio` — the wrapper routes
conservatively and protects user/recent messages, so the heavy transforms rarely fire on real
message content. This is why our backend benchmark was low.

**Calling the transforms directly ("as is"):** the published per-type numbers reproduce.

| Transform (direct call) | content | result | notes |
|---|---|--:|---|
| `LogCompressor.compress` | 200 repetitive log lines | **79.7%** (lossy) | matches their 80–95% claim; our Pro `pattern_compression` does 87% on the same logs |
| `SmartCrusher.crush` | 100-row homogeneous JSON array | **73% vs pretty** (lossless table) | columnar format; this is the one capability we lacked |
| `SmartCrusher.crush` | 50 *unique* (non-redundant) objects | ~47% (lossless) | falls back to ≈ our `json_minify` when rows aren't homogeneous |
| `CodeAwareCompressor.compress` | Python source | **broke** (AST bytes bug → `syntax_valid=False`) | not usable in this build |

Why their headline % looks bigger than ours: it is measured against **pretty-printed** JSON, and
on **ideal redundant content** (homogeneous arrays, repetitive logs). Real conversational traffic
is prose-dominated and we already minify, so the marginal win is smaller — except the columnar
table, which is genuinely additive (see below).

**Kompress (ML token-dropping)** is the one place Headroom wins on *prose*: ~12% on unique prose
(native: 0%), ~60% on repetitive boilerplate. But it is lossy (drops function words, fuses
sentences) and emits a `[... Retrieve more: hash=...]` marker that is **unrecoverable in our
wiring** (no `headroom_retrieve` endpoint). It stays `disabled` by default.

## Native columnar packing (`json_array_pack`) — the capability we adopted

The only reproducible Headroom win we lacked was SmartCrusher's columnar table for homogeneous
JSON arrays. We now do it natively in `aggressive` mode:

| 100-row homogeneous array | tokens | vs pretty |
|---|--:|--:|
| pretty JSON | 4,202 | — |
| `json_minify` (safe) | 2,302 | 45% |
| **`json_array_pack` (aggressive)** | **1,323** | **68%** |
| Headroom SmartCrusher | 1,119 | 73% |

It rewrites an array of same-keyed objects into a header (`⟦cols=[...]⟧`) plus one JSON
value-array per row, emitting each key once instead of N times. It is **information-lossless and
deterministically reversible** (`_unpack_table`), runs only when the array is strictly homogeneous
(≥ 5 rows, identical key sets) and only when it saves tokens, and **never runs in `safe` mode**
(it is not byte-identical JSON). The 5pp gap to SmartCrusher is format: it uses bare CSV rows; we
keep JSON-array rows so reversibility is robust across nested/special values.

**Caveat:** the public chat/coding datasets above barely contain homogeneous arrays
(`json_array_pack` fired once on glaive), so this does not move those totals. It targets
tool-output traffic — DB query results, API list responses, large `get_*` tool returns — which
production agent loops carry but these datasets do not.

## Code safety (correctness fix)

Testing the optimizer on raw source code surfaced a real bug: `whitespace_normalize`
collapsed the **leading indentation** of unfenced code (file-read tool outputs), flattening
nested Python into **invalid syntax** while reporting ~12–14% "savings". That apparent
code compression was the corruption — not real savings.

Fixed across all three optimizer copies (NadirClaw, Nadir Pro inherits it, getnadir): the
normalizer now preserves leading whitespace and only collapses interior multi-spaces.
Regression tests assert raw code stays `ast.parse`-valid in both safe and aggressive modes.

Corrected takeaway: honest lossless savings on clean source code are **~0%** — structural
optimization has nothing safe to remove from well-formatted code. (Headroom's CodeCompressor
is the only engine that targets code, and it errored to invalid output in this build.)

## Recommendations

1. **Keep the shipped posture.** Native default, headroom opt-in, Kompress off. Validated correct.
2. **For chat-heavy traffic, `aggressive` mode is the lever** — `semantic_dedup` is the only thing
   that moves prose, lossless-ish, already available in NadirClaw and Pro.
3. **Optional small lossless win (deferred):** a verbatim block-dedup transform would add ~2.5% on
   chat. Modest; weigh against the readability cost of inline reference markers.
4. **Large prose savings require investment:** ML token compression (Kompress) behind a real
   CCR `headroom_retrieve` recovery endpoint. Only worth it if prose-heavy traffic dominates the
   bill. Not built; lossy without it.

## Reproduction

```bash
# Headroom needs Python <= 3.13 (Rust/PyO3). Build a 3.13 venv:
python3.13 -m venv /tmp/hr-bench
/tmp/hr-bench/bin/pip install "headroom-ai>=0.23.0" tiktoken sentence-transformers
# Real-data benchmark (fetches 200 convs each from WildChat + glaive via HF datasets-server):
/tmp/hr-bench/bin/python NadirClaw/benchmarks/optimize_real_data.py
```

The benchmark script lives at [`benchmarks/optimize_real_data.py`](../benchmarks/optimize_real_data.py).
See also [context-optimize-savings.md](context-optimize-savings.md) for the transform-level
savings analysis and the `native` vs `headroom` backend reference.
