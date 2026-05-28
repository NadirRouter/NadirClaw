# Multi-provider routing — learnings + reproducibility

NadirClaw's default cascade profile (`nadirclaw/cascade_rules/profiles/default.yaml`)
is calibrated for a single-vendor ladder (Anthropic Haiku → Sonnet →
Opus). In practice many users want to route across vendors: Google
Gemini for the cheap tier, OpenAI or Anthropic in the middle, an Opus-
or reasoning-class model at the top, and an open-weight Llama-family
fallback. This document captures the failure modes we observed when
expanding Nadir's RouterArena submission from a single-provider menu to
a four-provider menu, the rules that mitigated them, and a
reproducibility recipe for re-running the experiment against cached
benchmark responses (no live API calls required).

The cross-vendor profile is shipped as
`nadirclaw/cascade_rules/profiles/multi_provider.yaml`. Load it with
`load_profile("multi_provider")`.

## What changes when you cross provider boundaries

Single-vendor cascades only have to model one safety policy, one
refusal style, and one tokeniser-driven length budget. Cross-vendor
cascades pick up four new variance sources:

1. **Refusal style drift.** Cheap-tier models from different vendors
   refuse slightly different prompts. Vendor A's small model may wrap
   a politically charged prompt in boilerplate while vendor B's small
   model answers cleanly. The heuristic verifier
   (`nadirclaw/heuristic_verifier.py`) catches most boilerplate
   refusals, but the empirical hit rate varies across vendors.
   Mitigation: `multi_provider.yaml` adds a `refusal_prone_jailbreak_keywords`
   rule that pre-empts the verifier round-trip on jailbreak-shaped
   prompts and escalates directly.
2. **Chain-of-thought ability gap.** Cheap-tier models diverge more on
   "step by step" prompts than on factual recall. The mid tier in any
   provider tends to be uniformly competent at multi-step reasoning,
   so escalation pays for itself. Mitigation:
   `chain_of_thought_trigger` and `math_proof_triggers` force-escalate.
3. **Structured-output drift.** "Return JSON" prompts produce wrapped-
   in-prose JSON on some cheap tiers and clean JSON on others. The
   verifier's JSON check helps, but the safer policy is to raise the
   acceptance bar so borderline scores escalate. Mitigation:
   `threshold_json_structured_output` sets τ to 0.85.
4. **Length-control drift.** "Summarize this" produces a 2-sentence
   summary from one vendor and a 6-paragraph essay from another, even
   at the same cheap-tier price point. Mitigation:
   `threshold_summarize_cross_provider` sets τ to 0.82 on
   summarisation prompts and `threshold_long_prompt` to 0.85 on long
   prompts.

The profile also keeps two `force_cheap` rules
(`force_cheap_trivial_greeting`, `force_cheap_thanks_acknowledge`) so
the cascade does not waste a verifier round-trip on prompts that
*every* vendor's cheap tier answers reliably.

## Reproducibility — routing across cached LLM benchmarks

NadirClaw's classifier + rule engine can populate predictions for any
public benchmark *without making live API calls* when the benchmark
ships pre-recorded responses. RouterArena is the reference case —
each row has the full per-model response cached in
`./cached_results/`. To reproduce Nadir's submission numbers (or the
multi-provider variant) on a laptop:

1. **Clone the benchmark with its response cache.**

   ```bash
   git clone https://github.com/RouteWorks/RouterArena
   cd RouterArena
   # cached_results/ ships pre-recorded responses for every model in the menu
   ```

2. **Run NadirClaw's pre-generation classifier over the prompts.**
   This is what produces the routing decision. No API keys, no
   network — just the bundled `wide_deep_asym_v3.pt` + BGE encoder.

   ```python
   from nadirclaw.wide_deep_classifier import get_wide_deep_classifier

   clf = get_wide_deep_classifier(
       checkpoint_variant="asym",
       decision_rule="cost_sensitive",
       cost_lambda=20.0,  # max-safe; matches Nadir Pro's RouterArena tuning
   )

   for row in benchmark_rows:
       result = clf.classify(row["prompt"])
       row["nadir_tier"] = result.tier            # simple / medium / complex
       row["nadir_model"] = TIER_TO_MODEL[result.tier]
   ```

3. **Layer the cascade rule engine on top.** This is where
   `multi_provider.yaml` (or your own custom profile) overrides the
   classifier in domains where the classifier is known to be weak.

   ```python
   from nadirclaw.cascade_rules import load_profile

   engine = load_profile("multi_provider")
   for row in benchmark_rows:
       decision = engine.evaluate(row["prompt"], predicted_tier=row["nadir_tier"])
       if decision.action == "force_escalate":
           row["nadir_tier"] = "complex"
           row["nadir_model"] = TIER_TO_MODEL["complex"]
       elif decision.action == "force_cheap":
           row["nadir_tier"] = "simple"
           row["nadir_model"] = TIER_TO_MODEL["simple"]
       # set_threshold is consumed by Cascade, not the offline router
   ```

4. **Read the cached response for the chosen model.** No
   `openai.ChatCompletion.create` call needed. This is how the
   benchmark scores cost / latency / quality without spending API
   budget on a re-run.

   ```python
   for row in benchmark_rows:
       row["response"] = row["cached_responses"][row["nadir_model"]]
   ```

5. **Score with the benchmark's own metric.** RouterArena ships its
   composite metric in-repo.

The end-to-end loop is offline, deterministic, and reproducible from
this codebase plus the benchmark repo. The path Nadir followed for the
public RouterArena submission is documented in
[`RouteWorks/RouterArena#112`](https://github.com/RouteWorks/RouterArena/pull/112)
— the same code path with a different rule profile.

## Verifying the classifier did not see the benchmark

The `verifier/contamination_audit.py` utility shipped with NadirClaw
runs a stdlib-only SHA256 overlap audit between the trained
classifier's labeled corpus and a held-out benchmark. The hash recipe
is `sha256(NFC(prompt).strip().casefold().utf8)`. For RouterArena
`sub_10` (n=809) and `full` (n=8,399), overlap is zero — published in
[`MODEL_CARD.md`](../MODEL_CARD.md). Re-running the audit on your own
held-out set is a single CLI call.

## Tuning your own cross-provider profile

Copy `multi_provider.yaml`, edit the patterns, and load it from an
absolute path:

```python
from nadirclaw.cascade_rules import load_profile
engine = load_profile("/path/to/my_custom_profile.yaml")
```

The profile is hot-reloaded from disk every 30 seconds (TTL cache),
so iteration is fast. Each rule's `meta.rationale` field is the place
to record *why* you added it — the engine ignores `meta` but the YAML
is the durable record.
