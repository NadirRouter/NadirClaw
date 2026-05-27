# Model Card — `wide_deep_asym_v3` (Nadir pre-generation classifier)

This card documents the architecture, training corpus, contamination
posture, and published benchmark numbers for the pre-generation tier
classifier that powers Nadir's routing decisions.

NadirClaw, the open-source router in this repo, ships **the architecture
description and the heuristics on top of it** (see
`nadirclaw/cascade.py`, `nadirclaw/cascade_rules/`,
`nadirclaw/heuristic_verifier.py`). The trained `wide_deep_asym_v3.pt`
artifact itself is proprietary to [Nadir Pro](https://getnadir.com).
NadirClaw users get the same routing topology with the simpler binary
centroid or DistilBERT classifier — the card below explains where the
Pro-tier numbers come from so users can decide whether to stay free or
upgrade.

- **Router name**: `nadir`
- **Classifier family**: wide-and-deep asymmetric (`wide_deep_asym`)
- **Production artifact (Nadir Pro)**: `wide_deep_asym_v3.pt`
- **Card last updated**: 2026-05-27
- **Schema version**: 1

---

## 1. Architecture

`wide_deep_asym_v3` is a wide-and-deep classifier trained on prompt
features to predict a routing tier in `{simple, medium, complex}`.

- **Wide branch** — structural and lexical features (length buckets,
  code-fence indicators, math symbol density, JSON shape hints,
  question-word counts).
- **Deep branch** — BGE sentence-transformer embedding (`bge-small-en`).
- **Head** — three-way softmax over `{simple, medium, complex}`.
- **Loss** — asymmetric cross-entropy with downgrade penalty
  `λ = 3` in v3. Downgrades (predicting `simple` when `complex` was
  correct) are penalised 3× more than upgrades.

Tier mapping for the reference deployment (Anthropic Claude 4.x ladder):

| Tier | Model |
| --- | --- |
| simple | `claude-haiku-4-5` |
| medium | `claude-sonnet-4-6` |
| complex | `claude-opus-4-6` |

### Inputs / outputs

**Input**: a single user message (string). Multi-turn messages are
concatenated by the production analyzer before classification.

**Output**:

- `tier` in `{simple, medium, complex}`
- `model` (the corresponding tier model name)
- `complexity_score` in `[0, 1]`
- `classifier_confidence` in `[0, 1]` (softmax top-class probability)
- `latency_ms` (single-core CPU)
- `classifier_version` (`wide_deep_asym_v3`)

---

## 2. Training data

Training is **deliberately disjoint** from RouterBench and RouterArena.

Sources for `wide_deep_asym_v3`:

- Internal Nadir labeled batches (`backend/labeled_data/v3/...`, not
  part of this open-source repo).
- Prior labeled batches under `v2/`, `raw/`, `batches/`.

The verifier corpus used to train the post-generation cascade verifier
is stored separately and was not used to train the pre-generation
classifier.

### Contamination audit status

| Held-out set | Audit run | Overlap | Verdict |
| --- | --- | --- | --- |
| RouterBench `0shot` | 2026-05-24 | 0 of 36,481 | DISJOINT |
| RouterArena `sub_10` | 2026-05-27 | 0 of 809 | DISJOINT |
| RouterArena `full` | 2026-05-27 | 0 of 8,399 | DISJOINT |

Audits are reproducible from this repo with the script in
`verifier/contamination_audit.py`. Hash recipe:
`sha256(NFC(prompt).strip().casefold().utf8)`.

---

## 3. Performance

### Pre-generation classifier (this card)

- **Held-out RouterBench** (n=11,420 prompts):
  - AUROC **0.961** for binary "should escalate?" decision composed
    with the post-generation verifier.
  - Expected Calibration Error (ECE) **0.016** at the production
    operating point.
- **RouterArena `sub_10`** (n=809, public leaderboard):
  - Composite score **0.7118**, currently projected #5 on the public
    leaderboard (ahead of NotDiamond, Auto Router, Martian).
  - RouterArena submission PR:
    https://github.com/RouteWorks/RouterArena/pull/112
- **Pre-generation, prompt-only** (no verifier): AUROC ~0.62 on
  RouterBench cross-family triples. The pre-generation ceiling is the
  architectural reason Nadir layers a post-generation cascade verifier
  on top.

### Cascade verifier (separately published)

The post-generation cross-encoder verifier is shipped in **Nadir Pro**
as DeBERTa-v3-small INT8 quantized. NadirClaw ships a rule-based
heuristic verifier with the same interface (see
`nadirclaw/heuristic_verifier.py`); the heuristic version reaches ~0.60
AUROC on the same held-out triples but catches the bulk of refusals,
truncations, and JSON-format failures.

Composed-system numbers (classifier + Pro verifier) on RouterBench:

- AUROC 0.961, ECE 0.016.
- At τ=0.80: **98% of always-Opus quality preserved** (catastrophic
  ≤ 1.7%), composed cost ~60% reduction vs always-Opus.
- Verifier latency 192.9 ms per call, single-core CPU, INT8 qnnpack.

τ-sweep (from the same held-out report):

| τ | accept rate | catastrophic | wasted escalation | quality preserved |
| --- | --- | --- | --- | --- |
| 0.70 | 0.69 | 0.024 | 0.078 | 97.6% |
| 0.75 | 0.67 | 0.019 | 0.089 | 98.1% |
| **0.80** | **0.67** | **0.017** | **0.092** | **98.3%** |
| 0.90 | 0.64 | 0.011 | 0.108 | 98.9% |

τ=0.80 is the production operating point. NadirClaw's cascade defaults
to τ=0.80 in `DEFAULT_ACCEPTANCE_THRESHOLD`.

---

## 4. Intended use

- Pre-generation tier selection for LLM routing on the Claude 4.x
  ladder, or any three-model ladder mapped to the same tiers.
- Public-benchmark evaluation (RouterBench, RouterArena).

### Out of scope

- Not a quality verifier on its own — the post-generation cascade
  verifier closes the pre-generation gap.
- Not a guarantee of model output correctness. The router's job is to
  pick a model.
- Not validated on languages other than English at the published
  thresholds.

---

## 5. Limitations

1. **Pre-generation ceiling.** Prompt-only classification has bounded
   AUROC on cross-family distributions (~0.62 on RouterBench). The
   router cannot know whether Haiku will get the answer right; it can
   only know whether Haiku *usually* gets that *kind* of prompt right.
   The post-generation cascade verifier is the architectural answer.
2. **Per-domain variance.** Verifier AUROC ranges from ~1.0 on
   factual-recall (MMLU-style) prompts down to ~0.65 on code
   generation and ~0.77 on long-form summarisation. The default
   `cascade_rules` profile encodes those weak-verifier domains as
   force-escalate / set-threshold rules so the cascade does not rely
   on the verifier where it is known to be unreliable.
3. **Training data is not adversarial.** The classifier has not been
   stress-tested against prompt-injection-style inputs designed to
   force a particular tier.
4. **Asymmetric loss at λ=3.** The router prefers upgrades over
   downgrades, which inflates the wasted-escalation rate on
   pure-cheap prompts. This is intentional: catastrophic downgrade is
   more expensive in customer trust than wasted Sonnet calls.

---

## 6. NadirClaw vs Nadir Pro on this card

| Component | NadirClaw (OSS) | Nadir Pro |
| --- | --- | --- |
| Pre-generation classifier | Binary centroid (~10 ms) or DistilBERT (3-class, opt-in) | `wide_deep_asym_v3` (the model documented in this card) |
| Post-generation verifier | Rule-based heuristic, ~1 ms | DeBERTa-v3-small INT8, ~193 ms, AUROC 0.96 |
| Cascade rule engine | Same engine, default profile bundled | Same engine, same default profile, plus per-tenant overrides |
| Default τ | 0.80 | 0.80 (env override `CASCADE_DEFAULT_THRESHOLD`) |
| Contamination audit utility | `verifier/contamination_audit.py` | Same script, plus internal corpus loader |

If you want the trained classifier numbers reproduced on your own
workload, the path is: run NadirClaw with the heuristic verifier first,
log decisions and outcomes, then use those logs as the labeled corpus
for training your own wide-and-deep classifier following the
architecture above. Nadir Pro automates this loop for hosted customers.

---

## 7. Contact

- Project: https://getnadir.com
- GitHub: https://github.com/NadirRouter/NadirClaw
- Email: hello@getnadir.dev
