# NadirClaw Pro/Enterprise Roadmap

> **Date:** March 25, 2026 | **Prepared by:** Product, Engineering, Sales, Data, ML, and Market Research

---

## Executive Summary

NadirClaw occupies a **unique position** in the LLM infrastructure market: the only open-source, fully-local, ML-powered router with a specific focus on coding agent cost optimization. The paid version differentiates on three axes:

1. **Intelligence** — the classifier gets smarter over time via feedback and custom training
2. **Trust** — quality scoring proves that cheaper routing doesn't sacrifice output quality
3. **Scale** — multi-tenant, RBAC, compliance, and team analytics make NadirClaw viable for organizations

**Pricing philosophy:** Our savings-based model ($9/mo base + share of savings) is a zero-risk value proposition. Customers only pay more when they save more. This aligns incentives perfectly and eliminates the biggest objection to paid developer tools.

**Market context:** The LLM middleware/gateway market is projected to grow from $18.9M (2026) to $189M (2034) at 49.6% CAGR. Enterprise LLM spend is $4.84B in 2025, reaching $48.25B by 2034. Over 90% of production AI teams now run 5+ LLMs simultaneously.

---

## Table of Contents

1. [Competitive Positioning](#1-competitive-positioning)
2. [Pricing & Tiers](#2-pricing--tiers)
3. [Phased Roadmap](#3-phased-roadmap)
4. [ML Algorithm Improvements (Deep Dive)](#4-ml-algorithm-improvements)
5. [Data & Analytics Strategy](#5-data--analytics-strategy)
6. [Go-to-Market & Sales](#6-go-to-market--sales)
7. [Revenue Projections](#7-revenue-projections)
8. [Key Risks & Mitigations](#8-key-risks--mitigations)
9. [Success Metrics](#9-success-metrics)
10. [Top 10 User Stories](#10-top-10-user-stories)

---

## 1. Competitive Positioning

### Landscape

| Competitor | Approach | Self-Hosted | ML Routing | Pricing |
|---|---|---|---|---|
| **OpenRouter** | Cloud proxy/marketplace | No | Weak (heuristics) | 5-5.5% markup |
| **Martian** | Model mapping (interpretability) | No | Strongest | $20+/mo |
| **Unify AI** | Benchmark-driven routing | No | Strong | $700+/mo |
| **Portkey** | Enterprise gateway (now OSS) | Yes | None | Free-$49/mo+ |
| **LiteLLM** | Unified proxy (NadirClaw uses it) | Yes | None | Free (OSS) |
| **Helicone** | Observability + gateway | Yes | None | Free-$20/seat |
| **Not Diamond** | ML routing + custom training | No | Strong | Free (basic) |
| **NadirClaw** | **ML routing + local-first** | **Yes** | **Strong** | **Free (OSS) / $9/mo + savings share** |

### NadirClaw's Defensible Moat

- **Only ML-powered router that runs fully locally** — keys never leave the user's machine
- **Coding agent specialization** — integrations with Cursor, Claude Code, Continue, Aider, Windsurf, Codex, Open WebUI
- **10ms classification overhead** — lightweight enough to sit in the critical path
- **Open-source trust** — community can inspect and modify routing logic
- **Savings-aligned pricing** — unlike flat-fee or markup competitors, we only charge when we deliver value

### Strategic Positioning

> "Portkey is the governance gateway. NadirClaw is the intelligence layer. Use both together."

Don't compete on governance features. **Own the "smart cost optimization that runs on your machine" category.**

---

## 2. Pricing & Tiers

### Pricing Overview

| | **Open Source (Free)** | **Pro ($9/mo + savings share)** | **Team ($29/mo + savings share)** | **Enterprise (Custom)** |
|---|---|---|---|---|
| **Deployment** | Self-hosted | Hosted proxy (api.getnadir.com) | Hosted proxy (dedicated) | Dedicated infrastructure |
| **Base Fee** | $0 | $9/month | $29/month | Custom contract |
| **Savings Share** | None | 25% of first $2K/mo, 10% above | 20% of first $5K/mo, 8% above | Negotiated (volume discounts) |
| **Support** | GitHub Issues | Email (48hr SLA) | Priority (24hr SLA) | Dedicated Slack + engineer |

### Feature Matrix

| | **Open Source** | **Pro** | **Team** | **Enterprise** |
|---|---|---|---|---|
| **Routing** | 4-tier, all profiles | + Adaptive classifier | + Per-team routing rules | + Custom classifier training |
| **Context Optimize** | Safe mode | + Aggressive semantic dedup | + Custom optimization rules | + Domain-specific optimizers |
| **Caching** | In-memory LRU | + Persistent SQLite cache | + Semantic cache (FAISS) | + Distributed cache (Redis) |
| **Analytics** | CLI dashboard, basic reports | + Web dashboard, 90-day history, cost forecast | + Per-team/user breakdown | + Unlimited retention, export |
| **Keys** | BYOK (single set) | + BYOK or use Nadir's keys, multi-key mgmt | + Per-team API keys | + Custom model registry |
| **Quality** | -- | + Quality scoring + feedback loop | + A/B testing framework | + LLM-as-judge eval pipeline |
| **Governance** | Single user | + Multi-API-key management | + RBAC, SSO (Google/Okta) | + SAML, audit trail, SOC2 |
| **Deployment** | Single instance | -- | + Multi-instance (Redis) | + Helm charts, air-gapped |
| **Requests** | Unlimited | Unlimited | Unlimited | Unlimited |
| **License** | MIT | Commercial | Commercial | Commercial |

### Savings-Based Pricing: How It Works

**Savings = benchmark model cost - actual routed cost**

The benchmark cost is what you would have paid using your default model (e.g., Claude Opus 4.6) for every request. The routed cost is what you actually paid after Nadir's intelligent routing + context optimization.

**Pro tier fee calculation:**
- 25% of the first $2,000 saved per month
- 10% of savings above $2,000 per month
- Plus $9/month base fee

**Examples (Pro):**
| Monthly Savings | Savings Fee | Base Fee | Total Bill | You Keep |
|---|---|---|---|---|
| $200 | $50 | $9 | $59 | $141 (71%) |
| $500 | $125 | $9 | $134 | $366 (73%) |
| $2,000 | $500 | $9 | $509 | $1,491 (75%) |
| $5,000 | $800 | $9 | $809 | $4,191 (84%) |
| $10,000 | $1,300 | $9 | $1,309 | $8,691 (87%) |

**Team tier fee calculation:**
- 20% of the first $5,000 saved per month (team aggregate)
- 8% of savings above $5,000 per month
- Plus $29/month base fee

**Examples (Team):**
| Monthly Savings | Savings Fee | Base Fee | Total Bill | You Keep |
|---|---|---|---|---|
| $2,000 | $400 | $29 | $429 | $1,571 (79%) |
| $5,000 | $1,000 | $29 | $1,029 | $3,971 (79%) |
| $10,000 | $1,400 | $29 | $1,429 | $8,571 (86%) |
| $20,000 | $2,200 | $29 | $2,229 | $17,771 (89%) |

### Price Justification

- **Pro $9/mo + savings share**: Zero risk. If Nadir doesn't save you money, you pay only $9. A dev saving 40% on $200/mo AI bill saves $80/mo and pays ~$29 total -- 2.7x ROI on total cost.
- **Team $29/mo + savings share**: For 10-50 devs, the base cost is negligible. The savings share scales with value delivered. Teams spending $10K/mo on LLMs save ~$4K and pay ~$1.4K -- 2.8x ROI.
- **Enterprise (Custom)**: For companies spending $100K+/mo on LLMs, negotiated volume discounts on savings share. Even at conservative 20% routing savings = $20K/mo savings.

### Why Savings-Based > Flat Fee

| Aspect | Flat Fee ($49/mo) | Savings-Based ($9 + share) |
|---|---|---|
| Customer risk | Pays even if no value | Only pays when saving |
| Alignment | Misaligned -- vendor gets paid regardless | Perfectly aligned -- more savings = more revenue |
| Expansion | Requires upgrade/upsell | Revenue grows automatically with usage |
| Sales objection | "What if it doesn't work?" | "You only pay when it works" |
| NRR potential | ~100% (flat) | >130% (usage-driven) |

### Add-On Revenue (Planned)

- **Domain Training Packs**: $299 one-time -- pre-trained centroids for legal, medical, financial, code
- **Priority Model Registry**: $19/mo -- 24hr model/pricing updates vs weekly

---

## 3. Phased Roadmap

### Feature Status Legend

| Status | Meaning |
|---|---|
| **Live** | Shipped and available in current release |
| **In Development** | Actively being built |
| **Planned Q2** | Scheduled for Apr-Jun 2026 |
| **Planned Q3** | Scheduled for Jul-Sep 2026 |
| **Planned Q4** | Scheduled for Oct-Dec 2026 |

### Phase 1: Pro Launch -- Q2 2026 (Apr-Jun)

> **Theme: "Smart Routing That Learns"**

#### April (Weeks 1-4): Foundation

| Feature | Description | Tier | Status |
|---|---|---|---|
| **4-Tier Routing** | Simple/mid/complex/expert routing with ML classification | Free | **Live** |
| **Context Optimize (Safe)** | Token-aware context optimization for cost reduction | Free | **Live** |
| **CLI Dashboard & Analytics** | `nadirclaw report`, `nadirclaw dashboard`, savings tracking | Free | **Live** |
| **Editor Integrations** | Cursor, Claude Code, Continue, Aider, Windsurf, Codex, Open WebUI | Free | **Live** |
| **Provider Health Monitor** | Rolling-window error rate per provider, auto-demotion to fallback | Free | **Live** |
| **X-Routed-* Headers** | Response headers for routing transparency | Free | **Live** |
| **Hosted Proxy (api.getnadir.com)** | Zero-setup hosted routing for Pro users | Pro | **In Development** |
| **Savings Tracking & Billing** | Per-request savings calculation, monthly invoicing via Stripe | Pro | **In Development** |
| **Aggressive Semantic Dedup** | Context Optimize aggressive mode for hosted Pro users | Pro | **In Development** |
| **Persistent Semantic Cache** | SQLite-backed embedding cache with configurable similarity threshold (0.95 default). FAISS IndexFlatIP for O(1) nearest-neighbor. | Pro | **Planned Q2** |
| **ONNX Runtime Inference** | Convert all-MiniLM-L6-v2 to ONNX for 2x speedup (~10ms to ~5ms) and 4x smaller footprint (~80MB to ~22MB INT8) | Free | **Planned Q2** |
| **Feedback Endpoint** | `POST /v1/feedback` + `nadirclaw flag <id> --reason misrouted` writes correction records to SQLite | Free | **Planned Q2** |

#### May (Weeks 5-8): Intelligence

| Feature | Description | Tier | Status |
|---|---|---|---|
| **Quality Scoring Engine** | Passive signals (empty response, retry detection, fallback rate) + active feedback (1-5 rating). Configurable sample-rate LLM-as-judge eval. | Pro | **Planned Q2** |
| **Multi-Dimensional Feature Extraction** | Extract 20+ structural features (code blocks, question marks, token count, tool count, URL count, language detection) alongside embeddings | Pro | **Planned Q2** |
| **Adaptive Classifier v1** | Nightly retrain script that regenerates centroid vectors from flagged corrections + quality scores. Versioned centroids with validation gates. | Pro | **Planned Q2** |
| **Custom Routing Rules (YAML)** | `~/.nadirclaw/rules.yaml` -- pin models by project, regex on system prompt, time-of-day rules, force tier overrides | Pro | **Planned Q2** |

#### June (Weeks 9-12): Launch

| Feature | Description | Tier | Status |
|---|---|---|---|
| **Web Dashboard** | Real-time feed, cost trends, savings waterfall, quality scores over time, model utilization heatmap | Pro | **In Development** |
| **Multi-API-Key Management** | `nadirclaw keys create --name "ci-runner" --budget 5.00/day` with per-key rate limits and budget caps | Pro | **Planned Q2** |
| **Data Retention + PII Redaction** | Configurable retention (30/90/730 days), prompt storage modes (full/redacted/hash_only/none), regex + API key pattern PII scrubbing | Pro | **Planned Q2** |
| **Pro Tier GA** | Stripe billing (savings-based), feature gating, docs site | -- | **Planned Q2** |

### Phase 2: Team & Business -- Q3 2026 (Jul-Sep)

> **Theme: "Quality-Aware Routing for Teams"**

#### July

| Feature | Description | Tier | Status |
|---|---|---|---|
| **Multi-Tenant Server Mode** | Auth middleware resolving tenant from API key; per-tenant settings, model config, budget; shared classifier | Team | **Planned Q3** |
| **Team Analytics** | Per-user and per-team cost attribution, department budgets, admin API for user management | Team | **Planned Q3** |
| **SSO (Google/Okta)** | Google and Okta SSO; map groups to NadirClaw roles (admin/viewer/api_only) | Team | **Planned Q3** |
| **Team Savings Billing** | Aggregate savings across team members, team-level invoicing with 20%/8% share structure | Team | **Planned Q3** |

#### August

| Feature | Description | Tier | Status |
|---|---|---|---|
| **Quality Prediction Network** | Multi-head MLP predicting per-model output quality. Shared trunk (384->128->64) with per-model heads (64->1). Trained on feedback + LLM-as-judge labels. | Team | **Planned Q3** |
| **Latency Prediction** | EMA-based with online learning. Predicts P50/P95 per model given estimated input/output tokens. | Team | **Planned Q3** |
| **Pareto Optimizer** | `X-Routing-Priority: quality=0.6, cost=0.3, latency=0.1` header. Selects the Pareto-optimal model per request. | Team | **Planned Q3** |
| **Task Type Detection** | Multi-label classifier (code/reasoning/creative/factual/summarization) replacing regex-based agentic/reasoning detection | Team | **Planned Q3** |

#### September

| Feature | Description | Tier | Status |
|---|---|---|---|
| **A/B Testing Framework** | Deterministic user-hash-based traffic split between routing strategies. Statistical significance reporting. | Team | **Planned Q3** |
| **Anomaly Detection** | Real-time cost spike, error rate surge, latency degradation, routing drift, budget breach detection with webhook alerts | Team | **Planned Q3** |
| **Advanced Cost Analytics API** | `/v1/analytics/cost`, `/v1/analytics/latency`, `/v1/analytics/cache` endpoints with period/groupby/percentile params. CSV/Parquet/JSON export. | Team | **Planned Q3** |

### Phase 3: Enterprise -- Q4 2026 (Oct-Dec)

> **Theme: "Self-Improving Routing at Scale"**

#### October

| Feature | Description | Tier | Status |
|---|---|---|---|
| **Custom Classifier Training (Two-Tower)** | `nadirclaw train --architecture two-tower --data labeled.jsonl` produces ~5.6MB artifact. Supports teacher-student distillation. | Enterprise | **Planned Q4** |
| **Contextual Bandit (LinUCB)** | Replace static tier mapping with exploration-exploitation model selection. Per-user adaptive routing with min 50 samples. | Enterprise | **Planned Q4** |
| **Distributed Rate Limiting** | Redis backend with atomic Lua scripts for sliding window. `NADIRCLAW_RATE_LIMIT_BACKEND=redis` | Enterprise | **Planned Q4** |

#### November

| Feature | Description | Tier | Status |
|---|---|---|---|
| **SAML + Audit Trail** | SAML SSO, immutable append-only audit log. SOC2-compatible event format. Data processing agreement template. | Enterprise | **Planned Q4** |
| **Routing Explainability** | Per-request decision chain: input signals -> classifier output -> candidate ranking -> final decision + human-readable reason | Enterprise | **Planned Q4** |
| **Cost Forecasting** | Linear trend + day-of-week seasonality on 14-day history. Weekly email digest with projected monthly cost. | Enterprise | **Planned Q4** |

#### December

| Feature | Description | Tier | Status |
|---|---|---|---|
| **Custom Model Registry** | Org-private model catalog with internal endpoints (vLLM, TGI, Bedrock). Custom pricing tables, model tagging ("approved-for-pii"). | Enterprise | **Planned Q4** |
| **Air-Gapped Installation** | Offline installer with bundled sentence-transformer, no PyPI/HuggingFace at runtime. Helm charts for K8s. | Enterprise | **Planned Q4** |
| **SDK / Client Libraries** | Python and TypeScript SDKs wrapping the NadirClaw API with retry, streaming, and type safety | Enterprise | **Planned Q4** |
| **99.9% SLA** | Enterprise SLA with dedicated infrastructure, guaranteed uptime, and incident response | Enterprise | **Planned Q4** |
| **v2.0 GA** | Stable API contract, full docs, migration guide from v1.x | -- | **Planned Q4** |

---

## 4. ML Algorithm Improvements

### Current Architecture (Live)

```
prompt -> SentenceTransformer("all-MiniLM-L6-v2") -> 384-dim embedding
       -> cosine_sim(embedding, simple_centroid) vs cosine_sim(embedding, complex_centroid)
       -> binary decision + confidence -> 4-tier mapping (simple/mid/complex/expert)
       -> regex overrides (agentic, reasoning, vision, context window)
       -> model selection from tier (health-aware, with per-tier fallbacks)
```

**Live features:** Two-Tower analyzer (default), BERT analyzer, Matrix Factorization analyzer, Ensemble analyzer, Gemini analyzer. Factory pattern via `analyzer_factory.py`. Provider health monitoring with rolling-window scores. Per-tier fallback chains.

**Weaknesses:** Single-dimension classification, static centroids from 140 hand-crafted prototypes, no online learning, arbitrary confidence scaling (`confidence * 5`), no task-type awareness, general-purpose encoder not fine-tuned for complexity.

### Target Architecture (Paid Version)

```
prompt -> [Shared Encoder (ONNX)] -> 384-dim embedding
       -> [Multi-Dimensional Scorer]
           |-- Cognitive Depth (Linear 384->5, learned)
           |-- Domain Detector (Linear 384->12, multi-label)
           |-- Output Length Predictor
           +-- Structural Feature Extractor (20+ features)
       -> [Quality Prediction Network]
           +-- Per-model quality heads (MLP: 404->128->64->1 x N models)
       -> [Latency Predictor] (EMA + token-to-latency coefficient)
       -> [Pareto Optimizer] (quality x w_q - cost x w_c - latency x w_l)
       -> [Contextual Bandit (LinUCB)] (exploration-exploitation)
       -> model selection with explainability trace
```

### Key ML Improvements by Phase

**Phase 1 -- Learned Classifier (Planned Q2):**
- Replace static centroids with a learned Linear(404, 64, 3) head trained on prototype data + real feedback
- ONNX Runtime inference: 2x faster, 4x smaller
- Semantic cache with FAISS nearest-neighbor (0.95 threshold)

**Phase 2 -- Quality-Aware Routing (Planned Q3):**
- Multi-head quality prediction network trained on LLM-as-judge + user feedback
- Task-type detection (12 categories) replacing regex-based detection
- Pareto optimization across cost/quality/latency with user-configurable weights
- Anomaly detection via Mahalanobis distance for OOD prompts

**Phase 3 -- Adaptive & Personalized (Planned Q4):**
- LinUCB contextual bandit with per-arm ridge regression for exploration-exploitation
- Per-user routing heads fine-tuned on individual feedback (min 50 samples)
- Online centroid drift detection (KL divergence threshold)
- Automated weekly retraining pipeline with validation gates and atomic model swap

### Training Data Pipeline

```
Production Traffic -> [fact_requests SQLite] -> [Nightly ETL]
                                                      |
User Feedback -> [feedback table] -----------> [Training Dataset]
                                                      |
LLM-as-Judge (5% sample) -------------------> [Feature Store]
                                                      |
                                               [Retrain + Validate]
                                                      |
                                               [Atomic Model Swap]
```

---

## 5. Data & Analytics Strategy

### Current State

**Live:**
- SQLite + JSONL dual logging for all routed requests
- CLI dashboard with savings reports, cost breakdown, model utilization
- Per-request savings tracking in Supabase (hosted version)
- Savings-based invoice generation

**Gaps:**
- No user/project/team segmentation -- flat single-user stream
- No response quality tracking or feedback loop
- No session/conversation tracking
- No data retention policy -- unbounded growth
- Prompt text stored in plaintext -- PII risk
- No export capability

### Data Warehouse Schema (Planned)

**Star schema** with `fact_requests` (30+ columns including routing explainability, quality score, latency breakdown) + dimension tables (`dim_models`, `dim_users`, `dim_projects`) + materialized aggregations (`agg_hourly`, `agg_daily`).

### Analytics Features by Tier

| Feature | Pro | Team | Enterprise |
|---|---|---|---|
| Cost breakdown by model/day | Y | Y | Y |
| Cost forecast (14-day trend) | Y | Y | Y |
| Cache hit rate insights | Y | Y | Y |
| Latency percentile breakdown | Y | Y | Y |
| Routing decision explainability | Y | Y | Y |
| Savings tracking + invoices | Y | Y | Y |
| Per-user/team attribution | -- | Y | Y |
| A/B testing for routing strategies | -- | Y | Y |
| Anomaly detection + alerting | -- | Y | Y |
| Custom reporting + webhooks | -- | -- | Y |
| Industry benchmarking | -- | -- | Y |
| CSV/Parquet/JSON export | Pro (90d) | Team (1yr) | Unlimited |

### Privacy & Compliance

| Level | Prompt Storage | Use Case |
|---|---|---|
| `full` | Raw text | Self-hosted dev |
| `redacted` | PII patterns replaced | Production |
| `hash_only` | SHA-256 hash only | High compliance |
| `none` | Nothing stored | Maximum privacy |

Enterprise: SOC2-compatible audit log, GDPR right-to-erasure endpoint, data retention policies (30/90/730 days configurable), encryption at rest via SQLCipher.

---

## 6. Go-to-Market & Sales

### Ideal Customer Profiles

| Segment | Monthly LLM Spend | Buyer | Typical Savings Fee | Key Pain |
|---|---|---|---|---|
| **AI-native startup** (20-200 eng) | $5K-$80K | VP Eng / CTO | $200-$2K/mo | Costs scaling linearly, no governance |
| **Mid-market with coding tools** (200-2K emp) | $15K-$150K | Platform Eng lead | $1K-$5K/mo | No per-team visibility, CFO pressure |
| **Enterprise / regulated** (2K+ emp) | $100K-$1M+ | Head of AI/ML | $5K-$25K/mo | Compliance blocks cloud proxies |
| **AI consultancy / agency** | $3K-$30K | CTO | $150-$1.5K/mo | Can't attribute costs per client |

### Sales Channels (by Priority)

1. **Product-Led Growth (70% of Y1 revenue)** -- `pip install nadirclaw`, in-CLI savings CTAs, Pro trial auto-activated on first hosted proxy request
2. **Content & SEO** -- "reduce LLM API costs", integration guides, real savings data
3. **Direct Sales (Q3+)** -- first AE when MRR hits $15K, identify multi-user companies via opt-in telemetry
4. **Cloud Marketplaces** -- AWS/GCP Marketplace for procurement-friendly purchasing
5. **Partnerships** -- Cursor/Continue/Windsurf bundling, Anthropic/Google technical partnerships

### Conversion Funnel

```
PyPI Install -> Setup Wizard (2 min) -> First Routed Request -> First Savings Report
    -> "You saved $X this week" CLI prompt -> Try hosted proxy (Pro trial)
    -> First invoice: "You saved $200, we charged $59" -> Paid Pro
    -> 2nd user same company detected -> Team pitch -> Enterprise QBR
```

**Target conversion rates:** Install->Activation: 60% | Activation->Engaged: 40% | Engaged->Trial: 15% | Trial->Paid: 25% | Pro->Team: 15%

### Key Objection Handling

| Objection | Response |
|---|---|
| "LiteLLM is free" | LiteLLM is a gateway, not a router -- you still pick the model. NadirClaw classifies and picks for you. They're complementary. |
| "OpenRouter does this" | Your keys/prompts pass through their servers and they charge 5% on all spend. NadirClaw runs locally (free) or via hosted proxy with savings-based pricing -- you only pay when we save you money. |
| "We'll build internally" | Teams typically spend 2-3 eng-months and end up with static rules. Our ML classifier adapts to prompt patterns. |
| "Prices are dropping" | Volume grows faster (Jevons paradox). Enterprise AI spend grew $500M->$8.4B in 2 years despite 80% price drops. |
| "10ms overhead?" | That's 0.2-2% of total response time (500ms-5s). Meanwhile you save up to 40% on cost. |
| "What if it doesn't save me money?" | Then you pay $9/month. Our pricing is designed so you only pay meaningfully when we deliver real savings. |

---

## 7. Revenue Projections

Revenue projections use the savings-based model. Assumptions: average Pro user saves $400/mo (savings fee ~$109), average Team saves $8K/mo (savings fee ~$1,269), average Enterprise saves $50K+/mo (custom contracts ~$5K-$15K/mo).

### Conservative

| | Month 6 | Month 12 | Month 24 |
|---|---|---|---|
| OSS Users | 2,000 | 5,000 | 15,000 |
| Pro | 35 | 80 | 200 |
| Team | 0 | 8 | 25 |
| Enterprise | 0 | 1 | 5 |
| Avg Pro ARPU | $109 | $130 | $155 |
| Avg Team ARPU | -- | $1,300 | $1,500 |
| Avg Enterprise ARPU | -- | $5,000 | $8,000 |
| **MRR** | **$3.8K** | **$19.8K** | **$78.5K** |
| **ARR** | **$46K** | **$238K** | **$942K** |

### Moderate

| | Month 6 | Month 12 | Month 24 |
|---|---|---|---|
| OSS Users | 4,000 | 12,000 | 40,000 |
| Pro | 70 | 200 | 600 |
| Team | 3 | 20 | 60 |
| Enterprise | 0 | 3 | 12 |
| Avg Pro ARPU | $120 | $145 | $170 |
| Avg Team ARPU | $1,200 | $1,400 | $1,600 |
| Avg Enterprise ARPU | -- | $7,000 | $10,000 |
| **MRR** | **$12.0K** | **$77.0K** | **$218K** |
| **ARR** | **$144K** | **$924K** | **$2.6M** |

### Aggressive (viral moment + enterprise traction)

| | Month 6 | Month 12 | Month 24 |
|---|---|---|---|
| OSS Users | 10,000 | 35,000 | 120,000 |
| Pro | 180 | 600 | 2,000 |
| Team | 8 | 50 | 150 |
| Enterprise | 1 | 8 | 30 |
| Avg Pro ARPU | $130 | $160 | $190 |
| Avg Team ARPU | $1,300 | $1,500 | $1,800 |
| Avg Enterprise ARPU | $5,000 | $8,000 | $12,000 |
| **MRR** | **$38.8K** | **$235K** | **$710K** |
| **ARR** | **$466K** | **$2.8M** | **$8.5M** |

**Key insight:** The savings-based model has significantly higher revenue potential than flat fees because ARPU grows automatically as customers route more traffic. NRR is projected at 130-150% vs ~100% for flat-fee models.

Revenue mix by Month 24 (moderate): Pro 47% | Team 36% | **Enterprise 17%** (Enterprise share grows rapidly in Year 2+)

---

## 8. Key Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **Portkey adds ML routing to their now-OSS gateway** | High | Position as complementary ("use NadirClaw as routing brain behind Portkey"). Build first-class Portkey integration. |
| **Providers build native model routing** (OpenAI auto-downgrade) | High | Multi-provider routing remains valuable. Focus on cross-provider optimization. |
| **Coding agents build native routing** (Cursor/Claude Code) | Medium | Deepen integrations, offer features agents can't (cost tracking, team governance, custom training). |
| **RouteLLM/vLLM Semantic Router productize** | Medium | Move fast on feedback loops and quality prediction -- features academic projects won't build. |
| **Model price collapse eliminates routing value** | Low | Token prices fall but volume rises (Jevons paradox). The spread between tiers persists. Savings-based pricing means our revenue adjusts proportionally -- we don't overcharge in a low-spread world. |
| **LiteLLM adds intelligent routing** | Medium | NadirClaw already uses LiteLLM as a library. Differentiate on ML quality, not proxy features. |
| **Savings-share model under-monetizes high-volume users** | Medium | Enterprise tier has negotiated pricing. Team tier has lower share rates at higher volumes. Monitor ARPU and adjust thresholds if needed. |

---

## 9. Success Metrics

| Feature | KPI | Target |
|---|---|---|
| Adaptive Classifier | Misroute rate (flagged / total) | <5% after 2 weeks |
| Quality Scoring | Simple-tier responses rated adequate+ | >90% |
| Persistent Cache | Cache hit rate across restarts | >15% |
| Provider Health | Unhandled provider failures | <0.1% |
| Custom Rules | Pro users creating 1+ rule in 30 days | >60% |
| Multi-Tenant | Requests/sec per instance | >500 rps at <20ms p99 |
| Custom Training | Accuracy improvement vs default | >10% relative |
| Savings Tracking | Median monthly savings per Pro user | >$300 |
| **Pro Conversion** | **Free->Pro rate** | **>5%** |
| **Team Conversion** | **Pro->Team upgrade** | **>12%** |
| **Enterprise Conversion** | **Team->Enterprise upgrade** | **>15%** |
| **Net Revenue Retention** | **Monthly expansion** | **>130% NRR** |

### Milestone Targets

| Milestone | Target | Timeline |
|---|---|---|
| 1,000 OSS users | PLG validation | Month 3 |
| $10K MRR | Savings model validation | Month 6-8 |
| First enterprise deal | Upmarket proof | Month 10 |
| $50K MRR | Sustainable business | Month 14-16 |
| $100K MRR | Series A readiness | Month 18-24 |

---

## 10. Top 10 User Stories

1. **Team Lead**: "I want to see how much each developer spends on LLM calls so I can set per-person budgets."

2. **Developer**: "I want NadirClaw to learn from my corrections when it misroutes, so accuracy improves over time."

3. **Platform Engineer**: "I want to deploy NadirClaw as a centralized service with per-team API keys and RBAC."

4. **Security Officer**: "I want PII redacted before prompts reach third-party providers to comply with our data policy."

5. **Developer**: "I want to define a rule that code review prompts always use Claude Sonnet for consistent quality."

6. **Finance Manager**: "I want to see exactly how our savings-based billing is calculated -- benchmark cost vs routed cost for every request."

7. **ML Engineer**: "I want to train a custom classifier on our domain-specific labeled data for better routing accuracy."

8. **Ops Engineer**: "I want NadirClaw to detect degraded providers and reroute automatically -- zero downtime."

9. **Developer**: "I want NadirClaw to measure whether the cheaper model produced a good response, so I trust the routing."

10. **CTO**: "I want an audit trail of every routed LLM request -- who, which model, cost -- for SOC2 compliance."

---

## Immediate Next Steps (First 30 Days)

1. [ ] Ship hosted proxy (api.getnadir.com) for Pro users with savings tracking
2. [ ] Finalize Stripe integration with savings-based billing ($9/mo + 25%/10% share)
3. [ ] Add savings CTA to CLI (`nadirclaw report` -> "Save more with aggressive dedup -- try Pro")
4. [ ] Launch waitlist -> beta conversion for Pro tier
5. [ ] Publish integration guides: Cursor, Claude Code, Continue
6. [ ] Implement `POST /v1/feedback` endpoint + SQLite feedback table
7. [ ] Convert sentence transformer to ONNX Runtime
8. [ ] Add data retention + PII redaction configuration
9. [ ] Build Team tier savings billing (20%/8% aggregate model)
10. [ ] Implement opt-in telemetry for multi-user detection

---

## 11. Horizen Backend Assessment (Ground Truth)

> Based on deep technical review of all Horizen components (March 25, 2026). This section documents what actually exists in the production backend, what's reusable, and what blocks the paid launch.

### 11.1 P0 Blockers for Pro Launch

These must be resolved before any paid tier ships:

| Blocker | Current State | Required Work | Effort |
|---|---|---|---|
| **Savings billing not in Horizen** | `SavingsBillingService` exists only in getnadir.dev. Horizen has no per-request savings calculation despite `benchmark_model` being on the user session. | Port savings_tracking + savings_invoices tables + billing service to Horizen. Wire into completion flow. | 1-2 weeks |
| **No Stripe integration** | No Stripe code in Horizen. Only a SQL migration comment references it. | Build Stripe Billing integration for savings-based invoicing ($9/mo + tiered share). | 1 week |
| **No streaming in production endpoint** | `request.stream` is parsed but SSE/streaming not implemented in `production_completion.py`. The dead-code `completion.py` has streaming. | Implement streaming in the production endpoint. Essential for Cursor/Claude Code/coding agents. | 1 week |
| **Fund deduction not atomic** | `deduct_funds` does read-then-write (get balance -> update). Concurrent requests can cause double-spend. | Create Supabase RPC function for atomic deduction (like `check_and_reserve_budget`). | 2 days |
| **Deployment config bugs** | `app.yaml` has `CORS_ORIGINS: "*"` (crashes prod) and entrypoint `uvicorn main:app` (should be `app.main:app`). | Fix CORS and entrypoint. | 1 hour |
| **Dead code: completion.py** | Two competing endpoints at `/v1/chat/completions`. `production_completion.py` wins due to router inclusion order. `completion.py` is dead code. | Delete or archive `completion.py`. | 1 hour |

### 11.2 ML Stack: What Already Exists in Horizen (Not in Roadmap)

The Horizen backend has **significantly more ML infrastructure** than the roadmap acknowledges. Key discoveries:

#### 11 Analyzer Types (vs 1 in NadirClaw)

| Analyzer | How It Works | Port to NadirClaw? |
|---|---|---|
| **BinaryComplexityClassifier** | Fine-tuned DistilBERT (3-class: simple/medium/complex) with centroid fallback. 500 prototypes (vs NadirClaw's 140). k-means sub-clustering for complex tier. | **Yes (P0)** -- upgrades from binary to ternary |
| **ConfidenceAwareAnalyzer** | Cascade: binary classifier first (~12ms). If low confidence, escalates to Two-Tower (~15ms more). ~12ms for 80% of requests. | **Yes (P1)** -- should be Pro default |
| **Two-Tower Neural** | TF-IDF prompt tower + model embedding tower + dot-product. 8,001 training interactions. Top-1: 52%, Top-3: 97.5%. Includes uncertainty estimation head. | **Yes (Q3)** -- for Enterprise custom training |
| **EnhancedBERT (heuristic)** | 20+ structural features across 5 dimensions (patterns, linguistic, domain, structure, intent). No actual BERT -- pure heuristics. | **Extract features (Q2)** -- ready-made for "Multi-Dimensional Feature Extraction" |
| **Matrix Factorization** | OpenAI `text-embedding-3-small` (1536-dim) collaborative filtering. | **No** -- requires external API, not local-first |
| **Gemini Analyzer** | Delegates ranking to Gemini via LLMRanker. ~500ms+ latency. | **No for routing** -- possible as calibration oracle for LLM-as-judge |
| **Ensemble** | Weighted average of sub-analyzers. Conservative (min confidence). | **Yes** -- useful for A/B testing framework |
| **HybridFast** | Regex fast-path for 80-90% of prompts, Gemma-3 fallback. | **Template only** -- rules too simplistic |

#### Training Pipeline (in `need_to_remove/training/`)

A nearly complete training pipeline exists but was archived:
- `train_two_tower_pure.py` -- Performance-based training
- `train_complexity_aware_model.py` -- Complexity-aware training with cost penalties
- `train_lightweight_student.py` -- Teacher-student distillation
- `generate_enhanced_training_dataset.py` -- Data generation
- Training data in JSONL format with model scores

**Implication:** The roadmap's "Custom Classifier Training (Two-Tower)" planned for Q4 could be **accelerated to Q3** by recovering this pipeline.

#### Distillation Pipeline (partially built in Horizen services)

- `training_data_service.py` -- Collects prompt/response pairs from production
- `fine_tuning_service.py` -- Manages OpenAI fine-tuning jobs + local LoRA training
- `quality_monitor.py` -- Quality gate using embedding similarity
- `distillation_monitor.py` -- Background poller for active training jobs
- `routing_quality_tracker.py` -- Detects misroutes (same user, same prompt, different model within 60s)
- `cost_anomaly_service.py` -- Rolling average cost anomaly detection (2x threshold)

#### Other Reusable Components

| Component | Horizen | NadirClaw Gap |
|---|---|---|
| **Provider Health Monitor** | 4-component composite score (success 40%, latency P95 30%, error trend 20%, circuit state 10%) + zero-completion detection | NadirClaw has simpler version. Port composite scoring. |
| **Model Ranker** | Capability filtering (tool schema, extended thinking, JSON mode, function calling) + version-aware fuzzy matching + existing Pareto weights (`quality=0.55, cost=0.30, latency=0.15`) | Roadmap's Pareto Optimizer is an evolution of this. |
| **Confidence Calibration** | Linear rescaling per analyzer type. TODO for full Platt scaling. | **Missing from roadmap entirely.** Add as Q2 task. |
| **Model Registry** | 40+ models across all providers with performance-to-API mappings. | Essential infrastructure not mentioned in roadmap. |
| **Cluster Routing Policies** | Per-cluster model pinning (user-defined). | Maps to "Custom Routing Rules" -- more sophisticated than YAML rules. |
| **Response Healer** | Progressive JSON repair pipeline. | Already ported to NadirClaw. |

### 11.3 Architecture Assessment: Team/Enterprise Readiness

**Verdict: Core routing intelligence is production-quality. Business/governance layer is not ready.**

| Area | Status | Gap |
|---|---|---|
| ML routing intelligence | Production-quality | None -- this is the hard part, and it's done |
| Middleware (circuit breaker, health, rate limit) | Solid | Rate limiting is in-memory only (needs Redis for multi-instance) |
| Auth | Single-user only | No org context, no RBAC enforcement, no API key caching (2 Supabase queries per request) |
| Multi-tenancy | Tables exist, not wired | `organizations`, `organization_members` in schema but auth doesn't resolve org_id |
| Schemas | OpenAI-compatible with Nadir extensions | Missing multi-part content (vision), missing tool_calls in response, legacy field bloat |
| Observability | Good Prometheus coverage | No OpenTelemetry, no alerting rules, no Grafana dashboards |
| Scalability | ~50-100 concurrent requests/instance ceiling | Auth caching, async Supabase client, Redis for shared state needed for 500 rps target |

**Estimated effort to Team-ready: 8-10 engineering weeks (1 engineer full-time)**

### 11.4 Data Layer Critical Issues

| Issue | Severity | Details |
|---|---|---|
| **Schema divergence** | HIGH | 3 codebases define Supabase schema (Horizen, admin-dash, seed files) with conflicting `usage_events` definitions |
| **No org_id on fact tables** | HIGH | `usage_events` and `cost_usage` have `user_id` but no `organization_id`. Per-team queries require N+1 lookups. |
| **Cost split inaccuracy** | MEDIUM | `litellm_cost_callback.py` uses hardcoded 30/70 input/output split instead of actual per-token costs |
| **Prompts stored in plaintext** | HIGH | `usage_events.prompt` and `usage_events.response` store raw text. No PII handling. Blocks SOC2/GDPR. |
| **No data retention** | MEDIUM | No partitioning, no cleanup jobs, no TTL. Tables grow unboundedly. |
| **Triple-write no transaction** | MEDIUM | Request logs write to `usage_events`, `cost_usage`, and `prompts` tables independently. No idempotency key. |
| **Python-side aggregation** | MEDIUM | All analytics queries do `SELECT *` then aggregate in Python. `usage_stats_daily` materialized view exists in admin-dash migrations but is never called. |

### 11.5 Priority Migration Plan (Data)

```
Week 1:  Fix input/output cost split (use LiteLLM actual costs)
         Port savings_tracking + SavingsBillingService from getnadir.dev
         Add prompt_storage_mode to profiles + PII redaction
Week 2:  Add organization_id to usage_events + cost_usage
         Create feedback + quality_scores tables
         Activate usage_stats_daily aggregation (pg_cron)
Week 3:  Reconcile schema divergence (Horizen vs admin-dash)
         Create audit_log table (append-only)
         Add Prometheus savings/quality metrics
Week 4:  Data retention cleanup job
         Replace pickle-based embedding cache with FAISS + SQLite
         Add data export endpoints (CSV/JSON)
```

### 11.6 Revised Roadmap Recommendations

Based on the Horizen review, the following changes to the phased roadmap are recommended:

**Accelerate to Phase 1 (Q2):**
- Port DistilBERT 3-class classifier from Horizen (replaces binary centroids)
- Port ConfidenceAwareAnalyzer cascade pattern (Pro default)
- Extract 20+ structural features from EnhancedBERT analyzer (ready-made code)
- Add confidence calibration (Platt scaling) -- missing from original roadmap
- Port model registry (40+ models) -- essential infrastructure

**Accelerate to Phase 2 (Q3, was Q4):**
- Custom Classifier Training -- recover training pipeline from `need_to_remove/training/`
- Teacher-student distillation -- infrastructure exists in Horizen services
- Routing quality tracking -- `routing_quality_tracker.py` detects misroutes automatically

**New P0 items not in original roadmap:**
- Implement streaming in production endpoint (blocks coding agent use)
- Fix fund deduction atomicity (double-spend vulnerability)
- Fix deployment config (CORS + entrypoint bugs)
- Delete dead-code `completion.py`
- Auth caching layer (2 DB queries per request is a scalability ceiling)

**Downgrade/defer:**
- Contextual Bandit (LinUCB) stays Q4 -- no foundation exists
- Air-gapped installation stays Q4 -- low priority vs revenue features
- SDK/client libraries stay Q4 -- API is already OpenAI-compatible

---

## 12. Launch Content

All waitlist launch content for LinkedIn, X, Product Hunt, Hacker News, and waitlist email has been prepared in [LAUNCH_CONTENT.md](LAUNCH_CONTENT.md).

---

_This roadmap synthesizes research from market analysis, competitive intelligence, ML architecture review, data infrastructure assessment, sales strategy, product management, and a deep technical audit of the Horizen production backend. All recommendations are grounded in the current NadirClaw codebase (v0.13.0), the Horizen backend, and the live pricing at getnadir.dev. Revenue projections use the savings-based billing model._
