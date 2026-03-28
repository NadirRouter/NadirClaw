#!/usr/bin/env python3
"""
Benchmark: NadirClaw routing vs always Opus on 50 customer success tickets.

Models:
  complex  = claude-opus-4-20250514
  mid      = claude-sonnet-4-20250514
  simple   = claude-3-5-haiku-20241022

Baseline  = always Opus for everything.
LLM-as-judge on simple/mid routed results (complex skipped — same model).
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import litellm
litellm.suppress_debug_info = True
os.environ.setdefault("LITELLM_LOG", "ERROR")

import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
import warnings
warnings.filterwarnings("ignore")

# ── Model config ──────────────────────────────────────────────────────────
# Using Gemini models (Anthropic/OpenAI keys exhausted)
# gemini-2.5-pro  ≈ Opus-tier   ($1.25/$10 per 1M tok → expensive, most capable)
# gemini-2.5-flash ≈ Sonnet-tier ($0.15/$0.60 per 1M tok → mid-range)
# gemini-2.0-flash ≈ Haiku-tier  ($0.10/$0.40 per 1M tok → cheapest)
BASELINE = "gemini/gemini-2.5-pro"          # always-expensive baseline
MID_MODEL = "gemini/gemini-2.5-flash"
CHEAP_MODEL = "gemini/gemini-2.0-flash"

TIER_MODELS = {
    "simple": CHEAP_MODEL,
    "mid": MID_MODEL,
    "medium": MID_MODEL,
    "complex": BASELINE,
    "reasoning": BASELINE,
}

# ── System prompt (realistic CS agent) ────────────────────────────────────
SYSTEM_PROMPT = """You are an AI-powered Customer Success Agent for CloudStack Pro, an enterprise cloud infrastructure platform serving over 2,000 B2B customers across financial services, healthcare, SaaS, and e-commerce verticals.

## Your Role & Responsibilities
You handle Tier 1-3 support tickets across the full product surface:
- **Infrastructure**: Kubernetes clusters, VM provisioning, load balancers, auto-scaling groups, VPC networking, DNS management, CDN configuration
- **Database Services**: Managed PostgreSQL, MySQL, Redis, MongoDB Atlas integration, DynamoDB-compatible tables, connection pooling, read replicas, automated backups
- **Security & Compliance**: IAM roles/policies, SSO/SAML integration, SOC2 controls, HIPAA BAA, PCI-DSS scope, encryption at rest/in transit, audit logging, vulnerability scanning
- **Billing & Account**: Usage-based pricing, committed-use discounts, invoice disputes, payment method updates, plan upgrades/downgrades, enterprise contract amendments
- **API & Developer Tools**: REST API v3, GraphQL endpoint, SDK issues (Python, Node, Go, Java), webhook configuration, CI/CD integration, Terraform provider, CLI tools
- **Monitoring & Observability**: Built-in metrics, custom dashboards, alerting rules, log aggregation, distributed tracing, SLO/SLA tracking, incident management integration

## Response Guidelines
1. **Tone**: Professional, empathetic, solution-oriented. Mirror the customer's urgency level.
2. **Structure**: Lead with acknowledgment → diagnose → provide actionable steps → set expectations.
3. **Technical depth**: Match the customer's technical sophistication. Use code snippets, CLI commands, or API examples when appropriate.
4. **Escalation**: If the issue requires engineering intervention, document clearly what you've tried and what needs escalation.
5. **SLA awareness**: P1 (production down) = 15min response, P2 (degraded) = 1hr, P3 (question) = 4hr, P4 (feature request) = 24hr.
6. **Knowledge cutoff**: You have access to documentation up to v4.2.1 (current release). For features in beta, note they may change.
7. **Data sensitivity**: Never expose other customers' data, internal pricing logic, or infrastructure details beyond what's documented.
8. **Follow-up**: Always suggest next steps and offer proactive recommendations when you spot configuration issues.

## Internal Tools Available
- `lookup_account(account_id)` — customer details, plan, usage, billing history
- `check_service_status(service, region)` — real-time service health
- `get_recent_incidents(hours=24)` — recent platform incidents
- `search_kb(query)` — internal knowledge base search
- `create_escalation(priority, summary)` — escalate to engineering
- `check_quota(account_id, resource)` — resource quota and usage
- `get_audit_log(account_id, hours=48)` — recent account activity

## Common Workflows You Handle
- Password resets and MFA recovery
- SSL certificate provisioning and renewal
- Database migration assistance
- Kubernetes cluster troubleshooting (CrashLoopBackOff, OOMKilled, ImagePullBackOff)
- Cost optimization recommendations
- API rate limit adjustments
- Cross-region replication setup
- Compliance documentation requests
- Integration debugging (Terraform, CI/CD, webhooks)
- Incident communication and post-mortem sharing"""

# ── 50 Customer Success Tickets ───────────────────────────────────────────
# (expected_tier, ticket_prompt)
TICKETS = [
    # ── SIMPLE (1-15): Quick lookups, password resets, status checks ───
    ("simple", "Hi, what's the current status of the us-east-1 region? We're seeing slow responses."),
    ("simple", "How do I reset my MFA device? I got a new phone."),
    ("simple", "What are your support hours? Do you have 24/7 support on the Pro plan?"),
    ("simple", "Can you tell me what plan we're on and when our contract renews?"),
    ("simple", "Where can I find the API documentation for your REST API v3?"),
    ("simple", "How do I add a new team member to our account with read-only access?"),
    ("simple", "What's the maximum file upload size for your object storage?"),
    ("simple", "Is there currently any scheduled maintenance planned for this week?"),
    ("simple", "How do I download our last 3 invoices?"),
    ("simple", "What regions do you support for database hosting?"),
    ("simple", "How do I rotate my API keys without downtime?"),
    ("simple", "Can you confirm our account is SOC2 compliant?"),
    ("simple", "What's the difference between your Standard and Pro plans?"),
    ("simple", "How do I enable email notifications for billing alerts?"),
    ("simple", "What's your uptime SLA for the Enterprise plan?"),

    # ── MEDIUM (16-35): Troubleshooting, config help, moderate analysis ──
    ("medium", "Our PostgreSQL read replica is lagging behind the primary by over 30 seconds during peak hours (2-4pm EST). We're on the db.r5.xlarge instance class with about 500 write transactions/sec. What can we do to reduce replication lag?"),
    ("medium", "I'm trying to set up SAML SSO with Okta but getting a 'Invalid SAML Response' error. Here's what I've configured:\n- Entity ID: https://cloudstack.pro/saml/metadata\n- ACS URL: https://cloudstack.pro/saml/acs\n- Name ID: email\nThe error occurs after the user authenticates in Okta. What am I missing?"),
    ("medium", "We need to set up automated database backups that are retained for 90 days for HIPAA compliance. Currently our backups only go back 7 days. Can you walk me through configuring this and confirm it meets HIPAA requirements?"),
    ("medium", "Our Kubernetes pods keep getting OOMKilled. Here's the deployment spec:\n```yaml\nresources:\n  requests:\n    memory: '256Mi'\n  limits:\n    memory: '512Mi'\n```\nThe application is a Java Spring Boot service. We've set -Xmx to 450m. What's going wrong?"),
    ("medium", "We're seeing intermittent 502 errors on our load balancer. It happens about 5% of the time, mostly during deployments. We're using rolling updates with maxUnavailable: 25%. How can we achieve zero-downtime deployments?"),
    ("medium", "I need to migrate our production database from us-east-1 to eu-west-1 for GDPR compliance. It's a 500GB PostgreSQL database with about 200 active connections. What's the best migration strategy with minimal downtime?"),
    ("medium", "Our Terraform deployment is failing with 'Error: Provider produced inconsistent result after apply' on the cloudstack_kubernetes_cluster resource. We're using provider version 2.3.1. Here's our config:\n```hcl\nresource \"cloudstack_kubernetes_cluster\" \"main\" {\n  name = \"production\"\n  version = \"1.28\"\n  node_count = 5\n  node_type = \"compute-optimized-8\"\n}\n```"),
    ("medium", "We want to implement a blue-green deployment strategy for our application. We have 3 services behind your load balancer. Can you explain how to set this up with your platform and what the switching mechanism looks like?"),
    ("medium", "Our webhook endpoint is receiving duplicate events. We've noticed the same event ID arriving 2-3 times within a 60-second window. Is this expected behavior? How should we handle idempotency?"),
    ("medium", "We need to set up cross-region replication for our Redis cluster for disaster recovery. Primary is in us-east-1, and we want a replica in us-west-2. What's the expected replication lag and does it support automatic failover?"),
    ("medium", "Our API is getting rate limited at 1000 req/min but we need 5000 req/min for our batch processing job that runs nightly. Can we get a temporary rate limit increase? Or is there a better way to handle batch operations?"),
    ("medium", "I'm seeing high p99 latency (>2s) on our GraphQL endpoint but p50 is fine (200ms). We suspect it's related to N+1 queries. Can you help me set up distributed tracing to identify the bottleneck?"),
    ("medium", "We need to set up a VPN connection between our on-premises data center and our CloudStack VPC. We have a Cisco ASA on our end. What are the IPsec parameters we need to configure?"),
    ("medium", "Our auto-scaling group isn't scaling up fast enough during traffic spikes. Current config: min=2, max=10, target CPU=70%, cooldown=300s. We get traffic spikes that 10x in under 2 minutes. How should we tune this?"),
    ("medium", "We want to implement mutual TLS (mTLS) between our microservices running on your Kubernetes platform. What's the recommended approach? Do you support Istio service mesh or do you have a native solution?"),
    ("medium", "We're getting 'connection pool exhausted' errors on our managed PostgreSQL. We have 3 application servers each configured with a pool size of 50, and our database plan supports 100 max connections. How should we restructure this?"),
    ("medium", "Can you help me set up a CI/CD pipeline using your CLI tools? We use GitHub Actions and want to deploy to staging on PR merge and production on tag push. We need approval gates for production."),
    ("medium", "Our CDN cache hit ratio dropped from 92% to 45% after we deployed a new version last week. We're using versioned assets with content hashing. What could cause this and how do we debug it?"),
    ("medium", "We need to implement IP allowlisting for our API but we have 200+ office locations with dynamic IPs. Is there a way to do this with your platform that doesn't require constant updates?"),
    ("medium", "Our monitoring alerts are too noisy — we're getting 50+ alerts per day and most are false positives. Can you help us tune our alerting rules? We're using your built-in monitoring with Slack integration."),

    # ── COMPLEX (36-50): Architecture, multi-system, incident response ──
    ("complex", "We're experiencing a production outage affecting our payment processing service. Our Kubernetes cluster in us-east-1 shows multiple pods in CrashLoopBackOff state. The service connects to a PostgreSQL database which is showing 100% CPU. Our error logs show 'FATAL: too many connections' mixed with 'connection refused' errors. We have 50,000 active users unable to complete transactions. This is P1 — we need immediate help.\n\nCluster info:\n- 12 nodes, 8 running payment-svc pods\n- HPA target: 80% CPU, currently at 95%\n- Database: db.r5.4xlarge, 100 max connections\n- Last deployment: 2 hours ago (rollback attempted, didn't help)\n- PgBouncer in front of DB with pool_mode=transaction, default_pool_size=25"),
    ("complex", "We're planning a major architecture migration. Currently we run a monolithic Django application on 4 EC2-equivalent VMs behind a load balancer with a single PostgreSQL database (2TB). We want to decompose into microservices on Kubernetes.\n\nOur services would be:\n1. User service (auth, profiles) — ~500 req/s\n2. Product catalog — ~2000 req/s, heavy reads\n3. Order processing — ~200 req/s, transactional\n4. Payment gateway integration — ~100 req/s, PCI scope\n5. Notification service — ~1000 events/s, async\n6. Analytics pipeline — ~5000 events/s, eventual consistency OK\n\nWe need: a phased migration plan, data decomposition strategy, service mesh recommendation, CI/CD pipeline design, observability stack, and cost estimate comparing current vs target architecture. Our budget for the migration is $200K over 6 months."),
    ("complex", "We need to design a disaster recovery strategy that achieves RPO < 1 minute and RTO < 5 minutes across 3 regions (us-east-1, eu-west-1, ap-southeast-1). Our stack includes:\n- 5 Kubernetes clusters (200+ services)\n- 12 PostgreSQL databases (largest is 5TB)\n- 3 Redis clusters (session store, cache, pub/sub)\n- Object storage (50TB)\n- Message queue (processing 10K msgs/sec)\n\nWe currently have no DR strategy and our compliance audit is in 6 weeks. We need a detailed implementation plan with your platform's capabilities, including automated failover, data sync strategies, and testing procedures."),
    ("complex", "We suspect we've had a security breach. Our audit logs show an IAM user 'deploy-bot' making API calls from an IP address (203.0.113.42) that isn't in our known IP ranges. The calls include:\n- ListAllUsers (succeeded)\n- GetDatabaseCredentials for prod-db (succeeded)\n- CreateAPIKey with admin scope (succeeded)\n- ModifySecurityGroup to allow 0.0.0.0/0 on port 22 (succeeded)\n\nThis happened between 2:00 AM and 2:47 AM UTC today. The deploy-bot service account was last legitimately used 3 days ago.\n\nWe need: immediate containment steps, forensic investigation guidance, impact assessment, remediation plan, and help preparing an incident report for our SOC2 auditor. We're also a HIPAA-covered entity with PHI in the affected database."),
    ("complex", "We want to implement a multi-tenant architecture on your platform for our SaaS product. We currently have 500 tenants and expect to grow to 5,000 within 18 months. Requirements:\n- Data isolation (some tenants require dedicated databases for compliance)\n- Per-tenant resource quotas and rate limiting\n- Tenant-aware routing at the load balancer level\n- Shared infrastructure for small tenants, dedicated for enterprise\n- Per-tenant billing and usage metering\n- Tenant provisioning API (new tenant live in < 5 minutes)\n- Cross-tenant analytics (aggregated, anonymized)\n\nOur stack: React frontend, Node.js API layer, PostgreSQL (with row-level security currently), Redis for caching. Budget: want to keep infrastructure cost under $50/tenant/month for shared-tier tenants."),
    ("complex", "We need to achieve PCI-DSS Level 1 compliance for our payment processing platform running on CloudStack. Our current architecture:\n- Frontend: React app on CDN\n- API: 6 Node.js microservices on Kubernetes\n- Database: PostgreSQL with card data (currently stored as encrypted fields)\n- Message queue for async processing\n- Third-party integrations: Stripe, PayPal, bank ACH\n\nWe need a gap analysis against PCI-DSS v4.0 requirements, a remediation plan covering: network segmentation (CDE isolation), encryption standards, access controls, logging/monitoring, vulnerability management, and a penetration testing schedule. Our QSA assessment is in 4 months."),
    ("complex", "We're building a real-time data pipeline on your platform that needs to process 500K events/second from IoT devices with end-to-end latency under 100ms. The pipeline:\n1. Ingest: MQTT → message broker\n2. Stream processing: windowed aggregations (1min, 5min, 1hr)\n3. Enrichment: join with reference data (50GB, updated daily)\n4. Anomaly detection: ML model inference (TensorFlow Serving)\n5. Storage: time-series DB for hot data (30 days), object storage for cold\n6. Real-time dashboards: WebSocket push to 10K concurrent viewers\n\nWhat infrastructure do you recommend? We need detailed sizing, cost projections, and architectural guidance. Current monthly budget: $80K."),
    ("complex", "We need to implement a comprehensive observability strategy for our microservices architecture (45 services, 200+ pods). Current problems:\n- No distributed tracing — can't debug cross-service issues\n- Metrics are siloed per service, no unified view\n- Log aggregation exists but search is slow (>30s for queries)\n- Alert fatigue: 200+ alerts/day, <5% actionable\n- MTTR for incidents is 4+ hours\n- No SLO tracking\n\nWe want to achieve: <15min MTTR, <10 actionable alerts/day, SLO-based alerting, full request tracing, unified dashboards per team, and automated runbooks for common issues. What's your recommended observability stack and implementation plan?"),
    ("complex", "Our application performance has degraded significantly over the past month. P99 latency went from 500ms to 8 seconds. We've already checked:\n- CPU utilization: 45% average (not the bottleneck)\n- Memory: 60% utilized, no OOM events\n- Database: query performance looks normal, connection count stable\n- Network: no packet loss, bandwidth well within limits\n\nBut we've noticed:\n- GC pause times increased from 50ms to 800ms\n- Thread pool utilization went from 30% to 95%\n- Connection pool wait times spiked\n- Heap dumps show growing number of unreleased HTTP client connections\n\nWe're running Java 17 on Kubernetes, 8 pods with 4GB heap each. The issue correlates with when we integrated a new third-party API (inventory service) that sometimes responds in 30+ seconds.\n\nNeed: root cause analysis, immediate mitigation, and long-term architecture recommendations."),
    ("complex", "We're designing a global content delivery architecture for our video streaming platform. Requirements:\n- 4K video streaming to 1M+ concurrent users\n- Content in 3 tiers: live (ultra-low latency <2s), VOD (standard), and user-generated (variable quality)\n- DRM integration (Widevine + FairPlay)\n- Adaptive bitrate streaming (HLS + DASH)\n- Geographic restrictions for licensed content\n- Real-time analytics: viewer count, quality metrics, engagement\n- Cost optimization: edge caching strategy, origin shield\n\nCurrent infrastructure handles 100K concurrent users. We need to 10x capacity in 3 months for a major content launch. What's the architecture and migration plan using your CDN and compute services?"),
    ("complex", "We need help designing a zero-trust security architecture for our platform. We're a healthcare company processing PHI across multiple services. Requirements:\n- Every service-to-service call must be authenticated and authorized\n- Network policies: deny-all default, explicit allow rules\n- Identity-based access (not network-based) for all resources\n- Continuous verification (not just at connection time)\n- Data classification and DLP policies\n- Encrypted everywhere (in transit, at rest, in use where possible)\n- Privileged access management for operations team\n- Compliance: HIPAA, SOC2 Type II, HITRUST\n\nCurrent state: flat network, shared service accounts, SSH key-based access, no service mesh. 30 microservices, 5 databases, 3 message queues."),
    ("complex", "We want to implement a sophisticated cost optimization strategy. Our monthly CloudStack bill is $450K and the CFO wants it under $300K without sacrificing performance or reliability. Current usage:\n- 150 Kubernetes nodes (mix of compute and memory optimized)\n- 25 database instances (PostgreSQL, Redis, MongoDB)\n- 50TB object storage with 10TB/day egress\n- 3 regions for redundancy (but traffic is 80% us-east-1)\n- Dev/staging environments running 24/7 (same config as prod)\n- No reserved capacity, everything on-demand\n- Auto-scaling configured but rarely triggers (conservative thresholds)\n\nNeed: detailed analysis of waste, rightsizing recommendations, reserved capacity strategy, environment scheduling, architecture changes for cost efficiency, and a month-by-month reduction plan."),
    ("complex", "We're implementing a data sovereignty solution for our global SaaS platform. We have customers in EU (GDPR), Brazil (LGPD), India (DPDPA), and California (CCPA). Requirements:\n- Customer data must reside in the customer's declared region\n- Cross-border data transfers must comply with applicable frameworks (SCCs, adequacy decisions)\n- Right to deletion must propagate across all systems within 72 hours\n- Data processing records must be maintained per Article 30\n- Consent management integration\n- Audit trail for all data access\n- Breach notification workflow (72hr GDPR, varies by jurisdiction)\n\nOur current architecture is US-only with a single PostgreSQL database. We need a migration plan to a multi-region, data-sovereign architecture."),
    ("complex", "We're seeing cascading failures across our microservices during peak traffic. The pattern:\n1. Inventory service slows down (p99 goes from 200ms to 10s)\n2. Order service threads get blocked waiting for inventory\n3. Order service health checks fail, pods restart\n4. Cart service can't reach order service, starts failing\n5. Frontend gets 503s, users retry, amplifying load\n6. Full cascade in ~3 minutes, recovery takes 20+ minutes\n\nThis has happened 4 times in the past month, always during peak hours (11am-1pm EST). We need:\n- Immediate fixes (circuit breakers, bulkheads, timeouts)\n- Architecture review for resilience patterns\n- Load testing strategy to validate fixes\n- Chaos engineering plan\n- Runbook for when cascades start\n\nOur stack: 12 services on Kubernetes, gRPC between services, no service mesh currently, PostgreSQL + Redis."),
    ("complex", "We need to build a comprehensive API gateway strategy. We currently have 30+ microservices each exposing their own endpoints directly. Problems:\n- No unified rate limiting (each service implements its own)\n- Authentication is duplicated across services\n- No request/response transformation\n- API versioning is inconsistent\n- No developer portal or API documentation aggregation\n- GraphQL federation attempted but abandoned\n- Partner integrations need different auth (OAuth2, API keys, mTLS)\n\nWe need: gateway architecture recommendation, migration plan from direct service exposure, unified auth strategy, rate limiting design, API versioning strategy, developer portal setup, and analytics/monitoring for API usage. We serve 500 API consumers (mix of internal teams, partners, and public developers)."),
]

# ── API key resolution ────────────────────────────────────────────────────
def _load_api_keys():
    """Load API keys from Horizen .env.local into environment."""
    env_path = os.path.join(os.path.dirname(__file__), "..", "Horizen", ".env.local")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                if k in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
                    os.environ.setdefault(k, v.strip())

_load_api_keys()


async def call_model(model: str, system: str, user_prompt: str, max_tokens: int = 2048) -> dict:
    """Call a model, return result with timing and cost."""
    start = time.time()
    try:
        # For gemini-2.5-pro, disable thinking to avoid token waste
        extra = {}
        if "2.5-pro" in model:
            extra["thinking"] = {"type": "disabled"}

        response = await asyncio.to_thread(
            litellm.completion,
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
            timeout=60,
            **extra,
        )
        elapsed = time.time() - start
        output = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            # Fallback: manual cost calc for Gemini models
            COST_TABLE = {
                "gemini/gemini-2.0-flash": (0.10, 0.40),      # per 1M tok
                "gemini/gemini-2.5-flash": (0.15, 0.60),
                "gemini/gemini-2.5-pro": (1.25, 10.00),
            }
            rates = COST_TABLE.get(model, (0.10, 0.40))
            cost = (prompt_tokens * rates[0] + completion_tokens * rates[1]) / 1_000_000
        return {
            "success": True,
            "model": model,
            "output": output,
            "cost": cost,
            "latency_ms": int(elapsed * 1000),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "success": False,
            "model": model,
            "output": "",
            "cost": 0.0,
            "latency_ms": int(elapsed * 1000),
            "error": str(e)[:200],
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }


async def classify_prompt(prompt: str) -> dict:
    """Classify a prompt using the cascade classifier."""
    os.environ["NADIRCLAW_CLASSIFIER"] = "cascade"
    from nadirclaw.classifier import get_classifier

    classifier = get_classifier()
    start = time.time()
    result = classifier.classify(prompt)
    classify_ms = int((time.time() - start) * 1000)

    if len(result) == 2:
        is_complex, conf = result
        tier = "complex" if is_complex else "simple"
        confidence = conf
    else:
        tier, confidence, meta = result
        # no-op

    model = TIER_MODELS.get(tier, MID_MODEL)
    return {
        "tier": tier,
        "confidence": round(confidence, 3),
        "model": model,
        "classify_ms": classify_ms,
    }


async def judge_quality(prompt: str, system: str, response_a: str, response_b: str) -> dict:
    """LLM-as-judge: compare NadirClaw response vs Opus baseline.
    Returns score 1-5 for each and a verdict."""
    judge_prompt = f"""You are an expert quality evaluator for customer support responses.

A customer sent this ticket:
<ticket>
{prompt}
</ticket>

The system prompt for the agent was a detailed CloudStack Pro customer success agent prompt.

Two AI agents responded. Rate EACH response on a 1-5 scale across these dimensions:
1. **Accuracy** — Is the information correct and relevant?
2. **Completeness** — Does it fully address the customer's question?
3. **Actionability** — Does it give clear, actionable steps?
4. **Tone** — Is it professional, empathetic, and appropriate?
5. **Efficiency** — Is it concise without being too terse?

<response_a>
{response_a[:3000]}
</response_a>

<response_b>
{response_b[:3000]}
</response_b>

Respond in this exact JSON format (no markdown, no explanation outside the JSON):
{{"response_a": {{"accuracy": N, "completeness": N, "actionability": N, "tone": N, "efficiency": N, "overall": N}}, "response_b": {{"accuracy": N, "completeness": N, "actionability": N, "tone": N, "efficiency": N, "overall": N}}, "verdict": "A_better" | "B_better" | "tie", "reasoning": "one sentence"}}"""

    try:
        response = await asyncio.to_thread(
            litellm.completion,
            model=MID_MODEL,  # Use Sonnet as judge (cost-efficient)
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=500,
            temperature=0.0,

        )
        text = response.choices[0].message.content or ""
        # Extract JSON
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(text)
        return result
    except Exception as e:
        return {"error": str(e)[:200], "verdict": "error"}


async def run_benchmark():
    print("=" * 80)
    print("  NadirClaw CS Ticket Benchmark — 50 Prompts")
    print(f"  Baseline: Always {BASELINE}")
    print(f"  Routing:  simple→2.0-flash, mid→2.5-flash, complex→2.5-pro")
    print(f"  System prompt: {len(SYSTEM_PROMPT)} chars")
    print("=" * 80)
    print()

    results = []
    judge_results = []
    total = len(TICKETS)

    for i, (expected_tier, ticket) in enumerate(TICKETS, 1):
        short = ticket[:80].replace("\n", " ")
        print(f"[{i}/{total}] ({expected_tier.upper()}) {short}...")

        # 1. Classify
        classification = await classify_prompt(ticket)
        routed_tier = classification["tier"]
        routed_model = classification["model"]
        confidence = classification["confidence"]
        classify_ms = classification["classify_ms"]

        correct = (routed_tier == expected_tier) or (
            routed_tier in ("simple", "mid", "medium") and expected_tier in ("simple", "mid", "medium") and routed_model == TIER_MODELS.get(expected_tier, MID_MODEL)
        )

        # 2. Call NadirClaw-routed model
        nadir_result = await call_model(routed_model, SYSTEM_PROMPT, ticket)

        # 3. If routed model != Opus, also call Opus for baseline + judging
        if routed_model == BASELINE:
            opus_result = nadir_result.copy()  # Same model, reuse
            needs_judge = False
        else:
            opus_result = await call_model(BASELINE, SYSTEM_PROMPT, ticket)
            needs_judge = True

        entry = {
            "index": i,
            "expected_tier": expected_tier,
            "routed_tier": routed_tier,
            "routed_model": routed_model.split("/")[-1],
            "confidence": confidence,
            "classify_ms": classify_ms,
            "correct_route": correct,
            "nadir": {
                "success": nadir_result["success"],
                "cost": nadir_result["cost"],
                "latency_ms": nadir_result["latency_ms"],
                "prompt_tokens": nadir_result["prompt_tokens"],
                "completion_tokens": nadir_result["completion_tokens"],
                "output": nadir_result["output"],
            },
            "opus": {
                "success": opus_result["success"],
                "cost": opus_result["cost"],
                "latency_ms": opus_result["latency_ms"],
                "prompt_tokens": opus_result["prompt_tokens"],
                "completion_tokens": opus_result["completion_tokens"],
                "output": opus_result["output"],
            },
            "ticket": ticket,
        }

        # Status line
        route_tag = "OK" if correct else "MISROUTE"
        if nadir_result["success"]:
            saved = opus_result["cost"] - nadir_result["cost"] if opus_result["success"] else 0
            print(f"  Route: {routed_tier} -> {routed_model.split('/')[-1]} (conf={confidence:.2f}) [{route_tag}]")
            print(f"  Nadir: ${nadir_result['cost']:.4f} ({nadir_result['latency_ms']}ms, {nadir_result['completion_tokens']} tok)")
            if routed_model != BASELINE:
                print(f"  Opus:  ${opus_result['cost']:.4f} ({opus_result['latency_ms']}ms, {opus_result['completion_tokens']} tok)")
                if saved > 0:
                    print(f"  Saved: ${saved:.4f} ({saved/opus_result['cost']*100:.1f}%)" if opus_result["cost"] > 0 else f"  Saved: ${saved:.4f}")
            else:
                print(f"  (Same model — no comparison needed)")
        else:
            print(f"  Route: {routed_tier} -> {routed_model.split('/')[-1]} [{route_tag}]")
            print(f"  ERROR: {nadir_result.get('error', 'unknown')[:100]}")

        # 4. Judge quality (only for non-Opus routes with successful responses)
        if needs_judge and nadir_result["success"] and opus_result["success"]:
            judge = await judge_quality(ticket, SYSTEM_PROMPT, nadir_result["output"], opus_result["output"])
            entry["judge"] = judge
            judge_results.append({
                "index": i,
                "tier": expected_tier,
                "routed_model": routed_model.split("/")[-1],
                "judge": judge,
            })
            verdict = judge.get("verdict", "error")
            a_score = judge.get("response_a", {}).get("overall", "?")
            b_score = judge.get("response_b", {}).get("overall", "?")
            print(f"  Judge: NadirClaw={a_score}/5, Opus={b_score}/5 → {verdict}")

        results.append(entry)
        print()

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 80)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r["nadir"]["success"] and r["opus"]["success"]]
    if not successful:
        print("\nNo successful results to analyze.")
        return

    # Routing accuracy
    correct_count = sum(1 for r in results if r["correct_route"])
    print(f"\n  Routing Accuracy: {correct_count}/{total} ({correct_count/total*100:.1f}%)")
    for tier in ["simple", "medium", "complex"]:
        tier_results = [r for r in results if r["expected_tier"] == tier]
        tier_correct = sum(1 for r in tier_results if r["correct_route"])
        print(f"    {tier:8s}: {tier_correct}/{len(tier_results)} correct")

    # Cost comparison
    # For complex-routed (Opus→Opus), opus cost = nadir cost, so net effect is $0 saved
    non_opus = [r for r in successful if r["routed_model"] != "gemini-2.5-pro"]
    total_opus_cost = sum(r["opus"]["cost"] for r in successful)
    total_nadir_cost = sum(r["nadir"]["cost"] for r in successful)
    total_saved = total_opus_cost - total_nadir_cost

    print(f"\n  Cost Comparison ({len(successful)} successful prompts):")
    print(f"    Always Opus:   ${total_opus_cost:.4f}")
    print(f"    NadirClaw:     ${total_nadir_cost:.4f}")
    if total_opus_cost > 0:
        print(f"    Total Saved:   ${total_saved:.4f} ({total_saved/total_opus_cost*100:.1f}%)")

    # Per-tier breakdown
    print(f"\n  Per-Tier Cost Breakdown:")
    for tier in ["simple", "medium", "complex"]:
        tier_s = [r for r in successful if r["expected_tier"] == tier]
        if tier_s:
            t_opus = sum(r["opus"]["cost"] for r in tier_s)
            t_nadir = sum(r["nadir"]["cost"] for r in tier_s)
            t_saved = t_opus - t_nadir
            pct = f"{t_saved/t_opus*100:.1f}%" if t_opus > 0 else "n/a"
            print(f"    {tier:8s}: Opus ${t_opus:.4f} → Nadir ${t_nadir:.4f} (saved ${t_saved:.4f}, {pct})")

    # Latency
    avg_opus_lat = sum(r["opus"]["latency_ms"] for r in successful) / len(successful)
    avg_nadir_lat = sum(r["nadir"]["latency_ms"] for r in successful) / len(successful)
    print(f"\n  Average Latency:")
    print(f"    Opus:      {avg_opus_lat:.0f}ms")
    print(f"    NadirClaw: {avg_nadir_lat:.0f}ms (+ {sum(r['classify_ms'] for r in successful)/len(successful):.0f}ms classify overhead)")

    # Model distribution
    print(f"\n  Models Used by NadirClaw:")
    model_counts = {}
    for r in results:
        m = r["routed_model"]
        model_counts[m] = model_counts.get(m, 0) + 1
    for m, c in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"    {m:45s}: {c}x")

    # LLM Judge summary
    judged = [j for j in judge_results if "error" not in j["judge"]]
    if judged:
        print(f"\n  LLM-as-Judge Quality Assessment ({len(judged)} judged, complex skipped):")
        a_wins = sum(1 for j in judged if j["judge"].get("verdict") == "A_better")
        b_wins = sum(1 for j in judged if j["judge"].get("verdict") == "B_better")
        ties = sum(1 for j in judged if j["judge"].get("verdict") == "tie")
        print(f"    NadirClaw wins: {a_wins}  |  Opus wins: {b_wins}  |  Ties: {ties}")

        avg_a = sum(j["judge"].get("response_a", {}).get("overall", 0) for j in judged) / len(judged)
        avg_b = sum(j["judge"].get("response_b", {}).get("overall", 0) for j in judged) / len(judged)
        print(f"    Avg quality: NadirClaw={avg_a:.2f}/5, Opus={avg_b:.2f}/5")

        # Per dimension
        dims = ["accuracy", "completeness", "actionability", "tone", "efficiency"]
        print(f"\n    Per-Dimension Averages:")
        print(f"    {'Dimension':20s} {'NadirClaw':>10s} {'Opus':>10s} {'Delta':>10s}")
        print(f"    {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
        for dim in dims:
            a_avg = sum(j["judge"].get("response_a", {}).get(dim, 0) for j in judged) / len(judged)
            b_avg = sum(j["judge"].get("response_b", {}).get(dim, 0) for j in judged) / len(judged)
            delta = a_avg - b_avg
            sign = "+" if delta >= 0 else ""
            print(f"    {dim:20s} {a_avg:10.2f} {b_avg:10.2f} {sign}{delta:9.2f}")

        # Per-tier judge breakdown
        print(f"\n    Quality by Routed Tier:")
        for tier in ["simple", "medium"]:
            tier_j = [j for j in judged if j["tier"] == tier]
            if tier_j:
                a_avg = sum(j["judge"].get("response_a", {}).get("overall", 0) for j in tier_j) / len(tier_j)
                b_avg = sum(j["judge"].get("response_b", {}).get("overall", 0) for j in tier_j) / len(tier_j)
                wins_a = sum(1 for j in tier_j if j["judge"].get("verdict") == "A_better")
                wins_b = sum(1 for j in tier_j if j["judge"].get("verdict") == "B_better")
                t = sum(1 for j in tier_j if j["judge"].get("verdict") == "tie")
                print(f"      {tier:8s} ({len(tier_j)} prompts): Nadir={a_avg:.2f}/5 vs Opus={b_avg:.2f}/5  (Nadir wins {wins_a}, Opus wins {wins_b}, ties {t})")

    # Misrouted prompts
    misrouted = [r for r in results if not r["correct_route"]]
    if misrouted:
        print(f"\n  Misrouted Prompts ({len(misrouted)}):")
        for r in misrouted:
            print(f"    #{r['index']}: expected={r['expected_tier']}, got={r['routed_tier']}")

    # Save everything
    output_file = os.path.join(os.path.dirname(__file__), "benchmark_cs50_results.json")
    save_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_prompts": total,
            "successful": len(successful),
            "system_prompt_chars": len(SYSTEM_PROMPT),
            "models": {"simple": CHEAP_MODEL, "mid": MID_MODEL, "complex": BASELINE},
            "baseline": BASELINE,
        },
        "summary": {
            "routing_accuracy": f"{correct_count}/{total}",
            "total_opus_cost": round(total_opus_cost, 6),
            "total_nadir_cost": round(total_nadir_cost, 6),
            "total_saved": round(total_saved, 6),
            "savings_pct": round(total_saved / total_opus_cost * 100, 1) if total_opus_cost > 0 else 0,
            "avg_opus_latency_ms": round(avg_opus_lat),
            "avg_nadir_latency_ms": round(avg_nadir_lat),
        },
        "results": results,
        "judge_results": judge_results,
    }
    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Full results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
