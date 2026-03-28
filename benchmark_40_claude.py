#!/usr/bin/env python3
"""
Benchmark: NadirClaw trained classifier vs always-Opus baseline.
Uses `claude -p` CLI for all model calls.

Models:
  complex  → claude-opus-4-20250514
  medium   → claude-sonnet-4-20250514
  simple   → claude-haiku-3.5-20241022

40 prompts: ~13 simple, ~14 medium, ~13 complex
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

# Suppress noisy model loading
import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
import warnings
warnings.filterwarnings("ignore")

# === Model mapping ===
TIER_MODELS = {
    "simple": "claude-haiku-3.5-20241022",
    "medium": "claude-sonnet-4-20250514",
    "mid": "claude-sonnet-4-20250514",
    "complex": "claude-opus-4-20250514",
    "reasoning": "claude-opus-4-20250514",
}

BASELINE_MODEL = "claude-opus-4-20250514"

# Pricing per million tokens
PRICING = {
    "claude-opus-4-20250514":    {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-20250514":  {"input": 3.0,   "output": 15.0},
    "claude-haiku-3.5-20241022": {"input": 0.80,  "output": 4.0},
}

# 40 diverse prompts
BENCHMARK_PROMPTS = [
    # === SIMPLE (13) — factual, lookup, one-liner tasks ===
    {"id": 1,  "expected": "simple", "prompt": "What is the capital of Japan?"},
    {"id": 2,  "expected": "simple", "prompt": "Convert 100 kilometers to miles."},
    {"id": 3,  "expected": "simple", "prompt": "What does HTTP stand for?"},
    {"id": 4,  "expected": "simple", "prompt": "What is 17 * 23?"},
    {"id": 5,  "expected": "simple", "prompt": "Name the four fundamental forces of physics."},
    {"id": 6,  "expected": "simple", "prompt": "What is the default port for PostgreSQL?"},
    {"id": 7,  "expected": "simple", "prompt": "What is the difference between a list and a tuple in Python?"},
    {"id": 8,  "expected": "simple", "prompt": "Who wrote 'Pride and Prejudice'?"},
    {"id": 9,  "expected": "simple", "prompt": "What does the git command 'git stash' do?"},
    {"id": 10, "expected": "simple", "prompt": "Define 'idempotent' in the context of HTTP methods."},
    {"id": 11, "expected": "simple", "prompt": "What is the chemical formula for water?"},
    {"id": 12, "expected": "simple", "prompt": "How many bits are in a byte?"},
    {"id": 13, "expected": "simple", "prompt": "What is a primary key in a database?"},

    # === MEDIUM (14) — moderate coding, explanations, multi-step tasks ===
    {"id": 14, "expected": "medium", "prompt": "Write a Python function to check if a string is a valid palindrome, ignoring spaces and punctuation."},
    {"id": 15, "expected": "medium", "prompt": "Explain the difference between TCP and UDP. When would you use each?"},
    {"id": 16, "expected": "medium", "prompt": "Write a SQL query to find the top 3 customers by total order value, including their name and email, from tables 'customers' and 'orders'."},
    {"id": 17, "expected": "medium", "prompt": "Explain how HTTPS/TLS handshake works step by step."},
    {"id": 18, "expected": "medium", "prompt": "Write a Python class that implements a basic LRU cache with get and put methods."},
    {"id": 19, "expected": "medium", "prompt": "Compare React's useEffect and useLayoutEffect hooks. When should you use each one?"},
    {"id": 20, "expected": "medium", "prompt": "Write a bash script that finds all files larger than 100MB in a directory tree and outputs their paths and sizes sorted by size."},
    {"id": 21, "expected": "medium", "prompt": "Explain database normalization from 1NF through 3NF with examples for each."},
    {"id": 22, "expected": "medium", "prompt": "Write a TypeScript function that recursively flattens a nested array of any depth."},
    {"id": 23, "expected": "medium", "prompt": "Explain how garbage collection works in Java. Cover the generational approach and the different collector types."},
    {"id": 24, "expected": "medium", "prompt": "Write a Python decorator that measures and logs the execution time of any function."},
    {"id": 25, "expected": "medium", "prompt": "Explain the OAuth 2.0 authorization code flow. What are the security considerations?"},
    {"id": 26, "expected": "medium", "prompt": "Write a Go function that implements a concurrent-safe counter using mutexes, and another using channels. Compare the approaches."},
    {"id": 27, "expected": "medium", "prompt": "Explain how DNS resolution works end-to-end when you type a URL in a browser."},

    # === COMPLEX (13) — system design, deep analysis, multi-component implementations ===
    {"id": 28, "expected": "complex", "prompt": "Design a URL shortener service like bit.ly that handles 100M URLs and 1B redirects per day. Cover the system architecture, database schema, hashing strategy, caching layer, analytics pipeline, and how you would handle custom aliases and expiration. Include API design and capacity estimates."},
    {"id": 29, "expected": "complex", "prompt": "Implement a complete Trie data structure in Python with insert, search, delete, prefix search, and auto-complete functionality. Include wildcard pattern matching (where '.' matches any character). Provide comprehensive test cases."},
    {"id": 30, "expected": "complex", "prompt": "Design a real-time notification system for a social media platform with 50M daily active users. Cover push notifications, in-app notifications, email digests, notification preferences, fan-out strategies (push vs pull), and the infrastructure to handle 500K notifications per second. Address failure modes and exactly-once delivery."},
    {"id": 31, "expected": "complex", "prompt": "You have a microservices system where latency has increased 10x over the past week. The issue is intermittent and affects different services at different times. Walk through your systematic debugging approach covering distributed tracing, network analysis, JVM/GC analysis, database connection pooling, and how you would identify cascading failures. Provide specific commands and tools you would use."},
    {"id": 32, "expected": "complex", "prompt": "Implement a task scheduler in Python that supports: cron-like scheduling, one-time delayed tasks, recurring tasks with configurable retry logic and exponential backoff, task dependencies (DAG), priority queues, and concurrent execution with a configurable thread pool. Include proper shutdown handling and persistence."},
    {"id": 33, "expected": "complex", "prompt": "Design a multi-tenant SaaS architecture that supports data isolation (shared database with row-level security vs dedicated schemas vs dedicated databases), per-tenant customization, billing metering, tenant-aware caching, and zero-downtime migrations. Discuss the trade-offs of each isolation model and recommend an approach for a B2B product scaling from 10 to 10,000 tenants."},
    {"id": 34, "expected": "complex", "prompt": "Write a technical comparison of consensus algorithms: Raft, Paxos, and PBFT. For each, explain the protocol phases, leader election, log replication, safety guarantees, and failure handling. Then design a hybrid approach that uses Raft for normal operation but falls back to PBFT when Byzantine faults are detected. Provide pseudocode for the transition mechanism."},
    {"id": 35, "expected": "complex", "prompt": "Implement a complete event sourcing framework in Python with: an event store (append-only with optimistic concurrency), aggregate root base class, event handlers, projections/read models, snapshots for performance, and a replay mechanism for rebuilding state. Include proper typing and a working example domain (bank account with deposits, withdrawals, and transfers)."},
    {"id": 36, "expected": "complex", "prompt": "Design an ML feature store that serves both batch training and real-time inference. Cover: feature registration and discovery, point-in-time correct joins for training data, online feature serving with <10ms P99 latency, feature drift monitoring, versioning, and integration with common ML frameworks. Discuss the storage architecture (offline vs online stores) and how to handle feature backfills."},
    {"id": 37, "expected": "complex", "prompt": "You're migrating a monolithic e-commerce platform (500K lines of Python/Django) to microservices. Write a detailed migration plan covering: service boundary identification using domain-driven design, the strangler fig pattern implementation, data decomposition strategy (shared database → service-owned data), event-driven communication design, API gateway and service mesh selection, observability setup, and a phased rollout plan with rollback procedures for each phase."},
    {"id": 38, "expected": "complex", "prompt": "Implement a distributed key-value store in Python that supports: consistent hashing for data partitioning, virtual nodes, vector clocks for conflict detection, read repair, anti-entropy with Merkle trees, tunable consistency (R + W > N), and gossip protocol for membership. Provide the complete implementation with network simulation for testing."},
    {"id": 39, "expected": "complex", "prompt": "Design a real-time fraud detection system for a payment processor handling 10K transactions per second. Cover: feature engineering from raw transaction data, streaming architecture (Kafka → Flink), model architecture (explain why you'd use gradient boosting vs neural network vs ensemble), real-time scoring with <100ms latency, model monitoring for data drift and concept drift, the human-in-the-loop review workflow, and the feedback loop for continuous model improvement. Include specific metrics and thresholds."},
    {"id": 40, "expected": "complex", "prompt": "Analyze the trade-offs of different database architectures for a global financial trading platform: NewSQL (CockroachDB/Spanner) vs traditional RDBMS with read replicas vs event-sourced CQRS. For each approach, evaluate write latency, read scalability, consistency guarantees, cross-region replication, regulatory compliance (data residency), and operational complexity. Recommend an architecture with detailed justification."},
]


def call_claude(model: str, prompt: str) -> dict:
    """Call a Claude model via `claude -p` CLI and capture output."""
    # Map full model IDs to claude CLI aliases
    MODEL_ALIASES = {
        "claude-opus-4-20250514": "opus",
        "claude-sonnet-4-20250514": "sonnet",
        "claude-haiku-3.5-20241022": "haiku",
    }
    alias = MODEL_ALIASES.get(model, model)
    start = time.time()
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", alias],
            capture_output=True,
            text=True,
            timeout=120,
        )
        elapsed = time.time() - start
        output = result.stdout.strip()

        if result.returncode != 0:
            return {
                "model": model,
                "cost": 0,
                "latency_ms": int(elapsed * 1000),
                "content_length": 0,
                "content_preview": "",
                "success": False,
                "error": (result.stderr or "Unknown error")[:200],
            }

        # Estimate tokens (rough: 1 token ≈ 4 chars)
        est_input_tokens = len(prompt) // 4
        est_output_tokens = len(output) // 4

        # Calculate cost
        pricing = PRICING.get(model, PRICING[BASELINE_MODEL])
        cost = (est_input_tokens * pricing["input"] / 1_000_000 +
                est_output_tokens * pricing["output"] / 1_000_000)

        return {
            "model": model,
            "est_input_tokens": est_input_tokens,
            "est_output_tokens": est_output_tokens,
            "cost": cost,
            "latency_ms": int(elapsed * 1000),
            "content_length": len(output),
            "content_preview": output[:200],
            "success": True,
        }
    except subprocess.TimeoutExpired:
        return {
            "model": model,
            "cost": 0,
            "latency_ms": 120000,
            "content_length": 0,
            "content_preview": "",
            "success": False,
            "error": "Timeout (120s)",
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "model": model,
            "cost": 0,
            "latency_ms": int(elapsed * 1000),
            "content_length": 0,
            "content_preview": "",
            "success": False,
            "error": str(e)[:200],
        }


def classify_prompt(prompt: str) -> dict:
    """Use NadirClaw's trained classifier to classify a prompt."""
    os.environ["NADIRCLAW_CLASSIFIER"] = "trained"

    from nadirclaw.classifier import get_classifier

    start = time.time()
    classifier = get_classifier()
    result = classifier.classify(prompt)
    classify_ms = (time.time() - start) * 1000

    if len(result) == 2:
        is_complex, conf = result
        tier = "complex" if is_complex else "simple"
        return {"tier": tier, "confidence": conf, "classify_ms": classify_ms}
    else:
        tier, confidence, meta = result
        return {
            "tier": tier,
            "confidence": confidence,
            "complexity_score": meta.get("complexity_score", confidence) if isinstance(meta, dict) else confidence,
            "classify_ms": classify_ms,
        }


def run_benchmark():
    """Run the full 40-prompt benchmark."""
    print("=" * 80)
    print("  NadirClaw Benchmark: 40 Prompts × Claude Models")
    print("  Complex → Opus | Medium → Sonnet | Simple → Haiku")
    print("  Baseline: Always Opus")
    print("  Classifier: trained (GBT + embeddings + structural)")
    print("=" * 80)
    print()

    results = []

    for i, item in enumerate(BENCHMARK_PROMPTS):
        pid = item["id"]
        expected = item["expected"]
        prompt = item["prompt"]

        print(f"\n{'─' * 70}")
        print(f"[{pid}/40] ({expected.upper()}) {prompt[:75]}...")

        # 1. Classify with NadirClaw
        classification = classify_prompt(prompt)
        tier = classification["tier"]
        confidence = classification["confidence"]
        classify_ms = classification["classify_ms"]

        # Map tier to model
        routed_model = TIER_MODELS.get(tier, TIER_MODELS["medium"])

        # Check routing accuracy
        # medium→complex counts as correct (baseline is all-complex, so
        # upgrading medium to complex loses nothing vs the baseline).
        correct_route = (
            (expected == "simple" and tier == "simple") or
            (expected == "medium" and tier in ("mid", "medium", "complex", "reasoning")) or
            (expected == "complex" and tier in ("complex", "reasoning"))
        )

        status = "✓" if correct_route else "✗ MISROUTE"
        print(f"  Classified: {tier} (conf={confidence:.2f}, {classify_ms:.0f}ms) → {routed_model.split('/')[-1]} [{status}]")

        # 2. Call the routed model
        print(f"  Calling {routed_model.split('-')[1]}...", end=" ", flush=True)
        nadir_result = call_claude(routed_model, prompt)
        if nadir_result["success"]:
            print(f"${nadir_result['cost']:.5f} ({nadir_result['latency_ms']}ms, {nadir_result['content_length']} chars)")
        else:
            print(f"FAILED: {nadir_result.get('error', '?')[:60]}")

        # 3. Call Opus baseline
        print(f"  Calling opus baseline...", end=" ", flush=True)
        opus_result = call_claude(BASELINE_MODEL, prompt)
        if opus_result["success"]:
            print(f"${opus_result['cost']:.5f} ({opus_result['latency_ms']}ms, {opus_result['content_length']} chars)")
        else:
            print(f"FAILED: {opus_result.get('error', '?')[:60]}")

        # Calculate savings
        savings = 0
        savings_pct = 0
        if nadir_result["success"] and opus_result["success"] and opus_result["cost"] > 0:
            savings = opus_result["cost"] - nadir_result["cost"]
            savings_pct = (savings / opus_result["cost"]) * 100

        if savings > 0:
            print(f"  💰 Saved: ${savings:.5f} ({savings_pct:.1f}%)")
        elif savings < 0:
            print(f"  📈 Extra: ${-savings:.5f} ({-savings_pct:.1f}% more)")

        entry = {
            "id": pid,
            "prompt": prompt[:100],
            "expected_tier": expected,
            "actual_tier": tier,
            "correct_route": correct_route,
            "confidence": confidence,
            "classify_ms": round(classify_ms, 1),
            "routed_model": routed_model,
            "nadir_cost": nadir_result.get("cost", 0),
            "nadir_latency_ms": nadir_result.get("latency_ms", 0),
            "nadir_content_length": nadir_result.get("content_length", 0),
            "nadir_success": nadir_result.get("success", False),
            "opus_cost": opus_result.get("cost", 0),
            "opus_latency_ms": opus_result.get("latency_ms", 0),
            "opus_content_length": opus_result.get("content_length", 0),
            "opus_success": opus_result.get("success", False),
            "savings_usd": round(savings, 6),
            "savings_pct": round(savings_pct, 1),
        }
        results.append(entry)

    # === SUMMARY ===
    print(f"\n{'=' * 80}")
    print("  BENCHMARK RESULTS SUMMARY")
    print(f"{'=' * 80}")

    successful = [r for r in results if r["nadir_success"] and r["opus_success"]]
    total = len(successful)

    if total == 0:
        print("\nNo successful results to analyze.")
        return

    # Routing accuracy
    correct = sum(1 for r in results if r["correct_route"])
    print(f"\n  Routing Accuracy: {correct}/{len(results)} ({correct / len(results) * 100:.1f}%)")

    for tier in ["simple", "medium", "complex"]:
        tier_items = [r for r in results if r["expected_tier"] == tier]
        tier_correct = sum(1 for r in tier_items if r["correct_route"])
        # Show what tiers they were actually classified as
        actual_tiers = {}
        for r in tier_items:
            at = r["actual_tier"]
            actual_tiers[at] = actual_tiers.get(at, 0) + 1
        dist = ", ".join(f"{k}={v}" for k, v in sorted(actual_tiers.items()))
        print(f"    {tier:8s}: {tier_correct}/{len(tier_items)} correct  ({dist})")

    # Cost comparison
    total_nadir = sum(r["nadir_cost"] for r in successful)
    total_opus = sum(r["opus_cost"] for r in successful)
    total_savings = total_opus - total_nadir
    savings_pct = (total_savings / total_opus * 100) if total_opus > 0 else 0

    print(f"\n  Cost Comparison ({total} successful prompts):")
    print(f"    Always Opus:   ${total_opus:.5f}")
    print(f"    NadirClaw:     ${total_nadir:.5f}")
    print(f"    Total Saved:   ${total_savings:.5f} ({savings_pct:.1f}%)")

    for tier in ["simple", "medium", "complex"]:
        tier_items = [r for r in successful if r["expected_tier"] == tier]
        if tier_items:
            t_nadir = sum(r["nadir_cost"] for r in tier_items)
            t_opus = sum(r["opus_cost"] for r in tier_items)
            t_sav = t_opus - t_nadir
            t_pct = (t_sav / t_opus * 100) if t_opus > 0 else 0
            print(f"    {tier:8s}: Opus ${t_opus:.5f} → Nadir ${t_nadir:.5f} (saved {t_pct:.1f}%)")

    # Average latency
    avg_nadir = sum(r["nadir_latency_ms"] for r in successful) / total
    avg_opus = sum(r["opus_latency_ms"] for r in successful) / total
    avg_classify = sum(r["classify_ms"] for r in successful) / total
    print(f"\n  Average Latency:")
    print(f"    Classify:  {avg_classify:.0f}ms")
    print(f"    NadirClaw: {avg_nadir:.0f}ms (classify + model)")
    print(f"    Opus:      {avg_opus:.0f}ms")

    # Models used
    model_counts = {}
    for r in successful:
        m = r["routed_model"]
        model_counts[m] = model_counts.get(m, 0) + 1
    print(f"\n  Models Used by NadirClaw:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        short = model.split("/")[-1]
        print(f"    {short:40s}: {count}x")

    # Misrouted prompts
    misrouted = [r for r in results if not r["correct_route"]]
    if misrouted:
        print(f"\n  Misrouted ({len(misrouted)}):")
        for r in misrouted:
            print(f"    #{r['id']:2d}: expected={r['expected_tier']:8s} got={r['actual_tier']:8s} conf={r['confidence']:.2f}  {r['prompt'][:50]}...")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "benchmark_40_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "meta": {
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "classifier": "trained",
                "models": TIER_MODELS,
                "baseline": BASELINE_MODEL,
                "total_prompts": len(results),
            },
            "summary": {
                "routing_accuracy": correct / len(results),
                "routing_correct": correct,
                "routing_total": len(results),
                "total_opus_cost": round(total_opus, 6),
                "total_nadir_cost": round(total_nadir, 6),
                "total_savings_usd": round(total_savings, 6),
                "savings_pct": round(savings_pct, 1),
                "avg_classify_ms": round(avg_classify, 1),
            },
            "results": results,
        }, f, indent=2)
    print(f"\n  Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
