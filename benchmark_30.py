#!/usr/bin/env python3
"""
Benchmark: NadirClaw routing vs always using Claude Opus 4.6
Sends 30 prompts of varying complexity through both paths and compares cost + quality.
"""

import asyncio
import json
import os
import time
import sys

# Ensure we can import nadirclaw
sys.path.insert(0, os.path.dirname(__file__))

import litellm

litellm.suppress_debug_info = True
os.environ.setdefault("LITELLM_LOG", "ERROR")

# Suppress noisy model loading output
import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
import warnings
warnings.filterwarnings("ignore")

# 30 prompts: ~10 simple, ~10 medium, ~10 complex
BENCHMARK_PROMPTS = [
    # === SIMPLE (expect cheap model) ===
    {"id": 1, "expected": "simple", "prompt": "What is the capital of France?"},
    {"id": 2, "expected": "simple", "prompt": "Convert 72 degrees Fahrenheit to Celsius."},
    {"id": 3, "expected": "simple", "prompt": "What does the acronym API stand for?"},
    {"id": 4, "expected": "simple", "prompt": "List 5 common Python data types."},
    {"id": 5, "expected": "simple", "prompt": "What is the difference between == and === in JavaScript?"},
    {"id": 6, "expected": "simple", "prompt": "How do you create a virtual environment in Python?"},
    {"id": 7, "expected": "simple", "prompt": "What HTTP status code means 'Not Found'?"},
    {"id": 8, "expected": "simple", "prompt": "What is a REST API?"},
    {"id": 9, "expected": "simple", "prompt": "How do you reverse a string in Python?"},
    {"id": 10, "expected": "simple", "prompt": "What is the time complexity of binary search?"},

    # === MEDIUM (expect mid-tier model) ===
    {"id": 11, "expected": "medium", "prompt": "Write a Python function that takes a list of integers and returns the second largest element. Handle edge cases like duplicates and lists with fewer than 2 elements."},
    {"id": 12, "expected": "medium", "prompt": "Explain the difference between SQL JOIN types (INNER, LEFT, RIGHT, FULL) with examples."},
    {"id": 13, "expected": "medium", "prompt": "Write a bash script that monitors disk usage and sends an alert if any partition exceeds 90%."},
    {"id": 14, "expected": "medium", "prompt": "Explain how Docker networking works. Cover bridge, host, and overlay network modes."},
    {"id": 15, "expected": "medium", "prompt": "Write a TypeScript generic function that deep-merges two objects, handling nested objects and arrays."},
    {"id": 16, "expected": "medium", "prompt": "Explain the CAP theorem and give real-world examples of databases that prioritize different combinations."},
    {"id": 17, "expected": "medium", "prompt": "Write a Python decorator that implements retry logic with exponential backoff."},
    {"id": 18, "expected": "medium", "prompt": "Compare and contrast Redis and Memcached. When would you choose one over the other?"},
    {"id": 19, "expected": "medium", "prompt": "Write a React custom hook that debounces an input value with a configurable delay."},
    {"id": 20, "expected": "medium", "prompt": "Explain database indexing strategies. When should you use B-tree vs hash vs GIN indexes?"},

    # === COMPLEX (expect premium model) ===
    {"id": 21, "expected": "complex", "prompt": "Design a distributed rate limiter service that can handle 1 million requests per second across 50 servers. Cover the architecture, data structures, consistency model, and failure modes. Include a system diagram and API design."},
    {"id": 22, "expected": "complex", "prompt": "Implement a lock-free concurrent hash map in C++ using atomic operations. Explain your approach to handling hash collisions, resizing, and memory reclamation. Provide the full implementation with comments."},
    {"id": 23, "expected": "complex", "prompt": "Design a real-time collaborative text editor like Google Docs. Cover the CRDT vs OT decision, conflict resolution, cursor synchronization, offline support, and the architecture for handling millions of concurrent users. Provide detailed technical trade-offs."},
    {"id": 24, "expected": "complex", "prompt": "You are debugging a production Kubernetes cluster where pods are randomly getting OOMKilled despite having plenty of memory allocated. The issue only happens under high load and affects different services. Walk me through your systematic debugging approach, covering cgroups, kernel memory accounting, JVM heap behavior, and potential kernel bugs."},
    {"id": 25, "expected": "complex", "prompt": "Implement a B+ tree in Python with insert, delete, search, and range query operations. The tree should support configurable order, handle all edge cases (underflow, overflow, redistribution, merging), and include proper leaf node linking for efficient range scans. Provide the complete implementation with tests."},
    {"id": 26, "expected": "complex", "prompt": "Design a multi-region, active-active database architecture for a financial trading platform that requires sub-millisecond latency, strong consistency for account balances, and eventual consistency for market data. Cover conflict resolution, failover, and regulatory compliance (data residency)."},
    {"id": 27, "expected": "complex", "prompt": "Write a comprehensive technical RFC for migrating a monolithic Django application to microservices. Cover service boundaries identification, data decomposition strategy, the strangler fig pattern, API gateway design, distributed tracing, and a phased rollout plan with rollback procedures."},
    {"id": 28, "expected": "complex", "prompt": "Implement a toy garbage collector in Rust that supports mark-and-sweep, generational collection, and concurrent marking. Explain the trade-offs between throughput and pause times. Include the full implementation."},
    {"id": 29, "expected": "complex", "prompt": "Design an ML pipeline that detects fraudulent transactions in real-time. Cover feature engineering from raw transaction data, model architecture (consider both gradient boosting and neural approaches), the serving infrastructure for <50ms P99 latency, model monitoring for data drift, and the feedback loop for continuous learning. Provide code samples for key components."},
    {"id": 30, "expected": "complex", "prompt": "Analyze the trade-offs between different consensus algorithms (Raft, Paxos, PBFT, HotStuff) for a blockchain-based supply chain system. Consider Byzantine fault tolerance, throughput, latency, and network partition handling. Recommend an approach with detailed justification and pseudo-code for the core protocol."},
]

# Opus 4.6 pricing (per million tokens)
OPUS_INPUT_COST = 15.0 / 1_000_000   # $15/M input
OPUS_OUTPUT_COST = 75.0 / 1_000_000  # $75/M output


def get_litellm_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Get cost from litellm's cost database, fallback to manual."""
    try:
        cost = litellm.completion_cost(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        if cost and cost > 0:
            return cost
    except Exception:
        pass
    # Fallback for opus
    if "opus" in model.lower():
        return prompt_tokens * OPUS_INPUT_COST + completion_tokens * OPUS_OUTPUT_COST
    return 0.0


def _resolve_api_key(model: str) -> dict:
    """Resolve API key — prefer env var, skip expired OAuth tokens."""
    # Prefer env var (already set by caller or shell)
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if env_key and not env_key.startswith("sk-ant-oat"):
        return {"api_key": env_key}
    # Fallback: Horizen .env.local
    horizen_env = os.path.join(os.path.dirname(__file__), "..", "Horizen", ".env.local")
    if os.path.exists(horizen_env):
        with open(horizen_env) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "ANTHROPIC_API_KEY":
                    return {"api_key": v.strip()}
    return {}


async def call_model(model: str, prompt: str, max_tokens: int = 300) -> dict:
    """Call a model and return result with timing and cost."""
    extra_kwargs = _resolve_api_key(model)
    start = time.time()
    try:
        response = await asyncio.to_thread(
            litellm.completion,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
            **extra_kwargs,
        )
        elapsed = time.time() - start
        usage = response.usage
        content = response.choices[0].message.content or ""
        cost = get_litellm_cost(model, usage.prompt_tokens, usage.completion_tokens)
        return {
            "model": model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cost": cost,
            "latency_ms": int(elapsed * 1000),
            "content_length": len(content),
            "content_preview": content[:150],
            "success": True,
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


async def classify_and_route(prompt: str) -> dict:
    """Use NadirClaw's classifier to pick a model, then call it."""
    from nadirclaw.classifier import get_classifier
    from nadirclaw.settings import settings
    from nadirclaw.features import StructuralFeatureExtractor, compute_complexity_score

    classifier = get_classifier()
    result = classifier.classify(prompt)

    # Handle both binary (2-tuple) and cascade/ternary (3-tuple) classifiers
    if len(result) == 2:
        is_complex, conf = result
        tier = "complex" if is_complex else "simple"
        confidence = conf
        complexity_score = conf
    else:
        tier, confidence, meta = result
        complexity_score = meta.get("complexity_score", confidence) if isinstance(meta, dict) else confidence

    # Also get structural features for display
    extractor = StructuralFeatureExtractor()
    features = extractor.extract([{"role": "user", "content": prompt}])
    structural_score = compute_complexity_score(features)

    # Get model for this tier
    tier_map = {
        "simple": settings.SIMPLE_MODEL,
        "mid": settings.MID_MODEL,
        "medium": settings.MID_MODEL,
        "complex": settings.COMPLEX_MODEL,
        "reasoning": settings.REASONING_MODEL,
    }
    model = tier_map.get(tier, settings.MID_MODEL)

    # Call the model
    result = await call_model(model, prompt)
    result["tier"] = tier
    result["confidence"] = confidence
    result["complexity_score"] = complexity_score
    result["structural_score"] = structural_score
    result["classifier_model"] = model
    return result


async def run_benchmark():
    """Run the full benchmark."""
    print("=" * 80)
    print("  NadirClaw Routing Benchmark: 30 Prompts")
    print("  Baseline: Always Claude Opus 4.6 (claude-opus-4-20250514)")
    print("=" * 80)
    print()

    # Check which models are available
    opus_model = "claude-opus-4-20250514"

    # Enable cascade classifier
    os.environ["NADIRCLAW_CLASSIFIER"] = "cascade"

    results = []

    for i, item in enumerate(BENCHMARK_PROMPTS):
        pid = item["id"]
        expected = item["expected"]
        prompt = item["prompt"]

        print(f"\n[{pid}/30] ({expected.upper()}) {prompt[:80]}...")

        # 1. Classify & route with NadirClaw
        nadir_result = await classify_and_route(prompt)

        # 2. Call Opus baseline
        opus_result = await call_model(opus_model, prompt)

        # Calculate savings
        savings = opus_result["cost"] - nadir_result["cost"] if opus_result["success"] and nadir_result["success"] else 0
        savings_pct = (savings / opus_result["cost"] * 100) if opus_result["cost"] > 0 else 0

        # Check routing accuracy
        tier = nadir_result.get("tier", "unknown")
        correct_route = (
            (expected == "simple" and tier == "simple") or
            (expected == "medium" and tier in ("mid", "medium")) or
            (expected == "complex" and tier == "complex")
        )

        entry = {
            "id": pid,
            "expected_tier": expected,
            "actual_tier": tier,
            "correct_route": correct_route,
            "nadir_model": nadir_result.get("classifier_model", "?"),
            "nadir_cost": nadir_result.get("cost", 0),
            "nadir_latency": nadir_result.get("latency_ms", 0),
            "nadir_tokens": nadir_result.get("prompt_tokens", 0) + nadir_result.get("completion_tokens", 0),
            "nadir_success": nadir_result.get("success", False),
            "opus_cost": opus_result.get("cost", 0),
            "opus_latency": opus_result.get("latency_ms", 0),
            "opus_tokens": opus_result.get("prompt_tokens", 0) + opus_result.get("completion_tokens", 0),
            "opus_success": opus_result.get("success", False),
            "savings_usd": savings,
            "savings_pct": savings_pct,
            "confidence": nadir_result.get("confidence", 0),
            "complexity_score": nadir_result.get("complexity_score", 0),
            "structural_score": nadir_result.get("structural_score", 0),
        }
        results.append(entry)

        status = "OK" if correct_route else "MISROUTE"
        model_short = nadir_result.get("classifier_model", "?").split("/")[-1][:30]
        print(f"  Route: {tier} -> {model_short} (conf={nadir_result.get('confidence', 0):.2f}) [{status}]")
        print(f"  Nadir: ${nadir_result.get('cost', 0):.4f} ({nadir_result.get('latency_ms', 0)}ms)")
        print(f"  Opus:  ${opus_result.get('cost', 0):.4f} ({opus_result.get('latency_ms', 0)}ms)")
        if savings > 0:
            print(f"  Saved: ${savings:.4f} ({savings_pct:.1f}%)")
        elif savings < 0:
            print(f"  Extra: ${-savings:.4f} ({-savings_pct:.1f}% more)")

    # === SUMMARY ===
    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r["nadir_success"] and r["opus_success"]]
    total = len(successful)

    if total == 0:
        print("\nNo successful results to analyze.")
        return

    # Routing accuracy
    correct = sum(1 for r in results if r["correct_route"])
    print(f"\n  Routing Accuracy: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")

    # Tier breakdown
    for tier in ["simple", "medium", "complex"]:
        tier_items = [r for r in results if r["expected_tier"] == tier]
        tier_correct = sum(1 for r in tier_items if r["correct_route"])
        print(f"    {tier:8s}: {tier_correct}/{len(tier_items)} correct")

    # Cost comparison
    total_nadir = sum(r["nadir_cost"] for r in successful)
    total_opus = sum(r["opus_cost"] for r in successful)
    total_savings = total_opus - total_nadir
    savings_pct = (total_savings / total_opus * 100) if total_opus > 0 else 0

    print(f"\n  Cost Comparison ({total} prompts):")
    print(f"    Always Opus:   ${total_opus:.4f}")
    print(f"    NadirClaw:     ${total_nadir:.4f}")
    print(f"    Total Saved:   ${total_savings:.4f} ({savings_pct:.1f}%)")

    # Per-tier cost breakdown
    print(f"\n  Per-Tier Breakdown:")
    for tier in ["simple", "medium", "complex"]:
        tier_items = [r for r in successful if r["expected_tier"] == tier]
        if tier_items:
            t_nadir = sum(r["nadir_cost"] for r in tier_items)
            t_opus = sum(r["opus_cost"] for r in tier_items)
            t_savings = t_opus - t_nadir
            t_pct = (t_savings / t_opus * 100) if t_opus > 0 else 0
            print(f"    {tier:8s}: Opus ${t_opus:.4f} -> Nadir ${t_nadir:.4f} (saved ${t_savings:.4f}, {t_pct:.1f}%)")

    # Latency comparison
    avg_nadir_latency = sum(r["nadir_latency"] for r in successful) / total
    avg_opus_latency = sum(r["opus_latency"] for r in successful) / total
    print(f"\n  Average Latency:")
    print(f"    Opus:     {avg_opus_latency:.0f}ms")
    print(f"    NadirClaw: {avg_nadir_latency:.0f}ms")

    # Models used
    model_counts = {}
    for r in successful:
        m = r["nadir_model"]
        model_counts[m] = model_counts.get(m, 0) + 1
    print(f"\n  Models Used by NadirClaw:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"    {model:40s}: {count}x")

    # Misrouted prompts
    misrouted = [r for r in results if not r["correct_route"]]
    if misrouted:
        print(f"\n  Misrouted Prompts ({len(misrouted)}):")
        for r in misrouted:
            print(f"    #{r['id']}: expected={r['expected_tier']}, got={r['actual_tier']}")

    # Save detailed results
    output_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total_prompts": len(results),
                "successful": total,
                "routing_accuracy": correct / len(results),
                "total_opus_cost": total_opus,
                "total_nadir_cost": total_nadir,
                "total_savings": total_savings,
                "savings_pct": savings_pct,
            },
            "results": results,
        }, f, indent=2)
    print(f"\n  Detailed results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
