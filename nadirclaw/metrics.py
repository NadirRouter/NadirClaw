"""Prometheus metrics for NadirClaw.

Zero-dependency Prometheus text format exporter. Tracks request counts,
latency histograms, token usage, cost, errors, cache hits, and fallbacks
— all labeled by model and tier.

Expose via GET /metrics in OpenMetrics text format.
"""

import time
from collections import defaultdict
from threading import Lock
from typing import Any, Dict, List, Optional

# Histogram bucket boundaries (milliseconds for latency)
# Includes 15000/30000 for tail latency visibility
LATENCY_BUCKETS = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 15000, 30000, float("inf")]


class _Counter:
    """Thread-safe counter with labels."""

    def __init__(self):
        self._lock = Lock()
        self._values: Dict[tuple, float] = defaultdict(float)

    def inc(self, labels: tuple = (), value: float = 1.0):
        with self._lock:
            self._values[labels] += value

    def items(self):
        with self._lock:
            return list(self._values.items())


class _Histogram:
    """Thread-safe histogram with labels and fixed buckets."""

    def __init__(self, buckets: List[float]):
        self._buckets = buckets
        self._lock = Lock()
        # Per label-set: {bucket_bound: count}
        self._counts: Dict[tuple, Dict[float, int]] = {}
        self._sums: Dict[tuple, float] = defaultdict(float)
        self._totals: Dict[tuple, int] = defaultdict(int)

    def observe(self, value: float, labels: tuple = ()):
        with self._lock:
            if labels not in self._counts:
                self._counts[labels] = {b: 0 for b in self._buckets}
            for b in self._buckets:
                if value <= b:
                    self._counts[labels][b] += 1
            self._sums[labels] += value
            self._totals[labels] += 1

    def items(self):
        with self._lock:
            return [
                (labels, dict(buckets), self._sums[labels], self._totals[labels])
                for labels, buckets in self._counts.items()
            ]


# ---------------------------------------------------------------------------
# Global metric instances
# ---------------------------------------------------------------------------

# Counters
requests_total = _Counter()         # labels: (model, tier, status)
tokens_prompt_total = _Counter()     # labels: (model,)
tokens_completion_total = _Counter() # labels: (model,)
cost_total = _Counter()              # labels: (model,)
cache_hits_total = _Counter()        # labels: ()
fallbacks_total = _Counter()         # labels: (from_model, to_model)
errors_total = _Counter()            # labels: (model, error_type)
tokens_saved_total = _Counter()      # labels: (optimization_mode,)
optimizations_total = _Counter()     # labels: (optimization_name,)
classifier_latency_total = _Counter()  # labels: (analyzer_type,) — sum of classify ms
classifier_calls_total = _Counter()    # labels: (analyzer_type,)

# Histograms
latency_ms = _Histogram(LATENCY_BUCKETS)  # labels: (model, tier)

# Uptime
_start_time = time.time()


def record_request(entry: Dict[str, Any]) -> None:
    """Record metrics from a log entry dict (called from _log_request)."""
    if entry.get("type") != "completion":
        return

    model = entry.get("selected_model", "unknown")
    tier = entry.get("tier", "unknown")
    status = entry.get("status", "ok")

    # Request count
    requests_total.inc((model, tier, status))

    # Tokens
    pt = entry.get("prompt_tokens", 0) or 0
    ct = entry.get("completion_tokens", 0) or 0
    if pt:
        tokens_prompt_total.inc((model,), pt)
    if ct:
        tokens_completion_total.inc((model,), ct)

    # Cost
    cost = entry.get("cost", 0) or 0
    if cost:
        cost_total.inc((model,), cost)

    # Latency
    total_lat = entry.get("total_latency_ms")
    if total_lat is not None:
        latency_ms.observe(float(total_lat), (model, tier))

    # Cache hit (check strategy field)
    strategy = entry.get("strategy") or ""
    if "cache-hit" in str(strategy) or "cache-hit" in str(entry.get("tier", "")):
        cache_hits_total.inc(())

    # Fallback
    fallback_from = entry.get("fallback_used")
    if fallback_from:
        fallbacks_total.inc((fallback_from, model))

    # Error
    if status != "ok":
        errors_total.inc((model, status))

    # Classifier latency
    clf_lat = entry.get("classifier_latency_ms")
    if clf_lat is not None:
        analyzer = entry.get("analyzer", "unknown")
        classifier_latency_total.inc((analyzer,), clf_lat)
        classifier_calls_total.inc((analyzer,))

    # Optimization
    saved = entry.get("tokens_saved", 0) or 0
    if saved > 0:
        opt_mode = entry.get("optimization_mode", "unknown")
        tokens_saved_total.inc((opt_mode,), saved)
        for opt_name in entry.get("optimizations_applied") or []:
            optimizations_total.inc((opt_name,))


def render_metrics() -> str:
    """Render all metrics in Prometheus text exposition format."""
    lines: List[str] = []

    # -- nadirclaw_requests_total --
    lines.append("# HELP nadirclaw_requests_total Total number of completed LLM requests.")
    lines.append("# TYPE nadirclaw_requests_total counter")
    for (model, tier, status), val in requests_total.items():
        lines.append(f'nadirclaw_requests_total{{model="{model}",tier="{tier}",status="{status}"}} {val}')

    # -- nadirclaw_tokens_prompt_total --
    lines.append("# HELP nadirclaw_tokens_prompt_total Total prompt tokens consumed.")
    lines.append("# TYPE nadirclaw_tokens_prompt_total counter")
    for (model,), val in tokens_prompt_total.items():
        lines.append(f'nadirclaw_tokens_prompt_total{{model="{model}"}} {int(val)}')

    # -- nadirclaw_tokens_completion_total --
    lines.append("# HELP nadirclaw_tokens_completion_total Total completion tokens generated.")
    lines.append("# TYPE nadirclaw_tokens_completion_total counter")
    for (model,), val in tokens_completion_total.items():
        lines.append(f'nadirclaw_tokens_completion_total{{model="{model}"}} {int(val)}')

    # -- nadirclaw_cost_dollars_total --
    lines.append("# HELP nadirclaw_cost_dollars_total Total estimated cost in USD.")
    lines.append("# TYPE nadirclaw_cost_dollars_total counter")
    for (model,), val in cost_total.items():
        lines.append(f'nadirclaw_cost_dollars_total{{model="{model}"}} {val:.6f}')

    # -- nadirclaw_cache_hits_total --
    lines.append("# HELP nadirclaw_cache_hits_total Total prompt cache hits.")
    lines.append("# TYPE nadirclaw_cache_hits_total counter")
    total_cache = sum(v for _, v in cache_hits_total.items())
    lines.append(f"nadirclaw_cache_hits_total {int(total_cache)}")

    # -- nadirclaw_fallbacks_total --
    lines.append("# HELP nadirclaw_fallbacks_total Total fallback events.")
    lines.append("# TYPE nadirclaw_fallbacks_total counter")
    for (from_model, to_model), val in fallbacks_total.items():
        lines.append(f'nadirclaw_fallbacks_total{{from_model="{from_model}",to_model="{to_model}"}} {int(val)}')

    # -- nadirclaw_errors_total --
    lines.append("# HELP nadirclaw_errors_total Total request errors.")
    lines.append("# TYPE nadirclaw_errors_total counter")
    for (model, error_type), val in errors_total.items():
        lines.append(f'nadirclaw_errors_total{{model="{model}",error_type="{error_type}"}} {int(val)}')

    # -- nadirclaw_request_latency_ms --
    lines.append("# HELP nadirclaw_request_latency_ms Request latency in milliseconds.")
    lines.append("# TYPE nadirclaw_request_latency_ms histogram")
    for (model, tier), buckets, s, count in latency_ms.items():
        cumulative = 0
        for bound in sorted(b for b in buckets if b != float("inf")):
            cumulative += buckets[bound]
            lines.append(
                f'nadirclaw_request_latency_ms_bucket{{model="{model}",tier="{tier}",le="{bound}"}} {cumulative}'
            )
        cumulative += buckets.get(float("inf"), 0)
        lines.append(
            f'nadirclaw_request_latency_ms_bucket{{model="{model}",tier="{tier}",le="+Inf"}} {cumulative}'
        )
        lines.append(f'nadirclaw_request_latency_ms_sum{{model="{model}",tier="{tier}"}} {s:.1f}')
        lines.append(f'nadirclaw_request_latency_ms_count{{model="{model}",tier="{tier}"}} {count}')

    # -- nadirclaw_tokens_saved_total --
    lines.append("# HELP nadirclaw_tokens_saved_total Total tokens saved by context optimization.")
    lines.append("# TYPE nadirclaw_tokens_saved_total counter")
    for (opt_mode,), val in tokens_saved_total.items():
        lines.append(f'nadirclaw_tokens_saved_total{{mode="{opt_mode}"}} {int(val)}')

    # -- nadirclaw_optimizations_total --
    lines.append("# HELP nadirclaw_optimizations_total Total optimization transform applications.")
    lines.append("# TYPE nadirclaw_optimizations_total counter")
    for (opt_name,), val in optimizations_total.items():
        lines.append(f'nadirclaw_optimizations_total{{transform="{opt_name}"}} {int(val)}')

    # -- nadirclaw_classifier_latency_ms --
    lines.append("# HELP nadirclaw_classifier_latency_ms_total Total classifier latency in milliseconds.")
    lines.append("# TYPE nadirclaw_classifier_latency_ms_total counter")
    for (analyzer,), val in classifier_latency_total.items():
        lines.append(f'nadirclaw_classifier_latency_ms_total{{analyzer="{analyzer}"}} {val:.1f}')

    # -- nadirclaw_classifier_calls_total --
    lines.append("# HELP nadirclaw_classifier_calls_total Total classifier invocations.")
    lines.append("# TYPE nadirclaw_classifier_calls_total counter")
    for (analyzer,), val in classifier_calls_total.items():
        lines.append(f'nadirclaw_classifier_calls_total{{analyzer="{analyzer}"}} {int(val)}')

    # -- nadirclaw_embed_cache (gauge from encoder module) --
    try:
        from nadirclaw.encoder import get_embed_cache_stats
        ec = get_embed_cache_stats()
        lines.append("# HELP nadirclaw_embed_cache_entries Current embedding cache entries.")
        lines.append("# TYPE nadirclaw_embed_cache_entries gauge")
        lines.append(f"nadirclaw_embed_cache_entries {ec['entries']}")
        lines.append("# HELP nadirclaw_embed_cache_hit_rate Embedding cache hit rate.")
        lines.append("# TYPE nadirclaw_embed_cache_hit_rate gauge")
        lines.append(f"nadirclaw_embed_cache_hit_rate {ec['hit_rate']}")
    except Exception:
        pass

    # -- nadirclaw_prompt_cache (gauge from cache module) --
    try:
        from nadirclaw.cache import get_prompt_cache
        pc = get_prompt_cache().get_stats()
        lines.append("# HELP nadirclaw_prompt_cache_entries Current prompt cache entries.")
        lines.append("# TYPE nadirclaw_prompt_cache_entries gauge")
        lines.append(f"nadirclaw_prompt_cache_entries {pc['entries']}")
        lines.append("# HELP nadirclaw_prompt_cache_memory_bytes Estimated prompt cache memory.")
        lines.append("# TYPE nadirclaw_prompt_cache_memory_bytes gauge")
        lines.append(f"nadirclaw_prompt_cache_memory_bytes {pc.get('estimated_memory_mb', 0) * 1_000_000:.0f}")
    except Exception:
        pass

    # -- nadirclaw_uptime_seconds --
    lines.append("# HELP nadirclaw_uptime_seconds Seconds since NadirClaw started.")
    lines.append("# TYPE nadirclaw_uptime_seconds gauge")
    lines.append(f"nadirclaw_uptime_seconds {time.time() - _start_time:.1f}")

    lines.append("")  # trailing newline
    return "\n".join(lines)
