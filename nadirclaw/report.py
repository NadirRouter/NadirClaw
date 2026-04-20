"""Log parsing and report generation for NadirClaw."""

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_since(since_str: str) -> datetime:
    """Parse a time filter string into a UTC datetime.

    Supports:
      - Duration: "24h", "7d", "30m"
      - ISO date: "2025-02-01"
      - ISO datetime: "2025-02-01T12:00:00"
    """
    since_str = since_str.strip()

    # Duration patterns: 30m, 24h, 7d
    match = re.fullmatch(r"(\d+)([mhd])", since_str)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        delta = {"m": timedelta(minutes=value), "h": timedelta(hours=value), "d": timedelta(days=value)}[unit]
        return datetime.now(timezone.utc) - delta

    # Try ISO date / datetime
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(since_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    raise ValueError(f"Cannot parse time filter: {since_str!r}. Use e.g. '24h', '7d', '2025-02-01'.")


def load_log_entries_sqlite(
    db_path: Path,
    since: Optional[datetime] = None,
    model_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read entries from the SQLite request log."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        query = "SELECT * FROM requests WHERE 1=1"
        params: List[Any] = []

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        if model_filter:
            query += " AND selected_model LIKE ?"
            params.append(f"%{model_filter}%")

        query += " ORDER BY timestamp ASC"

        cursor = conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def load_log_entries(
    log_path: Path,
    since: Optional[datetime] = None,
    model_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read JSONL log file and return filtered entries."""
    if not log_path.exists():
        return []

    entries: List[Dict[str, Any]] = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Filter by time
        if since:
            ts_str = entry.get("timestamp")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts < since:
                        continue
                except (ValueError, TypeError):
                    pass  # Keep entries with unparseable timestamps

        # Filter by model (substring match, case-insensitive)
        if model_filter:
            model = entry.get("selected_model", "") or ""
            if model_filter.lower() not in model.lower():
                continue

        entries.append(entry)

    return entries


def generate_report(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a structured report dict from log entries."""
    if not entries:
        return {
            "total_requests": 0,
            "time_range": None,
            "requests_by_type": {},
            "model_usage": {},
            "tier_distribution": {},
            "latency": {},
            "tokens": {},
            "fallback_count": 0,
            "error_count": 0,
            "streaming_count": 0,
            "tool_usage": {"requests_with_tools": 0, "total_tool_count": 0},
        }

    # Time range
    timestamps = []
    for e in entries:
        ts_str = e.get("timestamp")
        if ts_str:
            try:
                timestamps.append(datetime.fromisoformat(ts_str))
            except (ValueError, TypeError):
                pass

    time_range = None
    if timestamps:
        time_range = {
            "earliest": min(timestamps).isoformat(),
            "latest": max(timestamps).isoformat(),
        }

    # Requests by type
    requests_by_type: Dict[str, int] = {}
    for e in entries:
        req_type = e.get("type", "unknown")
        requests_by_type[req_type] = requests_by_type.get(req_type, 0) + 1

    # Model usage (with cost)
    model_usage: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        model = e.get("selected_model")
        if not model:
            continue
        if model not in model_usage:
            model_usage[model] = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}
        model_usage[model]["requests"] += 1
        pt = _safe_int(e.get("prompt_tokens", 0))
        ct = _safe_int(e.get("completion_tokens", 0))
        cost = _safe_float(e.get("cost")) or 0.0
        model_usage[model]["prompt_tokens"] += pt
        model_usage[model]["completion_tokens"] += ct
        model_usage[model]["total_tokens"] += pt + ct
        model_usage[model]["cost"] += cost

    # Total cost
    total_cost = sum(info["cost"] for info in model_usage.values())

    # Tier distribution
    tier_counts: Dict[str, int] = {}
    for e in entries:
        tier = e.get("tier")
        if tier:
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
    total_with_tier = sum(tier_counts.values())
    tier_distribution = {
        tier: {"count": count, "percentage": round(count / total_with_tier * 100, 1) if total_with_tier else 0}
        for tier, count in sorted(tier_counts.items())
    }

    # Latency stats
    classifier_latencies = [_safe_float(e.get("classifier_latency_ms")) for e in entries]
    classifier_latencies = [v for v in classifier_latencies if v is not None]
    total_latencies = [_safe_float(e.get("total_latency_ms")) for e in entries]
    total_latencies = [v for v in total_latencies if v is not None]

    latency: Dict[str, Any] = {}
    if classifier_latencies:
        latency["classifier"] = _percentile_stats(classifier_latencies)
    if total_latencies:
        latency["total"] = _percentile_stats(total_latencies)

    # Token totals
    all_prompt = sum(_safe_int(e.get("prompt_tokens", 0)) for e in entries)
    all_completion = sum(_safe_int(e.get("completion_tokens", 0)) for e in entries)
    tokens = {
        "prompt_tokens": all_prompt,
        "completion_tokens": all_completion,
        "total_tokens": all_prompt + all_completion,
    }

    # Fallback / error counts
    fallback_count = sum(1 for e in entries if e.get("fallback_used"))
    error_count = sum(1 for e in entries if e.get("status") == "error")

    # Streaming
    streaming_count = sum(1 for e in entries if e.get("stream"))

    # Tool usage
    requests_with_tools = sum(1 for e in entries if e.get("has_tools"))
    total_tool_count = sum(_safe_int(e.get("tool_count", 0)) for e in entries)

    return {
        "total_requests": len(entries),
        "time_range": time_range,
        "requests_by_type": requests_by_type,
        "model_usage": model_usage,
        "tier_distribution": tier_distribution,
        "latency": latency,
        "tokens": tokens,
        "total_cost": total_cost,
        "fallback_count": fallback_count,
        "error_count": error_count,
        "streaming_count": streaming_count,
        "tool_usage": {"requests_with_tools": requests_with_tools, "total_tool_count": total_tool_count},
    }


def format_report_text(report: Dict[str, Any]) -> str:
    """Format a report dict as human-readable text."""
    lines: List[str] = []
    lines.append("NadirClaw Report")
    lines.append("=" * 50)

    total = report.get("total_requests", 0)
    lines.append(f"Total requests: {total}")

    time_range = report.get("time_range")
    if time_range:
        lines.append(f"From: {time_range['earliest']}")
        lines.append(f"To:   {time_range['latest']}")

    # Requests by type
    rbt = report.get("requests_by_type", {})
    if rbt:
        lines.append("")
        lines.append("Requests by Type")
        lines.append("-" * 30)
        for typ, count in sorted(rbt.items()):
            lines.append(f"  {typ:20s} {count:>6}")

    # Tier distribution
    tiers = report.get("tier_distribution", {})
    if tiers:
        lines.append("")
        lines.append("Tier Distribution")
        lines.append("-" * 30)
        for tier, info in tiers.items():
            lines.append(f"  {tier:20s} {info['count']:>6}  ({info['percentage']}%)")

    # Total cost
    total_cost = report.get("total_cost", 0)
    if total_cost > 0:
        lines.append(f"Total cost: ${total_cost:.4f}")

    # Model usage (with cost breakdown)
    models = report.get("model_usage", {})
    if models:
        lines.append("")
        lines.append("Model Usage")
        lines.append("-" * 70)
        has_cost = any(info.get("cost", 0) > 0 for info in models.values())
        if has_cost:
            lines.append(f"  {'Model':35s} {'Reqs':>6}  {'Tokens':>10}  {'Cost':>10}")
            for model, info in sorted(models.items(), key=lambda x: x[1].get("cost", 0), reverse=True):
                cost_str = f"${info.get('cost', 0):.4f}"
                lines.append(f"  {model:35s} {info['requests']:>6}  {info['total_tokens']:>10}  {cost_str:>10}")
        else:
            lines.append(f"  {'Model':35s} {'Reqs':>6}  {'Tokens':>10}")
            for model, info in sorted(models.items(), key=lambda x: x[1]["requests"], reverse=True):
                lines.append(f"  {model:35s} {info['requests']:>6}  {info['total_tokens']:>10}")

    # Latency
    lat = report.get("latency", {})
    if lat:
        lines.append("")
        lines.append("Latency (ms)")
        lines.append("-" * 40)
        for key in ("classifier", "total"):
            stats = lat.get(key)
            if stats:
                lines.append(f"  {key:15s}  avg={stats['avg']:.0f}  p50={stats['p50']:.0f}  p95={stats['p95']:.0f}")

    # Tokens
    tok = report.get("tokens", {})
    if tok and tok.get("total_tokens", 0) > 0:
        lines.append("")
        lines.append("Token Usage")
        lines.append("-" * 30)
        lines.append(f"  Prompt:     {tok['prompt_tokens']:>10}")
        lines.append(f"  Completion: {tok['completion_tokens']:>10}")
        lines.append(f"  Total:      {tok['total_tokens']:>10}")

    # Fallback / errors / streaming / tools
    extras: List[str] = []
    if report.get("fallback_count", 0):
        extras.append(f"Fallbacks: {report['fallback_count']}")
    if report.get("error_count", 0):
        extras.append(f"Errors: {report['error_count']}")
    if report.get("streaming_count", 0):
        extras.append(f"Streaming requests: {report['streaming_count']}")
    tool_info = report.get("tool_usage", {})
    if tool_info.get("requests_with_tools", 0):
        extras.append(f"Requests with tools: {tool_info['requests_with_tools']} ({tool_info['total_tool_count']} tools total)")

    if extras:
        lines.append("")
        for line in extras:
            lines.append(f"  {line}")

    # Pro upsell at report-generation time, when the user has just seen their
    # cost number. Only shown if there is meaningful cost to upsell against.
    total_cost = report.get("total_cost", 0) or 0
    if total_cost > 0.50:
        lines.append("")
        lines.append("=" * 50)
        lines.append("This report updates in real time on Nadir Pro, with per-team")
        lines.append("breakdowns, alerting, and a trained classifier that cuts costs")
        lines.append("another 10-20% on the same traffic.")
        lines.append("30-day free trial: https://getnadir.com?ref=cli-report")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-model, per-day cost breakdown
# ---------------------------------------------------------------------------

def generate_cost_breakdown(
    entries: List[Dict[str, Any]],
    by_model: bool = False,
    by_day: bool = False,
) -> Dict[str, Any]:
    """Generate cost breakdown by model, by day, or both.

    Also flags anomalies: any model whose daily spend is > 2× its 7-day average.
    """
    if not entries:
        return {"total_cost": 0, "breakdown": [], "anomalies": []}

    # Build per-model-per-day aggregation
    buckets: Dict[str, Dict[str, Dict[str, Any]]] = {}  # model → day → stats
    for e in entries:
        model = e.get("selected_model") or "unknown"
        ts_str = e.get("timestamp")
        day = "all"
        if ts_str:
            try:
                day = datetime.fromisoformat(ts_str).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass
        cost = _safe_float(e.get("cost")) or 0.0
        pt = _safe_int(e.get("prompt_tokens", 0))
        ct = _safe_int(e.get("completion_tokens", 0))

        if model not in buckets:
            buckets[model] = {}
        if day not in buckets[model]:
            buckets[model][day] = {"requests": 0, "cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
        buckets[model][day]["requests"] += 1
        buckets[model][day]["cost"] += cost
        buckets[model][day]["prompt_tokens"] += pt
        buckets[model][day]["completion_tokens"] += ct

    # Build output rows
    rows: List[Dict[str, Any]] = []

    if by_model and by_day:
        for model in sorted(buckets.keys()):
            for day in sorted(buckets[model].keys()):
                row = {"model": model, "day": day, **buckets[model][day]}
                rows.append(row)
    elif by_model:
        for model in sorted(buckets.keys()):
            agg = {"requests": 0, "cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
            for day_stats in buckets[model].values():
                for k in agg:
                    agg[k] += day_stats[k]
            rows.append({"model": model, **agg})
    elif by_day:
        day_agg: Dict[str, Dict[str, Any]] = {}
        for model, days in buckets.items():
            for day, stats in days.items():
                if day not in day_agg:
                    day_agg[day] = {"requests": 0, "cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
                for k in day_agg[day]:
                    day_agg[day][k] += stats[k]
        for day in sorted(day_agg.keys()):
            rows.append({"day": day, **day_agg[day]})
    else:
        rows = [{"total": True, "requests": len(entries),
                 "cost": sum(_safe_float(e.get("cost")) or 0.0 for e in entries)}]

    # Anomaly detection: flag any model whose daily spend > 2× its 7-day average
    anomalies: List[Dict[str, Any]] = []
    for model, days in buckets.items():
        daily_costs = sorted(days.items())
        if len(daily_costs) < 2:
            continue
        # Use last 7 days for average
        recent = [c["cost"] for _, c in daily_costs[-7:]]
        avg = sum(recent) / len(recent) if recent else 0
        if avg > 0:
            latest_day, latest_stats = daily_costs[-1]
            if latest_stats["cost"] > 2 * avg:
                anomalies.append({
                    "model": model,
                    "day": latest_day,
                    "spend": round(latest_stats["cost"], 4),
                    "avg_7d": round(avg, 4),
                    "ratio": round(latest_stats["cost"] / avg, 1),
                })

    total_cost = sum(row.get("cost", 0) for row in rows)
    return {"total_cost": total_cost, "breakdown": rows, "anomalies": anomalies}


def format_cost_breakdown_text(data: Dict[str, Any]) -> str:
    """Format cost breakdown as human-readable text."""
    lines: List[str] = []
    lines.append("NadirClaw Cost Breakdown")
    lines.append("=" * 60)

    rows = data.get("breakdown", [])
    if not rows:
        lines.append("No data.")
        return "\n".join(lines)

    # Determine columns
    has_model = any("model" in r for r in rows)
    has_day = any("day" in r for r in rows)

    if has_model and has_day:
        lines.append(f"  {'Model':30s} {'Day':12s} {'Reqs':>6}  {'Cost':>10}")
        lines.append(f"  {'─' * 30} {'─' * 12} {'─' * 6}  {'─' * 10}")
        for r in rows:
            lines.append(f"  {r['model']:30s} {r['day']:12s} {r['requests']:>6}  ${r['cost']:.4f}")
    elif has_model:
        lines.append(f"  {'Model':35s} {'Reqs':>6}  {'Cost':>10}")
        lines.append(f"  {'─' * 35} {'─' * 6}  {'─' * 10}")
        for r in sorted(rows, key=lambda x: x.get("cost", 0), reverse=True):
            lines.append(f"  {r['model']:35s} {r['requests']:>6}  ${r['cost']:.4f}")
    elif has_day:
        lines.append(f"  {'Day':12s} {'Reqs':>6}  {'Cost':>10}")
        lines.append(f"  {'─' * 12} {'─' * 6}  {'─' * 10}")
        for r in rows:
            lines.append(f"  {r['day']:12s} {r['requests']:>6}  ${r['cost']:.4f}")

    total_cost = data.get("total_cost", 0)
    lines.append(f"\n  Total: ${total_cost:.4f}")

    anomalies = data.get("anomalies", [])
    if anomalies:
        lines.append("")
        lines.append("⚠ Cost Anomalies")
        lines.append("-" * 60)
        for a in anomalies:
            lines.append(
                f"  {a['model']:30s} {a['day']}: ${a['spend']:.4f} "
                f"({a['ratio']}× vs 7d avg ${a['avg_7d']:.4f})"
            )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(val: Any) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _percentile_stats(values: List[float]) -> Dict[str, float]:
    """Compute avg, p50, p95 from a list of numeric values."""
    values = sorted(values)
    n = len(values)
    avg = sum(values) / n

    def _percentile(p: float) -> float:
        k = (n - 1) * p / 100.0
        f = int(k)
        c = f + 1
        if c >= n:
            return values[-1]
        return values[f] + (k - f) * (values[c] - values[f])

    return {"avg": avg, "p50": _percentile(50), "p95": _percentile(95)}
