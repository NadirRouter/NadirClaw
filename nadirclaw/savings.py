"""Cost savings calculator for NadirClaw.

Analyzes request logs and calculates how much money was saved by routing
simple prompts to cheap models instead of sending everything to premium.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nadirclaw.report import load_log_entries, parse_since
from nadirclaw.routing import MODEL_REGISTRY


def get_model_cost(model: str) -> Tuple[float, float]:
    """Return (cost_per_m_input, cost_per_m_output) for a model.

    Falls back to reasonable defaults if model is unknown.
    """
    info = MODEL_REGISTRY.get(model)
    if info:
        return info.get("cost_per_m_input", 0), info.get("cost_per_m_output", 0)

    # Try partial matches
    model_lower = model.lower()
    for key, val in MODEL_REGISTRY.items():
        if key.lower() in model_lower or model_lower in key.lower():
            return val.get("cost_per_m_input", 0), val.get("cost_per_m_output", 0)

    return 0, 0


def calculate_actual_cost(entries: List[Dict[str, Any]]) -> float:
    """Calculate the actual cost of all requests using the models NadirClaw chose."""
    total = 0.0
    for e in entries:
        model = e.get("selected_model", "")
        pt = _safe_int(e.get("prompt_tokens", 0))
        ct = _safe_int(e.get("completion_tokens", 0))
        cost_in, cost_out = get_model_cost(model)
        total += (pt / 1_000_000) * cost_in + (ct / 1_000_000) * cost_out
    return total


def calculate_hypothetical_cost(entries: List[Dict[str, Any]], always_model: str) -> float:
    """Calculate what it would have cost if every request used one model."""
    cost_in, cost_out = get_model_cost(always_model)
    total = 0.0
    for e in entries:
        pt = _safe_int(e.get("prompt_tokens", 0))
        ct = _safe_int(e.get("completion_tokens", 0))
        total += (pt / 1_000_000) * cost_in + (ct / 1_000_000) * cost_out
    return total


def generate_savings_report(
    log_path: Path,
    since: Optional[str] = None,
    baseline_model: Optional[str] = None,
    entries: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate a cost savings report.

    Args:
        log_path: Path to the JSONL log file (used if entries is not provided).
        since: Optional time filter (e.g. "24h", "7d").
        baseline_model: Model to compare against (what you'd use without routing).
                       Defaults to the most expensive model seen in logs.
        entries: Pre-loaded log entries (skips file loading when provided).
    """
    if entries is None:
        since_dt = parse_since(since) if since else None
        entries = load_log_entries(log_path, since=since_dt)

    if not entries:
        return {"total_requests": 0, "message": "No requests found in logs."}

    # Find all models used
    models_used = {}
    for e in entries:
        model = e.get("selected_model", "")
        if model:
            models_used[model] = models_used.get(model, 0) + 1

    # Determine baseline: most expensive model in logs, or user-specified
    if not baseline_model:
        max_cost = 0
        for model in models_used:
            cost_in, cost_out = get_model_cost(model)
            avg_cost = (cost_in + cost_out) / 2
            if avg_cost > max_cost:
                max_cost = avg_cost
                baseline_model = model

    if not baseline_model:
        baseline_model = "claude-sonnet-4-5-20250929"

    actual_cost = calculate_actual_cost(entries)
    baseline_cost = calculate_hypothetical_cost(entries, baseline_model)

    savings = baseline_cost - actual_cost
    savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0

    # Per-model breakdown
    model_breakdown = []
    for model, count in sorted(models_used.items(), key=lambda x: x[1], reverse=True):
        model_entries = [e for e in entries if e.get("selected_model") == model]
        cost = calculate_actual_cost(model_entries)
        hypothetical = calculate_hypothetical_cost(model_entries, baseline_model)
        model_savings = hypothetical - cost
        total_tokens = sum(
            _safe_int(e.get("prompt_tokens", 0)) + _safe_int(e.get("completion_tokens", 0))
            for e in model_entries
        )
        model_breakdown.append({
            "model": model,
            "requests": count,
            "tokens": total_tokens,
            "actual_cost": round(cost, 4),
            "baseline_cost": round(hypothetical, 4),
            "saved": round(model_savings, 4),
        })

    # Tier breakdown
    tier_counts = {}
    for e in entries:
        tier = e.get("tier", "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    # Projection
    total_tokens = sum(
        _safe_int(e.get("prompt_tokens", 0)) + _safe_int(e.get("completion_tokens", 0))
        for e in entries
    )

    # Time span
    timestamps = []
    for e in entries:
        ts_str = e.get("timestamp")
        if ts_str:
            try:
                timestamps.append(datetime.fromisoformat(ts_str))
            except (ValueError, TypeError):
                pass

    hours_span = 1
    if len(timestamps) >= 2:
        delta = max(timestamps) - min(timestamps)
        hours_span = max(delta.total_seconds() / 3600, 1)

    daily_rate = actual_cost / hours_span * 24
    monthly_actual = daily_rate * 30
    monthly_baseline = (baseline_cost / hours_span * 24) * 30
    monthly_savings = monthly_baseline - monthly_actual

    return {
        "total_requests": len(entries),
        "total_tokens": total_tokens,
        "baseline_model": baseline_model,
        "actual_cost": round(actual_cost, 4),
        "baseline_cost": round(baseline_cost, 4),
        "savings": round(savings, 4),
        "savings_percentage": round(savings_pct, 1),
        "model_breakdown": model_breakdown,
        "tier_distribution": tier_counts,
        "projection": {
            "hours_analyzed": round(hours_span, 1),
            "monthly_actual": round(monthly_actual, 2),
            "monthly_baseline": round(monthly_baseline, 2),
            "monthly_savings": round(monthly_savings, 2),
        },
    }


def format_savings_text(report: Dict[str, Any]) -> str:
    """Format savings report as human-readable text."""
    lines = []
    lines.append("NadirClaw Savings Report")
    lines.append("=" * 50)

    if report.get("total_requests", 0) == 0:
        lines.append("No requests found. Start using NadirClaw to see savings!")
        lines.append("")
        lines.append("Tip: Nadir Pro tracks savings in a live dashboard, no CLI needed.")
        lines.append("     https://getnadir.com?ref=cli-savings")
        return "\n".join(lines)

    lines.append(f"Total requests:  {report['total_requests']}")
    lines.append(f"Total tokens:    {report['total_tokens']:,}")
    lines.append(f"Baseline model:  {report['baseline_model']}")
    lines.append("")

    # The money shot
    lines.append("Cost Comparison")
    lines.append("-" * 40)
    lines.append(f"  Without NadirClaw:  ${report['baseline_cost']:.4f}")
    lines.append(f"  With NadirClaw:     ${report['actual_cost']:.4f}")
    lines.append(f"  You saved:          ${report['savings']:.4f} ({report['savings_percentage']}%)")

    # Model breakdown
    breakdown = report.get("model_breakdown", [])
    if breakdown:
        lines.append("")
        lines.append("Cost by Model")
        lines.append("-" * 60)
        lines.append(f"  {'Model':35s} {'Reqs':>5} {'Cost':>8} {'Saved':>8}")
        for m in breakdown:
            lines.append(
                f"  {m['model']:35s} {m['requests']:>5} ${m['actual_cost']:>7.4f} ${m['saved']:>7.4f}"
            )

    # Tier distribution
    tiers = report.get("tier_distribution", {})
    if tiers:
        lines.append("")
        lines.append("Routing Distribution")
        lines.append("-" * 30)
        total = sum(tiers.values())
        for tier, count in sorted(tiers.items()):
            pct = count / total * 100 if total else 0
            bar = "█" * int(pct / 2)
            lines.append(f"  {tier:12s} {count:>5} ({pct:4.1f}%) {bar}")

    # Monthly projection
    proj = report.get("projection", {})
    if proj:
        lines.append("")
        lines.append(f"Monthly Projection (based on {proj['hours_analyzed']}h of data)")
        lines.append("-" * 40)
        lines.append(f"  Without NadirClaw:  ${proj['monthly_baseline']:.2f}/mo")
        lines.append(f"  With NadirClaw:     ${proj['monthly_actual']:.2f}/mo")
        lines.append(f"  Monthly savings:    ${proj['monthly_savings']:.2f}/mo")

    # Pro upsell at the moment of highest intent: when the user has just
    # seen their own savings number. Only shown if savings are positive.
    savings_value = report.get("savings", 0) or 0
    if savings_value > 0:
        monthly = (proj or {}).get("monthly_savings", 0) or 0
        lines.append("")
        lines.append("=" * 50)
        lines.append("Nadir Pro: trained classifier finds 10-20% more savings on the")
        lines.append("same traffic, plus a live dashboard and team analytics.")
        if monthly > 0:
            lines.append(
                f"At your current run rate, that is ~${monthly * 0.15:.2f}-${monthly * 0.20:.2f}/mo extra."
            )
        lines.append("Start the 30-day Pro trial: https://getnadir.com?ref=cli-savings")

    lines.append("")
    return "\n".join(lines)


def _safe_int(val: Any) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0
