"""Web-based dashboard for NadirClaw.

Serves a single-page HTML dashboard at /dashboard that shows:
- Real-time routing stats (requests, tier distribution)
- Cost tracking and savings
- Model usage breakdown
- Recent request log

Auto-refreshes every 5 seconds via fetch().
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from nadirclaw.auth import UserSession, validate_local_auth
from nadirclaw.settings import settings

router = APIRouter()


def _load_recent_logs(limit: int = 200) -> List[Dict[str, Any]]:
    """Load recent log entries."""
    log_path = settings.LOG_DIR / "requests.jsonl"
    if not log_path.exists():
        return []
    lines = log_path.read_text().strip().split("\n")
    recent = lines[-limit:] if len(lines) > limit else lines
    entries = []
    for line in reversed(recent):
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return entries


# Reason bucketing matches on error_type class names for stable classification.
# Schema: fallback_reasons entries are {"model", "error_type", "message"} (PR #47).
_REASON_BUCKETS: Dict[str, set] = {
    "rate_limit": {"RateLimitExhausted", "RateLimitError"},
    "timeout": {"ReadTimeout", "ConnectTimeout", "Timeout", "TimeoutError"},
    "connection_error": {"ConnectError", "APIConnectionError"},
    "disconnected": {"RemoteProtocolError", "EndOfStream"},
}


def _classify_error_type(error_type: str) -> str:
    """Map an error_type class name to a dashboard reason bucket."""
    for bucket, names in _REASON_BUCKETS.items():
        if error_type in names:
            return bucket
    if error_type.endswith("ServerError"):
        return "server_error"
    return "other"


def compute_model_stats(completions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-model success/fail counts and categorized reasons.

    Success = request landed on `selected_model`. Fail = every entry in
    `fallback_reasons`, attributed to the model that actually failed.
    """
    stats: Dict[str, Dict[str, Any]] = {}
    for e in completions:
        model = e.get("selected_model", "unknown")
        stats.setdefault(model, {"success": 0, "fail": 0, "fail_reasons": {}})
        stats[model]["success"] += 1
        for fr in e.get("fallback_reasons") or []:
            fm = fr.get("model", "unknown")
            stats.setdefault(fm, {"success": 0, "fail": 0, "fail_reasons": {}})
            stats[fm]["fail"] += 1
            bucket = _classify_error_type(fr.get("error_type", ""))
            reasons = stats[fm]["fail_reasons"]
            reasons[bucket] = reasons.get(bucket, 0) + 1
    return stats


@router.get("/dashboard/api/stats")
async def dashboard_stats(
    current_user: UserSession = Depends(validate_local_auth),
) -> Dict[str, Any]:
    """API endpoint for dashboard data."""
    from nadirclaw.budget import get_budget_tracker
    from nadirclaw.savings import calculate_actual_cost, get_model_cost

    entries = _load_recent_logs(500)
    completions = [e for e in entries if e.get("type") == "completion" and e.get("status") == "ok"]

    # Tier distribution
    tiers: Dict[str, int] = {}
    for e in completions:
        tier = e.get("tier", "unknown")
        tiers[tier] = tiers.get(tier, 0) + 1

    # Model usage
    models: Dict[str, Dict[str, Any]] = {}
    for e in completions:
        model = e.get("selected_model", "unknown")
        if model not in models:
            models[model] = {"requests": 0, "tokens": 0, "cost": 0.0, "avg_latency_ms": 0, "latencies": []}
        models[model]["requests"] += 1
        tokens = (e.get("prompt_tokens") or 0) + (e.get("completion_tokens") or 0)
        models[model]["tokens"] += tokens
        cost = e.get("cost", 0) or 0
        models[model]["cost"] += cost
        lat = e.get("total_latency_ms", 0) or 0
        if lat > 0:
            models[model]["latencies"].append(lat)

    # Calculate avg latency
    for m in models.values():
        lats = m.pop("latencies")
        m["avg_latency_ms"] = round(sum(lats) / len(lats)) if lats else 0

    # Recent requests (last 20)
    recent = []
    for e in completions[:20]:
        recent.append({
            "time": e.get("timestamp", ""),
            "model": e.get("selected_model", ""),
            "tier": e.get("tier", ""),
            "latency_ms": e.get("total_latency_ms", 0),
            "tokens": (e.get("prompt_tokens") or 0) + (e.get("completion_tokens") or 0),
            "cost": e.get("cost", 0),
            "prompt": (e.get("prompt", "") or "")[:60],
            "fallback": e.get("fallback_used"),
            "tokens_saved": e.get("tokens_saved", 0) or 0,
        })

    # Budget
    budget = get_budget_tracker().get_status()

    # Fallback stats
    fallbacks = sum(1 for e in completions if e.get("fallback_used"))

    # Model success/failure tracking from fallback_reasons (schema: PR #47).
    model_stats = compute_model_stats(completions)

    # Optimization stats
    total_tokens_saved = sum(e.get("tokens_saved", 0) or 0 for e in completions)
    total_original_tokens = sum(e.get("original_tokens", 0) or 0 for e in completions if e.get("original_tokens"))
    opt_savings_pct = (total_tokens_saved / max(total_original_tokens, 1) * 100) if total_original_tokens else 0
    optimized_requests = sum(1 for e in completions if e.get("optimization_mode") and e.get("optimization_mode") != "off")

    return {
        "total_requests": len(completions),
        "tier_distribution": tiers,
        "model_usage": dict(sorted(models.items(), key=lambda x: x[1]["requests"], reverse=True)),
        "recent_requests": recent,
        "budget": budget,
        "fallback_count": fallbacks,
        "simple_model": settings.SIMPLE_MODEL,
        "complex_model": settings.COMPLEX_MODEL,
        "optimization": {
            "total_tokens_saved": total_tokens_saved,
            "savings_pct": round(opt_savings_pct, 1),
            "optimized_requests": optimized_requests,
        },
        "model_stats": model_stats,
    }


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    """Serve the web dashboard HTML."""
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NadirClaw Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: #0f0f13; color: #e0e0e0; }
.header { padding: 1.5rem 2rem; border-bottom: 1px solid #1e1e2e; display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 1.3rem; font-weight: 700; color: #fff; }
.header h1 span { color: #a78bfa; }
.header .status { font-size: 0.8rem; color: #6b7280; }
.header .status .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #34d399; margin-right: 6px; }
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; padding: 1.5rem 2rem; }
.card { background: #1a1a24; border-radius: 12px; padding: 1.25rem; }
.card-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6b7280; margin-bottom: 0.5rem; }
.card-value { font-size: 1.8rem; font-weight: 700; color: #fff; }
.card-value.green { color: #34d399; }
.card-value.purple { color: #a78bfa; }
.card-value.amber { color: #fbbf24; }
.card-sub { font-size: 0.78rem; color: #6b7280; margin-top: 0.25rem; }
.section { padding: 0 2rem 1.5rem; }
.section-title { font-size: 0.85rem; font-weight: 600; color: #9ca3af; margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
.table-wrap { background: #1a1a24; border-radius: 12px; overflow: hidden; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
th { text-align: left; padding: 0.75rem 1rem; color: #6b7280; font-weight: 500; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; border-bottom: 1px solid #252535; }
td { padding: 0.6rem 1rem; border-bottom: 1px solid #1e1e2e; }
tr:last-child td { border: none; }
.tier-badge { display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 600; }
.tier-simple { background: #064e3b; color: #34d399; }
.tier-complex { background: #4c1d95; color: #a78bfa; }
.tier-reasoning { background: #78350f; color: #fbbf24; }
.tier-direct { background: #1e293b; color: #94a3b8; }
.tier-free { background: #1e3a2f; color: #6ee7b7; }
.bar-wrap { display: flex; gap: 4px; height: 24px; border-radius: 6px; overflow: hidden; }
.bar-seg { transition: width 0.3s ease; }
.bar-simple { background: #34d399; }
.bar-complex { background: #a78bfa; }
.bar-reasoning { background: #fbbf24; }
.bar-other { background: #4b5563; }
.model-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }
.model-card { background: #1a1a24; border-radius: 12px; padding: 1rem 1.25rem; }
.model-name { font-size: 0.82rem; font-weight: 600; color: #fff; margin-bottom: 0.5rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.model-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem; font-size: 0.75rem; color: #9ca3af; }
.model-stats span { color: #e0e0e0; font-weight: 500; }
.fallback-tag { color: #f87171; font-size: 0.7rem; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
@media (max-width: 900px) { .grid { grid-template-columns: repeat(2, 1fr); } .two-col { grid-template-columns: 1fr; } }
@media (max-width: 600px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="header">
  <h1>Nadir<span>Claw</span> Dashboard</h1>
  <div class="status"><span class="dot"></span>Live &mdash; refreshing every 5s</div>
</div>

<div class="grid" id="stats-cards">
  <div class="card"><div class="card-label">Total Requests</div><div class="card-value" id="total-reqs">-</div></div>
  <div class="card"><div class="card-label">Today's Spend</div><div class="card-value green" id="daily-spend">-</div><div class="card-sub" id="daily-budget"></div></div>
  <div class="card"><div class="card-label">Monthly Spend</div><div class="card-value purple" id="monthly-spend">-</div><div class="card-sub" id="monthly-budget"></div></div>
  <div class="card"><div class="card-label">Fallbacks</div><div class="card-value amber" id="fallback-count">-</div><div class="card-sub">auto-recovered</div></div>
</div>

<div class="section">
  <div class="section-title">Routing Distribution</div>
  <div class="bar-wrap" id="tier-bar" style="margin-bottom: 0.5rem;"></div>
  <div id="tier-legend" style="font-size: 0.75rem; color: #9ca3af;"></div>
</div>

<div class="section two-col">
  <div>
    <div class="section-title">Models</div>
    <div class="model-grid" id="model-grid"></div>
  </div>
  <div>
    <div class="section-title">Recent Requests</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Time</th><th>Model</th><th>Tier</th><th>Latency</th><th>Tokens</th><th>Prompt</th></tr></thead>
        <tbody id="recent-body"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const TIER_COLORS = { simple: '#34d399', complex: '#a78bfa', reasoning: '#fbbf24', direct: '#94a3b8', free: '#6ee7b7' };
const TIER_CLASSES = { simple: 'tier-simple', complex: 'tier-complex', reasoning: 'tier-reasoning', direct: 'tier-direct', free: 'tier-free' };

async function refresh() {
  try {
    const res = await fetch('/dashboard/api/stats');
    const d = await res.json();

    document.getElementById('total-reqs').textContent = d.total_requests.toLocaleString();
    document.getElementById('daily-spend').textContent = '$' + (d.budget.daily_spend || 0).toFixed(4);
    document.getElementById('monthly-spend').textContent = '$' + (d.budget.monthly_spend || 0).toFixed(4);
    document.getElementById('fallback-count').textContent = d.fallback_count;

    if (d.budget.daily_budget) document.getElementById('daily-budget').textContent = 'of $' + d.budget.daily_budget.toFixed(2) + ' budget';
    if (d.budget.monthly_budget) document.getElementById('monthly-budget').textContent = 'of $' + d.budget.monthly_budget.toFixed(2) + ' budget';

    // Tier bar
    const total = Object.values(d.tier_distribution).reduce((a,b) => a+b, 0) || 1;
    const bar = document.getElementById('tier-bar');
    const legend = document.getElementById('tier-legend');
    bar.innerHTML = '';
    legend.innerHTML = '';
    for (const [tier, count] of Object.entries(d.tier_distribution).sort((a,b) => b[1]-a[1])) {
      const pct = (count / total * 100);
      const seg = document.createElement('div');
      seg.className = 'bar-seg';
      seg.style.width = pct + '%';
      seg.style.background = TIER_COLORS[tier] || '#4b5563';
      bar.appendChild(seg);
      legend.innerHTML += '<span style="color:' + (TIER_COLORS[tier]||'#4b5563') + '">' + tier + ' ' + count + ' (' + pct.toFixed(0) + '%)</span>  ';
    }

    // Model cards (with success rate from model_stats)
    const ms = d.model_stats || {};
    const mg = document.getElementById('model-grid');
    mg.innerHTML = '';
    for (const [name, info] of Object.entries(d.model_usage)) {
      const msEntry = ms[name];
      const success = msEntry ? msEntry.success : info.requests;
      const fail = msEntry ? msEntry.fail : 0;
      const total = success + fail;
      const rate = total > 0 ? (success / total * 100) : 100;
      const rateColor = rate >= 95 ? '#34d399' : rate >= 80 ? '#fbbf24' : '#ef4444';
      let card = '<div class="model-card"><div class="model-name">' + name + '</div><div class="model-stats">' +
        '<div>Requests <span>' + info.requests + '</span></div>';
      if (fail > 0) {
        card += '<div>Success <span style="color:' + rateColor + '">' + success + '/' + total + ' (' + rate.toFixed(0) + '%)</span></div>';
        const reasons = Object.entries(msEntry.fail_reasons || {}).map(([r,c]) => r+':'+c).join(', ');
        if (reasons) card += '<div style="color:#6b7280;font-size:0.7rem;grid-column:span 2">' + reasons + '</div>';
      }
      card += '<div>Tokens <span>' + info.tokens.toLocaleString() + '</span></div>' +
        '<div>Cost <span>$' + info.cost.toFixed(4) + '</span></div>' +
        '<div>Avg latency <span>' + info.avg_latency_ms + 'ms</span></div>' +
        '</div></div>';
      mg.innerHTML += card;
    }

    // Recent
    const rb = document.getElementById('recent-body');
    rb.innerHTML = '';
    for (const r of d.recent_requests) {
      const t = r.time ? new Date(r.time).toLocaleTimeString() : '-';
      const tc = TIER_CLASSES[r.tier] || 'tier-direct';
      const fb = r.fallback ? ' <span class="fallback-tag">⚡fallback</span>' : '';
      rb.innerHTML += '<tr><td>' + t + '</td><td style="font-size:0.75rem">' + r.model + fb + '</td><td><span class="tier-badge ' + tc + '">' + r.tier + '</span></td><td>' + (r.latency_ms||0) + 'ms</td><td>' + r.tokens + '</td><td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + (r.prompt||'').replace(/</g,'&lt;') + '</td></tr>';
    }
  } catch(e) { console.error('Dashboard refresh error:', e); }
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""
