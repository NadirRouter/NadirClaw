"""NadirClaw CLI — serve, classify, onboard, and status commands."""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import click


@click.group(invoke_without_command=True)
@click.version_option(version=None, prog_name="nadirclaw", package_name="nadirclaw")
@click.pass_context
def main(ctx):
    """NadirClaw — Open-source LLM router."""
    if ctx.invoked_subcommand is None:
        click.echo("NadirClaw — Open-source LLM router\n")
        click.echo("Quick start:")
        click.echo(f"  {click.style('nadirclaw setup', bold=True)}      Configure providers and models")
        click.echo(f"  {click.style('nadirclaw serve', bold=True)}      Start the router on localhost:8856")
        click.echo(f"  {click.style('nadirclaw demo', bold=True)}       See routing in action (no API keys needed)")
        click.echo(f"  {click.style('nadirclaw classify', bold=True)}   Test the classifier on a prompt")
        click.echo()
        click.echo("Monitoring:")
        click.echo(f"  {click.style('nadirclaw report', bold=True)}     Cost and usage report")
        click.echo(f"  {click.style('nadirclaw dashboard', bold=True)}  Live terminal dashboard")
        click.echo(f"  {click.style('nadirclaw savings', bold=True)}    See how much you've saved")
        click.echo()
        click.echo(f"Run {click.style('nadirclaw --help', bold=True)} for all commands.")


@main.command()
@click.option("--count", default=10, help="Number of sample prompts to classify")
def demo(count):
    """Run a quick demo — classify sample prompts and show projected savings."""
    click.echo("NadirClaw Demo — classifying sample prompts...\n")

    sample_prompts = [
        ("What is 2+2?", "simple"),
        ("Format this JSON for me", "simple"),
        ("Translate 'hello' to French", "simple"),
        ("Write a docstring for this function", "simple"),
        ("List all files in the current directory", "simple"),
        ("Refactor this authentication module to use JWT tokens with refresh token rotation, add rate limiting per user, and write integration tests", "complex"),
        ("Debug this race condition in our async task queue that causes duplicate processing under high load", "complex"),
        ("Design a microservices architecture for a real-time multiplayer game with matchmaking, leaderboards, and replay storage", "complex"),
        ("Explain the difference between TCP and UDP", "medium"),
        ("Write a Python function to validate email addresses", "medium"),
        ("How do I set up a basic Express.js server?", "medium"),
        ("What are the pros and cons of NoSQL vs SQL databases?", "medium"),
        ("Create a React component that fetches and displays a list of users", "medium"),
    ]

    # Use at most `count` prompts
    prompts = sample_prompts[:count]

    from nadirclaw.classifier import get_classifier
    classifier = get_classifier()

    simple_count = 0
    mid_count = 0
    complex_count = 0

    # Cost assumptions (per request, approximate)
    premium_cost = 0.045  # avg cost if all went to premium model
    simple_cost = 0.002
    mid_cost = 0.012

    for prompt_text, expected in prompts:
        result = classifier.classify(prompt_text)
        # Handle different return types from different classifiers
        if isinstance(result, tuple):
            if len(result) == 2:
                is_complex, confidence = result
                tier = "complex" if is_complex else "simple"
            elif len(result) == 3:
                tier, confidence, _ = result
        elif isinstance(result, dict):
            tier = result.get("tier_name", "unknown")
            confidence = result.get("confidence", 0)
        else:
            tier = str(result)
            confidence = 0.0

        # Normalize tier name
        if tier in ("mid", "medium"):
            tier_display = "MEDIUM"
            mid_count += 1
        elif tier == "complex":
            tier_display = "COMPLEX"
            complex_count += 1
        else:
            tier_display = "SIMPLE"
            simple_count += 1

        # Color coding
        if tier_display == "SIMPLE":
            color = "green"
        elif tier_display == "MEDIUM":
            color = "yellow"
        else:
            color = "blue"

        # Truncate prompt for display
        display_prompt = prompt_text[:60] + "..." if len(prompt_text) > 60 else prompt_text
        click.echo(f"  {click.style(tier_display.ljust(7), fg=color, bold=True)}  \"{display_prompt}\"")

    total = len(prompts)
    routed_cheaper = simple_count + mid_count
    all_premium_cost = total * premium_cost
    routed_cost = (simple_count * simple_cost) + (mid_count * mid_cost) + (complex_count * premium_cost)
    savings_pct = ((all_premium_cost - routed_cost) / all_premium_cost * 100) if all_premium_cost > 0 else 0

    click.echo(f"\n{'─' * 50}")
    click.echo(f"  {click.style(str(routed_cheaper), fg='green', bold=True)} of {total} routed to cheaper models")
    click.echo(f"  Projected cost with routing:    {click.style(f'${routed_cost:.3f}', fg='green')}")
    click.echo(f"  Cost without routing:           ${all_premium_cost:.3f}")
    click.echo(f"  {click.style(f'Estimated savings: {savings_pct:.0f}%', fg='green', bold=True)}")
    click.echo(f"\n  Ready to start saving? Run:")
    click.echo(f"    {click.style('nadirclaw setup', bold=True)}    (configure your API keys)")
    click.echo(f"    {click.style('nadirclaw serve', bold=True)}    (start the router)")


@main.command()
@click.option("--reconfigure", is_flag=True, help="Re-run setup even if configured")
def setup(reconfigure):
    """Interactive setup wizard — configure providers and models."""
    from nadirclaw.setup import is_first_run, run_setup_wizard

    if not reconfigure and not is_first_run():
        if not click.confirm("Already configured. Re-run setup?", default=False):
            return
        reconfigure = True
    run_setup_wizard(reconfigure=reconfigure)


@main.command()
@click.option("--port", default=None, type=int, help="Port to listen on (default: 8856)")
@click.option("--simple-model", default=None, help="Model for simple prompts")
@click.option("--complex-model", default=None, help="Model for complex prompts")
@click.option("--models", default=None, help="Comma-separated model list (legacy)")
@click.option("--token", default=None, help="Auth token")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--log-raw", is_flag=True, help="Log full raw requests and responses to JSONL")
@click.option("--optimize", default=None, type=click.Choice(["off", "safe", "aggressive"]),
              help="Context optimization mode (default: off)")
def serve(port, simple_model, complex_model, models, token, verbose, log_raw, optimize):
    """Start the NadirClaw router server."""
    import logging

    from nadirclaw.setup import is_first_run

    if is_first_run():
        if click.confirm("No configuration found. Run setup wizard?", default=True):
            from nadirclaw.setup import run_setup_wizard
            run_setup_wizard()
        else:
            click.echo("Starting with defaults. Run 'nadirclaw setup' anytime.")

    # Override env vars from CLI flags
    if port:
        os.environ["NADIRCLAW_PORT"] = str(port)
    if simple_model:
        os.environ["NADIRCLAW_SIMPLE_MODEL"] = simple_model
    if complex_model:
        os.environ["NADIRCLAW_COMPLEX_MODEL"] = complex_model
    if models:
        os.environ["NADIRCLAW_MODELS"] = models
    if token:
        os.environ["NADIRCLAW_AUTH_TOKEN"] = token
    if log_raw:
        os.environ["NADIRCLAW_LOG_RAW"] = "true"
    if optimize:
        os.environ["NADIRCLAW_OPTIMIZE"] = optimize

    log_level = "debug" if verbose else "info"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    import uvicorn

    from nadirclaw.settings import settings

    actual_port = port or settings.PORT
    click.echo(f"Starting NadirClaw on port {actual_port}...")
    click.echo(f"  Simple model:  {settings.SIMPLE_MODEL}")
    click.echo(f"  Complex model: {settings.COMPLEX_MODEL}")
    if settings.OPTIMIZE != "off":
        click.echo(f"  Optimize:      {settings.OPTIMIZE}")

    click.echo()
    click.echo(f"  Point your tools to: {click.style(f'http://localhost:{actual_port}/v1', fg='blue', bold=True)}")
    click.echo(f"  Dashboard:           {click.style(f'http://localhost:{actual_port}/dashboard', fg='blue')}")
    click.echo(f"  Health check:        curl http://localhost:{actual_port}/health")
    click.echo()

    uvicorn.run(
        "nadirclaw.server:app",
        host="0.0.0.0",
        port=actual_port,
        log_level=log_level,
    )


@main.command()
@click.argument("prompt", nargs=-1, required=True)
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json"]), help="Output format")
@click.option("--cascade", is_flag=True, help="Use cascade classifier (ternary)")
def classify(prompt, fmt, cascade):
    """Classify a prompt as simple, medium, or complex (no server needed)."""
    import logging

    logging.basicConfig(level=logging.WARNING)

    from nadirclaw.settings import settings

    prompt_text = " ".join(prompt)

    use_cascade = cascade or settings.CLASSIFIER == "cascade"

    if use_cascade:
        from nadirclaw.classifier import ConfidenceAwareCascadeClassifier

        clf = ConfidenceAwareCascadeClassifier()
        tier_name, confidence, metadata = clf.classify(prompt_text)
        from nadirclaw.classifier import _tier_to_score
        score = _tier_to_score(tier_name, confidence)
        is_complex = tier_name in ("medium", "complex")
        escalated = metadata.get("confidence_escalated", False)
    else:
        from nadirclaw.classifier import BinaryComplexityClassifier

        clf = BinaryComplexityClassifier()
        is_complex, confidence = clf.classify(prompt_text)
        tier_name = "complex" if is_complex else "simple"
        score = clf._confidence_to_score(is_complex, confidence)
        escalated = False

    # Pick model from explicit tier config
    if tier_name == "complex":
        model = settings.COMPLEX_MODEL
    elif tier_name in ("mid", "medium"):
        model = settings.MID_MODEL
    else:
        model = settings.SIMPLE_MODEL

    if fmt == "json":
        result = {
            "tier": tier_name,
            "is_complex": is_complex,
            "confidence": round(confidence, 4),
            "score": round(score, 4),
            "model": model,
            "prompt": prompt_text,
            "classifier": "cascade" if use_cascade else "binary",
        }
        if use_cascade:
            result["escalated"] = escalated
        click.echo(json.dumps(result))
    else:
        click.echo(f"Classifier: {'cascade' if use_cascade else 'binary'}")
        click.echo(f"Tier:       {tier_name}")
        click.echo(f"Confidence: {confidence:.4f}")
        click.echo(f"Score:      {score:.4f}")
        click.echo(f"Model:      {model}")
        if use_cascade and escalated:
            click.echo(f"Escalated:  yes")


@main.command("optimize")
@click.argument("file", type=click.Path(exists=True), required=False)
@click.option("--mode", default="safe", type=click.Choice(["safe", "aggressive"]),
              help="Optimization mode")
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json"]),
              help="Output format")
def optimize_cmd(file, mode, fmt):
    """Test context optimization on a file (or stdin). Dry-run — shows before/after."""
    import sys

    from nadirclaw.optimize import optimize_messages

    if file:
        with open(file) as f:
            content = f.read()
    else:
        if sys.stdin.isatty():
            click.echo("Reading from stdin (Ctrl-D to end)...")
        content = sys.stdin.read()

    # Try to parse as JSON messages array, or wrap in a single user message
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "messages" in parsed:
            messages = parsed["messages"]
        elif isinstance(parsed, list):
            messages = parsed
        else:
            messages = [{"role": "user", "content": content}]
    except json.JSONDecodeError:
        messages = [{"role": "user", "content": content}]

    result = optimize_messages(messages, mode=mode)

    if fmt == "json":
        click.echo(json.dumps({
            "mode": result.mode,
            "original_tokens": result.original_tokens,
            "optimized_tokens": result.optimized_tokens,
            "tokens_saved": result.tokens_saved,
            "savings_pct": round(result.tokens_saved / max(result.original_tokens, 1) * 100, 1),
            "optimizations_applied": result.optimizations_applied,
            "messages": result.messages,
        }, indent=2))
    else:
        click.echo(f"Mode:          {result.mode}")
        click.echo(f"Original:      ~{result.original_tokens} tokens")
        click.echo(f"Optimized:     ~{result.optimized_tokens} tokens")
        savings_pct = result.tokens_saved / max(result.original_tokens, 1) * 100
        click.echo(f"Saved:         ~{result.tokens_saved} tokens ({savings_pct:.1f}%)")
        if result.optimizations_applied:
            click.echo(f"Transforms:    {', '.join(result.optimizations_applied)}")
        else:
            click.echo("Transforms:    (none applied)")


@main.command()
def status():
    """Check if NadirClaw server is running and show config."""
    import urllib.request

    from nadirclaw.credentials import list_credentials
    from nadirclaw.settings import settings

    click.echo("NadirClaw Status")
    click.echo("-" * 40)
    click.echo(f"Simple model:  {settings.SIMPLE_MODEL}")
    click.echo(f"Complex model: {settings.COMPLEX_MODEL}")
    if settings.has_explicit_tiers:
        click.echo("Tier config:   explicit (env vars)")
    else:
        click.echo("Tier config:   derived from NADIRCLAW_MODELS")
    click.echo(f"Port:          {settings.PORT}")
    click.echo(f"Threshold:     {settings.CONFIDENCE_THRESHOLD}")
    click.echo(f"Log dir:       {settings.LOG_DIR}")
    token = settings.AUTH_TOKEN
    if token:
        click.echo(f"Auth:          {token[:6]}***" if len(token) >= 6 else f"Auth:          {token}")
    else:
        click.echo("Auth:          disabled (local-only)")

    # Show credential status
    creds = list_credentials()
    if creds:
        click.echo(f"\nCredentials:   {len(creds)} provider(s)")
        for c in creds:
            click.echo(f"  {c['provider']:12s}  {c['masked_token']}  ({c['source']})")
    else:
        click.echo("\nCredentials:   none configured")
        click.echo("  Run 'nadirclaw auth add' or set env vars (ANTHROPIC_API_KEY, etc.)")

    # Check if server is running
    try:
        url = f"http://localhost:{settings.PORT}/health"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            click.echo(f"\nServer:        RUNNING ({data.get('status', '?')})")
    except Exception:
        click.echo("\nServer:        NOT RUNNING")


@main.command()
@click.option("--since", default=None, help="Time filter: '24h', '7d', '2025-02-01'")
@click.option("--model", default=None, help="Filter by model name (substring match)")
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json"]), help="Output format")
@click.option("--export", "export_path", default=None, type=click.Path(), help="Export report to file")
@click.option("--by-model", is_flag=True, help="Show per-model cost breakdown")
@click.option("--by-day", is_flag=True, help="Show per-day cost breakdown")
def report(since, model, fmt, export_path, by_model, by_day):
    """Show a summary report of request logs (reads SQLite first, falls back to JSONL)."""
    from nadirclaw.report import (
        format_cost_breakdown_text,
        format_report_text,
        generate_cost_breakdown,
        generate_report,
        load_log_entries,
        load_log_entries_sqlite,
        parse_since,
    )
    from nadirclaw.settings import settings

    db_path = settings.LOG_DIR / "requests.db"
    jsonl_path = settings.LOG_DIR / "requests.jsonl"

    if not db_path.exists() and not jsonl_path.exists():
        click.echo("No log file found. Start the server and make some requests first.")
        return

    since_dt = None
    if since:
        try:
            since_dt = parse_since(since)
        except ValueError as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)

    # Prefer SQLite (richer data), fall back to JSONL
    if db_path.exists():
        entries = load_log_entries_sqlite(db_path, since=since_dt, model_filter=model)
    else:
        entries = load_log_entries(jsonl_path, since=since_dt, model_filter=model)

    if by_model or by_day:
        # Cost breakdown mode
        breakdown_data = generate_cost_breakdown(entries, by_model=by_model, by_day=by_day)
        if fmt == "json":
            output = json.dumps(breakdown_data, indent=2, default=str)
        else:
            output = format_cost_breakdown_text(breakdown_data)
    else:
        report_data = generate_report(entries)
        if fmt == "json":
            output = json.dumps(report_data, indent=2, default=str)
        else:
            output = format_report_text(report_data)

    if export_path:
        Path(export_path).write_text(output)
        click.echo(f"Report exported to {export_path}")
    else:
        click.echo(output)


@main.command()
@click.option("--refresh", default=2.0, type=float, help="Refresh interval in seconds")
def dashboard(refresh):
    """Live terminal dashboard showing real-time routing stats.

    For a web-based dashboard, visit http://localhost:8856/dashboard
    while the server is running.
    """
    from nadirclaw.dashboard import run_dashboard
    from nadirclaw.settings import settings

    log_path = settings.LOG_DIR / "requests.jsonl"
    db_path = settings.LOG_DIR / "requests.db"
    run_dashboard(log_path, refresh=refresh, db_path=db_path)


@main.command()
@click.option("--since", default=None, help="Time filter: '24h', '7d', '2025-02-01'")
@click.option("--baseline", default=None, help="Model to compare against (default: most expensive in logs)")
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json"]), help="Output format")
def savings(since, baseline, fmt):
    """Show how much money NadirClaw saved you."""
    from nadirclaw.report import load_log_entries_sqlite, load_log_entries, parse_since
    from nadirclaw.savings import format_savings_text, generate_savings_report
    from nadirclaw.settings import settings

    db_path = settings.LOG_DIR / "requests.db"
    log_path = settings.LOG_DIR / "requests.jsonl"

    if not db_path.exists() and not log_path.exists():
        click.echo("No log file found. Start the server and make some requests first.")
        return

    since_dt = None
    if since:
        try:
            since_dt = parse_since(since)
        except ValueError as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)

    # Prefer SQLite (richer data), fall back to JSONL — mirrors the report command
    if db_path.exists():
        entries = load_log_entries_sqlite(db_path, since=since_dt)
    else:
        entries = load_log_entries(log_path, since=since_dt)

    report_data = generate_savings_report(log_path, since=since, baseline_model=baseline, entries=entries)

    if fmt == "json":
        output = json.dumps(report_data, indent=2, default=str)
    else:
        output = format_savings_text(report_data)

    click.echo(output)


@main.command()
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json"]), help="Output format")
def budget(fmt):
    """Show current spend and budget status."""
    from nadirclaw.budget import get_budget_tracker

    tracker = get_budget_tracker()
    status = tracker.get_status()

    if fmt == "json":
        click.echo(json.dumps(status, indent=2))
        return

    click.echo("NadirClaw Budget Status")
    click.echo("=" * 45)

    # Daily
    daily = status["daily_spend"]
    daily_budget = status["daily_budget"]
    click.echo(f"  Today:   ${daily:.4f}" + (f" / ${daily_budget:.2f}" if daily_budget else ""))
    click.echo(f"  Reqs:    {status['daily_requests']}")

    # Monthly
    monthly = status["monthly_spend"]
    monthly_budget = status["monthly_budget"]
    click.echo(f"  Month:   ${monthly:.4f}" + (f" / ${monthly_budget:.2f}" if monthly_budget else ""))
    click.echo(f"  Reqs:    {status['monthly_requests']}")

    # Top models
    top = status.get("top_models", [])
    if top:
        click.echo("")
        click.echo("Top Models by Spend")
        click.echo("-" * 45)
        for m in top[:5]:
            click.echo(f"  {m['model']:35s}  ${m['spend']:.4f}  ({m['requests']} reqs)")

    if not daily_budget and not monthly_budget:
        click.echo("")
        click.echo("Tip: Set NADIRCLAW_DAILY_BUDGET=5.00 and/or")
        click.echo("     NADIRCLAW_MONTHLY_BUDGET=50.00 to enable alerts.")


@main.command()
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json"]), help="Output format")
def cache(fmt):
    """Show prompt cache statistics (queries running server)."""
    import urllib.request

    from nadirclaw.settings import settings

    try:
        url = f"http://localhost:{settings.PORT}/v1/cache"
        headers = {}
        token = settings.AUTH_TOKEN
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        click.echo(f"Could not reach NadirClaw server: {e}")
        click.echo("Make sure the server is running (nadirclaw serve).")
        raise SystemExit(1)

    if fmt == "json":
        click.echo(json.dumps(data, indent=2))
        return

    click.echo("NadirClaw Prompt Cache")
    click.echo("=" * 40)
    click.echo(f"  Enabled:    {data.get('enabled', '?')}")
    click.echo(f"  Entries:    {data.get('entries', 0)} / {data.get('max_size', '?')}")
    click.echo(f"  TTL:        {data.get('ttl', '?')}s")
    click.echo(f"  Hits:       {data.get('hits', 0)}")
    click.echo(f"  Misses:     {data.get('misses', 0)}")
    hit_rate = data.get('hit_rate', 0)
    click.echo(f"  Hit rate:   {hit_rate:.1%}")
    click.echo(f"  Lookups:    {data.get('total_lookups', 0)}")


@main.command()
@click.option("--format", "fmt", default="csv", type=click.Choice(["csv", "jsonl"]), help="Export format")
@click.option("--since", default=None, help="Time filter: '24h', '7d', '2025-02-01'")
@click.option("--model", default=None, help="Filter by model name (substring match)")
@click.option("--output", "-o", "output_path", default=None, type=click.Path(), help="Output file (default: stdout)")
def export(fmt, since, model, output_path):
    """Export request logs for offline analysis."""
    import csv
    import io

    from nadirclaw.report import load_log_entries, load_log_entries_sqlite, parse_since
    from nadirclaw.settings import settings

    db_path = settings.LOG_DIR / "requests.db"
    jsonl_path = settings.LOG_DIR / "requests.jsonl"

    if not db_path.exists() and not jsonl_path.exists():
        click.echo("No log file found. Start the server and make some requests first.")
        return

    since_dt = None
    if since:
        try:
            since_dt = parse_since(since)
        except ValueError as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)

    # Prefer SQLite
    if db_path.exists():
        entries = load_log_entries_sqlite(db_path, since=since_dt, model_filter=model)
    else:
        entries = load_log_entries(jsonl_path, since=since_dt, model_filter=model)

    if not entries:
        click.echo("No matching entries found.")
        return

    if fmt == "csv":
        # Determine columns from first entry
        columns = list(entries[0].keys())

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)
        output = buf.getvalue()
    else:
        # JSONL
        lines = [json.dumps(entry, default=str) for entry in entries]
        output = "\n".join(lines) + "\n"

    if output_path:
        Path(output_path).write_text(output)
        click.echo(f"Exported {len(entries)} entries to {output_path}")
    else:
        click.echo(output, nl=False)


@main.command(name="export-labeled")
@click.option("--format", "fmt", default="json", type=click.Choice(["json", "jsonl"]), help="Export format")
@click.option("--since", default=None, help="Start date filter (ISO format or YYYY-MM-DD)")
@click.option("--until", default=None, help="End date filter (ISO format or YYYY-MM-DD)")
@click.option("--output", "-o", "output_path", default=None, type=click.Path(), help="Output file (default: stdout)")
def export_labeled(fmt, since, until, output_path):
    """Export human-labeled prompts for classifier retraining.

    \b
    Examples:
      nadirclaw export-labeled                              # all labels as JSON
      nadirclaw export-labeled --format jsonl -o train.jsonl
      nadirclaw export-labeled --since 2026-03-01 --until 2026-03-31
    """
    from nadirclaw.request_logger import export_labeled_prompts

    labels = export_labeled_prompts(since=since, until=until)

    if not labels:
        click.echo("No labeled prompts found. Use 'nadirclaw label' to label some requests.")
        return

    if fmt == "json":
        output = json.dumps(labels, indent=2, default=str)
    else:
        lines = [json.dumps(entry, default=str) for entry in labels]
        output = "\n".join(lines) + "\n"

    if output_path:
        Path(output_path).write_text(output)
        click.echo(f"Exported {len(labels)} labeled prompts to {output_path}")
    else:
        click.echo(output, nl=False)


@main.command()
@click.option("--last", "last_n", default=20, type=int, help="Number of recent requests to show (default: 20)")
def label(last_n):
    """Interactively label recent requests with correct tiers for classifier retraining.

    \b
    Shows recent requests from SQLite logs, lets you confirm or correct
    the predicted tier. Labels are stored in the labeled_prompts table
    and can be exported with 'nadirclaw export-labeled'.

    \b
    Examples:
      nadirclaw label            # label last 20 requests
      nadirclaw label --last 50  # label last 50 requests
    """
    from nadirclaw.request_logger import get_recent_requests, store_label
    from nadirclaw.settings import settings

    db_path = settings.LOG_DIR / "requests.db"
    if not db_path.exists():
        click.echo("No request database found. Start the server and make some requests first.")
        return

    requests = get_recent_requests(limit=last_n)
    if not requests:
        click.echo("No requests found in logs.")
        return

    click.echo(f"Found {len(requests)} recent requests. For each, enter the correct tier.")
    click.echo(f"Options: [s]imple, [m]edium, [c]omplex, [enter] to confirm predicted, [q]uit\n")

    labeled_count = 0
    skipped_count = 0
    tier_map = {"s": "simple", "m": "medium", "c": "complex"}

    for i, req in enumerate(requests, 1):
        request_id = req.get("request_id", "?")
        prompt = req.get("prompt", "")
        predicted_tier = req.get("tier", "?")
        confidence = req.get("confidence", 0)
        model = req.get("selected_model", "?")
        timestamp = req.get("timestamp", "?")

        # Truncate prompt for display
        display_prompt = prompt[:120] + "..." if len(prompt) > 120 else prompt
        display_prompt = display_prompt.replace("\n", " ")

        # Color-code the predicted tier
        tier_colors = {"simple": "green", "medium": "yellow", "complex": "blue", "mid": "yellow"}
        tier_color = tier_colors.get(predicted_tier, "white")

        click.echo(f"[{i}/{len(requests)}] {timestamp}")
        click.echo(f"  Prompt:    \"{display_prompt}\"")
        click.echo(f"  Predicted: {click.style(str(predicted_tier), fg=tier_color, bold=True)}  "
                    f"(confidence: {confidence:.3f})" if confidence else
                    f"  Predicted: {click.style(str(predicted_tier), fg=tier_color, bold=True)}")
        click.echo(f"  Model:     {model}")

        choice = click.prompt(
            "  Correct tier",
            default="",
            show_default=False,
            prompt_suffix=" [s/m/c/enter/q]: ",
        ).strip().lower()

        if choice == "q":
            click.echo("\nStopping.")
            break
        elif choice == "":
            # Confirm predicted tier
            correct_tier = predicted_tier if predicted_tier in ("simple", "medium", "complex") else None
            if correct_tier is None:
                # Normalize 'mid' -> 'medium'
                correct_tier = "medium" if predicted_tier == "mid" else "simple"
        elif choice in tier_map:
            correct_tier = tier_map[choice]
        else:
            click.echo("  Skipped (invalid input).")
            skipped_count += 1
            click.echo()
            continue

        result = store_label(
            request_id=request_id,
            correct_tier=correct_tier,
            prompt=prompt,
            system_prompt=req.get("system_prompt"),
            predicted_tier=predicted_tier,
            confidence=confidence,
        )

        if "error" in result:
            click.echo(f"  Error: {result['error']}")
            skipped_count += 1
        else:
            was_correct = correct_tier == predicted_tier or (
                correct_tier == "medium" and predicted_tier == "mid"
            )
            if was_correct:
                click.echo(f"  Confirmed: {click.style(correct_tier, fg='green')}")
            else:
                click.echo(
                    f"  Corrected: {click.style(str(predicted_tier), fg='red')} -> "
                    f"{click.style(correct_tier, fg='green')}"
                )
            labeled_count += 1

        click.echo()

    click.echo(f"{'─' * 50}")
    click.echo(f"Labeled: {labeled_count}  Skipped: {skipped_count}")
    if labeled_count > 0:
        click.echo(f"\nExport with: nadirclaw export-labeled -o training_data.json")


@main.command(name="build-centroids", hidden=True)
@click.option("--ternary", is_flag=True, help="Also build ternary centroids for cascade classifier")
def build_centroids(ternary):
    """Regenerate centroid .npy files from prototype prompts."""
    import logging

    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from nadirclaw.encoder import get_shared_encoder_sync
    from nadirclaw.prototypes import COMPLEX_PROTOTYPES, MEDIUM_PROTOTYPES, SIMPLE_PROTOTYPES

    click.echo("Loading encoder...")
    encoder = get_shared_encoder_sync()

    # --- Binary centroids (always built) ---
    click.echo(f"Encoding {len(SIMPLE_PROTOTYPES)} simple prototypes...")
    simple_embs = encoder.encode(SIMPLE_PROTOTYPES, show_progress_bar=False)
    simple_centroid = simple_embs.mean(axis=0)
    simple_centroid = simple_centroid / np.linalg.norm(simple_centroid)

    click.echo(f"Encoding {len(COMPLEX_PROTOTYPES)} complex prototypes...")
    complex_embs = encoder.encode(COMPLEX_PROTOTYPES, show_progress_bar=False)
    complex_centroid = complex_embs.mean(axis=0)
    complex_centroid = complex_centroid / np.linalg.norm(complex_centroid)

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    simple_path = os.path.join(pkg_dir, "simple_centroid.npy")
    complex_path = os.path.join(pkg_dir, "complex_centroid.npy")

    np.save(simple_path, simple_centroid.astype(np.float32))
    np.save(complex_path, complex_centroid.astype(np.float32))

    click.echo(f"\nSaved: {simple_path}")
    click.echo(f"Saved: {complex_path}")
    click.echo(f"Centroid dimension: {simple_centroid.shape[0]}")

    # --- Ternary centroids (for cascade classifier) ---
    if ternary:
        from nadirclaw.settings import settings

        click.echo(f"\nEncoding {len(MEDIUM_PROTOTYPES)} medium prototypes...")
        medium_embs = encoder.encode(MEDIUM_PROTOTYPES, show_progress_bar=False)
        medium_centroid = medium_embs.mean(axis=0)
        medium_centroid = medium_centroid / np.linalg.norm(medium_centroid)

        # K-means sub-clustering for complex tier
        k = settings.CASCADE_COMPLEX_SUB_CLUSTERS
        k = min(k, len(complex_embs))
        if k >= 2:
            try:
                from sklearn.cluster import KMeans

                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(complex_embs)
                sub_centroids = []
                for i in range(k):
                    cluster_embs = complex_embs[labels == i]
                    if len(cluster_embs) == 0:
                        continue
                    c = cluster_embs.mean(axis=0)
                    norm = np.linalg.norm(c)
                    if norm > 0:
                        c = c / norm
                    sub_centroids.append(c)
                complex_centroids_multi = np.array(sub_centroids)
                click.echo(f"Complex tier: {len(sub_centroids)} sub-centroids via k-means")
            except ImportError:
                click.echo("sklearn not available -- using single complex centroid")
                c = complex_centroid.copy()
                complex_centroids_multi = c.reshape(1, -1)
        else:
            complex_centroids_multi = complex_centroid.reshape(1, -1)

        ternary_path = os.path.join(pkg_dir, "ternary_centroids.npy")
        stacked = np.array(
            [simple_centroid.astype(np.float32),
             medium_centroid.astype(np.float32),
             complex_centroids_multi.astype(np.float32)],
            dtype=object,
        )
        np.save(ternary_path, stacked, allow_pickle=True)

        click.echo(f"Saved: {ternary_path}")
        click.echo(f"Ternary centroids: simple(1) + medium(1) + complex({len(complex_centroids_multi)})")


@main.command(name="train", hidden=True)
def train():
    """Retrain routing centroids. [Nadir Pro]"""
    try:
        from nadir.pro_cli import retrain_command
        retrain_command()
    except ImportError:
        click.echo("Classifier training requires Nadir Pro.")
        click.echo("Install with: pip install nadir")
        raise SystemExit(1)


@main.command(name="_train_legacy", hidden=True)
def _train_legacy():
    """Legacy train command — requires Nadir Pro."""
    click.echo("Classifier training requires Nadir Pro. Install with: pip install nadir")
    raise SystemExit(1)

@main.command()
def retrain():
    """Retrain the classifier from production feedback. [Nadir Pro]"""
    try:
        from nadir.pro_cli import retrain_command
        retrain_command()
    except ImportError:
        click.echo("Adaptive retraining requires Nadir Pro.")
        click.echo("Install with: pip install nadir")
        raise SystemExit(1)


@main.group()
def rules():
    """Manage routing rules."""
    pass


@rules.command("list")
def rules_list():
    """List active routing rules."""
    from nadirclaw.rules import get_rules_engine

    engine = get_rules_engine()
    if engine.rule_count == 0:
        click.echo("No rules configured. Create ~/.nadirclaw/rules.yaml to add rules.")
        return

    click.echo(f"Active rules ({engine.rule_count}):\n")
    for i, rule in enumerate(engine.rules, 1):
        name = getattr(rule, "name", f"rule_{i}")
        conditions = []
        if getattr(rule, "system_prompt_regex", None):
            conditions.append(f"system_prompt~/{rule.system_prompt_regex}/")
        if getattr(rule, "prompt_regex", None):
            conditions.append(f"prompt~/{rule.prompt_regex}/")
        if getattr(rule, "tier_match", None):
            conditions.append(f"tier={rule.tier_match}")
        if getattr(rule, "time_range", None):
            conditions.append(f"time={rule.time_range}")

        action_parts = []
        if getattr(rule, "force_model", None):
            action_parts.append(f"model={rule.force_model}")
        if getattr(rule, "force_tier", None):
            action_parts.append(f"tier={rule.force_tier}")
        if getattr(rule, "max_cost", None):
            action_parts.append(f"max_cost=${rule.max_cost}")

        cond_str = " AND ".join(conditions) or "(always)"
        action_str = ", ".join(action_parts) or "(no action)"
        click.echo(f"  {i}. {name}")
        click.echo(f"     When: {cond_str}")
        click.echo(f"     Then: {action_str}")
        click.echo()


@rules.command("test")
@click.argument("prompt", nargs=-1, required=True)
@click.option("--system", default="", help="System message to test with")
@click.option("--tier", default="", help="Pre-classification tier to simulate")
def rules_test(prompt, system, tier):
    """Test which rule would match a prompt (dry-run).

    \b
    Examples:
      nadirclaw rules test "write a haiku about cats"
      nadirclaw rules test "debug this code" --system "you are a coding agent"
      nadirclaw rules test "translate this" --tier simple
    """
    prompt_text = " ".join(prompt)

    from nadirclaw.rules import get_rules_engine

    engine = get_rules_engine()
    if engine.rule_count == 0:
        click.echo("No rules configured.")
        return

    # Build fake messages
    messages = []
    if system:
        messages.append(type("Msg", (), {"role": "system", "text_content": lambda: system, "content": system, "model_extra": {}})())
    messages.append(type("Msg", (), {"role": "user", "text_content": lambda: prompt_text, "content": prompt_text, "model_extra": {}})())

    result = engine.evaluate(
        messages=messages,
        system_prompt=system,
        tier=tier,
        metadata={"prompt": prompt_text},
    )

    if result.matched:
        click.echo(f"Matched rule: {result.rule_name}")
        if result.force_model:
            click.echo(f"  Force model: {result.force_model}")
        if result.force_tier:
            click.echo(f"  Force tier: {result.force_tier}")
        if result.max_cost_per_request is not None:
            click.echo(f"  Max cost: ${result.max_cost_per_request}")
    else:
        click.echo("No rule matched — routing will use classifier.")


@main.group()
def auth():
    """Manage provider credentials (API keys and tokens)."""
    pass


@auth.command(name="setup-token")
def setup_token():
    """Store a Claude subscription token from 'claude setup-token'."""
    from nadirclaw.credentials import get_credential_source, save_credential

    click.echo("Paste your Claude setup token (from 'claude setup-token'):")
    token = click.prompt("Token", hide_input=True)

    if not token or not token.strip():
        click.echo("Error: empty token provided.")
        raise SystemExit(1)

    token = token.strip()
    save_credential("anthropic", token, source="setup-token")

    click.echo("\nAnthropic credential saved (source: setup-token)")
    click.echo(f"  Token: {token[:8]}...{token[-4:]}" if len(token) > 12 else f"  Token: {token[:4]}***")
    click.echo("\nNadirClaw will use this token for Claude models.")
    click.echo("Verify with: nadirclaw auth status")


# ---------------------------------------------------------------------------
# nadirclaw auth openai — OpenAI subscription OAuth subgroup
# ---------------------------------------------------------------------------

@auth.group(name="openai")
def auth_openai():
    """OpenAI subscription commands (OAuth login with ChatGPT account)."""
    pass


@auth_openai.command(name="login")
@click.option("--timeout", "-t", default=300, help="Login timeout in seconds (default: 300)")
def openai_login(timeout):
    """Login via OAuth — use your ChatGPT subscription, no API key needed.

    Opens a browser for OAuth authorization. No external CLIs required.
    """
    import time as _time
    from nadirclaw.credentials import get_credential, get_credential_source, _read_credentials
    from nadirclaw.oauth import login_openai

    # First check if we already have a valid credential from any source
    existing_token = get_credential("openai-codex")
    existing_source = get_credential_source("openai-codex")
    if existing_token:
        # Check expiry from NadirClaw stored credentials
        stored = _read_credentials().get("openai-codex", {})
        expires_at = stored.get("expires_at", 0)


        if expires_at and _time.time() < (expires_at - 60):
            remaining = int(expires_at - _time.time())
            click.echo(f"You already have valid OpenAI Codex credentials (source: {existing_source}).")
            click.echo(f"  Token expires in: {remaining // 60} minutes")
            click.echo("  NadirClaw will use these automatically.")
            click.echo("\nTo force re-login, run: nadirclaw auth openai logout && nadirclaw auth openai login")
            return

    click.echo("Logging in to OpenAI...")
    click.echo("A browser window will open for you to sign in with your OpenAI account.\n")

    try:
        token_data = login_openai(timeout=timeout)
    except RuntimeError as e:
        click.echo(f"\nLogin failed: {e}")
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"\nUnexpected error during login: {e}")
        raise SystemExit(1)

    if not token_data:
        click.echo("\nLogin did not complete successfully.")
        raise SystemExit(1)

    access_token = token_data.get("access_token", "")
    refresh_token = token_data.get("refresh_token", "")
    expires_at = token_data.get("expires_at", 0)

    if access_token:
        # Also save a copy in NadirClaw's credential store
        from nadirclaw.credentials import save_oauth_credential
        import time as _time
        expires_in = max(int(expires_at - _time.time()), 3600) if expires_at else 3600
        save_oauth_credential("openai-codex", access_token, refresh_token, expires_in)

        click.echo("\nOpenAI login successful!")
        mask = f"{access_token[:12]}...{access_token[-4:]}" if len(access_token) > 16 else f"{access_token[:8]}***"
        click.echo(f"  Token: {mask}")
        if refresh_token:
            click.echo("  Auto-refresh: enabled")
        click.echo("\nNadirClaw will use this token for openai-codex models.")
        click.echo("Verify with: nadirclaw auth status")
    else:
        click.echo("\nLogin completed but no token was captured.")
        click.echo("Check with: nadirclaw auth status")


@auth_openai.command(name="logout")
def openai_logout():
    """Remove stored OpenAI OAuth credential."""
    from nadirclaw.credentials import remove_credential

    if remove_credential("openai-codex"):
        click.echo("OpenAI credential removed.")
    else:
        click.echo("No OpenAI credential found.")


# ---------------------------------------------------------------------------
# nadirclaw auth anthropic — Anthropic subscription OAuth subgroup
# ---------------------------------------------------------------------------

@auth.group(name="anthropic")
def auth_anthropic():
    """Anthropic commands (setup token or API key)."""
    pass


@auth_anthropic.command(name="login")
def anthropic_login():
    """Add Anthropic credentials — choose between setup token or API key."""
    from nadirclaw.credentials import get_credential, get_credential_source, save_credential
    from nadirclaw.oauth import validate_anthropic_setup_token

    # First check if we already have a valid credential from any source
    existing_token = get_credential("anthropic")
    existing_source = get_credential_source("anthropic")
    if existing_token:
        click.echo(f"You already have Anthropic credentials (source: {existing_source}).")
        click.echo("  NadirClaw will use these automatically.")
        if not click.confirm("\nReplace existing credentials?", default=False):
            return

    # Ask user which auth method they want
    click.echo("\nHow would you like to authenticate with Anthropic?\n")
    click.echo("  1. Setup token  — use your Claude subscription (run `claude setup-token`)")
    click.echo("  2. API key      — use an Anthropic API key")
    click.echo()

    choice = click.prompt(
        "Choose",
        type=click.Choice(["1", "2"], case_sensitive=False),
        default="1",
    )

    if choice == "1":
        # Setup token flow
        click.echo("\n--- Setup Token ---")
        click.echo("1. Open another terminal and run:  claude setup-token")
        click.echo("2. Copy the generated token (starts with sk-ant-oat01-...)")
        click.echo("3. Paste it below\n")

        token = click.prompt("Paste Anthropic setup-token", hide_input=True)
        token = token.strip()

        error = validate_anthropic_setup_token(token)
        if error:
            click.echo(f"\nInvalid token: {error}")
            raise SystemExit(1)

        save_credential("anthropic", token, source="setup-token")

        click.echo("\nAnthropic login successful!")
        mask = f"{token[:16]}...{token[-4:]}" if len(token) > 20 else f"{token[:8]}***"
        click.echo(f"  Token: {mask}")
        click.echo("  Source: setup-token")

    else:
        # API key flow
        click.echo()
        key = click.prompt("Enter Anthropic API key", hide_input=True)
        key = key.strip()

        if not key:
            click.echo("Error: empty key provided.")
            raise SystemExit(1)

        save_credential("anthropic", key, source="manual")

        click.echo("\nAnthropic API key saved!")
        mask = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else f"{key[:4]}***"
        click.echo(f"  Key: {mask}")
        click.echo("  Source: api-key")

    click.echo("\nNadirClaw will use this for Anthropic/Claude models.")
    click.echo("Verify with: nadirclaw auth status")


@auth_anthropic.command(name="logout")
def anthropic_logout():
    """Remove stored Anthropic OAuth credential."""
    from nadirclaw.credentials import remove_credential

    if remove_credential("anthropic"):
        click.echo("Anthropic credential removed.")
    else:
        click.echo("No Anthropic credential found.")


# ---------------------------------------------------------------------------
# nadirclaw auth antigravity — Google Antigravity OAuth subgroup
# ---------------------------------------------------------------------------

@auth.group(name="antigravity")
def auth_antigravity():
    """Google Antigravity subscription commands (OAuth login with Google account)."""
    pass


@auth_antigravity.command(name="login")
@click.option("--timeout", "-t", default=300, help="Login timeout in seconds (default: 300)")
def antigravity_login(timeout):
    """Login via OAuth — use your Google account, no API key needed.

    Opens a browser for OAuth authorization. No external CLIs or env vars required.
    """
    import time as _time
    from nadirclaw.credentials import get_credential, get_credential_source, _read_credentials
    from nadirclaw.oauth import login_antigravity

    # First check if we already have a valid credential
    existing_token = get_credential("antigravity")
    existing_source = get_credential_source("antigravity")
    if existing_token:
        stored = _read_credentials().get("antigravity", {})
        expires_at = stored.get("expires_at", 0)
        if expires_at and _time.time() < (expires_at - 60):
            remaining = int(expires_at - _time.time())
            click.echo(f"You already have valid Antigravity credentials (source: {existing_source}).")
            click.echo(f"  Token expires in: {remaining // 60} minutes")
            click.echo("  NadirClaw will use these automatically.")
            click.echo("\nTo force re-login, run: nadirclaw auth antigravity logout && nadirclaw auth antigravity login")
            return

    click.echo("Logging in to Google Antigravity...")
    click.echo("A browser window will open for you to sign in with your Google account.\n")

    try:
        token_data = login_antigravity(timeout=timeout)
    except RuntimeError as e:
        click.echo(f"\nLogin failed: {e}")
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"\nUnexpected error during login: {e}")
        raise SystemExit(1)

    if not token_data:
        click.echo("\nLogin did not complete successfully.")
        raise SystemExit(1)

    access_token = token_data.get("access_token", "")
    refresh_token = token_data.get("refresh_token", "")
    expires_at = token_data.get("expires_at", 0)
    project_id = token_data.get("project_id", "")
    email = token_data.get("email", "")

    if access_token:
        from nadirclaw.credentials import save_oauth_credential
        expires_in = max(int(expires_at - _time.time()), 3600) if expires_at else 3600
        save_oauth_credential("antigravity", access_token, refresh_token, expires_in, metadata={
            "project_id": project_id,
            "email": email,
        })

        click.echo("\nAntigravity login successful!")
        mask = f"{access_token[:12]}...{access_token[-4:]}" if len(access_token) > 16 else f"{access_token[:8]}***"
        click.echo(f"  Token: {mask}")
        if refresh_token:
            click.echo("  Auto-refresh: enabled")
        if project_id:
            click.echo(f"  Project ID: {project_id}")
        if email:
            click.echo(f"  Email: {email}")
        click.echo("\nNadirClaw will use this token for Antigravity models.")
        click.echo("Verify with: nadirclaw auth status")
    else:
        click.echo("\nLogin completed but no token was captured.")
        click.echo("Check with: nadirclaw auth status")


@auth_antigravity.command(name="logout")
def antigravity_logout():
    """Remove stored Antigravity OAuth credential."""
    from nadirclaw.credentials import remove_credential

    if remove_credential("antigravity"):
        click.echo("Antigravity credential removed.")
    else:
        click.echo("No Antigravity credential found.")


# ---------------------------------------------------------------------------
# nadirclaw auth gemini-cli — Google Gemini CLI OAuth subgroup
# ---------------------------------------------------------------------------

@auth.group(name="gemini")
def auth_gemini():
    """Google Gemini subscription commands (OAuth login with Google account)."""
    pass


@auth_gemini.command(name="login")
@click.option("--timeout", "-t", default=300, help="Login timeout in seconds (default: 300)")
def gemini_login(timeout):
    """Login via OAuth — use your Google account, no API key needed.

    Opens a browser for OAuth authorization. Requires the Gemini CLI to be
    installed so NadirClaw can extract OAuth client credentials.
    """
    import time as _time
    from nadirclaw.credentials import get_credential, get_credential_source, _read_credentials
    from nadirclaw.oauth import login_gemini

    # First check if we already have a valid credential
    existing_token = get_credential("gemini")
    existing_source = get_credential_source("gemini")
    if existing_token:
        stored = _read_credentials().get("gemini", {})
        expires_at = stored.get("expires_at", 0)
        if expires_at and _time.time() < (expires_at - 60):
            remaining = int(expires_at - _time.time())
            click.echo(f"You already have valid Gemini credentials (source: {existing_source}).")
            click.echo(f"  Token expires in: {remaining // 60} minutes")
            click.echo("  NadirClaw will use these automatically.")
            click.echo("\nTo force re-login, run: nadirclaw auth gemini logout && nadirclaw auth gemini login")
            return

    click.echo("Logging in to Google Gemini...")
    click.echo("A browser window will open for you to sign in with your Google account.\n")

    try:
        token_data = login_gemini(timeout=timeout)
    except RuntimeError as e:
        click.echo(f"\nLogin failed: {e}")
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"\nUnexpected error during login: {e}")
        raise SystemExit(1)

    if not token_data:
        click.echo("\nLogin did not complete successfully.")
        raise SystemExit(1)

    access_token = token_data.get("access_token", "")
    refresh_token = token_data.get("refresh_token", "")
    expires_at = token_data.get("expires_at", 0)
    project_id = token_data.get("project_id", "")
    email = token_data.get("email", "")

    if access_token:
        from nadirclaw.credentials import save_oauth_credential
        expires_in = max(int(expires_at - _time.time()), 3600) if expires_at else 3600
        save_oauth_credential("gemini", access_token, refresh_token, expires_in, metadata={
            "project_id": project_id,
            "email": email,
        })

        click.echo("\nGemini login successful!")
        mask = f"{access_token[:12]}...{access_token[-4:]}" if len(access_token) > 16 else f"{access_token[:8]}***"
        click.echo(f"  Token: {mask}")
        if refresh_token:
            click.echo("  Auto-refresh: enabled")
        if project_id:
            click.echo(f"  Project ID: {project_id}")
        if email:
            click.echo(f"  Email: {email}")
        click.echo("\nNadirClaw will use this token for Gemini models.")
        click.echo("Verify with: nadirclaw auth status")
    else:
        click.echo("\nLogin completed but no token was captured.")
        click.echo("Check with: nadirclaw auth status")


@auth_gemini.command(name="logout")
def gemini_logout():
    """Remove stored Gemini OAuth credential."""
    from nadirclaw.credentials import remove_credential

    if remove_credential("gemini"):
        click.echo("Gemini credential removed.")
    else:
        click.echo("No Gemini credential found.")


@auth.command(name="add")
@click.option("--provider", "-p", default=None, help="Provider name (e.g. anthropic, openai)")
@click.option("--key", "-k", default=None, help="API key or token")
def auth_add(provider, key):
    """Add an API key for a provider."""
    from nadirclaw.credentials import save_credential

    if not provider:
        provider = click.prompt(
            "Provider",
            type=click.Choice(["anthropic", "openai", "google", "cohere", "mistral"], case_sensitive=False),
        )

    if not key:
        key = click.prompt(f"API key for {provider}", hide_input=True)

    if not key or not key.strip():
        click.echo("Error: empty key provided.")
        raise SystemExit(1)

    key = key.strip()
    save_credential(provider, key, source="manual")
    click.echo(f"\n{provider} credential saved.")
    click.echo("Verify with: nadirclaw auth status")


@auth.command(name="status")
def auth_status():
    """Show configured credentials (tokens are masked)."""
    from nadirclaw.credentials import list_credentials

    creds = list_credentials()
    if not creds:
        click.echo("No credentials configured.")
        click.echo("\nAdd credentials with:")
        click.echo("  nadirclaw auth openai login      # OpenAI subscription (OAuth)")
        click.echo("  nadirclaw auth anthropic login   # Anthropic subscription (OAuth)")
        click.echo("  nadirclaw auth antigravity login # Google Antigravity (OAuth)")
        click.echo("  nadirclaw auth gemini login      # Google Gemini (OAuth)")
        click.echo("  nadirclaw auth setup-token        # Claude subscription token")
        click.echo("  nadirclaw auth add                # Any provider API key")
        click.echo("  Or set env vars: ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.")
        return

    click.echo("Configured Credentials")
    click.echo("-" * 50)
    for c in creds:
        click.echo(f"  {c['provider']:12s}  {c['masked_token']}  ({c['source']})")
    click.echo(f"\n{len(creds)} provider(s) configured.")


@auth.command(name="remove")
@click.argument("provider")
def auth_remove(provider):
    """Remove a stored credential for PROVIDER."""
    from nadirclaw.credentials import remove_credential

    if remove_credential(provider):
        click.echo(f"Removed stored credential for {provider}.")
    else:
        click.echo(f"No stored credential found for {provider}.")
        click.echo("Note: this only removes credentials stored via 'nadirclaw auth'. "
                    "Env vars are not affected.")


@main.group()
def openclaw():
    """OpenClaw integration commands."""
    pass


@openclaw.command()
def onboard():
    """Auto-configure OpenClaw to use NadirClaw as a provider."""
    from nadirclaw.settings import settings

    openclaw_dir = Path.home() / ".openclaw"
    config_path = openclaw_dir / "openclaw.json"

    # Read existing config or start fresh
    existing = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                existing = json.load(f)
            # Create backup
            backup_path = config_path.with_suffix(
                f".backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
            )
            shutil.copy2(config_path, backup_path)
            click.echo(f"Backed up existing config to {backup_path}")
        except Exception as e:
            click.echo(f"Warning: could not read existing config: {e}")

    # Build the NadirClaw provider config
    nadirclaw_provider = {
        "baseUrl": f"http://localhost:{settings.PORT}/v1",
        "apiKey": "local",
        "api": "openai-completions",
        "models": [
            {
                "id": "auto",
                "reasoning": True,
                "input": ["text"],
                "contextWindow": 200000,
                "maxTokens": 64000,
            }
        ],
    }

    # Merge into existing config
    if "models" not in existing:
        existing["models"] = {}
    if "mode" not in existing["models"]:
        existing["models"]["mode"] = "merge"
    if "providers" not in existing["models"]:
        existing["models"]["providers"] = {}

    existing["models"]["providers"]["nadirclaw"] = nadirclaw_provider

    # Register nadirclaw/auto as a known model (don't override primary)
    if "agents" not in existing:
        existing["agents"] = {}
    if "defaults" not in existing["agents"]:
        existing["agents"]["defaults"] = {}
    if "models" not in existing["agents"]["defaults"]:
        existing["agents"]["defaults"]["models"] = {}
    existing["agents"]["defaults"]["models"]["nadirclaw/auto"] = {"alias": "nadir"}

    # Write config
    openclaw_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(existing, f, indent=2)

    click.echo(f"\nWrote OpenClaw config to {config_path}")

    # Add nadirclaw provider to each agent's models.json
    agents_dir = openclaw_dir / "agents"
    agent_count = 0
    if agents_dir.exists():
        for agent_dir in sorted(agents_dir.iterdir()):
            models_path = agent_dir / "agent" / "models.json"
            if not models_path.exists():
                continue
            try:
                with open(models_path) as f:
                    agent_models = json.load(f)
                providers = agent_models.get("providers", {})
                if "nadirclaw" in providers:
                    click.echo(f"  {agent_dir.name}: nadirclaw provider already exists, skipping")
                    continue
                providers["nadirclaw"] = nadirclaw_provider
                agent_models["providers"] = providers
                with open(models_path, "w") as f:
                    json.dump(agent_models, f, indent=2)
                agent_count += 1
                click.echo(f"  {agent_dir.name}: added nadirclaw provider")
            except Exception as e:
                click.echo(f"  {agent_dir.name}: error — {e}")

    click.echo(f"\nNadirClaw provider added to {agent_count} agent(s)")
    click.echo("Model 'nadirclaw/auto' registered (alias: nadir)")
    click.echo("\nNext steps:")
    click.echo("  1. Start NadirClaw:  nadirclaw serve")
    click.echo("  2. Restart gateway:  openclaw gateway restart")
    click.echo("  3. Set agent model:  /model nadirclaw/auto (in agent session)")


@main.group()
def codex():
    """OpenAI Codex integration commands."""
    pass


@codex.command()
def onboard():
    """Auto-configure Codex to use NadirClaw as a provider."""
    from nadirclaw.settings import settings

    codex_dir = Path.home() / ".codex"
    config_path = codex_dir / "config.toml"

    # Backup existing config if present
    if config_path.exists():
        backup_path = config_path.with_suffix(
            f".backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}.toml"
        )
        shutil.copy2(config_path, backup_path)
        click.echo(f"Backed up existing config to {backup_path}")

    config_content = f"""\
model_provider = "nadirclaw"

[model_providers.nadirclaw]
base_url = "http://localhost:{settings.PORT}/v1"
api_key = "local"
"""

    codex_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(config_content)

    click.echo(f"\nWrote Codex config to {config_path}")
    click.echo("\nNadirClaw configured as Codex model provider.")
    click.echo(f"  Base URL: http://localhost:{settings.PORT}/v1")
    click.echo("\nNext steps:")
    click.echo("  1. Start NadirClaw:  nadirclaw serve")
    click.echo("  2. Run Codex:        codex")


@main.group()
def openwebui():
    """Open WebUI integration commands."""
    pass


@openwebui.command()
def onboard():
    """Show setup instructions for Open WebUI integration."""
    from nadirclaw.settings import settings

    url = f"http://localhost:{settings.PORT}/v1"

    click.echo("\nOpen WebUI + NadirClaw Setup")
    click.echo("=" * 50)
    click.echo()
    click.echo("1. Start NadirClaw:")
    click.echo(f"   nadirclaw serve")
    click.echo()
    click.echo("2. In Open WebUI, go to:")
    click.echo("   Admin Settings → Connections → OpenAI → Add Connection")
    click.echo()
    click.echo("3. Enter:")
    click.echo(f"   URL:     {url}")
    click.echo(f"   API Key: local")
    click.echo()
    click.echo("4. Select the 'auto' model in your chat — NadirClaw routes")
    click.echo("   each prompt to the right model automatically.")
    click.echo()
    click.echo("Available models:")
    click.echo("   auto      Smart routing (default)")
    click.echo("   eco       Always use cheap model")
    click.echo("   premium   Always use best model")
    click.echo()
    click.echo(f"Verify: curl {url}/models")
    click.echo()


@main.group()
def continue_dev():
    """Continue (continue.dev) integration commands."""
    pass


@continue_dev.command()
def onboard():
    """Auto-configure Continue to use NadirClaw as a provider."""
    from nadirclaw.settings import settings

    config_dir = Path.home() / ".continue"
    config_path = config_dir / "config.json"

    # Backup existing config if present
    existing = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                existing = json.load(f)
            backup_path = config_path.with_suffix(
                f".backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
            )
            shutil.copy2(config_path, backup_path)
            click.echo(f"Backed up existing config to {backup_path}")
        except Exception as e:
            click.echo(f"Warning: could not read existing config: {e}")

    # Build the NadirClaw model entry
    nadirclaw_model = {
        "title": "NadirClaw Auto",
        "provider": "openai",
        "model": "auto",
        "apiBase": f"http://localhost:{settings.PORT}/v1",
        "apiKey": "local",
    }

    # Merge into existing config
    if "models" not in existing:
        existing["models"] = []

    # Remove any existing NadirClaw entries
    existing["models"] = [
        m for m in existing["models"]
        if not (m.get("apiBase", "").startswith(f"http://localhost:{settings.PORT}"))
    ]
    existing["models"].insert(0, nadirclaw_model)

    # Write config
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(existing, f, indent=2)

    click.echo(f"\nWrote Continue config to {config_path}")
    click.echo(f"\nNadirClaw added as the first model in Continue.")
    click.echo(f"  Model: auto (smart routing)")
    click.echo(f"  API Base: http://localhost:{settings.PORT}/v1")
    click.echo("\nNext steps:")
    click.echo("  1. Start NadirClaw:  nadirclaw serve")
    click.echo("  2. Open Continue in your editor (VS Code / JetBrains)")
    click.echo("  3. Select 'NadirClaw Auto' from the model dropdown")


# Rename the Click group to use "continue" as CLI name (Python keyword workaround)
continue_dev.name = "continue"


@main.group()
def cursor():
    """Cursor editor integration commands."""
    pass


@cursor.command()
def onboard():
    """Auto-configure Cursor to use NadirClaw as an OpenAI-compatible provider."""
    from nadirclaw.settings import settings

    cursor_dir = Path.home() / ".cursor"
    config_path = cursor_dir / "mcp.json"

    click.echo("\nCursor + NadirClaw Setup")
    click.echo("=" * 50)
    click.echo()
    click.echo("Cursor supports OpenAI-compatible providers natively.")
    click.echo("Follow these steps to connect NadirClaw:\n")
    click.echo("1. Start NadirClaw:")
    click.echo("   nadirclaw serve")
    click.echo()
    click.echo("2. In Cursor, go to:")
    click.echo("   Settings → Models → OpenAI API Key")
    click.echo()
    click.echo("3. Enter these values:")
    click.echo(f"   API Key:      local")
    click.echo(f"   Base URL:     http://localhost:{settings.PORT}/v1")
    click.echo(f"   Model name:   auto")
    click.echo()
    click.echo("4. Click 'Verify' to test the connection, then save.")
    click.echo()
    click.echo("Available models:")
    click.echo("   auto      Smart routing (recommended)")
    click.echo("   eco       Always use cheap model")
    click.echo("   premium   Always use best model")
    click.echo()
    click.echo(f"Verify: curl http://localhost:{settings.PORT}/v1/models")
    click.echo()


@main.group()
def ollama():
    """Ollama discovery and management commands."""
    pass


@ollama.command()
@click.option("--scan-network", is_flag=True, help="Scan local network (slower)")
def discover(scan_network):
    """Discover Ollama instances on localhost and local network."""
    from nadirclaw.ollama_discovery import discover_ollama_instances, format_discovery_results

    click.echo("Scanning for Ollama instances...")
    if scan_network:
        click.echo("(network scan enabled — this may take a few seconds)")

    instances = discover_ollama_instances(scan_network=scan_network)

    click.echo()
    click.echo(format_discovery_results(instances))

    if instances:
        click.echo()
        click.echo("To use an instance, update your ~/.nadirclaw/.env:")
        click.echo(f"  OLLAMA_API_BASE={instances[0]['url']}")


@main.command()
@click.option("--simple-model", default=None, help="Override simple model for this test")
@click.option("--complex-model", default=None, help="Override complex model for this test")
@click.option("--timeout", default=30, type=int, help="Request timeout in seconds (default: 30)")
def test(simple_model, complex_model, timeout):
    """Send a probe request to each configured model and report results.

    Verifies that your API keys and model names work before running the server.
    """
    import time as _time

    from nadirclaw.settings import settings

    s_model = simple_model or settings.SIMPLE_MODEL
    c_model = complex_model or settings.COMPLEX_MODEL

    probe = [{"role": "user", "content": "Reply with the single word: ok"}]

    models_to_test = [("simple", s_model)]
    if c_model != s_model:
        models_to_test.append(("complex", c_model))

    click.echo("NadirClaw Model Test")
    click.echo("=" * 50)

    any_failed = False
    for tier, model in models_to_test:
        click.echo(f"\n  [{tier}] {model}")
        click.echo(f"  {'─' * 46}")
        t0 = _time.time()
        try:
            import litellm

            resp = litellm.completion(
                model=model,
                messages=probe,
                max_tokens=10,
                timeout=timeout,
            )
            latency = int((_time.time() - t0) * 1000)
            content = resp.choices[0].message.content or ""
            click.echo(f"  Status:   OK")
            click.echo(f"  Latency:  {latency}ms")
            click.echo(f"  Reply:    {content.strip()!r}")
        except Exception as e:
            latency = int((_time.time() - t0) * 1000)
            click.echo(f"  Status:   FAILED ({latency}ms)")
            click.echo(f"  Error:    {e}")
            any_failed = True

    click.echo("")
    if any_failed:
        click.echo("One or more models failed. Check credentials with: nadirclaw auth status")
        raise SystemExit(1)
    else:
        click.echo("All models OK. Start the router with: nadirclaw serve")


@main.command()
@click.argument("request_id")
@click.option("--reason", type=click.Choice(["misrouted", "slow", "bad_quality", "good", "other"]),
              default="misrouted", help="Reason for flagging (default: misrouted)")
@click.option("--rating", type=click.IntRange(1, 5), default=None, help="Quality rating (1-5)")
@click.option("--tier", type=click.Choice(["simple", "mid", "complex"]), default=None,
              help="What tier should this have been routed to")
@click.option("--model", default=None, help="What model should have been used")
def flag(request_id, reason, rating, tier, model):
    """Flag a request as misrouted or provide quality feedback.

    Looks up the request in the local SQLite log and stores your feedback.

    Example: nadirclaw flag abc123 --reason misrouted --tier simple
    """
    from nadirclaw.feedback import request_exists, store_feedback

    if not request_exists(request_id):
        click.echo(f"Error: Request '{request_id}' not found in logs.")
        click.echo("Use 'nadirclaw report' to see recent request IDs.")
        raise SystemExit(1)

    result = store_feedback(
        request_id=request_id,
        rating=rating,
        reason=reason,
        correct_tier=tier,
        correct_model=model,
    )

    if "error" in result:
        click.echo(f"Error: {result['error']}")
        raise SystemExit(1)

    click.echo("Feedback recorded:")
    click.echo(f"  Request:  {request_id}")
    click.echo(f"  Reason:   {reason}")
    if rating is not None:
        click.echo(f"  Rating:   {rating}/5")
    if tier:
        click.echo(f"  Tier:     {tier}")
    if model:
        click.echo(f"  Model:    {model}")


@main.command(name="feedback-stats")
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json"]), help="Output format")
def feedback_stats(fmt):
    """Show feedback and quality scoring statistics."""
    from nadirclaw.feedback import get_feedback_stats

    stats = get_feedback_stats()

    if fmt == "json":
        click.echo(json.dumps(stats, indent=2))
        return

    click.echo("NadirClaw Feedback Stats")
    click.echo("=" * 45)

    total = stats["total_feedback"]
    click.echo(f"  Total feedback:    {total}")

    avg = stats["average_rating"]
    click.echo(f"  Average rating:    {avg if avg is not None else 'n/a'}")

    misroute = stats["misroute_rate"]
    if misroute is not None:
        click.echo(f"  Misroute rate:     {misroute:.1%}")
    else:
        click.echo("  Misroute rate:     n/a")

    # Reason breakdown
    reasons = stats.get("reason_counts", {})
    if reasons:
        click.echo("")
        click.echo("Reasons:")
        for reason, count in reasons.items():
            click.echo(f"  {reason:15s}  {count}")

    # Quality scores
    click.echo("")
    click.echo("Quality (last 7 days)")
    click.echo("-" * 45)
    avg_q = stats.get("avg_quality_7d")
    total_7d = stats.get("total_requests_7d", 0)
    click.echo(f"  Avg score:         {avg_q if avg_q is not None else 'n/a'}")
    click.echo(f"  Scored requests:   {total_7d}")

    if total == 0:
        click.echo("")
        click.echo("Tip: Use 'nadirclaw flag <request_id>' to submit feedback.")


if __name__ == "__main__":
    main()