"""NadirClaw CLI — serve, classify, onboard, and status commands."""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import click


@click.group()
@click.version_option(version=None, prog_name="nadirclaw", package_name="nadirclaw")
def main():
    """NadirClaw — Open-source LLM router."""
    pass


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
    uvicorn.run(
        "nadirclaw.server:app",
        host="0.0.0.0",
        port=actual_port,
        log_level=log_level,
    )


@main.command()
@click.argument("prompt", nargs=-1, required=True)
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json"]), help="Output format")
def classify(prompt, fmt):
    """Classify a prompt as simple or complex (no server needed)."""
    import logging

    logging.basicConfig(level=logging.WARNING)

    from nadirclaw.classifier import BinaryComplexityClassifier
    from nadirclaw.settings import settings

    prompt_text = " ".join(prompt)
    classifier = BinaryComplexityClassifier()
    is_complex, confidence = classifier.classify(prompt_text)

    tier = "complex" if is_complex else "simple"
    score = classifier._confidence_to_score(is_complex, confidence)

    # Pick model from explicit tier config
    model = settings.COMPLEX_MODEL if is_complex else settings.SIMPLE_MODEL

    if fmt == "json":
        click.echo(json.dumps({
            "tier": tier,
            "is_complex": is_complex,
            "confidence": round(confidence, 4),
            "score": round(score, 4),
            "model": model,
            "prompt": prompt_text,
        }))
    else:
        click.echo(f"Tier:       {tier}")
        click.echo(f"Confidence: {confidence:.4f}")
        click.echo(f"Score:      {score:.4f}")
        click.echo(f"Model:      {model}")


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


@main.command("update-models")
@click.option("--output", type=click.Path(path_type=Path), default=None,
              help="Metadata file to write (default: ~/.nadirclaw/models.json)")
@click.option("--source-url", default=None,
              help="Optional registry JSON URL to merge before writing")
@click.option("--dry-run", is_flag=True, help="Show what would be written without saving")
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json"]),
              help="Output format")
def update_models(output, source_url, dry_run, fmt):
    """Refresh local model metadata used by the router."""
    import urllib.error
    import urllib.request

    from nadirclaw.model_metadata import (
        default_metadata_path,
        local_metadata_path,
        parse_model_metadata,
        write_model_metadata,
    )
    from nadirclaw.routing import BUILTIN_MODEL_REGISTRY

    output_path = output or default_metadata_path()
    models = {
        model: dict(info)
        for model, info in sorted(BUILTIN_MODEL_REGISTRY.items())
    }
    env_source = os.getenv("NADIRCLAW_MODEL_REGISTRY_URL", "")
    source = source_url or env_source
    if source and not source.startswith(("http://", "https://")):
        raise click.ClickException(f"Source URL must use http(s): {source}")

    if source:
        max_bytes = 10 * 1024 * 1024  # 10 MiB cap on registry payload
        try:
            with urllib.request.urlopen(source, timeout=15) as resp:
                raw = resp.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raise click.ClickException(
                    f"Registry payload exceeds {max_bytes} bytes: {source}"
                )
            remote_payload = json.loads(raw)
            remote_models = parse_model_metadata(remote_payload)
        except (OSError, ValueError, urllib.error.URLError) as e:
            raise click.ClickException(str(e)) from e
        for model, info in remote_models.items():
            models[model] = {**models.get(model, {}), **info}

    if not dry_run:
        write_model_metadata(models, output_path, source=source or "builtin")

    result = {
        "path": str(output_path),
        "local_override_path": str(local_metadata_path()),
        "model_count": len(models),
        "dry_run": dry_run,
        "source": source or "builtin",
        "models": list(models.keys()),
    }

    if fmt == "json":
        click.echo(json.dumps(result, indent=2))
        return

    action = "Would write" if dry_run else "Updated"
    plural = "entry" if len(models) == 1 else "entries"
    click.echo(f"{action} {len(models)} model metadata {plural}")
    click.echo(f"Path: {output_path}")
    click.echo(f"Source: {source or 'builtin'}")
    click.echo(f"Local overrides: {local_metadata_path()}")
    if models:
        click.echo("Sample models:")
        for model in list(models.keys())[:5]:
            click.echo(f"  - {model}")


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


@main.command(name="build-centroids")
def build_centroids():
    """Regenerate centroid .npy files from prototype prompts."""
    import logging

    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from nadirclaw.encoder import get_shared_encoder_sync
    from nadirclaw.prototypes import COMPLEX_PROTOTYPES, SIMPLE_PROTOTYPES

    click.echo("Loading encoder...")
    encoder = get_shared_encoder_sync()

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


if __name__ == "__main__":
    main()
