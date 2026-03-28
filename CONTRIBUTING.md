# Contributing to NadirClaw

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/NadirRouter/NadirClaw.git
cd NadirClaw
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest                    # full suite
pytest tests/test_credentials.py  # single file
pytest -x                 # stop on first failure
pytest -v                 # verbose output
```

Tests use temp directories for credential storage and don't touch your real `~/.nadirclaw/` config.

## Code Style

- Python 3.10+ (use modern syntax: `dict` not `Dict`, `list` not `List`, `X | None` not `Optional[X]` in new code)
- No auto-formatter enforced — just keep it readable and consistent with surrounding code
- Use `logging.getLogger(__name__)` for module loggers
- Async where the framework requires it (FastAPI endpoints); sync is fine elsewhere

## Making Changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add or update tests if you changed behavior
4. Run `pytest` and make sure everything passes
5. Open a pull request

## What to Work On

- Bug fixes are always welcome
- Check the GitHub issues for open tasks
- If you want to add a new provider or feature, open an issue first to discuss the approach

## Project Structure

```
nadirclaw/
  __init__.py        # Package version (single source of truth)
  cli.py             # CLI commands
  server.py          # FastAPI server
  classifier.py      # Binary complexity classifier
  credentials.py     # Credential storage and resolution
  oauth.py           # OAuth login flows
  auth.py            # Request authentication
  settings.py        # Environment configuration
  encoder.py         # Sentence transformer singleton
  prototypes.py      # Seed prompts for centroids
tests/
  test_classifier.py
  test_credentials.py
  test_oauth.py
  test_server.py
```

## Credential & OAuth Changes

If you're modifying OAuth flows or credential storage:

- Never hardcode real API keys or user tokens in tests
- Use `monkeypatch` and `tmp_path` fixtures to isolate credential file operations
- The Antigravity OAuth client ID/secret are public "installed app" credentials (same pattern as gcloud CLI) — this is intentional
- Gemini CLI credential extraction via regex is known to be fragile; prefer env var fallbacks

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
