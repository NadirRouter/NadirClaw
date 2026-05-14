"""Minimal env-based configuration for NadirClaw."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

_settings_logger = logging.getLogger(__name__)

# Load .env from ~/.nadirclaw/.env if it exists
_nadirclaw_dir = Path.home() / ".nadirclaw"
_env_file = _nadirclaw_dir / ".env"
if _env_file.exists():
    load_dotenv(_env_file)
else:
    # Fallback to current directory .env
    load_dotenv()


class Settings:
    """All configuration from environment variables."""

    @property
    def AUTH_TOKEN(self) -> str:
        return os.getenv("NADIRCLAW_AUTH_TOKEN", "")

    @property
    def SIMPLE_MODEL(self) -> str:
        """Model for simple prompts. Falls back to last model in MODELS list."""
        explicit = os.getenv("NADIRCLAW_SIMPLE_MODEL", "")
        if explicit:
            return explicit
        models = self.MODELS
        return models[-1] if models else "gemini-3-flash-preview"

    @property
    def COMPLEX_MODEL(self) -> str:
        """Model for complex prompts. Falls back to first model in MODELS list."""
        explicit = os.getenv("NADIRCLAW_COMPLEX_MODEL", "")
        if explicit:
            return explicit
        models = self.MODELS
        return models[0] if models else "openai-codex/gpt-5.3-codex"

    @property
    def MODELS(self) -> list[str]:
        raw = os.getenv(
            "NADIRCLAW_MODELS",
            "openai-codex/gpt-5.3-codex,gemini-3-flash-preview",
        )
        return [m.strip() for m in raw.split(",") if m.strip()]

    @property
    def ANTHROPIC_API_KEY(self) -> str:
        return os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def OPENAI_API_KEY(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def GEMINI_API_KEY(self) -> str:
        return os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")

    @property
    def OLLAMA_API_BASE(self) -> str:
        return os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

    @property
    def API_BASE(self) -> str:
        """Custom base URL for OpenAI-compatible endpoints (vLLM, LocalAI, etc.).

        When set, passed as api_base to all non-Ollama, non-Gemini LiteLLM calls.
        """
        return os.getenv("NADIRCLAW_API_BASE", "")

    @property
    def CONFIDENCE_THRESHOLD(self) -> float:
        return float(os.getenv("NADIRCLAW_CONFIDENCE_THRESHOLD", "0.06"))

    @property
    def MID_MODEL(self) -> str:
        """Model for mid-complexity prompts. Falls back to SIMPLE_MODEL."""
        return os.getenv("NADIRCLAW_MID_MODEL", "") or self.SIMPLE_MODEL

    @property
    def TIER_THRESHOLDS(self) -> tuple[float, float]:
        """Score thresholds for 3-tier routing: (simple_max, complex_min).

        Prompts with score <= simple_max → simple tier.
        Prompts with score >= complex_min → complex tier.
        Prompts in between → mid tier.

        Set NADIRCLAW_TIER_THRESHOLDS=0.35,0.65 to customize.
        Default: (0.35, 0.65).
        """
        raw = os.getenv("NADIRCLAW_TIER_THRESHOLDS", "")
        if raw:
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) == 2:
                try:
                    return (float(parts[0]), float(parts[1]))
                except ValueError:
                    _settings_logger.warning(
                        "Invalid NADIRCLAW_TIER_THRESHOLDS=%r — expected two floats "
                        "(e.g. '0.35,0.65'). Falling back to defaults.",
                        raw,
                    )
            else:
                _settings_logger.warning(
                    "Invalid NADIRCLAW_TIER_THRESHOLDS=%r — expected two comma-separated "
                    "values. Falling back to defaults.",
                    raw,
                )
        return (0.35, 0.65)

    @property
    def has_mid_tier(self) -> bool:
        """True if MID_MODEL is explicitly set via env."""
        return bool(os.getenv("NADIRCLAW_MID_MODEL"))

    @property
    def PORT(self) -> int:
        return int(os.getenv("NADIRCLAW_PORT", "8856"))

    @property
    def LOG_RAW(self) -> bool:
        """When True, log full raw request messages and response content."""
        return os.getenv("NADIRCLAW_LOG_RAW", "").lower() in ("1", "true", "yes")

    @property
    def LOG_DIR(self) -> Path:
        return Path(os.getenv("NADIRCLAW_LOG_DIR", "~/.nadirclaw/logs")).expanduser()

    @property
    def LOG_MAX_SIZE_MB(self) -> int:
        """Max size of requests.jsonl before rotation (MB)."""
        return int(os.getenv("NADIRCLAW_LOG_MAX_SIZE_MB", "50"))

    @property
    def LOG_RETENTION_DAYS(self) -> int:
        """Days to keep old log archives and SQLite rows."""
        return int(os.getenv("NADIRCLAW_LOG_RETENTION_DAYS", "30"))

    @property
    def LOG_COMPRESS(self) -> bool:
        """Gzip rotated JSONL files."""
        val = os.getenv("NADIRCLAW_LOG_COMPRESS", "true").lower()
        return val in ("1", "true", "yes")

    @property
    def CREDENTIALS_FILE(self) -> Path:
        return Path.home() / ".nadirclaw" / "credentials.json"

    @property
    def REASONING_MODEL(self) -> str:
        """Model for reasoning tasks. Falls back to COMPLEX_MODEL."""
        return os.getenv("NADIRCLAW_REASONING_MODEL", "") or self.COMPLEX_MODEL

    @property
    def FREE_MODEL(self) -> str:
        """Free fallback model. Falls back to SIMPLE_MODEL."""
        return os.getenv("NADIRCLAW_FREE_MODEL", "") or self.SIMPLE_MODEL

    @property
    def COMPLEXITY_ANALYZER_TYPE(self) -> str:
        """Which classifier to use for smart routing.

          - "binary"     (default) — fast 2-class centroid classifier (~10ms, 22MB)
          - "distilbert"           — 3-class fine-tuned DistilBERT (~30ms, 256MB)
                                     produces simple/mid/complex tiers natively.

        Set via NADIRCLAW_COMPLEXITY_ANALYZER. The DistilBERT artifact is not
        shipped in the package — install separately or train via
        `nadir.distilbert_classifier.DistilBertClassifier()`.
        """
        return os.getenv("NADIRCLAW_COMPLEXITY_ANALYZER", "binary").strip().lower()

    @property
    def FALLBACK_CHAIN(self) -> list[str]:
        """Ordered fallback chain. When a model fails, try the next one.

        Defaults to [COMPLEX_MODEL, SIMPLE_MODEL] (existing behavior).
        Set NADIRCLAW_FALLBACK_CHAIN to customize, e.g.:
          NADIRCLAW_FALLBACK_CHAIN=gpt-4.1,claude-sonnet-4-5-20250929,gemini-2.5-flash
        """
        raw = os.getenv("NADIRCLAW_FALLBACK_CHAIN", "")
        if raw:
            return [m.strip() for m in raw.split(",") if m.strip()]
        # Default: deduplicated list of all configured tier models
        chain = []
        for m in [self.COMPLEX_MODEL, self.MID_MODEL, self.SIMPLE_MODEL, self.REASONING_MODEL, self.FREE_MODEL]:
            if m and m not in chain:
                chain.append(m)
        return chain

    def get_tier_fallback_chain(self, tier: str) -> list[str]:
        """Get the fallback chain for a specific tier.

        Per-tier chains are configured via env vars:
          NADIRCLAW_SIMPLE_FALLBACK=gemini-2.5-flash,gemini-3-flash-preview
          NADIRCLAW_MID_FALLBACK=gpt-4.1-mini,gemini-2.5-flash
          NADIRCLAW_COMPLEX_FALLBACK=claude-sonnet-4-5-20250929,gpt-4.1

        When a per-tier chain is set, it is used instead of the global chain.
        If no per-tier chain is configured, falls back to the global FALLBACK_CHAIN.
        """
        env_key = f"NADIRCLAW_{tier.upper()}_FALLBACK"
        raw = os.getenv(env_key, "")
        if raw:
            return [m.strip() for m in raw.split(",") if m.strip()]
        return self.FALLBACK_CHAIN

    @property
    def MODEL_RATE_LIMITS(self) -> str:
        """Per-model rate limits. Format: model=rpm,model2=rpm2."""
        return os.getenv("NADIRCLAW_MODEL_RATE_LIMITS", "")

    @property
    def DEFAULT_MODEL_RPM(self) -> int:
        """Default max requests/minute per model. 0 = unlimited."""
        try:
            return max(0, int(os.getenv("NADIRCLAW_DEFAULT_MODEL_RPM", "0")))
        except ValueError:
            return 0

    @property
    def PROVIDER_HEALTH(self) -> bool:
        """Enable health-aware fallback routing."""
        return os.getenv("NADIRCLAW_PROVIDER_HEALTH", "").lower() in ("1", "true", "yes")

    @property
    def PROVIDER_HEALTH_COOLDOWN_SECONDS(self) -> int:
        """Seconds to skip unhealthy fallback candidates before re-admitting them."""
        try:
            return max(1, int(os.getenv("NADIRCLAW_PROVIDER_HEALTH_COOLDOWN_SECONDS", "60")))
        except ValueError:
            return 60

    @property
    def PROVIDER_HEALTH_FAILURE_THRESHOLD(self) -> int:
        """Consecutive health failures before a fallback candidate enters cooldown."""
        try:
            return max(1, int(os.getenv("NADIRCLAW_PROVIDER_HEALTH_FAILURE_THRESHOLD", "2")))
        except ValueError:
            return 2

    @property
    def OPTIMIZE(self) -> str:
        """Context optimization mode: off, safe, aggressive. Default: off."""
        val = os.getenv("NADIRCLAW_OPTIMIZE", "off").lower()
        if val not in ("off", "safe", "aggressive"):
            _settings_logger.warning(
                "Invalid NADIRCLAW_OPTIMIZE=%r — expected off|safe|aggressive. "
                "Falling back to 'off'.",
                val,
            )
            return "off"
        return val

    @property
    def OPTIMIZE_MAX_TURNS(self) -> int:
        """Max conversation turns to keep when trimming. Default: 40."""
        try:
            return max(4, int(os.getenv("NADIRCLAW_OPTIMIZE_MAX_TURNS", "40")))
        except ValueError:
            return 40

    @property
    def has_explicit_tiers(self) -> bool:
        """True if SIMPLE_MODEL and COMPLEX_MODEL are explicitly set via env."""
        return bool(
            os.getenv("NADIRCLAW_SIMPLE_MODEL") and os.getenv("NADIRCLAW_COMPLEX_MODEL")
        )

    @property
    def tier_models(self) -> list[str]:
        """Deduplicated list of tier models: [COMPLEX, MID, SIMPLE]."""
        models = [self.COMPLEX_MODEL]
        if self.has_mid_tier and self.MID_MODEL not in models:
            models.append(self.MID_MODEL)
        if self.SIMPLE_MODEL not in models:
            models.append(self.SIMPLE_MODEL)
        return models

    @property
    def CONTEXT_COMPRESSION(self) -> bool:
        """Enable context compression for long conversations."""
        return os.getenv("NADIRCLAW_CONTEXT_COMPRESSION", "false").lower() in ("true", "1", "yes")

    @property
    def COMPRESS_MIN_MESSAGES(self) -> int:
        """Minimum message count before compression kicks in."""
        return int(os.getenv("NADIRCLAW_COMPRESS_MIN_MESSAGES", "30"))

    @property
    def COMPRESS_RECENT_WINDOW(self) -> int:
        """Number of recent messages to preserve intact."""
        return int(os.getenv("NADIRCLAW_COMPRESS_RECENT_WINDOW", "20"))

    @property
    def COMPRESS_TOOL_OUTPUT_MAX(self) -> int:
        """Max characters for truncated tool output."""
        return int(os.getenv("NADIRCLAW_COMPRESS_TOOL_MAX", "500"))

    @property
    def AGENT_ROLE_DETECTION(self) -> bool:
        """Enable agent role detection for coding agents (opt-in)."""
        return os.getenv("NADIRCLAW_AGENT_ROLE_DETECTION", "false").lower() in ("true", "1", "yes")


settings = Settings()
