"""YAML loader for N-tier profiles, with TTL+mtime hot-reload cache.

Pattern mirrors :mod:`nadirclaw.cascade_rules.engine` so a single mental
model covers both: edit the YAML on disk, wait up to 30 s, the new
profile takes effect without a restart. Same fail-closed semantics:
parse errors leave the previously-loaded profile in place.

Profile resolution:

1. ``NADIRCLAW_TIERS_PROFILE`` env var, if set:
   - absolute path ending in ``.yaml`` / ``.yml`` → load that file.
   - bare name (e.g. ``n3_legacy``) → load
     ``nadirclaw/tier_config/profiles/<name>.yaml``.
2. Otherwise, fall back to the bundled ``n2_default.yaml``.

The env var is read at every ``get_default_profile_path()`` call so
tests can monkey-patch it without restarting the process.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover — pyyaml is in [dev] / [cascade-rules]
    yaml = None  # type: ignore[assignment]

from .schema import TierProfile

logger = logging.getLogger(__name__)


# Directory where bundled profiles live.
PROFILES_DIR: Path = Path(__file__).parent / "profiles"

_PROFILE_TTL_SEC: float = 30.0

_ENV_VAR: str = "NADIRCLAW_TIERS_PROFILE"
_DEFAULT_PROFILE_NAME: str = "n2_default"

# (mtime, ts_loaded, profile)
_PROFILE_CACHE: Dict[str, Tuple[float, float, TierProfile]] = {}
_PROFILE_CACHE_LOCK = threading.Lock()


def get_default_profile_path() -> Path:
    """Resolve the profile path that should be loaded with no explicit arg.

    Consults ``NADIRCLAW_TIERS_PROFILE`` each call (no caching). When the
    env var is unset, returns the bundled ``n2_default.yaml``.
    """
    raw = os.getenv(_ENV_VAR, "").strip()
    if not raw:
        return PROFILES_DIR / f"{_DEFAULT_PROFILE_NAME}.yaml"
    return _resolve_profile_path(raw)


def _resolve_profile_path(name_or_path: str) -> Path:
    """Resolve a profile reference to an absolute Path.

    Accepted forms:
    - absolute path ending in .yaml/.yml
    - relative path containing a separator
    - bare name (with or without .yaml suffix) → PROFILES_DIR/<name>.yaml
    """
    p = Path(name_or_path)
    if p.is_absolute() and p.suffix in (".yaml", ".yml"):
        return p
    if p.suffix in (".yaml", ".yml"):
        # Could be a name like "n3_legacy.yaml" or a relative path.
        if os.sep in name_or_path or "/" in name_or_path:
            return p
        return PROFILES_DIR / p.name
    return PROFILES_DIR / f"{name_or_path}.yaml"


def load_profile(name_or_path: Optional[str] = None) -> TierProfile:
    """Load a tier profile, returning a cached object when possible.

    ``name_or_path`` defaults to whatever ``get_default_profile_path()``
    returns. The cache invalidates after :data:`_PROFILE_TTL_SEC`
    seconds OR when the source file's mtime changes, whichever comes
    first. Parse errors return the last-good profile when one exists,
    falling back to a synthesised default otherwise.
    """
    if name_or_path is None:
        path = get_default_profile_path()
    else:
        path = _resolve_profile_path(name_or_path)
    key = str(path.resolve()) if path.exists() else str(path)

    now = time.time()
    with _PROFILE_CACHE_LOCK:
        cached = _PROFILE_CACHE.get(key)
        if cached is not None:
            mtime_cached, ts_loaded, profile = cached
            if (now - ts_loaded) < _PROFILE_TTL_SEC:
                return profile
            try:
                mtime_now = path.stat().st_mtime
            except OSError:
                mtime_now = mtime_cached
            if mtime_now == mtime_cached:
                _PROFILE_CACHE[key] = (mtime_cached, now, profile)
                return profile

    profile = _parse_profile_file(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    with _PROFILE_CACHE_LOCK:
        # Keep last-good on parse error: only overwrite on success.
        if profile is not None:
            _PROFILE_CACHE[key] = (mtime, now, profile)
            return profile
        cached = _PROFILE_CACHE.get(key)
        if cached is not None:
            return cached[2]

    # No previous good profile and parse failed — synthesise the bundled
    # default so the caller never sees a None. This is the same
    # fail-closed posture as cascade_rules.
    fallback = _bundled_default()
    with _PROFILE_CACHE_LOCK:
        _PROFILE_CACHE[key] = (0.0, now, fallback)
    return fallback


def _parse_profile_file(path: Path) -> Optional[TierProfile]:
    if yaml is None:
        logger.warning(
            "PyYAML not installed; tier profile %s cannot load. "
            "`pip install pyyaml` to enable YAML profiles.",
            path,
        )
        return None
    if not path.exists():
        logger.warning("Tier profile not found at %s.", path)
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to read tier profile %s: %s", path, e)
        return None
    if not isinstance(raw, dict):
        logger.error("Tier profile %s did not parse to a dict.", path)
        return None
    try:
        profile = TierProfile(**raw)
    except Exception as e:  # noqa: BLE001
        logger.error("Tier profile %s failed validation: %s", path, e)
        return None
    profile.profile_name = path.stem
    logger.info(
        "Loaded tier profile %s (mode=%s, n_tiers=%d)",
        profile.profile_name,
        profile.mode,
        profile.num_tiers,
    )
    return profile


def _bundled_default() -> TierProfile:
    """Synthesise the bundled n2 default in code, for when YAML is absent.

    This is the safety net: even with no PyYAML, no disk, and no env
    var, ``load_profile()`` returns a usable two-tier profile that
    reproduces the cheap-then-strong cascade we ship.
    """
    from .schema import CascadeConfig, SelectorConfig, Tier

    return TierProfile(
        version=1,
        mode="tiered",
        selector=SelectorConfig(),
        cascade=CascadeConfig(),
        tiers=[
            Tier(
                name="cheap",
                score_min=0.0,
                model_pool=[
                    "gpt-4o-mini",
                    "qwen3-235b-a22b-2507",
                    "deepseek-v3.2",
                    "claude-3-haiku-20240307",
                ],
                max_output_tokens=2048,
            ),
            Tier(
                name="strong",
                score_min=0.65,
                model_pool=[
                    "gpt-5-mini",
                    "deepseek-reasoner",
                    "deepseek-v4-flash",
                    "grok-4-1-fast-reasoning",
                    "claude-sonnet-4",
                ],
                max_output_tokens=4096,
            ),
        ],
        profile_name="n2_default",
    )


def _clear_profile_cache() -> None:
    """Drop the profile cache. Test-only — production hot-reload uses TTL."""
    with _PROFILE_CACHE_LOCK:
        _PROFILE_CACHE.clear()


__all__ = [
    "PROFILES_DIR",
    "get_default_profile_path",
    "load_profile",
]
