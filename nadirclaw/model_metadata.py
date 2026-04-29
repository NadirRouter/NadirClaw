"""Local model metadata helpers.

Model metadata is stored separately from code so users can refresh or override
model context windows, pricing, and capabilities without editing routing.py.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable


CONFIG_DIR = Path.home() / ".nadirclaw"
MODEL_METADATA_FILE = "models.json"
LOCAL_MODEL_METADATA_FILE = "models.local.json"


def default_metadata_path() -> Path:
    """Return the generated model metadata path."""
    override = os.getenv("NADIRCLAW_MODEL_METADATA_FILE", "")
    if override:
        return Path(override).expanduser()
    return CONFIG_DIR / MODEL_METADATA_FILE


def local_metadata_path() -> Path:
    """Return the user-managed model metadata override path."""
    override = os.getenv("NADIRCLAW_LOCAL_MODEL_METADATA_FILE", "")
    if override:
        return Path(override).expanduser()
    return CONFIG_DIR / LOCAL_MODEL_METADATA_FILE


def metadata_paths() -> Iterable[Path]:
    """Return metadata files in merge order."""
    return (default_metadata_path(), local_metadata_path())


def _extract_models(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Support both {"models": {...}} and direct {model_id: info} formats."""
    models = payload.get("models", payload)
    if not isinstance(models, dict):
        raise ValueError("model metadata must be a JSON object or contain a 'models' object")
    return models


def parse_model_metadata(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Normalize model metadata from a decoded JSON object."""
    models = _extract_models(data)
    normalized: Dict[str, Dict[str, Any]] = {}
    for model_id, info in models.items():
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        if not isinstance(info, dict):
            raise ValueError(f"metadata for {model_id!r} must be a JSON object")
        normalized[model_id.strip()] = _validate_model_info(model_id.strip(), info)
    return normalized


def _validate_model_info(model_id: str, info: Dict[str, Any]) -> Dict[str, Any]:
    """Validate known metadata fields while preserving unknown fields."""
    normalized = dict(info)
    if "context_window" in normalized:
        value = normalized["context_window"]
        if type(value) is not int or value < 0:
            raise ValueError(f"{model_id}.context_window must be a non-negative integer")

    for key in ("cost_per_m_input", "cost_per_m_output"):
        if key not in normalized:
            continue
        value = normalized[key]
        if not _is_non_negative_number(value):
            raise ValueError(f"{model_id}.{key} must be a non-negative number")

    if "has_vision" in normalized and type(normalized["has_vision"]) is not bool:
        raise ValueError(f"{model_id}.has_vision must be a boolean")

    return normalized


def _is_non_negative_number(value: Any) -> bool:
    return type(value) in (int, float) and value >= 0


def load_model_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load model metadata from a JSON file."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("model metadata root must be a JSON object")
    return parse_model_metadata(data)


def write_model_metadata(
    models: Dict[str, Dict[str, Any]],
    path: Path,
    *,
    source: str = "builtin",
) -> None:
    """Write model metadata in the generated file format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "source": source,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
