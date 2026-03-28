"""
Adaptive centroid retraining pipeline for NadirClaw.

Retrains routing centroids from production data + explicit feedback,
validates new centroids against held-out data, and manages versioned
centroid artifacts at ~/.nadirclaw/models/.
"""

import json
import logging
import random
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("nadirclaw")

MODELS_DIR = Path.home() / ".nadirclaw" / "models"
MAX_VERSIONS = 5
MIN_ACCURACY = 0.80
MAX_TIER_SHIFT = 0.20
VALIDATION_SPLIT = 0.2


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LabeledSample:
    """A prompt with a known tier label."""
    prompt: str
    tier: str  # "simple", "mid", or "complex"
    source: str = "prototype"  # prototype | feedback | misroute | file


@dataclass
class TrainingDataset:
    """Collection of labeled samples split into train/val."""
    train: List[LabeledSample] = field(default_factory=list)
    val: List[LabeledSample] = field(default_factory=list)
    sources: Dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val)

    def tier_distribution(self, subset: str = "all") -> Dict[str, float]:
        samples = self.train + self.val if subset == "all" else getattr(self, subset)
        if not samples:
            return {}
        counts: Dict[str, int] = {}
        for s in samples:
            counts[s.tier] = counts.get(s.tier, 0) + 1
        total = len(samples)
        return {t: c / total for t, c in sorted(counts.items())}


@dataclass
class ValidationResult:
    """Results from comparing new centroids against old."""
    accuracy: float
    tier_distribution: Dict[str, float]
    baseline_distribution: Dict[str, float]
    tier_shift: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Centroid metadata
# ---------------------------------------------------------------------------

def _metadata_path(version: int) -> Path:
    return MODELS_DIR / f"centroids_v{version}_meta.json"


def _centroids_path(version: int) -> Path:
    return MODELS_DIR / f"centroids_v{version}.npz"


def _read_metadata(version: int) -> Optional[Dict[str, Any]]:
    path = _metadata_path(version)
    if path.exists():
        return json.loads(path.read_text())
    return None


def _latest_version() -> int:
    """Scan ~/.nadirclaw/models/ for the highest centroid version number."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    versions = []
    for p in MODELS_DIR.glob("centroids_v*.npz"):
        try:
            v = int(p.stem.split("_v")[1])
            versions.append(v)
        except (IndexError, ValueError):
            continue
    return max(versions) if versions else 0


# ---------------------------------------------------------------------------
# CentroidTrainer
# ---------------------------------------------------------------------------

class CentroidTrainer:
    """Retrains routing centroids from production data + feedback."""

    def __init__(self):
        from nadirclaw.encoder import get_shared_encoder_sync
        self.encoder = get_shared_encoder_sync()

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def collect_training_data(
        self,
        db_path: Path,
        extra_file: Optional[Path] = None,
    ) -> TrainingDataset:
        """Collect labeled data from multiple sources.

        Sources (in priority order):
        1. Prototype prompts (always included as base dataset)
        2. Feedback corrections from SQLite (flagged with correct_tier)
        3. Misroute-detected entries (implicit negative feedback)
        4. Optional user-provided JSONL file
        """
        all_samples: List[LabeledSample] = []
        sources: Dict[str, int] = {}

        # 1. Prototype prompts
        from nadirclaw.prototypes import COMPLEX_PROTOTYPES, SIMPLE_PROTOTYPES
        try:
            from nadirclaw.prototypes import MEDIUM_PROTOTYPES
        except ImportError:
            MEDIUM_PROTOTYPES = []

        for p in SIMPLE_PROTOTYPES:
            all_samples.append(LabeledSample(prompt=p, tier="simple", source="prototype"))
        for p in MEDIUM_PROTOTYPES:
            all_samples.append(LabeledSample(prompt=p, tier="mid", source="prototype"))
        for p in COMPLEX_PROTOTYPES:
            all_samples.append(LabeledSample(prompt=p, tier="complex", source="prototype"))
        sources["prototype"] = len(SIMPLE_PROTOTYPES) + len(MEDIUM_PROTOTYPES) + len(COMPLEX_PROTOTYPES)

        # 2 + 3. Production data from SQLite
        if db_path.exists():
            feedback_samples, misroute_samples = self._collect_from_db(db_path)
            all_samples.extend(feedback_samples)
            all_samples.extend(misroute_samples)
            sources["feedback"] = len(feedback_samples)
            sources["misroute"] = len(misroute_samples)

        # 4. User-provided JSONL
        if extra_file and extra_file.exists():
            file_samples = self._load_jsonl(extra_file)
            all_samples.extend(file_samples)
            sources["file"] = len(file_samples)

        # Deduplicate by prompt text
        seen = set()
        deduped = []
        for s in all_samples:
            key = s.prompt.strip().lower()
            if key not in seen:
                seen.add(key)
                deduped.append(s)

        # Split into train/val
        random.shuffle(deduped)
        val_count = max(1, int(len(deduped) * VALIDATION_SPLIT))
        val = deduped[:val_count]
        train = deduped[val_count:]

        return TrainingDataset(train=train, val=val, sources=sources)

    def _collect_from_db(self, db_path: Path) -> Tuple[List[LabeledSample], List[LabeledSample]]:
        """Extract feedback corrections and misroute signals from SQLite."""
        feedback_samples: List[LabeledSample] = []
        misroute_samples: List[LabeledSample] = []

        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Check if feedback columns exist
            cursor.execute("PRAGMA table_info(requests)")
            columns = {row["name"] for row in cursor.fetchall()}

            # Feedback corrections: rows with correct_tier set
            if "correct_tier" in columns and "prompt" in columns:
                cursor.execute(
                    "SELECT prompt, correct_tier FROM requests "
                    "WHERE correct_tier IS NOT NULL AND prompt IS NOT NULL"
                )
                for row in cursor.fetchall():
                    tier = row["correct_tier"]
                    if tier in ("simple", "mid", "complex"):
                        feedback_samples.append(
                            LabeledSample(prompt=row["prompt"], tier=tier, source="feedback")
                        )

            # Misroute signals: rows flagged by MisrouteDetector
            if "misroute_type" in columns and "prompt" in columns and "tier" in columns:
                cursor.execute(
                    "SELECT prompt, tier, misroute_type FROM requests "
                    "WHERE misroute_type IS NOT NULL AND prompt IS NOT NULL"
                )
                for row in cursor.fetchall():
                    original_tier = row["tier"]
                    misroute_type = row["misroute_type"]
                    # Misrouted requests: the original tier was wrong, so invert
                    if original_tier == "simple" and misroute_type in ("retry", "override"):
                        misroute_samples.append(
                            LabeledSample(prompt=row["prompt"], tier="complex", source="misroute")
                        )
                    elif original_tier == "complex" and misroute_type == "override":
                        misroute_samples.append(
                            LabeledSample(prompt=row["prompt"], tier="simple", source="misroute")
                        )

            conn.close()
        except Exception as e:
            logger.warning("Could not read training data from SQLite: %s", e)

        return feedback_samples, misroute_samples

    @staticmethod
    def _load_jsonl(path: Path) -> List[LabeledSample]:
        """Load labeled samples from a JSONL file.

        Expected format: {"prompt": "...", "tier": "simple|mid|complex"}
        """
        samples = []
        for line_num, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                prompt = obj.get("prompt", "").strip()
                tier = obj.get("tier", "").strip().lower()
                if not prompt:
                    logger.warning("JSONL line %d: missing 'prompt', skipping", line_num)
                    continue
                if tier not in ("simple", "mid", "complex"):
                    logger.warning(
                        "JSONL line %d: invalid tier %r (expected simple/mid/complex), skipping",
                        line_num, tier,
                    )
                    continue
                samples.append(LabeledSample(prompt=prompt, tier=tier, source="file"))
            except json.JSONDecodeError as e:
                logger.warning("JSONL line %d: invalid JSON (%s), skipping", line_num, e)
        return samples

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_centroids(self, dataset: TrainingDataset) -> Dict[str, np.ndarray]:
        """Compute new centroids from labeled embeddings.

        Groups training samples by tier, encodes all prompts, computes the
        mean embedding per tier (normalized to unit length).

        Returns:
            {tier_name: centroid_vector} for each tier present in data.
        """
        tier_prompts: Dict[str, List[str]] = {}
        for sample in dataset.train:
            tier_prompts.setdefault(sample.tier, []).append(sample.prompt)

        centroids: Dict[str, np.ndarray] = {}
        for tier, prompts in tier_prompts.items():
            if not prompts:
                continue
            embeddings = self.encoder.encode(prompts, show_progress_bar=False)
            centroid = embeddings.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            centroids[tier] = centroid.astype(np.float32)
            logger.info(
                "Computed %s centroid from %d samples (dim=%d)",
                tier, len(prompts), centroid.shape[0],
            )

        return centroids

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        new_centroids: Dict[str, np.ndarray],
        dataset: TrainingDataset,
    ) -> ValidationResult:
        """Compare new centroids against held-out validation data.

        Classification rule: assign each sample to the tier whose centroid
        has the highest cosine similarity.

        Validation gates:
        - Accuracy must be >= 80%
        - Tier distribution shift must be <= 20%
        """
        if not dataset.val:
            return ValidationResult(
                accuracy=0.0,
                tier_distribution={},
                baseline_distribution=dataset.tier_distribution("train"),
                tier_shift=1.0,
                passed=False,
                details={"error": "No validation samples"},
            )

        # Encode validation prompts
        val_prompts = [s.prompt for s in dataset.val]
        val_embeddings = self.encoder.encode(val_prompts, show_progress_bar=False)

        # Normalize
        norms = np.linalg.norm(val_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        val_embeddings = val_embeddings / norms

        # Classify each validation sample
        tier_names = sorted(new_centroids.keys())
        centroid_matrix = np.stack([new_centroids[t] for t in tier_names])  # (n_tiers, dim)

        # Cosine similarity: (n_val, dim) @ (dim, n_tiers) -> (n_val, n_tiers)
        similarities = val_embeddings @ centroid_matrix.T
        predicted_indices = np.argmax(similarities, axis=1)
        predicted_tiers = [tier_names[i] for i in predicted_indices]

        # Compute accuracy
        correct = sum(
            1 for pred, sample in zip(predicted_tiers, dataset.val)
            if pred == sample.tier
        )
        accuracy = correct / len(dataset.val)

        # Compute tier distributions
        pred_dist: Dict[str, int] = {}
        for t in predicted_tiers:
            pred_dist[t] = pred_dist.get(t, 0) + 1
        total_val = len(dataset.val)
        new_distribution = {t: c / total_val for t, c in sorted(pred_dist.items())}

        baseline_distribution = dataset.tier_distribution("train")

        # Compute max tier shift
        all_tiers = set(list(new_distribution.keys()) + list(baseline_distribution.keys()))
        max_shift = 0.0
        for t in all_tiers:
            shift = abs(new_distribution.get(t, 0.0) - baseline_distribution.get(t, 0.0))
            max_shift = max(max_shift, shift)

        # Per-tier accuracy
        tier_correct: Dict[str, int] = {}
        tier_total: Dict[str, int] = {}
        for pred, sample in zip(predicted_tiers, dataset.val):
            tier_total[sample.tier] = tier_total.get(sample.tier, 0) + 1
            if pred == sample.tier:
                tier_correct[sample.tier] = tier_correct.get(sample.tier, 0) + 1
        per_tier_accuracy = {
            t: tier_correct.get(t, 0) / tier_total[t]
            for t in sorted(tier_total.keys())
        }

        passed = accuracy >= MIN_ACCURACY and max_shift <= MAX_TIER_SHIFT

        return ValidationResult(
            accuracy=accuracy,
            tier_distribution=new_distribution,
            baseline_distribution=baseline_distribution,
            tier_shift=max_shift,
            passed=passed,
            details={
                "per_tier_accuracy": per_tier_accuracy,
                "val_samples": len(dataset.val),
                "correct": correct,
            },
        )

    # ------------------------------------------------------------------
    # Deploy (versioned centroid storage)
    # ------------------------------------------------------------------

    def deploy(
        self,
        new_centroids: Dict[str, np.ndarray],
        validation: ValidationResult,
        dataset: TrainingDataset,
    ) -> int:
        """Save versioned centroids and metadata.

        Saves to ~/.nadirclaw/models/centroids_v{N}.npz with a JSON sidecar.
        Prunes old versions to keep at most MAX_VERSIONS.
        Also copies the new simple/complex centroids into the package directory
        so they are picked up by the classifier on next load.

        Returns the new version number.
        """
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        version = _latest_version() + 1
        npz_path = _centroids_path(version)
        meta_path = _metadata_path(version)

        # Save centroids as .npz
        np.savez(str(npz_path), **{t: v for t, v in new_centroids.items()})

        # Save metadata
        metadata = {
            "version": version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "samples_count": dataset.total,
            "train_samples": len(dataset.train),
            "val_samples": len(dataset.val),
            "accuracy": round(validation.accuracy, 4),
            "tier_distribution": {
                k: round(v, 4) for k, v in validation.tier_distribution.items()
            },
            "sources": dataset.sources,
            "tiers": sorted(new_centroids.keys()),
        }
        meta_path.write_text(json.dumps(metadata, indent=2))

        # Copy into package directory for immediate use
        self._install_centroids(new_centroids)

        # Prune old versions
        self._prune_old_versions(keep=MAX_VERSIONS)

        # Reset the classifier singleton so it reloads
        self._reset_classifier()

        logger.info("Deployed centroid version %d to %s", version, npz_path)
        return version

    @staticmethod
    def _install_centroids(centroids: Dict[str, np.ndarray]) -> None:
        """Copy centroids into the package directory."""
        import os
        pkg_dir = os.path.dirname(os.path.abspath(__file__))

        if "simple" in centroids:
            np.save(os.path.join(pkg_dir, "simple_centroid.npy"), centroids["simple"])
        if "complex" in centroids:
            np.save(os.path.join(pkg_dir, "complex_centroid.npy"), centroids["complex"])

    @staticmethod
    def _prune_old_versions(keep: int = MAX_VERSIONS) -> None:
        """Remove old centroid versions, keeping only the most recent N."""
        versions = []
        for p in MODELS_DIR.glob("centroids_v*.npz"):
            try:
                v = int(p.stem.split("_v")[1])
                versions.append(v)
            except (IndexError, ValueError):
                continue

        versions.sort(reverse=True)
        for v in versions[keep:]:
            npz = _centroids_path(v)
            meta = _metadata_path(v)
            if npz.exists():
                npz.unlink()
            if meta.exists():
                meta.unlink()
            logger.debug("Pruned centroid version %d", v)

    @staticmethod
    def _reset_classifier() -> None:
        """Reset the classifier singleton so it picks up new centroids."""
        try:
            from nadirclaw import classifier
            classifier._singleton = None
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self) -> Optional[int]:
        """Revert to the previous centroid version.

        Loads centroids from version N-1 and installs them into the
        package directory. Returns the rolled-back-to version, or None
        if no previous version exists.
        """
        current = _latest_version()
        if current <= 1:
            return None

        target = current - 1
        npz_path = _centroids_path(target)
        if not npz_path.exists():
            # Search for the highest version below current
            for v in range(current - 1, 0, -1):
                if _centroids_path(v).exists():
                    target = v
                    npz_path = _centroids_path(v)
                    break
            else:
                return None

        data = np.load(str(npz_path))
        centroids = {key: data[key] for key in data.files}
        self._install_centroids(centroids)
        self._reset_classifier()

        # Remove the current version
        current_npz = _centroids_path(current)
        current_meta = _metadata_path(current)
        if current_npz.exists():
            current_npz.unlink()
        if current_meta.exists():
            current_meta.unlink()

        logger.info("Rolled back from v%d to v%d", current, target)
        return target

    # ------------------------------------------------------------------
    # Load existing centroids for comparison
    # ------------------------------------------------------------------

    def load_current_centroids(self) -> Dict[str, np.ndarray]:
        """Load the currently installed centroids from the package directory."""
        import os
        pkg_dir = os.path.dirname(os.path.abspath(__file__))

        centroids = {}
        simple_path = os.path.join(pkg_dir, "simple_centroid.npy")
        complex_path = os.path.join(pkg_dir, "complex_centroid.npy")

        if os.path.exists(simple_path):
            centroids["simple"] = np.load(simple_path)
        if os.path.exists(complex_path):
            centroids["complex"] = np.load(complex_path)

        return centroids


# ---------------------------------------------------------------------------
# MisrouteDetector — ported from Horizen's routing_quality_tracker.py
# ---------------------------------------------------------------------------

OVERRIDE_WINDOW_SECONDS = 60
RETRY_WINDOW_SECONDS = 60


class MisrouteDetector:
    """Detects likely misroutes from implicit user signals.

    Two detection modes:
    1. **Retry detection**: same prompt sent again within 60 seconds
       (user retry = dissatisfaction with first response).
    2. **Override detection**: user explicitly specifies a model after
       an auto-routed request with the same prompt.

    Stores detected misroutes as implicit negative feedback in the
    requests table for use by the retraining pipeline.
    """

    def __init__(self):
        # In-memory ring buffer of recent auto-routed requests:
        # {prompt_hash: {model, tier, timestamp, request_id}}
        self._recent: Dict[str, Dict[str, Any]] = {}
        self._max_recent = 2000

    def check(
        self,
        prompt: str,
        selected_model: str,
        tier: str,
        strategy: str,
        request_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Check a completed request for misroute signals.

        Call this after every completion. Returns a misroute record
        if one is detected, or None otherwise.
        """
        now = time.time()
        prompt_key = self._hash_prompt(prompt)

        # Evict stale entries
        self._evict_stale(now)

        misroute = None

        if prompt_key in self._recent:
            prev = self._recent[prompt_key]
            delta = now - prev["timestamp"]

            if delta <= RETRY_WINDOW_SECONDS:
                if strategy in ("direct", "alias") and prev["strategy"] not in ("direct", "alias"):
                    # User overrode auto-routing with an explicit model
                    misroute = {
                        "type": "override",
                        "original_request_id": prev["request_id"],
                        "override_request_id": request_id,
                        "original_model": prev["model"],
                        "override_model": selected_model,
                        "original_tier": prev["tier"],
                        "delta_seconds": round(delta, 1),
                        "prompt_preview": prompt[:100],
                    }
                elif prev["model"] == selected_model and strategy not in ("direct", "alias"):
                    # Same prompt, same model, within window = retry
                    misroute = {
                        "type": "retry",
                        "original_request_id": prev["request_id"],
                        "retry_request_id": request_id,
                        "model": selected_model,
                        "tier": tier,
                        "delta_seconds": round(delta, 1),
                        "prompt_preview": prompt[:100],
                    }

        # Store current request
        self._recent[prompt_key] = {
            "model": selected_model,
            "tier": tier,
            "strategy": strategy,
            "timestamp": now,
            "request_id": request_id,
        }

        if misroute:
            logger.info(
                "Misroute detected (%s): %s -> %s (%.0fs)",
                misroute["type"],
                misroute.get("original_model", misroute.get("model", "?")),
                misroute.get("override_model", "same"),
                misroute["delta_seconds"],
            )

        return misroute

    def _evict_stale(self, now: float) -> None:
        """Remove entries older than the detection window."""
        max_age = max(OVERRIDE_WINDOW_SECONDS, RETRY_WINDOW_SECONDS) * 2
        stale_keys = [
            k for k, v in self._recent.items()
            if now - v["timestamp"] > max_age
        ]
        for k in stale_keys:
            del self._recent[k]

        # Hard cap on buffer size
        if len(self._recent) > self._max_recent:
            sorted_keys = sorted(self._recent, key=lambda k: self._recent[k]["timestamp"])
            for k in sorted_keys[: len(self._recent) - self._max_recent]:
                del self._recent[k]

    @staticmethod
    def _hash_prompt(prompt: str) -> str:
        """Create a stable hash for prompt deduplication."""
        normalized = prompt.strip().lower()[:500]
        return str(hash(normalized))


# Module-level singleton
_misroute_detector: Optional[MisrouteDetector] = None


def get_misroute_detector() -> MisrouteDetector:
    """Return the singleton MisrouteDetector instance."""
    global _misroute_detector
    if _misroute_detector is None:
        _misroute_detector = MisrouteDetector()
    return _misroute_detector


def record_misroute(db_path: Path, misroute: Dict[str, Any]) -> None:
    """Store a detected misroute in the SQLite database.

    Adds a row to the misroutes table (created if needed).
    """
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS misroutes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                original_request_id TEXT,
                override_request_id TEXT,
                original_model TEXT,
                override_model TEXT,
                original_tier TEXT,
                delta_seconds REAL,
                prompt_preview TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_misroutes_timestamp
            ON misroutes(timestamp)
        """)

        cursor.execute(
            """
            INSERT INTO misroutes (
                timestamp, type, original_request_id, override_request_id,
                original_model, override_model, original_tier,
                delta_seconds, prompt_preview
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                misroute.get("type", "unknown"),
                misroute.get("original_request_id"),
                misroute.get("override_request_id", misroute.get("retry_request_id")),
                misroute.get("original_model", misroute.get("model")),
                misroute.get("override_model"),
                misroute.get("original_tier", misroute.get("tier")),
                misroute.get("delta_seconds"),
                misroute.get("prompt_preview"),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug("Failed to record misroute: %s", e)


# ---------------------------------------------------------------------------
# Version listing helper (for CLI output)
# ---------------------------------------------------------------------------

def list_versions() -> List[Dict[str, Any]]:
    """List all available centroid versions with metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    versions = []
    for p in MODELS_DIR.glob("centroids_v*.npz"):
        try:
            v = int(p.stem.split("_v")[1])
        except (IndexError, ValueError):
            continue
        meta = _read_metadata(v)
        versions.append({
            "version": v,
            "path": str(p),
            "metadata": meta,
        })
    versions.sort(key=lambda x: x["version"], reverse=True)
    return versions
