"""Adaptive retraining pipeline for the trained classifier.

Collects training data from production (labeled prompts, misroute feedback,
low quality scores), performs stratified train/test splitting, trains a new
model with validation gating, and manages versioned artifacts.

CLI: ``nadirclaw retrain [--dry-run] [--min-samples N] [--validation-gate 0.90]``
"""

import json
import logging
import os
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("nadirclaw.retrain")

_TIER_MAP = {"simple": 0, "medium": 1, "mid": 1, "complex": 2}
_TIER_NAMES = {0: "simple", 1: "medium", 2: "complex"}
MODELS_DIR = Path.home() / ".nadirclaw" / "models"
MAX_VERSIONS = 5


def _collect_from_feedback(db_path: Path) -> List[Tuple[str, str, str]]:
    """Collect labeled prompts from feedback table.

    Returns list of (prompt, tier, source).
    """
    if not db_path.exists():
        return []

    results = []
    conn = sqlite3.connect(str(db_path))
    try:
        # From labeled_prompts table (explicit human labels)
        try:
            rows = conn.execute(
                "SELECT prompt, correct_tier FROM labeled_prompts WHERE prompt IS NOT NULL"
            ).fetchall()
            for prompt, tier in rows:
                if tier in _TIER_MAP and prompt:
                    results.append((prompt, tier, "label"))
        except sqlite3.OperationalError:
            pass

        # From feedback table (misroute flags → use correct_tier)
        try:
            rows = conn.execute(
                "SELECT r.prompt, f.correct_tier FROM feedback f "
                "JOIN requests r ON f.request_id = r.request_id "
                "WHERE f.reason = 'misrouted' AND f.correct_tier IS NOT NULL "
                "AND r.prompt IS NOT NULL"
            ).fetchall()
            for prompt, tier in rows:
                if tier in _TIER_MAP and prompt:
                    results.append((prompt, tier, "misroute_feedback"))
        except sqlite3.OperationalError:
            pass

    finally:
        conn.close()

    return results


def _collect_from_prototypes() -> List[Tuple[str, str, str]]:
    """Collect training data from built-in prototype files."""
    results = []
    pkg_dir = os.path.dirname(os.path.abspath(__file__))

    for tier, filename in [
        ("simple", "simple_prototypes.json"),
        ("complex", "complex_prototypes.json"),
    ]:
        path = os.path.join(pkg_dir, filename)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    prompts = json.load(f)
                for p in prompts:
                    if isinstance(p, str):
                        results.append((p, tier, "prototype"))
                    elif isinstance(p, dict) and "prompt" in p:
                        results.append((p["prompt"], tier, "prototype"))
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)

    return results


def collect_training_data(
    db_path: Optional[Path] = None,
    include_prototypes: bool = True,
) -> Tuple[List[str], List[int], Dict[str, int]]:
    """Collect all training data from available sources.

    Returns (prompts, labels, source_counts) where labels are 0/1/2.
    """
    from nadirclaw.settings import settings

    if db_path is None:
        db_path = settings.LOG_DIR / "requests.db"

    all_data: List[Tuple[str, str, str]] = []

    # Built-in prototypes
    if include_prototypes:
        all_data.extend(_collect_from_prototypes())

    # Production feedback
    all_data.extend(_collect_from_feedback(db_path))

    # Deduplicate by prompt text (keep latest source)
    seen: Dict[str, Tuple[str, str]] = {}
    for prompt, tier, source in all_data:
        seen[prompt.strip()] = (tier, source)

    prompts = list(seen.keys())
    labels = [_TIER_MAP[seen[p][0]] for p in prompts]
    sources: Dict[str, int] = {}
    for p in prompts:
        src = seen[p][1]
        sources[src] = sources.get(src, 0) + 1

    return prompts, labels, sources


def stratified_split(
    prompts: List[str],
    labels: List[int],
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Stratified train/test split ensuring proportional tier representation."""
    rng = np.random.RandomState(random_seed)

    # Group by label
    by_label: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    for i, label in enumerate(labels):
        by_label[label].append(i)

    train_idx = []
    test_idx = []

    for label, indices in by_label.items():
        rng.shuffle(indices)
        n_test = max(1, int(len(indices) * test_size))
        test_idx.extend(indices[:n_test])
        train_idx.extend(indices[n_test:])

    train_prompts = [prompts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_prompts = [prompts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_prompts, train_labels, test_prompts, test_labels


def compute_confusion_matrix(
    true_labels: List[int], pred_labels: List[int], n_classes: int = 3
) -> Dict[str, Any]:
    """Compute confusion matrix and per-class metrics."""
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        matrix[t][p] += 1

    # Per-class precision/recall
    per_class = {}
    for i in range(n_classes):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(n_classes)) - tp
        fn = sum(matrix[i]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class[_TIER_NAMES[i]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "support": int(sum(matrix[i])),
        }

    accuracy = sum(matrix[i][i] for i in range(n_classes)) / max(sum(sum(row) for row in matrix), 1)

    return {
        "accuracy": round(accuracy, 4),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def retrain(
    dry_run: bool = False,
    min_samples: int = 50,
    validation_gate: float = 0.85,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the adaptive retraining pipeline.

    Steps:
      1. Collect training data from prototypes + feedback
      2. Stratified train/test split
      3. Train new model
      4. Validate against gate threshold
      5. If passing, save versioned artifact (unless dry_run)

    Returns a report dict with metrics and status.
    """
    from nadirclaw.trained_classifier import train_and_save

    t0 = time.time()

    # 1. Collect data
    prompts, labels, sources = collect_training_data()
    total = len(prompts)

    if total < min_samples:
        return {
            "status": "skipped",
            "reason": f"Not enough samples ({total} < {min_samples})",
            "sources": sources,
        }

    logger.info("Retrain: collected %d samples from %s", total, sources)

    # 2. Stratified split
    train_prompts, train_labels, test_prompts, test_labels = stratified_split(
        prompts, labels, test_size=0.2,
    )

    logger.info("Retrain: train=%d, test=%d", len(train_prompts), len(test_prompts))

    # 3. Train
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_path = output_path or os.path.join(pkg_dir, "trained_model_candidate.joblib")

    try:
        result = train_and_save(tmp_path)
    except Exception as e:
        return {
            "status": "failed",
            "reason": f"Training failed: {e}",
            "sources": sources,
        }

    # 4. Validate
    from nadirclaw.trained_classifier import TrainedClassifier

    try:
        candidate = TrainedClassifier(model_path=tmp_path)
    except Exception as e:
        return {
            "status": "failed",
            "reason": f"Model load failed: {e}",
            "sources": sources,
        }

    pred_labels = []
    for prompt in test_prompts:
        tier, _, _ = candidate.classify(prompt)
        pred_labels.append(_TIER_MAP.get(tier, 1))

    metrics = compute_confusion_matrix(test_labels, pred_labels)
    accuracy = metrics["accuracy"]

    elapsed = time.time() - t0

    report = {
        "status": "validated",
        "accuracy": accuracy,
        "validation_gate": validation_gate,
        "passed_gate": accuracy >= validation_gate,
        "metrics": metrics,
        "sources": sources,
        "total_samples": total,
        "train_samples": len(train_prompts),
        "test_samples": len(test_prompts),
        "elapsed_seconds": round(elapsed, 1),
    }

    if accuracy < validation_gate:
        report["status"] = "rejected"
        report["reason"] = f"Accuracy {accuracy:.2%} < gate {validation_gate:.2%}"
        logger.warning("Retrain: model rejected (accuracy=%.2f%% < gate=%.2f%%)", accuracy * 100, validation_gate * 100)
        # Clean up candidate
        if os.path.exists(tmp_path) and not output_path:
            os.unlink(tmp_path)
        return report

    if dry_run:
        report["status"] = "dry_run"
        logger.info("Retrain (dry-run): accuracy=%.2f%%, would deploy", accuracy * 100)
        if os.path.exists(tmp_path) and not output_path:
            os.unlink(tmp_path)
        return report

    # 5. Deploy — version the artifact and replace current model
    final_path = os.path.join(pkg_dir, "trained_model.joblib")

    # Backup current model
    if os.path.exists(final_path):
        version_dir = MODELS_DIR / "versions"
        version_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = version_dir / f"trained_model_{ts}.joblib"
        shutil.copy2(final_path, str(backup))
        logger.info("Retrain: backed up current model → %s", backup)

        # Prune old versions
        versions = sorted(version_dir.glob("trained_model_*.joblib"))
        while len(versions) > MAX_VERSIONS:
            oldest = versions.pop(0)
            oldest.unlink()
            hash_file = oldest.with_suffix(".joblib.sha256")
            if hash_file.exists():
                hash_file.unlink()

    # Move candidate to production
    if tmp_path != final_path:
        shutil.move(tmp_path, final_path)
        # Move hash file too
        tmp_hash = tmp_path + ".sha256"
        if os.path.exists(tmp_hash):
            shutil.move(tmp_hash, final_path + ".sha256")

    report["status"] = "deployed"
    report["model_path"] = final_path
    logger.info(
        "Retrain: deployed new model (accuracy=%.2f%%, samples=%d, elapsed=%.1fs)",
        accuracy * 100, total, elapsed,
    )

    return report
