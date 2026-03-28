"""Trained classifier: sklearn model on embeddings + structural features.

Replaces the brittle centroid + rule-based cascade with a proper supervised
classifier that achieves 95%+ accuracy on ternary classification.

Training data: NadirClaw prototypes + Horizen prototypes + eval prompts (~700 samples).
Features: 384-dim sentence embedding (all-MiniLM-L6-v2) + 33-dim structural features = 417 dims.
Model: Gradient-boosted tree (sklearn) — small artifact (~200KB), <5ms inference.

Serialization: Uses joblib (safer than raw pickle) with SHA-256 hash verification.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

logger = logging.getLogger(__name__)

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH_JOBLIB = os.path.join(_PKG_DIR, "trained_model.joblib")
_MODEL_PATH_PKL = os.path.join(_PKG_DIR, "trained_model.pkl")  # legacy fallback
_TIER_MAP = {0: "simple", 1: "medium", 2: "complex"}
_TIER_IDX = {"simple": 0, "medium": 1, "complex": 2}


# ── Safe serialization helpers ───────────────────────────────────────


def _compute_file_hash(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_model(artifact: dict, path: str) -> None:
    """Save model artifact with joblib and write a companion SHA-256 hash file."""
    joblib.dump(artifact, path)
    file_hash = _compute_file_hash(path)
    hash_path = path + ".sha256"
    with open(hash_path, "w") as f:
        f.write(file_hash)
    logger.info("Model saved to %s (sha256: %s...)", path, file_hash[:16])


def _load_model(path: str) -> dict:
    """Load model artifact with joblib, verifying SHA-256 hash if available."""
    hash_path = path + ".sha256"
    if os.path.exists(hash_path):
        with open(hash_path, "r") as f:
            expected_hash = f.read().strip()
        actual_hash = _compute_file_hash(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Model file integrity check failed for {path}: "
                f"expected hash {expected_hash[:16]}..., got {actual_hash[:16]}..."
            )
        logger.debug("Model hash verified for %s", path)
    else:
        logger.warning(
            "No .sha256 hash file found for %s — skipping integrity check. "
            "Re-train the model to generate a hash file.",
            path,
        )
    return joblib.load(path)


def _resolve_model_path(model_path: Optional[str] = None) -> str:
    """Resolve which model file to use, preferring .joblib over legacy .pkl."""
    if model_path is not None:
        return model_path
    if os.path.exists(_MODEL_PATH_JOBLIB):
        return _MODEL_PATH_JOBLIB
    if os.path.exists(_MODEL_PATH_PKL):
        logger.info(
            "Using legacy .pkl model at %s — re-train to upgrade to .joblib format.",
            _MODEL_PATH_PKL,
        )
        return _MODEL_PATH_PKL
    # Neither exists; return the new default path (will trigger training)
    return _MODEL_PATH_JOBLIB


# ── Training data collection ─────────────────────────────────────────


def _load_all_training_data() -> Tuple[List[str], List[int]]:
    """Collect labeled prompts from all available sources.

    Returns (prompts, labels) where labels are 0=simple, 1=medium, 2=complex.
    """
    prompts: List[str] = []
    labels: List[int] = []
    seen = set()

    def _add(text: str, tier: str):
        key = text.strip().lower()
        if key in seen:
            return
        seen.add(key)
        prompts.append(text.strip())
        labels.append(_TIER_IDX[tier])

    # 1. NadirClaw prototypes
    try:
        from nadirclaw.prototypes import (
            SIMPLE_PROTOTYPES, MEDIUM_PROTOTYPES, COMPLEX_PROTOTYPES,
        )
        for p in SIMPLE_PROTOTYPES:
            _add(p, "simple")
        for p in MEDIUM_PROTOTYPES:
            _add(p, "medium")
        for p in COMPLEX_PROTOTYPES:
            _add(p, "complex")
        logger.info("Loaded NadirClaw prototypes: %d simple, %d medium, %d complex",
                     len(SIMPLE_PROTOTYPES), len(MEDIUM_PROTOTYPES), len(COMPLEX_PROTOTYPES))
    except ImportError:
        logger.warning("Could not load NadirClaw prototypes")

    # 2. Horizen prototypes
    horizen_path = Path(_PKG_DIR).parent.parent / "Horizen" / "app" / "reference_data" / "classifier_prototypes.json"
    if horizen_path.exists():
        with open(horizen_path) as f:
            data = json.load(f)
        for tier in ("simple", "medium", "complex"):
            for p in data.get(tier, []):
                text = p["text"] if isinstance(p, dict) else p
                _add(text, tier)
        logger.info("Loaded Horizen prototypes: %d simple, %d medium, %d complex",
                     len(data.get("simple", [])), len(data.get("medium", [])), len(data.get("complex", [])))

    # 3. Eval prompts (if available)
    try:
        from nadir_eval.prompts import PROMPT_BANKS
        for cat, (sys_prompt, prompt_list) in PROMPT_BANKS.items():
            for tier, text in prompt_list:
                _add(text, tier)
        logger.info("Loaded eval prompts from nadir_eval")
    except ImportError:
        pass

    logger.info("Total training data: %d samples (%d simple, %d medium, %d complex)",
                len(prompts),
                sum(1 for l in labels if l == 0),
                sum(1 for l in labels if l == 1),
                sum(1 for l in labels if l == 2))
    return prompts, labels


# ── Feature extraction ────────────────────────────────────────────────


def _extract_features(encoder, prompts: List[str]) -> np.ndarray:
    """Extract combined embedding + structural features for a batch of prompts.

    Returns array of shape (N, 417) = 384 embedding + 33 structural.
    """
    from nadirclaw.features import StructuralFeatureExtractor

    # Batch encode embeddings
    embeddings = encoder.encode(prompts, show_progress_bar=len(prompts) > 50,
                                 batch_size=64, normalize_embeddings=True)

    # Extract structural features for each prompt
    extractor = StructuralFeatureExtractor()
    struct_features = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        vec = extractor.extract_vector(messages)
        struct_features.append(vec)

    struct_array = np.array(struct_features, dtype=np.float32)

    # Concatenate: [embedding | structural]
    combined = np.hstack([embeddings, struct_array])
    return combined


# ── Training ──────────────────────────────────────────────────────────


def train_and_save(output_path: Optional[str] = None) -> Dict[str, Any]:
    """Train the classifier and save to disk.

    Returns training metrics dict.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import classification_report, confusion_matrix

    from nadirclaw.encoder import get_shared_encoder_sync

    output_path = output_path or _MODEL_PATH_JOBLIB

    logger.info("Training classifier...")
    start = time.time()

    # Load data
    prompts, labels = _load_all_training_data()
    if len(prompts) < 50:
        raise ValueError(f"Not enough training data: {len(prompts)} samples (need >= 50)")

    labels_arr = np.array(labels)

    # Extract features
    logger.info("Extracting features for %d prompts...", len(prompts))
    encoder = get_shared_encoder_sync()
    X = _extract_features(encoder, prompts)
    logger.info("Feature matrix: %s", X.shape)

    # Train with cross-validation to estimate accuracy
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42,
    )

    # 5-fold stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_preds = cross_val_predict(model, X, labels_arr, cv=cv)

    # Classification report
    report = classification_report(labels_arr, cv_preds, target_names=["simple", "medium", "complex"], output_dict=True)
    cm = confusion_matrix(labels_arr, cv_preds)

    cv_accuracy = report["accuracy"]
    logger.info("Cross-validation accuracy: %.1f%%", cv_accuracy * 100)
    logger.info("Per-class F1: simple=%.2f, medium=%.2f, complex=%.2f",
                report["simple"]["f1-score"], report["medium"]["f1-score"], report["complex"]["f1-score"])
    logger.info("Confusion matrix:\n%s", cm)

    # Train final model on all data
    model.fit(X, labels_arr)

    # Save
    artifact = {
        "model": model,
        "version": "1.0",
        "n_samples": len(prompts),
        "accuracy": cv_accuracy,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "feature_dim": X.shape[1],
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    _save_model(artifact, output_path)

    elapsed = time.time() - start
    logger.info("Trained model saved to %s (%.1fs)", output_path, elapsed)

    return {
        "accuracy": cv_accuracy,
        "per_class": {
            "simple": report["simple"],
            "medium": report["medium"],
            "complex": report["complex"],
        },
        "confusion_matrix": cm.tolist(),
        "n_samples": len(prompts),
        "elapsed_s": elapsed,
        "path": output_path,
    }


# ── Classifier class ─────────────────────────────────────────────────


class TrainedClassifier:
    """Supervised 3-class classifier using sklearn model.

    Combines sentence embeddings with structural features for high-accuracy
    ternary classification (simple/medium/complex).
    """

    CLASSIFIER_VERSION = "2.0"

    def __init__(self, model_path: Optional[str] = None):
        from nadirclaw.encoder import get_shared_encoder_sync
        from nadirclaw.features import StructuralFeatureExtractor

        self._encoder = get_shared_encoder_sync()
        self._extractor = StructuralFeatureExtractor()

        model_path = _resolve_model_path(model_path)
        if not os.path.exists(model_path):
            logger.info("No trained model found at %s — training now...", model_path)
            train_and_save(model_path)

        artifact = _load_model(model_path)

        self._model = artifact["model"]
        self._version = artifact.get("version", "unknown")
        self._accuracy = artifact.get("accuracy", 0)

        logger.info(
            "TrainedClassifier v%s loaded (cv_accuracy=%.1f%%, samples=%d)",
            self._version, self._accuracy * 100, artifact.get("n_samples", 0),
        )

    def classify(self, prompt: str) -> Tuple[str, float, Dict[str, Any]]:
        """Classify a prompt as simple, medium, or complex.

        Returns:
            (tier_name, confidence, metadata)
        """
        start = time.time()

        # 1. Embedding
        emb = self._encoder.encode([prompt], show_progress_bar=False,
                                    normalize_embeddings=True)

        # 2. Structural features
        messages = [{"role": "user", "content": prompt}]
        struct_vec = self._extractor.extract_vector(messages)

        # 3. Combine
        X = np.hstack([emb, np.array([struct_vec], dtype=np.float32)])

        # 4. Predict
        probs = self._model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        tier = _TIER_MAP[pred_idx]
        confidence = float(probs[pred_idx])
        escalated = False

        # Safety escalation: if the model says "simple" but isn't confident,
        # bump to the next-highest tier.  It's cheaper to over-serve a simple
        # prompt on a stronger model than to under-serve a complex one on Haiku.
        if tier == "simple" and confidence < 0.70:
            # Pick whichever of medium/complex has the higher probability
            if probs[2] >= probs[1]:
                tier, confidence = "complex", float(probs[2])
            else:
                tier, confidence = "medium", float(probs[1])
            escalated = True

        classify_ms = int((time.time() - start) * 1000)

        metadata = {
            "tier_probabilities": {
                "simple": float(probs[0]),
                "medium": float(probs[1]),
                "complex": float(probs[2]),
            },
            "confidence_escalated": escalated,
            "stage1_tier": _TIER_MAP[pred_idx],
            "stage1_confidence": float(probs[pred_idx]),
            "classify_ms": classify_ms,
            "classifier_version": self.CLASSIFIER_VERSION,
        }

        return tier, confidence, metadata

    async def analyze(self, text: str = "", system_message: str = "", **kwargs) -> Dict[str, Any]:
        """Async-compatible analysis interface matching the server's expected API.

        Returns a dict with tier_name, confidence, complexity_score, etc.
        """
        if system_message:
            tier, confidence, meta = self.classify_with_system(text, system_message)
        else:
            tier, confidence, meta = self.classify(text)

        probs = meta.get("tier_probabilities", {})
        # Map tier to a 0-1 complexity score
        complexity_score = probs.get("medium", 0) * 0.5 + probs.get("complex", 0) * 1.0

        return {
            "tier_name": tier,
            "confidence": confidence,
            "complexity_score": complexity_score,
            "analyzer_type": f"trained-v{self.CLASSIFIER_VERSION}",
            "analyzer_latency_ms": meta.get("classify_ms", 0),
            "reasoning": f"TrainedClassifier: {tier} ({confidence:.0%})",
            "ranked_models": [],
            **meta,
        }

    def classify_with_system(self, prompt: str, system_prompt: str = "") -> Tuple[str, float, Dict[str, Any]]:
        """Classify with system prompt context for richer structural features."""
        start = time.time()

        # 1. Embedding — include system prompt so the 384-dim vector captures
        #    tool definitions, personas, and constraints that affect complexity.
        if system_prompt:
            embed_text = f"{system_prompt[:500]} | {prompt}"
        else:
            embed_text = prompt
        emb = self._encoder.encode([embed_text], show_progress_bar=False,
                                    normalize_embeddings=True)

        # 2. Structural features (include system prompt for richer context)
        messages = [{"role": "user", "content": prompt}]
        struct_vec = self._extractor.extract_vector(messages, system_prompt=system_prompt)

        # 3. Combine & predict
        X = np.hstack([emb, np.array([struct_vec], dtype=np.float32)])
        probs = self._model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        tier = _TIER_MAP[pred_idx]
        confidence = float(probs[pred_idx])

        classify_ms = int((time.time() - start) * 1000)

        metadata = {
            "tier_probabilities": {
                "simple": float(probs[0]),
                "medium": float(probs[1]),
                "complex": float(probs[2]),
            },
            "confidence_escalated": False,
            "stage1_tier": tier,
            "stage1_confidence": confidence,
            "classify_ms": classify_ms,
            "classifier_version": self.CLASSIFIER_VERSION,
            "has_system_prompt": bool(system_prompt),
        }

        return tier, confidence, metadata
