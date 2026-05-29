"""Wide-and-Deep asymmetric pre-generation classifier (NadirClaw, MIT).

This is the open-source counterpart to the Wide&Deep analyzer that
powers Nadir's production routing. The trained weights (~900 KB,
`wide_deep_asym_v3.pt`) are bundled in `nadirclaw/models/` so users
get the trained classifier out of the box — no Nadir Pro account, no
HuggingFace download, no training run required.

Architecture
------------
- **Deep tower**: BGE-base-en-v1.5 (768-d sentence embedding) →
  two-layer MLP (768→hidden→hidden/2, ReLU + dropout).
- **Wide tower**: 33-d structural feature vector (length buckets, code
  fences, math symbols, tool calls, question words, ...). Batch-normed.
- **Head**: concat(deep, wide) → 3-way logits over
  `{simple, medium, complex}`.

Two checkpoints are bundled:

- ``wide_deep_asym_v3.pt`` — trained with asymmetric expected-cost
  loss (λ=3 in v3). The "shipped default" for backward compatibility
  with the published MODEL_CARD numbers, BUT note: the asym head has
  a known bug where the `simple` logit was globally suppressed during
  training (the model rationally learned to never predict `simple`).
  In production we always pair it with a `cost_sensitive` decoder, or
  we use the symmetric variant below.
- ``wide_deep_sym_v3.pt`` — same architecture, same training data,
  plain cross-entropy loss. Per-class F1 ≈ {simple 0.78, medium 0.64,
  complex 0.57}. Use this when you need correct `simple`-class
  behaviour with argmax decoding.

Decoders
--------
- ``argmax`` — pick the most-probable class.
- ``cost_sensitive`` — choose the class minimising expected
  asymmetric cost ``E[cost|j] = Σ_i P(i) * C[i, j]`` where downgrades
  cost ``λ * (i-j)`` and upgrades cost ``(j-i)``. ``λ=3`` is balanced,
  ``λ=20`` maxes out the safety margin.

Inference latency: ~40 ms / prompt on CPU once the BGE encoder is
warm (BGE load is amortised, ~5 s cold start).

Free/Pro split
--------------
NadirClaw ships the trained checkpoint and the architecture above. The
Pro version (`getnadir.dev/backend/app/complexity/wide_deep_asym_analyzer.py`)
shares the same loader contract and adds: per-tenant model menus,
provider-aware ranking, decision logging back to Postgres for
closed-loop retraining, and provider health-aware tier swaps.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bundled checkpoint paths
# ---------------------------------------------------------------------------
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_PKG_DIR, "models")
_MODEL_PATH_ASYM = os.path.join(_MODELS_DIR, "wide_deep_asym_v3.pt")
_MODEL_PATH_SYM = os.path.join(_MODELS_DIR, "wide_deep_sym_v3.pt")

_TIER_MAP: Dict[int, str] = {0: "simple", 1: "medium", 2: "complex"}
_TIER_NUM: Dict[str, int] = {"simple": 1, "medium": 2, "complex": 3}

# Process-level singletons. The encoder load is the expensive part; the
# checkpoint state-dict is tiny (~900 KB) so we cache one analyzer per
# `(checkpoint_variant)` pair.
_classifier_instances: Dict[str, "WideDeepClassifier"] = {}
_encoder = None  # SentenceTransformer instance; created lazily
_extractor = None  # StructuralFeatureExtractor; created lazily


# ---------------------------------------------------------------------------
# Lazy resource loaders
# ---------------------------------------------------------------------------
def _get_encoder():
    """Lazy-load the BGE-base-en-v1.5 sentence encoder (CPU)."""
    global _encoder
    if _encoder is None:
        t0 = time.time()
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover — surfaced as a clear error
            raise ImportError(
                "sentence-transformers is required for the trained classifier. "
                "Install via `pip install sentence-transformers` (already a NadirClaw dep)."
            ) from e

        # CPU is plenty fast and avoids the MPS hang we saw on py3.14.
        _encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
        logger.info(
            "BGE-base-en-v1.5 encoder loaded in %dms",
            int((time.time() - t0) * 1000),
        )
    return _encoder


def _get_extractor():
    """Lazy-load the 33-d structural feature extractor (pure regex)."""
    global _extractor
    if _extractor is None:
        from nadirclaw.structural_features import StructuralFeatureExtractor

        _extractor = StructuralFeatureExtractor()
    return _extractor


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
def _build_model(cfg: Dict[str, Any]):
    """Construct the exact module architecture used at training time."""
    import torch
    import torch.nn as nn

    class WideAndDeep(nn.Module):
        def __init__(self, emb_dim, struct_dim, hidden, dropout, out_dim=3):
            super().__init__()
            self.deep = nn.Sequential(
                nn.Linear(emb_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.struct_bn = nn.BatchNorm1d(struct_dim)
            self.head = nn.Linear(hidden // 2 + struct_dim, out_dim)

        def forward(self, emb, struct):  # noqa: D401
            return self.head(
                torch.cat([self.deep(emb), self.struct_bn(struct)], dim=1)
            )

    return WideAndDeep(
        cfg["emb_dim"], cfg["struct_dim"], cfg["hidden"], cfg["dropout"]
    )


def _cost_matrix(lam: float, k: int = 3):
    """C[i, j] = λ * (i - j) for downgrades, (j - i) for upgrades, 0 on diag."""
    import numpy as np

    C = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            if j < i:
                C[i, j] = lam * (i - j)
            elif j > i:
                C[i, j] = j - i
    return C


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------
@dataclass
class ClassificationResult:
    """Single-prompt classification outcome.

    Attributes
    ----------
    tier
        ``"simple"``, ``"medium"``, or ``"complex"``.
    tier_num
        1, 2, or 3 (matches the legacy NadirClaw tier integers).
    confidence
        Softmax probability of the chosen tier.
    probabilities
        Full per-class softmax: ``{"simple": p0, "medium": p1, "complex": p2}``.
    argmax_tier
        The argmax tier (may differ from ``tier`` under cost-sensitive decoding).
    decision_rule
        ``"argmax"`` or ``"cost_sensitive"``.
    cost_lambda
        Downgrade penalty (only meaningful under cost-sensitive decoding).
    latency_ms
        End-to-end classifier latency (encode + struct + forward).
    classifier_version
        e.g. ``"wide_deep_asym_v3"``.
    """

    tier: str
    tier_num: int
    confidence: float
    probabilities: Dict[str, float]
    argmax_tier: str
    decision_rule: str
    cost_lambda: float
    latency_ms: int
    classifier_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "tier_num": self.tier_num,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
            "argmax_tier": self.argmax_tier,
            "decision_rule": self.decision_rule,
            "cost_lambda": self.cost_lambda,
            "latency_ms": self.latency_ms,
            "classifier_version": self.classifier_version,
        }


class WideDeepClassifier:
    """Wide-and-Deep ternary complexity classifier.

    Parameters
    ----------
    checkpoint_variant
        ``"asym"`` (default, matches MODEL_CARD numbers) or ``"symmetric"``
        (recovers correct simple-class behaviour under argmax decoding).
    decision_rule
        ``"argmax"`` or ``"cost_sensitive"``. Pair the ``asym`` checkpoint
        with ``cost_sensitive`` for production-style behaviour.
    cost_lambda
        Downgrade penalty multiplier for cost-sensitive decoding. 3 is
        balanced; 20 is max-safe.
    model_path
        Override the bundled checkpoint location. When ``None`` (default),
        the bundled file under ``nadirclaw/models/`` is used.
    """

    ANALYZER_VERSION = "wide_deep_asym_v3"

    def __init__(
        self,
        checkpoint_variant: str = "asym",
        decision_rule: str = "argmax",
        cost_lambda: float = 3.0,
        model_path: Optional[str] = None,
    ) -> None:
        import torch
        import numpy as np

        if checkpoint_variant not in ("asym", "symmetric"):
            raise ValueError(
                f"checkpoint_variant must be 'asym' or 'symmetric', got {checkpoint_variant!r}"
            )
        if decision_rule not in ("argmax", "cost_sensitive"):
            raise ValueError(
                f"decision_rule must be 'argmax' or 'cost_sensitive', got {decision_rule!r}"
            )
        self.checkpoint_variant = checkpoint_variant
        self.decision_rule = decision_rule
        self.cost_lambda = float(cost_lambda)
        self._cost = (
            _cost_matrix(self.cost_lambda) if decision_rule == "cost_sensitive" else None
        )

        if model_path is not None:
            path = model_path
        elif checkpoint_variant == "symmetric":
            path = _MODEL_PATH_SYM
        else:
            path = _MODEL_PATH_ASYM
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Wide&Deep checkpoint not found at {path} "
                f"(variant={checkpoint_variant!r}). Did the package install "
                f"strip the model files? Reinstall NadirClaw or set model_path "
                f"explicitly."
            )

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        self._cfg = cfg
        self._trained_lambda = float(ckpt.get("lambda", 3.0))

        self._model = _build_model(cfg)
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()

        self._struct_mean = np.asarray(ckpt["struct_scaler_mean"], dtype=np.float32)
        self._struct_scale = np.asarray(ckpt["struct_scaler_scale"], dtype=np.float32)
        self._encoder_name = ckpt.get("encoder", "BAAI/bge-base-en-v1.5")
        self.model_path = path

        logger.info(
            "WideDeepClassifier loaded (variant=%s, encoder=%s, struct_dim=%d, "
            "trained_lambda=%.1f, rule=%s, lambda=%.1f)",
            self.checkpoint_variant,
            self._encoder_name,
            cfg["struct_dim"],
            self._trained_lambda,
            self.decision_rule,
            self.cost_lambda,
        )

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------
    def _predict_proba(
        self, prompt: str, system_prompt: str = ""
    ) -> Tuple["np.ndarray", int]:  # type: ignore[name-defined]
        import numpy as np
        import torch

        encoder = _get_encoder()
        extractor = _get_extractor()

        t0 = time.time()
        embed_text = f"{system_prompt[:500]} | {prompt}" if system_prompt else prompt

        emb = encoder.encode(
            [embed_text],
            show_progress_bar=False,
            normalize_embeddings=True,
            device="cpu",
        )
        emb = np.asarray(emb, dtype=np.float32)

        messages = [{"role": "user", "content": prompt}]
        struct_vec = np.asarray(
            extractor.extract_vector(messages, system_prompt=system_prompt),
            dtype=np.float32,
        )
        if struct_vec.shape[0] != self._cfg["struct_dim"]:
            raise ValueError(
                f"Structural feature vector has dim {struct_vec.shape[0]}, "
                f"checkpoint was trained with {self._cfg['struct_dim']}. "
                f"This usually means the StructuralFeatureExtractor has "
                f"drifted from the trained-time version."
            )
        struct_s = (struct_vec - self._struct_mean) / self._struct_scale
        struct_s = struct_s.reshape(1, -1)

        with torch.no_grad():
            logits = self._model(
                torch.from_numpy(emb),
                torch.from_numpy(struct_s),
            )
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        latency_ms = int((time.time() - t0) * 1000)
        return probs, latency_ms

    def classify(
        self, prompt: str, system_prompt: str = ""
    ) -> ClassificationResult:
        """Classify a single prompt; returns a :class:`ClassificationResult`."""
        import numpy as np

        probs, latency_ms = self._predict_proba(prompt, system_prompt=system_prompt)

        if self.decision_rule == "cost_sensitive":
            # E[cost | j] = Σ_i P(i) * C[i, j] → argmin_j
            expected_cost = probs @ self._cost
            pred_idx = int(np.argmin(expected_cost))
        else:
            pred_idx = int(np.argmax(probs))

        tier = _TIER_MAP[pred_idx]
        return ClassificationResult(
            tier=tier,
            tier_num=_TIER_NUM[tier],
            confidence=float(probs[pred_idx]),
            probabilities={
                "simple": float(probs[0]),
                "medium": float(probs[1]),
                "complex": float(probs[2]),
            },
            argmax_tier=_TIER_MAP[int(np.argmax(probs))],
            decision_rule=self.decision_rule,
            cost_lambda=self.cost_lambda,
            latency_ms=latency_ms,
            classifier_version=self.ANALYZER_VERSION,
        )

    # Convenience: legacy 3-tuple shape (matches getnadir's analyzer)
    def classify_tuple(
        self, prompt: str, system_prompt: str = ""
    ) -> Tuple[str, float, Dict[str, Any]]:
        r = self.classify(prompt, system_prompt=system_prompt)
        return r.tier, r.confidence, r.to_dict()


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------
def get_wide_deep_classifier(
    checkpoint_variant: str = "asym",
    decision_rule: str = "argmax",
    cost_lambda: float = 3.0,
    model_path: Optional[str] = None,
) -> WideDeepClassifier:
    """Singleton accessor keyed on ``checkpoint_variant``.

    The variant decides which checkpoint we deserialise, so we cache one
    instance per variant. Switching ``decision_rule`` or ``cost_lambda``
    on a cached instance is cheap (it only affects the decoder).
    """
    key = checkpoint_variant if model_path is None else f"{checkpoint_variant}:{model_path}"
    inst = _classifier_instances.get(key)
    if inst is None:
        inst = WideDeepClassifier(
            checkpoint_variant=checkpoint_variant,
            decision_rule=decision_rule,
            cost_lambda=cost_lambda,
            model_path=model_path,
        )
        _classifier_instances[key] = inst
        return inst
    # Hot-swap decoder if the caller asked for a different rule.
    if inst.decision_rule != decision_rule or inst.cost_lambda != float(cost_lambda):
        inst.decision_rule = decision_rule
        inst.cost_lambda = float(cost_lambda)
        inst._cost = (
            _cost_matrix(cost_lambda) if decision_rule == "cost_sensitive" else None
        )
    return inst


def bundled_model_paths() -> Dict[str, str]:
    """Return the absolute paths of the bundled checkpoints.

    Useful for diagnostics and for tests that want to assert the package
    data ships correctly.
    """
    return {
        "asym": _MODEL_PATH_ASYM,
        "symmetric": _MODEL_PATH_SYM,
    }


__all__ = [
    "ClassificationResult",
    "WideDeepClassifier",
    "bundled_model_paths",
    "get_wide_deep_classifier",
]
