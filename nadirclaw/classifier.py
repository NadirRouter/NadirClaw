"""
Complexity classifiers for NadirClaw.

Two classifier modes:

* **binary** (default, free tier) -- fast centroid-based simple/complex
  classification using pre-computed .npy centroid files (~10 ms).

* **cascade** (paid tier) -- confidence-aware ternary classifier that:
  1. Runs a fast 3-centroid classifier (simple/medium/complex) with
     k-means sub-clustering for the complex tier (~10 ms).
  2. If confidence < threshold, escalates to a structural feature
     analyzer that extracts 20+ textual signals for a second opinion.
  3. Returns calibrated confidence and ternary tier.

Activate cascade via ``NADIRCLAW_CLASSIFIER=cascade``.
"""

import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PKG_DIR = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# Confidence calibration (ported from Horizen analyzer_factory.py)
# ---------------------------------------------------------------------------

_CALIBRATION_PARAMS: Dict[str, Tuple[float, float]] = {
    # (scale, offset) -> calibrated = raw * scale + offset, clamped to [0, 1]
    "binary":           (1.0, 0.0),     # pass-through (legacy)
    "cascade_centroid": (1.0, 0.0),     # already calibrated via temperature scaling
    "cascade_struct":   (1.2, -0.05),   # slightly stretch structural analyzer
}


def calibrate_confidence(raw: float, analyzer_type: str) -> float:
    """Apply per-analyzer linear rescaling to a consistent [0, 1] scale."""
    scale, offset = _CALIBRATION_PARAMS.get(analyzer_type, (1.0, 0.0))
    return max(0.0, min(1.0, raw * scale + offset))


# ===================================================================
# BinaryComplexityClassifier (existing, backward-compatible)
# ===================================================================

class BinaryComplexityClassifier:
    """
    Classifies prompts as simple or complex using semantic prototype centroids.

    Loads pre-computed centroid vectors from .npy files (shipped with the
    package). At inference time, embeds the prompt (~10 ms on warm encoder),
    computes cosine similarity to both centroids, and returns a binary
    decision with a confidence score.
    """

    def __init__(self):
        from nadirclaw.encoder import get_shared_encoder_sync

        self.encoder = get_shared_encoder_sync()
        self._simple_centroid, self._complex_centroid = self._load_centroids()

        logger.info("BinaryComplexityClassifier ready (pre-computed centroids)")

    # ------------------------------------------------------------------
    # Load pre-computed centroids
    # ------------------------------------------------------------------

    @staticmethod
    def _load_centroids() -> Tuple[np.ndarray, np.ndarray]:
        """Load pre-computed centroid vectors from .npy files."""
        simple_path = os.path.join(_PKG_DIR, "simple_centroid.npy")
        complex_path = os.path.join(_PKG_DIR, "complex_centroid.npy")

        if not os.path.exists(simple_path) or not os.path.exists(complex_path):
            raise FileNotFoundError(
                "Pre-computed centroid files not found. "
                "Run 'nadirclaw build-centroids' to generate them."
            )

        simple_centroid = np.load(simple_path)
        complex_centroid = np.load(complex_path)

        return simple_centroid, complex_centroid

    # ------------------------------------------------------------------
    # Core classification
    # ------------------------------------------------------------------

    def classify(self, prompt: str) -> Tuple[bool, float]:
        """
        Classify a prompt as simple or complex.

        Borderline cases (confidence < threshold) are biased toward complex --
        it is cheaper to over-serve a simple prompt than to under-serve a
        complex one.

        Returns:
            (is_complex, confidence) where confidence is in [0, 1].
            confidence near 0 means borderline; near 1 means very clear.
        """
        from nadirclaw.settings import settings

        threshold = settings.CONFIDENCE_THRESHOLD

        emb = self.encoder.encode([prompt], show_progress_bar=False)[0]
        emb = emb / np.linalg.norm(emb)

        sim_simple = float(np.dot(emb, self._simple_centroid))
        sim_complex = float(np.dot(emb, self._complex_centroid))

        confidence = abs(sim_complex - sim_simple)

        if confidence < threshold:
            is_complex = True
        else:
            is_complex = sim_complex > sim_simple

        return is_complex, confidence

    # ------------------------------------------------------------------
    # Context-aware classification
    # ------------------------------------------------------------------

    def classify_with_context(
        self, prompt: str, system_prompt: Optional[str] = None,
    ) -> Tuple[bool, float]:
        """Classify a prompt, optionally incorporating system prompt context.

        When *system_prompt* is provided, the first 500 characters are
        prepended to the user prompt before embedding.  This gives the
        centroid classifier awareness of the conversation context (e.g. an
        agentic system prompt defining tools, personas, or constraints)
        even when the user message is very short.

        Returns:
            (is_complex, confidence) -- same contract as ``classify``.
        """
        if system_prompt:
            truncated = system_prompt[:500].strip()
            prompt = f"{truncated} | {prompt}"
        return self.classify(prompt)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def analyze(self, text: str, **kwargs) -> Dict[str, Any]:
        """Async analyse -- conforms to the analyzer interface."""
        return self._analyze_sync(text)

    def _analyze_sync(self, text: str) -> Dict[str, Any]:
        start = time.time()
        is_complex, confidence = self.classify(text)

        complexity_score = self._confidence_to_score(is_complex, confidence)

        # Three-tier routing: use score thresholds to determine tier
        tier_name, tier = self._score_to_tier(complexity_score)

        recommended_model, recommended_provider = self._select_model_by_tier(tier_name)

        latency_ms = int((time.time() - start) * 1000)

        return {
            "recommended_model": recommended_model,
            "recommended_provider": recommended_provider,
            "confidence": confidence,
            "complexity_score": complexity_score,
            "complexity_tier": tier,
            "complexity_name": tier_name,
            "tier": tier,
            "tier_name": tier_name,
            "reasoning": (
                f"Binary classifier: {tier_name} "
                f"(score={complexity_score:.3f}, confidence={confidence:.3f})"
            ),
            "ranked_models": [],
            "analyzer_latency_ms": latency_ms,
            "analyzer_type": "binary",
            "selection_method": "binary_classifier",
            "model_type": "binary_classifier",
        }

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_model(is_complex: bool) -> Tuple[str, str]:
        """Pick the model based on binary tier classification (legacy)."""
        from nadirclaw.settings import settings

        model = settings.COMPLEX_MODEL if is_complex else settings.SIMPLE_MODEL
        provider = model.split("/")[0] if "/" in model else "api"
        return model, provider

    @staticmethod
    def _select_model_by_tier(tier_name: str) -> Tuple[str, str]:
        """Pick the model based on three-tier classification."""
        from nadirclaw.settings import settings

        if tier_name == "complex":
            model = settings.COMPLEX_MODEL
        elif tier_name in ("mid", "medium"):
            model = settings.MID_MODEL
        else:
            model = settings.SIMPLE_MODEL
        provider = model.split("/")[0] if "/" in model else "api"
        return model, provider

    @staticmethod
    def _confidence_to_score(is_complex: bool, confidence: float) -> float:
        """Map binary decision + confidence to a 0-1 complexity score."""
        if is_complex:
            return 0.5 + min(confidence * 5, 0.5)
        else:
            return 0.5 - min(confidence * 5, 0.5)

    @staticmethod
    def _score_to_tier(complexity_score: float) -> Tuple[str, int]:
        """Map a 0-1 complexity score to a tier name and numeric tier.

        Uses configurable thresholds from NADIRCLAW_TIER_THRESHOLDS.
        If MID_MODEL is not set, falls back to binary (simple/complex).

        Returns (tier_name, tier_number).
        """
        from nadirclaw.settings import settings

        simple_max, complex_min = settings.TIER_THRESHOLDS

        if settings.has_mid_tier:
            if complexity_score <= simple_max:
                return "simple", 1
            elif complexity_score >= complex_min:
                return "complex", 3
            else:
                return "mid", 2
        else:
            # No mid model configured -- binary routing
            if complexity_score >= 0.5:
                return "complex", 3
            else:
                return "simple", 1


# ===================================================================
# Structural feature extractor (used as cascade escalation path)
# ===================================================================

# Pre-compiled patterns for structural analysis
_CODE_BLOCK_RE = re.compile(r"```")
_QUESTION_MARK_RE = re.compile(r"\?")
_BULLET_RE = re.compile(r"^\s*[-*]\s", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
_URL_RE = re.compile(r"https?://")
_TECHNICAL_KEYWORDS = re.compile(
    r"\b("
    # Software engineering
    r"implement|architecture|design|optimize|debug|refactor|deploy|migrate"
    r"|distributed|microservices?|kubernetes|k8s|docker|terraform|helm"
    # Databases
    r"|database|schema|migration|index(?:ing)?|postgres(?:ql)?|mysql|redis"
    r"|mongodb|dynamo|replica|replication|failover|backup|restore"
    r"|connection.?pool|query|transaction"
    # Infrastructure / ops
    r"|load.?balanc|auto.?scal|VPC|VPN|DNS|CDN|SSL|certificate|502|503|504"
    r"|nginx|ingress|proxy|gateway|endpoint|cluster|node|pod"
    r"|OOM|CrashLoop|health.?check|uptime|SLA|SLO|downtime|outage"
    r"|CPU|memory|disk|storage|bandwidth|throughput"
    # Security / compliance
    r"|security|authentication|authorization|OAuth|JWT|TLS|HTTPS"
    r"|SAML|SSO|IAM|HIPAA|SOC2|PCI|encryption|firewall|breach"
    r"|vulnerability|audit|compliance"
    # API / protocols
    r"|API|REST|GraphQL|gRPC|WebSocket|webhook|MQTT"
    r"|CI/CD|pipeline|monitoring|alerting|logging|tracing|metrics"
    # ML
    r"|machine learning|neural|model|training|inference"
    # Concurrency
    r"|concurrency|thread|async|parallel|race condition"
    # Performance
    r"|performance|latency|throughput|scalab|p99|p95|p50"
    # Testing
    r"|test(?:ing)?|coverage|integration|e2e|benchmark"
    # Config / deployment
    r"|container|image|registry|rollback|canary|blue.?green|rolling"
    r"|rate.?limit|timeout|retry|backoff|circuit.?breaker"
    r")\b",
    re.IGNORECASE,
)
_MULTI_STEP_KEYWORDS = re.compile(
    r"\b("
    r"step[- ]by[- ]step|first.*then.*finally"
    r"|compare and contrast|analyze.*and.*recommend"
    r"|design.*implement.*test"
    r"|identify.*fix.*verify"
    r"|create.*configure.*deploy"
    r"|walk me through|how do we set up|help me set up"
    r"|what can we do|what's the best strategy"
    r"|troubleshoot|diagnose|investigate|root cause"
    r"|production outage|incident|post.?mortem"
    r")\b",
    re.IGNORECASE,
)
_CONSTRAINT_KEYWORDS = re.compile(
    r"\b("
    r"million|billion|concurrent|real[- ]time|latency[- ]sensitive"
    r"|high[- ]availab|fault[- ]tolerant|zero[- ]downtime"
    r"|RPO|RTO|SLA|SLO|HIPAA|PCI|SOC2|GDPR"
    r"|budget|compliance|production|P[12]"
    r"|req/(?:min|sec|s)|transactions?/sec|TPS|QPS"
    r"|affected|outage|breach|incident"
    r"|\d+\s*(?:GB|TB|MB|Mi|Gi|users?|connections?|pods?|nodes?|replicas?)"
    r")\b",
    re.IGNORECASE,
)

# Patterns indicating tool-call / agentic JSON payloads in prompt text
_TOOL_CALL_JSON_RE = re.compile(
    r'"(?:function|tool_call|tool_use|parameters|tool_choice|tools)"',
    re.IGNORECASE,
)


def extract_structural_features(text: str) -> Dict[str, float]:
    """Extract 20+ structural features from prompt text.

    All features are normalised to roughly [0, 1] range.
    No ML model needed -- pure regex and counting.
    """
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    lines = text.split("\n")
    line_count = len(lines)

    # Counts
    code_blocks = len(_CODE_BLOCK_RE.findall(text)) // 2  # pairs
    question_marks = len(_QUESTION_MARK_RE.findall(text))
    bullets = len(_BULLET_RE.findall(text))
    numbered_items = len(_NUMBERED_RE.findall(text))
    urls = len(_URL_RE.findall(text))
    technical_matches = _TECHNICAL_KEYWORDS.findall(text)
    technical_count = len(technical_matches)
    multi_step_count = len(_MULTI_STEP_KEYWORDS.findall(text))
    constraint_count = len(_CONSTRAINT_KEYWORDS.findall(text))
    comma_count = text.count(",")
    semicolon_count = text.count(";")

    # Derived features
    avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
    sentence_count = max(1, text.count(".") + text.count("!") + text.count("?"))
    avg_sentence_length = word_count / sentence_count

    # Unique technical keywords (deduplicated)
    unique_technical = len(set(k.lower() for k in technical_matches))

    return {
        # Length features (log-scaled for normalisation)
        "word_count_log": min(math.log1p(word_count) / 7.0, 1.0),          # ~1100 words -> 1.0
        "char_count_log": min(math.log1p(char_count) / 9.0, 1.0),          # ~8100 chars -> 1.0
        "line_count_log": min(math.log1p(line_count) / 5.0, 1.0),          # ~148 lines -> 1.0
        # Structural complexity
        "code_blocks": min(code_blocks / 3.0, 1.0),
        "question_marks": min(question_marks / 5.0, 1.0),
        "bullets": min(bullets / 10.0, 1.0),
        "numbered_items": min(numbered_items / 8.0, 1.0),
        "urls": min(urls / 3.0, 1.0),
        "commas": min(comma_count / 15.0, 1.0),
        "semicolons": min(semicolon_count / 5.0, 1.0),
        # Technical depth
        "technical_keywords": min(technical_count / 10.0, 1.0),
        "unique_technical": min(unique_technical / 8.0, 1.0),
        "multi_step": min(multi_step_count / 3.0, 1.0),
        "constraints": min(constraint_count / 3.0, 1.0),
        # Linguistic complexity
        "avg_word_length": min(avg_word_length / 8.0, 1.0),
        "avg_sentence_length": min(avg_sentence_length / 30.0, 1.0),
        # Combined signals
        "has_code": 1.0 if code_blocks > 0 else 0.0,
        "has_list": 1.0 if (bullets + numbered_items) > 0 else 0.0,
        "has_urls": 1.0 if urls > 0 else 0.0,
        "has_constraints": 1.0 if constraint_count > 0 else 0.0,
        "has_multi_step": 1.0 if multi_step_count > 0 else 0.0,
    }


def structural_complexity_score(features: Dict[str, float]) -> float:
    """Compute a weighted complexity score from structural features.

    Returns a value in [0, 1].
    """
    weights = {
        "word_count_log": 0.10,
        "char_count_log": 0.05,
        "line_count_log": 0.05,
        "code_blocks": 0.08,
        "question_marks": -0.05,  # many questions often = simple FAQ
        "bullets": 0.04,
        "numbered_items": 0.04,
        "urls": 0.02,
        "commas": 0.03,
        "semicolons": 0.02,
        "technical_keywords": 0.15,
        "unique_technical": 0.12,
        "multi_step": 0.08,
        "constraints": 0.10,
        "avg_word_length": 0.03,
        "avg_sentence_length": 0.04,
        "has_code": 0.03,
        "has_list": 0.02,
        "has_urls": 0.01,
        "has_constraints": 0.06,
        "has_multi_step": 0.05,
    }

    score = sum(features.get(k, 0.0) * w for k, w in weights.items())
    # Clamp after summing (negative weights can push below 0)
    return max(0.0, min(1.0, score))


# ===================================================================
# TernaryClassifier -- ternary centroid classification (cascade stage 1)
# ===================================================================

class TernaryClassifier:
    """
    Classifies prompts as simple, medium, or complex using three centroid
    groups with temperature-scaled softmax probabilities.

    The complex tier uses k-means sub-clustering (ported from Horizen's
    BinaryComplexityClassifier v4.0) to capture diverse sub-categories.

    Does NOT require DistilBERT -- uses the same all-MiniLM-L6-v2 encoder
    as the binary classifier.
    """

    CLASSIFIER_VERSION = "1.0"

    def __init__(self):
        from nadirclaw.encoder import get_shared_encoder_sync
        from nadirclaw.settings import settings

        self.encoder = get_shared_encoder_sync()
        self._temperature = settings.CASCADE_SOFTMAX_TEMPERATURE
        self._sub_clusters = settings.CASCADE_COMPLEX_SUB_CLUSTERS

        # Try loading cached ternary centroids; fall back to computing them
        cached = self._try_load_centroid_cache()
        if cached is not None:
            self._simple_centroid, self._medium_centroid, self._complex_centroids = cached
            logger.info("TernaryClassifier v%s loaded cached centroids", self.CLASSIFIER_VERSION)
        else:
            (
                self._simple_centroid,
                self._medium_centroid,
                self._complex_centroids,
            ) = self._compute_centroids()
            self._save_centroid_cache()
            logger.info("TernaryClassifier v%s computed and cached centroids", self.CLASSIFIER_VERSION)

    # ------------------------------------------------------------------
    # Centroid cache
    # ------------------------------------------------------------------

    @staticmethod
    def _centroid_cache_path() -> str:
        return os.path.join(_PKG_DIR, "ternary_centroids.npy")

    def _try_load_centroid_cache(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        path = self._centroid_cache_path()
        try:
            data = np.load(path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object and len(data) == 3:
                return data[0], data[1], data[2]
        except Exception:
            pass
        return None

    def _save_centroid_cache(self) -> None:
        path = self._centroid_cache_path()
        try:
            stacked = np.array(
                [self._simple_centroid, self._medium_centroid, self._complex_centroids],
                dtype=object,
            )
            np.save(path, stacked, allow_pickle=True)
            logger.info("Saved ternary centroid cache to %s", path)
        except Exception as e:
            logger.warning("Could not save ternary centroid cache: %s", e)

    # ------------------------------------------------------------------
    # Centroid computation
    # ------------------------------------------------------------------

    def _compute_centroids(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Embed all prototypes and return L2-normalised centroids.

        Simple and medium get a single centroid each.
        Complex gets k sub-centroids via k-means.
        """
        from nadirclaw.prototypes import COMPLEX_PROTOTYPES, MEDIUM_PROTOTYPES, SIMPLE_PROTOTYPES

        simple_embs = self.encoder.encode(SIMPLE_PROTOTYPES, show_progress_bar=False)
        medium_embs = self.encoder.encode(MEDIUM_PROTOTYPES, show_progress_bar=False)
        complex_embs = self.encoder.encode(COMPLEX_PROTOTYPES, show_progress_bar=False)

        simple_centroid = simple_embs.mean(axis=0)
        simple_centroid = simple_centroid / np.linalg.norm(simple_centroid)

        medium_centroid = medium_embs.mean(axis=0)
        medium_centroid = medium_centroid / np.linalg.norm(medium_centroid)

        # Multi-centroid for complex tier via k-means
        k = min(self._sub_clusters, len(complex_embs))
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
                complex_centroids = np.array(sub_centroids)
                logger.info(
                    "Complex tier: %d sub-centroids from %d prototypes",
                    len(sub_centroids), len(complex_embs),
                )
            except ImportError:
                logger.warning("sklearn not available -- using single complex centroid")
                c = complex_embs.mean(axis=0)
                c = c / np.linalg.norm(c)
                complex_centroids = c.reshape(1, -1)
        else:
            c = complex_embs.mean(axis=0)
            c = c / np.linalg.norm(c)
            complex_centroids = c.reshape(1, -1)

        return simple_centroid, medium_centroid, complex_centroids

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, prompt: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify a prompt as simple, medium, or complex.

        Returns:
            (tier_name, confidence, tier_probabilities)
        """
        emb = self.encoder.encode([prompt], show_progress_bar=False)[0]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        # Complex tier: max similarity across sub-centroids
        complex_sims = self._complex_centroids @ emb
        complex_sim = float(np.max(complex_sims))

        sims = {
            "simple": float(np.dot(emb, self._simple_centroid)),
            "medium": float(np.dot(emb, self._medium_centroid)),
            "complex": complex_sim,
        }

        # Temperature-scaled softmax for calibrated probabilities
        logits = np.array([sims["simple"], sims["medium"], sims["complex"]])
        exp_logits = np.exp((logits - logits.max()) / self._temperature)
        probs = exp_logits / exp_logits.sum()

        tier_probs = {
            "simple": float(probs[0]),
            "medium": float(probs[1]),
            "complex": float(probs[2]),
        }

        tier_names = ["simple", "medium", "complex"]
        best_idx = int(np.argmax(probs))
        best_tier = tier_names[best_idx]
        confidence = float(probs[best_idx])

        confidence = calibrate_confidence(confidence, "cascade_centroid")

        return best_tier, confidence, tier_probs

    def embed(self, prompt: str) -> np.ndarray:
        """Return the normalised embedding for a prompt (reusable by cascade)."""
        emb = self.encoder.encode([prompt], show_progress_bar=False)[0]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def classify_from_embedding(
        self, emb: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:
        """Classify using a pre-computed normalised embedding."""
        complex_sims = self._complex_centroids @ emb
        complex_sim = float(np.max(complex_sims))

        sims = {
            "simple": float(np.dot(emb, self._simple_centroid)),
            "medium": float(np.dot(emb, self._medium_centroid)),
            "complex": complex_sim,
        }

        logits = np.array([sims["simple"], sims["medium"], sims["complex"]])
        exp_logits = np.exp((logits - logits.max()) / self._temperature)
        probs = exp_logits / exp_logits.sum()

        tier_probs = {
            "simple": float(probs[0]),
            "medium": float(probs[1]),
            "complex": float(probs[2]),
        }

        tier_names = ["simple", "medium", "complex"]
        best_idx = int(np.argmax(probs))
        best_tier = tier_names[best_idx]
        confidence = float(probs[best_idx])
        confidence = calibrate_confidence(confidence, "cascade_centroid")

        return best_tier, confidence, tier_probs


# ===================================================================
# ConfidenceAwareCascadeClassifier
# ===================================================================

class ConfidenceAwareCascadeClassifier:
    """
    Confidence-aware cascade classifier.

    Stage 1: Fast ternary centroid classification (~10 ms).
    Stage 2 (escalation): If confidence < threshold, runs structural feature
             analysis on the prompt text and blends the two signals.

    ~10 ms for clear cases (most requests), ~12 ms for ambiguous ones.
    No new dependencies required (no DistilBERT, no API calls).
    """

    CLASSIFIER_VERSION = "1.0"

    def __init__(self):
        from nadirclaw.settings import settings

        self._ternary = TernaryClassifier()
        self._threshold = settings.CASCADE_CONFIDENCE_THRESHOLD

        logger.info(
            "ConfidenceAwareCascadeClassifier v%s ready (threshold=%.2f)",
            self.CLASSIFIER_VERSION, self._threshold,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def analyze(self, text: str, **kwargs) -> Dict[str, Any]:
        """Async analyse -- conforms to the analyzer interface."""
        return self._analyze_sync(text)

    def classify_with_context(
        self, prompt: str, system_prompt: Optional[str] = None,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Classify a prompt, optionally incorporating system prompt context.

        When *system_prompt* is provided, the first 500 characters are
        prepended to the user prompt before embedding in the centroid
        stage.  This gives the classifier awareness of the conversation
        context (e.g. an agentic system prompt defining tools, personas,
        and constraints) even when the user message itself is very short
        (e.g. "do it").

        The structural feature analysis still runs on the original
        *prompt* combined with the full *system_prompt* for maximum
        signal extraction.

        Returns:
            (tier_name, confidence, metadata_dict) -- same contract as
            ``classify``.
        """
        if not system_prompt:
            return self.classify(prompt)

        # Build an augmented prompt for the centroid embedding stage
        truncated_sys = system_prompt[:500].strip()
        augmented_prompt = f"{truncated_sys} | {prompt}"

        # Stage 1: centroid classification on augmented prompt
        tier_name, confidence, tier_probs = self._ternary.classify(augmented_prompt)

        metadata: Dict[str, Any] = {
            "tier_probabilities": tier_probs,
            "confidence_escalated": False,
            "stage1_tier": tier_name,
            "stage1_confidence": confidence,
            "system_prompt_injected": True,
        }

        # Stage 2: structural features on FULL prompt + system prompt
        # for maximum signal extraction
        full_text = f"{system_prompt}\n{prompt}"
        features = extract_structural_features(full_text)
        struct_score = structural_complexity_score(features)
        struct_confidence = calibrate_confidence(
            abs(struct_score - 0.5) * 2.0, "cascade_struct"
        )

        tech_count = features.get("technical_keywords", 0)
        constraint_count = features.get("constraints", 0)
        multi_step = features.get("multi_step", 0)

        # Decision logic (same rules as classify, including agentic rules)
        if _TOOL_CALL_JSON_RE.search(full_text):
            final_tier = "complex"
        elif multi_step >= 0.3 and features.get("has_code", 0) and tech_count >= 0.2:
            final_tier = "complex" if tech_count >= 0.4 else "medium"
        elif tier_name == "complex" and confidence >= self._threshold:
            final_tier = "complex"
        elif tier_name == "simple" and confidence >= self._threshold and tech_count < 0.2:
            final_tier = "simple"
        elif (tech_count >= 0.3 and constraint_count >= 0.6) or \
             (tech_count >= 0.3 and multi_step >= 0.3) or \
             (tech_count >= 0.2 and constraint_count >= 0.9):
            final_tier = "complex"
        elif tech_count >= 0.1:
            final_tier = "medium"
        else:
            final_tier = "simple"

        final_confidence = max(confidence, struct_confidence)
        final_confidence = calibrate_confidence(final_confidence, "cascade_centroid")

        metadata["confidence_escalated"] = True
        metadata["structural_features"] = features
        metadata["structural_score"] = struct_score
        metadata["structural_confidence"] = struct_confidence
        metadata["centroid_tier"] = tier_name
        metadata["final_tier"] = final_tier
        metadata["tech_keywords"] = tech_count
        metadata["constraints"] = constraint_count

        return final_tier, final_confidence, metadata

    def classify(self, prompt: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Classify a prompt with confidence-aware cascade.

        Returns:
            (tier_name, confidence, metadata_dict)
        """
        # Stage 1: fast centroid classification
        tier_name, confidence, tier_probs = self._ternary.classify(prompt)

        metadata: Dict[str, Any] = {
            "tier_probabilities": tier_probs,
            "confidence_escalated": False,
            "stage1_tier": tier_name,
            "stage1_confidence": confidence,
        }

        # ── Stage 2: structural feature analysis ──
        # Always run structural features — even for "confident" centroid
        # results, because the centroid struggles with medium vs complex.
        features = extract_structural_features(prompt)
        struct_score = structural_complexity_score(features)
        struct_confidence = calibrate_confidence(
            abs(struct_score - 0.5) * 2.0, "cascade_struct"
        )

        tech_count = features.get("technical_keywords", 0)
        constraint_count = features.get("constraints", 0)
        multi_step = features.get("multi_step", 0)

        # ── Decision logic ──
        # Rule 0a: Prompt contains tool-call-like JSON patterns (agentic) → complex.
        # JSON payloads with "function", "tool_call", "parameters" indicate
        # agentic tool-use workflows that need a capable model.
        if _TOOL_CALL_JSON_RE.search(prompt):
            final_tier = "complex"

        # Rule 0b: Strong multi-step + code + technical signals → agentic → at least medium.
        # Multi-step reasoning with code and technical terms is likely an
        # agent orchestration prompt even without explicit tool JSON.
        elif multi_step >= 0.3 and features.get("has_code", 0) and tech_count >= 0.2:
            final_tier = "complex" if tech_count >= 0.4 else "medium"

        # Rule 1: Centroid says complex with HIGH confidence → trust it.
        # The centroid is reliable when it strongly says complex.
        elif tier_name == "complex" and confidence >= self._threshold:
            final_tier = "complex"

        # Rule 2: Centroid says simple with HIGH confidence AND no/low
        # technical keywords → trust it (simple FAQ/lookup).
        elif tier_name == "simple" and confidence >= self._threshold and tech_count < 0.2:
            final_tier = "simple"

        # Rule 3: High tech + high constraints/multi-step → complex.
        # Architecture design, production outages, compliance gaps.
        elif (tech_count >= 0.3 and constraint_count >= 0.6) or \
             (tech_count >= 0.3 and multi_step >= 0.3) or \
             (tech_count >= 0.2 and constraint_count >= 0.9):
            final_tier = "complex"

        # Rule 4: Has technical keywords but low constraints → medium.
        # Single-issue troubleshooting: OOMKilled, 502 errors, pool
        # exhaustion, config help.
        elif tech_count >= 0.1:
            final_tier = "medium"

        # Rule 5: No technical keywords → simple.
        else:
            final_tier = "simple"

        # Blended confidence
        final_confidence = max(confidence, struct_confidence)
        final_confidence = calibrate_confidence(final_confidence, "cascade_centroid")

        metadata["confidence_escalated"] = True
        metadata["structural_features"] = features
        metadata["structural_score"] = struct_score
        metadata["structural_confidence"] = struct_confidence
        metadata["centroid_tier"] = tier_name
        metadata["final_tier"] = final_tier
        metadata["tech_keywords"] = tech_count
        metadata["constraints"] = constraint_count

        return final_tier, final_confidence, metadata

    def _analyze_sync(self, text: str) -> Dict[str, Any]:
        start = time.time()

        tier_name, confidence, metadata = self.classify(text)
        escalated = metadata.get("confidence_escalated", False)

        tier_map = {"simple": 1, "medium": 2, "complex": 3}
        tier = tier_map.get(tier_name, 2)

        complexity_score = _tier_to_score(tier_name, confidence)
        recommended_model, recommended_provider = _select_model_by_tier(tier_name)

        latency_ms = int((time.time() - start) * 1000)

        method = "cascade_structural" if escalated else "cascade_centroid"
        reasoning = (
            f"Cascade v{self.CLASSIFIER_VERSION}: {tier_name} "
            f"(confidence={confidence:.3f}"
        )
        if escalated:
            reasoning += (
                f", escalated: struct_score={metadata.get('structural_score', 0):.3f}"
                f", struct_tier={metadata.get('structural_tier', '?')}"
            )
        reasoning += ")"

        result: Dict[str, Any] = {
            "recommended_model": recommended_model,
            "recommended_provider": recommended_provider,
            "confidence": confidence,
            "complexity_score": complexity_score,
            "complexity_tier": tier,
            "complexity_name": tier_name,
            "tier": tier,
            "tier_name": tier_name,
            "reasoning": reasoning,
            "ranked_models": [],
            "analyzer_latency_ms": latency_ms,
            "analyzer_type": "cascade",
            "selection_method": method,
            "model_type": "cascade_classifier",
            "confidence_escalated": escalated,
            "classifier_version": self.CLASSIFIER_VERSION,
        }

        # Include tier probabilities from centroid stage
        result["tier_probabilities"] = metadata.get("tier_probabilities", {})

        # If escalated, include structural details
        if escalated:
            result["structural_score"] = metadata.get("structural_score")
            result["structural_tier"] = metadata.get("structural_tier")
            result["blend_weights"] = metadata.get("blend_weights")

        return result


# ===================================================================
# Shared helpers
# ===================================================================

def _tier_to_score(tier_name: str, confidence: float) -> float:
    """Map ternary tier + confidence to a 0-1 complexity score."""
    base = {"simple": 0.15, "medium": 0.5, "complex": 0.85}.get(tier_name, 0.5)
    offset = min(confidence * 2, 0.15)
    if tier_name == "complex":
        return min(base + offset, 1.0)
    elif tier_name == "simple":
        return max(base - offset, 0.0)
    else:
        return base


def _select_model_by_tier(tier_name: str) -> Tuple[str, str]:
    """Pick the model based on tier classification."""
    from nadirclaw.settings import settings

    if tier_name == "complex":
        model = settings.COMPLEX_MODEL
    elif tier_name in ("mid", "medium"):
        model = settings.MID_MODEL
    else:
        model = settings.SIMPLE_MODEL
    provider = model.split("/")[0] if "/" in model else "api"
    return model, provider


# ===================================================================
# Singleton helpers
# ===================================================================
_binary_singleton: Optional[BinaryComplexityClassifier] = None
_cascade_singleton: Optional[ConfidenceAwareCascadeClassifier] = None


def get_binary_classifier() -> BinaryComplexityClassifier:
    """Return the singleton binary classifier instance."""
    global _binary_singleton
    if _binary_singleton is None:
        _binary_singleton = BinaryComplexityClassifier()
    return _binary_singleton


def get_cascade_classifier() -> ConfidenceAwareCascadeClassifier:
    """Return the singleton cascade classifier instance."""
    global _cascade_singleton
    if _cascade_singleton is None:
        _cascade_singleton = ConfidenceAwareCascadeClassifier()
    return _cascade_singleton


_trained_singleton = None


def get_trained_classifier():
    """Return the singleton trained classifier instance."""
    global _trained_singleton
    if _trained_singleton is None:
        from nadirclaw.trained_classifier import TrainedClassifier
        _trained_singleton = TrainedClassifier()
    return _trained_singleton


def get_classifier():
    """Return the active classifier based on NADIRCLAW_CLASSIFIER setting.

    Returns a BinaryComplexityClassifier, ConfidenceAwareCascadeClassifier,
    or TrainedClassifier, all of which expose a ``classify(prompt)`` method.

    Set NADIRCLAW_CLASSIFIER=trained for the sklearn-based classifier (95%+ accuracy).
    """
    from nadirclaw.settings import settings

    if settings.CLASSIFIER == "trained":
        return get_trained_classifier()
    if settings.CLASSIFIER == "cascade":
        return get_cascade_classifier()
    return get_binary_classifier()


def warmup() -> None:
    """Pre-warm the encoder and load centroids once at startup."""
    from nadirclaw.settings import settings

    if settings.CLASSIFIER == "trained":
        logger.info("Warming up TrainedClassifier ...")
        get_trained_classifier()
        logger.info("TrainedClassifier warmup complete")
    elif settings.CLASSIFIER == "cascade":
        logger.info("Warming up ConfidenceAwareCascadeClassifier ...")
        get_cascade_classifier()
        logger.info("ConfidenceAwareCascadeClassifier warmup complete")
    else:
        logger.info("Warming up BinaryComplexityClassifier ...")
        get_binary_classifier()
        logger.info("BinaryComplexityClassifier warmup complete")
