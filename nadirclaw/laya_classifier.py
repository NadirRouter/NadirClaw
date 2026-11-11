"""Laya typed-decisions complexity classifier (local CPU, opt-in).

``NADIRCLAW_COMPLEXITY_ANALYZER=laya``, after ``pip install "nadirclaw[laya]"``.

Laya (convaiinnovations/laya-typed-decisions, ModernBERT-large, Apache-2.0) is a
zero-shot decision model. It scores the work a prompt needs on a 3-level rubric
and returns P(simple, mid, complex). The tier comes from an under-routing risk
rule: escalate as soon as the probability of needing the higher tier reaches the
cutoff c (default 0.40).

    complex if p_complex >= c;  mid if p_mid + p_complex >= c;  else simple

Without a mid tier (NADIRCLAW_MID_MODEL unset) the rule has only one tier to
escalate to, so ``p_mid + p_complex >= c`` routes to complex.

Costs, all on the user's machine: torch + transformers, a one-time download of
the ~843 MB pinned weights into the Hugging Face cache, ~3 GB peak RAM, ~20 s to
load, and 0.15 s (short prompt) to 0.8 s (6,000 chars) of CPU per prompt on an
Apple M5 Pro; calls are serialized. Laya is conservative: at the default cutoff
it sends most prompts to complex, short ones like "hi" included, so expect far
fewer cheap-model routes than with the binary classifier. The cutoff was fitted on 249
labelled prompts, not on outcome data, so treat it as a setting to tune
(NADIRCLAW_LAYA_CUTOFF).

Fails closed: a missing package, a version other than the evaluated one, or a
load error makes ``get_classifier()`` serve the binary classifier, and an error
on one request serves that request from the binary classifier.
"""

import asyncio
import logging
import math
import threading
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

LAYA_REPO = "convaiinnovations/laya-typed-decisions"
LAYA_REVISION = "1a793eb568e6718f15941d08f85432581df534e3"
LAYA_VERSION = "0.3.20"
_MAX_CHARS = 6000  # the evaluated input was text[:6000]
_TIERS = ("simple", "mid", "complex")
# The files laya.load() reads; the rest of the repo (README, .gitattributes) is not needed.
_ALLOW_PATTERNS = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]

# The rubric the cutoff was fitted with. Changing a word changes the probabilities,
# so changing it means re-fitting the cutoff.
_INSTRUCTIONS = (
        "Size the work the answer to this prompt requires, not the length of the question. A "
        "one-sentence question is often medium or complex."
    )
_CRITERIA = [
    (
        "The answer is immediate: a lookup, a definition, one fact, one obvious step. For "
        "code: a single pass over the input or a direct built-in, no bookkeeping and no edge "
        "cases to reason about."
    ),
    (
        "You must choose an approach, know a specific domain, weigh a nuanced question, or "
        "produce several distinct items. For code: a particular algorithm, index or state "
        "bookkeeping, or edge cases, even when the request is one sentence."
    ),
    (
        "Multi-step reasoning, algorithm design, deep argumentation, or a long output with "
        "many constraints. For code: dynamic programming, graph or grid search, or a "
        "known-hard algorithm."
    ),
]
QUESTIONS: Dict[str, Dict[str, Any]] = {
    "tier_score": {"type": "score", "instructions": _INSTRUCTIONS, "criteria": _CRITERIA},
}

_warned_fallback = False


def _warn_once(message: str, *args: Any) -> None:
    global _warned_fallback
    if not _warned_fallback:
        logger.warning(message, *args)
        _warned_fallback = True


def risk_tier(p: Tuple[float, float, float], cutoff: float, has_mid: bool) -> str:
    """Escalate when the probability of needing a higher tier reaches the cutoff."""
    total = sum(p)
    if (len(p) != 3 or any(not math.isfinite(x) or not 0.0 <= x <= 1.0 for x in p)
            or not math.isclose(total, 1.0, abs_tol=0.015)
            or not math.isfinite(cutoff) or not 0.0 < cutoff <= 1.0):
        raise ValueError("expected three probabilities summing to one and a cutoff in (0, 1]")
    p = tuple(x / total for x in p)
    if p[2] >= cutoff:
        return "complex"
    if p[1] + p[2] >= cutoff:
        return "mid" if has_mid else "complex"
    return "simple"


def _snapshot_dir(explicit: str) -> str:
    """NADIRCLAW_LAYA_PATH if set, else the pinned revision from the HF cache (downloaded once)."""
    if explicit:
        return explicit
    from huggingface_hub import snapshot_download

    logger.info("Fetching Laya %s@%s (~843 MB on first use) ...", LAYA_REPO, LAYA_REVISION[:8])
    return snapshot_download(LAYA_REPO, revision=LAYA_REVISION, allow_patterns=_ALLOW_PATTERNS)


class LayaClassifier:
    """Laya tier classifier, falling back to the local binary classifier."""

    CLASSIFIER_VERSION = LAYA_VERSION

    def __init__(self):
        import laya

        from nadirclaw.settings import settings

        if laya.__version__ != LAYA_VERSION:
            raise RuntimeError(
                f"laya {laya.__version__} is installed but NadirClaw evaluated {LAYA_VERSION}; "
                f'run: pip install "nadirclaw[laya]"'
            )
        self._cutoff = settings.LAYA_CUTOFF  # a bad value raises: get_classifier() then serves binary
        self._path = _snapshot_dir(settings.LAYA_PATH)
        start = time.time()
        self._agent = laya.load(self._path, device="cpu")
        self._lock = threading.Lock()  # one prediction at a time on the shared model
        self._fallback: Optional[Any] = None
        self.predict("warm-up prompt")  # loads lazy imports now and checks the output shape
        logger.info(
            "LayaClassifier ready in %.1fs (cutoff %.2f, %s)",
            time.time() - start, self._cutoff, self._path,
        )

    def predict(self, text: str) -> Tuple[float, float, float]:
        with self._lock:
            answers = self._agent.predict((text or "")[:_MAX_CHARS], QUESTIONS)["answers"]
        probs = answers["tier_score"]["probabilities"]
        return tuple(float(probs[str(i)]) for i in range(3))

    async def _local_fallback(self, text: str, system_message: str) -> Dict[str, Any]:
        if self._fallback is None:
            from nadirclaw.classifier import get_binary_classifier
            self._fallback = get_binary_classifier()
        result = await self._fallback.analyze(text=text, system_message=system_message)
        result["analyzer_type"] = "laya-fallback-binary"
        return result

    async def analyze(self, text: str = "", system_message: str = "", **kwargs) -> Dict[str, Any]:
        """Async-compatible interface matching the server's expected API."""
        from nadirclaw.classifier import BinaryComplexityClassifier
        from nadirclaw.settings import settings

        start = time.time()
        try:
            # Off the event loop: a ~1 s CPU call would otherwise stall every open stream.
            p = await asyncio.to_thread(self.predict, text)
            tier_name = risk_tier(p, self._cutoff, settings.has_mid_tier)
        except Exception as e:
            _warn_once(
                "Laya classification failed (%s); serving the local binary classifier. "
                "Further failures are suppressed.",
                e,
            )
            return await self._local_fallback(text, system_message)

        tier = _TIERS.index(tier_name) + 1
        model, provider = BinaryComplexityClassifier._select_model_by_tier(tier_name)
        return {
            "recommended_model": model,
            "recommended_provider": provider,
            "confidence": max(p),  # Laya's top probability; uncalibrated, the cutoff is the calibration
            "complexity_score": (p[1] + 2 * p[2]) / 2,  # expected tier, scaled to 0-1
            "complexity_tier": tier,
            "complexity_name": tier_name,
            "tier": tier,
            "tier_name": tier_name,
            "tier_probabilities": dict(zip(_TIERS, p)),
            "reasoning": (
                f"Laya p=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) cutoff {self._cutoff:.2f} -> {tier_name}"
            ),
            "ranked_models": [],
            "analyzer_latency_ms": int((time.time() - start) * 1000),
            "analyzer_type": f"laya-v{self.CLASSIFIER_VERSION}",
            "selection_method": "laya",
            "model_type": "laya",
        }
