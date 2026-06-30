"""Trained DeBERTa-v3-small cross-encoder verifier (PolyForm Noncommercial snapshot).

This is the frozen snapshot used in the Nadir RouterArena submission
(PR #112, arena_F 0.7358). It is the same DeBERTa-v3-small cross-encoder
that powers the Nadir Pro hosted service, released here under the
PolyForm Noncommercial License 1.0.0 so that the RouterArena result is
reproducible end-to-end with the noncommercial NadirClaw router.

What's released
---------------
* Frozen INT8 / FP32 weights at the HuggingFace model id
  ``nadirclaw/cascade-verifier-v1``.
* Tokenizer + config to load via the standard
  ``transformers.AutoModelForSequenceClassification`` pipeline.

What's NOT released
-------------------
* The training pipeline (corpus builder, judge prompts, the curated
  RouterBench-derived triples).
* The adaptive retraining loop that keeps the production verifier
  current.

These remain proprietary to Nadir Pro. The frozen snapshot is enough
to reproduce the RouterArena number, but not enough to keep the
verifier fresh as new model families ship.

Interface
---------
``TrainedVerifier`` exposes the same shape as ``HeuristicVerifier``:

    >>> v = TrainedVerifier()                # downloads weights on first use
    >>> result = v.score(prompt, cheap_answer)
    >>> result.score, result.accepted        # float in [0, 1], bool

The ``reference_answer`` argument, when provided, is folded into the
structured ``text_pair`` the cross-encoder was trained on (see
``score()``). When ``None``, an empty ``EXPENSIVE:`` block is
substituted, matching the production backend's behaviour. The
``expect_json`` argument is accepted for parity with
``HeuristicVerifier`` but is currently ignored by the trained model.

Dependencies
------------
This module imports ``torch`` and ``transformers`` lazily. Install with::

    pip install nadirclaw[trained]

The heuristic verifier remains the default for the rule-engine cascade,
so users who do not want the transformer stack pay nothing for it.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# Default HuggingFace model id for the released snapshot. Override with
# the ``NADIRCLAW_TRAINED_VERIFIER_MODEL`` env var or the ``model_id``
# constructor argument if you mirror the weights elsewhere.
DEFAULT_MODEL_ID = "nadirclaw/cascade-verifier-v1"

# Default acceptance threshold. Calibrated against the same held-out
# RouterBench test split as the HeuristicVerifier default. The trained
# verifier is sharper, so the same 0.80 cutoff lands at higher precision
# and lower false-accept rate. Override via ``threshold=`` constructor arg.
DEFAULT_TRAINED_THRESHOLD = 0.80

# Maximum sequence length we feed to the cross-encoder. DeBERTa-v3-small
# is positional-embedding capped at 512; we leave room for the [CLS] +
# separator tokens.
_MAX_SEQ_LEN = 512


@dataclass
class TrainedScore:
    """Structured score from the trained verifier.

    Mirrors the public shape of ``HeuristicScore`` so downstream code
    (cascade, analytics) does not need to special-case the verifier
    type.
    """

    score: float
    accepted: bool
    threshold: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "accepted": self.accepted,
            "threshold": self.threshold,
            "reasons": list(self.reasons),
            "verifier": "trained",
        }


class TrainedVerifier:
    """Cross-encoder verifier loaded from a HuggingFace checkpoint.

    Parameters
    ----------
    model_id:
        HuggingFace model id or local path. Defaults to
        ``nadirclaw/cascade-verifier-v1``.
    threshold:
        Score cutoff for acceptance. Defaults to 0.80 (same as the
        heuristic, so cascade behaviour is consistent when you swap).
    device:
        ``"cpu"`` (default), ``"cuda"``, or ``"mps"``. ``"auto"`` picks
        the best available device.
    cache_dir:
        Optional local cache directory passed through to
        ``transformers.from_pretrained``.

    Notes
    -----
    Construction is lazy: the model is loaded on the first call to
    ``score()`` so that import time stays cheap. Pass
    ``preload=True`` to force the load at construction time (useful
    in test setup and warm-up).
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        threshold: float = DEFAULT_TRAINED_THRESHOLD,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        preload: bool = False,
    ):
        self.model_id = model_id or os.environ.get(
            "NADIRCLAW_TRAINED_VERIFIER_MODEL", DEFAULT_MODEL_ID
        )
        self.threshold = float(threshold)
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._tokenizer = None
        self._resolved_device = None
        if preload:
            self._ensure_loaded()

    # ------------------------------------------------------------------
    # Loading

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:  # pragma: no cover - hardware-dependent
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if (
                getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()
            ):
                return "mps"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch  # noqa: F401  (used implicitly by transformers)
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "TrainedVerifier requires the 'trained' extras. "
                "Install with: pip install nadirclaw[trained]"
            ) from exc

        logger.info(
            "Loading TrainedVerifier weights from %s (cache_dir=%s)",
            self.model_id,
            self.cache_dir,
        )
        kwargs = {}
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, **kwargs
        )
        self._resolved_device = self._resolve_device()
        self._model.to(self._resolved_device)
        self._model.eval()

    def is_available(self) -> bool:
        """Return True if the model can be loaded.

        Performs the lazy load if it hasn't happened yet. Returns False
        on any ImportError or download failure so callers can fall back
        to the heuristic without crashing.
        """
        try:
            self._ensure_loaded()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("TrainedVerifier not available: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Scoring

    def score(
        self,
        prompt: str,
        cheap_answer: str,
        reference_answer: Optional[str] = None,
        expect_json: bool = False,
    ) -> TrainedScore:
        """Score how acceptable ``cheap_answer`` is for ``prompt``.

        The cross-encoder was trained on inputs of the form
        ``(prompt, "CHEAP:\\n{cheap}\\n\\nEXPENSIVE:\\n{reference}")``.
        ``reference_answer`` is folded into the structured ``text_pair``
        when provided; when ``None`` an empty ``EXPENSIVE:`` block is
        substituted, matching the production backend at
        ``getnadir.dev/backend/app/services/verifier_model.py``.

        ``expect_json`` is accepted for interface parity with
        ``HeuristicVerifier`` and is currently ignored by the trained
        model.
        """
        self._ensure_loaded()

        # Empty response gets a deterministic 0.0 without bothering the
        # model. Matches HeuristicVerifier behaviour and saves a forward
        # pass on the obvious-reject case.
        cheap = (cheap_answer or "").strip()
        if not cheap:
            return TrainedScore(
                score=0.0,
                accepted=False,
                threshold=self.threshold,
                reasons=["empty_response"],
            )

        import torch

        # Match the training format used by the production backend
        # (getnadir.dev/backend/app/services/verifier_model.py) and
        # documented on the HuggingFace model card. Without the
        # ``CHEAP:``/``EXPENSIVE:`` wrapper the scores drift against
        # the calibrated tau=0.80 acceptance threshold.
        text_pair = (
            f"CHEAP:\n{cheap}\n\nEXPENSIVE:\n{(reference_answer or '').strip()}"
        )
        enc = self._tokenizer(
            prompt or "",
            text_pair,
            truncation=True,
            max_length=_MAX_SEQ_LEN,
            padding=False,
            return_tensors="pt",
        )
        enc = {k: v.to(self._resolved_device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._model(**enc).logits

        # Two-class head: probability of the "accept" class via softmax.
        # Single-class head: sigmoid of the logit.
        if logits.shape[-1] == 1:
            score = torch.sigmoid(logits.squeeze(-1)).item()
        else:
            probs = torch.softmax(logits, dim=-1)
            score = probs[..., 1].item()

        score = max(0.0, min(1.0, float(score)))
        return TrainedScore(
            score=score,
            accepted=score >= self.threshold,
            threshold=self.threshold,
            reasons=[],
        )


# ----------------------------------------------------------------------
# Convenience accessor mirroring ``get_heuristic_verifier``.

_singleton: Optional[TrainedVerifier] = None


def get_trained_verifier(
    threshold: float = DEFAULT_TRAINED_THRESHOLD,
    model_id: Optional[str] = None,
) -> TrainedVerifier:
    """Return a process-wide singleton TrainedVerifier instance.

    Threshold and model_id are pinned by the first caller; later calls
    that pass different values get a fresh instance (not cached).
    """
    global _singleton
    if (
        _singleton is not None
        and _singleton.threshold == threshold
        and (model_id is None or _singleton.model_id == model_id)
    ):
        return _singleton
    instance = TrainedVerifier(model_id=model_id, threshold=threshold)
    if _singleton is None:
        _singleton = instance
    return instance
