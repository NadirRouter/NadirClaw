"""Continuous-score adapter for the bundled wide_deep_asym_v3 classifier.

The classifier emits a 3-way softmax over ``{simple, medium, complex}``.
The N-tier selector consumes a single continuous score in ``[0,1]``. The
adapter is the mapping that lets the same weights serve N=2, N=3, N=5,
... with no retraining.

Formula
=======

For classes ``c ∈ {0=simple, 1=medium, 2=complex}`` with probabilities
``p_c``::

    expected_class = sum(c * p_c)        # in [0, 2]
    score          = expected_class / 2  # in [0, 1]

This is the standard expected-class adapter: monotone in complexity,
exact for one-hot distributions (``simple → 0.0``, ``medium → 0.5``,
``complex → 1.0``), and well-behaved under both argmax and
cost-sensitive decoders since both only change which class index gets
mass — they do not change the expectation formula itself.

Trade-off documented in ``N_TIER_ARCHITECTURE.md §2.1 / §8``: this
loses native multi-class calibration info compared to a 5-way head
trained from scratch, but the calibration we'd gain isn't worth the
training pipeline tax for OSS users. Operators who want exact per-tier
calibration should plug in their own classifier and emit ``score``
directly.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple


def softmax_to_score(probs: Iterable[float]) -> float:
    """Map a 3-way softmax to a continuous score in [0, 1].

    ``probs`` is an iterable of three floats summing to ~1.0, in the
    order ``[P(simple), P(medium), P(complex)]``. Robust to off-by-epsilon
    rounding (we don't re-normalise; we just clamp the output).
    """
    p = list(probs)
    if len(p) != 3:
        raise ValueError(
            f"softmax_to_score expects 3 probabilities, got {len(p)}"
        )
    expected_class = 0.0 * p[0] + 1.0 * p[1] + 2.0 * p[2]
    score = expected_class / 2.0
    # Defensive clamp: a slightly-negative input from numpy float32 noise
    # would otherwise propagate to the selector.
    return max(0.0, min(1.0, float(score)))


def probs_dict_to_score(probs: Dict[str, float]) -> float:
    """Same as :func:`softmax_to_score` but takes the dict shape used by
    ``WideDeepClassifier.classify(...).probabilities``.
    """
    try:
        ordered: Tuple[float, float, float] = (
            float(probs["simple"]),
            float(probs["medium"]),
            float(probs["complex"]),
        )
    except KeyError as e:
        raise ValueError(
            "probs_dict_to_score requires keys "
            "{'simple', 'medium', 'complex'}; missing %s" % e
        ) from None
    return softmax_to_score(ordered)


__all__ = ["probs_dict_to_score", "softmax_to_score"]
