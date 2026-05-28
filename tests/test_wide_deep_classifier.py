"""Integration tests for the bundled wide-and-deep classifier.

These tests load the actual `wide_deep_asym_v3.pt` / `wide_deep_sym_v3.pt`
weights shipped in `nadirclaw/models/` and run a real forward pass on a
handful of representative prompts. They are heavier than the other unit
tests (BGE-base load is ~5 s the first time the encoder is built) but
are cached as a process-level singleton, so the second-and-later
classify calls in the same pytest session are sub-100 ms each.

Skipped automatically if torch or sentence-transformers are not
installed in the running environment, so CI on a `pip install
nadirclaw[dev]` minimal venv still passes.
"""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")
pytest.importorskip("numpy")

from nadirclaw.wide_deep_classifier import (  # noqa: E402  (after importorskip)
    ClassificationResult,
    WideDeepClassifier,
    bundled_model_paths,
    get_wide_deep_classifier,
)


# ---------------------------------------------------------------------------
# Package data
# ---------------------------------------------------------------------------
def test_bundled_paths_exist():
    """Both checkpoints must be present on disk after a normal install."""
    paths = bundled_model_paths()
    assert set(paths) == {"asym", "symmetric"}
    for variant, path in paths.items():
        assert os.path.exists(path), (
            f"Bundled {variant!r} checkpoint missing at {path}. "
            f"Check pyproject.toml [tool.setuptools.package-data]."
        )
        # Sanity check: each shipped checkpoint is roughly 900 KB.
        size = os.path.getsize(path)
        assert 100_000 < size < 5_000_000, (
            f"Checkpoint {variant!r} at {path} has suspicious size: {size} bytes"
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def asym_clf() -> WideDeepClassifier:
    return get_wide_deep_classifier(
        checkpoint_variant="asym",
        decision_rule="cost_sensitive",
        cost_lambda=3.0,
    )


def test_loader_metadata(asym_clf: WideDeepClassifier):
    assert asym_clf.checkpoint_variant == "asym"
    assert asym_clf.decision_rule == "cost_sensitive"
    assert asym_clf._cfg["struct_dim"] == 33
    assert asym_clf._cfg["emb_dim"] == 768
    assert asym_clf._encoder_name == "BAAI/bge-base-en-v1.5"
    assert asym_clf.ANALYZER_VERSION == "wide_deep_asym_v3"


def test_classify_returns_valid_result(asym_clf: WideDeepClassifier):
    result = asym_clf.classify("What is 2 + 2?")
    assert isinstance(result, ClassificationResult)
    assert result.tier in {"simple", "medium", "complex"}
    assert result.tier_num in {1, 2, 3}
    assert 0.0 <= result.confidence <= 1.0
    probs = result.probabilities
    assert set(probs) == {"simple", "medium", "complex"}
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    # Softmax sums to ~1.
    assert abs(sum(probs.values()) - 1.0) < 1e-3
    assert result.classifier_version == "wide_deep_asym_v3"
    assert result.latency_ms >= 0


def test_classify_tuple_legacy_shape(asym_clf: WideDeepClassifier):
    tier, conf, info = asym_clf.classify_tuple("Explain how DNS resolution works")
    assert tier in {"simple", "medium", "complex"}
    assert 0.0 <= conf <= 1.0
    assert info["classifier_version"] == "wide_deep_asym_v3"
    assert "probabilities" in info


def test_singleton_cached_per_variant():
    a1 = get_wide_deep_classifier(checkpoint_variant="asym")
    a2 = get_wide_deep_classifier(checkpoint_variant="asym")
    assert a1 is a2  # same singleton

    s1 = get_wide_deep_classifier(checkpoint_variant="symmetric")
    assert s1 is not a1  # different singleton per variant
    assert s1.checkpoint_variant == "symmetric"


def test_decoder_hot_swap_does_not_reload():
    """Changing decision_rule should mutate the cached instance, not rebuild."""
    a1 = get_wide_deep_classifier(checkpoint_variant="asym", decision_rule="argmax")
    a2 = get_wide_deep_classifier(
        checkpoint_variant="asym", decision_rule="cost_sensitive", cost_lambda=20.0
    )
    assert a1 is a2
    assert a2.decision_rule == "cost_sensitive"
    assert a2.cost_lambda == 20.0
    assert a2._cost is not None  # cost matrix populated


def test_cost_sensitive_lambda20_never_predicts_simple_on_complex_prompt(
    asym_clf: WideDeepClassifier,
):
    """Sanity: a heavy code/architecture prompt should not land on `simple`
    under cost_sensitive λ=20 decoding. This is the safety bias the asym
    head was trained for."""
    clf = get_wide_deep_classifier(
        checkpoint_variant="asym",
        decision_rule="cost_sensitive",
        cost_lambda=20.0,
    )
    result = clf.classify(
        "Design a distributed rate limiter that survives a leader failover "
        "without dropping in-flight tokens. Walk through the consistency model, "
        "the failure cases, and how you would prove correctness."
    )
    assert result.tier in {"medium", "complex"}


def test_symmetric_variant_loads_independently():
    clf = WideDeepClassifier(checkpoint_variant="symmetric", decision_rule="argmax")
    assert clf.checkpoint_variant == "symmetric"
    # Forward pass works.
    result = clf.classify("hi")
    assert result.tier in {"simple", "medium", "complex"}


def test_invalid_args():
    with pytest.raises(ValueError):
        WideDeepClassifier(checkpoint_variant="badvariant")
    with pytest.raises(ValueError):
        WideDeepClassifier(decision_rule="badrule")


def test_missing_checkpoint_path():
    with pytest.raises(FileNotFoundError):
        WideDeepClassifier(model_path="/nonexistent/no_such_checkpoint.pt")
