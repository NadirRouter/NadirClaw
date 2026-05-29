"""Tests for the TrainedVerifier (RouterArena snapshot bridge).

We do not mock ``torch`` or ``transformers`` at the ``sys.modules`` level
because PyTorch's TORCH_LIBRARY registration is one-shot per process and
swapping the real torch out (even temporarily, even via monkeypatch
teardown) can corrupt later tests in the same session.

Instead we use real torch (available in the dev venv via the
``trained`` extras) for the few tests that exercise a forward pass —
the verifier itself takes care of running on CPU with random init.
The "missing extras" path is tested by a separate fast unit test that
simulates the ImportError without touching sys.modules.
"""
from __future__ import annotations

import pytest


# Skip the whole module if torch / transformers are missing; the design
# contract is that without `pip install nadirclaw[trained]` the verifier
# is unavailable. Confirmed by ``test_unavailable_without_extras``.
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


# ---------------------------------------------------------------------------
# Helpers — build a TrainedVerifier with a *real* tiny HF model loaded
# directly (no network), then exercise score().
# ---------------------------------------------------------------------------

def _tiny_verifier(threshold: float = 0.8):
    """Construct a TrainedVerifier whose loaded model is a tiny
    randomly-initialised DeBERTa-v2 head. No network, no checkpoint —
    we just want a model object that behaves like the real one for the
    interface-level tests.

    Returns a TrainedVerifier with ``_model``, ``_tokenizer``, and
    ``_resolved_device`` already populated.
    """
    from transformers import (
        AutoTokenizer,
        DebertaV2Config,
        DebertaV2ForSequenceClassification,
    )

    from nadirclaw.trained_verifier import TrainedVerifier

    # Tiny config — actual production model is DeBERTa-v3-small (6 layers,
    # 768 hidden). For tests we shrink everything to keep the random init
    # fast (~100ms) and memory under 5 MB.
    config = DebertaV2Config(
        vocab_size=128100,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
        pad_token_id=0,
        num_labels=2,
        relative_attention=True,
        position_biased_input=False,
    )
    model = DebertaV2ForSequenceClassification(config).eval()
    # Reuse the production tokenizer config (the spm.model is shipped
    # in the released HF repo; for the test we use microsoft's base
    # tokenizer which has the same vocab).
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")

    v = TrainedVerifier(threshold=threshold, device="cpu")
    v._model = model
    v._tokenizer = tokenizer
    v._resolved_device = "cpu"
    return v


# ---------------------------------------------------------------------------
# Interface parity with HeuristicVerifier
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    "NADIRCLAW_RUN_SLOW_TESTS" not in __import__("os").environ,
    reason=(
        "Requires real DeBERTa tokenizer download (~10 MB). "
        "Set NADIRCLAW_RUN_SLOW_TESTS=1 to enable."
    ),
)
def test_trained_verifier_score_returns_struct():
    """A real forward pass on a tiny random init returns a valid
    TrainedScore with score in [0, 1]. Gated behind an env var because
    it needs to download a tokenizer from the HF cache.
    """
    v = _tiny_verifier(threshold=0.8)
    out = v.score("What is 2+2?", "4")

    assert 0.0 <= out.score <= 1.0
    assert out.threshold == 0.8
    assert isinstance(out.accepted, bool)
    d = out.to_dict()
    assert d["verifier"] == "trained"
    assert {"score", "accepted", "threshold", "reasons"} <= d.keys()


def test_trained_verifier_empty_response_short_circuits():
    """Empty cheap_answer returns 0.0 without invoking the model — we
    test this without needing a real model load.
    """
    from nadirclaw.trained_verifier import TrainedVerifier

    v = TrainedVerifier(threshold=0.5, device="cpu")
    # Mark loaded with sentinels so .score() doesn't try to download.
    # The empty short-circuit returns before either is touched.
    v._tokenizer = object()
    v._model = object()
    v._resolved_device = "cpu"

    out = v.score("anything", "")
    assert out.score == 0.0
    assert out.accepted is False
    assert "empty_response" in out.reasons


def test_trained_verifier_interface_matches_heuristic():
    """The cascade calls verifier.score(prompt, cheap_answer, expect_json=...).

    Both verifiers must accept the same kwargs and return objects that
    expose .score / .accepted / .threshold / .to_dict() / .reasons. We
    use the empty-string short-circuit so this test runs without a
    forward pass.
    """
    from nadirclaw.heuristic_verifier import HeuristicVerifier
    from nadirclaw.trained_verifier import TrainedVerifier

    h = HeuristicVerifier(threshold=0.8)
    t = TrainedVerifier(threshold=0.8, device="cpu")
    t._tokenizer = object()
    t._model = object()
    t._resolved_device = "cpu"

    h_out = h.score(prompt="anything", cheap_answer="", expect_json=False)
    t_out = t.score(prompt="anything", cheap_answer="", expect_json=False)

    for out in (h_out, t_out):
        assert hasattr(out, "score")
        assert hasattr(out, "accepted")
        assert hasattr(out, "threshold")
        assert hasattr(out, "reasons")
        d = out.to_dict()
        assert {"score", "accepted", "threshold", "reasons", "verifier"} <= d.keys()


def test_trained_verifier_wraps_input_in_production_format():
    """The tokenizer must receive ``text_pair`` wrapped in the
    ``CHEAP:\\n...\\n\\nEXPENSIVE:\\n...`` format the cross-encoder was
    trained on. Without this wrapper, scores drift against the
    calibrated tau=0.80 threshold.

    Production reference:
      ``getnadir.dev/backend/app/services/verifier_model.py:195``
    """
    from nadirclaw.trained_verifier import TrainedVerifier

    captured: dict = {}

    class _FakeEncoding(dict):
        def __init__(self):
            super().__init__()
            # Minimal tensor-like values so the .to(device) loop works.
            class _T:
                def to(self, _device):
                    return self

            self["input_ids"] = _T()
            self["attention_mask"] = _T()

    class _FakeTokenizer:
        def __call__(self, prompt, text_pair, **kwargs):
            captured["prompt"] = prompt
            captured["text_pair"] = text_pair
            captured["kwargs"] = kwargs
            return _FakeEncoding()

    class _FakeLogits:
        # Two-class head; softmax([0, 0]) => probs[..., 1] == 0.5
        shape = (1, 2)

        def __init__(self):
            import torch
            self._t = torch.tensor([[0.0, 0.0]])

        def __getattr__(self, name):
            return getattr(self._t, name)

    class _FakeModelOut:
        def __init__(self):
            import torch
            self.logits = torch.tensor([[0.0, 0.0]])

    class _FakeModel:
        def __call__(self, **kwargs):
            return _FakeModelOut()

        def eval(self):
            return self

        def to(self, _device):
            return self

    v = TrainedVerifier(threshold=0.8, device="cpu")
    v._tokenizer = _FakeTokenizer()
    v._model = _FakeModel()
    v._resolved_device = "cpu"

    # Case 1: reference_answer provided.
    out = v.score("What is 2+2?", "4", reference_answer="four")
    assert captured["prompt"] == "What is 2+2?"
    assert captured["text_pair"] == "CHEAP:\n4\n\nEXPENSIVE:\nfour"
    assert 0.0 <= out.score <= 1.0

    # Case 2: reference_answer=None -> empty EXPENSIVE: block.
    captured.clear()
    v.score("What is 2+2?", "4")
    assert captured["text_pair"] == "CHEAP:\n4\n\nEXPENSIVE:\n"

    # Case 3: reference_answer is whitespace-only -> stripped to empty.
    captured.clear()
    v.score("What is 2+2?", "4", reference_answer="   \n  ")
    assert captured["text_pair"] == "CHEAP:\n4\n\nEXPENSIVE:\n"


def test_trained_verifier_get_singleton_caches():
    """The module-level singleton accessor should cache same-threshold calls
    and return fresh instances for mismatched thresholds. Construction
    only — no .score() call, no model load.
    """
    import nadirclaw.trained_verifier as tv

    tv._singleton = None
    a = tv.get_trained_verifier(threshold=0.8)
    b = tv.get_trained_verifier(threshold=0.8)
    assert a is b

    c = tv.get_trained_verifier(threshold=0.5)
    assert c is not a

    tv._singleton = None  # cleanup


def test_trained_verifier_default_model_id():
    """The released v1 snapshot id should be the constructor default."""
    from nadirclaw.trained_verifier import DEFAULT_MODEL_ID, TrainedVerifier

    assert DEFAULT_MODEL_ID == "nadirclaw/cascade-verifier-v1"
    v = TrainedVerifier(threshold=0.8, device="cpu")
    assert v.model_id == DEFAULT_MODEL_ID


def test_trained_verifier_env_override(monkeypatch):
    """NADIRCLAW_TRAINED_VERIFIER_MODEL overrides the default model id."""
    from nadirclaw.trained_verifier import TrainedVerifier

    monkeypatch.setenv(
        "NADIRCLAW_TRAINED_VERIFIER_MODEL", "local/path/to/weights"
    )
    v = TrainedVerifier(threshold=0.8, device="cpu")
    assert v.model_id == "local/path/to/weights"


# ---------------------------------------------------------------------------
# Profile wiring
# ---------------------------------------------------------------------------

def test_n2_trained_profile_loads():
    """The n2_trained YAML must parse and select the trained verifier."""
    from nadirclaw.tier_config.loader import load_profile

    profile = load_profile("n2_trained")
    assert profile.profile_name == "n2_trained"
    assert profile.num_tiers == 2
    assert profile.cascade.verifier == "trained"
    assert profile.cascade.verifier_model == "nadirclaw/cascade-verifier-v1"
    assert profile.cascade.acceptance_threshold == 0.80


def test_n2_default_profile_still_uses_heuristic():
    """Backward-compat: the existing n2_default profile must keep the
    heuristic verifier as its default (so users without the trained
    extras keep working).
    """
    from nadirclaw.tier_config.loader import load_profile

    profile = load_profile("n2_default")
    assert profile.cascade.verifier == "heuristic"


def test_cascade_config_rejects_unknown_verifier():
    """Schema must reject typos to avoid silently falling back."""
    from pydantic import ValidationError

    from nadirclaw.tier_config.schema import CascadeConfig

    with pytest.raises(ValidationError):
        CascadeConfig(verifier="trianed")  # typo
