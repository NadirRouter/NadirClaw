"""Tests for nadirclaw.laya_classifier (opt-in Laya typed-decisions classifier).

No test loads the real ~843 MB weights: a fake ``laya`` module stands in for the
package. Covers the risk rule, the pinned rubric, the analyze() result, the
fail-closed paths, and get_classifier() dispatch.
"""

import hashlib
import json
import sys
import types

import pytest


class _FakeAgent:
    def __init__(self, probs=(0.1, 0.3, 0.6), boom=False):
        self.probs = probs
        self.boom = boom
        self.calls = []

    def predict(self, text, questions):
        self.calls.append((text, questions))
        if self.boom and text != "warm-up prompt":
            raise RuntimeError("simulated inference failure")
        return {"answers": {"tier_score": {"probabilities": {str(i): p for i, p in enumerate(self.probs)}}}}


@pytest.fixture
def fake_laya(monkeypatch, tmp_path):
    """Install a fake `laya` package and a local snapshot path (no download)."""
    agent = _FakeAgent()
    module = types.SimpleNamespace(__version__="0.3.20", load=lambda path, device=None: agent)
    monkeypatch.setitem(sys.modules, "laya", module)
    monkeypatch.setenv("NADIRCLAW_LAYA_PATH", str(tmp_path))
    monkeypatch.delenv("NADIRCLAW_LAYA_CUTOFF", raising=False)
    monkeypatch.delenv("NADIRCLAW_MID_MODEL", raising=False)
    import nadirclaw.classifier as c
    import nadirclaw.laya_classifier as lc
    lc._warned_fallback = False
    c._active_classifier = None
    yield module, agent
    c._active_classifier = None


class TestRiskTier:
    def test_three_tiers(self):
        from nadirclaw.laya_classifier import risk_tier
        assert risk_tier((0.1, 0.3, 0.6), 0.4, True) == "complex"
        assert risk_tier((0.5, 0.3, 0.2), 0.4, True) == "mid"  # 0.3 + 0.2 >= 0.4
        assert risk_tier((0.7, 0.2, 0.1), 0.4, True) == "simple"

    def test_boundary_escalates(self):
        from nadirclaw.laya_classifier import risk_tier
        assert risk_tier((0.6, 0.0, 0.4), 0.4, True) == "complex"
        assert risk_tier((0.6, 0.2, 0.2), 0.4, True) == "mid"

    def test_no_mid_tier_escalates_to_complex(self):
        from nadirclaw.laya_classifier import risk_tier
        assert risk_tier((0.5, 0.3, 0.2), 0.4, False) == "complex"
        assert risk_tier((0.7, 0.2, 0.1), 0.4, False) == "simple"

    @pytest.mark.parametrize("p,cut", [
        ((0.5, 0.5), 0.4),
        ((0.2, 0.2, 0.2), 0.4),
        ((float("nan"), 0.5, 0.5), 0.4),
        ((0.2, 0.3, 0.5), 0.0),
        ((0.2, 0.3, 0.5), 1.5),
    ])
    def test_rejects_bad_input(self, p, cut):
        from nadirclaw.laya_classifier import risk_tier
        with pytest.raises(ValueError):
            risk_tier(p, cut, True)


def test_rubric_matches_the_evaluated_one():
    """The cutoff was fitted on this exact rubric; any edit must re-fit it."""
    from nadirclaw.laya_classifier import LAYA_REVISION, QUESTIONS
    digest = hashlib.sha256(json.dumps(QUESTIONS, sort_keys=True).encode()).hexdigest()
    assert digest == "1b56d50a1c2c783979de862b949990342c37dc9c2947751b66473eb45bf9a97e"
    assert LAYA_REVISION == "1a793eb568e6718f15941d08f85432581df534e3"


class TestCutoffSetting:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("NADIRCLAW_LAYA_CUTOFF", raising=False)
        from nadirclaw.settings import settings
        assert settings.LAYA_CUTOFF == 0.40

    @pytest.mark.parametrize("raw", ["0", "1.5", "nan", "abc"])
    def test_bad_value_raises(self, monkeypatch, raw):
        monkeypatch.setenv("NADIRCLAW_LAYA_CUTOFF", raw)
        from nadirclaw.settings import settings
        with pytest.raises(ValueError):
            settings.LAYA_CUTOFF


class TestAnalyze:
    @pytest.mark.asyncio
    async def test_complex_result(self, fake_laya, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_COMPLEX_MODEL", "anthropic/claude-opus-5")
        from nadirclaw.laya_classifier import LayaClassifier
        result = await LayaClassifier().analyze(text="design a scheduler")
        assert result["tier_name"] == "complex"
        assert result["tier"] == 3
        assert result["recommended_model"] == "anthropic/claude-opus-5"
        assert result["analyzer_type"] == "laya-v0.3.20"
        assert result["complexity_score"] == pytest.approx((0.3 + 2 * 0.6) / 2)
        assert result["confidence"] == pytest.approx(0.6)
        assert result["tier_probabilities"] == pytest.approx({"simple": 0.1, "mid": 0.3, "complex": 0.6})

    @pytest.mark.asyncio
    async def test_mid_needs_a_mid_model(self, fake_laya, monkeypatch):
        _, agent = fake_laya
        agent.probs = (0.5, 0.3, 0.2)
        from nadirclaw.laya_classifier import LayaClassifier
        clf = LayaClassifier()
        assert (await clf.analyze(text="x"))["tier_name"] == "complex"
        monkeypatch.setenv("NADIRCLAW_MID_MODEL", "openai/gpt-5-mini")
        result = await clf.analyze(text="x")
        assert result["tier_name"] == "mid"
        assert result["recommended_model"] == "openai/gpt-5-mini"

    @pytest.mark.asyncio
    async def test_input_is_truncated_to_the_evaluated_window(self, fake_laya):
        _, agent = fake_laya
        from nadirclaw.laya_classifier import LayaClassifier
        await LayaClassifier().analyze(text="a" * 10000)
        assert len(agent.calls[-1][0]) == 6000

    @pytest.mark.asyncio
    async def test_inference_error_serves_binary(self, fake_laya):
        _, agent = fake_laya
        agent.boom = True
        from nadirclaw.laya_classifier import LayaClassifier
        result = await LayaClassifier().analyze(text="What is 2+2?")
        assert result["analyzer_type"] == "laya-fallback-binary"
        assert result["tier_name"] in ("simple", "complex")

    @pytest.mark.asyncio
    async def test_bad_probabilities_serve_binary(self, fake_laya):
        _, agent = fake_laya
        from nadirclaw.laya_classifier import LayaClassifier
        clf = LayaClassifier()
        agent.probs = (0.9, 0.9, 0.9)
        result = await clf.analyze(text="What is 2+2?")
        assert result["analyzer_type"] == "laya-fallback-binary"


class TestDispatch:
    def test_get_classifier_returns_laya(self, fake_laya, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_COMPLEXITY_ANALYZER", "laya")
        from nadirclaw.classifier import get_classifier
        from nadirclaw.laya_classifier import LayaClassifier
        assert isinstance(get_classifier(), LayaClassifier)

    def test_missing_package_falls_back_to_binary(self, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_COMPLEXITY_ANALYZER", "laya")
        monkeypatch.setitem(sys.modules, "laya", None)  # import laya -> ImportError
        import nadirclaw.classifier as c
        c._active_classifier = None
        assert isinstance(c.get_classifier(), c.BinaryComplexityClassifier)
        c._active_classifier = None

    def test_other_laya_version_falls_back_to_binary(self, fake_laya, monkeypatch):
        module, _ = fake_laya
        module.__version__ = "0.4.30"
        monkeypatch.setenv("NADIRCLAW_COMPLEXITY_ANALYZER", "laya")
        import nadirclaw.classifier as c
        assert isinstance(c.get_classifier(), c.BinaryComplexityClassifier)

    def test_bad_cutoff_falls_back_to_binary(self, fake_laya, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_COMPLEXITY_ANALYZER", "laya")
        monkeypatch.setenv("NADIRCLAW_LAYA_CUTOFF", "2")
        import nadirclaw.classifier as c
        assert isinstance(c.get_classifier(), c.BinaryComplexityClassifier)
