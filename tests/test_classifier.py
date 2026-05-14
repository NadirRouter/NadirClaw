"""Tests for nadirclaw.classifier — binary complexity classification."""

import pytest


class TestBinaryClassifier:
    @pytest.fixture(autouse=True)
    def classifier(self):
        from nadirclaw.classifier import BinaryComplexityClassifier
        self.clf = BinaryComplexityClassifier()

    def test_simple_prompt(self):
        is_complex, confidence = self.clf.classify("What is 2+2?")
        assert is_complex is False
        assert 0.0 <= confidence <= 1.0

    def test_complex_prompt(self):
        is_complex, confidence = self.clf.classify(
            "Design a distributed database with sharding, replication, "
            "and consensus protocol for high availability"
        )
        assert is_complex is True
        assert 0.0 <= confidence <= 1.0

    def test_confidence_score_range(self):
        """Confidence-to-score should map to [0, 1]."""
        score_simple = self.clf._confidence_to_score(False, 0.5)
        score_complex = self.clf._confidence_to_score(True, 0.5)
        assert 0.0 <= score_simple <= 0.5
        assert 0.5 <= score_complex <= 1.0

    def test_analyze_sync_returns_expected_keys(self):
        result = self.clf._analyze_sync("Hello world")
        expected_keys = {
            "recommended_model", "confidence", "complexity_score",
            "tier_name", "reasoning", "analyzer_type",
        }
        assert expected_keys.issubset(result.keys())
        assert result["analyzer_type"] == "binary"

    @pytest.mark.asyncio
    async def test_analyze_async(self):
        result = await self.clf.analyze(text="What is Python?")
        assert result["tier_name"] in ("simple", "complex")


class TestClassifierFactory:
    """get_classifier() dispatches on NADIRCLAW_COMPLEXITY_ANALYZER."""

    def _reset_singleton(self):
        import nadirclaw.classifier as c
        c._active_classifier = None

    def test_default_is_binary(self, monkeypatch):
        monkeypatch.delenv("NADIRCLAW_COMPLEXITY_ANALYZER", raising=False)
        self._reset_singleton()
        from nadirclaw.classifier import get_classifier, BinaryComplexityClassifier
        clf = get_classifier()
        assert isinstance(clf, BinaryComplexityClassifier)

    def test_explicit_binary(self, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_COMPLEXITY_ANALYZER", "binary")
        self._reset_singleton()
        from nadirclaw.classifier import get_classifier, BinaryComplexityClassifier
        assert isinstance(get_classifier(), BinaryComplexityClassifier)

    def test_distilbert_load_failure_falls_back_to_binary(self, monkeypatch):
        """If the DistilBERT artifact can't load, we degrade to binary, not crash."""
        monkeypatch.setenv("NADIRCLAW_COMPLEXITY_ANALYZER", "distilbert")
        self._reset_singleton()

        import builtins
        real_import = builtins.__import__

        def _boom(name, *args, **kwargs):
            if name == "nadirclaw.distilbert_classifier":
                raise RuntimeError("simulated missing model artifact")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _boom)
        from nadirclaw.classifier import get_classifier, BinaryComplexityClassifier
        clf = get_classifier()
        assert isinstance(clf, BinaryComplexityClassifier)
        self._reset_singleton()

    def test_factory_singleton_is_cached(self, monkeypatch):
        monkeypatch.delenv("NADIRCLAW_COMPLEXITY_ANALYZER", raising=False)
        self._reset_singleton()
        from nadirclaw.classifier import get_classifier
        assert get_classifier() is get_classifier()
