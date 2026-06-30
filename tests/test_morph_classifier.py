"""Tests for nadirclaw.morph_classifier — Morph router classifier (issue #68).

No test hits the live Morph API; the HTTP layer is monkeypatched. Covers the
success path, the fail-closed-to-binary path, response parsing, caching, and
the get_classifier() dispatch + degradation when no key is set.
"""

import json
import urllib.error

import pytest


def _fake_response(payload: dict):
    """Build a context-manager stand-in for urllib.request.urlopen()."""
    class _Resp:
        def __enter__(self_):
            return self_
        def __exit__(self_, *a):
            return False
        def read(self_):
            return json.dumps(payload).encode("utf-8")
    return _Resp()


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Each test starts with a clean key, base, and warn-once guard."""
    monkeypatch.setenv("MORPH_API_KEY", "test-key")
    monkeypatch.delenv("MORPH_API_BASE", raising=False)
    monkeypatch.delenv("MORPH_TIMEOUT_MS", raising=False)
    import nadirclaw.morph_classifier as m
    m._warned_fallback = False
    import nadirclaw.classifier as c
    c._active_classifier = None
    yield
    c._active_classifier = None


class TestParseResponse:
    def test_difficulty_and_confidence(self):
        from nadirclaw.morph_classifier import MorphRouterClassifier
        diff, conf = MorphRouterClassifier._parse_response(
            {"difficulty": "hard", "confidence": 0.9}
        )
        assert diff == "hard"
        assert conf == pytest.approx(0.9)

    def test_confidence_from_ambiguity(self):
        from nadirclaw.morph_classifier import MorphRouterClassifier
        diff, conf = MorphRouterClassifier._parse_response(
            {"difficulty": "easy", "ambiguity": 0.25}
        )
        assert diff == "easy"
        assert conf == pytest.approx(0.75)

    def test_nested_classification_key(self):
        from nadirclaw.morph_classifier import MorphRouterClassifier
        diff, conf = MorphRouterClassifier._parse_response(
            {"classification": {"difficulty": "MEDIUM"}}
        )
        assert diff == "medium"
        assert conf == pytest.approx(1.0)

    def test_missing_difficulty_raises(self):
        from nadirclaw.morph_classifier import MorphRouterClassifier
        with pytest.raises(ValueError):
            MorphRouterClassifier._parse_response({"domain": "code"})

    def test_unknown_difficulty_raises(self):
        from nadirclaw.morph_classifier import MorphRouterClassifier
        with pytest.raises(ValueError):
            MorphRouterClassifier._parse_response({"difficulty": "trivial"})


class TestAnalyzeSuccess:
    @pytest.mark.asyncio
    async def test_hard_routes_complex(self, monkeypatch):
        import nadirclaw.morph_classifier as m
        monkeypatch.setattr(
            m.urllib.request, "urlopen",
            lambda *a, **k: _fake_response({"difficulty": "hard", "confidence": 0.95}),
        )
        clf = m.MorphRouterClassifier()
        result = await clf.analyze(text="Design a consensus protocol")
        assert result["tier_name"] == "complex"
        assert result["analyzer_type"].startswith("morph-v")
        assert result["morph_difficulty"] == "hard"
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_easy_routes_simple(self, monkeypatch):
        import nadirclaw.morph_classifier as m
        monkeypatch.setattr(
            m.urllib.request, "urlopen",
            lambda *a, **k: _fake_response({"difficulty": "easy", "ambiguity": 0.1}),
        )
        clf = m.MorphRouterClassifier()
        result = await clf.analyze(text="What is 2+2?")
        assert result["tier_name"] == "simple"

    @pytest.mark.asyncio
    async def test_response_is_cached(self, monkeypatch):
        import nadirclaw.morph_classifier as m
        calls = {"n": 0}

        def _urlopen(*a, **k):
            calls["n"] += 1
            return _fake_response({"difficulty": "hard", "confidence": 0.9})

        monkeypatch.setattr(m.urllib.request, "urlopen", _urlopen)
        clf = m.MorphRouterClassifier()
        await clf.analyze(text="same prompt")
        await clf.analyze(text="same prompt")
        assert calls["n"] == 1  # second call served from cache


class TestFailClosed:
    @pytest.mark.asyncio
    async def test_http_error_falls_back_to_binary(self, monkeypatch):
        import nadirclaw.morph_classifier as m

        def _boom(*a, **k):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(m.urllib.request, "urlopen", _boom)
        clf = m.MorphRouterClassifier()
        result = await clf.analyze(text="What is Python?")
        # Degrades, doesn't crash, and is clearly labelled as the fallback.
        assert result["analyzer_type"] == "morph-fallback-binary"
        assert result["tier_name"] in ("simple", "mid", "complex")

    @pytest.mark.asyncio
    async def test_timeout_falls_back_to_binary(self, monkeypatch):
        import nadirclaw.morph_classifier as m

        def _slow(*a, **k):
            raise TimeoutError("timed out")

        monkeypatch.setattr(m.urllib.request, "urlopen", _slow)
        clf = m.MorphRouterClassifier()
        result = await clf.analyze(text="What is Python?")
        assert result["analyzer_type"] == "morph-fallback-binary"

    @pytest.mark.asyncio
    async def test_bad_json_falls_back_to_binary(self, monkeypatch):
        import nadirclaw.morph_classifier as m

        class _BadResp:
            def __enter__(self_):
                return self_
            def __exit__(self_, *a):
                return False
            def read(self_):
                return b"not json"

        monkeypatch.setattr(m.urllib.request, "urlopen", lambda *a, **k: _BadResp())
        clf = m.MorphRouterClassifier()
        result = await clf.analyze(text="hi")
        assert result["analyzer_type"] == "morph-fallback-binary"


class TestDispatch:
    def test_missing_key_degrades_to_binary(self, monkeypatch):
        monkeypatch.delenv("MORPH_API_KEY", raising=False)
        monkeypatch.setenv("NADIRCLAW_COMPLEXITY_ANALYZER", "morph")
        import nadirclaw.classifier as c
        c._active_classifier = None
        from nadirclaw.classifier import get_classifier, BinaryComplexityClassifier
        clf = get_classifier()
        assert isinstance(clf, BinaryComplexityClassifier)
        c._active_classifier = None

    def test_with_key_selects_morph(self, monkeypatch):
        monkeypatch.setenv("MORPH_API_KEY", "test-key")
        monkeypatch.setenv("NADIRCLAW_COMPLEXITY_ANALYZER", "morph")
        import nadirclaw.classifier as c
        c._active_classifier = None
        from nadirclaw.classifier import get_classifier
        from nadirclaw.morph_classifier import MorphRouterClassifier
        clf = get_classifier()
        assert isinstance(clf, MorphRouterClassifier)
        c._active_classifier = None
