"""Tests for the classifier input cleaner (NADIRCLAW_CLASSIFIER_STRIP_PATTERNS).

Locks in the contract for ``_strip_classifier_input`` / ``_compile_strip_regex``
in ``nadirclaw.server``:

  (a) unset env var            -> identity (no stripping)
  (b) a configured pattern     -> the matched envelope is removed
  (c) an invalid regex         -> no-op + warning, never a crash
  (d) an over-broad pattern    -> falls back to the original prompt rather than
                                  emptying the classifier input

The module-level ``_strip_regex`` cache is reset before each case so the
pattern is recompiled from the (monkeypatched) environment.
"""

import pytest

import nadirclaw.server as server


@pytest.fixture(autouse=True)
def _reset_strip_cache(monkeypatch):
    """Clear the compiled-regex cache and env var before/after each test."""
    monkeypatch.delenv("NADIRCLAW_CLASSIFIER_STRIP_PATTERNS", raising=False)
    server._strip_regex = None
    yield
    server._strip_regex = None


def _set_pattern(monkeypatch, pattern: str):
    monkeypatch.setenv("NADIRCLAW_CLASSIFIER_STRIP_PATTERNS", pattern)
    server._strip_regex = None  # force recompile from the new env value


def test_unset_is_identity(monkeypatch):
    text = "<envelope>meta</envelope>What is the capital of France?"
    assert server._strip_classifier_input(text) == text


def test_empty_text_is_returned_unchanged(monkeypatch):
    _set_pattern(monkeypatch, r"<envelope>.*?</envelope>")
    assert server._strip_classifier_input("") == ""


def test_pattern_strips_envelope(monkeypatch):
    _set_pattern(monkeypatch, r"<envelope>.*?</envelope>")
    text = "<envelope>memory: 42 facts</envelope>Summarize this."
    assert server._strip_classifier_input(text) == "Summarize this."


def test_pattern_strips_across_newlines_dotall(monkeypatch):
    # re.DOTALL is applied internally, so '.' spans newlines.
    _set_pattern(monkeypatch, r"\[system note:.*?\]")
    text = "[system note:\nremember the user prefers\nterse answers]Hi"
    assert server._strip_classifier_input(text) == "Hi"


def test_invalid_regex_is_noop_and_warns(monkeypatch, caplog):
    _set_pattern(monkeypatch, r"<envelope>(unclosed")  # invalid: unbalanced (
    text = "<envelope>(unclosed keep me intact"
    with caplog.at_level("WARNING"):
        out = server._strip_classifier_input(text)
    assert out == text  # never crashes, returns input untouched
    assert any("NADIRCLAW_CLASSIFIER_STRIP_PATTERNS" in r.message for r in caplog.records)


def test_overbroad_pattern_falls_back_to_original(monkeypatch):
    # A greedy pattern that consumes the whole prompt must not empty the
    # classifier input (which would route everything to the cheapest tier).
    _set_pattern(monkeypatch, r".*")
    text = "Implement a distributed consensus algorithm."
    assert server._strip_classifier_input(text) == text


def test_partial_strip_leaving_content_is_kept(monkeypatch):
    # When stripping still leaves real content, that content is returned.
    _set_pattern(monkeypatch, r"<sys>.*?</sys>")
    text = "<sys>tooling</sys>   real question   <sys>more</sys>"
    assert server._strip_classifier_input(text) == "real question"
