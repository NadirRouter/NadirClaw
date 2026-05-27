"""Tests for the benchmark-contamination audit utility.

Exercises the public surface of `verifier.contamination_audit`: the
hashing convention, the three supported file formats (.jsonl / .json
/ .txt), the audit return-shape, the CLI exit codes, and the prompt-key
override.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from verifier.contamination_audit import (
    main,
    normalize_and_hash,
    run_audit,
)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def test_hash_is_nfc_casefold_strip_sha256():
    # Same prompt with different casing + whitespace + accent encoding
    # collapses to the same hash.
    a = normalize_and_hash(" Café ")
    b = normalize_and_hash("café")
    c = normalize_and_hash("CAFÉ")
    assert a == b == c
    # And distinct prompts produce distinct hashes.
    assert normalize_and_hash("foo") != normalize_and_hash("bar")


def test_hash_rejects_non_str():
    with pytest.raises(TypeError):
        normalize_and_hash(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Format readers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, prompts):
    path.write_text(
        "\n".join(json.dumps({"prompt": p}) for p in prompts) + "\n",
        encoding="utf-8",
    )


def _write_json_list(path: Path, prompts, key="prompt"):
    path.write_text(
        json.dumps([{key: p} for p in prompts]),
        encoding="utf-8",
    )


def _write_txt(path: Path, prompts):
    path.write_text("\n".join(prompts) + "\n", encoding="utf-8")


def test_run_audit_zero_overlap(tmp_path):
    bench = tmp_path / "bench.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    out = tmp_path / "report.json"
    _write_jsonl(bench, ["alpha question", "beta question"])
    _write_jsonl(corpus, ["unrelated training prompt"])
    report = run_audit([bench], [corpus], report_out=out)
    assert report["overlap_count"] == 0
    assert report["benchmark_unique_hashes"] == 2
    assert report["corpus_unique_hashes"] == 1
    # File is written.
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["overlap_count"] == 0


def test_run_audit_detects_overlap(tmp_path):
    bench = tmp_path / "bench.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(bench, ["alpha question", "beta question"])
    _write_jsonl(corpus, [" Alpha Question ", "unrelated"])
    report = run_audit([bench], [corpus], report_out=None)
    # Normalisation collapses casing/whitespace → 1 overlap.
    assert report["overlap_count"] == 1
    assert report["overlap_examples"][0]["benchmark_file"].endswith("bench.jsonl")


def test_run_audit_supports_json_and_txt(tmp_path):
    bench = tmp_path / "bench.json"
    corpus = tmp_path / "corpus.txt"
    _write_json_list(bench, ["foo prompt"])
    _write_txt(corpus, ["foo prompt", "ignored"])
    report = run_audit([bench], [corpus], report_out=None)
    assert report["overlap_count"] == 1


def test_run_audit_prompt_key_override(tmp_path):
    """When both files use a non-standard key, --prompt-key picks it up."""
    bench = tmp_path / "bench.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    bench.write_text(
        json.dumps({"user_input": "Tell me about X"}) + "\n", encoding="utf-8"
    )
    corpus.write_text(
        json.dumps({"user_input": "tell me about x"}) + "\n", encoding="utf-8"
    )
    # Without override, "user_input" is not in the default list → 0 hashes.
    report_default = run_audit([bench], [corpus], report_out=None)
    assert report_default["benchmark_unique_hashes"] == 0
    # With override, it picks up the field and finds the overlap.
    report_override = run_audit(
        [bench], [corpus], report_out=None, prompt_key="user_input"
    )
    assert report_override["benchmark_unique_hashes"] == 1
    assert report_override["overlap_count"] == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_exit_zero_on_clean_audit(tmp_path):
    bench = tmp_path / "bench.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(bench, ["a"])
    _write_jsonl(corpus, ["b"])
    rc = main([
        "--benchmark", str(bench),
        "--corpus", str(corpus),
    ])
    assert rc == 0


def test_cli_exit_two_on_overlap(tmp_path):
    bench = tmp_path / "bench.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(bench, ["shared prompt"])
    _write_jsonl(corpus, ["shared prompt"])
    rc = main([
        "--benchmark", str(bench),
        "--corpus", str(corpus),
    ])
    assert rc == 2


def test_cli_exit_one_on_missing_input(tmp_path):
    bench = tmp_path / "does_not_exist.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, ["a"])
    rc = main([
        "--benchmark", str(bench),
        "--corpus", str(corpus),
    ])
    assert rc == 1
