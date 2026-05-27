"""Benchmark-contamination audit utility for NadirClaw / Nadir verifier work.

Reproducibility tool: given (a) one or more benchmark prompt files and
(b) one or more training-corpus files, this script computes the
NFC-normalised SHA-256 of every prompt and writes a JSON report of any
overlapping hashes. Zero overlap is the hard FAIL gate that "no held-out
leakage" claims for Nadir's RouterBench and RouterArena results rest on;
this script is what makes those claims independently reproducible.

The script is intentionally dependency-light: stdlib only.

Hashing convention
==================

    sha256( prompt.strip().casefold().encode("utf-8") )

NFC normalisation is applied before casefold. This matches the
convention used by the upstream RouterArena reference harness and is
what we run internally for the Nadir verifier corpus, so hashes are
portable across the audit boundary.

Supported input formats
=======================

For both ``--benchmark`` and ``--corpus``, each file may be:

  * ``.jsonl`` — one JSON object per line. The prompt is read from
    one of: ``prompt``, ``input``, ``question``, ``query``, ``text``,
    in that priority order.
  * ``.json``  — either a single list of strings, or a list of objects
    that follow the same key-priority rule above.
  * ``.txt``   — one prompt per line.

Pass ``--prompt-key foo`` to override the auto-detected key.

Usage
=====

::

    # Audit a RouterArena snapshot against your training corpus.
    python -m verifier.contamination_audit \\
        --benchmark routerarena_sub10.jsonl routerarena_full.jsonl \\
        --corpus    my_train.jsonl my_val.jsonl \\
        --report-out reports/routerarena_overlap.json

The script exits with status ``0`` on zero overlap and ``2`` on any
overlap so it can be wired straight into a CI gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple


_DEFAULT_PROMPT_KEYS: Tuple[str, ...] = (
    "prompt",
    "input",
    "question",
    "query",
    "text",
)


def normalize_and_hash(text: str) -> str:
    """Return the canonical NadirClaw contamination-audit hash.

    NFC-normalise, strip leading/trailing whitespace, casefold, encode
    as UTF-8, SHA-256 hex. Matches Nadir's published verifier audits.
    """
    if not isinstance(text, str):
        raise TypeError(f"normalize_and_hash expects str, got {type(text).__name__}")
    nfc = unicodedata.normalize("NFC", text)
    canonical = nfc.strip().casefold().encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------


def _extract_prompt(
    obj: object, prompt_key: Optional[str]
) -> Optional[str]:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        if prompt_key is not None:
            v = obj.get(prompt_key)
            return v if isinstance(v, str) else None
        for k in _DEFAULT_PROMPT_KEYS:
            v = obj.get(k)
            if isinstance(v, str):
                return v
    return None


def _iter_prompts_from_file(
    path: Path, prompt_key: Optional[str]
) -> Iterator[Tuple[str, int]]:
    """Yield ``(prompt, row_index)`` for each prompt in ``path``."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"WARN: {path}:{idx} could not parse JSON line: {e}",
                        file=sys.stderr,
                    )
                    continue
                p = _extract_prompt(obj, prompt_key)
                if p is not None:
                    yield p, idx
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for idx, obj in enumerate(data):
                p = _extract_prompt(obj, prompt_key)
                if p is not None:
                    yield p, idx
        else:
            print(
                f"WARN: {path} is JSON but not a list of objects/strings; "
                "skipping.",
                file=sys.stderr,
            )
    elif suffix == ".txt":
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.rstrip("\n")
                if line:
                    yield line, idx
    else:
        raise ValueError(
            f"Unsupported file type {suffix!r} for {path}; "
            "expected .jsonl, .json, or .txt"
        )


def _hash_files(
    paths: Iterable[Path], prompt_key: Optional[str]
) -> Tuple[dict, int]:
    """Hash every prompt in every file. Returns ``(hash_to_source, raw_count)``."""
    hashes: dict = {}
    raw = 0
    for path in paths:
        for prompt, idx in _iter_prompts_from_file(path, prompt_key):
            raw += 1
            h = normalize_and_hash(prompt)
            # Keep first occurrence's source for the report.
            if h not in hashes:
                hashes[h] = {
                    "file": str(path),
                    "row": idx,
                    "preview": prompt[:120],
                }
    return hashes, raw


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def run_audit(
    benchmark_paths: List[Path],
    corpus_paths: List[Path],
    report_out: Optional[Path],
    prompt_key: Optional[str] = None,
    sample_overlaps: int = 50,
) -> dict:
    """Run the audit. Returns the in-memory report dict.

    ``sample_overlaps`` caps the number of overlap examples written into
    the report so a pathological full-overlap case does not blow up
    the JSON file. The full overlap-hash set is still reported as a count.
    """
    bench_hashes, bench_raw = _hash_files(benchmark_paths, prompt_key)
    corpus_hashes, corpus_raw = _hash_files(corpus_paths, prompt_key)

    overlap = set(bench_hashes) & set(corpus_hashes)
    examples = []
    for h in list(overlap)[:sample_overlaps]:
        examples.append({
            "hash": h,
            "benchmark_file": bench_hashes[h]["file"],
            "benchmark_row": bench_hashes[h]["row"],
            "corpus_file": corpus_hashes[h]["file"],
            "corpus_row": corpus_hashes[h]["row"],
            "preview": bench_hashes[h]["preview"],
        })

    report = {
        "audit_date": datetime.now(timezone.utc).isoformat(),
        "benchmark_files": [str(p) for p in benchmark_paths],
        "corpus_files": [str(p) for p in corpus_paths],
        "benchmark_raw_count": bench_raw,
        "benchmark_unique_hashes": len(bench_hashes),
        "corpus_raw_count": corpus_raw,
        "corpus_unique_hashes": len(corpus_hashes),
        "overlap_count": len(overlap),
        "overlap_examples": examples,
        "hash_recipe": "sha256(NFC(prompt).strip().casefold().utf8)",
        "tool": "nadirclaw.verifier.contamination_audit",
    }

    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="contamination_audit",
        description=(
            "Audit benchmark-vs-training-corpus prompt overlap. "
            "Reproduces the zero-leakage checks behind Nadir's RouterBench "
            "and RouterArena results."
        ),
    )
    p.add_argument(
        "--benchmark",
        nargs="+",
        required=True,
        type=Path,
        help="One or more benchmark prompt files (.jsonl / .json / .txt).",
    )
    p.add_argument(
        "--corpus",
        nargs="+",
        required=True,
        type=Path,
        help="One or more training-corpus prompt files.",
    )
    p.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Where to write the JSON report (parent dir auto-created).",
    )
    p.add_argument(
        "--prompt-key",
        default=None,
        help=(
            "JSON key to read prompts from. Default: try prompt, input, "
            "question, query, text in that order."
        ),
    )
    p.add_argument(
        "--sample-overlaps",
        type=int,
        default=50,
        help="Max number of overlap examples to include in the report.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    # Validate file existence up front so the audit either runs or fails
    # loudly — never a silent "0 overlap because both files were empty".
    missing = [p for p in [*args.benchmark, *args.corpus] if not p.exists()]
    if missing:
        print(
            "ERROR: input files not found:\n  " + "\n  ".join(str(m) for m in missing),
            file=sys.stderr,
        )
        return 1

    report = run_audit(
        benchmark_paths=list(args.benchmark),
        corpus_paths=list(args.corpus),
        report_out=args.report_out,
        prompt_key=args.prompt_key,
        sample_overlaps=args.sample_overlaps,
    )

    print(f"benchmark unique hashes: {report['benchmark_unique_hashes']:>9d}", file=sys.stderr)
    print(f"corpus    unique hashes: {report['corpus_unique_hashes']:>9d}", file=sys.stderr)
    print(f"OVERLAP                : {report['overlap_count']:>9d}", file=sys.stderr)

    return 2 if report["overlap_count"] > 0 else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
